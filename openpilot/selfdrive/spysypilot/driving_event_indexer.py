#!/usr/bin/env python3
"""Bounded, reconstructable SSH index for rlog-authoritative driving events."""
import argparse
import json
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog


POLL_INTERVAL = 30.0
EVENT_ROOT = Path("/data/community/driving_events")
MANIFEST_NAME = "manifest.jsonl"
ROTATED_MANIFEST_NAME = "manifest.jsonl.1"
STATE_NAME = "processed_segments.json"
REVIEWS_NAME = "reviews.jsonl"
MANIFEST_MAX_BYTES = 2 * 1024 * 1024
PRESERVE_ATTR_NAME = "user.preserve"
PRESERVE_ATTR_VALUE = b"1"
CONTEXT_BEFORE = 2
CONTEXT_AFTER = 1

# Primary-event designation (A4). Only manifest records with an envelope
# version >= DESIGNATION_MIN_VERSION participate in group_role computation;
# older (v5-era and earlier) records always self-designate as primary.
DESIGNATION_MIN_VERSION = 3
GROUP_PRIMARY_PRIORITY_MANUAL = 110
GROUP_PRIMARY_PRIORITY_DRIVER_ASSISTED_UNWIND = 100
GROUP_PRIMARY_PRIORITY_AUTONOMOUS_UNWIND = 90
GROUP_PRIMARY_PRIORITY_HANDOFF = 80
GROUP_PRIMARY_PRIORITY_TURN_STOP_AND_LONG = 70
GROUP_PRIMARY_PRIORITY_STALL_AND_LATE_UNWIND = 60
GROUP_PRIMARY_PRIORITY_DEFAULT = 0
# lat.unwindProgressDeficit's priority depends on driver-assisted vs
# autonomous (see _group_priority); long.* is matched by prefix, not listed.
GROUP_PRIMARY_PRIORITY: dict[str, int] = {
  "manual.general": GROUP_PRIMARY_PRIORITY_MANUAL,
  "lat.unwindProgressDeficit": GROUP_PRIMARY_PRIORITY_AUTONOMOUS_UNWIND,
  "lat.handoffMismatch": GROUP_PRIMARY_PRIORITY_HANDOFF,
  "lat.centerOvershoot": GROUP_PRIMARY_PRIORITY_HANDOFF,
  "lat.committedHandoffHarshness": GROUP_PRIMARY_PRIORITY_HANDOFF,
  "lat.torqueAuthority": GROUP_PRIMARY_PRIORITY_HANDOFF,
  "lat.turnStopTurn": GROUP_PRIMARY_PRIORITY_TURN_STOP_AND_LONG,
  "lat.stallRelease": GROUP_PRIMARY_PRIORITY_STALL_AND_LATE_UNWIND,
  "lat.lateUnwind": GROUP_PRIMARY_PRIORITY_STALL_AND_LATE_UNWIND,
}


@dataclass
class ScannedSegment:
  route: str
  segment: int
  start_mono_time: int | None
  route_start_mono_time: int | None
  events: list[tuple[int, Any]]
  acknowledgments: dict[str, tuple[int, Any]]


def find_rlog(segment_path: Path) -> Path | None:
  for name in ("rlog.zst", "rlog.bz2", "rlog"):
    path = segment_path / name
    if path.is_file():
      return path
  return None


def is_preserved(segment_path: Path) -> bool:
  try:
    return os.getxattr(segment_path, PRESERVE_ATTR_NAME) == PRESERVE_ATTR_VALUE
  except OSError:
    return False


def is_complete(segment_path: Path) -> bool:
  try:
    return not any(name.endswith(".lock") for name in os.listdir(segment_path))
  except OSError:
    return False


def parse_segment_name(name: str) -> tuple[str, int] | None:
  route, separator, segment = name.rpartition("--")
  if not separator or not route:
    return None
  try:
    return route, int(segment)
  except ValueError:
    return None


def _enum_name(value: Any) -> str:
  return str(value)


def _driver_confound_reasons(mask: int) -> list[str]:
  reasons = []
  if mask & 1:
    reasons.append("driverTorque")
  if mask & 2:
    reasons.append("steeringPressed")
  return reasons


def _payload(event: Any) -> dict[str, Any]:
  which = event.payload.which()
  if which == "lateral":
    value = event.payload.lateral
    return {
      "controller_version": int(value.controllerVersion),
      "reference_version": int(value.referenceVersion),
      "v_ego": round(float(value.vEgo), 4),
      "steering_angle_deg": round(float(value.steeringAngleDeg), 4),
      "steering_rate_deg": round(float(value.steeringRateDeg), 4),
      "desired_lateral_accel": round(float(value.desiredLateralAccel), 4),
      "actual_lateral_accel": round(float(value.actualLateralAccel), 4),
      "request_torque": round(float(value.requestTorque), 4),
      "applied_torque": round(float(value.appliedTorque), 4),
      "reference_target_torque": round(float(value.referenceTargetTorque), 4),
      "reference_rate": round(float(value.referenceRate), 4),
      "reference_unwind_scale": round(float(value.referenceUnwindScale), 4),
      "reference_sustained_unwind_scale": round(float(value.referenceSustainedUnwindScale), 4),
      "unwind_effective_phase": round(float(value.unwindEffectivePhase), 4),
      "unwind_overspeed": round(float(value.unwindOverspeed), 4),
      "unwind_same_episode": bool(value.unwindSameEpisode),
      "applied_target_gap": round(float(value.appliedTargetGap), 4),
      "p_term": round(float(value.pTerm), 4),
      "driver_torque": round(float(value.driverTorque), 4),
      "steering_pressed": bool(value.steeringPressed),
      "steering_torque_eps": round(float(value.steeringTorqueEps), 4),
      "damping_applied": round(float(value.dampingApplied), 4),
      "damping_state": str(value.dampingState),
      "trigger_driver_confounded": bool(value.triggerDriverConfounded),
      "trigger_road_confounded": bool(value.triggerRoadConfounded),
      "driver_confounded_fraction": round(float(value.driverConfoundedFraction), 4),
      "max_abs_driver_torque": round(float(value.maxAbsDriverTorque), 4),
      "steering_pressed_any": bool(value.steeringPressedAny),
      "road_confounded_fraction": round(float(value.roadConfoundedFraction), 4),
      "driver_confound_reason": int(value.driverConfoundReason),
      "driver_confound_reasons": _driver_confound_reasons(int(value.driverConfoundReason)),
      "evidence_start_mono_time": int(value.evidenceStartMonoTime),
      "evidence_end_mono_time": int(value.evidenceEndMonoTime),
      "stall_release_count": int(value.stallReleaseCount),
      "release_offsets_s": [round(float(item), 4) for item in value.releaseOffsetsS],
      "stall_durations_s": [round(float(item), 4) for item in value.stallDurationsS],
      "release_peak_rates_deg": [round(float(item), 4) for item in value.releasePeakRatesDeg],
      "stall_episode_phase": str(value.stallEpisodePhase),
      "late_unwind_duration_s": round(float(value.lateUnwindDurationS), 4),
      "previous_unwind_effective_phase": round(float(value.previousUnwindEffectivePhase), 4),
      "previous_unwind_same_episode": bool(value.previousUnwindSameEpisode),
      "tracking_inactive_time_s": round(float(value.trackingInactiveTimeS), 4),
      "raw_torque_above_threshold_fraction": round(float(value.rawTorqueAboveThresholdFraction), 4),
      "raw_torque_above_threshold_longest_s": round(float(value.rawTorqueAboveThresholdLongestS), 4),
      "steering_pressed_fraction": round(float(value.steeringPressedFraction), 4),
      "driver_torque_present_at_trigger": bool(value.driverTorquePresentAtTrigger),
      "driver_interaction": str(value.driverInteraction),
      "road_confounded_at_event_time": bool(value.roadConfoundedAtEventTime),
      "max_vertical_accel_deviation": round(float(value.maxVerticalAccelDeviation), 4),
      "longest_road_bump_interval_s": round(float(value.longestRoadBumpIntervalS), 4),
      "road_confound_extent": str(value.roadConfoundExtent),
      "center_crossing_mono_time": int(value.centerCrossingMonoTime),
      "classification_confirmed_mono_time": int(value.classificationConfirmedMonoTime),
      "established_outside_center": bool(value.establishedOutsideCenter),
      "center_crossing_wheel_rate_deg": round(float(value.centerCrossingWheelRateDeg), 4),
      "desired_lateral_accel_before_crossing": round(float(value.desiredLateralAccelBeforeCrossing), 4),
      "desired_lateral_accel_after_crossing": round(float(value.desiredLateralAccelAfterCrossing), 4),
      "desired_reversal_committed": bool(value.desiredReversalCommitted),
      "desired_reversal_commit_mono_time": int(value.desiredReversalCommitMonoTime),
      "tracking_error_at_crossing": round(float(value.trackingErrorAtCrossing), 4),
      "applied_target_gap_at_crossing": round(float(value.appliedTargetGapAtCrossing), 4),
      "handoff_observed": bool(value.handoffObserved),
      "handoff_mono_time": int(value.handoffMonoTime),
      "handoff_center_delta_s": round(float(value.handoffCenterDeltaS), 4),
      "handoff_consolidated": bool(value.handoffConsolidated),
      "signed_center_crossing_wheel_rate_deg": round(float(value.signedCenterCrossingWheelRateDeg), 4),
      "peak_abs_center_wheel_rate_deg": round(float(value.peakAbsCenterWheelRateDeg), 4),
      "peak_tracking_error": round(float(value.peakTrackingError), 4),
      "peak_applied_target_gap": round(float(value.peakAppliedTargetGap), 4),
      "steering_pressed_at_trigger": bool(value.steeringPressedAtTrigger),
      "requested_torque_at_crossing": round(float(value.requestedTorqueAtCrossing), 4),
      "applied_torque_at_crossing": round(float(value.appliedTorqueAtCrossing), 4),
      "reference_target_torque_at_crossing": round(float(value.referenceTargetTorqueAtCrossing), 4),
      "unwind_episode_start_mono_time": int(value.unwindEpisodeStartMonoTime),
      "unwind_deficit_start_mono_time": int(value.unwindDeficitStartMonoTime),
      "unwind_deficit_duration_s": round(float(value.unwindDeficitDurationS), 4),
      "initial_steering_angle_deg": round(float(value.initialSteeringAngleDeg), 4),
      "peak_steering_angle_deg": round(float(value.peakSteeringAngleDeg), 4),
      "trigger_steering_angle_deg": round(float(value.triggerSteeringAngleDeg), 4),
      "expected_unwind_direction": str(value.expectedUnwindDirection),
      "expected_angle_progress_deg": round(float(value.expectedAngleProgressDeg), 4),
      "actual_angle_progress_deg": round(float(value.actualAngleProgressDeg), 4),
      "unwind_progress_ratio": round(float(value.unwindProgressRatio), 4),
      "reference_rate_at_trigger": round(float(value.referenceRateAtTrigger), 4),
      "measurement_rate_at_trigger": round(float(value.measurementRateAtTrigger), 4),
      "peak_reference_measurement_rate_gap": round(float(value.peakReferenceMeasurementRateGap), 4),
      "requested_torque_neutral_cross_mono_time": int(value.requestedTorqueNeutralCrossMonoTime),
      "requested_torque_neutral_cross_present": bool(value.requestedTorqueNeutralCrossPresent),
      "applied_torque_neutral_cross_mono_time": int(value.appliedTorqueNeutralCrossMonoTime),
      "applied_torque_neutral_cross_present": bool(value.appliedTorqueNeutralCrossPresent),
      "unwind_command_delay_s": round(float(value.unwindCommandDelayS), 4),
      "driver_assisted_unwind": bool(value.driverAssistedUnwind),
      "driver_assist_mono_time": int(value.driverAssistMonoTime),
      "driver_assist_angle_deg": round(float(value.driverAssistAngleDeg), 4),
      "driver_assist_wheel_rate_deg": round(float(value.driverAssistWheelRateDeg), 4),
      "driver_assist_torque": round(float(value.driverAssistTorque), 4),
      "wheel_rate_increase_after_assist_deg_s": round(float(value.wheelRateIncreaseAfterAssistDegS), 4),
      "movement_start_mono_time": int(value.movementStartMonoTime),
      "dwell_start_mono_time": int(value.dwellStartMonoTime),
      "release_mono_time": int(value.releaseMonoTime),
      "dwell_duration_s": round(float(value.dwellDurationS), 4),
      "dwell_steering_angle_deg": round(float(value.dwellSteeringAngleDeg), 4),
      "pre_dwell_peak_rate_deg_s": round(float(value.preDwellPeakRateDegS), 4),
      "minimum_dwell_rate_deg_s": round(float(value.minimumDwellRateDegS), 4),
      "release_peak_rate_deg_s": round(float(value.releasePeakRateDegS), 4),
      "rate_reduction_ratio": round(float(value.rateReductionRatio), 4),
      "restart_classification": str(value.restartClassification),
      "request_torque_during_dwell": round(float(value.requestTorqueDuringDwell), 4),
      "applied_torque_during_dwell": round(float(value.appliedTorqueDuringDwell), 4),
      "reference_target_during_dwell": round(float(value.referenceTargetDuringDwell), 4),
      "tracking_active_during_dwell": bool(value.trackingActiveDuringDwell),
      "measurement_rate": round(float(value.measurementRate), 4),
      "driver_involved_before_dwell": bool(value.driverInvolvedBeforeDwell),
      "driver_involved_during_dwell": bool(value.driverInvolvedDuringDwell),
      "driver_involved_after_dwell": bool(value.driverInvolvedAfterDwell),
      "turn_stop_actual_damping_amount": round(float(value.turnStopActualDampingAmount), 4),
      "turn_stop_actual_damping_state": str(value.turnStopActualDampingState),
      "turn_stop_turn_in_blocked": bool(value.turnStopTurnInBlocked),
      "turn_stop_breakaway_latch": round(float(value.turnStopBreakawayLatch), 4),
      "turn_stop_sustain_floor_contribution": round(float(value.turnStopSustainFloorContribution), 4),
      "turn_stop_damping_version": int(value.turnStopDampingVersion),
      "turn_stop_damping_valid": bool(value.turnStopDampingValid),
      "unwind_requested_torque_at_trigger": round(float(value.unwindRequestedTorqueAtTrigger), 4),
      "unwind_applied_torque_at_trigger": round(float(value.unwindAppliedTorqueAtTrigger), 4),
      "unwind_reference_target_torque_at_trigger": round(float(value.unwindReferenceTargetTorqueAtTrigger), 4),
      "unwind_applied_target_gap_at_trigger": round(float(value.unwindAppliedTargetGapAtTrigger), 4),
      "unwind_peak_applied_target_gap": round(float(value.unwindPeakAppliedTargetGap), 4),
      "unwind_neutral_torque": round(float(value.unwindNeutralTorque), 4),
      "unwind_phase_direction": round(float(value.unwindPhaseDirection), 4),
      "high_angle_evidence_valid": bool(value.highAngleEvidenceValid),
      "high_angle_unwind_scale": round(float(value.highAngleUnwindScale), 4),
      "torque_command_before_high_angle_exit": round(float(value.torqueCommandBeforeHighAngleExit), 4),
      "high_angle_unwind_old_torque_correction": round(float(value.highAngleUnwindOldTorqueCorrection), 4),
      "high_angle_unwind_old_direction_torque": round(float(value.highAngleUnwindOldDirectionTorque), 4),
      "old_turn_sign": round(float(value.oldTurnSign), 4),
      "future_unwind_commit_mono_time": int(value.futureUnwindCommitMonoTime),
      "future_unwind_commit_present": bool(value.futureUnwindCommitPresent),
      "high_angle_exit_first_nonzero_mono_time": int(value.highAngleExitFirstNonzeroMonoTime),
      "high_angle_exit_first_nonzero_present": bool(value.highAngleExitFirstNonzeroPresent),
      "requested_crown_neutral_mono_time": int(value.requestedCrownNeutralMonoTime),
      "requested_crown_neutral_present": bool(value.requestedCrownNeutralPresent),
      "applied_crown_neutral_mono_time": int(value.appliedCrownNeutralMonoTime),
      "applied_crown_neutral_present": bool(value.appliedCrownNeutralPresent),
      "unwind_crown_command_delay_s": round(float(value.unwindCrownCommandDelayS), 4),
      "wheel_progress_5_mono_time": int(value.wheelProgress5MonoTime),
      "wheel_progress_5_present": bool(value.wheelProgress5Present),
      "wheel_progress_20_mono_time": int(value.wheelProgress20MonoTime),
      "wheel_progress_20_present": bool(value.wheelProgress20Present),
      "wheel_progress_50_mono_time": int(value.wheelProgress50MonoTime),
      "wheel_progress_50_present": bool(value.wheelProgress50Present),
      "unwind_rebound_max_magnitude": round(float(value.unwindReboundMaxMagnitude), 4),
      "unwind_rebound_start_mono_time": int(value.unwindReboundStartMonoTime),
      "unwind_rebound_start_present": bool(value.unwindReboundStartPresent),
      "unwind_rebound_duration_s": round(float(value.unwindReboundDurationS), 4),
      "unwind_rebound_same_episode": bool(value.unwindReboundSameEpisode),
      "driver_active_before_deficit": bool(value.driverActiveBeforeDeficit),
      "driver_active_at_deficit_start": bool(value.driverActiveAtDeficitStart),
      "driver_active_during_evaluation": bool(value.driverActiveDuringEvaluation),
      "driver_intervened_after_deficit": bool(value.driverIntervenedAfterDeficit),
      "driver_intervention_accelerated_progress": bool(value.driverInterventionAcceleratedProgress),
      "driver_causation": str(value.driverCausation),
      "turn_stop_pre_dwell_progress_deg": round(float(value.turnStopPreDwellProgressDeg), 4),
      "turn_stop_post_dwell_progress_deg": round(float(value.turnStopPostDwellProgressDeg), 4),
      "road_evidence_window_start_mono_time": int(value.roadEvidenceWindowStartMonoTime),
      "road_evidence_window_start_present": bool(value.roadEvidenceWindowStartPresent),
      "driver_assist_raw_torque_only": bool(value.driverAssistRawTorqueOnly),
      "stall_releases": [{
        "release_mono_time": int(release.releaseMonoTime),
        "offset_from_trigger_s": round(float(release.offsetFromTriggerS), 4),
        "stall_duration_s": round(float(release.stallDurationS), 4),
        "peak_signed_steering_rate_deg": round(float(release.peakSignedSteeringRateDeg), 4),
        "peak_abs_steering_rate_deg": round(float(release.peakAbsSteeringRateDeg), 4),
        "steering_angle_deg": round(float(release.steeringAngleDeg), 4),
        "v_ego": round(float(release.vEgo), 4),
        "desired_lateral_accel": round(float(release.desiredLateralAccel), 4),
        "actual_lateral_accel": round(float(release.actualLateralAccel), 4),
        "requested_torque": round(float(release.requestedTorque), 4),
        "applied_torque": round(float(release.appliedTorque), 4),
        "reference_target_torque": round(float(release.referenceTargetTorque), 4),
        "damping_applied": round(float(release.dampingApplied), 4),
        "damping_state": str(release.dampingState),
        "turn_in_blocked": bool(release.turnInBlocked),
        "breakaway_latch": round(float(release.breakawayLatch), 4),
        "sustain_floor": round(float(release.sustainFloor), 4),
        "damping_version": int(release.dampingVersion),
        "damping_valid": bool(release.dampingValid),
        "driver_evidence_active": bool(release.driverEvidenceActive),
        "road_evidence_active": bool(release.roadEvidenceActive),
        "phase": str(release.phase),
      } for release in value.stallReleases],
    }
  if which == "leadLaunch":
    value = event.payload.leadLaunch
    return {
      "forecast_to_lead_s": round(float(value.forecastToLeadS), 4),
      "plan_to_lead_s": round(float(value.planToLeadS), 4),
      "command_to_lead_s": round(float(value.commandToLeadS), 4),
      "lead_to_ego_s": round(float(value.leadToEgoS), 4),
      "command_to_ego_s": round(float(value.commandToEgoS), 4),
      "radar_discontinuity": bool(value.radarDiscontinuity),
      "radar_confidence": round(float(value.radarConfidence), 4),
      "attribution_detail": str(value.attributionDetail),
      "onsets": [{
        "kind": str(onset.kind),
        "mono_time": int(onset.monoTime),
        "d_rel": round(float(onset.dRel), 4),
        "v_lead": round(float(onset.vLead), 4),
        "v_ego": round(float(onset.vEgo), 4),
        "a_ego": round(float(onset.aEgo), 4),
        "output_accel": round(float(onset.outputAccel), 4),
        "brake_pressed": bool(onset.brakePressed),
        "brake_hold_active": bool(onset.brakeHoldActive),
      } for onset in value.onsets],
    }
  if which == "stopJolt":
    value = event.payload.stopJolt
    return {
      "episode_start_mono_time": int(value.episodeStartMonoTime),
      "standstill_mono_time": int(value.standstillMonoTime),
      "peak_jolt_mono_time": int(value.peakJoltMonoTime),
      "detection_mono_time": int(value.detectionMonoTime),
      "imu_jerk": round(float(value.imuJerk), 4),
      "abs_imu_jerk": round(float(value.absImuJerk), 4),
      "a_ego_jerk": round(float(value.aEgoJerk), 4),
      "abs_a_ego_jerk": round(float(value.absAEgoJerk), 4),
      "imu_accel_before": round(float(value.imuAccelBefore), 4),
      "imu_accel_after": round(float(value.imuAccelAfter), 4),
      "a_ego_accel_before": round(float(value.aEgoAccelBefore), 4),
      "a_ego_accel_after": round(float(value.aEgoAccelAfter), 4),
      "accel_change": round(float(value.accelChange), 4),
      "accel_at_02_mps": round(float(value.accelAt02Mps), 4),
      "v_ego_at_peak": round(float(value.vEgoAtPeak), 4),
      "plan_a_target": round(float(value.planATarget), 4),
      "requested_accel": round(float(value.requestedAccel), 4),
      "applied_accel": round(float(value.appliedAccel), 4),
      "plan_accel_change": round(float(value.planAccelChange), 4),
      "requested_accel_change": round(float(value.requestedAccelChange), 4),
      "applied_accel_change": round(float(value.appliedAccelChange), 4),
      "should_stop_before": bool(value.shouldStopBefore),
      "should_stop_at_peak": bool(value.shouldStopAtPeak),
      "should_stop_after": bool(value.shouldStopAfter),
      "long_control_state_before": str(value.longControlStateBefore),
      "long_control_state_at_peak": str(value.longControlStateAtPeak),
      "long_control_state_after": str(value.longControlStateAfter),
      "lead_present": bool(value.leadPresent),
      "d_rel": round(float(value.dRel), 4),
      "v_lead_k": round(float(value.vLeadK), 4),
      "brake_pressed": bool(value.brakePressed),
      "gas_pressed": bool(value.gasPressed),
      "brake_hold_active": bool(value.brakeHoldActive),
      "radar_valid": bool(value.radarValid),
      "imu_valid": bool(value.imuValid),
      "road_confounded": bool(value.roadConfounded),
      "classification": str(value.classification),
    }
  if which == "rollingLeadResponse":
    value = event.payload.rollingLeadResponse
    return {
      "attribution_detail": str(value.attributionDetail),
      "lead_commit_mono_time": int(value.leadCommitMonoTime),
      "planner_response_mono_time": int(value.plannerResponseMonoTime),
      "controller_response_mono_time": int(value.controllerResponseMonoTime),
      "ego_response_mono_time": int(value.egoResponseMonoTime),
      "planner_response_present": bool(value.plannerResponsePresent),
      "controller_response_present": bool(value.controllerResponsePresent),
      "ego_response_present": bool(value.egoResponsePresent),
      "detected_mono_time": int(value.detectedMonoTime),
      "lead_to_plan_s": round(float(value.leadToPlanS), 4),
      "lead_to_command_s": round(float(value.leadToCommandS), 4),
      "lead_to_ego_s": round(float(value.leadToEgoS), 4),
      "baseline_lead_speed": round(float(value.baselineLeadSpeed), 4),
      "peak_lead_speed": round(float(value.peakLeadSpeed), 4),
      "baseline_ego_speed": round(float(value.baselineEgoSpeed), 4),
      "peak_ego_speed": round(float(value.peakEgoSpeed), 4),
      "peak_relative_speed": round(float(value.peakRelativeSpeed), 4),
      "baseline_gap": round(float(value.baselineGap), 4),
      "final_gap": round(float(value.finalGap), 4),
      "max_gap_growth": round(float(value.maxGapGrowth), 4),
      "baseline_lead_accel": round(float(value.baselineLeadAccel), 4),
      "peak_lead_accel": round(float(value.peakLeadAccel), 4),
      "baseline_planner_accel": round(float(value.baselinePlannerAccel), 4),
      "peak_planner_accel": round(float(value.peakPlannerAccel), 4),
      "baseline_output_accel": round(float(value.baselineOutputAccel), 4),
      "peak_output_accel": round(float(value.peakOutputAccel), 4),
      "baseline_ego_accel": round(float(value.baselineEgoAccel), 4),
      "peak_ego_accel": round(float(value.peakEgoAccel), 4),
      "radar_track_id": int(value.radarTrackId),
      "radar_discontinuity": bool(value.radarDiscontinuity),
      "driver_confounded": bool(value.driverConfounded),
      "onsets": [{
        "kind": str(onset.kind),
        "mono_time": int(onset.monoTime),
        "d_rel": round(float(onset.dRel), 4),
        "v_lead": round(float(onset.vLead), 4),
        "v_lead_k": round(float(onset.vLeadK), 4),
        "a_lead_k": round(float(onset.aLeadK), 4),
        "v_ego": round(float(onset.vEgo), 4),
        "a_ego": round(float(onset.aEgo), 4),
        "planner_accel": round(float(onset.plannerAccel), 4),
        "output_accel": round(float(onset.outputAccel), 4),
        "should_stop": bool(onset.shouldStop),
        "long_active": bool(onset.longActive),
        "radar_valid": bool(onset.radarValid),
        "gas_pressed": bool(onset.gasPressed),
        "brake_pressed": bool(onset.brakePressed),
      } for onset in value.onsets],
    }
  return {}


def _group_priority(record: dict[str, Any]) -> int:
  """Priority table for choosing a group's primary event (A4). Highest wins."""
  event_type = record.get("event_type", "")
  if event_type == "lat.unwindProgressDeficit":
    payload = record.get("payload") or {}
    if payload.get("driver_assisted_unwind") or payload.get("driver_causation") == "interventionBacked":
      return GROUP_PRIMARY_PRIORITY_DRIVER_ASSISTED_UNWIND
    return GROUP_PRIMARY_PRIORITY_AUTONOMOUS_UNWIND
  if isinstance(event_type, str) and event_type.startswith("long."):
    return GROUP_PRIMARY_PRIORITY_TURN_STOP_AND_LONG
  return GROUP_PRIMARY_PRIORITY.get(event_type, GROUP_PRIMARY_PRIORITY_DEFAULT)


def _assign_group_role(record: dict[str, Any], groups: dict[str, list[dict[str, Any]]],
                       known_group_primaries: dict[str, str]) -> None:
  """Set group_role/primary_event_id on `record` in place (A4).

  Records older than DESIGNATION_MIN_VERSION always self-designate as
  primary (no priority computation, no cross-group lookup) so rebuild can
  never re-designate a v5-era group. `groups` must include every event
  scanned for this route this round, including a candidate's lookahead
  following-segment members, so a group that straddles a segment boundary
  is designated identically whether seen in one round or two (A4).
  """
  if int(record["version"]) < DESIGNATION_MIN_VERSION:
    record["group_role"] = "primary"
    record["primary_event_id"] = record["event_id"]
    return
  group_id = record["group_id"]
  primary_event_id = known_group_primaries.get(group_id)
  if primary_event_id is None:
    members = groups.get(group_id) or [record]
    chosen = min(
      members,
      key=lambda member: (-_group_priority(member), member["occurred_mono_time"], member["event_id"]),
    )
    primary_event_id = chosen["event_id"]
    # First-manifested-primary-wins: cache immediately so any other
    # write-eligible member of this same group in this same round agrees.
    known_group_primaries[group_id] = primary_event_id
  record["group_role"] = "primary" if record["event_id"] == primary_event_id else "secondary"
  record["primary_event_id"] = primary_event_id


def event_to_record(event: Any, acknowledgment: Any | None, route: str, segment: int,
                    marker_mono_time: int, segment_start_mono_time: int,
                    route_start_mono_time: int, acknowledgment_mono_time: int | None = None) -> dict[str, Any]:
  occurred_mono_time = int(event.occurredMonoTime)
  detected_mono_time = int(event.detectedMonoTime) or occurred_mono_time
  if acknowledgment is not None and int(acknowledgment.segmentStartMonoTime):
    segment_start_mono_time = int(acknowledgment.segmentStartMonoTime)
  ack_mono_time = 0
  if acknowledgment is not None:
    ack_mono_time = int(acknowledgment.ackMonoTime) or int(acknowledgment_mono_time or 0)
  requested_before = max(0, int(event.requestedContextBefore))
  requested_after = max(0, int(event.requestedContextAfter))
  effective_before = min(requested_before, CONTEXT_BEFORE)
  effective_after = min(requested_after, CONTEXT_AFTER)
  ack = {
    "seen": acknowledgment is not None,
    "marker_written": bool(acknowledgment.markerWritten) if acknowledgment is not None else False,
    "marker_accepted": bool(acknowledgment.markerAccepted or acknowledgment.markerWritten) if acknowledgment is not None else False,
    "current_segment_preserved": bool(acknowledgment.currentSegmentPreserved) if acknowledgment is not None else False,
    "following_segment_scheduled": bool(acknowledgment.followingSegmentScheduled) if acknowledgment is not None else False,
    "error": str(acknowledgment.error) if acknowledgment is not None else "",
    "ack_mono_time": ack_mono_time,
  }
  return {
    "version": int(event.version),
    "event_id": str(event.eventId),
    "group_id": str(event.groupId),
    "route": route,
    "segment": segment,
    "occurred_mono_time": occurred_mono_time,
    "detected_mono_time": detected_mono_time,
    "episode_start_mono_time": int(event.episodeStartMonoTime) or occurred_mono_time,
    "episode_key": str(event.episodeKey),
    "marker_mono_time": marker_mono_time,
    "segment_offset_s": round((occurred_mono_time - segment_start_mono_time) / 1e9, 6),
    "route_offset_s": round((occurred_mono_time - route_start_mono_time) / 1e9, 6),
    "marker_offset_s": round((marker_mono_time - segment_start_mono_time) / 1e9, 6),
    "detector_to_marker_ms": round((marker_mono_time - detected_mono_time) / 1e6, 3),
    "marker_to_ack_ms": round((ack_mono_time - marker_mono_time) / 1e6, 3) if ack_mono_time else None,
    "verified_in_completed_rlog": True,
    "domain": _enum_name(event.domain),
    "source": _enum_name(event.source),
    "event_type": str(event.eventType),
    "detector": str(event.detector),
    "detector_version": int(event.detectorVersion),
    "severity": _enum_name(event.severity),
    "confidence": round(float(event.confidence), 4),
    "reason": str(event.reason),
    "attribution": _enum_name(event.attribution),
    "driver_confounded": bool(event.driverConfounded),
    "road_confounded": bool(event.roadConfounded),
    "git_commit": str(event.gitCommit),
    "git_branch": str(event.gitBranch),
    "requested_context": {"before": requested_before, "after": requested_after},
    "effective_context": {"before": effective_before, "after": effective_after},
    "analysis_window": {
      "before_s": round(float(event.analysisWindowBeforeS), 3),
      "after_s": round(float(event.analysisWindowAfterS), 3),
    },
    "payload_type": event.payload.which(),
    "payload": _payload(event),
    "logger_acknowledgment": ack,
  }


def scan_segment(segment_path: Path, reader_factory: Callable[[str], Iterable[Any]]) -> ScannedSegment | None:
  parsed = parse_segment_name(segment_path.name)
  rlog = find_rlog(segment_path)
  if parsed is None or rlog is None:
    return None

  route, segment = parsed
  raw_events: list[tuple[int, Any]] = []
  acknowledgments: dict[str, tuple[int, Any]] = {}
  segment_start_mono_time: int | None = None
  route_start_mono_time: int | None = None
  first_mono_time: int | None = None
  for msg in reader_factory(str(rlog)):
    try:
      log_mono_time = int(msg.logMonoTime)
      if first_mono_time is None:
        first_mono_time = log_mono_time
      which = msg.which()
      if which == "initData" and route_start_mono_time is None:
        route_start_mono_time = log_mono_time
      elif which == "sentinel" and str(msg.sentinel.type) in ("startOfRoute", "startOfSegment"):
        segment_start_mono_time = log_mono_time
      elif which == "drivingEvent":
        raw_events.append((log_mono_time, msg.drivingEvent))
      elif which == "drivingEventRecorded":
        event_id = str(msg.drivingEventRecorded.eventId)
        prior = acknowledgments.get(event_id)
        if prior is None or log_mono_time >= prior[0]:
          acknowledgments[event_id] = (log_mono_time, msg.drivingEventRecorded)
    except Exception:
      cloudlog.exception(f"driving_event_indexer: malformed record in {rlog}")
      continue

  segment_start_mono_time = segment_start_mono_time or first_mono_time
  route_start_mono_time = route_start_mono_time or first_mono_time
  return ScannedSegment(route, segment, segment_start_mono_time, route_start_mono_time, raw_events, acknowledgments)


def protect_context(log_root: Path, route: str, segment: int,
                    before: int = CONTEXT_BEFORE, after: int = CONTEXT_AFTER) -> list[str]:
  names: list[str] = []
  for index in range(max(0, segment - before), segment + after + 1):
    name = f"{route}--{index}"
    path = log_root / name
    if path.is_dir():
      names.append(name)

  # The deleter protects two preceding segments around each xattr-marked segment.
  # Mark current/following only; this retains the full window without spending
  # four entries from its bounded preserved-segment quota.
  for index in range(segment, segment + after + 1):
    path = log_root / f"{route}--{index}"
    if not path.is_dir():
      continue
    try:
      os.setxattr(path, PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)
    except OSError:
      cloudlog.exception(f"driving_event_indexer: failed to preserve {path}")
  return names


def load_json(path: Path, default):
  try:
    with path.open(encoding="utf-8") as file:
      return json.load(file)
  except (OSError, ValueError):
    return default


def atomic_write_json(path: Path, value) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  with temporary.open("w", encoding="utf-8") as file:
    json.dump(value, file, sort_keys=True)
    file.write("\n")
    file.flush()
    os.fsync(file.fileno())
  os.replace(temporary, path)


def ensure_reviews_file(event_root: Path) -> None:
  fd = os.open(event_root / REVIEWS_NAME, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
  try:
    os.fsync(fd)
  finally:
    os.close(fd)


def load_event_ids(*manifest_paths: Path) -> set[str]:
  event_ids: set[str] = set()
  for manifest_path in manifest_paths:
    try:
      with manifest_path.open(encoding="utf-8") as file:
        for line in file:
          try:
            event = json.loads(line)
          except ValueError:
            continue
          if isinstance(event, dict) and isinstance(event.get("event_id"), str):
            event_ids.add(event["event_id"])
    except OSError:
      pass
  return event_ids


def load_group_primaries(*manifest_paths: Path) -> dict[str, str]:
  """Harvest group_id -> primary event_id from already-manifested lines (A4).

  A `group_role == "primary"` line claims its group directly; a
  `group_role == "secondary"` line's `primary_event_id` also claims the
  group (covers a secondary manifesting before its primary does). First
  manifested wins per group_id, so callers should pass paths oldest-first
  (rotated manifest, then the current active manifest).
  """
  primaries: dict[str, str] = {}
  for manifest_path in manifest_paths:
    try:
      with manifest_path.open(encoding="utf-8") as file:
        for line in file:
          try:
            event = json.loads(line)
          except ValueError:
            continue
          if not isinstance(event, dict):
            continue
          group_id = event.get("group_id")
          if not isinstance(group_id, str) or not group_id or group_id in primaries:
            continue
          role = event.get("group_role")
          if role == "primary":
            event_id = event.get("event_id")
            if isinstance(event_id, str) and event_id:
              primaries[group_id] = event_id
          elif role == "secondary":
            primary_event_id = event.get("primary_event_id")
            if isinstance(primary_event_id, str) and primary_event_id:
              primaries[group_id] = primary_event_id
    except OSError:
      pass
  return primaries


def rotate_manifest(manifest_path: Path, rotated_path: Path, incoming_bytes: int,
                    max_bytes: int = MANIFEST_MAX_BYTES) -> None:
  try:
    current_bytes = manifest_path.stat().st_size
  except OSError:
    current_bytes = 0
  if current_bytes and current_bytes + incoming_bytes > max_bytes:
    os.replace(manifest_path, rotated_path)


def append_events(manifest_path: Path, rotated_path: Path, events: list[dict[str, Any]],
                  known_ids: set[str], max_bytes: int = MANIFEST_MAX_BYTES) -> int:
  new_events: list[dict[str, Any]] = []
  batch_ids: set[str] = set()
  for event in events:
    event_id = event["event_id"]
    if event_id not in known_ids and event_id not in batch_ids:
      new_events.append(event)
      batch_ids.add(event_id)
  if not new_events:
    return 0
  payloads = [(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n").encode() for event in new_events]
  for event, payload in zip(new_events, payloads, strict=True):
    rotate_manifest(manifest_path, rotated_path, len(payload), max_bytes)
    fd = os.open(manifest_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
      view = memoryview(payload)
      while view:
        view = view[os.write(fd, view):]
      os.fsync(fd)
      known_ids.add(event["event_id"])
    finally:
      os.close(fd)
  return len(new_events)


def _segment_paths(log_root: Path) -> list[Path]:
  try:
    paths = [path for path in log_root.iterdir() if path.is_dir() and parse_segment_name(path.name) is not None]
    return sorted(paths, key=lambda path: parse_segment_name(path.name))
  except OSError:
    return []


def collect_records(log_root: Path, candidate_paths: list[Path],
                    all_segment_paths: list[Path],
                    reader_factory: Callable[[str], Iterable[Any]],
                    known_group_primaries: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], set[str]]:
  """Scan each candidate route in two passes, then join events to the newest acknowledgment by ID."""
  if known_group_primaries is None:
    known_group_primaries = {}
  path_by_key = {
    parsed: path
    for path in all_segment_paths
    if (parsed := parse_segment_name(path.name)) is not None
  }
  candidates_by_route: dict[str, list[Path]] = {}
  for path in candidate_paths:
    parsed = parse_segment_name(path.name)
    if parsed is not None:
      candidates_by_route.setdefault(parsed[0], []).append(path)

  records: list[dict[str, Any]] = []
  successfully_scanned: set[str] = set()
  for route, route_candidates in candidates_by_route.items():
    scan_paths = set(route_candidates)
    deferred: set[Path] = set()
    for candidate in route_candidates:
      parsed = parse_segment_name(candidate.name)
      assert parsed is not None
      following = path_by_key.get((route, parsed[1] + 1))
      if following is not None:
        if not is_complete(following) or find_rlog(following) is None:
          deferred.add(candidate)
        else:
          scan_paths.add(following)

    scanned: dict[Path, ScannedSegment] = {}
    failed: set[Path] = set()
    for path in sorted(scan_paths):
      try:
        result = scan_segment(path, reader_factory)
        if result is None:
          failed.add(path)
        else:
          scanned[path] = result
      except Exception:
        failed.add(path)
        cloudlog.exception(f"driving_event_indexer: failed to parse {path}")

    acknowledgments: dict[str, tuple[int, Any]] = {}
    for result in scanned.values():
      for event_id, acknowledgment in result.acknowledgments.items():
        prior = acknowledgments.get(event_id)
        if prior is None or acknowledgment[0] >= prior[0]:
          acknowledgments[event_id] = acknowledgment
    route_start_times = [
      result.route_start_mono_time for result in scanned.values()
      if result.route_start_mono_time is not None
    ]
    route_start_mono_time = min(route_start_times) if route_start_times else 0

    # Build every event's record across ALL scanned segments of this route —
    # write-eligible candidates AND their lookahead following segment — so
    # primary designation below sees group members that straddle a segment
    # boundary (A4), even though only write-eligible records are appended to
    # the output. `groups` feeds priority computation; only write-eligible
    # records are ever mutated with a group_role/primary_event_id or written.
    route_records_by_path: dict[Path, list[dict[str, Any]]] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path, result in scanned.items():
      if result.start_mono_time is None:
        continue
      path_records: list[dict[str, Any]] = []
      for marker_time, event in result.events:
        event_id = str(event.eventId)
        if not event_id:
          continue
        acknowledgment = acknowledgments.get(event_id)
        record = event_to_record(
          event,
          acknowledgment[1] if acknowledgment is not None else None,
          result.route,
          result.segment,
          marker_time,
          result.start_mono_time,
          route_start_mono_time,
          acknowledgment[0] if acknowledgment is not None else None,
        )
        path_records.append(record)
        groups[record["group_id"]].append(record)
      route_records_by_path[path] = path_records

    for candidate in route_candidates:
      if candidate in deferred or candidate in failed or candidate not in scanned:
        continue
      parsed = parse_segment_name(candidate.name)
      assert parsed is not None
      following = path_by_key.get((route, parsed[1] + 1))
      if following is not None and following in failed:
        continue

      result = scanned[candidate]
      if result.start_mono_time is not None:
        for record in route_records_by_path.get(candidate, []):
          _assign_group_role(record, groups, known_group_primaries)
          context = record["effective_context"]
          context_segments = protect_context(
            log_root, result.route, result.segment, context["before"], context["after"],
          )
          record["context_segments"] = context_segments
          context_indices = {
            parse_segment_name(name)[1]
            for name in context_segments
            if parse_segment_name(name) is not None
          }
          record["effective_context"] = {
            "before": sum(index < result.segment for index in context_indices),
            "after": sum(index > result.segment for index in context_indices),
          }
          records.append(record)
      successfully_scanned.add(candidate.name)

  return records, successfully_scanned


def _default_reader_factory() -> Callable[[str], Iterable[Any]]:
  from openpilot.tools.lib.logreader import LogReader

  def default_reader(path: str):
    return LogReader(path, sort_by_time=True)

  return default_reader


def scan_once(log_root: Path, event_root: Path = EVENT_ROOT,
              reader_factory: Callable[[str], Iterable[Any]] | None = None,
              scan_all: bool = False) -> int:
  if reader_factory is None:
    reader_factory = _default_reader_factory()

  event_root.mkdir(parents=True, exist_ok=True)
  ensure_reviews_file(event_root)
  manifest_path = event_root / MANIFEST_NAME
  rotated_path = event_root / ROTATED_MANIFEST_NAME
  state_path = event_root / STATE_NAME
  processed = set() if scan_all else set(load_json(state_path, []))
  known_ids = load_event_ids(manifest_path, rotated_path)
  # Oldest-first (rotated, then active) so first-manifested-primary-wins
  # reads chronologically (A4).
  known_group_primaries = load_group_primaries(rotated_path, manifest_path)
  all_segments = _segment_paths(log_root)
  present_segments = {path.name for path in all_segments}
  candidates = [
    path for path in all_segments
    if path.name not in processed
    and is_complete(path)
    and find_rlog(path) is not None
    and (scan_all or is_preserved(path))
  ]
  events, scanned = collect_records(log_root, candidates, all_segments, reader_factory, known_group_primaries)
  ids_before_append = set(known_ids)
  added = append_events(manifest_path, rotated_path, events, known_ids)
  if added:
    added_ids = known_ids - ids_before_append
    primary_count = sum(
      1 for event in events
      if event["event_id"] in added_ids and event.get("group_role") != "secondary"
    )
    cloudlog.info(f"driving_event_indexer: indexed {added} event(s) ({primary_count} primary)")
  processed.update(scanned)
  processed &= present_segments
  atomic_write_json(state_path, sorted(processed))
  return added


def rebuild(log_root: Path, event_root: Path = EVENT_ROOT,
            reader_factory: Callable[[str], Iterable[Any]] | None = None) -> int:
  if reader_factory is None:
    reader_factory = _default_reader_factory()

  event_root.mkdir(parents=True, exist_ok=True)
  ensure_reviews_file(event_root)
  manifest_path = event_root / MANIFEST_NAME
  rotated_path = event_root / ROTATED_MANIFEST_NAME
  state_path = event_root / STATE_NAME
  temporary_manifest = event_root / f"{MANIFEST_NAME}.rebuild"
  temporary_rotated = event_root / f"{ROTATED_MANIFEST_NAME}.rebuild"
  for path in (temporary_manifest, temporary_rotated):
    try:
      path.unlink()
    except FileNotFoundError:
      pass

  all_segments = _segment_paths(log_root)
  candidates = [
    path for path in all_segments
    if is_complete(path) and find_rlog(path) is not None
  ]
  # Rebuild recomputes designation from scratch: empty known-ids AND an
  # empty known-primary map, so it never inherits a stale primary choice.
  events, scanned = collect_records(log_root, candidates, all_segments, reader_factory, {})
  append_events(temporary_manifest, temporary_rotated, events, set())
  if not temporary_manifest.exists():
    temporary_manifest.touch(mode=0o600)

  if temporary_rotated.exists():
    os.replace(temporary_rotated, rotated_path)
  else:
    try:
      rotated_path.unlink()
    except FileNotFoundError:
      pass
  os.replace(temporary_manifest, manifest_path)
  directory_fd = os.open(event_root, os.O_RDONLY | os.O_DIRECTORY)
  try:
    os.fsync(directory_fd)
  finally:
    os.close(directory_fd)
  atomic_write_json(state_path, sorted(scanned))
  primary_count = sum(1 for event in events if event.get("group_role") != "secondary")
  cloudlog.info(f"driving_event_indexer: rebuilt {len(events)} event(s) ({primary_count} primary)")
  return len(events)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--rebuild", action="store_true", help="ignore processed state and inspect every supplied full rlog")
  parser.add_argument("--once", action="store_true")
  parser.add_argument("--log-root", type=Path, default=Path(Paths.log_root()))
  parser.add_argument("--event-root", type=Path, default=EVENT_ROOT)
  args = parser.parse_args()

  if args.rebuild:
    rebuild(args.log_root, args.event_root)
    return

  while True:
    try:
      scan_once(args.log_root, args.event_root)
    except Exception:
      cloudlog.exception("driving_event_indexer: scan failed")
    if args.once:
      return
    time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
  main()
