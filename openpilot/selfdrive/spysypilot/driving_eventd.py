#!/usr/bin/env python3
"""Observer-only runtime for manual, lateral, and longitudinal driving events."""
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.spysypilot.lat_event_detector import (
  DETECTOR_VERSION as LAT_DETECTOR_VERSION,
  LateralDetection,
  LateralEventDetector,
  LateralSample,
  RoadBumpClassifier,
  SteeringRateFilter,
)
from openpilot.selfdrive.spysypilot.long_event_detector import (
  DETECTOR_VERSION as LONG_DETECTOR_VERSION,
  LaunchSample,
  LeadLaunchDetector,
  LongEvent,
)
from openpilot.selfdrive.spysypilot.rolling_lead_response_detector import (
  DETECTOR_NAME as ROLLING_LEAD_DETECTOR_NAME,
  DETECTOR_VERSION as ROLLING_LEAD_DETECTOR_VERSION,
  RollingLeadResponseDetector,
  RollingLeadResponseEvent,
  RollingLeadSample,
)
from openpilot.selfdrive.spysypilot.stop_jolt_detector import (
  DETECTOR_VERSION as STOP_JOLT_DETECTOR_VERSION,
  StopJoltCarSample,
  StopJoltDetector,
  StopJoltEvent,
  StopJoltImuSample,
)


EVENT_VERSION = 4
GROUP_WINDOW_NS = 2_500_000_000
# Strictly greater than lat_event_detector.EPISODE_MAX_LIFETIME_S (20.0 s) plus
# occurred-time slack: a keyed group's physical episode can never legitimately
# span longer than the detector's own episode-lifetime cap (A3).
GROUP_MAX_LIFETIME_NS = 25_000_000_000
# Memory-only bookkeeping for _episode_groups entries; NOT a relatedness test
# (exact-key match within GROUP_MAX_LIFETIME_NS is the relatedness proof, A3).
GROUP_KEY_PRUNE_NS = 30_000_000_000
CONTEXT_BEFORE = 2
CONTEXT_AFTER = 1
ACK_RETRY_INTERVAL_NS = 1_000_000_000
# A7: confidence penalty for substantial road confounding, never applied to
# intervention-backed / driver-assisted unwind rescues (the 0.95 literal wins).
ROAD_SUBSTANTIAL_CONFIDENCE_PENALTY = 0.05
ROAD_CONFIDENCE_FLOOR = 0.50


@dataclass(frozen=True)
class EventCandidate:
  occurred_mono_time: int
  domain: str
  source: str
  event_type: str
  detector: str
  detector_version: int
  severity: str
  confidence: float
  reason: str
  attribution: str = "unknown"
  driver_confounded: bool = False
  road_confounded: bool = False
  payload: Any | None = None
  detector_evidence: Any | None = None
  detected_mono_time: int = 0
  episode_start_mono_time: int = 0
  analysis_window_before_s: float = 0.0
  analysis_window_after_s: float = 0.0
  episode_key: str = ""


@dataclass(frozen=True)
class AcceptedEvent:
  event_id: str
  group_id: str
  candidate: EventCandidate


@dataclass
class PendingEvent:
  event: AcceptedEvent
  last_sent_ns: int
  attempts: int = 1


class EventRecorder:
  """Assign stable IDs and short-lived physical-episode group IDs."""

  def __init__(self, id_factory: Callable[[], str] | None = None, group_window_ns: int = GROUP_WINDOW_NS):
    self.id_factory = id_factory or (lambda: uuid.uuid4().hex)
    self.group_window_ns = group_window_ns
    self._group_id = ""
    self._group_last_time = -group_window_ns
    # episode_key -> (group_id, last_seen_mono_time); last_seen refresh is
    # monotone (candidates can arrive with out-of-order occurred times within
    # one update() call since detection lag differs per detector, A3).
    self._episode_groups: dict[str, tuple[str, int]] = {}
    # group_id -> first-seen mono time, for the hard GROUP_MAX_LIFETIME_NS cap.
    self._group_first_time: dict[str, int] = {}

  def _mint_group(self, occurred_mono_time: int) -> str:
    group_id = self.id_factory()
    self._group_first_time[group_id] = occurred_mono_time
    return group_id

  def _group_expired(self, group_id: str, occurred_mono_time: int) -> bool:
    # Intentionally one-sided (forward-only): this expresses "has the group's
    # OWN lifetime, measured forward from its first-seen time, overflowed" --
    # the pathological-long-chain case (test_group_hard_lifetime_splits_a_
    # pathological_same_key_chain). It is only ever consulted (a) after the
    # window check below, which already uses abs() and rejects anything far
    # in the past, or (b) from the existing-key rebind path, where exact-key
    # match is itself the relatedness proof (A3) and an out-of-order-but-
    # earlier occurred time for the SAME key is expected (detection lag), not
    # a sign the group is stale. Making this symmetric would wrongly split a
    # still-live same-key group whenever a late-arriving-but-earlier-occurred
    # member showed up (see test_group_key_last_seen_refresh_is_monotone_
    # under_out_of_order_occurrence).
    first_time = self._group_first_time.get(group_id)
    return first_time is None or occurred_mono_time - first_time > GROUP_MAX_LIFETIME_NS

  def accept(self, candidate: EventCandidate) -> AcceptedEvent:
    if candidate.episode_key:
      episode_group = self._episode_groups.get(candidate.episode_key)
      if episode_group is None:
        # A brand-new key: chain onto the current running group only if it is
        # still fresh (keyless-style window) AND has not hit its hard
        # lifetime; otherwise mint a new group. No idle-staleness test is
        # applied to an EXISTING key's binding below — exact-key match is
        # itself the relatedness proof (A3).
        # abs() here: candidates from different detectors reach accept() in
        # different update() calls with different detection lag, so a later
        # accept() can carry an EARLIER occurred_mono_time (A3) -- a one-sided
        # check would let anything arbitrarily far in the PAST always satisfy
        # "<= window" (a negative delta is trivially <= a positive window),
        # incorrectly chaining unrelated maneuvers minutes apart.
        if (
          self._group_id
          and abs(candidate.occurred_mono_time - self._group_last_time) <= self.group_window_ns
          and not self._group_expired(self._group_id, candidate.occurred_mono_time)
        ):
          group_id = self._group_id
        else:
          group_id = self._mint_group(candidate.occurred_mono_time)
        self._group_id = group_id
        self._episode_groups[candidate.episode_key] = (group_id, candidate.occurred_mono_time)
      else:
        group_id, last_seen = episode_group
        if self._group_expired(group_id, candidate.occurred_mono_time):
          # Hard lifetime exceeded: mint a new group and re-bind the key to it.
          group_id = self._mint_group(candidate.occurred_mono_time)
        self._group_id = group_id
        self._episode_groups[candidate.episode_key] = (group_id, max(last_seen, candidate.occurred_mono_time))
    elif (
      not self._group_id
      or abs(candidate.occurred_mono_time - self._group_last_time) > self.group_window_ns
      or self._group_expired(self._group_id, candidate.occurred_mono_time)
    ):
      self._group_id = self._mint_group(candidate.occurred_mono_time)
    self._group_last_time = max(self._group_last_time, candidate.occurred_mono_time)
    # Bookkeeping-only pruning (not a relatedness/closure signal, A3).
    self._episode_groups = {
      key: value for key, value in self._episode_groups.items()
      if candidate.occurred_mono_time - value[1] <= GROUP_KEY_PRUNE_NS
    }
    active_group_ids = {value[0] for value in self._episode_groups.values()} | {self._group_id}
    self._group_first_time = {
      group_id: first_time for group_id, first_time in self._group_first_time.items()
      if group_id in active_group_ids
    }
    return AcceptedEvent(self.id_factory(), self._group_id, candidate)


def manual_candidate(occurred_mono_time: int) -> EventCandidate:
  return EventCandidate(
    occurred_mono_time=occurred_mono_time,
    domain="manual",
    source="user",
    event_type="manual.general",
    detector="bookmarkButton",
    detector_version=1,
    severity="info",
    confidence=1.0,
    reason="user marked a driving event",
    detected_mono_time=occurred_mono_time,
    episode_start_mono_time=occurred_mono_time,
    analysis_window_before_s=15.0,
    analysis_window_after_s=5.0,
  )


def lateral_candidate(sample: LateralSample, detection: LateralDetection) -> EventCandidate:
  is_takeover = detection.event_type == "driverTakeover"
  attribution = (
    "driver" if is_takeover
    else "actuator" if detection.event_type == "torqueAuthority"
    else "controller"
  )
  normalized_event_types = {
    "unwind_progress_deficit": "unwindProgressDeficit",
    "turn_stop_turn": "turnStopTurn",
  }
  event_type = normalized_event_types.get(detection.event_type, detection.event_type)
  evidence = detection.evidence
  # A takeover is the driver-interaction ground truth itself, not an
  # otherwise-valid controller event contaminated by driver input.
  driver_confounded = (
    False if is_takeover
    else evidence.driver_confounded_any if evidence is not None
    else sample.driver_confounded
  )
  road_confounded = evidence.road_confounded_any if evidence is not None else sample.road_confounded
  substantial_driver = evidence is not None and evidence.driver_interaction == "confirmedSteeringPressed"
  substantial_road = evidence is not None and evidence.road_interaction == "substantial"
  if not is_takeover and (substantial_driver or substantial_road):
    attribution = "mixed"
  # A7: bounded confidence penalty for substantial road confounding, applied
  # BEFORE constructing the frozen EventCandidate; never applied to an
  # intervention-backed / driver-assisted unwind rescue (the 0.95-confidence
  # literal wins there). The floor is an invariant, not a claimed engagement.
  confidence = detection.confidence
  if substantial_road and not is_takeover:
    rescued = evidence.driver_causation == "interventionBacked" or evidence.driver_assisted_unwind
    if not rescued:
      confidence = max(confidence - ROAD_SUBSTANTIAL_CONFIDENCE_PENALTY, ROAD_CONFIDENCE_FLOOR)
  occurred_time = detection.occurred_mono_time if detection.occurred_mono_time is not None else sample.mono_time
  detected_time = detection.detected_mono_time if detection.detected_mono_time is not None else sample.mono_time
  occurred_mono_time = round(occurred_time * 1e9)
  return EventCandidate(
    occurred_mono_time=occurred_mono_time,
    domain="lateral",
    source="automatic",
    event_type=f"lat.{event_type}",
    detector="blatLateralEventDetector",
    detector_version=LAT_DETECTOR_VERSION,
    severity=detection.severity,
    confidence=confidence,
    reason=detection.reason,
    attribution=attribution,
    driver_confounded=driver_confounded,
    road_confounded=road_confounded,
    payload=sample,
    detector_evidence=evidence,
    detected_mono_time=round(detected_time * 1e9),
    episode_start_mono_time=round(evidence.episode_start_mono_time * 1e9) if evidence is not None else occurred_mono_time,
    analysis_window_before_s=evidence.analysis_window_before_s if evidence is not None else 2.0,
    analysis_window_after_s=evidence.analysis_window_after_s if evidence is not None else 2.0,
    episode_key=evidence.episode_key if evidence is not None else "",
  )


def longitudinal_candidate(sample: LaunchSample, event: LongEvent) -> EventCandidate:
  names = {
    "late_lead_launch_planner": ("long.lateLeadLaunchPlanner", "planner"),
    "late_lead_launch_controller": ("long.lateLeadLaunchController", "controller"),
    "late_lead_launch_vehicle": ("long.lateLeadLaunchVehicle", "mixed"),
    "lead_launch_stall": ("long.leadLaunchStall", "unknown"),
  }
  event_type, attribution = names[event.event_type]
  severity = "critical" if event.severity >= 3 else ("warning" if event.severity else "info")
  occurred_mono_time = round(sample.t * 1e9)
  return EventCandidate(
    occurred_mono_time=occurred_mono_time,
    domain="longitudinal",
    source="automatic",
    event_type=event_type,
    detector="leadLaunchDetector",
    detector_version=LONG_DETECTOR_VERSION,
    severity=severity,
    confidence=event.confidence,
    reason=event.detail,
    attribution=attribution,
    payload=event,
    detected_mono_time=occurred_mono_time,
    episode_start_mono_time=round(event.episode_start_mono_time * 1e9) if event.episode_start_mono_time else occurred_mono_time,
    analysis_window_before_s=event.analysis_window_before_s,
    analysis_window_after_s=event.analysis_window_after_s,
    episode_key=f"long:{round(event.episode_start_mono_time * 1e9)}" if event.episode_start_mono_time else "",
  )


def stop_jolt_candidate(event: StopJoltEvent) -> EventCandidate:
  occurred_mono_time = round(event.peak_jolt_mono_time * 1e9)
  episode_start_mono_time = round(event.episode_start_mono_time * 1e9)
  return EventCandidate(
    occurred_mono_time=occurred_mono_time,
    domain="longitudinal",
    source="automatic",
    event_type="long.stopJolt",
    detector="smoothStopJoltDetector",
    detector_version=STOP_JOLT_DETECTOR_VERSION,
    severity=event.severity,
    confidence=event.confidence,
    reason=event.reason,
    attribution=event.attribution,
    road_confounded=event.road_confounded,
    payload=event,
    detected_mono_time=round(event.detection_mono_time * 1e9),
    episode_start_mono_time=episode_start_mono_time,
    analysis_window_before_s=5.0,
    analysis_window_after_s=2.0,
    episode_key=f"stop:{episode_start_mono_time}",
  )


def rolling_lead_candidate(event: RollingLeadResponseEvent) -> EventCandidate:
  names = {
    "late_rolling_lead_response_planner": ("long.lateRollingLeadResponsePlanner", "planner"),
    "late_rolling_lead_response_controller": ("long.lateRollingLeadResponseController", "controller"),
    "late_rolling_lead_response_vehicle": ("long.lateRollingLeadResponseVehicle", "vehicle"),
  }
  event_type, attribution = names[event.event_type]
  occurred_mono_time = round(event.occurred_mono_time * 1e9)
  episode_start_mono_time = round(event.episode_start_mono_time * 1e9)
  return EventCandidate(
    occurred_mono_time=occurred_mono_time,
    domain="longitudinal",
    source="automatic",
    event_type=event_type,
    detector=ROLLING_LEAD_DETECTOR_NAME,
    detector_version=ROLLING_LEAD_DETECTOR_VERSION,
    severity="critical" if event.severity >= 3 else "warning",
    confidence=event.confidence,
    reason=event.detail,
    attribution=attribution,
    driver_confounded=event.driver_confounded,
    payload=event,
    detected_mono_time=round(event.detected_mono_time * 1e9),
    episode_start_mono_time=episode_start_mono_time,
    analysis_window_before_s=event.analysis_window_before_s,
    analysis_window_after_s=event.analysis_window_after_s,
    episode_key=f"rolling-lead:{episode_start_mono_time}",
  )


class DrivingEventPlatform:
  """Runs detectors independently so one domain cannot disable another."""

  def __init__(self, recorder: EventRecorder | None = None,
               lateral_detector: LateralEventDetector | None = None,
               longitudinal_detector: LeadLaunchDetector | None = None,
               stop_jolt_detector: StopJoltDetector | None = None,
               rolling_lead_detector: RollingLeadResponseDetector | None = None,
               on_error: Callable[[str], None] | None = None):
    self.recorder = recorder or EventRecorder()
    self.lateral_detector = lateral_detector or LateralEventDetector()
    self.longitudinal_detector = longitudinal_detector or LeadLaunchDetector()
    self.stop_jolt_detector = stop_jolt_detector or StopJoltDetector()
    self.rolling_lead_detector = rolling_lead_detector or RollingLeadResponseDetector()
    self.on_error = on_error or (lambda domain: cloudlog.exception(f"driving_eventd: {domain} detector failed"))

  def update(self, lateral_sample: LateralSample | None = None, longitudinal_sample: LaunchSample | None = None,
             manual_pressed: bool = False, manual_time_ns: int | None = None,
             stop_car_sample: StopJoltCarSample | None = None,
             stop_imu_sample: StopJoltImuSample | None = None,
             rolling_lead_sample: RollingLeadSample | None = None) -> list[AcceptedEvent]:
    candidates: list[EventCandidate] = []
    if lateral_sample is not None:
      try:
        detection = self.lateral_detector.update(lateral_sample)
        if detection is not None:
          detections = detection if isinstance(detection, (list, tuple)) else (detection,)
          candidates.extend(
            lateral_candidate(lateral_sample, item) for item in detections
          )
      except Exception:
        self.on_error("lateral")

    if longitudinal_sample is not None:
      try:
        event = self.longitudinal_detector.update(longitudinal_sample)
        if event is not None:
          candidates.append(longitudinal_candidate(longitudinal_sample, event))
      except Exception:
        self.on_error("longitudinal")

    if rolling_lead_sample is not None:
      try:
        event = self.rolling_lead_detector.update(rolling_lead_sample)
        if event is not None:
          candidates.append(rolling_lead_candidate(event))
      except Exception:
        self.on_error("rolling_lead_response")

    if stop_imu_sample is not None:
      try:
        self.stop_jolt_detector.update_imu(stop_imu_sample)
      except Exception:
        self.on_error("stop_jolt_imu")

    if stop_car_sample is not None:
      try:
        event = self.stop_jolt_detector.update_car(stop_car_sample)
        if event is not None:
          candidates.append(stop_jolt_candidate(event))
      except Exception:
        self.on_error("stop_jolt")

    if manual_pressed:
      candidates.append(manual_candidate(manual_time_ns if manual_time_ns is not None else time.monotonic_ns()))

    candidates.sort(key=lambda candidate: candidate.occurred_mono_time)
    return [self.recorder.accept(candidate) for candidate in candidates]


def _param_text(params: Params, key: str) -> str:
  value = params.get(key)
  if value is None:
    return ""
  return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def build_message(event: AcceptedEvent, git_commit: str = "", git_branch: str = ""):
  msg = messaging.new_message("drivingEvent", valid=True)
  out = msg.drivingEvent
  candidate = event.candidate
  out.version = EVENT_VERSION
  out.eventId = event.event_id
  out.groupId = event.group_id
  out.occurredMonoTime = candidate.occurred_mono_time
  out.domain = candidate.domain
  out.source = candidate.source
  out.eventType = candidate.event_type
  out.detector = candidate.detector
  out.detectorVersion = candidate.detector_version
  out.severity = candidate.severity
  out.confidence = candidate.confidence
  out.reason = candidate.reason
  out.attribution = candidate.attribution
  out.driverConfounded = candidate.driver_confounded
  out.roadConfounded = candidate.road_confounded
  out.requestedContextBefore = CONTEXT_BEFORE
  out.requestedContextAfter = CONTEXT_AFTER
  out.gitCommit = git_commit
  out.gitBranch = git_branch
  out.detectedMonoTime = candidate.detected_mono_time or candidate.occurred_mono_time
  out.episodeStartMonoTime = candidate.episode_start_mono_time or candidate.occurred_mono_time
  out.analysisWindowBeforeS = candidate.analysis_window_before_s
  out.analysisWindowAfterS = candidate.analysis_window_after_s
  out.episodeKey = candidate.episode_key

  if candidate.domain == "lateral":
    sample: LateralSample = candidate.payload
    evidence = candidate.detector_evidence
    payload = out.payload.init("lateral")
    payload.controllerVersion = sample.controller_version
    payload.referenceVersion = sample.reference_version
    payload.vEgo = sample.v_ego
    payload.steeringAngleDeg = sample.steering_angle_deg
    payload.steeringRateDeg = sample.steering_rate_deg
    payload.desiredLateralAccel = sample.desired_lateral_accel
    payload.actualLateralAccel = sample.actual_lateral_accel
    payload.requestTorque = sample.request_torque
    payload.appliedTorque = sample.applied_torque
    payload.referenceTargetTorque = sample.reference_target_torque
    payload.referenceRate = sample.reference_rate
    payload.measurementRate = float(getattr(sample, "measurement_rate", 0.0))
    payload.referenceUnwindScale = sample.reference_unwind_scale
    payload.referenceSustainedUnwindScale = sample.reference_sustained_unwind_scale
    payload.unwindEffectivePhase = sample.unwind_effective_phase
    payload.unwindOverspeed = sample.unwind_overspeed
    payload.unwindSameEpisode = sample.unwind_same_episode
    payload.appliedTargetGap = sample.applied_target_gap
    payload.pTerm = sample.p_term
    payload.driverTorque = sample.driver_torque
    payload.steeringPressed = sample.steering_pressed
    payload.steeringTorqueEps = sample.steering_torque_eps
    payload.dampingApplied = sample.damping_applied
    payload.dampingState = sample.damping_state
    payload.triggerDriverConfounded = sample.driver_confounded
    payload.triggerRoadConfounded = sample.road_confounded
    if evidence is not None:
      payload.driverConfoundedFraction = evidence.driver_confounded_fraction
      payload.maxAbsDriverTorque = evidence.max_abs_driver_torque
      payload.steeringPressedAny = evidence.steering_pressed_any
      payload.roadConfoundedFraction = evidence.road_confounded_fraction
      payload.driverConfoundReason = evidence.driver_confound_reason
      payload.evidenceStartMonoTime = round(evidence.evidence_start_mono_time * 1e9)
      payload.evidenceEndMonoTime = round(evidence.evidence_end_mono_time * 1e9)
      payload.stallReleaseCount = evidence.stall_release_count
      payload.releaseOffsetsS = list(evidence.release_offsets_s)
      payload.stallDurationsS = list(evidence.stall_durations_s)
      payload.releasePeakRatesDeg = list(evidence.release_peak_rates_deg)
      payload.stallEpisodePhase = evidence.stall_episode_phase
      payload.lateUnwindDurationS = evidence.late_unwind_duration_s
      payload.previousUnwindEffectivePhase = evidence.previous_unwind_effective_phase
      payload.previousUnwindSameEpisode = evidence.previous_unwind_same_episode
      payload.trackingInactiveTimeS = evidence.tracking_inactive_time_s
      payload.rawTorqueAboveThresholdFraction = evidence.raw_torque_exceeded_fraction
      payload.rawTorqueAboveThresholdLongestS = evidence.longest_raw_torque_duration_s
      payload.steeringPressedFraction = evidence.steering_pressed_fraction
      payload.driverTorquePresentAtTrigger = evidence.driver_torque_active_at_trigger
      payload.steeringPressedAtTrigger = evidence.steering_pressed_at_trigger
      payload.driverInteraction = evidence.driver_interaction
      payload.roadConfoundedAtEventTime = evidence.road_confounded_at_trigger
      payload.maxVerticalAccelDeviation = evidence.max_vertical_accel_deviation
      payload.longestRoadBumpIntervalS = evidence.longest_road_confounded_duration_s
      payload.roadConfoundExtent = evidence.road_interaction
      payload.centerCrossingMonoTime = (
        round(evidence.center_crossing_mono_time * 1e9)
        if evidence.center_crossing_mono_time is not None else 0
      )
      payload.classificationConfirmedMonoTime = candidate.detected_mono_time
      payload.establishedOutsideCenter = evidence.established_outside_center
      payload.centerCrossingWheelRateDeg = evidence.signed_center_rate_deg
      payload.signedCenterCrossingWheelRateDeg = evidence.signed_center_rate_deg
      payload.peakAbsCenterWheelRateDeg = evidence.peak_abs_center_rate_deg
      payload.desiredLateralAccelBeforeCrossing = evidence.desired_lateral_accel_before_crossing
      payload.desiredLateralAccelAfterCrossing = evidence.desired_lateral_accel_after_crossing
      payload.desiredReversalCommitted = evidence.committed_reversal
      payload.desiredReversalCommitMonoTime = (
        round(evidence.desired_reversal_commit_mono_time * 1e9)
        if evidence.desired_reversal_commit_mono_time is not None else 0
      )
      payload.trackingErrorAtCrossing = evidence.tracking_error_at_crossing
      payload.appliedTargetGapAtCrossing = evidence.applied_target_gap_at_crossing
      payload.peakTrackingError = evidence.peak_tracking_error
      payload.peakAppliedTargetGap = evidence.peak_applied_target_gap
      payload.requestedTorqueAtCrossing = evidence.requested_torque_at_crossing
      payload.appliedTorqueAtCrossing = evidence.applied_torque_at_crossing
      payload.referenceTargetTorqueAtCrossing = evidence.reference_target_torque_at_crossing
      payload.unwindEpisodeStartMonoTime = round(
        float(getattr(evidence, "unwind_episode_start_mono_time", 0.0) or 0.0) * 1e9,
      )
      payload.unwindDeficitStartMonoTime = round(
        float(getattr(evidence, "unwind_deficit_start_mono_time", 0.0) or 0.0) * 1e9,
      )
      payload.unwindDeficitDurationS = float(getattr(evidence, "unwind_deficit_duration_s", 0.0))
      payload.initialSteeringAngleDeg = float(getattr(evidence, "initial_steering_angle_deg", 0.0))
      payload.peakSteeringAngleDeg = float(getattr(evidence, "peak_steering_angle_deg", 0.0))
      payload.triggerSteeringAngleDeg = float(getattr(evidence, "trigger_steering_angle_deg", 0.0))
      payload.expectedUnwindDirection = str(getattr(evidence, "expected_unwind_direction", ""))
      payload.expectedAngleProgressDeg = float(getattr(evidence, "expected_angle_progress_deg", 0.0))
      payload.actualAngleProgressDeg = float(getattr(evidence, "actual_angle_progress_deg", 0.0))
      payload.unwindProgressRatio = float(getattr(evidence, "unwind_progress_ratio", 0.0))
      payload.referenceRateAtTrigger = float(getattr(evidence, "reference_rate_at_trigger", 0.0))
      payload.measurementRateAtTrigger = float(getattr(evidence, "measurement_rate_at_trigger", 0.0))
      payload.peakReferenceMeasurementRateGap = float(
        getattr(evidence, "peak_reference_measurement_rate_gap", 0.0),
      )
      requested_neutral_cross = getattr(evidence, "requested_torque_neutral_cross_mono_time", None)
      requested_neutral_cross_present = (
        requested_neutral_cross is not None and float(requested_neutral_cross) > 0.0
      )
      payload.requestedTorqueNeutralCrossPresent = requested_neutral_cross_present
      payload.requestedTorqueNeutralCrossMonoTime = (
        round(float(requested_neutral_cross) * 1e9) if requested_neutral_cross_present else 0
      )
      applied_neutral_cross = getattr(evidence, "applied_torque_neutral_cross_mono_time", None)
      applied_neutral_cross_present = (
        applied_neutral_cross is not None and float(applied_neutral_cross) > 0.0
      )
      payload.appliedTorqueNeutralCrossPresent = applied_neutral_cross_present
      payload.appliedTorqueNeutralCrossMonoTime = (
        round(float(applied_neutral_cross) * 1e9) if applied_neutral_cross_present else 0
      )
      payload.unwindCommandDelayS = float(getattr(evidence, "unwind_command_delay_s", 0.0))
      payload.driverAssistedUnwind = bool(getattr(evidence, "driver_assisted_unwind", False))
      driver_assist_time = getattr(evidence, "driver_assist_mono_time", None)
      payload.driverAssistMonoTime = (
        round(float(driver_assist_time) * 1e9) if driver_assist_time is not None else 0
      )
      payload.driverAssistAngleDeg = float(getattr(evidence, "driver_assist_angle_deg", 0.0))
      payload.driverAssistWheelRateDeg = float(getattr(evidence, "driver_assist_wheel_rate_deg", 0.0))
      payload.driverAssistTorque = float(getattr(evidence, "driver_assist_torque", 0.0))
      payload.wheelRateIncreaseAfterAssistDegS = float(
        getattr(evidence, "wheel_rate_increase_after_assist_deg_s", 0.0),
      )
      payload.movementStartMonoTime = round(
        float(getattr(evidence, "movement_start_mono_time", 0.0) or 0.0) * 1e9,
      )
      payload.dwellStartMonoTime = round(
        float(getattr(evidence, "dwell_start_mono_time", 0.0) or 0.0) * 1e9,
      )
      payload.releaseMonoTime = round(
        float(getattr(evidence, "release_mono_time", 0.0) or 0.0) * 1e9,
      )
      payload.dwellDurationS = float(getattr(evidence, "dwell_duration_s", 0.0))
      payload.dwellSteeringAngleDeg = float(getattr(evidence, "dwell_steering_angle_deg", 0.0))
      payload.preDwellPeakRateDegS = float(getattr(evidence, "pre_dwell_peak_rate_deg_s", 0.0))
      payload.minimumDwellRateDegS = float(getattr(evidence, "minimum_dwell_rate_deg_s", 0.0))
      payload.releasePeakRateDegS = float(getattr(evidence, "release_peak_rate_deg_s", 0.0))
      payload.rateReductionRatio = float(getattr(evidence, "rate_reduction_ratio", 0.0))
      payload.restartClassification = str(getattr(evidence, "restart_classification", ""))
      payload.requestTorqueDuringDwell = float(getattr(evidence, "request_torque_during_dwell", 0.0))
      payload.appliedTorqueDuringDwell = float(getattr(evidence, "applied_torque_during_dwell", 0.0))
      payload.referenceTargetDuringDwell = float(
        getattr(evidence, "reference_target_during_dwell", 0.0),
      )
      payload.trackingActiveDuringDwell = bool(getattr(evidence, "tracking_active_during_dwell", False))
      payload.driverInvolvedBeforeDwell = bool(getattr(evidence, "driver_involved_before_dwell", False))
      payload.driverInvolvedDuringDwell = bool(getattr(evidence, "driver_involved_during_dwell", False))
      payload.driverInvolvedAfterDwell = bool(getattr(evidence, "driver_involved_after_dwell", False))
      turn_stop_damping_amount = getattr(evidence, "turn_stop_actual_damping_amount", None)
      turn_stop_damping_state = getattr(evidence, "turn_stop_actual_damping_state", None)
      turn_stop_damping_version = getattr(evidence, "turn_stop_damping_version", None)
      payload.turnStopActualDampingAmount = (
        float(turn_stop_damping_amount) if turn_stop_damping_amount is not None else 0.0
      )
      payload.turnStopActualDampingState = (
        str(turn_stop_damping_state) if turn_stop_damping_state is not None else "unavailable"
      )
      payload.turnStopTurnInBlocked = bool(getattr(evidence, "turn_stop_turn_in_blocked", False))
      turn_stop_breakaway_latch = getattr(evidence, "turn_stop_breakaway_latch", None)
      payload.turnStopBreakawayLatch = (
        float(turn_stop_breakaway_latch) if turn_stop_breakaway_latch is not None else 0.0
      )
      turn_stop_floor = getattr(evidence, "turn_stop_sustain_floor_contribution", None)
      payload.turnStopSustainFloorContribution = (
        float(turn_stop_floor) if turn_stop_floor is not None else 0.0
      )
      payload.turnStopDampingVersion = (
        int(turn_stop_damping_version) if turn_stop_damping_version is not None else 0
      )
      inferred_turn_stop_damping_valid = bool(
        turn_stop_damping_version is not None
        and int(turn_stop_damping_version) > 0
        and turn_stop_damping_amount is not None
        and turn_stop_damping_state is not None
      )
      turn_stop_damping_valid = getattr(evidence, "turn_stop_damping_valid", None)
      payload.turnStopDampingValid = (
        inferred_turn_stop_damping_valid
        if turn_stop_damping_valid is None else bool(turn_stop_damping_valid)
      )
      payload.unwindRequestedTorqueAtTrigger = float(
        getattr(evidence, "unwind_requested_torque_at_trigger", 0.0),
      )
      payload.unwindAppliedTorqueAtTrigger = float(
        getattr(evidence, "unwind_applied_torque_at_trigger", 0.0),
      )
      payload.unwindReferenceTargetTorqueAtTrigger = float(
        getattr(evidence, "unwind_reference_target_torque_at_trigger", 0.0),
      )
      payload.unwindAppliedTargetGapAtTrigger = float(
        getattr(evidence, "unwind_applied_target_gap_at_trigger", 0.0),
      )
      payload.unwindPeakAppliedTargetGap = float(
        getattr(evidence, "unwind_peak_applied_target_gap", 0.0),
      )
      payload.handoffObserved = evidence.phase_handoff_mono_time is not None
      payload.handoffMonoTime = (
        round(evidence.phase_handoff_mono_time * 1e9)
        if evidence.phase_handoff_mono_time is not None else 0
      )
      payload.handoffCenterDeltaS = evidence.handoff_center_delta_s
      payload.handoffConsolidated = evidence.handoff_consolidated
      # v6 crown-neutral trigger snapshots (@127-@134).
      payload.unwindNeutralTorque = float(getattr(evidence, "unwind_neutral_torque", 0.0))
      payload.unwindPhaseDirection = float(getattr(evidence, "unwind_phase_direction", 0.0))
      high_angle_valid = bool(getattr(evidence, "high_angle_evidence_valid", False))
      payload.highAngleEvidenceValid = high_angle_valid
      payload.highAngleUnwindScale = (
        float(getattr(evidence, "high_angle_unwind_scale", 0.0)) if high_angle_valid else 0.0
      )
      payload.torqueCommandBeforeHighAngleExit = (
        float(getattr(evidence, "torque_command_before_high_angle_exit", 0.0)) if high_angle_valid else 0.0
      )
      payload.highAngleUnwindOldTorqueCorrection = (
        float(getattr(evidence, "high_angle_unwind_old_torque_correction", 0.0)) if high_angle_valid else 0.0
      )
      payload.highAngleUnwindOldDirectionTorque = (
        float(getattr(evidence, "high_angle_unwind_old_direction_torque", 0.0)) if high_angle_valid else 0.0
      )
      payload.oldTurnSign = float(getattr(evidence, "old_turn_sign", 0.0))
      # v6 per-episode timestamps (@135-@149). Present test is `is not None`,
      # NOT `> 0.0`: unlike the legacy literal-zero torque crossings above,
      # these are real mono times that could theoretically land on exactly
      # 0.0 in a synthetic test and are never absent-vs-zero-ambiguous the
      # way torque is, so `> 0.0` would wrongly hide a legitimate zero time.
      future_unwind_commit = getattr(evidence, "future_unwind_commit_mono_time", None)
      future_unwind_commit_present = future_unwind_commit is not None
      payload.futureUnwindCommitPresent = future_unwind_commit_present
      payload.futureUnwindCommitMonoTime = (
        round(float(future_unwind_commit) * 1e9) if future_unwind_commit_present else 0
      )
      high_angle_exit_first_nonzero = getattr(evidence, "high_angle_exit_first_nonzero_mono_time", None)
      high_angle_exit_first_nonzero_present = high_angle_exit_first_nonzero is not None
      payload.highAngleExitFirstNonzeroPresent = high_angle_exit_first_nonzero_present
      payload.highAngleExitFirstNonzeroMonoTime = (
        round(float(high_angle_exit_first_nonzero) * 1e9) if high_angle_exit_first_nonzero_present else 0
      )
      requested_crown_neutral = getattr(evidence, "requested_crown_neutral_mono_time", None)
      requested_crown_neutral_present = requested_crown_neutral is not None
      payload.requestedCrownNeutralPresent = requested_crown_neutral_present
      payload.requestedCrownNeutralMonoTime = (
        round(float(requested_crown_neutral) * 1e9) if requested_crown_neutral_present else 0
      )
      applied_crown_neutral = getattr(evidence, "applied_crown_neutral_mono_time", None)
      applied_crown_neutral_present = applied_crown_neutral is not None
      payload.appliedCrownNeutralPresent = applied_crown_neutral_present
      payload.appliedCrownNeutralMonoTime = (
        round(float(applied_crown_neutral) * 1e9) if applied_crown_neutral_present else 0
      )
      payload.unwindCrownCommandDelayS = float(getattr(evidence, "unwind_crown_command_delay_s", 0.0))
      wheel_progress_5 = getattr(evidence, "wheel_progress_5_mono_time", None)
      wheel_progress_5_present = wheel_progress_5 is not None
      payload.wheelProgress5Present = wheel_progress_5_present
      payload.wheelProgress5MonoTime = round(float(wheel_progress_5) * 1e9) if wheel_progress_5_present else 0
      wheel_progress_20 = getattr(evidence, "wheel_progress_20_mono_time", None)
      wheel_progress_20_present = wheel_progress_20 is not None
      payload.wheelProgress20Present = wheel_progress_20_present
      payload.wheelProgress20MonoTime = round(float(wheel_progress_20) * 1e9) if wheel_progress_20_present else 0
      wheel_progress_50 = getattr(evidence, "wheel_progress_50_mono_time", None)
      wheel_progress_50_present = wheel_progress_50 is not None
      payload.wheelProgress50Present = wheel_progress_50_present
      payload.wheelProgress50MonoTime = round(float(wheel_progress_50) * 1e9) if wheel_progress_50_present else 0
      # v6 old-torque rebound tracker snapshot (@150-@154).
      payload.unwindReboundMaxMagnitude = float(getattr(evidence, "unwind_rebound_max_magnitude", 0.0))
      unwind_rebound_start = getattr(evidence, "unwind_rebound_start_mono_time", None)
      unwind_rebound_start_present = unwind_rebound_start is not None
      payload.unwindReboundStartPresent = unwind_rebound_start_present
      payload.unwindReboundStartMonoTime = (
        round(float(unwind_rebound_start) * 1e9) if unwind_rebound_start_present else 0
      )
      payload.unwindReboundDurationS = float(getattr(evidence, "unwind_rebound_duration_s", 0.0))
      payload.unwindReboundSameEpisode = bool(getattr(evidence, "unwind_rebound_same_episode", False))
      # v6 driver causation (@155-@160, @165).
      payload.driverActiveBeforeDeficit = bool(getattr(evidence, "driver_active_before_deficit", False))
      payload.driverActiveAtDeficitStart = bool(getattr(evidence, "driver_active_at_deficit_start", False))
      payload.driverActiveDuringEvaluation = bool(getattr(evidence, "driver_active_during_evaluation", False))
      payload.driverIntervenedAfterDeficit = bool(getattr(evidence, "driver_intervened_after_deficit", False))
      payload.driverInterventionAcceleratedProgress = bool(
        getattr(evidence, "driver_intervention_accelerated_progress", False),
      )
      payload.driverCausation = str(getattr(evidence, "driver_causation", ""))
      payload.driverAssistRawTorqueOnly = bool(getattr(evidence, "driver_assist_raw_torque_only", False))
      # v6 turn-stop lifetime progress (@161-@162).
      payload.turnStopPreDwellProgressDeg = float(getattr(evidence, "turn_stop_pre_dwell_progress_deg", 0.0))
      payload.turnStopPostDwellProgressDeg = float(getattr(evidence, "turn_stop_post_dwell_progress_deg", 0.0))
      # v6 consolidated road-metrics window start actually covered (@163-@164).
      road_evidence_window_start = getattr(evidence, "road_evidence_window_start_mono_time", None)
      road_evidence_window_start_present = road_evidence_window_start is not None
      payload.roadEvidenceWindowStartPresent = road_evidence_window_start_present
      payload.roadEvidenceWindowStartMonoTime = (
        round(float(road_evidence_window_start) * 1e9) if road_evidence_window_start_present else 0
      )
      # v7 under-delivery/takeover onset context (@166-@171).
      payload.demandedCurvature = float(getattr(evidence, "demanded_curvature", 0.0))
      payload.deliveredCurvatureFraction = float(
        getattr(evidence, "delivered_curvature_fraction", 0.0),
      )
      payload.torqueHeadroom = float(getattr(evidence, "torque_headroom", 0.0))
      payload.signedTrackingDeficit = float(
        getattr(evidence, "signed_tracking_deficit", 0.0),
      )
      payload.authorityUnderDeliveryDurationS = float(
        getattr(evidence, "authority_under_delivery_duration_s", 0.0),
      )
      payload.takeoverConfirmationDurationS = float(
        getattr(evidence, "takeover_confirmation_duration_s", 0.0),
      )
      releases = payload.init("stallReleases", len(evidence.stall_releases))
      for index, release in enumerate(evidence.stall_releases):
        releases[index].releaseMonoTime = round(release.mono_time * 1e9)
        releases[index].offsetFromTriggerS = release.offset_from_trigger_s
        releases[index].stallDurationS = release.stall_duration_s
        releases[index].peakSignedSteeringRateDeg = release.peak_signed_rate_deg
        releases[index].peakAbsSteeringRateDeg = release.peak_abs_rate_deg
        releases[index].steeringAngleDeg = release.steering_angle_deg
        releases[index].vEgo = release.v_ego
        releases[index].desiredLateralAccel = release.desired_lateral_accel
        releases[index].actualLateralAccel = release.actual_lateral_accel
        releases[index].requestedTorque = release.request_torque
        releases[index].appliedTorque = release.applied_torque
        releases[index].referenceTargetTorque = release.reference_target_torque
        releases[index].dampingApplied = (
          release.actual_damping_amount if release.actual_damping_amount is not None else 0.0
        )
        releases[index].dampingState = release.actual_damping_state or "unavailable"
        releases[index].turnInBlocked = bool(release.turn_in_blocked)
        releases[index].breakawayLatch = (
          release.breakaway_latch if release.breakaway_latch is not None else 0.0
        )
        releases[index].sustainFloor = (
          release.sustain_floor_contribution if release.sustain_floor_contribution is not None else 0.0
        )
        releases[index].dampingVersion = release.damping_version or 0
        releases[index].dampingValid = bool(
          release.damping_version is not None
          and release.damping_version > 0
          and release.actual_damping_amount is not None
          and release.actual_damping_state is not None
        )
        releases[index].driverEvidenceActive = release.driver_evidence_active
        releases[index].roadEvidenceActive = release.road_evidence_active
        releases[index].phase = release.phase
  elif candidate.event_type == "long.stopJolt":
    stop_event: StopJoltEvent = candidate.payload
    payload = out.payload.init("stopJolt")
    payload.episodeStartMonoTime = round(stop_event.episode_start_mono_time * 1e9)
    payload.standstillMonoTime = round(stop_event.standstill_mono_time * 1e9)
    payload.peakJoltMonoTime = round(stop_event.peak_jolt_mono_time * 1e9)
    payload.detectionMonoTime = round(stop_event.detection_mono_time * 1e9)
    payload.imuJerk = stop_event.imu_jerk
    payload.absImuJerk = stop_event.abs_imu_jerk
    payload.aEgoJerk = stop_event.a_ego_jerk
    payload.absAEgoJerk = stop_event.abs_a_ego_jerk
    payload.imuAccelBefore = stop_event.imu_accel_before
    payload.imuAccelAfter = stop_event.imu_accel_after
    payload.aEgoAccelBefore = stop_event.a_ego_accel_before
    payload.aEgoAccelAfter = stop_event.a_ego_accel_after
    payload.accelChange = stop_event.accel_change
    payload.accelAt02Mps = stop_event.accel_at_02_mps
    payload.vEgoAtPeak = stop_event.v_ego_at_peak
    payload.planATarget = stop_event.plan_a_target
    payload.requestedAccel = stop_event.requested_accel
    payload.appliedAccel = stop_event.applied_accel
    payload.planAccelChange = stop_event.plan_accel_change
    payload.requestedAccelChange = stop_event.requested_accel_change
    payload.appliedAccelChange = stop_event.applied_accel_change
    payload.shouldStopBefore = stop_event.should_stop_before
    payload.shouldStopAtPeak = stop_event.should_stop_at_peak
    payload.shouldStopAfter = stop_event.should_stop_after
    payload.longControlStateBefore = stop_event.long_control_state_before
    payload.longControlStateAtPeak = stop_event.long_control_state_at_peak
    payload.longControlStateAfter = stop_event.long_control_state_after
    payload.leadPresent = stop_event.lead_present
    payload.dRel = stop_event.d_rel
    payload.vLeadK = stop_event.v_lead_k
    payload.brakePressed = stop_event.brake_pressed
    payload.gasPressed = stop_event.gas_pressed
    payload.brakeHoldActive = stop_event.brake_hold_active
    payload.radarValid = stop_event.radar_valid
    payload.imuValid = stop_event.imu_valid
    payload.roadConfounded = stop_event.road_confounded
    payload.classification = stop_event.classification
  elif candidate.event_type.startswith("long.lateRollingLeadResponse"):
    rolling_event: RollingLeadResponseEvent = candidate.payload
    payload = out.payload.init("rollingLeadResponse")
    payload.attributionDetail = rolling_event.attribution_detail
    payload.leadCommitMonoTime = round(rolling_event.lead_commit_mono_time * 1e9)
    payload.plannerResponseMonoTime = (
      round(rolling_event.planner_response_mono_time * 1e9)
      if rolling_event.planner_response_mono_time is not None else 0
    )
    payload.controllerResponseMonoTime = (
      round(rolling_event.controller_response_mono_time * 1e9)
      if rolling_event.controller_response_mono_time is not None else 0
    )
    payload.egoResponseMonoTime = (
      round(rolling_event.ego_response_mono_time * 1e9)
      if rolling_event.ego_response_mono_time is not None else 0
    )
    payload.plannerResponsePresent = rolling_event.planner_response_mono_time is not None
    payload.controllerResponsePresent = rolling_event.controller_response_mono_time is not None
    payload.egoResponsePresent = rolling_event.ego_response_mono_time is not None
    payload.detectedMonoTime = round(rolling_event.detected_mono_time * 1e9)
    payload.leadToPlanS = rolling_event.lead_to_plan_s if rolling_event.lead_to_plan_s is not None else 0.0
    payload.leadToCommandS = rolling_event.lead_to_command_s if rolling_event.lead_to_command_s is not None else 0.0
    payload.leadToEgoS = rolling_event.lead_to_ego_s if rolling_event.lead_to_ego_s is not None else 0.0
    payload.baselineLeadSpeed = rolling_event.baseline_lead_speed
    payload.peakLeadSpeed = rolling_event.peak_lead_speed
    payload.baselineEgoSpeed = rolling_event.baseline_ego_speed
    payload.peakEgoSpeed = rolling_event.peak_ego_speed
    payload.peakRelativeSpeed = rolling_event.peak_relative_speed
    payload.baselineGap = rolling_event.baseline_gap
    payload.finalGap = rolling_event.final_gap
    payload.maxGapGrowth = rolling_event.max_gap_growth
    payload.baselineLeadAccel = rolling_event.baseline_lead_accel
    payload.peakLeadAccel = rolling_event.peak_lead_accel
    payload.baselinePlannerAccel = rolling_event.baseline_planner_accel
    payload.peakPlannerAccel = rolling_event.peak_planner_accel
    payload.baselineOutputAccel = rolling_event.baseline_output_accel
    payload.peakOutputAccel = rolling_event.peak_output_accel
    payload.baselineEgoAccel = rolling_event.baseline_ego_accel
    payload.peakEgoAccel = rolling_event.peak_ego_accel
    payload.radarTrackId = rolling_event.radar_track_id
    payload.radarDiscontinuity = rolling_event.radar_discontinuity
    payload.driverConfounded = rolling_event.driver_confounded
    onsets = payload.init("onsets", len(rolling_event.onsets))
    onset_names = {
      "leadCommit": "leadCommit",
      "plan": "plannerResponse",
      "command": "controllerResponse",
      "ego": "egoResponse",
    }
    for index, onset in enumerate(rolling_event.onsets):
      onsets[index].kind = onset_names[onset.kind]
      onsets[index].monoTime = round(onset.t * 1e9)
      onsets[index].dRel = onset.d_rel
      onsets[index].vLead = onset.v_lead
      onsets[index].vLeadK = onset.v_lead_k
      onsets[index].aLeadK = onset.a_lead_k
      onsets[index].vEgo = onset.v_ego
      onsets[index].aEgo = onset.a_ego
      onsets[index].plannerAccel = onset.a_target
      onsets[index].outputAccel = onset.output_accel
      onsets[index].shouldStop = onset.should_stop
      onsets[index].longActive = True
      onsets[index].radarValid = True
      onsets[index].gasPressed = onset.gas_pressed
      onsets[index].brakePressed = onset.brake_pressed
  elif candidate.domain == "longitudinal":
    long_event: LongEvent = candidate.payload
    payload = out.payload.init("leadLaunch")
    payload.forecastToLeadS = long_event.forecast_to_lead_s
    payload.planToLeadS = long_event.plan_to_lead_s
    payload.commandToLeadS = long_event.command_to_lead_s
    payload.leadToEgoS = long_event.lead_to_ego_s
    payload.commandToEgoS = long_event.command_to_ego_s
    payload.radarDiscontinuity = long_event.radar_discontinuity
    payload.radarConfidence = long_event.confidence
    payload.attributionDetail = long_event.attribution_detail
    onsets = payload.init("onsets", len(long_event.onsets))
    for index, onset in enumerate(long_event.onsets):
      onsets[index].kind = onset.kind
      onsets[index].monoTime = round(onset.t * 1e9)
      onsets[index].dRel = onset.d_rel
      onsets[index].vLead = onset.v_lead
      onsets[index].vEgo = onset.v_ego
      onsets[index].aEgo = onset.a_ego
      onsets[index].outputAccel = onset.output_accel
      onsets[index].brakePressed = onset.brake_pressed
      onsets[index].brakeHoldActive = onset.brake_hold_active
  else:
    out.payload.none = None
  return msg


class EventSubmitter:
  """Retain accepted events until loggerd acknowledges their stable event IDs."""

  def __init__(self, pm: messaging.PubMaster, git_commit: str = "", git_branch: str = "",
               retry_interval_ns: int = ACK_RETRY_INTERVAL_NS):
    self.pm = pm
    self.git_commit = git_commit
    self.git_branch = git_branch
    self.retry_interval_ns = retry_interval_ns
    self.pending: dict[str, PendingEvent] = {}

  def _send(self, pending: PendingEvent, now_ns: int) -> bool:
    pending.last_sent_ns = now_ns
    try:
      self.pm.send("drivingEvent", build_message(pending.event, self.git_commit, self.git_branch))
      return True
    except Exception:
      cloudlog.exception(f"driving_eventd: failed to submit event {pending.event.event_id}")
      return False

  def submit(self, event: AcceptedEvent, now_ns: int | None = None) -> None:
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    pending = PendingEvent(event, now_ns)
    self.pending[event.event_id] = pending
    self._send(pending, now_ns)

  def acknowledge(self, event_id: str) -> bool:
    return self.pending.pop(event_id, None) is not None

  def retry_due(self, now_ns: int | None = None) -> list[str]:
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    retried: list[str] = []
    for event_id, pending in self.pending.items():
      if now_ns - pending.last_sent_ns >= self.retry_interval_ns:
        pending.attempts += 1
        self._send(pending, now_ns)
        retried.append(event_id)
    return retried


def log_retry_warning(event_id: object) -> None:
  """Log a retry after converting Cap'n Proto/string-like IDs at the boundary.

  ``cloudlog.warning`` is the standard logging interface, not the structured
  ``cloudlog.event`` interface. Passing a DynamicStruct (or any other
  non-primitive) as a keyword argument reaches the cereal encoder and can
  terminate this passive process. Rendering here keeps the log useful and the
  process independent of the ID object's implementation type.
  """
  cloudlog.warning(
    f"driving_eventd: retrying unacknowledged event {event_id}"
  )


def longitudinal_sample(sm: messaging.SubMaster) -> LaunchSample:
  lead = sm["radarState"].leadOne
  leads_v3 = sm["modelV2"].leadsV3
  forecast_valid = sm.valid["modelV2"] and len(leads_v3) > 0 and leads_v3[0].prob > 0.5 and len(leads_v3[0].v) > 1
  predicted_lead_v_2s = 0.0
  if forecast_valid:
    predicted_lead_v_2s = float(lead.vLead) + float(leads_v3[0].v[1]) - float(leads_v3[0].v[0])

  now = sm.logMonoTime["carState"] * 1e-9 if sm.logMonoTime["carState"] else time.monotonic()
  return LaunchSample(
    t=now,
    active=bool(sm.valid["carState"] and sm.valid["carControl"] and sm["carControl"].longActive),
    standstill=bool(sm["carState"].standstill),
    v_ego=float(sm["carState"].vEgo),
    lead_present=bool(lead.present),
    radar_valid=bool(sm.valid["radarState"]),
    d_rel=float(lead.dRel),
    v_lead=float(lead.vLead),
    v_lead_k=float(lead.vLeadK),
    radar_track_id=int(lead.radarTrackId),
    plan_valid=bool(sm.valid["longitudinalPlan"]),
    plan_should_stop=bool(sm["longitudinalPlan"].shouldStop),
    output_valid=bool(sm.valid["carOutput"]),
    output_accel=float(sm["carOutput"].actuatorsOutput.accel),
    forecast_valid=forecast_valid,
    predicted_lead_v_2s=predicted_lead_v_2s,
    a_ego=float(getattr(sm["carState"], "aEgo", 0.0)),
    brake_pressed=bool(getattr(sm["carState"], "brakePressed", False)),
    brake_hold_active=bool(getattr(sm["carState"], "brakeHoldActive", False)),
  )


def rolling_lead_sample(sm: messaging.SubMaster) -> RollingLeadSample:
  car_state = sm["carState"]
  car_control = sm["carControl"]
  lead = sm["radarState"].leadOne
  plan = sm["longitudinalPlan"]
  output = sm["carOutput"].actuatorsOutput
  now = sm.logMonoTime["carState"] * 1e-9 if sm.logMonoTime["carState"] else time.monotonic()
  numeric_values = (
    float(car_state.vEgo),
    float(car_state.aEgo),
    float(lead.dRel),
    float(lead.vLead),
    float(lead.vLeadK),
    float(lead.aLeadK),
    float(plan.aTarget),
    float(output.accel),
  )
  radar_valid = bool(
    sm.valid["radarState"]
    and all(math.isfinite(value) for value in numeric_values[2:6])
  )
  track_id = int(lead.radarTrackId)
  return RollingLeadSample(
    t=now,
    long_active=bool(sm.valid["carState"] and sm.valid["carControl"] and car_control.longActive),
    v_ego=numeric_values[0],
    a_ego=numeric_values[1],
    gas_pressed=bool(car_state.gasPressed),
    brake_pressed=bool(car_state.brakePressed),
    lead_present=bool(lead.present),
    lead_valid=bool(radar_valid and lead.present and lead.radar and track_id >= 0),
    radar_valid=radar_valid,
    radar_track_id=track_id,
    d_rel=numeric_values[2],
    v_lead=numeric_values[3],
    v_lead_k=numeric_values[4],
    a_lead_k=numeric_values[5],
    plan_valid=bool(sm.valid["longitudinalPlan"] and math.isfinite(numeric_values[6])),
    a_target=numeric_values[6],
    should_stop=bool(plan.shouldStop),
    output_valid=bool(sm.valid["carOutput"] and math.isfinite(numeric_values[7])),
    output_accel=numeric_values[7],
  )


def live_pose_sample(sm: messaging.SubMaster, bump_classifier: RoadBumpClassifier) -> StopJoltImuSample:
  """Sample IMU once per actual deviceMotion update, using deviceMotion's timestamp."""
  live_pose = sm["deviceMotion"]
  acceleration = live_pose.accelerationDevice
  now = sm.logMonoTime["deviceMotion"] * 1e-9
  accel_x = float(acceleration.x)
  accel_z = float(acceleration.z)
  accel_x_valid = bool(acceleration.valid and math.isfinite(accel_x))
  accel_z_valid = bool(acceleration.valid and math.isfinite(accel_z))
  imu_valid = bool(
    sm.valid["deviceMotion"]
    and accel_x_valid
    and live_pose.inputsOK
    and live_pose.sensorsOK
  )
  road_confounded = (
    bump_classifier.update(accel_z, now)
    if accel_z_valid
    else now < bump_classifier.confounded_until
  )
  return StopJoltImuSample(
    t=now,
    accel_x=accel_x,
    valid=imu_valid,
    road_confounded=road_confounded,
  )


def stop_jolt_car_sample(sm: messaging.SubMaster, road_confounded: bool) -> StopJoltCarSample:
  car_state = sm["carState"]
  car_control = sm["carControl"]
  car_output = sm["carOutput"]
  plan = sm["longitudinalPlan"]
  controls_state = sm["controlsState"]
  lead = sm["radarState"].leadOne
  now = sm.logMonoTime["carState"] * 1e-9
  numeric_values = (
    float(car_state.vEgo),
    float(car_state.aEgo),
    float(plan.aTarget),
    float(car_control.actuators.accel),
    float(car_output.actuatorsOutput.accel),
  )
  required_valid = bool(
    all(sm.valid[service] for service in ("carState", "carControl", "carOutput", "controlsState", "longitudinalPlan"))
    and all(math.isfinite(value) for value in numeric_values)
  )
  d_rel = float(lead.dRel)
  v_lead_k = float(lead.vLeadK)
  radar_valid = bool(sm.valid["radarState"] and math.isfinite(d_rel) and math.isfinite(v_lead_k))
  return StopJoltCarSample(
    t=now,
    required_valid=required_valid,
    long_active=bool(car_control.longActive),
    should_stop=bool(plan.shouldStop),
    v_ego=numeric_values[0],
    a_ego=numeric_values[1],
    standstill=bool(car_state.standstill),
    brake_pressed=bool(car_state.brakePressed),
    gas_pressed=bool(car_state.gasPressed),
    brake_hold_active=bool(car_state.brakeHoldActive),
    plan_a_target=numeric_values[2],
    requested_accel=numeric_values[3],
    applied_accel=numeric_values[4],
    long_control_state=str(controls_state.longControlState),
    lead_present=bool(lead.present),
    d_rel=d_rel if math.isfinite(d_rel) else 0.0,
    v_lead_k=v_lead_k if math.isfinite(v_lead_k) else 0.0,
    radar_valid=radar_valid,
    road_confounded=road_confounded,
  )


def lateral_sample(sm: messaging.SubMaster, rate_filter: SteeringRateFilter,
                   road_confounded: bool, vertical_accel_deviation: float = 0.0) -> LateralSample:
  controls_state = sm["controlsState"]
  torque_state = controls_state.lateralControlState.torqueState if controls_state.lateralControlState.which() == "torqueState" else None
  car_state = sm["carState"]
  actuators_output = sm["carOutput"].actuatorsOutput
  now = sm.logMonoTime["controlsState"] * 1e-9 if sm.logMonoTime["controlsState"] else time.monotonic()
  angle = float(car_state.steeringAngleDeg)
  damping_version = int(getattr(actuators_output, "torqueDampingVersion", 0))
  damping_valid = bool(sm.valid["carOutput"] and damping_version > 0)
  actual_damping_amount = (
    float(getattr(actuators_output, "torqueDampingApplied", 0.0)) if damping_valid else None
  )
  actual_damping_state = (
    str(getattr(actuators_output, "torqueDampingState", "inactive")) if damping_valid else None
  )
  # v6 high-angle unwind evidence: torqueState fields, NOT gated on the
  # carOutput damping_valid check above. Grouped under one hasattr guard so
  # all four are present together (float, possibly 0.0) or absent together
  # (None) depending on whether the running schema carries them.
  high_angle_available = torque_state is not None and hasattr(torque_state, "highAngleUnwindScale")
  sample = LateralSample(
    mono_time=now,
    active=bool(torque_state is not None and torque_state.active and sm["carControl"].latActive),
    v_ego=float(car_state.vEgo),
    steering_angle_deg=angle,
    steering_rate_deg=rate_filter.update(angle, now),
    driver_torque=float(car_state.steeringTorque),
    steering_pressed=bool(car_state.steeringPressed),
    desired_lateral_accel=float(torque_state.desiredLateralAccel) if torque_state is not None else 0.0,
    actual_lateral_accel=float(torque_state.actualLateralAccel) if torque_state is not None else 0.0,
    request_torque=float(sm["carControl"].actuators.torque),
    applied_torque=float(sm["carOutput"].actuatorsOutput.torque),
    p_term=float(torque_state.p) if torque_state is not None else 0.0,
    steering_torque_eps=float(car_state.steeringTorqueEps),
    # Versions 2/3 serialized torqueState.d into these legacy fields. Version 4
    # leaves them explicitly unavailable and records actual CarController
    # damping in the append-only fields below.
    damping_applied=0.0,
    damping_state="legacyUnavailable",
    controller_version=int(torque_state.version) if torque_state is not None else 0,
    reference_version=int(getattr(torque_state, "referenceVersion", 0)) if torque_state is not None else 0,
    reference_rate=float(getattr(torque_state, "referenceRate", 0.0)) if torque_state is not None else 0.0,
    measurement_rate=(
      float(getattr(torque_state, "trackingMeasurementRate", 0.0))
      if torque_state is not None else 0.0
    ),
    reference_target_torque=float(getattr(torque_state, "referenceReachableTargetTorque", 0.0)) if torque_state is not None else 0.0,
    reference_unwind_scale=float(getattr(torque_state, "referenceUnwindScale", 0.0)) if torque_state is not None else 0.0,
    reference_sustained_unwind_scale=float(getattr(torque_state, "referenceSustainedUnwindScale", 0.0)) if torque_state is not None else 0.0,
    unwind_effective_phase=float(getattr(torque_state, "unwindEffectivePhase", 0.0)) if torque_state is not None else 0.0,
    unwind_overspeed=float(getattr(torque_state, "unwindPhaseOverspeed", 0.0)) if torque_state is not None else 0.0,
    unwind_same_episode=bool(getattr(torque_state, "unwindSameEpisode", False)) if torque_state is not None else False,
    unwind_neutral_torque=float(getattr(torque_state, "unwindNeutralTorque", 0.0)) if torque_state is not None else 0.0,
    unwind_phase_direction=float(getattr(torque_state, "unwindPhaseDirection", 0.0)) if torque_state is not None else 0.0,
    high_angle_unwind_scale=(
      float(torque_state.highAngleUnwindScale) if high_angle_available else None
    ),
    torque_command_before_high_angle_exit=(
      float(torque_state.torqueCommandBeforeHighAngleExit) if high_angle_available else None
    ),
    high_angle_unwind_old_torque_correction=(
      float(torque_state.highAngleUnwindOldTorqueCorrection) if high_angle_available else None
    ),
    high_angle_unwind_old_direction_torque=(
      float(torque_state.highAngleUnwindOldDirectionTorque) if high_angle_available else None
    ),
    road_confounded=road_confounded,
    vertical_accel_deviation=vertical_accel_deviation,
    actual_damping_amount=actual_damping_amount,
    actual_damping_state=actual_damping_state,
    turn_in_blocked=(
      bool(getattr(actuators_output, "torqueDampingBlocked", False)) if damping_valid else None
    ),
    breakaway_latch=(
      float(getattr(actuators_output, "torqueBreakawayLatch", 0.0)) if damping_valid else None
    ),
    sustain_floor_contribution=(
      float(getattr(actuators_output, "torqueDampingFloor", 0.0)) if damping_valid else None
    ),
    damping_version=damping_version if damping_valid else None,
  )
  return sample


def main() -> None:
  params = Params()
  git_commit = _param_text(params, "GitCommit")
  git_branch = _param_text(params, "GitBranch")
  pm = messaging.PubMaster(["drivingEvent"])
  ack_sock = messaging.sub_sock("drivingEventRecorded", conflate=False)
  sm = messaging.SubMaster(
    ["bookmarkButton", "carState", "carControl", "carOutput", "controlsState",
     "radarState", "longitudinalPlan", "modelV2", "deviceMotion"],
  )
  platform = DrivingEventPlatform()
  submitter = EventSubmitter(pm, git_commit, git_branch)
  rate_filter = SteeringRateFilter()
  bump_classifier = RoadBumpClassifier()
  road_confounded = False
  vertical_accel_deviation = 0.0

  if not pm.wait_for_readers_to_update("drivingEvent", timeout=10):
    cloudlog.warning("driving_eventd: loggerd reader not ready; retaining events for retry")

  while True:
    sm.update(100)
    for ack_msg in messaging.drain_sock(ack_sock):
      if ack_msg.valid and ack_msg.which() == "drivingEventRecorded":
        submitter.acknowledge(ack_msg.drivingEventRecorded.eventId)

    controls_updated = bool(sm.updated["controlsState"])
    car_state_updated = bool(sm.updated["carState"])
    live_pose_updated = bool(sm.updated["deviceMotion"])
    bookmark_updated = bool(sm.updated["bookmarkButton"])
    imu_sample = live_pose_sample(sm, bump_classifier) if live_pose_updated else None
    if imu_sample is not None:
      road_confounded = imu_sample.road_confounded
      vertical_accel_deviation = bump_classifier.last_deviation
    lat_sample = (
      lateral_sample(sm, rate_filter, road_confounded, vertical_accel_deviation)
      if controls_updated else None
    )
    long_sample = longitudinal_sample(sm) if car_state_updated else None
    rolling_sample = rolling_lead_sample(sm) if car_state_updated else None
    stop_sample = stop_jolt_car_sample(sm, road_confounded) if car_state_updated else None
    events = platform.update(
      lat_sample,
      long_sample,
      manual_pressed=bookmark_updated,
      manual_time_ns=sm.logMonoTime["bookmarkButton"] if bookmark_updated else None,
      stop_car_sample=stop_sample,
      stop_imu_sample=imu_sample,
      rolling_lead_sample=rolling_sample,
    )
    for event in events:
      submitter.submit(event)
      cloudlog.event("driving_event", event_id=event.event_id, group_id=event.group_id,
                     domain=event.candidate.domain, event_type=event.candidate.event_type)
    for event_id in submitter.retry_due():
      log_retry_warning(event_id)


if __name__ == "__main__":
  main()
