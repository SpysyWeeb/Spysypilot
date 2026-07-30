from types import SimpleNamespace

from openpilot.cereal import messaging
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.spysypilot.driving_eventd import (
  AcceptedEvent,
  build_message,
  lateral_candidate,
  longitudinal_candidate,
  manual_candidate,
)
from openpilot.selfdrive.spysypilot.lat_event_detector import (
  LateralDetection,
  LateralEvidence,
  LateralSample,
  StallReleaseEvidence,
)
from openpilot.selfdrive.spysypilot.long_event_detector import LaunchSample, LongEvent, OnsetSnapshot


def _assert_approx_equal(test_case: OpenpilotTestCase, actual: float, expected: float, *, delta: float | None = None) -> None:
  tolerance = max(abs(expected) * 1e-6, 1e-12) if delta is None else delta
  test_case.assertAlmostEqual(actual, expected, delta=tolerance)


def round_trip(msg):
  return messaging.log_from_bytes(msg.to_bytes())


class TestDrivingEventSchema(OpenpilotTestCase):
  def assertSequenceAlmostEqual(self, actual, expected):
    self.assertEqual(len(actual), len(expected))
    for actual_value, expected_value in zip(actual, expected, strict=True):
      _assert_approx_equal(self, actual_value, expected_value)

  def test_manual_event_round_trip(self):
    msg = build_message(AcceptedEvent("event", "group", manual_candidate(123)), "commit", "branch")
    event = round_trip(msg).drivingEvent
    assert event.eventId == "event"
    assert event.groupId == "group"
    assert event.domain == "manual"
    assert event.payload.which() == "none"
    assert event.gitCommit == "commit"
    assert event.detectedMonoTime == 123
    assert event.analysisWindowBeforeS == 15.0

  def test_lateral_payload_round_trip(self):
    sample = LateralSample(
      1.0, v_ego=12.0, steering_angle_deg=4.0, steering_rate_deg=-20.0,
      request_torque=0.5, applied_torque=0.4, reference_target_torque=0.1,
      controller_version=3, reference_version=4, road_confounded=True,
      driver_torque=75.0, steering_torque_eps=123.0, damping_applied=0.2, damping_state="applied",
    )
    sample.measurement_rate = -0.22
    evidence = LateralEvidence(
      0.0, 1.0, 0.5, 100.0, False, 0.25, 1, 0.0, "lat:episode", 2.0, 2.0,
      stall_release_count=3,
      release_offsets_s=(-1.0, -0.5, 0.0),
      stall_durations_s=(0.2, 0.3, 0.4),
      release_peak_rates_deg=(40.0, 50.0, 60.0),
      stall_episode_phase="mixed",
      raw_torque_exceeded_fraction=0.25,
      longest_raw_torque_duration_s=0.44,
      steering_pressed_fraction=0.0,
      driver_torque_active_at_trigger=True,
      driver_interaction="possibleRawTorque",
      road_confounded_at_trigger=True,
      max_vertical_accel_deviation=2.02,
      longest_road_confounded_duration_s=0.84,
      road_interaction="substantial",
      center_crossing_mono_time=0.75,
      phase_handoff_mono_time=0.5,
      committed_reversal=True,
      peak_abs_center_rate_deg=315.8,
      signed_center_rate_deg=300.0,
      established_outside_center=True,
      desired_lateral_accel_before_crossing=0.24,
      desired_lateral_accel_after_crossing=-0.36,
      desired_reversal_commit_mono_time=0.93,
      tracking_error_at_crossing=0.70,
      applied_target_gap_at_crossing=0.30,
      requested_torque_at_crossing=0.55,
      applied_torque_at_crossing=0.42,
      reference_target_torque_at_crossing=0.12,
      peak_tracking_error=0.776,
      peak_applied_target_gap=0.307,
      handoff_consolidated=True,
      handoff_center_delta_s=0.25,
      stall_releases=(StallReleaseEvidence(
        0.8, -0.2, 0.16, -62.5, 62.5, 13.8, 3.4, -0.13, -0.05,
        0.69, 0.52, 0.31, 0.0, "turnInAuthority", True, 0.0, 0.0, 2,
        True, False, "turnIn",
      ),),
    )
    evidence = SimpleNamespace(**(evidence.__dict__ | {
      "driver_confounded_any": evidence.driver_confounded_any,
      "road_confounded_any": evidence.road_confounded_any,
      "unwind_episode_start_mono_time": 0.20,
      "unwind_deficit_start_mono_time": 0.35,
      "unwind_deficit_duration_s": 0.65,
      "initial_steering_angle_deg": 42.0,
      "peak_steering_angle_deg": 45.0,
      "trigger_steering_angle_deg": 31.0,
      "expected_unwind_direction": "towardCenter",
      "expected_angle_progress_deg": 18.0,
      "actual_angle_progress_deg": 4.5,
      "unwind_progress_ratio": 0.25,
      "reference_rate_at_trigger": -0.8,
      "measurement_rate_at_trigger": -0.1,
      "peak_reference_measurement_rate_gap": 0.92,
      "requested_torque_neutral_cross_mono_time": 0.44,
      "applied_torque_neutral_cross_mono_time": 0.59,
      "unwind_command_delay_s": 0.15,
      "driver_assisted_unwind": True,
      "driver_assist_mono_time": 0.70,
      "driver_assist_angle_deg": 28.0,
      "driver_assist_wheel_rate_deg": -22.0,
      "driver_assist_torque": -85.0,
      "wheel_rate_increase_after_assist_deg_s": 35.0,
      "movement_start_mono_time": 0.10,
      "dwell_start_mono_time": 0.40,
      "release_mono_time": 0.90,
      "dwell_duration_s": 0.50,
      "dwell_steering_angle_deg": 12.0,
      "pre_dwell_peak_rate_deg_s": 75.0,
      "minimum_dwell_rate_deg_s": 2.0,
      "release_peak_rate_deg_s": 68.0,
      "rate_reduction_ratio": 0.0267,
      "restart_classification": "oppositeDirection",
      "request_torque_during_dwell": -0.4,
      "applied_torque_during_dwell": -0.3,
      "reference_target_during_dwell": -0.2,
      "tracking_active_during_dwell": True,
      "driver_involved_before_dwell": False,
      "driver_involved_during_dwell": True,
      "driver_involved_after_dwell": False,
      "turn_stop_actual_damping_amount": 0.029,
      "turn_stop_actual_damping_state": "damping",
      "turn_stop_turn_in_blocked": False,
      "turn_stop_breakaway_latch": 0.89,
      "turn_stop_sustain_floor_contribution": 0.80,
      "turn_stop_damping_version": 2,
      "turn_stop_damping_valid": True,
      "unwind_requested_torque_at_trigger": -0.55,
      "unwind_applied_torque_at_trigger": -0.42,
      "unwind_reference_target_torque_at_trigger": -0.12,
      "unwind_applied_target_gap_at_trigger": 0.30,
      "unwind_peak_applied_target_gap": 0.35,
      # v6 census (@127-@165): distinct nonzero values for every new field.
      "unwind_neutral_torque": 0.18,
      "unwind_phase_direction": -1.0,
      "high_angle_evidence_valid": True,
      "high_angle_unwind_scale": 0.42,
      "torque_command_before_high_angle_exit": 0.33,
      "high_angle_unwind_old_torque_correction": 0.11,
      "high_angle_unwind_old_direction_torque": 0.09,
      "old_turn_sign": -1.0,
      "future_unwind_commit_mono_time": 0.05,
      "high_angle_exit_first_nonzero_mono_time": 0.15,
      "requested_crown_neutral_mono_time": 0.48,
      "applied_crown_neutral_mono_time": 0.62,
      "unwind_crown_command_delay_s": 0.38,
      "wheel_progress_5_mono_time": 0.22,
      "wheel_progress_20_mono_time": 0.55,
      "wheel_progress_50_mono_time": 0.95,
      "unwind_rebound_max_magnitude": 0.55,
      "unwind_rebound_start_mono_time": 0.72,
      "unwind_rebound_duration_s": 0.28,
      "unwind_rebound_same_episode": True,
      "driver_active_before_deficit": True,
      "driver_active_at_deficit_start": False,
      "driver_active_during_evaluation": True,
      "driver_intervened_after_deficit": True,
      "driver_intervention_accelerated_progress": True,
      "driver_causation": "interventionBacked",
      "turn_stop_pre_dwell_progress_deg": 14.5,
      "turn_stop_post_dwell_progress_deg": 22.5,
      "road_evidence_window_start_mono_time": 0.08,
      "driver_assist_raw_torque_only": True,
    }))
    candidate = lateral_candidate(sample, LateralDetection(
      "committedHandoffHarshness", "warning", 0.9, "reason", evidence,
      occurred_mono_time=0.75, detected_mono_time=1.0,
    ))
    event = round_trip(build_message(AcceptedEvent("lat", "group", candidate))).drivingEvent
    assert event.eventId == "lat"
    assert event.payload.which() == "lateral"
    assert event.payload.lateral.controllerVersion == 3
    assert event.payload.lateral.steeringRateDeg == -20.0
    _assert_approx_equal(self, event.payload.lateral.appliedTargetGap, 0.3)
    assert event.roadConfounded
    assert event.driverConfounded
    assert event.payload.lateral.driverTorque == 75.0
    self.assertSequenceAlmostEqual(list(event.payload.lateral.stallDurationsS), [0.2, 0.3, 0.4])
    assert event.occurredMonoTime == 750_000_000
    assert event.detectedMonoTime == 1_000_000_000
    assert event.payload.lateral.driverInteraction == "possibleRawTorque"
    assert event.payload.lateral.roadConfoundExtent == "substantial"
    assert event.payload.lateral.centerCrossingMonoTime == event.occurredMonoTime
    assert event.payload.lateral.handoffConsolidated
    _assert_approx_equal(self, event.payload.lateral.peakAbsCenterWheelRateDeg, 315.8)
    _assert_approx_equal(self, event.payload.lateral.requestedTorqueAtCrossing, 0.55)
    _assert_approx_equal(self, event.payload.lateral.appliedTorqueAtCrossing, 0.42)
    _assert_approx_equal(self, event.payload.lateral.referenceTargetTorqueAtCrossing, 0.12)
    assert event.payload.lateral.unwindEpisodeStartMonoTime == 200_000_000
    assert event.payload.lateral.unwindDeficitStartMonoTime == 350_000_000
    _assert_approx_equal(self, event.payload.lateral.unwindDeficitDurationS, 0.65)
    _assert_approx_equal(self, event.payload.lateral.initialSteeringAngleDeg, 42.0)
    _assert_approx_equal(self, event.payload.lateral.peakSteeringAngleDeg, 45.0)
    _assert_approx_equal(self, event.payload.lateral.triggerSteeringAngleDeg, 31.0)
    assert event.payload.lateral.expectedUnwindDirection == "towardCenter"
    _assert_approx_equal(self, event.payload.lateral.expectedAngleProgressDeg, 18.0)
    _assert_approx_equal(self, event.payload.lateral.actualAngleProgressDeg, 4.5)
    _assert_approx_equal(self, event.payload.lateral.unwindProgressRatio, 0.25)
    _assert_approx_equal(self, event.payload.lateral.referenceRateAtTrigger, -0.8)
    _assert_approx_equal(self, event.payload.lateral.measurementRateAtTrigger, -0.1)
    _assert_approx_equal(self, event.payload.lateral.peakReferenceMeasurementRateGap, 0.92)
    assert event.payload.lateral.requestedTorqueNeutralCrossPresent
    assert event.payload.lateral.requestedTorqueNeutralCrossMonoTime == 440_000_000
    assert event.payload.lateral.appliedTorqueNeutralCrossPresent
    assert event.payload.lateral.appliedTorqueNeutralCrossMonoTime == 590_000_000
    _assert_approx_equal(self, event.payload.lateral.unwindCommandDelayS, 0.15)
    assert event.payload.lateral.driverAssistedUnwind
    assert event.payload.lateral.driverAssistMonoTime == 700_000_000
    _assert_approx_equal(self, event.payload.lateral.driverAssistAngleDeg, 28.0)
    _assert_approx_equal(self, event.payload.lateral.driverAssistWheelRateDeg, -22.0)
    _assert_approx_equal(self, event.payload.lateral.driverAssistTorque, -85.0)
    _assert_approx_equal(self, event.payload.lateral.wheelRateIncreaseAfterAssistDegS, 35.0)
    assert event.payload.lateral.movementStartMonoTime == 100_000_000
    assert event.payload.lateral.dwellStartMonoTime == 400_000_000
    assert event.payload.lateral.releaseMonoTime == 900_000_000
    _assert_approx_equal(self, event.payload.lateral.dwellDurationS, 0.5)
    _assert_approx_equal(self, event.payload.lateral.dwellSteeringAngleDeg, 12.0)
    _assert_approx_equal(self, event.payload.lateral.preDwellPeakRateDegS, 75.0)
    _assert_approx_equal(self, event.payload.lateral.minimumDwellRateDegS, 2.0)
    _assert_approx_equal(self, event.payload.lateral.releasePeakRateDegS, 68.0)
    _assert_approx_equal(self, event.payload.lateral.rateReductionRatio, 0.0267)
    assert event.payload.lateral.restartClassification == "oppositeDirection"
    _assert_approx_equal(self, event.payload.lateral.requestTorqueDuringDwell, -0.4)
    _assert_approx_equal(self, event.payload.lateral.appliedTorqueDuringDwell, -0.3)
    _assert_approx_equal(self, event.payload.lateral.referenceTargetDuringDwell, -0.2)
    assert event.payload.lateral.trackingActiveDuringDwell
    _assert_approx_equal(self, event.payload.lateral.measurementRate, -0.22)
    assert not event.payload.lateral.driverInvolvedBeforeDwell
    assert event.payload.lateral.driverInvolvedDuringDwell
    assert not event.payload.lateral.driverInvolvedAfterDwell
    _assert_approx_equal(self, event.payload.lateral.turnStopActualDampingAmount, 0.029)
    assert event.payload.lateral.turnStopActualDampingState == "damping"
    assert not event.payload.lateral.turnStopTurnInBlocked
    _assert_approx_equal(self, event.payload.lateral.turnStopBreakawayLatch, 0.89)
    _assert_approx_equal(self, event.payload.lateral.turnStopSustainFloorContribution, 0.80)
    assert event.payload.lateral.turnStopDampingVersion == 2
    assert event.payload.lateral.turnStopDampingValid
    _assert_approx_equal(self, event.payload.lateral.unwindRequestedTorqueAtTrigger, -0.55)
    _assert_approx_equal(self, event.payload.lateral.unwindAppliedTorqueAtTrigger, -0.42)
    _assert_approx_equal(self, event.payload.lateral.unwindReferenceTargetTorqueAtTrigger, -0.12)
    _assert_approx_equal(self, event.payload.lateral.unwindAppliedTargetGapAtTrigger, 0.30)
    _assert_approx_equal(self, event.payload.lateral.unwindPeakAppliedTargetGap, 0.35)
    release = event.payload.lateral.stallReleases[0]
    assert release.dampingValid
    assert release.dampingState == "turnInAuthority"
    assert release.dampingApplied == 0.0
    assert release.turnInBlocked
    assert event.episodeKey == "lat:episode"
    # v6 census (@127-@165): full field census, Present-bit pins, ns-conversion
    # pins for every new mono time.
    _assert_approx_equal(self, event.payload.lateral.unwindNeutralTorque, 0.18)
    _assert_approx_equal(self, event.payload.lateral.unwindPhaseDirection, -1.0)
    assert event.payload.lateral.highAngleEvidenceValid
    _assert_approx_equal(self, event.payload.lateral.highAngleUnwindScale, 0.42)
    _assert_approx_equal(self, event.payload.lateral.torqueCommandBeforeHighAngleExit, 0.33)
    _assert_approx_equal(self, event.payload.lateral.highAngleUnwindOldTorqueCorrection, 0.11)
    _assert_approx_equal(self, event.payload.lateral.highAngleUnwindOldDirectionTorque, 0.09)
    _assert_approx_equal(self, event.payload.lateral.oldTurnSign, -1.0)
    assert event.payload.lateral.futureUnwindCommitPresent
    assert event.payload.lateral.futureUnwindCommitMonoTime == 50_000_000
    assert event.payload.lateral.highAngleExitFirstNonzeroPresent
    assert event.payload.lateral.highAngleExitFirstNonzeroMonoTime == 150_000_000
    assert event.payload.lateral.requestedCrownNeutralPresent
    assert event.payload.lateral.requestedCrownNeutralMonoTime == 480_000_000
    assert event.payload.lateral.appliedCrownNeutralPresent
    assert event.payload.lateral.appliedCrownNeutralMonoTime == 620_000_000
    _assert_approx_equal(self, event.payload.lateral.unwindCrownCommandDelayS, 0.38)
    assert event.payload.lateral.wheelProgress5Present
    assert event.payload.lateral.wheelProgress5MonoTime == 220_000_000
    assert event.payload.lateral.wheelProgress20Present
    assert event.payload.lateral.wheelProgress20MonoTime == 550_000_000
    assert event.payload.lateral.wheelProgress50Present
    assert event.payload.lateral.wheelProgress50MonoTime == 950_000_000
    _assert_approx_equal(self, event.payload.lateral.unwindReboundMaxMagnitude, 0.55)
    assert event.payload.lateral.unwindReboundStartPresent
    assert event.payload.lateral.unwindReboundStartMonoTime == 720_000_000
    _assert_approx_equal(self, event.payload.lateral.unwindReboundDurationS, 0.28)
    assert event.payload.lateral.unwindReboundSameEpisode
    assert event.payload.lateral.driverActiveBeforeDeficit
    assert not event.payload.lateral.driverActiveAtDeficitStart
    assert event.payload.lateral.driverActiveDuringEvaluation
    assert event.payload.lateral.driverIntervenedAfterDeficit
    assert event.payload.lateral.driverInterventionAcceleratedProgress
    assert event.payload.lateral.driverCausation == "interventionBacked"
    _assert_approx_equal(self, event.payload.lateral.turnStopPreDwellProgressDeg, 14.5)
    _assert_approx_equal(self, event.payload.lateral.turnStopPostDwellProgressDeg, 22.5)
    assert event.payload.lateral.roadEvidenceWindowStartPresent
    assert event.payload.lateral.roadEvidenceWindowStartMonoTime == 80_000_000
    assert event.payload.lateral.driverAssistRawTorqueOnly

  def test_lateral_payload_v6_optional_evidence_absent_serializes_present_false(self):
    # Companion to the census above: when the detector never populates a v6
    # optional field (e.g. no rebound re-excursion occurred), the Present bit
    # must read False and the mono time 0 — distinct from the `is not None`
    # semantics pinned above where an explicit 0.0 mono time still reads
    # Present true.
    sample = LateralSample(1.0)
    evidence = LateralEvidence(
      0.0, 1.0, 0.0, 0.0, False, 0.0, 0, 0.0, "lat:episode", 2.0, 2.0,
    )
    candidate = lateral_candidate(sample, LateralDetection(
      "centerOvershoot", "warning", 0.9, "reason", evidence,
      occurred_mono_time=0.5, detected_mono_time=0.5,
    ))
    event = round_trip(build_message(AcceptedEvent("lat", "group", candidate))).drivingEvent
    assert not event.payload.lateral.highAngleEvidenceValid
    assert event.payload.lateral.highAngleUnwindScale == 0.0
    assert not event.payload.lateral.futureUnwindCommitPresent
    assert event.payload.lateral.futureUnwindCommitMonoTime == 0
    assert not event.payload.lateral.requestedCrownNeutralPresent
    assert not event.payload.lateral.appliedCrownNeutralPresent
    assert not event.payload.lateral.wheelProgress5Present
    assert not event.payload.lateral.wheelProgress20Present
    assert not event.payload.lateral.wheelProgress50Present
    assert not event.payload.lateral.unwindReboundStartPresent
    assert event.payload.lateral.unwindReboundMaxMagnitude == 0.0
    assert event.payload.lateral.driverCausation == ""
    assert not event.payload.lateral.driverAssistRawTorqueOnly
    assert not event.payload.lateral.roadEvidenceWindowStartPresent

  def test_longitudinal_payload_round_trip(self):
    sample = LaunchSample(2.0, True, True, 0.0, True, True, 5.0, 1.0, 1.0, 2, True, False, True, 0.1, True, 1.0)
    detected = LongEvent(
      "late_lead_launch_planner", "reason", 3, 0.55,
      lead_to_ego_s=1.5, command_to_ego_s=1.0, plan_to_lead_s=0.5,
      command_to_lead_s=0.7, forecast_to_lead_s=-0.2, radar_discontinuity=True,
      onsets=(OnsetSnapshot("lead", 1.0, 5.0, 0.5, 0.0, 0.0, 0.1, False, False),),
      episode_start_mono_time=0.5,
      attribution_detail="planner",
    )
    event = round_trip(build_message(AcceptedEvent("long", "group", longitudinal_candidate(sample, detected)))).drivingEvent
    assert event.payload.which() == "leadLaunch"
    assert event.payload.leadLaunch.leadToEgoS == 1.5
    assert event.payload.leadLaunch.radarDiscontinuity
    assert event.attribution == "planner"
    assert event.payload.leadLaunch.onsets[0].kind == "lead"
    assert event.payload.leadLaunch.onsets[0].monoTime == 1_000_000_000

  def test_acknowledgment_round_trip_preserves_identity_and_status(self):
    msg = messaging.new_message("drivingEventRecorded", valid=True)
    ack = msg.drivingEventRecorded
    ack.eventId = "event"
    ack.groupId = "group"
    ack.domain = "lateral"
    ack.source = "automatic"
    ack.eventType = "lat.centerOvershoot"
    ack.occurredMonoTime = 123
    ack.route = "route"
    ack.segment = 4
    ack.markerWritten = True
    ack.currentSegmentPreserved = True
    ack.followingSegmentScheduled = True
    ack.segmentStartMonoTime = 100
    ack.ackMonoTime = 150
    ack.markerAccepted = True
    result = round_trip(msg).drivingEventRecorded
    assert result.eventId == "event"
    assert result.groupId == "group"
    assert result.route == "route"
    assert result.segment == 4
    assert result.markerWritten and result.currentSegmentPreserved and result.followingSegmentScheduled
    assert result.markerAccepted
    assert result.segmentStartMonoTime == 100
    assert result.ackMonoTime == 150

  def test_legacy_bookmark_and_lateral_event_remain_readable(self):
    bookmark = round_trip(messaging.new_message("userBookmark", valid=True))
    assert bookmark.which() == "userBookmark"
    assert bookmark.userBookmark.eventType == ""

    lateral = messaging.new_message("lateralEvent", valid=True)
    lateral.lateralEvent.type = "lateUnwind"
    decoded = round_trip(lateral)
    assert decoded.which() == "lateralEvent"
    assert decoded.lateralEvent.type == "lateUnwind"
