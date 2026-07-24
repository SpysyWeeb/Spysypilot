import pytest

from openpilot.cereal import messaging
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


def round_trip(msg):
  return messaging.log_from_bytes(msg.to_bytes())


def test_manual_event_round_trip():
  msg = build_message(AcceptedEvent("event", "group", manual_candidate(123)), "commit", "branch")
  event = round_trip(msg).drivingEvent
  assert event.eventId == "event"
  assert event.groupId == "group"
  assert event.domain == "manual"
  assert event.payload.which() == "none"
  assert event.gitCommit == "commit"
  assert event.detectedMonoTime == 123
  assert event.analysisWindowBeforeS == 15.0


def test_lateral_payload_round_trip():
  sample = LateralSample(
    1.0, v_ego=12.0, steering_angle_deg=4.0, steering_rate_deg=-20.0,
    request_torque=0.5, applied_torque=0.4, reference_target_torque=0.1,
    controller_version=3, reference_version=4, road_confounded=True,
    driver_torque=75.0, steering_torque_eps=123.0, damping_applied=0.2, damping_state="applied",
  )
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
  candidate = lateral_candidate(sample, LateralDetection(
    "committedHandoffHarshness", "warning", 0.9, "reason", evidence,
    occurred_mono_time=0.75, detected_mono_time=1.0,
  ))
  event = round_trip(build_message(AcceptedEvent("lat", "group", candidate))).drivingEvent
  assert event.eventId == "lat"
  assert event.payload.which() == "lateral"
  assert event.payload.lateral.controllerVersion == 3
  assert event.payload.lateral.steeringRateDeg == -20.0
  assert event.payload.lateral.appliedTargetGap == pytest.approx(0.3)
  assert event.roadConfounded
  assert event.driverConfounded
  assert event.payload.lateral.driverTorque == 75.0
  assert list(event.payload.lateral.stallDurationsS) == pytest.approx([0.2, 0.3, 0.4])
  assert event.occurredMonoTime == 750_000_000
  assert event.detectedMonoTime == 1_000_000_000
  assert event.payload.lateral.driverInteraction == "possibleRawTorque"
  assert event.payload.lateral.roadConfoundExtent == "substantial"
  assert event.payload.lateral.centerCrossingMonoTime == event.occurredMonoTime
  assert event.payload.lateral.handoffConsolidated
  assert event.payload.lateral.peakAbsCenterWheelRateDeg == pytest.approx(315.8)
  assert event.payload.lateral.requestedTorqueAtCrossing == pytest.approx(0.55)
  assert event.payload.lateral.appliedTorqueAtCrossing == pytest.approx(0.42)
  assert event.payload.lateral.referenceTargetTorqueAtCrossing == pytest.approx(0.12)
  release = event.payload.lateral.stallReleases[0]
  assert release.dampingValid
  assert release.dampingState == "turnInAuthority"
  assert release.dampingApplied == 0.0
  assert release.turnInBlocked
  assert event.episodeKey == "lat:episode"


def test_longitudinal_payload_round_trip():
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


def test_acknowledgment_round_trip_preserves_identity_and_status():
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


def test_legacy_bookmark_and_lateral_event_remain_readable():
  bookmark = round_trip(messaging.new_message("userBookmark", valid=True))
  assert bookmark.which() == "userBookmark"
  assert bookmark.userBookmark.eventType == ""

  lateral = messaging.new_message("lateralEvent", valid=True)
  lateral.lateralEvent.type = "lateUnwind"
  decoded = round_trip(lateral)
  assert decoded.which() == "lateralEvent"
  assert decoded.lateralEvent.type == "lateUnwind"
