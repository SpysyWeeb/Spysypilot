import pytest

from openpilot.cereal import messaging
from openpilot.selfdrive.spysypilot.driving_eventd import (
  AcceptedEvent,
  build_message,
  lateral_candidate,
  longitudinal_candidate,
  manual_candidate,
)
from openpilot.selfdrive.spysypilot.lat_event_detector import LateralDetection, LateralSample
from openpilot.selfdrive.spysypilot.long_event_detector import LaunchSample, LongEvent


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


def test_lateral_payload_round_trip():
  sample = LateralSample(
    1.0, v_ego=12.0, steering_angle_deg=4.0, steering_rate_deg=-20.0,
    request_torque=0.5, applied_torque=0.4, reference_target_torque=0.1,
    controller_version=3, reference_version=4, road_confounded=True,
  )
  candidate = lateral_candidate(sample, LateralDetection("centerOvershoot", "warning", 0.9, "reason"))
  event = round_trip(build_message(AcceptedEvent("lat", "group", candidate))).drivingEvent
  assert event.eventId == "lat"
  assert event.payload.which() == "lateral"
  assert event.payload.lateral.controllerVersion == 3
  assert event.payload.lateral.steeringRateDeg == -20.0
  assert event.payload.lateral.appliedTargetGap == pytest.approx(0.3)
  assert event.roadConfounded


def test_longitudinal_payload_round_trip():
  sample = LaunchSample(2.0, True, True, 0.0, True, True, 5.0, 1.0, 1.0, 2, True, False, True, 0.1, True, 1.0)
  detected = LongEvent(
    "late_lead_launch_planner", "Long Event Logged", "reason", 3, 0.55,
    lead_to_ego_s=1.5, command_to_ego_s=1.0, plan_to_lead_s=0.5,
    command_to_lead_s=0.7, forecast_to_lead_s=-0.2, radar_discontinuity=True,
  )
  event = round_trip(build_message(AcceptedEvent("long", "group", longitudinal_candidate(sample, detected)))).drivingEvent
  assert event.payload.which() == "leadLaunch"
  assert event.payload.leadLaunch.leadToEgoS == 1.5
  assert event.payload.leadLaunch.radarDiscontinuity
  assert event.attribution == "planner"


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
  result = round_trip(msg).drivingEventRecorded
  assert result.eventId == "event"
  assert result.groupId == "group"
  assert result.route == "route"
  assert result.segment == 4
  assert result.markerWritten and result.currentSegmentPreserved and result.followingSegmentScheduled


def test_legacy_bookmark_and_lateral_event_remain_readable():
  bookmark = round_trip(messaging.new_message("userBookmark", valid=True))
  assert bookmark.which() == "userBookmark"
  assert bookmark.userBookmark.eventType == ""

  lateral = messaging.new_message("lateralEvent", valid=True)
  lateral.lateralEvent.type = "lateUnwind"
  decoded = round_trip(lateral)
  assert decoded.which() == "lateralEvent"
  assert decoded.lateralEvent.type == "lateUnwind"
