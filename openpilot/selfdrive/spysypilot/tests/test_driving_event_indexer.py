import json
import os
import tempfile
from pathlib import Path

from openpilot.cereal import messaging
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.spysypilot.driving_event_indexer import (
  MANIFEST_NAME,
  PRESERVE_ATTR_NAME,
  PRESERVE_ATTR_VALUE,
  REVIEWS_NAME,
  ROTATED_MANIFEST_NAME,
  append_events,
  event_to_record,
  load_group_primaries,
  rebuild,
  scan_once,
)
from openpilot.selfdrive.spysypilot.driving_eventd import AcceptedEvent, build_message, manual_candidate


def _unittest_test(test):
  def wrapped(_self):
    test()

  return wrapped


def _unittest_tmp_path_test(test):
  def wrapped(_self):
    with tempfile.TemporaryDirectory() as directory:
      test(Path(directory))

  return wrapped


def make_segments(log_root: Path, route: str, count: int):
  for segment in range(count):
    path = log_root / f"{route}--{segment}"
    path.mkdir(parents=True)
    (path / "rlog.zst").touch()


def event_message(event_id="event", group_id="group", occurred=102_500_000_000,
                  context_before=2, context_after=1):
  msg = build_message(AcceptedEvent(event_id, group_id, manual_candidate(occurred)))
  msg.logMonoTime = occurred + 10
  msg.drivingEvent.requestedContextBefore = context_before
  msg.drivingEvent.requestedContextAfter = context_after
  return msg.as_reader()


def stop_jolt_message(event_id="stop-event", group_id="stop-group", occurred=102_250_000_000):
  msg = build_message(AcceptedEvent(event_id, group_id, manual_candidate(occurred)))
  msg.logMonoTime = occurred + 500_000_000
  event = msg.drivingEvent
  event.domain = "longitudinal"
  event.source = "automatic"
  event.eventType = "long.stopJolt"
  event.detector = "smoothStopJoltDetector"
  event.detectorVersion = 1
  event.severity = "warning"
  event.confidence = 0.9
  event.attribution = "mixed"
  event.detectedMonoTime = occurred + 500_000_000
  event.episodeStartMonoTime = occurred - 1_000_000_000
  event.episodeKey = f"stop:{occurred - 1_000_000_000}"
  event.analysisWindowBeforeS = 5.0
  event.analysisWindowAfterS = 2.0
  payload = event.payload.init("stopJolt")
  payload.episodeStartMonoTime = event.episodeStartMonoTime
  payload.standstillMonoTime = occurred + 200_000_000
  payload.peakJoltMonoTime = occurred
  payload.detectionMonoTime = event.detectedMonoTime
  payload.imuJerk = -3.5
  payload.absImuJerk = 3.5
  payload.aEgoJerk = -3.0
  payload.absAEgoJerk = 3.0
  payload.accelChange = -0.8
  payload.vEgoAtPeak = 0.15
  payload.shouldStopBefore = True
  payload.shouldStopAtPeak = True
  payload.shouldStopAfter = True
  payload.longControlStateBefore = "stopping"
  payload.longControlStateAtPeak = "stopping"
  payload.longControlStateAfter = "pid"
  payload.imuValid = True
  payload.classification = "brakeGrab"
  return msg.as_reader()


def ack_message(event_id="event", group_id="group", occurred=102_500_000_000,
                current_preserved=True, log_offset=20, segment_start=100_000_000_000):
  msg = messaging.new_message("drivingEventRecorded", valid=True)
  msg.logMonoTime = occurred + log_offset
  ack = msg.drivingEventRecorded
  ack.eventId = event_id
  ack.groupId = group_id
  ack.domain = "manual"
  ack.source = "user"
  ack.eventType = "manual.general"
  ack.occurredMonoTime = occurred
  ack.route = "authoritative-route"
  ack.segment = 2
  ack.markerWritten = True
  ack.currentSegmentPreserved = current_preserved
  ack.followingSegmentScheduled = True
  ack.segmentStartMonoTime = segment_start
  ack.ackMonoTime = msg.logMonoTime
  ack.markerAccepted = True
  return msg.as_reader()


def baseline_message(mono_time=100_000_000_000):
  msg = messaging.new_message("carState", valid=True)
  msg.logMonoTime = mono_time
  return msg.as_reader()


def sentinel_message(mono_time, sentinel_type):
  msg = messaging.new_message("sentinel", valid=True)
  msg.logMonoTime = mono_time
  msg.sentinel.type = sentinel_type
  return msg.as_reader()


def driving_event_message(event_id, group_id, event_type, occurred, *, version=3,
                          domain="lateral", detector_version=6, driver_assisted_unwind=False,
                          driver_causation="", log_offset=10):
  """Hand-built drivingEvent message for primary-designation tests (A4)."""
  msg = messaging.new_message("drivingEvent", valid=True)
  msg.logMonoTime = occurred + log_offset
  event = msg.drivingEvent
  event.version = version
  event.eventId = event_id
  event.groupId = group_id
  event.occurredMonoTime = occurred
  event.detectedMonoTime = occurred
  event.domain = domain
  event.source = "user" if domain == "manual" else "automatic"
  event.eventType = event_type
  event.detector = "test"
  event.detectorVersion = detector_version
  event.severity = "warning"
  event.confidence = 0.9
  event.attribution = "controller"
  if event_type == "lat.unwindProgressDeficit":
    payload = event.payload.init("lateral")
    payload.driverAssistedUnwind = driver_assisted_unwind
    payload.driverCausation = driver_causation
  else:
    event.payload.none = None
  return msg.as_reader()


def _test_indexes_exact_segment_offset_ack_and_context_once(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 4)
  os.setxattr(log_root / f"{route}--2", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)

  def reader(path):
    segment = int(Path(path).parent.name.rsplit("--", 1)[1])
    if segment == 2:
      return [baseline_message(), event_message(), ack_message()]
    return [baseline_message()]

  assert scan_once(log_root, event_root, reader) == 1
  assert scan_once(log_root, event_root, reader) == 0
  record = json.loads((event_root / MANIFEST_NAME).read_text())
  assert record["event_id"] == "event"
  assert record["group_id"] == "group"
  assert record["route"] == route
  assert record["segment"] == 2
  assert record["segment_offset_s"] == 2.5
  assert record["route_offset_s"] == 2.5
  assert record["marker_offset_s"] == 2.5
  assert record["verified_in_completed_rlog"]
  assert record["logger_acknowledgment"]["marker_accepted"]
  assert record["logger_acknowledgment"]["ack_mono_time"] == 102_500_000_020
  assert record["detector_to_marker_ms"] == 0.0
  assert record["marker_to_ack_ms"] == 0.0
  assert "review" not in record
  assert record["logger_acknowledgment"]["marker_written"]
  assert record["context_segments"] == [f"{route}--{index}" for index in range(4)]
  assert os.getxattr(log_root / f"{route}--2", PRESERVE_ATTR_NAME) == PRESERVE_ATTR_VALUE
  assert os.getxattr(log_root / f"{route}--3", PRESERVE_ATTR_NAME) == PRESERVE_ATTR_VALUE
  assert (event_root / REVIEWS_NAME).exists()


def _test_segment_offset_uses_authoritative_ack_start_not_cached_init_data(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 3)
  os.setxattr(log_root / f"{route}--2", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)

  def reader(path):
    segment = int(Path(path).parent.name.rsplit("--", 1)[1])
    if segment == 2:
      return [
        baseline_message(10_000_000_000),
        sentinel_message(100_000_000_000, "startOfSegment"),
        event_message(occurred=102_500_000_000),
        ack_message(occurred=102_500_000_000, segment_start=101_000_000_000),
      ]
    return [baseline_message()]

  assert scan_once(log_root, event_root, reader) == 1
  record = json.loads((event_root / MANIFEST_NAME).read_text())
  assert record["route_offset_s"] == 92.5
  assert record["segment_offset_s"] == 1.5


def _test_stop_jolt_payload_serializes_to_manifest(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 3)
  os.setxattr(log_root / f"{route}--2", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)

  def reader(path):
    segment = int(Path(path).parent.name.rsplit("--", 1)[1])
    if segment == 2:
      return [baseline_message(), stop_jolt_message()]
    return [baseline_message()]

  assert scan_once(log_root, event_root, reader) == 1
  record = json.loads((event_root / MANIFEST_NAME).read_text())
  assert record["event_type"] == "long.stopJolt"
  assert record["payload_type"] == "stopJolt"
  assert record["detected_mono_time"] > record["occurred_mono_time"]
  assert record["analysis_window"] == {"before_s": 5.0, "after_s": 2.0}
  assert record["payload"]["classification"] == "brakeGrab"
  assert record["payload"]["imu_jerk"] == -3.5
  assert record["payload"]["a_ego_jerk"] == -3.0
  assert record["payload"]["peak_jolt_mono_time"] == record["occurred_mono_time"]


def _test_rolling_lead_payload_serializes_commit_and_response_evidence():
  occurred = 100_000_000_000
  msg = messaging.new_message("drivingEvent", valid=True)
  event = msg.drivingEvent
  event.version = 2
  event.eventId = "rolling"
  event.groupId = "group"
  event.occurredMonoTime = occurred
  event.detectedMonoTime = occurred + 2_500_000_000
  event.episodeStartMonoTime = occurred
  event.domain = "longitudinal"
  event.source = "automatic"
  event.eventType = "long.lateRollingLeadResponseVehicle"
  event.detector = "rollingLeadResponseDetector"
  event.detectorVersion = 1
  event.severity = "warning"
  event.confidence = 0.95
  event.attribution = "vehicle"
  event.analysisWindowBeforeS = 5.0
  event.analysisWindowAfterS = 8.0
  payload = event.payload.init("rollingLeadResponse")
  payload.attributionDetail = "vehicle"
  payload.leadCommitMonoTime = occurred
  payload.controllerResponseMonoTime = occurred + 400_000_000
  payload.egoResponseMonoTime = occurred + 1_100_000_000
  payload.detectedMonoTime = event.detectedMonoTime
  payload.leadToCommandS = 0.4
  payload.leadToEgoS = 1.1
  payload.maxGapGrowth = 4.9
  payload.peakRelativeSpeed = 3.5
  payload.radarTrackId = 42
  payload.controllerResponsePresent = True
  payload.egoResponsePresent = True
  onsets = payload.init("onsets", 1)
  onsets[0].kind = "leadCommit"
  onsets[0].monoTime = occurred
  onsets[0].radarValid = True

  record = event_to_record(
    msg.as_reader().drivingEvent, None, "route", 1, occurred + 2_600_000_000,
    60_000_000_000, 40_000_000_000,
  )
  assert record["payload_type"] == "rollingLeadResponse"
  assert record["occurred_mono_time"] == record["payload"]["lead_commit_mono_time"]
  assert record["detected_mono_time"] == record["payload"]["detected_mono_time"]
  assert record["payload"]["radar_track_id"] == 42
  assert record["payload"]["max_gap_growth"] == 4.9
  assert record["payload"]["onsets"][0]["kind"] == "leadCommit"


def _test_lateral_v5_payload_serializes_unwind_and_turn_stop_turn_evidence():
  occurred = 100_000_000_000
  msg = messaging.new_message("drivingEvent", valid=True)
  event = msg.drivingEvent
  event.version = 2
  event.eventId = "v5-evidence"
  event.groupId = "group"
  event.occurredMonoTime = occurred
  event.detectedMonoTime = occurred + 1_000_000_000
  event.episodeStartMonoTime = occurred - 2_000_000_000
  event.domain = "lateral"
  event.source = "automatic"
  event.eventType = "lat.unwindProgressDeficit"
  event.detector = "blatLateralEventDetector"
  event.detectorVersion = 5
  event.severity = "warning"
  event.confidence = 0.9
  event.attribution = "controller"
  event.analysisWindowBeforeS = 6.0
  event.analysisWindowAfterS = 4.0
  payload = event.payload.init("lateral")
  payload.unwindEpisodeStartMonoTime = occurred - 2_000_000_000
  payload.unwindDeficitStartMonoTime = occurred
  payload.unwindDeficitDurationS = 1.25
  payload.initialSteeringAngleDeg = 42.0
  payload.peakSteeringAngleDeg = 45.0
  payload.triggerSteeringAngleDeg = 31.0
  payload.expectedUnwindDirection = "towardCenter"
  payload.expectedAngleProgressDeg = 18.0
  payload.actualAngleProgressDeg = 4.5
  payload.unwindProgressRatio = 0.25
  payload.referenceRateAtTrigger = -0.8
  payload.measurementRateAtTrigger = -0.1
  payload.peakReferenceMeasurementRateGap = 0.92
  payload.requestedTorqueNeutralCrossMonoTime = occurred + 100_000_000
  payload.requestedTorqueNeutralCrossPresent = True
  payload.appliedTorqueNeutralCrossMonoTime = occurred + 300_000_000
  payload.appliedTorqueNeutralCrossPresent = True
  payload.unwindCommandDelayS = 0.2
  payload.driverAssistedUnwind = True
  payload.driverAssistMonoTime = occurred + 500_000_000
  payload.driverAssistAngleDeg = 28.0
  payload.driverAssistWheelRateDeg = -22.0
  payload.driverAssistTorque = -85.0
  payload.wheelRateIncreaseAfterAssistDegS = 35.0
  payload.movementStartMonoTime = occurred - 1_000_000_000
  payload.dwellStartMonoTime = occurred - 400_000_000
  payload.releaseMonoTime = occurred + 200_000_000
  payload.dwellDurationS = 0.6
  payload.dwellSteeringAngleDeg = 12.0
  payload.preDwellPeakRateDegS = 75.0
  payload.minimumDwellRateDegS = 2.0
  payload.releasePeakRateDegS = 68.0
  payload.rateReductionRatio = 0.0267
  payload.restartClassification = "oppositeDirection"
  payload.requestTorqueDuringDwell = -0.4
  payload.appliedTorqueDuringDwell = -0.3
  payload.referenceTargetDuringDwell = -0.2
  payload.trackingActiveDuringDwell = True
  payload.measurementRate = -0.22
  payload.driverInvolvedBeforeDwell = False
  payload.driverInvolvedDuringDwell = True
  payload.driverInvolvedAfterDwell = False
  payload.turnStopActualDampingAmount = 0.029
  payload.turnStopActualDampingState = "damping"
  payload.turnStopTurnInBlocked = False
  payload.turnStopBreakawayLatch = 0.89
  payload.turnStopSustainFloorContribution = 0.80
  payload.turnStopDampingVersion = 2
  payload.turnStopDampingValid = True
  payload.unwindRequestedTorqueAtTrigger = -0.55
  payload.unwindAppliedTorqueAtTrigger = -0.42
  payload.unwindReferenceTargetTorqueAtTrigger = -0.12
  payload.unwindAppliedTargetGapAtTrigger = 0.30
  payload.unwindPeakAppliedTargetGap = 0.35

  record = event_to_record(
    msg.as_reader().drivingEvent, None, "route", 1, event.detectedMonoTime,
    60_000_000_000, 40_000_000_000,
  )
  serialized = record["payload"]
  assert serialized["unwind_episode_start_mono_time"] == occurred - 2_000_000_000
  assert serialized["unwind_deficit_start_mono_time"] == occurred
  assert serialized["unwind_deficit_duration_s"] == 1.25
  assert serialized["initial_steering_angle_deg"] == 42.0
  assert serialized["peak_steering_angle_deg"] == 45.0
  assert serialized["trigger_steering_angle_deg"] == 31.0
  assert serialized["expected_unwind_direction"] == "towardCenter"
  assert serialized["expected_angle_progress_deg"] == 18.0
  assert serialized["actual_angle_progress_deg"] == 4.5
  assert serialized["unwind_progress_ratio"] == 0.25
  assert serialized["reference_rate_at_trigger"] == -0.8
  assert serialized["measurement_rate_at_trigger"] == -0.1
  assert serialized["peak_reference_measurement_rate_gap"] == 0.92
  assert serialized["requested_torque_neutral_cross_mono_time"] == occurred + 100_000_000
  assert serialized["requested_torque_neutral_cross_present"]
  assert serialized["applied_torque_neutral_cross_mono_time"] == occurred + 300_000_000
  assert serialized["applied_torque_neutral_cross_present"]
  assert serialized["unwind_command_delay_s"] == 0.2
  assert serialized["driver_assisted_unwind"]
  assert serialized["driver_assist_mono_time"] == occurred + 500_000_000
  assert serialized["driver_assist_angle_deg"] == 28.0
  assert serialized["driver_assist_wheel_rate_deg"] == -22.0
  assert serialized["driver_assist_torque"] == -85.0
  assert serialized["wheel_rate_increase_after_assist_deg_s"] == 35.0
  assert serialized["movement_start_mono_time"] == occurred - 1_000_000_000
  assert serialized["dwell_start_mono_time"] == occurred - 400_000_000
  assert serialized["release_mono_time"] == occurred + 200_000_000
  assert serialized["dwell_duration_s"] == 0.6
  assert serialized["dwell_steering_angle_deg"] == 12.0
  assert serialized["pre_dwell_peak_rate_deg_s"] == 75.0
  assert serialized["minimum_dwell_rate_deg_s"] == 2.0
  assert serialized["release_peak_rate_deg_s"] == 68.0
  assert serialized["rate_reduction_ratio"] == 0.0267
  assert serialized["restart_classification"] == "oppositeDirection"
  assert serialized["request_torque_during_dwell"] == -0.4
  assert serialized["applied_torque_during_dwell"] == -0.3
  assert serialized["reference_target_during_dwell"] == -0.2
  assert serialized["tracking_active_during_dwell"]
  assert serialized["measurement_rate"] == -0.22
  assert not serialized["driver_involved_before_dwell"]
  assert serialized["driver_involved_during_dwell"]
  assert not serialized["driver_involved_after_dwell"]
  assert serialized["turn_stop_actual_damping_amount"] == 0.029
  assert serialized["turn_stop_actual_damping_state"] == "damping"
  assert not serialized["turn_stop_turn_in_blocked"]
  assert serialized["turn_stop_breakaway_latch"] == 0.89
  assert serialized["turn_stop_sustain_floor_contribution"] == 0.8
  assert serialized["turn_stop_damping_version"] == 2
  assert serialized["turn_stop_damping_valid"]
  assert serialized["unwind_requested_torque_at_trigger"] == -0.55
  assert serialized["unwind_applied_torque_at_trigger"] == -0.42
  assert serialized["unwind_reference_target_torque_at_trigger"] == -0.12
  assert serialized["unwind_applied_target_gap_at_trigger"] == 0.3
  assert serialized["unwind_peak_applied_target_gap"] == 0.35


def _test_old_lateral_payload_defaults_remain_indexable_and_v5_release_is_structured():
  occurred = 100_000_000_000
  for detector_version in (2, 3, 4):
    old_msg = messaging.new_message("drivingEvent", valid=True)
    old = old_msg.drivingEvent
    old.version = 2
    old.eventId = f"old-v{detector_version}"
    old.groupId = "group"
    old.occurredMonoTime = occurred
    old.domain = "lateral"
    old.source = "automatic"
    old.eventType = "lat.stallRelease"
    old.detector = "blatLateralEventDetector"
    old.detectorVersion = detector_version
    old.severity = "warning"
    old.confidence = 0.8
    old.attribution = "controller"
    old_payload = old.payload.init("lateral")
    if detector_version >= 3:
      old_payload.stallReleaseCount = 3
      old_payload.dampingApplied = -0.72

    old_record = event_to_record(
      old_msg.as_reader().drivingEvent, None, "route", 1, occurred,
      60_000_000_000, 40_000_000_000,
    )
    assert old_record["detector_version"] == detector_version
    assert old_record["payload"]["damping_applied"] == (-0.72 if detector_version >= 3 else 0.0)
    assert old_record["payload"]["driver_interaction"] == "none"
    assert old_record["payload"]["requested_torque_at_crossing"] == 0.0
    assert old_record["payload"]["unwind_episode_start_mono_time"] == 0
    assert not old_record["payload"]["requested_torque_neutral_cross_present"]
    assert old_record["payload"]["restart_classification"] == ""
    assert not old_record["payload"]["tracking_active_during_dwell"]
    assert old_record["payload"]["measurement_rate"] == 0.0
    assert not old_record["payload"]["driver_involved_during_dwell"]
    assert not old_record["payload"]["turn_stop_damping_valid"]
    assert old_record["payload"]["unwind_requested_torque_at_trigger"] == 0.0
    assert old_record["payload"]["stall_releases"] == []

  new_msg = messaging.new_message("drivingEvent", valid=True)
  new = new_msg.drivingEvent
  new.version = 2
  new.eventId = "v5"
  new.groupId = "group"
  new.occurredMonoTime = occurred
  new.detectedMonoTime = occurred + 250_000_000
  new.domain = "lateral"
  new.source = "automatic"
  new.eventType = "lat.stallRelease"
  new.detector = "blatLateralEventDetector"
  new.detectorVersion = 5
  new.severity = "warning"
  new.confidence = 0.85
  new.attribution = "controller"
  new_payload = new.payload.init("lateral")
  new_payload.driverInteraction = "possibleRawTorque"
  new_payload.roadConfoundExtent = "substantial"
  releases = new_payload.init("stallReleases", 3)
  for index, release in enumerate(releases):
    release.releaseMonoTime = occurred - (2 - index) * 500_000_000
    release.dampingVersion = 2
    release.dampingValid = True
    release.dampingState = "turnInAuthority" if index < 2 else "damping"
    release.dampingApplied = 0.0 if index < 2 else 0.02934
    release.turnInBlocked = index < 2
    release.breakawayLatch = 0.0 if index < 2 else 0.894866

  new_record = event_to_record(
    new_msg.as_reader().drivingEvent, None, "route", 1, occurred + 250_000_000,
    60_000_000_000, 40_000_000_000,
  )
  serialized = new_record["payload"]["stall_releases"]
  assert len(serialized) == 3
  assert [release["damping_applied"] for release in serialized] == [0.0, 0.0, 0.0293]
  assert [release["turn_in_blocked"] for release in serialized] == [True, True, False]
  assert new_record["payload"]["driver_interaction"] == "possibleRawTorque"
  assert new_record["payload"]["road_confound_extent"] == "substantial"


def _test_joins_latest_acknowledgment_across_segment_boundary(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 4)
  os.setxattr(log_root / f"{route}--2", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)

  def reader(path):
    segment = int(Path(path).parent.name.rsplit("--", 1)[1])
    if segment == 2:
      return [baseline_message(), event_message()]
    if segment == 3:
      return [
        baseline_message(),
        ack_message(current_preserved=False, log_offset=20),
        ack_message(current_preserved=True, log_offset=30),
      ]
    return [baseline_message()]

  assert scan_once(log_root, event_root, reader) == 1
  record = json.loads((event_root / MANIFEST_NAME).read_text())
  assert record["logger_acknowledgment"]["seen"]
  assert record["logger_acknowledgment"]["current_segment_preserved"]


def _test_requested_context_is_honored_and_clamped(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 5)
  os.setxattr(log_root / f"{route}--2", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)

  def reader(path):
    segment = int(Path(path).parent.name.rsplit("--", 1)[1])
    if segment == 2:
      return [baseline_message(), event_message(context_before=99, context_after=99)]
    return [baseline_message()]

  assert scan_once(log_root, event_root, reader) == 1
  record = json.loads((event_root / MANIFEST_NAME).read_text())
  assert record["requested_context"] == {"before": 99, "after": 99}
  assert record["effective_context"] == {"before": 2, "after": 1}
  assert record["context_segments"] == [f"{route}--{index}" for index in range(4)]


def _test_rebuild_scans_unpreserved_full_rlogs(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 1)

  def reader(_path):
    return [baseline_message(), event_message(occurred=101_000_000_000)]

  assert rebuild(log_root, event_root, reader) == 1
  rebuilt = json.loads((event_root / MANIFEST_NAME).read_text())
  assert rebuilt["event_id"] == "event"
  assert rebuilt["effective_context"] == {"before": 0, "after": 0}
  assert rebuild(log_root, event_root, reader) == 1
  indexed_ids = [
    json.loads(line)["event_id"]
    for path in (event_root / MANIFEST_NAME, event_root / ROTATED_MANIFEST_NAME)
    if path.exists()
    for line in path.read_text().splitlines()
  ]
  assert indexed_ids == ["event"]


def _test_rebuild_never_erases_reviews(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 1)
  event_root.mkdir()
  review = '{"event_id":"event","status":"accepted","notes":"keep me"}\n'
  (event_root / REVIEWS_NAME).write_text(review)

  def reader(_path):
    return [baseline_message(), event_message()]

  assert rebuild(log_root, event_root, reader) == 1
  assert (event_root / REVIEWS_NAME).read_text() == review


def _test_manifest_rotation_and_dedup_across_rotated_file(tmp_path):
  manifest = tmp_path / MANIFEST_NAME
  rotated = tmp_path / ROTATED_MANIFEST_NAME
  known = set()
  first = {"event_id": "one", "reason": "x" * 100}
  second = {"event_id": "two", "reason": "y" * 100}
  assert append_events(manifest, rotated, [first], known, max_bytes=150) == 1
  assert append_events(manifest, rotated, [second], known, max_bytes=150) == 1
  assert rotated.exists()
  loaded = {json.loads(line)["event_id"] for path in (manifest, rotated) for line in path.read_text().splitlines()}
  assert loaded == {"one", "two"}


def _test_corrupt_manifest_and_state_do_not_stop_scan(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 1)
  event_root.mkdir()
  (event_root / MANIFEST_NAME).write_text("{not-json}\n")
  (event_root / "processed_segments.json").write_text("{not-json}")
  os.setxattr(log_root / f"{route}--0", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)

  def reader(_path):
    return [baseline_message(), event_message()]

  assert scan_once(log_root, event_root, reader) == 1


def _test_designation_version_gate_legacy_always_primary_current_computes_role(tmp_path, version):
  # A4: designation applies ONLY to version >= 3 (DESIGNATION_MIN_VERSION).
  # Legacy (v5-era and earlier) records always self-designate as primary,
  # regardless of any higher-priority sibling sharing their group_id.
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 1)
  os.setxattr(log_root / f"{route}--0", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)

  def reader(_path):
    return [
      baseline_message(),
      driving_event_message("low", "group", "lat.stallRelease", 100_000_000_000, version=version),
      driving_event_message(
        "high", "group", "manual.general", 100_500_000_000, version=version, domain="manual",
      ),
    ]

  assert scan_once(log_root, event_root, reader) == 2
  records = {
    json.loads(line)["event_id"]: json.loads(line)
    for line in (event_root / MANIFEST_NAME).read_text().splitlines()
  }
  if version < 3:
    assert records["low"]["group_role"] == "primary"
    assert records["low"]["primary_event_id"] == "low"
    assert records["high"]["group_role"] == "primary"
    assert records["high"]["primary_event_id"] == "high"
  else:
    assert records["high"]["group_role"] == "primary"
    assert records["high"]["primary_event_id"] == "high"
    assert records["low"]["group_role"] == "secondary"
    assert records["low"]["primary_event_id"] == "high"


def _test_designation_priority_ordering_manual_outranks_everything(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 1)
  os.setxattr(log_root / f"{route}--0", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)
  base = 100_000_000_000

  def reader(_path):
    return [
      baseline_message(),
      driving_event_message("stall", "group", "lat.stallRelease", base + 1),
      driving_event_message("long", "group", "long.stopJolt", base + 2, domain="longitudinal"),
      driving_event_message("turnstop", "group", "lat.turnStopTurn", base + 3),
      driving_event_message("handoff", "group", "lat.handoffMismatch", base + 4),
      driving_event_message(
        "autounwind", "group", "lat.unwindProgressDeficit", base + 5, driver_assisted_unwind=False,
      ),
      driving_event_message(
        "assistedunwind", "group", "lat.unwindProgressDeficit", base + 6, driver_assisted_unwind=True,
      ),
      driving_event_message("manual", "group", "manual.general", base + 7, domain="manual"),
    ]

  assert scan_once(log_root, event_root, reader) == 7
  records = {
    json.loads(line)["event_id"]: json.loads(line)
    for line in (event_root / MANIFEST_NAME).read_text().splitlines()
  }
  assert records["manual"]["group_role"] == "primary"
  assert records["manual"]["primary_event_id"] == "manual"
  for event_id in ("stall", "long", "turnstop", "handoff", "autounwind", "assistedunwind"):
    assert records[event_id]["group_role"] == "secondary"
    assert records[event_id]["primary_event_id"] == "manual"


def _test_designation_priority_ordering_among_automatic_tiers(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 1)
  os.setxattr(log_root / f"{route}--0", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)
  base = 100_000_000_000

  def reader(_path):
    return [
      baseline_message(),
      driving_event_message("stall", "group", "lat.stallRelease", base + 1),
      driving_event_message("long", "group", "long.stopJolt", base + 2, domain="longitudinal"),
      driving_event_message("turnstop", "group", "lat.turnStopTurn", base + 3),
      driving_event_message("handoff", "group", "lat.handoffMismatch", base + 4),
      driving_event_message(
        "autounwind", "group", "lat.unwindProgressDeficit", base + 5, driver_assisted_unwind=False,
      ),
      driving_event_message(
        "assistedunwind", "group", "lat.unwindProgressDeficit", base + 6, driver_assisted_unwind=True,
      ),
    ]

  assert scan_once(log_root, event_root, reader) == 6
  records = {
    json.loads(line)["event_id"]: json.loads(line)
    for line in (event_root / MANIFEST_NAME).read_text().splitlines()
  }
  assert records["assistedunwind"]["group_role"] == "primary"
  for event_id in ("stall", "long", "turnstop", "handoff", "autounwind"):
    assert records[event_id]["group_role"] == "secondary"
    assert records[event_id]["primary_event_id"] == "assistedunwind"
  # driver_causation == "interventionBacked" is equivalent to the
  # driver_assisted_unwind flag for this priority tier.
  assert records["autounwind"]["group_role"] == "secondary"


def _test_designation_driver_causation_intervention_backed_equivalent_to_flag(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 1)
  os.setxattr(log_root / f"{route}--0", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)
  base = 100_000_000_000

  def reader(_path):
    return [
      baseline_message(),
      driving_event_message("handoff", "group", "lat.handoffMismatch", base + 1),
      driving_event_message(
        "rescued", "group", "lat.unwindProgressDeficit", base + 2,
        driver_assisted_unwind=False, driver_causation="interventionBacked",
      ),
    ]

  assert scan_once(log_root, event_root, reader) == 2
  records = {
    json.loads(line)["event_id"]: json.loads(line)
    for line in (event_root / MANIFEST_NAME).read_text().splitlines()
  }
  assert records["rescued"]["group_role"] == "primary"
  assert records["handoff"]["primary_event_id"] == "rescued"


def _test_designation_tie_break_by_earliest_occurred_time(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 1)
  os.setxattr(log_root / f"{route}--0", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)
  base = 100_000_000_000

  def reader(_path):
    return [
      baseline_message(),
      # Same priority tier (both 80); "zzz-early" occurs first and must win
      # despite a lexicographically later event_id.
      driving_event_message("zzz-early", "group", "lat.handoffMismatch", base),
      driving_event_message("aaa-late", "group", "lat.centerOvershoot", base + 1_000_000_000),
    ]

  assert scan_once(log_root, event_root, reader) == 2
  records = {
    json.loads(line)["event_id"]: json.loads(line)
    for line in (event_root / MANIFEST_NAME).read_text().splitlines()
  }
  assert records["zzz-early"]["group_role"] == "primary"
  assert records["aaa-late"]["group_role"] == "secondary"
  assert records["aaa-late"]["primary_event_id"] == "zzz-early"


def _test_designation_tie_break_lexicographic_event_id_when_occurred_matches(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 1)
  os.setxattr(log_root / f"{route}--0", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)
  occurred = 100_000_000_000

  def reader(_path):
    return [
      baseline_message(),
      driving_event_message("bbb", "group", "lat.handoffMismatch", occurred, log_offset=10),
      driving_event_message("aaa", "group", "lat.centerOvershoot", occurred, log_offset=20),
    ]

  assert scan_once(log_root, event_root, reader) == 2
  records = {
    json.loads(line)["event_id"]: json.loads(line)
    for line in (event_root / MANIFEST_NAME).read_text().splitlines()
  }
  assert records["aaa"]["group_role"] == "primary"
  assert records["bbb"]["group_role"] == "secondary"
  assert records["bbb"]["primary_event_id"] == "aaa"


def _test_designation_segment_boundary_straddling_matches_rebuild(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 2)
  os.setxattr(log_root / f"{route}--0", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)

  group_id = "group"
  low_priority_event = driving_event_message(
    "turnstop", group_id, "lat.turnStopTurn", 100_000_000_000,
  )
  high_priority_event = driving_event_message(
    "assisted", group_id, "lat.unwindProgressDeficit", 101_000_000_000, driver_assisted_unwind=True,
  )

  def reader(path):
    segment = int(Path(path).parent.name.rsplit("--", 1)[1])
    if segment == 0:
      return [baseline_message(), low_priority_event]
    if segment == 1:
      return [baseline_message(), high_priority_event]
    raise AssertionError(f"unexpected segment {segment}")

  # Round 1: only segment 0 is preserved/a route_candidate. Segment 1 is
  # complete with an rlog, so it is visible as a LOOKAHEAD for priority
  # purposes (A4) without being written this round.
  assert scan_once(log_root, event_root, reader) == 1
  round_one_lines = (event_root / MANIFEST_NAME).read_text().splitlines()
  assert len(round_one_lines) == 1
  round_one_record = json.loads(round_one_lines[0])
  assert round_one_record["event_id"] == "turnstop"
  assert round_one_record["group_role"] == "secondary"
  assert round_one_record["primary_event_id"] == "assisted"

  # Round 2: segment 1 becomes preserved and is now its own candidate. It
  # consults the known-primary map (built from segment 0's already-written
  # secondary line) and self-designates as the primary it already was.
  os.setxattr(log_root / f"{route}--1", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)
  assert scan_once(log_root, event_root, reader) == 1
  all_lines = (event_root / MANIFEST_NAME).read_text().splitlines()
  assert len(all_lines) == 2
  incremental_by_id = {json.loads(line)["event_id"]: json.loads(line) for line in all_lines}
  assert incremental_by_id["assisted"]["group_role"] == "primary"
  assert incremental_by_id["assisted"]["primary_event_id"] == "assisted"
  assert incremental_by_id["turnstop"]["group_role"] == "secondary"
  assert incremental_by_id["turnstop"]["primary_event_id"] == "assisted"

  # Rebuild (single pass, both segments together) must reach the SAME
  # designation as the incremental two-round scan above.
  rebuilt_event_root = tmp_path / "rebuilt-events"
  assert rebuild(log_root, rebuilt_event_root, reader) == 2
  rebuilt_by_id = {
    json.loads(line)["event_id"]: json.loads(line)
    for line in (rebuilt_event_root / MANIFEST_NAME).read_text().splitlines()
  }
  for event_id in ("turnstop", "assisted"):
    assert rebuilt_by_id[event_id]["group_role"] == incremental_by_id[event_id]["group_role"]
    assert rebuilt_by_id[event_id]["primary_event_id"] == incremental_by_id[event_id]["primary_event_id"]


def _test_rebuild_recomputes_designation_ignoring_stale_prior_manifest(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 1)

  group_id = "group"
  low = driving_event_message("low", group_id, "lat.stallRelease", 100_000_000_000)
  high = driving_event_message("high", group_id, "manual.general", 100_500_000_000, domain="manual")

  def reader(_path):
    return [baseline_message(), low, high]

  # Seed a stale manifest that (incorrectly, by construction) claims the
  # LOW-priority event as primary, as an old/buggy scan might have.
  event_root.mkdir()
  stale = {
    "event_id": "low", "group_id": group_id, "version": 3,
    "group_role": "primary", "primary_event_id": "low",
  }
  (event_root / MANIFEST_NAME).write_text(json.dumps(stale) + "\n")

  assert rebuild(log_root, event_root, reader) == 2
  records = {
    json.loads(line)["event_id"]: json.loads(line)
    for line in (event_root / MANIFEST_NAME).read_text().splitlines()
  }
  # Rebuild recomputes with an EMPTY known-primary map: the priority table
  # alone decides, regardless of what a stale manifest claimed.
  assert records["high"]["group_role"] == "primary"
  assert records["high"]["primary_event_id"] == "high"
  assert records["low"]["group_role"] == "secondary"
  assert records["low"]["primary_event_id"] == "high"


def _test_designation_persists_across_manifest_rotation(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 1)
  os.setxattr(log_root / f"{route}--0", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)

  group_id = "group"
  low = driving_event_message("low", group_id, "lat.stallRelease", 100_000_000_000)
  high = driving_event_message("high", group_id, "manual.general", 100_500_000_000, domain="manual")

  def reader(_path):
    return [baseline_message(), low, high]

  assert scan_once(log_root, event_root, reader) == 2
  by_id = {
    json.loads(line)["event_id"]: json.loads(line)
    for line in (event_root / MANIFEST_NAME).read_text().splitlines()
  }
  assert by_id["high"]["group_role"] == "primary"
  assert by_id["low"]["primary_event_id"] == "high"

  # Simulate the active manifest having rotated: move today's content into
  # the ROTATED file and start a fresh, empty active manifest, as
  # rotate_manifest() would once the active file grows past its size cap.
  manifest_path = event_root / MANIFEST_NAME
  rotated_path = event_root / ROTATED_MANIFEST_NAME
  manifest_path.rename(rotated_path)
  manifest_path.touch()

  primaries = load_group_primaries(rotated_path, manifest_path)
  assert primaries[group_id] == "high"

  # A later sibling of the same group, scanned after rotation, must still
  # reference the correct (now-rotated) primary.
  (log_root / f"{route}--1").mkdir(parents=True)
  (log_root / f"{route}--1" / "rlog.zst").touch()
  os.setxattr(log_root / f"{route}--1", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)
  later = driving_event_message("later", group_id, "lat.stallRelease", 101_000_000_000)

  def reader_two(path):
    segment = int(Path(path).parent.name.rsplit("--", 1)[1])
    if segment == 0:
      return [baseline_message(), low, high]
    if segment == 1:
      return [baseline_message(), later]
    raise AssertionError(f"unexpected segment {segment}")

  assert scan_once(log_root, event_root, reader_two) == 1
  later_lines = (event_root / MANIFEST_NAME).read_text().splitlines()
  assert len(later_lines) == 1
  later_record = json.loads(later_lines[0])
  assert later_record["event_id"] == "later"
  assert later_record["group_role"] == "secondary"
  assert later_record["primary_event_id"] == "high"


def _test_designation_count_message_reports_primary_subset(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 1)
  os.setxattr(log_root / f"{route}--0", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)

  def reader(_path):
    return [
      baseline_message(),
      driving_event_message("low", "group", "lat.stallRelease", 100_000_000_000),
      driving_event_message("high", "group", "manual.general", 100_500_000_000, domain="manual"),
    ]

  assert scan_once(log_root, event_root, reader) == 2
  records = {
    json.loads(line)["event_id"]: json.loads(line)
    for line in (event_root / MANIFEST_NAME).read_text().splitlines()
  }
  primary_count = sum(1 for record in records.values() if record["group_role"] != "secondary")
  assert primary_count == 1


class TestDrivingEventIndexer(OpenpilotTestCase):
  test_indexes_exact_segment_offset_ack_and_context_once = _unittest_tmp_path_test(
    _test_indexes_exact_segment_offset_ack_and_context_once)
  test_segment_offset_uses_authoritative_ack_start_not_cached_init_data = _unittest_tmp_path_test(
    _test_segment_offset_uses_authoritative_ack_start_not_cached_init_data)
  test_stop_jolt_payload_serializes_to_manifest = _unittest_tmp_path_test(_test_stop_jolt_payload_serializes_to_manifest)
  test_rolling_lead_payload_serializes_commit_and_response_evidence = _unittest_test(
    _test_rolling_lead_payload_serializes_commit_and_response_evidence)
  test_lateral_v5_payload_serializes_unwind_and_turn_stop_turn_evidence = _unittest_test(
    _test_lateral_v5_payload_serializes_unwind_and_turn_stop_turn_evidence)
  test_old_lateral_payload_defaults_remain_indexable_and_v5_release_is_structured = _unittest_test(
    _test_old_lateral_payload_defaults_remain_indexable_and_v5_release_is_structured)
  test_joins_latest_acknowledgment_across_segment_boundary = _unittest_tmp_path_test(
    _test_joins_latest_acknowledgment_across_segment_boundary)
  test_requested_context_is_honored_and_clamped = _unittest_tmp_path_test(_test_requested_context_is_honored_and_clamped)
  test_rebuild_scans_unpreserved_full_rlogs = _unittest_tmp_path_test(_test_rebuild_scans_unpreserved_full_rlogs)
  test_rebuild_never_erases_reviews = _unittest_tmp_path_test(_test_rebuild_never_erases_reviews)
  test_manifest_rotation_and_dedup_across_rotated_file = _unittest_tmp_path_test(
    _test_manifest_rotation_and_dedup_across_rotated_file)
  test_corrupt_manifest_and_state_do_not_stop_scan = _unittest_tmp_path_test(
    _test_corrupt_manifest_and_state_do_not_stop_scan)

  @parameterized.expand([1, 2, 3, 4, 5], names=("version",))
  def test_designation_version_gate_legacy_always_primary_current_computes_role(self, version):
    with tempfile.TemporaryDirectory() as directory:
      _test_designation_version_gate_legacy_always_primary_current_computes_role(Path(directory), version)

  test_designation_priority_ordering_manual_outranks_everything = _unittest_tmp_path_test(
    _test_designation_priority_ordering_manual_outranks_everything)
  test_designation_priority_ordering_among_automatic_tiers = _unittest_tmp_path_test(
    _test_designation_priority_ordering_among_automatic_tiers)
  test_designation_driver_causation_intervention_backed_equivalent_to_flag = _unittest_tmp_path_test(
    _test_designation_driver_causation_intervention_backed_equivalent_to_flag)
  test_designation_tie_break_by_earliest_occurred_time = _unittest_tmp_path_test(
    _test_designation_tie_break_by_earliest_occurred_time)
  test_designation_tie_break_lexicographic_event_id_when_occurred_matches = _unittest_tmp_path_test(
    _test_designation_tie_break_lexicographic_event_id_when_occurred_matches)
  test_designation_segment_boundary_straddling_matches_rebuild = _unittest_tmp_path_test(
    _test_designation_segment_boundary_straddling_matches_rebuild)
  test_rebuild_recomputes_designation_ignoring_stale_prior_manifest = _unittest_tmp_path_test(
    _test_rebuild_recomputes_designation_ignoring_stale_prior_manifest)
  test_designation_persists_across_manifest_rotation = _unittest_tmp_path_test(
    _test_designation_persists_across_manifest_rotation)
  test_designation_count_message_reports_primary_subset = _unittest_tmp_path_test(
    _test_designation_count_message_reports_primary_subset)
