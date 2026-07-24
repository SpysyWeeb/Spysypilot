import json
import os
from pathlib import Path

from openpilot.cereal import messaging
from openpilot.selfdrive.spysypilot.driving_event_indexer import (
  MANIFEST_NAME,
  PRESERVE_ATTR_NAME,
  PRESERVE_ATTR_VALUE,
  REVIEWS_NAME,
  ROTATED_MANIFEST_NAME,
  append_events,
  event_to_record,
  rebuild,
  scan_once,
)
from openpilot.selfdrive.spysypilot.driving_eventd import AcceptedEvent, build_message, manual_candidate


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


def test_indexes_exact_segment_offset_ack_and_context_once(tmp_path):
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


def test_segment_offset_uses_authoritative_ack_start_not_cached_init_data(tmp_path):
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


def test_stop_jolt_payload_serializes_to_manifest(tmp_path):
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


def test_rolling_lead_payload_serializes_commit_and_response_evidence():
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


def test_old_lateral_payload_defaults_remain_indexable_and_v4_release_is_structured():
  occurred = 100_000_000_000
  for detector_version in (2, 3):
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
    assert old_record["payload"]["stall_releases"] == []

  new_msg = messaging.new_message("drivingEvent", valid=True)
  new = new_msg.drivingEvent
  new.version = 2
  new.eventId = "v4"
  new.groupId = "group"
  new.occurredMonoTime = occurred
  new.detectedMonoTime = occurred + 250_000_000
  new.domain = "lateral"
  new.source = "automatic"
  new.eventType = "lat.stallRelease"
  new.detector = "blatLateralEventDetector"
  new.detectorVersion = 4
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


def test_joins_latest_acknowledgment_across_segment_boundary(tmp_path):
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


def test_requested_context_is_honored_and_clamped(tmp_path):
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


def test_rebuild_scans_unpreserved_full_rlogs(tmp_path):
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


def test_rebuild_never_erases_reviews(tmp_path):
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


def test_manifest_rotation_and_dedup_across_rotated_file(tmp_path):
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


def test_corrupt_manifest_and_state_do_not_stop_scan(tmp_path):
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
