import json
import os
from pathlib import Path

from openpilot.cereal import messaging
from openpilot.selfdrive.spysypilot.driving_event_indexer import (
  MANIFEST_NAME,
  PRESERVE_ATTR_NAME,
  PRESERVE_ATTR_VALUE,
  ROTATED_MANIFEST_NAME,
  append_events,
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


def ack_message(event_id="event", group_id="group", occurred=102_500_000_000,
                current_preserved=True, log_offset=20):
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
  return msg.as_reader()


def baseline_message(mono_time=100_000_000_000):
  msg = messaging.new_message("carState", valid=True)
  msg.logMonoTime = mono_time
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
  assert record["logger_acknowledgment"]["marker_written"]
  assert record["context_segments"] == [f"{route}--{index}" for index in range(4)]
  assert os.getxattr(log_root / f"{route}--2", PRESERVE_ATTR_NAME) == PRESERVE_ATTR_VALUE
  assert os.getxattr(log_root / f"{route}--3", PRESERVE_ATTR_NAME) == PRESERVE_ATTR_VALUE


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
  assert json.loads((event_root / MANIFEST_NAME).read_text())["event_id"] == "event"
  assert rebuild(log_root, event_root, reader) == 1
  indexed_ids = [
    json.loads(line)["event_id"]
    for path in (event_root / MANIFEST_NAME, event_root / ROTATED_MANIFEST_NAME)
    if path.exists()
    for line in path.read_text().splitlines()
  ]
  assert indexed_ids == ["event"]


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
