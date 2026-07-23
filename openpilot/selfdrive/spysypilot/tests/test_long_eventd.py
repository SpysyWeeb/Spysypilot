import json
import os
from pathlib import Path

from openpilot.selfdrive.spysypilot.long_eventd import (MANIFEST_NAME, PRESERVE_ATTR_NAME,
                                                        PRESERVE_ATTR_VALUE, scan_once)


class Obj:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)


class Msg:
  def __init__(self, mono_time: int, which: str = "carState", bookmark=None):
    self.logMonoTime = mono_time
    self._which = which
    self.userBookmark = bookmark

  def which(self):
    return self._which


def bookmark(event_type="late_lead_launch_vehicle", source="automatic"):
  return Obj(
    source=source,
    eventType=event_type,
    alertText1="Long Event Logged",
    alertText2="Late launch +1.6 s - vehicle response",
    severity=3,
    confidence=0.95,
    leadToEgoS=1.6,
    commandToEgoS=2.1,
    planToLeadS=-0.7,
    commandToLeadS=-0.6,
    forecastToLeadS=-1.2,
  )


def make_segments(log_root: Path, route: str, count: int):
  for segment in range(count):
    path = log_root / f"{route}--{segment}"
    path.mkdir(parents=True)
    (path / "rlog.zst").touch()


def test_indexes_event_once_and_preserves_context(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 4)
  os.setxattr(log_root / f"{route}--2", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)

  def reader(path):
    segment = int(Path(path).parent.name.rsplit("--", 1)[1])
    messages = [Msg(100_000_000_000)]
    if segment == 2:
      messages.append(Msg(102_500_000_000, "userBookmark", bookmark()))
    return messages

  assert scan_once(log_root, event_root, reader) == 1
  assert scan_once(log_root, event_root, reader) == 0

  lines = (event_root / MANIFEST_NAME).read_text().splitlines()
  assert len(lines) == 1
  event = json.loads(lines[0])
  assert event["route"] == route
  assert event["segment"] == 2
  assert event["segment_offset_s"] == 2.5
  assert event["event_type"] == "late_lead_launch_vehicle"
  assert event["metrics"]["lead_to_ego_s"] == 1.6
  assert event["metrics"]["forecast_to_lead_s"] == -1.2
  assert event["context_segments"] == [f"{route}--{i}" for i in range(4)]

  for index in range(4):
    path = log_root / f"{route}--{index}"
    assert os.getxattr(path, PRESERVE_ATTR_NAME) == PRESERVE_ATTR_VALUE


def test_ignores_generic_bookmarks(tmp_path):
  log_root = tmp_path / "realdata"
  event_root = tmp_path / "events"
  route = "abc|2026-01-01--00-00-00"
  make_segments(log_root, route, 1)
  os.setxattr(log_root / f"{route}--0", PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)

  def reader(_):
    return [Msg(100), Msg(200, "userBookmark", bookmark(event_type="", source="generic"))]

  assert scan_once(log_root, event_root, reader) == 0
  assert not (event_root / MANIFEST_NAME).exists()
