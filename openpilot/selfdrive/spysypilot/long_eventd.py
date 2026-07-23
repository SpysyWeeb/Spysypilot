#!/usr/bin/env python3
"""Index structured longitudinal bookmarks from completed local rlogs.

The on-road path publishes userBookmark immediately for UI confirmation and loggerd
preservation. This off-road service turns those logged messages into a durable JSONL
manifest for SSH retrieval and protects adjacent local segments for context.
"""
import json
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog


POLL_INTERVAL = 30.0
EVENT_ROOT = Path("/data/community/long_events")
MANIFEST_NAME = "manifest.jsonl"
STATE_NAME = "processed_segments.json"
PRESERVE_ATTR_NAME = "user.preserve"
PRESERVE_ATTR_VALUE = b"1"
CONTEXT_BEFORE = 2
CONTEXT_AFTER = 1


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


def parse_segment_name(name: str) -> tuple[str, int] | None:
  route, separator, segment = name.rpartition("--")
  if not separator or not route:
    return None
  try:
    return route, int(segment)
  except ValueError:
    return None


def bookmark_to_event(bookmark: Any, route: str, segment: int, log_mono_time: int,
                      segment_start_mono_time: int) -> dict[str, Any] | None:
  event_type = str(bookmark.eventType)
  if not event_type:
    return None

  return {
    "id": f"{route}--{segment}:{log_mono_time}",
    "route": route,
    "segment": segment,
    "segment_offset_s": round((log_mono_time - segment_start_mono_time) / 1e9, 3),
    "source": str(bookmark.source),
    "event_type": event_type,
    "title": str(bookmark.alertText1),
    "detail": str(bookmark.alertText2),
    "severity": int(bookmark.severity),
    "confidence": round(float(bookmark.confidence), 3),
    "metrics": {
      "lead_to_ego_s": round(float(bookmark.leadToEgoS), 3),
      "command_to_ego_s": round(float(bookmark.commandToEgoS), 3),
      "plan_to_lead_s": round(float(bookmark.planToLeadS), 3),
      "command_to_lead_s": round(float(bookmark.commandToLeadS), 3),
      "forecast_to_lead_s": round(float(bookmark.forecastToLeadS), 3),
    },
    "review": "unreviewed",
  }


def scan_segment(segment_path: Path, reader_factory: Callable[[str], Iterable[Any]]) -> list[dict[str, Any]]:
  parsed = parse_segment_name(segment_path.name)
  rlog = find_rlog(segment_path)
  if parsed is None or rlog is None:
    return []

  route, segment = parsed
  events: list[dict[str, Any]] = []
  segment_start_mono_time: int | None = None
  for msg in reader_factory(str(rlog)):
    log_mono_time = int(msg.logMonoTime)
    if segment_start_mono_time is None:
      segment_start_mono_time = log_mono_time
    if msg.which() != "userBookmark":
      continue
    event = bookmark_to_event(msg.userBookmark, route, segment, log_mono_time, segment_start_mono_time)
    if event is not None:
      events.append(event)
  return events


def preserve_context(log_root: Path, route: str, segment: int) -> list[str]:
  preserved: list[str] = []
  start = max(0, segment - CONTEXT_BEFORE)
  for index in range(start, segment + CONTEXT_AFTER + 1):
    name = f"{route}--{index}"
    path = log_root / name
    if not path.is_dir():
      continue
    try:
      os.setxattr(path, PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE)
      preserved.append(name)
    except OSError:
      cloudlog.exception(f"long_eventd: failed to preserve {path}")
  return preserved


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
  os.replace(temporary, path)


def load_event_ids(manifest_path: Path) -> set[str]:
  event_ids: set[str] = set()
  try:
    with manifest_path.open(encoding="utf-8") as file:
      for line in file:
        try:
          event = json.loads(line)
        except ValueError:
          continue
        if isinstance(event, dict) and isinstance(event.get("id"), str):
          event_ids.add(event["id"])
  except OSError:
    pass
  return event_ids


def append_events(manifest_path: Path, events: list[dict[str, Any]], known_ids: set[str]) -> int:
  new_events = [event for event in events if event["id"] not in known_ids]
  if not new_events:
    return 0
  with manifest_path.open("a", encoding="utf-8") as file:
    for event in new_events:
      file.write(json.dumps(event, sort_keys=True) + "\n")
      known_ids.add(event["id"])
    file.flush()
    os.fsync(file.fileno())
  return len(new_events)


def scan_once(log_root: Path, event_root: Path = EVENT_ROOT,
              reader_factory: Callable[[str], Iterable[Any]] | None = None) -> int:
  if reader_factory is None:
    from openpilot.tools.lib.logreader import LogReader

    def default_reader(path: str):
      return LogReader(path, sort_by_time=True)

    reader_factory = default_reader

  event_root.mkdir(parents=True, exist_ok=True)
  manifest_path = event_root / MANIFEST_NAME
  state_path = event_root / STATE_NAME
  processed = set(load_json(state_path, []))
  known_ids = load_event_ids(manifest_path)
  present_segments: set[str] = set()
  added = 0

  try:
    segment_paths = sorted(path for path in log_root.iterdir() if path.is_dir() and parse_segment_name(path.name) is not None)
  except OSError:
    return 0

  for segment_path in segment_paths:
    present_segments.add(segment_path.name)
    if segment_path.name in processed:
      continue
    try:
      names = os.listdir(segment_path)
    except OSError:
      continue
    # Every structured event publishes userBookmark, so loggerd marks its live
    # segment. Restrict rlog decompression to preserved candidates instead of
    # rescanning every ordinary drive segment on the device.
    if (not is_preserved(segment_path) or any(name.endswith(".lock") for name in names)
        or find_rlog(segment_path) is None):
      continue

    try:
      events = scan_segment(segment_path, reader_factory)
    except Exception:
      cloudlog.exception(f"long_eventd: failed to parse {segment_path}")
      continue

    for event in events:
      event["context_segments"] = preserve_context(log_root, event["route"], event["segment"])
    added += append_events(manifest_path, events, known_ids)
    processed.add(segment_path.name)

  # Deleter may remove old segments. Keep the state bounded so a reused test/device
  # directory name is never silently treated as already analyzed forever.
  processed &= present_segments
  atomic_write_json(state_path, sorted(processed))
  return added


def main() -> None:
  log_root = Path(Paths.log_root())
  while True:
    try:
      added = scan_once(log_root)
      if added:
        cloudlog.info(f"long_eventd: indexed {added} new longitudinal event(s)")
    except Exception:
      cloudlog.exception("long_eventd: scan failed")
    time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
  main()
