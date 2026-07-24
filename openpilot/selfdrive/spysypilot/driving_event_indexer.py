#!/usr/bin/env python3
"""Bounded, reconstructable SSH index for rlog-authoritative driving events."""
import argparse
import json
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog


POLL_INTERVAL = 30.0
EVENT_ROOT = Path("/data/community/driving_events")
MANIFEST_NAME = "manifest.jsonl"
ROTATED_MANIFEST_NAME = "manifest.jsonl.1"
STATE_NAME = "processed_segments.json"
MANIFEST_MAX_BYTES = 2 * 1024 * 1024
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
    }
  return {}


def event_to_record(event: Any, acknowledgment: Any | None, route: str, segment: int,
                    marker_mono_time: int, segment_start_mono_time: int) -> dict[str, Any]:
  occurred_mono_time = int(event.occurredMonoTime)
  ack = {
    "seen": acknowledgment is not None,
    "marker_written": bool(acknowledgment.markerWritten) if acknowledgment is not None else False,
    "current_segment_preserved": bool(acknowledgment.currentSegmentPreserved) if acknowledgment is not None else False,
    "following_segment_scheduled": bool(acknowledgment.followingSegmentScheduled) if acknowledgment is not None else False,
    "error": str(acknowledgment.error) if acknowledgment is not None else "",
  }
  return {
    "version": int(event.version),
    "event_id": str(event.eventId),
    "group_id": str(event.groupId),
    "route": route,
    "segment": segment,
    "occurred_mono_time": occurred_mono_time,
    "marker_mono_time": marker_mono_time,
    "segment_offset_s": round((occurred_mono_time - segment_start_mono_time) / 1e9, 6),
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
    "payload_type": event.payload.which(),
    "payload": _payload(event),
    "logger_acknowledgment": ack,
    "review": "unreviewed",
  }


def scan_segment(segment_path: Path, reader_factory: Callable[[str], Iterable[Any]]) -> list[dict[str, Any]]:
  parsed = parse_segment_name(segment_path.name)
  rlog = find_rlog(segment_path)
  if parsed is None or rlog is None:
    return []

  route, segment = parsed
  raw_events: list[tuple[int, Any]] = []
  acknowledgments: dict[str, Any] = {}
  segment_start_mono_time: int | None = None
  for msg in reader_factory(str(rlog)):
    log_mono_time = int(msg.logMonoTime)
    if segment_start_mono_time is None:
      segment_start_mono_time = log_mono_time
    try:
      which = msg.which()
      if which == "drivingEvent":
        raw_events.append((log_mono_time, msg.drivingEvent))
      elif which == "drivingEventRecorded":
        acknowledgments[str(msg.drivingEventRecorded.eventId)] = msg.drivingEventRecorded
    except Exception:
      cloudlog.exception(f"driving_event_indexer: malformed record in {rlog}")
      continue

  if segment_start_mono_time is None:
    return []
  return [
    event_to_record(event, acknowledgments.get(str(event.eventId)), route, segment, marker_time, segment_start_mono_time)
    for marker_time, event in raw_events
    if str(event.eventId)
  ]


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
  new_events = [event for event in events if event["event_id"] not in known_ids]
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
    return sorted(path for path in log_root.iterdir() if path.is_dir() and parse_segment_name(path.name) is not None)
  except OSError:
    return []


def scan_once(log_root: Path, event_root: Path = EVENT_ROOT,
              reader_factory: Callable[[str], Iterable[Any]] | None = None,
              scan_all: bool = False) -> int:
  if reader_factory is None:
    from openpilot.tools.lib.logreader import LogReader

    def default_reader(path: str):
      return LogReader(path, sort_by_time=True)

    reader_factory = default_reader

  event_root.mkdir(parents=True, exist_ok=True)
  manifest_path = event_root / MANIFEST_NAME
  rotated_path = event_root / ROTATED_MANIFEST_NAME
  state_path = event_root / STATE_NAME
  processed = set() if scan_all else set(load_json(state_path, []))
  known_ids = set() if scan_all else load_event_ids(manifest_path, rotated_path)
  present_segments: set[str] = set()
  added = 0

  for segment_path in _segment_paths(log_root):
    present_segments.add(segment_path.name)
    if segment_path.name in processed or not is_complete(segment_path) or find_rlog(segment_path) is None:
      continue
    if not scan_all and not is_preserved(segment_path):
      continue
    try:
      events = scan_segment(segment_path, reader_factory)
    except Exception:
      cloudlog.exception(f"driving_event_indexer: failed to parse {segment_path}")
      continue

    for event in events:
      event["context_segments"] = protect_context(log_root, event["route"], event["segment"])
    added += append_events(manifest_path, rotated_path, events, known_ids)
    processed.add(segment_path.name)

  processed &= present_segments
  atomic_write_json(state_path, sorted(processed))
  return added


def rebuild(log_root: Path, event_root: Path = EVENT_ROOT,
            reader_factory: Callable[[str], Iterable[Any]] | None = None) -> int:
  event_root.mkdir(parents=True, exist_ok=True)
  manifest_path = event_root / MANIFEST_NAME
  rotated_path = event_root / ROTATED_MANIFEST_NAME
  if manifest_path.exists():
    os.replace(manifest_path, rotated_path)
  state_path = event_root / STATE_NAME
  try:
    state_path.unlink()
  except FileNotFoundError:
    pass
  return scan_once(log_root, event_root, reader_factory, scan_all=True)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--rebuild", action="store_true", help="ignore processed state and inspect every supplied full rlog")
  parser.add_argument("--once", action="store_true")
  parser.add_argument("--log-root", type=Path, default=Path(Paths.log_root()))
  parser.add_argument("--event-root", type=Path, default=EVENT_ROOT)
  args = parser.parse_args()

  if args.rebuild:
    added = rebuild(args.log_root, args.event_root)
    cloudlog.info(f"driving_event_indexer: rebuilt {added} event(s)")
    return

  while True:
    try:
      added = scan_once(args.log_root, args.event_root)
      if added:
        cloudlog.info(f"driving_event_indexer: indexed {added} new event(s)")
    except Exception:
      cloudlog.exception("driving_event_indexer: scan failed")
    if args.once:
      return
    time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
  main()
