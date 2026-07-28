"""Real-route regression for detector-v7 under-delivery and takeover events."""
import csv
import hashlib
import io
import json
from pathlib import Path

import pytest
import zstandard

from openpilot.selfdrive.spysypilot.lat_event_detector import (
  TAKEOVER_CONFIRM_S,
  UNDER_DELIVERY_MAX_DELIVERED_FRACTION,
  UNDER_DELIVERY_MIN_HEADROOM,
  UNDER_DELIVERY_PERSIST_S,
  LateralEventDetector,
  LateralSample,
)


DATA_DIR = Path(__file__).parent / "data"
META = json.loads((DATA_DIR / "b7_authority_windows_meta.json").read_text())
WINDOWS = META["windows"]
_COMPRESSED = (DATA_DIR / META["fixture_csv"]).read_bytes()
assert hashlib.sha256(_COMPRESSED).hexdigest() == META["fixture_sha256"]
_CSV = zstandard.ZstdDecompressor().decompress(_COMPRESSED, max_output_size=64 * 1024**2)
_ROWS = list(csv.DictReader(io.StringIO(_CSV.decode())))

_BOOL_FIELDS = {
  "active",
  "steering_pressed",
  "unwind_same_episode",
  "road_confounded",
  "turn_in_blocked",
}
_INT_FIELDS = {"controller_version", "reference_version", "damping_version"}
_TEXT_FIELDS = {"damping_state", "actual_damping_state"}
_OPTIONAL_FLOAT_FIELDS = {
  "high_angle_unwind_scale",
  "torque_command_before_high_angle_exit",
  "high_angle_unwind_old_torque_correction",
  "high_angle_unwind_old_direction_torque",
}


def _sample(row: dict[str, str]) -> LateralSample:
  values = {}
  for name in LateralSample.__dataclass_fields__:
    value = row[name]
    if name in _BOOL_FIELDS:
      values[name] = value == "True"
    elif name in _INT_FIELDS:
      values[name] = int(value)
    elif name in _TEXT_FIELDS:
      values[name] = value
    elif name in _OPTIONAL_FLOAT_FIELDS:
      values[name] = float(value) if value else None
    else:
      values[name] = float(value)
  return LateralSample(**values)


def _replay(window: dict) -> list:
  detector = LateralEventDetector()
  events = []
  for row in _ROWS:
    route_time = float(row["route_time"])
    if not window["route_start_s"] <= route_time <= window["route_end_s"]:
      continue
    detected = detector.update(_sample(row))
    if detected is None:
      continue
    events.extend(detected if isinstance(detected, tuple) else (detected,))
  return events


def _route_time(event, row: dict[str, str]) -> float:
  route_origin = float(row["mono_time"]) - float(row["route_time"])
  return event.occurred_mono_time - route_origin


@pytest.mark.parametrize("window", WINDOWS, ids=[window["label"] for window in WINDOWS])
def test_failed_turn_surfaces_under_delivery_and_takeover(window):
  events = _replay(window)
  under_delivery = [event for event in events if event.event_type == "torqueUnderDelivery"]
  takeovers = [event for event in events if event.event_type == "driverTakeover"]
  assert under_delivery
  assert takeovers
  assert not any(event.event_type == "torqueAuthority" for event in events)
  row = next(row for row in _ROWS if float(row["route_time"]) >= window["route_start_s"])

  authority = min(
    under_delivery,
    key=lambda event: abs(_route_time(event, row) - window["torque_under_delivery_occurred_s"]),
  )
  assert _route_time(authority, row) == pytest.approx(
    window["torque_under_delivery_occurred_s"], abs=0.02,
  )
  assert authority.severity == "critical"
  assert authority.evidence is not None
  assert authority.evidence.authority_under_delivery_duration_s >= UNDER_DELIVERY_PERSIST_S
  assert authority.evidence.delivered_curvature_fraction <= UNDER_DELIVERY_MAX_DELIVERED_FRACTION
  assert authority.evidence.torque_headroom >= UNDER_DELIVERY_MIN_HEADROOM

  takeover = min(
    takeovers,
    key=lambda event: abs(_route_time(event, row) - window["driver_takeover_occurred_s"]),
  )
  assert _route_time(takeover, row) == pytest.approx(
    window["driver_takeover_occurred_s"], abs=0.02,
  )
  assert takeover.severity == window["expected_takeover_severity"]
  assert takeover.evidence is not None
  assert takeover.evidence.takeover_confirmation_duration_s >= TAKEOVER_CONFIRM_S


def test_b7_full_route_census_preserves_all_pre_v7_events():
  assert META["full_route_before_census"] == {
    "turnStopTurn": 23,
    "committedHandoffHarshness": 2,
    "stallRelease": 1,
    "torqueAuthority": 0,
  }
  for event_type, count in META["full_route_before_census"].items():
    assert META["full_route_after_census"][event_type] == count
  assert META["full_route_after_census"]["torqueUnderDelivery"] == 18
  assert META["full_route_after_census"]["driverTakeover"] == 41
