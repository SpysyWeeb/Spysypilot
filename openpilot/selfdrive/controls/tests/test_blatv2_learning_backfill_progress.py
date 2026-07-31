from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  RouteCandidate,
  RouteSegment,
  _BackfillProgressTracker,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_progress import (
  BACKFILL_PROGRESS_PARAM,
  BACKFILL_PROGRESS_SCHEMA_VERSION,
  BackfillProgressPhase,
  BackfillProgressPublisher,
  build_backfill_progress_bytes,
  decode_backfill_progress,
  validate_backfill_progress_payload,
)


OPERATION_ID = "1" * 32
ROUTE_ID = hashlib.sha256(b"route").hexdigest()


class FakeParams:
  def __init__(self) -> None:
    self.values: dict[str, object] = {}

  def put(
    self,
    key: str,
    value: dict[str, object],
    *,
    block: bool,
  ) -> None:
    assert block is True
    self.values[key] = dict(value)

  def remove(self, key: str) -> None:
    self.values.pop(key, None)


def operation(sequence: int = 4) -> dict[str, object]:
  return {"operation_id": OPERATION_ID, "sequence": sequence}


def payload(**changes: object) -> dict[str, object]:
  result: dict[str, object] = {
    "approximate_remaining_seconds": 123,
    "completed_replay_segment_count": 3,
    "completed_work_units": 300,
    "current_route_identity": ROUTE_ID,
    "current_route_index": 2,
    "current_route_segment_count": 4,
    "current_segment_index": 3,
    "informational_only": True,
    "operation_id": OPERATION_ID,
    "operation_sequence": 4,
    "pass_count": 2,
    "pass_index": 1,
    "phase": "reading_segment",
    "schema_version": BACKFILL_PROGRESS_SCHEMA_VERSION,
    "sequence": 7,
    "total_replay_segment_count": 20,
    "total_route_count": 5,
    "total_work_units": 1000,
    "updated_mono_ns": 900,
  }
  result.update(changes)
  return result


def test_progress_schema_is_canonical_and_exact() -> None:
  encoded = build_backfill_progress_bytes(**payload())
  decoded = decode_backfill_progress(encoded)

  assert decoded == payload()
  assert encoded == build_backfill_progress_bytes(**decoded)
  with pytest.raises(ValueError, match="keys do not match"):
    validate_backfill_progress_payload({**decoded, "extra": 1})


@pytest.mark.parametrize(
  "changes, message",
  (
    ({"pass_index": 0}, "pass"),
    ({"pass_count": 3}, "pass"),
    ({"current_route_index": 0}, "coordinate"),
    ({"current_segment_index": 5}, "coordinate"),
    ({"completed_replay_segment_count": 21}, "totals"),
    ({"completed_work_units": 1001}, "totals"),
    (
      {
        "phase": "applying_route",
        "current_segment_index": 3,
      },
      "final segment",
    ),
    (
      {
        "phase": "comparing",
        "current_route_identity": None,
        "current_route_index": None,
        "current_segment_index": None,
        "current_route_segment_count": None,
        "approximate_remaining_seconds": None,
      },
      "completed replay",
    ),
  ),
)
def test_progress_schema_rejects_incoherent_shapes(
  changes: dict[str, object],
  message: str,
) -> None:
  with pytest.raises(ValueError, match=message):
    build_backfill_progress_bytes(**payload(**changes))


def test_comparing_and_publishing_require_complete_replay() -> None:
  complete = payload(
    approximate_remaining_seconds=None,
    completed_replay_segment_count=20,
    completed_work_units=1000,
    current_route_identity=None,
    current_route_index=None,
    current_route_segment_count=None,
    current_segment_index=None,
    pass_index=2,
    phase="comparing",
  )
  assert decode_backfill_progress(
    build_backfill_progress_bytes(**complete),
  )["phase"] == "comparing"
  complete["phase"] = "publishing"
  assert decode_backfill_progress(
    build_backfill_progress_bytes(**complete),
  )["phase"] == "publishing"


def test_publisher_binds_operation_and_allows_pass_two_route_reset() -> None:
  params = FakeParams()
  ticks = iter((100, 200, 300))
  publisher = BackfillProgressPublisher(
    params,
    monotonic_ns=lambda: next(ticks),
  )
  common = {
    "pass_count": 2,
    "total_route_count": 5,
    "total_replay_segment_count": 20,
    "total_work_units": 1000,
    "approximate_remaining_seconds": None,
  }
  publisher.publish(
    operation_status=operation(),
    phase=BackfillProgressPhase.READING_SEGMENT,
    pass_index=1,
    current_route_identity=ROUTE_ID,
    current_route_index=5,
    current_segment_index=4,
    current_route_segment_count=4,
    completed_replay_segment_count=9,
    completed_work_units=490,
    **common,
  )
  publisher.publish(
    operation_status=operation(5),
    phase=BackfillProgressPhase.READING_SEGMENT,
    pass_index=2,
    current_route_identity=ROUTE_ID,
    current_route_index=1,
    current_segment_index=1,
    current_route_segment_count=4,
    completed_replay_segment_count=10,
    completed_work_units=500,
    **common,
  )

  latest = params.values[BACKFILL_PROGRESS_PARAM]
  assert type(latest) is dict
  assert latest["pass_index"] == 2
  assert latest["current_route_index"] == 1
  assert latest["sequence"] == 1
  assert latest["operation_sequence"] == 5

  with pytest.raises(ValueError, match="inventory changed"):
    publisher.publish(
      operation_status=operation(6),
      phase=BackfillProgressPhase.READING_SEGMENT,
      pass_index=2,
      current_route_identity=ROUTE_ID,
      current_route_index=1,
      current_segment_index=2,
      current_route_segment_count=4,
      completed_replay_segment_count=10,
      completed_work_units=500,
      **{**common, "total_work_units": 1001},
    )

  publisher.clear()
  assert BACKFILL_PROGRESS_PARAM not in params.values


def test_tracker_counts_each_completed_segment_before_next_read() -> None:
  segments = (
    RouteSegment(
      index=0,
      path=Path("/route/rlog--00"),
      sha256="2" * 64,
      size_bytes=10,
    ),
    RouteSegment(
      index=1,
      path=Path("/route/rlog--01"),
      sha256="3" * 64,
      size_bytes=20,
    ),
  )
  route = RouteCandidate(
    route_name="00000001--0000000001",
    route_counter=1,
    segments=segments,
  )
  params = FakeParams()
  publisher = BackfillProgressPublisher(
    params,
    monotonic_ns=iter((100, 200)).__next__,
  )
  tracker = _BackfillProgressTracker(
    routes=(route,),
    operation_status=SimpleNamespace(last_payload=operation()),
    publisher=publisher,
    abort_requested=lambda: False,
    monotonic_ns=iter((1_000, 2_000, 3_000)).__next__,
  )

  tracker.segment_started(
    pass_index=1,
    route=route,
    segment=segments[0],
    segment_index=1,
    segment_count=2,
  )
  tracker.segment_completed(
    pass_index=1,
    route=route,
    segment=segments[0],
    segment_index=1,
    segment_count=2,
  )
  tracker.segment_started(
    pass_index=1,
    route=route,
    segment=segments[1],
    segment_index=2,
    segment_count=2,
  )

  latest = params.values[BACKFILL_PROGRESS_PARAM]
  assert type(latest) is dict
  assert latest["completed_replay_segment_count"] == 1
  assert latest["completed_work_units"] == 10
  assert latest["current_segment_index"] == 2


def test_eta_requires_three_independent_samples_and_uses_medians() -> None:
  segment = RouteSegment(
    index=0,
    path=Path("/route/rlog--00"),
    sha256="2" * 64,
    size_bytes=30,
  )
  tracker = _BackfillProgressTracker(
    routes=(RouteCandidate(
      route_name="00000001--0000000001",
      route_counter=1,
      segments=(segment,),
    ),),
    operation_status=SimpleNamespace(last_payload=operation()),
    publisher=BackfillProgressPublisher(FakeParams()),
    abort_requested=lambda: False,
  )
  tracker._completed_read_units = 10
  tracker._read_rates = [1.0, 3.0]
  tracker._apply_rates = [4.0, 6.0]

  assert tracker._remaining_seconds() is None

  tracker._read_rates.append(2.0)
  tracker._apply_rates.append(2.0)

  # Two passes contain 60 read and 60 apply byte-units. Ten read units are
  # complete, so median rates produce 50*2 + 60*4 = 340 seconds.
  assert tracker._remaining_seconds() == 340
