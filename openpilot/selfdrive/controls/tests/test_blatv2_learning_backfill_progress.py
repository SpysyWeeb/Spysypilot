from __future__ import annotations

import hashlib

import pytest  # noqa: TID251

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
