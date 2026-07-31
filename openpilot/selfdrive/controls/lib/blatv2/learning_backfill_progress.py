"""Display-only progress for deterministic historical-learning replay.

This projection is deliberately separate from learning evidence and from
``BLaTv2LearningOperationStatus``. It may describe timing and work already
performed, but it is never read by the learner, ledger, candidate builder, or
controller. Deleting or editing it cannot affect steering.
"""

from __future__ import annotations

from enum import StrEnum
import json
import re
import time
from typing import Any


BACKFILL_PROGRESS_PARAM = "BLaTv2BackfillProgress"
BACKFILL_PROGRESS_SCHEMA_VERSION = 1

_HEX_32_RE = re.compile(r"[0-9a-f]{32}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_KEYS = {
  "approximate_remaining_seconds",
  "completed_replay_segment_count",
  "completed_work_units",
  "current_route_identity",
  "current_route_index",
  "current_route_segment_count",
  "current_segment_index",
  "informational_only",
  "operation_id",
  "operation_sequence",
  "pass_count",
  "pass_index",
  "phase",
  "schema_version",
  "sequence",
  "total_replay_segment_count",
  "total_route_count",
  "total_work_units",
  "updated_mono_ns",
}


class BackfillProgressPhase(StrEnum):
  READING_SEGMENT = "reading_segment"
  APPLYING_ROUTE = "applying_route"
  COMPARING = "comparing"
  PUBLISHING = "publishing"


def _nonnegative_int(value: object, name: str) -> int:
  if type(value) is not int or value < 0:
    raise ValueError(f"{name} must be a nonnegative integer")
  return value


def _optional_nonnegative_int(
  value: object,
  name: str,
) -> int | None:
  if value is None:
    return None
  return _nonnegative_int(value, name)


def _optional_sha256(value: object, name: str) -> str | None:
  if value is None:
    return None
  if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
    raise ValueError(f"{name} must be null or a lowercase SHA-256")
  return value


def validate_backfill_progress_payload(
  payload: object,
) -> dict[str, object]:
  """Validate the exact, informational progress schema."""
  if type(payload) is not dict or set(payload) != _TOP_LEVEL_KEYS:
    raise ValueError("backfill progress keys do not match")
  if (
    type(payload["schema_version"]) is not int
    or payload["schema_version"] != BACKFILL_PROGRESS_SCHEMA_VERSION
    or payload["informational_only"] is not True
  ):
    raise ValueError("backfill progress schema is incompatible")

  operation_id = payload["operation_id"]
  if (
    type(operation_id) is not str
    or _HEX_32_RE.fullmatch(operation_id) is None
  ):
    raise ValueError("operation_id must be 32 lowercase hex characters")
  operation_sequence = _nonnegative_int(
    payload["operation_sequence"],
    "operation_sequence",
  )
  sequence = _nonnegative_int(payload["sequence"], "sequence")
  updated_mono_ns = _nonnegative_int(
    payload["updated_mono_ns"],
    "updated_mono_ns",
  )

  try:
    phase = BackfillProgressPhase(payload["phase"])
  except (TypeError, ValueError) as exc:
    raise ValueError("backfill progress phase is unknown") from exc

  pass_index = _nonnegative_int(payload["pass_index"], "pass_index")
  pass_count = _nonnegative_int(payload["pass_count"], "pass_count")
  if pass_count != 2 or not 1 <= pass_index <= pass_count:
    raise ValueError("backfill replay pass must be one or two of two")

  route_count = _nonnegative_int(
    payload["total_route_count"],
    "total_route_count",
  )
  if route_count == 0:
    raise ValueError("backfill progress requires at least one route")

  completed_segments = _nonnegative_int(
    payload["completed_replay_segment_count"],
    "completed_replay_segment_count",
  )
  total_segments = _nonnegative_int(
    payload["total_replay_segment_count"],
    "total_replay_segment_count",
  )
  completed_work = _nonnegative_int(
    payload["completed_work_units"],
    "completed_work_units",
  )
  total_work = _nonnegative_int(
    payload["total_work_units"],
    "total_work_units",
  )
  if (
    total_segments == 0
    or completed_segments > total_segments
    or total_work == 0
    or completed_work > total_work
  ):
    raise ValueError("backfill progress totals are outside their bounds")

  route_identity = _optional_sha256(
    payload["current_route_identity"],
    "current_route_identity",
  )
  route_index = _optional_nonnegative_int(
    payload["current_route_index"],
    "current_route_index",
  )
  segment_index = _optional_nonnegative_int(
    payload["current_segment_index"],
    "current_segment_index",
  )
  route_segment_count = _optional_nonnegative_int(
    payload["current_route_segment_count"],
    "current_route_segment_count",
  )
  coordinate = (
    route_identity,
    route_index,
    segment_index,
    route_segment_count,
  )
  replay_phase = phase in {
    BackfillProgressPhase.READING_SEGMENT,
    BackfillProgressPhase.APPLYING_ROUTE,
  }
  if replay_phase:
    if any(value is None for value in coordinate):
      raise ValueError("active replay progress requires a full coordinate")
    assert route_index is not None
    assert segment_index is not None
    assert route_segment_count is not None
    if (
      not 1 <= route_index <= route_count
      or route_segment_count == 0
      or not 1 <= segment_index <= route_segment_count
    ):
      raise ValueError("active replay coordinate is outside its bounds")
    if (
      phase is BackfillProgressPhase.APPLYING_ROUTE
      and segment_index != route_segment_count
    ):
      raise ValueError("route application must follow its final segment")
  elif any(value is not None for value in coordinate):
    raise ValueError("comparison/publication cannot claim a route coordinate")

  approximate_remaining = _optional_nonnegative_int(
    payload["approximate_remaining_seconds"],
    "approximate_remaining_seconds",
  )
  if not replay_phase and approximate_remaining is not None:
    raise ValueError("terminal replay stages cannot publish an ETA")
  if not replay_phase and (
    completed_segments != total_segments
    or completed_work != total_work
    or pass_index != pass_count
  ):
    raise ValueError("comparison/publication requires completed replay work")

  return {
    "approximate_remaining_seconds": approximate_remaining,
    "completed_replay_segment_count": completed_segments,
    "completed_work_units": completed_work,
    "current_route_identity": route_identity,
    "current_route_index": route_index,
    "current_route_segment_count": route_segment_count,
    "current_segment_index": segment_index,
    "informational_only": True,
    "operation_id": operation_id,
    "operation_sequence": operation_sequence,
    "pass_count": pass_count,
    "pass_index": pass_index,
    "phase": phase.value,
    "schema_version": BACKFILL_PROGRESS_SCHEMA_VERSION,
    "sequence": sequence,
    "total_replay_segment_count": total_segments,
    "total_route_count": route_count,
    "total_work_units": total_work,
    "updated_mono_ns": updated_mono_ns,
  }


def build_backfill_progress_bytes(**fields: object) -> bytes:
  payload = validate_backfill_progress_payload(fields)
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  ).encode("utf-8")


def decode_backfill_progress(
  encoded: str | bytes,
) -> dict[str, object]:
  try:
    payload = json.loads(encoded)
  except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise ValueError("backfill progress is not valid JSON") from exc
  return validate_backfill_progress_payload(payload)


class BackfillProgressPublisher:
  """Publish monotonic progress bound to one operation-status identity."""

  def __init__(
    self,
    params: Any,
    *,
    monotonic_ns: Any = time.monotonic_ns,
  ) -> None:
    self._params = params
    self._monotonic_ns = monotonic_ns
    self._operation_id: str | None = None
    self._sequence = -1
    self._last_payload: dict[str, object] | None = None

  @property
  def last_payload(self) -> dict[str, object] | None:
    return None if self._last_payload is None else dict(self._last_payload)

  def clear(self) -> None:
    self._params.remove(BACKFILL_PROGRESS_PARAM)
    self._operation_id = None
    self._sequence = -1
    self._last_payload = None

  def publish(
    self,
    *,
    operation_status: dict[str, object],
    phase: BackfillProgressPhase | str,
    pass_index: int,
    pass_count: int,
    current_route_identity: str | None,
    current_route_index: int | None,
    total_route_count: int,
    current_segment_index: int | None,
    current_route_segment_count: int | None,
    completed_replay_segment_count: int,
    total_replay_segment_count: int,
    completed_work_units: int,
    total_work_units: int,
    approximate_remaining_seconds: int | None,
  ) -> bytes:
    if type(operation_status) is not dict:
      raise TypeError("progress requires an operation-status payload")
    operation_id = operation_status.get("operation_id")
    operation_sequence = operation_status.get("sequence")
    if (
      type(operation_id) is not str
      or _HEX_32_RE.fullmatch(operation_id) is None
      or type(operation_sequence) is not int
      or operation_sequence < 0
    ):
      raise ValueError("progress operation binding is invalid")

    new_operation = operation_id != self._operation_id
    if new_operation:
      self._operation_id = operation_id
      self._sequence = 0
      self._last_payload = None
    else:
      self._sequence += 1

    encoded = build_backfill_progress_bytes(
      approximate_remaining_seconds=approximate_remaining_seconds,
      completed_replay_segment_count=(
        completed_replay_segment_count
      ),
      completed_work_units=completed_work_units,
      current_route_identity=current_route_identity,
      current_route_index=current_route_index,
      current_route_segment_count=current_route_segment_count,
      current_segment_index=current_segment_index,
      informational_only=True,
      operation_id=operation_id,
      operation_sequence=operation_sequence,
      pass_count=pass_count,
      pass_index=pass_index,
      phase=BackfillProgressPhase(phase).value,
      schema_version=BACKFILL_PROGRESS_SCHEMA_VERSION,
      sequence=self._sequence,
      total_replay_segment_count=total_replay_segment_count,
      total_route_count=total_route_count,
      total_work_units=total_work_units,
      updated_mono_ns=int(self._monotonic_ns()),
    )
    payload = decode_backfill_progress(encoded)
    previous = self._last_payload
    if previous is not None and not new_operation:
      if payload["sequence"] <= previous["sequence"]:
        raise ValueError("backfill progress sequence must advance")
      if payload["updated_mono_ns"] < previous["updated_mono_ns"]:
        raise ValueError("backfill progress timestamp moved backward")
      if payload["operation_sequence"] < previous["operation_sequence"]:
        raise ValueError("bound operation sequence moved backward")
      for total_name in (
        "pass_count",
        "total_route_count",
        "total_replay_segment_count",
        "total_work_units",
      ):
        if payload[total_name] != previous[total_name]:
          raise ValueError("backfill progress inventory changed")
      for completed_name in (
        "completed_replay_segment_count",
        "completed_work_units",
      ):
        if payload[completed_name] < previous[completed_name]:
          raise ValueError("backfill completed work moved backward")
      if payload["pass_index"] < previous["pass_index"]:
        raise ValueError("backfill replay pass moved backward")
      if (
        payload["pass_index"] == previous["pass_index"]
        and payload["current_route_index"] is not None
        and previous["current_route_index"] is not None
        and payload["current_route_index"]
        < previous["current_route_index"]
      ):
        raise ValueError("backfill route progress moved backward")
      if (
        payload["pass_index"] == previous["pass_index"]
        and payload["current_route_index"]
        == previous["current_route_index"]
        and payload["current_segment_index"] is not None
        and previous["current_segment_index"] is not None
        and payload["current_segment_index"]
        < previous["current_segment_index"]
      ):
        raise ValueError("backfill segment progress moved backward")

    self._params.put(BACKFILL_PROGRESS_PARAM, payload, block=True)
    self._last_payload = payload
    return encoded
