"""Canonical measured-frame construction shared by live and rlog learning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from openpilot.cereal.services import SERVICE_LIST
# Each source may be at most one-and-a-half declared publication periods older
# than a controls witness. This is an alignment bound, not controller tuning.
MAX_SOURCE_AGE_PERIODS = 1.5


def _copy_message(message: Any) -> Any:
  builder = getattr(message, "as_builder", None)
  return builder() if callable(builder) else message


@dataclass(frozen=True, slots=True)
class CanonicalSourceSnapshot:
  message: Any
  mono_ns: int
  previous_mono_ns: int | None
  valid: bool
  alive: bool


class CanonicalSourceHistory:
  """Retain the two newest snapshots and select one preceding a witness."""

  __slots__ = ("_current", "_previous")

  def __init__(self) -> None:
    self._current: CanonicalSourceSnapshot | None = None
    self._previous: CanonicalSourceSnapshot | None = None

  def update(
    self,
    *,
    message: Any,
    mono_ns: int,
    valid: bool,
    alive: bool,
  ) -> None:
    timestamp = int(mono_ns)
    if timestamp <= 0:
      return
    copied_message = _copy_message(message)
    if self._current is None or timestamp > self._current.mono_ns:
      self._previous = self._current
      self._current = CanonicalSourceSnapshot(
        message=copied_message,
        mono_ns=timestamp,
        previous_mono_ns=(
          None if self._previous is None else self._previous.mono_ns
        ),
        valid=bool(valid),
        alive=bool(alive),
      )
    elif timestamp == self._current.mono_ns:
      self._current = CanonicalSourceSnapshot(
        message=copied_message,
        mono_ns=timestamp,
        previous_mono_ns=self._current.previous_mono_ns,
        valid=bool(valid),
        alive=bool(alive),
      )
    elif (
      self._previous is None
      or timestamp >= self._previous.mono_ns
    ):
      # A late, out-of-order source publication can still be the canonical
      # predecessor of the current snapshot. Insert it into timestamp order:
      # its predecessor is the formerly retained previous snapshot, while it
      # becomes the current snapshot's predecessor. An equal-timestamp
      # replacement retains the already-established predecessor.
      if self._previous is None:
        previous_mono_ns = None
      elif timestamp == self._previous.mono_ns:
        previous_mono_ns = self._previous.previous_mono_ns
      else:
        previous_mono_ns = self._previous.mono_ns
      self._previous = CanonicalSourceSnapshot(
        message=copied_message,
        mono_ns=timestamp,
        previous_mono_ns=previous_mono_ns,
        valid=bool(valid),
        alive=bool(alive),
      )
      self._current = CanonicalSourceSnapshot(
        message=self._current.message,
        mono_ns=self._current.mono_ns,
        previous_mono_ns=timestamp,
        valid=self._current.valid,
        alive=self._current.alive,
      )

  def select(
    self,
    *,
    witness_mono_ns: int,
    maximum_age_ns: int,
  ) -> CanonicalSourceSnapshot | None:
    witness = int(witness_mono_ns)
    maximum_age = int(maximum_age_ns)
    if witness <= 0 or maximum_age < 0:
      return None
    eligible = tuple(
      snapshot
      for snapshot in (self._current, self._previous)
      if (
        snapshot is not None
        and snapshot.mono_ns <= witness
        and witness - snapshot.mono_ns <= maximum_age
      )
    )
    if not eligible:
      return None
    selected = max(eligible, key=lambda snapshot: snapshot.mono_ns)
    if not selected.valid or not selected.alive:
      return None
    return selected


def maximum_source_age_ns(service: str) -> int:
  frequency = float(SERVICE_LIST[service].frequency)
  if not math.isfinite(frequency) or frequency <= 0.0:
    raise ValueError("learning source must have a positive declared frequency")
  return int(round(MAX_SOURCE_AGE_PERIODS * 1e9 / frequency))
