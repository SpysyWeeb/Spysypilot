"""Read-only prediction of the production torque envelope.

This module answers one question: what normalized torque would the existing
runtime actuator envelope permit for each raw request? It does not own the raw
request and its projected values are diagnostics/predictions, not a second
live command path. The sole live output envelope remains
``blatv2.actuator.apply_torque_envelope``.

Every envelope decision delegates to
``actuator.apply_torque_envelope_counts``. Vehicle limits arrive through
``RuntimeTorqueLimits``; this module contains no platform torque or rate
literals and no duplicate limiter arithmetic.

The contract is count-exact. A raw normalized request is quantized with the
same ``round(request * steer_max)`` conversion as the live envelope. A request
is feasible when the production envelope returns that requested count
unchanged. Its normalized projection is then exactly
``requested_counts / steer_max``; it may differ from the original float by at
most unavoidable count quantization. ``unmet_torque`` retains that
quantization residual and any envelope residual as ``raw - feasible``.

The production limiter is Markov in the previous *applied* count only; it has
no previous-request state. Consequently the sequence API accepts one initial
applied count and publishes each derived requested count explicitly.
"""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from enum import IntEnum
import math
from numbers import Integral

from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
  apply_torque_envelope_counts,
)


class FeasibilityStatus(IntEnum):
  """Allocation-free status returned by the caller-buffer API."""

  OK = 0
  INVALID_INITIAL_STATE = 1
  INVALID_REQUEST = 2
  INVALID_DRIVER_TORQUE = 3


class ConstraintReason(IntEnum):
  """Per-sample explanation of projection state.

  ``ACTUATOR_ENVELOPE`` deliberately remains one reason: opendbc's production
  function owns the combined magnitude, driver, and rate decision. Recreating
  pieces of that decision here merely for classification would create a
  second implementation that could drift.
  """

  NONE = 0
  ACTUATOR_ENVELOPE = 1
  INVALID_INITIAL_STATE = 2
  INVALID_REQUEST = 3
  INVALID_DRIVER_TORQUE = 4
  INVALID_DEPENDENT_STATE = 5


@dataclass(frozen=True, slots=True)
class CurrentFeasibility:
  """Immutable diagnostic for one request.

  This convenience helper may allocate this result. The sequence hot path
  below constructs no per-sample result objects.
  """

  status: FeasibilityStatus
  raw_requested_torque: float
  feasible_applied_torque: float
  unmet_torque: float
  requested_counts: int | None
  feasible_counts: int | None
  constraint_active: bool
  constraint_reason: ConstraintReason

  @property
  def valid(self) -> bool:
    return self.status == FeasibilityStatus.OK

  @property
  def count_exactly_reachable(self) -> bool:
    return self.valid and self.requested_counts == self.feasible_counts and not self.constraint_active


def _coerce_float(value: float) -> float:
  try:
    return float(value)
  except (TypeError, ValueError, OverflowError):
    return math.nan


def _requested_counts(
  raw_requested_torque: float,
  limits: RuntimeTorqueLimits,
) -> int | None:
  scaled_request = raw_requested_torque * limits.steer_max
  if not math.isfinite(scaled_request):
    return None
  return int(round(scaled_request))


def _buffers_have_capacity(
  sample_count: int,
  *buffers: MutableSequence[object],
) -> bool:
  return all(len(buffer) >= sample_count for buffer in buffers)


def _invalidate_projection_suffix(
  raw_requested_torques: Sequence[float],
  start_index: int,
  sample_count: int,
  first_reason: ConstraintReason,
  output_raw_requested_torques: MutableSequence[float],
  output_feasible_applied_torques: MutableSequence[float],
  output_unmet_torques: MutableSequence[float],
  output_requested_counts: MutableSequence[int],
  output_feasible_counts: MutableSequence[int],
  output_constraint_active: MutableSequence[bool],
  output_constraint_reasons: MutableSequence[int],
) -> None:
  """Overwrite an invalid dependent suffix so reused buffers cannot look valid."""
  for index in range(start_index, sample_count):
    output_raw_requested_torques[index] = _coerce_float(
      raw_requested_torques[index],
    )
    output_feasible_applied_torques[index] = math.nan
    output_unmet_torques[index] = math.nan
    output_requested_counts[index] = 0
    output_feasible_counts[index] = 0
    output_constraint_active[index] = False
    output_constraint_reasons[index] = int(first_reason if index == start_index else ConstraintReason.INVALID_DEPENDENT_STATE)


def project_torque_feasibility_into(
  limits: RuntimeTorqueLimits,
  initial_applied_counts: int,
  raw_requested_torques: Sequence[float],
  driver_torques: Sequence[float],
  sample_count: int,
  output_raw_requested_torques: MutableSequence[float],
  output_feasible_applied_torques: MutableSequence[float],
  output_unmet_torques: MutableSequence[float],
  output_requested_counts: MutableSequence[int],
  output_feasible_counts: MutableSequence[int],
  output_constraint_active: MutableSequence[bool],
  output_constraint_reasons: MutableSequence[int],
) -> FeasibilityStatus:
  """Project raw demands through repeated production-envelope invocations.

  Each input sample represents one invocation of the live actuator envelope.
  The projected feasible count becomes the previous applied count for the next
  prediction. The raw demand sequence is copied, never modified or replaced.
  In particular, callers must not route ``output_feasible_applied_torques``
  back into the live command path; only the sole actuator envelope may shape
  the live request.

  The caller supplies all arrays. Configuration mistakes (negative count,
  short input/output buffers) raise ``ValueError`` before any output is
  touched. Runtime-invalid numeric input returns a status code and overwrites
  the invalid dependent suffix with non-finite projections plus explicit
  reasons, preventing stale finite buffer contents from masquerading as a
  command.
  """
  if sample_count < 0:
    raise ValueError("sample_count must be non-negative")
  if len(raw_requested_torques) < sample_count or len(driver_torques) < sample_count:
    raise ValueError("feasibility input buffers are shorter than sample_count")
  if not _buffers_have_capacity(
    sample_count,
    output_raw_requested_torques,
    output_feasible_applied_torques,
    output_unmet_torques,
    output_requested_counts,
    output_feasible_counts,
    output_constraint_active,
    output_constraint_reasons,
  ):
    raise ValueError("feasibility output buffers are shorter than sample_count")

  if isinstance(initial_applied_counts, bool) or not isinstance(
    initial_applied_counts,
    Integral,
  ):
    _invalidate_projection_suffix(
      raw_requested_torques,
      0,
      sample_count,
      ConstraintReason.INVALID_INITIAL_STATE,
      output_raw_requested_torques,
      output_feasible_applied_torques,
      output_unmet_torques,
      output_requested_counts,
      output_feasible_counts,
      output_constraint_active,
      output_constraint_reasons,
    )
    return FeasibilityStatus.INVALID_INITIAL_STATE

  previous_applied_counts = int(initial_applied_counts)
  for index in range(sample_count):
    raw_request = _coerce_float(raw_requested_torques[index])
    output_raw_requested_torques[index] = raw_request
    requested_counts = _requested_counts(raw_request, limits) if math.isfinite(raw_request) else None
    if requested_counts is None:
      _invalidate_projection_suffix(
        raw_requested_torques,
        index,
        sample_count,
        ConstraintReason.INVALID_REQUEST,
        output_raw_requested_torques,
        output_feasible_applied_torques,
        output_unmet_torques,
        output_requested_counts,
        output_feasible_counts,
        output_constraint_active,
        output_constraint_reasons,
      )
      return FeasibilityStatus.INVALID_REQUEST

    driver_torque = _coerce_float(driver_torques[index])
    if not math.isfinite(driver_torque):
      _invalidate_projection_suffix(
        raw_requested_torques,
        index,
        sample_count,
        ConstraintReason.INVALID_DRIVER_TORQUE,
        output_raw_requested_torques,
        output_feasible_applied_torques,
        output_unmet_torques,
        output_requested_counts,
        output_feasible_counts,
        output_constraint_active,
        output_constraint_reasons,
      )
      output_requested_counts[index] = requested_counts
      return FeasibilityStatus.INVALID_DRIVER_TORQUE

    feasible_counts = apply_torque_envelope_counts(
      limits,
      requested_counts,
      previous_applied_counts,
      driver_torque,
    )
    feasible_torque = feasible_counts / limits.steer_max
    constraint_active = feasible_counts != requested_counts

    output_feasible_applied_torques[index] = feasible_torque
    output_unmet_torques[index] = raw_request - feasible_torque
    output_requested_counts[index] = requested_counts
    output_feasible_counts[index] = feasible_counts
    output_constraint_active[index] = constraint_active
    output_constraint_reasons[index] = int(ConstraintReason.ACTUATOR_ENVELOPE if constraint_active else ConstraintReason.NONE)
    previous_applied_counts = feasible_counts

  return FeasibilityStatus.OK


def inspect_current_torque_feasibility(
  limits: RuntimeTorqueLimits,
  previous_applied_counts: int,
  raw_requested_torque: float,
  driver_torque: float,
) -> CurrentFeasibility:
  """Report, but never actuate, one raw request against the live envelope."""
  raw_request = _coerce_float(raw_requested_torque)
  if isinstance(previous_applied_counts, bool) or not isinstance(
    previous_applied_counts,
    Integral,
  ):
    return CurrentFeasibility(
      status=FeasibilityStatus.INVALID_INITIAL_STATE,
      raw_requested_torque=raw_request,
      feasible_applied_torque=math.nan,
      unmet_torque=math.nan,
      requested_counts=None,
      feasible_counts=None,
      constraint_active=False,
      constraint_reason=ConstraintReason.INVALID_INITIAL_STATE,
    )

  requested_counts = _requested_counts(raw_request, limits) if math.isfinite(raw_request) else None
  if requested_counts is None:
    return CurrentFeasibility(
      status=FeasibilityStatus.INVALID_REQUEST,
      raw_requested_torque=raw_request,
      feasible_applied_torque=math.nan,
      unmet_torque=math.nan,
      requested_counts=None,
      feasible_counts=None,
      constraint_active=False,
      constraint_reason=ConstraintReason.INVALID_REQUEST,
    )

  driver = _coerce_float(driver_torque)
  if not math.isfinite(driver):
    return CurrentFeasibility(
      status=FeasibilityStatus.INVALID_DRIVER_TORQUE,
      raw_requested_torque=raw_request,
      feasible_applied_torque=math.nan,
      unmet_torque=math.nan,
      requested_counts=requested_counts,
      feasible_counts=None,
      constraint_active=False,
      constraint_reason=ConstraintReason.INVALID_DRIVER_TORQUE,
    )

  feasible_counts = apply_torque_envelope_counts(
    limits,
    requested_counts,
    int(previous_applied_counts),
    driver,
  )
  feasible_torque = feasible_counts / limits.steer_max
  constraint_active = feasible_counts != requested_counts
  return CurrentFeasibility(
    status=FeasibilityStatus.OK,
    raw_requested_torque=raw_request,
    feasible_applied_torque=feasible_torque,
    unmet_torque=raw_request - feasible_torque,
    requested_counts=requested_counts,
    feasible_counts=feasible_counts,
    constraint_active=constraint_active,
    constraint_reason=(ConstraintReason.ACTUATOR_ENVELOPE if constraint_active else ConstraintReason.NONE),
  )
