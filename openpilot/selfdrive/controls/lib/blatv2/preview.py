"""Exact two-second count-space reachability for a surrounding horizon planner.

This module has no path objective, vehicle plant, confidence schedule, or policy
authority. It cannot decide that early torque is path-safe and cannot authorize
an early command. A surrounding plant/path-aware planner may use the facts here
to decide whether one exact count-space witness serves its own horizon objective.

The authored normalized demands are quantized and magnitude-clamped without
changing their sign. A backward pass constructs a neutral-future-driver
preparation witness, and a forward pass anchors that witness to the actual
previously applied CAN count. The present driver sample applies only to the
initial transition. Future transitions explicitly assume neutral driver torque
because future driver input is unknowable and the caller replans next frame.

Every reachable transition is constructed from cached production-envelope
lookups and then verified through ``apply_torque_envelope_counts``. Only the
first requested count is a live-frame candidate, not an authorization; opendbc
remains the final actuator-envelope authority.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
import math
from numbers import Integral

from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
  apply_torque_envelope_counts,
)


REACHABILITY_HORIZON_S = 2.0
REACHABILITY_FIXED_DT_S = 0.01
REACHABILITY_SAMPLE_COUNT = 201
REACHABILITY_LAST_SAMPLE_INDEX = REACHABILITY_SAMPLE_COUNT - 1
_DT_REPRESENTATION_TOLERANCE_S = math.ulp(REACHABILITY_FIXED_DT_S)


class ReachabilityStatus(IntEnum):
  """Status of one reused reachability result."""

  OK = 0
  DRIVER_SUPPRESSED = 2
  INVALID_INPUT = 3
  AUTHORED_SEQUENCE_MISS = 4
  ENVELOPE_MISMATCH = 5


class ReachabilityResult:
  """Reused scalar facts consumed by the surrounding horizon controller."""

  __slots__ = (
    "status",
    "valid",
    "preparation_active_now",
    "preparation_scheduled_later",
    "authored_sequence_exactly_reachable",
    "witness_sequence_reachable",
    "first_authored_miss_index",
    "first_authored_miss_time_s",
    "maximum_absolute_authored_residual_counts",
  )

  def __init__(self) -> None:
    self.clear(ReachabilityStatus.INVALID_INPUT)

  def clear(self, status: ReachabilityStatus) -> None:
    self.status = status
    self.valid = False
    self.preparation_active_now = False
    self.preparation_scheduled_later = False
    self.authored_sequence_exactly_reachable = False
    self.witness_sequence_reachable = False
    self.first_authored_miss_index = -1
    self.first_authored_miss_time_s = None
    self.maximum_absolute_authored_residual_counts = 0


def _configuration_float(value: object, field: str) -> float:
  if isinstance(value, (bool, str, bytes, bytearray)):
    raise TypeError(f"{field} must be numeric")
  try:
    converted = float(value)
  except (TypeError, ValueError, OverflowError) as error:
    raise TypeError(f"{field} must be numeric") from error
  if not math.isfinite(converted):
    raise ValueError(f"{field} must be finite")
  return converted


def _runtime_float(value: object) -> float | None:
  if isinstance(value, (bool, str, bytes, bytearray)):
    return None
  try:
    converted = float(value)
  except (TypeError, ValueError, OverflowError):
    return None
  return converted if math.isfinite(converted) else None


class ReachableCountProjector:
  """Preallocated reachability primitive for the exact 201-sample live grid.

  ``witness_counts[0]`` is only the first element of a mathematical
  witness. This class has insufficient information to decide whether requesting
  it early is correct for the model path or physical plant.
  """

  def __init__(
    self,
    *,
    fixed_dt_s: float,
    limits: RuntimeTorqueLimits,
  ) -> None:
    dt = _configuration_float(fixed_dt_s, "reachability fixed dt")
    if abs(dt - REACHABILITY_FIXED_DT_S) > _DT_REPRESENTATION_TOLERANCE_S:
      raise ValueError("reachability fixed dt must be the exact 100 Hz control period")
    if not isinstance(limits, RuntimeTorqueLimits):
      raise TypeError("reachability limits have the wrong type")
    for field in (
      "steer_max",
      "delta_up",
      "delta_down",
      "steer_step",
      "driver_allowance",
      "driver_multiplier",
      "driver_factor",
    ):
      value = getattr(limits, field)
      if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("reachability limits must use exact integral counts")
    if limits.steer_step != 1:
      raise ValueError("reachability requires the live 100 Hz steer-step contract")
    if limits.production_envelope_verified is not True:
      raise ValueError("reachability requires a verified production envelope")
    self.fixed_dt_s = REACHABILITY_FIXED_DT_S
    self.limits = limits
    self.result = ReachabilityResult()
    self.authored_counts = [0] * REACHABILITY_SAMPLE_COUNT
    self.reactive_counts = [0] * REACHABILITY_SAMPLE_COUNT
    self.backward_witness_counts = [0] * REACHABILITY_SAMPLE_COUNT
    self.witness_counts = [0] * REACHABILITY_SAMPLE_COUNT
    table_size = 2 * self.limits.steer_max + 1
    self._predecessor_lower = [0] * table_size
    self._predecessor_upper = [0] * table_size
    self._reachable_lower = [0] * table_size
    self._reachable_upper = [0] * table_size
    self._build_neutral_lookup()

  def _build_neutral_lookup(self) -> None:
    """Build and boundary-check exact neutral-driver predecessor lookups."""
    steer_max = self.limits.steer_max
    for previous_counts in range(-steer_max, steer_max + 1):
      table_index = previous_counts + steer_max
      lower = apply_torque_envelope_counts(
        self.limits,
        -steer_max,
        previous_counts,
        0.0,
      )
      upper = apply_torque_envelope_counts(
        self.limits,
        steer_max,
        previous_counts,
        0.0,
      )
      if (
        lower > upper
        or apply_torque_envelope_counts(self.limits, lower, previous_counts, 0.0) != lower
        or apply_torque_envelope_counts(self.limits, upper, previous_counts, 0.0) != upper
      ):
        raise ValueError("production envelope has a non-contiguous forward image")
      self._reachable_lower[table_index] = lower
      self._reachable_upper[table_index] = upper

    maximum_delta = max(self.limits.delta_up, self.limits.delta_down)
    for target_counts in range(-steer_max, steer_max + 1):
      lower_search = max(-steer_max, target_counts - maximum_delta - 1)
      upper_search = min(steer_max, target_counts + maximum_delta + 1)
      lower = steer_max + 1
      upper = -steer_max - 1
      for previous_counts in range(lower_search, upper_search + 1):
        if (
          apply_torque_envelope_counts(
            self.limits,
            target_counts,
            previous_counts,
            0.0,
          )
          == target_counts
        ):
          lower = min(lower, previous_counts)
          upper = max(upper, previous_counts)
      if (
        lower > upper
        or apply_torque_envelope_counts(self.limits, target_counts, lower, 0.0) != target_counts
        or apply_torque_envelope_counts(self.limits, target_counts, upper, 0.0) != target_counts
        or (lower > -steer_max and apply_torque_envelope_counts(self.limits, target_counts, lower - 1, 0.0) == target_counts)
        or (upper < steer_max and apply_torque_envelope_counts(self.limits, target_counts, upper + 1, 0.0) == target_counts)
      ):
        raise ValueError("production envelope has a non-contiguous predecessor image")
      table_index = target_counts + steer_max
      self._predecessor_lower[table_index] = lower
      self._predecessor_upper[table_index] = upper

  def _clear_arrays(self) -> None:
    for index in range(REACHABILITY_SAMPLE_COUNT):
      self.authored_counts[index] = 0
      self.reactive_counts[index] = 0
      self.backward_witness_counts[index] = 0
      self.witness_counts[index] = 0

  def _fail_closed(self, status: ReachabilityStatus) -> ReachabilityResult:
    self._clear_arrays()
    self.result.clear(status)
    return self.result

  def _apply_neutral_lookup(self, requested_counts: int, previous_counts: int) -> int:
    table_index = previous_counts + self.limits.steer_max
    return min(
      max(
        requested_counts,
        self._reachable_lower[table_index],
      ),
      self._reachable_upper[table_index],
    )

  def project_neutral_counts(self, requested_counts: int, previous_counts: int) -> int:
    """Apply the constructor-verified neutral-driver production image."""
    if (
      isinstance(requested_counts, bool)
      or not isinstance(requested_counts, Integral)
      or isinstance(previous_counts, bool)
      or not isinstance(previous_counts, Integral)
      or abs(int(previous_counts)) > self.limits.steer_max
    ):
      raise ValueError("neutral projection counts are outside the exact envelope")
    return self._apply_neutral_lookup(int(requested_counts), int(previous_counts))

  def _driver_arithmetic_is_finite(self, driver_torque: float) -> bool:
    scaled = driver_torque * self.limits.driver_factor
    if not math.isfinite(scaled):
      return False
    maximum_term = (self.limits.driver_allowance + scaled) * self.limits.driver_multiplier
    minimum_term = (-self.limits.driver_allowance + scaled) * self.limits.driver_multiplier
    return math.isfinite(maximum_term) and math.isfinite(minimum_term)

  def _set_first_command(
    self,
    *,
    status: ReachabilityStatus,
    authored_counts: int,
    reactive_counts: int,
  ) -> ReachabilityResult:
    """Publish only the reactive transition while future input is unknown."""
    residual = authored_counts - reactive_counts
    absolute_residual = abs(residual)
    self.backward_witness_counts[0] = reactive_counts
    self.witness_counts[0] = reactive_counts
    self.result.status = status
    self.result.valid = True
    self.result.maximum_absolute_authored_residual_counts = absolute_residual
    if residual != 0:
      self.result.first_authored_miss_index = 0
      self.result.first_authored_miss_time_s = 0.0
    return self.result

  def _update(
    self,
    *,
    ideal_torques: Sequence[float],
    previous_applied_counts: int,
    driver_torque: float,
    steering_pressed: bool,
  ) -> ReachabilityResult:
    if (
      isinstance(previous_applied_counts, bool)
      or not isinstance(previous_applied_counts, Integral)
      or type(steering_pressed) is not bool
      or isinstance(ideal_torques, (str, bytes, bytearray))
      or len(ideal_torques) != REACHABILITY_SAMPLE_COUNT
    ):
      return self.result
    previous_counts = int(previous_applied_counts)
    if abs(previous_counts) > self.limits.steer_max:
      return self.result
    driver = _runtime_float(driver_torque)
    if driver is None or not self._driver_arithmetic_is_finite(driver):
      return self.result

    steer_max = self.limits.steer_max
    for index in range(REACHABILITY_SAMPLE_COUNT):
      raw_demand = _runtime_float(ideal_torques[index])
      if raw_demand is None:
        return self._fail_closed(ReachabilityStatus.INVALID_INPUT)
      scaled_demand = raw_demand * steer_max
      if not math.isfinite(scaled_demand):
        return self._fail_closed(ReachabilityStatus.INVALID_INPUT)
      requested_counts = int(round(scaled_demand))
      authored_counts = min(max(requested_counts, -steer_max), steer_max)
      self.authored_counts[index] = authored_counts

    reactive_first = apply_torque_envelope_counts(
      self.limits,
      self.authored_counts[0],
      previous_counts,
      driver,
    )
    self.reactive_counts[0] = reactive_first
    if (
      apply_torque_envelope_counts(
        self.limits,
        reactive_first,
        previous_counts,
        driver,
      )
      != reactive_first
    ):
      return self._fail_closed(ReachabilityStatus.ENVELOPE_MISMATCH)
    scaled_driver = driver * self.limits.driver_factor
    driver_suppressed = steering_pressed or abs(scaled_driver) > self.limits.driver_allowance
    if driver_suppressed:
      return self._set_first_command(
        status=ReachabilityStatus.DRIVER_SUPPRESSED,
        authored_counts=self.authored_counts[0],
        reactive_counts=reactive_first,
      )

    applied_previous = reactive_first
    for index in range(1, REACHABILITY_SAMPLE_COUNT):
      applied_previous = self._apply_neutral_lookup(
        self.authored_counts[index],
        applied_previous,
      )
      self.reactive_counts[index] = applied_previous

    self.backward_witness_counts[REACHABILITY_LAST_SAMPLE_INDEX] = self.authored_counts[REACHABILITY_LAST_SAMPLE_INDEX]
    for index in range(REACHABILITY_LAST_SAMPLE_INDEX - 1, -1, -1):
      target_counts = self.backward_witness_counts[index + 1]
      table_index = target_counts + steer_max
      lower = self._predecessor_lower[table_index]
      upper = self._predecessor_upper[table_index]
      self.backward_witness_counts[index] = min(
        max(
          self.authored_counts[index],
          lower,
        ),
        upper,
      )

    self.witness_counts[0] = apply_torque_envelope_counts(
      self.limits,
      self.backward_witness_counts[0],
      previous_counts,
      driver,
    )
    for index in range(1, REACHABILITY_SAMPLE_COUNT):
      self.witness_counts[index] = self._apply_neutral_lookup(
        self.backward_witness_counts[index],
        self.witness_counts[index - 1],
      )

    if apply_torque_envelope_counts(self.limits, self.witness_counts[0], previous_counts, driver) != self.witness_counts[0]:
      return self._fail_closed(ReachabilityStatus.ENVELOPE_MISMATCH)

    first_miss_index = -1
    maximum_absolute_residual = 0
    authored_sequence_exactly_reachable = True
    preparation_scheduled_later = False
    for index in range(REACHABILITY_SAMPLE_COUNT):
      residual = self.authored_counts[index] - self.witness_counts[index]
      absolute_residual = abs(residual)
      if residual != 0 and first_miss_index < 0:
        first_miss_index = index
      maximum_absolute_residual = max(maximum_absolute_residual, absolute_residual)
      authored_sequence_exactly_reachable &= self.reactive_counts[index] == self.authored_counts[index]
      if index > 0 and self.witness_counts[index] != self.reactive_counts[index]:
        preparation_scheduled_later = True

    requested_first = self.witness_counts[0]
    self.result.status = ReachabilityStatus.OK if first_miss_index < 0 else ReachabilityStatus.AUTHORED_SEQUENCE_MISS
    self.result.valid = True
    self.result.preparation_active_now = requested_first != reactive_first
    self.result.preparation_scheduled_later = preparation_scheduled_later
    self.result.authored_sequence_exactly_reachable = authored_sequence_exactly_reachable
    self.result.witness_sequence_reachable = True
    self.result.first_authored_miss_index = first_miss_index
    self.result.first_authored_miss_time_s = None if first_miss_index < 0 else first_miss_index * REACHABILITY_FIXED_DT_S
    self.result.maximum_absolute_authored_residual_counts = maximum_absolute_residual
    return self.result

  def update(
    self,
    *,
    ideal_torques: Sequence[float],
    previous_applied_counts: int,
    driver_torque: float,
    steering_pressed: bool,
  ) -> ReachabilityResult:
    """Project one exact two-second prefix without raising on runtime input."""
    self.result.clear(ReachabilityStatus.INVALID_INPUT)
    self._clear_arrays()
    try:
      return self._update(
        ideal_torques=ideal_torques,
        previous_applied_counts=previous_applied_counts,
        driver_torque=driver_torque,
        steering_pressed=steering_pressed,
      )
    except Exception:
      # Runtime inputs and foreign Sequence implementations are untrusted. No
      # partially reused witness may escape when their coercion/indexing fails.
      self._clear_arrays()
      self.result.clear(ReachabilityStatus.INVALID_INPUT)
      return self.result
