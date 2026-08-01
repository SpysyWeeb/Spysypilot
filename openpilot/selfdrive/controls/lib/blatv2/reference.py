"""Stateless scalar-anchored sampling of the model's future path.

The scalar action is the sole path-placement authority. The native model path
contributes only shape and its analytic motion around that anchor:

  reference(t) = scalar + plan_curvature(t) - plan_curvature(action_time)

Curvature is reconstructed as ``orientationRate.z / velocity.x``. A
shape-preserving piecewise-cubic Hermite interpolant (PCHIP) supplies
curvature, rate, and acceleration without a least-squares fit, BLAS, or
overshoot on monotonic data. Derivatives are analytic inside each native plan
cell; acceleration can be discontinuous at a cell boundary, as expected for
a piecewise polynomial.

Speed uses the same analytic PCHIP, independently anchored to the measured
vehicle speed at the caller's explicit plan-relative ``plan_time_now_s``:

  speed_reference(t) = measured_v_ego + plan_speed(t) - plan_speed(plan_now)

The caller chooses every query timestamp. This module adds no actuation
delay, model smoothing offset, or other clock policy. The caller-buffer entry
point accepts all scratch and output storage from the caller. Convenience
wrappers allocate immutable tuples for tests, replay, and non-hot-path use.
No API retains state.
"""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
import math

from openpilot.selfdrive.controls.lib.blatv2.contracts import (
  ReferenceBuildStatus,
  ReferenceOutput,
  ReferenceQueryOutput,
)


def _buffers_have_capacity(
  capacity: int,
  *buffers: MutableSequence[float],
) -> bool:
  return all(len(buffer) >= capacity for buffer in buffers)


def _write_scalar_only(
  scalar_curvature: float,
  scalar_action_plan_s: float,
  output_times_s: MutableSequence[float],
  output_curvatures: MutableSequence[float],
  output_curvature_rates: MutableSequence[float],
  output_curvature_accelerations: MutableSequence[float],
  output_planned_speeds: MutableSequence[float],
) -> ReferenceBuildStatus:
  if not _buffers_have_capacity(
    1,
    output_times_s,
    output_curvatures,
    output_curvature_rates,
    output_curvature_accelerations,
    output_planned_speeds,
  ):
    raise ValueError("reference output buffers must hold at least one sample")
  scalar = float(scalar_curvature)
  if not math.isfinite(scalar):
    scalar = 0.0
  output_times_s[0] = float(scalar_action_plan_s)
  output_curvatures[0] = scalar
  output_curvature_rates[0] = 0.0
  output_curvature_accelerations[0] = 0.0
  output_planned_speeds[0] = 0.0
  return ReferenceBuildStatus(count=1, valid=False, scalar_only=True)


def _same_sign(left: float, right: float) -> bool:
  return (left > 0.0 and right > 0.0) or (left < 0.0 and right < 0.0)


def _limited_endpoint_tangent(
  first_width: float,
  second_width: float,
  first_slope: float,
  second_slope: float,
) -> float:
  tangent = ((2.0 * first_width + second_width) * first_slope - first_width * second_slope) / (first_width + second_width)
  if not _same_sign(tangent, first_slope):
    return 0.0
  if not _same_sign(first_slope, second_slope) and abs(tangent) > 3.0 * abs(first_slope):
    return 3.0 * first_slope
  return tangent


def _fill_pchip_tangents(
  times_s: Sequence[float],
  values: Sequence[float],
  count: int,
  tangents: MutableSequence[float],
) -> None:
  if len(tangents) < count:
    raise ValueError("PCHIP tangent scratch buffer is too small")
  if count == 2:
    slope = (float(values[1]) - float(values[0])) / (float(times_s[1]) - float(times_s[0]))
    tangents[0] = slope
    tangents[1] = slope
    return

  first_width = float(times_s[1]) - float(times_s[0])
  second_width = float(times_s[2]) - float(times_s[1])
  first_slope = (float(values[1]) - float(values[0])) / first_width
  second_slope = (float(values[2]) - float(values[1])) / second_width
  tangents[0] = _limited_endpoint_tangent(
    first_width,
    second_width,
    first_slope,
    second_slope,
  )

  previous_width = first_width
  previous_slope = first_slope
  for index in range(1, count - 1):
    next_width = float(times_s[index + 1]) - float(times_s[index])
    next_slope = (float(values[index + 1]) - float(values[index])) / next_width
    if previous_slope == 0.0 or next_slope == 0.0 or not _same_sign(previous_slope, next_slope):
      tangents[index] = 0.0
    else:
      previous_weight = 2.0 * next_width + previous_width
      next_weight = next_width + 2.0 * previous_width
      tangents[index] = (previous_weight + next_weight) / (previous_weight / previous_slope + next_weight / next_slope)
    previous_width = next_width
    previous_slope = next_slope

  last_width = float(times_s[count - 1]) - float(times_s[count - 2])
  penultimate_width = float(times_s[count - 2]) - float(times_s[count - 3])
  last_slope = (float(values[count - 1]) - float(values[count - 2])) / last_width
  penultimate_slope = (float(values[count - 2]) - float(values[count - 3])) / penultimate_width
  tangents[count - 1] = _limited_endpoint_tangent(
    last_width,
    penultimate_width,
    last_slope,
    penultimate_slope,
  )


def _segment_index(
  times_s: Sequence[float],
  count: int,
  sample_time_s: float,
) -> int:
  if sample_time_s <= float(times_s[0]):
    return 0
  if sample_time_s >= float(times_s[count - 1]):
    return count - 2
  low = 0
  high = count - 1
  while high - low > 1:
    middle = (low + high) // 2
    if float(times_s[middle]) < sample_time_s:
      low = middle
    else:
      high = middle
  # A native knot uses the polynomial cell ending at that knot. This makes
  # acceleration selection deterministic despite its legitimate PCHIP jump.
  return low


def _evaluate_pchip(
  times_s: Sequence[float],
  values: Sequence[float],
  tangents: Sequence[float],
  count: int,
  sample_time_s: float,
) -> tuple[float, float, float]:
  index = _segment_index(times_s, count, sample_time_s)
  left_time = float(times_s[index])
  right_time = float(times_s[index + 1])
  width = right_time - left_time
  fraction = (sample_time_s - left_time) / width
  left_value = float(values[index])
  right_value = float(values[index + 1])
  left_tangent = float(tangents[index])
  right_tangent = float(tangents[index + 1])
  fraction_squared = fraction * fraction
  fraction_cubed = fraction_squared * fraction

  value = (
    (2.0 * fraction_cubed - 3.0 * fraction_squared + 1.0) * left_value
    + (fraction_cubed - 2.0 * fraction_squared + fraction) * width * left_tangent
    + (-2.0 * fraction_cubed + 3.0 * fraction_squared) * right_value
    + (fraction_cubed - fraction_squared) * width * right_tangent
  )
  rate = (
    (6.0 * fraction_squared - 6.0 * fraction) * left_value
    + (3.0 * fraction_squared - 4.0 * fraction + 1.0) * width * left_tangent
    + (-6.0 * fraction_squared + 6.0 * fraction) * right_value
    + (3.0 * fraction_squared - 2.0 * fraction) * width * right_tangent
  ) / width
  acceleration = (
    (12.0 * fraction - 6.0) * left_value
    + (6.0 * fraction - 4.0) * width * left_tangent
    + (-12.0 * fraction + 6.0) * right_value
    + (6.0 * fraction - 2.0) * width * right_tangent
  ) / (width * width)
  return value, rate, acceleration


def _interpolate_linear(
  times_s: Sequence[float],
  values: Sequence[float],
  count: int,
  sample_time_s: float,
) -> float:
  index = _segment_index(times_s, count, sample_time_s)
  left_time = float(times_s[index])
  right_time = float(times_s[index + 1])
  fraction = (sample_time_s - left_time) / (right_time - left_time)
  left_value = float(values[index])
  return left_value + fraction * (float(values[index + 1]) - left_value)


def _write_query_scalar_only(
  query_times_s: Sequence[float],
  query_count: int,
  scalar_curvature: float,
  measured_v_ego: float,
  output_times_s: MutableSequence[float],
  output_curvatures: MutableSequence[float],
  output_curvature_rates: MutableSequence[float],
  output_curvature_accelerations: MutableSequence[float],
  output_planned_speeds: MutableSequence[float],
  output_planned_speed_rates: MutableSequence[float],
  output_planned_speed_accelerations: MutableSequence[float],
) -> ReferenceBuildStatus:
  """Fill every requested timestamp with the deterministic degraded value."""
  if not _buffers_have_capacity(
    query_count,
    output_times_s,
    output_curvatures,
    output_curvature_rates,
    output_curvature_accelerations,
    output_planned_speeds,
    output_planned_speed_rates,
    output_planned_speed_accelerations,
  ):
    raise ValueError("reference query output buffers are too small")

  scalar = float(scalar_curvature)
  if not math.isfinite(scalar):
    scalar = 0.0
  measured_speed = float(measured_v_ego)
  for index in range(query_count):
    output_times_s[index] = float(query_times_s[index])
    output_curvatures[index] = scalar
    output_curvature_rates[index] = 0.0
    output_curvature_accelerations[index] = 0.0
    output_planned_speeds[index] = measured_speed
    output_planned_speed_rates[index] = 0.0
    output_planned_speed_accelerations[index] = 0.0
  return ReferenceBuildStatus(
    count=query_count,
    valid=False,
    scalar_only=True,
  )


def sample_reference_into(
  native_times_s: Sequence[float],
  orientation_rates_z: Sequence[float],
  velocities_x: Sequence[float],
  scalar_curvature: float,
  scalar_action_plan_s: float,
  plan_time_now_s: float,
  measured_v_ego: float,
  query_times_s: Sequence[float],
  query_count: int,
  output_times_s: MutableSequence[float],
  output_curvatures: MutableSequence[float],
  output_curvature_rates: MutableSequence[float],
  output_curvature_accelerations: MutableSequence[float],
  output_planned_speeds: MutableSequence[float],
  output_planned_speed_rates: MutableSequence[float],
  output_planned_speed_accelerations: MutableSequence[float],
  scratch_curvatures: MutableSequence[float],
  scratch_curvature_tangents: MutableSequence[float],
  scratch_speed_tangents: MutableSequence[float],
) -> ReferenceBuildStatus:
  """Sample the anchored plan at caller-supplied plan-relative timestamps.

  The caller owns all output and scratch storage, making this the live
  hot-path API. ``scalar_action_plan_s`` and ``plan_time_now_s`` are authored
  timing inputs; this function never derives either from a delay constant.

  A query, curvature anchor, or speed anchor outside native plan support
  degrades the *whole* request to scalar-only output. It does not extrapolate
  the last PCHIP cell or clamp its derivative at the endpoint. Malformed
  model samples use the same explicit degraded result. Invalid API
  configuration (non-finite anchors, speed, or query timestamps; unordered
  queries; insufficient buffers) raises ``ValueError``.
  """
  action_time = float(scalar_action_plan_s)
  plan_now = float(plan_time_now_s)
  measured_speed = float(measured_v_ego)
  if not math.isfinite(action_time) or action_time < 0.0:
    raise ValueError("scalar_action_plan_s must be finite and non-negative")
  if not math.isfinite(plan_now) or plan_now < 0.0:
    raise ValueError("plan_time_now_s must be finite and non-negative")
  if not math.isfinite(measured_speed) or measured_speed < 0.0:
    raise ValueError("measured_v_ego must be finite and non-negative")
  if query_count <= 0 or query_count > len(query_times_s):
    raise ValueError("query_count must select a non-empty query prefix")

  previous_query_time = -math.inf
  for index in range(query_count):
    query_time = float(query_times_s[index])
    if not math.isfinite(query_time) or query_time < 0.0 or query_time <= previous_query_time:
      raise ValueError(
        "reference query times must be finite, non-negative, and increasing",
      )
    previous_query_time = query_time

  if not _buffers_have_capacity(
    query_count,
    output_times_s,
    output_curvatures,
    output_curvature_rates,
    output_curvature_accelerations,
    output_planned_speeds,
    output_planned_speed_rates,
    output_planned_speed_accelerations,
  ):
    raise ValueError("reference query output buffers are too small")

  scalar = float(scalar_curvature)
  native_count = len(native_times_s)
  if len(scratch_curvatures) < native_count or len(scratch_curvature_tangents) < native_count or len(scratch_speed_tangents) < native_count:
    raise ValueError("reference query scratch buffers are too small")

  model_data_valid = native_count >= 2 and len(orientation_rates_z) == native_count and len(velocities_x) == native_count and math.isfinite(scalar)
  previous_native_time = -math.inf
  if model_data_valid:
    for index in range(native_count):
      native_time = float(native_times_s[index])
      orientation_rate = float(orientation_rates_z[index])
      planned_speed = float(velocities_x[index])
      if (
        not math.isfinite(native_time)
        or native_time < 0.0
        or native_time <= previous_native_time
        or not math.isfinite(orientation_rate)
        or not math.isfinite(planned_speed)
        or planned_speed <= 0.0
      ):
        model_data_valid = False
        break
      scratch_curvatures[index] = orientation_rate / planned_speed
      previous_native_time = native_time

  if model_data_valid:
    support_start = float(native_times_s[0])
    support_end = float(native_times_s[native_count - 1])
    model_data_valid = (
      support_start <= action_time <= support_end
      and support_start <= plan_now <= support_end
      and support_start <= float(query_times_s[0])
      and float(query_times_s[query_count - 1]) <= support_end
    )
  if not model_data_valid:
    return _write_query_scalar_only(
      query_times_s,
      query_count,
      scalar,
      measured_speed,
      output_times_s,
      output_curvatures,
      output_curvature_rates,
      output_curvature_accelerations,
      output_planned_speeds,
      output_planned_speed_rates,
      output_planned_speed_accelerations,
    )

  _fill_pchip_tangents(
    native_times_s,
    scratch_curvatures,
    native_count,
    scratch_curvature_tangents,
  )
  _fill_pchip_tangents(
    native_times_s,
    velocities_x,
    native_count,
    scratch_speed_tangents,
  )
  curvature_anchor, _, _ = _evaluate_pchip(
    native_times_s,
    scratch_curvatures,
    scratch_curvature_tangents,
    native_count,
    action_time,
  )
  speed_anchor, _, _ = _evaluate_pchip(
    native_times_s,
    velocities_x,
    scratch_speed_tangents,
    native_count,
    plan_now,
  )

  for index in range(query_count):
    query_time = float(query_times_s[index])
    curvature, curvature_rate, curvature_acceleration = _evaluate_pchip(
      native_times_s,
      scratch_curvatures,
      scratch_curvature_tangents,
      native_count,
      query_time,
    )
    planned_speed, speed_rate, speed_acceleration = _evaluate_pchip(
      native_times_s,
      velocities_x,
      scratch_speed_tangents,
      native_count,
      query_time,
    )
    anchored_speed = measured_speed + planned_speed - speed_anchor
    if not math.isfinite(anchored_speed) or anchored_speed < 0.0:
      return _write_query_scalar_only(
        query_times_s,
        query_count,
        scalar,
        measured_speed,
        output_times_s,
        output_curvatures,
        output_curvature_rates,
        output_curvature_accelerations,
        output_planned_speeds,
        output_planned_speed_rates,
        output_planned_speed_accelerations,
      )

    output_times_s[index] = query_time
    output_curvatures[index] = scalar if query_time == action_time else scalar + curvature - curvature_anchor
    output_curvature_rates[index] = curvature_rate
    output_curvature_accelerations[index] = curvature_acceleration
    output_planned_speeds[index] = measured_speed if query_time == plan_now else anchored_speed
    output_planned_speed_rates[index] = speed_rate
    output_planned_speed_accelerations[index] = speed_acceleration

  return ReferenceBuildStatus(
    count=query_count,
    valid=True,
    scalar_only=False,
  )


def sample_reference(
  native_times_s: Sequence[float],
  orientation_rates_z: Sequence[float],
  velocities_x: Sequence[float],
  scalar_curvature: float,
  scalar_action_plan_s: float,
  plan_time_now_s: float,
  measured_v_ego: float,
  query_times_s: Sequence[float],
) -> ReferenceQueryOutput:
  """Convenience wrapper for explicit reference queries."""
  query_count = len(query_times_s)
  if query_count <= 0:
    raise ValueError("reference query must contain at least one sample")
  native_count = len(native_times_s)
  output_times_s = [0.0] * query_count
  output_curvatures = [0.0] * query_count
  output_curvature_rates = [0.0] * query_count
  output_curvature_accelerations = [0.0] * query_count
  output_planned_speeds = [0.0] * query_count
  output_planned_speed_rates = [0.0] * query_count
  output_planned_speed_accelerations = [0.0] * query_count
  scratch_capacity = max(native_count, 1)
  scratch_curvatures = [0.0] * scratch_capacity
  scratch_curvature_tangents = [0.0] * scratch_capacity
  scratch_speed_tangents = [0.0] * scratch_capacity

  status = sample_reference_into(
    native_times_s,
    orientation_rates_z,
    velocities_x,
    scalar_curvature,
    scalar_action_plan_s,
    plan_time_now_s,
    measured_v_ego,
    query_times_s,
    query_count,
    output_times_s,
    output_curvatures,
    output_curvature_rates,
    output_curvature_accelerations,
    output_planned_speeds,
    output_planned_speed_rates,
    output_planned_speed_accelerations,
    scratch_curvatures,
    scratch_curvature_tangents,
    scratch_speed_tangents,
  )
  scalar = float(scalar_curvature)
  if not math.isfinite(scalar):
    scalar = 0.0
  populated = slice(0, status.count)
  return ReferenceQueryOutput(
    times_s=tuple(output_times_s[populated]),
    curvatures=tuple(output_curvatures[populated]),
    curvature_rates=tuple(output_curvature_rates[populated]),
    curvature_accelerations=tuple(
      output_curvature_accelerations[populated],
    ),
    planned_speeds=tuple(output_planned_speeds[populated]),
    planned_speed_rates=tuple(output_planned_speed_rates[populated]),
    planned_speed_accelerations=tuple(
      output_planned_speed_accelerations[populated],
    ),
    scalar_curvature=scalar,
    scalar_action_plan_s=float(scalar_action_plan_s),
    plan_time_now_s=float(plan_time_now_s),
    measured_v_ego=float(measured_v_ego),
    valid=status.valid,
    scalar_only=status.scalar_only,
  )


def compile_reference_into(
  native_times_s: Sequence[float],
  orientation_rates_z: Sequence[float],
  velocities_x: Sequence[float],
  scalar_curvature: float,
  scalar_action_plan_s: float,
  horizon_s: float,
  output_times_s: MutableSequence[float],
  output_curvatures: MutableSequence[float],
  output_curvature_rates: MutableSequence[float],
  output_curvature_accelerations: MutableSequence[float],
  output_planned_speeds: MutableSequence[float],
  scratch_curvatures: MutableSequence[float],
  scratch_tangents: MutableSequence[float],
) -> ReferenceBuildStatus:
  """Compile into caller-owned buffers and return their populated prefix.

  Invalid model data is a normal runtime condition. It deterministically
  returns one scalar-only sample with ``valid=False`` rather than raising or
  retaining an earlier plan. Invalid configuration or insufficient caller
  storage raises ``ValueError``.
  """
  action_time = float(scalar_action_plan_s)
  horizon = float(horizon_s)
  if not math.isfinite(action_time) or action_time < 0.0:
    raise ValueError("scalar_action_plan_s must be finite and non-negative")
  if not math.isfinite(horizon) or horizon < action_time:
    raise ValueError(
      "horizon_s must be finite and reach the scalar action time",
    )

  scalar = float(scalar_curvature)
  count = len(native_times_s)
  if len(scratch_curvatures) < count or len(scratch_tangents) < count:
    raise ValueError("reference scratch buffers are too small")
  model_data_valid = count >= 2 and len(orientation_rates_z) == count and len(velocities_x) == count and math.isfinite(scalar)
  previous_time = -math.inf
  if model_data_valid:
    for index in range(count):
      time_value = float(native_times_s[index])
      rate_value = float(orientation_rates_z[index])
      speed_value = float(velocities_x[index])
      if (
        not math.isfinite(time_value)
        or time_value < 0.0
        or time_value <= previous_time
        or not math.isfinite(rate_value)
        or not math.isfinite(speed_value)
        or speed_value <= 0.0
      ):
        model_data_valid = False
        break
      scratch_curvatures[index] = rate_value / speed_value
      previous_time = time_value
    model_data_valid = model_data_valid and float(native_times_s[0]) <= action_time and action_time <= float(native_times_s[count - 1])
  if not model_data_valid:
    return _write_scalar_only(
      scalar,
      action_time,
      output_times_s,
      output_curvatures,
      output_curvature_rates,
      output_curvature_accelerations,
      output_planned_speeds,
    )

  _fill_pchip_tangents(
    native_times_s,
    scratch_curvatures,
    count,
    scratch_tangents,
  )
  anchor, _, _ = _evaluate_pchip(
    native_times_s,
    scratch_curvatures,
    scratch_tangents,
    count,
    action_time,
  )

  output_capacity = count + 1
  if not _buffers_have_capacity(
    output_capacity,
    output_times_s,
    output_curvatures,
    output_curvature_rates,
    output_curvature_accelerations,
    output_planned_speeds,
  ):
    raise ValueError(
      "reference output buffers must hold native samples plus the action point",
    )

  output_count = 0
  action_inserted = False
  for index in range(count):
    time_value = float(native_times_s[index])
    if time_value > horizon:
      break
    if not action_inserted and time_value > action_time:
      value, rate, acceleration = _evaluate_pchip(
        native_times_s,
        scratch_curvatures,
        scratch_tangents,
        count,
        action_time,
      )
      output_times_s[output_count] = action_time
      output_curvatures[output_count] = scalar
      output_curvature_rates[output_count] = rate
      output_curvature_accelerations[output_count] = acceleration
      output_planned_speeds[output_count] = _interpolate_linear(
        native_times_s,
        velocities_x,
        count,
        action_time,
      )
      output_count += 1
      action_inserted = True

    value, rate, acceleration = _evaluate_pchip(
      native_times_s,
      scratch_curvatures,
      scratch_tangents,
      count,
      time_value,
    )
    output_times_s[output_count] = time_value
    output_curvatures[output_count] = scalar if time_value == action_time else scalar + value - anchor
    output_curvature_rates[output_count] = rate
    output_curvature_accelerations[output_count] = acceleration
    output_planned_speeds[output_count] = float(velocities_x[index])
    output_count += 1
    if time_value == action_time:
      action_inserted = True

  if not action_inserted:
    # The action can coincide with ``horizon`` between two native samples, in
    # which case the loop stops at the first sample after it. Emit the exact
    # authored point explicitly rather than extending to that later sample.
    _, rate, acceleration = _evaluate_pchip(
      native_times_s,
      scratch_curvatures,
      scratch_tangents,
      count,
      action_time,
    )
    output_times_s[output_count] = action_time
    output_curvatures[output_count] = scalar
    output_curvature_rates[output_count] = rate
    output_curvature_accelerations[output_count] = acceleration
    output_planned_speeds[output_count] = _interpolate_linear(
      native_times_s,
      velocities_x,
      count,
      action_time,
    )
    output_count += 1
  return ReferenceBuildStatus(
    count=output_count,
    valid=True,
    scalar_only=False,
  )


def compile_reference(
  native_times_s: Sequence[float],
  orientation_rates_z: Sequence[float],
  velocities_x: Sequence[float],
  scalar_curvature: float,
  scalar_action_plan_s: float,
  horizon_s: float,
) -> ReferenceOutput:
  """Convenience wrapper returning one immutable reference value."""
  native_count = len(native_times_s)
  capacity = max(native_count + 1, 1)
  output_times_s = [0.0] * capacity
  output_curvatures = [0.0] * capacity
  output_curvature_rates = [0.0] * capacity
  output_curvature_accelerations = [0.0] * capacity
  output_planned_speeds = [0.0] * capacity
  scratch_curvatures = [0.0] * max(native_count, 1)
  scratch_tangents = [0.0] * max(native_count, 1)

  status = compile_reference_into(
    native_times_s,
    orientation_rates_z,
    velocities_x,
    scalar_curvature,
    scalar_action_plan_s,
    horizon_s,
    output_times_s,
    output_curvatures,
    output_curvature_rates,
    output_curvature_accelerations,
    output_planned_speeds,
    scratch_curvatures,
    scratch_tangents,
  )
  scalar = float(scalar_curvature)
  if not math.isfinite(scalar):
    scalar = 0.0
  populated = slice(0, status.count)
  return ReferenceOutput(
    times_s=tuple(output_times_s[populated]),
    curvatures=tuple(output_curvatures[populated]),
    curvature_rates=tuple(output_curvature_rates[populated]),
    curvature_accelerations=tuple(
      output_curvature_accelerations[populated],
    ),
    planned_speeds=tuple(output_planned_speeds[populated]),
    scalar_curvature=scalar,
    scalar_action_plan_s=float(scalar_action_plan_s),
    valid=status.valid,
    scalar_only=status.scalar_only,
  )
