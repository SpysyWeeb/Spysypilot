"""Stateless BLaTv2 model-reference construction and inverse statics."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.blatv2.plant import PlantParams

HORIZON_MARGIN_SECONDS = 0.2
MODEL_ACTION_OFFSET = 1.5 * DT_MDL


def _as_finite_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
  result = tuple(float(value) for value in values)
  if not result or not all(math.isfinite(value) for value in result):
    raise ValueError(f"{name} must be non-empty and finite")
  return result


def interpolate(times: Sequence[float], values: Sequence[float], sample_time: float) -> float:
  xs = tuple(float(value) for value in times)
  ys = tuple(float(value) for value in values)
  if len(xs) != len(ys) or not xs:
    raise ValueError("interpolation inputs must be non-empty and equal length")
  if sample_time <= xs[0]:
    return ys[0]
  if sample_time >= xs[-1]:
    return ys[-1]

  low = 0
  high = len(xs) - 1
  while high - low > 1:
    middle = (low + high) // 2
    if xs[middle] <= sample_time:
      low = middle
    else:
      high = middle
  fraction = (sample_time - xs[low]) / (xs[high] - xs[low])
  return ys[low] + fraction * (ys[high] - ys[low])


def interpolate_buffer(times: np.ndarray, values: np.ndarray, count: int, sample_time: float) -> float:
  """Allocation-free interpolation over the populated prefix of two buffers."""
  if count <= 0 or count > len(times) or count > len(values):
    raise ValueError("buffer interpolation count is outside buffer bounds")
  if sample_time <= times[0]:
    return float(values[0])
  if sample_time >= times[count - 1]:
    return float(values[count - 1])

  low = 0
  high = count - 1
  while high - low > 1:
    middle = (low + high) // 2
    if times[middle] <= sample_time:
      low = middle
    else:
      high = middle
  fraction = (sample_time - float(times[low])) / (float(times[high]) - float(times[low]))
  return float(values[low]) + fraction * (float(values[high]) - float(values[low]))


def build_reference_into(
  scalar_curvature: float,
  plan_times: np.ndarray,
  plan_curvatures: np.ndarray,
  plan_count: int,
  action_time_offset: float,
  horizon_seconds: float,
  output_times: np.ndarray,
  output_curvatures: np.ndarray,
) -> int:
  """Allocation-free form of :func:`build_reference` for the shadow hot path."""
  scalar = float(scalar_curvature)
  action_offset = float(action_time_offset)
  horizon_value = float(horizon_seconds)
  if not (math.isfinite(scalar) and math.isfinite(action_offset) and math.isfinite(horizon_value)):
    raise ValueError("reference scalars must be finite")
  if horizon_value < 0.0:
    raise ValueError("horizon must be non-negative")
  if plan_count <= 0 or plan_count > len(plan_times) or plan_count > len(plan_curvatures):
    raise ValueError("plan count is outside buffer bounds")

  anchor = interpolate_buffer(plan_times, plan_curvatures, plan_count, action_offset)
  output_count = 0
  previous_time = -math.inf
  for index in range(plan_count):
    time_value = float(plan_times[index])
    if time_value <= previous_time:
      raise ValueError("plan times must be strictly increasing")
    previous_time = time_value
    if 0.0 <= time_value <= horizon_value:
      if output_count >= len(output_times) or output_count >= len(output_curvatures):
        raise ValueError("reference output buffer is too small")
      output_times[output_count] = time_value
      output_curvatures[output_count] = scalar + (float(plan_curvatures[index]) - anchor)
      output_count += 1
  if output_count == 0:
    raise ValueError("plan has no samples inside the requested horizon")
  return output_count


def build_reference(
  scalar_curvature: float,
  plan_times: Sequence[float],
  plan_curvatures: Sequence[float],
  action_time_offset: float,
  horizon: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
  """Return scalar + (plan(t) - plan(action offset)); deliberately stateless."""
  scalar = float(scalar_curvature)
  times = _as_finite_tuple(plan_times, "plan_times")
  curvatures = _as_finite_tuple(plan_curvatures, "plan_curvatures")
  action_offset = float(action_time_offset)
  horizon_seconds = float(horizon)
  if not math.isfinite(scalar) or not math.isfinite(action_offset) or not math.isfinite(horizon_seconds):
    raise ValueError("reference scalars must be finite")
  if len(times) != len(curvatures):
    raise ValueError("plan times and curvatures must have equal length")
  if any(right <= left for left, right in zip(times, times[1:], strict=False)):
    raise ValueError("plan times must be strictly increasing")
  if horizon_seconds < 0.0:
    raise ValueError("horizon must be non-negative")

  anchor = interpolate(times, curvatures, action_offset)
  selected = tuple((time, curvature) for time, curvature in zip(times, curvatures, strict=True) if 0.0 <= time <= horizon_seconds)
  if not selected:
    raise ValueError("plan has no samples inside the requested horizon")
  output_times = tuple(item[0] for item in selected)
  output_curvatures = tuple(scalar + (item[1] - anchor) for item in selected)
  return output_times, output_curvatures


def horizon(params: PlantParams, actuation_delay: float | None = None) -> float:
  frame_seconds = DT_CTRL * params.steer_step
  full_sweep = (params.steer_max / params.delta_down + params.steer_max / params.delta_up) * frame_seconds
  delay = params.actuation_delay if actuation_delay is None else float(actuation_delay)
  if not math.isfinite(delay) or delay < 0.0:
    raise ValueError("actuation delay must be finite and non-negative")
  return float(full_sweep + delay + HORIZON_MARGIN_SECONDS)


def torque_demand(ref_curvature: float, v_ego: float, roll: float, torque_params: Any) -> float:
  curvature = float(ref_curvature)
  speed = float(v_ego)
  road_roll = float(roll)
  factor = float(torque_params.latAccelFactor)
  offset = float(torque_params.latAccelOffset)
  friction = float(torque_params.friction)
  if not (
    math.isfinite(curvature)
    and math.isfinite(speed)
    and math.isfinite(road_roll)
    and math.isfinite(factor)
    and math.isfinite(offset)
    and math.isfinite(friction)
  ):
    raise ValueError("inverse statics inputs must be finite")
  if factor <= 0.0:
    raise ValueError("latAccelFactor must be positive")

  geometric_lateral_accel = curvature * speed * speed
  gravity_adjusted = geometric_lateral_accel - road_roll * ACCELERATION_DUE_TO_GRAVITY - offset
  friction_torque = 0.0 if geometric_lateral_accel == 0.0 else math.copysign(friction, geometric_lateral_accel)
  demand = -(gravity_adjusted / factor + friction_torque)
  return min(max(demand, -1.0), 1.0)


def model_action_time(lateral_delay: float) -> float:
  delay = float(lateral_delay)
  if not math.isfinite(delay) or delay < 0.0:
    raise ValueError("lateral delay must be finite and non-negative")
  return delay + MODEL_ACTION_OFFSET


def plan_curvatures_from_model(model: Any, scalar_curvature: float) -> tuple[tuple[float, ...], tuple[float, ...], bool]:
  """Read curvature as orientationRate.z / velocity.x on the native grid.

  An exactly-zero or nonfinite speed/rate sample falls back to the scalar and
  makes the returned validity false, as pinned for shadow acceptance.
  """
  rates = tuple(float(value) for value in model.orientationRate.z)
  speeds = tuple(float(value) for value in model.velocity.x)
  native_times = tuple(float(value) for value in model.orientationRate.t)
  if len(native_times) != len(rates):
    native_times = tuple(float(value) for value in model.velocity.t)
  if len(native_times) != len(rates) or len(rates) != len(speeds) or not rates:
    raise ValueError("model orientation-rate, velocity, and native time grids must align")

  scalar = float(scalar_curvature)
  valid = math.isfinite(scalar)
  curvatures: list[float] = []
  for rate, speed in zip(rates, speeds, strict=True):
    if not math.isfinite(rate) or not math.isfinite(speed) or speed == 0.0:
      curvatures.append(scalar)
      valid = False
    else:
      curvature = rate / speed
      if math.isfinite(curvature):
        curvatures.append(curvature)
      else:
        curvatures.append(scalar)
        valid = False
  return native_times, tuple(curvatures), valid


def plan_curvatures_from_model_into(
  model: Any,
  scalar_curvature: float,
  times: np.ndarray,
  curvatures: np.ndarray,
) -> int:
  """Allocation-free form of :func:`plan_curvatures_from_model`.

  The magnitude of the return value is the populated count. A negative count
  marks an invalid plan, avoiding a per-frame tuple allocation in the daemon.
  """
  rates = model.orientationRate.z
  speeds = model.velocity.x
  native_times = model.orientationRate.t
  count = len(rates)
  if len(native_times) != count:
    native_times = model.velocity.t
  if count == 0 or len(native_times) != count or len(speeds) != count:
    raise ValueError("model orientation-rate, velocity, and native time grids must align")
  if count > len(times) or count > len(curvatures):
    raise ValueError("model plan exceeds preallocated shadow buffers")

  scalar = float(scalar_curvature)
  valid = math.isfinite(scalar)
  for index in range(count):
    time_value = float(native_times[index])
    rate = float(rates[index])
    speed = float(speeds[index])
    times[index] = time_value
    if not math.isfinite(time_value) or not math.isfinite(rate) or not math.isfinite(speed) or speed == 0.0:
      curvatures[index] = scalar
      valid = False
    else:
      curvature = rate / speed
      if math.isfinite(curvature):
        curvatures[index] = curvature
      else:
        curvatures[index] = scalar
        valid = False
  return count if valid else -count
