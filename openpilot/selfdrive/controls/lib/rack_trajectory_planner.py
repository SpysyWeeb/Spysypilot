"""Jerk-limited rack motion planner."""
from __future__ import annotations

import math
from collections.abc import Sequence

from openpilot.selfdrive.controls.lib.rack_trajectory_contracts import MotionLimits, RackPlan, RackTarget


def _clip(value: float, limit: float) -> float:
  return max(-limit, min(limit, value))


def _rate_viable_acceleration(headroom: float, jerk: float, dt: float) -> float:
  if headroom <= 0.0:
    return 0.0
  jerk_step = jerk * dt
  return -jerk_step + math.sqrt(jerk_step * jerk_step + 2.0 * jerk * headroom)


def _required_rate_headroom(rate: float, acceleration: float, jerk: float, dt: float) -> float:
  outward_acceleration = abs(acceleration) if rate * acceleration > 0.0 else 0.0
  return outward_acceleration * dt + outward_acceleration * outward_acceleration / (2.0 * jerk)


class JerkLimitedRackPlanner:
  def __init__(self, position_deg: float, rate_deg_s: float = 0.0) -> None:
    self.position_deg = float(position_deg)
    self.rate_deg_s = float(rate_deg_s)
    self.acceleration_deg_s2 = 0.0

  def update(self, target: RackTarget, limits: MotionLimits, dt: float,
             desired_acceleration_override: float | None = None) -> RackPlan:
    natural_frequency = 2.0 / limits.response_time_s
    desired_acceleration_raw = float(desired_acceleration_override) if desired_acceleration_override is not None else (
      natural_frequency * natural_frequency * (target.position_deg - self.position_deg)
      + 2.0 * natural_frequency * (target.rate_deg_s - self.rate_deg_s)
    )
    desired_acceleration = _clip(desired_acceleration_raw, limits.max_acceleration_deg_s2)
    jerk_step = limits.max_jerk_deg_s3 * dt
    jerk_lower = self.acceleration_deg_s2 - jerk_step
    jerk_upper = self.acceleration_deg_s2 + jerk_step
    rate_lower = -_rate_viable_acceleration(limits.max_rate_deg_s + self.rate_deg_s, limits.max_jerk_deg_s3, dt)
    rate_upper = _rate_viable_acceleration(limits.max_rate_deg_s - self.rate_deg_s, limits.max_jerk_deg_s3, dt)
    lower = max(-limits.max_acceleration_deg_s2, jerk_lower, rate_lower)
    upper = min(limits.max_acceleration_deg_s2, jerk_upper, rate_upper)
    if lower > upper + 1e-9:
      raise ValueError("rack planner outside motion envelope")
    acceleration = max(lower, min(upper, desired_acceleration))
    rate = self.rate_deg_s + acceleration * dt
    position = self.position_deg + .5 * (self.rate_deg_s + rate) * dt
    self.position_deg, self.rate_deg_s, self.acceleration_deg_s2 = position, rate, acceleration
    return RackPlan(
      position, rate, acceleration,
      desired_acceleration < rate_lower or desired_acceleration > rate_upper,
      desired_acceleration != desired_acceleration_raw,
      desired_acceleration < jerk_lower or desired_acceleration > jerk_upper,
    )


def horizon_desired_acceleration(
  planner: JerkLimitedRackPlanner,
  timed_targets: Sequence[tuple[float, RackTarget]],
) -> float:
  """Fit one current acceleration to the model-authored future rack states."""
  weighted_acceleration = 0.0
  weight_total = 0.0
  previous_time = 0.0
  for time_s, target in timed_targets:
    time = float(time_s)
    values = (time, target.position_deg, target.rate_deg_s)
    if not all(math.isfinite(value) for value in values) or time <= previous_time:
      raise ValueError("invalid rack horizon")
    # Initial acceleration of the cubic joining current position/rate to this
    # model target. Near knots carry more weight but later path phases still
    # influence preparation; the live planner applies the physical limits.
    acceleration = (
      6.0 * (target.position_deg - planner.position_deg) / time ** 2
      - (4.0 * planner.rate_deg_s + 2.0 * target.rate_deg_s) / time
    )
    weight = 1.0 / time
    weighted_acceleration += weight * acceleration
    weight_total += weight
    previous_time = time
  if weight_total == 0.0:
    raise ValueError("empty rack horizon")
  result = weighted_acceleration / weight_total
  if not math.isfinite(result):
    raise ValueError("non-finite rack horizon")
  return result
