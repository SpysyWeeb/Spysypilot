"""Model-path compilation and rack-reference governance."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from openpilot.selfdrive.controls.lib.drive_helpers import MIN_SPEED
from openpilot.selfdrive.controls.lib.rack_trajectory_contracts import PathTarget, RackTarget
from openpilot.selfdrive.controls.lib.rack_trajectory_planner import JerkLimitedRackPlanner

REFERENCE_REVERSAL_DISTANCE_DEG = 1.0
REFERENCE_REVERSAL_RC_S = .12
REFERENCE_MAX_RATE_DEG_S = 5.0


HORIZON_S = 2.0
HORIZON_STEP_S = .25
HORIZON_OFFSETS_S = tuple(index * HORIZON_STEP_S for index in range(round(HORIZON_S / HORIZON_STEP_S) + 1))


def model_path_targets(
  *,
  native_times_s: Sequence[float],
  orientation_rates_z: Sequence[float],
  velocities_x: Sequence[float],
  scalar_curvature: float,
  scalar_action_plan_s: float,
  plan_time_now_s: float,
  measured_v_ego: float,
  query_times_s: Sequence[float],
  vehicle_model,
  roll_rad: float,
  angle_offset_deg: float,
) -> tuple[PathTarget, ...]:
  scalar = float(scalar_curvature)
  measured_speed = float(measured_v_ego)
  queries = tuple(float(query) for query in query_times_s)
  if not queries or not math.isfinite(scalar) or not math.isfinite(measured_speed) or measured_speed < 0.0:
    raise ValueError("invalid scalar path target")

  count = len(native_times_s)
  valid = count >= 2 and len(orientation_rates_z) == count and len(velocities_x) == count
  times: list[float] = []
  curvatures: list[float] = []
  speeds: list[float] = []
  previous_time = -math.inf
  if valid:
    for native_time, orientation_rate, planned_speed in zip(
      native_times_s, orientation_rates_z, velocities_x, strict=True,
    ):
      time = float(native_time)
      speed = float(planned_speed)
      rate = float(orientation_rate)
      if not all(math.isfinite(value) for value in (time, speed, rate)) or time < 0.0 or time <= previous_time:
        valid = False
        break
      times.append(time)
      # a plan that stops inside the horizon still covers it. Below MIN_SPEED the yaw rate/speed
      # ratio is ill-conditioned, so hold the last well-conditioned curvature: the wheel stays put.
      if speed >= MIN_SPEED or not curvatures:
        curvatures.append(rate / max(speed, MIN_SPEED))
      else:
        curvatures.append(curvatures[-1])
      speeds.append(speed)
      previous_time = time

  if not valid or len(times) < 2:
    raise ValueError("invalid model path")
  if not all(times[0] <= float(query) <= times[-1] for query in (
    scalar_action_plan_s, plan_time_now_s, *queries,
  )):
    raise ValueError("model path does not cover requested timestamps")

  def angle_at(query: float) -> tuple[float, float, float]:
    plan_curvature = float(np.interp(query, times, curvatures))
    anchor_curvature = float(np.interp(float(scalar_action_plan_s), times, curvatures))
    curvature = scalar + plan_curvature - anchor_curvature
    speed = max(MIN_SPEED, measured_speed + float(np.interp(query, times, speeds)) - float(np.interp(float(plan_time_now_s), times, speeds)))
    angle = math.degrees(vehicle_model.get_steer_from_curvature(-curvature, speed, roll_rad)) + angle_offset_deg
    return curvature, speed, angle

  targets: list[PathTarget] = []
  for query in queries:
    curvature, speed, angle = angle_at(query)
    before = max(times[0], query - .05)
    after = min(times[-1], query + .05)
    rate = (angle_at(after)[2] - angle_at(before)[2]) / (after - before) if after > before else 0.0
    if not all(math.isfinite(value) for value in (curvature, speed, angle, rate)):
      raise ValueError("non-finite path target")
    targets.append(PathTarget(curvature, speed, angle, rate))
  return tuple(targets)


def model_path_target(
  *,
  native_times_s: Sequence[float],
  orientation_rates_z: Sequence[float],
  velocities_x: Sequence[float],
  scalar_curvature: float,
  scalar_action_plan_s: float,
  plan_time_now_s: float,
  measured_v_ego: float,
  query_time_s: float,
  vehicle_model,
  roll_rad: float,
  angle_offset_deg: float,
) -> PathTarget:
  return model_path_targets(
    native_times_s=native_times_s,
    orientation_rates_z=orientation_rates_z,
    velocities_x=velocities_x,
    scalar_curvature=scalar_curvature,
    scalar_action_plan_s=scalar_action_plan_s,
    plan_time_now_s=plan_time_now_s,
    measured_v_ego=measured_v_ego,
    query_times_s=(query_time_s,),
    vehicle_model=vehicle_model,
    roll_rad=roll_rad,
    angle_offset_deg=angle_offset_deg,
  )[0]


class RackReferenceGovernor:
  """Smooth short, small rack-reference reversals before trajectory planning."""

  def __init__(self) -> None:
    self.accepted: RackTarget | None = None
    self.last_model_timestamp_ns: int | None = None
    self.last_replan_position_deg: float | None = None
    self.direction = 0
    self.active = False
    self.limited = False

  def reset(self) -> None:
    self.accepted = None
    self.last_model_timestamp_ns = None
    self.last_replan_position_deg = None
    self.direction = 0
    self.active = False
    self.limited = False

  @staticmethod
  def _sign(value: float) -> int:
    return 1 if value > 1e-9 else -1 if value < -1e-9 else 0

  def _accept(self, target: RackTarget) -> RackTarget:
    self.accepted = target
    self.active = False
    self.limited = False
    return target

  def update(self, target: RackTarget, planner: JerkLimitedRackPlanner,
             neutral_position_deg: float, model_timestamp_ns: int, dt: float,
             bypass: bool = False) -> RackTarget:
    timestamp_ns = int(model_timestamp_ns)
    if self.accepted is None:
      self.last_model_timestamp_ns = timestamp_ns
      self.last_replan_position_deg = target.position_deg
      return self._accept(target)

    assert self.last_model_timestamp_ns is not None and self.last_replan_position_deg is not None
    new_model = timestamp_ns != self.last_model_timestamp_ns
    raw_change_direction = self._sign(target.position_deg - self.last_replan_position_deg) if new_model else 0
    if bypass:
      self.last_model_timestamp_ns = timestamp_ns
      self.last_replan_position_deg = target.position_deg
      self.direction = 0
      return self._accept(target)
    plan_error = abs(target.position_deg - planner.position_deg)
    filter_error = abs(target.position_deg - self.accepted.position_deg)
    coherent_motion = (
      plan_error >= REFERENCE_REVERSAL_DISTANCE_DEG
      or filter_error >= REFERENCE_REVERSAL_DISTANCE_DEG
      or abs(target.rate_deg_s) >= REFERENCE_MAX_RATE_DEG_S
    )
    reversal = new_model and raw_change_direction != 0 and self.direction != 0 and raw_change_direction != self.direction
    if coherent_motion:
      if new_model:
        self.last_model_timestamp_ns = timestamp_ns
        self.last_replan_position_deg = target.position_deg
        if raw_change_direction:
          self.direction = raw_change_direction
      return self._accept(target)
    if reversal:
      self.active = True
    if new_model:
      self.last_model_timestamp_ns = timestamp_ns
      self.last_replan_position_deg = target.position_deg
      if raw_change_direction:
        self.direction = raw_change_direction
    if not self.active:
      return self._accept(target)

    alpha = dt / (REFERENCE_REVERSAL_RC_S + dt)
    self.accepted = RackTarget(
      self.accepted.position_deg + alpha * (target.position_deg - self.accepted.position_deg),
      self.accepted.rate_deg_s + alpha * (target.rate_deg_s - self.accepted.rate_deg_s),
    )
    self.limited = (
      abs(self.accepted.position_deg - target.position_deg) > 1e-9
      or abs(self.accepted.rate_deg_s - target.rate_deg_s) > 1e-9
    )
    return self.accepted
