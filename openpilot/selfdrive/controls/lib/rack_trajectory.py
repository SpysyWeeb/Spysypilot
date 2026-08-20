"""Minimal Palisade rack-trajectory execution controller."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from opendbc.car.lateral import FRICTION_THRESHOLD, get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_CURVATURE, MAX_LATERAL_ACCEL_NO_ROLL, MIN_SPEED

DT = .01
PREVIEW_S = .5
RESPONSE_TIME_S = .4
RATE_HORIZON_S = .1
MAX_FEEDBACK_TORQUE = .35
MAX_TURN_IN_FEEDBACK_TORQUE = .7
MAX_DRIVER_ASSIST_TORQUE = .35
MODEL_ACTION_OFFSET_S = 1.5 * DT_MDL  # modeld frame delay + midpoint action delay
REFERENCE_REVERSAL_DISTANCE_DEG = 1.0
REFERENCE_REVERSAL_PERSISTENCE_S = .15
REFERENCE_REVERSAL_RC_S = .3
REFERENCE_PERSISTENT_RC_S = .05
REFERENCE_MAX_RATE_DEG_S = 5.0

STATUS_INACTIVE = 0
STATUS_ACTIVE = 1
STATUS_DRIVER_OVERRIDE = 2
STATUS_NO_MODEL = 3
STATUS_INVALID_VEHICLE_STATE = 4
STATUS_STALE_MODEL = 5
STATUS_INVALID_ACTION_TIME = 6
STATUS_INVALID_PATH = 7
STATUS_INVALID_OUTPUT = 8
STATUS_INVALID_PLANNER_STATE = 9
STATUS_MEASURED_OUT_OF_BOUNDS = 10
STATUS_PLANNED_OUT_OF_BOUNDS = 11

# Coherent-motion corpus: p99 rate/acceleration and p95 jerk by speed.
_SPEED_PROFILE_MPH = np.asarray([7.25, 12.5, 17.5, 22.5, 30.0, 45.0])
_RATE_PROFILE_DEG_S = np.asarray([315.848, 289.402, 128.744, 77.716, 35.104, 21.057])
_ACCEL_PROFILE_DEG_S2 = np.asarray([891.046, 827.645, 561.569, 334.476, 172.041, 97.740])
_JERK_PROFILE_DEG_S3 = np.asarray([4567.435, 5115.079, 3851.530, 2329.626, 1318.842, 743.710])

_STOCK_KP_SPEEDS = np.asarray([1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 30.0])
_STOCK_KP = np.asarray([250.0, 120.0, 65.0, 30.0, 11.5, 5.5, 3.5, 2.0, .8])
_LOW_SPEED_KP_END = float(np.float32(15.0 * 0.44704))
_LOW_SPEED_KP_SPEEDS = np.asarray([2.0, 3.0, 5.0, _LOW_SPEED_KP_END])
_LOW_SPEED_KP = np.asarray([65.0, 10.0, 10.0, np.interp(_LOW_SPEED_KP_END, _STOCK_KP_SPEEDS, _STOCK_KP)])


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


@dataclass(frozen=True, slots=True)
class PathTarget:
  curvature: float
  speed_mps: float
  angle_deg: float
  rate_deg_s: float


@dataclass(frozen=True, slots=True)
class MotionLimits:
  max_rate_deg_s: float
  max_acceleration_deg_s2: float
  max_jerk_deg_s3: float
  response_time_s: float = RESPONSE_TIME_S


@dataclass(frozen=True, slots=True)
class RackTarget:
  position_deg: float
  rate_deg_s: float


@dataclass(frozen=True, slots=True)
class RackPlan:
  position_deg: float
  rate_deg_s: float
  acceleration_deg_s2: float
  rate_limited: bool
  acceleration_limited: bool
  jerk_limited: bool


@dataclass(frozen=True, slots=True)
class RackTrajectoryOutput:
  torque: float
  target_curvature: float
  target_angle_deg: float
  target_rate_deg_s: float
  planned_angle_deg: float
  planned_rate_deg_s: float
  planned_acceleration_deg_s2: float
  measured_rate_deg_s: float
  lateral_accel_error: float
  rate_error_deg_s: float
  position_feedback_torque: float
  rate_feedback_torque: float
  feedforward_torque: float
  desired_lateral_accel: float
  actual_lateral_accel: float
  desired_lateral_jerk: float
  feedback_torque: float
  feedback_limited: bool
  motion_limited: bool
  torque_limited: bool
  rate_limit_deg_s: float
  acceleration_limit_deg_s2: float
  jerk_limit_deg_s3: float
  profile_transition: bool
  path_limited: bool
  infeasible: bool
  saturated: bool


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
  scalar = float(scalar_curvature)
  measured_speed = float(measured_v_ego)
  if not math.isfinite(scalar) or not math.isfinite(measured_speed) or measured_speed < 0.0:
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
      if speed <= 0.0:
        break
      times.append(time)
      curvatures.append(rate / speed)
      speeds.append(speed)
      previous_time = time

  if not valid or len(times) < 2:
    raise ValueError("invalid model path")
  if not all(times[0] <= float(query) <= times[-1] for query in (
    scalar_action_plan_s, plan_time_now_s, query_time_s,
  )):
    raise ValueError("model path does not cover requested timestamps")

  def angle_at(query: float) -> tuple[float, float, float]:
    plan_curvature = float(np.interp(query, times, curvatures))
    anchor_curvature = float(np.interp(float(scalar_action_plan_s), times, curvatures))
    curvature = scalar + plan_curvature - anchor_curvature
    speed = max(.1, measured_speed + float(np.interp(query, times, speeds)) - float(np.interp(float(plan_time_now_s), times, speeds)))
    angle = math.degrees(vehicle_model.get_steer_from_curvature(-curvature, speed, roll_rad)) + angle_offset_deg
    return curvature, speed, angle

  query = float(query_time_s)
  curvature, speed, angle = angle_at(query)
  before = max(times[0], query - .05)
  after = min(times[-1], query + .05)
  rate = (angle_at(after)[2] - angle_at(before)[2]) / (after - before) if after > before else 0.0
  if not all(math.isfinite(value) for value in (curvature, speed, angle, rate)):
    raise ValueError("non-finite path target")
  return PathTarget(curvature, speed, angle, rate)


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


class RackReferenceGovernor:
  """Smooth short, small rack-reference reversals before trajectory planning."""

  def __init__(self) -> None:
    self.accepted: RackTarget | None = None
    self.last_model_timestamp_ns: int | None = None
    self.last_replan_position_deg: float | None = None
    self.direction = 0
    self.reversal_start_model_ns: int | None = None
    self.reversal_s = 0.0
    self.active = False
    self.limited = False

  def reset(self) -> None:
    self.accepted = None
    self.last_model_timestamp_ns = None
    self.last_replan_position_deg = None
    self.direction = 0
    self.reversal_start_model_ns = None
    self.reversal_s = 0.0
    self.active = False
    self.limited = False

  @staticmethod
  def _sign(value: float) -> int:
    return 1 if value > 1e-9 else -1 if value < -1e-9 else 0

  def _accept(self, target: RackTarget) -> RackTarget:
    self.accepted = target
    self.reversal_start_model_ns = None
    self.reversal_s = 0.0
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
    crosses_neutral = (
      (self.accepted.position_deg - neutral_position_deg) * (target.position_deg - neutral_position_deg) <= 0.0
    )
    coherent_motion = (
      plan_error >= REFERENCE_REVERSAL_DISTANCE_DEG
      or filter_error >= REFERENCE_REVERSAL_DISTANCE_DEG
      or abs(target.rate_deg_s) >= REFERENCE_MAX_RATE_DEG_S
      or crosses_neutral
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
      self.reversal_start_model_ns = timestamp_ns
      self.reversal_s = 0.0
    if new_model:
      self.last_model_timestamp_ns = timestamp_ns
      self.last_replan_position_deg = target.position_deg
      if raw_change_direction:
        self.direction = raw_change_direction
    if not self.active:
      return self._accept(target)

    assert self.reversal_start_model_ns is not None
    self.reversal_s = max(0.0, (timestamp_ns - self.reversal_start_model_ns) * 1e-9)
    rc = REFERENCE_PERSISTENT_RC_S if self.reversal_s >= REFERENCE_REVERSAL_PERSISTENCE_S else REFERENCE_REVERSAL_RC_S
    alpha = dt / (rc + dt)
    self.accepted = RackTarget(
      self.accepted.position_deg + alpha * (target.position_deg - self.accepted.position_deg),
      self.accepted.rate_deg_s + alpha * (target.rate_deg_s - self.accepted.rate_deg_s),
    )
    self.limited = (
      abs(self.accepted.position_deg - target.position_deg) > 1e-9
      or abs(self.accepted.rate_deg_s - target.rate_deg_s) > 1e-9
    )
    return self.accepted


class PalisadeRackTrajectoryController:
  def __init__(self, dt: float = DT) -> None:
    self.dt = dt
    self.model = None
    self.state_mono_ns = 0
    self.planner: JerkLimitedRackPlanner | None = None
    self.transition_rate_limit: float | None = None
    self.transition_acceleration_limit: float | None = None
    self.previous_planned_lateral_accel: float | None = None
    self.previous_angle_deg: float | None = None
    self.rack_direction = 0
    self.raw_signed_episode = False
    self.driver_override_resume = False
    self.status = STATUS_INACTIVE
    self.jerk_filter = FirstOrderFilter(0.0, 1.0 / (2.0 * math.pi * 1.2), dt)
    self.reference_governor = RackReferenceGovernor()

  def set_model(self, model, state_mono_ns: int) -> None:
    self.model = model
    self.state_mono_ns = int(state_mono_ns)

  def reset(self) -> None:
    self.planner = None
    self.transition_rate_limit = None
    self.transition_acceleration_limit = None
    self.previous_planned_lateral_accel = None
    self.previous_angle_deg = None
    self.rack_direction = 0
    self.raw_signed_episode = False
    self.driver_override_resume = False
    self.status = STATUS_INACTIVE
    self.jerk_filter.x = 0.0
    self.reference_governor.reset()

  def _measured_rate(self, angle_deg: float, raw_rate_deg_s: float) -> tuple[float, bool]:
    magnitude = abs(raw_rate_deg_s)
    if magnitude == 0.0:
      self.rack_direction = 0
      self.raw_signed_episode = False
      rate, valid = 0.0, True
    elif raw_rate_deg_s < 0.0:
      self.rack_direction = -1
      self.raw_signed_episode = True
      rate, valid = -magnitude, True
    elif self.raw_signed_episode:
      self.rack_direction = 1
      rate, valid = magnitude, True
    elif self.previous_angle_deg is not None and angle_deg != self.previous_angle_deg:
      self.rack_direction = 1 if angle_deg > self.previous_angle_deg else -1
      rate, valid = self.rack_direction * magnitude, True
    elif self.rack_direction:
      rate, valid = self.rack_direction * magnitude, True
    else:
      rate, valid = 0.0, False
    self.previous_angle_deg = angle_deg
    return rate, valid

  @staticmethod
  def _limits(speed_mps: float) -> MotionLimits:
    speed_mph = speed_mps * 2.2369362920544
    return MotionLimits(
      float(np.interp(speed_mph, _SPEED_PROFILE_MPH, _RATE_PROFILE_DEG_S)),
      float(np.interp(speed_mph, _SPEED_PROFILE_MPH, _ACCEL_PROFILE_DEG_S2)),
      float(np.interp(speed_mph, _SPEED_PROFILE_MPH, _JERK_PROFILE_DEG_S3)),
    )

  def _motion_limits(self, profile: MotionLimits) -> tuple[MotionLimits, bool]:
    assert self.planner is not None
    rate_headroom = _required_rate_headroom(
      self.planner.rate_deg_s, self.planner.acceleration_deg_s2, profile.max_jerk_deg_s3, self.dt,
    )
    required_rate_limit = max(profile.max_rate_deg_s, abs(self.planner.rate_deg_s) + rate_headroom)
    required_acceleration_limit = max(profile.max_acceleration_deg_s2, abs(self.planner.acceleration_deg_s2))
    transition = (
      required_rate_limit > profile.max_rate_deg_s + 1e-6
      or required_acceleration_limit > profile.max_acceleration_deg_s2 + 1e-6
    )
    if not transition:
      self.transition_rate_limit = None
      self.transition_acceleration_limit = None
      return profile, False
    if self.transition_rate_limit is None or self.transition_acceleration_limit is None:
      self.transition_rate_limit = required_rate_limit
      self.transition_acceleration_limit = required_acceleration_limit
    else:
      self.transition_rate_limit = max(profile.max_rate_deg_s, min(self.transition_rate_limit, required_rate_limit))
      self.transition_acceleration_limit = max(
        profile.max_acceleration_deg_s2, min(self.transition_acceleration_limit, required_acceleration_limit),
      )
    return MotionLimits(
      self.transition_rate_limit, self.transition_acceleration_limit, profile.max_jerk_deg_s3,
    ), True

  def _recovery_acceleration(self, profile: MotionLimits, transition: bool) -> float | None:
    if not transition:
      return None
    assert self.planner is not None
    rate = self.planner.rate_deg_s
    acceleration = self.planner.acceleration_deg_s2
    if abs(rate) > profile.max_rate_deg_s + 1e-6 or rate * acceleration > 0.0:
      return -math.copysign(profile.max_acceleration_deg_s2, rate) if rate != 0.0 else 0.0
    if abs(acceleration) > profile.max_acceleration_deg_s2:
      return _clip(acceleration, profile.max_acceleration_deg_s2)
    return None

  @staticmethod
  def _feedback_gain(speed_mps: float) -> float:
    return float(
      np.interp(speed_mps, _LOW_SPEED_KP_SPEEDS, _LOW_SPEED_KP)
      if _LOW_SPEED_KP_SPEEDS[0] < speed_mps < _LOW_SPEED_KP_SPEEDS[-1]
      else np.interp(speed_mps, _STOCK_KP_SPEEDS, _STOCK_KP)
    )

  def update(self, active: bool, CS, VM, params, torque_params, torque_from_lateral_accel: Callable[[float, object], float],
             lat_delay: float, desired_curvature: float) -> RackTrajectoryOutput | None:
    if not active:
      self.reset()
      return None
    if self.model is None:
      self.reset()
      self.status = STATUS_NO_MODEL
      return None
    if not all(math.isfinite(float(value)) for value in (
      CS.vEgo, CS.steeringAngleDeg, CS.steeringRateDeg, CS.steeringTorque,
      params.roll, params.angleOffsetDeg, lat_delay, desired_curvature,
    )):
      self.reset()
      self.status = STATUS_INVALID_VEHICLE_STATE
      return None
    model_age_s = (self.state_mono_ns - int(self.model.timestampEof)) * 1e-9
    if not 0.0 <= model_age_s <= .2:
      self.reset()
      self.status = STATUS_STALE_MODEL
      return None

    try:
      scalar_action_plan_s = float(self.model.action.desiredCurvatureTime)
    except AttributeError:
      scalar_action_plan_s = float(lat_delay) + MODEL_ACTION_OFFSET_S
    if not math.isfinite(scalar_action_plan_s) or scalar_action_plan_s <= 0.0:
      scalar_action_plan_s = float(lat_delay) + MODEL_ACTION_OFFSET_S
    if scalar_action_plan_s <= 0.0:
      self.reset()
      self.status = STATUS_INVALID_ACTION_TIME
      return None

    try:
      target = model_path_target(
        native_times_s=self.model.orientationRate.t,
        orientation_rates_z=self.model.orientationRate.z,
        velocities_x=self.model.velocity.x,
        scalar_curvature=float(desired_curvature),
        scalar_action_plan_s=scalar_action_plan_s,
        plan_time_now_s=model_age_s,
        measured_v_ego=float(CS.vEgo),
        query_time_s=model_age_s + PREVIEW_S,
        vehicle_model=VM,
        roll_rad=float(params.roll),
        angle_offset_deg=float(params.angleOffsetDeg),
      )
    except (TypeError, ValueError, OverflowError):
      self.reset()
      self.status = STATUS_INVALID_PATH
      return None
    bound_speed = max(float(CS.vEgo), MIN_SPEED)
    roll_compensation = float(params.roll) * ACCELERATION_DUE_TO_GRAVITY
    minimum_curvature = max(-MAX_CURVATURE, (-MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation) / bound_speed ** 2)
    maximum_curvature = min(MAX_CURVATURE, (MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation) / bound_speed ** 2)
    measured_curvature = -VM.calc_curvature(
      math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), bound_speed, params.roll,
    )
    measured_out_of_bounds = not minimum_curvature - 1e-9 <= measured_curvature <= maximum_curvature + 1e-9
    bounded_curvature = float(np.clip(target.curvature, minimum_curvature, maximum_curvature))
    path_limited = bounded_curvature != target.curvature
    if not path_limited:
      current_target_curvature = -VM.calc_curvature(
        math.radians(target.angle_deg - params.angleOffsetDeg), bound_speed, params.roll,
      )
      bounded_curvature = float(np.clip(current_target_curvature, minimum_curvature, maximum_curvature))
      path_limited = bounded_curvature != current_target_curvature
    if path_limited:
      bounded_angle = math.degrees(VM.get_steer_from_curvature(-bounded_curvature, bound_speed, params.roll)) + params.angleOffsetDeg
      target = PathTarget(bounded_curvature, target.speed_mps, bounded_angle, 0.0)
    measured_rate, measured_rate_valid = self._measured_rate(float(CS.steeringAngleDeg), float(CS.steeringRateDeg))
    profile = self._limits(float(CS.vEgo))
    if self.planner is None or CS.steeringPressed:
      seed_rate = _clip(measured_rate, profile.max_rate_deg_s) if measured_rate_valid else 0.0
      self.planner = JerkLimitedRackPlanner(float(CS.steeringAngleDeg), seed_rate)
      self.reference_governor.reset()
    if 0.0 < abs(self.planner.rate_deg_s) - profile.max_rate_deg_s <= 1e-6:
      self.planner.rate_deg_s = math.copysign(profile.max_rate_deg_s, self.planner.rate_deg_s)
    if 0.0 < abs(self.planner.acceleration_deg_s2) - profile.max_acceleration_deg_s2 <= 1e-6:
      self.planner.acceleration_deg_s2 = math.copysign(profile.max_acceleration_deg_s2, self.planner.acceleration_deg_s2)
    limits, profile_transition = self._motion_limits(profile)
    rack_target = RackTarget(target.angle_deg, target.rate_deg_s)
    governed_target = self.reference_governor.update(
      rack_target, self.planner, float(params.angleOffsetDeg), int(self.model.timestampEof), self.dt,
      path_limited or measured_out_of_bounds or profile_transition,
    )
    recovery_acceleration = self._recovery_acceleration(profile, profile_transition)
    try:
      plan = self.planner.update(
        governed_target, limits, self.dt, recovery_acceleration,
      )
    except ValueError:
      self.reset()
      self.status = STATUS_INVALID_PLANNER_STATE
      return None
    planned_curvature = -VM.calc_curvature(math.radians(plan.position_deg - params.angleOffsetDeg), CS.vEgo, params.roll)
    planned_out_of_bounds = not minimum_curvature - 1e-9 <= planned_curvature <= maximum_curvature + 1e-9
    planned_lateral_accel = planned_curvature * CS.vEgo ** 2
    measured_lateral_accel = measured_curvature * CS.vEgo ** 2
    target_angle = target.angle_deg - params.angleOffsetDeg
    measured_angle = float(CS.steeringAngleDeg) - params.angleOffsetDeg
    target_motion = target_angle - measured_angle + RESPONSE_TIME_S * target.rate_deg_s
    unwinding = target_motion * target_angle < 0.0
    if self.driver_override_resume and not unwinding:
      self.driver_override_resume = False
    lateral_accel_error = planned_lateral_accel - measured_lateral_accel
    raw_lateral_jerk = (
      (planned_lateral_accel - self.previous_planned_lateral_accel) / self.dt
      if self.previous_planned_lateral_accel is not None else 0.0
    )
    self.previous_planned_lateral_accel = planned_lateral_accel
    desired_lateral_jerk = float(self.jerk_filter.update(raw_lateral_jerk))
    deadzone_curvature = abs(VM.calc_curvature(math.radians(torque_params.steeringAngleDeadzoneDeg), CS.vEgo, 0.0))
    friction = get_friction(
      lateral_accel_error + .3 * desired_lateral_jerk,
      deadzone_curvature * CS.vEgo ** 2,
      FRICTION_THRESHOLD,
      torque_params,
    )
    target_lateral_accel = target.curvature * CS.vEgo ** 2
    governed_curvature = -VM.calc_curvature(
      math.radians(governed_target.position_deg - params.angleOffsetDeg), CS.vEgo, params.roll,
    )
    governed_lateral_accel = governed_curvature * CS.vEgo ** 2
    trajectory_feedforward_lateral_accel = planned_lateral_accel
    if target_lateral_accel * planned_lateral_accel <= 0.0:
      trajectory_feedforward_lateral_accel = 0.0
    elif (governed_lateral_accel * planned_lateral_accel > 0.0
          and abs(governed_lateral_accel) < abs(planned_lateral_accel)):
      trajectory_feedforward_lateral_accel = governed_lateral_accel
    feedforward_lateral_accel = (
      trajectory_feedforward_lateral_accel - params.roll * ACCELERATION_DUE_TO_GRAVITY - torque_params.latAccelOffset + friction
    )
    feedforward_torque = 0.0 if self.driver_override_resume else -float(torque_from_lateral_accel(feedforward_lateral_accel, torque_params))

    curvature_per_degree = -VM.calc_curvature(math.radians(1.0), CS.vEgo, 0.0)
    lateral_accel_per_degree = curvature_per_degree * CS.vEgo ** 2
    gain = self._feedback_gain(float(CS.vEgo))
    position_feedback = -float(torque_from_lateral_accel(
      gain * lateral_accel_per_degree * (plan.position_deg - CS.steeringAngleDeg), torque_params,
    ))
    rate_feedback = -float(torque_from_lateral_accel(
      gain * lateral_accel_per_degree * RATE_HORIZON_S * (plan.rate_deg_s - measured_rate), torque_params,
    )) if measured_rate_valid else 0.0
    raw_feedback = position_feedback + rate_feedback
    turning_in = (
      target_angle * measured_angle >= 0.0
      and abs(target_angle) > abs(measured_angle)
      and target_motion * target_angle > 0.0
    )
    if turning_in:
      feedback_lower = -MAX_TURN_IN_FEEDBACK_TORQUE if target_angle < 0.0 else -MAX_FEEDBACK_TORQUE
      feedback_upper = MAX_TURN_IN_FEEDBACK_TORQUE if target_angle > 0.0 else MAX_FEEDBACK_TORQUE
    else:
      feedback_lower, feedback_upper = -MAX_FEEDBACK_TORQUE, MAX_FEEDBACK_TORQUE
    feedback = float(np.clip(raw_feedback, feedback_lower, feedback_upper))
    feedback_limited = feedback != raw_feedback
    raw_torque = feedforward_torque + feedback
    torque = float(np.clip(raw_torque, -1.0, 1.0))
    torque_limited = torque != raw_torque
    planned_angle = plan.position_deg - params.angleOffsetDeg
    if ((measured_out_of_bounds and torque * measured_curvature < 0.0)
        or (planned_out_of_bounds and torque * planned_curvature < 0.0)):
      torque_limited |= torque != 0.0
      torque = 0.0
    if planned_angle * target_angle < 0.0 and torque * target_angle < 0.0:
      torque_limited |= torque != 0.0
      torque = 0.0
    motion_limited = (
      plan.rate_limited or plan.acceleration_limited or plan.jerk_limited
      or measured_out_of_bounds or planned_out_of_bounds or self.reference_governor.limited
    )
    if not all(math.isfinite(value) for value in (
      torque, plan.position_deg, plan.rate_deg_s, plan.acceleration_deg_s2, measured_rate,
      lateral_accel_error, position_feedback, rate_feedback, feedforward_torque,
    )):
      self.reset()
      self.status = STATUS_INVALID_OUTPUT
      return None
    driver_torque = float(CS.steeringTorque)
    if CS.steeringPressed:
      if torque * driver_torque < 0.0 or target_motion * driver_torque <= 0.0:
        self.reset()
        self.driver_override_resume = True
        self.status = STATUS_DRIVER_OVERRIDE
        return None
      assisted_torque = _clip(torque, MAX_DRIVER_ASSIST_TORQUE)
      torque_limited |= assisted_torque != torque
      torque = assisted_torque
    self.status = STATUS_ACTIVE
    return RackTrajectoryOutput(
      torque=torque,
      target_curvature=target.curvature,
      target_angle_deg=target.angle_deg,
      target_rate_deg_s=target.rate_deg_s,
      planned_angle_deg=plan.position_deg,
      planned_rate_deg_s=plan.rate_deg_s,
      planned_acceleration_deg_s2=plan.acceleration_deg_s2,
      measured_rate_deg_s=measured_rate,
      lateral_accel_error=lateral_accel_error,
      rate_error_deg_s=plan.rate_deg_s - measured_rate,
      position_feedback_torque=position_feedback,
      rate_feedback_torque=rate_feedback,
      feedforward_torque=feedforward_torque,
      desired_lateral_accel=planned_lateral_accel,
      actual_lateral_accel=measured_lateral_accel,
      desired_lateral_jerk=desired_lateral_jerk,
      feedback_torque=feedback,
      feedback_limited=feedback_limited,
      motion_limited=motion_limited,
      torque_limited=torque_limited,
      rate_limit_deg_s=limits.max_rate_deg_s,
      acceleration_limit_deg_s2=limits.max_acceleration_deg_s2,
      jerk_limit_deg_s3=limits.max_jerk_deg_s3,
      profile_transition=profile_transition,
      path_limited=path_limited,
      infeasible=motion_limited or feedback_limited or torque_limited or profile_transition or path_limited,
      saturated=feedback_limited or torque_limited,
    )
