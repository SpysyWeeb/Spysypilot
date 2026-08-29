import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from opendbc.car.lateral import FRICTION_THRESHOLD, get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_CURVATURE, MAX_LATERAL_ACCEL_NO_ROLL, MIN_SPEED

# Palisade rack trajectory controller: compiles the model path into a steering-angle target,
# moves a jerk-limited virtual rack toward it and tracks that motion with feedforward torque
# plus angle and rate feedback. Ported unchanged from BLaTv2; see docs/BLaTv3_FAILURE_MODES.md.

RESPONSE_TIME_S = .4


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


MEASURED_RATE_FILTER_RC_S = .05


class RackRateEstimator:
  def __init__(self, dt: float, filter_rc_s: float = MEASURED_RATE_FILTER_RC_S) -> None:
    self.dt = dt
    self.filter_rc_s = filter_rc_s
    self.previous_angle_deg: float | None = None
    self.direction = 0
    self.raw_signed_episode = False
    self.rate_filter = FirstOrderFilter(0.0, filter_rc_s, dt)
    self.rate_filter_valid = False

  def reset(self) -> None:
    self.previous_angle_deg = None
    self.direction = 0
    self.raw_signed_episode = False
    self.rate_filter.x = 0.0
    self.rate_filter_valid = False

  def update(self, angle_deg: float, raw_rate_deg_s: float) -> tuple[float, bool]:
    magnitude = abs(raw_rate_deg_s)
    if magnitude == 0.0:
      self.direction = 0
      self.raw_signed_episode = False
      rate, valid = 0.0, True
    elif raw_rate_deg_s < 0.0:
      self.direction = -1
      self.raw_signed_episode = True
      rate, valid = -magnitude, True
    elif self.raw_signed_episode:
      self.direction = 1
      rate, valid = magnitude, True
    elif self.previous_angle_deg is not None and angle_deg != self.previous_angle_deg:
      self.direction = 1 if angle_deg > self.previous_angle_deg else -1
      rate, valid = self.direction * magnitude, True
    elif self.direction:
      rate, valid = self.direction * magnitude, True
    else:
      rate, valid = 0.0, False
    self.previous_angle_deg = angle_deg
    if not valid:
      return rate, False
    if not self.rate_filter_valid:
      self.rate_filter.x = rate
      self.rate_filter_valid = True
    else:
      rate = float(self.rate_filter.update(rate))
    return rate, True


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

  def update(self, target: RackTarget, planner: JerkLimitedRackPlanner, model_timestamp_ns: int, dt: float,
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


DT = .01
PREVIEW_S = .25
RATE_HORIZON_S = .1
HORIZON_POSITION_TOLERANCE_DEG = .01
HORIZON_RATE_TOLERANCE_DEG_S = .5
HORIZON_ACCELERATION_BLEND = .1
MAX_FEEDBACK_TORQUE = .35
MAX_TURN_IN_FEEDBACK_TORQUE = .7
MAX_DRIVER_ASSIST_TORQUE = .5
STALE_MODEL_S = 0.5  # SubMaster's alive window for modelV2: ten model frames
INACTIVE_HOLD_FRAMES = 5  # keep the planned rack through a short latActive blip, e.g. at the standstill gate

STATUS_INACTIVE = 0
STATUS_ACTIVE = 1
STATUS_NO_MODEL = 3
STATUS_INVALID_VEHICLE_STATE = 4
STATUS_STALE_MODEL = 5
STATUS_INVALID_ACTION_TIME = 6
STATUS_INVALID_PATH = 7
STATUS_INVALID_OUTPUT = 8
STATUS_INVALID_PLANNER_STATE = 9

# Coherent-motion corpus: p99 rate/acceleration and p95 jerk by speed. Re-derived from the owner's
# routes in docs/BLaTv3_FAILURE_MODES.md (FM2.6): the rate rows are the p99 steering-angle rate over
# all driving; the acceleration rows match no population. Retired by the learned rack-effort surfaces.
_SPEED_PROFILE_MPH = np.asarray([7.25, 12.5, 17.5, 22.5, 30.0, 45.0])
_RATE_PROFILE_DEG_S = np.asarray([315.848, 289.402, 128.744, 77.716, 35.104, 21.057])
_ACCEL_PROFILE_DEG_S2 = np.asarray([891.046, 827.645, 561.569, 334.476, 172.041, 97.740])
_JERK_PROFILE_DEG_S3 = np.asarray([4567.435, 5115.079, 3851.530, 2329.626, 1318.842, 743.710])

_STOCK_KP_SPEEDS = np.asarray([1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 30.0])
_STOCK_KP = np.asarray([250.0, 120.0, 65.0, 30.0, 11.5, 5.5, 3.5, 2.0, .8])
_LOW_SPEED_KP_END = float(np.float32(15.0 * 0.44704))
_LOW_SPEED_KP_SPEEDS = np.asarray([2.0, 3.0, 5.0, _LOW_SPEED_KP_END])
_LOW_SPEED_KP = np.asarray([65.0, 10.0, 10.0, np.interp(_LOW_SPEED_KP_END, _STOCK_KP_SPEEDS, _STOCK_KP)])


def horizon_candidate_preserves_immediate_path(
  planner_position_deg: float,
  target: RackTarget,
  baseline: RackPlan,
  candidate: RackPlan,
) -> bool:
  immediate_error = target.position_deg - planner_position_deg
  candidate_motion = candidate.position_deg - planner_position_deg
  wrong_side = (
    immediate_error * candidate_motion < 0.0
    and abs(candidate_motion) > HORIZON_POSITION_TOLERANCE_DEG
  )
  position_preserved = (
    abs(target.position_deg - candidate.position_deg)
    <= abs(target.position_deg - baseline.position_deg) + HORIZON_POSITION_TOLERANCE_DEG
  )
  rate_preserved = (
    abs(target.rate_deg_s - candidate.rate_deg_s)
    <= abs(target.rate_deg_s - baseline.rate_deg_s) + HORIZON_RATE_TOLERANCE_DEG_S
  )
  return not wrong_side and position_preserved and rate_preserved


class RackTrajectoryController:
  def __init__(self, dt: float = DT) -> None:
    self.dt = dt
    self.model = None
    self.state_mono_ns = 0
    self.inactive_frames = 0
    self.planner: JerkLimitedRackPlanner | None = None
    self.transition_rate_limit: float | None = None
    self.transition_acceleration_limit: float | None = None
    self.previous_planned_lateral_accel: float | None = None
    self.direction_guard_scale = 1.0
    self.status = STATUS_INACTIVE
    self.jerk_filter = FirstOrderFilter(0.0, 1.0 / (2.0 * math.pi * 1.2), dt)
    self.reference_governor = RackReferenceGovernor()
    self.rack_rate_estimator = RackRateEstimator(dt)

  def set_model(self, model, state_mono_ns: int) -> None:
    # a dropped or invalid model frame keeps the last good plan; staleness is judged by its age
    if model is not None:
      self.model = model
    self.state_mono_ns = int(state_mono_ns)

  def hold(self) -> None:
    # inactive for a frame: keep the planned rack through a short blip, start over after a real disengage
    self.inactive_frames += 1
    self.status = STATUS_INACTIVE
    if self.inactive_frames > INACTIVE_HOLD_FRAMES:
      self.reset()

  def reset(self) -> None:
    self.planner = None
    self.transition_rate_limit = None
    self.transition_acceleration_limit = None
    self.previous_planned_lateral_accel = None
    self.direction_guard_scale = 1.0
    self.status = STATUS_INACTIVE
    self.jerk_filter.x = 0.0
    self.reference_governor.reset()
    self.rack_rate_estimator.reset()

  def _invalidate(self, status: int) -> None:
    self.reset()
    self.status = status

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
      self.hold()
      return None
    self.inactive_frames = 0
    if self.model is None:
      self._invalidate(STATUS_NO_MODEL)
      return None
    if not all(math.isfinite(float(value)) for value in (
      CS.vEgo, CS.steeringAngleDeg, CS.steeringRateDeg, CS.steeringTorque,
      params.roll, params.angleOffsetDeg, lat_delay, desired_curvature,
    )):
      self._invalidate(STATUS_INVALID_VEHICLE_STATE)
      return None
    model_age_s = (self.state_mono_ns - int(self.model.timestampEof)) * 1e-9
    if not 0.0 <= model_age_s <= STALE_MODEL_S:
      self._invalidate(STATUS_STALE_MODEL)
      return None

    try:
      scalar_action_plan_s = float(self.model.action.desiredCurvatureTime)
    except (AttributeError, TypeError, ValueError):
      scalar_action_plan_s = math.nan
    if not math.isfinite(scalar_action_plan_s) or scalar_action_plan_s <= 0.0:
      self._invalidate(STATUS_INVALID_ACTION_TIME)
      return None

    try:
      raw_targets = model_path_targets(
        native_times_s=self.model.orientationRate.t,
        orientation_rates_z=self.model.orientationRate.z,
        velocities_x=self.model.velocity.x,
        scalar_curvature=float(desired_curvature),
        scalar_action_plan_s=scalar_action_plan_s,
        plan_time_now_s=model_age_s,
        measured_v_ego=float(CS.vEgo),
        query_times_s=tuple(model_age_s + offset for offset in HORIZON_OFFSETS_S),
        vehicle_model=VM,
        roll_rad=float(params.roll),
        angle_offset_deg=float(params.angleOffsetDeg),
      )
    except (TypeError, ValueError, OverflowError):
      self._invalidate(STATUS_INVALID_PATH)
      return None

    roll_compensation = float(params.roll) * ACCELERATION_DUE_TO_GRAVITY

    def bound_target(raw_target: PathTarget, speed_mps: float) -> tuple[PathTarget, bool]:
      bound_speed = max(float(speed_mps), MIN_SPEED)
      minimum = max(-MAX_CURVATURE, (-MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation) / bound_speed ** 2)
      maximum = min(MAX_CURVATURE, (MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation) / bound_speed ** 2)
      bounded_curvature = float(np.clip(raw_target.curvature, minimum, maximum))
      limited = bounded_curvature != raw_target.curvature
      if not limited:
        angle_curvature = -VM.calc_curvature(
          math.radians(raw_target.angle_deg - params.angleOffsetDeg), bound_speed, params.roll,
        )
        bounded_curvature = float(np.clip(angle_curvature, minimum, maximum))
        limited = bounded_curvature != angle_curvature
      if not limited:
        return raw_target, False
      bounded_angle = math.degrees(
        VM.get_steer_from_curvature(-bounded_curvature, bound_speed, params.roll),
      ) + params.angleOffsetDeg
      return PathTarget(bounded_curvature, raw_target.speed_mps, bounded_angle, 0.0), True

    targets: list[PathTarget] = []
    target_limits: list[bool] = []
    for offset, raw_target in zip(HORIZON_OFFSETS_S, raw_targets, strict=True):
      target_speed = float(CS.vEgo) if offset <= PREVIEW_S else raw_target.speed_mps
      bounded_target, target_limited = bound_target(raw_target, target_speed)
      targets.append(bounded_target)
      target_limits.append(target_limited)
    preview_index = round(PREVIEW_S / HORIZON_STEP_S)
    target = targets[preview_index]
    path_limited = target_limits[preview_index]

    bound_speed = max(float(CS.vEgo), MIN_SPEED)
    minimum_curvature = max(-MAX_CURVATURE, (-MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation) / bound_speed ** 2)
    maximum_curvature = min(MAX_CURVATURE, (MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation) / bound_speed ** 2)
    measured_curvature = -VM.calc_curvature(
      math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), bound_speed, params.roll,
    )
    measured_out_of_bounds = not minimum_curvature - 1e-9 <= measured_curvature <= maximum_curvature + 1e-9
    previous_angle_deg = self.rack_rate_estimator.previous_angle_deg
    measured_rate, measured_rate_valid = self.rack_rate_estimator.update(float(CS.steeringAngleDeg), float(CS.steeringRateDeg))
    profile = self._limits(float(CS.vEgo))
    if self.planner is None:
      seed_rate = _clip(measured_rate, profile.max_rate_deg_s) if measured_rate_valid else 0.0
      self.planner = JerkLimitedRackPlanner(float(CS.steeringAngleDeg), seed_rate)
      self.reference_governor.reset()
    elif CS.steeringPressed and previous_angle_deg is not None:
      self.planner.position_deg += float(CS.steeringAngleDeg) - previous_angle_deg
    if 0.0 < abs(self.planner.rate_deg_s) - profile.max_rate_deg_s <= 1e-6:
      self.planner.rate_deg_s = math.copysign(profile.max_rate_deg_s, self.planner.rate_deg_s)
    if 0.0 < abs(self.planner.acceleration_deg_s2) - profile.max_acceleration_deg_s2 <= 1e-6:
      self.planner.acceleration_deg_s2 = math.copysign(profile.max_acceleration_deg_s2, self.planner.acceleration_deg_s2)
    limits, profile_transition = self._motion_limits(profile)
    rack_target = RackTarget(target.angle_deg, target.rate_deg_s)
    governed_target = self.reference_governor.update(
      rack_target, self.planner, int(self.model.timestampEof), self.dt,
      path_limited or measured_out_of_bounds or profile_transition,
    )
    planner = self.planner
    assert planner is not None
    timed_targets = tuple(
      (offset, governed_target if index == preview_index else RackTarget(path_target.angle_deg, path_target.rate_deg_s))
      for index, (offset, path_target) in enumerate(zip(HORIZON_OFFSETS_S, targets, strict=True))
      if offset > 0.0
    )
    desired_acceleration = self._recovery_acceleration(profile, profile_transition)
    try:
      if desired_acceleration is None:
        fitted_acceleration = horizon_desired_acceleration(planner, timed_targets)
        natural_frequency = 2.0 / limits.response_time_s
        reactive_acceleration = (
          natural_frequency ** 2 * (governed_target.position_deg - planner.position_deg)
          + 2.0 * natural_frequency * (governed_target.rate_deg_s - planner.rate_deg_s)
        )
        horizon_acceleration = reactive_acceleration + HORIZON_ACCELERATION_BLEND * (
          fitted_acceleration - reactive_acceleration
        )

        def preview(acceleration_override: float | None) -> RackPlan:
          candidate = JerkLimitedRackPlanner(planner.position_deg, planner.rate_deg_s)
          candidate.acceleration_deg_s2 = planner.acceleration_deg_s2
          return candidate.update(governed_target, limits, self.dt, acceleration_override)

        baseline = preview(None)
        horizon = preview(horizon_acceleration)
        if horizon_candidate_preserves_immediate_path(planner.position_deg, governed_target, baseline, horizon):
          desired_acceleration = horizon_acceleration
      raw_plan = planner.update(
        governed_target, limits, self.dt, desired_acceleration,
      )
    except ValueError:
      self._invalidate(STATUS_INVALID_PLANNER_STATE)
      return None
    raw_planned_curvature = -VM.calc_curvature(math.radians(raw_plan.position_deg - params.angleOffsetDeg), CS.vEgo, params.roll)
    planned_out_of_bounds = not minimum_curvature - 1e-9 <= raw_planned_curvature <= maximum_curvature + 1e-9
    plan = raw_plan
    if planned_out_of_bounds:
      planned_curvature = float(np.clip(raw_planned_curvature, minimum_curvature, maximum_curvature))
      planned_angle = math.degrees(VM.get_steer_from_curvature(-planned_curvature, bound_speed, params.roll)) + params.angleOffsetDeg
      plan = RackPlan(planned_angle, 0.0, 0.0, True, raw_plan.acceleration_limited, raw_plan.jerk_limited)
    planned_curvature = -VM.calc_curvature(math.radians(plan.position_deg - params.angleOffsetDeg), CS.vEgo, params.roll)
    planned_lateral_accel = planned_curvature * CS.vEgo ** 2
    measured_lateral_accel = measured_curvature * CS.vEgo ** 2
    target_angle = target.angle_deg - params.angleOffsetDeg
    measured_angle = float(CS.steeringAngleDeg) - params.angleOffsetDeg
    target_motion = target_angle - measured_angle + RESPONSE_TIME_S * target.rate_deg_s
    turning_in = (
      target_angle * measured_angle >= 0.0
      and abs(target_angle) > abs(measured_angle)
      and target_motion * target_angle > 0.0
    )
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
    unwind_scale = 1.0
    planned_angle = plan.position_deg - params.angleOffsetDeg
    intended_angle = measured_angle + target_motion
    if measured_angle != 0.0:
      planned_hold_angle = abs(planned_angle) if planned_angle * measured_angle > 0.0 else 0.0
      turn_in_angle = abs(intended_angle) if intended_angle * measured_angle > 0.0 else 0.0
      unwind_scale = min(
        1.0, max(planned_hold_angle, turn_in_angle) / abs(measured_angle),
      )
    feedforward_lateral_accel = (
      trajectory_feedforward_lateral_accel - params.roll * ACCELERATION_DUE_TO_GRAVITY - torque_params.latAccelOffset + friction
    )
    feedforward_torque = -float(torque_from_lateral_accel(feedforward_lateral_accel, torque_params))

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
    if turning_in:
      feedback_lower = -MAX_TURN_IN_FEEDBACK_TORQUE if target_angle < 0.0 else -MAX_FEEDBACK_TORQUE
      feedback_upper = MAX_TURN_IN_FEEDBACK_TORQUE if target_angle > 0.0 else MAX_FEEDBACK_TORQUE
    else:
      feedback_lower, feedback_upper = -MAX_FEEDBACK_TORQUE, MAX_FEEDBACK_TORQUE
    feedback = float(np.clip(raw_feedback, feedback_lower, feedback_upper))
    feedback_limited = feedback != raw_feedback
    raw_torque = feedforward_torque + feedback
    if raw_torque * measured_angle > 0.0:
      raw_torque *= unwind_scale
    elif (measured_angle == 0.0 and planned_angle * intended_angle > 0.0
          and raw_torque * planned_angle < 0.0):
      raw_torque = 0.0
    torque = float(np.clip(raw_torque, -1.0, 1.0))
    torque_limited = torque != raw_torque
    if planned_angle * target_angle < 0.0 and torque * target_angle < 0.0:
      self.direction_guard_scale = 0.0
    else:
      self.direction_guard_scale = min(1.0, self.direction_guard_scale + self.dt / REFERENCE_REVERSAL_RC_S)
    guarded_torque = torque * self.direction_guard_scale
    torque_limited |= guarded_torque != torque
    torque = guarded_torque
    motion_limited = (
      plan.rate_limited or plan.acceleration_limited or plan.jerk_limited
      or measured_out_of_bounds or planned_out_of_bounds or self.reference_governor.limited
    )
    if not all(math.isfinite(value) for value in (
      torque, plan.position_deg, plan.rate_deg_s, plan.acceleration_deg_s2, measured_rate,
      lateral_accel_error, position_feedback, rate_feedback, feedforward_torque,
    )):
      self._invalidate(STATUS_INVALID_OUTPUT)
      return None
    if CS.steeringPressed:
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
