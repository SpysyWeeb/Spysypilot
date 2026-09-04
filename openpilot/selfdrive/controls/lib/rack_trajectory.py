import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from opendbc.car.lateral import FRICTION_THRESHOLD, get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_CURVATURE, MAX_LATERAL_ACCEL_NO_ROLL, MAX_LATERAL_JERK, MIN_SPEED

# Palisade rack trajectory controller: compiles the model path into a steering-angle target,
# moves a jerk-limited virtual rack toward it and tracks that motion with feedforward torque
# plus angle and rate feedback. Ported from BLaTv2; since phase 2 step 3 the path comes from modeld's
# curvature preview (the scalar's own function evaluated along the plan). See docs/BLaTv3_FAILURE_MODES.md.

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
  preview_time_s: float
  reference_limited: bool
  near_target_angle_deg: float
  direction_guarded: bool
  driver_assist_limited: bool
  driver_assist_cap: float
  early_release: bool
  direction_fraction: float
  envelope_open_rate_deg_s: float
  envelope_open_acceleration_deg_s2: float
  envelope_open_jerk_deg_s3: float
  envelope_preview_time_s: float
  hold_topup_torque: float
  hold_topup_growing: bool


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


def _smoothstep(value: float, edge0: float, edge1: float) -> float:
  blend = min(max((value - edge0) / (edge1 - edge0), 0.0), 1.0)
  return blend * blend * (3.0 - 2.0 * blend)


def _clip(value: float, limit: float) -> float:
  return max(-limit, min(limit, value))


def _hold_topup_step(state: float, error_deg: float, steady_gate: float, accumulating: bool, fast_leak: bool, dt: float) -> float:
  """One frame of the hold top-up's leaky integrator (FM3.14). The caller resolves the anti-windup gates
  (`accumulating`) and the steadiness weight (`steady_gate`); this owns only the ODE, the leak selection,
  the hard bound and the zero snap. Growth and the fast leak never apply in the same call (the caller
  passes accumulating=False whenever the driver's hand selects the fast leak), so the per-frame step is
  bounded by dt * (HOLD_TOPUP_RATE * HOLD_TOPUP_ERROR_CAP_DEG + MAX_HOLD_TOPUP_TORQUE / HOLD_TOPUP_OVERRIDE_DECAY_S)
  = 0.0117 even when a reversed error drains the state while growing the other way -- inside R7_MAX_TORQUE_STEP."""
  growth = HOLD_TOPUP_RATE * _clip(error_deg, HOLD_TOPUP_ERROR_CAP_DEG) * steady_gate if accumulating else 0.0
  leak_rc = HOLD_TOPUP_OVERRIDE_DECAY_S if fast_leak else HOLD_TOPUP_LEAK_RC_S
  state = _clip(state + dt * (growth - state / leak_rc), MAX_HOLD_TOPUP_TORQUE)
  return 0.0 if abs(state) < HOLD_TOPUP_ZERO_EPS_TORQUE else state


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


HORIZON_S = 2.0
HORIZON_STEP_S = .25
HORIZON_OFFSETS_S = tuple(index * HORIZON_STEP_S for index in range(round(HORIZON_S / HORIZON_STEP_S) + 1))


def model_path_targets(
  *,
  native_times_s: Sequence[float],
  velocities_x: Sequence[float],
  preview_times_s: Sequence[float],
  preview_curvatures: Sequence[float],
  scalar_curvature: float,
  plan_time_now_s: float,
  measured_v_ego: float,
  query_times_s: Sequence[float],
  vehicle_model,
  roll_rad: float,
  angle_offset_deg: float,
) -> tuple[PathTarget, ...]:
  """Steering targets along the plan: modeld's curvature preview, pinned to the scalar curvature
  controlsd hands down, converted to a wheel angle at the speed the plan expects at each query time.

  Queries are on the vehicle's timeline (seconds from the plan origin; the plan's age is now). The
  preview is desiredCurvature's own function from the action time on, so it is read one action time
  ahead of each query: at the plan's age it is the scalar the plan would publish now, and further out
  it is the scalar's future. Past the preview's end the last sample holds: the covered range is the
  horizon.
  """
  scalar = float(scalar_curvature)
  measured_speed = float(measured_v_ego)
  queries = tuple(float(query) for query in query_times_s)
  if not queries or not math.isfinite(scalar) or not math.isfinite(measured_speed) or measured_speed < 0.0:
    raise ValueError("invalid scalar path target")

  def series(raw_times: Sequence[float], raw_values: Sequence[float], name: str) -> tuple[np.ndarray, np.ndarray]:
    count = len(raw_times)
    if count < 2 or len(raw_values) != count:
      raise ValueError(f"invalid {name}")
    times = np.array(raw_times, dtype=np.float64)
    values = np.array(raw_values, dtype=np.float64)
    if not (np.isfinite(times).all() and np.isfinite(values).all()) or times[0] < 0.0 or not (np.diff(times) > 0.0).all():
      raise ValueError(f"invalid {name}")
    return times, values

  # a plan that stops inside the horizon still covers it; the speed is floored below
  times, speeds = series(native_times_s, velocities_x, "model path")
  preview_times, previews = series(preview_times_s, preview_curvatures, "curvature preview")
  if preview_times[0] <= 0.0:
    raise ValueError("invalid curvature preview")
  now = float(plan_time_now_s)
  if not all(times[0] <= query <= times[-1] for query in (now, *queries)):
    raise ValueError("model path does not cover requested timestamps")
  # controlsd's scalar is the model's first sample after the ISO clip, so the pin keeps the clip
  curvatures = np.array([scalar + preview - previews[0] for preview in previews])

  # every query and its rate stencil, interpolated in one pass per series
  count = len(queries)
  # the rate stencil stays inside the preview: at the action time it is the forward difference
  befores = [max(times[0], now, query - .05) for query in queries]
  afters = [min(times[-1], query + .05) for query in queries]
  points = np.array([*queries, *befores, *afters])
  curvature_at = np.interp(points - now + preview_times[0], preview_times, curvatures)
  speed_now = float(np.interp(now, times, speeds))
  speed_at = np.interp(points, times, speeds)

  def angle_at(index: int) -> tuple[float, float]:
    speed = max(MIN_SPEED, measured_speed + float(speed_at[index]) - speed_now)
    angle = math.degrees(vehicle_model.get_steer_from_curvature(-float(curvature_at[index]), speed, roll_rad)) + angle_offset_deg
    return speed, angle

  targets: list[PathTarget] = []
  for index in range(count):
    speed, angle = angle_at(index)
    before, after = befores[index], afters[index]
    rate = (angle_at(2 * count + index)[1] - angle_at(count + index)[1]) / (after - before) if after > before else 0.0
    curvature = float(curvature_at[index])
    if not all(math.isfinite(value) for value in (curvature, speed, angle, rate)):
      raise ValueError("non-finite path target")
    targets.append(PathTarget(curvature, speed, angle, rate))
  return tuple(targets)


REFERENCE_FILTER_RC_S = .1
REFERENCE_FILTER_PREVIEW_RC_S = .2  # added to the time constant at the full preview: a consistent road earns calmer tracking
REFERENCE_FILTER_TRAIL_LATERAL_ACCEL = .2  # m/s^2: how far the served target may trail the raw one
REFERENCE_FILTER_TRAIL_MAX_DEG = 3.0  # and in wheel angle, which is the tighter bound below ~12 m/s
DIRECTION_GUARD_RC_S = .12


def reference_trail_limit_deg(vehicle_model, speed_mps: float) -> float:
  """How far the served target may trail the model's: a lateral acceleration, capped in wheel angle."""
  speed = max(float(speed_mps), MIN_SPEED)
  return min(
    abs(math.degrees(vehicle_model.get_steer_from_curvature(REFERENCE_FILTER_TRAIL_LATERAL_ACCEL / speed ** 2, speed, 0.0))),
    REFERENCE_FILTER_TRAIL_MAX_DEG,
  )


class ReferenceFilter:
  """Smooth the immediate target without holding it back by more than a bounded amount.

  A first-order filter removes the frame-to-frame jitter of the model's target: a few degrees of
  wheel at low speed and nothing in lateral acceleration. The served target may trail the raw one
  by at most a lateral acceleration, capped in wheel angle, so a turn-in or an unwind passes with
  at most that trailing at once, and any trailing decays with the time constant once the raw target
  settles. The served rate is always the filtered rate -- the trail bound only clamps position, it
  never substitutes the raw target's own (unfiltered) rate for the served one, so a fast excursion
  that reaches the trail bound still hands the tracker a smoothed rate, not a step.
  """

  def __init__(self) -> None:
    self.target: RackTarget | None = None
    self.limited = False

  def reset(self) -> None:
    self.target = None
    self.limited = False

  def update(self, target: RackTarget, trail_limit_deg: float, dt: float, bypass: bool = False,
             rc_s: float = REFERENCE_FILTER_RC_S) -> RackTarget:
    if self.target is None or bypass:
      self.target = target
      self.limited = False
      return target
    alpha = dt / (rc_s + dt)
    position = self.target.position_deg + alpha * (target.position_deg - self.target.position_deg)
    rate = self.target.rate_deg_s + alpha * (target.rate_deg_s - self.target.rate_deg_s)
    trail = target.position_deg - position
    self.limited = abs(trail) > trail_limit_deg
    if self.limited:
      # Bound the position at the trail limit so a real change still passes at once (R5); leave
      # `rate` as the filtered value computed above instead of snapping it to the raw target's own
      # rate. Snapping removed rate smoothing exactly during the fast excursions that reach this
      # branch -- a step straight into the planner's rate-tracking term -- while the alpha blend
      # above is already bounded between the previous served rate and the raw one, so this still
      # converges to the raw rate (within one time constant) without the discontinuity.
      position = target.position_deg - math.copysign(trail_limit_deg, trail)
    self.target = RackTarget(position, rate)
    return self.target


PREVIEW_ADMIT_DEVIATION_M = .15  # path consistency, in metres, to read the target one step further ahead
PREVIEW_KEEP_DEVIATION_M = .2  # and to keep reading it there
PREVIEW_ADMIT_HEADING_DEG = 1.0
PREVIEW_KEEP_HEADING_DEG = 1.33
PREVIEW_FLICKER_TOLERANCE_DEG = .25  # a far target may not swing more than the near one between model frames
PREVIEW_MAX_DISTANCE_M = 40.0
PREVIEW_MAX_Y_STD_M = .35  # p99 of the path's yStd at 2 s on straight frames (routes 20/22)
PREVIEW_SHORTEN_UPDATES = 2  # model frames of disagreement before the preview shortens
PREVIEW_LENGTHEN_UPDATES = 2  # model frames of agreement before it lengthens by one step
RESPONSE_TIME_PREVIEW_S = .1  # extra tracker response time granted at the full preview
CLOTHOID_STEPS = 20

# R4 horizon-implied envelope opening (G-independent, docs/BLaTv3_FAILURE_MODES.md): a second,
# confidence-free PreviewScheduler (envelope_scheduler, below) reuses every gate above except the
# confidence check, corroborated instead by its own DCPC-graft frame-to-frame far-point stability
# check (previous_far_y / PREVIEW_ENVELOPE_DRIFT_M).
PREVIEW_ENVELOPE_DRIFT_M = .25  # route-20 decode (owner Q2, 31,107 model frames): |dy@2s| between
                                 # consecutive model frames swings 0.2-2.0 m through the island
                                 # window while the 1s point stays 0.03-0.25 m; this flat gate
                                 # admits the stable near approach and rejects the swinging far end
ENVELOPE_OPEN_MARGIN = 1.15  # demand is re-derived from the live horizon every frame, so a modest margin suffices
ENVELOPE_EASE_UP_RC_S = HORIZON_STEP_S  # owner Q4: tied to the scheduler's own lengthen pace so
                                         # ease-up can never lag PREVIEW_LENGTHEN_UPDATES's own admission rate


def _arc_y(curvature: float, x: np.ndarray) -> np.ndarray:
  if abs(curvature) < 1e-9:
    return 0.5 * curvature * x * x
  return (1.0 - np.cos(curvature * x)) / curvature


def _clothoid_y(curvature_near: float, curvature_far: float, x_action: float, x_far: float, x_query: np.ndarray) -> np.ndarray:
  """Lateral position of the path whose curvature is held at the near value to x_action and then
  ramps linearly to the far value at x_far: the shape a consistent plan draws between its targets."""
  heading = curvature_near * x_action
  y_action = float(_arc_y(curvature_near, np.array(x_action)))
  span = x_far - x_action
  if span <= 1e-6:
    return np.full_like(x_query, y_action)
  xs = np.linspace(x_action, x_far, CLOTHOID_STEPS + 1)
  curvatures = curvature_near + (curvature_far - curvature_near) * (xs - x_action) / span
  dx = xs[1] - xs[0]
  headings = heading + np.concatenate(([0.0], np.cumsum(dx * 0.5 * (curvatures[:-1] + curvatures[1:]))))
  slopes = np.sin(headings)
  ys = y_action + np.concatenate(([0.0], np.cumsum(dx * 0.5 * (slopes[:-1] + slopes[1:]))))
  return np.interp(x_query, xs, ys)


class PreviewScheduler:
  """Schedule how far ahead the plan is trusted to be consistent.

  On a path the model draws consistently -- a clothoid from the near curvature to the far one,
  within a lateral tolerance in metres, with the far target as steady as the near one -- the
  preview lengthens one horizon step at a time; the moment the path stops agreeing (a jog, a curve
  entry, a flickering far point, a lane change, the driver's hands, a limited path) it collapses
  within two model frames. The preview never replaces the near target: it only earns the tracker a
  calmer reference (a longer filter time constant) and a longer response time.
  """

  def __init__(self, require_confidence: bool = True) -> None:
    # require_confidence=False (R4's envelope_scheduler): every gate above still applies except
    # the confidence check, corroborated instead by previous_far_y's own temporal-stability check.
    self.require_confidence = require_confidence
    self.index = 0
    self.fail_updates = 0
    self.pass_updates = 0
    self.last_model_timestamp_ns: int | None = None
    self.previous_angles: tuple[float, ...] | None = None
    self.previous_far_y: tuple[float, ...] | None = None

  def reset(self) -> None:
    self.index = 0
    self.fail_updates = 0
    self.pass_updates = 0
    self.last_model_timestamp_ns = None
    self.previous_angles = None
    self.previous_far_y = None

  @property
  def preview_s(self) -> float:
    return HORIZON_OFFSETS_S[self.index]

  def update(self, model, model_timestamp_ns: int, action_time_s: float, targets: Sequence[PathTarget], forced: bool) -> int:
    if forced:
      self.index = 0
      self.fail_updates = 0
      self.pass_updates = 0
      # R4 fix: a forced frame skips _admissible entirely below, so it must clear the DCPC
      # baseline itself -- otherwise the next real frame compares against a pre-event far point
      # instead of treating the resumed data as having no baseline yet (previous_angles has no
      # analogous gap: it is always refreshed unconditionally a few lines down).
      self.previous_far_y = None
    timestamp = int(model_timestamp_ns)
    if timestamp == self.last_model_timestamp_ns:
      return self.index
    self.last_model_timestamp_ns = timestamp
    angles = tuple(target.angle_deg for target in targets)
    admissible = 0 if forced else self._admissible(model, action_time_s, targets, angles)
    self.previous_angles = angles
    if forced:
      return 0
    if admissible < self.index:
      self.fail_updates += 1
      if self.fail_updates >= PREVIEW_SHORTEN_UPDATES:
        self.index = admissible
        self.fail_updates = 0
        self.pass_updates = 0
    else:
      self.fail_updates = 0
      if admissible > self.index:
        self.pass_updates += 1
        if self.pass_updates >= PREVIEW_LENGTHEN_UPDATES:
          self.index += 1
          self.pass_updates = 0
      else:
        self.pass_updates = 0
    return self.index

  def _admissible(self, model, action_time_s: float, targets: Sequence[PathTarget], angles: tuple[float, ...]) -> int:
    position = model.position
    times = np.asarray(position.t, dtype=np.float64)
    if len(times) < 2 or len(position.x) != len(times) or len(position.y) != len(times):
      self.previous_far_y = None  # R4 fix: no valid position data -- drop the DCPC baseline
                                    # rather than let a later frame compare against it stale
      return 0
    if self.require_confidence and str(model.confidence) == "red":
      return 0
    xs = np.asarray(position.x, dtype=np.float64)
    ys = np.asarray(position.y, dtype=np.float64)
    y_std = np.asarray(position.yStd, dtype=np.float64) if len(position.yStd) == len(times) else None
    speed_times = np.asarray(model.velocity.t, dtype=np.float64)
    speeds = np.asarray(model.velocity.x, dtype=np.float64)
    # a path with a hole in it proves nothing: comparisons against NaN never fail, so check up front
    if not all(np.isfinite(array).all() for array in (times, xs, ys, speed_times, speeds)) or (y_std is not None and not np.isfinite(y_std).all()):
      self.previous_far_y = None  # R4 fix: same rationale -- a NaN/holed frame must not leave a
                                    # stale (or, worse, NaN-poisoned) baseline for the next frame
      return 0
    x_action = float(np.interp(action_time_s, times, xs))
    near = targets[0]
    admitted = 0
    for index in range(1, len(targets)):
      far_time = action_time_s + HORIZON_OFFSETS_S[index]
      keeping = index <= self.index
      window = (times >= action_time_s) & (times <= far_time)
      if window.any():
        x_far = float(np.interp(far_time, times, xs))
        deviation = float(np.max(np.abs(ys[window] - _clothoid_y(near.curvature, targets[index].curvature, x_action, x_far, xs[window]))))
        if deviation >= (PREVIEW_KEEP_DEVIATION_M if keeping else PREVIEW_ADMIT_DEVIATION_M):
          break
        if y_std is not None and float(np.max(y_std[window])) > PREVIEW_MAX_Y_STD_M:
          break
      if abs(angles[index] - angles[0]) > (PREVIEW_KEEP_HEADING_DEG if keeping else PREVIEW_ADMIT_HEADING_DEG):
        break
      if self.previous_angles is not None and len(self.previous_angles) == len(angles):
        if abs(angles[index] - self.previous_angles[index]) > abs(angles[0] - self.previous_angles[0]) + PREVIEW_FLICKER_TOLERANCE_DEG:
          break
      sample_times = np.linspace(action_time_s, far_time, 9)
      if float(np.trapezoid(np.interp(sample_times, speed_times, speeds), sample_times)) > PREVIEW_MAX_DISTANCE_M:
        break
      if not self.require_confidence:
        # DCPC graft: cheap insurance for the "smooth but wrong" defense that dropping confidence
        # removes -- this step's own far prediction must hold steady frame to frame, not just be
        # smooth against the current frame's own near-to-far shape.
        y_far = float(np.interp(far_time, times, ys))
        if self.previous_far_y is not None and abs(y_far - self.previous_far_y[index]) > PREVIEW_ENVELOPE_DRIFT_M:
          break
      admitted = index
    if not self.require_confidence:
      self.previous_far_y = tuple(float(np.interp(action_time_s + offset, times, ys)) for offset in HORIZON_OFFSETS_S)
    return admitted



DT = .01
RATE_HORIZON_S = .1
HORIZON_POSITION_TOLERANCE_DEG = .01
HORIZON_RATE_TOLERANCE_DEG_S = .5
HORIZON_ACCELERATION_BLEND = .1
MAX_FEEDBACK_TORQUE = .35
TURN_IN_BLEND_DEG = 3.0  # the feedback cap blends between its two values over this much angle, not a boolean jump
MAX_TURN_IN_FEEDBACK_TORQUE = .7
MAX_DRIVER_ASSIST_TORQUE = .5     # unchanged value; now the envelope's FLOOR, not its fixed cap
DRIVER_ASSIST_CEILING = 1.0       # == the ISO clip already applied upstream; not a new ceiling (R10)
# Direction guard v2 (target-referred bounded fallback, docs/BLaTv3_FAILURE_MODES.md FM3.5/FM3.9/R7/R10):
R7_MAX_TORQUE_STEP = .05  # existing rule bound (FM3.5: "R7 sweeps, jump < 0.05"), reused on the output verbatim
GUARD_TORQUE_BLEND = .05  # set equal to R7_MAX_TORQUE_STEP: an opposing torque smaller than the R7 step can't
                           # produce a meaningful "opposing push", so the sign-disagreement trip only arms above it
GUARD_UNWIND_BLEND = .15  # ramps the old direction_fraction>=1.0 snap over the top 15% of its range instead,
                           # matching the file's existing blend-width precedent (turn_in_fraction, rate_gain_scale)
GUARD_FALLBACK_TORQUE_CAP = .18  # empirically derived from real 2c/2d/2e replay: the uncapped target-referred
                                  # feedback at every historical exact-zero episode has p50=0.036, p75=0.082,
                                  # p90=0.467, p99=3.80; 0.18 sits at ~85th percentile, resolving the dominant
                                  # near-center regime without clipping while hard-capping the reversal-lag tail.
                                  # Fit to the old build's zero-episode geometry, which this change eliminates --
                                  # owner decision 2026-09-01: ship now, re-derive after the next field drive.
# Hold top-up (docs/BLaTv3_FAILURE_MODES.md FM3.14): the third torque term. Feedforward predicts, position/rate
# feedback correct, and this bounded leaky integrator makes up whatever standing shortfall is left while the
# wheel is meant to hold -- in angle space, deliberately NOT through gain(v) * lateral_accel_per_degree, since
# that v^2-scaled pipeline is what under-supplied the real hold effort on route 0x3e (request -0.57 -> -0.36 at a
# constant 35 deg as the car slowed 9.4 -> 7.9 m/s; the wheel crept out while the plan tracked it).
HOLD_TOPUP_RATE = .1                    # torque per (deg * s): a held 1.5 deg error is worth an R7 step within ~0.35 s
MAX_HOLD_TOPUP_TORQUE = .2              # the field event's own measured gap (0.21); must stay under MAX_FEEDBACK_TORQUE
HOLD_TOPUP_ERROR_CAP_DEG = 5.           # beyond this the error is no longer a small shortfall; P/D and the filter own it
HOLD_TOPUP_LEAK_RC_S = 3.               # always-on passive leak: a curve hold barely bleeds, nothing outlives its cause
HOLD_TOPUP_OVERRIDE_DECAY_S = .3        # fast leak while the driver presses and through the release cooldown below
HOLD_TOPUP_RELEASE_COOLDOWN_S = .3      # a brief grab must not hand a barely-decayed residual back to the slow leak
HOLD_TOPUP_ZERO_EPS_TORQUE = 1e-4       # snap to exact 0.0 below this so bit-identity with no top-up is reachable
HOLD_TOPUP_PLAN_RATE_DEG_S = 5.         # the virtual rack is holding, not chasing, below this planned rate ...
HOLD_TOPUP_PLAN_RATE_BLEND_DEG_S = 3.   # ... fading out continuously by 8 deg/s (R7: no boolean gate on a ramp)
HOLD_TOPUP_MEASURED_RATE_DEG_S = 8.     # the real wheel is not actively turning below this: above the shadow observer's
HOLD_TOPUP_MEASURED_RATE_BLEND_DEG_S = 4.  # 2 deg/s steady bar on purpose -- the 2 deg/s creep must still be corrected
HOLD_TOPUP_APPROACH_RATE_DEG_S = .25     # the wheel already closing on the plan faster than this is not a standing
HOLD_TOPUP_APPROACH_BLEND_DEG_S = .75    # shortfall: growth fades out by 1 deg/s of approach so the last additions are
                                         # allowed to finish their work before more is added (no integrator overshoot)
assert MAX_HOLD_TOPUP_TORQUE < MAX_FEEDBACK_TORQUE  # the two sum directly in raw_torque (R10: checked, not implicit)

# F3 highway feedforward taper (report.md S2 rank 3): on a highway-speed straight, a flat,
# speed-independent amount of angle-level noise (route audit torque_decomp/speed_vs_chatter.py:
# mean|d wheel| 0.082-0.095 deg/frame from 0-150 km/h) becomes a v^2-scaled feedforward swing (mean|d
# f| 0.00038 at 0-20 km/h to 0.01055 at 120-150 km/h, 28x; mean|d output| only 4.3x, because P/D
# partly cancels it -- segments 37-44, 128 km/h, are the route's roughest stretch by that measure).
# Re-verified directly against route 4d (impl_F3/speed_chatter_before_after.py,
# impl_F3/measured_vs_target_chatter.py): in the 120-150 km/h near-zero-curvature bucket, |d f| on
# real route data correlates 0.93 with |d measured_lateral_accel| (CS.steeringAngleDeg's own noise,
# fed in through the friction term's lateral_accel_error, opendbc/car/lateral.py get_friction) and
# only 0.02 with |d target_lateral_accel| (the model's own target dither) -- the measured-angle path
# through friction dominates the real signal by ~10x (mean|d measured_lateral_accel| 0.0169 vs
# mean|d target_lateral_accel| 0.0016 in that bucket), not the plan/target path alone. So this tapers
# the FULL feedforward_lateral_accel (trajectory term + friction, everything torque_from_lateral_accel
# actually receives) rather than only the plan-derived trajectory_feedforward_lateral_accel term --
# still exactly "the feedforward layer" the fix's own mandate names, just downstream of where friction
# joins it, so the taper reaches whichever of the two inputs is actually chattering that frame.
# Low-passes that combined input, but ONLY while every one of three smoothstep gates (R7) is open:
# near-zero curvature (reuses TURN_IN_BLEND_DEG, the same near-center deadband turn_in_fraction and
# direction_fraction already use, and the report's own "near-straight" definition -- the worst 128
# km/h windows in torque_decomp/rough_windows.json sit at 0.4-1.6 deg near/target range), holding
# rather than moving (reuses HOLD_TOPUP_PLAN_RATE_DEG_S/BLEND, the top-up's own "holding not chasing"
# threshold, applied to whichever of the served target's own rate and the plan's is larger -- the
# served rate leads the plan into a real highway turn-in, so the plan's rate alone leaves the taper
# open for the first frames of the move), and highway speed (FF_TAPER_SPEED_MPS matches
# _STOCK_KP_SPEEDS[-1] below: above it P/D gain no longer grows with speed while feedforward keeps
# scaling as v^2, so the same angle-level dither buys ever less P/D counter-authority as speed climbs
# further). A turn-in or unwind at any speed, a curve already held, or the driver's hands on the wheel
# drives the gate to exactly 0.0, so the low-passed term contributes nothing and today's code
# reproduces bit-for-bit (R5/R7).
FF_TAPER_SPEED_MPS = 30.0               # m/s (108 km/h); == _STOCK_KP_SPEEDS[-1] (below)
FF_TAPER_SPEED_BLEND_MPS = 8.0          # opens from 22 m/s (79 km/h): speed_vs_chatter.py's own
                                         # 60-80/80-100 km/h bucket boundary, already 13-19x the
                                         # 0-20 km/h mean|d f| baseline; fully open well inside
                                         # segments 37-44's 120-129 km/h (33.3-35.8 m/s)
FF_TAPER_ANGLE_DEG = TURN_IN_BLEND_DEG  # 3 deg: reused, not re-tuned -- see the comment block above
FF_TAPER_RATE_DEG_S = HOLD_TOPUP_PLAN_RATE_DEG_S              # 5 deg/s: reused, ditto
FF_TAPER_RATE_BLEND_DEG_S = HOLD_TOPUP_PLAN_RATE_BLEND_DEG_S  # 3 deg/s: reused, ditto
FF_TAPER_RC_S = REFERENCE_FILTER_RC_S   # reused verbatim, not re-tuned: measured on route 4d
                                         # (impl_F3/speed_chatter_before_after.py) against a slower
                                         # 0.2s candidate -- 0.1s gives the same highway-chatter
                                         # reduction (120-150 km/h bucket: mean|d output| -35.5% vs
                                         # -36.5%, mean|d f| -56.2% vs -57.9%) while cutting the worst
                                         # measured highway turn-in transient by ~26% (0.051 vs 0.069
                                         # peak |d torque| across 17 route-4d highway turn-ins,
                                         # impl_F3/highway_turnin_scan.py) -- the smaller value changes
                                         # response least for equivalent benefit (R7: transient bounded
                                         # by construction, decays with this same time constant once a
                                         # gate closes)
STALE_MODEL_S = 0.5  # SubMaster's alive window for modelV2: ten model frames
INACTIVE_HOLD_FRAMES = 5  # keep the planned rack through a short latActive blip, e.g. at the standstill gate

STATUS_INACTIVE = 0
STATUS_ACTIVE = 1
STATUS_NO_MODEL = 3
STATUS_INVALID_VEHICLE_STATE = 4
STATUS_STALE_MODEL = 5
STATUS_INVALID_PREVIEW = 6
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


def _direction_guard(
  scale: float, previous_output: float | None, torque: float,
  planned_angle: float, target_angle: float, measured_angle: float, direction_fraction: float,
  dt: float, gain: float, lateral_accel_per_degree: float, torque_params,
  torque_from_lateral_accel: Callable[[float, object], float],
) -> tuple[float, float, bool]:
  """Direction guard v2 (target-referred bounded fallback). Replaces the old zero-attractor scale
  (torque *= direction_guard_scale, snapping to a raw_torque==0 special case at measured_angle==0,
  FM3.9) with a continuous mix toward a capped, target-referred fallback -- "the model wins,
  bounded" -- built from the file's own torque_from_lateral_accel machinery, re-pointed at the
  served target instead of the plan. `scale` is a mix weight (0 = no conflict, 1 = full fallback)
  low-passed at DIRECTION_GUARD_RC_S toward a continuous conflict signal (fuzzy-OR of the straddle
  and full-unwind trips, each a ramp over TURN_IN_BLEND_DEG/GUARD_TORQUE_BLEND/GUARD_UNWIND_BLEND
  instead of a boolean). R7 (FM3.5: "rule boundaries are continuous, jump < 0.05") is enforced
  algebraically on the resultant *output* against the previous frame's output, but ONLY while this
  guard itself is blending (mix > 0) -- R7 bounds the discontinuity this rule's own transition could
  introduce, it is not a general-purpose rate limiter on every torque change the controller ever
  makes. Gating it on `mix` (rather than applying it unconditionally) is what keeps the mandatory
  "bit-identical outside conflict" replay invariant true: outside a conflict mix is exactly 0, so
  guarded_torque == torque bit-for-bit and there is nothing for R7 to clamp. `previous_output` is
  supplied by the caller and should be the torque actually committed on the previous frame -- this
  function only reads it, it does not own persisting the controller's R7 baseline (the caller must
  latch that from its own final, post-downstream-clip torque, not from this function's return, so a
  saturated frame elsewhere in the pipeline can't leave a phantom baseline behind). Returns
  (guarded_torque, new_scale, direction_guarded)."""
  toward = math.copysign(1.0, target_angle) if target_angle != 0.0 else 0.0
  # reference_conflict: degrees the plan sits past zero on the wrong side of target, ramped over
  # TURN_IN_BLEND_DEG (reused, not a new width) -- continuous form of planned_angle*target_angle<0
  reference_conflict = min(max(-planned_angle * toward / TURN_IN_BLEND_DEG, 0.0), 1.0)
  torque_away_from_target = min(max(-torque * toward / GUARD_TORQUE_BLEND, 0.0), 1.0)
  straddle_conflict = reference_conflict * torque_away_from_target

  measured_sign = math.copysign(1.0, measured_angle) if measured_angle != 0.0 else 0.0
  unwind_progress = _smoothstep(direction_fraction, 1.0 - GUARD_UNWIND_BLEND, 1.0)
  torque_deepens_unwind = min(max(torque * measured_sign / GUARD_TORQUE_BLEND, 0.0), 1.0)
  unwind_conflict = unwind_progress * torque_deepens_unwind

  guard_conflict = max(straddle_conflict, unwind_conflict)  # continuous OR, replaces `or`

  raw_target_feedback = -float(torque_from_lateral_accel(
    gain * lateral_accel_per_degree * (target_angle - measured_angle), torque_params,
  ))
  bounded_target_feedback = min(max(raw_target_feedback, -GUARD_FALLBACK_TORQUE_CAP), GUARD_FALLBACK_TORQUE_CAP)
  assert GUARD_FALLBACK_TORQUE_CAP < MAX_FEEDBACK_TORQUE  # R10: checked, not implicit

  new_scale = scale + min(max(guard_conflict - scale, -dt / DIRECTION_GUARD_RC_S), dt / DIRECTION_GUARD_RC_S)
  mix = min(max(new_scale, 0.0), 1.0)
  guarded_torque = torque * (1.0 - mix) + bounded_target_feedback * mix

  if previous_output is not None and mix > 0.0:  # R7 on the OUTPUT, scoped to this rule's own blend
    guarded_torque = min(max(guarded_torque, previous_output - R7_MAX_TORQUE_STEP), previous_output + R7_MAX_TORQUE_STEP)

  return guarded_torque, new_scale, guarded_torque != torque


@dataclass(frozen=True, slots=True)
class DriverAssistLimits:
  """The four opendbc.car.hyundai.values.CarControllerParams fields _driver_assist_envelope needs,
  threaded in by LatControlRack as plain floats -- this module never imports opendbc.car.hyundai
  (owner decision, docs/BLaTv3_FAILURE_MODES.md FM4.9): the driver-allowance term crosses the
  package boundary, not the platform import. Field names match the real dataclass's own so the
  formula below reads exactly like opendbc's."""
  STEER_MAX: float
  STEER_DRIVER_ALLOWANCE: float
  STEER_DRIVER_MULTIPLIER: float
  STEER_DRIVER_FACTOR: float


def _driver_assist_envelope(driver_torque_counts: float, commanded_torque: float, limits: DriverAssistLimits) -> float:
  """Mirrors opendbc.car.lateral.apply_driver_steer_torque_limits' own driver-allowance term for
  commanded_torque's direction (opendbc_repo/opendbc/car/lateral.py:76-79): a driver already
  pushing with the controller's own live intent widens the cap toward DRIVER_ASSIST_CEILING exactly
  as far as the platform's own driver-override limiter already would; an opposing driver still
  floors at MAX_DRIVER_ASSIST_TORQUE (never less than today's fixed cap -- the real limiter is
  re-enforced for real downstream at the CAN layer regardless, R10). Pure, stateless."""
  if commanded_torque == 0.0:
    return DRIVER_ASSIST_CEILING
  direction = math.copysign(1.0, commanded_torque)
  d = driver_torque_counts * direction            # d>0: driver pushes WITH the command
  allowed = limits.STEER_MAX + (limits.STEER_DRIVER_ALLOWANCE
                                 + d * limits.STEER_DRIVER_FACTOR) * limits.STEER_DRIVER_MULTIPLIER
  allowed = max(min(limits.STEER_MAX, allowed), 0.0)          # identical clamp to the real fn
  return max(allowed / limits.STEER_MAX, MAX_DRIVER_ASSIST_TORQUE)


def _ff_taper_gate(v_ego_mps: float, target_angle_deg: float, measured_angle_deg: float,
                   plan_rate_deg_s: float, target_rate_deg_s: float, driver_pressed: bool) -> float:
  """F3 highway feedforward taper: 1.0 only on a highway-speed, near-zero-curvature hold with the
  driver's hands off, 0.0 the instant any one of the underlying conditions lifts -- each its own
  smoothstep (R7), so a turn-in, an unwind, a curve already held, or anything below highway speed
  drives at least one factor to exactly 0.0 and the caller's blend contributes nothing
  (bit-identical, R5).

  The motion gate reads the SERVED target's rate alongside the plan's: the reference-filtered target
  is what the plan is chasing, so it leads the plan into a real move and the plan's own rate alone
  leaves the taper open through the lead-in frames of a highway turn-in or unwind (route 4d/4c:
  r2_impl_F3/gate_change_scan.py, 111/114 active frames where the served rate is the larger of the
  two, every one of them at 79-129 km/h inside a real leg, the gate dropping by up to 0.62/0.76).
  Whichever rate is larger governs, so this can only ever close the gate sooner, never open it.

  A hand on the wheel closes it outright, like every other steeringPressed site in this file: while
  the driver steers, the wheel's own motion -- not a model dither -- is what the friction term reads,
  and softening feedforward there subtracts from what the driver is being helped with (route 4d:
  2692.50-2694.06 s, 134 frames at 128 km/h with the gate fully open, up to 0.091 torque of
  softening, the largest single deviation the round-1 taper produced on either route)."""
  if driver_pressed:
    return 0.0
  return (
    _smoothstep(v_ego_mps, FF_TAPER_SPEED_MPS - FF_TAPER_SPEED_BLEND_MPS, FF_TAPER_SPEED_MPS)
    * (1.0 - _smoothstep(max(abs(target_angle_deg), abs(measured_angle_deg)), 0.0, FF_TAPER_ANGLE_DEG))
    * (1.0 - _smoothstep(max(abs(plan_rate_deg_s), abs(target_rate_deg_s)),
                         FF_TAPER_RATE_DEG_S, FF_TAPER_RATE_DEG_S + FF_TAPER_RATE_BLEND_DEG_S))
  )


class RackTrajectoryController:
  def __init__(self, dt: float = DT, driver_assist_limits: DriverAssistLimits | None = None) -> None:
    self.dt = dt
    self.driver_assist_limits = driver_assist_limits  # None preserves the fixed cap for callers without a CP
    self.model = None
    self.state_mono_ns = 0
    self.inactive_frames = 0
    self.hold_angle_deg: float | None = None
    self.planner: JerkLimitedRackPlanner | None = None
    self.transition_rate_limit: float | None = None
    self.transition_acceleration_limit: float | None = None
    self.previous_planned_lateral_accel: float | None = None
    self.direction_guard_scale = 0.0  # v2: a mix weight (0 = no conflict), not a survival scale
    self.hold_topup_torque = 0.0  # FM3.14: the third torque term, torque units, always starts at exactly 0.0
    self.hold_topup_cooldown_frames = 0  # frames of fast leak + frozen growth still owed after a release
    self.release_reconcile = False  # the assist branch's R7 slew carries on after a release until caught up
    self.previous_output_torque: float | None = None  # R7 baseline; None on a fresh engage (not a rule boundary)
    self.status = STATUS_INACTIVE
    self.jerk_filter = FirstOrderFilter(0.0, 1.0 / (2.0 * math.pi * 1.2), dt)
    self.reference_filter = ReferenceFilter()
    self.preview_scheduler = PreviewScheduler()
    self.envelope_scheduler = PreviewScheduler(require_confidence=False)  # R4: proactive, G-independent
    self.envelope_open_rate = self.envelope_open_accel = self.envelope_open_jerk = 0.0  # eased state; floors at comfort
    self.rack_rate_estimator = RackRateEstimator(dt)
    self.ff_taper_filter = FirstOrderFilter(0.0, FF_TAPER_RC_S, dt)  # F3: always warm, only ever blended in when gated

  def set_model(self, model, state_mono_ns: int) -> None:
    # a dropped or invalid model frame keeps the last good plan; staleness is judged by its age
    if model is not None:
      self.model = model
    self.state_mono_ns = int(state_mono_ns)

  def hold(self) -> None:
    # inactive for a frame: keep the planned rack through a short blip, start over after a real disengage
    self.inactive_frames += 1
    self.status = STATUS_INACTIVE
    if self.inactive_frames == 1:
      self.hold_angle_deg = self.rack_rate_estimator.previous_angle_deg
    elif self.inactive_frames == INACTIVE_HOLD_FRAMES + 1:
      self.reset()

  def reset(self) -> None:
    self.planner = None
    self.transition_rate_limit = None
    self.transition_acceleration_limit = None
    self.previous_planned_lateral_accel = None
    self.direction_guard_scale = 0.0
    self.previous_output_torque = None  # a fresh engage is not a rule boundary (R7); re-seeds freely
    self.hold_topup_torque = 0.0
    self.hold_topup_cooldown_frames = 0
    self.release_reconcile = False
    self.status = STATUS_INACTIVE
    self.jerk_filter.x = 0.0
    self.reference_filter.reset()
    self.preview_scheduler.reset()
    self.envelope_scheduler.reset()
    self.envelope_open_rate = self.envelope_open_accel = self.envelope_open_jerk = 0.0
    self.rack_rate_estimator.reset()
    self.ff_taper_filter.x = 0.0

  def _invalidate(self, status: int) -> None:
    self.reset()
    self.status = status

  @staticmethod
  def _limits(speed_mps: float, response_time_s: float = RESPONSE_TIME_S) -> MotionLimits:
    speed_mph = speed_mps * 2.2369362920544
    return MotionLimits(
      float(np.interp(speed_mph, _SPEED_PROFILE_MPH, _RATE_PROFILE_DEG_S)),
      float(np.interp(speed_mph, _SPEED_PROFILE_MPH, _ACCEL_PROFILE_DEG_S2)),
      float(np.interp(speed_mph, _SPEED_PROFILE_MPH, _JERK_PROFILE_DEG_S3)),
      response_time_s,
    )

  @staticmethod
  def _iso_ceiling(speed_mps: float, VM, roll: float, comfort: MotionLimits) -> MotionLimits:
    """R4/R9/R10: the envelope's ceiling is the ISO clamp already enforced upstream on
    desired_curvature (drive_helpers.clip_curvature: MAX_LATERAL_JERK, MAX_LATERAL_ACCEL_NO_ROLL),
    not a new number -- provably no more permissive than what the scalar curvature itself could
    ever ramp to. Owner Q1 (computed with the real Palisade VM, roll 0): clears the documented
    2-4x brisker-reaction band at the island scenario's speed (2.34x comfort at 40 mph), narrowing
    to 1.37x by 70 mph -- accepted as-is, no corpus p99.9 blend."""
    speed = max(speed_mps, MIN_SPEED)
    curvature_rate = MAX_LATERAL_JERK / speed ** 2  # clip_curvature's own formula
    deg_per_curvature = 1.0 / max(abs(-VM.calc_curvature(math.radians(1.0), speed, roll)), 1e-9)
    rate_ceiling = curvature_rate * deg_per_curvature
    # the acceleration and jerk legs open by the same factor the ISO rate ceiling allows over the
    # comfort table (never below it): deriving them by dividing the rate by the response time put
    # them under the comfort tables at nearly every speed, so the rate opened and the plan could not
    # accelerate into it -- exactly the blunting the design's dissent asked to check with numbers
    opening = max(rate_ceiling / max(comfort.max_rate_deg_s, 1e-9), 1.0)
    return MotionLimits(rate_ceiling, comfort.max_acceleration_deg_s2 * opening, comfort.max_jerk_deg_s3 * opening,
                        comfort.response_time_s)

  def _horizon_opened_profile(
    self, comfort: MotionLimits, targets: Sequence[PathTarget], g_env: int, ceiling: MotionLimits,
  ) -> MotionLimits:
    """R4 proactive opening (G-independent): how far the comfort envelope opens is driven by what
    the model's own admitted horizon (envelope_scheduler.index, g_env) already implies the plan
    will need -- margined (ENVELOPE_OPEN_MARGIN) and capped at the ISO ceiling -- never by the
    current lateral acceleration or error. A bounded one-pole state per limit keeps the OPENING
    side continuous (R7): snap down the same frame the demand or the admitted horizon falls, ease
    up over one horizon step otherwise (owner Q4, ENVELOPE_EASE_UP_RC_S = HORIZON_STEP_S). The
    resulting profile feeds _motion_limits unmodified, which only ever narrows further."""
    if g_env == 0:
      required_rate = required_accel = 0.0
    else:
      admitted = targets[:g_env + 1]
      required_rate = max(abs(target.rate_deg_s) for target in admitted[1:])
      required_accel = max(
        abs(admitted[index].rate_deg_s - admitted[index - 1].rate_deg_s) / HORIZON_STEP_S
        for index in range(1, len(admitted))
      )
    raw_rate = min(max(comfort.max_rate_deg_s, required_rate * ENVELOPE_OPEN_MARGIN), ceiling.max_rate_deg_s)
    raw_accel = min(max(comfort.max_acceleration_deg_s2, required_accel * ENVELOPE_OPEN_MARGIN), ceiling.max_acceleration_deg_s2)
    raw_jerk = min(max(comfort.max_jerk_deg_s3, raw_accel / RESPONSE_TIME_S), ceiling.max_jerk_deg_s3)

    def ease(attribute: str, raw: float, floor: float) -> float:
      current = getattr(self, attribute)
      new = raw if (g_env == 0 or raw <= current) else current + (self.dt / (ENVELOPE_EASE_UP_RC_S + self.dt)) * (raw - current)
      setattr(self, attribute, new)
      return max(new, floor)

    return MotionLimits(
      ease("envelope_open_rate", raw_rate, comfort.max_rate_deg_s),
      ease("envelope_open_accel", raw_accel, comfort.max_acceleration_deg_s2),
      ease("envelope_open_jerk", raw_jerk, comfort.max_jerk_deg_s3),
      comfort.response_time_s,
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
      self.transition_rate_limit, self.transition_acceleration_limit, profile.max_jerk_deg_s3, profile.response_time_s,
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
             lat_delay: float, desired_curvature: float, applied_torque: float = 0.0) -> RackTrajectoryOutput | None:
    if not active:
      self.hold()
      return None
    if self.inactive_frames and self.planner is not None and self.hold_angle_deg is not None:
      # the wheel may have moved while the plan was held: carry the plan along with it, once
      self.planner.position_deg += float(CS.steeringAngleDeg) - self.hold_angle_deg
      self.rack_rate_estimator.previous_angle_deg = float(CS.steeringAngleDeg)
    self.inactive_frames = 0
    self.hold_angle_deg = None
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

    preview_times = self.model.action.desiredCurvaturePreviewTimes
    preview = self.model.action.desiredCurvaturePreview
    if len(preview_times) < 2 or len(preview) != len(preview_times):
      # a modeld that publishes no usable preview: there is no path to build, stock steers
      self._invalidate(STATUS_INVALID_PREVIEW)
      return None

    try:
      raw_targets = model_path_targets(
        native_times_s=self.model.velocity.t,
        velocities_x=self.model.velocity.x,
        preview_times_s=preview_times,
        preview_curvatures=preview,
        scalar_curvature=float(desired_curvature),
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
      bounded_curvature = min(max(raw_target.curvature, minimum), maximum)
      limited = bounded_curvature != raw_target.curvature
      if not limited:
        angle_curvature = -VM.calc_curvature(
          math.radians(raw_target.angle_deg - params.angleOffsetDeg), bound_speed, params.roll,
        )
        bounded_curvature = min(max(angle_curvature, minimum), maximum)
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
      target_speed = float(CS.vEgo) if offset == 0.0 else raw_target.speed_mps
      bounded_target, target_limited = bound_target(raw_target, target_speed)
      targets.append(bounded_target)
      target_limits.append(target_limited)

    bound_speed = max(float(CS.vEgo), MIN_SPEED)
    minimum_curvature = max(-MAX_CURVATURE, (-MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation) / bound_speed ** 2)
    maximum_curvature = min(MAX_CURVATURE, (MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation) / bound_speed ** 2)
    measured_curvature = -VM.calc_curvature(
      math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), bound_speed, params.roll,
    )
    measured_out_of_bounds = not minimum_curvature - 1e-9 <= measured_curvature <= maximum_curvature + 1e-9
    # the driver's hands, a lane change or a limited immediate target pin the preview at the action time
    lane_changing = str(self.model.meta.laneChangeState) in ("laneChangeStarting", "laneChangeFinishing")
    forced = bool(CS.steeringPressed) or lane_changing or target_limits[0] or measured_out_of_bounds
    self.preview_scheduler.update(self.model, int(self.model.timestampEof), float(preview_times[0]), targets, forced)
    # R4 envelope scheduler: same gates and hysteresis, confidence-free (docs/BLaTv3_FAILURE_MODES.md R4)
    self.envelope_scheduler.update(self.model, int(self.model.timestampEof), float(preview_times[0]), targets, forced)
    preview_s = self.preview_scheduler.preview_s
    target = targets[0]
    path_limited = target_limits[0]
    previous_angle_deg = self.rack_rate_estimator.previous_angle_deg
    measured_rate, measured_rate_valid = self.rack_rate_estimator.update(float(CS.steeringAngleDeg), float(CS.steeringRateDeg))
    # a far, steady target may be approached more slowly
    profile = self._limits(float(CS.vEgo), RESPONSE_TIME_S + RESPONSE_TIME_PREVIEW_S * preview_s / HORIZON_S)
    if self.planner is None:
      seed_rate = _clip(measured_rate, profile.max_rate_deg_s) if measured_rate_valid else 0.0
      self.planner = JerkLimitedRackPlanner(float(CS.steeringAngleDeg), seed_rate)
      self.reference_filter.reset()
    elif CS.steeringPressed and previous_angle_deg is not None:
      self.planner.position_deg += float(CS.steeringAngleDeg) - previous_angle_deg
    if 0.0 < abs(self.planner.rate_deg_s) - profile.max_rate_deg_s <= 1e-6:
      self.planner.rate_deg_s = math.copysign(profile.max_rate_deg_s, self.planner.rate_deg_s)
    if 0.0 < abs(self.planner.acceleration_deg_s2) - profile.max_acceleration_deg_s2 <= 1e-6:
      self.planner.acceleration_deg_s2 = math.copysign(profile.max_acceleration_deg_s2, self.planner.acceleration_deg_s2)
    ceiling = self._iso_ceiling(float(CS.vEgo), VM, float(params.roll), profile)
    opened_profile = self._horizon_opened_profile(profile, targets, self.envelope_scheduler.index, ceiling)
    limits, profile_transition = self._motion_limits(opened_profile)  # opened profile feeds the ratchet, never the reverse (R4/R10)
    filtered_target = self.reference_filter.update(
      RackTarget(target.angle_deg, target.rate_deg_s), reference_trail_limit_deg(VM, CS.vEgo), self.dt,
      path_limited or measured_out_of_bounds or profile_transition,
      REFERENCE_FILTER_RC_S + REFERENCE_FILTER_PREVIEW_RC_S * preview_s / HORIZON_S,
    )
    planner = self.planner
    assert planner is not None
    timed_targets = tuple(
      (offset, RackTarget(path_target.angle_deg, path_target.rate_deg_s))
      for offset, path_target in zip(HORIZON_OFFSETS_S, targets, strict=True)
      if offset > 0.0
    )
    desired_acceleration = self._recovery_acceleration(profile, profile_transition)
    try:
      if desired_acceleration is None:
        fitted_acceleration = horizon_desired_acceleration(planner, timed_targets)
        natural_frequency = 2.0 / limits.response_time_s
        reactive_acceleration = (
          natural_frequency ** 2 * (filtered_target.position_deg - planner.position_deg)
          + 2.0 * natural_frequency * (filtered_target.rate_deg_s - planner.rate_deg_s)
        )
        horizon_acceleration = reactive_acceleration + HORIZON_ACCELERATION_BLEND * (
          fitted_acceleration - reactive_acceleration
        )

        def preview(acceleration_override: float | None) -> RackPlan:
          candidate = JerkLimitedRackPlanner(planner.position_deg, planner.rate_deg_s)
          candidate.acceleration_deg_s2 = planner.acceleration_deg_s2
          return candidate.update(filtered_target, limits, self.dt, acceleration_override)

        baseline = preview(None)
        horizon = preview(horizon_acceleration)
        if horizon_candidate_preserves_immediate_path(planner.position_deg, filtered_target, baseline, horizon):
          desired_acceleration = horizon_acceleration
      raw_plan = planner.update(
        filtered_target, limits, self.dt, desired_acceleration,
      )
    except ValueError:
      self._invalidate(STATUS_INVALID_PLANNER_STATE)
      return None
    raw_planned_curvature = -VM.calc_curvature(math.radians(raw_plan.position_deg - params.angleOffsetDeg), CS.vEgo, params.roll)
    planned_out_of_bounds = not minimum_curvature - 1e-9 <= raw_planned_curvature <= maximum_curvature + 1e-9
    plan = raw_plan
    if planned_out_of_bounds:
      planned_curvature = min(max(raw_planned_curvature, minimum_curvature), maximum_curvature)
      planned_angle = math.degrees(VM.get_steer_from_curvature(-planned_curvature, bound_speed, params.roll)) + params.angleOffsetDeg
      plan = RackPlan(planned_angle, 0.0, 0.0, True, raw_plan.acceleration_limited, raw_plan.jerk_limited)
    planned_curvature = -VM.calc_curvature(math.radians(plan.position_deg - params.angleOffsetDeg), CS.vEgo, params.roll)
    planned_lateral_accel = planned_curvature * CS.vEgo ** 2
    measured_lateral_accel = measured_curvature * CS.vEgo ** 2
    target_angle = filtered_target.position_deg - params.angleOffsetDeg
    measured_angle = float(CS.steeringAngleDeg) - params.angleOffsetDeg
    target_motion = target_angle - measured_angle + RESPONSE_TIME_S * filtered_target.rate_deg_s
    # how much of a turn-in this frame is, continuously: the wheel on (or near) the target's side, the
    # target beyond it, and the motion demanded toward it -- each condition a ramp, not a test (R7)
    toward = math.copysign(1.0, target_angle) if target_angle != 0.0 else 0.0
    turn_in_fraction = (
      (1.0 - min(max(-measured_angle * toward / TURN_IN_BLEND_DEG, 0.0), 1.0))
      * min(max((abs(target_angle) - abs(measured_angle)) / TURN_IN_BLEND_DEG, 0.0), 1.0)
      * min(max(target_motion * toward / TURN_IN_BLEND_DEG, 0.0), 1.0)
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
    filtered_curvature = -VM.calc_curvature(
      math.radians(filtered_target.position_deg - params.angleOffsetDeg), CS.vEgo, params.roll,
    )
    filtered_lateral_accel = filtered_curvature * CS.vEgo ** 2
    trajectory_feedforward_lateral_accel = planned_lateral_accel
    if target_lateral_accel * planned_lateral_accel <= 0.0:
      trajectory_feedforward_lateral_accel = 0.0
    elif (filtered_lateral_accel * planned_lateral_accel > 0.0
          and abs(filtered_lateral_accel) < abs(planned_lateral_accel)):
      trajectory_feedforward_lateral_accel = filtered_lateral_accel
    planned_angle = plan.position_deg - params.angleOffsetDeg
    intended_angle = measured_angle + target_motion
    # how far past what is still needed the wheel already is, signed and continuous: +1 a pure unwind
    # (neither the plan nor the commanded motion holds any of the current angle), 0 exactly at the need
    # (bit-identical to no relaxation), negative a turn-in still short of the need. Generalizes the
    # retired unwind magnitude clamp: instead of scaling the whole request, only the rate feedback
    # relaxes, so a return the rack's own self-aligning torque is already producing is not resisted.
    direction_fraction = 0.0
    if measured_angle != 0.0:
      planned_hold_angle = abs(planned_angle) if planned_angle * measured_angle > 0.0 else 0.0
      turn_in_angle = abs(intended_angle) if intended_angle * measured_angle > 0.0 else 0.0
      # faded out within TURN_IN_BLEND_DEG of center: a near-center dither must not pin the fraction
      # at its endpoints and strip the rate damping (or arm the guard) frame to frame (R7)
      direction_fraction = min(max(
        1.0 - max(planned_hold_angle, turn_in_angle) / abs(measured_angle), -1.0), 1.0,
      ) * min(abs(measured_angle) / TURN_IN_BLEND_DEG, 1.0)
    feedforward_lateral_accel = (
      trajectory_feedforward_lateral_accel - params.roll * ACCELERATION_DUE_TO_GRAVITY - torque_params.latAccelOffset + friction
    )
    # F3 highway feedforward taper -- see the FF_TAPER_* block comment above for the mechanism and data.
    filtered_ff_lateral_accel = float(self.ff_taper_filter.update(feedforward_lateral_accel))
    ff_taper_gate = _ff_taper_gate(float(CS.vEgo), target_angle, measured_angle, plan.rate_deg_s,
                                    filtered_target.rate_deg_s, bool(CS.steeringPressed))
    if ff_taper_gate != 0.0:  # exact 0.0 whenever any gate is fully closed: bit-identical to today's code (R7)
      feedforward_lateral_accel += ff_taper_gate * (filtered_ff_lateral_accel - feedforward_lateral_accel)
    feedforward_torque = -float(torque_from_lateral_accel(feedforward_lateral_accel, torque_params))

    # feedback keeps authority at standstill: the per-degree gain uses the floored speed (creep must correct)
    curvature_per_degree = -VM.calc_curvature(math.radians(1.0), bound_speed, 0.0)
    lateral_accel_per_degree = curvature_per_degree * bound_speed ** 2
    gain = self._feedback_gain(float(CS.vEgo))
    position_feedback = -float(torque_from_lateral_accel(
      gain * lateral_accel_per_degree * (plan.position_deg - CS.steeringAngleDeg), torque_params,
    ))
    rate_gain_scale = _smoothstep(-direction_fraction, -1.0, 0.0)
    rate_feedback = -float(torque_from_lateral_accel(
      gain * rate_gain_scale * lateral_accel_per_degree * RATE_HORIZON_S * (plan.rate_deg_s - measured_rate), torque_params,
    )) if measured_rate_valid else 0.0
    raw_feedback = position_feedback + rate_feedback
    turn_in_cap = MAX_FEEDBACK_TORQUE + turn_in_fraction * (MAX_TURN_IN_FEEDBACK_TORQUE - MAX_FEEDBACK_TORQUE)
    feedback_lower = -turn_in_cap if target_angle < 0.0 else -MAX_FEEDBACK_TORQUE
    feedback_upper = turn_in_cap if target_angle > 0.0 else MAX_FEEDBACK_TORQUE
    feedback = min(max(raw_feedback, feedback_lower), feedback_upper)
    feedback_limited = feedback != raw_feedback
    # the hold top-up applied cold from last frame's fully resolved state; this frame's growth is decided
    # below, after the platform clip, the guard and the driver-assist envelope have all had their say
    topup_applied = self.hold_topup_torque
    raw_torque = feedforward_torque + feedback + topup_applied
    # FM3.9 retired: the old measured_angle==0.0 exact-zero special case is now covered by the
    # continuous direction guard below -- a target-referred bounded fallback, not a zero spike.
    # The phase-3 slew-aware early release is RETIRED: in the field (route 2d, six owner bookmarks)
    # it latched during ordinary sustained curves -- the rate-flip test reads a visible curve exit as
    # a coming reversal, entry opened at benign wheel-past-target tracking dither, and the latch had
    # no exit when the target climbed back beyond the wheel -- shedding holding torque mid-curve
    # (output 0.44 -> 0.05 in 1 s while the curve tightened; the car ran wide). The closed-loop plant
    # A/B had already shown the mechanism binding almost never (0-0.8 % duty) with no measurable
    # reversal-lag benefit, so it is removed rather than re-gated; the applied-torque plumbing and
    # the earlyRelease log field remain for a future redesign against a true torque-reversal test.
    early_release = False
    torque = min(max(raw_torque, -1.0), 1.0)
    platform_saturated = torque != raw_torque
    # Direction guard v2: continuously blends torque toward a capped, target-referred fallback
    # instead of scaling it toward zero (FM3.5/FM3.9/R7/R10; docs/BLaTv3_FAILURE_MODES.md).
    torque, self.direction_guard_scale, direction_guarded = _direction_guard(
      self.direction_guard_scale, self.previous_output_torque, torque,
      planned_angle, target_angle, measured_angle, direction_fraction,
      self.dt, gain, lateral_accel_per_degree, torque_params, torque_from_lateral_accel,
    )
    motion_limited = (
      plan.rate_limited or plan.acceleration_limited or plan.jerk_limited
      or measured_out_of_bounds or planned_out_of_bounds or self.reference_filter.limited
    )
    if not all(math.isfinite(value) for value in (
      torque, plan.position_deg, plan.rate_deg_s, plan.acceleration_deg_s2, measured_rate,
      lateral_accel_error, position_feedback, rate_feedback, feedforward_torque, topup_applied,
    )):
      self._invalidate(STATUS_INVALID_OUTPUT)
      return None
    driver_assist_limited = False
    driver_assist_cap = DRIVER_ASSIST_CEILING
    if CS.steeringPressed:
      # Agreement relaxation (docs/BLaTv3_FAILURE_MODES.md FM4.9): a driver pushing with the
      # controller's own live intent widens the cap toward 1.0, exactly as far as the platform's
      # own driver-allowance limiter already would; an opposing driver still floors at
      # MAX_DRIVER_ASSIST_TORQUE. Falls back to the old fixed cap for callers with no CP (tests,
      # any future caller that hasn't threaded driver_assist_limits through).
      driver_assist_cap = (
        _driver_assist_envelope(CS.steeringTorque, torque, self.driver_assist_limits)
        if self.driver_assist_limits is not None else MAX_DRIVER_ASSIST_TORQUE
      )
      assisted_torque = _clip(torque, driver_assist_cap)
      if self.previous_output_torque is not None:  # same R7 idiom as _direction_guard, scoped to this branch
        assisted_torque = min(max(assisted_torque, self.previous_output_torque - R7_MAX_TORQUE_STEP),
                               self.previous_output_torque + R7_MAX_TORQUE_STEP)
      driver_assist_limited = assisted_torque != torque
      torque = assisted_torque
      self.release_reconcile = True
    elif self.release_reconcile and self.previous_output_torque is not None:
      # The hand comes off: the branch above has slewed the committed torque toward the cap, and the
      # composed request (the hold top-up's carried-in value included -- the fast leak only starts
      # this frame) may sit a full step or more away. Keep the same R7 slew until the request is
      # within a step of what was committed, then hand over cleanly (FM3.14 review: an unclamped
      # release frame jumped by the term's whole value; the pre-existing gap was smaller, same shape).
      reconciled_torque = min(max(torque, self.previous_output_torque - R7_MAX_TORQUE_STEP),
                              self.previous_output_torque + R7_MAX_TORQUE_STEP)
      self.release_reconcile = reconciled_torque != torque
      driver_assist_limited = self.release_reconcile
      torque = reconciled_torque
    # R7's baseline is the torque actually committed this frame, taken after the driver-assist clip
    # (and its own backstop above) -- not the guard's own pre-clip value -- so a saturated hand-off
    # can't leave a phantom-high baseline that forces an unwanted high-torque hold the instant the
    # driver releases the wheel.
    self.previous_output_torque = torque
    # Hold top-up, deferred write (FM3.14): grow only while the wheel is meant to be steady and nothing
    # upstream already binds in the error's own direction -- the platform clip and the feedback cap flags
    # above are the real applied-composition flags (last frame's top-up included), and the guard mix is this
    # frame's resolved value, not a stale one. A press freezes growth and selects the fast leak, and the
    # cooldown keeps both for HOLD_TOPUP_RELEASE_COOLDOWN_S after the hand comes off, so a brief grab can't
    # hand a barely-decayed residual back to the slow leak. Stock fallback never reaches here (reset() ran).
    in_release_cooldown = not CS.steeringPressed and self.hold_topup_cooldown_frames > 0
    if CS.steeringPressed:
      self.hold_topup_cooldown_frames = round(HOLD_TOPUP_RELEASE_COOLDOWN_S / self.dt)
    elif in_release_cooldown:
      self.hold_topup_cooldown_frames -= 1
    position_error_deg = plan.position_deg - float(CS.steeringAngleDeg)
    error_sign = math.copysign(1.0, position_error_deg) if position_error_deg != 0.0 else 0.0
    # a shortfall is only worth making up when the plan and the served target both lie on the same side
    # of the wheel: near center at speed the plan sits half a degree one way and the target the other,
    # and a term chasing the plan there pushes against the target (replay: the guard engaged twice as
    # often, at a mix too small to correct it). A push the target does not want drains at the fast rate,
    # as does a residual the current error already opposes -- a stale push never outlives ~0.3 s.
    aligned = position_error_deg * (target_angle - measured_angle) > 0.0
    reversed_residual = self.hold_topup_torque * position_error_deg < 0.0
    platform_bind = platform_saturated and error_sign != 0.0 and math.copysign(1.0, raw_torque) == error_sign
    feedback_bind = feedback_limited and error_sign != 0.0 and math.copysign(1.0, feedback) == error_sign
    accumulating = (
      aligned and not CS.steeringPressed and not in_release_cooldown and self.direction_guard_scale <= 0.0
      and not platform_bind and not feedback_bind
    )
    plan_rate_gate = 1.0 - _smoothstep(
      abs(plan.rate_deg_s), HOLD_TOPUP_PLAN_RATE_DEG_S, HOLD_TOPUP_PLAN_RATE_DEG_S + HOLD_TOPUP_PLAN_RATE_BLEND_DEG_S,
    )
    measured_rate_gate = (1.0 - _smoothstep(
      abs(measured_rate), HOLD_TOPUP_MEASURED_RATE_DEG_S, HOLD_TOPUP_MEASURED_RATE_DEG_S + HOLD_TOPUP_MEASURED_RATE_BLEND_DEG_S,
    )) if measured_rate_valid else 0.0
    approach_gate = (1.0 - _smoothstep(
      measured_rate * error_sign, HOLD_TOPUP_APPROACH_RATE_DEG_S, HOLD_TOPUP_APPROACH_RATE_DEG_S + HOLD_TOPUP_APPROACH_BLEND_DEG_S,
    )) if measured_rate_valid else 0.0
    # none against a wanted unwind (the rate feedback's own relaxation), none while the plan itself is moving,
    # none while the wheel is swinging, none while the wheel is already visibly closing on the plan. Deliberately
    # NOT the turn-in fraction: a wheel standing short of a static plan and target is exactly the standing
    # shortfall, and the turn-in fraction reads any wheel more than TURN_IN_BLEND_DEG short of the target as a
    # turn-in -- gated on it the term never acted on the route 0x3e window it exists for. What separates an
    # active turn-in from a standing shortfall is motion: the plan's, the wheel's, or the wheel's approach.
    steady_gate = rate_gain_scale * plan_rate_gate * measured_rate_gate * approach_gate
    self.hold_topup_torque = _hold_topup_step(
      self.hold_topup_torque, position_error_deg, steady_gate, accumulating,
      bool(CS.steeringPressed) or in_release_cooldown or not aligned or reversed_residual, self.dt,
    )
    self.status = STATUS_ACTIVE
    return RackTrajectoryOutput(
      torque=torque,
      target_curvature=target.curvature,
      target_angle_deg=filtered_target.position_deg,
      target_rate_deg_s=filtered_target.rate_deg_s,
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
      torque_limited=platform_saturated or direction_guarded or driver_assist_limited,
      rate_limit_deg_s=limits.max_rate_deg_s,
      acceleration_limit_deg_s2=limits.max_acceleration_deg_s2,
      jerk_limit_deg_s3=limits.max_jerk_deg_s3,
      profile_transition=profile_transition,
      path_limited=path_limited,
      infeasible=motion_limited or feedback_limited or platform_saturated or direction_guarded or driver_assist_limited
      or profile_transition or path_limited,
      saturated=platform_saturated,
      direction_guarded=direction_guarded,
      driver_assist_limited=driver_assist_limited,
      driver_assist_cap=driver_assist_cap,
      early_release=early_release,
      direction_fraction=direction_fraction,
      preview_time_s=preview_s,
      reference_limited=self.reference_filter.limited,
      near_target_angle_deg=target.angle_deg,
      # the floored (effective) values -- what _motion_limits actually received -- not the raw
      # internal one-pole state, which the jerk leg in particular can transiently sit below
      envelope_open_rate_deg_s=opened_profile.max_rate_deg_s,
      envelope_open_acceleration_deg_s2=opened_profile.max_acceleration_deg_s2,
      envelope_open_jerk_deg_s3=opened_profile.max_jerk_deg_s3,
      envelope_preview_time_s=self.envelope_scheduler.preview_s,
      hold_topup_torque=topup_applied,  # this frame's applied contribution, already inside `torque`
      hold_topup_growing=accumulating,
    )
