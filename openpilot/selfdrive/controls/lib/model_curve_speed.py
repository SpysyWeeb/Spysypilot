"""Curve longitudinal policy.

Two layers decide how the car should accelerate through the curves the model path shows and the one it is in:

* anticipation -- every path node gets the speed at which the steering demand there stays inside the torque budget and
  the lateral acceleration inside the owner's comfort; the candidate is the kinematic acceleration that meets the
  strictest node at its distance: small and positive near a limit, the needed deceleration beyond it;
* reaction -- the measured steering state (torque, tracking error) says what the car is doing right now: heavy but
  tracking, take the foot off (coast); pinned and understeering, brake to the speed that restores margin.

The steering authority the anticipation counts on is calibrated online from the steering's own share of the lateral
acceleration per unit of torque (the ground-plane value the rack reports, less the bank the limit adds back), measured
only in real corners at speed and bounded around the torque tuning's own factor.

The result is one plan candidate. The planner's arbitration can only lower the chosen acceleration with it, never raise
it, and the candidate is never positive while the steering is heavy. Nothing here touches the cruise target, the stop
bit or the mode.
"""
from dataclasses import dataclass
import math

import numpy as np

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants

# Steering authority and comfort.
TORQUE_BUDGET = 0.90          # of the EPS limit, the demand a path node may need at the speed the car will have there
A_LAT_COMFORT = 3.4           # m/s^2, deliberately above the owner's own turns (manual archive max 2.86, route 22 sweeper) so the
                              # calibrated steering authority is the ceiling that binds (owner ruling 2026-08-31: push closer to the
                              # limit, hold steady); comfort remains the backstop against an implausible learned authority
MIN_CURVATURE = 1e-4          # 1/m, straighter than this is a straight
MIN_MODEL_SPEED = 1.0         # m/s, avoids unstable curvature near predicted stops

# Online authority calibration: lateral acceleration achieved per unit of torque above friction.
AUTHORITY_RC = 5.0            # s, filter on the measured ratio
AUTHORITY_MIN_TORQUE = 0.40   # measure only when the steering is working ...
AUTHORITY_MAX_ERROR = 0.30    # m/s^2, ... and tracking ...
AUTHORITY_MIN_LATERAL = 1.0   # m/s^2, ... in a real corner (the friction term dominates the ratio below this) ...
AUTHORITY_MIN_SPEED = 10.0    # m/s, ... at a speed where the assist resembles the highway's: town corners at 5-8 m/s
                              # dragged the factor to 0.9x the tuning and a bend 300 s later was braked for (route 0x4c)
AUTHORITY_BOUNDS = (0.8, 1.8) # times the torque tuning's own factor
ROLL_HORIZON_S = 2.0          # s of travel over which the car's live roll still describes the road: farther path nodes get
                              # no bank either way. Route 0x4c: the approach's crown was charged to a node 143 m ahead that
                              # is banked the other way, and the limit there read 19.7 m/s instead of 25

# Anticipation.
T_APPROACH = 1.5              # s, proportional approach to a limit that is near or already reached
D_MIN = 1.0                   # m, shortest distance the kinematic candidate divides by
A_CURVE_MIN = -2.0            # m/s^2, comfort floor; harder braking is the lead and stop logic's business
A_CURVE_FREE = 4.0            # m/s^2, a candidate above this has nothing to say
V_HOLD_BAND = 0.3             # m/s, at the limit the candidate is a flat zero: the settled car sits still instead of
                              # stitching gas/brake corrections across the zero crossing; the band edges correct the drift
J_DOWN = 2.0                  # m/s^3, how fast the candidate may pull the acceleration down ...
J_UP = 3.0                    # m/s^3, ... and let it back up (curves release in about a second, not at 0.2 m/s^2)

# Reaction to the measured steering state.
T_COAST = 0.85                # torque above which the foot comes off, tracking or not ...
T_COAST_EXIT = 0.75           # ... and below which coasting ends (hysteresis)
COAST_ENTER_S = 0.3           # s, the torque must stay heavy this long before coasting starts (no breathing on a sweeper)
COAST_EXIT_S = 0.5            # s, coasting persists this long after the torque has dropped
T_PIN = 0.95                  # torque at which the steering is pinned (the field audit's own definition)
E_TRACK = 0.30                # m/s^2, understeer above which a pinned car is losing the line
E_TRACK_EXIT = 0.20           # m/s^2, understeer below which the brake regime hands back to coasting
BRAKE_ENTER_S = 0.3           # s, the loss must persist this long before the brake regime enters
T_RESTORE = 1.0               # s, time in which the brake regime aims to restore the torque margin
V_REACT_MIN = 3.0             # m/s, below this the reaction layer never brakes: measured curvature is noise there
CURVE_GAS_GRACE_S = 5.0       # s after a driver gas override in which the anticipation may hold but never brake: the owner
                              # pushed past the curve's limit on purpose (route 0x2c t=885: the still-active episode pulled the
                              # exit back to -1.0 mid-corner after the pedal was released); the reaction brake regime still runs

# The hold after a lift. Leaving the coast or brake regime used to hand the plan straight back to whichever candidate
# wanted to accelerate, and that candidate drove the car back into heavy steering: route 0x33 t=2524-2544, seven lifts in
# one 20 s bend, the request a square wave between +0.75 and the coast. Once the steering has said the curve is at its
# limit, the candidate holds zero for the rest of the bend instead
BEND_OPEN_A_LAT = 1.0         # m/s^2, measured lateral acceleration below which the bend has ended ...
BEND_OPEN_S = 1.0             # s, ... for this long: the crossover of an S does not release the hold
HOLD_MAX_S = 30.0             # s, the hold ends regardless. A backstop, not a release path: the open test reads the live
                              # measurement whenever the hold clamps, so it cannot be fooled by a frozen reading; a bend
                              # longer than this is held at the lifted speed for the rest of it (route 0x33's is 20 s)

REGIME_FREE, REGIME_ANTICIPATE, REGIME_COAST, REGIME_BRAKE = 'free', 'anticipate', 'coast', 'brake'


@dataclass
class LateralState:
  # the measured steering state, from either the stock torque controller or combo's rack controller
  active: bool = False
  torque: float = 0.0          # |output|, 0..1 of the EPS limit
  error: float = 0.0           # desired - actual lateral acceleration, m/s^2, signed as the controller reports it
  actual_lateral_accel: float = 0.0
  desired_lateral_accel: float = 0.0
  pinned: bool = False         # saturated or torque limited, as the controller reports it

  @property
  def understeer(self):
    # positive when the car turns less than asked, whichever way the curve goes; negative on an exit overshoot
    return math.copysign(1.0, self.desired_lateral_accel) * self.error

  @classmethod
  def from_controls_state(cls, controls_state):
    try:
      union = controls_state.lateralControlState
      kind = union.which()
      if kind not in ('torqueState', 'rackState'):
        return cls()
      st = getattr(union, kind)
      values = (float(st.output), float(st.error), float(st.actualLateralAccel), float(st.desiredLateralAccel))
    except (AttributeError, TypeError, ValueError, OverflowError):
      return cls()
    if not all(math.isfinite(v) for v in values):
      return cls()
    pinned = bool(st.saturated) or (kind == 'rackState' and bool(st.torqueLimited))
    return cls(bool(st.active), min(abs(values[0]), 1.0), values[1], values[2], values[3], pinned)


@dataclass
class CurveResult:
  a_target: float | None = None   # the plan candidate, None when the policy has nothing to say
  regime: str = REGIME_FREE
  v_limit: float = math.inf       # the strictest node's speed limit, for logging and tests
  distance: float = 0.0           # its distance along the path
  authority_factor: float = 0.0   # lateral acceleration per unit of torque the policy currently counts on
  holding: bool = False           # the hold after a lift is on: no acceleration until the bend reads open


def _median_filter_three(values):
  """Three-sample spatial median, including full windows at both edges."""
  filtered = np.empty_like(values)
  filtered[1:-1] = np.median(np.vstack((values[:-2], values[1:-1], values[2:])), axis=0)
  filtered[0] = np.median(values[:3])
  filtered[-1] = np.median(values[-3:])
  return filtered


def _torque_values(params):
  try:
    values = (float(params.latAccelFactorFiltered), float(params.latAccelOffsetFiltered),
              float(params.frictionCoefficientFiltered))
  except (AttributeError, TypeError, ValueError, OverflowError):
    try:
      values = (float(params.latAccelFactor), float(params.latAccelOffset), float(params.friction))
    except (AttributeError, TypeError, ValueError, OverflowError):
      return None
  return values if np.all(np.isfinite(values)) and values[0] > 0.0 and 0.0 <= values[2] < TORQUE_BUDGET else None


def curve_speed_limits(signed_curvature, torque_params, roll, lateral_active=True):
  """Per-node speed limits: the lower of the steering authority at the torque budget and the comfort lateral acceleration.

  torque_params is (lateral acceleration per unit torque, offset, friction); the authority applies only while openpilot steers.
  roll is a scalar or one value per node (the live roll near the car, zero beyond ROLL_HORIZON_S)."""
  curvature = np.abs(signed_curvature)
  comfort = np.where(curvature >= MIN_CURVATURE, np.sqrt(A_LAT_COMFORT / np.maximum(curvature, MIN_CURVATURE)), np.inf)
  if torque_params is None or not lateral_active:
    return comfort
  factor, offset, friction = torque_params
  bias = roll * ACCELERATION_DUE_TO_GRAVITY + offset
  margin = (TORQUE_BUDGET - friction) * factor
  available = np.sign(signed_curvature) * margin + bias          # lateral acceleration the budget can hold in this direction
  authority_sq = np.divide(available, signed_curvature, out=np.full_like(signed_curvature, np.inf),
                           where=curvature >= MIN_CURVATURE)
  authority = np.sqrt(np.maximum(authority_sq, 0.0))
  return np.minimum(comfort, authority)


class ModelCurveSpeedLimiter:
  """Curve longitudinal policy as a plan candidate (the class keeps its name for the planner wiring)."""

  def __init__(self, CP=None, dt=DT_MDL):
    self.dt = dt
    self.response_time = dt
    try:
      actuator_delay = float(getattr(CP, "longitudinalActuatorDelay", 0.0))
      if math.isfinite(actuator_delay) and actuator_delay >= 0.0:
        self.response_time += actuator_delay
    except (AttributeError, TypeError, ValueError, OverflowError):
      pass
    self.torque_params = None
    lateral_tuning = getattr(CP, "lateralTuning", None)
    if lateral_tuning is not None and lateral_tuning.which() == "torque":
      self.torque_params = _torque_values(lateral_tuning.torque)
    self.authority = None                        # FirstOrderFilter on the measured lateral acceleration per unit torque
    self.v_limit = math.inf
    self.distance = 0.0
    self._gas_grace_s = 0.0
    self.reset()

  def reset(self):
    # the car is not under our longitudinal control: no regime, no hold and no jerk anchor carries into the next engagement.
    # The calibrated authority and the gas grace are about the car and the driver, not the engagement, and stay
    self._candidate_history = [A_CURVE_FREE] * 3
    self._lateral_history = [0.0] * 3
    self._output = A_CURVE_FREE
    self._coast_enter_s = 0.0
    self._coast_exit_s = 0.0
    self._losing_s = 0.0
    self.regime = REGIME_FREE
    self.active = False
    self._holding = False
    self._hold_s = 0.0
    self._open_s = 0.0

  def _calibrated(self, params, state, v_ego, lateral_active, roll):
    # the torque tuning's factor is the prior; the measured ratio moves it inside AUTHORITY_BOUNDS while the steering
    # is working and tracking through a real corner at speed. The measurement is the steering's own share of the
    # lateral acceleration: the rack reports the ground-plane value, and the bank's share is the same bias the limit
    # adds back (route 0x4d: sweepers banked 3-4 deg railed the factor to 1.6x the tuning)
    if params is None:
      return None
    factor, offset, friction = params
    if self.authority is None:
      self.authority = FirstOrderFilter(factor, AUTHORITY_RC, self.dt)
    share = abs(state.actual_lateral_accel - (roll * ACCELERATION_DUE_TO_GRAVITY + offset))
    if (lateral_active and state.active and v_ego >= AUTHORITY_MIN_SPEED and state.torque >= AUTHORITY_MIN_TORQUE
        and share >= AUTHORITY_MIN_LATERAL and abs(state.error) <= AUTHORITY_MAX_ERROR and state.torque > friction + 0.05):
      measured = share / (state.torque - friction)
      self.authority.update(float(np.clip(measured, AUTHORITY_BOUNDS[0] * factor, AUTHORITY_BOUNDS[1] * factor)))
    return (float(np.clip(self.authority.x, AUTHORITY_BOUNDS[0] * factor, AUTHORITY_BOUNDS[1] * factor)), offset, friction)

  def _anticipate(self, model, v_ego, roll, params, lateral_active):
    try:
      position_x = np.asarray(model.position.x, dtype=float)
      position_y = np.asarray(model.position.y, dtype=float)
      velocity_x = np.asarray(model.velocity.x, dtype=float)
      yaw_rate = np.asarray(model.orientationRate.z, dtype=float)
    except (AttributeError, TypeError, ValueError, OverflowError):
      return None
    expected_shape = (ModelConstants.IDX_N,)
    if any(a.shape != expected_shape for a in (position_x, position_y, velocity_x, yaw_rate)):
      return None
    if not all(np.all(np.isfinite(a)) for a in (position_x, position_y, velocity_x, yaw_rate)):
      return None

    path_distance = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(position_x), np.diff(position_y)))))
    signed_curvature = _median_filter_three(yaw_rate / np.maximum(np.abs(velocity_x), MIN_MODEL_SPEED))
    node_roll = np.where(path_distance <= v_ego * ROLL_HORIZON_S, roll, 0.0)
    limits = curve_speed_limits(signed_curvature, params, node_roll, lateral_active)

    # per node, the less demanding of the kinematic acceleration that meets its limit at its distance and a proportional
    # approach: far limits are kinematic, near or reached ones proportional, and the two meet continuously. Small and
    # positive near a limit, the needed deceleration beyond it, never a burst toward it
    effective_distance = np.maximum(path_distance - v_ego * self.response_time, D_MIN)
    kinematic = (limits ** 2 - v_ego ** 2) / (2.0 * effective_distance)
    proportional = (limits - v_ego) / T_APPROACH
    per_node = np.maximum(kinematic, proportional)
    finite = np.isfinite(per_node)
    if not np.any(finite):
      self.v_limit = math.inf
      self.distance = 0.0
      return A_CURVE_FREE
    idx = int(np.argmin(np.where(finite, per_node, np.inf)))
    self.v_limit = float(limits[idx])
    self.distance = float(path_distance[idx])
    chosen = float(per_node[idx])
    # the hold band (owner ruling 2026-08-31): riding the limit means holding it, not correcting around it
    if math.isfinite(self.v_limit) and abs(v_ego - self.v_limit) <= V_HOLD_BAND:
      chosen = 0.0
    return chosen

  def _react(self, state, v_ego, accel_coast, params, roll):
    # the regime machine: free -> coast when the torque is heavy, coast -> brake when pinned and understeering, and back;
    # every entry and exit is dwelled so a single sample never switches it. Coasting does not wait for the tracking
    # error: heavy and understeering below the pin (route 25 t=216, 0.94 torque, 0.33 m/s^2 wide) still wants the foot off
    heavy = state.torque >= T_COAST
    pinned = state.pinned or state.torque >= T_PIN
    understeer = state.understeer
    if self.regime == REGIME_BRAKE and (understeer < E_TRACK_EXIT or not pinned):
      self.regime = REGIME_COAST
      self._losing_s = 0.0
    if self.regime != REGIME_BRAKE:
      self._losing_s = self._losing_s + self.dt if (pinned and understeer >= E_TRACK) else 0.0
      self._coast_enter_s = self._coast_enter_s + self.dt if heavy else 0.0
      if self._losing_s + 1e-9 >= BRAKE_ENTER_S and v_ego >= V_REACT_MIN:
        self.regime = REGIME_BRAKE
      elif self.regime != REGIME_COAST and self._coast_enter_s + 1e-9 >= COAST_ENTER_S:
        self.regime = REGIME_COAST
        self._coast_exit_s = 0.0
      elif self.regime == REGIME_COAST:
        self._coast_exit_s = self._coast_exit_s + self.dt if state.torque < T_COAST_EXIT else 0.0
        if self._coast_exit_s + 1e-9 >= COAST_EXIT_S:
          self.regime = REGIME_FREE

    if self.regime == REGIME_COAST:
      # continuous in the torque: hold speed at the exit threshold, fully off the throttle at the coast threshold, so the
      # car settles where the steering is comfortably heavy instead of breathing between lift-off and free -- and never a
      # net acceleration while the steering is heavy, downhill included
      return min(float(np.interp(state.torque, [T_COAST_EXIT, T_COAST], [0.0, accel_coast])), 0.0)
    if self.regime == REGIME_BRAKE:
      # the speed at which the curvature the path demands fits back inside the budget, approached in T_RESTORE. The
      # demanded lateral acceleration, not the achieved one: a pinned car achieves less than the curve asks
      lateral = float(np.median(self._lateral_history))
      curvature_now = abs(lateral) / max(v_ego, V_REACT_MIN) ** 2
      if params is not None:
        factor, offset, friction = params
        turn = math.copysign(1.0, state.desired_lateral_accel) if state.desired_lateral_accel != 0.0 else 1.0
        a_lat_ok = max((TORQUE_BUDGET - friction) * factor + turn * (roll * ACCELERATION_DUE_TO_GRAVITY + offset), 0.5)
      else:
        a_lat_ok = A_LAT_COMFORT
      v_ok = math.sqrt(a_lat_ok / max(curvature_now, MIN_CURVATURE))
      return min(accel_coast, 0.0, (v_ok - v_ego) / T_RESTORE)
    return A_CURVE_FREE

  def update(self, model, v_ego=0.0, a_ego=0.0, lateral_active=False, steering_pressed=False, roll=0.0, accel_coast=-0.3,
             torque_params=None, lateral_state=None, gas_pressed=False):
    try:
      v_ego = max(float(v_ego), 0.0)
      a_ego = float(a_ego)
      roll = float(roll)
      accel_coast = float(accel_coast)
    except (TypeError, ValueError, OverflowError):
      return CurveResult()
    if not all(math.isfinite(x) for x in (v_ego, a_ego, roll, accel_coast)):
      return CurveResult()
    state = lateral_state if lateral_state is not None else LateralState()
    if lateral_active and state.active:
      # an idle controller reports no lateral acceleration at all, which would read as an open road mid-corner: the last
      # measurement stands until the steering is back
      self._lateral_history = self._lateral_history[1:] + [max(abs(state.desired_lateral_accel), abs(state.actual_lateral_accel))]
    steering = lateral_active and state.active and not steering_pressed
    live = _torque_values(torque_params) if torque_params is not None else None
    params = self._calibrated(live or self.torque_params, state, v_ego, steering, roll)

    self._gas_grace_s = CURVE_GAS_GRACE_S if gas_pressed else max(self._gas_grace_s - self.dt, 0.0)
    anticipation = self._anticipate(model, v_ego, roll, params, steering)
    if anticipation is None:
      anticipation = A_CURVE_FREE          # no usable path: nothing to anticipate; the reaction layer alone still runs
    elif self._gas_grace_s > 0.0:
      # the grace after a gas override: the owner chose this speed; anticipation may hold it, never pull it back down
      anticipation = max(anticipation, 0.0)
    self._candidate_history = self._candidate_history[1:] + [anticipation]
    anticipation = float(np.median(self._candidate_history))

    lifting = self.regime in (REGIME_COAST, REGIME_BRAKE)
    if steering:
      reaction = self._react(state, v_ego, accel_coast, params, roll)
    else:
      self.regime = REGIME_FREE
      self._losing_s = self._coast_enter_s = self._coast_exit_s = 0.0
      reaction = A_CURVE_FREE

    # the hold: a lift ends into zero, not into whichever candidate wants to accelerate, until the bend reads open or the
    # hold times out. A steering dropout arms it too (the regime falls to free) and keeps it for the resumption; the driver's
    # own steering or gas is never held. The anticipation reading free does not end it: the path sees the exit before the
    # car is through the bend, and a straight road reads open within the dwell anyway
    if lifting and self.regime not in (REGIME_COAST, REGIME_BRAKE) and not self._holding:
      # a lift inside a hold does not restart its clock: the backstop measures the whole hold
      self._holding = True
      self._hold_s = self._open_s = 0.0
    if self._holding:
      self._hold_s += self.dt
      self._open_s = self._open_s + self.dt if float(np.median(self._lateral_history)) < BEND_OPEN_A_LAT else 0.0
      if self._open_s + 1e-9 >= BEND_OPEN_S or self._hold_s + 1e-9 >= HOLD_MAX_S:
        self._holding = False
    if self._holding and steering and self._gas_grace_s <= 0.0:
      reaction = min(reaction, 0.0)

    raw = max(min(anticipation, reaction), A_CURVE_MIN)
    factor = params[0] if params is not None else 0.0
    if not math.isfinite(raw) or raw >= A_CURVE_FREE:
      if self.regime == REGIME_ANTICIPATE:
        self.regime = REGIME_FREE
      if self._output < A_CURVE_FREE and steering:
        # nothing binds any more: the candidate ramps out at J_UP instead of vanishing, so a hold's end is a ramp
        # into the next candidate, not a step (route 0x4d t=2790: 0.00 -> +0.48 in one frame)
        self._output = min(self._output + J_UP * self.dt, A_CURVE_FREE)
        if self._output < A_CURVE_FREE:
          self.active = True
          return CurveResult(self._output, self.regime, self.v_limit, self.distance, factor, self._holding)
      # nothing binds: the limiter follows the car so a curve that appears starts pulling from where the car is
      self._output = A_CURVE_FREE
      self.active = False
      return CurveResult(None, self.regime, self.v_limit, self.distance, factor, self._holding)
    # the anchor is the car whenever the last output sat above both the car and the demand (a ramp-out, a value that
    # was not binding): a bend that pinches again right after a release must pull from where the car is, not from the ramp
    anchor = min(self._output, max(a_ego, raw) + J_DOWN * self.dt)
    self._output = float(np.clip(raw, anchor - J_DOWN * self.dt, anchor + J_UP * self.dt))
    self.active = True
    if self.regime == REGIME_FREE:
      self.regime = REGIME_ANTICIPATE
    return CurveResult(self._output, self.regime, self.v_limit, self.distance, factor, self._holding)
