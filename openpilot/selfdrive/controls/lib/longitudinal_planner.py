#!/usr/bin/env python3
import math
import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.blotv2 import (
  BLOTV2_ACCEL_MAX,
  BLOTV2_ACCEL_REQUEST_MAX,
  BLoTv2Supervisor,
  LeadDeparturePreRelease,
  model_predicted_acceleration,
  model_predicted_speed,
)
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.force_stops import ForceStops
from openpilot.selfdrive.controls.lib.model_curve_speed import ModelCurveSpeedLimiter
from openpilot.selfdrive.controls.lib.longitudinal_lead import LeadObservation
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LongitudinalMpc,
  LongitudinalPlanSource,
  get_T_FOLLOW,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

# A single convex curve retains launch authority without the sharp slope
# changes of the former [4.0, 1.2, 0.8, 0.6] piecewise-linear schedule.
A_CRUISE_MAX_CURVE_SPEED = 40.0
A_CRUISE_MAX_HIGH_SPEED = 0.6
A_CRUISE_MAX_CURVE_POWER = 3.0
# Keep BLoTv2's reaction-time ramp separate from sustained authority.
J_CRUISE_VALS = [2.0, 1.6, 1.0, 0.6]
J_CRUISE_BP = [0., 10.0, 25., 40.]
A_CRUISE_MIN = -1.2
# Ordinary set-speed corrections use proportional authority at road speed.
# This keeps a 5 mph correction near 0.4 m/s^2 while retaining the existing
# acceleration envelope for larger errors and low-speed launches.
CRUISE_COMFORT_ACCEL_KP = 0.18
CRUISE_COMFORT_SPEED_BP = [8.0, 15.0]
CRUISE_COMFORT_BLEND_V = [0.0, 1.0]
CRUISE_COMFORT_COAST_FULL_ERROR = 5.0 * CV.MPH_TO_MS
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

# Lookup table for turns
# Do not let the turn budget clip the requested straight-line launch authority.
# Lateral acceleration still consumes this shared budget in a turn.
_A_TOTAL_MAX_V = [BLOTV2_ACCEL_REQUEST_MAX, BLOTV2_ACCEL_REQUEST_MAX]
_A_TOTAL_MAX_BP = [20., 40.]

LAUNCH_DISARM_SPEED = 2.0
LAUNCH_COMMIT_T = 3.5
LAUNCH_MOVING_SPEED = 1.2
LAUNCH_MAX_ACCEL = 1.5
LAUNCH_OPEN_LENGTH = 20.0   # m, model path length that reads as "the way ahead is open" (red-light
                            # stubs sit at 2-5m with occasional 12-13m flickers; green opens 30-60m)
LAUNCH_OPEN_CONFIRM = 0.7   # filtered (RC 0.3s) open level to trust -- ~0.5s of sustained open path
LAUNCH_CLOSE_LENGTH = 10.0  # m, path re-collapse below this cancels anticipation (model changed its mind)


def get_requested_max_accel(v_ego):
  speed_fraction = float(np.clip(v_ego / A_CRUISE_MAX_CURVE_SPEED, 0.0, 1.0))
  remaining_fraction = 1.0 - speed_fraction
  return float(A_CRUISE_MAX_HIGH_SPEED +
               (BLOTV2_ACCEL_REQUEST_MAX - A_CRUISE_MAX_HIGH_SPEED) * remaining_fraction ** A_CRUISE_MAX_CURVE_POWER)

def get_max_accel(v_ego):
  return min(get_requested_max_accel(v_ego), BLOTV2_ACCEL_MAX)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py


def ordinary_cruise_comfort_enabled(experimental_mode, force_decel, radar_valid, lead_present, speed_limiter_active=False):
  """Limit comfort shaping to healthy, lead-free Chill cruise."""
  return not (experimental_mode or force_decel or not radar_valid or lead_present or speed_limiter_active)


def get_cruise_comfort_accel(v_cruise, v_ego, accel_coast):
  """Return the proportional acceleration target for an ordinary cruise correction."""
  speed_error = v_cruise - v_ego
  target_accel = CRUISE_COMFORT_ACCEL_KP * speed_error

  if speed_error < 0.0 and np.isfinite(accel_coast):
    # Blend toward a full throttle lift as the requested reduction reaches
    # 5 mph. On an uphill this permits natural coast-down without adding gas;
    # on a downhill the proportional target still requests gentle braking.
    coast_weight = np.interp(-speed_error, [0.0, CRUISE_COMFORT_COAST_FULL_ERROR], [0.0, 1.0])
    target_accel = min(target_accel, accel_coast * coast_weight)

  return float(target_accel)


def get_cruise_accel(e2e, v_cruise, v_ego, a_cruise_prev, angle_steers, CP, dt, accel_coast, allow_throttle,
                     comfort_enabled=False):
  max_accel = BLOTV2_ACCEL_MAX if e2e else get_max_accel(v_ego)

  if not e2e:
    a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
    a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
    a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))
    max_accel = min(max_accel, a_x_allowed)
    if not allow_throttle:
      clipped_accel_coast = max(accel_coast, ACCEL_MIN)
      coast_limit = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [max_accel, clipped_accel_coast])
      max_accel = min(max_accel, coast_limit)

  legacy_target_accel = float(np.clip(v_cruise - v_ego, A_CRUISE_MIN, max_accel))
  target_accel = legacy_target_accel
  if comfort_enabled:
    comfort_target_accel = float(np.clip(get_cruise_comfort_accel(v_cruise, v_ego, accel_coast),
                                         A_CRUISE_MIN, max_accel))
    comfort_weight = float(np.interp(v_ego, CRUISE_COMFORT_SPEED_BP, CRUISE_COMFORT_BLEND_V))
    target_accel = float(np.interp(comfort_weight, [0.0, 1.0], [legacy_target_accel, comfort_target_accel]))

  if not e2e:
    j_cruise = np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS)
    target_accel = float(np.clip(target_accel, a_cruise_prev - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  return target_accel


def limit_accel_for_torque(a_target, torque_veto):
  return min(a_target, 0.0) if torque_veto else a_target


def get_live_torque_params(sm):
  service = 'lateralTorqueParameters'
  try:
    params = sm[service]
    healthy = sm.alive[service] and sm.freq_ok[service] and sm.valid[service]
  except (AttributeError, KeyError):
    return None
  return params if healthy and params.useParams else None


class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True
    self.blotv2 = BLoTv2Supervisor(dt)
    self.lead_departure = LeadDeparturePreRelease(dt)
    self.curve_speed_limiter = ModelCurveSpeedLimiter(CP)
    self.force_stops = ForceStops(dt)

    self.a_desired = init_a
    self.last_mpc_a_target = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.a_cruise = 0.0
    self.output_a_target = 0.0
    self.output_should_stop = False
    self.launch_armed = False
    self.launch_open = FirstOrderFilter(0.0, 0.3, self.dt)
    self.anticipating = False

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  @staticmethod
  def parse_model(model_msg):
    """Return the model trajectory used by throttle gating and launch anticipation."""
    if (
      len(model_msg.position.x) == ModelConstants.IDX_N
      and len(model_msg.velocity.x) == ModelConstants.IDX_N
      and len(model_msg.acceleration.x) == ModelConstants.IDX_N
    ):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))

    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return x, v, a, j, throttle_prob

  def update(self, sm):
    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = BLOTV2_ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    force_decel = sm['controlsState'].forceDecel
    if force_decel:
      v_cruise = 0.0
    else:
      torque_params = get_live_torque_params(sm)
      lateral_active = sm['carControl'].latActive
      v_cruise = self.curve_speed_limiter.update(sm['modelV2'], v_cruise, v_ego=v_ego, lateral_active=lateral_active,
                                                 roll=sm['vehicleParameters'].roll, torque_params=torque_params)

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
    reset_state = reset_state or not v_cruise_initialized

    _, model_v, model_a, _, throttle_prob = self.parse_model(sm['modelV2'])
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['vehicleParameters'].angleOffsetDeg

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.a_desired = np.clip(sm['carState'].aEgo, ACCEL_MIN, BLOTV2_ACCEL_MAX)
      self.last_mpc_a_target = float(self.a_desired)
      self.blotv2.reset()
      self.lead_departure.reset()

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    force_stop_cap = self.force_stops.update(sm)
    stop_x = self.force_stops.remaining if self.force_stops.forcing else None

    personality = sm['selfdriveState'].personality
    radar_valid = sm.all_checks(['radarState'])
    lead = LeadObservation.from_radar(sm['radarState'].leadOne, radar_valid)
    radar_has_lead = radar_valid and (sm['radarState'].leadOne.present or sm['radarState'].leadTwo.present)
    model_leads = sm['modelV2'].leadsV3
    model_lead_0 = model_leads[0] if len(model_leads) > 0 else None
    policy = self.blotv2.update(
      lead,
      v_ego,
      self.last_mpc_a_target,
      get_T_FOLLOW(personality),
      model_predicted_acceleration(model_lead_0),
    )

    self.mpc.set_weights(
      prev_accel_constraint,
      personality=personality,
      jerk_factor_scale=policy.jerk_scale,
    )
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    self.mpc.update(
      sm['radarState'],
      personality=personality,
      t_follow=policy.t_follow,
      model_leads=model_leads,
      model_position=sm['modelV2'].position,
      allow_third_lead=(
        not reset_state
        and sm.all_checks(['modelV2'])
        and sm['modelV2'].meta.laneChangeState == log.LaneChangeState.off
        and not (sm['carState'].leftBlinker or sm['carState'].rightBlinker)
      ),
      stop_x=stop_x,
      v_ego=v_ego,
    )

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = (self.mpc.crash_cnt > 2 and not sm['carState'].standstill) or policy.emergency
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Save starting point for next iteration
    a_prev = self.a_desired

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                              action_t=action_t)
    self.last_mpc_a_target = float(output_a_target_mpc)
    output_should_stop_mpc = should_stop(v_ego, output_a_target_mpc)
    if self.lead_departure.update(
      active=self.CP.openpilotLongitudinalControl and not long_control_off,
      standstill=sm['carState'].standstill,
      lead=lead,
      predicted_speed=model_predicted_speed(model_lead_0, lead),
    ):
      # Begin only the MPC hold-release leg; preserve its acceleration target
      # and never override an e2e stop candidate below.
      output_should_stop_mpc = False
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    experimental_mode = sm['selfdriveState'].experimentalMode

    # Green-light anticipation: at a hold, the model's path length explodes (2-5m stub ->
    # 30-60m) about 1.5-2s BEFORE its shouldStop bit releases (field data, routes 37/38).
    # Read the path instead of the laggy bit -- the same trick Force Stops uses for stop
    # intent, mirrored for launch intent. Clearing shouldStop releases the hold; the car
    # creeps off, the model sees motion and commits its plan, and the launch assist below
    # takes it from there. If the path re-collapses, the hold re-engages at creep speed.
    xs = sm['modelV2'].position.x
    model_length = float(xs[-1]) if len(xs) else 0.0
    self.launch_open.update(1.0 if model_length > LAUNCH_OPEN_LENGTH else 0.0)
    if (sm['carState'].standstill and output_should_stop_e2e and
        experimental_mode and self.launch_open.x > LAUNCH_OPEN_CONFIRM):
      self.anticipating = True
    if self.anticipating:
      if model_length < LAUNCH_CLOSE_LENGTH or v_ego > LAUNCH_DISARM_SPEED or not experimental_mode:
        self.anticipating = False
      else:
        output_should_stop_e2e = False

    if sm['carState'].standstill:
      self.launch_armed = True
    elif v_ego > LAUNCH_DISARM_SPEED:
      self.launch_armed = False
    if (self.launch_armed and experimental_mode and not output_should_stop_e2e and
        np.interp(LAUNCH_COMMIT_T, T_IDXS_MPC, model_v) > LAUNCH_DISARM_SPEED):
      t_cut = min(float(T_IDXS_MPC[np.argmax(model_v > LAUNCH_MOVING_SPEED)]), LAUNCH_COMMIT_T)
      t_shifted = T_IDXS_MPC + t_cut
      v_shifted = np.interp(t_shifted, T_IDXS_MPC, model_v)
      a_shifted = np.interp(t_shifted, T_IDXS_MPC, model_a)
      a_launch = get_accel_from_plan(v_shifted, a_shifted, T_IDXS_MPC, action_t=action_t)
      a_launch_max = np.interp(v_ego, [LAUNCH_MOVING_SPEED, LAUNCH_DISARM_SPEED], [LAUNCH_MAX_ACCEL, 0.])
      output_a_target_e2e = max(output_a_target_e2e, min(a_launch, a_launch_max))

    # Optional Force Stops owns the committed approach point; CEM only selects
    # Experimental mode and Smooth Stops remains the final-landing owner.
    v_cruise = min(v_cruise, force_stop_cap)

    comfort_enabled = ordinary_cruise_comfort_enabled(
      experimental_mode,
      force_decel,
      radar_valid,
      radar_has_lead,
      speed_limiter_active=self.curve_speed_limiter.active,
    )
    self.a_cruise = get_cruise_accel(experimental_mode, v_cruise, v_ego,
                                     self.a_cruise, steer_angle_without_offset, self.CP, self.dt,
                                     accel_coast, self.allow_throttle, comfort_enabled=comfort_enabled)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)

    candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop)]
    if experimental_mode:
      candidates.append((output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e))

    output_a_target, self.mpc.source, _ = min(candidates, key=lambda c: c[0])
    self.output_should_stop = any(should_stop for _, _, should_stop in candidates)
    self.output_a_target = limit_accel_for_torque(np.clip(output_a_target, ACCEL_MIN, BLOTV2_ACCEL_MAX),
                                                  self.curve_speed_limiter.torque_veto)

    self.a_desired = float(self.output_a_target)
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.output_a_target + a_prev) / 2.0

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.present
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)
