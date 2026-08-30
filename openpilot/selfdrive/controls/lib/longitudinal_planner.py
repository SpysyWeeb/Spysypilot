#!/usr/bin/env python3
import math
import numpy as np

import openpilot.cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.force_stops import ForceStops
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_lead import LeadObservation, anchor_model_lead
from openpilot.selfdrive.controls.lib.model_curve_speed import LateralState, ModelCurveSpeedLimiter
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.necessity_supervisor import LeadDeparturePreRelease, NecessitySupervisor
from openpilot.selfdrive.controls.lib.stop_helpers import StopObservation, observe_model_stop
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

# one convex curve from launch authority to the high speed floor, no speed node corners;
# the deployed opendbc ACCEL_MAX clamps the request (2.0 stock, 4.0 with the fork's opendbc/panda)
A_CRUISE_MAX_LAUNCH = 4.0
A_CRUISE_MAX_HIGH_SPEED = 0.6
A_CRUISE_MAX_SPEED = 40.
J_CRUISE_VALS = [2.0, 1.6, 1.0, 0.6]
J_CRUISE_BP = [0., 10.0, 25., 40.]
A_CRUISE_MIN = -1.2
E2E_STOP_MARGIN = 0.5  # m/s^2, how much more urgent the model's request must be to override a committed stop profile
# ordinary set-speed corrections at road speed use a proportional target, so a 5 mph error asks for ~0.4 m/s^2
CRUISE_COMFORT_KP = 0.18
CRUISE_COMFORT_BP = [8.0, 15.0]
CRUISE_COMFORT_COAST_ERROR = 5.0 * CV.MPH_TO_MS
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

LAUNCH_DISARM_SPEED = 2.0
LAUNCH_COMMIT_T = 3.5
LAUNCH_MOVING_SPEED = 1.2
LAUNCH_MAX_ACCEL = 1.5
LAUNCH_OPEN_LENGTH = 20.0   # m, model path length that reads as "the way ahead is open" (red-light
                            # stubs sit at 2-5m with occasional 12-13m flickers; green opens 30-60m)
LAUNCH_OPEN_CONFIRM = 0.7   # filtered (RC 0.3s) open level to trust -- ~0.5s of sustained open path
LAUNCH_CLOSE_LENGTH = 10.0  # m, path re-collapse below this cancels anticipation (model changed its mind)

def get_max_accel_request(v_ego):
  remaining = 1.0 - np.clip(v_ego / A_CRUISE_MAX_SPEED, 0.0, 1.0)
  return float(A_CRUISE_MAX_HIGH_SPEED + (A_CRUISE_MAX_LAUNCH - A_CRUISE_MAX_HIGH_SPEED) * remaining ** 3)

def get_max_accel(v_ego):
  return min(get_max_accel_request(v_ego), ACCEL_MAX)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def get_cruise_comfort_accel(v_cruise, v_ego, accel_coast):
  speed_error = v_cruise - v_ego
  target_accel = CRUISE_COMFORT_KP * speed_error
  if speed_error < 0.0 and np.isfinite(accel_coast):
    # lift off toward a full coast by a 5 mph reduction: an uphill slows the car by itself, a downhill still gets gentle braking
    coast_weight = np.interp(-speed_error, [0.0, CRUISE_COMFORT_COAST_ERROR], [0.0, 1.0])
    target_accel = min(target_accel, accel_coast * coast_weight)
  return float(target_accel)

def get_live_torque_params(sm):
  service = 'lateralTorqueParameters'
  try:
    params = sm[service]
    healthy = sm.alive[service] and sm.freq_ok[service] and sm.valid[service]
  except (AttributeError, KeyError):
    return None
  return params if healthy and params.useParams else None


def ordinary_cruise_comfort_enabled(experimental_mode, force_decel, radar_valid):
  # comfort shaping only for ordinary cruise with a healthy radar; lead following and e2e keep their own targets
  return not (experimental_mode or force_decel or not radar_valid)

def get_cruise_accel(e2e, v_cruise, v_ego, a_cruise_prev, dt, accel_coast, allow_throttle, comfort=False):
  # the envelope alone bounds cruise acceleration; stock's shared lateral budget held the car back out of curves (owner ruling)
  max_accel = ACCEL_MAX if e2e else get_max_accel(v_ego)

  if not e2e:
    if not allow_throttle:
      clipped_accel_coast = max(accel_coast, ACCEL_MIN)
      coast_limit = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [max_accel, clipped_accel_coast])
      max_accel = min(max_accel, coast_limit)

  target_accel = np.clip(v_cruise - v_ego, A_CRUISE_MIN, max_accel)
  if comfort:
    comfort_accel = np.clip(get_cruise_comfort_accel(v_cruise, v_ego, accel_coast), A_CRUISE_MIN, max_accel)
    target_accel = np.interp(v_ego, CRUISE_COMFORT_BP, [target_accel, comfort_accel])
  j_cruise = np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS)
  target_accel = float(np.clip(target_accel, a_cruise_prev - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  return target_accel


class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True
    self.supervisor = NecessitySupervisor(dt)
    self.lead_departure = LeadDeparturePreRelease(dt)
    self.force_stops = ForceStops(dt)
    self.curve_speed_limiter = ModelCurveSpeedLimiter(CP)
    self.mpc_a_target = init_a

    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.a_cruise = init_a
    self.output_a_target = init_a
    self.output_should_stop = False
    self.launch_armed = False
    self.launch_open = FirstOrderFilter(0.0, 0.3, self.dt)
    self.anticipating = False

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  @staticmethod
  def parse_model(model_msg):
    # the model trajectory used by throttle gating and launch anticipation
    if (len(model_msg.position.x) == ModelConstants.IDX_N and len(model_msg.velocity.x) == ModelConstants.IDX_N
        and len(model_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
    j = np.zeros(len(T_IDXS_MPC))
    throttle_probs = model_msg.meta.disengagePredictions.gasPressProbs
    throttle_prob = throttle_probs[1] if len(throttle_probs) > 1 else 1.0
    return x, v, a, j, throttle_prob

  def update(self, sm):
    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    force_decel = sm['controlsState'].forceDecel
    if force_decel:
      v_cruise = 0.0

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
    reset_state = reset_state or not v_cruise_initialized

    _, model_v, model_a, _, throttle_prob = self.parse_model(sm['modelV2'])
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.output_a_target = np.clip(sm['carState'].aEgo, ACCEL_MIN, ACCEL_MAX)
      self.a_cruise = self.output_a_target
      self.mpc_a_target = float(self.output_a_target)
      self.supervisor.reset()
      self.lead_departure.reset()

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    # No change cost when user is controlling the speed; it stays on through standstill so launches start smooth
    prev_accel_constraint = not reset_state

    radar_valid = sm.all_checks(['radarState'])
    model_valid = sm.all_checks(['modelV2'])
    lead = LeadObservation.from_radar(sm['radarState'].leadOne, radar_valid)
    model_leads = sm['modelV2'].leadsV3 if model_valid else []
    lead0_anchor = anchor_model_lead(model_leads[0], sm['radarState'].leadOne) if len(model_leads) > 0 else None
    lead1_anchor = anchor_model_lead(model_leads[1], sm['radarState'].leadTwo) if len(model_leads) > 1 else None
    policy = self.supervisor.update(lead, v_ego, self.mpc_a_target, lead0_anchor.accel if lead0_anchor is not None else None)

    experimental_mode = sm['selfdriveState'].experimentalMode
    stop = observe_model_stop(sm['modelV2'], sm['carState'], sm['radarState']) if model_valid else StopObservation()
    force_stop = self.force_stops.update(stop, sm['carState'], experimental_mode, not reset_state, model_valid)
    stop_x = force_stop.stop_x if force_stop.stop_x is not None and math.isfinite(force_stop.stop_x) else None
    v_cruise = min(v_cruise, force_stop.v_cruise_cap)

    personality = sm['selfdriveState'].personality
    self.mpc.set_cur_state(self.v_desired_filter.x, self.output_a_target)
    self.mpc.update(sm['radarState'], personality, lead0_anchor, lead1_anchor, stop_x,
                    policy.jerk_scale, policy.t_follow_pad, prev_accel_constraint)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Save starting point for next iteration
    a_prev = self.output_a_target

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                              action_t=action_t)
    self.mpc_a_target = float(output_a_target_mpc)
    output_should_stop_mpc = should_stop(v_ego, output_a_target_mpc)
    # a stopped lead that is confirmed leaving releases the MPC's stop bit early; its acceleration target is untouched
    lead_departing = self.lead_departure.update(self.CP.openpilotLongitudinalControl and not long_control_off, sm['carState'].standstill,
                                                lead, lead0_anchor.speed if lead0_anchor is not None else None)
    if lead_departing:
      output_should_stop_mpc = False
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    # Green-light anticipation: at a hold, the model's path length explodes (2-5m stub ->
    # 30-60m) about 1.5-2s BEFORE its shouldStop bit releases (field data, routes 37/38).
    # Read the path instead of the laggy bit -- the same trick Force Stops uses for stop
    # intent, mirrored for launch intent. Clearing shouldStop releases the hold; the car
    # creeps off, the model sees motion and commits its plan, and the launch assist below
    # takes it from there. If the path re-collapses, the hold re-engages at creep speed.
    xs = sm['modelV2'].position.x
    model_length = float(xs[len(xs) - 1]) if len(xs) else 0.0
    self.launch_open.update(1.0 if model_length > LAUNCH_OPEN_LENGTH else 0.0)
    if (sm['carState'].standstill and output_should_stop_e2e and experimental_mode and self.launch_open.x > LAUNCH_OPEN_CONFIRM):
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

    comfort = ordinary_cruise_comfort_enabled(experimental_mode, force_decel, radar_valid)
    self.a_cruise = get_cruise_accel(experimental_mode, v_cruise, v_ego, self.a_cruise, self.dt, accel_coast, self.allow_throttle, comfort)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)

    # the curve policy: anticipation from the model path, reaction to the measured steering state; one candidate
    # that can only lower the chosen acceleration
    curve = self.curve_speed_limiter.update(sm['modelV2'], v_ego=v_ego, a_ego=sm['carState'].aEgo,
                                            lateral_active=sm['carControl'].latActive, steering_pressed=sm['carState'].steeringPressed,
                                            roll=sm['vehicleParameters'].roll, accel_coast=accel_coast,
                                            torque_params=get_live_torque_params(sm),
                                            lateral_state=LateralState.from_controls_state(sm['controlsState']))

    candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop)]
    if experimental_mode and model_valid:
      # while a committed stop's profile is moving the car, the model's own request joins only if it is clearly more urgent:
      # its late ramp used to overtake the profile through min() and put the heavy braking back at the end (route 27 t=250)
      if force_stop.a_target is None or output_a_target_e2e < force_stop.a_target - E2E_STOP_MARGIN:
        candidates.append((output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e))
    if force_stop.a_target is not None:
      # a committed stop's own approach profile competes like any candidate; the column and the hold still own the landing
      candidates.append((force_stop.a_target, LongitudinalPlanSource.stop, False))

    if curve.a_target is not None and not reset_state:
      # a frame after engaging the arbitration would otherwise seed the MPC with a stale floored candidate
      candidates.append((curve.a_target, LongitudinalPlanSource.curve, False))

    output_a_target, self.mpc.source, _ = min(candidates, key=lambda c: c[0])
    self.output_should_stop = force_stop.holding or any(should_stop for _, _, should_stop in candidates)
    self.output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)

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
