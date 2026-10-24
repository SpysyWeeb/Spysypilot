#!/usr/bin/env python3
import math
from collections import deque

import numpy as np

from opendbc.car import gen_empty_fingerprint
from opendbc.car.car_helpers import interfaces
from opendbc.car.interfaces import CarInterfaceBase
from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper, DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.stop_landing import STANDSTILL_SPEED
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU

RADAR_DREL_STEP = 0.1  # m, the resolution of the Palisade radar's track distance
# driveline creep is an assumed model, not an identified one: it fades out by DRIVELINE_CREEP_SPEED, and the brakes hold against
# it only while the request (not the braking the actuator applies) asks for at least CREEP_HOLD_REQUEST
DRIVELINE_CREEP_SPEED = 1.0  # m/s
CREEP_HOLD_REQUEST = -0.2  # m/s^2


class _PlantSubMaster:
  # mimics messaging.SubMaster's liveness contract so the planner (and future
  # BLoTv3 modules) see a real all_checks() instead of one that is always True
  def __init__(self, data, mono_time, invalid=()):
    self.data = data
    self.updated = dict.fromkeys(data, True)
    self.logMonoTime = dict.fromkeys(data, mono_time)
    self.valid = {s: s not in invalid for s in data}
    self.alive = dict(self.valid)
    self.freq_ok = dict.fromkeys(data, True)

  def __getitem__(self, s):
    return self.data[s]

  def all_checks(self, service_list=None):
    services = self.data.keys() if service_list is None else service_list
    return all(self.valid[s] and self.alive[s] and self.freq_ok[s] for s in services)


def _model_lead_v3(x, v, prob):
  n = len(ModelConstants.LEAD_T_IDXS)
  lead = log.ModelDataV2.LeadDataV3.new_message()
  lead.prob = float(prob)
  lead.probTime = 0.0
  lead.t = [float(t) for t in ModelConstants.LEAD_T_IDXS]
  lead.x = [float(p) for p in x]
  lead.xStd = [1.0] * n
  lead.y = [0.0] * n
  lead.yStd = [1.0] * n
  lead.v = [float(vv) for vv in v]
  lead.vStd = [0.5] * n
  return lead


def palisade_car_params():
  # the car BLoTv3 drives: a Palisade under openpilot longitudinal control
  from opendbc.car.hyundai.values import CAR
  return interfaces[CAR.HYUNDAI_PALISADE].get_params(CAR.HYUNDAI_PALISADE, gen_empty_fingerprint(), [], alpha_long=True, is_release=False, docs=False)


class ActuatorLag:
  """The Palisade's acceleration response to its SCC request: per speed band, the held-out best fit of a dead time and a first
  order with separate time constants for building and releasing braking, identified on braking requests against the wheel
  acceleration on its CAN bus. The data do not identify the dead time (0-0.24 s fit within 1 %) or the asymmetry below 4 m/s
  (a symmetric lag or a rate limit fits as well), so a result that moves within those spreads is not decided by this fit.
  It has no standstill regime: held wheels do not accelerate, and a launch starts from rest behind the dead time."""
  SPEED_BP = [4.0, 8.0]
  # per speed band: dead time (s), time constant building braking (s), releasing it (s), gain
  BANDS = [(0.12, 0.20, 0.60, 0.95), (0.08, 0.25, 0.30, 1.0), (0.16, 0.20, 0.30, 1.0)]

  def __init__(self):
    n = round(max(band[0] for band in self.BANDS) / DT_CTRL) + 1
    self.requests = deque([0.0] * n, maxlen=n)
    self.accel = 0.0

  def update(self, request, v_ego):
    dead_time, tau_bite, tau_release, gain = self.BANDS[int(np.searchsorted(self.SPEED_BP, v_ego, side='right'))]
    self.requests.append(request)
    target = gain * self.requests[-1 - round(dead_time / DT_CTRL)]
    tau = tau_bite if target < self.accel else tau_release
    self.accel += (target - self.accel) * (1.0 - math.exp(-DT_CTRL / tau))
    if v_ego <= 0.0:
      self.accel = max(self.accel, 0.0)
    return self.accel


class EscStandstill:
  """The Palisade's ESC standstill brake, fitted on its CAN bus from SCC12's StopReq and request to the engine torque's onset and
  the wheels' first motion. It holds the wheels of a car at rest under StopReq, and from the frame StopReq falls lets them go once
  the request has built up past its release level: the torque comes in at the onset, and the wheels turn BREAKAWAY_S later, more
  uphill. 'integral' is the best held-out fit (the onset INTEGRAL_DELAY after the request's excess over INTEGRAL_ACCEL integrates to
  INTEGRAL_BUILD, sooner the higher the request); 'threshold' is the level-blind fit that brackets it (THRESHOLD_DELAY after the
  request held THRESHOLD_ACCEL for THRESHOLD_DWELL). StopReq rising again before the wheels turn holds them and starts over. The
  actuator keeps answering the request under the held wheels, so they turn, at the fitted time, on what it has built."""
  INTEGRAL_ACCEL = 0.30    # m/s^2
  INTEGRAL_BUILD = 0.25    # m/s
  INTEGRAL_DELAY = 0.20    # s
  THRESHOLD_ACCEL = 0.50   # m/s^2
  THRESHOLD_DWELL = 0.10   # s
  THRESHOLD_DELAY = 0.65   # s
  BREAKAWAY_S = 0.556      # s from the onset to the wheels turning on the flat ...
  BREAKAWAY_PER_DEG = 0.075  # ... and this much longer per degree uphill
  FORMS = ('integral', 'threshold')

  def __init__(self, form='integral'):
    assert form in self.FORMS, form
    self.form = form
    self.holding = False
    self._start_over()

  def _start_over(self):
    self.build = 0.0       # the integral (m/s) or the dwell (s) toward the onset
    self.release_in = None  # s from the onset until the wheels turn

  def launch_delay(self, request, pitch=0.0):
    # s from StopReq falling under a steady request to the wheels turning
    if self.form == 'integral':
      onset = self.INTEGRAL_BUILD / (request - self.INTEGRAL_ACCEL) + self.INTEGRAL_DELAY if request > self.INTEGRAL_ACCEL else math.inf
    else:
      onset = self.THRESHOLD_DWELL + self.THRESHOLD_DELAY if request >= self.THRESHOLD_ACCEL else math.inf
    return onset + self.BREAKAWAY_S + self.BREAKAWAY_PER_DEG * max(math.degrees(pitch), 0.0)

  def update(self, stop_req, request, at_rest, pitch, dt):
    """One control step; returns whether the wheels are held."""
    if stop_req:
      self.holding = self.holding or at_rest
      self._start_over()
      return self.holding
    if not self.holding:
      return False
    if self.release_in is None:
      if self.form == 'integral':
        self.build += max(request - self.INTEGRAL_ACCEL, 0.0) * dt
        onset, delay = self.build + 1e-9 >= self.INTEGRAL_BUILD, self.INTEGRAL_DELAY
      else:
        self.build = self.build + dt if request >= self.THRESHOLD_ACCEL else 0.0
        onset, delay = self.build + 1e-9 >= self.THRESHOLD_DWELL, self.THRESHOLD_DELAY
      if onset:
        self.release_in = delay + self.BREAKAWAY_S + self.BREAKAWAY_PER_DEG * max(math.degrees(pitch), 0.0)
      return True
    self.release_in -= dt
    if self.release_in < 1e-9:
      self.holding = False
      self._start_over()
    return self.holding


E2E_PUSH_DISTANCE = 12.0  # m, the fake model's late ramp builds over this distance to the line
GREEN_PATH = 60.0  # m, at a green the fake model plans a long path, twice what reads as an open road at standstill
GREEN_SPEED = 5.0  # m/s, its terminal speed
GREEN_ACCEL = 0.5  # m/s^2, its request


class Plant:
  def __init__(self, lead_relevancy=False, speed=0.0, distance_lead=2.0,
               enabled=True, only_lead2=False, only_radar=False, e2e=False, personality=0, force_decel=False,
               stop_line=None, stop_line_horizon_s=5.0, e2e_landing_push=0.0, actuator_lag=False, CP=None, accel_error=0.0, creep=0.0,
               esc='integral'):
    self.rate = 1. / DT_MDL


    self.v_lead_prev = 0.0

    self.distance = 0.
    self.speed = speed
    self.should_stop = False
    self.acceleration = 0.0
    self.a_target = 0.0

    # lead car
    self.lead_relevancy = lead_relevancy
    self.distance_lead = distance_lead
    self.enabled = enabled
    self.only_lead2 = only_lead2
    self.only_radar = only_radar
    self.e2e = e2e
    self.personality = personality
    self.force_decel = force_decel
    # a world-fixed stop line the fake model plans to stop at once it is within stop_line_horizon_s of travel,
    # the way the real model calls a red light only a few seconds out
    self.stop_line = stop_line
    self.stop_line_horizon_s = stop_line_horizon_s
    # extra braking the fake model asks for over the last metres before the line, the way the real model's request
    # ramps late (-0.7 ... -1.3 -> -2.5 into the last seconds)
    self.e2e_landing_push = e2e_landing_push
    # without it the car applies the request at once
    self.actuator = ActuatorLag() if actuator_lag else None
    # added to the car's acceleration while it rolls, whatever it was asked: brakes that fall short, and the same push at cruise
    # and on a launch; held wheels do not feel it
    self.accel_error = accel_error
    # driveline creep, added to launch requests too
    self.creep = creep
    # the ESC's standstill brake under the actuator (None: wheels at rest go on any positive request)
    self.esc = EscStandstill(esc) if esc is not None else None
    self.pitch = 0.0

    self.rk = Ratekeeper(self.rate, print_delay_threshold=100.0)
    self.ts = 1. / self.rate

    if CP is None:
      from opendbc.car.honda.values import CAR
      CP = interfaces[CAR.HONDA_CIVIC].get_non_essential_params(CAR.HONDA_CIVIC)
    self.CP = CP
    self.CI = interfaces[CP.carFingerprint](CP)
    self.planner = LongitudinalPlanner(CP, init_v=self.speed)
    # under the stop bit the car follows this tree's LongControl, not the plan
    self.long_control = LongControl(CP)
    # carState's acceleration is the car's speed filter on its wheels, not the applied acceleration; the maneuver starts with
    # the car settled at its initial speed
    self.CI.CS.v_ego_kf.set_x([[self.speed], [0.0]])
    self.a_ego = 0.0

  def _plan_stop_line(self, model):
    # the model's path ends at the line with a constant-deceleration speed profile; its e2e request is
    # deliberately softer than that need, as the real model's early request is
    d_line = self.stop_line - self.distance
    t = np.array(ModelConstants.T_IDXS)
    if d_line <= 0.5 or self.speed < 0.3:
      x = np.zeros_like(t)
      v = np.zeros_like(t)
      model.action.desiredAcceleration = float(min(self.acceleration, 0.0)) if self.speed >= 0.3 else 0.0
    elif d_line <= self.speed * self.stop_line_horizon_s:
      a_req = self.speed ** 2 / (2.0 * d_line)
      t_stop = self.speed / a_req
      tc = np.minimum(t, t_stop)
      x = self.speed * tc - 0.5 * a_req * tc ** 2
      v = np.maximum(self.speed - a_req * tc, 0.0)
      push = self.e2e_landing_push * float(np.clip((E2E_PUSH_DISTANCE - d_line) / E2E_PUSH_DISTANCE, 0.0, 1.0))
      model.action.desiredAcceleration = float(-0.7 * a_req - push)
    else:
      return
    # modeld's own stop bit: below its stopping speed with the request braking
    model.action.shouldStop = should_stop(self.speed, model.action.desiredAcceleration)
    position = log.XYZTData.new_message()
    position.x = [float(xx) for xx in x]
    model.position = position
    velocity = log.XYZTData.new_message()
    velocity.x = [float(vv) for vv in v]
    velocity.x[0] = float(self.speed)
    model.velocity = velocity

  def _plan_open_road(self, model):
    # a green: the model drops its stop, plans a long moving path and asks to go
    n = len(ModelConstants.T_IDXS)
    position = log.XYZTData.new_message()
    position.x = [float(x) for x in np.linspace(0.0, GREEN_PATH, n)]
    model.position = position
    velocity = log.XYZTData.new_message()
    velocity.x = [float(v) for v in np.linspace(self.speed, GREEN_SPEED, n)]
    model.velocity = velocity
    model.action.desiredAcceleration = GREEN_ACCEL
    model.action.shouldStop = False

  @property
  def current_time(self):
    return float(self.rk.frame) / self.rate

  def step(self, v_lead=0.0, prob_lead=1.0, v_cruise=50., pitch=0.0, prob_throttle=1.0,
           radar_valid=True, model_valid=True, forecast=None, green=False):
    # forecast: the acceleration the model sees the lead hold when it differs from the radar's; green: the model reads the stop
    # line's light green and plans open road
    # ******** publish a fake model going straight and fake calibration ********
    # note that this is worst case for MPC, since model will delay long mpc by one time step
    radar = messaging.new_message('radarState')
    control = messaging.new_message('controlsState')
    ss = messaging.new_message('selfdriveState')
    car_state = messaging.new_message('carState')
    lp = messaging.new_message('vehicleParameters')
    car_control = messaging.new_message('carControl')
    model = messaging.new_message('modelV2')
    a_lead = (v_lead - self.v_lead_prev)/self.ts
    self.v_lead_prev = v_lead

    if self.lead_relevancy:
      d_rel = np.maximum(0., self.distance_lead - self.distance)
      v_rel = v_lead - self.speed
      if self.only_radar:
        status = True
      elif prob_lead > .5:
        status = True
      else:
        status = False
    else:
      d_rel = 200.
      v_rel = 0.
      prob_lead = 0.0
      status = False

    lead = log.RadarState.LeadData.new_message()
    lead.dRel = float(RADAR_DREL_STEP * round(d_rel / RADAR_DREL_STEP))
    lead.yRel = 0.0
    lead.vRel = float(v_rel)
    lead.vLead = float(v_lead)
    lead.vLeadK = float(v_lead)
    lead.aLeadK = float(a_lead)
    # TODO use real radard logic for this
    lead.aLeadTau = float(_LEAD_ACCEL_TAU)
    lead.present = status
    lead.modelProb = float(prob_lead)
    lead.radar = True
    if not self.only_lead2:
      radar.radarState.leadOne = lead
    radar.radarState.leadTwo = lead

    # Simulate model predicting slightly faster speed
    # this is to ensure lead policy is effective when model
    # does not predict slowdown in e2e mode
    position = log.XYZTData.new_message()
    position.x = [float(x) for x in (self.speed + 0.5) * np.array(ModelConstants.T_IDXS)]
    model.modelV2.position = position
    model.modelV2.action.desiredAcceleration = float(self.acceleration + 0.5)
    velocity = log.XYZTData.new_message()
    velocity.x = [float(x) for x in (self.speed + 0.5) * np.ones_like(ModelConstants.T_IDXS)]
    velocity.x[0] = float(self.speed) # always start at current speed
    model.modelV2.velocity = velocity
    acceleration = log.XYZTData.new_message()
    acceleration.x = [float(x) for x in np.zeros_like(ModelConstants.T_IDXS)]
    model.modelV2.acceleration = acceleration
    if green:
      self._plan_open_road(model.modelV2)
    elif self.stop_line is not None:
      self._plan_stop_line(model.modelV2)
    model.modelV2.meta.disengagePredictions.gasPressProbs = [float(prob_throttle) for _ in range(6)]

    # lead0 mirrors the radar lead above; lead1/lead2 are shaped but carry no probability
    lead_t = np.asarray(ModelConstants.LEAD_T_IDXS, dtype=np.float64)
    a_model = a_lead if forecast is None else forecast
    if a_model < 0.0:
      stop_t = np.minimum(lead_t, max(-v_lead / a_model, 0.0))
    else:
      stop_t = lead_t
    lead0_v = np.maximum(v_lead + a_model * lead_t, 0.0)
    lead0_x = np.maximum.accumulate(d_rel + v_lead * stop_t + 0.5 * a_model * stop_t ** 2)
    lead0_prob = float(prob_lead) if self.lead_relevancy else 0.0
    zeros = np.zeros_like(lead_t)
    model.modelV2.leadsV3 = [
      _model_lead_v3(lead0_x, lead0_v, lead0_prob),
      _model_lead_v3(zeros, zeros, 0.0),
      _model_lead_v3(zeros, zeros, 0.0),
    ]

    control.controlsState.longControlState = LongCtrlState.pid if self.enabled else LongCtrlState.off
    ss.selfdriveState.experimentalMode = self.e2e
    ss.selfdriveState.personality = self.personality
    ss.selfdriveState.enabled = bool(self.enabled)
    control.controlsState.forceDecel = self.force_decel
    car_state.carState.vEgo = float(self.speed)
    car_state.carState.aEgo = float(self.a_ego)
    # the car's standstill flag, under 0.104 m/s of wheel speed on Hyundai
    car_state.carState.standstill = bool(self.speed < STANDSTILL_SPEED)
    car_state.carState.vCruise = float(v_cruise * 3.6)
    car_control.carControl.orientationNED = [0., float(pitch), 0.]
    self.pitch = float(pitch)

    # ******** get controlsState messages for plotting ***
    invalid = set()
    if not radar_valid:
      invalid.add('radarState')
    if not model_valid:
      invalid.add('modelV2')
    sm = _PlantSubMaster({'radarState': radar.radarState,
                           'carState': car_state.carState,
                           'carControl': car_control.carControl,
                           'controlsState': control.controlsState,
                           'selfdriveState': ss.selfdriveState,
                           'vehicleParameters': lp.vehicleParameters,
                           'modelV2': model.modelV2},
                          mono_time=int(self.current_time * 1e9), invalid=invalid)
    self.last_sm = sm
    self.planner.update(sm)
    self.a_target = float(self.planner.output_a_target)
    self.should_stop = self.planner.output_should_stop
    fcw = self.planner.fcw
    self.distance_lead = self.distance_lead + v_lead * self.ts

    # ******** run the car ********
    # controlsd's LongControl on the held plan, then the ESC's standstill brake and the actuator, the wheels and the car's speed filter
    cs = messaging.new_message('carState').carState
    # opendbc's limits, the Palisade's own; a port's clamp (the Civic's Nidec near its set speed) is not what these judge
    accel_limits = CarInterfaceBase.get_pid_accel_limits(self.CP, self.speed, v_cruise)
    for _ in range(round(DT_MDL / DT_CTRL)):
      cs.vEgo, cs.aEgo = self.speed, self.a_ego
      request = float(self.long_control.update(self.enabled, cs, self.a_target, self.should_stop, accel_limits))
      stop_req = self.long_control.long_control_state == LongCtrlState.stopping
      held = self.esc is not None and self.esc.update(stop_req, request, self.speed <= 0.0, self.pitch, DT_CTRL)
      accel = self.actuator.update(request, self.speed) if self.actuator is not None else request
      if self.speed > 0.0:
        accel += self.accel_error
      if request > CREEP_HOLD_REQUEST:
        accel += self.creep * max(1.0 - self.speed / DRIVELINE_CREEP_SPEED, 0.0)
      if self.speed <= 0.0:
        accel = 0.0 if held else max(accel, 0.0)
      self.speed = max(self.speed + accel * DT_CTRL, 0.0)
      self.distance = self.distance + self.speed * DT_CTRL
      self.a_ego = self.CI.CS.update_speed_kf(self.speed)[1]
    self.acceleration = accel if self.speed > 0.0 else 0.0

    # *** radar model ***
    if self.lead_relevancy:
      d_rel = np.maximum(0., self.distance_lead - self.distance)
      v_rel = v_lead - self.speed
    else:
      d_rel = 200.
      v_rel = 0.

    # print at 5hz
    # if (self.rk.frame % (self.rate // 5)) == 0:
    #   print("%2.2f sec   %6.2f m  %6.2f m/s  %6.2f m/s2   lead_rel: %6.2f m  %6.2f m/s"
    #         % (self.current_time, self.distance, self.speed, self.acceleration, d_rel, v_rel))


    # ******** update prevs ********
    self.rk.monitor_time()

    return {
      "distance": self.distance,
      "speed": self.speed,
      "acceleration": self.acceleration,
      "a_target": self.a_target,
      "should_stop": self.should_stop,
      "forcing": self.planner.force_stops.forcing,
      "held": self.esc is not None and self.esc.holding,
      "distance_lead": self.distance_lead,
      "fcw": fcw,
    }

# simple engage in standalone mode
def plant_thread():
  plant = Plant()
  while 1:
    plant.step()


if __name__ == "__main__":
  plant_thread()
