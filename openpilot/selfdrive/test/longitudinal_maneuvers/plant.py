#!/usr/bin/env python3
import time
import numpy as np

from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper, DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU


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


class Plant:
  messaging_initialized = False

  def __init__(self, lead_relevancy=False, speed=0.0, distance_lead=2.0,
               enabled=True, only_lead2=False, only_radar=False, e2e=False, personality=0, force_decel=False,
               stop_line=None, stop_line_horizon_s=5.0):
    self.rate = 1. / DT_MDL

    if not Plant.messaging_initialized:
      Plant.radar = messaging.pub_sock('radarState')
      Plant.controls_state = messaging.pub_sock('controlsState')
      Plant.selfdrive_state = messaging.pub_sock('selfdriveState')
      Plant.car_state = messaging.pub_sock('carState')
      Plant.plan = messaging.sub_sock('longitudinalPlan')
      Plant.messaging_initialized = True

    self.v_lead_prev = 0.0

    self.distance = 0.
    self.speed = speed
    self.should_stop = False
    self.acceleration = 0.0

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

    self.rk = Ratekeeper(self.rate, print_delay_threshold=100.0)
    self.ts = 1. / self.rate
    time.sleep(0.1)
    self.sm = messaging.SubMaster(['longitudinalPlan'])

    from opendbc.car.honda.values import CAR
    from opendbc.car.honda.interface import CarInterface

    self.planner = LongitudinalPlanner(CarInterface.get_non_essential_params(CAR.HONDA_CIVIC), init_v=self.speed)

  def _plan_stop_line(self, model):
    # the model's path ends at the line with a constant-deceleration speed profile; its e2e request is
    # deliberately softer than that need, as the real model's early request is
    d_line = self.stop_line - self.distance
    t = np.array(ModelConstants.T_IDXS)
    if d_line <= 0.5 or self.speed < 0.3:
      x = np.zeros_like(t)
      v = np.zeros_like(t)
      model.action.shouldStop = True
      model.action.desiredAcceleration = float(min(self.acceleration, 0.0)) if self.speed >= 0.3 else 0.0
    elif d_line <= self.speed * self.stop_line_horizon_s:
      a_req = self.speed ** 2 / (2.0 * d_line)
      t_stop = self.speed / a_req
      tc = np.minimum(t, t_stop)
      x = self.speed * tc - 0.5 * a_req * tc ** 2
      v = np.maximum(self.speed - a_req * tc, 0.0)
      model.action.shouldStop = bool(d_line < 3.0)
      model.action.desiredAcceleration = float(-0.7 * a_req)
    else:
      return
    position = log.XYZTData.new_message()
    position.x = [float(xx) for xx in x]
    model.position = position
    velocity = log.XYZTData.new_message()
    velocity.x = [float(vv) for vv in v]
    velocity.x[0] = float(self.speed)
    model.velocity = velocity

  @property
  def current_time(self):
    return float(self.rk.frame) / self.rate

  def step(self, v_lead=0.0, prob_lead=1.0, v_cruise=50., pitch=0.0, prob_throttle=1.0,
           radar_valid=True, model_valid=True):
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
    lead.dRel = float(d_rel)
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
    if self.stop_line is not None:
      self._plan_stop_line(model.modelV2)
    model.modelV2.meta.disengagePredictions.gasPressProbs = [float(prob_throttle) for _ in range(6)]

    # lead0 mirrors the radar lead above; lead1/lead2 are shaped but carry no probability
    lead_t = np.asarray(ModelConstants.LEAD_T_IDXS, dtype=np.float64)
    if a_lead < 0.0:
      stop_t = np.minimum(lead_t, max(-v_lead / a_lead, 0.0))
    else:
      stop_t = lead_t
    lead0_v = np.maximum(v_lead + a_lead * lead_t, 0.0)
    lead0_x = np.maximum.accumulate(d_rel + v_lead * stop_t + 0.5 * a_lead * stop_t ** 2)
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
    car_state.carState.aEgo = float(self.acceleration)
    car_state.carState.standstill = bool(self.speed < 0.01)
    car_state.carState.vCruise = float(v_cruise * 3.6)
    car_control.carControl.orientationNED = [0., float(pitch), 0.]

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
    self.acceleration = self.planner.output_a_target
    if self.planner.output_should_stop:
      self.acceleration = min(-0.5, self.acceleration)
    self.speed = self.speed + self.acceleration * self.ts
    self.should_stop = self.planner.output_should_stop
    fcw = self.planner.fcw
    self.distance_lead = self.distance_lead + v_lead * self.ts

    # ******** run the car ********
    #print(self.distance, speed)
    if self.speed <= 0:
      self.speed = 0
      self.acceleration = 0
    self.distance = self.distance + self.speed * self.ts

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
      "should_stop": self.should_stop,
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
