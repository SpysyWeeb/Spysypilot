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


class Plant:
  messaging_initialized = False

  def __init__(self, lead_relevancy=False, speed=0.0, distance_lead=2.0,
               enabled=True, only_lead2=False, only_radar=False, e2e=False, personality=0, force_decel=False,
               curve=None, torque_factor=2.7, torque_friction=0.11, curve_model_scale=1.0):
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
    # a world-fixed curve (start distance, length, curvature) the fake model shows along its path; the fake steering
    # holds the path up to its torque authority and reports the torque controller state the curve policy reads
    self.curve = curve
    self.curve_model_scale = curve_model_scale       # the model reads the curve at this fraction of its true curvature
    self.torque_factor = torque_factor
    self.torque_friction = torque_friction
    self.lateral_accel = 0.0
    self.torque = 0.0

    self.rk = Ratekeeper(self.rate, print_delay_threshold=100.0)
    self.ts = 1. / self.rate
    time.sleep(0.1)
    self.sm = messaging.SubMaster(['longitudinalPlan'])

    from opendbc.car.honda.values import CAR
    from opendbc.car.honda.interface import CarInterface

    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC)
    if self.curve is not None:
      torque = CP.lateralTuning.init('torque')
      torque.latAccelFactor = self.torque_factor
      torque.latAccelOffset = 0.0
      torque.friction = self.torque_friction
    self.planner = LongitudinalPlanner(CP, init_v=self.speed)

  def _curvature_at(self, world_x):
    # one (start, length, curvature) or a list of them: a bend that tightens or opens along the way
    segments = self.curve if isinstance(self.curve[0], tuple | list) else [self.curve]
    for start, length, curvature in segments:
      if start <= world_x <= start + length:
        return curvature
    return 0.0

  def _plan_curve(self, model, controls_state, car_control):
    # path curvature along the model's own positions, and a steering state: the car holds the path up to the torque
    # authority factor * (1 - friction); beyond it the torque pins at 1 and the tracking error is the shortfall
    positions = np.asarray(model.position.x, dtype=float)
    speeds = np.asarray(model.velocity.x, dtype=float)
    model.position.y = [0.0] * len(positions)          # the policy measures arc length from x and y
    curvatures = np.array([self._curvature_at(self.distance + x) for x in positions])
    rate = log.XYZTData.new_message()
    rate.z = [float(k * self.curve_model_scale * v) for k, v in zip(curvatures, speeds, strict=True)]
    model.orientationRate = rate
    desired = self.speed ** 2 * self._curvature_at(self.distance)
    authority = self.torque_factor * (1.0 - self.torque_friction)
    self.lateral_accel = min(desired, authority)
    self.torque = min(desired / self.torque_factor + self.torque_friction, 1.0) if desired > 0.0 else 0.0
    state = controls_state.lateralControlState.init('torqueState')
    state.active = True
    state.output = float(self.torque)
    state.error = float(desired - self.lateral_accel)
    state.actualLateralAccel = float(self.lateral_accel)
    state.desiredLateralAccel = float(desired)
    state.saturated = bool(self.torque >= 1.0)
    car_control.latActive = True

  @property
  def current_time(self):
    return float(self.rk.frame) / self.rate

  def step(self, v_lead=0.0, prob_lead=1.0, v_cruise=50., pitch=0.0, prob_throttle=1.0):
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
    if self.curve is not None:
      self._plan_curve(model.modelV2, control.controlsState, car_control.carControl)
    model.modelV2.meta.disengagePredictions.gasPressProbs = [float(prob_throttle) for _ in range(6)]

    control.controlsState.longControlState = LongCtrlState.pid if self.enabled else LongCtrlState.off
    ss.selfdriveState.experimentalMode = self.e2e
    ss.selfdriveState.personality = self.personality
    control.controlsState.forceDecel = self.force_decel
    car_state.carState.vEgo = float(self.speed)
    car_state.carState.standstill = bool(self.speed < 0.01)
    car_state.carState.vCruise = float(v_cruise * 3.6)
    car_control.carControl.orientationNED = [0., float(pitch), 0.]

    # ******** get controlsState messages for plotting ***
    sm = {'radarState': radar.radarState,
          'carState': car_state.carState,
          'carControl': car_control.carControl,
          'controlsState': control.controlsState,
          'selfdriveState': ss.selfdriveState,
          'vehicleParameters': lp.vehicleParameters,
          'modelV2': model.modelV2}
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
      "lateral_accel": self.lateral_accel,
      "torque": self.torque,
    }

# simple engage in standalone mode
def plant_thread():
  plant = Plant()
  while 1:
    plant.step()


if __name__ == "__main__":
  plant_thread()
