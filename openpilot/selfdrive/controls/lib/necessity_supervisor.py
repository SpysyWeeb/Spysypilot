from dataclasses import dataclass
import math

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_lead import closing_speed, time_to_collision, total_decel_requirement

# The supervisor never commands acceleration. It only moves two solver inputs the MPC already
# owns, the acceleration-change/jerk cost and the following time, in proportion to measured need.
D_MIN = 4.0
MIN_GAP_BUDGET = 1.0

MIN_BOOST_BRAKE = 0.8
EXCESS_ON = 0.4
EXCESS_OFF = 0.15
EXCESS_DEBOUNCE = 0.4
MODEL_HARD_THRESHOLD = -0.5
MODEL_ARM_DEBOUNCE = 0.3

LAUNCH_VREL_ON = 0.5
LAUNCH_VREL_OFF = 0.2
LAUNCH_ALEAD_ON = 0.5
LAUNCH_SHORTFALL_ON = 0.6
LAUNCH_SHORTFALL_OFF = 0.2
LAUNCH_DEBOUNCE = 0.4
RATCHET_LEAD_BRAKE = 0.2

JERK_SCALE_MIN = 0.3
JERK_SCALE_RATE = 1.5
ONSET_LEAD_DECEL = 0.4
ONSET_PAD_MAX = 0.45
STOPPED_LEAD_PAD_MAX = 0.75
ONSET_FULL_DECEL = 1.5
ONSET_MAX_A_REQ = 1.5
STAND_DOWN_SHORTFALL_MIN = 0.15
ONSET_RATE_UP = 0.8
ONSET_RATE_DOWN = 0.5
MIN_TTC = 3.5
MIN_SPEED = 1.0

LEAD_DEPARTURE_SPEED = 0.5   # m/s of lead speed that releases at once; 0.3 released the hold for a lead that crept at
                             # 0.65 m/s and stopped again (route 0x2b t=1540: StopReq cycled off and back on under a
                             # standing car); slower creeps go through the confirmed path below
LEAD_MOVING_SPEED = 0.25
LEAD_DEPARTURE_CONFIRM = 0.2
LEAD_DEPARTURE_CANCEL = 0.2


@dataclass(frozen=True, slots=True)
class LongitudinalPolicy:
  jerk_scale: float
  t_follow_pad: float
  stand_down: bool


class DebouncedTrigger:
  def __init__(self, debounce, dt):
    self.debounce = debounce
    self.dt = dt
    self._seconds = 0.0

  def reset(self):
    self._seconds = 0.0

  def step(self, arm, disarm):
    if arm:
      self._seconds += self.dt
    elif disarm:
      self.reset()
    return self._seconds + 1e-9 >= self.debounce


class LeadDeparturePreRelease:
  # releases only the MPC stop bit once a stopped lead's departure is corroborated
  def __init__(self, dt=DT_MDL):
    self.dt = dt
    self.reset()

  def reset(self):
    self._prediction_s = 0.0
    self._cancel_s = 0.0
    self._released = False

  def update(self, active, standstill, lead, predicted_speed):
    if not (active and standstill and lead.present):
      self.reset()
      return False

    if lead.speed > LEAD_DEPARTURE_SPEED:
      self._released = True
      self._prediction_s = 0.0
      self._cancel_s = 0.0
      return True

    departure_valid = lead.speed > LEAD_MOVING_SPEED or (predicted_speed is not None and predicted_speed > LEAD_DEPARTURE_SPEED)
    if departure_valid:
      self._prediction_s += self.dt
      self._cancel_s = 0.0
      if self._prediction_s + 1e-9 >= LEAD_DEPARTURE_CONFIRM:
        self._released = True
    else:
      self._prediction_s = 0.0
      if self._released:
        self._cancel_s += self.dt
        if self._cancel_s + 1e-9 >= LEAD_DEPARTURE_CANCEL:
          self.reset()
    return self._released


class NecessitySupervisor:
  def __init__(self, dt=DT_MDL):
    self.dt = dt
    self._recovery = DebouncedTrigger(EXCESS_DEBOUNCE, dt)
    self._model = DebouncedTrigger(MODEL_ARM_DEBOUNCE, dt)
    self._launch = DebouncedTrigger(LAUNCH_DEBOUNCE, dt)
    self._triggers = (self._recovery, self._model, self._launch)
    self.reset()

  def reset(self):
    self.jerk_scale = 1.0
    self.t_follow_pad = 0.0
    self._responsive = False
    for trigger in self._triggers:
      trigger.reset()

  def _slew(self, current, target, rate):
    step = rate * self.dt
    return min(max(target, current - step), current + step)

  def update(self, lead, v_ego, a_mpc, predicted_lead_accel=None):
    scale_target = 1.0
    pad_target = 0.0
    stand_down = False

    if lead.present and v_ego > MIN_SPEED:
      required_decel = total_decel_requirement(v_ego, lead, D_MIN, MIN_GAP_BUDGET)
      closing = closing_speed(v_ego, lead)
      braking_shortfall = required_decel + a_mpc if math.isfinite(a_mpc) else math.inf
      # low TTC, high need and a real shortfall against the MPC's own braking: remove the adaptive policy
      stand_down = (time_to_collision(v_ego, lead) < MIN_TTC and required_decel >= ONSET_MAX_A_REQ
                    and braking_shortfall > STAND_DOWN_SHORTFALL_MIN)

      if not stand_down:
        excess = -a_mpc - required_decel
        recovery_active = self._recovery.step(arm=excess > EXCESS_ON and -a_mpc > MIN_BOOST_BRAKE,
                                              disarm=excess < EXCESS_OFF or -a_mpc <= MIN_BOOST_BRAKE)

        predicted_hard = predicted_lead_accel is not None and predicted_lead_accel < MODEL_HARD_THRESHOLD
        model_active = self._model.step(arm=predicted_hard, disarm=not predicted_hard)

        receding = lead.speed - v_ego
        shortfall = lead.acceleration - a_mpc
        launch_active = self._launch.step(arm=(receding > LAUNCH_VREL_ON and lead.acceleration > LAUNCH_ALEAD_ON
                                               and required_decel < 0.1 and shortfall > LAUNCH_SHORTFALL_ON),
                                          disarm=(shortfall < LAUNCH_SHORTFALL_OFF or lead.acceleration < LAUNCH_ALEAD_ON
                                                  or receding < LAUNCH_VREL_OFF))

        if recovery_active or model_active or launch_active:
          scale_target = JERK_SCALE_MIN

        # do not stiffen a responsive solution in the middle of a lead-braking / ego-closing reversal
        if scale_target > self.jerk_scale and lead.acceleration < -RATCHET_LEAD_BRAKE and closing > 0.0:
          scale_target = self.jerk_scale

        onset_lead_accel = lead.acceleration
        if model_active and predicted_lead_accel is not None:
          onset_lead_accel = min(onset_lead_accel, predicted_lead_accel)

        # open the obstacle cost earlier for mild lead braking or a stopped lead; the pads saturate, they never vanish
        recovering = v_ego <= lead.speed + 0.2 or lead.acceleration > 0.2
        if onset_lead_accel < -ONSET_LEAD_DECEL and not recovering:
          pad_target = ONSET_PAD_MAX * min(-onset_lead_accel / ONSET_FULL_DECEL, 1.0)
        if lead.speed < 2.0 and required_decel > 0.3 and not recovering:
          pad_target = max(pad_target, STOPPED_LEAD_PAD_MAX * min(required_decel / 1.2, 1.0))

        self._responsive = scale_target < 1.0 or pad_target > 0.0
      else:
        self._responsive = False
        for trigger in self._triggers:
          trigger.reset()
    else:
      if not lead.present:
        self._responsive = False
      for trigger in self._triggers:
        trigger.reset()

    # keep whatever softening was built while necessity-braking through the crawl and standstill transition
    if lead.present and v_ego <= MIN_SPEED and self._responsive:
      scale_target = min(scale_target, self.jerk_scale)

    self.jerk_scale = self._slew(self.jerk_scale, scale_target, JERK_SCALE_RATE)
    self.t_follow_pad = self._slew(self.t_follow_pad, pad_target, ONSET_RATE_UP if pad_target > self.t_follow_pad else ONSET_RATE_DOWN)
    return LongitudinalPolicy(self.jerk_scale, self.t_follow_pad, stand_down)
