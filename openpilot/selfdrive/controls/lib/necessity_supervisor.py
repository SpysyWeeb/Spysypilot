from dataclasses import dataclass
import math

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_lead import (MODEL_LEAD_PROB_MIN, LeadObservation, closing_speed, time_to_collision,
                                                                 total_decel_requirement)

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
PURSUIT_TAIL_S = 3.0          # s, after excess braking eases behind a lead that is pulling away, the low jerk cost stays this
                              # long: the recovery trigger disarms at the zero crossing, exactly where the MPC has to swing to
                              # acceleration as fast as it braked (route 0x3b t=392: +0.26 m/s^2 early in the pickup)

JERK_SCALE_MIN = 0.3
JERK_SCALE_RATE = 1.5
ONSET_LEAD_DECEL = 0.4
ONSET_PAD_MAX = 0.45
STOPPED_LEAD_PAD_MAX = 0.75
STOPPED_LEAD_FULL_DECEL = 1.2   # m/s^2 of required deceleration at which the near-stopped-lead pad reaches its ceiling; BLoTv2's
                                # field value, kept verbatim. BLoTv3 only removed the `required_decel < ONSET_MAX_A_REQ` gate that
                                # made this pad vanish above 1.5 instead of staying at its ceiling; the ramp still ends at 1.2
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
LEAD_SAME_CAR_JUMP = 2.0     # m; at standstill radard re-associates the car it has within 1.1 m, and another car taking the slot
                             # reads 3.3 m or more away
LEAD_STANDING_NOISE = 0.25   # m; a car standing ahead of the car at rest reads within two of the radar's 0.1 m range steps of where
                             # it stood (its quantisation and one step of noise), and a third step is motion


@dataclass(frozen=True, slots=True)
class LongitudinalPolicy:
  jerk_scale: float
  t_follow_pad: float


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
  # the one owner of the release at standstill behind the car the car rests behind: from the first frame the MPC's own plan
  # stops the car for the lead it came to rest behind its stop bit stays up until the lead's departure is corroborated, and a
  # release taken back while the lead still stands where it was released restores the hold it ended. All of it belongs to that
  # lead, a return on its radar track or where it stood. The lead missing is not evidence: the hold waits for it, a release
  # stands with its cancel window paused, the confirmation neither counts nor restarts, and only a car standing beyond it, or
  # the model's open road while it is missing, ends them, the hold as a launch; the lead's own track taking the slot back makes
  # it that lead again. A car that takes the slot at rest with no hold standing, where the MPC leaves it no room, is the car the
  # car now rests behind and is held from that frame; the car it took the slot in front of shows beyond it whenever the radar
  # misses it, so neither a car beyond it nor the open road is evidence that it left, and only its own departure releases it.
  # A measured speed counts only while the model confirms the return (radard's low-speed override reads a standing car as
  # moving) and from the second frame of its radar track. A lead read at the bumper is not released until it reads clear of it
  def __init__(self, dt=DT_MDL, absence_frames=1):
    self.dt = dt
    self.absence_frames = absence_frames
    self.reset()

  def reset(self):
    self._forget_lead()
    self._arrival = True
    self._arrival_tracks = set()

  def _forget_lead(self):
    self._prediction_s = 0.0
    self._cancel_s = 0.0
    self._released = False
    self._released_at = 0.0
    self._stopped_for = False
    self._arrival = False
    self.lead = LeadObservation()  # the lead it judges, as it last read it
    self._taken = False  # that lead took the slot at rest
    self._stood = 0.0
    self._absent_frames = 0
    self.holding = False
    self.cancelled = False
    self.handed_back = False

  def update(self, active, standstill, lead, predicted_speed, lead_stop=False, path_open=False, no_room=False):
    # lead_stop: the MPC's own plan stops the car for this lead this frame; path_open: the road ahead is confirmed open this
    # frame; no_room: the MPC binds this lead and from rest it leaves the car no room to move
    if not (active and standstill):
      self.reset()
      return False
    held = self.holding
    self.cancelled = self.handed_back = False
    same_track = not self.lead.present or lead.track_id == self.lead.track_id
    other = lead.present and not (self.lead.present and ((same_track and lead.track_id != -1) or
                                                         abs(lead.distance - self.lead.distance) <= LEAD_SAME_CAR_JUMP))
    beyond = other and self.lead.present and self.holding and lead.distance > self._stood + LEAD_STANDING_NOISE
    if not lead.present or (beyond and self._taken):
      # the hold waits, and a release stands with its cancel window and its confirmation as they were. Before any lead, the
      # MPC's own absence hold bounds how late the lead the car came to rest behind can appear
      if not self.lead.present:
        self._absent_frames += 1
        self._arrival = self._arrival and self._absent_frames < self.absence_frames
      elif path_open and not (self._taken and self.holding):
        self._forget_lead()
        self.handed_back = held
      return self._released
    if other:
      if no_room and not self.holding and (self.lead.present or not self._arrival) and lead.track_id not in self._arrival_tracks:
        # a car taking the slot at rest, no hold standing, where the MPC leaves no room: the car now rests behind it, held
        self._forget_lead()
        self._taken = self.holding = True
      elif self.lead.present and self.holding and not beyond:
        # a nearer car, or the held one back from behind it: nothing has left
        self._prediction_s = 0.0
      else:
        if self.lead.present:
          self._forget_lead()
        self._arrival = self._arrival or lead.track_id in self._arrival_tracks
    if self._arrival and lead.track_id != -1:
      self._arrival_tracks.add(lead.track_id)
    self.lead = lead
    stops_car = lead_stop and self._arrival
    self._stopped_for = self._stopped_for or self.holding or stops_car

    instant = lead.speed > LEAD_DEPARTURE_SPEED
    measured = lead.speed > LEAD_MOVING_SPEED and lead.model_prob > MODEL_LEAD_PROB_MIN and same_track
    forecast = predicted_speed is not None and predicted_speed > LEAD_DEPARTURE_SPEED
    released = self._released
    # a frame without departure evidence restarts the confirmation and counts toward a standing release's cancel
    self._prediction_s = self._prediction_s + self.dt if (measured or forecast) and not instant else 0.0
    self._cancel_s = self._cancel_s + self.dt if self._released and not (instant or measured or forecast) else 0.0
    if self._released and (lead.distance <= 0.0 or self._cancel_s + 1e-9 >= LEAD_DEPARTURE_CANCEL):
      # taken back after LEAD_DEPARTURE_CANCEL without departure evidence, and at once on a lead read at the bumper, which is not
      # released until it reads clear of it
      self._released = False
      self._cancel_s = 0.0
      # a lead that has moved since it was released is leaving: its launch stays the MPC's plan
      self.cancelled = self._stopped_for and lead.distance <= self._released_at + LEAD_STANDING_NOISE
    elif lead.distance > 0.0 and (instant or self._prediction_s + 1e-9 >= LEAD_DEPARTURE_CONFIRM):
      self._released = True
    if self._released and not released:
      self._released_at = lead.distance
    holding = not self._released and (self.holding or stops_car or self.cancelled)
    # where the car the hold is on stands: a nearer car taking the slot does not move it
    self._stood = max(self._stood, lead.distance) if self.holding and holding else lead.distance
    self.holding = holding
    self.handed_back = held and not self.holding and not self._released
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
    self._pursuit_s = 0.0
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

        # the pursuit tail: excess braking behind a lead that is already pulling away keeps the low jerk cost for a
        # bounded time after the braking eases, so the swing to acceleration is not made with the stiff cost
        if recovery_active and lead.acceleration > LAUNCH_ALEAD_ON:
          self._pursuit_s = PURSUIT_TAIL_S
        elif lead.acceleration <= 0.0:
          self._pursuit_s = 0.0
        else:
          self._pursuit_s = max(self._pursuit_s - self.dt, 0.0)

        if recovery_active or model_active or launch_active or self._pursuit_s > 0.0:
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
          pad_target = max(pad_target, STOPPED_LEAD_PAD_MAX * min(required_decel / STOPPED_LEAD_FULL_DECEL, 1.0))

        self._responsive = scale_target < 1.0 or pad_target > 0.0
      else:
        self._responsive = False
        self._pursuit_s = 0.0
        for trigger in self._triggers:
          trigger.reset()
    else:
      if not lead.present:
        self._responsive = False
      self._pursuit_s = 0.0
      for trigger in self._triggers:
        trigger.reset()

    # keep whatever softening was built while necessity-braking through the crawl and standstill transition
    if lead.present and v_ego <= MIN_SPEED and self._responsive:
      scale_target = min(scale_target, self.jerk_scale)

    self.jerk_scale = self._slew(self.jerk_scale, scale_target, JERK_SCALE_RATE)
    self.t_follow_pad = self._slew(self.t_follow_pad, pad_target, ONSET_RATE_UP if pad_target > self.t_follow_pad else ONSET_RATE_DOWN)
    return LongitudinalPolicy(self.jerk_scale, self.t_follow_pad)
