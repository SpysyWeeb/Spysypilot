import math

import numpy as np

from openpilot.selfdrive.controls.lib.necessity_supervisor import LEAD_SAME_CAR_JUMP, LEAD_STANDING_NOISE
from openpilot.selfdrive.controls.lib.stop_landing import KISS_SPEED

# At rest the stop bit is a hold or a committed launch. The Palisade's ESC keeps its standstill brake until the request has
# built up past about LAUNCH_EXIT_ACCEL, sooner the higher it is, so a bit that falls under a lower request leaves the car on
# the brake while the plan catches up. A launch commits on the planner's own release or on a plan asking LAUNCH_EXIT_ACCEL by
# itself, and asks at least LAUNCH_ACCEL until the wheels turn; until then it is taken back at no cost when a stop is asked for,
# the plan brakes past -LAUNCH_EXIT_ACCEL, the release has lapsed with a candidate's stop bit up, an owner holds the car again,
# or the MPC binds something new or a lead in contact. An owner holding the car again over a release, something new and a lead
# in contact spend every release standing, and nothing commits, the plan included, until an owner releases the car anew. Once
# the wheels turn the plan owns the car: the lift goes back at the planner's cruise jerk, at once on braking or a stop bit. A
# launch on evidence alone that its lead did not answer makes the next launch on evidence wait for the plan.
LAUNCH_ACCEL = 1.0       # m/s^2, the ESC lets go about 0.55-0.8 s after a step to it
LAUNCH_EXIT_ACCEL = 0.5  # m/s^2, about the least the ESC lets its standstill brake go under; at rest the lift does not override
                         # a plan braking firmer than that, whichever candidate asks it
NO_LEAD = (-1, math.inf)  # what a launch is given for with no lead in sight: no car is that one


def same_car(slot, reading, given):
  # given: (radar track, distance) of the pre-release's lead, in leadOne. The same car at rest: its radar track in either slot,
  # or a reading in leadOne within LEAD_SAME_CAR_JUMP of it, as radard re-associates a standing car within 1.1 m; a standing car
  # does not come nearer by more than that
  track_id, distance = given
  return reading.distance >= distance - LEAD_SAME_CAR_JUMP and ((track_id != -1 and reading.track_id == track_id) or
                                                               (slot == 0 and reading.distance <= distance + LEAD_SAME_CAR_JUMP))


class StandstillExit:
  def __init__(self, dt, release_jerk):
    self.dt = dt
    self.release_jerk = release_jerk  # (speed breakpoints, m/s^3), the planner's cruise jerk
    self.reset()

  def reset(self):
    self.at_rest = self.holding = self.launching = self.releasing = False
    self.committed = self.taken_back = False
    self.lift = 0.0
    self._forget_releases()
    self._unanswered = False
    self._lead_at_launch = self._lead_at_rest = None
    self._travel = 0.0

  def _forget_releases(self):
    self._edge = self._level_prev = self._owner_hold_prev = self._spent = False
    self._given = None

  def _spend(self):
    # every release standing is spent: only one given anew launches the car
    self._spent, self._edge = True, False

  def update(self, active, standstill, v_ego, held, owner_hold, stop_request, release_level, release_edge, plan_accel, lead, obstacle):
    # this frame's stop bit at rest (holding), and a launch committed or taken back on it. lead: the pre-release's lead (leadOne)
    # as it last read it; obstacle: (radar slot, reading) of the lead the MPC binds while from rest it leaves no room, else None
    self.committed = self.taken_back = False
    if not active or not math.isfinite(v_ego):
      self.reset()
      return
    if not math.isfinite(plan_accel):
      return
    rest = bool(standstill) and v_ego < KISS_SPEED
    left = self.at_rest and not rest
    self.at_rest = rest
    if not rest:
      # the wheels turned: the plan owns the car and the lift goes back from where it was. A stop bit hands the car to
      # LongControl's stop, which resumes from the plan
      if self.launching:
        self.launching = False
        self.releasing = self.lift > 0.0
      self.releasing = self.releasing and not held
      if left and self._travel > 0.0:
        self._lead_at_launch = None  # a judgement still pending at the rest the car leaves is dropped
      self._travel += max(v_ego, 0.0) * self.dt
      self.holding = False
      self._forget_releases()
      return
    self.releasing = False
    if self._lead_at_launch is not None and self._travel > 0.0 and lead.present:
      # back at rest after a launch on evidence alone that moved the car, on the first frame its lead is read: the same stop,
      # behind a lead that did not go. A missing lead is no answer
      moved = lead.distance + self._travel - self._lead_at_launch > LEAD_STANDING_NOISE
      self._unanswered = self._travel <= LEAD_SAME_CAR_JUMP and not moved
      self._lead_at_rest = lead.distance if self._unanswered else None
      self._lead_at_launch = None
    # the mark clears once the lead reads beyond the standing noise of where it stood at that rest
    if self._unanswered and lead.present and self._lead_at_rest is not None and lead.distance > self._lead_at_rest + LEAD_STANDING_NOISE:
      self._unanswered = False
    # a release given anew (the pre-release's rising, its hold handed back, a Force Stops green) ends a spent hold and is given
    # for the pre-release's lead as it reads it now; an owner holding the car again while a release stands, launched or not,
    # spends it, and one given while an owner holds is not one
    given_anew = (release_level and not self._level_prev) or (release_edge and not owner_hold)
    if given_anew:
      self._spent, self._given = False, (lead.track_id, lead.distance) if lead.present else NO_LEAD
    if owner_hold and not self._owner_hold_prev and (self.launching or self._edge or self._level_prev):
      self._spend()
    self._level_prev, self._owner_hold_prev = release_level, owner_hold
    self._edge = self._edge or (release_edge and not owner_hold)
    # something new: the MPC binds a lead with no room that is not the one the launch is given for, the pre-release's lead as read
    # when its release was given or the plan alone committed, before either as it reads it now. A lead in contact is no launch's
    given = self._given or ((lead.track_id, lead.distance) if lead.present else NO_LEAD)
    new = obstacle is not None and not same_car(*obstacle, given)
    contact = obstacle is not None and obstacle[1].distance <= 0.0
    if new:
      self._spend()
    usable = (release_level or self._edge) and not self._unanswered

    if self.launching:
      if owner_hold or new or contact or stop_request or plan_accel < -LAUNCH_EXIT_ACCEL or (not usable and held):
        self.launching = False
        self.taken_back = self.holding = True
        self.lift = 0.0
        self._lead_at_launch = None
        if contact:
          self._spend()
      return
    if held or self._spent or plan_accel < -LAUNCH_EXIT_ACCEL or not (usable or plan_accel >= LAUNCH_EXIT_ACCEL):
      self.holding = True
      return
    self.launching = self.committed = True
    self.holding = False
    self.lift = LAUNCH_ACCEL
    by_plan = plan_accel >= LAUNCH_EXIT_ACCEL
    self._unanswered = self._unanswered and not by_plan
    self._lead_at_launch = lead.distance if (lead.present and not by_plan) else None
    self._given = given
    self._travel = 0.0

  def apply(self, a_target, v_ego):
    # the published target: at least the lift while launching, the lift handed back once the wheels turn, else the plan
    if not math.isfinite(a_target):
      return a_target
    if self.launching:
      return max(a_target, self.lift)
    if self.releasing:
      if a_target < 0.0:
        self.releasing, self.lift = False, 0.0
        return a_target
      self.lift -= float(np.interp(v_ego, *self.release_jerk)) * self.dt
      if a_target >= self.lift:
        self.releasing, self.lift = False, 0.0
        return a_target
      return self.lift
    return a_target
