import math

import numpy as np

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_lead import total_decel_requirement

# The landing law: the planner's last word on braking as a stop closes.
#
# Every stop -- a committed red light, the MPC column behind a stopped lead, the model's own request in Experimental
# mode, the cruise floor under a shaping cap -- reaches the arbitration as its own candidate with its own idea of the
# last metres, and the field says they all land harder than the owner does: over routes 23-27 (2026-08-29) the lead-free
# stops exceeded this law in a third to two thirds of their last-3 m/s frames, by up to 1.1 m/s^2 at walking pace
# (route 27 t=1052, "head banging"). The owner brakes as hard as a stop needs early and lets off as the car settles, and
# always lets off at the very end. The law is a corridor on the arbitrated target through the landing:
#
# * a bound, falling linearly with speed, that only ever removes surplus braking (a close lead keeps full authority, the
#   physics of stopping behind a lead are never blocked, a watchdog opens it if the car stops slowing under it);
# * a floor, so the stop completes against the transmission's creep torque instead of coasting (the MPC column lets go
#   of the brake entirely by 0.2 m/s and hovers around zero);
# * both taper to one small "kiss" of braking at walking pace, so the wheels stop under a whisper of brake and the body
#   does not rock -- route 28 (2026-08-30): a flat 0.40 floor held to the wheel stop, and its on/off edge against the
#   MPC's hover made the target alternate -0.40 / +0.1 every frame for the last half second, a throttle blip and a clamp
#   inside 0.2 s. The landing therefore latches: once started it lasts through the hover and through standstill, and ends
#   only on a launch -- the lead-departure pre-release, a hold release, or the plan positive for LAUNCH_FRAMES in a row.
#
# This is the owner's original Smooth Stops design (June 2026, sunnypilot smooth-stops-dev), rehomed in the planner:
# the one place that knows the lead, the plan and the stop intent. LongControl's thin handoff keeps only the deferred
# clamp and its own kiss below the stop bit.
LANDING_SPEED = 3.5          # m/s, the law acts below this; at the top the bound (2.75 m/s^2) exceeds any comfort approach
STOP_INTENT_SPEED = 0.5      # m/s, the MPC's horizon must reach below this for a slowdown to count as a stop
KISS_SPEED = 0.15            # m/s, from here down the corridor is the kiss alone (the Palisade reports standstill at ~0.1)
KISS_DECEL = 0.15            # m/s^2, the braking the wheels stop under; the hold and the car's own clamp take over after
CREEP_SPEED = 1.0            # m/s, the floor peaks at CREEP_DECEL here: enough to keep slowing against creep torque, and
CREEP_DECEL = 0.40           # m/s^2, where the MPC column eases to anyway ...
CREEP_FADE_SPEED = 1.5       # m/s, ... and it is gone here: a queue rolling at 2 m/s is not held to a stop's floor
BOUND_BP = [KISS_SPEED, 0.5, LANDING_SPEED]    # allowed braking: 0.70 * v + 0.30 above 0.5 m/s (1.35 m/s^2 at 1.5 m/s,
BOUND_V = [KISS_DECEL, 0.65, 2.75]             # 0.65 at 0.5), straight down to the kiss below
FLOOR_BP = [KISS_SPEED, CREEP_SPEED, CREEP_FADE_SPEED]
FLOOR_V = [KISS_DECEL, CREEP_DECEL, 0.0]
LAUNCH_FRAMES = 3            # consecutive frames of a positive plan that end a landing: a hover alternates, a launch climbs
LEAD_FULL_AUTHORITY = 5.0    # m, a lead this close lifts the bound; the floor stays, coasting at it is no better
LEAD_LANDING_GAP = 4.0       # m, never soften the braking needed to stop this far behind the lead
LEAD_MIN_GAP_BUDGET = 0.5    # m, floor of that gap budget, keeps the requirement finite
STALL_S = 1.0                # s, the car not slowing under the corridor for this long ...
STALL_PROGRESS = 0.02        # m/s, ... (speed not falling by this below its low mark) ...
STALL_RELEASE_RATE = 0.15    # m/s^2 per s, ... shifts the whole corridor toward more braking this fast, while rolling


def landing_bound(v_ego):
  # the most braking the law allows at this speed
  return float(np.interp(v_ego, BOUND_BP, BOUND_V))


def landing_floor(v_ego):
  # the least braking a landing keeps on at this speed
  return float(np.interp(v_ego, FLOOR_BP, FLOOR_V))


class StopLanding:
  def __init__(self, dt=DT_MDL):
    self.dt = dt
    self.reset()

  def reset(self):
    self.landing = False
    self.active = False       # the law changed the plan this frame
    self._positive_frames = 0
    self._reset_watchdog()

  def _reset_watchdog(self):
    self._v_low = math.inf
    self._stall_s = 0.0

  def _stall_release(self, v_ego):
    # progress is the speed making a new low; without it for STALL_S the corridor shifts at STALL_RELEASE_RATE. Only while
    # rolling: at standstill the clamp holds the car, and a drifting target would only drag the next launch's start down
    if v_ego <= KISS_SPEED:
      self._reset_watchdog()
      return 0.0
    if v_ego < self._v_low - STALL_PROGRESS:
      self._v_low = v_ego
      self._stall_s = 0.0
    else:
      self._stall_s += self.dt
    return max(self._stall_s - STALL_S, 0.0) * STALL_RELEASE_RATE

  def update(self, a_target, v_ego, lead, stop_intent, launch=False):
    """Bound the arbitrated acceleration target through a landing; returns the plan unchanged otherwise.

    lead is the planner's LeadObservation, stop_intent says the plan ends in a stop this frame, launch says the planner
    itself is letting the car go (a lead-departure pre-release, a hold release). A landing starts on intent with the plan
    braking more than the kiss below LANDING_SPEED, and then lasts -- through the MPC's hover around zero and through standstill -- until a
    launch, the plan climbing above zero for LAUNCH_FRAMES in a row, or the speed leaving the window.
    """
    if v_ego >= LANDING_SPEED:
      self.reset()
      return a_target
    if not self.landing:
      # a landing starts on intent with the plan braking more than the kiss: a hover frame right after a launch is not a stop
      if not (stop_intent and a_target < -KISS_DECEL):
        self.active = False
        return a_target
      self.landing = True
      self._positive_frames = 0
      self._reset_watchdog()
    else:
      self._positive_frames = self._positive_frames + 1 if a_target > 0.0 else 0
      if launch or self._positive_frames >= LAUNCH_FRAMES:
        self.reset()
        return a_target

    release = self._stall_release(v_ego)
    floor = landing_floor(v_ego)
    floor = floor + release if floor > 0.0 else 0.0
    if lead.present and lead.distance < LEAD_FULL_AUTHORITY:
      bound = math.inf
    else:
      # the bound softens surplus braking only: whatever stopping LEAD_LANDING_GAP behind the lead needs always passes
      bound = max(landing_bound(v_ego) + release, total_decel_requirement(v_ego, lead, LEAD_LANDING_GAP, LEAD_MIN_GAP_BUDGET))
    output = min(a_target, -floor) if floor > 0.0 else a_target
    output = max(output, -bound, ACCEL_MIN)
    self.active = output != a_target
    return output
