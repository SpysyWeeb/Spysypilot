import math

import numpy as np

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.selfdrive.controls.lib.longitudinal_lead import closing_speed, total_decel_requirement
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE

# The landing law: the planner's last word on braking as a stop closes.
#
# Every stop -- a committed red light, the MPC column behind a stopped lead, the model's own request in Experimental
# mode, the cruise floor under a shaping cap -- reaches the arbitration as its own candidate with its own idea of the
# last metres, and they all land harder than a stop needs to. A stop should brake as hard as it needs early, let off as
# the car settles, and always let off at the very end. The law is a corridor on the car's deceleration through the landing:
#
# * a bound, falling with speed, that only removes surplus braking: whatever stopping STOP_DISTANCE behind a lead needs
#   always passes, and so does whatever stopping short of a closer lead needs in the room left after the actuator delay;
# * a floor, so the stop completes against the transmission's creep torque instead of coasting (the MPC column lets go
#   of the brake entirely by 0.2 m/s and hovers around zero);
# * both taper to one small "kiss" of braking at walking pace, so the wheels stop under a whisper of brake and the body
#   does not rock.
#
# The car answers a request one actuator delay later, so each edge is requested that much earlier (request_breakpoints).
# The landing latches, since an on/off floor against the MPC's hover would alternate the target every frame: once started
# it lasts through the hover and through standstill, and ends only on a launch -- the lead-departure pre-release or its
# hold handed back, a Force Stops release (the commitment over, not a hold rolling on into a moving commitment) -- the
# plan positive for LAUNCH_FRAMES in a row above KISS_SPEED, or the speed leaving the window; a launch the planner takes
# back at standstill latches it again. Whatever it still holds the plan below when it ends is handed back at the planner's
# cruise jerk, so no exit steps the car; a reset drops it.
LANDING_SPEED = 3.5          # m/s, the law acts below this; at the top the bound (2.75 m/s^2) exceeds any comfort approach
STOP_INTENT_SPEED = 0.5      # m/s, the MPC's horizon must reach below this for a slowdown to count as a stop
KISS_SPEED = 0.40            # m/s, from here down the car rides the kiss alone
KISS_DECEL = 0.15            # m/s^2, the braking the wheels stop under; the hold and the car's own clamp take over after
CREEP_SPEED = 1.0            # m/s, the floor peaks at CREEP_DECEL here: enough to keep slowing against creep torque, and
CREEP_DECEL = 0.40           # m/s^2, where the MPC column eases to anyway ...
CREEP_FADE_SPEED = 1.5       # m/s, ... and it is gone here: a queue rolling at 2 m/s is not held to a stop's floor
BOUND_BP = [KISS_SPEED, 0.9, LANDING_SPEED]    # the car's deceleration against its speed: at most 0.70 * v + 0.30 above
BOUND_V = [KISS_DECEL, 0.93, 2.75]             # 0.9 m/s, straight down to the kiss below ...
FLOOR_BP = [KISS_SPEED, CREEP_SPEED, CREEP_FADE_SPEED]
FLOOR_V = [KISS_DECEL, CREEP_DECEL, 0.0]       # ... and at least the floor
LAUNCH_FRAMES = 3            # consecutive frames of a positive plan that end a landing: a hover alternates, a launch climbs
LEAD_MIN_GAP_BUDGET = 0.5    # m, the least room a stop behind a lead is planned in, and the least it plans to leave to the lead
# Anti-stall: while rolling, the car's shortfall against the floor presses both edges toward more braking, and drains as
# the car slows again. It lives where the floor does and fades out with it, and it is held but not applied at standstill,
# where the hold clamps the car and a pressed target would only seed the next launch
STANDSTILL_SPEED = 0.10      # m/s, about the car's standstill flag (0.104 m/s of wheel speed on Hyundai)
STALL_GAIN = 1.0             # 1/s, keeps a gain margin above 3 against the ESP's dead time and bite lag
STALL_DEADBAND = 0.05        # m/s^2, about one standard deviation of aEgo while braking
STALL_MAX = 0.5              # m/s^2, the most the corridor is pressed down


def landing_bound(v_ego):
  # the most braking the car should carry at this speed
  return float(np.interp(v_ego, BOUND_BP, BOUND_V))


def landing_floor(v_ego):
  # the least braking the car should carry at this speed through a landing
  return float(np.interp(v_ego, FLOOR_BP, FLOOR_V))


def stop_within(speed, room):
  # the deceleration that sheds speed within room; with no room left any braking the plan asks is needed
  if speed <= 0.0:
    return 0.0
  return speed * speed / (2.0 * room) if room > 0.0 else math.inf


def request_breakpoints(bp, values, action_t, steps=100):
  # each breakpoint below LANDING_SPEED moves up to the speed from which a car riding that edge reaches it after action_t
  h = action_t / steps
  out = []
  for v in bp:
    if v < LANDING_SPEED:
      for _ in range(steps):
        v += h * float(np.interp(v + 0.5 * h * float(np.interp(v, bp, values)), bp, values))
    out.append(v)
  return out


class StopLanding:
  def __init__(self, dt, action_t, release_jerk):
    self.dt = dt
    self.action_t = action_t
    # (speed breakpoints, m/s^3): the rate a held output is handed back at, the planner's cruise jerk
    self.release_jerk = release_jerk
    self.bound_bp = request_breakpoints(BOUND_BP, BOUND_V, action_t)
    self.floor_bp = request_breakpoints(FLOOR_BP, FLOOR_V, action_t)
    self.reset()

  def reset(self):
    self.landing = False
    self.releasing = False
    self.output = 0.0
    self._held = False
    self._positive_frames = 0
    self.integral = 0.0

  def bound(self, v_ego):
    # the most braking requested at this speed
    return float(np.interp(v_ego, self.bound_bp, BOUND_V))

  def floor(self, v_ego):
    # the least braking requested at this speed through a landing
    return float(np.interp(v_ego, self.floor_bp, FLOOR_V))

  def lead_requirement(self, v_ego, lead):
    # the braking that stops the car STOP_DISTANCE behind the lead; a closer lead needs whatever stops the car short of it
    # in the room left once the requests already on their way have played out, which grows without limit as that room closes
    if not lead.present:
      return 0.0
    closing = closing_speed(v_ego, lead)
    requirement = max(total_decel_requirement(v_ego, lead, STOP_DISTANCE, LEAD_MIN_GAP_BUDGET),
                      stop_within(closing, lead.distance - closing * self.action_t - LEAD_MIN_GAP_BUDGET))
    if lead.acceleration < 0.0:
      lead_stop_distance = lead.speed * lead.speed / (-2.0 * lead.acceleration)
      room = lead.distance + lead_stop_distance - v_ego * self.action_t - LEAD_MIN_GAP_BUDGET
      requirement = max(requirement, stop_within(v_ego, room))
    return requirement

  def update(self, a_target, v_ego, lead, stop_intent, launch=False, a_ego=None, launch_cancelled=False):
    """Bound the arbitrated acceleration target through a landing; returns the plan unchanged otherwise.

    lead is the planner's LeadObservation, stop_intent says the plan ends in a stop this frame, launch says the planner
    itself is letting the car go (a lead-departure pre-release or its hold handed back, a Force Stops release), a_ego is
    the car's measured acceleration (None: the anti-stall integral holds), launch_cancelled says the planner took back a
    launch it issued at standstill. A landing starts on intent with the plan braking more than the kiss below
    LANDING_SPEED, unless the planner is issuing a launch that same frame, or at standstill on a cancelled launch, and then
    lasts -- through the MPC's hover around zero and through standstill -- until a launch, the plan climbing above zero for
    LAUNCH_FRAMES in a row above KISS_SPEED, or the speed leaving the window. Then an output it held below the plan rises
    at the release jerk until it meets the plan.
    """
    if not math.isfinite(v_ego):
      # a non-finite speed resets everything, a release included
      self.reset()
      return a_target
    v_ego = max(v_ego, 0.0)
    if not math.isfinite(a_target):
      # a NaN target (an MPC solve diverging before its own reset lands) must not enter the corridor's clamps -- min/max
      # return NaN regardless of the other side -- so it is passed through exactly as upstream's own NaN handling would,
      # touching neither the latch nor the integral: a transient bad frame must not end or restart a landing
      return a_target
    if self.landing:
      # a climbing plan ends the landing only while rolling: at standstill the MPC's hover can drift positive for
      # a few frames against a standing lead, and the launch authority there is the planner's own release
      self._positive_frames = self._positive_frames + 1 if a_target > 0.0 and v_ego > KISS_SPEED else 0
      if launch or v_ego >= LANDING_SPEED or self._positive_frames >= LAUNCH_FRAMES:
        self.landing = False
        self.releasing = self._held
    elif not launch and ((v_ego < LANDING_SPEED and stop_intent and a_target < -KISS_DECEL) or
                         (launch_cancelled and v_ego <= STANDSTILL_SPEED)):
      # a landing starts on intent with the plan braking more than the kiss: a hover frame right after a launch is not a stop,
      # and a launch frame never starts one, or every frame the pre-release ends would re-latch and toggle the stop bit. A
      # launch taken back at standstill restores the landing it ended, whatever the plan's hover
      self.landing = True
      self._positive_frames = 0
      self.integral = 0.0

    output = a_target
    if self.landing:
      if v_ego > STANDSTILL_SPEED and a_ego is not None and math.isfinite(a_ego):
        self.integral += STALL_GAIN * (a_ego + landing_floor(v_ego) - STALL_DEADBAND) * self.dt
      # the press fades out with the floor it is measured against: none of it is held where the landing asks for no braking
      self.integral = min(max(self.integral, 0.0), float(np.interp(v_ego, [CREEP_SPEED, CREEP_FADE_SPEED], [STALL_MAX, 0.0])))
      press = self.integral if v_ego > STANDSTILL_SPEED else 0.0
      floor = self.floor(v_ego) + press
      bound = max(self.bound(v_ego) + press, self.lead_requirement(v_ego, lead))
      output = min(a_target, -floor) if floor > 0.0 else a_target
      output = max(output, -bound, ACCEL_MIN)
    if self.releasing:
      # the floor and the press it held go back at the same rate, and a landing starting again meanwhile cannot drop them
      ceiling = self.output + float(np.interp(v_ego, *self.release_jerk)) * self.dt
      self.releasing = output > ceiling
      output = min(output, ceiling)
    self._held = output < a_target
    self.output = output
    return output
