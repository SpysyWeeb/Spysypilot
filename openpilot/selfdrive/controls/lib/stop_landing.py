import math

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_lead import total_decel_requirement

# The landing law: the planner's last bound on braking as a stop closes.
#
# Every stop -- a committed red light, the MPC column behind a stopped lead, the model's own request in Experimental
# mode, the cruise floor under a shaping cap -- reaches the arbitration as its own candidate with its own idea of the
# last metres, and the field says they all land harder than the owner does: over routes 23-27 (2026-08-29) the lead-free
# stops exceeded this law in a third to two thirds of their last-3 m/s frames, by up to 1.1 m/s^2 at walking pace
# (route 27 t=1052, "head banging"). The owner brakes as hard as a stop needs early and lets off as the car settles.
# Allowed braking falling linearly with speed is that shape for any stop: under the bound the speed decays
# exponentially into the stop (time constant 1/LANDING_K), the release is felt as a taper, never a step. The law only
# ever removes surplus braking. A close lead keeps full authority, the physics of stopping behind a lead are never
# blocked, and a watchdog releases the bound if the car stops slowing under it. This is the owner's original Smooth
# Stops design (June 2026), rehomed in the planner: the one place that knows the lead, the plan and the stop intent.
LANDING_SPEED = 3.5          # m/s, the law acts below this; at the top the bound (2.75 m/s^2) exceeds any comfort approach
STOP_INTENT_SPEED = 0.5      # m/s, the MPC's horizon must reach below this for a slowdown to count as a stop
LANDING_K = 0.70             # 1/s, allowed braking = K * v + C: at most 1.35 m/s^2 at 1.5 m/s, 0.65 at 0.5 m/s
LANDING_C = 0.30             # m/s^2, allowed at standstill; the hold and the car's own clamp take over there
STANDSTILL_SPEED = 0.1       # m/s, below this the stop bit, the hold and the ESP own everything
LEAD_FULL_AUTHORITY = 5.0    # m, a lead this close is not a landing matter
LEAD_LANDING_GAP = 4.0       # m, never soften the braking needed to stop this far behind the lead
LEAD_MIN_GAP_BUDGET = 0.5    # m, floor of that gap budget, keeps the requirement finite
STALL_S = 1.0                # s, the car not slowing under the bound for this long ...
STALL_PROGRESS = 0.02        # m/s, ... (speed not falling by this below its low mark) ...
STALL_RELEASE_RATE = 0.15    # m/s^2 per s, ... releases the bound this fast: the law must never hold speed
CREEP_SPEED = 1.0            # m/s, below this a landing plan brakes at least CREEP_DECEL, against the transmission's creep torque
CREEP_DECEL = 0.40           # m/s^2


def landing_bound(v_ego):
  # the most braking the law allows at this speed
  return LANDING_K * v_ego + LANDING_C


class StopLanding:
  def __init__(self, dt=DT_MDL):
    self.dt = dt
    self.reset()

  def reset(self):
    self.landing = False
    self.active = False       # the law changed the plan this frame
    self._v_low = math.inf
    self._stall_s = 0.0

  def _reset_watchdog(self):
    self._v_low = math.inf
    self._stall_s = 0.0

  def _stall_release(self, v_ego):
    # progress is the speed making a new low; without it for STALL_S the bound opens up at STALL_RELEASE_RATE
    if v_ego < self._v_low - STALL_PROGRESS:
      self._v_low = v_ego
      self._stall_s = 0.0
    else:
      self._stall_s += self.dt
    return max(self._stall_s - STALL_S, 0.0) * STALL_RELEASE_RATE

  def update(self, a_target, v_ego, lead, stop_intent):
    """Bound the arbitrated acceleration target through the last metres of a stop; returns the plan unchanged otherwise.

    lead is the planner's LeadObservation; stop_intent says the plan ends in a stop this frame. A landing starts on
    intent and then lasts until the plan lifts or the speed leaves the window: the intent evidence may flicker
    (the model's stop bit does), the landing must not.
    """
    self.landing = (stop_intent or self.landing) and a_target < 0.0 and STANDSTILL_SPEED < v_ego < LANDING_SPEED
    if not self.landing:
      self.reset()
      return a_target
    if lead.present and lead.distance < LEAD_FULL_AUTHORITY:
      self._reset_watchdog()
      self.active = False
      return a_target

    bound = landing_bound(v_ego) + self._stall_release(v_ego)
    # the law softens surplus braking only: whatever stopping LEAD_LANDING_GAP behind the lead needs always passes
    bound = max(bound, total_decel_requirement(v_ego, lead, LEAD_LANDING_GAP, LEAD_MIN_GAP_BUDGET))
    output = max(a_target, -bound, ACCEL_MIN)
    if v_ego < CREEP_SPEED:
      output = min(output, -CREEP_DECEL)
    self.active = output != a_target
    return output
