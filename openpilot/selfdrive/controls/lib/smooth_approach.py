"""
Original concept and implementation by SpysyWeeb (github.com/SpysyWeeb).

Smooth Approach. The stock ACC MPC can watch a lead decelerate for many seconds
while barely braking, then demand near-maximum deceleration once the gap finally
gets tight ("late, then hard"). Human drivers do the opposite: they commit to the
slowdown early and spread the same speed change over a gentle, single application.

This module is that early commitment. Each frame it computes a comfort *envelope*
speed: the fastest the car could be going right now and still match the lead's
speed at the desired following distance using no more than A_APPROACH of braking.
That envelope is min()'d into v_cruise before the MPC runs, so the MPC starts
shedding speed as soon as the lead's behavior calls for it and shapes the whole
deceleration itself. Parameters were fitted from the owner's own manual braking
(route 00000035, 19 stops: onset at ~2.9 s headway, plateau -1.4..-1.8 m/s^2).

Safety posture: the cap can only ever ask for *earlier, gentler* braking -- it
never raises v_cruise and never bypasses the MPC's lead constraint, which still
owns collision avoidance (COMFORT_BRAKE and the distance cost are untouched).
A wrong cap therefore degrades to a too-cautious slowdown, never a late brake.
"""
import math

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE, get_T_FOLLOW

A_APPROACH = 1.5   # m/s^2, envelope deceleration; the "plateau" braking level the cap plans around.
                   # Deliberately below the MPC's COMFORT_BRAKE (2.5) -- this is the comfort profile,
                   # the MPC keeps its full authority for when the world misbehaves.
NO_CAP = float('inf')


class SmoothApproach:
  """Stateless comfort envelope on v_cruise. min() the result into v_cruise pre-MPC."""

  def update(self, sm) -> float:
    lead = sm['radarState'].leadOne
    if not (lead.status and sm.valid['radarState']):
      return NO_CAP

    v_lead = max(float(lead.vLeadK), 0.0)  # Kalman-filtered lead speed; clamp radar noise at standstill

    # Match the lead's speed at the desired following distance (personality gap at the lead's
    # speed, plus the standstill margin) -- for a stopped lead this reduces to STOP_DISTANCE.
    t_follow = get_T_FOLLOW(sm['selfdriveState'].personality)
    gap_budget = float(lead.dRel) - (STOP_DISTANCE + t_follow * v_lead)

    # v_cap^2 = v_lead^2 + 2*a*d: the speed from which A_APPROACH of braking over the gap budget
    # lands exactly at the lead's speed. Faster/receding leads push the cap above v_cruise (inactive);
    # a closing gap pulls it down smoothly and the required deceleration never exceeds A_APPROACH.
    return math.sqrt(max(v_lead * v_lead + 2.0 * A_APPROACH * gap_budget, 0.0))
