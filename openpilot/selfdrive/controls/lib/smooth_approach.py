"""
Original concept and implementation by SpysyWeeb (github.com/SpysyWeeb).

Smooth Approach. The stock ACC MPC can watch a lead decelerate for many seconds
while barely braking, then demand near-maximum deceleration once the gap finally
gets tight ("late, then hard"). Human drivers do the opposite: they commit to the
slowdown early and spread the same speed change over a gentle, single application.

This module is that early commitment: a comfort envelope -- the fastest the car
could be going right now and still match the lead's speed at the stop margin using
no more than A_APPROACH of braking -- min()'d into v_cruise before the MPC runs.

FIELD LESSON (route 00000036): v_cruise is a *speed target the MPC erases within
seconds*, not a positional ceiling, so handing it the raw envelope commands a slam
whenever the envelope dips (it falls at ~the lead's own decel rate when the lead
brakes -- the MPC pinned ACCEL_MIN for 5s straight). The cap therefore may never
sit more than DV_MAX below the current speed: bounded, it can only ask for a
gentle, continuously-renewed trim (~-1..-1.5 m/s^2 by the measured error->accel
response), which IS the owner's plateau braking level. The same field test showed
the old 6m + 1.45s margin hard-enforced the MPC's *desired* gap (stops 6.6m back,
steady following pushed out); the margin now sits at the owner's measured stop gap
with a headway term deliberately below his normal following, so the cap only
shapes real approaches and the MPC's own soft costs keep owning the final meters.

Safety posture: the cap only ever lowers v_cruise -- earlier, gentler braking.
The MPC's lead constraint still owns collision avoidance (COMFORT_BRAKE and the
distance cost are untouched); a wrong cap degrades to a mild slowdown, never a
late brake.
"""
import math

A_APPROACH = 1.5   # m/s^2, envelope deceleration; the comfort profile the cap plans around,
                   # below the MPC's COMFORT_BRAKE (2.5) which keeps full authority
STOP_MARGIN = 4.0  # m, where the envelope reaches zero speed -- the owner's median standstill
                   # gap; NOT the MPC's STOP_DISTANCE (6.0), which is a soft-cost target the
                   # dynamics settle inside of, not a wall to stop behind
TIME_MARGIN = 1.0  # s, headway term of the margin -- deliberately UNDER the owner's steady
                   # following (~1.3-1.9s) so the cap never binds while just following
DV_MAX = 2.0       # m/s, the most the cap may sit below current speed (see field lesson)
NO_CAP = float('inf')


class SmoothApproach:
  """Stateless comfort envelope on v_cruise. min() the result into v_cruise pre-MPC."""

  def update(self, sm) -> float:
    lead = sm['radarState'].leadOne
    if not (lead.status and sm.valid['radarState']):
      return NO_CAP

    v_ego = max(float(sm['carState'].vEgo), 0.0)
    v_lead = max(float(lead.vLeadK), 0.0)  # Kalman-filtered lead speed; clamp radar noise at standstill

    # v_cap^2 = v_lead^2 + 2*a*d: the speed from which A_APPROACH of braking over the gap budget
    # lands exactly at the lead's speed at the margin. Faster/receding leads push it above v_ego.
    gap_budget = float(lead.dRel) - (STOP_MARGIN + TIME_MARGIN * v_lead)
    v_cap = math.sqrt(max(v_lead * v_lead + 2.0 * A_APPROACH * gap_budget, 0.0))

    if v_cap >= v_ego:
      return NO_CAP  # envelope satisfied: never touch cruising/following behavior

    # bounded error: however hard the envelope collapses (a braking lead drags it down at
    # ~its own decel rate), the MPC only ever sees a small trim to erase -- braking stays
    # gentle and simply renews as the car slows, trailing the envelope down
    return max(v_cap, v_ego - DV_MAX)
