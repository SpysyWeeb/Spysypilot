"""
Original concept and implementation by SpysyWeeb (github.com/SpysyWeeb).

BLT -- Better Longitudinal Tune. Design doc: docs/BLT.md.

A necessity supervisor over the stock longitudinal MPC. Each frame it computes what
braking the situation physically requires from the RAW measured lead (no pessimistic
extrapolation), then modulates knobs the solver already accepts at runtime -- the
jerk/accel-change cost weight, the t_follow gap parameter, and the lead-accel
extrapolation decay. It never wraps v_cruise and never clamps the controller output;
the solver keeps shaping everything, so intensity stays proportional by construction
and the behavior is model-independent (this is the radar+MPC leg, identical under
every driving model, in chill and experimental mode alike).

Three interventions, mapped to the three stock mechanisms found in field data:

  recovery boost   -- braking held past necessity (route 39 t=617: 2.2s of post-
                      necessity braking). While the plan demands meaningfully more
                      than necessity, scale the accel-change cost down so the solver
                      stops paying rent on stale deceleration and relaxes itself.
  pessimism floor  -- radard drives aLeadTau toward 0 while a lead brakes hard, which
                      makes the MPC plan against a lead that brakes forever across
                      the 10s horizon. While physics says the situation is mild,
                      floor the decay so the prediction stays a forecast, not a doom.
  dynamic onset    -- the quadratic obstacle cost barely moves for a mildly-slowing
                      lead, then explodes late (late-then-hard). When a lead starts
                      braking and necessity is still mild, pad the t_follow parameter
                      so the distance cost engages early and gently -- the owner's
                      "start smooth, peak mid" shape, produced inside the solver.

Safety posture: every intervention stands down entirely below MIN_TTC and only moves
within modest extensions of ranges the stock personalities already span. The solver's
hard constraints (ACCEL_MIN/MAX, danger-zone slack) are untouched, and the boost can
only make the solver more responsive to its own optimum -- braking deepens FASTER too
if the world worsens mid-boost. All outputs are rate-limited: no steps, ever.
"""
from openpilot.common.realtime import DT_MDL

# --- necessity (validated against the owner's stops: routes 35/37) ---
STOP_MARGIN = 4.0        # m, owner's median standstill gap
TIME_MARGIN = 1.0        # s, headway term of the necessity margin -- under his following
MIN_GAP_BUDGET = 1.0     # m, guards the division

# --- recovery boost ---
EXCESS_ON = 0.4          # m/s^2, plan demand beyond necessity that arms the boost
EXCESS_OFF = 0.15        # m/s^2, disarm level (hysteresis)
EXCESS_DEBOUNCE = 0.4    # s, excess must persist this long before boosting
JERK_SCALE_MIN = 0.3     # floor on the jerk/accel-change weight multiplier (stock
                         # personalities already span 0.5-1.0; this extends modestly)
JERK_SCALE_RATE = 1.5    # 1/s, multiplier slew (continuity)

# --- pessimism floor ---
TAU_FLOOR = 0.5          # min aLeadTau while active (larger tau = faster decay of the
                         # extrapolated lead deceleration = less pessimism)
TAU_MILD_A_REQ = 1.2     # m/s^2, only floor pessimism while necessity is mild
TAU_RATE = 1.5           # 1/s, floor slew

# --- dynamic onset ---
ONSET_LEAD_DECEL = 0.4   # m/s^2, lead braking at least this much opens the gap term early
ONSET_T_FOLLOW_MAX = 1.9 # s, padded gap parameter ceiling (stock relaxed is 1.75)
ONSET_MAX_A_REQ = 1.5    # m/s^2, above this the situation is real: stock gap, no padding
ONSET_RATE_UP = 0.4      # s per s, t_follow pad slew up
ONSET_RATE_DOWN = 0.25   # s per s, pad slew back down

# --- global safety gate ---
MIN_TTC = 3.5            # s, below this every intervention stands down (stock solver)
MIN_SPEED = 1.0          # m/s, nothing to supervise at crawl (settle/hold own that)


class BLTSupervisor:
  def __init__(self):
    self.jerk_scale = 1.0
    self.t_follow_pad = 0.0
    self.tau_floor = 0.0
    self._excess_s = 0.0

  def _slew(self, cur: float, target: float, rate: float) -> float:
    step = rate * DT_MDL
    return min(max(target, cur - step), cur + step)

  def update(self, sm, a_plan: float, t_follow_base: float):
    """a_plan: the planner's current desired acceleration (negative = braking).
    Returns (jerk_factor_scale, t_follow, tau_floor) for the MPC's runtime knobs."""
    lead = sm['radarState'].leadOne
    v_ego = max(float(sm['carState'].vEgo), 0.0)

    scale_target, pad_target, tau_target = 1.0, 0.0, 0.0

    if lead.status and sm.valid['radarState'] and v_ego > MIN_SPEED:
      v_lead = max(float(lead.vLeadK), 0.0)
      d_rel = float(lead.dRel)
      a_lead = float(lead.aLeadK)

      gap_budget = max(d_rel - (STOP_MARGIN + TIME_MARGIN * v_lead), MIN_GAP_BUDGET)
      a_req = max((v_ego * v_ego - v_lead * v_lead) / (2.0 * gap_budget), 0.0)
      closing = v_ego - v_lead
      ttc = d_rel / closing if closing > 0.3 else 100.0

      if ttc > MIN_TTC:
        # recovery boost: demand beyond necessity, sustained, means the solver is
        # holding stale braking -- let it move
        excess = -a_plan - a_req
        if excess > EXCESS_ON:
          self._excess_s += DT_MDL
        elif excess < EXCESS_OFF:
          self._excess_s = 0.0
        if self._excess_s >= EXCESS_DEBOUNCE:
          scale_target = JERK_SCALE_MIN

        # pessimism floor: the lead is braking hard enough that radard is driving its
        # extrapolation tau toward zero, but physics says we are not in trouble
        if a_lead < -0.5 and a_req < TAU_MILD_A_REQ:
          tau_target = TAU_FLOOR

        # dynamic onset: lead has started braking, situation still mild -- open the
        # gap term early so the distance cost pulls gently now instead of hard later
        if a_lead < -ONSET_LEAD_DECEL and a_req < ONSET_MAX_A_REQ:
          pad_target = max(ONSET_T_FOLLOW_MAX - t_follow_base, 0.0)
      else:
        self._excess_s = 0.0
    else:
      self._excess_s = 0.0

    self.jerk_scale = self._slew(self.jerk_scale, scale_target, JERK_SCALE_RATE)
    self.t_follow_pad = self._slew(self.t_follow_pad, pad_target, ONSET_RATE_UP if pad_target > self.t_follow_pad else ONSET_RATE_DOWN)
    self.tau_floor = self._slew(self.tau_floor, tau_target, TAU_RATE)

    return self.jerk_scale, t_follow_base + self.t_follow_pad, self.tau_floor
