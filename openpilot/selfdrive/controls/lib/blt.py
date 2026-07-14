"""
Original concept and implementation by SpysyWeeb (github.com/SpysyWeeb).

BLT -- Better Longitudinal Tune. Design doc: docs/BLT.md.

A necessity supervisor over the stock longitudinal MPC. Each frame it computes what
braking the situation physically requires from the RAW measured lead, then modulates
knobs the solver already accepts at runtime -- the jerk/accel-change cost weight and
the t_follow gap parameter. It never wraps v_cruise and never clamps the controller output;
the solver keeps shaping everything, so intensity stays proportional by construction
and the behavior is model-independent (this is the radar+MPC leg, identical under
every driving model, in chill and experimental mode alike).

Interventions, mapped to the stock mechanisms found in field data. Two others have
been retired: the lead-accel-extrapolation pessimism floor (2026-07-11, obsolete when
the MPC moved to model-predicted lead trajectories -- no extrapolation left to patch)
and gap forgiveness (2026-07-11, route 46: tuned on legacy-MPC route-3f data, it
ratcheted steady-following headway down to its floor and silently absorbed gap-button
personality changes; whether the fall-back ping-pong it patched even occurs under
model-predicted leads is now a field question, answered without the mask):

  recovery boost   -- braking held past necessity (route 39 t=617: 2.2s of post-
                      necessity braking). While the plan demands meaningfully more
                      than necessity, scale the accel-change cost down so the solver
                      stops paying rent on stale deceleration and relaxes itself. A
                      second, independent trigger arms the same boost off the model's
                      OWN predicted lead trajectory (leadsV3) for hard braking specifically
                      -- replay-validated against 14 real routes: even on the events where
                      the radar-driven trigger above eventually arms, the model would have
                      gotten there ~3s sooner, median, because it forecasts hard braking
                      before radar can measure it. Deliberately narrow: the threshold sits
                      well clear of the mild band dynamic onset owns below, so the model
                      never gets a vote on necessity or on subtle slowdowns -- only on
                      whether the solver's jerk cost is allowed to relax sooner.
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

# --- necessity, relative frame (route 3c lesson: the old headway-margin form exploded to
# nonsense at the owner's 1.2-1.6s following and BLT sat out the onset of every close-follow
# event; necessity is "match the lead's braking plus kill the closing energy") ---
D_MIN = 4.0              # m, owner's median standstill gap; the closing-energy denominator floor
MIN_GAP_BUDGET = 1.0     # m, guards the division

# --- recovery boost ---
MIN_BOOST_BRAKE = 0.8    # m/s^2, only meaningful braking is worth boosting out of -- the relative
                         # necessity reads near zero in ordinary following and small trims should
                         # not put the solver in its loose-cost mode
EXCESS_ON = 0.4          # m/s^2, plan demand beyond necessity that arms the boost
EXCESS_OFF = 0.15        # m/s^2, disarm level (hysteresis)
EXCESS_DEBOUNCE = 0.4    # s, excess must persist this long before boosting
JERK_SCALE_MIN = 0.3     # floor on the jerk/accel-change weight multiplier (stock
                         # personalities already span 0.5-1.0; this extends modestly)
JERK_SCALE_RATE = 1.5    # 1/s, multiplier slew (continuity)

# --- model-predicted hard-event early arm ---
MODEL_HARD_THRESH = -0.5    # m/s^2, predicted lead decel (leadsV3 v-slope, 0->2s) that arms
                             # the boost early. Calibrated + replay-validated on real routes:
                             # genuinely hard events (radar aLeadK sustained < -1.0) cluster
                             # here at the moment radar crosses threshold; the mild band the
                             # dynamic-onset pad owns rarely reaches half this magnitude, so
                             # this stays selective for hard events without extra logic
MODEL_ARM_DEBOUNCE = 0.3    # s, predicted decel must persist this long -- matched to the
                             # sustain window the replay validation used
MODEL_LEAD_PROB_MIN = 0.5   # matches NewLeadMpc's own trust gate (long_mpc.py) -- when the
                             # model isn't confident, this path is simply silent and behavior
                             # is identical to radar-only

# --- dynamic onset ---
ONSET_LEAD_DECEL = 0.4   # m/s^2, lead braking at least this much opens the gap term early
ONSET_PAD_MAX = 0.45     # s, max t_follow pad ON TOP of the personality base (route 3c: an
                         # absolute 1.9 ceiling meant +0.65 on aggressive -- a manufactured 5m
                         # deficit and phantom braking for a lead easing off by -0.5)
ONSET_FULL_DECEL = 1.5   # m/s^2, lead decel at which the pad reaches its full value; the pad is
                         # PROPORTIONAL below that, so a mild slowdown gets a mild gap-opening
ONSET_MAX_A_REQ = 1.5    # m/s^2, above this the situation is real: stock gap, no padding
ONSET_RATE_UP = 0.4      # s per s, t_follow pad slew up
ONSET_RATE_DOWN = 0.5    # s per s, pad slew back down (0.25 lingered 2-4s after events)

# --- global safety gate ---
MIN_TTC = 3.5            # s, below this every intervention stands down (stock solver)
MIN_SPEED = 1.0          # m/s, nothing to supervise at crawl (settle/hold own that)


class BLTSupervisor:
  def __init__(self):
    self.jerk_scale = 1.0
    self.t_follow_pad = 0.0
    self._excess_s = 0.0
    self._model_excess_s = 0.0

  def _slew(self, cur: float, target: float, rate: float) -> float:
    step = rate * DT_MDL
    return min(max(target, cur - step), cur + step)

  @staticmethod
  def _model_predicted_decel(sm):
    """0->2s slope of the model's own predicted lead speed (leadsV3), gated the same
    way NewLeadMpc trusts it. Only the shape is used, so no radar anchoring is needed."""
    leads_v3 = sm['modelV2'].leadsV3
    if len(leads_v3) < 1:
      return None
    l0 = leads_v3[0]
    if l0.prob < MODEL_LEAD_PROB_MIN or len(l0.v) < 2:
      return None
    return (float(l0.v[1]) - float(l0.v[0])) / 2.0

  def update(self, sm, a_plan: float, t_follow_base: float):
    """a_plan: the planner's current desired acceleration (negative = braking).
    Returns (jerk_factor_scale, t_follow) for the MPC's runtime knobs."""
    lead = sm['radarState'].leadOne
    v_ego = max(float(sm['carState'].vEgo), 0.0)

    scale_target, pad_target = 1.0, 0.0

    if lead.status and sm.valid['radarState'] and v_ego > MIN_SPEED:
      v_lead = max(float(lead.vLeadK), 0.0)
      d_rel = float(lead.dRel)
      a_lead = float(lead.aLeadK)

      closing = v_ego - v_lead
      # relative-frame necessity: hold the lead's own deceleration, plus enough to shed the
      # closing energy before the standstill margin
      a_req = max(-a_lead, 0.0) + max(closing, 0.0) ** 2 / (2.0 * max(d_rel - D_MIN, MIN_GAP_BUDGET))
      ttc = d_rel / closing if closing > 0.3 else 100.0

      # emergency = low TTC AND high necessity together. A controlled approach to a stopped
      # lead naturally runs TTC < 3.5 (10m at 3.6 m/s), and standing down there dropped the
      # stopped-lead pad mid-approach (route 3e replay) -- a desired-gap release right where
      # firmness matters most
      if ttc > MIN_TTC or a_req < ONSET_MAX_A_REQ:
        # recovery boost: meaningful braking beyond necessity, sustained, means the solver
        # is holding stale deceleration -- let it move
        excess = -a_plan - a_req
        if excess > EXCESS_ON and -a_plan > MIN_BOOST_BRAKE:
          self._excess_s += DT_MDL
        elif excess < EXCESS_OFF or -a_plan <= MIN_BOOST_BRAKE:
          self._excess_s = 0.0
        if self._excess_s >= EXCESS_DEBOUNCE:
          scale_target = JERK_SCALE_MIN

        # model early-arm: independent trigger, same target knob, never combined into the
        # excess magnitude above -- an unconfident or mild model reading simply never fires,
        # it can't stack with a marginal radar reading to produce an unwarranted early boost
        pred_decel = self._model_predicted_decel(sm)
        if pred_decel is not None and pred_decel < MODEL_HARD_THRESH:
          self._model_excess_s += DT_MDL
        else:
          self._model_excess_s = 0.0
        if self._model_excess_s >= MODEL_ARM_DEBOUNCE:
          scale_target = JERK_SCALE_MIN

        # dynamic onset: lead has started braking, situation still mild -- open the gap
        # term early, PROPORTIONALLY to how hard the lead is braking. Never pad during
        # the recovery (ego already at/below lead speed) or while the lead accelerates:
        # holding the gap open there is exactly the lag the owner felt (route 3c t=1227)
        recovering = v_ego <= v_lead + 0.2 or a_lead > 0.2
        if a_lead < -ONSET_LEAD_DECEL and a_req < ONSET_MAX_A_REQ and not recovering:
          pad_target = ONSET_PAD_MAX * min(-a_lead / ONSET_FULL_DECEL, 1.0)
        # stopped-lead onset (route 3e): a lead that is already stopped never trips the
        # decel-based pad, so nothing opened the gap term early and the car carried speed
        # to a 2.2m squeak-stop. Drive the pad from closing-energy necessity instead.
        if v_lead < 2.0 and 0.3 < a_req < ONSET_MAX_A_REQ and not recovering:
          pad_target = max(pad_target, ONSET_PAD_MAX * min(a_req / 1.2, 1.0))
      else:
        self._excess_s = 0.0
        self._model_excess_s = 0.0
    else:
      self._excess_s = 0.0
      self._model_excess_s = 0.0

    self.jerk_scale = self._slew(self.jerk_scale, scale_target, JERK_SCALE_RATE)
    self.t_follow_pad = self._slew(self.t_follow_pad, pad_target, ONSET_RATE_UP if pad_target > self.t_follow_pad else ONSET_RATE_DOWN)

    return self.jerk_scale, t_follow_base + self.t_follow_pad
