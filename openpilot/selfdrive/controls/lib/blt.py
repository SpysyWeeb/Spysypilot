"""
Original concept and implementation by SpysyWeeb (github.com/SpysyWeeb).

BLoT -- Better Longitudinal Tune (branch: BLoT). Design doc, history and rationale:
docs/BLoT.md (repo root).

A necessity supervisor over the stock longitudinal MPC. Each frame it computes what
braking the situation physically requires from the RAW measured lead, then modulates
knobs the solver already accepts at runtime -- the jerk/accel-change cost weight and
the t_follow gap parameter. It never wraps v_cruise and never clamps the controller
output; the solver keeps shaping everything, so intensity stays proportional by
construction and the behavior is model-independent.

Interventions (see docs/BLoT.md for the field data behind each):

  recovery boost   -- plan holds meaningfully more braking than necessity: relax the
                      jerk cost so the solver stops paying rent on stale deceleration.
                      A second, independent trigger arms the same boost early off the
                      model's own predicted lead trajectory (leadsV3) for hard braking
                      -- the model forecasts hard events ~3s before radar measures them.
  launch boost     -- the exact mirror: lead measurably pulling away AND accelerating,
                      necessity zero, plan accel lagging the lead's own -- stale LOW
                      acceleration, same knob. Radar-driven (for launches radar leads
                      the model, the reverse of hard braking). Never touches t_follow.
  lead pre-release -- at standstill only, release the MPC hold when the radar-anchored
                      model trajectory consistently predicts the tracked lead will be
                      moving within the Palisade's measured brake-bleed time. This
                      changes only shouldStop; the MPC acceleration target is untouched.
  whiplash ratchet -- the jerk scale may relax at any time but may only STIFFEN while
                      the measured lead is braking and ego closes, so a boost live when
                      a launch flips to braking keeps its relaxed cost through the
                      swing into decel.
  dynamic onset    -- a mildly-braking or stopped lead barely moves the quadratic
                      obstacle cost until too late (late-then-hard): pad t_follow
                      proportionally so the distance cost engages early and gently.

Safety posture: every intervention stands down entirely on low TTC with high necessity
(back to full stock stiffness -- the ratchet lives inside the non-emergency path only)
and moves only within modest extensions of ranges the stock personalities already span.
The solver's hard constraints (ACCEL_MIN/MAX, danger-zone slack) are untouched, and the
boost can only make the solver more responsive to its own optimum -- braking deepens
FASTER too if the world worsens mid-boost. All outputs are rate-limited: no steps, ever.
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

# --- launch boost ---
LAUNCH_VREL_ON = 0.5        # m/s, lead measurably faster than ego -- matches the pull-away
                            # floor's gate in long_mpc.py
LAUNCH_VREL_OFF = 0.2       # disarm hysteresis on the same
LAUNCH_ALEAD_ON = 0.5       # m/s^2, lead genuinely launching, not drifting ahead at steady
                            # speed (the constant-speed pull-away is the floor's job)
LAUNCH_SHORTFALL_ON = 0.6   # m/s^2, plan accel this far below the lead's own accel arms it
LAUNCH_SHORTFALL_OFF = 0.2  # disarm: plan has essentially caught the lead's acceleration
LAUNCH_DEBOUNCE = 0.4       # s, matches EXCESS_DEBOUNCE
RATCHET_LEAD_BRAKE = 0.2    # m/s^2, lead decel beyond this (while closing) freezes any
                            # upward slew of the jerk scale -- the whiplash guard

# --- standstill lead-departure pre-release (route 8d) ---
LEAD_DEPARTURE_LOOKAHEAD = 2.0  # s, first leadsV3 prediction sample and measured median
                                 # post-controller-to-wheel-motion delay on the Palisade
LEAD_DEPARTURE_SPEED = 0.3      # m/s, predicted speed that counts as a real departure
LEAD_MOVING_SPEED = 0.25        # m/s, measured lead motion is definitive and needs no debounce
LEAD_DEPARTURE_CONFIRM = 0.2    # s, reject one-frame launch forecasts before releasing the hold
LEAD_DEPARTURE_CANCEL = 0.2     # s, reject brief forecast/radar dropouts before reapplying the hold

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


class DebouncedTrigger:
  """Shared arm/hold/reset bookkeeping for the boost triggers: accrues time while `arm`
  holds, resets when `disarm` holds, and HOLDS the accrued time in the hysteresis band
  between them. Armed once the accrual crosses the debounce."""
  def __init__(self, debounce: float):
    self.debounce = debounce
    self._s = 0.0

  def reset(self) -> None:
    self._s = 0.0

  def step(self, arm: bool, disarm: bool) -> bool:
    if arm:
      self._s += DT_MDL
    elif disarm:
      self._s = 0.0
    return self._s >= self.debounce


class LeadDeparturePreRelease:
  """Standstill-only brake-bleed compensation for a tracked lead launch.

  NewLeadMpc anchors the model trajectory to radar at t=0; use the same anchoring
  here and inspect its first future sample (2 s). A sustained predicted departure
  releases only the MPC's shouldStop bit, allowing the existing starting state to
  begin brake bleed while leaving aTarget and the driving curve unchanged.

  Loss of both predicted and measured departure evidence cancels after a short
  confirmation, preventing a release-rehold pulse on a one-frame model/radar dropout.
  """
  def __init__(self):
    self._prediction_s = 0.0
    self._cancel_s = 0.0
    self._released = False

  def reset(self) -> None:
    self._prediction_s = 0.0
    self._cancel_s = 0.0
    self._released = False

  @staticmethod
  def _predicted_speed(sm):
    leads_v3 = sm['modelV2'].leadsV3
    if len(leads_v3) < 1:
      return None
    model_lead = leads_v3[0]
    if model_lead.prob < MODEL_LEAD_PROB_MIN or len(model_lead.v) < 2:
      return None

    # Match NewLeadMpc's radar anchoring: radar owns the current speed and the model
    # contributes only its predicted delta over the first two seconds.
    lead = sm['radarState'].leadOne
    return float(lead.vLead) + float(model_lead.v[1]) - float(model_lead.v[0])

  def update(self, sm, active: bool) -> bool:
    lead = sm['radarState'].leadOne
    if not (active and sm['carState'].standstill and sm.valid['radarState'] and lead.status):
      self.reset()
      return False

    # Measured motion is definitive and bypasses the model-arm debounce. Brief loss
    # of both radar and model evidence is handled by the cancellation debounce below.
    if float(lead.vLeadK) > LEAD_MOVING_SPEED:
      self._released = True
      self._prediction_s = 0.0
      self._cancel_s = 0.0
      return True

    predicted_speed = self._predicted_speed(sm)
    if predicted_speed is not None and predicted_speed > LEAD_DEPARTURE_SPEED:
      self._prediction_s += DT_MDL
      self._cancel_s = 0.0
      if self._prediction_s + 1e-9 >= LEAD_DEPARTURE_CONFIRM:
        self._released = True
    else:
      self._prediction_s = 0.0
      if self._released:
        self._cancel_s += DT_MDL
        if self._cancel_s + 1e-9 >= LEAD_DEPARTURE_CANCEL:
          self.reset()

    return self._released


class BLTSupervisor:
  def __init__(self):
    self.jerk_scale = 1.0
    self.t_follow_pad = 0.0
    self._excess = DebouncedTrigger(EXCESS_DEBOUNCE)
    self._model = DebouncedTrigger(MODEL_ARM_DEBOUNCE)
    self._launch = DebouncedTrigger(LAUNCH_DEBOUNCE)
    self._triggers = (self._excess, self._model, self._launch)

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
        if self._excess.step(arm=excess > EXCESS_ON and -a_plan > MIN_BOOST_BRAKE,
                             disarm=excess < EXCESS_OFF or -a_plan <= MIN_BOOST_BRAKE):
          scale_target = JERK_SCALE_MIN

        # model early-arm: independent trigger, same target knob, never combined into the
        # excess magnitude above -- an unconfident or mild model reading simply never fires,
        # it can't stack with a marginal radar reading to produce an unwarranted early boost
        pred_decel = self._model_predicted_decel(sm)
        pred_hard = pred_decel is not None and pred_decel < MODEL_HARD_THRESH
        if self._model.step(arm=pred_hard, disarm=not pred_hard):
          scale_target = JERK_SCALE_MIN

        # launch boost: mirror of recovery boost. Lead measurably pulling away and
        # accelerating, nothing to brake for, plan accel lagging the lead's own. The
        # a_req check is definitionally implied by the two lead gates (receding +
        # accelerating means both necessity terms are zero) but stays as the explicit
        # statement that this trigger only exists where there is nothing to brake for
        shortfall = a_lead - a_plan
        if self._launch.step(arm=(-closing > LAUNCH_VREL_ON and a_lead > LAUNCH_ALEAD_ON
                                  and a_req < 0.1 and shortfall > LAUNCH_SHORTFALL_ON),
                             disarm=(shortfall < LAUNCH_SHORTFALL_OFF or a_lead < LAUNCH_ALEAD_ON
                                     or -closing < LAUNCH_VREL_OFF)):
          scale_target = JERK_SCALE_MIN

        # whiplash ratchet: relaxing is always allowed, stiffening never happens while
        # the measured lead is braking and we're closing -- a boost live at the moment
        # a launch turns into a braking event keeps its relaxed cost through the swing
        # into decel (relaxed jerk helps that swing exactly as it helped the launch).
        # Deliberately inside the non-emergency path only: the MIN_TTC stand-down below
        # still returns the solver to full stock stiffness, unchanged
        if scale_target > self.jerk_scale and a_lead < -RATCHET_LEAD_BRAKE and closing > 0.0:
          scale_target = self.jerk_scale

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
        for trig in self._triggers:
          trig.reset()
    else:
      for trig in self._triggers:
        trig.reset()

    self.jerk_scale = self._slew(self.jerk_scale, scale_target, JERK_SCALE_RATE)
    self.t_follow_pad = self._slew(self.t_follow_pad, pad_target, ONSET_RATE_UP if pad_target > self.t_follow_pad else ONSET_RATE_DOWN)

    return self.jerk_scale, t_follow_base + self.t_follow_pad
