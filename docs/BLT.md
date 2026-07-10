# BLT — Better Longitudinal Tune

Goal, in the owner's words: braking events start smooth, peak around the middle of the
event, then release naturally; intensity proportional to what the situation actually
requires; hard braking held no longer than necessary; and the fix must work regardless
of which driving model is running. That last constraint fixes the architecture: BLT
tunes and supervises the **radar + MPC leg** of the planner (which runs identically
under every model, in both chill and experimental mode) and does not touch the e2e leg.

Reference targets, fitted from the owner's manual driving (route 00000035, 19 stops):
onset at ~2.9 s headway, single application, build ~0.5 m/s^3, plateau −1.4..−1.8
scaled by need, one release taper ~+0.2 m/s^3, landing −0.33, stop gap ~4.3 m median.

## Where the felt problems actually live (from reading the whole stack)

Three mechanisms in stock code produce everything field-tested so far:

### 1. Late-then-hard onset — the obstacle formulation

`long_mpc.py` never tracks the lead directly. It converts the lead into a virtual
*stopped* obstacle at `x_lead + v_lead^2 / (2*COMFORT_BRAKE)` and holds
`desired = v_ego^2 / (2*COMFORT_BRAKE) + t_follow*v_ego + STOP_DISTANCE` behind it,
with a quadratic cost on the distance error (X_EGO_OBSTACLE_COST = 3, normalized by
1/(v+10)) and a slack "danger zone" constraint at 0.75 * desired (DANGER_ZONE_COST = 100).

Consequence: while a lead decelerates *mildly*, the obstacle recedes only slowly
(the v_lead^2 term shrinks quadratically — barely at first), so the cost stays flat and
the MPC coasts. When the lead brakes *hard*, the same quadratic collapses fast and the
cost explodes — hence gentle situations get nothing and developing situations get a
wall of braking all at once. The onset shape is baked into COMFORT_BRAKE (2.5) and
t_follow, both of which are **runtime parameters**.

### 2. Braking held past necessity — lead-accel extrapolation plus recovery cost

Two stacked causes, measured at ~2.2 s of post-necessity braking in route 39 t=617:

- `radard.py`: `aLeadTau` starts at 1.5 but, while `|aLeadK| >= 0.5`, filters toward
  **0.0** with RC 0.45 s. `long_mpc.extrapolate_lead` decays predicted lead accel as
  `exp(-tau * T^2 / 2)` — so after ~1 s of sustained lead braking, tau ~ 0 and the MPC
  plans against a lead that **keeps braking across the entire 10 s horizon**. When the
  real lead recovers, `aLeadK` (Kalman) lags ~0.5-1 s before tau snaps back. This is the
  single biggest "holds the brakes" mechanism.
- `A_CHANGE_COST = 200` (x jerk_factor, applied only over the first 2 s of the horizon)
  limits how fast the *solution itself* may relax — observed recovery ~1.0 m/s^3 from
  deep braking, i.e. 2.5+ s from a slam back to neutral even after the obstacle clears.

The remaining ~0.5 s is hydraulic brake bleed on the Palisade (CAN JerkUpperLimit is
already 3.0 — verified not the bottleneck).

### 3. Proportionality — the personality knobs already exist, statically

`get_jerk_factor` (relaxed 1.0 / standard 1.0 / aggressive 0.5) scales both J_EGO_COST
and A_CHANGE_COST; `get_T_FOLLOW` (1.75 / 1.45 / 1.25) sets the gap term. Both are read
**every frame** (`set_weights(...)` and `params[:,4]`), which means the stock controller
already accepts per-frame modulation of exactly the quantities that determine onset
timing, intensity, and release rate. Nothing needs a solver regen to be scheduled.

## The BLT architecture: a necessity supervisor driving the MPC's own knobs

The failed pattern (retired): wrapping the MPC's inputs (v_cruise caps) or outputs
(release governors) — caps cannot express intensity, and every wrapper fights the
solver's internal state. The BLT pattern: compute, each frame, what physics actually
requires from the **raw measured lead** (no pessimistic extrapolation):

    a_req = required decel to land at desired gap given dRel, vRel, measured aLead trend

and answer the owner's two watchdog questions with it:

1. **"Is this braking actually necessary?"** — compare the plan's current demand to
   a_req. Excess = demand beyond `a_req + margin`, sustained beyond a debounce.
2. **"How much longer must it hold?"** — not a timer: the moment excess exists, start
   removing the *reasons* the solver holds, and let it converge itself.

Interventions, in priority order (all runtime-parameter modulation, no output clamping):

- **Recovery boost** (fixes mechanism 2b): while excess braking is detected and TTC is
  healthy, scale jerk_factor down (toward ~0.3) in `set_weights` so A_CHANGE_COST stops
  pinning the recovery; restore as demand converges to a_req. The solver still shapes
  everything — it just stops paying rent on stale deceleration.
- **Pessimism floor** (fixes mechanism 2a): floor the effective `a_lead_tau` in
  `process_lead` (e.g. tau >= 0.4) whenever a_req is mild and the measured lead accel is
  *rising* (recovering) — the "brakes forever" prediction only survives while the lead
  is actually still deep in its own braking. Small, surgical long_mpc change.
- **Dynamic onset** (fixes mechanism 1, replaces the retired Smooth Approach): when a
  tracked lead's measured decel first crosses a small threshold, ramp the per-frame
  t_follow parameter up briefly (standard -> ~1.7 over ~1 s) so the desired-distance
  term opens *early* and the quadratic cost starts pulling *gently* — early smooth
  commitment through the solver, intensity inherently proportional, no external cap to
  fight. Ramp back as the situation resolves or once ego is established on its own
  braking profile.
- **Static base-tune** (cheapest, do first): candidate COMFORT_BRAKE 2.5 -> ~2.2
  (earlier, gentler engagement of the distance cost at speed; stop distance unaffected —
  the stopped-equivalence term is zero for a stopped lead) and personality-grade
  jerk_factor/T_FOLLOW selection, scored offline before any road test.

Safety posture: the supervisor can only modulate within ranges the stock controller
already exposes (personalities span jerk_factor 0.5-1.0 and t_follow 1.25-1.75 today;
BLT extends those ranges modestly and schedules them). The solver's hard constraints
(ACCEL_MIN/MAX, danger-zone slack, crash distance) are untouched, and every intervention
de-activates on low TTC. The watchdog can never command braking below what the solver
itself decides under un-modulated weights in a genuine emergency.

## Verification

- Replay library (already built, real classes + logged leads): route 36 t=775/1858,
  route 38 t=1066, route 39 t=617 as the held-braking acceptance set; routes 35-39
  full-route sweeps for regressions (false releases, onset regressions).
- op-model-grader as the scorer: the owner's manual-stop template is the target
  distribution (onset headway, peak, peak-position, release slope, landing, stop gap).
- Field acceptance: the owner's seat.

## Non-goals

- The e2e leg (`min(e2e, mpc)` in experimental mode) — model behavior, out of scope.
- The stop clamp — `smooth-stops` branch owns the final landing (settle/hold), unchanged.
- Force Stops / green-light features — separate branches.
