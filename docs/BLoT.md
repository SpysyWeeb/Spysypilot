# BLoT — Better Longitudinal Tune (branch renamed from BLT 2026-07-17 to free the acronym; BLaT = Better Lateral Tune)

Goal, in the owner's words: braking events start smooth, peak around the middle of the
event, then release naturally; intensity proportional to what the situation actually
requires; hard braking held no longer than necessary; and the fix must work regardless
of which driving model is running. That last constraint fixes the architecture: BLT
tunes and supervises the **radar + MPC leg** of the planner (which runs identically
under every model, in both chill and experimental mode) and does not touch the e2e leg.
(One exception downstream of the planner: the launch chain — see Related machinery —
extends into longcontrol's starting state and the opendbc CAN limits, equally
model-agnostic.)

Reference targets, fitted from the owner's manual driving (route 00000035, 19 stops):
onset at ~2.9 s headway, single application, build ~0.5 m/s^3, plateau −1.4..−1.8
scaled by need, one release taper ~+0.2 m/s^3, landing −0.33, stop gap ~4.3 m median.

Code: `selfdrive/controls/lib/blt.py` (`BLTSupervisor`), wired in
`longitudinal_planner.py`, driving runtime knobs of
`longitudinal_mpc_lib/long_mpc.py`. Unit tests:
`selfdrive/controls/tests/test_blt.py`.

## Where the felt problems actually live (from reading the whole stack)

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

### 2. Solution stiffness — A_CHANGE_COST pins both directions of recovery

`A_CHANGE_COST = 200` (x jerk_factor, applied only over the first 2 s of the horizon)
limits how fast the *solution itself* may change, symmetrically:

- **Braking held past necessity** (route 39 t=617: ~2.2 s of post-necessity braking) —
  the solver keeps paying rent on stale deceleration after the world has recovered.
- **Acceleration lagging a launching lead** (route 59 seg 3: 3.5 s to swing from
  braking to peak accel while the lead opened a 35 m gap against a ~26 m request) —
  the same stiffness on the way up, measured at ~0.45 m/s^3 plan ramp.

(Historical: a second held-braking mechanism — `radard.aLeadTau` filtering to 0 during
sustained lead braking, making `extrapolate_lead` predict a lead that brakes across the
whole 10 s horizon — was the single biggest "holds the brakes" cause. It no longer
exists: the MPC now consumes the driving model's predicted lead trajectory directly,
see "Related machinery" below.)

The remaining ~0.5 s is hydraulic brake bleed on the Palisade (for *release*, the
pid-state CAN JerkUpperLimit of 3.0 was verified not the bottleneck; the standstill
exit was a different story — see "Launch chain" under Related machinery).

### 3. Proportionality — the personality knobs already exist, statically

`get_jerk_factor` (relaxed 1.0 / standard 1.0 / aggressive 0.5) scales both J_EGO_COST
and A_CHANGE_COST; `get_T_FOLLOW` (1.75 / 1.45 / 1.00) sets the gap term. Both are read
**every frame** (`set_weights(...)` and `params[:,4]`), which means the stock controller
already accepts per-frame modulation of exactly the quantities that determine onset
timing, intensity, and release rate. Nothing needs a solver regen to be scheduled.

## The BLT architecture: a necessity supervisor driving the MPC's own knobs

The failed pattern (retired): wrapping the MPC's inputs (v_cruise caps) or outputs
(release governors) — caps cannot express intensity, and every wrapper fights the
solver's internal state. The BLT pattern: compute, each frame, what physics actually
requires from the **raw measured lead**, in the relative frame (the route-3c lesson:
an absolute headway-margin form exploded at the owner's 1.2-1.6 s following distances):

    a_req = max(-aLead, 0) + max(closing, 0)^2 / (2 * max(dRel - D_MIN, budget))

("match the lead's own braking, plus shed the closing energy before the standstill
margin") — and use it to decide when the solver's own knobs may be relaxed or opened.
The supervisor never commands accelerations; it returns `(jerk_factor_scale, t_follow)`
and the solver keeps shaping everything.

### Current interventions

- **Recovery boost** — while the plan demands meaningfully more braking than `a_req`
  (excess > 0.4, braking > 0.8, sustained 0.4 s), scale the jerk/accel-change weight
  down toward 0.3 so A_CHANGE_COST stops pinning the recovery; restore as the demand
  converges. Two independent arming paths, one knob:
  - *Radar path*: the excess computation above.
  - *Model early-arm*: the model's own predicted lead trajectory (`leadsV3` 0→2 s
    v-slope < −0.5, prob > 0.5, sustained 0.3 s). For hard braking the model forecasts
    the event ~3 s (median) before radar can measure it — replay-validated over 14
    routes. Deliberately narrow: the threshold sits well clear of the mild band the
    dynamic onset owns, and an unconfident model reading is simply silent.
- **Launch boost** — the exact mirror (route 59). Lead measurably pulling away
  (vrel > 0.5, matching the pull-away floor's gate) AND genuinely accelerating
  (aLeadK > 0.5), necessity zero, and the plan's accel lagging the lead's own by > 0.6:
  same knob, same debounce and slew. Radar-driven by design — for launches radar leads
  the model by 1-1.5 s (measured), the reverse of hard braking. Never touches t_follow:
  shrinking the gap is a safety change; jerk relaxation only changes how fast the
  solver tracks the gap it already wants.
- **Whiplash ratchet** — the jerk scale may relax at any time but may only *stiffen*
  while the measured lead is braking (aLeadK < −0.2) and ego is closing. A boost that
  is live when a launch flips into a braking event keeps its relaxed cost through the
  swing into decel (relaxed jerk helps that swing exactly as it helped the launch).
  Lives inside the non-emergency path only — the stand-down below still returns full
  stock stiffness.
- **Dynamic onset** (fixes mechanism 1) — when a tracked lead's measured decel crosses
  0.4 and necessity is still mild, pad the per-frame t_follow parameter (up to +0.45 on
  top of the personality base, **proportional** to the lead's decel) so the
  desired-distance term opens *early* and the quadratic cost pulls *gently*. Cancels
  while recovering (ego at/below lead speed) or while the lead accelerates. A second
  form covers the already-stopped lead (route 3e: nothing else opens the gap term and
  the car carried speed to a 2.2 m squeak-stop): the pad is driven from closing-energy
  necessity when vLead < 2.
- **Emergency stand-down** — everything resets and the solver returns to stock weights
  when TTC < 3.5 **and** a_req ≥ 1.5 together. The conjunction matters: a controlled
  approach to a stopped lead naturally runs TTC < 3.5, and standing down on TTC alone
  dropped the stopped-lead pad mid-approach (route 3e replay) — a desired-gap release
  right where firmness matters most.

All outputs are rate-limited (jerk scale 1.5/s, pad 0.4/s up, 0.5/s down): no steps,
ever. Trigger bookkeeping is shared (`DebouncedTrigger`): accrue while arming
conditions hold, reset on disarm conditions, hold in the hysteresis band between.

### Retired interventions (kept for the record)

- ~~**Pessimism floor**~~ (2026-07-11) — floored `aLeadTau` so the legacy extrapolation
  stopped predicting a lead that brakes forever. Obsolete when the MPC moved to
  model-predicted lead trajectories: no extrapolation left to patch. First crutch
  actually deleted.
- ~~**Gap forgiveness**~~ (2026-07-11, route 46) — tuned on legacy-MPC route-3f data, it
  ratcheted steady-following headway down to its floor and silently absorbed gap-button
  personality changes. Whether the fall-back ping-pong it patched even occurs under
  model-predicted leads is a field question, answered without the mask.
- ~~**Smooth Approach / Release wrappers**~~ (2026-07-10) — v_cruise caps and output
  governors; could not express intensity. Superseded by this architecture entirely.

### Owner doctrine: the boosts are crutches

If the base tune were right, neither boost would ever fire. Boost-activity % per route
is the metric to drive toward zero — when a data-backed base-tune change makes a
trigger silent in the field, delete the trigger. (Recent duty readings: recovery boost
2.7-3.8 % on exp-mode routes; launch boost 1.01 % over the 14-route validation set.)

## Related machinery (lives in long_mpc.py, not BLT — but part of the same tune)

- **NewLeadMpc** (`process_lead_model`, upstream PR #37824 via IQPilot): the solver's
  lead input is the model's predicted `leadsV3` trajectory anchored to radar's h=0
  measurement — the single deepest lead-reaction-time lever in the stack. Always on;
  there is no legacy path anymore.
- **Pull-away floor** (`LEAD_PULLAWAY_*`): radar says the lead is genuinely receding
  and not braking → never predict it slower/closer than a constant-velocity radar
  continuation (phantom launch-braking guard).
- **Forecast trust ledger** (`TRUST_*`, routes 55/5b/5d/61/64 — launch guard v3,
  replaces both the confirm-gated gap credit and the planner's lead-launch
  anticipation machine): the model's forecast is trusted fully by default (stock PR
  behavior, no radar gatekeeping), and radar is the *auditor* — promised-but-unmeasured
  lead acceleration accrues debt that smoothly blends the obstacle toward the radar's
  constant-velocity continuation; radar-corroborated motion (vLead > 0.3 or aLeadK >
  0.4) pays debt down in ~0.3 s. Net: zero lag when the model is right (launches,
  creeps, ping-pong leads corroborate continuously), and a persistently wrong forecast
  can spend only ~0.5 s of optimism before easing off — a taper, not a slam. Replay
  library: route 55 ends up *safer* than the credit (slack more negative throughout
  the false window); route 61's creep-restop pays no confirmation tax on any swing;
  14-route census: trust < 0.99 on 1.6% of lead-frames, longest low-trust episode
  2.7 s, all in genuine launch-forecast ambiguity or stopped-lead approaches (where
  distrust errs conservative). Fresh/unauditable tracks reset to full trust — a new
  situation gets stock behavior and earns distrust only by breaking promises.
- **Launch chain** (the "quicker" leg downstream of the ledger — routes 5d/61/63):
  three layers between "the plan wants to go" and "the wheels move".
  - *Proportional starting command* (`longcontrol.py`): the starting state issues
    `clip(a_target, START_ACCEL_MIN, startAccel)` — the plan drives the launch and
    `startAccel` is a **ceiling**, not the value. a_target already scales with the
    situation (~0.1 for a lead that hasn't moved yet, 0.5–0.9 for a hesitant
    roll-away, 2+ for a committed launch); the stock fixed command blasted full
    startAccel at every standstill exit and the safety guard then caught it
    (route 63: lunge-and-catch on every slow lead). START_ACCEL_MIN 0.6 floors it —
    always enough to roll off brake bleed and reach vEgoStarting, a nudge rather
    than a lunge when the plan doesn't believe in the launch yet.
  - *startAccel 1.0 → 1.5* (opendbc `interface.py`): firmer ceiling off the line;
    pid takes over at vEgoStarting (0.1 m/s), so the exposure window is only the
    first ~0.1 m/s.
  - *Starting-state CAN JerkUpperLimit 1.0 → 5.0* (opendbc `carcontroller.py`): the
    real standstill-exit gate — at the stock 1.0 the car took ~1.4 s to physically
    deliver startAccel against brake bleed (field-measured, routes 5d/61: accCmd
    pinned while vEgo stayed 0). 5.0 is the platform's own factory-launch floor;
    pid keeps 3.0, stopping/off keep the gentle 1.0 for brake feathering.
- **Base tune**: STOP_DISTANCE 6.0→7.0 (owner preference, translates the whole decel
  plan), doubled gas schedule (A_CRUISE_MAX + opendbc ACCEL_MAX 4.0 + panda +
  turn-budget `_A_TOTAL_MAX`), aggressive t_follow 1.25→1.00.

## Verification

- Every change replays the **real production class** (never a transcription) against
  recorded rlogs before shipping: held-braking acceptance set (routes 36/38/39), the
  14-route sweep for regressions and duty-cycle, plus the motivating route for each
  feature (route 59 for the launch boost).
- `selfdrive/controls/tests/test_blt.py` locks in trigger semantics, debounce times,
  hysteresis holds, ratchet freeze/release, stand-down, and slew continuity.
- op-model-grader as the scorer: the owner's manual-stop template is the target
  distribution (onset headway, peak, peak-position, release slope, landing, stop gap).
- Field acceptance: the owner's seat. The replay harness cannot run the acados solver
  offline, so knob modulation is verified logic-level and directionally; the solver's
  felt response is what the road test checks.

## Non-goals

- The e2e leg (`min(e2e, mpc)` in experimental mode) — model behavior, out of scope.
- The stop clamp — `smooth-stops` branch owns the final landing (settle/hold), unchanged.
- Force Stops / green-light features — separate branches.
