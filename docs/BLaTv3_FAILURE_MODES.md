# BLaTv3 — design-for-failure catalog

Scope: the planned rewrite of Spysypilot's Palisade rack-trajectory lateral controller
(origin/BLaTv2 → "BLaTv3"). Lists how the system can fail and the design rule / test that
prevents each failure. v2 folds in a 7-lens red-team pass (30 additions, 15 accepted rule
critiques). This document is
the spec the implementation is tested against.

## 0. Design as agreed so far

Ownership (unchanged rulings): the driving model owns *where/when* (the path); the controller
owns *how the rack gets there*; opendbc/panda own platform limits and safety; torqued/lagd own
the learned static torque map and the lateral delay. No lane-placement opinions in the
controller.

Shape ("upstream-shaped controller"):
- `LatControlRack(LatControl)` in one file `openpilot/selfdrive/controls/lib/latcontrol_rack.py`.
- Selected in controlsd from `CP.lateralTuning.torque.useRackTrajectory`, which opendbc's
  `hyundai/interface.py:_get_params` sets from the LX platform code in `car_fw` (Telluride `ON`,
  mixed, or empty firmware → stock). `lateralTuning` stays `torque` so torqued keeps learning.
  The raised 409/+4/−7 envelope (`BLATV2_HIGH_LIMITS`) is gated by the **same** firmware test.
- Own `lateralControlState` union arm `rackState` (carries `saturated` for lagd, a `fallback`
  flag, status, `t_p`, planned/measured rack state, FF/P/D terms, restriction reason).
- Owns a stock `LatControlTorque` that runs in shadow every frame with `active = CC.latActive`
  while its output is discarded; any invalidation hands the frame to it (never zero torque).
  The shadow's request buffer, jerk filter and the shared saturation timer are warm; its
  integrator starts clean on handover. Since phase 2 step 2: a dropped model frame keeps the last
  plan, a model is stale past 0.5 s (SubMaster's alive window), stock holds for 0.5 s before the
  rack resumes (stale model only — a one-frame content fault hands back on the next good frame),
  and a `latActive` blip of up to five frames holds the planned rack and carries it along with any
  wheel motion meanwhile (R6, FM1.6, FM3.12, FM5.2). A warm integrator in the shadow is still open.
- modeld publishes a short curvature preview on `ModelDataV2.Action` next to
  `desiredCurvature`/`desiredCurvatureTime`, computed with the same function as the scalar
  (phase 2 step 3: `desiredCurvaturePreview[Times]`, 0.25 s apart from the action time to 2 s past it).
- The immediate target passes through a bounded reference filter (R5); a scheduled preview (R2–R4)
  measures how far the plan is consistent and earns the tracker a calmer reference and a longer
  response time, never replacing the near target — phase 2 step 4; the small-reversal governor is gone.

Layers (the input side is now explicit — see L8):
0. Upstream inputs — `clip_curvature` (ISO 3 m/s² / 5 m/s³ clamp with its own rate state),
   `lateralDelay` (only trusted when `status == estimated`), `lateralTorqueParameters`
   (`useParams`), `vehicleParameters` (roll, angle offset, steer ratio).
1. Reference — scalar-pinned target `κ_ref(t) = κ_scalar + κ_plan(t) − κ_plan(t_action)`,
   converted to steering angle + rate with the live VehicleModel; preview time `t_p`
   scheduled between `t_action` (≈0.25 s) and 2 s by *path consistency* (R2).
2. Motion — a virtual rack moved toward the target under a speed-dependent rate/accel/jerk
   envelope (comfort envelope, separate from the physical limits).
3. Tracking — torque = feedforward (planned lateral accel through the torque map + friction
   + roll + self-aligning term) + angle P + rate D, **in angle space** (no v² in the gain).
4. Output — torque trajectory planned inside the platform limiters (R9) and led by the
   measured lateral delay; `saturated` means |torque| at the platform limit.
5. Driver — plan position *and rate* follow the hand; assist torque capped from the first
   frame of real driver torque; re-anchor on release.
6. Fallback — warm stock shadow, switched with hysteresis, logged.
7. Adaptation — **two separately learned surfaces, used together** (owner decision 2026-08-28),
   replacing the static envelope tables: a *hold-torque* surface `H(v, angle, lateral load incl.
   bank, direction)` (prior = torqued's linear fit; anchors = residuals from steady frames) and a
   *rate-gain* surface `G(same axes)` (prior = physics shape: stiff at ~2 mph, easiest ~10 mph,
   stiffening with v², lighter when banked; anchors from clean turn-in/unwind frames). Decision
   each frame: `torque = H + planned_rate/G + feedback`; planner envelope = `(headroom − H) × G`,
   opened only as far as the *less-confident* surface supports at that cell, otherwise the comfort
   envelope. Quasi-static only (no inertia/breakaway); anchors from clean frames; decay; never
   lowers authority below stock; both confidences logged in `rackState`.

Phases: (0) safety fixes on today's branch + back-port combo's direction-guard fix;
(1) behavior-preserving port from **combo's** controller files, replay-identical;
(2) scheduled preview (measure first; retire the reversal governor if it works);
(3) rack-aware output + asymmetric feedforward (turn-in / unwind become two named knobs);
(4) bounded observer.

## 1. Governing rules

- **R1 Near-target authority.** The `t_action` target is the magnitude truth every frame. A
  longer preview may only be used when it *agrees* with the near target; it never replaces
  it. When `lateralManeuverPlan` is the active scalar source, `t_p = t_action` for its whole
  validity window (the model's shape channel describes a different lane).
- **R2 Preview by path consistency, in metres, with uncertainty.** `t_p` may extend beyond
  `t_action` only while all of: (a) the predicted lateral deviation over `[t_action, t_p]`
  between the model path (`modelV2.position.x/y`) and the path the car would follow steering
  to the far target — using a clothoid-consistent extrapolation (linear curvature buildup
  allowed), not a pure arc — stays under `δ_y` (≈0.15 m); (b) near and far target angles agree
  within `δ_θ` (≈1°); (c) the model's own position uncertainty (`position.yStd`) over the
  window stays under a threshold and `modelV2.confidence` is not red. Self-consistency proves
  smoothness, not correctness — (c) is what stops "smooth but wrong". `|κ(t_p)|` alone is
  never a straightness test: at 40 mph, `|κ| < 0.001` (3.9° at the wheel) still allows 0.64 m
  of drift over a 36 m preview. No cross-checks against lane lines/road edges (lane-placement
  ruling). *Measured 2026-08-29 on 49 owner routes (816,450 engaged model frames): on straight
  frames the 2 s target is 1.98× calmer than the 0.25 s target (frame-to-frame std 0.111° vs
  0.220°, holding in every speed band); `δ_y = 0.15 m` is the knee (admits 78 % of straight
  driving at 2 s with the clothoid idealization, 74 % arc; 0.25 m adds only 1.6 points) and
  rejects every one of the 475 island-class frames by 3–16×; the clothoid is uniformly the right
  idealization (arc p99 deviation 2.6 m vs 0.96 m). The far target moves the wheel only 0.48°
  RMS relative to the near one on admitted frames.*
  *Field 2026-08-30 (route 00000028, the first step-4 drive): the first cut served the far target
  whenever it agreed with the near one within 1°, which is exactly where the model's lateral-position
  corrections live at speed (1° ≈ 0.15 m/s² at 25 m/s) — the served target sat 0.77° (p90) / 1.2° (p99)
  from the near one while open, 0.05–0.14 m/s² of correction ignored, 52 drift-then-correct cycles in
  46 min; the owner felt them. Fixed the same day: the preview never replaces the near target; it schedules the
  reference filter's time constant (0.1 → 0.3 s) and the response time instead.*
  *Phase 2 step 4 (2026-08-29): `PreviewScheduler` walks the horizon grid one step at a time; a
  step is admitted only if (a) the model's own path samples inside `[t_action, t_action + step]`
  lie within 0.15 m of the clothoid from the near to the far curvature (0.20 m to keep an
  admitted step), (b) the far wheel angle is within 1° of the near one (1.33° to keep), (c) the
  path's `yStd` stays under 0.35 m (its p99 at 2 s on straight frames) and `confidence` is not red, (d) the far target's swing since
  the previous model frame is no larger than the near target's plus 0.25° (FM1.18), and (e) the
  preview covers at most 40 m at the plan's own speed (FM1.8). Hands on the wheel, a lane change
  starting or finishing, a limited immediate target or a measured curvature out of bounds pin the
  preview at the action time at once.*
- **R3 Shorten fast, lengthen slowly, don't starve.** `t_p` collapses to `t_action` within at
  most two consecutive failing model frames (≤100 ms ≈ 1.8 m at 40 mph); it grows back at a
  bounded rate. Growth progress is a decaying score, not reset by one isolated failing frame,
  so periodic texture (bridge joints, rumble strips) cannot pin `t_p` at `t_action`.
  *Phase 2 step 4: the preview shortens to the largest admissible step after two consecutive
  disagreeing model frames (≤ 100 ms) and lengthens one 0.25 s step per two agreeing frames
  (0 → 2 s in ~0.8 s); an isolated disagreeing frame does not reset the growth count (FM1.16).*
- **R4 The envelope bounds rate/accel/jerk, never amplitude — and opens ahead of need.** Two
  triggers open the comfort envelope: *proactive* — R2 has confirmed a real deviation ahead, so
  the envelope pre-opens to the rate the model path itself requires; *reactive* — measured
  near-target error. How far it opens depends on how the error arrived (growing over several
  frames → smooth opening; a genuinely sudden step → fast opening) with corroboration
  (position path agrees, confidence not red) before exceeding comfort. The ceiling is the
  upstream ISO clamp (`clip_curvature`: 3 m/s², 5 m/s³ — 36° and κ=0.0046 in 0.29 s at 40 mph),
  which stays in place. The comfort table above ~35 mph must be re-derived: the smoothest
  possible island jog (1.5 m over 40 m) needs 25.1 °/s at 40 mph against today's 25.7 °/s cap;
  a brisker one needs 2–4×.
  *Phase 3 step 5 (2026-09-01, horizon-implied envelope opening, G-independent): the proactive
  half ships. A second, confidence-free `PreviewScheduler` (`envelope_scheduler`) reuses every
  R2 gate above except the confidence check, corroborated instead by its own frame-to-frame
  far-point stability check (`PREVIEW_ENVELOPE_DRIFT_M = 0.25 m`, route-20 provenance below).
  `_horizon_opened_profile` derives the required rate/accel from the targets it admits, margins
  them 1.15×, and caps them at `_iso_ceiling` — the same upstream ISO clamp named above,
  evaluated exactly (not the placeholder 2.5×/1.5× multiplier this entry once proposed): 2.34×
  comfort at 40 mph, narrowing to 1.37× by 70 mph, so R9/R10 need no new number. A bounded
  one-pole state keeps the opening continuous (R7): snap down the instant the demand or the
  admitted horizon falls, ease up over one `HORIZON_STEP_S` (0.25 s) otherwise. Feeds
  `_motion_limits` unmodified, in that order, so the feasibility ratchet still narrows on top;
  never widens torque authority (R10) — see `rack_trajectory.py`.*
  *Phase 2 step 4 implements only the response-time side: the tracker's response time is
  scheduled from 0.4 s at the action time to 0.5 s at the full 2 s preview (`RESPONSE_TIME_PREVIEW_S`)
  and the reference filter's time constant from 0.1 s to 0.3 s (`REFERENCE_FILTER_PREVIEW_RC_S`),
  and carried through profile transitions (a branch of `_motion_limits` used to drop it). The
  proactive and corroborated opening of the comfort envelope itself is still open: it belongs to
  the phase-4 envelope `(headroom − H) × G`.*
- **R5 Filtering is bounded in amplitude *and* time.** Any smoothing of the reference may
  hold the served target back from the model's by at most a stated amplitude, and any such
  trailing must decay with a stated time constant; a real change always passes at once, short
  of the raw target by no more than the amplitude bound and at the raw target's own rate.
  *Phase 2 step 4 (2026-08-29): `ReferenceFilter` — first-order, τ = 0.10 s, bound
  `min(0.2 m/s² / v² in wheel angle, 3°)` (the 3° cap binds below ~12 m/s, the lateral
  acceleration above). Chosen by replaying three mechanisms on the logged targets of routes
  00000023/22 at 2–13.5 m/s (`phase2/step4_design/filter_sim/`): a one-model-frame hold of
  small reversals removed 6 % of the p95 step and froze the target for 20° mid-unwind (model
  noise reverses sign inside real moves); the old governor removed nothing (its 1° / 5 °/s gates
  sit below the jitter); the bounded-lag low-pass removed 60 % of the p95 step and 91 % of the
  reversals with the 3° hard bound. The owner's bound: ≤ 3° in the twelve largest low-speed
  turn-ins/unwinds, checked in replay. The reversal governor is retired; its 0.12 s constant
  survives as `DIRECTION_GUARD_RC_S` for the direction guard's recovery ramp.*
- **R6 No frame is ever invalid-to-zero; degrade to warm stock, not to a weaker tier.** The
  only fallback is the stock shadow (it has the integrator a scalar-only rack tier lacks).
  One staleness threshold, owned by the controller, checked against `timestampEof` **and**
  `sm.alive` (`sm.valid` never decays and cannot detect a hung publisher); below it, *hold*
  state, never reset; above it, hand over with hysteresis (≥0.5 s in stock).
- **R7 Continuity.** Every rule is continuous in its inputs; sweep tests across every rule
  boundary are required unit tests. No exact-zero special cases.
- **R8 Fail closed at selection — for the controller *and* the torque authority.** Unknown,
  mixed, or empty firmware → stock controller **and** stock 384/3/7 envelope, from the same test.
- **R9 The controller knows every platform limiter by name:** opendbc slew 409/+4/−7 with its
  sign-aware reversal cost (+409 → −409 takes ~1.6 s); the driver-torque allowance
  (`STEER_DRIVER_ALLOWANCE` = 50 counts) that clips authority before `steeringPressed` trips;
  panda's real-time rate window; the 85°/890 ms/2-frame LKAS angle-fault-avoidance cycle;
  the upstream ISO `clip_curvature`; the measured lateral delay (default when lagd reports
  `unestimated`). Anchor on `carOutput.actuatorsOutput.torque`, sanity-checked against
  `CS.steeringTorqueEps`.
- **R10 Adaptation is slow, bounded, single-purpose, and stays inside known ground.** One
  gain; hard limits; frozen when raw driver torque exceeds the allowance, under saturation,
  below torqued's validated speed floor (15 m/s — below it the band is fixed), during
  fallback frames, and while torqued's filtered params are moving; torqued's collection is
  down-weighted while the observer's gain is moving. Persisted state is versioned by the band
  boundaries and has a staleness bound; a maintenance reset clears both learners.
- **R11 Measured time.** The planner, rate estimator and reference filter use the measured control
  `dt` (clamped), not the compile-time constant; a lag spike must not shrink a planned motion.
- **R12 Evidence matches the layer.** Open-loop route replay grades the reference/planner
  layer only. Anything that depends on the vehicle's response to the candidate's *own* torque
  (tracking, feel, observer fitting) needs closed-loop counterfactual replay (route-audit's
  PlantTwin harness) or a field drive. Phase 1's baseline is combo's controller files.

## 2. Failure-mode catalog

Format: **ID — name.** Scenario. *Mechanism.* → Design rule. → Test. `[v2]` = added by the
red-team pass.

### L0/L8 Upstream inputs (new)

- **FM8.1 — `clip_curvature` precedes the controller. [v2]** A hard avoidance swerve, or a
  hands-off island jog. *controlsd rate-limits `desired_curvature` through the ISO clamp with
  its own persistent state before the controller sees it; `curvature_limited` only feeds the
  alert path.* → Name it in R4/R9; the ISO clamp stays (it is sufficient: 36°, 0.29 s to
  κ=0.0046 at 40 mph); the preview series must be built from the *model* curvature, pinned to
  the clamped scalar, so the clamp shapes magnitude, not the plan's shape. → Unit: step to
  MAX_CURVATURE; assert preview shape unaffected and scalar follows the ISO ramp.
- **FM8.2 — `lateralDelay` trusted while unestimated. [v2]** Fresh install, factory reset,
  lagd restart mid-drive. *controlsd reads `lateralDelay` with no `status` check, unlike
  `lateralTorqueParameters.useParams`; lagd publishes `steerActuatorDelay + 0.2` as the
  initial value.* → Gate on `status == estimated`; documented default otherwise;
  `rackState.delayEstimated`. → Unit with a mocked unestimated message.
- **FM8.3 — Roll estimate lags a banked ramp. [v2]** Cloverleaf/flyover superelevation building
  over 50–100 m at 30–45 mph. *The lateral-accel clamp uses paramsd's roll; if it lags the
  true bank, the clamp starves curvature authority at the tightest point.* → Roll's
  contribution to the clamp tracks at its natural rate with an uncertainty margin while
  paramsd is unconverged; the FM1.9 rate-limit applies to angle offset only. → Lagged-roll
  synthetic (1–2 s time constant): clamp never below the true budget by more than the margin.

### L1 Reference / preview

- **FM1.1 — Far-point blindness to lateral offset.** Pedestrian refuge island on a straight
  40 mph road (S Tamarac Dr): the path jogs ~1.5 m over ~40 m and back (≈18° at the wheel,
  1.5 m/s², 2.2 s per half); a lane shift around a parked car; construction shift; a lane
  change. *At 2 s (36 m) the path is straight again; curvature at the far point ≈ 0 while the
  path between is an S.* → R1 + R2. → Phase 1: fixed 0.25 s target stays authoritative through
  the jog. Phase 2: synthetic S-jog (1.5 m / 40 m / 17.9 m/s): `t_p == t_action` throughout;
  Tamarac route replay; and the recorded island-class event on route `00000020` seg 6 at 374 s
  (`θ_near −134.5°`, `θ_2s +1.6°`, `κ_2s −0.00018`, clothoid deviation 1.77 m) must be rejected at
  every `δ_y`.
- **FM1.2 — Preview chatter.** Curvature hovering at the consistency threshold. → R3. → Path
  oscillating ±10 % around `δ_y`: `t_p` changes ≤ once per second.
- **FM1.3 — Model replan flip-flop.** Faded lines, tar snakes, merges. → R4/R5: the envelope,
  not a filter, bounds what reaches the rack; any small-reversal filter bounded in time. →
  Alternating-path replay.
- **FM1.4 — Scalar/plan anchor mismatch.** Action head vs plan-derived curvature; the
  look-ahead formula vs interpolation (+12–14 % bias found on curve entry). → Preview
  computed in modeld by the same function; `preview[0] == desiredCurvature`. → Bit-exact unit.
  *Phase 2 step 3 (2026-08-29): done.* modeld's `get_action_from_model` evaluates
  `get_curvature_from_plan` at `LAT_PREVIEW_OFFSETS` past the action time and pins the list to the
  published scalar (`test_preview_is_the_scalar_function_along_the_plan` asserts the pin bit-exact,
  also through the action head and the low-speed hold); `model_path_targets` reads the preview and
  no longer touches `orientationRate`. The preview is the scalar's timeline from the action time on;
  lagd's action time already covers the model's age, so a query `offset` past now reads the preview
  `offset` past the action time (`PREVIEW_S` = 0: the immediate target is the scalar as published;
  later samples are interpolated 0.25 s apart). No query precedes the action time and a target's
  position and rate come from one curve — the review of the first cut caught a borrowed-rate variant
  (flat position, first-segment slope as rate) that fed the planner a fictitious lead on every frame
  at lagd's 0.375 s action time, and a second cut that added the plan's age on top of the action time
  double-counted lagd's delay. Past the preview's end the last sample holds. Missing preview →
  status 6, stock steers (R8).
- **FM1.5 — Truncated or invalid plan.** Approaching a stop the plan's velocity reaches ≤ 0
  inside the horizon. *Today: whole frame invalid → zero torque while still rolling.* → R6:
  clip the horizon to the covered range; `t_p → t_action`. → Plan hitting 0 at 1.5 s, vEgo
  3 m/s: torque continuous.
- **FM1.6 — Stale or dropped model frame.** One dropped camera frame; a 150–250 ms pipeline
  stall (3–5 skipped publishes, inside SubMaster's 0.5 s). *Today: 0.2 s hard cutoff, and
  `_invalidate()` resets planner/governor/estimator → zero torque and a re-seed.* → R6: one
  threshold, hold not reset below it; `modelV2.valid` reacts to same-frame drops but never
  decays with time, so freshness must come from `sm.alive`/timestamps. → Gaps of 50/150/250/
  450 ms: no reset, continuous torque; fallback only above the threshold.
- **FM1.7 — Lane change.** 3.5 m shift over 3–5 s; far point straight in the new lane. → Same
  as FM1.1; on `laneChangeStarting` shorten `t_p` — but a *nudgeless* start is provisional
  for its 0.5 s reconsideration window (see FM4.7). → Lane-change replay.
  *Phase 2 step 4: `laneChangeStarting` and `laneChangeFinishing` pin the preview at the action time.*
- **FM1.8 — Speed change during preview.** Braking toward a curve. → Preview in time at the
  planned speed, capped in distance (≈40 m). → Braking-into-curve replay.
  *Phase 2 step 4: the 40 m cap integrates the plan's own speed profile.*
- **FM1.9 — Angle-offset jumps.** paramsd re-converges; `angleOffsetDeg` steps. → Rate-limit
  the *change* of the offset's effect on the reference angle (roll excluded — FM8.3). → Step
  offset 1°: rack motion inside envelope.
- **FM1.10 — Model sees the obstacle late or not at all.** Out of scope for the controller;
  it must execute a late, hard correction at full (ISO-bounded) authority and the driver
  override must be effortless. → Late 30° step at 40 mph: time-to-target ≤ stock.
- **FM1.11 — Unfloored curvature division and single-sample `break`. [v2]** Parking-lot
  creep, drive-through, any stop approach; or one spurious non-positive velocity sample on a
  normal highway plan. *`curvatures.append(rate / speed)` with no floor: tiny planned speeds
  turn yaw-rate noise into curvature spikes; one bad sample truncates everything after it
  while `valid` stays True → coverage error → today, zero torque.* → Floor the denominator at
  `MIN_SPEED` at the division; never let one bad sample truncate coverage of earlier times.
  → Unit: 0.05 m/s sample bounded; injected 0.0 sample mid-array still yields output.
- **FM1.12 — Two model channels, no cross-check. [v2]** A feature revealed abruptly; a lane
  change onto a lane of different curvature under `lateralManeuverPlan`. *Scalar (action head)
  and shape (plan tensor) are decoded separately and may disagree for a frame or two; the
  anchor formula assumes they describe the same path.* → Disagreement = R2 failure → `t_p →
  t_action`, flagged distinctly; under `lateralManeuverPlan`, `t_p = t_action`. → S-jog replay
  with a 1-frame lag injected on one channel: no preview extension in the window.
- **FM1.13 — Smooth but wrong. [v2]** Wide-lane wander (4–6 s period), apex-cutting on a
  decreasing-radius ramp, a confidently wrong branch at an ambiguous exit. *A mean-path
  consistency test passes anything smooth.* → R2(c): uncertainty + confidence gates. →
  Synthetic 0.3–0.5 m sinusoidal wander: `t_p` never reaches max.
- **FM1.14 — Confidence never consulted. [v2]** Night, rain, glare. *`modelV2.confidence`
  rises before curvature error appears; nothing reads it.* → Yellow/red: no extension past
  `t_action`; stronger corroboration before R4 exceeds comfort. Never *reduces* authority.
  → Red-confidence frame: `t_p` pinned, reason logged.
- **FM1.15 — Clothoid entries rejected by an arc test. [v2]** Every standard highway curve
  entry. → R2(a) clothoid-consistent extrapolation; tolerance scales with the expected
  buildup-vs-arc gap. → Six AASHTO-typical transitions pass consistency.
- **FM1.16 — Periodic texture starves `t_p`. [v2]** Bridge joints every 15–20 m, rumble
  strips. → R3 decaying score. → 0.5–1.5 Hz periodic deviation for 30 s: time-averaged `t_p`
  above a target fraction.
  *Phase 2 step 4: an isolated disagreeing model frame neither shortens the preview nor resets its growth count.*
- **FM1.17 — Preview under-samples short features. [v2]** A quick jink or a narrow chicane
  shorter than the T_IDXS spacing in the 1–2 s range. → Consistency evaluated on the native
  model grid, not on a 0.25 s resample. → Synthetic 8 m chicane at 2 s: flagged inconsistent.
  *Phase 2 step 4: the clothoid deviation is evaluated on every native path sample inside each window, not only at the grid points.*
- **FM1.18 — The far target can be *noisier* than the near one. [measured 2026-08-29]** Highway
  speed (16–27 m/s) on two routes: `κ(2 s)` flickers across the ±0.0005 "straight" boundary from
  ordinary replan noise and the curvature→angle conversion's v² understeer term amplifies that
  into 1–2° swings of `θ_far` while `θ_near` barely moves. *A far target is only calmer when its
  curvature is stable, not merely small.* → The far target's curvature is admitted through the
  same smoothness/agreement gates as its position: R2(b)'s `δ_θ` compares `θ_far` frame-to-frame
  as well as to `θ_near`, and a far target whose own frame-to-frame swing exceeds the near
  target's is not used (the preview then shortens per R3). → Synthetic: `κ_far` dithering ±0.0006
  at 20 m/s on a straight path — assert the preview does not lengthen and the commanded angle
  jitter is ≤ the near target's.
  *Phase 3 step 5: `envelope_scheduler` shares this exact gate (unconditionally, not gated by
  confidence), so a dithering far target moves neither its admitted depth nor the
  `required_rate` `_horizon_opened_profile` derives from it, for either sign of the dither.*

### L2 Motion planner (virtual rack)

  *Phase 2 step 4: implemented as R2(d), the far target's frame-to-frame swing gate.*
- **FM2.1 — Virtual rack diverges from the real rack.** Driver holds, curb strike, ice, opendbc
  clipped the request. → Re-anchor when tracking error exceeds a bound **for ≥100–150 ms or
  together with driver torque** (a single-frame magnitude test misfires on speed humps and
  railroad crossings); plan follows the hand while pressed. → Hold 3 s then release: step
  < 0.1; speed-hump jolt: no re-anchor.
- **FM2.2 — Envelope tightens mid-motion.** Speed rises mid-turn. → One explicit decay-inside
  rule; continuous. → 5→20 m/s ramp holding ±10°: no discontinuity.
- **FM2.3 — Comfort envelope blocks an evasive motion.** → R4. → 30° step: time-to-target ≤
  stock.
- **FM2.4 — Numerical edge cases.** → Planner clamps its own state; no exceptions in the
  loop. → 100k random steps inside the envelope.
- **FM2.5 — Standstill and creep.** κ→angle scales as 1/v². → `MIN_SPEED` floor at every
  division (reference *and* feedback — see FM3.10); hold-angle behavior 0.3–1 m/s. → Creep at
  0.5 m/s: bounded target.
- **FM2.6 — Envelope tables mis-scaled.** p99 of how the rack *was* moved, not what the EPS
  can do; at 40 mph the cap equals the smoothest necessary jog. *Provenance re-derived
  2026-08-28 from 31 of the owner's routes (independently reproduced): the shipped rate table
  is p99 of the smoothed steering-angle derivative over ALL frames, driver and openpilot
  together (0.91–1.27× per bin; the 5–10 mph value is a human hand-over-hand signature); the
  acceleration table matches no population (real p99 is ~3× it at 35–55 mph); real driving
  exceeds the rate cap in < 1.1 % of frames, openpilot's own steering in < 0.9 %.* → The
  tables are retired in favour of the two learned surfaces (layer 7); until then they are the
  seed of the rate-gain prior only, cited to the derivation script. → Island jog inside the
  comfort envelope with margin.
  *Phase 3 step 5: the tables stay the comfort floor unchanged; R4's opened ceiling is the ISO
  clamp (owner Q1), not a re-derivation of these tables above 35 mph — a cross-check against a
  corpus p99.9 rerun remains open (dissent), but is no longer load-bearing for R4 to ship.
  Reconcile note (2026-09-01): swept `_iso_ceiling` against this table 5–85 mph — the RATE leg *Resolved the same day: the acceleration and jerk ceilings now scale with the rate opening ratio (never below the comfort table), so the three limits open as a family and the plan can accelerate into an opened rate; the rate ceiling itself is unchanged (ISO-derived).*
  opens as designed (owner Q1's 2.34×/1.37× figures reproduce exactly), but the ISO formula's
  JERK leg sits below comfort jerk at every speed from ~10 mph up, and its ACCELERATION leg
  drops below comfort accel above ~50–55 mph (e.g. 70 mph: comfort 97.74 vs ceiling 72.26
  °/s²). `ease()`'s own floor (never below comfort) makes this safe, not a bug, but it means
  the accel/jerk legs of the envelope structurally never open in the field at highway speed —
  only the rate leg delivers R4's intended benefit there. Flagging for owner sign-off; no code
  changed, since owner Q1 accepted the ISO ceiling formula as-is.*
- **FM2.7 — Ordinary tight turns trip the "evasive" trigger. [v2]** A signed right turn off a
  35 mph arterial: cross-street curvature appears in the last 0.5–1 s; ~370° of wheel. → R4's
  opening depends on how the error arrived (growing over frames → smooth opening). → Curb-
  return synthetic at 4.5 m/s: deviation < 0.3–0.5 m, no full-limit step.
- **FM2.8 — Reactive-only opening is late for a necessary jog. [v2]** The island at 40 mph.
  → R4 proactive trigger from R2's confirmed deviation. → Tamarac replay: rack rate tracks the
  path's required rate from the first consistent frame.
- **FM2.9 — SOL long stop freezes the scalar. [v2]** Under SOL, 30–90 s at a red light with
  the wheel off-center. *modeld holds `desiredCurvature` below 0.3 m/s; on resume the first
  live value can differ sharply and R4 would treat it as a hazard.* → Track time-since-frozen;
  after a freeze longer than one horizon, the first live value is a fresh reference with the
  comfort ramp. → 45 s at 0 then creep: comfort ramp, not full authority.
- **FM2.10 — Model swap or noise ramp read as a hazard. [v2]** Big↔small model failover
  mid-curve (`modelV2.big` never read today), action-head jitter with `LAT_SMOOTH_SECONDS = 0`.
  → R4 corroboration before exceeding comfort; a single frame may open partway. → Flip
  `modelV2.big` mid-curve: comfort ramp until the value persists.

### L3 Tracking and output

- **FM3.1 — Feedforward wrong.** Cold rack, tires, torqued defaults after reboot, trailer. →
  P feedback + bounded observer; correct `saturated`. → FF ±30 %: bounded error, no alert.
- **FM3.2 — Request clipped by opendbc unknowingly.** → R9; anchor on `carOutput`, check
  `steeringTorqueEps`. → Step demand: no overshoot after the ramp.
- **FM3.3 — Lateral-delay mismatch.** → Preview floor ≥ measured delay; FM8.2 gating. →
  ±50 ms: bounded, no oscillation.
- **FM3.4 — Sign or convention error.** → Mirror property tests.
- **FM3.5 — Discontinuity at a rule boundary.** → R7 sweeps, jump < 0.05.
  *Phase 3 step 1 (2026-08-30): the turn-in feedback cap blends continuously between 0.35 and 0.7
  over ±3° (`turn_in_fraction`, three ramps replacing three boolean tests) and the direction guard
  ramps down at its own time constant instead of snapping to zero (`test_turn_in_feedback_cap_is_continuous`,
  `test_direction_guard_ramps_down_not_snaps`). A flickering guard condition holds the scale mid-way
  instead of draining — accepted: the condition only flickers while the pre-guard torque crosses zero,
  where there is nothing worth suppressing.*
  *Phase 3 step 4 (2026-09-01, direction guard v2 — target-referred bounded fallback): the guard no
  longer drains toward zero — `direction_guard_scale` is repurposed from a survival scale into a mix
  weight blending torque toward a capped, target-referred fallback, and the two boolean trip
  conditions become continuous conflict fractions (fuzzy-OR via `max()`). R7 is now enforced
  algebraically on the output itself against the previous frame's output
  (`GUARD_FALLBACK_TORQUE_CAP` = 0.18, re-derive after the next drive), closing the scale-step ×
  torque gap this same phase's replay measured. `test_direction_guard_ramps_down_not_snaps` is
  replaced by `test_direction_guard_output_step_is_bounded_across_torque_magnitudes` plus the
  `test_guard_*` suite (conflict continuity through sign singularities, bit-identity outside
  conflict, target-following, no rate term, R10's cap invariant, R10's never-widen-authority bound).*
  *Reconcile pass (2026-09-01): the first cut applied R7's output clamp unconditionally, every
  frame, regardless of whether this guard was actually blending -- breaking the mandatory
  bit-identical-outside-conflict replay gate on an ordinary large torque swing with no direction
  conflict, and mislabeling `direction_guarded` on frames the guard never touched. Fixed by gating
  the clamp on `mix > 0` (R7 bounds this rule's own transition, not every torque change the
  controller makes -- see `test_guard_r7_inert_outside_conflict_even_across_a_large_torque_jump`).
  Separately, `previous_output_torque` was latched from the guard's pre-driver-assist-clip value;
  a saturated driver hand-off could then leave a phantom-high R7 baseline that forced an unwanted
  torque hold the instant the driver released the wheel. Fixed by latching it from the actual
  post-clip committed torque in `update()` instead. The never-widens-authority bound (R10) is
  confirmed NOT unconditional once R7 is active against a stale baseline -- see
  `test_guard_authority_decays_at_the_r7_rate_with_a_stale_baseline`, an accepted trade-off (R7
  continuity over instant suppression), now pinned by a test instead of an unqualified "always"
  claim.*
- **FM3.6 — Saturation semantics.** *Three consumers read one flag: the driver alert (via
  `curvature_limited` too), lagd's data-quality gate, R4.* → Separate signals:
  `saturated` = platform limit; `feedbackLimited` distinct; `curvature_limited` handled
  separately in the alert path. → Feedback clipped 1 s at 0.6 torque: no alert; lagd gate
  unaffected.
  *Phase 3 step 1: `saturated` now means the ±1.0 platform ceiling alone; the direction guard and
  the driver-assist cap report as their own log fields (`directionGuarded` @32, `driverAssistLimited`
  @33); `torqueLimited` stays their union for compatibility.*
- **FM3.7 — Aligning feedforward on banked roads.** → Roll-compensated lateral accel. → Roll
  sweep ±5°: monotonic.
- **FM3.8 — Rate signal quality.** Unsigned 4 °/s `SAS_Speed`; wrong sign 1–4 frames after a
  reversal. → Rate from the angle derivative, magnitude as validity; invalid until two
  consistent ticks. → No wrong-sign valid samples.
- **FM3.9 — Zero-crossing special cases.** → None; continuous through zero.
  *Phase 3 step 4 (2026-09-01): the last exact-zero special case (`measured_angle == 0.0` forcing
  `raw_torque = 0.0`, just above the direction guard) is deleted — direction guard v2's continuous
  conflict fractions cover the boundary without a discontinuous branch.
  `test_unwind_feedforward_releases_hold_torque_continuously`'s old zero-crossing fixture now
  asserts a nonzero, continuous output (swept on both sides of `measured_angle == 0.0`) instead of
  the old exact-zero spike.*
- **FM3.10 — Feedback authority is exactly zero at standstill. [v2]** SOL engaged while
  stopped with the wheel off-center; first creeping frames. *`lateral_accel_per_degree =
  curvature_per_degree × vEgo²` with raw vEgo → gain 0 at v = 0.* → Feedback in angle space
  (no v² in the gain), or floor the same speed variable. → vEgo = 0, 10° error: converges.
  *Phase 3 step 1: the per-degree feedback gain uses the floored speed (`bound_speed`), so creep and
  standstill keep corrective authority (`test_feedback_keeps_authority_at_standstill`).*
- **FM3.11 — Reversal cost invisible to the angle-space envelope. [v2]** Second half of the
  island jog right after firm torque the other way. *+4 up / −7 down per frame, sign-aware:
  a full reversal takes ~1.6 s.* → R9 reversal-cost model; start the return leg early enough
  to fit. → Unit reproducing the slew timing; saturated turn-in then reversal replay.
  *Phase 3 step 2: implemented as the slew-aware early release — walk the horizon for the earliest
  sign flip (angle sign, or a rate reversal above 1 °/s), budget `T_rev = |applied|/RATE_DOWN +
  |opposite|/RATE_UP` from the carOutput applied torque, fastest-release ceiling blended in over one
  budget. Closed-loop replay against the phase-3 plant caught it firing during turn-ins (route 2b
  s-turn: the 2 s horizon holds the next leg of an S while the wheel is still 200° short of this
  one, and the release starved the turn) . Review (wf_97492c7f) then confirmed
  three more: the flip time snapped to the 0.25 s horizon grid, swinging the blend by up to 0.53
  torque in one frame (R7) — the crossing is now interpolated inside its grid segment; the sign test
  was blind when the immediate target sat exactly at zero — it now falls back to the applied torque's
  own side; and `carOutput` was trusted without an alive/valid check — a dead `card` now reads as
  zero applied torque, which disables the release rather than latching it. The step-1 field drive
  (route 2c) settled the entry condition: a direction-fraction gate was blind to an S-leg REBUILT
  through center (wheel on the far side, ask cap-limited, next reversal visible — released, it cost
  0.4 s of crossing on the owner-flagged 63–74 s unwind). Final form: entry blocked while
  `raw · (served target − measured) > 0` (the ask still doing the plan's own work — subsumes turn-in
  and rebuild), then a one-bit latch rides the release through its own shed until the flip clears.
  All four replay windows (2b s-turn, route-23 owner unwind, 2c s-turn, 2c unwind) equal or better
  vs step 1; closed-loop release duty 0–0.8 %. **RETIRED after the first field drive (route 2d,
  2026-08-30, six owner bookmarks):** the rate-flip test reads a visible curve *exit* as a coming
  reversal — during ordinary sustained curves a "flip" is present for seconds, entry opened at benign
  wheel-past-target tracking dither, the latch had no exit when the target climbed back beyond the
  wheel, and holding torque was shed mid-curve (output 0.44 → 0.05 in 1 s while the target rose; the
  car ran wide). Lesson for the redesign: the flip detector must witness a true *torque sign
  reversal* (an angle crossing with commitment), never a rate reversal, and any latch needs the
  entry condition re-checked as an exit. The plumbing (carOutput applied torque, earlyRelease @34)
  remains. The anticipated-reversal `max_rate` reduction is DROPPED from phase 3 (owner decision 2026-08-30 after the
  65-agent panel): it may not reuse the retired t_flip/t_budget scaffolding; if it returns in phase 4 it needs a true
  torque-sign-commitment detector with an entry condition re-checked as an exit, validated on ordinary sustained
  curves at 15–35 m/s before any drive.*
- **FM3.12 — Undebounced EPS fault bit resets the controller. [v2]** A 1–3 frame
  `CF_Mdps_ToiUnavail`/`ToiFlt` flicker during a firm turn. *`steerFaultTemporary` has no
  debounce; latActive drops; today the rack state is wiped and re-seeded.* → Below a short
  debounce, hold state (R6); reserve reset for persistent faults. → 1/2/3-frame fault
  injection: state unchanged after.
- **FM3.13 — The 85°/890 ms/2-frame angle-fault-avoidance cycle. [v2]** Driveway, tight
  neighborhood corner, garage turn with lateral active. *Past ~0.9 s above 85° at the wheel
  (~4.7° road wheel) opendbc drops `CF_Lkas_ActToi` for 2 frames, repeatedly.* → Named in R9;
  a sustained high-angle hold is a budgeted state; empirically resolve whether the cycle
  round-trips into `steerFaultTemporary`. → Bench hold ≥85° for 3 s logging LKAS/MDPS bits.

### L4 Driver interaction

- **FM4.1 — Override then release.** → Re-anchor on release; torque ramps at the slew. →
  Step < 0.1.
- **FM4.2 — Resting hand.** → R10 freeze in raw counts tied to `STEER_DRIVER_ALLOWANCE`
  (the platform clips authority ~100 counts before `steeringPressed` trips). → Bias injection:
  gain unchanged.
- **FM4.3 — Driver fights a real jog.** → Assist cap, no windup, plan follows the hand.
- **FM4.4 — Engage mid-curve; hybrid states.** Engage at 20° with 10 °/s; and the common
  fork case: `longActive` drops on brake/gas while `latActive` stays (SOL/AOL). → Seed from
  the measured rack and rate; the hybrid state is a first-class tested state. → First-frame
  continuity; hybrid-state replay.
- **FM4.5 — The first 50 ms of a grab. [v2]** *`steeringPressed` is 5-frame debounced; the
  plan follows the hand and the assist cap apply only after it trips.* → "Back off now" keyed
  to raw torque/angle (≤1-frame); `steeringPressed` kept for UI/torqued. → Ramp torque past
  threshold with 150–250 °/s angle for 5 frames: assist cap from frame 1.
- **FM4.6 — Release during motion. [v2]** Driver lets go while still rotating back toward
  center mid-jog. *Only the plan's position follows the hand; its rate stays stale.* → Sync
  `planner.rate` to the measured rate every pressed frame. → 40 °/s ramp then release: plan
  rate within tolerance at the release frame.
- **FM4.7 — Nudgeless self-abort window. [v2]** Blinker tap fires the change; the model
  reverts within its 0.5 s reconsideration window. *Shortening `t_p` instantly commits to the
  new lane during the very window the model may abort.* → Nudgeless starts provisional:
  cap excursion rate until past the window; torque-confirmed starts commit immediately. →
  Abort at 0.5 s: excursion below bound.
- **FM4.8 — Brake mid-nudgeless-change under SOL/AOL. [v2]** Brake stab to abort. *Brake
  drops `enabled` but AOL keeps `latActive`; the auto change continues.* → **Owner decision
  2026-08-28: no** — a brake press does not abort an auto-started change; under SOL/AOL the
  brake is a longitudinal input only and the driver aborts with the wheel. Mitigations are
  therefore FM4.7's provisional window, the assist cap, and the wheel nudge. → Brake
  mid-change: lane change continues, longitudinal disengages, no steering discontinuity;
  wheel nudge mid-change: reverts within one model frame.
- **FM4.9 — Fixed assist cap ignores agreement. [v3]** *The flat `MAX_DRIVER_ASSIST_TORQUE`=0.5
  cap pinned even when the driver pushed with the controller's own live intent, discarding
  authority the platform's own driver-allowance limiter would already grant.* → `_driver_assist_envelope`
  mirrors `apply_driver_steer_torque_limits`'s driver-allowance term for the commanded direction
  (ceiling 1.0 pre-existing, R10; 0.5 is now the floor, never the cap); R7-backstopped on the
  output every pressed frame, same idiom as the direction guard but unscoped (every pressed-frame
  step ≤ 0.05, agreeing or not). → Envelope==1.0 exact when agreeing; opposing -300 still pins at
  0.5; envelope matches the real limiter across a driver-torque sweep; R7 holds across the ramp's
  102-count worst-case swing; unpressed frames stay bit-identical.
- **FM4.10 — Fresh-grab-to-oppose inherits an unpressed baseline. [v3]** A driver who grips to
  *oppose* right after an unpressed frame (no cap active) makes R7's baseline the prior unpressed
  torque, not the fixed cap the old build snapped to instantly, so a full-authority-to-floor
  transition costs up to 10 frames (0.1 s) instead of one. → Owner-accepted (Q3): the bound is
  exactly the ceiling-to-floor distance at the R7 step, and the platform's own driver-allowance
  limiter still yields immediately downstream (R10) regardless of this branch. → Route 2d's
  observed worst case (idx≈1780, unpressed -0.79 into a hard opposing grab) resolves in 6 of the
  10 allowed frames; `rack_log.driverAssistCap` now reports `DRIVER_ASSIST_CEILING` rather than the
  capnp `Float32` default on fallback (stock-steered) frames, so this field never reads as "capped
  to zero" when the cap simply wasn't evaluated that frame.

### L5 Runtime, fallback, integration

- **FM5.1 — Exception escapes controlsd.** → No exceptions as control flow; last-resort guard
  returns the stock shadow's torque and logs. → Fault injection.
- **FM5.2 — Fallback thrash / cold fallback.** → Shadow updated every frame with `active =
  CC.latActive` (integrator warm), output discarded while the rack drives, frozen while the
  rack is saturated; switch with hysteresis; logged. → Curvy replay with a forced
  invalidation mid-jog: first stock frame within bound; ≤1 switch/s.
- **FM5.3 — Runtime budget.** → p99 < 2 ms, no per-frame allocation. → Device CI timing.
- **FM5.4 — Firmware gate wrong.** → R8, incl. torque authority. → Real fingerprints:
  Telluride-only, mixed, empty → stock controller **and** 384/3/7.
- **FM5.5 — opendbc/panda pin drift.** *A clean upstream sync already dropped the fork's 409
  carve-out once (`df07a2de` restored it).* → Fork-owned golden test resolving
  `CarControllerParams` for the real Palisade fingerprint asserting (409, 4, 7), run on every
  sync workflow. → Sync dry-run.
- **FM5.6 — Upstream sync collision.** → Union-arm ordinal only; sync workflow raises an issue.
- **FM5.7 — Device/replay divergence.** → One code path. → Replay A/A bit-exact.
- **FM5.8 — Interplay with other fork features.** SOL, nudgeless, curve-speed limiter,
  hot-swap. → Combo replays; FM2.9/4.7/4.8 are the concrete cases.
- **FM5.9 — Stale-model timing hole. [v2]** modeld hangs without exiting (no restart, no
  terminal invalid message); a 20–30 ms controlsd stall. *`sm.valid` holds forever;
  `dt` assumed 10 ms.* → R6 freshness by alive+timestamp; R11 measured `dt`. → Freeze
  modelV2 for 5 s with `valid=True`: fallback well before selfdrived's 3 s soft-disable.

### L6 Adaptation and data

- **FM6.1 — Observer learns a disturbance.** → R10 bounds; offset stays torqued's job. →
  2 m/s² bias 10 s: gain moves < 5 %.
- **FM6.2 — Two learners fight (both directions).** → Disjoint variables; each frozen/
  down-weighted while the other moves. → Co-simulation: no limit cycle.
- **FM6.3 — Overfitting one route.** → Promotion bar: agreement within tolerance across ≥2
  geometrically distinct routes (one jog-heavy, one highway) — the single-route analysis rule
  stays for *feel* attribution, not for promotion.
- **FM6.4 — Bad learned state survives. [v2 expanded]** Alignment, tires, a band-boundary
  change. *torqued's restore key ignores physical events; its decay only slows.* → R10
  staleness bound + maintenance reset for both learners; state versioned by band boundaries.
- **FM6.6 — The two surfaces disagree at a cell.** One surface anchored, the other prior-only
  (e.g. many steady-curve frames but no turn-ins on a banked ramp); or anchors of opposite
  sign from a contaminated frame. *A combined decision mixes measured and guessed values.* →
  Envelope opening keyed to the less-confident surface; a residual anchor is admitted only
  if it agrees in sign with the physics prior or is corroborated by a neighbouring anchored
  cell; disagreement logged. → Synthetic: anchor `H` only on a cell, leave `G` prior; assert the
  envelope stays at comfort and `rackState` reports the limiting surface.
- **FM6.5 — Low-speed band outside torqued's domain. [v2]** Garage ramps, driveways at
  2–15 mph. *torqued never fits below 15 m/s; the observer would sit on an unfitted map.* →
  Bands ⊆ torqued's domain; below it fixed. → Low-speed battery: band gain does not move.

### L7 Sensing and timing

- **FM7.1 — Model latency varies.** → Times relative to `timestampEof`. → +100 ms: no
  invalidation.
- **FM7.2 — Clock domains.** → Verified same domain; replay assertion.
- **FM7.3 — CAN dropout.** → `CS.canValid`; rate estimator invalid. → Freeze angle 5 frames:
  no spike.

### L9 Process and evidence (new)

- **FM9.1 — Porting from the wrong baseline. [v2]** *combo carries `ecec8d387e` (direction-
  guard ramp) that BLaTv2 lacks.* → R12: phase-1 baseline = combo's controller files; CI:
  `git log <baseline>..origin/combo -- <controller files>` must be empty or accounted for.
- **FM9.2 — Open-loop replay grading a closed-loop change. [v2]** Phase 2/3 "validated by
  replay". *Logged steering response is the outcome of the *original* torque.* → R12:
  closed-loop counterfactual (PlantTwin) or field for tracking/feel. → Process gate on PRs.

## 3. Test strategy

1. Property/mirror/sweep unit tests for L2–L3 (continuity, symmetry, envelope), upstream style
   (`OpenpilotTestCase`, parameterized), one file, short tests.
2. Synthetic-scenario tests with hand-built model paths: island S-jog, lane change (confirmed
   and nudgeless-with-abort), stop approach, dropped/stalled frames, evasive step, tight turn
   revealed late, speed ramp mid-turn, SOL long stop, driver grab/release-in-motion.
3. Route replays on the route-audit harness: A/A bit-exact for the phase-1 port; open-loop
   replay for reference-layer changes only; closed-loop counterfactual for tracking changes.
4. Field checklist per phase, one behavior per drive, evidence = logged `rackState` (status,
   `t_p`, restriction reason, fallback, saturated/feedbackLimited, delayEstimated).
5. Fork-owned golden tests that survive upstream syncs: (409, 4, 7) for the Palisade
   fingerprint; firmware gate on real fingerprints; capnp union arm present.

## 4. Owner decisions

1. FM4.8 — brake press aborts an *auto-started* lane change under SOL/AOL? **No (2026-08-28).**
2. FM6.3 — two-route promotion bar alongside the single-route feel rule? **Yes (2026-08-28).**
3. FM2.6 — source of the comfort-envelope corpus and re-derivation above 35 mph: **closed
   (2026-08-30)** — the tables were re-derived from the owner's 31 frozen routes (see FM2.6's own
   entry above; under 1.1 % violation), and no further static tuning is planned because the
   tables are slated for retirement once the learned rate-gain surface exists (phase 4).
