# BLoTv3 — design

**Status: in progress. Not field validated. Do not mark complete before owner field testing and
explicit approval.** Phases 0–3 implemented on `BLoTv3` and replay-gated against BLoTv2; merged into `combo` on
2026-08-29 (phase 4a, replacing BLoTv2) for the owner's field test. Phases 4b–4d (smooth-stops
re-authoring, curve-limiter composition review, SOL/AOL hook review) follow the field verdict.

BLoTv3 restructures [BLoTv2](https://github.com/SpysyWeeb/Spysypilot/tree/BLoTv2) from `stock`.
It keeps BLoTv2's tuned behavior where BLoTv2 was right and fixes the verified defects; it does
not add tuning knobs or toggles. Standalone BLoTv3 runs on the stock opendbc pointer
(`ACCEL_MAX = 2.0`); `combo` runs the same code on the fork opendbc/panda lineage (4.0 m/s²).

## 1. Sole-owner contract

| Decision | Owner |
|---|---|
| Radar/model lead selection | `radard` and the driving model (stock, not edited) |
| Usable lead, lead presence, model-lead anchoring | `longitudinal_lead.py` (`LeadObservation`, `lead_present`, `relevant_lead`, `anchor_model_lead`) |
| Model stop intent classification (stateless) | `stop_helpers.py` — one copy of every stop constant |
| Effective Chill/Experimental mode | `selfdrived` via `conditional_experimental_mode.py`; owns nothing about stop points |
| Stop shaping, commitment, hold through standstill, release | `force_stops.py` (plannerd) |
| MPC response cost and dynamic headway | `necessity_supervisor.py` (`NecessitySupervisor`) |
| Lead trajectory and obstacle optimization | stock Acados MPC; `long_mpc.py` sets weights once per frame |
| Cruise acceleration target and final arbitration | `longitudinal_planner.py` |
| Stop profile and standstill handoff | stock `longcontrol.py`; `smooth-stops` in `combo` |
| Vehicle command limits / safety ceiling | opendbc / panda |

Only `selfdriveState.experimentalMode` crosses processes. Stock never had BLoTv2's
`SelfdriveState.conditionalStop{Qualified,Distance,ModelMonoTime,Latched}` fields, so this branch adds
nothing to `SelfdriveState`; on `combo` those fields move into a `deprecated` group at integration so
their ordinals stay reserved. `LongitudinalPlanSource.stop` is added.

## 2. Owner decisions (2026-08-29)

| # | Decision | Ruling |
|---|---|---|
| D1 | Architecture | mode in selfdrived, everything else in plannerd, shared stateless classifier |
| D2 | Turn budget | **removed after the 2026-08-29 field test** ("accelerating out of a curve feels held back"): cruise acceleration is bounded by the envelope alone; `curve-speed-limit`'s limiter and torque veto remain combo's only curve mechanism |
| D3 | `STOP_DISTANCE` | keep 7 m (owner prefers the extra distance); documented fork change |
| D4 | Acceleration-change cost through standstill | keep BLoTv2's behavior (cost stays on). Owner requirement: launches start smooth but grow quickly — acceptance metric: from a no-lead standstill launch, commanded acceleration reaches 50 % of the envelope within ~1.0 s with no dip; tune the low-speed cruise jerk or the supervisor launch response if not, never by removing the cost. Measured with the real MPC behind a departing lead: first step 0.13 vs 0.40 m/s² per frame with the cost off, half of peak at 1.35 s vs 0.70 s — the owner judges this in the phase-2 field test |
| D5 | Third model lead ("ponytail") | delete (owner never felt it act) |
| D6 | Supervisor lead speed | filtered `vLeadK`/`aLeadK`; MPC keeps raw `vLead` as stock; documented in one place |
| D7 | Supervisor stand-down → FCW alert | no; FCW keeps stock's form |
| D8 | Branches | retire `force-stops` (its README folded here); `BLoTv2` → `BLoTv2-Archive` after BLoTv3 is in combo; the radard low-speed age gate stays owned by `smooth-stops`; remove `force-stops` from `.github/workflows/sync-branches.yaml` (needs a PAT push) before deleting the branch |
| D9 | Module names | `longitudinal_lead.py` (kept — combo imports it), `necessity_supervisor.py`, `stop_helpers.py`, `force_stops.py`, `conditional_experimental_mode.py` |
| D10 | Hold release fallback | 4 s window in which ≥ 80 % of model frames show no stop tier, terminal speed ≥ 1 m/s and the stop corridor is lead-free — positive "clear" evidence, not absence |
| D11 | `selfdrived.py` on combo | collaborator (SOL/AOL) area; ask before phase 4d touches it; the CEM hook lands around the AOL calls without moving them |
| D13 | Committed approach profile (2026-08-29 field test 2) | Force Stops publishes its own plan candidate while a commitment is moving: the constant deceleration that lands `PROFILE_LANDING` short of the point, entered from the car's current deceleration at `PROFILE_JERK`, capped at `PROFILE_MAX_DECEL`, faded out between `PROFILE_HANDOVER_SPEED` and `PROFILE_FADE_SPEED` so the MPC column's own easing and the hold land the car. The MPC's quadratic stop column cannot be front-loaded (route 24: +1.45 → −1.86 over 1.5 s after a commit that needed 1.9 m/s²); the owner's own stops reach the needed deceleration within a second, hold it, ease off |
| D14 | Change-cost anchor on obstacle handoff | the MPC refills `a_prev` with the current acceleration whenever the binding obstacle column changes (lead0/lead1/stop), not only when an adaptive lead0 policy ends; a committed stop used to inherit the free run's accelerating solution |
| D15 | Commit speed | `QUALIFY_S` 1.0 → 0.3 s (world-fixed endpoint under strict evidence); a raw lead still blocks a *new* commitment, but only a *tracked* lead (`lead_filter` above `LEAD_GATE`) breaks an existing one — one radar frame reset a red-light commitment 0.5 s before the driver braked, and a flickering lead must not mint commitments between its frames |
| D18 | Landing taper (field test 4) | the profile never plans to reach its landing sooner than `PROFILE_MIN_TIME` (1 s): the need is at most v/2 near the end and only eases, instead of blowing up as the landing closes (route 27 t=1052, −2.4 m/s² at 2.6 m/s) |
| D19 | No speed cap once committed | `v_cruise_cap` is `NO_CAP` while forcing: the shaping cap's cruise floor used to land the car at −1.2 down to walking pace once the profile had faded (route 27 t=1053); the profile, the MPC column (which eases −0.9 → −0.4) and the hold own a committed stop |
| D20 | e2e against a committed profile | while the profile is moving the car, the model's own request joins the arbitration only if it is more urgent by `E2E_STOP_MARGIN` (0.5 m/s²): its late ramp used to overtake the flat profile through `min()` and put the heavy braking back at the end (route 27 t=250) |
| D21 | Green release and lane changes | a path longer than `RELEASE_OPEN_LENGTH` (30 m) for `RELEASE_OPEN_FRAMES` (3) releases a hold at once (saves ~0.2 s of the ~0.5 s the filtered release took; a one- or two-frame flash, route 27 t=263, does not); a lane change (`meta.laneChangeState`) drops shaping and a moving commitment so the stop re-qualifies on the new lane's endpoint (route 27 t=379: the through lane's line held a stop 15 m short of the left-turn lane's) |
| D17 | Latched point follows a drifting endpoint | the forward extension (`EXTEND_RATE`/`EXTEND_DEADBAND`) needs only the model still calling the stop with latch confidence, not the latch window: route 25 t=1547 (field test 3) drifted 3 m beyond a frozen commitment and, with the 5 m setback, headed for a stop ~10 m short of the line |
| D16 | Arbitration with e2e | unchanged: `min()` over MPC, cruise, e2e and the committed profile; with a front-loaded profile the car is slower when the model's late demand would come, so e2e loses the early phase and is milder late. Excluding e2e while committed stays an option if a drive shows otherwise |
| D12 | Fallback to "stop as MPC obstacle only, no mode switching" | if after two fix rounds the phase-3 field test still shows a resume pulse or `shouldStop` dither at a real stop, or the driver had to break a hold at a green more than once per ~10 stops |

## 3. Module contracts

### longitudinal_lead.py
`LeadObservation.from_radar(lead, service_valid)` (filtered speed/accel, finite, `dRel > 0`),
`lead_present(radar_state)`, `relevant_lead(radar_state, v_ego, path_end_m)` (BLoTv2's distance/time
relevance rule — the only filtered presence check in the tree), `anchor_model_lead(model_lead, radar_lead)`
(BLoTv2's validity gate plus first-horizon acceleration/speed, computed once per frame; since the
2026-08-29 field test the gate tolerates `MODEL_LEAD_STATIONARY_NOISE` = 0.2 m/s of below-zero sensor
noise on a stopped lead — the strict `>= 0` gate dropped the anchor for 0.1–0.5 s chunks through every
lead launch, collapsing the departure forecast, re-raising the stop bit and resetting the hold release;
a reversing lead still fails closed), and the
closing-speed / `total_decel_requirement` / TTC physics. `total_decel_requirement` is
`max(closing_requirement, stop_requirement)`, not the sum BLoTv2's doc stated.

### necessity_supervisor.py
`NecessitySupervisor.update(lead, v_ego, a_mpc_prev, t_follow_base, anchor) -> (jerk_scale, t_follow, stand_down)`.
Triggers, thresholds, slews, the whiplash ratchet (kept as its own guard) and both pad ceilings
(0.45 s onset braking, 0.75 s near-stopped lead) are BLoTv2's. Fixes: the pad ratio uses
`min(required_decel, ONSET_MAX_A_REQ)` so pads saturate instead of vanishing above 1.5 m/s²; the
low-speed hold latches only when the supervisor was necessity-braking in the frame before `v_ego`
crossed `MIN_SPEED`, and the emergency / lead-loss release paths clear it. `stand_down` never
reaches an alert. `JERK_SCALE_MIN` is the single clip source for `long_mpc.set_weights`.

### long_mpc.py
Stock functions untouched. Per frame: `set_cur_state(v, a)` then one
`update(radarstate, personality, lead0_anchor, lead1_anchor, stop_x, jerk_scale, t_follow_pad, prev_accel_constraint)`,
which in order: captures the previous lead0-adaptive flag; builds the lead0/lead1/stop obstacle
columns; `lead0_policy_active = argmin(x_obstacles[0]) == lead0`; on an adaptive lead0 → other
handoff refills `a_prev`; applies `jerk_scale`/`t_follow_pad` only while lead0 is active; calls
`set_weights` exactly once; solves; scores `crash_cnt` against the lead trajectory the solve used
(model-anchored when valid, radar extrapolation otherwise — a disclosed departure from stock FCW
sensitivity; scoring a radar-only path against a model-anchored solve would produce phantom
warnings). The third-lead machinery is removed. The MPC's acceleration bound stays opendbc's
`ACCEL_MAX` as in stock (BLoTv2's `min(ACCEL_MAX, 4.0)` always equalled it); only the cruise
envelope carries the 4.0 m/s² launch request.

### longitudinal_planner.py
Envelope (`a_max = 0.6 + 3.4 (1 − v/40)³`, clamped by opendbc's `ACCEL_MAX`), jerk schedule, ordinary-cruise
comfort shaping (with the `comfort_enabled` hook combo's curve limiter uses) as BLoTv2. No lateral
turn budget (D2). `update()` order: reset state → lead, anchors, stop observation (each once) →
`force_stops.update(...)` → `mpc.set_cur_state` → supervisor → `mpc.update(...)` →
`fcw = mpc.crash_cnt > 2 and not standstill` → candidates (MPC, cruise, e2e only in Experimental
mode with a valid model) → `output_should_stop = any(candidate stops) or force_stops.holding`.

### stop_helpers.py
`observe_model_stop(model, car_state, radar_state) -> StopObservation` — BLoTv2's tiers
(`shouldStop`, strict trajectory, early high-speed, early hint; BLoTv2's missing-velocity fallback
tier cannot occur with complete typed messages and is gone), straight-approach guard, relevant lead,
committed turn, and per frame the launch-evidence and corridor verdicts. `stop_release_open(model)` — one definition, non-braking not
required (combo's field-tested semantics). `leads_clear_of_stop_path(model, path_end_m)` — fails
closed unless **every** model-lead hypothesis with probability > 0 is outside the corridor, and on
any shape/finite irregularity (route-29 negative sentinel); a flat path, which is what the model
publishes at standstill, is a legal straight corridor, a reversing one is not. `MODEL_INVALID_RELEASE_S = 0.5` is
defined here and shared. Typed capnp access; no `getattr` guards.

### force_stops.py
`ForceStops.update(observation, car_state, experimental_mode, enabled, model_valid) -> (v_cruise_cap, stop_x, holding, a_target)`; the observation
carries lead presence/relevance, launch evidence and the corridor verdict, and `enabled` is the planner's own active signal.
States: `idle → shaping → committed → holding → (committed | idle)`.
- Entry requires Experimental mode (**entry only** — a later mode exit never releases a hold), no
  raw lead, a valid model, BLoTv2's tiers, path-length window and latch confidence, plus
  the committed-turn veto. Pre-latch shaping cap on the live endpoint as BLoTv2.
- `committed`: `remaining` decremented by ego travel, forward-ratcheted toward a re-extending
  endpoint (`EXTEND_RATE`/`EXTEND_DEADBAND`) and, below `DOWN_SPEED`, down-ratcheted toward a
  collapsing one (route 38 t=351); `LATCH_SETBACK`; `stop_x = max(remaining, −STOP_DISTANCE)`;
  `v_cruise_cap ≥ v_ego − DV_MAX`; `a_target` = the committed approach profile (D13), a plan
  candidate with source `stop` while the commitment is moving. Model invalid releases only after
  `MODEL_INVALID_RELEASE_S`. A tracked lead (not one radar frame) hands a moving commitment to the
  lead logic (D15).
- `holding`: entered at `CS.standstill` while committed, or within 10 s of a lead or a gas tap breaking
  a commitment or a hold when the car is stopped with stop evidence; `stop_x = 0` and `holding` forces
  `shouldStop`, so `controlsd`'s `cruiseControl.resume` cannot pulse. Leaves to `committed` (not
  idle) at `v_ego ≥ 0.8 m/s`, so an unsigned wheel-speed flicker on a grade never drops the latch.
- Release to idle: filtered launch evidence (`stop_release_open`, 0.30 s time constant); gas;
  brake; a **relevant** lead; model invalid ≥ 0.5 s; the D10 fallback. Fast re-entry: after a
  lead-triggered or gas-triggered release, if the car is at standstill again with stop evidence
  present, `holding` is re-entered directly; the 10 s gas grace suppresses only the shaping cap.
- Non-goal: no committed-lifetime + delay-projection scheme to move `shouldStop` earlier — tried
  on route 29 in BLoTv2 (0.450 s landed inside the 0.5 s actuator delay and weakened a fail-closed
  release). Holding engages at standstill only.

### conditional_experimental_mode.py
Runs every control tick (watchdogs, pedals, timers at `DT_CTRL`); evidence acquisition only on new
model frames. Entry filter, debounce, hysteresis, the 3 s recent-lead guard with corridor release,
committed-turn veto and post-stop/override suppression as BLoTv2. Fix: on control ticks a raw lead
may only revoke a *pending* recent-lead release, never wipe entry evidence; entry vetoes use
`relevant_lead`. Exits: resumed motion, stable clear, pedals, invalid model. selfdrived hook:
`experimental_mode = openpilotLong and (manual or (conditional and not driver_override))`, with
`driver_override` computed from `CS` in the 100 Hz hook.

## 4. Behavior: preserved vs changed
Preserved: envelope, comfort, jerk schedule, personalities (aggressive 1.0 s), supervisor triggers,
model-lead anchoring (lead0/lead1), lead-departure pre-release, CEM tiers/timers, Force Stops
shaping/commit/ratchets/setback/`DV_MAX`, `STOP_DISTANCE` 7 m, acceleration-change cost through
standstill, e2e candidate dropped while the model is invalid.
Changed: CEM can enter with an irrelevant lead in view; hold through standstill owned by Force
Stops and not released by a mode exit; FCW in stock's form, scored against the solved trajectory;
manual Experimental survives pedal taps; pad saturation and low-speed hold; third lead removed;
no turn budget; no cross-process stop fields; `longcontrol.py` is stock.

## 5. Phases and gates
Every phase: behavioral unit tests (real capnp messages, no constant echoes), the longitudinal
maneuver suite with an honest liveness shim, an rlog first-divergence replay against BLoTv2, then
the owner's field test before the next phase.
0. Branch, this document, harness (`plant.py` shim with real `valid/alive/freq_ok`, `enabled`,
   fully populated `leadsV3`, maneuvers that drop `radarState`/`modelV2`), longitudinal replay
   tooling with a pinned route manifest (d7, 29, d2, d9, 17, 27), and an audit of the combo-only
   Force Stops/CEM reconciliation commits.
1. Cruise layer. Field: Chill, no lead — urban and highway set-speed steps, one on-ramp; the D4
   launch metric.
2. Lead layer. Field: following, stopped-lead approach, pull-away, queue creep, a moderate lead
   brake from ~40 m (crosses 1.5 m/s²), a late low-speed lead brake (hold at partial softening).
3. Stop layer (also re-touches `long_mpc.py`/`longitudinal_planner.py` to wire `stop_x`/`holding`;
   the phase-2 lead-only replay is re-run with Force Stops idle first). Field: red lights with and
   without distant traffic, green release from hold, gas tap while holding then re-stop, signaled
   low-speed turn, a stop on a grade; D12 checkpoint.
4. Integration into `combo`, staged and each field-gated: (a) BLoTv3 alone on a scratch
   integration branch; (b) `smooth-stops` re-authored against BLoTv3's `LeadObservation` and its
   radard fix landed with a test, the dead BLoTv2 `longcontrol.py` clamp dropped; (c)
   `curve-speed-limit` composition; (d) SOL/AOL `selfdrived.py` merge (D11). Then retire branches
   and drop the audited combo-only commits.

## 6. Verification map
Envelope samples on the requested curve; comfort;
arbitration incl. hold; no FCW from the removed stand-down path; pad saturation across 1.5 m/s²;
low-speed hold at partial softening with emergency/lead-loss release still reaching 1.0; whiplash
ratchet and hold in one scenario; row-0 policy; single `set_weights`; `a_prev` refill on the exact
handoff frame; corridor rule with a 0.2-probability hypothesis; commit → hold → every release path;
flickering model stop signal while holding never drops `shouldStop`; grade flicker returns to
`committed`; lead passes through then fast re-entry; gas tap re-stop; CEM entry with a far lead;
model hang releases within 0.5 s; pedal latency; the selfdrived hook keeps manual mode under
override.

## 8. Field test log

**2026-08-29, route 23 (combo 83ccd10ab5), Palisade.** Owner: "accelerating out of a curve feels held back"; "at
stops with a lead, when it was time to accelerate, it felt like a harsh jolt rather than a smooth and quick
switch"; no difference noticed otherwise.

- Curve: the lateral turn budget (D2) was the only thing clipping cruise acceleration in bends. Removed.
- Lead launch, from the rlog and the CAN bus (TCS13 / SCC12 / SCC14 decoded):
  1. The car's ESP runs a fixed ~1.3–1.5 s standstill-exit sequence after `StopReq` drops, ignoring the
     acceleration request, then snaps its own reference to ~0.4 m/s² *above* the request at ~11 m/s³. The lurch
     therefore scales with whatever the plan asks at that instant; `JerkUpperLimit` does not gate it.
  2. Every launch's request had already climbed to 1.3–1.8 m/s² by then because the hold released late:
     `anchor_model_lead` rejected the lead's forecast on the −0.00…−0.04 m/s a stationary lead reads on both
     sensors, so the anchor flapped through the launch, the departure forecast collapsed, the stop bit flickered,
     the pre-release cancelled and the hold-release grace restarted. Fixed with a 0.2 m/s stationary-noise
     tolerance (reversing still fails closed). Like-for-like replay of the four launches: sustained release
     0.45–0.60 s earlier on three, stop-bit flicker 3 → 1, 37 short whole-route differences, all smoother.
  3. Smooth Stops' extra 0.5 s hold-release grace for a stopped radar lead was pure launch latency on top of the
     car's own sequence; removed on `smooth-stops` (10-frame debounce for every stop, immediate on a measured
     departing lead).
- Not changed: no launch-staging cap in LongControl — with the hold releasing ~0.9 s earlier the car should break
  free while the plan is still 0.3–0.5 m/s². Reassess after the next drive.

**2026-08-29, route 24 (combo 2f7ba629d0).** Two disengagements at red lights ("didn't feel like we'd stop in time"),
both Experimental-mode e2e stops without a lead; a launch felt odd. Owner baselines from the same drives: comfortable
stops from 18 m/s are ≈ −1.5 m/s² held for ~10 s then eased to −0.6; comfortable launches peak at 4 m/s² and hold ~3
to 4 m/s, tapering to 1.2 by 12 m/s. The car's SCC saturates at ≈ 1.6–1.9 m/s² for any request above 1.5 (CAN
`ACCEL_REF_ACC`), so openpilot launches are bounded by the vehicle.

- The model calls a red light 4–5 s out (strict evidence at ~70 m / 14 m/s, need ≥ 1.7 m/s²) and its e2e request
  ramps −0.7 … −1.3 → −2.5 into the last 3 s. Force Stops committed 4.5 s (#1) and 0.5 s (#2) before the driver braked:
  #1 lost every arbitration frame to e2e because the MPC's stop column started at +1.45 and took 1.5 s to reach −1.6
  (stale free-run `a_prev` under `A_CHANGE_COST`, quadratic obstacle cost); #2 was reset by one radar frame.
- Fixes: D13 committed approach profile, D14 change-cost re-anchor on any obstacle handoff, D15 faster commit and
  tracked-lead release. Closed-loop check: the maneuver plant gained a world-fixed stop line the fake model calls 5 s
  out (`Plant(stop_line=…)`); the red-light maneuver must stop short of the line with the needed deceleration reached
  within a second and eased off at the end.
- Launch: with the SCC's ~1.9 m/s² ceiling the remaining lever is starting early (field test 1); the green-light
  cap `LAUNCH_MAX_ACCEL` on combo (1.5) is worth raising toward the ceiling.

**2026-08-29, field test 3 (live report).** Stops follow the flat, eased profile but land earlier than the owner wants.
`LATCH_SETBACK` 5 → 2 m: the committed point sits that far short of the model's endpoint; 5 m compensated for the soft
column's overshoot, which the profile no longer has (route 24 red light 1: the endpoint sat ~1.5 m beyond the owner's own
stop). `PROFILE_LANDING` 4.5 → 2.5 m: the closed-loop plant stops about a metre past the landing, so the margin sets the
stop position almost 1:1 — the driven build landed ≈ 8.5 m short of the endpoint, this one ≈ 3.5 m with the same
−0.45 m/s² last second. Calibrated on routes 25/26 against the world-fixed committed point: the car stops ~0.6 m short of it (the plant's 3.5 m
column shortfall does not exist on the real ESP/hold), so the setback is the position lever and the landing margin is
feel; the owner's preferred spot sat ~2.7 m (route 25 t=1041), ~1.5 m (route 24) short of the model's endpoint and
once ~0.5 m past it (route 25 t=1558). `LATCH_SETBACK` 3 m ⇒ the car lands ≈ 3.6 m short of the endpoint, about a
metre before the owner's usual spot.

**2026-08-29, field test 4 (route 27, combo 6743a56844).** Owner: CEM stops jittery / not confident at the end, every green
about a second slow, one stale red (t=350), a lane change into a left-turn lane stopped far back (t=374), one harsh
landing (t=1052). Traces: the harsh end and the "heavy at the end" both came from the landing — the profile's need blew up
as the landing closed and, once the profile had faded, the shaping cap's −1.2 cruise floor drove the car to walking pace;
e2e's late ramp also overtook the flat profile through `min()`. D18–D20 fix those. Green launches: our release took
0.5–0.9 s after the model's path opened, the ESP a further 1.4 s (its standstill exit, not ours); D21 trims ours. t=350:
the model itself still called the stop until the driver pressed the gas — the path opened only afterwards. t=374: D21.
t=910 landed 3.5 m short of the endpoint by the 3 m setback; left as is.