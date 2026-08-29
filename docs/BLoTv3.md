# BLoTv3 — design

**Status: in progress. Not field validated. Do not mark complete before owner field testing and
explicit approval.** Phase 0 done and phase 1 (cruise layer) implemented 2026-08-29; phase 1 awaits the
owner's field test (Chill, no lead: urban and highway set-speed steps, one on-ramp, the D4 launch metric).

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

Only `selfdriveState.experimentalMode` crosses processes. The BLoTv2 fields
`SelfdriveState.conditionalStop{Qualified,Distance,ModelMonoTime,Latched}` are retired into a
`deprecated` group so their ordinals stay reserved and BLoTv2-era logs stay readable.
`LongitudinalPlanSource.stop` is kept.

## 2. Owner decisions (2026-08-29)

| # | Decision | Ruling |
|---|---|---|
| D1 | Architecture | mode in selfdrived, everything else in plannerd, shared stateless classifier |
| D2 | Turn budget | `a_total_max = max(envelope(v), interp(v, [20, 40], [1.7, 3.2]))` — launch never clipped, corners consume budget; composed before `curve-speed-limit`'s torque veto on combo |
| D3 | `STOP_DISTANCE` | keep 7 m (owner prefers the extra distance); documented fork change |
| D4 | Acceleration-change cost through standstill | keep BLoTv2's behavior (cost stays on). Owner requirement: launches start smooth but grow quickly — acceptance metric: from a no-lead standstill launch, commanded acceleration reaches 50 % of the envelope within ~1.0 s with no dip; tune the low-speed cruise jerk or the supervisor launch response if not, never by removing the cost |
| D5 | Third model lead ("ponytail") | delete (owner never felt it act) |
| D6 | Supervisor lead speed | filtered `vLeadK`/`aLeadK`; MPC keeps raw `vLead` as stock; documented in one place |
| D7 | Supervisor stand-down → FCW alert | no; FCW keeps stock's form |
| D8 | Branches | retire `force-stops` (its README folded here); `BLoTv2` → `BLoTv2-Archive` after BLoTv3 is in combo; the radard low-speed age gate stays owned by `smooth-stops`; remove `force-stops` from `.github/workflows/sync-branches.yaml` (needs a PAT push) before deleting the branch |
| D9 | Module names | `longitudinal_lead.py` (kept — combo imports it), `necessity_supervisor.py`, `stop_helpers.py`, `force_stops.py`, `conditional_experimental_mode.py` |
| D10 | Hold release fallback | 4 s window in which ≥ 80 % of model frames show no stop tier, terminal speed ≥ 1 m/s and the stop corridor is lead-free — positive "clear" evidence, not absence |
| D11 | `selfdrived.py` on combo | collaborator (SOL/AOL) area; ask before phase 4d touches it; the CEM hook lands around the AOL calls without moving them |
| D12 | Fallback to "stop as MPC obstacle only, no mode switching" | if after two fix rounds the phase-3 field test still shows a resume pulse or `shouldStop` dither at a real stop, or the driver had to break a hold at a green more than once per ~10 stops |

## 3. Module contracts

### longitudinal_lead.py
`LeadObservation.from_radar(lead, service_valid)` (filtered speed/accel, finite, `dRel > 0`),
`lead_present(radar_state)`, `relevant_lead(radar_state, v_ego, path_end_m)` (BLoTv2's distance/time
relevance rule — the only filtered presence check in the tree), `anchor_model_lead(model_lead, radar_lead)`
(BLoTv2's validity gate plus first-horizon acceleration/speed, computed once per frame), and the
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
warnings). The third-lead machinery is removed. `params[:, 1] = A_MAX`,
`A_MAX = min(ACCEL_MAX, A_REQUEST_MAX)` defined once in the planner.

### longitudinal_planner.py
Envelope (`a_max = 0.6 + 3.4 (1 − v/40)³`, clamped by `A_MAX`), jerk schedule, ordinary-cruise
comfort shaping (with the `comfort_enabled` hook combo's curve limiter uses) as BLoTv2. Turn budget
per D2. `update()` order: reset state → lead, anchors, stop observation (each once) →
`force_stops.update(...)` → `mpc.set_cur_state` → supervisor → `mpc.update(...)` →
`fcw = mpc.crash_cnt > 2 and not standstill` → candidates (MPC, cruise, e2e only in Experimental
mode with a valid model) → `output_should_stop = any(candidate stops) or force_stops.holding`.

### stop_helpers.py
`observe_model_stop(model, car_state, radar_state) -> StopObservation` — BLoTv2's tiers
(`shouldStop`, strict trajectory, early high-speed, fallback, early hint), straight-approach guard,
relevant lead, committed turn. `stop_release_open(model)` — one definition, non-braking not
required (combo's field-tested semantics). `leads_clear_of_stop_path(model, path_end_m)` — fails
closed unless **every** model-lead hypothesis with probability > 0 is outside the corridor, and on
any shape/finite irregularity (route-29 negative sentinel). `MODEL_INVALID_RELEASE_S = 0.5` is
defined here and shared. Typed capnp access; no `getattr` guards.

### force_stops.py
`ForceStops.update(observation, car_state, lead, experimental_mode, enabled, model_valid) -> (v_cruise_cap, stop_x, holding)`.
States: `idle → shaping → committed → holding → (committed | idle)`.
- Entry requires Experimental mode (**entry only** — a later mode exit never releases a hold), no
  relevant lead, a valid model, BLoTv2's tiers, path-length window and latch confidence, plus
  the committed-turn veto. Pre-latch shaping cap on the live endpoint as BLoTv2.
- `committed`: `remaining` decremented by ego travel, forward-ratcheted toward a re-extending
  endpoint (`EXTEND_RATE`/`EXTEND_DEADBAND`) and, below `DOWN_SPEED`, down-ratcheted toward a
  collapsing one (route 38 t=351); `LATCH_SETBACK`; `stop_x = max(remaining, −STOP_DISTANCE)`;
  `v_cruise_cap ≥ v_ego − DV_MAX`. Model invalid releases only after `MODEL_INVALID_RELEASE_S`.
- `holding`: entered at `CS.standstill` while committed; `stop_x = 0` and `holding` forces
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
turn budget active again; no cross-process stop fields; `longcontrol.py` is stock.

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
Envelope samples on the requested curve; comfort; turn budget with steering ≠ 0 (unit test);
arbitration incl. hold; no FCW from the removed stand-down path; pad saturation across 1.5 m/s²;
low-speed hold at partial softening with emergency/lead-loss release still reaching 1.0; whiplash
ratchet and hold in one scenario; row-0 policy; single `set_weights`; `a_prev` refill on the exact
handoff frame; corridor rule with a 0.2-probability hypothesis; commit → hold → every release path;
flickering model stop signal while holding never drops `shouldStop`; grade flicker returns to
`committed`; lead passes through then fast re-entry; gas tap re-stop; CEM entry with a far lead;
model hang releases within 0.5 s; pedal latency; the selfdrived hook keeps manual mode under
override.
