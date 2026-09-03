# BLaTv3

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: ⚠️ in progress — phase 1 (behavior-preserving port of BLaTv2) merged into combo and field-validated 2026-08-29; phase 2 in progress: steps 2 and 3 (hold through model gaps; modeld's curvature preview as the rack path) merged into combo and field-validated 2026-08-29 (routes 00000023/24: every model frame carried the preview, all engaged frames `active`, no fallbacks, all stop approaches clean; the path-compile fix took controlsd from 36 % to 25 % of its core with zero skipped control frames); step 4 (bounded reference filter + scheduled preview) merged 2026-08-29, corrected 2026-08-30 after its first
drive (the preview no longer replaces the near target) and field-validated 2026-08-30 (route 00000029: the served target within
0.03 m/s² of the near target's demand while the preview is open, all engaged frames `active`, no preview with hands on). Phase 3
in progress: step 1 (torque-tail continuity) merged and field-validated 2026-08-30; step 2 merged 2026-08-30 — its slew-aware early
release shed holding torque mid-curve on its first drives and was retired the same day, the rest of the step (unwind clamp replaced
by a direction fraction, applied-torque plumbing) stands and awaits a clean drive; step 3 is a log-only on-device hold-torque
learner (the global recalibration of torqued's prior was falsified per cell and never enters the torque path); a 65-agent panel
review of the whole phase is in `route-audit/phase3/panel_2026-08-30/`. Step 2's direction fraction field-validated 2026-09-01
(routes 34/35, 40 min on the release-retirement build: unpressed hold collapses 8–9 per route → 0–1). Step 4 (direction guard v2)
and step 3-C (the log-only rack-effort shadow observer) merged into combo 2026-09-01 and driven 2026-09-01 (routes 36/37, 15 min):
zero-torque tails while active 2.8–3.0 % → 0.4–1.0 %, none longer than 60 ms (was 1.2–2.3 s) and none guard-caused (was 99.8 %),
near-center exact-zero duty 1.7–2.0 % → 0.0 %, the R7 output-step bound held at exactly 0.05, the observer logging 23–25 cells with
finite biases; the guard now acts mostly at low speed with the driver's hands on, and a 149 ms hold-collapse flag at 12 m/s was
post-override recovery, not a shed. Step 4b (driver-assist envelope) and step 5 (R4 proactive envelope opening) merged into combo
2026-09-02, awaiting a drive — the post-release torque swing baseline to compare against is p50 0.18–0.37, p90 0.53–0.73
(`route-audit/phase3/verify_2026-09-02/`). `params_keys.h` changed with step 3-C: a device that crash-loops with `UnknownKeyName`
after updating needs `rm -f /data/openpilot/openpilot/common/params_pyx*.so` and a reboot.**

## What it does

BLaTv3 is the rewrite of the Palisade rack-trajectory lateral controller
([BLaTv2](https://github.com/SpysyWeeb/Spysypilot/tree/BLaTv2)). The goal is unchanged —
steer the Palisade better than stock by executing the model's path as one smooth, swift,
strong rack motion — but the controller is rebuilt in the shape upstream uses for a lateral
controller, designed from a written catalog of the ways it can fail, and it is meant to retire the static
"comfort envelope" tables in favor of two learned rack-effort surfaces (the tables still bound
the planned rate today; the learned surfaces are phase 4).

Kept from BLaTv2 by owner decision: turn-in boost and unwind boost (as feedforward physics
rather than sign rules), the model as the sole path authority, and the 2 s horizon — this
time as a real scheduled preview that looks far ahead on straights and pulls in as curvature
builds, never past a refuge island.

## How it works (planned)

- `LatControlRack(LatControl)` in one file, selected by a `CarParams` flag that opendbc sets
  from the Palisade `LX` firmware code; Telluride and unknown firmware stay on stock.
- Reference → motion → tracking → output layers; a warm stock `LatControlTorque` runs in
  shadow every frame and takes over on any invalidation (never zero torque).
- Two separately learned surfaces — hold torque and rate gain over speed, angle, lateral load
  (incl. bank) and direction — used together for feedforward and for a dynamic motion envelope.
- Twelve governing rules (near-target authority, preview by path consistency in metres, fail
  soft to stock, every platform limiter modeled, …) and the failure-mode catalog in
  [`docs/BLaTv3_FAILURE_MODES.md`](docs/BLaTv3_FAILURE_MODES.md) are the spec the
  implementation is tested against.
- Phases, each field-tested before the next: (0) branch, envelope pin, catalog;
  (1) behavior-preserving port of BLaTv2, replay-identical; (2) scheduled preview;
  (3) rack-aware output and asymmetric feedforward; (4) learned surfaces.

## What changed

- `openpilot/selfdrive/controls/lib/latcontrol_rack.py` — `LatControlRack(LatControl)`: steers with
  the rack trajectory controller when it has a request and with a stock `LatControlTorque` when it
  does not. The stock controller is stepped every frame so its request buffer and jerk filter follow
  the live history and the two share one saturation timer; as in BLaTv2, its integrator starts clean
  when it takes over. Logs into `lateralControlState.rackState`.
- Phase 3, step 6 — the hold top-up (FM3.14), the third torque term: feedforward predicts, position and
  rate feedback correct, and a bounded (0.20), leaky (3 s; 0.3 s under a press and through a 0.3 s
  release cooldown), angle-space integrator makes up whatever standing shortfall is left while the wheel
  is meant to hold — in degrees, deliberately outside the `gain(v) · lateral_accel_per_degree` pipeline
  whose v²-scaled feedforward under-supplied the real hold effort on route 0x3e (request −0.57 → −0.36 at a
  constant 35° as the car slowed 9.4 → 7.9 m/s; the wheel crept out within 1° of the plan and let go).
  Growth only while the plan and the wheel are still and the wheel is not already closing on the plan,
  never against a wanted unwind, never while a press, the release cooldown, the direction guard, the
  platform clip or the feedback cap already binds in the error's own direction; per-frame step ≤ 0.0067
  (R7), snaps to exact 0.0, reset with the controller, never persisted. Logged as `holdTopupTorque` /
  `holdTopupGrowing`. Design panel and refutation round in `route-audit/phase3/step6_topup_design/`.
- Phase 3, step 5 — horizon-implied envelope opening (R4, G-independent): a second,
  confidence-free `PreviewScheduler` (`envelope_scheduler`) reuses every R2 gate except the
  confidence check, corroborated instead by its own frame-to-frame far-point stability check
  (`PREVIEW_ENVELOPE_DRIFT_M = 0.25 m`, from route 20's near/far swing decode: the 2 s point
  swings 0.2-2.0 m through the island window while the 1 s point stays 0.03-0.25 m). The admitted
  horizon's own implied rate/acceleration, margined 1.15×, is capped at the upstream ISO clamp
  evaluated exactly with the real VM (2.34× comfort at 40 mph, narrowing to 1.37× by 70 mph — no
  corpus re-derivation needed) through a bounded one-pole state that snaps shut the instant the
  demand or the admitted horizon falls and eases open over one horizon step (0.25 s) otherwise;
  feeds `_motion_limits` unmodified, so the feasibility ratchet still narrows on top and torque
  authority is untouched (R10). Logs `envelopeRateDegS`/`AccelerationDegS2`/`JerkDegS3`/`PreviewTime`
  into `rackState`. Nine new unit tests (confidence independence, the opening/collapse continuity
  sweep, the drift-graft rejection, FM1.18 dither immunity, forced-zero parity, the R10
  architectural check, the call-order regression, and an island-jog walkthrough) join the 68
  carried over unchanged. Replay validation against the field routes is pending.
- Phase 2, step 2 — a dropped or invalid model frame keeps the last good plan (the reference already
  advances along it by its age); a model is stale only past SubMaster's 0.5 s alive window instead of a
  private 0.2 s; once stock has taken over it keeps steering for 0.5 s before the rack re-seeds and
  resumes; a `latActive` blip of up to five frames (the standstill gate) holds the planned rack instead
  of starting over. Replay on the two field routes: no change in ordinary driving.
- Phase 2, step 3 — modeld publishes a curvature preview on `ModelDataV2.Action`
  (`desiredCurvaturePreview` / `desiredCurvaturePreviewTimes`): `desiredCurvature`'s own function
  evaluated every 0.25 s from the action time to 2 s past it, pinned so the first sample *is*
  `desiredCurvature`. The rack controller builds its path from that preview (re-pinned to controlsd's
  ISO-clipped scalar) instead of re-deriving curvature from `orientationRate / velocity`. The preview
  is the scalar's own timeline from the action time on, and lagd's action time already covers the
  model's age, so the immediate target is the scalar as published (stock's target), the far targets are
  the same quantity further past the action time, and a target's position and rate come from one curve
  (catalog FM1.4). A model without a preview reports status 6 `invalid preview` and stock steers.
  Open-loop replay on routes 00000020/21/22 (571k engaged frames, all `active`, no fallbacks): at
  20–40 m/s the immediate target moves by 0.10° (p50) / 0.35° (p95) and torque by 0.013 / 0.055; below
  10 m/s the immediate target becomes the scalar itself instead of lagging it ~12 % while curvature
  builds — the intended fix, and the part a drive has to judge (intersection turns, roundabouts, creep).
  Field-validated 2026-08-29 (route 00000024, "no complaints").
- Phase 3, step 4b — the driver-assist cap relaxes when the driver agrees: instead of a fixed 0.5 clip
  whenever the wheel is held, the cap is a per-frame envelope mirroring opendbc's own driver-allowance
  term for the commanded direction (a driver pushing with the controller lifts the cap toward the ISO
  ceiling as far as the platform limiter would; an opposing driver still floors at 0.5), with the R7
  step bound inside the branch. Replay: unpressed frames byte-identical; cap-limited hold collapses
  2d 9 → 2, 2e 6 → 0. ⚠️ awaiting a drive.
- Phase 3, step 4 — direction guard v2: when the plan and the served target disagree in sign, torque
  blends toward a bounded, target-referred feedback (cap 0.18) instead of ramping to exactly zero;
  the two boolean trips became continuous conflict fractions and R7 is enforced on the output step
  within the guard's own blend. Replay vs step 3: exact-zero-while-active duty 1.2–1.7 % → 0.000 % on
  three routes, bit-identical outside conflict. Merged into combo 2026-09-01, awaiting a drive.
- Phase 3, step 3-C — the rack-effort shadow observer (`selfdrive/locationd/rack_effort_observer.py`):
  a log-only sibling of torqued recording the hold torque the EPS actually applies against the physics
  prior, per speed/angle/lateral-load/direction cell, bit-exact with the offline seed extractor; zero
  torque effect by construction and by test. Its cells earn authority only later, behind a promotion
  bar (two routes, no route over half the events, ten events). Merged into combo 2026-09-01.
- Phase 3, step 2 — the controller reads the torque the EPS is actually applying (`carOutput`) and
  uses it for the direction fraction below; a slew-aware early release was built, closed-loop
  validated, and then RETIRED after its first field drive (route 2d, six owner bookmarks): the
  rate-flip test read visible curve exits as coming reversals, so during ordinary sustained curves the
  release latched at benign wheel-past-target dither and shed holding torque while the curve tightened
  (output 0.44 → 0.05 in 1 s, the car ran wide toward the curb). With zero measured reversal-lag
  benefit in every plant window, it is removed rather than re-gated; the applied-torque plumbing and
  log fields stay for a future redesign against a true torque-reversal test. And the unwind magnitude clamp
  is retired: a signed continuous direction fraction relaxes only the rate feedback during unwinds, so a
  return the rack's own self-aligning torque is producing is not resisted, while the feedforward keeps
  following the plan. Closed-loop A/B vs step 1 on the four hardest windows (the 2b and 2c s-turns, the
  route-23 owner unwind, the 2c 63–74 s unwind): equal or better on every one, release duty 0–0.8 %.
  A 16-agent review then confirmed and fixed four more: the flip time is interpolated inside the
  horizon grid (was a 0.25 s snap that could swing the blend by half the torque scale in one frame),
  a target sitting exactly at zero no longer blinds the sign test, `carOutput` is alive/valid-guarded
  (a dead card disables the release instead of latching it), and the direction fraction fades out
  within 3° of center so a near-center dither cannot strip the rate damping frame to frame. One
  deliberate feel change to listen for on the drive: with the unwind clamp retired, holding torque
  through an ordinary curve exit is higher than step 1 (up to ~+0.2 on the ±1 scale at highway speed
  in a synthetic sweep; p95 +0.07 over the two replay routes) — the plan is held honestly instead of
  being bled off by the wheel's own position.
  ⚠️ on combo since 2026-08-30 with the release retired; awaiting a clean drive.
- Phase 3, step 1 — continuity and truth-telling in the torque tail, from the phase-3 design panel
  (`route-audit/phase3/design/phase3_design.md`): the turn-in feedback cap blends continuously between
  its two values instead of jumping 0.35 units at a boolean; the direction guard ramps down at its own
  time constant instead of snapping to zero; feedback keeps its authority at standstill (the per-degree
  gain used the raw speed, so creep steering sat inert); `saturated` now means the platform ceiling
  alone, with the direction guard and driver-assist cap logged as their own fields. Replay on routes
  20–24/28 (1.0M engaged frames): 2–4 % of frames change, exactly where intended — three quarters of
  all standstill frames (the restored authority), the direction-guard events (the old instant zero-torque
  holes become a 0.12 s ramp-out, the largest deltas), and the turn-in boundary (|Δ| p99 0.009–0.036
  at speed); everywhere else unchanged.
- Phase 2, step 4 — the immediate target passes through a bounded reference filter and is read at a
  scheduled preview time; the small-reversal governor is retired. `ReferenceFilter`: first-order,
  τ = 0.1 s, and the served target may trail the model's by at most `min(0.2 m/s² / v², 3°)` — so a
  turn-in or unwind passes at once, at its own rate, short of the raw target by no more than 3°, while
  the 4–7° per-frame jitter of the model's target at 5–30 mph is smoothed. `PreviewScheduler`: the
  plan is trusted up to 2 s past the action time, one 0.25 s step per two agreeing model frames, only
  while the model's own path stays within 0.15 m of the clothoid between the near and far targets
  (0.20 m to keep), the far wheel angle within 1° of the near one, the path's uncertainty low, the far
  target no jumpier than the near one, and at most 40 m ahead; two disagreeing frames, hands on the
  wheel, a lane change or a limited target collapse it. The preview never replaces the near target —
  the first cut did, within the 1° gate, and drifted-then-corrected on straights (route 00000028:
  52 such cycles in 46 min, 0.05–0.14 m/s² of the near target's correction ignored while open) — it only earns the tracker a calmer
  reference (filter time constant 0.1 → 0.3 s) and a longer response time (0.4 → 0.5 s). Costs
  +0.4 ms a frame on the device.
  Field-validated 2026-08-30 (route 00000029, "road test good").
  Open-loop replay on routes 00000020–24 (754k engaged frames, all `active`, no fallbacks): at 5–30 mph the served
  target's per-frame step p95 1.1–2.1° → 0.27–0.52° and its reversals −85–92 %, never more than 3° from the model's;
  the twelve largest low-speed turn-ins/unwinds per route pass within that bound with no far preview; the recorded
  refuge-island event keeps the preview at the action time; on highway straights the preview is ≥ 1 s for 30–54 % of
  the time in runs of ~2 s median (p90 11–14 s), held back mostly by gentle curvature inside the next 2 s.
- Phase 2, cost — `model_path_targets` interpolates every query and its rate stencil in one pass per
  series (81 scalar `np.interp` calls a frame became two) and the scalar clips in `bound_target` and the
  torque tail use `min`/`max`: bit-exact (20k randomized inputs; A/A replay on routes 00000020/21/22),
  `RackTrajectoryController.update` 4.0 → 1.75 ms per frame on the device's little cores. Motivation:
  route 00000022 ran core 4 (card + controlsd + selfdrived) at 100 % with card merging CAN batches at
  highway speed, and the rack controller was 5.7 ms of controlsd's 10 ms frame against 0.45 ms for stock.
- `openpilot/selfdrive/controls/lib/rack_trajectory.py` — BLaTv2's controller math (reference
  compile, jerk-limited rack planner, reversal governor, rate estimator, torque tail) moved unchanged
  into one module: the class is renamed `RackTrajectoryController`, the one-line `_measured_rate`
  wrapper is inlined at its call, and the re-export facade, an unused governor parameter and the
  test-only `model_path_target` helper are dropped. Replay against combo's controller on two field
  routes (312,983 frames) is bit-exact.
- `openpilot/selfdrive/controls/controlsd.py` — selects `LatControlRack` when
  `CarParams.lateralTuning.torque.useRackTrajectory` is set and fills the `rackState` union arm; no
  brand-specific imports, no `isinstance` hooks.
- `openpilot/selfdrive/controls/lib/latcontrol*.py` — `LatControl.update` gains `model` and
  `mono_time_ns` for controllers that consume the model plan (as `lat_delay` was added upstream).
- `openpilot/cereal/log.capnp` — `ControlsState.lateralControlState.rackState @67 :LateralRackState`;
  `ModelDataV2.Action.desiredCurvatureTime @3`.
- `openpilot/selfdrive/modeld/modeld.py` — publishes the plan time at which `desiredCurvature` is read.
- `.gitmodules` / `opendbc_repo` — tracks SpysyWeeb/opendbc `BLaTv3` (`a2c02f09`): adds
  `LateralTorqueTuning.useRackTrajectory`, set for the Palisade only when the queried platform code is
  `LX` (not Telluride `ON`, not unknown), and gates the 409/+4/−7 envelope and its panda safety flag on
  the same test, so unknown firmware fails closed to the stock controller and the stock 384/3/7 envelope.
- `docs/BLaTv3_FAILURE_MODES.md` — the design-for-failure catalog.
