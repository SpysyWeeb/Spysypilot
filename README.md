# BLaTv3

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: ⚠️ in progress — phase 1 (behavior-preserving port of BLaTv2) merged into combo and field-validated 2026-08-29; phase 2 in progress: steps 2 and 3 (hold through model gaps; modeld's curvature preview as the rack path) merged into combo and field-validated 2026-08-29 (routes 00000023/24: every model frame carried the preview, all engaged frames `active`, no fallbacks, all stop approaches clean; the path-compile fix took controlsd from 36 % to 25 % of its core with zero skipped control frames); step 4 (bounded reference filter + scheduled preview) next.**

## What it does

BLaTv3 is the rewrite of the Palisade rack-trajectory lateral controller
([BLaTv2](https://github.com/SpysyWeeb/Spysypilot/tree/BLaTv2)). The goal is unchanged —
steer the Palisade better than stock by executing the model's path as one smooth, swift,
strong rack motion — but the controller is rebuilt in the shape upstream uses for a lateral
controller, designed from a written catalog of the ways it can fail, and it retires the static
"comfort envelope" tables in favor of two learned rack-effort surfaces.

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
