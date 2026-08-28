# BLaTv3

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: ⚠️ in progress — phase 0, no controller code yet, nothing field-tested.**

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

- `.gitmodules` / `opendbc_repo` — tracks SpysyWeeb/opendbc `blatv2-409-horizon`
  (`69818202`), which selects the 409/+4/−7 Hyundai torque envelope and the matching panda
  safety flag for the Palisade platform. Same pin as BLaTv2.
- `docs/BLaTv3_FAILURE_MODES.md` — the design-for-failure catalog.
