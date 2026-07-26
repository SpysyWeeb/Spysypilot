# BLaTv2 — Better Lateral Tune v2

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot),
created from the untouched `stock` branch. See
[`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) for the full
fork overview.

## Status

⚠️ **In progress — shadow foundation.** This branch currently adds telemetry
and replay infrastructure only. It does not contain a BLaTv2 controller,
publish `carControl`, change actuation, or belong in `combo`.

The frozen v14 implementation remains on
[`BLaT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLaT). BLaTv2 is a
ground-up design and does not inherit that controller.

## Current scope

- a deterministic float64 steering-rack plant twin;
- a stateless scalar-anchored future-path reference;
- an onroad `blatv2_shadowd` process that publishes diagnostics only;
- route-audit replay using the identical library implementation.

The new shadow event is `blatV2Shadow`, version 1. It reports reference and
torque-demand values, actuator-feasible torque, one-step plant residual,
scalar/plan disagreement, horizon, validity, and per-frame runtime.

## Hyundai limits

This branch points `opendbc_repo` to
[`SpysyWeeb/opendbc:BLaTv2`](https://github.com/SpysyWeeb/opendbc/tree/BLaTv2).
That branch is based on stock opendbc and changes only the default Hyundai
command and panda-safety envelope from 384/3/7 to 409/4/7, plus the matching
safety expectations. It contains none of frozen BLaT's low-speed torque
damping.

## Not implemented

- controller or optimizer;
- observer state beyond the one-step residual;
- excitation module;
- UI;
- any merge into `combo`.

Shadow-foundation work stays **in progress** until device/harness bit-exactness,
the 2 ms p99 runtime budget, and the owner review all pass.
