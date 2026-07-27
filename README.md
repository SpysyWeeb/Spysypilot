# BLaTv2 — Better Lateral Tune v2

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot),
created from the untouched `stock` branch. See
[`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) for the full
fork overview.

## Status

⚠️ **In progress — shadow foundation.** This branch currently adds telemetry
and replay infrastructure only. It does not contain a BLaTv2 controller,
publish `carControl`, or change actuation. The telemetry-only daemon is carried
in `combo` so ordinary drives can collect plant-fit evidence.

The frozen v14 implementation remains on
[`BLaT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLaT). BLaTv2 is a
ground-up design and does not inherit that controller.

## Current scope

- a deterministic float64 steering-rack plant twin;
- a stateless scalar-anchored future-path reference;
- an onroad `blatv2_shadowd` process that publishes diagnostics only;
- route-audit replay using the identical library implementation.

The shadow event is `blatV2Shadow`, version 2. It reports reference and
torque-demand values, actuator-feasible torque, one-step plant residual,
scalar/plan disagreement, horizon, vehicle speed, the self-aligning torque
estimate, alignment-input validity, overall validity, and per-frame runtime.

Version 2 adds the self-aligning tire load omitted by version 1. It mirrors
frozen-v14's measurement pipeline: offset-corrected steering angle enters the
vehicle model with live roll/stiffness/steer ratio, and the resulting lateral
acceleration is mapped through the existing torque calibration without its
friction term. `t_breakaway` remains the sole Coulomb-friction model. Invalid
`liveParameters` uses zero roll/angle offset and nominal vehicle parameters for
that frame, with `alignInputsValid = false`; no stale values are carried.
`vEgo`, `aligningTorque`, and `alignInputsValid` describe the state-t side of
the reported state-t → state-t+1 residual (the bootstrap frame reports its
current state), so fit tooling can decompose each residual without shifting
telemetry streams.

The 100 Hz path constructs `PlantParams`, `PlantTwin`, fixed-size numpy plan
buffers, two plant-state slots, one result, and one Cap'n Proto message builder
once at startup. Each frame overwrites that storage. This removes the version-1
object churn that produced a uniform device timing tail.

All fields except `computeTimeSeconds` are deterministic replay fields and
must match the route-audit harness at the Float64 bit level. Runtime is an
environment measurement: only values logged during real onroad operation on
comma hardware gate the 2 ms p99 and 5 ms hard-maximum budgets. Workstation
replay timing is diagnostic only.

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
- any BLaTv2 actuation path in `combo`.

Shadow-foundation work stays **in progress** until device/harness bit-exactness,
the 2 ms p99 runtime budget, and the owner review all pass.
