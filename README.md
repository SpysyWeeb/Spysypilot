# BLaTv2 — Better Lateral Tune v2

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot),
created from the untouched `stock` branch. See
[`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) for the full
fork overview.

## Status

⚠️ **In progress — shadow controller tournament.** This branch computes two
BLaTv2 controller candidates as telemetry-only passengers. Neither candidate
publishes `carControl` or changes actuation. The shadow daemon is carried in
`combo` so ordinary drives can collect plant-fit and A/B evidence.

The frozen v14 implementation remains on
[`BLaT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLaT). BLaTv2 is a
ground-up design and does not inherit that controller.

## Current scope

- a deterministic float64 steering-rack plant twin;
- a stateless scalar-anchored future-path reference;
- one shared recorded-response disturbance observer;
- a deterministic sign-schedule torque MPC candidate;
- an analytic inverse-EPS finite-horizon LQI fallback candidate;
- an onroad `blatv2_shadowd` process that publishes diagnostics only;
- route-audit replay using the identical library implementation.

The shadow event is `blatV2Shadow`, version 3. It reports reference and
torque-demand values, actuator-feasible torque, one-step plant residual,
scalar/plan disagreement, horizon, vehicle speed, the self-aligning torque
estimate, alignment-input validity, overall validity, and per-frame runtime.
Version 3 adds the shared disturbance estimate and, for each candidate, the
commanded torque, status, candidate count, optimality residual, and isolated
device runtime. Solver status distinguishes invalid input, infeasibility,
non-convergence, and enumeration exhaustion. Equal-cost MPC schedules select
the lowest schedule index deterministically.

Candidate status values are `0=ok`, `1=input invalid`, `2=infeasible`,
`3=non-converged`, and `4=enumeration exhausted`. Observer status values are
`0=active`, `1=frozen by recorded actuator constraint`, and `2–6` the explicit
lateral-invalid, steering-pressed, standstill, model-invalid, and engagement
reset reasons, respectively.

Version 2 added the self-aligning tire load omitted by version 1. It mirrors
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

The 100 Hz path constructs `PlantParams`, `PlantTwin`, the observer, both
candidates, fixed-size numpy workspaces, two plant-state slots, one result, and
one Cap'n Proto message builder once at startup. Each frame overwrites that
storage.

All result fields except the three compute-time fields are deterministic replay
fields and must match the route-audit harness at the Float64/typed-integer bit
level. Runtime is an environment measurement: only per-candidate values logged
during real onroad operation on comma hardware gate the 2 ms p99 and 5 ms
hard-maximum budgets. Workstation replay timing is diagnostic only.

The candidates share one observer driven exclusively by recorded applied
torque and measured steering response; candidates read its estimate and never
write it. Observer learning resets on lateral/model invalidity, engagement,
steering press, and standstill, and freezes when the recorded actuator is
constrained. The fallback integral obeys the same reset/freeze lifecycle.

`SIGMA_Y = 0.05 m`, `SIGMA_HEADING = 0.01 rad`, and
`SIGMA_TORQUE_RATE = 0.5 normalized-torque/s` are provisional owner feel-dials
in `controller_seed_params.json`. They are a shared tuning vocabulary for both
candidates, not separately tuned gains. The provisional friction model always
charges the full `t_breakaway` against intended motion and independently bounds
the observer by another `t_breakaway`; the deliberate two-breakaway tolerance
is conservative until casual-drive evidence identifies a tighter value.

## Hyundai limits

This branch points `opendbc_repo` to
[`SpysyWeeb/opendbc:BLaTv2`](https://github.com/SpysyWeeb/opendbc/tree/BLaTv2).
That branch is based on stock opendbc and changes only the default Hyundai
command and panda-safety envelope from 384/3/7 to 409/4/7, plus the matching
safety expectations. It contains none of frozen BLaT's low-speed torque
damping.

## Not implemented

- any BLaTv2 actuation path;
- excitation module;
- UI;
- candidate feel tuning or architecture selection.

Shadow-foundation work stays **in progress** until device/harness bit-exactness,
the 2 ms p99 runtime budget, and the owner review all pass.
