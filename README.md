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

The shadow event is `blatV2Shadow`, version 6. It reports reference and
torque-demand values, actuator-feasible torque, one-step plant residual,
scalar/plan disagreement, horizon, vehicle speed, the self-aligning torque
estimate, alignment-input validity, overall validity, and per-frame runtime.
Version 6 makes MPC warm selection literal O(1): it reuses the previous
winning schedule ordinal when that ordinal still exists and otherwise selects
base schedule zero. Version 5 prepares the identical candidate workspace once
in shared setup and
lets both candidates consume it read-only. It also hard-bounds MPC to one
active-set solve per frame: `mpcCandidateCount` is the evaluated count and
`mpcAvailableScheduleCount` reports how many base/early/late schedules existed
before the bound. The selected schedule is the lowest-index sign sequence
closest to the previous converged solution; lifecycle resets return to the base
schedule. Version 4 split runtime into shared `begin_frame` setup, MPC-only
solve, fallback-only solve, and total dual-candidate frame time. Starting in
version 5, shared setup includes the once-per-frame candidate workspace;
candidate timings exclude it. Version 3 added the shared disturbance estimate
and, for each candidate, the commanded torque, status, candidate count,
optimality residual, and isolated device runtime. Solver status distinguishes
invalid input, infeasibility, non-convergence, and enumeration exhaustion.

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
storage. The device publisher converts every core result to its native Python
`float`, `int`, or `bool` type at the Cap'n Proto boundary because solver
outputs may be numpy scalars. `blatv2_shadowd` is manager-restartable: it is a
structurally passive telemetry process, so recovery after a crash preserves
route data without introducing a control path.

On-device stage profiling found the candidate floor in Python horizon loops,
not LDLT or object construction. The candidate workspace now walks the
monotonic reference once instead of binary-searching every sample. The fallback
uses a scalar-unrolled, arithmetic-order-preserving 3×3 Riccati recursion. When
MPC has one sign schedule, version 4 skips the 100 Hz rollout cost because that
cost only ranks competing schedules; the converged schedule is already the
unique winner. Seven device-side cases covering workspace, fallback, and
one-schedule MPC outputs remained bit-exact to the pre-optimization
implementation. That version-4 optimization changed no tuning constant or
candidate command; version 5's bounded multi-schedule selection is the
subsequent, explicitly telemetered architecture change.

Route `000000b2--be389808d5` showed that the remaining fallback tail was not a
fallback branch: timing had negligible correlation with candidate count,
speed, reference, torque demand, constraints, observer status, same-frame MPC
time, or adjacent MPC load. A sustained device benchmark recorded zero Python
garbage collections. Direct stage timing assigned about 80% of fallback CPU to
the candidate workspace (2.14 ms median offroad) and about 20% to Riccati
(0.52 ms). Version 5 therefore removes the duplicate workspace construction
and deletes the 100 Hz comparison-reference grid that became unused once MPC
comparison rollouts were bounded. The next ordinary drive remains the
authority for the 2 ms/5 ms runtime gates.

The same route found multiple schedules on 6.84% of frames, with nine
schedules costing 59 ms median and 17 costing 117 ms. Version 5 never makes
solve count depend on that population. Available schedules are represented as
one changed sign relative to base; warm selection is O(horizon + crossings),
and only the selected schedule is materialized and solved. A deliberately
harsh 23-schedule device benchmark reduced MPC solve median from 3.88 ms with a
schedule-wide warm scan to 1.46 ms with the compact representation. This is a
runtime architecture bound, not a feel dial or tuning constant.

Route `000000b3--618af36e75` verified that the algorithmic floor was gone
(shared/MPC/fallback medians 0.94/0.73/0.45 ms), but all three independent
phases retained an uncorrelated 9–12 ms p99 tail. Its `procLog` records
`blatv2_shadowd` as an ordinary priority-20 process migrating across CPUs
0/1/2/4/5, while `controlsd` is real-time on CPU 4 and planners are real-time
on CPU 5. Shadowd now uses the existing low-control real-time priority on CPU
4. The higher-priority `controlsd` always preempts it, preserving control
authority, while unrelated ordinary onroad work no longer interrupts whichever
candidate phase happens to be running. The standard real-time setup also
disables cyclic GC before the hot loop. No custom priority or affinity constant
was added.

Route `000000b4--ef6ec45105` showed that version 5's solve bound was not enough:
MPC time still rose monotonically from 0.73 ms with one available schedule to
38 ms with 23, while shared and fallback time did not correlate with schedule
population. Version 6 removes the remaining schedule-population scan entirely.
CPU 4 affinity remains unchanged for this experiment so the next route can
separate selector cost from the independently observed controlsState throughput
loss; affinity is reconsidered only after the O(1) selector is measured.

All result fields except the four compute-time fields are deterministic replay
fields and must match the route-audit harness at the Float64/typed-integer bit
level. Runtime is an environment measurement: only shared-plus-candidate values
logged during real onroad operation on comma hardware gate the 2 ms p99 and
5 ms hard-maximum budgets. Workstation replay timing is diagnostic only.

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
