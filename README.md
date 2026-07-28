# BLaTv2 — Better Lateral Tune v2

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot),
created from the untouched `stock` branch. See
[`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) for the full
fork overview.

## Status

⚠️ **In progress — LQI promotion build.** The tournament selected the analytic
inverse-EPS finite-horizon LQI. On Hyundai torque cars this branch now runs the
shared BLaTv2 reference, observer, plant workspace, and LQI directly inside
controlsd at its existing CTRL_HIGH 100 Hz position. Those numerical files are
the same artifact imported by route-audit; there is no separate live variant.

The MPC is retired from onroad execution. Its implementation and tests remain
in the repository so it can re-enter a future tournament after its real-world
schedule-construction cost is fixed.

The frozen v14 implementation remains on
[`BLaT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLaT). BLaTv2 is a
ground-up design and does not inherit that controller.

## Current scope

- a deterministic float64 steering-rack plant twin;
- a stateless scalar-anchored future-path reference;
- one shared recorded-response disturbance observer;
- a retained, non-running sign-schedule torque MPC challenger;
- the promoted analytic inverse-EPS finite-horizon LQI controller;
- an onroad `blatv2_shadowd` process that runs the complete frozen-v14
  controller passively for honest live A/B telemetry;
- route-audit replay using the identical library implementation.

The shadow event is `blatV2Shadow`, version 8. It reports reference and
torque-demand values, actuator-feasible torque, one-step plant residual,
scalar/plan disagreement, horizon, vehicle speed, the self-aligning torque
estimate, alignment-input validity, overall validity, and per-frame runtime.
Version 8 uses the live-field-calibrated plant schedule described below.
Version 7 removed both tournament candidates from shadowd. It copies the live
LQI command, status, recovery state, and in-controlsd compute time from
`controlsState`, and logs the command from the complete frozen-v14 pipeline.
The selected response-surface controller reports version `203`. Version 203 keeps
plant stiction and the observer clamp at the measured `0.09` while replacing
the moving-friction feedforward magnitude with the b7-selected `0.03`; full
`0.09` breakaway remains available when departing stiction. It adds the
curvature-space tracking tolerance `sigma_curvature = 0.00091683 1/m` while
leaving the three original sigma dials unchanged.

### b7 response-surface selection and accepted deviations

The owner selected kinetic feedforward `0.03` and
`sigma_curvature = 0.00091683 1/m` from the 16-cell b7 surface for staged field
evaluation. Plant stiction and the observer clamp remain `0.09`; only moving
decision cells use `0.03`. Curvature-space weighting naturally rolls tracking
authority down with the vehicle model's curvature-per-steering-degree response
at speed.

Two failed gates are owner-authorized deviations, not passes:

1. Raw worst-1-second torque-rate RMS is `28.89/s` versus v14's `26.08/s`.
   The applied domain—the slew-limited signal reaching the rack—passes all
   five metrics against v14, and the controller now reaches full authority in
   sharp turns by design. The owner chose field evidence over another replay
   iteration.
2. The required 20–25, 25–30, and >30 m/s oscillation bands are unscoreable on
   sole field route `000000b7--a6b3b1f175`, whose maximum speed is
   `18.43 m/s`. In the measurable 15.6–20.1 m/s band, v203 is the calmest of
   candidate/live/v14 (`0.02427 / 0.04268 / 0.05258` RMS). The
   curvature-per-angle term gives approximately `0.41×` high-speed weight at
   25 m/s versus low speed. The owner explicitly waived the unsatisfiable
   coverage gate and elected staged field evaluation.
The v14 controller and reference-planner source files are byte-identical to
frozen authority commit `5e533e3ec6`; the passive adapter does not substitute a
terminal FF/PID hybrid.

### Live invalid-output contract

An active non-OK or non-finite LQI result holds the previous request for one
frame, then decays it toward zero with the runtime actuator down-rate. At
250 ms continuously invalid, controlsd requests zero and publishes both
`controlsState.valid = false` and `carControl.valid = false`. This deliberately
surfaces as the existing `commIssue` event. It is not a communication defect:
for this branch, `commIssue` can mean the lateral controller remained invalid.

That tested event path produces the standard full-screen
**TAKE CONTROL IMMEDIATELY** visual alert, steering-required visual signal, and
`warningSoft` audible alert while the three-second soft-disable state counts
down; it also supplies the existing no-entry alert while invalid. Both the
controller and no-entry condition clear only after ten consecutive finite OK
frames. Re-entry resets the LQI integral and resumes through the same asymmetric
slew limiter from current applied torque.

Version 6 made MPC warm selection literal O(1): it reuses the previous
winning schedule ordinal when that ordinal still exists and otherwise selects
base schedule zero. Version 5 prepared the identical candidate workspace once
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
vehicle model with live roll/stiffness/steer ratio. Version 8 retains that
geometry, roll, offset, and platform-sign convention, but replaces the
constant live-calibration slope with the measured steady-state
torque-per-lateral-acceleration schedule in `plant_seed_params.json`. The six
nodes are 2.5/5.5/8.5/12.0/16.5/21.0 m/s with gains
0.85/0.39/0.38/0.36/0.286/0.288 normalized torque per m/s². Interpolation is
linear and extrapolation is flat. The fit used 104k clean engaged frames from
LQI route `000000b6--6d24c922f5` with v14 route
`000000b5--751a540298` as controller-independent cross-validation; the
low-speed node is b5-weighted because b6 was under-actuated there. The method
authority is route-audit `phase0/plant_fit_reference.py` at `313de6b`.

`t_breakaway = 0.09` remains the sole Coulomb-friction model and automatically
bounds the observer to ±0.09. `k_t = 4000 deg/s²`, `b_steer = 10 1/s`, the
0.12 s seed delay, and all sigma feel dials are unchanged. The plant parameters
remain provisional under continuous calibration from future casual routes.
Invalid `liveParameters` uses zero roll/angle offset and nominal vehicle
parameters for that frame, with `alignInputsValid = false`; no stale values are
carried.
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
That route also showed CPU 4 pinning doubling shadow inter-frame time from
9.5 ms to 19.0 ms. Route `000000b7--a6b3b1f175` then confirmed only about
32 shadow frames per second against controlsd's 100 Hz stream. The affinity is
therefore removed: shadowd retains `Priority.CTRL_LOW`/`SCHED_FIFO` but may roam
across cores 0-4. Core 5 is excluded because it carries equal-priority planning
work; cores 6 and 7 are excluded for camera/model workloads. Until a field
route demonstrates near-100 Hz throughput and materially improves the current
0.284 normalized-torque v14-shadow consistency RMS, logged v14 commands are
diagnostic only and never the A/B baseline.

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

- excitation module;
- UI;
- post-promotion feel tuning.

Promotion stays **in progress** until identity replay, delivered-curvature
gates, full test suites, device timing, and the first owner drive pass.
