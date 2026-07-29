# BLaTv2 — Better Lateral Tune v2

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot),
created from the untouched `stock` branch. See
[`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) for the full
fork overview.

## Status

⚠️ **In progress — v215 desired-load action controller.** Route bc exposed a
structural timing error in v207: the learned end-to-end lateral lag was also
used as the rack twin's pure command-transport delay. That advanced the
predicted rack response twice and could make valid feedforward and feedback
cancel. Version 208 keeps the model-authored action time unchanged while
predicting the measured rack through only the plant seed's physical delay.
Version 209 also removes the remaining feedforward/feedback conflict: the
steady aligning-load feedforward is evaluated at the model-requested rack
position, rather than at the lagging measured position whose old-direction
load could cancel entry or reversal feedback.

Version 214 tested separating closed-loop bandwidth from the plant twin's
provisional damping estimate. It exposed another conflation: one bandwidth
raised useful position authority and noisy rate feedback together. At
`20 1/s`, 9f complaint-band oscillation rose above v14 (`0.0315` versus
`0.0284`) and bc applied roughness also exceeded v14. Version 215 replaces the
coupled pole dial with an explicit position stiffness. Rate damping is derived
independently so one 4 deg/s sensor quantum can change feedback by no more than
one Hyundai torque build step. This preserves continuous path-error authority
without amplifying the quantized rate signal into another correction module.

A v210 experiment decoupled breakaway from `sigma_curvature` and armed it from
the measured 0.1-degree rack-angle resolution. It is reverted in v211: on 9f
it improved the direct-handoff delivered fraction but armed on 43% of valid
frames and raised 15.6–20.1 m/s command oscillation from `0.0145` to `0.0474`,
worse than v14 on that route. Repeated full-static-friction pulses are not the
smooth solution to a small-correction deadband.

V212 added a one-breakout-per-error-direction latch. It also remains rejected:
the 9f complaint-band oscillation improved from v210's `0.0474` only to
`0.0413`, still worse than v14's `0.0284` and v209's `0.0145`. Static
breakaway therefore remains reserved for large, persistent stiction events;
small continuous path error belongs to proportional feedback, not repeated or
latched friction compensation.

Route bb showed that v206's inverse target was strong but its one-frame slew
projection delivered sharp-turn torque late, while continuous static-friction
compensation caused near-center activity. Version 207 tried event-based
breakaway plus a horizon-reachability helper. Route bc proved the helper was
not a controller: it affected only four of 96,489 valid active replay frames,
while the real under-delivery was the local inverse's old-position load.
Version 209 deletes that second torque authority and retains event breakaway
only for the physical stiction it represents.

Field routes b9/ba
showed that v203 was smooth but slow and weak. Version 205 replaced the LQI
cost trade with a direct inverse-rack action controller inside controlsd at its
existing CTRL_HIGH 100 Hz position. The model scalar action remains the
unmodified path authority: v205 does not move the path earlier, preserve a
turn longer, or sample farther into the plan to create lead.

Version 206 removes a branch inconsistency inherited from the stock base:
modeld's lateral output smoothing is explicitly zero, matching combo, and
BLaTv2's fixed scalar-action timestamp no longer imports or depends on that
filter setting. A future smoothing change therefore cannot silently alter the
controller's reference timing.

The controller predicts the measured rack only through the physical command
transport delay, then uses one inverse-rack equation to combine the load at
the model-requested wheel position, the model-authored rate/acceleration, and
measured tracking error. Far-future curvature never pulls the current command
ahead of the scalar action. Tracking is not traded against a torque-rate cost
or an artificial arrival time. The same numerical files are imported by
route-audit; there is no separate live variant.

The MPC is retired from onroad execution. Its implementation and tests remain
in the repository so it can re-enter a future tournament after its real-world
schedule-construction cost is fixed.

### What LQI means

**LQI** means **Linear–Quadratic control with Integral action**. It predicts
how the steering rack responds, then chooses feedback that balances two costs:
tracking the requested path and avoiding excessive control effort. The
integral state gradually corrects a persistent tracking bias.

BLaTv2 versions through v203 used that architecture. It was smooth, but its
path-error-versus-torque trade made it reluctant to spend torque at low speed;
the wheel could remain far behind the requested angle while substantial EPS
authority went unused. LQI is therefore historical in v205. The active v205
controller has no LQI cost matrix and no integral state: it directly inverts
the physical rack dynamics for the model scalar action.

The frozen v14 implementation remains on
[`BLaT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLaT). BLaTv2 is a
ground-up design and does not inherit that controller.

## Current scope

- a deterministic float64 steering-rack plant twin;
- a stateless scalar-anchored future-path reference;
- one shared recorded-response disturbance observer;
- a retained, non-running sign-schedule torque MPC challenger;
- the promoted delay-compensated inverse-EPS action controller;
- an onroad `blatv2_shadowd` process that runs the complete frozen-v14
  controller passively for honest live A/B telemetry;
- route-audit replay using the identical library implementation.

The shadow event is `blatV2Shadow`, version 11. It reports reference and
torque-demand values, actuator-feasible torque, one-step plant residual,
scalar/plan disagreement, horizon, vehicle speed, the self-aligning torque
estimate, alignment-input validity, overall validity, and per-frame runtime.
Version 11 separates the logged model action time from the physical rack
prediction delay. Version 10 added event-breakaway state and the now-retired
horizon-helper diagnostics; their wire ordinals remain reserved and v209
publishes them as inactive zeros.
Version 9 adds the live v205 torque decomposition, action-point target, and
delay-predicted rack state. Version 8 uses the live-field-calibrated plant
schedule described below.
Version 7 removed both tournament candidates from shadowd. It copies the live
LQI command, status, recovery state, and in-controlsd compute time from
`controlsState`, and logs the command from the complete frozen-v14 pipeline.
The prior response-surface controller reports version `203`. Version 203 keeps
plant stiction and the observer clamp at the measured `0.09` while replacing
the moving-friction feedforward magnitude with the b7-selected `0.03`; full
`0.09` breakaway remains available when departing stiction. It adds the
curvature-space tracking tolerance `sigma_curvature = 0.00091683 1/m` while
leaving the three original sigma dials unchanged.

### v215 timing and authority contract

The action time is `liveDelay.lateralDelay + 1.5 × DT_MDL`. The fixed model
offset is independent of `LAT_SMOOTH_SECONDS`; model filtering cannot silently
move BLaTv2's reference. The scalar action and its local
angle/rate/acceleration stencil are sampled at that one time. Changing any plan
point beyond the local stencil cannot move that target or the current torque
request.

`liveDelay` is the measured desired-curvature-to-yaw lag and selects the model
reference; it is not a pure actuator transport delay. The controller advances
the measured steering-wheel angle/rate through only the independent physical
plant-seed delay (`0.12 s` provisionally) while holding measured applied
torque. `blatV2ActionTimeSeconds` and
`blatV2PredictionDelaySeconds` log both clocks independently so this contract
is per-frame auditable. The inverse combines the calibrated steady load at the
requested rack position with desired motion and direct position/rate feedback.
Plant damping `b_steer` remains only a physical rack-model parameter.
`tracking_stiffness` determines how strongly the controller removes measured
position error. Rate damping is not another feel dial: it is derived from the
runtime torque build step and the measured 4 deg/s rack-rate quantum.
Evaluating the load at the old measured position was the v207 conflict: during
turn entry or reversal it preserved old-direction torque while feedback tried
to leave it. The requested-position load removes that cancellation; the
independent stiffness supplies continuous authority without making rate
quantization authoritative.

The scalar action alone sets steering position. A fixed five-point quadratic
stencil derives coherent rate and acceleration from the native 50 ms model
grid; it adds no state, smoothing delay, or feel constant. The rest of the
far-future path remains available to the shared reference and harness, but it
has no independent torque authority. This is deliberate: for a second-order
rack, position, rate, and acceleration at the model action point are the
complete motion command. Letting later curvature pull current torque would
change path timing rather than reduce controller delay.

The inverse has no speed-scheduled feedback gain, torque-motion cost, arrival
horizon, or integral. `tracking_stiffness` is one continuous physical response
dial, not a smoothing or boost schedule. Smoothness is the exact 409/4/7
actuator trajectory. Low-speed authority arises naturally because the same
curvature requires much more steering-wheel angle at low speed, so the
physical angle error asks for more torque and may saturate.

Palisade steering rate is quantized in 4 deg/s increments. Version 207 does not
treat every zero-rate frame as stiction. Static breakaway arms only after one
model period of persistent same-direction curvature error at least as large as
the existing `sigma_curvature`, while measured rack rate is inside half a
sensor quantum. During that physical breakout episode it blends continuously
from static `0.09` to kinetic `0.03` across one full measured-rate quantum.
Sub-threshold center noise receives no static compensation. The persistence
period and rate quantum come from existing model timing and sensor resolution;
neither is a feel dial. A b9/ba attempt to identify an additional
road-wheel-angle scrub term was rejected: the two low-speed route fits did not
cross-validate, so v205 adds no unsupported coefficient.

### v207 route-bb staging record

Version 207 remains **in progress** pending field testing. The four frozen
routes pass every raw and applied regression metric against v14, and their A/A
replay is bit-exact. On route bb, hands-off controller evidence is encouraging:
global applied torque-rate RMS is `0.798/s` versus v14's `0.971/s`, and command
oscillation in the reported 15.6–20.1 m/s band is `0.0184` versus `0.0727`.
Sharp low-speed events reach the full raw torque target where the physical
need calls for it.

The route-bb delivered table is retained as an unresolved diagnostic, not
relabeled as a pass. Its worst one-second burst includes the replayed
engaged-to-inactive reset from `0.724` torque at a recorded driver takeover,
and its worst release event contains driver steering beginning 1.43 seconds
before the scalar peak. Counterfactual state after that intervention is not a
candidate-only trajectory. Truncating evidence at the first intervention
leaves seven clean turn-in windows, below the committed minimum of ten.
Consequently bb cannot close the delivered gate; the first v207 drive must
measure release timing and authority directly.

### v205 pre-field acceptance

The clean feature artifact at `14cac1022f` passes 48 BLaTv2 unit tests plus
12 parameterized subtests. A 6,000-frame 9d shadow A/A replay is bit-exact for
all deterministic fields. A 20,000-frame workstation diagnostic measured
0.134 ms median and 0.178 ms p99 for shared preparation plus the active solve;
only device-logged in-controlsd timing can satisfy the field timing gate.

The four-route counterfactual replay deliberately records three open
pre-field deviations instead of relabeling them as passes:

- the unconstrained internal target is substantially rougher than v14
  (12.370/s versus 2.315/s RMS), but it is not the command controlsd sends;
- the emitted/post-slew trajectory is modestly more active than v14
  (1.091/s versus 1.029/s RMS; worst one-second 1.423/s versus 1.329/s);
- delivered turn-in is 453.5 ms versus v14's 302.9 ms on the frozen set.
  V14's preview mechanisms include early/corner-cutting events; v205 refuses
  to move the scalar position target. The first v205 drive must determine
  whether the remaining replay lag is real with the controller observing its
  own rack response.

The newest field routes are diagnostic rather than formal gates because each
contains only nine `lat.turnStopTurn` events. Replaying v205 against their
recorded inputs shows the intended authority change: on b9, the internal target
is at or above 0.95 on 68.3% of sharp low-speed frames and the emitted command
is at or above 0.95 on 39.8%; on ba the corresponding figures are 89.7% and
15.6%. Median emitted torque in those sharp sets is 0.780 and 0.675. This is
the field-test reason to stage v205: it is no longer leaving large torque
headroom unused in the turns that convicted v203.

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

An active non-OK or non-finite action-controller result holds the previous request for one
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
frames. Re-entry resets the controller and resumes through the same asymmetric
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
constrained. The retired fallback integral obeys the same reset/freeze
lifecycle. The v205 action controller has no integral state; it consumes the
observer estimate read-only.

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
- unsupported scrub/load coefficients.

Version 207 stays **in progress** until identity replay, route gates, full test
suites, device timing, and the first owner drive pass.
