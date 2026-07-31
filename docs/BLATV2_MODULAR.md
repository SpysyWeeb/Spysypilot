# BLaTv2 modular controller contract

## Purpose

BLaTv2 is a ground-up, vehicle-portable lateral controller whose only path
authority is the driving model. It uses the model's scalar action and future
trajectory, live vehicle state, learned vehicle response, and runtime actuator
limits to produce one torque request.

The design deliberately avoids named maneuver patches. There is no turn-in
boost, unwind controller, handoff state machine, authority restoration,
low-speed controller, or smoothness filter. If a physical module is wrong, it
is fixed or replaced at its contract boundary.

All numerical modules are shared libraries called synchronously from the
100 Hz control loop and from replay. They are not separate onroad processes.
This keeps one frame clock and makes device/replay parity testable.

## Sole-owner rule

Every physical decision has exactly one owner.

| Decision | Owner |
|---|---|
| Lane/path placement | model scalar action |
| Model-authored time | canonical time/intent adapter |
| Future path shape | stateless reference compiler |
| Curvature/rack and rack/torque physics | vehicle plant |
| Speed-dependent physical parameters | vehicle profile |
| Unconstrained torque request | computed-torque core |
| Future reachability | optional feasibility constraint |
| Fast persistent physical bias | disturbance observer |
| Slow parameter learning | offroad profile learner |
| Live torque magnitude/rate | one output envelope |
| Platform command and safety enforcement | opendbc and panda |
| Quality judgment | replay metrics and event logger |

A downstream module may not compensate for an upstream module by changing a
decision it does not own.

## Data flow

```text
model + car state + live calibration
                 |
                 v
      canonical frame/time context
                 |
          +------+------+
          |             |
          v             v
 stateless reference   vehicle profile
          |             |
          +------v------+
              plant
                 |
                 v
       computed-torque core
                 |
        optional feasibility
                 |
                 v
        one output envelope
                 |
                 v
      opendbc command + panda safety
```

The observer and learner consume measured response beside this path. Neither
may create a second torque command.

## Module contracts

### Canonical time and intent

This module creates one immutable frame context. It alone converts model,
measurement, and actuator clocks into:

- measurement time;
- plan publication time and age;
- scalar action effect time;
- plant effect time;
- current and predicted speed;
- current and predicted rack state;
- engagement, override, validity, and constraint state.

No other module may add `DT_MDL`, smoothing latency, preview time, or live
delay. Model smoothing is not controller timing. The scalar action timestamp
is an interface fact, not a feel dial.

The live clock contract uses four distinct facts:

- the rack-state sample time from `carState.logMonoTime`;
- the control witness captured at precompute entry;
- the model plan origin from `modelV2.timestampEof`; and
- the scalar action effect time published by modeld as
  `action.desiredCurvatureTime`.

`plan_time_now = control_witness - timestampEof`. The plant prediction
horizon is `state_sample_age + profile_transport_delay`. Neither expression
reconstructs model latency from `LAT_SMOOTH_SECONDS`, `DT_MDL`, or
`liveDelay`.

### Stateless future reference

The reference compiler answers what path the model authored, not how hard the
wheel should move:

```text
reference(t) = scalar + plan_curvature(t) - plan_curvature(action_time)
```

The exact invariant is:

```text
reference(action_time) == scalar
```

The scalar owns placement; the plan contributes only surrounding shape and
analytic motion information. The compiler has no state, filters, thresholds,
speed gains, torque output, or maneuver classification. Invalid or stale plan
data yields an explicit scalar-only reference, never a retained old plan.

### Vehicle profile

The profile is a versioned snapshot of physical quantities at ordered speed
nodes. Values interpolate continuously; extrapolation is explicit. Initially
it may contain:

- steady torque per lateral acceleration;
- rack response gain and damping;
- transport delay;
- tracking authority derived from physical uncertainty.

Samples support only neighboring nodes, so driving at one speed cannot change
distant speed behavior. The controller has no hard low/high-speed mode.

### Plant

One artifact owns forward prediction and inverse torque physics. Feedforward,
feedback prediction, feasibility, live control, and replay use the same units,
signs, roll convention, offset-corrected rack state, and profile.

The plant separates:

- steady road/tire load;
- commanded rack acceleration;
- rack damping;
- static and moving friction;
- transport delay.

Forward/inverse reciprocity and deterministic float64 results are contract
tests. A physical seed may be provisional; its uncertainty must be visible
rather than hidden by a controller patch.

### Computed-torque core

This is the only owner of the unconstrained request:

```text
raw torque =
    steady requested-path load
  + requested rack-motion torque
  + position-error correction
  + rate-error correction
  + bounded observed disturbance
```

All terms refer to the same physical effect time. The future plan supplies
desired position, rate, and acceleration at that time. It does not move the
path earlier. Each term is logged separately.

There is no integral in the core when a disturbance observer is enabled, and
there is no internal torque-rate smoothing. Smooth motion must result from a
continuous reference, coherent physics, and the real actuator envelope.

### Future feasibility

Feasibility is optional and initially pass-through. If enabled later, it may
only report or constrain what current requests keep the authored future state
reachable under the shared plant and output envelope. It cannot add torque,
change path timing, preserve old-direction curvature, or retain episode state.
Disabling it leaves a complete controller.

### Output envelope and opendbc

There is one live controller-side limiter. It reads maximum torque, build
rate, release rate, and command step from runtime `CarControllerParams`.
Sign changes use one explicit decay-then-build calculation. Nothing after this
boundary may reshape the normalized request.

opendbc owns the vehicle-specific declaration and enforcement of those limits.
The Palisade's validated limits are currently 409/4/7 in integer command units;
those numbers belong in its opendbc platform/safety configuration, never in
the generic BLaTv2 controller. Other vehicles retain their own limits.
Panda remains an independent safety ceiling, not an authority discovery
mechanism.

The explicit opendbc compatibility bit covers the complete production
envelope, including command cadence—not merely the presence of `STEER_MAX`
fields. An actuation-capable port must also publish its measured rack-rate
resolution from `CarControllerParams`; that sensor fact controls learning
qualification and stick/slip classification. A port without both declarations
may shadow and learn against an explicitly provisional seed, but it can never
select modular actuation. Adding the vehicle-owned capability changes the
runtime identity, so evidence collected under a provisional sensor quantum
cannot silently become an active profile. Stock bootstrap remains available
everywhere.

### Observer

The fast observer estimates one bounded physical disturbance from recorded
applied torque and measured rack response. It resets on engagement lifecycle
resets and invalid input, and freezes during driver override or actuator
constraint. The controller reads the estimate but cannot mutate it.

### Slow learner

No managed learner runs onroad. Durable training happens offroad when
`blatv2_backfilld` replays complete, closed full rlogs containing clean,
hands-off measured data. A recorded applied-torque transition on the
vehicle-owned magnitude or slew envelope is retained as authority evidence:
the emitted torque is known exactly and sharp-turn/breakaway response must not
be discarded. Slew transients do not enter the instantaneous plant equality.
Settled full-magnitude motion enters a separate fit stratum only after the
profile's transport delay, and candidates must not regress either the ordinary
or authority validation stratum. Until an authority stratum has at least four
held-out rows, it remains stored but cannot alter fitted parameters.
Driver-limited and unreachable transitions remain invalid. Each node tracks
ordinary and authority support, excitation, validation error, uncertainty,
and provenance. The live learner adapter is retained only for deterministic
offline/harness work and cannot contribute field evidence.

Initial minimum exposure guidelines are:

- low-speed maneuvers: 2–3 clean minutes with multiple meaningful turns and
  reversals;
- urban/mid-speed: 3–5 clean minutes;
- highway: 5–10 minutes of actual curved or steering-active data.

Elapsed time is only a floor. A node qualifies only with enough information
and held-out validation. A profile is promoted only when all required speed
regions qualify, never mid-drive.

Turn-in, release, overshoot, roughness, burst, driver feedback, and event
bookmarks supervise a proposed profile. They are not directly learned torque
gains.

### Display-only learning and lifecycle status

The current field UI reads two rebuildable JSON caches:

- `BLaTv2LearningOperationStatus` reports the current operation instead of
  treating an unfinished replay as an empty learner. `blatv2_backfilld` owns
  offroad `preparing`, `finalizing`, `backfilling`, route progress, terminal
  idle/empty results, and failures. Route identities are hashed before
  publication. No managed process publishes live collection state.
- `BLaTv2LearningStatus` is projected by `blatv2_backfilld` only after the
  exact evidence, optional candidate, ledger, and manifest have persisted.
  It reports cumulative node evidence, qualification reasons, and only the
  four parameters the regression actually fits. Seed-carried delay, static
  friction, and rack-rate resolution are never labeled learned.

Both keys are `CLEAR_ON_MANAGER_START` display caches and are never
persistent authorities themselves. They are published only by the offroad
process owner from authenticated current-build, current-vehicle authorities.
No cache may be consumed by fitting, candidate creation, approval, controller
selection, feedback, rollback, the live controller, or a safety path.
Missing, stale, malformed, deleted, or edited display data therefore cannot
change steering.

The retained `BLaTv2LifecycleStatus` schema and `blatv2_profiled`
implementation are offline lifecycle-test surfaces in the current stock-only
field graph. Manager does not launch that process, so it cannot stage an
active profile or infer provisional-drive feedback. Re-enabling that
lifecycle requires a separately reviewed offroad exercise witness; the UI
must never infer it from raw activation Params.

Evidence stays cumulative against its immutable physical fit seed. A
qualified candidate's revision is an opaque monotone function of persisted
accepted evidence, so later qualified candidates advance across restarts
without pretending the already-accumulated sufficient statistics were
collected against a newer learned seed.

Complete local routes from previous software runs may be imported, but only
from full rlogs. Qlogs, incomplete or still-open routes, dirty or unreviewed
build provenance, identity/envelope mismatches, corrupt logs, and
nondeterministic replays fail closed. The importer replays every eligible
batch twice in fresh runtimes, records exactly-once route provenance in an
authenticated ledger, and publishes an immutable generation by atomically
replacing one `CURRENT` pointer. A rejected route does not block a later
eligible route. An older route discovered after the durable chronology has
advanced is recorded as late and skipped.

Each released clean build needs an explicit reviewed descriptor retained in
`historical_build_descriptors.json` before it becomes historical. The current
build descriptor can be synthesized only while that build is running; an
older route without its pinned descriptor correctly fails closed as
unreviewed.

The current manager process graph has exactly one BLaTv2 process:
`blatv2_backfilld`, offroad on a real car only. It is the sole durable
evidence writer and publishes learning/operation status. Manager never starts
`blatv2_shadowd`, `blatv2_learnerd`, or `blatv2_profiled`; their sources and
tests remain offline/harness tools. Consequently, BLaTv2 contributes no
managed process load while started/onroad.

All numerical actuation remains synchronous inside controlsd so device and
replay import the same artifact and share one control-frame clock.

## Bootstrap and rollback

With no qualified profile, the exact stock torque controller actuates while
the modular controller is evaluated offline. The first candidate is trained
only after coverage is complete. It must pass:

1. deterministic unit and contract tests;
2. device/harness A/A parity;
3. raw, applied, and delivered replay gates;
4. invalid-frame and lifecycle tests;
5. runtime actuator-limit agreement with opendbc;
6. shadow validation on held-out data.

Activation occurs only at an engagement boundary. A provisional profile is
atomically versioned and rollback remains available. `Worse` feedback
deactivates it at the next engagement without deleting evidence.

## Replacement and tuning discipline

One iteration changes one module or one parameter family. All other artifacts
remain pinned. A failed module is corrected or replaced; a downstream
compensator is not added.

The acceptance report names:

- source and opendbc commit identities;
- profile identity and provenance;
- exact controller/runtime limits;
- module outputs and active constraints;
- test and replay results;
- accepted deviations or unavailable evidence;
- whether stock or modular control is active.

The complete staged gate and artifact-identity requirements live in
[`BLATV2_ACCEPTANCE.md`](BLATV2_ACCEPTANCE.md).

The branch stays **in progress** until the owner field-tests a qualified build
and explicitly authorizes a status change.
