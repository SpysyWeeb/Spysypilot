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

The active learning artifact is a versioned observable calibration at ordered
`0/5/10/15/20/30 m/s` speed nodes. Values interpolate continuously and hold
flat beyond the end nodes. Each node contains:

- normalized torque per measured lateral acceleration;
- a signed residual lateral-acceleration offset correction;
- normalized moving-friction torque;
- normalized static breakaway torque;
- seed transport delay and measured rack-rate resolution as metadata; and
- evidence, held-out error, confidence, and qualification state.

Rack response gain and damping are deliberately absent. Casual road data
proved unable to identify them independently: the retired fit pinned
parameters at physical bounds and produced sign-invalid inverse gains. A
provisional transient model may remain in experimental code, but it is not a
learned fact, does not enter calibration identity, and cannot be emitted by
the observable learner.

Samples support only neighboring nodes, so driving at one speed cannot change
distant speed behavior. The controller has no hard low/high-speed mode.

### Plant

One artifact owns forward prediction and inverse torque physics. Feedforward,
feedback prediction, feasibility, live control, and replay use the same units,
signs, roll convention, offset-corrected rack state, and profile.

Any future plant used by an actuating controller must separate:

- steady road/tire load;
- commanded rack acceleration;
- rack damping;
- static and moving friction;
- transport delay.

Forward/inverse reciprocity and deterministic float64 results are contract
tests. A transient seed may be provisional; its uncertainty must be visible
rather than hidden by a controller patch. This learning milestone does not
claim casual driving has calibrated transient rack dynamics.

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
hands-off measured data. At each speed node it fits the directly observable
inverse relation:

```text
applied torque =
    gain * (-measured lateral acceleration + offset correction)
  + kinetic friction * moving direction
  + static breakaway * breakaway direction
```

The learner solves this as deterministic constrained least squares with
`gain >= 0`, `kinetic >= 0`, and
`static = kinetic + nonnegative breakaway excess`. Training routes evaluate a
nested model family: static only, friction, offset plus friction, then the full
map. Every populated training category must beat or match the seed and at
least one must improve; richer models replace simpler models only by Pareto
dominance. The seed is a comparator, never a candidate. The selected model is
frozen before one held-route validation pass, with no post-validation fallback.

Stationary/quantized rows set both direction terms to zero and identify the
effective settled inverse map; they are not presented as a pure decomposition
of every tire force. Breakaway is reconstructed once per vehicle-global
episode. Raw measured steering-angle motion at half one rate quantum marks the
earliest possible onset after a continuous dwell; a same-direction measured
rate quantum must confirm it within the existing transport delay. The midpoint
of last-stuck and first-motion response identifies static friction. Unconfirmed
or sensor-disagreeing episodes do not enter equality fitting. Lifecycle gaps,
driver override, invalid input, disengagement, and standstill clear episode and
direction continuity.

A recorded applied-torque transition on the vehicle-owned magnitude or slew
envelope is retained as authority evidence: the emitted torque is known
exactly and sharp-turn response must not be discarded. Slew transients and
stationary full-magnitude rows remain observations rather than equality-fit
rows. Settled full-magnitude response may enter the separate authority fit
only with resolved rack motion.
The learner retains controls-witness, response, torque-report, and
torque-effective clocks independently. Because `carOutput` carries the prior
card-cycle result, its payload is effective at the preceding publication.
Response at `t` uses the newest exact zero-order-held torque effective no later
than `t - profile_transport_delay(speed)`; response physics and speed-node
weights stay at `t`. Settled full-magnitude motion enters a separate fit
stratum after one aligned response interval of command-side dwell, and
candidates must not regress either the ordinary or authority validation
stratum. Every populated authority stratum can veto a regression. Authority
data enters fitting only at four training rows, and fitted authority use then
requires four independent validation rows.
Driver-limited and unreachable transitions remain invalid. Each node tracks
disjoint base, moving, breakaway, and authority support; whole-route
training/validation by immutable route-counter parity; excitation in both
directions; validation error; confidence; and provenance. No maneuver can
contribute frames to both fit and validation. The live learner adapter is
retained only for deterministic offline/harness work and cannot contribute
field evidence.

The learner uses one vehicle-generic signed rack coordinate. Native negative
rate values prove a signed source. Otherwise the raw rate remains a magnitude
and offset-corrected measured steering-angle motion establishes and
revalidates direction without changing that magnitude. The observable fit
does not consume rack acceleration, so a valid physical reversal can remain a
moving row and direction-coverage event without importing the quantized
sign-crossing acceleration. A reversal clears prior dwell and cannot fabricate
breakaway. Exact zero clears unsigned-source sign inference. A lifecycle
break, source gap, driver override, standstill, fault, or failed rack mapping
clears cross-frame continuity.

Calibration profile schema v2, evidence schema v6, coordinator artifact
schema v5, learning-status schema v2, canonical join schema v2, and inclusion
namespace `complete_full_rlog_authority_v4` bind this contract. The retired
v1, v2, and v3 namespaces remain byte-untouched and are never migrated or restored
as v4; retained compatible full rlogs are replayed into an initially empty v4
ledger. The separate calibration runtime identity excludes provisional rack
gain/damping but remains bound to the actual vehicle, measured mapping,
opendbc torque calibration and limits, delay, and rack-rate resolution.

Initial minimum exposure guidelines are:

- low-speed maneuvers: 2–3 clean minutes with multiple meaningful turns and
  reversals;
- urban/mid-speed: 3–5 clean minutes;
- highway: 5–10 minutes of actual curved or steering-active data.

Elapsed time is only a floor. A node qualifies only with enough information
and held-out validation. The current milestone may emit an unapproved
candidate only when all required speed regions qualify; it has no promotion
or activation path and stock remains active. A future consumer would still be
required to promote a separately approved profile only at an engagement
boundary, never mid-drive.

Turn-in, release, overshoot, roughness, burst, driver feedback, and event
bookmarks supervise a proposed profile. They are not directly learned torque
gains.

#### Off-device preparation boundary

The optional LAN worker replaces only the immutable, computationally costly
full-rlog decode and canonical-join stage. It does not replace the learner.
The device remains the sole owner of route selection, causal learner state,
the two-authority equality decision, durable ledger extension, finalization,
and atomic generation publication.

Every protocol message uses a versioned canonical-JSON envelope authenticated
with a dedicated 256-bit pre-shared secret. Requests are timestamp bounded and
nonce replay protected across worker restarts. Job creation binds the exact
superproject, opendbc, panda, descriptor-registry, runtime, vehicle, dongle,
and CarParams identities. Uploaded route segments and returned spools are
size-bounded and content addressed. The PC produces one private spool for
each independent preparation authority; an authority never consumes the
other authority's prepared bytes.

The native extractor follows the same content-authority rule as the rlogs.
Preparation hashes an `O_NOFOLLOW`-opened executable descriptor, invokes that
exact descriptor via `/proc/self/fd`, and verifies the held inode and pathname
afterward. The learner pins one extractor SHA across its complete transaction,
so neither a launch-time pathname swap nor a binary change between routes can
produce mixed evidence or dishonest publication provenance.

Cross-architecture trust is established before prepared frames are consumed.
For every unseen accepted compatibility domain, one locally retained route is
prepared with the device ARM extractor and its entire deterministic spool must
match the x86 spool byte-for-byte. The device stores a private, atomic,
HMAC-bound certificate tied to the two extractor binaries, worker process
instance, build/runtime/registry identities, decode schemas, CarParams, and
physical compatibility; immutable route hashes and selected-stream identity
remain the recorded test vector. Rejections get a route-specific certificate
only when ARM independently produces the same stable reason and message.
Archive-only unseen domains fall back local because protocol v1 has no rlog
download path.

The device projects remote progress onto its own monotonic clock and existing
display-only status. PC progress is never evidence. Loss of discovery or a
transient connection cancels the remote attempt and selects the unchanged
local preparation backend. Authentication, compatibility, content, or spool
validation failures are not downgraded to success. An onroad transition or
manager stop cancels either backend before publication.

One protocol-v1 job is limited to 128 selected routes; no batching is allowed
inside one atomic learner transaction, so a larger set uses the complete local
backend. The archive inventory uses authenticated, strictly monotonic
exclusive-cursor pages of at most 128 routes, so the job bound never truncates
the durable archive. The device resumably uploads every complete quiescent
local route missing from that archive, prioritizing active-job routes; archive
sync is transport only and has no selection, ledger, or publication authority.
Each uploaded segment remains in private worker staging. An explicit commit
binds and re-hashes the exact ordered segment manifest before atomically making
the complete route inventory-visible, so an interrupted prefix is always
resumable and never masquerades as a short complete route.
The device never sends cleanup traffic after an onroad transition.
The PC instead leases running work to authenticated status polling and expires
an abandoned job after 30 seconds. Ledger-only late-route publication names
the prior generation's actual extractor rather than either unused current
binary.

The worker's archive and implementation live outside the openpilot checkout
at `/home/alex/Documents/blatv2-remote-worker`, with full rlogs under
`data/routes/`. This keeps recordings out of temporary storage and prevents
source updates from pruning the archive.

### Display-only learning and lifecycle status

The current field UI reads three rebuildable JSON caches:

- `BLaTv2LearningOperationStatus` reports the current operation instead of
  treating an unfinished replay as an empty learner. `blatv2_backfilld` owns
  offroad `preparing`, `finalizing`, `backfilling`, route progress, terminal
  idle/empty results, and failures. Route identities are hashed before
  publication. No managed process publishes live collection state.
- `BLaTv2BackfillProgress` optionally adds pass, route, segment, read/apply
  phase, cumulative work, and an approximate remaining time. It is bound to
  the operation id and sequence so a torn two-key read falls back to the
  coarse status. Work spans reading and route application across both replay
  passes; timing never enters evidence, ledgers, replay ordering, A/A
  comparison, fitting, or actuation.
- `BLaTv2LearningStatus` is projected by `blatv2_backfilld` only after the
  exact evidence, optional candidate, ledger, and manifest have persisted.
  Schema 2 reports cumulative and last-drive base, moving, breakaway, and
  authority populations, qualification reasons, held-out errors, and only
  the four parameters the observable regression fits. Seed-carried delay and
  rack-rate resolution are never labeled learned; retired rack-gain/damping
  fields are rejected rather than compatibility-mapped.

All display keys are `CLEAR_ON_MANAGER_START` caches and are never
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

Evidence stays cumulative against its immutable observable calibration seed. A
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

The two required deterministic replay passes run concurrently in separate
forked Linux processes with independent replay state. Within each pass,
segment extraction, decoding,
source-history joining, frame construction, hash folding, and learner
ingestion remain strictly serial in canonical route/segment/frame order. The
workers share no prepared frames or learner state and have no durable-writer
authority. The parent process alone compares their `ReplayPass` artifacts,
extends the ledger, and performs the atomic publication. This preserves the
existing progress schema: it projects the primary replay, then verification,
without exposing worker scheduling. Production is fixed at two workers. A
four-worker design is queued only after two-worker device elapsed-time,
process-group peak-memory, storage-contention, thermal, and responsiveness
validation. It requires bounded per-route spooling so additional preparation
workers cannot reorder learning or multiply unbounded in-memory routes. Each
A/A authority must still prepare its own inputs; sharing one prepared spool
between the two passes would weaken the independence check.

Desktop reference measurement on the 21-segment
`000000b7--a6b3b1f175` route: two serial passes took 17.594 s and the
two-process path took 9.717 s (1.81x), with byte-identical evidence,
manifest, and ledger entries. On-device elapsed time, process-group RSS, I/O
contention, thermals, and responsiveness remain the deployment authority.

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

Any future modular actuation must remain synchronous inside controlsd so
device and replay import the same artifact and share one control-frame clock.
The current controlsd selection remains stock.

## Future bootstrap and rollback contract

The exact stock torque controller actuates throughout this learning milestone.
An all-node calibration is emitted only as an unapproved offline candidate.
Before a future controller consumer could use it, the exact candidate would
have to pass:

1. deterministic unit and contract tests;
2. device/harness A/A parity;
3. raw, applied, and delivered replay gates;
4. invalid-frame and lifecycle tests;
5. runtime actuator-limit agreement with opendbc;
6. shadow validation on held-out data.

Under that future contract, activation could occur only at an engagement
boundary. A provisional profile would be atomically versioned and rollback
would remain available. `Worse` feedback would deactivate it at the next
engagement without deleting evidence.

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
