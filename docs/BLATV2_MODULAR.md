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

The current learner is PC-only. The device records ordinary loggerd full rlogs
and runs no BLaTv2 learner, historical replay, route uploader, Wi-Fi bridge, or
local-processing fallback. An operator copies closed routes over read-only SSH
into durable PC storage. The PC owns deterministic A/A replay, physical and
behavioral qualification, and informational candidate generation. A candidate
has no approved activation path until separate manual review, installation,
and the remaining controller gates authorize that exact artifact. The
Palisade-only development parameter is a distinct owner-trial path for the
bundled provisional tune; it leaves every approval gate intact.

The present field build actuates this modular core only when that development
parameter was enabled offroad and the Palisade/runtime checks pass. Otherwise
controlsd constructs and runs the exact stock openpilot torque controller, not
a stock-shaped approximation or a modular controller with stock values. The
retired LQI design is also absent. **LQI** means
**Linear Quadratic Integral**: state feedback chosen from a mathematical
tracking/effort cost with an accumulated integral-error state. It is documented
only to make clear that neither its state nor its gain schedules are inherited.

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
| Slow parameter learning | PC offline profile learner |
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
At runtime the two neighboring node values are linearly interpolated; below
0 m/s and above 30 m/s the endpoint value is held. Interpolation is itself
validated on exact joint sufficient statistics. Two individually qualified
endpoints therefore cannot hide a bad mixed response between them.

Support is weighted clean response time rather than elapsed drive time:

| Nodes | Minimum clean support per node | Typical evidence source |
| --- | ---: | --- |
| 0 and 5 m/s | 150 s | low-speed turns, reversals, and breakaway episodes |
| 10 and 15 m/s | 240 s | urban curves and steering-active travel |
| 20 and 30 m/s | 420 s | genuinely steering-active higher-speed travel |

Every node also needs at least 20% held-out support, bidirectional lateral
acceleration and torque excitation, and moving plus complete breakaway
populations on both training and validation routes. Moving and breakaway each
need at least four training and four validation rows. The low-speed floor is
shorter because a sharp turn carries dense physical evidence, but waiting at a
light, driver override, invalid input, or straight motion without excitation
does not count. Thus “drive longer” is not a universal cure: missing support
needs more eligible time, missing breakaway needs more stuck-to-moving events,
and rank/conditioning problems need more varied excitation.

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

This runtime lookup is what makes the controller portable. The detected
CarParams selects the vehicle and opendbc supplies that port's actual maximum,
build/release rates, cadence, and driver-torque rules. Hardcoding the Palisade's
409/4/7 values would cause replay and control to disagree on another car and
would bypass the port review that proves the interface and panda share the same
contract.

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

No managed learner runs on the device. Durable training happens in an
operator-controlled PC workspace after complete, closed full rlogs are copied
from the comma over read-only SSH. The PC replays clean, hands-off measured
data twice with independent deterministic authorities before publication. At
each speed node the learner fits the directly observable
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
map. The seed is a first-class selectable result. A learned fit replaces it
only when paired whole-route loss clears route-level uncertainty without
regressing a populated category; otherwise the node is explicitly qualified
as `seed_retained`. Rank deficiency, ill conditioning, inconclusive
validation, and true regression remain different outcomes. The selected model
is frozen before one held-route validation pass, with no post-validation fallback.

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

The committed storage/wire identities are:

| Contract | Current value |
| --- | --- |
| calibration profile / evidence / coordinator | 2 / 9 / 9 |
| runtime vehicle / calibration identity / provisional dynamics | 1 / 1 / 1 |
| physical learning / operation / progress status | 4 / 1 / 1 |
| native extractor / canonical join | 5 / 5 |
| physical-frame encoding | 2 |
| route evidence | `BLATRE04`, version 4 |
| backfill ledger / commit / pointer | 3 / 2 / 1 |
| physical inclusion namespace | `complete_full_rlog_authority_v8` |
| controller policy | 1 |
| behavior gate spec / segmentation / replay input | 3 / 1 / 1 |
| behavior transaction / finalization | 2 / 1 |
| behavior generation / pointer / route-set | 1 / 1 / 1 |
| behavior learning status | 1 |
| future feedback / lifecycle status | 2 / 2 |
| future approved artifact / calibration selection / activation state | 5 / 2 / 1 |
| off-device protocol / cross-architecture certificate | 2 / 5 |
| off-device display progress | 2 |

These values are code contracts, not display labels. The retired physical v1
through v7 namespaces remain byte-untouched and are never migrated or restored
as v8; retained compatible full rlogs are replayed into an initially empty v8
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

### Shared route evidence and two qualification stages

One raw route is not decoded independently by every learner. Each of the two
A/A preparation authorities decodes and canonical-joins it exactly once into a
content-addressed `BLATRE02` version-2 artifact. That artifact stores the
physical measured-frame plane once and compact model, controls, live-torque,
delay, maneuver, and event planes beside it. The old physical-only `BLATSP01`
format is rejected. The authorities have separate extractor instances and
scratch artifacts; only byte-identical results may become durable. Physical
learning consumes the physical plane, while later behavioral replay reloads
the already authenticated shared artifact instead of parsing the rlog again.

Physical calibration is stage one. It publishes a selected profile only after
all six nodes and all five interpolation intervals qualify. A selected profile
can be entirely **seed retained**: that is a successful demonstration that the
recorded data did not justify changing the known physical map. A separate
learned candidate exists only when at least one node safely changes. Physical
publication is complete before behavior starts; a later behavior failure cannot
rewrite the physical route ledger or relabel its generation.

Behavioral qualification is stage two. Turn-in, release, overshoot, roughness,
burst, completion, correction latency, and delivered-path measurements are
scored from the shared route evidence. The learner may alter only two global,
speed-independent policy values:

- **natural frequency**: how quickly/stiffly the closed loop closes tracking
  error; and
- **damping ratio**: how the response settles without ringing or overshoot.

The provisional owner-trial center is `10.5 1/s` and damping ratio `1.0`;
it is explicitly unelected. The committed candidate bounds are
`5.0–15.0 1/s` and damping ratio `0.6–1.5`. These are behavior-search bounds,
not speed gates or vehicle torque values.

The explicit Palisade inverse-rack owner trial has one source-bound exception:
natural frequency is `11.0, 11.0, 10.5, 10.25, 10.0, 10.0 1/s` at
`0, 5, 10, 15, 20, 30 m/s`, with linear interpolation and flat endpoints.
Damping remains `1.0`. This unqualified reactive schedule is tied to the exact
source commit and cannot enter the global behavior search or approved preview
artifact path.

It cannot reinterpret the model path, vehicle calibration, observer, or
actuator envelope. The target is always the model's scalar-anchored reference;
lane-line estimates never enter it. Auto-logger events may locate maneuvers but
are not quality labels. Driver contact censors response after contact, because
the human then owns the trajectory, but the intervention itself is neither a
positive nor negative vote.

The behavioral population is the newest contiguous route cohort with one exact
recorded controller/source identity. A rejected, missing, corrupt, or
behavior-ineligible route interleaved in that cohort blocks qualification
instead of being skipped. An exact source boundary may end the cohort; routes
from different stock/modular builds are never pooled. Explicit
`late_older_skipped` entries are outside the append-only population and remain
ignored.

The committed partition keeps two validation routes, and the paired uncertainty
rule requires at least two routes in both partitions. The resulting minimum is
four homogeneous routes: two training and two held out. It is a floor, not a
promise of qualification. Every metric additionally needs its committed speed/
maneuver strata, at least two routes, and at least three windows, so rare
low-speed turns and genuine high-speed curves can require more routes.

Training evaluates exact stock, the currently accepted artifact, and the full
candidate grid against the same frozen segmentation. With no accepted modular
artifact, exact stock occupies both baseline roles and the provisional policy
is merely the candidate-grid center. One training winner is frozen before
held-out data is opened. Validation replays only exact stock, the incumbent,
and that winner; it cannot select a fallback after seeing the holdout.
Smooth, Swift, and Strong pass as separate contracts, and the path-error target
must materially improve beyond the observed whole-route paired uncertainty.
No weighted score can buy Swift or Strong by sacrificing Smooth, or vice versa.

The complete behavioral transaction is then rerun from independently reloaded
route artifacts and fresh replay cores. Only byte-identical A/A transaction
documents may publish. A passing result stores an informational policy; every
other safe result explicitly stores **stock retained** with no policy file.
Neither disposition is an activation decision.

### Immutable publication and `CURRENT`

Physical and behavior results have separate immutable generation stores. In
each store, all payload files are fsynced into a content-addressed generation
directory before one small `CURRENT` pointer is atomically replaced. Readers
authenticate the pointer, exact file inventory, every file hash, cross-file
identity, route set, source commits, physical profile, and configuration. A
generation whose bytes already exist must match byte-for-byte. An invalid
existing `CURRENT` is a fail-closed corruption report; publication never
overwrites it as though learning were starting fresh.

The physical `CURRENT` owns the route ledger, evidence, selected physical
profile, and optional changed candidate. Behavior uses its own
`behavior_generations_v1/CURRENT`, bound to that exact physical generation and
profile. A new physical generation, route cohort, source identity, config, core
identity, or incumbent policy requires a new behavioral transaction. Old
generations are deliberately retained because a safe reader-lifetime garbage
collection contract does not yet exist.

`BLaTv2LearningStatus`, `BLaTv2BehaviorLearningStatus`, and progress Params are
rebuildable views of these stores. Deleting or editing a display cache cannot
change `CURRENT`, select a policy, or affect actuation.

#### Retired device/PC preparation boundary (historical)

The automatic LAN worker and device-local fallback are retired. This section is
retained only to document old protocol/schema identities and generation
provenance; it does not describe the current manager graph. The optional LAN
worker replaced only the immutable, computationally costly
full-rlog decode and canonical-join stage. It does not replace the learner.
The device remained the sole owner of route selection, causal learner state,
the two-authority equality decision, durable ledger extension, finalization,
and atomic generation publication.

Every protocol message uses a versioned canonical-JSON envelope authenticated
with a dedicated 256-bit pre-shared secret. Requests are timestamp bounded and
nonce replay protected across worker restarts. Job creation binds the exact
superproject, opendbc, panda, descriptor-registry, runtime, vehicle, dongle,
and CarParams identities. Uploaded route segments and returned `BLATRE02`
route-evidence artifacts are size-bounded and content addressed. Each artifact
contains the exact physical frames once plus the compact behavior planes. The
PC produces one private artifact for each independent preparation authority;
an authority never consumes the other authority's prepared bytes.

The transport normally discovers the worker with a signed UDP broadcast. A
protected optional `worker_host.txt` beside the device secret supplies one
private IPv4 unicast target on mixed networks that suppress broadcasts. A
configured target is authoritative (responses from any other source fail
closed), but it changes no authentication or compatibility rule; lack of a
response remains a clean fallback to local preparation.

The native extractor follows the same content-authority rule as the rlogs.
Preparation hashes an `O_NOFOLLOW`-opened executable descriptor, invokes that
exact descriptor via `/proc/self/fd`, and verifies the held inode and pathname
afterward. The learner pins one extractor SHA across its complete transaction,
so neither a launch-time pathname swap nor a binary change between routes can
produce mixed evidence or dishonest publication provenance.

Cross-architecture trust is established before prepared frames are consumed.
For every unseen accepted compatibility domain, one locally retained route is
prepared with the device ARM extractor and its entire deterministic artifact
must match the x86 artifact byte-for-byte. The device stores a private, atomic,
HMAC-bound certificate tied to the two extractor binaries, immutable worker
implementation, build/runtime/registry identities, decode schemas, the
complete route runtime-vehicle bundle identity, and the validated
physical-vehicle projection. Full CarParams bytes remain verified
and content-addressed in each route artifact, but do not split a physical
domain merely because unrelated recorded fields changed. The authenticated
worker instance remains a per-job transport check and an otherwise identical
service restart reuses the numerical certificate. Immutable route hashes and
selected-stream identity remain the recorded test vector. Rejections get a route-specific certificate
only when ARM independently produces the same stable reason and message. A PC
rejection for an archive-only route is therefore not a device-authoritative
rejection: after the complete original two-authority outcome set agrees, that
route is removed from effective discovery and both authority inputs. It
contributes no ledger entry, watermark movement, learner evidence, readiness
count, or behavior-cohort vote. The UI may report its count as an unverified
exclusion. An archive-only accepted route may use a locally certified route
with the same parser/schema and physical numerical projection even when its
full CarParams artifact differs. A genuinely new archive-only compatibility
domain with no locally retained test vector still falls back local because
protocol v1 has no rlog download path.

The device projects remote progress onto its own monotonic clock in the
separate display-only `BLaTv2OffdeviceProgress` schema. It identifies PC
processing, bounded artifact download, ARM certification, prepared-data
handoff, and local fallback with a stable reason code. The ordinary
`BLaTv2BackfillProgress` remains the authority for local replay/application;
after handoff it retakes the display. PC progress is never evidence. Loss of
discovery or a transient connection cancels the remote attempt and selects the
unchanged local preparation backend. Authentication, compatibility, content,
or artifact validation failures are not downgraded to success. An onroad
transition or manager stop cancels either backend before publication.

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

### Retired device learning and lifecycle status (historical)

The former field UI read four rebuildable JSON caches. Their schema ordinals
and Params names remain reserved, but the current manager publishes none of
them:

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
  Schema 3 reports cumulative and last-drive base, moving, breakaway, and
  authority populations; explicit learned/seed-retained/evidence/rank/
  conditioning/validation outcomes; paired route-loss uncertainty; and
  interpolation qualification. Candidate availability is separate from a
  successful all-seed evaluation. Seed-carried delay and rack-rate resolution
  are never labeled learned; retired rack-gain/damping fields are rejected
  rather than compatibility-mapped.
- `BLaTv2BehaviorLearningStatus` is a schema-1 projection of the independent
  behavior stage. It reports waiting for a qualified physical profile or four
  homogeneous routes, then preparing, training, selecting, held-out
  validating, publishing, complete, or failed. Route/candidate/replay counts,
  source/profile hashes, and separate Smooth/Swift/Strong verdicts make the
  work legible. A complete result distinguishes `stock_retained` from
  `qualified_candidate_available`; neither means active.

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

### PC replay and publication

Evidence stays cumulative against its immutable observable calibration seed. A
qualified candidate's revision is an opaque monotone function of persisted
accepted evidence, so later qualified candidates advance across restarts
without pretending the already-accumulated sufficient statistics were
collected against a newer learned seed.

Complete archived routes from previous software runs may be imported, but only
from full rlogs. Qlogs, incomplete or still-open routes, dirty or unreviewed
build provenance, identity/envelope mismatches, corrupt logs, and
nondeterministic replays fail closed. The importer replays every eligible
batch twice in fresh runtimes, records exactly-once route provenance in an
authenticated ledger, and publishes an immutable generation by atomically
replacing one `CURRENT` pointer. A rejected route does not block a later
eligible route. An older route discovered after the durable chronology has
advanced is recorded as late and skipped.

Physical replay uses four worker lanes while preserving exactly two
independent A/A authorities. Each authority owns its causal learner state and
one private preparation helper. The helper may decode the next route into a
bounded `BLATRE02` artifact while its owner applies the current route, but
route application and learner ingestion remain serial in canonical order.
No authority consumes the other's route artifact, and no helper receives
ledger, Params, or publication access. The parent alone compares complete
results, extends the ledger, and atomically publishes. Worker counts 1 and 2
remain deterministic diagnostic modes; 3 is rejected because it would make the
two authorities asymmetric.

Behavior transactions independently use up to four fork workers across
whole-route/controller replay jobs. Results are folded in canonical job order,
so worker count cannot change transaction bytes. The complete behavior
transaction then runs a second time from freshly reloaded route artifacts and
fresh decoder/controller cores. This parallelizes pure replay without sharing
state between the A/A authorities.

The retired progress schema projected primary preparation/application and then
verification; helper scheduling was not exposed as extra passes. One
prefetched route per authority bounds live work, with at most two 512 MiB
artifacts per authority in the conservative worst case. Four-worker physical
processing previously completed on comma hardware without changing
deterministic artifacts; current production replay runs on the PC.

Desktop reference measurement on the 21-segment
`000000b7--a6b3b1f175` route: two serial passes took 17.594 s and the
two-process path took 9.717 s (1.81x), with byte-identical evidence,
manifest, and ledger entries. The integrated native-extractor/A/A/publication
benchmark across b7, b8, b9, and ca measured a 33.754 s four-lane median versus
42.359 s with two lanes, with identical hashes. These measurements are retained
as historical determinism and resource evidence, not as a device deployment
target.

Each released clean build needs an explicit reviewed descriptor retained in
`historical_build_descriptors.json` before it becomes historical. The current
build descriptor can be synthesized only while that build is running; an
older route without its pinned descriptor correctly fails closed as
unreviewed.

The current manager process graph has zero BLaTv2 background processes.
Learning, replay, transfer, bridge, shadow, and lifecycle entrypoints are not
registered. Retained numerical libraries and wire/schema identities are
offline compatibility surfaces. Consequently, BLaTv2 contributes no managed
process load onroad or offroad.

Modular actuation remains synchronous inside controlsd so device and replay
share one control-frame clock. It occurs only through the explicit provisional
trial or a future approved artifact; otherwise controlsd selects stock.

## Approved bootstrap and rollback contract

The exact stock torque controller remains the default throughout this learning
milestone; the explicit owner trial is an unapproved provisional exception. A
complete all-node physical selection and any qualified behavior policy are
emitted only as authenticated offline artifacts. An all-seed physical result
or failed behavior gate explicitly retains stock and emits no redundant policy.
Before an approved consumer could use a selected physical profile and qualified
behavior policy, that exact composed candidate would have to pass:

1. deterministic unit and contract tests;
2. device/harness A/A parity;
3. raw, applied, and delivered replay gates;
4. invalid-frame and lifecycle tests;
5. runtime actuator-limit agreement with opendbc;
6. shadow validation on held-out data.

Under that future contract, activation could occur only at an engagement
boundary. A provisional profile would be atomically versioned and rollback
would remain available. `Worse` feedback would deactivate it at the next
engagement without deleting evidence. Driver feedback is contextual only: it
does not enter either learner and cannot approve an artifact, waive a failed
objective or safety gate, or turn `stock_retained` into a candidate.

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
