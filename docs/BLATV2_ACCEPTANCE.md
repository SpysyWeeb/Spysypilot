# BLaTv2 modular acceptance

This document defines when the modular controller may progress from an
offline artifact to shadow collection and, eventually, actuation. A later
stage may not waive an earlier one. Until every activation gate passes, the
stock openpilot torque controller remains the sole actuator.

## Artifact identity

Every report identifies:

- the openpilot commit;
- the opendbc commit;
- the panda commit;
- the controller-policy artifact;
- the learned vehicle-profile SHA-256;
- the learner-evidence SHA-256;
- the replay-harness commit and input-timeline hashes.

The controller and replay import the same numerical source. Device-specific
copies or replay-only implementations are not acceptable.

## Opendbc and panda boundary

The generic controller contains no vehicle torque-count constants. At
runtime it receives the detected vehicle's maximum torque, build rate,
release rate, command cadence, and driver-torque limits from opendbc.
Projection tests compare the controller-side predictor directly with
opendbc's production limiter arithmetic.

Opendbc remains the platform command owner, including platform-specific
request-bit fault avoidance. Panda remains the independent safety ceiling.
Neither layer may add a steering feel adjustment. A submodule pointer change
is part of the steering artifact and is never absorbed during a merge.

## Foundation gate

Before shadow collection:

1. time, intent, reference, rack mapping, plant, observer, learner, actuator,
   feasibility, bootstrap, and invalid-output contracts pass their unit
   suites;
2. the model-published scalar action timestamp survives a cereal round trip;
3. malformed or stale plans cannot expose a prior plan;
4. forward and inverse plant equations are reciprocal on deterministic
   synthetic traces;
5. feasible actuator requests are transparent, and infeasible projections
   exactly match opendbc;
6. an unqualified or mismatched vehicle profile cannot activate;
7. source scans find no BLaTv1 mechanism, platform limit literal, hidden
   timing reconstruction, or second torque-shaping path.

## Shadow gate

Shadow mode is structurally incapable of publishing `carControl`. In the
current field graph it is an offline replay/harness surface only; manager
does not launch `blatv2_shadowd` onroad.

Device and harness recomputation must be bit-identical for deterministic
fields. Environment measurements, such as compute time, are excluded from
bit-exact comparison and reported separately. The device timing authority is
the on-device measurement, not workstation replay.

An offline shadow run may update an in-process slow-learner preview. That
preview is non-durable and cannot change a live profile or controller
selection. Field evidence comes only from a closed full rlog that passes the
offroad importer below.

## Learned-profile qualification

Training and artifact writes occur only offroad. `blatv2_backfilld` is the
sole durable evidence writer. Evidence is bound to the vehicle identity,
speed-node grid, observable calibration seed, profile schema, and learner
schema.

The importer accepts only complete, closed full rlogs; qlogs are insufficient.
It may discover routes recorded before the importer existed, but each route
must pass exact reviewed build/schema provenance, dongle/vehicle identity,
CarParams, controller-envelope, sensor-resolution, segment-continuity, and
source-coverage checks. A route-local rejection cannot prevent a later valid
route from being considered.

Each eligible batch is replayed twice in fresh runtimes and both results must
be byte-identical. An authenticated ledger binds every accepted, rejected, or
late-skipped route to its content and disposition so a route is never counted
twice. Publication writes a complete immutable generation before atomically
switching its `CURRENT` pointer.

An off-device worker may accelerate only route preparation. Before it is
field eligible, acceptance must additionally prove:

1. authenticated discovery, requests, progress, uploads, and downloads reject
   invalid HMACs, stale timestamps, replayed nonces, unknown keys, oversized
   bodies, path traversal, hash mismatches, and incompatible build/runtime
   identities;
2. a PC job reconstructs each route independently for both preparation
   authorities, while the device alone applies frames, compares A/A results,
   mutates the ledger, finalizes, and publishes;
3. local and remote preparation feed the identical device learner and yield
   byte-identical replay results, evidence, manifest, candidate, and ledger
   for the same frozen manifest; each mode is independently A/A exact;
4. disconnect, cancellation, service restart, PC-only-route rejection, and
   device onroad transition leave the previous `CURRENT` generation intact
   and do not leak scratch artifacts;
5. remote progress is monotonic after device-side restamping but remains
   display-only; and
6. with the worker absent, the original local importer remains fully
   functional without a toggle or recovery action.

Extractor identity is an execution property, not a pathname observation.
Both architectures must open the executable with `O_NOFOLLOW`, hash the held
file descriptor, execute that descriptor through `/proc/self/fd` with explicit
descriptor inheritance, and verify its inode, pathname, mode, and hash after
preparation. One extractor SHA is pinned for the entire learner transaction;
a pathname swap or a binary change between routes is a stable reader failure.

For each unseen accepted preparation-compatibility domain, the device first
prepares one locally retained test-vector route with the ARM extractor and
requires whole-spool SHA, size, frame count, and bytes to equal the PC
authority spool. The atomic HMAC-bound certificate is keyed by source,
opendbc, panda, runtime, historical/effective descriptor registries, both
extractors, the worker process instance, canonical join/extractor schemas, log
schema, CarParams, and physical compatibility. A worker restart therefore
forces recertification. A rejected route is consumable only after the same
local ARM preparation rejects it with the identical stable reason and message;
PC-only routes in an uncertified accepted or rejected domain fall back local.

Protocol v1 has a single-job 128-route bound and no batching. Exceeding it is
remote-unavailable and preserves the complete local transaction. Device
inventory itself is not truncated at that bound: authenticated, exclusive-
cursor pages enumerate the complete append-only PC archive. Every complete,
quiescent local route missing there is resumably archived, with routes needed
by the active job first. Completed segments remain private staging until an
authenticated exact ordered route manifest is re-hashed and atomically commits
the whole route; a partial prefix never enters inventory. This transfer cannot
select evidence or publish.
Device onroad handoff performs no network traffic; the server expires an abandoned
job after 30 seconds without authenticated status polling. Late-only ledger
updates preserve the prior authenticated generation's actual extractor
identity.

Cross-architecture equality is certified at the device-produced artifacts,
not by allowing an x86 worker to authorize its own final profile. The worker
has no Params, ledger, publication, controller-selection, or actuation API.
The final generation identity deliberately differs between local and remote
preparation because its provenance names the architecture-specific extractor
binary that actually decoded the rlogs. Treating those two binaries as the
same artifact would make provenance dishonest; semantic artifact equality is
the portability gate.

The current observable-calibration contract is profile schema 2, evidence
schema 6, coordinator artifact schema 5, and namespace
`complete_full_rlog_authority_v4`. It starts from empty evidence. Retired v1,
v2, and v3 artifact bytes are immutable and cannot be migrated into v4.

Each speed node independently requires its documented clean support,
bidirectional excitation, whole-route train/validation split, valid inverse
torque fit, and held-out improvement. An immutable route-counter parity owns
every category from a route, so one maneuver cannot train and validate itself.
Base, resolved-motion, confirmed stuck-to-motion breakaway, and
actuator-authority populations remain distinguishable.
Slew and stationary-full-torque observations cannot become equality-fit rows;
settled full magnitude may join only with resolved rack motion. Samples affect
only adjacent interpolation nodes. Consequently, extended highway use cannot
overwrite low-speed evidence.

The candidate may contain only torque per lateral acceleration, signed
lateral-acceleration offset correction, moving friction, and static breakaway,
plus seed delay/rate-resolution metadata. Casual driving may not promote rack
gain, rack damping, or another unidentifiable transient parameter. Changing
the retired provisional dynamics cannot change calibration identity.
The observable fit is constrained by construction to positive gain,
non-negative moving friction, and static breakaway no smaller than moving
friction. A boundary solution is re-solved on that physical face and remains
visible as `static == kinetic`; post-fit clipping is forbidden.

Raw measured steering-angle onset plus same-direction measured-rate
confirmation defines one physical breakaway episode. Training considers the
nested static-only, friction, offset-plus-friction, and full-map models. The
seed is comparator-only. A dense category cannot outvote a regression in a
sparse category, and held-out validation can reject only the frozen training
winner—it cannot choose a fallback model.

A candidate profile is emitted only when every required node qualifies.
Partial profiles remain evidence, not control artifacts.

Learning-status schema 2 is a strict display projection of those exact
populations and candidate values. Legacy rack-fit fields or an unknown schema
fail closed in the UI. The cache remains informational and has no approval or
activation authority.

`BLaTv2LearningOperationStatus` may expose logger finalization, historical
scanning/replay progress, and terminal diagnostics. No managed onroad process
publishes live collection state. It is a clear-on-manager-start display
cache, never evidence, approval, or a controller-selection input.

The field manager must contain exactly one BLaTv2 process on a real car:
`blatv2_backfilld`, and its predicate must be offroad-only. The shadow,
live-learner, and profile-lifecycle adapters remain unregistered offline
tools while stock is the sole active controller. A future activation build
must first add a reviewed offroad witness for exact provisional-profile
exercise and feedback; it may not restore an always-on lifecycle observer by
assumption.

This isolation is a measured field-load decision. On route `d1` from combo
build `ff842`, `blatv2_shadowd` exited with status `-6` 302 times at roughly a
5.5-second cadence and averaged 26.18% of one CPU core; `blatv2_learnerd`
averaged another 20.55%. Reused-`Text` Cap'n Proto arena accumulation is one
possible explanation for the shadow failures, but it is a hypothesis rather
than a proven cause. The acceptance invariant follows from the observed load,
not from that hypothesis: manager launches zero BLaTv2 processes while
started/onroad.

## Replay promotion gate

A complete profile and explicit controller policy are evaluated with the
same canonical input timeline for every controller. Route work is
parallelizable, but aggregation order, result bytes, and hashes are fixed so
worker count cannot change a result.

At minimum, promotion reports:

- signed steady-hold path error;
- signed turn-in timing;
- release timing and overshoot;
- aggregate and worst-window torque-rate roughness;
- raw request, opendbc-projected, and delivered-path domains;
- named sharp-turn, unwind, handoff, and archived blip vectors;
- constraint duty, invalid-frame behavior, and observer saturation;
- stock baseline and the last accepted reference implementation.

No single aggregate may hide an uncompleted sharp turn, early corner entry,
late release, or high-speed oscillation. A metric that is undefined remains
explicitly undefined; its denominator and exclusions are reported.

The modular candidate must improve the intended failure class without
regressing another core value. “Smooth. Swift. Strong.” is a joint contract,
not a weighted score that permits trading one word away.

## Future-only activation and rollback gate

The current observable-learning milestone does not execute this section. Its
all-node candidate is informational and hash-addressed only; it cannot write
`BLaTv2ApprovedArtifact`, stage a controller, or change stock selection. This
section pins the acceptance contract for a separately reviewed future
consumer.

An exact profile hash may eventually be staged only when raw/applied replay,
delivered-curvature replay, deterministic A/A, comma-device timing, and
safety approval all identify that same hash and source pair. Each result is a
separate fail-closed field in the canonical approval artifact; omitted,
non-boolean, or false fields cannot activate. Staging and selection occur
offroad or at an engagement boundary; never mid-drive.

The first active drive is provisional. On the next offroad transition the
driver is asked whether steering felt **Better**, **About same**, **Worse**,
or **Not sure**. The answer is evidence:

- **Worse** requests rollback at the next engagement;
- **Better** or **About same** may clear provisional status only if all
  objective gates remain valid;
- **Not sure** keeps the profile provisional;
- steering overrides are bookmarks, not automatic negative labels.

Invalid core output follows the tested hold, decay, comm-issue, and
ten-valid-frame recovery contract. That safety behavior is not a tuning
surface.

## Branch and field status

Local development commits do not make a build field eligible. The feature
branch is pushed only after its current artifact clears the applicable
foundation, replay, safety, and worktree-identity checks. Combo remains
untouched until a separately reviewed merge preserves combo's submodule
pointers and proves that no out-of-scope controller changed.

The README remains **in progress** until the owner field-tests the accepted
artifact and explicitly authorizes a status change.
