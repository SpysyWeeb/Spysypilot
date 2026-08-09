# BLaTv2 modular acceptance

This document defines when the modular controller may progress from an
offline artifact to shadow collection and, eventually, actuation. A later
stage may not waive an earlier one. Until every activation gate passes, the
stock openpilot torque controller remains the sole actuator.

The Palisade-only development parameter is a separate, explicit owner-trial
path for the bundled provisional tune. It does not pass or weaken any gate in
this document, create an approved artifact, or run beside stock. Stock remains
the default, and one controller is bound for the complete lateral session.

All current learning and qualification work is PC-only. The device records
ordinary full rlogs and runs no BLaTv2 learner, replay worker, route uploader,
or Wi-Fi bridge. An operator copies closed routes into durable PC storage over
read-only SSH. Any resulting profile remains informational until offline A/A,
the applicable gates, and a separate manual review and installation identify
the exact same artifact.

## Artifact identity

Every report identifies:

- the openpilot commit;
- the opendbc commit;
- the panda commit;
- the controller-policy artifact;
- the learned vehicle-profile SHA-256;
- the learner-evidence SHA-256;
- the physical generation and route-evidence-set SHA-256s;
- the behavior gate/segmentation, transaction, finalization, and generation
  SHA-256s when behavioral qualification has run;
- the exact recorded controller/source cohort identity; and
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

An offline PC shadow run may update an in-process slow-learner preview. That
preview is non-durable and cannot change a live profile or controller
selection. Field evidence comes only from a closed full rlog that passes the
PC importer below.

## Learned-profile qualification

Training and artifact writes occur only in the operator-controlled PC
workspace. One offline transaction owns route selection, both independent A/A
authorities, durable evidence, qualification, and immutable candidate
publication. Evidence is bound to the vehicle identity, speed-node grid,
observable calibration seed, profile schema, and learner schema. No output is
copied to the device without separate review.

The importer accepts only complete, closed full rlogs; qlogs are insufficient.
Routes enter its durable archive through an operator-initiated, read-only SSH
copy; there is no automatic upload. It may use routes recorded before the
importer existed, but each route
must pass exact reviewed build/schema provenance, dongle/vehicle identity,
CarParams, controller-envelope, sensor-resolution, segment-continuity, and
source-coverage checks. A route-local rejection cannot prevent a later valid
route from being considered.

Each eligible batch is replayed twice in fresh runtimes and both results must
be byte-identical. An authenticated ledger binds every accepted, rejected, or
late-skipped route to its content and disposition so a route is never counted
twice. Publication writes a complete immutable generation before atomically
switching its `CURRENT` pointer.

Each authority decodes and canonical-joins each raw route exactly once into an
independent `BLATRE02` version-2 route-evidence artifact. The physical frame
plane appears once alongside the compact model, controls, live-torque, delay,
maneuver, and event planes needed by behavioral replay. The physical-only
`BLATSP01` format is incompatible. No authority may consume its peer's prepared
bytes. Four production lanes consist of the two causal authority owners and one
private route-preparation helper per owner; worker counts 1 and 2 are diagnostic
modes, and 3 is invalid because it makes the authorities asymmetric.

### Retired device/PC bridge acceptance (historical)

The automatic LAN worker, device-side certification, upload, download, and
local-processing fallback are retired. The following requirements are retained
to explain historical protocol schemas and evidence provenance; they are not
current deployment requirements.

An off-device worker could accelerate only route preparation. Before it was
field eligible, acceptance additionally had to prove:

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
4. disconnect, cancellation, service restart, and device onroad transition
   leave the previous `CURRENT` generation intact and do not leak scratch
   artifacts; an A/A-identical PC-only rejection is absent from discovery,
   both authority inputs, the ledger, watermark, evidence, and cohorts;
5. remote processing, download, ARM certification, prepared-data handoff, and
   local-fallback progress are monotonic after device-side restamping but
   remain display-only; and
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
requires whole-artifact SHA, size, frame count, and bytes to equal the PC
authority route-evidence artifact. The atomic HMAC-bound certificate is keyed
by source, opendbc, panda, runtime, historical/effective descriptor registries,
both extractors, worker implementation, canonical join/extractor schemas, log
schema, the complete runtime-vehicle bundle identity, and the validated
physical-vehicle projection. Full recorded
CarParams bytes remain content-addressed per route but are not a separate
cross-architecture numerical domain when that physical projection is equal.
An otherwise identical worker restart therefore reuses the certificate; its
instance identity remains authenticated per job but is not a numerical input.
A rejected route is consumable only after the same
local ARM preparation rejects it with the identical stable reason and message.
An identically rejected PC-only route is not consumable and is instead
excluded before the effective route set reaches either learner authority; it
creates no durable disposition. A PC-only route in an uncertified accepted
domain falls back local.

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
The final generation identity deliberately differed between local and remote
preparation because its provenance names the architecture-specific extractor
binary that actually decoded the rlogs. Treating those two binaries as the
same artifact would make provenance dishonest; semantic artifact equality is
the portability gate.

### Retained contracts and PC qualification

The current contract identities are:

| Contract | Value |
| --- | --- |
| calibration profile / evidence / coordinator | 2 / 9 / 9 |
| runtime vehicle / calibration identity / provisional dynamics | 1 / 1 / 1 |
| physical learning / operation / progress status | 4 / 1 / 1 |
| native extractor / canonical join | 5 / 5 |
| physical-frame encoding | 2 |
| route evidence | `BLATRE04`, version 4 |
| backfill ledger / commit / pointer | 3 / 2 / 1 |
| inclusion namespace | `complete_full_rlog_authority_v8` |
| controller policy | 1 |
| behavior gate / segmentation / replay input | 3 / 1 / 1 |
| behavior transaction / finalization | 2 / 1 |
| behavior generation / pointer / route-set | 1 / 1 / 1 |
| behavior learning status | 1 |
| future feedback / lifecycle status | 2 / 2 |
| future approved artifact / calibration selection / activation state | 5 / 2 / 1 |
| off-device protocol / certification | 2 / 5 |
| off-device display progress | 2 |

The v8 physical namespace starts from empty evidence. Retired v1 through v7
artifact bytes are immutable and cannot be migrated into it.

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

The exact clean-support floors are 150 s at the 0/5 m/s nodes, 240 s at
10/15 m/s, and 420 s at 20/30 m/s. These are accepted weighted response
seconds, not drive-clock seconds. Every node also needs at least 20% held-out
support, bidirectional excitation, and at least four training and four
validation moving rows plus four training and four validation complete
breakaway episodes, with both directions represented. The shorter low-speed
floor recognizes that sharp turns are data-dense; it does not waive rare
breakaway or validation evidence. Missing support, missing variety, rank
deficiency, ill conditioning, inconclusive selection, and validation regression
must remain distinct reported outcomes.

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
seed is a selectable safe result, and a learned replacement must clear paired
whole-route uncertainty. A dense category cannot outvote a regression in a
sparse category, and held-out validation can reject only the frozen training
winner—it cannot choose a fallback model. Adjacent-node interpolation is
validated from exact sufficient statistics before a mixed profile qualifies.

A candidate profile is emitted only when every required node and interpolation
interval qualifies and at least one node differs from the seed. An all-seed
result is a complete qualified evaluation, not a candidate or an error.
Partial profiles remain evidence, not control artifacts.

Learning-status schema 4 is a strict display projection of those exact
populations, candidate values, and finite first-cause sample accounting.
Accepted plus every explicit rejection cause must equal the number of prepared
physical frames ingested. Legacy rack-fit fields, unknown rejection keys, or an
unknown schema fail closed in the UI. The cache remains informational and has
no approval or activation authority.

## Behavioral qualification

Behavior starts only after the physical `CURRENT` authenticates a fully
qualified selected profile. An all-seed physical selection is valid input; a
partial profile is not. Behavior failure or insufficient routes cannot mutate
the already published physical generation.

The only behavior dials are one speed-independent closed-loop natural frequency
and one damping ratio. Natural frequency represents response speed/stiffness;
damping ratio represents settling and resistance to ringing. No speed schedule,
maneuver rule, reference offset, observer change, or actuator-limit change is a
behavior candidate.

The route population must be the newest contiguous cohort with one exact
recorded controller/source identity. Missing, rejected, corrupt, or ineligible
evidence interleaved before a proven source boundary is blocking, not skippable.
The committed partition reserves two whole routes for validation, and paired
uncertainty requires two routes per side, so four homogeneous routes (two train,
two held out) is the minimum. Every metric also requires at least two routes,
three windows, and all of its committed speed/maneuver strata; four routes may
therefore remain insufficient.

Training scores exact stock, the currently accepted artifact, and every
candidate on the same scalar-anchored target and segmentation. Exact stock is
both bootstrap and incumbent when no approved modular artifact exists. One
winner is frozen before held-out data is examined. Held-out replay scores only
stock, incumbent, and that winner; it cannot choose a fallback. Logger events
locate evidence but do not label quality, and driver contact censors only the
post-contact response. Lane lines and driver interventions never become target
or vote inputs.

The following gate families are independent:

- **Smooth:** applied torque-rate RMS, worst one-second burst, and release
  overshoot;
- **Swift:** correction latency and signed delivered turn-in/release timing;
  early timing is not accepted as “fast”; and
- **Strong:** delivered-curvature fraction, maneuver completion, and integrated
  absolute path error.

Every candidate must satisfy absolute physical bounds, beat or match exact
stock and the incumbent beyond observed paired whole-route uncertainty, and
materially improve the committed path-error target. There is no weighted total
that can trade one contract away.

The entire transaction runs twice from independently reloaded `BLATRE02`
artifacts and fresh replay cores, using up to four fork workers within each
authority. Only canonical byte-identical results publish. The behavior store
uses immutable schema-1 generations under `behavior_generations_v1`; all files
are hash-bound before its own atomic `CURRENT` replacement. A passing result may
contain an informational policy. Any non-passing safe result is explicitly
`stock_retained` and omits `policy.json`. Neither result writes approval Params
or changes controller selection.

Retained `BLaTv2BehaviorLearningStatus` schema 1 historically exposed waiting for physical profile,
waiting for homogeneous routes, preparing, training, selecting, validating,
publishing, complete, or failed. It reports route/replay progress and separate
Smooth/Swift/Strong verdicts. Like all status Params, it is a rebuildable UI
projection, not evidence or authority. No current device process publishes it.

`BLaTv2LearningOperationStatus` historically exposed logger finalization,
historical scanning/replay progress, and terminal diagnostics. Its retained
schema is a clear-on-manager-start display-cache contract, never evidence,
approval, or a controller-selection input; no current device process publishes
it.

The field manager must contain zero BLaTv2 background processes on a real car.
Shadow, learning, replay, profile-lifecycle, transfer, and bridge entrypoints
remain absent whether stock or the explicit provisional trial owns the lateral
session. Retained libraries and wire/schema identities are offline
compatibility surfaces, not managed services. Any approved activation build
must add a reviewed offroad witness for exact candidate exercise and feedback;
it may not restore an always-on lifecycle observer by assumption.

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

The current learning milestone does not execute this section. Its selected
physical profile and optional behavior policy are informational and
hash-addressed only; neither pipeline can write `BLaTv2ApprovedArtifact`, stage
a controller, or change stock selection. `stock_retained` deliberately has no
policy file. This section pins the acceptance contract for a separately
reviewed future consumer.

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

Feedback is contextual only. It is never a learner input and cannot waive an
objective, timing, safety, source-identity, or portability gate. In particular,
**Better** cannot approve a failed artifact, and **About same** cannot erase a
measured regression. The prompt can request rollback or keep an already
approved future artifact provisional; it cannot create approval.

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
