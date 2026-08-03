# BLaTv2 learning and historical routes

BLaTv2 performs no managed data collection or learning work on the device.
Normal loggerd full rlogs contain the measured services needed later. An
operator copies closed routes over read-only SSH into durable PC storage, then
runs historical replay, physical learning, behavioral qualification, and
candidate generation on the PC. There is no automatic Wi-Fi upload, remote-job
bridge, or device-local fallback.

The numerical learner and replay libraries remain here as the shared offline
artifact. They produce informational, hash-addressed candidates only. A
candidate must pass deterministic A/A and the applicable replay/safety gates,
then receive separate manual review and installation before any device can
consume it. No learner API writes device Params, selects a controller, or
actuates.

The current generation learns an observable inverse-torque calibration, not
the retired dynamic rack model. At each `0/5/10/15/20/30 m/s` node it keeps
stationary/base, resolved-moving, confirmed stuck-to-motion breakaway, and
actuator-authority populations separate per immutable route. The only fitted
values are torque per lateral acceleration, signed lateral-acceleration offset
correction, moving friction, and static breakaway. Rack gain and damping are
neither fitted nor part of the calibration identity.

Breakaway is a vehicle-global physical episode, not one zero-rate frame. Raw
measured steering-angle displacement at half a declared rate quantum marks
the earliest possible motion, then a same-direction rate quantum must confirm
it within the existing transport delay. The midpoint of the last stuck and
first moving responses identifies static friction once per episode.

The numerical fit is deterministic constrained least squares. The math core
consumes exactly the route population supplied by its caller; it has no token,
partition service, or authority to claim that population is global TRAIN. The
PC trainer owns the immutable TRAIN manifest, authenticates route membership,
and passes only that sealed population to the core. Generic historical
backfill remains display-only evidence preparation and cannot promote a
profile. The learner evaluates the
nested `static only -> friction -> offset + friction -> full map` family with
route-grouped leave-one-route-out cross-fitting: each fold fits all other
TRAIN routes and scores only the omitted whole route. The route counter remains
immutable provenance and has no statistical meaning. Candidate choice uses
only aggregated out-of-fold paired losses with a whole-route uncertainty
envelope; global VALIDATION and TEST are inaccessible. After family selection,
that family is refit once on every TRAIN route for the published parameters.
The seed remains a first-class safe result when no family robustly improves.

Every coefficient-bearing stratum needs two independent contributing routes;
any nonempty stratum makes its route a cross-fit contributor, while four rows
are enforced only on each combined fold-fit population after the held route is
removed. A route with thousands of rows still counts once. Evidence schema 15
reports independent route counts,
cross-fit fold failures, paired out-of-fold losses, and the final all-route fit
separately. Every route also carries a source-assignment ledger whose accepted
base, moving, complete-breakaway, pending, fitted-authority, and unresolved-
authority counts must conserve both the interval strata and the durable global
accepted-frame count. Runtime interpolation evidence is also partitioned into
disjoint base, moving, complete-breakaway, and settled-authority strata. Every
populated stratum must independently avoid regression in every leave-one-route-out fold;
abundant ordinary rows cannot dilute a sparse physical failure. Earlier
evidence schemas are rejected rather than reinterpreted.

Historical replay assigns every controls witness a canonical ingestion
coordinate: route-content SHA-256, segment index, controls mono time, and
recorded ordinal. Replay verifies the witness against the same indexed
physical frame before the learner sees it. Every accepted and rejected
disposition then extends an ordered assignment hash chain. Accepted records
commit exact hexadecimal physical values and the exact node, interval,
support, training, and episode weights they contributed; rejected records
commit an empty contribution set. Each route commitment binds the distinct
route identity, route-content artifact, assignment chain, source accounting,
and sufficient statistics. The trainer supplies the complete ordered route
commitment mapping from its own independent replay; embedded roots are
structural integrity checks only and are never treated as population
authority. Live rows have no authenticated historical coordinate and therefore
cannot publish authoritative calibration evidence.

The `0/5/10/15/20/30 m/s` support floors are respectively
`150/150/240/240/420/420` accepted weighted seconds, not wall-clock drive
time. Every node also needs bidirectional torque and lateral-acceleration
excitation, at least two independent routes for every required stratum, and at
least four moving rows plus four complete breakaway episodes as numerical
floors. Low-speed sharp turns are rare but
data-dense; waiting, driver override, or unexcited straight travel does not fill
their node. Between nodes, evidence weights and runtime parameters interpolate
linearly, and every adjacent interval must validate independently. Highway data
therefore cannot erase a low-speed node.

## Retired device processing and bridge (historical)

The following records the former on-device status and LAN-preparation design.
Its schema ordinals, Params names, protocol versions, and generation formats
remain reserved for old rlogs and audit records, but manager no longer launches
the daemons and the current UI does not consume these caches.

The former UI could report `finalizing`, `backfilling`, and exact pass,
route, and segment progress instead of continuing to say that a first drive
is required while work is in progress. `BLaTv2BackfillProgress` is a separate
display-only projection bound to the current operation identity. Its work bar
includes both compressed-segment reading and prepared-route application for
both passes. Its approximate remaining time stays unavailable until each kind
of work has independent timing support. Operation and progress status are
never approval, evidence, or controller-selection inputs.

When the optional PC bridge was used, `BLaTv2OffdeviceProgress` independently
reports PC processing, bounded artifact download, ARM certification,
prepared-data handoff, or a stable local-fallback reason. A PC-only route that
both remote authorities reject cannot be certified without local bytes, so it
is excluded before effective discovery rather than recorded as a rejection.
It contributes no learner evidence, ledger entry, watermark movement,
readiness count, or behavior-cohort vote. Locally retained PC rejections fail
closed as `architecture_verification_rejection_unprovable`; the device never
recreates the removed full-route ARM replay merely to reproduce a reason and
message. After certified artifacts are handed off, the
ordinary local progress projection again owns route/application detail.
Accepted-route certificates are shared only across equal extractor/join
schemas, recorded log schema, complete runtime-vehicle bundle identities, and
validated physical-vehicle projections.
Full CarParams bytes remain verified per route, but do not create separate
numerical domains for physically identical recordings; an unchanged worker
implementation also retains certification across a service restart.

ARM certification is intentionally not another complete-route decode. The PC
authorities emit one deterministic whole-segment vector per prepared route;
segment 0 supplies authenticated CarParams and hash-selected interior/end
segments cover the preparation path. The source vector is capped at three
segments, 96 MiB compressed, and 30,000 controls witnesses. Its canonical
result is capped at 64 KiB. A killable child enforces a 120 s deadline, 450 MiB
child RSS, and 600 MiB combined RSS. The earlier full-route ARM reproduction
was removed after its overlapping decoded populations exhausted device memory.
One locally available canary proves each equal preparation/runtime domain on
ARM; it is an implementation/domain proof, not a claim that every route was
replayed on ARM. Per-route causality comes from the two independent PC
preparations and byte-exact route artifacts. That incident is why complete PC
artifacts are now authenticated and applied through bounded streams, while
unavailable-worker local fallback and remote artifact application use one
replay worker rather than recreating the four-PC-worker memory shape.

## Can previous routes be used?

Yes, including routes that predate this learner, when their complete full rlogs
exist in durable PC storage and all compatibility checks pass. Qlogs and
incomplete or currently open rlogs are not sufficient. Route transfer is a
manual, operator-initiated read-only SSH collection step for now.

Historical replay intentionally fails closed. A route is eligible only when:

- every segment is present, contiguous, closed, and structurally complete;
- the recording build is clean and has a reviewed descriptor for its exact
  superproject, opendbc, panda, and log-schema provenance;
- it belongs to the same dongle, vehicle fingerprint, and VIN when both VINs
  are available;
- its reviewed steering limits, rack-rate resolution, observable calibration
  seed, nominal mapping, and torque interpretation match the current physical
  runtime;
- CarParams are present and consistent throughout the route;
- all required measurement services cover the controls witnesses, with no more
  than one percent unresolved witnesses or inferred control gaps; and
- the bounded native reader accepts the complete raw rlog or its single zstd
  frame.

A route-local failure is recorded as rejected and does not prevent later valid
routes from being processed. If an older route appears only after a newer
route has already advanced the durable watermark, it is recorded as
`late_older_skipped`; inserting it later would change chronology and make
previous evidence non-reproducible.

The just-finished route receives one logger-quiescence poll before hashing,
measured from the first scan that discovers it unlocked. A prior locked scan
does not consume that guard. Pre-ledger evidence files are not silently
adopted or combined: the importer replays their full rlogs, and fails closed
if untracked legacy artifacts would otherwise be double-counted.

The current clean build descriptor is synthesized only while that build is
running. Before a shipped build becomes historical, its exact reviewed
descriptor must be retained in `historical_build_descriptors.json`; otherwise
first-time import of its remaining local routes correctly fails closed as
unreviewed.

## Retired device status meanings (historical)

| State | Meaning |
| --- | --- |
| `preparing` | Waiting for exact CarParams, restoring the runtime, or spending the bounded 30-second cold-boot grace discovering the optional PC worker. |
| `ready_no_evidence` | No eligible committed route exists yet. |
| `finalizing` | Waiting for logger closure, verifying pass 2/2 with route/segment progress, comparing, or publishing. |
| `backfilling` | Scanning complete routes or replaying pass 1/2 with route/segment progress. |
| `idle` | Authenticated evidence and its route ledger are committed. |
| `failed` | The stable diagnostic explains the fail-closed operation error. |

The historical process owned status only while manager Params said offroad and
checks that ownership again at each write boundary. `collecting` and
`drive_skipped_identity_mismatch` remain schema values for offline adapter
tests, but the current manager graph launches no BLaTv2 publisher or learner.
The optional progress projection is `CLEAR_ON_MANAGER_START`, is tied to the
operation id and sequence to reject torn reads, and is removed at terminal
idle/failure. Older UI code continues to use the coarse operation status.

Physical `BLaTv2LearningStatus` schema 10 distinguishes learned, seed retained,
missing support/variety, rank deficiency, ill conditioning, inconclusive
selection, cross-fit regression, fold completion, and per-stratum interpolation
state. The selected or seed-retained result carries its authoritative model
family, independent contributor counts, paired held-out loss, explicit
regressed-fold counts, final full-fit diagnostic, and unresolved diagnostics
rather than a summary with missing proof. It also carries
strict first-cause accounting for every prepared frame, so accepted evidence
plus the finite rejection-reason set always equals the ingested-frame count.
Frames absent before the first canonical poll and explicit source gaps remain
route-quality ledger facts because no physical frame exists to classify. The separately
rebuildable `BLaTv2BehaviorLearningStatus` schema 1 reports
`waiting_for_physical_profile`, `waiting_for_routes`, `preparing`, `training`,
`selecting`, `validating`, `publishing`, `complete`, or `failed`, plus route/job
progress and the independent Smooth/Swift/Strong verdicts. `complete` still
means either `qualified_candidate_available` or `stock_retained`; it does not
mean active. Both status documents are display-only.

## Determinism and storage

Only complete PC full-rlog replay owns durable evidence. Statistical
aggregation orders routes by `(route identity SHA-256, route content SHA-256,
canonical route counter)`. Identity and content hashes define immutable
membership; the counter is provenance and a final deterministic tie-breaker,
not a fold assignment. Within a route, selected events are ordered by
`(logMonoTime, segment, recorded ordinal)`, and a controls witness may use only
a source at or before its timestamp. Recorded ordinal deliberately resolves
equal-timestamp ties.

Every eligible batch is replayed in two fresh runtimes and must produce
byte-identical evidence, manifest, selected profile, optional learned
candidate, counters, ledger entries, and complete `BLATRE02` route artifacts.
Publication stages immutable, content-addressed artifacts and changes readers
with one atomic `CURRENT` pointer replacement. The SHA-bound ledger provides
exactly-once route ownership and rejects changed content for an already known
route.

Each A/A authority decodes each full rlog once into its own `BLATRE02`
version-2 artifact. The artifact contains the physical frame plane once plus
model, controls, torque, delay, maneuver, and event planes, so behavior replay
does not decode the raw route again. Four production lanes are the two
independent causal authority owners plus one private preparation helper per
owner. Helpers share neither artifacts nor learner state. Worker counts 1 and 2
remain deterministic diagnostics; 3 is rejected as asymmetric.

Behavior replay consumes only the newest contiguous exact-source cohort. The
newest normal ledger route establishes the source; missing, corrupt,
ineligible, or rejected evidence before a proven older source boundary blocks
behavior qualification instead of being skipped. Explicit `late_older_skipped`
entries predate the append-only watermark and are ignored. This strict rule
prevents route cherry-picking while leaving physical calibration free to record
and explain route-local rejections.

Behavior starts only after the physical `CURRENT` provides a fully qualified
selected profile, including a legitimate all-seed result. Its minimum cohort is
four homogeneous routes: two whole-route training and two held out, with more
required whenever the committed speed/maneuver strata lack support. Training
replays exact stock, the accepted artifact (exact stock again during bootstrap),
and all candidates. It freezes one winner before held-out replay of only stock,
incumbent, and winner. Auto-logger events are locators, driver contact censors
post-contact response, and neither is a quality vote.

The only candidate values are global closed-loop natural frequency (response
speed/stiffness) and damping ratio (settling/ringing). Smooth, Swift, and Strong
are separate gates, never a weighted score. The complete behavior transaction
is rebuilt twice from fresh artifacts/cores and must be byte-identical. A safe
failure publishes `stock_retained` with no policy file.

`accepted_sample_count` means a valid hands-off measured-response frame
incorporated into base, moving, breakaway, or authority evidence. Reachable driver-free full
magnitude and maximum-slew boundaries are retained because `carOutput`
records the actual CarController input to the rack. Slew rows are authority
observations only; stationary full-torque rows likewise remain unresolved
authority observations rather than equality-fit data.

The replay retains distinct controls-witness, car-state response,
`carOutput` report, and applied-command effective timestamps. Since `card.py`
publishes `last_actuators_output` before applying its next output, a
`carOutput` payload is effective at the preceding `carOutput` publication.
For rack response at time `t`, the input is the newest exact zero-order-held
command effective at or before `t - transport_delay(speed)`. No future,
interpolated, or same-frame command can enter the fit. Speed, mapping,
lateral acceleration, rack rate, and node weights remain at response time.
Rack acceleration is not an input to the observable fit. Full-magnitude rows
enter the separate authority equality fit after one aligned response interval
of settled command-side dwell and only when the rack is measurably moving.
Every populated authority stratum can veto a regressing model. Authority rows
alter the fit only after four training observations; that fitted use then
requires four independent held-out observations before qualification.
Driver-limited, lateral-inactive,
standstill, invalid/gapped, and physically unreachable transitions are
excluded.

Some vehicle interfaces publish signed steering rate; others publish only a
magnitude. Learning preserves that measured magnitude and reconstructs sign
from offset-corrected steering-angle motion when necessary. It rechecks angle
direction across nonzero reversals and bridges only sensor-quantization
plateaus. A valid reversal remains moving-response and direction-coverage
evidence because rack acceleration is not fitted; it clears prior dwell so it
cannot fabricate a breakaway. Lifecycle and mapping discontinuities clear
cross-frame direction.

The committed identities are calibration profile/evidence/coordinator
`3/14/14`, physical learning/operation/progress status `7/1/1`, native
extractor/canonical join `4/3`, route evidence `BLATRE02` version `2`,
backfill ledger/commit/pointer `3/2/1`, controller policy `1`, and namespace
`complete_full_rlog_authority_v7`. Behavior uses gate/
segmentation/replay-input `3/1/1`, transaction/finalization `2/1`, generation/
pointer/route-set `1/1/1`, and learning status `1`. Off-device protocol and
cross-architecture certification are `2/5`, and off-device display progress
is `2`; these retired wire identities stay reserved. Future feedback/lifecycle and
approved-artifact/selection/activation contracts are `2/2` and `5/2/1`.
Older evidence is never
reinterpreted or mixed silently.

A node can qualify either because a learned fit clears paired whole-route
uncertainty or because the existing seed remains the demonstrably safest
result. Rank deficiency, ill conditioning, inconclusive validation, and
validation regression remain distinct outcomes. Exact interpolation between
adjacent node results is also validated; endpoint success alone is not enough.

A selected physical profile is emitted whenever all six speed nodes and all
five interpolation intervals qualify, including an all-seed result. A learned
candidate is emitted only when that selected profile changes at least one node;
an all-seed result therefore remains complete and healthy without inventing a
redundant candidate. Any candidate remains an
unapproved, informational, content-addressed file: calibration code does not
write Params, cannot populate `BLaTv2ApprovedArtifact`, and has no approval or
activation API. Partial evidence emits no candidate, and even a complete
candidate leaves the exact stock torque controller selected until a separate
controller and its full acceptance contract are reviewed.

After physical publication, behavior has a separate immutable store under
`behavior_generations_v1`. Its generation binds the exact physical generation
and profile, route set/source, committed gate and segmentation files, replay
core identities, A/A transaction, finalization, and optional policy. Only after
all files are fsynced and hash-authenticated does its own atomic `CURRENT`
pointer change. A malformed existing `CURRENT` blocks publication rather than
being overwritten as an empty cache. Behavior candidates likewise have no
approval or activation API.

Immutable generations are not garbage-collected yet. A future collector must
be scoped to an offroad/boot reader lifetime, wait until no resolved artifact
paths can still reference an old generation, and retain at least `CURRENT` and
its predecessor. Until that reader-lifetime contract exists, retaining all
generations is the safe behavior.

The future offroad feedback prompt—**Better**, **About same**, **Worse**, or
**Not sure**—is contextual only. It is not a physical or behavior learner input
and can never waive objective, source, timing, or safety gates. In a separately
approved activation lifecycle it may request rollback or leave an artifact
provisional; it cannot create approval.

Durable storage is also versioned by evidence-inclusion policy:
`<storage root>/<calibration runtime identity>/complete_full_rlog_authority_v7`.
The predecessor v1/v2/v3/v4/v5/v6 namespaces and any earlier unnamespaced runtime
directory remain byte-untouched and are never restored or mixed into this
policy. Version 7 always starts with an empty ledger and independently replays
every eligible full rlog in the operator-selected PC archive.
