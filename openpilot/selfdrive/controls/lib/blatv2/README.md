# BLaTv2 learning and historical routes

BLaTv2 performs no managed data collection or learning work onroad. Normal
loggerd full rlogs contain the measured services needed later. After the
drive, the offroad-only `blatv2_backfilld` waits for loggerd to close the
route, reads the complete local full rlog, independently replays it twice,
and atomically publishes one authenticated evidence generation.

The current generation learns an observable inverse-torque calibration, not
the retired dynamic rack model. At each `0/5/10/15/20/30 m/s` node it keeps
stationary/base, resolved-moving, first-motion-after-dwell breakaway, held-out
validation, and actuator-authority populations separate. The only fitted
values are torque per lateral acceleration, signed lateral-acceleration offset
correction, moving friction, and static breakaway. Rack gain and damping are
neither fitted nor part of the calibration identity.

The numerical fit is deterministic constrained least squares. It
reparameterizes static breakaway as moving friction plus non-negative excess
and re-solves every active constraint face. A `static == kinetic` result is an
explicit statement that ordinary driving did not identify additional
breakaway; it is not a clipped or feel-tuned number.

This means the UI can report `finalizing`, `backfilling`, and exact pass,
route, and segment progress instead of continuing to say that a first drive
is required while work is in progress. `BLaTv2BackfillProgress` is a separate
display-only projection bound to the current operation identity. Its work bar
includes both compressed-segment reading and prepared-route application for
both passes. Its approximate remaining time stays unavailable until each kind
of work has independent timing support. Operation and progress status are
never approval, evidence, or controller-selection inputs.

## Can previous routes be used?

Yes, including routes that predate this importer, when their full rlogs still
exist locally and all compatibility checks pass. Qlogs and incomplete or
currently locked rlogs are not sufficient.

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

## Status meanings

| State | Meaning |
| --- | --- |
| `preparing` | Waiting for exact CarParams or restoring the runtime. |
| `ready_no_evidence` | No eligible committed route exists yet. |
| `finalizing` | Waiting for logger closure, verifying pass 2/2 with route/segment progress, comparing, or publishing. |
| `backfilling` | Scanning complete routes or replaying pass 1/2 with route/segment progress. |
| `idle` | Authenticated evidence and its route ledger are committed. |
| `failed` | The stable diagnostic explains the fail-closed operation error. |

The historical process owns status only while manager Params says offroad and
checks that ownership again at each write boundary. `collecting` and
`drive_skipped_identity_mismatch` remain schema values for offline adapter
tests, but the current manager graph never launches their onroad publisher.
The optional progress projection is `CLEAR_ON_MANAGER_START`, is tied to the
operation id and sequence to reject torn reads, and is removed at terminal
idle/failure. Older UI code continues to use the coarse operation status.

## Determinism and storage

Only complete full-rlog replay owns durable evidence. Routes are ordered by
their canonical route counter. Within a route, selected events are ordered by
`(logMonoTime, segment, recorded ordinal)`, and a controls witness may use only
a source at or before its timestamp. Recorded ordinal deliberately resolves
equal-timestamp ties.

Every eligible batch is replayed in two fresh runtimes and must produce
byte-identical evidence, manifest, candidate, counters, and ledger entries.
Publication stages immutable, content-addressed artifacts and changes readers
with one atomic `CURRENT` pointer replacement. The SHA-bound ledger provides
exactly-once route ownership and rejects changed content for an already known
route.

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
Rack acceleration is not an input to the observable fit. Full-magnitude rows enter the separate authority equality fit
after one aligned response interval of settled command-side dwell and only
when the rack is measurably moving. They remain deferred until at least four
held-out authority rows exist, then free and authority validation must each
beat or match the seed independently. Driver-limited, lateral-inactive,
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

Calibration profile schema v2, evidence schema v5, coordinator artifact schema
v4, learning-status schema v2, and canonical join schema v2 establish these
semantics. Older evidence is never reinterpreted or mixed silently.

A candidate is emitted only when all six speed nodes qualify. It remains an
unapproved, informational, content-addressed file: calibration code does not
write Params, cannot populate `BLaTv2ApprovedArtifact`, and has no approval or
activation API. Partial evidence emits no candidate, and even a complete
candidate leaves the exact stock torque controller selected until a separate
controller and its full acceptance contract are reviewed.

Immutable generations are not garbage-collected yet. A future collector must
be scoped to an offroad/boot reader lifetime, wait until no resolved artifact
paths can still reference an old generation, and retain at least `CURRENT` and
its predecessor. Until that reader-lifetime contract exists, retaining all
generations is the safe behavior.

Durable storage is also versioned by evidence-inclusion policy:
`<storage root>/<calibration runtime identity>/complete_full_rlog_authority_v3`.
The predecessor v1/v2 namespaces and any earlier unnamespaced runtime
directory remain byte-untouched and are never restored or mixed into this
policy. Version 3 always starts with an empty ledger and independently replays
every still-local eligible full rlog.
