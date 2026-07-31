# BLaTv2 learning and historical routes

BLaTv2 performs no managed data collection or learning work onroad. Normal
loggerd full rlogs contain the measured services needed later. After the
drive, the offroad-only `blatv2_backfilld` waits for loggerd to close the
route, reads the complete local full rlog, independently replays it twice,
and atomically publishes one authenticated evidence generation.

This means the UI can report `finalizing`, `backfilling`, and route-by-route
replay progress instead of continuing to say that a first drive is required
while work is in progress. The operation status is informational only. It is
never an approval, evidence, or controller-selection input.

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
- its reviewed steering limits, rack-rate resolution, seed profile, nominal
  mapping, and torque interpretation match the current physical runtime;
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
| `finalizing` | Waiting for logger closure, verifying the second replay, or publishing. |
| `backfilling` | Scanning complete routes or replaying a displayed route index/count. |
| `idle` | Authenticated evidence and its route ledger are committed. |
| `failed` | The stable diagnostic explains the fail-closed operation error. |

The historical process owns status only while manager Params says offroad and
checks that ownership again at each write boundary. `collecting` and
`drive_skipped_identity_mismatch` remain schema values for offline adapter
tests, but the current manager graph never launches their onroad publisher.

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

Immutable generations are not garbage-collected yet. A future collector must
be scoped to an offroad/boot reader lifetime, wait until no resolved artifact
paths can still reference an old generation, and retain at least `CURRENT` and
its predecessor. Until that reader-lifetime contract exists, retaining all
generations is the safe behavior.
