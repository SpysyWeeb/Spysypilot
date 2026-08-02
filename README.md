# BLaTv2 — modular adaptive lateral control

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot),
created from the current untouched `stock` tip.

## Status

**In progress — ground-up replacement. Not field eligible.**

**PC learner transition — in progress.** Historical replay, physical learning,
behavioral qualification, and candidate generation now run only on an
operator-controlled PC. The comma records ordinary full rlogs and may consume
only a separately reviewed and manually installed artifact. It runs no BLaTv2
learner and has no automatic Wi-Fi upload or processing fallback. Controller
and actuation behavior are unchanged in this milestone.

This branch replaces the previous BLaTv2 controller architecture. Git history
and useful test infrastructure remain available for audit, but no previous
BLaTv2 controller mechanism is part of the new active design by default.
Until the complete modular candidate passes replay, safety, timing, and
portability gates, the stock openpilot torque controller remains the only
active controller. The candidate is evaluated in offline replay and harness
tests; the field manager does not launch a BLaTv2 shadow process onroad.

The product target is **Smooth. Swift. Strong.**

- **Smooth:** the requested wheel motion is continuous and intentional, without
  jitter, ping-pong, or modules correcting one another.
- **Swift:** useful torque is requested quickly enough to meet the model's
  authored path timing. The path is not moved earlier to conceal controller
  delay.
- **Strong:** the controller uses the vehicle's available authority when the
  model path requires it and does not accept persistent tracking error merely
  to keep the torque trace quiet.

The architecture, module contracts, learning policy, opendbc boundary, and
acceptance sequence are documented in
[`docs/BLATV2_MODULAR.md`](docs/BLATV2_MODULAR.md) and
[`docs/BLATV2_ACCEPTANCE.md`](docs/BLATV2_ACCEPTANCE.md).

No BLaTv1 controller code or `HyundaiLowSpeedTorqueDamping` is inherited.
Vehicle-specific magnitude/rate limits come from the active opendbc
`CarControllerParams`; the controller contains no Palisade limit literals.
opendbc may enforce platform command limits and panda safety, but it may not
damp, boost, filter, or otherwise reinterpret the controller's normalized
request.

### What is built

The replacement is split into independently testable owners:

- a model-time/intent adapter and stateless scalar-anchored future reference;
- one measured rack mapping and forward/inverse physical plant;
- one computed-torque core;
- one exact opendbc command envelope and invalid-output guard;
- a PC-only full-rlog importer and observable, speed-local inverse-torque
  learner;
- offline-only artifact, feedback, promotion, and rollback test surfaces for a
  future reviewed consumer; none is manager-launched in this milestone.

The current learning milestone replaces the rejected dynamic-rack fit. Normal
driving did not independently identify rack gain and damping, so those
provisional values are no longer learned or allowed to influence calibration
identity. Evidence schema 9 instead learns only quantities the logs directly
support at the `0/5/10/15/20/30 m/s` nodes:

- normalized torque per measured lateral acceleration;
- a signed residual lateral-acceleration offset correction;
- moving (kinetic) friction from resolved rack motion; and
- static breakaway torque from a complete stuck-to-moving episode.

The physical fit is deterministic constrained least squares. A vehicle-global episode
detector uses raw measured steering-angle motion at half one declared
steering-rate quantum, then requires same-direction measured-rate confirmation
within the existing transport delay. This observes physical breakout before a
coarse rate sensor changes. Gain, moving friction, and the excess of static
over moving friction are non-negative by construction; the signed offset
remains free. This is not a post-fit clamp: each active constraint is
re-solved.

Whole routes—not adjacent frames—alternate between training and validation by
their canonical route counter: even counters train and odd counters validate.
The counter travels with the route through preparation, replay, persistence,
and restore; it is never reconstructed from replay order or a content hash.
The training side evaluates a nested model
family (`static only`, `friction`, `offset + friction`, then `full map`). The
seed is now a first-class selectable result: if the data cannot prove a
candidate is better beyond route-level uncertainty, the node is successfully
qualified as **seed retained** instead of being mislabeled unstable. A learned
model may advance only when its paired whole-route loss clears that uncertainty
without sacrificing a populated category; the frozen winner is checked once
on held-out routes. Dense straight-driving samples therefore cannot buy a
lower average error by sacrificing rare breakaway or full-authority behavior.

Stationary settled response, moving response, breakaway events, held-out
validation, and actuator-authority observations remain separate populations.
Rank and conditioning are explicit diagnostics, and exact runtime interpolation
between neighboring speed nodes is validated as its own response rather than
being assumed safe from endpoint fits. No partial node set can emit a candidate,
and no candidate produced by this milestone has an approval or activation path.

Controller feel is learned separately from physical calibration. A behavioral
learner is allowed to change only two global, physically legible response dials:
closed-loop natural frequency and damping ratio. It evaluates exact stock, the
currently accepted artifact, and candidates on identical cached route evidence,
with whole-route train/held-out splits. Smooth, Swift, and Strong are independent
contracts; no weighted score may trade one away. Auto-logger events locate
maneuvers but are not quality labels, driver intervention censors response after
contact but is not itself a good/bad vote, and lane lines never enter the target.

These are two separate qualification stages. Physical calibration first decides
what torque the detected vehicle needs at each speed. A fully qualified physical
result may legitimately carry the existing seed at every node when the data do
not prove a safer replacement. Behavioral qualification starts only after that
complete selected physical profile exists; it decides how quickly and how well a
future modular core should track the unchanged model target. A behavior failure
cannot relabel a successful physical generation, and neither stage can activate
its own output.

Natural frequency is the global response-speed/stiffness dial: higher values ask
the closed loop to close tracking error sooner. Damping ratio is the global
settling dial: it determines how that response settles instead of ringing or
overshooting. They are one pair for the full speed range, not hidden low/high
speed gains. Vehicle speed dependence remains in the measured physical profile
and the vehicle model.

The previous **LQI** controller—“Linear Quadratic Integral,” meaning a controller
that chose feedback from a mathematical error/effort cost and accumulated an
integral error state—is retired. It is not the controller being built here.
The exact stock torque-controller algorithm remains the active bootstrap, and a
future controller must consume one qualified calibration map in place of its
stock map—not add another correction loop around it.

The first combo build contains no approved modular profile. It therefore
drives with the exact current stock torque-controller algorithm and launches
no dedicated BLaTv2 process onroad. Normal loggerd full rlogs retain the
measured services needed for later learning. An operator periodically copies
closed routes from the device over read-only SSH into durable PC storage; no
device process uploads, replays, fits, or publishes them automatically. The PC
independently replays each complete full rlog before any evidence can become
durable. On opendbc's validated Palisade/Telluride
platform, that stock request passes through the
platform-selected **409/4/7** opendbc/panda envelope. Other cars keep their
own stock limits and remain stock-controlled unless their opendbc port
explicitly validates the complete modular command-envelope and rack-sensor
contract.

## Activation and learning

The first drive on a vehicle is stock-controlled. Normal loggerd is the only
BLaTv2 data source on the comma: it records the same full rlog openpilot already
uses, with no learner, replay worker, route uploader, or Wi-Fi bridge in the
manager process graph. Route collection is currently an operator-initiated,
read-only SSH copy into durable PC storage. The device neither initiates nor
retries that transfer.

The PC owns historical-route selection, deterministic preparation, both A/A
replay authorities, physical and behavioral qualification, and immutable
candidate generation. A candidate is informational until its exact source,
route set, evidence, gates, and output hashes have been reviewed. Installation
on the device is a separate manual action after that review; the learner cannot
write device Params, select a controller, or activate what it produces.

The pure learner, replay, evidence, and publication libraries remain in this
repository so the PC and later controller review use one numerical artifact.
The device-side daemons and network transport are retired. Their schema
ordinals, Params names, historical generation formats, and compatibility
records remain reserved so prior rlogs and audit records stay decodable.

### Retired on-device importer (historical)

The following records the former offroad importer for provenance and explains
the retained schemas and generation files. It is not current manager behavior.
Qualification mathematics described here remains applicable when invoked by
the PC, but every statement assigning ownership to `blatv2_backfilld`, the
comma, or a device progress UI is historical.

The retired `blatv2_backfilld` independently prepared and replayed each complete
full rlog twice and atomically committed evidence only when both authorities and
all compatibility checks agreed. It was the sole managed BLaTv2 process and
durable learning writer in that deployment.

The offroad importer used four worker lanes while retaining exactly two
independent deterministic
replay authorities: the parent and a verification process. Each authority now
has one private route-preparation lane that may decode the next route while its
owner applies the current route, so up to four Python lanes can use the four
comma CPU cores without parallelizing the learner's causal route/frame state.
Each helper writes one versioned, hash-bound `BLATRE02` route-evidence
artifact. It contains the exact physical frame plane once plus compact model,
control, live-torque, delay, maneuver, and event planes; the physical-only
`BLATSP01` format is rejected. A helper never sends a route-sized Python object
over IPC, never shares prepared data with the other replay, and never receives
durable-writer or Params authority. Only the parent compares the complete bytes
from both authorities, extends the ledger, and atomically publishes. One
prefetched route per authority bounds memory and scratch usage.
While a prefetched route is being applied, its helper may already be writing
the following route, so the hard scratch bound is two artifacts per authority.
Each artifact is independently capped at 512 MiB, making 2 GiB the deliberately
conservative four-artifact theoretical scratch ceiling; ordinary routes are
far smaller. Device validation measures the actual high-water mark.
The existing progress UI is deliberately unchanged: it projects the primary
pass, then moves to verification while the independently reconstructed pass
finishes; helper work is accounted in canonical route order and is not exposed
as a third or fourth replay pass. Four lanes improve a multi-route backlog but
cannot accelerate a single newly completed route because there is no next
route to prepare. Worker counts 1 and 2 remain deterministic diagnostic modes;
3 is rejected because it would make the A/A authorities asymmetric.
On the 21-segment `000000b7--a6b3b1f175` reference route, the same desktop
host completed the two passes in 17.594 s serially and 9.717 s with two
workers (1.81x), with identical evidence, manifest, and ledger content. That
is supporting evidence only. The integrated native-extractor/A/A/publication
benchmark over b7, b8, b9, and ca took a 33.754 s four-lane median versus a
42.359 s two-lane median: 20.3% less wall time (1.25x throughput), with exact
evidence, manifest, ledger, provenance, and generation hashes. Four lanes
peaked at about 1.21 GiB process-tree PSS and 26.8 MiB scratch, then left no
children or scratch behind. Four-worker processing has since completed on the
comma without changing deterministic artifacts; two-worker mode remains the
bounded fallback. The learner itself remains in progress because no candidate
may activate before every speed node and held-out gate qualifies.

Its separate display-only progress projection reports the current pass,
route, segment, and whether the route is being read or applied. A cumulative
bar spans both passes without reaching a pass boundary before the prepared
route has actually been ingested. An approximate remaining time appears only
after independent reading and application rates have enough observations;
none of these timing fields enter evidence or determinism comparisons.

The importer could also use older local routes, including routes recorded before
this importer existed, when their complete full rlogs remain on the device and
their exact build/schema, dongle, vehicle, CarParams, controller-envelope,
sensor-resolution, and source-coverage checks pass. Qlogs, incomplete or
open routes, unreviewed builds, incompatible routes, and late-discovered older
routes are not imported. A rejected route does not block a later compatible
one.

The reviewed historical registry includes the `3849a2f`, `2447667`, and
`9338f5b` combo builds recorded by routes d2-d6. It also retains `fdd5560`,
the clean combo build running immediately before this bridge landed, so any
of its remaining routes stay reviewable after the update. Their schema and
Palisade command/panda envelope are verified as the same 409/4/7 contract
used by the current learner. Older archived builds that actually used the
stock 384/3/7 envelope remain fail-closed; they are not mislabeled to make
their data pass.

### PC learner qualification

A selected physical profile becomes complete only when every speed node and
every adjacent interpolation interval has enough excitation, independent
validation, and bounded uncertainty. Runtime values are linearly interpolated
between the `0/5/10/15/20/30 m/s` nodes and held flat beyond the endpoints.
Each sample contributes only to its neighboring nodes, so highway mileage
cannot overwrite low-speed knowledge.

The support floors count accepted, weighted, hands-off response time—not route
duration. They are 150 s at the 0 and 5 m/s nodes, 240 s at 10 and 15 m/s, and
420 s at 20 and 30 m/s. Each node also needs at least 20% held-out support,
bidirectional lateral-acceleration and torque excitation, and independent
moving and complete breakaway evidence in both directions. Moving and
breakaway strata each require at least four training and four validation rows.
Low-speed turns are rare but information-dense: a few meaningful turns can add
valuable breakaway events, while minutes spent stopped, driver-limited, or
without useful excitation add no qualifying support. More driving is needed
only for the specific populations the UI reports as missing; rank-deficient,
ill-conditioned, inconclusive, or regressing data requires better variety or a
safer retained seed rather than simply forcing the node through.

The learner's current evidence namespace is
`complete_full_rlog_authority_v7`. It starts empty by design. The retired v1
through v6 namespaces and their artifact bytes are never migrated, edited, or
interpreted as schema-9 evidence; compatible retained full rlogs are replayed
from source. Runtime identity for this namespace excludes the retired
provisional rack-gain/damping seed while remaining bound to the detected
vehicle, torque mapping, rack mapping, sensor resolution, and opendbc command
envelope.

After physical publication, behavioral replay uses the newest contiguous
cohort recorded by one exact controller/source identity. The current gate needs
at least four homogeneous routes: two whole routes for training and two frozen
held-out routes. Four is only the partition minimum; every Smooth, Swift, and
Strong metric also requires its committed maneuver/speed strata, at least two
routes, and at least three windows, so qualification may need more routes. A
rejected, missing, corrupt, or behavior-ineligible route interleaved inside the
newest source cohort blocks qualification rather than being cherry-picked
around. Routes from an older controller build can form an older cohort, but are
never mixed into the newest one.

Behavior qualification is currently **fail-closed pending the
route-major streaming backend**. Cohort selection authenticates compact,
file-backed route summaries without decoding the route object graph. Once an
otherwise eligible cohort reaches the four-route partition minimum, the PC
reports `behavior_streaming_required` and emits no candidate before loading any
eager behavior artifact. This is a memory-safety diagnostic, not a failed
Smooth/Swift/Strong result and not a request for more driving. The old eager
path expanded the measured 146,363,115-byte CE evidence file to 909,200 KiB
before the
behavior decoder and replay outputs were constructed; no file-size threshold
can prove the full downstream transaction safe. The exact replacement and its
acceptance gates are documented in
[`docs/BLATV2_BEHAVIOR_STREAMING.md`](docs/BLATV2_BEHAVIOR_STREAMING.md).

Training replays exact stock, the currently accepted artifact (or exact stock
again during bootstrap), and the complete candidate grid. It freezes one
winner before the held-out routes are examined. Held-out replay then evaluates
only exact stock, the incumbent, and that frozen winner; validation cannot pick
a fallback. The target must materially improve beyond observed whole-route
uncertainty while every Smooth, Swift, and Strong gate passes independently.
Otherwise the immutable result is **stock retained** and contains no behavior
policy.

### Retired Wi-Fi preparation bridge (historical)

The automatic LAN bridge is retired and no service attempts discovery, route
upload, remote job creation, download, certification, or local fallback. The
following is retained only to explain historical protocol schemas, Params, and
audit records. The former paired PC worker could take over the expensive
immutable route-preparation stage when it was reachable on the local network. The
standalone worker project lives at
`/home/alex/Documents/blatv2-remote-worker`; its durable full-rlog archive is
the `data/routes/` subdirectory. The comma remains authoritative:

- the device freezes the ordered route manifest and explicitly selects any
  archive-only routes offered by the PC;
- authenticated inventory is read in strictly ordered 128-route pages, while
  every complete quiescent local route absent from the PC is resumably copied
  to private staging (job-required routes first). After every segment matches,
  the device submits the exact ordered route manifest and the worker atomically
  publishes the whole route into `data/routes/`; an interrupted prefix is never
  inventory-visible. Archive synchronization has no learner or publication
  authority;
- the PC authenticates every request, verifies exact source/runtime/build
  identities, and independently decodes/joins each selected route twice with
  four preparation workers;
- each ARM or x86 preparation opens its native extractor once with
  `O_NOFOLLOW`, hashes that held executable inode, and runs that exact inode
  through `/proc/self/fd`; a transaction pins one extractor hash across every
  route so pathname swaps or mixed extractor versions fail closed;
- the device downloads bounded, hash-bound full route-evidence artifacts and
  their independently reproduced certification vectors. Full artifacts are
  authenticated and applied as streams; the comma never calls `read_bytes()`
  or materializes a complete route object merely to consume PC output. The two
  authorities still have to match byte-for-byte before the device extends the
  ledger, finalizes, or atomically publishes `CURRENT`;
- before any accepted x86-prepared domain can reach that learner, the device
  snapshots and replays one deterministic whole-segment canary vector for that
  implementation/runtime domain with its ARM extractor and requires its exact
  result to match both PC authorities. This is a domain-level numerical proof,
  not a per-route ARM replay. Both PC authorities still prepare and compare
  every complete route independently. The
  vector always includes segment 0 for authenticated CarParams bootstrap, then
  chooses hash-stable interior/end coverage with a deterministic fallback. It
  is capped at three complete segments, 96 MiB of compressed source, 30,000
  controls witnesses, and a 64 KiB canonical result. This replaced the earlier
  full-route ARM check after that check exhausted device memory while
  duplicating hundreds of MiB of decoded route state;
- certification runs in a killable child with a 120 s deadline, 450 MiB child
  RSS and 600 MiB combined parent/child RSS ceilings. Source and artifact
  scratch use private directories, exact inode identities, bounded copy
  buffers, and a full-transaction disk-space preflight covering downloads,
  authority staging, and publication overlap. Abandoned scratch is quarantined
  rather than recursively deleted, and one existing quarantine blocks another
  transaction so repeated crashes cannot silently consume the disk. A timeout,
  resource excess, malformed vector, or
  certificate mismatch is a stable fail-closed result and never falls back to
  a full local replay of that remote outcome;
- the private atomic HMAC-bound certificate is scoped to both extractor
  binaries, the narrow byte-producing preparation implementation, Python
  executable/native numerical environment, immutable build/runtime identities,
  complete runtime-vehicle bundle, vector selection/result, and validated
  physical-vehicle compatibility domain. The complete recorded CarParams hash
  remains bound to each route artifact and A/A result, but irrelevant per-drive
  CarParams bytes do not fragment one physical numerical domain. An otherwise
  identical worker-service restart reuses the certificate because its
  authenticated session identity is transport state, not a numerical input;
  a rejected PC outcome has no bounded numerical result that ARM can compare.
  The device therefore never performs a hidden full-route ARM rejection replay:
  a locally retained rejection fails closed as
  `architecture_verification_rejection_unprovable`; an archive-only rejection
  is excluded from the effective discovery set and from both authority outputs.
  It creates no ledger row, watermark movement, learner count, or behavior-
  cohort vote and is reported only as an unverified exclusion;
- remote progress is restamped by the device onto its monotonic clock. The
  separate display-only `BLaTv2OffdeviceProgress` projection distinguishes PC
  processing, artifact download, ARM certification, prepared-data handoff,
  and an explicit local-fallback reason. The existing local progress resumes
  ownership once the device consumes prepared artifacts. Neither projection
  is evidence, and PC clocks and job identifiers never become evidence;
  and
- an unavailable or interrupted worker falls back to the existing local
  importer without changing the last authenticated generation. Device-local
  fallback deliberately uses one replay worker to stay inside comma memory;
  the PC retains four preparation workers. Incompatible, unauthenticated,
  replayed, oversized, or corrupt responses fail closed.

Discovery normally uses a signed LAN broadcast. Networks that suppress
broadcasts between Wi-Fi and Ethernet may instead place one canonical private
IPv4 address in the protected device file
`/data/params/blatv2-offdevice-bridge/worker_host.txt` (mode `0600`, beside the
raw 32-byte secret). When present, that address replaces only the UDP transport
target: discovery is still HMAC-authenticated, source-pinned, timestamped, and
bound to the exact source commit. An unreachable configured host is ordinary
worker unavailability and retains the four-worker local fallback.

The first offroad transaction probes the worker immediately, then retries only
transient discovery absence for up to 30 seconds before selecting local replay.
This bounded grace gives Wi-Fi time to associate without delaying an already
reachable worker. Offroad ownership is checked around every attempt; an
authentication, configuration, source, or protocol failure is never hidden by
the timer.

Certification replays selected whole segments independently. A contiguous
segment-start controls prefix whose selected poll precedes the first
segment-local `carState` or `carOutput` is unscoreable without borrowing state
from another segment, so it is excluded and counted explicitly as
`segment_local_measurement_context`. The ordinary whole-route canonical input
builder remains strict; this exception exists only at the certification
segment boundary.

Protocol v2 deliberately does not split one publication across worker jobs.
More than 128 selected replay routes therefore makes the remote backend
unavailable and runs the complete local transaction. Onroad handoff performs
no cleanup network I/O; a PC job left behind by that ownership transition is
stopped by the worker's 30-second authenticated-status-poll lease and can
never publish on the device. A late-only ledger generation carries forward
the extractor identity from the authenticated generation whose evidence it
retains rather than claiming an unused binary.

This boundary was intentional. It is superseded by PC-only processing: the
device no longer has a local processing fallback, and the PC still has no API
to write device Params, select a controller, or actuate.

### Offline publication and future activation

Both qualification stages stop at immutable, informational results. The
physical store publishes a content-addressed generation and atomically replaces
its `CURRENT` pointer only after exact A/A agreement. The behavior store then
does the same under `behavior_generations_v1`, with its own schema-1 generation
and `CURRENT` pointer. A behavior generation authenticates the selected physical
generation/profile, exact route set and source, committed gate/segmentation
files, both-authority transaction, and optional policy. A corrupt existing
pointer is never overwritten as if it were an empty cache, and old generations
are retained for rollback/audit.

Neither stage promotes, activates, or rolls back its output, and stock remains
selected even when both stages qualify. The following lifecycle is the
separately gated future consumer contract, not current behavior: a reviewed
profile could change only at an engagement boundary, and after its first active
validation drive the offroad review would ask:

> Compared with the previous steering profile, how did steering feel?

The choices are **Better**, **About same**, **Worse**, and **Not sure**.
Under that future contract, `Worse` would deactivate the provisional profile
for the next engagement while retaining its data for diagnosis. Driver
overrides remain evidence bookmarks, not automatic proof that a controller is
bad. The prompt is contextual evidence only: no answer trains the physical or
behavioral learner, converts a failed gate to a pass, approves an artifact, or
permits a safety exception. `Better` and `About same` can never override an
objective regression; `Not sure` leaves the future artifact provisional.

Profile revisions are opaque, monotonically increasing evidence generations,
not contiguous release numbers. Identical restored evidence produces the same
revision and hash; any additional accepted clean sample advances the next
qualified candidate. This lets later casual drives refine the profile without
rebasing accumulated physical statistics or letting highway data overwrite
low-speed nodes.

Measured evidence includes reachable, driver-free vehicle-owned torque
boundaries. Full-torque and maximum-slew frames are retained in a separate
speed-local authority stratum rather than being discarded. Slew transients
and stationary full-torque rows remain observations, not equality-fit rows;
only settled magnitude-boundary rows with resolved rack motion may join the
fit. Every response frame is paired
causally with the newest recorded torque effective no later than
`response time - seed transport delay`; it is never paired with the convenient
same-frame `carOutput`. `carOutput` itself reports the prior card cycle, so the
effective input clock is the preceding `carOutput` publication. Once aligned,
settled full-torque motion needs one response interval of command-side dwell,
not a second copy of the transport delay. Driver-limited and physically
impossible transitions remain excluded. This keeps sharp-turn/breakaway
evidence without turning limiter timing or human torque into false plant
parameters.

Vehicle steering-rate signals are also normalized before fitting. A natively
signed source keeps its sign. An unsigned magnitude source, including the
Palisade SAS rate, preserves the sensor magnitude and derives direction from
offset-corrected measured steering-angle motion. Quantized plateaus may retain
direction only within one continuous validity epoch. Because the observable
inverse map does not use rack acceleration, a valid measured reversal remains
a moving-response row and direction-coverage event without importing a
quantized acceleration impulse. It cannot manufacture a breakaway event from
stale dwell. Zero motion clears the unsigned-source direction latch; gaps,
disengagement, driver override, standstill, faults, and mapping failures break
cross-frame reversal continuity.
Every populated authority category is a no-regression surface. At least four
training authority rows are required before authority data may influence the
fit itself; if that happens, at least four independent validation rows are
required before qualification. Sparse authority evidence stays durable and
can reject a regressing model, but it cannot steer the fit.

The retired home-screen learning display read rebuildable
`BLaTv2LearningOperationStatus`, `BLaTv2BackfillProgress`,
`BLaTv2LearningStatus`, and `BLaTv2BehaviorLearningStatus` caches. The
offroad importer owned all four. Operation status distinguished logger
finalization, historical route scanning/replay progress, idle evidence, an
eligible empty state, and fail-closed diagnostics. Backfill progress shows
pass/route/segment plus read/apply progress and an evidence-based time estimate.

Physical learning-status schema 4 reports each speed node's total/base/moving/
breakaway/authority populations, train/validation state, rank and conditioning,
paired uncertainty, interpolation status, and—only when a fit exists—the four
observable values above. It distinguishes **learned**, **seed retained**,
**needs evidence/variety**, **ill-conditioned**, **inconclusive**, and
**validation regression** rather than calling every non-candidate unstable.
It never labels retired rack gain or damping as learned. It also publishes an
exact first-cause accounting for every prepared physical frame: accepted,
invalid numeric/timestamp, invalid vehicle input, lateral inactive, ineligible
speed/standstill, invalid live rack mapping, driver interaction, unavailable
causal alignment, measurement warm-up/discontinuity, learner-ineligible, or a
discarded breakaway episode. These counts explain evidence loss; they do not
change fitting or qualification.

Behavior learning-status schema 1 separately shows
`waiting_for_physical_profile`, `waiting_for_routes`, `preparing`, `training`,
`selecting`, `validating`, `publishing`, `complete`, or `failed`, together with
route/candidate/replay counts and the independent Smooth/Swift/Strong verdicts.
`complete` says either **qualified candidate available** or **stock retained**;
it never means activated. All displayed values remain informational, and the
UI cannot approve or select a controller.

These historical display caches were cleared at manager start and published
only after the offroad owner validated its phase and current vehicle/build
authority. Their schema and Params names remain reserved, but no current manager
process publishes them. They are
informational only: neither is evidence, approval, a profile,
controller-selection, or a safety input, and editing or deleting either one
cannot change steering. The `BLaTv2LifecycleStatus` schema and
`blatv2_profiled` implementation remain available for offline lifecycle
testing, but the stock-only field manager does not launch that process or
publish that cache.

Historically, on cold boot `blatv2_backfilld` published a
vehicle-bound **PREPARING LEARNER** projection immediately after it decodes
CarParams and before runtime construction, route discovery, PC inventory, or
uploads. Expensive preflight work therefore cannot be misreported as
**LEARNER STATUS UNAVAILABLE**; the cache still conveys no durable-learning
authority until authenticated evidence is restored or committed.

### Pre-merge real-route audit

Eighteen retained routes (`bd` through `d7`) were replayed twice from their
independently certified route-evidence artifacts. Both runs classified all
2,075,405 prepared frames identically: 820,768 accepted and 1,254,637 rejected.
The canonical counters formed 10 training routes and 8 validation routes. The
two runs produced byte-identical evidence
`cff67d0e984b2b8f8075e61193acd7ed9c5524273025acf27c362363dfc44dac`
and manifest
`400c8892ac5f5e41faeceeffa1a44269c10d17b0d058ace0e5fd3c48a70a7f98`.

The rejection ledger attributes 626,217 frames to lateral inactivity, 335,819
to unavailable causal command alignment, 260,313 to driver interaction or
allowance, 27,629 to measurement warm-up/discontinuity, 1,811 to invalid
numeric/timestamp input, 1,444 to learner eligibility, 1,401 to invalid vehicle
input, and 3 to discarded breakaway episodes; no frames failed live rack
mapping or the standstill/minimum-steer-speed gate in this set. The 0 through
20 m/s nodes all qualified by retaining their validated seed values. The
30 m/s node had five training but only three held-out breakaway episodes, below
the required four, so no six-node candidate was emitted and stock remained
selected.

---

<div align="center" style="text-align: center;">

<h1>openpilot</h1>

<p>
  <b>openpilot is an operating system for robotics.</b>
  <br>
  Currently, it upgrades the driver assistance system in 300+ supported cars.
</p>

<h3>
  <a href="https://docs.comma.ai">Docs</a>
  <span> · </span>
  <a href="https://docs.comma.ai/contributing/roadmap/">Roadmap</a>
  <span> · </span>
  <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Community</a>
  <span> · </span>
  <a href="https://comma.ai/shop">Try it on a comma four</a>
</h3>

Quick start: `bash <(curl -fsSL openpilot.comma.ai)`

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

</div>

<table>
  <tr>
    <td><a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772"></a></td>
    <td><a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c"></a></td>
    <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ce77dc6f63"></a></td>
  </tr>
</table>


Using openpilot in a car
------

To use openpilot in a car, you need four things:
1. **Supported Device:** a comma four, available at [comma.ai/shop/comma-four](https://www.comma.ai/shop/comma-four).
2. **Software:** The setup procedure for the comma four allows users to enter a URL for custom software. Use the URL `openpilot.comma.ai` to install the release version.
3. **Supported Car:** Ensure that you have one of [the 300+ supported cars](docs/CARS.md).
4. **Car Harness:** You will also need a [car harness](https://comma.ai/shop/car-harness) to connect your comma four to your car.

We have detailed instructions for [how to install the harness and device in a car](https://comma.ai/setup). Note that it's possible to run openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/), although it's not plug-and-play.


### Branches

Running `master` and other branches directly is supported, but it's recommended to run one of the following prebuilt branches:

| comma four branch      | comma 3X branch        | URL                                    | description                                                                         |
|------------------------|------------------------|----------------------------------------|-------------------------------------------------------------------------------------|
| `release-mici`         | `release-tizi`         | openpilot.comma.ai                     | This is openpilot's release branch.                                                 |
| `release-mici-staging` | `release-tizi-staging` | openpilot-test.comma.ai                | This is the staging branch for releases. Use it to get new releases slightly early. |
| `nightly`              | `nightly`              | openpilot-nightly.comma.ai             | This is the bleeding edge development branch. Do not expect this to be stable.      |
| `nightly-dev`          | `nightly-dev`          | installer.comma.ai/commaai/nightly-dev | Same as nightly, but includes experimental development features for some cars.      |

To start developing openpilot
------

openpilot is developed by [comma](https://comma.ai/) and by users like you. We welcome both pull requests and issues on [GitHub](http://github.com/commaai/openpilot).

* Join the [community Discord](https://discord.comma.ai)
* Check out [the contributing docs](docs/CONTRIBUTING.md)
* Check out the [openpilot tools](openpilot/tools/)
* Code documentation lives at https://docs.comma.ai
* Information about running openpilot lives on the [community wiki](https://github.com/commaai/openpilot/wiki)

Want to get paid to work on openpilot? [comma is hiring](https://comma.ai/jobs#open-positions) and offers lots of [bounties](https://comma.ai/bounties) for external contributors.

Safety and Testing
----

* openpilot observes [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines, see [SAFETY.md](docs/SAFETY.md) for more details.
* openpilot has software-in-the-loop [tests](.github/workflows/tests.yaml) that run on every commit.
* The code enforcing the safety model lives in panda and is written in C, see [code rigor](https://github.com/commaai/panda#code-rigor) for more details.
* panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
* Internally, we have a hardware-in-the-loop Jenkins test suite that builds and unit tests the various processes.
* panda has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
* We run the latest openpilot in a testing closet containing 10 comma devices continuously replaying routes.

<details>
<summary>MIT Licensed</summary>

openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
</details>

<details>
<summary>User Data and comma Account</summary>

By default, openpilot uploads driving data to our servers. You can also access your data through [comma connect](https://connect.comma.ai/). We use your data to train better models and improve openpilot for everyone.

openpilot is open source software, and users can disable data collection if they wish.

openpilot logs the road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The driver-facing camera and microphone are only logged if you explicitly opt-in in settings.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma for the use of this data.
</details>
