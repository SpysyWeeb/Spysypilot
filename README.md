# Universal Driving-Event Platform

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see
the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for
the full fork overview. This fork is entirely vibe-coded with
[Claude Code](https://claude.com/claude-code), is a personal project, and is
**not meant for others to use**; anyone who tries it does so at their own
risk.

Detailed implementation notes:
[`docs/DrivingEventPlatform.md`](docs/DrivingEventPlatform.md).

## Status

⚠️ **In progress.** The event pipeline, focused automated coverage, and
`combo` integration exist, but the platform remains in field validation. It
must not be considered done until its event detection, segment preservation,
off-road indexing, and review workflow have been verified on the device and
explicitly signed off.

🚧 **Automatic bad-stop detection is in progress.** The provisional
`long.stopJolt` detector, typed evidence, manifest serialization, and focused
tests are implemented on this branch. Its thresholds still require the
documented replay comparison and on-device field validation before this work
can be signed off.

## What it does

This branch is Spysypilot's shared evidence logger for driving behavior worth
reviewing. It accepts manual bookmarks and automatically detects selected
lateral and longitudinal behaviors, records structured evidence in the
authoritative full rlog, preserves the surrounding route segments, and builds
an on-device manifest that another orchestrator can inspect over SSH.

The platform is **observer-only**. It does not change the driving model,
planner, controller, CAN commands, panda safety behavior, or the behavior it
is measuring.

The word "snapshot" refers to an indexed event inside preserved full route
segments. The platform does not cut a standalone video clip, upload its
manifest, or directly notify another orchestrator.

### Event sources

| source | recorded event types | purpose |
|---|---|---|
| Manual sidebar flag | `manual.general` | Mark any moment the driver wants reviewed |
| BLaT lateral detector | `lat.stallRelease`, `lat.lateUnwind`, `lat.handoffMismatch`, `lat.centerOvershoot`, `lat.torqueAuthority` | Capture known steering failure shapes and their controller/actuator evidence |
| Lead-launch detector | `long.lateLeadLaunchPlanner`, `long.lateLeadLaunchController`, `long.lateLeadLaunchVehicle`, `long.leadLaunchStall` | Capture a delayed launch and attribute where the response chain lagged |
| Smooth-stop jolt detector | `long.stopJolt` | Capture an abrupt brake grab or brake-release rebound during the final low-speed landing |

Lateral detections include controller/reference versions, planned and measured
lateral acceleration, requested/applied/reference torque, steering rate,
driver input, road-bump confounders, unwind state, and event-specific evidence.
Lead-launch detections retain candidate, model forecast, plan, command, lead,
ego-motion, and ego-acceleration onset snapshots together with radar quality,
brake state, and timing between stages.
Stop-jolt detections retain the stop episode, standstill, peak, and detection
times; filtered IMU and wheel-speed-estimator jerk; before/after acceleration;
plan/request/applied acceleration snapshots and changes; longitudinal state;
lead state; driver inputs; and IMU, radar, brake-hold, and road-confounder
validity.

## How it works

### 1. `driving_eventd` observes and normalizes signals

`openpilot/selfdrive/spysypilot/driving_eventd.py` runs on-road for a real car.
It advances lateral detection only when `controlsState` updates and
longitudinal detectors only when `carState` updates. The stop-jolt car side
runs with the 100 Hz `carState` stream. Its IMU side advances only on an
actual 20 Hz `livePose` update, uses `livePose`'s own monotonic timestamp, and
never differentiates duplicated held samples. A manual
`bookmarkButton` update creates only the manual event at the button message's
monotonic timestamp.

Each accepted event receives:

- a stable UUID event ID;
- a group ID shared by evidence from the same physical episode;
- occurrence, detection, and episode-start monotonic times;
- detector/version, severity, confidence, attribution, and confounders;
- requested route context and analysis-window hints;
- the running git commit and branch.

Nearby generic events share a group for 2.5 seconds. Detector-defined episode
keys keep lateral maneuvers and lead launches grouped even when the underlying
episode lasts longer; each stop uses its approach start as its episode key.
The lateral, lead-launch, stop-jolt IMU, and stop-jolt car paths have separate
exception boundaries, so a failure in one detector cannot disable the others
or manual logging.

Accepted events remain in memory and are retried once per second with the same
event and group IDs until `loggerd` acknowledges them. `driving_eventd` does
not look up route names or write files.

### 2. `loggerd` is the recording authority

`loggerd` subscribes to `drivingEvent`, writes the exact message into the
active full rlog, assigns the authoritative route, segment, and segment-start
monotonic time, and publishes `drivingEventRecorded`.

For each new event it:

1. accepts the marker into the active rlog writer;
2. applies `user.preserve=1` to the current segment;
3. schedules the following segment for preservation after rotation;
4. publishes an acknowledgment containing the exact event ID and recording
   status;
5. deduplicates delivery retries by event ID.

Preservation failures enter a bounded retry queue. If a retry later succeeds,
`loggerd` republishes a corrected acknowledgment for the same event ID.

`markerAccepted` and its legacy alias `markerWritten` are low-latency
acceptance signals, not claims that the compressed rlog has already been
fsynced. Durable verification happens only when the off-road indexer later
finds the marker in a completed full rlog.

### 3. The UI reports logger-confirmed results

Both regular and mici on-road views listen only to
`drivingEventRecorded`—detector output alone never produces a success toast.

The notification renderer:

- shows manual, lateral, longitudinal, or failure-specific text;
- treats success as marker acceptance plus current-segment preservation;
- coalesces evidence from the same group;
- replaces an initial failure when a corrected acknowledgment arrives;
- queues every acknowledgment rather than conflating them;
- always yields to openpilot safety alerts.

### 4. The off-road indexer builds the review manifest

`openpilot/selfdrive/spysypilot/driving_event_indexer.py` runs while the device
is off-road. It reads completed **full rlogs**, not qlogs, and writes:

```text
/data/community/driving_events/manifest.jsonl
```

Each record contains the event/group IDs, exact route and segment, route,
segment, and marker offsets, detector-to-marker and marker-to-ack latency,
typed evidence, aggregate confounders, acknowledgment state, analysis-window
hints, and the names of context segments still present on the device.

Events and acknowledgments are joined by ID across adjacent segments, so a
segment rotation between marker acceptance and acknowledgment does not lose
recording status. A record's `verified_in_completed_rlog: true` is the durable
proof that the event marker was recovered from a completed rlog.

Normal scans prioritize preserved segments and remember completed work in an
atomically replaced, bounded processed-segment state file. To ignore that
optimization and reconstruct the manifest from every completed full rlog still
on the device:

```bash
python3 -m openpilot.selfdrive.spysypilot.driving_event_indexer --rebuild
```

### 5. Human review state stays independent

The manifest contains reconstructable event facts only. Review decisions are
append-only records in:

```text
/data/community/driving_events/reviews.jsonl
```

A review is keyed by `event_id` and may contain `status`, `labels`, `notes`,
`reviewer`, and `reviewed_at`. An event with no review entry is implicitly
unreviewed. Rebuilding or rotating the manifest never replaces or deletes the
review file.

This branch provides the review data contract, not a review UI or automatic
handoff service. Review orchestrators are expected to read the manifest and
preserved route data, then append their own review records.

## Preservation and retention

The requested evidence window is:

```text
two segments before → event segment → one segment after
```

`loggerd` marks the event segment and following segment. The deleter protects
two preceding segments around each marked segment, and the indexer repairs the
current/following xattrs after discovering a durable marker. Missing context
at the beginning/end of a route is reported accurately rather than invented.

Retention is intentionally bounded:

- the deleter keeps the ten newest xattr-marked segments, roughly five
  non-overlapping event windows;
- `manifest.jsonl` rotates near 2 MiB to `manifest.jsonl.1`;
- event IDs are deduplicated across the active and rotated manifests;
- processed-segment state is pruned to segments still present on the device;
- `reviews.jsonl` is never rotated or removed by the indexer.

Historical `/data/community/lat_events` and
`/data/community/long_events` indexes are left untouched.

## Detector behavior

### Lateral detector, version 3

The lateral detector is conservative and runs only while the torque controller
and lateral control are active. Driver steering immediately resets its
temporal detection state. Each event type has its own eight-second cooldown.

It detects:

- three steering stall-release cycles inside six seconds;
- an expected unwind that remains stalled for more than one second;
- a high-rate phase handoff with a large applied/reference torque gap;
- a fast center crossing while unwind braking still trails its target;
- sustained requested/applied torque saturation with growing tracking error.

Version 3 prevents stale stall arming from surviving inactive tracking and
keeps only 250 ms of transition hysteresis through the steering-rate boundary.
It also aggregates driver-torque, steering-press, and road-bump confounders
over the evidence window.

### Lead-launch detector, version 2

The lead-launch detector arms after at least 0.5 seconds stopped behind a valid
lead while openpilot longitudinal control is active. It records confirmed
model-forecast, plan-release, positive-command, lead-motion, and ego-motion
onsets.

A lead-to-ego delay over 0.25 seconds is classified by the first late stage:

- plan late or missing → planner;
- command late or missing after a timely plan → controller;
- timely plan and command but late ego response → downstream/vehicle response.

No ego movement for three seconds after confirmed lead movement becomes a
launch-stall event. Radar distance jumps or track-ID changes lower confidence
without hiding the event.

### Smooth-stop jolt detector, version 1

The pure `smoothStopJoltDetector` is observer-only and independent from the
lead-launch detector. It arms only when openpilot longitudinal control is
active, `longitudinalPlan.shouldStop` is true, required messages are valid,
the driver is not pressing either pedal, and an approach from above 0.5 m/s
crosses below 1.5 m/s. Hard braking before that low-speed landing is excluded.

Standstill must persist for about 0.2 seconds at `standstill` or no more than
0.05 m/s. The detector then observes another 0.45 seconds to retain both the
brake grab and brake-release rebound. A creeping queue therefore remains open
until the car actually stops. It emits at most one event and cannot re-arm
until longitudinal control resets or the vehicle moves above 1 m/s.

Both `carState.aEgo` and `livePose.accelerationDevice.x` are kept in short ring
buffers, smoothed over 0.3 seconds, and converted to jerk with a rolling
0.25-second slope using actual timestamps. A warning normally requires
same-sign IMU and aEgo peaks within 0.30 seconds, at least 3.0 and
2.5 m/s³ respectively, plus a meaningful acceleration change. A stricter
aEgo-only fallback is permitted when IMU data is unavailable; a wheel-speed
spike with valid but nonconfirming IMU data remains silent.

Replay also identified an IMU-dominant release shape in which the device sees
at least 3.0 m/s³ at standstill and the filtered aEgo estimate follows about
0.2 seconds later with a smaller sustained change. That narrowly scoped path
requires IMU confirmation, at least 0.5 m/s² of IMU acceleration change, and
at least 0.75 m/s³ plus 0.17 m/s² on aEgo; it is recorded at lower confidence.

Ordinary warning-level evidence overlapping the existing vertical-acceleration
bump classifier is suppressed. Severe evidence may still be recorded with
`roadConfounded=true` and reduced confidence. Any brake or gas intervention
during the landing blocks classification as a Smooth Stops failure. The
actual retained jolt peak is `occurredMonoTime`; finalization roughly
0.45 seconds later is `detectedMonoTime`. Requested review context is five
seconds before and two seconds after the peak.

Replay on the locally available full rlogs currently gives:

- route 22 segment 18: one critical `releaseSnap`, at the retained peak before
  delayed finalization;
- route 25 near 104, 493, and 670 seconds: three warning `releaseSnap` events;
- four automated post-fix stops on route 8d: three remain silent, while the
  stop near 428 seconds produces one lower-confidence borderline warning.
- route 92 (`00000092--c82a9cefa7`): 21 standstill transitions contain nine
  valid openpilot-controlled stop episodes. Seven remain silent. Two
  unconfounded lower-confidence `releaseSnap` warnings occur at route offsets
  2349.14 seconds (segment 39 + 9.14 seconds, IMU +3.40 m/s³ and aEgo
  +0.95 m/s³, controller-attributed) and 2809.18 seconds (segment 46 +
  49.18 seconds, IMU +3.26 m/s³ and aEgo +0.75 m/s³, mixed attribution).
  Driver review maps these to 09:34:26 and 09:42:06 respectively. The first
  was genuinely rough but situationally justified by traffic; retaining it
  as review evidence is acceptable rather than a false positive. The second
  was a confirmed unnecessarily harsh landing from a slow creep with ample
  stopping room. This labeled route therefore supports retaining the current
  lower-confidence boundary: it captures both rough landings while seven
  other eligible stops remain silent.

The complete requested population cannot yet be claimed: route 3c near 1343
seconds and route 38 near 972 seconds are no longer present on the device,
local disk, or the route listing returned for this dongle, and the post-fix
route-8d stops do not carry human good/bad labels. The thresholds therefore
remain provisional pending those archived route identifiers and manual review
of the borderline route-8d event. Route 92 now supplies positive field labels,
but broader field validation is still required before the detector can be
considered complete.

## Data model and compatibility

`openpilot/cereal/log.capnp` defines:

- `DrivingEvent` — the versioned observer event and typed evidence envelope;
- `DrivingEventRecorded` — `loggerd`'s route/segment/preservation
  acknowledgment.

Both services are logged to rlog and qlog. The retired `UserBookmark` and
`LateralEvent` schemas remain only so older routes can still be decoded. Their
old dedicated runtime loggers, indexers, and UI renderer are not part of this
platform. Legacy audio feedback may still emit a generic `userBookmark`.

Current versions:

| component | version |
|---|---:|
| universal event envelope | 2 |
| lateral detector | 3 |
| lead-launch detector | 2 |
| smooth-stop jolt detector | 1 |

## Main implementation files

- `openpilot/selfdrive/spysypilot/driving_eventd.py` — on-road orchestration,
  normalization, stable IDs, grouping, publication, and retry-until-ack.
- `openpilot/selfdrive/spysypilot/lat_event_detector.py` — pure lateral
  detector and signal conditioning.
- `openpilot/selfdrive/spysypilot/long_event_detector.py` — pure lead-launch
  detector and attribution.
- `openpilot/selfdrive/spysypilot/stop_jolt_detector.py` — pure low-speed
  stop-landing detector and typed evidence extraction.
- `openpilot/system/loggerd/loggerd.cc` — authoritative event write,
  deduplication, preservation, retry, and acknowledgment.
- `openpilot/selfdrive/spysypilot/driving_event_indexer.py` — completed-rlog
  scan, cross-segment join, manifest/review durability, and rebuild.
- `openpilot/selfdrive/ui/onroad/driving_event_notification.py` — shared
  acknowledgment toast queue.
- `openpilot/system/loggerd/deleter.py` — bounded preservation policy.
- `openpilot/cereal/log.capnp` and `openpilot/cereal/services.py` — event
  schemas and logged services.
- `openpilot/system/manager/process_config.py` — on-road detector and off-road
  indexer lifecycle.

Focused tests cover schema round trips, detector thresholds and cooldowns,
domain isolation, stable retry IDs, logger deduplication and preservation
retry, cross-segment acknowledgment joins, manifest rebuild/rotation,
independent reviews, UI queuing, and deleter behavior.

## Integration and current limitations

The stock-based platform was merged into `combo` in
[`8d663bf525`](https://github.com/SpysyWeeb/Spysypilot/commit/8d663bf525).
Changes should continue to land on this feature branch first and then be
integrated into `combo` without replacing combo-specific schemas, SOL/AOL
state, BLaT diagnostics, or other fork services.

Current boundaries to account for during validation:

- unacknowledged events live only in `driving_eventd` memory until `loggerd`
  accepts them;
- retention protects recent evidence, not every event forever;
- indexing waits until the device is off-road and the full rlog is complete;
- the platform has no built-in review UI, manifest uploader, or orchestrator
  notification transport;
- stop-jolt thresholds are provisional until the unavailable route-3c and
  route-38 rlogs and explicitly labeled good post-fix stops can complete the
  two-population replay comparison.

Automated tests are not a substitute for verifying real event timing,
preservation, manifest contents, and reviewability on the device.
