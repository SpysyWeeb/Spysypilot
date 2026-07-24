# Universal Driving-Event Platform

This observer-only platform records manual bookmarks and automatic BLaT,
lead-launch, and low-speed stop-jolt detections without changing controller,
planner, model, CAN, or safety behavior.

## Data flow

`driving_eventd` normalizes current signals, advances lateral detection only on
`controlsState` updates, and advances the lead-launch and stop-jolt car paths only
on 100 Hz `carState` updates. Stop-jolt IMU processing runs only for an actual
20 Hz `livePose` update and uses that message's monotonic timestamp. It assigns
stable event/group IDs and publishes `drivingEvent`. Bookmark-only updates create
only the manual event at the button message's monotonic timestamp. It never looks
up routes or writes files. Accepted events remain in memory and are retried with
the same ID until loggerd acknowledges them.

`loggerd` accepts that exact event into the active full rlog, assigns the authoritative
route, segment, and segment-start monotonic time, attempts `user.preserve` on the
current segment, schedules the next segment, and publishes `drivingEventRecorded`.
The UI listens only to this acknowledgment. Success requires both marker acceptance
and current-segment preservation; failures remain visible and preservation paths
remain retryable.
When a failed preservation later succeeds, loggerd republishes a corrected
acknowledgment for the same event ID.

`markerAccepted` (and the legacy `markerWritten` alias) is deliberately a low-latency
acceptance signal. It does not claim that the file was fsynced. The off-road indexer's
discovery of the marker in a completed full rlog is the durable verification, exposed
as `verified_in_completed_rlog`.

The old `userBookmark` and `lateralEvent` schemas remain solely for decoding old
routes. Their logger processes, indexers, and UI renderer have been removed. Audio
feedback may still emit a generic legacy bookmark.

## Off-road index

`driving_event_indexer` reads completed full rlogs, not qlogs, and appends:

```text
/data/community/driving_events/manifest.jsonl
```

Records include event/group IDs, exact route/segment, separate route/segment/marker
offsets, detector-to-marker and marker-to-ack latency, typed evidence, aggregate
confounders, acknowledgment state, analysis-window hints, and context segment names.
The manifest rotates near 2 MiB and deduplicates across active and rotated files.
Processed-segment state is atomic and bounded to segments still present on the
device. Events and acknowledgments are joined by ID across adjacent segments in a
route, so a logger rotation between marker acceptance and acknowledgment does not
lose acknowledgment status.

Normal scans prioritize preserved segments. A complete reconstruction ignores that
optimization:

```bash
python3 -m openpilot.selfdrive.spysypilot.driving_event_indexer --rebuild
```

The deleter protects two preceding segments around each marked event segment. The
indexer repairs the current and following xattrs, retaining the requested
two-before/current/one-after window without consuming four preservation-quota
entries per event.

The lateral detector is version 3. Its thresholds, three-release/six-second policy,
and per-event-type cooldown are unchanged. Version 3 fixes stale `stallRelease`
arming by clearing the current arm as soon as tracking becomes inactive and retaining
only a 250 ms transition hysteresis through the 8–30 deg/s wheel-rate band.

The event envelope carries detector-defined episode keys. Lateral maneuvers and lead
launches therefore retain one group across overlapping semantic evidence even when
they outlast the generic 2.5-second manual correlation window.

The lead-launch detector is version 2. Trigger timing is unchanged; its compact
payload now retains candidate/forecast/plan/command/lead/ego/ego-acceleration onset
snapshots, brake state, and neutral wording for downstream/vehicle response.

The smooth-stop jolt detector is version 1 and emits `long.stopJolt`. It retains
about two seconds of the final low-speed landing plus 0.45 seconds after sustained
standstill, smooths IMU and aEgo acceleration over 0.3 seconds, and calculates jerk
with actual-time rolling slopes. Its typed payload preserves peak and finalization
times, signed jerk and acceleration changes, plan/request/applied command evidence,
longitudinal and lead state, driver inputs, validity, and road-bump confounding.
Lead-launch, stop-jolt IMU, and stop-jolt car exceptions are isolated from one
another.

## Reviews

`manifest.jsonl` contains only reconstructable event facts. Human review state is an
independent append-only file:

```text
/data/community/driving_events/reviews.jsonl
```

Each review line is keyed by `event_id` and may contain `status`, `labels`, `notes`,
`reviewer`, and `reviewed_at`. Rebuild mode never rotates, replaces, or deletes this
file, so regenerating the manifest cannot erase review work. An event with no review
entry is implicitly unreviewed.

Historical `/data/community/lat_events` and `/data/community/long_events` indexes
are not deleted or overwritten.
