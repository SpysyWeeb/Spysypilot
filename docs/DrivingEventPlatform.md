# Universal Driving-Event Platform

This observer-only platform records manual bookmarks and automatic BLaT/lead-launch
detections without changing controller, planner, model, CAN, or safety behavior.

## Data flow

`driving_eventd` normalizes current signals, advances lateral detection only on
`controlsState` updates and longitudinal detection only on `carState` updates,
assigns stable event/group IDs, and publishes `drivingEvent`. Bookmark-only updates
create only the manual event at the button message's monotonic timestamp. It never
looks up routes or writes files. Accepted events remain in memory and are retried
with the same ID until loggerd acknowledges them.

`loggerd` writes that exact event to the active full rlog, assigns the authoritative
route and segment, attempts `user.preserve` on the current segment, schedules the next
segment, and publishes `drivingEventRecorded`. The UI listens only to this
acknowledgment. Success requires both marker acceptance and current-segment
preservation; failures remain visible and preservation paths remain retryable.
When a failed preservation later succeeds, loggerd republishes a corrected
acknowledgment for the same event ID.

The old `userBookmark` and `lateralEvent` schemas remain solely for decoding old
routes. Their logger processes, indexers, and UI renderer have been removed. Audio
feedback may still emit a generic legacy bookmark.

## Off-road index

`driving_event_indexer` reads completed full rlogs, not qlogs, and appends:

```text
/data/community/driving_events/manifest.jsonl
```

Records include event/group IDs, exact route/segment and monotonic offset, typed
metrics, confounders, acknowledgment state, context segment names, and an initial
`unreviewed` state. The manifest rotates near 2 MiB and deduplicates across active
and rotated files. Processed-segment state is atomic and bounded to segments still
present on the device. Events and acknowledgments are joined by ID across adjacent
segments in a route, so a logger rotation between marker acceptance and
acknowledgment does not lose acknowledgment status.

Normal scans prioritize preserved segments. A complete reconstruction ignores that
optimization:

```bash
python3 -m openpilot.selfdrive.spysypilot.driving_event_indexer --rebuild
```

The deleter protects two preceding segments around each marked event segment. The
indexer repairs the current and following xattrs, retaining the requested
two-before/current/one-after window without consuming four preservation-quota
entries per event.

The lateral detector is version 2. Its thresholds and classifications are unchanged,
but its cooldown is intentionally per event type: a stall/release event no longer
hides a separate handoff mismatch in the same episode.

Historical `/data/community/lat_events` and `/data/community/long_events` indexes
are not deleted or overwritten.
