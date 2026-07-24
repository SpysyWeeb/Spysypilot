# Universal Driving-Event Platform

This observer-only platform records manual bookmarks and automatic BLaT/lead-launch
detections without changing controller, planner, model, CAN, or safety behavior.

## Data flow

`driving_eventd` normalizes current signals, runs the pure lateral and longitudinal
detectors independently, assigns stable event/group IDs, and publishes
`drivingEvent`. It never looks up routes or writes files.

`loggerd` writes that exact event to the active full rlog, assigns the authoritative
route and segment, attempts `user.preserve` on the current segment, schedules the next
segment, and publishes `drivingEventRecorded`. The UI listens only to this
acknowledgment. Success requires both marker acceptance and current-segment
preservation; failures remain visible and preservation paths remain retryable.

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
present on the device.

Normal scans prioritize preserved segments. A complete reconstruction ignores that
optimization:

```bash
python3 -m openpilot.selfdrive.spysypilot.driving_event_indexer --rebuild
```

The deleter protects two preceding segments around each marked event segment. The
indexer repairs the current and following xattrs, retaining the requested
two-before/current/one-after window without consuming four preservation-quota
entries per event.

Historical `/data/community/lat_events` and `/data/community/long_events` indexes
are not deleted or overwritten.
