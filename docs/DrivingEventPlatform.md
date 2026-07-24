# Universal Driving-Event Platform

This observer-only platform records manual bookmarks and automatic BLaT,
standstill lead-launch, rolling lead-response, and low-speed stop-jolt
detections without changing controller, planner, model, CAN, or safety
behavior.

## Data flow

`driving_eventd` normalizes current signals, advances lateral detection only on
`controlsState` updates, and advances the standstill launch, rolling lead, and
stop-jolt car paths only on 100 Hz `carState` updates. Stop-jolt IMU processing runs only for an actual
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

The lateral detector version 5 is in progress. It preserves version 4's
interpolated signed steering
center crossing only after the wheel was established outside 25 degrees, then
waits for desired-trajectory commitment evidence. A requested reversal may
produce one `committedHandoffHarshness`; an unrequested rapid crossing may
produce `centerOvershoot`. A phase handoff and crossing in the same episode
within 500 ms are consolidated before publication. The physical crossing and
later classification times are retained separately.

Version 5 adds `unwindProgressDeficit`, which compares expected reference
unwind progress with actual timestamped steering-angle reduction instead of
requiring a nearly stationary wheel, and `turnStopTurn`, which records one
strong moving/slowing/dwell/release transition without weakening the existing
three-cycle `stallRelease`. Driver assistance confirms and attributes a
progress deficit; it never suppresses it. Both new events are observer-only,
request six seconds before and two seconds after the trigger, and remain in
progress pending new field review. Route 93 replay is complete.

Route `00000093--3198e2f719` replay captures both requested unwind cases with
their physical deficit onset and later driver-assist detection separated. The
12:41 case arms around 12:41:20.34 MDT, retains the roughly −368.5-degree peak
and interpolated request/applied neutral crossings, then finalizes at the
12:41:22.45 steering press. The 12:38 case emits one strong
`turnStopTurn` around 12:38:32.96 with about 1.23 seconds of dwell near
214 degrees and near-full requested/applied/reference torque. Replay
acceptance does not mark v5 complete; new field logs still require review.

Driver and road evidence no longer suppresses lateral events. The typed payload
records raw-torque and confirmed-steering-press fractions/durations plus
trigger-time state, and separates transient from substantial road confounding.
Each three-release/six-second `stallRelease` contains three structured release
snapshots. Actual Hyundai damping evidence comes only from optional
`carOutput.actuatorsOutput.torqueDamping*` fields; absent/non-Hyundai fields are
marked invalid, and the controller D-term is never reused as actual damping.
Historical detector-v2 and detector-v3 fields remain append-only decodable.

The event envelope carries detector-defined episode keys. Lateral maneuvers,
standstill launches, and rolling lead responses therefore retain one group across overlapping semantic evidence even when
they outlast the generic 2.5-second manual correlation window.

The lead-launch detector is version 2. Trigger timing is unchanged; its compact
payload now retains candidate/forecast/plan/command/lead/ego/ego-acceleration onset
snapshots, brake state, and neutral wording for downstream/vehicle response.

The rolling lead-response detector is version 1 and is independent from the
standstill detector. It requires a continuously radar-matched track for at
least 0.75 seconds while ego is moving, rejects driver pedals and radar
discontinuities, keeps a rolling baseline, and confirms meaningful lead
commitment from lead speed/acceleration plus increasing relative speed and gap.
It then records sustained plan, applied-output, and ego response onsets for
about 2.5 seconds. One planner/controller/vehicle event is emitted only after
the gap and a named bad-response threshold are both met. Occurrence remains the
lead-motion onset; final evaluation remains the later detection time. Rearming
requires settled motion and a 25-second cooldown.

The smooth-stop jolt detector is version 1 and emits `long.stopJolt`. It retains
about two seconds of the final low-speed landing plus 0.45 seconds after sustained
standstill, smooths IMU and aEgo acceleration over 0.3 seconds, and calculates jerk
with actual-time rolling slopes. Its typed payload preserves peak and finalization
times, signed jerk and acceleration changes, plan/request/applied command evidence,
longitudinal and lead state, driver inputs, validity, and road-bump confounding.
Standstill launch, rolling lead-response, stop-jolt IMU, and stop-jolt car
exceptions are isolated from one another.

Saved-route replay currently verifies that route 92 segment 52 becomes one
committed handoff, segment 28 retains all three actual damping snapshots, and
segment 26 remains visible but substantially road-confounded. The older
route-8f segment 4 and 5/6 transitions remain separate because their matching
physical events are about 95 seconds apart. Route 92 pull-aways near 09:06:49
and 09:17:10 both cross rolling-lead trigger criteria, with replay evidence
currently attributing the lag downstream/vehicle rather than forcing the
provisional planner expectation.

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
