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

The lateral detector version 7 is in progress. Version 5 preserves version 4's
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

### Lateral detector version 6

Route `00000094--8cec74a749` field review of version 5 found four
calibration/bookkeeping problems, all fixed without touching steering
behavior: unwind-release timing keyed off literal zero torque instead of the
road's actual crown-neutral offset, an unbounded `turnStopTurn` dwell that
let one steady curve masquerade as a 7–32 second "event," episode/grouping
bookkeeping that let unrelated maneuvers merge into one group, and an
envelope `driverConfounded` flag that conflated "the driver touched the
wheel at some point" with "the driver caused this." `DETECTOR_VERSION` is
now 6; the universal event envelope (`EVENT_VERSION`) is now 3. New
`LateralPayload` fields are strictly appended at `@127`–`@165` (v5's highest
field was `@126`); every existing field, struct, and enum is unchanged, and
pre-v6 messages keep decoding with safe zero/false/empty defaults.

**Crown-neutral-aware unwind timing.** The steering rack's actual neutral
point (`torqueState.unwindNeutralTorque`, wired defensively as
`unwind_neutral_torque`, defaulting to 0.0 on stock schemas) commonly sits
0.13–0.21 away from true zero on route 94, so "torque crossed zero" and
"torque reached the road's real crown-neutral offset" are measurably
different moments. `unwind_phase_direction` (`torqueState.unwindPhase
Direction`) captures the original turn-torque sign at arm time; when it
isn't populated strongly enough (`OLD_TURN_SIGN_PHASE_DIRECTION_MIN = 0.5`)
the detector falls back to the arm-time steering-angle sign.
`old_direction_component(torque) = max((torque - unwind_neutral_torque) *
old_turn_sign, 0.0)` is recomputed every sample against the CURRENT
sample's neutral — the neutral slews (toward ~0.18 in under a second on this
route) and is never cached as a route constant. "Requested torque reached
crown-neutral" is `old_direction_component(request_torque) <
CROWN_NEUTRAL_TOLERANCE (0.05)` held continuously for `CROWN_NEUTRAL_HOLD_S
(0.10 s)`; the same machinery runs independently for `applied_torque`. The
four high-angle fields (`high_angle_unwind_scale`,
`torque_command_before_high_angle_exit`,
`high_angle_unwind_old_torque_correction`,
`high_angle_unwind_old_direction_torque`, all from `torqueState`) are
grouped under one availability check so they are all-`None` (schema absent)
or all-`float` together, and are never gated on `carOutput` damping
validity. The old literal-zero interpolated crossings
(`requestedTorqueNeutralCrossMonoTime`/`appliedTorqueNeutralCrossMonoTime`,
payload `@86`–`@89`) are retained byte-for-byte as separate diagnostics;
`unwindCommandDelayS` (`@90`) keeps its old literal-zero definition, and a
new `unwindCrownCommandDelayS` (`@143`) reports the crown-aware version. An
old-torque rebound tracker arms the moment requested torque first reaches
crown-neutral (no hold requirement to arm) and tracks
`unwind_rebound_max_magnitude`, `unwind_rebound_start_mono_time`, and
`unwind_rebound_duration_s` for any re-excursion back into the old turn
direction above `UNWIND_REBOUND_MIN_COMPONENT (0.10)` — the damped,
controller-generated "spring back" visible after most releases.

Crown-neutral presence is data-dependent, not guaranteed on every episode.
Route-94 replay found `requested_crown_neutral_mono_time` present on 5 of 7
unwind keeps (absent twice when requested torque was still saturated or
oscillating at emission, never sustaining the 0.05 hold) and
`applied_crown_neutral_mono_time` present on only 1 of the 4 no-assist keeps
(the applied-torque rebound is still decaying at the no-assist emission
deadline on the others). Both are physically correct outcomes of a fast
emission racing a still-settling signal, not bugs — do not expect either
crown timestamp on every event.

**turnStopTurn lifetime.** A dwell that exceeds `TURN_STOP_MAX_DWELL_S
(2.75 s)` is discarded outright — the episode is forgotten and no later
movement can still publish it — which is what stops a single steady curve
from surviving as a 7–32 second "event." Entering `"dwell"` now requires
`TURN_STOP_MIN_PRE_PROGRESS_DEG (10°)` of travel since the episode's own
creation angle, and a released episode's post-release progress
(`turn_stop_post_dwell_progress_deg`) must reach
`TURN_STOP_MIN_POST_PROGRESS_DEG (10°)` by its pending-candidate deadline or
the candidate is dropped unpublished. In the `"dwell"` state, the existing
strong-release check is evaluated FIRST; the direction-flip/center-drift
resets apply only to samples that do NOT satisfy that release condition.
This ordering was a real bug caught in design review: checking
direction-flip first would have silently discarded every toward-center or
opposite-direction (`"unwind"`/`"reversal"`) release, since those releases
are by definition a flip relative to the dwell's own movement direction.

**Episode identity.** A physical maneuver now closes when the wheel holds
inside `EPISODE_CENTER_CLOSE_DEG (15°)` for `EPISODE_CENTER_CLOSE_S (0.5 s)`
regardless of other activity bits, in addition to the existing 2 s
quiet-gap rule and a new `EPISODE_MAX_LIFETIME_S (20 s)` hard cap. After a
center-close, `tracking_active` alone cannot re-open the same episode while
the wheel stays settled near center — re-arming needs the wheel back
outside the center band or a real rate spike (> 8 deg/s) — which stops
close/re-open churn right at center. Each detector-side episode object
(`UnwindProgressEpisode`, `TurnStopEpisode`) snapshots its `episode_start`/
`episode_key` at the maneuver's PHYSICAL time (arm, or dwell-episode
creation), not at emission. This matters because emission lags the
physical occurrence by up to ~2–3 s (`turnStopTurn` emits ~2.1 s after
release; a no-assist unwind deficit emits ~2 s after deficit start) — long
enough for the live episode to have already center-closed or churned into a
different object by the time it emits. Using the snapshot instead of live
state at emission time means a late-published event still carries the group
key that was actually live when the maneuver happened, not whatever key the
tracker happens to hold two seconds later.

**Grouping.** `EventRecorder` groups by exact `episode_key` match: a keyed
candidate whose key is already bound to a live group always rejoins that
group within `GROUP_MAX_LIFETIME_NS (25 s)` — there is no separate
idle-staleness test on an existing key's binding. This is deliberate: with
occurrence-time key snapshots (above) and detector-side episode closure
(center-settle + 2 s quiet gap + 20 s cap) already guaranteeing that the
same key means the same physical maneuver, a recorder-side "close after
~2.5–3 s without related evidence" heuristic would be redundant with
closure that already happens on the detector side. Key `last_seen` refresh
is monotone (`max(last_seen, occurred_mono_time)`, never rewound) because
candidates from different detectors can arrive out of order relative to
their own occurrence times within one update cycle. A group that exceeds
`GROUP_MAX_LIFETIME_NS` mints a new group and re-binds the key to it.
`GROUP_KEY_PRUNE_NS (30 s)` is memory-only bookkeeping (drops stale key
entries from the dict) and carries no relatedness meaning. The keyless
(manual-style) 2.5-second chain window (`GROUP_WINDOW_NS`) is unchanged.
`GROUP_MAX_LIFETIME_NS (25 s)` is deliberately greater than
`EPISODE_MAX_LIFETIME_S (20 s)` plus emission-lag slack, so a group can
never legitimately need to outlive its own hard cap.

**Primary designation.** Designation only applies to manifest records with
`version >= DESIGNATION_MIN_VERSION (3)`; older records (v5-era and
earlier) always self-designate `group_role: "primary"` with
`primary_event_id` equal to their own `event_id`, so `--rebuild` can never
re-designate a pre-v6 group. For eligible records, `driving_event_indexer.py`
groups a route's scanned events by `group_id` and picks the
highest-priority member as primary (module table `GROUP_PRIMARY_PRIORITY`,
ties broken by earliest `occurred_mono_time` then lexicographic
`event_id`):

| priority | event type / condition |
|---|---|
| 110 | `manual.general` |
| 100 | `lat.unwindProgressDeficit` with `driver_assisted_unwind` or `driver_causation == "interventionBacked"` |
| 90  | `lat.unwindProgressDeficit`, otherwise |
| 80  | `lat.handoffMismatch`, `lat.centerOvershoot`, `lat.committedHandoffHarshness`, `lat.torqueAuthority` |
| 70  | `lat.turnStopTurn` and every `long.*` event |
| 60  | `lat.stallRelease`, `lat.lateUnwind` |

Because the indexer only writes a segment's records after that segment and
its following segment are both scanned, a candidate's designation pass
includes the following segment's already-scanned events as group members
for priority purposes (without writing their records). Since a group's hard
25-second lifetime is far shorter than a 60-second segment, a group spans
at most two segments, so incremental (as-you-go) designation matches a
from-scratch `--rebuild`. The known-primary map is loaded from both the
active and rotated manifest: a `group_role: "primary"` line claims its
group directly, and a `secondary` line's own `primary_event_id` also claims
the group (covering a secondary manifesting before its primary).
First-manifested-primary wins across drives — once a group has a
manifested primary, every later member becomes secondary referencing it
even if it would otherwise outrank it. The user-facing event count is the
number of `group_role != "secondary"` lines; the indexer's `cloudlog` line
reports both totals (`indexed N event(s) (M primary)`). No envelope schema
field was added for this — designation state lives only in the manifest,
and `--rebuild` recomputes it from scratch.

**Driver causation vs. driver rescue.** Five granular evidence fields —
`driver_active_before_deficit`, `driver_active_at_deficit_start`,
`driver_active_during_evaluation`, `driver_intervened_after_deficit`,
`driver_intervention_accelerated_progress` — plus
`driver_assist_raw_torque_only` (true when the assist path confirmed via
raw torque without `steeringPressed` ever becoming true) feed a
`driver_causation` string (payload `@160`, one of `"driverCreated"`,
`"interventionBacked"`, `"autonomousOnly"`, `"mixed"`, or `""` when not
computed — it is computed only for `unwindProgressDeficit` and
`turnStopTurn`). `driver_active_at_deficit_start` is intentionally
ONE-SIDED: it only looks at `[deficit_start - DRIVER_AT_START_WINDOW_S
(0.20 s), deficit_start]`; anything after deficit start is exclusively
`driver_intervened_after_deficit`'s job. This means a rescue that begins
immediately (even ≤0.2 s) after the deficit is flagged still classifies
`"interventionBacked"` rather than being swept into `"driverCreated"` by
the at-start window.

The envelope-level `driverConfounded` boolean is UNCHANGED for every event
type in v6 — it is still exactly `evidence.driver_confounded_any` (true if
the driver was confounded anywhere in the analysis window), the same
definition as v5. It was deliberately NOT repurposed to mean "caused,"
which would have silently regressed 6 of the 8 lateral event types and
broken existing consumers pinned to the old meaning. Consumers that want
causation-based filtering must read the new `payload.driver_causation`
field instead of the envelope flag. The daemon's existing `attribution`
override (forcing `"mixed"` when `confirmedSteeringPressed` fired) also
keeps its pre-v6 meaning — an intervention-backed rescue still carries
`attribution: "mixed"`, so consumers filtering on causation should join on
`driver_causation`, not `attribution`.

Route-94 validation found that the 4 driver-rescue anchors on this route
all classify `"driverCreated"`, not `"interventionBacked"` as originally
hoped: this route is hands-on-heavy enough (see below) that the driver was
confounded continuously through arm, deficit-start, AND evaluation on all
4, which satisfies the stricter `driverCreated` branch before the
`interventionBacked` branch is ever reached. `driver_assisted_unwind=True`
and confidence 0.95 are still correctly preserved on all 4 — the v5-era
rescue mechanism itself is unaffected; only the causation label differs
from the design's original example set. This is a property of this
specific route's driving style, not a defect in the classifier: the
`interventionBacked` branch (hands-off, then a genuine later rescue) simply
never gets exercised on a route where the driver rarely lets go.

**Road confounding.** The four road-confounding metrics
(`road_confounded_fraction`, `max_vertical_accel_deviation`,
`longest_road_confounded_duration_s`, `road_interaction`) are now computed
over the consolidated physical-episode window
(`min(episode_start_mono_time, evidence_start)` through `evidence_end`)
instead of the narrower per-event analysis window alone, and the new
`roadEvidenceWindowStartMonoTime` payload field reports that window's
actual start. `EVIDENCE_HISTORY_S` is 25.0 (was 15.0), sized for
`EPISODE_MAX_LIFETIME_S (20 s)` plus after-window and emission-lag slack;
the serialized window start is additionally clamped to whatever history
sample was actually available at emission, so it never claims coverage the
metrics didn't really have. `road_confounded_at_trigger` keeps its old
per-trigger meaning unchanged. One new bounded confidence adjustment lives
in the daemon's `lateral_candidate`, not the detector: when
`road_interaction == "substantial"`, confidence drops by
`ROAD_SUBSTANTIAL_CONFIDENCE_PENALTY (0.05)`, floored at
`ROAD_CONFIDENCE_FLOOR (0.50)` — the floor is an invariant, not a claim
that road confounding ever pushes confidence that low in practice on real
routes (the lowest emit-site literal is 0.80). The penalty is never applied
when the event is intervention-backed or `driver_assisted_unwind` — the
existing 0.95-confidence rescue literal always wins.

**Route-94 validation.** Route `00000094--8cec74a749` (device branch
`combo`, detector v5, 46 live manifest rows) was replayed offline through
the current (v6) detector, and its raw detections were also fed through the
real `EventRecorder` and indexer designation logic (not a reimplementation),
per the committed fixture
(`openpilot/selfdrive/spysypilot/tests/data/route94_lat_windows.csv.zst` /
`route94_lat_windows_meta.json`, exercised by `test_route94_replay.py`).
Findings are reported honestly rather than tuned to match the original
design text:

- All 7 unwind-deficit anchors (00:25.6 through 11:04.0) still fire within
  0.03 s of occurrence, plus the 10:07.56 driver-rescue extra. 7/7 keeps.
- Of the 5 v5-era long-dwell `turnStopTurn` "rejects," only 3 genuinely
  exercise v6-specific logic, each via a DIFFERENT mechanism:
  never-enters-dwell (mono 466.36 — the episode never reaches `"dwell"` at
  all near the anchor), center-band-drift-reset (mono 581.20 — a real
  dwell is entered but resets straight to no-episode when the wheel drifts
  back inside the center band, never reaching `"released"`), and
  state-fragmentation (mono 649.0 — the would-be single dwell fragments
  into several short moving/slowing/dwell/released cycles instead of one
  sustained hold). The other 2 anchors (mono ~141.79 and ~211.68) are
  pre-existing live-daemon-only artifacts: neither was ever reproducible in
  any offline replay, v5 or v6 alike, so they are excluded from the
  fixture rather than asserted as v6 proof — testing their absence would
  only show "still absent, same as v5," not exercise v6-specific logic.
- Of the 20 short-dwell (0.3–1.6 s) `turnStopTurn` keeps, 19 still fire
  under the new progress gates. The 20th (v5 anchor mono 735.289) is lost
  not to a gate failure — its dwell/release happens exactly on schedule
  and clears both progress gates comfortably — but because a genuinely
  new v6-only `turnStopTurn` at mono 731.977 fires 5.4 s earlier and
  consumes the shared 8-second per-type `EVENT_COOLDOWN` first. This is the
  accepted cooldown-consumption tradeoff: a cooldown-suppressed emission
  still consumes the episode, so this loss is not attributable to the new
  discard rules and is documented rather than hidden.
- Both expected old-torque rebounds reproduce with real magnitudes: 0.908
  (03:59.8 episode) and 0.440 (04:50.0 episode).
- The 4 driver-rescue anchors all classify `"driverCreated"`, not
  `"interventionBacked"` (see the causation section above) — this route is
  hands-on-heavy enough that the driver never actually lets go before any
  of these deficits start.
- Grouping collapsed the route's 45 raw v6 detector events into 19
  physical-maneuver groups with 19 primaries (compared to 46 raw v5-era
  live events and roughly 23 human-labeled maneuvers in prior review) —
  `unwindProgressDeficit` (priority 90/100) wins primary in effectively
  every group it participates in, absorbing neighboring
  `turnStopTurn`/`handoffMismatch`/`stallRelease`/`centerOvershoot`
  detections as secondaries. No observed group spanned more than ~15 s.

None of these findings required weakening any named constant; where reality
differed from the original design text's expectations (crown-neutral
presence, the causation label on the 4 rescue anchors), the difference is
reported rather than papered over. Version 6 remains in progress pending
new on-device field review, same as every prior version on this branch.

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

### Lateral detector version 7

Route `000000b7--a6b3b1f175` showed that the original
`lat.torqueAuthority` condition encodes a controller-specific assumption:
both requested and applied torque must remain above 0.95 for one second while
tracking error grows. That remains the correct saturated-authority event, but
it cannot see a controller that leaves authority unused. W2, the route's
tightest failed turn, peaked at only 0.616 requested torque while delivering
65% of demanded curvature. W3 peaked at 0.733 while delivering 85%. W1
briefly approached saturation, but only five of 739 hands-off frames reached
full authority; it did not satisfy the existing detector's sustained
saturation requirement.

Version 7 adds the complementary `lat.torqueUnderDelivery` condition. It
requires all of the following continuously for 1.0 second:

- absolute desired lateral acceleration above 0.35 m/s²;
- direction-normalized lateral-acceleration deficit above 0.15 m/s²;
- delivered fraction at or below 0.80; and
- requested-torque headroom of at least 0.25.

The headroom and delivered-fraction envelope was selected from a b7 threshold
sweep, not by feel. It is the strongest tested envelope that retained W2 and
W3 without flooding; the continuous replay emits 18 events, while looser
candidate envelopes emitted 25–34. W1, W2, and W3 all surface.

Version 7 also emits `lat.driverTakeover` when `steeringPressed` begins while
lateral control is active and remains asserted for 0.30 seconds. B7 contained
75 raw active press onsets and 41 lasting at least 0.30 seconds, matching the
route's roughly forty genuine override episodes while rejecting incidental
touches. The event occurrence time is the press onset, detection time is the
later confirmation, and it is emitted once per continuous press. Its severity
uses the preceding two seconds of demand, tracking deficit, delivery, and
headroom context. The event is attributed to the driver and never suppressed,
downgraded, or marked driver-confounded: the interaction is the evidence, not
contamination of other evidence.

The universal envelope is now version 4. New `LateralPayload` fields are
strictly appended at `@166`–`@171`: demanded curvature, delivered-curvature
fraction, torque headroom, signed tracking deficit, under-delivery duration,
and takeover-confirmation duration. Existing ordinals and meanings are
unchanged.

Full b7 replay preserves the version-6 census exactly (23
`turnStopTurn`, two `committedHandoffHarshness`, one `stallRelease`, and zero
`torqueAuthority`) and adds 18 `torqueUnderDelivery` plus 41
`driverTakeover`. The committed W1/W2/W3 fixture pins both new events at each
owner-identified failed turn. Detector-only timing over 121,102 b7 frames
moves from 1.46 to 1.95 microseconds median and 2.72 to 3.38 microseconds p99;
the added checks do not create a material on-device scheduling load.

`turnStopTurn` keeps its existing six-second-before/two-second-after
detection window. Gate-side phase extension already owns complete-phase
measurement; widening every on-device event would duplicate that solution
and increase preserved/uploaded context.

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
