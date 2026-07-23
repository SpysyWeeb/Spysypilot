# Automatic Longitudinal Event Logger

## Purpose

Comma Connect is not the source of truth for this tool. Detection operates on the
device, loggerd preserves the original segment locally, and the final manifest points
to files under `/data/media/0/realdata` that can be retrieved directly over SSH.

Automatic classifications are review candidates, not unquestionable labels. A manual
bookmark is always recorded as ground truth from the driver's seat.

## On-road flow

`feedbackd` owns the single `userBookmark` publisher and runs the pure
`LeadLaunchDetector` on current logged signals. This keeps the detector outside all
control, planning, CAN, and safety processes.

For an openpilot-controlled stop behind a valid lead, the detector records:

- first sustained predicted lead departure;
- first sustained measured lead motion;
- first sustained MPC `shouldStop=False`;
- first sustained positive post-controller acceleration request;
- first sustained ego movement.

A launch is automatically logged when ego movement begins more than 0.25 seconds
after measured lead movement. The event is attributed in order:

1. planner, if the plan released more than 0.25 seconds after the lead;
2. controller, if the post-controller request arrived more than 0.25 seconds after;
3. vehicle response, if both were timely but the wheels were late.

Severity:

- 1: 0.25–0.75 seconds late;
- 2: 0.75–1.5 seconds late;
- 3: at least 1.5 seconds late, or no ego movement for 3 seconds.

Signals use a 0.15-second sustain window. Model forecast recognition uses 0.2 seconds.
A radar distance jump larger than 1.5 meters within 0.2 seconds or a radar track-ID
change lowers confidence to 0.55, but does not hide the event.

## UI behavior

Automatic and manual longitudinal captures display a silent, low-priority alert for
1.5 seconds:

```text
Long Event Logged
Late launch +1.6 s - vehicle response
```

The existing bookmark control creates:

```text
Long Event Logged
Manual bookmark
```

Higher-priority safety alerts remain dominant. Generic bookmarks used by audio
feedback retain the stock `Bookmark Saved` message.

## Local preservation and manifest

Publishing `userBookmark` immediately applies loggerd's existing `user.preserve`
attribute to the live segment. Once off-road, `long_eventd` scans completed rlogs and
also protects:

- the event segment;
- two segments before it;
- one segment after it, when present.

The indexer only decompresses segments already marked `user.preserve`, avoiding a
full rescan of ordinary drive logs. Events are appended to:

```text
/data/community/long_events/manifest.jsonl
```

Each JSON object includes:

- route and segment;
- event offset within the segment;
- manual or automatic source;
- type, severity, and confidence;
- forecast-to-lead, lead-to-ego, command-to-ego, plan-to-lead, and
  command-to-lead timing;
- exact preserved context segment names;
- an initial `unreviewed` status.

The indexer tracks completed segments in
`/data/community/long_events/processed_segments.json` and deduplicates by route,
segment, and monotonic timestamp.

## SSH retrieval

Inspect the queue:

```bash
ssh <device> cat /data/community/long_events/manifest.jsonl
```

Pull a listed segment:

```bash
rsync -av <device>:/data/media/0/realdata/<route--segment>/ ./long-events/<route--segment>/
```

The manifest deliberately does not clear preservation attributes after a pull.
Acknowledgement and retention controls should be added only with an explicit,
recoverable workflow so reviewing an event can never silently delete its source data.

## Initial scope

Version one recognizes lead-launch timing only. The structured event and manifest path
is intentionally general enough to add braking, acceleration-curve, lateral,
disengagement, override, FCW, and control-fault detectors without changing the storage
or UI pipeline.
