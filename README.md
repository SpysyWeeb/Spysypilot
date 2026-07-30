# custom-main-menu

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

## Status

**In progress.** The BLaTv2 dashboard is display-only. It does not activate,
approve, train, fit, reset, or otherwise influence a lateral controller.

## What it does

Replaces the "upgrade to comma prime" panel on the home screen with four
cycling windows. Tap the left half to go back or the right half to advance:

1. **BLaTv2 Learning** — a two-column grid of the vehicle's learned speed
   nodes. Each node shows credited clean support, its node-specific minimum,
   qualification state, and the last completed drive's contribution. A
   prominent banner reports live collection, finalization, retry, and
   historical-route processing without replacing the last validated snapshot.
2. **Readiness & Activation** — time, held-out validation, motion/excitation,
   and fit status for every node, plus an independent controller lifecycle
   rail.
3. **Live terminal** — a scrolling, colorized console of openpilot output.
4. **System usage** — CPU/RAM/power/fan history plus storage used/total.

The node list and speed labels are data-driven; current BLaTv2 vehicles use
six nodes at 0/5/10/15/20/30 m/s. Neighboring node values interpolate
continuously, so a node is not a hard speed-mode switch.

*(personal idea)*

## How it works

The two BLaTv2 pages share one rate-limited reader. It polls no faster than
once every two seconds and strictly decodes three versioned JSON Params caches:

- `BLaTv2LearningStatus` is an informational projection of persisted learner
  evidence and qualification reports.
- `BLaTv2LearningOperationStatus` is a separate, display-only projection of
  what the learner is doing now. It reports preparation, live collection,
  finalization, retry, deterministic historical-route backfill, completion,
  skip, and failure states. Backfill progress uses one-based route counts and
  cumulative accepted/rejected sample counts.
- `BLaTv2LifecycleStatus` is a sanitized projection produced by the owner of
  the validated activation state. The UI deliberately never parses
  `BLaTv2ActivationState` itself.

Both caches clear at manager start and remain unavailable until the current
build republishes them. The reader cross-checks their vehicle and runtime
identity. Missing, malformed, incompatible, or wrong-vehicle data is shown as
unavailable and never guessed.

Operational status never replaces persisted results. While a drive is being
finalized or older compatible routes are being replayed, both learning pages
continue to show the prior validated `BLaTv2LearningStatus` snapshot beneath
the processing banner. Missing operational and learning caches mean only
**status unavailable / awaiting learner**; they do not prove that the vehicle
has never driven. The first-drive instruction appears only when the learner
explicitly publishes `ready_no_evidence`.

The UI never parses rlogs, evidence, manifests, or profiles. It never trains,
fits, stages, approves, resets, or writes learning state. A full time bar means
only that the clean-support minimum is met; validation, steering variety, and
a physically valid fit remain separately visible. Likewise,
`all_nodes_qualified` means a complete fit exists, not that it is steering.
Only `BLaTv2LifecycleStatus` may label a controller provisional or approved.

## Status semantics

- **Blue:** collecting clean evidence.
- **Amber:** time is complete but validation, excitation, or the fit is still
  blocked.
- **Green:** the individual node is qualified.
- **Red:** an actual fit rejection, corrupt snapshot, or rollback condition.
- **Gray:** no evidence or unavailable current-build status.

Learner operation colors follow the same vocabulary: blue for active
preparation/collection/processing, amber for a pending retry or an
identity-change drive skip, green for ready persisted evidence, and red for an
explicit backend failure or rejected status schema. An identity-change skip
states that the learner is prepared for the next drive. Historical routes
older than the committed append-only watermark are reported as safely skipped;
legacy evidence without a route ledger reports backfill unavailable rather
than risking double-counting.

The activation rail is:

`Collecting → Complete profile → Replay/safety approval → Provisional → Approved`

Stock continues steering unless the lifecycle projection explicitly reports a
validated provisional or approved modular profile. A staged profile still
shows **Stock active**; rollback pending also reports effective stock.

## What changed

- `openpilot/selfdrive/ui/layouts/home.py` — four-page carousel with two BLaTv2
  pages followed by the existing terminal and system pages.
- `openpilot/selfdrive/ui/widgets/blatv2_learning_status.py` — dependency-free,
  strict learning, operation, and lifecycle display-schema parsers plus
  formatting/layout helpers.
- `openpilot/selfdrive/ui/widgets/blatv2_learning.py` — the two pure-reader
  pages and shared two-second cache.
- `openpilot/common/params_keys.h` — rebuildable learning, operation, and
  lifecycle display-status JSON keys.
- Removed the five route-analyzer widgets, `drive_statsd`, its process
  registration, and its now-unused `Spysy*Stats` Params.
- `terminal_widget.py`, `system_stats.py`, and their behavior are unchanged.
