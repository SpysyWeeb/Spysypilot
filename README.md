# custom-main-menu

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

## Status

**In progress.** The BLaTv2 dashboard is display-only. It does not activate,
approve, train, fit, reset, or otherwise influence a lateral controller.

## What it does

Replaces the "upgrade to comma prime" panel on the home screen with four
cycling windows. Tap the left half to go back or the right half to advance:

1. **BLaTv2 Learning** — a two-column grid of the vehicle's learned speed
   nodes. Each node shows overall clean support, independent base/moving/
   breakaway/authority evidence, its node-specific minimum, its explicit
   evaluation outcome, and the last completed drive's contribution. Separate
   summary rows report node outcomes, candidate-artifact availability, and
   interpolation qualification without conflating them. A
   prominent banner reports live collection, finalization, retry, and
   historical-route processing without replacing the last validated snapshot.
   A separate behavior panel reports whether a homogeneous route cohort is
   waiting, training, selecting, or undergoing held-out validation. Physical
   calibration and behavior qualification are deliberately never presented as
   the same operation.
2. **Readiness & Activation** — moving, breakaway, held-out validation,
   authority evidence, and the learner's plain-language result for every
   node. When a candidate exists it shows the observable inverse-torque gain,
   signed lateral-acceleration offset, and kinetic/static friction values,
   plus interval-level interpolation status. Independent **Smooth**, **Swift**,
   and **Strong** behavior gates appear below the physical matrix, followed by
   the independent controller lifecycle rail. A qualified behavior candidate
   is explicitly informational; only lifecycle status can say what is steering.
3. **Live terminal** — a scrolling, colorized console of openpilot output.
4. **System usage** — CPU/RAM/power/fan history plus storage used/total.

The node list and speed labels are data-driven; current BLaTv2 vehicles use
six nodes at 0/5/10/15/20/30 m/s. Neighboring node values interpolate
continuously, so a node is not a hard speed-mode switch.

*(personal idea)*

## How it works

The two BLaTv2 pages share one rate-limited reader. It polls no faster than
once every two seconds and strictly decodes versioned JSON Params caches:

- `BLaTv2LearningStatus` is an informational projection of persisted learner
  evidence and qualification reports.
- `BLaTv2LearningOperationStatus` is a separate, display-only projection of
  what the learner is doing now. It reports preparation, live collection,
  finalization, retry, deterministic historical-route backfill, completion,
  skip, and failure states. Backfill progress uses one-based route counts and
  cumulative accepted/rejected sample counts.
- `BLaTv2BackfillProgress` adds segment, pass, work-unit, and estimated-time
  detail for the current physical replay operation only.
- `BLaTv2OffdeviceProgress` independently identifies PC processing, prepared
  artifact download, ARM certification, prepared-data handoff, or an explicit
  local-fallback reason. During PC processing it retains the device-restamped
  pass/route/segment coordinate; after handoff the newer local replay status
  retakes the display. Archive-only PC rejections appear only as an unverified
  exclusion count and are never presented as learner evidence.
- `BLaTv2BehaviorLearningStatus` is the independent, informational-only
  behavior transaction projection. It reports homogeneous-route readiness,
  training and held-out-validation replay progress, immutable result hashes,
  and separate Smooth/Swift/Strong verdicts. A successful result means only
  that a candidate is available for external review; it does not stage or
  activate that candidate.
- `BLaTv2LifecycleStatus` is a sanitized projection produced by the owner of
  the validated activation state. The UI deliberately never parses
  `BLaTv2ActivationState` itself.

These caches clear at manager start and remain unavailable until the current
build republishes them. The reader cross-checks the common vehicle identity and
strictly validates each runtime hash within its own schema. Missing, malformed,
incompatible, or wrong-vehicle data is shown as unavailable and never guessed.

Learning schema v3 uses the observable-calibration runtime identity, behavior
status uses the full runtime-vehicle replay identity, and lifecycle status uses
the independently gated live-controller artifact identity. The UI strictly
validates every hash and their common vehicle identity, but does not equate
these deliberately separate identity namespaces.

Schema v3 keeps three decisions independent. A node may be **Learned** or
**Seed retained (calibration already good)**; either is a successful node
evaluation. Neighboring-node interpolation is then evaluated separately.
Finally, candidate-artifact availability says only whether learning produced
new calibration bytes. If every node and interval qualifies by retaining the
seed, completion is green and the UI says no new artifact is needed—it does
not report an unavailable snapshot or an error.

Operational status never replaces persisted results. While a drive is being
finalized or older compatible routes are being replayed, both learning pages
continue to show the prior validated `BLaTv2LearningStatus` snapshot beneath
the processing banner. Missing operational and learning caches mean only
**status unavailable / awaiting learner**; they do not prove that the vehicle
has never driven. The first-drive instruction appears only when the learner
explicitly publishes `ready_no_evidence`.

The UI never parses rlogs, evidence, manifests, or profiles. It never trains,
fits, stages, approves, resets, or writes learning state. A full time bar means
only that the clean-support minimum is met; moving-rack, breakaway, authority,
held-out validation, steering variety, and a valid observable calibration
remain separately visible. Likewise, `all_nodes_qualified` means every node
has a successful outcome and every interpolation interval qualifies, not that
new calibration was required or that it is steering.
Only `BLaTv2LifecycleStatus` may label a controller provisional or approved.
Behavior qualification cannot advance the activation rail and the dashboard
never opens route logs, evidence, profiles, or behavior artifacts to infer a
different result.

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

Behavior gates stay independent: **Smooth** covers command roughness and burst
quality, **Swift** covers signed turn-in/release timing without rewarding early
corner-cutting, and **Strong** covers delivered authority and tracking. A green
behavior candidate must pass all three plus the material-improvement target on
its frozen validation routes. A stock-retained result is a valid completed
transaction, not a missing snapshot and not permission to activate a weaker
candidate.

The activation rail is:

`Collecting → Complete profile → Replay/safety approval → Provisional → Approved`

Stock continues steering unless the lifecycle projection explicitly reports a
validated provisional or approved modular profile. A staged profile still
shows **Stock active**; rollback pending also reports effective stock.

## What changed

- `openpilot/selfdrive/ui/layouts/home.py` — four-page carousel with two BLaTv2
  pages followed by the existing terminal and system pages.
- `openpilot/selfdrive/ui/widgets/blatv2_learning_status.py` — dependency-free,
  strict physical-learning, behavior-learning, operation, and lifecycle
  display-schema parsers plus formatting/layout helpers.
- `openpilot/selfdrive/ui/widgets/blatv2_learning.py` — the two pure-reader
  pages and shared two-second cache.
- `openpilot/common/params_keys.h` — rebuildable learning, operation, and
  lifecycle display-status JSON keys.
- Removed the five route-analyzer widgets, `drive_statsd`, its process
  registration, and its now-unused `Spysy*Stats` Params.
- `terminal_widget.py`, `system_stats.py`, and their behavior are unchanged.
