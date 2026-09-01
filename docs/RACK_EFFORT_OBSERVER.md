# Rack Effort Shadow Observer (RESO) — step 3-C

Log-only, zero-torque-authority hold-torque (H) shadow learner for the BLaTv3 rack controller.
It never subscribes to or writes `carControl.actuators`; its only outputs are two log messages
and a `Params` cache. See `Documents/spysypilot-route-audit/phase3/step3c_design/
shadow_learner_design.md` for the full design/panel synthesis this implements, and
`selfdrive/locationd/torqued.py` for the process shape it mirrors.

Code: `selfdrive/locationd/rack_effort_classifier.py` (pure-python core, shared with the offline
tooling below) and `selfdrive/locationd/rack_effort_observer.py` (the manager process).

## What it measures

Per `controlsState` tick (100 Hz), on frames where `lateralControlState.which() == 'rackState'`:
a physics prior `hPrior` (torqued's linear hold-torque model, reproduced scalar-for-scalar from
`rack_effort_seed/extract.py`'s `calc_curvature_vec`/`h_prior_vec`) is compared against
`hMeasured` (`carOutput.actuatorsOutput.torque`, the torque actually applied). A **steady
candidate** frame requires (byte-identical to extract.py's own definition, modulo the threshold
noted below): `|steeringRateDeg| < 2.0` and the steering angle range over a centered 31-frame
(0.30s) window `< 0.5deg`, `|hMeasured| < 0.95`, `|Δframe hMeasured| < 0.008`, window-contiguous
(no gap `> 3x` nominal dt), `carControl.latActive`, `!steeringPressed`, and driver torque below a
hands-off allowance. Because the window is centered, every published frame is delayed
`STEADY_HALF_FRAMES` (15) frames behind the input that produced it — this is a log-only observer,
so that lag has no control-loop cost.

**One deliberate deviation from extract.py**: the offline seed's hands-off allowance is 30
(native CAN units); the live gate uses the platform's own `STEER_DRIVER_ALLOWANCE = 50`
(`learner_note.md` section 4). `steeringTorqueRaw` is logged per frame so either cut can be
re-derived offline (see `verify2.py`).

A cell tuple is `(vBandIdx, angleBandIdx, latAccelBinIdx, direction)` — the same `V_BANDS` /
`ANGLE_BANDS` / 0.5 m/s² signed latAccel bins (clipped ±8, `GRID_VERSION=1`) and 0.05deg direction
deadband as `extract.py`. A contiguous run of steady frames sharing one cell is one **event**;
`runId` increments whenever the mask drops or the cell changes. Events shorter than
`MIN_RUN_FRAMES=3` are discarded; an event's sample is `median(hResidual)` over the run, never a
per-frame value (`learner_note.md` section 2).

## Freeze gates

No run may fold into a cell's `biasHat`, and freeze gates 1/2/4 additionally keep a frame from
ever becoming a steady candidate at all (structural — the frame is simply never published):

1. **Driver override** — `steeringPressed` or `|steeringTorque| >= STEER_DRIVER_ALLOWANCE`.
   `freezeBits` bit0 is computed against the *offline* 30 cut independently of whichever
   threshold gated candidacy, so a frame between 30 and 50 can still be logged (candidate under
   the on-device 50) while carrying bit0=1 (would fail the stricter offline cut) — this is what
   `steeringTorqueRaw` is for.
2. **Saturated** — `|hMeasured| >= 0.95` (bit1).
3. **Params moving** — torqued's own `lateralTorqueParameters.calPerc < 100`, or `useParams`
   flipped within the last ~1s (bit2). This does **not** affect candidacy/cellId/runId (needed
   for bit-exact agreement with extract.py) — it only prevents the run's median from being folded
   into `biasHat`, since `hPrior` is not a clean target while torqued's own fit is unsettled.
4. **Wrong controller family** — `which() != 'rackState'` (torqueState-tagged cars). Structural,
   checked before classification, not a bit.

`rackFallback` (bit3, the real `rack_log.fallback`) is logged per candidate frame but **does
not gate** in v1 — v1 mirrors `extract.py`'s own classifier exactly, which never reads
`.fallback` either; `H_surface.json`'s seed already includes fallback-steered frames, so adding
an exclusion here would silently break reconstructability against that corpus (see the panel
synthesis's `chosen`/`dissent` sections). `DEF_VERSION` is reserved for a future coordinated fix
(exclude fallback in both extract.py and here, regenerate the seed, bump the version).

H is not frozen below 15 m/s: that floor is G-only (`learner_note.md` section 4), and this
design carries no G state at all.

## Update rule

Count-capped EMA, indexed by event (not frame), `learner_note.md` section 3:

```
n = cell.n_events (after incrementing for this event)
alpha = 1 / min(n, N_CAP_H)     # N_CAP_H = 20
cell.bias_hat += alpha * (event_median_hResidual - cell.bias_hat)
```

`RackEffortAccumulator` (in `rack_effort_classifier.py`) implements this once; the live process,
`fit.py`'s replay mode, and every test call the same code so the two paths cannot silently
diverge. The live process only *reads* `RackEffortAccumulator.cells` for its periodic snapshot —
it never calls `.flush()` except at shutdown, because flushing mid-run would fragment one real
dwell-event into two artificially short ones purely from snapshot timing.

## Route identity

`RackEffortSnapshot.Cell.routeSketch` tags each event's cell with a hash of the current route, so
promotion logic (a later, separate step) can eventually require multiple distinct routes per
cell. Per the owner's default answer: the route identity is `route_id_hash(Params.get(
"CurrentRoute"))` (a real key, `common/params_keys.h`, `CLEAR_ON_MANAGER_START |
CLEAR_ON_ONROAD_TRANSITION`, `STRING`) — no new UUID-minting key was needed. `route_id_hash` uses
a fixed SHA-256-derived digest, not Python's salted `hash()`, so it is stable across restarts.
The sketch is capped at `ROUTE_SKETCH_CAP=16` distinct routes per cell; beyond that, additional
routes' events still update `n_events`/`bias_hat` but are not separately tracked in the sketch —
a documented lower-bound undercount, not an error.

## Versions and invalidation

Three independent version bytes travel with every `RackEffortSnapshot`:
- `gridVersion` (`GRID_VERSION=1`) — bumps only if `V_BANDS`/`ANGLE_BANDS`/bin width change.
- `defVersion` (`DEF_VERSION=1`) — bumps only if the mask/hPrior *definition* changes (e.g. a
  future fallback exclusion).
- `version` (`SNAPSHOT_VERSION=1`) — the snapshot schema itself.

Restore/invalidate mirrors `TorqueEstimator.get_restore_key` (`torqued.py:129-134`) exactly in
shape and in the try/except-and-`remove()`-on-corruption pattern (`torqued.py:118-125`): on boot,
`(CP.carFingerprint, GRID_VERSION, DEF_VERSION, SNAPSHOT_VERSION)` is compared against the same
tuple computed from the cached `CarParamsPrevRoute` + `LiveRackEffortObserver` Params blobs. A
mismatch starts empty and leaves the cache alone (not corruption); a decode exception removes the
cache. Per the owner's default answer, **no other invalidation trigger exists** — torqued itself
has no live "calibration changed" signal to crib one from, and no verified alignment/tire-service
Params key exists in this fork to hang one on.

`LiveRackEffortObserver` (`common/params_keys.h`, `PERSISTENT | DONT_LOG, BYTES`) mirrors
`LiveTorqueParameters` exactly; `CarParamsPrevRoute` is reused as-is for fingerprint identity.

**Note:** adding `LiveRackEffortObserver` to `params_keys.h` changes the compiled Params key
table (`common/libparams_c.so`); a device needs a real rebuild of that extension before this key
is usable — accepted, per the task brief.

## Messages

`RackEffortFrame` (`cereal/log.capnp`, Event union field 154) — one per qualifying candidate
frame: `version, runId, vBandIdx, angleBandIdx, latAccelBinIdx, direction, hMeasured, hPrior,
freezeBits, steeringTorqueRaw, calPerc, vEgo`.

`RackEffortSnapshot` (field 155) — every 60s and at shutdown: `version, gridVersion, defVersion,
epoch, cells[]`, each cell carrying `biasHat, nEvents, routeSketch[], lastUpdateMonoTime`.

Both are additive to the Event union (next free ordinals after `chestnutGpuState@153`); no
existing field changed. `cereal/services.py`: `rackEffortFrame` up to 100 Hz (published only on
qualifying frames), `rackEffortSnapshot` at 1/60 Hz.

**Known merge collision.** `combo`'s `log.capnp` has independently taken ordinals 154-157 for
`drivingEventRecorded`, `blatV2Shadow`, `lateralEvent`, and `chestnutGpuState` (combo-side
features not yet rebased onto BLaTv3, whose own `chestnutGpuState` sits at `@153`). Capnp
ordinals must be sequential with no holes, so BLaTv3 cannot dodge this by skipping ahead to an
ordinal combo hasn't used yet -- `154`/`155` are its only legal next values today. This is a real
collision that whichever side merges second must resolve by hand-renumbering its new fields past
the other's current max, per the project's usual sync-conflict convention (flag it, don't paper
over it with a silent push). Re-check combo's actual max ordinal at merge time; it keeps moving.

## Rebuilding H from logs

Two independent ways to audit that the live and offline definitions still agree, both living in
`Documents/spysypilot-route-audit/phase4/rack_effort_seed/`:

- **From raw rlogs** (`verify2.py <route> [seg ...]`): decodes a route's raw signals the same way
  `extract.py` does, recomputes `cellId`/`hPrior`/`freezeBits`/`runId` through
  `rack_effort_classifier.py`'s own scalar code (the live classifier itself, not a
  reimplementation of it), and diffs against whatever `rackEffortFrame` events that route's rlog
  actually contains (via `extract.py`'s additive decode). No route in the current corpus has
  on-device RESO data yet, so this reports "nothing to diff yet" today — it becomes a live check
  the moment a real device log carries these events.
- **From a frame stream alone** (`fit.py --replay <route>`): given only `rack_effort_frames/
  <route>.json` (extract.py's additive decode of `rackEffortFrame`), rebuilds `nEvents`/`biasHat`
  via the exact same `RackEffortAccumulator` and diffs against that route's own device-computed
  `rackEffortSnapshot` — the auditability guarantee that survives even a qlog-only pull where the
  raw frames are gone but the device's own snapshot isn't.

## Known deviations / not implemented in v1

- `carParams` is read once via a blocking `Params.get("CarParams", block=True)` (torqued.py's own
  pattern), not an ongoing `SubMaster` subscription — `CarParams` does not change mid-drive.
- The CPU/latency bench (tests_required item 6, comma-bench-onroad mocked-CarParams recipe) was
  not implemented — it requires real on-device timing measurement this environment cannot
  produce; the classifier's per-frame cost is a fixed-size scalar computation with no
  unbounded loops, but that claim has not been benchmarked on target hardware.
- G (rate-gain) bookkeeping is entirely out of scope, per the owner's step-3 decision — no G
  state, no G freeze gates, nothing to promote.
