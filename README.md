# Spysypilot

This fork is **entirely vibe-coded using [Claude Code](https://claude.com/claude-code)** — including this README.

It's a personal side project for SpysyWeeb. It is **not meant for others to use**, but it's available for anyone who wants to try it **at their own risk**.

Any and all code and features generated in this project are free for others to use. SpysyWeeb can't take credit for the code itself, but if you build on an idea from here, a little credit for the idea would be appreciated. 🙏

## Branches

- **`stock`** — clean commaai/openpilot base, no changes
- **`combo`** — all features merged together for testing
- Feature branches are cut from `stock` and merged into `combo` when ready; each feature branch's own README explains that feature in depth

### Combo variants

Copies of `combo` that differ only in the driving model they run:

| branch | driving model |
|---|---|
| [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) | stock comma release model |
| [`deep-rl3-combo`](https://github.com/SpysyWeeb/Spysypilot/tree/deep-rl3-combo) | comma's [`deep_rl3`](https://github.com/commaai/openpilot/tree/deep_rl3) experimental RL model |
| [`divided-rl-combo`](https://github.com/SpysyWeeb/Spysypilot/tree/divided-rl-combo) | comma's [`divided-rl`](https://github.com/commaai/openpilot/tree/divided-rl) experimental RL model |
| [`michael-rl-combo`](https://github.com/SpysyWeeb/Spysypilot/tree/michael-rl-combo) | comma's [`michael-rl`](https://github.com/commaai/openpilot/tree/michael-rl) experimental RL model |

[`rl-combo`](https://github.com/SpysyWeeb/Spysypilot/tree/rl-combo) is a special case: it runs comma's Rebel Legion release model but is **not** a pure copy of `combo` — it also merges upstream `commaai/openpilot` master ahead of `stock`, plus small tuning deltas (e.g. `LONG_SMOOTH` 0.0), so expect it to differ beyond the model swap.

## BLoTv2 Conditional Experimental Mode

**Status: in progress; awaiting field validation.** Combo normally remains in
Chill mode, then hands its existing longitudinal planner to Experimental mode
for a confirmed, lead-free model stop prediction. It holds that handoff through
the stop and returns to Chill after a stable release or driver override.

There is no standalone traffic-light or stop-sign classifier in this build, so
the detector uses the model's existing `action.shouldStop`, path-end, and
desired-acceleration signals. The effective state has one owner and publishes
through `selfdriveState.experimentalMode`, which drives both the stock on-road
icon and planner strategy. The feature adds no UI or user-facing Params and
does not own target speed or braking. The implementation, signal mapping, and
ownership boundaries are documented in
[`docs/BLoTv2.md`](docs/BLoTv2.md#conditional-experimental-mode).

## BLaTv2 modular replacement

**Status: in progress.** The previous LQI-based BLaTv2 controller is retired.
LQI means “Linear Quadratic Integral”: state feedback plus an integral-error
state. The replacement is a ground-up, modular adaptive torque controller
whose target is **Smooth. Swift. Strong.**

This milestone deliberately has no approved learned profile or calibration
activation path. The active lateral controller remains the exact current stock
`LatControlTorque`, and no BLaTv2 helper runs onroad. On the validated
Palisade/Telluride platform that stock request uses the vehicle-selected
409/4/7 opendbc/panda envelope; other vehicles keep their own stock limits.

After logger closure, the offroad-only `blatv2_backfilld` replays compatible
complete full rlogs twice. It commits evidence only when both passes agree
bit-for-bit. Learning is measured-response-only at 0/5/10/15/20/30 m/s, and a
sample updates only neighboring nodes, so highway mileage cannot overwrite
low-speed knowledge. The current `complete_full_rlog_authority_v4` namespace
starts empty and never migrates or rewrites retired v1/v2/v3 evidence.

The retired dynamic-rack learner is replaced by an observable inverse-torque
fit: torque per measured lateral acceleration, a signed acceleration-offset
correction, moving friction, and static breakaway. Base, moving, breakaway,
held-out validation, and vehicle-owned full-authority evidence remain separate.
Breakaway is reconstructed as one physical episode: raw steering-angle motion
at half a rate quantum marks onset, and same-direction measured rate must
confirm it within the existing transport delay. Whole routes alternate between
training and validation by immutable route counter, so one maneuver cannot
train and validate itself. Training evaluates nested static-only, friction,
offset-plus-friction, and full-map models. The seed is comparator-only; a model
must improve something without regressing any populated category before its
frozen held-route check. Dense ordinary samples therefore cannot outvote rare
breakaway or authority failures.

A candidate file is emitted only if every node has enough independent evidence
and beats or matches its seed on every applicable held-out population. It is
still informational and cannot populate `BLaTv2ApprovedArtifact`, stage a
controller, or change stock selection. Any future consumer requires separate
raw/applied/delivered replay, deterministic A/A, safety, and device-timing
review.

The home-screen learning display reads rebuildable
`BLaTv2LearningOperationStatus` and `BLaTv2LearningStatus` caches. They clear
at manager start and are republished only by the offroad owner after validating
the current vehicle/build state. The operation cache distinguishes logger
finalization, historical scanning/replay, idle evidence, and fail-closed
diagnostics. Both caches are informational only: editing or deleting them
cannot train, approve, select, or change a steering controller. The retained
profile-lifecycle code is an offline test surface and is not manager-launched
while stock remains mandatory.

No BLaTv1 controller code or `HyundaiLowSpeedTorqueDamping` is inherited.
Vehicle limits come from runtime `CarControllerParams`; BLaTv2 contains no
Palisade limit literals. Full architecture, contracts, learning policy,
acceptance gates, and rollback behavior are documented in
[`docs/BLATV2_MODULAR.md`](docs/BLATV2_MODULAR.md) and
[`docs/BLATV2_ACCEPTANCE.md`](docs/BLATV2_ACCEPTANCE.md).

## BLaTv2 learning dashboard

**Status: in progress.** The custom home panel now cycles through exactly four
pages: **BLaTv2 Learning**, **Readiness & Activation**, the existing live
terminal, and the existing system-usage view. The five old route-analyzer
pages and their `drive_statsd` service have been removed.

The Learning page distinguishes active processing from an empty evidence set
and shows overall plus last-drive progress, with base/moving/breakaway/
authority populations at each speed node. Readiness shows independent motion,
breakaway, validation, and authority state and, when a fit exists, compact
gain/offset/kinetic/static values. A complete calibration never implies
activation; the lifecycle rail reports stock until a separately reviewed
activation path exists.

The historical learner registry now includes the verified 409/4/7 combo
builds recorded by routes d2-d6 (`3849a2f`, `2447667`, and `9338f5b`) plus
`fdd5560`, the clean combo build running immediately before the off-device
bridge landed. Older archived 384/3/7 builds remain fail-closed rather than
being mislabeled to admit their data.

### Off-device preparation bridge

**Status: in progress.** When its separately versioned worker is reachable on
the trusted private LAN, the comma may offload only deterministic full-rlog
preparation to `/home/alex/Documents/blatv2-remote-worker`. Uploaded segments
remain private and resumable until an authenticated exact route manifest
atomically publishes the whole route, so an interrupted prefix never appears
complete.

The device still freezes the manifest, runs both learner authorities, checks
A/A equality, owns the ledger/finalizer, and is the only profile publisher.
The PC has no Params, controller-selection, or actuation API. If the worker is
unavailable, the unchanged four-worker local path runs instead. See the BLaTv2
architecture and acceptance documents for the complete trust boundary.

The pre-merge b7/b8/b9/ca production replay was byte-identical with two and
four workers: generation
`1ff42b06de6480fc1e744702175167bd725849321e6e99d3e714596e16e56809`,
evidence `ad59ed7bc85a6d06d164185673110c592df80b44971377fe86b0bb4204993aa9`,
and manifest `3c0f072a6af25d44ae447a19b0011ed27435fb21d523cdd7d607b39f1f384631`.
Nodes 10 and 15 m/s qualified. The 5 m/s training fit selected a minimal
static-only change, but held routes rejected it; 0, 20, and 30 m/s remained
support- or episode-limited in this regression set. No candidate was emitted.

## To-Do

Progress legend: ✅ done &nbsp;•&nbsp; ⚠️ in progress &nbsp;•&nbsp; ❌ not started

Each feature links to its branch — the branch README has the full "what/how/what changed" story.

- ⚠️ **[Sometimes-On-Lateral (SOL)](https://github.com/SpysyWeeb/Spysypilot/tree/SOL)** — steering is toggled separately from cruise control, so the driver can use op lateral without enabling op long, or op long without enabling op lateral; runs its own state machine beside selfdrived's, with real panda/opendbc safety-layer support via the SpysyWeeb submodule forks; formerly Always-On-Lateral (AOL), which the code identifiers still use &nbsp;*(personal idea)*
- ✅ **[Hot-swap button between Chill/Experimental mode](https://github.com/SpysyWeeb/Spysypilot/tree/hot-swap-experimental)** — hold the steering-wheel distance button 0.5s to toggle Chill/Experimental without going into settings; a tap still cycles the follow personality &nbsp;*(inspired by sunnypilot)*
- ⚠️ **[Side panel quick-action buttons](https://github.com/SpysyWeeb/Spysypilot/tree/side-buttons)** — the home screen's right column is a stack of quick-access buttons: experimental-mode toggle, an update button with live download/install status, a screen-always-on toggle, and an error-log shortcut &nbsp;*(personal idea)*
- ⚠️ **[Nudgeless lane changes](https://github.com/SpysyWeeb/Spysypilot/tree/nudgless-lane-changes)** — lane changes trigger on turn signal alone, one automatic change per blinker event; pressing the brake cancels auto for that blinker event entirely (manual nudge still works) &nbsp;*(inspired by sunnypilot)*
- ⚠️ **[Better longitudinal tune v2 (BLoTv2)](https://github.com/SpysyWeeb/Spysypilot/tree/BLoTv2)** — combines a ground-up Smooth Stops replacement with a modular necessity supervisor and model-anchored MPC lead path for **Smooth. Swift. Strong.** longitudinal control; retains up to `4.0 m/s²` at launch under combo's existing opendbc/panda safety envelope, then fades the added authority into openpilot's exact stock acceleration gate by `10 m/s` (about `22 mph`) while preserving the reaction-time jerk tune; route `000000d2--a62f0c1831` predicts a 40 mph cruise cap of about `0.99 m/s²` instead of the observed `1.79 m/s²`; see [design](docs/BLoTv2.md) and [acceptance gates](docs/BLoTv2_ACCEPTANCE.md) &nbsp;*(in progress; awaiting owner field validation)*
- ✅\* **[Better boot screen](https://github.com/SpysyWeeb/Spysypilot/tree/better-boot-screen)** — the boot spinner shows live console output (build/manager), so hangs are immediately diagnosable from the device screen &nbsp;*(personal idea)*
- ✅ **[Error log viewer](https://github.com/SpysyWeeb/Spysypilot/tree/error-log-viewer)** — crashes are saved to an on-device log; a dev-menu button views it before/during/after a drive, with delete-on-close &nbsp;*(inspired by sunnypilot)*
- ✅ **[Auto-update](https://github.com/SpysyWeeb/Spysypilot/tree/auto-update)** — tapping "Check" automatically checks, downloads if an update is found, and reboots to install; background downloads (which already happen every ~1.5 hrs on non-metered connections) also auto-install the moment they finish while the car is parked &nbsp;*(personal idea)*
- ⚠️ **[Custom main menu windows](https://github.com/SpysyWeeb/Spysypilot/tree/custom-main-menu)** — replaces the "upgrade now" panel with two display-only BLaTv2 learning/readiness pages plus the existing live terminal and system graphs; the former route analyzer and `drive_statsd` are removed &nbsp;*(personal idea)*
- ✅ **[Swapped cruise speed adjustments](https://github.com/SpysyWeeb/Spysypilot/tree/swapped-cruise-speed)** — short press rounds to nearest 5 and jumps there (e.g. 42 → 45), long press steps by 1; reverses stock behavior &nbsp;*(inspired by sunnypilot)*
- ❌ **Quiet mode** — silence the engage and disengage sounds while leaving safety alerts audible; no branch yet &nbsp;*(inspired by sunnypilot)*
- ⚠️ **[Force Stops](https://github.com/SpysyWeeb/Spysypilot/tree/force-stops)** — legacy stop-strategy branch retained for reference; its independent cruise-speed cap is intentionally removed from `combo` and superseded there by BLoTv2 Conditional Experimental Mode &nbsp;*(inspired by IQPilot)*
- ⚠️ **[Better green lights](https://github.com/SpysyWeeb/Spysypilot/tree/better-green-lights)** — experimental-mode green-light launches start ~1.5–2s sooner by reading the model's path-length explosion instead of its laggy shouldStop bit, plus a launch assist that skips the dead time at the head of the model's speed plan &nbsp;*(personal idea)*
- ⚠️ **[Model curve speed limit](https://github.com/SpysyWeeb/Spysypilot/tree/curve-speed-limit)** — uses the model path and three owner-driven calibration points to cap cruise through curves, with spatial/temporal prediction-spike filtering and simple lookahead braking; see [docs/ModelCurveSpeedLimit.md](docs/ModelCurveSpeedLimit.md) &nbsp;*(personal idea)*
- ⚠️ **[Universal driving-event logger](https://github.com/SpysyWeeb/Spysypilot/tree/driving-event-platform)** — one rlog-first platform records automatic lateral and longitudinal failures plus general manual bookmarks, confirms preservation before showing UI success, and builds a bounded/reconstructable SSH manifest; lateral detector v7 adds b7-derived unused-authority and confirmed-driver-takeover events and remains in field validation; see [docs/DrivingEventPlatform.md](docs/DrivingEventPlatform.md) &nbsp;*(personal idea)*
- 🔒 **[Better lateral tune (BLaT)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaT)** — frozen reference implementation at the field-tested controller v14 tree from rollback authority `5e533e3ec6`; the rejected v15.x line is closed, and future ground-up lateral work belongs on stock-based `BLaTv2` &nbsp;*(personal idea)*
- ⚠️ **[Better lateral tune v2 (BLaTv2)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaTv2)** — ground-up modular adaptive lateral foundation; stock torque control is the active bootstrap, no BLaTv2 process runs onroad, and an offroad full-rlog importer builds the fully gated speed-local vehicle profile; no learned controller can actuate until replay, delivered-response, deterministic, safety, and device-timing approval all pass &nbsp;*(personal idea)*

  Offroad learning now exposes display-only progress for both deterministic replay passes, including route and segment position, route-application stages, and a conservative ETA once enough read/apply timing samples exist. Progress is never consumed by learning, evidence, candidate selection, or control.

  The two mandatory A/A passes remain independent and canonical. Each replay
  authority now has one private route-preparation helper, allowing up to four
  Python lanes without creating extra replay passes or parallelizing mutable
  learner state. Helpers use bounded, hash-verified scratch spools and never
  receive publication or Params authority; only the parent compares and
  publishes. Integrated b7/b8/b9/ca replay was byte-identical and 20.3% faster
  than two workers on the desktop, and four-worker processing has completed on
  the comma without changing deterministic artifacts.

  Evidence v6 in namespace v4 adds vehicle-global, angle-assisted physical
  breakaway episodes, whole-route held-out validation, and a nested
  no-regression model family. Existing v1/v2/v3 artifacts remain byte-for-byte
  untouched while retained compatible full rlogs replay into an empty v4
  namespace.
- ✅ **[Detailed system stats sidebar](https://github.com/SpysyWeeb/Spysypilot/tree/detailed-stats-sidebar)** — replace the "Temp Good / Vehicle Online / Connect Online" status pills with real data: actual CPU temp in °C, RAM usage, and power draw in watts &nbsp;*(inspired by FrogPilot)*

_\* = functional but could be better_
