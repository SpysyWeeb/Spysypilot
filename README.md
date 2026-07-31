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

## BLaTv2 modular replacement

**Status: in progress.** The previous LQI-based BLaTv2 controller is retired.
LQI means “Linear Quadratic Integral”: state feedback plus an integral-error
state. The replacement is a ground-up, modular adaptive torque controller
whose target is **Smooth. Swift. Strong.**

The system separates model-time intent, a stateless future reference, measured
rack mapping, physical plant/inverse torque, the exact opendbc command
envelope, invalid-output safety, offline shadow measurement, and speed-local
learning from closed full rlogs. Each owner has a narrow interface so it can
be audited, tuned, or replaced without layering a compensating controller on
another controller.

This first merged build deliberately contains no approved learned profile.
Consequently, its active lateral controller is the byte-identical current
stock openpilot `LatControlTorque`. No BLaTv2 helper process runs while the
car is onroad. Normal loggerd full rlogs retain the measured inputs, and the
real-car offroad worker replays only complete, closed routes after logger
finalization. On the validated Palisade/Telluride platform, the stock request
uses the platform-selected 409/4/7 opendbc and panda envelope. Other vehicles
retain their stock limits and stock controller until their own opendbc port
validates the full command-envelope and rack-sensor contract.

Learning is measured-response-only and speed-local. Evidence at one speed
updates only neighboring speed nodes, so highway mileage cannot overwrite
low-speed knowledge. `blatv2_backfilld` is the sole durable writer: it checks
exact route/build/vehicle compatibility, replays each eligible batch twice,
and commits only bit-identical results. Compatible older full-rlog routes can
be imported; qlogs, open/incomplete routes, and unreviewed builds cannot.
Candidates require enough evidence in every speed region and never change
during a drive. An artifact can actuate only after independent raw/applied
replay, delivered replay, deterministic A/A, safety, and comma-device timing
gates all pass.

Torque-envelope boundary frames are retained as incorporated authority
evidence instead of being discarded as merely constrained. Exact slew-limit
and full-magnitude transitions are classified from the vehicle's runtime
`CarControllerParams`; driver-influenced or physically impossible transitions
remain excluded. Slew and stuck-rack samples inform support without being
misrepresented as instantaneous equality measurements. Only continuously held
full torque with measurable rack motion, after the identified transport delay,
may enter the separate authority fit, and that fit must independently improve
ordinary and authority held-out validation before it can affect a candidate.

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

The Learning page shows credited clean support for every data-driven speed
node and now distinguishes active offroad processing from an empty evidence
set. The Readiness page keeps time, held-out validation, steering excitation,
physical-fit state, and controller lifecycle visibly separate. Blue means
processing, amber means enough time but another evidence gate is blocked,
green means a node qualified, red is reserved for a real fit/lifecycle
failure, and gray means no current validated snapshot. A complete learned
profile never implies activation; this build reports stock until a separately
reviewed activation path exists.

## To-Do

Progress legend: ✅ done &nbsp;•&nbsp; ⚠️ in progress &nbsp;•&nbsp; ❌ not started

Each feature links to its branch — the branch README has the full "what/how/what changed" story.

- ⚠️ **[Sometimes-On-Lateral (SOL)](https://github.com/SpysyWeeb/Spysypilot/tree/SOL)** — steering is toggled separately from cruise control, so the driver can use op lateral without enabling op long, or op long without enabling op lateral; runs its own state machine beside selfdrived's, with real panda/opendbc safety-layer support via the SpysyWeeb submodule forks; formerly Always-On-Lateral (AOL), which the code identifiers still use &nbsp;*(personal idea)*
- ✅ **[Hot-swap button between Chill/Experimental mode](https://github.com/SpysyWeeb/Spysypilot/tree/hot-swap-experimental)** — hold the steering-wheel distance button 0.5s to toggle Chill/Experimental without going into settings; a tap still cycles the follow personality &nbsp;*(inspired by sunnypilot)*
- ⚠️ **[Side panel quick-action buttons](https://github.com/SpysyWeeb/Spysypilot/tree/side-buttons)** — the home screen's right column is a stack of quick-access buttons: experimental-mode toggle, an update button with live download/install status, a screen-always-on toggle, and an error-log shortcut &nbsp;*(personal idea)*
- ⚠️ **[Nudgeless lane changes](https://github.com/SpysyWeeb/Spysypilot/tree/nudgless-lane-changes)** — lane changes trigger on turn signal alone, one automatic change per blinker event; pressing the brake cancels auto for that blinker event entirely (manual nudge still works) &nbsp;*(inspired by sunnypilot)*
- ⚠️ **[Better longitudinal tune v2 (BLoTv2)](https://github.com/SpysyWeeb/Spysypilot/tree/BLoTv2)** — combines a ground-up Smooth Stops replacement with a modular necessity supervisor and model-anchored MPC lead path for **Smooth. Swift. Strong.** longitudinal control; restores BLoT v1's launch-tapered request up to `4.0 m/s²` under combo's existing opendbc/panda safety envelope while retaining the BLoTv2 jerk/MPC tune for field evaluation; see [design](docs/BLoTv2.md) and [acceptance gates](docs/BLoTv2_ACCEPTANCE.md) &nbsp;*(in progress; awaiting owner field validation)*
- ✅\* **[Better boot screen](https://github.com/SpysyWeeb/Spysypilot/tree/better-boot-screen)** — the boot spinner shows live console output (build/manager), so hangs are immediately diagnosable from the device screen &nbsp;*(personal idea)*
- ✅ **[Error log viewer](https://github.com/SpysyWeeb/Spysypilot/tree/error-log-viewer)** — crashes are saved to an on-device log; a dev-menu button views it before/during/after a drive, with delete-on-close &nbsp;*(inspired by sunnypilot)*
- ✅ **[Auto-update](https://github.com/SpysyWeeb/Spysypilot/tree/auto-update)** — tapping "Check" automatically checks, downloads if an update is found, and reboots to install; background downloads (which already happen every ~1.5 hrs on non-metered connections) also auto-install the moment they finish while the car is parked &nbsp;*(personal idea)*
- ⚠️ **[Custom main menu windows](https://github.com/SpysyWeeb/Spysypilot/tree/custom-main-menu)** — replaces the "upgrade now" panel with two display-only BLaTv2 learning/readiness pages plus the existing live terminal and system graphs; the former route analyzer and `drive_statsd` are removed &nbsp;*(personal idea)*
- ✅ **[Swapped cruise speed adjustments](https://github.com/SpysyWeeb/Spysypilot/tree/swapped-cruise-speed)** — short press rounds to nearest 5 and jumps there (e.g. 42 → 45), long press steps by 1; reverses stock behavior &nbsp;*(inspired by sunnypilot)*
- ❌ **Quiet mode** — silence the engage and disengage sounds while leaving safety alerts audible; no branch yet &nbsp;*(inspired by sunnypilot)*
- ⚠️ **[Force Stops](https://github.com/SpysyWeeb/Spysypilot/tree/force-stops)** — makes experimental mode actually commit to red lights and stop signs instead of the model's indecisive crawl, by latching the model's own planned stop point and capping cruise speed to reach it &nbsp;*(inspired by IQPilot)*
- ⚠️ **[Better green lights](https://github.com/SpysyWeeb/Spysypilot/tree/better-green-lights)** — experimental-mode green-light launches start ~1.5–2s sooner by reading the model's path-length explosion instead of its laggy shouldStop bit, plus a launch assist that skips the dead time at the head of the model's speed plan &nbsp;*(personal idea)*
- ⚠️ **[Model curve speed limit](https://github.com/SpysyWeeb/Spysypilot/tree/curve-speed-limit)** — uses the model path and three owner-driven calibration points to cap cruise through curves, with spatial/temporal prediction-spike filtering and simple lookahead braking; see [docs/ModelCurveSpeedLimit.md](docs/ModelCurveSpeedLimit.md) &nbsp;*(personal idea)*
- ⚠️ **[Universal driving-event logger](https://github.com/SpysyWeeb/Spysypilot/tree/driving-event-platform)** — one rlog-first platform records automatic lateral and longitudinal failures plus general manual bookmarks, confirms preservation before showing UI success, and builds a bounded/reconstructable SSH manifest; lateral detector v7 adds b7-derived unused-authority and confirmed-driver-takeover events and remains in field validation; see [docs/DrivingEventPlatform.md](docs/DrivingEventPlatform.md) &nbsp;*(personal idea)*
- 🔒 **[Better lateral tune (BLaT)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaT)** — frozen reference implementation at the field-tested controller v14 tree from rollback authority `5e533e3ec6`; the rejected v15.x line is closed, and future ground-up lateral work belongs on stock-based `BLaTv2` &nbsp;*(personal idea)*
- ⚠️ **[Better lateral tune v2 (BLaTv2)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaTv2)** — ground-up modular adaptive lateral foundation; stock torque control is the active bootstrap, no BLaTv2 process runs onroad, and an offroad full-rlog importer builds the fully gated speed-local vehicle profile; no learned controller can actuate until replay, delivered-response, deterministic, safety, and device-timing approval all pass &nbsp;*(personal idea)*

  Offroad learning now exposes display-only progress for both deterministic replay passes, including route and segment position, route-application stages, and a conservative ETA once enough read/apply timing samples exist. Progress is never consumed by learning, evidence, candidate selection, or control.

  Evidence v2 reconstructs vehicle-generic signed rack motion without changing
  its measured magnitude: platforms with unsigned rate sensors derive direction
  from the offset-corrected steering angle, and physical reversals count toward
  speed-local coverage without fitting the sign-crossing acceleration impulse.
  Applied torque is paired causally using the previous `carOutput` publication
  and the speed-interpolated seed transport delay. Existing v1 artifacts remain
  byte-for-byte untouched while retained full rlogs replay into an initially
  empty v2 namespace.
- ✅ **[Detailed system stats sidebar](https://github.com/SpysyWeeb/Spysypilot/tree/detailed-stats-sidebar)** — replace the "Temp Good / Vehicle Online / Connect Online" status pills with real data: actual CPU temp in °C, RAM usage, and power draw in watts &nbsp;*(inspired by FrogPilot)*

_\* = functional but could be better_
