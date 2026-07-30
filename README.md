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
envelope, invalid-output safety, shadow measurement, and speed-local learning.
Each owner has a narrow interface so it can be audited, tuned, or replaced
without layering a compensating controller on another controller.

This first merged build deliberately contains no approved learned profile.
Consequently, its active lateral controller is the byte-identical current
stock openpilot `LatControlTorque`; the modular candidate is structurally
unable to actuate and runs only as shadow/learning infrastructure. On the
validated Palisade/Telluride platform, the stock request uses the
platform-selected 409/4/7 opendbc and panda envelope. Other vehicles retain
their stock limits and stock controller until their own opendbc port validates
the full command-envelope and rack-sensor contract.

Learning is measured-response-only and speed-local. Evidence at one speed
updates only neighboring speed nodes, so highway mileage cannot overwrite
low-speed knowledge. Candidates are trained and qualified offroad, require
enough evidence in every speed region, and never change during a drive. An
artifact can actuate only after independent raw/applied replay, delivered
replay, deterministic A/A, safety, and comma-device timing gates all pass.
Driver feedback after a provisional validation drive can approve, retain, or
roll back a profile at an engagement boundary.

No BLaTv1 controller code or `HyundaiLowSpeedTorqueDamping` is inherited.
Vehicle limits come from runtime `CarControllerParams`; BLaTv2 contains no
Palisade limit literals. Full architecture, contracts, learning policy,
acceptance gates, and rollback behavior are documented in
[`docs/BLATV2_MODULAR.md`](docs/BLATV2_MODULAR.md) and
[`docs/BLATV2_ACCEPTANCE.md`](docs/BLATV2_ACCEPTANCE.md).

## To-Do

Progress legend: ✅ done &nbsp;•&nbsp; ⚠️ in progress &nbsp;•&nbsp; ❌ not started

Each feature links to its branch — the branch README has the full "what/how/what changed" story.

- ⚠️ **[Sometimes-On-Lateral (SOL)](https://github.com/SpysyWeeb/Spysypilot/tree/SOL)** — steering is toggled separately from cruise control, so the driver can use op lateral without enabling op long, or op long without enabling op lateral; runs its own state machine beside selfdrived's, with real panda/opendbc safety-layer support via the SpysyWeeb submodule forks; formerly Always-On-Lateral (AOL), which the code identifiers still use &nbsp;*(personal idea)*
- ✅ **[Hot-swap button between Chill/Experimental mode](https://github.com/SpysyWeeb/Spysypilot/tree/hot-swap-experimental)** — hold the steering-wheel distance button 0.5s to toggle Chill/Experimental without going into settings; a tap still cycles the follow personality &nbsp;*(inspired by sunnypilot)*
- ⚠️ **[Side panel quick-action buttons](https://github.com/SpysyWeeb/Spysypilot/tree/side-buttons)** — the home screen's right column is a stack of quick-access buttons: experimental-mode toggle, an update button with live download/install status, a screen-always-on toggle, and an error-log shortcut &nbsp;*(personal idea)*
- ⚠️ **[Nudgeless lane changes](https://github.com/SpysyWeeb/Spysypilot/tree/nudgless-lane-changes)** — lane changes trigger on turn signal alone, one automatic change per blinker event; pressing the brake cancels auto for that blinker event entirely (manual nudge still works) &nbsp;*(inspired by sunnypilot)*
- ✅\* **[Smooth stops](https://github.com/SpysyWeeb/Spysypilot/tree/smooth-stops)** — feathers the brake down to a true standstill instead of stock's early −2 m/s² clamp, killing the Palisade's end-of-stop headbang; lead-aware with an anti-creep ratchet &nbsp;*(personal idea)*
- ✅\* **[Better boot screen](https://github.com/SpysyWeeb/Spysypilot/tree/better-boot-screen)** — the boot spinner shows live console output (build/manager), so hangs are immediately diagnosable from the device screen &nbsp;*(personal idea)*
- ✅ **[Error log viewer](https://github.com/SpysyWeeb/Spysypilot/tree/error-log-viewer)** — crashes are saved to an on-device log; a dev-menu button views it before/during/after a drive, with delete-on-close &nbsp;*(inspired by sunnypilot)*
- ✅ **[Auto-update](https://github.com/SpysyWeeb/Spysypilot/tree/auto-update)** — tapping "Check" automatically checks, downloads if an update is found, and reboots to install; background downloads (which already happen every ~1.5 hrs on non-metered connections) also auto-install the moment they finish while the car is parked &nbsp;*(personal idea)*
- ⚠️ **[Custom main menu windows](https://github.com/SpysyWeeb/Spysypilot/tree/custom-main-menu)** — replaces the "upgrade now" panel with tap-to-cycle windows: driver engagement stats (last drive + lifetime, computed on-device from stored routes by a new `drive_statsd` service), a driving breakdown (turn/curve/straight overrides), a live openpilot terminal, and system usage graphs &nbsp;*(personal idea)*
- ✅ **[Swapped cruise speed adjustments](https://github.com/SpysyWeeb/Spysypilot/tree/swapped-cruise-speed)** — short press rounds to nearest 5 and jumps there (e.g. 42 → 45), long press steps by 1; reverses stock behavior &nbsp;*(personal idea)*
- ❌ **Quiet mode** — silence the engage and disengage sounds while leaving safety alerts audible; no branch yet &nbsp;*(inspired by sunnypilot)*
- ⚠️ **[Force Stops](https://github.com/SpysyWeeb/Spysypilot/tree/force-stops)** — makes experimental mode actually commit to red lights and stop signs instead of the model's indecisive crawl, by latching the model's own planned stop point and capping cruise speed to reach it &nbsp;*(inspired by IQPilot)*
- ⚠️ **[Better green lights](https://github.com/SpysyWeeb/Spysypilot/tree/better-green-lights)** — experimental-mode green-light launches start ~1.5–2s sooner by reading the model's path-length explosion instead of its laggy shouldStop bit, plus a launch assist that skips the dead time at the head of the model's speed plan &nbsp;*(personal idea)*
- ⚠️ **[Better longitudinal tune (BLoT)](https://github.com/SpysyWeeb/Spysypilot/tree/BLoT)** — make braking and acceleration feel like my own driving: a necessity supervisor that drives the MPC's runtime knobs, model-predicted lead trajectories (comma PR [#37824](https://github.com/commaai/openpilot/pull/37824)), proportional launch control, and standstill lead pre-release that starts the Palisade's brake bleed from a sustained predicted departure without changing the acceleration curve; see [docs/BLoT.md](docs/BLoT.md); step 1 was [op-model-grader](https://github.com/SpysyWeeb/op-model-grader), which grades the model's driving from rlogs against my own manual driving &nbsp;*(personal idea)*
- ⚠️ **[Model curve speed limit](https://github.com/SpysyWeeb/Spysypilot/tree/curve-speed-limit)** — uses the model path and three owner-driven calibration points to cap cruise through curves, with spatial/temporal prediction-spike filtering and simple lookahead braking; see [docs/ModelCurveSpeedLimit.md](docs/ModelCurveSpeedLimit.md) &nbsp;*(personal idea)*
- ⚠️ **[Universal driving-event logger](https://github.com/SpysyWeeb/Spysypilot/tree/driving-event-platform)** — one rlog-first platform records automatic lateral and longitudinal failures plus general manual bookmarks, confirms preservation before showing UI success, and builds a bounded/reconstructable SSH manifest; lateral detector v7 adds b7-derived unused-authority and confirmed-driver-takeover events and remains in field validation; see [docs/DrivingEventPlatform.md](docs/DrivingEventPlatform.md) &nbsp;*(personal idea)*
- 🔒 **[Better lateral tune (BLaT)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaT)** — frozen reference implementation at the field-tested controller v14 tree from rollback authority `5e533e3ec6`; the rejected v15.x line is closed, and future ground-up lateral work belongs on stock-based `BLaTv2` &nbsp;*(personal idea)*
- ⚠️ **[Better lateral tune v2 (BLaTv2)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaTv2)** — ground-up modular adaptive lateral foundation; stock torque control is the active bootstrap while passive speed-local learning builds a fully gated vehicle profile; no learned controller can actuate until replay, delivered-response, deterministic, safety, and device-timing approval all pass &nbsp;*(personal idea)*
- ✅ **[Detailed system stats sidebar](https://github.com/SpysyWeeb/Spysypilot/tree/detailed-stats-sidebar)** — replace the "Temp Good / Vehicle Online / Connect Online" status pills with real data: actual CPU temp in °C, RAM usage, and power draw in watts &nbsp;*(inspired by FrogPilot)*

_\* = functional but could be better_
