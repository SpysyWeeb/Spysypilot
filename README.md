# Spysypilot

This fork is **entirely vibe-coded** — including this README.

It's a personal side project for SpysyWeeb. It is **not meant for others to use**, but it's available for anyone who wants to try it **at their own risk**.

Any and all code and features generated in this project are free for others to use. SpysyWeeb can't take credit for the code itself, but if you build on an idea from here, a little credit for the idea would be appreciated. 🙏

## Branches

- **`stock`** — clean commaai/openpilot base, no changes
- **`combo`** — all features merged together for testing
- Feature branches are cut from `stock` and merged into `combo` when ready; each feature branch's own README explains that feature in depth

### Combo variants

Only two combined testing branches are maintained:

| branch | driving model |
|---|---|
| [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) | stock comma release model |
| [`rdf-combo`](https://github.com/SpysyWeeb/Spysypilot/tree/rdf-combo) | comma's [`rdf-driving`](https://github.com/commaai/openpilot/tree/rdf-driving) experimental RDF model |

## BLoTv2 Conditional Experimental Mode

**Status: in progress; awaiting field validation.** Combo normally remains in
Chill mode, then hands its existing longitudinal planner to Experimental mode
for a confirmed, lead-free model stop prediction. It holds that handoff through
the stop and returns to Chill after a stable release or driver override.

There is no standalone traffic-light or stop-sign classifier in this build, so
the detector uses the model's existing `action.shouldStop`, predicted
position/velocity/orientation, and action signals. A route-derived high-speed
tier now recognizes a straight, lead-free near-stop trajectory before the
terminal prediction reaches zero; weaker evidence only precharges its existing
temporal filter. Replay of route
`000000d7--cc6308b4d0` moves the first handoff from `39.1 mph` to `42.2 mph`
without admitting its highway-exit slowdown or brief radar-dropout window.
This is counterfactual replay evidence, not field validation. The effective
state has one owner and publishes through `selfdriveState.experimentalMode`,
which drives both the stock on-road icon and planner strategy. The feature adds
no UI or user-facing Params and does not own target speed or braking. The
implementation, signal mapping, and ownership boundaries are documented in
[`docs/BLoTv2.md`](docs/BLoTv2.md#conditional-experimental-mode).

## BLoTv2 ordinary cruise comfort

**Status: in progress; awaiting field validation.** Route
`000000d9--6040563d1d` showed that an acceleration ceiling alone still makes a small
highway set-speed correction use all available acceleration or the full
`-1.2 m/s²` cruise deceleration limit. The current tune adds a lead-free Chill
cruise comfort response: above `15 m/s`, a `5 mph` error asks for about
`0.40 m/s²`, tapers continuously as the error closes, and uses the
pitch-compensated coast estimate during speed reductions. It blends in from
`8–15 m/s`, retaining low-speed launch response, while larger errors can still
reach the existing acceleration and braking envelope.

The former four-node acceleration gate is now one continuous cubic envelope:
`a_max = 0.6 + 3.4 × (1 − v/40)³` from `0–40 m/s`, then `0.6 m/s²` above it.
It retains the `4.0 m/s²` launch request without the abrupt 0-to-10 m/s drop,
stays near the prior urban-speed tune, and reaches its highway floor smoothly.
The deployed opendbc limit still clamps the request.

Experimental mode, forced deceleration, invalid radar or lead-following
operation, and an active model curve-speed limit bypass this shaping. The jerk
schedule and planner candidate arbitration are unchanged. Production-function
replay reduced the comparable route targets from approximately `+0.69` and
`-1.20 m/s²` to approximately `+0.40` and `-0.40 m/s²`; this is command replay,
not closed-loop or field validation. Design and remaining gates are documented
in [`docs/BLoTv2.md`](docs/BLoTv2.md#route-000000d9--6040563d1d-ordinary-cruise-comfort-refinement)
and [`docs/BLoTv2_ACCEPTANCE.md`](docs/BLoTv2_ACCEPTANCE.md).

## BLaTv2 modular replacement

**Status: in progress; PC-only learning while stock torque control remains
active.** The previous LQI controller is retired. LQI means “Linear Quadratic
Integral”: state feedback with an integral-error state. The replacement is a
modular learning system aimed at **Smooth. Swift. Strong.**, but this milestone
does not authorize learned steering. There is no approved profile or automatic
activation path. `LatControlTorque` remains byte-identical to the stock
bootstrap. The validated Palisade/Telluride uses the runtime-selected 409/4/7
opendbc/panda envelope; every other vehicle keeps the limits supplied by its
own `CarControllerParams`.

The comma now has one role in learning: ordinary loggerd records full rlogs.
No BLaTv2 learner, route replay worker, uploader, Wi-Fi bridge, or processing
fallback runs in the device manager graph. Closed routes are copied manually
over read-only SSH into durable PC storage. Historical replay, deterministic
A/A verification, physical learning, behavioral qualification, and immutable
candidate generation all run on the PC from a clean, commit-pinned checkout.
The retired bridge cannot publish Params, install a profile, or activate a
controller.

Physical calibration and behavioral qualification are deliberately separate.
The physical learner estimates only observable quantities—torque per measured
lateral acceleration, signed offset, moving friction, and static breakaway—at
0/5/10/15/20/30 m/s. Samples update adjacent nodes only, interpolation requires
support on both sides, and seed retention is a valid qualified result; long
highway drives therefore cannot erase low-speed knowledge. Support, numerical
rank/conditioning, training, held-route validation, and full-authority evidence
are reported independently. Each canonical route counter owns its complete
partition (even training, odd validation), and learning-status schema 4
accounts for every prepared frame as accepted or one explicit first rejection
cause. Eighteen retained routes reproduced byte-identically in two independent
passes; the 0–20 m/s nodes safely retained their seeds, while 30 m/s remains one
held-out breakaway episode short, so no candidate was emitted.

The behavior learner consumes a homogeneous, immutable route cohort (at least
four whole routes) and replays exact stock, incumbent, and candidate controllers
against the same inputs. It may adjust only global natural frequency and
damping, then must independently satisfy the Smooth, Swift, and Strong gates on
held-out routes. Driver interventions censor contaminated evidence rather than
voting a candidate up or down, and event logs locate windows but never define
the desired path. A complete physical fit or passing behavior candidate remains
informational: neither can populate an approved artifact or change actuation
without a later, separately reviewed activation phase.

The retired learner/readiness dashboard has been removed because the device no
longer owns learning state. The home panel retains the live terminal and system
usage views. The post-drive feedback prompt remains available for a future
manually installed, PC-generated profile; it cannot train, approve, or activate
one.

No BLaTv1 controller code or `HyundaiLowSpeedTorqueDamping` is inherited. Full
module boundaries, evidence contracts, process-safety rules, trust boundaries,
gates, and rollback behavior are documented in
[`docs/BLATV2_MODULAR.md`](docs/BLATV2_MODULAR.md) and
[`docs/BLATV2_ACCEPTANCE.md`](docs/BLATV2_ACCEPTANCE.md).

## To-Do

Progress legend: ✅ done &nbsp;•&nbsp; ⚠️ in progress &nbsp;•&nbsp; ❌ not started

Each feature links to its branch — the branch README has the full "what/how/what changed" story.

- ⚠️ **[Sometimes-On-Lateral (SOL)](https://github.com/SpysyWeeb/Spysypilot/tree/SOL)** — steering is toggled separately from cruise control, so the driver can use op lateral without enabling op long, or op long without enabling op lateral; runs its own state machine beside selfdrived's, with real panda/opendbc safety-layer support via the SpysyWeeb submodule forks; formerly Always-On-Lateral (AOL), which the code identifiers still use &nbsp;*(personal idea)*
- ✅ **[Hot-swap button between Chill/Experimental mode](https://github.com/SpysyWeeb/Spysypilot/tree/hot-swap-experimental)** — hold the steering-wheel distance button 0.5s to toggle Chill/Experimental without going into settings; a tap still cycles the follow personality &nbsp;*(inspired by sunnypilot)*
- ⚠️ **[Comma 3X torque bar](https://github.com/SpysyWeeb/Spysypilot/tree/torque-bar)** — shows the comma four steering-torque utilization arc by default on the comma 3X onroad display, scaled for its 2160×1080 UI with no settings toggle &nbsp;*(inspired by comma four and sunnypilot)*
- ⚠️ **[Comma 3X spinning steering wheel](https://github.com/SpysyWeeb/Spysypilot/tree/spinning-steering-wheel)** — rotates the existing top-right steering-wheel icon with the measured steering angle, with no settings toggle &nbsp;*(inspired by FrogPilot)*
- ⚠️ **[Side panel quick-action buttons](https://github.com/SpysyWeeb/Spysypilot/tree/side-buttons)** — the home screen's right column is a stack of quick-access buttons: experimental-mode toggle, an update button with live download/install status, a screen-always-on toggle, and an error-log shortcut &nbsp;*(personal idea)*
- ⚠️ **[Nudgeless lane changes](https://github.com/SpysyWeeb/Spysypilot/tree/nudgless-lane-changes)** — lane changes trigger on turn signal alone, one automatic change per blinker event; pressing the brake cancels auto for that blinker event entirely (manual nudge still works) &nbsp;*(inspired by sunnypilot)*
- ⚠️ **[Better longitudinal tune v2 (BLoTv2)](https://github.com/SpysyWeeb/Spysypilot/tree/BLoTv2)** — combines a ground-up Smooth Stops replacement with a modular necessity supervisor and model-anchored MPC lead path for **Smooth. Swift. Strong.** longitudinal control; retains up to `4.0 m/s²` at launch under combo's existing opendbc/panda safety envelope, then fades the added authority into openpilot's exact stock acceleration gate by `10 m/s` (about `22 mph`) while preserving the reaction-time jerk tune; route `000000d2--a62f0c1831` predicts a 40 mph cruise cap of about `0.99 m/s²` instead of the observed `1.79 m/s²`; see [design](docs/BLoTv2.md) and [acceptance gates](docs/BLoTv2_ACCEPTANCE.md) &nbsp;*(in progress; awaiting owner field validation)*
- ✅\* **[Better boot screen](https://github.com/SpysyWeeb/Spysypilot/tree/better-boot-screen)** — the boot spinner shows live console output (build/manager), so hangs are immediately diagnosable from the device screen &nbsp;*(personal idea)*
- ✅ **[Error log viewer](https://github.com/SpysyWeeb/Spysypilot/tree/error-log-viewer)** — crashes are saved to an on-device log; a dev-menu button views it before/during/after a drive, with delete-on-close &nbsp;*(inspired by sunnypilot)*
- ✅ **[Auto-update](https://github.com/SpysyWeeb/Spysypilot/tree/auto-update)** — tapping "Check" automatically checks, downloads if an update is found, and reboots to install; background downloads (which already happen every ~1.5 hrs on non-metered connections) also auto-install the moment they finish while the car is parked &nbsp;*(personal idea)*
- ⚠️ **[Custom main menu windows](https://github.com/SpysyWeeb/Spysypilot/tree/custom-main-menu)** — replaces the "upgrade now" panel with the existing live terminal and system graphs; the terminal feed starts with the initial home page instead of waiting for a page cycle, and the retired route analyzer, `drive_statsd`, and on-device BLaTv2 learner dashboard are removed &nbsp;*(personal idea)*
- ✅ **[Swapped cruise speed adjustments](https://github.com/SpysyWeeb/Spysypilot/tree/swapped-cruise-speed)** — short press rounds to nearest 5 and jumps there (e.g. 42 → 45), long press steps by 1; reverses stock behavior &nbsp;*(inspired by sunnypilot)*
- ❌ **Quiet mode** — silence the engage and disengage sounds while leaving safety alerts audible; no branch yet &nbsp;*(inspired by sunnypilot)*
- ⚠️ **[Force Stops](https://github.com/SpysyWeeb/Spysypilot/tree/force-stops)** — legacy stop-strategy branch retained for reference; its independent cruise-speed cap is intentionally removed from `combo` and superseded there by BLoTv2 Conditional Experimental Mode &nbsp;*(inspired by IQPilot)*
- ⚠️ **[Better green lights](https://github.com/SpysyWeeb/Spysypilot/tree/better-green-lights)** — experimental-mode green-light launches start ~1.5–2s sooner by reading the model's path-length explosion instead of its laggy shouldStop bit, plus a launch assist that skips the dead time at the head of the model's speed plan &nbsp;*(personal idea)*
- ⚠️ **[Model curve speed limit](https://github.com/SpysyWeeb/Spysypilot/tree/curve-speed-limit)** — uses the model path and three owner-driven calibration points to cap cruise through curves, with spatial/temporal prediction-spike filtering and simple lookahead braking; see [docs/ModelCurveSpeedLimit.md](docs/ModelCurveSpeedLimit.md) &nbsp;*(personal idea)*
- ⚠️ **[Universal driving-event logger](https://github.com/SpysyWeeb/Spysypilot/tree/driving-event-platform)** — one rlog-first platform records automatic lateral and longitudinal failures plus general manual bookmarks, confirms preservation before showing UI success, and builds a bounded/reconstructable SSH manifest; lateral detector v7 adds b7-derived unused-authority and confirmed-driver-takeover events and remains in field validation; see [docs/DrivingEventPlatform.md](docs/DrivingEventPlatform.md) &nbsp;*(personal idea)*
- 🔒 **[Better lateral tune (BLaT)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaT)** — frozen reference implementation at the field-tested controller v14 tree from rollback authority `5e533e3ec6`; the rejected v15.x line is closed, and future ground-up lateral work belongs on stock-based `BLaTv2` &nbsp;*(personal idea)*
- ⚠️ **[Better lateral tune v2 (BLaTv2)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaTv2)** — Palisade contextual controller field trial in progress; archived routes select one bounded, speed-scheduled inverse-rack candidate, which remains explicitly provisional and can run only through a Palisade-only development parameter read at controls startup; stock remains the default and no background learner or second live controller runs on the device &nbsp;*(personal idea; awaiting owner field validation)*
- ✅ **[Detailed system stats sidebar](https://github.com/SpysyWeeb/Spysypilot/tree/detailed-stats-sidebar)** — replace the "Temp Good / Vehicle Online / Connect Online" status pills with real data: actual CPU temp in °C, RAM usage, and power draw in watts &nbsp;*(inspired by FrogPilot)*

_\* = functional but could be better_
