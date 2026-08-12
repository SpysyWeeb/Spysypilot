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

## Force Stops

**Status: in progress; awaiting fresh route replay and owner field validation.**
After CEM selects Experimental mode for a lead-free model stop, Force Stops
tracks that model endpoint as ego advances, caps the planner's cruise candidate
with a `0.65 m/s²` kinematic profile, and passes a committed physical stop point
to longitudinal MPC. MPC retains that point through the profile's `6 m` boundary
and rolling landing, until Force Stops itself releases. Force Stops resists
ordinary farther-away endpoint noise, accepts bounded near-stop correction, and
holds the same point for `4 s` through brief model flicker. Either raw radar lead,
invalid model/radar data, driver gas, disengagement, or standstill releases it
immediately. Closer leads and stronger MPC/e2e braking
still win; Smooth Stops remains the sole final-landing and standstill-handoff
owner.

## BLoTv2 Conditional Experimental Mode

**Status: in progress; awaiting field validation.** Combo normally remains in
Chill mode, then hands its existing longitudinal planner to Experimental mode
for a confirmed, lead-free model stop prediction. It holds that handoff through
the stop and returns to Chill after a stable release or driver override.

There is no standalone traffic-light or stop-sign classifier in this build, so
the detector uses the model's existing `action.shouldStop`, predicted
position/velocity/orientation, and action signals. The route-derived tiers now
recognize a straight, lead-free near-stop trajectory earlier: weaker evidence
can qualify at or below `22 m/s` after the filter/debounce, while the known
`55 mph` highway slowdown remains filter-only. A qualifying stop also refreshes
a `4 s` mode hold so brief prediction flicker cannot return the approach to
Chill. The earlier route replay moved the first handoff from `39.1 mph` to
`42.2 mph`; the new earlier threshold still requires fresh replay and field
validation.

The effective state has one owner and publishes through
`selfdriveState.experimentalMode`, which drives both the stock on-road icon and
planner strategy. The feature adds no UI or user-facing Params and does not own
target speed, stop point, or braking. The implementation, signal mapping, and
ownership boundaries are documented in
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

## BLaTv2 archive

**Status: reverted from `combo`; stock lateral control restored.** `controlsd`
constructs and updates openpilot's stock `LatControlTorque` directly. The former
`BLaTv2ExperimentalController` activation key and post-drive BLaTv2 feedback
prompt are removed, so no BLaTv2 controller can own steering on `combo`.

The BLaTv2 branch, commits, source modules, tests, route evidence, and design
documents remain available as historical research. They are intentionally kept
out of the live command path rather than deleted or rewritten. Any future
Palisade work restarts from stock torque control and adds only a small,
evidence-backed intervention after route analysis identifies a falsifiable
cause. Existing research contracts are documented in
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
- ⚠️ **[Better longitudinal tune v2 (BLoTv2)](https://github.com/SpysyWeeb/Spysypilot/tree/BLoTv2)** — owns planner/MPC policy, lead response, Conditional Experimental Mode, and cruise behavior without implementing final stop landing; retains up to `4.0 m/s²` at launch under combo's existing opendbc/panda safety envelope, then fades the added authority into openpilot's exact stock acceleration gate by `10 m/s` (about `22 mph`) while preserving the reaction-time jerk tune; see [design](docs/BLoTv2.md) and [acceptance gates](docs/BLoTv2_ACCEPTANCE.md) &nbsp;*(in progress; awaiting owner field validation)*
- ⚠️ **[Smooth Stops](https://github.com/SpysyWeeb/Spysypilot/tree/smooth-stops)** — independently owns the final rolling landing and standstill handoff; planner/MPC necessity remains under BLoTv2 and stronger planner braking always passes through &nbsp;*(in progress; awaiting owner field validation)*
- ✅\* **[Better boot screen](https://github.com/SpysyWeeb/Spysypilot/tree/better-boot-screen)** — the boot spinner shows live console output (build/manager), so hangs are immediately diagnosable from the device screen &nbsp;*(personal idea)*
- ✅ **[Error log viewer](https://github.com/SpysyWeeb/Spysypilot/tree/error-log-viewer)** — crashes are saved to an on-device log; a dev-menu button views it before/during/after a drive, with delete-on-close &nbsp;*(inspired by sunnypilot)*
- ✅ **[Auto-update](https://github.com/SpysyWeeb/Spysypilot/tree/auto-update)** — tapping "Check" automatically checks, downloads if an update is found, and reboots to install; background downloads (which already happen every ~1.5 hrs on non-metered connections) also auto-install the moment they finish while the car is parked &nbsp;*(personal idea)*
- ⚠️ **[Custom main menu windows](https://github.com/SpysyWeeb/Spysypilot/tree/custom-main-menu)** — replaces the "upgrade now" panel with the existing live terminal and system graphs; the terminal feed starts with the initial home page instead of waiting for a page cycle, and the retired route analyzer, `drive_statsd`, and on-device BLaTv2 learner dashboard are removed &nbsp;*(personal idea)*
- ✅ **[Swapped cruise speed adjustments](https://github.com/SpysyWeeb/Spysypilot/tree/swapped-cruise-speed)** — short press rounds to nearest 5 and jumps there (e.g. 42 → 45), long press steps by 1; reverses stock behavior &nbsp;*(inspired by sunnypilot)*
- ❌ **Quiet mode** — silence the engage and disengage sounds while leaving safety alerts audible; no branch yet &nbsp;*(inspired by sunnypilot)*
- ⚠️ **[Force Stops](https://github.com/SpysyWeeb/Spysypilot/tree/force-stops)** — commits to a lead-free model stop endpoint across prediction flicker and applies a bounded planner cruise-speed cap; integrated separately from BLoTv2 CEM recognition and Smooth Stops final landing &nbsp;*(inspired by IQPilot; awaiting replay and field validation)*
- ⚠️ **[Better green lights](https://github.com/SpysyWeeb/Spysypilot/tree/better-green-lights)** — experimental-mode green-light launches start ~1.5–2s sooner by reading the model's path-length explosion instead of its laggy shouldStop bit, plus a launch assist that skips the dead time at the head of the model's speed plan &nbsp;*(personal idea)*
- ⚠️ **[Model curve speed limit](https://github.com/SpysyWeeb/Spysypilot/tree/curve-speed-limit)** — uses the model path and three owner-driven calibration points to cap cruise through curves, with spatial/temporal prediction-spike filtering and simple lookahead braking; see [docs/ModelCurveSpeedLimit.md](docs/ModelCurveSpeedLimit.md) &nbsp;*(personal idea)*
- 🔒 **[Better lateral tune (BLaT)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaT)** — frozen reference implementation at the field-tested controller v14 tree from rollback authority `5e533e3ec6`; the rejected v15.x line is closed, and future ground-up lateral work belongs on stock-based `BLaTv2` &nbsp;*(personal idea)*
- ⚠️ **[Better lateral tune v2 (BLaTv2)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaTv2)** — archived research branch; its live controller and activation path are reverted from `combo`, which now uses stock `LatControlTorque` exclusively. Route analysis continues before any small stock-plus Palisade experiment is proposed &nbsp;*(personal idea; in progress)*
- ✅ **[Detailed system stats sidebar](https://github.com/SpysyWeeb/Spysypilot/tree/detailed-stats-sidebar)** — replace the "Temp Good / Vehicle Online / Connect Online" status pills with real data: actual CPU temp in °C, RAM usage, and power draw in watts &nbsp;*(inspired by FrogPilot)*

_\* = functional but could be better_
