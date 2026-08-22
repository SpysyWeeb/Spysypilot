# Spysypilot

This fork is **entirely vibe-coded** — including this README.

It's a personal side project for SpysyWeeb. It is **not meant for others to use**, but it's available for anyone who wants to try it **at their own risk**.

Any and all code and features generated in this project are free for others to use. SpysyWeeb doesn't take credit for the code itself, but if you build on an idea from here, a little credit for the idea would be appreciated. 🙏

This branch keeps the existing model-path curve speed limiter and adds a
Palisade-only future-torque envelope. Valid torque parameters can lower the
future curve speed before the unchanged approach-distance calculation; the
existing two-of-three veto still clamps only positive acceleration when
predicted demand reaches 90%. The 50/22/13 mph field envelope and stronger
braking remain intact. See
[`docs/ModelCurveSpeedLimit.md`](docs/ModelCurveSpeedLimit.md) for the design
and remaining validation work.

## Branches

- **`stock`** — clean commaai/openpilot base, no changes
- **`combo`** — all features merged together for testing
- Feature branches are cut from `stock` and merged into `combo` when ready; each feature branch's own README explains that feature in depth

### Combo variants

Only two combined testing branches are maintained:

| branch | driving model |
|---|---|
| [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) | stock comma release model |
| [`tsfdo-combo`](https://github.com/SpysyWeeb/Spysypilot/tree/tsfdo-combo) | comma's [`tsfdo`](https://github.com/commaai/openpilot/tree/tsfdo) experimental small driving model |

## Engagement CI

Every `combo` push and pull request builds the tree, compiles Python, and runs the existing model-schema, torque-learning, all-car controller-construction, and platform-contract tests. Automated panda/opendbc bumps build the updated tree and run the model-schema, torque-learning, and controller-construction checks before they can commit or push a new gitlink.

The Python hardware contract uses AGNOS for all comma hardware, shares the current Chestnut USB IDs/topology API between `hardwared` and modeld, pins the matching firmware payload, and exposes the USB GPU only after its firmware product string matches.

## Force Stops

**Status: replay-rejected; owner testing only.**
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

One second of current, complete, lead-free stop evidence can grant Force Stops
authority before the classic horizon. That capability is bound to the exact
model frame and fails closed on stale timestamps, malformed trajectories,
health faults, raw leads, or pedals. Route-17 segment 20 stopped `1.324 m`
behind its internal retained target, which is not calibrated painted-line
ground truth. Offline native LongControl did not reproduce the old segment-25
no-standstill result, but neither result is road validation.

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
Chill. The original `3 s` recent-lead guard is unchanged. During that guard,
one current strict 33-point finite stop frame may mint a revocable release only
when both raw radar leads are absent and every positive-probability sample from
all three complete `leadsV3` hypotheses is outside the predicted stop corridor.
Malformed lead/path data, a raw lead on any control tick, health loss, or a
committed turn revokes the release.

The controller audit now admits a model lead future only from a live model
service and a vision-corresponding radar lead with finite, physically consistent
position and speed. Unconfirmed low-speed radar tracks and malformed forecasts
use the exact radar-physics fallback. When a committed Force Stops obstacle owns
MPC, it publishes its own planner source instead of masquerading as ordinary
cruise. Lead0-derived adaptive MPC policy stands down when lead1, lead2, or the
committed stop is the current MPC owner.
Route 29 adds only the two intended CEM entries and still rejects the lower-
confidence replacement vehicle near Connect `1440.989`; routes 17 and 27 add no
entries. Neither route-29 entry bypasses the separate one-second Force Stops
qualification.

After one full second of current evidence, CEM also publishes the frame-bound
qualification consumed by Force Stops. Recent-lead recovery neither shortens
that qualification nor changes its exact current-frame binding. CEM still owns
no target speed, stop point, acceleration, or brake command.
Force Stops retains reversible approach shaping and the committed stop point;
MPC arbitrates the obstacle, and Smooth Stops owns final landing. Signed target
tracking remains bounded at the
MPC's behind-ego limit until a native Force Stops release, while the conditional
stop latch keeps `shouldStop` asserted through standstill prediction flicker.

Route-29 landing replay confirms both reported twitch windows hand `shouldStop`
to control inside the logged `0.5 s` actuator delay. A combined lifetime/timing
experiment still left the first handoff only `0.450 s` before standstill and
weakened an existing finite-endpoint fail-closed release, so it is not shipped.
Cap, target geometry, MPC ownership, LongControl ramps, and hold pressure remain
unchanged.

The effective mode has one owner and publishes through
`selfdriveState.experimentalMode`, which drives both the stock on-road icon and
planner strategy. The feature adds no UI or user-facing Params. The
implementation, signal mapping, and ownership boundaries are documented in
[`docs/BLoTv2.md`](docs/BLoTv2.md#conditional-experimental-mode).

## BLoTv2 ordinary cruise comfort

**Status: in progress; awaiting field validation.** Route
`000000d9--6040563d1d` showed that an acceleration ceiling alone still makes a small
highway set-speed correction use all available acceleration or the full
`-1.2 m/s²` cruise deceleration limit. The current tune adds an ordinary Chill
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

Lead presence does not disable the shaped cruise candidate; lead MPC remains
the safety owner and wins whenever it requests lower acceleration. Experimental
mode, forced deceleration, invalid radar, and an active model curve-speed limit
bypass this shaping. The jerk schedule and planner candidate arbitration are
unchanged. Production-function replay reduced the comparable route targets from approximately `+0.69` and
`-1.20 m/s²` to approximately `+0.40` and `-0.40 m/s²`; this is command replay,
not closed-loop or field validation. Design and remaining gates are documented
in [`docs/BLoTv2.md`](docs/BLoTv2.md#route-000000d9--6040563d1d-ordinary-cruise-comfort-refinement)
and [`docs/BLoTv2_ACCEPTANCE.md`](docs/BLoTv2_ACCEPTANCE.md).

## BLaTv2 rack trajectory

**Status: in progress; replay-qualified, not field-approved.** `controlsd` uses
the current rack-trajectory controller only when the shared Palisade/Telluride
platform has queried `LX` firmware. Telluride `ON` and unknown firmware remain
on stock `LatControlTorque`.

The former modular BLaTv2 stack and its design documents remain historical
research; they are not the live command owner. The current controller owns one
normalized request and leaves Hyundai rate/driver limiting and panda safety
downstream. Combo pins the reviewed opendbc integration artifact, which selects
the `409/+4/-7` envelope for the shared Palisade/Telluride platform. BLaTv2
activation remains Palisade `LX` only; Telluride stays on stock `LatControlTorque`.

## To-Do

Progress legend: ✅ done &nbsp;•&nbsp; ⚠️ in progress &nbsp;•&nbsp; ❌ not started

Each feature links to its branch — the branch README has the full "what/how/what changed" story.

- ⚠️ **[Sometimes-On-Lateral (SOL)](https://github.com/SpysyWeeb/Spysypilot/tree/SOL)** — steering is toggled separately from cruise control, so the driver can use op lateral without enabling op long, or op long without enabling op lateral; runs its own state machine beside selfdrived's, with real panda/opendbc safety-layer support via the SpysyWeeb submodule forks; formerly Always-On-Lateral (AOL), which the code identifiers still use &nbsp;*(personal idea)*
- ✅ **[Hot-swap button between Chill/Experimental mode](https://github.com/SpysyWeeb/Spysypilot/tree/hot-swap-experimental)** — hold the steering-wheel distance button 0.5s to toggle Chill/Experimental without going into settings; a tap still cycles the follow personality &nbsp;*(inspired by sunnypilot)*
- ⚠️ **[Comma 3X torque bar](https://github.com/SpysyWeeb/Spysypilot/tree/torque-bar)** — shows the comma four steering-torque utilization arc by default on the comma 3X onroad display, scaled for its 2160×1080 UI with no settings toggle &nbsp;*(inspired by comma four and sunnypilot)*
- ⚠️ **[Comma 3X spinning steering wheel](https://github.com/SpysyWeeb/Spysypilot/tree/spinning-steering-wheel)** — rotates the existing top-right steering-wheel icon with the measured steering angle, with no settings toggle &nbsp;*(inspired by FrogPilot)*
- ⚠️ **[Side panel quick-action buttons](https://github.com/SpysyWeeb/Spysypilot/tree/side-buttons)** — the home screen's right column is a stack of quick-access buttons: experimental-mode toggle, an update button with live download/install status, a screen-always-on toggle, and an error-log shortcut &nbsp;*(personal idea)*
- ⚠️ **[Nudgeless lane changes](https://github.com/SpysyWeeb/Spysypilot/tree/nudgless-lane-changes)** — lane changes trigger on turn signal alone, one automatic change per blinker event; pressing the brake cancels auto for that blinker event entirely (manual nudge still works) &nbsp;*(inspired by sunnypilot)*
- ⚠️ **[Better longitudinal tune v2 (BLoTv2)](https://github.com/SpysyWeeb/Spysypilot/tree/BLoTv2)** — owns planner/MPC policy, lead response, Conditional Experimental Mode, and cruise behavior without implementing final stop landing; requests a continuous cubic acceleration envelope from `4.0 m/s²` at launch to `0.6 m/s²` at `40 m/s`, clamped by combo's opendbc/panda safety envelope, while preserving the separate reaction-time jerk tune; see [design](docs/BLoTv2.md) and [acceptance gates](docs/BLoTv2_ACCEPTANCE.md) &nbsp;*(in progress; awaiting owner field validation)*
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
- ⚠️ **[Better lateral tune v2 (BLaTv2)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaTv2)** — Palisade `LX`-scoped rack-trajectory controller with model-authored timing, bounded motion planning, driver-override release, and stock fallback for Telluride/unknown firmware; still awaiting owner field validation &nbsp;*(personal idea; in progress)*
- ✅ **[Detailed system stats sidebar](https://github.com/SpysyWeeb/Spysypilot/tree/detailed-stats-sidebar)** — replace the "Temp Good / Vehicle Online / Connect Online" status pills with real data: actual CPU temp in °C, RAM usage, and power draw in watts &nbsp;*(inspired by FrogPilot)*

_\* = functional but could be better_
