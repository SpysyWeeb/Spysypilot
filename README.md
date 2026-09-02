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

### Integration branch

[`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) is the only maintained combined testing branch and uses comma's stock release model. No standalone model-variant branch is active; the former `tsfdo-combo` branch was retired and deleted.

## Engagement CI

Every `combo` push and pull request builds the tree, compiles Python, and runs the existing model-schema, torque-learning, all-car controller-construction, and platform-contract tests. Automated panda/opendbc bumps build the updated tree and run the model-schema, torque-learning, and controller-construction checks before they can commit or push a new gitlink.

The Python hardware contract uses AGNOS for all comma hardware, shares the current Chestnut USB IDs/topology API between `hardwared` and modeld, pins the matching firmware payload, and exposes the USB GPU only after its firmware product string matches.

## BLoTv3 longitudinal stack

**Status: ⚠️ in progress; awaiting owner field validation.** BLoTv3 replaces BLoTv2 on
combo: the same road behavior where BLoTv2 was right — the continuous cubic acceleration
envelope (`0.6 + 3.4 (1 − v/40)³`, clamped by the deployed opendbc `ACCEL_MAX`, 4.0 m/s² here),
ordinary-cruise comfort shaping, the necessity supervisor that softens the MPC's jerk cost and pads
following time only when a lead requires it, radar-anchored model lead trajectories, the
lead-departure pre-release, Conditional Experimental Mode for lead-free model stops and Force
Stops' committed stop point — with one owner per longitudinal decision. The mode is decided in
`selfdrived`; the planner owns everything else, including the hold at standstill after a committed
stop, so no stop fields cross processes any more. There is no lateral turn budget (removed after
the 2026-08-29 field test: accelerating out of curves felt held back), Conditional Experimental Mode can engage with a distant vehicle in radar view,
a pedal tap no longer switches a manually enabled Experimental mode off, FCW keeps stock's form, and
the third-lead machinery is gone. Since 2026-08-30 the planner also bounds every stop's last metres with the
landing law (D22: a corridor — allowed braking 0.70·v + 0.30 m/s², a creep floor, both tapering to a 0.15 m/s² kiss at
walking pace — that latches through the MPC's hover and standstill until a real launch; lead physics never blocked; the owner's
original June Smooth Stops design rehomed in the planner), and since 2026-08-30 a one-way **release lift**: the ESP takes braking
up in ~0.2 s but lets it off in ~0.7 s, so when the car's measured deceleration exceeds the plan's the request is lifted in
proportion (the owner's accelerometer idea). radard's low-speed override no longer adopts an unconfirmed return closer than 0.6 s
of travel (a ground return the car drove over became the lead at 1.1 m on route 0x2a). The standstill hold stays on StopReq: a 2026-08-30 drive without it
(route 29) showed the SCC building brake pressure at standstill without end, chasing a deceleration it cannot measure on a stopped
car — the ESP's ~1.4 s standstill-exit sequence is the price of its own hold; Smooth Stops owns the standstill handoff (deferred
clamp, kiss below 0.3 m/s), and the curve policy shapes cruise through curves. Design, owner decisions
and acceptance gates: [`docs/BLoTv3.md`](https://github.com/SpysyWeeb/Spysypilot/blob/BLoTv3/docs/BLoTv3.md).

## BLaTv3 rack trajectory

**Status: ⚠️ in progress — phases 1 and 2 field-validated; phase 3 in progress.** On combo, `controlsd` selects
`LatControlRack` when opendbc sets `lateralTuning.torque.useRackTrajectory`, which it does only for the Palisade when the
queried platform code is `LX`; Telluride `ON`, mixed and unknown firmware fail closed to stock `LatControlTorque` and the
stock torque envelope. Any frame the rack controller cannot produce a request for is steered by a stock torque controller
stepped alongside it, and it logs under `lateralControlState.rackState`. Merged 2026-09-02 and not yet driven: the
driver-assist envelope and the R4 proactive envelope opening — watch for hands-on assist that never fights, earlier and
smoother turn-ins on long seen curves, and the wheel in the second after a release. A device that crash-loops with
`UnknownKeyName` after updating needs `rm -f /data/openpilot/openpilot/common/params_pyx*.so` and a reboot. Design,
step-by-step chronology with field numbers, and the failure-mode catalog live on the
[BLaTv3 branch README](https://github.com/SpysyWeeb/Spysypilot/tree/BLaTv3) and in
[`docs/BLaTv3_FAILURE_MODES.md`](https://github.com/SpysyWeeb/Spysypilot/blob/BLaTv3/docs/BLaTv3_FAILURE_MODES.md).

## To-Do

Progress legend: ✅ done &nbsp;•&nbsp; ⚠️ in progress &nbsp;•&nbsp; ❌ not started

Each feature links to its branch — the branch README has the full "what/how/what changed" story.

- ⚠️ **[Sometimes-On-Lateral (SOL)](https://github.com/SpysyWeeb/Spysypilot/tree/SOL)** — steering is toggled separately from cruise control, so the driver can use op lateral without enabling op long, or op long without enabling op lateral; runs its own state machine beside selfdrived's, with real panda/opendbc safety-layer support via the SpysyWeeb submodule forks; formerly Always-On-Lateral (AOL), which the code identifiers still use &nbsp;*(personal idea)*
- ✅ **[Hot-swap button between Chill/Experimental mode](https://github.com/SpysyWeeb/Spysypilot/tree/hot-swap-experimental)** — hold the steering-wheel distance button 0.5s to toggle Chill/Experimental without going into settings; a tap still cycles the follow personality &nbsp;*(inspired by sunnypilot)*
- ⚠️ **[Comma 3X torque bar](https://github.com/SpysyWeeb/Spysypilot/tree/torque-bar)** — shows the comma four steering-torque utilization arc by default on the comma 3X onroad display, scaled for its 2160×1080 UI with no settings toggle &nbsp;*(inspired by comma four and sunnypilot)*
- ⚠️ **[Comma 3X spinning steering wheel](https://github.com/SpysyWeeb/Spysypilot/tree/spinning-steering-wheel)** — rotates the existing top-right steering-wheel icon with the measured steering angle, with no settings toggle &nbsp;*(inspired by FrogPilot)*
- ⚠️ **[Side panel quick-action buttons](https://github.com/SpysyWeeb/Spysypilot/tree/side-buttons)** — the home screen's right column is a stack of quick-access buttons: experimental-mode toggle, an update button with live download/install status, a screen-always-on toggle, and an error-log shortcut &nbsp;*(personal idea)*
- ⚠️ **[Nudgeless lane changes](https://github.com/SpysyWeeb/Spysypilot/tree/nudgless-lane-changes)** — lane changes trigger on turn signal alone, one automatic change per blinker event; pressing the brake cancels auto for that blinker event entirely (manual nudge still works) &nbsp;*(inspired by sunnypilot)*
- 🔒 **[Better longitudinal tune v2 (BLoTv2)](https://github.com/SpysyWeeb/Spysypilot/tree/BLoTv2)** — superseded by BLoTv3, which reimplements its planner/MPC policy, lead response, Conditional Experimental Mode and cruise behavior with one owner per decision; the branch is kept as the reference the replay gates were proven against &nbsp;*(personal idea)*
- ⚠️ **[Better longitudinal tune v3 (BLoTv3)](https://github.com/SpysyWeeb/Spysypilot/tree/BLoTv3)** — rewrite of BLoTv2 in the upstream planner shape with one owner per longitudinal decision: the mode stays in selfdrived, stop commit *and* hold through standstill move to the planner, one shared stop classifier replaces the cross-process stop relay; keeps the cubic envelope, cruise comfort, necessity supervisor and model-lead anchoring, removes the third-lead "ponytail" and the inert `longcontrol.py` clamp, and fixes the reviewed BLoTv2 defects (CEM starved by any radar lead, standstill hold gap, ungated FCW path, manual Experimental lost on pedal taps, supervisor pad/ratchet cliffs; the inert turn budget is removed outright); see [docs/BLoTv3.md](https://github.com/SpysyWeeb/Spysypilot/blob/BLoTv3/docs/BLoTv3.md) &nbsp;*(personal idea; phases 1–3 implemented and replay-gated against BLoTv2; merged into combo 2026-08-29 for the owner's field test; landing law D22 added 2026-08-30, not yet driven)*
- ⚠️ **[Smooth Stops](https://github.com/SpysyWeeb/Spysypilot/tree/smooth-stops)** — the thin standstill handoff (clamp deferred to standstill, kiss taper below 0.3 m/s); the rolling landing is bounded by BLoTv3's planner-side landing law (D22), and stronger planner braking always passes through &nbsp;*(in progress; awaiting owner field validation)*
- ✅\* **[Better boot screen](https://github.com/SpysyWeeb/Spysypilot/tree/better-boot-screen)** — the boot spinner shows live console output (build/manager), so hangs are immediately diagnosable from the device screen &nbsp;*(personal idea)*
- ✅ **[Error log viewer](https://github.com/SpysyWeeb/Spysypilot/tree/error-log-viewer)** — crashes are saved to an on-device log; a dev-menu button views it before/during/after a drive, with delete-on-close &nbsp;*(inspired by sunnypilot)*
- ✅ **[Auto-update](https://github.com/SpysyWeeb/Spysypilot/tree/auto-update)** — tapping "Check" automatically checks, downloads if an update is found, and reboots to install; background downloads (which already happen every ~1.5 hrs on non-metered connections) also auto-install the moment they finish while the car is parked &nbsp;*(personal idea)*
- ⚠️ **[Custom main menu windows](https://github.com/SpysyWeeb/Spysypilot/tree/custom-main-menu)** — replaces the "upgrade now" panel with the existing live terminal and system graphs; the terminal feed starts with the initial home page instead of waiting for a page cycle, and the retired route analyzer, `drive_statsd`, and on-device BLaTv2 learner dashboard are removed &nbsp;*(personal idea)*
- ✅ **[Swapped cruise speed adjustments](https://github.com/SpysyWeeb/Spysypilot/tree/swapped-cruise-speed)** — short press rounds to nearest 5 and jumps there (e.g. 42 → 45), long press steps by 1; reverses stock behavior &nbsp;*(inspired by sunnypilot)*
- ❌ **Quiet mode** — silence the engage and disengage sounds while leaving safety alerts audible; no branch yet &nbsp;*(inspired by sunnypilot)*
- 🔒 **[Force Stops](https://github.com/SpysyWeeb/Spysypilot/tree/force-stops)** — commits to a lead-free model stop endpoint across prediction flicker and applies a bounded planner cruise-speed cap; now owned by BLoTv3's `force_stops.py`, which also holds the stop through standstill; the branch is kept as the reference its route-38 tuning came from &nbsp;*(inspired by IQPilot)*
- ⚠️ **[Better green lights](https://github.com/SpysyWeeb/Spysypilot/tree/better-green-lights)** — experimental-mode green-light launches start ~1.5–2s sooner by reading the model's path-length explosion instead of its laggy shouldStop bit, plus a launch assist that skips the dead time at the head of the model's speed plan; since 2026-08-30 the assist launches on the cruise ramp (under its cap) whenever the path is confirmed open but the model's own plan has not committed — route 28: +0.05 m/s² for a second after the release &nbsp;*(personal idea; cruise-ramp launch not yet driven)*
- ⚠️ **[Curve longitudinal policy](https://github.com/SpysyWeeb/Spysypilot/tree/curve-speed-limit)** — one plan candidate from two layers: anticipation (every path node gets the speed that keeps steering demand inside the torque budget, with the authority calibrated online from what the EPS actually delivers) and reaction to the measured steering (heavy-but-tracking coasts, pinned-and-understeering brakes). Since 2026-08-31 the calibrated authority is the ceiling that binds (comfort raised to 3.4 m/s² as backstop) and a 0.3 m/s hold band lets the car sit at the limit instead of stitching gas/brake corrections; see [docs/ModelCurveSpeedLimit.md](docs/ModelCurveSpeedLimit.md) &nbsp;*(personal idea; ceiling/hold band not yet driven)*
- 🔒 **[Better lateral tune (BLaT)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaT)** — frozen reference implementation at the field-tested controller v14 tree from rollback authority `5e533e3ec6`; the rejected v15.x line is closed, and future ground-up lateral work belongs on stock-based `BLaTv2` &nbsp;*(personal idea)*
- 🔒 **[Better lateral tune v2 (BLaTv2)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaTv2)** — superseded by BLaTv3: its rack-trajectory controller now runs on combo as `LatControlRack`, ported bit-exact; the branch is kept as the reference the port was proven against &nbsp;*(personal idea)*
- ⚠️ **[Better lateral tune v3 (BLaTv3)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaTv3)** — ground-up rewrite of the Palisade rack-trajectory controller in the upstream controller shape (`LatControlRack`, opendbc firmware flag, own log union arm, warm stock fallback), designed from a written failure-mode catalog: the model stays path authority, a scheduled preview never overrides the near target, and the static comfort-envelope tables give way to learned rack-effort surfaces &nbsp;*(personal idea; phases 1–2 field-validated 2026-08-29/30; phase 3 in progress — direction guard v2 and the shadow observer driven clean 2026-09-01, driver-assist envelope and R4 envelope opening merged 2026-09-02 and awaiting a drive)*
- ✅ **[Detailed system stats sidebar](https://github.com/SpysyWeeb/Spysypilot/tree/detailed-stats-sidebar)** — replace the "Temp Good / Vehicle Online / Connect Online" status pills with real data: actual CPU temp in °C, RAM usage, and power draw in watts &nbsp;*(inspired by FrogPilot)*

_\* = functional but could be better_
