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

⚠️ 2026-09-14: modeld/Chestnut build files, hardwared status and 60+ upstream-owned files re-synced to upstream; opendbc pins set by the consistency rule (docs/SYNC_POLICY.md); awaiting a drive — watch modeld lag and LDW re-arm timing. c3x-chestnut was merged on top the same day and changes the modeld build files, modeld.py and the Chestnut PCIe probe again; see its To-Do entry.

## Engagement CI

Every `combo` push and pull request builds the tree, compiles Python, and runs the existing model-schema, torque-learning, all-car controller-construction, and platform-contract tests. Automated panda/opendbc bumps build the updated tree and run the model-schema, torque-learning, and controller-construction checks before they can commit or push a new gitlink.

The Python hardware contract uses AGNOS for all comma hardware, shares the current Chestnut USB IDs/topology API between `hardwared` and modeld, pins the matching firmware payload, and exposes the USB GPU only after its firmware product string matches.

## BLoTv3 longitudinal stack

**Status: ⚠️ in progress; awaiting owner field validation.** [BLoTv3](https://github.com/SpysyWeeb/Spysypilot/tree/BLoTv3)
replaces BLoTv2 on combo: the same road behavior where BLoTv2 was right, with one owner per longitudinal decision —
the mode in `selfdrived`, everything else in the planner (stop commit, the hold through standstill, the landing).
[Smooth Stops](https://github.com/SpysyWeeb/Spysypilot/tree/smooth-stops) owns the standstill handoff and the
[curve policy](https://github.com/SpysyWeeb/Spysypilot/tree/curve-speed-limit) shapes cruise through curves. What each
decision is, why, and what the field said: the BLoTv3 README and
[`docs/BLoTv3.md`](https://github.com/SpysyWeeb/Spysypilot/blob/BLoTv3/docs/BLoTv3.md) (decisions, contracts, field log).
2026-09-17: the branch's cleanliness audit fixes (five stop-layer defects — a stale profile anchor after the slow release, the
hold's green release ignoring the model's stop call, a gas tap at speed arming no re-entry, a radar dropout hiding an invalid model
from CEM, a creep resume read as a launch — plus tests and docs) merged; awaiting a drive.

## To-Do

Progress legend: ✅ done &nbsp;•&nbsp; ⚠️ in progress &nbsp;•&nbsp; ❌ not started

Each feature links to its branch — the branch README has the full "what/how/what changed" story.

- ✅ **[Sometimes-On-Lateral (SOL)](https://github.com/SpysyWeeb/Spysypilot/tree/SOL)** — steering is toggled separately from cruise control, so the driver can use op lateral without enabling op long, or op long without enabling op lateral; runs its own state machine beside selfdrived's, with real panda/opendbc safety-layer support via the SpysyWeeb submodule forks; formerly Always-On-Lateral (AOL), which the code identifiers still use &nbsp;*(personal idea)*
- ✅ **[Hot-swap button between Chill/Experimental mode](https://github.com/SpysyWeeb/Spysypilot/tree/hot-swap-experimental)** — hold the steering-wheel distance button 0.5s to toggle Chill/Experimental without going into settings; a tap still cycles the follow personality &nbsp;*(inspired by sunnypilot)*
- ✅ **[Comma 3X torque bar](https://github.com/SpysyWeeb/Spysypilot/tree/torque-bar)** — shows the comma four steering-torque utilization arc by default on the comma 3X onroad display, scaled for its 2160×1080 UI with no settings toggle &nbsp;*(inspired by comma four and sunnypilot)*
- ✅ **[Comma 3X spinning steering wheel](https://github.com/SpysyWeeb/Spysypilot/tree/spinning-steering-wheel)** — rotates the existing top-right steering-wheel icon with the measured steering angle, with no settings toggle &nbsp;*(inspired by FrogPilot)*
- ⚠️ **[Comma 3X Chestnut big model](https://github.com/SpysyWeeb/Spysypilot/tree/c3x-chestnut)** — the big driving model runs on a Chestnut dock from a comma 3X: the device's own GPU warps the camera frames, so only model-size frames cross the dock's USB link, and modeld waits for the dock's PCIe link before loading the big model &nbsp;*(based on upstream's routing before #38684; merged into combo 2026-09-14; the big model ran on two drives, the PCIe link wait awaits a drive)*
- ✅ **[Side panel quick-action buttons](https://github.com/SpysyWeeb/Spysypilot/tree/side-buttons)** — the home screen's right column is a stack of quick-access buttons: experimental-mode toggle, an update button with live download/install status, a screen-always-on toggle, and an error-log shortcut &nbsp;*(personal idea)*
- ✅ **[Nudgeless lane changes](https://github.com/SpysyWeeb/Spysypilot/tree/nudgless-lane-changes)** — lane changes trigger on turn signal alone, one automatic change per blinker event; pressing the brake cancels auto for that blinker event entirely (manual nudge still works) &nbsp;*(inspired by sunnypilot)*
- ✅ **[ESC-active alert hold](https://github.com/SpysyWeeb/Spysypilot/tree/esp-active-debounce)** — the "Electronic Stability Control Active" take-control alert only fires after 0.5 s of continuous ESC activity, so the ~0.4 s ABS pulse the Palisade reports over bumps and pavement joints no longer triggers the 2 s full-screen alert and chime; a real skid still alerts within its first half second &nbsp;*(personal idea)*
- 🔒 **[Better longitudinal tune v2 (BLoTv2)](https://github.com/SpysyWeeb/Spysypilot/tree/BLoTv2)** — superseded by BLoTv3, which reimplements its planner/MPC policy, lead response, Conditional Experimental Mode and cruise behavior with one owner per decision; the branch is kept as the reference the replay gates were proven against &nbsp;*(personal idea)*
- ⚠️ **[Better longitudinal tune v3 (BLoTv3)](https://github.com/SpysyWeeb/Spysypilot/tree/BLoTv3)** — rewrite of BLoTv2 in the upstream planner shape with one owner per longitudinal decision: envelope, cruise comfort, necessity supervisor, model-lead anchoring, Conditional Experimental Mode, Force Stops and the landing law; see the branch README and [docs/BLoTv3.md](https://github.com/SpysyWeeb/Spysypilot/blob/BLoTv3/docs/BLoTv3.md) &nbsp;*(personal idea; merged into combo 2026-08-29, under owner field testing)*
- ⚠️ **[Smooth Stops](https://github.com/SpysyWeeb/Spysypilot/tree/smooth-stops)** — the thin standstill handoff in LongControl (clamp deferred to standstill, kiss below the stop bit) and radard's low-speed lead gates; the rolling landing itself is BLoTv3's &nbsp;*(in progress; awaiting owner field validation)*
- ✅\* **[Better boot screen](https://github.com/SpysyWeeb/Spysypilot/tree/better-boot-screen)** — the boot spinner shows live console output (build/manager), so hangs are immediately diagnosable from the device screen &nbsp;*(personal idea)*
- ✅ **[Error log viewer](https://github.com/SpysyWeeb/Spysypilot/tree/error-log-viewer)** — crashes are saved to an on-device log; a dev-menu button views it before/during/after a drive, with delete-on-close &nbsp;*(inspired by sunnypilot)*
- ✅ **[Auto-update](https://github.com/SpysyWeeb/Spysypilot/tree/auto-update)** — tapping "Check" automatically checks, downloads if an update is found, and reboots to install; background downloads (which already happen every ~1.5 hrs on non-metered connections) also auto-install the moment they finish while the car is parked &nbsp;*(personal idea)*
- ✅ **[Custom main menu windows](https://github.com/SpysyWeeb/Spysypilot/tree/custom-main-menu)** — replaces the "upgrade now" panel with the existing live terminal and system graphs; the terminal feed starts with the initial home page instead of waiting for a page cycle, and the retired route analyzer, `drive_statsd`, and on-device BLaTv2 learner dashboard are removed &nbsp;*(personal idea)*
- ✅ **[Swapped cruise speed adjustments](https://github.com/SpysyWeeb/Spysypilot/tree/swapped-cruise-speed)** — short press rounds to nearest 5 and jumps there (e.g. 42 → 45), long press steps by 1; reverses stock behavior &nbsp;*(inspired by sunnypilot)*
- 🔒 **Force Stops** — commits to a lead-free model stop endpoint across prediction flicker; the standalone branch was retired into BLoTv3's `force_stops.py` and deleted &nbsp;*(inspired by IQPilot)*
- ✅ **[Better green lights](https://github.com/SpysyWeeb/Spysypilot/tree/better-green-lights)** — experimental-mode green-light launches read the model's path opening instead of its laggy stop bit, with a launch assist for the head of the model's speed plan &nbsp;*(personal idea)*
- ⚠️ **[Curve longitudinal policy](https://github.com/SpysyWeeb/Spysypilot/tree/curve-speed-limit)** — one plan candidate: anticipation from the model path against the steering's calibrated authority, reaction to the measured steering (coast when heavy, brake when pinned and understeering); see [docs/ModelCurveSpeedLimit.md](https://github.com/SpysyWeeb/Spysypilot/blob/curve-speed-limit/docs/ModelCurveSpeedLimit.md) &nbsp;*(personal idea; under owner field testing)*
- 🔒 **[Better lateral tune (BLaT)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaT)** — frozen reference implementation at the field-tested controller v14 tree from rollback authority `5e533e3ec6`; the rejected v15.x line is closed, and future ground-up lateral work belongs on stock-based `BLaTv2` &nbsp;*(personal idea)*
- 🔒 **[Better lateral tune v2 (BLaTv2)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaTv2)** — superseded by BLaTv3, which ported its rack-trajectory controller bit-exact and has since been retired; the branch is kept as the reference the port was proven against &nbsp;*(personal idea)*
- 🔒 **[Better lateral tune v3 (BLaTv3)](https://github.com/SpysyWeeb/Spysypilot/tree/BLaTv3)** — retired 2026-09-14 and removed from combo: steering is upstream's `LatControlTorque` again, with the Palisade's 409/4/7 torque envelope kept from the opendbc fork; the branch is frozen as the reference for the rack-trajectory design, its field log and the [failure-mode catalog](https://github.com/SpysyWeeb/Spysypilot/blob/BLaTv3/docs/BLaTv3_FAILURE_MODES.md) &nbsp;*(personal idea)*
- ✅ **[Detailed system stats sidebar](https://github.com/SpysyWeeb/Spysypilot/tree/detailed-stats-sidebar)** — replace the "Temp Good / Vehicle Online / Connect Online" status pills with real data: actual CPU temp in °C, RAM usage, and power draw in watts &nbsp;*(inspired by FrogPilot)*

_\* = functional but could be better_
