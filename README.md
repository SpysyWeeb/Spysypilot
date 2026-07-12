# Spysypilot

This fork is **entirely vibe-coded using [Claude Code](https://claude.com/claude-code)** — including this README.

It's a personal side project for SpysyWeeb. It is **not meant for others to use**, but it's available for anyone who wants to try it **at their own risk**.

Any and all code and features generated in this project are free for others to use. SpysyWeeb can't take credit for the code itself, but if you build on an idea from here, a little credit for the idea would be appreciated. 🙏

## Branches

- **`stock`** — clean commaai/openpilot base, no changes
- **`combo`** — all features merged together for testing
- Feature branches are cut from `stock` and merged into `combo` when ready

## To-Do

Progress legend: ✅ done &nbsp;•&nbsp; ⚠️ in progress &nbsp;•&nbsp; ❌ not started

- ⚠️ **Always-On-Lateral (AOL)** — steering is toggled separately from cruise control, so the driver can choose to use op long without enabling op lateral &nbsp;*(personal idea)*
- ✅ **Hot-swap button between Chill/Experimental mode** — a button on the main driving screen toggles between Chill and Experimental longitudinal mode without going into settings &nbsp;*(inspired by sunnypilot)*
- ⚠️ **Side panel quick-action buttons** — a growing column of quick-access buttons on the driving screen; currently includes a screen-always-on toggle; planned additions: an error log shortcut that opens the viewer directly, and an update button that also reflects live download/install status &nbsp;*(personal idea)*
- ⚠️ **Nudgeless lane changes** — lane changes trigger on turn signal alone without requiring a steering nudge to confirm; brake pedal pressed now forces a manual nudge instead of auto-changing &nbsp;*(inspired by sunnypilot)*
- ✅\* **Smooth stops** — softer deceleration curve on approach to a full stop, reducing the lurch the Palisade tends to have at the very end of a stop &nbsp;*(personal idea)*
- ✅\* **Better boot screen** — show console output during startup so hangs are immediately diagnosable &nbsp;*(personal idea)*
- ✅ **Error log viewer** — button in dev menu to view the error log before/during/after a drive &nbsp;*(inspired by sunnypilot)*
- ✅ **Auto-update** — tapping "Check" automatically checks, downloads if an update is found, and reboots to install; background downloads (which already happen every ~1.5 hrs on non-metered connections) also auto-install the moment they finish while the car is parked &nbsp;*(personal idea)*
- ⚠️ **Custom main menu windows** — replaces the "upgrade now" panel; tap cycles between windows; default is a driver status screen showing engaged time for the last drive and lifetime average (pulled from on-board routes); other windows include a live terminal showing openpilot output and a system usage window (CPU/RAM/Power/Fan graphs, storage used/total) &nbsp;*(personal idea)*
- ✅ **Swapped cruise speed adjustments** — short press rounds to nearest 5 and jumps there (e.g. 42 → 45), long press steps by 1; reverses stock behavior &nbsp;*(personal idea)*
- ❌ **Quiet mode** — silence the engage and disengage sounds while leaving safety alerts audible &nbsp;*(inspired by sunnypilot)*
- ⚠️ **Force Stops** — ensure the car comes to a complete stop at lights and stop signs rather than a rolling stop &nbsp;*(inspired by IQPilot)*
- ⚠️ **Better longitudinal tune (BLT)** — make braking and acceleration feel like my own driving: commit to slowdowns early and smoothly, hold braking no longer than necessary, release as one human-like taper, react to lead takeoffs quickly, and keep follow distance stable (less speed hunting); step 1 is [op-model-grader](https://github.com/SpysyWeeb/op-model-grader), a standalone tool that grades the model's longitudinal (and lateral) performance from rlogs against my own manual driving; step 2 is the BLT necessity supervisor (branch `BLT`, see docs/BLT.md) — a watchdog that checks whether braking is actually necessary and modulates the MPC's own runtime knobs instead of wrapping it; also plans against the driving model's predicted lead trajectory (upstream PR #37824 via IQPilot, incl. my lead pull-away floor) instead of extrapolated radar, so launches are anticipated seconds before radar can measure them; absorbs the retired Smooth Approach / Smooth Release wrappers and the follow-personality tuning &nbsp;*(personal idea, lead trajectories based on comma PR [#37824](https://github.com/commaai/openpilot/pull/37824))*
- ✅ **Detailed system stats sidebar** — replace the "Temp Good / Vehicle Online / Connect Online" status pills with real data: actual CPU temp in °C, RAM usage, and power draw in watts &nbsp;*(inspired by FrogPilot)*

_\* = functional but could be better_
