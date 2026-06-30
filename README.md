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

- ⚠️ **Always-On-Lateral (AOL)** — steering can be actuated while not fully engaged with cruise control &nbsp;*(inspired by sunnypilot)*
- ✅ **Hot-swap button between Chill/Experimental mode** &nbsp;*(inspired by sunnypilot)*
- ✅ **Nudgeless lane changes** &nbsp;*(inspired by sunnypilot)*
- ✅\* **Smooth stops** &nbsp;*(personal idea)*
- ✅\* **Better boot screen** — show console output during startup so hangs are immediately diagnosable &nbsp;*(personal idea)*
- ✅ **Error log viewer** — button in dev menu to view the error log before/during/after a drive &nbsp;*(inspired by sunnypilot)*
- ✅ **Auto-update** — the background daemon already silently checks and downloads updates every ~1.5 hours on non-metered connections, but the install (reboot) still requires a manual tap; the manual "Check" flow also stops short and shows a "Download" button instead of proceeding automatically; goal: (1) when offroad and a finalized update is ready, auto-reboot to install it without any user interaction, and (2) when the user taps "Check", skip the extra "Download" and "Install" taps and complete the whole flow automatically &nbsp;*(personal idea)*
- ❌ **Custom main menu windows** — replaces the "upgrade now" panel; tap cycles between windows; default is a driver status screen showing engaged time for the last drive and lifetime average (pulled from on-board routes); other windows include a live terminal showing openpilot output &nbsp;*(personal idea)*
- ❌ **Swapped cruise speed adjustments** — short press rounds to nearest 5 and jumps there (e.g. 42 → 45), long press steps by 1; reverses stock behavior &nbsp;*(personal idea)*
- ❌ **Exaggerated follow personalities** — aggressive targets ~1s following distance and reacts quickly; relaxed targets ~3s and smooths out inputs more than stock &nbsp;*(personal idea)*
- ❌ **Quiet mode** &nbsp;*(inspired by sunnypilot)*
- ❌ **Force Stops** &nbsp;*(inspired by IQPilot)*
- ❌ **Earlier lead takeoffs** &nbsp;*(inspired by IQPilot)*
- ❌ **Better longitudinal tune** &nbsp;*(personal idea)*
- ✅ **Detailed system stats sidebar** — replace the "Temp Good / Vehicle Online / Connect Online" status pills with real data: actual CPU temp in °C, RAM usage, and power draw in watts &nbsp;*(inspired by FrogPilot)*
- ❌ **Replace the live on-road view with a different screen showing live stats** &nbsp;*(personal idea)*
- ❌ **Get Clip.py working on device with a route viewer** &nbsp;*(personal idea)*

_\* = functional but could be better_
