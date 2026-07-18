# custom-main-menu

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

## What it does

Replaces the useless "upgrade to comma prime" panel on the home screen with a set of **cycling info windows** — tap the panel to flip through them:

- **Driver status** — engaged vs. manual miles for the last drive and lifetime, computed from the routes stored on the device.
- **Driving breakdown** — where overrides happen (turns vs. curves vs. straight-line lane-position nudges), lane-change counts, average steering divergence, and turn-in/unwind timing versus the model.
- **Live terminal** — a scrolling console of openpilot's own output, colorized, right on the home screen.
- **System usage** — CPU/RAM/power/fan history graphs plus storage used/total.

*(personal idea)*

## How it works

A new off-road service, `drive_statsd`, polls for completed routes, parses their logs (engagement events, override events, steering angles), classifies each override by maneuver type — a "turn" is a low-speed ≥90° episode, a "curve" is model-steered road geometry, small commanded angles are lane-position preference — and writes per-drive and lifetime aggregates to Params (`SpysyLastDriveStats`, `SpysyLifetimeStats`, with an analyzer version stamp that forces reanalysis when the semantics change). The UI widgets are pure readers of those params, so the heavy log parsing never runs in the UI process; the terminal and system windows sample live sources (process output, `/sys` counters) on their own timers.

## What changed

- `openpilot/selfdrive/spysypilot/drive_statsd.py` *(new, the bulk of the branch)* — the stats analyzer/aggregator service, registered in `process_config.py` to run offroad.
- `openpilot/selfdrive/ui/layouts/home.py` — prime widget replaced by the tap-to-cycle window stack.
- `openpilot/selfdrive/ui/widgets/` *(new widgets)* — `drive_stats`, `override_stats`, `turn_stats`, `curve_stats`, `straight_stats`, `stats_common`, `terminal_widget`, `system_stats`.
- `openpilot/common/params_keys.h` — `Spysy*` stats params.
- `openpilot/common/swaglog.py` — log tee the terminal widget reads from.
