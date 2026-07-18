# detailed-stats-sidebar

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

## What it does

Replaces the sidebar's three vague status pills — "TEMP GOOD / VEHICLE ONLINE / CONNECT OFFLINE" — with **real numbers**:

- **CPU** — actual max component temperature in °C (warning ≥ 75, danger ≥ 90)
- **RAM** — memory usage in % (warning ≥ 75, danger ≥ 90)
- **POWER** — device power draw in watts (warning ≥ 13, danger ≥ 15)

Each metric keeps the stock green/yellow/red coloring, just driven by thresholds on the live value instead of a binary status.

*(inspired by FrogPilot)*

## How it works

All the data was already on the wire — `deviceState.maxTempC`, `memoryUsagePercent`, and `powerDrawW` arrive in the same message the stock sidebar already subscribes to. The stock `_update_temperature_status` / `_update_connection_status` / `_update_panda_status` handlers are replaced with three formatters that render the raw values with threshold-based colors.

## What changed

- `openpilot/selfdrive/ui/layouts/sidebar.py` — the only file: three new metric updaters reading `deviceState`, stock thermal/panda/Athena-ping status logic removed, same three metric slots in the render path.
