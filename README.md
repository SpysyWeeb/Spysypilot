# custom-main-menu

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

## Status

**In progress.** The on-device BLaTv2 learning dashboard has been retired with
the on-device learner. Learning now runs on a PC from pulled routes; the car is
responsible only for recording data and running a separately reviewed profile.

## What it does

Replaces the "upgrade to comma prime" panel on the home screen with two
cycling windows. Tap the left half to go back or the right half to advance:

1. **Live terminal** — a scrolling, colorized console of openpilot output.
2. **System usage** — CPU/RAM/power/fan history plus storage used/total.

The BLaTv2 post-drive feedback prompt remains in place. It evaluates a
manually installed, PC-generated profile and does not perform learning or route
processing on the device.

*(personal idea)*

## How it works

The home carousel owns a list of display widgets and cycles it in either
direction with modular indexing. Removing the learner pages therefore removes
their Params polling and parsing from the UI process rather than merely hiding
the rendered panels.

The retained feedback prompt reads only the feedback request produced by the
profile lifecycle. It cannot train, fit, approve, activate, or reset a profile.
Routes and learning artifacts are handled outside the device UI.

## What changed

- `openpilot/selfdrive/ui/layouts/home.py` — two-page carousel containing the
  existing terminal and system pages.
- Removed the retired BLaTv2 learner/readiness widgets and their Params reader.
- Preserved the BLaTv2 post-drive feedback prompt for PC-generated profiles.
- Removed the five route-analyzer widgets, `drive_statsd`, its process
  registration, and its now-unused `Spysy*Stats` Params.
- `terminal_widget.py`, `system_stats.py`, and their behavior are unchanged.
