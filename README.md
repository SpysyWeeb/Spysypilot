# auto-update

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

## What it does

Makes updates fully hands-off:

- **Tapping CHECK does everything** — check, and if an update is found, the download starts automatically instead of stopping at a "DOWNLOAD" button waiting for a second tap.
- **Downloads auto-install** — the moment a finalized update is ready and the car is parked (offroad), the device reboots to apply it. This covers both user-initiated checks and the background downloads the updater already runs every ~1.5 hours on non-metered connections.
- **Stale-cache recovery** — if the updater "fetches" an update that turns out to be the version already running (a stale consistency cache), it invalidates the cache and re-checks once so a genuinely newer remote isn't masked.

*(personal idea)*

## How it works

The Software settings page sets an `_auto_fetch_pending` flag when CHECK is tapped; when the updater reports `UpdaterFetchAvailable`, the page fires the download signal (`SIGHUP` to `updated`) itself instead of just relabeling the button. Separately, `ui_state.update_params()` watches for `UpdateAvailable && !started` and sets `DoReboot`, which is what turns "downloaded" into "installed" without a trip to settings. The stale-cache path lives in `updated.py`'s main loop: after a fetch, if the finalized overlay's commit and branch match what's currently running, it clears the consistency flag and runs one more check/fetch cycle.

## What changed

- `openpilot/selfdrive/ui/layouts/settings/software.py` — CHECK auto-fires the download when the check finds an update.
- `openpilot/selfdrive/ui/ui_state.py` — auto-reboot to install once a finalized update is ready and the car is offroad.
- `openpilot/system/updated/updated.py` — detect fetched-update == running-version (stale cache), invalidate and re-fetch once.
