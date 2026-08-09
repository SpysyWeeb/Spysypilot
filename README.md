# side-buttons

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

## What it does

Turns the right column of the home screen into a **stack of quick-action buttons**, so the most-used actions live one tap from the home screen instead of buried in settings:

- **Experimental mode** — the stock experimental-mode banner now *toggles* the mode directly on tap (stock sends you into settings).
- **Update** — one button that checks, downloads, and installs; its label tracks the updater live (`CHECKING…` / `DOWNLOADING…` / `INSTALL UPDATE` / `UPDATE FAILED`).
- **Screen timeout** — toggles a new persistent `ScreenAlwaysOn` param between "SCREEN ALWAYS ON" and "AUTO SCREEN TIMEOUT".
- **Error log** — opens the on-device crash log viewer (same modal as the [`error-log-viewer`](https://github.com/SpysyWeeb/Spysypilot/tree/error-log-viewer) branch), with delete-on-close.

*(personal idea)*

## How it works

`home.py`'s right column renders the four widgets stacked top-to-bottom above the setup widget. Each button is a small self-contained `Widget`: the update button drives `updated` with the same signals settings uses (`SIGUSR1` check → auto-`SIGHUP` download → `DoReboot` install) and mirrors `UpdaterState` into its label; the screen button flips `ScreenAlwaysOn`, which `ui_state`'s `Device` polls and translates into an interactive-timeout override so the screen never sleeps while it's on.

## What changed

- `openpilot/selfdrive/ui/layouts/home.py` — right column rebuilt as a four-button stack; experimental-mode tap now toggles the param.
- `openpilot/selfdrive/ui/widgets/update_button.py` *(new)* — check/download/install button with live updater status.
- `openpilot/selfdrive/ui/widgets/screen_timeout_button.py` *(new)* — `ScreenAlwaysOn` toggle.
- `openpilot/selfdrive/ui/widgets/error_log_button.py` *(new)* — error-log modal launcher.
- `openpilot/selfdrive/ui/widgets/exp_mode_button.py` — refreshes its state from Params every frame so the toggle reflects immediately.
- `openpilot/selfdrive/ui/ui_state.py` — `Device` honors `ScreenAlwaysOn` via an interactive-timeout override.
- `openpilot/common/params_keys.h` — new `ScreenAlwaysOn` param (plus the `Spysy*` stats keys shared with [`custom-main-menu`](https://github.com/SpysyWeeb/Spysypilot/tree/custom-main-menu)).
- `openpilot/system/ui/widgets/html_render.py` — `HtmlModal` `on_close` hook (shared with the error-log viewer).

## Cross-branch note

The error-log button depends on [`error-log-viewer`](https://github.com/SpysyWeeb/Spysypilot/tree/error-log-viewer): that branch's `sentry.py` hook is what writes `/data/community/crashes/error.log`. On this branch alone the button works but the log stays empty; both branches carry the identical `HtmlModal` `on_close` patch, so they merge cleanly. Everything else here is standalone.
