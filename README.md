# error-log-viewer

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

## What it does

Lets you **read crash/error logs on the device itself** — before, during, or after a drive — instead of needing SSH or comma connect. An "Error Log — VIEW" button in the Developer settings opens the log in a scrollable modal; closing it offers to delete the log.

*(inspired by sunnypilot)*

## How it works

Every process crash already flows through `sentry.capture_exception()`. This branch hooks that path with `save_exception()`, which appends the formatted traceback (timestamped, newest entry first) to `/data/community/crashes/error.log`, trimmed to 100 KB so it can never grow unbounded. The viewer is a `HtmlModal` fed the escaped log text; a new `on_close` callback on `HtmlModal` chains into a confirm dialog for deleting the log.

## What changed

- `openpilot/system/sentry.py` — `save_exception()` writes every captured crash to `/data/community/crashes/error.log` (newest first, 100 KB cap); hooked into `capture_exception()`.
- `openpilot/selfdrive/ui/layouts/settings/developer.py` — "Error Log" button that renders the log in a modal and offers deletion on close.
- `openpilot/system/ui/widgets/html_render.py` — `HtmlModal` gained an `on_close` callback so the viewer can chain the delete prompt.

## Cross-branch note

[`side-buttons`](https://github.com/SpysyWeeb/Spysypilot/tree/side-buttons) ships a home-screen shortcut to this viewer and carries the same `HtmlModal` `on_close` change (identical patch, no conflict on merge). This branch is fully standalone; side-buttons' error-log button only shows useful content when this branch's crash-writing hook is also present (as it is in combo).
