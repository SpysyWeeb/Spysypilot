# hot-swap-experimental

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

## What it does

Toggles between **Chill and Experimental longitudinal mode from the steering wheel**, without ever opening settings: **hold the distance/gap button for 0.5s** to flip modes. A quick tap still cycles the follow personality like stock.

*(inspired by sunnypilot)*

## How it works

Stock openpilot fires the personality change on the distance button's *release*. This branch adds a frame counter that starts on button *press*: if the button is still held after 50 frames (0.5s @ 100 Hz), `ExperimentalMode` is toggled and written to Params, and a latch (`experimental_mode_switched`) swallows the release event so the same hold doesn't also cycle the personality. Releases shorter than the hold threshold fall through to the stock tap behavior.

## What changed

- `openpilot/selfdrive/selfdrived/selfdrived.py` — the only file: hold-vs-tap detection on the `gapAdjustCruise` button (`DISTANCE_LONG_PRESS = 50` frames), `ExperimentalMode` param toggle on hold, personality cycle preserved on tap.
