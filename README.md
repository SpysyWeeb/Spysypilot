# better-boot-screen

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

## What it does

Shows **live console output on the boot spinner** instead of a bare progress bar, so when a boot or build hangs you can see *what* it's stuck on right from the device screen — no SSH needed to diagnose a bricked-looking startup.

*(personal idea)*

## How it works

The stock spinner only accepts a percentage or a single status line from the process that spawned it. This branch turns the spinner into a **long-lived singleton fed over a named FIFO** (`/tmp/spysypilot_boot.fifo` + a pidfile): whichever process starts first (manager, build) spawns the spinner in its own session and later processes attach to the same FIFO instead of spawning a second one, so the display survives the manager→build→manager handoffs during an update. `build.py` and `manager.py` stream their real output (scons lines, prestart steps, python exceptions) into it, and the on-screen spinner renders a scrolling console region under the stock spinner/progress UI.

## What changed

- `openpilot/common/spinner.py` — spinner client rewritten around the FIFO singleton: alive-check via pidfile, stale-file cleanup, non-blocking connect with a 15s handshake window, text streaming API.
- `openpilot/system/ui/spinner.py` — on-device spinner renders a scrolling console log alongside the stock progress display.
- `openpilot/system/manager/build.py` — build output (scons progress and errors) streams into the spinner console.
- `openpilot/system/manager/manager.py` — manager startup steps and failures stream into the spinner console.
