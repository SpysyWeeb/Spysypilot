# nudgless-lane-changes

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

*(Yes, the branch name is missing an "e". It's staying that way.)*

## What it does

**Lane changes trigger on the turn signal alone** — no steering nudge needed to confirm. With guardrails so it never fires when you didn't mean it:

- **One auto change per blinker activation.** A second change on the same blinker needs a manual nudge; cycling the blinker re-arms the automatic one.
- **Braking cancels the automatic change for the whole blinker event.** If you touch the brake with the blinker on, that blinker activation requires a manual nudge — the change will *not* fire the moment you lift off the pedal. Re-arm by cycling the blinker.

*(inspired by sunnypilot)*

## How it works

All in `DesireHelper`. After a short delay (`LANE_CHANGE_NUDGELESS_DELAY`, 0.05 s) in the pre-lane-change state, the change starts as if the driver had nudged — gated by `auto_allowed`, which requires: no auto change already used this blinker event (`nudgeless_used` latch) and no brake press seen this blinker event (`brake_cancelled` latch, set on `brakePressed` and only cleared when the blinker cycles off). Manual torque always works exactly as stock.

## What changed

- `openpilot/selfdrive/controls/lib/desire_helper.py` — the only file: nudgeless auto-start with delay, the one-per-blinker latch, and the brake latch-cancel.
