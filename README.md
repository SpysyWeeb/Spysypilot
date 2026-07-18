# swapped-cruise-speed

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

## What it does

Reverses stock openpilot's cruise-speed button behavior: a **short press jumps by 5 and rounds to the nearest 5** (e.g. 42 → 45), a **long press steps by 1**. Stock does the opposite (tap = 1, hold = 5), which makes the common case — big adjustments — the slow one.

*(personal idea)*

## How it works

Stock computes `v_cruise_delta * (5 if long_press else 1)` and snaps partial intervals to the nearest multiple on long press. This branch swaps the multiplier to `(1 if long_press else 5)` and moves the round-to-nearest-interval logic to the short-press path, so a tap from an off-multiple speed lands on the nearest 5 first instead of overshooting.

## What changed

- `openpilot/selfdrive/car/cruise.py` — swapped the long/short press multiplier and the partial-interval rounding condition in `VCruiseHelper`.
- `openpilot/selfdrive/car/tests/test_cruise_speed.py` — updated the expected decrement to match the new tap-equals-5 behavior.
