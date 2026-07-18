# BLaT — Better Lateral Tune

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

Companion longitudinal branch: [`BLoT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLoT).

## What it does

Gives the Palisade's steering **more authority and a faster wind-up** so it can hold tighter curves and take 90° turns without running wide:

- **Max steering torque 384 → 409** (+6.5%)
- **Torque ramp-up rate 3 → 4 per frame** — full torque reachable from zero in ~1.0 s instead of ~1.4 s
- Ramp-*down* stays at 7, so torque still sheds nearly twice as fast as it builds (the safety asymmetry is preserved)

The full Hyundai limit tuple is now **(409, 4, 7)** vs. stock **(384, 3, 7)**.

## How it works

This superproject branch carries no openpilot code changes — it is a submodule pointer: `.gitmodules` overrides `opendbc` to [SpysyWeeb/opendbc](https://github.com/SpysyWeeb/opendbc), branch [`BLaT`](https://github.com/SpysyWeeb/opendbc/tree/BLaT), where the actual changes live. Steering limits exist in **two enforcement layers** and both must agree, or the panda safety firmware blocks openpilot's own commands:

1. `opendbc/car/hyundai/values.py` — what openpilot *commands* (`STEER_MAX = 409`, `STEER_DELTA_UP = 4`; cars on `ALT_LIMITS` are untouched).
2. `opendbc/safety/modes/hyundai.h` — what the panda safety code *permits* (`HYUNDAI_LIMITS(409, 4, 7)`).
3. `opendbc/safety/tests/test_hyundai.py` — the safety tests asserting the new limits.

`hyundai_canfd.h` is deliberately untouched — the Palisade is CAN, not CANFD. The panda reflashes with the new limits on the next boot after installing.

## What changed

- `.gitmodules` + `opendbc_repo` pointer — the whole superproject delta; see the three opendbc files above for the real content.
