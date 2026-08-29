# smooth-stops

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: in progress; queue-physics fixes are not field validated.**

## What it does

Kills the **"headbang" at the end of every stop**. Stock openpilot jumps to the `stopping` state the moment the plan's speed drops below `vEgoStopping` and ramps the brake command to −2.0 m/s² *while the car is still rolling* — on the Palisade that lands as a lurch at the very end of an otherwise smooth stop. This branch feathers the brake all the way down to a true standstill and only then hands off to the hold clamp.

*(personal idea)*

## How it works

A `SmoothStopController` inside longcontrol (control rate, true `v_ego`) owns the final approach:

- **Settle feather** — the braking level you *entered* the stop with fades linearly down to a gentle "kiss" (0.12 m/s²) as v → 0, anchored at entry so there's no step at engagement. Only the settle pressure is feathered — braking demanded by the MPC's plan (which owns collision avoidance) always passes through immediately, so full braking force is never delayed.
- **Lead awareness** — the settle uses relative closing speed to firm up toward the deceleration needed to stop at least 2.5 m behind a lead. An equal-speed queue does not create a false braking floor, while stronger planner braking still passes through immediately.
- **Anti-creep ratchet** — if the car stops making progress toward the stop, pressure ratchets up at 0.5 m/s² per second, escalating up to 4× when the remaining gap to a lead is tight. Moving-lead hysteresis and brief radar-dropout grace identify a creeping queue; obsolete ratchet pressure then decays instead of holding the queue back.
- **Clean handoff** — the stopping/hold clamp is only armed once the car is actually stopped (≤ 0.05 m/s or a trusted standstill signal), and once holding, release requires `shouldStop` to stay false for 10 frames so a one-frame plan flicker can't blip the brake at a light. A stopped lead in radar view no longer stretches that wait to 0.5 s (removed 2026-08-29: the Palisade needs ~1.3 s of its own to exit standstill after the release, so the wait went straight onto launch latency).

## What changed

- `openpilot/selfdrive/controls/lib/smooth_stops.py` *(new)* — the `SmoothStopController`: settle feather, lead-gap term, anti-creep ratchet, hold arm/release logic.
- `openpilot/selfdrive/controls/lib/longcontrol.py` — the stopping-state transition and hold release route through the controller instead of the stock instant clamp.
- `openpilot/selfdrive/controls/controlsd.py`, `openpilot/selfdrive/controls/radard.py`, `openpilot/selfdrive/controls/lib/longitudinal_planner.py` — wiring: lead distance to the controller and the hold-release path.

History note: this branch once also carried Smooth Approach / Smooth Release wrappers for braking further out; those were retired in favor of the [`BLoT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLoT) supervisor, which drives the MPC's own knobs instead of wrapping its output. This branch is now the stop-landing piece only.
