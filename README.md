# force-stops

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: in progress; awaiting fresh route replay and owner field validation.**

## What it does

Makes the car **actually stop at red lights and stop signs in experimental mode** instead of the model's indecisive crawl. The stock end-to-end model often plans a stop but never commits — its `shouldStop` bit dithers and the car creeps toward the line indefinitely. Force Stops reads the model's stop *intent* and holds it to it.

*(original concept from IQPilot's IQForceStops, reimplemented from scratch)*

## How it works

The tell is the model's planned **path length**: when it intends to stop, the path endpoint closes in to a few meters even while the stop bit flickers. A filtered detector watches for `path end < v_ego × 3s` with no lead being tracked (a wider 4.5 s window applies while the model is actively braking, to catch back-loaded lead-less red lights). When the detector latches, the model's stop point is frozen and the cruise speed is capped at what reaches zero at that point. The stock planner converts that cap into its cruise acceleration candidate while stronger MPC/e2e braking still wins. Force Stops decides *that/where*, the planner *shapes*, and (on combo) Smooth Stops *lands* the last meter.

Safety/comfort properties baked in:

- The cap feeds the stock cruise candidate, never a synthetic brake command, and cannot weaken stronger MPC/e2e braking.
- The cap may never sit more than 2 m/s below current speed, so a collapsing model path can't command a slam.
- Before the latch, a live comfort envelope (√(2·1.2·d)) shapes lead-less red-light approaches onto the owner's fitted braking curve instead of the model's late ramp.
- The latched stop point follows the model's endpoint forward at a bounded rate (so a mid-collapse latch doesn't park the car short) and, below 3 m/s, downward too (so a stale latch can't roll past the model's stop line into a crosswalk).
- Qualifying evidence refreshes a 4 s position hold, keeping the same tracked point through brief model dropouts.
- Invalid model/radar data or a raw lead releases immediately instead of letting the position hold override lead handling.
- Driver gas cancels forcing for 10 s; at standstill the module steps aside entirely and the normal hold clamp owns the stop.

## What changed

- `openpilot/selfdrive/controls/lib/force_stops.py` — the `ForceStops` class: filtered path-length detector, position hold with forward-ratchet/down-follow, comfort envelope, validity/lead release, and gas override.
- `openpilot/selfdrive/controls/lib/longitudinal_planner.py` — one hook: `v_cruise = min(v_cruise, force_stops.update(sm))`.
- `openpilot/selfdrive/controls/tests/test_force_stops.py` — deterministic position-hold, raw-lead, and invalid-model coverage.
