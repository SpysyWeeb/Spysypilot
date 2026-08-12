# force-stops

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: in progress; awaiting fresh route replay and owner field validation.**

## What it does

Makes the car **actually stop at red lights and stop signs in experimental mode** instead of the model's indecisive crawl. The stock end-to-end model often plans a stop but never commits — its `shouldStop` bit dithers and the car creeps toward the line indefinitely. Force Stops reads the model's stop *intent* and holds it to it.

*(original concept from IQPilot's IQForceStops, reimplemented from scratch)*

## How it works

The tell is the model's planned **path length**: when it intends to stop, the path endpoint closes in to a few meters even while the stop bit flickers. A filtered detector watches for `path end < v_ego × 3s` with no lead being tracked (a wider 4.5 s window applies while the model is actively braking, to catch back-loaded lead-less red lights). A bounded StarPilot-style kinematic profile starts reducing the cruise target before the latch. After commitment, the same tracked stop point becomes an optional native MPC obstacle. MPC retains it through the 6 m profile boundary and the rolling landing, until Force Stops itself releases. Force Stops decides *that/where*, the planner/MPC *shapes*, and (on combo) Smooth Stops *lands* the last meter.

Safety/comfort properties baked in:

- Force Stops never emits acceleration or brake commands: its speed cap and committed position are inputs to the native planner/MPC, where closer leads and stronger e2e braking still win.
- The speed cap may never sit more than 2 m/s below current speed; after commitment, the position constraint lets MPC plan directly to the tracked point.
- The kinematic profile is `√(2·0.65·max(d−6, 0))`, matching StarPilot's model-stop approach without changing the active MPC's lead-spacing calibration.
- The latched stop point follows the model's endpoint forward at a bounded rate (so a mid-collapse latch doesn't park the car short) and, below 3 m/s, downward too (so a stale latch can't roll past the model's stop line into a crosswalk).
- Qualifying evidence refreshes a 4 s position hold, keeping the same tracked point through brief model dropouts.
- Invalid model/radar data or either raw radar lead releases immediately instead of letting the position hold override lead handling.
- Driver gas bypasses all shaping and cancels forcing for 10 s; at standstill the module steps aside entirely and the normal hold clamp owns the stop.

## What changed

- `openpilot/selfdrive/controls/lib/force_stops.py` — the `ForceStops` class: filtered path-length detector, position hold with forward-ratchet/down-follow, comfort envelope, validity/lead release, and gas override.
- `openpilot/selfdrive/controls/lib/longitudinal_planner.py` — applies the cap and passes only the committed remaining position to MPC.
- `openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py` — accepts the optional stop obstacle and preserves closer-lead priority; planner keeps it until Force Stops releases.
- `openpilot/selfdrive/controls/tests/test_force_stops.py` — deterministic position-hold, driver/lead release, invalid-model, profile, MPC-position, lead-priority, and handoff coverage.
