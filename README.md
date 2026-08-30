# smooth-stops

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: in progress — thin handoff rewrite of 2026-08-29, not yet field tested.**

## What it does

Kills the **"headbang" at the end of every stop**. Stock openpilot jumps to the `stopping` state the moment the
planner's stop bit sets (v < 0.3 m/s) and ramps the brake command to −2.0 m/s² *while the car is still rolling*. This
branch waits for a true standstill before the clamp, and owns the last few decimetres with two jobs: keep at least a
gentle "kiss" of braking on so the stop completes, and ratchet the pressure up if the car stops making progress.

*(personal idea)*

## How it works

`SmoothStopController` inside longcontrol (control rate, true `v_ego`):

- **Hold only once stopped** — the stopping/hold clamp arms at ≤ 0.05 m/s, or on the car's standstill flag while it is
  slow enough (≤ 0.15 m/s) to be believed.
- **The landing** — while the planner's stop bit is set and the car still rolls, the command is the planner's request
  bounded from below by the kiss (0.12 m/s²), jerk-limited at 2.5 m/s³. Harder planner braking passes straight through.
  If the car has not got slower for 0.5 s, 0.5 m/s² per stalled second is added until it does.
- **Release** — once holding, two frames of the stop bit clearing release the hold. The planner already corroborates a
  green; an audit of ~22 minutes of standstill found no stop-bit flicker to guard against.

What was here before (an entry-anchored taper, a lead floor, a queue-aware anti-creep ratchet with hysteresis and
radar-dropout grace, a radarState subscription in controlsd) never triggered in the 2026-08-29 field audit of 22 stops
and re-derived lead physics the planner owns; it was removed. Below ~0.3 m/s the Palisade's ESP brings the car to rest
with its own brake whatever is requested — the last moment of a stop is the car's.

## What changed

- `openpilot/selfdrive/controls/lib/smooth_stops.py` *(new)* — the `SmoothStopController`: hold arm/release and the landing.
- `openpilot/selfdrive/controls/lib/longcontrol.py` — the stopping-state transition and hold release route through the
  controller; the pid branch lands the car while the plan's stop bit is set.
- `openpilot/selfdrive/controls/radard.py` — the unconfirmed low-speed lead override needs a track that was tracked in
  from a distance (a separate fix that lived here: a 0.8 s ghost 3.6 m ahead once max-braked a stop).

History note: this branch once also carried Smooth Approach / Smooth Release wrappers for braking further out; those were retired in favor of the [`BLoT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLoT) supervisor, which drives the MPC's own knobs instead of wrapping its output. This branch is now the stop-landing piece only.
