# BLoTv3

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: ⚠️ in progress — phase 1 (cruise layer) implemented 2026-08-29, awaiting the owner's field test; phase 0 (branch, docs, honest harness, replay tooling) done.**

## What it does

BLoTv3 is the rewrite of the Palisade longitudinal tune
([BLoTv2](https://github.com/SpysyWeeb/Spysypilot/tree/BLoTv2)). The road behavior BLoTv2 got
right is kept — the continuous cubic acceleration envelope, ordinary-cruise comfort shaping, the
necessity supervisor that softens the MPC's jerk cost and pads following time only when a lead
requires it, radar-anchored model lead trajectories, lead-departure pre-release, Conditional
Experimental Mode for lead-free model stops, and Force Stops' committed stop point — but every
longitudinal decision gets exactly one owner in exactly one process, the code takes the shape
upstream uses for the planner, and the defects found in the 2026-08-28 review of BLoTv2 are fixed:

- Conditional Experimental Mode could only ever auto-engage when the radar saw **no** vehicle at
  all (entry evidence was wiped on every control tick a lead existed, at any distance).
- A committed stop had no owner once the car stood still, so `shouldStop` followed the model's
  flicker and could pulse the car's own ACC resume.
- The supervisor's stand-down raised the FCW "BRAKE!" alert in a single frame with no vision
  confirmation; the FCW distance check silently moved from radar extrapolation to the model path.
- Any pedal tap switched a manually enabled Experimental mode off for two seconds.
- The headway pad snapped to zero exactly as the required deceleration crossed 1.5 m/s², and the
  low-speed jerk hold only engaged if the scale had reached its exact floor.
- The turn budget was raised to a flat 4.0 m/s² and never limited anything; the `longcontrol.py`
  clamp could not bind in either opendbc configuration; the third-lead "ponytail" reset the
  supervisor's policy while lead0 was still the nearest car.
- `force_stops.py` existed in three divergent copies (`force-stops`, `BLoTv2`, `combo`).

## How it works (planned)

- **selfdrived** keeps owning the effective Chill/Experimental mode, exactly where stock publishes
  it. `ConditionalExperimentalMode` shrinks to that one job: mode request in, no stop fields out.
  It still steps every control tick (so a hung model still releases the mode) and only looks at
  new model frames for evidence.
- **plannerd** owns everything else. `stop_helpers.py` is one stateless classifier of model stop
  intent, imported by both processes, so the only signal that crosses processes is stock's
  `selfdriveState.experimentalMode`. `force_stops.py` becomes the sole owner of shaping →
  commit → **hold through standstill** → release; the four `conditionalStop*` cereal fields are
  retired. `necessity_supervisor.py` (BLoTv2's `blotv2.py`) keeps its triggers and fixes the two
  cliffs. `longitudinal_lead.py` is the one definition of a usable lead and anchors the model
  lead once per frame. `long_mpc.py` gets one `update()` call that resolves obstacles, sets the
  cost weights once, solves, and scores FCW against the trajectory it actually solved.
- `radard.py`, `longcontrol.py` and `controlsd.py` are not edited. Stock opendbc limits apply on
  this branch (2.0 m/s²); `combo`'s opendbc/panda lineage supplies the 4.0 m/s² envelope.
- Design, owner decisions, module contracts, the hold state machine rules and the acceptance
  gates live in [`docs/BLoTv3.md`](docs/BLoTv3.md). Phases, each field-tested by the owner before
  the next: (0) branch, docs, harness, replay tooling; (1) cruise layer; (2) lead layer;
  (3) stop layer; (4) staged integration into `combo`.

## What changed

- Phase 0 (2026-08-29): branch cut from `stock` `511f2b60b4`; `docs/BLoTv3.md` added; the
  longitudinal maneuver harness got a real `all_checks()` shim, radar/model validity schedules,
  `selfdriveState.enabled`, and three distinct, fully populated `leadsV3` messages.
- Phase 1 (2026-08-29) — `openpilot/selfdrive/controls/lib/longitudinal_planner.py`: the cruise
  acceleration ceiling is the continuous cubic envelope `0.6 + 3.4 (1 − v/40)³`, clamped by the
  deployed opendbc `ACCEL_MAX`; the jerk schedule is `[2.0, 1.6, 1.0, 0.6] m/s³`; ordinary Chill
  cruise above 15 m/s uses the proportional comfort target (5 mph ≈ 0.40 m/s², pitch-compensated
  coast on reductions, blended in from 8 m/s) whenever the radar is healthy; the turn budget is
  `max(envelope, stock [1.7, 3.2])` so a straight launch is never clipped but cornering consumes it
  again; the MPC's acceleration-change cost stays on through standstill. Behavioral tests in
  `openpilot/selfdrive/controls/tests/test_longitudinal_planner.py`. Replay against BLoTv2 on
  route d7 segment 0: every cruise-candidate difference is the turn budget; lead-candidate
  differences are phase 2's.
