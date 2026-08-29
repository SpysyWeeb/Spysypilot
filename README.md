# BLoTv3

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: ⚠️ in progress — phase 1 (cruise layer) implemented 2026-08-29 and awaiting the owner's field test on `BLoTv3`; phase 2 (lead layer) implemented on `BLoTv3-phase2`, to be merged after the phase 1 verdict.**

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
  `openpilot/selfdrive/controls/tests/test_longitudinal_planner.py`. The e2e candidate only enters
  arbitration with a valid model, as in BLoTv2. Replay against BLoTv2 on routes d7, d9 and d2:
  every cruise-candidate difference is the turn budget, and the three 75↔80 mph corrections on
  d9 settle at the same +0.40 / −0.40 / +0.40 m/s² in both; lead-candidate differences are
  phase 2's. Measured cost of D4 behind a departing lead: the first step is 3× gentler and the
  launch reaches half its peak at 1.35 s instead of 0.70 s — a phase-2 field item.
- Phase 2 (2026-08-29, branch `BLoTv3-phase2`) — the lead layer. `longitudinal_lead.py`: the one
  definition of a usable lead (`LeadObservation`, `lead_present`, `relevant_lead`) and
  `anchor_model_lead`, which validates the model's lead forecast once per frame and anchors it to
  radar. `necessity_supervisor.py`: BLoTv2's supervisor with two fixes — the following-time pads
  saturate instead of vanishing above 1.5 m/s² of required deceleration, and the low-speed hold
  keeps whatever softening was built while necessity-braking (a stand-down or lead loss still
  releases it); its stand-down never reaches an alert. `long_mpc.py`: one `update()` call takes
  the anchors, the supervisor's jerk scale and pad, and a committed stop point; it sets the cost
  weights exactly once, applies the policy only while lead0 owns the solve, re-anchors the
  change cost on a handoff, and has no third-lead machinery; `STOP_DISTANCE` is 7 m and the
  aggressive personality follows at 1.0 s as in BLoTv2. The planner wires the two and the
  lead-departure pre-release. `LongitudinalPlanSource.stop` is added to cereal. Replay against
  BLoTv2 on route d7 segment 0: every lead-candidate frame now matches to the float; only the 64
  turn-budget frames from phase 1 differ.
