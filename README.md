# BLoT — Better Longitudinal Tune

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

Full architecture doc: [`docs/BLoT.md`](docs/BLoT.md). Companion lateral branch: [`BLaT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLaT).

## What it does

Makes braking and acceleration feel like the owner's own driving: commit to slowdowns early and smoothly, hold braking no longer than necessary, release as one human-like taper, react to lead takeoffs in seconds-before-radar time, and keep follow distance stable. It builds on comma's model-predicted lead trajectories PR ([#37824](https://github.com/commaai/openpilot/pull/37824)): the driving model's `leadsV3` forecast — not radar extrapolation — is what the MPC plans against, and BLoT's guiding rule is to **drive the MPC's own knobs, never wrap its output**.

## How it works

Three cooperating pieces:

- **BLT necessity supervisor** (`blt.py`) — a watchdog that modulates the MPC's per-frame runtime knobs (`jerk_factor`, `t_follow`, an `aLeadTau` floor) instead of overriding its commands. A *recovery boost* relaxes the solver when braking is provably no longer necessary (measured excess, or the model forecasting the lead pulling away), a *launch boost* stiffens response when the plan lags a genuinely accelerating lead, and a *whiplash ratchet* lets stiffness relax at any time but never re-stiffen while the lead is braking toward us. Triggers share a debounced arm/hold/reset primitive (`DebouncedTrigger`).
- **Model-predicted lead trajectories** (`long_mpc.py`) — the MPC plans against the driving model's `leadsV3` forecast, currently in the *unmodified* PR #37824 form: the earlier forecast trust ledger and radar pull-away floor were removed 2026-07-18 for a stock-baseline A/B field test (both preserved in git history; see docs/BLoT.md).
- **Launch chain** (`blt.py` + `longitudinal_planner.py` + `longcontrol.py` + opendbc) —
  while stopped behind a tracked lead, a sustained radar-anchored model prediction of
  departure pre-releases only the MPC hold so the Palisade's measured brake bleed can
  start early. The starting state then issues a *proportional* command,
  `clip(a_target, 0.6, startAccel)`, so `startAccel` (raised to 1.5 on the opendbc side,
  with starting-state jerk limit 5.0) is a ceiling the plan can use, not a fixed shove.

## What changed

- `openpilot/selfdrive/controls/lib/blt.py` *(new)* — the necessity supervisor and `DebouncedTrigger`.
- `openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py` — model-predicted lead trajectories (PR #37824, currently stock form — ledger/floor removed for A/B testing), base-tune constants.
- `openpilot/selfdrive/controls/lib/longitudinal_planner.py` — BLT wiring into the MPC's weights, raised accel limits.
- `openpilot/selfdrive/controls/lib/longcontrol.py` — proportional starting command.
- `openpilot/selfdrive/controls/tests/test_blt.py` *(new)* — unit tests on trigger,
  ratchet, onset, and standstill pre-release semantics.
- `docs/BLoT.md` *(new)* — full architecture, retired interventions, and the "no crutches" doctrine.
- `opendbc_repo` (SpysyWeeb/opendbc, branch `BLoT`) — Hyundai `startAccel` 1.5 and starting-state CAN jerk limit 5.0.

## Automatic longitudinal event logging

- `long_event_detector.py` is a pure, lightweight state machine. It separates predicted
  lead motion, measured lead motion, MPC hold release, post-controller acceleration,
  and actual ego movement, then attributes a late launch to planner, controller, or
  vehicle response.
- `feedbackd.py` feeds the detector current logged signals. Automatic detections and
  manual bookmark presses publish a structured `userBookmark`; no control command,
  planner trajectory, or safety state is modified.
- `loggerd` already preserves the current segment whenever `userBookmark` is published.
- `long_eventd.py` runs only off-road, indexes structured bookmarks from completed
  rlogs, preserves two preceding segments plus one following segment, and writes
  `/data/community/long_events/manifest.jsonl`.
- `events.py` displays the bookmark's structured text. Generic/audio bookmarks retain
  the stock **Bookmark Saved** message.

Automatic detections and manual bookmark presses show **Long Event Logged** without
changing any control command, planner trajectory, or safety state. Full thresholds,
manifest format, and the SSH workflow are documented in
[`docs/AutoLongLogger.md`](docs/AutoLongLogger.md).

Step 1 of this effort was [op-model-grader](https://github.com/SpysyWeeb/op-model-grader), a standalone tool that grades the model's longitudinal (and lateral) performance from rlogs against the owner's manual driving.
