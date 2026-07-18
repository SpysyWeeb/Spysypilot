# BLoT — Better Longitudinal Tune

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

Full architecture doc: [`docs/BLoT.md`](docs/BLoT.md). Companion lateral branch: [`BLaT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLaT).

## What it does

Makes braking and acceleration feel like the owner's own driving: commit to slowdowns early and smoothly, hold braking no longer than necessary, release as one human-like taper, react to lead takeoffs in seconds-before-radar time, and keep follow distance stable. It builds on comma's model-predicted lead trajectories PR ([#37824](https://github.com/commaai/openpilot/pull/37824)): the driving model's `leadsV3` forecast — not radar extrapolation — is what the MPC plans against, and BLoT's guiding rule is to **drive the MPC's own knobs, never wrap its output**.

## How it works

Three cooperating pieces:

- **BLT necessity supervisor** (`blt.py`) — a watchdog that modulates the MPC's per-frame runtime knobs (`jerk_factor`, `t_follow`, an `aLeadTau` floor) instead of overriding its commands. A *recovery boost* relaxes the solver when braking is provably no longer necessary (measured excess, or the model forecasting the lead pulling away), a *launch boost* stiffens response when the plan lags a genuinely accelerating lead, and a *whiplash ratchet* lets stiffness relax at any time but never re-stiffen while the lead is braking toward us. Triggers share a debounced arm/hold/reset primitive (`DebouncedTrigger`).
- **Forecast trust ledger** (`long_mpc.py`) — the model's lead forecast is trusted fully by default (= stock PR behavior); radar is an *auditor*, not a gatekeeper. A per-frame ledger accrues debt when the forecast promises acceleration the radar-measured lead never delivers, and pays it down exponentially the moment radar corroborates real movement. Trust below 1.0 blends the MPC's lead trajectory toward a radar constant-velocity continuation — so a wrong forecast degrades gracefully (surge–soften–resume) instead of either ignoring radar or slamming the brakes. Kept from the earlier iterations: the radar pull-away floor.
- **Launch chain** (`longcontrol.py` + opendbc) — the starting state issues a *proportional* command, `clip(a_target, 0.6, startAccel)`, so `startAccel` (raised to 1.5 on the opendbc side, with starting-state jerk limit 5.0) is a ceiling the plan can use, not a fixed shove.

## What changed

- `openpilot/selfdrive/controls/lib/blt.py` *(new)* — the necessity supervisor and `DebouncedTrigger`.
- `openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py` — `LeadTrustLedger`, forecast/radar blend in `process_lead_model`, pull-away floor.
- `openpilot/selfdrive/controls/lib/longitudinal_planner.py` — BLT wiring into the MPC's weights, raised accel limits.
- `openpilot/selfdrive/controls/lib/longcontrol.py` — proportional starting command.
- `openpilot/selfdrive/controls/tests/test_blt.py` *(new)* — 24 unit tests on trigger/ratchet/onset semantics.
- `docs/BLoT.md` *(new)* — full architecture, retired interventions, and the "no crutches" doctrine.
- `opendbc_repo` (SpysyWeeb/opendbc, branch `BLoT`) — Hyundai `startAccel` 1.5 and starting-state CAN jerk limit 5.0.

Step 1 of this effort was [op-model-grader](https://github.com/SpysyWeeb/op-model-grader), a standalone tool that grades the model's longitudinal (and lateral) performance from rlogs against the owner's manual driving.
