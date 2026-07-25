# BLaT — Better Lateral Tune

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

Companion longitudinal branch: [`BLoT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLoT).

## Status

⚠️ **In progress.** BLaT has returned to the field-tested final controller v14
baseline from commit `e1a010eeef`. The v15–v19 committed-handoff, high-angle
release, breakout, remembered-intent, and catch-up interventions remain
removed.

The next architecture restructure has emitted and versioned its Phase 0
counterfactual replay baseline from frozen routes `0000008f--429bc635f2` and
`0000009b--217a1b70db` in route-audit commit `acdc165`. Phase 1 controller v15
remains in progress. Its scalar-anchored washout, one-replan quarantine, and
anchored-rate adapter have provisionally cleared the frozen-route replay gate;
the result remains subject to regeneration against the exact committed build
and a separate field drive followed by explicit sign-off. Later phases remain
in progress under the same replay-and-field requirements.

Replay rows are identified by both the Spysypilot commit and opendbc commit,
not by the logged controller `VERSION` alone. Beginning with v15, every
behavior-affecting controller change must bump `VERSION` in the same commit;
two builds that steer differently must never report the same version.

For v15 above 12 mph, the raw planner trajectory is washout input only; every
surviving downstream reference consumer must follow the final scalar-anchored
reference. Accordingly, `trajectoryReferenceRateValid = false` is nominal
above 12 mph and selects the controller's finite-difference rate from the
anchored command through the existing 0.18-second reference-rate innovation
filter—it does not indicate planner failure. The v14 trajectory-rate path
remains unchanged through 12 mph. The unanchored unwind and torque-target
diagnostics are a documented temporary exception pending their Phase 2
deletion.

Phase 2 must derive the slew-feasible feedforward's rate content analytically
from the plan through the scalar-anchor transform. Numerically differentiating
a replan-discontinuous command is prohibited: v15 demonstrated that even a
filtered finite difference can turn model-update boundaries into actuator-rate
roughness. This requirement carries forward independently of the temporary
v15 rate-cascade adapter, which Phase 3 deletes with the cascade itself.

The cumulative Phase 1 controller-source diff is +276/-65 lines (net +211),
within the approved Phase 1 whitelist. Phase 2's mandated mechanism deletions
must make the cumulative Phase 1–2 controller-source total net-negative at its
gate.

Automated rollback validation currently covers 87 lateral-controller/reference
tests, 16 Hyundai damping tests, and 1,367 Hyundai panda safety tests. These
checks confirm the restored V14 code and bounds, not its on-road steering feel.

## What it does

BLaT makes steering smooth without making it uniformly slow. It combines the
model's future path with measured wheel motion and the torque the Hyundai EPS
can actually receive through its fixed slew limits.

The controller:

- derives an all-speed curvature and steering-rate reference from the future
  model trajectory;
- aligns delayed lateral feedback to the car's current speed, avoiding false
  P error while the driver accelerates or brakes;
- uses a position/rate cascade so the wheel can move quickly for a real turn
  without blindly following short torque spikes;
- predicts whether requested torque is reachable through the Hyundai slew
  limiter before the path needs it;
- starts and hands off unwind episodes using future geometry, crown-neutral
  torque, wheel rate, and applied-torque delivery state;
- preserves turn-in authority when the wheel is still behind the planned path;
- records versioned controller diagnostics in `LateralTorqueState` for rlog
  analysis.

The Hyundai command and safety limits are also raised from stock **(384, 3, 7)**
to **(409, 4, 7)**:

- maximum steering torque: 384 → 409;
- torque build rate: 3 → 4 counts per frame;
- torque decay remains 7 counts per frame.

## How it works

The openpilot side is owned by this branch:

- `openpilot/selfdrive/controls/lib/lateral_reference_planner.py` builds the
  future trajectory reference and actuator-reachable torque preview.
- `openpilot/selfdrive/controls/lib/latcontrol_torque.py` applies the
  speed-aligned P loop, rate cascade, actuator-state correction, and
  future-unwind control.
- `openpilot/selfdrive/controls/controlsd.py` connects model trajectory,
  actuator limits, applied torque, and the torque controller.
- `openpilot/cereal/log.capnp` carries the BLaT diagnostics.

The branch also points `opendbc_repo` to
[SpysyWeeb/opendbc BLaT](https://github.com/SpysyWeeb/opendbc/tree/BLaT).
That branch owns the matching Hyundai command/safety limits and the low-speed
EPS-motion damping layer. Both command and panda safety limits must agree or
panda rejects openpilot's requests.

## What changed

- opendbc BLaT submodule pointer;
- torque-controller and future-reference implementation;
- controller integration and rlog diagnostics;
- three focused test suites covering delayed feedback, actuator/rate/unwind
  behavior, and the trajectory reference planner.

The universal driving-event logger is intentionally separate infrastructure.
BLaT supplies the lateral behavior and diagnostics that it observes.
