# BLaT — Better Lateral Tune

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

Companion longitudinal branch: [`BLoT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLoT).

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
