# BLaT — Better Lateral Tune

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

Companion longitudinal branch: [`BLoT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLoT).

## Status

⚠️ **In progress.** The implementation and focused automated tests are in
place, but BLaT remains in field validation. It is not considered done until
the behavior has been tested on the car and explicitly signed off.

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

Those limits apply to the default classic-CAN Hyundai safety profile used by
the Palisade. Existing CAN-FD, `ALT_LIMITS`, `ALT_LIMITS_2`, and vehicle-specific
lower command caps remain in place.

## How it works

### 1. Future-path reference

`openpilot/selfdrive/controls/lib/lateral_reference_planner.py` reconstructs a
continuous 1.5-second curvature trajectory from `modelV2.orientation.z` and
`modelV2.velocity.x`. It anchors predicted speed to `carState.vEgo`, fits the
model yaw horizon, and regularizes the solution differently at low and high
speed:

- low speed favors smooth curvature and preserves the proven sharp-turn
  behavior;
- high speed minimizes physical lateral jerk and jerk rate;
- every solution is anchored to current measured curvature and the previous
  time-shifted solution;
- malformed, non-finite, or physically implausible model trajectories reset
  the planner and fall back to the model's scalar curvature action.

The base path sample includes live lateral delay, openpilot's lateral smoothing
delay, the model action offset, and the age of the current solution. On Hyundai
torque-control cars, the planner also receives the real
`STEER_MAX/STEER_DELTA_UP/STEER_DELTA_DOWN/STEER_STEP` values and constructs a
backward rate-reachable torque envelope.

The actuator preview is deliberately asymmetric: it may look farther ahead to
prepare for a sustained release, but it does not look past turn-in and erase
needed steering authority. Preview correction is bounded in lateral-acceleration
space, while meaningful under-tracked turn-in is allowed to retain the stronger
reference.

### 2. Torque controller

`openpilot/selfdrive/controls/lib/latcontrol_torque.py` performs control in
lateral-acceleration space:

- delayed feedback buffers curvature, then evaluates desired and measured
  curvature at the same current speed; this removes false P error caused by
  accelerating or braking through a constant curve;
- feedforward projects speed over the bounded steering delay, with a separate
  cap on the resulting lateral-acceleration correction;
- a position/rate cascade combines the future trajectory's curvature rate,
  measured wheel curvature rate, and bounded position-error catch-up;
- a residual direct-P schedule retains high-speed position authority while
  moving more low-speed work into the rate loop;
- applied torque is treated as actuator state and only opposes torque that is
  still driving a wheel already outrunning the planned path;
- after the future horizon confirms a new maneuver for 200 ms, a crown-aware
  handoff cap progressively removes only controller command that still points
  into the old episode beyond the reachable target.

The future trajectory rate is used directly instead of differentiating every
20 Hz model replan. A low-frequency innovation path retains genuine changes in
the final command without turning small replan steps into steering-rate chatter.

### 3. Unwind episodes

The controller predicts where the wheel will be when already-applied torque can
reach crown-neutral torque. An `UnwindPhaseTracker` keeps one geometric maneuver
in control until torque delivery and wheel motion catch up, then restores direct
P smoothly.

Episode handoff requires the selected geometric target and a later horizon
sample to agree on the new direction for a sustained interval. This prevents a
single near-center sample, road crown, or friction sign flip from being mistaken
for a new turn. Once that commitment is confirmed, the old-direction cap ramps
in over 150 ms and can only subtract stale command; command already braking or
steering into the new maneuver is left unchanged. Driver steering immediately
resets the unwind state.

### 4. Hyundai EPS damping and panda safety

The branch points `opendbc_repo` to
[SpysyWeeb/opendbc BLaT](https://github.com/SpysyWeeb/opendbc/tree/BLaT), pinned
at `8876af47` for this implementation. That branch owns:

- the matching Hyundai command and panda safety limits;
- low-speed EPS-motion-gated torque damping;
- controller-output damping diagnostics in `CarControl.Actuators`.

Low-speed damping is fully available through 12 mph and fades to zero by
15 mph. It activates only after requested torque, measured EPS torque, and
signed wheel motion agree that static friction has been broken. Its adaptive
breakaway floor can limit how much damping subtracts, but can never hold the
command above the torque controller's current demand. A turn-in guard disables
damping whenever the wheel is meaningfully behind a same-direction path that
has not begun to unwind.

The turn-in guard also observes loaded stalls without weakening the torque that
is trying to break the rack free. If at least 150 ms of stationary, aligned EPS
behavior ends in a directionally consistent release above 30 degrees/second, a
bounded 300 ms breakaway-relief window becomes eligible. It uses the existing
speed fade and adaptive floor, never adds authority, and immediately resets for
driver input, sign disagreement, inactive control, or speeds at and above
15 mph.

Both the openpilot command layer and panda safety layer must agree on steering
limits. Changing only one side either has no effect or causes panda to reject
the request.

### 5. Integration and fallback behavior

`openpilot/selfdrive/controls/controlsd.py` connects the model trajectory,
actuator limits, applied torque, live torque parameters, and the torque
controller.

- Hyundai torque-control cars enable the full actuator preview, rate cascade,
  unwind state machine, and damping handoff.
- Other lateral controllers use the non-actuator future curvature reference
  where valid. Other torque platforms also receive the speed-aligned feedback
  improvements, but retain legacy actuator behavior until their delivery
  limits are measured.
- Lateral maneuver plans remain authoritative and reset the future-path
  planner.
- Invalid model data and inactive lateral control reset planner/controller
  state instead of retaining a stale future reference.
- `clip_curvature` remains the final openpilot curvature rate/acceleration
  boundary.

The BLaT implementation was merged into `combo` in
[`8dc765130`](https://github.com/SpysyWeeb/Spysypilot/commit/8dc765130).
`combo` wraps it with SOL/AOL and the fork's longitudinal features; changes
must continue to land in BLaT first and then be integrated without replacing
those combo-specific control paths.

## Diagnostics

`openpilot/cereal/log.capnp` carries enough attribution to separate model-path,
feedback, actuator-delivery, damping, and controller effects in an rlog.

Current diagnostic versions:

| layer | version | examples |
|---|---:|---|
| future reference planner | 5 | base/output curvature, preview timing, reachable/geometric/episode torque, unwind confidence |
| torque controller | 15 | delayed curvature, trajectory/measured rate, cascade terms, unwind ownership, committed-handoff cap |
| Hyundai EPS damping | 3 | requested/applied damping, gate state, signed wheel rate, breakaway stall and relief state |

The universal driving-event logger remains separate infrastructure. BLaT
supplies the behavior and diagnostics that it observes.

## Validation

Focused tests were rerun against the committed-handoff and EPS-breakaway
implementation with opendbc pin `8876af47`:

| suite | result |
|---|---:|
| delayed-feedback, reference-rate, actuator, unwind, and handoff controller tests | 55 passed |
| future-reference planner tests | 34 passed |
| Hyundai low-speed torque damping and breakaway tests | 20 passed |
| Hyundai panda safety tests | 342 passed, 55 skipped by suite configuration |
| **total passing** | **451 passed** |

The suites cover symmetry, bounds, speed schedules, invalid-model fallback,
torque reachability, crown-neutral transitions, episode handoff, driver
override, under-tracked turn-in protection, post-stall relief, stale-direction
handoff limiting, command/safety rate limits, and diagnostic schema exposure.

Automated tests are not a substitute for route analysis and field testing of
steering feel.

## Development history

The first BLaT changes raised Hyundai steering authority and build rate. A
later split-band low-pass P experiment reduced low-speed high-frequency torque
in replay, but it was intentionally reverted. The current consolidated
future-path controller supersedes that approach; the reverted split-band P
should not be treated as part of the active tune.

The implementation is primarily contained in:

- `openpilot/selfdrive/controls/lib/lateral_reference_planner.py`;
- `openpilot/selfdrive/controls/lib/latcontrol_torque.py`;
- `openpilot/selfdrive/controls/lib/lateral_torque_utils.py`;
- `openpilot/selfdrive/controls/controlsd.py`;
- `openpilot/cereal/log.capnp`;
- the pinned SpysyWeeb/opendbc BLaT commit.
