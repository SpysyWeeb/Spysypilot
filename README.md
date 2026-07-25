# BLaT — Better Lateral Tune

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

Companion longitudinal branch: [`BLoT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLoT).

## Status

⚠️ **In progress.** The implementation and focused automated tests are in
place, but BLaT remains in field validation. It is not considered done until
the behavior has been tested on the car and explicitly signed off.

Active tuning work:

- **In progress:** add angle-scaled exit urgency for low-speed, high-angle
  unwinds. Route `00000093--3198e2f719` showed that the future unwind was
  recognized before peak steering angle, but old-turn torque remained for
  roughly 1.2–1.4 seconds before crossing crown-neutral. The change must remove
  stale holding torque earlier without raising the existing torque or safety
  limits, and remains subject to replay and field validation.
- **In progress:** refine that exit using field route
  `00000094--8cec74a749`. Controller v16 removed stale torque safely, but its
  falling angle scale could reintroduce old-turn torque, its 200–280 degree
  release was sometimes too weak, and crown-neutral alone did not free the
  most tightly wound EPS cases. Controller v17 now protects immediate turn-in
  demand, keeps confirmed release monotonic, strengthens the 200–280 degree
  range, and permits only a bounded, evidence-gated breakout assist. Replay is
  complete, but the change remains in progress pending another field test.
- **In progress:** correct controller v17 using field route
  `00000095--23c2a1f7a2`. V17's rate-only present-demand guard could disappear
  while a strong turn merely plateaued, allowing its irreversible release
  latch to unwind 10 of 12 usable episodes too early. Its breakout assist also
  stopped after 30 degrees of progress while the EPS remained deeply wound.
  V18 must require a sustained present-turn release before latching, retain the
  monotonic ceiling only after that commitment, and hold the existing bounded
  breakout until rate recovery or the angle exit. Route 94/95 open-loop replay
  is complete; the task remains in progress pending another field test.
- **In progress:** validate controller v19 from low-speed field route
  `00000098--0cae131b10`. V19 adds a bounded, model-intent catch-up surge
  only after 200 ms of measured turn-in or unwind underperformance, connects
  the existing high-angle present-demand guard to Hyundai damping protection,
  and derives signed wheel motion from steering-angle change instead of the
  platform's unsigned steering-rate signal. The surge remains capped at 0.15
  normalized torque and cannot exceed the existing command, slew, or panda
  safety limits. Implementation, automated validation, and route replay are
  complete; this task remains in progress pending a field test.

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
- latches an angle-scaled old-direction ceiling when a same-episode, low-speed
  unwind is confirmed, then uses a separately gated, bounded breakout target
  only if an extreme-angle EPS remains stuck at crown-neutral;
- preserves turn-in authority when the wheel is still behind the planned path;
- applies one bounded, model-direction catch-up pulse only after a deeply wound
  wheel has underperformed for 200 ms during an extreme turn-in or unwind;
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
- during a confirmed high-angle unwind below 20 mph, a short persistence gate
  progressively lowers a crown-relative old-direction ceiling in proportion to
  steering angle, unwind rate, and sustained future-path evidence; the ceiling
  cannot rise again while that episode remains owned;
- a delay-aligned present-demand guard retains turn-in authority while the
  immediate old-direction request is still building and under-tracked;
- only after an extreme-angle wheel reaches applied crown-neutral and fails a
  300 ms progress check may a separate breakout target request up to 0.15
  normalized torque in the unwind direction;
- separately, a model-intent catch-up observer may request one ramped pulse
  after 200 ms of severe position/rate underperformance in an extreme turn-in
  or unwind; it uses steering-angle delta for signed wheel rate, shares the
  breakout controller's 0.15 normalized-torque budget, and aborts for driver
  input, intent loss, recovery, timeout, or episode loss;
- the immediate high-angle present-demand guard is also wired into Hyundai's
  low-speed damping gate so EPS damping cannot suppress a still-needed turn-in;
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

Before a direction handoff, high-angle exit urgency can act within the current
episode when the wheel and future reference both show a sustained unwind. It
starts above 120 degrees of old-turn steering angle, scales with angle and
unwind rate, is limited to low speed, and freezes integral growth while active.
Full one-sided release is available by 240 degrees, and confirmed authority is
latched as a monotonically tightening crown-relative ceiling. Brief evidence
dropout is tolerated for 250 ms without restoring old-direction torque. The
last ceiling bridges into a committed direction handoff, while breakout
authority resets immediately. Loss of lateral control or driver input resets
the full state immediately.

At angles above 300 degrees, a separate EPS breakout stage can act only after
applied torque has reached crown-neutral and the wheel has failed to unwind at
least 30 degrees during a 300 ms progress window. It ramps toward at most 0.15
normalized opposite torque, never weakens an already stronger unwind command,
and is blocked by building, under-tracked present turn demand.

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
| torque controller | 19 | delayed curvature and demand rate, trajectory/measured rate, unwind ownership, monotonic old-direction ceiling, signed wheel rate, catch-up mode/qualification/correction/termination, breakout progress, present-demand guard, and committed-handoff cap |
| Hyundai EPS damping | 3 | requested/applied damping, gate state, signed wheel rate, breakaway stall and relief state |

The universal driving-event logger remains separate infrastructure. BLaT
supplies the behavior and diagnostics that it observes.

## Validation

Focused tests were rerun against controller v19, the high-angle unwind exit,
committed-handoff, and EPS-breakaway implementation with opendbc pin
`8876af47`:

| suite | result |
|---|---:|
| delayed-feedback, reference-rate, catch-up, actuator, unwind, handoff, request-buffer, and future-reference planner tests | 192 passed |
| Hyundai low-speed torque damping and breakaway tests | 20 passed |
| Hyundai panda safety tests | 1367 passed, 162 skipped by suite configuration |
| **total passing** | **1579 passed** |

The suites cover symmetry, bounds, speed schedules, invalid-model fallback,
torque reachability, crown-neutral transitions, episode handoff, driver
override, under-tracked turn-in protection, post-stall relief, stale-direction
handoff limiting, one-sided high-angle unwind limiting, exact 200 ms catch-up
qualification, single-pulse cooldown/rearm, correction direction and shared
0.15 authority bounds, command/safety rate limits, and diagnostic schema
exposure.

An open-loop command-path replay of route `00000093--3198e2f719` modeled
crown-neutral crossings about 0.71 seconds earlier near 12:36:39 and 0.95
seconds earlier near 12:41:16. The cap remained one-sided in all three inspected
windows, including the weaker 12:38:30 event. This replay verifies controller
direction and gating against the logged inputs; it does not predict closed-loop
vehicle response and must be followed by a field test.

A second open-loop replay processed all 70,631 controller samples from route
`00000094--8cec74a749`. Across 19 logical release intervals, the latched
old-direction ceiling never increased, no release or breakout correction had
the wrong sign, and breakout never appeared with driver input, lost episode
ownership, or a committed handoff. Two recorded extreme-angle stalls satisfied
the counterfactual breakout gates, peaking at 0.048 and 0.019 normalized torque.
The present-demand guard also activated in the identified early-release
windows. Because recorded wheel motion came from controller v16, these results
verify v17 command invariants and eligibility—not closed-loop improvement.

A counterfactual observer replay processed 23,999 controller samples from
low-speed route `00000098--0cae131b10`. The intended extreme unwind windows at
1758 and 2487 seconds and the extreme turn-in at 2485 seconds sustained the
200 ms qualification and received correctly signed corrections no larger than
0.15 normalized torque. Candidate evidence at 2049 and 2196 seconds did not
persist for 200 ms and therefore did not trigger. This uses recorded v18 wheel
motion, so it validates v19 eligibility, direction, persistence, and bounds;
closed-loop improvement still requires the pending field test.

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
