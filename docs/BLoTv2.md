# BLoTv2 — modular longitudinal control

## Status

**In progress. Not field validated. Do not mark complete before owner field
testing and explicit approval.**

BLoTv2 starts from the untouched `stock` branch and deliberately reimplements
the useful mechanisms from `BLoT` and `smooth-stops`. It is not based on
`combo`, and it does not inherit either feature branch wholesale.

The product target is **Smooth. Swift. Strong.**

- **Smooth:** continuous acceleration and jerk through lead following, braking,
  final approach, hold, and release.
- **Swift:** react promptly to real lead/model changes without stale trigger
  state, unnecessary solution stiffness, or hidden brake-release delay.
- **Strong:** use the existing safe longitudinal envelope in proportion to
  need, while preserving planner emergency braking and platform limits.

## Sole-owner contract

| Decision | Owner |
|---|---|
| Radar/model lead selection | `radard` and the driving model |
| Finite/live filtered lead observation | `longitudinal_lead.py` |
| Lead trajectory and obstacle optimization | stock Acados longitudinal MPC |
| MPC response cost and dynamic headway | `BLoTv2Supervisor` |
| Cruise acceleration target | longitudinal planner |
| Final rolling landing | `SmoothStopController` |
| Standstill hold | stock `LongCtrlState.stopping` |
| Acceleration tracking | stock longitudinal PID |
| Vehicle command limits | opendbc |
| Independent safety ceiling | panda |

No downstream layer compensates by silently taking ownership of another
layer's decision. In particular:

- the supervisor never commands acceleration;
- Smooth Stops never weakens a stronger planner brake request;
- BLoTv2 does not raise opendbc or panda limits;
- model-independent BLoTv2 policy does not override the experimental/e2e
  acceleration candidate.

## Data flow

```text
model leadsV3 ----------+
                        |
radarState -- live/finite lead observation --> necessity supervisor
                        |                         |
                        +--> anchored lead path   +--> jerk-cost scale
                        |                         +--> dynamic t_follow
                        v
                 stock Acados MPC
                        |
             planner candidate arbitration
                        |
                 longitudinalPlan
                        |
          Smooth Stops final-approach policy
                        |
                stock longitudinal PID/hold
                        |
                  opendbc --> panda
```

The planner runs at model rate. Smooth Stops runs at control rate. The two
processes do not share mutable state; they import the same side-effect-free
relative lead physics.

## Shared lead contract

`LeadObservation.from_radar` accepts a lead only when:

- `radarState` passes `SubMaster.all_checks`, including alive, frequency, and
  valid;
- the lead is present;
- distance, filtered speed, filtered acceleration, and probability are finite;
- distance is positive.

Both longitudinal policy layers use relative closing motion:

```text
closing = max(v_ego - v_lead, 0)
closing_decel = closing² / (2 * max(d_rel - stop_margin, minimum_budget))
```

The MPC supervisor adds measured lead braking to this term. Smooth Stops uses
only closing deceleration because the planner remains the collision-avoidance
owner.

## Smooth

### Entry-anchored landing

When the plan wants a stop but ego is still rolling, longcontrol stays in the
PID state and enters a settle policy. It latches the current command and entry
speed, then releases pressure continuously toward a `0.12 m/s²` stop kiss as
speed reaches zero.

The stock hold clamp is entered only below `0.05 m/s`, or below `0.15 m/s` when
the car itself reports standstill. This avoids applying the full hold ramp to a
moving vehicle. It also works on an off-to-engaged rolling stop because current
stock openpilot no longer has a separate fixed starting command on that edge.

### Planner authority is preserved

Settle pressure is jerk-limited for comfort, but the returned command is:

```text
min(settle_command, planner_a_target)
```

A stronger MPC or e2e brake request therefore passes through immediately.

### Rolling-queue behavior

The previous Smooth Stops implementation treated a constant-speed creeping
queue as vehicle creep. A hard `0.3 m/s` threshold stopped new accumulation but
never released already accumulated anti-creep pressure. Noise or one radar
dropout could therefore leave permanent extra braking.

BLoTv2 uses:

- moving-lead entry at `0.30 m/s`;
- moving-lead exit at `0.18 m/s`;
- `0.50 s` radar-dropout grace;
- controlled anti-creep pressure decay at `1.0 m/s³`;
- a progress baseline re-anchored at the current queue speed.

The lead-distance floor also uses relative closing speed, so an equal-speed lead
at a close but stable gap no longer produces an absolute-ego-speed brake floor.

### Hold release

Once stopped, ten consecutive `shouldStop == false` control frames are required
to leave hold. Any true frame resets the count. This removes one-frame brake
blips while adding only `0.1 s` of confirmed launch delay.

## Swift

### Model-predicted MPC lead path

For a present radar lead and a confident, finite, correctly shaped model lead,
the MPC receives the model's full future lead trajectory anchored to radar at
the current frame. This lets visual lead braking and launch intent reach the
solver before a measured acceleration estimate catches up.

Malformed, non-finite, low-confidence, or unanchored model paths use the exact
stock radar-physics extrapolation. FCW remains gated by the present
vision-confirmed radar lead, not merely by model-path availability.

### Necessity supervisor

At 20 Hz the supervisor computes:

```text
required_decel =
    max(-a_lead, 0)
  + max(v_ego - v_lead, 0)²
    / (2 * max(d_rel - 4.0, 1.0))
```

It changes only solver inputs that already update at runtime:

- **recovery response:** sustained MPC braking beyond measured necessity
  relaxes acceleration-change and jerk cost;
- **model early response:** a sustained hard model lead forecast relaxes the
  same cost before radar acceleration catches up;
- **launch response:** a measured accelerating, receding lead and lagging MPC
  target relax the same cost;
- **dynamic onset:** mild measured lead braking or closing on a stopped lead
  adds up to `0.45 s` of following time so the obstacle cost opens earlier;
- **whiplash ratchet:** cost does not stiffen during a lead-braking/ego-closing
  reversal;
- **emergency stand-down:** low TTC plus high required deceleration removes all
  adaptive targets and returns toward stock solver policy.

All adaptive outputs slew. Trigger debouncing uses the planner's actual `dt`
instead of a hard-coded global frame period.

The supervisor compares necessity against the previous **MPC target**, not the
planner's final selected e2e/cruise output. This prevents another planner source
from accidentally arming MPC recovery state.

### Standstill lead pre-release

A valid lead that is already moving releases the MPC stop bit immediately. A
confident radar-anchored model prediction must persist for `0.2 s` before doing
the same. The MPC acceleration target remains unchanged, and an e2e stop
candidate is never overridden. Invalid radar, disengagement, or ego motion fails
closed.

## Strong

The stock cruise comfort schedule leaves some existing platform authority
unused at low speed. BLoTv2 changes the schedule from:

```text
[1.6, 1.2, 0.8, 0.6] m/s²
```

to:

```text
[2.0, 1.6, 1.0, 0.6] m/s²
at [0, 10, 25, 40] m/s
```

The low-speed jerk schedule follows the same taper. Straight-line total
acceleration allows `2.0 m/s²` at low speed.

This reaches, but never exceeds, stock `ACCEL_MAX = 2.0 m/s²`. A local
`BLOTV2_ACCEL_MAX = min(ACCEL_MAX, 2.0)` compatibility guard applies the same
ceiling to cruise, experimental/e2e, and MPC bounds when BLoTv2 is integrated
into a fork that still exposes BLoT v1's raised limit. Unlike BLoT v1, BLoTv2
changes no opendbc submodule, panda submodule, platform safety constant, or CAN
jerk limit. Strong braking remains available because Smooth Stops passes
stronger plan braking through.

Aggressive personality retains BLoT's `1.0 s` base follow setting; standard and
relaxed remain `1.45 s` and `1.75 s`. Dynamic onset can only add headway.

## Low-speed radar override qualification

The stock low-speed radar override can select an unconfirmed close track. A
field-observed short clutter/cross-traffic track previously caused maximum
braking near a stop. BLoTv2 requires an unconfirmed track to persist for 20
radar frames (about one second) before that override may select it.

Vision-confirmed lead selection is unchanged and bypasses this wait. This is a
global radar behavior change, so stopped-obstacle and cut-in coverage is a
mandatory field gate.

## Verification

Automated coverage includes:

- finite/live lead construction and relative-motion physics;
- settle entry continuity, jerk bounds, and stronger-plan pass-through;
- anti-creep progress, noisy rolling-lead thresholds, radar dropout, and
  stopped-to-moving queue transitions;
- true-stop handoff and hold-release debounce;
- every supervisor trigger, hysteresis, emergency stand-down, and slew rate;
- model lead anchoring and exact stock fallback for malformed/non-finite input;
- cruise authority bounded by stock `ACCEL_MAX`;
- low-speed radar track qualification;
- the full stock longitudinal maneuver matrix in ACC and experimental modes.

See `BLoTv2_ACCEPTANCE.md` for remaining replay, device, and field gates.

## Non-goals

- replacing the Acados optimization problem;
- overriding e2e acceleration or stop intent;
- raising vehicle or panda safety limits;
- changing the lateral controller;
- merging into `combo` as part of branch construction.
