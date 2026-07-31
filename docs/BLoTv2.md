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
unused at launch. BLoTv2 keeps BLoT v1's launch authority while handing back
to the exact stock speed gate by `10 m/s`, changing the schedule from:

```text
[1.6, 1.2, 0.8, 0.6] m/s²
at [0, 10, 25, 40] m/s
```

to:

```text
[4.0, 1.2, 0.8, 0.6] m/s²
at [0, 10, 25, 40] m/s
```

Authority and jerk remain separate tuning axes. The existing BLoTv2 jerk
schedule stays `[2.0, 1.6, 1.0, 0.6] m/s³`; it is not doubled with the
acceleration request and retains its original `[0, 10, 25, 40] m/s`
breakpoints. Only the acceleration schedule's launch endpoint differs from
stock: interpolation fades the added authority continuously from `4.0 m/s²`
at standstill to stock's `1.2 m/s²` at `10 m/s` (about `22 mph`). Every
acceleration ceiling at and above that handoff is exactly stock. The
straight-line total-acceleration budget is `4.0 m/s²`, while lateral
acceleration still consumes that shared budget in a turn.

### Route `000000d2--a62f0c1831` urban-speed refinement

The full-resolution route was recorded on `combo` commit `c43e130059`. Three
40-to-45 mph corrections reached approximately `1.31`, `1.64`, and
`1.79 m/s²`. In the strongest event, cruise selected `1.79 m/s²` at `39.4 mph`
for only a `5.4 mph` set-speed error; the planner target and final command
matched, so this was not PID overshoot or Smooth Stops behavior.

The revised schedule limits the same 40 mph operating point to about
`0.99 m/s²`, exactly matching stock there. It preserves the `4.0 m/s²` launch
endpoint but smoothly removes the extra authority by `10 m/s`. The separate
jerk schedule is unchanged, so a valid change still begins promptly; “Swift”
describes reaction time and low-speed launch response, not sustained high
throttle for a small urban-speed correction. This counterfactual cap is
route-derived, not a claim of field validation, and remains an explicit owner
test item.

`BLOTV2_ACCEL_REQUEST_MAX = 4.0 m/s²` records the requested policy.
`BLOTV2_ACCEL_MAX = min(ACCEL_MAX, BLOTV2_ACCEL_REQUEST_MAX)` applies the
deployed platform envelope to cruise, experimental/e2e, MPC bounds, and the
final longitudinal PID command. The stock-based feature branch therefore
remains limited to stock authority when run with its stock opendbc pointer.
`combo` already carries BLoT v1's opendbc and panda `4.0 m/s²` safety lineage,
so the same code can use the full request there without changing either
submodule in BLoTv2.

The necessity supervisor and MPC costs are unchanged. Their inputs are
relative lead physics, required deceleration, and the previous MPC solution,
not a fixed fraction of the positive acceleration ceiling, so no speculative
cost retune is needed merely to expose more positive authority. The resulting
launch feel, delivered acceleration, saturation duty, and speed overshoot still
require route review and owner field validation. Strong braking remains
available because Smooth Stops passes stronger plan braking through.

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
- BLoT v1 launch request and deployed-platform acceleration clamping;
- low-speed radar track qualification;
- the full stock longitudinal maneuver matrix in ACC and experimental modes.

See `BLoTv2_ACCEPTANCE.md` for remaining replay, device, and field gates.

## Non-goals

- replacing the Acados optimization problem;
- overriding e2e acceleration or stop intent;
- raising vehicle or panda safety limits;
- changing the lateral controller;
- declaring the `4.0 m/s²` tune field-proven before owner testing.
