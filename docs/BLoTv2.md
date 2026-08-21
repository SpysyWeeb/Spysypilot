# BLoTv2 — modular longitudinal control

## Status

**In progress. Not field validated. Do not mark complete before owner field
testing and explicit approval.**

BLoTv2 starts from the untouched `stock` branch and reimplements useful BLoT
planner/MPC mechanisms. It does not implement stop landing; `combo` integrates
that separately from the canonical `smooth-stops` branch.

The product target is **Smooth. Swift. Strong.**

- **Smooth:** continuous acceleration and jerk through cruise and lead following.
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
| Committed model stop point and approach cap | separate `force-stops` feature |
| Final rolling landing | separate `smooth-stops` feature |
| Standstill hold | stock `LongCtrlState.stopping` |
| Acceleration tracking | stock longitudinal PID |
| Effective Chill/Experimental mode | `selfdrived` |
| Vehicle command limits | opendbc |
| Independent safety ceiling | panda |

No downstream layer compensates by silently taking ownership of another
layer's decision. In particular:

- the supervisor never commands acceleration;
- Smooth Stops never weakens a stronger planner brake request;
- Conditional Experimental Mode never commands a speed, acceleration, brake,
  or stop point;
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
model stop + effective Experimental --> Force Stops cruise cap
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

The planner and optional Force Stops tracker run at model rate. Smooth Stops
runs at control rate. CEM selects the planning mode, Force Stops may cap the
approach, and Smooth Stops owns only the final landing.

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

## Integrated Smooth Stops

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

Smooth Stops uses:

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

The model service and radar/model association must both be healthy. Position
derivative and predicted speed must agree within the existing `5 m/s` model-lead
bound before and after radar anchoring; associated-lead position/speed uncertainty
must remain below `50 m` / `10 m/s`. Malformed, non-finite, low-confidence,
unconfirmed low-speed, physically inconsistent, or unanchored model paths use
the exact stock radar-physics extrapolation. FCW remains gated by the present
vision-confirmed radar lead, not merely by model-path availability. Invalid
runtime jerk scaling similarly restores the incumbent MPC weights. An invalid
model service contributes no model lead, model forecast, departure release, or
e2e candidate during the existing soft-disable interval.

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
  same cost and contributes to the bounded dynamic-onset pad before radar
  acceleration catches up;
- **launch response:** a measured accelerating, receding lead and lagging MPC
  target relax the same cost;
- **dynamic onset:** mild measured or confirmed predicted lead braking, or
  closing on a stopped lead, adds up to `0.45 s` of following time so the
  obstacle cost opens earlier;
- **whiplash ratchet:** cost does not stiffen during a lead-braking/ego-closing
  reversal;
- **emergency stand-down:** low TTC, high required deceleration, and a meaningful
  shortfall versus the existing MPC brake target remove adaptive targets and
  return toward stock solver policy.

All adaptive outputs slew. Trigger debouncing uses the planner's actual `dt`
instead of a hard-coded global frame period.

The supervisor compares necessity against the previous **MPC target**, not the
planner's final selected e2e/cruise output. This prevents another planner source
from accidentally arming MPC recovery state.

Its runtime jerk/headway policy is installed only when lead0 is the current MPC
obstacle source. Lead1, confirmed lead2, and the committed stop solve with the
incumbent personality weights and base following time, so one lead cannot shape
another owner's trajectory. If an actually adaptive lead0 solve hands off to a
competing source, the MPC change-cost reference reanchors to the current command;
nominal lead0 history is left intact.

### Standstill lead pre-release

A valid lead that is already moving releases the MPC stop bit immediately. A
confident radar-anchored model prediction must persist for `0.2 s` before doing
the same. The MPC acceleration target remains unchanged, and an e2e stop
candidate is never overridden. Invalid radar, disengagement, or ego motion fails
closed.

## Conditional Experimental Mode

**Status: in progress; not field validated.** Ordinary driving requests Chill.
A temporally confirmed, lead-free model stop prediction requests Experimental
early enough for the existing e2e candidate to participate. `selfdrived`
remains the only effective-mode owner and publishes the result through the
unchanged `selfdriveState.experimentalMode` field:

```text
modelV2 + radarState + carState
              |
   filtered stop-intent request
              |
          selfdrived
              |
 selfdriveState.experimentalMode
        |                 |
 longitudinal planner    stock on-road icon
```

The existing stock `ExperimentalMode` Param remains only a manual request.
The Params thread no longer writes the effective mode asynchronously;
`selfdrived` resolves manual and conditional requests in one place, with a
driver pedal override taking priority.
The conditional latch is also published separately from the effective mode.
Below the stock `0.3 m/s` stop boundary it preserves planner `shouldStop` through
brief model-clear flicker; manual Experimental mode does not mint that hold.

### BLoTv2 signal mapping

This model/cereal revision has no traffic-light class, stop-sign class,
dashboard stop signal, or calibrated stop-object probability. The detector
therefore describes a **generic model stop intent**, not a semantic claim about
what caused it.

| Stop evidence | BLoTv2 signal and use |
|---|---|
| Direct final-approach intent | `modelV2.action.shouldStop` |
| Strict prediction | `position.x[-1]` is inside a `5.0 s` ego-speed horizon and `velocity.x[-1] <= 1.0 m/s` |
| High-speed early prediction | above `13.0 m/s`, the path is inside a `1.3 m/s²` comfort-stop envelope plus `0.5 s` response distance, terminal speed is at most `6.5 m/s` and `35%` of ego speed, and `action.desiredAcceleration <= -0.5 m/s²` |
| Early urban prediction | inside an `8.0 s` horizon, terminal speed at most `55%` of ego speed, and desired acceleration at most `-0.25 m/s²`; at or below `22.0 m/s` it receives qualifying `0.70` confidence, while faster approaches remain a non-triggering `0.45` filter hint |
| Early geometry guard | no turn signal, valid `action.desiredCurvature` below `1.0 m/s²` lateral acceleration, and valid terminal `orientation.z` within `20 degrees` |
| Missing-velocity fallback | short path plus `action.desiredAcceleration <= -0.5 m/s²` |
| Lead ownership | relevant radar lead blocks a new handoff and starts an unchanged `3.0 s` grace. During that grace, one current strict 33-point stop may mint a revocable release only with both raw leads absent and every positive-probability model-lead hypothesis outside the predicted corridor; complete strong evidence may retain it through strict flicker |
| Committed turn guard | low-speed blinker plus large steering angle/model curvature blocks a new handoff |
| Release/override | stable model clear, resumed motion, gas, brake, invalid model, or disengagement |

The recent-lead release validates exactly 33 finite path `x/y` samples with
nondecreasing `x`, exactly three model-lead hypotheses, and exactly six finite
`x/y` samples per hypothesis. Probability must be finite in `[0, 1]`; every
value above zero counts. A hypothesis blocks the release at or inside both the
stop endpoint plus `10.0 m` and the interpolated path center plus/minus `1.5 m`.
The lateral path clamps at its authored endpoint over that extra margin rather
than extrapolating unknown curvature. Missing, malformed, stale, unhealthy, or
same-tick raw-lead data preserves the original grace and revokes any pending
release.

The evidence filter uses a `0.30 s` time constant, separate entry/release
thresholds, `0.20 s` entry debounce, and `0.75 s` release hysteresis. Every
qualifying stop sample refreshes a `4.0 s` intent hold so brief model flicker
cannot move the stop approach back to Chill. A mode handoff also has a `1.0 s`
minimum latch. Once standstill is reached, a separate `1.0 s` minimum prevents
an immediate clear; the refreshed `4.0 s` intent hold still applies. Valid stop
evidence keeps Experimental active indefinitely. A resume above `0.8 m/s`
releases immediately. Gas or brake suppresses a new handoff for `2.0 s`, and a
completed stop gets the same `2.0 s` re-entry guard.
All values live together in `conditional_experimental_mode.py`; no tuning Param
or UI control is added.

### Route `000000d7--cc6308b4d0` high-speed refinement

The first field attempt exposed a recognition-delay problem rather than a
planner handoff delay. At the lead-free red-light approach, the deployed
strict detector first produced qualifying evidence at `39.7 mph` and switched
at `39.1 mph`. The planner selected e2e within one model frame, but the driver
braked about `2.1 s` later because the remaining stop felt too late.

The recorded model had already developed coherent stop intent at `42.4 mph`:
`position.x[-1] = 138.0 m`, `velocity.x[-1] = 5.85 m/s`,
`action.desiredAcceleration = -0.50 m/s²`, and an essentially straight
predicted heading. The first early tier switches at `42.2 mph`, `3.8 s` and
roughly `69 m` before the deployed handoff. The earlier `44.7 mph` sample is
now eligible to request Experimental after the normal filter and debounce;
exact full-route timing still requires a new replay receipt.

A scan of all 62 route segments also exposed two ambiguous windows. A
`55 mph` highway-exit approach predicted slowing to about `8 m/s`, not a stop;
the `6.5 m/s` strict-tier terminal cap rejects it and the `22.0 m/s` urban-tier
ceiling keeps it filter-only. A separate intersection
window lost a relevant radar lead briefly; the `3.0 s` lead-release guard
keeps that slowdown lead-owned. Neither guard changes the first red-light
handoff or the later low-speed stop/green-release result.

This replay reuses recorded model, radar, and vehicle messages. It proves the
mode decision timing and rejection behavior, but it cannot close the loop to
show the trajectory the newly selected e2e planner would have driven. The
high-speed stop and false-handoff behavior therefore remain explicit owner
field-test gates.

### Route `00000029--74b84f1be9` lead-departure refinement

Exact production-class replay preserves the `3.0 s` timer and adds only two
strict-first, path-clear releases. CEM enters at route `291.790260` / Connect
`297.165686`, `0.526477 s` before recorded driver brake, and at route
`781.954128` / Connect `787.329554`, `1.805828 s` before brake. Neither entry
reaches the separate one-second Force Stops qualification before recorded
intervention.

The negative sentinel at route `1435.612171` / Connect `1440.987597` contains
three replacement hypotheses at roughly `36.82–36.93 m`, only
`0.270–0.378 m` from the predicted path center despite probabilities of
`0.20–0.38`. Every positive-probability hypothesis counts, so the release stays
revoked and no CEM entry occurs. Route-wide replay adds no entry on route 17 or
route 27. The conservative endpoint-placement case near Connect `364–380` is
unchanged: no target setback or global endpoint adjustment is introduced.

The same route proves both perceived twitch windows contain one late ownership
handoff, not repeated planner/LongControl oscillation. A combined committed-
lifetime and delay-projection experiment moved replay `shouldStop` to `0.450 s`
and `0.550 s` before standstill, but the first still landed inside the logged
`0.500 s` actuator delay and the experiment removed an existing finite-endpoint
fail-closed release. It is therefore rejected from this change. Force Stops,
planner acceleration, MPC geometry, stock LongControl, and integrated Smooth
Stops remain unchanged.

### Qualified Force Stops handoff

After one full second of current, complete, finite, lead-free stop evidence,
CEM publishes a capability bound to the exact current `modelV2` monotonic
timestamp and a fixed-world endpoint. Force Stops owns reversible cap shaping
and the committed point; longitudinal MPC admits that point only while its
native obstacle remains ahead and closer leads still win. Smooth Stops owns
combo's final landing and standstill handoff. CEM still owns no speed,
acceleration, brake, or stop point.

The standalone branch now carries the canonical Force Stops braking-confidence
gate already composed in `combo`: the widened `3.25 s` latch requires sustained
braking evidence, while the classic `3.0 s` latch retains its existing confidence.
When the committed position obstacle owns MPC, telemetry publishes the dedicated
`stop` source rather than the ordinary `cruise` label.

After commitment, signed remaining distance continues to represent the same
world point. The planner clamps only the MPC handoff at `-STOP_DISTANCE`; it does
not delete ownership at that geometry boundary. Lead, pedal, invalid-service,
mode, detector-clear, and standstill releases remain native Force Stops exits.

This handoff is in progress for testing only and its source candidate remains
replay-rejected. Route-17 segment 20 stopped `1.324 m` behind its internal
retained target; that target is not calibrated painted-line ground truth.
Offline native LongControl did not reproduce the earlier segment-25
no-standstill result, but neither result establishes road safety or stop-line
placement.

## Strong

The stock cruise comfort schedule leaves some existing platform authority
unused at launch. An earlier in-progress BLoTv2 revision changed its four
piecewise-linear nodes from:

```text
[1.6, 1.2, 0.8, 0.6] m/s²
at [0, 10, 25, 40] m/s
```

to:

```text
[4.0, 1.2, 0.8, 0.6] m/s²
at [0, 10, 25, 40] m/s
```

That revision corrected the route-derived urban lunge, but owner review found
the 0-to-10 m/s decline too abrupt. It is now superseded by one continuous
convex envelope:

```text
a_max(v) = 0.6 + 3.4 × (1 − v/40)³     for 0 ≤ v ≤ 40 m/s
a_max(v) = 0.6                          for v > 40 m/s
```

The requested curve is monotonic, has no internal speed-node corners, and
reaches the high-speed floor with zero slope. Representative requested values
before the deployed-platform clamp are:

| speed | requested maximum acceleration |
|---:|---:|
| 0 m/s (0 mph) | 4.000 m/s² |
| 5 m/s (11 mph) | 2.878 m/s² |
| 10 m/s (22 mph) | 2.034 m/s² |
| 15 m/s (34 mph) | 1.430 m/s² |
| 20 m/s (45 mph) | 1.025 m/s² |
| 25 m/s (56 mph) | 0.779 m/s² |
| 30 m/s (67 mph) | 0.653 m/s² |
| 35 m/s (78 mph) | 0.607 m/s² |
| 40 m/s (89 mph) | 0.600 m/s² |

Authority and jerk remain separate tuning axes. The existing BLoTv2 jerk
schedule stays `[2.0, 1.6, 1.0, 0.6] m/s³`; it retains its original
`[0, 10, 25, 40] m/s` breakpoints and is not replaced by the cubic acceleration
envelope. The straight-line total-acceleration budget remains `4.0 m/s²`,
while lateral acceleration still consumes that shared budget in a turn.

### Route `000000d2--a62f0c1831` urban-speed refinement

The full-resolution route was recorded on `combo` commit `c43e130059`. Three
40-to-45 mph corrections reached approximately `1.31`, `1.64`, and
`1.79 m/s²`. In the strongest event, cruise selected `1.79 m/s²` at `39.4 mph`
for only a `5.4 mph` set-speed error; the planner target and final command
matched, so this was not PID overshoot or Smooth Stops behavior.

The intermediate four-node revision limited the same 40 mph operating point
to about `0.99 m/s²`. The current cubic envelope permits about `1.17 m/s²`
there, but the later ordinary-cruise comfort response limits the route's
`5.4 mph` error to about `0.435 m/s²`, well below either ceiling. Smoothing the
envelope therefore does not restore the original `1.79 m/s²` Chill cruise
lunge. The separate jerk schedule is unchanged, so a valid change still begins
promptly; “Swift” describes reaction time and low-speed launch response, not
sustained high throttle for a small urban-speed correction. This remains
counterfactual evidence—not field validation—and an explicit owner test item.

### Route `000000d9--6040563d1d` ordinary-cruise comfort refinement

The route was recorded on `combo` commit `095714fd87` and contains three
comparable, engaged, lead-free Chill cruise corrections. Both 75-to-80 mph
increases selected about `+0.69 m/s²`; delivered acceleration peaked near
`+1.01` and `+0.80 m/s²`. The 80-to-75 mph reduction selected and commanded
`-1.20 m/s²`, delivered about `-1.54 m/s²`, crossed below the new set speed,
and then requested positive acceleration to recover. Its pitch-compensated
natural-coast estimate was only about `-0.36` to `-0.39 m/s²`.

The source in each comparable window was cruise. The planner target and
longitudinal command agreed, with no lead and Experimental mode off, so the
behavior did not originate in Smooth Stops, MPC lead following, Conditional
Experimental Mode, the curve-speed limiter, or an anomalous PID request. The
cause was the cruise candidate's unity-gain speed error: a 5 mph error is large
enough to saturate either the road-speed acceleration gate or the complete
`-1.2 m/s²` cruise deceleration limit. The jerk schedule delayed those limits
but did not reduce their sustained magnitude.

For ordinary Chill cruise, BLoTv2 now changes that candidate to:

```text
target acceleration = 0.18 s⁻¹ × set-speed error
```

At road speed, a 5 mph correction therefore requests about `0.40 m/s²` in
either direction instead of saturating. The target falls continuously as the
speed error closes. During a reduction, the proportional target blends toward
a full pitch-compensated throttle lift by a 5 mph error; this permits natural
uphill coast-down without adding gas, while a downhill still receives the
gentle proportional deceleration request. The response blends from the legacy
candidate at `8 m/s` to the comfort candidate at `15 m/s`, preserving the
existing low-speed launch behavior. Larger errors naturally reach the
configured cubic ceiling—roughly 8–13 mph upward depending on road speed—and
the existing `-1.2 m/s²` limit at about 15 mph downward, so this is not a
global authority reduction.

The comfort path is eligible only with healthy radar, no radar lead, Chill
mode, and no forced deceleration. In `combo`, an active model curve-speed limit
also disables it. Those paths retain the exact legacy cruise calculation, and
the planner's MPC/e2e candidate arbitration is unchanged. The existing jerk
schedule still governs onset and release, so “Swift” remains reaction time and
low-speed response while a routine five-mph correction becomes softer.

On the former four-node revision, a production-function trace replay—not a
copied equation—matched the recorded legacy peaks (`+0.688`, `-1.200`, and
`+0.685 m/s²`) and changed the same input timelines to `+0.400`, `-0.401`, and
`+0.353 m/s²`. Repeating that replay with the cubic envelope retains the same
three comfort-shaped values; they remain below the new ceiling. The third
value closes earlier because this command-only replay retains the vehicle
trajectory produced by the stronger recorded command. It is not a closed-loop
prediction of vehicle speed or delivered acceleration and is not field
validation.

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
- integrated Smooth Stops entry continuity, jerk bounds, and stronger-plan pass-through;
- integrated anti-creep progress, noisy rolling-lead thresholds, radar dropout, and
  stopped-to-moving queue transitions;
- true-stop handoff and hold-release debounce;
- Conditional Experimental entry filtering, release hysteresis, standstill
  latch, model-health failure, route-derived high-speed stop intent,
  highway-slowdown rejection, lead-dropout/curve/turn entry vetoes, pedal
  override, reset, and `selfdriveState.experimentalMode` publication;
- every supervisor trigger, hysteresis, emergency stand-down, and slew rate;
- model lead anchoring and exact stock fallback for malformed/non-finite input;
- BLoT v1 launch request and deployed-platform acceleration clamping;
- exact cubic acceleration samples, monotonicity, convexity, corner-free
  continuity, and smooth high-speed-floor coverage;
- route-derived ordinary-cruise proportional response, coast-down behavior,
  low-speed blend, large-error authority, jerk bounds, and strategy bypasses;
- low-speed radar track qualification;
- the full stock longitudinal maneuver matrix in ACC and experimental modes.

See `BLoTv2_ACCEPTANCE.md` for remaining replay, device, and field gates.

## Non-goals

- replacing the Acados optimization problem;
- overriding e2e acceleration or stop intent;
- owning final stop landing or standstill handoff;
- raising vehicle or panda safety limits;
- changing the lateral controller;
- classifying red lights or stop signs when the model publishes no such class;
- adding a second stop-point, target-speed, or brake-command owner;
- declaring the `4.0 m/s²` tune field-proven before owner testing.
