# BLoTv2 acceptance gates

The branch remains **in progress** until the owner field-tests it and explicitly
authorizes a status change. Passing offline tests does not make it complete or
field proven.

## Artifact identity

Every test report must identify:

- openpilot commit;
- opendbc commit;
- panda commit;
- route or synthetic scenario set;
- active personality and experimental mode;
- exact BLoTv2 constants;
- whether measurements are planner targets, control commands, or delivered
  vehicle acceleration.

## Foundation gate

- [x] Branch created from current untouched `stock`.
- [x] BLoTv2-owned changes are authored on `BLoTv2` before `combo` integration;
  Smooth Stops changes are authored on `smooth-stops`.
- [x] Shared live/finite lead contract.
- [x] Relative lead-physics unit coverage.
- [x] Integrated `smooth-stops` rolling-queue/dropout transition coverage.
- [x] Integrated `smooth-stops` stronger-planner-braking pass-through coverage.
- [x] MPC supervisor trigger, emergency, and slew coverage.
- [x] Model lead trajectory shape/finite/fallback coverage.
- [x] Model lead service validity, radar/model identity, probability boundaries,
  bounded uncertainty, and pre/post-anchor position-speed consistency fail
  closed to radar physics.
- [x] Invalid model service omits model forecasts and the e2e candidate while
  retaining healthy radar/cruise fallback.
- [x] Low-speed radar override qualification coverage.
- [x] `4.0 m/s²` request is clamped by the deployed opendbc envelope.
- [x] Existing BLoTv2 jerk and MPC tune retained for isolated evaluation.
- [x] The acceleration ceiling follows the owner-approved cubic envelope with
  deterministic endpoint, sample, monotonicity, convexity, continuity, and
  deployed-platform-clamp coverage.
- [x] Ordinary Chill cruise uses a route-derived proportional response with
  deterministic coverage for coast-down, low-speed blending, large-error
  authority, jerk, invalid radar, lead-independent shaping with MPC safety
  ownership, and strategy bypasses.
- [x] Conditional Experimental Mode uses only existing BLoTv2 model/cereal
  signals and publishes through `selfdriveState.experimentalMode`.
- [x] Conditional mode filtering, debounce, hysteresis, latch, lead/turn
  guards, pedal override, invalid-model release, reset, and publication have
  deterministic unit coverage.
- [x] Early urban intent, faster filter-only hints, intent-hold behavior,
  geometry validity, highway-slowdown rejection, and recent-lead hysteresis
  have deterministic unit coverage.
- [x] Qualified Force Stops authority is bound to the exact current model frame,
  fails closed on malformed trajectories, health faults, leads, and pedals, and
  retains combo's stable-braking latch, reversible cap, committed-point, MPC,
  and Smooth Stops ownership boundaries.
- [x] Force Stops widened-latch confidence matches the canonical composed
  implementation, and committed stop-obstacle ownership has a distinct source.
- [x] Lead0 adaptive MPC policy cannot shape lead1, lead2, or committed-stop
  solves; competing sources use incumbent weights and following time.
- [x] Signed committed-stop geometry survives the MPC lower boundary, is clamped
  at the solver handoff, and conditional stop intent preserves standstill hold.
- [x] Recent-lead recovery preserves the `3.0 s` timer, starts only from one
  strict 33-point frame, validates every positive-probability sample from all
  three complete model-lead hypotheses, and revokes on malformed data, health
  loss, turns, or either raw lead.
- [x] Stock longitudinal maneuver matrix passes.
- [x] Final static and generated-solver build audit.

## Replay gate

Replay must use the real production classes, not copied equations.

- [ ] Baseline and BLoTv2 replay the same route timelines.
- [ ] Integrated Smooth Stops routes cover no lead, stopped lead, and creeping queue.
- [ ] Integrated Smooth Stops rolling-lead noise and radar-dropout windows are reviewed.
- [ ] Mild lead-braking onset improves without early phantom braking.
- [ ] Hard lead braking has no collision-distance or peak-decel regression.
- [ ] `4.0 m/s²` launch response improves without lunge-and-catch behavior.
- [ ] Model lead false-positive launch/braking windows are reviewed.
- [ ] Low-speed radar ghost route is rejected without missing a real obstacle.
- [ ] ACC and experimental-mode candidate arbitration is reviewed.
- [x] Route `000000d2--a62f0c1831` identifies cruise authority—not PID or
  Smooth Stops—as the source of excessive 40-to-45 mph acceleration.
- [x] Route `000000d9--6040563d1d` identifies unity-gain cruise speed error as
  the source of both excessive 75-to-80 mph acceleration and full-limit
  80-to-75 mph braking.
- [x] The production cruise function replays the three comparable route
  timelines at `+0.400`, `-0.401`, and `+0.353 m/s²` instead of `+0.688`,
  `-1.200`, and `+0.685 m/s²`; this is command replay, not closed-loop proof.
- [x] Replacing the four-node ceiling with the cubic envelope retains those
  three comfort-shaped route outputs below the available acceleration ceiling.
- [x] Route `000000d7--cc6308b4d0` identifies late CEM recognition—not planner
  handoff latency—as the first high-speed red-light failure and replays the
  production class at `42.2 mph` instead of `39.1 mph`.
- [x] All 62 segments of that route reject the observed highway-exit slowdown
  and transient lead-dropout window while retaining both intended stop
  handoffs.
- [x] Production-class route-29 replay adds only the two intended strict-first
  entries, preserves nine Force Stops qualification transitions, and rejects
  the replacement-vehicle sentinel near Connect `1440.989`.
- [x] Production-class route-17 and route-27 replay add no early-release entries
  and preserve their `8/34` and `6/11` entry/qualification counts.
- [x] The route-29 landing experiment is rejected: it leaves the first handoff
  inside actuator delay and weakens an existing finite-endpoint fail-closed
  release. No landing, cap, target, MPC, or hold change ships here.
- [x] The conservative Connect `364–380` endpoint-placement case is documented
  without a global setback or endpoint adjustment.
- [x] Offline native LongControl removes the simplified-harness segment-25
  no-standstill artifact; this is diagnostic evidence, not field validation.
- [ ] Route-17 segment 20 remains `1.324 m` behind its internal retained target;
  physical stop-line placement remains an owner field gate because that target
  is not calibrated painted-line ground truth.
- [ ] The route-derived cubic envelope is repeated on-road without launch
  lunge, a 10 m/s authority cliff, delayed response, or added speed overshoot.
- [ ] Five-mph set-speed increases and reductions at urban and highway speed
  feel soft, do not overshoot, and do not create brake/throttle rebound.
- [ ] Ordinary no-stop driving remains in Chill without mode flicker.
- [ ] Lead-free red-light and stop-sign approaches enter Experimental before
  braking is needed and remain there through the stop.
- [ ] Green/resume and driver-pedal releases return to Chill without a
  one-frame relatch or delayed launch.
- [ ] Leads, curves, and signaled turns produce no false conditional handoff.

Report at least:

- response onset time and headway;
- peak acceleration/deceleration and its event position;
- jerk RMS and worst short window;
- release time after necessity falls;
- final landing acceleration;
- stop gap;
- supervisor trigger duty;
- raw MPC target, selected planner target, control command, and delivered
  acceleration separately.
- positive-command saturation duty and speed overshoot after launch.

No aggregate score may hide a collision, missed stop, late hard-braking wall,
launch lunge, or close-gap creep.

## Device gate

- [ ] Clean build completes on the target comma device.
- [ ] controlsd and longitudinal planner stay within timing budgets.
- [ ] radarState liveness loss fails closed as documented.
- [ ] No process crash, solver reset burst, or invalid longitudinal plan.
- [ ] Exact deployed openpilot, opendbc, and panda commits are recorded.
- [ ] Controller and panda limits support the requested authority end to end.
- [ ] Acceleration commands stay inside vehicle limits.
- [ ] The stock on-road icon matches `selfdriveState.experimentalMode` through
  entry, standstill, and release transitions.

## Field sequence

Run in a controlled environment with immediate driver takeover available:

1. Engage while already approaching a no-lead stop.
2. Repeat no-lead stops from gentle and moderate approaches.
3. Approach a stopped lead at low and urban speed.
4. Follow a queue at roughly `0.2–0.8 m/s`, including start/stop transitions.
5. Let a stopped lead depart slowly, then decisively.
6. Follow mild, moderate, and hard lead braking.
7. Run straight launches with progressively larger speed deltas, beginning
   below the `4.0 m/s²` ceiling; sample the curve near 11, 22, 34, 45, and
   56 mph, then repeat five-mph corrections near 45 and 80 mph in both
   directions.
8. Test a constant-speed and accelerating lead pull-away.
9. Exercise low-speed radar-only obstacle/cut-in cases.
10. Repeat relevant cases in ACC and experimental modes, then compare standard
    and aggressive personalities.

Stop the test immediately for:

- braking weaker than the selected planner target;
- acceleration toward an untrusted or stopped lead;
- unexpected launch overshoot or sustained positive-command saturation;
- oscillatory brake/throttle commands;
- a hold release without confirmed intent;
- repeated solver resets or invalid radar/planner state;
- any unexpected vehicle safety alert.

## Promotion

After the gates above:

- owner field feedback is recorded;
- unresolved regressions remain explicit;
- exact commits and submodule pointers are pinned;
- test integration into `combo` does not by itself satisfy these gates;
- only the owner may authorize changing README status from **in progress**.
