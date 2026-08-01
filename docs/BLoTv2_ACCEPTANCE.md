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
- [x] Feature changes are authored on `BLoTv2` before `combo` integration.
- [x] Shared live/finite lead contract.
- [x] Relative lead-physics unit coverage.
- [x] Smooth Stops rolling-queue/dropout transition coverage.
- [x] Stronger planner braking pass-through coverage.
- [x] MPC supervisor trigger, emergency, and slew coverage.
- [x] Model lead trajectory shape/finite/fallback coverage.
- [x] Low-speed radar override qualification coverage.
- [x] `4.0 m/s²` request is clamped by the deployed opendbc envelope.
- [x] Existing BLoTv2 jerk and MPC tune retained for isolated evaluation.
- [x] Conditional Experimental Mode uses only existing BLoTv2 model/cereal
  signals and publishes through `selfdriveState.experimentalMode`.
- [x] Conditional mode filtering, debounce, hysteresis, latch, lead/turn
  guards, pedal override, invalid-model release, reset, and publication have
  deterministic unit coverage.
- [x] High-speed early intent, filter-only hints, comfort-distance gating,
  geometry validity, highway-slowdown rejection, and recent-lead hysteresis
  have deterministic unit coverage.
- [x] BLoTv2 contains no legacy Force Stops speed-cap owner or planner hook.
- [x] Stock longitudinal maneuver matrix passes.
- [x] Final static and generated-solver build audit.

## Replay gate

Replay must use the real production classes, not copied equations.

- [ ] Baseline and BLoTv2 replay the same route timelines.
- [ ] Smooth-stop routes cover no lead, stopped lead, and creeping queue.
- [ ] Rolling-lead noise and radar-dropout motivating windows are reviewed.
- [ ] Mild lead-braking onset improves without early phantom braking.
- [ ] Hard lead braking has no collision-distance or peak-decel regression.
- [ ] `4.0 m/s²` launch response improves without lunge-and-catch behavior.
- [ ] Model lead false-positive launch/braking windows are reviewed.
- [ ] Low-speed radar ghost route is rejected without missing a real obstacle.
- [ ] ACC and experimental-mode candidate arbitration is reviewed.
- [x] Route `000000d2--a62f0c1831` identifies cruise authority—not PID or
  Smooth Stops—as the source of excessive 40-to-45 mph acceleration.
- [x] Route `000000d7--cc6308b4d0` identifies late CEM recognition—not planner
  handoff latency—as the first high-speed red-light failure and replays the
  production class at `42.2 mph` instead of `39.1 mph`.
- [x] All 62 segments of that route reject the observed highway-exit slowdown
  and transient lead-dropout window while retaining both intended stop
  handoffs.
- [ ] The route-derived stock-gate handoff is repeated on-road without
  excessive throttle feel, delayed response, or added speed overshoot.
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
   below the `4.0 m/s²` ceiling.
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
