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
- [x] `combo` is not modified or merged.
- [x] Shared live/finite lead contract.
- [x] Relative lead-physics unit coverage.
- [x] Smooth Stops rolling-queue/dropout transition coverage.
- [x] Stronger planner braking pass-through coverage.
- [x] MPC supervisor trigger, emergency, and slew coverage.
- [x] Model lead trajectory shape/finite/fallback coverage.
- [x] Low-speed radar override qualification coverage.
- [x] Stock platform acceleration ceiling retained.
- [x] Stock longitudinal maneuver matrix passes.
- [x] Final static and generated-solver build audit.

## Replay gate

Replay must use the real production classes, not copied equations.

- [ ] Baseline and BLoTv2 replay the same route timelines.
- [ ] Smooth-stop routes cover no lead, stopped lead, and creeping queue.
- [ ] Rolling-lead noise and radar-dropout motivating windows are reviewed.
- [ ] Mild lead-braking onset improves without early phantom braking.
- [ ] Hard lead braking has no collision-distance or peak-decel regression.
- [ ] Launch response improves without lunge-and-catch behavior.
- [ ] Model lead false-positive launch/braking windows are reviewed.
- [ ] Low-speed radar ghost route is rejected without missing a real obstacle.
- [ ] ACC and experimental-mode candidate arbitration is reviewed.

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

No aggregate score may hide a collision, missed stop, late hard-braking wall,
launch lunge, or close-gap creep.

## Device gate

- [ ] Clean build completes on the target comma device.
- [ ] controlsd and longitudinal planner stay within timing budgets.
- [ ] radarState liveness loss fails closed as documented.
- [ ] No process crash, solver reset burst, or invalid longitudinal plan.
- [ ] Opendbc and panda commits exactly match stock branch pointers.
- [ ] Acceleration commands stay inside vehicle limits.

## Field sequence

Run in a controlled environment with immediate driver takeover available:

1. Engage while already approaching a no-lead stop.
2. Repeat no-lead stops from gentle and moderate approaches.
3. Approach a stopped lead at low and urban speed.
4. Follow a queue at roughly `0.2–0.8 m/s`, including start/stop transitions.
5. Let a stopped lead depart slowly, then decisively.
6. Follow mild, moderate, and hard lead braking.
7. Test a constant-speed and accelerating lead pull-away.
8. Exercise low-speed radar-only obstacle/cut-in cases.
9. Repeat relevant cases in ACC and experimental modes.
10. Compare standard and aggressive personalities.

Stop the test immediately for:

- braking weaker than the selected planner target;
- acceleration toward an untrusted or stopped lead;
- oscillatory brake/throttle commands;
- a hold release without confirmed intent;
- repeated solver resets or invalid radar/planner state;
- any unexpected vehicle safety alert.

## Promotion

After the gates above:

- owner field feedback is recorded;
- unresolved regressions remain explicit;
- exact commits and submodule pointers are pinned;
- only then may the owner request a merge into `combo`;
- only the owner may authorize changing README status from **in progress**.
