# lane-centering

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: in progress; awaiting owner field validation.** Watch for a slow weave with a period of 8–10 s on a straight highway (the sign of too much gain), for the car running wider than before through left curves, and for how much room it still gives a cyclist or a parked car.

## What it does

Keeps the car in the middle of the lane lines the model sees. The driving model is content to sit off-center: on the 2021 Hyundai Palisade it rides 0.14 m left of the midpoint between the lines on average, more than 0.47 m off for a tenth of the time, and it cuts curves by about 0.4 m to the inside. Left to itself it drifts back toward *its own* preferred spot, not the center, over about 7 s.

This branch adds a small steering correction toward the lane center on top of the model's. The model still drives: the correction is at most 0.2 m/s² of lateral acceleration (the model's own demand is 1.1 m/s² at the 90th percentile), it needs both lane lines, and it lets go when the driver or the model means to leave the lane.

## How it works

Every model frame, `get_lane_center_curvature` looks one second ahead (at least 10 m) and compares the midpoint of the two lane lines with where the model's plan puts the car. The road's curvature is in both, so only the offset between them is left. The correction is the curvature of the arc that would close 20 % of that offset by the lookahead, `0.2 · 2 · offset / lookahead²`, added to `modelV2.action.desiredCurvature` in `controlsd` before `clip_curvature`, so stock's lateral jerk and acceleration limits still apply to the sum. A 0.25 s first-order filter at 100 Hz smooths the 20 Hz steps.

It fades out instead of fighting:

- **No lane:** either line below 0.5 probability or above 0.3 m standard deviation, or a lane narrower than 2.5 m or wider than 4.5 m (merges, splits, misread lines) — no correction.
- **Leaving the lane on purpose:** steering pressed, either blinker, or a lane change in progress — the correction decays to zero over the filter's time constant.
- **The model means it:** full correction up to 0.5 m between the plan and the center, tapering to none at 0.9 m. Corner cutting sits below 0.5 m and is corrected; a plan 0.9 m off center has the wheels over the line, which is a decision (an obstacle, a lane change without a blinker), not drift.
- **Low speed:** below 10 m/s the fixed 10 m lookahead makes the correction fade with speed squared. The car is already centered there (0.01 m mean offset) and town lane lines are the least trustworthy.
- **Lateral control inactive, or the lateral maneuver tool supplying the curvature:** nothing is added and the filter resets, so the correction always ramps back in from zero.
- **A modelV2 message that is not shaped like one** (missing lines, x and y of different lengths, a non-finite result): no correction. Stock reads a single number from the model here, so a bad message cannot crash `controlsd`; this reads arrays, and returns zero for the malformed shapes in the tests rather than raising.

The idea of correcting the plan toward the lane lines comes from StarPilot's lane centering; the control law, gates and numbers here are this branch's own, from the measurements below.

There is no toggle and nothing new in the log. While engaged, the correction is `controlsState.desiredCurvature` minus `modelV2.action.desiredCurvature` wherever the limits are not active, and it can be recomputed exactly from `modelV2` and `carState`. `lagd` keeps learning the steering delay from `controlsState.desiredCurvature`, which is the command the car actually receives, correction included; `torqued` does not read it.

### Why these numbers

Measured on four drives (113 min of logs, 68 min engaged and hands-off, town to highway) on the Palisade:

| | |
|---|---|
| Car's offset from the lane center | mean 0.14 m left, σ 0.26 m, p90 0.47 m; the same on all four drives |
| Plan's offset 0.5 / 1 / 2 s ahead | 0.14 / 0.13 / 0.12 m left — the plan runs parallel to the lane, it does not return to center |
| Plan vs car's offset, straights | slope 0.86 at 1 s: the model closes 14 % of an offset per second, toward its own preferred spot |
| Persistence of the offset | autocorrelation 0.81 at 2 s, 0.59 at 5 s — slow hugging, not twitching, so no deadband is needed (frame-to-frame noise is 0.018 m) |
| In curves | plan 0.42 m inside in left curves above 1 m/s², 0.37 m inside in right curves; 77 % of the time beyond 0.5 m is in curves |
| Both lines usable | 80 % of engaged time; width p1–p99 2.67–4.49 m |
| Steering latency, desired to actual curvature | 0.30 s (cross-correlation), 0.29 s (`lateralDelay`) |

The correction adds position stiffness but almost no damping of its own, so latency between the command and the car turns into negative damping that the model's own heading control has to absorb. In a closed-loop simulation fitted to the numbers above, a gain of 0.3 with 0.5 s of smoothing starts to weave at 1.75× the measured latency or 2.5× the gain; 0.2 with 0.25 s tolerates 2.25× the latency and 5.75× the gain, and still takes the standing offset from 0.14 m to about 0.03 m in under 4 s. The shorter filter is where most of the gain margin comes from. The simulation's driving model is a two-number fit, so the margins are estimates; that is why they are wide, and why the weave is the first thing to look for on the road.

Replaying the branch's code open-loop over 52 engaged minutes of the same drives: correcting 56 % of the time, |lateral accel| p50 0.02 / p90 0.15 / max 0.199 m/s², mean 0.03 m/s² to the right, jerk at most 0.8 m/s³, 3 µs per `controlsd` frame on a PC.

### What it costs

- A deliberate move of less than 0.9 m gets smaller. Against a model holding a 0.6 m offset the estimate is about 0.5 m; beyond 0.9 m nothing changes.
- The lane center is measured from the camera. A device mounted off the car's centerline moves "center" by the same amount; the near-zero offset at low speed suggests most of the 0.14 m is how the model drives rather than the mount, but that is 3 minutes of data; check the mount if the car ends up right of center.
- With one line missing (20 % of engaged time) it does nothing.

## What changed

- `openpilot/selfdrive/controls/lib/lane_centering.py` — `get_lane_center_curvature` and the `LaneCentering` filter and gates.
- `openpilot/selfdrive/controls/controlsd.py` — adds the correction to the model's desired curvature, resets it while the lateral maneuver tool drives (four lines).
- `openpilot/selfdrive/controls/tests/test_lane_centering.py` — geometry, the 0.2 m/s² authority bound at every speed, each gate, malformed model messages, the hand-over to the driver, and a closed-loop run at 15/25/35 m/s with twice the measured steering latency that must settle within 5 cm of center without overshoot. Each of 19 single-line mutations of the module fails a test, a gain of 0.21 included.
