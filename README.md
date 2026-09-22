# lane-centering

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: in progress; awaiting owner field validation.** Watch for a slow weave with a period of 8–10 s on a straight highway, for the car running wider than before through left curves, for how much room it still gives a cyclist or a parked car, and on roads with only one painted line for whether it stays centred or pulls toward the missing line.

## What it does

Keeps the car in the middle of the lane lines the model sees. The driving model is content to sit off-center: on the 2021 Hyundai Palisade it rides 0.21 m left of the midpoint between the lines on average (0.14 m with the previous model), is more than 0.56 m off for a tenth of the time, and cuts curves by about 0.4 m to the inside. Left to itself it drifts back toward *its own* preferred spot, not the center, and slowly: it closes about 6 % of an offset per second.

This branch adds a small steering correction toward the lane center on top of the model's. The model still drives: the correction is at most 0.2 m/s² of lateral acceleration (the model's own demand is 1.1 m/s² at the 90th percentile), and it lets go when the driver or the model means to leave the lane. With one line painted over or worn away it keeps going from the other line and the lane width it learned while both were visible.

## How it works

Every model frame, `get_lane_center` looks one second ahead (at least 10 m) and compares the midpoint of the two lane lines with where the model's plan puts the car, over the plan points up to that distance. The road's curvature is in both, so only the offset between them is left. A straight line fitted to that offset gives where the plan sits and which way it is heading relative to the lane; the target error is the offset plus twice the heading's contribution at the lookahead, bounded to ±0.5 m, so a plan heading away from the center is corrected before the gap has opened and a plan already coming back is left alone. The correction is the curvature of the arc that closes 20 % of that error by the lookahead, `0.2 · 2 · error / lookahead²`, added to `modelV2.action.desiredCurvature` in `controlsd` before `clip_curvature`, so stock's lateral jerk and acceleration limits still apply to the sum. A 0.25 s first-order filter at 100 Hz smooths the 20 Hz steps.

**One line missing.** While both lines are visible the lane width at the lookahead is learned through a 5 s filter; a width that differs by more than 0.4 m for five consecutive frames replaces it outright, because that is a new road, not noise. When exactly one line is valid, the center is that line plus or minus half the learned width, for as long as the line stays valid and the width is less than 60 s old. The learned width cannot be checked against the line that is gone, only against the model's plan, which sees the whole road: if the center it gives moves more than 0.25 m relative to the plan compared with the last frame that had both lines, the width is wrong for this road and the correction stops.

It fades out instead of fighting:

- **No lane:** a line below 0.5 probability or above 0.3 m standard deviation does not count; with both lines, a lane narrower than 2.5 m or wider than 4.5 m (merges, splits, misread lines) gives no correction; with one line and no learned width, none either.
- **Leaving the lane on purpose:** steering pressed, either blinker, or a lane change in progress — the correction decays to zero over the filter's time constant.
- **The model means it:** full correction up to 0.5 m between the plan and the center, tapering to none at 0.9 m. Corner cutting sits below 0.5 m and is corrected; a plan 0.9 m off center has the wheels over the line, which is a decision (an obstacle, a lane change without a blinker), not drift.
- **Low speed:** below 10 m/s the fixed 10 m lookahead makes the correction fade with speed squared. The car is already centered there and town lane lines are the least trustworthy.
- **Lateral control inactive, or the lateral maneuver tool supplying the curvature:** nothing is added and the filter resets, so the correction always ramps back in from zero.
- **A modelV2 message that is not shaped like one** (missing lines, x and y of different lengths, a plan with non-finite distances, a non-finite result): no correction. Stock reads a single number from the model here, so a bad message cannot crash `controlsd`; this reads arrays, and returns zero for the malformed shapes in the tests rather than raising.

The idea of correcting the plan toward the lane lines comes from StarPilot's lane centering, the heading term and the one-line hold from gm1500's `lp-e2e-blend`; the control law, gates and numbers here are this branch's own, from the measurements below.

There is no toggle and nothing new in the log. While engaged, the correction is `controlsState.desiredCurvature` minus `modelV2.action.desiredCurvature` wherever the limits are not active, and everything above, the learned width included, can be recomputed exactly from `modelV2` and `carState`. `lagd` keeps learning the steering delay from `controlsState.desiredCurvature`, which is the command the car actually receives, correction included; `torqued` does not read it.

### Why these numbers

Measured on the Palisade: four drives of 2026-09-10 to 09-14 (68 min engaged and hands-off) for the first design, then the two drives of 2026-09-19 (52 min, 25 min measurable) on the current upstream model, which sits further left.

| | |
|---|---|
| Car's offset from the lane center | mean 0.21 m left, rms 0.34 m; 39 % of the time more than 0.3 m off; the body within 15 cm of a line 17 % of the time |
| Model's own recentring | the plan keeps 94 % of an offset after 1 s: it returns to its own spot over about 17 s |
| In curves | 0.39 m inside in left curves, 0.19 m inside in firm right curves |
| Lines | both usable 72 % of engaged time, exactly one 21 % (left only 13 %, right only 7 %), neither 8 %; one-line stretches are mostly under a second but the long ones hold most of the time: a 1 s hold covers 44 % of it, 20 s covers 97 % |
| Lane width | median 3.29 m, p1–p99 2.67–4.49 m; it drifts 0.12 m over 1 s and 0.24 m over 30 s at the 90th percentile |
| Steering latency, desired to actual curvature | 0.30–0.35 s (cross-correlation), 0.29 s (`lateralDelay`) |

The correction adds position stiffness, so latency between the command and the car turns into negative damping that the model's own heading control has to absorb. The heading term supplies the damping the lookahead alone lacks. In a closed-loop replay of the two drives, with the recorded lane lines and a driving model fitted to the numbers above, the plain lookahead law weaves with an 8–10 s period once the model straightens up slowly (2 s) or the latency reaches 0.45 s; with the heading term at twice the lookahead it does not weave in any case tried, including 3 s and 0.45 s, and the car moves sideways 15 % less than it did in the recording instead of 2 % more. A heading gain of 1.5 still weaves in the worst case; leaving the error unbounded reaches 0.4 m/s²; a longer lookahead removes the weave but centres worse, since authority falls with its square. The rest of the replay: rms offset 0.34 → 0.13 m, time more than 0.3 m off 39 % → 4 %, body within 15 cm of a line 17 % → 1 %, left curves 0.39 → 0.05 m; active 78 % of engaged time (62 % without the hold); correction p99 0.18 and max 0.202 m/s²; jerk p99 0.29 and max 0.68 m/s³.

The one-line hold was checked on 1,482 recorded stretches where the lost line came back: the center rebuilt from the other line and the learned width was within 0.16 m of the true one 90 % of the time, and the error does not grow with how long the hold lasts — it is set by why the line was lost. Where it was lost at an exit or a merge the width is wrong, which is what the plan check is for: it drops 87 % of the holds that would have ended more than 0.3 m off and 3 % of the good ones. The snap rule comes from a right turn off a 4.2 m road onto a 3.3 m one with only the left line: the 5 s filter kept the wide width for two minutes and the check had to drop nearly every hold, while a width taken from a 0.4 s glimpse of both lines makes the rest of that road centred. Fleet-wide the snap fires about once a minute and makes the hold error smaller, not larger.

### What it costs

- A deliberate move of less than 0.9 m gets smaller. Against a model holding a 0.6 m offset the estimate is about 0.5 m; beyond 0.9 m nothing changes.
- The lane center is measured from the camera. A device mounted off the car's centerline moves "center" by the same amount; the left offset grows with speed, which a mount error would not do, so most of it is how the model drives — but check the mount if the car ends up right of center.
- With one line and no width learned in the last 60 s, or with neither line (8 % of engaged time), it does nothing.
- Real lanes narrower than 2.5 m exist (a 2.4 m stretch in the logs); the width gate stands the two-line correction down there.

## What changed

- `openpilot/selfdrive/controls/lib/lane_centering.py` — `get_lane_center` (the fit, the fade and the gates), `LaneCentering` (the filter, the learned width and the plan check).
- `openpilot/selfdrive/controls/controlsd.py` — adds the correction to the model's desired curvature, resets it while the lateral maneuver tool drives (four lines).
- `openpilot/selfdrive/controls/tests/test_lane_centering.py` — geometry, the heading term, the 0.2 m/s² bound over offsets and headings at every speed, each gate, malformed model messages, the hand-over to the driver, the hold (needs a width, a valid line and a plan that agrees; expires; snaps to a new road within five frames and filters small changes), and a closed-loop run at 15/25/35 m/s with 0.4–0.45 s of steering latency and a model straightening up over 1–2 s that must settle within 5 cm of center without overshoot. Each of 29 single-line mutations of the module fails a test, a gain of 0.21 and a heading gain of 1 or 3 included.
