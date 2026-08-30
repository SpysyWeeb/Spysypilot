# Curve longitudinal policy

Status: **in progress pending field testing** (rebuilt 2026-08-29; the previous speed-cap limiter is described at the end).

## What it does

`model_curve_speed.py` decides how the car should accelerate through the curves the model path shows ahead and the
one it is in, and hands the planner **one acceleration candidate** (plan source `curve`). The planner's `min()`
arbitration can only lower the chosen acceleration with it. The cruise target, the stop bit and the mode are untouched.

### Layer 1 — anticipation (the path ahead)
Every model path node gets a curvature (yaw rate ÷ speed, 3-node spatial median) and two speed limits:
- **authority** — the speed at which the steering demand *at that node, at that node's speed* stays inside
  `TORQUE_BUDGET` (0.90) of the EPS limit: `v² = ((budget − friction)·factor ± roll·g + offset) / κ`;
- **comfort** — `v² = A_LAT_COMFORT / κ` with `A_LAT_COMFORT = 3.0 m/s²`, the owner's own turns (p90 of manual driving).
  Inert on the Palisade today because the authority binds first; it is the ceiling should the torque headroom grow.

Per node the candidate is the less demanding of the kinematic acceleration that meets the limit at its distance and a
proportional approach `(v_limit − v)/T_APPROACH`; the strictest node wins. That is small and positive near a limit
(no burst toward it), zero at it, and the needed deceleration beyond it. It is floored at `A_CURVE_MIN` (−2.0: harder
braking is the lead and stop logic's business), 3-frame median filtered, and jerk-limited from the car's own
acceleration (`J_DOWN` 2.0 m/s³ down, `J_UP` 3.0 up: a curve releases in about a second).

### Steering authority calibrated online
The feedforward torque model under-reads what the steering achieves (route 22: 3.1 m/s² lateral at 0.9 torque
against 2.14 predicted). The lateral acceleration achieved per unit of torque above friction is measured whenever the
steering is working (torque ≥ 0.40) and tracking (error ≤ 0.30), filtered with a 5 s time constant, and bounded to
0.8–1.8 × the torque tuning's own factor. It starts at the tuning's factor on every boot, so the first heavy curve of a
drive is approached conservatively.

### Layer 2 — reaction (the steering now)
Reads `controlsState.lateralControlState` (`torqueState` on stock, `rackState` on combo): torque, tracking error,
actual and desired lateral acceleration, saturated / torque-limited. Only while openpilot steers and the driver is not.
- **coast** — torque ≥ `T_COAST` (0.85) for `COAST_ENTER_S` (0.3 s): throttle off, no brake (`min(coast, 0)`, so never a
  net acceleration downhill); demand falls with v². Ends after `COAST_EXIT_S` below `T_COAST_EXIT` (0.75).
- **brake** — pinned (`T_PIN` 0.95, saturated, or torque limited) *and* understeering (`sign(desired)·error ≥ E_TRACK`
  0.30, so an exit overshoot never triggers it) for `BRAKE_ENTER_S` (0.3 s): decelerate toward the speed at which the
  measured curvature fits the budget within `T_RESTORE` (1 s); back to coast below `E_TRACK_EXIT`. Never below
  `V_REACT_MIN` (3 m/s), where the measured curvature is noise.
- **free** otherwise. Regimes are dwelled at every edge; a single sample never switches them.

The final candidate is `max(min(anticipation, reaction), A_CURVE_MIN)`, jerk-limited as one signal, and absent
(`None`) whenever it would not bind or is not finite. The planner adds it only while engaged (`not reset_state`).

## Why (field evidence, 2026-08-29, routes 22–26)
Steering pinned (≥ 0.95) 8.4 s in ~2 min of engaged turning, tracking ~75 % of that time. The old limiter's predicted
torque (max over the 10 s path at the *current* speed) ran +0.15–0.2 above actual (p90 +0.4); of its 18 braking episodes
6 were needed and 12 not; its 0.2 m/s² cap release produced 186 s of crawl after curves; 7 bursts of ≥ 1.5 m/s² toward a
cap less than 3 m/s above the car. Route 22 t=2934–2986: the driver held a banked sweeper at 23–25 m/s under
always-on lateral with the steering at 0.45–0.9 torque and error ≤ 0.3 while the limiter wanted 23 m/s with the veto on.

## Validation
- `test_model_curve_speed.py`: limits, kinematic/proportional candidate, spikes and invalid models, jerk limits and floor,
  coast/brake dwell and hysteresis, understeer sign, both lateral state kinds, authority calibration and bounds, the
  candidate never raising the plan, and a guard that the legacy veto/cap surface is gone from the planner.
- `test_longitudinal.py::TestCurvePolicy`: the maneuver plant gained a world-fixed curve with a synthetic torque state
  (`Plant(curve=(start, length, curvature))`); approach without a burst within the floor, coast when heavy, brake when the
  steering cannot hold the line.
- Replay (`~/Documents/blotv3/replay/curve_gates.py`, like-for-like on this branch): crawl 84/24/28/62 s → 0/0.1/0.3/0 s,
  hard-braking frames unchanged, 12 of the 18 audit episodes exact, the rest between; the route-22 sweeper is approached
  conservatively on a cold calibration (limit 20 m/s before the filter has learned that curve) — the first-drive check.

## Open
- Persist the calibrated authority across boots (torqued-style) so the first curve of a drive is not the cold case.
- A speed-dependent authority if the field shows the single factor too low at highway speed.
- Experimental mode: the e2e candidate also slows for curves; the two simply share the `min()`.

## Previous design (replaced)
A model-path curve speed cap fed in as a lower `v_cruise` (three field points 50/22/13 mph, a Palisade torque-budget
speed, 0.5 m/s² approach, 0.2 m/s² release) plus a two-of-three predicted-torque veto that clamped positive acceleration.

## 2026-08-31 — ride the authority, hold it steady (owner ruling)

`A_LAT_COMFORT` 3.0 → 3.4 m/s²: the owner's manual cornering tops out at 2.86 (181 manual cornering frames in the whole
archive — always-on-lateral steers everything else), and the ruling is to push closer to the limit. With comfort above the
calibrated authority, the steering's own per-curve, bank-aware ceiling binds nearly everywhere; comfort stays as the
backstop against an implausible learned authority. And `V_HOLD_BAND` 0.3 m/s: within the band of the binding limit the
candidate is a flat zero, so the settled car holds the limit instead of stitching gas/brake corrections across the zero
crossing; drift is corrected at the band edges by the proportional approach. Route 22 sweeper: the in-curve limit moves
from ~25–26 (comfort-bound) to ~27–28 m/s (authority-bound with the bank).

## 2026-08-31 — a gas override earns a grace

`CURVE_GAS_GRACE_S` 5.0: after the driver's gas press, the anticipation layer may hold the speed but never pull it back down
for five seconds — route 0x2c t=885: the owner released the pedal 1.4 m/s above the in-curve limit and the still-active episode
dragged the exit from +1.8 back to −1.0 mid-corner. The reaction brake regime (pinned and understeering) still runs inside the
grace, and the grace re-arms on every gas frame.
