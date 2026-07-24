# Model Curve Speed Limit

Status: **in progress pending field testing**.

## Goal

Use the driving model's predicted path to impose an owner-calibrated maximum
speed through curves without introducing a separate turn-control state
machine. The existing longitudinal planner remains responsible for
acceleration, braking, and resuming the driver's cruise speed.

The limiter must:

- look ahead using the model path;
- reject isolated curvature prediction spikes;
- allow normal acceleration while below the curve maximum;
- stop accelerating at the maximum;
- reduce speed before a curve only as early as required to meet the maximum;
- release the cap automatically as the predicted path straightens; and
- work for low-speed turns from a stop.

## Field calibration

The initial maximum-speed envelope comes from three owner-driven route points
recorded on 2026-07-24:

| Measured vehicle curvature | Model-path curvature | Maximum speed |
|---:|---:|---:|
| 0.00524 1/m | 0.00501 1/m | 38 mph |
| 0.04339 1/m | 0.04666 1/m | 19 mph |
| 0.07056 1/m | 0.08188 1/m | 11 mph |

The model-path values are used by the online limiter because that is its input;
the measured vehicle values document the physical curves that produced them.
These are field-test starting points, not completed tuning.

## Design

For each valid model horizon sample:

1. Compute curvature from predicted yaw rate and speed.
2. Apply a three-sample spatial median to reject an isolated bad path point.
3. Convert curvature to a maximum speed using the field envelope.
4. Compute the maximum speed allowed now to reach that future maximum with
   comfortable deceleration over the predicted path distance.
5. Apply a three-frame temporal median to the resulting allowance so one bad
   model prediction cannot cause a sudden brake request.

The lowest future allowance caps the driver's cruise setpoint before the
existing longitudinal planner runs. No acceleration command is overridden,
and no entering/turning/leaving state machine is added.

The legacy instantaneous steering-angle total-acceleration limiter is retired;
the calibrated speed cap now determines when curve-related acceleration must
stop.
