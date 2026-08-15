# Model Curve Speed Limit

Status: **in progress pending field testing**.

## Goal

Keep the existing model-path curve speed limiter while testing whether future
path curvature can prevent additional throttle before steering authority is
exhausted.

## Existing speed envelope

The 50/22/13 mph curve envelope, spatial median, temporal median, and 0.5 m/s²
approach calculation remain unchanged. They continue to own curve-entry speed
and braking during this experiment.

## Future-torque veto

For the Palisade's linear torque mapping, every valid future path sample is
converted to normalized feedforward demand at the current vehicle speed using
the controller's lateral-acceleration factor, offset, friction allowance, and
live roll compensation. Valid filtered live torque parameters replace the
static `CarParams` values.

When predicted demand reaches `TORQUE_BUDGET = 0.95` on two of three consecutive
model frames, the longitudinal planner clamps only positive final acceleration
to zero:

```python
output_a_target = min(output_a_target, 0.0)
```

Existing braking, stronger MPC/e2e/lead/stop deceleration, and driver override
remain authoritative. Invalid geometry, inactive lateral control, or unusable
parameters release the veto through the same two-of-three debounce rather than
one-frame chatter.

This estimates path-driven feedforward and friction demand—not future PID
feedback, tire grip, or a guaranteed full-controller 95% ceiling. The manual
speed envelope must remain until exact-hash closed-loop route evidence and owner
field testing show that the predictor activates early enough without needless
throttle suppression.
