# Model Curve Speed Limit

Status: **in progress pending field testing**.

## Goal

Slow for model-predicted curves before the Palisade exhausts steering authority.

## Curve-speed envelope

The existing 50/22/13 mph field envelope, spatial median, temporal median,
target-release rate, and `0.5 m/s²` approach calculation remain unchanged.
For valid Palisade torque parameters while lateral control is active, each
future signed-curvature sample also gets the maximum speed that keeps predicted
feedforward plus friction demand at or below `TORQUE_BUDGET = 0.93`. The lower
of the field and torque-budget speeds enters the existing approach-distance
calculation.

Valid filtered live torque parameters replace the static `CarParams` values.
Invalid geometry or unusable parameters retain field-envelope behavior rather
than inventing a torque limit.

## Torque veto

The existing two-of-three future-torque veto remains a backstop. When predicted
demand reaches the budget, the longitudinal planner clamps only positive final
acceleration to zero. Stronger MPC/e2e/lead/stop braking remains authoritative.

This predicts path-driven feedforward and friction demand—not future PID
feedback, tire grip, or a guaranteed full-controller 93% ceiling. Route replay
must check final acceleration and jerk, false slowing on ordinary bends, and
actual entry torque before owner field testing.
