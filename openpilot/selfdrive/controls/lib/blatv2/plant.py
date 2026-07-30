"""Shared deterministic rack plant and computed-torque inverse.

The same equation is used for forward replay and live inverse control:

```
angle_acceleration =
    rack_gain * (applied - road_load - friction - disturbance)
  - rack_damping * angle_rate
```

The controller's unconstrained request is the exact inverse of that equation
with one critically damped tracking correction. There is no maneuver
classifier, low-speed mode, torque boost, integral, or smoothing filter.
Speed dependence enters only through the immutable physical vehicle profile.

Friction uses a continuous presliding transition whose width is the measured
rack-rate resolution stored in the vehicle profile. At zero measured motion,
static breakaway is available; as observable motion develops, the load moves
continuously to kinetic friction. This is physical sensor/model handling, not
a turn-state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
  curvature_from_measured_angle,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  PhysicalParameters,
)


@dataclass(frozen=True, slots=True)
class RackState:
  angle_deg: float
  rate_deg_s: float
  applied_torque: float

  def __post_init__(self) -> None:
    if not all(math.isfinite(value) for value in (
      self.angle_deg, self.rate_deg_s, self.applied_torque,
    )):
      raise ValueError("rack state must be finite")


@dataclass(frozen=True, slots=True)
class TrackingPolicy:
  """Speed-independent closed-loop response objective.

  Physical speed variation is already represented by the profile. Keeping
  this policy speed-independent prevents a second hidden speed schedule.
  """

  natural_frequency_per_s: float
  damping_ratio: float = 1.0

  def __post_init__(self) -> None:
    if (
      not math.isfinite(self.natural_frequency_per_s)
      or self.natural_frequency_per_s <= 0.0
      or not math.isfinite(self.damping_ratio)
      or self.damping_ratio <= 0.0
    ):
      raise ValueError("tracking policy must be finite and positive")


@dataclass(frozen=True, slots=True)
class PlantStep:
  state: RackState
  acceleration_deg_s2: float
  aligning_torque: float
  friction_torque: float
  disturbance_torque: float
  stuck: bool


@dataclass(frozen=True, slots=True)
class ComputedTorque:
  raw_torque: float
  aligning_torque: float
  friction_torque: float
  motion_feedforward_torque: float
  position_feedback_torque: float
  rate_feedback_torque: float
  disturbance_torque: float
  position_error_deg: float
  rate_error_deg_s: float
  required_acceleration_deg_s2: float

  def __post_init__(self) -> None:
    if not all(math.isfinite(value) for value in (
      self.raw_torque,
      self.aligning_torque,
      self.friction_torque,
      self.motion_feedforward_torque,
      self.position_feedback_torque,
      self.rate_feedback_torque,
      self.disturbance_torque,
      self.position_error_deg,
      self.rate_error_deg_s,
      self.required_acceleration_deg_s2,
    )):
      raise ValueError("computed torque terms must be finite")


def steady_road_load_torque(
  curvature: float,
  speed_mps: float,
  roll_rad: float,
  lateral_accel_offset: float,
  torque_per_lateral_accel: float,
) -> float:
  """Mirror openpilot's gravity/roll/offset torque convention."""
  values = (
    curvature,
    speed_mps,
    roll_rad,
    lateral_accel_offset,
    torque_per_lateral_accel,
  )
  if not all(math.isfinite(value) for value in values):
    raise ValueError("steady road-load inputs must be finite")
  if speed_mps < 0.0 or torque_per_lateral_accel <= 0.0:
    raise ValueError("steady road-load speed/gain is outside its domain")
  gravity_adjusted_lateral_accel = (
    curvature * speed_mps * speed_mps
    - roll_rad * ACCELERATION_DUE_TO_GRAVITY
    - lateral_accel_offset
  )
  # Positive model curvature maps to negative platform steering/torque.
  return -gravity_adjusted_lateral_accel * torque_per_lateral_accel


def presliding_friction_magnitude(
  rate_deg_s: float,
  parameters: PhysicalParameters,
) -> float:
  """Continuously move from static to kinetic load over one sensor quantum."""
  rate = float(rate_deg_s)
  if not math.isfinite(rate):
    raise ValueError("rack rate must be finite")
  resolution = parameters.rack_rate_resolution_deg_s
  if resolution == 0.0:
    moving_fraction = 1.0 if rate != 0.0 else 0.0
  else:
    moving_fraction = min(abs(rate) / resolution, 1.0)
  return (
    parameters.static_friction_torque
    + moving_fraction
    * (
      parameters.kinetic_friction_torque
      - parameters.static_friction_torque
    )
  )


def departure_friction_torque(
  measured_rate_deg_s: float,
  departure_direction: float,
  parameters: PhysicalParameters,
) -> float:
  """Return the signed friction command needed for the intended rack motion."""
  rate = float(measured_rate_deg_s)
  direction = rate if rate != 0.0 else float(departure_direction)
  if not math.isfinite(direction):
    raise ValueError("departure direction must be finite")
  if direction == 0.0:
    return 0.0
  return math.copysign(
    presliding_friction_magnitude(rate, parameters),
    direction,
  )


def step_plant(
  state: RackState,
  applied_torque: float,
  speed_mps: float,
  mapping: RackMappingSnapshot,
  nominal_mapping: RackMappingSnapshot,
  lateral_accel_offset: float,
  parameters: PhysicalParameters,
  disturbance_torque: float,
  dt: float,
) -> PlantStep:
  """Advance one semi-implicit Euler step of the shared rack equation."""
  applied = float(applied_torque)
  speed = float(speed_mps)
  disturbance = float(disturbance_torque)
  step_seconds = float(dt)
  if not all(math.isfinite(value) for value in (
    applied, speed, lateral_accel_offset, disturbance, step_seconds,
  )):
    raise ValueError("plant step inputs must be finite")
  if speed < 0.0 or step_seconds <= 0.0:
    raise ValueError("plant speed/dt is outside its valid domain")

  measured_curvature = curvature_from_measured_angle(
    state.angle_deg,
    speed,
    mapping,
    nominal_mapping,
  ).curvature
  selected_mapping = mapping if mapping.valid else nominal_mapping
  aligning = steady_road_load_torque(
    measured_curvature,
    speed,
    selected_mapping.roll_rad,
    lateral_accel_offset,
    parameters.torque_per_lateral_accel,
  )
  net_before_friction = applied - aligning - disturbance
  friction_magnitude = presliding_friction_magnitude(
    state.rate_deg_s, parameters,
  )

  stuck = False
  if state.rate_deg_s == 0.0:
    if abs(net_before_friction) <= parameters.static_friction_torque:
      friction = net_before_friction
      effective_torque = 0.0
      stuck = True
    else:
      friction = math.copysign(
        parameters.static_friction_torque,
        net_before_friction,
      )
      effective_torque = net_before_friction - friction
  else:
    friction = math.copysign(friction_magnitude, state.rate_deg_s)
    effective_torque = net_before_friction - friction

  acceleration = (
    parameters.rack_gain_deg_s2_per_torque * effective_torque
    - parameters.rack_damping_per_s * state.rate_deg_s
  )
  next_rate = state.rate_deg_s + acceleration * step_seconds
  # Semi-implicit Euler can step across zero while Coulomb friction still has
  # the sign selected from the *old* rack rate. Re-evaluate the force at zero:
  # friction and damping may stop a rack, but cannot reverse it. A reversal is
  # possible only when the non-friction load exceeds static breakaway in the
  # opposite direction. In that case, split the frame at the zero crossing so
  # the post-crossing acceleration uses the new direction's static friction.
  if state.rate_deg_s * next_rate < 0.0:
    old_direction = math.copysign(1.0, state.rate_deg_s)
    load_direction = (
      math.copysign(1.0, net_before_friction)
      if net_before_friction != 0.0
      else 0.0
    )
    supports_reversal = (
      abs(net_before_friction) > parameters.static_friction_torque
      and load_direction == -old_direction
    )
    if not supports_reversal:
      next_rate = 0.0
      stuck = True
    else:
      time_to_zero = min(
        max(-state.rate_deg_s / acceleration, 0.0),
        step_seconds,
      )
      remaining_time = step_seconds - time_to_zero
      post_crossing_effective_torque = (
        net_before_friction
        - math.copysign(
          parameters.static_friction_torque,
          net_before_friction,
        )
      )
      post_crossing_acceleration = (
        parameters.rack_gain_deg_s2_per_torque
        * post_crossing_effective_torque
      )
      next_rate = post_crossing_acceleration * remaining_time
    # Report the acceleration actually represented by the discrete step.
    acceleration = (
      next_rate - state.rate_deg_s
    ) / step_seconds
  next_angle = state.angle_deg + next_rate * step_seconds
  return PlantStep(
    state=RackState(next_angle, next_rate, applied),
    acceleration_deg_s2=acceleration,
    aligning_torque=aligning,
    friction_torque=friction,
    disturbance_torque=disturbance,
    stuck=stuck,
  )


def predict_applied_history(
  state: RackState,
  applied_torques: Sequence[float],
  speed_mps: float,
  mapping: RackMappingSnapshot,
  nominal_mapping: RackMappingSnapshot,
  lateral_accel_offset: float,
  parameters: PhysicalParameters,
  disturbance_torque: float,
  dt: float,
) -> RackState:
  """Advance through commands already committed inside transport delay."""
  predicted = state
  for applied_torque in applied_torques:
    predicted = step_plant(
      predicted,
      applied_torque,
      speed_mps,
      mapping,
      nominal_mapping,
      lateral_accel_offset,
      parameters,
      disturbance_torque,
      dt,
    ).state
  return predicted


def compute_inverse_torque(
  predicted_state: RackState,
  desired_curvature: float,
  desired_angle_deg: float,
  desired_rate_deg_s: float,
  desired_acceleration_deg_s2: float,
  speed_mps: float,
  roll_rad: float,
  lateral_accel_offset: float,
  parameters: PhysicalParameters,
  policy: TrackingPolicy,
  disturbance_torque: float,
) -> ComputedTorque:
  """Compute the sole unconstrained torque request for one physical time."""
  values = (
    desired_curvature,
    desired_angle_deg,
    desired_rate_deg_s,
    desired_acceleration_deg_s2,
    speed_mps,
    roll_rad,
    lateral_accel_offset,
    disturbance_torque,
  )
  if not all(math.isfinite(value) for value in values):
    raise ValueError("inverse torque inputs must be finite")
  if speed_mps < 0.0:
    raise ValueError("inverse torque speed must be non-negative")

  position_error = desired_angle_deg - predicted_state.angle_deg
  rate_error = desired_rate_deg_s - predicted_state.rate_deg_s
  frequency = policy.natural_frequency_per_s
  position_acceleration = frequency * frequency * position_error
  rate_acceleration = (
    2.0 * policy.damping_ratio * frequency * rate_error
  )
  required_acceleration = (
    desired_acceleration_deg_s2
    + position_acceleration
    + rate_acceleration
  )

  aligning = steady_road_load_torque(
    desired_curvature,
    speed_mps,
    roll_rad,
    lateral_accel_offset,
    parameters.torque_per_lateral_accel,
  )
  departure_direction = (
    desired_rate_deg_s
    if desired_rate_deg_s != 0.0
    else position_error
  )
  friction = departure_friction_torque(
    predicted_state.rate_deg_s,
    departure_direction,
    parameters,
  )
  motion_feedforward = (
    desired_acceleration_deg_s2
    + parameters.rack_damping_per_s * desired_rate_deg_s
  ) / parameters.rack_gain_deg_s2_per_torque
  position_feedback = (
    position_acceleration
    / parameters.rack_gain_deg_s2_per_torque
  )
  rate_feedback = (
    rate_acceleration
    + parameters.rack_damping_per_s
    * (predicted_state.rate_deg_s - desired_rate_deg_s)
  ) / parameters.rack_gain_deg_s2_per_torque
  raw_torque = (
    aligning
    + friction
    + motion_feedforward
    + position_feedback
    + rate_feedback
    + disturbance_torque
  )
  return ComputedTorque(
    raw_torque=raw_torque,
    aligning_torque=aligning,
    friction_torque=friction,
    motion_feedforward_torque=motion_feedforward,
    position_feedback_torque=position_feedback,
    rate_feedback_torque=rate_feedback,
    disturbance_torque=disturbance_torque,
    position_error_deg=position_error,
    rate_error_deg_s=rate_error,
    required_acceleration_deg_s2=required_acceleration,
  )
