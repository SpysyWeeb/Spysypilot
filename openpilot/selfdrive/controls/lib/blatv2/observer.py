"""Optional fast physical-disturbance observer for modular BLaTv2.

This observer owns one persistent estimate of unmodelled normalized rack
torque. It is not a controller and cannot see desired path or candidate
commands. Its only update source is the recorded actuator/rack response:

  rack_acceleration =
      rack_gain * (
        applied_torque - aligning_torque - friction_torque - disturbance
      )
    - rack_damping * rack_rate

Solving that same plant equation gives the instantaneous disturbance used
below. A first-order exact discrete update then estimates a slowly varying
physical bias without depending on scheduler timestamp jitter.

The observer has no default policy. It remains exactly zero when no explicit
``ObserverPolicy`` is supplied or while the physical vehicle profile is
unqualified. The disturbance bound is deliberately independent of static or
kinetic friction.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
import math

from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  PhysicalParameters,
)


class ObserverStatus(IntEnum):
  ACTIVE = 0
  DISABLED_NO_POLICY = 1
  DISABLED_UNQUALIFIED_PROFILE = 2
  RESET_ENGAGEMENT_BOUNDARY = 3
  RESET_LATERAL_INACTIVE = 4
  RESET_LATERAL_INVALID = 5
  RESET_STANDSTILL = 6
  RESET_MODEL_INVALID = 7
  RESET_VEHICLE_INVALID = 8
  RESET_LIVE_PARAMETERS_INVALID = 9
  RESET_MEASUREMENT_INVALID = 10
  FROZEN_STEERING_PRESSED = 11
  FROZEN_OUTPUT_CONSTRAINT = 12


@dataclass(frozen=True, slots=True)
class ObserverPolicy:
  """Acceptance-supplied physical estimator policy; no implicit defaults."""

  time_constant_s: float
  max_abs_disturbance_torque: float

  def __post_init__(self) -> None:
    if (
      not math.isfinite(self.time_constant_s)
      or self.time_constant_s <= 0.0
      or not math.isfinite(self.max_abs_disturbance_torque)
      or self.max_abs_disturbance_torque <= 0.0
    ):
      raise ValueError("observer policy values must be finite and positive")


@dataclass(frozen=True, slots=True)
class ObserverMeasurement:
  """Recorded-response inputs plus centralized lifecycle facts."""

  applied_torque: float
  rack_rate_deg_s: float
  rack_acceleration_deg_s2: float
  aligning_torque: float
  friction_torque: float
  lateral_active: bool
  lateral_valid: bool
  engagement_boundary: bool
  model_valid: bool
  vehicle_state_valid: bool
  live_parameters_valid: bool
  steering_pressed: bool
  actuator_constrained: bool
  output_constrained: bool
  standstill: bool

  @property
  def numeric_inputs_finite(self) -> bool:
    return all(math.isfinite(value) for value in (
      self.applied_torque,
      self.rack_rate_deg_s,
      self.rack_acceleration_deg_s2,
      self.aligning_torque,
      self.friction_torque,
    ))


@dataclass(frozen=True, slots=True)
class ObserverResult:
  status: ObserverStatus
  instantaneous_disturbance_torque: float
  estimated_disturbance_torque: float
  saturated: bool

  def __post_init__(self) -> None:
    if not all(math.isfinite(value) for value in (
      self.instantaneous_disturbance_torque,
      self.estimated_disturbance_torque,
    )):
      raise ValueError("observer result must be finite")


def instantaneous_disturbance_torque(
  measurement: ObserverMeasurement,
  parameters: PhysicalParameters,
) -> float:
  """Solve the shared rack equation for its unmodelled torque term."""
  if not measurement.numeric_inputs_finite:
    raise ValueError("observer measurement must be finite")
  return (
    measurement.applied_torque
    - measurement.aligning_torque
    - measurement.friction_torque
    - (
      measurement.rack_acceleration_deg_s2
      + parameters.rack_damping_per_s * measurement.rack_rate_deg_s
    ) / parameters.rack_gain_deg_s2_per_torque
  )


class DisturbanceObserver:
  """One read-only-to-candidates estimate updated from recorded response."""

  def __init__(
    self,
    policy: ObserverPolicy | None,
    fixed_dt_s: float,
  ) -> None:
    dt = float(fixed_dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
      raise ValueError("observer fixed dt must be finite and positive")
    self._policy = policy
    self._fixed_dt_s = dt
    self._estimate_torque = 0.0
    self._alpha = (
      0.0
      if policy is None
      else -math.expm1(-dt / policy.time_constant_s)
    )

  @property
  def estimate_torque(self) -> float:
    """Current estimate; consumers have no public mutation API."""
    return self._estimate_torque

  @property
  def fixed_dt_s(self) -> float:
    return self._fixed_dt_s

  def _zero_result(self, status: ObserverStatus) -> ObserverResult:
    self._estimate_torque = 0.0
    return ObserverResult(
      status=status,
      instantaneous_disturbance_torque=0.0,
      estimated_disturbance_torque=0.0,
      saturated=False,
    )

  def _frozen_result(self, status: ObserverStatus) -> ObserverResult:
    if self._policy is None:
      raise AssertionError("disabled observer cannot enter frozen state")
    return ObserverResult(
      status=status,
      instantaneous_disturbance_torque=0.0,
      estimated_disturbance_torque=self._estimate_torque,
      saturated=(
        abs(self._estimate_torque)
        >= self._policy.max_abs_disturbance_torque
      ),
    )

  def update(
    self,
    measurement: ObserverMeasurement,
    parameters: PhysicalParameters,
  ) -> ObserverResult:
    """Apply lifecycle policy, then one exact first-order observer update."""
    if self._policy is None:
      return self._zero_result(ObserverStatus.DISABLED_NO_POLICY)
    if not parameters.qualified:
      return self._zero_result(
        ObserverStatus.DISABLED_UNQUALIFIED_PROFILE,
      )
    if measurement.engagement_boundary:
      return self._zero_result(
        ObserverStatus.RESET_ENGAGEMENT_BOUNDARY,
      )
    if not measurement.lateral_active:
      return self._zero_result(ObserverStatus.RESET_LATERAL_INACTIVE)
    if not measurement.lateral_valid:
      return self._zero_result(ObserverStatus.RESET_LATERAL_INVALID)
    if measurement.standstill:
      return self._zero_result(ObserverStatus.RESET_STANDSTILL)
    if not measurement.model_valid:
      return self._zero_result(ObserverStatus.RESET_MODEL_INVALID)
    if not measurement.vehicle_state_valid:
      return self._zero_result(ObserverStatus.RESET_VEHICLE_INVALID)
    if not measurement.live_parameters_valid:
      return self._zero_result(
        ObserverStatus.RESET_LIVE_PARAMETERS_INVALID,
      )
    if not measurement.numeric_inputs_finite:
      return self._zero_result(
        ObserverStatus.RESET_MEASUREMENT_INVALID,
      )
    if measurement.steering_pressed:
      return self._frozen_result(
        ObserverStatus.FROZEN_STEERING_PRESSED,
      )
    if (
      measurement.actuator_constrained
      or measurement.output_constrained
    ):
      return self._frozen_result(
        ObserverStatus.FROZEN_OUTPUT_CONSTRAINT,
      )

    instantaneous = instantaneous_disturbance_torque(
      measurement, parameters,
    )
    unconstrained = (
      self._estimate_torque
      + self._alpha * (instantaneous - self._estimate_torque)
    )
    bound = self._policy.max_abs_disturbance_torque
    estimate = min(max(unconstrained, -bound), bound)
    saturated = abs(estimate) >= bound
    self._estimate_torque = estimate
    return ObserverResult(
      status=ObserverStatus.ACTIVE,
      instantaneous_disturbance_torque=instantaneous,
      estimated_disturbance_torque=estimate,
      saturated=saturated,
    )


def observer_measurement_field_names() -> tuple[str, ...]:
  """Expose the recorded-response-only input contract for audit/tests."""
  return tuple(field.name for field in fields(ObserverMeasurement))
