"""Pure analytic mapping between BLaTv2 curvature and steering-wheel motion.

The mapper mirrors opendbc's :class:`VehicleModel` scalar conventions:

  desired_angle = degrees(
    VM.get_steer_from_curvature(-reference_curvature, speed, roll)
  ) + angleOffsetDeg

Rate and acceleration are propagated through the same algebra with a
second-order dual number. No controller-frame finite difference, filter,
delay, measured-state feedback, torque calculation, or retained state exists
in this module.

VehicleModel's steady-state formula has physical singularities when its
curvature or roll denominator is zero. Those inputs raise ``ValueError``;
they are never silently clamped. Zero speed itself is supported because the
scalar VehicleModel mapping is well-defined there.
"""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
import math
from typing import Any

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY


@dataclass(frozen=True, slots=True)
class RackMappingSnapshot:
  """Frozen VehicleModel coefficients and live alignment for one frame."""

  mass_kg: float
  wheelbase_m: float
  center_to_front_m: float
  center_to_rear_m: float
  tire_stiffness_front: float
  tire_stiffness_rear: float
  steer_ratio_rear: float
  steer_ratio: float
  roll_rad: float
  angle_offset_deg: float
  valid: bool

  def __post_init__(self) -> None:
    values = (
      self.mass_kg,
      self.wheelbase_m,
      self.center_to_front_m,
      self.center_to_rear_m,
      self.tire_stiffness_front,
      self.tire_stiffness_rear,
      self.steer_ratio_rear,
      self.steer_ratio,
      self.roll_rad,
      self.angle_offset_deg,
    )
    if not all(math.isfinite(value) for value in values):
      raise ValueError("rack mapping snapshot must be finite")
    if self.mass_kg <= 0.0:
      raise ValueError("vehicle mass must be positive")
    if self.wheelbase_m <= 0.0:
      raise ValueError("wheelbase must be positive")
    if self.center_to_front_m < 0.0 or self.center_to_rear_m < 0.0:
      raise ValueError("axle distances must be non-negative")
    if self.tire_stiffness_front <= 0.0 or self.tire_stiffness_rear <= 0.0:
      raise ValueError("tire stiffness must be positive")
    if self.steer_ratio <= 0.0:
      raise ValueError("steer ratio must be positive")
    if 1.0 - self.steer_ratio_rear == 0.0:
      raise ValueError("rear steer ratio makes curvature mapping singular")

  @classmethod
  def from_vehicle_model(
    cls,
    vehicle_model: Any,
    *,
    roll_rad: float,
    angle_offset_deg: float,
    valid: bool,
  ) -> RackMappingSnapshot:
    """Snapshot the current, already-updated opendbc VehicleModel."""
    return cls(
      mass_kg=float(vehicle_model.m),
      wheelbase_m=float(vehicle_model.l),
      center_to_front_m=float(vehicle_model.aF),
      center_to_rear_m=float(vehicle_model.aR),
      tire_stiffness_front=float(vehicle_model.cF),
      tire_stiffness_rear=float(vehicle_model.cR),
      steer_ratio_rear=float(vehicle_model.chi),
      steer_ratio=float(vehicle_model.sR),
      roll_rad=float(roll_rad),
      angle_offset_deg=float(angle_offset_deg),
      valid=bool(valid),
    )


@dataclass(frozen=True, slots=True)
class RackReference:
  angle_deg: float
  rate_deg_s: float
  acceleration_deg_s2: float
  valid: bool
  degraded: bool

  def __post_init__(self) -> None:
    if not all(math.isfinite(value) for value in (
      self.angle_deg, self.rate_deg_s, self.acceleration_deg_s2,
    )):
      raise ValueError("rack reference must be finite")
    if self.valid == self.degraded:
      raise ValueError("valid and degraded must be logical opposites")


@dataclass(frozen=True, slots=True)
class RackMappingStatus:
  count: int
  valid: bool
  degraded: bool

  def __post_init__(self) -> None:
    if self.count <= 0:
      raise ValueError("rack mapping count must be positive")
    if self.valid == self.degraded:
      raise ValueError("valid and degraded must be logical opposites")


@dataclass(frozen=True, slots=True)
class CurvatureDiagnostic:
  curvature: float
  valid: bool
  degraded: bool

  def __post_init__(self) -> None:
    if not math.isfinite(self.curvature):
      raise ValueError("diagnostic curvature must be finite")
    if self.valid == self.degraded:
      raise ValueError("valid and degraded must be logical opposites")


@dataclass(frozen=True, slots=True)
class _Jet2:
  value: float
  first: float = 0.0
  second: float = 0.0

  @staticmethod
  def coerce(value: float | _Jet2) -> _Jet2:
    return value if isinstance(value, _Jet2) else _Jet2(float(value))

  def __add__(self, other: float | _Jet2) -> _Jet2:
    right = self.coerce(other)
    return _Jet2(
      self.value + right.value,
      self.first + right.first,
      self.second + right.second,
    )

  __radd__ = __add__

  def __neg__(self) -> _Jet2:
    return _Jet2(-self.value, -self.first, -self.second)

  def __sub__(self, other: float | _Jet2) -> _Jet2:
    return self + (-self.coerce(other))

  def __rsub__(self, other: float | _Jet2) -> _Jet2:
    return self.coerce(other) - self

  def __mul__(self, other: float | _Jet2) -> _Jet2:
    right = self.coerce(other)
    return _Jet2(
      self.value * right.value,
      self.first * right.value + self.value * right.first,
      (
        self.second * right.value
        + 2.0 * self.first * right.first
        + self.value * right.second
      ),
    )

  __rmul__ = __mul__

  def __truediv__(self, other: float | _Jet2) -> _Jet2:
    right = self.coerce(other)
    if right.value == 0.0:
      raise ValueError("rack mapping denominator is zero")
    denominator_squared = right.value * right.value
    denominator_cubed = denominator_squared * right.value
    return _Jet2(
      self.value / right.value,
      (
        self.first * right.value - self.value * right.first
      ) / denominator_squared,
      (
        self.second / right.value
        - 2.0 * self.first * right.first / denominator_squared
        - self.value * right.second / denominator_squared
        + 2.0 * self.value * right.first * right.first
        / denominator_cubed
      ),
    )

  def __rtruediv__(self, other: float | _Jet2) -> _Jet2:
    return self.coerce(other) / self


def _slip_factor(snapshot: RackMappingSnapshot) -> float:
  return (
    snapshot.mass_kg
    * (
      snapshot.tire_stiffness_front * snapshot.center_to_front_m
      - snapshot.tire_stiffness_rear * snapshot.center_to_rear_m
    )
    / (
      snapshot.wheelbase_m * snapshot.wheelbase_m
      * snapshot.tire_stiffness_front
      * snapshot.tire_stiffness_rear
    )
  )


def _select_snapshot(
  live_snapshot: RackMappingSnapshot | None,
  nominal_snapshot: RackMappingSnapshot,
) -> tuple[RackMappingSnapshot, bool]:
  if not nominal_snapshot.valid:
    raise ValueError("nominal rack mapping snapshot must be valid")
  if live_snapshot is None or not live_snapshot.valid:
    return nominal_snapshot, False
  return live_snapshot, True


def _map_jet(
  curvature: _Jet2,
  speed: _Jet2,
  snapshot: RackMappingSnapshot,
) -> _Jet2:
  if speed.value < 0.0:
    raise ValueError("reference speed must be non-negative")

  slip_factor = _slip_factor(snapshot)
  speed_squared = speed * speed
  curvature_denominator = 1.0 - slip_factor * speed_squared
  if curvature_denominator.value == 0.0:
    raise ValueError("VehicleModel curvature denominator is zero")
  curvature_factor = (
    (1.0 - snapshot.steer_ratio_rear)
    / curvature_denominator
    / snapshot.wheelbase_m
  )
  if curvature_factor.value == 0.0:
    raise ValueError("VehicleModel curvature factor is zero")

  if abs(slip_factor) < 1e-6:
    roll_compensation = _Jet2(0.0)
  else:
    roll_denominator = 1.0 / slip_factor - speed_squared
    if roll_denominator.value == 0.0:
      raise ValueError("VehicleModel roll denominator is zero")
    roll_compensation = (
      ACCELERATION_DUE_TO_GRAVITY * snapshot.roll_rad
      / roll_denominator
    )

  # Negative curvature is the stock controlsd/opendbc platform convention.
  return (
    (-curvature - roll_compensation)
    * snapshot.steer_ratio
    * 1.0
    / curvature_factor
  )


def map_reference(
  curvature: float,
  curvature_rate: float,
  curvature_acceleration: float,
  speed: float,
  speed_rate: float,
  speed_acceleration: float,
  live_snapshot: RackMappingSnapshot | None,
  nominal_snapshot: RackMappingSnapshot,
) -> RackReference:
  """Map one analytic curvature/speed sample into rack motion."""
  values = (
    curvature,
    curvature_rate,
    curvature_acceleration,
    speed,
    speed_rate,
    speed_acceleration,
  )
  if not all(math.isfinite(value) for value in values):
    raise ValueError("rack reference inputs must be finite")
  snapshot, inputs_valid = _select_snapshot(
    live_snapshot, nominal_snapshot,
  )
  angle = _map_jet(
    _Jet2(curvature, curvature_rate, curvature_acceleration),
    _Jet2(speed, speed_rate, speed_acceleration),
    snapshot,
  )
  return RackReference(
    angle_deg=math.degrees(angle.value) + snapshot.angle_offset_deg,
    rate_deg_s=math.degrees(angle.first),
    acceleration_deg_s2=math.degrees(angle.second),
    valid=inputs_valid,
    degraded=not inputs_valid,
  )


def map_reference_into(
  curvatures: Sequence[float],
  curvature_rates: Sequence[float],
  curvature_accelerations: Sequence[float],
  speeds: Sequence[float],
  speed_rates: Sequence[float],
  speed_accelerations: Sequence[float],
  count: int,
  live_snapshot: RackMappingSnapshot | None,
  nominal_snapshot: RackMappingSnapshot,
  output_angles_deg: MutableSequence[float],
  output_rates_deg_s: MutableSequence[float],
  output_accelerations_deg_s2: MutableSequence[float],
) -> RackMappingStatus:
  """Allocation-free mapping over caller-owned populated buffer prefixes."""
  if count <= 0:
    raise ValueError("rack mapping count must be positive")
  inputs = (
    curvatures,
    curvature_rates,
    curvature_accelerations,
    speeds,
    speed_rates,
    speed_accelerations,
  )
  outputs = (
    output_angles_deg,
    output_rates_deg_s,
    output_accelerations_deg_s2,
  )
  if any(len(buffer) < count for buffer in (*inputs, *outputs)):
    raise ValueError("rack mapping buffer is shorter than count")

  snapshot, inputs_valid = _select_snapshot(
    live_snapshot, nominal_snapshot,
  )
  for index in range(count):
    values = tuple(float(buffer[index]) for buffer in inputs)
    if not all(math.isfinite(value) for value in values):
      raise ValueError(f"rack reference sample {index} must be finite")
    angle = _map_jet(
      _Jet2(values[0], values[1], values[2]),
      _Jet2(values[3], values[4], values[5]),
      snapshot,
    )
    output_angles_deg[index] = (
      math.degrees(angle.value) + snapshot.angle_offset_deg
    )
    output_rates_deg_s[index] = math.degrees(angle.first)
    output_accelerations_deg_s2[index] = math.degrees(angle.second)

  return RackMappingStatus(
    count=count,
    valid=inputs_valid,
    degraded=not inputs_valid,
  )


def curvature_from_measured_angle(
  measured_angle_deg: float,
  speed: float,
  live_snapshot: RackMappingSnapshot | None,
  nominal_snapshot: RackMappingSnapshot,
) -> CurvatureDiagnostic:
  """Reverse diagnostic using the same stock VehicleModel convention."""
  if not math.isfinite(measured_angle_deg) or not math.isfinite(speed):
    raise ValueError("measured rack diagnostic inputs must be finite")
  if speed < 0.0:
    raise ValueError("measured rack diagnostic speed must be non-negative")
  snapshot, inputs_valid = _select_snapshot(
    live_snapshot, nominal_snapshot,
  )
  slip_factor = _slip_factor(snapshot)
  speed_squared = speed * speed
  curvature_denominator = 1.0 - slip_factor * speed_squared
  if curvature_denominator == 0.0:
    raise ValueError("VehicleModel curvature denominator is zero")
  curvature_factor = (
    (1.0 - snapshot.steer_ratio_rear)
    / curvature_denominator
    / snapshot.wheelbase_m
  )
  if abs(slip_factor) < 1e-6:
    roll_compensation = 0.0
  else:
    roll_denominator = 1.0 / slip_factor - speed_squared
    if roll_denominator == 0.0:
      raise ValueError("VehicleModel roll denominator is zero")
    roll_compensation = (
      ACCELERATION_DUE_TO_GRAVITY
      * snapshot.roll_rad
      / roll_denominator
    )
  steering_angle_rad = math.radians(
    measured_angle_deg - snapshot.angle_offset_deg
  )
  vehicle_model_curvature = (
    curvature_factor * steering_angle_rad / snapshot.steer_ratio
    + roll_compensation
  )
  return CurvatureDiagnostic(
    curvature=-vehicle_model_curvature,
    valid=inputs_valid,
    degraded=not inputs_valid,
  )
