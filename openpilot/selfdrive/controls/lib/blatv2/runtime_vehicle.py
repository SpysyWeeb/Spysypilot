"""Construction-time adaptation from detected opendbc vehicle facts.

This module is intentionally platform-neutral. It copies the detected
actuator envelope, verifies that the car interface's production torque
conversion fits the scalar-linear profile schema, snapshots the detected
VehicleModel, and builds an unqualified physical seed. Unsupported mappings
fail closed so stock control remains the only eligible controller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

from opendbc.car.structs import CarParams
from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  VehicleProfile,
  make_seed_profile,
)


RUNTIME_VEHICLE_SCHEMA_VERSION = 1
PROVISIONAL_RACK_DYNAMICS_SCHEMA_VERSION = 1
# This multiplier admits only accumulated binary64/API representation error.
# It is not a steering tolerance or feel dial.
_CALLBACK_REPRESENTATION_EPSILON_MULTIPLIER = 128.0


class RuntimeVehicleCompatibility(StrEnum):
  SUPPORTED = "supported"
  INVALID_IDENTITY = "invalid_identity"
  INVALID_PROVISIONAL_DYNAMICS = "invalid_provisional_dynamics"
  UNSUPPORTED_STEER_CONTROL = "unsupported_steer_control"
  UNSUPPORTED_LATERAL_TUNING = "unsupported_lateral_tuning"
  INVALID_CONTROLLER_LIMITS = "invalid_controller_limits"
  MISSING_RACK_RATE_RESOLUTION = "missing_rack_rate_resolution"
  INVALID_VEHICLE_CALIBRATION = "invalid_vehicle_calibration"
  MISSING_TORQUE_CALLBACK = "missing_torque_callback"
  INCOMPATIBLE_TORQUE_CALLBACK = "incompatible_torque_callback"


class RuntimeVehicleCompatibilityError(ValueError):
  """Fail-closed construction error with a stable compatibility reason."""

  def __init__(
    self,
    status: RuntimeVehicleCompatibility,
    message: str,
  ) -> None:
    super().__init__(message)
    self.status = status


def _fail(
  status: RuntimeVehicleCompatibility,
  message: str,
) -> None:
  raise RuntimeVehicleCompatibilityError(status, message)


@dataclass(frozen=True, slots=True)
class ProvisionalRackDynamics:
  """Explicit unqualified dynamics used only for shadowing and training."""

  rack_gain_deg_s2_per_torque: float
  rack_damping_per_s: float
  rack_rate_resolution_deg_s: float
  provenance: str

  def __post_init__(self) -> None:
    gain = float(self.rack_gain_deg_s2_per_torque)
    damping = float(self.rack_damping_per_s)
    resolution = float(self.rack_rate_resolution_deg_s)
    provenance = str(self.provenance).strip()
    if (
      not math.isfinite(gain)
      or gain <= 0.0
      or not math.isfinite(damping)
      or damping < 0.0
      or not math.isfinite(resolution)
      or resolution < 0.0
      or not provenance
    ):
      _fail(
        RuntimeVehicleCompatibility.INVALID_PROVISIONAL_DYNAMICS,
        "provisional rack dynamics and provenance must be explicit and valid",
      )
    object.__setattr__(
      self,
      "rack_gain_deg_s2_per_torque",
      gain,
    )
    object.__setattr__(self, "rack_damping_per_s", damping)
    object.__setattr__(
      self,
      "rack_rate_resolution_deg_s",
      resolution,
    )
    object.__setattr__(self, "provenance", provenance)

  @classmethod
  def from_json_file(
    cls,
    path: str | Path,
  ) -> ProvisionalRackDynamics:
    """Load an explicit, schema-pinned shadow/training seed."""
    try:
      payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
      raise RuntimeVehicleCompatibilityError(
        RuntimeVehicleCompatibility.INVALID_PROVISIONAL_DYNAMICS,
        "provisional rack dynamics file is unreadable or malformed",
      ) from exc
    expected_keys = {
      "schema_version",
      "rack_gain_deg_s2_per_torque",
      "rack_damping_per_s",
      "rack_rate_resolution_deg_s",
      "provenance",
      "provisional",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
      _fail(
        RuntimeVehicleCompatibility.INVALID_PROVISIONAL_DYNAMICS,
        "provisional rack dynamics schema keys do not match",
      )
    if (
      type(payload["schema_version"]) is not int
      or payload["schema_version"]
      != PROVISIONAL_RACK_DYNAMICS_SCHEMA_VERSION
      or payload["provisional"] is not True
    ):
      _fail(
        RuntimeVehicleCompatibility.INVALID_PROVISIONAL_DYNAMICS,
        "rack dynamics must use the current explicitly provisional schema",
      )
    try:
      return cls(
        rack_gain_deg_s2_per_torque=float(
          payload["rack_gain_deg_s2_per_torque"],
        ),
        rack_damping_per_s=float(payload["rack_damping_per_s"]),
        rack_rate_resolution_deg_s=float(
          payload["rack_rate_resolution_deg_s"],
        ),
        provenance=payload["provenance"],
      )
    except (TypeError, ValueError, OverflowError) as exc:
      if isinstance(exc, RuntimeVehicleCompatibilityError):
        raise
      raise RuntimeVehicleCompatibilityError(
        RuntimeVehicleCompatibility.INVALID_PROVISIONAL_DYNAMICS,
        "provisional rack dynamics values are invalid",
      ) from exc


@dataclass(frozen=True, slots=True)
class RuntimeVehicleBundle:
  """Immutable detected-vehicle construction artifact."""

  vehicle_identity: str
  car_fingerprint: str
  provisional_rack_provenance: str
  torque_limits: RuntimeTorqueLimits
  nominal_rack_mapping: RackMappingSnapshot
  seed_profile: VehicleProfile
  torque_callback_slope: float
  torque_callback_max_abs_residual: float
  torque_callback_representation_tolerance: float
  compatibility: RuntimeVehicleCompatibility = (
    RuntimeVehicleCompatibility.SUPPORTED
  )
  schema_version: int = RUNTIME_VEHICLE_SCHEMA_VERSION

  def to_dict(self) -> dict[str, Any]:
    limits = self.torque_limits
    mapping = self.nominal_rack_mapping
    return {
      "car_fingerprint": self.car_fingerprint,
      "compatibility": self.compatibility.value,
      "nominal_rack_mapping": {
        "angle_offset_deg": mapping.angle_offset_deg,
        "center_to_front_m": mapping.center_to_front_m,
        "center_to_rear_m": mapping.center_to_rear_m,
        "mass_kg": mapping.mass_kg,
        "roll_rad": mapping.roll_rad,
        "steer_ratio": mapping.steer_ratio,
        "steer_ratio_rear": mapping.steer_ratio_rear,
        "tire_stiffness_front": mapping.tire_stiffness_front,
        "tire_stiffness_rear": mapping.tire_stiffness_rear,
        "valid": mapping.valid,
        "wheelbase_m": mapping.wheelbase_m,
      },
      "provisional_rack_provenance": self.provisional_rack_provenance,
      "schema_version": self.schema_version,
      "seed_profile": self.seed_profile.to_dict(),
      "torque_callback_max_abs_residual": (
        self.torque_callback_max_abs_residual
      ),
      "torque_callback_representation_tolerance": (
        self.torque_callback_representation_tolerance
      ),
      "torque_callback_slope": self.torque_callback_slope,
      "torque_limits": {
        "delta_down": limits.delta_down,
        "delta_up": limits.delta_up,
        "driver_allowance": limits.driver_allowance,
        "driver_factor": limits.driver_factor,
        "driver_multiplier": limits.driver_multiplier,
        "production_envelope_verified": (
          limits.production_envelope_verified
        ),
        "steer_max": limits.steer_max,
        "steer_step": limits.steer_step,
      },
      "vehicle_identity": self.vehicle_identity,
    }

  def to_json(self) -> str:
    return json.dumps(
      self.to_dict(),
      sort_keys=True,
      separators=(",", ":"),
    )

  @property
  def identity_sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _VerifiedTorqueMapping:
  slope: float
  max_abs_residual: float
  representation_tolerance: float


def _nonempty_text(
  value: object,
  name: str,
  status: RuntimeVehicleCompatibility,
) -> str:
  if type(value) is not str or not value.strip():
    _fail(status, f"{name} must be a nonempty string")
  return value.strip()


def _validate_integer_limit(value: object, name: str) -> None:
  if isinstance(value, bool):
    _fail(
      RuntimeVehicleCompatibility.INVALID_CONTROLLER_LIMITS,
      f"{name} must be an integer",
    )
  try:
    converted = int(value)
  except (TypeError, ValueError, OverflowError):
    _fail(
      RuntimeVehicleCompatibility.INVALID_CONTROLLER_LIMITS,
      f"{name} must be an integer",
    )
  if converted != value:
    _fail(
      RuntimeVehicleCompatibility.INVALID_CONTROLLER_LIMITS,
      f"{name} must be an exact integer",
    )


def _runtime_limits(controller_params: object) -> RuntimeTorqueLimits:
  field_names = (
    "STEER_MAX",
    "STEER_DELTA_UP",
    "STEER_DELTA_DOWN",
    "STEER_STEP",
    "STEER_DRIVER_ALLOWANCE",
    "STEER_DRIVER_MULTIPLIER",
    "STEER_DRIVER_FACTOR",
  )
  for name in field_names:
    if not hasattr(controller_params, name):
      _fail(
        RuntimeVehicleCompatibility.INVALID_CONTROLLER_LIMITS,
        f"controller params are missing {name}",
      )
    _validate_integer_limit(getattr(controller_params, name), name)
  try:
    limits = RuntimeTorqueLimits.from_controller_params(controller_params)
  except (TypeError, ValueError, OverflowError) as exc:
    raise RuntimeVehicleCompatibilityError(
      RuntimeVehicleCompatibility.INVALID_CONTROLLER_LIMITS,
      "detected controller limits are outside their valid domain",
    ) from exc
  if limits.production_envelope_verified and limits.steer_step != 1:
    _fail(
      RuntimeVehicleCompatibility.INVALID_CONTROLLER_LIMITS,
      "verified production envelope must execute at the 100 Hz control cadence",
    )
  return limits


def _rack_rate_resolution(
  controller_params: object,
  provisional_rack_dynamics: ProvisionalRackDynamics,
  production_envelope_verified: bool,
) -> tuple[float, str]:
  """Resolve a vehicle-owned sensor quantum, or retain a shadow-only seed.

  Actuation-eligible platforms must publish this measurement fact from
  opendbc. An unverified platform may collect passive data with the explicit
  provisional seed. Adding a capability later changes the runtime identity,
  so an artifact learned under the provisional value cannot silently actuate.
  """
  value = getattr(
    controller_params,
    "BLATV2_RACK_RATE_RESOLUTION_DEG_S",
    None,
  )
  if value is None:
    if production_envelope_verified:
      _fail(
        RuntimeVehicleCompatibility.MISSING_RACK_RATE_RESOLUTION,
        "verified production envelope lacks a vehicle rack-rate resolution",
      )
    return (
      provisional_rack_dynamics.rack_rate_resolution_deg_s,
      f"provisional:{provisional_rack_dynamics.provenance}",
    )
  if type(value) not in (int, float):
    _fail(
      RuntimeVehicleCompatibility.MISSING_RACK_RATE_RESOLUTION,
      "vehicle rack-rate resolution must be numeric",
    )
  resolution = float(value)
  if not math.isfinite(resolution) or resolution < 0.0:
    _fail(
      RuntimeVehicleCompatibility.MISSING_RACK_RATE_RESOLUTION,
      "vehicle rack-rate resolution must be finite and non-negative",
    )
  return resolution, "detected-opendbc"


def _resolve_torque_callback(
  car_interface_or_callback: object,
) -> Callable[[float, Any], float]:
  if callable(car_interface_or_callback):
    return car_interface_or_callback
  callback_factory = getattr(
    car_interface_or_callback,
    "torque_from_lateral_accel",
    None,
  )
  if not callable(callback_factory):
    _fail(
      RuntimeVehicleCompatibility.MISSING_TORQUE_CALLBACK,
      "car interface does not expose a torque conversion callback",
    )
  try:
    callback = callback_factory()
  except Exception as exc:
    raise RuntimeVehicleCompatibilityError(
      RuntimeVehicleCompatibility.MISSING_TORQUE_CALLBACK,
      "car interface could not construct its torque conversion callback",
    ) from exc
  if not callable(callback):
    _fail(
      RuntimeVehicleCompatibility.MISSING_TORQUE_CALLBACK,
      "car interface returned a non-callable torque conversion",
    )
  return callback


def _verify_linear_torque_mapping(
  callback: Callable[[float, Any], float],
  torque_tuning: Any,
  maximum_lateral_accel: float,
) -> _VerifiedTorqueMapping:
  span = float(maximum_lateral_accel)
  if not math.isfinite(span) or span <= 0.0:
    _fail(
      RuntimeVehicleCompatibility.INVALID_VEHICLE_CALIBRATION,
      "maxLateralAccel must be finite and positive",
    )
  half_span = span / 2.0
  samples = (-span, -half_span, 0.0, half_span, span)
  outputs = []
  try:
    for sample in samples:
      output = float(callback(sample, torque_tuning))
      if not math.isfinite(output):
        raise ValueError("non-finite callback output")
      outputs.append(output)
  except Exception as exc:
    raise RuntimeVehicleCompatibilityError(
      RuntimeVehicleCompatibility.INCOMPATIBLE_TORQUE_CALLBACK,
      "torque conversion callback failed its symmetric probe",
    ) from exc

  denominator = sum(sample * sample for sample in samples)
  slope = sum(
    sample * output
    for sample, output in zip(samples, outputs, strict=True)
  ) / denominator
  if not math.isfinite(slope) or slope <= 0.0:
    _fail(
      RuntimeVehicleCompatibility.INCOMPATIBLE_TORQUE_CALLBACK,
      "torque conversion must have a finite positive scalar slope",
    )
  residuals = tuple(
    output - slope * sample
    for sample, output in zip(samples, outputs, strict=True)
  )
  max_abs_residual = max(abs(residual) for residual in residuals)
  representation_scale = max(
    1.0,
    max(abs(output) for output in outputs),
    abs(slope) * span,
  )
  representation_tolerance = (
    _CALLBACK_REPRESENTATION_EPSILON_MULTIPLIER
    * sys.float_info.epsilon
    * representation_scale
  )
  if max_abs_residual > representation_tolerance:
    _fail(
      RuntimeVehicleCompatibility.INCOMPATIBLE_TORQUE_CALLBACK,
      "torque conversion is nonlinear, asymmetric, or offset and cannot " +
      "be represented by the scalar profile schema",
    )
  return _VerifiedTorqueMapping(
    slope=slope,
    max_abs_residual=max_abs_residual,
    representation_tolerance=representation_tolerance,
  )


def _vehicle_model_snapshot(car_params: CarParams) -> RackMappingSnapshot:
  numeric_fields = (
    "mass",
    "rotationalInertia",
    "wheelbase",
    "centerToFront",
    "steerRatio",
    "steerRatioRear",
    "tireStiffnessFront",
    "tireStiffnessRear",
    "steerActuatorDelay",
  )
  for name in numeric_fields:
    try:
      value = float(getattr(car_params, name))
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
      raise RuntimeVehicleCompatibilityError(
        RuntimeVehicleCompatibility.INVALID_VEHICLE_CALIBRATION,
        f"CarParams {name} is missing or invalid",
      ) from exc
    if not math.isfinite(value):
      _fail(
        RuntimeVehicleCompatibility.INVALID_VEHICLE_CALIBRATION,
        f"CarParams {name} must be finite",
      )
  if float(car_params.rotationalInertia) <= 0.0:
    _fail(
      RuntimeVehicleCompatibility.INVALID_VEHICLE_CALIBRATION,
      "CarParams rotational inertia must be positive",
    )
  try:
    model = VehicleModel(car_params)
    return RackMappingSnapshot.from_vehicle_model(
      model,
      roll_rad=0.0,
      angle_offset_deg=0.0,
      valid=True,
    )
  except (TypeError, ValueError, ZeroDivisionError) as exc:
    raise RuntimeVehicleCompatibilityError(
      RuntimeVehicleCompatibility.INVALID_VEHICLE_CALIBRATION,
      "CarParams geometry cannot construct a valid VehicleModel mapping",
    ) from exc


def build_runtime_vehicle_bundle(
  *,
  car_params: CarParams,
  car_interface_or_callback: object,
  controller_params: object,
  vehicle_identity: str,
  provisional_rack_dynamics: ProvisionalRackDynamics,
) -> RuntimeVehicleBundle:
  """Build a generic runtime bundle or fail closed with a stable reason."""
  if car_params is None:
    _fail(
      RuntimeVehicleCompatibility.INVALID_VEHICLE_CALIBRATION,
      "runtime adaptation requires detected CarParams",
    )
  identity = _nonempty_text(
    vehicle_identity,
    "vehicle identity",
    RuntimeVehicleCompatibility.INVALID_IDENTITY,
  )
  if not isinstance(provisional_rack_dynamics, ProvisionalRackDynamics):
    _fail(
      RuntimeVehicleCompatibility.INVALID_PROVISIONAL_DYNAMICS,
      "provisional rack dynamics must be supplied explicitly",
    )

  try:
    steer_control_type = car_params.steerControlType
  except AttributeError as exc:
    raise RuntimeVehicleCompatibilityError(
      RuntimeVehicleCompatibility.INVALID_VEHICLE_CALIBRATION,
      "CarParams steer control type is unavailable",
    ) from exc
  if steer_control_type != CarParams.SteerControlType.torque:
    _fail(
      RuntimeVehicleCompatibility.UNSUPPORTED_STEER_CONTROL,
      "only torque-control vehicles are supported by this profile schema",
    )
  try:
    lateral_tuning_kind = car_params.lateralTuning.which()
  except Exception as exc:
    raise RuntimeVehicleCompatibilityError(
      RuntimeVehicleCompatibility.UNSUPPORTED_LATERAL_TUNING,
      "CarParams lateral tuning is unavailable",
    ) from exc
  if lateral_tuning_kind != "torque":
    _fail(
      RuntimeVehicleCompatibility.UNSUPPORTED_LATERAL_TUNING,
      "only lateral torque tuning is supported by this profile schema",
    )

  car_fingerprint = _nonempty_text(
    car_params.carFingerprint,
    "car fingerprint",
    RuntimeVehicleCompatibility.INVALID_VEHICLE_CALIBRATION,
  )
  limits = _runtime_limits(controller_params)
  rack_rate_resolution, rack_resolution_source = _rack_rate_resolution(
    controller_params,
    provisional_rack_dynamics,
    limits.production_envelope_verified,
  )
  nominal_mapping = _vehicle_model_snapshot(car_params)
  delay = float(car_params.steerActuatorDelay)
  if delay < 0.0:
    _fail(
      RuntimeVehicleCompatibility.INVALID_VEHICLE_CALIBRATION,
      "steering actuator delay must be non-negative",
    )

  torque_tuning = car_params.lateralTuning.torque
  friction = float(torque_tuning.friction)
  if not math.isfinite(friction) or friction < 0.0:
    _fail(
      RuntimeVehicleCompatibility.INVALID_VEHICLE_CALIBRATION,
      "stock normalized friction must be finite and non-negative",
    )
  callback = _resolve_torque_callback(car_interface_or_callback)
  verified_mapping = _verify_linear_torque_mapping(
    callback,
    torque_tuning,
    float(car_params.maxLateralAccel),
  )

  base_seed = make_seed_profile(
    vehicle_identity=identity,
    torque_per_lateral_accel=verified_mapping.slope,
    rack_gain_deg_s2_per_torque=(
      provisional_rack_dynamics.rack_gain_deg_s2_per_torque
    ),
    rack_damping_per_s=provisional_rack_dynamics.rack_damping_per_s,
    transport_delay_s=delay,
    static_friction_torque=friction,
    kinetic_friction_torque=friction,
    rack_rate_resolution_deg_s=rack_rate_resolution,
  )
  seed_profile = VehicleProfile(
    vehicle_identity=base_seed.vehicle_identity,
    revision=base_seed.revision,
    provenance=(
      "unqualified detected-opendbc runtime seed; " +
      f"car_fingerprint={car_fingerprint}; " +
      f"provisional_rack={provisional_rack_dynamics.provenance}; " +
      f"rack_rate_resolution_source={rack_resolution_source}"
    ),
    nodes=base_seed.nodes,
    schema_version=base_seed.schema_version,
  )
  return RuntimeVehicleBundle(
    vehicle_identity=identity,
    car_fingerprint=car_fingerprint,
    provisional_rack_provenance=(
      provisional_rack_dynamics.provenance
    ),
    torque_limits=limits,
    nominal_rack_mapping=nominal_mapping,
    seed_profile=seed_profile,
    torque_callback_slope=verified_mapping.slope,
    torque_callback_max_abs_residual=(
      verified_mapping.max_abs_residual
    ),
    torque_callback_representation_tolerance=(
      verified_mapping.representation_tolerance
    ),
  )
