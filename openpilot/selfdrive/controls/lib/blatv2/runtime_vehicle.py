"""Construction-time adaptation from detected opendbc vehicle facts.

This module is intentionally platform-neutral. It copies the detected
actuator envelope, verifies that the car interface's production torque
conversion fits the scalar-linear profile schema, snapshots the detected
VehicleModel, and builds an unqualified physical seed. Unsupported mappings
fail closed so stock control remains the only eligible controller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
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
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  VehicleCalibrationProfile,
  make_calibration_seed_profile,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  DEFAULT_SPEED_NODES_MPS,
  VehicleProfile,
  make_seed_profile,
)


RUNTIME_VEHICLE_SCHEMA_VERSION = 1
CALIBRATION_RUNTIME_IDENTITY_SCHEMA_VERSION = 1
PROVISIONAL_RACK_DYNAMICS_SCHEMA_VERSION = 2
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
class RackDynamicsNode:
  speed_mps: float
  rack_gain_deg_s2_per_torque: float
  rack_damping_per_s: float

  def __post_init__(self) -> None:
    speed = float(self.speed_mps)
    gain = float(self.rack_gain_deg_s2_per_torque)
    damping = float(self.rack_damping_per_s)
    if (
      not math.isfinite(speed)
      or speed < 0.0
      or not math.isfinite(gain)
      or gain <= 0.0
      or not math.isfinite(damping)
      or damping < 0.0
    ):
      _fail(
        RuntimeVehicleCompatibility.INVALID_PROVISIONAL_DYNAMICS,
        "rack dynamics node must be finite and physical",
      )
    object.__setattr__(self, "speed_mps", speed)
    object.__setattr__(self, "rack_gain_deg_s2_per_torque", gain)
    object.__setattr__(self, "rack_damping_per_s", damping)

  def to_dict(self) -> dict[str, float]:
    return {
      "rack_damping_per_s": self.rack_damping_per_s,
      "rack_gain_deg_s2_per_torque": self.rack_gain_deg_s2_per_torque,
      "speed_mps": self.speed_mps,
    }


@dataclass(frozen=True, slots=True)
class ProvisionalRackDynamics:
  """Explicit unqualified dynamics for study and authorized field trials."""

  rack_gain_deg_s2_per_torque: float
  rack_damping_per_s: float
  rack_rate_resolution_deg_s: float
  provenance: str
  nodes: tuple[RackDynamicsNode, ...] = ()

  def __post_init__(self) -> None:
    gain = float(self.rack_gain_deg_s2_per_torque)
    damping = float(self.rack_damping_per_s)
    resolution = float(self.rack_rate_resolution_deg_s)
    provenance = str(self.provenance).strip()
    nodes = self.nodes
    if (
      not nodes
      or (
        type(nodes) is tuple
        and nodes
        and type(nodes[0]) is RackDynamicsNode
        and (
          nodes[0].rack_gain_deg_s2_per_torque != gain
          or nodes[0].rack_damping_per_s != damping
        )
      )
    ):
      # Preserve the legacy scalar constructor and dataclasses.replace
      # semantics: changing either scalar intentionally requests a uniform
      # schedule. Schema-2 files derive both scalars from their first node.
      nodes = tuple(
        RackDynamicsNode(speed, gain, damping)
        for speed in DEFAULT_SPEED_NODES_MPS
      )
    if (
      not math.isfinite(gain)
      or gain <= 0.0
      or not math.isfinite(damping)
      or damping < 0.0
      or not math.isfinite(resolution)
      or resolution < 0.0
      or not provenance
      or type(nodes) is not tuple
      or any(type(node) is not RackDynamicsNode for node in nodes)
      or tuple(node.speed_mps for node in nodes) != DEFAULT_SPEED_NODES_MPS
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
    object.__setattr__(self, "nodes", nodes)

  def parameters_at_speed(self, speed_mps: float) -> tuple[float, float]:
    speed = float(speed_mps)
    if not math.isfinite(speed) or speed < 0.0:
      _fail(
        RuntimeVehicleCompatibility.INVALID_PROVISIONAL_DYNAMICS,
        "rack dynamics interpolation speed must be finite and non-negative",
      )
    if speed <= self.nodes[0].speed_mps:
      node = self.nodes[0]
      return node.rack_gain_deg_s2_per_torque, node.rack_damping_per_s
    for lower, upper in zip(self.nodes, self.nodes[1:], strict=True):
      if speed <= upper.speed_mps:
        weight = (speed - lower.speed_mps) / (upper.speed_mps - lower.speed_mps)
        gain = lower.rack_gain_deg_s2_per_torque + weight * (
          upper.rack_gain_deg_s2_per_torque
          - lower.rack_gain_deg_s2_per_torque
        )
        damping = lower.rack_damping_per_s + weight * (
          upper.rack_damping_per_s - lower.rack_damping_per_s
        )
        return gain, damping
    node = self.nodes[-1]
    return node.rack_gain_deg_s2_per_torque, node.rack_damping_per_s

  def to_dict(self) -> dict[str, Any]:
    return {
      "provenance": self.provenance,
      "provisional": True,
      "nodes": [node.to_dict() for node in self.nodes],
      "rack_rate_resolution_deg_s": self.rack_rate_resolution_deg_s,
      "schema_version": PROVISIONAL_RACK_DYNAMICS_SCHEMA_VERSION,
    }

  def to_json(self) -> str:
    return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

  @property
  def identity_sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

  @classmethod
  def from_json_file(
    cls,
    path: str | Path,
  ) -> ProvisionalRackDynamics:
    """Load an explicit, schema-pinned provisional seed."""
    try:
      payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
      raise RuntimeVehicleCompatibilityError(
        RuntimeVehicleCompatibility.INVALID_PROVISIONAL_DYNAMICS,
        "provisional rack dynamics file is unreadable or malformed",
      ) from exc
    expected_keys = {
      "schema_version",
      "nodes",
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
      raw_nodes = payload["nodes"]
      if type(raw_nodes) is not list:
        raise TypeError("rack dynamics nodes must be a list")
      nodes = tuple(
        RackDynamicsNode(
          speed_mps=float(raw_node["speed_mps"]),
          rack_gain_deg_s2_per_torque=float(
            raw_node["rack_gain_deg_s2_per_torque"],
          ),
          rack_damping_per_s=float(raw_node["rack_damping_per_s"]),
        )
        for raw_node in raw_nodes
        if type(raw_node) is dict and set(raw_node) == {
          "speed_mps",
          "rack_gain_deg_s2_per_torque",
          "rack_damping_per_s",
        }
      )
      if len(nodes) != len(raw_nodes) or not nodes:
        raise ValueError("rack dynamics nodes are malformed")
      return cls(
        rack_gain_deg_s2_per_torque=nodes[0].rack_gain_deg_s2_per_torque,
        rack_damping_per_s=nodes[0].rack_damping_per_s,
        rack_rate_resolution_deg_s=float(
          payload["rack_rate_resolution_deg_s"],
        ),
        provenance=payload["provenance"],
        nodes=nodes,
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
  calibration_seed_profile: VehicleCalibrationProfile
  seed_profile: VehicleProfile
  stock_lateral_accel_offset_mps2: float
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

  def calibration_identity_dict(self) -> dict[str, Any]:
    """Return only facts that can change observable calibration evidence.

    The retired rack-gain/damping seed remains in ``to_dict`` for legacy
    controller-artifact compatibility, but it is deliberately absent here.
    Changing an unidentifiable provisional rack model must not invalidate or
    fork an otherwise identical inverse-torque evidence set.
    """
    payload = self.to_dict()
    return {
      "calibration_identity_schema_version": (
        CALIBRATION_RUNTIME_IDENTITY_SCHEMA_VERSION
      ),
      "calibration_seed_profile": (
        self.calibration_seed_profile.to_dict()
      ),
      "car_fingerprint": payload["car_fingerprint"],
      "nominal_rack_mapping": payload["nominal_rack_mapping"],
      "stock_lateral_accel_offset_mps2": (
        self.stock_lateral_accel_offset_mps2
      ),
      "torque_callback_max_abs_residual": (
        payload["torque_callback_max_abs_residual"]
      ),
      "torque_callback_representation_tolerance": (
        payload["torque_callback_representation_tolerance"]
      ),
      "torque_callback_slope": payload["torque_callback_slope"],
      "torque_limits": payload["torque_limits"],
      "vehicle_identity": payload["vehicle_identity"],
    }

  @property
  def calibration_identity_sha256(self) -> str:
    encoded = json.dumps(
      self.calibration_identity_dict(),
      sort_keys=True,
      separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

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
  stock_lateral_accel_offset = float(torque_tuning.latAccelOffset)
  if not math.isfinite(friction) or friction < 0.0:
    _fail(
      RuntimeVehicleCompatibility.INVALID_VEHICLE_CALIBRATION,
      "stock normalized friction must be finite and non-negative",
    )
  if not math.isfinite(stock_lateral_accel_offset):
    _fail(
      RuntimeVehicleCompatibility.INVALID_VEHICLE_CALIBRATION,
      "stock lateral-acceleration offset must be finite",
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
  dynamics_by_speed = {
    node.speed_mps: provisional_rack_dynamics.parameters_at_speed(
      node.speed_mps,
    )
    for node in base_seed.nodes
  }
  base_seed = replace(
    base_seed,
    nodes=tuple(
      replace(
        node,
        parameters=replace(
          node.parameters,
          rack_gain_deg_s2_per_torque=dynamics_by_speed[node.speed_mps][0],
          rack_damping_per_s=dynamics_by_speed[node.speed_mps][1],
        ),
      )
      for node in base_seed.nodes
    ),
  )
  calibration_seed_profile = make_calibration_seed_profile(
    vehicle_identity=identity,
    torque_callback_slope=verified_mapping.slope,
    # opendbc currently stores this value in normalized-torque space. Its
    # stock friction helper temporarily converts it to lateral acceleration,
    # and the torque callback converts it back before actuation.
    stock_friction_torque=friction,
    transport_delay_s=delay,
    rack_rate_resolution_deg_s=rack_rate_resolution,
  )
  calibration_rack_resolution_source = (
    "provisional"
    if rack_resolution_source.startswith("provisional:")
    else rack_resolution_source
  )
  calibration_seed_profile = VehicleCalibrationProfile(
    vehicle_identity=calibration_seed_profile.vehicle_identity,
    revision=calibration_seed_profile.revision,
    provenance="; ".join((
      "unqualified observable detected-opendbc calibration seed",
      f"car_fingerprint={car_fingerprint}",
      f"rack_rate_resolution_source={calibration_rack_resolution_source}",
    )),
    nodes=calibration_seed_profile.nodes,
    schema_version=calibration_seed_profile.schema_version,
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
    calibration_seed_profile=calibration_seed_profile,
    seed_profile=seed_profile,
    stock_lateral_accel_offset_mps2=stock_lateral_accel_offset,
    torque_callback_slope=verified_mapping.slope,
    torque_callback_max_abs_residual=(
      verified_mapping.max_abs_residual
    ),
    torque_callback_representation_tolerance=(
      verified_mapping.representation_tolerance
    ),
  )
