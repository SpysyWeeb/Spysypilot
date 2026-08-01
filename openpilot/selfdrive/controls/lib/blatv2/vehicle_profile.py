"""Versioned, speed-local physical profile for modular BLaTv2.

This module owns every speed-dependent physical value used by the controller.
It deliberately does not contain turn, release, unwind, or other maneuver
gates. Values are linearly interpolated between nodes and held flat outside
the node range, so crossing a node never changes controller mode.

Profiles are immutable snapshots. Learning produces a new snapshot offroad;
the live controller never mutates one in place. The schema and parameter key
are intentionally incompatible with every earlier BLaTv2 adaptive profile.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  VehicleCalibrationProfile,
)


PROFILE_SCHEMA_VERSION = 2
PROFILE_PARAM_KEY = "BLaTv2ModularVehicleProfile"
DEFAULT_SPEED_NODES_MPS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)

_PROFILE_KEYS = frozenset({
  "nodes",
  "provenance",
  "revision",
  "schema_version",
  "vehicle_identity",
})
_NODE_KEYS = frozenset({
  "clean_support_s",
  "parameters",
  "sample_count",
  "speed_mps",
  "validation_count",
  "validation_rms",
})
_PARAMETER_KEYS = frozenset({
  "confidence",
  "kinetic_friction_torque",
  "lateral_accel_offset_correction_mps2",
  "qualified",
  "rack_damping_per_s",
  "rack_gain_deg_s2_per_torque",
  "rack_rate_resolution_deg_s",
  "static_friction_torque",
  "torque_per_lateral_accel",
  "transport_delay_s",
})


def _require_exact_keys(
  value: object,
  expected: frozenset[str],
  context: str,
) -> dict[str, Any]:
  if type(value) is not dict:
    raise ValueError(f"{context} must be an object")
  keys = frozenset(value)
  if keys != expected:
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    raise ValueError(
      f"{context} keys are incompatible: missing={missing}, unknown={unknown}",
    )
  return value


def _require_int(value: object, context: str) -> int:
  if type(value) is not int:
    raise ValueError(f"{context} must be an integer")
  return value


def _require_float(value: object, context: str) -> float:
  if type(value) not in (int, float):
    raise ValueError(f"{context} must be numeric")
  numeric = float(value)
  if not math.isfinite(numeric):
    raise ValueError(f"{context} must be finite")
  return numeric


def _require_bool(value: object, context: str) -> bool:
  if type(value) is not bool:
    raise ValueError(f"{context} must be a boolean")
  return value


@dataclass(frozen=True, slots=True)
class PhysicalParameters:
  """Physical plant values evaluated at one speed."""

  torque_per_lateral_accel: float
  rack_gain_deg_s2_per_torque: float
  rack_damping_per_s: float
  transport_delay_s: float
  static_friction_torque: float
  kinetic_friction_torque: float
  rack_rate_resolution_deg_s: float
  confidence: float
  qualified: bool
  lateral_accel_offset_correction_mps2: float = 0.0

  def __post_init__(self) -> None:
    if type(self.qualified) is not bool:
      raise ValueError("qualified must be a boolean")
    values = (
      self.torque_per_lateral_accel,
      self.rack_gain_deg_s2_per_torque,
      self.rack_damping_per_s,
      self.transport_delay_s,
      self.static_friction_torque,
      self.kinetic_friction_torque,
      self.rack_rate_resolution_deg_s,
      self.confidence,
      self.lateral_accel_offset_correction_mps2,
    )
    if not all(math.isfinite(value) for value in values):
      raise ValueError("physical parameters must be finite")
    if (
      self.torque_per_lateral_accel <= 0.0
      or self.rack_gain_deg_s2_per_torque <= 0.0
      or self.rack_damping_per_s < 0.0
      or self.transport_delay_s < 0.0
      or self.static_friction_torque < 0.0
      or self.kinetic_friction_torque < 0.0
      or self.rack_rate_resolution_deg_s < 0.0
      or not 0.0 <= self.confidence <= 1.0
    ):
      raise ValueError("physical parameters are outside their valid domain")
    if self.kinetic_friction_torque > self.static_friction_torque:
      raise ValueError("kinetic friction cannot exceed static breakaway")


@dataclass(frozen=True, slots=True)
class ProfileNode:
  """One independently learned speed node and its evidence."""

  speed_mps: float
  parameters: PhysicalParameters
  clean_support_s: float
  sample_count: int
  validation_count: int
  validation_rms: float

  def __post_init__(self) -> None:
    if not isinstance(self.parameters, PhysicalParameters):
      raise ValueError("profile-node parameters must be PhysicalParameters")
    if type(self.sample_count) is not int or type(self.validation_count) is not int:
      raise ValueError("profile-node counts must be integers")
    if not all(math.isfinite(value) for value in (
      self.speed_mps,
      self.clean_support_s,
      self.validation_rms,
    )):
      raise ValueError("profile-node values must be finite")
    if (
      self.speed_mps < 0.0
      or self.clean_support_s < 0.0
      or self.sample_count < 0
      or self.validation_count < 0
      or self.validation_rms < 0.0
    ):
      raise ValueError("profile-node evidence is outside its valid domain")


@dataclass(frozen=True, slots=True)
class InterpolatedProfile:
  """Continuous live parameters plus transparent node provenance."""

  parameters: PhysicalParameters
  lower_node: int
  upper_node: int
  upper_weight: float


@dataclass(frozen=True, slots=True)
class VehicleProfile:
  """Atomic vehicle-specific physical profile."""

  vehicle_identity: str
  revision: int
  provenance: str
  nodes: tuple[ProfileNode, ...]
  schema_version: int = PROFILE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != PROFILE_SCHEMA_VERSION:
      raise ValueError("vehicle profile schema is incompatible")
    if type(self.schema_version) is not int:
      raise ValueError("vehicle profile schema must be an integer")
    if type(self.vehicle_identity) is not str or not self.vehicle_identity:
      raise ValueError("vehicle identity must not be empty")
    if type(self.revision) is not int or self.revision < 0:
      raise ValueError("profile revision must be non-negative")
    if type(self.provenance) is not str or not self.provenance:
      raise ValueError("profile provenance must not be empty")
    if type(self.nodes) is not tuple or len(self.nodes) < 2:
      raise ValueError("vehicle profile requires at least two speed nodes")
    if not all(isinstance(node, ProfileNode) for node in self.nodes):
      raise ValueError("vehicle profile nodes have an incompatible type")
    if any(
      right.speed_mps <= left.speed_mps
      for left, right in zip(self.nodes, self.nodes[1:], strict=False)
    ):
      raise ValueError("profile speed nodes must be strictly increasing")

  @property
  def qualified(self) -> bool:
    """A profile is eligible only when every speed region is qualified."""
    return all(node.parameters.qualified for node in self.nodes)

  @property
  def speed_nodes_mps(self) -> tuple[float, ...]:
    return tuple(node.speed_mps for node in self.nodes)

  def parameters_at(self, v_ego: float) -> InterpolatedProfile:
    """Return continuous physical parameters with flat extrapolation."""
    speed = abs(float(v_ego))
    if not math.isfinite(speed):
      raise ValueError("vehicle speed must be finite")

    if speed <= self.nodes[0].speed_mps:
      return InterpolatedProfile(self.nodes[0].parameters, 0, 0, 0.0)
    if speed >= self.nodes[-1].speed_mps:
      last = len(self.nodes) - 1
      return InterpolatedProfile(self.nodes[last].parameters, last, last, 0.0)

    upper = 1
    while self.nodes[upper].speed_mps < speed:
      upper += 1
    lower = upper - 1
    lower_node = self.nodes[lower]
    upper_node = self.nodes[upper]
    weight = (
      (speed - lower_node.speed_mps)
      / (upper_node.speed_mps - lower_node.speed_mps)
    )
    lower_params = lower_node.parameters
    upper_params = upper_node.parameters

    def blend(lower_value: float, upper_value: float) -> float:
      return lower_value + weight * (upper_value - lower_value)

    parameters = PhysicalParameters(
      torque_per_lateral_accel=blend(
        lower_params.torque_per_lateral_accel,
        upper_params.torque_per_lateral_accel,
      ),
      rack_gain_deg_s2_per_torque=blend(
        lower_params.rack_gain_deg_s2_per_torque,
        upper_params.rack_gain_deg_s2_per_torque,
      ),
      rack_damping_per_s=blend(
        lower_params.rack_damping_per_s,
        upper_params.rack_damping_per_s,
      ),
      transport_delay_s=blend(
        lower_params.transport_delay_s,
        upper_params.transport_delay_s,
      ),
      static_friction_torque=blend(
        lower_params.static_friction_torque,
        upper_params.static_friction_torque,
      ),
      kinetic_friction_torque=blend(
        lower_params.kinetic_friction_torque,
        upper_params.kinetic_friction_torque,
      ),
      rack_rate_resolution_deg_s=blend(
        lower_params.rack_rate_resolution_deg_s,
        upper_params.rack_rate_resolution_deg_s,
      ),
      confidence=blend(lower_params.confidence, upper_params.confidence),
      qualified=lower_params.qualified and upper_params.qualified,
      lateral_accel_offset_correction_mps2=blend(
        lower_params.lateral_accel_offset_correction_mps2,
        upper_params.lateral_accel_offset_correction_mps2,
      ),
    )
    return InterpolatedProfile(parameters, lower, upper, weight)

  def to_dict(self) -> dict[str, Any]:
    return {
      "schema_version": self.schema_version,
      "vehicle_identity": self.vehicle_identity,
      "revision": self.revision,
      "provenance": self.provenance,
      "nodes": [
        {
          "speed_mps": node.speed_mps,
          "parameters": {
            "torque_per_lateral_accel": node.parameters.torque_per_lateral_accel,
            "rack_gain_deg_s2_per_torque": node.parameters.rack_gain_deg_s2_per_torque,
            "rack_damping_per_s": node.parameters.rack_damping_per_s,
            "transport_delay_s": node.parameters.transport_delay_s,
            "static_friction_torque": node.parameters.static_friction_torque,
            "kinetic_friction_torque": node.parameters.kinetic_friction_torque,
            "rack_rate_resolution_deg_s": node.parameters.rack_rate_resolution_deg_s,
            "confidence": node.parameters.confidence,
            "qualified": node.parameters.qualified,
            "lateral_accel_offset_correction_mps2": (
              node.parameters.lateral_accel_offset_correction_mps2
            ),
          },
          "clean_support_s": node.clean_support_s,
          "sample_count": node.sample_count,
          "validation_count": node.validation_count,
          "validation_rms": node.validation_rms,
        }
        for node in self.nodes
      ],
    }

  def to_json(self) -> str:
    return json.dumps(
      self.to_dict(),
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
    )

  @classmethod
  def from_json(
    cls,
    encoded: str | bytes | dict[str, Any],
    expected_vehicle_identity: str,
  ) -> VehicleProfile:
    payload: object = encoded if type(encoded) is dict else json.loads(encoded)
    raw_profile = _require_exact_keys(payload, _PROFILE_KEYS, "vehicle profile")
    schema_version = _require_int(
      raw_profile["schema_version"],
      "vehicle profile schema_version",
    )
    if schema_version != PROFILE_SCHEMA_VERSION:
      raise ValueError("vehicle profile schema is incompatible")
    identity = raw_profile["vehicle_identity"]
    if type(identity) is not str:
      raise ValueError("vehicle identity must be a string")
    if identity != expected_vehicle_identity:
      raise ValueError("vehicle profile belongs to a different vehicle")

    provenance = raw_profile["provenance"]
    if type(provenance) is not str:
      raise ValueError("profile provenance must be a string")
    raw_nodes = raw_profile["nodes"]
    if type(raw_nodes) is not list:
      raise ValueError("vehicle profile nodes must be a list")

    nodes: list[ProfileNode] = []
    for index, raw_node_value in enumerate(raw_nodes):
      raw_node = _require_exact_keys(
        raw_node_value,
        _NODE_KEYS,
        f"vehicle profile node {index}",
      )
      raw_params = _require_exact_keys(
        raw_node["parameters"],
        _PARAMETER_KEYS,
        f"vehicle profile node {index} parameters",
      )
      nodes.append(ProfileNode(
        speed_mps=_require_float(
          raw_node["speed_mps"],
          f"vehicle profile node {index} speed_mps",
        ),
        parameters=PhysicalParameters(
          torque_per_lateral_accel=_require_float(
            raw_params["torque_per_lateral_accel"],
            f"vehicle profile node {index} torque_per_lateral_accel",
          ),
          rack_gain_deg_s2_per_torque=_require_float(
            raw_params["rack_gain_deg_s2_per_torque"],
            f"vehicle profile node {index} rack_gain_deg_s2_per_torque",
          ),
          rack_damping_per_s=_require_float(
            raw_params["rack_damping_per_s"],
            f"vehicle profile node {index} rack_damping_per_s",
          ),
          transport_delay_s=_require_float(
            raw_params["transport_delay_s"],
            f"vehicle profile node {index} transport_delay_s",
          ),
          static_friction_torque=_require_float(
            raw_params["static_friction_torque"],
            f"vehicle profile node {index} static_friction_torque",
          ),
          kinetic_friction_torque=_require_float(
            raw_params["kinetic_friction_torque"],
            f"vehicle profile node {index} kinetic_friction_torque",
          ),
          rack_rate_resolution_deg_s=_require_float(
            raw_params["rack_rate_resolution_deg_s"],
            f"vehicle profile node {index} rack_rate_resolution_deg_s",
          ),
          confidence=_require_float(
            raw_params["confidence"],
            f"vehicle profile node {index} confidence",
          ),
          qualified=_require_bool(
            raw_params["qualified"],
            f"vehicle profile node {index} qualified",
          ),
          lateral_accel_offset_correction_mps2=_require_float(
            raw_params["lateral_accel_offset_correction_mps2"],
            f"vehicle profile node {index} lateral_accel_offset_correction_mps2",
          ),
        ),
        clean_support_s=_require_float(
          raw_node["clean_support_s"],
          f"vehicle profile node {index} clean_support_s",
        ),
        sample_count=_require_int(
          raw_node["sample_count"],
          f"vehicle profile node {index} sample_count",
        ),
        validation_count=_require_int(
          raw_node["validation_count"],
          f"vehicle profile node {index} validation_count",
        ),
        validation_rms=_require_float(
          raw_node["validation_rms"],
          f"vehicle profile node {index} validation_rms",
        ),
      ))
    return cls(
      vehicle_identity=identity,
      revision=_require_int(raw_profile["revision"], "profile revision"),
      provenance=provenance,
      nodes=tuple(nodes),
      schema_version=schema_version,
    )


def make_seed_profile(
  vehicle_identity: str,
  torque_per_lateral_accel: float,
  rack_gain_deg_s2_per_torque: float,
  rack_damping_per_s: float,
  transport_delay_s: float,
  static_friction_torque: float,
  kinetic_friction_torque: float,
  rack_rate_resolution_deg_s: float = 0.0,
  speed_nodes_mps: tuple[float, ...] = DEFAULT_SPEED_NODES_MPS,
) -> VehicleProfile:
  """Create an explicitly unqualified seed for shadow/bootstrap operation."""
  parameters = PhysicalParameters(
    torque_per_lateral_accel=float(torque_per_lateral_accel),
    rack_gain_deg_s2_per_torque=float(rack_gain_deg_s2_per_torque),
    rack_damping_per_s=float(rack_damping_per_s),
    transport_delay_s=float(transport_delay_s),
    static_friction_torque=float(static_friction_torque),
    kinetic_friction_torque=float(kinetic_friction_torque),
    rack_rate_resolution_deg_s=float(rack_rate_resolution_deg_s),
    confidence=0.0,
    qualified=False,
  )
  nodes = tuple(
    ProfileNode(
      speed_mps=float(speed),
      parameters=parameters,
      clean_support_s=0.0,
      sample_count=0,
      validation_count=0,
      validation_rms=0.0,
    )
    for speed in speed_nodes_mps
  )
  return VehicleProfile(
    vehicle_identity=vehicle_identity,
    revision=0,
    provenance="unqualified runtime calibration seed",
    nodes=nodes,
  )


def compose_controller_profile(
  calibration_profile: VehicleCalibrationProfile,
  transient_seed_profile: VehicleProfile,
) -> VehicleProfile:
  """Compose the sole controller profile from observable and transient facts.

  Ordinary driving qualifies only the observable inverse-torque map.  Rack
  gain and damping remain the explicitly provisional runtime seed, but they
  are copied through this one deterministic boundary rather than being
  smuggled into a second learned artifact.  Exact node-grid equality keeps the
  speed interpolation seen by calibration, replay, and live control identical.
  """
  if not isinstance(calibration_profile, VehicleCalibrationProfile):
    raise TypeError("controller composition requires observable calibration")
  if not isinstance(transient_seed_profile, VehicleProfile):
    raise TypeError("controller composition requires a transient seed profile")
  if calibration_profile.vehicle_identity != transient_seed_profile.vehicle_identity:
    raise ValueError("calibration and transient seed belong to different vehicles")
  if calibration_profile.speed_nodes_mps != transient_seed_profile.speed_nodes_mps:
    raise ValueError("calibration and transient seed speed grids differ")

  nodes: list[ProfileNode] = []
  for calibration_node, transient_node in zip(
    calibration_profile.nodes,
    transient_seed_profile.nodes,
    strict=True,
  ):
    observable = calibration_node.parameters
    transient = transient_node.parameters
    nodes.append(ProfileNode(
      speed_mps=calibration_node.speed_mps,
      parameters=PhysicalParameters(
        torque_per_lateral_accel=observable.torque_per_lateral_accel,
        rack_gain_deg_s2_per_torque=transient.rack_gain_deg_s2_per_torque,
        rack_damping_per_s=transient.rack_damping_per_s,
        transport_delay_s=observable.transport_delay_s,
        static_friction_torque=observable.static_breakaway_torque,
        kinetic_friction_torque=observable.kinetic_friction_torque,
        rack_rate_resolution_deg_s=observable.rack_rate_resolution_deg_s,
        confidence=observable.confidence,
        qualified=observable.qualified,
        lateral_accel_offset_correction_mps2=(
          observable.lateral_accel_offset_correction_mps2
        ),
      ),
      clean_support_s=(
        calibration_node.base_support_s
        + calibration_node.moving_support_s
        + calibration_node.breakaway_support_s
      ),
      sample_count=(
        calibration_node.base_sample_count
        + calibration_node.moving_sample_count
        + calibration_node.breakaway_sample_count
      ),
      validation_count=calibration_node.validation_count,
      validation_rms=calibration_node.inverse_calibration_validation_rms,
    ))

  calibration_sha = hashlib.sha256(
    calibration_profile.to_json().encode("utf-8"),
  ).hexdigest()
  transient_sha = hashlib.sha256(
    transient_seed_profile.to_json().encode("utf-8"),
  ).hexdigest()
  return VehicleProfile(
    vehicle_identity=calibration_profile.vehicle_identity,
    revision=calibration_profile.revision,
    provenance="; ".join((
      "deterministic observable/transient composition",
      f"calibration_sha256={calibration_sha}",
      f"transient_seed_sha256={transient_sha}",
    )),
    nodes=tuple(nodes),
  )
