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
import json
import math
from typing import Any


PROFILE_SCHEMA_VERSION = 1
PROFILE_PARAM_KEY = "BLaTv2ModularVehicleProfile"
DEFAULT_SPEED_NODES_MPS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)


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

  def __post_init__(self) -> None:
    values = (
      self.torque_per_lateral_accel,
      self.rack_gain_deg_s2_per_torque,
      self.rack_damping_per_s,
      self.transport_delay_s,
      self.static_friction_torque,
      self.kinetic_friction_torque,
      self.rack_rate_resolution_deg_s,
      self.confidence,
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
    if not self.vehicle_identity:
      raise ValueError("vehicle identity must not be empty")
    if self.revision < 0:
      raise ValueError("profile revision must be non-negative")
    if len(self.nodes) < 2:
      raise ValueError("vehicle profile requires at least two speed nodes")
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
    return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

  @classmethod
  def from_json(
    cls,
    encoded: str | bytes | dict[str, Any],
    expected_vehicle_identity: str,
  ) -> VehicleProfile:
    payload = encoded if isinstance(encoded, dict) else json.loads(encoded)
    if int(payload["schema_version"]) != PROFILE_SCHEMA_VERSION:
      raise ValueError("vehicle profile schema is incompatible")
    identity = str(payload["vehicle_identity"])
    if identity != expected_vehicle_identity:
      raise ValueError("vehicle profile belongs to a different vehicle")

    nodes = []
    for raw_node in payload["nodes"]:
      raw_params = raw_node["parameters"]
      nodes.append(ProfileNode(
        speed_mps=float(raw_node["speed_mps"]),
        parameters=PhysicalParameters(
          torque_per_lateral_accel=float(raw_params["torque_per_lateral_accel"]),
          rack_gain_deg_s2_per_torque=float(raw_params["rack_gain_deg_s2_per_torque"]),
          rack_damping_per_s=float(raw_params["rack_damping_per_s"]),
          transport_delay_s=float(raw_params["transport_delay_s"]),
          static_friction_torque=float(raw_params["static_friction_torque"]),
          kinetic_friction_torque=float(raw_params["kinetic_friction_torque"]),
          rack_rate_resolution_deg_s=float(raw_params["rack_rate_resolution_deg_s"]),
          confidence=float(raw_params["confidence"]),
          qualified=bool(raw_params["qualified"]),
        ),
        clean_support_s=float(raw_node["clean_support_s"]),
        sample_count=int(raw_node["sample_count"]),
        validation_count=int(raw_node["validation_count"]),
        validation_rms=float(raw_node["validation_rms"]),
      ))
    return cls(
      vehicle_identity=identity,
      revision=int(payload["revision"]),
      provenance=str(payload["provenance"]),
      nodes=tuple(nodes),
      schema_version=int(payload["schema_version"]),
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
