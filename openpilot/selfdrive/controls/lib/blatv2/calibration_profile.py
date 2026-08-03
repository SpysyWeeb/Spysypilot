"""Observable, speed-local inverse-torque calibration for BLaTv2.

This profile deliberately contains only quantities that can be identified from
ordinary driving.  ``torque_per_lateral_accel`` maps lateral acceleration in
m/s^2 to normalized steering torque.  The offset remains in m/s^2, while both
friction values are normalized steering torque.  Transport delay and rack-rate
resolution are seed/measurement metadata; they are not rack dynamics.

The explicit units resolve an ambiguity in the retired physical profile.  In
the current openpilot/opendbc convention, torque-tuning ``friction`` is stored
as normalized torque.  ``get_friction`` temporarily multiplies it by
``latAccelFactor`` and the vehicle callback divides by that factor, recovering
the same normalized-torque magnitude.  The seed builder therefore copies that
value without applying the callback slope a second time.

Profiles are immutable snapshots.  Learning publishes a new snapshot offroad;
live code may interpolate one but never mutate it.  The schema and Params key
are intentionally incompatible with every prior BLaTv2 profile.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any


CALIBRATION_PROFILE_SCHEMA_VERSION = 3
CALIBRATION_PROFILE_PARAM_KEY = "BLaTv2ObservableCalibrationProfile"
DEFAULT_SPEED_NODES_MPS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)


def _require_exact_keys(payload: object, expected: frozenset[str], context: str) -> dict[str, Any]:
  if not isinstance(payload, dict):
    raise ValueError(f"{context} must be an object")
  keys = frozenset(payload)
  if keys != expected:
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    raise ValueError(f"{context} keys are incompatible: missing={missing}, unknown={unknown}")
  return payload


def _require_int(value: object, context: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise ValueError(f"{context} must be an integer")
  return value


def _require_bool(value: object, context: str) -> bool:
  if not isinstance(value, bool):
    raise ValueError(f"{context} must be a boolean")
  return value


@dataclass(frozen=True, slots=True)
class CalibrationParameters:
  """Observable inverse-torque values evaluated at one speed.

  ``torque_per_lateral_accel`` is normalized torque per m/s^2.
  ``lateral_accel_offset_correction_mps2`` is a signed residual correction in
  m/s^2 after the detected stock ``latAccelOffset`` has already been removed.
  ``kinetic_friction_torque`` and ``static_breakaway_torque`` are normalized
  torque magnitudes.  Static breakaway cannot be smaller than moving friction.
  """

  torque_per_lateral_accel: float
  lateral_accel_offset_correction_mps2: float
  kinetic_friction_torque: float
  static_breakaway_torque: float
  transport_delay_s: float
  rack_rate_resolution_deg_s: float
  confidence: float
  qualified: bool

  def __post_init__(self) -> None:
    values = (
      self.torque_per_lateral_accel,
      self.lateral_accel_offset_correction_mps2,
      self.kinetic_friction_torque,
      self.static_breakaway_torque,
      self.transport_delay_s,
      self.rack_rate_resolution_deg_s,
      self.confidence,
    )
    if not all(math.isfinite(value) for value in values):
      raise ValueError("calibration parameters must be finite")
    if self.torque_per_lateral_accel <= 0.0:
      raise ValueError("torque per lateral acceleration must be positive")
    if self.kinetic_friction_torque < 0.0 or self.static_breakaway_torque < 0.0:
      raise ValueError("friction torque magnitudes must be non-negative")
    if self.kinetic_friction_torque > self.static_breakaway_torque:
      raise ValueError("kinetic friction cannot exceed static breakaway")
    if self.transport_delay_s < 0.0 or self.rack_rate_resolution_deg_s < 0.0:
      raise ValueError("seed metadata must be non-negative")
    if not 0.0 <= self.confidence <= 1.0:
      raise ValueError("confidence must be within [0, 1]")
    if not isinstance(self.qualified, bool):
      raise ValueError("qualified must be a boolean")


@dataclass(frozen=True, slots=True)
class CalibrationProfileNode:
  """One speed node plus independent base, motion, and breakaway evidence."""

  speed_mps: float
  parameters: CalibrationParameters
  base_support_s: float
  base_sample_count: int
  moving_support_s: float
  moving_sample_count: int
  breakaway_support_s: float
  breakaway_sample_count: int
  cross_fit_route_count: int
  full_fit_candidate_rms: float
  breakaway_full_fit_candidate_rms: float | None

  def __post_init__(self) -> None:
    if not isinstance(self.parameters, CalibrationParameters):
      raise ValueError("profile-node parameters must be CalibrationParameters")
    values = (
      self.speed_mps,
      self.base_support_s,
      self.moving_support_s,
      self.breakaway_support_s,
      self.full_fit_candidate_rms,
    )
    if self.breakaway_full_fit_candidate_rms is not None:
      values += (self.breakaway_full_fit_candidate_rms,)
    if not all(math.isfinite(value) for value in values):
      raise ValueError("profile-node evidence must be finite")
    if (
      self.speed_mps < 0.0
      or self.base_support_s < 0.0
      or self.moving_support_s < 0.0
      or self.breakaway_support_s < 0.0
      or self.full_fit_candidate_rms < 0.0
      or (self.breakaway_full_fit_candidate_rms is not None and self.breakaway_full_fit_candidate_rms < 0.0)
    ):
      raise ValueError("profile-node evidence is outside its valid domain")
    for count_name in (
      "base_sample_count",
      "moving_sample_count",
      "breakaway_sample_count",
      "cross_fit_route_count",
    ):
      count = getattr(self, count_name)
      if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{count_name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class InterpolatedCalibration:
  """Continuous parameters plus the bounding-node provenance."""

  parameters: CalibrationParameters
  lower_node: int
  upper_node: int
  upper_weight: float


@dataclass(frozen=True, slots=True)
class VehicleCalibrationProfile:
  """Atomic vehicle-specific observable calibration snapshot."""

  vehicle_identity: str
  revision: int
  provenance: str
  nodes: tuple[CalibrationProfileNode, ...]
  schema_version: int = CALIBRATION_PROFILE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != CALIBRATION_PROFILE_SCHEMA_VERSION:
      raise ValueError("calibration profile schema is incompatible")
    if not isinstance(self.vehicle_identity, str) or not self.vehicle_identity:
      raise ValueError("vehicle identity must not be empty")
    if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
      raise ValueError("profile revision must be a non-negative integer")
    if not isinstance(self.provenance, str) or not self.provenance:
      raise ValueError("profile provenance must not be empty")
    if not isinstance(self.nodes, tuple) or len(self.nodes) < 2:
      raise ValueError("calibration profile requires at least two speed nodes")
    if not all(isinstance(node, CalibrationProfileNode) for node in self.nodes):
      raise ValueError("calibration profile nodes have an incompatible type")
    if any(
      right.speed_mps <= left.speed_mps
      for left, right in zip(self.nodes, self.nodes[1:], strict=False)
    ):
      raise ValueError("profile speed nodes must be strictly increasing")

  @property
  def qualified(self) -> bool:
    """A full profile is eligible only when every speed region is qualified."""
    return all(node.parameters.qualified for node in self.nodes)

  @property
  def speed_nodes_mps(self) -> tuple[float, ...]:
    return tuple(node.speed_mps for node in self.nodes)

  def parameters_at(self, v_ego: float) -> InterpolatedCalibration:
    """Linearly interpolate speed-local values, holding both ends flat."""
    speed = abs(float(v_ego))
    if not math.isfinite(speed):
      raise ValueError("vehicle speed must be finite")

    if speed <= self.nodes[0].speed_mps:
      return InterpolatedCalibration(self.nodes[0].parameters, 0, 0, 0.0)
    if speed >= self.nodes[-1].speed_mps:
      last = len(self.nodes) - 1
      return InterpolatedCalibration(self.nodes[last].parameters, last, last, 0.0)

    upper = 1
    while self.nodes[upper].speed_mps < speed:
      upper += 1
    lower = upper - 1
    lower_node = self.nodes[lower]
    upper_node = self.nodes[upper]
    weight = (speed - lower_node.speed_mps) / (upper_node.speed_mps - lower_node.speed_mps)
    lower_params = lower_node.parameters
    upper_params = upper_node.parameters

    def blend(lower_value: float, upper_value: float) -> float:
      return lower_value + weight * (upper_value - lower_value)

    parameters = CalibrationParameters(
      torque_per_lateral_accel=blend(
        lower_params.torque_per_lateral_accel,
        upper_params.torque_per_lateral_accel,
      ),
      lateral_accel_offset_correction_mps2=blend(
        lower_params.lateral_accel_offset_correction_mps2,
        upper_params.lateral_accel_offset_correction_mps2,
      ),
      kinetic_friction_torque=blend(
        lower_params.kinetic_friction_torque,
        upper_params.kinetic_friction_torque,
      ),
      static_breakaway_torque=blend(
        lower_params.static_breakaway_torque,
        upper_params.static_breakaway_torque,
      ),
      transport_delay_s=blend(lower_params.transport_delay_s, upper_params.transport_delay_s),
      rack_rate_resolution_deg_s=blend(
        lower_params.rack_rate_resolution_deg_s,
        upper_params.rack_rate_resolution_deg_s,
      ),
      confidence=blend(lower_params.confidence, upper_params.confidence),
      qualified=lower_params.qualified and upper_params.qualified,
    )
    return InterpolatedCalibration(parameters, lower, upper, weight)

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
            "lateral_accel_offset_correction_mps2": node.parameters.lateral_accel_offset_correction_mps2,
            "kinetic_friction_torque": node.parameters.kinetic_friction_torque,
            "static_breakaway_torque": node.parameters.static_breakaway_torque,
            "transport_delay_s": node.parameters.transport_delay_s,
            "rack_rate_resolution_deg_s": node.parameters.rack_rate_resolution_deg_s,
            "confidence": node.parameters.confidence,
            "qualified": node.parameters.qualified,
          },
          "base_support_s": node.base_support_s,
          "base_sample_count": node.base_sample_count,
          "moving_support_s": node.moving_support_s,
          "moving_sample_count": node.moving_sample_count,
          "breakaway_support_s": node.breakaway_support_s,
          "breakaway_sample_count": node.breakaway_sample_count,
          "cross_fit_route_count": node.cross_fit_route_count,
          "full_fit_candidate_rms": node.full_fit_candidate_rms,
          "breakaway_full_fit_candidate_rms": node.breakaway_full_fit_candidate_rms,
        }
        for node in self.nodes
      ],
    }

  def to_json(self) -> str:
    return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

  @classmethod
  def from_json(
    cls,
    encoded: str | bytes | dict[str, Any],
    expected_vehicle_identity: str,
  ) -> VehicleCalibrationProfile:
    payload: object = encoded if isinstance(encoded, dict) else json.loads(encoded)
    raw_profile = _require_exact_keys(payload, frozenset({
      "schema_version", "vehicle_identity", "revision", "provenance", "nodes",
    }), "calibration profile")
    schema_version = _require_int(raw_profile["schema_version"], "schema_version")
    if schema_version != CALIBRATION_PROFILE_SCHEMA_VERSION:
      raise ValueError("calibration profile schema is incompatible")
    identity = raw_profile["vehicle_identity"]
    if not isinstance(identity, str):
      raise ValueError("vehicle identity must be a string")
    if identity != expected_vehicle_identity:
      raise ValueError("calibration profile belongs to a different vehicle")
    provenance = raw_profile["provenance"]
    if not isinstance(provenance, str):
      raise ValueError("profile provenance must be a string")
    raw_nodes = raw_profile["nodes"]
    if not isinstance(raw_nodes, list):
      raise ValueError("calibration profile nodes must be a list")

    nodes: list[CalibrationProfileNode] = []
    node_keys = frozenset({
      "speed_mps", "parameters", "base_support_s", "base_sample_count",
      "moving_support_s", "moving_sample_count", "breakaway_support_s",
      "breakaway_sample_count", "cross_fit_route_count", "full_fit_candidate_rms",
      "breakaway_full_fit_candidate_rms",
    })
    parameter_keys = frozenset({
      "torque_per_lateral_accel", "lateral_accel_offset_correction_mps2", "kinetic_friction_torque",
      "static_breakaway_torque", "transport_delay_s", "rack_rate_resolution_deg_s",
      "confidence", "qualified",
    })
    for index, raw_node_value in enumerate(raw_nodes):
      raw_node = _require_exact_keys(raw_node_value, node_keys, f"calibration node {index}")
      raw_params = _require_exact_keys(raw_node["parameters"], parameter_keys, f"calibration node {index} parameters")
      breakaway_rms_value = raw_node["breakaway_full_fit_candidate_rms"]
      nodes.append(CalibrationProfileNode(
        speed_mps=float(raw_node["speed_mps"]),
        parameters=CalibrationParameters(
          torque_per_lateral_accel=float(raw_params["torque_per_lateral_accel"]),
          lateral_accel_offset_correction_mps2=float(raw_params["lateral_accel_offset_correction_mps2"]),
          kinetic_friction_torque=float(raw_params["kinetic_friction_torque"]),
          static_breakaway_torque=float(raw_params["static_breakaway_torque"]),
          transport_delay_s=float(raw_params["transport_delay_s"]),
          rack_rate_resolution_deg_s=float(raw_params["rack_rate_resolution_deg_s"]),
          confidence=float(raw_params["confidence"]),
          qualified=_require_bool(raw_params["qualified"], f"calibration node {index} qualified"),
        ),
        base_support_s=float(raw_node["base_support_s"]),
        base_sample_count=_require_int(raw_node["base_sample_count"], f"calibration node {index} base_sample_count"),
        moving_support_s=float(raw_node["moving_support_s"]),
        moving_sample_count=_require_int(raw_node["moving_sample_count"], f"calibration node {index} moving_sample_count"),
        breakaway_support_s=float(raw_node["breakaway_support_s"]),
        breakaway_sample_count=_require_int(raw_node["breakaway_sample_count"], f"calibration node {index} breakaway_sample_count"),
        cross_fit_route_count=_require_int(raw_node["cross_fit_route_count"], f"calibration node {index} cross_fit_route_count"),
        full_fit_candidate_rms=float(raw_node["full_fit_candidate_rms"]),
        breakaway_full_fit_candidate_rms=(None if breakaway_rms_value is None else float(breakaway_rms_value)),
      ))
    return cls(
      vehicle_identity=identity,
      revision=_require_int(raw_profile["revision"], "profile revision"),
      provenance=provenance,
      nodes=tuple(nodes),
      schema_version=schema_version,
    )


def make_calibration_seed_profile(
  vehicle_identity: str,
  torque_callback_slope: float,
  stock_friction_torque: float,
  transport_delay_s: float,
  rack_rate_resolution_deg_s: float,
  lateral_accel_offset_correction_mps2: float = 0.0,
  speed_nodes_mps: tuple[float, ...] = DEFAULT_SPEED_NODES_MPS,
) -> VehicleCalibrationProfile:
  """Create an unqualified seed using stock's current friction convention.

  ``torque_callback_slope`` is normalized torque per m/s^2.  Current opendbc
  ``friction`` is already normalized torque: stock's ``get_friction`` scales it
  into lateral-acceleration space, then the callback scales it back.  Copying
  the value here preserves that round trip and avoids multiplying it by the
  callback slope twice.  The explicit ``_torque`` suffix records the units
  until opendbc's documented future convention change is implemented.
  """
  slope = float(torque_callback_slope)
  stock_friction = float(stock_friction_torque)
  if not math.isfinite(stock_friction) or stock_friction < 0.0:
    raise ValueError("stock friction torque must be finite and non-negative")
  friction_torque = stock_friction
  parameters = CalibrationParameters(
    torque_per_lateral_accel=slope,
    lateral_accel_offset_correction_mps2=float(lateral_accel_offset_correction_mps2),
    kinetic_friction_torque=friction_torque,
    static_breakaway_torque=friction_torque,
    transport_delay_s=float(transport_delay_s),
    rack_rate_resolution_deg_s=float(rack_rate_resolution_deg_s),
    confidence=0.0,
    qualified=False,
  )
  nodes = tuple(
    CalibrationProfileNode(
      speed_mps=float(speed),
      parameters=parameters,
      base_support_s=0.0,
      base_sample_count=0,
      moving_support_s=0.0,
      moving_sample_count=0,
      breakaway_support_s=0.0,
      breakaway_sample_count=0,
      cross_fit_route_count=0,
      full_fit_candidate_rms=0.0,
      breakaway_full_fit_candidate_rms=None,
    )
    for speed in speed_nodes_mps
  )
  return VehicleCalibrationProfile(
    vehicle_identity=vehicle_identity,
    revision=0,
    provenance="unqualified observable runtime calibration seed",
    nodes=nodes,
  )
