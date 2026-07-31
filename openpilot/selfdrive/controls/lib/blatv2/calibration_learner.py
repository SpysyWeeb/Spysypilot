"""Deterministic learner for BLaTv2's observable inverse-torque map.

The fitted law at each speed node is::

  applied = gain * (-measured_lataccel + offset_correction)
            + kinetic * motion_sign
            + static * breakaway_sign

Only one friction sign is nonzero per row. Quantized/stationary rows use both
signs as zero and intentionally identify an *effective settled inverse map*;
the coefficients are not claimed to be a pure physical decomposition. The
first resolved motion following a continuous stationary dwell of at least the
seed transport delay is a breakaway row. Later moving rows are kinetic rows.

Inputs are the measured-only :class:`learner.LearningSample`. Slew build and
release frames are authority evidence but never equality-fit. Evidence schema
5 is deliberately incompatible with the retired dynamic-rack schema 4.

The inverse map is solved as deterministic physically constrained least
squares. Gain and moving friction are non-negative, and static breakaway is
moving friction plus a non-negative excess. A boundary solution is re-solved
on its active face; raw coefficients are never clipped after fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import hmac
import json
import math
from typing import Any

from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  CALIBRATION_PROFILE_SCHEMA_VERSION,
  CalibrationParameters,
  CalibrationProfileNode,
  VehicleCalibrationProfile,
)
from openpilot.selfdrive.controls.lib.blatv2.learner import (
  ActuatorBoundary,
  LearningSample,
  MIN_APPLIED_TORQUE_SPAN,
  MIN_EXCITATION_NODE_WEIGHT,
  MIN_LATERAL_ACCEL_RMS_MPS2,
  MIN_LATERAL_ACCEL_SPAN_MPS2,
  TRAIN_VALIDATION_BLOCK_SAMPLES,
  minimum_clean_support_s,
)


CALIBRATION_EVIDENCE_SCHEMA_VERSION = 5
MIN_VALIDATION_SUPPORT_FRACTION = 0.20
MIN_STRATUM_TRAINING_ROWS = 4
MIN_STRATUM_VALIDATION_ROWS = 4
NORMAL_MATRIX_RELATIVE_PIVOT_MIN = 1e-10
VALIDATION_RMS_ABSOLUTE_TOLERANCE = 1e-12


class CalibrationQualificationReason(StrEnum):
  QUALIFIED = "qualified"
  INSUFFICIENT_SUPPORT = "insufficient_support"
  INSUFFICIENT_VALIDATION = "insufficient_validation"
  INSUFFICIENT_EXCITATION = "insufficient_excitation"
  INSUFFICIENT_MOVING_EVIDENCE = "insufficient_moving_evidence"
  INSUFFICIENT_BREAKAWAY_EVIDENCE = "insufficient_breakaway_evidence"
  SINGULAR_FIT = "singular_fit"
  INVALID_PARAMETERS = "invalid_parameters"
  VALIDATION_REGRESSION = "validation_regression"
  MOVING_VALIDATION_REGRESSION = "moving_validation_regression"
  BREAKAWAY_VALIDATION_REGRESSION = "breakaway_validation_regression"
  AUTHORITY_VALIDATION_REGRESSION = "authority_validation_regression"


class _Regression:
  __slots__ = ("normal", "rhs", "target_squared", "weight_s", "count")

  def __init__(self) -> None:
    self.normal = [0.0] * 16
    self.rhs = [0.0] * 4
    self.target_squared = 0.0
    self.weight_s = 0.0
    self.count = 0

  def add(self, x: tuple[float, float, float, float], y: float, weight: float) -> None:
    for row in range(4):
      self.rhs[row] += weight * x[row] * y
      for column in range(4):
        self.normal[row * 4 + column] += weight * x[row] * x[column]
    self.target_squared += weight * y * y
    self.weight_s += weight
    self.count += 1

  def rms(self, coefficients: tuple[float, float, float, float]) -> float | None:
    if self.weight_s <= 0.0:
      return None
    linear = sum(coefficients[i] * self.rhs[i] for i in range(4))
    quadratic = sum(coefficients[i] * self.normal[i * 4 + j] * coefficients[j] for i in range(4) for j in range(4))
    error = self.target_squared - 2.0 * linear + quadratic
    scale = max(self.target_squared, quadratic, 1.0)
    if error < 0.0 and abs(error) <= 1e-12 * scale:
      error = 0.0
    return None if error < 0.0 or not math.isfinite(error) else math.sqrt(error / self.weight_s)

  def encoded(self) -> dict[str, Any]:
    return {
      "count": self.count,
      "normal": [value.hex() for value in self.normal],
      "rhs": [value.hex() for value in self.rhs],
      "target_squared": self.target_squared.hex(),
      "weight_s": self.weight_s.hex(),
    }

  @classmethod
  def decoded(cls, raw: object, context: str) -> _Regression:
    payload = _exact(raw, {"count", "normal", "rhs", "target_squared", "weight_s"}, context)
    if type(payload["count"]) is not int or payload["count"] < 0:
      raise ValueError(f"{context}.count is invalid")
    normal = _hex_list(payload["normal"], 16, f"{context}.normal")
    rhs = _hex_list(payload["rhs"], 4, f"{context}.rhs")
    result = cls()
    result.count = payload["count"]
    result.normal[:] = normal
    result.rhs[:] = rhs
    result.target_squared = _hex(payload["target_squared"], f"{context}.target_squared")
    result.weight_s = _hex(payload["weight_s"], f"{context}.weight_s")
    if result.target_squared < 0.0 or result.weight_s < 0.0 or (result.count > 0) != (result.weight_s > 0.0):
      raise ValueError(f"{context} has inconsistent support")
    if result.count == 0 and any(value != 0.0 for value in (*normal, *rhs, result.target_squared)):
      raise ValueError(f"{context} empty statistics are nonzero")
    return result


def _combine(*parts: _Regression) -> _Regression:
  result = _Regression()
  for part in parts:
    for i in range(16):
      result.normal[i] += part.normal[i]
    for i in range(4):
      result.rhs[i] += part.rhs[i]
    result.target_squared += part.target_squared
    result.weight_s += part.weight_s
    result.count += part.count
  return result


class _Node:
  def __init__(self) -> None:
    self.clean_support_s = 0.0
    self.supported_sample_count = 0
    self.base_support_s = 0.0
    self.base_sample_count = 0
    self.moving_support_s = 0.0
    self.moving_sample_count = 0
    self.breakaway_support_s = 0.0
    self.breakaway_sample_count = 0
    self.authority_support_s = 0.0
    self.authority_sample_count = 0
    self.authority_magnitude_sample_count = 0
    self.authority_slew_build_sample_count = 0
    self.authority_slew_release_sample_count = 0
    self.authority_fit_support_s = 0.0
    self.authority_fit_sample_count = 0
    self.authority_unresolved_sample_count = 0
    self.lat_min = math.inf
    self.lat_max = -math.inf
    self.lat_energy = 0.0
    self.torque_min = math.inf
    self.torque_max = -math.inf
    self.rack_travel_deg = 0.0
    self.rack_reversals = 0
    self.last_direction = 0
    self.stationary_dwell_s = 0.0
    self.fit_count = 0
    self.authority_fit_count = 0
    self.lateral_accel_direction_mask = 0
    self.applied_torque_direction_mask = 0
    self.moving_direction_mask = 0
    self.breakaway_direction_mask = 0
    self.training = _Regression()
    self.validation = _Regression()
    self.moving_training = _Regression()
    self.moving_validation = _Regression()
    self.breakaway_training = _Regression()
    self.breakaway_validation = _Regression()
    self.authority_training = _Regression()
    self.authority_validation = _Regression()


@dataclass(frozen=True, slots=True)
class CalibrationNodeEvidenceSnapshot:
  clean_support_s: float
  supported_sample_count: int
  base_support_s: float
  base_sample_count: int
  moving_support_s: float
  moving_sample_count: int
  breakaway_support_s: float
  breakaway_sample_count: int
  authority_support_s: float
  authority_sample_count: int
  authority_magnitude_sample_count: int
  authority_slew_build_sample_count: int
  authority_slew_release_sample_count: int
  authority_unresolved_sample_count: int
  authority_fit_support_s: float
  authority_fit_sample_count: int
  stationary_dwell_s: float
  training_support_s: float
  training_count: int
  validation_support_s: float
  validation_count: int
  moving_training_support_s: float
  moving_training_count: int
  moving_validation_support_s: float
  moving_validation_count: int
  breakaway_training_support_s: float
  breakaway_training_count: int
  breakaway_validation_support_s: float
  breakaway_validation_count: int
  authority_training_support_s: float
  authority_training_count: int
  authority_validation_support_s: float
  authority_validation_count: int
  lateral_accel_span_mps2: float
  applied_torque_span: float
  lateral_accel_directions: int
  applied_torque_directions: int
  rack_reversals: int


@dataclass(frozen=True, slots=True)
class CalibrationNodeQualificationReport:
  node_index: int
  speed_mps: float
  minimum_support_s: float
  clean_support_s: float
  supported_sample_count: int
  training_count: int
  validation_count: int
  validation_support_s: float
  base_support_s: float
  base_sample_count: int
  moving_support_s: float
  moving_sample_count: int
  moving_training_count: int
  moving_validation_count: int
  breakaway_support_s: float
  breakaway_sample_count: int
  breakaway_training_count: int
  breakaway_validation_count: int
  lateral_accel_span_mps2: float
  lateral_accel_rms_mps2: float
  rack_travel_deg: float
  applied_torque_span: float
  rack_reversals: int
  lateral_accel_directions: int
  applied_torque_directions: int
  seed_validation_rms: float | None
  candidate_validation_rms: float | None
  moving_seed_validation_rms: float | None
  moving_candidate_validation_rms: float | None
  breakaway_seed_validation_rms: float | None
  breakaway_candidate_validation_rms: float | None
  confidence: float
  reasons: tuple[CalibrationQualificationReason, ...]
  candidate_parameters: CalibrationParameters | None
  authority_support_s: float = 0.0
  authority_sample_count: int = 0
  authority_fit_support_s: float = 0.0
  authority_fit_sample_count: int = 0
  authority_training_count: int = 0
  authority_validation_count: int = 0
  authority_seed_validation_rms: float | None = None
  authority_candidate_validation_rms: float | None = None
  authority_magnitude_sample_count: int = 0
  authority_slew_build_sample_count: int = 0
  authority_slew_release_sample_count: int = 0
  authority_unresolved_sample_count: int = 0

  @property
  def qualified(self) -> bool:
    return self.reasons == (CalibrationQualificationReason.QUALIFIED,)


@dataclass(frozen=True, slots=True)
class CalibrationLearningResult:
  node_reports: tuple[CalibrationNodeQualificationReport, ...]
  candidate_profile: VehicleCalibrationProfile | None

  @property
  def all_nodes_qualified(self) -> bool:
    return self.candidate_profile is not None


def _canonical(payload: object) -> bytes:
  return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f"duplicate evidence key: {key}")
    result[key] = value
  return result


def _exact(raw: object, keys: set[str], context: str) -> dict[str, Any]:
  if type(raw) is not dict or set(raw) != keys:
    raise ValueError(f"{context} keys are incompatible")
  return raw


def _hex(raw: object, context: str) -> float:
  if type(raw) is not str:
    raise ValueError(f"{context} must be a canonical hexadecimal float")
  try:
    value = float.fromhex(raw)
  except ValueError as exc:
    raise ValueError(f"{context} is invalid") from exc
  if not math.isfinite(value) or value.hex() != raw:
    raise ValueError(f"{context} is not canonical and finite")
  return value


def _hex_list(raw: object, length: int, context: str) -> list[float]:
  if type(raw) is not list or len(raw) != length:
    raise ValueError(f"{context} has the wrong length")
  return [_hex(value, f"{context}[{index}]") for index, value in enumerate(raw)]


def _extreme(value: float) -> str | None:
  return None if not math.isfinite(value) else value.hex()


def _decode_extreme(raw: object, positive: bool, context: str) -> float:
  return (math.inf if positive else -math.inf) if raw is None else _hex(raw, context)


def _solve_free_system(
  normal: tuple[float, ...],
  rhs: tuple[float, ...],
  free: tuple[int, ...],
) -> tuple[float, ...] | None:
  """Solve one deterministic face of the constrained least-squares fit."""
  scales: list[float] = []
  for index in free:
    diagonal = normal[index * 4 + index]
    if not math.isfinite(diagonal) or diagonal <= 0.0:
      return None
    scales.append(math.sqrt(diagonal))
  size = len(free)
  matrix = [
    [
      normal[row * 4 + column] / (scales[r] * scales[c])
      for c, column in enumerate(free)
    ]
    + [rhs[row] / scales[r]]
    for r, row in enumerate(free)
  ]
  for column in range(size):
    pivot = max(
      range(column, size),
      key=lambda row: abs(matrix[row][column]),
    )
    if abs(matrix[pivot][column]) < NORMAL_MATRIX_RELATIVE_PIVOT_MIN:
      return None
    matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
    divisor = matrix[column][column]
    for c in range(column, size + 1):
      matrix[column][c] /= divisor
    for row in range(size):
      if row == column:
        continue
      factor = matrix[row][column]
      for c in range(column, size + 1):
        matrix[row][c] -= factor * matrix[column][c]
  answer = tuple(matrix[index][size] / scales[index] for index in range(size))
  return answer if all(math.isfinite(value) for value in answer) else None


def _weighted_error(
  evidence: _Regression,
  coefficients: tuple[float, float, float, float],
) -> float | None:
  linear = sum(
    coefficients[index] * evidence.rhs[index]
    for index in range(4)
  )
  quadratic = sum(
    coefficients[row]
    * evidence.normal[row * 4 + column]
    * coefficients[column]
    for row in range(4)
    for column in range(4)
  )
  error = evidence.target_squared - 2.0 * linear + quadratic
  scale = max(evidence.target_squared, quadratic, 1.0)
  if error < 0.0 and abs(error) <= 1e-12 * scale:
    return 0.0
  return error if math.isfinite(error) and error >= 0.0 else None


def _solve(evidence: _Regression) -> tuple[float, float, float, float] | None:
  """Fit the closest physically admissible observable inverse-torque map.

  Reparameterize ``static = kinetic + breakaway_excess`` and enumerate the
  eight deterministic active sets for the three non-negative coefficients:
  gain, kinetic friction, and breakaway excess. The signed intercept remains
  unconstrained. This is a constrained least-squares solve, not a post-fit
  clamp: each boundary solution is re-solved on its own face and compared in
  the original weighted objective. A zero breakaway excess is the honest
  result when ordinary driving does not independently resolve extra stiction.
  Equal-cost faces resolve to the lowest mask by construction.
  """
  # Original coefficients c=[gain, intercept, kinetic, static] are produced
  # from d=[gain, intercept, kinetic, breakaway_excess] by c=A*d.
  transform = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0, 1.0),
  )
  transformed_normal = tuple(
    sum(
      transform[source_row][row]
      * evidence.normal[source_row * 4 + source_column]
      * transform[source_column][column]
      for source_row in range(4)
      for source_column in range(4)
    )
    for row in range(4)
    for column in range(4)
  )
  transformed_rhs = tuple(
    sum(
      transform[source][index] * evidence.rhs[source]
      for source in range(4)
    )
    for index in range(4)
  )
  constrained = (0, 2, 3)
  best: tuple[float, float, float, float] | None = None
  best_error: float | None = None
  for mask in range(1 << len(constrained)):
    fixed = {
      index
      for bit, index in enumerate(constrained)
      if mask & (1 << bit)
    }
    free = tuple(index for index in range(4) if index not in fixed)
    solved = _solve_free_system(
      transformed_normal,
      transformed_rhs,
      free,
    )
    if solved is None:
      continue
    parameters = [0.0] * 4
    for index, value in zip(free, solved, strict=True):
      parameters[index] = value
    if any(parameters[index] < 0.0 for index in constrained):
      continue
    coefficients = (
      parameters[0],
      parameters[1],
      parameters[2],
      parameters[2] + parameters[3],
    )
    error = _weighted_error(evidence, coefficients)
    if error is not None and (best_error is None or error < best_error):
      best = coefficients
      best_error = error
  return best


def _seed_coefficients(parameters: CalibrationParameters) -> tuple[float, float, float, float]:
  return (
    parameters.torque_per_lateral_accel,
    parameters.torque_per_lateral_accel * parameters.lateral_accel_offset_correction_mps2,
    parameters.kinetic_friction_torque,
    parameters.static_breakaway_torque,
  )


def minimum_calibration_support_s(speed_mps: float) -> float:
  return minimum_clean_support_s(speed_mps)


def calibration_evidence_sha256(encoded: bytes) -> str:
  if type(encoded) is not bytes:
    raise TypeError("calibration evidence identity requires bytes")
  return hashlib.sha256(encoded).hexdigest()


class CalibrationProfileLearner:
  """Speed-local evidence accumulator and offroad qualification engine."""

  def __init__(self, seed_profile: VehicleCalibrationProfile) -> None:
    if not isinstance(seed_profile, VehicleCalibrationProfile):
      raise TypeError("calibration learner requires VehicleCalibrationProfile")
    if seed_profile.schema_version != CALIBRATION_PROFILE_SCHEMA_VERSION:
      raise ValueError("calibration seed schema is incompatible")
    self.seed_profile = seed_profile
    self._nodes = tuple(_Node() for _ in seed_profile.nodes)

  @property
  def speed_nodes_mps(self) -> tuple[float, ...]:
    return self.seed_profile.speed_nodes_mps

  def evidence_for_node(self, index: int) -> CalibrationNodeEvidenceSnapshot:
    node = self._nodes[index]
    lateral_accel_span = 0.0 if not math.isfinite(node.lat_min) else node.lat_max - node.lat_min
    applied_torque_span = 0.0 if not math.isfinite(node.torque_min) else node.torque_max - node.torque_min
    return CalibrationNodeEvidenceSnapshot(
      clean_support_s=node.clean_support_s,
      supported_sample_count=node.supported_sample_count,
      base_support_s=node.base_support_s,
      base_sample_count=node.base_sample_count,
      moving_support_s=node.moving_support_s,
      moving_sample_count=node.moving_sample_count,
      breakaway_support_s=node.breakaway_support_s,
      breakaway_sample_count=node.breakaway_sample_count,
      authority_support_s=node.authority_support_s,
      authority_sample_count=node.authority_sample_count,
      authority_magnitude_sample_count=node.authority_magnitude_sample_count,
      authority_slew_build_sample_count=node.authority_slew_build_sample_count,
      authority_slew_release_sample_count=node.authority_slew_release_sample_count,
      authority_unresolved_sample_count=node.authority_unresolved_sample_count,
      authority_fit_support_s=node.authority_fit_support_s,
      authority_fit_sample_count=node.authority_fit_sample_count,
      stationary_dwell_s=node.stationary_dwell_s,
      training_support_s=node.training.weight_s,
      training_count=node.training.count,
      validation_support_s=node.validation.weight_s,
      validation_count=node.validation.count,
      moving_training_support_s=node.moving_training.weight_s,
      moving_training_count=node.moving_training.count,
      moving_validation_support_s=node.moving_validation.weight_s,
      moving_validation_count=node.moving_validation.count,
      breakaway_training_support_s=node.breakaway_training.weight_s,
      breakaway_training_count=node.breakaway_training.count,
      breakaway_validation_support_s=node.breakaway_validation.weight_s,
      breakaway_validation_count=node.breakaway_validation.count,
      authority_training_support_s=node.authority_training.weight_s,
      authority_training_count=node.authority_training.count,
      authority_validation_support_s=node.authority_validation.weight_s,
      authority_validation_count=node.authority_validation.count,
      lateral_accel_span_mps2=lateral_accel_span,
      applied_torque_span=applied_torque_span,
      lateral_accel_directions=node.lateral_accel_direction_mask.bit_count(),
      applied_torque_directions=node.applied_torque_direction_mask.bit_count(),
      rack_reversals=node.rack_reversals,
    )

  def reset_route_transients(self) -> None:
    for node in self._nodes:
      node.stationary_dwell_s = 0.0
      node.last_direction = 0

  def _supports(self, speed: float) -> tuple[tuple[int, float], ...]:
    nodes = self.speed_nodes_mps
    if speed <= nodes[0]:
      return ((0, 1.0),)
    if speed >= nodes[-1]:
      return ((len(nodes) - 1, 1.0),)
    upper = 1
    while nodes[upper] < speed:
      upper += 1
    lower = upper - 1
    weight = (speed - nodes[lower]) / (nodes[upper] - nodes[lower])
    if weight <= 0.0:
      return ((lower, 1.0),)
    if weight >= 1.0:
      return ((upper, 1.0),)
    return ((lower, 1.0 - weight), (upper, weight))

  def _row(self, node: _Node, seed: CalibrationParameters, sample: LearningSample) -> tuple[tuple[float, float, float, float], str]:
    threshold = max(seed.rack_rate_resolution_deg_s, 1e-12)
    # The declared resolution is the first physically resolved sensor value,
    # not part of the unresolved deadband.
    direction = 1 if sample.rack_rate_deg_s >= threshold else -1 if sample.rack_rate_deg_s <= -threshold else 0
    if direction == 0:
      node.stationary_dwell_s += sample.dt_s
      return (-sample.measured_lateral_accel_mps2, 1.0, 0.0, 0.0), "base"
    is_breakaway = node.stationary_dwell_s + 1e-12 >= seed.transport_delay_s
    node.stationary_dwell_s = 0.0
    if node.last_direction != 0 and direction != node.last_direction:
      node.rack_reversals += 1
    node.last_direction = direction
    if is_breakaway:
      node.breakaway_direction_mask |= 1 if direction < 0 else 2
      return (-sample.measured_lateral_accel_mps2, 1.0, 0.0, float(direction)), "breakaway"
    node.moving_direction_mask |= 1 if direction < 0 else 2
    return (-sample.measured_lateral_accel_mps2, 1.0, float(direction), 0.0), "moving"

  @staticmethod
  def _reset_node(node: _Node) -> None:
    node.stationary_dwell_s = 0.0
    node.last_direction = 0

  def add_sample(self, sample: LearningSample) -> bool:
    if not isinstance(sample, LearningSample):
      raise TypeError("calibration learner accepts LearningSample only")
    if not sample.valid or not sample.engaged or sample.steering_pressed or sample.standstill:
      self.reset_route_transients()
      return False
    reversal_evidence = (
      sample.rack_direction_reversal
      and sample._base_valid
      and not sample.actuator_constrained
      and sample.actuator_boundary == ActuatorBoundary.NONE
      and sample.magnitude_boundary_dwell_s == 0.0
    )
    if not sample.clean and not sample.authority_evidence and not reversal_evidence:
      self.reset_route_transients()
      return False
    if reversal_evidence:
      # A signed reversal is valid for the acceleration-free inverse map, but
      # it ends prior dwell/direction continuity. It may be a moving row; it
      # must not manufacture a breakaway from stale stationary history.
      self.reset_route_transients()
      for node_index, _ in self._supports(sample.speed_mps):
        self._nodes[node_index].rack_reversals += 1

    accepted = False
    for node_index, node_weight in self._supports(sample.speed_mps):
      node = self._nodes[node_index]
      seed = self.seed_profile.nodes[node_index].parameters
      weight = sample.dt_s * node_weight
      if sample.authority_evidence:
        node.authority_support_s += weight
        node.authority_sample_count += 1
        node.authority_magnitude_sample_count += bool(sample.actuator_boundary & ActuatorBoundary.MAGNITUDE)
        node.authority_slew_build_sample_count += bool(sample.actuator_boundary & ActuatorBoundary.SLEW_BUILD)
        node.authority_slew_release_sample_count += bool(sample.actuator_boundary & ActuatorBoundary.SLEW_RELEASE)
        resolved_motion = abs(sample.rack_rate_deg_s) >= max(seed.rack_rate_resolution_deg_s, 1e-12)
        equality_fit = sample.actuator_boundary == ActuatorBoundary.MAGNITUDE and sample.magnitude_boundary_dwell_s + 1e-12 >= sample.dt_s and resolved_motion
        if not equality_fit:
          node.authority_unresolved_sample_count += bool(sample.actuator_boundary & ActuatorBoundary.MAGNITUDE)
          self._reset_node(node)
          accepted = True
          continue
        predictors, _ = self._row(node, seed, sample)
        validation = (node.authority_fit_count // TRAIN_VALIDATION_BLOCK_SAMPLES) % 2 == 1
        evidence = node.authority_validation if validation else node.authority_training
        evidence.add(predictors, sample.applied_torque, weight)
        node.authority_fit_count += 1
        node.authority_fit_sample_count += 1
        node.authority_fit_support_s += weight
        accepted = True
        continue

      predictors, category = self._row(node, seed, sample)
      validation = (node.fit_count // TRAIN_VALIDATION_BLOCK_SAMPLES) % 2 == 1
      total = node.validation if validation else node.training
      total.add(predictors, sample.applied_torque, weight)
      node.fit_count += 1
      node.clean_support_s += weight
      node.supported_sample_count += 1
      node.lat_energy += weight * sample.measured_lateral_accel_mps2**2
      node.rack_travel_deg += weight * abs(sample.rack_rate_deg_s)
      if category == "base":
        node.base_support_s += weight
        node.base_sample_count += 1
      elif category == "moving":
        node.moving_support_s += weight
        node.moving_sample_count += 1
        (node.moving_validation if validation else node.moving_training).add(predictors, sample.applied_torque, weight)
      else:
        node.breakaway_support_s += weight
        node.breakaway_sample_count += 1
        (node.breakaway_validation if validation else node.breakaway_training).add(predictors, sample.applied_torque, weight)
      if node_weight >= MIN_EXCITATION_NODE_WEIGHT:
        lat = sample.measured_lateral_accel_mps2
        torque = sample.applied_torque
        node.lat_min, node.lat_max = min(node.lat_min, lat), max(node.lat_max, lat)
        node.torque_min, node.torque_max = min(node.torque_min, torque), max(node.torque_max, torque)
        node.lateral_accel_direction_mask |= 1 if lat < 0.0 else 2 if lat > 0.0 else 0
        node.applied_torque_direction_mask |= 1 if torque < 0.0 else 2 if torque > 0.0 else 0
      accepted = True
    return accepted

  def _node_report(self, index: int) -> CalibrationNodeQualificationReport:
    node = self._nodes[index]
    seed = self.seed_profile.nodes[index].parameters
    speed = self.speed_nodes_mps[index]
    minimum = minimum_calibration_support_s(speed)
    min_validation = minimum * MIN_VALIDATION_SUPPORT_FRACTION
    lat_span = 0.0 if not math.isfinite(node.lat_min) else node.lat_max - node.lat_min
    torque_span = 0.0 if not math.isfinite(node.torque_min) else node.torque_max - node.torque_min
    lat_rms = math.sqrt(node.lat_energy / node.clean_support_s) if node.clean_support_s > 0.0 else 0.0
    reasons: list[CalibrationQualificationReason] = []
    if node.clean_support_s < minimum:
      reasons.append(CalibrationQualificationReason.INSUFFICIENT_SUPPORT)
    if node.validation.weight_s < min_validation or node.validation.count < 4:
      reasons.append(CalibrationQualificationReason.INSUFFICIENT_VALIDATION)
    if (
      lat_span < MIN_LATERAL_ACCEL_SPAN_MPS2
      or lat_rms < MIN_LATERAL_ACCEL_RMS_MPS2
      or torque_span < MIN_APPLIED_TORQUE_SPAN
      or node.lateral_accel_direction_mask != 3
      or node.applied_torque_direction_mask != 3
    ):
      reasons.append(CalibrationQualificationReason.INSUFFICIENT_EXCITATION)
    if node.moving_training.count < MIN_STRATUM_TRAINING_ROWS or node.moving_validation.count < MIN_STRATUM_VALIDATION_ROWS or node.moving_direction_mask != 3:
      reasons.append(CalibrationQualificationReason.INSUFFICIENT_MOVING_EVIDENCE)
    if (
      node.breakaway_training.count < MIN_STRATUM_TRAINING_ROWS
      or node.breakaway_validation.count < MIN_STRATUM_VALIDATION_ROWS
      or node.breakaway_direction_mask != 3
    ):
      reasons.append(CalibrationQualificationReason.INSUFFICIENT_BREAKAWAY_EVIDENCE)

    authority_active = node.authority_training.count >= 4 and node.authority_validation.count >= 4
    training = _combine(node.training, node.authority_training) if authority_active else node.training
    validation = _combine(node.validation, node.authority_validation) if authority_active else node.validation
    coefficients = _solve(training)
    seed_coefficients = _seed_coefficients(seed)
    seed_rms = validation.rms(seed_coefficients)
    candidate_rms = validation.rms(coefficients) if coefficients is not None else None
    moving_seed = node.moving_validation.rms(seed_coefficients)
    moving_candidate = node.moving_validation.rms(coefficients) if coefficients is not None else None
    breakaway_seed = node.breakaway_validation.rms(seed_coefficients)
    breakaway_candidate = node.breakaway_validation.rms(coefficients) if coefficients is not None else None
    authority_seed = node.authority_validation.rms(seed_coefficients)
    authority_candidate = node.authority_validation.rms(coefficients) if coefficients is not None else None
    candidate_parameters: CalibrationParameters | None = None
    if coefficients is None:
      reasons.append(CalibrationQualificationReason.SINGULAR_FIT)
    else:
      gain, intercept, kinetic, static = coefficients
      offset = intercept / gain if gain != 0.0 else math.inf
      offset_bound = max(1.0, lat_span)
      if not all(math.isfinite(value) for value in (*coefficients, offset)) or gain <= 0.0 or static < kinetic or kinetic < 0.0 or abs(offset) > offset_bound:
        reasons.append(CalibrationQualificationReason.INVALID_PARAMETERS)
      else:
        comparisons = (
          (candidate_rms, seed_rms, CalibrationQualificationReason.VALIDATION_REGRESSION),
          (moving_candidate, moving_seed, CalibrationQualificationReason.MOVING_VALIDATION_REGRESSION),
          (breakaway_candidate, breakaway_seed, CalibrationQualificationReason.BREAKAWAY_VALIDATION_REGRESSION),
        )
        for candidate_value, seed_value, reason in comparisons:
          if candidate_value is not None and seed_value is not None and candidate_value > seed_value + VALIDATION_RMS_ABSOLUTE_TOLERANCE:
            reasons.append(reason)
        if (
          authority_active
          and authority_candidate is not None
          and authority_seed is not None
          and authority_candidate > authority_seed + VALIDATION_RMS_ABSOLUTE_TOLERANCE
        ):
          reasons.append(CalibrationQualificationReason.AUTHORITY_VALIDATION_REGRESSION)
        confidence = min(
          1.0,
          max(
            0.0,
            min(
              node.clean_support_s / minimum,
              node.validation.weight_s / min_validation,
              lat_span / MIN_LATERAL_ACCEL_SPAN_MPS2,
              lat_rms / MIN_LATERAL_ACCEL_RMS_MPS2,
              torque_span / MIN_APPLIED_TORQUE_SPAN,
            ),
          ),
        )
        candidate_parameters = CalibrationParameters(
          torque_per_lateral_accel=gain,
          lateral_accel_offset_correction_mps2=offset,
          kinetic_friction_torque=kinetic,
          static_breakaway_torque=static,
          transport_delay_s=seed.transport_delay_s,
          rack_rate_resolution_deg_s=seed.rack_rate_resolution_deg_s,
          confidence=confidence,
          qualified=False,
        )
    unique = tuple(dict.fromkeys(reasons))
    qualified = not unique
    if candidate_parameters is not None and qualified:
      candidate_parameters = CalibrationParameters(
        candidate_parameters.torque_per_lateral_accel,
        candidate_parameters.lateral_accel_offset_correction_mps2,
        candidate_parameters.kinetic_friction_torque,
        candidate_parameters.static_breakaway_torque,
        candidate_parameters.transport_delay_s,
        candidate_parameters.rack_rate_resolution_deg_s,
        candidate_parameters.confidence,
        True,
      )
    return CalibrationNodeQualificationReport(
      index,
      speed,
      minimum,
      node.clean_support_s,
      node.supported_sample_count,
      training.count,
      validation.count,
      validation.weight_s,
      node.base_support_s,
      node.base_sample_count,
      node.moving_support_s,
      node.moving_sample_count,
      node.moving_training.count,
      node.moving_validation.count,
      node.breakaway_support_s,
      node.breakaway_sample_count,
      node.breakaway_training.count,
      node.breakaway_validation.count,
      lat_span,
      lat_rms,
      node.rack_travel_deg,
      torque_span,
      node.rack_reversals,
      node.lateral_accel_direction_mask.bit_count(),
      node.applied_torque_direction_mask.bit_count(),
      seed_rms,
      candidate_rms,
      moving_seed,
      moving_candidate,
      breakaway_seed,
      breakaway_candidate,
      candidate_parameters.confidence if candidate_parameters is not None else 0.0,
      (CalibrationQualificationReason.QUALIFIED,) if qualified else unique,
      candidate_parameters,
      node.authority_support_s,
      node.authority_sample_count,
      node.authority_fit_support_s,
      node.authority_fit_sample_count,
      node.authority_training.count,
      node.authority_validation.count,
      authority_seed,
      authority_candidate,
      node.authority_magnitude_sample_count,
      node.authority_slew_build_sample_count,
      node.authority_slew_release_sample_count,
      node.authority_unresolved_sample_count,
    )

  def qualify(self, provenance: str) -> CalibrationLearningResult:
    source = str(provenance).strip()
    if not source:
      raise ValueError("candidate provenance must not be empty")
    reports = tuple(self._node_report(index) for index in range(len(self._nodes)))
    if not all(report.qualified for report in reports):
      return CalibrationLearningResult(reports, None)
    revision = self.seed_profile.revision + 1 + sum(node.supported_sample_count + node.authority_sample_count for node in self._nodes)
    nodes = tuple(
      CalibrationProfileNode(
        speed_mps=report.speed_mps,
        parameters=report.candidate_parameters,  # type: ignore[arg-type]
        base_support_s=report.base_support_s,
        base_sample_count=report.base_sample_count,
        moving_support_s=report.moving_support_s,
        moving_sample_count=report.moving_sample_count,
        breakaway_support_s=report.breakaway_support_s,
        breakaway_sample_count=report.breakaway_sample_count,
        validation_count=report.validation_count,
        inverse_calibration_validation_rms=report.candidate_validation_rms or 0.0,
        breakaway_validation_rms=report.breakaway_candidate_validation_rms,
      )
      for report in reports
    )
    profile = VehicleCalibrationProfile(
      self.seed_profile.vehicle_identity,
      revision,
      f"{source}; observable-inverse-torque-learner-v1; evidence_revision={revision}",
      nodes,
    )
    return CalibrationLearningResult(reports, profile)

  def export_evidence(self) -> bytes:
    seed_json = self.seed_profile.to_json()
    node_fields = (
      "clean_support_s",
      "supported_sample_count",
      "base_support_s",
      "base_sample_count",
      "moving_support_s",
      "moving_sample_count",
      "breakaway_support_s",
      "breakaway_sample_count",
      "authority_support_s",
      "authority_sample_count",
      "authority_magnitude_sample_count",
      "authority_slew_build_sample_count",
      "authority_slew_release_sample_count",
      "authority_fit_support_s",
      "authority_fit_sample_count",
      "authority_unresolved_sample_count",
      "lat_energy",
      "rack_travel_deg",
      "rack_reversals",
      "last_direction",
      "stationary_dwell_s",
      "fit_count",
      "authority_fit_count",
      "lateral_accel_direction_mask",
      "applied_torque_direction_mask",
      "moving_direction_mask",
      "breakaway_direction_mask",
    )
    regression_fields = (
      "training",
      "validation",
      "moving_training",
      "moving_validation",
      "breakaway_training",
      "breakaway_validation",
      "authority_training",
      "authority_validation",
    )
    nodes = []
    for index, node in enumerate(self._nodes):
      raw: dict[str, Any] = {
        "node_index": index,
        "lat_min": _extreme(node.lat_min),
        "lat_max": _extreme(node.lat_max),
        "torque_min": _extreme(node.torque_min),
        "torque_max": _extreme(node.torque_max),
      }
      for name in node_fields:
        value = getattr(node, name)
        raw[name] = value.hex() if type(value) is float else value
      for name in regression_fields:
        raw[name] = getattr(node, name).encoded()
      nodes.append(raw)
    payload = {
      "evidence_schema_version": CALIBRATION_EVIDENCE_SCHEMA_VERSION,
      "profile_schema_version": self.seed_profile.schema_version,
      "vehicle_identity": self.seed_profile.vehicle_identity,
      "seed_profile_json": seed_json,
      "seed_profile_sha256": hashlib.sha256(seed_json.encode()).hexdigest(),
      "speed_nodes_mps": [speed.hex() for speed in self.speed_nodes_mps],
      "nodes": nodes,
    }
    envelope = {"payload": payload, "payload_sha256": hashlib.sha256(_canonical(payload)).hexdigest()}
    return _canonical(envelope)

  @classmethod
  def from_evidence(cls, seed_profile: VehicleCalibrationProfile, encoded: bytes) -> CalibrationProfileLearner:
    if type(encoded) is not bytes:
      raise TypeError("calibration evidence must be bytes")
    try:
      decoded = json.loads(encoded, object_pairs_hook=_pairs, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
      raise ValueError("calibration evidence is invalid") from exc
    envelope = _exact(decoded, {"payload", "payload_sha256"}, "evidence envelope")
    if _canonical(envelope) != encoded:
      raise ValueError("calibration evidence is not canonical")
    payload = _exact(
      envelope["payload"],
      {"evidence_schema_version", "profile_schema_version", "vehicle_identity", "seed_profile_json", "seed_profile_sha256", "speed_nodes_mps", "nodes"},
      "evidence payload",
    )
    if payload["evidence_schema_version"] != CALIBRATION_EVIDENCE_SCHEMA_VERSION:
      raise ValueError("calibration evidence schema is incompatible")
    if payload["profile_schema_version"] != CALIBRATION_PROFILE_SCHEMA_VERSION:
      raise ValueError("calibration profile schema is incompatible")
    actual_hash = hashlib.sha256(_canonical(payload)).hexdigest()
    if type(envelope["payload_sha256"]) is not str or not hmac.compare_digest(envelope["payload_sha256"], actual_hash):
      raise ValueError("calibration evidence payload hash mismatch")
    seed_json = seed_profile.to_json()
    seed_hash = hashlib.sha256(seed_json.encode()).hexdigest()
    if payload["vehicle_identity"] != seed_profile.vehicle_identity or payload["seed_profile_json"] != seed_json:
      raise ValueError("calibration evidence belongs to a different seed")
    if type(payload["seed_profile_sha256"]) is not str or not hmac.compare_digest(payload["seed_profile_sha256"], seed_hash):
      raise ValueError("calibration seed hash mismatch")
    if payload["speed_nodes_mps"] != [speed.hex() for speed in seed_profile.speed_nodes_mps]:
      raise ValueError("calibration speed grid mismatch")
    learner = cls(seed_profile)
    raw_nodes = payload["nodes"]
    if type(raw_nodes) is not list or len(raw_nodes) != len(learner._nodes):
      raise ValueError("calibration node count mismatch")
    node_fields = (
      "clean_support_s",
      "supported_sample_count",
      "base_support_s",
      "base_sample_count",
      "moving_support_s",
      "moving_sample_count",
      "breakaway_support_s",
      "breakaway_sample_count",
      "authority_support_s",
      "authority_sample_count",
      "authority_magnitude_sample_count",
      "authority_slew_build_sample_count",
      "authority_slew_release_sample_count",
      "authority_fit_support_s",
      "authority_fit_sample_count",
      "authority_unresolved_sample_count",
      "lat_energy",
      "rack_travel_deg",
      "rack_reversals",
      "last_direction",
      "stationary_dwell_s",
      "fit_count",
      "authority_fit_count",
      "lateral_accel_direction_mask",
      "applied_torque_direction_mask",
      "moving_direction_mask",
      "breakaway_direction_mask",
    )
    float_fields = {name for name in node_fields if name.endswith("_s") or name in {"lat_energy", "rack_travel_deg"}}
    regressions = (
      "training",
      "validation",
      "moving_training",
      "moving_validation",
      "breakaway_training",
      "breakaway_validation",
      "authority_training",
      "authority_validation",
    )
    expected = {"node_index", "lat_min", "lat_max", "torque_min", "torque_max", *node_fields, *regressions}
    for index, raw in enumerate(raw_nodes):
      values = _exact(raw, expected, f"nodes[{index}]")
      if values["node_index"] != index:
        raise ValueError("calibration node ordering is corrupt")
      node = learner._nodes[index]
      node.lat_min = _decode_extreme(values["lat_min"], True, "lat_min")
      node.lat_max = _decode_extreme(values["lat_max"], False, "lat_max")
      node.torque_min = _decode_extreme(values["torque_min"], True, "torque_min")
      node.torque_max = _decode_extreme(values["torque_max"], False, "torque_max")
      for name in node_fields:
        raw_value = values[name]
        if name in float_fields:
          setattr(node, name, _hex(raw_value, name))
        elif type(raw_value) is not int or raw_value < 0 and name not in {"last_direction"}:
          raise ValueError(f"{name} is invalid")
        else:
          setattr(node, name, raw_value)
      if node.last_direction not in (-1, 0, 1):
        raise ValueError("last_direction is invalid")
      for name in regressions:
        setattr(node, name, _Regression.decoded(values[name], name))
      if node.fit_count != node.training.count + node.validation.count:
        raise ValueError("calibration fit counts are inconsistent")
      if node.authority_fit_count != node.authority_training.count + node.authority_validation.count:
        raise ValueError("authority fit counts are inconsistent")
      if node.supported_sample_count != node.base_sample_count + node.moving_sample_count + node.breakaway_sample_count:
        raise ValueError("calibration stratum counts are inconsistent")
      if node.moving_sample_count != node.moving_training.count + node.moving_validation.count:
        raise ValueError("moving evidence counts are inconsistent")
      if node.breakaway_sample_count != node.breakaway_training.count + node.breakaway_validation.count:
        raise ValueError("breakaway evidence counts are inconsistent")
      if node.authority_fit_sample_count != node.authority_fit_count or node.authority_fit_sample_count > node.authority_sample_count:
        raise ValueError("authority sample counts are inconsistent")
      if node.authority_unresolved_sample_count > node.authority_magnitude_sample_count:
        raise ValueError("authority unresolved count is inconsistent")
      support_sum = node.base_support_s + node.moving_support_s + node.breakaway_support_s
      if not math.isclose(node.clean_support_s, support_sum, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("calibration stratum support is inconsistent")
      if node.authority_fit_support_s > node.authority_support_s + 1e-12:
        raise ValueError("authority fit support exceeds total authority support")
      if any(
        getattr(node, name) < 0.0
        for name in (
          "clean_support_s",
          "base_support_s",
          "moving_support_s",
          "breakaway_support_s",
          "authority_support_s",
          "authority_fit_support_s",
          "stationary_dwell_s",
          "lat_energy",
          "rack_travel_deg",
        )
      ):
        raise ValueError("calibration support or energy is negative")
      if any(
        getattr(node, name) not in range(4)
        for name in (
          "lateral_accel_direction_mask",
          "applied_torque_direction_mask",
          "moving_direction_mask",
          "breakaway_direction_mask",
        )
      ):
        raise ValueError("calibration direction mask is invalid")
      lat_empty = math.isinf(node.lat_min) and node.lat_min > 0.0 and math.isinf(node.lat_max) and node.lat_max < 0.0
      torque_empty = math.isinf(node.torque_min) and node.torque_min > 0.0 and math.isinf(node.torque_max) and node.torque_max < 0.0
      if not (lat_empty or math.isfinite(node.lat_min) and math.isfinite(node.lat_max) and node.lat_min <= node.lat_max):
        raise ValueError("lateral-acceleration extrema are inconsistent")
      if not (torque_empty or math.isfinite(node.torque_min) and math.isfinite(node.torque_max) and node.torque_min <= node.torque_max):
        raise ValueError("applied-torque extrema are inconsistent")
      if (lat_empty and node.lateral_accel_direction_mask != 0) or (torque_empty and node.applied_torque_direction_mask != 0):
        raise ValueError("empty calibration extrema carry a direction mask")
    return learner


ObservableCalibrationLearner = CalibrationProfileLearner


def calibration_learning_sample_field_names() -> tuple[str, ...]:
  """Expose the unchanged measured-only contract for audits."""
  return (
    "speed_mps",
    "dt_s",
    "applied_torque",
    "measured_lateral_accel_mps2",
    "rack_rate_deg_s",
    "rack_acceleration_deg_s2",
    "engaged",
    "valid",
    "steering_pressed",
    "actuator_constrained",
    "standstill",
    "rack_direction_reversal",
  )
