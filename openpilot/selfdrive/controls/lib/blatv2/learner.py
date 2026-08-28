"""Deterministic offroad physical-profile learner for modular BLaTv2.

This module consumes measured response only. Desired path, controller request,
and candidate-controller output are deliberately absent from ``LearningSample``
so the learner cannot turn a quality error into a hidden live torque modifier.

For each speed node, moving-rack samples fit the linear inverse plant

  applied_torque
    = -torque_per_lateral_accel * measured_lateral_accel
    + rack_acceleration / rack_gain
    + rack_damping * rack_rate / rack_gain
    + kinetic_friction * sign(rack_rate)

using deterministic scalar normal equations. Applied torque is causally
aligned to rack response with the immutable seed transport delay before a row
reaches this module. Static friction, transport delay, and measured rack-rate
resolution remain exactly the seed values.
Moving friction is independently fit from already-moving samples. A candidate
must improve or match the complete seed model on held-out chronological
blocks. Driver-free limiter boundaries are retained separately: slew
transients are authority observations, while settled full-magnitude motion may
join the fit only after the seed transport delay and with resolved rack motion.
Free and authority validation are checked independently.

Readiness thresholds below describe data coverage and matrix validity, not
steering feel. Learning never changes the live profile and promotion remains a
separate engagement-boundary operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from enum import IntFlag, StrEnum
import hashlib
import hmac
import json
import math
import struct
from typing import Any

from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  PROFILE_SCHEMA_VERSION,
  PhysicalParameters,
  ProfileNode,
  VehicleProfile,
)


# Clean support floors are deliberately in the middle of the documented
# exposure ranges: low 2-3 min, urban/mid 3-5 min, highway 5-10 min.
LOW_SPEED_MAX_MPS = 5.0
MID_SPEED_MAX_MPS = 15.0
LOW_SPEED_MIN_CLEAN_SUPPORT_S = 150.0
MID_SPEED_MIN_CLEAN_SUPPORT_S = 240.0
HIGH_SPEED_MIN_CLEAN_SUPPORT_S = 420.0

# Alternating local chronological blocks prevent adjacent route samples from
# leaking between fit and validation. The count is a partitioning constant,
# not a controller or feel parameter.
TRAIN_VALIDATION_BLOCK_SAMPLES = 128
MIN_VALIDATION_SUPPORT_FRACTION = 0.20
MIN_AUTHORITY_TRAINING_SAMPLES = 4
MIN_AUTHORITY_VALIDATION_SAMPLES = 4

# A sample gap longer than ten 100 Hz frames is not continuous evidence and
# cannot be allowed to manufacture minutes of support.
MAX_CLEAN_SAMPLE_DT_S = 0.10
MAX_NORMALIZED_TORQUE = 1.0

# Excitation minima establish that all four regression columns are exercised.
# They carry physical units and are intentionally unrelated to controller
# quality metrics or desired-path tracking.
MIN_LATERAL_ACCEL_SPAN_MPS2 = 0.80
MIN_LATERAL_ACCEL_RMS_MPS2 = 0.20
MIN_RACK_TRAVEL_DEG = 180.0
MIN_APPLIED_TORQUE_SPAN = 0.15
MIN_RACK_DIRECTION_REVERSALS = 4
MIN_EXCITATION_NODE_WEIGHT = 0.10

# Scaled normal equations make this a dimensionless numerical-rank test.
NORMAL_MATRIX_RELATIVE_PIVOT_MIN = 1e-10
VALIDATION_RMS_ABSOLUTE_TOLERANCE = 1e-12

# Evidence is not a learned profile. This independent schema identifies the
# exact sufficient statistics needed to continue fitting across drives.
LEARNING_EVIDENCE_SCHEMA_VERSION = 4

_EMPTY_POSITIVE_SENTINEL = {"empty": "positive_infinity"}
_EMPTY_NEGATIVE_SENTINEL = {"empty": "negative_infinity"}


class QualificationReason(StrEnum):
  QUALIFIED = "qualified"
  INSUFFICIENT_SUPPORT = "insufficient_support"
  INSUFFICIENT_VALIDATION = "insufficient_validation"
  INSUFFICIENT_EXCITATION = "insufficient_excitation"
  SINGULAR_FIT = "singular_fit"
  INVALID_PARAMETERS = "invalid_parameters"
  CROSS_FIT_REGRESSION = "cross_fit_regression"
  AUTHORITY_CROSS_FIT_REGRESSION = "authority_cross_fit_regression"


class ActuatorBoundary(IntFlag):
  """Measured vehicle-owned torque-envelope boundary classification."""

  NONE = 0
  MAGNITUDE = 1
  SLEW_BUILD = 2
  SLEW_RELEASE = 4
  DRIVER = 8


_KNOWN_ACTUATOR_BOUNDARIES = (
  ActuatorBoundary.MAGNITUDE
  | ActuatorBoundary.SLEW_BUILD
  | ActuatorBoundary.SLEW_RELEASE
  | ActuatorBoundary.DRIVER
)


@dataclass(frozen=True, slots=True)
class LearningSample:
  """One measured physical-response sample; no desired-path field exists."""

  speed_mps: float
  dt_s: float
  applied_torque: float
  measured_lateral_accel_mps2: float
  rack_rate_deg_s: float
  rack_acceleration_deg_s2: float
  engaged: bool
  valid: bool
  steering_pressed: bool
  actuator_constrained: bool
  standstill: bool
  # A signed physical reversal is valid coverage evidence, but the
  # quantized sign-crossing acceleration is not a plant-regression row.
  rack_direction_reversal: bool = False
  # Raw physical steering-angle sensor coordinate.  It is retained solely so
  # the calibration learner can observe sub-rate-quantum rack motion; desired
  # angle, model path, and controller request remain absent from this contract.
  measured_rack_angle_deg: float = 0.0
  actuator_boundary: ActuatorBoundary = field(
    default=ActuatorBoundary.NONE,
    init=False,
  )
  magnitude_boundary_dwell_s: float = field(default=0.0, init=False)
  _authority_attested: bool = field(
    default=False,
    init=False,
    repr=False,
    compare=False,
  )

  @property
  def _base_valid(self) -> bool:
    numeric_values = (
      self.speed_mps,
      self.dt_s,
      self.applied_torque,
      self.measured_lateral_accel_mps2,
      self.rack_rate_deg_s,
      self.rack_acceleration_deg_s2,
      self.measured_rack_angle_deg,
      self.magnitude_boundary_dwell_s,
    )
    return (
      all(math.isfinite(value) for value in numeric_values)
      and self.speed_mps >= 0.0
      and 0.0 < self.dt_s <= MAX_CLEAN_SAMPLE_DT_S
      and abs(self.applied_torque) <= MAX_NORMALIZED_TORQUE
      and self.magnitude_boundary_dwell_s >= 0.0
      and self.engaged
      and self.valid
      and not self.steering_pressed
      and not self.standstill
      and not (
        int(self.actuator_boundary)
        & ~int(_KNOWN_ACTUATOR_BOUNDARIES)
      )
      and (
        not bool(self.actuator_boundary & ActuatorBoundary.MAGNITUDE)
        or abs(abs(self.applied_torque) - 1.0) <= 1e-6
      )
      and (
        bool(self.actuator_boundary & ActuatorBoundary.MAGNITUDE)
        or self.magnitude_boundary_dwell_s == 0.0
      )
    )

  @property
  def clean(self) -> bool:
    """Whether this sample may enter the unconstrained equality regression."""
    return (
      self._base_valid
      and not self.rack_direction_reversal
      and not self.actuator_constrained
      and self.actuator_boundary == ActuatorBoundary.NONE
      and self.magnitude_boundary_dwell_s == 0.0
    )

  @property
  def authority_evidence(self) -> bool:
    """Whether this is valid, driver-free vehicle-boundary evidence."""
    return (
      self._base_valid
      and not self.rack_direction_reversal
      and self._authority_attested
      and self.actuator_constrained
      and self.actuator_boundary != ActuatorBoundary.NONE
      and not bool(self.actuator_boundary & ActuatorBoundary.DRIVER)
      and (
        bool(self.actuator_boundary & ActuatorBoundary.MAGNITUDE)
        or self.magnitude_boundary_dwell_s == 0.0
      )
    )


def _attest_authority_sample(
  sample: LearningSample,
  *,
  boundary: ActuatorBoundary,
  magnitude_boundary_dwell_s: float,
) -> LearningSample:
  """Attach runtime-classified authority facts to one physical sample.

  This is deliberately private. Public measurement callers can flag a sample
  as constrained, but only the exact runtime envelope classifier may attest
  its boundary kind and dwell for authority accumulation or fitting.
  """
  if not isinstance(sample, LearningSample):
    raise TypeError("authority attestation requires a LearningSample")
  if (
    not sample.actuator_constrained
    or boundary == ActuatorBoundary.NONE
    or bool(boundary & ActuatorBoundary.DRIVER)
    or int(boundary) & ~int(_KNOWN_ACTUATOR_BOUNDARIES)
    or not math.isfinite(magnitude_boundary_dwell_s)
    or magnitude_boundary_dwell_s < 0.0
  ):
    raise ValueError("authority attestation is not driver-free boundary data")
  if (
    not bool(boundary & ActuatorBoundary.MAGNITUDE)
    and magnitude_boundary_dwell_s != 0.0
  ):
    raise ValueError("only a magnitude boundary may carry dwell")

  attested = replace(sample)
  object.__setattr__(attested, "actuator_boundary", boundary)
  object.__setattr__(
    attested,
    "magnitude_boundary_dwell_s",
    float(magnitude_boundary_dwell_s),
  )
  object.__setattr__(attested, "_authority_attested", True)
  return attested


@dataclass(frozen=True, slots=True)
class NodeEvidenceSnapshot:
  """Immutable diagnostics for proving node-local evidence isolation."""

  clean_support_s: float
  supported_sample_count: int
  authority_support_s: float
  authority_sample_count: int
  authority_magnitude_sample_count: int
  authority_slew_build_sample_count: int
  authority_slew_release_sample_count: int
  authority_fit_support_s: float
  authority_fit_sample_count: int
  authority_unresolved_sample_count: int
  lateral_accel_min_mps2: float
  lateral_accel_max_mps2: float
  lateral_accel_energy_mps4_s: float
  rack_travel_deg: float
  applied_torque_min: float
  applied_torque_max: float
  rack_reversals: int
  last_rack_direction: int
  training_values: tuple[float, ...]
  validation_values: tuple[float, ...]
  authority_training_values: tuple[float, ...]
  authority_validation_values: tuple[float, ...]

  def to_bytes(self) -> bytes:
    scalar_values = (
      self.clean_support_s,
      float(self.supported_sample_count),
      self.authority_support_s,
      float(self.authority_sample_count),
      float(self.authority_magnitude_sample_count),
      float(self.authority_slew_build_sample_count),
      float(self.authority_slew_release_sample_count),
      self.authority_fit_support_s,
      float(self.authority_fit_sample_count),
      float(self.authority_unresolved_sample_count),
      self.lateral_accel_min_mps2,
      self.lateral_accel_max_mps2,
      self.lateral_accel_energy_mps4_s,
      self.rack_travel_deg,
      self.applied_torque_min,
      self.applied_torque_max,
      float(self.rack_reversals),
      float(self.last_rack_direction),
      *self.training_values,
      *self.validation_values,
      *self.authority_training_values,
      *self.authority_validation_values,
    )
    return struct.pack(f"<{len(scalar_values)}d", *scalar_values)


@dataclass(frozen=True, slots=True)
class NodeQualificationReport:
  node_index: int
  speed_mps: float
  minimum_support_s: float
  clean_support_s: float
  supported_sample_count: int
  training_count: int
  validation_count: int
  validation_support_s: float
  lateral_accel_span_mps2: float
  lateral_accel_rms_mps2: float
  rack_travel_deg: float
  applied_torque_span: float
  rack_reversals: int
  seed_full_fit_candidate_rms: float | None
  candidate_full_fit_candidate_rms: float | None
  confidence: float
  reasons: tuple[QualificationReason, ...]
  candidate_parameters: PhysicalParameters | None
  authority_support_s: float = 0.0
  authority_sample_count: int = 0
  authority_fit_support_s: float = 0.0
  authority_fit_sample_count: int = 0
  authority_training_count: int = 0
  authority_validation_count: int = 0
  authority_fit_active: bool = False
  authority_seed_full_fit_candidate_rms: float | None = None
  authority_candidate_full_fit_candidate_rms: float | None = None

  @property
  def qualified(self) -> bool:
    return self.reasons == (QualificationReason.QUALIFIED,)


@dataclass(frozen=True, slots=True)
class LearningResult:
  node_reports: tuple[NodeQualificationReport, ...]
  candidate_profile: VehicleProfile | None

  @property
  def all_nodes_qualified(self) -> bool:
    return self.candidate_profile is not None


class _RegressionEvidence:
  __slots__ = ("normal", "rhs", "target_squared", "weight_s", "count")

  def __init__(self) -> None:
    self.normal = [0.0] * 16
    self.rhs = [0.0] * 4
    self.target_squared = 0.0
    self.weight_s = 0.0
    self.count = 0

  def add(
    self,
    predictors: tuple[float, float, float, float],
    target: float,
    weight_s: float,
  ) -> None:
    for row in range(4):
      self.rhs[row] += weight_s * predictors[row] * target
      for column in range(4):
        self.normal[row * 4 + column] += (
          weight_s * predictors[row] * predictors[column]
        )
    self.target_squared += weight_s * target * target
    self.weight_s += weight_s
    self.count += 1

  def values(self) -> tuple[float, ...]:
    return (
      *self.normal,
      *self.rhs,
      self.target_squared,
      self.weight_s,
      float(self.count),
    )

  def rms(
    self,
    coefficients: tuple[float, float, float, float],
  ) -> float | None:
    if self.weight_s <= 0.0:
      return None
    quadratic = 0.0
    linear = 0.0
    for row in range(4):
      linear += coefficients[row] * self.rhs[row]
      for column in range(4):
        quadratic += (
          coefficients[row]
          * self.normal[row * 4 + column]
          * coefficients[column]
        )
    squared_error = self.target_squared - 2.0 * linear + quadratic
    roundoff_scale = max(self.target_squared, quadratic, 1.0)
    if squared_error < 0.0 and abs(squared_error) <= 1e-12 * roundoff_scale:
      squared_error = 0.0
    if squared_error < 0.0 or not math.isfinite(squared_error):
      return None
    return math.sqrt(squared_error / self.weight_s)


def _combined_regression(
  primary: _RegressionEvidence,
  authority: _RegressionEvidence,
) -> _RegressionEvidence:
  """Combine fixed-order sufficient statistics without mutating evidence."""
  combined = _RegressionEvidence()
  for index in range(16):
    combined.normal[index] = (
      primary.normal[index] + authority.normal[index]
    )
  for index in range(4):
    combined.rhs[index] = primary.rhs[index] + authority.rhs[index]
  combined.target_squared = (
    primary.target_squared + authority.target_squared
  )
  combined.weight_s = primary.weight_s + authority.weight_s
  combined.count = primary.count + authority.count
  return combined


class _NodeAccumulator:
  __slots__ = (
    "clean_support_s",
    "supported_sample_count",
    "authority_support_s",
    "authority_sample_count",
    "authority_magnitude_sample_count",
    "authority_slew_build_sample_count",
    "authority_slew_release_sample_count",
    "authority_fit_support_s",
    "authority_fit_sample_count",
    "authority_unresolved_sample_count",
    "lateral_accel_min",
    "lateral_accel_max",
    "lateral_accel_energy",
    "rack_travel_deg",
    "applied_torque_min",
    "applied_torque_max",
    "rack_reversals",
    "last_rack_direction",
    "training",
    "validation",
    "authority_training",
    "authority_validation",
  )

  def __init__(self) -> None:
    self.clean_support_s = 0.0
    self.supported_sample_count = 0
    self.authority_support_s = 0.0
    self.authority_sample_count = 0
    self.authority_magnitude_sample_count = 0
    self.authority_slew_build_sample_count = 0
    self.authority_slew_release_sample_count = 0
    self.authority_fit_support_s = 0.0
    self.authority_fit_sample_count = 0
    self.authority_unresolved_sample_count = 0
    self.lateral_accel_min = math.inf
    self.lateral_accel_max = -math.inf
    self.lateral_accel_energy = 0.0
    self.rack_travel_deg = 0.0
    self.applied_torque_min = math.inf
    self.applied_torque_max = -math.inf
    self.rack_reversals = 0
    self.last_rack_direction = 0
    self.training = _RegressionEvidence()
    self.validation = _RegressionEvidence()
    self.authority_training = _RegressionEvidence()
    self.authority_validation = _RegressionEvidence()

  def snapshot(self) -> NodeEvidenceSnapshot:
    return NodeEvidenceSnapshot(
      clean_support_s=self.clean_support_s,
      supported_sample_count=self.supported_sample_count,
      authority_support_s=self.authority_support_s,
      authority_sample_count=self.authority_sample_count,
      authority_magnitude_sample_count=(
        self.authority_magnitude_sample_count
      ),
      authority_slew_build_sample_count=(
        self.authority_slew_build_sample_count
      ),
      authority_slew_release_sample_count=(
        self.authority_slew_release_sample_count
      ),
      authority_fit_support_s=self.authority_fit_support_s,
      authority_fit_sample_count=self.authority_fit_sample_count,
      authority_unresolved_sample_count=(
        self.authority_unresolved_sample_count
      ),
      lateral_accel_min_mps2=self.lateral_accel_min,
      lateral_accel_max_mps2=self.lateral_accel_max,
      lateral_accel_energy_mps4_s=self.lateral_accel_energy,
      rack_travel_deg=self.rack_travel_deg,
      applied_torque_min=self.applied_torque_min,
      applied_torque_max=self.applied_torque_max,
      rack_reversals=self.rack_reversals,
      last_rack_direction=self.last_rack_direction,
      training_values=self.training.values(),
      validation_values=self.validation.values(),
      authority_training_values=self.authority_training.values(),
      authority_validation_values=self.authority_validation.values(),
    )


def _canonical_json_bytes(payload: object) -> bytes:
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
  ).encode("utf-8")


def _reject_duplicate_keys(
  pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f"duplicate evidence key: {key}")
    result[key] = value
  return result


def _reject_non_json_number(value: str) -> None:
  raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _require_exact_keys(
  payload: object,
  expected: frozenset[str],
  context: str,
) -> dict[str, Any]:
  if type(payload) is not dict:
    raise ValueError(f"{context} must be an object")
  actual = frozenset(payload)
  if actual != expected:
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    raise ValueError(
      f"{context} keys mismatch; missing={missing}, unknown={unknown}",
    )
  return payload


def _require_int(
  value: object,
  context: str,
  *,
  minimum: int = 0,
) -> int:
  if type(value) is not int or value < minimum:
    raise ValueError(f"{context} must be an integer >= {minimum}")
  return value


def _encode_finite_float(value: float) -> str:
  numeric = float(value)
  if not math.isfinite(numeric):
    raise ValueError("non-finite learner evidence is not serializable")
  return numeric.hex()


def _decode_finite_float(value: object, context: str) -> float:
  if type(value) is not str:
    raise ValueError(f"{context} must be an exact hexadecimal float")
  try:
    numeric = float.fromhex(value)
  except ValueError as exc:
    raise ValueError(
      f"{context} is not a hexadecimal float",
    ) from exc
  if not math.isfinite(numeric) or numeric.hex() != value:
    raise ValueError(f"{context} is not a canonical finite float")
  return numeric


def _encode_extreme(value: float) -> str | dict[str, str]:
  numeric = float(value)
  if math.isfinite(numeric):
    return numeric.hex()
  if numeric == math.inf:
    return dict(_EMPTY_POSITIVE_SENTINEL)
  if numeric == -math.inf:
    return dict(_EMPTY_NEGATIVE_SENTINEL)
  raise ValueError("NaN learner extrema are not serializable")


def _decode_extreme(
  value: object,
  context: str,
) -> float:
  if type(value) is str:
    return _decode_finite_float(value, context)
  sentinel = _require_exact_keys(
    value,
    frozenset({"empty"}),
    context,
  )
  marker = sentinel["empty"]
  if marker == _EMPTY_POSITIVE_SENTINEL["empty"]:
    return math.inf
  if marker == _EMPTY_NEGATIVE_SENTINEL["empty"]:
    return -math.inf
  raise ValueError(f"{context} has an unknown empty sentinel")


def _encode_regression(
  evidence: _RegressionEvidence,
) -> dict[str, Any]:
  return {
    "count": evidence.count,
    "normal": [_encode_finite_float(value) for value in evidence.normal],
    "rhs": [_encode_finite_float(value) for value in evidence.rhs],
    "target_squared": _encode_finite_float(evidence.target_squared),
    "weight_s": _encode_finite_float(evidence.weight_s),
  }


def _decode_float_list(
  value: object,
  expected_length: int,
  context: str,
) -> list[float]:
  if type(value) is not list or len(value) != expected_length:
    raise ValueError(
      f"{context} must contain exactly {expected_length} values",
    )
  return [
    _decode_finite_float(item, f"{context}[{index}]")
    for index, item in enumerate(value)
  ]


def _restore_regression(
  raw: object,
  context: str,
) -> _RegressionEvidence:
  payload = _require_exact_keys(
    raw,
    frozenset({
      "count",
      "normal",
      "rhs",
      "target_squared",
      "weight_s",
    }),
    context,
  )
  normal = _decode_float_list(payload["normal"], 16, f"{context}.normal")
  rhs = _decode_float_list(payload["rhs"], 4, f"{context}.rhs")
  target_squared = _decode_finite_float(
    payload["target_squared"],
    f"{context}.target_squared",
  )
  weight_s = _decode_finite_float(
    payload["weight_s"],
    f"{context}.weight_s",
  )
  count = _require_int(payload["count"], f"{context}.count")
  if target_squared < 0.0 or weight_s < 0.0:
    raise ValueError(f"{context} weights must be non-negative")
  if count == 0 and (
    target_squared != 0.0
    or weight_s != 0.0
    or any(value != 0.0 for value in (*normal, *rhs))
  ):
    raise ValueError(f"{context} empty evidence must be all zero")
  if count > 0 and weight_s <= 0.0:
    raise ValueError(f"{context} populated evidence needs positive weight")

  evidence = _RegressionEvidence()
  evidence.normal[:] = normal
  evidence.rhs[:] = rhs
  evidence.target_squared = target_squared
  evidence.weight_s = weight_s
  evidence.count = count
  return evidence


def _validate_extrema_pair(
  minimum: float,
  maximum: float,
  context: str,
) -> None:
  empty = minimum == math.inf and maximum == -math.inf
  finite = (
    math.isfinite(minimum)
    and math.isfinite(maximum)
    and minimum <= maximum
  )
  if not empty and not finite:
    raise ValueError(
      f"{context} must be a finite ordered pair or an explicit empty pair",
    )


def _seed_profile_sha256(seed_profile: VehicleProfile) -> str:
  return hashlib.sha256(
    seed_profile.to_json().encode("utf-8"),
  ).hexdigest()


def learner_evidence_sha256(encoded: bytes) -> str:
  """Return the identity of the complete canonical evidence artifact."""
  if type(encoded) is not bytes:
    raise TypeError("learner evidence identity requires bytes")
  return hashlib.sha256(encoded).hexdigest()


def minimum_clean_support_s(speed_mps: float) -> float:
  """Return the documented evidence floor for one profile node."""
  speed = float(speed_mps)
  if not math.isfinite(speed) or speed < 0.0:
    raise ValueError("profile-node speed must be finite and non-negative")
  if speed <= LOW_SPEED_MAX_MPS:
    return LOW_SPEED_MIN_CLEAN_SUPPORT_S
  if speed <= MID_SPEED_MAX_MPS:
    return MID_SPEED_MIN_CLEAN_SUPPORT_S
  return HIGH_SPEED_MIN_CLEAN_SUPPORT_S


def _solve_scaled_normal_equations(
  evidence: _RegressionEvidence,
) -> tuple[float, float, float, float] | None:
  scales = []
  for index in range(4):
    diagonal = evidence.normal[index * 4 + index]
    if not math.isfinite(diagonal) or diagonal <= 0.0:
      return None
    scales.append(math.sqrt(diagonal))

  matrix = [[0.0] * 5 for _ in range(4)]
  for row in range(4):
    for column in range(4):
      matrix[row][column] = (
        evidence.normal[row * 4 + column]
        / (scales[row] * scales[column])
      )
    matrix[row][4] = evidence.rhs[row] / scales[row]

  for pivot_column in range(4):
    pivot_row = pivot_column
    pivot_magnitude = abs(matrix[pivot_row][pivot_column])
    for candidate_row in range(pivot_column + 1, 4):
      candidate_magnitude = abs(matrix[candidate_row][pivot_column])
      if candidate_magnitude > pivot_magnitude:
        pivot_row = candidate_row
        pivot_magnitude = candidate_magnitude
    if (
      not math.isfinite(pivot_magnitude)
      or pivot_magnitude < NORMAL_MATRIX_RELATIVE_PIVOT_MIN
    ):
      return None
    if pivot_row != pivot_column:
      matrix[pivot_column], matrix[pivot_row] = (
        matrix[pivot_row],
        matrix[pivot_column],
      )
    pivot = matrix[pivot_column][pivot_column]
    for column in range(pivot_column, 5):
      matrix[pivot_column][column] /= pivot
    for row in range(4):
      if row == pivot_column:
        continue
      factor = matrix[row][pivot_column]
      for column in range(pivot_column, 5):
        matrix[row][column] -= factor * matrix[pivot_column][column]

  scaled_solution = [matrix[index][4] for index in range(4)]
  coefficients = tuple(
    scaled_solution[index] / scales[index] for index in range(4)
  )
  if not all(math.isfinite(value) for value in coefficients):
    return None
  return coefficients


def _seed_coefficients(
  parameters: PhysicalParameters,
) -> tuple[float, float, float, float]:
  inverse_gain = 1.0 / parameters.rack_gain_deg_s2_per_torque
  return (
    parameters.torque_per_lateral_accel,
    inverse_gain,
    parameters.rack_damping_per_s * inverse_gain,
    parameters.kinetic_friction_torque,
  )


def _span(minimum: float, maximum: float) -> float:
  if not math.isfinite(minimum) or not math.isfinite(maximum):
    return 0.0
  return maximum - minimum


class ProfileLearner:
  """Node-local evidence accumulator and offroad qualification engine."""

  def __init__(self, seed_profile: VehicleProfile) -> None:
    if not isinstance(seed_profile, VehicleProfile):
      raise TypeError(
        "learner accepts only the current VehicleProfile schema",
      )
    if seed_profile.schema_version != PROFILE_SCHEMA_VERSION:
      raise ValueError("learner seed profile schema is incompatible")
    self.seed_profile = seed_profile
    self._nodes = tuple(_NodeAccumulator() for _ in seed_profile.nodes)

  @property
  def speed_nodes_mps(self) -> tuple[float, ...]:
    return self.seed_profile.speed_nodes_mps

  def evidence_for_node(self, node_index: int) -> NodeEvidenceSnapshot:
    return self._nodes[node_index].snapshot()

  def reset_route_transients(self) -> None:
    """Break reversal continuity without changing cumulative evidence."""
    for node in self._nodes:
      node.last_rack_direction = 0

  def export_evidence(self) -> bytes:
    """Export exact fitting evidence as a canonical, self-verifying artifact."""
    seed_profile_json = self.seed_profile.to_json()
    payload = {
      "evidence_schema_version": LEARNING_EVIDENCE_SCHEMA_VERSION,
      "nodes": [
        {
          "applied_torque_max": _encode_extreme(
            node.applied_torque_max,
          ),
          "applied_torque_min": _encode_extreme(
            node.applied_torque_min,
          ),
          "clean_support_s": _encode_finite_float(
            node.clean_support_s,
          ),
          "authority_fit_sample_count": node.authority_fit_sample_count,
          "authority_fit_support_s": _encode_finite_float(
            node.authority_fit_support_s,
          ),
          "authority_magnitude_sample_count": (
            node.authority_magnitude_sample_count
          ),
          "authority_sample_count": node.authority_sample_count,
          "authority_slew_build_sample_count": (
            node.authority_slew_build_sample_count
          ),
          "authority_slew_release_sample_count": (
            node.authority_slew_release_sample_count
          ),
          "authority_support_s": _encode_finite_float(
            node.authority_support_s,
          ),
          "authority_training": _encode_regression(
            node.authority_training,
          ),
          "authority_unresolved_sample_count": (
            node.authority_unresolved_sample_count
          ),
          "authority_validation": _encode_regression(
            node.authority_validation,
          ),
          "last_rack_direction": node.last_rack_direction,
          "lateral_accel_energy_mps4_s": _encode_finite_float(
            node.lateral_accel_energy,
          ),
          "lateral_accel_max_mps2": _encode_extreme(
            node.lateral_accel_max,
          ),
          "lateral_accel_min_mps2": _encode_extreme(
            node.lateral_accel_min,
          ),
          "node_index": node_index,
          "rack_reversals": node.rack_reversals,
          "rack_travel_deg": _encode_finite_float(
            node.rack_travel_deg,
          ),
          "supported_sample_count": node.supported_sample_count,
          "training": _encode_regression(node.training),
          "validation": _encode_regression(node.validation),
        }
        for node_index, node in enumerate(self._nodes)
      ],
      "profile_schema_version": self.seed_profile.schema_version,
      "seed_profile_json": seed_profile_json,
      "seed_profile_sha256": _seed_profile_sha256(self.seed_profile),
      "speed_nodes_mps": [
        _encode_finite_float(speed)
        for speed in self.speed_nodes_mps
      ],
      "vehicle_identity": self.seed_profile.vehicle_identity,
    }
    payload_bytes = _canonical_json_bytes(payload)
    envelope = {
      "payload": payload,
      "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
    }
    return _canonical_json_bytes(envelope)

  @classmethod
  def from_evidence(
    cls,
    seed_profile: VehicleProfile,
    encoded: bytes,
  ) -> ProfileLearner:
    """Restore exact sufficient statistics for the exact supplied seed."""
    if not isinstance(seed_profile, VehicleProfile):
      raise TypeError(
        "learner evidence requires the current VehicleProfile schema",
      )
    if seed_profile.schema_version != PROFILE_SCHEMA_VERSION:
      raise ValueError("learner evidence seed profile schema is incompatible")
    if type(encoded) is not bytes:
      raise TypeError("learner evidence must be canonical bytes")
    try:
      decoded = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_json_number,
      )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
      raise ValueError("learner evidence is not valid UTF-8 JSON") from exc
    envelope = _require_exact_keys(
      decoded,
      frozenset({"payload", "payload_sha256"}),
      "evidence envelope",
    )
    if _canonical_json_bytes(envelope) != encoded:
      raise ValueError("learner evidence is not canonically encoded")

    stored_payload_hash = envelope["payload_sha256"]
    if type(stored_payload_hash) is not str:
      raise ValueError("payload_sha256 must be a string")
    payload = _require_exact_keys(
      envelope["payload"],
      frozenset({
        "evidence_schema_version",
        "nodes",
        "profile_schema_version",
        "seed_profile_json",
        "seed_profile_sha256",
        "speed_nodes_mps",
        "vehicle_identity",
      }),
      "evidence payload",
    )
    actual_payload_hash = hashlib.sha256(
      _canonical_json_bytes(payload),
    ).hexdigest()
    if not hmac.compare_digest(stored_payload_hash, actual_payload_hash):
      raise ValueError("learner evidence payload hash mismatch")

    evidence_schema = _require_int(
      payload["evidence_schema_version"],
      "evidence_schema_version",
    )
    if evidence_schema != LEARNING_EVIDENCE_SCHEMA_VERSION:
      raise ValueError("learner evidence schema is incompatible")
    profile_schema = _require_int(
      payload["profile_schema_version"],
      "profile_schema_version",
    )
    if profile_schema != PROFILE_SCHEMA_VERSION:
      raise ValueError("learner evidence profile schema is incompatible")
    if profile_schema != seed_profile.schema_version:
      raise ValueError("learner evidence seed schema mismatch")

    vehicle_identity = payload["vehicle_identity"]
    if type(vehicle_identity) is not str:
      raise ValueError("vehicle_identity must be a string")
    if vehicle_identity != seed_profile.vehicle_identity:
      raise ValueError("learner evidence belongs to a different vehicle")

    raw_speed_nodes = payload["speed_nodes_mps"]
    if type(raw_speed_nodes) is not list:
      raise ValueError("speed_nodes_mps must be a list")
    decoded_speed_nodes = tuple(
      _decode_finite_float(
        value,
        f"speed_nodes_mps[{index}]",
      )
      for index, value in enumerate(raw_speed_nodes)
    )
    if decoded_speed_nodes != seed_profile.speed_nodes_mps:
      raise ValueError("learner evidence speed-node grid mismatch")

    seed_profile_json = payload["seed_profile_json"]
    seed_profile_hash = payload["seed_profile_sha256"]
    if type(seed_profile_json) is not str or type(seed_profile_hash) is not str:
      raise ValueError("seed profile content and hash must be strings")
    stored_seed_hash = hashlib.sha256(
      seed_profile_json.encode("utf-8"),
    ).hexdigest()
    if not hmac.compare_digest(seed_profile_hash, stored_seed_hash):
      raise ValueError("learner evidence seed-profile hash is corrupt")
    expected_seed_json = seed_profile.to_json()
    expected_seed_hash = _seed_profile_sha256(seed_profile)
    if (
      seed_profile_json != expected_seed_json
      or not hmac.compare_digest(seed_profile_hash, expected_seed_hash)
    ):
      raise ValueError("learner evidence belongs to a different seed profile")

    raw_nodes = payload["nodes"]
    if (
      type(raw_nodes) is not list
      or len(raw_nodes) != len(seed_profile.nodes)
    ):
      raise ValueError("learner evidence node count mismatch")

    learner = cls(seed_profile)
    node_keys = frozenset({
      "applied_torque_max",
      "applied_torque_min",
      "authority_fit_sample_count",
      "authority_fit_support_s",
      "authority_magnitude_sample_count",
      "authority_sample_count",
      "authority_slew_build_sample_count",
      "authority_slew_release_sample_count",
      "authority_support_s",
      "authority_training",
      "authority_unresolved_sample_count",
      "authority_validation",
      "clean_support_s",
      "last_rack_direction",
      "lateral_accel_energy_mps4_s",
      "lateral_accel_max_mps2",
      "lateral_accel_min_mps2",
      "node_index",
      "rack_reversals",
      "rack_travel_deg",
      "supported_sample_count",
      "training",
      "validation",
    })
    for node_index, raw_node in enumerate(raw_nodes):
      node_payload = _require_exact_keys(
        raw_node,
        node_keys,
        f"nodes[{node_index}]",
      )
      stored_index = _require_int(
        node_payload["node_index"],
        f"nodes[{node_index}].node_index",
      )
      if stored_index != node_index:
        raise ValueError("learner evidence node ordering is corrupt")

      clean_support_s = _decode_finite_float(
        node_payload["clean_support_s"],
        f"nodes[{node_index}].clean_support_s",
      )
      supported_sample_count = _require_int(
        node_payload["supported_sample_count"],
        f"nodes[{node_index}].supported_sample_count",
      )
      authority_support_s = _decode_finite_float(
        node_payload["authority_support_s"],
        f"nodes[{node_index}].authority_support_s",
      )
      authority_sample_count = _require_int(
        node_payload["authority_sample_count"],
        f"nodes[{node_index}].authority_sample_count",
      )
      authority_magnitude_sample_count = _require_int(
        node_payload["authority_magnitude_sample_count"],
        f"nodes[{node_index}].authority_magnitude_sample_count",
      )
      authority_slew_build_sample_count = _require_int(
        node_payload["authority_slew_build_sample_count"],
        f"nodes[{node_index}].authority_slew_build_sample_count",
      )
      authority_slew_release_sample_count = _require_int(
        node_payload["authority_slew_release_sample_count"],
        f"nodes[{node_index}].authority_slew_release_sample_count",
      )
      authority_fit_support_s = _decode_finite_float(
        node_payload["authority_fit_support_s"],
        f"nodes[{node_index}].authority_fit_support_s",
      )
      authority_fit_sample_count = _require_int(
        node_payload["authority_fit_sample_count"],
        f"nodes[{node_index}].authority_fit_sample_count",
      )
      authority_unresolved_sample_count = _require_int(
        node_payload["authority_unresolved_sample_count"],
        f"nodes[{node_index}].authority_unresolved_sample_count",
      )
      lateral_accel_min = _decode_extreme(
        node_payload["lateral_accel_min_mps2"],
        f"nodes[{node_index}].lateral_accel_min_mps2",
      )
      lateral_accel_max = _decode_extreme(
        node_payload["lateral_accel_max_mps2"],
        f"nodes[{node_index}].lateral_accel_max_mps2",
      )
      lateral_accel_energy = _decode_finite_float(
        node_payload["lateral_accel_energy_mps4_s"],
        f"nodes[{node_index}].lateral_accel_energy_mps4_s",
      )
      rack_travel_deg = _decode_finite_float(
        node_payload["rack_travel_deg"],
        f"nodes[{node_index}].rack_travel_deg",
      )
      applied_torque_min = _decode_extreme(
        node_payload["applied_torque_min"],
        f"nodes[{node_index}].applied_torque_min",
      )
      applied_torque_max = _decode_extreme(
        node_payload["applied_torque_max"],
        f"nodes[{node_index}].applied_torque_max",
      )
      rack_reversals = _require_int(
        node_payload["rack_reversals"],
        f"nodes[{node_index}].rack_reversals",
      )
      last_rack_direction = node_payload["last_rack_direction"]
      if (
        type(last_rack_direction) is not int
        or last_rack_direction not in (-1, 0, 1)
      ):
        raise ValueError("last rack direction must be -1, 0, or 1")
      training = _restore_regression(
        node_payload["training"],
        f"nodes[{node_index}].training",
      )
      validation = _restore_regression(
        node_payload["validation"],
        f"nodes[{node_index}].validation",
      )
      authority_training = _restore_regression(
        node_payload["authority_training"],
        f"nodes[{node_index}].authority_training",
      )
      authority_validation = _restore_regression(
        node_payload["authority_validation"],
        f"nodes[{node_index}].authority_validation",
      )

      if (
        clean_support_s < 0.0
        or authority_support_s < 0.0
        or authority_fit_support_s < 0.0
        or lateral_accel_energy < 0.0
        or rack_travel_deg < 0.0
      ):
        raise ValueError("learner evidence accumulators must be non-negative")
      _validate_extrema_pair(
        lateral_accel_min,
        lateral_accel_max,
        f"nodes[{node_index}].lateral_accel",
      )
      _validate_extrema_pair(
        applied_torque_min,
        applied_torque_max,
        f"nodes[{node_index}].applied_torque",
      )
      if rack_reversals > supported_sample_count:
        raise ValueError("rack reversal count exceeds sample count")
      if training.count + validation.count > supported_sample_count:
        raise ValueError("regression count exceeds supported sample count")
      if (
        authority_magnitude_sample_count > authority_sample_count
        or authority_slew_build_sample_count > authority_sample_count
        or authority_slew_release_sample_count > authority_sample_count
        or (
          authority_magnitude_sample_count
          + authority_slew_build_sample_count
          + authority_slew_release_sample_count
          < authority_sample_count
        )
        or authority_fit_sample_count > authority_magnitude_sample_count
        or (
          authority_fit_sample_count + authority_unresolved_sample_count
          > authority_magnitude_sample_count
        )
        or authority_training.count + authority_validation.count
        != authority_fit_sample_count
        or authority_fit_support_s > authority_support_s + 1e-12
        or not math.isclose(
          authority_fit_support_s,
          (
            authority_training.weight_s
            + authority_validation.weight_s
          ),
          rel_tol=1e-12,
          abs_tol=1e-12,
        )
      ):
        raise ValueError("authority evidence counts are inconsistent")
      if supported_sample_count == 0 and (
        clean_support_s != 0.0
        or lateral_accel_energy != 0.0
        or rack_travel_deg != 0.0
        or lateral_accel_min != math.inf
        or lateral_accel_max != -math.inf
        or applied_torque_min != math.inf
        or applied_torque_max != -math.inf
        or rack_reversals != 0
        or last_rack_direction != 0
        or training.count != 0
        or validation.count != 0
      ):
        raise ValueError("empty node evidence is internally inconsistent")
      if supported_sample_count > 0 and clean_support_s <= 0.0:
        raise ValueError("populated node evidence needs positive support")
      if authority_sample_count == 0 and (
        authority_support_s != 0.0
        or authority_fit_support_s != 0.0
        or authority_fit_sample_count != 0
        or authority_magnitude_sample_count != 0
        or authority_slew_build_sample_count != 0
        or authority_slew_release_sample_count != 0
        or authority_unresolved_sample_count != 0
        or authority_training.count != 0
        or authority_validation.count != 0
      ):
        raise ValueError("empty authority evidence is internally inconsistent")
      if authority_sample_count > 0 and authority_support_s <= 0.0:
        raise ValueError("populated authority evidence needs positive support")
      if (
        authority_fit_sample_count == 0
        and authority_fit_support_s != 0.0
      ):
        raise ValueError("empty authority fit has nonzero support")
      if (
        authority_fit_sample_count > 0
        and authority_fit_support_s <= 0.0
      ):
        raise ValueError("populated authority fit needs positive support")

      node = learner._nodes[node_index]
      node.clean_support_s = clean_support_s
      node.supported_sample_count = supported_sample_count
      node.authority_support_s = authority_support_s
      node.authority_sample_count = authority_sample_count
      node.authority_magnitude_sample_count = (
        authority_magnitude_sample_count
      )
      node.authority_slew_build_sample_count = (
        authority_slew_build_sample_count
      )
      node.authority_slew_release_sample_count = (
        authority_slew_release_sample_count
      )
      node.authority_fit_support_s = authority_fit_support_s
      node.authority_fit_sample_count = authority_fit_sample_count
      node.authority_unresolved_sample_count = (
        authority_unresolved_sample_count
      )
      node.lateral_accel_min = lateral_accel_min
      node.lateral_accel_max = lateral_accel_max
      node.lateral_accel_energy = lateral_accel_energy
      node.rack_travel_deg = rack_travel_deg
      node.applied_torque_min = applied_torque_min
      node.applied_torque_max = applied_torque_max
      node.rack_reversals = rack_reversals
      node.last_rack_direction = last_rack_direction
      node.training = training
      node.validation = validation
      node.authority_training = authority_training
      node.authority_validation = authority_validation
    return learner

  def _node_support(
    self,
    speed_mps: float,
  ) -> tuple[tuple[int, float], ...]:
    nodes = self.speed_nodes_mps
    if speed_mps <= nodes[0]:
      return ((0, 1.0),)
    if speed_mps >= nodes[-1]:
      return ((len(nodes) - 1, 1.0),)
    upper = 1
    while nodes[upper] < speed_mps:
      upper += 1
    lower = upper - 1
    upper_weight = (
      (speed_mps - nodes[lower]) / (nodes[upper] - nodes[lower])
    )
    if upper_weight == 0.0:
      return ((lower, 1.0),)
    if upper_weight == 1.0:
      return ((upper, 1.0),)
    return ((lower, 1.0 - upper_weight), (upper, upper_weight))

  def add_sample(self, sample: LearningSample) -> bool:
    """Add one measured sample to its physical or authority evidence stratum."""
    if (
      not sample.valid
      or not sample.engaged
      or sample.steering_pressed
      or sample.standstill
    ):
      # Rack direction is meaningful only within one uninterrupted measured
      # lifecycle. A gap, driver override, fault, disengagement, or derivative
      # warm-up must not turn the first motion of the next epoch into a
      # fabricated reversal.
      for node in self._nodes:
        node.last_rack_direction = 0
      return False
    if sample.rack_direction_reversal:
      # Direction belongs to this uninterrupted measured lifecycle, but
      # differentiating a quantized rate through its sign crossing would
      # create a false acceleration impulse. Retain only speed-local reversal
      # coverage and never admit this frame to either equality fit.
      if not sample._base_valid:
        for node in self._nodes:
          node.last_rack_direction = 0
        return False
      accepted = False
      for node_index, node_weight in self._node_support(sample.speed_mps):
        if node_weight < MIN_EXCITATION_NODE_WEIGHT:
          continue
        node = self._nodes[node_index]
        seed = self.seed_profile.nodes[node_index].parameters
        motion_threshold = max(
          seed.rack_rate_resolution_deg_s, 1e-12,
        )
        rack_direction = (
          1
          if sample.rack_rate_deg_s >= motion_threshold
          else -1
          if sample.rack_rate_deg_s <= -motion_threshold
          else 0
        )
        if rack_direction == 0:
          continue
        if (
          node.last_rack_direction != 0
          and rack_direction != node.last_rack_direction
        ):
          node.rack_reversals += 1
        node.last_rack_direction = rack_direction
        accepted = True
      return accepted
    if not sample.clean and not sample.authority_evidence:
      return False

    for node_index, node_weight in self._node_support(sample.speed_mps):
      node = self._nodes[node_index]
      seed = self.seed_profile.nodes[node_index].parameters
      weight_s = sample.dt_s * node_weight

      if sample.authority_evidence:
        node.authority_support_s += weight_s
        node.authority_sample_count += 1
        if sample.actuator_boundary & ActuatorBoundary.MAGNITUDE:
          node.authority_magnitude_sample_count += 1
        if sample.actuator_boundary & ActuatorBoundary.SLEW_BUILD:
          node.authority_slew_build_sample_count += 1
        if sample.actuator_boundary & ActuatorBoundary.SLEW_RELEASE:
          node.authority_slew_release_sample_count += 1

        motion_threshold = max(
          seed.rack_rate_resolution_deg_s, 1e-12,
        )
        settled_magnitude = (
          sample.actuator_boundary == ActuatorBoundary.MAGNITUDE
          and sample.magnitude_boundary_dwell_s + 1e-12
          >= sample.dt_s
        )
        if settled_magnitude and (
          abs(sample.rack_rate_deg_s) > motion_threshold
        ):
          validation_block = (
            node.authority_fit_sample_count
            // TRAIN_VALIDATION_BLOCK_SAMPLES
          ) % 2 == 1
          predictors = (
            -sample.measured_lateral_accel_mps2,
            sample.rack_acceleration_deg_s2,
            sample.rack_rate_deg_s,
            math.copysign(1.0, sample.rack_rate_deg_s),
          )
          evidence = (
            node.authority_validation
            if validation_block
            else node.authority_training
          )
          evidence.add(predictors, sample.applied_torque, weight_s)
          node.authority_fit_support_s += weight_s
          node.authority_fit_sample_count += 1
        elif settled_magnitude:
          # At full torque with a quantized/stationary rack, the input is
          # known but friction direction is not. Preserve it as stiction/
          # scrub evidence without pretending it is an equality row.
          node.authority_unresolved_sample_count += 1
        continue

      validation_block = (
        node.supported_sample_count // TRAIN_VALIDATION_BLOCK_SAMPLES
      ) % 2 == 1

      node.clean_support_s += weight_s
      node.supported_sample_count += 1
      node.lateral_accel_energy += (
        weight_s * sample.measured_lateral_accel_mps2**2
      )
      node.rack_travel_deg += weight_s * abs(sample.rack_rate_deg_s)

      if node_weight >= MIN_EXCITATION_NODE_WEIGHT:
        node.lateral_accel_min = min(
          node.lateral_accel_min,
          sample.measured_lateral_accel_mps2,
        )
        node.lateral_accel_max = max(
          node.lateral_accel_max,
          sample.measured_lateral_accel_mps2,
        )
        node.applied_torque_min = min(
          node.applied_torque_min, sample.applied_torque,
        )
        node.applied_torque_max = max(
          node.applied_torque_max, sample.applied_torque,
        )

        motion_threshold = max(
          seed.rack_rate_resolution_deg_s, 1e-12,
        )
        rack_direction = (
          1
          if sample.rack_rate_deg_s >= motion_threshold
          else -1
          if sample.rack_rate_deg_s <= -motion_threshold
          else 0
        )
        if rack_direction != 0:
          if (
            node.last_rack_direction != 0
            and rack_direction != node.last_rack_direction
          ):
            node.rack_reversals += 1
          node.last_rack_direction = rack_direction

      fit_motion_threshold = max(
        seed.rack_rate_resolution_deg_s, 1e-12,
      )
      if abs(sample.rack_rate_deg_s) > fit_motion_threshold:
        predictors = (
          -sample.measured_lateral_accel_mps2,
          sample.rack_acceleration_deg_s2,
          sample.rack_rate_deg_s,
          math.copysign(1.0, sample.rack_rate_deg_s),
        )
        target = sample.applied_torque
        evidence = node.validation if validation_block else node.training
        evidence.add(predictors, target, weight_s)
    return True

  def _node_report(self, node_index: int) -> NodeQualificationReport:
    node = self._nodes[node_index]
    seed = self.seed_profile.nodes[node_index].parameters
    node_speed = self.speed_nodes_mps[node_index]
    minimum_support = minimum_clean_support_s(node_speed)
    lateral_span = _span(
      node.lateral_accel_min, node.lateral_accel_max,
    )
    lateral_rms = (
      math.sqrt(node.lateral_accel_energy / node.clean_support_s)
      if node.clean_support_s > 0.0
      else 0.0
    )
    torque_span = _span(
      node.applied_torque_min, node.applied_torque_max,
    )
    minimum_validation_support = (
      minimum_support * MIN_VALIDATION_SUPPORT_FRACTION
    )

    reasons: list[QualificationReason] = []
    if node.clean_support_s < minimum_support:
      reasons.append(QualificationReason.INSUFFICIENT_SUPPORT)
    if (
      node.validation.weight_s < minimum_validation_support
      or node.validation.count < 4
    ):
      reasons.append(QualificationReason.INSUFFICIENT_VALIDATION)
    if (
      lateral_span < MIN_LATERAL_ACCEL_SPAN_MPS2
      or lateral_rms < MIN_LATERAL_ACCEL_RMS_MPS2
      or node.rack_travel_deg < MIN_RACK_TRAVEL_DEG
      or torque_span < MIN_APPLIED_TORQUE_SPAN
      or node.rack_reversals < MIN_RACK_DIRECTION_REVERSALS
    ):
      reasons.append(QualificationReason.INSUFFICIENT_EXCITATION)

    authority_fit_active = (
      node.authority_training.count >= MIN_AUTHORITY_TRAINING_SAMPLES
      and node.authority_validation.count
      >= MIN_AUTHORITY_VALIDATION_SAMPLES
    )
    empty_authority = _RegressionEvidence()
    active_authority_training = (
      node.authority_training
      if authority_fit_active
      else empty_authority
    )
    active_authority_validation = (
      node.authority_validation
      if authority_fit_active
      else empty_authority
    )
    combined_training = _combined_regression(
      node.training,
      active_authority_training,
    )
    combined_validation = _combined_regression(
      node.validation,
      active_authority_validation,
    )
    coefficients = _solve_scaled_normal_equations(combined_training)
    candidate_parameters: PhysicalParameters | None = None
    candidate_rms: float | None = None
    seed_coefficients = _seed_coefficients(seed)
    seed_rms = combined_validation.rms(seed_coefficients)
    if coefficients is None:
      reasons.append(QualificationReason.SINGULAR_FIT)
    else:
      (
        torque_per_lataccel,
        inverse_gain,
        damping_over_gain,
        kinetic_friction,
      ) = coefficients
      if inverse_gain <= 0.0:
        reasons.append(QualificationReason.INVALID_PARAMETERS)
      else:
        rack_gain = 1.0 / inverse_gain
        rack_damping = damping_over_gain / inverse_gain
        if (
          torque_per_lataccel <= 0.0
          or rack_gain <= 0.0
          or rack_damping < 0.0
          or kinetic_friction < 0.0
          or kinetic_friction > seed.static_friction_torque
          or not all(math.isfinite(value) for value in (
            torque_per_lataccel,
            rack_gain,
            rack_damping,
            kinetic_friction,
          ))
        ):
          reasons.append(QualificationReason.INVALID_PARAMETERS)
        else:
          candidate_rms = combined_validation.rms(coefficients)
          if candidate_rms is None or seed_rms is None:
            if QualificationReason.INSUFFICIENT_VALIDATION not in reasons:
              reasons.append(QualificationReason.INSUFFICIENT_VALIDATION)
          elif (
            candidate_rms
            > seed_rms + VALIDATION_RMS_ABSOLUTE_TOLERANCE
          ):
            reasons.append(QualificationReason.CROSS_FIT_REGRESSION)
          free_seed_rms = node.validation.rms(seed_coefficients)
          free_candidate_rms = node.validation.rms(coefficients)
          if (
            free_seed_rms is not None
            and free_candidate_rms is not None
            and free_candidate_rms
            > free_seed_rms + VALIDATION_RMS_ABSOLUTE_TOLERANCE
          ):
            reasons.append(QualificationReason.CROSS_FIT_REGRESSION)
          authority_seed_rms = node.authority_validation.rms(
            seed_coefficients,
          )
          authority_candidate_rms = node.authority_validation.rms(
            coefficients,
          )
          if (
            authority_fit_active
            and
            authority_seed_rms is not None
            and authority_candidate_rms is not None
            and authority_candidate_rms
            > authority_seed_rms + VALIDATION_RMS_ABSOLUTE_TOLERANCE
          ):
            reasons.append(
              QualificationReason.AUTHORITY_CROSS_FIT_REGRESSION,
            )

          ratios = (
            node.clean_support_s / minimum_support,
            node.validation.weight_s / minimum_validation_support,
            lateral_span / MIN_LATERAL_ACCEL_SPAN_MPS2,
            lateral_rms / MIN_LATERAL_ACCEL_RMS_MPS2,
            node.rack_travel_deg / MIN_RACK_TRAVEL_DEG,
            torque_span / MIN_APPLIED_TORQUE_SPAN,
            node.rack_reversals / MIN_RACK_DIRECTION_REVERSALS,
          )
          confidence = min(max(min(ratios), 0.0), 1.0)
          candidate_parameters = PhysicalParameters(
            torque_per_lateral_accel=torque_per_lataccel,
            rack_gain_deg_s2_per_torque=rack_gain,
            rack_damping_per_s=rack_damping,
            transport_delay_s=seed.transport_delay_s,
            static_friction_torque=seed.static_friction_torque,
            kinetic_friction_torque=kinetic_friction,
            rack_rate_resolution_deg_s=seed.rack_rate_resolution_deg_s,
            confidence=confidence,
            qualified=False,
          )

    unique_reasons = tuple(dict.fromkeys(reasons))
    qualified = not unique_reasons
    if candidate_parameters is not None and qualified:
      candidate_parameters = PhysicalParameters(
        torque_per_lateral_accel=candidate_parameters.torque_per_lateral_accel,
        rack_gain_deg_s2_per_torque=(
          candidate_parameters.rack_gain_deg_s2_per_torque
        ),
        rack_damping_per_s=candidate_parameters.rack_damping_per_s,
        transport_delay_s=candidate_parameters.transport_delay_s,
        static_friction_torque=candidate_parameters.static_friction_torque,
        kinetic_friction_torque=candidate_parameters.kinetic_friction_torque,
        rack_rate_resolution_deg_s=(
          candidate_parameters.rack_rate_resolution_deg_s
        ),
        confidence=candidate_parameters.confidence,
        qualified=True,
      )
    return NodeQualificationReport(
      node_index=node_index,
      speed_mps=node_speed,
      minimum_support_s=minimum_support,
      clean_support_s=node.clean_support_s,
      supported_sample_count=node.supported_sample_count,
      training_count=combined_training.count,
      validation_count=combined_validation.count,
      validation_support_s=combined_validation.weight_s,
      lateral_accel_span_mps2=lateral_span,
      lateral_accel_rms_mps2=lateral_rms,
      rack_travel_deg=node.rack_travel_deg,
      applied_torque_span=torque_span,
      rack_reversals=node.rack_reversals,
      seed_full_fit_candidate_rms=seed_rms,
      candidate_full_fit_candidate_rms=candidate_rms,
      confidence=(
        candidate_parameters.confidence
        if candidate_parameters is not None
        else 0.0
      ),
      reasons=(
        (QualificationReason.QUALIFIED,)
        if qualified
        else unique_reasons
      ),
      candidate_parameters=candidate_parameters,
      authority_support_s=node.authority_support_s,
      authority_sample_count=node.authority_sample_count,
      authority_fit_support_s=node.authority_fit_support_s,
      authority_fit_sample_count=node.authority_fit_sample_count,
      authority_training_count=node.authority_training.count,
      authority_validation_count=node.authority_validation.count,
      authority_fit_active=authority_fit_active,
      authority_seed_full_fit_candidate_rms=(
        node.authority_validation.rms(seed_coefficients)
      ),
      authority_candidate_full_fit_candidate_rms=(
        None
        if coefficients is None
        else node.authority_validation.rms(coefficients)
      ),
    )

  def qualify(self, provenance: str) -> LearningResult:
    """Fit and validate all nodes; emit a profile only when every node passes."""
    source = str(provenance).strip()
    if not source:
      raise ValueError("candidate provenance must not be empty")
    reports = tuple(
      self._node_report(index) for index in range(len(self._nodes))
    )
    if not all(report.qualified for report in reports):
      return LearningResult(node_reports=reports, candidate_profile=None)

    profile_nodes = []
    for report in reports:
      if (
        report.candidate_parameters is None
        or report.candidate_full_fit_candidate_rms is None
      ):
        raise AssertionError("qualified node lacks validated parameters")
      profile_nodes.append(ProfileNode(
        speed_mps=report.speed_mps,
        parameters=report.candidate_parameters,
        clean_support_s=report.clean_support_s,
        sample_count=report.supported_sample_count,
        cross_fit_route_count=report.validation_count,
        full_fit_candidate_rms=report.candidate_full_fit_candidate_rms,
      ))
    # Revisions are opaque monotone evidence generations, not a count of
    # approvals. The sufficient statistics are cumulative and restored
    # exactly, so the sum advances whenever any accepted clean sample changes
    # a candidate and remains identical for identical evidence after restart.
    # Keeping the original physical seed avoids an invalid "rebase" of
    # already-accumulated normal equations onto a learned profile.
    evidence_revision = (
      self.seed_profile.revision
      + 1
      + sum(
        node.supported_sample_count + node.authority_sample_count
        for node in self._nodes
      )
    )
    candidate = VehicleProfile(
      vehicle_identity=self.seed_profile.vehicle_identity,
      revision=evidence_revision,
      provenance=(
        f"{source}; modular-offroad-learner-v3; " +
        f"fit_seed_revision={self.seed_profile.revision}; " +
        f"evidence_revision={evidence_revision}"
      ),
      nodes=tuple(profile_nodes),
    )
    return LearningResult(node_reports=reports, candidate_profile=candidate)


def learning_sample_field_names() -> tuple[str, ...]:
  """Expose the physical-only input contract for audit/tests."""
  return tuple(field.name for field in fields(LearningSample) if field.init)
