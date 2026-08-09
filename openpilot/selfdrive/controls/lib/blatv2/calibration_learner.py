"""Deterministic learner for BLaTv2's observable inverse-torque map.

The fitted law at each speed node is::

  applied = gain * (-measured_lataccel + offset_correction)
            + kinetic * motion_sign
            + static * breakaway_sign

Only one friction sign is nonzero per row. Quantized/stationary rows remain a
legacy no-regression surface, but never decide the moving inverse map. Static
breakaway is identified from a complete physical episode: the causally
aligned last-stuck response and the earliest angle-resolved motion, confirmed
by a same-direction rate quantum. Later moving rows identify kinetic motion.

Inputs are the measured-only :class:`learner.LearningSample`. Slew build and
release frames are authority evidence but never equality-fit. Evidence schema
14 is deliberately incompatible with older evidence. Route uncertainty is
bound to immutable route/content identities and per-route sufficient
statistics. The canonical route counter remains provenance only; it never
assigns a statistical role. Every route supplied here already belongs to the
caller's sealed TRAIN partition.

The inverse map is selected from a deterministic nested family by
route-grouped leave-one-route-out cross-fitting. Every fold is fit on all but
one TRAIN route and scored only on the omitted route. Selection uses only
out-of-fold scores and must clear the conservative paired whole-route loss
envelope in every required stratum. The frozen family is then refit once on
all TRAIN routes for publication. Global VALIDATION and TEST are neither
accepted nor representable by this learner.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import hmac
import json
import math
import sys
from typing import Any

from openpilot.selfdrive.controls.lib.blatv2.breakaway_episode import (
  BreakawayEpisode,
  BreakawayEpisodeDetector,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  CALIBRATION_PROFILE_SCHEMA_VERSION,
  CalibrationParameters,
  CalibrationProfileNode,
  VehicleCalibrationProfile,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_source import (
  CalibrationIngestionCoordinate,
)
from openpilot.selfdrive.controls.lib.blatv2.learner import (
  ActuatorBoundary,
  LearningSample,
  MIN_APPLIED_TORQUE_SPAN,
  MIN_EXCITATION_NODE_WEIGHT,
  MIN_LATERAL_ACCEL_RMS_MPS2,
  MIN_LATERAL_ACCEL_SPAN_MPS2,
  minimum_clean_support_s,
)


CALIBRATION_EVIDENCE_SCHEMA_VERSION = 15
MAX_CALIBRATION_ROUTE_COUNT = 1 << 32
MAX_CALIBRATION_ROWS_PER_ROUTE = 1_000_000
MAX_CALIBRATION_EVIDENCE_ROWS = (
  MAX_CALIBRATION_ROUTE_COUNT * MAX_CALIBRATION_ROWS_PER_ROUTE
)
MAX_CALIBRATION_WIRE_FLOAT_ABS = math.sqrt(sys.float_info.max) / 16.0
MIN_STRATUM_TRAINING_ROWS = 4
MIN_INDEPENDENT_ROUTES = 2
MIN_TERMINAL_SEED_AUTHORITY_ROUTES = 1
TERMINAL_SEED_AUTHORITY_VEHICLE_IDENTITY = "HYUNDAI_PALISADE"
TERMINAL_SEED_AUTHORITY_SPEED_MPS = 30.0
TERMINAL_SEED_AUTHORITY_POLICY_ID = (
  "palisade-terminal-30mps-authority-one-route-field-test-v2"
)
NORMAL_MATRIX_RELATIVE_PIVOT_MIN = 1e-10
# Floating-point cancellation guard only. Physical acceptance uses the
# route-level paired uncertainty interval implemented below.
NUMERICAL_LOSS_EPSILON_MULTIPLIER = 64.0
FIT_CONDITION_LIMIT = 1.0 / math.sqrt(sys.float_info.epsilon)


def _bounded_count(value: object, context: str) -> int:
  if (
    type(value) is not int
    or value < 0
    or value > MAX_CALIBRATION_EVIDENCE_ROWS
  ):
    raise ValueError(f"{context} is outside the evidence schema bound")
  return value


def terminal_seed_authority_allowed(
  *,
  vehicle_identity: str,
  speed_mps: float,
  terminal: bool,
  seed_retained: bool,
) -> bool:
  return (
    vehicle_identity == TERMINAL_SEED_AUTHORITY_VEHICLE_IDENTITY
    and terminal
    and speed_mps == TERMINAL_SEED_AUTHORITY_SPEED_MPS
    and seed_retained
  )


def _stat_tolerance(scale: float, count: int, multiplier: float) -> float:
  """Finite count-scaled tolerance for already-bounded wire statistics."""
  relative = min(
    1e-6,
    multiplier * sys.float_info.epsilon * max(count, 1),
  )
  return relative * max(scale, 1.0)


def _finite_fsum(values: Iterable[float], context: str) -> float:
  try:
    result = math.fsum(values)
  except OverflowError as exc:
    raise ValueError(f"{context} arithmetic overflow") from exc
  if not math.isfinite(result):
    raise ValueError(f"{context} arithmetic overflow")
  return result


class CalibrationSampleDisposition(StrEnum):
  """One bounded, mutually-exclusive outcome for every measured frame.

  The runtime assigns upstream failures in physical pipeline order: malformed
  input, vehicle validity, lateral lifecycle, speed eligibility, rack mapping,
  driver interaction, causal command alignment, and measurement continuity.
  Only frames surviving that chain reach the learner's two terminal rejection
  outcomes. This first-cause ordering keeps counts additive and reproducible.
  """

  ACCEPTED = "accepted"
  INVALID_NUMERIC_OR_TIMESTAMP = "invalid_numeric_or_timestamp"
  VEHICLE_INPUT_INVALID = "vehicle_input_invalid"
  LATERAL_INACTIVE = "lateral_inactive"
  STANDSTILL_OR_BELOW_MIN_STEER_SPEED = "standstill_or_below_min_steer_speed"
  LIVE_RACK_MAPPING_INVALID = "live_rack_mapping_invalid"
  DRIVER_OVERRIDE_OR_ALLOWANCE = "driver_override_or_allowance"
  CAUSAL_COMMAND_ALIGNMENT_UNAVAILABLE = "causal_command_alignment_unavailable"
  MEASUREMENT_WARMUP_OR_DISCONTINUITY = "measurement_warmup_or_discontinuity"
  LEARNER_INELIGIBLE = "learner_ineligible"
  BREAKAWAY_EPISODE_DISCARDED = "breakaway_episode_discarded"


class CalibrationIntervalStratum(StrEnum):
  BASE = "base"
  MOVING = "moving"
  BREAKAWAY_EPISODE = "breakaway_episode"
  AUTHORITY = "authority"


_REJECTED_SAMPLE_DISPOSITIONS = tuple(
  disposition
  for disposition in CalibrationSampleDisposition
  if disposition is not CalibrationSampleDisposition.ACCEPTED
)
_SAMPLE_DISPOSITIONS = tuple(CalibrationSampleDisposition)
_SAMPLE_DISPOSITION_INDEX = {
  disposition: index
  for index, disposition in enumerate(_SAMPLE_DISPOSITIONS)
}
_LEARNER_OWNED_SAMPLE_DISPOSITIONS = frozenset({
  CalibrationSampleDisposition.LEARNER_INELIGIBLE,
  CalibrationSampleDisposition.BREAKAWAY_EPISODE_DISCARDED,
})


@dataclass(frozen=True, slots=True)
class CalibrationSampleAccounting:
  """Canonical durable totals for the finite sample-disposition vocabulary."""

  accepted_sample_count: int
  rejection_counts: tuple[int, ...]

  def __post_init__(self) -> None:
    if (
      type(self.accepted_sample_count) is not int
      or not 0 <= self.accepted_sample_count <= MAX_CALIBRATION_EVIDENCE_ROWS
      or type(self.rejection_counts) is not tuple
      or len(self.rejection_counts) != len(_REJECTED_SAMPLE_DISPOSITIONS)
      or any(
        type(count) is not int
        or not 0 <= count <= MAX_CALIBRATION_EVIDENCE_ROWS
        for count in self.rejection_counts
      )
    ):
      raise ValueError("calibration sample accounting is invalid")

  @classmethod
  def empty(cls) -> CalibrationSampleAccounting:
    return cls(0, (0,) * len(_REJECTED_SAMPLE_DISPOSITIONS))

  @property
  def rejected_sample_count(self) -> int:
    return sum(self.rejection_counts)

  @property
  def ingested_sample_count(self) -> int:
    return self.accepted_sample_count + self.rejected_sample_count

  def count(self, disposition: CalibrationSampleDisposition) -> int:
    if disposition is CalibrationSampleDisposition.ACCEPTED:
      return self.accepted_sample_count
    return self.rejection_counts[_REJECTED_SAMPLE_DISPOSITIONS.index(disposition)]

  def with_disposition(
    self,
    disposition: CalibrationSampleDisposition,
  ) -> CalibrationSampleAccounting:
    if not isinstance(disposition, CalibrationSampleDisposition):
      raise TypeError("sample disposition must be CalibrationSampleDisposition")
    if disposition is CalibrationSampleDisposition.ACCEPTED:
      return CalibrationSampleAccounting(
        self.accepted_sample_count + 1,
        self.rejection_counts,
      )
    index = _REJECTED_SAMPLE_DISPOSITIONS.index(disposition)
    counts = list(self.rejection_counts)
    counts[index] += 1
    return CalibrationSampleAccounting(self.accepted_sample_count, tuple(counts))

  def to_payload(self) -> dict[str, object]:
    return {
      "accepted_sample_count": self.accepted_sample_count,
      "ingested_sample_count": self.ingested_sample_count,
      "rejected_sample_count": self.rejected_sample_count,
      "rejection_reasons": {
        disposition.value: self.rejection_counts[index]
        for index, disposition in enumerate(_REJECTED_SAMPLE_DISPOSITIONS)
      },
    }

  @classmethod
  def from_payload(cls, raw: object) -> CalibrationSampleAccounting:
    if type(raw) is not dict or set(raw) != {
      "accepted_sample_count",
      "ingested_sample_count",
      "rejected_sample_count",
      "rejection_reasons",
    }:
      raise ValueError("calibration sample-accounting schema does not match")
    accepted = raw["accepted_sample_count"]
    ingested = raw["ingested_sample_count"]
    rejected = raw["rejected_sample_count"]
    reasons = raw["rejection_reasons"]
    expected_reasons = {
      disposition.value for disposition in _REJECTED_SAMPLE_DISPOSITIONS
    }
    if (
      type(reasons) is not dict
      or set(reasons) != expected_reasons
    ):
      raise ValueError("calibration sample accounting is invalid")
    accepted = _bounded_count(accepted, "sample_accounting.accepted_sample_count")
    ingested = _bounded_count(ingested, "sample_accounting.ingested_sample_count")
    rejected = _bounded_count(rejected, "sample_accounting.rejected_sample_count")
    counts_by_reason = {
      reason: _bounded_count(count, f"sample_accounting.rejection_reasons.{reason}")
      for reason, count in reasons.items()
    }
    counts = tuple(counts_by_reason[disposition.value] for disposition in _REJECTED_SAMPLE_DISPOSITIONS)
    accounting = cls(accepted, counts)
    if (
      accounting.rejected_sample_count != rejected
      or accounting.ingested_sample_count != ingested
    ):
      raise ValueError("calibration sample-accounting totals disagree")
    return accounting


class CalibrationModelId(StrEnum):
  STATIC_ONLY = "static_only"
  FRICTION_MAP = "friction_map"
  OFFSET_AND_FRICTION = "offset_and_friction"
  FULL_MAP = "full_map"


class CalibrationQualificationReason(StrEnum):
  QUALIFIED = "qualified"
  LEARNED = "learned"
  SEED_RETAINED = "seed_retained"
  INSUFFICIENT_SUPPORT = "insufficient_support"
  INSUFFICIENT_CROSS_FIT = "insufficient_cross_fit"
  INSUFFICIENT_EXCITATION = "insufficient_excitation"
  INSUFFICIENT_MOVING_EVIDENCE = "insufficient_moving_evidence"
  INSUFFICIENT_BREAKAWAY_EVIDENCE = "insufficient_breakaway_evidence"
  RANK_DEFICIENT_FIT = "rank_deficient_fit"
  ILL_CONDITIONED_FIT = "ill_conditioned_fit"
  SINGULAR_FIT = "singular_fit"
  INVALID_PARAMETERS = "invalid_parameters"
  CROSS_FIT_INCONCLUSIVE = "cross_fit_inconclusive"
  CROSS_FIT_REGRESSION = "cross_fit_regression"
  MOVING_CROSS_FIT_REGRESSION = "moving_cross_fit_regression"
  BREAKAWAY_CROSS_FIT_REGRESSION = "breakaway_cross_fit_regression"
  AUTHORITY_CROSS_FIT_REGRESSION = "authority_cross_fit_regression"
  INTERPOLATION_TRAINING_INCONCLUSIVE = "interpolation_training_inconclusive"
  INTERPOLATION_TRAINING_REGRESSION = "interpolation_training_regression"
  INTERPOLATION_CROSS_FIT_INCONCLUSIVE = "interpolation_cross_fit_inconclusive"
  INTERPOLATION_CROSS_FIT_REGRESSION = "interpolation_cross_fit_regression"
  INSUFFICIENT_INDEPENDENT_ROUTES = "insufficient_independent_routes"
  CROSS_FIT_FOLD_FAILURE = "cross_fit_fold_failure"


CALIBRATION_NODE_FAILURE_REASON_ORDER = (
  CalibrationQualificationReason.INSUFFICIENT_SUPPORT,
  CalibrationQualificationReason.INSUFFICIENT_EXCITATION,
  CalibrationQualificationReason.INSUFFICIENT_MOVING_EVIDENCE,
  CalibrationQualificationReason.INSUFFICIENT_BREAKAWAY_EVIDENCE,
  CalibrationQualificationReason.INSUFFICIENT_INDEPENDENT_ROUTES,
  CalibrationQualificationReason.CROSS_FIT_FOLD_FAILURE,
  CalibrationQualificationReason.ILL_CONDITIONED_FIT,
  CalibrationQualificationReason.RANK_DEFICIENT_FIT,
  CalibrationQualificationReason.SINGULAR_FIT,
  CalibrationQualificationReason.CROSS_FIT_REGRESSION,
  CalibrationQualificationReason.INVALID_PARAMETERS,
)


def calibration_node_failure_reasons(
  *,
  insufficient_support: bool,
  insufficient_excitation: bool,
  insufficient_moving_evidence: bool,
  insufficient_breakaway_evidence: bool,
  insufficient_independent_routes: bool,
  cross_fit_fold_failure: bool,
  fit_statuses: tuple[CalibrationFitStatus, ...],
  full_fit_safe: bool,
  parameters_valid: bool,
) -> tuple[CalibrationQualificationReason, ...]:
  """One ordered qualification-failure vocabulary for producer and decoder."""
  reasons: set[CalibrationQualificationReason] = set()
  if insufficient_support:
    reasons.add(CalibrationQualificationReason.INSUFFICIENT_SUPPORT)
  if insufficient_excitation:
    reasons.add(CalibrationQualificationReason.INSUFFICIENT_EXCITATION)
  if insufficient_moving_evidence:
    reasons.add(CalibrationQualificationReason.INSUFFICIENT_MOVING_EVIDENCE)
  if insufficient_breakaway_evidence:
    reasons.add(CalibrationQualificationReason.INSUFFICIENT_BREAKAWAY_EVIDENCE)
  if insufficient_independent_routes:
    reasons.add(CalibrationQualificationReason.INSUFFICIENT_INDEPENDENT_ROUTES)
  if cross_fit_fold_failure:
    reasons.add(CalibrationQualificationReason.CROSS_FIT_FOLD_FAILURE)
  if CalibrationFitStatus.IDENTIFIABLE not in fit_statuses:
    if CalibrationFitStatus.ILL_CONDITIONED in fit_statuses:
      reasons.add(CalibrationQualificationReason.ILL_CONDITIONED_FIT)
    elif CalibrationFitStatus.RANK_DEFICIENT in fit_statuses:
      reasons.add(CalibrationQualificationReason.RANK_DEFICIENT_FIT)
    else:
      reasons.add(CalibrationQualificationReason.SINGULAR_FIT)
  if not full_fit_safe:
    reasons.add(CalibrationQualificationReason.CROSS_FIT_REGRESSION)
  if not parameters_valid:
    reasons.add(CalibrationQualificationReason.INVALID_PARAMETERS)
  return tuple(
    reason for reason in CALIBRATION_NODE_FAILURE_REASON_ORDER
    if reason in reasons
  )


class CalibrationFitStatus(StrEnum):
  IDENTIFIABLE = "identifiable"
  RANK_DEFICIENT = "rank_deficient"
  ILL_CONDITIONED = "ill_conditioned"
  NO_SOLUTION = "no_solution"


class CalibrationCrossFitStatus(StrEnum):
  SCORED = "scored"
  INSUFFICIENT_INDEPENDENT_ROUTES = "insufficient_independent_routes"
  FOLD_FIT_FAILURE = "fold_fit_failure"
  HELD_OUT_REGRESSION = "held_out_regression"
  NO_ROBUST_IMPROVEMENT = "no_robust_improvement"


@dataclass(frozen=True, slots=True)
class CalibrationModelFitDiagnostic:
  model: CalibrationModelId
  status: CalibrationFitStatus
  moving_rank: int
  moving_parameter_count: int
  condition_estimate: float | None
  breakaway_rank: int
  breakaway_parameter_count: int


@dataclass(frozen=True, slots=True)
class CalibrationPairedLossDiagnostic:
  route_count: int
  mean_candidate_minus_seed_mse: float | None
  uncertainty_mse: float | None
  lower_bound_mse: float | None
  upper_bound_mse: float | None
  numerical_tolerance_mse: float | None


@dataclass(frozen=True, slots=True)
class CalibrationIndependentRouteCounts:
  all: int
  base: int
  moving: int
  breakaway: int
  breakaway_episode: int
  authority: int


@dataclass(frozen=True, slots=True)
class CalibrationRouteCommitment:
  route_index: int
  route_counter: int
  route_identity_sha256: str
  route_content_sha256: str
  assignment_record_count: int
  assignment_chain_sha256: str
  route_commitment_sha256: str


@dataclass(frozen=True, slots=True)
class CalibrationCrossFitModelDiagnostic:
  model: CalibrationModelId
  status: CalibrationCrossFitStatus
  contributing_route_count: int
  successful_fold_count: int
  failed_fold_count: int
  regressed_fold_count: int
  paired_loss: CalibrationPairedLossDiagnostic


def _validate_augmented_gram(
  normal: list[float],
  rhs: list[float],
  target_squared: float,
  dimension: int,
  context: str,
) -> None:
  """Reject sufficient statistics that no weighted real rows can produce."""
  def finite_product(*values: float) -> float:
    result = 1.0
    try:
      for value in values:
        result *= value
    except OverflowError as exc:
      raise ValueError(f"{context} arithmetic overflow") from exc
    if not math.isfinite(result):
      raise ValueError(f"{context} arithmetic overflow")
    return result

  scale = max([abs(value) for value in (*normal, *rhs, target_squared)], default=1.0)
  tolerance = _stat_tolerance(scale, 1, 512.0 * (dimension + 1))
  for row in range(dimension):
    diagonal = normal[row * dimension + row]
    if diagonal < -tolerance:
      raise ValueError(f"{context} has a negative predictor energy")
    for column in range(dimension):
      if abs(normal[row * dimension + column] - normal[column * dimension + row]) > tolerance:
        raise ValueError(f"{context} normal matrix is not symmetric")
      bound = finite_product(
        max(diagonal, 0.0),
        max(normal[column * dimension + column], 0.0),
      )
      squared = finite_product(
        normal[row * dimension + column],
        normal[row * dimension + column],
      )
      energy_tolerance = _stat_tolerance(
        max(bound, squared),
        1,
        512.0 * (dimension + 1),
      )
      if squared > bound and squared - bound > energy_tolerance:
        raise ValueError(f"{context} violates predictor Cauchy bounds")
    rhs_squared = finite_product(rhs[row], rhs[row])
    target_bound = finite_product(max(diagonal, 0.0), target_squared)
    target_tolerance = _stat_tolerance(
      max(target_bound, rhs_squared),
      1,
      512.0 * (dimension + 1),
    )
    if rhs_squared > target_bound and rhs_squared - target_bound > target_tolerance:
      raise ValueError(f"{context} violates target Cauchy bounds")

  augmented = [0.0] * ((dimension + 1) ** 2)
  augmented_dimension = dimension + 1
  for row in range(dimension):
    for column in range(dimension):
      augmented[row * augmented_dimension + column] = normal[row * dimension + column]
    augmented[row * augmented_dimension + dimension] = rhs[row]
    augmented[dimension * augmented_dimension + row] = rhs[row]
  augmented[-1] = target_squared
  _validate_scaled_psd(augmented, augmented_dimension, context)


def _validate_scaled_psd(
  matrix: list[float],
  dimension: int,
  context: str,
) -> None:
  """Validate PSD up to a deterministic binary64 backward-error bound.

  Diagonal congruence removes physical units and absolute scale before a
  max-residual-diagonal pivoted LDL decomposition. This avoids declaring a
  valid low-energy direction null merely because its raw pivot is below a
  global absolute tolerance. Equal pivots select the lowest original index.
  """
  if len(matrix) != dimension * dimension:
    raise ValueError(f"{context} PSD matrix dimension is invalid")
  row_norms = tuple(
    _finite_fsum(
      (abs(matrix[row * dimension + column]) for column in range(dimension)),
      f"{context} PSD row norm",
    )
    for row in range(dimension)
  )
  matrix_norm = max(row_norms, default=0.0)
  relative_tolerance = 512.0 * dimension * sys.float_info.epsilon
  raw_tolerance = relative_tolerance * matrix_norm
  diagonal = [matrix[index * dimension + index] for index in range(dimension)]
  for index, value in enumerate(diagonal):
    if value < -raw_tolerance:
      raise ValueError(f"{context} augmented Gram matrix is not PSD")
    if value <= 0.0:
      diagonal[index] = 0.0

  normalized = [0.0] * (dimension * dimension)
  roots = [math.sqrt(value) for value in diagonal]
  for row in range(dimension):
    for column in range(row, dimension):
      value = matrix[row * dimension + column]
      if roots[row] == 0.0 or roots[column] == 0.0:
        if abs(value) > raw_tolerance:
          raise ValueError(f"{context} augmented Gram nullspace is inconsistent")
        normalized_value = 0.0
      else:
        normalized_value = value / (roots[row] * roots[column])
        if not math.isfinite(normalized_value):
          raise ValueError(f"{context} arithmetic overflow")
      normalized[row * dimension + column] = normalized_value
      normalized[column * dimension + row] = normalized_value

  remaining = list(range(dimension))
  while remaining:
    pivot_index = min(
      remaining,
      key=lambda index: (-normalized[index * dimension + index], index),
    )
    pivot = normalized[pivot_index * dimension + pivot_index]
    if pivot < -relative_tolerance:
      raise ValueError(f"{context} augmented Gram matrix is not PSD")
    if pivot <= relative_tolerance:
      if any(
        abs(normalized[row * dimension + column]) > relative_tolerance
        for row in remaining
        for column in remaining
      ):
        raise ValueError(f"{context} augmented Gram nullspace is inconsistent")
      return
    remaining.remove(pivot_index)
    for row_offset, row in enumerate(remaining):
      for column in remaining[row_offset:]:
        update = (
          normalized[row * dimension + pivot_index]
          * normalized[column * dimension + pivot_index]
          / pivot
        )
        value = normalized[row * dimension + column] - update
        if not math.isfinite(value):
          raise ValueError(f"{context} arithmetic overflow")
        normalized[row * dimension + column] = value
        normalized[column * dimension + row] = value


def _regression_close(left: float, right: float, regression: _Regression) -> bool:
  tolerance = _stat_tolerance(
    max(abs(left), abs(right), regression.weight_s),
    regression.count,
    NUMERICAL_LOSS_EPSILON_MULTIPLIER,
  )
  return abs(left - right) <= tolerance


def _validate_sign_predictor(
  regression: _Regression,
  active_index: int | None,
  context: str,
) -> None:
  """Validate the exact {-1, 0, +1} sign-predictor alphabet."""
  for sign_index in (2, 3):
    expected_energy = regression.weight_s if sign_index == active_index else 0.0
    if not _regression_close(
      regression.normal[sign_index * 4 + sign_index],
      expected_energy,
      regression,
    ):
      raise ValueError(f"{context} sign-predictor energy is invalid")
    if sign_index == active_index:
      continue
    if not _regression_close(regression.rhs[sign_index], 0.0, regression):
      raise ValueError(f"{context} inactive sign target cross term is nonzero")
    for other in range(4):
      if not _regression_close(
        regression.normal[sign_index * 4 + other],
        0.0,
        regression,
      ) or not _regression_close(
        regression.normal[other * 4 + sign_index],
        0.0,
        regression,
      ):
        raise ValueError(f"{context} inactive sign cross term is nonzero")


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

  def mse(self, coefficients: tuple[float, float, float, float]) -> float | None:
    error = _weighted_error(self, coefficients)
    if error is None or self.weight_s <= 0.0:
      return None
    return error / self.weight_s

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
    count = _bounded_count(payload["count"], f"{context}.count")
    normal = _hex_list(payload["normal"], 16, f"{context}.normal")
    rhs = _hex_list(payload["rhs"], 4, f"{context}.rhs")
    result = cls()
    result.count = count
    result.normal[:] = normal
    result.rhs[:] = rhs
    result.target_squared = _hex(payload["target_squared"], f"{context}.target_squared")
    result.weight_s = _hex(payload["weight_s"], f"{context}.weight_s")
    if result.target_squared < 0.0 or result.weight_s < 0.0 or (result.count > 0) != (result.weight_s > 0.0):
      raise ValueError(f"{context} has inconsistent support")
    if result.count == 0 and any(value != 0.0 for value in (*normal, *rhs, result.target_squared)):
      raise ValueError(f"{context} empty statistics are nonzero")
    _validate_augmented_gram(
      result.normal,
      result.rhs,
      result.target_squared,
      4,
      context,
    )
    tolerance = 512.0 * sys.float_info.epsilon * max(result.weight_s, 1.0)
    if abs(result.normal[5] - result.weight_s) > tolerance:
      raise ValueError(f"{context} intercept energy disagrees with support")
    if abs(result.normal[11]) > tolerance:
      raise ValueError(f"{context} moving and breakaway predictors overlap")
    return result


class _JointRegression:
  """Sufficient statistics for the exact runtime interpolation interval.

  Runtime interpolates gain and lateral-acceleration offset independently,
  then multiplies them in ``gain * (-lataccel + offset)``.  The resulting
  offset torque is quadratic in speed weight; linearly interpolating the four
  regression coefficients would therefore validate a controller that does
  not exist.  Ten expanded predictors retain the exact endpoint monomials
  ``g0``, ``g1``, ``g0*o0``, ``g0*o1``, ``g1*o0``, ``g1*o1``, plus the two
  endpoint values for moving and static friction.  This is profile validation,
  not a joint constrained fitter: endpoint models remain independently fit
  and any incoherent runtime interpolation is rejected.
  """

  __slots__ = (
    "normal",
    "predictor_sums",
    "rhs",
    "target_squared",
    "weight_s",
    "count",
  )

  def __init__(self) -> None:
    self.normal = [0.0] * 100
    self.predictor_sums = [0.0] * 10
    self.rhs = [0.0] * 10
    self.target_squared = 0.0
    self.weight_s = 0.0
    self.count = 0

  def add(
    self,
    x: tuple[float, float, float, float],
    y: float,
    weight: float,
    upper_weight: float,
  ) -> None:
    lower_weight = 1.0 - upper_weight
    cross_weight = lower_weight * upper_weight
    predictors = (
      lower_weight * x[0],
      upper_weight * x[0],
      lower_weight * lower_weight * x[1],
      cross_weight * x[1],
      cross_weight * x[1],
      upper_weight * upper_weight * x[1],
      lower_weight * x[2],
      upper_weight * x[2],
      lower_weight * x[3],
      upper_weight * x[3],
    )
    for row in range(10):
      self.predictor_sums[row] += weight * predictors[row]
      self.rhs[row] += weight * predictors[row] * y
      for column in range(10):
        self.normal[row * 10 + column] += (
          weight * predictors[row] * predictors[column]
        )
    self.target_squared += weight * y * y
    self.weight_s += weight
    self.count += 1

  def mse(
    self,
    lower: CalibrationParameters,
    upper: CalibrationParameters,
  ) -> float | None:
    if self.weight_s <= 0.0:
      return None
    coefficients = (
      lower.torque_per_lateral_accel,
      upper.torque_per_lateral_accel,
      lower.torque_per_lateral_accel
      * lower.lateral_accel_offset_correction_mps2,
      lower.torque_per_lateral_accel
      * upper.lateral_accel_offset_correction_mps2,
      upper.torque_per_lateral_accel
      * lower.lateral_accel_offset_correction_mps2,
      upper.torque_per_lateral_accel
      * upper.lateral_accel_offset_correction_mps2,
      lower.kinetic_friction_torque,
      upper.kinetic_friction_torque,
      lower.static_breakaway_torque,
      upper.static_breakaway_torque,
    )
    linear = sum(coefficients[index] * self.rhs[index] for index in range(10))
    quadratic = sum(
      coefficients[row]
      * self.normal[row * 10 + column]
      * coefficients[column]
      for row in range(10)
      for column in range(10)
    )
    error = self.target_squared - 2.0 * linear + quadratic
    scale = max(self.target_squared, quadratic, 1.0)
    if error < 0.0 and abs(error) <= 1e-12 * scale:
      error = 0.0
    return (
      error / self.weight_s
      if error >= 0.0 and math.isfinite(error)
      else None
    )

  def encoded(self) -> dict[str, Any]:
    return {
      "count": self.count,
      "normal": [value.hex() for value in self.normal],
      "predictor_sums": [value.hex() for value in self.predictor_sums],
      "rhs": [value.hex() for value in self.rhs],
      "target_squared": self.target_squared.hex(),
      "weight_s": self.weight_s.hex(),
    }

  @classmethod
  def decoded(cls, raw: object, context: str) -> _JointRegression:
    payload = _exact(
      raw,
      {
        "count",
        "normal",
        "predictor_sums",
        "rhs",
        "target_squared",
        "weight_s",
      },
      context,
    )
    result = cls()
    result.count = _bounded_count(payload["count"], f"{context}.count")
    result.normal[:] = _hex_list(payload["normal"], 100, f"{context}.normal")
    result.predictor_sums[:] = _hex_list(
      payload["predictor_sums"],
      10,
      f"{context}.predictor_sums",
    )
    result.rhs[:] = _hex_list(payload["rhs"], 10, f"{context}.rhs")
    result.target_squared = _hex(
      payload["target_squared"], f"{context}.target_squared"
    )
    result.weight_s = _hex(payload["weight_s"], f"{context}.weight_s")
    if (
      result.target_squared < 0.0
      or result.weight_s < 0.0
      or (result.count > 0) != (result.weight_s > 0.0)
    ):
      raise ValueError(f"{context} has inconsistent support")
    if result.count == 0 and any(
      value != 0.0
      for value in (
        *result.normal,
        *result.predictor_sums,
        *result.rhs,
        result.target_squared,
      )
    ):
      raise ValueError(f"{context} empty statistics are nonzero")
    _validate_augmented_gram(
      result.normal,
      result.rhs,
      result.target_squared,
      10,
      context,
    )
    _validate_joint_predictor_moments(result, context)
    return result


def _validate_joint_predictor_moments(
  regression: _JointRegression,
  context: str,
) -> None:
  """Bind the expanded interpolation Gram to its exact row algebra."""
  tolerance = _stat_tolerance(
    max(
      regression.weight_s,
      *(abs(value) for value in regression.predictor_sums),
      *(abs(value) for value in regression.normal),
    ),
    regression.count,
    NUMERICAL_LOSS_EPSILON_MULTIPLIER,
  )
  if abs(regression.predictor_sums[3] - regression.predictor_sums[4]) > tolerance:
    raise ValueError(f"{context} duplicate interpolation moments disagree")
  if any(regression.predictor_sums[index] < -tolerance for index in (2, 3, 4, 5)):
    raise ValueError(f"{context} interpolation basis moment is negative")
  intercept_sum = _finite_fsum(
    (regression.predictor_sums[index] for index in (2, 3, 4, 5)),
    f"{context} interpolation first moment",
  )
  if abs(intercept_sum - regression.weight_s) > tolerance:
    raise ValueError(f"{context} interpolation first moment is invalid")

  # The interpolation basis is [(1-t)^2, t(1-t), t(1-t), t^2] for
  # t in [0, 1]. Recover its weighted moments through degree four and
  # validate the finite Hausdorff moment conditions. Positive semidefinite
  # predictor statistics alone only prove that some real vector generated
  # the row; these localizing matrices prove that vector is representable by
  # a bounded interpolation coordinate.
  moment_0 = regression.weight_s
  moment_2 = regression.predictor_sums[5]
  moment_1 = _finite_fsum(
    (regression.predictor_sums[3], moment_2),
    f"{context} interpolation linear moment",
  )
  moment_4 = regression.normal[5 * 10 + 5]
  moment_3 = _finite_fsum(
    (regression.normal[3 * 10 + 5], moment_4),
    f"{context} interpolation cubic moment",
  )
  _validate_augmented_gram(
    [
      moment_0, moment_1, moment_2,
      moment_1, moment_2, moment_3,
      moment_2, moment_3, moment_4,
    ],
    [0.0, 0.0, 0.0],
    0.0,
    3,
    f"{context} interpolation Hausdorff moment matrix",
  )
  _validate_augmented_gram(
    [
      moment_1 - moment_2, moment_2 - moment_3,
      moment_2 - moment_3, moment_3 - moment_4,
    ],
    [0.0, 0.0],
    0.0,
    2,
    f"{context} interpolation localizing matrix",
  )
  expected_basis_gram = (
    moment_0 - 4.0 * moment_1 + 6.0 * moment_2 - 4.0 * moment_3 + moment_4,
    moment_1 - 3.0 * moment_2 + 3.0 * moment_3 - moment_4,
    moment_1 - 3.0 * moment_2 + 3.0 * moment_3 - moment_4,
    moment_2 - 2.0 * moment_3 + moment_4,
    moment_1 - 3.0 * moment_2 + 3.0 * moment_3 - moment_4,
    moment_2 - 2.0 * moment_3 + moment_4,
    moment_2 - 2.0 * moment_3 + moment_4,
    moment_3 - moment_4,
    moment_1 - 3.0 * moment_2 + 3.0 * moment_3 - moment_4,
    moment_2 - 2.0 * moment_3 + moment_4,
    moment_2 - 2.0 * moment_3 + moment_4,
    moment_3 - moment_4,
    moment_2 - 2.0 * moment_3 + moment_4,
    moment_3 - moment_4,
    moment_3 - moment_4,
    moment_4,
  )
  actual_basis_gram = tuple(
    regression.normal[row * 10 + column]
    for row in (2, 3, 4, 5)
    for column in (2, 3, 4, 5)
  )
  if any(
    abs(actual - expected) > tolerance
    for actual, expected in zip(
      actual_basis_gram,
      expected_basis_gram,
      strict=True,
    )
  ):
    raise ValueError(f"{context} interpolation polynomial Gram is invalid")
  for row in range(10):
    recovered = _finite_fsum(
      (
        regression.normal[row * 10 + column]
        for column in (2, 3, 4, 5)
      ),
      f"{context} interpolation row moment",
    )
    if abs(recovered - regression.predictor_sums[row]) > tolerance:
      raise ValueError(f"{context} interpolation moments are inconsistent")


def _validate_joint_sign_predictor(
  regression: _JointRegression,
  active_indices: tuple[int, int] | None,
  context: str,
) -> None:
  """Validate interpolated sign features without assuming a node weight."""
  tolerance = _stat_tolerance(
    regression.weight_s,
    regression.count,
    NUMERICAL_LOSS_EPSILON_MULTIPLIER,
  )
  active = set(() if active_indices is None else active_indices)
  if abs(regression.rhs[3] - regression.rhs[4]) > tolerance or any(
    abs(regression.normal[3 * 10 + index] - regression.normal[4 * 10 + index])
    > tolerance
    or abs(regression.normal[index * 10 + 3] - regression.normal[index * 10 + 4])
    > tolerance
    for index in range(10)
  ):
    raise ValueError(f"{context} duplicate interpolation predictors disagree")
  try:
    intercept_energy = math.fsum(
      regression.normal[row * 10 + column]
      for row in (2, 3, 4, 5)
      for column in (2, 3, 4, 5)
    )
  except OverflowError as exc:
    raise ValueError(f"{context} interpolation arithmetic overflow") from exc
  if (
    not math.isfinite(intercept_energy)
    or abs(intercept_energy - regression.weight_s) > tolerance
  ):
    raise ValueError(f"{context} interpolation intercept energy is invalid")
  for index in (6, 7, 8, 9):
    if index in active:
      continue
    if abs(regression.rhs[index]) > tolerance or any(
      abs(regression.normal[index * 10 + other]) > tolerance
      or abs(regression.normal[other * 10 + index]) > tolerance
      for other in range(10)
    ):
      raise ValueError(f"{context} inactive sign cross term is nonzero")
  if active_indices is None:
    return
  lower, upper = active_indices
  for actual, expected in (
    (regression.normal[lower * 10 + lower], regression.predictor_sums[2]),
    (regression.normal[lower * 10 + upper], regression.predictor_sums[3]),
    (regression.normal[upper * 10 + upper], regression.predictor_sums[5]),
  ):
    if abs(actual - expected) > tolerance:
      raise ValueError(f"{context} sign/interpolation moments disagree")
  energy = math.fsum((
    regression.normal[lower * 10 + lower],
    2.0 * regression.normal[lower * 10 + upper],
    regression.normal[upper * 10 + upper],
  ))
  if not math.isfinite(energy) or abs(energy - regression.weight_s) > tolerance:
    raise ValueError(f"{context} sign-predictor energy is invalid")


_INTERVAL_STRATA = tuple(CalibrationIntervalStratum)


class _IntervalEvidence:
  """Disjoint runtime-interpolation evidence; strata never dilute each other."""

  __slots__ = tuple(stratum.value for stratum in _INTERVAL_STRATA)

  def __init__(self) -> None:
    for stratum in _INTERVAL_STRATA:
      setattr(self, stratum.value, _JointRegression())

  def regression(self, stratum: CalibrationIntervalStratum) -> _JointRegression:
    return getattr(self, stratum.value)

  def encoded(self) -> dict[str, Any]:
    return {
      stratum.value: self.regression(stratum).encoded()
      for stratum in _INTERVAL_STRATA
    }

  @classmethod
  def decoded(cls, raw: object, context: str) -> _IntervalEvidence:
    payload = _exact(raw, {stratum.value for stratum in _INTERVAL_STRATA}, context)
    result = cls()
    for stratum in _INTERVAL_STRATA:
      regression = _JointRegression.decoded(
        payload[stratum.value],
        f"{context}.{stratum.value}",
      )
      _validate_joint_sign_predictor(
        regression,
        (6, 7)
        if stratum in (
          CalibrationIntervalStratum.MOVING,
          CalibrationIntervalStratum.AUTHORITY,
        )
        else (8, 9)
        if stratum is CalibrationIntervalStratum.BREAKAWAY_EPISODE
        else None,
        f"{context}.{stratum.value}",
      )
      setattr(result, stratum.value, regression)
    return result


@dataclass(slots=True)
class _RouteSourceAccounting:
  """Physical source assignment committed before sufficient-statistic fitting."""

  base: int = 0
  moving: int = 0
  breakaway_episode: int = 0
  pending: int = 0
  authority_fit: int = 0
  authority_unresolved: int = 0

  @property
  def accepted(self) -> int:
    return (
      self.base
      + self.moving
      + self.breakaway_episode
      + self.pending
      + self.authority_fit
      + self.authority_unresolved
    )

  def encoded(self) -> dict[str, int]:
    return {
      "accepted": self.accepted,
      "authority_fit": self.authority_fit,
      "authority_unresolved": self.authority_unresolved,
      "base": self.base,
      "breakaway_episode": self.breakaway_episode,
      "moving": self.moving,
      "pending": self.pending,
    }

  @classmethod
  def decoded(cls, raw: object, context: str) -> _RouteSourceAccounting:
    payload = _exact(raw, {
      "accepted",
      "authority_fit",
      "authority_unresolved",
      "base",
      "breakaway_episode",
      "moving",
      "pending",
    }, context)
    values = {
      field: payload[field]
      for field in (
        "authority_fit",
        "authority_unresolved",
        "base",
        "breakaway_episode",
        "moving",
        "pending",
      )
    }
    values = {
      field: _bounded_count(value, f"{context}.{field}")
      for field, value in values.items()
    }
    result = cls(**values)
    accepted = _bounded_count(payload["accepted"], f"{context}.accepted")
    if accepted != result.accepted:
      raise ValueError(f"{context} accepted partition disagrees")
    return result


@dataclass(slots=True)
class _RouteEvidence:
  route_index: int
  route_counter: int
  route_identity_sha256: str
  route_content_sha256: str
  source_accounting: _RouteSourceAccounting
  nodes: tuple[_Node, ...]
  intervals: tuple[_IntervalEvidence, ...]
  assignment_record_count: int
  assignment_chain_sha256: str
  route_commitment_sha256: str
  non_authoritative_record_count: int


def _combine(*parts: _Regression) -> _Regression:
  result = _Regression()
  for i in range(16):
    result.normal[i] = math.fsum(part.normal[i] for part in parts)
  for i in range(4):
    result.rhs[i] = math.fsum(part.rhs[i] for part in parts)
  result.target_squared = math.fsum(part.target_squared for part in parts)
  result.weight_s = math.fsum(part.weight_s for part in parts)
  result.count = sum(part.count for part in parts)
  return result


def _subtract(whole: _Regression, *parts: _Regression) -> _Regression:
  """Recover one exactly accumulated disjoint stratum."""
  result = _Regression()
  result.normal[:] = whole.normal
  result.rhs[:] = whole.rhs
  result.target_squared = whole.target_squared
  result.weight_s = whole.weight_s
  result.count = whole.count
  for part in parts:
    for i in range(16):
      result.normal[i] -= part.normal[i]
    for i in range(4):
      result.rhs[i] -= part.rhs[i]
    result.target_squared -= part.target_squared
    result.weight_s -= part.weight_s
    result.count -= part.count
  if not all(
    math.isfinite(value)
    for value in (
      *result.normal,
      *result.rhs,
      result.target_squared,
      result.weight_s,
    )
  ):
    raise ValueError("calibration stratum arithmetic overflow")
  if result.count < 0 or result.weight_s < -1e-12:
    raise ValueError("calibration strata are not a disjoint partition")
  if result.weight_s < 0.0:
    result.weight_s = 0.0
  if result.target_squared < 0.0 and abs(result.target_squared) <= 1e-12:
    result.target_squared = 0.0
  _validate_augmented_gram(
    result.normal,
    result.rhs,
    result.target_squared,
    4,
    "derived calibration stratum",
  )
  tolerance = _stat_tolerance(
    max(result.weight_s, *(abs(value) for value in result.normal)),
    result.count,
    512.0,
  )
  if abs(result.normal[5] - result.weight_s) > tolerance:
    raise ValueError("derived calibration stratum intercept is invalid")
  if abs(result.normal[11]) > tolerance:
    raise ValueError("derived calibration stratum predictors overlap")
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
    self.moving_training_direction_mask = 0
    self.breakaway_direction_mask = 0
    self.breakaway_episode_direction_mask = 0
    self.breakaway_episode_training_direction_mask = 0
    self.breakaway_episode_count = 0
    self.breakaway_episode_dwell_s = 0.0
    self.breakaway_angle_assisted_count = 0
    self.breakaway_bracket_width_sum = 0.0
    self.training = _Regression()
    self.moving_training = _Regression()
    self.breakaway_training = _Regression()
    self.breakaway_episode_training = _Regression()
    self.authority_training = _Regression()


_NODE_FLOAT_SUM_FIELDS = (
  "clean_support_s",
  "base_support_s",
  "moving_support_s",
  "breakaway_support_s",
  "authority_support_s",
  "authority_fit_support_s",
  "lat_energy",
  "rack_travel_deg",
  "breakaway_episode_dwell_s",
  "breakaway_bracket_width_sum",
)
_NODE_INT_SUM_FIELDS = (
  "supported_sample_count",
  "base_sample_count",
  "moving_sample_count",
  "breakaway_sample_count",
  "authority_sample_count",
  "authority_magnitude_sample_count",
  "authority_slew_build_sample_count",
  "authority_slew_release_sample_count",
  "authority_fit_sample_count",
  "authority_unresolved_sample_count",
  "rack_reversals",
  "fit_count",
  "authority_fit_count",
  "breakaway_episode_count",
  "breakaway_angle_assisted_count",
)
_NODE_MASK_FIELDS = (
  "lateral_accel_direction_mask",
  "applied_torque_direction_mask",
  "moving_direction_mask",
  "moving_training_direction_mask",
  "breakaway_direction_mask",
  "breakaway_episode_direction_mask",
  "breakaway_episode_training_direction_mask",
)
_NODE_REGRESSION_FIELDS = (
  "training",
  "moving_training",
  "breakaway_training",
  "breakaway_episode_training",
  "authority_training",
)


def _aggregate_nodes(parts: tuple[_Node, ...]) -> _Node:
  """Combine complete route summaries in canonical order without drift."""
  result = _Node()
  if not parts:
    return result
  for name in _NODE_FLOAT_SUM_FIELDS:
    setattr(result, name, math.fsum(getattr(part, name) for part in parts))
  for name in _NODE_INT_SUM_FIELDS:
    setattr(result, name, sum(getattr(part, name) for part in parts))
  for name in _NODE_MASK_FIELDS:
    value = 0
    for part in parts:
      value |= getattr(part, name)
    setattr(result, name, value)
  finite_lat_min = tuple(part.lat_min for part in parts if math.isfinite(part.lat_min))
  finite_lat_max = tuple(part.lat_max for part in parts if math.isfinite(part.lat_max))
  finite_torque_min = tuple(part.torque_min for part in parts if math.isfinite(part.torque_min))
  finite_torque_max = tuple(part.torque_max for part in parts if math.isfinite(part.torque_max))
  result.lat_min = min(finite_lat_min, default=math.inf)
  result.lat_max = max(finite_lat_max, default=-math.inf)
  result.torque_min = min(finite_torque_min, default=math.inf)
  result.torque_max = max(finite_torque_max, default=-math.inf)
  for name in _NODE_REGRESSION_FIELDS:
    setattr(result, name, _combine(*(getattr(part, name) for part in parts)))
  # Schema 14 has no parity/global-validation population or persisted route
  # transient. These fields remain empty only for compatibility with internal
  # helpers shared by the live accumulator.
  return result


_NODE_SUMMARY_KEYS = {
  "node_index",
  "lat_min",
  "lat_max",
  "torque_min",
  "torque_max",
  *_NODE_FLOAT_SUM_FIELDS,
  *_NODE_INT_SUM_FIELDS,
  *_NODE_MASK_FIELDS,
  *_NODE_REGRESSION_FIELDS,
}


def _encode_node_summary(node: _Node, node_index: int) -> dict[str, Any]:
  result: dict[str, Any] = {
    "node_index": node_index,
    "lat_min": _extreme(node.lat_min),
    "lat_max": _extreme(node.lat_max),
    "torque_min": _extreme(node.torque_min),
    "torque_max": _extreme(node.torque_max),
  }
  for name in _NODE_FLOAT_SUM_FIELDS:
    result[name] = getattr(node, name).hex()
  for name in (*_NODE_INT_SUM_FIELDS, *_NODE_MASK_FIELDS):
    result[name] = getattr(node, name)
  for name in _NODE_REGRESSION_FIELDS:
    result[name] = getattr(node, name).encoded()
  return result


def _route_commitment_payload(
  route: _RouteEvidence,
) -> dict[str, object]:
  """Bounded replay commitment; embedded values are structural, not trust."""
  return {
    "assignment_chain_sha256": route.assignment_chain_sha256,
    "assignment_record_count": route.assignment_record_count,
    "intervals": [interval.encoded() for interval in route.intervals],
    "nodes": [
      _encode_node_summary(node, index)
      for index, node in enumerate(route.nodes)
    ],
    "non_authoritative_record_count": route.non_authoritative_record_count,
    "route_content_sha256": route.route_content_sha256,
    "route_counter": route.route_counter,
    "route_identity_sha256": route.route_identity_sha256,
    "source_accounting": route.source_accounting.encoded(),
  }


def _route_commitment(route: _RouteEvidence) -> str:
  return hashlib.sha256(
    b"blatv2-calibration-route-commitment-v1\0"
    + _canonical(_route_commitment_payload(route))
  ).hexdigest()


def _decode_node_summary(raw: object, node_index: int, context: str) -> _Node:
  payload = _exact(raw, _NODE_SUMMARY_KEYS, context)
  if payload["node_index"] != node_index:
    raise ValueError(f"{context} ordering is corrupt")
  node = _Node()
  node.lat_min = _decode_extreme(payload["lat_min"], True, f"{context}.lat_min")
  node.lat_max = _decode_extreme(payload["lat_max"], False, f"{context}.lat_max")
  node.torque_min = _decode_extreme(payload["torque_min"], True, f"{context}.torque_min")
  node.torque_max = _decode_extreme(payload["torque_max"], False, f"{context}.torque_max")
  for name in _NODE_FLOAT_SUM_FIELDS:
    value = _hex(payload[name], f"{context}.{name}")
    if value < 0.0:
      raise ValueError(f"{context}.{name} is negative")
    setattr(node, name, value)
  for name in _NODE_INT_SUM_FIELDS:
    value = _bounded_count(payload[name], f"{context}.{name}")
    setattr(node, name, value)
  for name in _NODE_MASK_FIELDS:
    value = payload[name]
    if type(value) is not int or value not in range(4):
      raise ValueError(f"{context}.{name} is invalid")
    setattr(node, name, value)
  for name in _NODE_REGRESSION_FIELDS:
    setattr(node, name, _Regression.decoded(payload[name], f"{context}.{name}"))
  _validate_sign_predictor(node.moving_training, 2, f"{context}.moving_training")
  _validate_sign_predictor(node.authority_training, 2, f"{context}.authority_training")
  _validate_sign_predictor(node.breakaway_training, 3, f"{context}.breakaway_training")
  _validate_sign_predictor(
    node.breakaway_episode_training,
    3,
    f"{context}.breakaway_episode_training",
  )
  base_training = _subtract(
    node.training,
    node.moving_training,
    node.breakaway_training,
  )
  _validate_sign_predictor(base_training, None, f"{context}.base_training")
  if node.fit_count != node.training.count:
    raise ValueError(f"{context} fit counts are inconsistent")
  if node.authority_fit_count != node.authority_training.count:
    raise ValueError(f"{context} authority counts are inconsistent")
  if node.supported_sample_count != node.base_sample_count + node.moving_sample_count + node.breakaway_sample_count:
    raise ValueError(f"{context} stratum counts are inconsistent")
  if node.moving_sample_count != node.moving_training.count:
    raise ValueError(f"{context} moving counts are inconsistent")
  if node.breakaway_sample_count != node.breakaway_training.count:
    raise ValueError(f"{context} breakaway counts are inconsistent")
  if node.breakaway_episode_count != node.breakaway_episode_training.count:
    raise ValueError(f"{context} breakaway episode counts are inconsistent")
  if node.breakaway_angle_assisted_count > node.breakaway_episode_count:
    raise ValueError(f"{context} angle-assisted count is inconsistent")
  if node.authority_fit_sample_count != node.authority_fit_count or node.authority_fit_sample_count > node.authority_sample_count:
    raise ValueError(f"{context} authority sample counts are inconsistent")
  if node.authority_unresolved_sample_count > node.authority_magnitude_sample_count:
    raise ValueError(f"{context} unresolved authority count is inconsistent")
  if not math.isclose(
    node.clean_support_s,
    math.fsum((node.base_support_s, node.moving_support_s, node.breakaway_support_s)),
    rel_tol=1e-12,
    abs_tol=1e-12,
  ):
    raise ValueError(f"{context} support is inconsistent")
  if node.authority_fit_support_s > node.authority_support_s + 1e-12:
    raise ValueError(f"{context} authority support is inconsistent")
  lat_empty = math.isinf(node.lat_min) and node.lat_min > 0.0 and math.isinf(node.lat_max) and node.lat_max < 0.0
  torque_empty = math.isinf(node.torque_min) and node.torque_min > 0.0 and math.isinf(node.torque_max) and node.torque_max < 0.0
  if not (lat_empty or math.isfinite(node.lat_min) and math.isfinite(node.lat_max) and node.lat_min <= node.lat_max):
    raise ValueError(f"{context} lateral extrema are inconsistent")
  if not (torque_empty or math.isfinite(node.torque_min) and math.isfinite(node.torque_max) and node.torque_min <= node.torque_max):
    raise ValueError(f"{context} torque extrema are inconsistent")
  return node


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
  full_fit_support_s: float
  full_fit_count: int
  moving_full_fit_support_s: float
  moving_full_fit_count: int
  breakaway_full_fit_support_s: float
  breakaway_full_fit_count: int
  breakaway_episode_training_weight: float
  breakaway_episode_full_fit_count: int
  breakaway_episode_dwell_s: float
  breakaway_angle_assisted_count: int
  authority_full_fit_support_s: float
  authority_full_fit_count: int
  completed_route_count: int
  base_completed_route_count: int
  moving_completed_route_count: int
  breakaway_episode_completed_route_count: int
  authority_completed_route_count: int
  lateral_accel_span_mps2: float
  applied_torque_span: float
  lateral_accel_directions: int
  applied_torque_directions: int
  rack_reversals: int


@dataclass(frozen=True, slots=True)
class CalibrationNodeQualificationReport:
  """Schema-14 full-fit result plus route-grouped cross-fit diagnostics."""
  node_index: int
  speed_mps: float
  minimum_support_s: float
  clean_support_s: float
  supported_sample_count: int
  full_fit_count: int
  cross_fit_route_count: int
  base_support_s: float
  base_sample_count: int
  moving_support_s: float
  moving_sample_count: int
  moving_full_fit_count: int
  moving_cross_fit_route_count: int
  breakaway_support_s: float
  breakaway_sample_count: int
  breakaway_full_fit_count: int
  breakaway_cross_fit_route_count: int
  lateral_accel_span_mps2: float
  lateral_accel_rms_mps2: float
  rack_travel_deg: float
  applied_torque_span: float
  rack_reversals: int
  lateral_accel_directions: int
  applied_torque_directions: int
  full_fit_seed_rms: float | None
  full_fit_candidate_rms: float | None
  moving_full_fit_seed_rms: float | None
  moving_full_fit_candidate_rms: float | None
  breakaway_full_fit_seed_rms: float | None
  breakaway_full_fit_candidate_rms: float | None
  confidence: float
  reasons: tuple[CalibrationQualificationReason, ...]
  candidate_parameters: CalibrationParameters | None
  authority_support_s: float = 0.0
  authority_sample_count: int = 0
  authority_fit_support_s: float = 0.0
  authority_fit_sample_count: int = 0
  authority_full_fit_count: int = 0
  authority_cross_fit_route_count: int = 0
  authority_full_fit_seed_rms: float | None = None
  authority_full_fit_candidate_rms: float | None = None
  authority_magnitude_sample_count: int = 0
  authority_slew_build_sample_count: int = 0
  authority_slew_release_sample_count: int = 0
  authority_unresolved_sample_count: int = 0
  selected_model: CalibrationModelId | None = None
  base_full_fit_seed_rms: float | None = None
  base_full_fit_candidate_rms: float | None = None
  breakaway_episode_full_fit_count: int = 0
  breakaway_episode_cross_fit_route_count: int = 0
  breakaway_episode_dwell_s: float = 0.0
  breakaway_angle_assisted_count: int = 0
  breakaway_episode_full_fit_seed_rms: float | None = None
  breakaway_episode_full_fit_candidate_rms: float | None = None
  breakaway_mean_bracket_width: float | None = None
  fit_diagnostics: tuple[CalibrationModelFitDiagnostic, ...] = ()
  full_fit_paired_loss: CalibrationPairedLossDiagnostic | None = None
  cross_fit_paired_loss: CalibrationPairedLossDiagnostic | None = None
  selection_outcome: CalibrationQualificationReason | None = None
  independent_route_counts: CalibrationIndependentRouteCounts | None = None
  cross_fit_diagnostics: tuple[CalibrationCrossFitModelDiagnostic, ...] = ()
  full_fit_diagnostic: CalibrationModelFitDiagnostic | None = None
  unresolved_diagnostics: tuple[CalibrationQualificationReason, ...] = ()
  full_fit_stratum_paired_losses: tuple[CalibrationPairedLossDiagnostic, ...] = ()

  @property
  def qualified(self) -> bool:
    return self.reasons in (
      (CalibrationQualificationReason.QUALIFIED,),
      (CalibrationQualificationReason.LEARNED,),
      (CalibrationQualificationReason.SEED_RETAINED,),
    )

  @property
  def learned(self) -> bool:
    return self.reasons == (CalibrationQualificationReason.LEARNED,)

  @property
  def seed_retained(self) -> bool:
    return self.reasons == (CalibrationQualificationReason.SEED_RETAINED,)


@dataclass(frozen=True, slots=True)
class CalibrationIntervalStratumDiagnostic:
  stratum: CalibrationIntervalStratum
  full_fit_paired_loss: CalibrationPairedLossDiagnostic
  cross_fit_paired_loss: CalibrationPairedLossDiagnostic
  contributing_route_count: int
  successful_fold_count: int
  failed_fold_count: int
  regressed_fold_count: int
  cross_fit_status: CalibrationCrossFitStatus


@dataclass(frozen=True, slots=True)
class CalibrationInterpolationQualificationReport:
  """Interval full-fit and route-grouped out-of-fold diagnostics."""
  interval_index: int
  lower_speed_mps: float
  upper_speed_mps: float
  stratum_diagnostics: tuple[CalibrationIntervalStratumDiagnostic, ...]
  reasons: tuple[CalibrationQualificationReason, ...]
  contributing_route_count: int = 0
  successful_fold_count: int = 0
  failed_fold_count: int = 0
  regressed_fold_count: int = 0
  cross_fit_status: CalibrationCrossFitStatus = CalibrationCrossFitStatus.INSUFFICIENT_INDEPENDENT_ROUTES

  @property
  def qualified(self) -> bool:
    return self.reasons == (CalibrationQualificationReason.QUALIFIED,)


@dataclass(frozen=True, slots=True)
class CalibrationLearningResult:
  node_reports: tuple[CalibrationNodeQualificationReport, ...]
  candidate_profile: VehicleCalibrationProfile | None
  interpolation_reports: tuple[
    CalibrationInterpolationQualificationReport, ...
  ] = ()
  # The evidence-qualified physical map selected by training and held-out
  # validation.  This exists for both learned and all-seed outcomes.  A
  # separate ``candidate_profile`` exists only when a controller-affecting
  # physical value changed, preserving the UI/artifact meaning of "new
  # calibration" without throwing away the proof needed by behavior replay.
  selected_profile: VehicleCalibrationProfile | None = None

  @property
  def all_nodes_qualified(self) -> bool:
    return all(report.qualified for report in self.node_reports) and all(
      report.qualified for report in self.interpolation_reports
    )

  @property
  def contains_learned_change(self) -> bool:
    return any(report.learned for report in self.node_reports)


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
  except (OverflowError, ValueError) as exc:
    raise ValueError(f"{context} is invalid") from exc
  if (
    not math.isfinite(value)
    or abs(value) > MAX_CALIBRATION_WIRE_FLOAT_ABS
    or value.hex() != raw
  ):
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


def _matrix_rank_and_condition(
  evidence: _Regression,
  free: tuple[int, ...],
) -> tuple[int, float | None]:
  """Return deterministic scaled rank and a pivot-ratio condition estimate.

  This is diagnostic and eligibility logic only. No ridge term or other
  regularization is permitted: an unidentifiable model must yield to a simpler
  nested family, never be manufactured into identifiability.
  """
  size = len(free)
  if size == 0:
    return 0, 1.0
  scales: list[float] = []
  for index in free:
    diagonal = evidence.normal[index * 4 + index]
    if not math.isfinite(diagonal) or diagonal <= 0.0:
      return 0, None
    scales.append(math.sqrt(diagonal))
  matrix = [
    [
      evidence.normal[row * 4 + column] / (scales[r] * scales[c])
      for c, column in enumerate(free)
    ]
    for r, row in enumerate(free)
  ]
  pivots: list[float] = []
  rank = 0
  for column in range(size):
    pivot = max(range(column, size), key=lambda row: abs(matrix[row][column]))
    magnitude = abs(matrix[pivot][column])
    if magnitude < NORMAL_MATRIX_RELATIVE_PIVOT_MIN:
      continue
    matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
    divisor = matrix[column][column]
    pivots.append(abs(divisor))
    rank += 1
    for row in range(column + 1, size):
      factor = matrix[row][column] / divisor
      for c in range(column, size):
        matrix[row][c] -= factor * matrix[column][c]
  condition = None if not pivots else max(pivots) / min(pivots)
  return rank, condition


def _model_fit_diagnostic(
  model: CalibrationModelId,
  moving: _Regression,
  episodes: _Regression,
  free: tuple[int, ...],
  solved: tuple[float, float, float, float] | None,
) -> CalibrationModelFitDiagnostic:
  moving_rank, condition = _matrix_rank_and_condition(moving, free)
  moving_count = len(free)
  breakaway_rank = int(
    episodes.count > 0
    and math.isfinite(episodes.normal[15])
    and episodes.normal[15] > 0.0
  )
  if moving_rank < moving_count or breakaway_rank < 1:
    status = CalibrationFitStatus.RANK_DEFICIENT
  elif condition is not None and condition > FIT_CONDITION_LIMIT:
    status = CalibrationFitStatus.ILL_CONDITIONED
  elif solved is None:
    status = CalibrationFitStatus.NO_SOLUTION
  else:
    status = CalibrationFitStatus.IDENTIFIABLE
  return CalibrationModelFitDiagnostic(
    model=model,
    status=status,
    moving_rank=moving_rank,
    moving_parameter_count=moving_count,
    condition_estimate=condition,
    breakaway_rank=breakaway_rank,
    breakaway_parameter_count=1,
  )


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


def _fit_bounded_subset(
  evidence: _Regression,
  seed: tuple[float, float, float, float],
  free: tuple[int, ...],
  *,
  kinetic_upper: float | None = None,
) -> tuple[float, float, float, float] | None:
  """Fit one nested moving-map model on deterministic active faces.

  Gain and kinetic friction carry their physical non-negative bounds.  When
  static friction remains fixed to the seed, kinetic also cannot exceed that
  fixed static value.  Coefficients outside ``free`` remain byte-identical to
  the seed comparator.
  """
  if evidence.count == 0:
    return None
  if any(index not in range(4) for index in free) or 3 in free:
    raise ValueError("moving-map subset contains an invalid free coefficient")

  gain_faces: tuple[float | None, ...] = (
    (None, 0.0) if 0 in free else (seed[0],)
  )
  if 2 not in free:
    kinetic_faces: tuple[float | None, ...] = (seed[2],)
  elif kinetic_upper is None:
    kinetic_faces = (None, 0.0)
  else:
    kinetic_faces = (None, 0.0, kinetic_upper)

  best: tuple[float, float, float, float] | None = None
  best_error: float | None = None
  for gain_face in gain_faces:
    for kinetic_face in kinetic_faces:
      coefficients = list(seed)
      fixed = {index for index in range(4) if index not in free}
      if 0 in free and gain_face is not None:
        coefficients[0] = gain_face
        fixed.add(0)
      if 2 in free and kinetic_face is not None:
        coefficients[2] = kinetic_face
        fixed.add(2)
      solve_free = tuple(index for index in free if index not in fixed)
      adjusted_rhs = list(evidence.rhs)
      for row in solve_free:
        adjusted_rhs[row] -= sum(
          evidence.normal[row * 4 + column] * coefficients[column]
          for column in fixed
        )
      solved = _solve_free_system(
        tuple(evidence.normal),
        tuple(adjusted_rhs),
        solve_free,
      )
      if solved is None:
        continue
      for index, value in zip(solve_free, solved, strict=True):
        coefficients[index] = value
      gain = coefficients[0]
      kinetic = coefficients[2]
      if gain < 0.0 or kinetic < 0.0:
        continue
      if kinetic_upper is not None and kinetic > kinetic_upper:
        continue
      candidate = tuple(coefficients)
      error = _weighted_error(evidence, candidate)
      if error is not None and (best_error is None or error < best_error):
        best = candidate
        best_error = error
  return best


def _fit_episode_static(
  episode_evidence: _Regression,
  moving_coefficients: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
  """Fit the static term from one midpoint row per physical episode."""
  denominator = episode_evidence.normal[15]
  if episode_evidence.count == 0 or denominator <= 0.0:
    return None
  gain, intercept, kinetic, _ = moving_coefficients
  unconstrained = (
    episode_evidence.rhs[3]
    - episode_evidence.normal[12] * gain
    - episode_evidence.normal[13] * intercept
    - episode_evidence.normal[14] * kinetic
  ) / denominator
  if not math.isfinite(unconstrained):
    return None
  # This is the exact active-face solution of S >= K, not post-fit clipping.
  static = unconstrained if unconstrained >= kinetic else kinetic
  candidate = (gain, intercept, kinetic, static)
  return candidate if _weighted_error(episode_evidence, candidate) is not None else None


@dataclass(frozen=True, slots=True)
class _ModelCandidate:
  model: CalibrationModelId
  coefficients: tuple[float, float, float, float]
  training_rms: tuple[float, ...]


def _rms_vector(
  regressions: tuple[_Regression, ...],
  coefficients: tuple[float, float, float, float],
) -> tuple[float, ...] | None:
  values: list[float] = []
  for regression in regressions:
    if regression.count == 0:
      continue
    value = regression.rms(coefficients)
    if value is None:
      return None
    values.append(value)
  return tuple(values)


class _PairedLossVerdict(StrEnum):
  IMPROVED = "improved"
  NO_REGRESSION = "no_regression"
  INCONCLUSIVE = "inconclusive"
  REGRESSION = "regression"
  NO_DATA = "no_data"


def calibration_cross_fit_status(
  *,
  contributing_route_count: int,
  successful_fold_count: int,
  failed_fold_count: int,
  regressed_fold_count: int,
  paired_loss_verdict: _PairedLossVerdict,
  coverage_sufficient: bool,
  interval: bool,
) -> CalibrationCrossFitStatus:
  """Derive the one authoritative status from observable fold accounting."""
  counts = (
    contributing_route_count,
    successful_fold_count,
    failed_fold_count,
    regressed_fold_count,
  )
  if any(type(count) is not int or count < 0 for count in counts):
    raise ValueError("cross-fit fold counts must be nonnegative integers")
  if (
    type(coverage_sufficient) is not bool
    or type(interval) is not bool
    or type(paired_loss_verdict) is not _PairedLossVerdict
  ):
    raise ValueError("cross-fit status inputs are invalid")

  # No fold was attempted when the required independent population is absent.
  # Contributing evidence is coverage, not a fabricated fit failure.
  if not coverage_sufficient:
    if successful_fold_count != 0 or failed_fold_count != 0 or regressed_fold_count != 0:
      raise ValueError("insufficient cross-fit coverage has attempted folds")
    if paired_loss_verdict is not _PairedLossVerdict.NO_DATA:
      raise ValueError("unattempted cross-fit carries paired-loss evidence")
    return CalibrationCrossFitStatus.INSUFFICIENT_INDEPENDENT_ROUTES
  if successful_fold_count == failed_fold_count == 0:
    raise ValueError("sufficient cross-fit coverage has no attempted folds")
  if successful_fold_count + failed_fold_count != contributing_route_count:
    raise ValueError("cross-fit attempted-fold counts disagree with contributors")
  if regressed_fold_count > successful_fold_count:
    raise ValueError("cross-fit regressed-fold count exceeds successful folds")
  if failed_fold_count:
    return CalibrationCrossFitStatus.FOLD_FIT_FAILURE
  if paired_loss_verdict is _PairedLossVerdict.NO_DATA:
    raise ValueError("attempted cross-fit lacks paired-loss evidence")
  if regressed_fold_count:
    return CalibrationCrossFitStatus.HELD_OUT_REGRESSION
  if paired_loss_verdict is _PairedLossVerdict.REGRESSION:
    raise ValueError("aggregate regression lacks a regressed-fold witness")
  if paired_loss_verdict is _PairedLossVerdict.IMPROVED or (
    interval and paired_loss_verdict is _PairedLossVerdict.NO_REGRESSION
  ):
    return CalibrationCrossFitStatus.SCORED
  return CalibrationCrossFitStatus.NO_ROBUST_IMPROVEMENT


def _loss_tolerance(candidate_mse: float, comparator_mse: float) -> float:
  return (
    NUMERICAL_LOSS_EPSILON_MULTIPLIER
    * sys.float_info.epsilon
    * max(abs(candidate_mse), abs(comparator_mse), 1.0)
  )


def _paired_route_losses(
  regressions: tuple[_Regression, ...],
  candidate: tuple[float, float, float, float],
  comparator: tuple[float, float, float, float],
) -> CalibrationPairedLossDiagnostic:
  deltas: list[float] = []
  tolerance = 0.0
  for evidence in regressions:
    if evidence.count == 0:
      continue
    candidate_mse = evidence.mse(candidate)
    comparator_mse = evidence.mse(comparator)
    if candidate_mse is None or comparator_mse is None:
      continue
    route_tolerance = _loss_tolerance(candidate_mse, comparator_mse)
    tolerance = max(tolerance, route_tolerance)
    delta = candidate_mse - comparator_mse
    deltas.append(0.0 if abs(delta) <= route_tolerance else delta)
  if not deltas:
    return CalibrationPairedLossDiagnostic(0, None, None, None, None, None)
  mean = math.fsum(deltas) / len(deltas)
  if len(deltas) < 2:
    uncertainty = None
    lower = None
    upper = None
  else:
    # The observed whole-route envelope is the uncertainty margin. It does not
    # shrink merely because more correlated 100 Hz rows (or more similar
    # routes) arrive, and it introduces no selected confidence multiplier.
    # Thus a supported improvement must survive every observed route.
    uncertainty = max(abs(delta - mean) for delta in deltas)
    lower = mean - uncertainty
    upper = mean + uncertainty
  return CalibrationPairedLossDiagnostic(
    len(deltas), mean, uncertainty, lower, upper, tolerance
  )


def _paired_loss_verdict(
  diagnostic: CalibrationPairedLossDiagnostic,
  *,
  identical: bool = False,
) -> _PairedLossVerdict:
  if diagnostic.route_count == 0:
    return _PairedLossVerdict.NO_DATA
  if identical:
    if diagnostic.mean_candidate_minus_seed_mse != 0.0:
      raise ValueError("identical paired loss is nonzero")
    return _PairedLossVerdict.NO_REGRESSION
  if diagnostic.upper_bound_mse is None or diagnostic.lower_bound_mse is None:
    return _PairedLossVerdict.INCONCLUSIVE
  tolerance = diagnostic.numerical_tolerance_mse or 0.0
  if diagnostic.lower_bound_mse > tolerance:
    return _PairedLossVerdict.REGRESSION
  if diagnostic.upper_bound_mse < -tolerance:
    return _PairedLossVerdict.IMPROVED
  if diagnostic.upper_bound_mse <= tolerance:
    return _PairedLossVerdict.NO_REGRESSION
  return _PairedLossVerdict.INCONCLUSIVE


def _route_regressions_for_node(
  routes: tuple[_RouteEvidence, ...],
  node_index: int,
  field: str,
) -> tuple[_Regression, ...]:
  result: list[_Regression] = []
  for route in _canonical_routes(routes):
    route_node = route.nodes[node_index]
    if field == "base_training":
      regression = _subtract(
        route_node.training,
        route_node.moving_training,
        route_node.breakaway_training,
      )
    else:
      regression = getattr(route_node, field)
    if regression.count > 0:
      result.append(regression)
  return tuple(result)


def _safe_on_route_strata(
  routes: tuple[_RouteEvidence, ...],
  node_index: int,
  fields: tuple[str, ...],
  candidate: tuple[float, float, float, float],
  comparator: tuple[float, float, float, float],
  *,
  require_improvement: bool,
) -> tuple[bool, tuple[CalibrationPairedLossDiagnostic, ...]]:
  diagnostics = tuple(
    _paired_route_losses(
      _route_regressions_for_node(routes, node_index, field),
      candidate,
      comparator,
    )
    for field in fields
  )
  verdicts = tuple(
    _paired_loss_verdict(
      diagnostic,
      identical=candidate == comparator,
    )
    for diagnostic in diagnostics
  )
  if not diagnostics or any(
    verdict in (
      _PairedLossVerdict.REGRESSION,
      _PairedLossVerdict.INCONCLUSIVE,
      _PairedLossVerdict.NO_DATA,
    )
    for verdict in verdicts
  ):
    return False, diagnostics
  improved = any(verdict is _PairedLossVerdict.IMPROVED for verdict in verdicts)
  return improved or not require_improvement, diagnostics


def _paired_joint_route_losses(
  routes: tuple[_RouteEvidence, ...],
  interval_index: int,
  stratum: CalibrationIntervalStratum,
  candidate_lower: CalibrationParameters,
  candidate_upper: CalibrationParameters,
  seed_lower: CalibrationParameters,
  seed_upper: CalibrationParameters,
) -> CalibrationPairedLossDiagnostic:
  deltas: list[float] = []
  tolerance = 0.0
  for route in _canonical_routes(routes):
    evidence = route.intervals[interval_index].regression(stratum)
    if evidence.count == 0:
      continue
    candidate_mse = evidence.mse(candidate_lower, candidate_upper)
    seed_mse = evidence.mse(seed_lower, seed_upper)
    if candidate_mse is None or seed_mse is None:
      continue
    route_tolerance = _loss_tolerance(candidate_mse, seed_mse)
    tolerance = max(tolerance, route_tolerance)
    delta = candidate_mse - seed_mse
    deltas.append(0.0 if abs(delta) <= route_tolerance else delta)
  if not deltas:
    return CalibrationPairedLossDiagnostic(0, None, None, None, None, None)
  mean = math.fsum(deltas) / len(deltas)
  if len(deltas) < 2:
    uncertainty = lower = upper = None
  else:
    uncertainty = max(abs(delta - mean) for delta in deltas)
    lower, upper = mean - uncertainty, mean + uncertainty
  return CalibrationPairedLossDiagnostic(
    len(deltas), mean, uncertainty, lower, upper, tolerance
  )


_MODEL_FAMILIES: tuple[tuple[CalibrationModelId, tuple[int, ...]], ...] = (
  (CalibrationModelId.STATIC_ONLY, ()),
  (CalibrationModelId.FRICTION_MAP, (2,)),
  (CalibrationModelId.OFFSET_AND_FRICTION, (1, 2)),
  (CalibrationModelId.FULL_MAP, (0, 1, 2)),
)


def _canonical_routes(routes: tuple[_RouteEvidence, ...]) -> tuple[_RouteEvidence, ...]:
  return tuple(sorted(
    routes,
    key=lambda route: (
      route.route_identity_sha256,
      route.route_content_sha256,
      route.route_counter,
    ),
  ))


def _canonical_committed_routes(
  routes: tuple[_RouteEvidence, ...],
) -> tuple[_RouteEvidence, ...]:
  committed: list[_RouteEvidence] = []
  for route_index, route in enumerate(_canonical_routes(routes)):
    canonical = replace(
      route,
      route_index=route_index,
      route_commitment_sha256="",
    )
    canonical.route_commitment_sha256 = _route_commitment(canonical)
    committed.append(canonical)
  return tuple(committed)


def _route_node_regression(
  route: _RouteEvidence,
  node_index: int,
  field: str,
) -> _Regression:
  node = route.nodes[node_index]
  if field == "base_training":
    return _subtract(node.training, node.moving_training, node.breakaway_training)
  return getattr(node, field)


def _contributing_routes(
  routes: tuple[_RouteEvidence, ...],
  node_index: int,
  field: str,
) -> tuple[_RouteEvidence, ...]:
  return tuple(
    route for route in _canonical_routes(routes)
    if _route_node_regression(route, node_index, field).count > 0
  )


# These are mutually-exclusive physical scoring populations. ``training`` is
# their legacy aggregate and ``breakaway_training`` is the first-motion row of
# the same episode represented by ``breakaway_episode_training``; including
# either would count observations twice during family selection.
_DISJOINT_CROSS_FIT_FIELDS = (
  "base_training",
  "moving_training",
  "breakaway_episode_training",
  "authority_training",
)


def _independent_route_counts(
  routes: tuple[_RouteEvidence, ...],
  node_index: int,
) -> CalibrationIndependentRouteCounts:
  canonical = _canonical_routes(routes)
  contributors = {
    route.route_identity_sha256
    for route in canonical
    if any(
      _route_node_regression(route, node_index, field).count > 0
      for field in _DISJOINT_CROSS_FIT_FIELDS
    )
  }
  return CalibrationIndependentRouteCounts(
    all=len(contributors),
    base=len(_contributing_routes(canonical, node_index, "base_training")),
    moving=len(_contributing_routes(canonical, node_index, "moving_training")),
    breakaway=len(_contributing_routes(canonical, node_index, "breakaway_training")),
    breakaway_episode=len(_contributing_routes(canonical, node_index, "breakaway_episode_training")),
    authority=len(_contributing_routes(canonical, node_index, "authority_training")),
  )


_INTERVAL_NODE_FIELDS = {
  CalibrationIntervalStratum.BASE: "base_training",
  CalibrationIntervalStratum.MOVING: "moving_training",
  CalibrationIntervalStratum.BREAKAWAY_EPISODE: "breakaway_episode_training",
  CalibrationIntervalStratum.AUTHORITY: "authority_training",
}


def _accumulator_close(left: float, right: float, row_count: int) -> bool:
  tolerance = _stat_tolerance(
    max(abs(left), abs(right)),
    _bounded_count(row_count, "calibration accumulator row_count"),
    NUMERICAL_LOSS_EPSILON_MULTIPLIER,
  )
  return abs(left - right) <= tolerance


def _validate_interval_conservation(route: _RouteEvidence) -> None:
  """Prove interval strata are the same disjoint rows as node evidence."""
  for stratum, node_field in _INTERVAL_NODE_FIELDS.items():
    node_regressions = tuple(
      _route_node_regression(route, node_index, node_field)
      for node_index in range(len(route.nodes))
    )
    interval_regressions = tuple(
      interval.regression(stratum) for interval in route.intervals
    )
    node_count = sum(regression.count for regression in node_regressions)
    interval_count = sum(regression.count for regression in interval_regressions)
    source_field = (
      "breakaway_episode"
      if stratum is CalibrationIntervalStratum.BREAKAWAY_EPISODE
      else "authority_fit"
      if stratum is CalibrationIntervalStratum.AUTHORITY
      else stratum.value
    )
    source_count = getattr(route.source_accounting, source_field)
    if (
      interval_count != source_count
      or (interval_count == 0) != (node_count == 0)
      or node_count < interval_count
      or node_count > 2 * interval_count
    ):
      raise ValueError(f"route interval {stratum.value} row conservation failed")
    node_weight = math.fsum(regression.weight_s for regression in node_regressions)
    interval_weight = math.fsum(
      regression.weight_s for regression in interval_regressions
    )
    node_target_squared = math.fsum(
      regression.target_squared for regression in node_regressions
    )
    interval_target_squared = math.fsum(
      regression.target_squared for regression in interval_regressions
    )
    if not _accumulator_close(node_weight, interval_weight, node_count) or not (
      _accumulator_close(
        node_target_squared,
        interval_target_squared,
        node_count,
      )
    ):
      raise ValueError(f"route interval {stratum.value} support conservation failed")


def _family_fit_fields(model: CalibrationModelId, include_authority: bool) -> tuple[str, ...]:
  return tuple(
    f"{population}_training"
    for population in calibration_cross_fit_required_populations(
      model,
      include_authority=include_authority,
    )
  )


def calibration_cross_fit_required_populations(
  model: CalibrationModelId,
  *,
  include_authority: bool,
) -> tuple[str, ...]:
  if type(model) is not CalibrationModelId or type(include_authority) is not bool:
    raise ValueError("cross-fit population selector is invalid")
  populations = ("breakaway_episode",)
  if model is not CalibrationModelId.STATIC_ONLY:
    populations = ("moving", *populations)
    if include_authority:
      populations = (*populations, "authority")
  return populations


def _family_population_fields(include_authority: bool) -> tuple[str, ...]:
  return tuple(
    field for field in _DISJOINT_CROSS_FIT_FIELDS
    if include_authority or field != "authority_training"
  )


def _fit_model_family(
  routes: tuple[_RouteEvidence, ...],
  node_index: int,
  model: CalibrationModelId,
  seed: tuple[float, float, float, float],
  *,
  include_authority: bool,
) -> tuple[tuple[float, float, float, float] | None, CalibrationModelFitDiagnostic]:
  canonical = _canonical_routes(routes)
  route_nodes = tuple(route.nodes[node_index] for route in canonical)
  moving = _combine(*(node.moving_training for node in route_nodes))
  if include_authority:
    moving = _combine(moving, *(node.authority_training for node in route_nodes))
  episodes = _combine(*(node.breakaway_episode_training for node in route_nodes))
  free = dict(_MODEL_FAMILIES)[model]
  combined = {
    field: _combine(*(_route_node_regression(route, node_index, field) for route in canonical))
    for field in _family_fit_fields(model, include_authority)
  }
  if any(regression.count < MIN_STRATUM_TRAINING_ROWS for regression in combined.values()):
    return None, _model_fit_diagnostic(model, moving, episodes, free, None)
  if model is CalibrationModelId.STATIC_ONLY:
    coefficients = _fit_episode_static(episodes, seed)
  else:
    moving_coefficients = _fit_bounded_subset(moving, seed, free)
    coefficients = None if moving_coefficients is None else _fit_episode_static(episodes, moving_coefficients)
  return coefficients, _model_fit_diagnostic(model, moving, episodes, free, coefficients)


def _diagnostic_from_deltas(
  deltas: tuple[tuple[float, float], ...],
) -> CalibrationPairedLossDiagnostic:
  if not deltas:
    return CalibrationPairedLossDiagnostic(0, None, None, None, None, None)
  values = tuple(delta for delta, _ in deltas)
  tolerance = max(tolerance for _, tolerance in deltas)
  mean = math.fsum(values) / len(values)
  if len(values) < MIN_INDEPENDENT_ROUTES:
    uncertainty = lower = upper = None
  else:
    uncertainty = max(abs(delta - mean) for delta in values)
    lower, upper = mean - uncertainty, mean + uncertainty
  return CalibrationPairedLossDiagnostic(len(values), mean, uncertainty, lower, upper, tolerance)


@dataclass(frozen=True, slots=True)
class _CrossFitFamily:
  model: CalibrationModelId
  coefficients_by_route: tuple[tuple[str, tuple[float, float, float, float]], ...]
  diagnostic: CalibrationCrossFitModelDiagnostic


def _cross_fit_family(
  routes: tuple[_RouteEvidence, ...],
  node_index: int,
  model: CalibrationModelId,
  seed: tuple[float, float, float, float],
  fields: tuple[str, ...],
) -> _CrossFitFamily:
  del fields  # Selection owns one disjoint physical objective vocabulary.
  canonical = _canonical_routes(routes)
  include_authority = any(route.nodes[node_index].authority_training.count > 0 for route in canonical)
  population_fields = _family_population_fields(include_authority)
  contributing = tuple(
    route for route in canonical
    if any(_route_node_regression(route, node_index, field).count > 0 for field in population_fields)
  )
  required_fields = _family_fit_fields(model, include_authority)
  lacks_independence = (
    len(contributing) < MIN_INDEPENDENT_ROUTES
    or any(len(_contributing_routes(canonical, node_index, field)) < MIN_INDEPENDENT_ROUTES for field in required_fields)
  )
  if lacks_independence:
    paired = _diagnostic_from_deltas(())
    return _CrossFitFamily(
      model,
      (),
      CalibrationCrossFitModelDiagnostic(
        model,
        calibration_cross_fit_status(
          contributing_route_count=len(contributing),
          successful_fold_count=0,
          failed_fold_count=0,
          regressed_fold_count=0,
          paired_loss_verdict=_paired_loss_verdict(paired),
          coverage_sufficient=False,
          interval=False,
        ),
        len(contributing),
        0,
        0,
        0,
        paired,
      ),
    )

  fold_coefficients: list[tuple[str, tuple[float, float, float, float]]] = []
  deltas: list[tuple[float, float]] = []
  failures = 0
  regressed_folds = 0
  for held_out in contributing:
    fit_routes = tuple(route for route in canonical if route is not held_out)
    coefficients, fit_diagnostic = _fit_model_family(
      fit_routes,
      node_index,
      model,
      seed,
      include_authority=include_authority,
    )
    if coefficients is None or fit_diagnostic.status is not CalibrationFitStatus.IDENTIFIABLE:
      failures += 1
      continue
    candidate_errors: list[float] = []
    seed_errors: list[float] = []
    weights: list[float] = []
    fold_safe = True
    fold_regression = False
    fold_tolerance = 0.0
    for field in population_fields:
      evidence = _route_node_regression(held_out, node_index, field)
      if evidence.count == 0:
        continue
      candidate_mse = evidence.mse(coefficients)
      seed_mse = evidence.mse(seed)
      if candidate_mse is None or seed_mse is None:
        fold_safe = False
        break
      tolerance = _loss_tolerance(candidate_mse, seed_mse)
      fold_tolerance = max(fold_tolerance, tolerance)
      if candidate_mse - seed_mse > tolerance:
        fold_regression = True
      candidate_errors.append(candidate_mse * evidence.weight_s)
      seed_errors.append(seed_mse * evidence.weight_s)
      weights.append(evidence.weight_s)
    total_weight = math.fsum(weights)
    if not fold_safe or total_weight <= 0.0:
      failures += 1
      continue
    delta = (math.fsum(candidate_errors) - math.fsum(seed_errors)) / total_weight
    deltas.append((0.0 if abs(delta) <= fold_tolerance else delta, fold_tolerance))
    fold_coefficients.append((held_out.route_identity_sha256, coefficients))
    if fold_regression:
      regressed_folds += 1

  paired = _diagnostic_from_deltas(tuple(deltas))
  verdict = _paired_loss_verdict(paired)
  status = calibration_cross_fit_status(
    contributing_route_count=len(contributing),
    successful_fold_count=len(deltas),
    failed_fold_count=failures,
    regressed_fold_count=regressed_folds,
    paired_loss_verdict=verdict,
    coverage_sufficient=True,
    interval=False,
  )
  return _CrossFitFamily(
    model,
    tuple(fold_coefficients),
    CalibrationCrossFitModelDiagnostic(
      model,
      status,
      len(contributing),
      len(deltas),
      failures,
      regressed_folds,
      paired,
    ),
  )


def _cross_fit_dominates(
  routes: tuple[_RouteEvidence, ...],
  node_index: int,
  candidate: _CrossFitFamily,
  comparator: _CrossFitFamily,
) -> bool:
  candidate_by_route = dict(candidate.coefficients_by_route)
  comparator_by_route = dict(comparator.coefficients_by_route)
  if set(candidate_by_route) != set(comparator_by_route):
    return False
  include_authority = any(route.nodes[node_index].authority_training.count > 0 for route in routes)
  fields = _family_population_fields(include_authority)
  deltas: list[tuple[float, float]] = []
  for route in _canonical_routes(routes):
    candidate_coefficients = candidate_by_route.get(route.route_identity_sha256)
    comparator_coefficients = comparator_by_route.get(route.route_identity_sha256)
    if candidate_coefficients is None or comparator_coefficients is None:
      continue
    candidate_errors: list[float] = []
    comparator_errors: list[float] = []
    weights: list[float] = []
    tolerance = 0.0
    for field in fields:
      evidence = _route_node_regression(route, node_index, field)
      if evidence.count == 0:
        continue
      candidate_mse = evidence.mse(candidate_coefficients)
      comparator_mse = evidence.mse(comparator_coefficients)
      if candidate_mse is None or comparator_mse is None:
        return False
      field_tolerance = _loss_tolerance(candidate_mse, comparator_mse)
      tolerance = max(tolerance, field_tolerance)
      if candidate_mse - comparator_mse > field_tolerance:
        return False
      candidate_errors.append(candidate_mse * evidence.weight_s)
      comparator_errors.append(comparator_mse * evidence.weight_s)
      weights.append(evidence.weight_s)
    weight = math.fsum(weights)
    if weight <= 0.0:
      return False
    delta = (math.fsum(candidate_errors) - math.fsum(comparator_errors)) / weight
    deltas.append((0.0 if abs(delta) <= tolerance else delta, tolerance))
  return _paired_loss_verdict(_diagnostic_from_deltas(tuple(deltas))) is _PairedLossVerdict.IMPROVED


def _cross_fit_interval_loss(
  routes: tuple[_RouteEvidence, ...],
  interval_index: int,
  stratum: CalibrationIntervalStratum,
  lower_model: CalibrationModelId | None,
  upper_model: CalibrationModelId | None,
  seed_lower: CalibrationParameters,
  seed_upper: CalibrationParameters,
  *,
  allow_single_exact_seed: bool = False,
) -> tuple[CalibrationPairedLossDiagnostic, int, int, int, int, CalibrationCrossFitStatus]:
  exact_seed = lower_model is None and upper_model is None
  if allow_single_exact_seed and not exact_seed:
    raise ValueError("single-route interval authority requires exact seed")
  minimum_independent_routes = (
    MIN_TERMINAL_SEED_AUTHORITY_ROUTES
    if allow_single_exact_seed
    else MIN_INDEPENDENT_ROUTES
  )
  canonical = _canonical_routes(routes)
  contributing = tuple(
    route for route in canonical
    if route.intervals[interval_index].regression(stratum).count > 0
  )
  if len(contributing) < minimum_independent_routes:
    diagnostic = _diagnostic_from_deltas(())
    return (
      diagnostic, len(contributing), 0, 0, 0,
      calibration_cross_fit_status(
        contributing_route_count=len(contributing),
        successful_fold_count=0,
        failed_fold_count=0,
        regressed_fold_count=0,
        paired_loss_verdict=_paired_loss_verdict(diagnostic),
        coverage_sufficient=False,
        interval=True,
      ),
    )
  deltas: list[tuple[float, float]] = []
  failures = 0
  regressed_folds = 0
  for held_out in contributing:
    fit_routes = tuple(route for route in canonical if route is not held_out)
    lower_coefficients = _seed_coefficients(seed_lower)
    upper_coefficients = _seed_coefficients(seed_upper)
    if lower_model is not None:
      lower_coefficients, lower_diagnostic = _fit_model_family(
        fit_routes,
        interval_index,
        lower_model,
        lower_coefficients,
        include_authority=any(route.nodes[interval_index].authority_training.count > 0 for route in canonical),
      )
      if lower_coefficients is None or lower_diagnostic.status is not CalibrationFitStatus.IDENTIFIABLE:
        failures += 1
        continue
    if upper_model is not None:
      upper_coefficients, upper_diagnostic = _fit_model_family(
        fit_routes,
        interval_index + 1,
        upper_model,
        upper_coefficients,
        include_authority=any(route.nodes[interval_index + 1].authority_training.count > 0 for route in canonical),
      )
      if upper_coefficients is None or upper_diagnostic.status is not CalibrationFitStatus.IDENTIFIABLE:
        failures += 1
        continue

    def parameters(
      seed: CalibrationParameters,
      coefficients: tuple[float, float, float, float],
    ) -> CalibrationParameters:
      gain, intercept, kinetic, static = coefficients
      return CalibrationParameters(
        gain,
        intercept / gain,
        kinetic,
        static,
        seed.transport_delay_s,
        seed.rack_rate_resolution_deg_s,
        seed.confidence,
        False,
      )

    held_out_evidence = held_out.intervals[interval_index].regression(stratum)
    candidate_mse = held_out_evidence.mse(
      parameters(seed_lower, lower_coefficients),
      parameters(seed_upper, upper_coefficients),
    )
    seed_mse = held_out_evidence.mse(seed_lower, seed_upper)
    if candidate_mse is None or seed_mse is None:
      failures += 1
      continue
    tolerance = _loss_tolerance(candidate_mse, seed_mse)
    delta = candidate_mse - seed_mse
    deltas.append((0.0 if abs(delta) <= tolerance else delta, tolerance))
    if delta > tolerance:
      regressed_folds += 1
  diagnostic = _diagnostic_from_deltas(tuple(deltas))
  verdict = _paired_loss_verdict(
    diagnostic,
    identical=exact_seed,
  )
  status = calibration_cross_fit_status(
    contributing_route_count=len(contributing),
    successful_fold_count=len(deltas),
    failed_fold_count=failures,
    regressed_fold_count=regressed_folds,
    paired_loss_verdict=verdict,
    coverage_sufficient=True,
    interval=True,
  )
  return diagnostic, len(contributing), len(deltas), failures, regressed_folds, status


def minimum_calibration_support_s(speed_mps: float) -> float:
  return minimum_clean_support_s(speed_mps)


def calibration_evidence_sha256(encoded: bytes) -> str:
  if type(encoded) is not bytes:
    raise TypeError("calibration evidence identity requires bytes")
  return hashlib.sha256(encoded).hexdigest()


def _route_sha256(value: str, name: str) -> str:
  if (
    type(value) is not str
    or len(value) != 64
    or any(character not in "0123456789abcdef" for character in value)
  ):
    raise ValueError(f"calibration {name} must be lowercase SHA-256")
  return value


def _canonical_route_counter(value: int) -> int:
  if type(value) is not int or value < 0 or value > 0xFFFFFFFF:
    raise ValueError("calibration route counter must be an unsigned 32-bit integer")
  return value


class CalibrationProfileLearner:
  """Speed-local evidence accumulator and offroad qualification engine."""

  def __init__(self, seed_profile: VehicleCalibrationProfile) -> None:
    if not isinstance(seed_profile, VehicleCalibrationProfile):
      raise TypeError("calibration learner requires VehicleCalibrationProfile")
    if seed_profile.schema_version != CALIBRATION_PROFILE_SCHEMA_VERSION:
      raise ValueError("calibration seed schema is incompatible")
    self.seed_profile = seed_profile
    # Aggregates are derived from immutable route summaries in canonical route
    # order. Keeping no mutable cross-route accumulator eliminates ingestion-
    # order floating-point drift.
    self._nodes = tuple(_Node() for _ in seed_profile.nodes)
    self._breakaway_detector = BreakawayEpisodeDetector()
    self._route_active = False
    self._active_route_counter: int | None = None
    self._active_route_identity_sha256: str | None = None
    self._active_route_content_sha256: str | None = None
    self._routes: list[_RouteEvidence] = []
    # Single-owner mutable counters keep the 100 Hz path allocation-free.
    # Immutable snapshots are materialized only for export/status consumers.
    self._sample_disposition_counts = [0] * len(_SAMPLE_DISPOSITIONS)
    self._active_route_source_accounting: _RouteSourceAccounting | None = None
    self._active_route_nodes: tuple[_Node, ...] | None = None
    self._active_route_intervals: tuple[_IntervalEvidence, ...] | None = None
    self._active_assignment_count = 0
    self._active_assignment_chain = b""
    self._active_last_coordinate: CalibrationIngestionCoordinate | None = None
    self._active_non_authoritative_count = 0
    self._last_assignment_category = ""
    self._last_assignment_episode: BreakawayEpisode | None = None
    self._restore_authoritative = True

  @property
  def speed_nodes_mps(self) -> tuple[float, ...]:
    return self.seed_profile.speed_nodes_mps

  @property
  def sample_accounting(self) -> CalibrationSampleAccounting:
    return CalibrationSampleAccounting(
      self._sample_disposition_counts[
        _SAMPLE_DISPOSITION_INDEX[CalibrationSampleDisposition.ACCEPTED]
      ],
      tuple(
        self._sample_disposition_counts[_SAMPLE_DISPOSITION_INDEX[disposition]]
        for disposition in _REJECTED_SAMPLE_DISPOSITIONS
      ),
    )

  @property
  def route_commitments(self) -> tuple[CalibrationRouteCommitment, ...]:
    """Bounded structural commitments for independent authenticated replay."""
    return tuple(
      CalibrationRouteCommitment(
        route.route_index,
        route.route_counter,
        route.route_identity_sha256,
        route.route_content_sha256,
        route.assignment_record_count,
        route.assignment_chain_sha256,
        route.route_commitment_sha256,
      )
      for route in _canonical_committed_routes(tuple(self._routes))
    )

  @property
  def evidence_authoritative(self) -> bool:
    return self._restore_authoritative and not any(
      route.non_authoritative_record_count for route in self._routes
    )

  def evidence_for_node(self, index: int) -> CalibrationNodeEvidenceSnapshot:
    completed_routes = _canonical_routes(tuple(self._routes))
    route_parts = tuple(route.nodes[index] for route in completed_routes)
    if self._active_route_nodes is not None:
      route_parts += (self._active_route_nodes[index],)
    node = _aggregate_nodes(route_parts)
    route_counts = _independent_route_counts(completed_routes, index)
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
      full_fit_support_s=node.training.weight_s,
      full_fit_count=node.training.count,
      moving_full_fit_support_s=node.moving_training.weight_s,
      moving_full_fit_count=node.moving_training.count,
      breakaway_full_fit_support_s=node.breakaway_training.weight_s,
      breakaway_full_fit_count=node.breakaway_training.count,
      breakaway_episode_training_weight=(
        node.breakaway_episode_training.weight_s
      ),
      breakaway_episode_full_fit_count=(
        node.breakaway_episode_training.count
      ),
      breakaway_episode_dwell_s=node.breakaway_episode_dwell_s,
      breakaway_angle_assisted_count=node.breakaway_angle_assisted_count,
      authority_full_fit_support_s=node.authority_training.weight_s,
      authority_full_fit_count=node.authority_training.count,
      completed_route_count=route_counts.all,
      base_completed_route_count=route_counts.base,
      moving_completed_route_count=route_counts.moving,
      breakaway_episode_completed_route_count=route_counts.breakaway_episode,
      authority_completed_route_count=route_counts.authority,
      lateral_accel_span_mps2=lateral_accel_span,
      applied_torque_span=applied_torque_span,
      lateral_accel_directions=node.lateral_accel_direction_mask.bit_count(),
      applied_torque_directions=node.applied_torque_direction_mask.bit_count(),
      rack_reversals=node.rack_reversals,
    )

  def reset_route_transients(self) -> None:
    self._breakaway_detector.reset()
    for node in self._active_route_nodes or ():
      node.stationary_dwell_s = 0.0
      node.last_direction = 0

  def begin_route(
    self,
    route_identity_sha256: str,
    route_content_sha256: str | None = None,
    *,
    route_counter: int,
  ) -> None:
    """Pin one canonical route counter and independent identity."""
    if self._route_active:
      raise RuntimeError("calibration learner route is already active")
    counter = _canonical_route_counter(route_counter)
    route_identity = _route_sha256(route_identity_sha256, "route identity")
    route_content = _route_sha256(
      route_identity if route_content_sha256 is None else route_content_sha256,
      "route content identity",
    )
    if any(
      route.route_counter == counter
      or route.route_identity_sha256 == route_identity
      or route.route_content_sha256 == route_content
      for route in self._routes
    ):
      raise ValueError("calibration route counter, identity, or content was already ingested")
    self.reset_route_transients()
    self._active_route_counter = counter
    self._active_route_identity_sha256 = route_identity
    self._active_route_content_sha256 = route_content
    self._active_route_source_accounting = _RouteSourceAccounting()
    self._active_route_nodes = tuple(_Node() for _ in self.seed_profile.nodes)
    self._active_route_intervals = tuple(
      _IntervalEvidence() for _ in range(len(self._nodes) - 1)
    )
    self._active_assignment_count = 0
    self._active_assignment_chain = hashlib.sha256(
      b"blatv2-calibration-assignment-chain-v1\0"
      + _canonical({
        "route_content_sha256": route_content,
        "route_counter": counter,
        "route_identity_sha256": route_identity,
      })
    ).digest()
    self._active_last_coordinate = None
    self._active_non_authoritative_count = 0
    self._route_active = True

  def end_route(self) -> None:
    if not self._route_active:
      raise RuntimeError("calibration learner route is not active")
    route_nodes = self._active_route_nodes
    route_intervals = self._active_route_intervals
    route_source_accounting = self._active_route_source_accounting
    if route_nodes is None or route_intervals is None or route_source_accounting is None:
      raise AssertionError("active calibration route lacks route statistics")
    if self._active_route_identity_sha256 is None or self._active_route_content_sha256 is None:
      raise AssertionError("active calibration route lacks immutable identity")
    if self._active_route_counter is None:
      raise AssertionError("active calibration route lacks canonical counter")
    route = _RouteEvidence(
        route_index=len(self._routes),
        route_counter=self._active_route_counter,
        route_identity_sha256=self._active_route_identity_sha256,
        route_content_sha256=self._active_route_content_sha256,
        source_accounting=route_source_accounting,
        nodes=route_nodes,
        intervals=route_intervals,
        assignment_record_count=self._active_assignment_count,
        assignment_chain_sha256=self._active_assignment_chain.hex(),
        route_commitment_sha256="",
        non_authoritative_record_count=self._active_non_authoritative_count,
      )
    route.route_commitment_sha256 = _route_commitment(route)
    self._routes.append(route)
    self.reset_route_transients()
    self._active_route_nodes = None
    self._active_route_source_accounting = None
    self._active_route_intervals = None
    self._active_route_counter = None
    self._active_route_identity_sha256 = None
    self._active_route_content_sha256 = None
    self._active_assignment_count = 0
    self._active_assignment_chain = b""
    self._active_last_coordinate = None
    self._active_non_authoritative_count = 0
    self._route_active = False

  def _add_regression(
    self,
    node_index: int,
    aggregate_field: str,
    predictors: tuple[float, float, float, float],
    target: float,
    weight: float,
  ) -> None:
    route_nodes = self._active_route_nodes
    if route_nodes is None:
      raise AssertionError("calibration route regression lacks active route")
    if "validation" in aggregate_field:
      raise AssertionError("schema-14 learner cannot accumulate validation rows")
    getattr(route_nodes[node_index], aggregate_field).add(
      predictors, target, weight
    )

  def _interval_support(self, speed: float) -> tuple[int, float]:
    nodes = self.speed_nodes_mps
    if speed <= nodes[0]:
      return 0, 0.0
    if speed >= nodes[-1]:
      return len(nodes) - 2, 1.0
    upper = 1
    while nodes[upper] < speed:
      upper += 1
    lower = upper - 1
    return lower, (speed - nodes[lower]) / (nodes[upper] - nodes[lower])

  def _add_joint_regression(
    self,
    speed: float,
    stratum: CalibrationIntervalStratum,
    predictors: tuple[float, float, float, float],
    target: float,
    weight: float,
  ) -> None:
    intervals = self._active_route_intervals
    if intervals is None:
      raise AssertionError("joint calibration regression lacks active route")
    interval, upper_weight = self._interval_support(speed)
    intervals[interval].regression(stratum).add(
      predictors, target, weight, upper_weight
    )

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

  @staticmethod
  def _rate_direction(
    sample: LearningSample,
    seed: CalibrationParameters,
  ) -> int:
    threshold = max(seed.rack_rate_resolution_deg_s, 1e-12)
    return (
      1
      if sample.rack_rate_deg_s >= threshold
      else -1
      if sample.rack_rate_deg_s <= -threshold
      else 0
    )

  @staticmethod
  def _row(sample: LearningSample, category: str, direction: int) -> tuple[float, float, float, float]:
    if category == "moving":
      return (-sample.measured_lateral_accel_mps2, 1.0, float(direction), 0.0)
    if category == "breakaway":
      return (-sample.measured_lateral_accel_mps2, 1.0, 0.0, float(direction))
    return (-sample.measured_lateral_accel_mps2, 1.0, 0.0, 0.0)

  @staticmethod
  def _episode_row(
    episode: BreakawayEpisode,
  ) -> tuple[tuple[float, float, float, float], float]:
    lateral_accel = 0.5 * (
      episode.last_stuck.measured_lateral_accel_mps2
      + episode.first_motion.measured_lateral_accel_mps2
    )
    torque = 0.5 * (
      episode.last_stuck.applied_torque
      + episode.first_motion.applied_torque
    )
    return (
      (-lateral_accel, 1.0, 0.0, float(episode.direction)),
      torque,
    )

  @staticmethod
  def _reset_node(node: _Node) -> None:
    node.stationary_dwell_s = 0.0
    node.last_direction = 0

  def _add_sample_without_accounting(
    self,
    sample: LearningSample,
    source_coordinate: CalibrationIngestionCoordinate | None,
  ) -> CalibrationSampleDisposition:
    if not isinstance(sample, LearningSample):
      raise TypeError("calibration learner accepts LearningSample only")
    if not self._route_active:
      raise RuntimeError("calibration sample has no pinned route partition")
    self._last_assignment_category = "rejected"
    self._last_assignment_episode = None
    if not sample.valid or not sample.engaged or sample.steering_pressed or sample.standstill:
      self.reset_route_transients()
      return CalibrationSampleDisposition.LEARNER_INELIGIBLE
    reversal_evidence = (
      sample.rack_direction_reversal
      and sample._base_valid
      and not sample.actuator_constrained
      and sample.actuator_boundary == ActuatorBoundary.NONE
      and sample.magnitude_boundary_dwell_s == 0.0
    )
    if not sample.clean and not sample.authority_evidence and not reversal_evidence:
      self.reset_route_transients()
      return CalibrationSampleDisposition.LEARNER_INELIGIBLE
    if reversal_evidence:
      # A signed reversal is valid for the acceleration-free inverse map, but
      # it ends prior dwell/direction continuity. It may be a moving row; it
      # must not manufacture a breakaway from stale stationary history.
      self.reset_route_transients()
      route_nodes = self._active_route_nodes
      if route_nodes is None:
        raise AssertionError("active route lacks node summaries")
      for node_index, _ in self._supports(sample.speed_mps):
        route_nodes[node_index].rack_reversals += 1

    route_nodes = self._active_route_nodes
    route_source_accounting = self._active_route_source_accounting
    if route_nodes is None or route_source_accounting is None:
      raise AssertionError("active route lacks node summaries")
    if sample.authority_evidence:
      # A limiter boundary interrupts free breakaway causality. Settled,
      # full-magnitude motion may still identify the moving map.
      self._breakaway_detector.reset()
      seed_at_speed = self.seed_profile.parameters_at(sample.speed_mps).parameters
      joint_direction = self._rate_direction(sample, seed_at_speed)
      equality_fit = (
        sample.actuator_boundary == ActuatorBoundary.MAGNITUDE
        and sample.magnitude_boundary_dwell_s + 1e-12 >= sample.dt_s
        and joint_direction != 0
      )
      if equality_fit:
        self._add_joint_regression(
          sample.speed_mps,
          CalibrationIntervalStratum.AUTHORITY,
          self._row(sample, "moving", joint_direction),
          sample.applied_torque,
          sample.dt_s,
        )
        route_source_accounting.authority_fit += 1
        self._last_assignment_category = "authority_fit"
      else:
        route_source_accounting.authority_unresolved += 1
        self._last_assignment_category = "authority_unresolved"
      accepted = False
      for node_index, node_weight in self._supports(sample.speed_mps):
        node = route_nodes[node_index]
        weight = sample.dt_s * node_weight
        node.authority_support_s += weight
        node.authority_sample_count += 1
        node.authority_magnitude_sample_count += bool(
          sample.actuator_boundary & ActuatorBoundary.MAGNITUDE
        )
        node.authority_slew_build_sample_count += bool(
          sample.actuator_boundary & ActuatorBoundary.SLEW_BUILD
        )
        node.authority_slew_release_sample_count += bool(
          sample.actuator_boundary & ActuatorBoundary.SLEW_RELEASE
        )
        if not equality_fit:
          node.authority_unresolved_sample_count += bool(
            sample.actuator_boundary & ActuatorBoundary.MAGNITUDE
          )
          self._reset_node(node)
          accepted = True
          continue
        predictors = self._row(sample, "moving", joint_direction)
        self._add_regression(
          node_index,
          "authority_training",
          predictors,
          sample.applied_torque,
          weight,
        )
        node.authority_fit_count += 1
        node.authority_fit_sample_count += 1
        node.authority_fit_support_s += weight
        accepted = True
      return (
        CalibrationSampleDisposition.ACCEPTED
        if accepted
        else CalibrationSampleDisposition.LEARNER_INELIGIBLE
      )

    seed_at_speed = self.seed_profile.parameters_at(
      sample.speed_mps,
    ).parameters
    decision = self._breakaway_detector.update(
      sample,
      rack_rate_resolution_deg_s=(
        seed_at_speed.rack_rate_resolution_deg_s
      ),
      transport_delay_s=seed_at_speed.transport_delay_s,
      source_coordinate=source_coordinate,
    )
    category = decision.category.value
    episode = decision.episode
    self._last_assignment_category = category
    self._last_assignment_episode = episode
    if category == "discarded":
      for node in route_nodes:
        self._reset_node(node)
      return CalibrationSampleDisposition.BREAKAWAY_EPISODE_DISCARDED
    if category == "pending":
      route_source_accounting.pending += 1
    elif category == "breakaway":
      route_source_accounting.breakaway_episode += 1
    elif category == "moving":
      route_source_accounting.moving += 1
    else:
      route_source_accounting.base += 1
    direction = decision.direction
    support_speed = (
      episode.onset_speed_mps if episode is not None else sample.speed_mps
    )
    if category != "pending":
      joint_torque = sample.applied_torque
      joint_weight = sample.dt_s
      if category == "breakaway":
        if episode is None:
          raise AssertionError("breakaway category lacks a complete episode")
        joint_predictors, joint_torque = self._episode_row(episode)
        # Match the node-family objective exactly: one physical episode gets
        # unit total weight. Duration and dwell cannot manufacture authority.
        joint_weight = 1.0
      else:
        joint_predictors = self._row(sample, category, direction)
      self._add_joint_regression(
        support_speed,
        (
          CalibrationIntervalStratum.BREAKAWAY_EPISODE
          if category == "breakaway"
          else CalibrationIntervalStratum.MOVING
          if category == "moving"
          else CalibrationIntervalStratum.BASE
        ),
        joint_predictors,
        joint_torque,
        joint_weight,
      )
    accepted = False
    for node_index, node_weight in self._supports(support_speed):
      node = route_nodes[node_index]
      weight = sample.dt_s * node_weight
      node.clean_support_s += weight
      node.supported_sample_count += 1
      node.lat_energy += weight * sample.measured_lateral_accel_mps2**2
      node.rack_travel_deg += weight * abs(sample.rack_rate_deg_s)
      if direction != 0:
        if node.last_direction != 0 and direction != node.last_direction:
          node.rack_reversals += 1
        node.last_direction = direction

      # An angle-only onset is physical motion that still carries static
      # breakout. Retain its support, but do not contaminate either equality
      # model while same-direction rate confirmation is pending.
      if category == "pending":
        node.base_support_s += weight
        node.base_sample_count += 1
      else:
        fit_torque = sample.applied_torque
        fit_weight = weight
        if category == "breakaway":
          if episode is None:
            raise AssertionError("breakaway category lacks a complete episode")
          # Angle-assisted onset is intentionally confirmed one or more
          # frames later.  Keep the legacy breakaway surface tied to the
          # earliest measured motion, never to that delayed rate quantum.
          predictors = (
            -episode.first_motion.measured_lateral_accel_mps2,
            1.0,
            0.0,
            float(direction),
          )
          fit_torque = episode.first_motion.applied_torque
          fit_weight = episode.first_motion.dt_s * node_weight
        else:
          predictors = self._row(sample, category, direction)
        self._add_regression(
          node_index,
          "training",
          predictors,
          fit_torque,
          fit_weight,
        )
        node.fit_count += 1
        if category == "base":
          node.base_support_s += weight
          node.base_sample_count += 1
        elif category == "moving":
          node.moving_support_s += weight
          node.moving_sample_count += 1
          node.moving_direction_mask |= 1 if direction < 0 else 2
          node.moving_training_direction_mask |= (
            1 if direction < 0 else 2
          )
          self._add_regression(
            node_index,
            "moving_training",
            predictors,
            sample.applied_torque,
            weight,
          )
        else:
          node.breakaway_support_s += weight
          node.breakaway_sample_count += 1
          node.breakaway_direction_mask |= 1 if direction < 0 else 2
          self._add_regression(
            node_index,
            "breakaway_training",
            predictors,
            fit_torque,
            fit_weight,
          )

          episode_predictors, episode_torque = self._episode_row(episode)
          # One physical episode gets unit total weight, split only by its
          # onset-speed interpolation. Dwell length cannot outvote independent
          # episodes.
          self._add_regression(
            node_index,
            "breakaway_episode_training",
            episode_predictors,
            episode_torque,
            node_weight,
          )
          node.breakaway_episode_count += 1
          node.breakaway_episode_dwell_s += (
            episode.dwell_s * node_weight
          )
          node.breakaway_angle_assisted_count += int(
            episode.angle_assisted
          )
          node.breakaway_bracket_width_sum += node_weight * abs(
            episode.first_motion.applied_torque
            - episode.last_stuck.applied_torque
          )
          node.breakaway_episode_direction_mask |= (
            1 if direction < 0 else 2
          )
          node.breakaway_episode_training_direction_mask |= (
            1 if direction < 0 else 2
          )
      if node_weight >= MIN_EXCITATION_NODE_WEIGHT:
        lat = sample.measured_lateral_accel_mps2
        torque = sample.applied_torque
        node.lat_min, node.lat_max = min(node.lat_min, lat), max(node.lat_max, lat)
        node.torque_min, node.torque_max = min(node.torque_min, torque), max(node.torque_max, torque)
        node.lateral_accel_direction_mask |= 1 if lat < 0.0 else 2 if lat > 0.0 else 0
        node.applied_torque_direction_mask |= 1 if torque < 0.0 else 2 if torque > 0.0 else 0
      accepted = True
    return (
      CalibrationSampleDisposition.ACCEPTED
      if accepted
      else CalibrationSampleDisposition.LEARNER_INELIGIBLE
    )

  def add_sample_with_disposition(
    self,
    sample: LearningSample,
    *,
    source_coordinate: CalibrationIngestionCoordinate | None = None,
    upstream_rejection: CalibrationSampleDisposition | None = None,
  ) -> CalibrationSampleDisposition:
    """Accumulate one frame and durably record exactly one first cause."""
    if upstream_rejection is not None and (
      not isinstance(upstream_rejection, CalibrationSampleDisposition)
      or upstream_rejection is CalibrationSampleDisposition.ACCEPTED
      or upstream_rejection in _LEARNER_OWNED_SAMPLE_DISPOSITIONS
    ):
      raise ValueError("upstream rejection must identify an upstream pipeline failure")
    if not isinstance(sample, LearningSample):
      raise TypeError("calibration learner accepts LearningSample only")
    if not self._route_active:
      raise RuntimeError("calibration sample has no pinned route partition")
    self._last_assignment_category = "rejected"
    self._last_assignment_episode = None
    if upstream_rejection is not None:
      # Runtime first-cause classification is authoritative. An upstream
      # failure resets physical continuity and never reaches accumulation,
      # even if a contradictory test caller attaches it to a clean sample.
      self.reset_route_transients()
      self._sample_disposition_counts[
        _SAMPLE_DISPOSITION_INDEX[upstream_rejection]
      ] += 1
      self._commit_assignment(sample, upstream_rejection, source_coordinate)
      return upstream_rejection
    learner_disposition = self._add_sample_without_accounting(
      sample,
      source_coordinate,
    )
    self._sample_disposition_counts[
      _SAMPLE_DISPOSITION_INDEX[learner_disposition]
    ] += 1
    disposition = learner_disposition
    self._commit_assignment(sample, disposition, source_coordinate)
    return disposition

  def _commit_assignment(
    self,
    sample: LearningSample,
    disposition: CalibrationSampleDisposition,
    source_coordinate: CalibrationIngestionCoordinate | None,
  ) -> None:
    if source_coordinate is None:
      self._active_non_authoritative_count += 1
      return
    if source_coordinate.route_content_sha256 != self._active_route_content_sha256:
      raise ValueError("calibration source coordinate belongs to another route")
    if (
      self._active_last_coordinate is not None
      and source_coordinate.ordering_key
      <= self._active_last_coordinate.ordering_key
    ):
      raise ValueError("calibration source coordinates are duplicate or unordered")
    self._active_last_coordinate = source_coordinate
    accepted = disposition is CalibrationSampleDisposition.ACCEPTED
    category = self._last_assignment_category if accepted else disposition.value
    episode = self._last_assignment_episode if accepted else None
    support_speed = episode.onset_speed_mps if episode is not None else sample.speed_mps
    node_supports = self._supports(support_speed) if accepted else ()
    contributes_interval = category in (
      "authority_fit",
      "base",
      "breakaway",
      "moving",
    )
    interval_support = (
      self._interval_support(support_speed)
      if accepted and contributes_interval
      else None
    )

    def node_contribution(index: int, node_weight: float) -> dict[str, object]:
      support_weight = sample.dt_s * node_weight
      training_weight: float | None = support_weight
      episode_weight: float | None = None
      if category in ("authority_unresolved", "pending"):
        training_weight = None
      elif category == "breakaway":
        if episode is None:
          raise AssertionError("breakaway category lacks a complete episode")
        training_weight = episode.first_motion.dt_s * node_weight
        episode_weight = node_weight
      return {
        "episode_training_weight": (
          None if episode_weight is None else episode_weight.hex()
        ),
        "node_index": index,
        "node_weight": node_weight.hex(),
        "support_weight": support_weight.hex(),
        "training_weight": (
          None if training_weight is None else training_weight.hex()
        ),
      }

    payload: dict[str, object] = {
      "actuator_boundary": int(sample.actuator_boundary),
      "actuator_constrained": bool(sample.actuator_constrained),
      "category": category,
      "coordinate": source_coordinate.payload(),
      "disposition": disposition.value,
      "interval_contribution": (
        {
          "interval_index": interval_support[0],
          "upper_weight": interval_support[1].hex(),
          "weight": (
            (1.0 if category == "breakaway" else sample.dt_s).hex()
          ),
        }
        if interval_support is not None
        else None
      ),
      "magnitude_boundary_dwell_s": sample.magnitude_boundary_dwell_s.hex(),
      "node_contributions": [
        node_contribution(index, weight)
        for index, weight in node_supports
      ],
      "physical": {
        "applied_torque": sample.applied_torque.hex(),
        "dt_s": sample.dt_s.hex(),
        "measured_lateral_accel_mps2": sample.measured_lateral_accel_mps2.hex(),
        "measured_rack_angle_deg": sample.measured_rack_angle_deg.hex(),
        "rack_acceleration_deg_s2": sample.rack_acceleration_deg_s2.hex(),
        "rack_rate_deg_s": sample.rack_rate_deg_s.hex(),
        "speed_mps": sample.speed_mps.hex(),
      },
    }
    if episode is not None:
      coordinates = (
        episode.last_stuck.source_coordinate,
        episode.first_motion.source_coordinate,
        episode.rate_confirmation.source_coordinate,
      )
      if any(coordinate is None for coordinate in coordinates):
        raise ValueError("authoritative breakaway lacks endpoint coordinates")
      payload["breakaway_coordinates"] = [
        coordinate.payload()
        for coordinate in coordinates
        if coordinate is not None
      ]
    else:
      payload["breakaway_coordinates"] = None
    self._active_assignment_chain = hashlib.sha256(
      b"blatv2-calibration-assignment-record-v1\0"
      + self._active_assignment_chain
      + _canonical(payload)
    ).digest()
    self._active_assignment_count += 1

  def add_sample(self, sample: LearningSample) -> bool:
    """Compatibility wrapper for callers that need only accepted/rejected."""
    return (
      self.add_sample_with_disposition(sample)
      is CalibrationSampleDisposition.ACCEPTED
    )

  def _node_report(self, index: int) -> CalibrationNodeQualificationReport:
    routes = _canonical_routes(tuple(self._routes))
    node = _aggregate_nodes(tuple(route.nodes[index] for route in routes))
    seed = self.seed_profile.nodes[index].parameters
    seed_coefficients = _seed_coefficients(seed)
    speed = self.speed_nodes_mps[index]
    minimum = minimum_calibration_support_s(speed)
    lat_span = 0.0 if not math.isfinite(node.lat_min) else node.lat_max - node.lat_min
    torque_span = 0.0 if not math.isfinite(node.torque_min) else node.torque_max - node.torque_min
    lat_rms = math.sqrt(node.lat_energy / node.clean_support_s) if node.clean_support_s > 0.0 else 0.0

    route_counts = _independent_route_counts(routes, index)
    reasons: list[CalibrationQualificationReason] = []
    unresolved: list[CalibrationQualificationReason] = []
    insufficient_support = node.clean_support_s < minimum
    if insufficient_support:
      reasons.append(CalibrationQualificationReason.INSUFFICIENT_SUPPORT)
    insufficient_excitation = (
      lat_span < MIN_LATERAL_ACCEL_SPAN_MPS2
      or lat_rms < MIN_LATERAL_ACCEL_RMS_MPS2
      or torque_span < MIN_APPLIED_TORQUE_SPAN
      or node.lateral_accel_direction_mask != 3
      or node.applied_torque_direction_mask != 3
    )
    if insufficient_excitation:
      reasons.append(CalibrationQualificationReason.INSUFFICIENT_EXCITATION)
    insufficient_moving = node.moving_training.count < MIN_STRATUM_TRAINING_ROWS
    if insufficient_moving:
      reasons.append(CalibrationQualificationReason.INSUFFICIENT_MOVING_EVIDENCE)
    insufficient_breakaway = (
      node.breakaway_training.count < MIN_STRATUM_TRAINING_ROWS
      or node.breakaway_episode_training.count < MIN_STRATUM_TRAINING_ROWS
    )
    if insufficient_breakaway:
      reasons.append(CalibrationQualificationReason.INSUFFICIENT_BREAKAWAY_EVIDENCE)

    required_counts = (
      route_counts.all,
      route_counts.base,
      route_counts.moving,
      route_counts.breakaway,
      route_counts.breakaway_episode,
    )
    insufficient_required_routes = any(
      count < MIN_INDEPENDENT_ROUTES for count in required_counts
    )
    if insufficient_required_routes:
      reasons.append(CalibrationQualificationReason.INSUFFICIENT_INDEPENDENT_ROUTES)
      unresolved.append(CalibrationQualificationReason.INSUFFICIENT_INDEPENDENT_ROUTES)

    fields = _family_population_fields(route_counts.authority > 0)
    cross_fit_families = tuple(
      _cross_fit_family(routes, index, model, seed_coefficients, fields)
      for model, _ in _MODEL_FAMILIES
    )
    selected_cross_fit: _CrossFitFamily | None = None
    for family in cross_fit_families:
      if family.diagnostic.status is not CalibrationCrossFitStatus.SCORED:
        continue
      if selected_cross_fit is None or _cross_fit_dominates(
        routes, index, family, selected_cross_fit
      ):
        selected_cross_fit = family
    safe_seed_cross_fit = tuple(
      family for family in cross_fit_families
      if family.diagnostic.status is CalibrationCrossFitStatus.NO_ROBUST_IMPROVEMENT
    )
    cross_fit_fold_failure = (
      selected_cross_fit is None
      and not safe_seed_cross_fit
      and any(family.diagnostic.failed_fold_count > 0 for family in cross_fit_families)
    )
    if cross_fit_fold_failure:
      reasons.append(CalibrationQualificationReason.CROSS_FIT_FOLD_FAILURE)
      unresolved.append(CalibrationQualificationReason.CROSS_FIT_FOLD_FAILURE)
    heldout_regression = (
      selected_cross_fit is None
      and not safe_seed_cross_fit
      and any(
        family.diagnostic.regressed_fold_count > 0
        for family in cross_fit_families
      )
    )

    include_authority = route_counts.authority > 0
    full_fit_results = tuple(
      (
        model,
        *_fit_model_family(
          routes,
          index,
          model,
          seed_coefficients,
          include_authority=include_authority,
        ),
      )
      for model, _ in _MODEL_FAMILIES
    )
    fit_diagnostics = tuple(result[2] for result in full_fit_results)
    if not any(
      diagnostic.status is CalibrationFitStatus.IDENTIFIABLE
      for diagnostic in fit_diagnostics
    ):
      statuses = {diagnostic.status for diagnostic in fit_diagnostics}
      if CalibrationFitStatus.ILL_CONDITIONED in statuses:
        reasons.append(CalibrationQualificationReason.ILL_CONDITIONED_FIT)
      elif CalibrationFitStatus.RANK_DEFICIENT in statuses:
        reasons.append(CalibrationQualificationReason.RANK_DEFICIENT_FIT)
      else:
        reasons.append(CalibrationQualificationReason.SINGULAR_FIT)
    selected_model = None if selected_cross_fit is None else selected_cross_fit.model
    terminal_seed_fallback = next(
      (
        family for family in cross_fit_families
        if (
          selected_cross_fit is None
          and not safe_seed_cross_fit
          and route_counts.authority == MIN_TERMINAL_SEED_AUTHORITY_ROUTES
          and terminal_seed_authority_allowed(
            vehicle_identity=self.seed_profile.vehicle_identity,
            speed_mps=speed,
            terminal=index == len(self._nodes) - 1,
            seed_retained=True,
          )
          and family.diagnostic.status
          is CalibrationCrossFitStatus.HELD_OUT_REGRESSION
          and family.diagnostic.successful_fold_count > 0
          and family.diagnostic.failed_fold_count == 0
          and any(
            model is family.model
            and fitted == seed_coefficients
            and diagnostic.status is CalibrationFitStatus.IDENTIFIABLE
            for model, fitted, diagnostic in full_fit_results
          )
        )
      ),
      None,
    )
    selection_cross_fit = (
      selected_cross_fit
      if selected_cross_fit is not None
      else safe_seed_cross_fit[0]
      if safe_seed_cross_fit
      else terminal_seed_fallback
    )
    selected_full_fit = next(
      (
        result for result in full_fit_results
        if selection_cross_fit is not None and result[0] is selection_cross_fit.model
      ),
      None,
    )
    seed_retained = selected_cross_fit is None and selection_cross_fit is not None
    terminal_seed_authority = terminal_seed_authority_allowed(
      vehicle_identity=self.seed_profile.vehicle_identity,
      speed_mps=speed,
      terminal=index == len(self._nodes) - 1,
      seed_retained=seed_retained,
    )
    minimum_authority_routes = (
      MIN_TERMINAL_SEED_AUTHORITY_ROUTES
      if terminal_seed_authority
      else MIN_INDEPENDENT_ROUTES
    )
    insufficient_authority_routes = (
      0 < route_counts.authority < minimum_authority_routes
    )
    if insufficient_authority_routes:
      reasons.append(CalibrationQualificationReason.INSUFFICIENT_INDEPENDENT_ROUTES)
      unresolved.append(CalibrationQualificationReason.INSUFFICIENT_INDEPENDENT_ROUTES)
    insufficient_independent = (
      insufficient_required_routes or insufficient_authority_routes
    )
    coefficients = seed_coefficients
    full_fit_diagnostic: CalibrationModelFitDiagnostic | None = None
    full_fit_safe = seed_retained or not heldout_regression
    full_fit_stratum_losses: tuple[CalibrationPairedLossDiagnostic, ...] = ()
    if selected_full_fit is not None:
      _, fitted, full_fit_diagnostic = selected_full_fit
      if fitted is None or full_fit_diagnostic.status is not CalibrationFitStatus.IDENTIFIABLE:
        reasons.append(CalibrationQualificationReason.CROSS_FIT_FOLD_FAILURE)
        cross_fit_fold_failure = True
      elif not seed_retained:
        coefficients = fitted
        full_fit_safe, full_fit_stratum_losses = _safe_on_route_strata(
          routes,
          index,
          fields,
          coefficients,
          seed_coefficients,
          require_improvement=False,
        )
        if not full_fit_safe:
          reasons.append(CalibrationQualificationReason.CROSS_FIT_REGRESSION)

    gain, intercept, kinetic, static = coefficients
    offset = intercept / gain if gain != 0.0 else math.inf
    offset_bound = max(1.0, lat_span)
    parameters_valid = not (
      not all(math.isfinite(value) for value in (*coefficients, offset))
      or gain <= 0.0
      or static < kinetic
      or kinetic < 0.0
      or abs(offset) > offset_bound
    )
    if not parameters_valid:
      reasons.append(CalibrationQualificationReason.INVALID_PARAMETERS)

    all_evidence = node.training
    base_evidence = _subtract(node.training, node.moving_training, node.breakaway_training)
    candidate_parameters = CalibrationParameters(
      torque_per_lateral_accel=seed.torque_per_lateral_accel if seed_retained else gain,
      lateral_accel_offset_correction_mps2=(
        seed.lateral_accel_offset_correction_mps2 if seed_retained else offset
      ),
      kinetic_friction_torque=seed.kinetic_friction_torque if seed_retained else kinetic,
      static_breakaway_torque=seed.static_breakaway_torque if seed_retained else static,
      transport_delay_s=seed.transport_delay_s,
      rack_rate_resolution_deg_s=seed.rack_rate_resolution_deg_s,
      confidence=min(
        1.0,
        max(
          0.0,
          min(
            node.clean_support_s / minimum,
            route_counts.all / MIN_INDEPENDENT_ROUTES,
            lat_span / MIN_LATERAL_ACCEL_SPAN_MPS2,
            lat_rms / MIN_LATERAL_ACCEL_RMS_MPS2,
            torque_span / MIN_APPLIED_TORQUE_SPAN,
          ),
        ),
      ),
      qualified=False,
    )
    unique = calibration_node_failure_reasons(
      insufficient_support=insufficient_support,
      insufficient_excitation=insufficient_excitation,
      insufficient_moving_evidence=insufficient_moving,
      insufficient_breakaway_evidence=insufficient_breakaway,
      insufficient_independent_routes=insufficient_independent,
      cross_fit_fold_failure=cross_fit_fold_failure,
      fit_statuses=tuple(diagnostic.status for diagnostic in fit_diagnostics),
      full_fit_safe=full_fit_safe,
      parameters_valid=parameters_valid,
    )
    qualified = not unique
    outcome = (
      CalibrationQualificationReason.SEED_RETAINED
      if seed_retained
      else CalibrationQualificationReason.LEARNED
    )
    if qualified:
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
    report_reasons = (outcome,) if qualified else unique
    full_loss = _paired_route_losses(
      _route_regressions_for_node(routes, index, "training"),
      coefficients,
      seed_coefficients,
    )
    cross_fit_loss = (
      selection_cross_fit.diagnostic.paired_loss
      if selection_cross_fit is not None
      else _diagnostic_from_deltas(())
    )
    return CalibrationNodeQualificationReport(
      node_index=index,
      speed_mps=speed,
      minimum_support_s=minimum,
      clean_support_s=node.clean_support_s,
      supported_sample_count=node.supported_sample_count,
      full_fit_count=node.training.count,
      cross_fit_route_count=(
        selection_cross_fit.diagnostic.contributing_route_count
        if selection_cross_fit is not None
        else 0
      ),
      base_support_s=node.base_support_s,
      base_sample_count=node.base_sample_count,
      moving_support_s=node.moving_support_s,
      moving_sample_count=node.moving_sample_count,
      moving_full_fit_count=node.moving_training.count,
      moving_cross_fit_route_count=route_counts.moving,
      breakaway_support_s=node.breakaway_support_s,
      breakaway_sample_count=node.breakaway_sample_count,
      breakaway_full_fit_count=node.breakaway_training.count,
      breakaway_cross_fit_route_count=route_counts.breakaway,
      lateral_accel_span_mps2=lat_span,
      lateral_accel_rms_mps2=lat_rms,
      rack_travel_deg=node.rack_travel_deg,
      applied_torque_span=torque_span,
      rack_reversals=node.rack_reversals,
      lateral_accel_directions=node.lateral_accel_direction_mask.bit_count(),
      applied_torque_directions=node.applied_torque_direction_mask.bit_count(),
      full_fit_seed_rms=all_evidence.rms(seed_coefficients),
      full_fit_candidate_rms=all_evidence.rms(coefficients),
      moving_full_fit_seed_rms=node.moving_training.rms(seed_coefficients),
      moving_full_fit_candidate_rms=node.moving_training.rms(coefficients),
      breakaway_full_fit_seed_rms=node.breakaway_training.rms(seed_coefficients),
      breakaway_full_fit_candidate_rms=node.breakaway_training.rms(coefficients),
      confidence=candidate_parameters.confidence,
      reasons=report_reasons,
      candidate_parameters=candidate_parameters,
      authority_support_s=node.authority_support_s,
      authority_sample_count=node.authority_sample_count,
      authority_fit_support_s=node.authority_fit_support_s,
      authority_fit_sample_count=node.authority_fit_sample_count,
      authority_full_fit_count=node.authority_training.count,
      authority_cross_fit_route_count=route_counts.authority,
      authority_full_fit_seed_rms=node.authority_training.rms(seed_coefficients),
      authority_full_fit_candidate_rms=node.authority_training.rms(coefficients),
      authority_magnitude_sample_count=node.authority_magnitude_sample_count,
      authority_slew_build_sample_count=node.authority_slew_build_sample_count,
      authority_slew_release_sample_count=node.authority_slew_release_sample_count,
      authority_unresolved_sample_count=node.authority_unresolved_sample_count,
      selected_model=selected_model,
      base_full_fit_seed_rms=base_evidence.rms(seed_coefficients),
      base_full_fit_candidate_rms=base_evidence.rms(coefficients),
      breakaway_episode_full_fit_count=node.breakaway_episode_training.count,
      breakaway_episode_cross_fit_route_count=route_counts.breakaway_episode,
      breakaway_episode_dwell_s=node.breakaway_episode_dwell_s,
      breakaway_angle_assisted_count=node.breakaway_angle_assisted_count,
      breakaway_episode_full_fit_seed_rms=node.breakaway_episode_training.rms(seed_coefficients),
      breakaway_episode_full_fit_candidate_rms=node.breakaway_episode_training.rms(coefficients),
      breakaway_mean_bracket_width=(
        None if node.breakaway_episode_training.weight_s <= 0.0
        else node.breakaway_bracket_width_sum / node.breakaway_episode_training.weight_s
      ),
      fit_diagnostics=fit_diagnostics,
      full_fit_paired_loss=full_loss,
      cross_fit_paired_loss=cross_fit_loss,
      selection_outcome=outcome,
      independent_route_counts=route_counts,
      cross_fit_diagnostics=tuple(family.diagnostic for family in cross_fit_families),
      full_fit_diagnostic=full_fit_diagnostic,
      unresolved_diagnostics=tuple(dict.fromkeys(unresolved)),
      full_fit_stratum_paired_losses=full_fit_stratum_losses,
    )
  def qualify(self, provenance: str) -> CalibrationLearningResult:
    if self._route_active:
      raise RuntimeError("active route evidence cannot be qualified")
    source = str(provenance).strip()
    if not source:
      raise ValueError("candidate provenance must not be empty")
    reports = tuple(self._node_report(index) for index in range(len(self._nodes)))
    if not all(report.qualified for report in reports):
      return CalibrationLearningResult(reports, None)
    candidate_parameters = tuple(
      report.candidate_parameters  # type: ignore[misc]
      for report in reports
    )
    if any(parameters is None for parameters in candidate_parameters):
      raise AssertionError("qualified node lacks calibration parameters")
    candidate_parameters = tuple(
      parameters for parameters in candidate_parameters if parameters is not None
    )
    seed_parameters = tuple(node.parameters for node in self.seed_profile.nodes)
    # Confidence/support are evidence metadata and legitimately advance even
    # when the four physical values remain the exact seed. Artifact emission
    # is keyed only to values that change controller behavior.
    profile_identical = tuple(
      _seed_coefficients(parameters) for parameters in candidate_parameters
    ) == tuple(
      _seed_coefficients(parameters) for parameters in seed_parameters
    )
    routes = _canonical_routes(tuple(self._routes))
    interpolation_reports: list[CalibrationInterpolationQualificationReport] = []
    for interval_index in range(len(self._nodes) - 1):
      interval_reasons: list[CalibrationQualificationReason] = []
      stratum_diagnostics: list[CalibrationIntervalStratumDiagnostic] = []
      contributors: set[str] = set()
      for stratum in _INTERVAL_STRATA:
        terminal_seed_authority = terminal_seed_authority_allowed(
          vehicle_identity=self.seed_profile.vehicle_identity,
          speed_mps=self.speed_nodes_mps[interval_index + 1],
          terminal=(
            interval_index == len(self._nodes) - 2
            and stratum is CalibrationIntervalStratum.AUTHORITY
          ),
          seed_retained=(
            reports[interval_index].seed_retained
            and reports[interval_index + 1].seed_retained
          ),
        )
        minimum_independent_routes = (
          MIN_TERMINAL_SEED_AUTHORITY_ROUTES
          if terminal_seed_authority
          else MIN_INDEPENDENT_ROUTES
        )
        stratum_routes = tuple(
          route for route in routes
          if route.intervals[interval_index].regression(stratum).count > 0
        )
        if not stratum_routes:
          continue
        contributors.update(route.route_identity_sha256 for route in stratum_routes)
        full_fit = _paired_joint_route_losses(
          routes,
          interval_index,
          stratum,
          candidate_parameters[interval_index],
          candidate_parameters[interval_index + 1],
          seed_parameters[interval_index],
          seed_parameters[interval_index + 1],
        )
        (
          cross_fit,
          contributing_route_count,
          successful_fold_count,
          failed_fold_count,
          regressed_fold_count,
          cross_fit_status,
        ) = _cross_fit_interval_loss(
          routes,
          interval_index,
          stratum,
          reports[interval_index].selected_model,
          reports[interval_index + 1].selected_model,
          seed_parameters[interval_index],
          seed_parameters[interval_index + 1],
          allow_single_exact_seed=terminal_seed_authority,
        )
        stratum_diagnostics.append(CalibrationIntervalStratumDiagnostic(
          stratum=stratum,
          full_fit_paired_loss=full_fit,
          cross_fit_paired_loss=cross_fit,
          contributing_route_count=contributing_route_count,
          successful_fold_count=successful_fold_count,
          failed_fold_count=failed_fold_count,
          regressed_fold_count=regressed_fold_count,
          cross_fit_status=cross_fit_status,
        ))
        if contributing_route_count < minimum_independent_routes:
          interval_reasons.append(
            CalibrationQualificationReason.INSUFFICIENT_INDEPENDENT_ROUTES
          )
        if failed_fold_count:
          interval_reasons.append(CalibrationQualificationReason.CROSS_FIT_FOLD_FAILURE)
        full_fit_verdict = _paired_loss_verdict(
          full_fit,
          identical=terminal_seed_authority,
        )
        cross_fit_verdict = _paired_loss_verdict(
          cross_fit,
          identical=terminal_seed_authority,
        )
        if full_fit_verdict is _PairedLossVerdict.REGRESSION:
          interval_reasons.append(
            CalibrationQualificationReason.INTERPOLATION_TRAINING_REGRESSION
          )
        elif full_fit_verdict in (_PairedLossVerdict.INCONCLUSIVE, _PairedLossVerdict.NO_DATA):
          interval_reasons.append(
            CalibrationQualificationReason.INTERPOLATION_TRAINING_INCONCLUSIVE
          )
        if regressed_fold_count or cross_fit_verdict is _PairedLossVerdict.REGRESSION:
          interval_reasons.append(
            CalibrationQualificationReason.INTERPOLATION_CROSS_FIT_REGRESSION
          )
        elif cross_fit_verdict in (_PairedLossVerdict.INCONCLUSIVE, _PairedLossVerdict.NO_DATA):
          interval_reasons.append(
            CalibrationQualificationReason.INTERPOLATION_CROSS_FIT_INCONCLUSIVE
          )
      if not stratum_diagnostics:
        interval_reasons.append(
          CalibrationQualificationReason.INTERPOLATION_TRAINING_INCONCLUSIVE
        )
      interval_statuses = tuple(
        diagnostic.cross_fit_status for diagnostic in stratum_diagnostics
      )
      cross_fit_status = next(
        (
          status for status in (
            CalibrationCrossFitStatus.FOLD_FIT_FAILURE,
            CalibrationCrossFitStatus.HELD_OUT_REGRESSION,
            CalibrationCrossFitStatus.NO_ROBUST_IMPROVEMENT,
            CalibrationCrossFitStatus.INSUFFICIENT_INDEPENDENT_ROUTES,
          )
          if status in interval_statuses
        ),
        CalibrationCrossFitStatus.SCORED,
      )
      interpolation_reports.append(
        CalibrationInterpolationQualificationReport(
          interval_index=interval_index,
          lower_speed_mps=self.speed_nodes_mps[interval_index],
          upper_speed_mps=self.speed_nodes_mps[interval_index + 1],
          stratum_diagnostics=tuple(stratum_diagnostics),
          reasons=(
            (CalibrationQualificationReason.QUALIFIED,)
            if not interval_reasons
            else tuple(dict.fromkeys(interval_reasons))
          ),
          contributing_route_count=len(contributors),
          successful_fold_count=sum(
            diagnostic.successful_fold_count for diagnostic in stratum_diagnostics
          ),
          failed_fold_count=sum(
            diagnostic.failed_fold_count for diagnostic in stratum_diagnostics
          ),
          regressed_fold_count=sum(
            diagnostic.regressed_fold_count for diagnostic in stratum_diagnostics
          ),
          cross_fit_status=cross_fit_status,
        )
      )
    interpolation_tuple = tuple(interpolation_reports)
    if not all(report.qualified for report in interpolation_tuple):
      return CalibrationLearningResult(reports, None, interpolation_tuple)
    revision = self.seed_profile.revision + 1 + sum(
      report.supported_sample_count + report.authority_sample_count
      for report in reports
    )
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
        cross_fit_route_count=report.cross_fit_route_count,
        full_fit_candidate_rms=report.full_fit_candidate_rms or 0.0,
        breakaway_full_fit_candidate_rms=report.breakaway_full_fit_candidate_rms,
      )
      for report in reports
    )
    model_ids = ",".join(
      report.selected_model.value
      for report in reports
      if report.selected_model is not None
    )
    candidate_provenance = (
      f"{source}; observable-inverse-torque-crossfit-v6; " +
      f"terminal_authority_policy={TERMINAL_SEED_AUTHORITY_POLICY_ID}; " +
      f"models={model_ids}; evidence_revision={revision}"
    )
    profile = VehicleCalibrationProfile(
      self.seed_profile.vehicle_identity,
      revision,
      candidate_provenance,
      nodes,
    )
    return CalibrationLearningResult(
      reports,
      None if profile_identical else profile,
      interpolation_tuple,
      selected_profile=profile,
    )

  def export_evidence(self) -> bytes:
    """Serialize structurally validated evidence without claiming authority."""
    if self._route_active:
      raise RuntimeError("active route evidence cannot be exported")
    seed_json = self.seed_profile.to_json()
    canonical_routes = _canonical_committed_routes(tuple(self._routes))
    aggregate_nodes = tuple(
      _aggregate_nodes(tuple(route.nodes[index] for route in canonical_routes))
      for index in range(len(self.seed_profile.nodes))
    )
    nodes = [
      _encode_node_summary(node, index)
      for index, node in enumerate(aggregate_nodes)
    ]
    routes = [
      {
        "route_index": route_index,
        "route_counter": route.route_counter,
        "route_identity_sha256": route.route_identity_sha256,
        "route_content_sha256": route.route_content_sha256,
        "source_accounting": route.source_accounting.encoded(),
        "nodes": [
          _encode_node_summary(route_node, node_index)
          for node_index, route_node in enumerate(route.nodes)
        ],
        "intervals": [interval.encoded() for interval in route.intervals],
        "assignment_record_count": route.assignment_record_count,
        "assignment_chain_sha256": route.assignment_chain_sha256,
        "route_commitment_sha256": route.route_commitment_sha256,
        "non_authoritative_record_count": route.non_authoritative_record_count,
      }
      for route_index, route in enumerate(canonical_routes)
    ]
    payload = {
      "evidence_schema_version": CALIBRATION_EVIDENCE_SCHEMA_VERSION,
      "profile_schema_version": self.seed_profile.schema_version,
      "vehicle_identity": self.seed_profile.vehicle_identity,
      "seed_profile_json": seed_json,
      "seed_profile_sha256": hashlib.sha256(seed_json.encode()).hexdigest(),
      "sample_accounting": self.sample_accounting.to_payload(),
      "speed_nodes_mps": [speed.hex() for speed in self.speed_nodes_mps],
      "nodes": nodes,
      "routes": routes,
    }
    envelope = {"payload": payload, "payload_sha256": hashlib.sha256(_canonical(payload)).hexdigest()}
    return _canonical(envelope)

  def export_authoritative_evidence(self) -> bytes:
    """Export only evidence backed by authenticated historical coordinates."""
    if not self.evidence_authoritative:
      raise RuntimeError(
        "non-authoritative live calibration rows cannot publish offline evidence"
      )
    return self.export_evidence()

  @classmethod
  def from_evidence(
    cls,
    seed_profile: VehicleCalibrationProfile,
    encoded: bytes,
    *,
    expected_route_commitments: tuple[tuple[str, str], ...] | None = None,
  ) -> CalibrationProfileLearner:
    """Restore evidence; only external expected commitments confer authority."""
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
      {
        "evidence_schema_version",
        "profile_schema_version",
        "vehicle_identity",
        "seed_profile_json",
        "seed_profile_sha256",
        "sample_accounting",
        "speed_nodes_mps",
        "nodes",
        "routes",
      },
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
    sample_accounting = CalibrationSampleAccounting.from_payload(
      payload["sample_accounting"],
    )
    learner._sample_disposition_counts = [
      sample_accounting.count(disposition)
      for disposition in _SAMPLE_DISPOSITIONS
    ]
    raw_nodes = payload["nodes"]
    if type(raw_nodes) is not list or len(raw_nodes) != len(learner._nodes):
      raise ValueError("calibration node count mismatch")
    aggregate_nodes = tuple(
      _decode_node_summary(raw, index, f"nodes[{index}]")
      for index, raw in enumerate(raw_nodes)
    )
    raw_routes = payload["routes"]
    if type(raw_routes) is not list:
      raise ValueError("calibration route statistics must be a list")
    raw_route_keys = tuple(
      (
        raw.get("route_identity_sha256"),
        raw.get("route_content_sha256"),
        raw.get("route_counter"),
      )
      if type(raw) is dict
      else (None, None, None)
      for raw in raw_routes
    )
    if raw_route_keys != tuple(sorted(raw_route_keys)):
      raise ValueError("calibration routes are not in canonical identity order")
    for route_index, raw_route in enumerate(raw_routes):
      route_payload = _exact(
        raw_route,
        {
          "route_index",
          "route_counter",
          "route_identity_sha256",
          "route_content_sha256",
          "source_accounting",
          "nodes",
          "intervals",
          "assignment_record_count",
          "assignment_chain_sha256",
          "route_commitment_sha256",
          "non_authoritative_record_count",
        },
        f"routes[{route_index}]",
      )
      if route_payload["route_index"] != route_index:
        raise ValueError("calibration route ordering is corrupt")
      route_counter = _canonical_route_counter(route_payload["route_counter"])
      route_identity = _route_sha256(
        route_payload["route_identity_sha256"],
        "route identity",
      )
      route_content = _route_sha256(
        route_payload["route_content_sha256"],
        "route content identity",
      )
      source_accounting = _RouteSourceAccounting.decoded(
        route_payload["source_accounting"],
        f"routes[{route_index}].source_accounting",
      )
      assignment_record_count = _bounded_count(
        route_payload["assignment_record_count"],
        f"routes[{route_index}].assignment_record_count",
      )
      non_authoritative_count = _bounded_count(
        route_payload["non_authoritative_record_count"],
        f"routes[{route_index}].non_authoritative_record_count",
      )
      assignment_chain = _route_sha256(
        route_payload["assignment_chain_sha256"],
        "assignment chain",
      )
      route_commitment = _route_sha256(
        route_payload["route_commitment_sha256"],
        "route commitment",
      )
      if any(
        route.route_counter == route_counter
        or route.route_identity_sha256 == route_identity
        or route.route_content_sha256 == route_content
        for route in learner._routes
      ):
        raise ValueError("calibration evidence contains a duplicate route")
      raw_route_nodes = route_payload["nodes"]
      if type(raw_route_nodes) is not list or len(raw_route_nodes) != len(
        learner._nodes
      ):
        raise ValueError("calibration route node count is incompatible")
      route_nodes = tuple(
        _decode_node_summary(
          raw_route_node,
          node_index,
          f"routes[{route_index}].nodes[{node_index}]",
        )
        for node_index, raw_route_node in enumerate(raw_route_nodes)
      )
      raw_intervals = route_payload["intervals"]
      if type(raw_intervals) is not list or len(raw_intervals) != len(
        learner._nodes
      ) - 1:
        raise ValueError("calibration route interval count is incompatible")
      route_intervals = tuple(
        _IntervalEvidence.decoded(
          raw_interval,
          f"routes[{route_index}].intervals[{interval_index}]",
        )
        for interval_index, raw_interval in enumerate(raw_intervals)
      )
      route = _RouteEvidence(
        route_index=route_index,
        route_counter=route_counter,
        route_identity_sha256=route_identity,
        route_content_sha256=route_content,
        source_accounting=source_accounting,
        nodes=route_nodes,
        intervals=route_intervals,
        assignment_record_count=assignment_record_count,
        assignment_chain_sha256=assignment_chain,
        route_commitment_sha256=route_commitment,
        non_authoritative_record_count=non_authoritative_count,
      )
      _validate_interval_conservation(route)
      learner._routes.append(route)
    actual_commitments = tuple(
      (route.route_identity_sha256, route.route_commitment_sha256)
      for route in learner._routes
    )
    if expected_route_commitments is not None:
      if (
        type(expected_route_commitments) is not tuple
        or any(
          type(item) is not tuple or len(item) != 2
          for item in expected_route_commitments
        )
        or expected_route_commitments != actual_commitments
      ):
        raise ValueError("calibration expected route commitments are incomplete or reordered")
      if any(route.non_authoritative_record_count for route in learner._routes):
        raise ValueError("non-authoritative calibration rows cannot be authenticated")
      learner._restore_authoritative = True
    else:
      learner._restore_authoritative = False

    canonical_keys = tuple(
      (
        route.route_identity_sha256,
        route.route_content_sha256,
        route.route_counter,
      )
      for route in _canonical_routes(tuple(learner._routes))
    )
    serialized_keys = tuple(
      (
        route.route_identity_sha256,
        route.route_content_sha256,
        route.route_counter,
      )
      for route in learner._routes
    )
    if serialized_keys != canonical_keys:
      raise ValueError("calibration routes are not in canonical identity order")
    if sum(route.source_accounting.accepted for route in learner._routes) != (
      sample_accounting.accepted_sample_count
    ):
      raise ValueError("calibration route source accounting disagrees with accepted samples")
    if any(
      route.source_accounting.accepted
      > route.assignment_record_count + route.non_authoritative_record_count
      for route in learner._routes
    ) or sum(
      route.assignment_record_count + route.non_authoritative_record_count
      for route in learner._routes
    ) != sample_accounting.ingested_sample_count:
      raise ValueError("calibration route assignment accounting is incomplete")
    if any(
      not hmac.compare_digest(route.route_commitment_sha256, _route_commitment(route))
      for route in learner._routes
    ):
      raise ValueError("calibration route commitment disagrees with evidence")

    reconstructed_nodes = tuple(
      _aggregate_nodes(tuple(route.nodes[index] for route in learner._routes))
      for index in range(len(learner._nodes))
    )
    if any(
      _encode_node_summary(reconstructed, index)
      != _encode_node_summary(aggregate, index)
      for index, (reconstructed, aggregate) in enumerate(
        zip(reconstructed_nodes, aggregate_nodes, strict=True)
      )
    ):
      raise ValueError("route-level calibration statistics are inconsistent")
    learner._nodes = reconstructed_nodes
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
    "measured_rack_angle_deg",
  )
