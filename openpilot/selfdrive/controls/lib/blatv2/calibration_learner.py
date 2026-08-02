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
9 is deliberately incompatible with older evidence because route uncertainty
is bound to immutable route/content identities as well as per-route sufficient
statistics and the canonical route counter owns the train/validation split.
The route boundary, rather than individual
100 Hz samples, is the uncertainty unit used by selection and validation.

The inverse map is selected from a deterministic nested family. Every model
is fit on training routes only and must clear the paired whole-route loss
envelope in every training stratum; a denser population can therefore never
buy improvement by sacrificing rare breakaway evidence. The selected model is
then frozen and checked once against wholly held-out routes. Validation never
participates in selection or fallback choice.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from openpilot.selfdrive.controls.lib.blatv2.learner import (
  ActuatorBoundary,
  LearningSample,
  MIN_APPLIED_TORQUE_SPAN,
  MIN_EXCITATION_NODE_WEIGHT,
  MIN_LATERAL_ACCEL_RMS_MPS2,
  MIN_LATERAL_ACCEL_SPAN_MPS2,
  minimum_clean_support_s,
)


CALIBRATION_EVIDENCE_SCHEMA_VERSION = 9
MIN_VALIDATION_SUPPORT_FRACTION = 0.20
MIN_STRATUM_TRAINING_ROWS = 4
MIN_STRATUM_VALIDATION_ROWS = 4
NORMAL_MATRIX_RELATIVE_PIVOT_MIN = 1e-10
# Floating-point cancellation guard only. Physical acceptance uses the
# route-level paired uncertainty interval implemented below.
NUMERICAL_LOSS_EPSILON_MULTIPLIER = 64.0
FIT_CONDITION_LIMIT = 1.0 / math.sqrt(sys.float_info.epsilon)


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
      or self.accepted_sample_count < 0
      or type(self.rejection_counts) is not tuple
      or len(self.rejection_counts) != len(_REJECTED_SAMPLE_DISPOSITIONS)
      or any(type(count) is not int or count < 0 for count in self.rejection_counts)
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
      type(accepted) is not int
      or accepted < 0
      or type(ingested) is not int
      or ingested < 0
      or type(rejected) is not int
      or rejected < 0
      or type(reasons) is not dict
      or set(reasons) != expected_reasons
      or any(type(count) is not int or count < 0 for count in reasons.values())
    ):
      raise ValueError("calibration sample accounting is invalid")
    counts = tuple(reasons[disposition.value] for disposition in _REJECTED_SAMPLE_DISPOSITIONS)
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
  INSUFFICIENT_VALIDATION = "insufficient_validation"
  INSUFFICIENT_EXCITATION = "insufficient_excitation"
  INSUFFICIENT_MOVING_EVIDENCE = "insufficient_moving_evidence"
  INSUFFICIENT_BREAKAWAY_EVIDENCE = "insufficient_breakaway_evidence"
  RANK_DEFICIENT_FIT = "rank_deficient_fit"
  ILL_CONDITIONED_FIT = "ill_conditioned_fit"
  SINGULAR_FIT = "singular_fit"
  INVALID_PARAMETERS = "invalid_parameters"
  VALIDATION_INCONCLUSIVE = "validation_inconclusive"
  VALIDATION_REGRESSION = "validation_regression"
  MOVING_VALIDATION_REGRESSION = "moving_validation_regression"
  BREAKAWAY_VALIDATION_REGRESSION = "breakaway_validation_regression"
  AUTHORITY_VALIDATION_REGRESSION = "authority_validation_regression"
  INTERPOLATION_TRAINING_INCONCLUSIVE = "interpolation_training_inconclusive"
  INTERPOLATION_TRAINING_REGRESSION = "interpolation_training_regression"
  INTERPOLATION_VALIDATION_INCONCLUSIVE = "interpolation_validation_inconclusive"
  INTERPOLATION_VALIDATION_REGRESSION = "interpolation_validation_regression"


class CalibrationFitStatus(StrEnum):
  IDENTIFIABLE = "identifiable"
  RANK_DEFICIENT = "rank_deficient"
  ILL_CONDITIONED = "ill_conditioned"
  NO_SOLUTION = "no_solution"


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

  __slots__ = ("normal", "rhs", "target_squared", "weight_s", "count")

  def __init__(self) -> None:
    self.normal = [0.0] * 100
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
      "rhs": [value.hex() for value in self.rhs],
      "target_squared": self.target_squared.hex(),
      "weight_s": self.weight_s.hex(),
    }

  @classmethod
  def decoded(cls, raw: object, context: str) -> _JointRegression:
    payload = _exact(
      raw,
      {"count", "normal", "rhs", "target_squared", "weight_s"},
      context,
    )
    if type(payload["count"]) is not int or payload["count"] < 0:
      raise ValueError(f"{context}.count is invalid")
    result = cls()
    result.count = payload["count"]
    result.normal[:] = _hex_list(payload["normal"], 100, f"{context}.normal")
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
      for value in (*result.normal, *result.rhs, result.target_squared)
    ):
      raise ValueError(f"{context} empty statistics are nonzero")
    return result


class _RouteNodeRegressions:
  __slots__ = (
    "training",
    "moving_training",
    "breakaway_training",
    "breakaway_episode_training",
    "authority_training",
  )

  def __init__(self) -> None:
    # Names retain their aggregate counterparts. The route's population is
    # carried separately, so these are training or validation according to
    # _RouteEvidence.validation.
    self.training = _Regression()
    self.moving_training = _Regression()
    self.breakaway_training = _Regression()
    self.breakaway_episode_training = _Regression()
    self.authority_training = _Regression()


@dataclass(slots=True)
class _RouteEvidence:
  route_index: int
  route_counter: int
  route_identity_sha256: str
  route_content_sha256: str
  validation: bool
  nodes: tuple[_RouteNodeRegressions, ...]
  intervals: tuple[_JointRegression, ...]


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
  if result.count < 0 or result.weight_s < -1e-12:
    raise ValueError("calibration strata are not a disjoint partition")
  if result.weight_s < 0.0:
    result.weight_s = 0.0
  if result.target_squared < 0.0 and abs(result.target_squared) <= 1e-12:
    result.target_squared = 0.0
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
    self.moving_validation_direction_mask = 0
    self.breakaway_direction_mask = 0
    self.breakaway_episode_direction_mask = 0
    self.breakaway_episode_training_direction_mask = 0
    self.breakaway_episode_validation_direction_mask = 0
    self.breakaway_episode_count = 0
    self.breakaway_episode_dwell_s = 0.0
    self.breakaway_angle_assisted_count = 0
    self.breakaway_bracket_width_sum = 0.0
    self.training = _Regression()
    self.validation = _Regression()
    self.moving_training = _Regression()
    self.moving_validation = _Regression()
    self.breakaway_training = _Regression()
    self.breakaway_validation = _Regression()
    self.breakaway_episode_training = _Regression()
    self.breakaway_episode_validation = _Regression()
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
  breakaway_episode_training_weight: float
  breakaway_episode_training_count: int
  breakaway_episode_validation_weight: float
  breakaway_episode_validation_count: int
  breakaway_episode_dwell_s: float
  breakaway_angle_assisted_count: int
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
  selected_model: CalibrationModelId | None = None
  base_seed_validation_rms: float | None = None
  base_candidate_validation_rms: float | None = None
  breakaway_episode_training_count: int = 0
  breakaway_episode_validation_count: int = 0
  breakaway_episode_dwell_s: float = 0.0
  breakaway_angle_assisted_count: int = 0
  breakaway_episode_seed_validation_rms: float | None = None
  breakaway_episode_candidate_validation_rms: float | None = None
  breakaway_mean_bracket_width: float | None = None
  fit_diagnostics: tuple[CalibrationModelFitDiagnostic, ...] = ()
  training_paired_loss: CalibrationPairedLossDiagnostic | None = None
  validation_paired_loss: CalibrationPairedLossDiagnostic | None = None
  training_outcome: CalibrationQualificationReason | None = None

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
class CalibrationInterpolationQualificationReport:
  interval_index: int
  lower_speed_mps: float
  upper_speed_mps: float
  training_paired_loss: CalibrationPairedLossDiagnostic
  validation_paired_loss: CalibrationPairedLossDiagnostic
  reasons: tuple[CalibrationQualificationReason, ...]

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
  if identical:
    return _PairedLossVerdict.NO_REGRESSION
  if diagnostic.route_count == 0:
    return _PairedLossVerdict.NO_DATA
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
  *,
  validation: bool,
) -> tuple[_Regression, ...]:
  result: list[_Regression] = []
  for route in routes:
    if route.validation != validation:
      continue
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
  validation: bool,
  require_improvement: bool,
) -> tuple[bool, tuple[CalibrationPairedLossDiagnostic, ...]]:
  diagnostics = tuple(
    _paired_route_losses(
      _route_regressions_for_node(
        routes, node_index, field, validation=validation
      ),
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
  candidate_lower: CalibrationParameters,
  candidate_upper: CalibrationParameters,
  seed_lower: CalibrationParameters,
  seed_upper: CalibrationParameters,
  *,
  validation: bool,
) -> CalibrationPairedLossDiagnostic:
  deltas: list[float] = []
  tolerance = 0.0
  for route in routes:
    if route.validation != validation:
      continue
    evidence = route.intervals[interval_index]
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
    self._nodes = tuple(_Node() for _ in seed_profile.nodes)
    self._breakaway_detector = BreakawayEpisodeDetector()
    self._route_active = False
    self._route_validation = False
    self._active_route_counter: int | None = None
    self._active_route_identity_sha256: str | None = None
    self._active_route_content_sha256: str | None = None
    self._routes: list[_RouteEvidence] = []
    # Single-owner mutable counters keep the 100 Hz path allocation-free.
    # Immutable snapshots are materialized only for export/status consumers.
    self._sample_disposition_counts = [0] * len(_SAMPLE_DISPOSITIONS)
    self._active_route_nodes: tuple[_RouteNodeRegressions, ...] | None = None
    self._active_route_intervals: tuple[_JointRegression, ...] | None = None

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
      breakaway_episode_training_weight=(
        node.breakaway_episode_training.weight_s
      ),
      breakaway_episode_training_count=(
        node.breakaway_episode_training.count
      ),
      breakaway_episode_validation_weight=(
        node.breakaway_episode_validation.weight_s
      ),
      breakaway_episode_validation_count=(
        node.breakaway_episode_validation.count
      ),
      breakaway_episode_dwell_s=node.breakaway_episode_dwell_s,
      breakaway_angle_assisted_count=node.breakaway_angle_assisted_count,
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
    self._breakaway_detector.reset()
    for node in self._nodes:
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
    self._route_validation = bool(counter & 1)
    self._active_route_counter = counter
    self._active_route_identity_sha256 = route_identity
    self._active_route_content_sha256 = route_content
    self._active_route_nodes = tuple(
      _RouteNodeRegressions() for _ in self._nodes
    )
    self._active_route_intervals = tuple(
      _JointRegression() for _ in range(len(self._nodes) - 1)
    )
    self._route_active = True

  def end_route(self) -> None:
    if not self._route_active:
      raise RuntimeError("calibration learner route is not active")
    route_nodes = self._active_route_nodes
    route_intervals = self._active_route_intervals
    if route_nodes is None or route_intervals is None:
      raise AssertionError("active calibration route lacks route statistics")
    if self._active_route_identity_sha256 is None or self._active_route_content_sha256 is None:
      raise AssertionError("active calibration route lacks immutable identity")
    if self._active_route_counter is None:
      raise AssertionError("active calibration route lacks canonical counter")
    self._routes.append(
      _RouteEvidence(
        route_index=len(self._routes),
        route_counter=self._active_route_counter,
        route_identity_sha256=self._active_route_identity_sha256,
        route_content_sha256=self._active_route_content_sha256,
        validation=self._route_validation,
        nodes=route_nodes,
        intervals=route_intervals,
      )
    )
    self.reset_route_transients()
    self._active_route_nodes = None
    self._active_route_intervals = None
    self._active_route_counter = None
    self._active_route_identity_sha256 = None
    self._active_route_content_sha256 = None
    self._route_active = False

  def _add_regression(
    self,
    node_index: int,
    aggregate_field: str,
    predictors: tuple[float, float, float, float],
    target: float,
    weight: float,
  ) -> None:
    getattr(self._nodes[node_index], aggregate_field).add(
      predictors, target, weight
    )
    route_nodes = self._active_route_nodes
    if route_nodes is None:
      raise AssertionError("calibration route regression lacks active route")
    route_field = aggregate_field.replace("validation", "training")
    getattr(route_nodes[node_index], route_field).add(
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
    predictors: tuple[float, float, float, float],
    target: float,
    weight: float,
  ) -> None:
    intervals = self._active_route_intervals
    if intervals is None:
      raise AssertionError("joint calibration regression lacks active route")
    interval, upper_weight = self._interval_support(speed)
    intervals[interval].add(
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
  ) -> CalibrationSampleDisposition:
    if not isinstance(sample, LearningSample):
      raise TypeError("calibration learner accepts LearningSample only")
    if not self._route_active:
      raise RuntimeError("calibration sample has no pinned route partition")
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
      for node_index, _ in self._supports(sample.speed_mps):
        self._nodes[node_index].rack_reversals += 1

    validation = self._route_validation
    if sample.authority_evidence:
      # A limiter boundary interrupts free breakaway causality. Settled,
      # full-magnitude motion may still identify the moving map.
      self._breakaway_detector.reset()
      seed_at_speed = self.seed_profile.parameters_at(sample.speed_mps).parameters
      joint_direction = self._rate_direction(sample, seed_at_speed)
      if (
        sample.actuator_boundary == ActuatorBoundary.MAGNITUDE
        and sample.magnitude_boundary_dwell_s + 1e-12 >= sample.dt_s
        and joint_direction != 0
      ):
        self._add_joint_regression(
          sample.speed_mps,
          self._row(sample, "moving", joint_direction),
          sample.applied_torque,
          sample.dt_s,
        )
      accepted = False
      for node_index, node_weight in self._supports(sample.speed_mps):
        node = self._nodes[node_index]
        seed = self.seed_profile.nodes[node_index].parameters
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
        direction = self._rate_direction(sample, seed)
        equality_fit = (
          sample.actuator_boundary == ActuatorBoundary.MAGNITUDE
          and sample.magnitude_boundary_dwell_s + 1e-12 >= sample.dt_s
          and direction != 0
        )
        if not equality_fit:
          node.authority_unresolved_sample_count += bool(
            sample.actuator_boundary & ActuatorBoundary.MAGNITUDE
          )
          self._reset_node(node)
          accepted = True
          continue
        predictors = self._row(sample, "moving", direction)
        self._add_regression(
          node_index,
          "authority_validation" if validation else "authority_training",
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
    )
    category = decision.category.value
    episode = decision.episode
    if category == "discarded":
      for node in self._nodes:
        self._reset_node(node)
      return CalibrationSampleDisposition.BREAKAWAY_EPISODE_DISCARDED
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
        joint_predictors = (
          -episode.first_motion.measured_lateral_accel_mps2,
          1.0,
          0.0,
          float(direction),
        )
        joint_torque = episode.first_motion.applied_torque
        joint_weight = episode.first_motion.dt_s
      else:
        joint_predictors = self._row(sample, category, direction)
      self._add_joint_regression(
        support_speed,
        joint_predictors,
        joint_torque,
        joint_weight,
      )
    accepted = False
    for node_index, node_weight in self._supports(support_speed):
      node = self._nodes[node_index]
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
          "validation" if validation else "training",
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
          if validation:
            node.moving_validation_direction_mask |= (
              1 if direction < 0 else 2
            )
          else:
            node.moving_training_direction_mask |= (
              1 if direction < 0 else 2
            )
          self._add_regression(
            node_index,
            "moving_validation" if validation else "moving_training",
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
            (
              "breakaway_validation"
              if validation
              else "breakaway_training"
            ),
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
            (
              "breakaway_episode_validation"
              if validation
              else "breakaway_episode_training"
            ),
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
          if validation:
            node.breakaway_episode_validation_direction_mask |= (
              1 if direction < 0 else 2
            )
          else:
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
    if upstream_rejection is not None:
      # Runtime first-cause classification is authoritative. An upstream
      # failure resets physical continuity and never reaches accumulation,
      # even if a contradictory test caller attaches it to a clean sample.
      self.reset_route_transients()
      self._sample_disposition_counts[
        _SAMPLE_DISPOSITION_INDEX[upstream_rejection]
      ] += 1
      return upstream_rejection
    learner_disposition = self._add_sample_without_accounting(sample)
    self._sample_disposition_counts[
      _SAMPLE_DISPOSITION_INDEX[learner_disposition]
    ] += 1
    return learner_disposition

  def add_sample(self, sample: LearningSample) -> bool:
    """Compatibility wrapper for callers that need only accepted/rejected."""
    return (
      self.add_sample_with_disposition(sample)
      is CalibrationSampleDisposition.ACCEPTED
    )

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
    if (
      node.moving_training.count < MIN_STRATUM_TRAINING_ROWS
      or node.moving_validation.count < MIN_STRATUM_VALIDATION_ROWS
      or node.moving_training_direction_mask != 3
      or node.moving_validation_direction_mask != 3
    ):
      reasons.append(CalibrationQualificationReason.INSUFFICIENT_MOVING_EVIDENCE)
    if (
      node.breakaway_training.count < MIN_STRATUM_TRAINING_ROWS
      or node.breakaway_validation.count < MIN_STRATUM_VALIDATION_ROWS
      or node.breakaway_episode_training.count < MIN_STRATUM_TRAINING_ROWS
      or node.breakaway_episode_validation.count < MIN_STRATUM_VALIDATION_ROWS
      or node.breakaway_episode_training_direction_mask != 3
      or node.breakaway_episode_validation_direction_mask != 3
    ):
      reasons.append(CalibrationQualificationReason.INSUFFICIENT_BREAKAWAY_EVIDENCE)

    # Whether authority evidence participates in fitting is a training-only
    # decision.  Held-out row presence may reject that frozen choice, but it
    # must never change which model or coefficients training selects.
    authority_fit_active = (
      node.authority_training.count >= MIN_STRATUM_TRAINING_ROWS
    )
    if (
      authority_fit_active
      and node.authority_validation.count
      < MIN_STRATUM_VALIDATION_ROWS
    ):
      reasons.append(CalibrationQualificationReason.INSUFFICIENT_VALIDATION)
    seed_coefficients = _seed_coefficients(seed)
    base_training = _subtract(
      node.training,
      node.moving_training,
      node.breakaway_training,
    )
    base_validation = _subtract(
      node.validation,
      node.moving_validation,
      node.breakaway_validation,
    )
    moving_fit = (
      _combine(node.moving_training, node.authority_training)
      if authority_fit_active
      else node.moving_training
    )
    training_strata = (
      base_training,
      node.moving_training,
      node.breakaway_training,
      node.breakaway_episode_training,
      *(
        (node.authority_training,)
        if node.authority_training.count > 0
        else ()
      ),
      node.training,
    )
    seed_training_vector = _rms_vector(
      training_strata,
      seed_coefficients,
    )
    training_fields = (
      "base_training",
      "moving_training",
      "breakaway_training",
      "breakaway_episode_training",
      *(("authority_training",) if node.authority_training.count > 0 else ()),
      "training",
    )
    selected: _ModelCandidate | None = None
    generated_raw: list[
      tuple[
        CalibrationModelId,
        tuple[int, ...],
        tuple[float, float, float, float] | None,
      ]
    ] = [
      (
        CalibrationModelId.STATIC_ONLY,
        (),
        _fit_episode_static(
          node.breakaway_episode_training,
          seed_coefficients,
        ),
      ),
    ]
    for model, free in (
      (CalibrationModelId.FRICTION_MAP, (2,)),
      (CalibrationModelId.OFFSET_AND_FRICTION, (1, 2)),
      (CalibrationModelId.FULL_MAP, (0, 1, 2)),
    ):
      moving_coefficients = _fit_bounded_subset(
        moving_fit,
        seed_coefficients,
        free,
      )
      generated_raw.append((
        model,
        free,
        (
          None
          if moving_coefficients is None
          else _fit_episode_static(
            node.breakaway_episode_training,
            moving_coefficients,
          )
        ),
      ))
    fit_diagnostics = tuple(
      _model_fit_diagnostic(
        model,
        moving_fit,
        node.breakaway_episode_training,
        free,
        coefficients,
      )
      for model, free, coefficients in generated_raw
    )
    generated = tuple(
      (model, coefficients)
      for (model, _, coefficients), diagnostic in zip(
        generated_raw, fit_diagnostics, strict=True
      )
      if diagnostic.status is CalibrationFitStatus.IDENTIFIABLE
    )
    if seed_training_vector is not None:
      for model, coefficients in generated:
        if coefficients is None:
          continue
        vector = _rms_vector(training_strata, coefficients)
        safe_against_seed, _ = _safe_on_route_strata(
          tuple(self._routes),
          index,
          training_fields,
          coefficients,
          seed_coefficients,
          validation=False,
          require_improvement=True,
        )
        if vector is None or not safe_against_seed:
          continue
        candidate = _ModelCandidate(model, coefficients, vector)
        if selected is None:
          selected = candidate
          continue
        dominates, _ = _safe_on_route_strata(
          tuple(self._routes),
          index,
          training_fields,
          coefficients,
          selected.coefficients,
          validation=False,
          require_improvement=True,
        )
        if dominates:
          selected = candidate

    identifiable_model_exists = any(
      diagnostic.status is CalibrationFitStatus.IDENTIFIABLE
      for diagnostic in fit_diagnostics
    )
    seed_retained = selected is None and identifiable_model_exists
    coefficients = (
      selected.coefficients
      if selected is not None
      else seed_coefficients
      if seed_retained
      else None
    )
    seed_rms = node.validation.rms(seed_coefficients)
    candidate_rms = node.validation.rms(coefficients) if coefficients is not None else None
    base_seed = base_validation.rms(seed_coefficients)
    base_candidate = base_validation.rms(coefficients) if coefficients is not None else None
    moving_seed = node.moving_validation.rms(seed_coefficients)
    moving_candidate = node.moving_validation.rms(coefficients) if coefficients is not None else None
    breakaway_seed = node.breakaway_validation.rms(seed_coefficients)
    breakaway_candidate = node.breakaway_validation.rms(coefficients) if coefficients is not None else None
    episode_seed = node.breakaway_episode_validation.rms(seed_coefficients)
    episode_candidate = (
      node.breakaway_episode_validation.rms(coefficients)
      if coefficients is not None
      else None
    )
    authority_seed = node.authority_validation.rms(seed_coefficients)
    authority_candidate = node.authority_validation.rms(coefficients) if coefficients is not None else None
    training_paired_loss = (
      None
      if coefficients is None
      else _paired_route_losses(
        _route_regressions_for_node(
          tuple(self._routes), index, "training", validation=False
        ),
        coefficients,
        seed_coefficients,
      )
    )
    validation_paired_loss = (
      None
      if coefficients is None
      else _paired_route_losses(
        _route_regressions_for_node(
          tuple(self._routes), index, "training", validation=True
        ),
        coefficients,
        seed_coefficients,
      )
    )
    candidate_parameters: CalibrationParameters | None = None
    if coefficients is None:
      statuses = {diagnostic.status for diagnostic in fit_diagnostics}
      if CalibrationFitStatus.ILL_CONDITIONED in statuses:
        reasons.append(CalibrationQualificationReason.ILL_CONDITIONED_FIT)
      elif CalibrationFitStatus.RANK_DEFICIENT in statuses:
        reasons.append(CalibrationQualificationReason.RANK_DEFICIENT_FIT)
      else:
        reasons.append(CalibrationQualificationReason.SINGULAR_FIT)
    else:
      gain, intercept, kinetic, static = coefficients
      offset = intercept / gain if gain != 0.0 else math.inf
      offset_bound = max(1.0, lat_span)
      if not all(math.isfinite(value) for value in (*coefficients, offset)) or gain <= 0.0 or static < kinetic or kinetic < 0.0 or abs(offset) > offset_bound:
        reasons.append(CalibrationQualificationReason.INVALID_PARAMETERS)
      else:
        validation_comparisons = (
          ("training", CalibrationQualificationReason.VALIDATION_REGRESSION),
          ("base_training", CalibrationQualificationReason.VALIDATION_REGRESSION),
          ("moving_training", CalibrationQualificationReason.MOVING_VALIDATION_REGRESSION),
          ("breakaway_training", CalibrationQualificationReason.BREAKAWAY_VALIDATION_REGRESSION),
          ("breakaway_episode_training", CalibrationQualificationReason.BREAKAWAY_VALIDATION_REGRESSION),
          *((
            ("authority_training", CalibrationQualificationReason.AUTHORITY_VALIDATION_REGRESSION),
          ) if node.authority_validation.count > 0 else ()),
        )
        for field, regression_reason in validation_comparisons:
          diagnostic = _paired_route_losses(
            _route_regressions_for_node(
              tuple(self._routes), index, field, validation=True
            ),
            coefficients,
            seed_coefficients,
          )
          verdict = _paired_loss_verdict(
            diagnostic,
            identical=seed_retained,
          )
          if verdict is _PairedLossVerdict.REGRESSION:
            reasons.append(regression_reason)
          elif verdict in (
            _PairedLossVerdict.INCONCLUSIVE,
            _PairedLossVerdict.NO_DATA,
          ):
            reasons.append(
              CalibrationQualificationReason.VALIDATION_INCONCLUSIVE
            )
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
          torque_per_lateral_accel=(
            seed.torque_per_lateral_accel if seed_retained else gain
          ),
          lateral_accel_offset_correction_mps2=(
            seed.lateral_accel_offset_correction_mps2
            if seed_retained
            else offset
          ),
          kinetic_friction_torque=(
            seed.kinetic_friction_torque if seed_retained else kinetic
          ),
          static_breakaway_torque=(
            seed.static_breakaway_torque if seed_retained else static
          ),
          transport_delay_s=seed.transport_delay_s,
          rack_rate_resolution_deg_s=seed.rack_rate_resolution_deg_s,
          confidence=confidence,
          qualified=False,
        )
    unique = tuple(dict.fromkeys(reasons))
    qualified = not unique
    training_outcome = (
      CalibrationQualificationReason.SEED_RETAINED
      if seed_retained
      else CalibrationQualificationReason.LEARNED
      if selected is not None
      else None
    )
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
    report_reasons = (
      (training_outcome,)
      if qualified and training_outcome is not None
      else (CalibrationQualificationReason.QUALIFIED,)
      if qualified
      else unique
    )
    return CalibrationNodeQualificationReport(
      index,
      speed,
      minimum,
      node.clean_support_s,
      node.supported_sample_count,
      node.training.count,
      node.validation.count,
      node.validation.weight_s,
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
      report_reasons,
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
      None if selected is None else selected.model,
      base_seed,
      base_candidate,
      node.breakaway_episode_training.count,
      node.breakaway_episode_validation.count,
      node.breakaway_episode_dwell_s,
      node.breakaway_angle_assisted_count,
      episode_seed,
      episode_candidate,
      (
        None
        if (
          node.breakaway_episode_training.weight_s
          + node.breakaway_episode_validation.weight_s
        ) <= 0.0
        else node.breakaway_bracket_width_sum
        / (
          node.breakaway_episode_training.weight_s
          + node.breakaway_episode_validation.weight_s
        )
      ),
      fit_diagnostics,
      training_paired_loss,
      validation_paired_loss,
      training_outcome,
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
    interpolation_reports: list[CalibrationInterpolationQualificationReport] = []
    for interval_index in range(len(self._nodes) - 1):
      training = _paired_joint_route_losses(
        tuple(self._routes),
        interval_index,
        candidate_parameters[interval_index],
        candidate_parameters[interval_index + 1],
        seed_parameters[interval_index],
        seed_parameters[interval_index + 1],
        validation=False,
      )
      validation = _paired_joint_route_losses(
        tuple(self._routes),
        interval_index,
        candidate_parameters[interval_index],
        candidate_parameters[interval_index + 1],
        seed_parameters[interval_index],
        seed_parameters[interval_index + 1],
        validation=True,
      )
      interval_reasons: list[CalibrationQualificationReason] = []
      for diagnostic, regression_reason, inconclusive_reason in (
        (
          training,
          CalibrationQualificationReason.INTERPOLATION_TRAINING_REGRESSION,
          CalibrationQualificationReason.INTERPOLATION_TRAINING_INCONCLUSIVE,
        ),
        (
          validation,
          CalibrationQualificationReason.INTERPOLATION_VALIDATION_REGRESSION,
          CalibrationQualificationReason.INTERPOLATION_VALIDATION_INCONCLUSIVE,
        ),
      ):
        verdict = _paired_loss_verdict(
          diagnostic,
          identical=profile_identical,
        )
        if verdict is _PairedLossVerdict.REGRESSION:
          interval_reasons.append(regression_reason)
        elif verdict in (
          _PairedLossVerdict.INCONCLUSIVE,
          _PairedLossVerdict.NO_DATA,
        ):
          interval_reasons.append(inconclusive_reason)
      interpolation_reports.append(
        CalibrationInterpolationQualificationReport(
          interval_index=interval_index,
          lower_speed_mps=self.speed_nodes_mps[interval_index],
          upper_speed_mps=self.speed_nodes_mps[interval_index + 1],
          training_paired_loss=training,
          validation_paired_loss=validation,
          reasons=(
            (CalibrationQualificationReason.QUALIFIED,)
            if not interval_reasons
            else tuple(dict.fromkeys(interval_reasons))
          ),
        )
      )
    interpolation_tuple = tuple(interpolation_reports)
    if not all(report.qualified for report in interpolation_tuple):
      return CalibrationLearningResult(reports, None, interpolation_tuple)
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
    model_ids = ",".join(
      report.selected_model.value
      for report in reports
      if report.selected_model is not None
    )
    candidate_provenance = (
      f"{source}; observable-inverse-torque-learner-v2; " +
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
    if self._route_active:
      raise RuntimeError("active route evidence cannot be exported")
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
      "moving_training_direction_mask",
      "moving_validation_direction_mask",
      "breakaway_direction_mask",
      "breakaway_episode_direction_mask",
      "breakaway_episode_training_direction_mask",
      "breakaway_episode_validation_direction_mask",
      "breakaway_episode_count",
      "breakaway_episode_dwell_s",
      "breakaway_angle_assisted_count",
      "breakaway_bracket_width_sum",
    )
    regression_fields = (
      "training",
      "validation",
      "moving_training",
      "moving_validation",
      "breakaway_training",
      "breakaway_validation",
      "breakaway_episode_training",
      "breakaway_episode_validation",
      "authority_training",
      "authority_validation",
    )
    route_regression_fields = (
      "training",
      "moving_training",
      "breakaway_training",
      "breakaway_episode_training",
      "authority_training",
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
    routes = [
      {
        "route_index": route.route_index,
        "route_counter": route.route_counter,
        "route_identity_sha256": route.route_identity_sha256,
        "route_content_sha256": route.route_content_sha256,
        "validation": route.validation,
        "nodes": [
          {
            name: getattr(route_node, name).encoded()
            for name in route_regression_fields
          }
          for route_node in route.nodes
        ],
        "intervals": [interval.encoded() for interval in route.intervals],
      }
      for route in self._routes
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
      "moving_training_direction_mask",
      "moving_validation_direction_mask",
      "breakaway_direction_mask",
      "breakaway_episode_direction_mask",
      "breakaway_episode_training_direction_mask",
      "breakaway_episode_validation_direction_mask",
      "breakaway_episode_count",
      "breakaway_episode_dwell_s",
      "breakaway_angle_assisted_count",
      "breakaway_bracket_width_sum",
    )
    float_fields = {
      name
      for name in node_fields
      if name.endswith("_s")
      or name in {
        "lat_energy",
        "rack_travel_deg",
        "breakaway_bracket_width_sum",
      }
    }
    regressions = (
      "training",
      "validation",
      "moving_training",
      "moving_validation",
      "breakaway_training",
      "breakaway_validation",
      "breakaway_episode_training",
      "breakaway_episode_validation",
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
      if (
        node.breakaway_episode_count != node.breakaway_sample_count
        or node.breakaway_episode_count
        != node.breakaway_episode_training.count
        + node.breakaway_episode_validation.count
      ):
        raise ValueError("breakaway episode counts are inconsistent")
      if node.breakaway_angle_assisted_count > node.breakaway_episode_count:
        raise ValueError("angle-assisted breakaway count is inconsistent")
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
          "breakaway_episode_dwell_s",
          "breakaway_bracket_width_sum",
        )
      ):
        raise ValueError("calibration support or energy is negative")
      if any(
        getattr(node, name) not in range(4)
        for name in (
          "lateral_accel_direction_mask",
          "applied_torque_direction_mask",
          "moving_direction_mask",
          "moving_training_direction_mask",
          "moving_validation_direction_mask",
          "breakaway_direction_mask",
          "breakaway_episode_direction_mask",
          "breakaway_episode_training_direction_mask",
          "breakaway_episode_validation_direction_mask",
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
    raw_routes = payload["routes"]
    if type(raw_routes) is not list:
      raise ValueError("calibration route statistics must be a list")
    route_regression_fields = (
      "training",
      "moving_training",
      "breakaway_training",
      "breakaway_episode_training",
      "authority_training",
    )
    for route_index, raw_route in enumerate(raw_routes):
      route_payload = _exact(
        raw_route,
        {
          "route_index",
          "route_counter",
          "route_identity_sha256",
          "route_content_sha256",
          "validation",
          "nodes",
          "intervals",
        },
        f"routes[{route_index}]",
      )
      if route_payload["route_index"] != route_index:
        raise ValueError("calibration route ordering is corrupt")
      route_counter = _canonical_route_counter(route_payload["route_counter"])
      if type(route_payload["validation"]) is not bool:
        raise ValueError("calibration route partition is invalid")
      route_identity = _route_sha256(
        route_payload["route_identity_sha256"],
        "route identity",
      )
      route_content = _route_sha256(
        route_payload["route_content_sha256"],
        "route content identity",
      )
      if route_payload["validation"] != bool(route_counter & 1):
        raise ValueError("calibration route partition does not match counter")
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
      route_nodes: list[_RouteNodeRegressions] = []
      for node_index, raw_route_node in enumerate(raw_route_nodes):
        route_node_payload = _exact(
          raw_route_node,
          set(route_regression_fields),
          f"routes[{route_index}].nodes[{node_index}]",
        )
        route_node = _RouteNodeRegressions()
        for name in route_regression_fields:
          setattr(
            route_node,
            name,
            _Regression.decoded(
              route_node_payload[name],
              f"routes[{route_index}].nodes[{node_index}].{name}",
            ),
          )
        route_nodes.append(route_node)
      raw_intervals = route_payload["intervals"]
      if type(raw_intervals) is not list or len(raw_intervals) != len(
        learner._nodes
      ) - 1:
        raise ValueError("calibration route interval count is incompatible")
      route_intervals = tuple(
        _JointRegression.decoded(
          raw_interval,
          f"routes[{route_index}].intervals[{interval_index}]",
        )
        for interval_index, raw_interval in enumerate(raw_intervals)
      )
      learner._routes.append(
        _RouteEvidence(
          route_index=route_index,
          route_counter=route_counter,
          route_identity_sha256=route_identity,
          route_content_sha256=route_content,
          validation=route_payload["validation"],
          nodes=tuple(route_nodes),
          intervals=route_intervals,
        )
      )

    aggregate_pairs = (
      ("training", "training", False),
      ("validation", "training", True),
      ("moving_training", "moving_training", False),
      ("moving_validation", "moving_training", True),
      ("breakaway_training", "breakaway_training", False),
      ("breakaway_validation", "breakaway_training", True),
      (
        "breakaway_episode_training",
        "breakaway_episode_training",
        False,
      ),
      (
        "breakaway_episode_validation",
        "breakaway_episode_training",
        True,
      ),
      ("authority_training", "authority_training", False),
      ("authority_validation", "authority_training", True),
    )
    for node_index, node in enumerate(learner._nodes):
      for aggregate_name, route_name, validation in aggregate_pairs:
        parts = tuple(
          getattr(route.nodes[node_index], route_name)
          for route in learner._routes
          if route.validation == validation
        )
        reconstructed = _combine(*parts)
        aggregate = getattr(node, aggregate_name)
        if reconstructed.count != aggregate.count or not math.isclose(
          reconstructed.weight_s,
          aggregate.weight_s,
          rel_tol=1e-12,
          abs_tol=1e-12,
        ):
          raise ValueError("route-level calibration counts are inconsistent")
        values = (
          (reconstructed.target_squared, aggregate.target_squared),
          *zip(reconstructed.normal, aggregate.normal, strict=True),
          *zip(reconstructed.rhs, aggregate.rhs, strict=True),
        )
        if any(
          not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
          for left, right in values
        ):
          raise ValueError("route-level calibration statistics are inconsistent")
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
