"""Strict display-only projection of observable BLaTv2 calibration.

This Params value is an informational projection of already-finalized
calibration evidence.  It is outside controller selection, approval, fitting,
and actuation: deleting or corrupting it cannot change which controller runs.

Schema 3 deliberately rejects the retired physical rack-fit vocabulary.  The
only candidate values it exposes are the four observable inverse-torque
calibration values, while independent base, moving, breakaway, and authority
populations remain visible for audit and UI progress reporting. It also keeps
"fully evaluated" separate from "new artifact available": retaining the seed
at every node is a successful, qualified result with no redundant candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

from openpilot.selfdrive.controls.lib.blatv2.calibration_coordinator import (
  CalibrationLearningFinalization,
  CalibrationNodeSupportDiagnostic,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_learner import (
  MIN_VALIDATION_SUPPORT_FRACTION,
  CalibrationFitStatus,
  CalibrationModelId,
  CalibrationNodeQualificationReport,
  CalibrationQualificationReason,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  RuntimeVehicleBundle,
)


LEARNING_STATUS_PARAM = "BLaTv2LearningStatus"
LEARNING_STATUS_SCHEMA_VERSION = 3
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_KEYS = {
  "all_intervals_qualified",
  "all_nodes_evaluated",
  "all_nodes_qualified",
  "candidate_profile_available",
  "candidate_profile_revision",
  "candidate_profile_sha256",
  "evidence_sha256",
  "informational_only",
  "interpolation_reports",
  "last_drive_complete",
  "manifest_sha256",
  "nodes",
  "runtime_identity_sha256",
  "schema_version",
  "seed_profile_sha256",
  "vehicle_identity",
}
_NODE_KEYS = {
  "applied_torque_directions",
  "applied_torque_span",
  "authority_candidate_validation_rms",
  "authority_fit_sample_count",
  "authority_fit_support_s",
  "authority_sample_count",
  "authority_seed_validation_rms",
  "authority_support_s",
  "authority_training_count",
  "authority_validation_count",
  "base_sample_count",
  "base_support_s",
  "breakaway_candidate_validation_rms",
  "breakaway_sample_count",
  "breakaway_seed_validation_rms",
  "breakaway_support_s",
  "breakaway_training_count",
  "breakaway_validation_count",
  "candidate_parameters",
  "candidate_validation_rms",
  "clean_support_s",
  "confidence",
  "evaluation_status",
  "fit_diagnostics",
  "last_drive_accepted_sample_count",
  "last_drive_authority_fit_sample_count",
  "last_drive_authority_fit_support_s",
  "last_drive_authority_sample_count",
  "last_drive_authority_support_s",
  "last_drive_base_sample_count",
  "last_drive_base_support_s",
  "last_drive_breakaway_sample_count",
  "last_drive_breakaway_support_s",
  "last_drive_clean_support_s",
  "last_drive_moving_sample_count",
  "last_drive_moving_support_s",
  "lateral_accel_directions",
  "lateral_accel_rms_mps2",
  "lateral_accel_span_mps2",
  "minimum_support_s",
  "minimum_validation_support_s",
  "moving_candidate_validation_rms",
  "moving_sample_count",
  "moving_seed_validation_rms",
  "moving_support_s",
  "moving_training_count",
  "moving_validation_count",
  "node_index",
  "qualified",
  "rack_reversals",
  "rack_travel_deg",
  "reasons",
  "seed_validation_rms",
  "speed_mps",
  "supported_sample_count",
  "training_count",
  "training_outcome",
  "training_paired_loss",
  "validation_count",
  "validation_paired_loss",
  "validation_support_s",
}
_FIT_DIAGNOSTIC_KEYS = {
  "breakaway_parameter_count",
  "breakaway_rank",
  "condition_estimate",
  "model",
  "moving_parameter_count",
  "moving_rank",
  "status",
}
_PAIRED_LOSS_KEYS = {
  "lower_bound_mse",
  "mean_candidate_minus_seed_mse",
  "numerical_tolerance_mse",
  "route_count",
  "uncertainty_mse",
  "upper_bound_mse",
}
_INTERPOLATION_KEYS = {
  "interval_index",
  "lower_speed_mps",
  "qualified",
  "reasons",
  "training_paired_loss",
  "upper_speed_mps",
  "validation_paired_loss",
}
_CANDIDATE_PARAMETER_KEYS = {
  "kinetic_friction_torque",
  "lateral_accel_offset_correction_mps2",
  "static_breakaway_torque",
  "torque_per_lateral_accel",
}
_SUPPORT_SUM_REL_TOL = 1e-12
_SUPPORT_SUM_ABS_TOL = 1e-12
_EVALUATION_STATUSES = {
  "evidence_insufficient",
  "ill_conditioned",
  "invalid_parameters",
  "learned",
  "numerical_failure",
  "rank_deficient",
  "seed_retained",
  "validation_inconclusive",
  "validation_regressed",
}


def _support_populations_match(
  clean_support_s: float,
  base_support_s: float,
  moving_support_s: float,
  breakaway_support_s: float,
) -> bool:
  """Match the authoritative evidence validator's accumulation tolerance."""
  return math.isclose(
    clean_support_s,
    base_support_s + moving_support_s + breakaway_support_s,
    rel_tol=_SUPPORT_SUM_REL_TOL,
    abs_tol=_SUPPORT_SUM_ABS_TOL,
  )


def _finite_float(value: object, name: str) -> float:
  if type(value) not in (int, float):
    raise TypeError(f"{name} must be a JSON number")
  numeric = float(value)
  if not math.isfinite(numeric):
    raise ValueError(f"{name} must be finite")
  return 0.0 if numeric == 0.0 else numeric


def _positive_float(value: object, name: str) -> float:
  numeric = _finite_float(value, name)
  if numeric <= 0.0:
    raise ValueError(f"{name} must be positive")
  return numeric


def _nonnegative_float(value: object, name: str) -> float:
  numeric = _finite_float(value, name)
  if numeric < 0.0:
    raise ValueError(f"{name} must be nonnegative")
  return numeric


def _nonnegative_int(value: object, name: str) -> int:
  if type(value) is not int or value < 0:
    raise ValueError(f"{name} must be a nonnegative integer")
  return value


def _sha256(value: object, name: str) -> str:
  if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
    raise ValueError(f"{name} must be a lowercase SHA-256")
  return value


def _canonical_json_bytes(payload: object) -> bytes:
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class DriveNodeBaseline:
  """Cumulative calibration populations before one real drive."""

  node_index: int
  speed_mps: float
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
  authority_fit_support_s: float
  authority_fit_sample_count: int


@dataclass(frozen=True, slots=True)
class DriveEvidenceBaseline:
  """All node-local populations captured before a drive.

  A full snapshot is required so a last-drive value can never silently include
  evidence from an earlier route.  Authority fit support is distinct from all
  authority support because slew build/release rows are deliberately retained
  as authority observations without pretending they are equality-fit rows.
  """

  nodes: tuple[DriveNodeBaseline, ...]

  @classmethod
  def from_support_diagnostics(
    cls,
    diagnostics: tuple[CalibrationNodeSupportDiagnostic, ...],
  ) -> DriveEvidenceBaseline:
    if not diagnostics:
      raise ValueError("learning display requires at least one speed node")
    nodes: list[DriveNodeBaseline] = []
    for diagnostic in diagnostics:
      context = f"baseline.nodes[{diagnostic.node_index}]"
      nodes.append(DriveNodeBaseline(
        node_index=_nonnegative_int(
          diagnostic.node_index,
          f"{context}.node_index",
        ),
        speed_mps=_nonnegative_float(
          diagnostic.speed_mps,
          f"{context}.speed_mps",
        ),
        clean_support_s=_nonnegative_float(
          diagnostic.clean_support_s,
          f"{context}.clean_support_s",
        ),
        supported_sample_count=_nonnegative_int(
          diagnostic.supported_sample_count,
          f"{context}.supported_sample_count",
        ),
        base_support_s=_nonnegative_float(
          diagnostic.base_support_s,
          f"{context}.base_support_s",
        ),
        base_sample_count=_nonnegative_int(
          diagnostic.base_sample_count,
          f"{context}.base_sample_count",
        ),
        moving_support_s=_nonnegative_float(
          diagnostic.moving_support_s,
          f"{context}.moving_support_s",
        ),
        moving_sample_count=_nonnegative_int(
          diagnostic.moving_sample_count,
          f"{context}.moving_sample_count",
        ),
        breakaway_support_s=_nonnegative_float(
          diagnostic.breakaway_support_s,
          f"{context}.breakaway_support_s",
        ),
        breakaway_sample_count=_nonnegative_int(
          diagnostic.breakaway_sample_count,
          f"{context}.breakaway_sample_count",
        ),
        authority_support_s=_nonnegative_float(
          diagnostic.authority_support_s,
          f"{context}.authority_support_s",
        ),
        authority_sample_count=_nonnegative_int(
          diagnostic.authority_sample_count,
          f"{context}.authority_sample_count",
        ),
        authority_fit_support_s=_nonnegative_float(
          diagnostic.authority_fit_support_s,
          f"{context}.authority_fit_support_s",
        ),
        authority_fit_sample_count=_nonnegative_int(
          diagnostic.authority_fit_sample_count,
          f"{context}.authority_fit_sample_count",
        ),
      ))
      node = nodes[-1]
      if not _support_populations_match(
        node.clean_support_s,
        node.base_support_s,
        node.moving_support_s,
        node.breakaway_support_s,
      ):
        raise ValueError(f"{context} clean support populations disagree")
      if node.supported_sample_count != (
        node.base_sample_count
        + node.moving_sample_count
        + node.breakaway_sample_count
      ):
        raise ValueError(f"{context} clean sample populations disagree")
      if (
        node.authority_fit_support_s > node.authority_support_s + 1e-12
        or node.authority_fit_sample_count > node.authority_sample_count
      ):
        raise ValueError(f"{context} authority fit exceeds authority evidence")
    return cls(tuple(nodes))


def _candidate_parameters(
  report: CalibrationNodeQualificationReport,
) -> dict[str, float] | None:
  parameters = report.candidate_parameters
  if parameters is None:
    return None
  return {
    "kinetic_friction_torque": _nonnegative_float(
      parameters.kinetic_friction_torque,
      "candidate_parameters.kinetic_friction_torque",
    ),
    "lateral_accel_offset_correction_mps2": _finite_float(
      parameters.lateral_accel_offset_correction_mps2,
      "candidate_parameters.lateral_accel_offset_correction_mps2",
    ),
    "static_breakaway_torque": _nonnegative_float(
      parameters.static_breakaway_torque,
      "candidate_parameters.static_breakaway_torque",
    ),
    "torque_per_lateral_accel": _positive_float(
      parameters.torque_per_lateral_accel,
      "candidate_parameters.torque_per_lateral_accel",
    ),
  }


def _delta_float(current: float, baseline: float, name: str) -> float:
  delta = _finite_float(current, f"{name}.current") - _finite_float(
    baseline,
    f"{name}.baseline",
  )
  if delta < -1e-12:
    raise ValueError("cumulative learning evidence moved backwards")
  return max(0.0, delta)


def _delta_int(current: int, baseline: int, name: str) -> int:
  delta = _nonnegative_int(current, f"{name}.current") - _nonnegative_int(
    baseline,
    f"{name}.baseline",
  )
  if delta < 0:
    raise ValueError("cumulative learning evidence moved backwards")
  return delta


def _optional_rms(value: float | None, name: str) -> float | None:
  return None if value is None else _nonnegative_float(value, name)


def _paired_loss_payload(diagnostic: object | None, context: str) -> dict[str, object] | None:
  if diagnostic is None:
    return None
  return {
    "lower_bound_mse": (
      None
      if diagnostic.lower_bound_mse is None
      else _finite_float(diagnostic.lower_bound_mse, f"{context}.lower_bound_mse")
    ),
    "mean_candidate_minus_seed_mse": (
      None
      if diagnostic.mean_candidate_minus_seed_mse is None
      else _finite_float(
        diagnostic.mean_candidate_minus_seed_mse,
        f"{context}.mean_candidate_minus_seed_mse",
      )
    ),
    "numerical_tolerance_mse": (
      None
      if diagnostic.numerical_tolerance_mse is None
      else _nonnegative_float(
        diagnostic.numerical_tolerance_mse,
        f"{context}.numerical_tolerance_mse",
      )
    ),
    "route_count": _nonnegative_int(
      diagnostic.route_count,
      f"{context}.route_count",
    ),
    "uncertainty_mse": (
      None
      if diagnostic.uncertainty_mse is None
      else _nonnegative_float(
        diagnostic.uncertainty_mse,
        f"{context}.uncertainty_mse",
      )
    ),
    "upper_bound_mse": (
      None
      if diagnostic.upper_bound_mse is None
      else _finite_float(diagnostic.upper_bound_mse, f"{context}.upper_bound_mse")
    ),
  }


def _fit_diagnostics_payload(report: CalibrationNodeQualificationReport) -> list[dict[str, object]]:
  result: list[dict[str, object]] = []
  for index, diagnostic in enumerate(report.fit_diagnostics):
    context = f"nodes[{report.node_index}].fit_diagnostics[{index}]"
    result.append({
      "breakaway_parameter_count": _nonnegative_int(
        diagnostic.breakaway_parameter_count,
        f"{context}.breakaway_parameter_count",
      ),
      "breakaway_rank": _nonnegative_int(
        diagnostic.breakaway_rank,
        f"{context}.breakaway_rank",
      ),
      "condition_estimate": (
        None
        if diagnostic.condition_estimate is None
        else _nonnegative_float(
          diagnostic.condition_estimate,
          f"{context}.condition_estimate",
        )
      ),
      "model": diagnostic.model.value,
      "moving_parameter_count": _nonnegative_int(
        diagnostic.moving_parameter_count,
        f"{context}.moving_parameter_count",
      ),
      "moving_rank": _nonnegative_int(
        diagnostic.moving_rank,
        f"{context}.moving_rank",
      ),
      "status": diagnostic.status.value,
    })
  return result


def _node_evaluation_status(report: CalibrationNodeQualificationReport) -> str:
  if report.seed_retained:
    return "seed_retained"
  if report.learned:
    return "learned"
  reasons = set(report.reasons)
  if any(reason.value.startswith("insufficient_") for reason in reasons):
    return "evidence_insufficient"
  if CalibrationQualificationReason.RANK_DEFICIENT_FIT in reasons:
    return "rank_deficient"
  if CalibrationQualificationReason.ILL_CONDITIONED_FIT in reasons:
    return "ill_conditioned"
  if CalibrationQualificationReason.VALIDATION_INCONCLUSIVE in reasons:
    return "validation_inconclusive"
  if any("validation_regression" in reason.value for reason in reasons):
    return "validation_regressed"
  if CalibrationQualificationReason.INVALID_PARAMETERS in reasons:
    return "invalid_parameters"
  return "numerical_failure"


def _node_payload(
  report: CalibrationNodeQualificationReport,
  baseline: DriveNodeBaseline | None,
) -> dict[str, object]:
  context = f"nodes[{report.node_index}]"
  last_values: dict[str, float | int | None] = {
    "last_drive_clean_support_s": None,
    "last_drive_accepted_sample_count": None,
    "last_drive_base_support_s": None,
    "last_drive_base_sample_count": None,
    "last_drive_moving_support_s": None,
    "last_drive_moving_sample_count": None,
    "last_drive_breakaway_support_s": None,
    "last_drive_breakaway_sample_count": None,
    "last_drive_authority_support_s": None,
    "last_drive_authority_sample_count": None,
    "last_drive_authority_fit_support_s": None,
    "last_drive_authority_fit_sample_count": None,
  }
  if baseline is not None:
    if baseline.node_index != report.node_index or baseline.speed_mps != report.speed_mps:
      raise ValueError("drive baseline node grid changed")
    last_values = {
      "last_drive_clean_support_s": _delta_float(
        report.clean_support_s,
        baseline.clean_support_s,
        f"{context}.last_drive_clean_support_s",
      ),
      "last_drive_accepted_sample_count": _delta_int(
        report.supported_sample_count,
        baseline.supported_sample_count,
        f"{context}.last_drive_accepted_sample_count",
      ),
      "last_drive_base_support_s": _delta_float(
        report.base_support_s,
        baseline.base_support_s,
        f"{context}.last_drive_base_support_s",
      ),
      "last_drive_base_sample_count": _delta_int(
        report.base_sample_count,
        baseline.base_sample_count,
        f"{context}.last_drive_base_sample_count",
      ),
      "last_drive_moving_support_s": _delta_float(
        report.moving_support_s,
        baseline.moving_support_s,
        f"{context}.last_drive_moving_support_s",
      ),
      "last_drive_moving_sample_count": _delta_int(
        report.moving_sample_count,
        baseline.moving_sample_count,
        f"{context}.last_drive_moving_sample_count",
      ),
      "last_drive_breakaway_support_s": _delta_float(
        report.breakaway_support_s,
        baseline.breakaway_support_s,
        f"{context}.last_drive_breakaway_support_s",
      ),
      "last_drive_breakaway_sample_count": _delta_int(
        report.breakaway_sample_count,
        baseline.breakaway_sample_count,
        f"{context}.last_drive_breakaway_sample_count",
      ),
      "last_drive_authority_support_s": _delta_float(
        report.authority_support_s,
        baseline.authority_support_s,
        f"{context}.last_drive_authority_support_s",
      ),
      "last_drive_authority_sample_count": _delta_int(
        report.authority_sample_count,
        baseline.authority_sample_count,
        f"{context}.last_drive_authority_sample_count",
      ),
      "last_drive_authority_fit_support_s": _delta_float(
        report.authority_fit_support_s,
        baseline.authority_fit_support_s,
        f"{context}.last_drive_authority_fit_support_s",
      ),
      "last_drive_authority_fit_sample_count": _delta_int(
        report.authority_fit_sample_count,
        baseline.authority_fit_sample_count,
        f"{context}.last_drive_authority_fit_sample_count",
      ),
    }

  minimum_support = _nonnegative_float(
    report.minimum_support_s,
    f"{context}.minimum_support_s",
  )
  return {
    "applied_torque_directions": _nonnegative_int(
      report.applied_torque_directions,
      f"{context}.applied_torque_directions",
    ),
    "applied_torque_span": _nonnegative_float(
      report.applied_torque_span,
      f"{context}.applied_torque_span",
    ),
    "authority_candidate_validation_rms": _optional_rms(
      report.authority_candidate_validation_rms,
      f"{context}.authority_candidate_validation_rms",
    ),
    "authority_fit_sample_count": _nonnegative_int(
      report.authority_fit_sample_count,
      f"{context}.authority_fit_sample_count",
    ),
    "authority_fit_support_s": _nonnegative_float(
      report.authority_fit_support_s,
      f"{context}.authority_fit_support_s",
    ),
    "authority_sample_count": _nonnegative_int(
      report.authority_sample_count,
      f"{context}.authority_sample_count",
    ),
    "authority_seed_validation_rms": _optional_rms(
      report.authority_seed_validation_rms,
      f"{context}.authority_seed_validation_rms",
    ),
    "authority_support_s": _nonnegative_float(
      report.authority_support_s,
      f"{context}.authority_support_s",
    ),
    "authority_training_count": _nonnegative_int(
      report.authority_training_count,
      f"{context}.authority_training_count",
    ),
    "authority_validation_count": _nonnegative_int(
      report.authority_validation_count,
      f"{context}.authority_validation_count",
    ),
    "base_sample_count": _nonnegative_int(
      report.base_sample_count,
      f"{context}.base_sample_count",
    ),
    "base_support_s": _nonnegative_float(
      report.base_support_s,
      f"{context}.base_support_s",
    ),
    "breakaway_candidate_validation_rms": _optional_rms(
      report.breakaway_candidate_validation_rms,
      f"{context}.breakaway_candidate_validation_rms",
    ),
    "breakaway_sample_count": _nonnegative_int(
      report.breakaway_sample_count,
      f"{context}.breakaway_sample_count",
    ),
    "breakaway_seed_validation_rms": _optional_rms(
      report.breakaway_seed_validation_rms,
      f"{context}.breakaway_seed_validation_rms",
    ),
    "breakaway_support_s": _nonnegative_float(
      report.breakaway_support_s,
      f"{context}.breakaway_support_s",
    ),
    "breakaway_training_count": _nonnegative_int(
      report.breakaway_training_count,
      f"{context}.breakaway_training_count",
    ),
    "breakaway_validation_count": _nonnegative_int(
      report.breakaway_validation_count,
      f"{context}.breakaway_validation_count",
    ),
    "candidate_parameters": _candidate_parameters(report),
    "candidate_validation_rms": _optional_rms(
      report.candidate_validation_rms,
      f"{context}.candidate_validation_rms",
    ),
    "clean_support_s": _nonnegative_float(
      report.clean_support_s,
      f"{context}.clean_support_s",
    ),
    "confidence": _nonnegative_float(
      report.confidence,
      f"{context}.confidence",
    ),
    "evaluation_status": _node_evaluation_status(report),
    "fit_diagnostics": _fit_diagnostics_payload(report),
    **last_values,
    "lateral_accel_directions": _nonnegative_int(
      report.lateral_accel_directions,
      f"{context}.lateral_accel_directions",
    ),
    "lateral_accel_rms_mps2": _nonnegative_float(
      report.lateral_accel_rms_mps2,
      f"{context}.lateral_accel_rms_mps2",
    ),
    "lateral_accel_span_mps2": _nonnegative_float(
      report.lateral_accel_span_mps2,
      f"{context}.lateral_accel_span_mps2",
    ),
    "minimum_support_s": minimum_support,
    "minimum_validation_support_s": (
      minimum_support * MIN_VALIDATION_SUPPORT_FRACTION
    ),
    "moving_candidate_validation_rms": _optional_rms(
      report.moving_candidate_validation_rms,
      f"{context}.moving_candidate_validation_rms",
    ),
    "moving_sample_count": _nonnegative_int(
      report.moving_sample_count,
      f"{context}.moving_sample_count",
    ),
    "moving_seed_validation_rms": _optional_rms(
      report.moving_seed_validation_rms,
      f"{context}.moving_seed_validation_rms",
    ),
    "moving_support_s": _nonnegative_float(
      report.moving_support_s,
      f"{context}.moving_support_s",
    ),
    "moving_training_count": _nonnegative_int(
      report.moving_training_count,
      f"{context}.moving_training_count",
    ),
    "moving_validation_count": _nonnegative_int(
      report.moving_validation_count,
      f"{context}.moving_validation_count",
    ),
    "node_index": _nonnegative_int(report.node_index, f"{context}.node_index"),
    "qualified": report.qualified,
    "rack_reversals": _nonnegative_int(
      report.rack_reversals,
      f"{context}.rack_reversals",
    ),
    "rack_travel_deg": _nonnegative_float(
      report.rack_travel_deg,
      f"{context}.rack_travel_deg",
    ),
    "reasons": [reason.value for reason in report.reasons],
    "seed_validation_rms": _optional_rms(
      report.seed_validation_rms,
      f"{context}.seed_validation_rms",
    ),
    "speed_mps": _nonnegative_float(report.speed_mps, f"{context}.speed_mps"),
    "supported_sample_count": _nonnegative_int(
      report.supported_sample_count,
      f"{context}.supported_sample_count",
    ),
    "training_count": _nonnegative_int(
      report.training_count,
      f"{context}.training_count",
    ),
    "training_outcome": (
      None if report.training_outcome is None else report.training_outcome.value
    ),
    "training_paired_loss": _paired_loss_payload(
      report.training_paired_loss,
      f"{context}.training_paired_loss",
    ),
    "validation_count": _nonnegative_int(
      report.validation_count,
      f"{context}.validation_count",
    ),
    "validation_paired_loss": _paired_loss_payload(
      report.validation_paired_loss,
      f"{context}.validation_paired_loss",
    ),
    "validation_support_s": _nonnegative_float(
      report.validation_support_s,
      f"{context}.validation_support_s",
    ),
  }


def _interpolation_payload(report: object) -> dict[str, object]:
  context = f"interpolation_reports[{report.interval_index}]"
  return {
    "interval_index": _nonnegative_int(
      report.interval_index,
      f"{context}.interval_index",
    ),
    "lower_speed_mps": _nonnegative_float(
      report.lower_speed_mps,
      f"{context}.lower_speed_mps",
    ),
    "qualified": report.qualified,
    "reasons": [reason.value for reason in report.reasons],
    "training_paired_loss": _paired_loss_payload(
      report.training_paired_loss,
      f"{context}.training_paired_loss",
    ),
    "upper_speed_mps": _nonnegative_float(
      report.upper_speed_mps,
      f"{context}.upper_speed_mps",
    ),
    "validation_paired_loss": _paired_loss_payload(
      report.validation_paired_loss,
      f"{context}.validation_paired_loss",
    ),
  }


def build_learning_status_payload(
  *,
  finalization: CalibrationLearningFinalization,
  runtime_bundle: RuntimeVehicleBundle,
  drive_baseline: DriveEvidenceBaseline | None,
) -> dict[str, object]:
  """Project one already-persisted calibration finalization for the UI."""
  reports = finalization.learning_result.node_reports
  profile_nodes = runtime_bundle.calibration_seed_profile.nodes
  if not reports or len(reports) != len(profile_nodes):
    raise ValueError("learning reports and runtime speed grid differ")
  for index, report in enumerate(reports):
    if report.node_index != index:
      raise ValueError("learning display node indices must be contiguous from zero")
    if index and report.speed_mps <= reports[index - 1].speed_mps:
      raise ValueError("learning display node speeds must increase")
    if report.speed_mps != profile_nodes[index].speed_mps:
      raise ValueError("learning report speed differs from runtime grid")

  baselines: tuple[DriveNodeBaseline | None, ...]
  if drive_baseline is None:
    baselines = (None,) * len(reports)
  else:
    if len(drive_baseline.nodes) != len(reports):
      raise ValueError("drive baseline and runtime speed grid differ")
    baselines = drive_baseline.nodes

  candidate = finalization.learning_result.candidate_profile
  interpolation_reports = finalization.learning_result.interpolation_reports
  # Every emitted report is a terminal evaluation, including explicit
  # insufficient/rank/validation-regression outcomes. Qualification is a
  # separate fact; conflating the two made a fully processed rejection look
  # like the learner had not run.
  all_nodes_evaluated = all(bool(report.reasons) for report in reports)
  all_nodes_qualified = all(report.qualified for report in reports)
  if interpolation_reports and len(interpolation_reports) != len(reports) - 1:
    raise ValueError("interpolation reports and runtime speed grid differ")
  if not all_nodes_qualified and interpolation_reports:
    raise ValueError("unqualified node grid cannot carry interpolation reports")
  if all_nodes_qualified and len(interpolation_reports) != len(reports) - 1:
    raise ValueError("qualified node grid lacks interpolation reports")
  all_intervals_qualified = (
    len(interpolation_reports) == len(reports) - 1
    and all(report.qualified for report in interpolation_reports)
  )
  all_qualified = all_nodes_qualified and all_intervals_qualified
  if finalization.all_nodes_qualified != all_qualified:
    raise ValueError("final qualification and interval reports disagree")
  candidate_available = candidate is not None
  if candidate_available and not all_qualified:
    raise ValueError("candidate profile exists before full qualification")
  if (
    all_qualified
    and finalization.learning_result.contains_learned_change
    and not candidate_available
  ):
    raise ValueError("qualified learned values lack a candidate profile")
  candidate_hash = finalization.candidate_profile_sha256
  if candidate_available != (candidate_hash is not None):
    raise ValueError("candidate hash and profile availability disagree")

  payload = {
    "all_intervals_qualified": all_intervals_qualified,
    "all_nodes_evaluated": all_nodes_evaluated,
    "all_nodes_qualified": all_nodes_qualified,
    "candidate_profile_available": candidate_available,
    "candidate_profile_revision": None if candidate is None else candidate.revision,
    "candidate_profile_sha256": candidate_hash,
    "evidence_sha256": _sha256(finalization.evidence_sha256, "evidence_sha256"),
    "informational_only": True,
    "interpolation_reports": [
      _interpolation_payload(report) for report in interpolation_reports
    ],
    "last_drive_complete": drive_baseline is not None,
    "manifest_sha256": _sha256(finalization.manifest_sha256, "manifest_sha256"),
    "nodes": [
      _node_payload(report, baselines[index])
      for index, report in enumerate(reports)
    ],
    "runtime_identity_sha256": _sha256(
      runtime_bundle.calibration_identity_sha256,
      "runtime_identity_sha256",
    ),
    "schema_version": LEARNING_STATUS_SCHEMA_VERSION,
    "seed_profile_sha256": _sha256(
      hashlib.sha256(
        runtime_bundle.calibration_seed_profile.to_json().encode("utf-8"),
      ).hexdigest(),
      "seed_profile_sha256",
    ),
    "vehicle_identity": runtime_bundle.vehicle_identity,
  }
  validate_learning_status_payload(payload)
  return payload


def build_learning_status_bytes(
  *,
  finalization: CalibrationLearningFinalization,
  runtime_bundle: RuntimeVehicleBundle,
  drive_baseline: DriveEvidenceBaseline | None,
) -> bytes:
  return _canonical_json_bytes(build_learning_status_payload(
    finalization=finalization,
    runtime_bundle=runtime_bundle,
    drive_baseline=drive_baseline,
  ))


def _optional_nonnegative_float(value: object, name: str) -> None:
  if value is not None:
    _nonnegative_float(value, name)


def _validate_paired_loss(value: object, context: str, *, optional: bool) -> None:
  if value is None:
    if optional:
      return
    raise ValueError(f"{context} must be present")
  if type(value) is not dict or set(value) != _PAIRED_LOSS_KEYS:
    raise ValueError(f"{context} schema does not match")
  route_count = _nonnegative_int(value["route_count"], f"{context}.route_count")
  mean = value["mean_candidate_minus_seed_mse"]
  tolerance = value["numerical_tolerance_mse"]
  uncertainty = value["uncertainty_mse"]
  lower = value["lower_bound_mse"]
  upper = value["upper_bound_mse"]
  if route_count == 0:
    if any(item is not None for item in (mean, tolerance, uncertainty, lower, upper)):
      raise ValueError(f"{context} empty route loss carries values")
    return
  mean_value = _finite_float(mean, f"{context}.mean_candidate_minus_seed_mse")
  _nonnegative_float(tolerance, f"{context}.numerical_tolerance_mse")
  if route_count == 1:
    if any(item is not None for item in (uncertainty, lower, upper)):
      raise ValueError(f"{context} one-route loss invents uncertainty")
    return
  uncertainty_value = _nonnegative_float(
    uncertainty,
    f"{context}.uncertainty_mse",
  )
  lower_value = _finite_float(lower, f"{context}.lower_bound_mse")
  upper_value = _finite_float(upper, f"{context}.upper_bound_mse")
  if lower_value > upper_value or uncertainty_value < 0.0:
    raise ValueError(f"{context} uncertainty bounds are invalid")
  if not math.isclose(
    lower_value,
    mean_value - uncertainty_value,
    rel_tol=1e-12,
    abs_tol=1e-12,
  ) or not math.isclose(
    upper_value,
    mean_value + uncertainty_value,
    rel_tol=1e-12,
    abs_tol=1e-12,
  ):
    raise ValueError(f"{context} uncertainty bounds disagree")


def _validate_fit_diagnostics(value: object, context: str) -> set[str]:
  if type(value) is not list or not value:
    raise ValueError(f"{context} must be a nonempty list")
  expected_models = {model.value for model in CalibrationModelId}
  statuses = {status.value for status in CalibrationFitStatus}
  observed_models: set[str] = set()
  observed_statuses: set[str] = set()
  for index, diagnostic in enumerate(value):
    item_context = f"{context}[{index}]"
    if type(diagnostic) is not dict or set(diagnostic) != _FIT_DIAGNOSTIC_KEYS:
      raise ValueError(f"{item_context} schema does not match")
    model = diagnostic["model"]
    if type(model) is not str or model not in expected_models or model in observed_models:
      raise ValueError(f"{item_context}.model is invalid or duplicated")
    observed_models.add(model)
    if type(diagnostic["status"]) is not str or diagnostic["status"] not in statuses:
      raise ValueError(f"{item_context}.status is invalid")
    status = diagnostic["status"]
    observed_statuses.add(status)
    moving_rank = _nonnegative_int(
      diagnostic["moving_rank"], f"{item_context}.moving_rank"
    )
    moving_count = _nonnegative_int(
      diagnostic["moving_parameter_count"],
      f"{item_context}.moving_parameter_count",
    )
    breakaway_rank = _nonnegative_int(
      diagnostic["breakaway_rank"], f"{item_context}.breakaway_rank"
    )
    breakaway_count = _nonnegative_int(
      diagnostic["breakaway_parameter_count"],
      f"{item_context}.breakaway_parameter_count",
    )
    if moving_rank > moving_count or breakaway_rank > breakaway_count:
      raise ValueError(f"{item_context} rank exceeds parameter count")
    if diagnostic["condition_estimate"] is not None:
      _nonnegative_float(
        diagnostic["condition_estimate"],
        f"{item_context}.condition_estimate",
      )
    full_rank = moving_rank == moving_count and breakaway_rank == breakaway_count
    if status == CalibrationFitStatus.IDENTIFIABLE.value and not full_rank:
      raise ValueError(f"{item_context} identifiable fit lacks full rank")
    if status == CalibrationFitStatus.RANK_DEFICIENT.value and full_rank:
      raise ValueError(f"{item_context} rank-deficient fit has full rank")
  if observed_models != expected_models:
    raise ValueError(f"{context} model family is incomplete")
  return observed_statuses


def validate_learning_status_payload(payload: object) -> dict[str, object]:
  """Strictly validate an existing display snapshot; extra keys fail closed."""
  if type(payload) is not dict or set(payload) != _TOP_LEVEL_KEYS:
    raise ValueError("learning status top-level schema does not match")
  if (
    type(payload["schema_version"]) is not int
    or payload["schema_version"] != LEARNING_STATUS_SCHEMA_VERSION
    or payload["informational_only"] is not True
  ):
    raise ValueError("learning status version/authority marker is invalid")
  if type(payload["vehicle_identity"]) is not str or not payload["vehicle_identity"].strip():
    raise ValueError("learning status vehicle identity is invalid")
  for field in ("runtime_identity_sha256", "seed_profile_sha256", "evidence_sha256", "manifest_sha256"):
    _sha256(payload[field], field)
  if any(
    type(payload[field]) is not bool
    for field in (
      "all_intervals_qualified",
      "all_nodes_evaluated",
      "all_nodes_qualified",
      "candidate_profile_available",
      "last_drive_complete",
    )
  ):
    raise ValueError("learning status booleans are invalid")

  nodes = payload["nodes"]
  if type(nodes) is not list or not nodes:
    raise ValueError("learning status requires a nonempty node grid")
  qualified: list[bool] = []
  previous_speed = -1.0
  drive_values_present: list[bool] = []
  reason_values = {reason.value for reason in CalibrationQualificationReason}
  for index, node in enumerate(nodes):
    context = f"nodes[{index}]"
    if type(node) is not dict or set(node) != _NODE_KEYS:
      raise ValueError(f"{context} schema does not match")
    if node["node_index"] != index or type(node["node_index"]) is not int:
      raise ValueError(f"{context}.node_index is invalid")
    speed = _nonnegative_float(node["speed_mps"], f"{context}.speed_mps")
    if speed <= previous_speed:
      raise ValueError("learning node speeds must strictly increase")
    previous_speed = speed
    for field in (
      "minimum_support_s", "clean_support_s", "validation_support_s",
      "minimum_validation_support_s", "base_support_s", "moving_support_s",
      "breakaway_support_s", "authority_support_s", "authority_fit_support_s",
      "lateral_accel_span_mps2", "lateral_accel_rms_mps2",
      "rack_travel_deg", "applied_torque_span", "confidence",
    ):
      _nonnegative_float(node[field], f"{context}.{field}")
    if _finite_float(node["confidence"], f"{context}.confidence") > 1.0:
      raise ValueError(f"{context}.confidence must not exceed one")
    for field in (
      "supported_sample_count", "training_count", "validation_count",
      "base_sample_count", "moving_sample_count", "moving_training_count",
      "moving_validation_count", "breakaway_sample_count",
      "breakaway_training_count", "breakaway_validation_count",
      "authority_sample_count", "authority_fit_sample_count",
      "authority_training_count", "authority_validation_count",
      "rack_reversals", "lateral_accel_directions",
      "applied_torque_directions",
    ):
      _nonnegative_int(node[field], f"{context}.{field}")
    if not _support_populations_match(
      float(node["clean_support_s"]),
      float(node["base_support_s"]),
      float(node["moving_support_s"]),
      float(node["breakaway_support_s"]),
    ):
      raise ValueError(f"{context} clean support populations disagree")
    if node["supported_sample_count"] != (
      node["base_sample_count"]
      + node["moving_sample_count"]
      + node["breakaway_sample_count"]
    ):
      raise ValueError(f"{context} clean sample populations disagree")
    if (
      float(node["authority_fit_support_s"])
      > float(node["authority_support_s"]) + 1e-12
      or node["authority_fit_sample_count"] > node["authority_sample_count"]
    ):
      raise ValueError(f"{context} authority fit exceeds authority evidence")
    for field in (
      "seed_validation_rms", "candidate_validation_rms",
      "moving_seed_validation_rms", "moving_candidate_validation_rms",
      "breakaway_seed_validation_rms", "breakaway_candidate_validation_rms",
      "authority_seed_validation_rms", "authority_candidate_validation_rms",
    ):
      _optional_nonnegative_float(node[field], f"{context}.{field}")

    delta_fields = (
      "last_drive_clean_support_s", "last_drive_accepted_sample_count",
      "last_drive_base_support_s", "last_drive_base_sample_count",
      "last_drive_moving_support_s", "last_drive_moving_sample_count",
      "last_drive_breakaway_support_s", "last_drive_breakaway_sample_count",
      "last_drive_authority_support_s", "last_drive_authority_sample_count",
      "last_drive_authority_fit_support_s",
      "last_drive_authority_fit_sample_count",
    )
    present = [node[field] is not None for field in delta_fields]
    if len(set(present)) != 1:
      raise ValueError(f"{context} has partial last-drive deltas")
    if present[0]:
      for field in (
        "last_drive_clean_support_s", "last_drive_base_support_s",
        "last_drive_moving_support_s",
        "last_drive_breakaway_support_s", "last_drive_authority_support_s",
        "last_drive_authority_fit_support_s",
      ):
        _nonnegative_float(node[field], f"{context}.{field}")
      for field in (
        "last_drive_accepted_sample_count", "last_drive_base_sample_count",
        "last_drive_moving_sample_count",
        "last_drive_breakaway_sample_count", "last_drive_authority_sample_count",
        "last_drive_authority_fit_sample_count",
      ):
        _nonnegative_int(node[field], f"{context}.{field}")
    drive_values_present.append(present[0])

    if type(node["qualified"]) is not bool:
      raise ValueError(f"{context}.qualified must be boolean")
    qualified.append(node["qualified"])
    evaluation_status = node["evaluation_status"]
    if type(evaluation_status) is not str or evaluation_status not in _EVALUATION_STATUSES:
      raise ValueError(f"{context}.evaluation_status is invalid")
    fit_statuses = _validate_fit_diagnostics(
      node["fit_diagnostics"], f"{context}.fit_diagnostics"
    )
    if evaluation_status == "rank_deficient" and CalibrationFitStatus.RANK_DEFICIENT.value not in fit_statuses:
      raise ValueError(f"{context}.rank-deficient status lacks fit evidence")
    if evaluation_status == "ill_conditioned" and CalibrationFitStatus.ILL_CONDITIONED.value not in fit_statuses:
      raise ValueError(f"{context}.ill-conditioned status lacks fit evidence")
    training_outcome = node["training_outcome"]
    if training_outcome not in (
      None,
      CalibrationQualificationReason.LEARNED.value,
      CalibrationQualificationReason.SEED_RETAINED.value,
    ):
      raise ValueError(f"{context}.training_outcome is invalid")
    _validate_paired_loss(
      node["training_paired_loss"],
      f"{context}.training_paired_loss",
      optional=True,
    )
    _validate_paired_loss(
      node["validation_paired_loss"],
      f"{context}.validation_paired_loss",
      optional=True,
    )
    reasons = node["reasons"]
    if type(reasons) is not list or not reasons or any(type(reason) is not str or not reason for reason in reasons):
      raise ValueError(f"{context}.reasons must be nonempty text values")
    qualified_reason = reasons in (
      [CalibrationQualificationReason.LEARNED.value],
      [CalibrationQualificationReason.SEED_RETAINED.value],
    )
    if node["qualified"] != qualified_reason:
      raise ValueError(f"{context}.qualification reasons disagree")
    if any(reason not in reason_values for reason in reasons):
      raise ValueError(f"{context}.qualification reason is unknown")
    if node["qualified"] and (
      evaluation_status != reasons[0] or training_outcome != reasons[0]
    ):
      raise ValueError(f"{context}.qualified outcome projection disagrees")
    if not node["qualified"] and evaluation_status in ("learned", "seed_retained"):
      raise ValueError(f"{context}.failed node has a qualified status")

    parameters = node["candidate_parameters"]
    if node["qualified"] and parameters is None:
      raise ValueError(f"{context}.qualified node lacks candidate parameters")
    if parameters is not None:
      if type(parameters) is not dict or set(parameters) != _CANDIDATE_PARAMETER_KEYS:
        raise ValueError(f"{context}.candidate_parameters schema differs")
      gain = _positive_float(
        parameters["torque_per_lateral_accel"],
        f"{context}.candidate_parameters.torque_per_lateral_accel",
      )
      del gain
      _finite_float(
        parameters["lateral_accel_offset_correction_mps2"],
        f"{context}.candidate_parameters.lateral_accel_offset_correction_mps2",
      )
      kinetic = _nonnegative_float(
        parameters["kinetic_friction_torque"],
        f"{context}.candidate_parameters.kinetic_friction_torque",
      )
      static = _nonnegative_float(
        parameters["static_breakaway_torque"],
        f"{context}.candidate_parameters.static_breakaway_torque",
      )
      if kinetic > static:
        raise ValueError(f"{context}.candidate_parameters friction ordering is invalid")

  drive_complete = payload["last_drive_complete"]
  if any(drive_values_present) != drive_complete:
    raise ValueError("last-drive completeness and node deltas disagree")
  if len(set(drive_values_present)) != 1:
    raise ValueError("last-drive node deltas cannot be partially present")
  interpolation_reports = payload["interpolation_reports"]
  if type(interpolation_reports) is not list:
    raise ValueError("interpolation reports must be a list")
  interval_qualified: list[bool] = []
  for index, report in enumerate(interpolation_reports):
    context = f"interpolation_reports[{index}]"
    if type(report) is not dict or set(report) != _INTERPOLATION_KEYS:
      raise ValueError(f"{context} schema does not match")
    if type(report["interval_index"]) is not int or report["interval_index"] != index:
      raise ValueError(f"{context}.interval_index is invalid")
    lower_speed = _nonnegative_float(
      report["lower_speed_mps"], f"{context}.lower_speed_mps"
    )
    upper_speed = _nonnegative_float(
      report["upper_speed_mps"], f"{context}.upper_speed_mps"
    )
    if (
      index + 1 >= len(nodes)
      or lower_speed != nodes[index]["speed_mps"]
      or upper_speed != nodes[index + 1]["speed_mps"]
    ):
      raise ValueError(f"{context} speed bounds disagree with node grid")
    if type(report["qualified"]) is not bool:
      raise ValueError(f"{context}.qualified must be boolean")
    interval_qualified.append(report["qualified"])
    reasons = report["reasons"]
    if type(reasons) is not list or not reasons or any(
      type(reason) is not str or reason not in reason_values
      for reason in reasons
    ):
      raise ValueError(f"{context}.reasons are invalid")
    if report["qualified"] != (
      reasons == [CalibrationQualificationReason.QUALIFIED.value]
    ):
      raise ValueError(f"{context}.qualification reasons disagree")
    _validate_paired_loss(
      report["training_paired_loss"],
      f"{context}.training_paired_loss",
      optional=False,
    )
    _validate_paired_loss(
      report["validation_paired_loss"],
      f"{context}.validation_paired_loss",
      optional=False,
    )
  all_nodes_evaluated = all(
    type(node["evaluation_status"]) is str
    and bool(node["evaluation_status"])
    and type(node["reasons"]) is list
    and bool(node["reasons"])
    for node in nodes
  )
  all_nodes_qualified = all(qualified)
  if not all_nodes_qualified and interpolation_reports:
    raise ValueError("unqualified node grid carries interpolation reports")
  if payload["all_nodes_evaluated"] != all_nodes_evaluated:
    raise ValueError("all-node evaluation invariant failed")
  all_intervals_qualified = (
    len(interpolation_reports) == len(nodes) - 1
    and all(interval_qualified)
  )
  if payload["all_intervals_qualified"] != all_intervals_qualified:
    raise ValueError("all-interval qualification invariant failed")
  if payload["all_nodes_qualified"] != all_nodes_qualified:
    raise ValueError("node qualification invariant failed")
  all_qualified = all_nodes_qualified and all_intervals_qualified
  candidate_hash = payload["candidate_profile_sha256"]
  candidate_revision = payload["candidate_profile_revision"]
  candidate_present = candidate_hash is not None
  if candidate_present:
    _sha256(candidate_hash, "candidate_profile_sha256")
    _nonnegative_int(candidate_revision, "candidate_profile_revision")
  elif candidate_revision is not None:
    raise ValueError("candidate revision exists without candidate hash")
  if payload["candidate_profile_available"] != candidate_present:
    raise ValueError("candidate availability and identity disagree")
  if candidate_present and not all_qualified:
    raise ValueError("candidate exists before full qualification")
  if all_qualified and not candidate_present and any(
    node["training_outcome"] == CalibrationQualificationReason.LEARNED.value
    for node in nodes
  ):
    raise ValueError("qualified learned node lacks candidate artifact")
  return payload


def decode_learning_status(value: object) -> dict[str, object]:
  """Decode canonical bytes/text and reject malformed/noncanonical input."""
  if type(value) is bytes:
    encoded = value
  elif type(value) is str:
    encoded = value.encode("utf-8")
  else:
    raise TypeError("learning status must be canonical JSON bytes or text")
  try:
    payload = json.loads(encoded)
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise ValueError("learning status JSON is malformed") from exc
  validate_learning_status_payload(payload)
  if _canonical_json_bytes(payload) != encoded:
    raise ValueError("learning status JSON is not canonical")
  return payload
