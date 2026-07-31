"""Strict display-only projection of observable BLaTv2 calibration.

This Params value is an informational projection of already-finalized
calibration evidence.  It is outside controller selection, approval, fitting,
and actuation: deleting or corrupting it cannot change which controller runs.

Schema 2 deliberately rejects the retired physical rack-fit vocabulary.  The
only candidate values it exposes are the four observable inverse-torque
calibration values, while independent base, moving, breakaway, and authority
populations remain visible for audit and UI progress reporting.
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
  CalibrationNodeQualificationReport,
  CalibrationQualificationReason,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  RuntimeVehicleBundle,
)


LEARNING_STATUS_PARAM = "BLaTv2LearningStatus"
LEARNING_STATUS_SCHEMA_VERSION = 2
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_KEYS = {
  "all_nodes_qualified",
  "candidate_profile_revision",
  "candidate_profile_sha256",
  "evidence_sha256",
  "informational_only",
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
  "validation_count",
  "validation_support_s",
}
_CANDIDATE_PARAMETER_KEYS = {
  "kinetic_friction_torque",
  "lateral_accel_offset_correction_mps2",
  "static_breakaway_torque",
  "torque_per_lateral_accel",
}


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
      clean_parts = (
        node.base_support_s + node.moving_support_s + node.breakaway_support_s
      )
      if not math.isclose(
        node.clean_support_s,
        clean_parts,
        rel_tol=0.0,
        abs_tol=1e-12,
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
    "validation_count": _nonnegative_int(
      report.validation_count,
      f"{context}.validation_count",
    ),
    "validation_support_s": _nonnegative_float(
      report.validation_support_s,
      f"{context}.validation_support_s",
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
  all_qualified = all(report.qualified for report in reports)
  if finalization.all_nodes_qualified != all_qualified:
    raise ValueError("candidate identity and node qualification disagree")
  if all_qualified != (candidate is not None):
    raise ValueError("candidate profile and node qualification disagree")
  candidate_hash = finalization.candidate_profile_sha256
  if all_qualified != (candidate_hash is not None):
    raise ValueError("candidate hash and node qualification disagree")

  payload = {
    "all_nodes_qualified": all_qualified,
    "candidate_profile_revision": None if candidate is None else candidate.revision,
    "candidate_profile_sha256": candidate_hash,
    "evidence_sha256": _sha256(finalization.evidence_sha256, "evidence_sha256"),
    "informational_only": True,
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
  if type(payload["all_nodes_qualified"]) is not bool or type(payload["last_drive_complete"]) is not bool:
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
    clean_parts = (
      float(node["base_support_s"])
      + float(node["moving_support_s"])
      + float(node["breakaway_support_s"])
    )
    if not math.isclose(
      float(node["clean_support_s"]),
      clean_parts,
      rel_tol=0.0,
      abs_tol=1e-12,
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
    reasons = node["reasons"]
    if type(reasons) is not list or not reasons or any(type(reason) is not str or not reason for reason in reasons):
      raise ValueError(f"{context}.reasons must be nonempty text values")
    if node["qualified"] != (reasons == [CalibrationQualificationReason.QUALIFIED.value]):
      raise ValueError(f"{context}.qualification reasons disagree")
    if any(reason not in reason_values for reason in reasons):
      raise ValueError(f"{context}.qualification reason is unknown")

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
  all_qualified = payload["all_nodes_qualified"]
  if all(qualified) != all_qualified:
    raise ValueError("all-node qualification invariant failed")
  candidate_hash = payload["candidate_profile_sha256"]
  candidate_revision = payload["candidate_profile_revision"]
  candidate_present = candidate_hash is not None
  if candidate_present:
    _sha256(candidate_hash, "candidate_profile_sha256")
    _nonnegative_int(candidate_revision, "candidate_profile_revision")
  elif candidate_revision is not None:
    raise ValueError("candidate revision exists without candidate hash")
  if candidate_present != all_qualified:
    raise ValueError("candidate identity and qualification disagree")
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
