"""Strict display-only projection of persisted BLaTv2 learning evidence.

This module is deliberately outside every controller-selection, approval,
fitting, and safety path.  It projects an already-finalized learner artifact
into one small Params value for the offroad UI.  The UI status is never an
authority source: deleting, corrupting, or editing it cannot change learning,
candidate creation, controller selection, or actuation.

The projection is written only after the canonical evidence, optional
candidate, and manifest have persisted successfully.  Floats remain JSON
numbers for the C++ UI, while canonical key ordering and finite-value checks
make identical finalizations byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

from openpilot.selfdrive.controls.lib.blatv2.learner import (
  MIN_VALIDATION_SUPPORT_FRACTION,
  NodeQualificationReport,
  QualificationReason,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_coordinator import (
  LearningFinalization,
  NodeSupportDiagnostic,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  RuntimeVehicleBundle,
)


LEARNING_STATUS_PARAM = "BLaTv2LearningStatus"
LEARNING_STATUS_SCHEMA_VERSION = 1
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
  "applied_torque_span",
  "candidate_parameters",
  "candidate_validation_rms",
  "clean_support_s",
  "confidence",
  "last_drive_accepted_sample_count",
  "last_drive_clean_support_s",
  "lateral_accel_rms_mps2",
  "lateral_accel_span_mps2",
  "minimum_support_s",
  "minimum_validation_support_s",
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
  "rack_damping_per_s",
  "rack_gain_deg_s2_per_torque",
  "torque_per_lateral_accel",
}


def _finite_float(value: object, name: str) -> float:
  if type(value) not in (int, float):
    raise TypeError(f"{name} must be a JSON number")
  numeric = float(value)
  if not math.isfinite(numeric):
    raise ValueError(f"{name} must be finite")
  return 0.0 if numeric == 0.0 else numeric


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
  """Cumulative evidence at a real offroad-to-onroad transition."""

  node_index: int
  speed_mps: float
  clean_support_s: float
  supported_sample_count: int


@dataclass(frozen=True, slots=True)
class DriveEvidenceBaseline:
  """All node-local cumulative counters captured before a drive."""

  nodes: tuple[DriveNodeBaseline, ...]

  @classmethod
  def from_support_diagnostics(
    cls,
    diagnostics: tuple[NodeSupportDiagnostic, ...],
  ) -> DriveEvidenceBaseline:
    if not diagnostics:
      raise ValueError("learning display requires at least one speed node")
    return cls(
      nodes=tuple(
        DriveNodeBaseline(
          node_index=diagnostic.node_index,
          speed_mps=_nonnegative_float(
            diagnostic.speed_mps,
            f"baseline.nodes[{diagnostic.node_index}].speed_mps",
          ),
          clean_support_s=_nonnegative_float(
            diagnostic.clean_support_s,
            f"baseline.nodes[{diagnostic.node_index}].clean_support_s",
          ),
          supported_sample_count=_nonnegative_int(
            diagnostic.supported_sample_count,
            f"baseline.nodes[{diagnostic.node_index}].supported_sample_count",
          ),
        )
        for diagnostic in diagnostics
      ),
    )


def _candidate_parameters(
  report: NodeQualificationReport,
) -> dict[str, float] | None:
  parameters = report.candidate_parameters
  if parameters is None:
    return None
  # Only values independently fit by ProfileLearner are exposed as learned.
  # Seed-carried delay, static friction, and rack-rate resolution are omitted.
  return {
    "kinetic_friction_torque": _nonnegative_float(
      parameters.kinetic_friction_torque,
      "candidate_parameters.kinetic_friction_torque",
    ),
    "rack_damping_per_s": _nonnegative_float(
      parameters.rack_damping_per_s,
      "candidate_parameters.rack_damping_per_s",
    ),
    "rack_gain_deg_s2_per_torque": _nonnegative_float(
      parameters.rack_gain_deg_s2_per_torque,
      "candidate_parameters.rack_gain_deg_s2_per_torque",
    ),
    "torque_per_lateral_accel": _nonnegative_float(
      parameters.torque_per_lateral_accel,
      "candidate_parameters.torque_per_lateral_accel",
    ),
  }


def _node_payload(
  report: NodeQualificationReport,
  baseline: DriveNodeBaseline | None,
) -> dict[str, object]:
  context = f"nodes[{report.node_index}]"
  last_support: float | None = None
  last_count: int | None = None
  if baseline is not None:
    if (
      baseline.node_index != report.node_index
      or baseline.speed_mps != report.speed_mps
    ):
      raise ValueError("drive baseline node grid changed")
    support_delta = report.clean_support_s - baseline.clean_support_s
    count_delta = (
      report.supported_sample_count - baseline.supported_sample_count
    )
    if support_delta < -1e-12 or count_delta < 0:
      raise ValueError("cumulative learning evidence moved backwards")
    last_support = max(0.0, _finite_float(
      support_delta,
      f"{context}.last_drive_clean_support_s",
    ))
    last_count = count_delta

  minimum_support = _nonnegative_float(
    report.minimum_support_s,
    f"{context}.minimum_support_s",
  )
  return {
    "applied_torque_span": _nonnegative_float(
      report.applied_torque_span,
      f"{context}.applied_torque_span",
    ),
    "candidate_parameters": _candidate_parameters(report),
    "candidate_validation_rms": (
      None
      if report.candidate_validation_rms is None
      else _nonnegative_float(
        report.candidate_validation_rms,
        f"{context}.candidate_validation_rms",
      )
    ),
    "clean_support_s": _nonnegative_float(
      report.clean_support_s,
      f"{context}.clean_support_s",
    ),
    "confidence": _nonnegative_float(
      report.confidence,
      f"{context}.confidence",
    ),
    "last_drive_accepted_sample_count": last_count,
    "last_drive_clean_support_s": last_support,
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
    "node_index": report.node_index,
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
    "seed_validation_rms": (
      None
      if report.seed_validation_rms is None
      else _nonnegative_float(
        report.seed_validation_rms,
        f"{context}.seed_validation_rms",
      )
    ),
    "speed_mps": _nonnegative_float(
      report.speed_mps,
      f"{context}.speed_mps",
    ),
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
  finalization: LearningFinalization,
  runtime_bundle: RuntimeVehicleBundle,
  drive_baseline: DriveEvidenceBaseline | None,
) -> dict[str, object]:
  """Project one already-persisted finalization into the UI-only schema."""
  reports = finalization.learning_result.node_reports
  profile_nodes = runtime_bundle.seed_profile.nodes
  if not reports or len(reports) != len(profile_nodes):
    raise ValueError("learning reports and runtime speed grid differ")
  for index, report in enumerate(reports):
    if report.node_index != index:
      raise ValueError(
        "learning display node indices must be contiguous from zero",
      )
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
  all_qualified = finalization.all_nodes_qualified
  if all_qualified != all(report.qualified for report in reports):
    raise ValueError("candidate identity and node qualification disagree")
  if all_qualified != (candidate is not None):
    raise ValueError("candidate profile and finalization disagree")
  candidate_hash = finalization.candidate_profile_sha256
  if all_qualified != (candidate_hash is not None):
    raise ValueError("candidate identity and qualification disagree")

  payload = {
    "all_nodes_qualified": all_qualified,
    "candidate_profile_revision": (
      None if candidate is None else candidate.revision
    ),
    "candidate_profile_sha256": candidate_hash,
    "evidence_sha256": _sha256(
      finalization.evidence_sha256,
      "evidence_sha256",
    ),
    "informational_only": True,
    "last_drive_complete": drive_baseline is not None,
    "manifest_sha256": _sha256(
      finalization.manifest_sha256,
      "manifest_sha256",
    ),
    "nodes": [
      _node_payload(report, baselines[index])
      for index, report in enumerate(reports)
    ],
    "runtime_identity_sha256": _sha256(
      runtime_bundle.identity_sha256,
      "runtime_identity_sha256",
    ),
    "schema_version": LEARNING_STATUS_SCHEMA_VERSION,
    "seed_profile_sha256": _sha256(
      hashlib.sha256(
        runtime_bundle.seed_profile.to_json().encode("utf-8"),
      ).hexdigest(),
      "seed_profile_sha256",
    ),
    "vehicle_identity": runtime_bundle.vehicle_identity,
  }
  # Strict validation is applied to our own output so schema changes cannot
  # accidentally loosen the informational/authority boundary.
  validate_learning_status_payload(payload)
  return payload


def build_learning_status_bytes(
  *,
  finalization: LearningFinalization,
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
  if (
    type(payload["vehicle_identity"]) is not str
    or not payload["vehicle_identity"].strip()
  ):
    raise ValueError("learning status vehicle identity is invalid")
  for field in (
    "runtime_identity_sha256",
    "seed_profile_sha256",
    "evidence_sha256",
    "manifest_sha256",
  ):
    _sha256(payload[field], field)
  if (
    type(payload["all_nodes_qualified"]) is not bool
    or type(payload["last_drive_complete"]) is not bool
  ):
    raise ValueError("learning status booleans are invalid")

  nodes = payload["nodes"]
  if type(nodes) is not list or not nodes:
    raise ValueError("learning status requires a nonempty node grid")
  qualified: list[bool] = []
  previous_speed = -1.0
  drive_values_present: list[bool] = []
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
      "minimum_support_s",
      "clean_support_s",
      "validation_support_s",
      "minimum_validation_support_s",
      "lateral_accel_span_mps2",
      "lateral_accel_rms_mps2",
      "rack_travel_deg",
      "applied_torque_span",
      "confidence",
    ):
      _nonnegative_float(node[field], f"{context}.{field}")
    if _finite_float(node["confidence"], f"{context}.confidence") > 1.0:
      raise ValueError(f"{context}.confidence must not exceed one")
    for field in (
      "supported_sample_count",
      "training_count",
      "validation_count",
      "rack_reversals",
    ):
      _nonnegative_int(node[field], f"{context}.{field}")
    _optional_nonnegative_float(
      node["seed_validation_rms"],
      f"{context}.seed_validation_rms",
    )
    _optional_nonnegative_float(
      node["candidate_validation_rms"],
      f"{context}.candidate_validation_rms",
    )
    support_delta = node["last_drive_clean_support_s"]
    count_delta = node["last_drive_accepted_sample_count"]
    if (support_delta is None) != (count_delta is None):
      raise ValueError(f"{context} has a partial last-drive delta")
    if support_delta is not None:
      _nonnegative_float(
        support_delta,
        f"{context}.last_drive_clean_support_s",
      )
      _nonnegative_int(
        count_delta,
        f"{context}.last_drive_accepted_sample_count",
      )
    drive_values_present.append(support_delta is not None)
    if type(node["qualified"]) is not bool:
      raise ValueError(f"{context}.qualified must be boolean")
    qualified.append(node["qualified"])
    reasons = node["reasons"]
    if (
      type(reasons) is not list
      or not reasons
      or any(type(reason) is not str or not reason for reason in reasons)
    ):
      raise ValueError(f"{context}.reasons must be nonempty text values")
    qualified_reason = QualificationReason.QUALIFIED.value
    if node["qualified"] != (reasons == [qualified_reason]):
      raise ValueError(f"{context}.qualification reasons disagree")
    known_reasons = {reason.value for reason in QualificationReason}
    if any(reason not in known_reasons for reason in reasons):
      raise ValueError(f"{context}.qualification reason is unknown")
    parameters = node["candidate_parameters"]
    if parameters is not None:
      if (
        type(parameters) is not dict
        or set(parameters) != _CANDIDATE_PARAMETER_KEYS
      ):
        raise ValueError(f"{context}.candidate_parameters schema differs")
      for field in _CANDIDATE_PARAMETER_KEYS:
        _nonnegative_float(
          parameters[field],
          f"{context}.candidate_parameters.{field}",
        )

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
  """Decode canonical bytes/text and reject malformed or noncanonical input."""
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
