"""Strict, display-only decoding for the BLaTv2 learning dashboard.

The UI must never infer controller activation from learner progress. Learning
and lifecycle snapshots are independent, rebuildable Params caches produced by
the owning daemons. This module contains no controller imports and performs no
file, route, evidence, fit, approval, or actuation work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import re


LEARNING_STATUS_SCHEMA_VERSION = 1
LIFECYCLE_STATUS_SCHEMA_VERSION = 1
LEARNING_OPERATION_STATUS_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_OPERATION_ID_RE = re.compile(r"[0-9a-f]{32}")
_TOP_LEVEL_KEYS = frozenset((
  "schema_version",
  "informational_only",
  "vehicle_identity",
  "runtime_identity_sha256",
  "seed_profile_sha256",
  "evidence_sha256",
  "manifest_sha256",
  "all_nodes_qualified",
  "candidate_profile_sha256",
  "candidate_profile_revision",
  "last_drive_complete",
  "nodes",
))
_NODE_KEYS = frozenset((
  "node_index",
  "speed_mps",
  "minimum_support_s",
  "clean_support_s",
  "last_drive_clean_support_s",
  "supported_sample_count",
  "last_drive_accepted_sample_count",
  "training_count",
  "validation_count",
  "validation_support_s",
  "minimum_validation_support_s",
  "lateral_accel_span_mps2",
  "lateral_accel_rms_mps2",
  "rack_travel_deg",
  "applied_torque_span",
  "rack_reversals",
  "seed_validation_rms",
  "candidate_validation_rms",
  "confidence",
  "qualified",
  "reasons",
  "candidate_parameters",
))
_PARAMETER_KEYS = frozenset((
  "torque_per_lateral_accel",
  "rack_gain_deg_s2_per_torque",
  "rack_damping_per_s",
  "kinetic_friction_torque",
))
_LIFECYCLE_KEYS = frozenset((
  "schema_version",
  "informational_only",
  "vehicle_identity",
  "runtime_identity_sha256",
  "source_openpilot_commit",
  "opendbc_commit",
  "activation_state_sha256",
  "diagnostic",
  "controller_state",
  "effective_controller",
  "production_envelope_verified",
  "active_profile",
  "staged_profile",
  "rejected_profile_count",
))
_OPERATION_KEYS = frozenset((
  "schema_version",
  "informational_only",
  "state",
  "diagnostic",
  "operation_id",
  "sequence",
  "started_mono_ns",
  "updated_mono_ns",
  "terminal",
  "vehicle_identity",
  "runtime_identity_sha256",
  "current_route_identity",
  "current_route_index",
  "total_route_count",
  "last_route_identity",
  "accepted_sample_count",
  "rejected_sample_count",
  "retry_count",
  "evidence_sha256",
  "ledger_sha256",
))
_PROFILE_IDENTITY_KEYS = frozenset((
  "artifact_sha256",
  "profile_sha256",
  "profile_revision",
))
_REASONS = frozenset((
  "qualified",
  "insufficient_support",
  "insufficient_validation",
  "insufficient_excitation",
  "singular_fit",
  "invalid_parameters",
  "validation_regression",
))
_REASON_LABELS = {
  "qualified": "Qualified",
  "insufficient_support": "Collecting clean driving",
  "insufficient_validation": "Needs held-out validation",
  "insufficient_excitation": "Needs more steering variety",
  "singular_fit": "Fit not identifiable",
  "invalid_parameters": "Rejected: invalid fit",
  "validation_regression": "Rejected: validation regressed",
}
_REASON_PRIORITY = (
  "insufficient_support",
  "insufficient_validation",
  "insufficient_excitation",
  "invalid_parameters",
  "validation_regression",
  "singular_fit",
  "qualified",
)
_CONTROLLER_STATES = frozenset((
  "stock",
  "staged",
  "provisional",
  "approved",
  "rollback_pending",
  "unavailable",
))
_EFFECTIVE_CONTROLLERS = frozenset(("stock", "modular"))
_ARTIFACT_DIAGNOSTICS = frozenset((
  "ok",
  "absent",
  "param_read_error",
  "malformed",
  "profile_hash_mismatch",
  "policy_hash_mismatch",
  "unqualified_profile",
  "provisional_policy",
  "vehicle_mismatch",
  "runtime_vehicle_mismatch",
  "source_commit_mismatch",
  "opendbc_commit_mismatch",
  "unverified_actuation_envelope",
  "gate_failed",
  "state_invalid",
  "state_stale_build",
))
_OPERATION_STATE_DIAGNOSTICS = {
  "preparing": frozenset(("waiting_for_car_params", "restoring_runtime")),
  "ready_no_evidence": frozenset(("ready_for_first_drive",)),
  "collecting": frozenset(("collecting_current_drive",)),
  "finalizing": frozenset((
    "finalizing_drive",
    "verifying_backfill",
    "publishing_backfill",
  )),
  "retry_pending": frozenset(("persist_retry_pending",)),
  "backfilling": frozenset(("scanning_routes", "replaying_route")),
  "idle": frozenset((
    "evidence_ready",
    "backfill_complete",
    "backfill_complete_late_older_skipped",
    "backfill_complete_with_rejections",
  )),
  "drive_skipped_identity_mismatch": frozenset(("car_params_identity_mismatch",)),
  "failed": frozenset((
    "runtime_restore_failed",
    "backfill_reader_unavailable",
    "backfill_route_incompatible",
    "backfill_corrupt_log",
    "backfill_nondeterministic",
    "backfill_publish_failed",
    "backfill_untracked_evidence",
    "backfill_no_complete_routes",
    "unexpected_error",
  )),
}
_TERMINAL_OPERATION_STATES = frozenset((
  "ready_no_evidence",
  "idle",
  "drive_skipped_identity_mismatch",
  "failed",
))
_OPERATION_DIAGNOSTIC_LABELS = {
  "waiting_for_car_params": "Waiting for vehicle configuration",
  "restoring_runtime": "Restoring the prepared learning runtime",
  "ready_for_first_drive": "Ready to collect the first drive",
  "collecting_current_drive": "Collecting clean evidence from this drive",
  "finalizing_drive": "Validating and saving the completed drive",
  "verifying_backfill": "Verifying replayed route evidence",
  "publishing_backfill": "Publishing the rebuilt evidence snapshot",
  "persist_retry_pending": "Saving failed; a safe retry is pending",
  "scanning_routes": "Scanning compatible routes on device",
  "replaying_route": "Replaying a compatible historical route",
  "evidence_ready": "Validated learning evidence is ready",
  "backfill_complete": "Historical route processing is complete",
  "backfill_complete_late_older_skipped": ("Backfill complete; late older routes were safely skipped"),
  "backfill_complete_with_rejections": ("Backfill complete; incompatible routes were rejected"),
  "car_params_identity_mismatch": ("Vehicle configuration changed; prepared for the next drive"),
  "runtime_restore_failed": "Prepared learner runtime could not be restored",
  "backfill_reader_unavailable": "Historical route reader is unavailable",
  "backfill_route_incompatible": "A historical route is incompatible",
  "backfill_corrupt_log": "A historical route log is corrupt",
  "backfill_nondeterministic": "Historical replay did not reproduce exactly",
  "backfill_publish_failed": "Rebuilt evidence could not be published",
  "backfill_untracked_evidence": ("Backfill unavailable: stored evidence has no route ledger"),
  "backfill_no_complete_routes": "No complete compatible routes were found",
  "unexpected_error": "The learner reported an unexpected error",
}


class LearningStatusError(ValueError):
  """A stable fail-closed reason for an unavailable learning display."""

  def __init__(self, code: str, message: str):
    super().__init__(message)
    self.code = code


@dataclass(frozen=True, slots=True)
class CandidateParameters:
  torque_per_lateral_accel: float
  rack_gain_deg_s2_per_torque: float
  rack_damping_per_s: float
  kinetic_friction_torque: float


@dataclass(frozen=True, slots=True)
class LearningNodeStatus:
  node_index: int
  speed_mps: float
  minimum_support_s: float
  clean_support_s: float
  last_drive_clean_support_s: float | None
  supported_sample_count: int
  last_drive_accepted_sample_count: int | None
  training_count: int
  validation_count: int
  validation_support_s: float
  minimum_validation_support_s: float
  lateral_accel_span_mps2: float
  lateral_accel_rms_mps2: float
  rack_travel_deg: float
  applied_torque_span: float
  rack_reversals: int
  seed_validation_rms: float | None
  candidate_validation_rms: float | None
  confidence: float
  qualified: bool
  reasons: tuple[str, ...]
  candidate_parameters: CandidateParameters | None

  @property
  def support_fraction(self) -> float:
    if self.minimum_support_s <= 0.0:
      return 0.0
    return max(0.0, min(1.0, self.clean_support_s / self.minimum_support_s))

  @property
  def validation_fraction(self) -> float:
    if self.minimum_validation_support_s <= 0.0:
      return 0.0
    return max(
      0.0,
      min(1.0, self.validation_support_s / self.minimum_validation_support_s),
    )

  @property
  def primary_reason(self) -> str:
    # Sparse evidence can make the provisional normal equations singular or
    # physically invalid long before the node has enough coverage to judge a
    # fit. Collection blockers therefore take display priority; a red fit
    # failure is meaningful only after time, validation, and excitation pass.
    for reason in _REASON_PRIORITY:
      if reason in self.reasons:
        return reason
    return "insufficient_support"

  @property
  def collection_complete(self) -> bool:
    return not any(
      reason in self.reasons
      for reason in (
        "insufficient_support",
        "insufficient_validation",
        "insufficient_excitation",
      )
    )


@dataclass(frozen=True, slots=True)
class LearningStatus:
  vehicle_identity: str
  runtime_identity_sha256: str
  seed_profile_sha256: str
  evidence_sha256: str
  manifest_sha256: str
  all_nodes_qualified: bool
  candidate_profile_sha256: str | None
  candidate_profile_revision: int | None
  last_drive_complete: bool
  nodes: tuple[LearningNodeStatus, ...]

  @property
  def qualified_node_count(self) -> int:
    return sum(node.qualified for node in self.nodes)


@dataclass(frozen=True, slots=True)
class LearningOperationStatus:
  """Display-only state of learner collection, finalization, and backfill."""

  state: str
  diagnostic: str
  operation_id: str
  sequence: int
  started_mono_ns: int
  updated_mono_ns: int
  terminal: bool
  vehicle_identity: str | None
  runtime_identity_sha256: str | None
  current_route_identity: str | None
  current_route_index: int | None
  total_route_count: int | None
  last_route_identity: str | None
  accepted_sample_count: int
  rejected_sample_count: int
  retry_count: int
  evidence_sha256: str | None
  ledger_sha256: str | None

  @property
  def active(self) -> bool:
    return not self.terminal


@dataclass(frozen=True, slots=True)
class OperationPresentation:
  """Pure presentation model shared by both dashboard pages."""

  title: str
  detail: str
  tone: str
  show_banner: bool


@dataclass(frozen=True, slots=True)
class ProfileIdentity:
  artifact_sha256: str
  profile_sha256: str
  profile_revision: int


@dataclass(frozen=True, slots=True)
class LifecycleStatus:
  vehicle_identity: str
  runtime_identity_sha256: str
  source_openpilot_commit: str
  opendbc_commit: str
  activation_state_sha256: str | None
  diagnostic: str
  controller_state: str
  effective_controller: str
  production_envelope_verified: bool
  active_profile: ProfileIdentity | None
  staged_profile: ProfileIdentity | None
  rejected_profile_count: int

  @property
  def badge(self) -> str:
    return {
      "stock": "STOCK ACTIVE",
      "staged": "STOCK ACTIVE",
      "provisional": "BLATV2 PROVISIONAL",
      "approved": "BLATV2 APPROVED",
      "rollback_pending": "ROLLBACK PENDING · STOCK ACTIVE",
      "unavailable": "CONTROLLER STATUS UNAVAILABLE",
    }[self.controller_state]

  @property
  def lifecycle_position(self) -> int:
    return {
      "stock": 0,
      "staged": 2,
      "provisional": 3,
      "approved": 4,
      "rollback_pending": 0,
      "unavailable": -1,
    }[self.controller_state]


@dataclass(frozen=True, slots=True)
class GridCell:
  x: float
  y: float
  width: float
  height: float


def _exact_object(
  value: object,
  keys: frozenset[str],
  field: str,
) -> dict[str, object]:
  if type(value) is not dict or set(value) != keys:
    raise LearningStatusError("malformed", f"{field} keys do not match schema")
  return value


def select_value_provider(
  default_provider: Callable[[], bool],
  supplied_provider: Callable[[], bool] | None,
) -> Callable[[], bool]:
  """Select a callback without truth-testing the callback object itself."""
  return default_provider if supplied_provider is None else supplied_provider


def _bool(value: object, field: str) -> bool:
  if type(value) is not bool:
    raise LearningStatusError("malformed", f"{field} must be boolean")
  return value


def _integer(value: object, field: str, *, nullable: bool = False) -> int | None:
  if nullable and value is None:
    return None
  if type(value) is not int or value < 0:
    raise LearningStatusError(
      "malformed",
      f"{field} must be a non-negative integer",
    )
  return value


def _number(value: object, field: str, *, nullable: bool = False) -> float | None:
  if nullable and value is None:
    return None
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise LearningStatusError("malformed", f"{field} must be numeric")
  result = float(value)
  if not math.isfinite(result) or result < 0.0:
    raise LearningStatusError(
      "malformed",
      f"{field} must be finite and non-negative",
    )
  return result


def _text(value: object, field: str) -> str:
  if type(value) is not str or not value.strip():
    raise LearningStatusError("malformed", f"{field} must be non-empty text")
  return value.strip()


def _nullable_text(value: object, field: str) -> str | None:
  if value is None:
    return None
  result = _text(value, field)
  if len(result) > 256:
    raise LearningStatusError("malformed", f"{field} exceeds 256 characters")
  return result


def _sha256(value: object, field: str, *, nullable: bool = False) -> str | None:
  if nullable and value is None:
    return None
  result = _text(value, field)
  if _SHA256_RE.fullmatch(result) is None:
    raise LearningStatusError("malformed", f"{field} must be lowercase SHA-256")
  return result


def _operation_id(value: object, field: str) -> str:
  result = _text(value, field)
  if _OPERATION_ID_RE.fullmatch(result) is None:
    raise LearningStatusError(
      "malformed",
      f"{field} must be 32 lowercase hexadecimal characters",
    )
  return result


def _commit(value: object, field: str) -> str:
  result = _text(value, field)
  if _GIT_COMMIT_RE.fullmatch(result) is None:
    raise LearningStatusError("malformed", f"{field} must be a full Git commit")
  return result


def _candidate_parameters(
  value: object,
  field: str,
) -> CandidateParameters | None:
  if value is None:
    return None
  payload = _exact_object(value, _PARAMETER_KEYS, field)
  return CandidateParameters(
    torque_per_lateral_accel=_number(
      payload["torque_per_lateral_accel"],
      f"{field}.torque_per_lateral_accel",
    ),
    rack_gain_deg_s2_per_torque=_number(
      payload["rack_gain_deg_s2_per_torque"],
      f"{field}.rack_gain_deg_s2_per_torque",
    ),
    rack_damping_per_s=_number(
      payload["rack_damping_per_s"],
      f"{field}.rack_damping_per_s",
    ),
    kinetic_friction_torque=_number(
      payload["kinetic_friction_torque"],
      f"{field}.kinetic_friction_torque",
    ),
  )


def _node(value: object, position: int) -> LearningNodeStatus:
  field = f"nodes[{position}]"
  payload = _exact_object(value, _NODE_KEYS, field)
  index = _integer(payload["node_index"], f"{field}.node_index")
  if index != position:
    raise LearningStatusError("malformed", "node indices must be contiguous")

  raw_reasons = payload["reasons"]
  if type(raw_reasons) is not list or not raw_reasons:
    raise LearningStatusError("malformed", f"{field}.reasons must be non-empty")
  reasons: list[str] = []
  for reason_position, raw_reason in enumerate(raw_reasons):
    reason = _text(raw_reason, f"{field}.reasons[{reason_position}]")
    if reason not in _REASONS or reason in reasons:
      raise LearningStatusError(
        "malformed",
        f"{field}.reasons contains an unknown or duplicate reason",
      )
    reasons.append(reason)

  qualified = _bool(payload["qualified"], f"{field}.qualified")
  if qualified != (tuple(reasons) == ("qualified",)):
    raise LearningStatusError(
      "malformed",
      f"{field} qualified flag and reasons disagree",
    )

  confidence = _number(payload["confidence"], f"{field}.confidence")
  if confidence > 1.0:
    raise LearningStatusError("malformed", f"{field}.confidence exceeds one")

  return LearningNodeStatus(
    node_index=index,
    speed_mps=_number(payload["speed_mps"], f"{field}.speed_mps"),
    minimum_support_s=_number(
      payload["minimum_support_s"],
      f"{field}.minimum_support_s",
    ),
    clean_support_s=_number(
      payload["clean_support_s"],
      f"{field}.clean_support_s",
    ),
    last_drive_clean_support_s=_number(
      payload["last_drive_clean_support_s"],
      f"{field}.last_drive_clean_support_s",
      nullable=True,
    ),
    supported_sample_count=_integer(
      payload["supported_sample_count"],
      f"{field}.supported_sample_count",
    ),
    last_drive_accepted_sample_count=_integer(
      payload["last_drive_accepted_sample_count"],
      f"{field}.last_drive_accepted_sample_count",
      nullable=True,
    ),
    training_count=_integer(
      payload["training_count"],
      f"{field}.training_count",
    ),
    validation_count=_integer(
      payload["validation_count"],
      f"{field}.validation_count",
    ),
    validation_support_s=_number(
      payload["validation_support_s"],
      f"{field}.validation_support_s",
    ),
    minimum_validation_support_s=_number(
      payload["minimum_validation_support_s"],
      f"{field}.minimum_validation_support_s",
    ),
    lateral_accel_span_mps2=_number(
      payload["lateral_accel_span_mps2"],
      f"{field}.lateral_accel_span_mps2",
    ),
    lateral_accel_rms_mps2=_number(
      payload["lateral_accel_rms_mps2"],
      f"{field}.lateral_accel_rms_mps2",
    ),
    rack_travel_deg=_number(
      payload["rack_travel_deg"],
      f"{field}.rack_travel_deg",
    ),
    applied_torque_span=_number(
      payload["applied_torque_span"],
      f"{field}.applied_torque_span",
    ),
    rack_reversals=_integer(
      payload["rack_reversals"],
      f"{field}.rack_reversals",
    ),
    seed_validation_rms=_number(
      payload["seed_validation_rms"],
      f"{field}.seed_validation_rms",
      nullable=True,
    ),
    candidate_validation_rms=_number(
      payload["candidate_validation_rms"],
      f"{field}.candidate_validation_rms",
      nullable=True,
    ),
    confidence=confidence,
    qualified=qualified,
    reasons=tuple(reasons),
    candidate_parameters=_candidate_parameters(
      payload["candidate_parameters"],
      f"{field}.candidate_parameters",
    ),
  )


def parse_learning_status(
  raw: object,
  *,
  expected_vehicle_identity: str | None,
) -> LearningStatus:
  """Decode the exact current display schema or fail closed.

  ``expected_vehicle_identity`` comes from the UI's already-decoded current
  CarParams. A missing expected identity is unavailable rather than permission
  to display a potentially stale snapshot from another vehicle.
  """
  if raw is None:
    raise LearningStatusError("absent", "Learning data is not available yet")
  if expected_vehicle_identity is None or not expected_vehicle_identity.strip():
    raise LearningStatusError(
      "vehicle_unavailable",
      "Current vehicle identity is unavailable",
    )
  if type(raw) is not dict:
    raise LearningStatusError(
      "malformed",
      "learning status must be a Params JSON object",
    )
  if (
    type(raw.get("schema_version")) is not int
    or raw["schema_version"] != LEARNING_STATUS_SCHEMA_VERSION
  ):
    raise LearningStatusError(
      "schema_mismatch",
      "Learning snapshot version is not supported",
    )
  data = _exact_object(raw, _TOP_LEVEL_KEYS, "learning status")
  if data["informational_only"] is not True:
    raise LearningStatusError(
      "malformed",
      "Learning snapshot is not marked display-only",
    )

  vehicle_identity = _text(data["vehicle_identity"], "vehicle_identity")
  if vehicle_identity != expected_vehicle_identity.strip():
    raise LearningStatusError(
      "wrong_vehicle",
      "Learning data belongs to a different vehicle",
    )

  raw_nodes = data["nodes"]
  if type(raw_nodes) is not list or not raw_nodes:
    raise LearningStatusError("malformed", "nodes must be a non-empty list")
  nodes = tuple(_node(value, index) for index, value in enumerate(raw_nodes))
  if any(
    right.speed_mps <= left.speed_mps
    for left, right in zip(nodes, nodes[1:], strict=False)
  ):
    raise LearningStatusError(
      "malformed",
      "node speeds must be strictly increasing",
    )

  last_drive_complete = _bool(
    data["last_drive_complete"],
    "last_drive_complete",
  )
  for node in nodes:
    node_complete = (
      node.last_drive_clean_support_s is not None
      and node.last_drive_accepted_sample_count is not None
    )
    node_empty = (
      node.last_drive_clean_support_s is None
      and node.last_drive_accepted_sample_count is None
    )
    if (last_drive_complete and not node_complete) or (
      not last_drive_complete and not node_empty
    ):
      raise LearningStatusError(
        "malformed",
        "last-drive completeness and node deltas disagree",
      )

  all_nodes_qualified = _bool(
    data["all_nodes_qualified"],
    "all_nodes_qualified",
  )
  if all_nodes_qualified != all(node.qualified for node in nodes):
    raise LearningStatusError(
      "malformed",
      "all_nodes_qualified disagrees with node reports",
    )
  candidate_sha = _sha256(
    data["candidate_profile_sha256"],
    "candidate_profile_sha256",
    nullable=True,
  )
  candidate_revision = _integer(
    data["candidate_profile_revision"],
    "candidate_profile_revision",
    nullable=True,
  )
  candidate_complete = candidate_sha is not None and candidate_revision is not None
  candidate_empty = candidate_sha is None and candidate_revision is None
  if (
    (all_nodes_qualified and not candidate_complete)
    or (not all_nodes_qualified and not candidate_empty)
  ):
    raise LearningStatusError(
      "malformed",
      "candidate identity and qualification state disagree",
    )

  return LearningStatus(
    vehicle_identity=vehicle_identity,
    runtime_identity_sha256=_sha256(
      data["runtime_identity_sha256"],
      "runtime_identity_sha256",
    ),
    seed_profile_sha256=_sha256(
      data["seed_profile_sha256"],
      "seed_profile_sha256",
    ),
    evidence_sha256=_sha256(
      data["evidence_sha256"],
      "evidence_sha256",
    ),
    manifest_sha256=_sha256(
      data["manifest_sha256"],
      "manifest_sha256",
    ),
    all_nodes_qualified=all_nodes_qualified,
    candidate_profile_sha256=candidate_sha,
    candidate_profile_revision=candidate_revision,
    last_drive_complete=last_drive_complete,
    nodes=nodes,
  )


def parse_learning_operation_status(
  raw: object,
  *,
  expected_vehicle_identity: str | None,
  expected_runtime_identity_sha256: str | None,
  now_mono_ns: int,
) -> LearningOperationStatus:
  """Decode the learner's operational projection without inferring history.

  The status is rebuildable and display-only. Its monotonic timestamps share
  the current manager boot because the Params key clears on manager start.
  There is deliberately no UI-defined age timeout: the backend owns
  transitions, retry, and failure. A timestamp from a future monotonic epoch
  is rejected as stale rather than displayed indefinitely.
  """
  if raw is None:
    raise LearningStatusError(
      "operation_absent",
      "Learner operation status has not been published",
    )
  if type(now_mono_ns) is not int or now_mono_ns < 0:
    raise ValueError("now_mono_ns must be a non-negative integer")
  if type(raw) is not dict:
    raise LearningStatusError(
      "malformed",
      "learner operation status must be a Params JSON object",
    )
  if (
    type(raw.get("schema_version")) is not int
    or raw["schema_version"] != LEARNING_OPERATION_STATUS_SCHEMA_VERSION
  ):
    raise LearningStatusError(
      "schema_mismatch",
      "Learner operation status version is not supported",
    )
  data = _exact_object(raw, _OPERATION_KEYS, "learner operation status")
  if data["informational_only"] is not True:
    raise LearningStatusError(
      "malformed",
      "Learner operation status is not marked display-only",
    )

  state = _text(data["state"], "state")
  if state not in _OPERATION_STATE_DIAGNOSTICS:
    raise LearningStatusError("malformed", "operation state is not recognized")
  diagnostic = _text(data["diagnostic"], "diagnostic")
  if diagnostic not in _OPERATION_STATE_DIAGNOSTICS[state]:
    raise LearningStatusError(
      "malformed",
      "operation diagnostic does not match its state",
    )

  terminal = _bool(data["terminal"], "terminal")
  if terminal != (state in _TERMINAL_OPERATION_STATES):
    raise LearningStatusError(
      "malformed",
      "operation terminal flag does not match its state",
    )

  started_mono_ns = _integer(data["started_mono_ns"], "started_mono_ns")
  updated_mono_ns = _integer(data["updated_mono_ns"], "updated_mono_ns")
  if updated_mono_ns < started_mono_ns:
    raise LearningStatusError(
      "stale",
      "Learner operation timestamp precedes its start",
    )
  if updated_mono_ns > now_mono_ns:
    raise LearningStatusError(
      "stale",
      "Learner operation status belongs to another monotonic epoch",
    )

  vehicle_identity = _nullable_text(
    data["vehicle_identity"],
    "vehicle_identity",
  )
  identity_optional_states = frozenset(("preparing", "failed"))
  if vehicle_identity is None and state not in identity_optional_states:
    raise LearningStatusError(
      "malformed",
      "operation state requires a vehicle identity",
    )
  if vehicle_identity is not None:
    if expected_vehicle_identity is None or not expected_vehicle_identity.strip():
      raise LearningStatusError(
        "vehicle_unavailable",
        "Current vehicle identity is unavailable",
      )
    if vehicle_identity != expected_vehicle_identity.strip():
      raise LearningStatusError(
        "wrong_vehicle",
        "Learner operation belongs to a different vehicle",
      )

  runtime_identity = _sha256(
    data["runtime_identity_sha256"],
    "runtime_identity_sha256",
    nullable=True,
  )
  runtime_optional_states = frozenset((
    "preparing",
    "failed",
    "drive_skipped_identity_mismatch",
  ))
  if runtime_identity is None and state not in runtime_optional_states:
    raise LearningStatusError(
      "malformed",
      "operation state requires a runtime identity",
    )
  if (
    runtime_identity is not None
    and expected_runtime_identity_sha256 is not None
    and runtime_identity != expected_runtime_identity_sha256
  ):
    raise LearningStatusError(
      "runtime_mismatch",
      "Learning snapshot and operation describe different runtimes",
    )

  current_route_identity = _sha256(
    data["current_route_identity"],
    "current_route_identity",
    nullable=True,
  )
  current_route_index = _integer(
    data["current_route_index"],
    "current_route_index",
    nullable=True,
  )
  total_route_count = _integer(
    data["total_route_count"],
    "total_route_count",
    nullable=True,
  )
  index_pair_complete = (
    current_route_index is not None and total_route_count is not None
  )
  index_pair_empty = current_route_index is None and total_route_count is None
  if not (index_pair_complete or index_pair_empty):
    raise LearningStatusError(
      "malformed",
      "route index and total must both be present or absent",
    )
  if index_pair_complete and (
    current_route_index < 1
    or total_route_count < 1
    or current_route_index > total_route_count
  ):
    raise LearningStatusError(
      "malformed",
      "route progress is outside its one-based bounds",
    )
  if index_pair_complete and current_route_identity is None:
    raise LearningStatusError(
      "malformed",
      "indexed route progress lacks a route identity",
    )
  if state != "backfilling" and not index_pair_empty:
    raise LearningStatusError(
      "malformed",
      "only backfill operations may publish route progress",
    )
  if (
    current_route_identity is not None
    and state not in ("collecting", "backfilling")
  ):
    raise LearningStatusError(
      "malformed",
      "operation state cannot claim a current route",
    )
  if (
    state == "backfilling"
    and diagnostic == "replaying_route"
    and (not index_pair_complete or current_route_identity is None)
  ):
    raise LearningStatusError(
      "malformed",
      "route replay requires complete progress",
    )
  if (
    state == "backfilling"
    and diagnostic == "scanning_routes"
    and (not index_pair_empty or current_route_identity is not None)
  ):
    raise LearningStatusError(
      "malformed",
      "route scanning cannot claim replay progress",
    )

  evidence_sha256 = _sha256(
    data["evidence_sha256"],
    "evidence_sha256",
    nullable=True,
  )
  ledger_sha256 = _sha256(
    data["ledger_sha256"],
    "ledger_sha256",
    nullable=True,
  )
  if state == "idle" and evidence_sha256 is None:
    raise LearningStatusError(
      "malformed",
      "idle operation status lacks persisted evidence identity",
    )
  if state == "ready_no_evidence" and (
    evidence_sha256 is not None
    or ledger_sha256 is not None
    or data["accepted_sample_count"] != 0
    or data["rejected_sample_count"] != 0
  ):
    raise LearningStatusError(
      "malformed",
      "ready-no-evidence operation unexpectedly claims evidence",
    )
  if ledger_sha256 is not None and evidence_sha256 is None:
    raise LearningStatusError(
      "malformed",
      "backfill ledger identity lacks persisted evidence identity",
    )

  return LearningOperationStatus(
    state=state,
    diagnostic=diagnostic,
    operation_id=_operation_id(data["operation_id"], "operation_id"),
    sequence=_integer(data["sequence"], "sequence"),
    started_mono_ns=started_mono_ns,
    updated_mono_ns=updated_mono_ns,
    terminal=terminal,
    vehicle_identity=vehicle_identity,
    runtime_identity_sha256=runtime_identity,
    current_route_identity=current_route_identity,
    current_route_index=current_route_index,
    total_route_count=total_route_count,
    last_route_identity=_sha256(
      data["last_route_identity"],
      "last_route_identity",
      nullable=True,
    ),
    accepted_sample_count=_integer(
      data["accepted_sample_count"],
      "accepted_sample_count",
    ),
    rejected_sample_count=_integer(
      data["rejected_sample_count"],
      "rejected_sample_count",
    ),
    retry_count=_integer(data["retry_count"], "retry_count"),
    evidence_sha256=evidence_sha256,
    ledger_sha256=ledger_sha256,
  )


def validate_operation_update(
  previous: LearningOperationStatus | None,
  current: LearningOperationStatus,
) -> None:
  """Reject a regressed or mutated sequence within one operation."""
  if previous is None or previous.operation_id != current.operation_id:
    return
  if current.sequence < previous.sequence:
    raise LearningStatusError(
      "stale",
      "Learner operation sequence moved backward",
    )
  if current.sequence == previous.sequence and current != previous:
    raise LearningStatusError(
      "stale",
      "Learner operation changed without advancing its sequence",
    )
  if (
    current.sequence > previous.sequence
    and current.updated_mono_ns <= previous.updated_mono_ns
  ):
    raise LearningStatusError(
      "stale",
      "Learner operation advanced with a stale timestamp",
    )


def _profile_identity(value: object, field: str) -> ProfileIdentity | None:
  if value is None:
    return None
  payload = _exact_object(value, _PROFILE_IDENTITY_KEYS, field)
  return ProfileIdentity(
    artifact_sha256=_sha256(
      payload["artifact_sha256"],
      f"{field}.artifact_sha256",
    ),
    profile_sha256=_sha256(
      payload["profile_sha256"],
      f"{field}.profile_sha256",
    ),
    profile_revision=_integer(
      payload["profile_revision"],
      f"{field}.profile_revision",
    ),
  )


def parse_lifecycle_status(
  raw: object,
  *,
  expected_vehicle_identity: str | None,
  expected_runtime_identity_sha256: str | None,
) -> LifecycleStatus:
  """Decode the profiled-owned lifecycle projection, never raw authority."""
  if raw is None:
    raise LearningStatusError(
      "activation_absent",
      "Controller status is not available yet",
    )
  if expected_vehicle_identity is None or not expected_vehicle_identity.strip():
    raise LearningStatusError(
      "vehicle_unavailable",
      "Current vehicle identity is unavailable",
    )
  if type(raw) is not dict:
    raise LearningStatusError(
      "malformed",
      "lifecycle status must be a Params JSON object",
    )
  if (
    type(raw.get("schema_version")) is not int
    or raw["schema_version"] != LIFECYCLE_STATUS_SCHEMA_VERSION
  ):
    raise LearningStatusError(
      "schema_mismatch",
      "Controller status version is not supported",
    )
  data = _exact_object(raw, _LIFECYCLE_KEYS, "lifecycle status")
  if data["informational_only"] is not True:
    raise LearningStatusError(
      "malformed",
      "Controller status is not marked display-only",
    )

  vehicle_identity = _text(data["vehicle_identity"], "vehicle_identity")
  if vehicle_identity != expected_vehicle_identity.strip():
    raise LearningStatusError(
      "wrong_vehicle",
      "Controller status belongs to a different vehicle",
    )
  runtime_identity = _sha256(
    data["runtime_identity_sha256"],
    "runtime_identity_sha256",
  )
  if (
    expected_runtime_identity_sha256 is not None
    and runtime_identity != expected_runtime_identity_sha256
  ):
    raise LearningStatusError(
      "runtime_mismatch",
      "Learning and controller snapshots describe different runtimes",
    )

  diagnostic = _text(data["diagnostic"], "diagnostic")
  if diagnostic not in _ARTIFACT_DIAGNOSTICS:
    raise LearningStatusError("malformed", "diagnostic is not recognized")
  controller_state = _text(data["controller_state"], "controller_state")
  if controller_state not in _CONTROLLER_STATES:
    raise LearningStatusError("malformed", "controller_state is not recognized")
  effective_controller = _text(
    data["effective_controller"],
    "effective_controller",
  )
  if effective_controller not in _EFFECTIVE_CONTROLLERS:
    raise LearningStatusError(
      "malformed",
      "effective_controller is not recognized",
    )

  active_profile = _profile_identity(data["active_profile"], "active_profile")
  staged_profile = _profile_identity(data["staged_profile"], "staged_profile")
  production_verified = _bool(
    data["production_envelope_verified"],
    "production_envelope_verified",
  )
  active_states = frozenset(("provisional", "approved", "rollback_pending"))
  if (controller_state in active_states) != (active_profile is not None):
    raise LearningStatusError(
      "malformed",
      "controller state and active profile disagree",
    )
  if controller_state == "stock" and staged_profile is not None:
    raise LearningStatusError(
      "malformed",
      "stock lifecycle unexpectedly contains a staged profile",
    )
  if controller_state == "staged" and staged_profile is None:
    raise LearningStatusError(
      "malformed",
      "staged lifecycle lacks a staged profile",
    )
  expected_effective = (
    "modular"
    if controller_state in ("provisional", "approved")
    else "stock"
  )
  if effective_controller != expected_effective:
    raise LearningStatusError(
      "malformed",
      "controller state and effective controller disagree",
    )
  if controller_state in ("provisional", "approved") and (
    diagnostic != "ok" or not production_verified
  ):
    raise LearningStatusError(
      "malformed",
      "active modular lifecycle is not validated",
    )
  if controller_state in ("staged", "rollback_pending") and (
    diagnostic != "ok" or not production_verified
  ):
    raise LearningStatusError(
      "malformed",
      "profile-bearing stock lifecycle is not validated",
    )
  if controller_state == "unavailable" and diagnostic in ("ok", "absent"):
    raise LearningStatusError(
      "malformed",
      "unavailable lifecycle lacks a failure diagnostic",
    )
  activation_state_sha256 = _sha256(
    data["activation_state_sha256"],
    "activation_state_sha256",
    nullable=True,
  )
  if diagnostic in ("ok", "absent") and activation_state_sha256 is None:
    raise LearningStatusError(
      "malformed",
      "validated lifecycle lacks its activation-state identity",
    )

  return LifecycleStatus(
    vehicle_identity=vehicle_identity,
    runtime_identity_sha256=runtime_identity,
    source_openpilot_commit=_commit(
      data["source_openpilot_commit"],
      "source_openpilot_commit",
    ),
    opendbc_commit=_commit(data["opendbc_commit"], "opendbc_commit"),
    activation_state_sha256=activation_state_sha256,
    diagnostic=diagnostic,
    controller_state=controller_state,
    effective_controller=effective_controller,
    production_envelope_verified=production_verified,
    active_profile=active_profile,
    staged_profile=staged_profile,
    rejected_profile_count=_integer(
      data["rejected_profile_count"],
      "rejected_profile_count",
    ),
  )


def operation_presentation(
  status: LearningOperationStatus | None,
  *,
  error_code: str | None,
  error_message: str | None,
  has_learning_snapshot: bool,
) -> OperationPresentation:
  """Return truthful copy without treating absence as an empty history."""
  if status is None:
    unavailable = error_message or "Awaiting the learner's current status"
    if error_code == "operation_absent":
      unavailable = "Awaiting learner status; drive history is unknown"
    return OperationPresentation(
      title="LEARNER STATUS UNAVAILABLE",
      detail=unavailable,
      tone=("gray" if error_code in (None, "operation_absent", "vehicle_unavailable") else "red"),
      show_banner=has_learning_snapshot,
    )

  prior = "Prior snapshot shown · " if has_learning_snapshot and status.active else ""
  counts = f"{status.accepted_sample_count:,} accepted · {status.rejected_sample_count:,} rejected"
  diagnostic = _OPERATION_DIAGNOSTIC_LABELS[status.diagnostic]

  if status.state == "preparing":
    return OperationPresentation(
      title="PREPARING LEARNER",
      detail=f"{prior}{diagnostic}",
      tone="blue",
      show_banner=has_learning_snapshot,
    )
  if status.state == "ready_no_evidence":
    return OperationPresentation(
      title="READY FOR FIRST DRIVE",
      detail="Complete one drive to begin collecting clean support",
      tone="blue",
      show_banner=has_learning_snapshot,
    )
  if status.state == "collecting":
    return OperationPresentation(
      title="COLLECTING THIS DRIVE",
      detail=f"{prior}{counts}",
      tone="blue",
      show_banner=has_learning_snapshot,
    )
  if status.state == "finalizing":
    return OperationPresentation(
      title="FINALIZING LEARNING DATA",
      detail=f"{prior}{diagnostic}",
      tone="blue",
      show_banner=has_learning_snapshot,
    )
  if status.state == "retry_pending":
    retry = f"Retry {status.retry_count} pending"
    return OperationPresentation(
      title="SAVE RETRY PENDING",
      detail=f"{prior}{retry} · {counts}",
      tone="amber",
      show_banner=has_learning_snapshot,
    )
  if status.state == "backfilling":
    if status.diagnostic == "scanning_routes":
      progress = "Scanning compatible routes"
    else:
      progress = f"Route {status.current_route_index}/{status.total_route_count}"
    return OperationPresentation(
      title="PROCESSING PRIOR ROUTES",
      detail=f"{prior}{progress} · {counts}",
      tone="blue",
      show_banner=has_learning_snapshot,
    )
  if status.state == "idle":
    return OperationPresentation(
      title="LEARNER READY",
      detail=diagnostic,
      tone="green",
      show_banner=has_learning_snapshot,
    )
  if status.state == "drive_skipped_identity_mismatch":
    return OperationPresentation(
      title="DRIVE SKIPPED · NEXT DRIVE READY",
      detail=diagnostic,
      tone="amber",
      show_banner=has_learning_snapshot,
    )
  return OperationPresentation(
    title="LEARNER FAILED",
    detail=diagnostic,
    tone="red",
    show_banner=has_learning_snapshot,
  )


def learning_panel_presentation(
  status: LearningOperationStatus | None,
  *,
  operation_error_code: str | None,
  operation_error_message: str | None,
  learning_error_code: str | None,
  learning_error_message: str | None,
  has_learning_snapshot: bool,
) -> OperationPresentation:
  """Plan the banner or empty state without discarding a valid snapshot."""
  operation = operation_presentation(
    status,
    error_code=operation_error_code,
    error_message=operation_error_message,
    has_learning_snapshot=has_learning_snapshot,
  )
  if has_learning_snapshot:
    return operation

  # An explicit ready_no_evidence status is the only authority for first-drive
  # copy. Missing caches reveal nothing about drive history.
  if status is not None and status.state != "idle":
    return operation
  if learning_error_code not in (None, "absent", "vehicle_unavailable"):
    return OperationPresentation(
      title="LEARNING DATA UNAVAILABLE",
      detail=(learning_error_message or "Validated current-build data could not be read"),
      tone="red",
      show_banner=False,
    )
  if status is not None and status.state == "idle":
    return OperationPresentation(
      title="LEARNING SNAPSHOT UNAVAILABLE",
      detail=(learning_error_message or "Learner is ready, but its validated snapshot is unavailable"),
      tone="red",
      show_banner=False,
    )
  return operation


def reason_label(reason: str) -> str:
  return _REASON_LABELS.get(reason, "Status unavailable")


def format_duration(seconds: float) -> str:
  total = max(0, int(round(seconds)))
  hours, remainder = divmod(total, 3600)
  minutes, secs = divmod(remainder, 60)
  if hours:
    return f"{hours:d}h {minutes:02d}m"
  return f"{minutes:d}:{secs:02d}"


def format_speed(speed_mps: float, *, metric: bool) -> str:
  converted = speed_mps * (3.6 if metric else 2.2369362920544)
  return f"{converted:.0f} {'km/h' if metric else 'mph'}"


def grid_cells(
  width: float,
  height: float,
  count: int,
  *,
  columns: int = 2,
  gap: float = 14.0,
) -> tuple[GridCell, ...]:
  if count <= 0 or columns <= 0 or width <= 0.0 or height <= 0.0:
    return ()
  rows = (count + columns - 1) // columns
  cell_width = (width - gap * (columns - 1)) / columns
  cell_height = (height - gap * (rows - 1)) / rows
  if cell_width <= 0.0 or cell_height <= 0.0:
    return ()
  return tuple(
    GridCell(
      x=(index % columns) * (cell_width + gap),
      y=(index // columns) * (cell_height + gap),
      width=cell_width,
      height=cell_height,
    )
    for index in range(count)
  )


def cycle_page_index(current: int, direction: int, page_count: int) -> int:
  if page_count <= 0:
    raise ValueError("page_count must be positive")
  if direction not in (-1, 1):
    raise ValueError("direction must be -1 or 1")
  if not 0 <= current < page_count:
    raise ValueError("current page is outside the carousel")
  return (current + direction) % page_count
