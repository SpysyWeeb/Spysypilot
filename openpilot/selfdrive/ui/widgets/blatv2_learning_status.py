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

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
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


def _sha256(value: object, field: str, *, nullable: bool = False) -> str | None:
  if nullable and value is None:
    return None
  result = _text(value, field)
  if _SHA256_RE.fullmatch(result) is None:
    raise LearningStatusError("malformed", f"{field} must be lowercase SHA-256")
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
