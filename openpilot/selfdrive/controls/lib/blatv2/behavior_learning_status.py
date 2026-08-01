"""Strict display-only status for offroad BLaTv2 behavior qualification.

``BLaTv2BehaviorLearningStatus`` is intentionally separate from the physical
``BLaTv2LearningStatus`` projection.  This document may drive UI progress and
diagnostics only.  It is never an input to replay, selection, approval, live
controller construction, or activation; deleting it cannot change actuation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import json
import re
import secrets
import time
from typing import Any

from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import canonical_json


BEHAVIOR_LEARNING_STATUS_PARAM = "BLaTv2BehaviorLearningStatus"
BEHAVIOR_LEARNING_STATUS_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_OPERATION_ID_RE = re.compile(r"[0-9a-f]{32}\Z")


class BehaviorLearningState(StrEnum):
  WAITING_FOR_PHYSICAL_PROFILE = "waiting_for_physical_profile"
  WAITING_FOR_ROUTES = "waiting_for_routes"
  PREPARING = "preparing"
  TRAINING = "training"
  SELECTING = "selecting"
  VALIDATING = "validating"
  PUBLISHING = "publishing"
  COMPLETE = "complete"
  FAILED = "failed"


class BehaviorLearningDiagnostic(StrEnum):
  PHYSICAL_PROFILE_UNQUALIFIED = "physical_profile_unqualified"
  INSUFFICIENT_HOMOGENEOUS_ROUTES = "insufficient_homogeneous_routes"
  VALIDATING_ROUTE_EVIDENCE = "validating_route_evidence"
  REPLAYING_TRAINING_GRID = "replaying_training_grid"
  SELECTING_TRAINING_WINNER = "selecting_training_winner"
  REPLAYING_FROZEN_WINNER = "replaying_frozen_winner"
  PUBLISHING_BEHAVIOR_GENERATION = "publishing_behavior_generation"
  CANDIDATE_QUALIFIED = "candidate_qualified"
  STOCK_RETAINED = "stock_retained"
  ROUTE_EVIDENCE_INVALID = "route_evidence_invalid"
  REPLAY_NONDETERMINISTIC = "replay_nondeterministic"
  BEHAVIOR_TRANSACTION_FAILED = "behavior_transaction_failed"
  BEHAVIOR_PUBLISH_FAILED = "behavior_publish_failed"


class BehaviorQualificationDisposition(StrEnum):
  STOCK_RETAINED = "stock_retained"
  QUALIFIED_CANDIDATE_AVAILABLE = "qualified_candidate_available"


_STATE_DIAGNOSTICS = {
  BehaviorLearningState.WAITING_FOR_PHYSICAL_PROFILE: {
    BehaviorLearningDiagnostic.PHYSICAL_PROFILE_UNQUALIFIED,
  },
  BehaviorLearningState.WAITING_FOR_ROUTES: {
    BehaviorLearningDiagnostic.INSUFFICIENT_HOMOGENEOUS_ROUTES,
  },
  BehaviorLearningState.PREPARING: {
    BehaviorLearningDiagnostic.VALIDATING_ROUTE_EVIDENCE,
  },
  BehaviorLearningState.TRAINING: {
    BehaviorLearningDiagnostic.REPLAYING_TRAINING_GRID,
  },
  BehaviorLearningState.SELECTING: {
    BehaviorLearningDiagnostic.SELECTING_TRAINING_WINNER,
  },
  BehaviorLearningState.VALIDATING: {
    BehaviorLearningDiagnostic.REPLAYING_FROZEN_WINNER,
  },
  BehaviorLearningState.PUBLISHING: {
    BehaviorLearningDiagnostic.PUBLISHING_BEHAVIOR_GENERATION,
  },
  BehaviorLearningState.COMPLETE: {
    BehaviorLearningDiagnostic.CANDIDATE_QUALIFIED,
    BehaviorLearningDiagnostic.STOCK_RETAINED,
  },
  BehaviorLearningState.FAILED: {
    BehaviorLearningDiagnostic.ROUTE_EVIDENCE_INVALID,
    BehaviorLearningDiagnostic.REPLAY_NONDETERMINISTIC,
    BehaviorLearningDiagnostic.BEHAVIOR_TRANSACTION_FAILED,
    BehaviorLearningDiagnostic.BEHAVIOR_PUBLISH_FAILED,
  },
}


def _optional_sha256(value: str | None, name: str) -> None:
  if value is not None and (
    type(value) is not str or _SHA256_RE.fullmatch(value) is None
  ):
    raise ValueError(f"{name} must be lowercase SHA-256 or None")


def _strict_nonnegative(value: int, name: str) -> None:
  if type(value) is not int or value < 0:
    raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class BehaviorLearningStatus:
  """Immutable informational projection of one behavior-learning operation."""

  schema_version: int
  informational_only: bool
  operation_id: str
  sequence: int
  state: BehaviorLearningState
  diagnostic: BehaviorLearningDiagnostic
  terminal: bool
  started_mono_ns: int
  updated_mono_ns: int
  vehicle_identity: str
  runtime_vehicle_identity_sha256: str
  physical_generation_sha256: str | None
  physical_profile_sha256: str | None
  recorded_source_identity_sha256: str | None
  eligible_route_count: int
  required_route_count: int
  training_route_count: int
  validation_route_count: int
  current_route_identity: str | None
  current_route_index: int | None
  total_route_count: int
  current_candidate_index: int | None
  total_candidate_count: int
  completed_replay_jobs: int
  total_replay_jobs: int
  gate_spec_sha256: str
  segmentation_config_sha256: str
  transaction_sha256: str | None
  behavior_finalization_sha256: str | None
  behavior_selection_sha256: str | None
  selected_behavior_policy_sha256: str | None
  smooth_passed: bool | None
  swift_passed: bool | None
  strong_passed: bool | None
  target_materially_improved: bool | None
  qualification_disposition: BehaviorQualificationDisposition | None
  reasons: tuple[str, ...]

  def __post_init__(self) -> None:
    if self.schema_version != BEHAVIOR_LEARNING_STATUS_SCHEMA_VERSION:
      raise ValueError("behavior learning status schema is incompatible")
    if self.informational_only is not True:
      raise ValueError("behavior learning status must remain informational-only")
    if type(self.operation_id) is not str or _OPERATION_ID_RE.fullmatch(self.operation_id) is None:
      raise ValueError("operation_id must be 128-bit lowercase hex")
    if not isinstance(self.state, BehaviorLearningState):
      raise TypeError("state must be BehaviorLearningState")
    if not isinstance(self.diagnostic, BehaviorLearningDiagnostic):
      raise TypeError("diagnostic must be BehaviorLearningDiagnostic")
    if self.diagnostic not in _STATE_DIAGNOSTICS[self.state]:
      raise ValueError("diagnostic does not belong to behavior-learning state")
    terminal_state = self.state in (
      BehaviorLearningState.COMPLETE,
      BehaviorLearningState.FAILED,
    )
    if type(self.terminal) is not bool or self.terminal != terminal_state:
      raise ValueError("terminal flag and state disagree")
    for name in (
      "sequence",
      "started_mono_ns",
      "updated_mono_ns",
      "eligible_route_count",
      "required_route_count",
      "training_route_count",
      "validation_route_count",
      "total_route_count",
      "total_candidate_count",
      "completed_replay_jobs",
      "total_replay_jobs",
    ):
      _strict_nonnegative(getattr(self, name), name)
    if self.updated_mono_ns < self.started_mono_ns:
      raise ValueError("status update cannot precede operation start")
    if type(self.vehicle_identity) is not str or not self.vehicle_identity.strip():
      raise ValueError("vehicle_identity must not be empty")
    for name in (
      "runtime_vehicle_identity_sha256",
      "gate_spec_sha256",
      "segmentation_config_sha256",
    ):
      value = getattr(self, name)
      if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    for name in (
      "physical_generation_sha256",
      "physical_profile_sha256",
      "recorded_source_identity_sha256",
      "transaction_sha256",
      "behavior_finalization_sha256",
      "behavior_selection_sha256",
      "selected_behavior_policy_sha256",
    ):
      _optional_sha256(getattr(self, name), name)
    if self.training_route_count + self.validation_route_count > self.eligible_route_count:
      raise ValueError("route partition counts exceed eligible evidence")
    if self.current_route_identity is None:
      if self.current_route_index is not None:
        raise ValueError("route index requires a current route identity")
    elif (
      type(self.current_route_identity) is not str
      or not self.current_route_identity.strip()
      or type(self.current_route_index) is not int
      or not 0 <= self.current_route_index < self.total_route_count
    ):
      raise ValueError("current route progress is invalid")
    if self.current_candidate_index is not None and (
      type(self.current_candidate_index) is not int
      or not 0 <= self.current_candidate_index < self.total_candidate_count
    ):
      raise ValueError("current candidate progress is invalid")
    if self.total_candidate_count == 0 and self.current_candidate_index is not None:
      raise ValueError("candidate index requires a non-empty grid")
    if self.completed_replay_jobs > self.total_replay_jobs:
      raise ValueError("completed replay jobs exceed total")
    if type(self.reasons) is not tuple or any(
      type(reason) is not str for reason in self.reasons
    ) or tuple(sorted(set(self.reasons))) != self.reasons or any(
      not reason.strip() for reason in self.reasons
    ):
      raise ValueError("reasons must be non-empty, unique, and sorted")

    gates = (
      self.smooth_passed,
      self.swift_passed,
      self.strong_passed,
      self.target_materially_improved,
    )
    if any(value is not None and type(value) is not bool for value in gates):
      raise ValueError("behavior gate verdicts must be boolean or None")
    if self.qualification_disposition is not None and not isinstance(
      self.qualification_disposition,
      BehaviorQualificationDisposition,
    ):
      raise TypeError("qualification disposition enum is invalid")
    if not self.terminal:
      if any(value is not None for value in gates):
        raise ValueError("non-terminal status cannot publish gate verdicts")
      if self.qualification_disposition is not None or self.reasons:
        raise ValueError("non-terminal status cannot publish a disposition")
      if any(value is not None for value in (
        self.transaction_sha256,
        self.behavior_finalization_sha256,
        self.behavior_selection_sha256,
        self.selected_behavior_policy_sha256,
      )):
        raise ValueError("non-terminal status cannot publish selection hashes")
      return

    if self.qualification_disposition is None or not self.reasons:
      raise ValueError("terminal status needs a disposition and reasons")
    if self.state is BehaviorLearningState.COMPLETE and any(
      type(value) is not bool for value in gates
    ):
      raise ValueError("completed status needs every gate verdict")
    if self.state is BehaviorLearningState.COMPLETE and (
      self.transaction_sha256 is None
      or self.behavior_finalization_sha256 is None
    ):
      raise ValueError("completed status needs transaction provenance")

    qualified = (
      self.qualification_disposition
      is BehaviorQualificationDisposition.QUALIFIED_CANDIDATE_AVAILABLE
    )
    if qualified:
      if gates != (True, True, True, True):
        raise ValueError("qualified candidate must pass every behavior gate")
      if (
        self.state is not BehaviorLearningState.COMPLETE
        or self.behavior_selection_sha256 is None
        or self.selected_behavior_policy_sha256 is None
      ):
        raise ValueError("qualified candidate lacks immutable selection provenance")
    elif (
      self.behavior_selection_sha256 is not None
      or self.selected_behavior_policy_sha256 is not None
    ):
      raise ValueError("stock-retained status cannot expose a selected policy")

  def to_dict(self) -> dict[str, Any]:
    return {
      "behaviorFinalizationSha256": self.behavior_finalization_sha256,
      "behaviorSelectionSha256": self.behavior_selection_sha256,
      "completedReplayJobs": self.completed_replay_jobs,
      "currentCandidateIndex": self.current_candidate_index,
      "currentRouteIdentity": self.current_route_identity,
      "currentRouteIndex": self.current_route_index,
      "diagnostic": self.diagnostic.value,
      "eligibleRouteCount": self.eligible_route_count,
      "gateSpecSha256": self.gate_spec_sha256,
      "informationalOnly": self.informational_only,
      "operationId": self.operation_id,
      "physicalGenerationSha256": self.physical_generation_sha256,
      "physicalProfileSha256": self.physical_profile_sha256,
      "qualificationDisposition": (
        None
        if self.qualification_disposition is None
        else self.qualification_disposition.value
      ),
      "reasons": list(self.reasons),
      "recordedSourceIdentitySha256": self.recorded_source_identity_sha256,
      "requiredRouteCount": self.required_route_count,
      "runtimeVehicleIdentitySha256": self.runtime_vehicle_identity_sha256,
      "schemaVersion": self.schema_version,
      "segmentationConfigSha256": self.segmentation_config_sha256,
      "selectedBehaviorPolicySha256": self.selected_behavior_policy_sha256,
      "sequence": self.sequence,
      "smoothPassed": self.smooth_passed,
      "startedMonoNs": self.started_mono_ns,
      "state": self.state.value,
      "strongPassed": self.strong_passed,
      "swiftPassed": self.swift_passed,
      "targetMateriallyImproved": self.target_materially_improved,
      "terminal": self.terminal,
      "totalCandidateCount": self.total_candidate_count,
      "totalReplayJobs": self.total_replay_jobs,
      "totalRouteCount": self.total_route_count,
      "trainingRouteCount": self.training_route_count,
      "transactionSha256": self.transaction_sha256,
      "updatedMonoNs": self.updated_mono_ns,
      "validationRouteCount": self.validation_route_count,
      "vehicleIdentity": self.vehicle_identity,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @classmethod
  def from_json(cls, encoded: str) -> BehaviorLearningStatus:
    try:
      payload = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
      raise ValueError("behavior learning status is invalid JSON") from exc
    expected_keys = frozenset(cls._json_keys())
    if type(payload) is not dict or frozenset(payload) != expected_keys:
      raise ValueError("behavior learning status keys do not match schema")

    def enum_or_none(enum_type, value):
      return None if value is None else enum_type(value)

    reasons = payload["reasons"]
    if type(reasons) is not list or any(type(value) is not str for value in reasons):
      raise ValueError("behavior learning reasons must be a text array")
    return cls(
      schema_version=payload["schemaVersion"],
      informational_only=payload["informationalOnly"],
      operation_id=payload["operationId"],
      sequence=payload["sequence"],
      state=BehaviorLearningState(payload["state"]),
      diagnostic=BehaviorLearningDiagnostic(payload["diagnostic"]),
      terminal=payload["terminal"],
      started_mono_ns=payload["startedMonoNs"],
      updated_mono_ns=payload["updatedMonoNs"],
      vehicle_identity=payload["vehicleIdentity"],
      runtime_vehicle_identity_sha256=payload["runtimeVehicleIdentitySha256"],
      physical_generation_sha256=payload["physicalGenerationSha256"],
      physical_profile_sha256=payload["physicalProfileSha256"],
      recorded_source_identity_sha256=payload["recordedSourceIdentitySha256"],
      eligible_route_count=payload["eligibleRouteCount"],
      required_route_count=payload["requiredRouteCount"],
      training_route_count=payload["trainingRouteCount"],
      validation_route_count=payload["validationRouteCount"],
      current_route_identity=payload["currentRouteIdentity"],
      current_route_index=payload["currentRouteIndex"],
      total_route_count=payload["totalRouteCount"],
      current_candidate_index=payload["currentCandidateIndex"],
      total_candidate_count=payload["totalCandidateCount"],
      completed_replay_jobs=payload["completedReplayJobs"],
      total_replay_jobs=payload["totalReplayJobs"],
      gate_spec_sha256=payload["gateSpecSha256"],
      segmentation_config_sha256=payload["segmentationConfigSha256"],
      transaction_sha256=payload["transactionSha256"],
      behavior_finalization_sha256=payload["behaviorFinalizationSha256"],
      behavior_selection_sha256=payload["behaviorSelectionSha256"],
      selected_behavior_policy_sha256=payload["selectedBehaviorPolicySha256"],
      smooth_passed=payload["smoothPassed"],
      swift_passed=payload["swiftPassed"],
      strong_passed=payload["strongPassed"],
      target_materially_improved=payload["targetMateriallyImproved"],
      qualification_disposition=enum_or_none(
        BehaviorQualificationDisposition,
        payload["qualificationDisposition"],
      ),
      reasons=tuple(reasons),
    )

  @staticmethod
  def _json_keys() -> tuple[str, ...]:
    # One authority for strict decode and serialization keys.
    return (
      "behaviorFinalizationSha256",
      "behaviorSelectionSha256",
      "completedReplayJobs",
      "currentCandidateIndex",
      "currentRouteIdentity",
      "currentRouteIndex",
      "diagnostic",
      "eligibleRouteCount",
      "gateSpecSha256",
      "informationalOnly",
      "operationId",
      "physicalGenerationSha256",
      "physicalProfileSha256",
      "qualificationDisposition",
      "reasons",
      "recordedSourceIdentitySha256",
      "requiredRouteCount",
      "runtimeVehicleIdentitySha256",
      "schemaVersion",
      "segmentationConfigSha256",
      "selectedBehaviorPolicySha256",
      "sequence",
      "smoothPassed",
      "startedMonoNs",
      "state",
      "strongPassed",
      "swiftPassed",
      "targetMateriallyImproved",
      "terminal",
      "totalCandidateCount",
      "totalReplayJobs",
      "totalRouteCount",
      "trainingRouteCount",
      "transactionSha256",
      "updatedMonoNs",
      "validationRouteCount",
      "vehicleIdentity",
    )


_PUBLISH_CONTEXT_DEFAULTS: dict[str, object] = {
  "vehicle_identity": "",
  "runtime_vehicle_identity_sha256": "",
  "physical_generation_sha256": None,
  "physical_profile_sha256": None,
  "recorded_source_identity_sha256": None,
  "eligible_route_count": 0,
  "required_route_count": 0,
  "training_route_count": 0,
  "validation_route_count": 0,
  "current_route_identity": None,
  "current_route_index": None,
  "total_route_count": 0,
  "current_candidate_index": None,
  "total_candidate_count": 0,
  "completed_replay_jobs": 0,
  "total_replay_jobs": 0,
  "gate_spec_sha256": "",
  "segmentation_config_sha256": "",
  "transaction_sha256": None,
  "behavior_finalization_sha256": None,
  "behavior_selection_sha256": None,
  "selected_behavior_policy_sha256": None,
  "smooth_passed": None,
  "swift_passed": None,
  "strong_passed": None,
  "target_materially_improved": None,
  "qualification_disposition": None,
  "reasons": (),
}

# These fields identify the exact evidence and executable/configuration inputs
# of an operation.  A nullable identity may be established once (for example,
# after waiting for a physical profile), but it may never subsequently change.
_STABLE_IDENTITY_CONTEXT_FIELDS = (
  "vehicle_identity",
  "runtime_vehicle_identity_sha256",
  "physical_generation_sha256",
  "physical_profile_sha256",
  "recorded_source_identity_sha256",
  "gate_spec_sha256",
  "segmentation_config_sha256",
)
_STABLE_TOTAL_CONTEXT_FIELDS = (
  "eligible_route_count",
  "required_route_count",
  "training_route_count",
  "validation_route_count",
  "total_route_count",
  "total_candidate_count",
  "total_replay_jobs",
)


class BehaviorLearningStatusPublisher:
  """Publish one strict, sequenced, informational behavior operation.

  Publication is transactional with respect to this object's state: validation
  and the blocking Params write both complete before the operation advances.
  This class deliberately has no dependency on approval or activation code.
  """

  def __init__(
    self,
    params: Any,
    *,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    operation_id_factory: Callable[[], str] = lambda: secrets.token_hex(16),
  ) -> None:
    self._params = params
    self._monotonic_ns = monotonic_ns
    self._operation_id_factory = operation_id_factory
    self._last_status: BehaviorLearningStatus | None = None
    self._established_context: dict[str, object] = {}

  @property
  def last_status(self) -> BehaviorLearningStatus | None:
    # BehaviorLearningStatus is frozen and contains only immutable members.
    return self._last_status

  def publish(
    self,
    state: BehaviorLearningState | str,
    diagnostic: BehaviorLearningDiagnostic | str,
    *,
    new_operation: bool = False,
    **context: object,
  ) -> BehaviorLearningStatus:
    unknown = set(context) - set(_PUBLISH_CONTEXT_DEFAULTS)
    if unknown:
      raise ValueError(f"unknown behavior-learning context: {sorted(unknown)}")

    previous = self._last_status
    starting = new_operation or previous is None
    if not starting and previous is not None and previous.terminal:
      raise ValueError("terminal behavior operation requires a new operation")

    now = self._monotonic_ns()
    if type(now) is not int or now < 0:
      raise ValueError("monotonic clock must return a non-negative integer")
    if previous is not None and not starting and now < previous.updated_mono_ns:
      raise ValueError("behavior operation time cannot move backwards")

    if starting:
      operation_id = self._operation_id_factory()
      if type(operation_id) is not str or _OPERATION_ID_RE.fullmatch(operation_id) is None:
        raise ValueError("operation id factory returned an invalid identity")
      sequence = 0
      started_mono_ns = now
      established: dict[str, object] = {}
    else:
      assert previous is not None
      operation_id = previous.operation_id
      sequence = previous.sequence + 1
      started_mono_ns = previous.started_mono_ns
      established = dict(self._established_context)

    fields = dict(_PUBLISH_CONTEXT_DEFAULTS)
    # Established operation context is inherited when callers update only a
    # progress coordinate.  Transient route/candidate and terminal fields use
    # their explicit defaults on every publication.
    fields.update(established)
    if previous is not None and not starting:
      fields["completed_replay_jobs"] = previous.completed_replay_jobs
    fields.update(context)

    stable_fields = (
      *_STABLE_IDENTITY_CONTEXT_FIELDS,
      *_STABLE_TOTAL_CONTEXT_FIELDS,
    )
    for name in stable_fields:
      if name not in context:
        continue
      value = context[name]
      if name in established and value != established[name]:
        raise ValueError(f"behavior operation {name} cannot change")
      # Optional provenance is not established by an absent value.  Counts and
      # required identities are established whenever explicitly supplied.
      if value is not None:
        established[name] = value

    completed_jobs = fields["completed_replay_jobs"]
    if previous is not None and not starting and (
      type(completed_jobs) is not int
      or completed_jobs < previous.completed_replay_jobs
    ):
      raise ValueError("completed replay jobs cannot move backwards")

    resolved_state = BehaviorLearningState(state)
    resolved_diagnostic = BehaviorLearningDiagnostic(diagnostic)
    status = BehaviorLearningStatus(
      schema_version=BEHAVIOR_LEARNING_STATUS_SCHEMA_VERSION,
      informational_only=True,
      operation_id=operation_id,
      sequence=sequence,
      state=resolved_state,
      diagnostic=resolved_diagnostic,
      terminal=resolved_state in (
        BehaviorLearningState.COMPLETE,
        BehaviorLearningState.FAILED,
      ),
      started_mono_ns=started_mono_ns,
      updated_mono_ns=now,
      **fields,
    )
    payload = status.to_dict()
    self._params.put(BEHAVIOR_LEARNING_STATUS_PARAM, payload, block=True)

    # Commit lifecycle state only after Params confirms the blocking write.
    self._last_status = status
    self._established_context = established
    return status
