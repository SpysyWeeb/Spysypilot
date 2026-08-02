"""Offroad-only composition of physical evidence and behavioral replay.

This is the narrow production seam between four independently testable
owners: the immutable physical generation, the homogeneous route-evidence
cohort, the pure replay/selection transaction, and the immutable behavior
generation store.  It has no live-controller, approval, staging, or
activation API.  A successful result is still only an informational candidate;
stock remains the actuator until the separately reviewed approval lifecycle
accepts a complete artifact.

The retired eager implementation expanded whole cohorts into Python object
graphs and is intentionally absent from this production seam.  Authenticated
cached generations remain readable, but a new eligible cohort retains stock
with ``behavior_streaming_required`` until the independently gated route-major
backend replaces it.  The pure replay/transaction modules remain as the
numerical reference; they are not called by this device pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from openpilot.selfdrive.controls.lib.blatv2.approved_artifact import (
  ArtifactDiagnostic,
  PersistentProfileActivation,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_configuration import (
  load_behavior_gate_spec,
  load_behavior_segmentation_config,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  ReplayCoreIdentity,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorSourceIdentity,
  canonical_json,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_generation import (
  BehaviorGenerationError,
  LoadedBehaviorGeneration,
  load_current_behavior_generation,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_learning_status import (
  BehaviorLearningDiagnostic,
  BehaviorLearningState,
  BehaviorLearningStatusPublisher,
  BehaviorQualificationDisposition,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import (
  BehaviorPolicy,
  build_candidate_grid,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay import (
  BEHAVIOR_REPLAY_INPUT_SCHEMA_VERSION,
  behavior_source_identity_from_route_source,
  reviewed_replay_core_identity,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_transaction import (
  BehaviorLearningTransactionResult,
  QualificationDisposition,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  VehicleCalibrationProfile,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  BehaviorEvidenceCohortSelection,
  load_ledger,
  select_homogeneous_behavior_cohort,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  PersistentLearningRuntime,
)
from openpilot.selfdrive.controls.lib.blatv2.policy import ControllerPolicy
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  RouteEvidenceFileSummary,
  RouteEvidenceStore,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
)


BEHAVIOR_GENERATION_DIRECTORY = "behavior_generations_v1"
PROVISIONAL_CONTROLLER_POLICY_PATH = (
  Path(__file__).resolve().parent / "provisional_controller_policy.json"
)
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class BehaviorPipelineError(RuntimeError):
  """Behavior qualification could not produce an authoritative result."""


class BehaviorPipelineAborted(BehaviorPipelineError):
  """Offroad ownership ended before behavior publication."""


@dataclass(frozen=True, slots=True)
class BehaviorPipelineResult:
  state: str
  generation_sha256: str | None
  transaction_sha256: str | None
  route_count: int
  diagnostic: str

  def __post_init__(self) -> None:
    if self.state not in {"waiting", "cached", "published", "failed", "aborted"}:
      raise ValueError("unknown behavior pipeline result state")
    if self.route_count < 0:
      raise ValueError("behavior route count must be non-negative")


def _sha256_json(payload: object) -> str:
  return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_replay_core_identity(
  *,
  controller_name: str,
  implementation_contract: str,
  source_openpilot_commit: str,
  opendbc_commit: str,
  panda_commit: str,
) -> ReplayCoreIdentity:
  """Bind replay semantics to exact clean-build commits and input schema.

  The manager's clean-build guard makes each commit an identity for all source
  bytes.  The additional implementation contract prevents stock and modular
  adapters at the same commit from aliasing one another.
  """
  commits = (source_openpilot_commit, opendbc_commit, panda_commit)
  if any(type(value) is not str or _COMMIT_RE.fullmatch(value) is None for value in commits):
    raise BehaviorPipelineError("behavior replay requires full clean-build commits")
  if not controller_name.strip() or not implementation_contract.strip():
    raise BehaviorPipelineError("behavior replay implementation identity is empty")
  return ReplayCoreIdentity.compose(
    controller_name=controller_name,
    implementation_contract=implementation_contract,
    replay_input_schema_version=BEHAVIOR_REPLAY_INPUT_SCHEMA_VERSION,
    source_openpilot_commit=source_openpilot_commit,
    opendbc_commit=opendbc_commit,
    panda_commit=panda_commit,
  )


def _minimum_behavior_route_count(gate_spec: Any) -> int:
  """Smallest population giving both partitions the paired-route floor."""
  minimum = int(gate_spec.minimum_paired_route_count)
  for route_count in range(2, 10_001):
    partition = gate_spec.route_partition
    if partition.validation_route_count is not None:
      validation_count = int(partition.validation_route_count)
    else:
      validation_count = math.ceil(
        route_count * float(partition.validation_fraction),
      )
    training_count = route_count - validation_count
    if training_count >= minimum and validation_count >= minimum:
      return route_count
  raise BehaviorPipelineError("behavior route partition has no bounded viable population")


def _partition_counts(route_count: int, gate_spec: Any) -> tuple[int, int]:
  partition = gate_spec.route_partition
  if partition.validation_route_count is not None:
    validation_count = int(partition.validation_route_count)
  else:
    validation_count = math.ceil(
      route_count * float(partition.validation_fraction),
    )
  # Before the minimum population is reached the committed validation count
  # can exceed the evidence on hand.  Status reports collection progress, not
  # a fictitious partition, so its two counts still sum to the actual cohort.
  validation_count = min(route_count, validation_count)
  return route_count - validation_count, validation_count


def _transaction_replay_job_count(
  transaction: BehaviorLearningTransactionResult,
) -> int:
  """Count only replay jobs evidenced by an immutable transaction.

  A stock-retained training result deliberately skips held-out validation, so
  the theoretical maximum is not its completed-work count. Each evaluation is
  one controller/policy over its exact route partition; summing those route
  counts reconstructs the jobs that actually produced the transaction.
  """
  try:
    count = sum(
      len(evaluation.route_ids)
      for evaluation in transaction.evaluations
    )
  except (AttributeError, TypeError) as exc:
    raise BehaviorPipelineError(
      "behavior transaction replay accounting is invalid",
    ) from exc
  if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
    raise BehaviorPipelineError(
      "behavior transaction contains no replay authority",
    )
  return count


def _physical_generation(
  runtime: PersistentLearningRuntime,
) -> tuple[str | None, VehicleCalibrationProfile | None]:
  """Read the already-authenticated physical selection from this snapshot."""
  paths = runtime.artifact_paths.resolved()
  if not paths.backfill_pointer.is_file() or not paths.backfill_commit.is_file():
    return None, None
  finalization = runtime.coordinator.finalize()
  encoded = finalization.selected_profile_json
  identity = finalization.selected_profile_sha256
  if encoded is None or identity is None:
    return paths.backfill_commit.parent.name, None
  if hashlib.sha256(encoded).hexdigest() != identity:
    raise BehaviorPipelineError("physical selected-profile identity is inconsistent")
  try:
    profile = VehicleCalibrationProfile.from_json(
      encoded,
      expected_vehicle_identity=runtime.runtime_bundle.vehicle_identity,
    )
  except (TypeError, ValueError, json.JSONDecodeError) as exc:
    raise BehaviorPipelineError("physical selected profile is invalid") from exc
  if not profile.qualified:
    raise BehaviorPipelineError("physical selected profile is not fully qualified")
  return paths.backfill_commit.parent.name, profile


def _active_behavior_policy(
  *,
  params: Any,
  runtime: PersistentLearningRuntime,
  source_openpilot_commit: str,
  opendbc_commit: str,
  panda_commit: str,
) -> BehaviorPolicy | None:
  """Return only the exact artifact that could drive this current build."""
  bundle = runtime.runtime_bundle
  activation = PersistentProfileActivation(
    params,
    expected_vehicle_identity=bundle.vehicle_identity,
    expected_runtime_vehicle_identity_sha256=bundle.identity_sha256,
    expected_source_openpilot_commit=source_openpilot_commit,
    expected_opendbc_commit=opendbc_commit,
    expected_panda_commit=panda_commit,
    production_envelope_verified=(
      bundle.torque_limits.production_envelope_verified
    ),
  )
  if (
    activation.diagnostic is not ArtifactDiagnostic.OK
    or activation.active_artifact is None
    or activation.rollback_pending
  ):
    return None
  return BehaviorPolicy.from_controller_policy(
    activation.active_artifact.controller_policy,
  )


def _recorded_source(cohort: BehaviorEvidenceCohortSelection) -> BehaviorSourceIdentity:
  if not cohort.ready or not cohort.summaries:
    raise BehaviorPipelineError("behavior cohort is not ready")
  sources = tuple(
    behavior_source_identity_from_route_source(summary.source_identity)
    for summary in cohort.summaries
  )
  if len(set(sources)) != 1:
    raise BehaviorPipelineError("cohort source projection is not homogeneous")
  return sources[0]


def _generation_matches_inputs(
  generation: LoadedBehaviorGeneration,
  *,
  physical_generation_sha256: str,
  physical_profile_sha256: str,
  route_summaries: tuple[RouteEvidenceFileSummary, ...],
  recorded_source: BehaviorSourceIdentity,
  gate_spec: Any,
  segmentation_config: Any,
  stock_identity: ReplayCoreIdentity,
  modular_identity: ReplayCoreIdentity,
  accepted_policy: BehaviorPolicy | None,
) -> bool:
  route_set = tuple(sorted(
    (summary.source_identity.route_id, summary.sha256)
    for summary in route_summaries
  ))
  if (
    generation.physical_generation_sha256 != physical_generation_sha256
    or generation.physical_profile_sha256 != physical_profile_sha256
    or generation.route_evidence_sha256s != route_set
    or generation.recorded_source != recorded_source
    or generation.gate_spec.sha256 != gate_spec.sha256
    or generation.segmentation_config.sha256 != segmentation_config.sha256
  ):
    return False

  expected_policy_sha = None if accepted_policy is None else accepted_policy.sha256
  roles: dict[str, set[tuple[str, str | None]]] = {}
  try:
    for evaluation in generation.transaction.evaluations:
      identity = json.loads(evaluation.artifact_identity)
      role = identity["role"]
      core = identity["core"]
      core_json = canonical_json(core)
      roles.setdefault(role, set()).add((
        core_json,
        identity["behaviorPolicySha256"],
      ))
  except (KeyError, TypeError, ValueError, json.JSONDecodeError):
    return False
  stock_core_json = canonical_json(stock_identity.to_dict())
  modular_core_json = canonical_json(modular_identity.to_dict())
  if roles.get("exact_stock") != {(stock_core_json, None)}:
    return False
  if roles.get("currently_accepted") != {(
    stock_core_json if accepted_policy is None else modular_core_json,
    expected_policy_sha,
  )}:
    return False
  candidate_roles = roles.get("candidate")
  return bool(candidate_roles) and all(
    core_json == modular_core_json and policy_sha is not None
    for core_json, policy_sha in candidate_roles
  )


class OffroadBehaviorLearningPipeline:
  """Run or restore one complete informational behavior qualification."""

  def __init__(
    self,
    *,
    params: Any,
    status_publisher: BehaviorLearningStatusPublisher,
    provisional_dynamics: ProvisionalRackDynamics,
    source_openpilot_commit: str,
    opendbc_commit: str,
    panda_commit: str,
    abort_requested: Callable[[], bool],
    offroad_confirmed: Callable[[], bool],
    logger: Any | None = None,
  ) -> None:
    if not isinstance(provisional_dynamics, ProvisionalRackDynamics):
      raise TypeError("behavior pipeline requires provisional rack dynamics")
    if not callable(abort_requested) or not callable(offroad_confirmed):
      raise TypeError("behavior pipeline ownership guards must be callable")
    self.params = params
    self.status_publisher = status_publisher
    self.provisional_dynamics = provisional_dynamics
    self.source_openpilot_commit = source_openpilot_commit
    self.opendbc_commit = opendbc_commit
    self.panda_commit = panda_commit
    self.abort_requested = abort_requested
    self.offroad_confirmed = offroad_confirmed
    self.logger = logger

  def _log_exception(self, message: str) -> None:
    callback = getattr(self.logger, "exception", None)
    if callable(callback):
      callback(message)

  def _abort_if_needed(self) -> None:
    try:
      aborted = self.abort_requested()
      offroad = self.offroad_confirmed()
    except BaseException as exc:
      raise BehaviorPipelineAborted("behavior ownership guard failed") from exc
    if type(aborted) is not bool or type(offroad) is not bool:
      raise BehaviorPipelineAborted("behavior ownership guards must return booleans")
    if aborted or not offroad:
      raise BehaviorPipelineAborted("offroad ownership ended")

  def _status(self, state, diagnostic, *, new_operation=False, **context) -> None:
    try:
      self.status_publisher.publish(
        state,
        diagnostic,
        new_operation=new_operation,
        **context,
      )
    except Exception:
      # Display status is rebuildable and has no authority over qualification.
      self._log_exception("blatv2 behavior status publication failed")

  @staticmethod
  def _status_base(
    *,
    runtime: PersistentLearningRuntime,
    gate_spec: Any,
    segmentation_config: Any,
    physical_generation_sha256: str | None,
    physical_profile_sha256: str | None,
    recorded_source_identity_sha256: str | None,
    route_count: int,
    required_route_count: int,
    training_route_count: int,
    validation_route_count: int,
    candidate_count: int,
    total_jobs: int,
  ) -> dict[str, object]:
    return {
      "vehicle_identity": runtime.runtime_bundle.vehicle_identity,
      "runtime_vehicle_identity_sha256": runtime.runtime_bundle.identity_sha256,
      "physical_generation_sha256": physical_generation_sha256,
      "physical_profile_sha256": physical_profile_sha256,
      "recorded_source_identity_sha256": recorded_source_identity_sha256,
      "eligible_route_count": route_count,
      "required_route_count": required_route_count,
      "training_route_count": training_route_count,
      "validation_route_count": validation_route_count,
      "total_route_count": route_count,
      "total_candidate_count": candidate_count,
      "total_replay_jobs": total_jobs,
      "gate_spec_sha256": gate_spec.sha256,
      "segmentation_config_sha256": segmentation_config.sha256,
    }

  def _terminal_status(
    self,
    *,
    transaction: BehaviorLearningTransactionResult,
    status_base: dict[str, object],
    completed_jobs: int,
  ) -> None:
    finalization = transaction.finalization
    qualified = (
      transaction.qualification_disposition
      is QualificationDisposition.QUALIFIED_CANDIDATE_AVAILABLE
    )
    self._status(
      BehaviorLearningState.COMPLETE,
      (
        BehaviorLearningDiagnostic.CANDIDATE_QUALIFIED
        if qualified
        else BehaviorLearningDiagnostic.STOCK_RETAINED
      ),
      **status_base,
      completed_replay_jobs=completed_jobs,
      transaction_sha256=transaction.sha256,
      behavior_finalization_sha256=finalization.sha256,
      behavior_selection_sha256=finalization.behavior_selection_sha256,
      selected_behavior_policy_sha256=(
        finalization.final_behavior_policy_sha256
      ),
      smooth_passed=finalization.smooth_passed,
      swift_passed=finalization.swift_passed,
      strong_passed=finalization.strong_passed,
      target_materially_improved=finalization.target_materially_improved,
      qualification_disposition=(
        BehaviorQualificationDisposition.QUALIFIED_CANDIDATE_AVAILABLE
        if qualified
        else BehaviorQualificationDisposition.STOCK_RETAINED
      ),
      reasons=tuple(sorted(reason.value for reason in finalization.reasons)),
    )

  def run(self, runtime: PersistentLearningRuntime) -> BehaviorPipelineResult:
    if not isinstance(runtime, PersistentLearningRuntime):
      raise TypeError("behavior pipeline requires a persistent learning runtime")
    self._abort_if_needed()
    gate_spec = load_behavior_gate_spec()
    segmentation_config = load_behavior_segmentation_config()
    seed_controller_policy = ControllerPolicy.from_json_file(
      PROVISIONAL_CONTROLLER_POLICY_PATH,
    )
    bootstrap_policy = BehaviorPolicy.from_controller_policy(
      seed_controller_policy,
    )
    candidate_count = len(build_candidate_grid(
      gate_spec.candidate_grid.policy_grid(bootstrap_policy),
    ))
    required_routes = _minimum_behavior_route_count(gate_spec)
    physical_generation_sha256, physical_profile = _physical_generation(runtime)
    if physical_profile is None:
      base = self._status_base(
        runtime=runtime,
        gate_spec=gate_spec,
        segmentation_config=segmentation_config,
        physical_generation_sha256=physical_generation_sha256,
        physical_profile_sha256=None,
        recorded_source_identity_sha256=None,
        route_count=0,
        required_route_count=required_routes,
        training_route_count=0,
        validation_route_count=0,
        candidate_count=candidate_count,
        total_jobs=0,
      )
      self._status(
        BehaviorLearningState.WAITING_FOR_PHYSICAL_PROFILE,
        BehaviorLearningDiagnostic.PHYSICAL_PROFILE_UNQUALIFIED,
        new_operation=True,
        **base,
      )
      return BehaviorPipelineResult(
        "waiting", None, None, 0, "physical_profile_unqualified",
      )

    assert physical_generation_sha256 is not None
    profile_sha256 = hashlib.sha256(
      physical_profile.to_json().encode("utf-8"),
    ).hexdigest()
    paths = runtime.artifact_paths.resolved()
    try:
      ledger = load_ledger(
        paths,
        runtime_identity_sha256=(
          runtime.runtime_bundle.calibration_identity_sha256
        ),
      )
      store = RouteEvidenceStore(paths.root / "route_evidence_v2")
      cohort = select_homogeneous_behavior_cohort(ledger=ledger, store=store)
    except Exception as exc:
      raise BehaviorPipelineError("behavior route ledger could not be authenticated") from exc

    route_count = len(cohort.summaries)
    if not cohort.ready:
      if cohort.status == "empty":
        base = self._status_base(
          runtime=runtime,
          gate_spec=gate_spec,
          segmentation_config=segmentation_config,
          physical_generation_sha256=physical_generation_sha256,
          physical_profile_sha256=profile_sha256,
          recorded_source_identity_sha256=None,
          route_count=0,
          required_route_count=required_routes,
          training_route_count=0,
          validation_route_count=0,
          candidate_count=candidate_count,
          total_jobs=0,
        )
        self._status(
          BehaviorLearningState.WAITING_FOR_ROUTES,
          BehaviorLearningDiagnostic.INSUFFICIENT_HOMOGENEOUS_ROUTES,
          new_operation=True,
          **base,
        )
        return BehaviorPipelineResult("waiting", None, None, 0, cohort.reason)
      base = self._status_base(
        runtime=runtime,
        gate_spec=gate_spec,
        segmentation_config=segmentation_config,
        physical_generation_sha256=physical_generation_sha256,
        physical_profile_sha256=profile_sha256,
        recorded_source_identity_sha256=cohort.source_identity_sha256,
        route_count=0,
        required_route_count=required_routes,
        training_route_count=0,
        validation_route_count=0,
        candidate_count=candidate_count,
        total_jobs=0,
      )
      self._status(
        BehaviorLearningState.FAILED,
        BehaviorLearningDiagnostic.ROUTE_EVIDENCE_INVALID,
        new_operation=True,
        **base,
        qualification_disposition=BehaviorQualificationDisposition.STOCK_RETAINED,
        reasons=(cohort.reason,),
      )
      return BehaviorPipelineResult("failed", None, None, 0, cohort.reason)

    recorded_source = _recorded_source(cohort)
    if route_count < required_routes:
      training_count, validation_count = _partition_counts(
        route_count,
        gate_spec,
      )
      base = self._status_base(
        runtime=runtime,
        gate_spec=gate_spec,
        segmentation_config=segmentation_config,
        physical_generation_sha256=physical_generation_sha256,
        physical_profile_sha256=profile_sha256,
        recorded_source_identity_sha256=recorded_source.sha256,
        route_count=route_count,
        required_route_count=required_routes,
        training_route_count=max(0, training_count),
        validation_route_count=max(0, validation_count),
        candidate_count=candidate_count,
        total_jobs=0,
      )
      self._status(
        BehaviorLearningState.WAITING_FOR_ROUTES,
        BehaviorLearningDiagnostic.INSUFFICIENT_HOMOGENEOUS_ROUTES,
        new_operation=True,
        **base,
      )
      return BehaviorPipelineResult(
        "waiting", None, None, route_count,
        "insufficient_homogeneous_routes",
      )

    training_count, validation_count = _partition_counts(route_count, gate_spec)
    accepted_policy = _active_behavior_policy(
      params=self.params,
      runtime=runtime,
      source_openpilot_commit=self.source_openpilot_commit,
      opendbc_commit=self.opendbc_commit,
      panda_commit=self.panda_commit,
    )
    if accepted_policy is None:
      search_center_policy = bootstrap_policy
    else:
      search_center_policy = accepted_policy
    grid = build_candidate_grid(
      gate_spec.candidate_grid.policy_grid(search_center_policy),
    )
    candidate_count = len(grid)
    training_jobs = (candidate_count + 2) * training_count
    validation_jobs = 3 * validation_count
    authority_jobs = training_jobs + validation_jobs
    total_jobs = authority_jobs * 2
    base = self._status_base(
      runtime=runtime,
      gate_spec=gate_spec,
      segmentation_config=segmentation_config,
      physical_generation_sha256=physical_generation_sha256,
      physical_profile_sha256=profile_sha256,
      recorded_source_identity_sha256=recorded_source.sha256,
      route_count=route_count,
      required_route_count=required_routes,
      training_route_count=training_count,
      validation_route_count=validation_count,
      candidate_count=candidate_count,
      total_jobs=total_jobs,
    )

    stock_identity = reviewed_replay_core_identity(
      exact_stock=True,
      source_openpilot_commit=self.source_openpilot_commit,
      opendbc_commit=self.opendbc_commit,
      panda_commit=self.panda_commit,
    )
    modular_identity = reviewed_replay_core_identity(
      exact_stock=False,
      source_openpilot_commit=self.source_openpilot_commit,
      opendbc_commit=self.opendbc_commit,
      panda_commit=self.panda_commit,
    )
    behavior_root = paths.root / BEHAVIOR_GENERATION_DIRECTORY

    current: LoadedBehaviorGeneration | None = None
    if behavior_root.exists() or behavior_root.is_symlink():
      try:
        current = load_current_behavior_generation(behavior_root)
      except (BehaviorGenerationError, OSError):
        # An existing but unauthenticated CURRENT is evidence corruption, not
        # cache absence. Never overwrite it and pretend this is a fresh run.
        self._status(
          BehaviorLearningState.FAILED,
          BehaviorLearningDiagnostic.BEHAVIOR_PUBLISH_FAILED,
          new_operation=True,
          **base,
          qualification_disposition=BehaviorQualificationDisposition.STOCK_RETAINED,
          reasons=("current_behavior_generation_invalid",),
        )
        return BehaviorPipelineResult(
          "failed", None, None, route_count,
          "current_behavior_generation_invalid",
        )
    if current is not None and _generation_matches_inputs(
      current,
      physical_generation_sha256=physical_generation_sha256,
      physical_profile_sha256=profile_sha256,
      route_summaries=cohort.summaries,
      recorded_source=recorded_source,
      gate_spec=gate_spec,
      segmentation_config=segmentation_config,
      stock_identity=stock_identity,
      modular_identity=modular_identity,
      accepted_policy=accepted_policy,
    ):
      self._status(
        BehaviorLearningState.PREPARING,
        BehaviorLearningDiagnostic.VALIDATING_ROUTE_EVIDENCE,
        new_operation=True,
        **base,
      )
      self._terminal_status(
        transaction=current.transaction,
        status_base=base,
        completed_jobs=2 * _transaction_replay_job_count(current.transaction),
      )
      return BehaviorPipelineResult(
        "cached",
        current.generation_sha256,
        current.transaction.sha256,
        route_count,
        "behavior_generation_current",
      )

    # The retired eager transaction expanded every route into several complete
    # Python object graphs before replay.  A single measured artifact reached
    # 909,200 KiB RSS at the first decode stage alone; the four-route gate
    # cannot enter that path safely.  Descriptor authentication and cache
    # restoration are bounded, so stop here and retain stock until the
    # route-major streaming contract is implemented.  There is deliberately
    # no compressed-size or free-memory bypass: neither proves the full
    # downstream transaction safe.
    streaming_base = dict(base)
    streaming_base["total_replay_jobs"] = 0
    self._status(
      BehaviorLearningState.FAILED,
      BehaviorLearningDiagnostic.BEHAVIOR_STREAMING_REQUIRED,
      new_operation=True,
      **streaming_base,
      completed_replay_jobs=0,
      qualification_disposition=BehaviorQualificationDisposition.STOCK_RETAINED,
      reasons=("behavior_streaming_required",),
    )
    return BehaviorPipelineResult(
      "failed", None, None, route_count, "behavior_streaming_required",
    )
