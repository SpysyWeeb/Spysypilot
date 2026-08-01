"""Offroad-only composition of physical evidence and behavioral replay.

This is the narrow production seam between four independently testable
owners: the immutable physical generation, the homogeneous route-evidence
cohort, the pure replay/selection transaction, and the immutable behavior
generation store.  It has no live-controller, approval, staging, or
activation API.  A successful result is still only an informational candidate;
stock remains the actuator until the separately reviewed approval lifecycle
accepts a complete artifact.

Every qualification is run twice from independently reloaded route artifacts.
Only byte-identical transaction documents can be published.  Exact stock is
the bootstrap baseline.  If a valid modular artifact is currently active, its
two behavioral dials become the incumbent and search center; no other part of
that artifact is learnable here.
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
  publish_behavior_generation,
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
  behavior_source_identity_from_route_artifact,
  make_behavior_route_evidence_decoder,
  make_exact_stock_behavior_replay_core,
  make_modular_behavior_replay_core,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_transaction import (
  BehaviorLearningTransactionResult,
  BehaviorReplayProgress,
  BehaviorReplayProgressPhase,
  BehaviorTransactionError,
  QualificationDisposition,
  run_behavior_learning_transaction,
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
  RouteEvidenceArtifact,
  RouteEvidenceError,
  RouteEvidenceStore,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
)


BEHAVIOR_REPLAY_WORKER_COUNT = 4
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
  core_sha256 = _sha256_json({
    "behaviorReplayInputSchemaVersion": BEHAVIOR_REPLAY_INPUT_SCHEMA_VERSION,
    "controllerName": controller_name,
    "implementationContract": implementation_contract,
    "opendbcCommit": opendbc_commit,
    "pandaCommit": panda_commit,
    "sourceOpenpilotCommit": source_openpilot_commit,
  })
  return ReplayCoreIdentity(
    controller_name=controller_name,
    core_artifact_sha256=core_sha256,
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
  if not cohort.ready or not cohort.artifacts:
    raise BehaviorPipelineError("behavior cohort is not ready")
  sources = tuple(
    behavior_source_identity_from_route_artifact(artifact)
    for artifact in cohort.artifacts
  )
  if len(set(sources)) != 1:
    raise BehaviorPipelineError("cohort source projection is not homogeneous")
  return sources[0]


def _generation_matches_inputs(
  generation: LoadedBehaviorGeneration,
  *,
  physical_generation_sha256: str,
  physical_profile_sha256: str,
  route_artifacts: tuple[RouteEvidenceArtifact, ...],
  recorded_source: BehaviorSourceIdentity,
  gate_spec: Any,
  segmentation_config: Any,
  stock_identity: ReplayCoreIdentity,
  modular_identity: ReplayCoreIdentity,
  accepted_policy: BehaviorPolicy | None,
) -> bool:
  route_set = tuple(sorted(
    (artifact.source_identity.route_id, artifact.sha256)
    for artifact in route_artifacts
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
    worker_count: int = BEHAVIOR_REPLAY_WORKER_COUNT,
    logger: Any | None = None,
    transaction_runner: Callable[..., BehaviorLearningTransactionResult] = (
      run_behavior_learning_transaction
    ),
    generation_publisher: Callable[..., str] = publish_behavior_generation,
  ) -> None:
    if not isinstance(provisional_dynamics, ProvisionalRackDynamics):
      raise TypeError("behavior pipeline requires provisional rack dynamics")
    if not callable(abort_requested) or not callable(offroad_confirmed):
      raise TypeError("behavior pipeline ownership guards must be callable")
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or not 1 <= worker_count <= 4:
      raise ValueError("behavior pipeline worker count must be in [1, 4]")
    self.params = params
    self.status_publisher = status_publisher
    self.provisional_dynamics = provisional_dynamics
    self.source_openpilot_commit = source_openpilot_commit
    self.opendbc_commit = opendbc_commit
    self.panda_commit = panda_commit
    self.abort_requested = abort_requested
    self.offroad_confirmed = offroad_confirmed
    self.worker_count = worker_count
    self.logger = logger
    self.transaction_runner = transaction_runner
    self.generation_publisher = generation_publisher

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

  def _transaction_abort_requested(self) -> bool:
    """Adapt both ownership guards to the transaction's boolean contract."""
    self._abort_if_needed()
    return False

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

    route_count = len(cohort.artifacts)
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

    stock_identity = build_replay_core_identity(
      controller_name="openpilot.LatControlTorque.exact-stock",
      implementation_contract="behavior-replay-full-stock-v1",
      source_openpilot_commit=self.source_openpilot_commit,
      opendbc_commit=self.opendbc_commit,
      panda_commit=self.panda_commit,
    )
    modular_identity = build_replay_core_identity(
      controller_name="blatv2.ModularControllerCore",
      implementation_contract="behavior-replay-modular-core-v1",
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
      route_artifacts=cohort.artifacts,
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

    self._status(
      BehaviorLearningState.PREPARING,
      BehaviorLearningDiagnostic.VALIDATING_ROUTE_EVIDENCE,
      new_operation=True,
      **base,
    )

    completed_jobs = 0

    def make_progress(authority_offset: int):
      def progress(update: BehaviorReplayProgress) -> None:
        nonlocal completed_jobs
        completed_jobs = authority_offset + update.completed_jobs
        state = (
          BehaviorLearningState.TRAINING
          if update.phase is BehaviorReplayProgressPhase.TRAINING
          else BehaviorLearningState.VALIDATING
        )
        diagnostic = (
          BehaviorLearningDiagnostic.REPLAYING_TRAINING_GRID
          if update.phase is BehaviorReplayProgressPhase.TRAINING
          else BehaviorLearningDiagnostic.REPLAYING_FROZEN_WINNER
        )
        self._status(
          state,
          diagnostic,
          **base,
          completed_replay_jobs=completed_jobs,
        )
      return progress

    def load_authority_artifacts() -> tuple[RouteEvidenceArtifact, ...]:
      artifacts = tuple(
        store.load(artifact.sha256)
        for artifact in cohort.artifacts
      )
      if tuple(artifact.canonical_bytes for artifact in artifacts) != tuple(
        artifact.canonical_bytes for artifact in cohort.artifacts
      ):
        raise BehaviorPipelineError("route evidence changed between authorities")
      return artifacts

    def make_authority_replay():
      """Construct one authority's independent executable artifacts.

      Freshly loading route bytes is not enough: decoders and controller
      adapters may own lifecycle state or caches. Reusing them would let pass
      one contaminate pass two while still producing superficially matching
      output. Each numerical authority therefore gets fresh instances.
      """
      decoder = make_behavior_route_evidence_decoder(
        provisional_dynamics=self.provisional_dynamics,
      )
      exact_stock = make_exact_stock_behavior_replay_core(
        stock_identity,
        provisional_dynamics=self.provisional_dynamics,
      )
      modular = make_modular_behavior_replay_core(
        modular_identity,
        provisional_dynamics=self.provisional_dynamics,
      )
      return decoder, exact_stock, modular

    try:
      first_decoder, first_stock, first_modular = make_authority_replay()
      first = self.transaction_runner(
        route_evidence_artifacts=load_authority_artifacts(),
        decode_route_evidence=first_decoder,
        physical_profile=physical_profile,
        accepted_policy=accepted_policy,
        search_center_policy=search_center_policy,
        exact_stock=first_stock,
        currently_accepted=(
          None if accepted_policy is None else first_modular
        ),
        candidate=first_modular,
        segmentation_config=segmentation_config,
        gate_spec=gate_spec,
        worker_count=self.worker_count,
        progress_callback=make_progress(0),
        abort_requested=self._transaction_abort_requested,
      )
      first_authority_jobs = _transaction_replay_job_count(first)
      if first_authority_jobs > authority_jobs:
        raise BehaviorPipelineError(
          "first behavior authority exceeded its replay job bound",
        )
      completed_jobs = first_authority_jobs
      self._abort_if_needed()
      second_authority_offset = first_authority_jobs
      # A fresh disk load plus fresh whole-route controller instances forms
      # the second independent numerical authority.
      second_decoder, second_stock, second_modular = make_authority_replay()
      second = self.transaction_runner(
        route_evidence_artifacts=load_authority_artifacts(),
        decode_route_evidence=second_decoder,
        physical_profile=physical_profile,
        accepted_policy=accepted_policy,
        search_center_policy=search_center_policy,
        exact_stock=second_stock,
        currently_accepted=(
          None if accepted_policy is None else second_modular
        ),
        candidate=second_modular,
        segmentation_config=segmentation_config,
        gate_spec=gate_spec,
        worker_count=self.worker_count,
        progress_callback=make_progress(second_authority_offset),
        abort_requested=self._transaction_abort_requested,
      )
      second_authority_jobs = _transaction_replay_job_count(second)
      if second_authority_jobs > authority_jobs:
        raise BehaviorPipelineError(
          "second behavior authority exceeded its replay job bound",
        )
      completed_jobs = first_authority_jobs + second_authority_jobs
      self._abort_if_needed()
      if first.to_json().encode("utf-8") != second.to_json().encode("utf-8"):
        self._status(
          BehaviorLearningState.FAILED,
          BehaviorLearningDiagnostic.REPLAY_NONDETERMINISTIC,
          **base,
          completed_replay_jobs=completed_jobs,
          qualification_disposition=BehaviorQualificationDisposition.STOCK_RETAINED,
          reasons=("independent_replay_mismatch",),
        )
        return BehaviorPipelineResult(
          "failed", None, None, route_count, "replay_nondeterministic",
        )
      self._status(
        BehaviorLearningState.PUBLISHING,
        BehaviorLearningDiagnostic.PUBLISHING_BEHAVIOR_GENERATION,
        **base,
        completed_replay_jobs=completed_jobs,
      )
      generation_sha256 = self.generation_publisher(
        behavior_root=behavior_root,
        first_authority=first,
        second_authority=second,
        physical_generation_sha256=physical_generation_sha256,
        physical_profile_sha256=profile_sha256,
        recorded_source=recorded_source,
        abort_requested=self.abort_requested,
        offroad_confirmed=self.offroad_confirmed,
      )
      self._terminal_status(
        transaction=first,
        status_base=base,
        completed_jobs=completed_jobs,
      )
      return BehaviorPipelineResult(
        "published",
        generation_sha256,
        first.sha256,
        route_count,
        first.qualification_disposition.value,
      )
    except BehaviorPipelineAborted:
      return BehaviorPipelineResult(
        "aborted", None, None, route_count, "offroad_ownership_ended",
      )
    except (BehaviorTransactionError, BehaviorGenerationError, RouteEvidenceError) as exc:
      diagnostic = (
        BehaviorLearningDiagnostic.BEHAVIOR_PUBLISH_FAILED
        if isinstance(exc, BehaviorGenerationError)
        else BehaviorLearningDiagnostic.BEHAVIOR_TRANSACTION_FAILED
      )
      self._status(
        BehaviorLearningState.FAILED,
        diagnostic,
        **base,
        completed_replay_jobs=completed_jobs,
        qualification_disposition=BehaviorQualificationDisposition.STOCK_RETAINED,
        reasons=(type(exc).__name__,),
      )
      self._log_exception("blatv2 behavior qualification failed")
      return BehaviorPipelineResult(
        "failed", None, None, route_count, diagnostic.value,
      )
    except Exception as exc:
      # Behavior qualification is an offroad informational subsystem. A
      # malformed candidate must retain stock without relabeling the already
      # committed physical generation as failed.
      self._status(
        BehaviorLearningState.FAILED,
        BehaviorLearningDiagnostic.BEHAVIOR_TRANSACTION_FAILED,
        **base,
        completed_replay_jobs=completed_jobs,
        qualification_disposition=BehaviorQualificationDisposition.STOCK_RETAINED,
        reasons=(type(exc).__name__,),
      )
      self._log_exception("blatv2 behavior qualification failed unexpectedly")
      return BehaviorPipelineResult(
        "failed", None, None, route_count,
        BehaviorLearningDiagnostic.BEHAVIOR_TRANSACTION_FAILED.value,
      )
