from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest  # noqa: TID251

import openpilot.selfdrive.controls.lib.blatv2.behavior_pipeline as pipeline
import openpilot.selfdrive.controls.blatv2_backfilld as backfilld
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  FinalizationReason,
  ReplayArtifactIdentity,
  ReplayRole,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorSourceIdentity,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_learning_status import (
  BehaviorLearningDiagnostic,
  BehaviorLearningState,
  BehaviorQualificationDisposition,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import BehaviorPolicy
from openpilot.selfdrive.controls.lib.blatv2.behavior_transaction import (
  BehaviorReplayProgress,
  BehaviorReplayProgressPhase,
  BehaviorTransactionError,
  QualificationDisposition,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  CalibrationParameters,
  CalibrationProfileNode,
  VehicleCalibrationProfile,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  BackfillRunResult,
  BehaviorEvidenceCohortSelection,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
)


SHA = "a" * 64
GENERATION_SHA = "b" * 64
SOURCE_COMMIT = "1" * 40
OPENDBC_COMMIT = "2" * 40
PANDA_COMMIT = "3" * 40


def qualified_profile() -> VehicleCalibrationProfile:
  parameters = CalibrationParameters(
    torque_per_lateral_accel=0.4,
    lateral_accel_offset_correction_mps2=0.0,
    kinetic_friction_torque=0.03,
    static_breakaway_torque=0.09,
    transport_delay_s=0.12,
    rack_rate_resolution_deg_s=4.0,
    confidence=1.0,
    qualified=True,
  )
  return VehicleCalibrationProfile(
    vehicle_identity="test-car",
    revision=1,
    provenance="pipeline fixture",
    nodes=tuple(
      CalibrationProfileNode(
        speed_mps=speed,
        parameters=parameters,
        base_support_s=600.0,
        base_sample_count=60_000,
        moving_support_s=300.0,
        moving_sample_count=30_000,
        breakaway_support_s=30.0,
        breakaway_sample_count=3_000,
        validation_count=10_000,
        inverse_calibration_validation_rms=0.01,
        breakaway_validation_rms=0.01,
      )
      for speed in (0.0, 30.0)
    ),
  )


RECORDED_SOURCE = BehaviorSourceIdentity(
  controller_name="recorded-controller",
  controller_artifact_sha256="4" * 64,
  source_openpilot_commit="5" * 40,
  opendbc_commit="6" * 40,
  panda_commit="7" * 40,
  evidence_schema_version=2,
)


@dataclass(frozen=True)
class FakeArtifact:
  sha256: str
  canonical_bytes: bytes
  source_identity: object


def artifacts(count: int) -> tuple[FakeArtifact, ...]:
  return tuple(
    FakeArtifact(
      sha256=f"{index + 16:064x}",
      canonical_bytes=f"route-{index}".encode(),
      source_identity=SimpleNamespace(route_id=f"route-{index}"),
    )
    for index in range(count)
  )


class FakeStore:
  def __init__(self, values: tuple[FakeArtifact, ...]) -> None:
    self.values = {value.sha256: value for value in values}
    self.loads: list[FakeArtifact] = []

  def load(self, digest: str) -> FakeArtifact:
    original = self.values[digest]
    # A load represents an independent disk reconstruction, never the cohort
    # selector's in-memory object and never the other authority's object.
    loaded = FakeArtifact(
      original.sha256,
      bytes(bytearray(original.canonical_bytes)),
      original.source_identity,
    )
    self.loads.append(loaded)
    return loaded


class FakePaths:
  def __init__(self, root: Path, *, physical: bool) -> None:
    self.root = root
    self.backfill_pointer = root / "backfill_current.json"
    self.backfill_commit = root / "backfill_generations" / GENERATION_SHA / "commit.json"
    if physical:
      self.backfill_pointer.parent.mkdir(parents=True, exist_ok=True)
      self.backfill_pointer.write_text("present")
      self.backfill_commit.parent.mkdir(parents=True, exist_ok=True)
      self.backfill_commit.write_text("present")

  def resolved(self) -> FakePaths:
    return self


class FakeRuntime:
  def __init__(self, root: Path, *, physical: bool = True) -> None:
    profile = qualified_profile()
    encoded = profile.to_json().encode()
    finalization = SimpleNamespace(
      selected_profile_json=encoded,
      selected_profile_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    self.coordinator = SimpleNamespace(finalize=lambda: finalization)
    self.runtime_bundle = SimpleNamespace(
      vehicle_identity=profile.vehicle_identity,
      identity_sha256="8" * 64,
      calibration_identity_sha256="9" * 64,
      torque_limits=SimpleNamespace(production_envelope_verified=True),
    )
    self.artifact_paths = FakePaths(root, physical=physical)


class StatusRecorder:
  def __init__(self, *, fail: bool = False) -> None:
    self.fail = fail
    self.calls: list[tuple[BehaviorLearningState, BehaviorLearningDiagnostic, dict]] = []

  def publish(self, state, diagnostic, **context):
    self.calls.append((BehaviorLearningState(state), BehaviorLearningDiagnostic(diagnostic), context))
    if self.fail:
      raise RuntimeError("display-only status failed")


class DaemonParams:
  def __init__(self) -> None:
    self.values: dict[str, object] = {
      "IsOffroad": True,
      "GitCommit": SOURCE_COMMIT,
    }

  def get_bool(self, key: str, block: bool = False):
    assert block is False
    return self.values.get(key)

  def get(self, key: str, *, block: bool):
    assert block is False
    return self.values.get(key)

  def put(self, key: str, value, *, block: bool) -> None:
    assert block is True
    self.values[key] = value

  def remove(self, key: str) -> None:
    self.values.pop(key, None)


class FakeFinalization:
  def __init__(self, *, qualified: bool) -> None:
    self.sha256 = "c" * 64
    self.behavior_selection_sha256 = "d" * 64 if qualified else None
    self.final_behavior_policy_sha256 = "e" * 64 if qualified else None
    self.smooth_passed = qualified
    self.swift_passed = qualified
    self.strong_passed = qualified
    self.target_materially_improved = qualified
    self.reasons = (
      FinalizationReason.PASSED if qualified else FinalizationReason.NO_TRAINING_WINNER,
    )


class FakeTransaction:
  def __init__(
    self,
    *,
    qualified: bool = True,
    token: str = "same",
    replay_job_count: int = 42,
  ) -> None:
    self.qualification_disposition = (
      QualificationDisposition.QUALIFIED_CANDIDATE_AVAILABLE
      if qualified
      else QualificationDisposition.STOCK_RETAINED
    )
    self.finalization = FakeFinalization(qualified=qualified)
    self.evaluations = (
      SimpleNamespace(route_ids=tuple(range(replay_job_count))),
    )
    self._encoded = f'{{"token":"{token}"}}'
    self.sha256 = hashlib.sha256(self._encoded.encode()).hexdigest()

  def to_json(self) -> str:
    return self._encoded


@dataclass
class CoreInstance:
  identity: object
  serial: int


@dataclass
class PipelineHarness:
  runtime: FakeRuntime
  pipeline: pipeline.OffroadBehaviorLearningPipeline
  status: StatusRecorder
  store: FakeStore
  transaction_calls: list[dict]
  publish_calls: list[dict]
  factory_calls: dict[str, list[object]]


def ready_cohort(count: int = 4) -> BehaviorEvidenceCohortSelection:
  return BehaviorEvidenceCohortSelection(
    status="ready",
    reason="ready",
    blocking_route_name=None,
    source_identity_sha256=RECORDED_SOURCE.sha256,
    artifacts=artifacts(count),
  )


def make_harness(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  *,
  cohort: BehaviorEvidenceCohortSelection | None = None,
  physical: bool = True,
  qualified: bool = True,
  mismatch: bool = False,
  worker_count: int = 4,
  status_fail: bool = False,
  transaction_error: Exception | None = None,
  abort_requested=lambda: False,
) -> PipelineHarness:
  selected = ready_cohort() if cohort is None else cohort
  store = FakeStore(selected.artifacts)
  runtime = FakeRuntime(tmp_path, physical=physical)
  status = StatusRecorder(fail=status_fail)
  transaction_calls: list[dict] = []
  publish_calls: list[dict] = []
  factory_calls: dict[str, list[object]] = {
    "decoder": [],
    "stock": [],
    "modular": [],
  }

  monkeypatch.setattr(pipeline, "PersistentLearningRuntime", FakeRuntime)
  monkeypatch.setattr(pipeline, "load_ledger", lambda *_args, **_kwargs: {})
  monkeypatch.setattr(pipeline, "RouteEvidenceStore", lambda _root: store)
  monkeypatch.setattr(
    pipeline,
    "select_homogeneous_behavior_cohort",
    lambda **_kwargs: selected,
  )
  monkeypatch.setattr(
    pipeline,
    "behavior_source_identity_from_route_artifact",
    lambda _artifact: RECORDED_SOURCE,
  )
  monkeypatch.setattr(pipeline, "_active_behavior_policy", lambda **_kwargs: None)

  def decoder_factory(**_kwargs):
    value = object()
    factory_calls["decoder"].append(value)
    return value

  def core_factory(kind: str):
    def make(identity, **_kwargs):
      value = CoreInstance(identity, len(factory_calls[kind]))
      factory_calls[kind].append(value)
      return value
    return make

  monkeypatch.setattr(pipeline, "make_behavior_route_evidence_decoder", decoder_factory)
  monkeypatch.setattr(pipeline, "make_exact_stock_behavior_replay_core", core_factory("stock"))
  monkeypatch.setattr(pipeline, "make_modular_behavior_replay_core", core_factory("modular"))

  def run_transaction(**kwargs):
    transaction_calls.append(kwargs)
    if transaction_error is not None:
      raise transaction_error
    gate_spec = kwargs["gate_spec"]
    route_count = len(kwargs["route_evidence_artifacts"])
    training_count, validation_count = pipeline._partition_counts(route_count, gate_spec)
    candidate_count = len(pipeline.build_candidate_grid(
      gate_spec.candidate_grid.policy_grid(kwargs["search_center_policy"]),
    ))
    training_jobs = (candidate_count + 2) * training_count
    validation_jobs = 3 * validation_count
    total_jobs = training_jobs + validation_jobs
    callback = kwargs["progress_callback"]
    callback(BehaviorReplayProgress(
      phase=BehaviorReplayProgressPhase.TRAINING,
      completed_jobs=training_jobs,
      total_jobs=total_jobs,
      phase_completed_jobs=training_jobs,
      phase_total_jobs=training_jobs,
    ))
    if qualified:
      callback(BehaviorReplayProgress(
        phase=BehaviorReplayProgressPhase.VALIDATION,
        completed_jobs=total_jobs,
        total_jobs=total_jobs,
        phase_completed_jobs=validation_jobs,
        phase_total_jobs=validation_jobs,
      ))
    token = "different" if mismatch and len(transaction_calls) == 2 else "same"
    return FakeTransaction(
      qualified=qualified,
      token=token,
      replay_job_count=(total_jobs if qualified else training_jobs),
    )

  def publish(**kwargs):
    publish_calls.append(kwargs)
    return "f" * 64

  instance = pipeline.OffroadBehaviorLearningPipeline(
    params=object(),
    status_publisher=status,
    provisional_dynamics=ProvisionalRackDynamics(4000.0, 10.0, 4.0, "test"),
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
    panda_commit=PANDA_COMMIT,
    abort_requested=abort_requested,
    offroad_confirmed=lambda: True,
    worker_count=worker_count,
    transaction_runner=run_transaction,
    generation_publisher=publish,
  )
  return PipelineHarness(
    runtime,
    instance,
    status,
    store,
    transaction_calls,
    publish_calls,
    factory_calls,
  )


def test_replay_core_identity_binds_commits_schema_and_contract() -> None:
  base = {
    "controller_name": "controller",
    "implementation_contract": "contract-v1",
    "source_openpilot_commit": SOURCE_COMMIT,
    "opendbc_commit": OPENDBC_COMMIT,
    "panda_commit": PANDA_COMMIT,
  }
  identity = pipeline.build_replay_core_identity(**base)
  assert identity.core_artifact_sha256 == pipeline.build_replay_core_identity(**base).core_artifact_sha256
  for field, value in (
    ("implementation_contract", "contract-v2"),
    ("source_openpilot_commit", "4" * 40),
    ("opendbc_commit", "5" * 40),
    ("panda_commit", "6" * 40),
  ):
    changed = dict(base)
    changed[field] = value
    assert pipeline.build_replay_core_identity(**changed).core_artifact_sha256 != identity.core_artifact_sha256
  with pytest.raises(pipeline.BehaviorPipelineError):
    pipeline.build_replay_core_identity(**(base | {"source_openpilot_commit": "short"}))


def test_minimum_population_and_partition_counts_match_committed_gate() -> None:
  gate = pipeline.load_behavior_gate_spec()
  assert pipeline._minimum_behavior_route_count(gate) == 4
  assert pipeline._partition_counts(3, gate) == (1, 2)
  assert pipeline._partition_counts(4, gate) == (2, 2)

  fractional = SimpleNamespace(
    minimum_paired_route_count=2,
    route_partition=SimpleNamespace(
      validation_route_count=None,
      validation_fraction=0.25,
    ),
  )
  assert pipeline._minimum_behavior_route_count(fractional) == 5
  assert pipeline._partition_counts(5, fractional) == (3, 2)


def test_generation_cache_identity_matches_every_authority_input() -> None:
  route_artifacts = artifacts(4)
  gate = pipeline.load_behavior_gate_spec()
  segmentation = pipeline.load_behavior_segmentation_config()
  stock = pipeline.build_replay_core_identity(
    controller_name="stock",
    implementation_contract="stock-v1",
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
    panda_commit=PANDA_COMMIT,
  )
  modular = pipeline.build_replay_core_identity(
    controller_name="modular",
    implementation_contract="modular-v1",
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
    panda_commit=PANDA_COMMIT,
  )
  candidate_policy = BehaviorPolicy(10.0, 1.0)
  evaluations = tuple(
    SimpleNamespace(artifact_identity=identity.to_json())
    for identity in (
      ReplayArtifactIdentity.compose(ReplayRole.EXACT_STOCK, stock, None),
      ReplayArtifactIdentity.compose(ReplayRole.CURRENTLY_ACCEPTED, stock, None),
      ReplayArtifactIdentity.compose(ReplayRole.CANDIDATE, modular, candidate_policy),
    )
  )
  generation = SimpleNamespace(
    physical_generation_sha256=GENERATION_SHA,
    physical_profile_sha256=SHA,
    route_evidence_sha256s=tuple(sorted(
      (artifact.source_identity.route_id, artifact.sha256)
      for artifact in route_artifacts
    )),
    recorded_source=RECORDED_SOURCE,
    gate_spec=gate,
    segmentation_config=segmentation,
    transaction=SimpleNamespace(evaluations=evaluations),
  )
  inputs = {
    "physical_generation_sha256": GENERATION_SHA,
    "physical_profile_sha256": SHA,
    "route_artifacts": route_artifacts,
    "recorded_source": RECORDED_SOURCE,
    "gate_spec": gate,
    "segmentation_config": segmentation,
    "stock_identity": stock,
    "modular_identity": modular,
    "accepted_policy": None,
  }
  assert pipeline._generation_matches_inputs(generation, **inputs)
  changed_stock = pipeline.build_replay_core_identity(
    controller_name="stock",
    implementation_contract="stock-v2",
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
    panda_commit=PANDA_COMMIT,
  )
  assert not pipeline._generation_matches_inputs(
    generation,
    **(inputs | {"stock_identity": changed_stock}),
  )
  assert not pipeline._generation_matches_inputs(
    generation,
    **(inputs | {"physical_profile_sha256": "f" * 64}),
  )


def test_unqualified_physical_profile_waits_without_route_or_replay(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  harness = make_harness(monkeypatch, tmp_path, physical=False)
  result = harness.pipeline.run(harness.runtime)
  assert result.state == "waiting"
  assert result.diagnostic == "physical_profile_unqualified"
  assert harness.transaction_calls == []
  assert harness.store.loads == []
  assert harness.status.calls[-1][:2] == (
    BehaviorLearningState.WAITING_FOR_PHYSICAL_PROFILE,
    BehaviorLearningDiagnostic.PHYSICAL_PROFILE_UNQUALIFIED,
  )


@pytest.mark.parametrize("count", [0, 3])
def test_empty_or_insufficient_cohort_waits(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  count: int,
) -> None:
  cohort = (
    BehaviorEvidenceCohortSelection("empty", "no_ingested_routes", None, None, ())
    if count == 0
    else ready_cohort(count)
  )
  harness = make_harness(monkeypatch, tmp_path, cohort=cohort)
  result = harness.pipeline.run(harness.runtime)
  assert result.state == "waiting"
  assert result.route_count == count
  assert harness.transaction_calls == []
  _, diagnostic, context = harness.status.calls[-1]
  assert diagnostic is BehaviorLearningDiagnostic.INSUFFICIENT_HOMOGENEOUS_ROUTES
  assert context["eligible_route_count"] == count
  if count:
    assert context["training_route_count"] + context["validation_route_count"] == count


def test_blocked_cohort_is_terminal_and_never_replays(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  cohort = BehaviorEvidenceCohortSelection(
    "blocked", "route_evidence_corrupt", "route-x", RECORDED_SOURCE.sha256, (),
  )
  harness = make_harness(monkeypatch, tmp_path, cohort=cohort)
  result = harness.pipeline.run(harness.runtime)
  assert result.state == "failed"
  assert harness.transaction_calls == []
  state, diagnostic, context = harness.status.calls[-1]
  assert state is BehaviorLearningState.FAILED
  assert diagnostic is BehaviorLearningDiagnostic.ROUTE_EVIDENCE_INVALID
  assert context["qualification_disposition"] is BehaviorQualificationDisposition.STOCK_RETAINED


@pytest.mark.parametrize("worker_count", [1, 4])
def test_success_runs_two_fresh_authorities_and_publishes_terminal_candidate(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  worker_count: int,
) -> None:
  harness = make_harness(monkeypatch, tmp_path, worker_count=worker_count)
  result = harness.pipeline.run(harness.runtime)
  assert result.state == "published"
  assert result.generation_sha256 == "f" * 64
  assert len(harness.transaction_calls) == 2
  assert [call["worker_count"] for call in harness.transaction_calls] == [worker_count, worker_count]
  assert all(call["abort_requested"]() is False for call in harness.transaction_calls)
  assert len(harness.store.loads) == 8
  assert all(
    first is not second
    for first, second in zip(harness.store.loads[:4], harness.store.loads[4:], strict=True)
  )
  assert len(harness.factory_calls["decoder"]) == 2
  assert len(harness.factory_calls["stock"]) == 2
  assert len(harness.factory_calls["modular"]) == 2
  assert harness.transaction_calls[0]["decode_route_evidence"] is not harness.transaction_calls[1]["decode_route_evidence"]
  assert harness.transaction_calls[0]["exact_stock"] is not harness.transaction_calls[1]["exact_stock"]
  assert harness.transaction_calls[0]["candidate"] is not harness.transaction_calls[1]["candidate"]
  assert len(harness.publish_calls) == 1
  state, diagnostic, context = harness.status.calls[-1]
  assert state is BehaviorLearningState.COMPLETE
  assert diagnostic is BehaviorLearningDiagnostic.CANDIDATE_QUALIFIED
  assert context["completed_replay_jobs"] == context["total_replay_jobs"]


def test_failed_qualification_is_published_as_stock_retained_with_skipped_validation_progress(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  harness = make_harness(monkeypatch, tmp_path, qualified=False)
  result = harness.pipeline.run(harness.runtime)
  assert result.state == "published"
  assert len(harness.publish_calls) == 1
  state, diagnostic, context = harness.status.calls[-1]
  assert state is BehaviorLearningState.COMPLETE
  assert diagnostic is BehaviorLearningDiagnostic.STOCK_RETAINED
  assert context["qualification_disposition"] is BehaviorQualificationDisposition.STOCK_RETAINED
  # Each authority completed the complete training grid, but correctly did
  # not claim the held-out work that was never run after no winner existed.
  assert context["completed_replay_jobs"] == 72
  assert context["total_replay_jobs"] == 84


def test_aa_mismatch_never_publishes(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  harness = make_harness(monkeypatch, tmp_path, mismatch=True)
  result = harness.pipeline.run(harness.runtime)
  assert result.state == "failed"
  assert result.diagnostic == "replay_nondeterministic"
  assert harness.publish_calls == []
  assert harness.status.calls[-1][1] is BehaviorLearningDiagnostic.REPLAY_NONDETERMINISTIC


def test_status_failures_cannot_change_transaction_or_publication(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  harness = make_harness(monkeypatch, tmp_path, status_fail=True)
  result = harness.pipeline.run(harness.runtime)
  assert result.state == "published"
  assert len(harness.transaction_calls) == 2
  assert len(harness.publish_calls) == 1


def test_current_matching_generation_is_a_cache_hit(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  harness = make_harness(monkeypatch, tmp_path)
  behavior_root = tmp_path / pipeline.BEHAVIOR_GENERATION_DIRECTORY
  behavior_root.mkdir()
  cached_transaction = FakeTransaction()
  current = SimpleNamespace(
    generation_sha256="0" * 64,
    transaction=cached_transaction,
  )
  monkeypatch.setattr(pipeline, "load_current_behavior_generation", lambda _root: current)
  monkeypatch.setattr(pipeline, "_generation_matches_inputs", lambda *_args, **_kwargs: True)
  result = harness.pipeline.run(harness.runtime)
  assert result.state == "cached"
  assert result.generation_sha256 == "0" * 64
  assert harness.transaction_calls == []
  assert harness.publish_calls == []
  # Cache validation relies only on hash-bound identities; replay artifacts
  # are not constructed merely to rediscover an exact current generation.
  assert all(not calls for calls in harness.factory_calls.values())


def test_cached_stock_retained_generation_reports_only_completed_training(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  harness = make_harness(monkeypatch, tmp_path)
  (tmp_path / pipeline.BEHAVIOR_GENERATION_DIRECTORY).mkdir()
  current = SimpleNamespace(
    generation_sha256="0" * 64,
    transaction=FakeTransaction(qualified=False, replay_job_count=36),
  )
  monkeypatch.setattr(pipeline, "load_current_behavior_generation", lambda _root: current)
  monkeypatch.setattr(pipeline, "_generation_matches_inputs", lambda *_args, **_kwargs: True)
  result = harness.pipeline.run(harness.runtime)
  assert result.state == "cached"
  _, diagnostic, context = harness.status.calls[-1]
  assert diagnostic is BehaviorLearningDiagnostic.STOCK_RETAINED
  assert context["completed_replay_jobs"] == 72
  assert context["total_replay_jobs"] == 84


def test_stale_valid_generation_reruns_and_republishes(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  harness = make_harness(monkeypatch, tmp_path)
  (tmp_path / pipeline.BEHAVIOR_GENERATION_DIRECTORY).mkdir()
  monkeypatch.setattr(
    pipeline,
    "load_current_behavior_generation",
    lambda _root: SimpleNamespace(transaction=FakeTransaction()),
  )
  monkeypatch.setattr(pipeline, "_generation_matches_inputs", lambda *_args, **_kwargs: False)
  result = harness.pipeline.run(harness.runtime)
  assert result.state == "published"
  assert len(harness.transaction_calls) == 2
  assert len(harness.publish_calls) == 1


def test_corrupt_existing_current_fails_closed_without_overwrite(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  harness = make_harness(monkeypatch, tmp_path)
  (tmp_path / pipeline.BEHAVIOR_GENERATION_DIRECTORY).mkdir()
  monkeypatch.setattr(
    pipeline,
    "load_current_behavior_generation",
    lambda _root: (_ for _ in ()).throw(pipeline.BehaviorGenerationError("corrupt")),
  )
  result = harness.pipeline.run(harness.runtime)
  assert result.state == "failed"
  assert result.diagnostic == "current_behavior_generation_invalid"
  assert harness.transaction_calls == []
  assert harness.publish_calls == []


def test_abort_between_authorities_never_publishes(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  checks = iter((False, True))
  harness = make_harness(
    monkeypatch,
    tmp_path,
    abort_requested=lambda: next(checks),
  )
  result = harness.pipeline.run(harness.runtime)
  assert result.state == "aborted"
  assert len(harness.transaction_calls) == 1
  assert harness.publish_calls == []


def test_transaction_ownership_guard_aborts_inside_replay_without_publication(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  ownership_lost = False
  harness = make_harness(
    monkeypatch,
    tmp_path,
    abort_requested=lambda: ownership_lost,
  )

  def abort_inside_transaction(**kwargs):
    nonlocal ownership_lost
    harness.transaction_calls.append(kwargs)
    ownership_lost = True
    kwargs["abort_requested"]()
    raise AssertionError("ownership guard did not abort transaction")

  harness.pipeline.transaction_runner = abort_inside_transaction
  result = harness.pipeline.run(harness.runtime)
  assert result.state == "aborted"
  assert len(harness.transaction_calls) == 1
  assert harness.publish_calls == []


def test_generic_transaction_failure_isolated_from_physical_result(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  harness = make_harness(
    monkeypatch,
    tmp_path,
    transaction_error=RuntimeError("candidate bug"),
  )
  physical_before = harness.runtime.coordinator.finalize().selected_profile_sha256
  result = harness.pipeline.run(harness.runtime)
  assert result.state == "failed"
  assert result.diagnostic == BehaviorLearningDiagnostic.BEHAVIOR_TRANSACTION_FAILED.value
  assert harness.runtime.coordinator.finalize().selected_profile_sha256 == physical_before
  assert harness.publish_calls == []


def test_typed_transaction_failure_is_fail_closed_and_stock_retained(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  harness = make_harness(
    monkeypatch,
    tmp_path,
    transaction_error=BehaviorTransactionError("bad replay"),
  )
  result = harness.pipeline.run(harness.runtime)
  assert result.state == "failed"
  state, diagnostic, context = harness.status.calls[-1]
  assert state is BehaviorLearningState.FAILED
  assert diagnostic is BehaviorLearningDiagnostic.BEHAVIOR_TRANSACTION_FAILED
  assert context["qualification_disposition"] is BehaviorQualificationDisposition.STOCK_RETAINED
  assert harness.publish_calls == []


def test_daemon_closes_remote_session_before_one_behavior_stage(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  params = DaemonParams()
  daemon = backfilld.BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
  )
  events: list[str] = []
  result = BackfillRunResult(publication=SimpleNamespace(), pending_logger_close=False)
  local_engine = SimpleNamespace(name="local")
  remote_engine = SimpleNamespace(run_once=lambda: result)
  session = SimpleNamespace(
    build_engine=lambda: remote_engine,
    preserve_transaction_state=lambda engine: events.append(
      "preserve" if engine is remote_engine else "wrong-engine",
    ),
    close=lambda: events.append("close"),
  )
  car_params = SimpleNamespace(to_bytes=lambda: b"car-params")
  daemon._wait_for_car_params = lambda: car_params
  daemon._build_engine = lambda _cp: local_engine
  daemon._prepare_remote = lambda *_args: session
  daemon._project_learning_status = lambda *_args: events.append("physical")
  daemon._run_behavior_learning = lambda engine: events.append(
    "behavior" if engine is local_engine else "wrong-behavior-engine",
  )

  def leave_offroad(_delay: float) -> None:
    params.values["IsOffroad"] = False

  monkeypatch.setattr(backfilld.time, "sleep", leave_offroad)
  daemon.run()
  assert events == ["physical", "preserve", "close", "behavior"]


def test_daemon_never_runs_behavior_while_logger_close_is_pending(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  params = DaemonParams()
  daemon = backfilld.BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
  )
  results = iter((
    BackfillRunResult(publication=None, pending_logger_close=True),
    BackfillRunResult(publication=SimpleNamespace(), pending_logger_close=False),
  ))
  engine = SimpleNamespace(run_once=lambda: next(results))
  car_params = SimpleNamespace(to_bytes=lambda: b"car-params")
  behavior_calls: list[object] = []
  daemon._wait_for_car_params = lambda: car_params
  daemon._build_engine = lambda _cp: engine
  daemon._prepare_remote = lambda *_args: None
  daemon._project_learning_status = lambda *_args: None
  daemon._run_behavior_learning = behavior_calls.append
  monkeypatch.setattr(backfilld, "has_pending_full_rlog", lambda *_args, **_kwargs: False)
  sleep_count = 0

  def advance(_delay: float) -> None:
    nonlocal sleep_count
    sleep_count += 1
    if sleep_count == 2:
      params.values["IsOffroad"] = False

  monkeypatch.setattr(backfilld.time, "sleep", advance)
  daemon.run()
  assert behavior_calls == [engine]


def test_onroad_transition_before_behavior_prevents_pipeline_construction(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  params = DaemonParams()
  factories: list[dict] = []

  def factory(**kwargs):
    factories.append(kwargs)
    return SimpleNamespace(run=lambda _runtime: None)

  daemon = backfilld.BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
    behavior_pipeline_factory=factory,
  )
  result = BackfillRunResult(publication=SimpleNamespace(), pending_logger_close=False)
  engine = SimpleNamespace(run_once=lambda: result)
  car_params = SimpleNamespace(to_bytes=lambda: b"car-params")
  daemon._wait_for_car_params = lambda: car_params
  daemon._build_engine = lambda _cp: engine
  daemon._prepare_remote = lambda *_args: None

  def transition(*_args) -> None:
    params.values["IsOffroad"] = False

  daemon._project_learning_status = transition
  daemon.run()
  assert factories == []


def test_behavior_construction_failure_cannot_relabel_physical_success(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  params = DaemonParams()
  physical_status = {"state": "complete", "diagnostic": "profile_selected"}
  params.values["BLaTv2LearningStatus"] = physical_status

  def fail_factory(**_kwargs):
    raise RuntimeError("behavior-only construction failure")

  daemon = backfilld.BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
    behavior_pipeline_factory=fail_factory,
  )
  monkeypatch.setattr(backfilld, "get_commit", lambda path: OPENDBC_COMMIT if "opendbc" in path else PANDA_COMMIT)
  monkeypatch.setattr(
    backfilld.ProvisionalRackDynamics,
    "from_json_file",
    lambda _path: ProvisionalRackDynamics(4000.0, 10.0, 4.0, "test"),
  )
  assert daemon._run_behavior_learning(SimpleNamespace(runtime_factory=MagicMock())) is None
  assert params.values["BLaTv2LearningStatus"] is physical_status
