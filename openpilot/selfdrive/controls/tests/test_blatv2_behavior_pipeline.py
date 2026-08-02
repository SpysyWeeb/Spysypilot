from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest  # noqa: TID251

import openpilot.selfdrive.controls.lib.blatv2.behavior_pipeline as pipeline
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
  QualificationDisposition,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  CalibrationParameters,
  CalibrationProfileNode,
  VehicleCalibrationProfile,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import BehaviorEvidenceCohortSelection
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
    self.loads: list[str] = []

  def load(self, digest: str) -> FakeArtifact:
    self.loads.append(digest)
    raise AssertionError("device behavior pipeline attempted an eager artifact load")


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
class PipelineHarness:
  runtime: FakeRuntime
  pipeline: pipeline.OffroadBehaviorLearningPipeline
  status: StatusRecorder
  store: FakeStore


def ready_cohort(count: int = 4) -> BehaviorEvidenceCohortSelection:
  return BehaviorEvidenceCohortSelection(
    status="ready",
    reason="ready",
    blocking_route_name=None,
    source_identity_sha256=RECORDED_SOURCE.sha256,
    summaries=artifacts(count),
  )


def make_harness(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  *,
  cohort: BehaviorEvidenceCohortSelection | None = None,
  physical: bool = True,
  status_fail: bool = False,
  abort_requested=lambda: False,
) -> PipelineHarness:
  selected = ready_cohort() if cohort is None else cohort
  store = FakeStore(selected.summaries)
  runtime = FakeRuntime(tmp_path, physical=physical)
  status = StatusRecorder(fail=status_fail)

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
    "behavior_source_identity_from_route_source",
    lambda _source: RECORDED_SOURCE,
  )
  monkeypatch.setattr(pipeline, "_active_behavior_policy", lambda **_kwargs: None)

  instance = pipeline.OffroadBehaviorLearningPipeline(
    params=object(),
    status_publisher=status,
    provisional_dynamics=ProvisionalRackDynamics(4000.0, 10.0, 4.0, "test"),
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
    panda_commit=PANDA_COMMIT,
    abort_requested=abort_requested,
    offroad_confirmed=lambda: True,
  )
  return PipelineHarness(runtime, instance, status, store)


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
  route_summaries = artifacts(4)
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
      (summary.source_identity.route_id, summary.sha256)
      for summary in route_summaries
    )),
    recorded_source=RECORDED_SOURCE,
    gate_spec=gate,
    segmentation_config=segmentation,
    transaction=SimpleNamespace(evaluations=evaluations),
  )
  inputs = {
    "physical_generation_sha256": GENERATION_SHA,
    "physical_profile_sha256": SHA,
    "route_summaries": route_summaries,
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
  assert harness.store.loads == []
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
  assert harness.store.loads == []
  state, diagnostic, context = harness.status.calls[-1]
  assert state is BehaviorLearningState.FAILED
  assert diagnostic is BehaviorLearningDiagnostic.ROUTE_EVIDENCE_INVALID
  assert context["qualification_disposition"] is BehaviorQualificationDisposition.STOCK_RETAINED


@pytest.mark.parametrize("route_count", [4, 128])
def test_ready_cohort_requires_streaming_and_retains_stock_without_eager_load(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  route_count: int,
) -> None:
  harness = make_harness(monkeypatch, tmp_path, cohort=ready_cohort(route_count))
  result = harness.pipeline.run(harness.runtime)
  assert result == pipeline.BehaviorPipelineResult(
    "failed", None, None, route_count, "behavior_streaming_required",
  )
  assert harness.store.loads == []
  state, diagnostic, context = harness.status.calls[-1]
  assert state is BehaviorLearningState.FAILED
  assert diagnostic is BehaviorLearningDiagnostic.BEHAVIOR_STREAMING_REQUIRED
  assert context["qualification_disposition"] is BehaviorQualificationDisposition.STOCK_RETAINED
  assert context["reasons"] == ("behavior_streaming_required",)
  assert context["completed_replay_jobs"] == 0
  assert context["total_replay_jobs"] == 0


def test_status_failure_cannot_bypass_or_change_streaming_guard(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  harness = make_harness(monkeypatch, tmp_path, status_fail=True)
  result = harness.pipeline.run(harness.runtime)
  assert result.diagnostic == "behavior_streaming_required"
  assert harness.store.loads == []


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
  # Cache validation relies only on hash-bound summary identities; no decoded
  # route artifact is constructed merely to rediscover an exact generation.
  assert harness.store.loads == []


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


def test_stale_valid_generation_requires_streaming_without_overwrite(
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
  assert result.diagnostic == "behavior_streaming_required"
  assert harness.store.loads == []


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
  assert harness.store.loads == []


def test_abort_before_descriptor_selection_stops_without_artifact_load(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  harness = make_harness(
    monkeypatch,
    tmp_path,
    abort_requested=lambda: True,
  )
  with pytest.raises(pipeline.BehaviorPipelineAborted):
    harness.pipeline.run(harness.runtime)
  assert harness.store.loads == []
