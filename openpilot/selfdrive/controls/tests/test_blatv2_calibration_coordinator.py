from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from openpilot.selfdrive.controls.lib.blatv2.approved_artifact import (
  APPROVED_ARTIFACT_PARAM,
  ApprovedArtifactReader,
  ApprovedProfileArtifact,
  ArtifactDiagnostic,
  ArtifactValidationError,
  PersistentProfileActivation,
)
from openpilot.selfdrive.controls.lib.blatv2.bootstrap import (
  ControllerSelection,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_coordinator import (
  CALIBRATION_COORDINATOR_ARTIFACT_SCHEMA_VERSION,
  CalibrationLearningCoordinator,
  CalibrationLearningLifecycleState,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_learner import (
  CalibrationSampleDisposition,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  CALIBRATION_PROFILE_SCHEMA_VERSION,
  CalibrationProfileNode,
  VehicleCalibrationProfile,
  make_calibration_seed_profile,
)
from openpilot.selfdrive.controls.lib.blatv2.learner import LearningSample


DT = 0.01


def route_sha(counter: int) -> str:
  return hashlib.sha256(f"route-{counter}".encode()).hexdigest()


class _MemoryParams:
  def __init__(self) -> None:
    self.values: dict[str, object] = {}

  def get(self, key: str, block: bool = False):
    del block
    return self.values.get(key)

  def put(self, key: str, value: object, block: bool = False) -> None:
    del block
    self.values[key] = value

  def remove(self, key: str) -> None:
    self.values.pop(key, None)


def seed_profile() -> VehicleCalibrationProfile:
  return make_calibration_seed_profile(
    vehicle_identity="calibration-coordinator-test-platform",
    torque_callback_slope=0.32,
    stock_friction_torque=0.09,
    transport_delay_s=0.12,
    rack_rate_resolution_deg_s=4.0,
  )


def measured_sample(speed_mps: float, index: int, *, engaged: bool = True) -> LearningSample:
  direction = -1.0 if index % 2 else 1.0
  return LearningSample(
    speed_mps=speed_mps,
    dt_s=DT,
    applied_torque=direction * (0.15 + 0.001 * (index % 10)),
    measured_lateral_accel_mps2=-direction * (0.3 + 0.01 * (index % 8)),
    rack_rate_deg_s=direction * 8.0,
    rack_acceleration_deg_s2=direction * 20.0,
    engaged=engaged,
    valid=True,
    steering_pressed=False,
    actuator_constrained=False,
    standstill=False,
  )


class _Reason(StrEnum):
  QUALIFIED = "qualified"
  INSUFFICIENT_SUPPORT = "insufficient_support"
  INSUFFICIENT_MOVING_EVIDENCE = "insufficient_moving_evidence"
  INSUFFICIENT_BREAKAWAY_EVIDENCE = "insufficient_breakaway_evidence"


@dataclass(frozen=True)
class _Snapshot:
  clean_support_s: float
  supported_sample_count: int
  base_support_s: float
  base_sample_count: int
  full_fit_support_s: float
  full_fit_count: int
  completed_route_count: int
  base_completed_route_count: int
  moving_support_s: float
  moving_sample_count: int
  moving_full_fit_support_s: float
  moving_full_fit_count: int
  moving_completed_route_count: int
  breakaway_support_s: float
  breakaway_sample_count: int
  breakaway_full_fit_support_s: float
  breakaway_full_fit_count: int
  breakaway_episode_completed_route_count: int
  authority_support_s: float
  authority_sample_count: int
  authority_magnitude_sample_count: int
  authority_slew_build_sample_count: int
  authority_slew_release_sample_count: int
  authority_unresolved_sample_count: int
  authority_fit_support_s: float
  authority_fit_sample_count: int
  authority_full_fit_support_s: float
  authority_full_fit_count: int
  authority_completed_route_count: int
  lateral_accel_span_mps2: float
  applied_torque_span: float
  lateral_accel_directions: int
  applied_torque_directions: int
  rack_reversals: int


@dataclass(frozen=True)
class _Report:
  node_index: int
  speed_mps: float
  qualified: bool
  reasons: tuple[_Reason, ...]
  base_support_s: float
  base_sample_count: int
  moving_support_s: float
  moving_sample_count: int
  moving_reasons: tuple[_Reason, ...]
  moving_full_fit_seed_rms: float | None
  moving_full_fit_candidate_rms: float | None
  breakaway_support_s: float
  breakaway_sample_count: int
  breakaway_reasons: tuple[_Reason, ...]
  breakaway_full_fit_seed_rms: float | None
  breakaway_full_fit_candidate_rms: float | None
  full_fit_candidate_rms: float | None
  lateral_accel_span_mps2: float
  applied_torque_span: float
  lateral_accel_directions: int
  applied_torque_directions: int
  confidence: float


class _FakeCalibrationLearner:
  """Small deterministic learner double; coordinator policy is under test."""

  def __init__(self, seed: VehicleCalibrationProfile) -> None:
    self.seed = seed
    self.speed_nodes_mps = seed.speed_nodes_mps
    self.counts = [0] * len(self.speed_nodes_mps)
    self.active_route_counter: int | None = None
    self.begun_route_counters: list[int] = []
    self.ended_route_counters: list[int] = []

  @classmethod
  def from_evidence(cls, seed: VehicleCalibrationProfile, encoded: bytes) -> _FakeCalibrationLearner:
    payload = json.loads(encoded)
    if payload["evidence_schema_version"] != 10:
      raise ValueError("evidence schema is incompatible")
    if payload["vehicle_identity"] != seed.vehicle_identity:
      raise ValueError("evidence belongs to a different vehicle")
    learner = cls(seed)
    learner.counts[:] = payload["counts"]
    return learner

  def reset_route_transients(self) -> None:
    pass

  def begin_route(
    self,
    route_identity_sha256: str,
    route_content_sha256: str | None = None,
    *,
    route_counter: int,
  ) -> None:
    if self.active_route_counter is not None:
      raise AssertionError("fake learner already owns a route")
    del route_content_sha256
    del route_identity_sha256
    self.active_route_counter = route_counter
    self.begun_route_counters.append(route_counter)

  def end_route(self) -> None:
    if self.active_route_counter is None:
      raise AssertionError("fake learner has no route to end")
    self.ended_route_counters.append(self.active_route_counter)
    self.active_route_counter = None

  def _node(self, speed: float) -> int:
    return min(range(len(self.speed_nodes_mps)), key=lambda index: abs(self.speed_nodes_mps[index] - speed))

  def add_sample(self, sample: LearningSample) -> bool:
    if self.active_route_counter is None:
      raise AssertionError("fake learner sample has no route owner")
    if not sample.clean:
      return False
    self.counts[self._node(sample.speed_mps)] += 1
    return True

  def evidence_for_node(self, node_index: int) -> _Snapshot:
    count = self.counts[node_index]
    return _Snapshot(
      clean_support_s=count * DT,
      supported_sample_count=count,
      base_support_s=count * DT,
      base_sample_count=count,
      full_fit_support_s=count * DT,
      full_fit_count=count,
      completed_route_count=int(bool(count)),
      base_completed_route_count=int(bool(count)),
      moving_support_s=count * DT,
      moving_sample_count=count,
      moving_full_fit_support_s=count * DT,
      moving_full_fit_count=count,
      moving_completed_route_count=int(bool(count)),
      breakaway_support_s=count * DT,
      breakaway_sample_count=count,
      breakaway_full_fit_support_s=count * DT,
      breakaway_full_fit_count=count,
      breakaway_episode_completed_route_count=int(bool(count)),
      authority_support_s=0.0,
      authority_sample_count=0,
      authority_magnitude_sample_count=0,
      authority_slew_build_sample_count=0,
      authority_slew_release_sample_count=0,
      authority_unresolved_sample_count=0,
      authority_fit_support_s=0.0,
      authority_fit_sample_count=0,
      authority_full_fit_support_s=0.0,
      authority_full_fit_count=0,
      authority_completed_route_count=0,
      lateral_accel_span_mps2=0.8 if count else 0.0,
      applied_torque_span=0.4 if count else 0.0,
      lateral_accel_directions=2 if count else 0,
      applied_torque_directions=2 if count else 0,
      rack_reversals=0,
    )

  def export_evidence(self) -> bytes:
    return json.dumps(
      {
        "counts": self.counts,
        "evidence_schema_version": 10,
        "vehicle_identity": self.seed.vehicle_identity,
      },
      sort_keys=True,
      separators=(",", ":"),
    ).encode()

  def qualify(self, provenance: str) -> SimpleNamespace:
    reports = []
    for index, (speed, count) in enumerate(zip(self.speed_nodes_mps, self.counts, strict=True)):
      qualified = count >= 2
      reasons = (_Reason.QUALIFIED,) if qualified else (_Reason.INSUFFICIENT_SUPPORT,)
      reports.append(
        _Report(
          node_index=index,
          speed_mps=speed,
          qualified=qualified,
          reasons=reasons,
          base_support_s=count * DT,
          base_sample_count=count,
          moving_support_s=count * DT,
          moving_sample_count=count,
          moving_reasons=(() if qualified else (_Reason.INSUFFICIENT_MOVING_EVIDENCE,)),
          moving_full_fit_seed_rms=(0.02 if qualified else None),
          moving_full_fit_candidate_rms=(0.01 if qualified else None),
          breakaway_support_s=count * DT,
          breakaway_sample_count=count,
          breakaway_reasons=(() if qualified else (_Reason.INSUFFICIENT_BREAKAWAY_EVIDENCE,)),
          breakaway_full_fit_seed_rms=(0.03 if qualified else None),
          breakaway_full_fit_candidate_rms=(0.02 if qualified else None),
          full_fit_candidate_rms=(0.03 if qualified else None),
          lateral_accel_span_mps2=(0.8 if count else 0.0),
          applied_torque_span=(0.4 if count else 0.0),
          lateral_accel_directions=(2 if count else 0),
          applied_torque_directions=(2 if count else 0),
          confidence=(1.0 if qualified else 0.0),
        )
      )
    all_qualified = all(report.qualified for report in reports)
    candidate = None
    if all_qualified:
      nodes: list[CalibrationProfileNode] = []
      for node, report in zip(self.seed.nodes, reports, strict=True):
        nodes.append(
          CalibrationProfileNode(
            speed_mps=node.speed_mps,
            parameters=replace(node.parameters, confidence=report.confidence, qualified=True),
            base_support_s=report.base_support_s,
            base_sample_count=report.base_sample_count,
            moving_support_s=report.moving_support_s,
            moving_sample_count=report.moving_sample_count,
            breakaway_support_s=report.breakaway_support_s,
            breakaway_sample_count=report.breakaway_sample_count,
            cross_fit_route_count=report.base_sample_count,
            full_fit_candidate_rms=0.03,
            breakaway_full_fit_candidate_rms=0.02,
          )
        )
      candidate = VehicleCalibrationProfile(
        vehicle_identity=self.seed.vehicle_identity,
        revision=self.seed.revision + sum(self.counts),
        provenance=provenance,
        nodes=tuple(nodes),
      )
    return SimpleNamespace(
      all_nodes_qualified=all_qualified,
      candidate_profile=candidate,
      node_reports=tuple(reports),
      interpolation_reports=(),
      contains_learned_change=all_qualified,
    )


def _fake_evidence_sha256(encoded: bytes) -> str:
  return hashlib.sha256(encoded).hexdigest()


class TestBLaTv2CalibrationCoordinator(unittest.TestCase):
  def setUp(self) -> None:
    learner_patch = patch(
      "openpilot.selfdrive.controls.lib.blatv2.calibration_coordinator.CalibrationProfileLearner",
      _FakeCalibrationLearner,
    )
    hash_patch = patch(
      "openpilot.selfdrive.controls.lib.blatv2.calibration_coordinator.calibration_evidence_sha256",
      _fake_evidence_sha256,
    )
    self.addCleanup(learner_patch.stop)
    self.addCleanup(hash_patch.stop)
    learner_patch.start()
    hash_patch.start()

  def test_lifecycle_enforces_measured_only_in_memory_learning(self) -> None:
    coordinator = CalibrationLearningCoordinator(seed_profile())
    self.assertIs(coordinator.state, CalibrationLearningLifecycleState.OFFROAD)
    self.assertFalse(hasattr(coordinator, "approve"))
    self.assertFalse(hasattr(coordinator, "activate"))
    self.assertFalse(hasattr(coordinator, "select_controller"))
    with self.assertRaisesRegex(RuntimeError, "only while onroad"):
      coordinator.ingest(measured_sample(10.0, 0))

    coordinator.transition_onroad(route_sha(0x17), route_counter=0x17)
    self.assertEqual(
      coordinator._learner.begun_route_counters,
      [0x17],
    )
    self.assertEqual(coordinator._learner.active_route_counter, 0x17)
    with self.assertRaisesRegex(TypeError, "measured-only"):
      coordinator.ingest({"desiredCurvature": 0.1})  # type: ignore[arg-type]
    self.assertFalse(coordinator.ingest(measured_sample(10.0, 0, engaged=False)))
    self.assertTrue(coordinator.ingest(measured_sample(10.0, 1)))
    self.assertEqual(coordinator.ingested_sample_count, 2)
    self.assertEqual(coordinator.clean_sample_count, 1)
    self.assertEqual(coordinator.accepted_sample_count, 1)
    with self.assertRaisesRegex(RuntimeError, "only while offroad"):
      coordinator.finalize()
    coordinator.transition_offroad()
    self.assertEqual(
      coordinator._learner.ended_route_counters,
      [0x17],
    )
    self.assertIsNone(coordinator._learner.active_route_counter)

    diagnostic = coordinator.support_diagnostics[2]
    self.assertEqual(diagnostic.base_sample_count, 1)
    self.assertEqual(diagnostic.moving_sample_count, 1)
    self.assertEqual(diagnostic.breakaway_sample_count, 1)
    self.assertEqual(diagnostic.full_fit_count, 1)
    self.assertEqual(diagnostic.completed_route_count, 1)
    self.assertEqual(diagnostic.base_completed_route_count, 1)
    self.assertEqual(diagnostic.moving_completed_route_count, 1)
    self.assertEqual(diagnostic.breakaway_episode_completed_route_count, 1)
    self.assertEqual(diagnostic.authority_sample_count, 0)
    self.assertEqual(diagnostic.authority_fit_sample_count, 0)
    self.assertEqual(diagnostic.lateral_accel_directions, 2)
    self.assertEqual(diagnostic.applied_torque_directions, 2)

  def test_finalize_is_deterministic_cached_and_restorable(self) -> None:
    first = CalibrationLearningCoordinator(seed_profile())
    first.transition_onroad(route_sha(0), route_counter=0)
    self.assertTrue(first.ingest(measured_sample(10.0, 0)))
    first.transition_offroad()
    first_final = first.finalize()
    self.assertIs(first.finalize(), first_final)

    restored = CalibrationLearningCoordinator(seed_profile(), first_final.evidence_bytes)
    restored_final = restored.finalize()
    self.assertEqual(restored_final.evidence_bytes, first_final.evidence_bytes)
    self.assertEqual(restored_final.evidence_sha256, first_final.evidence_sha256)
    self.assertEqual(restored_final.manifest_bytes, first_final.manifest_bytes)

    restored.transition_onroad(route_sha(1), route_counter=1)
    self.assertTrue(restored.ingest(measured_sample(10.0, 1)))
    restored.transition_offroad()
    advanced = restored.finalize()
    self.assertIsNot(advanced, restored_final)
    self.assertNotEqual(advanced.evidence_sha256, restored_final.evidence_sha256)

  def test_rejection_invalidates_cached_finalization_immediately(self) -> None:
    coordinator = CalibrationLearningCoordinator(seed_profile())
    initial = coordinator.finalize()
    self.assertIs(coordinator.finalize(), initial)
    coordinator.transition_onroad(route_sha(0), route_counter=0)
    generation = coordinator._evidence_generation
    self.assertFalse(coordinator.ingest(measured_sample(10.0, 0, engaged=False)))
    self.assertEqual(coordinator._evidence_generation, generation + 1)
    self.assertIsNone(coordinator._cached_finalization)
    self.assertEqual(coordinator.ingested_sample_count, 1)
    self.assertEqual(coordinator.rejected_sample_count, 1)

  def test_upstream_rejection_never_reaches_legacy_learner_evidence(self) -> None:
    coordinator = CalibrationLearningCoordinator(seed_profile())
    coordinator.transition_onroad(route_sha(0), route_counter=0)
    counts_before = tuple(coordinator._learner.counts)
    self.assertFalse(coordinator.ingest(
      measured_sample(10.0, 0),
      upstream_rejection=CalibrationSampleDisposition.LIVE_RACK_MAPPING_INVALID,
    ))
    self.assertEqual(tuple(coordinator._learner.counts), counts_before)
    self.assertEqual(coordinator.ingested_sample_count, 1)
    self.assertEqual(
      coordinator.sample_accounting.count(
        CalibrationSampleDisposition.LIVE_RACK_MAPPING_INVALID,
      ),
      1,
    )

  def test_partial_evidence_never_writes_candidate(self) -> None:
    coordinator = CalibrationLearningCoordinator(seed_profile())
    coordinator.transition_onroad(route_sha(0), route_counter=0)
    for index in range(4):
      self.assertTrue(coordinator.ingest(measured_sample(0.0, index)))
    coordinator.transition_offroad()
    finalization = coordinator.finalize()
    self.assertFalse(finalization.all_nodes_qualified)
    self.assertIsNone(finalization.candidate_profile_json)

    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      with self.assertRaisesRegex(RuntimeError, "partial"):
        coordinator.persist_finalized(
          evidence_path=root / "evidence.json",
          candidate_profile_path=root / "candidate.json",
          manifest_path=root / "manifest.json",
        )
      self.assertEqual(tuple(root.iterdir()), ())

  def test_full_candidate_has_exact_identity_and_versioned_manifest(self) -> None:
    seed = seed_profile()
    coordinator = CalibrationLearningCoordinator(seed, candidate_provenance="coordinator integration test")
    coordinator.transition_onroad(route_sha(0), route_counter=0)
    for node_index, speed in enumerate(seed.speed_nodes_mps):
      self.assertTrue(coordinator.ingest(measured_sample(speed, node_index * 2)))
      self.assertTrue(coordinator.ingest(measured_sample(speed, node_index * 2 + 1)))
    coordinator.transition_offroad()
    finalization = coordinator.finalize()
    candidate_bytes = finalization.candidate_profile_json
    self.assertIsNotNone(candidate_bytes)
    if candidate_bytes is None:
      self.fail("full evidence did not emit a calibration candidate")
    candidate = VehicleCalibrationProfile.from_json(candidate_bytes, expected_vehicle_identity=seed.vehicle_identity)
    self.assertTrue(candidate.qualified)
    self.assertEqual(finalization.candidate_profile_sha256, hashlib.sha256(candidate_bytes).hexdigest())

    manifest = json.loads(finalization.manifest_bytes)
    self.assertEqual(manifest["artifact_schema_version"], CALIBRATION_COORDINATOR_ARTIFACT_SCHEMA_VERSION)
    self.assertEqual(manifest["artifact_schema_version"], 14)
    self.assertEqual(manifest["evidence_schema_version"], 14)
    self.assertEqual(manifest["seed_profile_schema_version"], CALIBRATION_PROFILE_SCHEMA_VERSION)
    self.assertEqual(manifest["seed_profile_schema_version"], 3)
    self.assertEqual(manifest["seed_profile_sha256"], hashlib.sha256(seed.to_json().encode()).hexdigest())
    self.assertEqual(manifest["evidence_sha256"], finalization.evidence_sha256)
    self.assertEqual(manifest["candidate_profile"]["profile_sha256"], finalization.candidate_profile_sha256)
    for report in manifest["node_reports"]:
      self.assertIn("base_support_s", report)
      self.assertIn("moving_reasons", report)
      self.assertIn("moving_full_fit_seed_rms", report)
      self.assertIn("moving_full_fit_candidate_rms", report)
      self.assertIn("breakaway_reasons", report)
      self.assertIn("breakaway_full_fit_seed_rms", report)
      self.assertIn("breakaway_full_fit_candidate_rms", report)
      self.assertNotIn("rack_gain", json.dumps(report))
      self.assertNotIn("rack_damping", json.dumps(report))

    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      write_order: list[str] = []
      real_write = __import__(
        "openpilot.selfdrive.controls.lib.blatv2.calibration_coordinator",
        fromlist=["_atomic_write_bytes"],
      )._atomic_write_bytes

      def recording_write(path: str | Path, encoded: bytes) -> None:
        write_order.append(Path(path).name)
        real_write(path, encoded)

      with patch(
        "openpilot.selfdrive.controls.lib.blatv2.calibration_coordinator._atomic_write_bytes",
        side_effect=recording_write,
      ):
        coordinator.persist_finalized(
          evidence_path=root / "evidence.json",
          candidate_profile_path=root / "candidate.json",
          manifest_path=root / "manifest.json",
        )
      self.assertEqual(write_order, ["evidence.json", "candidate.json", "manifest.json"])
      self.assertEqual((root / "candidate.json").read_bytes(), candidate_bytes)

    # Calibration output is a different schema and storage boundary from an
    # externally gate-approved controller artifact. Producing and persisting a
    # complete calibration must leave the live Params surface empty.
    params = _MemoryParams()
    self.assertNotIn(APPROVED_ARTIFACT_PARAM, params.values)
    read_result = ApprovedArtifactReader(params).read(
      expected_vehicle_identity=seed.vehicle_identity,
      expected_runtime_vehicle_identity_sha256="1" * 64,
      expected_source_openpilot_commit="2" * 40,
      expected_opendbc_commit="3" * 40,
      expected_panda_commit="4" * 40,
    )
    self.assertIs(read_result.diagnostic, ArtifactDiagnostic.ABSENT)
    self.assertIsNone(read_result.artifact)

    with self.assertRaises(ArtifactValidationError) as error:
      ApprovedProfileArtifact.from_param(
        json.loads(candidate_bytes),
        expected_vehicle_identity=seed.vehicle_identity,
      )
    self.assertIs(error.exception.reason, ArtifactDiagnostic.MALFORMED)

    activation = PersistentProfileActivation(
      params,
      expected_vehicle_identity=seed.vehicle_identity,
      expected_runtime_vehicle_identity_sha256="1" * 64,
      expected_source_openpilot_commit="2" * 40,
      expected_opendbc_commit="3" * 40,
      expected_panda_commit="4" * 40,
      production_envelope_verified=True,
    )
    decision = activation.begin_engagement()
    self.assertIs(decision.selection, ControllerSelection.STOCK)
    self.assertIsNone(decision.artifact)
    self.assertNotIn(APPROVED_ARTIFACT_PARAM, params.values)

  def test_calibration_modules_have_no_params_or_activation_boundary(self) -> None:
    module_root = Path(__file__).parents[1] / "lib" / "blatv2"
    for name in (
      "calibration_profile.py",
      "calibration_learner.py",
      "calibration_coordinator.py",
    ):
      source = (module_root / name).read_text(encoding="utf-8")
      with self.subTest(module=name):
        self.assertNotIn("BLaTv2ApprovedArtifact", source)
        self.assertNotIn("approved_artifact", source)
        self.assertNotIn("openpilot.common.params", source)
        self.assertNotIn("PersistentProfileActivation", source)


class TestBLaTv2CalibrationCoordinatorRealLearner(unittest.TestCase):
  def test_real_v12_report_is_manifest_compatible_and_restorable(self) -> None:
    seed = seed_profile()
    first = CalibrationLearningCoordinator(seed).finalize()
    restored = CalibrationLearningCoordinator(
      seed,
      first.evidence_bytes,
      expected_route_commitments=(),
    ).finalize()
    self.assertEqual(restored.evidence_bytes, first.evidence_bytes)
    self.assertEqual(restored.manifest_bytes, first.manifest_bytes)
    manifest = json.loads(first.manifest_bytes)
    self.assertEqual(manifest["artifact_schema_version"], 14)
    self.assertEqual(manifest["evidence_schema_version"], 14)
    self.assertEqual(manifest["seed_profile_schema_version"], 3)
    for report in manifest["node_reports"]:
      self.assertIn("moving_reasons", report)
      self.assertIn("moving_full_fit_candidate_rms", report)
      self.assertIn("breakaway_reasons", report)
      self.assertIn("breakaway_full_fit_candidate_rms", report)
      self.assertIn("independent_route_counts", report)
      self.assertIn("cross_fit_diagnostics", report)
      self.assertIn("full_fit_diagnostic", report)
      self.assertIn("unresolved_diagnostics", report)
    self.assertEqual(manifest["interpolation_reports"], [])

  def test_real_learner_support_diagnostics_use_completed_route_counts(self) -> None:
    coordinator = CalibrationLearningCoordinator(seed_profile())
    self.assertEqual(coordinator.support_diagnostics[2].completed_route_count, 0)
    coordinator.transition_onroad(route_sha(0x62), route_counter=0x62)
    self.assertTrue(coordinator.ingest(measured_sample(10.0, 1)))
    self.assertEqual(coordinator.support_diagnostics[2].completed_route_count, 0)
    coordinator.transition_offroad()
    diagnostic = coordinator.support_diagnostics[2]
    self.assertEqual(diagnostic.completed_route_count, 1)
    self.assertEqual(
      diagnostic.base_completed_route_count
      + diagnostic.moving_completed_route_count
      + diagnostic.breakaway_episode_completed_route_count
      + diagnostic.authority_completed_route_count,
      1,
    )


if __name__ == "__main__":
  unittest.main()
