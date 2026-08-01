from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openpilot.selfdrive.controls.lib.blatv2.learner import (
  LEARNING_EVIDENCE_SCHEMA_VERSION,
  TRAIN_VALIDATION_BLOCK_SAMPLES,
  LearningSample,
  minimum_clean_support_s,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_coordinator import (
  LEARNING_COORDINATOR_ARTIFACT_SCHEMA_VERSION,
  LearningCoordinator,
  LearningLifecycleState,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  DEFAULT_SPEED_NODES_MPS,
  VehicleProfile,
  make_seed_profile,
)


DT = 0.1
TRUE_TORQUE_PER_LATACCEL = 0.32
TRUE_RACK_GAIN = 1800.0
TRUE_RACK_DAMPING = 7.0
TRUE_KINETIC_FRICTION = 0.03


def seed_profile(
  vehicle_identity: str = "coordinator-test-platform",
  *,
  torque_per_lateral_accel: float = 0.50,
  speed_nodes_mps: tuple[float, ...] = DEFAULT_SPEED_NODES_MPS,
) -> VehicleProfile:
  return make_seed_profile(
    vehicle_identity=vehicle_identity,
    torque_per_lateral_accel=torque_per_lateral_accel,
    rack_gain_deg_s2_per_torque=1000.0,
    rack_damping_per_s=3.0,
    transport_delay_s=0.12,
    static_friction_torque=0.09,
    kinetic_friction_torque=0.06,
    rack_rate_resolution_deg_s=1.0,
    speed_nodes_mps=speed_nodes_mps,
  )


def excitation_values(index: int) -> tuple[float, float, float]:
  lateral_levels = (-1.2, -0.8, -0.35, 0.25, 0.65, 1.0, 1.3)
  rate_levels = (-18.0, -11.0, -5.0, 4.0, 9.0, 15.0, 20.0, -7.0)
  acceleration_levels = (
    -140.0, -95.0, -55.0, -15.0, 20.0, 50.0,
    85.0, 120.0, 65.0, -35.0, 105.0,
  )
  return (
    lateral_levels[index % len(lateral_levels)],
    rate_levels[(index * 3) % len(rate_levels)],
    acceleration_levels[(index * 5) % len(acceleration_levels)],
  )


def measured_sample(
  speed_mps: float,
  index: int,
  *,
  engaged: bool = True,
) -> LearningSample:
  lateral_accel, rack_rate, rack_acceleration = excitation_values(index)
  applied_torque = (
    math.copysign(TRUE_KINETIC_FRICTION, rack_rate)
    - TRUE_TORQUE_PER_LATACCEL * lateral_accel
    + rack_acceleration / TRUE_RACK_GAIN
    + TRUE_RACK_DAMPING * rack_rate / TRUE_RACK_GAIN
  )
  return LearningSample(
    speed_mps=speed_mps,
    dt_s=DT,
    applied_torque=applied_torque,
    measured_lateral_accel_mps2=lateral_accel,
    rack_rate_deg_s=rack_rate,
    rack_acceleration_deg_s2=rack_acceleration,
    engaged=engaged,
    valid=True,
    steering_pressed=False,
    actuator_constrained=False,
    standstill=False,
  )


def canonical_json_bytes(payload: object) -> bytes:
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
  ).encode("utf-8")


def evidence_with_schema(encoded: bytes, schema: int) -> bytes:
  envelope = json.loads(encoded)
  envelope["payload"]["evidence_schema_version"] = schema
  payload_bytes = canonical_json_bytes(envelope["payload"])
  envelope["payload_sha256"] = hashlib.sha256(payload_bytes).hexdigest()
  return canonical_json_bytes(envelope)


def add_qualified_evidence(coordinator: LearningCoordinator) -> None:
  for node_index, speed in enumerate(DEFAULT_SPEED_NODES_MPS):
    sample_count = (
      math.ceil(minimum_clean_support_s(speed) / DT)
      + 2 * TRAIN_VALIDATION_BLOCK_SAMPLES
    )
    for sample_index in range(sample_count):
      if not coordinator.ingest(measured_sample(
        speed,
        sample_index + node_index,
      )):
        raise AssertionError("known-clean qualification sample was rejected")


class TestBLaTv2LearningCoordinator(unittest.TestCase):
  def test_lifecycle_counters_and_support_are_measured_only(self) -> None:
    profile = seed_profile()
    coordinator = LearningCoordinator(profile)
    self.assertIs(coordinator.state, LearningLifecycleState.OFFROAD)
    self.assertEqual(coordinator.ingested_sample_count, 0)
    self.assertEqual(coordinator.clean_sample_count, 0)
    self.assertEqual(coordinator.accepted_sample_count, 0)

    coordinator.transition_onroad()
    self.assertFalse(coordinator.ingest(
      measured_sample(10.0, 0, engaged=False),
    ))
    self.assertEqual(coordinator.ingested_sample_count, 1)
    self.assertEqual(coordinator.clean_sample_count, 0)
    self.assertEqual(coordinator.accepted_sample_count, 0)
    self.assertEqual(coordinator.rejected_sample_count, 1)

    self.assertTrue(coordinator.ingest(measured_sample(7.5, 1)))
    self.assertEqual(coordinator.ingested_sample_count, 2)
    self.assertEqual(coordinator.clean_sample_count, 1)
    self.assertEqual(coordinator.accepted_sample_count, 1)
    self.assertEqual(coordinator.rejected_sample_count, 1)
    diagnostics = coordinator.support_diagnostics
    self.assertEqual(len(diagnostics), len(DEFAULT_SPEED_NODES_MPS))
    self.assertAlmostEqual(diagnostics[1].clean_support_s, DT * 0.5)
    self.assertAlmostEqual(diagnostics[2].clean_support_s, DT * 0.5)
    self.assertEqual(diagnostics[0].clean_support_s, 0.0)
    self.assertGreater(diagnostics[1].minimum_clean_support_s, 0.0)
    coordinator.transition_offroad()

  def test_cross_drive_restore_matches_uninterrupted_evidence(self) -> None:
    profile = seed_profile()
    stream = tuple(measured_sample(10.0, index) for index in range(800))

    uninterrupted = LearningCoordinator(profile)
    uninterrupted.transition_onroad()
    for sample in stream[:317]:
      self.assertTrue(uninterrupted.ingest(sample))
    # Match the real route boundary without serializing/restoring. Transient
    # rack-direction continuity deliberately ends between drives.
    uninterrupted.transition_offroad()
    uninterrupted.transition_onroad()
    for sample in stream[317:]:
      self.assertTrue(uninterrupted.ingest(sample))
    uninterrupted.transition_offroad()
    uninterrupted_final = uninterrupted.finalize()

    first_drive = LearningCoordinator(profile)
    first_drive.transition_onroad()
    for sample in stream[:317]:
      self.assertTrue(first_drive.ingest(sample))
    first_drive.transition_offroad()
    first_final = first_drive.finalize()

    second_drive = LearningCoordinator(
      profile,
      first_final.evidence_bytes,
    )
    self.assertEqual(second_drive.accepted_sample_count, 0)
    second_drive.transition_onroad()
    for sample in stream[317:]:
      self.assertTrue(second_drive.ingest(sample))
    second_drive.transition_offroad()
    second_final = second_drive.finalize()

    self.assertEqual(
      second_final.evidence_bytes,
      uninterrupted_final.evidence_bytes,
    )
    self.assertEqual(
      second_final.evidence_sha256,
      uninterrupted_final.evidence_sha256,
    )
    self.assertEqual(
      second_final.manifest_bytes,
      uninterrupted_final.manifest_bytes,
    )

  def test_partial_coverage_never_emits_partial_profile(self) -> None:
    coordinator = LearningCoordinator(seed_profile())
    coordinator.transition_onroad()
    for index in range(300):
      self.assertTrue(coordinator.ingest(measured_sample(0.0, index)))
    coordinator.transition_offroad()

    finalization = coordinator.finalize()
    self.assertFalse(finalization.all_nodes_qualified)
    self.assertIsNone(finalization.candidate_profile_json)
    self.assertIsNone(finalization.candidate_profile_sha256)
    self.assertIsNone(finalization.learning_result.candidate_profile)
    manifest = json.loads(finalization.manifest_bytes)
    self.assertFalse(manifest["all_nodes_qualified"])
    self.assertIsNone(manifest["candidate_profile"])
    self.assertEqual(
      len(manifest["node_reports"]),
      len(DEFAULT_SPEED_NODES_MPS),
    )
    self.assertTrue(
      any(
        "insufficient_support" in report["reasons"]
        for report in manifest["node_reports"]
      ),
    )

    with tempfile.TemporaryDirectory() as directory:
      evidence_path = Path(directory) / "evidence.json"
      manifest_path = Path(directory) / "manifest.json"
      candidate_path = Path(directory) / "candidate.json"
      with self.assertRaisesRegex(RuntimeError, "partial"):
        coordinator.persist_finalized(
          evidence_path=evidence_path,
          manifest_path=manifest_path,
          candidate_profile_path=candidate_path,
        )
      self.assertFalse(evidence_path.exists())
      self.assertFalse(manifest_path.exists())
      self.assertFalse(candidate_path.exists())

  def test_all_node_candidate_is_canonical_complete_and_separate(self) -> None:
    seed = seed_profile()
    original_seed_json = seed.to_json()
    coordinator = LearningCoordinator(
      seed,
      candidate_provenance="synthetic coordinator validation",
    )
    coordinator.transition_onroad()
    add_qualified_evidence(coordinator)
    coordinator.transition_offroad()
    finalization = coordinator.finalize()

    self.assertTrue(finalization.all_nodes_qualified)
    self.assertIsNotNone(finalization.candidate_profile_json)
    candidate_json = finalization.candidate_profile_json
    if candidate_json is None:
      self.fail("qualified evidence did not produce candidate JSON")
    candidate = VehicleProfile.from_json(
      candidate_json,
      expected_vehicle_identity=seed.vehicle_identity,
    )
    self.assertTrue(candidate.qualified)
    expected_revision = seed.revision + 1 + sum(
      report.supported_sample_count
      for report in finalization.learning_result.node_reports
    )
    self.assertEqual(candidate.revision, expected_revision)
    self.assertIn(
      f"fit_seed_revision={seed.revision}",
      candidate.provenance,
    )
    self.assertIn(
      f"evidence_revision={expected_revision}",
      candidate.provenance,
    )
    self.assertIn(
      "synthetic coordinator validation",
      candidate.provenance,
    )
    self.assertEqual(seed.to_json(), original_seed_json)
    self.assertFalse(seed.qualified)
    self.assertEqual(
      finalization.candidate_profile_sha256,
      hashlib.sha256(candidate_json).hexdigest(),
    )

    manifest = json.loads(finalization.manifest_bytes)
    self.assertEqual(
      manifest["artifact_schema_version"],
      LEARNING_COORDINATOR_ARTIFACT_SCHEMA_VERSION,
    )
    self.assertEqual(
      manifest["evidence_schema_version"],
      LEARNING_EVIDENCE_SCHEMA_VERSION,
    )
    self.assertEqual(manifest["vehicle_identity"], seed.vehicle_identity)
    self.assertEqual(
      manifest["candidate_profile"]["profile_sha256"],
      finalization.candidate_profile_sha256,
    )
    self.assertTrue(
      all(report["qualified"] for report in manifest["node_reports"]),
    )

    with tempfile.TemporaryDirectory() as directory:
      evidence_path = Path(directory) / "evidence.json"
      manifest_path = Path(directory) / "manifest.json"
      candidate_path = Path(directory) / "candidate.json"
      persisted = coordinator.persist_finalized(
        evidence_path=evidence_path,
        manifest_path=manifest_path,
        candidate_profile_path=candidate_path,
      )
      self.assertIs(persisted, finalization)
      self.assertEqual(
        evidence_path.read_bytes(),
        finalization.evidence_bytes,
      )
      self.assertEqual(
        manifest_path.read_bytes(),
        finalization.manifest_bytes,
      )
      self.assertEqual(candidate_path.read_bytes(), candidate_json)

  def test_candidate_revision_advances_with_evidence_and_survives_restore(
    self,
  ) -> None:
    seed = seed_profile()
    first = LearningCoordinator(seed)
    first.transition_onroad()
    add_qualified_evidence(first)
    first.transition_offroad()
    first_final = first.finalize()
    first_candidate = first_final.learning_result.candidate_profile
    self.assertIsNotNone(first_candidate)

    restored = LearningCoordinator(seed, first_final.evidence_bytes)
    restored_same = restored.finalize()
    restored_candidate = restored_same.learning_result.candidate_profile
    self.assertIsNotNone(restored_candidate)
    self.assertEqual(
      restored_same.candidate_profile_sha256,
      first_final.candidate_profile_sha256,
    )
    self.assertEqual(
      restored_candidate.revision,
      first_candidate.revision,
    )

    restored.transition_onroad()
    self.assertTrue(restored.ingest(measured_sample(0.0, 100_000)))
    restored.transition_offroad()
    advanced = restored.finalize()
    advanced_candidate = advanced.learning_result.candidate_profile
    self.assertIsNotNone(advanced_candidate)
    self.assertGreater(
      advanced_candidate.revision,
      first_candidate.revision,
    )
    self.assertNotEqual(
      advanced.candidate_profile_sha256,
      first_final.candidate_profile_sha256,
    )

  def test_highway_evidence_cannot_mutate_low_speed_nodes(self) -> None:
    coordinator = LearningCoordinator(seed_profile())
    coordinator.transition_onroad()
    for index in range(300):
      self.assertTrue(coordinator.ingest(measured_sample(2.5, index)))
    low_before = coordinator.support_diagnostics[:3]
    for index in range(1000):
      self.assertTrue(coordinator.ingest(measured_sample(30.0, index)))
    low_after = coordinator.support_diagnostics[:3]
    coordinator.transition_offroad()

    self.assertEqual(low_after, low_before)
    self.assertGreater(
      coordinator.support_diagnostics[5].clean_support_s,
      0.0,
    )

  def test_finalize_is_idempotent_until_new_evidence_arrives(self) -> None:
    coordinator = LearningCoordinator(seed_profile())
    first = coordinator.finalize()
    second = coordinator.finalize()
    self.assertIs(second, first)
    self.assertEqual(second.manifest_bytes, first.manifest_bytes)
    self.assertEqual(second.evidence_bytes, first.evidence_bytes)

    coordinator.transition_onroad()
    self.assertFalse(coordinator.ingest(
      measured_sample(10.0, 0, engaged=False),
    ))
    coordinator.transition_offroad()
    after_rejection = coordinator.finalize()
    self.assertIs(after_rejection, first)

    coordinator.transition_onroad()
    self.assertTrue(coordinator.ingest(measured_sample(10.0, 1)))
    coordinator.transition_offroad()
    after_evidence = coordinator.finalize()
    self.assertIsNot(after_evidence, first)
    self.assertNotEqual(after_evidence.evidence_bytes, first.evidence_bytes)
    self.assertNotEqual(after_evidence.manifest_bytes, first.manifest_bytes)
    self.assertIs(coordinator.finalize(), after_evidence)

  def test_corrupt_mismatched_and_wrong_schema_evidence_is_rejected(self) -> None:
    profile = seed_profile("vehicle-A")
    encoded = LearningCoordinator(profile).finalize().evidence_bytes

    with self.assertRaises(ValueError):
      LearningCoordinator(profile, encoded[:-1])
    with self.assertRaisesRegex(ValueError, "different vehicle"):
      LearningCoordinator(seed_profile("vehicle-B"), encoded)
    with self.assertRaisesRegex(ValueError, "different seed profile"):
      LearningCoordinator(
        seed_profile("vehicle-A", torque_per_lateral_accel=0.51),
        encoded,
      )
    wrong_schema = evidence_with_schema(
      encoded,
      LEARNING_EVIDENCE_SCHEMA_VERSION + 1,
    )
    with self.assertRaisesRegex(ValueError, "evidence schema"):
      LearningCoordinator(profile, wrong_schema)

  def test_transition_misuse_and_onroad_finalize_are_rejected(self) -> None:
    coordinator = LearningCoordinator(seed_profile())
    with self.assertRaisesRegex(RuntimeError, "only while onroad"):
      coordinator.ingest(measured_sample(10.0, 0))
    with self.assertRaisesRegex(RuntimeError, "already offroad"):
      coordinator.transition_offroad()

    coordinator.transition_onroad()
    with self.assertRaisesRegex(RuntimeError, "already onroad"):
      coordinator.transition_onroad()
    with self.assertRaisesRegex(RuntimeError, "only while offroad"):
      coordinator.finalize()
    coordinator.transition_offroad()
    with self.assertRaisesRegex(RuntimeError, "already offroad"):
      coordinator.transition_offroad()

  def test_onroad_persist_refusal_makes_no_write_attempt(self) -> None:
    coordinator = LearningCoordinator(seed_profile())
    coordinator.transition_onroad()
    with tempfile.TemporaryDirectory() as directory:
      evidence_path = Path(directory) / "evidence.json"
      manifest_path = Path(directory) / "manifest.json"
      candidate_path = Path(directory) / "candidate.json"
      with patch(
        "openpilot.selfdrive.controls.lib.blatv2.learning_coordinator._atomic_write_bytes",
      ) as atomic_write:
        with self.assertRaisesRegex(RuntimeError, "only while offroad"):
          coordinator.persist_finalized(
            evidence_path=evidence_path,
            manifest_path=manifest_path,
            candidate_profile_path=candidate_path,
          )
        atomic_write.assert_not_called()
      self.assertFalse(evidence_path.exists())
      self.assertFalse(manifest_path.exists())
      self.assertFalse(candidate_path.exists())
