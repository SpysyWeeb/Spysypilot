from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openpilot.selfdrive.controls.lib.blatv2.learner import (
  LEARNING_EVIDENCE_SCHEMA_VERSION,
  TRAIN_VALIDATION_BLOCK_SAMPLES,
  LearningSample,
  ProfileLearner,
  learner_evidence_sha256,
  minimum_clean_support_s,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_store import (
  read_learner_evidence,
  write_learner_evidence,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  DEFAULT_SPEED_NODES_MPS,
  make_seed_profile,
)


DT = 0.1
TRUE_TORQUE_PER_LATACCEL = 0.32
TRUE_RACK_GAIN = 1800.0
TRUE_RACK_DAMPING = 7.0
KINETIC_FRICTION = 0.03


def seed_profile(
  vehicle_identity: str = "persistent-test-platform",
  *,
  torque_per_lateral_accel: float = 0.50,
  speed_nodes_mps: tuple[float, ...] = DEFAULT_SPEED_NODES_MPS,
):
  return make_seed_profile(
    vehicle_identity=vehicle_identity,
    torque_per_lateral_accel=torque_per_lateral_accel,
    rack_gain_deg_s2_per_torque=1000.0,
    rack_damping_per_s=3.0,
    transport_delay_s=0.12,
    static_friction_torque=0.09,
    kinetic_friction_torque=KINETIC_FRICTION,
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


def measured_sample(speed_mps: float, index: int) -> LearningSample:
  lateral_accel, rack_rate, rack_acceleration = excitation_values(index)
  friction = math.copysign(KINETIC_FRICTION, rack_rate)
  applied_torque = (
    friction
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
    engaged=True,
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


def resign(
  encoded: bytes,
  mutation,
) -> bytes:
  envelope = json.loads(encoded)
  mutation(envelope["payload"])
  envelope["payload_sha256"] = hashlib.sha256(
    canonical_json_bytes(envelope["payload"]),
  ).hexdigest()
  return canonical_json_bytes(envelope)


class TestBLaTv2LearningStore(unittest.TestCase):
  def test_empty_evidence_is_canonical_exact_and_explicit(self) -> None:
    learner = ProfileLearner(seed_profile())
    first = learner.export_evidence()
    second = learner.export_evidence()
    self.assertEqual(first, second)
    self.assertEqual(
      learner_evidence_sha256(first),
      hashlib.sha256(first).hexdigest(),
    )
    text = first.decode("utf-8")
    self.assertNotIn("Infinity", text)
    self.assertNotIn("NaN", text)
    self.assertIn('"empty":"positive_infinity"', text)
    self.assertIn('"empty":"negative_infinity"', text)
    for forbidden in (
      "desired",
      "candidate",
      "request",
      "tracking_error",
      "roughness",
      "burst",
    ):
      self.assertNotIn(forbidden, text)

    restored = ProfileLearner.from_evidence(
      learner.seed_profile,
      first,
    )
    self.assertEqual(restored.export_evidence(), first)
    for node_index in range(len(DEFAULT_SPEED_NODES_MPS)):
      self.assertEqual(
        restored.evidence_for_node(node_index).to_bytes(),
        learner.evidence_for_node(node_index).to_bytes(),
      )

  def test_cross_drive_continuity_is_bit_exact(self) -> None:
    profile = seed_profile()
    sample_counts = tuple(
      math.ceil(minimum_clean_support_s(speed) / DT)
      + 2 * TRAIN_VALIDATION_BLOCK_SAMPLES
      for speed in profile.speed_nodes_mps
    )
    stream = [
      measured_sample(speed, sample_index)
      for sample_index in range(max(sample_counts))
      for node_index, speed in enumerate(profile.speed_nodes_mps)
      if sample_index < sample_counts[node_index]
    ]
    split_index = len(stream) // 2

    uninterrupted = ProfileLearner(profile)
    for sample in stream:
      self.assertTrue(uninterrupted.add_sample(sample))

    cross_drive = ProfileLearner(profile)
    for sample in stream[:split_index]:
      self.assertTrue(cross_drive.add_sample(sample))
    halfway = cross_drive.export_evidence()
    cross_drive = ProfileLearner.from_evidence(profile, halfway)
    self.assertEqual(cross_drive.export_evidence(), halfway)
    for sample in stream[split_index:]:
      self.assertTrue(cross_drive.add_sample(sample))

    for node_index in range(len(profile.nodes)):
      self.assertEqual(
        cross_drive.evidence_for_node(node_index).to_bytes(),
        uninterrupted.evidence_for_node(node_index).to_bytes(),
      )
    self.assertEqual(
      cross_drive.export_evidence(),
      uninterrupted.export_evidence(),
    )
    uninterrupted_result = uninterrupted.qualify("cross-drive")
    cross_drive_result = cross_drive.qualify("cross-drive")
    self.assertEqual(cross_drive_result, uninterrupted_result)
    self.assertEqual(
      repr(cross_drive_result).encode("utf-8"),
      repr(uninterrupted_result).encode("utf-8"),
    )
    self.assertIsNotNone(cross_drive_result.candidate_profile)
    self.assertEqual(
      cross_drive_result.candidate_profile.to_json().encode("utf-8"),
      uninterrupted_result.candidate_profile.to_json().encode("utf-8"),
    )

  def test_highway_evidence_cannot_touch_low_speed_nodes(self) -> None:
    learner = ProfileLearner(seed_profile())
    for index in range(300):
      self.assertTrue(learner.add_sample(measured_sample(2.5, index)))
    low_before = tuple(
      learner.evidence_for_node(index).to_bytes() for index in (0, 1)
    )
    for index in range(5000):
      self.assertTrue(learner.add_sample(measured_sample(30.0, index)))
    low_after = tuple(
      learner.evidence_for_node(index).to_bytes() for index in (0, 1)
    )
    self.assertEqual(low_after, low_before)
    self.assertGreater(
      learner.evidence_for_node(5).clean_support_s,
      0.0,
    )

    restored = ProfileLearner.from_evidence(
      learner.seed_profile,
      learner.export_evidence(),
    )
    self.assertEqual(
      tuple(
        restored.evidence_for_node(index).to_bytes()
        for index in (0, 1)
      ),
      low_before,
    )

  def test_wrong_vehicle_seed_grid_and_schema_are_refused(self) -> None:
    profile = seed_profile("vehicle-A")
    encoded = ProfileLearner(profile).export_evidence()
    with self.assertRaisesRegex(ValueError, "different vehicle"):
      ProfileLearner.from_evidence(
        seed_profile("vehicle-B"),
        encoded,
      )
    with self.assertRaisesRegex(ValueError, "different seed profile"):
      ProfileLearner.from_evidence(
        seed_profile("vehicle-A", torque_per_lateral_accel=0.51),
        encoded,
      )
    with self.assertRaisesRegex(ValueError, "speed-node grid mismatch"):
      ProfileLearner.from_evidence(
        seed_profile(
          "vehicle-A",
          speed_nodes_mps=(0.0, 4.0, 10.0, 15.0, 20.0, 30.0),
        ),
        encoded,
      )

    wrong_evidence_schema = resign(
      encoded,
      lambda payload: payload.__setitem__(
        "evidence_schema_version",
        LEARNING_EVIDENCE_SCHEMA_VERSION + 1,
      ),
    )
    with self.assertRaisesRegex(ValueError, "evidence schema"):
      ProfileLearner.from_evidence(profile, wrong_evidence_schema)

    version_one_evidence = resign(
      encoded,
      lambda payload: payload.__setitem__(
        "evidence_schema_version",
        1,
      ),
    )
    with self.assertRaisesRegex(ValueError, "evidence schema"):
      ProfileLearner.from_evidence(profile, version_one_evidence)

    wrong_profile_schema = resign(
      encoded,
      lambda payload: payload.__setitem__(
        "profile_schema_version",
        999,
      ),
    )
    with self.assertRaisesRegex(ValueError, "profile schema"):
      ProfileLearner.from_evidence(profile, wrong_profile_schema)

    wrong_seed_hash = resign(
      encoded,
      lambda payload: payload.__setitem__(
        "seed_profile_sha256",
        "0" * 64,
      ),
    )
    with self.assertRaisesRegex(ValueError, "seed-profile hash"):
      ProfileLearner.from_evidence(profile, wrong_seed_hash)

  def test_truncation_tampering_and_malformed_payloads_are_refused(self) -> None:
    profile = seed_profile()
    learner = ProfileLearner(profile)
    learner.add_sample(measured_sample(10.0, 0))
    encoded = learner.export_evidence()
    with self.assertRaises(ValueError):
      ProfileLearner.from_evidence(profile, encoded[:-1])

    tampered_envelope = json.loads(encoded)
    tampered_envelope["payload"]["nodes"][2]["clean_support_s"] = (
      (9.0).hex()
    )
    tampered = canonical_json_bytes(tampered_envelope)
    with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
      ProfileLearner.from_evidence(profile, tampered)

    malformed_mutations = (
      lambda payload: payload.__setitem__("unknown", 1),
      lambda payload: payload.pop("vehicle_identity"),
      lambda payload: payload["nodes"][0]["training"]["normal"].pop(),
      lambda payload: payload["nodes"][0].__setitem__(
        "supported_sample_count",
        True,
      ),
      lambda payload: payload["nodes"][0].__setitem__(
        "clean_support_s",
        "inf",
      ),
    )
    for mutation in malformed_mutations:
      with self.subTest(mutation=mutation):
        malformed = resign(encoded, mutation)
        with self.assertRaises(ValueError):
          ProfileLearner.from_evidence(profile, malformed)

  def test_onroad_write_is_refused_without_touching_path(self) -> None:
    learner = ProfileLearner(seed_profile())
    with tempfile.TemporaryDirectory() as directory:
      target = Path(directory) / "evidence.json"
      target.write_bytes(b"existing")
      with self.assertRaisesRegex(RuntimeError, "only while offroad"):
        write_learner_evidence(target, learner, offroad=False)
      self.assertEqual(target.read_bytes(), b"existing")

      missing = Path(directory) / "missing.json"
      with self.assertRaisesRegex(RuntimeError, "only while offroad"):
        write_learner_evidence(missing, learner, offroad=False)
      self.assertFalse(missing.exists())

  def test_atomic_replacement_and_verified_read(self) -> None:
    profile = seed_profile()
    learner = ProfileLearner(profile)
    with tempfile.TemporaryDirectory() as directory:
      target = Path(directory) / "evidence.json"
      first_identity = write_learner_evidence(
        target,
        learner,
        offroad=True,
      )
      first = target.read_bytes()
      self.assertEqual(first_identity, learner_evidence_sha256(first))

      learner.add_sample(measured_sample(10.0, 0))
      second = learner.export_evidence()
      real_replace = os.replace
      observations = []

      def checked_replace(source, destination):
        observations.append((Path(source), Path(destination)))
        self.assertEqual(target.read_bytes(), first)
        self.assertEqual(Path(source).read_bytes(), second)
        real_replace(source, destination)

      with (
        patch(
          "openpilot.selfdrive.controls.lib.blatv2.learning_store.os.replace",
          side_effect=checked_replace,
        ) as replace_mock,
        patch(
          "openpilot.selfdrive.controls.lib.blatv2.learning_store.os.fsync",
          wraps=os.fsync,
        ) as fsync_mock,
      ):
        second_identity = write_learner_evidence(
          target,
          learner,
          offroad=True,
        )
      replace_mock.assert_called_once()
      self.assertGreaterEqual(fsync_mock.call_count, 2)
      self.assertEqual(len(observations), 1)
      self.assertEqual(target.read_bytes(), second)
      self.assertEqual(second_identity, learner_evidence_sha256(second))
      self.assertFalse(any(
        path.name.endswith(".tmp")
        for path in Path(directory).iterdir()
      ))

      restored = read_learner_evidence(target, profile)
      self.assertEqual(restored.export_evidence(), second)
      target.write_bytes(second[:-1])
      with self.assertRaises(ValueError):
        read_learner_evidence(target, profile)


if __name__ == "__main__":
  unittest.main()
