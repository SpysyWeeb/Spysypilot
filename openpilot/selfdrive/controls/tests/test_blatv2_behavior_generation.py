from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import canonical_json
from openpilot.selfdrive.controls.lib.blatv2.behavior_generation import (
  BehaviorGenerationError,
  load_behavior_generation,
  load_current_behavior_generation,
  publish_behavior_generation,
)
from openpilot.selfdrive.controls.tests.test_blatv2_behavior_transaction import (
  decoded_route,
  gate_spec,
  run,
  segmentation_config,
  source,
)


PHYSICAL_GENERATION_SHA256 = "9" * 64


class TestBehaviorGeneration(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    self.base = Path(self.temporary.name)
    self.root = self.base / "behavior"
    self.gate_path = self.base / "behavior_gate_spec.json"
    self.segmentation_path = self.base / "behavior_segmentation_config.json"
    self._write_configs()
    self.qualified = run(tuple(decoded_route(index) for index in range(4)))

  def tearDown(self) -> None:
    self.temporary.cleanup()

  def _write_configs(self, *, gate=None, segmentation=None) -> None:
    gate_value = gate_spec() if gate is None else gate
    segmentation_value = segmentation_config() if segmentation is None else segmentation
    self.gate_path.write_text(gate_value.to_json() + "\n", encoding="utf-8")
    self.segmentation_path.write_text(
      canonical_json(segmentation_value.to_dict()) + "\n",
      encoding="utf-8",
    )

  def _publish(
    self,
    first=None,
    second=None,
    *,
    physical_generation_sha256: str = PHYSICAL_GENERATION_SHA256,
    abort_requested=lambda: False,
    offroad_confirmed=lambda: True,
  ) -> str:
    first_value = self.qualified if first is None else first
    second_value = first_value if second is None else second
    return publish_behavior_generation(
      behavior_root=self.root,
      first_authority=first_value,
      second_authority=second_value,
      physical_generation_sha256=physical_generation_sha256,
      physical_profile_sha256=first_value.physical_profile_sha256,
      recorded_source=source(),
      abort_requested=abort_requested,
      offroad_confirmed=offroad_confirmed,
      gate_spec_path=self.gate_path,
      segmentation_config_path=self.segmentation_path,
    )

  def test_qualified_publication_and_authenticated_load(self) -> None:
    generation = self._publish()

    loaded = load_current_behavior_generation(self.root)
    direct = load_behavior_generation(self.root, generation)
    self.assertEqual(loaded, direct)
    self.assertEqual(loaded.generation_sha256, generation)
    self.assertEqual(loaded.physical_generation_sha256, PHYSICAL_GENERATION_SHA256)
    self.assertEqual(loaded.physical_profile_sha256, self.qualified.physical_profile_sha256)
    self.assertEqual(loaded.transaction.to_json(), self.qualified.to_json())
    self.assertEqual(loaded.finalization, self.qualified.finalization)
    self.assertEqual(loaded.route_evidence_sha256s, self.qualified.route_evidence_sha256s)
    self.assertEqual(loaded.recorded_source, source())
    self.assertEqual(loaded.selected_policy, self.qualified.selected_policy)
    self.assertFalse(loaded.stock_retained)
    generation_files = {
      path.name for path in (self.root / "generations" / generation).iterdir()
    }
    self.assertIn("policy.json", generation_files)
    self.assertNotIn("approved.json", generation_files)
    self.assertNotIn("activation.json", generation_files)

  def test_balanced_target_diagnostic_is_audited_with_policy(self) -> None:
    qualified = run(tuple(
      decoded_route(index, hard=index == 2)
      for index in range(4)
    ))
    self.assertFalse(qualified.stock_retained)
    self.assertFalse(qualified.finalization.target_materially_improved)

    generation = self._publish(qualified)
    loaded = load_current_behavior_generation(self.root)
    self.assertEqual(loaded.generation_sha256, generation)
    self.assertFalse(loaded.stock_retained)
    self.assertEqual(loaded.selected_policy, qualified.selected_policy)
    self.assertTrue((self.root / "generations" / generation / "policy.json").exists())

  def test_aa_mismatch_creates_nothing(self) -> None:
    changed = replace(self.qualified, physical_profile_sha256="8" * 64)

    with self.assertRaisesRegex(BehaviorGenerationError, "not byte-identical"):
      self._publish(self.qualified, changed)
    self.assertFalse(self.root.exists())

  def test_abort_leaves_current_unchanged_and_removes_staging(self) -> None:
    first_generation = self._publish()
    current_before = (self.root / "CURRENT").read_bytes()
    calls = 0

    def abort() -> bool:
      nonlocal calls
      calls += 1
      return calls >= 3

    with self.assertRaisesRegex(BehaviorGenerationError, "aborted"):
      self._publish(
        physical_generation_sha256="7" * 64,
        abort_requested=abort,
      )
    self.assertEqual((self.root / "CURRENT").read_bytes(), current_before)
    self.assertEqual(load_current_behavior_generation(self.root).generation_sha256, first_generation)
    self.assertFalse(any(
      path.name.startswith(".staging-")
      for path in (self.root / "generations").iterdir()
    ))

  def test_offroad_guard_fails_before_creating_store(self) -> None:
    with self.assertRaisesRegex(BehaviorGenerationError, "offroad"):
      self._publish(offroad_confirmed=lambda: False)
    self.assertFalse(self.root.exists())

  def test_corrupt_partial_extra_and_symlink_artifacts_are_rejected(self) -> None:
    mutators = {
      "corrupt": lambda directory: (
        directory / "transaction.json"
      ).write_bytes((directory / "transaction.json").read_bytes() + b" "),
      "partial": lambda directory: (directory / "finalization.json").unlink(),
      "extra": lambda directory: (directory / "unexpected.json").write_bytes(b"{}"),
      "symlink": self._replace_transaction_with_symlink,
    }
    for name, mutate in mutators.items():
      with self.subTest(name=name):
        if self.root.exists():
          shutil.rmtree(self.root)
        generation = self._publish()
        directory = self.root / "generations" / generation
        mutate(directory)
        with self.assertRaises(BehaviorGenerationError):
          load_current_behavior_generation(self.root)

  def _replace_transaction_with_symlink(self, directory: Path) -> None:
    transaction = directory / "transaction.json"
    target = self.base / "symlink-target.json"
    target.write_bytes(transaction.read_bytes())
    transaction.unlink()
    transaction.symlink_to(target)

  def test_corrupt_current_and_generation_hash_are_rejected(self) -> None:
    generation = self._publish()
    (self.root / "CURRENT").write_bytes(b'{"generationSha256":"bad","schemaVersion":1}')
    with self.assertRaises(BehaviorGenerationError):
      load_current_behavior_generation(self.root)

    (self.root / "CURRENT").write_bytes(canonical_json({
      "generationSha256": generation,
      "schemaVersion": 1,
    }).encode())
    commit = self.root / "generations" / generation / "commit.json"
    commit.write_bytes(commit.read_bytes() + b" ")
    with self.assertRaisesRegex(BehaviorGenerationError, "directory identity"):
      load_current_behavior_generation(self.root)

  def test_idempotent_content_address_collision_requires_identical_bytes(self) -> None:
    first = self._publish()
    second = self._publish()
    self.assertEqual(first, second)
    generations = tuple(
      path for path in (self.root / "generations").iterdir()
      if not path.name.startswith(".staging-")
    )
    self.assertEqual(tuple(path.name for path in generations), (first,))

    transaction = self.root / "generations" / first / "transaction.json"
    transaction.write_bytes(transaction.read_bytes() + b" ")
    with self.assertRaisesRegex(BehaviorGenerationError, "not byte-identical"):
      self._publish()

  def test_physical_and_configuration_identity_mismatches_publish_nothing(self) -> None:
    with self.assertRaisesRegex(BehaviorGenerationError, "physical profile"):
      publish_behavior_generation(
        behavior_root=self.root,
        first_authority=self.qualified,
        second_authority=self.qualified,
        physical_generation_sha256=PHYSICAL_GENERATION_SHA256,
        physical_profile_sha256="6" * 64,
        recorded_source=source(),
        abort_requested=lambda: False,
        offroad_confirmed=lambda: True,
        gate_spec_path=self.gate_path,
        segmentation_config_path=self.segmentation_path,
      )
    self.assertFalse(self.root.exists())

    alternate_gate = replace(gate_spec(), provenance="different committed gate")
    self._write_configs(gate=alternate_gate)
    with self.assertRaisesRegex(BehaviorGenerationError, "gate-spec"):
      self._publish()
    self.assertFalse(self.root.exists())

    self._write_configs(segmentation=replace(
      segmentation_config(),
      maximum_phase_extension_s=3.0,
    ))
    with self.assertRaisesRegex(BehaviorGenerationError, "segmentation"):
      self._publish()
    self.assertFalse(self.root.exists())

  def test_recorded_source_identity_mismatch_is_rejected(self) -> None:
    with self.assertRaisesRegex(BehaviorGenerationError, "recorded source"):
      publish_behavior_generation(
        behavior_root=self.root,
        first_authority=self.qualified,
        second_authority=self.qualified,
        physical_generation_sha256=PHYSICAL_GENERATION_SHA256,
        physical_profile_sha256=self.qualified.physical_profile_sha256,
        recorded_source=source("f"),
        abort_requested=lambda: False,
        offroad_confirmed=lambda: True,
        gate_spec_path=self.gate_path,
        segmentation_config_path=self.segmentation_path,
      )
    self.assertFalse(self.root.exists())

  def test_resealed_semantic_mismatch_is_rejected(self) -> None:
    generation = self._publish()
    old_directory = self.root / "generations" / generation
    commit = json.loads((old_directory / "commit.json").read_bytes())
    commit["physicalProfileSha256"] = "5" * 64
    encoded = canonical_json(commit).encode()
    new_generation = hashlib.sha256(encoded).hexdigest()
    new_directory = self.root / "generations" / new_generation
    shutil.copytree(old_directory, new_directory)
    (new_directory / "commit.json").write_bytes(encoded)

    with self.assertRaisesRegex(BehaviorGenerationError, "physical profile"):
      load_behavior_generation(self.root, new_generation)


if __name__ == "__main__":
  unittest.main()
