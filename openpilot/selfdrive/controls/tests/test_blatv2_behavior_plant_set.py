from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import canonical_json
from openpilot.selfdrive.controls.lib.blatv2.behavior_plant_set import (
  ROBUST_PLANT_SET_DOMAIN,
  RobustPlantSetError,
  load_robust_plant_set,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay_authority import ReviewedReplaySource
from openpilot.selfdrive.controls.lib.blatv2.behavior_partition import (
  FrozenBehaviorPartition,
  build_behavior_partition,
)
from openpilot.selfdrive.controls.lib.blatv2.counterfactual_plant import CounterfactualPlantMember


def source() -> ReviewedReplaySource:
  return ReviewedReplaySource(
    source_openpilot_commit="1" * 40,
    opendbc_commit="2" * 40,
    panda_commit="3" * 40,
    source_composition_sha256="4" * 64,
    runtime_identity_sha256="5" * 64,
    module_closure_sha256="6" * 64,
  )


def member(gain: float) -> CounterfactualPlantMember:
  return CounterfactualPlantMember.create(
    rack_gain_deg_s2_per_torque=gain,
    rack_damping_per_s=10.0,
    delay_offset_s=0.0,
    unresolved_load_torque=0.0,
  )


def wire(value: CounterfactualPlantMember) -> dict[str, object]:
  return {
    "delayOffsetS": value.delay_offset_s.hex(),
    "loadUncertaintyTorque": value.unresolved_load_torque.hex(),
    "memberId": value.member_id,
    "rackDampingPerS": value.rack_damping_per_s.hex(),
    "rackGainDegS2PerTorque": value.rack_gain_deg_s2_per_torque.hex(),
  }


def partition() -> FrozenBehaviorPartition:
  return build_behavior_partition(
    tuple(
      (
        f"{index:08x}--{index:010x}",
        hashlib.sha256(f"partition-route-{index}".encode()).hexdigest(),
      )
      for index in range(1, 201)
    ),
    seed_identity_sha256="9" * 64,
    minimum_route_count=1,
  )


def payload() -> dict[str, object]:
  frozen_partition = partition()
  members = sorted((member(2000.0), member(4000.0)), key=lambda value: value.member_id)
  member_wire = [wire(value) for value in members]
  member_ids = [value.member_id for value in members]
  return {
    "activationEligible": False,
    "importManifestSha256": "7" * 64,
    "memberSetSha256": hashlib.sha256(
      b"blatv2-robust-member-set-v1\0" + canonical_json(member_wire).encode(),
    ).hexdigest(),
    "members": member_wire,
    "membershipFrozenOn": "training",
    "moduleClosureSha256": "6" * 64,
    "partition": frozen_partition.to_dict(),
    "partitionExclusions": [],
    "partitionReceiptSha256": "8" * 64,
    "partitionSha256": frozen_partition.sha256,
    "physicalGenerationSha256": "a" * 64,
    "physicalModuleClosureSha256": "b" * 64,
    "physicalProfileSha256": "c" * 64,
    "reviewedReplaySource": source().to_dict(),
    "schemaVersion": 1,
    "status": "qualified",
    "testFalsificationPassed": True,
    "testMemberIds": member_ids,
    "trainingMemberIds": member_ids,
    "transientModuleClosureSha256": "d" * 64,
    "transientReportSha256": "e" * 64,
    "transientRulesSha256": "f" * 64,
    "validationFalsificationPassed": True,
    "validationMemberIds": member_ids,
  }


def persist(root: Path, value: dict[str, object]) -> Path:
  encoded = canonical_json(value).encode()
  identity = hashlib.sha256(ROBUST_PLANT_SET_DOMAIN + encoded).hexdigest()
  directory = root / identity
  directory.mkdir()
  path = directory / "report.json"
  path.write_bytes(encoded)
  os.chmod(path, 0o400)
  return path


def load(path: Path):
  return load_robust_plant_set(
    path,
    import_manifest_sha256="7" * 64,
    partition_receipt_sha256="8" * 64,
    partition=partition(),
    physical_generation_sha256="a" * 64,
    physical_module_closure_sha256="b" * 64,
    physical_profile_sha256="c" * 64,
    replay_source=source(),
  )


class TestRobustPlantSet(unittest.TestCase):
  def test_loads_exact_frozen_multi_member_set(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      result = load(persist(Path(directory), payload()))
    self.assertEqual(len(result.members), 2)
    self.assertEqual(result.member_ids, tuple(sorted(result.member_ids)))

  def test_singleton_and_held_out_member_tamper_fail(self) -> None:
    singleton = payload()
    singleton["members"] = singleton["members"][:1]
    singleton["trainingMemberIds"] = singleton["trainingMemberIds"][:1]
    singleton["validationMemberIds"] = singleton["validationMemberIds"][:1]
    singleton["testMemberIds"] = singleton["testMemberIds"][:1]
    singleton["memberSetSha256"] = hashlib.sha256(
      b"blatv2-robust-member-set-v1\0" + canonical_json(singleton["members"]).encode(),
    ).hexdigest()
    with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
      RobustPlantSetError, "outside its bound",
    ):
      load(persist(Path(directory), singleton))

    changed = payload()
    changed["testMemberIds"] = changed["testMemberIds"][:-1]
    with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
      RobustPlantSetError, "changed across held-out",
    ):
      load(persist(Path(directory), changed))

  def test_partition_and_source_provenance_mismatch_fail(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      path = persist(Path(directory), payload())
      with self.assertRaisesRegex(RobustPlantSetError, "another experiment"):
        load_robust_plant_set(
          path,
          import_manifest_sha256="0" * 64,
          partition_receipt_sha256="8" * 64,
          partition=partition(),
          physical_generation_sha256="a" * 64,
          physical_module_closure_sha256="b" * 64,
          physical_profile_sha256="c" * 64,
          replay_source=source(),
        )
