"""Persisted robust plant-set authority for offline behavior training.

The trainer publishes this receipt only after fitting transient members on the
global TRAINING split and falsifying that unchanged set on VALIDATION and TEST.
The behavior trainer accepts the immutable receipt, never caller-constructed
member objects, and evaluates every controller against every listed member.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import canonical_json
from openpilot.selfdrive.controls.lib.blatv2.behavior_partition import (
  BehaviorPartitionError,
  FrozenBehaviorPartition,
  frozen_behavior_partition_from_dict,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay_authority import ReviewedReplaySource
from openpilot.selfdrive.controls.lib.blatv2.counterfactual_plant import CounterfactualPlantMember


ROBUST_PLANT_SET_SCHEMA_VERSION = 1
ROBUST_PLANT_SET_DOMAIN = b"blatv2-robust-plant-set-v1\0"
ROBUST_PLANT_SET_MINIMUM_MEMBERS = 2
ROBUST_PLANT_SET_MAXIMUM_MEMBERS = 64
_MAXIMUM_RECEIPT_BYTES = 8 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class RobustPlantSetError(RuntimeError):
  """A persisted identified set cannot authorize robust replay."""


def _sha256(value: object, description: str) -> str:
  if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
    raise RobustPlantSetError(f"{description} is not a lowercase SHA-256")
  return value


def _float_hex(value: object, description: str) -> float:
  if type(value) is not str:
    raise RobustPlantSetError(f"{description} is not a float-hex string")
  try:
    parsed = float.fromhex(value)
  except ValueError as error:
    raise RobustPlantSetError(f"{description} is not a float-hex string") from error
  if not math.isfinite(parsed) or parsed.hex() != value:
    raise RobustPlantSetError(f"{description} is not canonical finite float-hex")
  return parsed


def _read_immutable_canonical(path: Path) -> tuple[dict[str, object], bytes]:
  if not isinstance(path, Path) or not path.is_absolute() or path.name != "report.json":
    raise RobustPlantSetError("robust plant-set path must select an absolute report.json")
  try:
    parent = path.parent
    metadata = path.lstat()
    parent_metadata = parent.lstat()
    if (
      path.is_symlink()
      or parent.is_symlink()
      or not stat.S_ISREG(metadata.st_mode)
      or not stat.S_ISDIR(parent_metadata.st_mode)
      or metadata.st_mode & 0o222
      or metadata.st_size <= 0
      or metadata.st_size > _MAXIMUM_RECEIPT_BYTES
      or parent.resolve(strict=True) != parent
      or _SHA256_RE.fullmatch(parent.name) is None
      or {entry.name for entry in parent.iterdir()} != {"report.json"}
    ):
      raise RobustPlantSetError("robust plant-set receipt is not immutable and content-addressed")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
      before = os.fstat(descriptor)
      chunks: list[bytes] = []
      size = 0
      while block := os.read(descriptor, 1024 * 1024):
        size += len(block)
        if size > _MAXIMUM_RECEIPT_BYTES:
          raise RobustPlantSetError("robust plant-set receipt exceeds its size bound")
        chunks.append(block)
      after = os.fstat(descriptor)
    finally:
      os.close(descriptor)
  except RobustPlantSetError:
    raise
  except OSError as error:
    raise RobustPlantSetError("robust plant-set receipt is unavailable") from error
  if (
    (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    or before.st_size != metadata.st_size
  ):
    raise RobustPlantSetError("robust plant-set receipt changed while reading")
  encoded = b"".join(chunks)
  try:
    value = json.loads(encoded)
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise RobustPlantSetError("robust plant-set receipt is not JSON") from error
  if type(value) is not dict or encoded != canonical_json(value).encode("utf-8"):
    raise RobustPlantSetError("robust plant-set receipt is not canonical JSON")
  return value, encoded


@dataclass(frozen=True, slots=True)
class RobustPlantSet:
  receipt_sha256: str
  import_manifest_sha256: str
  partition_receipt_sha256: str
  partition_sha256: str
  physical_generation_sha256: str
  physical_profile_sha256: str
  physical_module_closure_sha256: str
  transient_report_sha256: str
  transient_rules_sha256: str
  transient_module_closure_sha256: str
  members: tuple[CounterfactualPlantMember, ...]

  def __post_init__(self) -> None:
    for name in (
      "receipt_sha256", "import_manifest_sha256", "partition_receipt_sha256",
      "partition_sha256", "physical_generation_sha256", "physical_profile_sha256",
      "physical_module_closure_sha256",
      "transient_report_sha256", "transient_rules_sha256",
      "transient_module_closure_sha256",
    ):
      _sha256(getattr(self, name), f"robust plant set {name}")
    member_ids = tuple(member.member_id for member in self.members)
    if (
      not ROBUST_PLANT_SET_MINIMUM_MEMBERS
      <= len(self.members)
      <= ROBUST_PLANT_SET_MAXIMUM_MEMBERS
      or member_ids != tuple(sorted(set(member_ids)))
    ):
      raise ValueError("robust plant set is empty, singleton, duplicate, or over its bound")

  @property
  def member_ids(self) -> tuple[str, ...]:
    return tuple(member.member_id for member in self.members)


def load_robust_plant_set(
  path: Path,
  *,
  import_manifest_sha256: str,
  partition_receipt_sha256: str,
  partition: FrozenBehaviorPartition,
  physical_generation_sha256: str,
  physical_profile_sha256: str,
  physical_module_closure_sha256: str,
  replay_source: ReviewedReplaySource,
) -> RobustPlantSet:
  """Authenticate one frozen member population and all held-out verdicts."""
  payload, encoded = _read_immutable_canonical(path)
  keys = {
    "activationEligible",
    "importManifestSha256",
    "memberSetSha256",
    "members",
    "membershipFrozenOn",
    "moduleClosureSha256",
    "partition",
    "partitionExclusions",
    "partitionReceiptSha256",
    "partitionSha256",
    "physicalGenerationSha256",
    "physicalModuleClosureSha256",
    "physicalProfileSha256",
    "reviewedReplaySource",
    "schemaVersion",
    "status",
    "testFalsificationPassed",
    "testMemberIds",
    "trainingMemberIds",
    "transientModuleClosureSha256",
    "transientReportSha256",
    "transientRulesSha256",
    "validationFalsificationPassed",
    "validationMemberIds",
  }
  if type(payload) is not dict or set(payload) != keys:
    raise RobustPlantSetError("robust plant-set schema is malformed")
  if (
    payload["schemaVersion"] != ROBUST_PLANT_SET_SCHEMA_VERSION
    or payload["activationEligible"] is not False
    or payload["status"] != "qualified"
    or payload["membershipFrozenOn"] != "training"
    or payload["validationFalsificationPassed"] is not True
    or payload["testFalsificationPassed"] is not True
  ):
    raise RobustPlantSetError("robust plant set has not passed unchanged held-out falsification")
  receipt_sha256 = hashlib.sha256(ROBUST_PLANT_SET_DOMAIN + encoded).hexdigest()
  if path.parent.name != receipt_sha256:
    raise RobustPlantSetError("robust plant-set content address differs")
  for name in (
    "importManifestSha256", "memberSetSha256", "moduleClosureSha256",
    "partitionReceiptSha256", "partitionSha256", "physicalGenerationSha256",
    "physicalModuleClosureSha256", "physicalProfileSha256", "transientModuleClosureSha256",
    "transientReportSha256", "transientRulesSha256",
  ):
    _sha256(payload[name], f"robust plant-set {name}")
  if payload["reviewedReplaySource"] != replay_source.to_dict():
    raise RobustPlantSetError("robust plant set was produced by another replay source")
  if payload["moduleClosureSha256"] != replay_source.module_closure_sha256:
    raise RobustPlantSetError("robust plant set module closure differs from replay")
  try:
    persisted_partition = frozen_behavior_partition_from_dict(payload["partition"])
  except BehaviorPartitionError as error:
    raise RobustPlantSetError("robust plant-set partition is malformed") from error
  if (
    persisted_partition != partition
    or payload["partitionExclusions"] != partition.to_dict()["exclusions"]
  ):
    raise RobustPlantSetError("robust plant set carries another route partition")
  expected = (
    (payload["importManifestSha256"], import_manifest_sha256),
    (payload["partitionReceiptSha256"], partition_receipt_sha256),
    (payload["partitionSha256"], partition.sha256),
    (payload["physicalGenerationSha256"], physical_generation_sha256),
    (payload["physicalProfileSha256"], physical_profile_sha256),
    (payload["physicalModuleClosureSha256"], physical_module_closure_sha256),
  )
  if any(left != right for left, right in expected):
    raise RobustPlantSetError("robust plant set belongs to another experiment")
  raw_members = payload["members"]
  if type(raw_members) is not list:
    raise RobustPlantSetError("robust plant-set members are malformed")
  members: list[CounterfactualPlantMember] = []
  for value in raw_members:
    if type(value) is not dict or set(value) != {
      "delayOffsetS", "loadUncertaintyTorque", "memberId",
      "rackDampingPerS", "rackGainDegS2PerTorque",
    }:
      raise RobustPlantSetError("robust plant-set member is malformed")
    member = CounterfactualPlantMember.create(
      rack_gain_deg_s2_per_torque=_float_hex(value["rackGainDegS2PerTorque"], "rack gain"),
      rack_damping_per_s=_float_hex(value["rackDampingPerS"], "rack damping"),
      delay_offset_s=_float_hex(value["delayOffsetS"], "delay offset"),
      unresolved_load_torque=_float_hex(value["loadUncertaintyTorque"], "load uncertainty"),
    )
    if value["memberId"] != member.member_id:
      raise RobustPlantSetError("robust plant-set member identity differs")
    members.append(member)
  members_tuple = tuple(members)
  member_ids = tuple(member.member_id for member in members_tuple)
  if member_ids != tuple(sorted(set(member_ids))):
    raise RobustPlantSetError("robust plant-set members are not canonical")
  if not ROBUST_PLANT_SET_MINIMUM_MEMBERS <= len(member_ids) <= ROBUST_PLANT_SET_MAXIMUM_MEMBERS:
    raise RobustPlantSetError("robust plant-set population is outside its bound")
  member_wire = [
    {
      "delayOffsetS": member.delay_offset_s.hex(),
      "loadUncertaintyTorque": member.unresolved_load_torque.hex(),
      "memberId": member.member_id,
      "rackDampingPerS": member.rack_damping_per_s.hex(),
      "rackGainDegS2PerTorque": member.rack_gain_deg_s2_per_torque.hex(),
    }
    for member in members_tuple
  ]
  member_set_sha256 = hashlib.sha256(
    b"blatv2-robust-member-set-v1\0" + canonical_json(member_wire).encode(),
  ).hexdigest()
  if payload["memberSetSha256"] != member_set_sha256:
    raise RobustPlantSetError("robust plant-set population identity differs")
  for name in ("trainingMemberIds", "validationMemberIds", "testMemberIds"):
    if payload[name] != list(member_ids):
      raise RobustPlantSetError("robust plant set changed across held-out stages")
  return RobustPlantSet(
    receipt_sha256=receipt_sha256,
    import_manifest_sha256=payload["importManifestSha256"],
    partition_receipt_sha256=payload["partitionReceiptSha256"],
    partition_sha256=payload["partitionSha256"],
    physical_generation_sha256=payload["physicalGenerationSha256"],
    physical_profile_sha256=payload["physicalProfileSha256"],
    physical_module_closure_sha256=payload["physicalModuleClosureSha256"],
    transient_report_sha256=payload["transientReportSha256"],
    transient_rules_sha256=payload["transientRulesSha256"],
    transient_module_closure_sha256=payload["transientModuleClosureSha256"],
    members=members_tuple,
  )
