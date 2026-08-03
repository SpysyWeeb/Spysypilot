"""Append-stable whole-route partitions for offline BLaTv2 training.

This module is the sole numerical authority for assigning authenticated route
artifacts to training, validation, or untouched test. Assignment depends only
on the immutable artifact bytes and the committed experiment seed. Recorded
controller identity and route-derived quality are deliberately outside this
interface: they cannot turn a usable scenario into a behavior target or move a
route after its split is known.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import re


BEHAVIOR_PARTITION_SCHEMA_VERSION = 1
BEHAVIOR_PARTITION_ALGORITHM = "sha256_content_bucket_6000_2000_2000_v1"

_PARTITION_DOMAIN = b"blatv2-training-partition-v1\0"
_ROUTE_ID_RE = re.compile(r"[0-9a-f]{8}--[0-9a-f]{10}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REJECTION_REASON_RE = re.compile(r"[a-z0-9][a-z0-9._:-]{0,127}\Z")


class BehaviorPartitionError(ValueError):
  """A route population cannot form the committed whole-route partition."""


class BehaviorPartitionSplit(StrEnum):
  TRAINING = "training"
  VALIDATION = "validation"
  TEST = "test"


def _route_id(value: object, description: str) -> str:
  if type(value) is not str or _ROUTE_ID_RE.fullmatch(value) is None:
    raise BehaviorPartitionError(f"{description} is not a canonical route ID")
  return value


def _sha256(value: object, description: str) -> str:
  if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
    raise BehaviorPartitionError(f"{description} is not a lowercase SHA-256")
  return value


def _canonical_json(value: object) -> bytes:
  return json.dumps(
    value,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("ascii")


@dataclass(frozen=True, slots=True)
class BehaviorPartitionBoundaries:
  bucket_count: int
  training_end: int
  validation_end: int

  def __post_init__(self) -> None:
    if any(type(value) is not int for value in (
      self.bucket_count,
      self.training_end,
      self.validation_end,
    )):
      raise BehaviorPartitionError("partition boundaries must be integers")
    if not 0 < self.training_end < self.validation_end < self.bucket_count:
      raise BehaviorPartitionError("partition boundaries are not strictly ordered")

  def to_dict(self) -> dict[str, int]:
    return {
      "bucketCount": self.bucket_count,
      "trainingEnd": self.training_end,
      "validationEnd": self.validation_end,
    }


COMMITTED_BEHAVIOR_PARTITION_BOUNDARIES = BehaviorPartitionBoundaries(
  bucket_count=10_000,
  training_end=6_000,
  validation_end=8_000,
)


@dataclass(frozen=True, slots=True)
class BehaviorPartitionExclusion:
  route_id: str
  reasons: tuple[str, ...]

  def __post_init__(self) -> None:
    _route_id(self.route_id, "excluded route ID")
    if (
      type(self.reasons) is not tuple
      or not self.reasons
      or any(
        type(reason) is not str or _REJECTION_REASON_RE.fullmatch(reason) is None
        for reason in self.reasons
      )
      or self.reasons != tuple(sorted(set(self.reasons)))
    ):
      raise BehaviorPartitionError("route exclusion reasons are not canonical")

  def to_dict(self) -> dict[str, object]:
    return {"reasons": list(self.reasons), "routeId": self.route_id}


@dataclass(frozen=True, slots=True)
class BehaviorPartitionAssignment:
  route_id: str
  artifact_sha256: str
  bucket: int
  split: BehaviorPartitionSplit

  def __post_init__(self) -> None:
    _route_id(self.route_id, "assigned route ID")
    _sha256(self.artifact_sha256, "assigned route artifact")
    if type(self.bucket) is not int or self.bucket < 0:
      raise BehaviorPartitionError("assigned route bucket is invalid")
    if type(self.split) is not BehaviorPartitionSplit:
      raise BehaviorPartitionError("assigned route split is invalid")

  def to_dict(self) -> dict[str, object]:
    return {
      "artifactSha256": self.artifact_sha256,
      "bucket": self.bucket,
      "routeId": self.route_id,
      "split": self.split.value,
    }


@dataclass(frozen=True, slots=True)
class FrozenBehaviorPartition:
  seed_identity_sha256: str
  boundaries: BehaviorPartitionBoundaries
  minimum_route_count: int
  assignments: tuple[BehaviorPartitionAssignment, ...]
  exclusions: tuple[BehaviorPartitionExclusion, ...] = ()

  def __post_init__(self) -> None:
    _sha256(self.seed_identity_sha256, "partition seed")
    if type(self.boundaries) is not BehaviorPartitionBoundaries:
      raise BehaviorPartitionError("partition boundaries are malformed")
    if type(self.minimum_route_count) is not int or self.minimum_route_count <= 0:
      raise BehaviorPartitionError("minimum route count must be positive")
    if (
      type(self.assignments) is not tuple
      or not self.assignments
      or any(type(row) is not BehaviorPartitionAssignment for row in self.assignments)
      or self.assignments != tuple(sorted(self.assignments, key=lambda row: row.route_id))
    ):
      raise BehaviorPartitionError("partition assignments are not canonical")
    if (
      type(self.exclusions) is not tuple
      or any(type(row) is not BehaviorPartitionExclusion for row in self.exclusions)
      or self.exclusions != tuple(sorted(self.exclusions, key=lambda row: row.route_id))
    ):
      raise BehaviorPartitionError("partition exclusions are not canonical")

    assigned_ids = tuple(row.route_id for row in self.assignments)
    artifact_ids = tuple(row.artifact_sha256 for row in self.assignments)
    excluded_ids = tuple(row.route_id for row in self.exclusions)
    if len(set(assigned_ids)) != len(assigned_ids):
      raise BehaviorPartitionError("partition contains duplicate route IDs")
    if len(set(artifact_ids)) != len(artifact_ids):
      raise BehaviorPartitionError("partition contains duplicate route artifacts")
    if len(set(excluded_ids)) != len(excluded_ids):
      raise BehaviorPartitionError("partition contains duplicate exclusions")
    if set(assigned_ids) & set(excluded_ids):
      raise BehaviorPartitionError("partition both assigns and excludes a route")

    for row in self.assignments:
      if row.bucket >= self.boundaries.bucket_count:
        raise BehaviorPartitionError("assigned route bucket exceeds the partition")
      if row.split is not _split_for_bucket(row.bucket, self.boundaries):
        raise BehaviorPartitionError("assigned route split disagrees with its bucket")
    counts = self.counts
    if any(count < self.minimum_route_count for count in counts.values()):
      detail = ", ".join(f"{split.value}={counts[split]}" for split in BehaviorPartitionSplit)
      raise BehaviorPartitionError(f"partition lacks minimum route support ({detail})")

  @property
  def counts(self) -> dict[BehaviorPartitionSplit, int]:
    return {
      split: sum(row.split is split for row in self.assignments)
      for split in BehaviorPartitionSplit
    }

  def route_ids(self, split: BehaviorPartitionSplit) -> tuple[str, ...]:
    if type(split) is not BehaviorPartitionSplit:
      raise TypeError("split must use BehaviorPartitionSplit")
    return tuple(row.route_id for row in self.assignments if row.split is split)

  def to_dict(self) -> dict[str, object]:
    return {
      "algorithm": BEHAVIOR_PARTITION_ALGORITHM,
      "assignments": [row.to_dict() for row in self.assignments],
      "boundaries": self.boundaries.to_dict(),
      "exclusions": [row.to_dict() for row in self.exclusions],
      "minimumRouteCount": self.minimum_route_count,
      "schemaVersion": BEHAVIOR_PARTITION_SCHEMA_VERSION,
      "seedIdentitySha256": self.seed_identity_sha256,
      "splitCounts": {
        split.value: self.counts[split]
        for split in BehaviorPartitionSplit
      },
    }

  @property
  def canonical_bytes(self) -> bytes:
    return _canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.canonical_bytes).hexdigest()


def _split_for_bucket(
  bucket: int,
  boundaries: BehaviorPartitionBoundaries,
) -> BehaviorPartitionSplit:
  if bucket < boundaries.training_end:
    return BehaviorPartitionSplit.TRAINING
  if bucket < boundaries.validation_end:
    return BehaviorPartitionSplit.VALIDATION
  return BehaviorPartitionSplit.TEST


def _partition_bucket(
  artifact_sha256: str,
  seed_identity_sha256: str,
  boundaries: BehaviorPartitionBoundaries,
) -> int:
  digest = hashlib.sha256(
    _PARTITION_DOMAIN
    + bytes.fromhex(seed_identity_sha256)
    + bytes.fromhex(artifact_sha256),
  ).digest()
  return int.from_bytes(digest[:8], "big") % boundaries.bucket_count


def build_behavior_partition(
  route_rows: Iterable[tuple[str, str]],
  *,
  seed_identity_sha256: str,
  boundaries: BehaviorPartitionBoundaries = COMMITTED_BEHAVIOR_PARTITION_BOUNDARIES,
  exclusions: Iterable[BehaviorPartitionExclusion] = (),
  minimum_route_count: int = 2,
) -> FrozenBehaviorPartition:
  """Assign every supplied artifact exactly once without opening route data."""
  seed = _sha256(seed_identity_sha256, "partition seed")
  if type(boundaries) is not BehaviorPartitionBoundaries:
    raise TypeError("boundaries must use BehaviorPartitionBoundaries")
  if type(minimum_route_count) is not int or minimum_route_count <= 0:
    raise BehaviorPartitionError("minimum route count must be positive")

  canonical_rows: list[tuple[str, str]] = []
  for index, row in enumerate(route_rows):
    if type(row) is not tuple or len(row) != 2:
      raise BehaviorPartitionError(f"route row {index} is not a canonical pair")
    route_id, artifact_sha256 = row
    canonical_rows.append((
      _route_id(route_id, f"route row {index} ID"),
      _sha256(artifact_sha256, f"route row {index} artifact"),
    ))
  if not canonical_rows:
    raise BehaviorPartitionError("partition route population is empty")
  if len({row[0] for row in canonical_rows}) != len(canonical_rows):
    raise BehaviorPartitionError("partition input contains duplicate route IDs")
  if len({row[1] for row in canonical_rows}) != len(canonical_rows):
    raise BehaviorPartitionError("partition input contains duplicate route artifacts")

  canonical_exclusions = tuple(exclusions)
  if any(type(row) is not BehaviorPartitionExclusion for row in canonical_exclusions):
    raise TypeError("exclusions must contain BehaviorPartitionExclusion values")
  canonical_exclusions = tuple(sorted(canonical_exclusions, key=lambda row: row.route_id))
  if len({row.route_id for row in canonical_exclusions}) != len(canonical_exclusions):
    raise BehaviorPartitionError("partition input contains duplicate exclusions")
  if {row[0] for row in canonical_rows} & {row.route_id for row in canonical_exclusions}:
    raise BehaviorPartitionError("partition input both includes and excludes a route")

  assignments = []
  for route_id, artifact_sha256 in sorted(canonical_rows):
    bucket = _partition_bucket(artifact_sha256, seed, boundaries)
    assignments.append(BehaviorPartitionAssignment(
      route_id=route_id,
      artifact_sha256=artifact_sha256,
      bucket=bucket,
      split=_split_for_bucket(bucket, boundaries),
    ))
  return FrozenBehaviorPartition(
    seed_identity_sha256=seed,
    boundaries=boundaries,
    minimum_route_count=minimum_route_count,
    assignments=tuple(assignments),
    exclusions=canonical_exclusions,
  )


def frozen_behavior_partition_from_dict(value: object) -> FrozenBehaviorPartition:
  """Reconstruct and independently verify one serialized partition receipt."""
  if type(value) is not dict or set(value) != {
    "algorithm",
    "assignments",
    "boundaries",
    "exclusions",
    "minimumRouteCount",
    "schemaVersion",
    "seedIdentitySha256",
    "splitCounts",
  }:
    raise BehaviorPartitionError("serialized partition shape is incompatible")
  if (
    value["algorithm"] != BEHAVIOR_PARTITION_ALGORITHM
    or value["schemaVersion"] != BEHAVIOR_PARTITION_SCHEMA_VERSION
  ):
    raise BehaviorPartitionError("serialized partition authority is incompatible")
  boundaries = value["boundaries"]
  if type(boundaries) is not dict or set(boundaries) != {
    "bucketCount", "trainingEnd", "validationEnd",
  }:
    raise BehaviorPartitionError("serialized partition boundaries are malformed")
  parsed_boundaries = BehaviorPartitionBoundaries(
    boundaries["bucketCount"],
    boundaries["trainingEnd"],
    boundaries["validationEnd"],
  )
  assignments = value["assignments"]
  exclusions = value["exclusions"]
  if type(assignments) is not list or type(exclusions) is not list:
    raise BehaviorPartitionError("serialized partition rows are malformed")
  parsed_assignments: list[BehaviorPartitionAssignment] = []
  for row in assignments:
    if type(row) is not dict or set(row) != {
      "artifactSha256", "bucket", "routeId", "split",
    }:
      raise BehaviorPartitionError("serialized partition assignment is malformed")
    try:
      split = BehaviorPartitionSplit(row["split"])
    except (TypeError, ValueError) as error:
      raise BehaviorPartitionError("serialized partition split is invalid") from error
    parsed_assignments.append(BehaviorPartitionAssignment(
      route_id=row["routeId"],
      artifact_sha256=row["artifactSha256"],
      bucket=row["bucket"],
      split=split,
    ))
  parsed_exclusions: list[BehaviorPartitionExclusion] = []
  for row in exclusions:
    if type(row) is not dict or set(row) != {"reasons", "routeId"}:
      raise BehaviorPartitionError("serialized partition exclusion is malformed")
    reasons = row["reasons"]
    if type(reasons) is not list:
      raise BehaviorPartitionError("serialized partition exclusion reasons are malformed")
    parsed_exclusions.append(BehaviorPartitionExclusion(row["routeId"], tuple(reasons)))
  result = FrozenBehaviorPartition(
    seed_identity_sha256=value["seedIdentitySha256"],
    boundaries=parsed_boundaries,
    minimum_route_count=value["minimumRouteCount"],
    assignments=tuple(parsed_assignments),
    exclusions=tuple(parsed_exclusions),
  )
  expected_counts = {
    split.value: result.counts[split]
    for split in BehaviorPartitionSplit
  }
  if value["splitCounts"] != expected_counts:
    raise BehaviorPartitionError("serialized partition counts disagree")
  rebuilt = build_behavior_partition(
    tuple((row.route_id, row.artifact_sha256) for row in result.assignments),
    seed_identity_sha256=result.seed_identity_sha256,
    boundaries=result.boundaries,
    exclusions=result.exclusions,
    minimum_route_count=result.minimum_route_count,
  )
  if result != rebuilt:
    raise BehaviorPartitionError("serialized partition differs from shared assignment math")
  return result
