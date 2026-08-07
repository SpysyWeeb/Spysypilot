from __future__ import annotations

from dataclasses import replace
import inspect
import json
import unittest

from openpilot.selfdrive.controls.lib.blatv2.behavior_partition import (
  BEHAVIOR_PARTITION_ALGORITHM,
  BehaviorPartitionAssignment,
  BehaviorPartitionBoundaries,
  BehaviorPartitionError,
  BehaviorPartitionExclusion,
  BehaviorPartitionSplit,
  COMMITTED_BEHAVIOR_PARTITION_BOUNDARIES,
  FrozenBehaviorPartition,
  build_behavior_partition,
)


SEED = "a" * 64


def _route(index: int) -> tuple[str, str]:
  return f"{index:08x}--{index:010x}", f"{index:064x}"


def _routes(count: int = 11) -> tuple[tuple[str, str], ...]:
  return tuple(_route(index) for index in range(1, count + 1))


class TestBehaviorPartition(unittest.TestCase):
  def test_partition_matches_existing_authority_vectors_exactly(self) -> None:
    partition = build_behavior_partition(_routes(), seed_identity_sha256=SEED)

    self.assertEqual([
      (row.route_id, row.bucket, row.split.value)
      for row in partition.assignments
    ], [
      (_route(1)[0], 3027, "training"),
      (_route(2)[0], 1368, "training"),
      (_route(3)[0], 7818, "validation"),
      (_route(4)[0], 1859, "training"),
      (_route(5)[0], 730, "training"),
      (_route(6)[0], 4828, "training"),
      (_route(7)[0], 7416, "validation"),
      (_route(8)[0], 2377, "training"),
      (_route(9)[0], 9597, "test"),
      (_route(10)[0], 2958, "training"),
      (_route(11)[0], 8621, "test"),
    ])
    self.assertEqual(partition.to_dict()["algorithm"], BEHAVIOR_PARTITION_ALGORITHM)


  def test_partition_is_order_independent_and_append_stable(self) -> None:
    original = build_behavior_partition(_routes(), seed_identity_sha256=SEED)
    reversed_input = build_behavior_partition(reversed(_routes()), seed_identity_sha256=SEED)
    extended = build_behavior_partition(_routes(30), seed_identity_sha256=SEED)

    self.assertEqual(original.canonical_bytes, reversed_input.canonical_bytes)
    self.assertEqual(original.sha256, reversed_input.sha256)
    retained = tuple(
      row for row in extended.assignments if row.route_id in {route[0] for route in _routes()}
    )
    self.assertEqual(original.assignments, retained)
    self.assertTrue(all(
      original_row.to_dict() == extended_row.to_dict()
      for original_row, extended_row in zip(original.assignments, retained, strict=True)
    ))


  def test_every_eligible_route_is_assigned_exactly_once(self) -> None:
    partition = build_behavior_partition(_routes(), seed_identity_sha256=SEED)
    assigned = [route_id for split in BehaviorPartitionSplit for route_id in partition.route_ids(split)]

    self.assertEqual(set(assigned), {row[0] for row in _routes()})
    self.assertEqual(len(assigned), len(set(assigned)))
    self.assertEqual(len(assigned), len(_routes()))
    self.assertEqual(partition.counts, {
      BehaviorPartitionSplit.TRAINING: 7,
      BehaviorPartitionSplit.VALIDATION: 2,
      BehaviorPartitionSplit.TEST: 2,
    })


  def test_explicit_exclusions_remain_separate_and_canonical(self) -> None:
    exclusions = (
      BehaviorPartitionExclusion("00000020--0000000020", ("raw_route_incomplete",)),
      BehaviorPartitionExclusion(
        "0000001f--000000001f",
        ("missing_authority_pair", "raw_route_incomplete"),
      ),
    )
    partition = build_behavior_partition(
      _routes(),
      seed_identity_sha256=SEED,
      exclusions=reversed(exclusions),
    )

    self.assertEqual(partition.exclusions, tuple(sorted(exclusions, key=lambda row: row.route_id)))
    self.assertFalse({row.route_id for row in partition.assignments} & {
      row.route_id for row in partition.exclusions
    })
    self.assertEqual(
      json.loads(partition.canonical_bytes)["exclusions"][0]["routeId"],
      "0000001f--000000001f",
    )


  def test_duplicate_route_ids_or_artifacts_fail(self) -> None:
    populations = (
      (_route(1), _route(1), *_routes()[1:]),
      ((_route(1)[0], _route(2)[1]), _route(2), *_routes()[2:]),
    )
    for routes in populations:
      with self.subTest(routes=routes), self.assertRaisesRegex(
        BehaviorPartitionError,
        "duplicate route",
      ):
        build_behavior_partition(routes, seed_identity_sha256=SEED)


  def test_invalid_hash_rows_and_noncanonical_direct_construction_fail(self) -> None:
    with self.assertRaisesRegex(BehaviorPartitionError, "lowercase SHA-256"):
      build_behavior_partition(((_route(1)[0], "A" * 64),), seed_identity_sha256=SEED)
    with self.assertRaisesRegex(BehaviorPartitionError, "canonical pair"):
      build_behavior_partition([list(_route(1))], seed_identity_sha256=SEED)  # type: ignore[list-item]

    valid = build_behavior_partition(_routes(), seed_identity_sha256=SEED)
    with self.assertRaisesRegex(BehaviorPartitionError, "canonical"):
      replace(valid, assignments=tuple(reversed(valid.assignments)))


  def test_too_small_split_fails_closed(self) -> None:
    with self.assertRaisesRegex(BehaviorPartitionError, "minimum route support"):
      build_behavior_partition(_routes(4), seed_identity_sha256=SEED)
    with self.assertRaisesRegex(BehaviorPartitionError, "minimum route support"):
      build_behavior_partition(_routes(), seed_identity_sha256=SEED, minimum_route_count=3)


  def test_boundaries_and_assignment_are_self_validating(self) -> None:
    with self.assertRaisesRegex(BehaviorPartitionError, "strictly ordered"):
      BehaviorPartitionBoundaries(10_000, 8_000, 6_000)
    with self.assertRaisesRegex(BehaviorPartitionError, "disagrees"):
      FrozenBehaviorPartition(
        seed_identity_sha256=SEED,
        boundaries=COMMITTED_BEHAVIOR_PARTITION_BOUNDARIES,
        minimum_route_count=1,
        assignments=(
          BehaviorPartitionAssignment(
            route_id=_route(1)[0],
            artifact_sha256=_route(1)[1],
            bucket=3027,
            split=BehaviorPartitionSplit.TEST,
          ),
        ),
      )


  def test_authority_interface_cannot_receive_controller_or_quality_metadata(self) -> None:
    self.assertEqual(tuple(inspect.signature(build_behavior_partition).parameters), (
      "route_rows",
      "seed_identity_sha256",
      "boundaries",
      "exclusions",
      "minimum_route_count",
    ))
    self.assertFalse({
      "controller",
      "source",
      "quality",
      "score",
    } & set(inspect.signature(build_behavior_partition).parameters))
