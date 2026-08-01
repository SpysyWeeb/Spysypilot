from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from openpilot.selfdrive.controls.lib.blatv2 import learning_backfill
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  BACKFILL_COMMIT_SCHEMA_VERSION,
  CANONICAL_JOIN_SCHEMA_VERSION,
  BACKFILL_LEDGER_SCHEMA_VERSION,
  BACKFILL_POINTER_SCHEMA_VERSION,
  BackfillError,
  ReplayResult,
  RouteCandidate,
  RouteSegment,
  extend_ledger,
  load_ledger,
  publish_generation,
)
from openpilot.selfdrive.controls.lib.blatv2.learner import (
  LearningResult,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_learner import (
  CALIBRATION_EVIDENCE_SCHEMA_VERSION,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_coordinator import (
  CALIBRATION_COORDINATOR_ARTIFACT_SCHEMA_VERSION,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_coordinator import (
  LearningFinalization,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  FULL_RLOG_INCLUSION_POLICY_NAMESPACE,
  LearningArtifactPaths,
  artifact_paths_for_bundle,
)


RUNTIME_IDENTITY = hashlib.sha256(b"runtime").hexdigest()
DESCRIPTOR_REGISTRY_IDENTITY = hashlib.sha256(b"descriptors").hexdigest()
EXTRACTOR_IDENTITY = hashlib.sha256(b"extractor").hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
  ).encode("utf-8")


def make_finalization(tag: str) -> LearningFinalization:
  evidence = canonical_json_bytes({"evidence": tag})
  manifest = canonical_json_bytes({"manifest": tag})
  return LearningFinalization(
    manifest_bytes=manifest,
    manifest_sha256=hashlib.sha256(manifest).hexdigest(),
    evidence_bytes=evidence,
    evidence_sha256=hashlib.sha256(evidence).hexdigest(),
    candidate_profile_json=None,
    candidate_profile_sha256=None,
    learning_result=LearningResult(
      node_reports=(),
      candidate_profile=None,
    ),
  )


def empty_ledger() -> dict[str, object]:
  return {
    "entries": [],
    "runtime_identity_sha256": RUNTIME_IDENTITY,
    "schema_version": BACKFILL_LEDGER_SCHEMA_VERSION,
    "watermark_route_counter": None,
  }


def make_route(counter: int) -> RouteCandidate:
  route_name = f"{counter:08x}--{counter:010x}"
  segment_identity = hashlib.sha256(route_name.encode("ascii")).hexdigest()
  return RouteCandidate(
    route_name=route_name,
    route_counter=counter,
    segments=(
      RouteSegment(
        index=0,
        path=Path("/route-not-read-by-ledger-tests"),
        sha256=segment_identity,
        size_bytes=1024 + counter,
      ),
    ),
  )


def replay_result(
  route: RouteCandidate,
  *,
  accepted: int = 1,
) -> ReplayResult:
  return ReplayResult(
    route=route,
    disposition="ingested",
    diagnostic="ingested",
    provenance={
      "canonical_join_schema_version": CANONICAL_JOIN_SCHEMA_VERSION,
      "car_params_sha256": hashlib.sha256(b"car-params").hexdigest(),
      "dongle_id_sha256": hashlib.sha256(b"dongle").hexdigest(),
      "extractor_schema_version": 1,
      "log_schema_blob": "4" * 40,
      "opendbc_commit": "2" * 40,
      "panda_commit": "3" * 40,
      "physical_compatibility_sha256": hashlib.sha256(
        b"physical",
      ).hexdigest(),
      "route_version": "test-version",
      "selected_event_stream_sha256": hashlib.sha256(
        route.route_name.encode("ascii"),
      ).hexdigest(),
      "superproject_commit": "1" * 40,
    },
    accepted_sample_count=accepted,
    rejected_sample_count=2,
    controls_witness_count=accepted + 2,
    unresolved_witness_count=0,
  )


def publish(
  paths: LearningArtifactPaths,
  finalization: LearningFinalization,
  ledger: dict[str, object],
  *,
  abort_requested=lambda: False,
) -> tuple[str, str]:
  return publish_generation(
    artifact_paths=paths,
    runtime_identity_sha256=RUNTIME_IDENTITY,
    finalization=finalization,
    ledger=ledger,
    descriptor_registry_sha256=DESCRIPTOR_REGISTRY_IDENTITY,
    extractor_sha256=EXTRACTOR_IDENTITY,
    abort_requested=abort_requested,
  )


class TestBLaTv2BackfillGeneration(unittest.TestCase):
  def test_signed_causal_learning_contract_versions_move_together(
    self,
  ) -> None:
    self.assertEqual(
      (
        CALIBRATION_EVIDENCE_SCHEMA_VERSION,
        CALIBRATION_COORDINATOR_ARTIFACT_SCHEMA_VERSION,
        CANONICAL_JOIN_SCHEMA_VERSION,
        FULL_RLOG_INCLUSION_POLICY_NAMESPACE,
      ),
      (6, 5, 2, "complete_full_rlog_authority_v4"),
    )

  def test_inclusion_policy_namespace_ignores_legacy_runtime_root(
    self,
  ) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      storage_root = Path(temporary)
      legacy_root = storage_root / RUNTIME_IDENTITY
      legacy_root.mkdir()
      legacy_pointer = canonical_json_bytes({
        "generation_sha256": hashlib.sha256(b"legacy").hexdigest(),
        "schema_version": BACKFILL_POINTER_SCHEMA_VERSION,
      })
      legacy_evidence = b'{"legacy":"evidence"}'
      legacy_manifest = b'{"legacy":"manifest"}'
      (legacy_root / "backfill_current.json").write_bytes(legacy_pointer)
      (legacy_root / "evidence.json").write_bytes(legacy_evidence)
      (legacy_root / "manifest.json").write_bytes(legacy_manifest)
      predecessor_namespaces = []
      predecessor_before = {}
      for version in (1, 2, 3):
        predecessor_namespace = (
          legacy_root / f"complete_full_rlog_authority_v{version}"
        )
        predecessor_namespace.mkdir()
        predecessor_pointer = canonical_json_bytes({
          "generation_sha256": hashlib.sha256(
            f"predecessor-generation-v{version}".encode(),
          ).hexdigest(),
          "schema_version": BACKFILL_POINTER_SCHEMA_VERSION,
        })
        (predecessor_namespace / "backfill_current.json").write_bytes(
          predecessor_pointer,
        )
        (predecessor_namespace / "evidence.json").write_bytes(
          f'{{"v{version}":"evidence"}}'.encode(),
        )
        (predecessor_namespace / "manifest.json").write_bytes(
          f'{{"v{version}":"manifest"}}'.encode(),
        )
        predecessor_generation = (
          predecessor_namespace
          / "generations"
          / f"frozen-v{version}"
        )
        predecessor_generation.mkdir(parents=True)
        (predecessor_generation / "commit.json").write_bytes(
          f'{{"v{version}":"commit"}}'.encode(),
        )
        predecessor_namespaces.append(predecessor_namespace)
        predecessor_before[version] = {
          path.relative_to(predecessor_namespace): path.read_bytes()
          for path in predecessor_namespace.rglob("*")
          if path.is_file()
        }

      paths = artifact_paths_for_bundle(
        storage_root,
        SimpleNamespace(calibration_identity_sha256=RUNTIME_IDENTITY),
      )

      self.assertEqual(
        paths.root,
        legacy_root / FULL_RLOG_INCLUSION_POLICY_NAMESPACE,
      )
      self.assertFalse(paths.root.exists())
      self.assertFalse(paths.backfill_pointer.exists())
      self.assertEqual(
        load_ledger(
          paths,
          runtime_identity_sha256=RUNTIME_IDENTITY,
        ),
        empty_ledger(),
      )
      self.assertEqual(
        (legacy_root / "backfill_current.json").read_bytes(),
        legacy_pointer,
      )
      self.assertEqual(
        (legacy_root / "evidence.json").read_bytes(),
        legacy_evidence,
      )
      self.assertEqual(
        (legacy_root / "manifest.json").read_bytes(),
        legacy_manifest,
      )
      publish(
        paths,
        make_finalization("v4"),
        empty_ledger(),
      )
      self.assertTrue(paths.backfill_pointer.is_file())
      for version, predecessor_namespace in zip(
        (1, 2, 3), predecessor_namespaces, strict=True,
      ):
        predecessor_after = {
          path.relative_to(predecessor_namespace): path.read_bytes()
          for path in predecessor_namespace.rglob("*")
          if path.is_file()
        }
        self.assertEqual(predecessor_after, predecessor_before[version])

  def test_publish_is_content_addressed_and_hash_bound(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      paths = LearningArtifactPaths(Path(temporary))
      ledger = extend_ledger(
        empty_ledger(),
        late_routes=(),
        replay_results=(replay_result(make_route(1)),),
      )
      finalization = make_finalization("first")

      generation_identity, ledger_identity = publish(
        paths,
        finalization,
        ledger,
      )

      pointer_bytes = paths.backfill_pointer.read_bytes()
      self.assertEqual(
        pointer_bytes,
        canonical_json_bytes(
          {
            "generation_sha256": generation_identity,
            "schema_version": BACKFILL_POINTER_SCHEMA_VERSION,
          }
        ),
      )
      generation = paths.backfill_generations / generation_identity
      commit_bytes = (generation / "commit.json").read_bytes()
      self.assertEqual(
        hashlib.sha256(commit_bytes).hexdigest(),
        generation_identity,
      )
      commit = json.loads(commit_bytes)
      self.assertEqual(commit["schema_version"], BACKFILL_COMMIT_SCHEMA_VERSION)
      self.assertEqual(
        commit["evidence_sha256"],
        hashlib.sha256((generation / "evidence.json").read_bytes()).hexdigest(),
      )
      self.assertEqual(
        commit["manifest_sha256"],
        hashlib.sha256((generation / "manifest.json").read_bytes()).hexdigest(),
      )
      self.assertEqual(
        commit["ledger_sha256"],
        hashlib.sha256((generation / "ledger.json").read_bytes()).hexdigest(),
      )
      self.assertEqual(commit["ledger_sha256"], ledger_identity)
      self.assertEqual(
        commit["provenance_sha256"],
        hashlib.sha256(
          (generation / "provenance.json").read_bytes(),
        ).hexdigest(),
      )
      self.assertEqual(
        load_ledger(paths, runtime_identity_sha256=RUNTIME_IDENTITY),
        ledger,
      )

  def test_resolved_paths_snapshot_current_without_mixed_generations(
    self,
  ) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      paths = LearningArtifactPaths(Path(temporary))
      first = make_finalization("first")
      first_generation, _ = publish(paths, first, empty_ledger())
      snapshot = paths.resolved()

      second = make_finalization("second")
      second_generation, _ = publish(paths, second, empty_ledger())

      self.assertNotEqual(first_generation, second_generation)
      self.assertEqual(snapshot.evidence.read_bytes(), first.evidence_bytes)
      self.assertEqual(snapshot.manifest.read_bytes(), first.manifest_bytes)
      self.assertEqual(paths.evidence.read_bytes(), second.evidence_bytes)
      self.assertEqual(paths.manifest.read_bytes(), second.manifest_bytes)
      self.assertEqual(
        snapshot.backfill_commit.parent.name,
        first_generation,
      )
      self.assertEqual(paths.backfill_commit.parent.name, second_generation)

  def test_all_abort_checkpoints_keep_old_generation_active(self) -> None:
    for abort_on_call in (1, 2, 3, 4):
      with self.subTest(abort_on_call=abort_on_call):
        with tempfile.TemporaryDirectory() as temporary:
          paths = LearningArtifactPaths(Path(temporary))
          first = make_finalization("first")
          publish(paths, first, empty_ledger())
          old_pointer = paths.backfill_pointer.read_bytes()
          old_generations = set(paths.backfill_generations.iterdir())
          abort_call_count = 0

          def abort_at_checkpoint(
            checkpoint: int = abort_on_call,
          ) -> bool:
            nonlocal abort_call_count
            abort_call_count += 1
            return abort_call_count >= checkpoint

          with self.assertRaises(BackfillError) as raised:
            publish(
              paths,
              make_finalization(f"aborted-{abort_on_call}"),
              empty_ledger(),
              abort_requested=abort_at_checkpoint,
            )

          self.assertEqual(raised.exception.diagnostic, "unexpected_error")
          self.assertEqual(paths.backfill_pointer.read_bytes(), old_pointer)
          self.assertEqual(paths.evidence.read_bytes(), first.evidence_bytes)
          self.assertEqual(paths.manifest.read_bytes(), first.manifest_bytes)
          new_generations = set(paths.backfill_generations.iterdir())
          self.assertEqual(
            len(new_generations - old_generations),
            int(abort_on_call >= 3),
          )

  def test_pointer_write_failure_keeps_old_generation_active(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      paths = LearningArtifactPaths(Path(temporary))
      first = make_finalization("first")
      publish(paths, first, empty_ledger())
      old_pointer = paths.backfill_pointer.read_bytes()

      with patch.object(
        learning_backfill,
        "_atomic_write_bytes",
        side_effect=OSError("injected pointer failure"),
      ):
        with self.assertRaises(BackfillError) as raised:
          publish(paths, make_finalization("failed"), empty_ledger())

      self.assertEqual(
        raised.exception.diagnostic,
        "backfill_publish_failed",
      )
      self.assertEqual(paths.backfill_pointer.read_bytes(), old_pointer)
      self.assertEqual(paths.evidence.read_bytes(), first.evidence_bytes)
      self.assertEqual(paths.manifest.read_bytes(), first.manifest_bytes)

  def test_corrupt_existing_generation_collision_is_rejected(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      paths = LearningArtifactPaths(Path(temporary))
      finalization = make_finalization("collision")
      generation_identity, _ = publish(
        paths,
        finalization,
        empty_ledger(),
      )
      paths.backfill_pointer.unlink()
      generation = paths.backfill_generations / generation_identity
      (generation / "evidence.json").write_bytes(b"corrupt")

      with self.assertRaises(BackfillError) as raised:
        publish(paths, finalization, empty_ledger())

      self.assertEqual(
        raised.exception.diagnostic,
        "backfill_publish_failed",
      )
      self.assertFalse(paths.backfill_pointer.exists())

  def test_legacy_nonempty_artifacts_without_ledger_fail_closed(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      for name in ("evidence.json", "manifest.json"):
        with self.subTest(name=name):
          artifact_root = root / name
          artifact_root.mkdir()
          (artifact_root / name).write_bytes(b"legacy")
          paths = LearningArtifactPaths(artifact_root)

          with self.assertRaises(BackfillError) as raised:
            load_ledger(
              paths,
              runtime_identity_sha256=RUNTIME_IDENTITY,
            )

          self.assertEqual(
            raised.exception.diagnostic,
            "backfill_untracked_evidence",
          )

  def test_extend_ledger_is_exactly_once_and_tracks_late_ordering(
    self,
  ) -> None:
    base_route = make_route(0x20)
    base = extend_ledger(
      empty_ledger(),
      late_routes=(),
      replay_results=(replay_result(base_route),),
    )
    late_routes = (make_route(0x05), make_route(0x10))
    new_results = (
      replay_result(make_route(0x21), accepted=3),
      replay_result(make_route(0x30), accepted=4),
    )

    extended = extend_ledger(
      base,
      late_routes=late_routes,
      replay_results=new_results,
    )

    self.assertEqual(len(base["entries"]), 1)
    self.assertEqual(
      [entry["route_name"] for entry in extended["entries"]],
      [
        base_route.route_name,
        late_routes[0].route_name,
        late_routes[1].route_name,
        new_results[0].route.route_name,
        new_results[1].route.route_name,
      ],
    )
    self.assertEqual(
      [entry["disposition"] for entry in extended["entries"]],
      [
        "ingested",
        "late_older_skipped",
        "late_older_skipped",
        "ingested",
        "ingested",
      ],
    )
    self.assertEqual(extended["watermark_route_counter"], 0x30)
    for entry in extended["entries"][1:3]:
      self.assertEqual(entry["accepted_sample_count"], 0)
      self.assertEqual(entry["rejected_sample_count"], 0)

    with self.assertRaises(BackfillError) as raised:
      extend_ledger(
        extended,
        late_routes=(),
        replay_results=(new_results[-1],),
      )
    self.assertEqual(
      raised.exception.diagnostic,
      "backfill_untracked_evidence",
    )

  def test_ledger_semantics_fail_closed_under_tampering(self) -> None:
    base = extend_ledger(
      empty_ledger(),
      late_routes=(),
      replay_results=(replay_result(make_route(0x20)),),
    )
    valid = extend_ledger(
      base,
      late_routes=(make_route(0x10),),
      replay_results=(replay_result(make_route(0x30)),),
    )
    cases = {
      "empty_segments": lambda payload: payload["entries"][2].update(
        segments=[],
      ),
      "noncontiguous_segments": lambda payload: payload["entries"][2][
        "segments"
      ][0].update(index=1),
      "missing_provenance": lambda payload: payload["entries"][2].update(
        provenance=None,
      ),
      "bad_provenance_hash": lambda payload: payload["entries"][2][
        "provenance"
      ].update(selected_event_stream_sha256="0"),
      "bool_provenance_version": lambda payload: payload["entries"][2][
        "provenance"
      ].update(extractor_schema_version=True),
      "malformed_route_version": lambda payload: payload["entries"][2][
        "provenance"
      ].update(route_version=" leading"),
      "accepted_over_controls": lambda payload: payload["entries"][2].update(
        accepted_sample_count=4,
      ),
      "unresolved_over_controls": lambda payload: payload["entries"][2].update(
        unresolved_witness_count=4,
      ),
      "rejected_underived": lambda payload: payload["entries"][2].update(
        rejected_sample_count=0,
      ),
      "late_has_provenance": lambda payload: payload["entries"][1].update(
        provenance={"forged": True},
      ),
      "late_has_counter": lambda payload: payload["entries"][1].update(
        controls_witness_count=1,
      ),
      "late_bad_diagnostic": lambda payload: payload["entries"][1].update(
        diagnostic="skipped",
      ),
    }
    for name, mutate in cases.items():
      with self.subTest(name=name):
        tampered = copy.deepcopy(valid)
        mutate(tampered)
        with self.assertRaises(BackfillError) as raised:
          learning_backfill.validate_ledger(
            tampered,
            runtime_identity_sha256=RUNTIME_IDENTITY,
          )
        self.assertEqual(
          raised.exception.diagnostic,
          "backfill_untracked_evidence",
        )

    with self.assertRaises(BackfillError) as raised:
      extend_ledger(
        empty_ledger(),
        late_routes=(make_route(0x10),),
        replay_results=(),
      )
    self.assertEqual(
      raised.exception.diagnostic,
      "backfill_untracked_evidence",
    )


if __name__ == "__main__":
  unittest.main()
