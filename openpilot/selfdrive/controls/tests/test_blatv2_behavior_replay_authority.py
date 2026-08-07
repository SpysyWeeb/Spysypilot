from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from openpilot.selfdrive.controls.lib.blatv2.certification_vector import (
  CERTIFICATION_VECTOR_SCHEMA_VERSION,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import canonical_json
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay_authority import (
  AuthenticatedBehaviorReplayReceipt,
  BehaviorReplayAuthorityError,
  NonAuthoritativeBehaviorReplayResult,
  ReviewedReplaySource,
  _execute_authority_once_common,
  _execute_authority_once_for_test,
  _read_immutable_regular,
  _require_clean_checkout,
  _run_authenticated_behavior_replay_with_registry_for_test,
  _verify_replay_source,
  run_authenticated_behavior_replay,
)
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  RouteEvidenceArtifact,
)
from openpilot.selfdrive.controls.tests.test_blatv2_behavior_replay import (
  INTERFACES,
  behavior_policy,
  physical_profile,
)
from openpilot.selfdrive.controls.tests.test_blatv2_behavior_route_evaluator import (
  _artifact,
)


ROUTE_IDS = (
  "00000001--0000000001",
  "00000002--0000000002",
  "00000003--0000000003",
)


def _source_row(artifact: RouteEvidenceArtifact) -> dict[str, object]:
  value = artifact.source_identity.manifest_dict()
  value["routeId"] = value.pop("route_id")
  return value


def _route_artifact(route_id: str, index: int) -> RouteEvidenceArtifact:
  original = _artifact()
  source = replace(
    original.source_identity,
    route_id=route_id,
    route_segment_sha256=(f"{index + 1:x}" * 64,),
    preparation_cache_key=f"{index + 7:x}" * 64,
    controller_source_kind="stock_canonical",
    behavior_eligible=True,
    behavior_ineligible_reason="eligible",
  )
  return RouteEvidenceArtifact(
    source,
    original.car_params_bytes,
    original.physical_bytes,
    original.model_publications,
    original.control_witnesses,
    original.live_torque_parameters,
    original.live_delays,
    original.lateral_maneuver_plans,
    original.event_locators,
  )


def _write_immutable(path: Path, encoded: bytes) -> None:
  path.write_bytes(encoded)
  path.chmod(stat.S_IRUSR)


def _store(
  root: Path,
  *,
  rejected_route: bool = False,
) -> tuple[Path, str, Path, dict[str, object]]:
  store = root / "evidence"
  objects = store / "objects"
  imports = store / "imports"
  objects.mkdir(parents=True)
  imports.mkdir()
  rows: list[dict[str, object]] = []
  scenario_sources: list[dict[str, object]] = []
  for index, route_id in enumerate(ROUTE_IDS):
    from openpilot.selfdrive.controls.tests.test_blatv2_behavior_training_authority import (
      _certification_for_route,
    )

    artifact = _route_artifact(route_id, index)
    object_path = objects / f"{artifact.sha256}.route-evidence"
    _write_immutable(object_path, artifact.canonical_bytes)
    source = _source_row(artifact)
    vector = _certification_for_route(
      route_id,
      artifact.source_identity.route_segment_sha256[0],
      artifact.source_identity.route_segment_size_bytes[0],
      artifact.sha256,
      behavior_eligible=True,
    )
    _write_immutable(objects / f"{vector.sha256}.cert-vector", vector.canonical_bytes)
    authority_ids = [f"{index + 1:x}" * 64, f"{index + 4:x}" * 64]
    vector_authority_ids = [f"{index + 7:x}" * 64, f"{index + 10:x}" * 64]
    rows.append({
      "artifact": {
        "authorityArtifactIds": authority_ids,
        "certificationVector": {
          "authorityArtifactIds": vector_authority_ids,
          "path": f"objects/{vector.sha256}.cert-vector",
          "schemaVersion": CERTIFICATION_VECTOR_SCHEMA_VERSION,
          "selectionIdentitySha256": vector.manifest["selection_identity_sha256"],
          "sha256": vector.sha256,
          "sizeBytes": len(vector.canonical_bytes),
        },
        "path": f"objects/{artifact.sha256}.route-evidence",
        "sha256": artifact.sha256,
        "sizeBytes": len(artifact.canonical_bytes),
      },
      "archiveContentSha256": f"{index + 1:x}" * 64,
      "rejectionReasons": [],
      "routeId": route_id,
      "source": source,
      "status": "imported",
    })
    scenario_sources.append({"routeId": route_id, "source": source})
  if rejected_route:
    rows.append({
      "artifact": None,
      "archiveContentSha256": "e" * 64,
      "rejectionReasons": ["remote_invalid_route"],
      "routeId": "00000004--0000000004",
      "source": None,
      "status": "rejected",
    })
  manifest: dict[str, object] = {
    "importedRouteCount": len(ROUTE_IDS),
    "inspector": {
      "runtimeIdentitySha256": "8" * 64,
      "sourceCompositionSha256": "9" * 64,
      "sourceIdentitySha256": "a" * 64,
      "sourceOpenpilotCommit": "1" * 40,
    },
    "jobStateSha256": "b" * 64,
    "rejectedRouteCount": int(rejected_route),
    "remoteWorker": {
      "jobId": "c" * 32,
      "requestSha256": "d" * 64,
      "workerExtractorSha256": "e" * 64,
      "workerImplementationCommit": "2" * 40,
      "workerImplementationSha256": "f" * 64,
      "workerInstanceId": "0" * 64,
    },
    "routes": rows,
    "scenarioSourceSetIdentity": hashlib.sha256(canonical_json({
      "domain": "blatv2-trainer-scenario-source-set-v1",
      "routes": scenario_sources,
    }).encode()).hexdigest(),
    "schemaVersion": 2,
  }
  encoded = (canonical_json(manifest) + "\n").encode()
  manifest_sha256 = hashlib.sha256(encoded).hexdigest()
  _write_immutable(imports / f"{manifest_sha256}.json", encoded)
  profile_path = root / "physical-profile.json"
  _write_immutable(profile_path, physical_profile().to_json().encode())
  return store, manifest_sha256, profile_path, manifest


def _source() -> ReviewedReplaySource:
  return ReviewedReplaySource(
    source_openpilot_commit="1" * 40,
    opendbc_commit="2" * 40,
    panda_commit="3" * 40,
    source_composition_sha256="4" * 64,
    runtime_identity_sha256="5" * 64,
    module_closure_sha256="6" * 64,
  )


def _run(root: Path) -> NonAuthoritativeBehaviorReplayResult:
  store, manifest_sha256, profile_path, _ = _store(root)
  return _run_authenticated_behavior_replay_with_registry_for_test(
    evidence_store_root=store,
    import_manifest_sha256=manifest_sha256,
    physical_profile_path=profile_path,
    candidate=behavior_policy(),
    replay_source=_source(),
    interface_registry=INTERFACES,
  )


class TestBehaviorReplayAuthority(unittest.TestCase):
  def test_public_boundary_has_no_route_config_callback_or_receipt_inputs(self) -> None:
    self.assertEqual(
      tuple(inspect.signature(run_authenticated_behavior_replay).parameters),
      (
        "evidence_store_root",
        "import_manifest_sha256",
        "physical_profile_path",
        "candidate",
        "replay_source",
      ),
    )

  def test_injected_registry_reopens_cas_but_cannot_issue_authority(self) -> None:
    with self._temporary_path() as tmp_path:
      with patch(
        "openpilot.selfdrive.controls.lib.blatv2.behavior_replay_authority._execute_authority_once_for_test",
        wraps=_execute_authority_once_for_test,
      ) as execute:
        result = _run(tmp_path)

      self.assertEqual(execute.call_count, 2)
      self.assertEqual(result.independent_aa_sha256, result.first.sha256)
      self.assertIs(result.authoritative, False)
      self.assertFalse(hasattr(result, "to_dict"))
      self.assertEqual(result.first.aggregate_spec.scenarios.sources[0].route_id, ROUTE_IDS[0])
      self.assertEqual(result.first.aggregate_spec.partition.route_ids, ROUTE_IDS)
      self.assertIsNone(result.first.stock_training.policy)
      self.assertIsNone(result.first.stock_validation.policy)
      self.assertEqual(result.first.candidate_training.policy, behavior_policy())
      self.assertEqual(result.first.candidate_validation.policy, behavior_policy())

  @staticmethod
  @contextmanager
  def _temporary_path():
    with tempfile.TemporaryDirectory() as directory:
      yield Path(directory)

  def test_rejected_manifest_rows_remain_bound_but_never_become_scenarios(self) -> None:
    with self._temporary_path() as tmp_path:
      store, manifest_sha256, profile_path, _ = _store(tmp_path, rejected_route=True)
      result = _run_authenticated_behavior_replay_with_registry_for_test(
        evidence_store_root=store,
        import_manifest_sha256=manifest_sha256,
        physical_profile_path=profile_path,
        candidate=behavior_policy(),
        replay_source=_source(),
        interface_registry=INTERFACES,
      )
      self.assertEqual(result.first.aggregate_spec.partition.route_ids, ROUTE_IDS)

  def test_large_route_objects_are_streamed_not_read_into_authority_memory(self) -> None:
    with self._temporary_path() as tmp_path:
      with patch(
        "openpilot.selfdrive.controls.lib.blatv2.behavior_replay_authority._read_immutable_regular",
        wraps=_read_immutable_regular,
      ) as read_regular:
        _run(tmp_path)
      read_paths = tuple(call.args[0] for call in read_regular.call_args_list)
      self.assertTrue(read_paths)
      self.assertFalse(any(path.suffix == ".route-evidence" for path in read_paths))

  def test_receipt_constructor_cannot_mint_from_an_existing_run(self) -> None:
    with self._temporary_path() as tmp_path:
      run = _run(tmp_path).first
      with self.assertRaisesRegex(TypeError, "production authority"):
        AuthenticatedBehaviorReplayReceipt(
          first=run,
          independent_aa_sha256=run.sha256,
          authority=object(),
        )
      forged = object.__new__(AuthenticatedBehaviorReplayReceipt)
      forged._first = run
      forged._independent_aa_sha256 = run.sha256
      with self.assertRaisesRegex(BehaviorReplayAuthorityError, "invariants"):
        forged.to_dict()

  def test_injected_interface_cannot_be_composed_with_production_mode(self) -> None:
    with self._temporary_path() as tmp_path:
      store, manifest_sha256, profile_path, _ = _store(tmp_path)
      with self.assertRaisesRegex(
        BehaviorReplayAuthorityError,
        "execution mode and interface authority disagree",
      ):
        _execute_authority_once_common(
          evidence_store_root=store,
          import_manifest_sha256=manifest_sha256,
          physical_profile_path=profile_path,
          candidate=behavior_policy(),
          replay_source=_source(),
          interface_registry=INTERFACES,
          verify_execution_source=True,
        )

  def test_verified_public_path_is_the_only_receipt_issuer(self) -> None:
    with self._temporary_path() as tmp_path:
      run = replace(_run(tmp_path).first, production_mode=True)
      with (
        patch(
          "openpilot.selfdrive.controls.lib.blatv2.behavior_replay_authority._verify_replay_source",
          return_value=_source(),
        ),
        patch(
          "openpilot.selfdrive.controls.lib.blatv2.behavior_replay_authority._run_authenticated_behavior_replay",
          return_value=(run, run.sha256),
        ),
      ):
        receipt = run_authenticated_behavior_replay(
          evidence_store_root=Path("/tmp/evidence"),
          import_manifest_sha256="1" * 64,
          physical_profile_path=Path("/tmp/profile"),
          candidate=behavior_policy(),
          replay_source=_source(),
        )
        self.assertIsInstance(receipt, AuthenticatedBehaviorReplayReceipt)
        self.assertIs(receipt.to_dict()["bitExactIndependentAA"], True)

  def test_candidate_must_belong_to_the_committed_grid(self) -> None:
    with self._temporary_path() as tmp_path:
      store, manifest_sha256, profile_path, _ = _store(tmp_path)
      with self.assertRaisesRegex(BehaviorReplayAuthorityError, "outside the committed policy grid"):
        _run_authenticated_behavior_replay_with_registry_for_test(
          evidence_store_root=store,
          import_manifest_sha256=manifest_sha256,
          physical_profile_path=profile_path,
          candidate=replace(behavior_policy(), natural_frequency_per_s=7.123),
          replay_source=_source(),
          interface_registry=INTERFACES,
        )

  def test_source_claims_are_compared_to_actual_head_gitlinks_and_runtime(self) -> None:
    expected = _source()
    cases = (
      replace(expected, source_openpilot_commit="a" * 40),
      replace(expected, opendbc_commit="b" * 40),
      replace(expected, panda_commit="c" * 40),
      replace(expected, source_composition_sha256="d" * 64),
      replace(expected, runtime_identity_sha256="e" * 64),
    )
    for observed in cases:
      with self.subTest(observed=observed), patch(
        "openpilot.selfdrive.controls.lib.blatv2.behavior_replay_authority.inspect_current_replay_source",
        return_value=observed,
      ), self.assertRaisesRegex(BehaviorReplayAuthorityError, "executing bytes"):
        _verify_replay_source(expected)

  def test_dirty_checkout_is_never_execution_authority(self) -> None:
    with self._temporary_path() as root:
      subprocess.run(("git", "init", "-q", root), check=True)
      subprocess.run(("git", "-C", root, "config", "user.name", "test"), check=True)
      subprocess.run(("git", "-C", root, "config", "user.email", "test@example.invalid"), check=True)
      tracked = root / "tracked.py"
      tracked.write_text("value = 1\n", encoding="utf-8")
      subprocess.run(("git", "-C", root, "add", "tracked.py"), check=True)
      subprocess.run(("git", "-C", root, "commit", "-qm", "seed"), check=True)
      _require_clean_checkout(root, "test")
      tracked.write_text("value = 2\n", encoding="utf-8")
      with self.assertRaisesRegex(BehaviorReplayAuthorityError, "not clean"):
        _require_clean_checkout(root, "test")

  def test_public_replay_fails_before_execution_when_source_is_unverified(self) -> None:
    with patch(
      "openpilot.selfdrive.controls.lib.blatv2.behavior_replay_authority._verify_replay_source",
      side_effect=BehaviorReplayAuthorityError("unverified source"),
    ), patch(
      "openpilot.selfdrive.controls.lib.blatv2.behavior_replay_authority._run_authenticated_behavior_replay",
    ) as execute, self.assertRaisesRegex(BehaviorReplayAuthorityError, "unverified source"):
      run_authenticated_behavior_replay(
        evidence_store_root=Path("/tmp/evidence"),
        import_manifest_sha256="1" * 64,
        physical_profile_path=Path("/tmp/profile"),
        candidate=behavior_policy(),
        replay_source=_source(),
      )
    execute.assert_not_called()

  def test_manifest_mismatch_fails_before_any_receipt(self) -> None:
    cases = (
      (lambda manifest: manifest.update(importedRouteCount=2), "counts disagree"),
      (
        lambda manifest: manifest.update(scenarioSourceSetIdentity="0" * 64),
        "source-set identity differs",
      ),
      (
        lambda manifest: manifest["routes"][0]["artifact"].update(path="objects/../escape"),
        "not content-addressed",
      ),
      (
        lambda manifest: manifest["routes"][0]["source"].update(vehicle_identity="other"),
        "source differs from evidence",
      ),
    )
    for mutate, message in cases:
      with self.subTest(message=message), self._temporary_path() as tmp_path:
        store, _, profile_path, manifest = _store(tmp_path)
        mutate(manifest)
        encoded = (canonical_json(manifest) + "\n").encode()
        manifest_sha256 = hashlib.sha256(encoded).hexdigest()
        _write_immutable(store / "imports" / f"{manifest_sha256}.json", encoded)
        with self.assertRaisesRegex(BehaviorReplayAuthorityError, message):
          _run_authenticated_behavior_replay_with_registry_for_test(
            evidence_store_root=store,
            import_manifest_sha256=manifest_sha256,
            physical_profile_path=profile_path,
            candidate=behavior_policy(),
            replay_source=_source(),
            interface_registry=INTERFACES,
          )

  def test_certification_vector_must_bind_the_paired_route_artifact(self) -> None:
    with self._temporary_path() as tmp_path:
      from openpilot.selfdrive.controls.tests.test_blatv2_behavior_training_authority import (
        _certification_for_route,
      )

      store, _, profile_path, manifest = _store(tmp_path)
      row = manifest["routes"][0]
      source = row["source"]
      wrong_vector = _certification_for_route(
        row["routeId"],
        source["route_segment_sha256"][0],
        source["route_segment_size_bytes"][0],
        "f" * 64,
      )
      _write_immutable(
        store / "objects" / f"{wrong_vector.sha256}.cert-vector",
        wrong_vector.canonical_bytes,
      )
      row["artifact"]["certificationVector"].update({
        "path": f"objects/{wrong_vector.sha256}.cert-vector",
        "selectionIdentitySha256": wrong_vector.manifest[
          "selection_identity_sha256"
        ],
        "sha256": wrong_vector.sha256,
        "sizeBytes": len(wrong_vector.canonical_bytes),
      })
      encoded = (canonical_json(manifest) + "\n").encode()
      manifest_sha256 = hashlib.sha256(encoded).hexdigest()
      _write_immutable(store / "imports" / f"{manifest_sha256}.json", encoded)

      with self.assertRaisesRegex(
        BehaviorReplayAuthorityError,
        "certification does not bind its route evidence",
      ):
        _run_authenticated_behavior_replay_with_registry_for_test(
          evidence_store_root=store,
          import_manifest_sha256=manifest_sha256,
          physical_profile_path=profile_path,
          candidate=behavior_policy(),
          replay_source=_source(),
          interface_registry=INTERFACES,
        )

  def test_writable_or_replaced_cas_object_fails_closed(self) -> None:
    with self._temporary_path() as tmp_path:
      store, manifest_sha256, profile_path, manifest = _store(tmp_path)
      first = manifest["routes"][0]["artifact"]
      object_path = store / first["path"]
      object_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
      with self.assertRaisesRegex(BehaviorReplayAuthorityError, "immutable regular file"):
        _run_authenticated_behavior_replay_with_registry_for_test(
          evidence_store_root=store,
          import_manifest_sha256=manifest_sha256,
          physical_profile_path=profile_path,
          candidate=behavior_policy(),
          replay_source=_source(),
          interface_registry=INTERFACES,
        )

  def test_one_bad_imported_route_aborts_the_whole_receipt(self) -> None:
    with self._temporary_path() as tmp_path:
      store, _, profile_path, manifest = _store(tmp_path)
      bad_route = ROUTE_IDS[1]
      row = manifest["routes"][1]
      original_path = store / row["artifact"]["path"]
      artifact = RouteEvidenceArtifact.from_file(original_path)
      controls = tuple(
        replace(
          witness,
          live_torque_parameters_health_exact=(
            witness.live_torque_parameters_health_exact
            and not witness.lateral_active
          ),
        )
        for witness in artifact.control_witnesses
      )
      invalid = RouteEvidenceArtifact(
        artifact.source_identity,
        artifact.car_params_bytes,
        artifact.physical_bytes,
        artifact.model_publications,
        controls,
        artifact.live_torque_parameters,
        artifact.live_delays,
        artifact.lateral_maneuver_plans,
        artifact.event_locators,
      )
      replacement = store / "objects" / f"{invalid.sha256}.route-evidence"
      _write_immutable(replacement, invalid.canonical_bytes)
      row["artifact"].update({
        "path": f"objects/{invalid.sha256}.route-evidence",
        "sha256": invalid.sha256,
        "sizeBytes": len(invalid.canonical_bytes),
      })
      row["source"] = _source_row(invalid)
      from openpilot.selfdrive.controls.tests.test_blatv2_behavior_training_authority import (
        _certification_for_route,
      )
      vector = _certification_for_route(
        bad_route,
        invalid.source_identity.route_segment_sha256[0],
        invalid.source_identity.route_segment_size_bytes[0],
        invalid.sha256,
      )
      _write_immutable(
        store / "objects" / f"{vector.sha256}.cert-vector",
        vector.canonical_bytes,
      )
      row["artifact"]["certificationVector"].update({
        "path": f"objects/{vector.sha256}.cert-vector",
        "selectionIdentitySha256": vector.manifest[
          "selection_identity_sha256"
        ],
        "sha256": vector.sha256,
        "sizeBytes": len(vector.canonical_bytes),
      })
      encoded = (canonical_json(manifest) + "\n").encode()
      changed_manifest_sha256 = hashlib.sha256(encoded).hexdigest()
      _write_immutable(
        store / "imports" / f"{changed_manifest_sha256}.json",
        encoded,
      )
      with self.assertRaisesRegex(
        BehaviorReplayAuthorityError,
        rf"route {bad_route} failed authenticated preparation or replay: active lateral scenario lacks exact stock calibration health",
      ):
        _run_authenticated_behavior_replay_with_registry_for_test(
          evidence_store_root=store,
          import_manifest_sha256=changed_manifest_sha256,
          physical_profile_path=profile_path,
          candidate=behavior_policy(),
          replay_source=_source(),
          interface_registry=INTERFACES,
        )

  def test_independent_aa_mismatch_never_issues_a_receipt(self) -> None:
    with self._temporary_path() as tmp_path:
      store, manifest_sha256, profile_path, _ = _store(tmp_path)
      original = _execute_authority_once_for_test
      calls = 0

      def disagree(**kwargs):
        nonlocal calls
        calls += 1
        result = original(**kwargs)
        if calls == 2:
          return replace(
            result,
            replay_source=replace(
              result.replay_source,
              runtime_identity_sha256="6" * 64,
            ),
          )
        return result

      with (
        patch(
          "openpilot.selfdrive.controls.lib.blatv2.behavior_replay_authority._execute_authority_once_for_test",
          side_effect=disagree,
        ),
        self.assertRaisesRegex(BehaviorReplayAuthorityError, "independent test replay A/A differs"),
      ):
        _run_authenticated_behavior_replay_with_registry_for_test(
          evidence_store_root=store,
          import_manifest_sha256=manifest_sha256,
          physical_profile_path=profile_path,
          candidate=behavior_policy(),
          replay_source=_source(),
          interface_registry=INTERFACES,
        )
      self.assertEqual(calls, 2)

  def test_physical_profile_must_be_immutable_canonical_and_qualified(self) -> None:
    with self._temporary_path() as tmp_path:
      store, manifest_sha256, profile_path, _ = _store(tmp_path)
      profile_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
      profile_path.write_text(json.dumps(physical_profile().to_dict(), indent=2))
      profile_path.chmod(stat.S_IRUSR)
      with self.assertRaisesRegex(BehaviorReplayAuthorityError, "not canonical JSON"):
        _run_authenticated_behavior_replay_with_registry_for_test(
          evidence_store_root=store,
          import_manifest_sha256=manifest_sha256,
          physical_profile_path=profile_path,
          candidate=behavior_policy(),
          replay_source=_source(),
          interface_registry=INTERFACES,
        )
