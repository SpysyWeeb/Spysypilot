from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import canonical_json
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  BehaviorContract,
  BehaviorMetricName,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import (
  BehaviorPolicy,
  CandidateGateVerdict,
  ContractGateVerdict,
  MetricGateVerdict,
  PairedRouteUncertainty,
  PolicyCandidate,
)
from openpilot.selfdrive.controls.lib.blatv2.certification_vector import (
  CERTIFICATION_VECTOR_SCHEMA_VERSION,
  CertificationVector,
  _VECTOR_DOMAIN,
  _scenario_proof_summary,
  certification_vector_selection_identity,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay_authority import (
  BehaviorReplayAuthorityError,
  ReviewedReplaySource,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay_authority import (
  _validate_imported_object as _replay_validate_imported_object,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority import (
  AuthenticatedBehaviorTrainingReceipt,
  BehaviorTrainingAuthorityError,
  BehaviorTrainingRun,
  FrozenTrainingPartition,
  NonAuthoritativeBehaviorTrainingResult,
  RejectedBehaviorTrainingEvidence,
  RobustCandidateGateVerdict,
  RobustHeldOutValidation,
  RobustTrainingSelection,
  TrainingDisposition,
  TrainingSplit,
  _AuthenticatedManifest,
  _PendingRoute,
  _activate_routes,
  _authenticate_manifest,
  _execute_epoch_once_for_test,
  _execute_epoch_once_common,
  _load_authenticated_behavior_training_receipt_common,
  _load_rejected_behavior_training_evidence_common,
  _ordered_map_for_test,
  _run_authenticated_behavior_training_for_test,
  _robust_candidate_verdict,
  _select,
  _strict_cli_request,
  publish_rejected_behavior_training_evidence,
  run_authenticated_behavior_training,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_partition import (
  BehaviorPartitionExclusion,
  BehaviorPartitionError,
  FrozenBehaviorPartition,
  build_behavior_partition,
)
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import RouteEvidenceArtifact
from openpilot.selfdrive.controls.lib.blatv2.counterfactual_plant import CounterfactualPlantMember
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import RouteCandidate, RouteSegment
from openpilot.selfdrive.controls.tests.test_blatv2_behavior_replay_authority import (
  ROUTE_IDS as STORED_ROUTE_IDS,
  _route_artifact,
  _source_row,
  _write_immutable,
)
from openpilot.selfdrive.controls.tests.test_blatv2_behavior_replay import physical_profile
from openpilot.selfdrive.controls.tests.test_blatv2_certification_vector import (
  SEGMENT as _CERTIFICATION_SEGMENT,
  _prepared as _certification_prepared,
  _signed_vector,
  _without_maneuver_plan,
  _with_planes,
)
from openpilot.selfdrive.controls.tests.test_blatv2_route_evidence import artifact as _certification_artifact
from openpilot.selfdrive.controls.lib.blatv2.certification_vector import _segment_vector


def _source() -> ReviewedReplaySource:
  return ReviewedReplaySource(
    source_openpilot_commit="1" * 40,
    opendbc_commit="2" * 40,
    panda_commit="3" * 40,
    source_composition_sha256="4" * 64,
    runtime_identity_sha256="5" * 64,
    module_closure_sha256="6" * 64,
  )


def _pending(index: int) -> _PendingRoute:
  route_id = f"{index:08x}--{index:010x}"
  digest = hashlib.sha256(f"route-{index}".encode()).hexdigest()
  return _PendingRoute(route_id, digest, {}, {"vehicle_identity": "vehicle"}, {})


def _manifest(route_count: int = 200) -> _AuthenticatedManifest:
  return _AuthenticatedManifest(
    "6" * 64,
    "7" * 64,
    "8" * 64,
    Path("/tmp"),
    tuple(_pending(index) for index in range(1, route_count + 1)),
  )


def _test_partition(
  authenticated: _AuthenticatedManifest,
  seed: str,
  minimum_route_count: int,
) -> FrozenBehaviorPartition:
  return build_behavior_partition(
    tuple((route.route_id, route.artifact_sha256) for route in authenticated.pending_routes),
    seed_identity_sha256=seed,
    exclusions=tuple(
      BehaviorPartitionExclusion(route_id, (reason,))
      for route_id, reason in authenticated.excluded_routes
    ),
    minimum_route_count=minimum_route_count,
  )


def _partition() -> FrozenTrainingPartition:
  return build_behavior_partition(
    tuple(
      (
        f"{index:08x}--{index:010x}",
        hashlib.sha256(f"receipt-partition-{index}".encode()).hexdigest(),
      )
      for index in range(1, 201)
    ),
    seed_identity_sha256="9" * 64,
    minimum_route_count=1,
  )


def _empty_run(seed: str = "a", *, transcript_marker: str = "same") -> BehaviorTrainingRun:
  return BehaviorTrainingRun(
    import_manifest_sha256="6" * 64,
    replay_source=_source(),
    physical_profile_sha256="7" * 64,
    physical_generation_sha256="8" * 64,
    robust_plant_set_sha256="9" * 64,
    plant_member_ids=("a" * 64, "b" * 64),
    partition=_partition(),
    controller_seed_sha256=seed * 64,
    coarse_selection=None,
    refinement_selection=None,
    validation=None,
    outer_test=None,
    winner=None,
    disposition=TrainingDisposition.NO_TRAINING_WINNER,
    transcript={
      "coarseSearch": {"candidateArrays": transcript_marker},
      "configuration": {},
      "outerTest": None,
      "refinementSearch": None,
      "schemaVersion": 2,
      "validation": None,
    },
    worker_count=1,
    production_mode=False,
    preparation_count=1,
    controller_replay_count=2,
    route_scan_count=3,
  )


def _passing_candidate_verdict(
  candidate: PolicyCandidate,
  route_ids: tuple[str, ...],
) -> CandidateGateVerdict:
  metric_names = {
    BehaviorContract.SMOOTH: BehaviorMetricName.APPLIED_TORQUE_RATE_RMS,
    BehaviorContract.SWIFT: BehaviorMetricName.CORRECTION_LATENCY_S,
    BehaviorContract.STRONG: BehaviorMetricName.INTEGRATED_CURVATURE_ERROR,
  }
  paired = PairedRouteUncertainty(route_ids, 0.2, 0.0, 0.2, 0.2, 0.0)
  contracts = tuple(
    ContractGateVerdict(
      contract,
      True,
      0.1,
      (MetricGateVerdict(
        metric_names[contract].value,
        contract,
        True,
        0.1,
        (),
        paired,
        paired,
      ),),
    )
    for contract in BehaviorContract
  )
  return CandidateGateVerdict(
    candidate,
    True,
    BehaviorMetricName.INTEGRATED_CURVATURE_ERROR.value,
    0.2,
    0.0,
    True,
    0.1,
    contracts,
    route_ids,
    "1" * 64,
    "2" * 64,
    "3" * 64,
    "4" * 64,
  )


def _accepted_run() -> BehaviorTrainingRun:
  partition = _partition()
  member_ids = ("a" * 64, "b" * 64)
  candidate = PolicyCandidate(0, BehaviorPolicy(8.0, 0.9), 0.0)

  def robust(route_ids: tuple[str, ...]) -> RobustCandidateGateVerdict:
    verdicts = tuple(
      (member_id, _passing_candidate_verdict(candidate, route_ids))
      for member_id in member_ids
    )
    return RobustCandidateGateVerdict(candidate, verdicts, True, member_ids[0], 0.2, 0.1, ())

  training_routes = partition.route_ids(TrainingSplit.TRAINING)
  validation_routes = partition.route_ids(TrainingSplit.VALIDATION)
  test_routes = partition.route_ids(TrainingSplit.TEST)
  training_verdict = robust(training_routes)
  selection = RobustTrainingSelection(
    training_routes,
    member_ids,
    candidate,
    training_verdict,
    (training_verdict,),
    hashlib.sha256(canonical_json([training_verdict.to_dict()["candidate"]]).encode()).hexdigest(),
  )
  validation_verdict = robust(validation_routes)
  test_verdict = robust(test_routes)
  validation = RobustHeldOutValidation(
    selection.sha256, validation_routes, member_ids, True, validation_verdict,
  )
  outer_test = RobustHeldOutValidation(
    selection.sha256, test_routes, member_ids, True, test_verdict,
  )

  def stage_rows(route_ids: tuple[str, ...], split: TrainingSplit, policy: BehaviorPolicy | None):
    return [
      {
        "evaluationSha256": "5" * 64,
        "plantMemberId": member_id,
        "policySha256": None if policy is None else policy.sha256,
        "routeEvaluationSha256s": ["6" * 64 for _ in route_ids],
        "routeIds": list(route_ids),
        "selectorSha256": "7" * 64,
        "split": split.value,
      }
      for member_id in member_ids
    ]

  candidate_row = {
    "candidate": training_verdict.to_dict()["candidate"],
    "memberEvaluations": stage_rows(training_routes, TrainingSplit.TRAINING, candidate.policy),
    "verdict": training_verdict.to_dict(),
  }
  def search(name: str) -> dict[str, object]:
    return {
      "candidateGridSha256": selection.candidate_grid_sha256,
      "candidates": [candidate_row],
      "failureReasons": [],
      "name": name,
      "selection": selection.to_dict(),
      "stockMemberEvaluations": stage_rows(training_routes, TrainingSplit.TRAINING, None),
    }

  def held_out(
    evidence: RobustHeldOutValidation,
    routes: tuple[str, ...],
    split: TrainingSplit,
  ) -> dict[str, object]:
    return {
      "candidateMemberEvaluations": stage_rows(routes, split, candidate.policy),
      "stockMemberEvaluations": stage_rows(routes, split, None),
      "validation": evidence.to_dict(),
    }
  configuration = {
    "controllerSeedSha256": "8" * 64,
    "gateSpecSha256": "9" * 64,
    "metricConfigSha256": "a" * 64,
    "physicalGenerationSha256": "b" * 64,
    "physicalModuleClosureSha256": "c" * 64,
    "physicalProfileSha256": "d" * 64,
    "plantMemberIds": list(member_ids),
    "provisionalDynamicsSha256": "e" * 64,
    "robustPlantSetSha256": "f" * 64,
    "segmentationConfigSha256": "0" * 64,
    "transientReportSha256": "1" * 64,
    "transientRulesSha256": "2" * 64,
  }
  return BehaviorTrainingRun(
    import_manifest_sha256="3" * 64,
    replay_source=_source(),
    physical_profile_sha256="d" * 64,
    physical_generation_sha256="b" * 64,
    robust_plant_set_sha256="f" * 64,
    plant_member_ids=member_ids,
    partition=partition,
    controller_seed_sha256="8" * 64,
    coarse_selection=selection,
    refinement_selection=selection,
    validation=validation,
    outer_test=outer_test,
    winner=candidate.policy,
    disposition=TrainingDisposition.EXTERNAL_REVIEW_CANDIDATE,
    transcript={
      "coarseSearch": search("coarse"),
      "configuration": configuration,
      "outerTest": held_out(outer_test, test_routes, TrainingSplit.TEST),
      "refinementSearch": search("refinement"),
      "schemaVersion": 2,
      "validation": held_out(validation, validation_routes, TrainingSplit.VALIDATION),
    },
    worker_count=4,
    production_mode=True,
    preparation_count=3,
    controller_replay_count=12,
    route_scan_count=6,
  )


def _persist_training_receipt(root: Path, run: BehaviorTrainingRun) -> Path:
  payload = {
    "activationEligible": False,
    "bitExactIndependentAA": True,
    "independentAASha256": run.sha256,
    "result": run.to_dict(),
    "schemaVersion": 2,
  }
  encoded = canonical_json(payload).encode()
  identity = hashlib.sha256(b"blatv2-authenticated-training-receipt-v2\0" + encoded).hexdigest()
  directory = root / identity
  directory.mkdir()
  path = directory / "receipt.json"
  path.write_bytes(encoded)
  path.chmod(0o400)
  return path


def _certification_for_route(
  route_id: str,
  segment_sha256: str,
  segment_size: int,
  route_evidence_sha256: str,
  *,
  behavior_eligible: bool = True,
) -> CertificationVector:
  if behavior_eligible:
    active = _certification_artifact()
    evidence = _with_planes(
      active,
      controls=tuple(replace(value, lateral_active=False) for value in active.control_witnesses),
    )
  else:
    evidence = _without_maneuver_plan()
  base = _signed_vector(_segment_vector(
    segment=_CERTIFICATION_SEGMENT,
    prepared=_certification_prepared(evidence),
  ))
  unsigned = dict(base.manifest)
  unsigned.pop("vector_identity_sha256")
  unsigned["route_name"] = route_id
  unsigned["source_manifest"] = [{
    "index": 0,
    "sha256": segment_sha256,
    "size_bytes": segment_size,
  }]
  unsigned["route_provenance_seed"] = {
    **unsigned["route_provenance_seed"],
    "source_segment_sha256": segment_sha256,
  }
  results = json.loads(json.dumps(unsigned["segment_results"]))
  results[0]["segment"] = {
    "index": 0,
    "sha256": segment_sha256,
    "size_bytes": segment_size,
  }
  results[0]["encoded_source_plane_sha256"][
    "route_evidence_complete"
  ] = route_evidence_sha256
  unsigned["segment_results"] = results
  unsigned["scenario_proof"] = _scenario_proof_summary(results)
  unsigned["bounds"] = {
    **unsigned["bounds"],
    "selected_compressed_bytes": segment_size,
  }
  segment = RouteSegment(
    0,
    _CERTIFICATION_SEGMENT.path,
    segment_sha256,
    segment_size,
  )
  unsigned["selection_identity_sha256"] = certification_vector_selection_identity(
    RouteCandidate(route_id, 1, (segment,)),
    (segment,),
  )
  unsigned_vector = CertificationVector.from_manifest(unsigned)
  signed = dict(unsigned)
  signed["vector_identity_sha256"] = hashlib.sha256(
    _VECTOR_DOMAIN + unsigned_vector.canonical_bytes,
  ).hexdigest()
  return CertificationVector.from_manifest(signed)


def _training_store(
  root: Path,
  *,
  physical_only_last: bool = False,
) -> tuple[Path, str, Path, dict[str, object]]:
  store = root / "evidence"
  objects = store / "objects"
  imports = store / "imports"
  objects.mkdir(parents=True)
  imports.mkdir()
  rows: list[dict[str, object]] = []
  sources: list[dict[str, object]] = []
  for index, route_id in enumerate(STORED_ROUTE_IDS):
    original = _route_artifact(route_id, index)
    source_identity = original.source_identity
    behavior_eligible = not (physical_only_last and index == len(STORED_ROUTE_IDS) - 1)
    eligible_source = type(source_identity)(
      **{
        **source_identity.manifest_dict(),
        "controller_source_kind": "stock_canonical" if behavior_eligible else "ineligible",
        "behavior_eligible": behavior_eligible,
        "behavior_ineligible_reason": (
          "eligible" if behavior_eligible else "lateral_maneuver_plan_missing"
        ),
      },
    )
    artifact = RouteEvidenceArtifact(
      eligible_source,
      original.car_params_bytes,
      original.physical_bytes,
      original.model_publications,
      original.control_witnesses,
      original.live_torque_parameters,
      original.live_delays,
      original.lateral_maneuver_plans,
      original.event_locators,
    )
    _write_immutable(objects / f"{artifact.sha256}.route-evidence", artifact.canonical_bytes)
    segment_sha256 = eligible_source.route_segment_sha256[0]
    segment_size = eligible_source.route_segment_size_bytes[0]
    vector = _certification_for_route(
      route_id,
      segment_sha256,
      segment_size,
      artifact.sha256,
      behavior_eligible=behavior_eligible,
    )
    _write_immutable(objects / f"{vector.sha256}.cert-vector", vector.canonical_bytes)
    source = _source_row(artifact)
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
    sources.append({"routeId": route_id, "source": source})
  manifest: dict[str, object] = {
    "importedRouteCount": len(rows),
    "inspector": {
      "runtimeIdentitySha256": "8" * 64,
      "sourceCompositionSha256": "9" * 64,
      "sourceIdentitySha256": "a" * 64,
      "sourceOpenpilotCommit": "1" * 40,
    },
    "jobStateSha256": "b" * 64,
    "rejectedRouteCount": 0,
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
      "routes": sources,
    }).encode()).hexdigest(),
    "schemaVersion": 2,
  }
  encoded = (canonical_json(manifest) + "\n").encode()
  manifest_sha256 = hashlib.sha256(encoded).hexdigest()
  _write_immutable(imports / f"{manifest_sha256}.json", encoded)
  profile_path = root / "physical-profile.json"
  _write_immutable(profile_path, physical_profile().to_json().encode())
  return store, manifest_sha256, profile_path, manifest


def _twin_trust_generation(
  root: Path,
  *,
  import_manifest_sha256: str = "6" * 64,
  physical_profile_sha256: str = "7" * 64,
  transient_dynamics_sha256: str = "a" * 64,
  source: ReviewedReplaySource | None = None,
  omit_result: bool = False,
) -> Path:
  replay_source = _source() if source is None else source
  fit = {
    "contentSha256": "1" * 64,
    "routeCounter": 1,
    "routeId": "00000001--0000000001",
    "validation": False,
  }
  validation = {
    "contentSha256": "2" * 64,
    "routeCounter": 2,
    "routeId": "00000002--0000000002",
    "validation": True,
  }
  partition = {
    "fitRoutes": [fit],
    "importManifestSha256": import_manifest_sha256,
    "physicalEvidenceSha256": "3" * 64,
    "physicalGenerationSha256": "4" * 64,
    "physicalLedgerSha256": "5" * 64,
    "physicalProfileSha256": physical_profile_sha256,
    "schemaVersion": 1,
    "validationRoutes": [validation],
  }
  partition_sha256 = hashlib.sha256(
    b"blatv2-twin-physical-partition-v1\0" + canonical_json(partition).encode(),
  ).hexdigest()
  parameter_sha256 = hashlib.sha256(
    b"blatv2-twin-parameter-composition-v1\0" + canonical_json({
      "calibrationProfileSha256": physical_profile_sha256,
      "transientDynamicsSha256": transient_dynamics_sha256,
    }).encode(),
  ).hexdigest()
  provenance = {
    "bootstrapMethod": "route_resample_with_replacement",
    "bootstrapResampleCount": 100,
    "bootstrapSeed": 123,
    "datasetManifestSha256": import_manifest_sha256,
    "metricImplementationSha256": "6" * 64,
    "metricSpecSha256": "7" * 64,
    "schemaVersion": 1,
    "sourceCompositionSha256": replay_source.source_composition_sha256,
    "splitManifestSha256": partition_sha256,
    "twinImplementationSha256": "8" * 64,
    "twinParameterProfileSha256": parameter_sha256,
  }
  stratum = {"identity": "low_speed_stick"}
  rules = {
    "horizonRules": [],
    "maximumOneStepRateP99DegS": "1",
    "maximumOneStepRateRmsDegS": "1",
    "maximumOneStepUncertaintyDegS": "1",
    "minimumBootstrapConfidence": "0.95",
    "minimumBootstrapResamples": 100,
    "minimumFitRoutes": 1,
    "minimumOneStepFrames": 1,
    "minimumValidationRoutes": 1,
    "minimumWindows": 1,
    "provenance": provenance,
    "requiredStrata": [stratum],
    "schemaVersion": 1,
  }
  rules_sha256 = hashlib.sha256(
    b"blatv2-twin-trust-rules-v1\0" + canonical_json(rules).encode(),
  ).hexdigest()
  results = [] if omit_result else [{
    "evidence": {"provenance": provenance, "stratum": stratum},
    "passed": True,
    "reasons": [],
    "stratum": stratum,
  }]
  trust_report = {
    "globalReasons": [],
    "results": results,
    "rules": rules,
    "rulesSha256": rules_sha256,
    "schemaVersion": 1,
    "trusted": True,
  }
  report_sha256 = hashlib.sha256(
    b"blatv2-twin-trust-report-v1\0" + canonical_json(trust_report).encode(),
  ).hexdigest()
  route_measurements = [{
    "evaluatedRecordCount": 1,
    "route": {
      "contentSha256": route["contentSha256"],
      "routeId": route["routeId"],
    },
    "sourceVehicleIdentity": "vehicle",
    "strata": [],
  } for route in (fit, validation)]
  measurements_sha256 = hashlib.sha256(
    b"blatv2-twin-route-measurements-v1\0" + canonical_json(route_measurements).encode(),
  ).hexdigest()
  evidence = {
    "builderSchemaVersion": 1,
    "metricImplementationSha256": provenance["metricImplementationSha256"],
    "partition": partition,
    "partitionSha256": partition_sha256,
    "routeMeasurementAABitExact": True,
    "routeMeasurements": route_measurements,
    "routeMeasurementsSha256": measurements_sha256,
    "runtimeIdentitySha256": replay_source.runtime_identity_sha256,
    "sourceCompositionSha256": replay_source.source_composition_sha256,
    "transientDynamicsSha256": transient_dynamics_sha256,
    "trustReport": trust_report,
    "trustReportCanonicalSha256": report_sha256,
  }
  evidence_bytes = canonical_json(evidence).encode()
  generation_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
  generation = root / generation_sha256
  generation.mkdir()
  _write_immutable(generation / "evidence.json", evidence_bytes)
  attestation = {
    "importManifestSha256": import_manifest_sha256,
    "physicalProfileSha256": physical_profile_sha256,
    "runtimeIdentitySha256": replay_source.runtime_identity_sha256,
    "schemaVersion": 1,
    "sourceCompositionSha256": replay_source.source_composition_sha256,
    "trustReportSha256": generation_sha256,
    "trusted": True,
  }
  _write_immutable(generation / "report.json", canonical_json(attestation).encode())
  return generation / "report.json"


@dataclass(frozen=True)
class _WorkerRoute:
  route_id: str


def _fake_worker(route: _WorkerRoute, *unused: object) -> object:
  return SimpleNamespace(route_id=route.route_id)


class TestBehaviorTrainingAuthority(unittest.TestCase):
  def test_public_boundary_has_only_persisted_evidence_inputs(self) -> None:
    self.assertEqual(
      tuple(inspect.signature(run_authenticated_behavior_training).parameters),
      (
        "evidence_store_root",
        "import_manifest_sha256",
        "partition_receipt_path",
        "training_physical_receipt_path",
        "robust_plant_set_path",
        "replay_source",
      ),
    )

  def test_injectable_training_mode_cannot_be_composed_as_production(self) -> None:
    for worker_count, registry in ((1, None), (4, {"CAR": object})):
      with self.subTest(worker_count=worker_count, registry=registry):
        with self.assertRaisesRegex(
          BehaviorTrainingAuthorityError,
          "production training requires four workers and the detected interface",
        ):
          _execute_epoch_once_common(
            evidence_store_root=Path("/tmp/evidence"),
            import_manifest_sha256="1" * 64,
            partition_receipt_path=Path("/tmp/partition-receipt"),
            training_physical_receipt_path=Path("/tmp/physical-receipt"),
            robust_plant_set_path=Path("/tmp/plant-set"),
            replay_source=_source(),
            worker_count=worker_count,
            interface_registry=registry,
            production_mode=True,
          )

  def test_partition_is_disjoint_deterministic_and_append_stable(self) -> None:
    seed = "a" * 64
    initial = _test_partition(_manifest(200), seed, 2)
    repeated = _test_partition(_manifest(200), seed, 2)
    extended = _test_partition(_manifest(260), seed, 2)
    self.assertEqual(initial, repeated)
    self.assertEqual(
      initial.assignments,
      tuple(row for row in extended.assignments if int(row.route_id[:8], 16) <= 200),
    )
    populations = [set(initial.route_ids(split)) for split in TrainingSplit]
    self.assertTrue(all(populations))
    self.assertFalse(populations[0] & populations[1])
    self.assertFalse(populations[0] & populations[2])
    self.assertFalse(populations[1] & populations[2])

  def test_partition_fails_closed_when_a_split_lacks_support(self) -> None:
    with self.assertRaisesRegex(BehaviorPartitionError, "minimum route support"):
      _test_partition(_manifest(2), "a" * 64, 2)

  def test_manifest_authentication_does_not_open_unreleased_route_objects(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      store, manifest_sha256, _, _ = _training_store(Path(directory))
      target = "openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._validate_imported_object"
      with patch(target, wraps=_replay_validate_imported_object) as validate:
        manifest = _authenticate_manifest(store, manifest_sha256)
        self.assertEqual(validate.call_count, 0)
        released = _activate_routes(manifest, (STORED_ROUTE_IDS[0],))
      self.assertEqual(validate.call_count, 1)
      self.assertEqual(tuple(route.route_id for route in released), (STORED_ROUTE_IDS[0],))

  def test_schema_two_keeps_controller_ineligible_route_in_scenario_partition(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      store, manifest_sha256, _, _ = _training_store(
        Path(directory),
        physical_only_last=True,
      )
      authenticated = _authenticate_manifest(store, manifest_sha256)
    self.assertEqual(
      tuple(route.route_id for route in authenticated.pending_routes),
      STORED_ROUTE_IDS,
    )
    self.assertEqual(authenticated.excluded_routes, ())

  def test_empty_bounded_scenario_proof_blocks_instead_of_dropping_route(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      store, manifest_sha256, _, manifest = _training_store(Path(directory))
      authority_ids = manifest["routes"][0]["artifact"]["authorityArtifactIds"]
      with patch(
        "openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._certification_vector_descriptor",
        return_value=({"authorityArtifactIds": authority_ids}, False),
      ), self.assertRaisesRegex(
        BehaviorTrainingAuthorityError,
        "regenerate certification or adjudicate full-route evidence",
      ):
        _authenticate_manifest(store, manifest_sha256)

  def test_certification_descriptor_tamper_fails_before_route_activation(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      store, _, _, original = _training_store(Path(directory))
      manifest = deepcopy(original)
      artifact = manifest["routes"][0]["artifact"]
      artifact["certificationVector"]["selectionIdentitySha256"] = "f" * 64
      encoded = (canonical_json(manifest) + "\n").encode()
      digest = hashlib.sha256(encoded).hexdigest()
      _write_immutable(store / "imports" / f"{digest}.json", encoded)
      with self.assertRaisesRegex(BehaviorTrainingAuthorityError, "certification provenance"):
        _authenticate_manifest(store, digest)

  def test_certification_vector_must_bind_the_paired_route_artifact(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      store, _, _, original = _training_store(Path(directory))
      manifest = deepcopy(original)
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
      digest = hashlib.sha256(encoded).hexdigest()
      _write_immutable(store / "imports" / f"{digest}.json", encoded)

      with self.assertRaisesRegex(
        BehaviorTrainingAuthorityError,
        "certification does not bind its route evidence",
      ):
        _authenticate_manifest(store, digest)

  def test_receipt_cannot_be_minted_from_json_or_existing_run(self) -> None:
    run = _empty_run()
    with self.assertRaisesRegex(TypeError, "production authority"):
      AuthenticatedBehaviorTrainingReceipt(run, run.sha256, object())
    self.assertFalse(hasattr(AuthenticatedBehaviorTrainingReceipt, "from_dict"))
    self.assertFalse(hasattr(AuthenticatedBehaviorTrainingReceipt, "from_json"))
    forged = object.__new__(AuthenticatedBehaviorTrainingReceipt)
    forged._first = run
    forged._independent_aa_sha256 = run.sha256
    with self.assertRaisesRegex(BehaviorTrainingAuthorityError, "cannot serialize"):
      forged.to_dict()
    non_authority = NonAuthoritativeBehaviorTrainingResult(run, run.sha256)
    self.assertIs(non_authority.authoritative, False)
    self.assertFalse(hasattr(non_authority, "to_dict"))

  def test_persisted_receipt_loader_binds_aa_partition_members_and_contracts(self) -> None:
    run = _accepted_run()
    with tempfile.TemporaryDirectory() as directory:
      path = _persist_training_receipt(Path(directory), run)
      evidence = _load_authenticated_behavior_training_receipt_common(
        path,
        verify_replay_source=False,
      )
    self.assertEqual(evidence.result.sha256, run.sha256)
    self.assertEqual(evidence.result.selection.sha256, run.refinement_selection.sha256)
    self.assertEqual(evidence.result.controller_seed_sha256, run.controller_seed_sha256)
    self.assertTrue(all(
      evidence.behavior_contract_passed(contract) for contract in BehaviorContract
    ))

  def test_persisted_receipt_tamper_and_wrong_domain_fail_closed(self) -> None:
    run = _accepted_run()
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      path = _persist_training_receipt(root, run)
      payload = json.loads(path.read_bytes())
      path.chmod(0o600)
      payload["independentAASha256"] = "0" * 64
      path.write_bytes(canonical_json(payload).encode())
      path.chmod(0o400)
      with self.assertRaisesRegex(BehaviorTrainingAuthorityError, "address differs"):
        _load_authenticated_behavior_training_receipt_common(
          path,
          verify_replay_source=False,
        )

  def test_rejection_report_is_persisted_but_never_candidate_authority(self) -> None:
    run = replace(_empty_run(), production_mode=True, worker_count=4)
    rejection = RejectedBehaviorTrainingEvidence(run, run.sha256)
    with tempfile.TemporaryDirectory() as directory, patch(
      "openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._verify_replay_source",
      return_value=run.replay_source,
    ):
      path = publish_rejected_behavior_training_evidence(
        rejection,
        root=Path(directory),
      )
      evidence = _load_rejected_behavior_training_evidence_common(
        path,
        verify_replay_source=False,
      )
      self.assertEqual(path.name, "rejection.json")
      self.assertEqual(evidence.disposition, TrainingDisposition.NO_TRAINING_WINNER)
      self.assertEqual(evidence.result_sha256, run.sha256)
      self.assertFalse(evidence.activation_eligible)
      with self.assertRaisesRegex(BehaviorTrainingAuthorityError, "path is malformed"):
        _load_authenticated_behavior_training_receipt_common(
          path,
          verify_replay_source=False,
        )

  def test_full_aa_rejects_nondeterminism(self) -> None:
    first = _empty_run("a", transcript_marker="first arrays")
    second = _empty_run("a", transcript_marker="different arrays")
    with patch(
      "openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._execute_epoch_once_for_test",
      side_effect=(first, second),
    ) as execute:
      with self.assertRaisesRegex(BehaviorTrainingAuthorityError, "A/A differs"):
        _run_authenticated_behavior_training_for_test(
          evidence_store_root=Path("/tmp"),
          import_manifest_sha256="1" * 64,
          partition_receipt_path=Path("/tmp/partition-receipt"),
          training_physical_receipt_path=Path("/tmp/physical-receipt"),
          robust_plant_set_path=Path("/tmp/plant-set"),
          replay_source=_source(),
          worker_count=1,
          interface_registry=None,
        )
    self.assertEqual(execute.call_count, 2)

  def test_one_and_four_worker_reduction_is_identical(self) -> None:
    calls = tuple((
      _WorkerRoute(f"{index:08x}--{index:010x}"),
    ) for index in range(12, 0, -1))
    one = _ordered_map_for_test(calls, 1, None, worker=_fake_worker)
    four = _ordered_map_for_test(calls, 4, None, worker=_fake_worker)
    self.assertEqual(one, four)
    self.assertEqual(tuple(row.route_id for row in four), tuple(sorted(row.route_id for row in four)))

  def test_cli_surface_rejects_any_extra_control_input(self) -> None:
    request = {
      "evidenceStoreRoot": "/tmp/evidence",
      "importManifestSha256": "1" * 64,
      "partitionReceiptPath": "/tmp/partition-receipt",
      "trainingPhysicalReceiptPath": "/tmp/physical-receipt",
      "replaySource": _source().to_dict(),
      "robustPlantSetPath": "/tmp/plant-set",
    }
    _strict_cli_request(request)
    for key in ("routes", "split", "candidateGrid", "winner", "testRelease"):
      with self.assertRaisesRegex(BehaviorTrainingAuthorityError, "keys"):
        _strict_cli_request({**request, key: True})

  def test_public_training_verifies_source_before_epoch_execution(self) -> None:
    with patch(
      "openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._verify_replay_source",
      side_effect=BehaviorReplayAuthorityError("unverified source"),
    ), patch(
      "openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._run_authenticated_behavior_training",
    ) as execute, self.assertRaisesRegex(BehaviorReplayAuthorityError, "unverified source"):
      run_authenticated_behavior_training(
        evidence_store_root=Path("/tmp/evidence"),
        import_manifest_sha256="1" * 64,
        partition_receipt_path=Path("/tmp/partition-receipt"),
        training_physical_receipt_path=Path("/tmp/physical-receipt"),
        robust_plant_set_path=Path("/tmp/plant-set"),
        replay_source=_source(),
      )
    execute.assert_not_called()

  def test_failed_validation_never_activates_test_route_objects(self) -> None:
    self._exercise_stages(validation_accepted=False, expected_activations=2)

  def test_no_winner_retains_complete_coarse_transcript(self) -> None:
    result, activations = self._exercise_stages(
      validation_accepted=False,
      expected_activations=1,
      coarse_winner=False,
    )
    self.assertEqual(result.disposition, TrainingDisposition.NO_TRAINING_WINNER)
    self.assertEqual(result.transcript["coarseSearch"], {"name": "coarse"})
    self.assertIsNone(result.transcript["refinementSearch"])
    self.assertEqual(len(activations), 1)

  def test_coarse_refinement_validation_and_test_run_in_order(self) -> None:
    result, activations = self._exercise_stages(
      validation_accepted=True,
      expected_activations=3,
    )
    self.assertEqual(result.disposition, TrainingDisposition.EXTERNAL_REVIEW_CANDIDATE)
    self.assertIsNotNone(result.outer_test)
    self.assertEqual(
      [split for split, _ in activations],
      [TrainingSplit.TRAINING, TrainingSplit.VALIDATION, TrainingSplit.TEST],
    )

  def test_module_has_no_live_or_actuation_imports(self) -> None:
    module_path = Path(inspect.getsourcefile(run_authenticated_behavior_training))
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = {
      node.module
      for node in ast.walk(tree)
      if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    forbidden = (
      "cereal.messaging",
      "common.params",
      "openpilot.system.manager",
      "openpilot.selfdrive.car",
    )
    self.assertFalse(any(name.startswith(forbidden) for name in imported))

  def test_offline_authority_import_does_not_load_messaging_or_msgq(self) -> None:
    script = "; ".join((
      "import sys",
      "import openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority",
      "assert 'openpilot.cereal.messaging' not in sys.modules",
      "assert not any(name == 'msgq' or name.startswith('msgq.') for name in sys.modules)",
    ))
    result = subprocess.run(
      (sys.executable, "-c", script),
      cwd=Path(__file__).resolve().parents[4],
      capture_output=True,
      check=False,
      text=True,
      timeout=30,
    )
    self.assertEqual(result.returncode, 0, result.stderr)

  def test_worker_population_is_bounded_and_crashes_are_normalized(self) -> None:
    calls = ((_WorkerRoute("00000001--0000000001"),),)
    with self.assertRaisesRegex(BehaviorTrainingAuthorityError, "worker count"):
      _ordered_map_for_test(calls, 5, None)
    with patch(
      "openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._evaluate_policies_for_route_for_test",
      side_effect=RuntimeError("worker detail"),
    ):
      with self.assertRaisesRegex(BehaviorTrainingAuthorityError, "worker crashed"):
        _ordered_map_for_test(calls, 1, None)

  def test_worker_timeout_cancels_the_bounded_stage(self) -> None:
    calls = ((_WorkerRoute("00000001--0000000001"),),)
    executor = MagicMock()
    future = MagicMock()
    executor.submit.return_value = future
    with (
      patch(
        "openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority.ProcessPoolExecutor",
        return_value=executor,
      ),
      patch(
        "openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority.as_completed",
        side_effect=TimeoutError("deadline"),
      ),
      self.assertRaisesRegex(BehaviorTrainingAuthorityError, "bounded route worker stage failed"),
    ):
      _ordered_map_for_test(calls, 2, None)
    future.cancel.assert_called_once_with()
    executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

  def test_candidate_that_passes_nominal_but_fails_another_member_is_rejected(self) -> None:
    candidate = PolicyCandidate(0, BehaviorPolicy(8.0, 0.9), 0.0)
    member_ids = ("1" * 64, "2" * 64)
    stock = {
      member_id: SimpleNamespace(plant_member_id=member_id, selector=object())
      for member_id in member_ids
    }
    candidates = {
      member_id: SimpleNamespace(plant_member_id=member_id, selector=object())
      for member_id in member_ids
    }
    verdicts = tuple(
      SimpleNamespace(
        candidate=candidate,
        passed=passed,
        target_improvement=0.1,
        worst_contract_margin=0.1,
      )
      for passed in (True, False)
    )
    with (
      patch("openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._bootstrap_alias", return_value=object()),
      patch("openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority.evaluate_candidate", side_effect=verdicts),
    ):
      result = _robust_candidate_verdict(candidate, candidates, stock, SimpleNamespace(
        metric_rules=(), target_metric_name="target",
        paired_uncertainty_method="paired", minimum_paired_route_count=2,
      ))
    self.assertFalse(result.passed)
    self.assertEqual(result.failing_member_ids, (member_ids[1],))

  def test_candidate_is_eligible_only_when_every_member_passes(self) -> None:
    candidate = PolicyCandidate(0, BehaviorPolicy(8.0, 0.9), 0.0)
    member_ids = ("1" * 64, "2" * 64)
    stock = {member_id: SimpleNamespace(plant_member_id=member_id, selector=object()) for member_id in member_ids}
    candidates = {member_id: SimpleNamespace(plant_member_id=member_id, selector=object()) for member_id in member_ids}
    verdicts = tuple(
      SimpleNamespace(
        candidate=candidate,
        passed=True,
        target_improvement=value,
        worst_contract_margin=0.2,
      )
      for value in (0.2, 0.1)
    )
    with (
      patch("openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._bootstrap_alias", return_value=object()),
      patch("openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority.evaluate_candidate", side_effect=verdicts),
    ):
      result = _robust_candidate_verdict(candidate, candidates, stock, SimpleNamespace(
        metric_rules=(), target_metric_name="target",
        paired_uncertainty_method="paired", minimum_paired_route_count=2,
      ))
    self.assertTrue(result.passed)
    self.assertEqual(result.worst_target_improvement, 0.1)

  def test_candidate_and_stock_member_mismatch_fails_closed(self) -> None:
    candidate = PolicyCandidate(0, BehaviorPolicy(8.0, 0.9), 0.0)
    stock = {
      "1" * 64: SimpleNamespace(plant_member_id="1" * 64),
      "2" * 64: SimpleNamespace(plant_member_id="2" * 64),
    }
    candidates = {"1" * 64: SimpleNamespace(plant_member_id="1" * 64)}
    with self.assertRaisesRegex(BehaviorTrainingAuthorityError, "populations differ"):
      _robust_candidate_verdict(candidate, candidates, stock, SimpleNamespace())

  def test_robust_selection_uses_deterministic_worst_member_score(self) -> None:
    first = PolicyCandidate(0, BehaviorPolicy(8.0, 0.8), 0.0)
    second = PolicyCandidate(1, BehaviorPolicy(8.0, 0.9), 0.0)
    member_ids = ("1" * 64, "2" * 64)
    stock = {
      member_id: SimpleNamespace(
        plant_member_id=member_id,
        selector=SimpleNamespace(route_ids=("00000001--0000000001", "00000002--0000000002")),
      )
      for member_id in member_ids
    }

    def robust(candidate: PolicyCandidate, target: float) -> RobustCandidateGateVerdict:
      member_verdicts = tuple(
        (member_id, SimpleNamespace(candidate=candidate, passed=True))
        for member_id in member_ids
      )
      return RobustCandidateGateVerdict(
        candidate, member_verdicts, True, member_ids[0], target, 0.2, (),
      )

    verdicts = (robust(first, 0.1), robust(second, 0.2))
    candidates = {
      first.policy: {member_id: object() for member_id in member_ids},
      second.policy: {member_id: object() for member_id in member_ids},
    }
    with patch(
      "openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._robust_verdicts",
      return_value=verdicts,
    ):
      result = _select((first, second), candidates, stock, SimpleNamespace())
    assert result is not None
    self.assertEqual(result.winner, second)

  def _exercise_stages(
    self,
    *,
    validation_accepted: bool,
    expected_activations: int,
    coarse_winner: bool = True,
  ) -> tuple[BehaviorTrainingRun, list[tuple[TrainingSplit, tuple[str, ...]]]]:
    manifest = _manifest(200)
    gate_seed = "61bf35bd8e920dfaf10042ff76cd29e0ccd854138801300677fa5eabbab33f1d"
    partition = _test_partition(manifest, gate_seed, 2)
    split_by_ids = {
      partition.route_ids(split): split for split in TrainingSplit
    }
    activations: list[tuple[TrainingSplit, tuple[str, ...]]] = []

    def activate(unused_manifest: object, route_ids: tuple[str, ...]):
      split = split_by_ids[route_ids]
      activations.append((split, route_ids))
      return tuple(
        SimpleNamespace(
          route_id=route_id,
          scenario=SimpleNamespace(vehicle_identity="vehicle"),
        )
        for route_id in route_ids
      )

    def ordered(calls: tuple[tuple[object, ...], ...], *unused: object):
      return tuple(
        SimpleNamespace(
          route_id=call[0].route_id,
          preparation=SimpleNamespace(route_id=call[0].route_id),
        )
        for call in calls
      )

    candidate = PolicyCandidate(0, BehaviorPolicy(8.0, 0.9), 0.0)
    selection = SimpleNamespace(winner=candidate, to_dict=lambda: {"winner": candidate.policy.to_dict()})
    validation = SimpleNamespace(
      accepted=validation_accepted,
      to_dict=lambda: {"accepted": validation_accepted},
    )
    accepted_test = SimpleNamespace(accepted=True, to_dict=lambda: {"accepted": True})
    stage = SimpleNamespace(selector=object(), route_results=(SimpleNamespace(),))
    members = tuple(sorted((
      CounterfactualPlantMember.create(
        rack_gain_deg_s2_per_torque=gain,
        rack_damping_per_s=10.0,
        delay_offset_s=0.0,
        unresolved_load_torque=0.0,
      )
      for gain in (2000.0, 4000.0)
    ), key=lambda member: member.member_id))
    physical_receipt = SimpleNamespace(
      generation_sha256="8" * 64,
      evidence_compatibility_sha256="9" * 64,
      partition_receipt_sha256="c" * 64,
      partition_sha256=partition.sha256,
      profile_path=Path("/tmp/profile"),
      profile_sha256="7" * 64,
      module_closure_sha256="d" * 64,
      training_algorithm_identity_sha256="a" * 64,
      training_algorithm_schema_version=11,
    )
    partition_receipt = SimpleNamespace(
      receipt_sha256="c" * 64,
      partition=partition,
      source_composition_sha256="4" * 64,
      runtime_identity_sha256="5" * 64,
    )
    plant_set = SimpleNamespace(
      receipt_sha256="e" * 64,
      transient_report_sha256="f" * 64,
      transient_rules_sha256="0" * 64,
      members=members,
      member_ids=tuple(member.member_id for member in members),
    )
    with (
      patch("openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._authenticate_manifest", return_value=manifest),
      patch("openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._load_training_partition_receipt", return_value=partition_receipt),
      patch("openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._activate_routes", side_effect=activate),
      patch("openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._load_training_physical_receipt", return_value=physical_receipt),
      patch("openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._load_physical_profile", return_value=(object(), "7" * 64)),
      patch("openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority.load_robust_plant_set", return_value=plant_set),
      patch("openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._ordered_map_for_test", side_effect=ordered),
      patch(
        "openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._aggregate_member_stage",
        return_value={member.member_id: stage for member in members},
      ),
      patch("openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._validate_split_coverage"),
      patch(
        "openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._select",
        side_effect=((selection, selection) if coarse_winner else (None,)),
      ),
      patch("openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._refinement_grid", return_value=(candidate,)),
      patch("openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._validate", side_effect=(validation, accepted_test)),
      patch(
        "openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._search_stage_report",
        side_effect=lambda name, *unused: {"name": name},
      ),
      patch(
        "openpilot.selfdrive.controls.lib.blatv2.behavior_training_authority._held_out_stage_report",
        side_effect=lambda *unused: {"heldOut": True},
      ),
    ):
      result = _execute_epoch_once_for_test(
        evidence_store_root=Path("/tmp"),
        import_manifest_sha256="6" * 64,
        partition_receipt_path=Path("/tmp/partition-receipt"),
        training_physical_receipt_path=Path("/tmp/physical-receipt"),
        robust_plant_set_path=Path("/tmp/plant-set"),
        replay_source=_source(),
        worker_count=1,
        interface_registry=None,
      )
    self.assertEqual(len(activations), expected_activations)
    expected_preparations = sum(len(route_ids) for _, route_ids in activations)
    expected_route_scans = expected_preparations * len(members)
    self.assertEqual(result.route_scan_count, expected_route_scans)
    self.assertEqual(result.preparation_count, expected_preparations)
    if not coarse_winner:
      return result, activations
    if not validation_accepted:
      self.assertEqual(result.disposition, TrainingDisposition.VALIDATION_REJECTED)
      self.assertIsNone(result.outer_test)
      self.assertNotIn(TrainingSplit.TEST, [split for split, _ in activations])
    return result, activations


if __name__ == "__main__":
  unittest.main()
