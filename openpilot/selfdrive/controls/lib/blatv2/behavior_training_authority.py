"""End-to-end, non-actuating authority for offline BLaTv2 policy training.

The production boundary accepts immutable route evidence and identities only.
It derives its partition and candidate populations internally, replays the
reviewed stock and modular cores, freezes one training winner before opening
validation, and never reads the outer-test routes unless validation passes.

This module deliberately has no Params, messaging, process-manager, schema,
device, or actuation surface.  A successful receipt is evidence for a later
review; it is never permission to install or enable a controller. The current
single-profile, provisional-dynamics trust input is deliberately emitted with
``activationEligible`` false until an authenticated transient-model set can
score every candidate against its worst-case member.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sys
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from openpilot.selfdrive.controls.lib.blatv2.behavior_configuration import (
  BEHAVIOR_GATE_SPEC_PATH,
  BEHAVIOR_SEGMENTATION_CONFIG_PATH,
  load_behavior_gate_spec,
  load_behavior_segmentation_config,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  ReplayArtifactIdentity,
  ReplayCoreIdentity,
  ReplayRole,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import canonical_json
from openpilot.selfdrive.controls.lib.blatv2.behavior_metrics import (
  BehaviorContract,
  BehaviorMetricName,
  BehaviorScorecard,
  aggregate_behavior_metrics,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import (
  BehaviorPolicy,
  CandidateGateVerdict,
  PolicyCandidate,
  PolicyEvaluation,
  PolicyGridSpec,
  PolicyMetric,
  build_candidate_grid,
  evaluate_candidate,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_partition import (
  BehaviorPartitionError,
  BehaviorPartitionSplit,
  FrozenBehaviorPartition,
  frozen_behavior_partition_from_dict,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_plant_set import (
  RobustPlantSetError,
  load_robust_plant_set,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay import (
  reviewed_replay_core_identity,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay_authority import (
  BehaviorReplayAuthorityError,
  ReviewedReplaySource,
  _ImportedRoute,
  _IMPORT_MANIFEST_MAXIMUM_BYTES,
  _MAXIMUM_MANIFEST_ROUTES,
  _REJECTION_RE,
  _ROUTE_ID_RE,
  _canonical_sha256,
  _load_physical_profile,
  _parse_canonical_file,
  _read_immutable_regular,
  _safe_directory,
  _sha256,
  _strict_import_inspector,
  _uint,
  _validate_imported_object,
  _validate_remote_worker,
  _verify_replay_source,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_route_evaluator import (
  BehaviorRouteEvaluation,
  BehaviorRouteEvaluationError,
  BehaviorRoutePreparation,
  _evaluate_behavior_route_policies_with_registry_for_test,
  evaluate_behavior_route_policies,
)
from openpilot.selfdrive.controls.lib.blatv2.certification_vector import (
  CERTIFICATION_VECTOR_MAX_BYTES,
  CERTIFICATION_VECTOR_SCHEMA_VERSION,
  CertificationVector,
  CertificationVectorError,
)
from openpilot.selfdrive.controls.lib.blatv2.counterfactual_plant import (
  CounterfactualPlantMember,
)
from openpilot.selfdrive.controls.lib.blatv2.policy import ControllerPolicy
from openpilot.selfdrive.controls.lib.blatv2.preparation_contract import (
  BLATV2_LIBRARY_ROOT,
  PROVISIONAL_RACK_DYNAMICS_PATH,
)
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import RouteEvidenceError
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import ProvisionalRackDynamics


BEHAVIOR_TRAINING_AUTHORITY_SCHEMA_VERSION = 2
PROVISIONAL_CONTROLLER_POLICY_PATH = BLATV2_LIBRARY_ROOT / "provisional_controller_policy.json"

_WORKER_COUNT = 4
_TRAINING_PHYSICAL_RECEIPT_MAXIMUM_BYTES = 16 * 1024 * 1024
_TRAINING_PHYSICAL_PROFILE_MAXIMUM_BYTES = 4 * 1024 * 1024
_TRAINING_PHYSICAL_GENERATION_DOMAIN = b"blatv2-training-scoped-physical-generation-v1\0"
_TRAINING_PARTITION_RECEIPT_DOMAIN = b"blatv2-trainer-partition-receipt-v2\0"
_TRAINING_PARTITION_RECEIPT_MAXIMUM_BYTES = 16 * 1024 * 1024
_RECEIPT_DOMAIN = b"blatv2-authenticated-training-receipt-v2\0"
_REJECTION_EVIDENCE_DOMAIN = b"blatv2-rejected-training-evidence-v2\0"
_TRAINING_RECEIPT_MAXIMUM_BYTES = 256 * 1024 * 1024
_TRAINER_IMPORT_MANIFEST_SCHEMA_VERSION = 2
_STAGE_TIMEOUT_SECONDS = 6 * 60 * 60
_MAXIMUM_STAGE_CALLS = _MAXIMUM_MANIFEST_ROUTES


class BehaviorTrainingAuthorityError(RuntimeError):
  """The authenticated training epoch cannot issue a receipt."""


TrainingSplit = BehaviorPartitionSplit
FrozenTrainingPartition = FrozenBehaviorPartition


class TrainingDisposition(StrEnum):
  NO_TRAINING_WINNER = "no_training_winner"
  VALIDATION_REJECTED = "validation_rejected"
  TEST_REJECTED = "test_rejected"
  EXTERNAL_REVIEW_CANDIDATE = "external_review_candidate"


@dataclass(frozen=True, slots=True)
class _PendingRoute:
  route_id: str
  artifact_sha256: str
  artifact: Mapping[str, object]
  source: Mapping[str, object]
  certification_vector: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _AuthenticatedManifest:
  manifest_sha256: str
  scenario_source_set_identity: str
  inspector_identity_sha256: str
  store_root: Path
  pending_routes: tuple[_PendingRoute, ...]
  excluded_routes: tuple[tuple[str, str], ...] = ()
  import_excluded_routes: tuple[tuple[str, tuple[str, ...]], ...] = ()


def _certification_vector_descriptor(
  store_root: Path,
  route_id: str,
  value: object,
  source: Mapping[str, object],
  route_evidence_sha256: str,
) -> tuple[dict[str, object], bool]:
  keys = {
    "authorityArtifactIds",
    "path",
    "schemaVersion",
    "selectionIdentitySha256",
    "sha256",
    "sizeBytes",
  }
  if type(value) is not dict or set(value) != keys:
    raise BehaviorTrainingAuthorityError(f"route {route_id} certification vector is malformed")
  vector_sha256 = _sha256(value["sha256"], f"route {route_id} certification vector")
  selection_sha256 = _sha256(
    value["selectionIdentitySha256"],
    f"route {route_id} certification selection",
  )
  size = _uint(
    value["sizeBytes"],
    f"route {route_id} certification vector size",
    CERTIFICATION_VECTOR_MAX_BYTES,
  )
  if size == 0 or value["schemaVersion"] != CERTIFICATION_VECTOR_SCHEMA_VERSION:
    raise BehaviorTrainingAuthorityError(f"route {route_id} certification version is incompatible")
  if value["path"] != f"objects/{vector_sha256}.cert-vector":
    raise BehaviorTrainingAuthorityError(
      f"route {route_id} certification vector is not content-addressed",
    )
  authority_ids = value["authorityArtifactIds"]
  if type(authority_ids) is not list or len(authority_ids) != 2:
    raise BehaviorTrainingAuthorityError(f"route {route_id} certification authority is malformed")
  for index, authority_id in enumerate(authority_ids):
    _sha256(authority_id, f"route {route_id} certification authority {index + 1}")
  try:
    encoded, _ = _read_immutable_regular(
      store_root / "objects" / f"{vector_sha256}.cert-vector",
      f"route {route_id} certification vector",
      CERTIFICATION_VECTOR_MAX_BYTES,
    )
    vector = CertificationVector.from_bytes(encoded)
  except (BehaviorReplayAuthorityError, CertificationVectorError) as error:
    raise BehaviorTrainingAuthorityError(
      f"route {route_id} certification vector failed authentication",
    ) from error
  if len(encoded) != size or vector.sha256 != vector_sha256:
    raise BehaviorTrainingAuthorityError(f"route {route_id} certification identity differs")
  manifest = vector.manifest
  if (
    manifest.get("route_name") != route_id
    or manifest.get("selection_identity_sha256") != selection_sha256
  ):
    raise BehaviorTrainingAuthorityError(f"route {route_id} certification provenance differs")
  source_segments = manifest.get("source_manifest")
  if type(source_segments) is not list:
    raise BehaviorTrainingAuthorityError(f"route {route_id} certification source is malformed")
  expected_segments = [
    {"index": index, "sha256": sha256, "size_bytes": size_bytes}
    for index, (sha256, size_bytes) in enumerate(zip(
      source.get("route_segment_sha256", ()),
      source.get("route_segment_size_bytes", ()),
      strict=True,
    ))
  ]
  if source_segments != expected_segments:
    raise BehaviorTrainingAuthorityError(f"route {route_id} certification segments differ")
  behavior_eligible = source.get("behavior_eligible")
  behavior_reason = source.get("behavior_ineligible_reason")
  if type(behavior_eligible) is not bool or type(behavior_reason) is not str:
    raise BehaviorTrainingAuthorityError(f"route {route_id} behavior eligibility is malformed")
  if behavior_eligible != (behavior_reason == "eligible"):
    raise BehaviorTrainingAuthorityError(f"route {route_id} behavior eligibility disagrees")
  segment_results = manifest.get("segment_results")
  if type(segment_results) is not list or not segment_results:
    raise BehaviorTrainingAuthorityError(f"route {route_id} certification coverage is empty")
  for segment in segment_results:
    # Recorded-controller reproducibility is immutable provenance, while the
    # independent scenario plane determines counterfactual input eligibility.
    behavior = segment.get("behavior_plane") if type(segment) is dict else None
    scenario = segment.get("scenario_plane") if type(segment) is dict else None
    source_hashes = (
      segment.get("encoded_source_plane_sha256")
      if type(segment) is dict
      else None
    )
    if (
      type(source_hashes) is not dict
      or source_hashes.get("route_evidence_complete")
      != route_evidence_sha256
    ):
      raise BehaviorTrainingAuthorityError(
        f"route {route_id} certification does not bind its route evidence",
      )
    if (
      type(behavior) is not dict
      or behavior.get("source_eligible") is not behavior_eligible
      or behavior.get("source_eligibility_reason") != behavior_reason
    ):
      raise BehaviorTrainingAuthorityError(
        f"route {route_id} recorded-controller certification provenance differs",
      )
    if (
      type(scenario) is not dict
      or type(scenario.get("controls_retained")) is not int
      or scenario["controls_retained"] < 0
      or type(scenario.get("active_controls_retained")) is not int
      or not 0 <= scenario["active_controls_retained"] <= scenario["controls_retained"]
      or type(scenario.get("proof_eligible")) is not bool
      or scenario["proof_eligible"] is not (scenario["controls_retained"] > 0)
    ):
      raise BehaviorTrainingAuthorityError(
        f"route {route_id} selected certification segment lacks scenario input proof",
      )
  scenario_proof = manifest.get("scenario_proof")
  if type(scenario_proof) is not dict or set(scenario_proof) != {
    "active_controls_retained",
    "controls_retained",
    "proof_eligible",
    "selected_inputs_sha256",
  }:
    raise BehaviorTrainingAuthorityError(f"route {route_id} scenario proof is malformed")
  controls_retained = scenario_proof["controls_retained"]
  active_controls_retained = scenario_proof["active_controls_retained"]
  proof_eligible = scenario_proof["proof_eligible"]
  if (
    type(controls_retained) is not int
    or controls_retained < 0
    or type(active_controls_retained) is not int
    or not 0 <= active_controls_retained <= controls_retained
    or type(proof_eligible) is not bool
    or proof_eligible is not (controls_retained > 0)
  ):
    raise BehaviorTrainingAuthorityError(f"route {route_id} scenario proof is inconsistent")
  _sha256(
    scenario_proof["selected_inputs_sha256"],
    f"route {route_id} selected scenario inputs",
  )
  # The bounded vector proves decoding, not maneuver coverage. Complete route
  # replay and metric retention independently require lateral-active support.
  return value, proof_eligible


def _authenticate_manifest(
  store_root: Path,
  manifest_sha256: str,
) -> _AuthenticatedManifest:
  """Authenticate the manifest without opening unreleased route objects."""
  root = _safe_directory(store_root, "route-evidence store")
  _safe_directory(root / "imports", "route-evidence manifest directory")
  _safe_directory(root / "objects", "route-evidence object directory")
  selected_sha256 = _sha256(manifest_sha256, "import manifest")
  manifest, observed_sha256 = _parse_canonical_file(
    root / "imports" / f"{selected_sha256}.json",
    "import manifest",
    _IMPORT_MANIFEST_MAXIMUM_BYTES,
  )
  if observed_sha256 != selected_sha256:
    raise BehaviorTrainingAuthorityError("import manifest content address is invalid")
  if set(manifest) != {
    "importedRouteCount",
    "inspector",
    "jobStateSha256",
    "rejectedRouteCount",
    "remoteWorker",
    "routes",
    "scenarioSourceSetIdentity",
    "schemaVersion",
  } or manifest["schemaVersion"] != _TRAINER_IMPORT_MANIFEST_SCHEMA_VERSION:
    raise BehaviorTrainingAuthorityError("import manifest shape/version is incompatible")
  inspector = _strict_import_inspector(manifest["inspector"])
  _validate_remote_worker(manifest["remoteWorker"])
  _sha256(manifest["jobStateSha256"], "import job state")
  declared_source_set = _sha256(
    manifest["scenarioSourceSetIdentity"],
    "import scenario source set",
  )
  rows = manifest["routes"]
  if type(rows) is not list or not rows or len(rows) > _MAXIMUM_MANIFEST_ROUTES:
    raise BehaviorTrainingAuthorityError("import route population is invalid")
  imported_count = _uint(
    manifest["importedRouteCount"],
    "imported route count",
    _MAXIMUM_MANIFEST_ROUTES,
  )
  rejected_count = _uint(
    manifest["rejectedRouteCount"],
    "rejected route count",
    _MAXIMUM_MANIFEST_ROUTES,
  )
  if imported_count + rejected_count != len(rows):
    raise BehaviorTrainingAuthorityError("import route counts disagree")
  route_ids: list[str] = []
  source_rows: list[dict[str, object]] = []
  pending: list[_PendingRoute] = []
  import_excluded: list[tuple[str, tuple[str, ...]]] = []
  observed_rejected = 0
  for row in rows:
    if type(row) is not dict or set(row) != {
      "artifact", "archiveContentSha256", "rejectionReasons", "routeId", "source", "status",
    }:
      raise BehaviorTrainingAuthorityError("import route row is malformed")
    route_id = row["routeId"]
    if type(route_id) is not str or _ROUTE_ID_RE.fullmatch(route_id) is None:
      raise BehaviorTrainingAuthorityError("import route ID is malformed")
    route_ids.append(route_id)
    archive_sha256 = row["archiveContentSha256"]
    if archive_sha256 is not None:
      _sha256(archive_sha256, f"route {route_id} archive content")
    reasons = row["rejectionReasons"]
    if type(reasons) is not list or any(
      type(reason) is not str or _REJECTION_RE.fullmatch(reason) is None for reason in reasons
    ) or len(reasons) != len(set(reasons)):
      raise BehaviorTrainingAuthorityError(f"route {route_id} rejection reasons are malformed")
    if row["status"] == "rejected":
      observed_rejected += 1
      if not reasons or row["artifact"] is not None or row["source"] is not None:
        raise BehaviorTrainingAuthorityError(f"route {route_id} rejection row is inconsistent")
      import_excluded.append((route_id, tuple(sorted(reasons))))
      continue
    if row["status"] != "imported" or reasons or archive_sha256 is None:
      raise BehaviorTrainingAuthorityError(f"route {route_id} import row is inconsistent")
    artifact = row["artifact"]
    source = row["source"]
    if type(artifact) is not dict or set(artifact) != {
      "authorityArtifactIds", "certificationVector", "path", "sha256", "sizeBytes",
    } or type(source) is not dict:
      raise BehaviorTrainingAuthorityError(f"route {route_id} deferred object row is malformed")
    artifact_sha256 = _sha256(artifact["sha256"], f"route {route_id} artifact")
    _uint(artifact["sizeBytes"], f"route {route_id} artifact size", 1 << 40)
    if artifact["path"] != f"objects/{artifact_sha256}.route-evidence":
      raise BehaviorTrainingAuthorityError(f"route {route_id} object path is not content-addressed")
    authority_ids = artifact["authorityArtifactIds"]
    if type(authority_ids) is not list or len(authority_ids) != 2:
      raise BehaviorTrainingAuthorityError(f"route {route_id} authority pair is malformed")
    for index, value in enumerate(authority_ids):
      _sha256(value, f"route {route_id} authority artifact {index + 1}")
    certification, scenario_eligible = _certification_vector_descriptor(
      root,
      route_id,
      artifact["certificationVector"],
      source,
      artifact_sha256,
    )
    if not scenario_eligible:
      reason = "bounded certification lacks scenario proof; regenerate certification or adjudicate full-route evidence"
      raise BehaviorTrainingAuthorityError(f"route {route_id} {reason}")
    pending.append(_PendingRoute(route_id, artifact_sha256, artifact, source, certification))
    source_rows.append({"routeId": route_id, "source": source})
  if route_ids != sorted(set(route_ids)):
    raise BehaviorTrainingAuthorityError("import routes must be unique and sorted")
  if len(pending) != imported_count or observed_rejected != rejected_count:
    raise BehaviorTrainingAuthorityError("observed import dispositions disagree")
  if len(pending) < 2:
    raise BehaviorTrainingAuthorityError("authenticated training needs multiple routes")
  expected_source_set = _canonical_sha256({
    "domain": "blatv2-trainer-scenario-source-set-v1",
    "routes": source_rows,
  })
  if declared_source_set != expected_source_set:
    raise BehaviorTrainingAuthorityError("import scenario source-set identity differs")
  return _AuthenticatedManifest(
    selected_sha256,
    declared_source_set,
    _canonical_sha256(inspector),
    root,
    tuple(pending),
    (),
    tuple(import_excluded),
  )


def _activate_routes(
  manifest: _AuthenticatedManifest,
  route_ids: Sequence[str],
) -> tuple[_ImportedRoute, ...]:
  pending = {route.route_id: route for route in manifest.pending_routes}
  output: list[_ImportedRoute] = []
  for route_id in route_ids:
    route = pending[route_id]
    try:
      output.append(_validate_imported_object(
        manifest.store_root,
        route.route_id,
        {
          key: value
          for key, value in route.artifact.items()
          if key != "certificationVector"
        },
        route.source,
      ))
    except BehaviorReplayAuthorityError as error:
      raise BehaviorTrainingAuthorityError(str(error)) from error
  identities = {route.scenario.vehicle_identity for route in output}
  if len(identities) != 1:
    raise BehaviorTrainingAuthorityError("released routes mix vehicle identities")
  return tuple(output)


def _sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
  return _sha256_bytes(canonical_json(value).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class _TrainingPhysicalReceipt:
  generation_sha256: str
  partition_receipt_sha256: str
  partition_sha256: str
  profile_path: Path
  profile_sha256: str
  module_closure_sha256: str


@dataclass(frozen=True, slots=True)
class _TrainingPartitionReceipt:
  receipt_sha256: str
  partition: FrozenTrainingPartition
  source_composition_sha256: str
  runtime_identity_sha256: str


def _load_training_partition_receipt(
  path: Path,
  *,
  authenticated: _AuthenticatedManifest,
  expected_seed_identity_sha256: str,
  replay_source: ReviewedReplaySource,
) -> _TrainingPartitionReceipt:
  """Load the pipeline's sole persisted partition instead of re-splitting."""
  if not isinstance(path, Path) or not path.is_absolute() or path.name != "receipt.json":
    raise BehaviorTrainingAuthorityError("training partition receipt path is malformed")
  try:
    _sha256(path.parent.name, "training partition receipt directory")
    if (
      path.parent.resolve(strict=True) != path.parent
      or path.parent.is_symlink()
      or {entry.name for entry in path.parent.iterdir()} != {"receipt.json"}
    ):
      raise BehaviorTrainingAuthorityError("training partition receipt directory is unsafe")
    encoded, _ = _read_immutable_regular(
      path,
      "training partition receipt",
      _TRAINING_PARTITION_RECEIPT_MAXIMUM_BYTES,
    )
    payload = json.loads(encoded)
  except BehaviorTrainingAuthorityError:
    raise
  except BehaviorReplayAuthorityError as error:
    raise BehaviorTrainingAuthorityError(str(error)) from error
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise BehaviorTrainingAuthorityError("training partition receipt is malformed") from error
  keys = {
    "activationEligible", "aaBitExact", "exclusions", "importExclusions",
    "importManifestSha256", "manifestAuthorityModuleSha256", "partition",
    "partitionModuleSha256", "partitionSha256", "productionMode", "routeArtifacts",
    "runtimeIdentitySha256", "schemaVersion", "sourceCompositionSha256",
    "sourceIdentitySha256", "sourceOpenpilotCommit",
  }
  if (
    type(payload) is not dict
    or set(payload) != keys
    or encoded != canonical_json(payload).encode()
    or payload["schemaVersion"] != 2
    or payload["activationEligible"] is not False
    or payload["aaBitExact"] is not True
    or payload["productionMode"] is not True
  ):
    raise BehaviorTrainingAuthorityError("training partition receipt is not production authority")
  receipt_sha256 = hashlib.sha256(
    _TRAINING_PARTITION_RECEIPT_DOMAIN + encoded,
  ).hexdigest()
  if path.parent.name != receipt_sha256:
    raise BehaviorTrainingAuthorityError("training partition receipt address differs")
  for name in (
    "importManifestSha256", "manifestAuthorityModuleSha256", "partitionModuleSha256",
    "partitionSha256", "runtimeIdentitySha256", "sourceCompositionSha256",
    "sourceIdentitySha256",
  ):
    try:
      _sha256(payload[name], f"training partition {name}")
    except BehaviorReplayAuthorityError as error:
      raise BehaviorTrainingAuthorityError(str(error)) from error
  try:
    partition = frozen_behavior_partition_from_dict(payload["partition"])
  except BehaviorPartitionError as error:
    raise BehaviorTrainingAuthorityError("training partition payload is malformed") from error
  partition_module_path = Path(__file__).with_name("behavior_partition.py")
  if (
    payload["partitionSha256"] != partition.sha256
    or partition.seed_identity_sha256 != expected_seed_identity_sha256
    or payload["importManifestSha256"] != authenticated.manifest_sha256
    or payload["sourceOpenpilotCommit"] != replay_source.source_openpilot_commit
    or payload["partitionModuleSha256"] != _sha256_bytes(partition_module_path.read_bytes())
    or payload["manifestAuthorityModuleSha256"] != _sha256_bytes(Path(__file__).read_bytes())
  ):
    raise BehaviorTrainingAuthorityError("training partition receipt belongs to another authority")
  expected_assignments = tuple(
    (route.route_id, route.artifact_sha256) for route in authenticated.pending_routes
  )
  observed_assignments = tuple(
    (row.route_id, row.artifact_sha256) for row in partition.assignments
  )
  expected_exclusions = [
    {"reasons": [reason], "routeId": route_id}
    for route_id, reason in authenticated.excluded_routes
  ]
  expected_import_exclusions = [
    {"reasons": list(reasons), "routeId": route_id}
    for route_id, reasons in authenticated.import_excluded_routes
  ]
  expected_artifacts = [
    {
      "artifactSha256": route.artifact_sha256,
      "routeId": route.route_id,
      "sizeBytes": route.artifact["sizeBytes"],
    }
    for route in authenticated.pending_routes
  ]
  if (
    observed_assignments != expected_assignments
    or payload["exclusions"] != expected_exclusions
    or payload["exclusions"] != partition.to_dict()["exclusions"]
    or payload["importExclusions"] != expected_import_exclusions
    or payload["routeArtifacts"] != expected_artifacts
  ):
    raise BehaviorTrainingAuthorityError("training partition population differs from import authority")
  return _TrainingPartitionReceipt(
    receipt_sha256,
    partition,
    payload["sourceCompositionSha256"],
    payload["runtimeIdentitySha256"],
  )


def _load_training_physical_receipt(
  path: Path,
  *,
  import_manifest_sha256: str,
  partition_receipt: _TrainingPartitionReceipt,
) -> _TrainingPhysicalReceipt:
  partition = partition_receipt.partition
  if not isinstance(path, Path) or not path.is_absolute() or path.name != "receipt.json":
    raise BehaviorTrainingAuthorityError("training physical receipt path is malformed")
  try:
    _sha256(path.parent.name, "training physical generation directory")
  except BehaviorReplayAuthorityError as error:
    raise BehaviorTrainingAuthorityError("training physical receipt is not content-addressed") from error
  try:
    encoded, _ = _read_immutable_regular(
      path,
      "training-scoped physical receipt",
      _TRAINING_PHYSICAL_RECEIPT_MAXIMUM_BYTES,
    )
    payload = json.loads(encoded)
  except BehaviorReplayAuthorityError as error:
    raise BehaviorTrainingAuthorityError(str(error)) from error
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise BehaviorTrainingAuthorityError("training physical receipt is malformed") from error
  keys = {
    "aaBitExact", "acceptedSampleCount", "activationEligible", "allNodesQualified",
    "candidateProfileSha256", "evidenceSha256", "heldOutArtifactsOpened",
    "importManifestSha256", "interpolationQualified", "manifestSha256",
    "moduleClosure", "moduleClosureSha256", "partition", "partitionReceiptSha256",
    "partitionSha256", "productionMode", "rejectedSampleCount",
    "runtimeCarParamsRouteId", "runtimeIdentitySha256", "schemaVersion",
    "selectedProfile", "selectedProfileSha256", "sourceCompositionSha256",
    "sourceIdentitySha256", "trainingRoutes",
  }
  if type(payload) is not dict or set(payload) != keys or encoded != canonical_json(payload).encode():
    raise BehaviorTrainingAuthorityError("training physical receipt is not canonical")
  if (
    payload["schemaVersion"] != 1
    or payload["activationEligible"] is not False
    or payload["heldOutArtifactsOpened"] is not False
    or payload["productionMode"] is not True
    or payload["aaBitExact"] is not True
    or payload["allNodesQualified"] is not True
    or payload["interpolationQualified"] is not True
  ):
    raise BehaviorTrainingAuthorityError("training physical receipt is not qualified")
  generation_sha256 = hashlib.sha256(
    _TRAINING_PHYSICAL_GENERATION_DOMAIN + encoded,
  ).hexdigest()
  if path.parent.name != generation_sha256:
    raise BehaviorTrainingAuthorityError("training physical receipt address differs")
  for name in (
    "evidenceSha256", "importManifestSha256", "manifestSha256", "moduleClosureSha256",
    "partitionReceiptSha256", "partitionSha256", "runtimeIdentitySha256",
    "selectedProfileSha256", "sourceCompositionSha256", "sourceIdentitySha256",
  ):
    _sha256(payload[name], f"training physical {name}")
  candidate = payload["candidateProfileSha256"]
  if candidate is not None:
    _sha256(candidate, "training physical candidate profile")
  for name in ("acceptedSampleCount", "rejectedSampleCount"):
    if type(payload[name]) is not int or payload[name] < 0:
      raise BehaviorTrainingAuthorityError("training physical sample accounting is malformed")
  try:
    persisted_partition = frozen_behavior_partition_from_dict(payload["partition"])
  except BehaviorPartitionError as error:
    raise BehaviorTrainingAuthorityError("training physical partition is malformed") from error
  if (
    payload["importManifestSha256"] != import_manifest_sha256
    or persisted_partition != partition
    or payload["partitionSha256"] != partition.sha256
    or payload["partitionReceiptSha256"] != partition_receipt.receipt_sha256
    or payload["sourceCompositionSha256"] != partition_receipt.source_composition_sha256
    or payload["runtimeIdentitySha256"] != partition_receipt.runtime_identity_sha256
  ):
    raise BehaviorTrainingAuthorityError("training physical receipt belongs to another experiment")
  expected_training = [
    {"artifactSha256": row.artifact_sha256, "routeId": row.route_id}
    for row in partition.assignments
    if row.split is TrainingSplit.TRAINING
  ]
  if payload["trainingRoutes"] != expected_training:
    raise BehaviorTrainingAuthorityError("physical profile was not fit on exact TRAINING only")
  closure = payload["moduleClosure"]
  if (
    type(closure) is not list
    or not closure
    or any(
      type(row) is not dict
      or set(row) != {"path", "sha256", "sizeBytes"}
      or type(row["path"]) is not str
      or not row["path"]
      or type(row["sizeBytes"]) is not int
      or row["sizeBytes"] <= 0
      for row in closure
    )
  ):
    raise BehaviorTrainingAuthorityError("training physical module closure is malformed")
  for row in closure:
    _sha256(row["sha256"], "training physical module")
  if _sha256_json(closure) != payload["moduleClosureSha256"]:
    raise BehaviorTrainingAuthorityError("training physical module closure identity differs")
  profile_path = path.with_name("selected-profile.json")
  try:
    profile_encoded, _ = _read_immutable_regular(
      profile_path,
      "training-scoped selected profile",
      _TRAINING_PHYSICAL_PROFILE_MAXIMUM_BYTES,
    )
    profile_payload = json.loads(profile_encoded)
  except BehaviorReplayAuthorityError as error:
    raise BehaviorTrainingAuthorityError(str(error)) from error
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise BehaviorTrainingAuthorityError("training selected profile is malformed") from error
  if (
    type(profile_payload) is not dict
    or profile_encoded != canonical_json(profile_payload).encode()
    or profile_payload != payload["selectedProfile"]
    or hashlib.sha256(profile_encoded).hexdigest() != payload["selectedProfileSha256"]
  ):
    raise BehaviorTrainingAuthorityError("training selected profile identity differs")
  return _TrainingPhysicalReceipt(
    generation_sha256=generation_sha256,
    partition_receipt_sha256=payload["partitionReceiptSha256"],
    partition_sha256=payload["partitionSha256"],
    profile_path=profile_path,
    profile_sha256=payload["selectedProfileSha256"],
    module_closure_sha256=payload["moduleClosureSha256"],
  )


@dataclass(frozen=True, slots=True)
class _StageEvaluation:
  split: TrainingSplit
  plant_member_id: str
  policy: BehaviorPolicy | None
  route_results: tuple[BehaviorRouteEvaluation, ...]
  scorecard: BehaviorScorecard
  selector: PolicyEvaluation

  def __post_init__(self) -> None:
    _sha256(self.plant_member_id, "stage plant member")
    if not self.route_results:
      raise ValueError("stage evaluation requires route results")
    if any(result.plant_member_id != self.plant_member_id for result in self.route_results):
      raise ValueError("stage evaluation mixes counterfactual plant members")
    route_ids = tuple(sorted(result.scenario.route_id for result in self.route_results))
    if self.selector.route_ids != route_ids:
      raise ValueError("stage selector routes disagree")
    if self.selector.policy != self.policy:
      raise ValueError("stage selector policy disagrees")

  def to_dict(self) -> dict[str, object]:
    return {
      "evaluationSha256": self.sha256,
      "plantMemberId": self.plant_member_id,
      "policySha256": None if self.policy is None else self.policy.sha256,
      "routeEvaluationSha256s": [result.sha256 for result in self.route_results],
      "routeIds": list(self.selector.route_ids),
      "selectorSha256": self.selector.sha256,
      "split": self.split.value,
    }

  @property
  def sha256(self) -> str:
    return _sha256_json({
      "policy": None if self.policy is None else self.policy.to_dict(),
      "plantMemberId": self.plant_member_id,
      "routeEvaluations": [result.sha256 for result in self.route_results],
      "scorecard": self.scorecard.to_dict(),
      "selector": self.selector.to_dict(),
      "split": self.split.value,
    })


def _aggregate_stage(
  split: TrainingSplit,
  plant_member_id: str,
  policy: BehaviorPolicy | None,
  route_results: Sequence[BehaviorRouteEvaluation],
) -> _StageEvaluation:
  results = tuple(sorted(route_results, key=lambda result: result.scenario.route_id))
  if not results:
    raise BehaviorTrainingAuthorityError(f"{split.value} produced no route evaluations")
  artifact = results[0].artifact_identity
  if any(result.artifact_identity != artifact for result in results):
    raise BehaviorTrainingAuthorityError("stage route evaluations mix controller artifacts")
  metric_sha = results[0].metric_config_sha256
  if any(result.metric_config_sha256 != metric_sha for result in results):
    raise BehaviorTrainingAuthorityError("stage route evaluations mix metric authorities")
  # The config object is committed and reloaded by the enclosing authority.
  gate = load_behavior_gate_spec(BEHAVIOR_GATE_SPEC_PATH)
  if gate.metric_config.sha256 != metric_sha:
    raise BehaviorTrainingAuthorityError("route metric authority differs from committed gate")
  scorecard = aggregate_behavior_metrics(
    (window for result in results for window in result.windows),
    gate.metric_config,
  )
  selector = PolicyEvaluation(
    artifact_identity=artifact.to_json(),
    policy=policy,
    route_ids=tuple(result.scenario.route_id for result in results),
    metrics=tuple(
      PolicyMetric.from_scorecard(scorecard, name)
      for name in BehaviorMetricName
    ),
  )
  return _StageEvaluation(split, plant_member_id, policy, results, scorecard, selector)


def _aggregate_member_stage(
  split: TrainingSplit,
  policy: BehaviorPolicy | None,
  policy_index: int,
  batches: Sequence[_RouteBatchResult],
  member_ids: tuple[str, ...],
) -> dict[str, _StageEvaluation]:
  if not batches or member_ids != tuple(sorted(set(member_ids))):
    raise BehaviorTrainingAuthorityError("member stage population is malformed")
  output: dict[str, _StageEvaluation] = {}
  for member_index, member_id in enumerate(member_ids):
    results: list[BehaviorRouteEvaluation] = []
    for batch in batches:
      if (
        len(batch.member_evaluations) != len(member_ids)
        or batch.member_evaluations[member_index].plant_member_id != member_id
      ):
        raise BehaviorTrainingAuthorityError("route batch member ordering differs")
      evaluations = batch.member_evaluations[member_index].evaluations
      if not 0 <= policy_index < len(evaluations):
        raise BehaviorTrainingAuthorityError("route batch policy population differs")
      results.append(evaluations[policy_index])
    output[member_id] = _aggregate_stage(split, member_id, policy, results)
  return output


def _required_strata(gate: object) -> tuple[str, ...]:
  return tuple(sorted({
    stratum
    for rule in gate.metric_rules
    for stratum in rule.required_strata
  }))


def _validate_split_coverage(evaluation: _StageEvaluation, gate: object) -> None:
  present = {
    stratum
    for metric in evaluation.selector.metrics
    for stratum in metric.strata
  }
  missing = tuple(stratum for stratum in _required_strata(gate) if stratum not in present)
  if missing:
    raise BehaviorTrainingAuthorityError(
      f"{evaluation.split.value} lacks required behavior strata: {', '.join(missing)}",
    )


def _bootstrap_alias(stock: _StageEvaluation) -> PolicyEvaluation:
  identity = ReplayArtifactIdentity.compose(
    ReplayRole.CURRENTLY_ACCEPTED,
    stock.route_results[0].artifact_identity.core,
    None,
  )
  return PolicyEvaluation(
    artifact_identity=identity.to_json(),
    policy=None,
    route_ids=stock.selector.route_ids,
    metrics=stock.selector.metrics,
  )


@dataclass(frozen=True, slots=True)
class _MemberRouteEvaluations:
  plant_member_id: str
  evaluations: tuple[BehaviorRouteEvaluation, ...]

  def __post_init__(self) -> None:
    _sha256(self.plant_member_id, "route batch plant member")
    if not self.evaluations or any(
      evaluation.plant_member_id != self.plant_member_id
      for evaluation in self.evaluations
    ):
      raise ValueError("route batch mixes counterfactual plant members")


@dataclass(frozen=True, slots=True)
class _RouteBatchResult:
  route_id: str
  preparation: BehaviorRoutePreparation
  member_evaluations: tuple[_MemberRouteEvaluations, ...]
  source_before: ReviewedReplaySource | None = None
  source_after: ReviewedReplaySource | None = None

  def __post_init__(self) -> None:
    if not self.member_evaluations:
      raise ValueError("route batch has no plant-member evaluations")
    member_ids = tuple(row.plant_member_id for row in self.member_evaluations)
    if member_ids != tuple(sorted(set(member_ids))):
      raise ValueError("route batch plant-member population is not canonical")


def _evaluate_policies_for_route_common(
  route: _ImportedRoute,
  preparation: BehaviorRoutePreparation | None,
  profile: object,
  dynamics: ProvisionalRackDynamics,
  segmentation: object,
  policies: tuple[BehaviorPolicy | None, ...],
  roles: tuple[ReplayRole, ...],
  gate: object,
  exact_stock_core: ReplayCoreIdentity,
  modular_core: ReplayCoreIdentity,
  plant_members: tuple[CounterfactualPlantMember, ...],
  interface_registry: Mapping[str, type] | None,
) -> _RouteBatchResult:
  try:
    if (
      not plant_members
      or plant_members != tuple(sorted(plant_members, key=lambda member: member.member_id))
      or len({member.member_id for member in plant_members}) != len(plant_members)
    ):
      raise BehaviorTrainingAuthorityError("route worker plant-member set is malformed")
    core_identities = tuple(
      exact_stock_core if role is ReplayRole.EXACT_STOCK else modular_core
      for role in roles
    )
    evaluate = (
      evaluate_behavior_route_policies
      if interface_registry is None
      else _evaluate_behavior_route_policies_with_registry_for_test
    )
    frozen_preparation = preparation
    member_results: list[_MemberRouteEvaluations] = []
    for member in plant_members:
      keyword: dict[str, object] = {
        "opponent_roles": roles,
        "core_identities": core_identities,
        "plant_member": member,
        "segmentation_config": segmentation if frozen_preparation is None else None,
      }
      if interface_registry is not None:
        keyword["interface_registry"] = interface_registry
      prepared, results = evaluate(
        route.object_path,
        frozen_preparation,
        profile,
        dynamics,
        policies,
        gate.metric_config,
        **keyword,
      )
      if prepared.scenario != route.scenario:
        raise BehaviorTrainingAuthorityError(
          f"route {route.route_id} preparation changed scenario provenance",
        )
      if frozen_preparation is None:
        frozen_preparation = prepared
      elif prepared != frozen_preparation:
        raise BehaviorTrainingAuthorityError(
          f"route {route.route_id} preparation differs across plant members",
        )
      member_results.append(_MemberRouteEvaluations(member.member_id, results))
    assert frozen_preparation is not None
    return _RouteBatchResult(route.route_id, frozen_preparation, tuple(member_results))
  except BehaviorTrainingAuthorityError:
    raise
  except (BehaviorRouteEvaluationError, RouteEvidenceError, TypeError, ValueError) as error:
    raise BehaviorTrainingAuthorityError(
      f"route {route.route_id} failed preparation or replay: {error}",
    ) from error


def _evaluate_policies_for_route(
  route: _ImportedRoute,
  preparation: BehaviorRoutePreparation | None,
  profile: object,
  dynamics: ProvisionalRackDynamics,
  segmentation: object,
  policies: tuple[BehaviorPolicy | None, ...],
  roles: tuple[ReplayRole, ...],
  gate: object,
  exact_stock_core: ReplayCoreIdentity,
  modular_core: ReplayCoreIdentity,
  plant_members: tuple[CounterfactualPlantMember, ...],
  replay_source: ReviewedReplaySource,
) -> _RouteBatchResult:
  before = _verify_replay_source(replay_source)
  result = _evaluate_policies_for_route_common(
    route,
    preparation,
    profile,
    dynamics,
    segmentation,
    policies,
    roles,
    gate,
    exact_stock_core,
    modular_core,
    plant_members,
    None,
  )
  after = _verify_replay_source(replay_source)
  if before != replay_source or after != replay_source:
    raise BehaviorTrainingAuthorityError(
      f"route {route.route_id} worker source identity changed",
    )
  return replace(result, source_before=before, source_after=after)


def _evaluate_policies_for_route_for_test(
  route: _ImportedRoute,
  preparation: BehaviorRoutePreparation | None,
  profile: object,
  dynamics: ProvisionalRackDynamics,
  segmentation: object,
  policies: tuple[BehaviorPolicy | None, ...],
  roles: tuple[ReplayRole, ...],
  gate: object,
  exact_stock_core: ReplayCoreIdentity,
  modular_core: ReplayCoreIdentity,
  plant_members: tuple[CounterfactualPlantMember, ...],
  interface_registry: Mapping[str, type] | None,
) -> _RouteBatchResult:
  return _evaluate_policies_for_route_common(
    route,
    preparation,
    profile,
    dynamics,
    segmentation,
    policies,
    roles,
    gate,
    exact_stock_core,
    modular_core,
    plant_members,
    interface_registry,
  )


def _ordered_map(
  calls: tuple[tuple[object, ...], ...],
  replay_source: ReviewedReplaySource,
) -> tuple[_RouteBatchResult, ...]:
  """Run every production batch in the fixed four-process pool."""
  if not calls or len(calls) > _MAXIMUM_STAGE_CALLS:
    raise BehaviorTrainingAuthorityError("training stage route population is outside its bound")
  if not isinstance(replay_source, ReviewedReplaySource):
    raise TypeError("production route map requires reviewed replay source")
  try:
    executor = ProcessPoolExecutor(max_workers=_WORKER_COUNT)
    futures = tuple(
      executor.submit(_evaluate_policies_for_route_star, (*call, replay_source))
      for call in calls
    )
    collected: list[_RouteBatchResult] = []
    try:
      for future in as_completed(futures, timeout=_STAGE_TIMEOUT_SECONDS):
        collected.append(future.result())
    except Exception:
      for future in futures:
        future.cancel()
      executor.shutdown(wait=False, cancel_futures=True)
      raise
    else:
      executor.shutdown(wait=True)
  except BehaviorTrainingAuthorityError:
    raise
  except (BrokenProcessPool, FuturesTimeoutError, MemoryError, OSError) as error:
    raise BehaviorTrainingAuthorityError("bounded route worker stage failed") from error
  except Exception as error:
    raise BehaviorTrainingAuthorityError("route worker crashed") from error
  values = tuple(sorted(collected, key=lambda value: value.route_id))
  if any(
    value.source_before != replay_source or value.source_after != replay_source
    for value in values
  ):
    raise BehaviorTrainingAuthorityError("worker module/source identity differs from parent")
  return values


def _ordered_map_for_test(
  calls: tuple[tuple[object, ...], ...],
  worker_count: int,
  interface_registry: Mapping[str, type] | None,
) -> tuple[_RouteBatchResult, ...]:
  if not calls or len(calls) > _MAXIMUM_STAGE_CALLS:
    raise BehaviorTrainingAuthorityError("training stage route population is outside its bound")
  if type(worker_count) is not int or not 1 <= worker_count <= _WORKER_COUNT:
    raise BehaviorTrainingAuthorityError("training stage worker count is outside its bound")
  try:
    if interface_registry is not None or worker_count == 1:
      values = tuple(
        _evaluate_policies_for_route_for_test(*call, interface_registry)
        for call in calls
      )
    else:
      executor = ProcessPoolExecutor(max_workers=worker_count)
      futures = tuple(
        executor.submit(
          _evaluate_policies_for_route_for_test_star,
          (*call, interface_registry),
        )
        for call in calls
      )
      collected: list[_RouteBatchResult] = []
      try:
        for future in as_completed(futures, timeout=_STAGE_TIMEOUT_SECONDS):
          collected.append(future.result())
      except Exception:
        for future in futures:
          future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
      else:
        executor.shutdown(wait=True)
      values = tuple(collected)
  except BehaviorTrainingAuthorityError:
    raise
  except (BrokenProcessPool, FuturesTimeoutError, MemoryError, OSError) as error:
    raise BehaviorTrainingAuthorityError("bounded route worker stage failed") from error
  except Exception as error:
    raise BehaviorTrainingAuthorityError("route worker crashed") from error
  return tuple(sorted(values, key=lambda value: value.route_id))


def _evaluate_policies_for_route_star(arguments: tuple[object, ...]) -> _RouteBatchResult:
  return _evaluate_policies_for_route(*arguments)


def _evaluate_policies_for_route_for_test_star(
  arguments: tuple[object, ...],
) -> _RouteBatchResult:
  return _evaluate_policies_for_route_for_test(*arguments)


def _calls(
  routes: Sequence[_ImportedRoute],
  preparations: Mapping[str, BehaviorRoutePreparation],
  profile: object,
  dynamics: ProvisionalRackDynamics,
  segmentation: object,
  policies: tuple[BehaviorPolicy | None, ...],
  roles: tuple[ReplayRole, ...],
  gate: object,
  exact_stock_core: ReplayCoreIdentity,
  modular_core: ReplayCoreIdentity,
  plant_members: tuple[CounterfactualPlantMember, ...],
) -> tuple[tuple[object, ...], ...]:
  return tuple(
    (
      route,
      preparations.get(route.route_id),
      profile,
      dynamics,
      segmentation,
      policies,
      roles,
      gate,
      exact_stock_core,
      modular_core,
      plant_members,
    )
    for route in routes
  )


def _refinement_grid(gate: object, winner: PolicyCandidate) -> tuple[PolicyCandidate, ...]:
  coarse = gate.candidate_grid

  def local(offsets: tuple[float, ...]) -> tuple[float, ...]:
    negative = max(value for value in offsets if value < 0.0)
    positive = min(value for value in offsets if value > 0.0)
    return negative * 0.5, 0.0, positive * 0.5

  return build_candidate_grid(PolicyGridSpec(
    incumbent=winner.policy,
    natural_frequency_log_offsets=local(coarse.natural_frequency_log_offsets),
    damping_ratio_log_offsets=local(coarse.damping_ratio_log_offsets),
    minimum_natural_frequency_per_s=coarse.minimum_natural_frequency_per_s,
    maximum_natural_frequency_per_s=coarse.maximum_natural_frequency_per_s,
    minimum_damping_ratio=coarse.minimum_damping_ratio,
    maximum_damping_ratio=coarse.maximum_damping_ratio,
  ))


@dataclass(frozen=True, slots=True)
class RobustCandidateGateVerdict:
  candidate: PolicyCandidate
  member_verdicts: tuple[tuple[str, CandidateGateVerdict], ...]
  passed: bool
  worst_member_id: str
  worst_target_improvement: float | None
  worst_contract_margin: float | None
  failing_member_ids: tuple[str, ...]

  def __post_init__(self) -> None:
    if not self.member_verdicts:
      raise ValueError("robust verdict requires plant members")
    member_ids = tuple(member_id for member_id, _ in self.member_verdicts)
    if member_ids != tuple(sorted(set(member_ids))):
      raise ValueError("robust verdict member population is not canonical")
    if any(verdict.candidate != self.candidate for _, verdict in self.member_verdicts):
      raise ValueError("robust verdict mixes policy candidates")
    expected_failing = tuple(
      member_id for member_id, verdict in self.member_verdicts if not verdict.passed
    )
    if self.failing_member_ids != expected_failing or self.passed != (not expected_failing):
      raise ValueError("robust verdict disposition disagrees with its members")
    if self.worst_member_id not in member_ids:
      raise ValueError("robust verdict worst member is absent")
    if self.passed and (
      self.worst_target_improvement is None or self.worst_contract_margin is None
    ):
      raise ValueError("passing robust verdict lacks a finite worst-member score")

  def to_dict(self) -> dict[str, object]:
    return {
      "candidate": {
        "canonicalIndex": self.candidate.canonical_index,
        "policy": self.candidate.policy.to_dict(),
        "squaredLogDisplacement": self.candidate.squared_log_displacement,
      },
      "failingMemberIds": list(self.failing_member_ids),
      "memberVerdicts": [
        {"memberId": member_id, "verdict": verdict.to_dict()}
        for member_id, verdict in self.member_verdicts
      ],
      "passed": self.passed,
      "worstContractMargin": self.worst_contract_margin,
      "worstMemberId": self.worst_member_id,
      "worstTargetImprovement": self.worst_target_improvement,
    }

  @property
  def sha256(self) -> str:
    return _sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class RobustTrainingSelection:
  training_route_ids: tuple[str, ...]
  plant_member_ids: tuple[str, ...]
  winner: PolicyCandidate
  winner_verdict: RobustCandidateGateVerdict
  all_verdicts: tuple[RobustCandidateGateVerdict, ...]
  candidate_grid_sha256: str

  def __post_init__(self) -> None:
    if not self.training_route_ids or self.training_route_ids != tuple(sorted(set(self.training_route_ids))):
      raise ValueError("robust selection training routes are not canonical")
    if (
      len(self.plant_member_ids) < 2
      or self.plant_member_ids != tuple(sorted(set(self.plant_member_ids)))
    ):
      raise ValueError("robust selection requires a bounded canonical member set")
    if self.winner_verdict.candidate != self.winner or not self.winner_verdict.passed:
      raise ValueError("robust selection winner is not unanimously passing")
    if self.winner_verdict not in self.all_verdicts:
      raise ValueError("robust selection winner is absent from the transcript")
    _sha256(self.candidate_grid_sha256, "robust selection candidate grid")

  def to_dict(self) -> dict[str, object]:
    return {
      "allVerdicts": [verdict.to_dict() for verdict in self.all_verdicts],
      "candidateGridSha256": self.candidate_grid_sha256,
      "plantMemberIds": list(self.plant_member_ids),
      "trainingRouteIds": list(self.training_route_ids),
      "winnerCanonicalIndex": self.winner.canonical_index,
      "winnerPolicy": self.winner.policy.to_dict(),
      "winnerVerdict": self.winner_verdict.to_dict(),
    }

  @property
  def sha256(self) -> str:
    return _sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class RobustHeldOutValidation:
  selection_sha256: str
  validation_route_ids: tuple[str, ...]
  plant_member_ids: tuple[str, ...]
  accepted: bool
  frozen_winner_verdict: RobustCandidateGateVerdict

  def __post_init__(self) -> None:
    _sha256(self.selection_sha256, "robust held-out selection")
    if not self.validation_route_ids or self.validation_route_ids != tuple(sorted(set(self.validation_route_ids))):
      raise ValueError("robust held-out routes are not canonical")
    if self.plant_member_ids != tuple(
      member_id for member_id, _ in self.frozen_winner_verdict.member_verdicts
    ):
      raise ValueError("robust held-out member set differs from training")
    if self.accepted != self.frozen_winner_verdict.passed:
      raise ValueError("robust held-out disposition disagrees")

  def to_dict(self) -> dict[str, object]:
    return {
      "accepted": self.accepted,
      "frozenWinnerVerdict": self.frozen_winner_verdict.to_dict(),
      "plantMemberIds": list(self.plant_member_ids),
      "selectionSha256": self.selection_sha256,
      "validationRouteIds": list(self.validation_route_ids),
    }


def _robust_candidate_verdict(
  candidate: PolicyCandidate,
  candidates: Mapping[str, _StageEvaluation],
  stock: Mapping[str, _StageEvaluation],
  gate: object,
) -> RobustCandidateGateVerdict:
  member_ids = tuple(sorted(stock))
  if (
    len(member_ids) < 2
    or set(candidates) != set(member_ids)
    or any(stock[member_id].plant_member_id != member_id for member_id in member_ids)
    or any(candidates[member_id].plant_member_id != member_id for member_id in member_ids)
  ):
    raise BehaviorTrainingAuthorityError("candidate/stock plant-member populations differ")
  verdicts = tuple(
    (
      member_id,
      evaluate_candidate(
        candidate,
        candidates[member_id].selector,
        stock[member_id].selector,
        _bootstrap_alias(stock[member_id]),
        gate.metric_rules,
        gate.target_metric_name,
        gate.paired_uncertainty_method,
        gate.minimum_paired_route_count,
      ),
    )
    for member_id in member_ids
  )
  failing = tuple(member_id for member_id, verdict in verdicts if not verdict.passed)
  scored = tuple(
    (member_id, verdict.target_improvement, verdict.worst_contract_margin)
    for member_id, verdict in verdicts
    if verdict.target_improvement is not None and verdict.worst_contract_margin is not None
  )
  if scored:
    worst_member_id = min(
      scored,
      key=lambda row: (row[1], row[2], row[0]),
    )[0]
    worst_target = min(row[1] for row in scored)
    worst_margin = min(row[2] for row in scored)
  else:
    worst_member_id, worst_target, worst_margin = member_ids[0], None, None
  return RobustCandidateGateVerdict(
    candidate=candidate,
    member_verdicts=verdicts,
    passed=not failing,
    worst_member_id=worst_member_id,
    worst_target_improvement=worst_target,
    worst_contract_margin=worst_margin,
    failing_member_ids=failing,
  )


def _robust_verdicts(
  grid: tuple[PolicyCandidate, ...],
  candidates: Mapping[BehaviorPolicy, Mapping[str, _StageEvaluation]],
  stock: Mapping[str, _StageEvaluation],
  gate: object,
) -> tuple[RobustCandidateGateVerdict, ...]:
  if set(candidates) != {candidate.policy for candidate in grid}:
    raise BehaviorTrainingAuthorityError("robust candidate evaluations do not cover the grid")
  return tuple(
    _robust_candidate_verdict(candidate, candidates[candidate.policy], stock, gate)
    for candidate in grid
  )


def _select(
  grid: tuple[PolicyCandidate, ...],
  candidates: Mapping[BehaviorPolicy, Mapping[str, _StageEvaluation]],
  stock: Mapping[str, _StageEvaluation],
  gate: object,
) -> RobustTrainingSelection | None:
  verdicts = _robust_verdicts(grid, candidates, stock, gate)
  passing = tuple(verdict for verdict in verdicts if verdict.passed)
  if not passing:
    return None
  winner = min(
    passing,
    key=lambda verdict: (
      -float(verdict.worst_target_improvement),
      -float(verdict.worst_contract_margin),
      verdict.candidate.squared_log_displacement,
      verdict.candidate.canonical_index,
    ),
  )
  member_ids = tuple(sorted(stock))
  route_ids = stock[member_ids[0]].selector.route_ids
  if any(value.selector.route_ids != route_ids for value in stock.values()):
    raise BehaviorTrainingAuthorityError("plant members use different training routes")
  return RobustTrainingSelection(
    training_route_ids=route_ids,
    plant_member_ids=member_ids,
    winner=winner.candidate,
    winner_verdict=winner,
    all_verdicts=verdicts,
    candidate_grid_sha256=_sha256_json([
      {
        "canonicalIndex": candidate.canonical_index,
        "policy": candidate.policy.to_dict(),
        "squaredLogDisplacement": candidate.squared_log_displacement,
      }
      for candidate in grid
    ]),
  )


def _search_stage_report(
  name: str,
  grid: tuple[PolicyCandidate, ...],
  candidates: Mapping[BehaviorPolicy, Mapping[str, _StageEvaluation]],
  stock: Mapping[str, _StageEvaluation],
  gate: object,
  selection: RobustTrainingSelection | None,
) -> dict[str, object]:
  verdicts = _robust_verdicts(grid, candidates, stock, gate)
  if selection is not None and tuple(selection.all_verdicts) != verdicts:
    raise BehaviorTrainingAuthorityError(f"{name} selector transcript differs")
  rows = []
  for candidate, verdict in zip(grid, verdicts, strict=True):
    evaluations = candidates[candidate.policy]
    rows.append({
      "candidate": {
        "canonicalIndex": candidate.canonical_index,
        "policy": candidate.policy.to_dict(),
        "squaredLogDisplacement": candidate.squared_log_displacement,
      },
      "memberEvaluations": [
        evaluations[member_id].to_dict() for member_id in sorted(evaluations)
      ],
      "verdict": verdict.to_dict(),
    })
  grid_sha256 = _sha256_json([row["candidate"] for row in rows])
  return {
    "candidateGridSha256": grid_sha256,
    "candidates": rows,
    "failureReasons": sorted({
      reason
      for verdict in verdicts
      for _, member_verdict in verdict.member_verdicts
      for contract in member_verdict.contracts
      for metric in contract.metrics
      for reason in metric.reasons
    }),
    "name": name,
    "selection": None if selection is None else selection.to_dict(),
    "stockMemberEvaluations": [stock[member_id].to_dict() for member_id in sorted(stock)],
  }


def _held_out_stage_report(
  stock: Mapping[str, _StageEvaluation],
  winner: Mapping[str, _StageEvaluation],
  validation: RobustHeldOutValidation,
) -> dict[str, object]:
  return {
    "candidateMemberEvaluations": [winner[member_id].to_dict() for member_id in sorted(winner)],
    "stockMemberEvaluations": [stock[member_id].to_dict() for member_id in sorted(stock)],
    "validation": validation.to_dict(),
  }


def _validate(
  selection: RobustTrainingSelection,
  winner: Mapping[str, _StageEvaluation],
  stock: Mapping[str, _StageEvaluation],
  gate: object,
) -> RobustHeldOutValidation:
  verdict = _robust_candidate_verdict(selection.winner, winner, stock, gate)
  member_ids = tuple(sorted(stock))
  if member_ids != selection.plant_member_ids:
    raise BehaviorTrainingAuthorityError("held-out plant-member set changed after training")
  route_ids = stock[member_ids[0]].selector.route_ids
  if set(selection.training_route_ids) & set(route_ids):
    raise BehaviorTrainingAuthorityError("training and held-out routes overlap")
  return RobustHeldOutValidation(
    selection_sha256=selection.sha256,
    validation_route_ids=route_ids,
    plant_member_ids=member_ids,
    accepted=verdict.passed,
    frozen_winner_verdict=verdict,
  )


@dataclass(frozen=True, slots=True)
class BehaviorTrainingRun:
  import_manifest_sha256: str
  replay_source: ReviewedReplaySource
  physical_profile_sha256: str
  physical_generation_sha256: str
  robust_plant_set_sha256: str
  plant_member_ids: tuple[str, ...]
  partition: FrozenTrainingPartition
  controller_seed_sha256: str
  coarse_selection: RobustTrainingSelection | None
  refinement_selection: RobustTrainingSelection | None
  validation: RobustHeldOutValidation | None
  outer_test: RobustHeldOutValidation | None
  winner: BehaviorPolicy | None
  disposition: TrainingDisposition
  transcript: Mapping[str, object]
  worker_count: int
  production_mode: bool
  preparation_count: int
  controller_replay_count: int
  route_scan_count: int

  def __post_init__(self) -> None:
    if not isinstance(self.replay_source, ReviewedReplaySource):
      raise TypeError("training run requires reviewed replay source")
    for name in (
      "import_manifest_sha256", "physical_profile_sha256", "physical_generation_sha256",
      "robust_plant_set_sha256", "controller_seed_sha256",
    ):
      _sha256(getattr(self, name), f"training run {name}")
    if not isinstance(self.partition, FrozenTrainingPartition):
      raise TypeError("training run requires a frozen partition")
    if (
      len(self.plant_member_ids) < 2
      or self.plant_member_ids != tuple(sorted(set(self.plant_member_ids)))
    ):
      raise ValueError("training run plant-member set is not robust and canonical")
    if type(self.worker_count) is not int or not 1 <= self.worker_count <= _WORKER_COUNT:
      raise ValueError("training run worker count is outside its bound")
    if type(self.production_mode) is not bool:
      raise TypeError("training run production mode must be boolean")
    for name in ("preparation_count", "controller_replay_count", "route_scan_count"):
      if type(getattr(self, name)) is not int or getattr(self, name) < 0:
        raise ValueError(f"training run {name} must be nonnegative")
    if self.route_scan_count < self.preparation_count:
      raise ValueError("training run scan count is inconsistent")
    expected_transcript = {
      "coarseSearch",
      "configuration",
      "outerTest",
      "refinementSearch",
      "schemaVersion",
      "validation",
    }
    if type(self.transcript) is not dict or set(self.transcript) != expected_transcript:
      raise ValueError("training run transcript is incomplete")
    if self.transcript["schemaVersion"] != 2 or type(self.transcript["coarseSearch"]) is not dict:
      raise ValueError("training run transcript schema is invalid")
    if self.disposition is TrainingDisposition.NO_TRAINING_WINNER:
      valid = (
        self.coarse_selection is None
        and self.refinement_selection is None
        and self.validation is None
        and self.outer_test is None
        and self.winner is None
        and self.transcript["refinementSearch"] is None
        and self.transcript["validation"] is None
        and self.transcript["outerTest"] is None
      )
    elif self.disposition is TrainingDisposition.VALIDATION_REJECTED:
      valid = (
        self.coarse_selection is not None
        and self.refinement_selection is not None
        and self.validation is not None
        and not self.validation.accepted
        and self.outer_test is None
        and self.winner == self.refinement_selection.winner.policy
        and self.transcript["validation"] is not None
        and self.transcript["outerTest"] is None
      )
    elif self.disposition is TrainingDisposition.TEST_REJECTED:
      valid = (
        self.coarse_selection is not None
        and self.refinement_selection is not None
        and self.validation is not None
        and self.validation.accepted
        and self.outer_test is not None
        and not self.outer_test.accepted
        and self.winner == self.refinement_selection.winner.policy
        and self.transcript["validation"] is not None
        and self.transcript["outerTest"] is not None
      )
    else:
      valid = (
        self.disposition is TrainingDisposition.EXTERNAL_REVIEW_CANDIDATE
        and self.coarse_selection is not None
        and self.refinement_selection is not None
        and self.validation is not None
        and self.validation.accepted
        and self.outer_test is not None
        and self.outer_test.accepted
        and self.winner == self.refinement_selection.winner.policy
        and self.transcript["validation"] is not None
        and self.transcript["outerTest"] is not None
      )
    if not valid:
      raise ValueError("training run disposition and stage state disagree")

  def to_dict(self) -> dict[str, object]:
    return {
      "acceptedIncumbentMode": "bootstrap_exact_stock_only",
      "activationEligible": False,
      "controllerReplayCount": self.controller_replay_count,
      "controllerSeedSha256": self.controller_seed_sha256,
      "coarseSelection": None if self.coarse_selection is None else self.coarse_selection.to_dict(),
      "disposition": self.disposition.value,
      "importManifestSha256": self.import_manifest_sha256,
      "outerTest": None if self.outer_test is None else self.outer_test.to_dict(),
      "outerTestReleased": self.outer_test is not None,
      "partition": self.partition.to_dict(),
      "partitionSha256": self.partition.sha256,
      "physicalGenerationSha256": self.physical_generation_sha256,
      "physicalProfileSha256": self.physical_profile_sha256,
      "plantMemberIds": list(self.plant_member_ids),
      "preparationCount": self.preparation_count,
      "productionMode": self.production_mode,
      "refinementSelection": (
        None if self.refinement_selection is None else self.refinement_selection.to_dict()
      ),
      "replaySource": self.replay_source.to_dict(),
      "routeScanCount": self.route_scan_count,
      "schemaVersion": BEHAVIOR_TRAINING_AUTHORITY_SCHEMA_VERSION,
      "robustPlantSetSha256": self.robust_plant_set_sha256,
      "trainingTranscript": dict(self.transcript),
      "validation": None if self.validation is None else self.validation.to_dict(),
      "winner": None if self.winner is None else self.winner.to_dict(),
      "workerCount": self.worker_count,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return _sha256_bytes(self.to_json().encode("utf-8"))


def _execute_epoch_once_common(
  *,
  evidence_store_root: Path,
  import_manifest_sha256: str,
  partition_receipt_path: Path,
  training_physical_receipt_path: Path,
  robust_plant_set_path: Path,
  replay_source: ReviewedReplaySource,
  worker_count: int,
  interface_registry: Mapping[str, type] | None,
  production_mode: bool,
) -> BehaviorTrainingRun:
  if type(production_mode) is not bool:
    raise TypeError("training production mode must be boolean")
  if production_mode and (worker_count != _WORKER_COUNT or interface_registry is not None):
    raise BehaviorTrainingAuthorityError(
      "production training requires four workers and the detected interface",
    )
  try:
    authenticated = _authenticate_manifest(evidence_store_root, import_manifest_sha256)
  except BehaviorReplayAuthorityError as error:
    raise BehaviorTrainingAuthorityError(str(error)) from error
  try:
    gate = load_behavior_gate_spec(BEHAVIOR_GATE_SPEC_PATH)
    segmentation = load_behavior_segmentation_config(BEHAVIOR_SEGMENTATION_CONFIG_PATH)
    dynamics = ProvisionalRackDynamics.from_json_file(PROVISIONAL_RACK_DYNAMICS_PATH)
    controller_seed = ControllerPolicy.from_json_file(PROVISIONAL_CONTROLLER_POLICY_PATH)
    center = BehaviorPolicy.from_controller_policy(controller_seed)
    coarse_grid = build_candidate_grid(gate.candidate_grid.policy_grid(center))
    exact_stock_core = reviewed_replay_core_identity(
      exact_stock=True,
      source_openpilot_commit=replay_source.source_openpilot_commit,
      opendbc_commit=replay_source.opendbc_commit,
      panda_commit=replay_source.panda_commit,
    )
    modular_core = reviewed_replay_core_identity(
      exact_stock=False,
      source_openpilot_commit=replay_source.source_openpilot_commit,
      opendbc_commit=replay_source.opendbc_commit,
      panda_commit=replay_source.panda_commit,
    )
  except (OSError, TypeError, ValueError) as error:
    raise BehaviorTrainingAuthorityError("committed training authority is invalid") from error
  partition_receipt = _load_training_partition_receipt(
    partition_receipt_path,
    authenticated=authenticated,
    expected_seed_identity_sha256=gate.route_partition.seed_identity_sha256,
    replay_source=replay_source,
  )
  partition = partition_receipt.partition
  vehicle_identities = {
    route.source.get("vehicle_identity") for route in authenticated.pending_routes
  }
  if len(vehicle_identities) != 1 or type(next(iter(vehicle_identities))) is not str:
    raise BehaviorTrainingAuthorityError("scenario-eligible routes mix vehicle identities")
  vehicle_identity = next(iter(vehicle_identities))
  try:
    physical_receipt = _load_training_physical_receipt(
      training_physical_receipt_path,
      import_manifest_sha256=authenticated.manifest_sha256,
      partition_receipt=partition_receipt,
    )
    profile, profile_sha256 = _load_physical_profile(
      physical_receipt.profile_path,
      vehicle_identity,
    )
  except BehaviorReplayAuthorityError as error:
    raise BehaviorTrainingAuthorityError(str(error)) from error
  if profile_sha256 != physical_receipt.profile_sha256:
    raise BehaviorTrainingAuthorityError("training physical profile loader identity differs")
  if physical_receipt.partition_receipt_sha256 != partition_receipt.receipt_sha256:
    raise BehaviorTrainingAuthorityError("physical profile uses another partition receipt")
  try:
    plant_set = load_robust_plant_set(
      robust_plant_set_path,
      import_manifest_sha256=authenticated.manifest_sha256,
      partition_receipt_sha256=physical_receipt.partition_receipt_sha256,
      partition=partition,
      physical_generation_sha256=physical_receipt.generation_sha256,
      physical_profile_sha256=profile_sha256,
      physical_module_closure_sha256=physical_receipt.module_closure_sha256,
      replay_source=replay_source,
    )
  except RobustPlantSetError as error:
    raise BehaviorTrainingAuthorityError(str(error)) from error
  plant_members = plant_set.members
  plant_member_ids = plant_set.member_ids
  training_routes = _activate_routes(
    authenticated,
    partition.route_ids(TrainingSplit.TRAINING),
  )

  preparations: dict[str, BehaviorRoutePreparation] = {}
  preparation_count = 0
  controller_replay_count = 0
  route_scan_count = 0
  coarse_report: dict[str, object] | None = None
  refinement_report: dict[str, object] | None = None
  validation_report: dict[str, object] | None = None
  outer_test_report: dict[str, object] | None = None
  configuration_report = {
    "controllerSeedSha256": controller_seed.sha256,
    "gateSpecSha256": gate.sha256,
    "metricConfigSha256": gate.metric_config.sha256,
    "physicalProfileSha256": profile_sha256,
    "provisionalDynamicsSha256": dynamics.identity_sha256,
    "segmentationConfigSha256": segmentation.sha256,
    "physicalGenerationSha256": physical_receipt.generation_sha256,
    "physicalModuleClosureSha256": physical_receipt.module_closure_sha256,
    "plantMemberIds": list(plant_member_ids),
    "robustPlantSetSha256": plant_set.receipt_sha256,
    "transientReportSha256": plant_set.transient_report_sha256,
    "transientRulesSha256": plant_set.transient_rules_sha256,
  }

  def map_batches(
    calls: tuple[tuple[object, ...], ...],
  ) -> tuple[_RouteBatchResult, ...]:
    if production_mode:
      return _ordered_map(calls, replay_source)
    return _ordered_map_for_test(calls, worker_count, interface_registry)

  def finish(
    disposition: TrainingDisposition,
    coarse_selection: RobustTrainingSelection | None,
    refinement_selection: RobustTrainingSelection | None,
    validation: RobustHeldOutValidation | None,
    outer_test: RobustHeldOutValidation | None,
    winner: BehaviorPolicy | None,
  ) -> BehaviorTrainingRun:
    if coarse_report is None:
      raise BehaviorTrainingAuthorityError("coarse training transcript is missing")
    transcript = {
      "coarseSearch": coarse_report,
      "configuration": configuration_report,
      "outerTest": outer_test_report,
      "refinementSearch": refinement_report,
      "schemaVersion": 2,
      "validation": validation_report,
    }
    return BehaviorTrainingRun(
      import_manifest_sha256=authenticated.manifest_sha256,
      replay_source=replay_source,
      physical_profile_sha256=profile_sha256,
      physical_generation_sha256=physical_receipt.generation_sha256,
      robust_plant_set_sha256=plant_set.receipt_sha256,
      plant_member_ids=plant_member_ids,
      partition=partition,
      controller_seed_sha256=controller_seed.sha256,
      coarse_selection=coarse_selection,
      refinement_selection=refinement_selection,
      validation=validation,
      outer_test=outer_test,
      winner=winner,
      disposition=disposition,
      transcript=transcript,
      worker_count=worker_count,
      production_mode=production_mode,
      preparation_count=preparation_count,
      controller_replay_count=controller_replay_count,
      route_scan_count=route_scan_count,
    )

  coarse_policies: tuple[BehaviorPolicy | None, ...] = (
    None,
    *(candidate.policy for candidate in coarse_grid),
  )
  coarse_roles = (ReplayRole.EXACT_STOCK,) + (ReplayRole.CANDIDATE,) * len(coarse_grid)
  coarse_batches = map_batches(
    _calls(
      training_routes,
      preparations,
      profile,
      dynamics,
      segmentation,
      coarse_policies,
      coarse_roles,
      gate,
      exact_stock_core,
      modular_core,
      plant_members,
    ),
  )
  preparations.update({batch.route_id: batch.preparation for batch in coarse_batches})
  preparation_count += len(coarse_batches)
  route_scan_count += len(coarse_batches) * len(plant_members)
  controller_replay_count += len(coarse_batches) * len(coarse_policies) * len(plant_members)
  stock_training = _aggregate_member_stage(
    TrainingSplit.TRAINING, None, 0, coarse_batches, plant_member_ids,
  )
  for evaluation in stock_training.values():
    _validate_split_coverage(evaluation, gate)
  coarse_evaluations = {
    candidate.policy: _aggregate_member_stage(
      TrainingSplit.TRAINING, candidate.policy, index + 1,
      coarse_batches, plant_member_ids,
    )
    for index, candidate in enumerate(coarse_grid)
  }
  coarse_selection = _select(coarse_grid, coarse_evaluations, stock_training, gate)
  coarse_report = _search_stage_report(
    "coarse",
    coarse_grid,
    coarse_evaluations,
    stock_training,
    gate,
    coarse_selection,
  )
  if coarse_selection is None:
    return finish(
      TrainingDisposition.NO_TRAINING_WINNER,
      None, None, None, None, None,
    )

  refinement_grid = _refinement_grid(gate, coarse_selection.winner)
  reused = {coarse_selection.winner.policy: coarse_evaluations[coarse_selection.winner.policy]}
  new_refinement = tuple(
    candidate for candidate in refinement_grid if candidate.policy not in reused
  )
  if new_refinement:
    refinement_batches = map_batches(
      _calls(
        training_routes,
        preparations,
        profile,
        dynamics,
        segmentation,
        tuple(candidate.policy for candidate in new_refinement),
        (ReplayRole.CANDIDATE,) * len(new_refinement),
        gate,
        exact_stock_core,
        modular_core,
        plant_members,
      ),
    )
    controller_replay_count += len(refinement_batches) * len(new_refinement) * len(plant_members)
    route_scan_count += len(refinement_batches) * len(plant_members)
    reused.update({
      candidate.policy: _aggregate_member_stage(
        TrainingSplit.TRAINING, candidate.policy, index,
        refinement_batches, plant_member_ids,
      )
      for index, candidate in enumerate(new_refinement)
    })
  refinement_selection = _select(refinement_grid, reused, stock_training, gate)
  if refinement_selection is None:
    raise BehaviorTrainingAuthorityError("refinement lost its passing coarse center")
  refinement_report = _search_stage_report(
    "refinement",
    refinement_grid,
    reused,
    stock_training,
    gate,
    refinement_selection,
  )
  winner_policy = refinement_selection.winner.policy

  validation_routes = _activate_routes(
    authenticated,
    partition.route_ids(TrainingSplit.VALIDATION),
  )
  if any(route.scenario.vehicle_identity != vehicle_identity for route in validation_routes):
    raise BehaviorTrainingAuthorityError("validation routes use another vehicle identity")
  validation_batches = map_batches(
    _calls(
      validation_routes,
      preparations,
      profile,
      dynamics,
      segmentation,
      (None, winner_policy),
      (ReplayRole.EXACT_STOCK, ReplayRole.CANDIDATE),
      gate,
      exact_stock_core,
      modular_core,
      plant_members,
    ),
  )
  preparations.update({batch.route_id: batch.preparation for batch in validation_batches})
  preparation_count += len(validation_batches)
  route_scan_count += len(validation_batches) * len(plant_members)
  controller_replay_count += 2 * len(validation_batches) * len(plant_members)
  stock_validation = _aggregate_member_stage(
    TrainingSplit.VALIDATION, None, 0, validation_batches, plant_member_ids,
  )
  for evaluation in stock_validation.values():
    _validate_split_coverage(evaluation, gate)
  winner_validation = _aggregate_member_stage(
    TrainingSplit.VALIDATION, winner_policy, 1, validation_batches, plant_member_ids,
  )
  validation = _validate(refinement_selection, winner_validation, stock_validation, gate)
  validation_report = _held_out_stage_report(
    stock_validation,
    winner_validation,
    validation,
  )
  if not validation.accepted:
    return finish(
      TrainingDisposition.VALIDATION_REJECTED,
      coarse_selection, refinement_selection, validation, None, winner_policy,
    )

  test_routes = _activate_routes(
    authenticated,
    partition.route_ids(TrainingSplit.TEST),
  )
  if any(route.scenario.vehicle_identity != vehicle_identity for route in test_routes):
    raise BehaviorTrainingAuthorityError("outer-test routes use another vehicle identity")
  test_batches = map_batches(
    _calls(
      test_routes,
      preparations,
      profile,
      dynamics,
      segmentation,
      (None, winner_policy),
      (ReplayRole.EXACT_STOCK, ReplayRole.CANDIDATE),
      gate,
      exact_stock_core,
      modular_core,
      plant_members,
    ),
  )
  preparation_count += len(test_batches)
  route_scan_count += len(test_batches) * len(plant_members)
  controller_replay_count += 2 * len(test_batches) * len(plant_members)
  stock_test = _aggregate_member_stage(
    TrainingSplit.TEST, None, 0, test_batches, plant_member_ids,
  )
  for evaluation in stock_test.values():
    _validate_split_coverage(evaluation, gate)
  winner_test = _aggregate_member_stage(
    TrainingSplit.TEST, winner_policy, 1, test_batches, plant_member_ids,
  )
  outer_test = _validate(refinement_selection, winner_test, stock_test, gate)
  outer_test_report = _held_out_stage_report(stock_test, winner_test, outer_test)
  disposition = (
    TrainingDisposition.EXTERNAL_REVIEW_CANDIDATE
    if outer_test.accepted
    else TrainingDisposition.TEST_REJECTED
  )
  return finish(
    disposition,
    coarse_selection,
    refinement_selection,
    validation,
    outer_test,
    winner_policy,
  )


def _execute_epoch_once(
  *,
  evidence_store_root: Path,
  import_manifest_sha256: str,
  partition_receipt_path: Path,
  training_physical_receipt_path: Path,
  robust_plant_set_path: Path,
  replay_source: ReviewedReplaySource,
) -> BehaviorTrainingRun:
  """Execute one fixed four-process production epoch."""
  return _execute_epoch_once_common(
    evidence_store_root=evidence_store_root,
    import_manifest_sha256=import_manifest_sha256,
    partition_receipt_path=partition_receipt_path,
    training_physical_receipt_path=training_physical_receipt_path,
    robust_plant_set_path=robust_plant_set_path,
    replay_source=replay_source,
    worker_count=_WORKER_COUNT,
    interface_registry=None,
    production_mode=True,
  )


def _execute_epoch_once_for_test(
  *,
  evidence_store_root: Path,
  import_manifest_sha256: str,
  partition_receipt_path: Path,
  training_physical_receipt_path: Path,
  robust_plant_set_path: Path,
  replay_source: ReviewedReplaySource,
  worker_count: int = 1,
  interface_registry: Mapping[str, type] | None = None,
) -> BehaviorTrainingRun:
  """Execute an injectable epoch that is structurally non-authoritative."""
  return _execute_epoch_once_common(
    evidence_store_root=evidence_store_root,
    import_manifest_sha256=import_manifest_sha256,
    partition_receipt_path=partition_receipt_path,
    training_physical_receipt_path=training_physical_receipt_path,
    robust_plant_set_path=robust_plant_set_path,
    replay_source=replay_source,
    worker_count=worker_count,
    interface_registry=interface_registry,
    production_mode=False,
  )


class AuthenticatedBehaviorTrainingReceipt:
  """Opaque proof of two complete, bit-identical training executions."""

  __slots__ = ("_first", "_independent_aa_sha256")

  def __init__(
    self,
    *_: object,
    **__: object,
  ) -> None:
    raise TypeError("training receipts are issued only by the production authority")

  @property
  def result(self) -> BehaviorTrainingRun:
    return self._first

  @property
  def independent_aa_sha256(self) -> str:
    return self._independent_aa_sha256

  def to_dict(self) -> dict[str, object]:
    if (
      self.result.disposition is not TrainingDisposition.EXTERNAL_REVIEW_CANDIDATE
      or not self.result.production_mode
      or self.result.worker_count != _WORKER_COUNT
      or self.independent_aa_sha256 != self.result.sha256
    ):
      raise BehaviorTrainingAuthorityError("a rejected epoch cannot serialize as a receipt")
    before = _verify_replay_source(self.result.replay_source)
    payload = {
      "activationEligible": False,
      "bitExactIndependentAA": True,
      "independentAASha256": self.independent_aa_sha256,
      "result": self.result.to_dict(),
      "schemaVersion": BEHAVIOR_TRAINING_AUTHORITY_SCHEMA_VERSION,
    }
    after = _verify_replay_source(self.result.replay_source)
    if before != after:
      raise BehaviorTrainingAuthorityError("reviewed source changed during receipt serialization")
    return payload

  @property
  def canonical_sha256(self) -> str:
    return _sha256_bytes(_RECEIPT_DOMAIN + canonical_json(self.to_dict()).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class ValidatedRobustSelectionEvidence:
  sha256: str
  winner: BehaviorPolicy
  winner_canonical_index: int
  plant_member_ids: tuple[str, ...]
  training_route_ids: tuple[str, ...]
  passed_contracts: frozenset[BehaviorContract]
  payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ValidatedHeldOutBehaviorEvidence:
  sha256: str
  selection_sha256: str
  route_ids: tuple[str, ...]
  plant_member_ids: tuple[str, ...]
  passed_contracts: frozenset[BehaviorContract]
  payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ValidatedBehaviorTrainingRun:
  sha256: str
  import_manifest_sha256: str
  controller_seed_sha256: str
  partition: FrozenBehaviorPartition
  replay_source: ReviewedReplaySource
  physical_generation_sha256: str
  physical_profile_sha256: str
  robust_plant_set_sha256: str
  plant_member_ids: tuple[str, ...]
  selection: ValidatedRobustSelectionEvidence
  validation: ValidatedHeldOutBehaviorEvidence
  outer_test: ValidatedHeldOutBehaviorEvidence
  winner: BehaviorPolicy
  payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ValidatedBehaviorTrainingEvidence:
  """Read-only authenticated DTO consumed by later packaging authorities."""

  receipt_sha256: str
  independent_aa_sha256: str
  result: ValidatedBehaviorTrainingRun
  canonical_bytes: bytes
  payload: Mapping[str, object]

  @property
  def canonical_sha256(self) -> str:
    return self.receipt_sha256

  def behavior_contract_passed(self, contract: BehaviorContract) -> bool:
    if type(contract) is not BehaviorContract:
      raise TypeError("contract must use BehaviorContract")
    return all(
      contract in evidence.passed_contracts
      for evidence in (self.result.selection, self.result.validation, self.result.outer_test)
    )


def _freeze_receipt_value(value: object) -> object:
  if type(value) is dict:
    return MappingProxyType({key: _freeze_receipt_value(item) for key, item in value.items()})
  if type(value) is list:
    return tuple(_freeze_receipt_value(item) for item in value)
  return value


def _receipt_number(value: object, description: str) -> float:
  if type(value) not in (int, float):
    raise BehaviorTrainingAuthorityError(f"{description} is not numeric")
  result = float(value)
  if not math.isfinite(result):
    raise BehaviorTrainingAuthorityError(f"{description} is not finite")
  return result


def _receipt_policy(value: object, description: str) -> BehaviorPolicy:
  if type(value) is not dict or set(value) != {"dampingRatio", "naturalFrequencyPerS"}:
    raise BehaviorTrainingAuthorityError(f"{description} is malformed")
  try:
    return BehaviorPolicy(
      _receipt_number(value["naturalFrequencyPerS"], f"{description} frequency"),
      _receipt_number(value["dampingRatio"], f"{description} damping"),
    )
  except ValueError as error:
    raise BehaviorTrainingAuthorityError(f"{description} is outside its domain") from error


def _receipt_candidate(value: object, description: str) -> tuple[int, BehaviorPolicy, float]:
  if type(value) is not dict or set(value) != {
    "canonicalIndex", "policy", "squaredLogDisplacement",
  }:
    raise BehaviorTrainingAuthorityError(f"{description} is malformed")
  index = value["canonicalIndex"]
  if type(index) is not int or index < 0:
    raise BehaviorTrainingAuthorityError(f"{description} index is malformed")
  displacement = _receipt_number(value["squaredLogDisplacement"], f"{description} displacement")
  if displacement < 0.0:
    raise BehaviorTrainingAuthorityError(f"{description} displacement is negative")
  return index, _receipt_policy(value["policy"], f"{description} policy"), displacement


def _validate_paired_uncertainty(value: object, description: str) -> None:
  if value is None:
    return
  if type(value) is not dict or set(value) != {
    "lower", "mean", "routeCount", "routeIds", "tolerance", "uncertainty", "upper",
  }:
    raise BehaviorTrainingAuthorityError(f"{description} is malformed")
  route_ids = value["routeIds"]
  if (
    type(route_ids) is not list
    or route_ids != sorted(set(route_ids))
    or value["routeCount"] != len(route_ids)
    or any(type(route_id) is not str or _ROUTE_ID_RE.fullmatch(route_id) is None for route_id in route_ids)
  ):
    raise BehaviorTrainingAuthorityError(f"{description} route population is malformed")
  values = tuple(
    _receipt_number(value[name], f"{description} {name}")
    for name in ("lower", "mean", "tolerance", "uncertainty", "upper")
  )
  lower, mean, tolerance, uncertainty, upper = values
  if tolerance < 0.0 or uncertainty < 0.0 or lower > mean or mean > upper:
    raise BehaviorTrainingAuthorityError(f"{description} bounds are inconsistent")


def _validate_member_gate_verdict(
  value: object,
  *,
  expected_candidate: object,
  require_pass: bool,
) -> frozenset[BehaviorContract]:
  keys = {
    "acceptedEvaluationSha256", "candidate", "candidateEvaluationSha256", "contracts",
    "exactStockEvaluationSha256", "gateSpecSha256", "passed", "routeIds",
    "targetImprovement", "targetMateriallyImproved", "targetMetricName",
    "targetNoiseFloor", "worstContractMargin",
  }
  if type(value) is not dict or set(value) != keys or value["candidate"] != expected_candidate:
    raise BehaviorTrainingAuthorityError("plant-member gate verdict is malformed")
  for name in (
    "acceptedEvaluationSha256", "candidateEvaluationSha256",
    "exactStockEvaluationSha256", "gateSpecSha256",
  ):
    try:
      _sha256(value[name], f"plant-member verdict {name}")
    except BehaviorReplayAuthorityError as error:
      raise BehaviorTrainingAuthorityError(str(error)) from error
  routes = value["routeIds"]
  if (
    type(routes) is not list
    or not routes
    or routes != sorted(set(routes))
    or any(type(route_id) is not str or _ROUTE_ID_RE.fullmatch(route_id) is None for route_id in routes)
  ):
    raise BehaviorTrainingAuthorityError("plant-member verdict routes are malformed")
  contracts = value["contracts"]
  if type(contracts) is not list or len(contracts) != len(BehaviorContract):
    raise BehaviorTrainingAuthorityError("plant-member verdict contracts are incomplete")
  passed_contracts: set[BehaviorContract] = set()
  for contract in contracts:
    if type(contract) is not dict or set(contract) != {"contract", "margin", "metrics", "passed"}:
      raise BehaviorTrainingAuthorityError("plant-member contract verdict is malformed")
    try:
      contract_name = BehaviorContract(contract["contract"])
    except (TypeError, ValueError) as error:
      raise BehaviorTrainingAuthorityError("plant-member contract is unknown") from error
    if contract_name in passed_contracts:
      raise BehaviorTrainingAuthorityError("plant-member contract is duplicated")
    metrics = contract["metrics"]
    if type(metrics) is not list or not metrics:
      raise BehaviorTrainingAuthorityError("plant-member contract metrics are empty")
    metric_passes = []
    margins: list[float] = []
    for metric in metrics:
      if type(metric) is not dict or set(metric) != {
        "contract", "margin", "metricName", "pairedAgainstAccepted",
        "pairedAgainstStock", "passed", "reasons",
      } or metric["contract"] != contract_name.value:
        raise BehaviorTrainingAuthorityError("plant-member metric verdict is malformed")
      try:
        BehaviorMetricName(metric["metricName"])
      except (TypeError, ValueError) as error:
        raise BehaviorTrainingAuthorityError("plant-member metric is unknown") from error
      if type(metric["passed"]) is not bool or type(metric["reasons"]) is not list:
        raise BehaviorTrainingAuthorityError("plant-member metric disposition is malformed")
      if any(type(reason) is not str or not reason for reason in metric["reasons"]):
        raise BehaviorTrainingAuthorityError("plant-member metric reasons are malformed")
      _validate_paired_uncertainty(metric["pairedAgainstAccepted"], "accepted paired uncertainty")
      _validate_paired_uncertainty(metric["pairedAgainstStock"], "stock paired uncertainty")
      if metric["margin"] is not None:
        margins.append(_receipt_number(metric["margin"], "plant-member metric margin"))
      metric_passes.append(metric["passed"])
    if type(contract["passed"]) is not bool or contract["passed"] is not all(metric_passes):
      raise BehaviorTrainingAuthorityError("plant-member contract disposition disagrees")
    if contract["margin"] is not None:
      contract_margin = _receipt_number(contract["margin"], "plant-member contract margin")
      if margins and contract_margin != min(margins):
        raise BehaviorTrainingAuthorityError("plant-member contract margin disagrees")
    if contract["passed"]:
      passed_contracts.add(contract_name)
  if type(value["targetMetricName"]) is not str:
    raise BehaviorTrainingAuthorityError("plant-member target metric is malformed")
  try:
    BehaviorMetricName(value["targetMetricName"])
  except ValueError as error:
    raise BehaviorTrainingAuthorityError("plant-member target metric is unknown") from error
  if type(value["targetMateriallyImproved"]) is not bool:
    raise BehaviorTrainingAuthorityError("plant-member target disposition is malformed")
  expected_pass = all(
    contract in passed_contracts for contract in BehaviorContract
  ) and value["targetMateriallyImproved"]
  if type(value["passed"]) is not bool or value["passed"] is not expected_pass:
    raise BehaviorTrainingAuthorityError("plant-member gate disposition disagrees")
  if require_pass and (
    value["passed"] is not True
    or value["targetImprovement"] is None
    or value["worstContractMargin"] is None
  ):
    raise BehaviorTrainingAuthorityError("plant-member gate is not passing")
  _receipt_number(value["targetNoiseFloor"], "plant-member target noise floor")
  if value["targetImprovement"] is not None:
    _receipt_number(value["targetImprovement"], "plant-member target improvement")
  if value["worstContractMargin"] is not None:
    _receipt_number(value["worstContractMargin"], "plant-member worst margin")
  return frozenset(passed_contracts)


def _validate_robust_verdict(
  value: object,
  *,
  expected_member_ids: tuple[str, ...],
  expected_candidate: object | None = None,
  require_pass: bool,
) -> tuple[frozenset[BehaviorContract], object]:
  if type(value) is not dict or set(value) != {
    "candidate", "failingMemberIds", "memberVerdicts", "passed", "worstContractMargin",
    "worstMemberId", "worstTargetImprovement",
  }:
    raise BehaviorTrainingAuthorityError("robust candidate verdict is malformed")
  _receipt_candidate(value["candidate"], "robust candidate")
  if expected_candidate is not None and value["candidate"] != expected_candidate:
    raise BehaviorTrainingAuthorityError("robust verdict candidate differs")
  rows = value["memberVerdicts"]
  if type(rows) is not list:
    raise BehaviorTrainingAuthorityError("robust member verdicts are malformed")
  member_ids: list[str] = []
  passed_contracts: set[BehaviorContract] | None = None
  failing: list[str] = []
  scored: list[tuple[str, float, float]] = []
  for row in rows:
    if type(row) is not dict or set(row) != {"memberId", "verdict"}:
      raise BehaviorTrainingAuthorityError("robust member verdict row is malformed")
    try:
      member_id = _sha256(row["memberId"], "robust plant member")
    except BehaviorReplayAuthorityError as error:
      raise BehaviorTrainingAuthorityError(str(error)) from error
    member_ids.append(member_id)
    contracts = _validate_member_gate_verdict(
      row["verdict"],
      expected_candidate=value["candidate"],
      require_pass=require_pass,
    )
    if passed_contracts is None:
      passed_contracts = set(contracts)
    else:
      passed_contracts &= set(contracts)
    if row["verdict"]["passed"] is not True:
      failing.append(member_id)
    target = row["verdict"]["targetImprovement"]
    margin = row["verdict"]["worstContractMargin"]
    if target is not None and margin is not None:
      scored.append((member_id, float(target), float(margin)))
  if tuple(member_ids) != expected_member_ids:
    raise BehaviorTrainingAuthorityError("robust plant-member population changed")
  if (
    value["failingMemberIds"] != failing
    or type(value["passed"]) is not bool
    or value["passed"] is not (not failing)
    or value["worstMemberId"] not in expected_member_ids
  ):
    raise BehaviorTrainingAuthorityError("robust verdict disposition disagrees")
  if scored:
    worst_member = min(scored, key=lambda row: (row[1], row[2], row[0]))[0]
    if (
      value["worstMemberId"] != worst_member
      or _receipt_number(value["worstTargetImprovement"], "robust worst improvement")
      != min(row[1] for row in scored)
      or _receipt_number(value["worstContractMargin"], "robust worst margin")
      != min(row[2] for row in scored)
    ):
      raise BehaviorTrainingAuthorityError("robust worst-member accounting differs")
  elif require_pass:
    raise BehaviorTrainingAuthorityError("passing robust verdict has no scored members")
  return frozenset(passed_contracts or ()), value["candidate"]


def _validate_selection_receipt(
  value: object,
  *,
  expected_member_ids: tuple[str, ...],
  expected_route_ids: tuple[str, ...],
) -> ValidatedRobustSelectionEvidence:
  if type(value) is not dict or set(value) != {
    "allVerdicts", "candidateGridSha256", "plantMemberIds", "trainingRouteIds",
    "winnerCanonicalIndex", "winnerPolicy", "winnerVerdict",
  }:
    raise BehaviorTrainingAuthorityError("robust selection is malformed")
  if value["plantMemberIds"] != list(expected_member_ids) or value["trainingRouteIds"] != list(expected_route_ids):
    raise BehaviorTrainingAuthorityError("robust selection population differs")
  all_verdicts = value["allVerdicts"]
  if type(all_verdicts) is not list or not all_verdicts:
    raise BehaviorTrainingAuthorityError("robust selection transcript is empty")
  candidates = []
  winner_match = None
  for verdict in all_verdicts:
    _, candidate = _validate_robust_verdict(
      verdict,
      expected_member_ids=expected_member_ids,
      require_pass=False,
    )
    candidates.append(candidate)
    if verdict == value["winnerVerdict"]:
      winner_match = verdict
  if winner_match is None:
    raise BehaviorTrainingAuthorityError("robust selection winner is absent")
  candidate_grid_sha256 = _sha256_json(candidates)
  if value["candidateGridSha256"] != candidate_grid_sha256:
    raise BehaviorTrainingAuthorityError("robust candidate-grid identity differs")
  contracts, winner_candidate = _validate_robust_verdict(
    value["winnerVerdict"],
    expected_member_ids=expected_member_ids,
    require_pass=True,
  )
  winner_index, winner, _ = _receipt_candidate(winner_candidate, "robust winner")
  if value["winnerCanonicalIndex"] != winner_index or value["winnerPolicy"] != winner.to_dict():
    raise BehaviorTrainingAuthorityError("robust winner identity differs")
  frozen = _freeze_receipt_value(value)
  assert isinstance(frozen, Mapping)
  return ValidatedRobustSelectionEvidence(
    _sha256_json(value),
    winner,
    winner_index,
    expected_member_ids,
    expected_route_ids,
    contracts,
    frozen,
  )


def _validate_held_out_receipt(
  value: object,
  *,
  selection: ValidatedRobustSelectionEvidence,
  expected_route_ids: tuple[str, ...],
) -> ValidatedHeldOutBehaviorEvidence:
  if type(value) is not dict or set(value) != {
    "accepted", "frozenWinnerVerdict", "plantMemberIds", "selectionSha256",
    "validationRouteIds",
  }:
    raise BehaviorTrainingAuthorityError("held-out robust evidence is malformed")
  if (
    value["accepted"] is not True
    or value["selectionSha256"] != selection.sha256
    or value["plantMemberIds"] != list(selection.plant_member_ids)
    or value["validationRouteIds"] != list(expected_route_ids)
  ):
    raise BehaviorTrainingAuthorityError("held-out robust evidence belongs to another selection")
  expected_candidate = {
    "canonicalIndex": selection.winner_canonical_index,
    "policy": selection.winner.to_dict(),
    "squaredLogDisplacement": selection.payload["winnerVerdict"]["candidate"]["squaredLogDisplacement"],
  }
  contracts, _ = _validate_robust_verdict(
    value["frozenWinnerVerdict"],
    expected_member_ids=selection.plant_member_ids,
    expected_candidate=expected_candidate,
    require_pass=True,
  )
  frozen = _freeze_receipt_value(value)
  assert isinstance(frozen, Mapping)
  return ValidatedHeldOutBehaviorEvidence(
    _sha256_json(value),
    selection.sha256,
    expected_route_ids,
    selection.plant_member_ids,
    contracts,
    frozen,
  )


def _validate_stage_summary_rows(
  rows: object,
  *,
  expected_member_ids: tuple[str, ...],
  expected_route_ids: tuple[str, ...],
  expected_split: str,
  expected_policy_sha256: str | None,
) -> None:
  if type(rows) is not list or len(rows) != len(expected_member_ids):
    raise BehaviorTrainingAuthorityError("stage member summaries are incomplete")
  observed = []
  for row in rows:
    if type(row) is not dict or set(row) != {
      "evaluationSha256", "plantMemberId", "policySha256", "routeEvaluationSha256s",
      "routeIds", "selectorSha256", "split",
    }:
      raise BehaviorTrainingAuthorityError("stage member summary is malformed")
    try:
      member_id = _sha256(row["plantMemberId"], "stage plant member")
      _sha256(row["evaluationSha256"], "stage evaluation")
      _sha256(row["selectorSha256"], "stage selector")
      for sha256 in row["routeEvaluationSha256s"]:
        _sha256(sha256, "stage route evaluation")
    except (BehaviorReplayAuthorityError, TypeError) as error:
      raise BehaviorTrainingAuthorityError("stage evidence identity is malformed") from error
    if (
      type(row["routeEvaluationSha256s"]) is not list
      or len(row["routeEvaluationSha256s"]) != len(expected_route_ids)
      or row["routeIds"] != list(expected_route_ids)
      or row["split"] != expected_split
      or row["policySha256"] != expected_policy_sha256
    ):
      raise BehaviorTrainingAuthorityError("stage member summary belongs to another stage")
    observed.append(member_id)
  if tuple(observed) != expected_member_ids:
    raise BehaviorTrainingAuthorityError("stage plant-member set changed")


def _parse_reviewed_replay_source(value: object) -> ReviewedReplaySource:
  if type(value) is not dict or set(value) != {
    "moduleClosureSha256", "opendbcCommit", "pandaCommit", "runtimeIdentitySha256",
    "schemaVersion", "sourceCompositionSha256", "sourceOpenpilotCommit",
  }:
    raise BehaviorTrainingAuthorityError("training receipt replay source is malformed")
  try:
    return ReviewedReplaySource(
      source_openpilot_commit=value["sourceOpenpilotCommit"],
      opendbc_commit=value["opendbcCommit"],
      panda_commit=value["pandaCommit"],
      source_composition_sha256=value["sourceCompositionSha256"],
      runtime_identity_sha256=value["runtimeIdentitySha256"],
      module_closure_sha256=value["moduleClosureSha256"],
      schema_version=value["schemaVersion"],
    )
  except (TypeError, ValueError) as error:
    raise BehaviorTrainingAuthorityError("training receipt replay source is invalid") from error


def _load_authenticated_behavior_training_receipt_common(
  path: Path,
  *,
  verify_replay_source: bool,
) -> ValidatedBehaviorTrainingEvidence:
  if not isinstance(path, Path) or not path.is_absolute() or path.name != "receipt.json":
    raise BehaviorTrainingAuthorityError("authenticated training receipt path is malformed")
  try:
    _sha256(path.parent.name, "authenticated training receipt directory")
    if (
      path.parent.resolve(strict=True) != path.parent
      or path.parent.is_symlink()
      or {entry.name for entry in path.parent.iterdir()} != {"receipt.json"}
    ):
      raise BehaviorTrainingAuthorityError("authenticated training receipt directory is unsafe")
    encoded, _ = _read_immutable_regular(
      path,
      "authenticated training receipt",
      _TRAINING_RECEIPT_MAXIMUM_BYTES,
    )
    payload = json.loads(encoded)
  except BehaviorTrainingAuthorityError:
    raise
  except BehaviorReplayAuthorityError as error:
    raise BehaviorTrainingAuthorityError(str(error)) from error
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise BehaviorTrainingAuthorityError("authenticated training receipt is malformed") from error
  if (
    type(payload) is not dict
    or set(payload) != {
      "activationEligible", "bitExactIndependentAA", "independentAASha256", "result",
      "schemaVersion",
    }
    or encoded != canonical_json(payload).encode()
    or payload["schemaVersion"] != BEHAVIOR_TRAINING_AUTHORITY_SCHEMA_VERSION
    or payload["activationEligible"] is not False
    or payload["bitExactIndependentAA"] is not True
  ):
    raise BehaviorTrainingAuthorityError("authenticated training receipt is not canonical authority")
  receipt_sha256 = _sha256_bytes(_RECEIPT_DOMAIN + encoded)
  if path.parent.name != receipt_sha256:
    raise BehaviorTrainingAuthorityError("authenticated training receipt address differs")
  try:
    independent_aa_sha256 = _sha256(payload["independentAASha256"], "independent training A/A")
  except BehaviorReplayAuthorityError as error:
    raise BehaviorTrainingAuthorityError(str(error)) from error
  result = payload["result"]
  result_keys = {
    "acceptedIncumbentMode", "activationEligible", "controllerReplayCount",
    "controllerSeedSha256", "coarseSelection", "disposition", "importManifestSha256",
    "outerTest", "outerTestReleased", "partition", "partitionSha256",
    "physicalGenerationSha256", "physicalProfileSha256", "plantMemberIds",
    "preparationCount", "productionMode", "refinementSelection", "replaySource",
    "robustPlantSetSha256", "routeScanCount", "schemaVersion", "trainingTranscript",
    "validation", "winner", "workerCount",
  }
  if type(result) is not dict or set(result) != result_keys:
    raise BehaviorTrainingAuthorityError("authenticated training result schema is malformed")
  result_sha256 = _sha256_json(result)
  if result_sha256 != independent_aa_sha256:
    raise BehaviorTrainingAuthorityError("authenticated training A/A identity differs")
  if (
    result["schemaVersion"] != BEHAVIOR_TRAINING_AUTHORITY_SCHEMA_VERSION
    or result["acceptedIncumbentMode"] != "bootstrap_exact_stock_only"
    or result["activationEligible"] is not False
    or result["productionMode"] is not True
    or result["workerCount"] != _WORKER_COUNT
    or result["disposition"] != TrainingDisposition.EXTERNAL_REVIEW_CANDIDATE.value
    or result["outerTestReleased"] is not True
  ):
    raise BehaviorTrainingAuthorityError("authenticated training result is not reviewable")
  for name in (
    "controllerSeedSha256", "importManifestSha256", "partitionSha256",
    "physicalGenerationSha256", "physicalProfileSha256", "robustPlantSetSha256",
  ):
    try:
      _sha256(result[name], f"authenticated training {name}")
    except BehaviorReplayAuthorityError as error:
      raise BehaviorTrainingAuthorityError(str(error)) from error
  for name in ("controllerReplayCount", "preparationCount", "routeScanCount"):
    if type(result[name]) is not int or result[name] < 0:
      raise BehaviorTrainingAuthorityError("authenticated training accounting is malformed")
  if result["routeScanCount"] < result["preparationCount"]:
    raise BehaviorTrainingAuthorityError("authenticated training accounting disagrees")
  try:
    partition = frozen_behavior_partition_from_dict(result["partition"])
  except BehaviorPartitionError as error:
    raise BehaviorTrainingAuthorityError("authenticated training partition is malformed") from error
  if result["partitionSha256"] != partition.sha256:
    raise BehaviorTrainingAuthorityError("authenticated training partition identity differs")
  replay_source = _parse_reviewed_replay_source(result["replaySource"])
  if verify_replay_source:
    try:
      _verify_replay_source(replay_source)
    except BehaviorReplayAuthorityError as error:
      raise BehaviorTrainingAuthorityError(str(error)) from error
  raw_member_ids = result["plantMemberIds"]
  if type(raw_member_ids) is not list:
    raise BehaviorTrainingAuthorityError("authenticated training plant members are malformed")
  try:
    plant_member_ids = tuple(_sha256(value, "authenticated plant member") for value in raw_member_ids)
  except BehaviorReplayAuthorityError as error:
    raise BehaviorTrainingAuthorityError(str(error)) from error
  if not 2 <= len(plant_member_ids) <= 64 or plant_member_ids != tuple(sorted(set(plant_member_ids))):
    raise BehaviorTrainingAuthorityError("authenticated training plant-member set is not robust")
  training_routes = partition.route_ids(TrainingSplit.TRAINING)
  validation_routes = partition.route_ids(TrainingSplit.VALIDATION)
  test_routes = partition.route_ids(TrainingSplit.TEST)
  coarse = _validate_selection_receipt(
    result["coarseSelection"],
    expected_member_ids=plant_member_ids,
    expected_route_ids=training_routes,
  )
  selection = _validate_selection_receipt(
    result["refinementSelection"],
    expected_member_ids=plant_member_ids,
    expected_route_ids=training_routes,
  )
  winner = _receipt_policy(result["winner"], "authenticated winner")
  if winner != selection.winner:
    raise BehaviorTrainingAuthorityError("authenticated winner differs from frozen selection")
  validation = _validate_held_out_receipt(
    result["validation"],
    selection=selection,
    expected_route_ids=validation_routes,
  )
  outer_test = _validate_held_out_receipt(
    result["outerTest"],
    selection=selection,
    expected_route_ids=test_routes,
  )
  transcript = result["trainingTranscript"]
  if type(transcript) is not dict or set(transcript) != {
    "coarseSearch", "configuration", "outerTest", "refinementSearch",
    "schemaVersion", "validation",
  } or transcript["schemaVersion"] != 2:
    raise BehaviorTrainingAuthorityError("authenticated training transcript is malformed")
  for name, expected_name, expected_selection in (
    ("coarseSearch", "coarse", result["coarseSelection"]),
    ("refinementSearch", "refinement", result["refinementSelection"]),
  ):
    stage = transcript[name]
    if type(stage) is not dict or set(stage) != {
      "candidateGridSha256", "candidates", "failureReasons", "name", "selection",
      "stockMemberEvaluations",
    } or stage["name"] != expected_name or stage["selection"] != expected_selection:
      raise BehaviorTrainingAuthorityError("authenticated search transcript differs from selection")
    selected = coarse if name == "coarseSearch" else selection
    if stage["candidateGridSha256"] != selected.payload["candidateGridSha256"]:
      raise BehaviorTrainingAuthorityError("authenticated search grid identity differs")
    _validate_stage_summary_rows(
      stage["stockMemberEvaluations"],
      expected_member_ids=plant_member_ids,
      expected_route_ids=training_routes,
      expected_split=TrainingSplit.TRAINING.value,
      expected_policy_sha256=None,
    )
    candidates = stage["candidates"]
    if type(candidates) is not list or len(candidates) != len(selected.payload["allVerdicts"]):
      raise BehaviorTrainingAuthorityError("authenticated search candidate transcript differs")
    for candidate_row, verdict in zip(candidates, selected.payload["allVerdicts"], strict=True):
      if type(candidate_row) is not dict or set(candidate_row) != {
        "candidate", "memberEvaluations", "verdict",
      } or (
        _freeze_receipt_value(candidate_row["candidate"]) != verdict["candidate"]
        or _freeze_receipt_value(candidate_row["verdict"]) != verdict
      ):
        raise BehaviorTrainingAuthorityError("authenticated search candidate row differs")
      _, policy, _ = _receipt_candidate(candidate_row["candidate"], "search candidate")
      _validate_stage_summary_rows(
        candidate_row["memberEvaluations"],
        expected_member_ids=plant_member_ids,
        expected_route_ids=training_routes,
        expected_split=TrainingSplit.TRAINING.value,
        expected_policy_sha256=policy.sha256,
      )
  for name, expected, route_ids, split in (
    ("validation", result["validation"], validation_routes, TrainingSplit.VALIDATION.value),
    ("outerTest", result["outerTest"], test_routes, TrainingSplit.TEST.value),
  ):
    stage = transcript[name]
    if type(stage) is not dict or set(stage) != {
      "candidateMemberEvaluations", "stockMemberEvaluations", "validation",
    } or stage["validation"] != expected:
      raise BehaviorTrainingAuthorityError("authenticated held-out transcript differs")
    _validate_stage_summary_rows(
      stage["stockMemberEvaluations"],
      expected_member_ids=plant_member_ids,
      expected_route_ids=route_ids,
      expected_split=split,
      expected_policy_sha256=None,
    )
    _validate_stage_summary_rows(
      stage["candidateMemberEvaluations"],
      expected_member_ids=plant_member_ids,
      expected_route_ids=route_ids,
      expected_split=split,
      expected_policy_sha256=winner.sha256,
    )
  configuration = transcript["configuration"]
  if type(configuration) is not dict or set(configuration) != {
    "controllerSeedSha256", "gateSpecSha256", "metricConfigSha256",
    "physicalGenerationSha256", "physicalModuleClosureSha256", "physicalProfileSha256",
    "plantMemberIds", "provisionalDynamicsSha256", "robustPlantSetSha256",
    "segmentationConfigSha256", "transientReportSha256", "transientRulesSha256",
  }:
    raise BehaviorTrainingAuthorityError("authenticated training configuration is malformed")
  if (
    configuration["controllerSeedSha256"] != result["controllerSeedSha256"]
    or configuration["physicalGenerationSha256"] != result["physicalGenerationSha256"]
    or configuration["physicalProfileSha256"] != result["physicalProfileSha256"]
    or configuration["plantMemberIds"] != list(plant_member_ids)
    or configuration["robustPlantSetSha256"] != result["robustPlantSetSha256"]
  ):
    raise BehaviorTrainingAuthorityError("authenticated training configuration bindings differ")
  for name, value in configuration.items():
    if name != "plantMemberIds":
      try:
        _sha256(value, f"authenticated configuration {name}")
      except BehaviorReplayAuthorityError as error:
        raise BehaviorTrainingAuthorityError(str(error)) from error
  frozen_result = _freeze_receipt_value(result)
  frozen_payload = _freeze_receipt_value(payload)
  assert isinstance(frozen_result, Mapping) and isinstance(frozen_payload, Mapping)
  validated_result = ValidatedBehaviorTrainingRun(
    result_sha256,
    result["importManifestSha256"],
    result["controllerSeedSha256"],
    partition,
    replay_source,
    result["physicalGenerationSha256"],
    result["physicalProfileSha256"],
    result["robustPlantSetSha256"],
    plant_member_ids,
    selection,
    validation,
    outer_test,
    winner,
    frozen_result,
  )
  evidence = ValidatedBehaviorTrainingEvidence(
    receipt_sha256,
    independent_aa_sha256,
    validated_result,
    encoded,
    frozen_payload,
  )
  if not all(evidence.behavior_contract_passed(contract) for contract in BehaviorContract):
    raise BehaviorTrainingAuthorityError("authenticated training did not pass every behavior contract")
  return evidence


def load_authenticated_behavior_training_receipt(
  path: Path,
) -> ValidatedBehaviorTrainingEvidence:
  """Load one immutable schema-v2 training proof and verify current replay bytes."""
  return _load_authenticated_behavior_training_receipt_common(path, verify_replay_source=True)


def publish_authenticated_behavior_training_receipt(
  receipt: AuthenticatedBehaviorTrainingReceipt,
  *,
  root: Path,
) -> Path:
  """Persist one production-issued receipt under its semantic content address."""
  if type(receipt) is not AuthenticatedBehaviorTrainingReceipt:
    raise TypeError("receipt must be an authenticated production receipt")
  if not isinstance(root, Path) or not root.is_absolute():
    raise TypeError("authenticated training receipt root must be an absolute Path")
  root.mkdir(parents=True, exist_ok=True)
  if root.resolve(strict=True) != root or root.is_symlink() or not root.is_dir():
    raise BehaviorTrainingAuthorityError("authenticated training receipt root is unsafe")
  encoded = canonical_json(receipt.to_dict()).encode()
  receipt_sha256 = _sha256_bytes(_RECEIPT_DOMAIN + encoded)
  target = root / receipt_sha256
  receipt_path = target / "receipt.json"
  if target.exists() or target.is_symlink():
    loaded = _load_authenticated_behavior_training_receipt_common(
      receipt_path,
      verify_replay_source=True,
    )
    if loaded.canonical_bytes != encoded:
      raise BehaviorTrainingAuthorityError("authenticated training receipt collision differs")
    return receipt_path
  staging = root / f".{receipt_sha256}.{secrets.token_hex(8)}.tmp"
  try:
    staging.mkdir(mode=0o700)
    descriptor = os.open(
      staging / "receipt.json",
      os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
      0o600,
    )
    try:
      view = memoryview(encoded)
      while view:
        written = os.write(descriptor, view)
        if written <= 0:
          raise OSError("authenticated training receipt write made no progress")
        view = view[written:]
      os.fsync(descriptor)
      os.fchmod(descriptor, 0o400)
    finally:
      os.close(descriptor)
    os.chmod(staging, 0o500)
    os.rename(staging, target)
  except OSError as error:
    if staging.exists() and not staging.is_symlink():
      try:
        os.chmod(staging, 0o700)
        child = staging / "receipt.json"
        if child.exists() and not child.is_symlink():
          child.unlink()
        staging.rmdir()
      except OSError:
        pass
    raise BehaviorTrainingAuthorityError("authenticated training receipt publication failed") from error
  loaded = _load_authenticated_behavior_training_receipt_common(
    receipt_path,
    verify_replay_source=True,
  )
  if loaded.canonical_bytes != encoded:
    raise BehaviorTrainingAuthorityError("published authenticated training receipt differs")
  return receipt_path


@dataclass(frozen=True, slots=True)
class RejectedBehaviorTrainingEvidence:
  result: BehaviorTrainingRun
  independent_aa_sha256: str

  def __post_init__(self) -> None:
    if (
      not isinstance(self.result, BehaviorTrainingRun)
      or self.result.disposition is TrainingDisposition.EXTERNAL_REVIEW_CANDIDATE
      or self.independent_aa_sha256 != self.result.sha256
    ):
      raise ValueError("rejected training evidence state is inconsistent")

  def to_dict(self) -> dict[str, object]:
    before = _verify_replay_source(self.result.replay_source)
    payload = {
      "activationEligible": False,
      "authorityKind": "diagnostic_rejection_only",
      "bitExactIndependentAA": True,
      "independentAASha256": self.independent_aa_sha256,
      "result": self.result.to_dict(),
      "schemaVersion": BEHAVIOR_TRAINING_AUTHORITY_SCHEMA_VERSION,
    }
    after = _verify_replay_source(self.result.replay_source)
    if before != after:
      raise BehaviorTrainingAuthorityError("reviewed source changed during evidence serialization")
    return payload


@dataclass(frozen=True, slots=True)
class ValidatedRejectedBehaviorTrainingEvidence:
  """Immutable fail-closed transcript; never candidate or activation authority."""

  receipt_sha256: str
  result_sha256: str
  independent_aa_sha256: str
  disposition: TrainingDisposition
  replay_source: ReviewedReplaySource
  canonical_bytes: bytes
  payload: Mapping[str, object]

  @property
  def activation_eligible(self) -> bool:
    return False


def _load_rejected_behavior_training_evidence_common(
  path: Path,
  *,
  verify_replay_source: bool,
) -> ValidatedRejectedBehaviorTrainingEvidence:
  if not isinstance(path, Path) or not path.is_absolute() or path.name != "rejection.json":
    raise BehaviorTrainingAuthorityError("rejected training evidence path is malformed")
  try:
    _sha256(path.parent.name, "rejected training evidence directory")
    if (
      path.parent.resolve(strict=True) != path.parent
      or path.parent.is_symlink()
      or {entry.name for entry in path.parent.iterdir()} != {"rejection.json"}
    ):
      raise BehaviorTrainingAuthorityError("rejected training evidence directory is unsafe")
    encoded, _ = _read_immutable_regular(
      path,
      "rejected training evidence",
      _TRAINING_RECEIPT_MAXIMUM_BYTES,
    )
    payload = json.loads(encoded)
  except BehaviorTrainingAuthorityError:
    raise
  except BehaviorReplayAuthorityError as error:
    raise BehaviorTrainingAuthorityError(str(error)) from error
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise BehaviorTrainingAuthorityError("rejected training evidence is malformed") from error
  if (
    type(payload) is not dict
    or set(payload) != {
      "activationEligible", "authorityKind", "bitExactIndependentAA",
      "independentAASha256", "result", "schemaVersion",
    }
    or encoded != canonical_json(payload).encode()
    or payload["schemaVersion"] != BEHAVIOR_TRAINING_AUTHORITY_SCHEMA_VERSION
    or payload["activationEligible"] is not False
    or payload["authorityKind"] != "diagnostic_rejection_only"
    or payload["bitExactIndependentAA"] is not True
  ):
    raise BehaviorTrainingAuthorityError("rejected training evidence is not fail-closed authority")
  receipt_sha256 = _sha256_bytes(_REJECTION_EVIDENCE_DOMAIN + encoded)
  if path.parent.name != receipt_sha256:
    raise BehaviorTrainingAuthorityError("rejected training evidence address differs")
  result = payload["result"]
  result_keys = {
    "acceptedIncumbentMode", "activationEligible", "controllerReplayCount",
    "controllerSeedSha256", "coarseSelection", "disposition", "importManifestSha256",
    "outerTest", "outerTestReleased", "partition", "partitionSha256",
    "physicalGenerationSha256", "physicalProfileSha256", "plantMemberIds",
    "preparationCount", "productionMode", "refinementSelection", "replaySource",
    "robustPlantSetSha256", "routeScanCount", "schemaVersion", "trainingTranscript",
    "validation", "winner", "workerCount",
  }
  if type(result) is not dict or set(result) != result_keys:
    raise BehaviorTrainingAuthorityError("rejected training result schema is malformed")
  try:
    independent_aa_sha256 = _sha256(payload["independentAASha256"], "rejected training A/A")
    disposition = TrainingDisposition(result["disposition"])
  except (BehaviorReplayAuthorityError, TypeError, ValueError) as error:
    raise BehaviorTrainingAuthorityError("rejected training disposition is malformed") from error
  result_sha256 = _sha256_json(result)
  if result_sha256 != independent_aa_sha256:
    raise BehaviorTrainingAuthorityError("rejected training A/A identity differs")
  if (
    disposition is TrainingDisposition.EXTERNAL_REVIEW_CANDIDATE
    or result["schemaVersion"] != BEHAVIOR_TRAINING_AUTHORITY_SCHEMA_VERSION
    or result["acceptedIncumbentMode"] != "bootstrap_exact_stock_only"
    or result["activationEligible"] is not False
    or result["productionMode"] is not True
    or result["workerCount"] != _WORKER_COUNT
  ):
    raise BehaviorTrainingAuthorityError("rejected training result is not diagnostic-only")
  for name in (
    "controllerSeedSha256", "importManifestSha256", "partitionSha256",
    "physicalGenerationSha256", "physicalProfileSha256", "robustPlantSetSha256",
  ):
    try:
      _sha256(result[name], f"rejected training {name}")
    except BehaviorReplayAuthorityError as error:
      raise BehaviorTrainingAuthorityError(str(error)) from error
  try:
    partition = frozen_behavior_partition_from_dict(result["partition"])
  except BehaviorPartitionError as error:
    raise BehaviorTrainingAuthorityError("rejected training partition is malformed") from error
  if result["partitionSha256"] != partition.sha256:
    raise BehaviorTrainingAuthorityError("rejected training partition identity differs")
  replay_source = _parse_reviewed_replay_source(result["replaySource"])
  if verify_replay_source:
    try:
      _verify_replay_source(replay_source)
    except BehaviorReplayAuthorityError as error:
      raise BehaviorTrainingAuthorityError(str(error)) from error
  raw_member_ids = result["plantMemberIds"]
  if type(raw_member_ids) is not list:
    raise BehaviorTrainingAuthorityError("rejected training plant members are malformed")
  try:
    member_ids = tuple(_sha256(value, "rejected plant member") for value in raw_member_ids)
  except BehaviorReplayAuthorityError as error:
    raise BehaviorTrainingAuthorityError(str(error)) from error
  if not 2 <= len(member_ids) <= 64 or member_ids != tuple(sorted(set(member_ids))):
    raise BehaviorTrainingAuthorityError("rejected training plant-member set is not robust")
  transcript = result["trainingTranscript"]
  if type(transcript) is not dict or set(transcript) != {
    "coarseSearch", "configuration", "outerTest", "refinementSearch",
    "schemaVersion", "validation",
  } or transcript["schemaVersion"] != 2 or type(transcript["coarseSearch"]) is not dict:
    raise BehaviorTrainingAuthorityError("rejected training transcript is malformed")
  if disposition is TrainingDisposition.NO_TRAINING_WINNER:
    coherent = (
      result["coarseSelection"] is None
      and result["refinementSelection"] is None
      and result["validation"] is None
      and result["outerTest"] is None
      and result["winner"] is None
      and result["outerTestReleased"] is False
      and transcript["refinementSearch"] is None
      and transcript["validation"] is None
      and transcript["outerTest"] is None
    )
  elif disposition is TrainingDisposition.VALIDATION_REJECTED:
    coherent = (
      type(result["coarseSelection"]) is dict
      and type(result["refinementSelection"]) is dict
      and type(result["validation"]) is dict
      and result["validation"].get("accepted") is False
      and result["outerTest"] is None
      and type(result["winner"]) is dict
      and result["outerTestReleased"] is False
      and type(transcript["refinementSearch"]) is dict
      and type(transcript["validation"]) is dict
      and transcript["outerTest"] is None
    )
  else:
    coherent = (
      type(result["coarseSelection"]) is dict
      and type(result["refinementSelection"]) is dict
      and type(result["validation"]) is dict
      and result["validation"].get("accepted") is True
      and type(result["outerTest"]) is dict
      and result["outerTest"].get("accepted") is False
      and type(result["winner"]) is dict
      and result["outerTestReleased"] is True
      and type(transcript["refinementSearch"]) is dict
      and type(transcript["validation"]) is dict
      and type(transcript["outerTest"]) is dict
    )
  if not coherent:
    raise BehaviorTrainingAuthorityError("rejected training stages disagree with disposition")
  for name in ("controllerReplayCount", "preparationCount", "routeScanCount"):
    if type(result[name]) is not int or result[name] < 0:
      raise BehaviorTrainingAuthorityError("rejected training accounting is malformed")
  if result["routeScanCount"] < result["preparationCount"]:
    raise BehaviorTrainingAuthorityError("rejected training accounting disagrees")
  frozen = _freeze_receipt_value(payload)
  assert isinstance(frozen, Mapping)
  return ValidatedRejectedBehaviorTrainingEvidence(
    receipt_sha256,
    result_sha256,
    independent_aa_sha256,
    disposition,
    replay_source,
    encoded,
    frozen,
  )


def load_rejected_behavior_training_evidence(
  path: Path,
) -> ValidatedRejectedBehaviorTrainingEvidence:
  """Load a diagnostic rejection transcript and verify current replay bytes."""
  return _load_rejected_behavior_training_evidence_common(path, verify_replay_source=True)


def publish_rejected_behavior_training_evidence(
  evidence: RejectedBehaviorTrainingEvidence,
  *,
  root: Path,
) -> Path:
  """Persist deterministic fail-closed evidence without creating a candidate."""
  if type(evidence) is not RejectedBehaviorTrainingEvidence:
    raise TypeError("evidence must be rejected production training evidence")
  if not isinstance(root, Path) or not root.is_absolute():
    raise TypeError("rejected training evidence root must be an absolute Path")
  root.mkdir(parents=True, exist_ok=True)
  if root.resolve(strict=True) != root or root.is_symlink() or not root.is_dir():
    raise BehaviorTrainingAuthorityError("rejected training evidence root is unsafe")
  encoded = canonical_json(evidence.to_dict()).encode()
  receipt_sha256 = _sha256_bytes(_REJECTION_EVIDENCE_DOMAIN + encoded)
  target = root / receipt_sha256
  report_path = target / "rejection.json"
  if target.exists() or target.is_symlink():
    loaded = _load_rejected_behavior_training_evidence_common(
      report_path,
      verify_replay_source=True,
    )
    if loaded.canonical_bytes != encoded:
      raise BehaviorTrainingAuthorityError("rejected training evidence collision differs")
    return report_path
  staging = root / f".{receipt_sha256}.{secrets.token_hex(8)}.tmp"
  try:
    staging.mkdir(mode=0o700)
    descriptor = os.open(
      staging / "rejection.json",
      os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
      0o600,
    )
    try:
      view = memoryview(encoded)
      while view:
        written = os.write(descriptor, view)
        if written <= 0:
          raise OSError("rejected training evidence write made no progress")
        view = view[written:]
      os.fsync(descriptor)
      os.fchmod(descriptor, 0o400)
    finally:
      os.close(descriptor)
    os.chmod(staging, 0o500)
    os.rename(staging, target)
  except OSError as error:
    if staging.exists() and not staging.is_symlink():
      try:
        os.chmod(staging, 0o700)
        child = staging / "rejection.json"
        if child.exists() and not child.is_symlink():
          child.unlink()
        staging.rmdir()
      except OSError:
        pass
    raise BehaviorTrainingAuthorityError("rejected training evidence publication failed") from error
  loaded = _load_rejected_behavior_training_evidence_common(
    report_path,
    verify_replay_source=True,
  )
  if loaded.canonical_bytes != encoded:
    raise BehaviorTrainingAuthorityError("published rejected training evidence differs")
  return report_path


@dataclass(frozen=True, slots=True)
class NonAuthoritativeBehaviorTrainingResult:
  """Explicitly test-only result; it can never serialize as authority evidence."""

  first: BehaviorTrainingRun
  independent_aa_sha256: str

  def __post_init__(self) -> None:
    if (
      not isinstance(self.first, BehaviorTrainingRun)
      or self.first.production_mode
      or self.independent_aa_sha256 != self.first.sha256
    ):
      raise ValueError("non-authoritative training result state is inconsistent")

  @property
  def authoritative(self) -> bool:
    return False


def _run_authenticated_behavior_training(
  *,
  evidence_store_root: Path,
  import_manifest_sha256: str,
  partition_receipt_path: Path,
  training_physical_receipt_path: Path,
  robust_plant_set_path: Path,
  replay_source: ReviewedReplaySource,
) -> tuple[BehaviorTrainingRun, str]:
  if not isinstance(replay_source, ReviewedReplaySource):
    raise TypeError("replay_source must be ReviewedReplaySource")
  arguments: dict[str, object] = {
    "evidence_store_root": evidence_store_root,
    "import_manifest_sha256": import_manifest_sha256,
    "partition_receipt_path": partition_receipt_path,
    "training_physical_receipt_path": training_physical_receipt_path,
    "robust_plant_set_path": robust_plant_set_path,
    "replay_source": replay_source,
  }
  _verify_replay_source(replay_source)
  first = _execute_epoch_once(**arguments)
  _verify_replay_source(replay_source)
  independent = _execute_epoch_once(**arguments)
  _verify_replay_source(replay_source)
  if first.to_json() != independent.to_json():
    raise BehaviorTrainingAuthorityError(
      "independent full training A/A differs; no receipt was issued",
    )
  return first, independent.sha256


def run_authenticated_behavior_training(
  *,
  evidence_store_root: Path,
  import_manifest_sha256: str,
  partition_receipt_path: Path,
  training_physical_receipt_path: Path,
  robust_plant_set_path: Path,
  replay_source: ReviewedReplaySource,
) -> AuthenticatedBehaviorTrainingReceipt | RejectedBehaviorTrainingEvidence:
  """Run the fixed four-worker production authority from immutable evidence."""
  observed_source = _verify_replay_source(replay_source)
  first, independent_sha256 = _run_authenticated_behavior_training(
    evidence_store_root=evidence_store_root,
    import_manifest_sha256=import_manifest_sha256,
    partition_receipt_path=partition_receipt_path,
    training_physical_receipt_path=training_physical_receipt_path,
    robust_plant_set_path=robust_plant_set_path,
    replay_source=observed_source,
  )
  _verify_replay_source(observed_source)
  if first.disposition is TrainingDisposition.EXTERNAL_REVIEW_CANDIDATE:
    if independent_sha256 != first.sha256:
      raise BehaviorTrainingAuthorityError("training result is not eligible for a receipt")
    # Python private objects are not security boundaries. Clean-source module
    # identity and transcript verification provide the authority; construction
    # is kept here to separate it from every injectable test workflow.
    outcome = object.__new__(AuthenticatedBehaviorTrainingReceipt)
    outcome._first = first
    outcome._independent_aa_sha256 = independent_sha256
  else:
    outcome = RejectedBehaviorTrainingEvidence(first, independent_sha256)
  outcome.to_dict()
  _verify_replay_source(observed_source)
  return outcome


def _run_authenticated_behavior_training_for_test(
  *,
  evidence_store_root: Path,
  import_manifest_sha256: str,
  partition_receipt_path: Path,
  training_physical_receipt_path: Path,
  robust_plant_set_path: Path,
  replay_source: ReviewedReplaySource,
  worker_count: int = 1,
  interface_registry: Mapping[str, type] | None = None,
) -> NonAuthoritativeBehaviorTrainingResult:
  arguments: dict[str, object] = {
    "evidence_store_root": evidence_store_root,
    "import_manifest_sha256": import_manifest_sha256,
    "partition_receipt_path": partition_receipt_path,
    "training_physical_receipt_path": training_physical_receipt_path,
    "robust_plant_set_path": robust_plant_set_path,
    "replay_source": replay_source,
    "worker_count": worker_count,
    "interface_registry": interface_registry,
  }
  first = _execute_epoch_once_for_test(**arguments)
  independent = _execute_epoch_once_for_test(**arguments)
  if first.to_json() != independent.to_json():
    raise BehaviorTrainingAuthorityError("independent test training A/A differs")
  return NonAuthoritativeBehaviorTrainingResult(first, independent.sha256)


def _strict_cli_request(
  value: object,
) -> tuple[Path, str, Path, Path, Path, ReviewedReplaySource]:
  keys = {
    "evidenceStoreRoot",
    "importManifestSha256",
    "partitionReceiptPath",
    "robustPlantSetPath",
    "replaySource",
    "trainingPhysicalReceiptPath",
  }
  if type(value) is not dict or set(value) != keys:
    raise BehaviorTrainingAuthorityError("training CLI request keys are incompatible")
  source = value["replaySource"]
  source_keys = {
    "moduleClosureSha256",
    "opendbcCommit",
    "pandaCommit",
    "runtimeIdentitySha256",
    "sourceCompositionSha256",
    "sourceOpenpilotCommit",
    "schemaVersion",
  }
  if type(source) is not dict or set(source) != source_keys:
    raise BehaviorTrainingAuthorityError("training CLI replay source is malformed")
  return (
    Path(value["evidenceStoreRoot"]),
    value["importManifestSha256"],
    Path(value["partitionReceiptPath"]),
    Path(value["trainingPhysicalReceiptPath"]),
    Path(value["robustPlantSetPath"]),
    ReviewedReplaySource(
      source_openpilot_commit=source["sourceOpenpilotCommit"],
      opendbc_commit=source["opendbcCommit"],
      panda_commit=source["pandaCommit"],
      source_composition_sha256=source["sourceCompositionSha256"],
      runtime_identity_sha256=source["runtimeIdentitySha256"],
      module_closure_sha256=source["moduleClosureSha256"],
      schema_version=source["schemaVersion"],
    ),
  )


def main() -> int:
  """Read one strict canonical request from stdin and emit receipt evidence."""
  try:
    encoded = sys.stdin.buffer.read(1024 * 1024 + 1)
    if not encoded or len(encoded) > 1024 * 1024:
      raise BehaviorTrainingAuthorityError("training CLI request is outside its size bound")
    request = json.loads(encoded)
    if encoded != (canonical_json(request) + "\n").encode("utf-8"):
      raise BehaviorTrainingAuthorityError("training CLI request is not canonical JSON")
    store, manifest, partition_receipt, physical_receipt, plant_set, source = _strict_cli_request(request)
    outcome = run_authenticated_behavior_training(
      evidence_store_root=store,
      import_manifest_sha256=manifest,
      partition_receipt_path=partition_receipt,
      training_physical_receipt_path=physical_receipt,
      robust_plant_set_path=plant_set,
      replay_source=source,
    )
  except (BehaviorTrainingAuthorityError, BehaviorReplayAuthorityError, TypeError, ValueError) as error:
    sys.stderr.write(f"behavior training failed: {error}\n")
    return 1
  sys.stdout.write(canonical_json(outcome.to_dict()) + "\n")
  return 0 if isinstance(outcome, AuthenticatedBehaviorTrainingReceipt) else 2


if __name__ == "__main__":
  raise SystemExit(main())
