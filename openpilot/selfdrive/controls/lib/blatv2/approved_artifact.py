"""Fail-closed approved-artifact loading and engagement-boundary activation.

This module is persistence and lifecycle policy only. It does not construct a
controller or calculate a command. An artifact is eligible only when its exact
profile, policy, vehicle, source, evidence, replay-harness, and gate identities
all validate. Any absent, malformed, stale, or partially-passed artifact
returns stock bootstrap with a diagnostic reason instead of raising into the
live process.

Activation changes happen only in ``prepare_offroad``. ``begin_engagement``
is a read-only binding operation: it performs no Params write, does not
promote staged content, and does not execute rollback. Explicit driver
feedback is consumed only offroad and only when it identifies the exact
provisional profile. No driver-intervention signal exists in this API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import hmac
import json
import math
import re
from typing import Any, Protocol

from openpilot.selfdrive.controls.lib.blatv2.bootstrap import (
  ControllerSelection,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  BehaviorLearningFinalization,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import (
  BehaviorPolicy,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  VehicleCalibrationProfile,
)
from openpilot.selfdrive.controls.lib.blatv2.device_acceptance import (
  DeviceAcceptanceError,
  DeviceAcceptanceReceipt,
)
from openpilot.selfdrive.controls.lib.blatv2.feedback import (
  FEEDBACK_REQUEST_PARAM,
  FEEDBACK_RESPONSE_PARAM,
  FeedbackChoice,
  FeedbackRequest,
  FeedbackResponse,
)
from openpilot.selfdrive.controls.lib.blatv2.policy import ControllerPolicy
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  VehicleProfile,
)


APPROVED_ARTIFACT_SCHEMA_VERSION = 7
CALIBRATION_SELECTION_SCHEMA_VERSION = 2
ACTIVATION_STATE_SCHEMA_VERSION = 1
APPROVED_ARTIFACT_PARAM = "BLaTv2ApprovedArtifact"
ACTIVATION_STATE_PARAM = "BLaTv2ActivationState"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_ARTIFACT_KEYS = frozenset((
  "schemaVersion",
  "vehicleProfileJson",
  "vehicleProfileSha256",
  "calibrationProfileJson",
  "calibrationProfileSha256",
  "controllerPolicyJson",
  "controllerPolicySha256",
  "horizonPolicySha256",
  "runtimeVehicleIdentitySha256",
  "sourceOpenpilotCommit",
  "opendbcCommit",
  "pandaCommit",
  "calibrationSelectionManifestJson",
  "calibrationSelectionManifestSha256",
  "learnerEvidenceSha256",
  "behaviorFinalizationJson",
  "behaviorFinalizationSha256",
  "behaviorSelectionSha256",
  "replayHarnessCommit",
  "replayPassed",
  "deliveredReplayPassed",
  "deterministicAaPassed",
  "deviceAcceptanceReceiptJson",
  "deviceAcceptanceReceiptSha256",
  "smoothPassed",
  "swiftPassed",
  "strongPassed",
))
_POLICY_KEYS = frozenset((
  "damping_ratio",
  "natural_frequency_per_s",
  "observer",
  "provenance",
  "provisional",
  "revision",
  "schema_version",
))
_OBSERVER_KEYS = frozenset((
  "max_abs_disturbance_torque",
  "time_constant_s",
))
_IDENTITY_KEYS = frozenset((
  "artifactSha256",
  "profileSha256",
  "profileRevision",
))
_STATE_KEYS = frozenset((
  "schemaVersion",
  "activeArtifact",
  "activeProfileIdentity",
  "previousArtifact",
  "stagedArtifact",
  "provisional",
  "rollbackPending",
  "rejectedProfileIdentities",
))
_CALIBRATION_SELECTION_KEYS = frozenset((
  "allNodesQualified",
  "candidateCalibrationProfileSha256",
  "learnerEvidenceSha256",
  "qualificationManifestSha256",
  "schemaVersion",
  "selectedControllerProfileSha256",
))


class ParamsLike(Protocol):
  def get(self, key: str, block: bool = False): ...

  def put(self, key: str, value, block: bool = False): ...

  def remove(self, key: str): ...


class ArtifactDiagnostic(StrEnum):
  OK = "ok"
  ABSENT = "absent"
  PARAM_READ_ERROR = "param_read_error"
  MALFORMED = "malformed"
  PROFILE_HASH_MISMATCH = "profile_hash_mismatch"
  POLICY_HASH_MISMATCH = "policy_hash_mismatch"
  CALIBRATION_PROOF_MISMATCH = "calibration_proof_mismatch"
  BEHAVIOR_PROOF_MISMATCH = "behavior_proof_mismatch"
  DEVICE_ACCEPTANCE_PROOF_MISMATCH = "device_acceptance_proof_mismatch"
  EXTERNAL_SAFETY_AUTHORITY_UNAVAILABLE = (
    "external_safety_authority_unavailable"
  )
  UNQUALIFIED_PROFILE = "unqualified_profile"
  PROVISIONAL_POLICY = "provisional_policy"
  VEHICLE_MISMATCH = "vehicle_mismatch"
  RUNTIME_VEHICLE_MISMATCH = "runtime_vehicle_mismatch"
  SOURCE_COMMIT_MISMATCH = "source_commit_mismatch"
  OPENDBC_COMMIT_MISMATCH = "opendbc_commit_mismatch"
  PANDA_COMMIT_MISMATCH = "panda_commit_mismatch"
  UNVERIFIED_ACTUATION_ENVELOPE = "unverified_actuation_envelope"
  GATE_FAILED = "gate_failed"
  STATE_INVALID = "state_invalid"
  STATE_STALE_BUILD = "state_stale_build"


class ArtifactValidationError(ValueError):
  def __init__(self, reason: ArtifactDiagnostic, message: str):
    super().__init__(message)
    self.reason = reason


def _fail(reason: ArtifactDiagnostic, message: str) -> None:
  raise ArtifactValidationError(reason, message)


def _canonical_json(payload: object) -> str:
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  )


def _sha256_text(encoded: str) -> str:
  return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _strict_object(
  value: object,
  expected_keys: frozenset[str],
  name: str,
) -> dict[str, Any]:
  if type(value) is not dict or frozenset(value) != expected_keys:
    _fail(ArtifactDiagnostic.MALFORMED, f"{name} keys are not canonical")
  return value


def _strict_sha256(value: object, name: str) -> str:
  if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
    _fail(
      ArtifactDiagnostic.MALFORMED,
      f"{name} must be a lowercase SHA-256",
    )
  return value


def _strict_commit(value: object, name: str) -> str:
  if type(value) is not str or _GIT_COMMIT_RE.fullmatch(value) is None:
    _fail(
      ArtifactDiagnostic.MALFORMED,
      f"{name} must be a full lowercase Git commit",
    )
  return value


def _strict_bool(value: object, name: str) -> bool:
  if type(value) is not bool:
    _fail(ArtifactDiagnostic.MALFORMED, f"{name} must be boolean")
  return value


def _strict_number(value: object, name: str) -> float:
  if type(value) not in (int, float):
    _fail(ArtifactDiagnostic.MALFORMED, f"{name} must be numeric")
  numeric = float(value)
  if not math.isfinite(numeric):
    _fail(ArtifactDiagnostic.MALFORMED, f"{name} must be finite")
  return numeric


def _policy_from_canonical_json(encoded: object) -> ControllerPolicy:
  if type(encoded) is not str:
    _fail(
      ArtifactDiagnostic.MALFORMED,
      "controllerPolicyJson must be text",
    )
  try:
    payload = json.loads(encoded)
  except (TypeError, ValueError, json.JSONDecodeError) as exc:
    raise ArtifactValidationError(
      ArtifactDiagnostic.MALFORMED,
      "controllerPolicyJson is invalid JSON",
    ) from exc
  policy_payload = _strict_object(payload, _POLICY_KEYS, "controller policy")
  if type(policy_payload["schema_version"]) is not int:
    _fail(
      ArtifactDiagnostic.MALFORMED,
      "controller policy schema_version must be integer",
    )
  if type(policy_payload["revision"]) is not int:
    _fail(
      ArtifactDiagnostic.MALFORMED,
      "controller policy revision must be integer",
    )
  if type(policy_payload["provenance"]) is not str:
    _fail(
      ArtifactDiagnostic.MALFORMED,
      "controller policy provenance must be text",
    )
  provisional = _strict_bool(
    policy_payload["provisional"],
    "controller policy provisional",
  )
  observer = policy_payload["observer"]
  observer_time_constant_s = None
  observer_max_abs_disturbance_torque = None
  if observer is not None:
    observer_payload = _strict_object(
      observer,
      _OBSERVER_KEYS,
      "controller observer",
    )
    observer_time_constant_s = _strict_number(
      observer_payload["time_constant_s"],
      "observer time_constant_s",
    )
    observer_max_abs_disturbance_torque = _strict_number(
      observer_payload["max_abs_disturbance_torque"],
      "observer max_abs_disturbance_torque",
    )
  try:
    policy = ControllerPolicy(
      revision=policy_payload["revision"],
      provenance=policy_payload["provenance"],
      provisional=provisional,
      natural_frequency_per_s=_strict_number(
        policy_payload["natural_frequency_per_s"],
        "controller natural_frequency_per_s",
      ),
      damping_ratio=_strict_number(
        policy_payload["damping_ratio"],
        "controller damping_ratio",
      ),
      observer_time_constant_s=observer_time_constant_s,
      observer_max_abs_disturbance_torque=(
        observer_max_abs_disturbance_torque
      ),
      schema_version=policy_payload["schema_version"],
    )
  except (TypeError, ValueError, OverflowError) as exc:
    raise ArtifactValidationError(
      ArtifactDiagnostic.MALFORMED,
      "controller policy values are invalid",
    ) from exc
  if policy.to_json() != encoded:
    _fail(
      ArtifactDiagnostic.MALFORMED,
      "controllerPolicyJson is not canonical",
    )
  return policy


def _profile_from_canonical_json(
  encoded: object,
  expected_vehicle_identity: str,
) -> VehicleProfile:
  if type(encoded) is not str:
    _fail(
      ArtifactDiagnostic.MALFORMED,
      "vehicleProfileJson must be text",
    )
  try:
    profile = VehicleProfile.from_json(
      encoded,
      expected_vehicle_identity=expected_vehicle_identity,
    )
  except (KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
    message = str(exc)
    reason = (
      ArtifactDiagnostic.VEHICLE_MISMATCH
      if "different vehicle" in message
      else ArtifactDiagnostic.MALFORMED
    )
    raise ArtifactValidationError(
      reason,
      "vehicleProfileJson is invalid",
    ) from exc
  if profile.to_json() != encoded:
    _fail(
      ArtifactDiagnostic.MALFORMED,
      "vehicleProfileJson is not canonical",
    )
  return profile


def _calibration_profile_from_canonical_json(
  encoded: object,
  expected_vehicle_identity: str,
) -> VehicleCalibrationProfile:
  if type(encoded) is not str:
    _fail(
      ArtifactDiagnostic.MALFORMED,
      "calibrationProfileJson must be text",
    )
  try:
    profile = VehicleCalibrationProfile.from_json(
      encoded,
      expected_vehicle_identity=expected_vehicle_identity,
    )
  except (KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
    message = str(exc)
    reason = (
      ArtifactDiagnostic.VEHICLE_MISMATCH
      if "different vehicle" in message
      else ArtifactDiagnostic.MALFORMED
    )
    raise ArtifactValidationError(
      reason,
      "calibrationProfileJson is invalid",
    ) from exc
  if profile.to_json() != encoded:
    _fail(
      ArtifactDiagnostic.MALFORMED,
      "calibrationProfileJson is not canonical",
    )
  return profile


@dataclass(frozen=True, slots=True)
class CalibrationSelectionManifest:
  """Canonical preimage binding learned evidence to the selected profile.

  The calibration coordinator's complete qualification report remains a
  separately versioned artifact.  This compact selection manifest is the
  promotion boundary: its hash is derived here, never supplied without its
  preimage, and both profile identities must name the exact live profile.
  """

  selected_controller_profile_sha256: str
  candidate_calibration_profile_sha256: str
  learner_evidence_sha256: str
  qualification_manifest_sha256: str
  all_nodes_qualified: bool
  schema_version: int = CALIBRATION_SELECTION_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != CALIBRATION_SELECTION_SCHEMA_VERSION:
      _fail(
        ArtifactDiagnostic.MALFORMED,
        "calibration selection schema is incompatible",
      )
    for name, value in (
      (
        "selectedControllerProfileSha256",
        self.selected_controller_profile_sha256,
      ),
      (
        "candidateCalibrationProfileSha256",
        self.candidate_calibration_profile_sha256,
      ),
      ("learnerEvidenceSha256", self.learner_evidence_sha256),
      ("qualificationManifestSha256", self.qualification_manifest_sha256),
    ):
      _strict_sha256(value, name)
    if type(self.all_nodes_qualified) is not bool:
      _fail(
        ArtifactDiagnostic.MALFORMED,
        "calibration allNodesQualified must be boolean",
      )
    if not self.all_nodes_qualified:
      _fail(
        ArtifactDiagnostic.GATE_FAILED,
        "calibration selection requires every node to qualify",
      )
  def to_param(self) -> dict[str, object]:
    return {
      "allNodesQualified": self.all_nodes_qualified,
      "candidateCalibrationProfileSha256": (
        self.candidate_calibration_profile_sha256
      ),
      "learnerEvidenceSha256": self.learner_evidence_sha256,
      "qualificationManifestSha256": self.qualification_manifest_sha256,
      "schemaVersion": self.schema_version,
      "selectedControllerProfileSha256": (
        self.selected_controller_profile_sha256
      ),
    }

  def to_json(self) -> str:
    return _canonical_json(self.to_param())

  @property
  def sha256(self) -> str:
    return _sha256_text(self.to_json())

  @classmethod
  def from_json(cls, encoded: object) -> CalibrationSelectionManifest:
    if type(encoded) is not str:
      _fail(
        ArtifactDiagnostic.MALFORMED,
        "calibrationSelectionManifestJson must be text",
      )
    try:
      raw = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
      raise ArtifactValidationError(
        ArtifactDiagnostic.MALFORMED,
        "calibration selection manifest is invalid JSON",
      ) from exc
    payload = _strict_object(
      raw,
      _CALIBRATION_SELECTION_KEYS,
      "calibration selection manifest",
    )
    if type(payload["schemaVersion"]) is not int:
      _fail(
        ArtifactDiagnostic.MALFORMED,
        "calibration selection schemaVersion must be integer",
      )
    manifest = cls(
      selected_controller_profile_sha256=_strict_sha256(
        payload["selectedControllerProfileSha256"],
        "selectedControllerProfileSha256",
      ),
      candidate_calibration_profile_sha256=_strict_sha256(
        payload["candidateCalibrationProfileSha256"],
        "candidateCalibrationProfileSha256",
      ),
      learner_evidence_sha256=_strict_sha256(
        payload["learnerEvidenceSha256"],
        "learnerEvidenceSha256",
      ),
      qualification_manifest_sha256=_strict_sha256(
        payload["qualificationManifestSha256"],
        "qualificationManifestSha256",
      ),
      all_nodes_qualified=_strict_bool(
        payload["allNodesQualified"],
        "allNodesQualified",
      ),
      schema_version=payload["schemaVersion"],
    )
    if manifest.to_json() != encoded:
      _fail(
        ArtifactDiagnostic.MALFORMED,
        "calibrationSelectionManifestJson is not canonical",
      )
    return manifest


def _behavior_finalization_from_canonical_json(
  encoded: object,
) -> BehaviorLearningFinalization:
  if type(encoded) is not str:
    _fail(
      ArtifactDiagnostic.MALFORMED,
      "behaviorFinalizationJson must be text",
    )
  try:
    finalization = BehaviorLearningFinalization.from_json(encoded)
  except (KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
    raise ArtifactValidationError(
      ArtifactDiagnostic.MALFORMED,
      "behaviorFinalizationJson is invalid",
    ) from exc
  if finalization.to_json() != encoded:
    _fail(
      ArtifactDiagnostic.MALFORMED,
      "behaviorFinalizationJson is not canonical",
    )
  return finalization


def _device_acceptance_from_canonical_json(
  encoded: object,
) -> DeviceAcceptanceReceipt:
  if type(encoded) is not str:
    _fail(
      ArtifactDiagnostic.MALFORMED,
      "deviceAcceptanceReceiptJson must be text",
    )
  try:
    receipt = DeviceAcceptanceReceipt.from_json(encoded)
  except DeviceAcceptanceError as error:
    raise ArtifactValidationError(
      ArtifactDiagnostic.MALFORMED,
      "deviceAcceptanceReceiptJson is invalid",
    ) from error
  return receipt


def _controller_profile_matches_calibration(
  controller: VehicleProfile,
  calibration: VehicleCalibrationProfile,
) -> bool:
  """Verify every learned observable survives controller composition exactly.

  Rack gain and damping are runtime-seed facts and are checked against the
  detected bundle by ``ModularLiveController``.  Everything learned from road
  evidence is checked here, including the residual offset that earlier profile
  formats silently dropped.
  """
  if (
    not isinstance(controller, VehicleProfile)
    or not isinstance(calibration, VehicleCalibrationProfile)
    or controller.speed_nodes_mps != calibration.speed_nodes_mps
    or controller.vehicle_identity != calibration.vehicle_identity
    or controller.revision != calibration.revision
    or len(controller.nodes) != len(calibration.nodes)
  ):
    return False
  for controller_node, calibration_node in zip(
    controller.nodes,
    calibration.nodes,
    strict=True,
  ):
    physical = controller_node.parameters
    observable = calibration_node.parameters
    if (
      controller_node.speed_mps != calibration_node.speed_mps
      or physical.torque_per_lateral_accel
      != observable.torque_per_lateral_accel
      or physical.lateral_accel_offset_correction_mps2
      != observable.lateral_accel_offset_correction_mps2
      or physical.kinetic_friction_torque
      != observable.kinetic_friction_torque
      or physical.static_friction_torque
      != observable.static_breakaway_torque
      or physical.transport_delay_s != observable.transport_delay_s
      or physical.rack_rate_resolution_deg_s
      != observable.rack_rate_resolution_deg_s
      or physical.confidence != observable.confidence
      or physical.qualified is not observable.qualified
      or controller_node.clean_support_s
      != (
        calibration_node.base_support_s
        + calibration_node.moving_support_s
        + calibration_node.breakaway_support_s
      )
      or controller_node.sample_count
      != (
        calibration_node.base_sample_count
        + calibration_node.moving_sample_count
        + calibration_node.breakaway_sample_count
      )
      or controller_node.cross_fit_route_count != calibration_node.cross_fit_route_count
      or controller_node.full_fit_candidate_rms
      != calibration_node.full_fit_candidate_rms
    ):
      return False
  return True


@dataclass(frozen=True, slots=True)
class ApprovedProfileArtifact:
  """Exact, fail-closed proof bundle for one learned controller artifact.

  ``replay_passed`` means that the raw-command and post-envelope applied
  five-metric gates passed. ``delivered_replay_passed`` is deliberately
  separate: it proves the vehicle/twin delivered-curvature gates rather than
  allowing a command-space pass to imply physical path authority.
  Device timing/comms is carried by a content-addressed route-evidence
  receipt; no caller boolean can approve the runtime budget. External safety
  authority remains unresolved, so persisted production activation fails
  closed while candidate bundles remain parseable for offline inspection.
  ``smooth_passed``,
  ``swift_passed``, and ``strong_passed`` are independent behavioral contracts:
  no aggregate score may trade one product value for another. The exact
  behavioral selection evidence is bound by the canonical behavioral
  finalization proof. The physical profile is independently bound to its
  canonical calibration selection manifest. Neither proof can be replaced by
  an opaque caller-supplied hash.
  All gates are mandatory.
  """

  vehicle_profile: VehicleProfile
  calibration_profile: VehicleCalibrationProfile
  controller_policy: ControllerPolicy
  horizon_policy_sha256: str
  runtime_vehicle_identity_sha256: str
  source_openpilot_commit: str
  opendbc_commit: str
  panda_commit: str
  calibration_selection_manifest: CalibrationSelectionManifest
  behavior_finalization: BehaviorLearningFinalization
  replay_harness_commit: str
  replay_passed: bool
  delivered_replay_passed: bool
  deterministic_aa_passed: bool
  device_acceptance_receipt: DeviceAcceptanceReceipt
  smooth_passed: bool
  swift_passed: bool
  strong_passed: bool
  schema_version: int = APPROVED_ARTIFACT_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if self.schema_version != APPROVED_ARTIFACT_SCHEMA_VERSION:
      _fail(
        ArtifactDiagnostic.MALFORMED,
        "approved artifact schema is incompatible",
      )
    _strict_sha256(
      self.runtime_vehicle_identity_sha256,
      "runtimeVehicleIdentitySha256",
    )
    _strict_commit(
      self.source_openpilot_commit,
      "sourceOpenpilotCommit",
    )
    _strict_commit(self.opendbc_commit, "opendbcCommit")
    _strict_commit(self.panda_commit, "pandaCommit")
    _strict_commit(self.replay_harness_commit, "replayHarnessCommit")
    _strict_sha256(self.horizon_policy_sha256, "horizonPolicySha256")
    if not isinstance(self.vehicle_profile, VehicleProfile):
      _fail(
        ArtifactDiagnostic.MALFORMED,
        "approved controller profile has the wrong type",
      )
    if not self.vehicle_profile.qualified or self.vehicle_profile.revision < 1:
      _fail(
        ArtifactDiagnostic.UNQUALIFIED_PROFILE,
        "approved profile must be complete, qualified, and learned",
      )
    if (
      not isinstance(self.calibration_profile, VehicleCalibrationProfile)
      or not self.calibration_profile.qualified
      or self.calibration_profile.revision < 1
    ):
      _fail(
        ArtifactDiagnostic.UNQUALIFIED_PROFILE,
        "approved observable calibration must be complete and qualified",
      )
    if not _controller_profile_matches_calibration(
      self.vehicle_profile,
      self.calibration_profile,
    ):
      _fail(
        ArtifactDiagnostic.CALIBRATION_PROOF_MISMATCH,
        "controller profile does not preserve the observable calibration",
      )
    if not isinstance(self.controller_policy, ControllerPolicy):
      _fail(
        ArtifactDiagnostic.MALFORMED,
        "approved controller policy has the wrong type",
      )
    if self.controller_policy.provisional:
      _fail(
        ArtifactDiagnostic.PROVISIONAL_POLICY,
        "provisional controller policy cannot be approved",
      )
    gates = (
      self.replay_passed,
      self.delivered_replay_passed,
      self.deterministic_aa_passed,
      self.smooth_passed,
      self.swift_passed,
      self.strong_passed,
    )
    if any(type(gate) is not bool for gate in gates):
      _fail(
        ArtifactDiagnostic.MALFORMED,
        "artifact gate values must be boolean",
      )
    if not all(gates):
      _fail(
        ArtifactDiagnostic.GATE_FAILED,
        "every artifact acceptance gate must pass",
      )
    if not isinstance(
      self.device_acceptance_receipt,
      DeviceAcceptanceReceipt,
    ):
      _fail(
        ArtifactDiagnostic.MALFORMED,
        "device acceptance receipt has the wrong type",
      )
    receipt = self.device_acceptance_receipt
    if not receipt.passed:
      _fail(
        ArtifactDiagnostic.GATE_FAILED,
        "device timing/comms acceptance did not pass",
      )
    if not isinstance(
      self.calibration_selection_manifest,
      CalibrationSelectionManifest,
    ):
      _fail(
        ArtifactDiagnostic.MALFORMED,
        "calibration selection manifest has the wrong type",
      )
    if not hmac.compare_digest(
      self.calibration_selection_manifest.selected_controller_profile_sha256,
      self.vehicle_profile_sha256,
    ):
      _fail(
        ArtifactDiagnostic.CALIBRATION_PROOF_MISMATCH,
        "calibration selection does not name the active profile",
      )
    if not hmac.compare_digest(
      self.calibration_selection_manifest.candidate_calibration_profile_sha256,
      self.calibration_profile_sha256,
    ):
      _fail(
        ArtifactDiagnostic.CALIBRATION_PROOF_MISMATCH,
        "calibration selection does not name the learned calibration",
      )
    if not isinstance(
      self.behavior_finalization,
      BehaviorLearningFinalization,
    ):
      _fail(
        ArtifactDiagnostic.MALFORMED,
        "behavior finalization has the wrong type",
      )
    finalization = self.behavior_finalization
    if not finalization.passed:
      _fail(
        ArtifactDiagnostic.GATE_FAILED,
        "behavior finalization did not pass every required gate",
      )
    behavior_projection = BehaviorPolicy.from_controller_policy(
      self.controller_policy,
    )
    if (
      finalization.final_behavior_policy != behavior_projection
      or finalization.final_behavior_policy_sha256
      != behavior_projection.sha256
    ):
      _fail(
        ArtifactDiagnostic.BEHAVIOR_PROOF_MISMATCH,
        "behavior finalization does not name the active controller policy",
      )
    if (
      self.smooth_passed != finalization.smooth_passed
      or self.swift_passed != finalization.swift_passed
      or self.strong_passed != finalization.strong_passed
    ):
      _fail(
        ArtifactDiagnostic.BEHAVIOR_PROOF_MISMATCH,
        "artifact behavioral gates disagree with the finalization proof",
      )
    if (
      receipt.vehicle_identity != self.vehicle_profile.vehicle_identity
      or not hmac.compare_digest(
        receipt.source_openpilot_commit,
        self.source_openpilot_commit,
      )
      or not hmac.compare_digest(receipt.opendbc_commit, self.opendbc_commit)
      or not hmac.compare_digest(receipt.panda_commit, self.panda_commit)
      or not hmac.compare_digest(
        receipt.runtime_identity_sha256,
        self.runtime_vehicle_identity_sha256,
      )
      or not hmac.compare_digest(
        receipt.profile_sha256,
        self.vehicle_profile_sha256,
      )
      or not hmac.compare_digest(
        receipt.controller_policy_sha256,
        self.controller_policy_sha256,
      )
      or not hmac.compare_digest(
        receipt.horizon_policy_sha256,
        self.horizon_policy_sha256,
      )
    ):
      _fail(
        ArtifactDiagnostic.DEVICE_ACCEPTANCE_PROOF_MISMATCH,
        "device acceptance receipt names another controller identity",
      )

  @property
  def vehicle_profile_sha256(self) -> str:
    return _sha256_text(self.vehicle_profile.to_json())

  @property
  def controller_policy_sha256(self) -> str:
    return self.controller_policy.sha256

  @property
  def calibration_profile_sha256(self) -> str:
    return _sha256_text(self.calibration_profile.to_json())

  @property
  def learner_evidence_sha256(self) -> str:
    return self.calibration_selection_manifest.learner_evidence_sha256

  @property
  def behavior_selection_sha256(self) -> str:
    selection = self.behavior_finalization.behavior_selection_sha256
    if selection is None:
      raise AssertionError("approved behavior finalization lacks a selection")
    return selection

  def to_param(self) -> dict[str, object]:
    return {
      "schemaVersion": self.schema_version,
      "vehicleProfileJson": self.vehicle_profile.to_json(),
      "vehicleProfileSha256": self.vehicle_profile_sha256,
      "calibrationProfileJson": self.calibration_profile.to_json(),
      "calibrationProfileSha256": self.calibration_profile_sha256,
      "controllerPolicyJson": self.controller_policy.to_json(),
      "controllerPolicySha256": self.controller_policy_sha256,
      "horizonPolicySha256": self.horizon_policy_sha256,
      "runtimeVehicleIdentitySha256": (
        self.runtime_vehicle_identity_sha256
      ),
      "sourceOpenpilotCommit": self.source_openpilot_commit,
      "opendbcCommit": self.opendbc_commit,
      "pandaCommit": self.panda_commit,
      "calibrationSelectionManifestJson": (
        self.calibration_selection_manifest.to_json()
      ),
      "calibrationSelectionManifestSha256": (
        self.calibration_selection_manifest.sha256
      ),
      "learnerEvidenceSha256": self.learner_evidence_sha256,
      "behaviorFinalizationJson": self.behavior_finalization.to_json(),
      "behaviorFinalizationSha256": self.behavior_finalization.sha256,
      "behaviorSelectionSha256": self.behavior_selection_sha256,
      "replayHarnessCommit": self.replay_harness_commit,
      "replayPassed": self.replay_passed,
      "deliveredReplayPassed": self.delivered_replay_passed,
      "deterministicAaPassed": self.deterministic_aa_passed,
      "deviceAcceptanceReceiptJson": (
        self.device_acceptance_receipt.to_json()
      ),
      "deviceAcceptanceReceiptSha256": (
        self.device_acceptance_receipt.sha256
      ),
      "smoothPassed": self.smooth_passed,
      "swiftPassed": self.swift_passed,
      "strongPassed": self.strong_passed,
    }

  @property
  def artifact_sha256(self) -> str:
    return _sha256_text(_canonical_json(self.to_param()))

  @classmethod
  def from_param(
    cls,
    value: object,
    *,
    expected_vehicle_identity: str,
  ) -> ApprovedProfileArtifact:
    payload = _strict_object(value, _ARTIFACT_KEYS, "approved artifact")
    if (
      type(payload["schemaVersion"]) is not int
      or payload["schemaVersion"] != APPROVED_ARTIFACT_SCHEMA_VERSION
    ):
      _fail(
        ArtifactDiagnostic.MALFORMED,
        "approved artifact schemaVersion is incompatible",
      )

    profile_hash = _strict_sha256(
      payload["vehicleProfileSha256"],
      "vehicleProfileSha256",
    )
    calibration_profile_hash = _strict_sha256(
      payload["calibrationProfileSha256"],
      "calibrationProfileSha256",
    )
    policy_hash = _strict_sha256(
      payload["controllerPolicySha256"],
      "controllerPolicySha256",
    )
    horizon_policy_hash = _strict_sha256(
      payload["horizonPolicySha256"],
      "horizonPolicySha256",
    )
    profile = _profile_from_canonical_json(
      payload["vehicleProfileJson"],
      expected_vehicle_identity,
    )
    calibration_profile = _calibration_profile_from_canonical_json(
      payload["calibrationProfileJson"],
      expected_vehicle_identity,
    )
    policy = _policy_from_canonical_json(
      payload["controllerPolicyJson"],
    )
    calibration_manifest_hash = _strict_sha256(
      payload["calibrationSelectionManifestSha256"],
      "calibrationSelectionManifestSha256",
    )
    calibration_manifest = CalibrationSelectionManifest.from_json(
      payload["calibrationSelectionManifestJson"],
    )
    behavior_finalization_hash = _strict_sha256(
      payload["behaviorFinalizationSha256"],
      "behaviorFinalizationSha256",
    )
    behavior_finalization = _behavior_finalization_from_canonical_json(
      payload["behaviorFinalizationJson"],
    )
    device_acceptance_hash = _strict_sha256(
      payload["deviceAcceptanceReceiptSha256"],
      "deviceAcceptanceReceiptSha256",
    )
    device_acceptance = _device_acceptance_from_canonical_json(
      payload["deviceAcceptanceReceiptJson"],
    )
    if not hmac.compare_digest(
      profile_hash,
      _sha256_text(profile.to_json()),
    ):
      _fail(
        ArtifactDiagnostic.PROFILE_HASH_MISMATCH,
        "vehicle profile hash does not match its JSON",
      )
    if not hmac.compare_digest(
      calibration_profile_hash,
      _sha256_text(calibration_profile.to_json()),
    ):
      _fail(
        ArtifactDiagnostic.CALIBRATION_PROOF_MISMATCH,
        "calibration profile hash does not match its JSON",
      )
    if not hmac.compare_digest(policy_hash, policy.sha256):
      _fail(
        ArtifactDiagnostic.POLICY_HASH_MISMATCH,
        "controller policy hash does not match its JSON",
      )
    if not hmac.compare_digest(
      device_acceptance_hash,
      device_acceptance.sha256,
    ):
      _fail(
        ArtifactDiagnostic.DEVICE_ACCEPTANCE_PROOF_MISMATCH,
        "device acceptance receipt hash does not match its JSON",
      )
    if not device_acceptance.passed:
      _fail(
        ArtifactDiagnostic.GATE_FAILED,
        "device timing/comms acceptance did not pass",
      )
    if not hmac.compare_digest(
      calibration_manifest_hash,
      calibration_manifest.sha256,
    ):
      _fail(
        ArtifactDiagnostic.CALIBRATION_PROOF_MISMATCH,
        "calibration selection hash does not match its manifest",
      )
    if not hmac.compare_digest(
      calibration_manifest.learner_evidence_sha256,
      _strict_sha256(
        payload["learnerEvidenceSha256"],
        "learnerEvidenceSha256",
      ),
    ):
      _fail(
        ArtifactDiagnostic.CALIBRATION_PROOF_MISMATCH,
        "learner evidence identity does not match calibration selection",
      )
    if not hmac.compare_digest(
      behavior_finalization_hash,
      behavior_finalization.sha256,
    ):
      _fail(
        ArtifactDiagnostic.BEHAVIOR_PROOF_MISMATCH,
        "behavior finalization hash does not match its JSON",
      )
    if not behavior_finalization.passed:
      _fail(
        ArtifactDiagnostic.GATE_FAILED,
        "behavior finalization did not pass every required gate",
      )
    selection_sha = behavior_finalization.behavior_selection_sha256
    if selection_sha is None or not hmac.compare_digest(
      selection_sha,
      _strict_sha256(
        payload["behaviorSelectionSha256"],
        "behaviorSelectionSha256",
      ),
    ):
      _fail(
        ArtifactDiagnostic.BEHAVIOR_PROOF_MISMATCH,
        "behavior selection identity does not match its finalization",
      )
    return cls(
      vehicle_profile=profile,
      calibration_profile=calibration_profile,
      controller_policy=policy,
      horizon_policy_sha256=horizon_policy_hash,
      runtime_vehicle_identity_sha256=_strict_sha256(
        payload["runtimeVehicleIdentitySha256"],
        "runtimeVehicleIdentitySha256",
      ),
      source_openpilot_commit=_strict_commit(
        payload["sourceOpenpilotCommit"],
        "sourceOpenpilotCommit",
      ),
      opendbc_commit=_strict_commit(
        payload["opendbcCommit"],
        "opendbcCommit",
      ),
      panda_commit=_strict_commit(
        payload["pandaCommit"],
        "pandaCommit",
      ),
      calibration_selection_manifest=calibration_manifest,
      behavior_finalization=behavior_finalization,
      replay_harness_commit=_strict_commit(
        payload["replayHarnessCommit"],
        "replayHarnessCommit",
      ),
      replay_passed=_strict_bool(
        payload["replayPassed"],
        "replayPassed",
      ),
      delivered_replay_passed=_strict_bool(
        payload["deliveredReplayPassed"],
        "deliveredReplayPassed",
      ),
      deterministic_aa_passed=_strict_bool(
        payload["deterministicAaPassed"],
        "deterministicAaPassed",
      ),
      device_acceptance_receipt=device_acceptance,
      smooth_passed=_strict_bool(
        payload["smoothPassed"],
        "smoothPassed",
      ),
      swift_passed=_strict_bool(
        payload["swiftPassed"],
        "swiftPassed",
      ),
      strong_passed=_strict_bool(
        payload["strongPassed"],
        "strongPassed",
      ),
      schema_version=payload["schemaVersion"],
    )


@dataclass(frozen=True, slots=True)
class ArtifactReadResult:
  artifact: ApprovedProfileArtifact | None
  diagnostic: ArtifactDiagnostic


class ApprovedArtifactReader:
  """Injectable Params reader that always fails back to no artifact."""

  def __init__(self, params: ParamsLike):
    self._params = params

  def read(
    self,
    *,
    expected_vehicle_identity: str,
    expected_runtime_vehicle_identity_sha256: str,
    expected_source_openpilot_commit: str,
    expected_opendbc_commit: str,
    expected_panda_commit: str,
  ) -> ArtifactReadResult:
    try:
      value = self._params.get(APPROVED_ARTIFACT_PARAM, block=False)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError):
      return ArtifactReadResult(
        artifact=None,
        diagnostic=ArtifactDiagnostic.PARAM_READ_ERROR,
      )
    if value is None:
      return ArtifactReadResult(
        artifact=None,
        diagnostic=ArtifactDiagnostic.ABSENT,
      )
    try:
      expected_runtime_hash = _strict_sha256(
        expected_runtime_vehicle_identity_sha256,
        "expected runtime vehicle identity",
      )
      expected_source = _strict_commit(
        expected_source_openpilot_commit,
        "expected source openpilot commit",
      )
      expected_opendbc = _strict_commit(
        expected_opendbc_commit,
        "expected opendbc commit",
      )
      expected_panda = _strict_commit(
        expected_panda_commit,
        "expected panda commit",
      )
      artifact = ApprovedProfileArtifact.from_param(
        value,
        expected_vehicle_identity=expected_vehicle_identity,
      )
      if not hmac.compare_digest(
        artifact.runtime_vehicle_identity_sha256,
        expected_runtime_hash,
      ):
        _fail(
          ArtifactDiagnostic.RUNTIME_VEHICLE_MISMATCH,
          "approved artifact belongs to another runtime vehicle",
        )
      if artifact.source_openpilot_commit != expected_source:
        _fail(
          ArtifactDiagnostic.SOURCE_COMMIT_MISMATCH,
          "approved artifact was gated against another openpilot commit",
        )
      if artifact.opendbc_commit != expected_opendbc:
        _fail(
          ArtifactDiagnostic.OPENDBC_COMMIT_MISMATCH,
          "approved artifact was gated against another opendbc commit",
        )
      if artifact.panda_commit != expected_panda:
        _fail(
          ArtifactDiagnostic.PANDA_COMMIT_MISMATCH,
          "approved artifact was gated against another panda commit",
        )
    except ArtifactValidationError as exc:
      return ArtifactReadResult(artifact=None, diagnostic=exc.reason)
    return ArtifactReadResult(
      artifact=None,
      diagnostic=(
        ArtifactDiagnostic.EXTERNAL_SAFETY_AUTHORITY_UNAVAILABLE
      ),
    )


@dataclass(frozen=True, slots=True)
class ProfileIdentity:
  artifact_sha256: str
  profile_sha256: str
  profile_revision: int

  def __post_init__(self) -> None:
    _strict_sha256(self.artifact_sha256, "artifactSha256")
    _strict_sha256(self.profile_sha256, "profileSha256")
    if type(self.profile_revision) is not int or self.profile_revision < 1:
      _fail(
        ArtifactDiagnostic.MALFORMED,
        "profileRevision must be a positive integer",
      )

  @classmethod
  def from_artifact(
    cls,
    artifact: ApprovedProfileArtifact,
  ) -> ProfileIdentity:
    return cls(
      artifact_sha256=artifact.artifact_sha256,
      profile_sha256=artifact.vehicle_profile_sha256,
      profile_revision=artifact.vehicle_profile.revision,
    )

  @classmethod
  def from_param(cls, value: object) -> ProfileIdentity:
    payload = _strict_object(value, _IDENTITY_KEYS, "profile identity")
    return cls(
      artifact_sha256=_strict_sha256(
        payload["artifactSha256"],
        "artifactSha256",
      ),
      profile_sha256=_strict_sha256(
        payload["profileSha256"],
        "profileSha256",
      ),
      profile_revision=payload["profileRevision"],
    )

  def to_param(self) -> dict[str, object]:
    return {
      "artifactSha256": self.artifact_sha256,
      "profileSha256": self.profile_sha256,
      "profileRevision": self.profile_revision,
    }


@dataclass(frozen=True, slots=True)
class _ActivationState:
  active_artifact: ApprovedProfileArtifact | None = None
  previous_artifact: ApprovedProfileArtifact | None = None
  staged_artifact: ApprovedProfileArtifact | None = None
  provisional: bool = False
  rollback_pending: bool = False
  rejected_profile_identities: tuple[ProfileIdentity, ...] = ()

  def to_param(self) -> dict[str, object]:
    current_identity = (
      None
      if self.active_artifact is None
      else ProfileIdentity.from_artifact(self.active_artifact).to_param()
    )
    return {
      "schemaVersion": ACTIVATION_STATE_SCHEMA_VERSION,
      "activeArtifact": (
        None
        if self.active_artifact is None
        else self.active_artifact.to_param()
      ),
      "activeProfileIdentity": current_identity,
      "previousArtifact": (
        None
        if self.previous_artifact is None
        else self.previous_artifact.to_param()
      ),
      "stagedArtifact": (
        None
        if self.staged_artifact is None
        else self.staged_artifact.to_param()
      ),
      "provisional": self.provisional,
      "rollbackPending": self.rollback_pending,
      "rejectedProfileIdentities": [
        identity.to_param()
        for identity in self.rejected_profile_identities
      ],
    }


@dataclass(frozen=True, slots=True)
class ApprovedEngagementDecision:
  selection: ControllerSelection
  artifact: ApprovedProfileArtifact | None
  provisional: bool


class PersistentProfileActivation:
  """Persisted profile selection prepared offroad and bound read-only live."""

  def __init__(
    self,
    params: ParamsLike,
    *,
    expected_vehicle_identity: str,
    expected_runtime_vehicle_identity_sha256: str,
    expected_source_openpilot_commit: str,
    expected_opendbc_commit: str,
    expected_panda_commit: str,
    production_envelope_verified: bool,
  ):
    self._params = params
    self._expected_vehicle_identity = expected_vehicle_identity
    self._expected_runtime_vehicle_identity_sha256 = (
      expected_runtime_vehicle_identity_sha256
    )
    self._expected_source_openpilot_commit = (
      expected_source_openpilot_commit
    )
    self._expected_opendbc_commit = expected_opendbc_commit
    self._expected_panda_commit = expected_panda_commit
    if type(production_envelope_verified) is not bool:
      raise TypeError("production envelope verification must be boolean")
    self._production_envelope_verified = production_envelope_verified
    self._engaged = False
    self._state = _ActivationState()
    self._state_valid = True
    self._state_stale_build = False
    self.diagnostic = ArtifactDiagnostic.ABSENT
    self._restore()

  @property
  def engaged(self) -> bool:
    return self._engaged

  @property
  def active_artifact(self) -> ApprovedProfileArtifact | None:
    return self._state.active_artifact

  @property
  def staged_artifact(self) -> ApprovedProfileArtifact | None:
    return self._state.staged_artifact

  @property
  def provisional(self) -> bool:
    return self._state.provisional

  @property
  def rollback_pending(self) -> bool:
    return self._state.rollback_pending

  @property
  def production_envelope_verified(self) -> bool:
    return self._production_envelope_verified

  @property
  def stale_build_state(self) -> bool:
    return self._state_stale_build

  @property
  def rejected_profile_identities(self) -> tuple[ProfileIdentity, ...]:
    return self._state.rejected_profile_identities

  @property
  def state_sha256(self) -> str | None:
    """Identity of the already-validated activation state for display caches."""
    if not self._state_valid:
      return None
    return _sha256_text(_canonical_json(self._state.to_param()))

  def _parse_artifact(
    self,
    value: object,
  ) -> ApprovedProfileArtifact | None:
    if value is None:
      return None
    artifact = self._parse_canonical_artifact(value)
    self._validate_current_artifact(artifact)
    return artifact

  @staticmethod
  def _parse_canonical_artifact(
    value: object,
  ) -> ApprovedProfileArtifact:
    # The embedded vehicle identity is needed to validate a canonical state
    # left by another car/build without pretending it belongs to this runtime.
    try:
      payload = _strict_object(
        value,
        _ARTIFACT_KEYS,
        "activation artifact",
      )
      encoded_profile = payload["vehicleProfileJson"]
      if type(encoded_profile) is not str:
        _fail(
          ArtifactDiagnostic.MALFORMED,
          "activation vehicleProfileJson must be text",
        )
      profile_payload = json.loads(encoded_profile)
      vehicle_identity = profile_payload["vehicle_identity"]
      if type(vehicle_identity) is not str or not vehicle_identity:
        _fail(
          ArtifactDiagnostic.MALFORMED,
          "activation vehicle identity must be nonempty text",
        )
      return ApprovedProfileArtifact.from_param(
        payload,
        expected_vehicle_identity=vehicle_identity,
      )
    except ArtifactValidationError:
      raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
      raise ArtifactValidationError(
        ArtifactDiagnostic.MALFORMED,
        "activation artifact is not canonical",
      ) from exc

  def _validate_current_artifact(
    self,
    artifact: ApprovedProfileArtifact,
  ) -> None:
    if (
      artifact.vehicle_profile.vehicle_identity
      != self._expected_vehicle_identity
    ):
      _fail(
        ArtifactDiagnostic.VEHICLE_MISMATCH,
        "activation artifact belongs to another vehicle",
      )
    if not hmac.compare_digest(
      artifact.runtime_vehicle_identity_sha256,
      self._expected_runtime_vehicle_identity_sha256,
    ):
      _fail(
        ArtifactDiagnostic.RUNTIME_VEHICLE_MISMATCH,
        "activation artifact runtime vehicle does not match",
      )
    if (
      artifact.source_openpilot_commit
      != self._expected_source_openpilot_commit
    ):
      _fail(
        ArtifactDiagnostic.SOURCE_COMMIT_MISMATCH,
        "activation artifact source commit does not match",
      )
    if artifact.opendbc_commit != self._expected_opendbc_commit:
      _fail(
        ArtifactDiagnostic.OPENDBC_COMMIT_MISMATCH,
        "activation artifact opendbc commit does not match",
      )
    if artifact.panda_commit != self._expected_panda_commit:
      _fail(
        ArtifactDiagnostic.PANDA_COMMIT_MISMATCH,
        "activation artifact panda commit does not match",
      )
    _fail(
      ArtifactDiagnostic.EXTERNAL_SAFETY_AUTHORITY_UNAVAILABLE,
      "external safety approval authority is unavailable",
    )

  def _state_from_param(self, value: object) -> _ActivationState:
    payload = _strict_object(value, _STATE_KEYS, "activation state")
    if (
      type(payload["schemaVersion"]) is not int
      or payload["schemaVersion"] != ACTIVATION_STATE_SCHEMA_VERSION
    ):
      _fail(
        ArtifactDiagnostic.STATE_INVALID,
        "activation state schema is incompatible",
      )
    provisional = _strict_bool(payload["provisional"], "provisional")
    rollback_pending = _strict_bool(
      payload["rollbackPending"],
      "rollbackPending",
    )
    rejected_payload = payload["rejectedProfileIdentities"]
    if type(rejected_payload) is not list:
      _fail(
        ArtifactDiagnostic.STATE_INVALID,
        "rejected profile identities must be a list",
      )
    try:
      rejected = tuple(
        ProfileIdentity.from_param(identity)
        for identity in rejected_payload
      )
      active = (
        None
        if payload["activeArtifact"] is None
        else self._parse_canonical_artifact(payload["activeArtifact"])
      )
      previous = (
        None
        if payload["previousArtifact"] is None
        else self._parse_canonical_artifact(payload["previousArtifact"])
      )
      staged = (
        None
        if payload["stagedArtifact"] is None
        else self._parse_canonical_artifact(payload["stagedArtifact"])
      )
    except ArtifactValidationError as exc:
      raise ArtifactValidationError(
        ArtifactDiagnostic.STATE_INVALID,
        f"activation state artifact is invalid: {exc}",
      ) from exc
    current_identity_payload = payload["activeProfileIdentity"]
    current_identity = (
      None
      if current_identity_payload is None
      else ProfileIdentity.from_param(current_identity_payload)
    )
    expected_current = (
      None if active is None else ProfileIdentity.from_artifact(active)
    )
    if current_identity != expected_current:
      _fail(
        ArtifactDiagnostic.STATE_INVALID,
        "active profile identity does not match active artifact",
      )
    if len(set(rejected)) != len(rejected):
      _fail(
        ArtifactDiagnostic.STATE_INVALID,
        "rejected profile identities must be unique",
      )
    if active is None and (provisional or rollback_pending):
      _fail(
        ArtifactDiagnostic.STATE_INVALID,
        "stock state cannot be provisional or pending rollback",
      )
    if rollback_pending and not provisional:
      _fail(
        ArtifactDiagnostic.STATE_INVALID,
        "rollback requires a provisional active profile",
      )
    if (
      active is not None
      and ProfileIdentity.from_artifact(active) in rejected
      and not rollback_pending
    ):
      _fail(
        ArtifactDiagnostic.STATE_INVALID,
        "rejected active profile must be pending rollback",
      )
    artifacts = tuple(
      artifact
      for artifact in (active, previous, staged)
      if artifact is not None
    )
    artifact_hashes = tuple(
      artifact.artifact_sha256
      for artifact in artifacts
    )
    if len(set(artifact_hashes)) != len(artifact_hashes):
      _fail(
        ArtifactDiagnostic.STATE_INVALID,
        "activation artifact roles must be distinct",
      )
    state = _ActivationState(
      active_artifact=active,
      previous_artifact=previous,
      staged_artifact=staged,
      provisional=provisional,
      rollback_pending=rollback_pending,
      rejected_profile_identities=rejected,
    )
    for artifact in artifacts:
      self._validate_current_artifact(artifact)
    return state

  def _restore(self) -> None:
    try:
      value = self._params.get(ACTIVATION_STATE_PARAM, block=False)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError):
      self._state_valid = False
      self.diagnostic = ArtifactDiagnostic.PARAM_READ_ERROR
      return
    if value is None:
      self.diagnostic = ArtifactDiagnostic.ABSENT
      return
    try:
      self._state = self._state_from_param(value)
    except ArtifactValidationError as exc:
      if exc.reason in (
        ArtifactDiagnostic.VEHICLE_MISMATCH,
        ArtifactDiagnostic.RUNTIME_VEHICLE_MISMATCH,
        ArtifactDiagnostic.SOURCE_COMMIT_MISMATCH,
        ArtifactDiagnostic.OPENDBC_COMMIT_MISMATCH,
        ArtifactDiagnostic.PANDA_COMMIT_MISMATCH,
      ):
        self._state_stale_build = True
        self._state_valid = False
        self._state = _ActivationState()
        self.diagnostic = ArtifactDiagnostic.STATE_STALE_BUILD
        return
      if (
        exc.reason
        is ArtifactDiagnostic.EXTERNAL_SAFETY_AUTHORITY_UNAVAILABLE
      ):
        self._state_valid = False
        self._state = _ActivationState()
        self.diagnostic = exc.reason
        return
      self._state_valid = False
      self._state = _ActivationState()
      self.diagnostic = ArtifactDiagnostic.STATE_INVALID
      return
    except (KeyError, TypeError, ValueError):
      self._state_valid = False
      self._state = _ActivationState()
      self.diagnostic = ArtifactDiagnostic.STATE_INVALID
      return
    self.diagnostic = ArtifactDiagnostic.OK

  def _persist(self, state: _ActivationState) -> None:
    if not self._state_valid:
      raise RuntimeError(
        "invalid persisted activation state must be repaired explicitly",
      )
    self._params.put(
      ACTIVATION_STATE_PARAM,
      state.to_param(),
      block=True,
    )
    self._state = state
    self._state_stale_build = False
    self.diagnostic = ArtifactDiagnostic.OK

  def retire_stale_build_offroad(self, *, offroad: bool) -> bool:
    """Atomically retire a canonical old-build state to prepared stock.

    This is deliberately unavailable for corruption. Only a state that fully
    passed its own canonical schema/hash/invariant checks and then failed a
    current vehicle/source/opendbc identity check can take this path.
    """
    if not offroad or self._engaged:
      raise RuntimeError(
        "stale activation state may be retired only while offroad",
      )
    if not self._state_stale_build:
      return False
    state = _ActivationState()
    self._params.put(
      ACTIVATION_STATE_PARAM,
      state.to_param(),
      block=True,
    )
    self._state = state
    self._state_valid = True
    self._state_stale_build = False
    self.diagnostic = ArtifactDiagnostic.OK
    return True

  def stage(
    self,
    artifact: ApprovedProfileArtifact,
    *,
    offroad: bool,
  ) -> None:
    if not offroad or self._engaged:
      raise RuntimeError(
        "approved artifacts may be staged only while offroad",
      )
    if not self._production_envelope_verified:
      raise RuntimeError(
        "approved artifact requires a verified production actuator envelope",
      )
    if not self._state_valid:
      raise RuntimeError(
        "invalid persisted activation state must be repaired explicitly",
      )
    reparsed = self._parse_artifact(artifact.to_param())
    if reparsed is None:
      raise AssertionError("approved artifact unexpectedly parsed as absent")
    identity = ProfileIdentity.from_artifact(reparsed)
    if identity in self._state.rejected_profile_identities:
      raise ValueError("rejected profile cannot be staged again")
    active = self._state.active_artifact
    if (
      active is not None
      and reparsed.vehicle_profile.revision
      <= active.vehicle_profile.revision
    ):
      raise ValueError("staged profile revision must advance")
    self._persist(_ActivationState(
      active_artifact=active,
      previous_artifact=self._state.previous_artifact,
      staged_artifact=reparsed,
      provisional=self._state.provisional,
      rollback_pending=self._state.rollback_pending,
      rejected_profile_identities=(
        self._state.rejected_profile_identities
      ),
    ))

  def prepare_offroad(self, *, offroad: bool) -> bool:
    """Atomically prepare one pending state transition while safely parked.

    Rollback has priority over promotion. Each successful call writes the
    complete activation state once through Params' atomic replacement. A
    caller may invoke this method again to prepare a separately-staged
    artifact after a rollback, but no call can perform both operations.
    """
    if not offroad or self._engaged:
      raise RuntimeError(
        "activation state may be prepared only while offroad",
      )
    if not self._state_valid:
      return False

    state = self._state
    if state.rollback_pending:
      self._persist(_ActivationState(
        active_artifact=state.previous_artifact,
        previous_artifact=None,
        staged_artifact=state.staged_artifact,
        provisional=False,
        rollback_pending=False,
        rejected_profile_identities=state.rejected_profile_identities,
      ))
      return True

    if state.staged_artifact is None:
      return False
    if not self._production_envelope_verified:
      return False
    # Never chain an unreviewed live trial on top of another provisional
    # profile. The staged artifact remains inert until explicit feedback
    # resolves the current trial.
    if state.active_artifact is not None and state.provisional:
      return False
    self._persist(_ActivationState(
      active_artifact=state.staged_artifact,
      previous_artifact=state.active_artifact,
      staged_artifact=None,
      provisional=True,
      rollback_pending=False,
      rejected_profile_identities=state.rejected_profile_identities,
    ))
    return True

  def begin_engagement(self) -> ApprovedEngagementDecision:
    """Bind the already-prepared state without persistence or mutation."""
    if self._engaged:
      raise RuntimeError("engagement already active")
    state = self._state
    self._engaged = True
    artifact = None
    if (
      self._state_valid
      and self._production_envelope_verified
      and not state.rollback_pending
    ):
      candidate = state.active_artifact
      if (
        candidate is not None
        and ProfileIdentity.from_artifact(candidate)
        not in state.rejected_profile_identities
      ):
        artifact = candidate
    return ApprovedEngagementDecision(
      selection=(
        ControllerSelection.MODULAR
        if artifact is not None
        else ControllerSelection.STOCK
      ),
      artifact=artifact,
      provisional=state.provisional if artifact is not None else False,
    )

  def end_engagement(self) -> None:
    if not self._engaged:
      raise RuntimeError("no active engagement")
    self._engaged = False

  def consume_feedback(self, *, offroad: bool) -> FeedbackChoice | None:
    """Consume explicit feedback for the exact provisional active profile."""
    if not offroad or self._engaged:
      return None
    active = self._state.active_artifact
    if (
      not self._state_valid
      or active is None
      or not self._state.provisional
    ):
      return None
    try:
      request = FeedbackRequest.from_param(
        self._params.get(FEEDBACK_REQUEST_PARAM, block=False),
      )
      response = FeedbackResponse.from_param(
        self._params.get(FEEDBACK_RESPONSE_PARAM, block=False),
      )
    except (
      KeyError,
      TypeError,
      ValueError,
      RuntimeError,
      OSError,
    ):
      return None
    identity = ProfileIdentity.from_artifact(active)
    if (
      not response.matches(request)
      or request.artifact_sha256 != identity.artifact_sha256
      or request.profile_sha256 != identity.profile_sha256
      or request.profile_revision != identity.profile_revision
      or response.artifact_sha256 != identity.artifact_sha256
      or response.profile_sha256 != identity.profile_sha256
      or response.profile_revision != identity.profile_revision
    ):
      return None

    if response.choice == FeedbackChoice.WORSE:
      rejected = self._state.rejected_profile_identities
      if identity not in rejected:
        rejected += (identity,)
      state = _ActivationState(
        active_artifact=active,
        previous_artifact=self._state.previous_artifact,
        staged_artifact=self._state.staged_artifact,
        provisional=True,
        rollback_pending=True,
        rejected_profile_identities=rejected,
      )
    elif response.choice in (
      FeedbackChoice.BETTER,
      FeedbackChoice.ABOUT_SAME,
    ):
      state = _ActivationState(
        active_artifact=active,
        previous_artifact=None,
        staged_artifact=self._state.staged_artifact,
        provisional=False,
        rollback_pending=False,
        rejected_profile_identities=(
          self._state.rejected_profile_identities
        ),
      )
    else:
      # NOT_SURE deliberately retains the exact provisional state.
      state = self._state
    if state != self._state:
      self._persist(state)
    # Remove the response first. If power is lost between these two atomic
    # removals, the still-present request can be presented again; the reverse
    # order could leave a stale response suppressing a future request for the
    # same provisional profile. NOT_SURE relies on this cleanup so another
    # completed drive can legitimately request fresh feedback.
    try:
      self._params.remove(FEEDBACK_RESPONSE_PARAM)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError):
      pass
    try:
      self._params.remove(FEEDBACK_REQUEST_PARAM)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError):
      pass
    return response.choice
