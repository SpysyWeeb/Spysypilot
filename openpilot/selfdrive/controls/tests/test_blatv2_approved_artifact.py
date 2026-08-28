from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import unittest

from openpilot.selfdrive.controls.lib.blatv2.approved_artifact import (
  ACTIVATION_STATE_PARAM,
  APPROVED_ARTIFACT_PARAM,
  ApprovedArtifactReader,
  ApprovedProfileArtifact,
  ArtifactDiagnostic,
  ArtifactValidationError,
  CalibrationSelectionManifest,
  PersistentProfileActivation,
  ProfileIdentity,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  BEHAVIOR_FINALIZATION_SCHEMA_VERSION,
  BehaviorLearningFinalization,
  FinalizationReason,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import BehaviorPolicy
from openpilot.selfdrive.controls.lib.blatv2.bootstrap import (
  ControllerSelection,
)
from openpilot.selfdrive.controls.lib.blatv2.policy import ControllerPolicy
from openpilot.selfdrive.controls.tests.blatv2_artifact_test_helpers import (
  calibration_profile_for_controller,
  passing_device_acceptance_receipt,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  PhysicalParameters,
  ProfileNode,
  VehicleProfile,
)


VEHICLE = "approved-artifact-test-car"
RUNTIME_HASH = "1" * 64
SOURCE_COMMIT = "2" * 40
OPENDBC_COMMIT = "3" * 40
PANDA_COMMIT = "4" * 40
EVIDENCE_HASH = "4" * 64
HARNESS_COMMIT = "5" * 40
CALIBRATION_MANIFEST_HASH = "5" * 64
HORIZON_POLICY_HASH = "6" * 64


class MemoryParams:
  def __init__(self):
    self.values = {}
    self.puts = []
    self.removes = []
    self.raise_reads = False

  def get(self, key, block=False):
    del block
    if self.raise_reads:
      raise RuntimeError("read failed")
    return self.values.get(key)

  def put(self, key, value, block=False):
    self.values[key] = copy.deepcopy(value)
    self.puts.append((key, copy.deepcopy(value), block))

  def remove(self, key):
    self.values.pop(key, None)
    self.removes.append(key)


def profile(
  revision: int = 1,
  *,
  vehicle: str = VEHICLE,
  qualified: bool = True,
) -> VehicleProfile:
  parameters = PhysicalParameters(
    torque_per_lateral_accel=0.31,
    rack_gain_deg_s2_per_torque=1600.0,
    rack_damping_per_s=7.0,
    transport_delay_s=0.12,
    static_friction_torque=0.09,
    kinetic_friction_torque=0.03,
    rack_rate_resolution_deg_s=4.0,
    confidence=0.9,
    qualified=qualified,
    lateral_accel_offset_correction_mps2=-0.04,
  )
  return VehicleProfile(
    vehicle_identity=vehicle,
    revision=revision,
    provenance="qualified synthetic evidence",
    nodes=(
      ProfileNode(0.0, parameters, 180.0, 18000, 4000, 0.02),
      ProfileNode(30.0, parameters, 600.0, 60000, 12000, 0.03),
    ),
  )


def policy(
  revision: int = 1,
  *,
  provisional: bool = False,
) -> ControllerPolicy:
  return ControllerPolicy(
    revision=revision,
    provenance="replay-qualified response policy",
    provisional=provisional,
    natural_frequency_per_s=8.0,
    damping_ratio=1.0,
    observer_time_constant_s=None,
    observer_max_abs_disturbance_torque=None,
  )


def behavior_finalization(
  selected_policy: ControllerPolicy | None = None,
  *,
  smooth_passed: bool = True,
  swift_passed: bool = True,
  strong_passed: bool = True,
) -> BehaviorLearningFinalization:
  controller = policy() if selected_policy is None else selected_policy
  behavior = BehaviorPolicy.from_controller_policy(controller)
  gate_spec = "a" * 64
  route_partition = "b" * 64
  recorded_source = "c" * 64
  training = "d" * 64
  validation = "e" * 64
  selection = hashlib.sha256(json.dumps({
    "finalBehaviorPolicySha256": behavior.sha256,
    "gateSpecSha256": gate_spec,
    "recordedSourceIdentitySha256": recorded_source,
    "routePartitionSha256": route_partition,
    "trainingSelectionSha256": training,
    "validationSha256": validation,
  }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  passed = smooth_passed and swift_passed and strong_passed
  return BehaviorLearningFinalization(
    schema_version=BEHAVIOR_FINALIZATION_SCHEMA_VERSION,
    gate_spec_sha256=gate_spec,
    route_partition_sha256=route_partition,
    recorded_source_identity_sha256=recorded_source,
    training_selection_sha256=training,
    validation_sha256=validation,
    smooth_passed=smooth_passed,
    swift_passed=swift_passed,
    strong_passed=strong_passed,
    target_materially_improved=passed,
    final_behavior_policy=behavior if passed else None,
    final_behavior_policy_sha256=behavior.sha256 if passed else None,
    behavior_selection_sha256=selection if passed else None,
    reasons=(FinalizationReason.PASSED,) if passed else (
      FinalizationReason.SMOOTH_CROSS_FIT_REGRESSION,
    ),
  )


def calibration_manifest(selected_profile: VehicleProfile) -> CalibrationSelectionManifest:
  calibration = calibration_profile_for_controller(selected_profile)
  profile_sha = hashlib.sha256(selected_profile.to_json().encode()).hexdigest()
  calibration_sha = hashlib.sha256(calibration.to_json().encode()).hexdigest()
  return CalibrationSelectionManifest(
    selected_controller_profile_sha256=profile_sha,
    candidate_calibration_profile_sha256=calibration_sha,
    learner_evidence_sha256=EVIDENCE_HASH,
    qualification_manifest_sha256=CALIBRATION_MANIFEST_HASH,
    all_nodes_qualified=True,
  )


def artifact(revision: int = 1) -> ApprovedProfileArtifact:
  selected_profile = profile(revision)
  calibration = calibration_profile_for_controller(selected_profile)
  selected_policy = policy()
  profile_hash = hashlib.sha256(
    selected_profile.to_json().encode(),
  ).hexdigest()
  return ApprovedProfileArtifact(
    vehicle_profile=selected_profile,
    calibration_profile=calibration,
    controller_policy=selected_policy,
    horizon_policy_sha256=HORIZON_POLICY_HASH,
    runtime_vehicle_identity_sha256=RUNTIME_HASH,
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
    panda_commit=PANDA_COMMIT,
    calibration_selection_manifest=calibration_manifest(selected_profile),
    behavior_finalization=behavior_finalization(selected_policy),
    replay_harness_commit=HARNESS_COMMIT,
    replay_passed=True,
    delivered_replay_passed=True,
    deterministic_aa_passed=True,
    device_acceptance_receipt=passing_device_acceptance_receipt(
      vehicle_identity=selected_profile.vehicle_identity,
      runtime_identity_sha256=RUNTIME_HASH,
      profile_sha256=profile_hash,
      controller_policy_sha256=selected_policy.sha256,
      horizon_policy_sha256=HORIZON_POLICY_HASH,
      source_openpilot_commit=SOURCE_COMMIT,
      opendbc_commit=OPENDBC_COMMIT,
      panda_commit=PANDA_COMMIT,
    ),
    smooth_passed=True,
    swift_passed=True,
    strong_passed=True,
  )


def reader_result(params: MemoryParams, **changes):
  expected = {
    "expected_vehicle_identity": VEHICLE,
    "expected_runtime_vehicle_identity_sha256": RUNTIME_HASH,
    "expected_source_openpilot_commit": SOURCE_COMMIT,
    "expected_opendbc_commit": OPENDBC_COMMIT,
    "expected_panda_commit": PANDA_COMMIT,
  }
  expected.update(changes)
  return ApprovedArtifactReader(params).read(**expected)


def activation(
  params: MemoryParams,
  *,
  production_envelope_verified: bool = True,
  expected_source_openpilot_commit: str = SOURCE_COMMIT,
  expected_opendbc_commit: str = OPENDBC_COMMIT,
  expected_panda_commit: str = PANDA_COMMIT,
) -> PersistentProfileActivation:
  return PersistentProfileActivation(
    params,
    expected_vehicle_identity=VEHICLE,
    expected_runtime_vehicle_identity_sha256=RUNTIME_HASH,
    expected_source_openpilot_commit=expected_source_openpilot_commit,
    expected_opendbc_commit=expected_opendbc_commit,
    expected_panda_commit=expected_panda_commit,
    production_envelope_verified=production_envelope_verified,
  )


class TestApprovedProfileArtifact(unittest.TestCase):
  def setUp(self):
    self.params = MemoryParams()
    self.approved = artifact()
    self.params.values[APPROVED_ARTIFACT_PARAM] = (
      self.approved.to_param()
    )

  def test_candidate_round_trip_binds_identity_but_cannot_approve(self):
    parsed = ApprovedProfileArtifact.from_param(
      self.approved.to_param(),
      expected_vehicle_identity=VEHICLE,
    )
    self.assertEqual(parsed, self.approved)
    self.assertEqual(
      parsed.behavior_selection_sha256,
      self.approved.behavior_finalization.behavior_selection_sha256,
    )
    self.assertEqual(
      parsed.artifact_sha256,
      hashlib.sha256(
        json.dumps(
          self.approved.to_param(),
          sort_keys=True,
          separators=(",", ":"),
        ).encode(),
      ).hexdigest(),
    )

    result = reader_result(self.params)
    self.assertIsNone(result.artifact)
    self.assertIs(
      result.diagnostic,
      ArtifactDiagnostic.EXTERNAL_SAFETY_AUTHORITY_UNAVAILABLE,
    )

  def test_absent_and_read_error_fail_to_stock_without_exception(self):
    self.params.values.clear()
    result = reader_result(self.params)
    self.assertIsNone(result.artifact)
    self.assertIs(result.diagnostic, ArtifactDiagnostic.ABSENT)
    self.params.raise_reads = True
    result = reader_result(self.params)
    self.assertIsNone(result.artifact)
    self.assertIs(
      result.diagnostic,
      ArtifactDiagnostic.PARAM_READ_ERROR,
    )

  def test_exact_keys_types_lowercase_hashes_and_canonical_json(self):
    mutations = []
    missing = self.approved.to_param()
    missing.pop("replayHarnessCommit")
    mutations.append(missing)
    extra = self.approved.to_param()
    extra["automaticApproval"] = True
    mutations.append(extra)
    retired_safety_bool = self.approved.to_param()
    retired_safety_bool["externalSafetyEvidencePassed"] = True
    mutations.append(retired_safety_bool)
    upper_hash = self.approved.to_param()
    upper_hash["learnerEvidenceSha256"] = "A" * 64
    mutations.append(upper_hash)
    malformed_behavior_hash = self.approved.to_param()
    malformed_behavior_hash["behaviorSelectionSha256"] = "6" * 63
    mutations.append(malformed_behavior_hash)
    abbreviated_commit = self.approved.to_param()
    abbreviated_commit["sourceOpenpilotCommit"] = SOURCE_COMMIT[:12]
    mutations.append(abbreviated_commit)
    noncanonical_profile = self.approved.to_param()
    noncanonical_profile["vehicleProfileJson"] = json.dumps(
      json.loads(noncanonical_profile["vehicleProfileJson"]),
    )
    noncanonical_profile["vehicleProfileSha256"] = hashlib.sha256(
      noncanonical_profile["vehicleProfileJson"].encode(),
    ).hexdigest()
    mutations.append(noncanonical_profile)

    for payload in mutations:
      with self.subTest(payload=payload):
        self.params.values[APPROVED_ARTIFACT_PARAM] = payload
        result = reader_result(self.params)
        self.assertIsNone(result.artifact)
        self.assertIs(result.diagnostic, ArtifactDiagnostic.MALFORMED)

  def test_profile_and_policy_hash_mismatches_are_distinct(self):
    payload = self.approved.to_param()
    payload["vehicleProfileSha256"] = "6" * 64
    self.params.values[APPROVED_ARTIFACT_PARAM] = payload
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.PROFILE_HASH_MISMATCH,
    )

    payload = self.approved.to_param()
    payload["controllerPolicySha256"] = "7" * 64
    self.params.values[APPROVED_ARTIFACT_PARAM] = payload
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.POLICY_HASH_MISMATCH,
    )

    payload = self.approved.to_param()
    payload["calibrationProfileSha256"] = "8" * 64
    self.params.values[APPROVED_ARTIFACT_PARAM] = payload
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.CALIBRATION_PROOF_MISMATCH,
    )

  def test_calibration_json_tampering_and_dropped_offset_fail_closed(self):
    payload = self.approved.to_param()
    calibration = json.loads(payload["calibrationProfileJson"])
    calibration["nodes"][0]["parameters"][
      "lateral_accel_offset_correction_mps2"
    ] = 0.125
    payload["calibrationProfileJson"] = json.dumps(
      calibration,
      sort_keys=True,
      separators=(",", ":"),
    )
    self.params.values[APPROVED_ARTIFACT_PARAM] = payload
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.CALIBRATION_PROOF_MISMATCH,
    )

    # Recomputing the sibling hash is insufficient: the selection preimage
    # still names the exact qualified calibration that was gated.
    payload["calibrationProfileSha256"] = hashlib.sha256(
      payload["calibrationProfileJson"].encode(),
    ).hexdigest()
    self.params.values[APPROVED_ARTIFACT_PARAM] = payload
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.CALIBRATION_PROOF_MISMATCH,
    )

    dropped = self.approved.to_param()
    calibration = json.loads(dropped["calibrationProfileJson"])
    calibration["nodes"][0]["parameters"].pop(
      "lateral_accel_offset_correction_mps2",
    )
    dropped["calibrationProfileJson"] = json.dumps(
      calibration,
      sort_keys=True,
      separators=(",", ":"),
    )
    dropped["calibrationProfileSha256"] = hashlib.sha256(
      dropped["calibrationProfileJson"].encode(),
    ).hexdigest()
    self.params.values[APPROVED_ARTIFACT_PARAM] = dropped
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.MALFORMED,
    )

  def test_controller_profile_cannot_drop_learned_offset(self):
    payload = self.approved.to_param()
    controller = json.loads(payload["vehicleProfileJson"])
    controller["nodes"][0]["parameters"][
      "lateral_accel_offset_correction_mps2"
    ] = 0.25
    payload["vehicleProfileJson"] = json.dumps(
      controller,
      sort_keys=True,
      separators=(",", ":"),
    )
    payload["vehicleProfileSha256"] = hashlib.sha256(
      payload["vehicleProfileJson"].encode(),
    ).hexdigest()
    selection = json.loads(payload["calibrationSelectionManifestJson"])
    selection["selectedControllerProfileSha256"] = payload[
      "vehicleProfileSha256"
    ]
    payload["calibrationSelectionManifestJson"] = json.dumps(
      selection,
      sort_keys=True,
      separators=(",", ":"),
    )
    payload["calibrationSelectionManifestSha256"] = hashlib.sha256(
      payload["calibrationSelectionManifestJson"].encode(),
    ).hexdigest()
    self.params.values[APPROVED_ARTIFACT_PARAM] = payload
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.CALIBRATION_PROOF_MISMATCH,
    )

  def test_unqualified_profile_and_provisional_policy_are_rejected(self):
    payload = self.approved.to_param()
    unqualified_json = profile(qualified=False).to_json()
    payload["vehicleProfileJson"] = unqualified_json
    payload["vehicleProfileSha256"] = hashlib.sha256(
      unqualified_json.encode(),
    ).hexdigest()
    self.params.values[APPROVED_ARTIFACT_PARAM] = payload
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.UNQUALIFIED_PROFILE,
    )

    payload = self.approved.to_param()
    provisional_json = policy(provisional=True).to_json()
    payload["controllerPolicyJson"] = provisional_json
    payload["controllerPolicySha256"] = hashlib.sha256(
      provisional_json.encode(),
    ).hexdigest()
    self.params.values[APPROVED_ARTIFACT_PARAM] = payload
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.PROVISIONAL_POLICY,
    )

  def test_vehicle_runtime_source_opendbc_and_panda_mismatch_reasons(self):
    cases = (
      (
        {"expected_vehicle_identity": "different-car"},
        ArtifactDiagnostic.VEHICLE_MISMATCH,
      ),
      (
        {"expected_runtime_vehicle_identity_sha256": "8" * 64},
        ArtifactDiagnostic.RUNTIME_VEHICLE_MISMATCH,
      ),
      (
        {"expected_source_openpilot_commit": "9" * 40},
        ArtifactDiagnostic.SOURCE_COMMIT_MISMATCH,
      ),
      (
        {"expected_opendbc_commit": "a" * 40},
        ArtifactDiagnostic.OPENDBC_COMMIT_MISMATCH,
      ),
      (
        {"expected_panda_commit": "b" * 40},
        ArtifactDiagnostic.PANDA_COMMIT_MISMATCH,
      ),
    )
    for changes, expected in cases:
      with self.subTest(changes=changes):
        result = reader_result(self.params, **changes)
        self.assertIsNone(result.artifact)
        self.assertIs(result.diagnostic, expected)

  def test_every_gate_is_explicit_and_false_is_never_approved(self):
    for key in (
      "replayPassed",
      "deliveredReplayPassed",
      "deterministicAaPassed",
      "smoothPassed",
      "swiftPassed",
      "strongPassed",
    ):
      with self.subTest(key=key):
        payload = self.approved.to_param()
        payload[key] = False
        self.params.values[APPROVED_ARTIFACT_PARAM] = payload
        result = reader_result(self.params)
        self.assertIsNone(result.artifact)
        self.assertIs(
          result.diagnostic,
          ArtifactDiagnostic.GATE_FAILED,
        )

      with self.subTest(key=key, case="omitted"):
        payload = self.approved.to_param()
        payload.pop(key)
        self.params.values[APPROVED_ARTIFACT_PARAM] = payload
        result = reader_result(self.params)
        self.assertIsNone(result.artifact)
        self.assertIs(
          result.diagnostic,
          ArtifactDiagnostic.MALFORMED,
        )

      with self.subTest(key=key, case="non_boolean"):
        payload = self.approved.to_param()
        payload[key] = 1
        self.params.values[APPROVED_ARTIFACT_PARAM] = payload
        result = reader_result(self.params)
        self.assertIsNone(result.artifact)
        self.assertIs(
          result.diagnostic,
          ArtifactDiagnostic.MALFORMED,
        )

  def test_device_receipt_is_hash_and_identity_bound_without_boolean_escape(self):
    forged = self.approved.to_param()
    forged["deviceTimingPassed"] = True
    self.params.values[APPROVED_ARTIFACT_PARAM] = forged
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.MALFORMED,
    )

    tampered = self.approved.to_param()
    decoded = json.loads(tampered["deviceAcceptanceReceiptJson"])
    decoded["sampleCount"] = 2
    tampered["deviceAcceptanceReceiptJson"] = json.dumps(
      decoded,
      sort_keys=True,
      separators=(",", ":"),
    )
    self.params.values[APPROVED_ARTIFACT_PARAM] = tampered
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.DEVICE_ACCEPTANCE_PROOF_MISMATCH,
    )

    mismatched = self.approved.to_param()
    receipt = replace(
      self.approved.device_acceptance_receipt,
      horizon_policy_sha256="7" * 64,
    )
    mismatched["deviceAcceptanceReceiptJson"] = receipt.to_json()
    mismatched["deviceAcceptanceReceiptSha256"] = receipt.sha256
    self.params.values[APPROVED_ARTIFACT_PARAM] = mismatched
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.DEVICE_ACCEPTANCE_PROOF_MISMATCH,
    )

  def test_old_schema_artifact_fails_closed_in_reader_and_activation(self):
    old_payload = self.approved.to_param()
    old_payload["schemaVersion"] = 3
    for key in (
      "pandaCommit",
      "calibrationSelectionManifestJson",
      "calibrationSelectionManifestSha256",
      "behaviorFinalizationJson",
      "behaviorFinalizationSha256",
    ):
      old_payload.pop(key)

    self.params.values[APPROVED_ARTIFACT_PARAM] = old_payload
    result = reader_result(self.params)
    self.assertIsNone(result.artifact)
    self.assertIs(result.diagnostic, ArtifactDiagnostic.MALFORMED)

    old_calibration = self.approved.to_param()
    calibration = json.loads(old_calibration["calibrationProfileJson"])
    calibration["schema_version"] = 1
    old_calibration["calibrationProfileJson"] = json.dumps(
      calibration,
      sort_keys=True,
      separators=(",", ":"),
    )
    old_calibration["calibrationProfileSha256"] = hashlib.sha256(
      old_calibration["calibrationProfileJson"].encode(),
    ).hexdigest()
    self.params.values[APPROVED_ARTIFACT_PARAM] = old_calibration
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.MALFORMED,
    )

    old_selection = self.approved.to_param()
    selection = json.loads(old_selection["calibrationSelectionManifestJson"])
    selection["schemaVersion"] = 1
    old_selection["calibrationSelectionManifestJson"] = json.dumps(
      selection,
      sort_keys=True,
      separators=(",", ":"),
    )
    old_selection["calibrationSelectionManifestSha256"] = hashlib.sha256(
      old_selection["calibrationSelectionManifestJson"].encode(),
    ).hexdigest()
    self.params.values[APPROVED_ARTIFACT_PARAM] = old_selection
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.MALFORMED,
    )

    state_params = MemoryParams()
    persisted = {
      "schemaVersion": 1,
      "activeArtifact": None,
      "activeProfileIdentity": None,
      "previousArtifact": None,
      "stagedArtifact": self.approved.to_param(),
      "provisional": False,
      "rollbackPending": False,
      "rejectedProfileIdentities": [],
    }
    state_params.values[ACTIVATION_STATE_PARAM] = persisted
    persisted["stagedArtifact"]["schemaVersion"] = 3
    for key in (
      "pandaCommit",
      "calibrationSelectionManifestJson",
      "calibrationSelectionManifestSha256",
      "behaviorFinalizationJson",
      "behaviorFinalizationSha256",
    ):
      persisted["stagedArtifact"].pop(key)
    restored = activation(state_params)
    self.assertIs(restored.diagnostic, ArtifactDiagnostic.STATE_INVALID)
    self.assertIs(
      restored.begin_engagement().selection,
      ControllerSelection.STOCK,
    )
    restored.end_engagement()

    current = self.approved.to_param()
    current["schemaVersion"] = 3
    self.params.values[APPROVED_ARTIFACT_PARAM] = current
    result = reader_result(self.params)
    self.assertIsNone(result.artifact)
    self.assertIs(result.diagnostic, ArtifactDiagnostic.MALFORMED)

  def test_behavior_selection_identity_is_derived_from_canonical_proof(self):
    for mutation in (None, "", "A" * 64, "7" * 63, "7" * 64):
      with self.subTest(mutation=mutation):
        payload = self.approved.to_param()
        if mutation is None:
          payload.pop("behaviorSelectionSha256")
        else:
          payload["behaviorSelectionSha256"] = mutation
        self.params.values[APPROVED_ARTIFACT_PARAM] = payload
        result = reader_result(self.params)
        self.assertIsNone(result.artifact)
        self.assertIn(result.diagnostic, (
          ArtifactDiagnostic.MALFORMED,
          ArtifactDiagnostic.BEHAVIOR_PROOF_MISMATCH,
        ))

  def test_behavior_proof_cannot_be_paired_with_an_unrelated_policy(self):
    payload = self.approved.to_param()
    other_policy = ControllerPolicy(
      revision=2,
      provenance="different replay-qualified response policy",
      provisional=False,
      natural_frequency_per_s=12.0,
      damping_ratio=0.7,
      observer_time_constant_s=None,
      observer_max_abs_disturbance_torque=None,
    )
    payload["controllerPolicyJson"] = other_policy.to_json()
    payload["controllerPolicySha256"] = other_policy.sha256
    self.params.values[APPROVED_ARTIFACT_PARAM] = payload
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.BEHAVIOR_PROOF_MISMATCH,
    )

  def test_modified_behavior_proof_and_false_proof_gate_fail_closed(self):
    payload = self.approved.to_param()
    decoded = json.loads(payload["behaviorFinalizationJson"])
    decoded["validationSha256"] = "f" * 64
    payload["behaviorFinalizationJson"] = json.dumps(
      decoded,
      sort_keys=True,
      separators=(",", ":"),
    )
    self.params.values[APPROVED_ARTIFACT_PARAM] = payload
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.MALFORMED,
    )

    failed = behavior_finalization(smooth_passed=False)
    payload = self.approved.to_param()
    payload["behaviorFinalizationJson"] = failed.to_json()
    payload["behaviorFinalizationSha256"] = failed.sha256
    payload["behaviorSelectionSha256"] = "7" * 64
    self.params.values[APPROVED_ARTIFACT_PARAM] = payload
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.GATE_FAILED,
    )

  def test_calibration_manifest_is_hash_and_profile_bound(self):
    payload = self.approved.to_param()
    payload["calibrationSelectionManifestSha256"] = "7" * 64
    self.params.values[APPROVED_ARTIFACT_PARAM] = payload
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.CALIBRATION_PROOF_MISMATCH,
    )

    payload = self.approved.to_param()
    decoded = json.loads(payload["calibrationSelectionManifestJson"])
    decoded["selectedControllerProfileSha256"] = "8" * 64
    changed = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    payload["calibrationSelectionManifestJson"] = changed
    payload["calibrationSelectionManifestSha256"] = hashlib.sha256(
      changed.encode(),
    ).hexdigest()
    self.params.values[APPROVED_ARTIFACT_PARAM] = payload
    self.assertIs(
      reader_result(self.params).diagnostic,
      ArtifactDiagnostic.CALIBRATION_PROOF_MISMATCH,
    )


class TestPersistentProfileActivationAuthorityWall(unittest.TestCase):
  def test_stage_and_preseeded_state_cannot_bypass_missing_authority(self):
    params = MemoryParams()
    selected = artifact()
    manager = activation(params)
    with self.assertRaises(ArtifactValidationError) as raised:
      manager.stage(selected, offroad=True)
    self.assertIs(
      raised.exception.reason,
      ArtifactDiagnostic.EXTERNAL_SAFETY_AUTHORITY_UNAVAILABLE,
    )
    self.assertEqual(params.puts, [])
    self.assertFalse(manager.prepare_offroad(offroad=True))
    decision = manager.begin_engagement()
    self.assertIs(decision.selection, ControllerSelection.STOCK)
    self.assertIsNone(decision.artifact)
    manager.end_engagement()

    identity = ProfileIdentity.from_artifact(selected)
    params.values[ACTIVATION_STATE_PARAM] = {
      "schemaVersion": 1,
      "activeArtifact": selected.to_param(),
      "activeProfileIdentity": identity.to_param(),
      "previousArtifact": None,
      "stagedArtifact": None,
      "provisional": False,
      "rollbackPending": False,
      "rejectedProfileIdentities": [],
    }
    restored = activation(params)
    self.assertIs(
      restored.diagnostic,
      ArtifactDiagnostic.EXTERNAL_SAFETY_AUTHORITY_UNAVAILABLE,
    )
    self.assertIsNone(restored.active_artifact)
    self.assertFalse(restored.prepare_offroad(offroad=True))
    decision = restored.begin_engagement()
    self.assertIs(decision.selection, ControllerSelection.STOCK)
    self.assertIsNone(decision.artifact)


class TestPersistentProfileActivationGuards(unittest.TestCase):
  def setUp(self):
    self.params = MemoryParams()

  def test_unverified_production_envelope_can_construct_but_only_selects_stock(self):
    unverified = activation(
      self.params,
      production_envelope_verified=False,
    )
    put_count = len(self.params.puts)
    decision = unverified.begin_engagement()
    self.assertIs(decision.selection, ControllerSelection.STOCK)
    self.assertIsNone(decision.artifact)
    unverified.end_engagement()
    self.assertEqual(len(self.params.puts), put_count)
    with self.assertRaisesRegex(RuntimeError, "verified production"):
      unverified.stage(artifact(2), offroad=True)

  def test_malformed_persisted_state_stays_stock_and_is_not_overwritten(self):
    corrupt = {"schemaVersion": 1}
    self.params.values[ACTIVATION_STATE_PARAM] = corrupt
    manager = activation(self.params)
    self.assertIs(
      manager.diagnostic,
      ArtifactDiagnostic.STATE_INVALID,
    )
    decision = manager.begin_engagement()
    self.assertIs(decision.selection, ControllerSelection.STOCK)
    manager.end_engagement()
    self.assertFalse(manager.stale_build_state)
    self.assertFalse(manager.retire_stale_build_offroad(offroad=True))
    with self.assertRaisesRegex(RuntimeError, "repaired"):
      manager.stage(artifact(), offroad=True)
    self.assertEqual(
      self.params.values[ACTIVATION_STATE_PARAM],
      corrupt,
    )

  def test_persisted_state_invariants_fail_closed(self):
    selected = artifact()
    base = {
      "schemaVersion": 1,
      "activeArtifact": selected.to_param(),
      "activeProfileIdentity": ProfileIdentity.from_artifact(
        selected,
      ).to_param(),
      "previousArtifact": None,
      "stagedArtifact": None,
      "provisional": False,
      "rollbackPending": False,
      "rejectedProfileIdentities": [],
    }
    cases = {
      "identity": {
        "activeProfileIdentity": ProfileIdentity.from_artifact(
          artifact(2),
        ).to_param(),
      },
      "rollback": {"rollbackPending": True},
      "duplicate role": {"stagedArtifact": selected.to_param()},
    }
    for name, mutation in cases.items():
      with self.subTest(name=name):
        persisted = copy.deepcopy(base)
        persisted.update(mutation)
        params = MemoryParams()
        params.values[ACTIVATION_STATE_PARAM] = persisted
        manager = activation(params)
        self.assertIs(manager.diagnostic, ArtifactDiagnostic.STATE_INVALID)
        self.assertIsNone(manager.active_artifact)
        self.assertFalse(manager.prepare_offroad(offroad=True))
        self.assertIs(
          manager.begin_engagement().selection,
          ControllerSelection.STOCK,
        )
        self.assertEqual(params.values[ACTIVATION_STATE_PARAM], persisted)

  def test_activation_params_are_persistent_json_without_clear_flags(self):
    root = Path(__file__).resolve().parents[3]
    keys = (root / "common" / "params_keys.h").read_text()
    self.assertIn(
      '{"BLaTv2ApprovedArtifact", {PERSISTENT, JSON}}',
      keys,
    )
    self.assertIn(
      '{"BLaTv2ActivationState", {PERSISTENT, JSON}}',
      keys,
    )


if __name__ == "__main__":
  unittest.main()
