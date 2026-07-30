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
  PersistentProfileActivation,
  ProfileIdentity,
)
from openpilot.selfdrive.controls.lib.blatv2.bootstrap import (
  ControllerSelection,
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
  PhysicalParameters,
  ProfileNode,
  VehicleProfile,
)


VEHICLE = "approved-artifact-test-car"
RUNTIME_HASH = "1" * 64
SOURCE_COMMIT = "2" * 40
OPENDBC_COMMIT = "3" * 40
EVIDENCE_HASH = "4" * 64
HARNESS_COMMIT = "5" * 40


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


def artifact(revision: int = 1) -> ApprovedProfileArtifact:
  return ApprovedProfileArtifact(
    vehicle_profile=profile(revision),
    controller_policy=policy(),
    runtime_vehicle_identity_sha256=RUNTIME_HASH,
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
    learner_evidence_sha256=EVIDENCE_HASH,
    replay_harness_commit=HARNESS_COMMIT,
    replay_passed=True,
    delivered_replay_passed=True,
    safety_passed=True,
    deterministic_aa_passed=True,
    device_timing_passed=True,
  )


def reader_result(params: MemoryParams, **changes):
  expected = {
    "expected_vehicle_identity": VEHICLE,
    "expected_runtime_vehicle_identity_sha256": RUNTIME_HASH,
    "expected_source_openpilot_commit": SOURCE_COMMIT,
    "expected_opendbc_commit": OPENDBC_COMMIT,
  }
  expected.update(changes)
  return ApprovedArtifactReader(params).read(**expected)


def activation(
  params: MemoryParams,
  *,
  production_envelope_verified: bool = True,
  expected_source_openpilot_commit: str = SOURCE_COMMIT,
  expected_opendbc_commit: str = OPENDBC_COMMIT,
) -> PersistentProfileActivation:
  return PersistentProfileActivation(
    params,
    expected_vehicle_identity=VEHICLE,
    expected_runtime_vehicle_identity_sha256=RUNTIME_HASH,
    expected_source_openpilot_commit=expected_source_openpilot_commit,
    expected_opendbc_commit=expected_opendbc_commit,
    production_envelope_verified=production_envelope_verified,
  )


def set_feedback(
  params: MemoryParams,
  selected: ApprovedProfileArtifact,
  choice: FeedbackChoice,
  *,
  profile_sha256: str | None = None,
) -> None:
  request = FeedbackRequest(
    artifact_sha256=selected.artifact_sha256,
    profile_sha256=(
      selected.vehicle_profile_sha256
      if profile_sha256 is None
      else profile_sha256
    ),
    profile_revision=selected.vehicle_profile.revision,
  )
  params.values[FEEDBACK_REQUEST_PARAM] = request.to_param()
  params.values[FEEDBACK_RESPONSE_PARAM] = (
    FeedbackResponse.for_request(request, choice).to_param()
  )


class TestApprovedProfileArtifact(unittest.TestCase):
  def setUp(self):
    self.params = MemoryParams()
    self.approved = artifact()
    self.params.values[APPROVED_ARTIFACT_PARAM] = (
      self.approved.to_param()
    )

  def test_valid_artifact_round_trip_binds_every_identity(self):
    result = reader_result(self.params)
    self.assertIs(result.diagnostic, ArtifactDiagnostic.OK)
    self.assertEqual(result.artifact, self.approved)
    self.assertEqual(
      result.artifact.artifact_sha256,
      hashlib.sha256(
        json.dumps(
          self.approved.to_param(),
          sort_keys=True,
          separators=(",", ":"),
        ).encode(),
      ).hexdigest(),
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
    wrong_bool = self.approved.to_param()
    wrong_bool["safetyPassed"] = 1
    mutations.append(wrong_bool)
    upper_hash = self.approved.to_param()
    upper_hash["learnerEvidenceSha256"] = "A" * 64
    mutations.append(upper_hash)
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

  def test_vehicle_runtime_source_and_opendbc_mismatch_reasons(self):
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
      "safetyPassed",
      "deterministicAaPassed",
      "deviceTimingPassed",
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


class TestPersistentProfileActivation(unittest.TestCase):
  def setUp(self):
    self.params = MemoryParams()

  def _activate_provisional(
    self,
    candidate: ApprovedProfileArtifact | None = None,
  ) -> tuple[
    PersistentProfileActivation,
    ApprovedProfileArtifact,
  ]:
    selected = artifact() if candidate is None else candidate
    manager = activation(self.params)
    manager.stage(selected, offroad=True)
    self.assertTrue(manager.prepare_offroad(offroad=True))
    decision = manager.begin_engagement()
    self.assertIs(decision.selection, ControllerSelection.MODULAR)
    self.assertEqual(decision.artifact, selected)
    self.assertTrue(decision.provisional)
    manager.end_engagement()
    return manager, selected

  def test_empty_state_is_stock_and_stage_survives_restart(self):
    manager = activation(self.params)
    self.assertIs(
      manager.begin_engagement().selection,
      ControllerSelection.STOCK,
    )
    manager.end_engagement()
    selected = artifact()
    manager.stage(selected, offroad=True)
    restored = activation(self.params)
    self.assertEqual(restored.staged_artifact, selected)
    self.assertIs(
      restored.begin_engagement().selection,
      ControllerSelection.STOCK,
    )
    restored.end_engagement()
    self.assertTrue(restored.prepare_offroad(offroad=True))
    decision = restored.begin_engagement()
    self.assertEqual(decision.artifact, selected)

  def test_stage_and_feedback_cannot_switch_mid_engagement(self):
    manager = activation(self.params)
    first = artifact()
    manager.stage(first, offroad=True)
    manager.prepare_offroad(offroad=True)
    decision = manager.begin_engagement()
    self.assertEqual(decision.artifact, first)
    with self.assertRaisesRegex(RuntimeError, "offroad"):
      manager.stage(artifact(2), offroad=True)
    response = FeedbackResponse(
      first.artifact_sha256,
      first.vehicle_profile_sha256,
      first.vehicle_profile.revision,
      FeedbackChoice.WORSE,
    )
    self.params.values[FEEDBACK_RESPONSE_PARAM] = response.to_param()
    self.assertIsNone(manager.consume_feedback(offroad=True))
    self.assertFalse(manager.rollback_pending)

  def test_worse_rejects_exact_identity_then_rolls_back_at_boundary(self):
    manager, selected = self._activate_provisional()
    set_feedback(self.params, selected, FeedbackChoice.WORSE)
    self.assertIs(
      manager.consume_feedback(offroad=True),
      FeedbackChoice.WORSE,
    )
    identity = ProfileIdentity.from_artifact(selected)
    self.assertIn(identity, manager.rejected_profile_identities)
    self.assertTrue(manager.rollback_pending)
    self.assertEqual(manager.active_artifact, selected)

    restored = activation(self.params)
    self.assertTrue(restored.rollback_pending)
    put_count = len(self.params.puts)
    decision = restored.begin_engagement()
    self.assertIs(decision.selection, ControllerSelection.STOCK)
    self.assertIsNone(decision.artifact)
    self.assertEqual(len(self.params.puts), put_count)
    restored.end_engagement()
    self.assertTrue(restored.prepare_offroad(offroad=True))
    decision = restored.begin_engagement()
    self.assertIs(decision.selection, ControllerSelection.STOCK)
    restored.end_engagement()
    with self.assertRaisesRegex(ValueError, "rejected"):
      restored.stage(selected, offroad=True)

  def test_better_and_about_same_accept_exact_provisional_profile(self):
    for choice in (
      FeedbackChoice.BETTER,
      FeedbackChoice.ABOUT_SAME,
    ):
      with self.subTest(choice=choice):
        params = MemoryParams()
        manager = activation(params)
        selected = artifact()
        manager.stage(selected, offroad=True)
        manager.prepare_offroad(offroad=True)
        manager.begin_engagement()
        manager.end_engagement()
        set_feedback(params, selected, choice)
        self.assertIs(manager.consume_feedback(offroad=True), choice)
        self.assertFalse(manager.provisional)
        restored = activation(params)
        decision = restored.begin_engagement()
        self.assertEqual(decision.artifact, selected)
        self.assertFalse(decision.provisional)

  def test_not_sure_preserves_provisional_state(self):
    manager, selected = self._activate_provisional()
    set_feedback(self.params, selected, FeedbackChoice.NOT_SURE)
    put_count = len(self.params.puts)
    self.assertIs(
      manager.consume_feedback(offroad=True),
      FeedbackChoice.NOT_SURE,
    )
    self.assertTrue(manager.provisional)
    self.assertEqual(len(self.params.puts), put_count)
    self.assertNotIn(FEEDBACK_RESPONSE_PARAM, self.params.values)

  def test_feedback_must_match_hash_revision_and_be_offroad(self):
    manager, selected = self._activate_provisional()
    set_feedback(
      self.params,
      selected,
      FeedbackChoice.WORSE,
      profile_sha256="b" * 64,
    )
    self.assertIsNone(manager.consume_feedback(offroad=True))
    self.assertIsNone(manager.consume_feedback(offroad=False))
    self.assertFalse(manager.rollback_pending)

  def test_feedback_requires_the_exact_request_as_well_as_response(self):
    manager, selected = self._activate_provisional()
    self.params.values[FEEDBACK_RESPONSE_PARAM] = FeedbackResponse(
      selected.artifact_sha256,
      selected.vehicle_profile_sha256,
      selected.vehicle_profile.revision,
      FeedbackChoice.WORSE,
    ).to_param()
    self.assertIsNone(manager.consume_feedback(offroad=True))
    self.assertFalse(manager.rollback_pending)

  def test_same_profile_in_a_different_artifact_cannot_resolve_feedback(self):
    manager, selected = self._activate_provisional()
    different_wrapper = replace(
      selected,
      controller_policy=policy(revision=2),
    )
    self.assertEqual(
      different_wrapper.vehicle_profile_sha256,
      selected.vehicle_profile_sha256,
    )
    self.assertNotEqual(
      different_wrapper.artifact_sha256,
      selected.artifact_sha256,
    )
    set_feedback(
      self.params,
      different_wrapper,
      FeedbackChoice.BETTER,
    )
    self.assertIsNone(manager.consume_feedback(offroad=True))
    self.assertTrue(manager.provisional)

  def test_newer_artifact_switches_only_at_next_engagement(self):
    manager, first = self._activate_provisional()
    set_feedback(self.params, first, FeedbackChoice.BETTER)
    manager.consume_feedback(offroad=True)
    second = artifact(2)
    manager.stage(second, offroad=True)
    self.assertEqual(manager.active_artifact, first)
    decision = manager.begin_engagement()
    self.assertEqual(decision.artifact, first)
    manager.end_engagement()
    self.assertTrue(manager.prepare_offroad(offroad=True))
    decision = manager.begin_engagement()
    self.assertEqual(decision.artifact, second)
    self.assertTrue(decision.provisional)

  def test_begin_engagement_is_strictly_read_only(self):
    selected = artifact()
    manager = activation(self.params)
    manager.stage(selected, offroad=True)

    def forbid_put(*_args, **_kwargs):
      raise AssertionError("live engagement attempted a Params write")

    original_put = self.params.put
    self.params.put = forbid_put
    try:
      decision = manager.begin_engagement()
      self.assertIs(decision.selection, ControllerSelection.STOCK)
      manager.end_engagement()
    finally:
      self.params.put = original_put

    self.assertTrue(manager.prepare_offroad(offroad=True))
    self.params.put = forbid_put
    try:
      decision = manager.begin_engagement()
      self.assertIs(decision.selection, ControllerSelection.MODULAR)
      self.assertEqual(decision.artifact, selected)
      manager.end_engagement()
    finally:
      self.params.put = original_put

  def test_unverified_production_envelope_can_construct_but_only_selects_stock(self):
    selected = artifact()
    prepared = activation(self.params)
    prepared.stage(selected, offroad=True)
    prepared.prepare_offroad(offroad=True)

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

  def test_canonical_old_source_or_opendbc_state_retires_offroad(self):
    cases = (
      (
        replace(
          artifact(),
          source_openpilot_commit="6" * 40,
        ),
        {
          "expected_source_openpilot_commit": "6" * 40,
          "expected_opendbc_commit": OPENDBC_COMMIT,
        },
      ),
      (
        replace(
          artifact(),
          opendbc_commit="7" * 40,
        ),
        {
          "expected_source_openpilot_commit": SOURCE_COMMIT,
          "expected_opendbc_commit": "7" * 40,
        },
      ),
    )
    for old_artifact, old_expected in cases:
      with self.subTest(old_expected=old_expected):
        params = MemoryParams()
        old_manager = activation(params, **old_expected)
        old_manager.stage(old_artifact, offroad=True)
        old_manager.prepare_offroad(offroad=True)

        current = activation(params)
        self.assertTrue(current.stale_build_state)
        self.assertIs(
          current.diagnostic,
          ArtifactDiagnostic.STATE_STALE_BUILD,
        )
        put_count = len(params.puts)
        decision = current.begin_engagement()
        self.assertIs(decision.selection, ControllerSelection.STOCK)
        current.end_engagement()
        self.assertEqual(len(params.puts), put_count)
        with self.assertRaisesRegex(RuntimeError, "offroad"):
          current.retire_stale_build_offroad(offroad=False)
        self.assertTrue(
          current.retire_stale_build_offroad(offroad=True),
        )
        self.assertFalse(current.stale_build_state)
        current.stage(artifact(), offroad=True)
        current.prepare_offroad(offroad=True)
        self.assertEqual(current.active_artifact, artifact())

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
