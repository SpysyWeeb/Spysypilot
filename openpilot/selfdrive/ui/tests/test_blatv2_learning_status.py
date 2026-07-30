from copy import deepcopy
import unittest

from openpilot.selfdrive.ui.widgets.blatv2_learning_status import (
  LearningStatusError,
  cycle_page_index,
  format_duration,
  format_speed,
  grid_cells,
  learning_panel_presentation,
  operation_presentation,
  parse_learning_operation_status,
  parse_learning_status,
  parse_lifecycle_status,
  reason_label,
  select_value_provider,
  validate_operation_update,
)


VEHICLE = "HYUNDAI PALISADE 2020"
RUNTIME_HASH = "1" * 64
HASH = "2" * 64
COMMIT = "3" * 40
NOW_MONO_NS = 10_000_000_000
OPERATION_ID = "d" * 32
ROUTE_HASH = "e" * 64


def node_fixture(
  index: int,
  *,
  clean_support_s: float = 60.0,
  minimum_support_s: float = 150.0,
  qualified: bool = False,
  reasons: list[str] | None = None,
  last_drive_complete: bool = True,
) -> dict:
  return {
    "node_index": index,
    "speed_mps": float((0, 5, 10, 15, 20, 30)[index]),
    "minimum_support_s": minimum_support_s,
    "clean_support_s": clean_support_s,
    "last_drive_clean_support_s": 12.5 if last_drive_complete else None,
    "supported_sample_count": 6000,
    "last_drive_accepted_sample_count": 1250 if last_drive_complete else None,
    "training_count": 4800,
    "validation_count": 1200,
    "validation_support_s": 24.0,
    "minimum_validation_support_s": minimum_support_s * 0.2,
    "lateral_accel_span_mps2": 0.9,
    "lateral_accel_rms_mps2": 0.3,
    "rack_travel_deg": 200.0,
    "applied_torque_span": 0.2,
    "rack_reversals": 6,
    "seed_validation_rms": 0.1,
    "candidate_validation_rms": 0.08,
    "confidence": 0.8,
    "qualified": qualified,
    "reasons": (
      ["qualified"]
      if qualified
      else ["insufficient_support"]
      if reasons is None
      else reasons
    ),
    "candidate_parameters": (
      {
        "torque_per_lateral_accel": 0.3,
        "rack_gain_deg_s2_per_torque": 4000.0,
        "rack_damping_per_s": 10.0,
        "kinetic_friction_torque": 0.03,
      }
      if qualified
      else None
    ),
  }


def learning_fixture(*, last_drive_complete: bool = True) -> dict:
  nodes = [
    node_fixture(index, last_drive_complete=last_drive_complete)
    for index in range(6)
  ]
  # A full time bar must not be confused with node qualification.
  nodes[0] = node_fixture(
    0,
    clean_support_s=150.0,
    minimum_support_s=150.0,
    reasons=["insufficient_excitation"],
    last_drive_complete=last_drive_complete,
  )
  nodes[1] = node_fixture(
    1,
    clean_support_s=150.0,
    minimum_support_s=150.0,
    qualified=True,
    last_drive_complete=last_drive_complete,
  )
  return {
    "schema_version": 1,
    "informational_only": True,
    "vehicle_identity": VEHICLE,
    "runtime_identity_sha256": RUNTIME_HASH,
    "seed_profile_sha256": HASH,
    "evidence_sha256": "4" * 64,
    "manifest_sha256": "5" * 64,
    "all_nodes_qualified": False,
    "candidate_profile_sha256": None,
    "candidate_profile_revision": None,
    "last_drive_complete": last_drive_complete,
    "nodes": nodes,
  }


def lifecycle_fixture(
  state: str = "stock",
  *,
  runtime_hash: str = RUNTIME_HASH,
) -> dict:
  active = (
    {
      "artifact_sha256": "6" * 64,
      "profile_sha256": "7" * 64,
      "profile_revision": 42,
    }
    if state in ("provisional", "approved", "rollback_pending")
    else None
  )
  staged = (
    {
      "artifact_sha256": "8" * 64,
      "profile_sha256": "9" * 64,
      "profile_revision": 43,
    }
    if state == "staged"
    else None
  )
  return {
    "schema_version": 1,
    "informational_only": True,
    "vehicle_identity": VEHICLE,
    "runtime_identity_sha256": runtime_hash,
    "source_openpilot_commit": COMMIT,
    "opendbc_commit": "a" * 40,
    "activation_state_sha256": "b" * 64,
    "diagnostic": "absent" if state == "stock" else "ok",
    "controller_state": state,
    "effective_controller": (
      "modular" if state in ("provisional", "approved") else "stock"
    ),
    "production_envelope_verified": True,
    "active_profile": active,
    "staged_profile": staged,
    "rejected_profile_count": 0,
  }


_OPERATION_DIAGNOSTICS = {
  "preparing": "waiting_for_car_params",
  "ready_no_evidence": "ready_for_first_drive",
  "collecting": "collecting_current_drive",
  "finalizing": "finalizing_drive",
  "retry_pending": "persist_retry_pending",
  "backfilling": "replaying_route",
  "idle": "evidence_ready",
  "drive_skipped_identity_mismatch": "car_params_identity_mismatch",
  "failed": "unexpected_error",
}
_TERMINAL_OPERATION_STATES = {
  "ready_no_evidence",
  "idle",
  "drive_skipped_identity_mismatch",
  "failed",
}


def operation_fixture(
  state: str = "idle",
  *,
  diagnostic: str | None = None,
) -> dict:
  replaying = state == "backfilling" and (diagnostic is None or diagnostic == "replaying_route")
  collecting = state == "collecting"
  finalizing = state == "finalizing"
  idle = state == "idle"
  no_evidence = state == "ready_no_evidence"
  return {
    "schema_version": 1,
    "informational_only": True,
    "state": state,
    "diagnostic": (_OPERATION_DIAGNOSTICS[state] if diagnostic is None else diagnostic),
    "operation_id": OPERATION_ID,
    "sequence": 7,
    "started_mono_ns": NOW_MONO_NS - 2_000_000_000,
    "updated_mono_ns": NOW_MONO_NS - 100_000_000,
    "terminal": state in _TERMINAL_OPERATION_STATES,
    "vehicle_identity": VEHICLE,
    "runtime_identity_sha256": RUNTIME_HASH,
    "current_route_identity": ROUTE_HASH if replaying or collecting else None,
    "current_route_index": 2 if replaying else None,
    "total_route_count": 5 if replaying else None,
    "last_route_identity": ROUTE_HASH if finalizing or idle else None,
    "accepted_sample_count": 0 if no_evidence else 12345,
    "rejected_sample_count": 0 if no_evidence else 678,
    "retry_count": 2 if state == "retry_pending" else 0,
    "evidence_sha256": HASH if idle else None,
    "ledger_sha256": None,
  }


class TestLearningStatusParser(unittest.TestCase):
  def test_six_node_fixture_and_full_time_blocked_node(self) -> None:
    status = parse_learning_status(
      learning_fixture(),
      expected_vehicle_identity=VEHICLE,
    )
    self.assertEqual(len(status.nodes), 6)
    self.assertEqual(status.nodes[0].support_fraction, 1.0)
    self.assertFalse(status.nodes[0].qualified)
    self.assertEqual(status.nodes[0].primary_reason, "insufficient_excitation")
    self.assertEqual(status.qualified_node_count, 1)
    self.assertEqual(status.nodes[-1].speed_mps, 30.0)

  def test_sparse_fit_errors_remain_collection_state(self) -> None:
    payload = learning_fixture()
    payload["nodes"][2]["reasons"] = [
      "insufficient_support",
      "insufficient_validation",
      "insufficient_excitation",
      "singular_fit",
      "invalid_parameters",
    ]
    status = parse_learning_status(payload, expected_vehicle_identity=VEHICLE)
    node = status.nodes[2]
    self.assertEqual(node.primary_reason, "insufficient_support")
    self.assertFalse(node.collection_complete)

    payload["nodes"][2]["reasons"] = ["singular_fit"]
    status = parse_learning_status(payload, expected_vehicle_identity=VEHICLE)
    node = status.nodes[2]
    self.assertEqual(node.primary_reason, "singular_fit")
    self.assertTrue(node.collection_complete)

  def test_last_drive_unavailable_is_null_not_zero(self) -> None:
    status = parse_learning_status(
      learning_fixture(last_drive_complete=False),
      expected_vehicle_identity=VEHICLE,
    )
    self.assertFalse(status.last_drive_complete)
    self.assertTrue(
      all(node.last_drive_clean_support_s is None for node in status.nodes),
    )
    self.assertTrue(
      all(
        node.last_drive_accepted_sample_count is None
        for node in status.nodes
      ),
    )

  def test_mixed_last_drive_completeness_is_rejected(self) -> None:
    payload = learning_fixture(last_drive_complete=False)
    payload["nodes"][0]["last_drive_clean_support_s"] = 0.0
    with self.assertRaisesRegex(
      LearningStatusError,
      "last-drive completeness",
    ):
      parse_learning_status(payload, expected_vehicle_identity=VEHICLE)

  def test_malformed_and_schema_mismatch_are_unavailable(self) -> None:
    with self.assertRaises(LearningStatusError) as malformed:
      parse_learning_status("not a Params JSON object", expected_vehicle_identity=VEHICLE)
    self.assertEqual(malformed.exception.code, "malformed")

    payload = learning_fixture()
    payload["schema_version"] = 2
    with self.assertRaises(LearningStatusError) as mismatch:
      parse_learning_status(payload, expected_vehicle_identity=VEHICLE)
    self.assertEqual(mismatch.exception.code, "schema_mismatch")

  def test_wrong_vehicle_fails_closed(self) -> None:
    with self.assertRaises(LearningStatusError) as mismatch:
      parse_learning_status(
        learning_fixture(),
        expected_vehicle_identity="ANOTHER CAR",
      )
    self.assertEqual(mismatch.exception.code, "wrong_vehicle")

  def test_node_list_is_dynamic_but_ordered(self) -> None:
    payload = learning_fixture()
    payload["nodes"] = payload["nodes"][:3]
    status = parse_learning_status(payload, expected_vehicle_identity=VEHICLE)
    self.assertEqual([node.node_index for node in status.nodes], [0, 1, 2])

    broken = deepcopy(payload)
    broken["nodes"][2]["speed_mps"] = 4.0
    with self.assertRaisesRegex(LearningStatusError, "strictly increasing"):
      parse_learning_status(broken, expected_vehicle_identity=VEHICLE)

  def test_qualification_summary_is_not_activation(self) -> None:
    payload = learning_fixture()
    for node in payload["nodes"]:
      node["qualified"] = True
      node["reasons"] = ["qualified"]
      node["candidate_parameters"] = {
        "torque_per_lateral_accel": 0.3,
        "rack_gain_deg_s2_per_torque": 4000.0,
        "rack_damping_per_s": 10.0,
        "kinetic_friction_torque": 0.03,
      }
    payload["all_nodes_qualified"] = True
    payload["candidate_profile_sha256"] = "c" * 64
    payload["candidate_profile_revision"] = 10
    status = parse_learning_status(payload, expected_vehicle_identity=VEHICLE)
    self.assertTrue(status.all_nodes_qualified)
    # LearningStatus intentionally has no active-controller property.
    self.assertFalse(hasattr(status, "active"))


class TestLearningOperationStatusParser(unittest.TestCase):
  def parse(self, payload: object):
    return parse_learning_operation_status(
      payload,
      expected_vehicle_identity=VEHICLE,
      expected_runtime_identity_sha256=RUNTIME_HASH,
      now_mono_ns=NOW_MONO_NS,
    )

  def test_every_state_has_a_strict_terminal_contract(self) -> None:
    for state in _OPERATION_DIAGNOSTICS:
      with self.subTest(state=state):
        status = self.parse(operation_fixture(state))
        self.assertEqual(status.state, state)
        self.assertEqual(
          status.terminal,
          state in _TERMINAL_OPERATION_STATES,
        )
        self.assertEqual(status.active, not status.terminal)

  def test_every_state_has_truthful_display_copy(self) -> None:
    expected = {
      "preparing": ("PREPARING LEARNER", "blue"),
      "ready_no_evidence": ("READY FOR FIRST DRIVE", "blue"),
      "collecting": ("COLLECTING THIS DRIVE", "blue"),
      "finalizing": ("FINALIZING LEARNING DATA", "blue"),
      "retry_pending": ("SAVE RETRY PENDING", "amber"),
      "backfilling": ("PROCESSING PRIOR ROUTES", "blue"),
      "idle": ("LEARNER READY", "green"),
      "drive_skipped_identity_mismatch": (
        "DRIVE SKIPPED · NEXT DRIVE READY",
        "amber",
      ),
      "failed": ("LEARNER FAILED", "red"),
    }
    for state, (title, tone) in expected.items():
      with self.subTest(state=state):
        presentation = operation_presentation(
          self.parse(operation_fixture(state)),
          error_code=None,
          error_message=None,
          has_learning_snapshot=False,
        )
        self.assertEqual(presentation.title, title)
        self.assertEqual(presentation.tone, tone)

  def test_backfill_progress_is_one_based_and_cumulative(self) -> None:
    status = self.parse(operation_fixture("backfilling"))
    self.assertEqual(status.current_route_index, 2)
    self.assertEqual(status.total_route_count, 5)
    self.assertEqual(status.accepted_sample_count, 12345)
    self.assertEqual(status.rejected_sample_count, 678)

    presentation = operation_presentation(
      status,
      error_code=None,
      error_message=None,
      has_learning_snapshot=False,
    )
    self.assertEqual(presentation.title, "PROCESSING PRIOR ROUTES")
    self.assertIn("Route 2/5", presentation.detail)
    self.assertIn("12,345 accepted", presentation.detail)

  def test_scanning_has_no_invented_route_progress(self) -> None:
    payload = operation_fixture(
      "backfilling",
      diagnostic="scanning_routes",
    )
    payload["current_route_identity"] = None
    payload["current_route_index"] = None
    payload["total_route_count"] = None
    status = self.parse(payload)
    self.assertIsNone(status.current_route_index)
    presentation = operation_presentation(
      status,
      error_code=None,
      error_message=None,
      has_learning_snapshot=False,
    )
    self.assertIn("Scanning compatible routes", presentation.detail)

  def test_progress_bounds_and_shapes_fail_closed(self) -> None:
    bad_payloads = []
    zero_index = operation_fixture("backfilling")
    zero_index["current_route_index"] = 0
    bad_payloads.append(zero_index)
    past_total = operation_fixture("backfilling")
    past_total["current_route_index"] = 6
    bad_payloads.append(past_total)
    missing_total = operation_fixture("backfilling")
    missing_total["total_route_count"] = None
    bad_payloads.append(missing_total)
    missing_route = operation_fixture("backfilling")
    missing_route["current_route_identity"] = None
    bad_payloads.append(missing_route)
    scanning_progress = operation_fixture(
      "backfilling",
      diagnostic="scanning_routes",
    )
    scanning_progress["current_route_identity"] = ROUTE_HASH
    scanning_progress["current_route_index"] = 1
    scanning_progress["total_route_count"] = 5
    bad_payloads.append(scanning_progress)
    non_backfill_progress = operation_fixture("collecting")
    non_backfill_progress["current_route_index"] = 1
    non_backfill_progress["total_route_count"] = 1
    bad_payloads.append(non_backfill_progress)
    idle_current_route = operation_fixture("idle")
    idle_current_route["current_route_identity"] = ROUTE_HASH
    bad_payloads.append(idle_current_route)

    for position, payload in enumerate(bad_payloads):
      with self.subTest(position=position):
        with self.assertRaises(LearningStatusError) as malformed:
          self.parse(payload)
        self.assertEqual(malformed.exception.code, "malformed")

  def test_malformed_unknown_and_extra_fields_fail_closed(self) -> None:
    unknown_state = operation_fixture()
    unknown_state["state"] = "training"
    with self.assertRaises(LearningStatusError) as state_error:
      self.parse(unknown_state)
    self.assertEqual(state_error.exception.code, "malformed")

    wrong_diagnostic = operation_fixture("collecting")
    wrong_diagnostic["diagnostic"] = "evidence_ready"
    with self.assertRaises(LearningStatusError) as diagnostic_error:
      self.parse(wrong_diagnostic)
    self.assertEqual(diagnostic_error.exception.code, "malformed")

    extra = operation_fixture()
    extra["surprise"] = True
    with self.assertRaises(LearningStatusError) as extra_error:
      self.parse(extra)
    self.assertEqual(extra_error.exception.code, "malformed")

    schema = operation_fixture()
    schema["schema_version"] = 2
    with self.assertRaises(LearningStatusError) as schema_error:
      self.parse(schema)
    self.assertEqual(schema_error.exception.code, "schema_mismatch")

  def test_stale_timestamps_fail_closed_without_an_age_timeout(self) -> None:
    inverted = operation_fixture("collecting")
    inverted["updated_mono_ns"] = inverted["started_mono_ns"] - 1
    with self.assertRaises(LearningStatusError) as inverted_error:
      self.parse(inverted)
    self.assertEqual(inverted_error.exception.code, "stale")

    future = operation_fixture("collecting")
    future["updated_mono_ns"] = NOW_MONO_NS + 1
    with self.assertRaises(LearningStatusError) as future_error:
      self.parse(future)
    self.assertEqual(future_error.exception.code, "stale")

    # The UI does not invent an inactivity timeout; transition/failure
    # ownership remains in the backend.
    old = operation_fixture("collecting")
    old["started_mono_ns"] = 1
    old["updated_mono_ns"] = 2
    self.assertEqual(self.parse(old).updated_mono_ns, 2)

  def test_sequence_regressions_and_mutations_are_stale(self) -> None:
    previous = self.parse(operation_fixture("collecting"))
    same = self.parse(operation_fixture("collecting"))
    validate_operation_update(previous, same)

    regressed_payload = operation_fixture("collecting")
    regressed_payload["sequence"] = previous.sequence - 1
    regressed = self.parse(regressed_payload)
    with self.assertRaises(LearningStatusError) as regression_error:
      validate_operation_update(previous, regressed)
    self.assertEqual(regression_error.exception.code, "stale")

    mutated_payload = operation_fixture("collecting")
    mutated_payload["accepted_sample_count"] += 1
    mutated = self.parse(mutated_payload)
    with self.assertRaises(LearningStatusError) as mutation_error:
      validate_operation_update(previous, mutated)
    self.assertEqual(mutation_error.exception.code, "stale")

    advanced_payload = operation_fixture("collecting")
    advanced_payload["sequence"] += 1
    advanced_payload["updated_mono_ns"] += 1
    advanced = self.parse(advanced_payload)
    validate_operation_update(previous, advanced)

  def test_wrong_vehicle_and_runtime_fail_closed(self) -> None:
    vehicle = operation_fixture()
    vehicle["vehicle_identity"] = "ANOTHER CAR"
    with self.assertRaises(LearningStatusError) as vehicle_error:
      self.parse(vehicle)
    self.assertEqual(vehicle_error.exception.code, "wrong_vehicle")

    runtime = operation_fixture()
    runtime["runtime_identity_sha256"] = "f" * 64
    with self.assertRaises(LearningStatusError) as runtime_error:
      self.parse(runtime)
    self.assertEqual(runtime_error.exception.code, "runtime_mismatch")

  def test_nullable_identity_is_limited_to_early_states(self) -> None:
    preparing = operation_fixture("preparing")
    preparing["vehicle_identity"] = None
    preparing["runtime_identity_sha256"] = None
    self.assertIsNone(self.parse(preparing).vehicle_identity)

    failed = operation_fixture("failed")
    failed["vehicle_identity"] = None
    failed["runtime_identity_sha256"] = None
    self.assertIsNone(self.parse(failed).runtime_identity_sha256)

    skipped = operation_fixture("drive_skipped_identity_mismatch")
    skipped["runtime_identity_sha256"] = None
    self.assertIsNone(self.parse(skipped).runtime_identity_sha256)

    collecting = operation_fixture("collecting")
    collecting["runtime_identity_sha256"] = None
    with self.assertRaises(LearningStatusError) as identity_error:
      self.parse(collecting)
    self.assertEqual(identity_error.exception.code, "malformed")

  def test_idle_requires_persisted_evidence(self) -> None:
    missing = operation_fixture("idle")
    missing["evidence_sha256"] = None
    with self.assertRaises(LearningStatusError) as missing_error:
      self.parse(missing)
    self.assertEqual(missing_error.exception.code, "malformed")

  def test_ready_no_evidence_cannot_claim_prior_evidence(self) -> None:
    for field, value in (
      ("evidence_sha256", HASH),
      ("ledger_sha256", "f" * 64),
      ("accepted_sample_count", 1),
      ("rejected_sample_count", 1),
    ):
      with self.subTest(field=field):
        payload = operation_fixture("ready_no_evidence")
        payload[field] = value
        if field == "ledger_sha256":
          payload["evidence_sha256"] = HASH
        with self.assertRaises(LearningStatusError) as evidence_error:
          self.parse(payload)
        self.assertEqual(evidence_error.exception.code, "malformed")

  def test_backfill_exclusions_and_missing_ledger_are_explicit(self) -> None:
    skipped = self.parse(
      operation_fixture(
        "idle",
        diagnostic="backfill_complete_late_older_skipped",
      ),
    )
    skipped_display = operation_presentation(
      skipped,
      error_code=None,
      error_message=None,
      has_learning_snapshot=True,
    )
    self.assertIn("late older routes", skipped_display.detail)

    untracked = operation_fixture(
      "failed",
      diagnostic="backfill_untracked_evidence",
    )
    untracked["evidence_sha256"] = HASH
    untracked_status = self.parse(untracked)
    untracked_display = operation_presentation(
      untracked_status,
      error_code=None,
      error_message=None,
      has_learning_snapshot=True,
    )
    self.assertEqual(untracked_display.tone, "red")
    self.assertIn("Backfill unavailable", untracked_display.detail)

  def test_absence_never_claims_that_no_drive_occurred(self) -> None:
    presentation = learning_panel_presentation(
      None,
      operation_error_code="operation_absent",
      operation_error_message="not published",
      learning_error_code="absent",
      learning_error_message="not published",
      has_learning_snapshot=False,
    )
    self.assertEqual(presentation.title, "LEARNER STATUS UNAVAILABLE")
    self.assertIn("history is unknown", presentation.detail)
    self.assertNotIn("Complete one drive", presentation.detail)
    self.assertEqual(presentation.tone, "gray")

  def test_only_explicit_ready_state_requests_a_first_drive(self) -> None:
    status = self.parse(operation_fixture("ready_no_evidence"))
    presentation = learning_panel_presentation(
      status,
      operation_error_code=None,
      operation_error_message=None,
      learning_error_code="absent",
      learning_error_message="not published",
      has_learning_snapshot=False,
    )
    self.assertEqual(presentation.title, "READY FOR FIRST DRIVE")
    self.assertIn("Complete one drive", presentation.detail)

  def test_processing_keeps_prior_snapshot_and_adds_banner(self) -> None:
    for state in (
      "preparing",
      "collecting",
      "finalizing",
      "retry_pending",
      "backfilling",
    ):
      with self.subTest(state=state):
        status = self.parse(operation_fixture(state))
        presentation = learning_panel_presentation(
          status,
          operation_error_code=None,
          operation_error_message=None,
          learning_error_code=None,
          learning_error_message=None,
          has_learning_snapshot=True,
        )
        self.assertTrue(presentation.show_banner)
        self.assertIn("Prior snapshot shown", presentation.detail)

  def test_skipped_and_failed_messages_have_distinct_error_tones(self) -> None:
    skipped = self.parse(
      operation_fixture("drive_skipped_identity_mismatch"),
    )
    skipped_display = operation_presentation(
      skipped,
      error_code=None,
      error_message=None,
      has_learning_snapshot=True,
    )
    self.assertEqual(skipped_display.tone, "amber")
    self.assertIn("prepared for the next drive", skipped_display.detail)

    failed = self.parse(operation_fixture("failed"))
    failed_display = operation_presentation(
      failed,
      error_code=None,
      error_message=None,
      has_learning_snapshot=True,
    )
    self.assertEqual(failed_display.tone, "red")
    self.assertIn("unexpected error", failed_display.detail)

    for code in ("malformed", "stale", "wrong_vehicle", "runtime_mismatch"):
      with self.subTest(error_code=code):
        error_display = operation_presentation(
          None,
          error_code=code,
          error_message=f"{code} detail",
          has_learning_snapshot=False,
        )
        self.assertEqual(error_display.tone, "red")
        self.assertEqual(error_display.detail, f"{code} detail")


class TestLifecycleStatusParser(unittest.TestCase):
  def test_stock_staged_provisional_and_approved_badges(self) -> None:
    expected = {
      "stock": ("STOCK ACTIVE", "stock"),
      "staged": ("STOCK ACTIVE", "stock"),
      "provisional": ("BLATV2 PROVISIONAL", "modular"),
      "approved": ("BLATV2 APPROVED", "modular"),
      "rollback_pending": ("ROLLBACK PENDING · STOCK ACTIVE", "stock"),
    }
    for state, (badge, effective) in expected.items():
      with self.subTest(state=state):
        status = parse_lifecycle_status(
          lifecycle_fixture(state),
          expected_vehicle_identity=VEHICLE,
          expected_runtime_identity_sha256=RUNTIME_HASH,
        )
        self.assertEqual(status.badge, badge)
        self.assertEqual(status.effective_controller, effective)

  def test_learning_lifecycle_runtime_mismatch_fails_closed(self) -> None:
    with self.assertRaises(LearningStatusError) as mismatch:
      parse_lifecycle_status(
        lifecycle_fixture(runtime_hash="d" * 64),
        expected_vehicle_identity=VEHICLE,
        expected_runtime_identity_sha256=RUNTIME_HASH,
      )
    self.assertEqual(mismatch.exception.code, "runtime_mismatch")

  def test_approved_requires_validated_modular_state(self) -> None:
    payload = lifecycle_fixture("approved")
    payload["production_envelope_verified"] = False
    with self.assertRaisesRegex(LearningStatusError, "not validated"):
      parse_lifecycle_status(
        payload,
        expected_vehicle_identity=VEHICLE,
        expected_runtime_identity_sha256=RUNTIME_HASH,
      )

  def test_staged_and_rollback_require_validated_state(self) -> None:
    for state in ("staged", "rollback_pending"):
      with self.subTest(state=state):
        payload = lifecycle_fixture(state)
        payload["diagnostic"] = "absent"
        with self.assertRaisesRegex(
          LearningStatusError,
          "profile-bearing stock lifecycle",
        ):
          parse_lifecycle_status(
            payload,
            expected_vehicle_identity=VEHICLE,
            expected_runtime_identity_sha256=RUNTIME_HASH,
          )
        payload = lifecycle_fixture(state)
        payload["production_envelope_verified"] = False
        with self.assertRaisesRegex(
          LearningStatusError,
          "profile-bearing stock lifecycle",
        ):
          parse_lifecycle_status(
            payload,
            expected_vehicle_identity=VEHICLE,
            expected_runtime_identity_sha256=RUNTIME_HASH,
          )

  def test_impossible_stock_and_missing_state_identity_are_rejected(self) -> None:
    impossible = lifecycle_fixture("stock")
    impossible["staged_profile"] = {
      "artifact_sha256": "8" * 64,
      "profile_sha256": "9" * 64,
      "profile_revision": 43,
    }
    with self.assertRaisesRegex(LearningStatusError, "unexpectedly contains"):
      parse_lifecycle_status(
        impossible,
        expected_vehicle_identity=VEHICLE,
        expected_runtime_identity_sha256=RUNTIME_HASH,
      )

    missing_identity = lifecycle_fixture("stock")
    missing_identity["activation_state_sha256"] = None
    with self.assertRaisesRegex(LearningStatusError, "activation-state identity"):
      parse_lifecycle_status(
        missing_identity,
        expected_vehicle_identity=VEHICLE,
        expected_runtime_identity_sha256=RUNTIME_HASH,
      )


class TestLearningStatusPresentationHelpers(unittest.TestCase):
  def test_duration_speed_and_reasons(self) -> None:
    self.assertEqual(format_duration(0.0), "0:00")
    self.assertEqual(format_duration(150.0), "2:30")
    self.assertEqual(format_duration(3661.0), "1h 01m")
    self.assertEqual(format_speed(10.0, metric=True), "36 km/h")
    self.assertEqual(format_speed(10.0, metric=False), "22 mph")
    self.assertEqual(
      reason_label("insufficient_validation"),
      "Needs held-out validation",
    )
    labels = {
      reason_label(reason)
      for reason in (
        "qualified",
        "insufficient_support",
        "insufficient_validation",
        "insufficient_excitation",
        "singular_fit",
        "invalid_parameters",
        "validation_regression",
      )
    }
    self.assertEqual(len(labels), 7)

  def test_supplied_false_metric_provider_is_not_truth_tested(self) -> None:
    selected = select_value_provider(lambda: True, lambda: False)
    self.assertFalse(selected())

  def test_two_by_three_grid_geometry(self) -> None:
    cells = grid_cells(1000.0, 600.0, 6, columns=2, gap=20.0)
    self.assertEqual(len(cells), 6)
    self.assertEqual(cells[0].x, cells[2].x)
    self.assertGreater(cells[1].x, cells[0].x + cells[0].width)
    self.assertGreater(cells[2].y, cells[0].y + cells[0].height)
    self.assertLessEqual(
      max(cell.y + cell.height for cell in cells),
      600.0,
    )
    portable_cells = grid_cells(1000.0, 600.0, 8, columns=3, gap=14.0)
    self.assertEqual(len(portable_cells), 8)
    self.assertLessEqual(
      max(cell.x + cell.width for cell in portable_cells),
      1000.0,
    )

  def test_four_page_navigation_wraps_in_both_directions(self) -> None:
    self.assertEqual(cycle_page_index(0, 1, 4), 1)
    self.assertEqual(cycle_page_index(3, 1, 4), 0)
    self.assertEqual(cycle_page_index(0, -1, 4), 3)
    self.assertEqual(cycle_page_index(2, -1, 4), 1)


if __name__ == "__main__":
  unittest.main()
