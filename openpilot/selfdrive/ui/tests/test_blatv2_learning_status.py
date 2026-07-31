from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from openpilot.selfdrive.ui.widgets.blatv2_learning_status import (
  LearningStatusError,
  cycle_page_index,
  format_duration,
  format_speed,
  grid_cells,
  learning_panel_presentation,
  operation_presentation,
  parse_backfill_progress_status,
  parse_learning_operation_status,
  parse_learning_status,
  parse_lifecycle_status,
  reason_label,
  select_value_provider,
  validate_backfill_progress_update,
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
  base_support_s = clean_support_s * 0.5
  moving_support_s = clean_support_s * 0.3
  breakaway_support_s = clean_support_s - base_support_s - moving_support_s
  last_drive_clean_support_s = 12.5 if last_drive_complete else None
  return {
    "node_index": index,
    "speed_mps": float((0, 5, 10, 15, 20, 30)[index]),
    "minimum_support_s": minimum_support_s,
    "clean_support_s": clean_support_s,
    "last_drive_clean_support_s": last_drive_clean_support_s,
    "supported_sample_count": 6000,
    "last_drive_accepted_sample_count": 1250 if last_drive_complete else None,
    "base_support_s": base_support_s,
    "base_sample_count": 3000,
    "last_drive_base_support_s": 6.25 if last_drive_complete else None,
    "last_drive_base_sample_count": 625 if last_drive_complete else None,
    "moving_support_s": moving_support_s,
    "moving_sample_count": 1800,
    "moving_training_count": 1440,
    "moving_validation_count": 360,
    "last_drive_moving_support_s": 3.75 if last_drive_complete else None,
    "last_drive_moving_sample_count": 375 if last_drive_complete else None,
    "breakaway_support_s": breakaway_support_s,
    "breakaway_sample_count": 1200,
    "breakaway_training_count": 960,
    "breakaway_validation_count": 240,
    "last_drive_breakaway_support_s": 2.5 if last_drive_complete else None,
    "last_drive_breakaway_sample_count": 250 if last_drive_complete else None,
    "authority_support_s": 5.0,
    "authority_sample_count": 500,
    "authority_fit_support_s": 3.0,
    "authority_fit_sample_count": 300,
    "authority_training_count": 240,
    "authority_validation_count": 60,
    "last_drive_authority_support_s": 1.0 if last_drive_complete else None,
    "last_drive_authority_sample_count": 100 if last_drive_complete else None,
    "last_drive_authority_fit_support_s": 0.8 if last_drive_complete else None,
    "last_drive_authority_fit_sample_count": 80 if last_drive_complete else None,
    "training_count": 4800,
    "validation_count": 1200,
    "validation_support_s": 24.0,
    "minimum_validation_support_s": minimum_support_s * 0.2,
    "lateral_accel_span_mps2": 0.9,
    "lateral_accel_rms_mps2": 0.3,
    "rack_travel_deg": 200.0,
    "applied_torque_span": 0.2,
    "rack_reversals": 6,
    "lateral_accel_directions": 2,
    "applied_torque_directions": 2,
    "seed_validation_rms": 0.1,
    "candidate_validation_rms": 0.08,
    "moving_seed_validation_rms": 0.11,
    "moving_candidate_validation_rms": 0.07,
    "breakaway_seed_validation_rms": 0.12,
    "breakaway_candidate_validation_rms": 0.075,
    "authority_seed_validation_rms": 0.13,
    "authority_candidate_validation_rms": 0.09,
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
        "lateral_accel_offset_correction_mps2": -0.04,
        "kinetic_friction_torque": 0.03,
        "static_breakaway_torque": 0.09,
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
    "schema_version": 2,
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


def backfill_progress_fixture(
  phase: str = "reading_segment",
  *,
  pass_index: int = 1,
  operation_sequence: int = 7,
  approximate_remaining_seconds: int | None = None,
) -> dict:
  replaying = phase in ("reading_segment", "applying_route")
  final = phase in ("comparing", "publishing")
  return {
    "schema_version": 1,
    "informational_only": True,
    "operation_id": OPERATION_ID,
    "operation_sequence": operation_sequence,
    "sequence": 11,
    "updated_mono_ns": NOW_MONO_NS - 50_000_000,
    "phase": phase,
    "pass_index": 2 if final else pass_index,
    "pass_count": 2,
    "current_route_identity": ROUTE_HASH if replaying else None,
    "current_route_index": 2 if replaying else None,
    "total_route_count": 5,
    "current_segment_index": (
      26 if phase == "applying_route" else 4 if replaying else None
    ),
    "current_route_segment_count": 26 if replaying else None,
    "completed_replay_segment_count": (
      400 if final else 242 if pass_index == 2 else 42
    ),
    "total_replay_segment_count": 400,
    "completed_work_units": (
      1000 if final else 650 if pass_index == 2 else 250
    ),
    "total_work_units": 1000,
    "approximate_remaining_seconds": (
      None if final else approximate_remaining_seconds
    ),
  }


class TestLearningStatusParser(unittest.TestCase):
  def test_schema_v2_exposes_observable_calibration_and_evidence(self) -> None:
    status = parse_learning_status(
      learning_fixture(),
      expected_vehicle_identity=VEHICLE,
    )
    node = status.nodes[1]
    self.assertEqual(node.base_support_s, 75.0)
    self.assertEqual(node.moving_support_s, 45.0)
    self.assertEqual(node.breakaway_support_s, 30.0)
    self.assertEqual(node.authority_fit_sample_count, 300)
    self.assertEqual(node.last_drive_base_sample_count, 625)
    self.assertEqual(node.last_drive_moving_sample_count, 375)
    self.assertEqual(node.last_drive_breakaway_sample_count, 250)
    self.assertEqual(node.last_drive_authority_fit_sample_count, 80)
    self.assertEqual(node.moving_candidate_validation_rms, 0.07)
    self.assertEqual(node.breakaway_candidate_validation_rms, 0.075)
    self.assertEqual(node.authority_candidate_validation_rms, 0.09)
    self.assertIsNotNone(node.candidate_parameters)
    self.assertEqual(
      node.candidate_parameters.lateral_accel_offset_correction_mps2,
      -0.04,
    )
    self.assertEqual(node.candidate_parameters.static_breakaway_torque, 0.09)
    self.assertFalse(hasattr(node.candidate_parameters, "rack_gain_deg_s2_per_torque"))

  def test_schema_v2_rejects_legacy_rack_parameters(self) -> None:
    payload = learning_fixture()
    parameters = payload["nodes"][1]["candidate_parameters"]
    parameters["rack_gain_deg_s2_per_torque"] = 4000.0
    with self.assertRaisesRegex(LearningStatusError, "keys do not match"):
      parse_learning_status(payload, expected_vehicle_identity=VEHICLE)

  def test_population_invariants_fail_closed(self) -> None:
    mutations = (
      ("clean_support_s", 61.0, "clean support populations"),
      ("supported_sample_count", 6001, "clean sample populations"),
      ("authority_fit_sample_count", 501, "authority fit exceeds"),
    )
    for field, value, message in mutations:
      with self.subTest(field=field):
        payload = learning_fixture()
        payload["nodes"][2][field] = value
        with self.assertRaisesRegex(LearningStatusError, message):
          parse_learning_status(payload, expected_vehicle_identity=VEHICLE)

  def test_device_accumulation_rounding_remains_displayable(self) -> None:
    # Exact six-node support values from the first four-worker device
    # generation. The producer's authoritative validator accepts these
    # ordinary binary64 accumulation differences, so the UI must as well.
    populations = (
      ("0x1.0a44139e49b93p+8", "0x1.d3042eaf59bcfp+7", "0x1.fd8f2a8005819p+4", "0x1.d2133d3905d82p-1"),
      ("0x1.62b638afc1237p+9", "0x1.3bfb958f8bbd7p+9", "0x1.2dba9292cfb84p+6", "0x1.0350cddb6e06dp+1"),
      ("0x1.03a6a45be81c4p+10", "0x1.d8618befd2dcap+9", "0x1.6d15043517f6ap+6", "0x1.491c415a53a3bp+1"),
      ("0x1.22eb64ebea10ep+10", "0x1.0e8ec3b25b6e8p+10", "0x1.3be940d524edfp+6", "0x1.3c1a5878ad3f9p+1"),
      ("0x1.8de1131e56d21p+9", "0x1.77a8f0c2c4053p+9", "0x1.5a9fb39f2b29fp+5", "0x1.1c4e434035d3dp+0"),
      ("0x1.21c8b4ef5f334p+8", "0x1.1730de571cafcp+8", "0x1.38815f1017158p+3", "0x1.a7973f83963a6p-1"),
    )
    for node_index, encoded in enumerate(populations):
      with self.subTest(node_index=node_index):
        payload = learning_fixture()
        node = payload["nodes"][node_index]
        (
          node["clean_support_s"],
          node["base_support_s"],
          node["moving_support_s"],
          node["breakaway_support_s"],
        ) = tuple(float.fromhex(value) for value in encoded)
        parsed = parse_learning_status(
          payload,
          expected_vehicle_identity=VEHICLE,
        )
        self.assertEqual(
          parsed.nodes[node_index].clean_support_s,
          float.fromhex(encoded[0]),
        )

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

  def test_moving_and_breakaway_are_independent_readiness_blockers(self) -> None:
    payload = learning_fixture()
    payload["nodes"][2]["reasons"] = [
      "insufficient_moving_evidence",
      "insufficient_breakaway_evidence",
    ]
    node = parse_learning_status(
      payload,
      expected_vehicle_identity=VEHICLE,
    ).nodes[2]
    self.assertEqual(node.primary_reason, "insufficient_moving_evidence")
    self.assertFalse(node.moving_ready)
    self.assertFalse(node.breakaway_ready)
    self.assertFalse(node.collection_complete)

  def test_authority_validation_regression_is_specific_and_complete(
    self,
  ) -> None:
    payload = learning_fixture()
    payload["nodes"][2]["reasons"] = [
      "validation_regression",
      "authority_validation_regression",
    ]
    status = parse_learning_status(payload, expected_vehicle_identity=VEHICLE)
    node = status.nodes[2]
    self.assertEqual(
      node.primary_reason,
      "authority_validation_regression",
    )
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
    self.assertTrue(
      all(
        node.last_drive_base_support_s is None
        and node.last_drive_moving_support_s is None
        and node.last_drive_breakaway_support_s is None
        and node.last_drive_authority_support_s is None
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
    payload["schema_version"] = 1
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
        "lateral_accel_offset_correction_mps2": -0.04,
        "kinetic_friction_torque": 0.03,
        "static_breakaway_torque": 0.09,
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
        "DRIVE SKIPPED | NEXT DRIVE READY",
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
    self.assertIn("12,345 incorporated", presentation.detail)
    self.assertIn("678 excluded", presentation.detail)

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

  def test_cumulative_progress_cannot_regress_within_an_operation(self) -> None:
    previous = self.parse(operation_fixture("backfilling"))
    mutations = (
      {"accepted_sample_count": previous.accepted_sample_count - 1},
      {"rejected_sample_count": previous.rejected_sample_count - 1},
      {"current_route_index": previous.current_route_index - 1},
      {"total_route_count": previous.total_route_count + 1},
    )
    for mutation in mutations:
      with self.subTest(mutation=mutation):
        payload = operation_fixture("backfilling")
        payload.update(mutation)
        payload["sequence"] += 1
        payload["updated_mono_ns"] += 1
        with self.assertRaises(LearningStatusError) as stale:
          validate_operation_update(previous, self.parse(payload))
        self.assertEqual(stale.exception.code, "stale")

    retry_previous = self.parse(operation_fixture("retry_pending"))
    retry_payload = operation_fixture("retry_pending")
    retry_payload["sequence"] += 1
    retry_payload["updated_mono_ns"] += 1
    retry_payload["retry_count"] -= 1
    with self.assertRaises(LearningStatusError) as retry_stale:
      validate_operation_update(retry_previous, self.parse(retry_payload))
    self.assertEqual(retry_stale.exception.code, "stale")

  def test_terminal_operation_cannot_resume_without_a_new_identity(self) -> None:
    previous = self.parse(operation_fixture("idle"))
    resumed = operation_fixture("preparing")
    resumed["sequence"] += 1
    resumed["updated_mono_ns"] += 1
    with self.assertRaises(LearningStatusError) as stale:
      validate_operation_update(previous, self.parse(resumed))
    self.assertEqual(stale.exception.code, "stale")

    resumed["operation_id"] = "a" * 32
    validate_operation_update(previous, self.parse(resumed))

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

  def test_backfill_progress_uses_display_safe_ascii_separators(self) -> None:
    payload = operation_fixture("backfilling")
    payload["current_route_index"] = 1
    payload["total_route_count"] = 20
    payload["accepted_sample_count"] = 0
    payload["rejected_sample_count"] = 0
    status = self.parse(payload)
    presentation = operation_presentation(
      status,
      error_code=None,
      error_message=None,
      has_learning_snapshot=False,
    )
    self.assertEqual(
      presentation.detail,
      "Route 1/20 | 0 incorporated | 0 excluded",
    )
    self.assertTrue(presentation.title.isascii())
    self.assertTrue(presentation.detail.isascii())

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


class TestBackfillProgressStatusParser(unittest.TestCase):
  @staticmethod
  def operation(
    state: str = "backfilling",
    *,
    diagnostic: str | None = None,
  ):
    return parse_learning_operation_status(
      operation_fixture(state, diagnostic=diagnostic),
      expected_vehicle_identity=VEHICLE,
      expected_runtime_identity_sha256=RUNTIME_HASH,
      now_mono_ns=NOW_MONO_NS,
    )

  def parse(self, payload: object, operation=None):
    return parse_backfill_progress_status(
      payload,
      operation_status=self.operation() if operation is None else operation,
      now_mono_ns=NOW_MONO_NS,
    )

  def test_pass_one_and_pass_two_detail_bind_to_the_operation(self) -> None:
    first = self.parse(backfill_progress_fixture())
    self.assertEqual(first.pass_index, 1)
    self.assertEqual(first.current_segment_index, 4)
    self.assertEqual(first.progress_fraction, 0.25)

    verifying = self.operation(
      "finalizing",
      diagnostic="verifying_backfill",
    )
    second = self.parse(
      backfill_progress_fixture(pass_index=2),
      operation=verifying,
    )
    self.assertEqual(second.pass_index, 2)
    self.assertEqual(second.completed_work_units, 650)

  def test_route_segment_pass_and_work_bounds_fail_closed(self) -> None:
    mutations = (
      {"pass_index": 0},
      {"pass_count": 3},
      {"current_route_index": None},
      {"current_route_index": 0},
      {"current_route_index": 6},
      {"current_segment_index": None},
      {"current_segment_index": 0},
      {"current_segment_index": 27},
      {"current_route_segment_count": 0},
      {"completed_replay_segment_count": 401},
      {"total_replay_segment_count": 0},
      {"completed_work_units": 1000},
      {"total_work_units": 0},
    )
    for mutation in mutations:
      with self.subTest(mutation=mutation):
        payload = backfill_progress_fixture()
        payload.update(mutation)
        with self.assertRaises(LearningStatusError) as malformed:
          self.parse(payload)
        self.assertEqual(malformed.exception.code, "malformed")

    applying = backfill_progress_fixture("applying_route")
    applying["current_segment_index"] = 25
    with self.assertRaises(LearningStatusError) as malformed_applying:
      self.parse(applying)
    self.assertEqual(malformed_applying.exception.code, "malformed")

  def test_exact_schema_phase_and_informational_marker_are_strict(self) -> None:
    payloads = []
    extra = backfill_progress_fixture()
    extra["surprise"] = True
    payloads.append(extra)
    schema = backfill_progress_fixture()
    schema["schema_version"] = 2
    payloads.append(schema)
    marker = backfill_progress_fixture()
    marker["informational_only"] = False
    payloads.append(marker)
    phase = backfill_progress_fixture()
    phase["phase"] = "estimating"
    payloads.append(phase)

    for position, payload in enumerate(payloads):
      with self.subTest(position=position):
        with self.assertRaises(LearningStatusError):
          self.parse(payload)

  def test_comparing_and_publishing_have_complete_work_but_no_bar_eta_or_route(
    self,
  ) -> None:
    cases = (
      (
        "comparing",
        self.operation("finalizing", diagnostic="verifying_backfill"),
      ),
      (
        "publishing",
        self.operation("finalizing", diagnostic="publishing_backfill"),
      ),
    )
    for phase, operation in cases:
      with self.subTest(phase=phase):
        status = self.parse(
          backfill_progress_fixture(phase),
          operation=operation,
        )
        self.assertIsNone(status.current_route_identity)
        self.assertIsNone(status.current_segment_index)
        self.assertEqual(
          status.completed_work_units,
          status.total_work_units,
        )
        self.assertIsNone(status.approximate_remaining_seconds)

    invalid = backfill_progress_fixture("comparing")
    invalid["current_route_identity"] = ROUTE_HASH
    with self.assertRaises(LearningStatusError):
      self.parse(
        invalid,
        operation=self.operation(
          "finalizing",
          diagnostic="verifying_backfill",
        ),
      )

  def test_operation_binding_state_and_monotonic_epoch_fail_closed(self) -> None:
    wrong_id = backfill_progress_fixture()
    wrong_id["operation_id"] = "a" * 32
    wrong_sequence = backfill_progress_fixture()
    wrong_sequence["operation_sequence"] += 1
    wrong_state = backfill_progress_fixture()
    future = backfill_progress_fixture()
    future["updated_mono_ns"] = NOW_MONO_NS + 1
    before_operation = backfill_progress_fixture()
    before_operation["updated_mono_ns"] = (
      operation_fixture()["updated_mono_ns"] - 1
    )

    for payload, operation, code in (
      (wrong_id, self.operation(), "progress_mismatch"),
      (wrong_sequence, self.operation(), "progress_mismatch"),
      (
        wrong_state,
        self.operation("finalizing", diagnostic="verifying_backfill"),
        "progress_mismatch",
      ),
      (future, self.operation(), "stale"),
      (before_operation, self.operation(), "stale"),
    ):
      with self.subTest(code=code, payload=payload):
        with self.assertRaises(LearningStatusError) as error:
          self.parse(payload, operation=operation)
        self.assertEqual(error.exception.code, code)

  def test_progress_update_totals_coordinates_and_work_cannot_regress(
    self,
  ) -> None:
    previous = self.parse(backfill_progress_fixture())
    advanced = replace(
      previous,
      sequence=previous.sequence + 1,
      updated_mono_ns=previous.updated_mono_ns + 1,
      current_segment_index=previous.current_segment_index + 1,
      completed_replay_segment_count=(
        previous.completed_replay_segment_count + 1
      ),
      completed_work_units=previous.completed_work_units + 10,
    )
    validate_backfill_progress_update(previous, advanced)
    validate_backfill_progress_update(
      previous,
      replace(advanced, updated_mono_ns=previous.updated_mono_ns),
    )

    regressions = (
      {"sequence": previous.sequence - 1},
      {"pass_index": 0},
      {"current_route_index": previous.current_route_index - 1},
      {"current_segment_index": previous.current_segment_index - 1},
      {
        "completed_replay_segment_count":
          previous.completed_replay_segment_count - 1,
      },
      {"completed_work_units": previous.completed_work_units - 1},
      {"total_route_count": previous.total_route_count + 1},
      {"total_replay_segment_count": previous.total_replay_segment_count + 1},
      {"total_work_units": previous.total_work_units + 1},
    )
    for mutation in regressions:
      with self.subTest(mutation=mutation):
        fields = {
          "sequence": previous.sequence + 1,
          "updated_mono_ns": previous.updated_mono_ns + 1,
        }
        fields.update(mutation)
        current = replace(previous, **fields)
        with self.assertRaises(LearningStatusError) as stale:
          validate_backfill_progress_update(previous, current)
        self.assertEqual(stale.exception.code, "stale")

  def test_route_index_can_reset_only_when_the_pass_advances(self) -> None:
    first = self.parse(backfill_progress_fixture())
    verifying = self.operation(
      "finalizing",
      diagnostic="verifying_backfill",
    )
    payload = backfill_progress_fixture(pass_index=2)
    payload["sequence"] = first.sequence + 1
    payload["updated_mono_ns"] = first.updated_mono_ns + 1
    payload["current_route_index"] = 1
    second = self.parse(payload, operation=verifying)
    validate_backfill_progress_update(first, second)

    same_pass = replace(second, current_route_index=0)
    with self.assertRaises(LearningStatusError):
      validate_backfill_progress_update(second, same_pass)


class TestBackfillProgressPresentation(unittest.TestCase):
  @staticmethod
  def operation(
    state: str = "backfilling",
    *,
    diagnostic: str | None = None,
  ):
    return parse_learning_operation_status(
      operation_fixture(state, diagnostic=diagnostic),
      expected_vehicle_identity=VEHICLE,
      expected_runtime_identity_sha256=RUNTIME_HASH,
      now_mono_ns=NOW_MONO_NS,
    )

  @staticmethod
  def progress(payload: dict, operation):
    return parse_backfill_progress_status(
      payload,
      operation_status=operation,
      now_mono_ns=NOW_MONO_NS,
    )

  def presentation(self, operation, progress, *, has_snapshot: bool = True):
    return operation_presentation(
      operation,
      error_code=None,
      error_message=None,
      has_learning_snapshot=has_snapshot,
      backfill_progress=progress,
    )

  def test_both_passes_have_exact_primary_copy_and_determinate_bar(self) -> None:
    first_operation = self.operation()
    first_progress = self.progress(
      backfill_progress_fixture(),
      first_operation,
    )
    first = self.presentation(first_operation, first_progress)
    self.assertEqual(first.title, "PROCESSING PRIOR ROUTES")
    self.assertEqual(
      first.detail,
      "Pass 1/2 | Route 2/5 | Segment 4/26",
    )
    self.assertIn("Reading and validating", first.phase_detail)
    self.assertEqual(first.compact_meta, "Estimating time")
    self.assertNotIn("%", first.meta)
    self.assertEqual(first.progress_fraction, 0.25)

    second_operation = self.operation(
      "finalizing",
      diagnostic="verifying_backfill",
    )
    payload = backfill_progress_fixture(
      pass_index=2,
      approximate_remaining_seconds=601,
    )
    payload["current_route_index"] = 3
    payload["total_route_count"] = 20
    second_progress = self.progress(payload, second_operation)
    second = self.presentation(second_operation, second_progress)
    self.assertEqual(second.title, "VERIFYING PRIOR ROUTES")
    self.assertEqual(
      second.detail,
      "Pass 2/2 | Route 3/20 | Segment 4/26",
    )
    self.assertEqual(second.progress_fraction, 0.65)
    self.assertEqual(second.compact_meta, "65% | About 11 min left")
    self.assertIn("12,345 incorporated", second.meta)
    self.assertIn("678 excluded", second.meta)

  def test_applying_phase_and_initial_no_snapshot_screen_keep_detail(self) -> None:
    operation = self.operation()
    progress = self.progress(
      backfill_progress_fixture("applying_route"),
      operation,
    )
    presentation = self.presentation(
      operation,
      progress,
      has_snapshot=False,
    )
    self.assertFalse(presentation.show_banner)
    self.assertEqual(
      presentation.detail,
      "Pass 1/2 | Route 2/5 | Segment 26/26",
    )
    self.assertEqual(
      presentation.phase_detail,
      "Applying validated route evidence",
    )
    self.assertIsNotNone(presentation.progress_fraction)

  def test_comparing_and_publishing_are_distinct_and_have_no_false_bar(
    self,
  ) -> None:
    cases = (
      (
        "comparing",
        "verifying_backfill",
        "COMPARING REPLAY PASSES",
      ),
      (
        "publishing",
        "publishing_backfill",
        "SAVING VERIFIED LEARNING DATA",
      ),
    )
    for phase, diagnostic, title in cases:
      with self.subTest(phase=phase):
        operation = self.operation("finalizing", diagnostic=diagnostic)
        progress = self.progress(
          backfill_progress_fixture(phase),
          operation,
        )
        presentation = self.presentation(operation, progress)
        self.assertEqual(presentation.title, title)
        self.assertIsNone(presentation.progress_fraction)

  def test_missing_or_mismatched_progress_uses_existing_coarse_copy(self) -> None:
    scanning = self.operation("backfilling", diagnostic="scanning_routes")
    fallback = self.presentation(scanning, None)
    self.assertEqual(fallback.title, "PROCESSING PRIOR ROUTES")
    self.assertIn("Scanning compatible routes", fallback.detail)
    self.assertIsNone(fallback.progress_fraction)

    operation = self.operation()
    progress = self.progress(backfill_progress_fixture(), operation)
    mismatched = replace(progress, operation_sequence=operation.sequence + 1)
    fallback = self.presentation(operation, mismatched)
    self.assertEqual(
      fallback.detail,
      " | ".join((
        "Prior snapshot shown",
        "Route 2/5",
        "12,345 incorporated",
        "678 excluded",
      )),
    )
    self.assertIsNone(fallback.progress_fraction)

  def test_progress_copy_is_ascii_only(self) -> None:
    operation = self.operation()
    progress = self.progress(
      backfill_progress_fixture(
        approximate_remaining_seconds=120,
      ),
      operation,
    )
    presentation = self.presentation(operation, progress)
    for value in (
      presentation.title,
      presentation.detail,
      presentation.phase_detail,
      presentation.meta,
      presentation.compact_meta,
    ):
      self.assertTrue(value.isascii())

  def test_large_backend_eta_formats_without_float_overflow(self) -> None:
    operation = self.operation()
    progress = self.progress(
      backfill_progress_fixture(
        approximate_remaining_seconds=10**400,
      ),
      operation,
    )
    presentation = self.presentation(operation, progress)
    self.assertTrue(presentation.compact_meta.startswith("25% | About "))
    self.assertTrue(presentation.compact_meta.endswith(" min left"))


def _load_learning_widget_module():
  """Load the status source without requiring raylib in an off-car test."""
  fake_pyray = types.ModuleType("pyray")
  fake_pyray.Color = lambda *channels: tuple(channels)
  fake_pyray.draw_calls = []

  class FakeRectangle:
    def __init__(self, x, y, width, height):
      self.x = x
      self.y = y
      self.width = width
      self.height = height

  class FakeVector2:
    def __init__(self, x, y):
      self.x = x
      self.y = y

  def record_rounded(rect, *args):
    fake_pyray.draw_calls.append(("rounded", rect, args))

  fake_pyray.Rectangle = FakeRectangle
  fake_pyray.Vector2 = FakeVector2
  fake_pyray.fade = lambda color, _alpha: color
  fake_pyray.draw_rectangle_rounded = record_rounded
  fake_pyray.draw_rectangle_rounded_lines_ex = lambda *args: (
    fake_pyray.draw_calls.append(("rounded_lines", args))
  )
  fake_pyray.draw_text_ex = lambda *args: (
    fake_pyray.draw_calls.append(("text", args))
  )

  fake_params = types.ModuleType("openpilot.common.params")
  fake_params.Params = object

  fake_ui_state = types.ModuleType("openpilot.selfdrive.ui.ui_state")
  fake_ui_state.ui_state = types.SimpleNamespace(is_metric=False, CP=None)

  fake_application = types.ModuleType(
    "openpilot.system.ui.lib.application",
  )
  fake_application.FontWeight = types.SimpleNamespace(
    BOLD=0,
    MEDIUM=1,
    NORMAL=2,
  )
  fake_application.gui_app = types.SimpleNamespace(font=lambda _weight: object())

  fake_text_measure = types.ModuleType(
    "openpilot.system.ui.lib.text_measure",
  )
  fake_text_measure.measure_text_cached = lambda _font, text, size: (
    types.SimpleNamespace(x=len(text) * size * 0.5, y=float(size))
  )

  fake_widgets = types.ModuleType("openpilot.system.ui.widgets")
  fake_widgets.Widget = type("Widget", (), {})

  module_name = "_blatv2_learning_widget_status_source_test"
  module_path = (
    Path(__file__).resolve().parents[1]
    / "widgets"
    / "blatv2_learning.py"
  )
  spec = importlib.util.spec_from_file_location(module_name, module_path)
  if spec is None or spec.loader is None:
    raise RuntimeError("could not load BLaTv2 learning widget module")
  module = importlib.util.module_from_spec(spec)
  stubs = {
    "pyray": fake_pyray,
    "openpilot.common.params": fake_params,
    "openpilot.selfdrive.ui.ui_state": fake_ui_state,
    "openpilot.system.ui.lib.application": fake_application,
    "openpilot.system.ui.lib.text_measure": fake_text_measure,
    "openpilot.system.ui.widgets": fake_widgets,
  }
  with patch.dict(sys.modules, stubs):
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
  return module


class _DashboardParams:
  def __init__(
    self,
    values: dict[str, object],
    *,
    failing_key: str | None = None,
  ):
    self.values = values
    self.failing_key = failing_key
    self.calls: list[str] = []

  def get(self, key: str, *, block: bool) -> object:
    self.calls.append(key)
    if key == self.failing_key:
      raise OSError(f"{key} read failed")
    return self.values.get(key)


class TestLearningStatusSource(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.widget_module = _load_learning_widget_module()

  @staticmethod
  def params_values(
    operation: dict | None = None,
    backfill_progress: dict | None = None,
  ) -> dict[str, object]:
    return {
      "BLaTv2LearningStatus": learning_fixture(),
      "BLaTv2LearningOperationStatus": (
        operation_fixture("idle") if operation is None else operation
      ),
      "BLaTv2BackfillProgress": backfill_progress,
      "BLaTv2LifecycleStatus": lifecycle_fixture(),
    }

  def source(self, params: _DashboardParams):
    return self.widget_module.BLaTv2LearningStatusSource(
      params,
      vehicle_identity_provider=lambda: VEHICLE,
      metric_provider=lambda: False,
    )

  def test_authority_validation_regression_renders_as_regressed(self) -> None:
    payload = learning_fixture()
    payload["nodes"][2]["reasons"] = [
      "authority_validation_regression",
    ]
    node = parse_learning_status(
      payload,
      expected_vehicle_identity=VEHICLE,
    ).nodes[2]
    self.assertEqual(
      self.widget_module.BLaTv2ReadinessWidget._fit_text(node),
      "REGRESSED",
    )

  def test_calibration_text_reports_only_observable_values(self) -> None:
    node = parse_learning_status(
      learning_fixture(),
      expected_vehicle_identity=VEHICLE,
    ).nodes[1]
    text = self.widget_module.BLaTv2ReadinessWidget._calibration_text(node)
    self.assertEqual(text, "G .300 O -.040 K/S .030/.090")
    self.assertNotIn("DAMP", text)
    self.assertNotIn("RACK", text)

  def test_calibration_and_lifecycle_identity_namespaces_are_independent(
    self,
  ) -> None:
    values = self.params_values()
    values["BLaTv2LifecycleStatus"] = lifecycle_fixture(
      runtime_hash="f" * 64,
    )
    with patch.object(
      self.widget_module.time,
      "monotonic_ns",
      return_value=NOW_MONO_NS,
    ):
      snapshot = self.source(_DashboardParams(values)).snapshot
    self.assertEqual(snapshot.learning.runtime_identity_sha256, RUNTIME_HASH)
    self.assertEqual(snapshot.lifecycle.runtime_identity_sha256, "f" * 64)
    self.assertIsNone(snapshot.lifecycle_error_code)

  def test_param_read_exceptions_remain_distinct_and_render_red(self) -> None:
    expectations = (
      ("BLaTv2LearningStatus", "learning_error_code"),
      ("BLaTv2LearningOperationStatus", "operation_error_code"),
      ("BLaTv2LifecycleStatus", "lifecycle_error_code"),
    )
    for key, error_attribute in expectations:
      with self.subTest(key=key):
        snapshot = self.source(
          _DashboardParams(self.params_values(), failing_key=key),
        ).snapshot
        self.assertEqual(
          getattr(snapshot, error_attribute),
          "param_read_error",
        )
        if key == "BLaTv2LearningOperationStatus":
          presentation = operation_presentation(
            snapshot.operation,
            error_code=snapshot.operation_error_code,
            error_message=snapshot.operation_error,
            has_learning_snapshot=snapshot.learning is not None,
          )
          self.assertEqual(presentation.tone, "red")
        elif key == "BLaTv2LearningStatus":
          presentation = learning_panel_presentation(
            snapshot.operation,
            operation_error_code=snapshot.operation_error_code,
            operation_error_message=snapshot.operation_error,
            learning_error_code=snapshot.learning_error_code,
            learning_error_message=snapshot.learning_error,
            has_learning_snapshot=False,
          )
          self.assertEqual(presentation.tone, "red")
        else:
          color = self.widget_module._BLaTv2Page._error_color(
            snapshot.lifecycle_error_code,
          )
          self.assertIs(color, self.widget_module._RED)

  def test_optional_progress_failure_falls_back_without_poisoning_operation(
    self,
  ) -> None:
    operation = operation_fixture("backfilling")
    values = self.params_values(
      operation,
      backfill_progress=backfill_progress_fixture(),
    )
    for raw_progress in (
      {"malformed": True},
      backfill_progress_fixture(operation_sequence=operation["sequence"] + 1),
    ):
      values["BLaTv2BackfillProgress"] = raw_progress
      source = self.source(_DashboardParams(values))
      with patch.object(
        self.widget_module.time,
        "monotonic_ns",
        return_value=NOW_MONO_NS,
      ):
        snapshot = source.snapshot
      self.assertIsNotNone(snapshot.operation)
      self.assertIsNone(snapshot.backfill_progress)
      presentation = operation_presentation(
        snapshot.operation,
        error_code=snapshot.operation_error_code,
        error_message=snapshot.operation_error,
        has_learning_snapshot=True,
        backfill_progress=snapshot.backfill_progress,
      )
      self.assertEqual(presentation.title, "PROCESSING PRIOR ROUTES")
      self.assertIsNone(presentation.progress_fraction)

  def test_source_exposes_only_progress_bound_to_the_current_operation(
    self,
  ) -> None:
    operation = operation_fixture("backfilling")
    progress = backfill_progress_fixture(
      operation_sequence=operation["sequence"],
    )
    source = self.source(
      _DashboardParams(self.params_values(operation, progress)),
    )
    with patch.object(
      self.widget_module.time,
      "monotonic_ns",
      return_value=NOW_MONO_NS,
    ):
      snapshot = source.snapshot
    self.assertIsNotNone(snapshot.backfill_progress)
    self.assertEqual(snapshot.backfill_progress.operation_id, OPERATION_ID)

  def test_rejected_operation_update_is_not_exposed_to_the_dashboard(
    self,
  ) -> None:
    operation = operation_fixture("collecting")
    params = _DashboardParams(self.params_values(operation))
    source = self.source(params)
    with patch.object(
      self.widget_module.time,
      "monotonic_ns",
      return_value=NOW_MONO_NS,
    ):
      self.assertIsNotNone(source.snapshot.operation)
      mutated = deepcopy(operation)
      mutated["accepted_sample_count"] += 1
      params.values["BLaTv2LearningOperationStatus"] = mutated
      source._refresh()
    self.assertIsNone(source._snapshot.operation)
    self.assertEqual(source._snapshot.operation_error_code, "stale")

  def test_transient_fallback_does_not_discard_the_validation_watermark(
    self,
  ) -> None:
    operation = operation_fixture("backfilling")
    progress = backfill_progress_fixture()
    params = _DashboardParams(self.params_values(operation, progress))
    source = self.source(params)
    with patch.object(
      self.widget_module.time,
      "monotonic_ns",
      return_value=NOW_MONO_NS,
    ):
      first = source.snapshot
      self.assertIsNotNone(first.backfill_progress)

      params.values["BLaTv2BackfillProgress"] = {"malformed": True}
      source._refresh()
      self.assertIsNone(source._snapshot.backfill_progress)
      self.assertEqual(
        source._last_backfill_progress.completed_work_units,
        progress["completed_work_units"],
      )

      regressed = backfill_progress_fixture()
      regressed["sequence"] = progress["sequence"] + 1
      regressed["updated_mono_ns"] = progress["updated_mono_ns"] + 1
      regressed["completed_work_units"] -= 1
      params.values["BLaTv2BackfillProgress"] = regressed
      source._refresh()
      self.assertIsNone(source._snapshot.backfill_progress)
      self.assertEqual(
        source._snapshot.backfill_progress_error_code,
        "stale",
      )

  def test_monotonic_epoch_is_sampled_after_each_companion_read(self) -> None:
    events: list[str] = []
    operation = operation_fixture("backfilling")
    progress = backfill_progress_fixture()
    params = _DashboardParams(self.params_values(operation, progress))
    original_get = params.get

    def recording_get(key: str, *, block: bool):
      events.append(f"read:{key}")
      return original_get(key, block=block)

    def recording_clock() -> int:
      events.append("clock")
      return NOW_MONO_NS

    params.get = recording_get
    source = self.source(params)
    with patch.object(
      self.widget_module.time,
      "monotonic_ns",
      side_effect=recording_clock,
    ):
      snapshot = source.snapshot
    self.assertIsNotNone(snapshot.backfill_progress)
    self.assertLess(
      events.index("read:BLaTv2LearningOperationStatus"),
      events.index("clock"),
    )
    first_clock = events.index("clock")
    self.assertLess(
      events.index("read:BLaTv2BackfillProgress"),
      events.index("clock", first_clock + 1),
    )

  def test_reader_refreshes_immediately_then_no_faster_than_two_seconds(
    self,
  ) -> None:
    preparing = operation_fixture("preparing")
    params = _DashboardParams(self.params_values(preparing))
    source = self.source(params)

    collecting = operation_fixture("collecting")
    collecting["sequence"] = preparing["sequence"] + 1
    collecting["updated_mono_ns"] = preparing["updated_mono_ns"] + 1

    with (
      patch.object(
        self.widget_module.time,
        "monotonic",
        side_effect=(0.0, 1.999, 2.0),
      ),
      patch.object(
        self.widget_module.time,
        "monotonic_ns",
        return_value=NOW_MONO_NS,
      ),
    ):
      first = source.snapshot
      self.assertEqual(first.operation.state, "preparing")
      self.assertEqual(len(params.calls), 4)

      params.values["BLaTv2LearningOperationStatus"] = collecting
      cached = source.snapshot
      self.assertIs(cached, first)
      self.assertEqual(cached.operation.state, "preparing")
      self.assertEqual(len(params.calls), 4)

      refreshed = source.snapshot
      self.assertIsNot(refreshed, first)
      self.assertEqual(refreshed.operation.state, "collecting")
      self.assertEqual(len(params.calls), 8)


class TestBackfillProgressRendering(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.widget_module = _load_learning_widget_module()

  def snapshot(self, *, has_learning: bool):
    operation = parse_learning_operation_status(
      operation_fixture("backfilling"),
      expected_vehicle_identity=VEHICLE,
      expected_runtime_identity_sha256=RUNTIME_HASH,
      now_mono_ns=NOW_MONO_NS,
    )
    progress = parse_backfill_progress_status(
      backfill_progress_fixture(),
      operation_status=operation,
      now_mono_ns=NOW_MONO_NS,
    )
    learning = (
      parse_learning_status(
        learning_fixture(),
        expected_vehicle_identity=VEHICLE,
      )
      if has_learning
      else None
    )
    return self.widget_module.DashboardSnapshot(
      learning=learning,
      learning_error_code=None if has_learning else "absent",
      learning_error=None if has_learning else "not published",
      operation=operation,
      operation_error_code=None,
      operation_error=None,
      backfill_progress=progress,
      backfill_progress_error_code=None,
      backfill_progress_error=None,
      lifecycle=None,
      lifecycle_error_code="activation_absent",
      lifecycle_error="not published",
      metric=False,
    )

  def test_both_snapshot_pages_draw_the_same_determinate_track(self) -> None:
    snapshot = self.snapshot(has_learning=True)
    rectangle = self.widget_module.rl.Rectangle(0, 0, 1305, 855)
    for page_class in (
      self.widget_module.BLaTv2LearningOverviewWidget,
      self.widget_module.BLaTv2ReadinessWidget,
    ):
      with self.subTest(page=page_class.__name__):
        self.widget_module.rl.draw_calls.clear()
        page = page_class(types.SimpleNamespace(snapshot=snapshot))
        bottom = page._draw_operation_banner(rectangle, 72, snapshot)
        tracks = [
          call[1]
          for call in self.widget_module.rl.draw_calls
          if call[0] == "rounded" and call[1].height == 7
        ]
        self.assertEqual(len(tracks), 2)
        self.assertAlmostEqual(
          tracks[1].width / tracks[0].width,
          snapshot.backfill_progress.progress_fraction,
        )
        self.assertGreater(bottom, 72 + 68)

  def test_initial_no_snapshot_screen_draws_the_determinate_track(self) -> None:
    snapshot = self.snapshot(has_learning=False)
    rectangle = self.widget_module.rl.Rectangle(0, 0, 1305, 855)
    self.widget_module.rl.draw_calls.clear()
    page = self.widget_module.BLaTv2LearningOverviewWidget(
      types.SimpleNamespace(snapshot=snapshot),
    )
    page._draw_unavailable(rectangle, snapshot)
    tracks = [
      call[1]
      for call in self.widget_module.rl.draw_calls
      if call[0] == "rounded" and call[1].height == 12
    ]
    self.assertEqual(len(tracks), 2)
    self.assertAlmostEqual(
      tracks[1].width / tracks[0].width,
      snapshot.backfill_progress.progress_fraction,
    )


class TestLifecycleStatusParser(unittest.TestCase):
  def test_stock_staged_provisional_and_approved_badges(self) -> None:
    expected = {
      "stock": ("STOCK ACTIVE", "stock"),
      "staged": ("STOCK ACTIVE", "stock"),
      "provisional": ("BLATV2 PROVISIONAL", "modular"),
      "approved": ("BLATV2 APPROVED", "modular"),
      "rollback_pending": ("ROLLBACK PENDING | STOCK ACTIVE", "stock"),
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
        "authority_validation_regression",
      )
    }
    self.assertEqual(len(labels), 8)
    self.assertEqual(
      reason_label("authority_validation_regression"),
      "Rejected: authority validation regressed",
    )

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
