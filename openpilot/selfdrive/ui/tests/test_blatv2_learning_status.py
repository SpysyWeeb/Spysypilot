from copy import deepcopy
import unittest

from openpilot.selfdrive.ui.widgets.blatv2_learning_status import (
  LearningStatusError,
  cycle_page_index,
  format_duration,
  format_speed,
  grid_cells,
  parse_learning_status,
  parse_lifecycle_status,
  reason_label,
  select_value_provider,
)


VEHICLE = "HYUNDAI PALISADE 2020"
RUNTIME_HASH = "1" * 64
HASH = "2" * 64
COMMIT = "3" * 40


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
