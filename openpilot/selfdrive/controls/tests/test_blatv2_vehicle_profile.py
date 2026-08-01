import math
import json
import unittest

from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  CalibrationProfileNode,
  VehicleCalibrationProfile,
  make_calibration_seed_profile,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  DEFAULT_SPEED_NODES_MPS,
  PhysicalParameters,
  ProfileNode,
  VehicleProfile,
  compose_controller_profile,
  make_seed_profile,
)


def params(value: float, qualified: bool = True) -> PhysicalParameters:
  return PhysicalParameters(
    torque_per_lateral_accel=value,
    rack_gain_deg_s2_per_torque=1000.0 + value,
    rack_damping_per_s=5.0 + value,
    transport_delay_s=0.1 + value / 100.0,
    static_friction_torque=0.1,
    kinetic_friction_torque=0.05,
    rack_rate_resolution_deg_s=4.0,
    confidence=0.8,
    qualified=qualified,
  )


def node(speed: float, value: float, qualified: bool = True) -> ProfileNode:
  return ProfileNode(
    speed_mps=speed,
    parameters=params(value, qualified),
    clean_support_s=180.0,
    sample_count=1000,
    validation_count=200,
    validation_rms=0.1,
  )


class TestBlatV2VehicleProfile(unittest.TestCase):
  def test_profile_interpolates_continuously_and_extrapolates_flat(self):
    profile = VehicleProfile("car", 1, "test", (node(0.0, 1.0), node(10.0, 3.0), node(20.0, 5.0)))

    self.assertTrue(math.isclose(profile.parameters_at(-5.0).parameters.torque_per_lateral_accel, 2.0))
    self.assertTrue(math.isclose(profile.parameters_at(5.0).parameters.torque_per_lateral_accel, 2.0))
    self.assertTrue(math.isclose(profile.parameters_at(10.0).parameters.torque_per_lateral_accel, 3.0))
    self.assertTrue(math.isclose(profile.parameters_at(15.0).parameters.torque_per_lateral_accel, 4.0))
    self.assertTrue(math.isclose(profile.parameters_at(25.0).parameters.torque_per_lateral_accel, 5.0))
    self.assertTrue(math.isclose(
      profile.parameters_at(10.0 - 1e-9).parameters.torque_per_lateral_accel,
      profile.parameters_at(10.0 + 1e-9).parameters.torque_per_lateral_accel,
      abs_tol=1e-8,
    ))

  def test_qualification_requires_every_node_and_both_interpolation_neighbors(self):
    profile = VehicleProfile("car", 1, "test", (node(0.0, 1.0), node(10.0, 2.0, False), node(20.0, 3.0)))
    self.assertFalse(profile.qualified)
    self.assertIs(profile.parameters_at(5.0).parameters.qualified, False)
    self.assertIs(profile.parameters_at(15.0).parameters.qualified, False)
    self.assertIs(profile.parameters_at(20.0).parameters.qualified, True)

  def test_seed_is_unqualified_at_every_independent_speed_node(self):
    profile = make_seed_profile("car", 0.4, 1500.0, 8.0, 0.12, 0.09, 0.03)
    self.assertEqual(profile.speed_nodes_mps, DEFAULT_SPEED_NODES_MPS)
    self.assertFalse(profile.qualified)
    self.assertTrue(all(not item.parameters.qualified for item in profile.nodes))
    self.assertTrue(all(item.sample_count == 0 for item in profile.nodes))

  def test_profile_round_trip_is_deterministic_and_vehicle_bound(self):
    profile = VehicleProfile("car", 4, "fit", (node(0.0, 1.0), node(10.0, 2.0)))
    encoded = profile.to_json()
    decoded = VehicleProfile.from_json(encoded, "car")
    self.assertEqual(decoded, profile)
    self.assertEqual(decoded.to_json(), encoded)
    with self.assertRaisesRegex(ValueError, "different vehicle"):
      VehicleProfile.from_json(encoded, "other")

  def test_observable_calibration_composes_exactly_into_controller_profile(self):
    transient = make_seed_profile(
      "car", 0.4, 1500.0, 8.0, 0.12, 0.09, 0.03,
      speed_nodes_mps=(0.0, 10.0),
    )
    seed = make_calibration_seed_profile(
      "car", 0.4, 0.06, 0.12, 4.0,
      speed_nodes_mps=(0.0, 10.0),
    )
    learned_nodes = tuple(
      CalibrationProfileNode(
        speed_mps=source.speed_mps,
        parameters=source.parameters.__class__(
          torque_per_lateral_accel=0.31 + 0.02 * index,
          lateral_accel_offset_correction_mps2=-0.04 + 0.01 * index,
          kinetic_friction_torque=0.02 + 0.01 * index,
          static_breakaway_torque=0.08 + 0.01 * index,
          transport_delay_s=source.parameters.transport_delay_s,
          rack_rate_resolution_deg_s=source.parameters.rack_rate_resolution_deg_s,
          confidence=0.8,
          qualified=True,
        ),
        base_support_s=20.0,
        base_sample_count=200,
        moving_support_s=10.0,
        moving_sample_count=100,
        breakaway_support_s=2.0,
        breakaway_sample_count=20,
        validation_count=50,
        inverse_calibration_validation_rms=0.03,
        breakaway_validation_rms=0.02,
      )
      for index, source in enumerate(seed.nodes)
    )
    calibration = VehicleCalibrationProfile(
      "car", 1, "test calibration", learned_nodes,
    )

    composed = compose_controller_profile(calibration, transient)

    self.assertTrue(composed.qualified)
    self.assertEqual(composed.revision, calibration.revision)
    for calibration_node, transient_node, controller_node in zip(
      calibration.nodes, transient.nodes, composed.nodes, strict=True,
    ):
      self.assertEqual(
        controller_node.parameters.torque_per_lateral_accel,
        calibration_node.parameters.torque_per_lateral_accel,
      )
      self.assertEqual(
        controller_node.parameters.lateral_accel_offset_correction_mps2,
        calibration_node.parameters.lateral_accel_offset_correction_mps2,
      )
      self.assertEqual(
        controller_node.parameters.rack_gain_deg_s2_per_torque,
        transient_node.parameters.rack_gain_deg_s2_per_torque,
      )
      self.assertEqual(controller_node.clean_support_s, 32.0)
      self.assertEqual(controller_node.sample_count, 320)

    for speed in (0.0, 2.5, 5.0, 7.5, 10.0):
      observable = calibration.parameters_at(speed).parameters
      transient_parameters = transient.parameters_at(speed).parameters
      controller = composed.parameters_at(speed).parameters
      self.assertEqual(
        controller.torque_per_lateral_accel,
        observable.torque_per_lateral_accel,
      )
      self.assertEqual(
        controller.lateral_accel_offset_correction_mps2,
        observable.lateral_accel_offset_correction_mps2,
      )
      self.assertEqual(
        controller.kinetic_friction_torque,
        observable.kinetic_friction_torque,
      )
      self.assertEqual(
        controller.static_friction_torque,
        observable.static_breakaway_torque,
      )
      self.assertEqual(
        controller.transport_delay_s,
        observable.transport_delay_s,
      )
      self.assertEqual(
        controller.rack_rate_resolution_deg_s,
        observable.rack_rate_resolution_deg_s,
      )
      self.assertEqual(controller.confidence, observable.confidence)
      self.assertIs(controller.qualified, observable.qualified)
      self.assertEqual(
        controller.rack_gain_deg_s2_per_torque,
        transient_parameters.rack_gain_deg_s2_per_torque,
      )
      self.assertEqual(
        controller.rack_damping_per_s,
        transient_parameters.rack_damping_per_s,
      )

    other = make_seed_profile(
      "other", 0.4, 1500.0, 8.0, 0.12, 0.09, 0.03,
      speed_nodes_mps=(0.0, 10.0),
    )
    with self.assertRaisesRegex(ValueError, "different vehicles"):
      compose_controller_profile(calibration, other)

  def test_old_or_unknown_profile_schema_cannot_seed_replacement(self):
    profile = VehicleProfile("car", 4, "fit", (node(0.0, 1.0), node(10.0, 2.0)))
    payload = profile.to_dict()
    payload["schema_version"] = 5
    with self.assertRaisesRegex(ValueError, "incompatible"):
      VehicleProfile.from_json(payload, "car")

  def test_profile_parser_rejects_alternate_or_lossy_preimages(self):
    profile = VehicleProfile(
      "car", 4, "fit", (node(0.0, 1.0), node(10.0, 2.0)),
    )
    cases: list[tuple[str, object]] = []

    old_schema = profile.to_dict()
    old_schema["schema_version"] = 1
    cases.append(("incompatible", old_schema))

    missing_offset = profile.to_dict()
    missing_offset["nodes"][0]["parameters"].pop(
      "lateral_accel_offset_correction_mps2",
    )
    cases.append(("keys are incompatible", missing_offset))

    unknown_parameter = profile.to_dict()
    unknown_parameter["nodes"][0]["parameters"]["learnedRackGain"] = 1.0
    cases.append(("keys are incompatible", unknown_parameter))

    coerced_qualified = profile.to_dict()
    coerced_qualified["nodes"][0]["parameters"]["qualified"] = 1
    cases.append(("must be a boolean", coerced_qualified))

    coerced_revision = profile.to_dict()
    coerced_revision["revision"] = "4"
    cases.append(("must be an integer", coerced_revision))

    for match, payload in cases:
      with self.subTest(match=match), self.assertRaisesRegex(ValueError, match):
        VehicleProfile.from_json(
          json.dumps(payload, sort_keys=True, separators=(",", ":")),
          "car",
        )

  def test_composition_requires_exact_speed_grid(self):
    transient = make_seed_profile(
      "car", 0.4, 1500.0, 8.0, 0.12, 0.09, 0.03,
      speed_nodes_mps=(0.0, 5.0, 10.0),
    )
    calibration = make_calibration_seed_profile(
      "car", 0.4, 0.06, 0.12, 4.0,
      speed_nodes_mps=(0.0, 10.0),
    )
    with self.assertRaisesRegex(ValueError, "speed grids differ"):
      compose_controller_profile(calibration, transient)

  def test_nonfinite_profile_values_are_rejected(self):
    for bad_value in (math.nan, math.inf, -math.inf):
      with self.subTest(bad_value=bad_value), self.assertRaisesRegex(ValueError, "finite"):
        params(bad_value)
