import json
import math
import unittest

from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  CALIBRATION_PROFILE_PARAM_KEY,
  CALIBRATION_PROFILE_SCHEMA_VERSION,
  DEFAULT_SPEED_NODES_MPS,
  CalibrationParameters,
  CalibrationProfileNode,
  VehicleCalibrationProfile,
  make_calibration_seed_profile,
)


def parameters(value: float, *, qualified: bool = True) -> CalibrationParameters:
  return CalibrationParameters(
    torque_per_lateral_accel=value,
    lateral_accel_offset_correction_mps2=-0.2 + value / 10.0,
    kinetic_friction_torque=0.03 + value / 100.0,
    static_breakaway_torque=0.08 + value / 100.0,
    transport_delay_s=0.10 + value / 100.0,
    rack_rate_resolution_deg_s=3.0 + value,
    confidence=0.7 + value / 100.0,
    qualified=qualified,
  )


def node(speed: float, value: float, *, qualified: bool = True) -> CalibrationProfileNode:
  return CalibrationProfileNode(
    speed_mps=speed,
    parameters=parameters(value, qualified=qualified),
    base_support_s=100.0 + speed,
    base_sample_count=1000 + int(speed),
    moving_support_s=50.0 + speed,
    moving_sample_count=500 + int(speed),
    breakaway_support_s=10.0 + speed,
    breakaway_sample_count=20 + int(speed),
    validation_count=100 + int(speed),
    inverse_calibration_validation_rms=0.02 + speed / 1000.0,
    breakaway_validation_rms=0.03 + speed / 1000.0,
  )


class TestBlatV2CalibrationProfile(unittest.TestCase):
  def test_round_trip_is_deterministic_and_current_schema_is_strict(self):
    profile = VehicleCalibrationProfile(
      vehicle_identity="palisade",
      revision=7,
      provenance="held-out route fit",
      nodes=(node(0.0, 0.3), node(10.0, 0.5)),
    )

    encoded = profile.to_json()
    decoded = VehicleCalibrationProfile.from_json(encoded, "palisade")
    self.assertEqual(decoded, profile)
    self.assertEqual(decoded.to_json(), encoded)
    self.assertEqual(json.loads(encoded)["schema_version"], CALIBRATION_PROFILE_SCHEMA_VERSION)
    self.assertEqual(CALIBRATION_PROFILE_PARAM_KEY, "BLaTv2ObservableCalibrationProfile")

    payload = profile.to_dict()
    payload["unexpected"] = True
    with self.assertRaisesRegex(ValueError, "keys are incompatible"):
      VehicleCalibrationProfile.from_json(payload, "palisade")

  def test_interpolation_is_linear_flat_at_ends_and_qualification_uses_bounding_nodes(self):
    profile = VehicleCalibrationProfile(
      "car",
      1,
      "test",
      (node(0.0, 0.2), node(10.0, 0.4, qualified=False), node(20.0, 0.8)),
    )

    below = profile.parameters_at(0.0)
    middle = profile.parameters_at(5.0)
    upper_middle = profile.parameters_at(15.0)
    above = profile.parameters_at(100.0)
    self.assertEqual(below.parameters, profile.nodes[0].parameters)
    self.assertTrue(math.isclose(middle.parameters.torque_per_lateral_accel, 0.3))
    self.assertTrue(math.isclose(middle.parameters.lateral_accel_offset_correction_mps2, -0.17))
    self.assertFalse(middle.parameters.qualified)
    self.assertFalse(upper_middle.parameters.qualified)
    self.assertTrue(above.parameters.qualified)
    self.assertEqual(above.parameters, profile.nodes[-1].parameters)
    self.assertEqual((middle.lower_node, middle.upper_node, middle.upper_weight), (0, 1, 0.5))

  def test_wrong_schema_and_vehicle_are_rejected(self):
    profile = VehicleCalibrationProfile("car", 1, "test", (node(0.0, 0.2), node(10.0, 0.4)))
    payload = profile.to_dict()
    payload["schema_version"] = 1
    with self.assertRaisesRegex(ValueError, "schema is incompatible"):
      VehicleCalibrationProfile.from_json(payload, "car")
    with self.assertRaisesRegex(ValueError, "different vehicle"):
      VehicleCalibrationProfile.from_json(profile.to_json(), "other")

  def test_kinetic_friction_cannot_exceed_static_breakaway(self):
    with self.assertRaisesRegex(ValueError, "kinetic friction cannot exceed static breakaway"):
      CalibrationParameters(
        torque_per_lateral_accel=0.4,
        lateral_accel_offset_correction_mps2=0.0,
        kinetic_friction_torque=0.10,
        static_breakaway_torque=0.09,
        transport_delay_s=0.12,
        rack_rate_resolution_deg_s=4.0,
        confidence=0.5,
        qualified=False,
      )

  def test_seed_copies_stock_normalized_friction_without_slope_multiplication(self):
    slope = 0.4
    stock_friction_torque = 0.12
    profile = make_calibration_seed_profile(
      "car",
      torque_callback_slope=slope,
      stock_friction_torque=stock_friction_torque,
      transport_delay_s=0.11,
      rack_rate_resolution_deg_s=4.0,
      lateral_accel_offset_correction_mps2=-0.02,
    )

    self.assertEqual(profile.speed_nodes_mps, DEFAULT_SPEED_NODES_MPS)
    self.assertFalse(profile.qualified)
    self.assertTrue(all(not item.parameters.qualified for item in profile.nodes))
    self.assertTrue(all(item.parameters.torque_per_lateral_accel == slope for item in profile.nodes))
    self.assertTrue(all(item.parameters.kinetic_friction_torque == stock_friction_torque for item in profile.nodes))
    self.assertTrue(all(item.parameters.static_breakaway_torque == stock_friction_torque for item in profile.nodes))
    self.assertNotEqual(stock_friction_torque, slope * stock_friction_torque)

    # Stock get_friction enters lateral-acceleration space by multiplying by
    # latAccelFactor=(1/slope); the callback multiplies by slope and therefore
    # recovers the original normalized-torque friction exactly.
    stock_lat_accel_friction = stock_friction_torque / slope
    self.assertTrue(math.isclose(slope * stock_lat_accel_friction, stock_friction_torque))

  def test_schema_has_no_unobservable_rack_gain_or_damping_fields(self):
    profile = make_calibration_seed_profile("car", 0.4, 0.1, 0.12, 4.0)
    parameter_payload = profile.to_dict()["nodes"][0]["parameters"]
    self.assertNotIn("rack_gain_deg_s2_per_torque", parameter_payload)
    self.assertNotIn("rack_damping_per_s", parameter_payload)
    self.assertFalse(hasattr(profile.nodes[0].parameters, "rack_gain_deg_s2_per_torque"))
    self.assertFalse(hasattr(profile.nodes[0].parameters, "rack_damping_per_s"))


if __name__ == "__main__":
  unittest.main()
