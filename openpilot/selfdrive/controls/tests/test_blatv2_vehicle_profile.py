import math
import unittest

from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  DEFAULT_SPEED_NODES_MPS,
  PhysicalParameters,
  ProfileNode,
  VehicleProfile,
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

  def test_old_or_unknown_profile_schema_cannot_seed_replacement(self):
    profile = VehicleProfile("car", 4, "fit", (node(0.0, 1.0), node(10.0, 2.0)))
    payload = profile.to_dict()
    payload["schema_version"] = 5
    with self.assertRaisesRegex(ValueError, "incompatible"):
      VehicleProfile.from_json(payload, "car")

  def test_nonfinite_profile_values_are_rejected(self):
    for bad_value in (math.nan, math.inf, -math.inf):
      with self.subTest(bad_value=bad_value), self.assertRaisesRegex(ValueError, "finite"):
        params(bad_value)
