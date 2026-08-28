import math
import unittest

from openpilot.selfdrive.controls.lib.blatv2.plant import (
  RackState,
  TrackingPolicy,
  compute_inverse_torque,
  departure_friction_torque,
  predict_applied_history,
  presliding_friction_magnitude,
  steady_road_load_torque,
  step_rack_dynamics,
  step_plant,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
  curvature_from_measured_angle,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  PhysicalParameters,
)


def mapping() -> RackMappingSnapshot:
  return RackMappingSnapshot(
    mass_kg=2100.0,
    wheelbase_m=2.9,
    center_to_front_m=1.2,
    center_to_rear_m=1.7,
    tire_stiffness_front=100000.0,
    tire_stiffness_rear=110000.0,
    steer_ratio_rear=0.0,
    steer_ratio=15.0,
    roll_rad=0.0,
    angle_offset_deg=0.0,
    valid=True,
  )


def parameters() -> PhysicalParameters:
  return PhysicalParameters(
    torque_per_lateral_accel=0.3,
    rack_gain_deg_s2_per_torque=1500.0,
    rack_damping_per_s=8.0,
    transport_delay_s=0.12,
    static_friction_torque=0.09,
    kinetic_friction_torque=0.03,
    rack_rate_resolution_deg_s=4.0,
    confidence=1.0,
    qualified=True,
  )


class TestBlatV2Plant(unittest.TestCase):
  def test_steady_inverse_load_holds_rack(self):
    rack_mapping = mapping()
    physical_parameters = parameters()
    state = RackState(angle_deg=-30.0, rate_deg_s=0.0, applied_torque=0.0)
    from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import curvature_from_measured_angle
    curvature = curvature_from_measured_angle(state.angle_deg, 10.0, rack_mapping, rack_mapping).curvature
    load = steady_road_load_torque(curvature, 10.0, 0.0, 0.0, physical_parameters.torque_per_lateral_accel)
    result = step_plant(state, load, 10.0, rack_mapping, rack_mapping, 0.0, physical_parameters, 0.0, 0.01)
    self.assertTrue(result.stuck)
    self.assertEqual(result.state.angle_deg, state.angle_deg)
    self.assertEqual(result.state.rate_deg_s, 0.0)

  def test_static_load_holds_then_breaks_away(self):
    rack_mapping = mapping()
    physical_parameters = parameters()
    state = RackState(0.0, 0.0, 0.0)
    held = step_plant(state, 0.08, 0.0, rack_mapping, rack_mapping, 0.0, physical_parameters, 0.0, 0.01)
    moving = step_plant(state, 0.10, 0.0, rack_mapping, rack_mapping, 0.0, physical_parameters, 0.0, 0.01)
    self.assertTrue(held.stuck)
    self.assertEqual(held.state.rate_deg_s, 0.0)
    self.assertFalse(moving.stuck)
    self.assertGreater(moving.state.rate_deg_s, 0.0)

  def test_friction_can_stop_but_never_reverse_a_coasting_rack(self):
    rack_mapping = mapping()
    physical_parameters = parameters()
    state = RackState(0.0, 0.1, 0.0)
    stopped = step_plant(
      state, 0.0, 0.0, rack_mapping, rack_mapping, 0.0, physical_parameters, 0.0, 0.01,
    )
    self.assertTrue(stopped.stuck)
    self.assertEqual(stopped.state.rate_deg_s, 0.0)

    held = step_plant(
      stopped.state, 0.0, 0.0, rack_mapping, rack_mapping, 0.0,
      physical_parameters, 0.0, 0.01,
    )
    self.assertTrue(held.stuck)
    self.assertEqual(held.state.rate_deg_s, 0.0)
    self.assertEqual(held.state.angle_deg, stopped.state.angle_deg)

  def test_external_load_above_breakaway_can_reverse_rack(self):
    rack_mapping = mapping()
    physical_parameters = parameters()
    state = RackState(0.0, 0.1, 0.0)
    reversed_step = step_plant(
      state, -0.2, 0.0, rack_mapping, rack_mapping, 0.0, physical_parameters, 0.0, 0.01,
    )
    self.assertFalse(reversed_step.stuck)
    self.assertLess(reversed_step.state.rate_deg_s, 0.0)

  def test_presliding_transition_is_continuous_and_physical(self):
    physical_parameters = parameters()
    self.assertEqual(presliding_friction_magnitude(0.0, physical_parameters), 0.09)
    self.assertAlmostEqual(presliding_friction_magnitude(2.0, physical_parameters), 0.06)
    self.assertEqual(presliding_friction_magnitude(4.0, physical_parameters), 0.03)
    self.assertEqual(presliding_friction_magnitude(20.0, physical_parameters), 0.03)
    self.assertEqual(departure_friction_torque(0.0, -1.0, physical_parameters), -0.09)

  def test_inverse_terms_reproduce_requested_acceleration(self):
    physical_parameters = parameters()
    state = RackState(angle_deg=5.0, rate_deg_s=2.0, applied_torque=0.0)
    policy = TrackingPolicy(natural_frequency_per_s=6.0)
    terms = compute_inverse_torque(
      state,
      desired_curvature=-0.001,
      desired_angle_deg=6.0,
      desired_rate_deg_s=3.0,
      desired_acceleration_deg_s2=4.0,
      speed_mps=8.0,
      roll_rad=0.0,
      lateral_accel_offset=0.0,
      parameters=physical_parameters,
      policy=policy,
      disturbance_torque=0.02,
    )
    expected_required = 4.0 + 6.0**2 * 1.0 + 2.0 * 6.0 * 1.0
    self.assertEqual(terms.required_acceleration_deg_s2, expected_required)
    self.assertAlmostEqual(
      terms.raw_torque,
      terms.aligning_torque
      + terms.friction_torque
      + terms.motion_feedforward_torque
      + terms.position_feedback_torque
      + terms.rate_feedback_torque
      + terms.disturbance_torque,
    )

  def test_zero_error_constant_reference_requests_only_steady_load(self):
    physical_parameters = parameters()
    state = RackState(0.0, 0.0, 0.0)
    terms = compute_inverse_torque(
      state, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0,
      physical_parameters, TrackingPolicy(6.0), 0.0,
    )
    self.assertEqual(terms.raw_torque, 0.0)
    self.assertEqual(terms.friction_torque, 0.0)
    self.assertEqual(terms.motion_feedforward_torque, 0.0)
    self.assertEqual(terms.position_feedback_torque, 0.0)
    self.assertEqual(terms.rate_feedback_torque, 0.0)

  def test_tracking_law_is_speed_invariant_in_angle_space(self):
    physical_parameters = parameters()
    policy = TrackingPolicy(6.0)
    state = RackState(0.0, 0.0, 0.0)
    low = compute_inverse_torque(state, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, physical_parameters, policy, 0.0)
    high = compute_inverse_torque(state, 0.0, 1.0, 0.0, 0.0, 25.0, 0.0, 0.0, physical_parameters, policy, 0.0)
    self.assertEqual(low.position_feedback_torque, high.position_feedback_torque)

  def test_applied_history_prediction_is_deterministic(self):
    rack_mapping = mapping()
    physical_parameters = parameters()
    state = RackState(0.0, 0.0, 0.0)
    commands = (0.0, 0.1, 0.2, 0.3)
    first = predict_applied_history(state, commands, 5.0, rack_mapping, rack_mapping, 0.0, physical_parameters, 0.0, 0.01)
    second = predict_applied_history(state, commands, 5.0, rack_mapping, rack_mapping, 0.0, physical_parameters, 0.0, 0.01)
    self.assertEqual(first, second)
    self.assertGreater(first.rate_deg_s, 0.0)

  def test_step_plant_delegates_without_changing_transient_semantics(self):
    rack_mapping = mapping()
    physical_parameters = parameters()
    state = RackState(-12.0, 3.0, 0.1)
    curvature = curvature_from_measured_angle(
      state.angle_deg, 9.0, rack_mapping, rack_mapping,
    ).curvature
    aligning = steady_road_load_torque(
      curvature, 9.0, rack_mapping.roll_rad, 0.0,
      physical_parameters.torque_per_lateral_accel,
    )
    wrapped = step_plant(
      state, -0.4, 9.0, rack_mapping, rack_mapping, 0.0,
      physical_parameters, 0.02, 0.01,
    )
    direct = step_rack_dynamics(
      state, -0.4, aligning, physical_parameters, 0.02, 0.01,
    )
    self.assertEqual(wrapped, direct)

  def test_nonfinite_inverse_input_rejected(self):
    physical_parameters = parameters()
    for bad in (math.nan, math.inf, -math.inf):
      with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "finite"):
        compute_inverse_torque(
          RackState(0.0, 0.0, 0.0), 0.0, bad, 0.0, 0.0,
          5.0, 0.0, 0.0, physical_parameters, TrackingPolicy(6.0), 0.0,
        )
