import unittest

from openpilot.selfdrive.controls.lib.blatv2.counterfactual_plant import (
  AppliedTorqueDelayLine,
  CounterfactualPlantMember,
  step_counterfactual_plant,
)
from openpilot.selfdrive.controls.lib.blatv2.plant import RackState
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import RackMappingSnapshot
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import PhysicalParameters


def member(**overrides) -> CounterfactualPlantMember:
  values = {
    "rack_gain_deg_s2_per_torque": 2000.0,
    "rack_damping_per_s": 10.0,
    "delay_offset_s": 0.02,
    "unresolved_load_torque": 0.03,
  }
  values.update(overrides)
  return CounterfactualPlantMember.create(**values)


def parameters() -> PhysicalParameters:
  return PhysicalParameters(
    torque_per_lateral_accel=0.3,
    rack_gain_deg_s2_per_torque=4000.0,
    rack_damping_per_s=8.0,
    transport_delay_s=0.1,
    static_friction_torque=0.09,
    kinetic_friction_torque=0.03,
    rack_rate_resolution_deg_s=4.0,
    confidence=1.0,
    qualified=True,
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


class TestCounterfactualPlant(unittest.TestCase):
  def test_member_changes_only_transient_parameters(self):
    base = parameters()
    selected = member().parameters_for(base)
    self.assertEqual(selected.rack_gain_deg_s2_per_torque, 2000.0)
    self.assertEqual(selected.rack_damping_per_s, 10.0)
    self.assertEqual(selected.torque_per_lateral_accel, base.torque_per_lateral_accel)
    self.assertEqual(selected.static_friction_torque, base.static_friction_torque)
    self.assertEqual(selected.transport_delay_s, base.transport_delay_s)
    self.assertAlmostEqual(member().effective_delay_s(base.transport_delay_s), 0.12)

  def test_zero_delay_uses_new_can_torque(self):
    history = AppliedTorqueDelayLine(fixed_dt_s=0.01, maximum_delay_s=0.02)
    history.reset(-0.2)
    self.assertEqual(history.latest_can_applied_torque, -0.2)
    self.assertEqual(history.commit_and_sample(0.4, 0.0), 0.4)
    self.assertEqual(history.latest_can_applied_torque, 0.4)
    self.assertEqual(history.latest_rack_effective_torque, 0.4)

  def test_positive_subframe_delay_uses_previous_torque(self):
    history = AppliedTorqueDelayLine(fixed_dt_s=0.01, maximum_delay_s=0.02)
    history.reset(0.1)
    self.assertEqual(history.commit_and_sample(0.2, 0.005), 0.1)
    self.assertEqual(history.commit_and_sample(0.3, 0.005), 0.2)

  def test_integer_delay_and_bootstrap_are_exact(self):
    history = AppliedTorqueDelayLine(fixed_dt_s=0.01, maximum_delay_s=0.02)
    history.reset(-0.4)
    self.assertEqual(history.commit_and_sample(0.1, 0.02), -0.4)
    self.assertEqual(history.commit_and_sample(0.2, 0.02), -0.4)
    self.assertEqual(history.commit_and_sample(0.3, 0.02), 0.1)

  def test_dynamic_delay_resamples_history_without_rewriting_it(self):
    history = AppliedTorqueDelayLine(fixed_dt_s=0.01, maximum_delay_s=0.02)
    history.reset(0.0)
    self.assertEqual(history.commit_and_sample(0.1, 0.0), 0.1)
    self.assertEqual(history.commit_and_sample(0.2, 0.01), 0.1)
    self.assertEqual(history.commit_and_sample(0.3, 0.02), 0.1)
    self.assertEqual(history.commit_and_sample(0.4, 0.01), 0.3)

  def test_controller_anchor_is_can_torque_not_rack_effective_torque(self):
    history = AppliedTorqueDelayLine(fixed_dt_s=0.01, maximum_delay_s=0.02)
    history.reset(0.0)
    effective = history.commit_and_sample(0.5, 0.02)
    self.assertEqual(effective, 0.0)
    self.assertEqual(history.latest_can_applied_torque, 0.5)
    self.assertNotEqual(history.latest_can_applied_torque, effective)

  def test_member_load_and_dynamics_drive_plant(self):
    selected_member = member(
      rack_gain_deg_s2_per_torque=1000.0,
      rack_damping_per_s=5.0,
      unresolved_load_torque=0.1,
    )
    result = step_counterfactual_plant(
      state=RackState(0.0, 0.0, 0.0),
      rack_effective_torque=0.2,
      speed_mps=0.0,
      mapping=mapping(),
      nominal_mapping=mapping(),
      lateral_accel_offset=0.0,
      base_parameters=parameters(),
      member=selected_member,
      dt=0.01,
    )
    # 0.2 applied - 0.1 unresolved load - 0.09 static breakaway leaves
    # 0.01 effective torque at a 1000 deg/s^2/torque member gain.
    self.assertAlmostEqual(result.acceleration_deg_s2, 10.0)
    self.assertAlmostEqual(result.state.rate_deg_s, 0.1)
    self.assertEqual(result.state.applied_torque, 0.2)

  def test_invalid_member_and_delay_fail_closed(self):
    with self.assertRaisesRegex(ValueError, "member"):
      CounterfactualPlantMember(
        member_id="1" * 64,
        rack_gain_deg_s2_per_torque=2000.0,
        rack_damping_per_s=10.0,
        delay_offset_s=0.02,
        unresolved_load_torque=0.03,
      )
    with self.assertRaisesRegex(ValueError, "delay"):
      member(delay_offset_s=-0.2).effective_delay_s(0.1)
    history = AppliedTorqueDelayLine(fixed_dt_s=0.01, maximum_delay_s=0.02)
    with self.assertRaisesRegex(RuntimeError, "primed"):
      history.commit_and_sample(0.1, 0.0)
    history.reset(0.0)
    with self.assertRaisesRegex(ValueError, "exceeds"):
      history.commit_and_sample(0.1, 0.03)


if __name__ == "__main__":
  unittest.main()
