from dataclasses import replace
import unittest

from openpilot.selfdrive.controls.lib.blatv2.bootstrap import (
  ControllerSelection,
  DriverFeedback,
  ProfileActivationManager,
  ProfileApproval,
  profile_sha256,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  PhysicalParameters,
  ProfileNode,
  VehicleProfile,
)


def profile(revision: int, qualified: bool = True) -> VehicleProfile:
  parameters = PhysicalParameters(
    torque_per_lateral_accel=0.3,
    rack_gain_deg_s2_per_torque=1500.0,
    rack_damping_per_s=8.0,
    transport_delay_s=0.12,
    static_friction_torque=0.09,
    kinetic_friction_torque=0.03,
    rack_rate_resolution_deg_s=4.0,
    confidence=0.9,
    qualified=qualified,
  )
  return VehicleProfile(
    "car",
    revision,
    "test",
    (
      ProfileNode(0.0, parameters, 180.0, 1000, 200, 0.1),
      ProfileNode(30.0, parameters, 600.0, 1000, 200, 0.1),
    ),
  )


def approval(candidate: VehicleProfile, **changes) -> ProfileApproval:
  base = ProfileApproval(
    profile_sha256(candidate),
    "source",
    "opendbc",
    True,
    True,
    True,
  )
  return replace(base, **changes)


class TestBlatV2Bootstrap(unittest.TestCase):
  def test_stock_is_exact_default_until_complete_profile_is_approved(self):
    manager = ProfileActivationManager("car")
    decision = manager.begin_engagement()
    self.assertEqual(decision.selection, ControllerSelection.STOCK)
    self.assertIsNone(decision.profile)

  def test_incomplete_or_failed_candidate_cannot_stage(self):
    manager = ProfileActivationManager("car")
    incomplete = profile(1, False)
    with self.assertRaisesRegex(ValueError, "Incomplete|incomplete"):
      manager.stage_candidate(incomplete, approval(incomplete), onroad=False)
    complete = profile(1)
    with self.assertRaisesRegex(ValueError, "did not all pass"):
      manager.stage_candidate(
        complete,
        approval(complete, replay_passed=False),
        onroad=False,
      )

  def test_profile_hash_binds_approval_and_prevents_mutation_after_gate(self):
    manager = ProfileActivationManager("car")
    candidate = profile(1)
    wrong = profile(2)
    with self.assertRaisesRegex(ValueError, "exact profile"):
      manager.stage_candidate(candidate, approval(wrong), onroad=False)

  def test_staged_profile_switches_only_at_engagement_boundary(self):
    manager = ProfileActivationManager("car")
    candidate = profile(1)
    manager.stage_candidate(candidate, approval(candidate), onroad=False)
    self.assertIsNone(manager.active_profile)
    decision = manager.begin_engagement()
    self.assertEqual(decision.selection, ControllerSelection.MODULAR)
    self.assertEqual(decision.profile, candidate)
    self.assertTrue(decision.provisional)
    with self.assertRaisesRegex(RuntimeError, "offroad"):
      manager.stage_candidate(profile(2), approval(profile(2)), onroad=True)

  def test_worse_feedback_rolls_back_to_stock_on_next_engagement(self):
    manager = ProfileActivationManager("car")
    candidate = profile(1)
    manager.stage_candidate(candidate, approval(candidate), onroad=False)
    manager.begin_engagement()
    manager.end_engagement()
    manager.record_feedback(DriverFeedback.WORSE, offroad=True)
    # Active evidence is preserved until the next explicit boundary.
    self.assertEqual(manager.active_profile, candidate)
    decision = manager.begin_engagement()
    self.assertEqual(decision.selection, ControllerSelection.STOCK)
    self.assertIsNone(decision.profile)

  def test_better_accepts_without_waiving_pre_activation_gates(self):
    manager = ProfileActivationManager("car")
    candidate = profile(1)
    manager.stage_candidate(candidate, approval(candidate), onroad=False)
    manager.begin_engagement()
    manager.end_engagement()
    manager.record_feedback(DriverFeedback.BETTER, offroad=True)
    decision = manager.begin_engagement()
    self.assertEqual(decision.selection, ControllerSelection.MODULAR)
    self.assertFalse(decision.provisional)

  def test_not_sure_keeps_profile_provisional_and_vehicle_identity_is_strict(self):
    manager = ProfileActivationManager("car")
    other = replace(profile(1), vehicle_identity="other")
    with self.assertRaisesRegex(ValueError, "different vehicle"):
      manager.stage_candidate(other, approval(other), onroad=False)
    candidate = profile(1)
    manager.stage_candidate(candidate, approval(candidate), onroad=False)
    manager.begin_engagement()
    manager.end_engagement()
    manager.record_feedback(DriverFeedback.NOT_SURE, offroad=True)
    self.assertTrue(manager.begin_engagement().provisional)
