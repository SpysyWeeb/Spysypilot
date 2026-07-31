import math
from types import SimpleNamespace
import unittest

from openpilot.selfdrive.controls.lib.longitudinal_lead import (
  LeadObservation,
  closing_decel_requirement,
  time_to_collision,
  total_decel_requirement,
)


def radar_lead(**overrides):
  values = {
    "present": True,
    "dRel": 12.0,
    "vLeadK": 4.0,
    "aLeadK": 0.0,
    "modelProb": 0.9,
  }
  values.update(overrides)
  return SimpleNamespace(**values)


class TestLeadObservation(unittest.TestCase):
  def test_requires_live_valid_service(self):
    self.assertFalse(LeadObservation.from_radar(radar_lead(), False).present)
    self.assertFalse(LeadObservation.from_radar(radar_lead(present=False), True).present)

  def test_rejects_nonfinite_or_impossible_values(self):
    self.assertFalse(LeadObservation.from_radar(radar_lead(dRel=math.nan), True).present)
    self.assertFalse(LeadObservation.from_radar(radar_lead(dRel=0.0), True).present)
    self.assertFalse(LeadObservation.from_radar(radar_lead(vLeadK=math.inf), True).present)

  def test_sanitizes_filtered_values(self):
    lead = LeadObservation.from_radar(
      radar_lead(vLeadK=-0.2, aLeadK=-20.0, modelProb=1.4),
      True,
    )
    self.assertTrue(lead.present)
    self.assertEqual(lead.speed, 0.0)
    self.assertEqual(lead.acceleration, -10.0)
    self.assertEqual(lead.model_prob, 1.0)


class TestRelativeLeadPhysics(unittest.TestCase):
  def test_equal_speed_has_no_closing_requirement(self):
    lead = LeadObservation(True, distance=2.9, speed=0.6)
    self.assertEqual(closing_decel_requirement(0.6, lead, 2.5, 0.15), 0.0)

  def test_stopped_lead_matches_v_squared_over_two_d(self):
    lead = LeadObservation(True, distance=6.5, speed=0.0)
    self.assertAlmostEqual(
      closing_decel_requirement(2.0, lead, 2.5, 0.15),
      0.5,
    )

  def test_total_requirement_includes_lead_braking(self):
    lead = LeadObservation(True, distance=14.0, speed=8.0, acceleration=-1.0)
    closing_only = closing_decel_requirement(10.0, lead, 4.0, 1.0)
    self.assertAlmostEqual(
      total_decel_requirement(10.0, lead, 4.0, 1.0),
      1.0 + closing_only,
    )

  def test_ttc_ignores_nonclosing_lead(self):
    lead = LeadObservation(True, distance=10.0, speed=10.0)
    self.assertTrue(math.isinf(time_to_collision(9.0, lead)))
    self.assertAlmostEqual(time_to_collision(12.0, lead), 5.0)


if __name__ == "__main__":
  unittest.main()
