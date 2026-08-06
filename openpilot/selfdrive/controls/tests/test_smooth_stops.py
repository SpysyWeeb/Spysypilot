import unittest

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longitudinal_lead import LeadObservation
from openpilot.selfdrive.controls.lib.smooth_stops import (
  HOLD_RELEASE_FRAMES,
  LEAD_DROPOUT_GRACE,
  SETTLE_JERK,
  SmoothStopController,
)


def run_settle(controller, seconds, *, v_ego, a_target=-0.05, last_output=-0.15, lead=None):
  output = last_output
  for _ in range(round(seconds / DT_CTRL)):
    output = controller.settle(a_target, v_ego, output, lead)
  return output


class TestSmoothStopHandoff(unittest.TestCase):
  def test_hold_only_arms_at_true_standstill(self):
    controller = SmoothStopController()
    self.assertFalse(controller.want_hold(True, 0.16, True))
    self.assertTrue(controller.want_hold(True, 0.14, True))
    self.assertTrue(controller.want_hold(True, 0.04, False))
    self.assertFalse(controller.want_hold(False, 0.0, True))

  def test_hold_release_is_consecutive_and_debounced(self):
    controller = SmoothStopController()
    controller.arm_hold()
    for _ in range(HOLD_RELEASE_FRAMES - 1):
      self.assertFalse(controller.hold_release(False))
    self.assertTrue(controller.hold_release(False))
    self.assertFalse(controller.hold_release(True))


class TestSmoothStopSettle(unittest.TestCase):
  def test_entry_is_continuous(self):
    controller = SmoothStopController()
    output = controller.settle(-0.05, 1.0, -0.4)
    self.assertLessEqual(abs(output + 0.4), SETTLE_JERK * DT_CTRL + 1e-9)

  def test_settle_pressure_respects_jerk_limit(self):
    controller = SmoothStopController()
    previous = controller.settle(-0.05, 1.0, -0.4)
    output = controller.settle(-0.05, 0.5, previous)
    self.assertLessEqual(abs(output - previous), SETTLE_JERK * DT_CTRL + 1e-9)

  def test_ordinary_plan_braking_is_profiled(self):
    controller = SmoothStopController()
    output = controller.settle(-3.0, 15.0, 0.0)
    self.assertAlmostEqual(output, -SETTLE_JERK * DT_CTRL)

  def test_emergency_plan_braking_passes_through(self):
    controller = SmoothStopController()
    self.assertEqual(controller.settle(-3.0, 15.0, -0.1, emergency=True), -3.0)

  def test_limo_profile_reaches_plateau_then_releases(self):
    controller = SmoothStopController()
    lead = LeadObservation(True, distance=100.0, speed=15.0)
    output = 0.0
    high_speed_outputs = []
    for _ in range(80):
      output = controller.settle(-1.0, 15.0, output, lead)
      high_speed_outputs.append(output)
    self.assertAlmostEqual(high_speed_outputs[-1], -1.0, places=6)

    release_outputs = []
    for speed in (4.0, 3.0, 2.0, 1.0, 0.3, 0.15):
      for _ in range(20):
        output = controller.settle(-1.0, speed, output, lead)
      release_outputs.append(output)
    self.assertTrue(all(a <= b for a, b in zip(release_outputs, release_outputs[1:], strict=False)))
    self.assertAlmostEqual(release_outputs[-1], -0.12, places=6)

  def test_stationary_vehicle_creep_ratchets_firmer(self):
    output = run_settle(SmoothStopController(), 2.0, v_ego=0.6)
    self.assertLess(output, -0.8)

  def test_continuously_moving_lead_does_not_ratchet(self):
    lead = LeadObservation(True, distance=8.0, speed=0.31)
    output = run_settle(SmoothStopController(), 10.0, v_ego=0.6, lead=lead)
    self.assertGreater(output, -0.3)

  def test_moving_threshold_noise_has_hysteresis(self):
    controller = SmoothStopController()
    output = -0.15
    for frame in range(round(10.0 / DT_CTRL)):
      lead_speed = 0.31 if frame % 2 else 0.29
      lead = LeadObservation(True, distance=8.0, speed=lead_speed)
      output = controller.settle(-0.05, 0.6, output, lead)
    self.assertGreater(output, -0.3)

  def test_short_radar_dropout_does_not_add_permanent_brake(self):
    controller = SmoothStopController()
    moving = LeadObservation(True, distance=8.0, speed=0.4)
    output = run_settle(controller, 1.0, v_ego=0.6, lead=moving)
    output = run_settle(
      controller,
      LEAD_DROPOUT_GRACE,
      v_ego=0.6,
      last_output=output,
      lead=LeadObservation(),
    )
    output = run_settle(controller, 1.0, v_ego=0.6, last_output=output, lead=moving)
    self.assertGreater(output, -0.3)

  def test_stopped_then_moving_lead_releases_ratchet_smoothly(self):
    controller = SmoothStopController()
    stopped = LeadObservation(True, distance=5.0, speed=0.0)
    output = run_settle(controller, 2.0, v_ego=0.6, lead=stopped)
    self.assertLess(output, -0.8)

    moving = LeadObservation(True, distance=5.0, speed=0.5)
    output = run_settle(controller, 2.0, v_ego=0.6, last_output=output, lead=moving)
    self.assertGreater(output, -0.3)

  def test_close_equal_speed_lead_uses_relative_motion(self):
    controller = SmoothStopController()
    lead = LeadObservation(True, distance=2.9, speed=0.6)
    output = controller.settle(-0.05, 0.6, -0.15, lead)
    self.assertGreater(output, -0.2)


if __name__ == "__main__":
  unittest.main()
