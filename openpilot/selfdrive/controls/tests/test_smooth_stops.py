import math

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_CTRL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
from openpilot.selfdrive.controls.lib.smooth_stops import (HOLD_RELEASE_FRAMES, SETTLE_JERK, STANDSTILL_SPEED, STOP_KISS_DECEL,
                                                           SmoothStopController)


class TestSmoothStopController(OpenpilotTestCase):

  def test_hold_arms_when_stopped(self):
    controller = SmoothStopController()
    assert not controller.want_hold(True, 0.2)                     # still rolling: the kiss keeps the stop, not the car
    assert controller.want_hold(True, STANDSTILL_SPEED)
    assert not controller.want_hold(True, STANDSTILL_SPEED + 0.05)
    assert not controller.want_hold(False, 0.0)
    assert should_stop(STANDSTILL_SPEED, 0.0)                      # the gates nest: the hold speed sits inside the stop window

  def test_kiss_floor(self):
    controller = SmoothStopController()
    eased = controller.settle(-0.02, -STOP_KISS_DECEL)             # the plan has faded to nothing: the kiss stays
    assert math.isclose(eased, -STOP_KISS_DECEL, abs_tol=1e-9)

  def test_hard_braking_passes_through(self):
    controller = SmoothStopController()
    hard = controller.settle(-1.5, -1.5)
    assert math.isclose(hard, -1.5, abs_tol=1e-9)

  def test_jerk_limit(self):
    controller = SmoothStopController()
    step = SETTLE_JERK * DT_CTRL
    assert math.isclose(controller.settle(-2.0, -0.2), -0.2 - step, abs_tol=1e-9)
    assert math.isclose(controller.settle(-0.02, -1.0), -1.0 + step, abs_tol=1e-9)
    assert controller.settle(-9.0, ACCEL_MIN) >= ACCEL_MIN

  def test_launch_release(self):
    controller = SmoothStopController()
    controller.arm_hold()
    assert controller.hold_release(False, 0.1)                     # the stop bit drops with the plan asking to move

  def test_flicker_keeps_hold(self):
    controller = SmoothStopController()
    controller.arm_hold()
    for _ in range(5):                                             # a few frames of dropped bit, plan at the kiss
      assert not controller.hold_release(False, -0.15)
    assert not controller.hold_release(True, -0.15)

  def test_backstop_release(self):
    controller = SmoothStopController()
    controller.arm_hold()
    for _ in range(HOLD_RELEASE_FRAMES - 1):
      assert not controller.hold_release(False, 0.0)
    assert controller.hold_release(False, 0.0)
    assert not controller.hold_release(True, 0.0)                  # the stop bit back on zeroes the count

  def test_arm_resets_counter(self):
    controller = SmoothStopController()
    controller.arm_hold()
    for _ in range(HOLD_RELEASE_FRAMES):
      controller.hold_release(False, 0.0)
    controller.arm_hold()
    assert not controller.hold_release(False, 0.0)
