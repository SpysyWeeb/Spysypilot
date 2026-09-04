import math

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
from openpilot.selfdrive.controls.lib.smooth_stops import (HOLD_RELEASE_FRAMES, SETTLE_JERK, STANDSTILL_SPEED, STOP_KISS_DECEL,
                                                            SmoothStopController)


def test_the_hold_only_arms_on_a_stopped_car_and_inside_the_plans_stop_window():
  controller = SmoothStopController()
  assert not controller.want_hold(True, 0.2)                              # still rolling: the kiss keeps the stop, not the car
  assert controller.want_hold(True, STANDSTILL_SPEED)
  assert not controller.want_hold(True, STANDSTILL_SPEED + 0.05)
  assert not controller.want_hold(False, 0.0)
  assert should_stop(STANDSTILL_SPEED, 0.0)                               # the two gates nest: the hold speed sits inside the stop window


def test_the_landing_keeps_the_kiss_on_and_passes_harder_plan_braking_through():
  controller = SmoothStopController()
  eased = controller.settle(-0.02, 0.25, -STOP_KISS_DECEL)                # the plan has faded to nothing: the kiss stays
  assert math.isclose(eased, -STOP_KISS_DECEL, abs_tol=1e-9)
  hard = controller.settle(-1.5, 0.25, -1.5)                              # the plan still braking: passed through
  assert math.isclose(hard, -1.5, abs_tol=1e-9)


def test_the_landing_never_steps():
  controller = SmoothStopController()
  step = SETTLE_JERK * DT_CTRL
  assert math.isclose(controller.settle(-2.0, 0.25, -0.2), -0.2 - step, abs_tol=1e-9)
  assert math.isclose(controller.settle(-0.02, 0.25, -1.0), -1.0 + step, abs_tol=1e-9)
  assert controller.settle(-9.0, 0.25, ACCEL_MIN) >= ACCEL_MIN



def test_a_launch_releases_the_hold_on_its_first_frame():
  controller = SmoothStopController()
  controller.arm_hold()
  assert controller.hold_release(False, 0.1)                                # the stop bit drops with the plan asking to move


def test_a_stop_bit_flicker_with_the_plan_still_braking_does_not_release():
  controller = SmoothStopController()
  controller.arm_hold()
  for _ in range(5):                                                       # a planner frame of dropped bit, plan at the kiss
    assert not controller.hold_release(False, -0.15)
  assert not controller.hold_release(True, -0.15)


def test_a_plan_sitting_at_zero_releases_after_the_backstop_and_arms_fresh():
  controller = SmoothStopController()
  controller.arm_hold()
  for _ in range(HOLD_RELEASE_FRAMES - 1):
    assert not controller.hold_release(False, 0.0)
  assert controller.hold_release(False, 0.0)
  assert not controller.hold_release(True, 0.0)
  controller.arm_hold()
  assert not controller.hold_release(False, 0.0)
