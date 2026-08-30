import math

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
from openpilot.selfdrive.controls.lib.smooth_stops import (HOLD_RELEASE_FRAMES, SETTLE_JERK, STALL_RATE, STALL_S, STANDSTILL_HOLD_SPEED,
                                                            STANDSTILL_SPEED, STOP_KISS_DECEL, SmoothStopController)


def test_the_hold_only_arms_on_a_stopped_car_and_inside_the_plans_stop_window():
  controller = SmoothStopController()
  assert not controller.want_hold(True, 0.2, standstill=False)
  assert controller.want_hold(True, STANDSTILL_SPEED, standstill=False)
  assert controller.want_hold(True, 0.1, standstill=True)
  assert not controller.want_hold(True, 0.2, standstill=True)             # the car's flag is not believed while rolling
  assert not controller.want_hold(False, 0.0, standstill=True)
  assert should_stop(STANDSTILL_HOLD_SPEED, 0.0)                          # the two gates nest: hold speeds sit inside the stop window


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


def test_a_stall_ratchets_the_pressure_up_until_the_car_moves_again():
  controller = SmoothStopController()
  output = -STOP_KISS_DECEL
  for _ in range(round(STALL_S / DT_CTRL) - 1):
    output = controller.settle(-0.02, 0.2, output)
  assert math.isclose(output, -STOP_KISS_DECEL, abs_tol=1e-9)             # a grace period first
  stalled = 100
  for _ in range(stalled):
    output = controller.settle(-0.02, 0.2, output)                        # a further second stalled
  stall_time = (round(STALL_S / DT_CTRL) - 2 + stalled) * DT_CTRL          # the first call is progress from rest
  expected = STOP_KISS_DECEL + STALL_RATE * (stall_time - STALL_S)
  assert math.isclose(output, -expected, abs_tol=1e-6)
  for _ in range(20):
    output = controller.settle(-0.02, 0.1, output)                        # progress again: the ratchet releases
  assert math.isclose(output, -STOP_KISS_DECEL, abs_tol=1e-6)


def test_the_hold_releases_after_the_debounce_and_arms_fresh():
  controller = SmoothStopController()
  controller.arm_hold()
  for _ in range(HOLD_RELEASE_FRAMES - 1):
    assert not controller.hold_release(False)
  assert controller.hold_release(False)
  assert not controller.hold_release(True)
  controller.arm_hold()
  assert not controller.hold_release(False)
