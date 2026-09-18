from opendbc.car.structs import car

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState, long_control_state_trans
from openpilot.selfdrive.controls.lib.smooth_stops import HOLD_RELEASE_FRAMES, STOP_KISS_DECEL


class TestLongControlStateTransition(OpenpilotTestCase):

  def test_stay_stopped(self):
    active = True
    current_state = LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
    active = False
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.off

  def test_engage(self):
    active = True
    current_state = LongCtrlState.off
    next_state = long_control_state_trans(active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid


def long_control(ki=0.0):
  CP = car.CarParams.new_message()
  CP.longitudinalTuning.kiBP = [0.0]
  CP.longitudinalTuning.kiV = [ki]
  CP.stopAccel = -2.0
  return LongControl(CP)


def car_state(v_ego=0.0, a_ego=0.0):
  CS = car.CarState.new_message()
  CS.vEgo = v_ego
  CS.aEgo = a_ego
  return CS


class TestSmoothStopHandoff(OpenpilotTestCase):

  def test_clamp_waits_for_stop(self):
    control = long_control()
    control.long_control_state = LongCtrlState.pid
    control.last_output_accel = -0.4
    output = control.update(True, car_state(v_ego=0.25), -0.05, True, (-3.5, 2.0))
    assert control.long_control_state == LongCtrlState.pid          # still rolling: no clamp
    assert output <= -STOP_KISS_DECEL                               # the landing keeps the kiss on
    control.update(True, car_state(), -0.05, True, (-3.5, 2.0))
    assert control.long_control_state == LongCtrlState.stopping     # stopped: the clamp arms

  def test_chatter_keeps_hold(self):
    control = long_control()
    control.long_control_state = LongCtrlState.stopping
    control.last_output_accel = -0.75
    control.smooth_stop.arm_hold()
    states = []
    for should_stop in [True] * 20 + [False] * (HOLD_RELEASE_FRAMES - 1) + [True] * 20:
      control.update(True, car_state(), -0.24, should_stop, (-3.5, 2.0))
      states.append(control.long_control_state)
    assert set(states) == {LongCtrlState.stopping}

  def test_launch_release(self):
    control = long_control()
    control.long_control_state = LongCtrlState.stopping
    control.smooth_stop.arm_hold()
    control.update(True, car_state(), 0.2, True, (-3.5, 2.0))
    control.update(True, car_state(), 0.2, False, (-3.5, 2.0))      # the stop bit drops with the plan asking to move
    assert control.long_control_state == LongCtrlState.pid

  def test_flicker_keeps_hold(self):
    control = long_control()
    control.long_control_state = LongCtrlState.stopping
    control.smooth_stop.arm_hold()
    for _ in range(5):                                              # a few frames of dropped bit, plan at the kiss
      control.update(True, car_state(), -0.15, False, (-3.5, 2.0))
      assert control.long_control_state == LongCtrlState.stopping
    control.update(True, car_state(), -0.15, True, (-3.5, 2.0))
    assert control.long_control_state == LongCtrlState.stopping

  def test_backstop_release(self):
    control = long_control()
    control.long_control_state = LongCtrlState.stopping
    control.smooth_stop.arm_hold()
    for _ in range(HOLD_RELEASE_FRAMES - 1):
      control.update(True, car_state(), 0.0, False, (-3.5, 2.0))
      assert control.long_control_state == LongCtrlState.stopping
    control.update(True, car_state(), 0.0, False, (-3.5, 2.0))
    assert control.long_control_state == LongCtrlState.pid

  def test_arm_resets_backstop(self):
    control = long_control()
    control.long_control_state = LongCtrlState.stopping
    control.smooth_stop.arm_hold()
    for _ in range(HOLD_RELEASE_FRAMES):
      control.update(True, car_state(), 0.0, False, (-3.5, 2.0))
    assert control.long_control_state == LongCtrlState.pid
    control.update(True, car_state(), -0.05, True, (-3.5, 2.0))     # back into the hold
    assert control.long_control_state == LongCtrlState.stopping
    control.update(True, car_state(), 0.0, False, (-3.5, 2.0))      # the debounce starts over
    assert control.long_control_state == LongCtrlState.stopping

  def test_landing_resets_pid(self):
    control = long_control(ki=0.5)
    control.long_control_state = LongCtrlState.pid
    for _ in range(10):
      control.update(True, car_state(v_ego=1.0), 1.0, False, (-3.5, 2.0))
    assert control.pid.i != 0.0
    control.update(True, car_state(v_ego=0.25), -0.05, True, (-3.5, 2.0))
    assert control.pid.i == 0.0
    assert control.long_control_state == LongCtrlState.pid
