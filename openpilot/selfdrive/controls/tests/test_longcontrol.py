from types import SimpleNamespace

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.blotv2 import BLOTV2_ACCEL_MAX
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState, long_control_state_trans
from openpilot.selfdrive.controls.lib.longitudinal_lead import LeadObservation
from openpilot.selfdrive.controls.lib.smooth_stops import HOLD_RELEASE_FRAMES


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


def long_control():
  tuning = SimpleNamespace(
    kpBP=[0.0],
    kpV=[1.0],
    kiBP=[0.0],
    kiV=[0.0],
  )
  return LongControl(SimpleNamespace(longitudinalTuning=tuning, stopAccel=-2.0))


def car_state(v_ego, standstill=False):
  return SimpleNamespace(
    vEgo=v_ego,
    aEgo=0.0,
    brakePressed=False,
    standstill=standstill,
    cruiseState=SimpleNamespace(standstill=standstill),
  )


class TestSmoothStopLongControlIntegration(OpenpilotTestCase):
  def test_final_command_honors_blotv2_platform_ceiling(self):
    control = long_control()
    CS = car_state(5.0)
    CS.aEgo = -2.0
    output = control.update(
      True,
      CS,
      BLOTV2_ACCEL_MAX,
      False,
      (-3.5, 10.0),
      LeadObservation(),
    )
    assert control.pid.pos_limit == BLOTV2_ACCEL_MAX
    assert output == BLOTV2_ACCEL_MAX

  def test_engaged_rolling_stop_stays_in_pid_settle(self):
    control = long_control()
    output = control.update(
      True,
      car_state(0.6),
      -0.4,
      True,
      (-3.5, 2.0),
      LeadObservation(),
    )
    assert control.long_control_state == LongCtrlState.pid
    assert output <= -0.4

  def test_true_standstill_hands_off_to_stock_hold(self):
    control = long_control()
    control.long_control_state = LongCtrlState.pid
    control.update(
      True,
      car_state(0.04),
      -0.1,
      True,
      (-3.5, 2.0),
      LeadObservation(),
    )
    assert control.long_control_state == LongCtrlState.stopping

  def test_hold_release_is_debounced_before_pid(self):
    control = long_control()
    control.long_control_state = LongCtrlState.stopping
    control.smooth_stop.arm_hold()
    for _ in range(HOLD_RELEASE_FRAMES - 1):
      control.update(
        True,
        car_state(0.0),
        0.2,
        False,
        (-3.5, 2.0),
        LeadObservation(),
      )
      assert control.long_control_state == LongCtrlState.stopping

    control.update(
      True,
      car_state(0.0),
      0.2,
      False,
      (-3.5, 2.0),
      LeadObservation(),
    )
    assert control.long_control_state == LongCtrlState.pid
