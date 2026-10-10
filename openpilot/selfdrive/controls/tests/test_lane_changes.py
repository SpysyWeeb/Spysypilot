from types import SimpleNamespace

import pytest

from opendbc.car.structs import car
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper, LaneChangeState, LANE_CHANGE_SPEED_MIN


def test_brake_on_blinker_activation_cancels_nudgeless_change():
  car_state = SimpleNamespace(
    vEgo=LANE_CHANGE_SPEED_MIN + 1,
    leftBlinker=False,
    rightBlinker=False,
    brakePressed=False,
    steeringPressed=False,
    steeringTorque=0.0,
    leftBlindspot=False,
    rightBlindspot=False,
  )
  helper = DesireHelper()
  helper.update(car_state, True, 1.0)

  car_state.leftBlinker = True
  car_state.brakePressed = True
  helper.update(car_state, True, 1.0)
  car_state.brakePressed = False
  for _ in range(10):
    helper.update(car_state, True, 1.0)

  assert helper.lane_change_state == LaneChangeState.preLaneChange
  assert helper.brake_cancelled


@pytest.mark.parametrize(('blinker', 'torque'), [('leftBlinker', 1.0), ('rightBlinker', -1.0)])
def test_early_manual_start_does_not_leave_an_automatic_retry(blinker, torque):
  car_state = car.CarState.new_message(vEgo=LANE_CHANGE_SPEED_MIN + 1)
  helper = DesireHelper()
  helper.update(car_state, True, 1.0)

  setattr(car_state, blinker, True)
  car_state.steeringPressed = True
  car_state.steeringTorque = torque
  helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.preLaneChange

  # The nudge starts on the first waiting frame, before the auto delay expires.
  helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.laneChangeStarting
  car_state.steeringPressed = False
  car_state.steeringTorque = 0.0

  for _ in range(20):
    helper.update(car_state, True, 0.0)
    if helper.lane_change_state == LaneChangeState.preLaneChange:
      break
  assert helper.lane_change_state == LaneChangeState.preLaneChange

  # Keeping the same signal on must not initiate another change without input.
  for _ in range(40):
    helper.update(car_state, True, 0.0)
    assert helper.lane_change_state == LaneChangeState.preLaneChange

  # A later manual nudge is still allowed under the same signal.
  car_state.steeringPressed = True
  car_state.steeringTorque = torque
  helper.update(car_state, True, 0.0)
  assert helper.lane_change_state == LaneChangeState.laneChangeStarting


@pytest.mark.parametrize('blinker', ['leftBlinker', 'rightBlinker'])
def test_automatic_start_requires_signal_cycle_to_rearm(blinker):
  car_state = car.CarState.new_message(vEgo=LANE_CHANGE_SPEED_MIN + 1)
  setattr(car_state, blinker, True)
  helper = DesireHelper()
  helper.update(car_state, True, 1.0)
  helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.preLaneChange
  helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.laneChangeStarting

  for _ in range(20):
    helper.update(car_state, True, 0.0)
    if helper.lane_change_state == LaneChangeState.preLaneChange:
      break
  for _ in range(40):
    helper.update(car_state, True, 0.0)
    assert helper.lane_change_state == LaneChangeState.preLaneChange

  setattr(car_state, blinker, False)
  helper.update(car_state, True, 0.0)
  assert helper.lane_change_state == LaneChangeState.off
  setattr(car_state, blinker, True)
  for _ in range(3):
    helper.update(car_state, True, 0.0)
  assert helper.lane_change_state == LaneChangeState.laneChangeStarting


def test_brake_cancellation_preserves_manual_start():
  car_state = car.CarState.new_message(vEgo=LANE_CHANGE_SPEED_MIN + 1, leftBlinker=True, brakePressed=True)
  helper = DesireHelper()
  helper.update(car_state, True, 1.0)
  car_state.brakePressed = False
  for _ in range(10):
    helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.preLaneChange

  car_state.steeringPressed = True
  car_state.steeringTorque = 1.0
  helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.laneChangeStarting


@pytest.mark.parametrize('manual_start', [False, True])
def test_blindspot_blocks_start_without_consuming_allowance(manual_start):
  car_state = car.CarState.new_message(vEgo=LANE_CHANGE_SPEED_MIN + 1, leftBlinker=True, leftBlindspot=True,
                                     steeringPressed=manual_start, steeringTorque=1.0 if manual_start else 0.0)
  helper = DesireHelper()
  for _ in range(10):
    helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.preLaneChange
  assert not helper.nudgeless_used

  car_state.leftBlindspot = False
  helper.update(car_state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.laneChangeStarting
