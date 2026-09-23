from types import SimpleNamespace

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
