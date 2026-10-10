from opendbc.car.structs import car
from openpilot.selfdrive.controls.controlsd import get_steer_limited_by_safety


class TestAolSteeringLimitFeedback:
  def test_torque_limit_is_updated_from_lat_active(self):
    CP = car.CarParams.new_message()
    CP.steerControlType = car.CarParams.SteerControlType.torque
    CC = car.CarControl.new_message()
    CO = car.CarOutput.new_message()
    CC.enabled = False
    CC.latActive = True
    CC.actuators.torque = 0.8
    CO.actuatorsOutput.torque = 0.5
    assert get_steer_limited_by_safety(CP, CC, CO)

    CC.latActive = False
    assert not get_steer_limited_by_safety(CP, CC, CO)
