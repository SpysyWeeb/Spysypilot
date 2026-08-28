from opendbc.car.structs import car
from openpilot.selfdrive.controls.controlsd import get_steer_limited_by_safety
from openpilot.spysypilot.aol.aol import AolDriver
from openpilot.spysypilot.aol.state import State


class FakeSubMaster:
  def __init__(self) -> None:
    self.valid = {'controlsState': False, 'pandaStates': True}
    self.alive = {'pandaStates': True}

  def __getitem__(self, service: str):
    assert service == 'pandaStates'
    return []


class FakeSelfdrived:
  def __init__(self, *, enabled: bool) -> None:
    self.CP = car.CarParams.new_message()
    self.enabled = enabled
    self.sm = FakeSubMaster()


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

  def test_invalid_controls_state_disables_only_aol_session(self):
    CS = car.CarState.new_message()
    for selfdrive_enabled in (False, True):
      sd = FakeSelfdrived(enabled=selfdrive_enabled)
      aol = AolDriver(sd)
      aol.state_machine.state = State.enabled
      aol.enabled = True
      aol.active = True

      aol.update_events(CS)
      aol.update(CS)

      if selfdrive_enabled:
        assert aol.enabled and aol.active
        assert aol.create_aol_alerts() == []
      else:
        assert not aol.enabled and not aol.active
        assert [alert.alert_type for alert in aol.create_aol_alerts()] == [
          "aolDisengaged/warning",
        ]
