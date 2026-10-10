from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from opendbc.car.structs import car
from openpilot.selfdrive.monitoring.policy import DriverMonitoring


@pytest.mark.parametrize('stock_enabled,lat_active,expected', [(False, True, True), (False, False, False), (True, False, True)])
def test_dm_treats_sol_steering_as_engaged(stock_enabled, lat_active, expected):
  dm = DriverMonitoring()
  # Observe the real run_step dispatch into the existing DM policy, not a duplicate rule.
  dm._set_pose_strictness = Mock()
  dm._update_states = Mock()
  dm._update_events = Mock()
  sm = {
    'carState': car.CarState.new_message(vEgo=20.0, gearShifter='drive'),
    'selfdriveState': SimpleNamespace(enabled=stock_enabled),
    'carControl': car.CarControl.new_message(latActive=lat_active),
    'modelV2': SimpleNamespace(meta=SimpleNamespace(disengagePredictions=SimpleNamespace(brakeDisengageProbs=[0.0]))),
    'extrinsicsCalibration': SimpleNamespace(rpyCalib=[0.0, 0.0, 0.0]),
    'driverStateV2': SimpleNamespace(),
  }
  dm.run_step(sm)
  assert dm._update_states.call_args.kwargs['op_engaged'] is expected
  assert dm._update_events.call_args.kwargs['op_engaged'] is expected


def test_stale_carcontrol_invalidates_monitoring_without_using_old_enable(monkeypatch):
  from openpilot.selfdrive.monitoring import dmonitoringd

  dm = DriverMonitoring()
  dm.run_step = Mock(wraps=dm.run_step)
  published = []

  class StopDaemon(Exception):
    pass

  class Publisher:
    @staticmethod
    def send(topic, packet):
      assert topic == 'driverMonitoringState'
      published.append(packet.valid)
      raise StopDaemon

  def subscriber(services, poll):
    assert 'carControl' in services and poll == 'driverStateV2'
    # SubMaster's aggregate health is false when carControl is stale, even if
    # its cached latActive field still says true. Exercise the real daemon gate.
    return SimpleNamespace(update=lambda: None, updated={'driverStateV2': True}, all_checks=lambda: False)

  monkeypatch.setattr(dmonitoringd, 'config_realtime_process', lambda *args: None)
  monkeypatch.setattr(dmonitoringd, 'Params', lambda: SimpleNamespace(get_bool=lambda _: False))
  monkeypatch.setattr(dmonitoringd, 'DriverMonitoring', lambda **kwargs: dm)
  monkeypatch.setattr(dmonitoringd.messaging, 'PubMaster', lambda _: Publisher())
  monkeypatch.setattr(dmonitoringd.messaging, 'SubMaster', subscriber)
  with pytest.raises(StopDaemon):
    dmonitoringd.dmonitoringd_thread()
  assert published == [False]
  dm.run_step.assert_not_called()
