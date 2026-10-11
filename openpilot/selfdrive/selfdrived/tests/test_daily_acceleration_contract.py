"""Distinguish the shared command ceiling from the existing 2x measured-acceleration monitor."""
import pytest

import openpilot.cereal.messaging as messaging
from opendbc.car.hyundai.values import CarControllerParams
from opendbc.car.interfaces import ACCEL_MAX, ACCEL_MIN
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel
from openpilot.selfdrive.locationd.helpers import Pose
from openpilot.selfdrive.selfdrived.helpers import ExcessiveActuationCheck, ExcessiveActuationType, MIN_EXCESSIVE_ACTUATION_COUNT


def test_daily_host_hyundai_and_launch_envelope_match_the_selected_ceiling():
  assert ACCEL_MAX == CarControllerParams.ACCEL_MAX == get_max_accel(0.0) == 4.0
  assert ACCEL_MIN == CarControllerParams.ACCEL_MIN == -3.5
  assert get_max_accel(40.0) == 0.6


@pytest.mark.parametrize('acceleration, expected', [
  (4.0, None), (8.0, None), (8.01, ExcessiveActuationType.LONGITUDINAL),
  (-7.0, None), (-7.01, ExcessiveActuationType.LONGITUDINAL),
])
def test_measured_acceleration_monitor_keeps_combo_ratio_and_existing_debounce(acceleration, expected):
  # Use the real message accessor and Pose representation without opening live sockets.
  sm = messaging.SubMaster.__new__(messaging.SubMaster)
  sm.data = {s: getattr(messaging.new_message(s), s) for s in ('carControl', 'vehicleParameters')}
  cs = messaging.new_message('carState').carState
  cs.aEgo = acceleration
  sm['carControl'].longActive = True
  pose = Pose.from_device_motion(messaging.new_message('deviceMotion').deviceMotion)
  pose.acceleration.xyz[0] = acceleration
  monitor = ExcessiveActuationCheck()
  for _ in range(MIN_EXCESSIVE_ACTUATION_COUNT):
    assert monitor.update(sm, cs, pose) is None
  assert monitor.update(sm, cs, pose) == expected
  # An inactive longitudinal channel or inconsistent measurement cannot retain the counter.
  sm['carControl'].longActive = False
  assert monitor.update(sm, cs, pose) is None
  assert monitor._excessive_counter == 0
  sm['carControl'].longActive = True
  cs.aEgo = 0.0
  if abs(acceleration) >= 2.0:
    for _ in range(MIN_EXCESSIVE_ACTUATION_COUNT + 1):
      assert monitor.update(sm, cs, pose) is None
    assert monitor._excessive_counter == 0
