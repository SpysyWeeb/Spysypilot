from types import SimpleNamespace
from typing import Any

import pytest

from opendbc.car.structs import car
from openpilot.selfdrive.controls.controlsd import Controls
from openpilot.selfdrive.controls.controlsd_ext import ControlsExt
from openpilot.selfdrive.controls.lib.lane_centering import LaneCentering
from openpilot.selfdrive.controls.tests.test_lane_centering import get_model


def make_controls(torque_request):
  controls: Any = Controls.__new__(Controls)
  controls.CP = SimpleNamespace(minSteerSpeed=0.0, steerAtStandstill=False, openpilotLongitudinalControl=True,
                               steerControlType=car.CarParams.SteerControlType.torque,
                               lateralTuning=SimpleNamespace(which=lambda: 'pid'))
  CS = car.CarState.new_message(vEgo=10.0, canValid=True)
  data = {
    'carState': CS,
    'vehicleParameters': SimpleNamespace(stiffnessFactor=1.0, steerRatio=15.0, angleOffsetDeg=0.0, roll=0.0),
    'longitudinalPlan': SimpleNamespace(aTarget=0.0, shouldStop=False),
    'modelV2': get_model(lane_center=0.3, v_ego=CS.vEgo),
    'selfdriveState': SimpleNamespace(enabled=False, active=False),
    'spysydriveStateSP': SimpleNamespace(aol=SimpleNamespace(available=True, active=True)),
    'onroadEvents': [],
    'lateralDelay': SimpleNamespace(lateralDelay=0.2),
  }

  class Sm(dict):
    valid = {'lateralManeuverPlan': False, 'spysydriveStateSP': True}
    alive = {'spysydriveStateSP': True}

  controls.sm = Sm(data)
  controls.VM = SimpleNamespace(update_params=lambda *a: None, calc_curvature=lambda *a: 0.0)
  controls.LoC = SimpleNamespace(long_control_state=car.CarControl.Actuators.LongControlState.off,
                                reset=lambda: None, update=lambda *a: 0.0)
  controls.CI = SimpleNamespace(get_pid_accel_limits=lambda *a: (-3.5, 2.0))
  controls.LaC = SimpleNamespace(reset=lambda: None, update=lambda *a: (torque_request, 0.0, None))
  controls.controls_ext = ControlsExt()
  controls.lane_centering = LaneCentering()
  controls.lane_centering_enabled = True
  controls.desired_curvature = 0.0
  controls.steer_limited_by_safety = False
  return controls, CS


@pytest.mark.parametrize('torque_request,expected', [(0.8, 0.5), (-0.8, -0.5), (0.3, 0.3), (-0.3, -0.3), (0.0, 0.0)])
@pytest.mark.parametrize('driver_torque', [-250.0, 250.0])
def test_driver_touch_caps_not_scales_and_has_no_release_latch(torque_request, expected, driver_torque):
  controls, CS = make_controls(torque_request)
  CS.steeringPressed = True
  CS.steeringTorque = driver_torque
  cc, _ = controls.state_control()
  assert cc.latActive and not cc.enabled and not cc.longActive
  assert cc.actuators.torque == pytest.approx(expected)
  CS.steeringPressed = False
  cc, _ = controls.state_control()
  assert cc.actuators.torque == pytest.approx(torque_request)


def test_stock_fallback_does_not_gain_sol_driver_cap():
  controls, CS = make_controls(0.8)
  controls.sm['spysydriveStateSP'].aol.available = False
  controls.sm['selfdriveState'].active = True
  CS.steeringPressed = True
  cc, _ = controls.state_control()
  assert cc.actuators.torque == pytest.approx(0.8)


@pytest.mark.parametrize('torque_request', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_command_fails_closed_before_driver_cap(torque_request):
  controls, CS = make_controls(torque_request)
  CS.steeringPressed = True
  cc, _ = controls.state_control()
  assert cc.actuators.torque == 0.0
