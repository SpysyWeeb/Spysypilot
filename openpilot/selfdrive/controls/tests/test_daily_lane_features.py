import threading
import time
from unittest.mock import patch

from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR
from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.common.parameterized import parameterized
from openpilot.common.params import Params, ParamKeyType
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.controlsd import Controls
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper, LaneChangeState
from openpilot.selfdrive.controls.tests.test_lane_centering import get_car_state, get_model, V_EGO


class TestDailyLaneFeatures(OpenpilotTestCase):
  def test_centering_default_is_registered(self):
    params = Params()
    assert params.get_type("LaneCenteringEnabled") == ParamKeyType.BOOL
    assert params.get_default_value("LaneCenteringEnabled") is True

  def make_controls(self):
    CP = interfaces[CAR.HYUNDAI_PALISADE].get_non_essential_params(CAR.HYUNDAI_PALISADE)
    Params().put("CarParams", CP.to_bytes(), block=True)
    controls = Controls()
    state = messaging.new_message('spysydriveStateSP', valid=True)
    state.spysydriveStateSP.aol.available = True
    state.spysydriveStateSP.aol.active = True
    controls.sm.update_msgs(time.monotonic(), [state.as_reader()])
    controls.sm.data['carState'] = get_car_state(canValid=True)
    controls.sm.data['selfdriveState'] = log.SelfdriveState.new_message(enabled=False, active=False)
    controls.sm.data['vehicleParameters'] = log.VehicleParameters.new_message(stiffnessFactor=1.0, steerRatio=CP.steerRatio)
    controls.sm.data['modelV2'] = get_model(lane_center=0.3)
    controls.sm['modelV2'].action.desiredCurvature = 0.001
    return controls

  def test_sol_only_centering_and_toggle(self):
    controls = self.make_controls()
    model_curvature = controls.sm['modelV2'].action.desiredCurvature
    assert controls.lane_centering_enabled
    for _ in range(300):
      CC, _ = controls.state_control()
    assert CC.latActive and not CC.enabled and not CC.longActive
    assert controls.desired_curvature > model_curvature

    for enabled in (False, True):
      Params().put_bool("LaneCenteringEnabled", enabled, block=True)
      stop = threading.Event()
      with patch.object(stop, 'wait', side_effect=lambda _, event=stop: event.set()):
        controls.params_thread(stop)
      assert controls.lane_centering_enabled == enabled
      for _ in range(300):
        CC, _ = controls.state_control()
      assert CC.latActive and not CC.enabled and not CC.longActive
      if enabled:
        assert controls.desired_curvature > model_curvature
      else:
        assert controls.lane_centering.curvature.x == 0.0
        assert controls.desired_curvature == model_curvature
    assert controls.sm['modelV2'].action.desiredCurvature == model_curvature

  @parameterized.expand([('alive',), ('valid',)])
  def test_stale_sol_state_removes_centering(self, field):
    controls = self.make_controls()
    for _ in range(300):
      controls.state_control()
    assert controls.lane_centering.curvature.x > 0.0
    # Ordinary engagement must not bypass the independent authority's freshness gate.
    controls.sm['selfdriveState'].active = True
    getattr(controls.sm, field)['spysydriveStateSP'] = False
    CC, _ = controls.state_control()
    assert not CC.latActive
    assert controls.lane_centering.curvature.x == 0.0
    assert CC.actuators.torque == 0.0

  @parameterized.expand([('leftBlinker',), ('rightBlinker',)])
  def test_nudgeless_handoff_once_and_recenter(self, blinker):
    controls = self.make_controls()
    helper = DesireHelper()
    CS, model = controls.sm['carState'], controls.sm['modelV2']

    def step(probability=1.0):
      CC, _ = controls.state_control()
      helper.update(CS, CC.latActive, probability)
      model.frameId += 1
      model.meta.laneChangeState = helper.lane_change_state
      model.meta.laneChangeDirection = helper.lane_change_direction
      for _ in range(round(DT_MDL / DT_CTRL)):
        CC, _ = controls.state_control()
        assert CC.latActive and not CC.enabled and not CC.longActive
      return helper.lane_change_state

    for _ in range(60):
      step()
    centered = controls.lane_centering.curvature.x
    assert centered > 0.0 and CS.vEgo == V_EGO

    setattr(CS, blinker, True)
    for _ in range(3):
      state = step()
    assert state == LaneChangeState.laneChangeStarting
    assert not CS.steeringPressed and helper.nudgeless_used
    assert 0.0 < controls.lane_centering.curvature.x < centered

    for _ in range(20):
      if step(0.0) == LaneChangeState.preLaneChange:
        break
    assert helper.lane_change_state == LaneChangeState.preLaneChange
    for _ in range(40):
      assert step(0.0) == LaneChangeState.preLaneChange
    assert controls.lane_centering.curvature.x < centered * 0.001

    setattr(CS, blinker, False)
    assert step(0.0) == LaneChangeState.off
    for _ in range(60):
      step()
    assert controls.lane_centering.curvature.x > centered * 0.99

    # Only a new signal event restores the automatic allowance.
    setattr(CS, blinker, True)
    for _ in range(3):
      state = step()
    assert state == LaneChangeState.laneChangeStarting
