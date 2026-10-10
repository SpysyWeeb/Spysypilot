import threading
import time
from unittest.mock import patch

from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR
from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.common.parameterized import parameterized
from openpilot.common.params import Params, ParamKeyFlag, ParamKeyType
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.controlsd import Controls
from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature
from openpilot.selfdrive.controls.tests.test_lane_centering import get_model, get_car_state, V_EGO


class TestLaneCenteringToggle(OpenpilotTestCase):
  def test_param_defaults_on_and_persists(self):
    params = Params()
    assert params.get_type("LaneCenteringEnabled") == ParamKeyType.BOOL
    assert params.get_default_value("LaneCenteringEnabled") is True
    for enabled in (False, True):
      params.put_bool("LaneCenteringEnabled", enabled, block=True)
      for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
                   ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION):
        params.clear_all(flag)
        assert Params().get_bool("LaneCenteringEnabled") == enabled

  def make_controls(self, enabled: bool | None = True):
    params = Params()
    if enabled is not None:
      params.put_bool("LaneCenteringEnabled", enabled, block=True)
    CP = interfaces[CAR.HYUNDAI_PALISADE].get_non_essential_params(CAR.HYUNDAI_PALISADE)
    params.put("CarParams", CP.to_bytes(), block=True)
    controls = Controls()
    controls.sm.data['carState'] = get_car_state()
    controls.sm.data['selfdriveState'] = log.SelfdriveState.new_message(enabled=True, active=True)
    controls.sm.data['vehicleParameters'] = log.VehicleParameters.new_message(stiffnessFactor=1.0, steerRatio=CP.steerRatio)
    controls.sm.data['modelV2'] = get_model(lane_center=0.3)
    controls.sm['modelV2'].action.desiredCurvature = 0.001
    self.update_controls(controls)
    return controls

  def update_controls(self, controls):
    # Daily requires a fresh SOL status even for ordinary cruise engagement.
    state = messaging.new_message('spysydriveStateSP', valid=True)
    controls.sm.update_msgs(time.monotonic(), [state.as_reader()])
    controls.update()

  def test_control_loop_does_not_read_params(self):
    controls = self.make_controls()
    with patch.object(controls.params, 'get_bool', side_effect=AssertionError('Param I/O in control loop')), \
         patch.object(controls.params, 'get', side_effect=AssertionError('Param I/O in control loop')):
      for _ in range(20):
        self.update_controls(controls)
        controls.state_control()

  def test_live_toggle_uses_model_and_keeps_curvature_limits(self):
    controls = self.make_controls()
    stop = threading.Event()
    worker = threading.Thread(target=lambda: controls.params_thread(stop))
    worker.start()
    def cleanup():
      stop.set()
      worker.join(timeout=2)
      assert not worker.is_alive()
    self.addCleanup(cleanup)

    def set_enabled(enabled):
      Params().put_bool("LaneCenteringEnabled", enabled, block=True)
      deadline = time.monotonic() + 2
      while controls.lane_centering_enabled != enabled and time.monotonic() < deadline:
        self.update_controls(controls)
      assert controls.lane_centering_enabled == enabled

    model_curvature = controls.sm['modelV2'].action.desiredCurvature
    for _ in range(300):
      controls.state_control()
    assert controls.desired_curvature > model_curvature

    # A separate Params instance stands in for the UI process; no Controls restart.
    set_enabled(False)
    previous = controls.desired_curvature
    CC, _ = controls.state_control()
    assert CC.latActive
    assert controls.lane_centering.curvature.x == 0.0
    assert controls.desired_curvature == clip_curvature(V_EGO, previous, model_curvature, 0.0)[0]
    for _ in range(100):
      controls.state_control()
    assert controls.desired_curvature == model_curvature

    set_enabled(True)
    controls.state_control()
    assert 0.0 < controls.lane_centering.curvature.x < 0.05 * controls.lane_centering.target
    for _ in range(300):
      controls.state_control()
    assert controls.desired_curvature > model_curvature
    assert controls.sm['modelV2'].action.desiredCurvature == model_curvature

  def test_starts_disabled(self):
    controls = self.make_controls(enabled=False)
    for _ in range(300):
      controls.state_control()
    assert controls.lane_centering.curvature.x == 0.0
    assert controls.desired_curvature == controls.sm['modelV2'].action.desiredCurvature

  def test_missing_param_uses_default_in_startup_and_worker(self):
    controls = self.make_controls(enabled=None)
    assert controls.lane_centering_enabled is True
    stop = threading.Event()
    with patch.object(stop, 'wait', side_effect=lambda _: stop.set()):
      controls.params_thread(stop)
    assert controls.lane_centering_enabled is True
    assert Params().get("LaneCenteringEnabled") is None

  def test_run_stops_param_worker_on_exit(self):
    controls = self.make_controls()
    threads = []
    thread_class = threading.Thread

    def make_thread(*args, **kwargs):
      thread = thread_class(*args, **kwargs)
      threads.append(thread)
      return thread

    with patch('openpilot.selfdrive.controls.controlsd.threading.Thread', side_effect=make_thread), \
         patch.object(controls, 'update', side_effect=RuntimeError('test exit')):
      with self.assertRaisesRegex(RuntimeError, 'test exit'):
        controls.run()
    assert len(threads) == 1
    assert not threads[0].is_alive()

  @parameterized.expand([(False,), (True,)])
  def test_inactive_and_maneuver_precedence(self, enabled):
    controls = self.make_controls(enabled=enabled)
    for _ in range(300):
      controls.state_control()
    controls.sm.data['lateralManeuverPlan'] = log.LateralManeuverPlan.new_message(desiredCurvature=-0.001)
    controls.sm.valid['lateralManeuverPlan'] = True
    previous = controls.desired_curvature
    controls.state_control()
    assert controls.desired_curvature == clip_curvature(V_EGO, previous, controls.sm['lateralManeuverPlan'].desiredCurvature, 0.0)[0]
    assert controls.lane_centering.curvature.x == 0.0

    controls.sm.valid['lateralManeuverPlan'] = False
    controls.sm['selfdriveState'].active = False
    CC, _ = controls.state_control()
    assert not CC.latActive
    assert controls.lane_centering.curvature.x == 0.0
