"""Exercise host activation -> Hyundai command history -> unchanged native Panda checks."""
from collections import defaultdict
from types import SimpleNamespace

import pytest

from opendbc.car import Bus
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.carcontroller import CarController
from opendbc.car.hyundai.values import CAR, DBC
from opendbc.car.structs import car
from opendbc.safety.tests.common import CANPackerSafety
from opendbc.safety.tests.libsafety import libsafety_py
from openpilot.cereal import log, messaging
from openpilot.selfdrive.controls.controlsd_ext import ControlsExt


class EngagementSubMaster(messaging.SubMaster):
  def __init__(self):
    super().__init__(['spysydriveStateSP', 'selfdriveState', 'pandaStates'])
    for service in self.services:
      self.alive[service] = self.valid[service] = self.freq_ok[service] = True
    self.sol(False, 1)
    self.health(False, 2)

  def sol(self, active, timestamp, *, available=True):
    msg = messaging.new_message('spysydriveStateSP', valid=True, logMonoTime=timestamp)
    msg.spysydriveStateSP.aol.available = available
    msg.spysydriveStateSP.aol.enabled = active
    msg.spysydriveStateSP.aol.active = active
    self.data['spysydriveStateSP'] = msg.spysydriveStateSP.as_reader()
    self.logMonoTime['spysydriveStateSP'] = timestamp

  def health(self, allowed, timestamp):
    self.data['pandaStates'] = [log.PandaState.new_message(
      safetyModel=car.CarParams.SafetyModel.hyundai, lateralAllowed=allowed).as_reader()]
    self.logMonoTime['pandaStates'] = timestamp


def make_controller():
  cp = interfaces[CAR.HYUNDAI_PALISADE].get_non_essential_params(CAR.HYUNDAI_PALISADE)
  cp.openpilotLongitudinalControl = True
  controller = CarController(DBC[CAR.HYUNDAI_PALISADE], cp)
  cs = SimpleNamespace(out=car.CarState.new_message(vEgo=10.0, steeringAngleDeg=100.0, steeringTorque=0.0),
                       is_metric=True, lkas11=defaultdict(int))
  return controller, cs


def test_post_request_snapshot_cannot_reuse_permission_before_queued_off_heartbeat():
  native = libsafety_py.libsafety
  native.set_safety_hooks(8, 4)
  native.init_tests()
  native.set_alternative_experience(1024)
  native.set_aol_heartbeat(True)
  packer = CANPackerSafety(DBC[CAR.HYUNDAI_PALISADE][Bus.pt])
  for pressed in (False, True, False):
    assert native.safety_rx_hook(packer.make_can_msg_safety('CLU11', 0, {'CF_Clu_CruiseSwMain': int(pressed)}))
  assert native.get_lateral_allowed()
  sm, ext = EngagementSubMaster(), ControlsExt()
  sm.sol(False, 10)
  assert not ext.get_lat_active(sm)
  # pandad consumed off, but the host re-enables while its health query is in flight.
  sm.sol(True, 12)
  sm.health(native.get_lateral_allowed(), 13)
  native.set_aol_heartbeat(False)  # Sent after publishing the old true permission.
  cc = car.CarControl.new_message()
  cc.latActive = ext.get_lat_active(sm)
  cc.actuators.torque = 0.5
  controller, cs = make_controller()
  _, sends = controller.update(cc.as_reader(), cs, 14)
  addr, data, bus = next(msg for msg in sends if msg[0] == 0x340)
  assert native.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, data))
  assert not cc.latActive
  assert controller.apply_torque_last == 0 and controller.angle_limit_counter == 0
  # A later true permission follows a new native grant, not the queued off.
  native.set_aol_heartbeat(True)
  assert native.safety_rx_hook(packer.make_can_msg_safety('CLU11', 0, {'CF_Clu_CruiseSwMain': 1}))
  assert native.get_lateral_allowed()
  sm.health(native.get_lateral_allowed(), 15)
  assert ext.get_lat_active(sm)


def test_permission_expires_before_generic_one_second_alive_timeout():
  sm, ext = EngagementSubMaster(), ControlsExt()
  sm.sol(True, 10)
  sm.health(True, 11)
  ext.get_lat_active(sm)
  sm.health(True, 12)
  assert ext.get_lat_active(sm)
  sm.sol(True, 250_000_011)
  assert ext.get_lat_active(sm)
  sm.sol(True, 250_000_012)
  assert sm.all_checks(['pandaStates'])  # Broad messaging health still says alive.
  assert not ext.get_lat_active(sm)
  sm.health(True, 250_000_013)
  assert not ext.get_lat_active(sm)
  sm.sol(True, 350_000_012)
  sm.health(True, 350_000_013)
  assert ext.get_lat_active(sm)


def test_reenable_does_not_reuse_a_previous_permission_report():
  sm, ext = EngagementSubMaster(), ControlsExt()
  sm.health(True, 2)
  assert not ext.get_lat_active(sm)
  sm.sol(True, 10)
  assert not ext.get_lat_active(sm)  # Cached permission predates this request.
  sm.health(True, 11)
  for _ in range(10):
    assert not ext.get_lat_active(sm)  # Re-reading one publication is not another cycle.
  sm.health(True, 12)
  assert ext.get_lat_active(sm)
  sm.sol(False, 13)
  assert not ext.get_lat_active(sm)
  sm.sol(True, 14)
  assert not ext.get_lat_active(sm)
  sm.health(True, 15)
  assert not ext.get_lat_active(sm)
  sm.health(True, 16)
  assert ext.get_lat_active(sm)


@pytest.mark.parametrize('heartbeat_phase', range(10))
@pytest.mark.parametrize('angle', [0.0, 100.0, -400.0])
@pytest.mark.parametrize('torque_request', [-0.5, 0.5])
def test_delayed_panda_confirmation_never_rejects_the_host_ramp(heartbeat_phase, angle, torque_request):
  native = libsafety_py.libsafety
  native.set_safety_hooks(8, 4)
  native.init_tests()
  native.set_alternative_experience(1024)
  packer = CANPackerSafety(DBC[CAR.HYUNDAI_PALISADE][Bus.pt])
  sm = EngagementSubMaster()
  ext = ControlsExt()
  controller, cs = make_controller()
  cs.out.steeringAngleDeg = angle
  cc = car.CarControl.new_message()
  cc.actuators.torque = torque_request
  rejected, cuts, nonzero = [], [], []
  for frame in range(260):
    timestamp = (frame + 1) * 10_000_000
    native.set_timer(timestamp // 1000)
    enabled = frame >= 20
    sm.sol(enabled, timestamp)
    assert native.safety_rx_hook(packer.make_can_msg_safety('CLU11', 0, {
      'CF_Clu_CruiseSwMain': int(frame == 20), 'CF_Clu_AliveCnt1': frame % 16}))
    # Model pandad's real order: sample health, then send heartbeat at 10 Hz.
    if frame % 10 == heartbeat_phase:
      sm.health(native.get_lateral_allowed(), timestamp + 1)
      native.set_aol_heartbeat(enabled)
    cc.latActive = ext.get_lat_active(sm)
    _, sends = controller.update(cc.as_reader(), cs, timestamp)
    addr, data, bus = next(msg for msg in sends if msg[0] == 0x340)
    raw = int.from_bytes(data, 'little')
    torque, request = ((raw >> 16) & 0x7ff) - 1024, (raw >> 27) & 1
    if torque:
      nonzero.append(frame)
      if not request:
        cuts.append(frame)
    if not native.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, data)):
      rejected.append((frame, torque, request))
  assert not rejected, f'host advanced before authorization or desynchronized high-angle history: {rejected[:5]}'
  assert nonzero
  assert bool(cuts) == (abs(angle) >= 85)  # Exercise the real high-angle request-bit workaround.


@pytest.mark.parametrize('health_field', ['alive', 'valid', 'freq_ok'])
def test_unhealthy_panda_cannot_authorize_or_keep_sol_active(health_field):
  sm, ext = EngagementSubMaster(), ControlsExt()
  sm.sol(True, 10)
  sm.health(True, 11)
  assert not ext.get_lat_active(sm)
  sm.health(True, 12)
  assert ext.get_lat_active(sm)
  getattr(sm, health_field)['pandaStates'] = False
  assert not ext.get_lat_active(sm)
  getattr(sm, health_field)['pandaStates'] = True
  sm.health(False, 13)
  assert not ext.get_lat_active(sm)
  sm.health(True, 14)
  assert ext.get_lat_active(sm)


@pytest.mark.parametrize('states,expected', [
  ([], False),
  ([('silent', False, False)], False),
  ([('noOutput', False, False)], False),
  ([('hyundai', False, False)], False),
  ([('hyundai', True, False)], True),
  ([('hyundaiLegacy', True, False)], True),
  ([('hyundaiCanfd', True, False)], True),
  ([('hyundai', False, True)], True),
  ([('toyota', True, True)], False),
  ([('hyundai', True, False), ('noOutput', False, False)], True),
  ([('hyundai', True, False), ('hyundai', False, False)], False),
])
def test_all_actuating_pandas_must_authorize(states, expected):
  sm, ext = EngagementSubMaster(), ControlsExt()
  sm.sol(True, 10)
  sm.logMonoTime['pandaStates'] = 11
  sm.data['pandaStates'] = [log.PandaState.new_message(safetyModel=model, lateralAllowed=lateral,
                                                      controlsAllowed=controls).as_reader()
                          for model, lateral, controls in states]
  first = ext.get_lat_active(sm)
  assert first == (expected and all(controls for model, _, controls in states if model not in ('silent', 'noOutput')))
  sm.logMonoTime['pandaStates'] = 12
  assert ext.get_lat_active(sm) == expected


def test_stale_sol_recovery_requires_a_new_permission_report():
  sm, ext = EngagementSubMaster(), ControlsExt()
  sm.sol(True, 10)
  sm.health(True, 11)
  assert not ext.get_lat_active(sm)
  sm.health(True, 12)
  assert ext.get_lat_active(sm)
  sm.alive['spysydriveStateSP'] = False
  assert not ext.get_lat_active(sm)
  sm.alive['spysydriveStateSP'] = True
  sm.sol(True, 13)
  assert not ext.get_lat_active(sm)
  sm.health(True, 14)
  assert not ext.get_lat_active(sm)
  sm.health(True, 15)
  assert ext.get_lat_active(sm)


@pytest.mark.parametrize('release', ['steerFaultTemporary', 'steerFaultPermanent', 'aol_paused', 'panda_revoked'])
def test_real_controlsd_release_resets_lateral_control(release):
  from openpilot.common.prefix import OpenpilotPrefix
  from openpilot.selfdrive.controls.tests.test_daily_lane_features import TestDailyLaneFeatures

  with OpenpilotPrefix():
    controls = TestDailyLaneFeatures().make_controls()
    controls.LaC.pid.i = 0.2
    controls.LaC.sat_time = 0.2
    if release.startswith('steerFault'):
      setattr(controls.sm['carState'], release, True)
    elif release == 'aol_paused':
      state = controls.sm['spysydriveStateSP'].as_builder()
      state.aol.active = False
      controls.sm.data['spysydriveStateSP'] = state.as_reader()
    else:
      panda = log.PandaState.new_message(safetyModel='hyundai', lateralAllowed=False)
      controls.sm.data['pandaStates'] = [panda.as_reader()]
    cc, _ = controls.state_control()
    assert not cc.latActive and not cc.enabled and not cc.longActive
    assert cc.actuators.torque == 0.0
    assert controls.LaC.sat_time == 0.0
    assert controls.LaC.pid.i == 0.2  # Preserve the stock torque PID's inactive freeze semantics.
    controller, cs = make_controller()
    cs.out = controls.sm['carState']
    controller.apply_torque_last, controller.angle_limit_counter = 100, 50
    controller.update(cc.as_reader(), cs, 1)
    assert controller.apply_torque_last == 0 and controller.angle_limit_counter == 0


def test_controlsd_runtime_and_process_replay_receive_permission():
  from openpilot.common.params import Params
  from openpilot.common.prefix import OpenpilotPrefix
  from openpilot.selfdrive.controls.controlsd import Controls
  from openpilot.selfdrive.test.process_replay.process_replay import get_process_config

  cp = interfaces[CAR.HYUNDAI_PALISADE].get_non_essential_params(CAR.HYUNDAI_PALISADE)
  with OpenpilotPrefix():
    Params().put('CarParams', cp.to_bytes())
    controls = Controls()
    assert 'pandaStates' in controls.sm.services
    assert {'pandaStates', 'spysydriveStateSP'} <= set(get_process_config('controlsd').pubs)


def test_waiting_for_permission_does_not_clear_the_heartbeat_request():
  sm, ext = EngagementSubMaster(), ControlsExt()
  sm.sol(True, 10)
  sm.health(False, 11)
  assert not ext.get_lat_active(sm)
  assert sm['spysydriveStateSP'].aol.enabled
  assert sm['spysydriveStateSP'].aol.active
