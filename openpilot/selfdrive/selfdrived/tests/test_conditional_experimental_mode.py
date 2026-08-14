from types import SimpleNamespace

import openpilot.selfdrive.selfdrived.selfdrived as selfdrived
from openpilot.selfdrive.controls.lib.conditional_experimental_mode import ConditionalExperimentalMode
from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveD


class FakeSubMaster(dict):
  def __init__(self):
    super().__init__(
      modelV2=SimpleNamespace(),
      radarState=SimpleNamespace(),
    )
    self.updated = {'modelV2': True}
    self.valid = {'modelV2': True, 'radarState': True}
    self.alive = {'modelV2': True, 'radarState': True}
    self.freq_ok = {'modelV2': True, 'radarState': True}


class FakeConditionalMode:
  def __init__(self, conditional_request, driver_override=False, stop_qualified=False, stop_distance=None):
    self.conditional_request = conditional_request
    self.driver_override_active = driver_override
    self.stop_qualified = stop_qualified
    self.stop_distance = stop_distance
    self.calls = []

  def update(self, *args, **kwargs):
    self.calls.append((args, kwargs))
    return self.conditional_request


def selfdrive_for_mode_update(*, conditional_request, manual_request=False, driver_override=False, longitudinal=True):
  instance = SelfdriveD.__new__(SelfdriveD)
  instance.CP = SimpleNamespace(openpilotLongitudinalControl=longitudinal)
  instance.enabled = True
  instance.manual_experimental_mode = manual_request
  instance.experimental_mode = False
  instance.conditional_experimental_mode = FakeConditionalMode(conditional_request, driver_override)
  instance.sm = FakeSubMaster()
  return instance


def test_selfdrived_is_single_effective_mode_owner():
  CS = SimpleNamespace()
  instance = selfdrive_for_mode_update(conditional_request=True)

  instance.update_experimental_mode(CS)

  assert instance.experimental_mode
  assert len(instance.conditional_experimental_mode.calls) == 1
  args, kwargs = instance.conditional_experimental_mode.calls[0]
  assert args == (instance.sm['modelV2'], CS, instance.sm['radarState'])
  assert kwargs == {
    'controls_enabled': True,
    'model_updated': True,
    'model_valid': True,
    'radar_valid': True,
  }


def test_selfdrived_passes_invalid_radar_health_to_conditional_mode():
  instance = selfdrive_for_mode_update(conditional_request=False)
  instance.sm.alive['radarState'] = False

  instance.update_experimental_mode(SimpleNamespace())

  _, kwargs = instance.conditional_experimental_mode.calls[0]
  assert kwargs['radar_valid'] is False


def test_manual_request_and_conditional_request_resolve_in_one_place():
  CS = SimpleNamespace()
  manual = selfdrive_for_mode_update(conditional_request=False, manual_request=True)
  manual.update_experimental_mode(CS)
  assert manual.experimental_mode

  chill = selfdrive_for_mode_update(conditional_request=False, manual_request=False)
  chill.update_experimental_mode(CS)
  assert not chill.experimental_mode


def test_manual_toggle_ignores_temporary_conditional_state():
  writes = []
  instance = SelfdriveD.__new__(SelfdriveD)
  instance.params = SimpleNamespace(put_bool=lambda key, value: writes.append((key, value)))
  instance.manual_experimental_mode = False
  instance.experimental_mode = True

  instance.toggle_manual_experimental_mode()

  assert instance.manual_experimental_mode
  assert writes == [('ExperimentalMode', True)]


def test_driver_override_has_priority_over_manual_and_conditional_requests():
  instance = selfdrive_for_mode_update(conditional_request=True, manual_request=True, driver_override=True)
  instance.update_experimental_mode(SimpleNamespace())
  assert not instance.experimental_mode


def test_mode_is_disabled_without_openpilot_longitudinal_control():
  instance = selfdrive_for_mode_update(conditional_request=True, manual_request=True, longitudinal=False)
  instance.update_experimental_mode(SimpleNamespace())
  assert not instance.experimental_mode


def test_real_stop_detector_request_reaches_effective_mode_and_releases():
  instance = selfdrive_for_mode_update(conditional_request=False)
  instance.conditional_experimental_mode = ConditionalExperimentalMode(control_dt=0.05, model_dt=0.05)
  instance.sm['modelV2'] = SimpleNamespace(
    action=SimpleNamespace(shouldStop=False, desiredAcceleration=-0.3, desiredCurvature=0.0),
    position=SimpleNamespace(x=[0.0, 30.0]),
    velocity=SimpleNamespace(x=[10.0, 0.2]),
  )
  instance.sm['radarState'] = SimpleNamespace(leadOne=SimpleNamespace(present=False, dRel=1000.0))
  CS = SimpleNamespace(
    vEgo=10.0,
    standstill=False,
    gasPressed=False,
    brakePressed=False,
    leftBlinker=False,
    rightBlinker=False,
    steeringAngleDeg=0.0,
  )

  for _ in range(30):
    instance.update_experimental_mode(CS)
  assert instance.experimental_mode

  instance.sm['modelV2'].action.shouldStop = False
  instance.sm['modelV2'].action.desiredAcceleration = 0.0
  instance.sm['modelV2'].position.x = [0.0, 90.0]
  instance.sm['modelV2'].velocity.x = [10.0, 10.0]
  for _ in range(100):
    instance.update_experimental_mode(CS)
  assert not instance.experimental_mode


def test_publish_selfdrive_state_uses_effective_experimental_mode(monkeypatch):
  selfdrive_state = SimpleNamespace()
  message = SimpleNamespace(valid=False, selfdriveState=selfdrive_state)
  monkeypatch.setattr(selfdrived.messaging, 'new_message', lambda *args, **kwargs: message)

  sent = {}
  instance = SelfdriveD.__new__(SelfdriveD)
  instance.pm = SimpleNamespace(send=lambda service, msg: sent.update({service: msg}))
  instance.enabled = True
  instance.active = True
  instance.experimental_mode = True
  instance.conditional_experimental_mode = FakeConditionalMode(True, stop_qualified=True, stop_distance=42.5)
  instance.personality = 0
  instance.state_machine = SimpleNamespace(state=1)
  instance.events = SimpleNamespace(contains=lambda event_type: False, names=[])
  instance.events_prev = []
  instance.sm = SimpleNamespace(frame=1, logMonoTime={'modelV2': 123456789})
  instance.AM = SimpleNamespace(current_alert=SimpleNamespace(
    alert_text_1='',
    alert_text_2='',
    alert_size=0,
    alert_status=0,
    alert_type='',
    audible_alert=0,
    visual_alert=0,
  ))

  instance.publish_selfdriveState(SimpleNamespace())

  assert sent['selfdriveState'].selfdriveState.experimentalMode is True
  assert sent['selfdriveState'].selfdriveState.conditionalStopQualified is True
  assert sent['selfdriveState'].selfdriveState.conditionalStopDistance == 42.5
  assert sent['selfdriveState'].selfdriveState.conditionalStopModelMonoTime == 123456789
