from types import SimpleNamespace

import openpilot.selfdrive.selfdrived.selfdrived as selfdrived
from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveD


class FakeSubMaster(dict):
  def __init__(self):
    super().__init__(
      modelV2=SimpleNamespace(),
      radarState=SimpleNamespace(),
    )
    self.updated = {'modelV2': True}
    self.valid = {'modelV2': True}
    self.alive = {'modelV2': True}
    self.freq_ok = {'modelV2': True}


class FakeConditionalMode:
  def __init__(self, conditional_request, driver_override=False):
    self.conditional_request = conditional_request
    self.driver_override_active = driver_override
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
  }


def test_manual_request_and_conditional_request_resolve_in_one_place():
  CS = SimpleNamespace()
  manual = selfdrive_for_mode_update(conditional_request=False, manual_request=True)
  manual.update_experimental_mode(CS)
  assert manual.experimental_mode

  chill = selfdrive_for_mode_update(conditional_request=False, manual_request=False)
  chill.update_experimental_mode(CS)
  assert not chill.experimental_mode


def test_driver_override_has_priority_over_manual_and_conditional_requests():
  instance = selfdrive_for_mode_update(conditional_request=True, manual_request=True, driver_override=True)
  instance.update_experimental_mode(SimpleNamespace())
  assert not instance.experimental_mode


def test_mode_is_disabled_without_openpilot_longitudinal_control():
  instance = selfdrive_for_mode_update(conditional_request=True, manual_request=True, longitudinal=False)
  instance.update_experimental_mode(SimpleNamespace())
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
  instance.personality = 0
  instance.state_machine = SimpleNamespace(state=1)
  instance.events = SimpleNamespace(contains=lambda event_type: False, names=[])
  instance.events_prev = []
  instance.sm = SimpleNamespace(frame=1)
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
