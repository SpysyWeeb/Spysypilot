from types import SimpleNamespace

from opendbc.car.structs import car
import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveD


class FakeConditionalMode:
  def __init__(self, request):
    self.request = request
    self.calls = []

  def update(self, model, car_state, radar_state, **kwargs):
    self.calls.append(kwargs)
    return self.request


def selfdrived(manual, conditional, enabled=True, long_control=True, model_ok=True):
  sm = SimpleNamespace(updated={'modelV2': True}, valid={'modelV2': model_ok, 'radarState': True},
                       alive={'modelV2': True, 'radarState': True}, freq_ok={'modelV2': model_ok, 'radarState': True})
  sm.__getitem__ = None
  data = {'modelV2': messaging.new_message('modelV2').modelV2, 'radarState': messaging.new_message('radarState').radarState}
  sm = type('SM', (), {'__getitem__': lambda self, k: data[k], 'updated': sm.updated, 'valid': sm.valid, 'alive': sm.alive, 'freq_ok': sm.freq_ok})()
  return SimpleNamespace(sm=sm, enabled=enabled, CP=car.CarParams.new_message(openpilotLongitudinalControl=long_control),
                         conditional_experimental_mode=FakeConditionalMode(conditional), manual_experimental_mode=manual and long_control,
                         experimental_mode=False)


def test_manual_and_conditional_requests_resolve_in_one_place():
  for manual, conditional, expected in ((False, False, False), (True, False, True), (False, True, True), (True, True, True)):
    sd = selfdrived(manual, conditional)
    SelfdriveD.update_experimental_mode(sd, messaging.new_message('carState').carState)
    assert sd.experimental_mode is expected


def test_manual_mode_survives_a_pedal_tap():
  # the conditional request already drops on a pedal; the manual setting must not
  sd = selfdrived(True, False)
  cs = messaging.new_message('carState').carState
  cs.gasPressed = True
  SelfdriveD.update_experimental_mode(sd, cs)
  assert sd.experimental_mode


def test_conditional_mode_needs_openpilot_longitudinal_control_and_engagement():
  sd = selfdrived(False, True, long_control=False)
  SelfdriveD.update_experimental_mode(sd, messaging.new_message('carState').carState)
  assert not sd.conditional_experimental_mode.calls[-1]['controls_enabled']
  sd = selfdrived(False, True, enabled=False)
  SelfdriveD.update_experimental_mode(sd, messaging.new_message('carState').carState)
  assert not sd.conditional_experimental_mode.calls[-1]['controls_enabled']


def test_model_health_reaches_the_conditional_mode():
  sd = selfdrived(False, True, model_ok=False)
  SelfdriveD.update_experimental_mode(sd, messaging.new_message('carState').carState)
  assert not sd.conditional_experimental_mode.calls[-1]['model_valid']
  assert sd.conditional_experimental_mode.calls[-1]['radar_valid']
