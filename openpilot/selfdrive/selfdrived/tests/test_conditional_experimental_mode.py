from types import SimpleNamespace

from opendbc.car.structs import car
import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.conditional_experimental_mode import ConditionalExperimentalMode
from openpilot.selfdrive.controls.tests.test_conditional_experimental_mode import (LOW_SPEED_APPROACH, TICKS_PER_FRAME, YELLOW_LIGHT_SPEED,
                                                                                   creeping_stop, model, red_light_request, yellow_light_lead,
                                                                                   yellow_light_stop)
from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveD


class FakeConditionalMode:
  def __init__(self, request):
    self.request = request
    self.calls = []

  def update(self, model, car_state, radar_state, **kwargs):
    self.calls.append(kwargs)
    return self.request


def selfdrived(manual, conditional, enabled=True, long_control=True, model_ok=True, data=None):
  sm = SimpleNamespace(updated={'modelV2': True}, valid={'modelV2': model_ok, 'radarState': True},
                       alive={'modelV2': True, 'radarState': True}, freq_ok={'modelV2': model_ok, 'radarState': True})
  sm.__getitem__ = None
  if data is None:
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


def test_a_yellow_light_the_lead_goes_through_requests_experimental():
  # route 0x1f1 t=544.9 (2026-11-09, D46): selfdrived hands the real mode the car ahead holding 13.5 m/s and the model's strict
  # stop moving between 47 and 57 m; Experimental is requested within the entry filter and debounce, as with no lead
  data = {'modelV2': model(path_end=140.0, terminal_speed=YELLOW_LIGHT_SPEED), 'radarState': yellow_light_lead()}
  sd = selfdrived(False, False, data=data)
  sd.conditional_experimental_mode = ConditionalExperimentalMode()
  cs = messaging.new_message('carState').carState
  cs.vEgo = YELLOW_LIGHT_SPEED
  for tick in range(round(1.0 / DT_CTRL)):
    sd.sm.updated['modelV2'] = tick % TICKS_PER_FRAME == 0
    SelfdriveD.update_experimental_mode(sd, cs)
  assert not sd.experimental_mode
  for tick in range(round(0.5 / DT_CTRL)):
    data['modelV2'] = yellow_light_stop(tick // TICKS_PER_FRAME)
    sd.sm.updated['modelV2'] = tick % TICKS_PER_FRAME == 0
    SelfdriveD.update_experimental_mode(sd, cs)
  assert sd.experimental_mode


def creeping_red_light_request_time():
  data = {'modelV2': model(), 'radarState': messaging.new_message('radarState').radarState}
  sd = selfdrived(False, False, data=data)
  sd.conditional_experimental_mode = ConditionalExperimentalMode()
  cs = messaging.new_message('carState').carState
  cs.vEgo = LOW_SPEED_APPROACH
  requested = None
  for tick in range(round(1.5 / DT_CTRL)):
    i = tick // TICKS_PER_FRAME
    data['modelV2'] = creeping_stop(i, desired_accel=red_light_request(i))
    sd.sm.updated['modelV2'] = tick % TICKS_PER_FRAME == 0
    SelfdriveD.update_experimental_mode(sd, cs)
    if sd.experimental_mode and requested is None:
      requested = tick * DT_CTRL
  return requested


def test_a_creeping_red_light_with_no_car_ahead_requests_experimental(monkeypatch):
  # route 0x1ef t=210.5 (2026-11-09, D47): at 10 m/s with no lead, the plan dipping to a walking pace and creeping on above 1 m/s while
  # the model's request falls through -0.25 on the ninth frame; Experimental is requested within the hint tier's filter and debounce.
  # Since D50 the plan, ending slowly within 60 m with nobody ahead, is entry evidence from its first frame: 0.60 s
  requested = creeping_red_light_request_time()
  assert requested is not None and 0.55 <= requested <= 0.65
  # D47's reading alone: the first braking frame is the ninth (0.40 s); the filter and the debounce take 0.60 s more
  monkeypatch.setattr('openpilot.selfdrive.controls.lib.stop_helpers.SLOW_PLAN_MAX_DISTANCE', 0.0)
  requested = creeping_red_light_request_time()
  assert requested is not None and 0.95 <= requested <= 1.05
