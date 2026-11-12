import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui import ui_state as ui_state_module
from openpilot.selfdrive.ui.ui_state import (BACKLIGHT_OFFROAD, Device, UIStatus, ONROAD_ALERT_BRIGHTNESS, ONROAD_ALERT_RELEASE_TIME,
                                             ONROAD_DIM_BRIGHTNESS, ONROAD_DIM_KNEE, onroad_brightness, ui_state)

AlertSize = log.SelfdriveState.AlertSize
AlertStatus = log.SelfdriveState.AlertStatus
DT = 0.05  # 20 Hz frames, the comma 3X UI rate
STARTED_FRAME = 10


def cie(light_sensor):
  return light_sensor / 903.3 if light_sensor <= 8 else ((light_sensor + 16.0) / 116.0) ** 3.0


def old_onroad_brightness(light_sensor):
  # the automatic onroad brightness before this change (upstream's): the CIE curve mapped linearly onto 30..100 percent
  return float(np.interp(cie(light_sensor), [0, 1], [30, 100]))


class Sm(dict):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.alive = {'selfdriveState': True}
    self.valid = {'selfdriveState': True}
    self.recv_frame = {'selfdriveState': STARTED_FRAME + 1}


class Clock:
  def __init__(self):
    self.t = 1000.0

  def __call__(self):
    return self.t


@pytest.fixture
def env(monkeypatch):
  clock = Clock()
  monkeypatch.setattr(ui_state_module.time, 'monotonic', clock)
  sm = Sm(selfdriveState=log.SelfdriveState.new_message())
  monkeypatch.setattr(ui_state, 'sm', sm)
  monkeypatch.setattr(ui_state, 'started', True)
  monkeypatch.setattr(ui_state, 'status', UIStatus.ENGAGED)
  monkeypatch.setattr(ui_state, 'started_frame', STARTED_FRAME)
  monkeypatch.setattr(ui_state, 'light_sensor', 0.0)

  def make_device(light_sensor=0.0):
    monkeypatch.setattr(ui_state, 'light_sensor', light_sensor)
    dev = Device()
    dev._brightness_filter.x = onroad_brightness(light_sensor)  # the slow ambient filter has settled
    return dev
  return clock, sm, make_device


def step(clock, dev, n=1, dt=DT):
  out = []
  for _ in range(n):
    clock.t += dt
    dev._update_brightness()
    out.append(dev._last_brightness)
  return out


def test_owner_numbers_are_pinned():
  # 5 percent in the dark, knee at 0.1 (37 percent), alerts at least the old 30 percent floor, 1 s back
  assert (ONROAD_DIM_BRIGHTNESS, ONROAD_DIM_KNEE, ONROAD_ALERT_BRIGHTNESS, ONROAD_ALERT_RELEASE_TIME) == (5, 0.1, 30, 1.0)


@pytest.mark.parametrize('status', list(UIStatus))
def test_dark_gives_the_dim_level(env, monkeypatch, status):
  clock, _, make_device = env
  monkeypatch.setattr(ui_state, 'status', status)
  assert onroad_brightness(0.0) == ONROAD_DIM_BRIGHTNESS
  dev = make_device(0.0)
  assert set(step(clock, dev, 200)) == {ONROAD_DIM_BRIGHTNESS}
  assert dev._brightness_target == ONROAD_DIM_BRIGHTNESS


def test_mapping_is_the_old_line_at_and_above_the_knee():
  readings = np.linspace(0.0, 100.0, 20001)
  above = [L for L in readings if cie(L) >= ONROAD_DIM_KNEE]
  assert len(above) > 10000
  for L in above:
    assert onroad_brightness(L) == old_onroad_brightness(L), L
  assert onroad_brightness(100.0) == old_onroad_brightness(100.0) == 100


def test_below_the_knee_it_runs_from_the_knee_value_down_to_the_dim_level():
  knee_value = float(np.interp(ONROAD_DIM_KNEE, [0, 1], [30, 100]))
  assert knee_value == pytest.approx(37)
  below = [L for L in np.linspace(0.0, 40.0, 4001) if cie(L) < ONROAD_DIM_KNEE]
  assert len(below) > 3000
  for L in below:
    x = cie(L)
    assert onroad_brightness(L) == pytest.approx(ONROAD_DIM_BRIGHTNESS + (knee_value - ONROAD_DIM_BRIGHTNESS) * x / ONROAD_DIM_KNEE)
    assert ONROAD_DIM_BRIGHTNESS <= onroad_brightness(L) <= old_onroad_brightness(L)
  # continuous at the knee and never decreasing with more light
  values = [onroad_brightness(L) for L in np.linspace(0.0, 100.0, 20001)]
  assert all(a <= b for a, b in zip(values, values[1:], strict=False))
  assert onroad_brightness(below[-1]) == pytest.approx(knee_value, abs=0.05)


@pytest.mark.parametrize('status', list(UIStatus))
def test_above_the_knee_the_screen_is_frame_for_frame_the_old_automatic_brightness(env, monkeypatch, status):
  # the old code dimmed only while engaged; disengaged it showed exactly round(filter(old line)). Alerts on a bright screen change nothing.
  clock, sm, make_device = env
  monkeypatch.setattr(ui_state, 'status', status)
  readings = 69.0 + 31.0 * np.sin(np.linspace(0.0, 12.0, 600))  # 38..100, all at or above the knee
  assert min(cie(L) for L in readings) >= ONROAD_DIM_KNEE
  dev = make_device(readings[0])
  ref = FirstOrderFilter(dev._brightness_filter.x, 10.0, dev._brightness_filter.dt)
  for i, L in enumerate(readings):
    monkeypatch.setattr(ui_state, 'light_sensor', float(L))
    sm['selfdriveState'].alertSize = AlertSize.mid if (i // 50) % 2 else AlertSize.none
    assert step(clock, dev)[-1] == round(ref.update(old_onroad_brightness(L)))


def run_drive(clock, sm, dev, monkeypatch):
  # dark, an alert, its release, a stale selfdriveState, then daylight
  out = []
  for light, size, alive, n in [(0.0, AlertSize.none, True, 100), (0.0, AlertSize.small, True, 30), (0.0, AlertSize.none, True, 40),
                                (0.0, AlertSize.none, False, 10), (14.5, AlertSize.none, True, 100), (99.5, AlertSize.none, True, 300)]:
    monkeypatch.setattr(ui_state, 'light_sensor', light)
    sm['selfdriveState'].alertSize = size
    sm.alive['selfdriveState'] = alive
    out += step(clock, dev, n)
  return out


def test_engaged_or_not_makes_no_difference(env, monkeypatch):
  clock, sm, make_device = env
  runs = {}
  for status in UIStatus:
    monkeypatch.setattr(ui_state, 'status', status)
    clock.t = 1000.0
    runs[status] = run_drive(clock, sm, make_device(0.0), monkeypatch)
  assert all(r == runs[UIStatus.ENGAGED] for r in runs.values())
  assert min(runs[UIStatus.ENGAGED]) == ONROAD_DIM_BRIGHTNESS


@pytest.mark.parametrize('size', [AlertSize.small, AlertSize.mid, AlertSize.full])
@pytest.mark.parametrize('status', [AlertStatus.normal, AlertStatus.userPrompt, AlertStatus.critical])
def test_any_alert_lifts_a_dark_screen_on_its_first_frame(env, size, status):
  clock, sm, make_device = env
  dev = make_device(0.0)
  assert step(clock, dev, 20)[-1] == ONROAD_DIM_BRIGHTNESS
  sm['selfdriveState'].alertSize = size
  sm['selfdriveState'].alertStatus = status
  assert step(clock, dev) == [ONROAD_ALERT_BRIGHTNESS]
  assert dev._brightness_target == ONROAD_ALERT_BRIGHTNESS
  assert set(step(clock, dev, round(10.0 / DT))) == {ONROAD_ALERT_BRIGHTNESS}  # held for as long as the alert is up


@pytest.mark.parametrize('stale', ['alive', 'valid', 'recv_frame_equal', 'recv_frame_old'])
def test_stale_selfdrive_state_lifts_a_dark_screen_on_its_first_frame(env, stale):
  # the UI's own "Waiting to start" / "System Unresponsive" (TAKE CONTROL IMMEDIATELY) alerts come from a stale selfdriveState
  clock, sm, make_device = env
  dev = make_device(0.0)
  assert step(clock, dev, 20)[-1] == ONROAD_DIM_BRIGHTNESS
  if stale in ('alive', 'valid'):
    getattr(sm, stale)['selfdriveState'] = False
  else:
    sm.recv_frame['selfdriveState'] = STARTED_FRAME if stale == 'recv_frame_equal' else STARTED_FRAME - 5
  assert step(clock, dev) == [ONROAD_ALERT_BRIGHTNESS]
  assert set(step(clock, dev, round(20.0 / DT))) == {ONROAD_ALERT_BRIGHTNESS}


def test_after_the_alert_it_eases_back_over_the_release_time(env):
  clock, sm, make_device = env
  dev = make_device(0.0)
  sm['selfdriveState'].alertSize = AlertSize.small
  step(clock, dev, 20)
  sm['selfdriveState'].alertSize = AlertSize.none
  n = round(ONROAD_ALERT_RELEASE_TIME / DT)
  vals = step(clock, dev, n)
  assert all(a >= b for a, b in zip(vals, vals[1:], strict=False))
  assert ONROAD_DIM_BRIGHTNESS < vals[0] < ONROAD_ALERT_BRIGHTNESS  # not held, not dropped in one frame
  assert abs(vals[n // 2 - 1] - (ONROAD_ALERT_BRIGHTNESS + ONROAD_DIM_BRIGHTNESS) / 2) <= 1  # linear: halfway at half the time
  assert vals[-1] == ONROAD_DIM_BRIGHTNESS
  assert set(step(clock, dev, 200)) == {ONROAD_DIM_BRIGHTNESS}

  # an alert that comes back during the release lifts it at once again
  sm['selfdriveState'].alertSize = AlertSize.small
  step(clock, dev, 20)
  sm['selfdriveState'].alertSize = AlertSize.none
  assert step(clock, dev, n // 2)[-1] < ONROAD_ALERT_BRIGHTNESS
  sm['selfdriveState'].alertSize = AlertSize.full
  sm['selfdriveState'].alertStatus = AlertStatus.critical
  assert step(clock, dev) == [ONROAD_ALERT_BRIGHTNESS]


def test_an_alert_never_dims_a_brighter_screen(env):
  clock, sm, make_device = env
  dev = make_device(99.5)
  bright = step(clock, dev, 20)[-1]
  assert bright > ONROAD_ALERT_BRIGHTNESS
  sm['selfdriveState'].alertSize = AlertSize.full
  sm['selfdriveState'].alertStatus = AlertStatus.critical
  assert set(step(clock, dev, 40)) == {bright}
  sm['selfdriveState'].alertSize = AlertSize.none
  assert set(step(clock, dev, 40)) == {bright}


def test_asleep_gives_zero(env):
  clock, sm, make_device = env
  for light in (0.0, 99.5):
    dev = make_device(light)
    dev._awake = False
    assert set(step(clock, dev, 50)) == {0}
    sm['selfdriveState'].alertSize = AlertSize.full
    sm['selfdriveState'].alertStatus = AlertStatus.critical
    assert set(step(clock, dev, 50)) == {0}
    assert dev._brightness_target == 0
    sm['selfdriveState'].alertSize = AlertSize.none


def test_offroad_is_unchanged(env, monkeypatch):
  # offroad: the offroad brightness through the ambient filter, nothing else; alerts and the light sensor are ignored
  clock, sm, make_device = env
  dev = make_device(0.0)
  sm['selfdriveState'].alertSize = AlertSize.full
  assert step(clock, dev)[-1] == ONROAD_ALERT_BRIGHTNESS
  monkeypatch.setattr(ui_state, 'started', False)
  ref = FirstOrderFilter(dev._brightness_filter.x, 10.0, dev._brightness_filter.dt)
  assert step(clock, dev, 300) == [round(ref.update(BACKLIGHT_OFFROAD)) for _ in range(300)]  # the lift ends at once
  dev.set_offroad_brightness(100)
  sm.alive['selfdriveState'] = False
  assert step(clock, dev, 300) == [round(ref.update(100)) for _ in range(300)]

  # back onroad without a light reading: the offroad brightness, as before
  monkeypatch.setattr(ui_state, 'started', True)
  monkeypatch.setattr(ui_state, 'light_sensor', -1)
  sm.alive['selfdriveState'] = True
  sm['selfdriveState'].alertSize = AlertSize.none
  assert step(clock, dev, 300) == [round(ref.update(100)) for _ in range(300)]
