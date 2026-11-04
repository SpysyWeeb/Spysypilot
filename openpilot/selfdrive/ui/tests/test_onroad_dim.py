import pytest

from openpilot.cereal import log
from openpilot.selfdrive.ui import ui_state as ui_state_module
from openpilot.selfdrive.ui.ui_state import Device, UIStatus, ONROAD_DIM_BRIGHTNESS, ONROAD_DIM_FADE_TIME, ONROAD_DIM_WAKE_TIME, ui_state

AlertSize = log.SelfdriveState.AlertSize
DT = 0.05  # 20 Hz steps, well under the 0.25 s clamp
STARTED_FRAME = 10


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
  monkeypatch.setattr(ui_state, 'light_sensor', -1)  # ambient target is the deterministic offroad value

  def make_device(ambient=50):
    dev = Device()
    dev._offroad_brightness = ambient
    dev._brightness_filter.x = ambient
    dev._interaction_time = clock.t - 1  # last touch long ago
    return dev
  return clock, sm, make_device


def step(clock, dev, n=1, dt=DT):
  out = []
  for _ in range(n):
    clock.t += dt
    dev._update_brightness()
    out.append(dev._last_brightness)
  return out


def fade_fully(clock, dev):
  vals = step(clock, dev, round(ONROAD_DIM_FADE_TIME / DT))
  assert vals[-1] == ONROAD_DIM_BRIGHTNESS
  return vals


@pytest.mark.parametrize('ambient', [30, 50, 100])
def test_quiet_fades_to_dim_over_fade_time_and_holds(env, ambient):
  clock, _, make_device = env
  dev = make_device(ambient)
  vals = step(clock, dev, round(ONROAD_DIM_FADE_TIME / DT))
  assert all(a >= b for a, b in zip(vals, vals[1:], strict=False))
  assert vals[0] >= ambient - 1  # starts at ambient, no jump
  half = len(vals) // 2 - 1  # 2.5 s into the fade: linear blend is halfway
  assert abs(vals[half] - (ambient + ONROAD_DIM_BRIGHTNESS) / 2) <= 1
  assert vals[-1] == ONROAD_DIM_BRIGHTNESS
  assert dev._brightness_target == ONROAD_DIM_BRIGHTNESS
  assert set(step(clock, dev, 200)) == {ONROAD_DIM_BRIGHTNESS}


def test_owner_numbers_are_pinned(env):
  # owner decisions are fixed: 10 percent, 5 s fade, 1 s wake; literals here so a constant change fails
  assert (ONROAD_DIM_BRIGHTNESS, ONROAD_DIM_FADE_TIME, ONROAD_DIM_WAKE_TIME) == (10, 5.0, 1.0)
  clock, sm, make_device = env
  dev = make_device(100)
  assert abs(step(clock, dev, round(4.0 / DT))[-1] - 28) <= 1  # 4 s in: 100 - 90 * 0.8, not yet dim
  assert step(clock, dev, round(1.0 / DT))[-1] == 10  # 5 s in: fully dim
  sm['selfdriveState'].alertSize = AlertSize.small
  assert abs(step(clock, dev, round(0.5 / DT))[-1] - 55) <= 1  # 0.5 s into the wake: halfway, not already 100
  assert step(clock, dev, round(0.5 / DT))[-1] == 100  # 1 s: fully awake


def test_already_dim_ambient_never_brightens(env):
  clock, _, make_device = env
  dev = make_device(5)
  assert set(step(clock, dev, 200)) == {5}


@pytest.mark.parametrize('size', [AlertSize.small, AlertSize.mid, AlertSize.full])
def test_any_alert_restores_within_wake_time(env, size):
  clock, sm, make_device = env
  dev = make_device(30)
  fade_fully(clock, dev)
  sm['selfdriveState'].alertSize = size
  vals = step(clock, dev, round(ONROAD_DIM_WAKE_TIME / DT))
  assert all(a <= b for a, b in zip(vals, vals[1:], strict=False))
  assert vals[-1] == 30
  assert dev._brightness_target == 30
  assert set(step(clock, dev, 200)) == {30}


@pytest.mark.parametrize('status', [UIStatus.OVERRIDE, UIStatus.DISENGAGED, UIStatus.AOL_ACTIVE])
def test_non_engaged_status_never_dims_and_wakes(env, monkeypatch, status):
  clock, _, make_device = env
  monkeypatch.setattr(ui_state, 'status', status)
  dev = make_device()
  assert set(step(clock, dev, 200)) == {50}

  monkeypatch.setattr(ui_state, 'status', UIStatus.ENGAGED)
  fade_fully(clock, dev)
  monkeypatch.setattr(ui_state, 'status', status)
  assert step(clock, dev, round(ONROAD_DIM_WAKE_TIME / DT))[-1] == 50


@pytest.mark.parametrize('stale', ['alive', 'valid', 'recv_frame_equal', 'recv_frame_old'])
def test_stale_selfdrive_state_never_dims_and_wakes(env, stale):
  clock, sm, make_device = env

  def make_stale():
    if stale in ('alive', 'valid'):
      getattr(sm, stale)['selfdriveState'] = False
    else:
      sm.recv_frame['selfdriveState'] = STARTED_FRAME if stale == 'recv_frame_equal' else STARTED_FRAME - 5

  dev = make_device()
  fade_fully(clock, dev)
  make_stale()
  assert step(clock, dev, round(ONROAD_DIM_WAKE_TIME / DT))[-1] == 50
  assert set(step(clock, dev, 200)) == {50}


def test_touch_wakes_and_blocks_until_interactive_timeout(env):
  clock, _, make_device = env
  dev = make_device()
  fade_fully(clock, dev)
  dev._interaction_time = clock.t + 10  # touch with a 10 s interactive timeout
  vals = step(clock, dev, round(ONROAD_DIM_WAKE_TIME / DT))
  assert vals[-1] == 50
  vals = step(clock, dev, round(8.5 / DT))
  assert set(vals) == {50}
  # after the timeout expires, fading resumes
  vals = step(clock, dev, round(1.0 / DT) + round(ONROAD_DIM_FADE_TIME / DT))
  assert vals[-1] == ONROAD_DIM_BRIGHTNESS


def test_going_offroad_resets_at_once(env, monkeypatch):
  clock, _, make_device = env
  dev = make_device()
  fade_fully(clock, dev)
  monkeypatch.setattr(ui_state, 'started', False)
  assert step(clock, dev)[-1] == 50
  assert dev._dim_fraction == 0.0
  assert dev._brightness_target == 50

  # coming back onroad starts the fade from full ambient, not from the old fraction
  monkeypatch.setattr(ui_state, 'started', True)
  assert step(clock, dev)[-1] >= 49


def test_stall_between_updates_is_clamped(env):
  clock, sm, make_device = env
  dev = make_device(100)
  step(clock, dev)
  before = dev._dim_fraction
  step(clock, dev, dt=10.0)
  assert dev._dim_fraction - before == pytest.approx(0.25 / ONROAD_DIM_FADE_TIME)

  fade_fully(clock, dev)
  sm['selfdriveState'].alertSize = AlertSize.small
  step(clock, dev, dt=10.0)
  assert dev._dim_fraction == pytest.approx(1.0 - 0.25 / ONROAD_DIM_WAKE_TIME)


def test_not_awake_still_yields_zero(env):
  clock, sm, make_device = env
  dev = make_device()
  dev._awake = False
  assert set(step(clock, dev, 50)) == {0}
  sm['selfdriveState'].alertSize = AlertSize.full
  assert set(step(clock, dev, 50)) == {0}
  assert dev._brightness_target == 0
