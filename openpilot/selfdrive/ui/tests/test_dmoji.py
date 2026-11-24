"""The comma four driver-monitoring badge on the comma 3X onroad screen (selfdrive/ui/onroad/dmoji.py)."""
import math
from types import SimpleNamespace as NS
from unittest.mock import patch

import pyray as rl
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.ui.onroad import dmoji
from openpilot.selfdrive.ui.onroad.driver_state import BTN_SIZE
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app

AlertSize = log.SelfdriveState.AlertSize
VISION = log.DriverMonitoringState.MonitoringPolicy.vision
CONTENT = rl.Rectangle(30, 30, 2100, 1020)  # the 3X road view inside its 30 px border


class FakeSm(dict):
  def __init__(self, **kw):
    super().__init__(**kw)
    self.recv_frame = {'driverStateV2': 10}


def observation(rhd=False, face=True, yaw=0.2, pitch=0.0, awareness=100.0):
  monitor = NS(isRHD=rhd, activePolicy=VISION,
               visionPolicyState=NS(faceDetected=face, awarenessPercent=awareness, pose=NS(pitch=pitch, yaw=yaw)))
  return monitor, NS(leftDriverData=NS(), rightDriverData=NS())


@pytest.fixture
def env():
  textures = []

  def texture(path, width, height, *args, **kwargs):
    textures.append((path.rsplit('/', 1)[-1], width, height))
    return NS(width=width, height=height)

  monitor, driver = observation()
  sm = FakeSm(driverMonitoringState=monitor, driverStateV2=driver, selfdriveState=NS(alertSize=AlertSize.none))
  with patch.object(gui_app, 'texture', side_effect=texture), patch.object(ui_state, 'sm', sm, create=True), \
       patch.object(ui_state, 'started_frame', 0, create=True):
    yield NS(sm=sm, textures=textures, renderer=dmoji.DMojiRenderer())


def test_drawn_at_the_icons_native_size(env):
  r = env.renderer
  assert (r._rect.width, r._rect.height) == (128, 128)
  # the last load is at 3X size: background at 128, person and cone at the comma four's 52/60 of it
  assert env.textures[-3:] == [('dm_person.png', 111, 111), ('dm_cone.png', 111, 111), ('dm_background.png', 128, 128)]


def test_takes_the_face_icons_place_and_mirrors_for_right_hand_drive(env):
  assert dmoji.FACE_ICON_BUTTON == BTN_SIZE
  r = env.renderer
  r.place(CONTENT)
  # the 3X face icon is centred UI_BORDER_SIZE + BTN_SIZE / 2 = 126 px in from the content's bottom corner
  assert (r._rect.x + 64, r._rect.y + 64) == (30 + 126, 30 + 1020 - 126)
  env.sm['driverMonitoringState'], env.sm['driverStateV2'] = observation(rhd=True)
  r._update_state()
  r.place(CONTENT)
  assert (r._rect.x + 64, r._rect.y + 64) == (30 + 2100 - 126, 30 + 1020 - 126)
  assert (r._rect.width, r._rect.height) == (128, 128)


@pytest.mark.parametrize('rhd', [False, True])
def test_head_yaw_is_negated_on_the_3x(env, rhd):
  env.sm['driverMonitoringState'], env.sm['driverStateV2'] = observation(rhd=rhd, yaw=0.2)
  env.renderer.get_driver_data()
  assert env.renderer._face_yaw == pytest.approx(-0.2)


def test_full_with_a_face_and_dim_without(env):
  r = env.renderer
  r.place(CONTENT)
  r._update_state()
  assert r._fade_filter.x == 1.0
  env.sm['driverMonitoringState'], _ = observation(face=False)
  for _ in range(50):
    r._update_state()
    assert r._fade_filter.x <= dmoji.NO_FACE_FADE
  env.sm['driverMonitoringState'], _ = observation(face=True)
  r._update_state()
  assert r._fade_filter.x == 1.0


@pytest.mark.parametrize('break_it', ['nan_pose', 'awareness_out_of_range', 'stale'])
def test_bad_or_stale_monitoring_keeps_it_dim_and_still(env, break_it):
  r = env.renderer
  r.place(CONTENT)
  r._update_state()
  yaw_before = r._yaw_filter.x
  if break_it == 'nan_pose':
    env.sm['driverMonitoringState'], _ = observation(yaw=math.nan)
  elif break_it == 'awareness_out_of_range':
    env.sm['driverMonitoringState'], _ = observation(awareness=101.0, yaw=0.5)
  else:
    env.sm['driverMonitoringState'], _ = observation(yaw=0.5)
    env.sm.recv_frame['driverStateV2'] = 0  # nothing received since this drive started
  for _ in range(200):
    r._update_state()
  assert not r._is_active and not r._awareness_unfull
  assert r._fade_filter.x == pytest.approx(dmoji.NO_FACE_FADE, abs=0.01)
  assert r._yaw_filter.x == yaw_before


def test_hidden_while_an_alert_is_shown(env):
  r = env.renderer
  r.place(CONTENT)
  r._update_state()
  assert r.should_draw and r._fade_filter.x == 1.0
  env.sm['selfdriveState'] = NS(alertSize=AlertSize.small)
  r.place(CONTENT)
  assert not r.should_draw
  for _ in range(200):
    r._update_state()
  assert r._fade_filter.x < 0.01


def test_valid_observation():
  monitor, driver = observation()
  assert dmoji.valid_observation(monitor, driver)
  for broken in (observation(pitch=math.inf)[0], observation(awareness=-1.0)[0], NS(isRHD=1, visionPolicyState=monitor.visionPolicyState),
                 NS(isRHD=False)):
    assert not dmoji.valid_observation(broken, driver)
  assert not dmoji.valid_observation(monitor, NS(leftDriverData=None, rightDriverData=NS()))
