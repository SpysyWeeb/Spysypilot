from unittest.mock import patch

import pyray as rl

from openpilot.cereal import log
from openpilot.common.parameterized import parameterized
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.layouts.settings.developer import DeveloperLayout
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, MousePos


class TestDeveloperLaneCentering(OpenpilotTestCase):
  @parameterized.expand([(False,), (True,)])
  def test_toggle_available_engaged_and_refreshes(self, seed_param):
    params = Params()
    if seed_param:
      params.put_bool("LaneCenteringEnabled", params.get_default_value("LaneCenteringEnabled"), block=True)
    # Exercise real widgets/callbacks without a graphics context or asset uploads.
    with patch.object(gui_app, 'font', return_value=rl.Font()), \
         patch.object(gui_app, 'texture', return_value=rl.Texture()), \
         patch.object(ui_state, 'started', True), \
         patch.dict(ui_state.sm.data, selfdriveState=log.SelfdriveState.new_message(enabled=True)), \
         patch.object(ui_state, '_offroad_transition_callbacks', []):
      layout = DeveloperLayout()
      layout.show_event()
      items = [item for item in layout._scroller._items if item.title == "Lane Centering"]
      assert len(items) == 1
      item = items[0]
      assert ui_state.engaged
      assert item.is_visible and item.action_item.enabled
      assert item.action_item.get_state()

      item.action_item.toggle._handle_mouse_release(MousePos(0, 0))
      assert not Params().get_bool("LaneCenteringEnabled")
      assert not params.get_bool("OnroadCycleRequested")
      item.action_item.toggle._handle_mouse_release(MousePos(0, 0))
      assert Params().get_bool("LaneCenteringEnabled")

      params.put_bool("LaneCenteringEnabled", False, block=True)
      layout.show_event()
      assert not item.action_item.get_state()

  @parameterized.expand([(False,), (True,)])
  def test_mici_toggle_available_engaged_and_refreshes(self, seed_param):
    from openpilot.selfdrive.ui.mici.layouts.settings.developer import DeveloperLayoutMici

    params = Params()
    if seed_param:
      params.put_bool("LaneCenteringEnabled", True, block=True)
    with patch.object(gui_app, 'font', return_value=rl.Font()), \
         patch.object(gui_app, 'texture', return_value=rl.Texture()), \
         patch.object(ui_state, 'params', params), \
         patch.object(ui_state, 'started', True), \
         patch.dict(ui_state.sm.data, selfdriveState=log.SelfdriveState.new_message(enabled=True)), \
         patch.object(ui_state, '_offroad_transition_callbacks', []):
      layout = DeveloperLayoutMici()
      layout.show_event()
      item = layout._lane_centering_toggle
      assert ui_state.engaged
      assert item.is_visible and item.enabled and item._checked
      item._handle_mouse_release(MousePos(0, 0))
      assert not Params().get_bool("LaneCenteringEnabled")
      assert not params.get_bool("OnroadCycleRequested")
      item._handle_mouse_release(MousePos(0, 0))
      assert Params().get_bool("LaneCenteringEnabled")
      params.put_bool("LaneCenteringEnabled", False, block=True)
      layout.show_event()
      assert not item._checked
