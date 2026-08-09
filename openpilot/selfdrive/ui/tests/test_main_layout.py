import unittest
from unittest.mock import Mock

from openpilot.selfdrive.ui.layouts.main import MainLayout, MainState
from openpilot.system.ui.widgets import Widget


class TestMainLayout(unittest.TestCase):
  def test_show_event_forwards_to_active_layout(self):
    main = MainLayout.__new__(MainLayout)
    Widget.__init__(main)
    active_layout = Mock(spec=Widget)
    inactive_layout = Mock(spec=Widget)
    main._current_mode = MainState.HOME
    main._layouts = {
      MainState.HOME: active_layout,
      MainState.SETTINGS: inactive_layout,
    }

    main.show_event()

    active_layout.show_event.assert_called_once_with()
    inactive_layout.show_event.assert_not_called()
