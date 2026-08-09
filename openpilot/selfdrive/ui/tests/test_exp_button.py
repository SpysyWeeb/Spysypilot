from types import SimpleNamespace
from unittest.mock import patch

import pyray as rl

from openpilot.selfdrive.ui.onroad import exp_button
from openpilot.selfdrive.ui.onroad.exp_button import ExpButton
from openpilot.system.ui.widgets import Widget


def test_steering_angle_rotates_current_icon():
  button = ExpButton.__new__(ExpButton)
  Widget.__init__(button)
  button._rect = rl.Rectangle(100, 200, 192, 192)
  button._white_color = rl.Color(255, 255, 255, 255)
  button._black_bg = rl.Color(0, 0, 0, 166)
  button._txt_wheel = SimpleNamespace(width=144, height=144)
  button._txt_exp = SimpleNamespace(width=144, height=144)
  button._hold_end_time = None
  button._held_mode = None

  sm = {
    "selfdriveState": SimpleNamespace(experimentalMode=False, engageable=True, enabled=False),
    "carState": SimpleNamespace(steeringAngleDeg=37.5),
  }
  with patch.object(exp_button, "ui_state", SimpleNamespace(sm=sm)):
    button._update_state()

  with patch.object(exp_button.rl, "draw_circle"), patch.object(exp_button.rl, "draw_texture_pro") as draw_texture:
    button._render(button._rect)

  texture, _, destination, origin, rotation, _ = draw_texture.call_args.args
  assert texture is button._txt_wheel
  assert (destination.x, destination.y) == (196, 296)
  assert (origin.x, origin.y) == (72, 72)
  assert rotation == -37.5
