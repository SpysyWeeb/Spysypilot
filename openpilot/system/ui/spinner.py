#!/usr/bin/env python3
import os
import pyray as rl
import re
import select
import sys

from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.text import wrap_text
from openpilot.system.ui.widgets import Widget

# Constants
if gui_app.big_ui():
  PROGRESS_BAR_WIDTH = 1000
  PROGRESS_BAR_HEIGHT = 20
  TEXTURE_SIZE = 360
  WRAPPED_SPACING = 50
  CENTERED_SPACING = 150
  CONSOLE_FONT_SIZE = 38
  CONSOLE_LINE_HEIGHT = 46
else:
  PROGRESS_BAR_WIDTH = 268
  PROGRESS_BAR_HEIGHT = 10
  TEXTURE_SIZE = 140
  WRAPPED_SPACING = 10
  CENTERED_SPACING = 20
  CONSOLE_FONT_SIZE = 20
  CONSOLE_LINE_HEIGHT = 25
DEGREES_PER_SECOND = 360.0  # one full rotation per second
MARGIN_H = 100
FONT_SIZE = 96
LINE_HEIGHT = 104
DARKGRAY = (55, 55, 55, 255)
LOG_PREFIX = "LOG:"
MAX_CONSOLE_LINES = 100   # history buffer; rendered lines are capped by screen height
MAX_LINE_CHARS = 140
CONSOLE_MARGIN = 20
CONSOLE_ALPHA = 191  # ~75% opacity
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[mKJH]')


def clamp(value, min_value, max_value):
  return max(min(value, max_value), min_value)


class Spinner(Widget):
  def __init__(self):
    super().__init__()
    self._comma_texture = gui_app.texture("images/spinner_comma.png", TEXTURE_SIZE, TEXTURE_SIZE)
    self._spinner_texture = gui_app.texture("images/spinner_track.png", TEXTURE_SIZE, TEXTURE_SIZE, alpha_premultiply=True)
    self._rotation = 0.0
    self._progress: int | None = None
    self._wrapped_lines: list[str] = []
    # Show a default line immediately so the screen isn't blank while waiting for FIFO
    self._console_lines: list[str] = ["Starting spysypilot..."]

  def set_text(self, text: str) -> None:
    if text.startswith(LOG_PREFIX):
      line = ANSI_ESCAPE.sub('', text[len(LOG_PREFIX):]).strip()[:MAX_LINE_CHARS]
      if line:
        self._console_lines.append(line)
        if len(self._console_lines) > MAX_CONSOLE_LINES:
          self._console_lines.pop(0)
    elif text.isdigit():
      self._progress = clamp(int(text), 0, 100)
      self._wrapped_lines = []
    else:
      self._progress = None
      self._wrapped_lines = wrap_text(text, FONT_SIZE, gui_app.width - MARGIN_H)

  def _render(self, rect: rl.Rectangle):
    if self._wrapped_lines:
      spacing = WRAPPED_SPACING
      total_height = TEXTURE_SIZE + spacing + len(self._wrapped_lines) * LINE_HEIGHT
      center_y = (rect.height - total_height) / 2.0 + TEXTURE_SIZE / 2.0
    else:
      spacing = CENTERED_SPACING
      center_y = rect.height / 2.0
    y_pos = center_y + TEXTURE_SIZE / 2.0 + spacing

    center = rl.Vector2(rect.width / 2.0, center_y)
    spinner_origin = rl.Vector2(TEXTURE_SIZE / 2.0, TEXTURE_SIZE / 2.0)
    comma_position = rl.Vector2(center.x - TEXTURE_SIZE / 2.0, center.y - TEXTURE_SIZE / 2.0)

    delta_time = rl.get_frame_time()
    self._rotation = (self._rotation + DEGREES_PER_SECOND * delta_time) % 360.0

    # Draw rotating spinner and static comma logo
    rl.draw_texture_pro(self._spinner_texture, rl.Rectangle(0, 0, TEXTURE_SIZE, TEXTURE_SIZE),
                        rl.Rectangle(center.x, center.y, TEXTURE_SIZE, TEXTURE_SIZE),
                        spinner_origin, self._rotation, rl.WHITE)
    rl.draw_texture_v(self._comma_texture, comma_position, rl.WHITE)

    # Display the progress bar or text based on user input
    if self._progress is not None:
      bar = rl.Rectangle(center.x - PROGRESS_BAR_WIDTH / 2.0, y_pos, PROGRESS_BAR_WIDTH, PROGRESS_BAR_HEIGHT)
      rl.draw_rectangle_rounded(bar, 1, 10, DARKGRAY)
      bar.width *= self._progress / 100.0
      rl.draw_rectangle_rounded(bar, 1, 10, rl.WHITE)
    elif self._wrapped_lines:
      for i, line in enumerate(self._wrapped_lines):
        text_size = measure_text_cached(gui_app.font(), line, FONT_SIZE)
        rl.draw_text_ex(gui_app.font(), line, rl.Vector2(center.x - text_size.x / 2, y_pos + i * LINE_HEIGHT),
                        FONT_SIZE, 0.0, rl.WHITE)

    # Console log overlay at ~75% opacity — fill from bottom up, as many lines as the screen fits
    if self._console_lines:
      max_visible = max(1, (int(rect.height) - CONSOLE_MARGIN * 2) // CONSOLE_LINE_HEIGHT)
      visible = self._console_lines[-max_visible:]
      n = len(visible)
      total_h = n * CONSOLE_LINE_HEIGHT + CONSOLE_MARGIN * 2
      bg_rect = rl.Rectangle(0, rect.height - total_h, rect.width, total_h)
      rl.draw_rectangle_rec(bg_rect, rl.Color(0, 0, 0, 128))
      text_color = rl.Color(255, 255, 255, CONSOLE_ALPHA)
      for i, line in enumerate(visible):
        y = rect.height - total_h + CONSOLE_MARGIN + i * CONSOLE_LINE_HEIGHT
        rl.draw_text_ex(gui_app.font(), line, rl.Vector2(CONSOLE_MARGIN, y), CONSOLE_FONT_SIZE, 0.0, text_color)


def _read_input(f):
  """Non-blocking read of available lines from f (stdin or FIFO)."""
  lines = []
  while True:
    rlist, _, _ = select.select([f], [], [], 0.0)
    if not rlist:
      break
    try:
      line = f.readline()
    except (BlockingIOError, OSError):
      break
    if line == "":
      break
    line = line.strip()
    if line:
      lines.append(line)
  return lines


def main():
  gui_app.init_window("Spinner")
  spinner = Spinner()

  # Use FIFO if given as argument (boot console mode), otherwise fall back to stdin
  if len(sys.argv) > 1:
    fifo_path = sys.argv[1]
    # O_RDWR: keeps write end open so reader never sees EOF between writers (keepalive trick)
    raw_fd = os.open(fifo_path, os.O_RDWR)
    input_file = os.fdopen(raw_fd, 'r')
  else:
    input_file = sys.stdin

  for _ in gui_app.render():
    for text in _read_input(input_file):
      spinner.set_text(text)

    spinner.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))


if __name__ == "__main__":
  main()
