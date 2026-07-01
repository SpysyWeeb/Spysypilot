"""
Custom drag-to-seek playback position bar for the Clip Viewer.

system/ui/widgets/slider.py's SliderBase/LargerSlider/BigSlider are swipe-*to-confirm* controls
(e.g. "swipe to power off"), not a scrubber, so they are not reused here -- only the general drag
tracking technique (via _handle_mouse_event, following mouse_event.pos across frames) is borrowed
from there. Frame-accurate scrubbing is not required; a few-seconds seek granularity is fine.
"""
from collections.abc import Callable

import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight, MouseEvent
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

TRACK_COLOR = rl.Color(70, 70, 70, 255)
FILL_COLOR = rl.Color(70, 91, 234, 255)
KNOB_COLOR = rl.WHITE
KNOB_RADIUS = 14
TRACK_HEIGHT = 8
LABEL_FONT_SIZE = 32
LABEL_ROW_HEIGHT = LABEL_FONT_SIZE + 10


def _fmt_time(seconds: float) -> str:
  seconds = max(0, int(seconds))
  m, s = divmod(seconds, 60)
  h, m = divmod(m, 60)
  return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


class ClipSeekBar(Widget):
  def __init__(self):
    super().__init__()
    self._progress = 0.0  # 0..1, currently displayed position (live playback or active drag)
    self._current_s = 0.0
    self._total_s = 0.0
    self._dragging = False
    self._seek_callback: Callable[[float], None] | None = None  # called with a 0..1 fraction

  def set_seek_callback(self, cb: Callable[[float], None]) -> None:
    self._seek_callback = cb

  def set_progress(self, progress: float, current_s: float, total_s: float) -> None:
    """Called every frame by the owning screen with the player's current state. Ignored while
    the user is actively dragging so the drag position isn't fought over."""
    self._current_s = current_s
    self._total_s = total_s
    if not self._dragging:
      self._progress = max(0.0, min(progress, 1.0))

  def _track_rect(self, rect: rl.Rectangle) -> rl.Rectangle:
    pad = KNOB_RADIUS
    return rl.Rectangle(rect.x + pad, rect.y + rect.height / 2 - TRACK_HEIGHT / 2, rect.width - 2 * pad, TRACK_HEIGHT)

  def _progress_from_x(self, track: rl.Rectangle, x: float) -> float:
    if track.width <= 0:
      return 0.0
    return max(0.0, min((x - track.x) / track.width, 1.0))

  def _handle_mouse_event(self, mouse_event: MouseEvent) -> None:
    """Only commit a seek on release, not on press or on every intermediate drag sample.
    CONFIRMED ON-DEVICE (live SSH session, 2026-07-01): each seek commit tears down and recreates
    the entire VisionIpcServer/FrameQueue/GL textures (ClipPlayer._reload -> unconditional
    _teardown_frame_feed + recreate), and mouse events can arrive in bursts of a dozen-plus for
    what was physically a single touch (observed on the back button: ~18 duplicate release events
    in ~20ms when the render loop is already stalling) -- committing on every press+move+release
    sample turned one drag gesture into a storm of full server/texture rebuilds, which was the
    dominant cause of sustained stutter (FPS logged as low as 0, sustained 3-7 for a minute-plus),
    not a one-time hiccup. Dragging still updates the visually-displayed position every sample
    (cheap, local-only), just without touching the player until the finger lifts."""
    super()._handle_mouse_event(mouse_event)
    track = self._track_rect(self._slider_rect())

    if mouse_event.left_pressed:
      self._dragging = True
      self._progress = self._progress_from_x(track, mouse_event.pos.x)
    elif mouse_event.left_released:
      if self._dragging and self._seek_callback:
        self._seek_callback(self._progress)
      self._dragging = False
    elif self._dragging:
      self._progress = self._progress_from_x(track, mouse_event.pos.x)

  def _slider_rect(self) -> rl.Rectangle:
    return rl.Rectangle(self._rect.x, self._rect.y + LABEL_ROW_HEIGHT, self._rect.width, self._rect.height - LABEL_ROW_HEIGHT)

  def _render(self, rect: rl.Rectangle) -> None:
    font = gui_app.font(FontWeight.NORMAL)
    current_txt = _fmt_time(self._current_s)
    total_txt = _fmt_time(self._total_s)
    rl.draw_text_ex(font, current_txt, rl.Vector2(rect.x, rect.y), LABEL_FONT_SIZE, 0, rl.WHITE)
    total_size = measure_text_cached(font, total_txt, LABEL_FONT_SIZE)
    rl.draw_text_ex(font, total_txt, rl.Vector2(rect.x + rect.width - total_size.x, rect.y), LABEL_FONT_SIZE, 0, rl.WHITE)

    track = self._track_rect(self._slider_rect())
    rl.draw_rectangle_rounded(track, 1.0, 10, TRACK_COLOR)

    fill_w = track.width * self._progress
    if fill_w > 0:
      rl.draw_rectangle_rounded(rl.Rectangle(track.x, track.y, fill_w, track.height), 1.0, 10, FILL_COLOR)

    knob_x = track.x + fill_w
    knob_y = track.y + track.height / 2
    rl.draw_circle(int(knob_x), int(knob_y), KNOB_RADIUS, KNOB_COLOR)
