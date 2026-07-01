"""
ClipViewerButton: home screen right-hand side-column entry point for the Clip Viewer.

This fork's other right-column buttons (error_log_button.py / screen_timeout_button.py) don't
exist on this branch (see clip-viewer report for why -- clip-route-viewer is cut from stock, not
combo), so this follows their established shape from direct inspection rather than an import:
a plain Widget with a colored rounded-rect background/border and a left-aligned ~45px label,
`alpha = 0xCC if self.is_pressed else 0xFF` press feedback, sized/stacked by home.py the same way.
"""
import pyray as rl

from openpilot.selfdrive.ui.widgets.clip_viewer_screen import ClipViewerScreen, entry_allowed
from openpilot.system.ui.lib.application import gui_app, FontWeight, FONT_SCALE
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget


class ClipViewerButton(Widget):
  def __init__(self):
    super().__init__()
    # Constructed lazily (on first tap) rather than here: AugmentedRoadView() (owned by
    # ClipViewerScreen -> ClipPlayer) does real EGL/shader/VisionIpcClient setup in its
    # constructor, not worth paying for at app startup for a screen most drives will never open.
    # Cached (not rebuilt per tap) because CameraView.__init__ permanently registers an
    # ui_state.add_offroad_transition_callback with no matching remove -- constructing a fresh
    # AugmentedRoadView on every visit would leak one such callback per visit.
    self._screen: ClipViewerScreen | None = None
    self.set_enabled(entry_allowed)

  def _handle_mouse_release(self, mouse_pos) -> None:
    super()._handle_mouse_release(mouse_pos)
    if not entry_allowed():
      # Framework already prevents this callback from firing while disabled (Widget.render()
      # only calls _process_mouse_events() when self.enabled); this is just defense in depth
      # against entry_allowed() flipping between the enabled-check and the tap landing.
      return
    if self._screen is None:
      self._screen = ClipViewerScreen()
    gui_app.push_widget(self._screen)

  def _render(self, rect: rl.Rectangle) -> None:
    enabled = self.enabled
    alpha = 0xCC if self.is_pressed else 0xFF
    bg_alpha = alpha if enabled else 0x55

    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), int(rect.height), rl.Color(51, 51, 51, bg_alpha))
    rl.draw_rectangle_rounded_lines_ex(self._rect, 0.19, 10, 5, rl.BLACK)
    rl.end_scissor_mode()

    text = tr("CLIP VIEWER")
    text_color = rl.WHITE if enabled else rl.Color(255, 255, 255, 120)
    text_y = rect.y + rect.height / 2 - 45 * FONT_SCALE // 2
    rl.draw_text_ex(gui_app.font(FontWeight.NORMAL), text, rl.Vector2(int(rect.x + 25), int(text_y)), 45, 0, text_color)
