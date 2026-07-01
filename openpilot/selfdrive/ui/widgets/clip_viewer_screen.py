"""
ClipViewerScreen: full-screen route browser + synchronized onroad-UI playback preview.

Pushed onto the nav stack by ClipViewerButton (selfdrive/ui/widgets/clip_viewer_button.py).
Two view modes, toggled by tapping the preview -- never via push_widget/pop_widget, so toggling
never touches the ui_state.sm patch (see clip_playback.py's module docstring):
 - list view: back button, scrollable route list, small top-right preview + seek bar below it.
 - fullscreen: preview fills the screen, seek bar pinned near the bottom.
"""
import pyray as rl

from openpilot.common.params import Params
from openpilot.selfdrive.spysypilot.clip_playback import ClipPlayer
from openpilot.selfdrive.spysypilot.clip_routes import RouteSummary, list_route_summaries
from openpilot.selfdrive.ui.onroad.augmented_road_view import AugmentedRoadView
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.widgets.clip_seek_bar import ClipSeekBar
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.widgets.list_view import button_item
from openpilot.system.ui.widgets.scroller_tici import Scroller

MARGIN = 40
PREVIEW_W = 900
PREVIEW_H = 506
SEEK_BAR_HEIGHT = 90
BACK_BTN_W = 200
BACK_BTN_H = 90
FULLSCREEN_SEEK_MARGIN = 40


def entry_allowed() -> bool:
  """Refuse to allow Clip Viewer whenever a real camerad process might be running or about to
  start -- see selfdrive/spysypilot/clip_playback.py's module docstring for the full reasoning.
  Guards:
   - ui_state.started: already onroad, the real camerad is running.
   - IsDriverViewEnabled / IsLiveStreaming: these also start the real camerad
     (system/manager/process_config.py's driverview()/livestream() gates on the "camerad" entry),
     so creating a second VisionIpcServer("camerad") for playback would be an untested conflict.
  Checked both when greying out the entry button and again on screen show (defense in depth).
  """
  if ui_state.started:
    return False
  params = Params()
  if params.get_bool("IsDriverViewEnabled") or params.get_bool("IsLiveStreaming"):
    return False
  return True


class ClipViewerScreen(Widget):
  def __init__(self):
    super().__init__()
    self._player = ClipPlayer()
    self._routes: list[RouteSummary] = []
    self._selected_route: RouteSummary | None = None
    self._fullscreen = False
    self._guard_failed = False

    self._road_view = self._child(AugmentedRoadView())
    self._road_view.set_click_callback(self._toggle_fullscreen)

    self._seek_bar = self._child(ClipSeekBar())
    self._seek_bar.set_seek_callback(self._on_seek)

    self._back_button = self._child(
      Button(tr("BACK"), click_callback=self._on_back, button_style=ButtonStyle.TRANSPARENT_WHITE_BORDER)
    )

    self._scroller = self._child(Scroller([], line_separator=True))

  # -- nav stack lifecycle ------------------------------------------------------------------
  def show_event(self):
    self._fullscreen = False
    self._guard_failed = not entry_allowed()
    if not self._guard_failed:
      self._refresh_routes()
    super().show_event()

  def hide_event(self):
    super().hide_event()
    self._player.close()
    self._selected_route = None
    self._fullscreen = False

  # -- route list ------------------------------------------------------------------
  def _refresh_routes(self):
    self._routes = list_route_summaries()

    # Scroller (system/ui/widgets/scroller_tici.py) only exposes add_widget(), no way to replace
    # its item list wholesale -- reach into _items directly to rebuild it. Rebuilding (rather
    # than keeping one Scroller/list forever) lets newly-recorded routes show up on a later visit
    # to this screen within the same app session.
    self._scroller._items = []
    for route in self._routes:
      label = f"{route.date_str}   ({route.num_segments} seg, ~{route.approx_duration_s // 60} min)"
      item = button_item(label, lambda: tr("VIEW"), callback=lambda r=route: self._select_route(r))
      self._scroller.add_widget(item)

  def _select_route(self, route: RouteSummary):
    self._selected_route = route
    self._player.load_route(route)
    self._player.play()

  # -- controls ------------------------------------------------------------------
  def _on_seek(self, fraction: float):
    self._player.seek_fraction(fraction)

  def _toggle_fullscreen(self):
    if self._selected_route is None:
      return  # nothing loaded yet, no point going fullscreen on a blank preview
    self._fullscreen = not self._fullscreen

  def _on_back(self):
    gui_app.pop_widget()

  # -- per-frame update ------------------------------------------------------------------
  def _update_state(self):
    if self._guard_failed:
      gui_app.pop_widget()
      return

    still_ok = self._player.tick()
    if not still_ok:
      # Ignition watchdog tripped: car is actually being driven. Bail to Home immediately so the
      # app falls through to the real onroad transition.
      self._fullscreen = False
      self._selected_route = None
      gui_app.pop_widget()
      return

    if self._player.is_loaded:
      self._seek_bar.set_progress(self._player.progress, self._player.current_time_s, self._player.total_time_s)

  # -- rendering ------------------------------------------------------------------
  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, rl.BLACK)
    if self._fullscreen:
      self._render_fullscreen(rect)
    else:
      self._render_list(rect)

  def _render_list(self, rect: rl.Rectangle):
    back_rect = rl.Rectangle(rect.x + MARGIN, rect.y + MARGIN, BACK_BTN_W, BACK_BTN_H)
    self._back_button.render(back_rect)

    preview_rect = rl.Rectangle(rect.x + rect.width - PREVIEW_W - MARGIN, rect.y + MARGIN, PREVIEW_W, PREVIEW_H)
    self._render_preview(preview_rect)

    seek_rect = rl.Rectangle(preview_rect.x, preview_rect.y + PREVIEW_H + 15, PREVIEW_W, SEEK_BAR_HEIGHT)
    self._seek_bar.render(seek_rect)

    list_top = back_rect.y + back_rect.height + 20
    list_rect = rl.Rectangle(
      rect.x + MARGIN,
      list_top,
      rect.width - PREVIEW_W - MARGIN * 3,
      rect.y + rect.height - MARGIN - list_top,
    )
    self._scroller.render(list_rect)

  def _render_fullscreen(self, rect: rl.Rectangle):
    self._render_preview(rect)
    seek_rect = rl.Rectangle(
      rect.x + FULLSCREEN_SEEK_MARGIN,
      rect.y + rect.height - SEEK_BAR_HEIGHT - FULLSCREEN_SEEK_MARGIN,
      rect.width - 2 * FULLSCREEN_SEEK_MARGIN,
      SEEK_BAR_HEIGHT,
    )
    self._seek_bar.render(seek_rect)

  def _render_preview(self, rect: rl.Rectangle):
    self._road_view.render(rect)
    # AugmentedRoadView._render() no-ops entirely while ui_state.started is False, which is the
    # real (unpatched) value until a route is loaded -- draw a placeholder so the preview area
    # isn't just blank before the user picks something.
    if self._selected_route is None:
      rl.draw_rectangle_rec(rect, rl.Color(20, 20, 20, 255))
      gui_label(rect, tr("Select a route to preview"), font_size=28, color=rl.Color(180, 180, 180, 255),
                alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER, alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE)
