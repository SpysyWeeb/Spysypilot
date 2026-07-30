"""Offroad-only BLaTv2 profile feedback modal shared by tici and mici."""

from __future__ import annotations

from collections.abc import Callable

import pyray as rl

from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.blatv2.feedback import (
  FeedbackChoice,
  FeedbackPromptState,
  FeedbackRequest,
)
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.system.ui.widgets.label import Label


QUESTION = "Compared with the previous steering profile, how did steering feel?"
_BUTTONS = (
  ("Better", FeedbackChoice.BETTER),
  ("About same", FeedbackChoice.ABOUT_SAME),
  ("Worse", FeedbackChoice.WORSE),
  ("Not sure", FeedbackChoice.NOT_SURE),
)


class BlatV2FeedbackDialog(Widget):
  """A non-dismissible four-choice modal; no choice is inferred or defaulted."""

  def __init__(
    self,
    request: FeedbackRequest,
    callback: Callable[[FeedbackRequest, FeedbackChoice], None],
  ):
    super().__init__()
    self.request = request
    self._label = self._child(Label(
      QUESTION,
      font_size=64 if gui_app.big_ui() else 27,
      font_weight=FontWeight.BOLD,
    ))
    self._buttons = tuple(
      self._child(Button(
        text,
        click_callback=lambda selected=choice: callback(request, selected),
        font_size=52 if gui_app.big_ui() else 24,
        font_weight=FontWeight.MEDIUM,
        button_style=ButtonStyle.PRIMARY if choice == FeedbackChoice.BETTER else ButtonStyle.NORMAL,
      ))
      for text, choice in _BUTTONS
    )

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, rl.Color(0, 0, 0, 235))

    big = gui_app.big_ui()
    outer_x = 190 if big else 12
    outer_y = 120 if big else 8
    card = rl.Rectangle(
      rect.x + outer_x,
      rect.y + outer_y,
      rect.width - 2 * outer_x,
      rect.height - 2 * outer_y,
    )
    rl.draw_rectangle_rounded(card, 0.04 if big else 0.08, 16, rl.Color(27, 27, 27, 255))

    pad = 60 if big else 12
    gap = 28 if big else 8
    title_height = 260 if big else 72
    self._label.render(rl.Rectangle(
      card.x + pad,
      card.y + pad,
      card.width - 2 * pad,
      title_height,
    ))

    grid_y = card.y + pad + title_height + gap
    grid_height = card.y + card.height - pad - grid_y
    button_width = (card.width - 2 * pad - gap) / 2
    button_height = (grid_height - gap) / 2
    for index, button in enumerate(self._buttons):
      row, column = divmod(index, 2)
      button.render(rl.Rectangle(
        card.x + pad + column * (button_width + gap),
        grid_y + row * (button_height + gap),
        button_width,
        button_height,
      ))


class BlatV2FeedbackPrompt:
  """Rendering adapter around the pure, profile-bound prompt state."""

  def __init__(
    self,
    root_widget: Widget,
    is_offroad: Callable[[], bool],
    params: Params | None = None,
  ):
    self._root_widget = root_widget
    self._is_offroad = is_offroad
    self._params = params if params is not None else Params()
    self._state = FeedbackPromptState()
    self._dialog: BlatV2FeedbackDialog | None = None

  def update(self) -> None:
    offroad = self._is_offroad()
    if not offroad:
      self._state.update(self._params, offroad=False)
      self._dismiss_for_onroad()
      return

    if self._dialog is not None:
      self._state.update(self._params, offroad=True)
      if self._state.presented_request is None:
        self._dismiss_active_dialog()
      elif self._state.presented_request != self._dialog.request:
        replacement = self._state.presented_request
        if self._dismiss_active_dialog():
          self._show_dialog(replacement)
      return

    # Never cover onboarding or another generic UI dialog. The normal render
    # loop retries after that widget closes.
    if self._dialog is None and gui_app.get_active_widget() is self._root_widget:
      request = self._state.update(self._params, offroad=True)
      if request is not None:
        self._show_dialog(request)

  def _show_dialog(self, request: FeedbackRequest) -> None:
    if gui_app.get_active_widget() is not self._root_widget:
      return
    self._dialog = BlatV2FeedbackDialog(request, self._on_choice)
    gui_app.push_widget(self._dialog)

  def _on_choice(
    self,
    request: FeedbackRequest,
    choice: FeedbackChoice,
  ) -> None:
    # The callback is bound to the displayed request. Recheck both the offroad
    # state and pending Params request before writing.
    if (
      self._is_offroad()
      and self._state.presented_request == request
      and self._state.submit(self._params, choice, offroad=True)
    ):
      self._dismiss_active_dialog()

  def _dismiss_active_dialog(self) -> bool:
    if self._dialog is not None and gui_app.get_active_widget() is self._dialog:
      gui_app.pop_widget()
    elif self._dialog is not None and gui_app.widget_in_stack(self._dialog):
      return False
    self._dialog = None
    return True

  def _dismiss_for_onroad(self) -> None:
    if self._dialog is None:
      return
    if gui_app.get_active_widget() is self._dialog:
      gui_app.pop_widget()
    elif gui_app.widget_in_stack(self._dialog):
      # An onroad transition already dismisses generic UI state. If another
      # modal covered this prompt, remove the complete modal stack immediately
      # so this offroad-only prompt can never become visible onroad.
      gui_app.pop_widgets_to(self._root_widget, instant=True)
    self._dialog = None
