"""Two-page, display-only BLaTv2 learning dashboard."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

import pyray as rl

from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.widgets.blatv2_learning_status import (
  GridCell,
  LearningOperationStatus,
  LearningNodeStatus,
  LearningStatus,
  LearningStatusError,
  LifecycleStatus,
  format_duration,
  format_speed,
  grid_cells,
  learning_panel_presentation,
  parse_learning_operation_status,
  parse_learning_status,
  parse_lifecycle_status,
  reason_label,
  select_value_provider,
  validate_operation_update,
)
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget


REFRESH_INTERVAL_S = 2.0

_BG = rl.Color(40, 40, 40, 255)
_PANEL = rl.Color(53, 53, 53, 255)
_TRACK = rl.Color(255, 255, 255, 32)
_DIVIDER = rl.Color(255, 255, 255, 28)
_WHITE = rl.Color(255, 255, 255, 255)
_DIM = rl.Color(255, 255, 255, 155)
_GRAY = rl.Color(155, 155, 155, 255)
_BLUE = rl.Color(82, 132, 255, 255)
_AMBER = rl.Color(234, 160, 50, 255)
_GREEN = rl.Color(70, 200, 100, 255)
_RED = rl.Color(225, 70, 70, 255)


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
  learning: LearningStatus | None
  learning_error_code: str | None
  learning_error: str | None
  operation: LearningOperationStatus | None
  operation_error_code: str | None
  operation_error: str | None
  lifecycle: LifecycleStatus | None
  lifecycle_error_code: str | None
  lifecycle_error: str | None
  metric: bool


class BLaTv2LearningStatusSource:
  """Rate-limited Params reader shared by both pages.

  JSON decoding is handled by Params according to the registered JSON type.
  The UI never opens evidence, manifests, profiles, or route logs.
  """

  def __init__(
    self,
    params: Params | None = None,
    *,
    vehicle_identity_provider: Callable[[], str | None] | None = None,
    metric_provider: Callable[[], bool] | None = None,
  ):
    self._params = Params() if params is None else params
    self._vehicle_identity_provider = (
      self._default_vehicle_identity
      if vehicle_identity_provider is None
      else vehicle_identity_provider
    )
    self._metric_provider = select_value_provider(
      lambda: bool(ui_state.is_metric),
      metric_provider,
    )
    self._last_refresh = float("-inf")
    self._snapshot = DashboardSnapshot(
      learning=None,
      learning_error_code="absent",
      learning_error="Learning data is not available yet",
      operation=None,
      operation_error_code="operation_absent",
      operation_error="Learner operation status has not been published",
      lifecycle=None,
      lifecycle_error_code="activation_absent",
      lifecycle_error="Controller status is not available yet",
      metric=False,
    )

  @staticmethod
  def _default_vehicle_identity() -> str | None:
    try:
      identity = str(ui_state.CP.carFingerprint).strip()
    except Exception:
      return None
    return identity or None

  @property
  def snapshot(self) -> DashboardSnapshot:
    now = time.monotonic()
    if now - self._last_refresh >= REFRESH_INTERVAL_S:
      self._refresh()
      self._last_refresh = now
    return self._snapshot

  def _read_param(self, key: str) -> object:
    try:
      return self._params.get(key, block=False)
    except Exception as exc:
      raise LearningStatusError(
        "param_read_error",
        f"{key} could not be read",
      ) from exc

  def _refresh(self) -> None:
    try:
      expected_vehicle = self._vehicle_identity_provider()
    except Exception:
      expected_vehicle = None

    learning: LearningStatus | None = None
    learning_error_code: str | None = None
    learning_error: str | None = None
    try:
      learning = parse_learning_status(
        self._read_param("BLaTv2LearningStatus"),
        expected_vehicle_identity=expected_vehicle,
      )
    except LearningStatusError as exc:
      learning_error_code = exc.code
      learning_error = str(exc)
    except Exception:
      learning_error_code = "malformed"
      learning_error = "Learning snapshot could not be read"

    operation: LearningOperationStatus | None = None
    operation_error_code: str | None = None
    operation_error: str | None = None
    try:
      operation = parse_learning_operation_status(
        self._read_param("BLaTv2LearningOperationStatus"),
        expected_vehicle_identity=expected_vehicle,
        expected_runtime_identity_sha256=(
          learning.runtime_identity_sha256
          if learning is not None
          else (
            self._snapshot.learning.runtime_identity_sha256
            if self._snapshot.learning is not None
            else None
          )
        ),
        now_mono_ns=time.monotonic_ns(),
      )
      validate_operation_update(self._snapshot.operation, operation)
    except LearningStatusError as exc:
      operation_error_code = exc.code
      operation_error = str(exc)
    except Exception:
      operation_error_code = "malformed"
      operation_error = "Learner operation status could not be read"

    # Params snapshots are atomic individually, not across keys. If an active
    # operation is observed between its status writes, preserve the most
    # recent validated in-memory learning snapshot rather than blanking the
    # dashboard. Runtime and vehicle identity must still match.
    prior_learning = self._snapshot.learning
    if (
      learning is None
      and learning_error_code == "absent"
      and operation is not None
      and operation.active
      and prior_learning is not None
      and operation.vehicle_identity == prior_learning.vehicle_identity
      and operation.runtime_identity_sha256
      == prior_learning.runtime_identity_sha256
    ):
      learning = prior_learning

    lifecycle: LifecycleStatus | None = None
    lifecycle_error_code: str | None = None
    lifecycle_error: str | None = None
    try:
      lifecycle = parse_lifecycle_status(
        self._read_param("BLaTv2LifecycleStatus"),
        expected_vehicle_identity=expected_vehicle,
        expected_runtime_identity_sha256=(
          None if learning is None else learning.runtime_identity_sha256
        ),
      )
    except LearningStatusError as exc:
      lifecycle_error_code = exc.code
      lifecycle_error = str(exc)
    except Exception:
      lifecycle_error_code = "malformed"
      lifecycle_error = "Controller status could not be read"

    try:
      metric = bool(self._metric_provider())
    except Exception:
      metric = False

    self._snapshot = DashboardSnapshot(
      learning=learning,
      learning_error_code=learning_error_code,
      learning_error=learning_error,
      operation=operation,
      operation_error_code=operation_error_code,
      operation_error=operation_error,
      lifecycle=lifecycle,
      lifecycle_error_code=lifecycle_error_code,
      lifecycle_error=lifecycle_error,
      metric=metric,
    )


class _BLaTv2Page(Widget):
  def __init__(self, source: BLaTv2LearningStatusSource):
    super().__init__()
    self._source = source
    self._background_tap_callback = None

  def set_background_tap_callback(self, callback) -> None:
    self._background_tap_callback = callback

  def _handle_mouse_release(self, mouse_pos) -> None:
    if self._background_tap_callback:
      self._background_tap_callback(mouse_pos)

  @staticmethod
  def _tone_for_node(node: LearningNodeStatus) -> rl.Color:
    if node.qualified:
      return _GREEN
    if node.collection_complete and node.primary_reason in (
      "invalid_parameters",
      "validation_regression",
      "singular_fit",
    ):
      return _RED
    if node.clean_support_s <= 0.0:
      return _GRAY
    if node.support_fraction >= 1.0:
      return _AMBER
    return _BLUE

  @staticmethod
  def _error_color(code: str | None) -> rl.Color:
    return (
      _RED
      if code
      not in (
        None,
        "absent",
        "activation_absent",
        "operation_absent",
        "vehicle_unavailable",
      )
      else _GRAY
    )

  @staticmethod
  def _tone_color(tone: str) -> rl.Color:
    return {
      "gray": _GRAY,
      "blue": _BLUE,
      "amber": _AMBER,
      "green": _GREEN,
      "red": _RED,
    }.get(tone, _RED)

  @staticmethod
  def _lifecycle_color(lifecycle: LifecycleStatus | None) -> rl.Color:
    if lifecycle is None or lifecycle.controller_state in ("stock", "staged"):
      return _GRAY
    if lifecycle.controller_state == "approved":
      return _GREEN
    if lifecycle.controller_state == "provisional":
      return _AMBER
    if lifecycle.controller_state == "rollback_pending":
      return _RED
    if lifecycle.controller_state == "unavailable" and lifecycle.diagnostic in (
      "malformed",
      "profile_hash_mismatch",
      "policy_hash_mismatch",
      "state_invalid",
    ):
      return _RED
    return _GRAY

  def _draw_background(self, rect: rl.Rectangle) -> None:
    rl.draw_rectangle_rounded(rect, 0.025, 10, _BG)

  def _draw_header(
    self,
    rect: rl.Rectangle,
    title: str,
    snapshot: DashboardSnapshot,
  ) -> float:
    font = gui_app.font(FontWeight.BOLD)
    x = int(rect.x + 32)
    y = int(rect.y + 24)
    rl.draw_text_ex(font, title, rl.Vector2(x, y), 31, 0, _BLUE)

    badge = (
      snapshot.lifecycle.badge
      if snapshot.lifecycle is not None
      else "CONTROLLER STATUS UNAVAILABLE"
    )
    badge_color = (
      self._lifecycle_color(snapshot.lifecycle)
      if snapshot.lifecycle is not None
      else self._error_color(snapshot.lifecycle_error_code)
    )
    badge_font = gui_app.font(FontWeight.MEDIUM)
    text_size = measure_text_cached(badge_font, badge, 22)
    badge_width = text_size.x + 28
    badge_rect = rl.Rectangle(
      rect.x + rect.width - 32 - badge_width,
      rect.y + 18,
      badge_width,
      42,
    )
    rl.draw_rectangle_rounded(badge_rect, 0.35, 8, rl.fade(badge_color, 0.24))
    rl.draw_rectangle_rounded_lines_ex(
      badge_rect,
      0.35,
      8,
      1,
      badge_color,
    )
    rl.draw_text_ex(
      badge_font,
      badge,
      rl.Vector2(int(badge_rect.x + 14), int(badge_rect.y + 8)),
      22,
      0,
      badge_color,
    )
    return rect.y + 72

  def _draw_operation_banner(
    self,
    rect: rl.Rectangle,
    y: float,
    snapshot: DashboardSnapshot,
  ) -> float:
    presentation = learning_panel_presentation(
      snapshot.operation,
      operation_error_code=snapshot.operation_error_code,
      operation_error_message=snapshot.operation_error,
      learning_error_code=snapshot.learning_error_code,
      learning_error_message=snapshot.learning_error,
      has_learning_snapshot=snapshot.learning is not None,
    )
    if not presentation.show_banner:
      return y

    color = self._tone_color(presentation.tone)
    banner = rl.Rectangle(rect.x + 32, y, rect.width - 64, 58)
    rl.draw_rectangle_rounded(banner, 0.15, 8, rl.fade(color, 0.16))
    rl.draw_rectangle_rounded_lines_ex(banner, 0.15, 8, 1, color)
    medium = gui_app.font(FontWeight.MEDIUM)
    normal = gui_app.font(FontWeight.NORMAL)
    rl.draw_text_ex(
      medium,
      presentation.title,
      rl.Vector2(int(banner.x + 16), int(banner.y + 7)),
      18,
      0,
      color,
    )
    detail_font_size = 18
    detail_width = measure_text_cached(
      normal,
      presentation.detail,
      detail_font_size,
    ).x
    if detail_width > banner.width - 32:
      detail_font_size = max(
        14,
        int(detail_font_size * (banner.width - 32) / detail_width),
      )
    rl.draw_text_ex(
      normal,
      presentation.detail,
      rl.Vector2(int(banner.x + 16), int(banner.y + 32)),
      detail_font_size,
      0,
      _DIM,
    )
    return y + 68

  def _draw_unavailable(
    self,
    rect: rl.Rectangle,
    snapshot: DashboardSnapshot,
  ) -> None:
    font = gui_app.font(FontWeight.NORMAL)
    bold = gui_app.font(FontWeight.BOLD)
    presentation = learning_panel_presentation(
      snapshot.operation,
      operation_error_code=snapshot.operation_error_code,
      operation_error_message=snapshot.operation_error,
      learning_error_code=snapshot.learning_error_code,
      learning_error_message=snapshot.learning_error,
      has_learning_snapshot=False,
    )
    color = self._tone_color(presentation.tone)
    center_y = int(rect.y + rect.height * 0.46)
    title = presentation.title
    title_size = measure_text_cached(bold, title, 36)
    rl.draw_text_ex(
      bold,
      title,
      rl.Vector2(
        int(rect.x + (rect.width - title_size.x) / 2),
        center_y,
      ),
      36,
      0,
      color,
    )
    detail = presentation.detail
    detail_font_size = 27
    detail_size = measure_text_cached(font, detail, detail_font_size)
    if detail_size.x > rect.width - 80:
      detail_font_size = max(
        18,
        int(detail_font_size * (rect.width - 80) / detail_size.x),
      )
      detail_size = measure_text_cached(font, detail, detail_font_size)
    rl.draw_text_ex(
      font,
      detail,
      rl.Vector2(
        int(rect.x + (rect.width - detail_size.x) / 2),
        center_y + 54,
      ),
      detail_font_size,
      0,
      _DIM,
    )


class BLaTv2LearningOverviewWidget(_BLaTv2Page):
  """Six-node overview; node count and speeds come from the snapshot."""

  def _render(self, rect: rl.Rectangle) -> None:
    snapshot = self._source.snapshot
    self._draw_background(rect)
    content_y = self._draw_header(rect, "BLATV2 LEARNING", snapshot)
    learning = snapshot.learning
    if learning is None:
      self._draw_unavailable(rect, snapshot)
      return
    content_y = self._draw_operation_banner(rect, content_y, snapshot)

    normal = gui_app.font(FontWeight.NORMAL)
    bold = gui_app.font(FontWeight.BOLD)
    pad = 32
    x = rect.x + pad
    width = rect.width - pad * 2
    summary = f"{learning.qualified_node_count} OF {len(learning.nodes)} SPEED NODES QUALIFIED"
    rl.draw_text_ex(
      bold,
      summary,
      rl.Vector2(int(x), int(content_y)),
      25,
      0,
      _WHITE,
    )
    summary_detail = (
      "Complete physical fit; activation is tracked separately"
      if learning.all_nodes_qualified
      else "Clean support is credited between neighboring speed nodes"
    )
    rl.draw_text_ex(
      normal,
      summary_detail,
      rl.Vector2(int(x), int(content_y + 33)),
      22,
      0,
      _DIM,
    )

    grid_y = content_y + 72
    footer_height = 48
    grid_height = rect.y + rect.height - footer_height - grid_y
    # Six current nodes use the intended 2x3 layout. A future portable
    # profile with more nodes gets a compact three-column grid rather than
    # overflowing or silently dropping evidence.
    columns = 2 if len(learning.nodes) <= 6 else 3
    cells = grid_cells(
      width,
      grid_height,
      len(learning.nodes),
      columns=columns,
    )
    for cell, node in zip(cells, learning.nodes, strict=True):
      self._draw_node_card(
        GridCell(
          x=x + cell.x,
          y=grid_y + cell.y,
          width=cell.width,
          height=cell.height,
        ),
        node,
        snapshot.metric,
        learning.last_drive_complete,
      )

    footer = "Nodes blend continuously. Full time can still need motion, validation, or a valid fit."
    rl.draw_text_ex(
      normal,
      footer,
      rl.Vector2(int(x), int(rect.y + rect.height - 34)),
      21,
      0,
      _DIM,
    )

  def _draw_node_card(
    self,
    cell: GridCell,
    node: LearningNodeStatus,
    metric: bool,
    last_drive_complete: bool,
  ) -> None:
    rect = rl.Rectangle(cell.x, cell.y, cell.width, cell.height)
    rl.draw_rectangle_rounded(rect, 0.08, 8, _PANEL)
    tone = self._tone_for_node(node)
    normal = gui_app.font(FontWeight.NORMAL)
    medium = gui_app.font(FontWeight.MEDIUM)
    bold = gui_app.font(FontWeight.BOLD)
    pad = 18
    x = rect.x + pad
    y = rect.y + 13
    compact = rect.width < 390 or rect.height < 130

    speed = format_speed(node.speed_mps, metric=metric)
    rl.draw_text_ex(
      bold,
      speed,
      rl.Vector2(int(x), int(y)),
      22 if compact else 28,
      0,
      _WHITE,
    )
    reason = reason_label(node.primary_reason).upper()
    reason_font_size = 14 if compact else 18
    reason_size = measure_text_cached(medium, reason, reason_font_size)
    reason_x = (
      x
      if compact
      else rect.x + rect.width - pad - reason_size.x
    )
    reason_y = y + (27 if compact else 5)
    rl.draw_text_ex(
      medium,
      reason,
      rl.Vector2(int(reason_x), int(reason_y)),
      reason_font_size,
      0,
      tone,
    )

    support = f"{format_duration(node.clean_support_s)} / {format_duration(node.minimum_support_s)} CLEAN"
    rl.draw_text_ex(
      medium,
      support,
      rl.Vector2(int(x), int(y + (48 if compact else 40))),
      16 if compact else 23,
      0,
      tone,
    )
    progress_rect = rl.Rectangle(
      x,
      y + (70 if compact else 72),
      rect.width - pad * 2,
      9 if compact else 12,
    )
    rl.draw_rectangle_rounded(progress_rect, 0.5, 6, _TRACK)
    fill_width = progress_rect.width * node.support_fraction
    if fill_width > 1.0:
      rl.draw_rectangle_rounded(
        rl.Rectangle(
          progress_rect.x,
          progress_rect.y,
          fill_width,
          progress_rect.height,
        ),
        0.5,
        6,
        tone,
      )

    if last_drive_complete and node.last_drive_clean_support_s is not None:
      drive_text = (
        f"+{format_duration(node.last_drive_clean_support_s)} LAST DRIVE"
      )
    else:
      drive_text = "LAST-DRIVE CONTRIBUTION UNAVAILABLE"
    drive_y = y + (84 if compact else 94)
    if drive_y + (14 if compact else 19) <= rect.y + rect.height - 4:
      rl.draw_text_ex(
        normal,
        drive_text,
        rl.Vector2(int(x), int(drive_y)),
        14 if compact else 19,
        0,
        _DIM,
      )


class BLaTv2ReadinessWidget(_BLaTv2Page):
  """Compact qualification matrix and independent activation lifecycle."""

  def _render(self, rect: rl.Rectangle) -> None:
    snapshot = self._source.snapshot
    self._draw_background(rect)
    content_y = self._draw_header(
      rect,
      "READINESS & ACTIVATION",
      snapshot,
    )
    learning = snapshot.learning
    if learning is None:
      self._draw_unavailable(rect, snapshot)
      return
    content_y = self._draw_operation_banner(rect, content_y, snapshot)

    pad = 32
    x = rect.x + pad
    width = rect.width - pad * 2
    matrix_bottom = rect.y + rect.height - 165
    matrix_height = matrix_bottom - content_y
    self._draw_matrix(
      x,
      content_y,
      width,
      matrix_height,
      learning,
      snapshot.metric,
    )
    self._draw_lifecycle(
      x,
      matrix_bottom + 18,
      width,
      learning,
      snapshot,
    )

  def _draw_matrix(
    self,
    x: float,
    y: float,
    width: float,
    height: float,
    learning: LearningStatus,
    metric: bool,
  ) -> None:
    normal = gui_app.font(FontWeight.NORMAL)
    medium = gui_app.font(FontWeight.MEDIUM)
    columns = (0.00, 0.18, 0.37, 0.58, 0.78)
    headers = ("NODE", "TIME", "VALIDATION", "MOTION", "FIT / STATE")
    for offset, header in zip(columns, headers, strict=True):
      rl.draw_text_ex(
        medium,
        header,
        rl.Vector2(int(x + width * offset), int(y)),
        21,
        0,
        _DIM,
      )
    line_y = y + 32
    rl.draw_line_ex(
      rl.Vector2(x, line_y),
      rl.Vector2(x + width, line_y),
      1,
      _DIVIDER,
    )

    row_height = (height - 34) / len(learning.nodes)
    row_font_size = max(13, min(21, int(row_height * 0.46)))
    for position, node in enumerate(learning.nodes):
      row_y = line_y + position * row_height
      if position:
        rl.draw_line_ex(
          rl.Vector2(x, row_y),
          rl.Vector2(x + width, row_y),
          1,
          _DIVIDER,
        )
      text_y = row_y + (row_height - row_font_size) / 2
      tone = self._tone_for_node(node)
      values = (
        format_speed(node.speed_mps, metric=metric),
        f"{node.support_fraction * 100:.0f}%",
        (
          "READY"
          if "insufficient_validation" not in node.reasons
          else f"{node.validation_fraction * 100:.0f}%"
        ),
        (
          "READY"
          if "insufficient_excitation" not in node.reasons
          else "MORE RANGE"
        ),
        self._fit_text(node),
      )
      value_colors = (_WHITE, tone, tone, tone, tone)
      for offset, value, color in zip(
        columns,
        values,
        value_colors,
        strict=True,
      ):
        rl.draw_text_ex(
          normal,
          value,
          rl.Vector2(int(x + width * offset), int(text_y)),
          row_font_size,
          0,
          color,
        )

  @staticmethod
  def _fit_text(node: LearningNodeStatus) -> str:
    if node.qualified:
      return "PASSED"
    if not node.collection_complete:
      return "WAITING"
    if "invalid_parameters" in node.reasons:
      return "REJECTED"
    if "validation_regression" in node.reasons:
      return "REGRESSED"
    if "singular_fit" in node.reasons:
      return "UNSTABLE"
    return "WAITING"

  def _draw_lifecycle(
    self,
    x: float,
    y: float,
    width: float,
    learning: LearningStatus,
    snapshot: DashboardSnapshot,
  ) -> None:
    normal = gui_app.font(FontWeight.NORMAL)
    medium = gui_app.font(FontWeight.MEDIUM)
    labels = ("COLLECT", "PROFILE", "GATES", "PROVISIONAL", "APPROVED")
    stage = (
      (1 if learning.all_nodes_qualified else 0)
      if snapshot.lifecycle is None
      else max(
        snapshot.lifecycle.lifecycle_position,
        1 if learning.all_nodes_qualified else 0,
      )
    )
    rail_left = x + 24
    rail_right = x + width - 24
    rail_y = y + 14
    spacing = (rail_right - rail_left) / (len(labels) - 1)
    rl.draw_line_ex(
      rl.Vector2(rail_left, rail_y),
      rl.Vector2(rail_right, rail_y),
      3,
      _TRACK,
    )
    for index, label in enumerate(labels):
      px = rail_left + index * spacing
      color = _GREEN if stage >= index else _GRAY
      if (
        snapshot.lifecycle is not None
        and snapshot.lifecycle.controller_state == "rollback_pending"
        and index > 0
      ):
        color = _RED
      rl.draw_circle(int(px), int(rail_y), 8, color)
      size = measure_text_cached(medium, label, 17)
      rl.draw_text_ex(
        medium,
        label,
        rl.Vector2(int(px - size.x / 2), int(rail_y + 15)),
        17,
        0,
        color,
      )

    statement = self._activation_statement(snapshot)
    rl.draw_text_ex(
      normal,
      statement,
      rl.Vector2(int(x), int(y + 62)),
      21,
      0,
      (
        self._lifecycle_color(snapshot.lifecycle)
        if snapshot.lifecycle is not None
        else _GRAY
      ),
    )

  @staticmethod
  def _activation_statement(snapshot: DashboardSnapshot) -> str:
    lifecycle = snapshot.lifecycle
    if lifecycle is None:
      return "Controller activation unavailable; learner progress never activates steering."
    if lifecycle.controller_state == "approved":
      return "Approved BLaTv2 profile is steering; all runtime gates remain authoritative."
    if lifecycle.controller_state == "provisional":
      return "Provisional BLaTv2 profile is steering under the rollback contract."
    if lifecycle.controller_state == "staged":
      return "Stock is steering; an externally gated profile is staged for activation."
    if lifecycle.controller_state == "rollback_pending":
      return "Stock is steering; the rejected provisional profile is pending retirement."
    if lifecycle.controller_state == "unavailable":
      return "Controller status is unavailable; no activation is inferred."
    return "Stock is steering; collected learning data is display-only."
