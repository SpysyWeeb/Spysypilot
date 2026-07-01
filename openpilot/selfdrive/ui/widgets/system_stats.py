import collections
import os
import time
import pyray as rl

from openpilot.common.hardware.hw import Paths
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget

SAMPLE_INTERVAL = 1.0
HISTORY_LEN = 60

_BG      = rl.Color(40, 40, 40, 255)
_PANEL   = rl.Color(55, 55, 55, 255)
_DIM     = rl.Color(255, 255, 255, 150)
_GRID    = rl.Color(255, 255, 255, 25)
_TRACK   = rl.Color(255, 255, 255, 35)
_GOOD    = rl.Color(70, 200, 100, 255)
_WARNING = rl.Color(234, 160, 50, 255)
_DANGER  = rl.Color(220, 60, 60, 255)


class _Metric:
  def __init__(self, label: str, unit: str, warn: float, danger: float, plot_max: float = 100.0, dynamic_max: bool = False):
    self.label = label
    self.unit = unit
    self.warn = warn
    self.danger = danger
    self.plot_max_base = plot_max
    self.dynamic_max = dynamic_max
    self.history: collections.deque[float] = collections.deque(maxlen=HISTORY_LEN)
    self.current = 0.0

  def push(self, value: float) -> None:
    self.current = value
    self.history.append(value)

  def color(self) -> rl.Color:
    if self.current >= self.danger:
      return _DANGER
    if self.current >= self.warn:
      return _WARNING
    return _GOOD

  def plot_max(self) -> float:
    if not self.dynamic_max:
      return self.plot_max_base
    peak = max(self.history, default=0.0)
    return max(self.plot_max_base, peak * 1.15)


class SystemStatsWidget(Widget):
  def __init__(self):
    super().__init__()
    self._background_tap_callback = None
    self._last_sample = 0.0

    self._cpu = _Metric("CPU", "%", 75, 90)
    self._ram = _Metric("RAM", "%", 75, 90)
    self._power = _Metric("POWER", "W", 13, 15, plot_max=20.0, dynamic_max=True)
    self._fan = _Metric("FAN", "%", 80, 95)
    self._metrics = [self._cpu, self._ram, self._power, self._fan]

    self._used_gb = 0.0
    self._total_gb = 0.0

  def set_background_tap_callback(self, cb) -> None:
    self._background_tap_callback = cb

  def _handle_mouse_release(self, mouse_pos) -> None:
    if self._background_tap_callback:
      self._background_tap_callback(mouse_pos)

  def _update_state(self):
    now = time.monotonic()
    if now - self._last_sample < SAMPLE_INTERVAL:
      return
    self._last_sample = now

    sm = ui_state.sm
    device_state = sm['deviceState']
    cores = list(device_state.cpuUsagePercent)
    self._cpu.push(sum(cores) / len(cores) if cores else 0.0)
    self._ram.push(float(device_state.memoryUsagePercent))
    self._power.push(device_state.powerDrawW)
    self._fan.push(float(device_state.fanSpeedPercentDesired))

    self._refresh_storage()

  def _refresh_storage(self) -> None:
    try:
      st = os.statvfs(Paths.log_root())
      total = st.f_blocks * st.f_frsize
      free = st.f_bavail * st.f_frsize
      self._total_gb = total / 1e9
      self._used_gb = (total - free) / 1e9
    except OSError:
      pass

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rounded(rect, 0.025, 10, _BG)

    pad = 40
    gap = 20
    storage_h = 130

    grid_rect = rl.Rectangle(
      rect.x + pad, rect.y + pad,
      rect.width - pad * 2,
      rect.height - pad * 2 - storage_h - gap,
    )

    cell_w = (grid_rect.width - gap) / 2
    cell_h = (grid_rect.height - gap) / 2

    for i, metric in enumerate(self._metrics):
      col, row = i % 2, i // 2
      cell = rl.Rectangle(
        grid_rect.x + col * (cell_w + gap),
        grid_rect.y + row * (cell_h + gap),
        cell_w, cell_h,
      )
      self._draw_graph_panel(cell, metric)

    storage_rect = rl.Rectangle(
      rect.x + pad, grid_rect.y + grid_rect.height + gap,
      rect.width - pad * 2, storage_h,
    )
    self._draw_storage_panel(storage_rect)

  def _draw_graph_panel(self, rect: rl.Rectangle, metric: _Metric):
    rl.draw_rectangle_rounded(rect, 0.08, 8, _PANEL)

    label_pad = 18
    fn = gui_app.font(FontWeight.NORMAL)
    fb = gui_app.font(FontWeight.BOLD)
    color = metric.color()

    rl.draw_text_ex(fn, metric.label, rl.Vector2(int(rect.x + label_pad), int(rect.y + 14)), 28, 0, _DIM)

    value_text = f"{metric.current:.0f}{metric.unit}"
    vw = rl.measure_text_ex(fb, value_text, 34, 0).x
    rl.draw_text_ex(
      fb, value_text,
      rl.Vector2(int(rect.x + rect.width - label_pad - vw), int(rect.y + 10)),
      34, 0, color,
    )

    plot_top = rect.y + 58
    plot_rect = rl.Rectangle(
      rect.x + label_pad, plot_top,
      rect.width - label_pad * 2, rect.y + rect.height - 16 - plot_top,
    )
    self._draw_sparkline(plot_rect, metric, color)

  def _draw_sparkline(self, rect: rl.Rectangle, metric: _Metric, color: rl.Color):
    rl.draw_line_ex(
      rl.Vector2(rect.x, rect.y + rect.height), rl.Vector2(rect.x + rect.width, rect.y + rect.height), 1, _GRID,
    )

    pts = list(metric.history)
    if len(pts) < 2:
      return

    vmax = metric.plot_max()
    step = rect.width / (HISTORY_LEN - 1)
    start_x = rect.x + rect.width - (len(pts) - 1) * step

    prev = None
    for i, v in enumerate(pts):
      frac = max(0.0, min(1.0, v / vmax)) if vmax > 0 else 0.0
      point = rl.Vector2(start_x + i * step, rect.y + rect.height - frac * rect.height)
      if prev is not None:
        rl.draw_line_ex(prev, point, 2, color)
      prev = point

  def _draw_storage_panel(self, rect: rl.Rectangle):
    rl.draw_rectangle_rounded(rect, 0.08, 8, _PANEL)
    pad = 24
    fn = gui_app.font(FontWeight.NORMAL)
    fb = gui_app.font(FontWeight.BOLD)

    rl.draw_text_ex(fn, "STORAGE", rl.Vector2(int(rect.x + pad), int(rect.y + 14)), 28, 0, _DIM)

    pct = (self._used_gb / self._total_gb * 100) if self._total_gb > 0 else 0.0
    color = _DANGER if pct >= 90 else _WARNING if pct >= 75 else _GOOD

    value_text = f"{self._used_gb:.0f} GB / {self._total_gb:.0f} GB"
    vw = rl.measure_text_ex(fb, value_text, 32, 0).x
    rl.draw_text_ex(
      fb, value_text,
      rl.Vector2(int(rect.x + rect.width - pad - vw), int(rect.y + 8)),
      32, 0, color,
    )

    bar_rect = rl.Rectangle(rect.x + pad, rect.y + 64, rect.width - pad * 2, 28)
    rl.draw_rectangle_rounded(bar_rect, 0.5, 8, _TRACK)
    fill_w = bar_rect.width * max(0.0, min(1.0, pct / 100))
    if fill_w > 1:
      fill_rect = rl.Rectangle(bar_rect.x, bar_rect.y, fill_w, bar_rect.height)
      rl.draw_rectangle_rounded(fill_rect, 0.5, 8, color)
