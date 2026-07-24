import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import pyray as rl

from openpilot.cereal import messaging
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached


DISPLAY_SECONDS = 2.5
STALE_SECONDS = 30.0


@dataclass
class EventNotice:
  group_id: str
  event_id: str
  text: str
  failed: bool
  received_at: float


def notification_text(recorded: Any) -> tuple[str, bool]:
  if not (bool(recorded.markerWritten) and bool(recorded.currentSegmentPreserved)):
    return "Event Save Failed", True
  if str(recorded.source) == "user" or str(recorded.domain) == "manual":
    return "Event Logged", False
  if str(recorded.domain) == "lateral":
    return "Lat Event Logged", False
  if str(recorded.domain) == "longitudinal":
    return "Long Event Logged", False
  return "Event Logged", False


class EventNotificationQueue:
  def __init__(self):
    self.pending: deque[EventNotice] = deque()
    self.active: EventNotice | None = None
    self.visible_until = 0.0

  def push(self, recorded: Any, now: float) -> None:
    text, failed = notification_text(recorded)
    group_id = str(recorded.groupId) or str(recorded.eventId)
    event_id = str(recorded.eventId)
    notice = EventNotice(group_id, event_id, text, failed, now)

    if self.active is not None and self.active.group_id == group_id:
      if event_id == self.active.event_id or failed or self.active.text == "Event Logged":
        self.active.event_id = event_id
        self.active.text = text
        self.active.failed = failed
      self.visible_until = max(self.visible_until, now + DISPLAY_SECONDS)
      return

    for queued in self.pending:
      if queued.group_id == group_id:
        if event_id == queued.event_id or failed or queued.text == "Event Logged":
          queued.event_id = event_id
          queued.text = text
          queued.failed = failed
        return
    self.pending.append(notice)

  def current(self, safety_alert_visible: bool, now: float) -> EventNotice | None:
    while self.pending and now - self.pending[0].received_at > STALE_SECONDS:
      self.pending.popleft()

    if safety_alert_visible:
      if self.active is not None:
        self.pending.appendleft(self.active)
        self.active = None
      return None

    if self.active is not None and now >= self.visible_until:
      self.active = None
    if self.active is None and self.pending:
      self.active = self.pending.popleft()
      self.visible_until = now + DISPLAY_SECONDS
    return self.active


class DrivingEventNotification:
  def __init__(self):
    self._font = gui_app.font(FontWeight.SEMI_BOLD)
    self._sock = messaging.sub_sock("drivingEventRecorded", conflate=False)
    self.queue = EventNotificationQueue()

  def render(self, rect: rl.Rectangle, safety_alert_visible: bool = False) -> None:
    now = time.monotonic()
    for msg in messaging.drain_sock(self._sock):
      if msg.valid and msg.which() == "drivingEventRecorded":
        self.queue.push(msg.drivingEventRecorded, now)

    notice = self.queue.current(safety_alert_visible, now)
    if notice is None:
      return

    remaining = self.queue.visible_until - now
    scale = max(0.65, min(rect.width / 1920.0, rect.height / 1080.0))
    font_size = round(56 * scale)
    padding_x = round(38 * scale)
    padding_y = round(20 * scale)
    text_size = measure_text_cached(self._font, notice.text, font_size)
    width = text_size.x + 2 * padding_x
    height = text_size.y + 2 * padding_y
    x = rect.x + (rect.width - width) / 2
    y = rect.y + rect.height * 0.76

    fade = min(1.0, remaining / 0.35)
    alpha = round(220 * fade)
    border = rl.Color(220, 60, 60, alpha) if notice.failed else rl.Color(51, 205, 122, alpha)
    toast_rect = rl.Rectangle(x, y, width, height)
    rl.draw_rectangle_rounded(toast_rect, 0.35, 12, rl.Color(18, 24, 28, alpha))
    rl.draw_rectangle_rounded_lines_ex(toast_rect, 0.35, 12, max(2, round(3 * scale)), border)
    text_pos = rl.Vector2(x + padding_x, y + padding_y)
    rl.draw_text_ex(self._font, notice.text, text_pos, font_size, 0, rl.Color(255, 255, 255, round(255 * fade)))
