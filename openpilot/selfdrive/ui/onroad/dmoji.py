"""The comma four's driver-monitoring badge (the "DMoji": a person and a cone that turns with the driver's head) on the
comma 3X onroad screen, in place of the 3X face icon, shown the way StarPilot shows it on the 3X
(StarPilot `openpilot/starpilot/ui/onroad_dm.py`: LargeDriverStateRenderer, StableDriverStateRenderer, valid_observation):

- drawn at 128 px, the native size of its 128x128 icons, so they stay sharp on the 3X screen;
- the head yaw is negated on the 3X so the cone turns the way the driver's head turns;
- full strength while a face is detected, at most 35 % while none is;
- monitoring data that is missing or not plausible keeps it dim and neutral instead of moving it.

It is centred where the 3X face icon was (bottom left; bottom right for right-hand drive) and, like that icon, hidden while an
alert is shown, because the 3X alerts are drawn across the bottom of the screen. StarPilot hides it only under full-screen or
critical alerts, but StarPilot draws its own alerts elsewhere.
"""
import math

import pyray as rl

from openpilot.cereal import log
from openpilot.selfdrive.ui import UI_BORDER_SIZE
from openpilot.selfdrive.ui.mici.onroad.driver_state import DriverStateRenderer
from openpilot.selfdrive.ui.ui_state import ui_state

AlertSize = log.SelfdriveState.AlertSize

DMOJI_SIZE = 128        # px, the icons' native size
FACE_ICON_BUTTON = 192  # px, the 3X face icon's button; the DMoji takes over its centre
NO_FACE_FADE = 0.35


def valid_observation(monitor, driver) -> bool:
  """True when the monitoring message is complete and plausible enough to move the badge (StarPilot's check)."""
  try:
    vision = monitor.visionPolicyState
    return (type(monitor.isRHD) is bool and type(vision.faceDetected) is bool and
            all(math.isfinite(float(number)) for number in (vision.pose.pitch, vision.pose.yaw, vision.awarenessPercent)) and
            0 <= vision.awarenessPercent <= 100 and
            getattr(driver, 'rightDriverData' if monitor.isRHD else 'leftDriverData') is not None)
  except (AttributeError, TypeError, ValueError, OverflowError):
    return False


def dmoji_position(content_rect: rl.Rectangle, is_rhd: bool) -> tuple[float, float]:
  """Top-left corner that centres the badge where the 3X face icon was drawn."""
  offset = UI_BORDER_SIZE + FACE_ICON_BUTTON // 2
  center_x = content_rect.x + (content_rect.width - offset if is_rhd else offset)
  center_y = content_rect.y + content_rect.height - offset
  return center_x - DMOJI_SIZE / 2, center_y - DMOJI_SIZE / 2


class DMojiRenderer(DriverStateRenderer):
  def __init__(self):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, DMOJI_SIZE, DMOJI_SIZE))
    self.load_icons()

  @property
  def should_draw(self) -> bool:
    # visibility is decided by place(); freshness is handled in _update_state
    return self._should_draw

  def get_driver_data(self):
    driver_data = super().get_driver_data()
    self._face_yaw = -ui_state.sm['driverMonitoringState'].visionPolicyState.pose.yaw
    return driver_data

  def _update_state(self):
    sm = ui_state.sm
    fresh = sm.recv_frame['driverStateV2'] > ui_state.started_frame
    if fresh and valid_observation(sm['driverMonitoringState'], sm['driverStateV2']):
      super()._update_state()
      if self.should_draw:
        self._fade_filter.x = 1.0 if self._face_detected else min(self._fade_filter.x, NO_FACE_FADE)
    else:
      self._is_active = False
      self._awareness_unfull = False
      self._fade_filter.update(NO_FACE_FADE if self.should_draw else 0.0)

  def place(self, content_rect: rl.Rectangle) -> None:
    self.set_should_draw(ui_state.sm['selfdriveState'].alertSize == AlertSize.none)
    self.set_position(*dmoji_position(content_rect, self._is_rhd))

  def draw_in(self, content_rect: rl.Rectangle) -> None:
    self.place(content_rect)
    self.render()
