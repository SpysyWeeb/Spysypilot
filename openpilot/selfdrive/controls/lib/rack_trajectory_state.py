"""Measured rack-state estimation for trajectory feedback."""
from __future__ import annotations

from openpilot.common.filter_simple import FirstOrderFilter

MEASURED_RATE_FILTER_RC_S = .05


class RackRateEstimator:
  def __init__(self, dt: float, filter_rc_s: float = MEASURED_RATE_FILTER_RC_S) -> None:
    self.dt = dt
    self.filter_rc_s = filter_rc_s
    self.previous_angle_deg: float | None = None
    self.direction = 0
    self.raw_signed_episode = False
    self.rate_filter = FirstOrderFilter(0.0, filter_rc_s, dt)
    self.rate_filter_valid = False

  def reset(self) -> None:
    self.previous_angle_deg = None
    self.direction = 0
    self.raw_signed_episode = False
    self.rate_filter.x = 0.0
    self.rate_filter_valid = False

  def update(self, angle_deg: float, raw_rate_deg_s: float) -> tuple[float, bool]:
    magnitude = abs(raw_rate_deg_s)
    if magnitude == 0.0:
      self.direction = 0
      self.raw_signed_episode = False
      rate, valid = 0.0, True
    elif raw_rate_deg_s < 0.0:
      self.direction = -1
      self.raw_signed_episode = True
      rate, valid = -magnitude, True
    elif self.raw_signed_episode:
      self.direction = 1
      rate, valid = magnitude, True
    elif self.previous_angle_deg is not None and angle_deg != self.previous_angle_deg:
      self.direction = 1 if angle_deg > self.previous_angle_deg else -1
      rate, valid = self.direction * magnitude, True
    elif self.direction:
      rate, valid = self.direction * magnitude, True
    else:
      rate, valid = 0.0, False
    self.previous_angle_deg = angle_deg
    if not valid:
      return rate, False
    if not self.rate_filter_valid:
      self.rate_filter.x = rate
      self.rate_filter_valid = True
    else:
      rate = float(self.rate_filter.update(rate))
    return rate, True
