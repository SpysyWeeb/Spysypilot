from openpilot.common.realtime import DT_CTRL

# Matches DesireHelper.LANE_CHANGE_SPEED_MIN (desire_helper.py). modeld feeds DesireHelper
# the post-blinker-pause carControl.latActive, so a higher threshold here than DesireHelper's
# own floor creates a speed band where DesireHelper would attempt a lane change but this pause
# has already zeroed latActive first, permanently blocking lane changes below this fork's
# BLINKER_MIN_SPEED regardless of DesireHelper's own gating. Keep the two in sync.
BLINKER_MIN_SPEED = 20 * 0.44704  # 20 mph in m/s

class BlinkerPauseLateral:
  def __init__(self):
    self.reengage_delay = 1.0  # seconds to wait after blinker turns off
    self.blinker_off_timer = 0.0

  def update(self, CS) -> bool:
    """Return True if lateral should be paused due to blinker."""
    speed = CS.vEgo
    below_speed = speed < BLINKER_MIN_SPEED
    one_blinker = CS.leftBlinker != CS.rightBlinker

    if one_blinker and below_speed:
      self.blinker_off_timer = self.reengage_delay
    elif self.blinker_off_timer > 0:
      self.blinker_off_timer -= DT_CTRL

    return bool((one_blinker and below_speed) or self.blinker_off_timer > 0)