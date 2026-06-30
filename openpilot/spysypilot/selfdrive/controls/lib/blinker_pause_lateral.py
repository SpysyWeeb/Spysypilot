from openpilot.common.realtime import DT_CTRL

BLINKER_MIN_SPEED = 30 * 0.44704  # 30 mph in m/s

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