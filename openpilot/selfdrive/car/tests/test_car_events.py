from opendbc.car import DT_CTRL, structs
from openpilot.cereal import log
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.car.car_events import CarEvents

EventName = log.OnroadEvent.EventName


class TestCarEvents(OpenpilotTestCase):
  def setup_method(self):
    CP = structs.CarParams()
    CP.carFingerprint = "HYUNDAI_PALISADE"
    CP.brand = "hyundai"
    self.car_events = CarEvents(CP)
    self.CC = structs.CarControl()

  def esp_active(self, active: bool) -> bool:
    CS = structs.CarState()
    CS.espActive = active
    return EventName.espActive in self.car_events.update(CS, CS, self.CC).names

  def test_esp_active_pulse_ignored(self):
    # the ESC pulses espActive for 0.36-0.39 s over a bump
    for frames in (36, 37, 38, 39, int(0.5 / DT_CTRL) - 1):
      assert not any(self.esp_active(True) for _ in range(frames))
      assert not self.esp_active(False)

  def test_esp_active_sustained(self):
    hold = int(0.5 / DT_CTRL)
    assert not any(self.esp_active(True) for _ in range(hold - 1))
    assert all(self.esp_active(True) for _ in range(hold))
    assert not self.esp_active(False)
    # the hold restarts after a release
    assert not any(self.esp_active(True) for _ in range(hold - 1))
