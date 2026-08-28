import unittest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.radard import (
  LOW_SPEED_LEAD_MIN_CNT,
  KalmanParams,
  Track,
  V_EGO_STATIONARY,
)


def radar_track():
  return Track(1, 0.0, KalmanParams(DT_MDL))


def update_track(track, d_rel=3.6, y_rel=0.0):
  track.update(
    d_rel=d_rel,
    y_rel=y_rel,
    v_rel=0.0,
    v_lead=0.0,
  )


class TestLowSpeedRadarOverride(unittest.TestCase):
  def test_unconfirmed_close_track_must_persist(self):
    track = radar_track()
    for _ in range(LOW_SPEED_LEAD_MIN_CNT - 1):
      update_track(track)
      self.assertFalse(track.potential_low_speed_lead(2.0))

    update_track(track)
    self.assertTrue(track.potential_low_speed_lead(2.0))

  def test_existing_spatial_and_speed_guards_remain(self):
    track = radar_track()
    for _ in range(LOW_SPEED_LEAD_MIN_CNT):
      update_track(track)

    self.assertFalse(track.potential_low_speed_lead(V_EGO_STATIONARY))
    track.yRel = 1.1
    self.assertFalse(track.potential_low_speed_lead(2.0))
    track.yRel = 0.0
    track.dRel = 0.7
    self.assertFalse(track.potential_low_speed_lead(2.0))


if __name__ == "__main__":
  unittest.main()
