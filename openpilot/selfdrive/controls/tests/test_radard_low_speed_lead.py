from openpilot.common.test import OpenpilotTestCase
import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.radard import (LOW_SPEED_LEAD_MIN_CNT, LOW_SPEED_LEAD_MIN_TIME, RADAR_TO_CAMERA, V_EGO_STATIONARY,
                                                 KalmanParams, Track, get_lead)


def track(d_rel, y_rel=0.1, v_ego=2.5, frames=LOW_SPEED_LEAD_MIN_CNT + 1, identifier=1):
  # a stationary radar return ahead, seen for `frames` cycles
  t = Track(identifier, 0.0, KalmanParams(DT_MDL))
  for _ in range(frames):
    t.update(d_rel, y_rel, -v_ego, 0.0)
  return t


def vision_lead(d_rel, y_rel=0.1, prob=0.9):
  # a model lead on a stationary object at d_rel, with stds tight enough to match only the track there
  lead_msg = messaging.new_message('modelV2').modelV2.init('leadsV3', 1)[0]
  lead_msg.prob = prob
  lead_msg.x = [d_rel + RADAR_TO_CAMERA]
  lead_msg.y = [-y_rel]
  lead_msg.v = [0.0]
  lead_msg.a = [0.0]
  lead_msg.xStd = [1.0]
  lead_msg.yStd = [1.0]
  lead_msg.vStd = [1.0]
  return lead_msg


class TestLowSpeedLeadOverride(OpenpilotTestCase):

  def test_age_and_distance(self):
    assert not track(2.0, frames=LOW_SPEED_LEAD_MIN_CNT - 1).potential_low_speed_lead(2.5)
    assert track(2.0).potential_low_speed_lead(2.5)
    # a return at 1.1 m while rolling at 2.5 m/s is under the bumper, not a lead
    assert not track(1.1).potential_low_speed_lead(2.5)
    assert track(1.1).potential_low_speed_lead(1.0)                    # crawling, the stock 0.75 m floor rules
    assert not track(0.7).potential_low_speed_lead(0.5)
    assert not track(2.0, y_rel=1.2).potential_low_speed_lead(2.5)

  def test_age_boundary(self):
    # the helper defaults to LOW_SPEED_LEAD_MIN_CNT + 1 frames, so build the boundary tracks explicitly
    old = track(2.0, frames=LOW_SPEED_LEAD_MIN_CNT)
    young = track(2.0, frames=LOW_SPEED_LEAD_MIN_CNT - 1)
    assert old.cnt == LOW_SPEED_LEAD_MIN_CNT
    assert young.cnt == LOW_SPEED_LEAD_MIN_CNT - 1
    assert old.potential_low_speed_lead(2.5)
    assert not young.potential_low_speed_lead(2.5)

  def test_range_ceiling(self):
    assert track(24.9).potential_low_speed_lead(2.5)
    assert not track(25.0).potential_low_speed_lead(2.5)
    assert not track(25.1).potential_low_speed_lead(2.5)

  def test_speed_off_switch(self):
    assert track(5.0).potential_low_speed_lead(V_EGO_STATIONARY - 0.1)
    assert not track(5.0).potential_low_speed_lead(V_EGO_STATIONARY)
    assert not track(5.0).potential_low_speed_lead(V_EGO_STATIONARY + 1.0)

  def test_floor_scales_with_speed(self):
    for v_ego in (1.0, 2.0, 3.5):
      floor = max(0.75, LOW_SPEED_LEAD_MIN_TIME * v_ego)
      assert not track(floor - 0.05).potential_low_speed_lead(v_ego)
      assert track(floor + 0.05).potential_low_speed_lead(v_ego)

  def test_bumper_return_ignored(self):
    lead_msg = messaging.new_message('modelV2').modelV2.init('leadsV3', 1)[0]
    tracks = {1: track(1.1, identifier=1), 2: track(9.0, identifier=2)}
    lead = get_lead(2.5, False, tracks, lead_msg, 2.5, 0.0, low_speed_override=True)
    assert lead['present'] and abs(lead['dRel'] - 9.0) < 1e-6

  def test_closer_track_wins(self):
    # a qualifying track closer than the vision matched lead takes the lead
    lead_msg = vision_lead(9.0)
    tracks = {1: track(2.0, identifier=1), 2: track(9.0, identifier=2)}
    lead = get_lead(2.5, True, tracks, lead_msg, 2.5, 0.9, low_speed_override=True)
    assert lead['present'] and abs(lead['dRel'] - 2.0) < 1e-6
    assert lead['radarTrackId'] == 1

  def test_vision_lead_kept(self):
    # the closer return is under the bumper, so the vision matched lead stands
    lead_msg = vision_lead(9.0)
    tracks = {1: track(1.1, identifier=1), 2: track(9.0, identifier=2)}
    lead = get_lead(2.5, True, tracks, lead_msg, 2.5, 0.9, low_speed_override=True)
    assert lead['present'] and abs(lead['dRel'] - 9.0) < 1e-6
    assert lead['radarTrackId'] == 2 and abs(lead['modelProb'] - 0.9) < 1e-6
