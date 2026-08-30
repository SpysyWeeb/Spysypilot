import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.radard import LOW_SPEED_LEAD_MIN_CNT, LOW_SPEED_LEAD_MIN_TIME, KalmanParams, Track, get_lead


def track(d_rel, y_rel=0.1, v_ego=2.5, frames=LOW_SPEED_LEAD_MIN_CNT + 1, identifier=1):
  # a stationary radar return ahead, seen for `frames` cycles
  t = Track(identifier, 0.0, KalmanParams(DT_MDL))
  for _ in range(frames):
    t.update(d_rel, y_rel, -v_ego, 0.0)
  return t


class TestLowSpeedOverride:
  def test_a_close_return_needs_age_and_distance_before_it_can_be_the_lead(self):
    assert not track(2.0, frames=LOW_SPEED_LEAD_MIN_CNT - 1).potential_low_speed_lead(2.5)
    assert track(2.0).potential_low_speed_lead(2.5)
    # route 0x2a: a return at 1.1 m while rolling at 2.5 m/s is under the bumper, not a lead
    assert not track(1.1).potential_low_speed_lead(2.5)
    assert track(1.1).potential_low_speed_lead(1.0)                    # crawling, the stock 0.75 m floor rules
    assert not track(0.7).potential_low_speed_lead(0.5)
    assert not track(2.0, y_rel=1.2).potential_low_speed_lead(2.5)

  def test_the_floor_scales_with_speed(self):
    for v_ego in (1.0, 2.0, 3.5):
      floor = max(0.75, LOW_SPEED_LEAD_MIN_TIME * v_ego)
      assert not track(floor - 0.05).potential_low_speed_lead(v_ego)
      assert track(floor + 0.05).potential_low_speed_lead(v_ego)

  def test_get_lead_ignores_a_return_under_the_bumper(self):
    lead_msg = messaging.new_message('modelV2').modelV2.init('leadsV3', 1)[0]
    tracks = {1: track(1.1, identifier=1), 2: track(9.0, identifier=2)}
    lead = get_lead(2.5, False, tracks, lead_msg, 2.5, 0.0, low_speed_override=True)
    assert lead['present'] and abs(lead['dRel'] - 9.0) < 1e-6
