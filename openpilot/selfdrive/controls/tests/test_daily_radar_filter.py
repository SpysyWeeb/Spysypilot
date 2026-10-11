import pytest

from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.radard import RadarD
from openpilot.selfdrive.controls.tests.test_daily_lead_departure import rest_behind
from openpilot.selfdrive.controls.tests.test_longitudinal_planner import PALISADE_CP, frame
from openpilot.selfdrive.controls.tests.test_radard_low_speed_lead import vision_lead


def radar_frame(v_ego=2.5):
  sm = frame(v_ego)
  sm['modelV2'].leadsV3 = [log.ModelDataV2.LeadDataV3.new_message() for _ in range(2)]
  sm.seen = {'modelV2': True}
  sm.recv_frame = {'carState': 0}
  return sm


def update_radar(radar, sm, distance=9.0):
  points = [] if distance is None else [{'trackId': 7, 'dRel': distance, 'yRel': 0.1, 'vRel': -sm['carState'].vEgo}]
  sm.recv_frame['carState'] += 1
  radar.update(sm, car.RadarData(points=points))
  return radar.radar_state


def test_native_radar_age_gate_and_one_cycle_loss_restarts_qualification():
  radar, sm = RadarD(), radar_frame()
  for _ in range(19):
    assert not update_radar(radar, sm).leadOne.present
  assert update_radar(radar, sm).leadOne.present
  assert radar.tracks[7].cnt == 20
  assert not update_radar(radar, sm, None).leadOne.present
  assert 7 not in radar.tracks
  for _ in range(19):
    assert not update_radar(radar, sm).leadOne.present
  assert update_radar(radar, sm).leadOne.present


@pytest.mark.parametrize('slot', [0, 1])
def test_confirmed_lead_bypasses_age_and_range_gates_and_reaches_planner(slot):
  radar, sm = RadarD(), radar_frame()
  leads = list(sm['modelV2'].leadsV3)
  leads[slot] = vision_lead(1.1)
  sm['modelV2'].leadsV3 = leads
  state = update_radar(radar, sm, 1.1)
  lead = (state.leadOne, state.leadTwo)[slot]
  assert radar.tracks[7].cnt == 1
  assert lead.present and lead.radar and lead.radarTrackId == 7
  assert lead.dRel == pytest.approx(1.1) and lead.modelProb > 0.5
  sm.data['radarState'] = state
  planner = LongitudinalPlanner(PALISADE_CP, init_v=2.5)
  planner.update(sm)
  assert planner.mpc.binding_obstacle == 'lead'
  assert planner.output_a_target < 0.0


def test_held_car_is_not_released_while_unconfirmed_radar_requalifies():
  planner, distance = rest_behind(0)
  radar, sm = RadarD(), radar_frame(0.0)
  for _ in range(20):
    state = update_radar(radar, sm, distance)
  assert state.leadOne.present
  sm.data['radarState'] = state
  planner.update(sm)
  assert planner.output_should_stop and planner.lead_departure.holding
  # One missed radar cycle removes the track. Recreated tracks must age in again.
  for raw_distance in [None] + [distance] * 19:
    sm.data['radarState'] = update_radar(radar, sm, raw_distance)
    assert not sm['radarState'].leadOne.present
    planner.update(sm)
    assert planner.output_should_stop and planner.lead_departure.holding
  sm.data['radarState'] = update_radar(radar, sm, distance)
  assert sm['radarState'].leadOne.present
  planner.update(sm)
  assert planner.output_should_stop and planner.lead_departure.holding
