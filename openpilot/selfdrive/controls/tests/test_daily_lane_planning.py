import pytest

from opendbc.car.hyundai.values import HyundaiFlags
from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.tests.test_daily_blotv3_integration import PlannerMessages, car_params
from openpilot.selfdrive.controls.tests.test_lane_change_gap import get_model, get_radar_state, get_tracks
from openpilot.selfdrive.modeld.constants import ModelConstants


def lane_params():
  cp = car_params()
  cp.flags |= HyundaiFlags.HAS_BSM.value
  cp.radarUnavailable = False
  return cp


def lane_frame(path=0.0, v_ego=15.0, v_lead=14.0, distance=40.0):
  sm = PlannerMessages(v_ego=v_ego, experimental=False, e2e_accel=1.0)
  sm['carState'].rightBlinker = True
  sm['carState'].canValid = True
  sm['modelV2'] = get_model(direction=log.LaneChangeDirection.right, path=path, lead_y=0.0, lead_std=0.3)
  sm['modelV2'].velocity.x = [v_ego] * ModelConstants.IDX_N
  sm['modelV2'].action.desiredAcceleration = 1.0
  sm['radarState'] = get_radar_state(d_rel=distance, v_rel=v_lead - v_ego, v_ego=v_ego)
  sm['radarState'].leadOne.vLeadK = v_lead
  sm['radarState'].leadOne.modelProb = 1.0
  sm['radarState'].leadOne.aLeadTau = 1.5
  sm['radarTracks'] = get_tracks((distance, 0.0, v_lead - v_ego, 7))
  sm.alive = dict.fromkeys(sm, True)
  sm.valid = dict.fromkeys(sm, True)
  sm.freq_ok = dict.fromkeys(sm, True)
  sm.logMonoTime = dict.fromkeys(sm, 1_000_000_000)
  return sm


def test_passing_window_reaches_the_real_mpc_following_gap():
  planner = LongitudinalPlanner(lane_params(), init_v=15.0)
  sm = lane_frame()
  planner.update(sm)
  assert planner.mpc.params[0, 4] == pytest.approx(0.6)
  assert sm['radarState'].leadOne.present


def test_raw_radar_subscription_is_optional_for_plans_but_required_for_passing():
  from openpilot.selfdrive.controls import plannerd
  from openpilot.selfdrive.controls.lib.longitudinal_planner import get_radar_tracks
  from openpilot.selfdrive.controls.tests.test_lane_change_release import frame, get_cp
  sm = plannerd.get_submaster()
  assert 'radarTracks' in sm.services
  good = frame(12.0, 0.0, 40.0, 11.0, y_ego=1.5, path=2.0)
  for service in sm.services:
    if service in good.data:
      sm.data[service] = good[service]
      sm.alive[service] = sm.freq_ok[service] = sm.valid[service] = True
  for health in ('alive', 'freq_ok', 'valid'):
    sm.alive['radarTracks'] = sm.freq_ok['radarTracks'] = sm.valid['radarTracks'] = True
    getattr(sm, health)['radarTracks'] = False
    assert sm.all_checks()
    assert get_radar_tracks(sm) is None
    planner = LongitudinalPlanner(get_cp(), init_v=12.0)
    planner.update(sm)
    assert not any(planner.lane_change_gap.released)
  sm.alive['radarTracks'] = sm.freq_ok['radarTracks'] = sm.valid['radarTracks'] = True
  assert get_radar_tracks(sm) is sm['radarTracks']


def anchored_release_frame():
  from openpilot.selfdrive.controls.tests.test_lane_change_release import frame
  from openpilot.selfdrive.controls.tests.test_daily_blotv3_integration import model_lead
  from openpilot.selfdrive.modeld.constants import ModelConstants
  sm = frame(12.0, 0.0, 40.0, 11.0, y_ego=1.5, path=2.0)
  times = ModelConstants.LEAD_T_IDXS
  forecasts = [model_lead([11.0] * len(times), x=[40.0 + 11.0 * t for t in times]) for _ in range(2)]
  for lead in forecasts:
    lead.y = [-1.5] * len(ModelConstants.LEAD_T_IDXS)
    lead.yStd = [0.3] * len(ModelConstants.LEAD_T_IDXS)
  sm['modelV2'].leadsV3 = forecasts
  return sm


def test_release_masks_both_valid_forecasts_without_mutating_raw_leads():
  from openpilot.selfdrive.controls.lib.longitudinal_lead import anchor_model_lead
  from openpilot.selfdrive.controls.tests.test_lane_change_release import mpc_follows
  from openpilot.selfdrive.controls.tests.test_daily_curve_integration import Publisher
  sm = anchored_release_frame()
  for model, radar in zip(sm['modelV2'].leadsV3, (sm['radarState'].leadOne, sm['radarState'].leadTwo), strict=True):
    assert anchor_model_lead(model, radar) is not None
  planner = LongitudinalPlanner(lane_params(), init_v=12.0)
  for _ in range(5):
    planner.update(sm)
  assert all(planner.lane_change_gap.released)
  assert not mpc_follows(planner, 40.0, 11.0)
  assert not planner.fcw
  pm = Publisher()
  planner.publish(sm, pm)
  assert pm.messages['longitudinalPlan'].longitudinalPlan.hasLead
  assert sm['radarState'].leadOne.present and sm['radarState'].leadTwo.present


@pytest.mark.parametrize('service', ['radarTracks', 'radarState', 'modelV2', 'carState'])
@pytest.mark.parametrize('health', ['alive', 'freq_ok', 'valid'])
def test_unhealthy_inputs_restore_both_leads_in_the_current_frame(service, health):
  from openpilot.selfdrive.controls.tests.test_lane_change_release import mpc_follows
  sm = anchored_release_frame()
  planner = LongitudinalPlanner(lane_params(), init_v=12.0)
  planner.update(sm)
  assert all(planner.lane_change_gap.released)
  getattr(sm, health)[service] = False
  planner.update(sm)
  assert not any(planner.lane_change_gap.released)
  assert planner.lane_change_gap.t_follow_pad == 0.0
  assert mpc_follows(planner, 40.0, 11.0)


@pytest.mark.parametrize('error', ['canError', 'radarFault', 'wrongConfig', 'radarUnavailableTemporary'])
def test_radar_error_flags_are_not_clearance_even_when_service_flags_are_healthy(error):
  sm = anchored_release_frame()
  planner = LongitudinalPlanner(lane_params(), init_v=12.0)
  planner.update(sm)
  assert all(planner.lane_change_gap.released)
  setattr(sm['radarTracks'].errors, error, True)
  planner.update(sm)
  assert not any(planner.lane_change_gap.released)
  assert not planner.lane_change_gap.accelerate


@pytest.mark.parametrize('pedal', ['gasPressed', 'brakePressed'])
def test_pedal_on_first_change_frame_cannot_release_leads(pedal):
  sm = anchored_release_frame()
  setattr(sm['carState'], pedal, True)
  planner = LongitudinalPlanner(lane_params(), init_v=12.0)
  planner.update(sm)
  assert not any(planner.lane_change_gap.released)
  assert planner.lane_change_gap.t_follow_pad == 0.0


def test_passing_relaxation_preserves_additional_blotv3_risk_padding(monkeypatch):
  from dataclasses import replace
  planner = LongitudinalPlanner(lane_params(), init_v=15.0)
  original = planner.supervisor.update
  monkeypatch.setattr(planner.supervisor, 'update', lambda *args: replace(original(*args), t_follow_pad=0.3))
  planner.update(lane_frame())
  assert planner.lane_change_gap.t_follow_pad < 0.0
  assert planner.mpc.params[0, 4] == pytest.approx(0.6 + 0.3)


@pytest.mark.parametrize('competitor', ['curve', 'e2e', 'committed_stop'])
def test_releasing_old_lane_leads_cannot_override_other_braking_owners(monkeypatch, competitor):
  from openpilot.selfdrive.controls.lib.force_stops import ForceStopsResult
  from openpilot.selfdrive.controls.tests.test_daily_curve_integration import set_path
  sm = anchored_release_frame()
  planner = LongitudinalPlanner(lane_params(), init_v=12.0)
  if competitor == 'e2e':
    sm['selfdriveState'].experimentalMode = True
    sm['modelV2'].action.desiredAcceleration = -3.0
  elif competitor == 'committed_stop':
    monkeypatch.setattr(planner.force_stops, 'update', lambda *args: ForceStopsResult(a_target=-3.0))
  else:
    set_path(sm, curvature=0.06, v_ego=12.0)
    sm['carControl'].latActive = True
    state = sm['controlsState'].lateralControlState.init('torqueState')
    state.active = True
    state.output = 1.0
    state.desiredLateralAccel = 3.0
    state.actualLateralAccel = 2.0
    state.error = 1.0
  for _ in range(20):
    planner.update(sm)
  assert all(planner.lane_change_gap.released)
  if competitor == 'curve':
    assert planner.curve.a_target is not None
    assert planner.output_a_target <= planner.curve.a_target
    assert planner.output_a_target <= -0.8
  else:
    assert planner.output_a_target <= -3.0
