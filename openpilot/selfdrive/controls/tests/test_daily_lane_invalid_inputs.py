import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.tests.test_lane_change_release import frame, get_cp, mpc_follows

from openpilot.selfdrive.controls.lib.lane_change_gap import LaneChangeGap, LaneLines, inputs_valid
from openpilot.selfdrive.controls.tests.test_lane_change_gap import (CP, DT_MDL, FAR, V_EGO, get_car_state, get_model,
                                                                  get_radar_state, get_tracks, path_for, step)
from openpilot.selfdrive.modeld.constants import ModelConstants


POLYLINES = ['position', 'lane0', 'lane1', 'lane2', 'lane3']
BAD_GEOMETRY = [0, 1, 2, ModelConstants.IDX_N - 1, ModelConstants.IDX_N + 1,
                'short_x', 'short_y', 'duplicate_first', 'duplicate_middle', 'duplicate_last',
                'reversed', 'nonmonotonic', 'stationary', 'nan_x', 'nan_y', 'inf_x', 'inf_y']


def corrupt_polyline(model, polyline, malformed):
  line = model.position if polyline == 'position' else model.laneLines[int(polyline[-1])]
  x, y = list(line.x), list(line.y)
  if isinstance(malformed, int):
    # Keep all surviving samples intact; an extra sample still extends forward.
    x, y = (x + [x[-1] + 1.0])[:malformed], (y + [y[-1]])[:malformed]
  elif malformed == 'short_x':
    x = x[:-1]
  elif malformed == 'short_y':
    y = y[:-1]
  elif malformed.startswith('duplicate_'):
    i = {'duplicate_first': 1, 'duplicate_middle': len(x) // 2, 'duplicate_last': len(x) - 1}[malformed]
    x[i] = x[i - 1]
  elif malformed == 'reversed':
    x, y = x[::-1], y[::-1]
  elif malformed == 'nonmonotonic':
    i = len(x) // 2
    x[i], x[i + 1] = x[i + 1], x[i]
  elif malformed == 'stationary':
    x = [0.0] * len(x)
  else:
    values = x if malformed.endswith('_x') else y
    values[len(values) // 2] = float(malformed.split('_')[0])
  line.x, line.y = x, y


@pytest.mark.parametrize('polyline', POLYLINES)
@pytest.mark.parametrize('malformed', BAD_GEOMETRY)
@pytest.mark.parametrize('warm', [False, True], ids=['cold', 'released'])
def test_unusable_polyline_resets_helper_without_clearance(polyline, malformed, warm):
  model = get_model(path=path_for(1.5, FAR))
  radar = get_radar_state(d_rel=FAR, lead_two='same')
  CS, tracks = get_car_state(), get_tracks()
  gap = LaneChangeGap(CP, DT_MDL)
  assert inputs_valid(model, CS, radar, tracks, V_EGO, V_EGO + 5.0)
  if warm:
    assert step(gap, model=model, radar_state=radar) < 0.0
    assert gap.accelerate and all(gap.released)
    assert gap.crossed is not None
  corrupt_polyline(model, polyline, malformed)
  assert not inputs_valid(model, CS, radar, tracks, V_EGO, V_EGO + 5.0)
  for _ in range(5):
    assert step(gap, model=model, radar_state=radar) == 0.0
    assert not gap.accelerate and not gap.armed and not any(gap.released)
    assert gap.followed(radar) is radar
    assert gap.lines_seen == [None, None] and gap.lines_age == [np.inf, np.inf]
    assert gap.crossed is None and gap.lane_width is None
    assert gap.relax_timer == gap.blocked_timer == 0.0
  # Recovery cannot revive a warm release during the same change.
  recovered = get_model(path=path_for(1.5, FAR), probs=(0.0,) * 4)
  if warm:
    assert step(gap, model=recovered, radar_state=radar) == 0.0
    assert not gap.accelerate and not any(gap.released)
  recovered.meta.laneChangeState = 'off'
  step(gap, model=recovered, radar_state=radar)
  recovered.meta.laneChangeState = 'laneChangeStarting'
  assert step(gap, model=recovered, radar_state=radar) < 0.0
  assert gap.accelerate and all(gap.released)


@pytest.mark.parametrize('polyline', POLYLINES)
@pytest.mark.parametrize('malformed', BAD_GEOMETRY)
@pytest.mark.parametrize('warm', [False, True], ids=['cold', 'released'])
def test_unusable_polyline_preserves_real_planner_mpc_and_fcw(polyline, malformed, warm, monkeypatch):
  planner = LongitudinalPlanner(get_cp(), init_v=12.0)
  ordinary = LongitudinalPlanner(get_cp(bsm=warm), init_v=12.0)
  sm = frame(12.0, 0.0, 40.0, 11.0, y_ego=1.5, path=2.0)
  if warm:
    # Both solvers have the same pre-fault history; the control then disables passing.
    for _ in range(5):
      planner.update(sm)
      ordinary.update(sm)
    assert all(planner.lane_change_gap.released) and all(ordinary.lane_change_gap.released)
    assert not mpc_follows(planner, 40.0, 11.0)
  ordinary.lane_change_gap.enabled = False
  corrupt_polyline(sm['modelV2'], polyline, malformed)
  assert sm.all_checks(['modelV2'])
  raw = sm['radarState'].to_dict()
  seen = []
  process_lead = planner.mpc.process_lead_model

  def record(anchor, lead):
    seen.append(lead.to_dict())
    return process_lead(anchor, lead)

  monkeypatch.setattr(planner.mpc, 'process_lead_model', record)
  for _ in range(5):
    seen.clear()
    planner.update(sm)
    ordinary.update(sm)
    assert planner.lane_change_gap.t_follow_pad == 0.0
    assert not planner.lane_change_gap.accelerate and not any(planner.lane_change_gap.released)
    assert planner.lane_change_gap.followed(sm['radarState']) is sm['radarState']
    assert seen == [raw['leadOne'], raw['leadTwo']]
    assert sm['radarState'].to_dict() == raw
    assert mpc_follows(planner, 40.0, 11.0)
    np.testing.assert_array_equal(planner.mpc.params, ordinary.mpc.params)
    assert planner.mpc.crash_cnt == ordinary.mpc.crash_cnt
    assert planner.fcw == ordinary.fcw
    assert planner.output_a_target == ordinary.output_a_target
  # Malformed model geometry must not hide an imminent collision from the real crash check.
  for lead in (sm['radarState'].leadOne, sm['radarState'].leadTwo):
    lead.dRel, lead.vLead, lead.vLeadK, lead.vRel = 5.0, 0.0, 0.0, -12.0
  for _ in range(5):
    planner.update(sm)
    ordinary.update(sm)
  assert planner.fcw and ordinary.fcw
  assert planner.mpc.crash_cnt == ordinary.mpc.crash_cnt
  np.testing.assert_array_equal(planner.mpc.params, ordinary.mpc.params)
  assert sm['radarState'].leadOne.present and sm['radarState'].leadTwo.present


def test_absent_position_is_not_a_neutral_path():
  model = get_model(path=None)
  radar = get_radar_state(d_rel=FAR, lead_two='same')
  gap = LaneChangeGap(CP, DT_MDL)
  assert not inputs_valid(model, get_car_state(), radar, get_tracks(), V_EGO, V_EGO + 5.0)
  assert step(gap, model=model, radar_state=radar) == 0.0
  assert not gap.accelerate and not any(gap.released)
  assert gap.followed(radar) is radar


@pytest.mark.parametrize('low_confidence', [False, True])
def test_fully_shaped_geometry_keeps_real_planner_passing(low_confidence):
  sm = frame(12.0, 0.0, 40.0, 11.0, y_ego=1.5, path=2.0)
  if low_confidence:
    sm['modelV2'].laneLineProbs = [0.0] * 4
    sm['modelV2'].laneLineStds = [1.0] * 4
  planner = LongitudinalPlanner(get_cp(), init_v=12.0)
  assert inputs_valid(sm['modelV2'], sm['carState'], sm['radarState'], sm['radarTracks'], 12.0, 22.35)
  for _ in range(5):
    planner.update(sm)
    assert planner.lane_change_gap.accelerate and all(planner.lane_change_gap.released)
    assert planner.lane_change_gap.t_follow_pad == pytest.approx(-0.4)
    assert not mpc_follows(planner, 40.0, 11.0) and not planner.fcw
    assert sm['radarState'].leadOne.present and sm['radarState'].leadTwo.present


@pytest.mark.parametrize('malformed', ['path_shape', 'lane_shape', 'track_nan', 'lead_nan'])
def test_malformed_geometry_or_nonfinite_radar_cannot_relax_or_release(malformed):
  model = get_model(path=path_for(1.5, FAR))
  radar = get_radar_state(d_rel=FAR)
  tracks = get_tracks()
  radar_ok = True
  if malformed == 'path_shape':
    model.position.y = []
  elif malformed == 'lane_shape':
    model.laneLines[1].y = []
    radar_ok = False
  elif malformed == 'track_nan':
    tracks = get_tracks((30.0, float('nan'), -1.0))
  else:
    radar.leadOne.vLead = float('nan')
  gap = LaneChangeGap(CP, DT_MDL)
  assert step(gap, model=model, radar_state=radar, tracks=tracks, radar_ok=radar_ok) == 0.0
  assert not gap.accelerate
  assert not any(gap.released)


@pytest.mark.parametrize('line_count', [0, 1, 2, 3, 5])
@pytest.mark.parametrize('warm', [False, True])
def test_lane_line_count_fails_closed_and_discards_cached_geometry(line_count, warm):
  model = get_model(path=path_for(1.5, FAR))
  radar = get_radar_state(d_rel=FAR, lead_two='same')
  gap = LaneChangeGap(CP, DT_MDL)
  if warm:
    step(gap, model=model, radar_state=radar)
    assert all(gap.released)
  model.init('laneLines', line_count)
  for line in model.laneLines:
    line.x, line.y = [0.0, 30.0], [-1.6, -1.6]
  model.laneLineProbs = [1.0] * line_count
  assert step(gap, model=model, radar_state=radar) == 0.0
  assert not gap.accelerate
  assert not any(gap.released)
  assert gap.followed(radar) is radar
  assert gap.lines_seen == [None, None]
  assert gap.crossed is None and gap.lane_width is None


@pytest.mark.parametrize('prob_count', [0, 1, 2, 3, 5])
def test_lane_probability_count_fails_closed(prob_count):
  model = get_model(path=path_for(1.5, FAR))
  model.laneLineProbs = [1.0] * prob_count
  gap = LaneChangeGap(CP, DT_MDL)
  assert step(gap, model=model, radar_state=get_radar_state(d_rel=FAR)) == 0.0
  assert not gap.accelerate and not any(gap.released)


@pytest.mark.parametrize('line_index', range(4))
def test_empty_lane_coordinate_pair_is_not_clearance(line_index):
  model = get_model(path=path_for(1.5, FAR))
  model.laneLines[line_index].x = []
  model.laneLines[line_index].y = []
  gap = LaneChangeGap(CP, DT_MDL)
  assert step(gap, model=model, radar_state=get_radar_state(d_rel=FAR)) == 0.0
  assert not gap.accelerate and not any(gap.released)


def test_four_line_low_confidence_fallback_still_allows_the_valid_pass():
  model = get_model(path=path_for(1.5, FAR), probs=(0.0, 0.0, 0.0, 0.0))
  gap = LaneChangeGap(CP, DT_MDL)
  assert step(gap, model=model, radar_state=get_radar_state(d_rel=FAR, lead_two='same')) < 0.0
  assert gap.accelerate and all(gap.released)


@pytest.mark.parametrize('line_count', [0, 1, 2, 3])
def test_truncated_lanes_keep_both_raw_leads_in_actual_planner_mpc_and_fcw(line_count):
  planner = LongitudinalPlanner(get_cp(), init_v=12.0)
  ordinary = LongitudinalPlanner(get_cp(bsm=False), init_v=12.0)
  sm = frame(12.0, 0.0, 40.0, 11.0, y_ego=1.5, path=2.0)
  # Keep the remaining geometry intact: missing structure, not a malformed fixture, is the defect.
  sm['modelV2'].laneLines = [line.to_dict() for line in sm['modelV2'].laneLines][:line_count]
  sm['modelV2'].laneLineProbs = list(sm['modelV2'].laneLineProbs)[:line_count]
  assert sm.all_checks(['modelV2'])
  for _ in range(5):
    planner.update(sm)
    ordinary.update(sm)
    assert planner.lane_change_gap.t_follow_pad == 0.0
    assert not planner.lane_change_gap.accelerate
    assert not any(planner.lane_change_gap.released)
    assert planner.lane_change_gap.followed(sm['radarState']) is sm['radarState']
    assert mpc_follows(planner, 40.0, 11.0)
    np.testing.assert_array_equal(planner.mpc.params, ordinary.mpc.params)
    assert planner.fcw == ordinary.fcw
    assert planner.output_a_target == ordinary.output_a_target
  # The same malformed geometry must not hide an imminent collision from the real crash check.
  for lead in (sm['radarState'].leadOne, sm['radarState'].leadTwo):
    lead.dRel, lead.vLead, lead.vLeadK, lead.vRel = 5.0, 0.0, 0.0, -12.0
  for _ in range(5):
    planner.update(sm)
    ordinary.update(sm)
  assert planner.fcw and ordinary.fcw
  np.testing.assert_array_equal(planner.mpc.params, ordinary.mpc.params)
  assert sm['radarState'].leadOne.present and sm['radarState'].leadTwo.present


def test_release_recovery_guard_cannot_assume_current_braking_will_continue():
  from openpilot.selfdrive.controls.lib.lane_change_gap import too_close
  # Releasing the lead withdraws the MPC braking owner, so its negative measured
  # acceleration cannot count as future headroom for the accelerating pass.
  assert too_close(45.0, -7.0, 22.0, -2.6, 1.0, 7.0)


def test_invalid_input_discards_cached_lane_geometry_before_recovery():
  gap = LaneChangeGap(CP, DT_MDL)
  step(gap, model=get_model(left_y=-1.1, right_y=2.4))
  malformed = get_model()
  malformed.position.x = [0.0, 1.0]
  malformed.position.y = []
  step(gap, model=malformed)
  # After the failed evidence, an unconfident frame must use the same fallback as a cold policy.
  recovered = get_model(probs=(0.0, 0.0, 0.0, 0.0))
  cold = LaneChangeGap(CP, DT_MDL)
  step(gap, model=recovered)
  step(cold, model=recovered)
  assert LaneLines(recovered, gap.crossed, gap.lane_width).band(recovered.meta.laneChangeDirection, 30.0) == \
         LaneLines(recovered, cold.crossed, cold.lane_width).band(recovered.meta.laneChangeDirection, 30.0)
