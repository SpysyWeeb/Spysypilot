"""Intentional lane-release recovery, not a change to ordinary radar dropout handling."""
import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_lead import LeadObservation
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LEAD_ABSENCE_FRAMES, LEAD_PRESENCE_FRAMES, LongitudinalMpc
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.tests.test_lane_change_release import (LaneChangePlant, LaneChangeState, frame, get_cp,
                                                                        mpc_follows, released_frames, summary)


def update(mpc, radar, **kwargs):
  # one MPC update as the planner makes it, on its one reading of each radar slot
  mpc.update(radar, usable=tuple(LeadObservation.from_radar(lead).present for lead in (radar.leadOne, radar.leadTwo)), **kwargs)


def test_restored_lead_discards_free_run_change_cost_through_the_reanchor(monkeypatch):
  planner = LongitudinalPlanner(get_cp(), init_v=12.0)
  released_frames(planner, n=20)
  assert planner.mpc.binding_obstacle == 'fake'
  assert planner.mpc.a_prev.max() > planner.output_a_target
  costs = []
  original = planner.mpc.set_cost_weights

  def record(cost_weights, constraint_weights):
    costs.append((cost_weights.copy(), constraint_weights.copy()))
    original(cost_weights, constraint_weights)

  monkeypatch.setattr(planner.mpc, 'set_cost_weights', record)
  for _ in range(LEAD_PRESENCE_FRAMES + 1):
    planner.update(frame(12.0, 0.0, 40.0, 11.0, y_ego=1.5, path=2.0, blinker=False))
    assert not any(planner.lane_change_gap.released)
    assert mpc_follows(planner, 40.0, 11.0)
    assert planner.mpc.solution_status == 0
  assert [weights[4] for weights, _ in costs[:-1]] == [0.0] * LEAD_PRESENCE_FRAMES
  assert costs[-1][0][4] > 0.0
  assert all(weights[5] > 0.0 for weights, _ in costs)  # jerk cost still owns the transition
  assert all(np.array_equal(constraints, costs[0][1]) for _, constraints in costs)


@pytest.mark.parametrize('distance, kwargs, gap_loss, extra_braking', [
  (28.0, {'y_stall': 1.2}, 4.0, 0.6),
  (40.0, {'abort_at': 1.5}, 5.6, 0.75),
  (21.4, {'y_stall': 1.2}, 4.0, 0.4),
], ids=['stalled_near_floor', 'turn_signal_off', 'one_second_gap'])
def test_recovery_preserves_maneuver_bounds(distance, kwargs, gap_loss, extra_braking):
  # Keep the tighter of the imported source contract and this task's requested bounds.
  today = summary(LaneChangePlant(False, 12.8, 11.1, distance, **kwargs).run())
  design = summary(LaneChangePlant(True, 12.8, 11.1, distance, **kwargs).run())
  assert design['onsets'] == 1 and not design['fcw']
  assert design['min_gap'] >= today['min_gap'] - gap_loss
  assert design['min_a'] >= today['min_a'] - extra_braking
  assert design['first_release'] <= 1.0
  if 'abort_at' in kwargs:
    aborted = design['t'] >= kwargs['abort_at'] - 1e-6
    assert design['released'][~aborted][-1]
    assert not design['released'][aborted].any()


def test_real_radar_dropout_keeps_the_daily_change_cost(monkeypatch):
  planner = LongitudinalPlanner(get_cp(), init_v=12.0)
  costs = []
  original = planner.mpc.set_cost_weights

  def record(cost_weights, constraint_weights):
    costs.append(cost_weights[4])
    original(cost_weights, constraint_weights)

  monkeypatch.setattr(planner.mpc, 'set_cost_weights', record)
  for present, count in ((True, 20), (False, 20), (True, 5)):
    for _ in range(count):
      planner.update(frame(12.0, 0.0, 40.0, 11.0, state=LaneChangeState.off, blinker=False, lead_present=present))
      assert not any(planner.lane_change_gap.released)
  assert all(cost > 0.0 for cost in costs)


@pytest.mark.parametrize('restored_slot', [0, 1])
def test_restoring_a_nonbinding_lead_does_not_change_the_incumbent(restored_slot):
  restored, ordinary = LongitudinalMpc(), LongitudinalMpc()
  radar = frame(12.0, 0.0, 18.0, 0.0)['radarState']
  leads = (radar.leadOne, radar.leadTwo)
  leads[restored_slot].dRel = 80.0
  mask = tuple(i == restored_slot for i in range(2))
  for _ in range(20):
    for mpc in (restored, ordinary):
      mpc.set_cur_state(12.0, 0.0, 0.0)
    update(restored, radar, restored_leads=mask)
    update(ordinary, radar)
    assert restored.restoring_lead is None
    np.testing.assert_array_equal(restored.params, ordinary.params)
    np.testing.assert_array_equal(restored.a_solution, ordinary.a_solution)
    assert restored.crash_cnt == ordinary.crash_cnt


@pytest.mark.parametrize('interruption', ['lead_missing', 'other_slot', 'stop', 'reset'])
def test_pending_restore_cannot_leak_to_another_obstacle(monkeypatch, interruption):
  mpc = LongitudinalMpc()
  absent = frame(12.0, 0.0, 40.0, 11.0, lead_present=False)['radarState']
  radar = frame(12.0, 0.0, 40.0, 11.0)['radarState']
  for _ in range(20):
    mpc.set_cur_state(12.0, 1.0, 1.0)
    update(mpc, absent)
  update(mpc, radar, restored_leads=(True, True))
  assert mpc.restoring_lead == 0
  kwargs = {}
  if interruption == 'lead_missing':
    radar = absent
  elif interruption == 'other_slot':
    radar.leadTwo.dRel = 10.0
  elif interruption == 'stop':
    kwargs['stop_x'] = 5.0
  else:
    mpc.reset()
    assert mpc.restoring_lead is None
  costs = []
  original = mpc.set_cost_weights

  def record(cost_weights, constraint_weights):
    costs.append(cost_weights[4])
    original(cost_weights, constraint_weights)

  monkeypatch.setattr(mpc, 'set_cost_weights', record)
  mpc.set_cur_state(12.0, 1.0, 1.0)
  update(mpc, radar, **kwargs)
  assert mpc.restoring_lead is None
  assert costs[0] > 0.0


def test_a_missing_released_lead_is_not_a_restored_obstacle():
  planner = LongitudinalPlanner(get_cp(), init_v=12.0)
  released_frames(planner, n=20)
  planner.update(frame(12.0, 0.0, 40.0, 11.0, y_ego=3.5, path=0.0, lead_present=False))
  assert planner.mpc.restoring_lead is None
  assert planner.mpc.binding_obstacle == 'fake'


def test_a_released_lead_is_not_kept_as_a_dropout():
  # the MPC keeps a lead the radar loses through its absence hold; a lead the lane-change gap releases has left the plan on purpose, so
  # from its release frame on its slot is the free run and nothing is kept for it, while the identity hands over after the absence
  # hold as before. A lead restored later starts its own count again, so the keep never spans a release
  planner = LongitudinalPlanner(get_cp(), init_v=12.0)
  for _ in range(20):
    planner.update(frame(12.0, 0.0, 40.0, 11.0, state=LaneChangeState.off, blinker=False))
  assert mpc_follows(planner, 40.0, 11.0)
  frames = []
  for _ in range(10):
    planner.update(frame(12.0, 0.0, 40.0, 11.0, y_ego=1.5, path=2.0))
    frames.append((tuple(planner.lane_change_gap.released), tuple(planner.mpc.kept), planner.mpc.binding_obstacle))
  released = [k for k, (slots, _, _) in enumerate(frames) if all(slots)]
  assert released and released == list(range(released[0], len(frames))), frames
  assert not any(any(kept) for _, kept, _ in frames), frames
  assert not mpc_follows(planner, 40.0, 11.0)
  first = next(k for k, (slots, _, _) in enumerate(frames) if any(slots))
  assert [identity for _, _, identity in frames[first:first + LEAD_ABSENCE_FRAMES]] == ['lead'] * (LEAD_ABSENCE_FRAMES - 1) + ['fake'], frames
  # the change called off: the gap restores the car, which the radar then loses after one frame back. One frame is no confirmation
  planner.update(frame(12.0, 0.0, 40.0, 11.0, state=LaneChangeState.off, blinker=False))
  assert not any(planner.lane_change_gap.released) and mpc_follows(planner, 40.0, 11.0)
  planner.update(frame(12.0, 0.0, 40.0, 11.0, state=LaneChangeState.off, blinker=False, lead_present=False))
  assert not any(planner.mpc.kept) and not planner.lead_kept.present
