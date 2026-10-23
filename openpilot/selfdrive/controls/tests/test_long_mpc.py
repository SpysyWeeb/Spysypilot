import math
import numpy as np
import pytest

from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.interfaces import ACCEL_MIN
from opendbc.car.hyundai.values import CAR
from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.controls.lib.longitudinal_lead import LeadObservation, ModelLeadAnchor, LEAD_T_IDXS
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (LEAD_ABSENCE_FRAMES, LEAD_PRESENCE_FRAMES, LongitudinalMpc, STOP_DISTANCE,
                                                                            T_IDXS, get_T_FOLLOW)
from openpilot.selfdrive.modeld.constants import ModelConstants

LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource
STANDARD = log.LongitudinalPersonality.standard
AGGRESSIVE = log.LongitudinalPersonality.aggressive
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ACTION_T = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE).longitudinalActuatorDelay + DT_MDL


def radar_state(d_one=None, v_one=10.0, d_two=None, v_two=10.0):
  rs = messaging.new_message('radarState').radarState
  for lead, d, v in ((rs.leadOne, d_one, v_one), (rs.leadTwo, d_two, v_two)):
    if d is not None:
      lead.present = True
      lead.dRel = d
      lead.vLead = v
      lead.vLeadK = v
      lead.modelProb = 0.95
      lead.aLeadTau = 1.5
  return rs


def update(mpc, rs, *args, **kwargs):
  # one MPC update as the planner makes it, on its one reading of each radar slot
  mpc.update(rs, *args, usable=tuple(LeadObservation.from_radar(lead).present for lead in (rs.leadOne, rs.leadTwo)), **kwargs)


def anchor(d, v0, v1):
  v = np.linspace(v0, v1, len(LEAD_T_IDXS))
  x = d + np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2.0 * np.diff(LEAD_T_IDXS))])
  return ModelLeadAnchor(x, v, (v[1] - v[0]) / (LEAD_T_IDXS[1] - LEAD_T_IDXS[0]), v[1])


def candidate(mpc):
  # the planner's MPC candidate on the Palisade
  v = np.interp(CONTROL_N_T_IDX, T_IDXS, mpc.v_solution)
  a = np.interp(CONTROL_N_T_IDX, T_IDXS, mpc.a_solution)
  return float(get_accel_from_plan(v, a, CONTROL_N_T_IDX, action_t=ACTION_T))


def converged(mpc):
  # SQP-RTI iterated to convergence on the frame's own data
  for _ in range(40):
    mpc.run()
  return candidate(mpc)


def free_run(mpc, v, a, personality=AGGRESSIVE, frames=40):
  # another candidate owns the output: the MPC plans toward a_max on the fake lead
  for _ in range(frames):
    mpc.set_cur_state(v, a, a)
    update(mpc, radar_state(), personality)


def record_iterations(mpc):
  iterations = []
  run = mpc.run

  def counted_run(*args):
    iterations.append(args[0] if args else 1)
    run(*args)
  mpc.run = counted_run
  return iterations


def lead_after_a_free_run(mpc):
  free_run(mpc, 14.0, 0.0)
  mpc.set_cur_state(14.0, 0.0, 0.0)
  update(mpc, radar_state(d_one=60.0, v_one=0.0), AGGRESSIVE)


def stopping_at_a_column(mpc):
  for _ in range(20):
    mpc.set_cur_state(12.0, -1.5, -1.5)
    update(mpc, radar_state(), AGGRESSIVE, stop_x=45.0)


def braking_for_a_column(mpc, v0, personality=AGGRESSIVE, frames=20):
  # a commit after a free run, the car doing what the MPC asks: returns (v, a, stop_x) of the next frame
  free_run(mpc, v0, 0.0, personality)
  v, a, stop_x = v0, 0.0, v0 ** 2 / 2.0 + 5.0
  for _ in range(frames):
    mpc.set_cur_state(v, a, a)
    update(mpc, radar_state(), personality, stop_x=stop_x)
    a = max(candidate(mpc), ACCEL_MIN)
    v = max(v + a * DT_MDL, 0.0)
    stop_x -= v * DT_MDL
  return v, a, stop_x


def drive(mpc, v, told, rs, personality, stop_x=None):
  # the planner while the MPC drives: it continues its own plan within the car's limits, the car was told the clipped candidate;
  # an ideal car
  mpc.set_cur_state(v, float(np.clip(mpc.a_next, ACCEL_MIN, long_mpc.ACCEL_MAX)), told)
  update(mpc, rs, personality, stop_x=stop_x)
  a = float(np.clip(candidate(mpc), ACCEL_MIN, long_mpc.ACCEL_MAX))
  return max(v + (told + a) / 2.0 * DT_MDL, 0.0), a


@pytest.fixture(params=[2.0, 4.0])
def accel_max(request, monkeypatch):
  # the free run's iterate climbs toward ACCEL_MAX: stock opendbc's and the fork's
  monkeypatch.setattr(long_mpc, 'ACCEL_MAX', request.param)


class TestUpdateProtocol:
  def test_weights_are_set_exactly_once_per_update(self):
    mpc = LongitudinalMpc()
    calls = []
    set_weights = mpc.set_weights
    mpc.set_weights = lambda *args: (calls.append(args), set_weights(*args))
    mpc.set_cur_state(10.0, 0.0, 0.0)
    update(mpc, radar_state(d_one=20.0, v_one=8.0), STANDARD, jerk_scale=0.3, t_follow_pad=0.5)
    assert len(calls) == 1

  def test_supervisor_policy_shapes_lead0_only(self):
    mpc = LongitudinalMpc()
    weights = []
    set_weights = mpc.set_weights
    mpc.set_weights = lambda *args: (weights.append(args), set_weights(*args))
    mpc.set_cur_state(10.0, 0.0, 0.0)
    update(mpc, radar_state(d_one=20.0, v_one=8.0), STANDARD, jerk_scale=0.3, t_follow_pad=0.5)
    assert mpc.source == LongitudinalPlanSource.lead0
    assert weights[-1][2] == 0.3
    assert math.isclose(mpc.params[0, 4], get_T_FOLLOW(STANDARD) + 0.5, rel_tol=1e-6, abs_tol=1e-9)

    update(mpc, radar_state(d_one=80.0, v_one=8.0, d_two=20.0, v_two=8.0), STANDARD, jerk_scale=0.3, t_follow_pad=0.5)
    assert mpc.source == LongitudinalPlanSource.lead1
    assert weights[-1][2] == 1.0
    assert math.isclose(mpc.params[0, 4], get_T_FOLLOW(STANDARD), rel_tol=1e-6, abs_tol=1e-9)

  def test_handoff_from_an_adaptive_lead0_to_the_stop_column_reanchors_the_change_cost(self):
    mpc = LongitudinalMpc()
    mpc.set_cur_state(10.0, -1.0, -1.0)
    for _ in range(5):
      update(mpc, radar_state(d_one=15.0, v_one=6.0), STANDARD, jerk_scale=0.3)
    assert not np.allclose(mpc.params[:, 3], -1.0)
    mpc.set_cur_state(10.0, -1.0, -1.0)
    update(mpc, radar_state(d_one=15.0, v_one=6.0), STANDARD, stop_x=10.0, jerk_scale=0.3)
    assert mpc.source == LongitudinalPlanSource.stop
    assert np.allclose(mpc.params[:, 3], -1.0)

  def test_a_new_binding_obstacle_reanchors_the_change_cost(self):
    # a committed stop used to inherit the free run's accelerating solution through the change cost and start 1.5 s late
    mpc = LongitudinalMpc()
    for _ in range(20):
      mpc.set_cur_state(13.0, 0.0, 0.0)
      update(mpc, radar_state(), STANDARD)
    assert mpc.a_prev.max() > 0.5
    mpc.set_cur_state(13.0, -1.2, -1.2)
    update(mpc, radar_state(), STANDARD, stop_x=50.0)
    assert mpc.source == LongitudinalPlanSource.stop
    assert np.allclose(mpc.params[:, 3], -1.2)
    mpc.set_cur_state(13.0, -1.2, -1.2)
    update(mpc, radar_state(), STANDARD, stop_x=49.4)
    assert not np.allclose(mpc.params[:, 3], -1.2)

  def test_a_next_is_the_plan_one_frame_on(self):
    mpc = LongitudinalMpc()
    mpc.set_cur_state(12.0, -0.5, -0.5)
    update(mpc, radar_state(), STANDARD, stop_x=40.0)
    assert math.isclose(mpc.a_next, mpc.a_solution[0] + mpc.j_solution[0] * DT_MDL, abs_tol=1e-6)
    assert mpc.a_next < mpc.a_solution[0]

  def test_model_anchor_replaces_the_radar_extrapolation(self):
    mpc = LongitudinalMpc()
    mpc.set_cur_state(10.0, 0.0, 0.0)
    update(mpc, radar_state(d_one=30.0, v_one=10.0), STANDARD)
    radar_only = np.array(mpc.params[:, 2])
    update(mpc, radar_state(d_one=30.0, v_one=10.0), STANDARD, lead0_anchor=anchor(30.0, 10.0, 2.0))
    braking_forecast = np.array(mpc.params[:, 2])
    assert math.isclose(braking_forecast[0], radar_only[0], rel_tol=1e-6, abs_tol=1e-9)
    assert np.all(braking_forecast[3:] < radar_only[3:])

  def test_committed_stop_is_a_fixed_obstacle(self):
    mpc = LongitudinalMpc()
    mpc.set_cur_state(5.0, 0.0, 0.0)
    update(mpc, radar_state(), STANDARD, stop_x=10.0)
    assert mpc.source == LongitudinalPlanSource.stop
    # D3: owner keeps STOP_DISTANCE at 7 m instead of stock's 6 m -- pin the literal
    # so a revert to stock's 6.0 fails here instead of only in the tautological
    # live-imported comparison below.
    assert STOP_DISTANCE == 7.0
    assert np.allclose(mpc.params[:, 2], 10.0 + STOP_DISTANCE)
    assert np.min(mpc.a_solution) < -0.5
    assert mpc.v_solution[-1] < 1.0

  def test_fcw_counter_needs_a_confirmed_present_lead(self):
    mpc = LongitudinalMpc()
    mpc.set_cur_state(20.0, 0.0, 0.0)
    update(mpc, radar_state(d_one=6.0, v_one=0.0), STANDARD)
    assert mpc.crash_cnt == 1
    unconfirmed = radar_state(d_one=6.0, v_one=0.0)
    unconfirmed.leadOne.modelProb = 0.5
    update(mpc, unconfirmed, STANDARD)
    assert mpc.crash_cnt == 0
    ghost = radar_state(d_one=6.0, v_one=0.0)
    ghost.leadOne.present = False
    update(mpc, ghost, STANDARD)
    assert mpc.crash_cnt == 0


class TestNewBindingObstacle:
  @pytest.mark.parametrize('personality', [AGGRESSIVE, STANDARD])
  @pytest.mark.parametrize('v, a, stop_x', [(11.6, -1.26, 46.0), (11.6, -1.26, 50.0), (11.6, -1.26, 55.0),
                                            (14.0, -1.2, 50.0), (14.0, -0.5, 50.0), (14.0, 0.0, 50.0), (20.0, 0.0, 50.0)])
  def test_a_commit_after_a_free_run_starts_at_the_converged_plan(self, accel_max, personality, v, a, stop_x):
    mpc = LongitudinalMpc()
    free_run(mpc, v, a, personality)
    iterations = record_iterations(mpc)
    mpc.set_cur_state(v, a, a)
    update(mpc, radar_state(), personality, stop_x=stop_x)
    commit_iterations = list(iterations)
    first = candidate(mpc)
    assert mpc.source == LongitudinalPlanSource.stop
    assert abs(first - converged(mpc)) < 0.1
    assert np.allclose(mpc.params[:, 3], a)
    iterations.clear()
    mpc.set_cur_state(v, first, first)
    update(mpc, radar_state(), personality, stop_x=stop_x - v * DT_MDL)
    assert commit_iterations[0] > 1 and iterations == [1]

  @pytest.mark.parametrize('history, lead, stop_x', [(lambda mpc: free_run(mpc, 14.0, 0.0), None, 50.0),
                                                     (lead_after_a_free_run, 59.3, None),
                                                     (stopping_at_a_column, 22.0, 45.0)],
                           ids=['fake lead to stop column', 'fake lead to real lead', 'stop column to stopped lead'])
  def test_the_first_solve_of_a_new_obstacle_does_not_depend_on_the_previous_one(self, accel_max, history, lead, stop_x):
    handed_over, fresh = LongitudinalMpc(), LongitudinalMpc()
    history(handed_over)
    for mpc in (handed_over, fresh):
      mpc.set_cur_state(12.0, -1.0, -1.0)
      update(mpc, radar_state(d_one=lead, v_one=0.0), AGGRESSIVE, stop_x=stop_x)
    assert np.allclose(handed_over.params, fresh.params)
    # to the QP's tolerance
    assert np.allclose(handed_over.a_solution, fresh.a_solution, atol=0.01)

  @pytest.mark.parametrize('d', [60.0, 80.0])
  def test_a_real_lead_after_a_free_run_is_a_new_obstacle(self, accel_max, d):
    mpc = LongitudinalMpc()
    free_run(mpc, 14.0, 0.0)
    iterations = record_iterations(mpc)
    mpc.set_cur_state(14.0, 0.0, 0.0)
    update(mpc, radar_state(d_one=d, v_one=0.0), AGGRESSIVE)
    mpc.set_cur_state(14.0, 0.0, 0.0)
    update(mpc, radar_state(d_one=d - 0.7, v_one=0.0), AGGRESSIVE)
    assert abs(candidate(mpc) - converged(mpc)) < 0.1
    assert np.allclose(mpc.params[:, 3], 0.0)
    assert iterations[0] == 1 and iterations[1] > 1

  # a stopped car 18-40 m ahead: two SQP-RTI iterations leave an urgent hand-over short of converged, standard's the most
  @pytest.mark.parametrize('personality, tolerance', [(AGGRESSIVE, 0.11), (STANDARD, 0.145)])
  @pytest.mark.parametrize('v', [10.0, 12.0, 14.0])
  @pytest.mark.parametrize('d', [18.0, 25.0, 32.0, 40.0])
  def test_a_stopped_lead_appearing_after_a_free_run_is_braked_for_from_its_second_frame(self, accel_max, personality, tolerance, v, d):
    mpc = LongitudinalMpc()
    free_run(mpc, v, 0.0, personality)
    mpc.set_cur_state(v, 0.0, 0.0)
    update(mpc, radar_state(d_one=d, v_one=0.0), personality)
    before = max(candidate(mpc), ACCEL_MIN)
    mpc.set_cur_state(v, 0.0, 0.0)
    update(mpc, radar_state(d_one=d - v * DT_MDL, v_one=0.0), personality)
    first = max(candidate(mpc), ACCEL_MIN)
    assert first - max(converged(mpc), ACCEL_MIN) < tolerance
    assert first <= before

  # a stopped car 18-25 m ahead of a committed stop: the same limit, larger for standard
  @pytest.mark.parametrize('personality, tolerance', [(AGGRESSIVE, 0.11), (STANDARD, 0.25)])
  @pytest.mark.parametrize('v0', [10.0, 12.0, 14.0])
  @pytest.mark.parametrize('d', [18.0, 20.0, 22.0, 25.0])
  def test_a_stopped_lead_appearing_before_the_stop_point_is_braked_for_at_once(self, accel_max, personality, tolerance, v0, d):
    mpc = LongitudinalMpc()
    v, a, stop_x = braking_for_a_column(mpc, v0, personality)
    mpc.set_cur_state(v, a, a)
    update(mpc, radar_state(d_one=d, v_one=0.0), personality, stop_x=stop_x)
    first = max(candidate(mpc), ACCEL_MIN)
    assert mpc.source == LongitudinalPlanSource.lead0
    assert first - max(converged(mpc), ACCEL_MIN) < tolerance
    assert first <= a

  def test_a_commit_behind_a_lead_is_a_new_obstacle(self):
    # the column appears nearer than a lead the car was following: the committed stop starts from the current state at once
    mpc = LongitudinalMpc()
    for _ in range(40):
      mpc.set_cur_state(12.0, 0.5, 0.5)
      update(mpc, radar_state(d_one=20.0, v_one=12.0), AGGRESSIVE)
    iterations = record_iterations(mpc)
    mpc.set_cur_state(12.0, 0.5, 0.5)
    update(mpc, radar_state(d_one=20.0, v_one=12.0), AGGRESSIVE, stop_x=25.0)
    assert mpc.source == LongitudinalPlanSource.stop
    assert np.allclose(mpc.params[:, 3], 0.5)
    assert iterations[0] > 1
    assert abs(candidate(mpc) - converged(mpc)) < 0.1

  @pytest.mark.parametrize('column', [False, True], ids=['lead alone', 'lead before a column'])
  def test_a_lead_that_drops_out_is_the_same_obstacle_until_it_has_left(self, column):
    stop_x = 40.0 if column else None

    def follow(mpc, frames, lead=True):
      for _ in range(frames):
        mpc.set_cur_state(10.0, -0.5, -0.5)
        update(mpc, radar_state(d_one=25.0, v_one=8.0) if lead else radar_state(), STANDARD, stop_x=stop_x)

    mpc = LongitudinalMpc()
    follow(mpc, 20)
    assert mpc.source == LongitudinalPlanSource.lead0
    iterations = record_iterations(mpc)
    # radar drops a lead for one to a few frames far more often than it loses it (half the short absences last two frames)
    for dropout in (1, 2, 5):
      follow(mpc, dropout, lead=False)
      follow(mpc, 1)
      assert not np.allclose(mpc.params[:, 3], -0.5)
    assert iterations == [1] * 11
    iterations.clear()
    # gone for a second: it has left
    follow(mpc, 20, lead=False)
    assert sum(1 for n in iterations if n > 1) == 1

  def test_a_column_and_a_stopped_lead_in_the_same_place_are_one_obstacle(self):
    # radar noise swaps which of the two binds every frame; neither swap is a hand-over
    mpc = LongitudinalMpc()
    v, a, stop_x = braking_for_a_column(mpc, 12.0)
    iterations, sources, refills = record_iterations(mpc), [], []
    for k in range(20):
      a_prev = mpc.a_prev.copy()
      mpc.set_cur_state(v, a, a)
      update(mpc, radar_state(d_one=stop_x + STOP_DISTANCE + (-0.1 if k % 2 else 0.1), v_one=0.0), AGGRESSIVE, stop_x=stop_x)
      sources.append(mpc.source)
      refills.append(not np.allclose(mpc.params[:, 3], a_prev))
      a = max(candidate(mpc), ACCEL_MIN)
      v = max(v + a * DT_MDL, 0.0)
      stop_x -= v * DT_MDL
    assert sources[::2] == [LongitudinalPlanSource.stop] * 10 and sources[1::2] == [LongitudinalPlanSource.lead0] * 10
    assert iterations == [1] * 20 and not any(refills)

  def test_the_same_car_changing_slots_is_the_same_obstacle(self):
    same_slot, swapped = LongitudinalMpc(), LongitudinalMpc()
    for mpc in (same_slot, swapped):
      for _ in range(5):
        mpc.set_cur_state(10.0, -1.0, -1.0)
        update(mpc, radar_state(d_one=15.0, v_one=6.0), STANDARD)
      mpc.set_cur_state(10.0, -1.0, -1.0)
    iterations = record_iterations(swapped)
    update(same_slot, radar_state(d_one=15.0, v_one=6.0), STANDARD)
    update(swapped, radar_state(d_one=80.0, v_one=8.0, d_two=15.0, v_two=6.0), STANDARD)
    assert swapped.source == LongitudinalPlanSource.lead1
    assert np.allclose(swapped.params, same_slot.params)
    assert np.allclose(swapped.a_solution, same_slot.a_solution, atol=1e-6)
    assert iterations == [1]

  @pytest.mark.parametrize('personality', [AGGRESSIVE, STANDARD])
  def test_a_lead_that_has_left_hands_over_from_what_the_car_was_told(self, accel_max, personality):
    # braking behind a lead that brakes hard, then the radar loses it for good: through the absence hold the plan keeps the lead
    # where its last reading goes, and on the hold's last frame the free run takes the car on from the published target, not from
    # the lead's plan one frame on, which lies well below it while that plan climbs out of the braking
    def lead_lost(in_sight):
      mpc = LongitudinalMpc()
      v, told, x, x_lead, v_lead = 15.0, 0.0, 0.0, 25.0, 15.0
      for k in range(90):
        v_lead = max(v_lead - 3.0 * DT_MDL, 0.0) if k >= 20 else v_lead
        x_lead += v_lead * DT_MDL
        v, told = drive(mpc, v, told, radar_state(d_one=x_lead - x, v_one=v_lead), personality)
        x += v * DT_MDL
      iterations, frames = record_iterations(mpc), []
      for _ in range(2 * LEAD_ABSENCE_FRAMES):
        # in sight: the radar still reads the lead where its last reading goes on
        x_lead += v_lead * DT_MDL
        v, published = drive(mpc, v, told, radar_state(d_one=x_lead - x, v_one=v_lead) if in_sight else radar_state(), personality)
        x += v * DT_MDL
        frames.append((told, published, float(mpc.params[0, 3]), mpc.lead_handover))
        told = published
      return frames, iterations

    lost, iterations = lead_lost(False)
    kept, _ = lead_lost(True)
    last = LEAD_ABSENCE_FRAMES - 1
    assert all(abs(a[1] - b[1]) <= 0.02 for a, b in zip(lost[:last], kept[:last], strict=True)), (lost[:last], kept[:last])
    assert [k for k, n in enumerate(iterations) if n > 1] == [last], iterations
    assert math.isclose(lost[last][2], lost[last][0], abs_tol=1e-9), lost[last]
    assert [k for k, frame in enumerate(lost) if frame[3]] == [last], lost

  @pytest.mark.parametrize('personality', [AGGRESSIVE, STANDARD])
  @pytest.mark.parametrize('lead', [True, False], ids=['to a stopped lead ahead of it', 'to the free run'])
  def test_a_stop_column_easing_off_hands_over_from_what_the_car_was_told(self, accel_max, personality, lead):
    # late in a stop the column's plan eases off, so the target the car was told, read action_t ahead, lies above the plan one
    # frame on; whatever takes over publishes what a planner that had told the car the same target would
    mpc = LongitudinalMpc()
    free_run(mpc, 14.0, 0.0, personality)
    v, told, stop_x = 14.0, 0.0, 14.0 ** 2 / (2 * 1.6)
    while v > 4.0:
      v, told = drive(mpc, v, told, radar_state(), personality, stop_x)
      stop_x -= v * DT_MDL
    assert told - mpc.a_next > 0.1
    # a car stopped 1 m short of where the column holds the car, or the column released
    rs = radar_state(d_one=stop_x + STOP_DISTANCE - 1.0, v_one=0.0) if lead else radar_state()
    stop_x = stop_x if lead else None
    iterations = record_iterations(mpc)
    first = drive(mpc, v, told, rs, personality, stop_x)[1]
    fresh = LongitudinalMpc()
    fresh.set_cur_state(v, told, told)
    update(fresh, rs, personality, stop_x=stop_x)
    assert iterations[0] > 1
    assert abs(first - np.clip(candidate(fresh), ACCEL_MIN, long_mpc.ACCEL_MAX)) < 0.01

  def test_what_the_car_was_told_starts_a_new_obstacle_only(self):
    told_less, told_more = LongitudinalMpc(), LongitudinalMpc()
    for mpc in (told_less, told_more):
      for _ in range(20):
        mpc.set_cur_state(10.0, -1.0, -1.0)
        update(mpc, radar_state(d_one=15.0, v_one=6.0), STANDARD)
    # the same obstacle continues its plan from the seed
    for mpc, told in ((told_less, -1.5), (told_more, -0.5)):
      mpc.set_cur_state(10.0, -1.0, told)
      update(mpc, radar_state(d_one=15.0, v_one=6.0), STANDARD)
    assert np.allclose(told_less.params, told_more.params)
    assert np.allclose(told_less.a_solution, told_more.a_solution, atol=1e-9)
    # a new obstacle, the stop column nearer than the lead, publishes what a planner that had told the car the same target would
    firsts = []
    for mpc, told in ((told_less, -1.5), (told_more, -0.5)):
      fresh = LongitudinalMpc()
      fresh.set_cur_state(10.0, told, told)
      mpc.set_cur_state(10.0, -1.0, told)
      for m in (mpc, fresh):
        update(m, radar_state(d_one=15.0, v_one=6.0), STANDARD, stop_x=8.0)
      assert mpc.source == LongitudinalPlanSource.stop
      assert abs(candidate(mpc) - candidate(fresh)) < 0.01
      firsts.append(candidate(mpc))
    assert firsts[0] < firsts[1]


def one_frame_lead(history, lead):
  # a car driven by an MPC that sees lead for one frame, and an MPC that never sees it fed the same car: each frame after the flash,
  # the two QP obstacles, the column's obstacle if one is committed, and how far the plans lie apart
  seen, unseen = LongitudinalMpc(), LongitudinalMpc()
  for mpc in (seen, unseen):
    v, told, stop_x = history(mpc)
  frames = []
  for k in range(LEAD_ABSENCE_FRAMES + 2):
    seed = float(np.clip(seen.a_next, ACCEL_MIN, long_mpc.ACCEL_MAX))
    for mpc in (seen, unseen):
      mpc.set_cur_state(v, seed, told)
      update(mpc, lead if mpc is seen and k == 0 else radar_state(), STANDARD, stop_x=stop_x)
    if k > 0:
      frames.append((np.array(seen.params[:, 2]), np.array(unseen.params[:, 2]), None if stop_x is None else stop_x + STOP_DISTANCE,
                     candidate(seen) - candidate(unseen)))
    told = float(np.clip(candidate(seen), ACCEL_MIN, long_mpc.ACCEL_MAX))
    v = max(v + told * DT_MDL, 0.0)
    stop_x = None if stop_x is None else stop_x - v * DT_MDL
  return frames


class TestLeadKeep:
  # the radar loses a lead for a frame or a few far more often than the lead leaves: the MPC keeps a lead it confirmed where its last
  # trajectory goes, less the car's travel, through the absence hold, and the plan's identity reads it as the lead meanwhile
  def test_a_one_frame_lead_against_the_stop_column_is_not_held(self):
    # braking for a committed stop 40 m ahead, a standing car 25 m ahead for one frame. It is not kept: from the next frame the QP
    # plans against the column, and while the identity waits out the absence hold the flash's pull on the plan only fades
    frames = one_frame_lead(lambda mpc: braking_for_a_column(mpc, 10.0, STANDARD, frames=30), radar_state(d_one=25.0, v_one=0.0))
    for seen, unseen, column, _ in frames:
      assert np.allclose(seen, unseen, atol=1e-9) and np.allclose(seen, column, atol=1e-9), (seen[0], unseen[0], column)
    pulls = [abs(pull) for _, _, _, pull in frames[:LEAD_ABSENCE_FRAMES - 1]]
    assert all(b < a for a, b in zip(pulls, pulls[1:], strict=False)), np.round(pulls, 4)

  def test_a_one_frame_lead_in_free_run_is_not_held(self):
    def cruising(mpc):
      free_run(mpc, 15.0, 0.0, STANDARD)
      return 15.0, 0.0, None

    frames = one_frame_lead(cruising, radar_state(d_one=25.0, v_one=10.0))
    for seen, unseen, _, _ in frames:
      assert np.allclose(seen, unseen, atol=1e-9), (seen[0], unseen[0])

  def test_a_kept_lead_hands_over_on_its_own_seventh_absent_frame(self):
    # following leadOne, a nearer car in leadTwo for three frames; then leadTwo is missing and, three frames later, leadOne too. The
    # nearer car's keep binds while it lasts and leadOne's keep after it: the plan hands over to the free run once, on leadOne's own
    # last absent frame of the hold
    mpc = LongitudinalMpc()

    def step(rs):
      a = candidate(mpc) if mpc.v_solution.any() else 0.0
      mpc.set_cur_state(12.0, a, a)
      update(mpc, rs, STANDARD)
      return mpc.source

    for _ in range(60):
      step(radar_state(d_one=40.0, v_one=10.0))
    for _ in range(LEAD_PRESENCE_FRAMES + 1):
      step(radar_state(d_one=40.0, v_one=10.0, d_two=30.0, v_two=10.0))
    gone = 3
    sources, handed_over, handovers = [], [], []
    for k in range(2 * LEAD_ABSENCE_FRAMES + 2):
      before = mpc.binding_obstacle
      sources.append(step(radar_state(d_one=40.0 - 2.0 * DT_MDL * k, v_one=10.0) if k < gone else radar_state()))
      if before == 'lead' and mpc.binding_obstacle == 'fake':
        handed_over.append(k)
      if mpc.lead_handover:
        handovers.append(k)
    assert sources[:LEAD_ABSENCE_FRAMES - 1] == [LongitudinalPlanSource.lead1] * (LEAD_ABSENCE_FRAMES - 1), sources
    assert handed_over == handovers == [gone + LEAD_ABSENCE_FRAMES - 1], (handed_over, handovers)

  def test_fcw_ignores_a_kept_lead(self):
    # closing at 20 m/s on a standing car, the radar drops it but its last reading stays in the message: the plan keeps braking for
    # the kept lead as it comes closer, and FCW, which counts only a lead the radar reports, does not count it
    mpc = LongitudinalMpc()
    counts, obstacles = [], []
    for k in range(LEAD_ABSENCE_FRAMES + 1):
      rs = radar_state(d_one=12.0 - 20.0 * DT_MDL * k, v_one=0.0)
      rs.leadOne.present = k < LEAD_PRESENCE_FRAMES
      mpc.set_cur_state(20.0, 0.0, 0.0)
      update(mpc, rs, STANDARD)
      counts.append(mpc.crash_cnt)
      obstacles.append(float(mpc.params[0, 2]))
    kept = slice(LEAD_PRESENCE_FRAMES, LEAD_PRESENCE_FRAMES + LEAD_ABSENCE_FRAMES - 1)
    assert counts[:LEAD_PRESENCE_FRAMES] == list(range(1, LEAD_PRESENCE_FRAMES + 1)), counts
    assert np.all(np.diff(obstacles[LEAD_PRESENCE_FRAMES - 1:kept.stop]) < 0.0), obstacles
    assert counts[kept] == [0] * (LEAD_ABSENCE_FRAMES - 1), counts

  @pytest.mark.parametrize('seen', [1, LEAD_PRESENCE_FRAMES], ids=['unconfirmed', 'confirmed'])
  def test_a_lead_the_planner_cannot_use_is_a_missing_lead(self, seen):
    # closing at 20 m/s on a car standing 12 m ahead and another 8 m beyond it, both reported for `seen` frames; then the message still
    # holds them on frames the planner cannot use: the MPC plans, names its obstacle, hands over and counts FCW exactly as it does with
    # both cars missing from the message
    unusable, missing, frames = LongitudinalMpc(), LongitudinalMpc(), []
    for k in range(seen + LEAD_ABSENCE_FRAMES + 1):
      rs = radar_state(d_one=12.0 - 20.0 * DT_MDL * k, v_one=0.0, d_two=20.0 - 20.0 * DT_MDL * k, v_two=0.0)
      unusable.set_cur_state(20.0, 0.0, 0.0)
      unusable.update(rs, STANDARD, usable=(k < seen, k < seen))
      missing.set_cur_state(20.0, 0.0, 0.0)
      update(missing, rs if k < seen else radar_state(), STANDARD)
      frames.append([(np.array(m.params), m.source, m.binding_obstacle, m.lead_handover, m.crash_cnt, tuple(m.kept)) for m in (unusable, missing)])
    for k, (u, m) in enumerate(frames):
      assert np.allclose(u[0], m[0], rtol=0.0, atol=1e-9) and u[1:] == m[1:], (k, u[1:], m[1:])

  def test_a_lead_back_within_the_hold_is_the_same_lead(self):
    # braking behind a lead that slows to a crawl, the radar loses it for three frames and finds it again where it went: nothing
    # changed hands, so the plan goes on from its own state -- no second solve, no change cost re-anchored at the told target, no step
    mpc = LongitudinalMpc()
    v, told, x, x_lead, v_lead = 12.0, 0.0, 0.0, 25.0, 12.0
    lost, frames = range(60, 63), []
    for k in range(80):
      a_lead = -1.5 if k >= 20 and v_lead > 5.0 else 0.0
      v_lead = max(v_lead + a_lead * DT_MDL, 5.0)
      x_lead += v_lead * DT_MDL
      rs = radar_state(d_one=x_lead - x, v_one=v_lead)
      rs.leadOne.aLeadK = a_lead
      if k == lost.start - 1:
        iterations = record_iterations(mpc)
      v, published = drive(mpc, v, told, radar_state() if k in lost else rs, STANDARD)
      x += v * DT_MDL
      if k >= lost.start - 1:
        frames.append((published - told, mpc.binding_obstacle, np.allclose(mpc.params[:, 3], told)))
      told = published
    assert all(identity == 'lead' for _, identity, _ in frames) and iterations == [1] * len(frames), (frames, iterations)
    assert not any(anchored for _, _, anchored in frames[1:]), frames
    assert max(abs(step) for step, _, _ in frames) <= 0.08, frames

  def test_a_kept_lead_goes_on_along_its_last_trajectory(self):
    # a lead holding 10 m/s 30 m ahead is lost: frame k plans against its last trajectory k frames on, less the car's travel since
    mpc = LongitudinalMpc()
    for _ in range(40):
      mpc.set_cur_state(12.0, 0.0, 0.0)
      update(mpc, radar_state(d_one=30.0, v_one=10.0), STANDARD)
    last, travel = np.array(mpc.params[:, 2]), 0.0
    for k in range(1, LEAD_ABSENCE_FRAMES):
      v = 12.0 - 0.2 * k
      travel += v * DT_MDL
      mpc.set_cur_state(v, 0.0, 0.0)
      update(mpc, radar_state(), STANDARD)
      assert mpc.source == LongitudinalPlanSource.lead0
      assert np.allclose(mpc.params[:, 2], np.interp(T_IDXS + k * DT_MDL, T_IDXS, last) - travel, rtol=0.0, atol=1e-6), k
