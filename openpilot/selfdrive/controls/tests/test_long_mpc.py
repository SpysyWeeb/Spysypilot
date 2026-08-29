import math
import numpy as np

from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.controls.lib.longitudinal_lead import ModelLeadAnchor, LEAD_T_IDXS
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, STOP_DISTANCE, get_T_FOLLOW

LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource
STANDARD = log.LongitudinalPersonality.standard


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


def anchor(d, v0, v1):
  v = np.linspace(v0, v1, len(LEAD_T_IDXS))
  x = d + np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2.0 * np.diff(LEAD_T_IDXS))])
  return ModelLeadAnchor(x, v, (v[1] - v[0]) / (LEAD_T_IDXS[1] - LEAD_T_IDXS[0]), v[1])


class TestUpdateProtocol:
  def test_weights_are_set_exactly_once_per_update(self):
    mpc = LongitudinalMpc()
    calls = []
    set_weights = mpc.set_weights
    mpc.set_weights = lambda *args: (calls.append(args), set_weights(*args))
    mpc.set_cur_state(10.0, 0.0)
    mpc.update(radar_state(d_one=20.0, v_one=8.0), STANDARD, jerk_scale=0.3, t_follow_pad=0.5)
    assert len(calls) == 1

  def test_supervisor_policy_shapes_lead0_only(self):
    mpc = LongitudinalMpc()
    weights = []
    set_weights = mpc.set_weights
    mpc.set_weights = lambda *args: (weights.append(args), set_weights(*args))
    mpc.set_cur_state(10.0, 0.0)
    mpc.update(radar_state(d_one=20.0, v_one=8.0), STANDARD, jerk_scale=0.3, t_follow_pad=0.5)
    assert mpc.source == LongitudinalPlanSource.lead0
    assert mpc.lead0_policy_active
    assert weights[-1][2] == 0.3
    assert math.isclose(mpc.params[0, 4], get_T_FOLLOW(STANDARD) + 0.5, rel_tol=1e-6, abs_tol=1e-9)

    mpc.update(radar_state(d_one=80.0, v_one=8.0, d_two=20.0, v_two=8.0), STANDARD, jerk_scale=0.3, t_follow_pad=0.5)
    assert mpc.source == LongitudinalPlanSource.lead1
    assert not mpc.lead0_policy_active
    assert weights[-1][2] == 1.0
    assert math.isclose(mpc.params[0, 4], get_T_FOLLOW(STANDARD), rel_tol=1e-6, abs_tol=1e-9)

  def test_handoff_from_an_adaptive_lead0_reanchors_the_change_cost(self):
    mpc = LongitudinalMpc()
    mpc.set_cur_state(10.0, -1.0)
    for _ in range(5):
      mpc.update(radar_state(d_one=15.0, v_one=6.0), STANDARD, jerk_scale=0.3)
    assert mpc.lead0_policy_adaptive
    assert not np.allclose(mpc.params[:, 3], -1.0)
    mpc.set_cur_state(10.0, -1.0)
    mpc.update(radar_state(d_one=80.0, v_one=8.0, d_two=15.0, v_two=6.0), STANDARD, jerk_scale=0.3)
    assert not mpc.lead0_policy_adaptive
    assert np.allclose(mpc.params[:, 3], -1.0)

  def test_model_anchor_replaces_the_radar_extrapolation(self):
    mpc = LongitudinalMpc()
    mpc.set_cur_state(10.0, 0.0)
    mpc.update(radar_state(d_one=30.0, v_one=10.0), STANDARD)
    radar_only = np.array(mpc.params[:, 2])
    mpc.update(radar_state(d_one=30.0, v_one=10.0), STANDARD, lead0_anchor=anchor(30.0, 10.0, 2.0))
    braking_forecast = np.array(mpc.params[:, 2])
    assert math.isclose(braking_forecast[0], radar_only[0], rel_tol=1e-6, abs_tol=1e-9)
    assert np.all(braking_forecast[3:] < radar_only[3:])

  def test_committed_stop_is_a_fixed_obstacle(self):
    mpc = LongitudinalMpc()
    mpc.set_cur_state(5.0, 0.0)
    mpc.update(radar_state(), STANDARD, stop_x=10.0)
    assert mpc.source == LongitudinalPlanSource.stop
    assert np.allclose(mpc.params[:, 2], 10.0 + STOP_DISTANCE)
    assert np.min(mpc.a_solution) < -0.5
    assert mpc.v_solution[-1] < 1.0

  def test_fcw_counter_needs_a_confirmed_present_lead(self):
    mpc = LongitudinalMpc()
    mpc.set_cur_state(20.0, 0.0)
    mpc.update(radar_state(d_one=6.0, v_one=0.0), STANDARD)
    assert mpc.crash_cnt == 1
    unconfirmed = radar_state(d_one=6.0, v_one=0.0)
    unconfirmed.leadOne.modelProb = 0.5
    mpc.update(unconfirmed, STANDARD)
    assert mpc.crash_cnt == 0
    ghost = radar_state(d_one=6.0, v_one=0.0)
    ghost.leadOne.present = False
    mpc.update(ghost, STANDARD)
    assert mpc.crash_cnt == 0
