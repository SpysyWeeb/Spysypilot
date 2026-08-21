from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch

import numpy as np
from opendbc.car.interfaces import ACCEL_MAX
from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.blotv2 import BLOTV2_ACCEL_MAX, BLOTV2_ACCEL_REQUEST_MAX, LongitudinalPolicy

from openpilot.selfdrive.controls.lib import longitudinal_planner
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LEAD_THREE_CONFIRM,
  LEAD_T_IDXS_MODEL,
  T_IDXS,
  LongitudinalMpc,
  LongitudinalPlanSource,
  get_T_FOLLOW,
  get_safe_obstacle_distance,
  get_stopped_equivalence_factor,
)
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  get_cruise_accel,
  get_cruise_comfort_accel,
  get_lead_jerk_scale,
  get_max_accel,
  get_requested_max_accel,
  ordinary_cruise_comfort_enabled,
)


def radar_lead(**overrides):
  values = {
    "present": True,
    "dRel": 20.0,
    "vLead": 8.0,
    "aLeadK": 0.0,
    "aLeadTau": 1.5,
    "modelProb": 1.0,
  }
  values.update(overrides)
  return SimpleNamespace(**values)


def model_lead(**overrides):
  values = {
    "prob": 1.0,
    "probTime": 0.0,
    "t": LEAD_T_IDXS_MODEL,
    "x": 8.0 * LEAD_T_IDXS_MODEL,
    "xStd": np.full_like(LEAD_T_IDXS_MODEL, 0.5),
    "y": np.zeros_like(LEAD_T_IDXS_MODEL),
    "yStd": np.full_like(LEAD_T_IDXS_MODEL, 0.2),
    "v": np.full_like(LEAD_T_IDXS_MODEL, 8.0),
    "vStd": np.full_like(LEAD_T_IDXS_MODEL, 0.5),
  }
  values.update(overrides)
  return SimpleNamespace(**values)


class TestLaunchContinuity(unittest.TestCase):
  def test_standstill_launch_keeps_previous_accel_constraint(self):
    class SubMaster(dict):
      def __init__(self):
        lead = radar_lead(present=False)
        super().__init__(
          carState=SimpleNamespace(vEgo=0.0, vCruise=50.0, aEgo=0.0, standstill=True,
                                   gasPressed=False, brakePressed=False, leftBlinker=False, rightBlinker=False,
                                   steeringAngleDeg=0.0),
          carControl=SimpleNamespace(orientationNED=[], latActive=False),
          controlsState=SimpleNamespace(forceDecel=False, longControlState=longitudinal_planner.LongCtrlState.pid),
          selfdriveState=SimpleNamespace(enabled=True, experimentalMode=True, personality=1),
          vehicleParameters=SimpleNamespace(angleOffsetDeg=0.0, roll=0.0),
          radarState=SimpleNamespace(leadOne=lead, leadTwo=lead),
          modelV2=SimpleNamespace(
            position=SimpleNamespace(x=np.full(longitudinal_planner.ModelConstants.IDX_N, 50.0)),
            velocity=SimpleNamespace(x=np.full(longitudinal_planner.ModelConstants.IDX_N, 5.0)),
            acceleration=SimpleNamespace(x=np.zeros(longitudinal_planner.ModelConstants.IDX_N)),
            meta=SimpleNamespace(disengagePredictions=SimpleNamespace(gasPressProbs=[]),
                                 laneChangeState=longitudinal_planner.log.LaneChangeState.off),
            action=SimpleNamespace(desiredAcceleration=1.0, shouldStop=False),
            leadsV3=[],
          ),
        )
        self.valid = dict.fromkeys(self, True)
        self.alive = dict.fromkeys(self, True)
        self.freq_ok = dict.fromkeys(self, True)
        self.logMonoTime = {"modelV2": 1}

      def all_checks(self, services=None):
        return True

    planner = longitudinal_planner.LongitudinalPlanner(
      SimpleNamespace(openpilotLongitudinalControl=True, longitudinalActuatorDelay=0.2,
                      steerRatio=15.0, wheelbase=2.9),
    )
    planner.force_stops.update = lambda sm: math.inf
    planner.blotv2.update = lambda *args: SimpleNamespace(jerk_scale=1.0, t_follow=1.45, emergency=False)
    planner.lead_departure.update = lambda **kwargs: False
    constraints = []
    planner.mpc.set_weights = lambda prev_accel_constraint, **kwargs: constraints.append(prev_accel_constraint)
    planner.mpc.update = lambda *args, **kwargs: None

    planner.update(SubMaster())

    self.assertEqual(constraints, [True])


class TestModelLeadTrajectory(unittest.TestCase):
  def setUp(self):
    self.mpc = LongitudinalMpc()
    self.mpc.set_cur_state(10.0, 0.0)

  def test_valid_trajectory_is_anchored_to_radar(self):
    trajectory = self.mpc.process_lead_model(model_lead(), radar_lead())
    self.assertAlmostEqual(trajectory[0, 0], 20.0)
    self.assertAlmostEqual(trajectory[0, 1], 8.0)
    np.testing.assert_allclose(trajectory[:, 0], 20.0 + 8.0 * T_IDXS)

  def test_model_contributes_future_shape_not_absolute_offset(self):
    model = model_lead(
      x=100.0 + 8.0 * LEAD_T_IDXS_MODEL,
      v=30.0 + LEAD_T_IDXS_MODEL,
    )
    trajectory = self.mpc.process_lead_model(model, radar_lead())
    self.assertAlmostEqual(trajectory[0, 0], 20.0)
    self.assertAlmostEqual(trajectory[0, 1], 8.0)
    self.assertGreater(trajectory[-1, 1], trajectory[0, 1])

  def test_invalid_model_shape_uses_exact_stock_radar_fallback(self):
    radar = radar_lead(aLeadK=-1.0)
    expected = self.mpc.process_lead(radar)
    actual = self.mpc.process_lead_model(model_lead(x=[0.0], v=[0.0]), radar)
    np.testing.assert_allclose(actual, expected)

  def test_nonfinite_model_uses_exact_stock_radar_fallback(self):
    radar = radar_lead(aLeadK=-1.0)
    expected = self.mpc.process_lead(radar)
    x = np.array(model_lead().x)
    x[2] = np.nan
    actual = self.mpc.process_lead_model(model_lead(x=x), radar)
    np.testing.assert_allclose(actual, expected)

  def test_missing_radar_cannot_activate_model_trajectory(self):
    radar = radar_lead(present=False)
    expected = self.mpc.process_lead(radar)
    actual = self.mpc.process_lead_model(model_lead(), radar)
    np.testing.assert_allclose(actual, expected)

  def test_third_model_lead_only_promotes_when_future_y_enters_path(self):
    radar_state = SimpleNamespace(
      leadOne=radar_lead(dRel=45.0, vLead=12.0),
      leadTwo=radar_lead(present=False),
    )
    lead_0 = model_lead(x=12.0 * LEAD_T_IDXS_MODEL, v=np.full_like(LEAD_T_IDXS_MODEL, 12.0))
    lead_1 = model_lead(prob=0.0)
    adjacent = model_lead(probTime=4.0, x=20.0 + 6.0 * LEAD_T_IDXS_MODEL, y=np.full_like(LEAD_T_IDXS_MODEL, 3.0),
                          v=np.full_like(LEAD_T_IDXS_MODEL, 6.0))
    entering = model_lead(probTime=4.0, x=20.0 + 6.0 * LEAD_T_IDXS_MODEL, y=np.array([3.0, 2.5, 0.4, 0.2, 0.1, 0.0]),
                         v=np.full_like(LEAD_T_IDXS_MODEL, 6.0))
    model_position = SimpleNamespace(x=np.linspace(0.0, 200.0, 33), y=np.zeros(33))

    adjacent_xv = self.mpc.process_third_model_lead(adjacent, model_position)
    entering_xv = self.mpc.process_third_model_lead(entering, model_position)
    curved_position = SimpleNamespace(x=model_position.x, y=0.01 * np.asarray(model_position.x))
    curved_y = np.interp(entering.x, curved_position.x, curved_position.y) + np.array([3.0, 2.5, 0.4, 0.2, 0.1, 0.0])

    self.assertIsNone(adjacent_xv)
    self.assertIsNotNone(entering_xv)
    self.assertIsNotNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"y": curved_y})), curved_position))
    assert entering_xv is not None
    self.assertTrue(np.all(entering_xv[T_IDXS < 4.0, 0] == 1e8))
    entry_idx = np.flatnonzero(T_IDXS >= 4.0)[0]
    self.assertAlmostEqual(entering_xv[entry_idx, 0], 20.0 + 6.0 * T_IDXS[entry_idx] - 1.52)

    crossing_between_samples = model_lead(**(vars(entering) | {"y": np.array([3.0, 1.0, 0.0, 0.0, 0.0, 0.0])}))
    crossing_xv = self.mpc.process_third_model_lead(crossing_between_samples, model_position)
    assert crossing_xv is not None
    crossing_time = 3.0
    self.assertTrue(np.all(crossing_xv[T_IDXS < crossing_time, 0] == 1e8))
    self.assertTrue(np.all(crossing_xv[T_IDXS >= crossing_time, 0] < 1e8))

    centerline_crossing = model_lead(**(vars(entering) | {"y": np.array([3.0, 1.0, -0.4, -0.2, -0.1, 0.0])}))
    centerline_xv = self.mpc.process_third_model_lead(centerline_crossing, model_position)
    assert centerline_xv is not None
    centerline_entry = 2.0 + (1.0 - 0.5) / (1.0 + 0.4) * 2.0
    self.assertTrue(np.all(centerline_xv[T_IDXS < centerline_entry, 0] == 1e8))
    self.assertTrue(np.all(centerline_xv[T_IDXS >= centerline_entry, 0] < 1e8))

    existing_obstacle = 45.0 + 12.0 * T_IDXS + 12.0 ** 2 / 5.0
    for _ in range(4):
      self.mpc.update(radar_state, model_leads=[lead_0, lead_1, entering], model_position=model_position)
      np.testing.assert_allclose(self.mpc.params[:, 2], existing_obstacle)
    tracked = entering
    for frame in range(5):
      age = frame * self.mpc.dt
      future_t = LEAD_T_IDXS_MODEL + age
      tracked = model_lead(**(vars(entering) | {
        "x": 20.0 + 6.0 * future_t - 10.0 * age,
        "y": np.interp(np.minimum(future_t, LEAD_T_IDXS_MODEL[-1]), LEAD_T_IDXS_MODEL, entering.y),
      }))
      self.mpc.update(radar_state, model_leads=[lead_0, lead_1, tracked], model_position=model_position, allow_third_lead=True)
      if frame < 4:
        np.testing.assert_allclose(self.mpc.params[:, 2], existing_obstacle)
    tracked_xv = self.mpc.process_third_model_lead(tracked, model_position, require_outside=False)
    assert tracked_xv is not None
    expected_obstacle = tracked_xv[:, 0] + tracked_xv[:, 1] ** 2 / 5.0
    np.testing.assert_allclose(self.mpc.params[:, 2], np.minimum(existing_obstacle, expected_obstacle))
    self.assertEqual(self.mpc.source, LongitudinalPlanSource.lead2)

    self.mpc.update(radar_state, model_leads=[lead_0, lead_1, entering], model_position=model_position, allow_third_lead=False)
    self.assertEqual(self.mpc.lead_three_confirm, 0.0)
    np.testing.assert_allclose(self.mpc.params[:, 2], existing_obstacle)

    alternate = model_lead(**(vars(entering) | {"x": np.asarray(entering.x) + 10.0}))
    for lead in (entering, alternate) * 3:
      self.mpc.update(radar_state, model_leads=[lead_0, lead_1, lead], model_position=model_position, allow_third_lead=True)
      self.assertLess(self.mpc.lead_three_confirm, LEAD_THREE_CONFIRM)
      np.testing.assert_allclose(self.mpc.params[:, 2], existing_obstacle)

    self.mpc.update(radar_state, model_leads=[lead_0, lead_1, entering], model_position=model_position, allow_third_lead=False)
    self.mpc.set_cur_state(31.0, 0.0)
    fast_closing = entering
    for frame in range(5):
      moved = model_lead(**(vars(fast_closing) | {"x": np.asarray(fast_closing.x) - 1.25 * frame}))
      self.mpc.update(radar_state, model_leads=[lead_0, lead_1, moved], model_position=model_position, allow_third_lead=True)
    self.assertGreaterEqual(self.mpc.lead_three_confirm, LEAD_THREE_CONFIRM)

    self.mpc.update(radar_state, model_leads=[lead_0, lead_1, entering], model_position=model_position, allow_third_lead=False)
    changed_entry = model_lead(**(vars(entering) | {"y": np.array([3.0, 0.4, 0.3, 0.2, 0.1, 0.0])}))
    for lead in (entering, entering, changed_entry, changed_entry, changed_entry):
      self.mpc.update(radar_state, model_leads=[lead_0, lead_1, lead], model_position=model_position, allow_third_lead=True)
    self.assertLess(self.mpc.lead_three_confirm, LEAD_THREE_CONFIRM)

    self.mpc.update(radar_state, model_leads=[lead_0, lead_1, entering], model_position=model_position, allow_third_lead=False)
    shifted_path = SimpleNamespace(x=model_position.x, y=np.full_like(model_position.x, 0.75))
    for path in (model_position, model_position, shifted_path, shifted_path, shifted_path):
      self.mpc.update(radar_state, model_leads=[lead_0, lead_1, entering], model_position=path, allow_third_lead=True)
    self.assertLess(self.mpc.lead_three_confirm, LEAD_THREE_CONFIRM)

    self.mpc.source = LongitudinalPlanSource.lead2
    self.mpc.reset()
    self.assertEqual(self.mpc.source, LongitudinalPlanSource.cruise)
    self.assertEqual(self.mpc.lead_three_confirm, 0.0)
    self.assertIsNone(self.mpc.lead_three_signature)
    self.assertEqual(self.mpc.lead_three_ego_distance, 0.0)
    self.assertEqual(self.mpc.lead_three_ego_speed, 0.0)

  def test_third_model_lead_rejects_uncertain_or_malformed_path(self):
    model_position = SimpleNamespace(x=np.linspace(0.0, 200.0, 33), y=np.zeros(33))
    entering = model_lead(probTime=4.0, x=20.0 + 6.0 * LEAD_T_IDXS_MODEL, y=np.array([3.0, 2.5, 0.4, 0.2, 0.1, 0.0]),
                         v=np.full_like(LEAD_T_IDXS_MODEL, 6.0))

    self.assertIsNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"yStd": np.full(6, 3.0)})), model_position))
    self.assertIsNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"xStd": np.full(6, 1e6)})), model_position))
    self.assertIsNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"vStd": np.full(6, 1e6)})), model_position))
    self.assertIsNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"xStd": np.full(6, np.nan)})), model_position))
    self.assertIsNone(self.mpc.process_third_model_lead(SimpleNamespace(**{k: v for k, v in vars(entering).items() if k != "xStd"}), model_position))
    self.assertIsNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"vStd": -np.ones(6)})), model_position))
    self.assertIsNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"yStd": -np.ones(6)})), model_position))
    self.assertIsNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"t": np.linspace(0.0, 0.5, 6)})), model_position))
    near_t = np.array(LEAD_T_IDXS_MODEL)
    near_t[2] += 1e-6
    self.assertIsNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"t": near_t})), model_position))
    self.assertIsNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"prob": np.inf})), model_position))
    self.assertIsNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"probTime": 3.9})), model_position))
    self.assertIsNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"probTime": 4.1})), model_position))
    self.assertIsNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"v": np.full(6, 80.0)})), model_position))
    self.assertIsNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"y": [3.0]})), model_position))
    short_path = SimpleNamespace(x=np.linspace(0.0, 45.0, 33), y=np.zeros(33))
    self.assertIsNone(self.mpc.process_third_model_lead(entering, short_path))
    bad_y = np.array(entering.y)
    bad_y[2] = np.nan
    self.assertIsNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"y": bad_y})), model_position))
    decreasing_x = np.array(entering.x)
    decreasing_x[2] = decreasing_x[1] - 1.0
    self.assertIsNone(self.mpc.process_third_model_lead(model_lead(**(vars(entering) | {"x": decreasing_x})), model_position))

  def test_third_model_lead_continuity_tracks_evolving_trajectory(self):
    radar_state = SimpleNamespace(leadOne=radar_lead(present=False), leadTwo=radar_lead(present=False))
    model_position = SimpleNamespace(x=np.linspace(0.0, 300.0, 33), y=np.zeros(33))
    lead_0 = model_lead(prob=0.0)
    lead_1 = model_lead(prob=0.0)
    self.mpc.set_cur_state(10.0, 0.0)

    y_shape = np.array([3.15, 1.15, 0.4, 0.2, 0.1, 0.0])
    accelerating = model_lead()
    for frame in range(12):
      age = frame * self.mpc.dt
      absolute_t = LEAD_T_IDXS_MODEL + age
      accelerating = model_lead(
        probTime=4.0,
        x=20.0 + 6.0 * absolute_t + 0.5 * absolute_t ** 2 - 10.0 * age,
        y=np.interp(np.minimum(absolute_t, LEAD_T_IDXS_MODEL[-1]), LEAD_T_IDXS_MODEL, y_shape),
        v=6.0 + absolute_t,
      )
      self.mpc.update(radar_state, model_leads=[lead_0, lead_1, accelerating], model_position=model_position, allow_third_lead=True)
    self.assertGreaterEqual(self.mpc.lead_three_confirm, LEAD_THREE_CONFIRM)

    self.mpc.update(radar_state, model_leads=[lead_0, lead_1, accelerating], model_position=model_position, allow_third_lead=False)
    candidate_a = model_lead(probTime=4.0, x=20.0 + 6.0 * LEAD_T_IDXS_MODEL,
                             y=np.array([3.0, 2.6, 0.49, 0.2, 0.1, 0.0]), v=np.full(6, 6.0))
    candidate_b = model_lead(probTime=4.0, x=20.8 + 6.0 * LEAD_T_IDXS_MODEL,
                             y=np.array([3.0, 0.49, 0.2, 0.1, 0.0, 0.0]), v=np.full(6, 6.0))
    for lead in (candidate_a, candidate_b, candidate_a, candidate_b, candidate_a):
      self.mpc.update(radar_state, model_leads=[lead_0, lead_1, lead], model_position=model_position, allow_third_lead=True)
    self.assertLess(self.mpc.lead_three_confirm, LEAD_THREE_CONFIRM)
    inside_b = model_lead(probTime=4.0, x=21.25 + 6.0 * LEAD_T_IDXS_MODEL,
                          y=np.array([0.4, 0.3, 0.2, 0.1, 0.0, 0.0]), v=np.full(6, 6.0))
    for _ in range(5):
      self.mpc.update(radar_state, model_leads=[lead_0, lead_1, inside_b], model_position=model_position, allow_third_lead=True)
    self.assertEqual(self.mpc.lead_three_confirm, 0.0)
    self.assertNotEqual(self.mpc.source, LongitudinalPlanSource.lead2)

    self.mpc.update(radar_state, model_leads=[lead_0, lead_1, candidate_a], model_position=model_position, allow_third_lead=False)
    approaching = candidate_a
    for frame in range(12):
      age = frame * self.mpc.dt
      future_t = LEAD_T_IDXS_MODEL + age
      approaching = model_lead(
        probTime=4.0,
        x=20.0 + 6.0 * future_t - 10.0 * age,
        y=np.interp(np.minimum(future_t, LEAD_T_IDXS_MODEL[-1]), LEAD_T_IDXS_MODEL, y_shape),
        v=np.full(6, 6.0),
      )
      self.mpc.update(radar_state, model_leads=[lead_0, lead_1, approaching], model_position=model_position, allow_third_lead=True)
    self.assertGreater(self.mpc.lead_three_confirm, LEAD_THREE_CONFIRM)
    self.assertEqual(self.mpc.source, LongitudinalPlanSource.lead2)

    self.mpc.update(radar_state, model_leads=[lead_0, lead_1, approaching], model_position=model_position, allow_third_lead=False)
    ego_distance = 0.0
    previous_ego_speed = 10.0
    accelerating_ego = approaching
    for frame in range(16):
      ego_speed = 10.0 + 2.0 * frame * self.mpc.dt
      if frame:
        ego_distance += 0.5 * (previous_ego_speed + ego_speed) * self.mpc.dt
      self.mpc.set_cur_state(ego_speed, 0.0)
      age = frame * self.mpc.dt
      future_t = LEAD_T_IDXS_MODEL + age
      accelerating_ego = model_lead(probTime=4.0, x=20.0 + 6.0 * future_t - ego_distance,
                                    y=np.interp(np.minimum(future_t, LEAD_T_IDXS_MODEL[-1]), LEAD_T_IDXS_MODEL, y_shape),
                                    v=np.full(6, 6.0))
      self.mpc.set_cur_state(10.0, 0.0)  # production MPC state is filtered; identity must use measured v_ego below
      self.mpc.update(radar_state, model_leads=[lead_0, lead_1, accelerating_ego], model_position=model_position,
                      allow_third_lead=True, v_ego=ego_speed)
      previous_ego_speed = ego_speed
    self.assertGreater(self.mpc.lead_three_confirm, LEAD_THREE_CONFIRM)

    self.mpc.update(radar_state, model_leads=[lead_0, lead_1, accelerating_ego], model_position=model_position,
                    allow_third_lead=True, v_ego=np.nan)
    self.assertEqual(self.mpc.lead_three_confirm, 0.0)
    self.assertIsNone(self.mpc.lead_three_signature)

    self.mpc.update(radar_state, model_leads=[lead_0, lead_1, approaching], model_position=model_position, allow_third_lead=False)
    same_y = np.array([3.0, 2.5, 0.4, 0.2, 0.1, 0.0])
    for frame in range(5):
      age = frame * self.mpc.dt
      future_t = LEAD_T_IDXS_MODEL + age
      if frame % 2 == 0:
        x = 20.0 + 6.0 * future_t - 10.0 * age
        v = np.full(6, 6.0)
      else:
        x = 21.0 + 5.75 * future_t - 10.0 * age
        v = np.full(6, 5.75)
      lead = model_lead(probTime=4.0, x=x,
                        y=np.interp(np.minimum(future_t, LEAD_T_IDXS_MODEL[-1]), LEAD_T_IDXS_MODEL, same_y), v=v)
      self.mpc.update(radar_state, model_leads=[lead_0, lead_1, lead], model_position=model_position, allow_third_lead=True)
    self.assertLess(self.mpc.lead_three_confirm, LEAD_THREE_CONFIRM)

    for gap in (0.5, np.nextafter(0.5, np.inf), 0.500001, 0.75, 1.0, 1.5):
      self.mpc.update(radar_state, model_leads=[lead_0, lead_1, candidate_a], model_position=model_position, allow_third_lead=False)
      self.mpc.set_cur_state(10.0, 0.0)
      for frame in range(8):
        age = frame * self.mpc.dt
        future_t = LEAD_T_IDXS_MODEL + age
        x = 20.0 + 6.0 * future_t - 10.0 * age + (gap if frame % 2 else 0.0)
        alternating = model_lead(probTime=4.0, x=x,
                                 y=np.interp(np.minimum(future_t, LEAD_T_IDXS_MODEL[-1]), LEAD_T_IDXS_MODEL, same_y),
                                 v=np.full(6, 6.0))
        self.mpc.update(radar_state, model_leads=[lead_0, lead_1, alternating], model_position=model_position, allow_third_lead=True)
        self.assertLess(self.mpc.lead_three_confirm, LEAD_THREE_CONFIRM)
        self.assertNotEqual(self.mpc.source, LongitudinalPlanSource.lead2)

    self.mpc.update(radar_state, model_leads=[lead_0, lead_1, candidate_a], model_position=model_position, allow_third_lead=False)
    self.mpc.set_cur_state(0.0, 0.0)
    for frame in range(202):
      age = frame * self.mpc.dt
      future_t = LEAD_T_IDXS_MODEL + age
      long_lived = model_lead(probTime=4.0, x=20.0 + 6.0 * future_t,
                              y=np.interp(np.minimum(future_t, LEAD_T_IDXS_MODEL[-1]), LEAD_T_IDXS_MODEL, same_y),
                              v=np.full(6, 6.0))
      self.mpc.update(radar_state, model_leads=[lead_0, lead_1, long_lived], model_position=model_position, allow_third_lead=True)
    self.assertIsNone(self.mpc.lead_three_signature)
    self.assertNotEqual(self.mpc.source, LongitudinalPlanSource.lead2)

  def test_runtime_policy_range_keeps_solver_finite(self):
    radar_state = SimpleNamespace(
      leadOne=radar_lead(),
      leadTwo=radar_lead(present=False),
    )
    models = [model_lead(), model_lead(prob=0.0)]
    for jerk_scale, t_follow in ((1.0, 1.0), (0.5, 1.45), (0.3, 1.9)):
      self.mpc.set_weights(jerk_factor_scale=jerk_scale)
      self.mpc.update(
        radar_state,
        t_follow=t_follow,
        model_leads=models,
      )
      self.assertEqual(self.mpc.solution_status, 0)
      self.assertTrue(np.all(np.isfinite(self.mpc.v_solution)))
      self.assertTrue(np.all(np.isfinite(self.mpc.a_solution)))
      np.testing.assert_allclose(self.mpc.params[:, 4], t_follow)

  def test_aggressive_lead_taper_keeps_gap_and_releases_for_risk(self):
    policy = LongitudinalPolicy(1.0, 1.0, 0.0, False, False, False, False)
    lead = longitudinal_planner.LeadObservation(True, 26.0, 13.2, 0.2, 1.0)

    self.assertEqual(get_T_FOLLOW(log.LongitudinalPersonality.aggressive), 1.0)
    self.assertGreater(get_lead_jerk_scale(log.LongitudinalPersonality.aggressive, lead, 13.0,
                                           policy, 0.1, True), 1.0)
    for risky_policy, model_accel, model_valid in (
      (policy, -0.1, True),
      (policy, None, True),
      (policy, 0.1, False),
      (LongitudinalPolicy(1.0, 1.0, 0.1, False, False, False, False), 0.1, True),
      (LongitudinalPolicy(1.0, 1.0, 0.0, True, False, False, False), 0.1, True),
    ):
      self.assertEqual(get_lead_jerk_scale(log.LongitudinalPersonality.aggressive, lead, 13.0,
                                           risky_policy, model_accel, model_valid), 1.0)
    desired_distance = get_safe_obstacle_distance(13.0, policy.t_follow) - get_stopped_equivalence_factor(13.0)
    self.assertEqual(get_lead_jerk_scale(
      log.LongitudinalPersonality.aggressive,
      longitudinal_planner.LeadObservation(True, desired_distance + 1.0, 13.0, 0.2, 1.0),
      13.0, policy, 0.1, True,
    ), 2.0)
    self.assertEqual(get_lead_jerk_scale(
      log.LongitudinalPersonality.aggressive,
      longitudinal_planner.LeadObservation(True, 26.0, 12.9, 0.2, 1.0),
      13.0, policy, 0.1, True,
    ), 1.0)
    self.assertEqual(get_lead_jerk_scale(log.LongitudinalPersonality.aggressive,
                                         longitudinal_planner.LeadObservation(True, desired_distance, 13.0, 0.2, 1.0),
                                         13.0, policy, 0.1, True), 1.0)
    self.assertEqual(get_lead_jerk_scale(log.LongitudinalPersonality.aggressive, lead,
                                         longitudinal_planner.CRUISE_COMFORT_SPEED_BP[0], policy, 0.1, True), 1.0)


class TestStrongCruiseEnvelope(unittest.TestCase):
  def test_low_speed_requests_blotv1_authority_with_platform_clamp(self):
    self.assertEqual(BLOTV2_ACCEL_REQUEST_MAX, 4.0)
    self.assertEqual(BLOTV2_ACCEL_MAX, min(ACCEL_MAX, BLOTV2_ACCEL_REQUEST_MAX))
    self.assertEqual(get_requested_max_accel(0.0), BLOTV2_ACCEL_REQUEST_MAX)
    self.assertEqual(get_max_accel(0.0), BLOTV2_ACCEL_MAX)
    speeds = np.linspace(0.0, 50.0, 101)
    self.assertLessEqual(max(get_max_accel(speed) for speed in speeds), BLOTV2_ACCEL_MAX)

  def test_cubic_curve_matches_agreed_speed_samples(self):
    speeds = np.arange(0.0, 45.0, 5.0)
    expected = [4.0, 2.877734375, 2.034375, 1.430078125, 1.025,
                0.779296875, 0.653125, 0.606640625, 0.6]
    np.testing.assert_allclose([get_requested_max_accel(speed) for speed in speeds], expected)

  def test_cubic_curve_is_monotonic_convex_and_has_no_speed_node_corners(self):
    speeds = np.linspace(0.0, longitudinal_planner.A_CRUISE_MAX_CURVE_SPEED, 401)
    accels = np.array([get_requested_max_accel(speed) for speed in speeds])
    slopes = np.diff(accels) / np.diff(speeds)

    self.assertTrue(np.all(np.diff(accels) <= 0.0))
    self.assertTrue(np.all(np.diff(slopes) >= -1e-12))

    step = 1e-4
    for old_node in (10.0, 25.0):
      left_slope = (get_requested_max_accel(old_node) - get_requested_max_accel(old_node - step)) / step
      right_slope = (get_requested_max_accel(old_node + step) - get_requested_max_accel(old_node)) / step
      self.assertAlmostEqual(left_slope, right_slope, places=5)

  def test_curve_reaches_a_smooth_high_speed_floor(self):
    self.assertEqual(get_requested_max_accel(40.0), 0.6)
    self.assertEqual(get_requested_max_accel(50.0), 0.6)
    near_floor_slope = (get_requested_max_accel(40.0) - get_requested_max_accel(39.99)) / 0.01
    self.assertAlmostEqual(near_floor_slope, 0.0, places=5)

  def test_route_d2_urban_correction_remains_comfort_limited(self):
    # Route 000000d2--a62f0c1831 reached 1.79 m/s² near 40 mph for a
    # 5.4 mph set-speed error. The ordinary-cruise response remains below the
    # new cubic envelope, so smoothing the envelope cannot restore that lunge.
    v_ego = 40.0 * CV.MPH_TO_MS
    v_cruise = v_ego + 5.4 * CV.MPH_TO_MS
    comfort_target = get_cruise_comfort_accel(v_cruise, v_ego, -0.3)

    self.assertAlmostEqual(comfort_target, 0.43452288, places=6)
    self.assertLess(comfort_target, get_max_accel(v_ego))

  def test_turn_budget_does_not_clip_straight_launch(self):
    np.testing.assert_allclose(longitudinal_planner._A_TOTAL_MAX_V, [4.0, 4.0])

  def test_existing_blotv2_jerk_tune_is_preserved(self):
    np.testing.assert_allclose(
      longitudinal_planner.J_CRUISE_BP,
      [0.0, 10.0, 25.0, 40.0],
    )
    np.testing.assert_allclose(
      longitudinal_planner.J_CRUISE_VALS,
      [2.0, 1.6, 1.0, 0.6],
    )


class TestOrdinaryCruiseComfort(unittest.TestCase):
  CP = SimpleNamespace(steerRatio=15.0, wheelbase=2.7)
  DT = 0.05

  def settled_target(self, v_cruise, v_ego, accel_coast=-0.3, comfort_enabled=True, e2e=False):
    target = 0.0
    for _ in range(200):
      target = get_cruise_accel(
        e2e,
        v_cruise,
        v_ego,
        target,
        0.0,
        self.CP,
        self.DT,
        accel_coast,
        True,
        comfort_enabled=comfort_enabled,
      )
    return target

  def test_route_d9_five_mph_increase_uses_proportional_target(self):
    v_ego = 74.5 * CV.MPH_TO_MS
    v_cruise = 79.5 * CV.MPH_TO_MS
    expected = longitudinal_planner.CRUISE_COMFORT_ACCEL_KP * (v_cruise - v_ego)

    self.assertAlmostEqual(self.settled_target(v_cruise, v_ego, accel_coast=-0.25), expected)
    self.assertAlmostEqual(expected, 0.402336, places=6)
    self.assertLess(expected, get_max_accel(v_ego))

  def test_route_d9_five_mph_reduction_coasts_instead_of_saturating_brakes(self):
    v_ego = 79.6 * CV.MPH_TO_MS
    v_cruise = 74.6 * CV.MPH_TO_MS
    target = self.settled_target(v_cruise, v_ego, accel_coast=-0.39)

    self.assertAlmostEqual(target, -0.402336, places=6)
    self.assertGreater(target, longitudinal_planner.A_CRUISE_MIN)

  def test_reduction_uses_natural_uphill_coast_and_not_downhill_acceleration(self):
    v_ego = 80.0 * CV.MPH_TO_MS
    v_cruise = 75.0 * CV.MPH_TO_MS

    self.assertAlmostEqual(get_cruise_comfort_accel(v_cruise, v_ego, -0.6), -0.6)
    self.assertAlmostEqual(get_cruise_comfort_accel(v_cruise, v_ego, 0.2), -0.402336, places=6)

  def test_small_corrections_taper_continuously(self):
    v_ego = 75.0 * CV.MPH_TO_MS
    target_2_5_mph = get_cruise_comfort_accel(v_ego + 2.5 * CV.MPH_TO_MS, v_ego, -0.3)
    target_5_mph = get_cruise_comfort_accel(v_ego + 5.0 * CV.MPH_TO_MS, v_ego, -0.3)

    self.assertAlmostEqual(target_2_5_mph, target_5_mph / 2.0)

  def test_large_errors_retain_existing_acceleration_and_braking_limits(self):
    v_ego = 75.0 * CV.MPH_TO_MS

    self.assertAlmostEqual(
      self.settled_target(v_ego + 15.0 * CV.MPH_TO_MS, v_ego),
      get_max_accel(v_ego),
    )
    self.assertAlmostEqual(
      self.settled_target(v_ego - 15.0 * CV.MPH_TO_MS, v_ego),
      longitudinal_planner.A_CRUISE_MIN,
    )

  def test_low_speed_launch_retains_legacy_authority(self):
    v_ego = 5.0
    v_cruise = v_ego + 5.0 * CV.MPH_TO_MS

    comfort = self.settled_target(v_cruise, v_ego, comfort_enabled=True)
    legacy = self.settled_target(v_cruise, v_ego, comfort_enabled=False)
    self.assertAlmostEqual(comfort, legacy)

  def test_existing_jerk_schedule_still_bounds_response(self):
    v_ego = 75.0 * CV.MPH_TO_MS
    jerk = np.interp(v_ego, longitudinal_planner.J_CRUISE_BP, longitudinal_planner.J_CRUISE_VALS)
    first_target = get_cruise_accel(
      False,
      v_ego + 5.0 * CV.MPH_TO_MS,
      v_ego,
      0.0,
      0.0,
      self.CP,
      self.DT,
      -0.3,
      True,
      comfort_enabled=True,
    )

    self.assertAlmostEqual(first_target, jerk * self.DT)

  def test_experimental_cruise_deceleration_uses_existing_jerk_schedule(self):
    v_ego = 75.0 * CV.MPH_TO_MS
    previous_target = 0.4
    jerk = np.interp(v_ego, longitudinal_planner.J_CRUISE_BP, longitudinal_planner.J_CRUISE_VALS)
    first_target = get_cruise_accel(
      True,
      v_ego - 5.0 * CV.MPH_TO_MS,
      v_ego,
      previous_target,
      0.0,
      self.CP,
      self.DT,
      -0.3,
      True,
    )

    self.assertAlmostEqual(first_target, previous_target - jerk * self.DT)

  def test_present_lead_does_not_disable_cruise_comfort(self):
    class SubMaster(dict):
      def __init__(self):
        lead = radar_lead(dRel=100.0, vLead=80.0 * CV.MPH_TO_MS)
        absent_lead = radar_lead(present=False)
        super().__init__(
          carState=SimpleNamespace(vEgo=80.0 * CV.MPH_TO_MS, vCruise=75.0 * CV.MPH_TO_KPH,
                                   aEgo=0.0, standstill=False, gasPressed=False, brakePressed=False,
                                   leftBlinker=False, rightBlinker=False, steeringAngleDeg=0.0),
          carControl=SimpleNamespace(orientationNED=[], latActive=False),
          controlsState=SimpleNamespace(forceDecel=False, longControlState=longitudinal_planner.LongCtrlState.pid),
          selfdriveState=SimpleNamespace(enabled=True, experimentalMode=False, personality=1),
          vehicleParameters=SimpleNamespace(angleOffsetDeg=0.0, roll=0.0),
          radarState=SimpleNamespace(leadOne=lead, leadTwo=absent_lead),
          modelV2=SimpleNamespace(
            position=SimpleNamespace(x=np.full(longitudinal_planner.ModelConstants.IDX_N, 50.0)),
            velocity=SimpleNamespace(x=np.full(longitudinal_planner.ModelConstants.IDX_N, 5.0)),
            acceleration=SimpleNamespace(x=np.zeros(longitudinal_planner.ModelConstants.IDX_N)),
            meta=SimpleNamespace(disengagePredictions=SimpleNamespace(gasPressProbs=[1.0, 1.0]),
                                 laneChangeState=longitudinal_planner.log.LaneChangeState.off),
            action=SimpleNamespace(desiredAcceleration=1.0, shouldStop=False),
            leadsV3=[],
          ),
        )

      def all_checks(self, services=None):
        return True

    planner = longitudinal_planner.LongitudinalPlanner(
      SimpleNamespace(openpilotLongitudinalControl=True, longitudinalActuatorDelay=0.2,
                      steerRatio=15.0, wheelbase=2.9),
    )
    sm = SubMaster()
    policy = SimpleNamespace(jerk_scale=1.0, t_follow=1.45, emergency=False)
    with (
      patch.object(planner.force_stops, "update", return_value=math.inf),
      patch.object(planner.blotv2, "update", return_value=policy),
      patch.object(planner.lead_departure, "update", return_value=False),
      patch.object(planner.mpc, "set_weights"),
      patch.object(planner.mpc, "update"),
    ):
      with patch.object(longitudinal_planner, "get_accel_from_plan", return_value=0.0):
        for _ in range(200):
          planner.update(sm)

      self.assertEqual(planner.mpc.source, LongitudinalPlanSource.cruise)
      self.assertAlmostEqual(planner.output_a_target, -0.402336, places=6)

      planner.mpc.source = LongitudinalPlanSource.lead0
      with patch.object(longitudinal_planner, "get_accel_from_plan", return_value=-1.0):
        planner.update(sm)

      self.assertEqual(planner.mpc.source, LongitudinalPlanSource.lead0)
      self.assertEqual(planner.output_a_target, -1.0)

  def test_comfort_eligibility_preserves_other_longitudinal_strategies(self):
    self.assertTrue(ordinary_cruise_comfort_enabled(False, False, True))
    self.assertFalse(ordinary_cruise_comfort_enabled(True, False, True))
    self.assertFalse(ordinary_cruise_comfort_enabled(False, True, True))
    self.assertFalse(ordinary_cruise_comfort_enabled(False, False, False))
    self.assertFalse(ordinary_cruise_comfort_enabled(False, False, True, speed_limiter_active=True))

  def test_disabled_comfort_matches_legacy_targets(self):
    v_ego = 80.0 * CV.MPH_TO_MS

    self.assertAlmostEqual(
      self.settled_target(v_ego + 5.0 * CV.MPH_TO_MS, v_ego, comfort_enabled=False),
      get_max_accel(v_ego),
    )
    self.assertAlmostEqual(
      self.settled_target(v_ego - 5.0 * CV.MPH_TO_MS, v_ego, comfort_enabled=False),
      longitudinal_planner.A_CRUISE_MIN,
    )


if __name__ == "__main__":
  unittest.main()
