import numpy as np

from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.common.parameterized import parameterized
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.lane_centering import LaneCentering, get_lane_center_curvature, \
                                                            GAIN, LOOKAHEAD_T, MIN_LOOKAHEAD, OFFSET_BP, SMOOTH_TAU
from openpilot.selfdrive.modeld.constants import ModelConstants

X_IDXS = np.array(ModelConstants.X_IDXS)
T_IDXS = np.array(ModelConstants.T_IDXS)

V_EGO = 25.0
MAX_LAT_ACCEL = GAIN * 2 * OFFSET_BP[0] / LOOKAHEAD_T**2
CM = GAIN * 2 * 0.01 / (V_EGO * LOOKAHEAD_T)**2  # curvature for a centimeter of offset


def get_model(lane_center=0.0, plan=0.0, v_ego=V_EGO, width=3.5, curvature=0.0, probs=(1.0, 1.0), stds=(0.05, 0.05), frame_id=1):
  # lane_center and plan are lateral positions in the car's frame, a number or one per point
  model = log.ModelDataV2.new_message()
  model.frameId = frame_id
  model.init('laneLines', 4)
  for line, lanes in zip(model.laneLines, (-1.5, -0.5, 0.5, 1.5), strict=True):
    line.x = X_IDXS.tolist()
    line.y = (lane_center + lanes * width + curvature * X_IDXS**2 / 2).tolist()
  model.laneLineProbs = [1.0, *probs, 1.0]
  model.laneLineStds = [0.05, *stds, 0.05]

  plan_x = v_ego * T_IDXS
  model.position.x = plan_x.tolist()
  model.position.y = (plan + curvature * plan_x**2 / 2).tolist()
  return model


def get_car_state(v_ego=V_EGO, **kwargs):
  return car.CarState.new_message(vEgo=v_ego, **kwargs)


class TestLaneCenterCurvature(OpenpilotTestCase):
  def test_steers_toward_center(self):
    assert get_lane_center_curvature(get_model(), V_EGO) == 0.0
    # positive curvature is to the right, like the lane center
    right = get_lane_center_curvature(get_model(lane_center=0.3), V_EGO)
    assert right == -get_lane_center_curvature(get_model(lane_center=-0.3), V_EGO)
    np.testing.assert_allclose(right, GAIN * 2 * 0.3 / (V_EGO * LOOKAHEAD_T)**2, rtol=1e-5)

  def test_plan_already_centering(self):
    np.testing.assert_allclose(get_lane_center_curvature(get_model(lane_center=0.3, plan=0.3), V_EGO), 0.0, atol=1e-3 * CM)

  @parameterized.expand([(0.002,), (-0.002,)])
  def test_road_curvature_cancels(self, curvature):
    # the lines and the plan are sampled at different distances, a curve between two points does not cancel exactly
    np.testing.assert_allclose(get_lane_center_curvature(get_model(curvature=curvature), V_EGO), 0.0, atol=CM)
    np.testing.assert_allclose(get_lane_center_curvature(get_model(lane_center=0.3, curvature=curvature), V_EGO),
                               get_lane_center_curvature(get_model(lane_center=0.3), V_EGO), atol=CM)

  def test_authority(self):
    for v_ego in (5.0, 10.0, 20.0, 40.0):
      lat_accels = [get_lane_center_curvature(get_model(lane_center=o, v_ego=v_ego), v_ego) * v_ego**2 for o in np.arange(0, 1.5, 0.01)]
      np.testing.assert_allclose(max(lat_accels), MAX_LAT_ACCEL * min(v_ego * LOOKAHEAD_T / MIN_LOOKAHEAD, 1.0)**2, rtol=1e-6)
      assert max(lat_accels) <= 0.2 + 1e-6
      assert lat_accels[-1] == 0.0

  def test_fades_out_far_from_center(self):
    # halfway through the fade half of the offset counts, at the end of it none
    np.testing.assert_allclose(get_lane_center_curvature(get_model(lane_center=0.7), V_EGO),
                               get_lane_center_curvature(get_model(lane_center=0.35), V_EGO))
    assert get_lane_center_curvature(get_model(lane_center=OFFSET_BP[1]), V_EGO) == 0.0

  @parameterized.expand([
    ({'probs': (0.4, 1.0)},),
    ({'probs': (1.0, 0.4)},),
    ({'stds': (0.4, 0.05)},),
    ({'stds': (0.05, 0.4)},),
    ({'width': 2.4},),
    ({'width': 4.6},),
  ])
  def test_needs_a_lane(self, kwargs):
    assert get_lane_center_curvature(get_model(lane_center=0.3), V_EGO) != 0.0
    assert get_lane_center_curvature(get_model(lane_center=0.3, **kwargs), V_EGO) == 0.0

  def test_needs_a_plan(self):
    assert get_lane_center_curvature(log.ModelDataV2.new_message(), V_EGO) == 0.0
    # plan shorter than the lookahead
    assert get_lane_center_curvature(get_model(lane_center=0.3, v_ego=0.5), 0.5) == 0.0

  def test_malformed_model(self):
    def malformed(edit):
      model = get_model(lane_center=0.3)
      edit(model)
      return get_lane_center_curvature(model, V_EGO)

    assert malformed(lambda m: setattr(m.laneLines[1], 'y', [0.0] * 10)) == 0.0
    assert malformed(lambda m: setattr(m.laneLines[2], 'x', [])) == 0.0
    assert malformed(lambda m: setattr(m.position, 'y', [0.0] * 20)) == 0.0
    assert malformed(lambda m: setattr(m, 'laneLineProbs', [1.0, 1.0])) == 0.0
    assert malformed(lambda m: setattr(m, 'laneLineStds', [0.05, 0.05])) == 0.0

  @parameterized.expand([({'lane_center': np.nan},), ({'plan': np.nan},)])
  def test_bad_model_output(self, kwargs):
    assert get_lane_center_curvature(get_model(**kwargs), V_EGO) == 0.0


class TestLaneCentering(OpenpilotTestCase):
  def setup_method(self):
    self.lane_centering = LaneCentering()
    self.target = get_lane_center_curvature(get_model(lane_center=0.3), V_EGO)

  def run_frames(self, seconds, lat_active=True, **car_state):
    model = get_model(lane_center=0.3)
    for _ in range(round(seconds / DT_CTRL)):
      curvature = self.lane_centering.update(lat_active, get_car_state(**car_state), model)
    return curvature

  def test_smooth(self):
    assert 0.0 < self.run_frames(DT_CTRL) < 0.05 * self.target
    np.testing.assert_allclose(self.run_frames(SMOOTH_TAU), self.target * (1 - np.exp(-1)), rtol=0.05)
    np.testing.assert_allclose(self.run_frames(3.0), self.target, rtol=1e-4)

  def test_lat_inactive(self):
    self.run_frames(3.0)
    assert self.run_frames(DT_CTRL, lat_active=False) == 0.0
    assert self.run_frames(DT_CTRL) < 0.05 * self.target

  def test_reset(self):
    self.run_frames(3.0)
    self.lane_centering.reset()
    assert self.run_frames(DT_CTRL) < 0.05 * self.target

  @parameterized.expand([({'steeringPressed': True},), ({'leftBlinker': True},), ({'rightBlinker': True},)])
  def test_driver_leaving_lane(self, car_state):
    self.run_frames(3.0)
    assert 0.0 < self.run_frames(DT_CTRL, **car_state) < self.target
    np.testing.assert_allclose(self.run_frames(3.0, **car_state), 0.0, atol=1e-3 * self.target)
    np.testing.assert_allclose(self.run_frames(3.0), self.target, rtol=1e-4)

  def test_lane_change(self):
    self.run_frames(3.0)
    model = get_model(lane_center=0.3)
    model.meta.laneChangeState = log.LaneChangeState.laneChangeStarting
    for _ in range(300):
      curvature = self.lane_centering.update(True, get_car_state(), model)
    np.testing.assert_allclose(curvature, 0.0, atol=1e-3 * self.target)

  def test_updates_with_the_model(self):
    self.run_frames(3.0)
    # same frame, the lane moved: nothing to do until the next model frame
    moved = get_model(lane_center=-0.3)
    np.testing.assert_allclose(self.lane_centering.update(True, get_car_state(), moved), self.target, rtol=1e-4)
    moved.frameId = 2
    assert self.lane_centering.update(True, get_car_state(), moved) < self.target * (1 - 1e-3)

  @parameterized.expand([(15.0,), (25.0,), (35.0,)])
  def test_closed_loop(self, v_ego):
    # a driving model that holds its heading along the lane within a second and drifts to 0.15 m left of
    # center over 7 s, steering that lags twice the 0.3 s measured on the Palisade, car starts 0.4 m left
    preferred, recenter_tau, heading_tau = -0.15, 7.0, 1.0
    delay_frames, lag_tau = round(0.4 / DT_CTRL), 0.2

    def model_accel(y, y_rate):
      return -(y - preferred) / (recenter_tau * heading_tau) - y_rate / heading_tau

    y, y_rate, lat_accel = -0.4, 0.0, 0.0
    pending = [0.0] * delay_frames
    ys = []
    for frame in range(round(60 / DT_CTRL)):
      if frame % round(DT_MDL / DT_CTRL) == 0:
        # the model's plan from here, in the lane's frame, then seen from the car
        plan_y, plan_rate, plan = y, y_rate, [y]
        for dt in np.diff(T_IDXS):
          plan_rate += model_accel(plan_y, plan_rate) * dt
          plan_y += plan_rate * dt
          plan.append(plan_y)
        heading = y_rate / v_ego
        model = get_model(lane_center=-y - heading * X_IDXS, plan=np.array(plan) - y - heading * v_ego * T_IDXS,
                          v_ego=v_ego, frame_id=frame)
        desired_accel = model_accel(y, y_rate)

      curvature = self.lane_centering.update(True, get_car_state(v_ego), model)
      pending.append(desired_accel + curvature * v_ego**2)
      lat_accel += (pending.pop(0) - lat_accel) * DT_CTRL / lag_tau
      y_rate += lat_accel * DT_CTRL
      y += y_rate * DT_CTRL
      ys.append(y)

    ys = np.array(ys)
    assert abs(ys[-1]) < 0.05 < abs(preferred)
    assert max(ys) < 0.05, "overshoots the center"
    assert np.ptp(ys[round(-20 / DT_CTRL):]) < 0.01, "does not settle"
