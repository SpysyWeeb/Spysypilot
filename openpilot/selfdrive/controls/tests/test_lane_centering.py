import numpy as np
from pytest import approx

from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.common.parameterized import parameterized
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.lane_centering import LaneCentering, get_lane_center, GAIN, LOOKAHEAD_T, MIN_LOOKAHEAD, \
                                                            MAX_OFFSET, OFFSET_BP, SMOOTH_TAU, MAX_WIDTH_AGE, NUDGE_SETTLE
from openpilot.selfdrive.modeld.constants import ModelConstants

X_IDXS = np.array(ModelConstants.X_IDXS)
T_IDXS = np.array(ModelConstants.T_IDXS)

V_EGO = 25.0
LOOKAHEAD = V_EGO * LOOKAHEAD_T
MAX_LAT_ACCEL = GAIN * 2 * MAX_OFFSET / LOOKAHEAD_T**2
CM = GAIN * 2 * 0.01 / LOOKAHEAD**2  # curvature for a centimeter of offset


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


def get_curvature(model, v_ego=V_EGO, width=None):
  lane = get_lane_center(model, v_ego, width)
  return lane.curvature if lane is not None else 0.0


class TestLaneCenter(OpenpilotTestCase):
  def test_steers_toward_center(self):
    assert get_curvature(get_model()) == 0.0
    # positive curvature is to the right, like the lane center
    right = get_curvature(get_model(lane_center=0.3))
    assert right == -get_curvature(get_model(lane_center=-0.3))
    np.testing.assert_allclose(right, GAIN * 2 * 0.3 / LOOKAHEAD**2, rtol=1e-5)

  def test_plan_already_centering(self):
    np.testing.assert_allclose(get_curvature(get_model(lane_center=0.3, plan=0.3)), 0.0, atol=1e-3 * CM)

  def test_plan_heading(self):
    # a plan closing half the gap by the lookahead is left alone, one opening it by as much is corrected for twice that
    plan_x = V_EGO * T_IDXS
    np.testing.assert_allclose(get_curvature(get_model(lane_center=0.2, plan=0.1 * plan_x / LOOKAHEAD)), 0.0, atol=CM)
    np.testing.assert_allclose(get_curvature(get_model(lane_center=0.2, plan=-0.1 * plan_x / LOOKAHEAD)), 40 * CM, rtol=1e-3)

  def test_plan_moving_over(self):
    # a plan heading off center fades like one that is already there, before the lookahead shows it
    plan_x = V_EGO * T_IDXS
    np.testing.assert_allclose(get_curvature(get_model(plan=0.35 * plan_x / LOOKAHEAD)), get_curvature(get_model(lane_center=-0.7)), rtol=1e-3)
    assert get_curvature(get_model(plan=0.45 * plan_x / LOOKAHEAD)) == 0.0
    assert get_curvature(get_model(plan=-0.45 * plan_x / LOOKAHEAD)) == 0.0

  @parameterized.expand([(0.002,), (-0.002,)])
  def test_road_curvature_cancels(self, curvature):
    # the lines and the plan are sampled at different distances, a curve between two points does not cancel exactly
    np.testing.assert_allclose(get_curvature(get_model(curvature=curvature)), 0.0, atol=CM)
    np.testing.assert_allclose(get_curvature(get_model(lane_center=0.3, curvature=curvature)), get_curvature(get_model(lane_center=0.3)), atol=CM)

  def test_authority(self):
    for v_ego in (5.0, 10.0, 20.0, 40.0):
      plan_x = v_ego * T_IDXS
      lat_accels = [get_curvature(get_model(lane_center=o, plan=s * plan_x, v_ego=v_ego), v_ego) * v_ego**2
                    for o in np.arange(0, 1.5, 0.01) for s in np.linspace(-0.1, 0.1, 9)]
      np.testing.assert_allclose(max(lat_accels), MAX_LAT_ACCEL * min(v_ego * LOOKAHEAD_T / MIN_LOOKAHEAD, 1.0)**2, rtol=1e-6)
      assert max(np.abs(lat_accels)) <= 0.2 + 1e-6
      assert get_curvature(get_model(lane_center=1.5, v_ego=v_ego), v_ego) == 0.0

  def test_fades_out_far_from_center(self):
    # halfway through the fade half of the bounded offset counts, at the end of it none
    np.testing.assert_allclose(get_curvature(get_model(lane_center=0.7)), get_curvature(get_model(lane_center=MAX_OFFSET)) / 2, rtol=1e-5)
    assert get_curvature(get_model(lane_center=OFFSET_BP[1])) == 0.0

  @parameterized.expand([
    ({'probs': (0.4, 1.0)},),
    ({'probs': (1.0, 0.4)},),
    ({'stds': (0.4, 0.05)},),
    ({'stds': (0.05, 0.4)},),
    ({'width': 2.4},),
    ({'width': 4.6},),
  ])
  def test_needs_a_lane(self, kwargs):
    assert get_curvature(get_model(lane_center=0.3)) != 0.0
    assert get_curvature(get_model(lane_center=0.3, **kwargs)) == 0.0

  @parameterized.expand([({'probs': (1.0, 0.4)},), ({'probs': (0.4, 1.0)},), ({'stds': (0.05, 0.4)},), ({'stds': (0.4, 0.05)},)])
  def test_one_line_with_a_learned_width(self, kwargs):
    lane = get_lane_center(get_model(lane_center=0.3, **kwargs), V_EGO, width=3.5)
    np.testing.assert_allclose(lane.curvature, get_curvature(get_model(lane_center=0.3)), rtol=1e-6)
    assert lane.width is None
    assert get_lane_center(get_model(lane_center=0.3, probs=(0.4, 0.4)), V_EGO, width=3.5) is None

  def test_needs_a_plan(self):
    assert get_lane_center(log.ModelDataV2.new_message(), V_EGO) is None
    # plan shorter than the lookahead
    assert get_lane_center(get_model(lane_center=0.3, v_ego=0.5), 0.5) is None

  @parameterized.expand([({'lane_center': np.nan},), ({'plan': np.nan},)])
  def test_bad_model_output(self, kwargs):
    assert get_lane_center(get_model(**kwargs), V_EGO) is None

  def test_malformed_model(self):
    def malformed(edit):
      model = get_model(lane_center=0.3)
      edit(model)
      return get_lane_center(model, V_EGO)

    assert malformed(lambda m: setattr(m.laneLines[1], 'y', [0.0] * 10)) is None
    assert malformed(lambda m: setattr(m.laneLines[2], 'x', [])) is None
    assert malformed(lambda m: setattr(m.position, 'y', [0.0] * 20)) is None
    assert malformed(lambda m: setattr(m, 'laneLineProbs', [1.0, 1.0])) is None
    assert malformed(lambda m: setattr(m, 'laneLineStds', [0.05, 0.05])) is None
    assert malformed(lambda m: setattr(m.position, 'x', [np.nan] * len(m.position.x))) is None
    # a plan that starts beyond the lookahead is still a plan
    lane = malformed(lambda m: setattr(m.position, 'x', (np.array(m.position.x) + 2 * LOOKAHEAD).tolist()))
    assert lane is None or np.isfinite(lane.curvature)


class TestLaneCentering(OpenpilotTestCase):
  def setup_method(self):
    self.lane_centering = LaneCentering()
    self.frame_id = 0
    self.target = get_curvature(get_model(lane_center=0.3))

  def run_frames(self, seconds, lat_active=True, **car_state):
    model = get_model(lane_center=0.3)
    for _ in range(round(seconds / DT_CTRL)):
      curvature = self.lane_centering.update(lat_active, get_car_state(**car_state), model)
    return curvature

  def run_model_frames(self, frames, car_state=None, **model_kwargs):
    for _ in range(frames):
      self.frame_id += 1
      model = get_model(frame_id=self.frame_id, **model_kwargs)
      for _ in range(round(DT_MDL / DT_CTRL)):
        curvature = self.lane_centering.update(True, get_car_state(**(car_state or {})), model)
    return curvature

  def press(self, frames, toward, **model_kwargs):
    # the driver's torque is positive to the left, and the car moving left moves the lane center right
    torque = float(100.0 * np.sign(toward))
    return self.run_model_frames(frames, car_state={'steeringPressed': True, 'steeringTorque': torque}, **model_kwargs)

  def run_nudge(self, start, end, toward=None):
    # the plan sits start off the lane center, the driver pushes it to end and lets go
    self.run_model_frames(60, lane_center=start)
    self.press(20, end - start if toward is None else toward, lane_center=end)
    return self.run_model_frames(round(NUDGE_SETTLE / DT_MDL), lane_center=end)

  def centering(self, lane_center, frames=60):
    target = get_curvature(get_model(lane_center=lane_center))
    curvature = self.run_model_frames(frames, lane_center=lane_center)
    assert target != 0.0
    return curvature == approx(target, rel=1e-3)

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

  @parameterized.expand([({'probs': (1.0, 0.4)},), ({'probs': (0.4, 1.0)},)])
  def test_one_line_hold(self, kwargs):
    two_lines = self.run_model_frames(60, lane_center=0.3)
    assert two_lines > 0.0
    # the other line and the learned width give the same center
    np.testing.assert_allclose(self.run_model_frames(60, lane_center=0.3, **kwargs), two_lines, rtol=1e-3)

  def test_hold_needs_a_learned_width(self):
    assert self.run_model_frames(60, lane_center=0.3, probs=(1.0, 0.4)) == 0.0

  def test_hold_needs_a_valid_line(self):
    self.run_model_frames(60, lane_center=0.3)
    np.testing.assert_allclose(self.run_model_frames(60, lane_center=0.3, probs=(1.0, 0.4), stds=(0.4, 0.05)), 0.0, atol=1e-3 * self.target)
    np.testing.assert_allclose(self.run_model_frames(60, lane_center=0.3, probs=(0.4, 0.4)), 0.0, atol=1e-3 * self.target)

  def test_hold_expires(self):
    self.run_model_frames(60, lane_center=0.3)
    self.frame_id += round(MAX_WIDTH_AGE / DT_MDL)
    np.testing.assert_allclose(self.run_model_frames(60, lane_center=0.3, probs=(1.0, 0.4)), 0.0, atol=1e-3 * self.target)

  def test_hold_follows_the_plan(self):
    # the learned width cannot be checked against the lost line, only against where the plan expects the center: the
    # reference is the last frame that had both lines, not the previous hold frame, so a wrong width cannot creep past it
    self.run_model_frames(60, lane_center=0.3)
    assert self.run_model_frames(60, lane_center=0.45, probs=(1.0, 0.4)) > 0.0
    np.testing.assert_allclose(self.run_model_frames(60, lane_center=0.6, probs=(1.0, 0.4)), 0.0, atol=1e-3 * self.target)

  def test_width_snaps_to_a_new_road(self):
    self.run_model_frames(60, width=3.5)
    # a new road is taken on within a quarter second, a glitch is not
    self.run_model_frames(4, width=2.9)
    np.testing.assert_allclose(self.lane_centering.width.x, 3.5, atol=1e-6)
    self.run_model_frames(1, width=2.9)
    np.testing.assert_allclose(self.lane_centering.width.x, 2.9, atol=1e-6)
    # glimpses of a different width with a line missing in between are not a new road
    self.run_model_frames(3, width=3.5)
    self.run_model_frames(1, width=3.5, probs=(1.0, 0.4))
    self.run_model_frames(3, width=3.5)
    np.testing.assert_allclose(self.lane_centering.width.x, 2.9, atol=1e-6)
    # a small change is filtered instead
    self.run_model_frames(20, width=3.1)
    assert 2.9 < self.lane_centering.width.x < 3.0

  def closed_loop(self, seconds, v_ego=V_EGO, heading_tau=1.0, delay=0.4, preferred=lambda t: -0.15, push=lambda t: 0.0):
    # a driving model that aligns with the lane over heading_tau and drifts to its preferred spot over 7 s, steering
    # that lags delay, a driver pushing with push(t) of lateral acceleration, car starts 0.4 m left
    recenter_tau, delay_frames, lag_tau = 7.0, round(delay / DT_CTRL), 0.2

    def model_accel(y, y_rate, t):
      return -(y - preferred(t)) / (recenter_tau * heading_tau) - y_rate / heading_tau

    y, y_rate, lat_accel = -0.4, 0.0, 0.0
    pending = [0.0] * delay_frames
    ys = []
    for frame in range(round(seconds / DT_CTRL)):
      t = frame * DT_CTRL
      if frame % round(DT_MDL / DT_CTRL) == 0:
        # the model's plan from here, in the lane's frame, then seen from the car
        plan_y, plan_rate, plan = y, y_rate, [y]
        for dt in np.diff(T_IDXS):
          plan_rate += model_accel(plan_y, plan_rate, t) * dt
          plan_y += plan_rate * dt
          plan.append(plan_y)
        heading = y_rate / v_ego
        model = get_model(lane_center=-y - heading * X_IDXS, plan=np.array(plan) - y - heading * v_ego * T_IDXS,
                          v_ego=v_ego, frame_id=frame // round(DT_MDL / DT_CTRL))
        desired_accel = model_accel(y, y_rate, t)

      # the driver's torque is positive to the left, the lane's y to the right
      CS = get_car_state(v_ego, steeringPressed=push(t) != 0.0, steeringTorque=float(-100.0 * np.sign(push(t))))
      curvature = self.lane_centering.update(True, CS, model)
      pending.append(desired_accel + curvature * v_ego**2)
      lat_accel += (pending.pop(0) - lat_accel) * DT_CTRL / lag_tau
      y_rate += (lat_accel + push(t)) * DT_CTRL
      y += y_rate * DT_CTRL
      ys.append(y)
    return np.array(ys)

  @parameterized.expand([(15.0, 1.0, 0.4), (25.0, 1.0, 0.4), (35.0, 1.0, 0.4), (25.0, 2.0, 0.45), (35.0, 2.0, 0.45)])
  def test_closed_loop(self, v_ego, heading_tau, delay):
    # steering that lags more than the 0.35 s measured on the Palisade
    ys = self.closed_loop(60, v_ego, heading_tau, delay)
    assert abs(ys[-1]) < 0.05, "not centered"
    assert max(ys) < 0.05, "overshoots the center"
    assert np.ptp(ys[round(-20 / DT_CTRL):]) < 0.01, "does not settle"

  def test_closed_loop_push(self):
    # the model holds 0.4 m left past cones and gets centered, the driver pushes left for a second: the model drives
    # until its plan comes back to the center
    def preferred(t):
      return -0.4 if t < 90 else 0.0

    def push(t):
      return -0.5 if 30 <= t < 31 else 0.0

    ys = self.closed_loop(150, preferred=preferred, push=push)
    assert abs(ys[round(29 / DT_CTRL)]) < 0.1, "centered against the model"
    assert max(abs(ys[round(t / DT_CTRL)] + 0.4) for t in (60, 89)) < 0.05, "held where the model wants"
    assert abs(ys[-1]) < 0.05, "centered again"

  def test_push_the_way_the_model_leans(self):
    # the model leans left of center and gets corrected, the driver pushes further left: the model means it
    self.run_nudge(0.3, 0.5)
    np.testing.assert_allclose(self.run_model_frames(600, lane_center=0.5), 0.0, atol=1e-3 * self.target)
    # until the plan comes back near the center
    np.testing.assert_allclose(self.run_model_frames(60, lane_center=0.2), 0.0, atol=1e-3 * self.target)
    assert self.centering(0.1)
    assert self.centering(0.3)

  def test_yield_expires(self):
    self.run_nudge(0.3, 0.5)
    self.frame_id += round(110 / DT_MDL)
    np.testing.assert_allclose(self.run_model_frames(60, lane_center=0.5), 0.0, atol=1e-3 * self.target)
    self.frame_id += round(10 / DT_MDL)
    assert self.centering(0.5)

  @parameterized.expand([({'leftBlinker': True},), ({'rightBlinker': True},)])
  def test_yield_ends_with_a_blinker(self, car_state):
    self.run_nudge(0.3, 0.5)
    self.run_model_frames(1, car_state=car_state, lane_center=0.5)
    assert self.centering(0.5)

  def test_yield_ends_on_reset(self):
    self.run_nudge(0.3, 0.5)
    self.lane_centering.reset()
    assert self.centering(0.5)

  def test_push_back_to_center(self):
    self.run_nudge(0.3, 0.5)
    self.run_nudge(0.5, 0.3)
    assert self.centering(0.3)

  def test_push_further_renews(self):
    self.run_nudge(0.3, 0.5)
    self.frame_id += round(110 / DT_MDL)
    self.run_nudge(0.5, 0.7)
    np.testing.assert_allclose(self.run_model_frames(200, lane_center=0.7), 0.0, atol=1e-3 * self.target)

  @parameterized.expand([
    (0.3, 0.1),  # toward the center
    (0.3, -0.3),  # across it
    (0.3, 0.35),  # barely
    (0.1, 0.22),  # not far enough off center
    (0.0, 0.4),  # the model did not lean
    (0.3, 3.8),  # into the next lane
  ])
  def test_push_that_is_not_agreement(self, start, end):
    self.run_nudge(start, end)
    assert self.centering(0.5)

  def test_push_in_several_presses(self):
    # presses less than half a second apart are one push
    self.run_model_frames(60, lane_center=0.3)
    self.press(5, 1, lane_center=0.37)
    self.run_model_frames(8, lane_center=0.37)
    self.press(5, 1, lane_center=0.44)
    self.run_model_frames(round(NUDGE_SETTLE / DT_MDL), lane_center=0.44)
    np.testing.assert_allclose(self.run_model_frames(200, lane_center=0.44), 0.0, atol=1e-3 * self.target)

  def test_pushes_apart_count_apart(self):
    self.run_model_frames(60, lane_center=0.3)
    self.press(5, 1, lane_center=0.37)
    self.run_model_frames(round(NUDGE_SETTLE / DT_MDL), lane_center=0.37)
    self.press(5, 1, lane_center=0.44)
    self.run_model_frames(round(NUDGE_SETTLE / DT_MDL), lane_center=0.44)
    assert self.centering(0.44)

  def test_judged_after_the_last_press(self):
    # the push is judged half a second after the last press, the correction comes back right away
    self.run_model_frames(60, lane_center=0.3)
    self.press(20, 1, lane_center=0.5)
    self.run_model_frames(8, lane_center=0.5)
    self.press(1, 1, lane_center=0.5)
    assert self.run_model_frames(9, lane_center=0.5) > 0.9 * get_curvature(get_model(lane_center=0.5))
    np.testing.assert_allclose(self.run_model_frames(60, lane_center=0.5), 0.0, atol=1e-3 * self.target)

  def test_touches_keep_centering(self):
    # a hand on the wheel costs the time it is pressed, not more
    self.run_model_frames(60, lane_center=0.3)
    for _ in range(10):
      self.press(1, 1, lane_center=0.3)
      assert self.run_model_frames(7, lane_center=0.3) > 0.9 * self.target

  def test_push_needs_the_lines(self):
    # where the push left the plan is unknown without the lines
    self.run_model_frames(60, lane_center=0.3)
    self.press(20, 1, lane_center=0.5)
    self.run_model_frames(round(NUDGE_SETTLE / DT_MDL), lane_center=0.5, probs=(0.4, 0.4))
    assert self.centering(0.5)

  @parameterized.expand([(0,), (-1,)])
  def test_the_model_moved_the_car(self, toward):
    # with the correction off while the wheel is held the model takes the car its way, the driver did not push it there
    self.run_nudge(0.3, 0.5, toward=toward)
    assert self.centering(0.5)

  def test_each_push_counts_its_own_torque(self):
    # a long press against the model, then a short push with it
    self.run_model_frames(60, lane_center=0.3)
    self.press(20, -1, lane_center=0.3)
    self.run_model_frames(round(NUDGE_SETTLE / DT_MDL), lane_center=0.3)
    self.press(2, 1, lane_center=0.5)
    self.run_model_frames(round(NUDGE_SETTLE / DT_MDL), lane_center=0.5)
    np.testing.assert_allclose(self.run_model_frames(60, lane_center=0.5), 0.0, atol=1e-3 * self.target)

  def test_push_after_lateral_comes_back(self):
    # the plan the car had before lateral control went inactive is not where a push starts
    self.run_model_frames(60, lane_center=0.3)
    self.lane_centering.update(False, get_car_state(), get_model(lane_center=0.3, frame_id=self.frame_id))
    self.press(20, 1, lane_center=0.5)
    self.run_model_frames(round(NUDGE_SETTLE / DT_MDL), lane_center=0.5)
    assert self.centering(0.5)

  def test_push_then_lane_change(self):
    self.run_model_frames(60, lane_center=0.3)
    self.press(20, 1, lane_center=0.5)
    self.run_model_frames(round(NUDGE_SETTLE / DT_MDL), car_state={'leftBlinker': True}, lane_center=0.5)
    assert self.centering(0.5)
