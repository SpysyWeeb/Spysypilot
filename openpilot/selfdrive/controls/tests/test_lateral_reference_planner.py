import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.lateral_reference_planner import (
  LateralReferencePlanner,
)
from openpilot.selfdrive.modeld.constants import ModelConstants


MODEL_T = np.asarray(ModelConstants.T_IDXS)
V_EGO = 3.0
LATERAL_DELAY = 0.385


def build_model(yaws, speeds=V_EGO):
  model = log.ModelDataV2.new_message()
  model.orientation.z = np.asarray(yaws, dtype=float).tolist()
  if np.isscalar(speeds):
    speeds = np.full_like(MODEL_T, speeds, dtype=float)
  model.velocity.x = np.asarray(speeds, dtype=float).tolist()
  return model


def constant_curvature_model(curvature, speed=V_EGO):
  return build_model(speed * curvature * MODEL_T, speed)


class TestLateralReferencePlanner:
  def test_straight(self):
    planner = LateralReferencePlanner()
    assert planner.update(constant_curvature_model(0.0), 0.0, V_EGO)
    assert planner.get_curvature(0.0, V_EGO, LATERAL_DELAY) == pytest.approx(0.0, abs=1e-10)

  @pytest.mark.parametrize(("speed", "curvature"), [(3.0, -0.01), (3.0, 0.01), (15.0, 0.002), (30.0, 0.0005)])
  def test_constant_curvature(self, speed, curvature):
    planner = LateralReferencePlanner()
    assert planner.update(constant_curvature_model(curvature, speed), 0.0, speed)
    assert planner.get_curvature(curvature, speed, LATERAL_DELAY) == pytest.approx(curvature, rel=0.05)

  def test_symmetry(self):
    outputs = []
    for sign in (-1.0, 1.0):
      planner = LateralReferencePlanner()
      planner.update(constant_curvature_model(sign * 0.01), 0.0, V_EGO)
      outputs.append(planner.get_curvature(sign * 0.01, V_EGO, LATERAL_DELAY))
    assert outputs[0] == pytest.approx(-outputs[1], abs=1e-10)

  def test_uses_trajectory_beyond_action_time(self):
    straight = build_model(np.zeros_like(MODEL_T))
    future_turn_yaw = np.where(MODEL_T < 0.5, 0.0, V_EGO * 0.02 * (MODEL_T - 0.5))
    future_turn = build_model(future_turn_yaw)

    straight_planner = LateralReferencePlanner()
    turn_planner = LateralReferencePlanner()
    straight_planner.update(straight, 0.0, V_EGO)
    turn_planner.update(future_turn, 0.0, V_EGO)

    straight_output = straight_planner.get_curvature(0.0, V_EGO, LATERAL_DELAY)
    turn_output = turn_planner.get_curvature(0.0, V_EGO, LATERAL_DELAY)
    assert straight_output == pytest.approx(0.0, abs=1e-10)
    assert turn_output > 0.005

  def test_replan_reduces_reference_jump(self):
    planner = LateralReferencePlanner()
    planner.update(constant_curvature_model(0.01), 0.0, V_EGO)
    old_output = planner.get_curvature(0.01, V_EGO, LATERAL_DELAY)
    for _ in range(4):
      planner.get_curvature(0.01, V_EGO, LATERAL_DELAY)

    planner.update(constant_curvature_model(-0.01), 0.0, V_EGO)
    new_output = planner.get_curvature(-0.01, V_EGO, LATERAL_DELAY)
    assert abs(new_output - old_output) < 0.02
    assert new_output < old_output

  def test_advances_at_control_rate(self):
    yaw = V_EGO * 0.01 * MODEL_T + 0.5 * V_EGO * 0.01 * MODEL_T**2
    planner = LateralReferencePlanner()
    planner.update(build_model(yaw), 0.0, V_EGO)
    outputs = [planner.get_curvature(0.01, V_EGO, LATERAL_DELAY) for _ in range(5)]
    assert np.all(np.isfinite(outputs))
    assert np.all(np.diff(outputs) > 0.0)

  def test_active_at_highway_speed(self):
    speed = 30.0
    future_turn_yaw = np.where(MODEL_T < 0.5, 0.0, speed * 0.001 * (MODEL_T - 0.5))
    planner = LateralReferencePlanner()
    assert planner.update(build_model(future_turn_yaw, speed), 0.0, speed)
    assert planner.get_curvature(0.0, speed, LATERAL_DELAY) > 0.0002

  def test_low_speed_sharp_turn_behavior_is_preserved(self):
    speed = 12.0 * 0.44704
    planner = LateralReferencePlanner()
    assert planner.update(constant_curvature_model(0.01, speed), 0.0, speed)
    assert planner.get_curvature(0.04, speed, LATERAL_DELAY) == pytest.approx(0.04)
    assert planner.solution is None

  def test_reference_becomes_fully_active_by_15_mph(self):
    outputs = []
    for speed_mph in (12.0, 13.5, 15.0):
      speed = speed_mph * 0.44704
      planner = LateralReferencePlanner()
      assert planner.update(constant_curvature_model(0.01, speed), 0.0, speed)
      outputs.append(planner.get_curvature(0.04, speed, LATERAL_DELAY))

    assert outputs[0] == pytest.approx(0.04)
    assert outputs[0] > outputs[1] > outputs[2]
    assert outputs[2] == pytest.approx(0.01, rel=0.05)

  def test_speed_scaled_replan_responds_immediately(self):
    lateral_accel = 0.6
    for speed in (3.0, 10.0, 20.0, 30.0):
      curvature = lateral_accel / speed**2
      planner = LateralReferencePlanner()
      planner.update(constant_curvature_model(curvature, speed), 0.0, speed)
      old_output = planner.get_curvature(curvature, speed, LATERAL_DELAY)
      for _ in range(4):
        planner.get_curvature(curvature, speed, LATERAL_DELAY)

      planner.update(constant_curvature_model(-curvature, speed), 0.0, speed)
      new_output = planner.get_curvature(-curvature, speed, LATERAL_DELAY)
      assert new_output < 0.0
      assert abs(new_output - old_output) * speed**2 > 0.5

  @pytest.mark.parametrize(("speed", "curvature"), [(30.0, 0.02), (20.0, 0.03), (3.0, 0.3)])
  def test_unphysical_solution_falls_back(self, speed, curvature):
    planner = LateralReferencePlanner()
    assert not planner.update(constant_curvature_model(curvature, speed), 0.0, speed)
    assert planner.get_curvature(curvature, speed, LATERAL_DELAY) == curvature

  @pytest.mark.parametrize("malformed", ["empty", "nan"])
  def test_invalid_model_falls_back(self, malformed):
    planner = LateralReferencePlanner()
    planner.update(constant_curvature_model(0.01), 0.0, V_EGO)

    model = constant_curvature_model(0.01)
    if malformed == "empty":
      model.orientation.z = []
    else:
      yaws = np.asarray(model.orientation.z)
      yaws[5] = np.nan
      model.orientation.z = yaws.tolist()

    assert not planner.update(model, 0.0, V_EGO)
    assert planner.get_curvature(0.012, V_EGO, LATERAL_DELAY) == 0.012

  def test_predicted_speed_change_is_finite(self):
    speeds = np.linspace(5.0, 1.0, len(MODEL_T))
    yaws = 0.01 * speeds[0] * MODEL_T
    planner = LateralReferencePlanner()
    assert planner.update(build_model(yaws, speeds), 0.0, speeds[0])
    assert np.isfinite(planner.get_curvature(0.01, speeds[0], LATERAL_DELAY))

  def test_reset_removes_previous_solution(self):
    planner = LateralReferencePlanner()
    planner.update(constant_curvature_model(0.01), 0.0, V_EGO)
    planner.reset()
    assert planner.get_curvature(-0.004, V_EGO, LATERAL_DELAY) == -0.004
