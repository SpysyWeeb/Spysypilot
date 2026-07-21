import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.lateral_reference_planner import (
  FULL_PLANNER_CURVATURE,
  FULL_PLANNER_SPEED,
  PLANNER_RESET_CURVATURE,
  PLANNER_RESET_SPEED,
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

  @pytest.mark.parametrize("curvature", [-0.01, 0.01])
  def test_constant_curvature(self, curvature):
    planner = LateralReferencePlanner()
    assert planner.update(constant_curvature_model(curvature), 0.0, V_EGO)
    assert planner.get_curvature(curvature, V_EGO, LATERAL_DELAY) == pytest.approx(curvature, rel=0.02)

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

  def test_speed_blend_and_reset(self):
    model = constant_curvature_model(0.02)
    raw_curvature = -0.01

    planner = LateralReferencePlanner()
    planner.update(model, 0.0, FULL_PLANNER_SPEED)
    full_output = planner.get_curvature(raw_curvature, FULL_PLANNER_SPEED, LATERAL_DELAY)
    assert full_output != pytest.approx(raw_curvature)

    blend_speed = (FULL_PLANNER_SPEED + PLANNER_RESET_SPEED) / 2.0
    planner.reset()
    planner.update(model, 0.0, blend_speed)
    blend_output = planner.get_curvature(raw_curvature, blend_speed, LATERAL_DELAY)
    assert min(full_output, raw_curvature) < blend_output < max(full_output, raw_curvature)

    assert not planner.update(model, 0.0, PLANNER_RESET_SPEED)
    assert planner.get_curvature(raw_curvature, PLANNER_RESET_SPEED, LATERAL_DELAY) == raw_curvature

  def test_curvature_blend_and_reset(self):
    model = constant_curvature_model(0.01)
    planner = LateralReferencePlanner()
    planner.update(model, 0.0, V_EGO)
    full_output = planner.get_curvature(FULL_PLANNER_CURVATURE, V_EGO, LATERAL_DELAY)

    blend_curvature = (FULL_PLANNER_CURVATURE + PLANNER_RESET_CURVATURE) / 2.0
    planner.reset()
    planner.update(model, 0.0, V_EGO)
    blend_output = planner.get_curvature(blend_curvature, V_EGO, LATERAL_DELAY)
    assert min(full_output, blend_curvature) < blend_output < max(full_output, blend_curvature)

    planner.update(model, 0.0, V_EGO)
    assert planner.get_curvature(PLANNER_RESET_CURVATURE, V_EGO, LATERAL_DELAY) == PLANNER_RESET_CURVATURE
    assert planner.solution is None

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
