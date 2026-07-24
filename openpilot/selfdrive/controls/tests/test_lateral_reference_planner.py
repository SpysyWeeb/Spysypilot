import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.controls.lib.lateral_reference_planner import (
  ActuatorPreviewConfig,
  EPISODE_COMMITMENT_LOOKAHEAD,
  LateralReferencePlanner,
  MAX_PREVIEW_LATERAL_ACCEL_DELTA,
  TRAJECTORY_DT,
  TRAJECTORY_T,
  UNWIND_RELEASE_PREVIEW_TIME,
)
from openpilot.selfdrive.modeld.constants import ModelConstants


MODEL_T = np.asarray(ModelConstants.T_IDXS)
V_EGO = 3.0
LATERAL_DELAY = 0.385
HYUNDAI_ACTUATOR = ActuatorPreviewConfig(max_torque=409, delta_up=4, delta_down=7)


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

  def test_actuator_preview_leads_low_speed_sharp_unwind_without_erasing_turn(self):
    speed = 3.0
    curvature = 0.04
    # Hold the turn through the action horizon, then unwind in the future path.
    yaws = speed * curvature * np.minimum(MODEL_T, 0.7)
    planner = LateralReferencePlanner()
    planner.configure_actuator(HYUNDAI_ACTUATOR)
    assert planner.update(build_model(yaws, speed), curvature, speed)

    output = planner.get_curvature(curvature, speed, LATERAL_DELAY, applied_torque=-0.8, lat_accel_factor=2.5, friction=0.115)
    assert output < curvature
    assert (curvature - output) * speed**2 <= MAX_PREVIEW_LATERAL_ACCEL_DELTA + 1e-9
    assert planner.solution is not None
    assert planner.diagnostics.extra_time > 0.0
    assert planner.diagnostics.unwind_scale > 0.0

    # These values are copied into cereal at 100 Hz. pycapnp rejects NumPy
    # scalar types even though they behave like Python floats.
    torque_log = log.ControlsState.LateralTorqueState.new_message()
    torque_log.referenceBaseCurvature = planner.diagnostics.base_curvature
    torque_log.referenceOutputCurvature = planner.diagnostics.output_curvature
    torque_log.trajectoryReferenceCurvatureRate = planner.diagnostics.trajectory_curvature_rate
    torque_log.trajectoryReferenceRateValid = planner.diagnostics.trajectory_rate_valid
    torque_log.referencePreviewTime = planner.diagnostics.sample_time
    torque_log.referencePreviewExtraTime = planner.diagnostics.extra_time
    torque_log.referenceTargetTorque = planner.diagnostics.target_torque
    torque_log.referenceAppliedTorque = planner.diagnostics.applied_torque
    torque_log.referenceUnwindScale = planner.diagnostics.unwind_scale
    torque_log.referenceAuthorityRestored = planner.diagnostics.authority_restored
    torque_log.referencePreviewCorrection = planner.diagnostics.preview_correction
    torque_log.referenceGeometricTargetTorque = planner.diagnostics.geometric_target_torque
    torque_log.referenceNeutralTorque = planner.diagnostics.neutral_torque
    torque_log.referenceReachableTargetTorque = planner.diagnostics.reachable_target_torque
    torque_log.referenceSustainedUnwindScale = planner.diagnostics.sustained_unwind_scale
    torque_log.referenceEpisodeTargetTorque = planner.diagnostics.episode_target_torque
    torque_log.referenceEpisodeLateralAccel = planner.diagnostics.episode_lateral_accel

  def test_sustained_unwind_leads_release_with_coherent_future_target(self):
    speed = 4.0
    curvature = 0.06
    yaws = speed * curvature * np.minimum(MODEL_T, 0.55)
    planner = LateralReferencePlanner()
    planner.configure_actuator(HYUNDAI_ACTUATOR)
    assert planner.update(build_model(yaws, speed), curvature, speed)

    output = planner.get_curvature(
      curvature,
      speed,
      0.2,
      applied_torque=-0.8,
      lat_accel_factor=2.7,
      friction=0.13,
      roll=0.05,
      lat_accel_offset=-0.1,
    )
    diagnostics = planner.diagnostics
    assert diagnostics.sustained_unwind_scale > 0.9
    assert diagnostics.extra_time == pytest.approx(UNWIND_RELEASE_PREVIEW_TIME)
    assert output < diagnostics.base_curvature
    assert abs(diagnostics.preview_correction) <= MAX_PREVIEW_LATERAL_ACCEL_DELTA + 1e-9

    _, reachable = planner._reachable_torque_trajectory(speed, 2.7, 0.13, 0.05, -0.1)
    expected_target = np.interp(diagnostics.sample_time, np.arange(len(reachable)) * 0.05, reachable)
    assert diagnostics.reachable_target_torque == pytest.approx(expected_target)

  def test_episode_target_samples_later_horizon_for_handoff_commitment(self):
    speed = 4.0
    curvature = 0.05
    # The future path crosses through neutral into a sustained opposite curve.
    curvature_path = np.where(
      MODEL_T < 0.35,
      curvature,
      np.where(MODEL_T < 0.6, curvature - 0.09 * (MODEL_T - 0.35) / 0.25, -0.04),
    )
    yaws = np.zeros_like(MODEL_T)
    yaws[1:] = np.cumsum(speed * curvature_path[:-1] * np.diff(MODEL_T))
    planner = LateralReferencePlanner()
    planner.configure_actuator(HYUNDAI_ACTUATOR)
    assert planner.update(build_model(yaws, speed), curvature, speed)

    planner.get_curvature(
      curvature,
      speed,
      0.2,
      applied_torque=-0.5,
      lat_accel_factor=2.7,
      friction=0.13,
    )
    diagnostics = planner.diagnostics
    assert EPISODE_COMMITMENT_LOOKAHEAD > 0.0
    assert diagnostics.episode_target_torque > diagnostics.neutral_torque
    assert diagnostics.episode_lateral_accel < 0.0

  def test_actuator_preview_preserves_turn_in_authority(self):
    speed = 3.0
    future_turn_yaw = np.where(MODEL_T < 0.7, 0.0, speed * 0.04 * (MODEL_T - 0.7))
    planner = LateralReferencePlanner()
    planner.configure_actuator(HYUNDAI_ACTUATOR)
    assert planner.update(build_model(future_turn_yaw, speed), 0.0, speed)

    raw_curvature = 0.02
    output = planner.get_curvature(raw_curvature, speed, LATERAL_DELAY, applied_torque=0.0, lat_accel_factor=2.5, friction=0.115)
    assert output >= raw_curvature
    assert planner.diagnostics.extra_time == 0.0
    assert planner.diagnostics.authority_restored > 0.0

  def test_trajectory_rate_preserves_quick_planned_motion(self):
    speed = 10.0
    curvature_rate = 0.001
    # Integral of yaw_rate = speed * curvature_rate * time.
    yaws = 0.5 * speed * curvature_rate * MODEL_T**2
    planner = LateralReferencePlanner()
    assert planner.update(build_model(yaws, speed), 0.0, speed)

    planner.get_curvature(0.0, speed, LATERAL_DELAY)
    assert planner.diagnostics.trajectory_rate_valid
    assert planner.diagnostics.trajectory_curvature_rate == pytest.approx(curvature_rate, rel=0.10)

  def test_trajectory_rate_rejects_constant_curve_replan_step(self):
    speed = 10.0
    planner = LateralReferencePlanner()
    assert planner.update(constant_curvature_model(0.003, speed), 0.0, speed)
    previous_output = planner.get_curvature(0.003, speed, LATERAL_DELAY)
    for _ in range(4):
      previous_output = planner.get_curvature(0.003, speed, LATERAL_DELAY)

    # A small replan-to-replan curvature step looks like a very fast rate when
    # the final 100 Hz output is differentiated. The horizon itself still
    # describes a steady curve and should report almost no planned motion.
    assert planner.update(constant_curvature_model(0.0035, speed), 0.003, speed)
    output = planner.get_curvature(0.0035, speed, LATERAL_DELAY)
    finite_difference_rate = (output - previous_output) / planner.dt
    assert abs(finite_difference_rate) > 0.01
    assert abs(planner.diagnostics.trajectory_curvature_rate) < 0.001

  def test_actuator_transition_accounts_for_slow_sign_reversal(self):
    planner = LateralReferencePlanner()
    planner.configure_actuator(HYUNDAI_ACTUATOR)
    same_direction_time = planner._transition_time(-0.8, -0.4)
    reversal_time = planner._transition_time(-0.8, 0.4)
    assert reversal_time > same_direction_time

  def test_closed_form_reachability_matches_transition_definition(self):
    planner = LateralReferencePlanner()
    planner.configure_actuator(HYUNDAI_ACTUATOR)
    rng = np.random.default_rng(0)

    for desired_previous, reachable_next, duration in zip(
      rng.uniform(-1.0, 1.0, 1000),
      rng.uniform(-1.0, 1.0, 1000),
      rng.uniform(0.0, 0.2, 1000),
      strict=True,
    ):
      actual = planner._reachable_previous_torque(desired_previous, reachable_next, duration)

      # Independent high-precision version of the previous monotonic bisection.
      if planner._transition_time(desired_previous, reachable_next) <= duration:
        expected = desired_previous
      else:
        feasible_fraction = 0.0
        infeasible_fraction = 1.0
        for _ in range(60):
          fraction = 0.5 * (feasible_fraction + infeasible_fraction)
          candidate = reachable_next + fraction * (desired_previous - reachable_next)
          if planner._transition_time(candidate, reachable_next) <= duration:
            feasible_fraction = fraction
          else:
            infeasible_fraction = fraction
        expected = reachable_next + feasible_fraction * (desired_previous - reachable_next)

      assert actual == pytest.approx(expected, abs=1e-12)
      assert planner._transition_time(actual, reachable_next) <= duration + 1e-12

  def test_vectorized_torque_demand_matches_scalar_definition(self):
    planner = LateralReferencePlanner()
    planner.configure_actuator(HYUNDAI_ACTUATOR)
    rng = np.random.default_rng(1)
    planner.solution = rng.uniform(-0.04, 0.04, len(TRAJECTORY_T))
    planner.predicted_speeds = rng.uniform(1.0, 15.0, len(TRAJECTORY_T))
    v_ego = 5.0
    lat_accel_factor = 2.7
    friction = 0.13
    roll = 0.06
    lat_accel_offset = -0.1

    desired, _ = planner._reachable_torque_trajectory(v_ego, lat_accel_factor, friction, roll, lat_accel_offset)
    scalar_desired = np.asarray(
      [
        planner._target_torque(curvature, sample_time, v_ego, lat_accel_factor, friction, roll, lat_accel_offset)
        for curvature, sample_time in zip(planner.solution, TRAJECTORY_T, strict=True)
      ]
    )
    np.testing.assert_array_equal(desired, scalar_desired)

  def test_actuator_preview_keeps_road_tested_single_pass(self, monkeypatch):
    speed = 4.0
    curvature = 0.05
    yaws = speed * curvature * np.minimum(MODEL_T, 0.6)
    planner = LateralReferencePlanner()
    planner.configure_actuator(HYUNDAI_ACTUATOR)
    assert planner.update(build_model(yaws, speed), curvature, speed)

    transition_calls = 0
    transition_time = planner._transition_time

    def count_transition(current, target):
      nonlocal transition_calls
      transition_calls += 1
      return transition_time(current, target)

    monkeypatch.setattr(planner, "_transition_time", count_transition)
    planner.get_curvature(curvature, speed, 0.2, applied_torque=-0.8, lat_accel_factor=2.7, friction=0.13)
    assert transition_calls == 1

  def test_reachable_torque_envelope_breaks_saturated_preview_deadlock(self):
    speed = 6.0
    curvature = 0.08
    # At the fixed-delay sample both applied and geometric target torque are
    # saturated in the turn direction, while the full path clearly unwinds.
    yaws = speed * curvature * np.minimum(MODEL_T, 0.6)
    planner = LateralReferencePlanner()
    planner.configure_actuator(HYUNDAI_ACTUATOR)
    assert planner.update(build_model(yaws, speed), curvature, speed)

    planner.get_curvature(curvature, speed, 0.2, applied_torque=-1.0, lat_accel_factor=2.7, friction=0.13, roll=0.07, lat_accel_offset=-0.103)
    diagnostics = planner.diagnostics
    assert diagnostics.geometric_target_torque < -0.85
    assert diagnostics.reachable_target_torque > diagnostics.geometric_target_torque + 0.5
    assert diagnostics.unwind_scale > 0.9
    # A sustained path release now selects a bounded future sample even when
    # applied torque and the old scalar target are both saturated.
    assert diagnostics.extra_time == pytest.approx(UNWIND_RELEASE_PREVIEW_TIME)

  def test_reference_target_includes_crown_adjusted_neutral(self):
    speed = 5.0
    planner = LateralReferencePlanner()
    planner.configure_actuator(HYUNDAI_ACTUATOR)
    assert planner.update(constant_curvature_model(0.0, speed), 0.0, speed)

    planner.get_curvature(0.0, speed, 0.2, applied_torque=0.0, lat_accel_factor=2.7, friction=0.13, roll=0.07, lat_accel_offset=-0.103)
    expected_neutral = (0.07 * ACCELERATION_DUE_TO_GRAVITY - 0.103) / 2.7
    assert planner.diagnostics.neutral_torque == pytest.approx(expected_neutral)
    assert planner.diagnostics.geometric_target_torque == pytest.approx(expected_neutral, abs=1e-6)
    assert planner.diagnostics.reachable_target_torque == pytest.approx(expected_neutral, abs=1e-6)

  def test_backward_torque_envelope_is_rate_reachable(self):
    speed = 6.0
    curvature = 0.08
    yaws = speed * curvature * np.minimum(MODEL_T, 0.6)
    planner = LateralReferencePlanner()
    planner.configure_actuator(HYUNDAI_ACTUATOR)
    assert planner.update(build_model(yaws, speed), curvature, speed)

    _, reachable = planner._reachable_torque_trajectory(speed, 2.7, 0.13, 0.07, -0.103)
    transition_times = [planner._transition_time(current, following) for current, following in zip(reachable[:-1], reachable[1:], strict=True)]
    assert max(transition_times) <= TRAJECTORY_DT + 1e-12

  def test_speed_change_alone_does_not_trigger_path_unwind(self):
    speed = 8.0
    curvature = 0.01
    speeds = np.linspace(speed, 3.0, len(MODEL_T))
    planner = LateralReferencePlanner()
    planner.configure_actuator(HYUNDAI_ACTUATOR)
    assert planner.update(build_model(speed * curvature * MODEL_T, speeds), curvature, speed)

    planner.get_curvature(curvature, speed, LATERAL_DELAY, applied_torque=-0.4, lat_accel_factor=2.5, friction=0.115)
    assert planner.diagnostics.unwind_scale == pytest.approx(0.0, abs=1e-6)
    assert planner.diagnostics.extra_time == pytest.approx(0.0, abs=1e-6)

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
    assert not planner.diagnostics.trajectory_rate_valid
