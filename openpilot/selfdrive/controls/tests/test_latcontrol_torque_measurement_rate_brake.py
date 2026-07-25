import numpy as np
import pytest

from openpilot.cereal import log
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol_torque import (
  ACTUATION_LATERAL_ACCEL_CORRECTION_MAX,
  ACTUATION_SPEED_PROJECTION_MAX_DELTA,
  CASCADE_ACTUATOR_CORRECTION_MAX,
  CASCADE_ACTIVATION_MIN_SPEED,
  CASCADE_CATCHUP_RATE_MAX,
  CASCADE_DESIRED_RATE_MAX,
  CASCADE_P_SCALES,
  CASCADE_P_SCALE_SPEEDS,
  CASCADE_RATE_CORRECTION_MAX,
  CATCHUP_SURGE_COOLDOWN_TIME,
  CATCHUP_SURGE_MAX_ACTIVE_TIME,
  CATCHUP_SURGE_QUALIFY_TIME,
  CATCHUP_SURGE_REARM_HEALTHY_TIME,
  CATCHUP_SURGE_RECOVERY_TIME,
  CATCHUP_SURGE_TORQUE_MAX,
  HANDOFF_TORQUE_CAP_RAMP_TIME,
  HIGH_ANGLE_BREAKOUT_APPLIED_NEUTRAL_TOL,
  HIGH_ANGLE_BREAKOUT_PROGRESS_TIME,
  HIGH_ANGLE_BREAKOUT_START_DEG,
  HIGH_ANGLE_BREAKOUT_TORQUE_MAX,
  HIGH_ANGLE_UNWIND_CONFIRM_TIME,
  HIGH_ANGLE_UNWIND_EVIDENCE_HOLD_TIME,
  HIGH_ANGLE_UNWIND_FULL_DEG,
  HIGH_ANGLE_UNWIND_FULL_SPEED,
  HIGH_ANGLE_UNWIND_FUTURE_WAIT_TIME,
  HIGH_ANGLE_UNWIND_MAX_SPEED,
  HIGH_ANGLE_UNWIND_MIN_SPEED,
  HIGH_ANGLE_UNWIND_PRESENT_RELEASE_TIME,
  HIGH_ANGLE_UNWIND_RAMP_IN_TIME,
  HIGH_ANGLE_UNWIND_START_DEG,
  REFERENCE_RATE_MIN_SPEED,
  REFERENCE_RATE_TRACKING_SCALES,
  REFERENCE_RATE_TRACKING_SPEEDS,
  UNWIND_EPISODE_OPPOSITE_TIME,
  UNWIND_TORQUE_BLEND_MAX,
  UNWIND_TORQUE_DELTA_DOWN,
  UNWIND_TORQUE_MAX,
  VERSION,
  CatchupSurgeState,
  HighAngleUnwindExitState,
  LatControlTorque,
  MeasurementRateFilter,
  SignedSteeringRateFilter,
  UnwindPhaseTracker,
  apply_high_angle_unwind_limits,
  cascade_p_scale,
  get_actuation_speed,
  get_committed_handoff_torque_cap,
  get_future_unwind_brake,
  get_high_angle_unwind_exit,
  get_projected_lateral_accel,
  get_reference_rate_cascade,
  reference_rate_tracking_speed_scale,
  should_block_undertracked_turn_in_damping,
  smoothstep,
  torque_transition_time,
)


def get_controller(car_name):
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CI = CarInterface(CP)
  return LatControlTorque(CP.as_reader(), CI, DT_CTRL), VehicleModel(CP)


class TestReferenceRateCascade:
  def test_zero_error_has_no_correction(self):
    rate, actuator, catchup, desired, error, scale = get_reference_rate_cascade(1.0, 1.0, 0.0, 0.0, 5.0)
    assert rate == 0.0
    assert actuator == 0.0
    assert catchup == 0.0
    assert desired == 1.0
    assert error == 0.0
    assert scale == pytest.approx(1.0)

  @pytest.mark.parametrize(
    ("reference_rate", "measurement_rate", "position_error", "sign"),
    [
      (1.0, 0.25, 0.0, 1.0),  # drive a wheel lagging the future path
      (0.25, 1.0, 0.0, -1.0),  # brake a wheel outrunning the future path
      (-1.0, 1.0, 0.0, -1.0),  # start a planned reversal before measured motion reverses
      (0.0, 0.0, 0.25, 1.0),  # turn position error into a catch-up rate, not raw torque
    ],
  )
  def test_tracks_reference_and_position_in_both_directions(self, reference_rate, measurement_rate, position_error, sign):
    rate, _, _, _, _, _ = get_reference_rate_cascade(reference_rate, measurement_rate, position_error, 0.0, 5.0)
    assert rate * sign > 0.0

  def test_cascade_is_bounded(self):
    positive = get_reference_rate_cascade(100.0, 0.0, 100.0, 0.0, 5.0)
    negative = get_reference_rate_cascade(-100.0, 0.0, -100.0, 0.0, 5.0)
    assert positive[0] == pytest.approx(CASCADE_RATE_CORRECTION_MAX)
    assert negative[0] == pytest.approx(-CASCADE_RATE_CORRECTION_MAX)
    assert positive[2] == pytest.approx(CASCADE_CATCHUP_RATE_MAX)
    assert negative[2] == pytest.approx(-CASCADE_CATCHUP_RATE_MAX)
    assert positive[3] == pytest.approx(CASCADE_DESIRED_RATE_MAX)
    assert negative[3] == pytest.approx(-CASCADE_DESIRED_RATE_MAX)

  def test_cascade_is_suppressed_at_standstill(self):
    result = get_reference_rate_cascade(100.0, 0.0, 100.0, 0.0, CASCADE_ACTIVATION_MIN_SPEED)
    assert result[0] == 0.0
    assert result[-1] == 0.0

  def test_applied_state_only_brakes_excess_motion(self):
    # Measured motion outruns the desired rate while applied torque still drives it.
    rate, actuator, *_ = get_reference_rate_cascade(0.0, 2.0, 0.0, 2.0, 5.0)
    assert rate < 0.0
    assert -CASCADE_ACTUATOR_CORRECTION_MAX <= actuator < 0.0

    # The same applied torque must not be opposed while the wheel lags a quick planned ramp.
    _, actuator, *_ = get_reference_rate_cascade(3.0, 1.0, 0.0, 2.0, 5.0)
    assert actuator == 0.0

  def test_all_speed_schedule(self):
    full = get_reference_rate_cascade(100.0, 0.0, 0.0, 0.0, 12.0 * CV.MPH_TO_MS)
    casual = get_reference_rate_cascade(100.0, 0.0, 0.0, 0.0, 30.0 * CV.MPH_TO_MS)
    highway = get_reference_rate_cascade(100.0, 0.0, 0.0, 0.0, 60.0 * CV.MPH_TO_MS)
    very_high = get_reference_rate_cascade(100.0, 0.0, 0.0, 0.0, 100.0 * CV.MPH_TO_MS)

    assert casual[0] == pytest.approx(full[0] * 0.50)
    assert highway[0] == pytest.approx(full[0] * 0.35)
    assert very_high[0] == pytest.approx(full[0] * 0.25)
    assert very_high[-1] == pytest.approx(0.25)

  def test_speed_schedule_is_continuous_monotonic_and_nonzero(self):
    samples = [reference_rate_tracking_speed_scale(speed) for speed in np.linspace(0.0, 50.0, 1001)]
    assert all(current <= previous for previous, current in zip(samples[:-1], samples[1:], strict=True))
    assert min(samples) == pytest.approx(REFERENCE_RATE_TRACKING_SCALES[-1])
    for speed, scale in zip(REFERENCE_RATE_TRACKING_SPEEDS, REFERENCE_RATE_TRACKING_SCALES, strict=True):
      assert reference_rate_tracking_speed_scale(speed) == pytest.approx(scale)

  def test_symmetry(self):
    positive = get_reference_rate_cascade(1.0, -1.0, 0.1, -0.5, 5.0)
    negative = get_reference_rate_cascade(-1.0, 1.0, -0.1, 0.5, 5.0)
    assert positive[:-1] == pytest.approx(tuple(-value for value in negative[:-1]))

  def test_residual_p_schedule_is_continuous_and_restores_high_speed_gain(self):
    samples = [cascade_p_scale(speed) for speed in np.linspace(0.0, 40.0, 1001)]
    assert all(current >= previous for previous, current in zip(samples[:-1], samples[1:], strict=True))
    for speed, scale in zip(CASCADE_P_SCALE_SPEEDS, CASCADE_P_SCALES, strict=True):
      assert cascade_p_scale(speed) == pytest.approx(scale)


class TestFutureUnwindBrake:
  def test_inactive_outside_future_path_unwind(self):
    p_factor, torque_blend, activation, *_ = get_future_unwind_brake(
      0.0,
      -0.8,
      -2.0,
      1.0,
      0.2,
      0.8,
      2.5,
      1.0,
    )
    assert p_factor == 1.0
    assert torque_blend == 0.0
    assert activation == 0.0

  def test_stalled_unwind_uses_reachable_target_without_waiting_for_wheel_motion(self):
    p_factor, torque_blend, activation, *_ = get_future_unwind_brake(
      1.0,
      -0.8,
      0.0,
      -2.0,
      0.5,
      0.8,
      2.5,
      1.0,
    )
    assert p_factor == pytest.approx(0.25)
    assert torque_blend == pytest.approx(UNWIND_TORQUE_BLEND_MAX)
    assert activation == pytest.approx(1.0)

  def test_applied_torque_brakes_predicted_unwind_overshoot(self):
    p_factor, torque_blend, activation, zero_time, projected_error = get_future_unwind_brake(
      1.0,
      -0.8,
      -2.0,
      1.0,
      0.2,
      0.8,
      2.5,
      1.0,
    )
    expected_zero_time = 0.8 / (UNWIND_TORQUE_DELTA_DOWN / UNWIND_TORQUE_MAX / DT_CTRL)
    assert zero_time == pytest.approx(expected_zero_time)
    assert projected_error > 0.2
    assert activation == pytest.approx(1.0)
    assert p_factor == pytest.approx(0.25)
    assert torque_blend == pytest.approx(UNWIND_TORQUE_BLEND_MAX)

  def test_confirmed_unwind_can_use_target_that_starts_the_planned_release(self):
    _, torque_blend, activation, *_ = get_future_unwind_brake(
      1.0,
      1.0,
      -2.0,
      1.0,
      0.2,
      0.8,
      2.5,
      1.0,
    )
    assert torque_blend == pytest.approx(UNWIND_TORQUE_BLEND_MAX)
    assert activation == pytest.approx(1.0)

  def test_overspeed_uses_raw_path_rate_not_catchup_target(self):
    # The raw path is nearly stationary while the wheel is moving quickly.
    # A position catch-up target in the wheel's direction must not redefine
    # that motion as on-rate for unwind braking.
    raw_path = get_future_unwind_brake(
      1.0,
      0.0,
      -4.0,
      3.0,
      0.0,
      0.0,
      2.5,
      1.0,
    )
    catchup_inflated = get_future_unwind_brake(
      1.0,
      0.0,
      -4.0,
      -1.0,
      0.0,
      0.0,
      2.5,
      1.0,
    )
    assert raw_path[2] == pytest.approx(1.0)
    assert catchup_inflated[2] == pytest.approx(0.0)

  def test_all_speed_schedule_retains_more_direct_p_at_high_speed(self):
    low_speed = get_future_unwind_brake(1.0, -0.8, -2.0, 1.0, 0.2, 0.8, 2.5, 1.0)
    high_speed = get_future_unwind_brake(1.0, -0.8, -2.0, 1.0, 0.2, 0.8, 2.5, 0.25)
    assert high_speed[0] > low_speed[0]
    assert high_speed[1] < low_speed[1]

  def test_neutral_aware_transition_explains_directional_crown_asymmetry(self):
    build_rate = 4 / 409 / DT_CTRL
    decay_rate = 7 / 409 / DT_CTRL
    right_to_neutral = torque_transition_time(-1.0, 0.25, build_rate, decay_rate)
    left_to_neutral = torque_transition_time(1.0, 0.25, build_rate, decay_rate)
    to_zero = torque_transition_time(-1.0, 0.0, build_rate, decay_rate)
    assert right_to_neutral > to_zero > left_to_neutral

    *_, right_time, _ = get_future_unwind_brake(
      1.0,
      0.25,
      0.0,
      0.0,
      0.0,
      -1.0,
      2.5,
      1.0,
      decay_rate,
      build_rate,
      0.25,
    )
    *_, left_time, _ = get_future_unwind_brake(
      1.0,
      0.25,
      0.0,
      0.0,
      0.0,
      1.0,
      2.5,
      1.0,
      decay_rate,
      build_rate,
      0.25,
    )
    assert right_time == pytest.approx(right_to_neutral)
    assert left_time == pytest.approx(left_to_neutral)


class TestHighAngleUnwindExit:
  def test_inactive_is_identity(self):
    assert get_high_angle_unwind_exit(0.8, 0.1, 1.0, 0.0) == (0.8, 0.0, 0.0)
    assert get_high_angle_unwind_exit(0.8, 0.1, 0.0, 1.0) == (0.8, 0.0, 0.0)

  def test_releases_only_old_direction_torque_without_crossing_neutral(self):
    torque, correction, old_direction = get_high_angle_unwind_exit(
      0.9,
      0.1,
      1.0,
      0.75,
    )
    assert old_direction == pytest.approx(0.8)
    assert correction == pytest.approx(-0.8 * 0.75)
    assert torque == pytest.approx(0.1 + 0.8 * 0.25)
    assert torque > 0.1

  def test_neutral_and_opposite_requests_are_unchanged(self):
    neutral = get_high_angle_unwind_exit(0.1, 0.1, 1.0, 1.0)
    torque, correction, old_direction = get_high_angle_unwind_exit(-0.9, 0.1, 1.0, 1.0)
    assert neutral == (0.1, 0.0, 0.0)
    assert torque == pytest.approx(-0.9)
    assert correction == 0.0
    assert old_direction == 0.0

  def test_crown_aware_behavior_is_symmetric(self):
    positive = get_high_angle_unwind_exit(0.9, 0.1, 1.0, 0.75)
    negative = get_high_angle_unwind_exit(-0.9, -0.1, -1.0, 0.75)
    assert positive[:2] == pytest.approx(tuple(-value for value in negative[:2]))
    assert positive[2] == pytest.approx(negative[2])

  @pytest.mark.parametrize("sign", [-1.0, 1.0])
  def test_one_sided_invariant_across_dense_inputs(self, sign):
    neutral = 0.13 * sign
    for command in np.linspace(-1.0, 1.0, 101):
      old_before = max((command - neutral) * sign, 0.0)
      for scale in np.linspace(0.0, 1.0, 21):
        output, correction, old_logged = get_high_angle_unwind_exit(command, neutral, sign, scale)
        old_after = max((output - neutral) * sign, 0.0)
        assert old_after <= old_before + 1e-12
        assert old_logged == pytest.approx(old_before if scale > 0.0 else 0.0)
        assert correction * sign <= 0.0
        if old_before == 0.0:
          assert output == pytest.approx(command)
        else:
          assert (output - neutral) * sign >= -1e-12

  @pytest.mark.parametrize("sign", [-1.0, 1.0])
  def test_latched_limit_and_breakout_are_crown_symmetric(self, sign):
    neutral = 0.13 * sign
    command = neutral + 0.8 * sign
    output, old_correction, old_direction, breakout_correction = apply_high_angle_unwind_limits(
      command,
      neutral,
      sign,
      0.25,
      0.5,
    )
    assert old_direction == pytest.approx(0.8)
    assert (output - neutral) * sign == pytest.approx(-0.5 * HIGH_ANGLE_BREAKOUT_TORQUE_MAX)
    assert old_correction * sign == pytest.approx(-0.55)
    assert breakout_correction * sign == pytest.approx(-(0.25 + 0.5 * HIGH_ANGLE_BREAKOUT_TORQUE_MAX))

  def test_existing_stronger_opposite_request_is_unchanged_by_breakout(self):
    output, old_correction, old_direction, breakout_correction = apply_high_angle_unwind_limits(
      -0.4,
      0.1,
      1.0,
      0.0,
      1.0,
    )
    assert output == pytest.approx(-0.4)
    assert old_correction == 0.0
    assert old_direction == 0.0
    assert breakout_correction == 0.0

  @pytest.mark.parametrize(("neutral", "old_sign"), [(1.0, -1.0), (-1.0, 1.0)])
  def test_breakout_never_pushes_absolute_command_outside_normalized_bounds(self, neutral, old_sign):
    output, _, _, breakout = apply_high_angle_unwind_limits(neutral, neutral, old_sign, None, 1.0)
    assert output == pytest.approx(neutral)
    assert breakout == 0.0

  @pytest.mark.parametrize("sign", [-1.0, 1.0])
  def test_dense_latched_and_breakout_limits_remain_bounded(self, sign):
    for neutral in np.linspace(-0.25, 0.25, 11):
      for command in np.linspace(-1.0, 1.0, 41):
        for old_limit in (None, 0.0, 0.2, 0.8):
          for breakout_scale in (0.0, 0.25, 1.0):
            output, old_correction, old_direction, breakout_correction = apply_high_angle_unwind_limits(
              command,
              neutral,
              sign,
              old_limit,
              breakout_scale,
            )
            assert np.isfinite(output)
            assert -1.0 <= output <= 1.0
            assert old_direction >= 0.0
            assert old_correction * sign <= 1e-12
            assert breakout_correction * sign <= 1e-12
            opposite_before = max(-(command - neutral) * sign, 0.0)
            opposite_after = max(-(output - neutral) * sign, 0.0)
            assert opposite_after + 1e-12 >= opposite_before
            if breakout_scale > 0.0 and opposite_before < breakout_scale * HIGH_ANGLE_BREAKOUT_TORQUE_MAX:
              assert opposite_after == pytest.approx(breakout_scale * HIGH_ANGLE_BREAKOUT_TORQUE_MAX)


class TestHighAngleUnwindExitState:
  @staticmethod
  def run_confirmed(
    state,
    angle=HIGH_ANGLE_UNWIND_FULL_DEG,
    speed=HIGH_ANGLE_UNWIND_FULL_SPEED,
    turn_sign=1.0,
    reference_rate=None,
  ):
    reference_rate = turn_sign if reference_rate is None else reference_rate
    samples = int(np.ceil((HIGH_ANGLE_UNWIND_CONFIRM_TIME + HIGH_ANGLE_UNWIND_PRESENT_RELEASE_TIME + HIGH_ANGLE_UNWIND_RAMP_IN_TIME) / DT_CTRL)) + 2
    return [state.update(True, angle, speed, 1.0, 1.0, True, turn_sign, reference_rate) for _ in range(samples)]

  def test_requires_persistent_future_confirmed_unwind(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    preconfirm_samples = int(np.floor(HIGH_ANGLE_UNWIND_CONFIRM_TIME / DT_CTRL)) - 1
    for _ in range(preconfirm_samples):
      assert state.update(True, HIGH_ANGLE_UNWIND_FULL_DEG, 3.0, 1.0, 1.0, True, 1.0, 1.0) == 0.0

    samples = self.run_confirmed(state)
    assert any(scale > 0.0 for scale in samples)
    assert samples[-1] == pytest.approx(1.0)

  def test_angle_schedule_is_smooth_and_stronger_at_larger_angles(self):
    moderate = HighAngleUnwindExitState(DT_CTRL)
    full = HighAngleUnwindExitState(DT_CTRL)
    moderate_scale = self.run_confirmed(moderate, angle=(HIGH_ANGLE_UNWIND_START_DEG + HIGH_ANGLE_UNWIND_FULL_DEG) / 2.0)[-1]
    full_scale = self.run_confirmed(full)[-1]
    assert 0.0 < moderate_scale < full_scale
    assert full_scale == pytest.approx(1.0)

  @pytest.mark.parametrize(
    ("angle", "expected"),
    [
      (120.0, 0.0),
      (200.0, 20.0 / 27.0),
      (220.0, 25.0 / 27.0),
      (240.0, 1.0),
      (280.0, 1.0),
    ],
  )
  def test_strengthened_angle_schedule(self, angle, expected):
    state = HighAngleUnwindExitState(DT_CTRL)
    self.run_confirmed(state, angle=angle)
    assert state.latched_scale == pytest.approx(expected)

  def test_speed_fades_to_zero_by_maximum(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    assert all(scale == 0.0 for scale in self.run_confirmed(state, speed=HIGH_ANGLE_UNWIND_MAX_SPEED))

  def test_speed_below_controller_motion_floor_never_arms(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    assert all(scale == 0.0 for scale in self.run_confirmed(state, speed=HIGH_ANGLE_UNWIND_MIN_SPEED - 0.01))

  @pytest.mark.parametrize(
    ("angle", "same_episode", "turn_sign", "reference_rate"),
    [
      (HIGH_ANGLE_UNWIND_START_DEG, True, 1.0, 1.0),
      (-HIGH_ANGLE_UNWIND_FULL_DEG, True, 1.0, 1.0),
      (HIGH_ANGLE_UNWIND_FULL_DEG, False, 1.0, 1.0),
      (HIGH_ANGLE_UNWIND_FULL_DEG, True, 0.0, 1.0),
      (HIGH_ANGLE_UNWIND_FULL_DEG, True, 1.0, -1.0),
    ],
  )
  def test_ineligible_geometry_never_arms(self, angle, same_episode, turn_sign, reference_rate):
    state = HighAngleUnwindExitState(DT_CTRL)
    samples = [state.update(True, angle, 3.0, 1.0, 1.0, same_episode, turn_sign, reference_rate) for _ in range(100)]
    assert all(scale == 0.0 for scale in samples)

  def test_route_like_negative_unwind_reaches_full_scale(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    scales = self.run_confirmed(state, angle=-336.0, speed=3.63, turn_sign=-1.0, reference_rate=-1.69)
    assert scales[-1] > 0.95
    output, correction, old_direction = get_high_angle_unwind_exit(-0.785, 0.0, -1.0, scales[-1])
    assert old_direction == pytest.approx(0.785)
    assert correction > 0.0
    assert -0.04 < output <= 0.0

  def test_single_reference_dropout_is_held_without_command_step(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    assert self.run_confirmed(state)[-1] == pytest.approx(1.0)
    dropout_scale = state.update(True, HIGH_ANGLE_UNWIND_FULL_DEG, 3.0, 0.0, 1.0, True, 1.0, 1.0)
    assert dropout_scale == pytest.approx(1.0)
    assert state.confirmed_time >= HIGH_ANGLE_UNWIND_CONFIRM_TIME

  def test_uncommitted_future_intent_expires_after_bounded_wait(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    for _ in range(30):
      state.update(True, 400.0, 3.0, 1.0, 1.0, True, 1.0, 1.0, delayed_lateral_accel=-1.2)
    assert state.future_confirmed

    for _ in range(int(np.ceil(HIGH_ANGLE_UNWIND_FUTURE_WAIT_TIME / DT_CTRL)) + 2):
      state.update(True, 400.0, 3.0, 0.0, 1.0, True, 1.0, 0.0, delayed_lateral_accel=-1.2)
    assert not state.future_confirmed
    assert state.confirmed_target_scale == 0.0

    for _ in range(30):
      state.update(True, 400.0, 3.0, 0.0, 1.0, True, 1.0, 0.0, delayed_lateral_accel=0.0)
    assert not state.release_committed
    assert state.latched_scale == 0.0

  def test_sparse_unconfirmed_future_blips_cannot_refresh_old_intent(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    for _ in range(30):
      state.update(True, 400.0, 3.0, 1.0, 1.0, True, 1.0, 1.0, delayed_lateral_accel=-1.2)
    assert state.future_confirmed

    dropout_samples = int(np.floor((HIGH_ANGLE_UNWIND_FUTURE_WAIT_TIME - 0.05) / DT_CTRL))
    for _ in range(5):
      for _ in range(dropout_samples):
        state.update(True, 400.0, 3.0, 0.0, 1.0, True, 1.0, 0.0, delayed_lateral_accel=-1.2)
      state.update(True, 400.0, 3.0, 1.0, 1.0, True, 1.0, 1.0, delayed_lateral_accel=-1.2)
    assert not state.future_confirmed
    assert state.confirmed_target_scale == 0.0

    for _ in range(30):
      state.update(True, 400.0, 3.0, 0.0, 1.0, True, 1.0, 0.0, delayed_lateral_accel=0.0)
    assert not state.release_committed

  def test_delayed_commit_cannot_exceed_current_angle_authority(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    for _ in range(30):
      state.update(True, 400.0, 3.0, 1.0, 1.0, True, 1.0, 1.0, delayed_lateral_accel=-1.2)
    assert state.confirmed_target_scale == pytest.approx(1.0)

    for _ in range(int(np.ceil(HIGH_ANGLE_UNWIND_PRESENT_RELEASE_TIME / DT_CTRL)) + 2):
      state.update(
        True,
        130.0,
        3.0,
        0.0,
        1.0,
        True,
        1.0,
        0.0,
        present_demand_rate=0.5,
        delayed_lateral_accel=-0.4,
      )
    current_angle_scale = smoothstep((130.0 - HIGH_ANGLE_UNWIND_START_DEG) / (HIGH_ANGLE_UNWIND_FULL_DEG - HIGH_ANGLE_UNWIND_START_DEG))
    assert state.release_committed
    assert state.latched_scale == pytest.approx(current_angle_scale)

  @pytest.mark.parametrize("sign", [-1.0, 1.0])
  def test_strong_present_plateau_blocks_all_release_authority(self, sign):
    state = HighAngleUnwindExitState(DT_CTRL)
    for _ in range(100):
      state.update(
        True,
        235.0 * sign,
        4.4,
        1.0,
        1.0,
        True,
        sign,
        sign,
        present_demand_rate=0.0,
        tracking_position_error=-0.25 * sign,
        delayed_lateral_accel=-1.9 * sign,
        current_lateral_accel=-1.9 * sign,
        geometric_lateral_accel=-1.9 * sign,
        applied_torque=0.8 * sign,
      )
    assert state.turn_in_guard == pytest.approx(1.0)
    assert not state.release_committed
    assert state.latched_scale == 0.0
    assert state.scale == 0.0
    assert state.old_direction_limit is None
    output, correction, old_direction, breakout = state.apply(0.9 * sign, 0.1 * sign)
    assert old_direction == pytest.approx(0.8)
    assert output == pytest.approx(0.9 * sign)
    assert correction == 0.0
    assert breakout == 0.0

  def test_brief_present_demand_rate_noise_cannot_commit_release(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    for _ in range(40):
      state.update(True, -235.0, 4.4, 1.0, 1.0, True, -1.0, -1.0, delayed_lateral_accel=1.9)
    for _ in range(int(np.floor(HIGH_ANGLE_UNWIND_PRESENT_RELEASE_TIME / DT_CTRL)) - 1):
      state.update(
        True,
        -235.0,
        4.4,
        1.0,
        1.0,
        True,
        -1.0,
        -1.0,
        present_demand_rate=-0.5,
        delayed_lateral_accel=0.6,
      )
    for _ in range(40):
      state.update(True, -235.0, 4.4, 1.0, 1.0, True, -1.0, -1.0, delayed_lateral_accel=1.9)
    assert not state.release_committed
    assert state.release_candidate_time == 0.0
    assert state.latched_scale == 0.0

  @pytest.mark.parametrize("sign", [-1.0, 1.0])
  def test_true_present_demand_decline_commits_without_wheel_motion(self, sign):
    state = HighAngleUnwindExitState(DT_CTRL)
    for _ in range(30):
      state.update(True, 400.0 * sign, 3.0, 1.0, 1.0, True, sign, sign, delayed_lateral_accel=-1.2 * sign)
    release_samples = int(np.ceil(HIGH_ANGLE_UNWIND_PRESENT_RELEASE_TIME / DT_CTRL)) + 2
    for _ in range(release_samples):
      state.update(
        True,
        400.0 * sign,
        3.0,
        1.0,
        1.0,
        True,
        sign,
        sign,
        present_demand_rate=0.4 * sign,
        tracking_position_error=-0.25 * sign,
        delayed_lateral_accel=-0.45 * sign,
        applied_torque=0.8 * sign,
        neutral_torque=0.1 * sign,
      )
    assert state.release_committed
    assert state.release_owned
    assert state.latched_scale == pytest.approx(1.0)

  def test_route_603_like_gradual_extreme_release_commits_before_angle_exit(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    for _ in range(30):
      state.update(True, -425.0, 2.9, 1.0, 1.0, True, -1.0, -1.0, delayed_lateral_accel=0.85)
    for demand in np.linspace(0.85, 0.26, 50):
      state.update(
        True,
        -425.0,
        2.9,
        0.0,
        1.0,
        True,
        -1.0,
        0.0,
        present_demand_rate=-0.8,
        delayed_lateral_accel=float(demand),
      )
    assert state.release_committed
    assert state.latched_scale == pytest.approx(1.0)

  def test_confirmed_future_intent_waits_without_torque_then_survives_preview_dropout(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    for _ in range(30):
      state.update(True, 400.0, 3.0, 1.0, 1.0, True, 1.0, 1.0, delayed_lateral_accel=-1.2)
    assert state.future_confirmed
    assert not state.release_committed
    output, correction, *_ = state.apply(0.9, 0.1)
    assert output == pytest.approx(0.9)
    assert correction == 0.0

    for _ in range(int(np.ceil(HIGH_ANGLE_UNWIND_PRESENT_RELEASE_TIME / DT_CTRL)) + 2):
      state.update(
        True,
        400.0,
        3.0,
        0.0,
        1.0,
        True,
        1.0,
        0.0,
        present_demand_rate=0.5,
        delayed_lateral_accel=-0.4,
      )
    assert state.release_committed
    assert state.latched_scale == pytest.approx(1.0)

  def test_route_577_plateau_dip_does_not_commit_release(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    for demand, demand_rate in [(1.99, 0.0)] * 30 + [(1.86, -0.3)] * 3 + [(1.9, 0.0)] * 50:
      state.update(
        True,
        -235.0,
        4.4,
        1.0,
        1.0,
        True,
        -1.0,
        -1.0,
        present_demand_rate=demand_rate,
        tracking_position_error=0.25,
        delayed_lateral_accel=demand,
        current_lateral_accel=demand,
        geometric_lateral_accel=demand,
        applied_torque=-0.78,
      )
    assert not state.release_committed
    assert state.latched_scale == 0.0
    assert state.old_direction_limit is None

  def test_release_ceiling_never_relaxes_with_falling_angle_or_rising_command(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    self.run_confirmed(state)
    previous_old = float("inf")
    for angle, command in zip((240.0, 220.0, 200.0, 160.0, 121.0, 100.0), (0.9, 1.0, 0.7, 1.0, 0.8, 1.0), strict=True):
      state.update(True, angle, 3.0, 1.0, 1.0, True, 1.0, 1.0)
      output, *_ = state.apply(command, 0.1)
      old_after = max(output - 0.1, 0.0)
      assert old_after <= previous_old + 1e-12
      previous_old = old_after

  def test_long_evidence_dropout_stops_strengthening_without_rebound(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    self.run_confirmed(state, angle=200.0)
    initial_scale = state.latched_scale
    state.apply(0.9, 0.1)
    initial_limit = state.old_direction_limit
    dropout_samples = int(np.ceil((HIGH_ANGLE_UNWIND_EVIDENCE_HOLD_TIME + DT_CTRL) / DT_CTRL))
    for _ in range(dropout_samples):
      state.update(True, HIGH_ANGLE_UNWIND_FULL_DEG, 3.0, 0.0, 1.0, True, 1.0, 1.0)
      output, *_ = state.apply(1.0, 0.1)
      assert max(output - 0.1, 0.0) <= initial_limit + 1e-12
    assert not state.evidence_held
    assert state.confirmed_time == 0.0
    assert state.latched_scale == pytest.approx(initial_scale)
    assert state.old_direction_limit == pytest.approx(initial_limit)

  def test_ownership_change_bridges_release_ceiling_without_breakout(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    self.run_confirmed(state)
    state.apply(0.9, 0.1)
    old_limit = state.old_direction_limit
    assert state.update(True, HIGH_ANGLE_UNWIND_FULL_DEG, 3.0, 1.0, 1.0, False, 1.0, 1.0) == 0.0
    output, _, _, breakout = state.apply(1.0, 0.1)
    assert max(output - 0.1, 0.0) <= old_limit + 1e-12
    assert breakout == 0.0

  def test_breakout_waits_for_applied_crown_neutral(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    samples = int(np.ceil((HIGH_ANGLE_UNWIND_CONFIRM_TIME + HIGH_ANGLE_UNWIND_PRESENT_RELEASE_TIME + HIGH_ANGLE_BREAKOUT_PROGRESS_TIME + 0.5) / DT_CTRL))
    for _ in range(samples):
      state.update(
        True,
        HIGH_ANGLE_BREAKOUT_START_DEG + 100.0,
        3.0,
        1.0,
        1.0,
        True,
        1.0,
        1.0,
        applied_torque=0.1 + HIGH_ANGLE_BREAKOUT_APPLIED_NEUTRAL_TOL + 0.01,
        neutral_torque=0.1,
      )
    assert state.breakout_scale == 0.0
    assert state.neutral_dwell_time == 0.0

  def test_extreme_stall_earns_bounded_breakout_after_neutral_progress_window(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    samples = (
      int(
        np.ceil(
          (HIGH_ANGLE_UNWIND_CONFIRM_TIME + HIGH_ANGLE_UNWIND_PRESENT_RELEASE_TIME + HIGH_ANGLE_BREAKOUT_PROGRESS_TIME + HIGH_ANGLE_UNWIND_RAMP_IN_TIME)
          / DT_CTRL
        )
      )
      + 4
    )
    for _ in range(samples):
      state.update(
        True,
        400.0,
        3.0,
        1.0,
        1.0,
        True,
        1.0,
        1.0,
        applied_torque=0.1,
        neutral_torque=0.1,
      )
    assert state.breakout_scale > 0.0
    output, _, _, breakout = state.apply(0.1, 0.1)
    assert -HIGH_ANGLE_BREAKOUT_TORQUE_MAX <= output - 0.1 < 0.0
    assert breakout < 0.0

  def test_earned_breakout_survives_future_evidence_dropout(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    for _ in range(100):
      state.update(True, 400.0, 3.0, 1.0, 1.0, True, 1.0, 1.0, applied_torque=0.1, neutral_torque=0.1)
    assert state.breakout_earned
    earned_scale = state.breakout_scale
    for _ in range(40):
      state.update(True, 400.0, 3.0, 0.0, 1.0, True, 1.0, 1.0, applied_torque=0.1, neutral_torque=0.1)
    assert state.breakout_earned
    assert not state.breakout_completed
    assert state.breakout_scale == pytest.approx(earned_scale)

  def test_newly_earned_breakout_freezes_during_short_future_dropout(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    for _ in range(200):
      state.update(True, 400.0, 3.0, 1.0, 1.0, True, 1.0, 1.0, applied_torque=0.1, neutral_torque=0.1)
      if state.breakout_earned:
        break
    assert state.breakout_earned
    earned_scale = state.breakout_scale
    catch_time = state.breakout_catch_time
    for _ in range(10):
      state.update(
        True,
        400.0,
        3.0,
        0.0,
        1.0,
        True,
        1.0,
        1.0,
        tracking_measurement_rate=1.0,
        signed_steering_rate_deg=-70.0,
        applied_torque=0.1,
        neutral_torque=0.1,
      )
    assert state.breakout_scale == pytest.approx(earned_scale)
    assert state.breakout_catch_time == pytest.approx(catch_time)

  def test_breakout_ends_only_after_sustained_measured_rate_recovery(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    for _ in range(100):
      state.update(True, 400.0, 3.0, 1.0, 1.0, True, 1.0, 1.0, applied_torque=0.1, neutral_torque=0.1)
    assert state.breakout_earned
    recovery_samples = int(np.ceil(0.15 / DT_CTRL)) + 1
    for _ in range(recovery_samples):
      state.update(
        True,
        400.0,
        3.0,
        1.0,
        1.0,
        True,
        1.0,
        1.0,
        tracking_measurement_rate=1.0,
        signed_steering_rate_deg=-70.0,
        applied_torque=0.1,
        neutral_torque=0.1,
      )
    assert state.breakout_completed
    assert state.breakout_target_scale == 0.0

  def test_sufficient_progress_or_moderate_angle_prevents_breakout(self):
    moderate = HighAngleUnwindExitState(DT_CTRL)
    for _ in range(100):
      moderate.update(True, 280.0, 3.0, 1.0, 1.0, True, 1.0, 1.0, applied_torque=0.1, neutral_torque=0.1)
    assert moderate.breakout_scale == 0.0

    progressing = HighAngleUnwindExitState(DT_CTRL)
    for _ in range(40):
      progressing.update(True, 400.0, 3.0, 1.0, 1.0, True, 1.0, 1.0, applied_torque=0.2, neutral_torque=0.1)
    for angle in np.linspace(400.0, 350.0, 40):
      progressing.update(True, angle, 3.0, 1.0, 1.0, True, 1.0, 1.0, applied_torque=0.1, neutral_torque=0.1)
    assert progressing.progress_deg >= 30.0
    assert progressing.breakout_scale == 0.0

  def test_committed_handoff_resets_immediately(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    assert self.run_confirmed(state)[-1] == pytest.approx(1.0)
    assert state.update(True, HIGH_ANGLE_UNWIND_FULL_DEG, 3.0, 1.0, 1.0, False, 1.0, 1.0) == 0.0

  def test_driver_or_inactive_reset_is_immediate(self):
    state = HighAngleUnwindExitState(DT_CTRL)
    assert self.run_confirmed(state)[-1] == pytest.approx(1.0)
    assert state.update(False, HIGH_ANGLE_UNWIND_FULL_DEG, 3.0, 1.0, 1.0, True, 1.0, 1.0) == 0.0
    assert state.confirmed_time == 0.0
    assert state.old_direction_limit is None
    assert state.breakout_scale == 0.0


class TestCatchupSurgeState:
  @staticmethod
  def turn_in_case(sign=1.0):
    high_angle_state = HighAngleUnwindExitState(DT_CTRL)
    return high_angle_state, {
      "enabled": True,
      "steering_pressed": False,
      "driver_torque": 0.0,
      "steering_angle_deg": 300.0 * sign,
      "signed_steering_rate": 5.0 * sign,
      "v_ego": 2.0,
      "delayed_lateral_accel": -0.9 * sign,
      "current_lateral_accel": -0.9 * sign,
      "measured_lateral_accel": -0.3 * sign,
      "tracking_position_error": -0.6 * sign,
      "reference_rate": -1.0 * sign,
      "tracking_measurement_rate": -0.1 * sign,
      "torque_command": 0.7 * sign,
      "neutral_torque": 0.0,
      "handoff_committed": False,
      "unwind_same_episode": False,
      "unwind_direction": 0.0,
      "high_angle_state": high_angle_state,
    }

  @staticmethod
  def unwind_case(sign=1.0, fresh_future=True):
    high_angle_state = HighAngleUnwindExitState(DT_CTRL)
    high_angle_state.peak_aligned_angle = 360.0
    high_angle_state.future_confirmed = fresh_future
    high_angle_state.present_peak_demand = 0.8
    return high_angle_state, {
      "enabled": True,
      "steering_pressed": False,
      "driver_torque": 0.0,
      "steering_angle_deg": 360.0 * sign,
      "signed_steering_rate": -5.0 * sign,
      "v_ego": 2.0,
      "delayed_lateral_accel": -0.1 * sign,
      "current_lateral_accel": -0.1 * sign,
      "measured_lateral_accel": -0.7 * sign,
      "tracking_position_error": 0.0,
      "reference_rate": 1.0 * sign,
      "tracking_measurement_rate": 0.2 * sign,
      "torque_command": -0.3 * sign,
      "neutral_torque": 0.0,
      "handoff_committed": False,
      "unwind_same_episode": True,
      "unwind_direction": sign,
      "high_angle_state": high_angle_state,
    }

  @staticmethod
  def qualify(state, case):
    output = case["torque_command"]
    for _ in range(int(round(CATCHUP_SURGE_QUALIFY_TIME / DT_CTRL))):
      output = state.update(**case)
    return output

  @pytest.mark.parametrize("sign", [-1.0, 1.0])
  def test_turn_in_qualification_timing_and_crown_symmetry(self, sign):
    state = CatchupSurgeState(DT_CTRL)
    _, case = self.turn_in_case(sign)
    for _ in range(int(round(CATCHUP_SURGE_QUALIFY_TIME / DT_CTRL)) - 1):
      assert state.update(**case) == pytest.approx(case["torque_command"])
      assert not state.active

    output = state.update(**case)
    assert state.active
    assert state.mode == state.MODE_TURN_IN
    assert state.candidate_time >= CATCHUP_SURGE_QUALIFY_TIME
    assert state.correction * sign > 0.0
    assert output == pytest.approx(case["torque_command"] + state.correction)

  @pytest.mark.parametrize(
    ("field", "value"),
    [
      ("steering_angle_deg", 119.0),
      ("current_lateral_accel", -0.49),
      ("delayed_lateral_accel", -0.49),
      ("tracking_position_error", -0.19),
      ("torque_command", -0.7),
      ("handoff_committed", True),
    ],
  )
  def test_turn_in_requires_all_structural_gates(self, field, value):
    state = CatchupSurgeState(DT_CTRL)
    _, case = self.turn_in_case()
    case[field] = value
    for _ in range(30):
      assert state.update(**case) == pytest.approx(case["torque_command"])
    assert not state.active
    assert state.correction == 0.0

  def test_turn_in_rejects_committed_high_angle_release(self):
    state = CatchupSurgeState(DT_CTRL)
    high_angle_state, case = self.turn_in_case()
    high_angle_state.release_committed = True
    for _ in range(30):
      state.update(**case)
    assert not state.active

  @pytest.mark.parametrize("sign", [-1.0, 1.0])
  def test_route98_position_undertrack_qualifies_despite_smaller_accel_gap(self, sign):
    state = CatchupSurgeState(DT_CTRL)
    high_angle_state, case = self.turn_in_case(sign)
    case.update(
      steering_angle_deg=400.0 * sign,
      delayed_lateral_accel=-0.64 * sign,
      current_lateral_accel=-0.64 * sign,
      measured_lateral_accel=-0.53 * sign,
      tracking_position_error=-0.62 * sign,
      torque_command=0.75 * sign,
    )
    high_angle_state.future_confirmed = True
    high_angle_state.turn_in_guard = 1.0
    output = self.qualify(state, case)
    assert state.active
    assert state.mode == state.MODE_TURN_IN
    assert state.position_error == pytest.approx(0.62)
    assert state.correction * sign > 0.0
    assert abs(output) > abs(case["torque_command"])

  @pytest.mark.parametrize("sign", [-1.0, 1.0])
  def test_turn_in_correction_is_bounded_and_saturates_at_normal_limit(self, sign):
    state = CatchupSurgeState(DT_CTRL)
    _, case = self.turn_in_case(sign)
    case["torque_command"] = 0.95 * sign
    outputs = [self.qualify(state, case)]
    outputs.extend(state.update(**case) for _ in range(20))
    assert max(abs(output) for output in outputs) <= 1.0
    assert max(abs(output - case["torque_command"]) for output in outputs) <= CATCHUP_SURGE_TORQUE_MAX
    assert max(abs(output) for output in outputs) == pytest.approx(1.0)

  @pytest.mark.parametrize("sign", [-1.0, 1.0])
  @pytest.mark.parametrize("fresh_future", [False, True])
  def test_unwind_accepts_fresh_future_or_strong_present_release(self, sign, fresh_future):
    state = CatchupSurgeState(DT_CTRL)
    high_angle_state, case = self.unwind_case(sign, fresh_future)
    if fresh_future:
      high_angle_state.present_peak_demand = 0.0
    output = self.qualify(state, case)
    assert state.active
    assert state.mode == state.MODE_UNWIND
    assert state.correction * sign < 0.0
    assert abs(output) > abs(case["torque_command"])

  @pytest.mark.parametrize("sign", [-1.0, 1.0])
  def test_unwind_can_begin_from_natural_crown_neutral_command(self, sign):
    state = CatchupSurgeState(DT_CTRL)
    _, case = self.unwind_case(sign)
    case["torque_command"] = 0.0
    output = self.qualify(state, case)
    assert state.active
    assert state.mode == state.MODE_UNWIND
    assert state.correction * sign < 0.0
    assert output == pytest.approx(state.correction)

  @pytest.mark.parametrize("sign", [-1.0, 1.0])
  def test_route98_extreme_low_speed_release_uses_angle_proof_with_moderate_peak_demand(self, sign):
    state = CatchupSurgeState(DT_CTRL)
    high_angle_state, case = self.unwind_case(sign, fresh_future=False)
    high_angle_state.present_peak_demand = 0.36
    high_angle_state.present_old_demand = 0.19
    output = self.qualify(state, case)
    assert state.mode == state.MODE_UNWIND
    assert state.active
    assert state.correction * sign < 0.0
    assert abs(output) > abs(case["torque_command"])

    blocked = CatchupSurgeState(DT_CTRL)
    high_angle_state.present_old_demand = 0.21
    for _ in range(30):
      blocked.update(**case)
    assert not blocked.active

  def test_catchup_and_breakout_share_one_bounded_assist_budget(self):
    state = CatchupSurgeState(DT_CTRL)
    high_angle_state, case = self.unwind_case()
    high_angle_state.breakout_scale = 0.6
    self.qualify(state, case)
    for _ in range(10):
      state.update(**case)
    breakout_reserve = HIGH_ANGLE_BREAKOUT_TORQUE_MAX * high_angle_state.breakout_scale
    assert abs(state.correction) + breakout_reserve <= CATCHUP_SURGE_TORQUE_MAX

  @pytest.mark.parametrize(
    ("field", "value"),
    [
      ("steering_angle_deg", 235.0),
      ("steering_angle_deg", 279.0),
      ("torque_command", 0.3),
      ("reference_rate", 0.35),
      ("tracking_measurement_rate", 0.8),
      ("unwind_same_episode", False),
    ],
  )
  def test_unwind_rejects_route95_and_missing_proof_gates(self, field, value):
    state = CatchupSurgeState(DT_CTRL)
    _, case = self.unwind_case()
    case[field] = value
    for _ in range(30):
      state.update(**case)
    assert not state.active

  def test_unwind_requires_peak_angle_and_rejects_model_preview_alone(self):
    for mutation in ("peak", "model_only"):
      state = CatchupSurgeState(DT_CTRL)
      high_angle_state, case = self.unwind_case()
      if mutation == "peak":
        high_angle_state.peak_aligned_angle = 299.0
      else:
        high_angle_state.present_peak_demand = 0.0
        case["measured_lateral_accel"] = case["current_lateral_accel"]
      for _ in range(30):
        state.update(**case)
      assert not state.active

  @pytest.mark.parametrize(
    ("mutation", "termination"),
    [
      ({"steering_pressed": True}, CatchupSurgeState.TERMINATION_DRIVER),
      ({"driver_torque": 100.0}, CatchupSurgeState.TERMINATION_DRIVER),
      ({"enabled": False}, CatchupSurgeState.TERMINATION_INACTIVE),
      ({"v_ego": 6.0}, CatchupSurgeState.TERMINATION_SPEED),
      ({"torque_command": -0.7}, CatchupSurgeState.TERMINATION_INTENT),
      (
        {
          "steering_angle_deg": -300.0,
          "delayed_lateral_accel": 0.9,
          "current_lateral_accel": 0.9,
          "measured_lateral_accel": 0.3,
          "tracking_position_error": 0.6,
          "reference_rate": 1.0,
          "tracking_measurement_rate": 0.1,
          "torque_command": -0.7,
        },
        CatchupSurgeState.TERMINATION_INTENT,
      ),
    ],
  )
  def test_turn_in_abort_conditions_clear_within_one_update(self, mutation, termination):
    state = CatchupSurgeState(DT_CTRL)
    _, case = self.turn_in_case()
    self.qualify(state, case)
    assert state.active
    case.update(mutation)
    output = state.update(**case)
    assert output == pytest.approx(case["torque_command"])
    assert not state.active
    assert state.scale == 0.0
    assert state.termination_reason == termination

  def test_unwind_episode_change_aborts_within_one_update(self):
    state = CatchupSurgeState(DT_CTRL)
    _, case = self.unwind_case()
    self.qualify(state, case)
    case["unwind_same_episode"] = False
    assert state.update(**case) == pytest.approx(case["torque_command"])
    assert not state.active
    assert state.termination_reason == state.TERMINATION_EPISODE

  @pytest.mark.parametrize(
    ("exit_kind", "termination"),
    [
      ("rate", CatchupSurgeState.TERMINATION_RECOVERED),
      ("position", CatchupSurgeState.TERMINATION_POSITION),
      ("progress", CatchupSurgeState.TERMINATION_PROGRESS),
      ("timeout", CatchupSurgeState.TERMINATION_TIMEOUT),
    ],
  )
  def test_recovery_position_progress_and_timeout_exit(self, exit_kind, termination):
    state = CatchupSurgeState(DT_CTRL)
    _, case = self.turn_in_case()
    self.qualify(state, case)
    if exit_kind == "rate":
      case["tracking_measurement_rate"] = -0.9
      updates = int(round(CATCHUP_SURGE_RECOVERY_TIME / DT_CTRL)) + 1
    elif exit_kind == "position":
      case["tracking_position_error"] = -0.1
      updates = 1
    elif exit_kind == "progress":
      case["steering_angle_deg"] += 10.0
      updates = 1
    else:
      updates = int(round(CATCHUP_SURGE_MAX_ACTIVE_TIME / DT_CTRL)) + 1
    for _ in range(updates):
      state.update(**case)
      if not state.pulse_active:
        break
    assert not state.pulse_active
    assert state.termination_reason == termination

  def test_cooldown_requires_healthy_rearm_before_second_pulse(self):
    state = CatchupSurgeState(DT_CTRL)
    _, case = self.turn_in_case()
    self.qualify(state, case)
    case["steering_angle_deg"] += 10.0
    state.update(**case)
    assert state.used
    assert state.cooldown == pytest.approx(CATCHUP_SURGE_COOLDOWN_TIME)

    case["steering_angle_deg"] -= 10.0
    for _ in range(int(round(CATCHUP_SURGE_COOLDOWN_TIME / DT_CTRL)) + 5):
      state.update(**case)
      assert not state.pulse_active
    assert state.used

    healthy_case = dict(case)
    healthy_case["current_lateral_accel"] = 0.0
    healthy_case["delayed_lateral_accel"] = 0.0
    healthy_case["torque_command"] = 0.0
    for _ in range(int(round(CATCHUP_SURGE_REARM_HEALTHY_TIME / DT_CTRL)) + 1):
      state.update(**healthy_case)
    assert not state.used
    self.qualify(state, case)
    assert state.pulse_active


class TestCommittedHandoffTorqueCap:
  def test_inactive_is_identity(self):
    assert get_committed_handoff_torque_cap(0.8, -0.4, 0.1, 1.0, False, 1.0) == (0.8, 0.0, 0.0, 0.0)

  @pytest.mark.parametrize("sign", [-1.0, 1.0])
  def test_full_cap_removes_only_old_direction_excess(self, sign):
    torque, correction, limit, scale = get_committed_handoff_torque_cap(
      0.9 * sign,
      0.4 * sign,
      0.1 * sign,
      sign,
      True,
      HANDOFF_TORQUE_CAP_RAMP_TIME,
    )
    assert torque == pytest.approx(0.4 * sign)
    assert correction == pytest.approx(-0.5 * sign)
    assert limit == pytest.approx(0.3)
    assert scale == pytest.approx(1.0)

  def test_cap_ramps_after_commitment(self):
    torque, correction, limit, scale = get_committed_handoff_torque_cap(
      0.9,
      0.4,
      0.1,
      1.0,
      True,
      HANDOFF_TORQUE_CAP_RAMP_TIME / 2.0,
    )
    assert scale == pytest.approx(0.5)
    assert limit == pytest.approx(0.3)
    assert correction == pytest.approx(-0.25)
    assert torque == pytest.approx(0.65)

  def test_never_opposes_new_direction_or_increases_old_torque(self):
    new_direction = get_committed_handoff_torque_cap(-0.4, -0.7, 0.1, 1.0, True, 1.0)
    below_limit = get_committed_handoff_torque_cap(0.3, 0.7, 0.1, 1.0, True, 1.0)
    assert new_direction[0] == pytest.approx(-0.4)
    assert new_direction[1] == 0.0
    assert below_limit[0] == pytest.approx(0.3)
    assert below_limit[1] == 0.0


class TestUnwindPhaseTracker:
  def test_holds_through_delivery_then_releases_smoothly(self):
    tracker = UnwindPhaseTracker(DT_CTRL)
    phase, direction, gap, *_ = tracker.update(
      True,
      1.0,
      0.02,
      -0.2,
      -0.8,
      0.2,
      -1.0,
      0.0,
    )
    assert phase == pytest.approx(1.0)
    assert direction == -1.0
    assert gap == pytest.approx(0.6)

    # Geometry has passed its instantaneous knee, but old-turn applied torque
    # still has not caught the reachable target.
    phase, *_ = tracker.update(True, 0.0, 0.02, 0.0, -0.5, 0.2, -1.0, 0.0)
    assert phase == pytest.approx(1.0)

    # Once delivery and rate have caught up, P authority returns gradually.
    phase, *_ = tracker.update(True, 0.0, 0.02, 0.2, 0.2, 0.2, 0.0, 0.0)
    assert 0.0 < phase < 1.0
    previous = phase
    phase, *_ = tracker.update(True, 0.0, 0.02, 0.2, 0.2, 0.2, 0.0, 0.0)
    assert 0.0 < phase < previous

  def test_active_episode_ignores_far_side_retrigger(self):
    tracker = UnwindPhaseTracker(DT_CTRL)
    tracker.update(True, 1.0, 0.02, -0.2, -0.8, 0.2, -1.0, 0.0)
    phase, direction, *_ = tracker.update(
      True,
      1.0,
      -0.02,
      0.2,
      0.2,
      0.2,
      0.0,
      0.0,
    )
    assert direction == -1.0
    assert phase < 1.0

  def test_opposite_geometric_maneuver_releases_delivery_hold(self):
    tracker = UnwindPhaseTracker(DT_CTRL)
    tracker.update(
      True,
      1.0,
      0.02,
      -0.2,
      -0.8,
      0.2,
      -1.0,
      0.0,
      -0.5,
    )

    samples = int(np.ceil(UNWIND_EPISODE_OPPOSITE_TIME / DT_CTRL)) + 1
    for _ in range(samples):
      phase, direction, gap, _, same_episode, opposite_time, episode_armed = tracker.update(
        True,
        1.0,
        -0.02,
        0.5,
        0.2,
        0.2,
        0.0,
        0.0,
        0.5,
      )

    assert direction == -1.0
    assert gap > 0.05
    assert not same_episode
    assert opposite_time >= UNWIND_EPISODE_OPPOSITE_TIME
    assert not episode_armed
    assert tracker.handoff_time > 0.0
    assert phase < 1.0

    # The old geometry indication must not immediately start a second episode
    # after the smooth handoff reaches zero.
    for _ in range(100):
      state = tracker.update(
        True,
        1.0,
        -0.02,
        0.5,
        0.2,
        0.2,
        0.0,
        0.0,
        0.5,
      )
      phase = state.phase
      episode_armed = state.episode_armed
    assert phase == 0.0
    assert not episode_armed

    state = tracker.update(
      True,
      0.0,
      -0.02,
      0.5,
      0.2,
      0.2,
      0.0,
      0.0,
      0.5,
    )
    assert state.episode_armed

  def test_transient_opposite_sample_does_not_transfer_episode(self):
    tracker = UnwindPhaseTracker(DT_CTRL)
    tracker.update(
      True,
      1.0,
      0.02,
      -0.2,
      -0.8,
      0.2,
      -1.0,
      0.0,
      -0.5,
      -0.5,
    )

    samples = int(np.ceil(UNWIND_EPISODE_OPPOSITE_TIME / DT_CTRL)) + 2
    for _ in range(samples):
      phase, _, gap, _, same_episode, opposite_time, _ = tracker.update(
        True,
        0.0,
        -0.02,
        0.2,
        0.5,
        0.2,
        0.0,
        0.0,
        0.5,
        -0.5,
      )

    assert phase == pytest.approx(1.0)
    assert gap == pytest.approx(0.3)
    assert same_episode
    assert opposite_time == 0.0

  def test_friction_sign_flip_on_nearly_straight_path_does_not_transfer_episode(self):
    tracker = UnwindPhaseTracker(DT_CTRL)
    tracker.update(
      True,
      1.0,
      0.02,
      -0.2,
      -0.8,
      0.2,
      -1.0,
      0.0,
      -0.5,
      -0.5,
      0.5,
      0.5,
    )

    samples = int(np.ceil(UNWIND_EPISODE_OPPOSITE_TIME / DT_CTRL)) + 2
    for _ in range(samples):
      phase, _, _, _, same_episode, opposite_time, _ = tracker.update(
        True,
        0.0,
        -0.02,
        0.2,
        0.5,
        0.2,
        0.0,
        0.0,
        0.5,
        0.5,
        -0.13,
        -0.15,
      )

    assert phase == pytest.approx(1.0)
    assert same_episode
    assert opposite_time == 0.0

  def test_symmetric_delivery_gap_holds_far_side_torque_mismatch(self):
    tracker = UnwindPhaseTracker(DT_CTRL)
    tracker.update(
      True,
      1.0,
      0.02,
      -0.2,
      -0.8,
      0.2,
      -1.0,
      0.0,
      -0.5,
      -0.5,
    )

    # Applied torque has crossed to the far side of the original turn. The old
    # directional gap was zero here despite a large mismatch to the future
    # crown-adjusted target.
    phase, _, gap, _, same_episode, *_ = tracker.update(
      True,
      0.0,
      0.0,
      0.2,
      0.5,
      0.2,
      0.0,
      0.0,
      0.2,
      -0.5,
    )
    assert gap == pytest.approx(0.3)
    assert phase == pytest.approx(1.0)
    assert same_episode

  def test_geometric_deadband_does_not_chatter_episode_ownership(self):
    tracker = UnwindPhaseTracker(DT_CTRL)
    tracker.update(
      True,
      1.0,
      0.02,
      -0.2,
      -0.8,
      0.2,
      -1.0,
      0.0,
      -0.5,
    )

    for geometric_target in (0.27, 0.13) * 20:  # +/-0.07 around the 0.20 crown-neutral torque
      phase, _, _, _, same_episode, opposite_time, _ = tracker.update(
        True,
        0.0,
        -0.02,
        0.5,
        0.2,
        0.2,
        0.0,
        0.0,
        geometric_target,
      )

    assert phase == pytest.approx(1.0)
    assert same_episode
    assert opposite_time == 0.0

  def test_driver_override_resets_phase(self):
    tracker = UnwindPhaseTracker(DT_CTRL)
    tracker.update(True, 1.0, 0.02, -0.2, -0.8, 0.2, -1.0, 0.0)
    assert tracker.update(False, 1.0, 0.02, -0.2, -0.8, 0.2, -1.0, 0.0) == (0.0, 0.0, 0.0, 0.0, False, 0.0, True)


class TestTurnInDampingGuard:
  def test_blocks_only_meaningful_same_direction_undertracking(self):
    assert should_block_undertracked_turn_in_damping(0.02, 0.2, 0.5, 0.0)
    assert not should_block_undertracked_turn_in_damping(-0.02, 0.2, 0.5, 0.0)
    assert not should_block_undertracked_turn_in_damping(0.02, 0.02, 0.5, 0.0)

  def test_future_unwind_remains_eligible_for_damping(self):
    assert not should_block_undertracked_turn_in_damping(0.02, 0.2, -0.5, 0.0)
    assert not should_block_undertracked_turn_in_damping(0.02, 0.2, 0.5, 1.0)


class TestMeasurementRateFilter:
  def test_initial_sample_and_inactive_reset_do_not_spike(self):
    rate_filter = MeasurementRateFilter(DT_CTRL)

    assert rate_filter.update(0.01, 10.0, True) == 0.0
    assert rate_filter.update(0.011, 10.0, True) > 0.0
    assert rate_filter.update(0.04, 10.0, False) == 0.0
    assert rate_filter.update(0.04, 10.0, True) == 0.0

  def test_constant_curvature_speed_change_has_no_motion_rate(self):
    rate_filter = MeasurementRateFilter(DT_CTRL)

    assert rate_filter.update(0.01, 3.0, True) == 0.0
    assert rate_filter.update(0.01, 20.0, True) == pytest.approx(0.0)

  def test_curvature_motion_is_scaled_to_lateral_acceleration_rate(self):
    slow_filter = MeasurementRateFilter(DT_CTRL)
    fast_filter = MeasurementRateFilter(DT_CTRL)
    slow_filter.update(0.0, 5.0, True)
    fast_filter.update(0.0, 10.0, True)

    slow_rate = slow_filter.update(0.001, 5.0, True)
    fast_rate = fast_filter.update(0.001, 10.0, True)
    assert slow_rate > 0.0
    assert fast_rate == pytest.approx(4.0 * slow_rate)

  def test_exposes_filtered_curvature_rate_for_common_tracking_speed(self):
    rate_filter = MeasurementRateFilter(DT_CTRL)
    rate_filter.update(0.0, 1.0, True)
    rate_filter.update(0.001, 1.0, True)
    assert rate_filter.curvature_rate > 0.0
    assert rate_filter.curvature_rate * REFERENCE_RATE_MIN_SPEED**2 > rate_filter.curvature_rate


class TestSignedSteeringRateFilter:
  @pytest.mark.parametrize("initial_angle", [-320.0, 320.0])
  def test_first_sample_does_not_spike(self, initial_angle):
    rate_filter = SignedSteeringRateFilter(DT_CTRL)
    assert rate_filter.update(initial_angle, True) == 0.0

  @pytest.mark.parametrize("sign", [-1.0, 1.0])
  def test_angle_delta_produces_crown_symmetric_signed_rate(self, sign):
    rate_filter = SignedSteeringRateFilter(DT_CTRL)
    assert rate_filter.update(200.0 * sign, True) == 0.0
    rate = rate_filter.update(201.0 * sign, True)
    assert rate * sign > 0.0

    mirrored_filter = SignedSteeringRateFilter(DT_CTRL)
    mirrored_filter.update(-200.0 * sign, True)
    mirrored_rate = mirrored_filter.update(-201.0 * sign, True)
    assert rate == pytest.approx(-mirrored_rate)

  def test_inactive_reset_prevents_reenable_spike(self):
    rate_filter = SignedSteeringRateFilter(DT_CTRL)
    rate_filter.update(200.0, True)
    assert rate_filter.update(201.0, True) > 0.0
    assert rate_filter.update(-300.0, False) == 0.0
    assert rate_filter.update(-300.0, True) == 0.0
    assert rate_filter.update(-301.0, True) < 0.0


class TestActuationSpeedProjection:
  def test_zero_acceleration_is_identity(self):
    assert get_actuation_speed(10.0, 0.0, 0.2) == pytest.approx(10.0)

  def test_acceleration_and_braking_project_in_expected_direction(self):
    assert get_actuation_speed(10.0, 2.0, 0.2) == pytest.approx(10.4)
    assert get_actuation_speed(10.0, -2.0, 0.2) == pytest.approx(9.6)

  def test_projection_is_suppressed_at_standstill(self):
    assert get_actuation_speed(0.0, 3.0, 0.2) == 0.0
    assert get_actuation_speed(0.3, 3.0, 0.2) == pytest.approx(0.3)
    assert 0.3 < get_actuation_speed(0.65, 3.0, 0.2) < 1.25

  def test_projection_is_bounded(self):
    assert get_actuation_speed(10.0, 100.0, 10.0) == pytest.approx(10.0 + ACTUATION_SPEED_PROJECTION_MAX_DELTA)
    assert get_actuation_speed(0.5, -100.0, 10.0) >= 0.0

  @pytest.mark.parametrize("desired_curvature", [-1.0, 1.0])
  def test_lateral_acceleration_correction_is_separately_bounded(self, desired_curvature):
    _, projected_lateral_accel = get_projected_lateral_accel(desired_curvature, 5.0, 3.0, 0.35)
    current_lateral_accel = desired_curvature * 5.0**2
    assert abs(projected_lateral_accel - current_lateral_accel) == pytest.approx(ACTUATION_LATERAL_ACCEL_CORRECTION_MAX)


class TestReferenceRateTrackingIntegration:
  def test_catchup_state_is_wired_to_output_diagnostics_and_damping(self, monkeypatch):
    controller, vehicle_model = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    car_state = car.CarState.new_message()
    car_state.vEgo = 2.0
    params = log.LiveParametersData.new_message()
    state = controller.catchup_surge_state

    def active_catchup_update(*args):
      torque_command = args[12]
      state.mode = state.MODE_TURN_IN
      state.active = True
      state.scale = 0.5
      state.correction = 0.05
      state.planned_rate = 1.0
      state.actual_rate = 0.1
      state.position_error = 0.6
      state.underperform = True
      return torque_command + state.correction

    monkeypatch.setattr(state, "update", active_catchup_update)
    _, _, torque_log = controller.update(True, car_state, vehicle_model, params, False, 0.0, False, 0.2)
    assert torque_log.catchupMode == state.MODE_TURN_IN
    assert torque_log.catchupActive
    assert torque_log.catchupScale == pytest.approx(0.5)
    assert torque_log.catchupCorrection == pytest.approx(0.05)
    assert torque_log.catchupPlannedRate == pytest.approx(1.0)
    assert torque_log.catchupActualRate == pytest.approx(0.1)
    assert torque_log.catchupPositionError == pytest.approx(0.6)
    assert torque_log.catchupUnderperform
    assert torque_log.dampingTurnInBlocked

  def test_high_angle_present_guard_is_included_in_damping_block(self, monkeypatch):
    controller, vehicle_model = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    car_state = car.CarState.new_message()
    car_state.vEgo = 2.0
    params = log.LiveParametersData.new_message()

    def guarded_high_angle_update(*args, **kwargs):
      controller.high_angle_unwind_exit_state.turn_in_guard = 1.0
      return 0.0

    monkeypatch.setattr(controller.high_angle_unwind_exit_state, "update", guarded_high_angle_update)
    _, _, torque_log = controller.update(True, car_state, vehicle_model, params, False, 0.0, False, 0.2)
    assert torque_log.highAngleUnwindTurnInGuard == pytest.approx(1.0)
    assert torque_log.dampingTurnInBlocked

  @pytest.mark.parametrize("steer_limited_by_safety", [False, True])
  def test_hyundai_controller_logs_tracking_attribution(self, steer_limited_by_safety):
    controller, vehicle_model = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    car_state = car.CarState.new_message()
    car_state.vEgo = 3.0
    params = log.LiveParametersData.new_message()

    controller.update(True, car_state, vehicle_model, params, steer_limited_by_safety, 0.0, False, 0.2)
    _, _, torque_log = controller.update(True, car_state, vehicle_model, params, steer_limited_by_safety, 0.01, False, 0.2)

    assert torque_log.version == VERSION
    assert torque_log.referenceRate > 0.0
    assert not torque_log.trajectoryReferenceRateValid
    assert torque_log.referenceCurvatureRate == pytest.approx(torque_log.finiteDifferenceReferenceCurvatureRate)
    assert torque_log.trackingMeasurementRate == pytest.approx(0.0)
    assert torque_log.rateTrackingError > 0.0
    assert torque_log.rateTrackingCorrection > 0.0
    assert torque_log.cascadeDesiredRate == pytest.approx(torque_log.referenceRate)
    assert torque_log.cascadeRateError > 0.0
    assert torque_log.cascadePScale == pytest.approx(0.5)
    assert torque_log.rateTrackingSpeedScale == pytest.approx(1.0)
    assert torque_log.d == pytest.approx(torque_log.rateTrackingCorrection + torque_log.actuatorStateCorrection)
    assert torque_log.rateBrake == 0.0
    assert torque_log.rateBrakeScale == 0.0
    assert torque_log.highAngleUnwindScale == 0.0
    assert torque_log.highAngleUnwindOldTorqueCorrection == 0.0
    assert torque_log.highAngleUnwindOldDirectionTorque == 0.0
    assert torque_log.highAngleUnwindLatchedScale == 0.0
    assert torque_log.highAngleUnwindOldDirectionLimit == 0.0
    assert torque_log.highAngleUnwindTurnInGuard == 0.0
    assert not torque_log.highAngleUnwindEvidenceHeld
    assert torque_log.highAngleUnwindBreakoutScale == 0.0
    assert torque_log.highAngleUnwindBreakoutCorrection == 0.0
    assert torque_log.highAngleUnwindPresentDemandRate == 0.0
    assert not torque_log.highAngleUnwindFutureConfirmed
    assert not torque_log.highAngleUnwindReleaseCommitted
    assert torque_log.highAngleUnwindReleaseCandidateTime == 0.0
    assert not torque_log.highAngleUnwindBreakoutEarned
    assert not torque_log.highAngleUnwindBreakoutCompleted
    assert not torque_log.highAngleUnwindNeutralDelivered
    assert torque_log.highAngleUnwindFutureWaitAge == 0.0

    car_state.steeringPressed = True
    _, _, torque_log = controller.update(True, car_state, vehicle_model, params, False, 0.02, False, 0.2)
    assert torque_log.rateTrackingCorrection == 0.0
    assert torque_log.rateTrackingSpeedScale == 0.0

  def test_hyundai_controller_keeps_bounded_tracking_at_high_speed(self):
    controller, vehicle_model = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    car_state = car.CarState.new_message()
    car_state.vEgo = 30.0
    params = log.LiveParametersData.new_message()

    controller.update(True, car_state, vehicle_model, params, False, 0.0, False, 0.2)
    _, _, torque_log = controller.update(True, car_state, vehicle_model, params, False, 0.01, False, 0.2)

    assert 0.25 < torque_log.rateTrackingSpeedScale < 0.35
    assert torque_log.rateTrackingCorrection > 0.0
    assert torque_log.rateTrackingCorrection == pytest.approx(CASCADE_RATE_CORRECTION_MAX * torque_log.rateTrackingSpeedScale)

  def test_hyundai_controller_uses_trajectory_rate_without_replan_spike(self):
    controller, vehicle_model = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    car_state = car.CarState.new_message()
    car_state.vEgo = 3.0
    params = log.LiveParametersData.new_message()

    controller.update(
      True,
      car_state,
      vehicle_model,
      params,
      False,
      0.0,
      False,
      0.2,
      trajectory_reference_curvature_rate=0.0,
    )
    _, _, torque_log = controller.update(
      True,
      car_state,
      vehicle_model,
      params,
      False,
      0.01,
      False,
      0.2,
      trajectory_reference_curvature_rate=0.001,
    )

    assert torque_log.trajectoryReferenceRateValid
    assert torque_log.trajectoryReferenceCurvatureRate == pytest.approx(0.001)
    assert torque_log.finiteDifferenceReferenceCurvatureRate > 0.05
    assert torque_log.trajectoryReferenceInnovation > 0.05
    assert 0.001 < torque_log.referenceCurvatureRate < 0.01
    assert torque_log.filteredTrajectoryReferenceInnovation < torque_log.trajectoryReferenceInnovation
    assert torque_log.referenceRate == pytest.approx(
      torque_log.referenceCurvatureRate * REFERENCE_RATE_MIN_SPEED**2,
    )

  def test_feedforward_projection_is_logged_separately(self):
    controller, vehicle_model = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    car_state = car.CarState.new_message()
    car_state.vEgo = 5.0
    car_state.aEgo = 2.0
    params = log.LiveParametersData.new_message()

    _, _, torque_log = controller.update(True, car_state, vehicle_model, params, False, 0.01, False, 0.2)

    assert torque_log.actuationSpeed == pytest.approx(5.4)
    assert torque_log.currentSpeedDesiredLateralAccel == pytest.approx(0.25)
    assert torque_log.speedProjectionCorrection == pytest.approx(0.01 * (5.4**2 - 5.0**2))

  def test_future_target_torque_brakes_unwind_before_applied_torque_catches_up(self):
    controller, vehicle_model = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    car_state = car.CarState.new_message()
    car_state.vEgo = 3.0
    params = log.LiveParametersData.new_message()

    controller.update(
      True, car_state, vehicle_model, params, False, 0.02, False, 0.2, applied_torque=-0.8, reference_unwind_scale=0.0, reference_target_torque=-0.8
    )
    car_state.steeringAngleDeg = -5.0
    torque, _, torque_log = controller.update(
      True,
      car_state,
      vehicle_model,
      params,
      False,
      0.01,
      False,
      0.2,
      applied_torque=-0.8,
      reference_unwind_scale=1.0,
      reference_target_torque=0.8,
    )

    assert torque_log.unwindBrakeActivation == pytest.approx(1.0)
    assert torque_log.unwindTorqueNeutralTime > 0.4
    assert torque_log.cascadePScale < torque_log.cascadeBasePScale
    assert torque_log.unwindTorqueCorrection > 0.0
    assert torque_log.unwindEffectivePhase == pytest.approx(1.0)
    assert torque_log.unwindPhaseDirection == -1.0
    assert torque_log.unwindSameEpisode
    assert torque_log.unwindOppositeTime == 0.0
    assert torque_log.unwindEpisodeArmed
    assert not torque_log.handoffCommitted
    assert torque_log.handoffTime == 0.0
    assert torque_log.handoffTorqueCorrection == 0.0
    assert 0.0 < torque < 0.8

  def test_confirmed_handoff_caps_stale_old_direction_controller_output(self):
    controller, vehicle_model = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    car_state = car.CarState.new_message()
    car_state.vEgo = 3.0
    params = log.LiveParametersData.new_message()

    controller.update(
      True,
      car_state,
      vehicle_model,
      params,
      False,
      0.02,
      False,
      0.2,
      applied_torque=-0.8,
      reference_unwind_scale=1.0,
      reference_target_torque=-0.5,
      reference_geometric_target_torque=-0.5,
      reference_episode_target_torque=-0.5,
      reference_geometric_lateral_accel=0.5,
      reference_episode_lateral_accel=0.5,
    )

    torque = 0.0
    torque_log = None
    samples = int(np.ceil(UNWIND_EPISODE_OPPOSITE_TIME / DT_CTRL)) + 2
    for _ in range(samples):
      torque, _, torque_log = controller.update(
        True,
        car_state,
        vehicle_model,
        params,
        False,
        0.02,
        False,
        0.2,
        applied_torque=-0.5,
        reference_unwind_scale=1.0,
        reference_target_torque=0.5,
        reference_geometric_target_torque=0.5,
        reference_episode_target_torque=0.5,
        reference_geometric_lateral_accel=-0.5,
        reference_episode_lateral_accel=-0.5,
      )

    assert torque_log is not None
    assert torque_log.handoffCommitted
    assert torque_log.handoffTime > 0.0
    assert torque_log.handoffTorqueCapScale > 0.0
    assert torque_log.handoffTorqueCorrection > 0.0
    assert torque == pytest.approx(torque_log.output)
    assert abs(torque) < abs(torque_log.torqueCommandBeforeHandoffCap)
