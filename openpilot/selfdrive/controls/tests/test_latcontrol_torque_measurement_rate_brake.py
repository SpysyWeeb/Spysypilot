import numpy as np

from openpilot.cereal import log
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.constants import CV
from openpilot.common.parameterized import parameterized
from openpilot.common.realtime import DT_CTRL
from openpilot.common.test import OpenpilotTestCase
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
  REFERENCE_RATE_MIN_SPEED,
  REFERENCE_RATE_TRACKING_SCALES,
  REFERENCE_RATE_TRACKING_SPEEDS,
  UNWIND_EPISODE_OPPOSITE_TIME,
  UNWIND_TORQUE_BLEND_MAX,
  UNWIND_TORQUE_DELTA_DOWN,
  UNWIND_TORQUE_MAX,
  VERSION,
  LatControlTorque,
  MeasurementRateFilter,
  UnwindPhaseTracker,
  cascade_p_scale,
  get_actuation_speed,
  get_future_unwind_brake,
  get_projected_lateral_accel,
  get_reference_rate_cascade,
  reference_rate_tracking_speed_scale,
  should_block_undertracked_turn_in_damping,
  torque_transition_time,
)


def get_controller(car_name):
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CI = CarInterface(CP)
  return LatControlTorque(CP.as_reader(), CI, DT_CTRL), VehicleModel(CP)


class ApproxTestCase(OpenpilotTestCase):
  def assertApproxEqual(self, actual, expected, *, rel=1e-6, abs_=1e-12):
    if isinstance(expected, (list, tuple)):
      self.assertEqual(len(actual), len(expected))
      for actual_value, expected_value in zip(actual, expected, strict=True):
        self.assertApproxEqual(actual_value, expected_value, rel=rel, abs_=abs_)
      return

    self.assertAlmostEqual(actual, expected, delta=max(abs_, rel * abs(expected)))


class TestReferenceRateCascade(ApproxTestCase):
  def test_zero_error_has_no_correction(self):
    rate, actuator, catchup, desired, error, scale = get_reference_rate_cascade(1.0, 1.0, 0.0, 0.0, 5.0)
    assert rate == 0.0
    assert actuator == 0.0
    assert catchup == 0.0
    assert desired == 1.0
    assert error == 0.0
    self.assertApproxEqual(scale, 1.0)

  @parameterized.expand(
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
    self.assertApproxEqual(positive[0], CASCADE_RATE_CORRECTION_MAX)
    self.assertApproxEqual(negative[0], -CASCADE_RATE_CORRECTION_MAX)
    self.assertApproxEqual(positive[2], CASCADE_CATCHUP_RATE_MAX)
    self.assertApproxEqual(negative[2], -CASCADE_CATCHUP_RATE_MAX)
    self.assertApproxEqual(positive[3], CASCADE_DESIRED_RATE_MAX)
    self.assertApproxEqual(negative[3], -CASCADE_DESIRED_RATE_MAX)

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

    self.assertApproxEqual(casual[0], full[0] * 0.50)
    self.assertApproxEqual(highway[0], full[0] * 0.35)
    self.assertApproxEqual(very_high[0], full[0] * 0.25)
    self.assertApproxEqual(very_high[-1], 0.25)

  def test_speed_schedule_is_continuous_monotonic_and_nonzero(self):
    samples = [reference_rate_tracking_speed_scale(speed) for speed in np.linspace(0.0, 50.0, 1001)]
    assert all(current <= previous for previous, current in zip(samples[:-1], samples[1:], strict=True))
    self.assertApproxEqual(min(samples), REFERENCE_RATE_TRACKING_SCALES[-1])
    for speed, scale in zip(REFERENCE_RATE_TRACKING_SPEEDS, REFERENCE_RATE_TRACKING_SCALES, strict=True):
      self.assertApproxEqual(reference_rate_tracking_speed_scale(speed), scale)

  def test_symmetry(self):
    positive = get_reference_rate_cascade(1.0, -1.0, 0.1, -0.5, 5.0)
    negative = get_reference_rate_cascade(-1.0, 1.0, -0.1, 0.5, 5.0)
    self.assertApproxEqual(positive[:-1], tuple(-value for value in negative[:-1]))

  def test_residual_p_schedule_is_continuous_and_restores_high_speed_gain(self):
    samples = [cascade_p_scale(speed) for speed in np.linspace(0.0, 40.0, 1001)]
    assert all(current >= previous for previous, current in zip(samples[:-1], samples[1:], strict=True))
    for speed, scale in zip(CASCADE_P_SCALE_SPEEDS, CASCADE_P_SCALES, strict=True):
      self.assertApproxEqual(cascade_p_scale(speed), scale)


class TestFutureUnwindBrake(ApproxTestCase):
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
    self.assertApproxEqual(p_factor, 0.25)
    self.assertApproxEqual(torque_blend, UNWIND_TORQUE_BLEND_MAX)
    self.assertApproxEqual(activation, 1.0)

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
    self.assertApproxEqual(zero_time, expected_zero_time)
    assert projected_error > 0.2
    self.assertApproxEqual(activation, 1.0)
    self.assertApproxEqual(p_factor, 0.25)
    self.assertApproxEqual(torque_blend, UNWIND_TORQUE_BLEND_MAX)

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
    self.assertApproxEqual(torque_blend, UNWIND_TORQUE_BLEND_MAX)
    self.assertApproxEqual(activation, 1.0)

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
    self.assertApproxEqual(raw_path[2], 1.0)
    self.assertApproxEqual(catchup_inflated[2], 0.0)

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
    self.assertApproxEqual(right_time, right_to_neutral)
    self.assertApproxEqual(left_time, left_to_neutral)


class TestUnwindPhaseTracker(ApproxTestCase):
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
    self.assertApproxEqual(phase, 1.0)
    assert direction == -1.0
    self.assertApproxEqual(gap, 0.6)

    # Geometry has passed its instantaneous knee, but old-turn applied torque
    # still has not caught the reachable target.
    phase, *_ = tracker.update(True, 0.0, 0.02, 0.0, -0.5, 0.2, -1.0, 0.0)
    self.assertApproxEqual(phase, 1.0)

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
    assert phase < 1.0

    # The old geometry indication must not immediately start a second episode
    # after the smooth handoff reaches zero.
    for _ in range(100):
      phase, *_, episode_armed = tracker.update(
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
    assert phase == 0.0
    assert not episode_armed

    *_, episode_armed = tracker.update(
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
    assert episode_armed

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

    self.assertApproxEqual(phase, 1.0)
    self.assertApproxEqual(gap, 0.3)
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

    self.assertApproxEqual(phase, 1.0)
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
    self.assertApproxEqual(gap, 0.3)
    self.assertApproxEqual(phase, 1.0)
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

    self.assertApproxEqual(phase, 1.0)
    assert same_episode
    assert opposite_time == 0.0

  def test_driver_override_resets_phase(self):
    tracker = UnwindPhaseTracker(DT_CTRL)
    tracker.update(True, 1.0, 0.02, -0.2, -0.8, 0.2, -1.0, 0.0)
    assert tracker.update(False, 1.0, 0.02, -0.2, -0.8, 0.2, -1.0, 0.0) == (0.0, 0.0, 0.0, 0.0, False, 0.0, True)


class TestTurnInDampingGuard(ApproxTestCase):
  def test_blocks_only_meaningful_same_direction_undertracking(self):
    assert should_block_undertracked_turn_in_damping(0.02, 0.2, 0.5, 0.0)
    assert not should_block_undertracked_turn_in_damping(-0.02, 0.2, 0.5, 0.0)
    assert not should_block_undertracked_turn_in_damping(0.02, 0.02, 0.5, 0.0)

  def test_future_unwind_remains_eligible_for_damping(self):
    assert not should_block_undertracked_turn_in_damping(0.02, 0.2, -0.5, 0.0)
    assert not should_block_undertracked_turn_in_damping(0.02, 0.2, 0.5, 1.0)


class TestMeasurementRateFilter(ApproxTestCase):
  def test_initial_sample_and_inactive_reset_do_not_spike(self):
    rate_filter = MeasurementRateFilter(DT_CTRL)

    assert rate_filter.update(0.01, 10.0, True) == 0.0
    assert rate_filter.update(0.011, 10.0, True) > 0.0
    assert rate_filter.update(0.04, 10.0, False) == 0.0
    assert rate_filter.update(0.04, 10.0, True) == 0.0

  def test_constant_curvature_speed_change_has_no_motion_rate(self):
    rate_filter = MeasurementRateFilter(DT_CTRL)

    assert rate_filter.update(0.01, 3.0, True) == 0.0
    self.assertApproxEqual(rate_filter.update(0.01, 20.0, True), 0.0)

  def test_curvature_motion_is_scaled_to_lateral_acceleration_rate(self):
    slow_filter = MeasurementRateFilter(DT_CTRL)
    fast_filter = MeasurementRateFilter(DT_CTRL)
    slow_filter.update(0.0, 5.0, True)
    fast_filter.update(0.0, 10.0, True)

    slow_rate = slow_filter.update(0.001, 5.0, True)
    fast_rate = fast_filter.update(0.001, 10.0, True)
    assert slow_rate > 0.0
    self.assertApproxEqual(fast_rate, 4.0 * slow_rate)

  def test_exposes_filtered_curvature_rate_for_common_tracking_speed(self):
    rate_filter = MeasurementRateFilter(DT_CTRL)
    rate_filter.update(0.0, 1.0, True)
    rate_filter.update(0.001, 1.0, True)
    assert rate_filter.curvature_rate > 0.0
    assert rate_filter.curvature_rate * REFERENCE_RATE_MIN_SPEED**2 > rate_filter.curvature_rate


class TestActuationSpeedProjection(ApproxTestCase):
  def test_zero_acceleration_is_identity(self):
    self.assertApproxEqual(get_actuation_speed(10.0, 0.0, 0.2), 10.0)

  def test_acceleration_and_braking_project_in_expected_direction(self):
    self.assertApproxEqual(get_actuation_speed(10.0, 2.0, 0.2), 10.4)
    self.assertApproxEqual(get_actuation_speed(10.0, -2.0, 0.2), 9.6)

  def test_projection_is_suppressed_at_standstill(self):
    assert get_actuation_speed(0.0, 3.0, 0.2) == 0.0
    self.assertApproxEqual(get_actuation_speed(0.3, 3.0, 0.2), 0.3)
    assert 0.3 < get_actuation_speed(0.65, 3.0, 0.2) < 1.25

  def test_projection_is_bounded(self):
    self.assertApproxEqual(get_actuation_speed(10.0, 100.0, 10.0), 10.0 + ACTUATION_SPEED_PROJECTION_MAX_DELTA)
    assert get_actuation_speed(0.5, -100.0, 10.0) >= 0.0

  @parameterized.expand([-1.0, 1.0])
  def test_lateral_acceleration_correction_is_separately_bounded(self, desired_curvature):
    _, projected_lateral_accel = get_projected_lateral_accel(desired_curvature, 5.0, 3.0, 0.35)
    current_lateral_accel = desired_curvature * 5.0**2
    self.assertApproxEqual(abs(projected_lateral_accel - current_lateral_accel), ACTUATION_LATERAL_ACCEL_CORRECTION_MAX)


class TestReferenceRateTrackingIntegration(ApproxTestCase):
  @parameterized.expand([False, True])
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
    self.assertApproxEqual(torque_log.referenceCurvatureRate, torque_log.finiteDifferenceReferenceCurvatureRate)
    self.assertApproxEqual(torque_log.trackingMeasurementRate, 0.0)
    assert torque_log.rateTrackingError > 0.0
    assert torque_log.rateTrackingCorrection > 0.0
    self.assertApproxEqual(torque_log.cascadeDesiredRate, torque_log.referenceRate)
    assert torque_log.cascadeRateError > 0.0
    self.assertApproxEqual(torque_log.cascadePScale, 0.5)
    self.assertApproxEqual(torque_log.rateTrackingSpeedScale, 1.0)
    self.assertApproxEqual(torque_log.d, torque_log.rateTrackingCorrection + torque_log.actuatorStateCorrection)
    assert torque_log.rateBrake == 0.0
    assert torque_log.rateBrakeScale == 0.0

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
    self.assertApproxEqual(torque_log.rateTrackingCorrection, CASCADE_RATE_CORRECTION_MAX * torque_log.rateTrackingSpeedScale)

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
    self.assertApproxEqual(torque_log.trajectoryReferenceCurvatureRate, 0.001)
    assert torque_log.finiteDifferenceReferenceCurvatureRate > 0.05
    assert torque_log.trajectoryReferenceInnovation > 0.05
    assert 0.001 < torque_log.referenceCurvatureRate < 0.01
    assert torque_log.filteredTrajectoryReferenceInnovation < torque_log.trajectoryReferenceInnovation
    self.assertApproxEqual(
      torque_log.referenceRate,
      torque_log.referenceCurvatureRate * REFERENCE_RATE_MIN_SPEED**2,
    )

  def test_feedforward_projection_is_logged_separately(self):
    controller, vehicle_model = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    car_state = car.CarState.new_message()
    car_state.vEgo = 5.0
    car_state.aEgo = 2.0
    params = log.LiveParametersData.new_message()

    _, _, torque_log = controller.update(True, car_state, vehicle_model, params, False, 0.01, False, 0.2)

    self.assertApproxEqual(torque_log.actuationSpeed, 5.4)
    self.assertApproxEqual(torque_log.currentSpeedDesiredLateralAccel, 0.25)
    self.assertApproxEqual(torque_log.speedProjectionCorrection, 0.01 * (5.4**2 - 5.0**2))

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

    self.assertApproxEqual(torque_log.unwindBrakeActivation, 1.0)
    assert torque_log.unwindTorqueNeutralTime > 0.4
    assert torque_log.cascadePScale < torque_log.cascadeBasePScale
    assert torque_log.unwindTorqueCorrection > 0.0
    self.assertApproxEqual(torque_log.unwindEffectivePhase, 1.0)
    assert torque_log.unwindPhaseDirection == -1.0
    assert torque_log.unwindSameEpisode
    assert torque_log.unwindOppositeTime == 0.0
    assert torque_log.unwindEpisodeArmed
    assert 0.0 < torque < 0.8
