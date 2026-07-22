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
  REFERENCE_RATE_MIN_SPEED,
  REFERENCE_RATE_TRACKING_MAX,
  REFERENCE_RATE_TRACKING_SCALES,
  REFERENCE_RATE_TRACKING_SPEEDS,
  VERSION,
  LatControlTorque,
  MeasurementRateFilter,
  get_actuation_speed,
  get_projected_lateral_accel,
  get_reference_rate_tracking,
  reference_rate_tracking_speed_scale,
)


def get_controller(car_name):
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CI = CarInterface(CP)
  return LatControlTorque(CP.as_reader(), CI, DT_CTRL), VehicleModel(CP)


class TestReferenceRateTracking:
  def test_zero_error_has_no_correction(self):
    correction, scale = get_reference_rate_tracking(1.0, 1.0, 5.0)
    assert correction == 0.0
    assert scale == pytest.approx(1.0)

  @pytest.mark.parametrize(("reference_rate", "measurement_rate", "sign"), [
    (1.0, 0.25, 1.0),    # drive a wheel lagging the future path
    (0.25, 1.0, -1.0),   # brake a wheel outrunning the future path
    (-1.0, 1.0, -1.0),   # start a planned reversal without waiting for measured motion to reverse
  ])
  def test_tracks_reference_in_both_directions(self, reference_rate, measurement_rate, sign):
    correction, _ = get_reference_rate_tracking(reference_rate, measurement_rate, 5.0)
    assert correction * sign > 0.0

  def test_correction_is_bounded(self):
    positive, _ = get_reference_rate_tracking(100.0, 0.0, 5.0)
    negative, _ = get_reference_rate_tracking(-100.0, 0.0, 5.0)
    assert positive == pytest.approx(REFERENCE_RATE_TRACKING_MAX)
    assert negative == pytest.approx(-REFERENCE_RATE_TRACKING_MAX)

  def test_all_speed_schedule(self):
    full, _ = get_reference_rate_tracking(100.0, 0.0, 12.0 * CV.MPH_TO_MS)
    casual, _ = get_reference_rate_tracking(100.0, 0.0, 30.0 * CV.MPH_TO_MS)
    highway, _ = get_reference_rate_tracking(100.0, 0.0, 60.0 * CV.MPH_TO_MS)
    very_high, scale = get_reference_rate_tracking(100.0, 0.0, 100.0 * CV.MPH_TO_MS)

    assert casual == pytest.approx(full * 0.50)
    assert highway == pytest.approx(full * 0.35)
    assert very_high == pytest.approx(full * 0.25)
    assert scale == pytest.approx(0.25)

  def test_speed_schedule_is_continuous_monotonic_and_nonzero(self):
    samples = [reference_rate_tracking_speed_scale(speed) for speed in np.linspace(0.0, 50.0, 1001)]
    assert all(current <= previous for previous, current in zip(samples[:-1], samples[1:], strict=True))
    assert min(samples) == pytest.approx(REFERENCE_RATE_TRACKING_SCALES[-1])
    for speed, scale in zip(REFERENCE_RATE_TRACKING_SPEEDS, REFERENCE_RATE_TRACKING_SCALES, strict=True):
      assert reference_rate_tracking_speed_scale(speed) == pytest.approx(scale)

  def test_symmetry(self):
    positive, _ = get_reference_rate_tracking(1.0, -1.0, 5.0)
    negative, _ = get_reference_rate_tracking(-1.0, 1.0, 5.0)
    assert positive == pytest.approx(-negative)


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
    assert torque_log.trackingMeasurementRate == pytest.approx(0.0)
    assert torque_log.rateTrackingError > 0.0
    assert torque_log.rateTrackingCorrection > 0.0
    assert torque_log.rateTrackingSpeedScale == pytest.approx(1.0)
    assert torque_log.d == pytest.approx(torque_log.rateTrackingCorrection)
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
    assert torque_log.rateTrackingCorrection == pytest.approx(REFERENCE_RATE_TRACKING_MAX * torque_log.rateTrackingSpeedScale)

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
