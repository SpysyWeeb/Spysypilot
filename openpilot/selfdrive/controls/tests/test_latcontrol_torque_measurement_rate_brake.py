import pytest

from openpilot.cereal import log
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol_torque import (
  MEASUREMENT_RATE_BRAKE_FULL_SPEED,
  MEASUREMENT_RATE_BRAKE_MAX,
  MEASUREMENT_RATE_BRAKE_ZERO_SPEED,
  VERSION,
  LatControlTorque,
  MeasurementRateFilter,
  get_measurement_rate_brake,
)


def get_controller(car_name):
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CI = CarInterface(CP)
  return LatControlTorque(CP.as_reader(), CI, DT_CTRL), VehicleModel(CP)


class TestMeasurementRateBrake:
  @pytest.mark.parametrize(("output_lataccel", "measurement_rate"), [(0.5, 1.0), (-0.5, -1.0)])
  def test_preserves_command_driving_measured_motion(self, output_lataccel, measurement_rate):
    brake, scale = get_measurement_rate_brake(output_lataccel, measurement_rate, 5.0)
    assert brake == 0.0
    assert scale == 0.0

  @pytest.mark.parametrize(("output_lataccel", "measurement_rate"), [(0.5, -1.0), (-0.5, 1.0)])
  def test_augments_command_opposing_measured_motion(self, output_lataccel, measurement_rate):
    brake, scale = get_measurement_rate_brake(output_lataccel, measurement_rate, 5.0)
    assert brake * output_lataccel > 0.0
    assert brake * measurement_rate < 0.0
    assert scale == pytest.approx(1.0)

  def test_opposing_demand_gate_is_smooth_and_monotonic(self):
    weak_brake, weak_scale = get_measurement_rate_brake(-0.05, 1.0, 5.0)
    medium_brake, medium_scale = get_measurement_rate_brake(-0.10, 1.0, 5.0)
    full_brake, full_scale = get_measurement_rate_brake(-0.15, 1.0, 5.0)

    assert 0.0 < weak_scale < medium_scale < full_scale
    assert abs(weak_brake) < abs(medium_brake) < abs(full_brake)
    assert full_scale == pytest.approx(1.0)

  def test_brake_is_bounded(self):
    brake, _ = get_measurement_rate_brake(-1.0, 100.0, 5.0)
    assert brake == pytest.approx(-MEASUREMENT_RATE_BRAKE_MAX)

  def test_speed_fade(self):
    full, _ = get_measurement_rate_brake(-0.5, 1.0, MEASUREMENT_RATE_BRAKE_FULL_SPEED)
    midpoint, _ = get_measurement_rate_brake(-0.5, 1.0, 13.5 * CV.MPH_TO_MS)
    zero, scale = get_measurement_rate_brake(-0.5, 1.0, MEASUREMENT_RATE_BRAKE_ZERO_SPEED)

    assert midpoint == pytest.approx(full / 2.0)
    assert zero == 0.0
    assert scale == 0.0

  def test_symmetry(self):
    positive, _ = get_measurement_rate_brake(0.5, -1.0, 5.0)
    negative, _ = get_measurement_rate_brake(-0.5, 1.0, 5.0)
    assert positive == pytest.approx(-negative)


class TestMeasurementRateFilter:
  def test_initial_sample_and_inactive_reset_do_not_spike(self):
    rate_filter = MeasurementRateFilter(DT_CTRL)

    assert rate_filter.update(1.0, True) == 0.0
    assert rate_filter.update(1.1, True) > 0.0
    assert rate_filter.update(4.0, False) == 0.0
    assert rate_filter.update(4.0, True) == 0.0


class TestMeasurementRateBrakeIntegration:
  @pytest.mark.parametrize("steer_limited_by_safety", [False, True])
  def test_hyundai_controller_logs_brake_attribution(self, steer_limited_by_safety):
    controller, vehicle_model = get_controller(HYUNDAI.HYUNDAI_PALISADE)
    car_state = car.CarState.new_message()
    car_state.vEgo = 3.0
    params = log.LiveParametersData.new_message()

    controller.update(True, car_state, vehicle_model, params, steer_limited_by_safety, 0.0, False, 0.2)
    car_state.steeringAngleDeg = 10.0
    _, _, torque_log = controller.update(True, car_state, vehicle_model, params, steer_limited_by_safety, 0.0, False, 0.2)

    assert torque_log.version == VERSION
    assert torque_log.measurementRate < 0.0
    assert torque_log.rateBrake > 0.0
    assert torque_log.rateBrakeScale > 0.0
    assert torque_log.d == pytest.approx(torque_log.rateBrake)

    car_state.steeringPressed = True
    car_state.steeringAngleDeg = 20.0
    _, _, torque_log = controller.update(True, car_state, vehicle_model, params, False, 0.0, False, 0.2)
    assert torque_log.rateBrake == 0.0
    assert torque_log.rateBrakeScale == 0.0
