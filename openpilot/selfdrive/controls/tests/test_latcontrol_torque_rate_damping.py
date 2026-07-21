import pytest

from opendbc.car.structs import car
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.controlsd import get_steer_limited_by_safety
from openpilot.selfdrive.controls.lib.latcontrol_torque import (
  LOW_SPEED_RATE_DAMPING_GAIN,
  LOW_SPEED_RATE_DAMPING_MAX,
  SteeringRateDamping,
  apply_low_speed_steering_rate_damping,
)


class TestLowSpeedSteeringRateDamping:
  @pytest.mark.parametrize("torque", [-1.0, -0.3, 0.0, 0.3, 1.0])
  def test_stationary_wheel_preserves_torque(self, torque):
    assert apply_low_speed_steering_rate_damping(torque, 0.0, 5.0, False) == torque

  @pytest.mark.parametrize(("torque", "steering_rate"), [(0.8, -100.0), (-0.8, 100.0)])
  def test_wheel_moving_against_command_preserves_torque(self, torque, steering_rate):
    assert apply_low_speed_steering_rate_damping(torque, steering_rate, 5.0, False) == torque

  @pytest.mark.parametrize(("torque", "steering_rate"), [(0.8, 100.0), (-0.8, -100.0)])
  def test_safety_limited_command_preserves_torque(self, torque, steering_rate):
    assert apply_low_speed_steering_rate_damping(torque, steering_rate, 5.0, True) == torque

  @pytest.mark.parametrize(("torque", "steering_rate"), [(0.8, 100.0), (-0.8, -100.0)])
  def test_wheel_moving_with_command_is_damped(self, torque, steering_rate):
    expected_damping = min(abs(steering_rate) * LOW_SPEED_RATE_DAMPING_GAIN, LOW_SPEED_RATE_DAMPING_MAX)
    expected = (abs(torque) - expected_damping) * (1 if torque > 0 else -1)
    assert apply_low_speed_steering_rate_damping(torque, steering_rate, 5.0, False) == pytest.approx(expected)

  def test_speed_fade(self):
    torque = 0.8
    steering_rate = 100.0
    full = apply_low_speed_steering_rate_damping(torque, steering_rate, 12.0 * CV.MPH_TO_MS, False)
    midpoint = apply_low_speed_steering_rate_damping(torque, steering_rate, 13.5 * CV.MPH_TO_MS, False)
    zero = apply_low_speed_steering_rate_damping(torque, steering_rate, 15.0 * CV.MPH_TO_MS, False)

    assert full < midpoint < zero
    assert midpoint == pytest.approx((full + zero) / 2.0)
    assert zero == torque

  @pytest.mark.parametrize(("torque", "steering_rate"), [(0.1, 500.0), (-0.1, -500.0)])
  def test_damping_cannot_reverse_command(self, torque, steering_rate):
    assert apply_low_speed_steering_rate_damping(torque, steering_rate, 5.0, False) == 0.0

  def test_symmetry(self):
    positive = apply_low_speed_steering_rate_damping(0.7, 80.0, 5.0, False)
    negative = apply_low_speed_steering_rate_damping(-0.7, -80.0, 5.0, False)
    assert positive == pytest.approx(-negative)

  def test_signed_rate_filter_resets_while_inactive(self):
    damping = SteeringRateDamping(DT_CTRL)
    assert damping.update(10.0, 0.0, False) == 0.0
    assert damping.update(10.0, 0.0, True) == 0.0
    assert damping.update(11.0, 100.0, True) > 0.0
    assert damping.update(11.0, 0.0, True) == 0.0

    assert damping.update(100.0, 0.0, False) == 0.0
    assert damping.update(100.0, 0.0, True) == 0.0
    assert damping.update(99.0, 100.0, True) < 0.0


class TestAolSteeringLimitFeedback:
  def test_torque_limit_is_updated_from_lat_active(self):
    CP = car.CarParams.new_message()
    CP.steerControlType = car.CarParams.SteerControlType.torque
    CC = car.CarControl.new_message()
    CO = car.CarOutput.new_message()
    CC.enabled = False
    CC.latActive = True
    CC.actuators.torque = 0.8
    CO.actuatorsOutput.torque = 0.5
    assert get_steer_limited_by_safety(CP, CC, CO)

    CC.latActive = False
    assert not get_steer_limited_by_safety(CP, CC, CO)
