import math
import numpy as np
from collections import deque

from openpilot.cereal import log
from opendbc.car.lateral import FRICTION_THRESHOLD, get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY, CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.pid import PIDController

# At higher speeds (25+mph) we can assume:
# Lateral acceleration achieved by a specific car correlates to
# torque applied to the steering rack. It does not correlate to
# wheel slip, or to speed.

# This controller applies torque to achieve desired lateral
# accelerations. To compensate for the low speed effects the
# proportional gain is increased at low speeds by the PID controller.
# Additionally, there is friction in the steering wheel that needs
# to be overcome to move it at all, this is compensated for too.

KP = 0.8
KI = 0.15

INTERP_SPEEDS = [1, 1.5, 2.0, 3.0, 5, 7.5, 10, 15, 30]
KP_INTERP = [250, 120, 65, 30, 11.5, 5.5, 3.5, 2.0, KP]

LP_FILTER_CUTOFF_HZ = 1.2
JERK_LOOKAHEAD_SECONDS = 0.19
JERK_GAIN = 0.3
LAT_ACCEL_REQUEST_BUFFER_SECONDS = 1.0
LOW_SPEED_RATE_DAMPING_FULL_SPEED = 12.0 * CV.MPH_TO_MS
LOW_SPEED_RATE_DAMPING_ZERO_SPEED = 15.0 * CV.MPH_TO_MS
LOW_SPEED_RATE_DAMPING_GAIN = 0.002  # normalized torque per steering-wheel deg/s
LOW_SPEED_RATE_DAMPING_MAX = 0.20
STEERING_RATE_FILTER_TAU = 0.08
VERSION = 2


def apply_low_speed_steering_rate_damping(torque: float, steering_rate: float, v_ego: float, steer_limited: bool) -> float:
  """Reduce torque already accelerating the wheel, without delaying breakaway or reversing the command."""
  if steer_limited or torque * steering_rate <= 0.0:
    return torque

  speed_fraction = np.clip(
    (v_ego - LOW_SPEED_RATE_DAMPING_FULL_SPEED) / (LOW_SPEED_RATE_DAMPING_ZERO_SPEED - LOW_SPEED_RATE_DAMPING_FULL_SPEED),
    0.0,
    1.0,
  )
  speed_scale = 1.0 - speed_fraction * speed_fraction * (3.0 - 2.0 * speed_fraction)
  damping = min(abs(steering_rate) * LOW_SPEED_RATE_DAMPING_GAIN * speed_scale, LOW_SPEED_RATE_DAMPING_MAX)
  return math.copysign(max(abs(torque) - damping, 0.0), torque)


class SteeringRateDamping:
  """Build a signed, filtered rate from steering angle; Hyundai's native rate is unsigned."""

  def __init__(self, dt: float):
    self.dt = dt
    self.previous_angle: float | None = None
    self.direction = 0.0
    self.rate_filter = FirstOrderFilter(0.0, STEERING_RATE_FILTER_TAU, dt)

  def update(self, steering_angle: float, steering_rate: float, active: bool) -> float:
    if not active or self.previous_angle is None:
      self.previous_angle = steering_angle
      self.direction = 0.0
      self.rate_filter.x = 0.0
      return 0.0

    angle_delta = steering_angle - self.previous_angle
    self.previous_angle = steering_angle
    if abs(angle_delta) > 1e-3:
      self.direction = math.copysign(1.0, angle_delta)

    if abs(steering_rate) > 1e-3 and self.direction != 0.0:
      raw_rate = self.direction * abs(steering_rate)
    elif abs(angle_delta) > 1e-3:
      raw_rate = angle_delta / self.dt
    else:
      self.rate_filter.x = 0.0
      return 0.0
    return float(self.rate_filter.update(raw_rate))

class LatControlTorque(LatControl):
  def __init__(self, CP, CI, dt):
    super().__init__(CP, CI, dt)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    self.pid = PIDController([INTERP_SPEEDS, KP_INTERP], KI, rate=1/self.dt)
    self.update_limits()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    self.lat_accel_request_buffer_len = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / self.dt)
    self.lat_accel_request_buffer = deque([0.] * self.lat_accel_request_buffer_len , maxlen=self.lat_accel_request_buffer_len)
    self.lookahead_frames = int(JERK_LOOKAHEAD_SECONDS / self.dt)
    self.jerk_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), self.dt)
    self.steering_rate_damping = SteeringRateDamping(self.dt)

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction
    self.update_limits()

  def update_limits(self):
    self.pid.set_limits(self.lateral_accel_from_torque(self.steer_max, self.torque_params),
                        self.lateral_accel_from_torque(-self.steer_max, self.torque_params))

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay):
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION
    signed_steering_rate = self.steering_rate_damping.update(CS.steeringAngleDeg, CS.steeringRateDeg, active)
    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    measurement = measured_curvature * CS.vEgo ** 2
    future_desired_lateral_accel = desired_curvature * CS.vEgo ** 2
    self.lat_accel_request_buffer.append(future_desired_lateral_accel)

    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
    curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
    lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

    delay_frames = int(np.clip(lat_delay / self.dt + 1, 1, self.lat_accel_request_buffer_len))
    expected_lateral_accel = self.lat_accel_request_buffer[-delay_frames]
    setpoint = expected_lateral_accel
    error = setpoint - measurement

    lookahead_idx = int(np.clip(-delay_frames + self.lookahead_frames, -self.lat_accel_request_buffer_len+1, -2))
    raw_lateral_jerk = (self.lat_accel_request_buffer[lookahead_idx+1] - self.lat_accel_request_buffer[lookahead_idx-1]) / (2 * self.dt)
    desired_lateral_jerk = self.jerk_filter.update(raw_lateral_jerk)
    gravity_adjusted_future_lateral_accel = future_desired_lateral_accel - roll_compensation
    ff = gravity_adjusted_future_lateral_accel
    # latAccelOffset corrects roll compensation bias from device roll misalignment relative to car roll
    ff -= self.torque_params.latAccelOffset
    ff += get_friction(error + JERK_GAIN * desired_lateral_jerk, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)

    if not active:
      torque_command = 0.0
      pid_log.active = False
    else:
      # do error correction in lateral acceleration space, convert at end to handle non-linear torque responses correctly
      pid_log.error = float(error)

      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5
      output_lataccel = self.pid.update(pid_log.error, speed=CS.vEgo, feedforward=ff, freeze_integrator=freeze_integrator)
      output_torque = self.torque_from_lateral_accel(output_lataccel, self.torque_params)
      torque_command = apply_low_speed_steering_rate_damping(-output_torque, signed_steering_rate, CS.vEgo, steer_limited_by_safety)

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.d = float(self.pid.d)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(torque_command) # TODO: log lat accel?
      pid_log.actualLateralAccel = float(measurement)
      pid_log.desiredLateralAccel = float(setpoint)
      pid_log.desiredLateralJerk = float(desired_lateral_jerk)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    # TODO left is positive in this convention
    return torque_command, 0.0, pid_log
