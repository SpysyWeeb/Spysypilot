import math
import numpy as np
from collections import deque

from openpilot.cereal import log
from opendbc.car.lateral import FRICTION_THRESHOLD, get_friction
from opendbc.car.hyundai.values import CAR as HYUNDAI, get_platform_codes
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.rack_trajectory import PalisadeRackTrajectoryController

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
VERSION = 1


def palisade_rack_trajectory_compatible(CP) -> bool:
  if CP.carFingerprint != HYUNDAI.HYUNDAI_PALISADE:
    return False
  platform_codes = {
    code.split(b"-", 1)[0][:2]
    for firmware in CP.carFw
    for code, _ in get_platform_codes([bytes(firmware.fwVersion)])
  }
  # HYUNDAI_PALISADE is shared with the Kia Telluride; unknown firmware fails closed to stock.
  return b"LX" in platform_codes and b"ON" not in platform_codes

class LatControlTorque(LatControl):
  def __init__(self, CP, CI, dt, use_rack_trajectory=False):
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
    self.rack_trajectory = PalisadeRackTrajectoryController(dt) if use_rack_trajectory else None
    self.rack_trajectory_output = None

  def set_rack_trajectory_model(self, model, state_mono_ns):
    if self.rack_trajectory is not None:
      self.rack_trajectory.set_model(model, state_mono_ns)

  def reset(self):
    super().reset()
    self.rack_trajectory_output = None
    if self.rack_trajectory is not None:
      self.rack_trajectory.reset()

  def update_torque_parameters(self, latAccelFactor, latAccelOffset, friction):
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
    if self.rack_trajectory is not None:
      pid_log.version = 6
      output = self.rack_trajectory.update(
        active, CS, VM, params, self.torque_params, self.torque_from_lateral_accel, lat_delay, desired_curvature,
      )
      self.rack_trajectory_output = output
      if output is None:
        pid_log.active = False
        return 0.0, 0.0, pid_log
      pid_log.active = True
      pid_log.error = float(output.lateral_accel_error)
      pid_log.errorRate = float(output.rate_error_deg_s)
      pid_log.p = float(output.position_feedback_torque)
      pid_log.i = 0.0
      pid_log.d = float(output.rate_feedback_torque)
      pid_log.f = float(output.feedforward_torque)
      pid_log.output = float(output.torque)
      pid_log.actualLateralAccel = float(output.actual_lateral_accel)
      pid_log.desiredLateralAccel = float(output.desired_lateral_accel)
      pid_log.desiredLateralJerk = float(output.desired_lateral_jerk)

      pid_log.saturated = bool(self._check_saturation(
        output.saturated or self.steer_max - abs(output.torque) < 1e-3,
        CS,
        steer_limited_by_safety,
        curvature_limited,
      ))
      return output.torque, output.planned_angle_deg, pid_log

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
      output_torque = 0.0
      pid_log.active = False
    else:
      # do error correction in lateral acceleration space, convert at end to handle non-linear torque responses correctly
      pid_log.error = float(error)

      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5
      output_lataccel = self.pid.update(pid_log.error, speed=CS.vEgo, feedforward=ff, freeze_integrator=freeze_integrator)
      output_torque = self.torque_from_lateral_accel(output_lataccel, self.torque_params)

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.d = float(self.pid.d)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(-output_torque) # TODO: log lat accel?
      pid_log.actualLateralAccel = float(measurement)
      pid_log.desiredLateralAccel = float(setpoint)
      pid_log.desiredLateralJerk = float(desired_lateral_jerk)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    # TODO left is positive in this convention
    return -output_torque, 0.0, pid_log
