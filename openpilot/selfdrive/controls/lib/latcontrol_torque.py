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
MEASUREMENT_RATE_FILTER_TAU = 0.12
MEASUREMENT_RATE_BRAKE_GAIN = 0.18  # seconds; converts measured lateral jerk to lateral acceleration
MEASUREMENT_RATE_BRAKE_MAX = 0.16  # m/s^2
MEASUREMENT_RATE_BRAKE_OPPOSING_FULL = 0.15  # m/s^2 of controller demand opposing wheel motion
MEASUREMENT_RATE_BRAKE_FULL_SPEED = 12.0 * CV.MPH_TO_MS
MEASUREMENT_RATE_BRAKE_SPEEDS = tuple(speed * CV.MPH_TO_MS for speed in (0.0, 12.0, 15.0, 30.0, 60.0, 90.0))
MEASUREMENT_RATE_BRAKE_SCALES = (1.0, 1.0, 0.25, 0.15, 0.08, 0.05)
ACTUATION_SPEED_PROJECTION_MIN_SPEED = 0.3  # m/s; do not project from a standstill
ACTUATION_SPEED_PROJECTION_FULL_SPEED = 1.0  # m/s
ACTUATION_SPEED_PROJECTION_MAX_TIME = 0.35  # seconds
ACTUATION_SPEED_PROJECTION_MAX_DELTA = 0.75  # m/s
ACTUATION_SPEED_PROJECTION_ACCEL_MIN = -4.0  # m/s^2
ACTUATION_SPEED_PROJECTION_ACCEL_MAX = 3.0  # m/s^2
ACTUATION_LATERAL_ACCEL_CORRECTION_MAX = 0.20  # m/s^2
VERSION = 6


def smoothstep(value: float) -> float:
  value = float(np.clip(value, 0.0, 1.0))
  return value * value * (3.0 - 2.0 * value)


def measurement_rate_brake_speed_scale(v_ego: float) -> float:
  """Return the conservative all-speed gain schedule for motion feedback."""
  speed = max(v_ego, 0.0)
  for i in range(1, len(MEASUREMENT_RATE_BRAKE_SPEEDS)):
    if speed <= MEASUREMENT_RATE_BRAKE_SPEEDS[i]:
      lower_speed = MEASUREMENT_RATE_BRAKE_SPEEDS[i - 1]
      upper_speed = MEASUREMENT_RATE_BRAKE_SPEEDS[i]
      fraction = (speed - lower_speed) / (upper_speed - lower_speed)
      blend = smoothstep(fraction)
      lower_scale = MEASUREMENT_RATE_BRAKE_SCALES[i - 1]
      upper_scale = MEASUREMENT_RATE_BRAKE_SCALES[i]
      return lower_scale + blend * (upper_scale - lower_scale)
  return MEASUREMENT_RATE_BRAKE_SCALES[-1]


def get_actuation_speed(v_ego: float, a_ego: float, lat_delay: float) -> float:
  """Project speed over the bounded steering delay for feedforward only."""
  speed = max(v_ego, 0.0)
  projection_scale = smoothstep((speed - ACTUATION_SPEED_PROJECTION_MIN_SPEED) /
                                (ACTUATION_SPEED_PROJECTION_FULL_SPEED - ACTUATION_SPEED_PROJECTION_MIN_SPEED))
  projection_time = float(np.clip(lat_delay, 0.0, ACTUATION_SPEED_PROJECTION_MAX_TIME))
  acceleration = float(np.clip(a_ego, ACTUATION_SPEED_PROJECTION_ACCEL_MIN, ACTUATION_SPEED_PROJECTION_ACCEL_MAX))
  speed_delta = float(np.clip(acceleration * projection_time,
                              -ACTUATION_SPEED_PROJECTION_MAX_DELTA, ACTUATION_SPEED_PROJECTION_MAX_DELTA))
  return max(speed + projection_scale * speed_delta, 0.0)


def get_projected_lateral_accel(desired_curvature: float, v_ego: float, a_ego: float, lat_delay: float) -> tuple[float, float]:
  """Return projected speed and a separately bounded feedforward reference."""
  actuation_speed = get_actuation_speed(v_ego, a_ego, lat_delay)
  current_speed_lateral_accel = desired_curvature * v_ego ** 2
  projection_correction = desired_curvature * (actuation_speed ** 2 - v_ego ** 2)
  projection_correction = float(np.clip(projection_correction,
                                        -ACTUATION_LATERAL_ACCEL_CORRECTION_MAX, ACTUATION_LATERAL_ACCEL_CORRECTION_MAX))
  return actuation_speed, current_speed_lateral_accel + projection_correction


def get_measurement_rate_brake(output_lataccel: float, measurement_rate: float, v_ego: float) -> tuple[float, float]:
  """Return a bounded lateral-acceleration brake and its combined gate scale.

  The brake can only add to a controller command that is already opposing the
  measured lateral motion. It cannot reduce or reverse the controller's demand.
  """
  if abs(measurement_rate) < 1e-6:
    return 0.0, 0.0

  motion_sign = math.copysign(1.0, measurement_rate)
  opposing_lataccel = max(-output_lataccel * motion_sign, 0.0)
  opposing_scale = smoothstep(opposing_lataccel / MEASUREMENT_RATE_BRAKE_OPPOSING_FULL)
  brake_scale = opposing_scale * measurement_rate_brake_speed_scale(v_ego)
  brake_magnitude = min(MEASUREMENT_RATE_BRAKE_GAIN * abs(measurement_rate), MEASUREMENT_RATE_BRAKE_MAX)
  return -motion_sign * brake_magnitude * brake_scale, brake_scale


class MeasurementRateFilter:
  def __init__(self, dt: float):
    self.dt = dt
    self.filter = FirstOrderFilter(0.0, MEASUREMENT_RATE_FILTER_TAU, dt, initialized=False)
    self.previous_curvature: float | None = None

  def update(self, measured_curvature: float, v_ego: float, active: bool) -> float:
    """Return curvature-motion lateral-acceleration rate, excluding 2*v*a*kappa."""
    if not active:
      self.filter.x = measured_curvature
      self.filter.initialized = True
      self.previous_curvature = measured_curvature
      return 0.0

    filtered_curvature = float(self.filter.update(measured_curvature))
    curvature_rate = 0.0 if self.previous_curvature is None else (filtered_curvature - self.previous_curvature) / self.dt
    self.previous_curvature = filtered_curvature
    return curvature_rate * max(v_ego, 0.0) ** 2


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
    self.curvature_request_buffer = deque([0.] * self.lat_accel_request_buffer_len, maxlen=self.lat_accel_request_buffer_len)
    self.lookahead_frames = int(JERK_LOOKAHEAD_SECONDS / self.dt)
    self.jerk_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), self.dt)
    self.measurement_rate_filter = MeasurementRateFilter(self.dt)
    # Route-derived for the affected Hyundai EPS. Other torque platforms keep
    # their existing controller behavior until they are analyzed independently.
    self.measurement_rate_brake_enabled = CP.brand == "hyundai"

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
    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    measurement = measured_curvature * CS.vEgo ** 2
    measurement_rate = self.measurement_rate_filter.update(measured_curvature, CS.vEgo, active)
    longitudinal_lateral_accel_rate = 2.0 * CS.vEgo * CS.aEgo * measured_curvature
    current_speed_desired_lateral_accel = desired_curvature * CS.vEgo ** 2
    actuation_speed, future_desired_lateral_accel = get_projected_lateral_accel(desired_curvature, CS.vEgo, CS.aEgo, lat_delay)
    # Keep the delay-reference shadow and jerk path at current speed so this
    # change remains isolated to feedforward.
    self.lat_accel_request_buffer.append(current_speed_desired_lateral_accel)
    self.curvature_request_buffer.append(desired_curvature)

    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
    curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
    lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

    delay_frames = int(np.clip(lat_delay / self.dt + 1, 1, self.lat_accel_request_buffer_len))
    legacy_expected_lateral_accel = self.lat_accel_request_buffer[-delay_frames]
    delayed_desired_curvature = self.curvature_request_buffer[-delay_frames]
    # Compare delayed desired curvature and current measured curvature at the
    # same speed. Buffering lateral acceleration directly compares an old-speed
    # reference against a current-speed measurement, creating a false P error
    # whenever the driver accelerates or brakes through an otherwise unchanged
    # path.
    expected_lateral_accel = delayed_desired_curvature * CS.vEgo ** 2
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
      rate_brake = 0.0
      rate_brake_scale = 0.0
    else:
      # do error correction in lateral acceleration space, convert at end to handle non-linear torque responses correctly
      pid_log.error = float(error)

      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5
      rate_brake = 0.0
      rate_brake_scale = 0.0
      # The requested/applied torque gap represented by steer_limited_by_safety
      # is expected while Hyundai's downstream rate limiter catches up. Gating
      # on that gap suppresses rate feedback through nearly the entire release,
      # so leave downstream actuator limits authoritative and only suppress for
      # a real driver override. A decreasing gain schedule keeps the correction
      # small but nonzero above the low-speed release regime.
      rate_brake_allowed = self.measurement_rate_brake_enabled and not CS.steeringPressed
      if rate_brake_allowed:
        # Get the undamped command without moving the integrator, then add rate
        # feedback only when that command is already braking the measured motion.
        base_output_lataccel = self.pid.update(pid_log.error, speed=CS.vEgo, feedforward=ff, freeze_integrator=True)
        rate_brake, rate_brake_scale = get_measurement_rate_brake(base_output_lataccel, measurement_rate, CS.vEgo)
        output_lataccel = self.pid.update(pid_log.error, speed=CS.vEgo, feedforward=ff + rate_brake, freeze_integrator=freeze_integrator)
      else:
        # Preserve the existing single-update path exactly on other platforms
        # and during driver intervention.
        output_lataccel = self.pid.update(pid_log.error, speed=CS.vEgo, feedforward=ff, freeze_integrator=freeze_integrator)
      output_torque = self.torque_from_lateral_accel(output_lataccel, self.torque_params)
      torque_command = -output_torque

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.d = float(rate_brake)
      pid_log.f = float(self.pid.f - rate_brake)
      pid_log.output = float(torque_command) # TODO: log lat accel?
      pid_log.actualLateralAccel = float(measurement)
      pid_log.desiredLateralAccel = float(setpoint)
      pid_log.desiredLateralJerk = float(desired_lateral_jerk)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    pid_log.measurementRate = float(measurement_rate)
    pid_log.rateBrake = float(rate_brake)
    pid_log.rateBrakeScale = float(rate_brake_scale)
    pid_log.delayedDesiredCurvature = float(delayed_desired_curvature)
    pid_log.legacyDesiredLateralAccel = float(legacy_expected_lateral_accel)
    pid_log.speedAlignmentCorrection = float(expected_lateral_accel - legacy_expected_lateral_accel)
    pid_log.actuationSpeed = float(actuation_speed)
    pid_log.currentSpeedDesiredLateralAccel = float(current_speed_desired_lateral_accel)
    pid_log.speedProjectionCorrection = float(future_desired_lateral_accel - current_speed_desired_lateral_accel)
    pid_log.longitudinalLateralAccelRate = float(longitudinal_lateral_accel_rate)
    pid_log.rateBrakeSpeedScale = float(measurement_rate_brake_speed_scale(CS.vEgo))

    # TODO left is positive in this convention
    return torque_command, 0.0, pid_log
