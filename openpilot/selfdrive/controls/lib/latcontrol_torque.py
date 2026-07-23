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
# Two-degree-of-freedom cascade: the future path supplies reference motion,
# position error supplies only a bounded catch-up rate, and the inner rate loop
# supplies torque. This keeps quick planned turns while preventing raw position
# error from repeatedly commanding full torque into the Hyundai slew limiter.
CASCADE_POSITION_RATE_GAIN = 6.0  # 1/s
CASCADE_CATCHUP_RATE_MAX = 4.0  # m/s^3 at the common tracking speed
CASCADE_DESIRED_RATE_MAX = 8.0  # m/s^3 at the common tracking speed
CASCADE_RATE_GAIN = 0.35  # seconds; converts rate error to lateral acceleration
CASCADE_RATE_CORRECTION_MAX = 0.85  # m/s^2 before the speed schedule
CASCADE_ACTIVATION_MIN_SPEED = 0.3  # m/s; do not steer a stationary rack from the synthetic speed floor
CASCADE_ACTIVATION_FULL_SPEED = 1.0  # m/s
CASCADE_P_SCALE_SPEEDS = tuple(speed * CV.MPH_TO_MS for speed in (0.0, 22.0, 45.0, 67.0))
CASCADE_P_SCALES = (0.50, 0.50, 0.75, 1.0)
CASCADE_ACTUATOR_OVERSPEED_FULL = 0.8  # m/s^3
CASCADE_ACTUATOR_STATE_GAIN = 0.15
CASCADE_ACTUATOR_CORRECTION_MAX = 0.15  # m/s^2 before the speed schedule
REFERENCE_RATE_MIN_SPEED = 5.0  # m/s; retain steering-rate authority where v^2 would otherwise vanish
REFERENCE_RATE_TRACKING_SPEEDS = tuple(speed * CV.MPH_TO_MS for speed in (0.0, 12.0, 15.0, 30.0, 60.0, 90.0))
REFERENCE_RATE_TRACKING_SCALES = (1.0, 1.0, 0.65, 0.50, 0.35, 0.25)
ACTUATION_SPEED_PROJECTION_MIN_SPEED = 0.3  # m/s; do not project from a standstill
ACTUATION_SPEED_PROJECTION_FULL_SPEED = 1.0  # m/s
ACTUATION_SPEED_PROJECTION_MAX_TIME = 0.35  # seconds
ACTUATION_SPEED_PROJECTION_MAX_DELTA = 0.75  # m/s
ACTUATION_SPEED_PROJECTION_ACCEL_MIN = -4.0  # m/s^2
ACTUATION_SPEED_PROJECTION_ACCEL_MAX = 3.0  # m/s^2
ACTUATION_LATERAL_ACCEL_CORRECTION_MAX = 0.20  # m/s^2
VERSION = 9


def smoothstep(value: float) -> float:
  value = float(np.clip(value, 0.0, 1.0))
  return value * value * (3.0 - 2.0 * value)


def reference_rate_tracking_speed_scale(v_ego: float) -> float:
  """Return the all-speed gain schedule for future-reference rate tracking."""
  speed = max(v_ego, 0.0)
  for i in range(1, len(REFERENCE_RATE_TRACKING_SPEEDS)):
    if speed <= REFERENCE_RATE_TRACKING_SPEEDS[i]:
      lower_speed = REFERENCE_RATE_TRACKING_SPEEDS[i - 1]
      upper_speed = REFERENCE_RATE_TRACKING_SPEEDS[i]
      fraction = (speed - lower_speed) / (upper_speed - lower_speed)
      blend = smoothstep(fraction)
      lower_scale = REFERENCE_RATE_TRACKING_SCALES[i - 1]
      upper_scale = REFERENCE_RATE_TRACKING_SCALES[i]
      return lower_scale + blend * (upper_scale - lower_scale)
  return REFERENCE_RATE_TRACKING_SCALES[-1]


def cascade_p_scale(v_ego: float) -> float:
  """Return the residual direct-P share; most low-speed authority moves to the rate cascade."""
  speed = max(v_ego, 0.0)
  for i in range(1, len(CASCADE_P_SCALE_SPEEDS)):
    if speed <= CASCADE_P_SCALE_SPEEDS[i]:
      lower_speed = CASCADE_P_SCALE_SPEEDS[i - 1]
      upper_speed = CASCADE_P_SCALE_SPEEDS[i]
      blend = smoothstep((speed - lower_speed) / (upper_speed - lower_speed))
      return CASCADE_P_SCALES[i - 1] + blend * (CASCADE_P_SCALES[i] - CASCADE_P_SCALES[i - 1])
  return CASCADE_P_SCALES[-1]


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


def get_reference_rate_cascade(reference_rate: float, measurement_rate: float, position_error: float,
                               applied_lateral_accel: float, v_ego: float) -> tuple[float, float, float, float, float, float]:
  """Return future-path rate control and actuator-state feedback diagnostics."""
  activation_scale = smoothstep((v_ego - CASCADE_ACTIVATION_MIN_SPEED) /
                                (CASCADE_ACTIVATION_FULL_SPEED - CASCADE_ACTIVATION_MIN_SPEED))
  speed_scale = reference_rate_tracking_speed_scale(v_ego) * activation_scale
  catchup_rate = float(np.clip(CASCADE_POSITION_RATE_GAIN * position_error,
                               -CASCADE_CATCHUP_RATE_MAX, CASCADE_CATCHUP_RATE_MAX))
  desired_rate = float(np.clip(reference_rate + catchup_rate,
                               -CASCADE_DESIRED_RATE_MAX, CASCADE_DESIRED_RATE_MAX))
  rate_error = desired_rate - measurement_rate
  rate_correction = float(np.clip(CASCADE_RATE_GAIN * rate_error,
                                  -CASCADE_RATE_CORRECTION_MAX, CASCADE_RATE_CORRECTION_MAX)) * speed_scale

  # Applied torque is useful as a plant state, not as a duplicate request
  # limiter. Only oppose torque that is still driving a wheel which is already
  # outrunning the desired future-path rate. This naturally sits out a fast
  # planned turn and does not pull torque from a stalled wheel.
  actuator_correction = 0.0
  if abs(measurement_rate) > 1e-3:
    motion_sign = math.copysign(1.0, measurement_rate)
    overspeed = max(-rate_error * motion_sign, 0.0)
    applied_driving_motion = max(applied_lateral_accel * motion_sign, 0.0)
    overspeed_scale = smoothstep(overspeed / CASCADE_ACTUATOR_OVERSPEED_FULL)
    actuator_magnitude = min(CASCADE_ACTUATOR_STATE_GAIN * applied_driving_motion,
                             CASCADE_ACTUATOR_CORRECTION_MAX)
    actuator_correction = -motion_sign * actuator_magnitude * overspeed_scale * speed_scale

  return rate_correction, actuator_correction, catchup_rate, desired_rate, rate_error, speed_scale


class MeasurementRateFilter:
  def __init__(self, dt: float):
    self.dt = dt
    self.filter = FirstOrderFilter(0.0, MEASUREMENT_RATE_FILTER_TAU, dt, initialized=False)
    self.previous_curvature: float | None = None
    self.curvature_rate = 0.0

  def update(self, measured_curvature: float, v_ego: float, active: bool) -> float:
    """Return curvature-motion lateral-acceleration rate, excluding 2*v*a*kappa."""
    if not active:
      self.filter.x = measured_curvature
      self.filter.initialized = True
      self.previous_curvature = measured_curvature
      self.curvature_rate = 0.0
      return 0.0

    filtered_curvature = float(self.filter.update(measured_curvature))
    self.curvature_rate = 0.0 if self.previous_curvature is None else (filtered_curvature - self.previous_curvature) / self.dt
    self.previous_curvature = filtered_curvature
    return self.curvature_rate * max(v_ego, 0.0) ** 2


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
    self.reference_rate_filter = MeasurementRateFilter(self.dt)
    # Route-derived for the affected Hyundai EPS. Other torque platforms keep
    # their existing controller behavior until they are analyzed independently.
    self.reference_rate_tracking_enabled = CP.brand == "hyundai"

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction
    self.update_limits()

  def update_limits(self):
    self.pid.set_limits(self.lateral_accel_from_torque(self.steer_max, self.torque_params),
                        self.lateral_accel_from_torque(-self.steer_max, self.torque_params))

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay,
             applied_torque=0.0):
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION
    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    measurement = measured_curvature * CS.vEgo ** 2
    measurement_rate = self.measurement_rate_filter.update(measured_curvature, CS.vEgo, active)
    self.reference_rate_filter.update(desired_curvature, CS.vEgo, active)
    rate_reference_speed = max(CS.vEgo, REFERENCE_RATE_MIN_SPEED)
    tracking_measurement_rate = self.measurement_rate_filter.curvature_rate * rate_reference_speed ** 2
    reference_rate = self.reference_rate_filter.curvature_rate * rate_reference_speed ** 2
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
    tracking_position_error = (delayed_desired_curvature - measured_curvature) * rate_reference_speed ** 2
    applied_lateral_accel = -applied_torque * self.torque_params.latAccelFactor

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
      rate_tracking_correction = 0.0
      actuator_state_correction = 0.0
      rate_tracking_scale = 0.0
      catchup_rate = 0.0
      cascade_desired_rate = 0.0
      cascade_rate_error = 0.0
      direct_p_scale = cascade_p_scale(CS.vEgo) if self.reference_rate_tracking_enabled else 1.0
    else:
      # do error correction in lateral acceleration space, convert at end to handle non-linear torque responses correctly
      pid_log.error = float(error)

      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5
      rate_tracking_correction = 0.0
      actuator_state_correction = 0.0
      rate_tracking_scale = 0.0
      catchup_rate = 0.0
      cascade_desired_rate = reference_rate
      cascade_rate_error = reference_rate - tracking_measurement_rate
      direct_p_scale = cascade_p_scale(CS.vEgo) if self.reference_rate_tracking_enabled else 1.0
      if self.reference_rate_tracking_enabled and not CS.steeringPressed:
        cascade_values = get_reference_rate_cascade(reference_rate, tracking_measurement_rate, tracking_position_error,
                                                     applied_lateral_accel, CS.vEgo)
        rate_tracking_correction, actuator_state_correction, catchup_rate, cascade_desired_rate, cascade_rate_error, rate_tracking_scale = cascade_values
      controller_error = pid_log.error * direct_p_scale
      cascade_correction = rate_tracking_correction + actuator_state_correction
      output_lataccel = self.pid.update(controller_error, speed=CS.vEgo, feedforward=ff + cascade_correction,
                                       freeze_integrator=freeze_integrator)
      output_torque = self.torque_from_lateral_accel(output_lataccel, self.torque_params)
      torque_command = -output_torque

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.d = float(cascade_correction)
      pid_log.f = float(self.pid.f - cascade_correction)
      pid_log.output = float(torque_command) # TODO: log lat accel?
      pid_log.actualLateralAccel = float(measurement)
      pid_log.desiredLateralAccel = float(setpoint)
      pid_log.desiredLateralJerk = float(desired_lateral_jerk)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    pid_log.measurementRate = float(measurement_rate)
    # Retain the old fields as explicit zeros so mixed-version route tooling
    # cannot mistake reference-rate tracking for the retired motion-only brake.
    pid_log.rateBrake = 0.0
    pid_log.rateBrakeScale = 0.0
    pid_log.delayedDesiredCurvature = float(delayed_desired_curvature)
    pid_log.legacyDesiredLateralAccel = float(legacy_expected_lateral_accel)
    pid_log.speedAlignmentCorrection = float(expected_lateral_accel - legacy_expected_lateral_accel)
    pid_log.actuationSpeed = float(actuation_speed)
    pid_log.currentSpeedDesiredLateralAccel = float(current_speed_desired_lateral_accel)
    pid_log.speedProjectionCorrection = float(future_desired_lateral_accel - current_speed_desired_lateral_accel)
    pid_log.longitudinalLateralAccelRate = float(longitudinal_lateral_accel_rate)
    pid_log.rateBrakeSpeedScale = 0.0
    pid_log.referenceRate = float(reference_rate)
    pid_log.trackingMeasurementRate = float(tracking_measurement_rate)
    pid_log.rateTrackingError = float(reference_rate - tracking_measurement_rate)
    pid_log.rateTrackingCorrection = float(rate_tracking_correction)
    pid_log.rateTrackingSpeedScale = float(rate_tracking_scale)
    pid_log.referenceCurvatureRate = float(self.reference_rate_filter.curvature_rate)
    pid_log.measurementCurvatureRate = float(self.measurement_rate_filter.curvature_rate)
    pid_log.cascadePositionError = float(tracking_position_error)
    pid_log.cascadeCatchupRate = float(catchup_rate)
    pid_log.cascadeDesiredRate = float(cascade_desired_rate)
    pid_log.cascadeRateError = float(cascade_rate_error)
    pid_log.actuatorAppliedLateralAccel = float(applied_lateral_accel)
    pid_log.actuatorStateCorrection = float(actuator_state_correction)
    pid_log.cascadePScale = float(direct_p_scale)

    # TODO left is positive in this convention
    return torque_command, 0.0, pid_log
