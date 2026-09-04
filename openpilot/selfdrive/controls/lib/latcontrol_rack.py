from opendbc.car.hyundai.values import CarControllerParams
from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.rack_trajectory import (
  DRIVER_ASSIST_CEILING, DriverAssistLimits, INACTIVE_HOLD_FRAMES, STATUS_STALE_MODEL, RackTrajectoryController,
)

# Executes the model path as a planned rack motion (see rack_trajectory.py) and tracks it with
# torque. A stock torque controller is stepped alongside every frame, so any frame the rack
# controller cannot produce a request for (no or stale model, invalid path, infeasible plan)
# is steered by stock instead of dropping torque. Its request buffer and jerk filter follow the
# live history; its integrator starts clean when it takes over, and the two controllers share
# one steering saturation timer. Once stock has taken over because the model went stale it keeps
# steering for a hold time, so the two controllers cannot trade places every frame around the
# staleness threshold; a one-frame content fault hands back on the next good frame.

VERSION = 1
FALLBACK_HOLD_S = 0.5


class LatControlRack(LatControl):
  def __init__(self, CP, CI, dt):
    super().__init__(CP, CI, dt)
    self.torque = LatControlTorque(CP, CI, dt)
    # driver-assist agreement relaxation (docs/BLaTv3_FAILURE_MODES.md FM4.9): build the platform's
    # own driver-override limits here (this class already holds CP/CI) and hand the rack controller
    # only the four constants it needs, as plain floats -- rack_trajectory.py never imports
    # opendbc.car.hyundai.
    limits = CarControllerParams(CP)
    self.rack = RackTrajectoryController(dt, driver_assist_limits=DriverAssistLimits(
      STEER_MAX=float(limits.STEER_MAX),
      STEER_DRIVER_ALLOWANCE=float(limits.STEER_DRIVER_ALLOWANCE),
      STEER_DRIVER_MULTIPLIER=float(limits.STEER_DRIVER_MULTIPLIER),
      STEER_DRIVER_FACTOR=float(limits.STEER_DRIVER_FACTOR),
    ))
    self.fallback_hold_frames = int(FALLBACK_HOLD_S / dt)
    self.fallback_frames = 0
    self.output = None

  def update_torque_parameters(self, latAccelFactor, latAccelOffset, friction):
    self.torque.update_torque_parameters(latAccelFactor, latAccelOffset, friction)

  def reset(self):
    super().reset()
    self.torque.reset()
    self.rack.hold()
    if self.rack.inactive_frames > INACTIVE_HOLD_FRAMES:
      self.fallback_frames = 0
    self.output = None

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay,
             model=None, mono_time_ns=0, applied_torque=0.0):
    stock_torque, _, stock_log = self.torque.update(active, CS, VM, params, steer_limited_by_safety, desired_curvature,
                                                    curvature_limited, lat_delay)
    self.rack.set_model(model, mono_time_ns)
    if not active:
      self.output = None
    elif self.fallback_frames > 0:
      # stock keeps steering through the hold; the rack controller re-seeds when it resumes
      self.fallback_frames -= 1
      self.output = None
    else:
      self.output = self.rack.update(active, CS, VM, params, self.torque.torque_params, self.torque.torque_from_lateral_accel,
                                     lat_delay, desired_curvature, applied_torque=applied_torque)
      if self.output is None and self.rack.status == STATUS_STALE_MODEL:
        self.fallback_frames = self.fallback_hold_frames

    rack_log = log.ControlsState.LateralRackState.new_message()
    rack_log.version = VERSION
    rack_log.status = self.rack.status
    if self.output is None:
      # steered by the stock controller this frame
      self.sat_time = self.torque.sat_time
      rack_log.active = stock_log.active
      rack_log.fallback = bool(active)
      rack_log.error = stock_log.error
      rack_log.p = stock_log.p
      rack_log.d = stock_log.d
      rack_log.f = stock_log.f
      rack_log.output = stock_log.output
      rack_log.actualLateralAccel = stock_log.actualLateralAccel
      rack_log.desiredLateralAccel = stock_log.desiredLateralAccel
      rack_log.desiredLateralJerk = stock_log.desiredLateralJerk
      rack_log.saturated = stock_log.saturated
      # no rack-computed cap applies this frame (stock is steering); log the ceiling rather than
      # the Float32 default of 0.0, which would misread as "capped to zero" instead of "not active"
      rack_log.driverAssistCap = DRIVER_ASSIST_CEILING
      return stock_torque, 0.0, rack_log

    # the stock controller is idle while the rack controller steers; its integrator starts clean if it takes over
    self.torque.pid.reset()
    output = self.output
    rack_log.active = True
    rack_log.error = float(output.lateral_accel_error)
    rack_log.errorRate = float(output.rate_error_deg_s)
    rack_log.p = float(output.position_feedback_torque)
    rack_log.d = float(output.rate_feedback_torque)
    rack_log.f = float(output.feedforward_torque)
    rack_log.output = float(output.torque)
    rack_log.actualLateralAccel = float(output.actual_lateral_accel)
    rack_log.desiredLateralAccel = float(output.desired_lateral_accel)
    rack_log.desiredLateralJerk = float(output.desired_lateral_jerk)
    rack_log.targetCurvature = float(output.target_curvature)
    rack_log.targetSteeringAngleDeg = float(output.target_angle_deg)
    rack_log.targetSteeringRateDegS = float(output.target_rate_deg_s)
    rack_log.plannedSteeringAngleDeg = float(output.planned_angle_deg)
    rack_log.plannedSteeringRateDegS = float(output.planned_rate_deg_s)
    rack_log.plannedSteeringAccelerationDegS2 = float(output.planned_acceleration_deg_s2)
    rack_log.measuredSteeringRateDegS = float(output.measured_rate_deg_s)
    rack_log.rateLimitDegS = float(output.rate_limit_deg_s)
    rack_log.accelerationLimitDegS2 = float(output.acceleration_limit_deg_s2)
    rack_log.jerkLimitDegS3 = float(output.jerk_limit_deg_s3)
    rack_log.feedbackLimited = bool(output.feedback_limited)
    rack_log.motionLimited = bool(output.motion_limited)
    rack_log.torqueLimited = bool(output.torque_limited)
    rack_log.pathLimited = bool(output.path_limited)
    rack_log.profileTransition = bool(output.profile_transition)
    rack_log.previewTime = float(output.preview_time_s)
    rack_log.referenceLimited = bool(output.reference_limited)
    rack_log.nearSteeringAngleDeg = float(output.near_target_angle_deg)
    rack_log.directionGuarded = bool(output.direction_guarded)
    rack_log.driverAssistLimited = bool(output.driver_assist_limited)
    rack_log.driverAssistCap = float(output.driver_assist_cap)
    rack_log.earlyRelease = bool(output.early_release)
    rack_log.directionFraction = float(output.direction_fraction)
    rack_log.envelopeRateDegS = float(output.envelope_open_rate_deg_s)
    rack_log.envelopeAccelerationDegS2 = float(output.envelope_open_acceleration_deg_s2)
    rack_log.envelopeJerkDegS3 = float(output.envelope_open_jerk_deg_s3)
    rack_log.envelopePreviewTime = float(output.envelope_preview_time_s)
    rack_log.holdTopupTorque = float(output.hold_topup_torque)
    rack_log.holdTopupGrowing = bool(output.hold_topup_growing)
    rack_log.saturated = bool(self._check_saturation(output.saturated or self.steer_max - abs(output.torque) < 1e-3, CS,
                                                     steer_limited_by_safety, curvature_limited))
    self.torque.sat_time = self.sat_time
    return output.torque, output.planned_angle_deg, rack_log
