from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.rack_trajectory import RackTrajectoryController

# Executes the model path as a planned rack motion (see rack_trajectory.py) and tracks it with
# torque. A stock torque controller is stepped alongside every frame, so any frame the rack
# controller cannot produce a request for (no or stale model, invalid path, infeasible plan)
# is steered by stock instead of dropping torque. Its request buffer and jerk filter follow the
# live history; its integrator starts clean when it takes over, and the two controllers share
# one steering saturation timer.

VERSION = 1


class LatControlRack(LatControl):
  def __init__(self, CP, CI, dt):
    super().__init__(CP, CI, dt)
    self.torque = LatControlTorque(CP, CI, dt)
    self.rack = RackTrajectoryController(dt)
    self.output = None

  def update_torque_parameters(self, latAccelFactor, latAccelOffset, friction):
    self.torque.update_torque_parameters(latAccelFactor, latAccelOffset, friction)

  def reset(self):
    super().reset()
    self.torque.reset()
    self.rack.reset()
    self.output = None

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay,
             model=None, mono_time_ns=0):
    stock_torque, _, stock_log = self.torque.update(active, CS, VM, params, steer_limited_by_safety, desired_curvature,
                                                    curvature_limited, lat_delay)
    self.rack.set_model(model, mono_time_ns)
    self.output = self.rack.update(active, CS, VM, params, self.torque.torque_params, self.torque.torque_from_lateral_accel,
                                   lat_delay, desired_curvature)

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
    rack_log.saturated = bool(self._check_saturation(output.saturated or self.steer_max - abs(output.torque) < 1e-3, CS,
                                                     steer_limited_by_safety, curvature_limited))
    self.torque.sat_time = self.sat_time
    return output.torque, output.planned_angle_deg, rack_log
