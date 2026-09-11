from opendbc.car.hyundai.values import CarControllerParams
from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.rack_trajectory import (
  DRIVER_ASSIST_CEILING, DriverAssistLimits, INACTIVE_HOLD_FRAMES, R7_MAX_TORQUE_STEP, STATUS_STALE_MODEL,
  RackTrajectoryController,
)

# Executes the model path as a planned rack motion (see rack_trajectory.py) and tracks it with
# torque. A stock torque controller is stepped alongside every frame, so any frame the rack
# controller loses its plan on (no or stale model, an infeasible plan, a non-finite request) is
# steered by stock instead of dropping torque. Its request buffer and jerk filter follow the
# live history; its integrator starts clean when it takes over, and the two controllers share
# one steering saturation timer. Once stock has taken over because the model went stale it keeps
# steering for a hold time, so the two controllers cannot trade places every frame around the
# staleness threshold.
#
# A content fault in one frame's inputs is not one of those frames: the controller holds its plan
# (R6) and the output is state too, so this class keeps steering with the torque it last committed
# for the few held frames (rack.holding, INACTIVE_HOLD_FRAMES at most) instead of changing hands
# twice for one bad model frame. Past that budget the controller has reset and the hand-over below
# applies as usual.
#
# This class is also the only place that sees both controllers, so it owns R7 across the hand-over
# between them (_commit below).

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
    self.committed_torque = None  # the torque last committed to the car while active, whichever controller made it
    self.rack_steering = False    # which one that was, so a change of source is visible
    self.handover_reconcile = False
    self.output_altered = False   # this frame's committed torque is not the source's own request

  def update_torque_parameters(self, latAccelFactor, latAccelOffset, friction):
    self.torque.update_torque_parameters(latAccelFactor, latAccelOffset, friction)

  def reset(self):
    super().reset()
    self.torque.reset()
    self.rack.hold()
    if self.rack.inactive_frames > INACTIVE_HOLD_FRAMES:
      self.fallback_frames = 0
    self.output = None
    self.committed_torque = None
    self.handover_reconcile = False
    self.output_altered = False

  def _commit(self, active, torque, rack_steering):
    """R7 across the rack/stock hand-over (FM3.5, R7). Each controller bounds the steps of its own
    rules, but neither can see the other's: the frame stock takes over, and the frame the rack takes it
    back, the output jumped by the whole difference between two independently composed requests --
    0.20-0.23 measured on a one-frame content fault, 4x the R7 step (audit F12). Keep the torque
    actually committed to the car and, once the source changes, slew the new source's request toward
    it by at most R7_MAX_TORQUE_STEP a frame until its own request is within one step, then hand over
    cleanly. It costs up to four frames of lag on a hand-over and nothing at all when there is none --
    outside a hand-over the committed torque is the source's own request, bit for bit. A fresh engage
    is not a rule boundary (the same reading as the controller's own R7 baseline), so an inactive
    frame drops the committed value instead of slewing from it.

    Since a content fault holds the wheel here rather than handing it over, the source can only change
    when the controller has actually lost its plan: a stale or missing model, an infeasible plan, a
    non-finite request, or a content fault past the hold budget. Every one of those is followed by a
    stock hold (FALLBACK_HOLD_S) or by the controller re-seeding its plan from the wheel, so the source
    cannot flap frame to frame and the slew always has a settled value to start from.

    The rack controller's own R7 baseline is deliberately left alone. Every hand-over is preceded by a
    reset that clears it, so it can never be stale from before a stock interval, and on the reconcile
    frames the clamp here is the binding one: a probe that presses the driver's hand through a
    stale-model hand-over delivers bit-identical torque whether or not the controller is told what was
    committed (route-audit phase3/hygiene_batch_2026-09-11/probe_seed_effect.py).

    `output_altered` records whether this frame's committed torque is the source's own request or
    something this class changed, for the log's torqueLimited flag."""
    request = torque
    if not active:
      self.committed_torque = None
      self.handover_reconcile = False
      self.rack_steering = rack_steering
      self.output_altered = False
      return torque
    if self.committed_torque is not None:
      if rack_steering != self.rack_steering:
        self.handover_reconcile = True
      if self.handover_reconcile:
        slewed = min(max(torque, self.committed_torque - R7_MAX_TORQUE_STEP),
                     self.committed_torque + R7_MAX_TORQUE_STEP)
        self.handover_reconcile = slewed != torque
        torque = slewed
    self.committed_torque = torque
    self.rack_steering = rack_steering
    self.output_altered = torque != request
    return torque

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
    # a content fault: the plan is held, so the wheel is too (R6). Only while engaged, and never on the
    # frame a fault lands before the first plan, where there is no committed torque yet -- stock takes
    # that one.
    held = active and self.output is None and self.rack.holding and self.committed_torque is not None

    rack_log = log.ControlsState.LateralRackState.new_message()
    rack_log.version = VERSION
    rack_log.status = self.rack.status
    if held:
      # keep steering with what was already committed; stock stays in shadow with a clean integrator,
      # exactly as while the rack steers. p/d/f and the top-up are a frame's composition and this frame
      # has none, so they stay at 0.0 rather than repeat the last frame's; torqueLimited says the
      # committed torque is not this frame's composition, which is what it already means for the
      # platform clip, the guard and driver assist.
      torque = self._commit(active, self.committed_torque, True)
      self.torque.pid.reset()
      rack_log.active = True
      rack_log.fallback = False
      rack_log.output = torque
      rack_log.torqueLimited = True
      rack_log.driverAssistCap = DRIVER_ASSIST_CEILING  # not "capped to zero"; see the stock branch below
      rack_log.saturated = bool(self._check_saturation(self.steer_max - abs(torque) < 1e-3, CS,
                                                       steer_limited_by_safety, curvature_limited))
      self.torque.sat_time = self.sat_time
      return torque, 0.0, rack_log
    if self.output is None:
      # steered by the stock controller this frame
      stock_torque = self._commit(active, float(stock_torque), False)
      self.sat_time = self.torque.sat_time
      rack_log.active = stock_log.active
      rack_log.fallback = bool(active)
      rack_log.error = stock_log.error
      rack_log.p = stock_log.p
      rack_log.d = stock_log.d
      rack_log.f = stock_log.f
      rack_log.output = stock_torque  # what was committed, not stock's own request: they differ across a hand-over
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
    torque = self._commit(active, float(output.torque), True)
    rack_log.active = True
    rack_log.error = float(output.lateral_accel_error)
    rack_log.errorRate = float(output.rate_error_deg_s)
    rack_log.p = float(output.position_feedback_torque)
    rack_log.d = float(output.rate_feedback_torque)
    rack_log.f = float(output.feedforward_torque)
    rack_log.output = torque
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
    # the wrapper's own slew is one more alteration downstream of the composition, like the clip,
    # the guard and driver assist: the same flag covers it
    rack_log.torqueLimited = bool(output.torque_limited) or self.output_altered
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
    rack_log.saturated = bool(self._check_saturation(output.saturated or self.steer_max - abs(torque) < 1e-3, CS,
                                                     steer_limited_by_safety, curvature_limited))
    self.torque.sat_time = self.sat_time
    return torque, output.planned_angle_deg, rack_log
