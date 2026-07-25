#!/usr/bin/env python3
import math
from numbers import Number

from openpilot.cereal import log
from opendbc.car.structs import car
import openpilot.cereal.messaging as messaging
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_CTRL, Priority, Ratekeeper
from openpilot.common.swaglog import cloudlog

from opendbc.car.car_helpers import interfaces
from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle, STEER_ANGLE_SATURATION_THRESHOLD
from openpilot.selfdrive.controls.lib.latcontrol_curvature import LatControlCurvature
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.lateral_reference_planner import ActuatorPreviewConfig, LateralReferencePlanner
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.modeld.modeld import LAT_SMOOTH_SECONDS
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose

State = log.SelfdriveState.OpenpilotState
LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

ACTUATOR_FIELDS = tuple(car.CarControl.Actuators.schema.fields.keys())


def get_steer_limited_by_safety(CP, CC, CO) -> bool:
  if not CC.latActive:
    return False
  if CP.steerControlType == car.CarParams.SteerControlType.angle:
    return abs(CC.actuators.steeringAngleDeg - CO.actuatorsOutput.steeringAngleDeg) > STEER_ANGLE_SATURATION_THRESHOLD
  return abs(CC.actuators.torque - CO.actuatorsOutput.torque) > 1e-2


class Controls:
  def __init__(self) -> None:
    self.params = Params()
    cloudlog.info("controlsd is waiting for CarParams")
    self.CP = messaging.log_from_bytes(self.params.get("CarParams", block=True), car.CarParams)
    cloudlog.info("controlsd got CarParams")

    self.CI = interfaces[self.CP.carFingerprint](self.CP)

    self.sm = messaging.SubMaster(['liveDelay', 'liveParameters', 'liveTorqueParameters', 'modelV2', 'selfdriveState',
                                   'liveCalibration', 'livePose', 'longitudinalPlan', 'lateralManeuverPlan', 'carState', 'carOutput',
                                   'driverMonitoringState', 'onroadEvents', 'driverAssistance'], poll='selfdriveState')
    self.pm = messaging.PubMaster(['carControl', 'controlsState'])

    self.steer_limited_by_safety = False
    self.curvature = 0.0
    self.desired_curvature = 0.0
    self.lateral_reference_planner = LateralReferencePlanner(DT_CTRL)

    self.pose_calibrator = PoseCalibrator()
    self.calibrated_pose: Pose | None = None

    self.LoC = LongControl(self.CP)
    self.VM = VehicleModel(self.CP)
    self.lateral_tuning_type = self.CP.lateralTuning.which()
    self.is_torque_lateral = self.lateral_tuning_type == 'torque'
    self.LaC: LatControl
    if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
      self.LaC = LatControlAngle(self.CP, self.CI, DT_CTRL)
    elif self.CP.steerControlType == car.CarParams.SteerControlType.curvature:
      self.LaC = LatControlCurvature(self.CP, self.CI, DT_CTRL)
    elif self.lateral_tuning_type == 'pid':
      self.LaC = LatControlPID(self.CP, self.CI, DT_CTRL)
    elif self.is_torque_lateral:
      self.LaC = LatControlTorque(self.CP, self.CI, DT_CTRL)

    # BLaT's actuator-aware path timing is currently calibrated from the
    # Hyundai controller limits. Keep every other platform on the legacy path
    # reference until its delivery limits have been measured independently.
    if self.CP.brand == "hyundai" and self.is_torque_lateral:
      hyundai_params = self.CI.CC.params
      self.lateral_reference_planner.configure_actuator(
        ActuatorPreviewConfig(
          max_torque=hyundai_params.STEER_MAX,
          delta_up=hyundai_params.STEER_DELTA_UP,
          delta_down=hyundai_params.STEER_DELTA_DOWN,
          steer_step=hyundai_params.STEER_STEP,
        )
      )

  def update(self):
    self.sm.update(15)
    if self.sm.updated["liveCalibration"]:
      self.pose_calibrator.feed_live_calib(self.sm['liveCalibration'])
    if self.sm.updated["livePose"]:
      device_pose = Pose.from_live_pose(self.sm['livePose'])
      self.calibrated_pose = self.pose_calibrator.build_calibrated_pose(device_pose)

  def state_control(self):
    CS = self.sm['carState']

    # Update VehicleModel
    lp = self.sm['liveParameters']
    x = max(lp.stiffnessFactor, 0.1)
    sr = max(lp.steerRatio, 0.1)
    self.VM.update_params(x, sr)

    steer_angle_without_offset = math.radians(CS.steeringAngleDeg - lp.angleOffsetDeg)
    self.curvature = -self.VM.calc_curvature(steer_angle_without_offset, CS.vEgo, lp.roll)

    # Update Torque Params
    if self.is_torque_lateral:
      torque_params = self.sm['liveTorqueParameters']
      if self.sm.all_checks(['liveTorqueParameters']) and torque_params.useParams:
        self.LaC.update_live_torque_params(
          torque_params.latAccelFactorFiltered, torque_params.latAccelOffsetFiltered, torque_params.frictionCoefficientFiltered
        )

    long_plan = self.sm['longitudinalPlan']
    model_v2 = self.sm['modelV2']

    CC = car.CarControl.new_message()
    CC.enabled = self.sm['selfdriveState'].enabled

    # Check which actuators can be enabled
    standstill = abs(CS.vEgo) <= max(self.CP.minSteerSpeed, 0.3) or CS.standstill
    CC.latActive = self.sm['selfdriveState'].active and not CS.steerFaultTemporary and not CS.steerFaultPermanent and \
                   (not standstill or self.CP.steerAtStandstill)
    CC.longActive = CC.enabled and not any(e.overrideLongitudinal for e in self.sm['onroadEvents']) and self.CP.openpilotLongitudinalControl

    actuators = CC.actuators
    actuators.longControlState = self.LoC.long_control_state

    # Enable blinkers while lane changing
    if model_v2.meta.laneChangeState != LaneChangeState.off:
      CC.leftBlinker = model_v2.meta.laneChangeDirection == LaneChangeDirection.left
      CC.rightBlinker = model_v2.meta.laneChangeDirection == LaneChangeDirection.right

    if not CC.latActive:
      self.LaC.reset()
    if not CC.longActive:
      self.LoC.reset()

    # accel PID loop
    pid_accel_limits = self.CI.get_pid_accel_limits(self.CP, CS.vEgo, CS.vCruise * CV.KPH_TO_MS)
    actuators.accel = float(self.LoC.update(CC.longActive, CS, long_plan.aTarget, long_plan.shouldStop, pid_accel_limits))

    # Steering PID loop and future-aware lateral reference planner
    # Reset desired curvature to current to avoid violating the limits on engage
    lat_delay = self.sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS
    applied_torque = self.sm['carOutput'].actuatorsOutput.torque
    if self.sm.valid['lateralManeuverPlan']:
      new_desired_curvature = self.sm['lateralManeuverPlan'].desiredCurvature if CC.latActive else self.curvature
      self.lateral_reference_planner.reset()
    else:
      raw_desired_curvature = model_v2.action.desiredCurvature if CC.latActive else self.curvature
      if not CC.latActive or not self.sm.valid['modelV2']:
        self.lateral_reference_planner.reset()
      elif self.sm.updated['modelV2']:
        self.lateral_reference_planner.update(model_v2, self.curvature, CS.vEgo)
      if self.is_torque_lateral:
        new_desired_curvature = self.lateral_reference_planner.get_curvature(
          raw_desired_curvature,
          CS.vEgo,
          lat_delay,
          applied_torque,
          self.LaC.torque_params.latAccelFactor,
          self.LaC.torque_params.friction,
          lp.roll,
          self.LaC.torque_params.latAccelOffset,
        )
      else:
        new_desired_curvature = self.lateral_reference_planner.get_curvature(raw_desired_curvature, CS.vEgo, lat_delay)
    self.desired_curvature, curvature_limited = clip_curvature(CS.vEgo, self.desired_curvature, new_desired_curvature, lp.roll)

    actuators.curvature = self.desired_curvature
    if self.is_torque_lateral:
      reference_log = self.lateral_reference_planner.diagnostics
      steer, lateral_output, lac_log = self.LaC.update(
        CC.latActive,
        CS,
        self.VM,
        lp,
        self.steer_limited_by_safety,
        self.desired_curvature,
        curvature_limited,
        lat_delay,
        applied_torque,
        reference_log.unwind_scale,
        reference_log.target_torque,
        reference_log.geometric_target_torque,
        reference_log.episode_target_torque,
        reference_log.output_curvature * CS.vEgo**2,
        reference_log.episode_lateral_accel,
        (reference_log.trajectory_curvature_rate if reference_log.trajectory_rate_valid else None),
        reference_log.scalar_anchor_active,
      )
    else:
      steer, lateral_output, lac_log = self.LaC.update(
        CC.latActive, CS, self.VM, lp, self.steer_limited_by_safety, self.desired_curvature, curvature_limited, lat_delay
      )
    actuators.torqueDampingBlocked = bool(self.is_torque_lateral and lac_log.dampingTurnInBlocked)
    if self.is_torque_lateral:
      lac_log.referenceVersion = reference_log.version
      # np.interp/np.clip may return NumPy scalar types, which pycapnp refuses.
      # Convert every planner diagnostic at the cereal boundary.
      lac_log.referenceBaseCurvature = float(reference_log.base_curvature)
      lac_log.referenceOutputCurvature = float(reference_log.output_curvature)
      lac_log.trajectoryReferenceCurvatureRate = float(reference_log.trajectory_curvature_rate)
      lac_log.trajectoryReferenceRateValid = bool(reference_log.trajectory_rate_valid)
      lac_log.referencePreviewTime = float(reference_log.sample_time)
      lac_log.referencePreviewExtraTime = float(reference_log.extra_time)
      lac_log.referenceTargetTorque = float(reference_log.target_torque)
      lac_log.referenceAppliedTorque = float(reference_log.applied_torque)
      lac_log.referenceUnwindScale = float(reference_log.unwind_scale)
      lac_log.referenceAuthorityRestored = float(reference_log.authority_restored)
      lac_log.referencePreviewCorrection = float(reference_log.preview_correction)
      lac_log.referenceGeometricTargetTorque = float(reference_log.geometric_target_torque)
      lac_log.referenceNeutralTorque = float(reference_log.neutral_torque)
      lac_log.referenceReachableTargetTorque = float(reference_log.reachable_target_torque)
      lac_log.referenceSustainedUnwindScale = float(reference_log.sustained_unwind_scale)
      lac_log.referenceEpisodeTargetTorque = float(reference_log.episode_target_torque)
      lac_log.referenceEpisodeLateralAccel = float(reference_log.episode_lateral_accel)
      lac_log.scalarAnchorDeviation = float(reference_log.scalar_anchor_deviation)
      lac_log.referencePersistenceGateHold = bool(reference_log.persistence_gate_hold)
    actuators.torque = float(steer)
    if self.CP.steerControlType == car.CarParams.SteerControlType.curvature:
      actuators.curvature = float(lateral_output)
    else:
      actuators.steeringAngleDeg = float(lateral_output)
    # Ensure no NaNs/Infs
    for p in ACTUATOR_FIELDS:
      attr = getattr(actuators, p)
      if not isinstance(attr, Number):
        continue

      if not math.isfinite(attr):
        cloudlog.error(f"actuators.{p} not finite {actuators.to_dict()}")
        setattr(actuators, p, 0.0)

    return CC, lac_log

  def publish(self, CC, lac_log):
    CS = self.sm['carState']

    # Orientation and angle rates can be useful for carcontroller
    # Only calibrated (car) frame is relevant for the carcontroller
    CC.currentCurvature = self.curvature
    if self.calibrated_pose is not None:
      CC.orientationNED = self.calibrated_pose.orientation.xyz.tolist()
      CC.angularVelocity = self.calibrated_pose.angular_velocity.xyz.tolist()

    CC.cruiseControl.override = CC.enabled and not CC.longActive and self.CP.openpilotLongitudinalControl
    CC.cruiseControl.cancel = CS.cruiseState.enabled and (not CC.enabled or not self.CP.pcmCruise)
    CC.cruiseControl.resume = CC.enabled and CS.cruiseState.standstill and not self.sm['longitudinalPlan'].shouldStop

    hudControl = CC.hudControl
    hudControl.setSpeed = float(CS.vCruiseCluster * CV.KPH_TO_MS)
    hudControl.speedVisible = CC.enabled
    hudControl.lanesVisible = CC.enabled
    hudControl.leadVisible = self.sm['longitudinalPlan'].hasLead
    hudControl.leadDistanceBars = self.sm['selfdriveState'].personality.raw + 1
    hudControl.visualAlert = self.sm['selfdriveState'].alertHudVisual

    hudControl.rightLaneVisible = True
    hudControl.leftLaneVisible = True
    if self.sm.valid['driverAssistance']:
      hudControl.leftLaneDepart = self.sm['driverAssistance'].leftLaneDeparture
      hudControl.rightLaneDepart = self.sm['driverAssistance'].rightLaneDeparture

    # AOL can steer while selfdriveState is inactive. Keep limiter feedback live whenever lateral control is active.
    self.steer_limited_by_safety = get_steer_limited_by_safety(self.CP, CC, self.sm['carOutput'])

    # TODO: both controlsState and carControl valids should be set by
    #       sm.all_checks(), but this creates a circular dependency

    # controlsState
    dat = messaging.new_message('controlsState')
    dat.valid = CS.canValid
    cs = dat.controlsState

    cs.curvature = self.curvature
    cs.longitudinalPlanMonoTime = self.sm.logMonoTime['longitudinalPlan']
    cs.lateralPlanMonoTime = self.sm.logMonoTime['modelV2']
    cs.desiredCurvature = self.desired_curvature
    cs.longControlState = self.LoC.long_control_state
    cs.upAccelCmd = float(self.LoC.pid.p)
    cs.uiAccelCmd = float(self.LoC.pid.i)
    cs.ufAccelCmd = float(self.LoC.pid.f)
    cs.forceDecel = bool(self.sm['driverMonitoringState'].noResponseForceDecel or (self.sm['selfdriveState'].state == State.softDisabling))

    # trigger the car's stock driver monitoring escalation
    CC.driverMonitoringEscalation = cs.forceDecel

    lat_tuning = self.lateral_tuning_type
    if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
      cs.lateralControlState.angleState = lac_log
    elif self.CP.steerControlType == car.CarParams.SteerControlType.curvature:
      cs.lateralControlState.curvatureState = lac_log
    elif lat_tuning == 'pid':
      cs.lateralControlState.pidState = lac_log
    elif lat_tuning == 'torque':
      cs.lateralControlState.torqueState = lac_log

    self.pm.send('controlsState', dat)

    # carControl
    cc_send = messaging.new_message('carControl')
    cc_send.valid = CS.canValid
    cc_send.carControl = CC
    self.pm.send('carControl', cc_send)

  def run(self):
    rk = Ratekeeper(100, print_delay_threshold=None)
    while True:
      self.update()
      CC, lac_log = self.state_control()
      self.publish(CC, lac_log)
      rk.monitor_time()


def main():
  config_realtime_process(4, Priority.CTRL_HIGH)
  controls = Controls()
  controls.run()


if __name__ == "__main__":
  main()
