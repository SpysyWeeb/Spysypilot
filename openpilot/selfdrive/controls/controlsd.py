#!/usr/bin/env python3
import math
from numbers import Number
import os

from openpilot.cereal import log
from opendbc.car.structs import car
import openpilot.cereal.messaging as messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.constants import CV
from openpilot.common.git import get_commit
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_CTRL, Priority, Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.common.version import get_build_metadata

from opendbc.car.car_helpers import interfaces
from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.controls.lib.blatv2.bootstrap import ControllerSelection
from openpilot.selfdrive.controls.lib.blatv2.live_controller import (
  construct_modular_live_controller,
)
from openpilot.selfdrive.controls.lib.blatv2.live_telemetry import (
  build_modular_lateral_state,
)
from openpilot.selfdrive.controls.lib.blatv2.stock_bootstrap import (
  fresh_stock_torque_controller,
)
from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle, STEER_ANGLE_SATURATION_THRESHOLD
from openpilot.selfdrive.controls.lib.latcontrol_curvature import LatControlCurvature
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.modeld.modeld import LAT_SMOOTH_SECONDS
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose

State = log.SelfdriveState.OpenpilotState
LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

ACTUATOR_FIELDS = tuple(car.CarControl.Actuators.schema.fields.keys())


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

    self.pose_calibrator = PoseCalibrator()
    self.calibrated_pose: Pose | None = None

    self.LoC = LongControl(self.CP)
    self.VM = VehicleModel(self.CP)
    self.LaC: LatControl
    if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
      self.LaC = LatControlAngle(self.CP, self.CI, DT_CTRL)
    elif self.CP.steerControlType == car.CarParams.SteerControlType.curvature:
      self.LaC = LatControlCurvature(self.CP, self.CI, DT_CTRL)
    elif self.CP.lateralTuning.which() == 'pid':
      self.LaC = LatControlPID(self.CP, self.CI, DT_CTRL)
    elif self.CP.lateralTuning.which() == 'torque':
      self.LaC = fresh_stock_torque_controller(self.CP, self.CI)

    # Optional modular construction is process-start-only. No approved
    # artifact exists on a fresh install, so the exact stock controller above
    # remains the sole actuator unless profiled has prepared an exact active
    # artifact or the explicit development-trial parameter is enabled.
    try:
      source_openpilot_commit = (
        get_build_metadata().openpilot.git_commit
      )
    except Exception:
      source_openpilot_commit = ""
    try:
      opendbc_commit = get_commit(os.path.join(BASEDIR, "opendbc_repo"))
    except Exception:
      opendbc_commit = ""
    try:
      panda_commit = get_commit(os.path.join(BASEDIR, "panda"))
    except Exception:
      panda_commit = ""
    self.blatv2_live = construct_modular_live_controller(
      car_params=self.CP,
      car_interface=self.CI,
      params=self.params,
      source_openpilot_commit=source_openpilot_commit,
      opendbc_commit=opendbc_commit,
      panda_commit=panda_commit,
    )
    self.lateral_maneuver_mode = self.params.get_bool(
      "LateralManeuverMode",
    )
    self.blatv2_messages_valid = True
    self.blatv2_modular_session = False
    cloudlog.info(
      f"BLaTv2 modular bootstrap: eligibility={int(self.blatv2_live.eligibility)} " +
      f"experimental={self.blatv2_live.experimental_active} " +
      f"runtime_identity={self.blatv2_live.runtime_identity_sha256}"
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
    if self.CP.lateralTuning.which() == 'torque':
      torque_params = self.sm['liveTorqueParameters']
      if self.sm.all_checks(['liveTorqueParameters']) and torque_params.useParams:
        self.LaC.update_live_torque_params(torque_params.latAccelFactorFiltered, torque_params.latAccelOffsetFiltered,
                                           torque_params.frictionCoefficientFiltered)

    long_plan = self.sm['longitudinalPlan']
    model_v2 = self.sm['modelV2']

    CC = car.CarControl.new_message()
    CC.enabled = self.sm['selfdriveState'].enabled

    # Check which actuators can be enabled
    standstill = abs(CS.vEgo) <= max(self.CP.minSteerSpeed, 0.3) or CS.standstill
    CC.latActive = self.sm['selfdriveState'].active and not CS.steerFaultTemporary and not CS.steerFaultPermanent and \
                   (not standstill or self.CP.steerAtStandstill)
    CC.longActive = CC.enabled and not any(e.overrideLongitudinal for e in self.sm['onroadEvents']) and self.CP.openpilotLongitudinalControl

    # Bind one controller architecture at the AOL/MADS lateral-session
    # boundary (enabled OR latActive). An exact
    # torqueOutputCan count plus the emitted steering-request witness are the
    # actuator Markov state accepted by the modular path; failure leaves a new
    # engagement on exact stock.
    self.blatv2_live.observe_previous_applied(
      self.sm['carOutput'],
      car_output_mono_ns=int(self.sm.logMonoTime['carOutput']),
    )
    lateral_maneuver_active = bool(
      self.lateral_maneuver_mode
      or self.sm.valid['lateralManeuverPlan']
    )
    controller_selection = self.blatv2_live.update_engagement(
      enabled=CC.enabled,
      lateral_active=CC.latActive,
      lateral_maneuver_active=lateral_maneuver_active,
    )
    if not CC.enabled and not CC.latActive:
      self.blatv2_live.observe_inactive_state(
        state_sample_mono_ns=int(self.sm.logMonoTime['carState']),
        car_state=CS,
        inputs_valid=bool(
          self.sm.seen['carState']
          and self.sm.valid['carState']
          and CS.canValid
        ),
      )
    if (
      controller_selection == ControllerSelection.MODULAR
      and not self.blatv2_modular_session
    ):
      # A stock controller that is merely left idle retains its PID integral,
      # jerk filter, and request deque. Reconstruct it at the modular boundary
      # so a later rollback starts from a genuinely fresh stock state rather
      # than hours-old hidden state. This object never actuates this session.
      torque_params = self.sm['liveTorqueParameters']
      self.LaC = fresh_stock_torque_controller(
        self.CP,
        self.CI,
        (
          torque_params
          if self.sm.all_checks(['liveTorqueParameters'])
          else None
        ),
      )
      self.desired_curvature = self.curvature
      self.blatv2_modular_session = True
    elif not self.blatv2_live.enabled_bound:
      self.blatv2_modular_session = False

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

    # Steering controller. The STOCK arm below is the existing stock block:
    # modular state, exceptions, and telemetry never participate in it.
    if controller_selection == ControllerSelection.STOCK:
      # Reset desired curvature to current to avoid violating the limits on engage
      if self.sm.valid['lateralManeuverPlan']:
        new_desired_curvature = self.sm['lateralManeuverPlan'].desiredCurvature if CC.latActive else self.curvature
      else:
        new_desired_curvature = model_v2.action.desiredCurvature if CC.latActive else self.curvature
      self.desired_curvature, curvature_limited = clip_curvature(CS.vEgo, self.desired_curvature, new_desired_curvature, lp.roll)
      lat_delay = self.sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS

      actuators.curvature = self.desired_curvature
      steer, lateral_output, lac_log = self.LaC.update(CC.latActive, CS, self.VM, lp,
                                                       self.steer_limited_by_safety, self.desired_curvature,
                                                       curvature_limited, lat_delay)
      actuators.torque = float(steer)
      if self.CP.steerControlType == car.CarParams.SteerControlType.curvature:
        actuators.curvature = float(lateral_output)
      else:
        actuators.steeringAngleDeg = float(lateral_output)
      self.blatv2_messages_valid = True
    else:
      result = self.blatv2_live.update_modular(
        state_sample_mono_ns=int(self.sm.logMonoTime['carState']),
        model_publication_mono_ns=int(self.sm.logMonoTime['modelV2']),
        model_message=model_v2,
        car_state=CS,
        live_parameters=lp,
        model_message_valid=bool(self.sm.valid['modelV2']),
        model_message_alive=bool(self.sm.alive['modelV2']),
        vehicle_inputs_valid=bool(
          self.sm.all_checks(['carState', 'carOutput'])
          and CS.canValid
        ),
        live_parameters_inputs_valid=bool(
          self.sm.all_checks(['liveParameters'])
        ),
        lateral_active=CC.latActive,
        actuator_constrained_previous=(
          not self.blatv2_live.final_count_match_valid
          or self.blatv2_live.final_limiter_altered
        ),
        lateral_maneuver_active=lateral_maneuver_active,
      )
      core_result = result.core_result
      reference_curvature = (
        float(model_v2.action.desiredCurvature)
        if core_result is None
        else float(core_result.desired_curvature)
      )
      self.desired_curvature = (
        reference_curvature
        if math.isfinite(reference_curvature)
        else self.curvature
      )
      actuators.curvature = self.desired_curvature
      actuators.torque = float(
        result.command_torque if result.command_available else 0.0
      )
      actuators.steeringAngleDeg = float(
        0.0 if core_result is None else core_result.desired_angle_deg
      )
      lac_log = build_modular_lateral_state(
        self.blatv2_live,
        lateral_active=CC.latActive,
        measured_curvature=self.curvature,
        v_ego_m_s=CS.vEgo,
      )
      self.blatv2_messages_valid = self.blatv2_live.messages_valid
    # Ensure no NaNs/Infs
    for p in ACTUATOR_FIELDS:
      attr = getattr(actuators, p)
      if not isinstance(attr, Number):
        continue

      if not math.isfinite(attr):
        cloudlog.error(f"actuators.{p} not finite {actuators.to_dict()}")
        setattr(actuators, p, 0.0)

    if controller_selection == ControllerSelection.MODULAR:
      command_recorded = self.blatv2_live.record_requested_command(
        actuators.torque,
      )
      self.blatv2_messages_valid = (
        self.blatv2_messages_valid and command_recorded
      )

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

    if self.sm['selfdriveState'].active:
      CO = self.sm['carOutput']
      if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
        self.steer_limited_by_safety = abs(CC.actuators.steeringAngleDeg - CO.actuatorsOutput.steeringAngleDeg) > \
                                              STEER_ANGLE_SATURATION_THRESHOLD
      elif self.blatv2_live.selection == ControllerSelection.MODULAR:
        self.steer_limited_by_safety = (
          not self.blatv2_live.final_count_match_valid
          or self.blatv2_live.final_limiter_altered
        )
      else:
        self.steer_limited_by_safety = abs(CC.actuators.torque - CO.actuatorsOutput.torque) > 1e-2

    # TODO: both controlsState and carControl valids should be set by
    #       sm.all_checks(), but this creates a circular dependency

    # controlsState
    dat = messaging.new_message('controlsState')
    dat.valid = CS.canValid and self.blatv2_messages_valid
    cs = dat.controlsState

    cs.curvature = self.curvature
    cs.longitudinalPlanMonoTime = self.sm.logMonoTime['longitudinalPlan']
    cs.lateralPlanMonoTime = self.sm.logMonoTime['modelV2']
    cs.desiredCurvature = self.desired_curvature
    cs.longControlState = self.LoC.long_control_state
    cs.upAccelCmd = float(self.LoC.pid.p)
    cs.uiAccelCmd = float(self.LoC.pid.i)
    cs.ufAccelCmd = float(self.LoC.pid.f)
    cs.forceDecel = bool(self.sm['driverMonitoringState'].noResponseForceDecel or
                         (self.sm['selfdriveState'].state == State.softDisabling))

    # trigger the car's stock driver monitoring escalation
    CC.driverMonitoringEscalation = cs.forceDecel

    lat_tuning = self.CP.lateralTuning.which()
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
    cc_send.valid = CS.canValid and self.blatv2_messages_valid
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
