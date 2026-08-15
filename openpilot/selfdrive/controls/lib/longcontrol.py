import numpy as np
from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.blotv2 import BLOTV2_ACCEL_MAX
from openpilot.selfdrive.controls.lib.longitudinal_lead import LeadObservation
from openpilot.selfdrive.controls.lib.smooth_stops import SmoothStopController
from openpilot.selfdrive.modeld.constants import ModelConstants

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

LongCtrlState = car.CarControl.Actuators.LongControlState


def long_control_state_trans(active, long_control_state, should_stop, brake_pressed, cruise_standstill):
  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.pid:
      if should_stop:
        long_control_state = LongCtrlState.stopping

  return long_control_state

class LongControl:
  def __init__(self, CP):
    self.CP = CP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController(0.0, (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             rate=1 / DT_CTRL)
    self.last_output_accel = 0.0
    self.smooth_stop = SmoothStopController()

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, a_target, should_stop, accel_limits, lead: LeadObservation | None = None):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    lead = lead if lead is not None else LeadObservation()
    pos_limit = min(accel_limits[1], BLOTV2_ACCEL_MAX)
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = pos_limit

    # Smooth Stops owns only a rolling final approach. This is safe from both
    # pid and off: current openpilot has no separate starting command that can
    # blip on the off -> pid edge. Once truly stopped, the stock stopping clamp
    # takes over. Hold release remains debounced against one-frame plan flicker.
    if active and should_stop and self.long_control_state in (LongCtrlState.off, LongCtrlState.pid):
      stop_now = self.smooth_stop.want_hold(should_stop, CS.vEgo, CS.standstill)
    elif active and self.long_control_state == LongCtrlState.stopping:
      stop_now = not self.smooth_stop.hold_release(should_stop, lead)
    else:
      stop_now = should_stop

    previous_state = self.long_control_state
    self.long_control_state = long_control_state_trans(active, self.long_control_state, stop_now,
                                                       CS.brakePressed, CS.cruiseState.standstill)
    if self.long_control_state == LongCtrlState.stopping and previous_state != LongCtrlState.stopping:
      self.smooth_stop.arm_hold()

    if self.long_control_state == LongCtrlState.off:
      self.reset()
      self.smooth_stop.reset()
      output_accel = 0.

    elif self.long_control_state == LongCtrlState.stopping:
      output_accel = self.last_output_accel
      if output_accel > self.CP.stopAccel:
        output_accel = min(output_accel, 0.0)
        # TODO: can we just go straight to stopAccel?
        output_accel -= 1.0 * DT_CTRL  # m/s^2/s while trying to stop
      self.reset()
      self.smooth_stop.reset()

    else:  # LongCtrlState.pid
      if active and should_stop:
        output_accel = self.smooth_stop.settle(a_target, CS.vEgo, self.last_output_accel, lead)
        self.reset()
      else:
        error = a_target - CS.aEgo
        output_accel = self.pid.update(error, speed=CS.vEgo,
                                       feedforward=a_target)
        self.smooth_stop.reset()

    self.last_output_accel = np.clip(output_accel, accel_limits[0], pos_limit)
    return self.last_output_accel
