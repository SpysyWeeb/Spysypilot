import numpy as np
from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.smooth_release import SmoothRelease
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.smooth_stops import SmoothStopController

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

LongCtrlState = car.CarControl.Actuators.LongControlState


def long_control_state_trans(CP, active, long_control_state, v_ego,
                             should_stop, brake_pressed, cruise_standstill):
  stopping_condition = should_stop
  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)
  started_condition = v_ego > CP.vEgoStarting

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        if starting_condition and CP.startingState:
          long_control_state = LongCtrlState.starting
        else:
          long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition and CP.startingState:
        long_control_state = LongCtrlState.starting
      elif starting_condition:
        long_control_state = LongCtrlState.pid

    elif long_control_state in [LongCtrlState.starting, LongCtrlState.pid]:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping
      elif started_condition:
        long_control_state = LongCtrlState.pid
  return long_control_state

class LongControl:
  def __init__(self, CP):
    self.CP = CP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV),
                             (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             rate=1 / DT_CTRL)
    self.last_output_accel = 0.0
    self.smooth = SmoothStopController()
    self.smooth_release = SmoothRelease()

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, a_target, should_stop, accel_limits, lead_distance=0.0, has_lead=False, lead_speed=0.0):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    # Smooth Stops owns the final approach: while the plan wants to stop but the car is
    # still rolling, defer the hold clamp and settle in the pid branch below. The clamp
    # (stopping state) only arms once we're actually stopped, so it never headbangs.
    # starting is exempt: if the lead re-stops mid-launch, take the stock path (immediate
    # stopping) rather than keep commanding startAccel toward a stopped lead.
    if active and self.long_control_state not in (LongCtrlState.stopping, LongCtrlState.starting):
      stop_now = self.smooth.want_hold(should_stop, CS.vEgo, CS.standstill)
    elif active and self.long_control_state == LongCtrlState.stopping:
      # debounced hold exit: a one-frame should_stop flicker must not blip the brake at standstill
      stop_now = not self.smooth.hold_release(should_stop)
    else:
      stop_now = should_stop

    self.long_control_state = long_control_state_trans(self.CP, active, self.long_control_state, CS.vEgo,
                                                       stop_now, CS.brakePressed,
                                                       CS.cruiseState.standstill)
    if self.long_control_state == LongCtrlState.off:
      self.reset()
      self.smooth.reset()
      self.smooth_release.reset()
      output_accel = 0.

    elif self.long_control_state == LongCtrlState.stopping:
      output_accel = self.last_output_accel
      if output_accel > self.CP.stopAccel:
        output_accel = min(output_accel, 0.0)
        output_accel -= self.CP.stoppingDecelRate * DT_CTRL
      self.reset()
      self.smooth.reset()
      self.smooth_release.reset()

    elif self.long_control_state == LongCtrlState.starting:
      output_accel = self.CP.startAccel
      self.reset()
      self.smooth.reset()
      self.smooth_release.reset()

    else:  # LongCtrlState.pid
      if active and should_stop:
        # SETTLE: feather to a true standstill instead of clamping while still rolling.
        # Open-loop accel command (like stopping/starting), so keep the PID reset.
        # Smooth Release is deliberately NOT applied here: settle's entry-anchored taper
        # must be free to rise toward the kiss faster than the release governor allows.
        output_accel = self.smooth.settle(a_target, CS.vEgo, lead_distance, has_lead, self.last_output_accel, lead_speed)
        self.reset()
      else:
        error = a_target - CS.aEgo
        # freeze the integrator while Smooth Release is clamping, else it winds up against the hold
        output_accel = self.pid.update(error, speed=CS.vEgo,
                                       feedforward=a_target,
                                       freeze_integrator=self.smooth_release.engaged)
        # Smooth Release: brake releases are bled off as one human-like taper, never a pump
        output_accel = self.smooth_release.govern(output_accel, a_target, self.last_output_accel, CS.vEgo, lead_speed, has_lead)
        self.smooth.reset()

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel
