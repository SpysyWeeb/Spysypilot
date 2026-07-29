"""Delay-compensated inverse-EPS computed-torque controller.

The model scalar action is the authoritative steering-position target. The
plan supplies coherent rate/acceleration at that action point. The measured
rack is predicted through the physical actuator delay, then inverse rack
dynamics combine the requested-position load, desired motion, and tracking
error into one torque request.

Two clocks are deliberately independent. ``action_time`` selects the
model-authored scalar target and includes the learned end-to-end lateral lag.
``prediction_delay`` advances the measured rack only until this command can
physically affect it. Treating the learned lag as pure rack transport predicts
the rack response twice and can make feedback cancel valid feedforward.

This is not an arrival-time controller, preview boost, or
tracking-versus-smoothness cost. Far-future curvature cannot pull the wheel
ahead of the scalar action. Position error therefore asks for whatever torque
the physical model requires now, including full normalized authority. The
exact Hyundai 409/4/7 limiter remains the sole command-smoothing authority.
"""

from __future__ import annotations

import math

import numpy as np

from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.blatv2.candidate_common import (
  CandidateWorkspace,
  RACK_ANGLE_QUANTUM_DEG,
  RACK_RATE_QUANTUM_DEG_S,
  measured_rack_friction,
)
from openpilot.selfdrive.controls.lib.blatv2.controller import (
  DECISION_DT,
  CandidateResult,
  CandidateStatus,
  ControllerParams,
  ObserverStatus,
)
from openpilot.selfdrive.controls.lib.blatv2.plant import (
  AlignInputs,
  PlantState,
  PlantTwin,
)
from openpilot.selfdrive.controls.lib.blatv2.reference import MODEL_ACTION_OFFSET


class InverseEpsActionController:
  """Scalar-faithful inverse with event breakaway and slew reachability."""

  def __init__(
    self,
    twin: PlantTwin,
    controller_params: ControllerParams,
    workspace: CandidateWorkspace | None = None,
  ):
    self.twin = twin
    # Retained in the constructor because the shared artifact and seed contract
    # are also consumed by the retired tournament MPC. The active action
    # controller uses only the physical kinetic-friction parameter.
    self.params = controller_params
    self.workspace = CandidateWorkspace() if workspace is None else workspace
    self.result = CandidateResult()
    self.predicted_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.breakaway_direction = 0.0
    self.breakaway_persistence_frames = 0
    self.breakaway_active = False
    self.breakaway_completed = False

  def reset(self) -> None:
    self.result.invalidate()
    self.breakaway_direction = 0.0
    self.breakaway_persistence_frames = 0
    self.breakaway_active = False
    self.breakaway_completed = False

  @staticmethod
  def _interpolate(values: np.ndarray, position: float, count: int) -> float:
    bounded = min(max(float(position), 0.0), float(count - 1))
    lower = int(bounded)
    upper = min(lower + 1, count - 1)
    fraction = bounded - lower
    return float(values[lower]) + fraction * (
      float(values[upper]) - float(values[lower])
    )

  def _update_breakaway(
    self,
    measured_rate_deg_s: float,
    departure_direction: float,
  ) -> float:
    """Return friction for a real stick/slip episode, never plan jitter.

    Route bb showed that treating every sub-quantum rate as static applied
    nearly full breakaway on 99.6% of highway frames. Static compensation is
    now armed only by one model period of persistent, same-direction tracking
    displacement of at least one measured angle count while the rack is inside
    half a rate count. It remains active through the first full rate count so
    the 0.09 -> 0.03 transition is continuous, then latches complete until the
    error resolves or reverses. That episode latch prevents repeated static
    pulses from creating their own stick/slip limit cycle.
    """
    measured_rate = float(measured_rate_deg_s)
    departure = float(departure_direction)
    direction = 0.0 if departure == 0.0 else math.copysign(1.0, departure)
    stationary = abs(measured_rate) <= 0.5 * RACK_RATE_QUANTUM_DEG_S
    displacement_resolved = abs(departure) >= RACK_ANGLE_QUANTUM_DEG
    same_direction = direction != 0.0 and direction == self.breakaway_direction

    if (
      direction == 0.0
      or direction != self.breakaway_direction
      or not displacement_resolved
    ):
      self.breakaway_active = False
      self.breakaway_completed = False
      self.breakaway_persistence_frames = 0
      self.breakaway_direction = direction

    if self.breakaway_active:
      if abs(measured_rate) >= RACK_RATE_QUANTUM_DEG_S:
        self.breakaway_active = False
        self.breakaway_completed = True
        self.breakaway_persistence_frames = 0
    elif (
      not self.breakaway_completed
      and stationary
      and displacement_resolved
      and direction != 0.0
    ):
      if same_direction:
        self.breakaway_persistence_frames += 1
      else:
        self.breakaway_direction = direction
        self.breakaway_persistence_frames = 1
      if self.breakaway_persistence_frames >= int(round(DT_MDL / DT_CTRL)):
        self.breakaway_active = True
    elif not self.breakaway_completed:
      self.breakaway_persistence_frames = 0
      self.breakaway_direction = direction

    if self.breakaway_active:
      return measured_rack_friction(
        measured_rate,
        self.breakaway_direction,
        self.twin.params.t_breakaway,
        self.params.kinetic_friction,
      )
    if abs(measured_rate) > 0.5 * RACK_RATE_QUANTUM_DEG_S:
      return math.copysign(self.params.kinetic_friction, measured_rate)
    return 0.0

  def compute(
    self,
    state: PlantState,
    align_inputs: AlignInputs,
    reference_times: np.ndarray,
    reference_curvatures: np.ndarray,
    reference_count: int,
    horizon_seconds: float,
    prediction_delay: float,
    disturbance_torque: float,
    observer_status: ObserverStatus,
    *,
    workspace_prepared: bool = False,
    action_time: float | None = None,
  ) -> CandidateResult:
    del observer_status  # The action controller has no integral state to freeze.
    result = self.result
    try:
      if not workspace_prepared:
        self.workspace.fill(
          self.twin,
          state,
          align_inputs,
          reference_times,
          reference_curvatures,
          reference_count,
          horizon_seconds,
          disturbance_torque,
          self.params.kinetic_friction,
        )

      action_sample_time = (
        float(prediction_delay) + MODEL_ACTION_OFFSET
        if action_time is None
        else float(action_time)
      )
      if not math.isfinite(action_sample_time) or action_sample_time < 0.0:
        raise ValueError("model action point must be finite and non-negative")
      physical_prediction_delay = float(prediction_delay)
      if (
        not math.isfinite(physical_prediction_delay)
        or physical_prediction_delay < 0.0
      ):
        raise ValueError(
          "physical prediction delay must be finite and non-negative"
        )

      sample_position = action_sample_time / DECISION_DT
      count = self.workspace.decision_count
      desired_angle = self._interpolate(
        self.workspace.desired_angles, sample_position, count,
      )
      desired_rate = self._interpolate(
        self.workspace.desired_rates, sample_position, count,
      )
      desired_acceleration = self._interpolate(
        self.workspace.desired_accelerations, sample_position, count,
      )
      action_speed = self._interpolate(
        self.workspace.reference_speeds, sample_position, count,
      )
      # Compensate only the physical control-to-rack delay. This is state
      # estimation, not path lead: the position target remains the scalar.
      self.twin.predict_held_state_into(
        state,
        physical_prediction_delay,
        align_inputs,
        float(disturbance_torque),
        self.predicted_state,
        DECISION_DT,
      )
      self.predicted_state.v_ego = action_speed

      # Computed-torque pole placement using the plant's physical damping.
      # There is deliberately no response-time constant. Cancelling the rack
      # pole and placing both error poles at b_steer makes authority
      # speed-independent while curvature-to-angle geometry naturally asks for
      # more wheel travel, and therefore more torque, at low speed.
      pole_rate = self.twin.params.b_steer
      angle_error = desired_angle - self.predicted_state.angle_deg
      rate_error = desired_rate - self.predicted_state.rate_deg_s
      required_acceleration = (
        desired_acceleration
        + 2.0 * pole_rate * rate_error
        + pole_rate * pole_rate * angle_error
      )

      # Feed the load required by the model-requested rack position, not the
      # load at the lagging measured position. Using the latter makes the
      # steady-load term preserve the old turn while feedback tries to enter
      # or reverse it; route bc exposed that cancellation directly. The
      # calibrated aligning map is already the inverse static plant, so its
      # desired-state value is the physically required trajectory feedforward.
      aligning = self.twin.aligning_torque_values(
        desired_angle,
        action_speed,
        align_inputs,
      )
      friction = self._update_breakaway(
        state.rate_deg_s,
        angle_error,
      )
      dynamic = (
        required_acceleration
        + self.twin.params.b_steer * self.predicted_state.rate_deg_s
      ) / self.twin.params.k_t
      nominal_dynamic = (
        desired_acceleration
        + self.twin.params.b_steer * desired_rate
      ) / self.twin.params.k_t
      feedforward = (
        aligning
        + float(disturbance_torque)
        + friction
        + nominal_dynamic
      )
      raw_command = (
        aligning
        + float(disturbance_torque)
        + friction
        + dynamic
      )
      if not math.isfinite(raw_command):
        result.invalidate(state.applied_torque, CandidateStatus.NON_CONVERGED)
        result.candidate_count = 1
        result.available_schedule_count = 1
        return result

      local_target = min(max(raw_command, -1.0), 1.0)
      command = self.twin.apply_slew(
        state.applied_torque, local_target,
      )
      result.command_torque = command
      result.raw_command_torque = local_target
      result.feedforward_torque = feedforward
      result.feedback_torque = dynamic - nominal_dynamic
      result.desired_angle_deg = desired_angle
      result.desired_rate_deg_s = desired_rate
      result.desired_acceleration_deg_s2 = desired_acceleration
      result.predicted_angle_deg = self.predicted_state.angle_deg
      result.predicted_rate_deg_s = self.predicted_state.rate_deg_s
      result.required_acceleration_deg_s2 = required_acceleration
      result.action_speed_mps = action_speed
      result.aligning_torque = aligning
      result.friction_torque = friction
      result.dynamic_torque = dynamic
      result.action_time_seconds = action_sample_time
      result.prediction_delay_seconds = physical_prediction_delay
      result.slew_constrained = command != local_target
      result.breakaway_active = self.breakaway_active
      result.breakaway_persistence_frames = (
        self.breakaway_persistence_frames
      )
      # Wire-compatible retired-v207 diagnostics. The corresponding ordinals
      # remain reserved, but the second torque authority is gone.
      result.horizon_assist_active = False
      result.horizon_torque_demand = 0.0
      result.horizon_demand_time_seconds = 0.0
      result.no_lead_limited = False
      result.status = CandidateStatus.OK
      result.candidate_count = 1
      result.available_schedule_count = 1
      result.optimality_residual = 0.0
      return result
    except (ValueError, OverflowError):
      result.invalidate(state.applied_torque, CandidateStatus.INPUT_INVALID)
      return result


# Compatibility name for route-audit tooling that loads the shared numerical
# artifact by its tournament-era symbol. Both names are the same class/bytes.
InverseEpsLQIFallback = InverseEpsActionController
