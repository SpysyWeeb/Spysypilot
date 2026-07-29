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

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.blatv2.candidate_common import (
  CandidateWorkspace,
  RACK_RATE_QUANTUM_DEG_S,
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
  """One scalar-faithful inverse-EPS command path.

  The linear position term keeps small corrections calm. A stateless,
  continuously differentiable authority map supplies the rack's identified
  static-load envelope as model-angle error becomes physically meaningful.
  There is no persistence gate, boost state, integral, or preview scheduler:
  the same error-to-torque law handles center corrections, sharp turns, and
  unwinds.
  """

  def __init__(
    self,
    twin: PlantTwin,
    controller_params: ControllerParams,
    workspace: CandidateWorkspace | None = None,
  ):
    self.twin = twin
    self.params = controller_params
    self.workspace = CandidateWorkspace() if workspace is None else workspace
    self.result = CandidateResult()
    self.predicted_state = PlantState(0.0, 0.0, 0.0, 0.0)

  def reset(self) -> None:
    self.result.invalidate()

  @staticmethod
  def _interpolate(values: np.ndarray, position: float, count: int) -> float:
    bounded = min(max(float(position), 0.0), float(count - 1))
    lower = int(bounded)
    upper = min(lower + 1, count - 1)
    fraction = bounded - lower
    return float(values[lower]) + fraction * (
      float(values[upper]) - float(values[lower])
    )

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
    del observer_status  # This controller has no observer-writable state.
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

      # Position stiffness and rate damping have different physical jobs and
      # different measurement quality. Versions through v214 tied both to one
      # pole-rate dial; raising useful angle authority also amplified the
      # 4-deg/s-quantized rack-rate signal into highway chatter. Position
      # stiffness is explicit. Rate damping is bounded so one sensor quantum
      # can change feedback by at most one Hyundai torque build step.
      angle_error = desired_angle - self.predicted_state.angle_deg
      rate_error = desired_rate - self.predicted_state.rate_deg_s
      rate_damping = (
        self.twin.params.delta_up
        / self.twin.params.steer_max
        / RACK_RATE_QUANTUM_DEG_S
      )
      transition_error_deg = self.params.authority_transition_error_deg
      normalized_error = min(
        abs(angle_error) / transition_error_deg, 1.0,
      )
      smoothstep = normalized_error * normalized_error * (
        3.0 - 2.0 * normalized_error
      )
      static_authority = max(
        self.twin.params.t_breakaway
        - self.params.tracking_stiffness * transition_error_deg,
        0.0,
      )
      position_feedback = (
        math.copysign(
          self.params.tracking_stiffness * abs(angle_error)
          + static_authority * smoothstep,
          angle_error,
        )
        if angle_error != 0.0
        else 0.0
      )
      feedback = position_feedback + rate_damping * rate_error
      required_acceleration = (
        desired_acceleration
        + self.twin.params.b_steer * rate_error
        + self.twin.params.k_t * feedback
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
      friction = (
        math.copysign(self.params.kinetic_friction, state.rate_deg_s)
        if abs(state.rate_deg_s) > 0.5 * RACK_RATE_QUANTUM_DEG_S
        else 0.0
      )
      nominal_dynamic = (
        desired_acceleration
        + self.twin.params.b_steer * desired_rate
      ) / self.twin.params.k_t
      dynamic = nominal_dynamic + feedback
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
      result.feedback_torque = feedback
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
      result.breakaway_active = False
      result.breakaway_persistence_frames = 0
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
