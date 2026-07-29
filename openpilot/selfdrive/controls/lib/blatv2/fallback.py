"""Delay-compensated inverse-EPS computed-torque controller.

The model scalar action is the authoritative steering-position target. The
plan contributes only the local rate and acceleration at that same action
point; no later path sample can move the target or start a turn early. The
measured rack is predicted through the physical actuator delay, then inverse
rack dynamics cancel the measured load and place both tracking-error poles at
the rack's identified physical damping rate.

This is not an arrival-time controller and has no tracking-versus-smoothness
cost. Position error therefore asks for whatever torque the physical model
requires now, including full normalized authority. The exact Hyundai 409/4/7
limiter remains the sole command-smoothing authority.
"""

from __future__ import annotations

import math

import numpy as np

from openpilot.selfdrive.controls.lib.blatv2.candidate_common import (
  CandidateWorkspace,
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
  """No-dial computed-torque inverse with physical rack-pole feedback."""

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
    actuation_delay: float,
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
        float(actuation_delay) + MODEL_ACTION_OFFSET
        if action_time is None
        else float(action_time)
      )
      if not math.isfinite(action_sample_time) or action_sample_time < 0.0:
        raise ValueError("model action point must be finite and non-negative")

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
        float(actuation_delay),
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

      aligning = self.twin.aligning_torque_values(
        self.predicted_state.angle_deg,
        action_speed,
        align_inputs,
      )
      friction = measured_rack_friction(
        self.predicted_state.rate_deg_s,
        required_acceleration,
        self.twin.params.t_breakaway,
        self.params.kinetic_friction,
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

      torque_target = min(max(raw_command, -1.0), 1.0)
      command = self.twin.apply_slew(state.applied_torque, torque_target)
      result.command_torque = command
      result.raw_command_torque = torque_target
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
      result.slew_constrained = command != torque_target
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
