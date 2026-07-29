"""Delay-compensated inverse-EPS computed-torque controller.

The model scalar action is the authoritative steering-position target. The
plan supplies coherent rate/acceleration at that action point and a future
physical-torque workspace. Future samples may start using otherwise-idle slew
capacity only when their same-direction demand is unreachable if the controller
waits; they never move the scalar-pinned position target. The measured rack is
predicted through the physical actuator delay, then inverse rack dynamics
cancel the measured load and place both tracking-error poles at the rack's
identified physical damping rate.

This is not an arrival-time controller and has no tracking-versus-smoothness
cost. Position error therefore asks for whatever torque the physical model
requires now, including full normalized authority. The exact Hyundai 409/4/7
limiter remains the sole command-smoothing authority.
"""

from __future__ import annotations

import math

import numpy as np

from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.blatv2.candidate_common import (
  CandidateWorkspace,
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
    self.command_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.command_next_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.breakaway_direction = 0.0
    self.breakaway_persistence_frames = 0
    self.breakaway_active = False

  def reset(self) -> None:
    self.result.invalidate()
    self.breakaway_direction = 0.0
    self.breakaway_persistence_frames = 0
    self.breakaway_active = False

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
    curvature_error: float,
    departure_direction: float,
  ) -> float:
    """Return friction for a real stick/slip episode, never plan jitter.

    Route bb showed that treating every sub-quantum rate as static applied
    nearly full breakaway on 99.6% of highway frames. Static compensation is
    now armed only by one model period of persistent, same-direction tracking
    error while the measured rack is inside half a sensor quantum. It remains
    active through the first full quantum so the 0.09 -> 0.03 transition is
    continuous, then moving-rack compensation is kinetic only.
    """
    measured_rate = float(measured_rate_deg_s)
    error = float(curvature_error)
    departure = float(departure_direction)
    direction = 0.0 if departure == 0.0 else math.copysign(1.0, departure)
    stationary = abs(measured_rate) <= 0.5 * RACK_RATE_QUANTUM_DEG_S
    error_large = abs(error) >= self.params.sigma_curvature
    same_direction = direction != 0.0 and direction == self.breakaway_direction

    if self.breakaway_active:
      if (
        direction == 0.0
        or direction != self.breakaway_direction
        or not error_large
      ):
        self.breakaway_active = False
        self.breakaway_persistence_frames = 0
        self.breakaway_direction = direction
      elif abs(measured_rate) >= RACK_RATE_QUANTUM_DEG_S:
        self.breakaway_active = False
        self.breakaway_persistence_frames = 0
    elif stationary and error_large and direction != 0.0:
      if same_direction:
        self.breakaway_persistence_frames += 1
      else:
        self.breakaway_direction = direction
        self.breakaway_persistence_frames = 1
      if self.breakaway_persistence_frames >= int(round(DT_MDL / DT_CTRL)):
        self.breakaway_active = True
    else:
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

  def _unreachable_future_demand(
    self,
    applied_torque: float,
    action_sample_time: float,
    position_direction: float,
  ) -> tuple[bool, float, float]:
    """Find the first same-direction demand outside the exact slew envelope."""
    if position_direction == 0.0:
      return False, 0.0, 0.0
    positive = float(applied_torque)
    negative = float(applied_torque)
    decision_index = max(
      int(math.ceil(action_sample_time / DECISION_DT - 1e-12)), 0,
    )
    decision_index = min(decision_index, self.workspace.decision_count - 1)
    max_steps = int(math.ceil(
      float(self.workspace.decision_times[self.workspace.decision_count - 1])
      / DT_CTRL
    ))
    for step in range(1, max_steps + 1):
      positive = self.twin.apply_slew(positive, 1.0)
      negative = self.twin.apply_slew(negative, -1.0)
      reachable_time = step * DT_CTRL
      while (
        decision_index < self.workspace.decision_count
        and float(self.workspace.decision_times[decision_index])
        <= reachable_time + 1e-12
      ):
        # The workspace retains static departure friction for the retired MPC.
        # A future *prediction* cannot know that the measured rack is stuck,
        # so the active controller removes that cell term and restores only
        # kinetic friction when the planned rack is actually moving. Static
        # authority is owned exclusively by this controller's measured-rack
        # breakaway state machine.
        desired_rate = float(
          self.workspace.desired_rates[decision_index]
        )
        desired_acceleration = float(
          self.workspace.desired_accelerations[decision_index]
        )
        motion_direction = desired_rate
        if motion_direction == 0.0:
          motion_direction = desired_acceleration
        future_friction = (
          0.0
          if motion_direction == 0.0
          else math.copysign(
            self.params.kinetic_friction, motion_direction,
          )
        )
        demand = min(max(
          float(self.workspace.feedforward[decision_index])
          - float(self.workspace.friction_torques[decision_index])
          + future_friction,
          -1.0,
        ), 1.0)
        if (
          math.copysign(1.0, demand) == position_direction
          and (
            demand > positive + 1e-12
            or demand < negative - 1e-12
          )
        ):
          return True, demand, float(
            self.workspace.decision_times[decision_index]
          )
        decision_index += 1
      if decision_index >= self.workspace.decision_count:
        break
    return False, 0.0, 0.0

  def _next_angle(
    self,
    command: float,
    align_inputs: AlignInputs,
    disturbance_torque: float,
  ) -> float:
    state = self.command_state
    state.angle_deg = self.predicted_state.angle_deg
    state.rate_deg_s = self.predicted_state.rate_deg_s
    state.applied_torque = float(command)
    state.v_ego = self.predicted_state.v_ego
    self.twin.predict_held_state_into(
      state,
      DT_CTRL,
      align_inputs,
      disturbance_torque,
      self.command_next_state,
      DT_CTRL,
    )
    return self.command_next_state.angle_deg

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
      desired_curvature = self._interpolate(
        self.workspace.reference_curvatures, sample_position, count,
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
      predicted_curvature = self.twin.curvature_from_angle(
        self.predicted_state.angle_deg,
        action_speed,
        align_inputs,
      )
      friction = self._update_breakaway(
        state.rate_deg_s,
        desired_curvature - predicted_curvature,
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
      local_command = self.twin.apply_slew(
        state.applied_torque, local_target,
      )
      position_direction = (
        0.0 if angle_error == 0.0 else math.copysign(1.0, angle_error)
      )
      (
        horizon_assist,
        horizon_demand,
        horizon_demand_time,
      ) = self._unreachable_future_demand(
        state.applied_torque,
        action_sample_time,
        position_direction,
      )
      torque_target = horizon_demand if horizon_assist else local_target
      command = self.twin.apply_slew(state.applied_torque, torque_target)
      horizon_assist = bool(
        horizon_assist
        and position_direction * (command - local_command) > 0.0
      )
      if not horizon_assist:
        torque_target = local_target
        command = local_command
        horizon_demand = 0.0
        horizon_demand_time = 0.0

      # Prediction may spend torque earlier, but it may never move the wheel
      # past the scalar-pinned path. Restrict only the anticipatory increment;
      # the local inverse remains the fallback at the exact action point.
      no_lead_limited = False
      if horizon_assist:
        next_sample_position = (
          action_sample_time + DT_CTRL
        ) / DECISION_DT
        next_desired_angle = self._interpolate(
          self.workspace.desired_angles,
          next_sample_position,
          count,
        )
        if (
          position_direction
          * (
            self._next_angle(
              command, align_inputs, disturbance_torque,
            )
            - next_desired_angle
          )
          > 0.0
        ):
          no_lead_limited = True
          low = local_command
          high = command
          if position_direction < 0.0:
            low, high = high, low
          # Twelve deterministic bisections resolve far below one normalized
          # torque count without introducing a tuning tolerance.
          for _ in range(12):
            middle = 0.5 * (low + high)
            leads = (
              position_direction
              * (
                self._next_angle(
                  middle, align_inputs, disturbance_torque,
                )
                - next_desired_angle
              )
              > 0.0
            )
            if position_direction > 0.0:
              if leads:
                high = middle
              else:
                low = middle
            elif leads:
              low = middle
            else:
              high = middle
          command = low if position_direction > 0.0 else high
          # ``raw_command_torque`` is the final pre-actuator request, not the
          # rejected horizon demand. The bisection result is already inside
          # this frame's reachable interval, so requesting it reproduces the
          # selected command exactly through apply_slew. Keep the original
          # unreachable demand in its dedicated diagnostic.
          torque_target = command
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
      result.breakaway_active = self.breakaway_active
      result.breakaway_persistence_frames = (
        self.breakaway_persistence_frames
      )
      result.horizon_assist_active = horizon_assist
      result.horizon_torque_demand = horizon_demand
      result.horizon_demand_time_seconds = horizon_demand_time
      result.no_lead_limited = no_lead_limited
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
