"""Scalar-anchored, actuator-feasible inverse-EPS trajectory controller.

The model scalar remains the authoritative curvature at ``action_time``.
The surrounding plan supplies the time-indexed rack position, rate,
acceleration, and inverse physical torque on both sides of that anchor. The
measured rack is predicted only to ``prediction_delay``, the instant this
frame's request can affect it, and is compared with the reference at that same
instant. Earlier versions compared the rack at the physical-effect time with
the later scalar-action position. That made the wheel begin a turn early while
the error-grown authority map and one-frame limiter still delivered its useful
torque late.

The full inverse-torque path is now the command path, not a helper. The plant
twin solves the request that reaches the scalar action at its existing
authored timestamp, then rolls the same inverse law across the surrounding
plan so finite rack dynamics are part of the demand. A deterministic backward
reachability pass applies the exact asymmetric 409/4/7 transition physics and
moves each build, release, or sign handoff no earlier than required. One
linear residual term corrects measured and predicted rack error; steady
holding authority comes from the requested-path load, never from tracking
error remaining nonzero. A one-frame plant check prevents an anticipatory
increment from moving the rack beyond the time-aligned model reference.

There is no turn detector, persistence timer, preview boost, low-speed branch,
integral, or torque-rate smoothing controller. Smoothness is the single exact
actuator-feasible trajectory; swiftness is its latest feasible transition;
strength is the unclipped inverse demand up to the normalized torque limit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.blatv2.candidate_common import (
  CandidateWorkspace,
  MAX_DECISION_STEPS,
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


@dataclass(slots=True)
class _InverseTerms:
  angle_error: float = 0.0
  rate_error: float = 0.0
  required_acceleration: float = 0.0
  aligning: float = 0.0
  friction: float = 0.0
  nominal_dynamic: float = 0.0
  feedback: float = 0.0
  feedforward: float = 0.0
  dynamic: float = 0.0
  target: float = 0.0


class InverseEpsActionController:
  """One scalar-faithful, slew-feasible inverse-EPS command path."""

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
    self.command_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.command_next_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.rollout_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.rollout_next_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.arrival_source_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.arrival_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.local_terms = _InverseTerms()
    self.rollout_terms = _InverseTerms()
    self.trajectory_times = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.trajectory_demands = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.trajectory_targets = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.trajectory_constraint_indices = np.empty(
      MAX_DECISION_STEPS, dtype=np.int32,
    )

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

  def _torque_transition_time(self, source: float, target: float) -> float:
    """Exact continuous-time counterpart of :meth:`PlantTwin.apply_slew`."""
    start = min(max(float(source), -1.0), 1.0)
    finish = min(max(float(target), -1.0), 1.0)
    if start == finish:
      return 0.0
    frame_seconds = DT_CTRL * self.twin.params.steer_step
    build = self.twin.params.delta_up / self.twin.params.steer_max
    decay = self.twin.params.delta_down / self.twin.params.steer_max
    if start * finish >= 0.0:
      budget = build if abs(finish) > abs(start) else decay
      return abs(finish - start) / budget * frame_seconds
    return (
      abs(start) / decay + abs(finish) / build
    ) * frame_seconds

  def _latest_feasible_predecessor(
    self,
    desired_now: float,
    feasible_next: float,
    interval: float,
  ) -> float:
    """Move ``desired_now`` only enough to reach ``feasible_next`` on time."""
    current = min(max(float(desired_now), -1.0), 1.0)
    future = min(max(float(feasible_next), -1.0), 1.0)
    duration = float(interval)
    if not math.isfinite(duration) or duration <= 0.0:
      raise ValueError("trajectory interval must be finite and positive")

    # Solve the target's exact backward-reachable interval analytically. This
    # is the inverse of apply_slew's decay-then-build sign crossing: going
    # backward from a positive target first consumes build time down to zero,
    # then permits an opposite-sign predecessor at the decay rate (mirrored
    # for a negative target). Clamping the model demand to this interval is the
    # unique latest feasible predecessor, with no search or tuning parameter.
    frame_seconds = DT_CTRL * self.twin.params.steer_step
    build_rate = (
      self.twin.params.delta_up
      / self.twin.params.steer_max
      / frame_seconds
    )
    decay_rate = (
      self.twin.params.delta_down
      / self.twin.params.steer_max
      / frame_seconds
    )
    if future > 0.0:
      build_to_zero_time = future / build_rate
      lower = (
        future - build_rate * duration
        if duration <= build_to_zero_time
        else -decay_rate * (duration - build_to_zero_time)
      )
      upper = future + decay_rate * duration
    elif future < 0.0:
      build_to_zero_time = -future / build_rate
      lower = future - decay_rate * duration
      upper = (
        future + build_rate * duration
        if duration <= build_to_zero_time
        else decay_rate * (duration - build_to_zero_time)
      )
    else:
      lower = -decay_rate * duration
      upper = decay_rate * duration
    return min(max(current, max(lower, -1.0)), min(upper, 1.0))

  def _project_inverse_trajectory(
    self,
    physical_effect_time: float,
    local_target: float,
    action_target: float,
    action_time: float,
    align_inputs: AlignInputs,
    disturbance_torque: float,
  ) -> tuple[float, bool, float, float]:
    """Return the current target of the latest state/slew-feasible path."""
    count = self.workspace.decision_count
    last_time = float(self.workspace.decision_times[count - 1])
    start_time = min(max(float(physical_effect_time), 0.0), last_time)
    trajectory_count = min(
      int(math.floor((last_time - start_time) / DECISION_DT + 1e-12)) + 1,
      MAX_DECISION_STEPS,
    )
    if trajectory_count <= 0:
      return float(local_target), False, 0.0, 0.0

    rollout = self.rollout_state
    rollout.angle_deg = self.predicted_state.angle_deg
    rollout.rate_deg_s = self.predicted_state.rate_deg_s
    rollout.applied_torque = self.predicted_state.applied_torque
    rollout.v_ego = self.predicted_state.v_ego
    next_rollout = self.rollout_next_state
    for index in range(trajectory_count):
      sample_time = start_time + index * DECISION_DT
      self.trajectory_times[index] = sample_time
      sample_position = sample_time / DECISION_DT
      desired_angle = self._interpolate(
        self.workspace.desired_angles, sample_position, count,
      )
      desired_rate = self._interpolate(
        self.workspace.desired_rates, sample_position, count,
      )
      desired_acceleration = self._interpolate(
        self.workspace.desired_accelerations, sample_position, count,
      )
      sample_speed = self._interpolate(
        self.workspace.reference_speeds, sample_position, count,
      )
      rollout.v_ego = sample_speed
      self._fill_inverse_terms(
        self.rollout_terms,
        rollout,
        desired_angle,
        desired_rate,
        desired_acceleration,
        sample_speed,
        align_inputs,
        disturbance_torque,
      )
      demand = self.rollout_terms.target
      if index == 0:
        # The action-point solver supplies the first target; the forward
        # rollout then exposes any later state-feedback demand caused by it.
        demand = float(action_target)
      self.trajectory_demands[index] = min(max(demand, -1.0), 1.0)
      if index + 1 < trajectory_count:
        self.twin.predict_constant_request_into(
          rollout,
          DECISION_DT,
          self.trajectory_demands[index],
          align_inputs,
          disturbance_torque,
          next_rollout,
          DT_CTRL,
        )
        rollout, next_rollout = next_rollout, rollout

    last = trajectory_count - 1
    self.trajectory_targets[last] = self.trajectory_demands[last]
    self.trajectory_constraint_indices[last] = last
    for index in range(last - 1, -1, -1):
      desired = float(self.trajectory_demands[index])
      future = float(self.trajectory_targets[index + 1])
      projected = self._latest_feasible_predecessor(
        desired, future, DECISION_DT,
      )
      self.trajectory_targets[index] = projected
      if projected == desired:
        self.trajectory_constraint_indices[index] = index
      else:
        self.trajectory_constraint_indices[index] = (
          self.trajectory_constraint_indices[index + 1]
        )

    target = float(self.trajectory_targets[0])
    projected = target != float(self.trajectory_demands[0])
    active = target != float(local_target)
    if not active:
      return target, False, 0.0, 0.0
    if projected:
      constraint_index = int(self.trajectory_constraint_indices[0])
      binding_demand = float(self.trajectory_demands[constraint_index])
      binding_time = float(self.trajectory_times[constraint_index])
    else:
      binding_demand = float(action_target)
      binding_time = float(action_time)
    return (
      target,
      True,
      binding_demand,
      binding_time,
    )

  def _action_point_target(
    self,
    action_time: float,
    physical_effect_time: float,
    desired_action_angle: float,
    action_speed: float,
    local_target: float,
    align_inputs: AlignInputs,
    disturbance_torque: float,
  ) -> float:
    """Solve the constant request that reaches the scalar action on time.

    The model owns the terminal angle and timestamp. The plant owns how much
    torque is needed to get there. Receding-horizon evaluation at 100 Hz means
    only this frame's exact slew step is committed; subsequent frames solve
    again from measured response. Twelve fixed bisections resolve finer than
    one Hyundai torque count and introduce no response-time or feel constant.
    """
    duration = float(action_time) - float(physical_effect_time)
    if duration <= 0.0:
      return float(local_target)
    low = -1.0
    high = 1.0
    source = self.arrival_source_state
    source.angle_deg = self.predicted_state.angle_deg
    source.rate_deg_s = self.predicted_state.rate_deg_s
    source.applied_torque = self.predicted_state.applied_torque
    source.v_ego = float(action_speed)
    self.twin.predict_constant_request_into(
      source,
      duration,
      low,
      align_inputs,
      disturbance_torque,
      self.arrival_state,
      DT_CTRL,
    )
    low_angle = self.arrival_state.angle_deg
    self.twin.predict_constant_request_into(
      source,
      duration,
      high,
      align_inputs,
      disturbance_torque,
      self.arrival_state,
      DT_CTRL,
    )
    high_angle = self.arrival_state.angle_deg
    if desired_action_angle <= low_angle:
      return low
    if desired_action_angle >= high_angle:
      return high
    for _ in range(12):
      request = 0.5 * (low + high)
      self.twin.predict_constant_request_into(
        source,
        duration,
        request,
        align_inputs,
        disturbance_torque,
        self.arrival_state,
        DT_CTRL,
      )
      if self.arrival_state.angle_deg < float(desired_action_angle):
        low = request
      else:
        high = request
    return 0.5 * (low + high)

  def _fill_inverse_terms(
    self,
    terms: _InverseTerms,
    rack_state: PlantState,
    desired_angle: float,
    desired_rate: float,
    desired_acceleration: float,
    speed: float,
    align_inputs: AlignInputs,
    disturbance_torque: float,
  ) -> None:
    """Evaluate the one inverse-EPS law for measured or predicted rack state."""
    angle_error = float(desired_angle) - rack_state.angle_deg
    rate_error = float(desired_rate) - rack_state.rate_deg_s
    rate_damping = (
      self.twin.params.delta_up
      / self.twin.params.steer_max
      / RACK_RATE_QUANTUM_DEG_S
    )
    feedback = self.params.tracking_stiffness * angle_error
    feedback += rate_damping * rate_error
    required_acceleration = (
      float(desired_acceleration)
      + self.twin.params.b_steer * rate_error
      + self.twin.params.k_t * feedback
    )
    aligning = self.twin.aligning_torque_values(
      desired_angle, speed, align_inputs,
    )
    friction = measured_rack_friction(
      rack_state.rate_deg_s,
      required_acceleration,
      self.twin.params.t_breakaway,
      self.params.kinetic_friction,
    )
    nominal_dynamic = (
      float(desired_acceleration)
      + self.twin.params.b_steer * float(desired_rate)
    ) / self.twin.params.k_t
    feedforward = (
      aligning
      + float(disturbance_torque)
      + friction
      + nominal_dynamic
    )
    terms.angle_error = angle_error
    terms.rate_error = rate_error
    terms.required_acceleration = required_acceleration
    terms.aligning = aligning
    terms.friction = friction
    terms.nominal_dynamic = nominal_dynamic
    terms.feedback = feedback
    terms.feedforward = feedforward
    terms.dynamic = nominal_dynamic + feedback
    terms.target = feedforward + feedback

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

  def _limit_anticipatory_lead(
    self,
    applied_torque: float,
    local_command: float,
    trajectory_command: float,
    desired_next_angle: float,
    align_inputs: AlignInputs,
    disturbance_torque: float,
  ) -> tuple[float, bool]:
    """Prevent only the future-trajectory increment from leading the path."""
    path_delta = desired_next_angle - self.predicted_state.angle_deg
    if path_delta == 0.0:
      return local_command, trajectory_command != local_command
    direction = math.copysign(1.0, path_delta)
    if direction * (trajectory_command - local_command) <= 0.0:
      return trajectory_command, False
    if direction * (
      self._next_angle(
        trajectory_command, align_inputs, disturbance_torque,
      )
      - desired_next_angle
    ) <= 0.0:
      return trajectory_command, False

    # The local time-aligned inverse is the lower-authority bound. Restricting
    # the future increment can never make the local controller less capable.
    safe = float(local_command)
    leading = float(trajectory_command)
    if direction < 0.0:
      safe, leading = leading, safe
    for _ in range(20):
      middle = 0.5 * (safe + leading)
      leads = direction * (
        self._next_angle(middle, align_inputs, disturbance_torque)
        - desired_next_angle
      ) > 0.0
      if direction > 0.0:
        if leads:
          leading = middle
        else:
          safe = middle
      elif leads:
        safe = middle
      else:
        leading = middle
    target = safe if direction > 0.0 else leading
    reachable_low = self.twin.apply_slew(applied_torque, -1.0)
    reachable_high = self.twin.apply_slew(applied_torque, 1.0)
    return min(max(target, reachable_low), reachable_high), True

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

      count = self.workspace.decision_count
      # The scalar is anchored at action_sample_time, but the command computed
      # now first affects the rack at physical_prediction_delay. Compare states
      # at that same physical instant; never ask the earlier rack state to be
      # at the later scalar position.
      tracking_sample_time = min(
        physical_prediction_delay,
        float(self.workspace.decision_times[count - 1]),
      )
      sample_position = tracking_sample_time / DECISION_DT
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
      action_position = action_sample_time / DECISION_DT
      desired_action_angle = self._interpolate(
        self.workspace.desired_angles, action_position, count,
      )
      action_point_speed = self._interpolate(
        self.workspace.reference_speeds, action_position, count,
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

      # The same inverse law evaluates the measured state now and each
      # predicted state in the horizon. The rollout therefore exposes future
      # tracking demand before the slew projection, instead of assuming the
      # rack will remain exactly on the feedforward trajectory.
      self._fill_inverse_terms(
        self.local_terms,
        self.predicted_state,
        desired_angle,
        desired_rate,
        desired_acceleration,
        action_speed,
        align_inputs,
        disturbance_torque,
      )
      raw_command = self.local_terms.target
      if not math.isfinite(raw_command):
        result.invalidate(state.applied_torque, CandidateStatus.NON_CONVERGED)
        result.candidate_count = 1
        result.available_schedule_count = 1
        return result

      local_target = min(max(raw_command, -1.0), 1.0)
      local_command = self.twin.apply_slew(
        state.applied_torque, local_target,
      )
      action_target = self._action_point_target(
        action_sample_time,
        tracking_sample_time,
        desired_action_angle,
        action_point_speed,
        local_target,
        align_inputs,
        disturbance_torque,
      )
      (
        trajectory_target,
        trajectory_active,
        trajectory_demand,
        trajectory_demand_time,
      ) = self._project_inverse_trajectory(
        tracking_sample_time,
        local_target,
        action_target,
        action_sample_time,
        align_inputs,
        disturbance_torque,
      )
      trajectory_command = self.twin.apply_slew(
        state.applied_torque, trajectory_target,
      )
      no_lead_limited = False
      final_target = trajectory_target
      if trajectory_active and trajectory_command != local_command:
        next_sample_position = (
          tracking_sample_time + DT_CTRL
        ) / DECISION_DT
        desired_next_angle = self._interpolate(
          self.workspace.desired_angles,
          next_sample_position,
          count,
        )
        final_target, no_lead_limited = self._limit_anticipatory_lead(
          state.applied_torque,
          local_command,
          trajectory_command,
          desired_next_angle,
          align_inputs,
          disturbance_torque,
        )
      command = self.twin.apply_slew(
        state.applied_torque, final_target,
      )
      trajectory_active = bool(
        trajectory_active
        and (
          command != local_command
          if no_lead_limited
          else trajectory_target != local_target
        )
      )
      if not trajectory_active:
        trajectory_demand = 0.0
        trajectory_demand_time = 0.0

      result.command_torque = command
      result.raw_command_torque = final_target
      result.feedforward_torque = self.local_terms.feedforward
      result.feedback_torque = self.local_terms.feedback
      result.desired_angle_deg = desired_angle
      result.desired_rate_deg_s = desired_rate
      result.desired_acceleration_deg_s2 = desired_acceleration
      result.predicted_angle_deg = self.predicted_state.angle_deg
      result.predicted_rate_deg_s = self.predicted_state.rate_deg_s
      result.required_acceleration_deg_s2 = (
        self.local_terms.required_acceleration
      )
      result.action_speed_mps = action_speed
      result.aligning_torque = self.local_terms.aligning
      result.friction_torque = self.local_terms.friction
      result.dynamic_torque = self.local_terms.dynamic
      result.action_time_seconds = action_sample_time
      result.prediction_delay_seconds = physical_prediction_delay
      result.slew_constrained = command != final_target
      result.breakaway_active = False
      result.breakaway_persistence_frames = 0
      # Re-activate the original wire meanings. These diagnostics describe the
      # future demand that made the current exact-slew trajectory differ from
      # the local inverse; they are no longer a second command authority.
      result.horizon_assist_active = trajectory_active
      result.horizon_torque_demand = trajectory_demand
      result.horizon_demand_time_seconds = trajectory_demand_time
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
