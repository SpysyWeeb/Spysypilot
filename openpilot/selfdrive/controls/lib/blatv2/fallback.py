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
)
from openpilot.selfdrive.controls.lib.blatv2.controller import (
  DECISION_DT,
  CandidateResult,
  CandidateStatus,
  ControllerParams,
  ObserverStatus,
)
from openpilot.selfdrive.controls.lib.blatv2.plant import (
  AlignRuntimeTerms,
  AlignInputs,
  PlantSensitivity,
  PlantState,
  PlantTwin,
  RACK_RATE_QUANTUM_DEG_S,
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
    if twin.kinetic_friction != controller_params.kinetic_friction:
      raise ValueError(
        "controller and plant must share one kinetic-friction value"
      )
    self.twin = twin
    self.params = controller_params
    self.workspace = CandidateWorkspace() if workspace is None else workspace
    self.result = CandidateResult()
    self.predicted_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.command_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.command_next_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.arrival_source_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.arrival_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.arrival_align_terms = AlignRuntimeTerms()
    self.arrival_sensitivity = PlantSensitivity()
    self.arrival_low_request = -1.0
    self.arrival_high_request = 1.0
    self.local_terms = _InverseTerms()
    self.trajectory_demands = [0.0] * MAX_DECISION_STEPS
    self.command_history_capacity = int(
      math.ceil(self.twin.params.actuation_delay / DT_CTRL)
    )
    self.command_history = [0.0] * max(
      self.command_history_capacity, 1,
    )
    self.command_history_start = 0
    self.command_history_count = 0
    frame_seconds = DT_CTRL * self.twin.params.steer_step
    self.trajectory_build_rate = (
      self.twin.params.delta_up
      / self.twin.params.steer_max
      / frame_seconds
    )
    self.trajectory_decay_rate = (
      self.twin.params.delta_down
      / self.twin.params.steer_max
      / frame_seconds
    )
    self.trajectory_build_delta = (
      self.trajectory_build_rate * DECISION_DT
    )
    self.trajectory_decay_delta = (
      self.trajectory_decay_rate * DECISION_DT
    )

  def reset(self) -> None:
    self.result.invalidate()
    self.command_history_start = 0
    self.command_history_count = 0

  def _remember_applied_torque(self, applied_torque: float) -> None:
    """Append the last hardware/safety-limited command to the delay line."""
    if self.command_history_capacity == 0:
      return
    if self.command_history_count < self.command_history_capacity:
      index = (
        self.command_history_start + self.command_history_count
      ) % self.command_history_capacity
      self.command_history_count += 1
    else:
      index = self.command_history_start
      self.command_history_start = (
        self.command_history_start + 1
      ) % self.command_history_capacity
    self.command_history[index] = float(applied_torque)

  def _predict_to_effect_time(
    self,
    state: PlantState,
    duration: float,
    align_inputs: AlignInputs,
    disturbance_torque: float,
  ) -> None:
    """Advance through commands already committed inside the pure delay."""
    self.twin.prepare_align_runtime_terms(
      state.v_ego,
      align_inputs,
      self.arrival_align_terms,
    )
    self.twin.predict_applied_history_prepared_into(
      state,
      duration,
      self.command_history,
      self.command_history_start,
      self.command_history_count,
      disturbance_torque,
      self.arrival_align_terms,
      self.predicted_state,
      DT_CTRL,
    )

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

  def _constant_request_endpoint(
    self,
    source: float,
    direction: float,
    frame_count: int,
  ) -> float:
    """Exact endpoint of repeated full-authority slew applications.

    This is the closed form of calling ``apply_slew(source, direction)`` once
    per controller frame. It is used only to remove requests that produce an
    identical terminal rollout; the plant rollout itself still applies the
    limiter frame by frame.
    """
    start = min(max(float(source), -1.0), 1.0)
    sign = math.copysign(1.0, float(direction))
    count = int(frame_count)
    if count <= 0:
      return start
    build = self.twin.params.delta_up / self.twin.params.steer_max
    decay = self.twin.params.delta_down / self.twin.params.steer_max
    oriented_start = sign * start
    if oriented_start >= 0.0:
      return sign * min(oriented_start + count * build, 1.0)
    decay_frames = -oriented_start / decay
    if count <= decay_frames:
      remaining = -oriented_start - count * decay
      return -sign * remaining
    return sign * min(
      (count - decay_frames) * build,
      1.0,
    )

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
  ) -> tuple[float, bool, float, float]:
    """Return the current target of one planned, slew-feasible torque path.

    The future cells are the model-authored inverse-EPS feedforward, not a
    second closed-loop controller simulated against its own predictions.
    Current measured-rack error is already applied once in ``action_target``;
    feeding predicted future error backward through the slew projector makes
    the present request chase corrections that have not happened. Receding
    100 Hz evaluation handles the real residual on the next frame.
    """
    count = self.workspace.decision_count
    last_time = self.workspace.decision_times[count - 1]
    start_time = float(physical_effect_time)
    if start_time < 0.0:
      start_time = 0.0
    elif start_time > last_time:
      start_time = last_time
    trajectory_count = min(
      int(math.floor((last_time - start_time) / DECISION_DT + 1e-12)) + 1,
      MAX_DECISION_STEPS,
    )
    if trajectory_count <= 0:
      return float(local_target), False, 0.0, 0.0

    start_position = start_time / DECISION_DT
    lower_index = int(start_position)
    sample_fraction = start_position - lower_index
    for index in range(trajectory_count):
      sample_index = min(lower_index + index, count - 1)
      upper_index = min(sample_index + 1, count - 1)
      lower_demand = self.workspace.feedforward[sample_index]
      demand = lower_demand + sample_fraction * (
        self.workspace.feedforward[upper_index]
        - lower_demand
      )
      if index == 0:
        # The scalar-pinned terminal solve supplies this frame's planned target
        # with the one measured-rack correction already included.
        demand = float(action_target)
      if demand < -1.0:
        demand = -1.0
      elif demand > 1.0:
        demand = 1.0
      self.trajectory_demands[index] = demand

    last = trajectory_count - 1
    future = self.trajectory_demands[last]
    constraint_index = last
    build_rate = self.trajectory_build_rate
    decay_rate = self.trajectory_decay_rate
    build_delta = self.trajectory_build_delta
    decay_delta = self.trajectory_decay_delta
    for index in range(last - 1, -1, -1):
      desired = self.trajectory_demands[index]
      if future > 0.0:
        build_to_zero_time = future / build_rate
        lower = (
          future - build_delta
          if DECISION_DT <= build_to_zero_time
          else -decay_rate * (
            DECISION_DT - build_to_zero_time
          )
        )
        upper = future + decay_delta
      elif future < 0.0:
        build_to_zero_time = -future / build_rate
        lower = future - decay_delta
        upper = (
          future + build_delta
          if DECISION_DT <= build_to_zero_time
          else decay_rate * (
            DECISION_DT - build_to_zero_time
          )
        )
      else:
        lower = -decay_delta
        upper = decay_delta
      if lower < -1.0:
        lower = -1.0
      if upper > 1.0:
        upper = 1.0
      if desired < lower:
        projected = lower
      elif desired > upper:
        projected = upper
      else:
        projected = desired
      if projected == desired:
        constraint_index = index
      future = projected

    target = future
    projected = target != self.trajectory_demands[0]
    active = target != float(local_target)
    if not active:
      return target, False, 0.0, 0.0
    if projected:
      binding_demand = self.trajectory_demands[constraint_index]
      binding_time = start_time + constraint_index * DECISION_DT
    else:
      binding_demand = float(action_target)
      binding_time = float(action_time)
    return (
      target,
      True,
      binding_demand,
      binding_time,
    )

  def _solve_terminal_state_request(
    self,
    source: PlantState,
    duration: float,
    target_angle: float,
    target_rate: float,
    nominal_request: float,
    disturbance_torque: float,
    direction: float,
    enforce_rate: bool = True,
  ) -> float:
    """Return the least request reaching both terminal rack constraints.

    The rack and limiter are piecewise affine in a constant request. Their
    exact local sensitivities solve terminal angle and rate together. Earlier
    code independently rebuilt the identical 30-step plant rollout for each
    constraint, then selected the more authoritative request. In the oriented
    direction of motion that selection is exactly the smallest request for
    which both constraints hold, so one safeguarded solve is equivalent.

    Equal-cost roots resolve toward the angle constraint by the fixed
    ``>=`` comparison below. Secant and midpoint fallbacks preserve
    convergence across slew, stiction, and zero-rate branch boundaries.
    Precision is derived from one quarter of a physical Hyundai torque count;
    it is numerical, not a steering-feel dial. Three evaluated corrections
    plus one final local-affine extrapolation cap plant work, so
    an unusual stiction branch cannot consume a variable part of the 10 ms
    control frame.
    """
    low_request = self.arrival_low_request
    high_request = self.arrival_high_request
    target_angle_value = float(target_angle)
    target_rate_value = float(target_rate)
    direction_value = float(direction)
    endpoint_prepared = direction_value != 0.0
    if endpoint_prepared:
      motion_direction = math.copysign(1.0, direction_value)
      oriented_low = min(
        motion_direction * low_request,
        motion_direction * high_request,
      )
      oriented_high = max(
        motion_direction * low_request,
        motion_direction * high_request,
      )
      # Fast maneuvers dominate the fault route, and most action targets are
      # physically unreachable inside one scalar-action interval. Prove that
      # from the maximum reachable endpoint first: one rollout then replaces
      # the old nominal-plus-endpoint pair on every saturated frame.
      endpoint_request = motion_direction * oriented_high
      self.twin.predict_constant_request_prepared_into(
        source,
        duration,
        endpoint_request,
        disturbance_torque,
        self.arrival_align_terms,
        self.arrival_state,
        DT_CTRL,
      )
      high_angle_error = motion_direction * (
        self.arrival_state.angle_deg - target_angle_value
      )
      high_rate_error = (
        motion_direction * (
          self.arrival_state.rate_deg_s - target_rate_value
        )
        if enforce_rate else math.inf
      )
      if high_angle_error < 0.0 or high_rate_error < 0.0:
        return motion_direction

    bounded_nominal = min(max(
      min(max(float(nominal_request), -1.0), 1.0),
      low_request,
    ), high_request)
    self.twin.predict_constant_request_sensitivity_into(
      source,
      duration,
      bounded_nominal,
      disturbance_torque,
      self.arrival_align_terms,
      self.arrival_state,
      self.arrival_sensitivity,
      DT_CTRL,
    )
    if direction_value == 0.0:
      # A steady authored rack target has no motion sign. Its sole angle root
      # still has a physical direction relative to the nominal terminal state.
      angle_delta = target_angle_value - self.arrival_state.angle_deg
      if angle_delta == 0.0:
        return bounded_nominal
      direction_value = angle_delta
      motion_direction = math.copysign(1.0, direction_value)
      oriented_low = min(
        motion_direction * low_request,
        motion_direction * high_request,
      )
      oriented_high = max(
        motion_direction * low_request,
        motion_direction * high_request,
      )
    oriented_nominal = motion_direction * bounded_nominal
    angle_error = motion_direction * (
      self.arrival_state.angle_deg - target_angle_value
    )
    rate_error = (
      motion_direction * (
        self.arrival_state.rate_deg_s - target_rate_value
      )
      if enforce_rate else math.inf
    )
    angle_sensitivity = self.arrival_sensitivity.angle_per_torque
    rate_sensitivity = self.arrival_sensitivity.rate_per_torque
    nominal_satisfies = angle_error >= 0.0 and rate_error >= 0.0

    if nominal_satisfies:
      oriented_high = oriented_nominal
      high_angle_error = angle_error
      high_rate_error = rate_error
      request = motion_direction * oriented_low
      self.twin.predict_constant_request_sensitivity_into(
        source,
        duration,
        request,
        disturbance_torque,
        self.arrival_align_terms,
        self.arrival_state,
        self.arrival_sensitivity,
        DT_CTRL,
      )
      low_angle_error = motion_direction * (
        self.arrival_state.angle_deg - target_angle_value
      )
      low_rate_error = (
        motion_direction * (
          self.arrival_state.rate_deg_s - target_rate_value
        )
        if enforce_rate else math.inf
      )
      if low_angle_error >= 0.0 and low_rate_error >= 0.0:
        # Match the former independent solvers' full-authority saturation
        # result when even the opposite reachable endpoint exceeds the target.
        return -motion_direction
      oriented_current = oriented_low
      angle_error = low_angle_error
      rate_error = low_rate_error
      angle_sensitivity = self.arrival_sensitivity.angle_per_torque
      rate_sensitivity = self.arrival_sensitivity.rate_per_torque
    else:
      oriented_low = oriented_nominal
      low_angle_error = angle_error
      low_rate_error = rate_error
      if not endpoint_prepared:
        request = motion_direction * oriented_high
        self.twin.predict_constant_request_sensitivity_into(
          source,
          duration,
          request,
          disturbance_torque,
          self.arrival_align_terms,
          self.arrival_state,
          self.arrival_sensitivity,
          DT_CTRL,
        )
        high_angle_error = motion_direction * (
          self.arrival_state.angle_deg - target_angle_value
        )
        high_rate_error = (
          motion_direction * (
            self.arrival_state.rate_deg_s - target_rate_value
          )
          if enforce_rate else math.inf
        )
        if high_angle_error < 0.0 or high_rate_error < 0.0:
          return motion_direction
      oriented_current = oriented_nominal

    torque_resolution = 0.25 / self.twin.params.steer_max
    for _ in range(3):
      if oriented_high - oriented_low <= torque_resolution:
        break

      if angle_sensitivity > 0.0 and math.isfinite(angle_sensitivity):
        angle_root = (
          oriented_current - angle_error / angle_sensitivity
        )
      elif high_angle_error > low_angle_error:
        angle_root = oriented_low + (
          -low_angle_error
          * (oriented_high - oriented_low)
          / (high_angle_error - low_angle_error)
        )
      else:
        angle_root = 0.5 * (oriented_low + oriented_high)

      if not enforce_rate:
        rate_root = oriented_low
      elif rate_sensitivity > 0.0 and math.isfinite(rate_sensitivity):
        rate_root = (
          oriented_current - rate_error / rate_sensitivity
        )
      elif high_rate_error > low_rate_error:
        rate_root = oriented_low + (
          -low_rate_error
          * (oriented_high - oriented_low)
          / (high_rate_error - low_rate_error)
        )
      else:
        rate_root = 0.5 * (oriented_low + oriented_high)

      # Both terminal constraints become monotone in oriented request. The
      # later root is therefore the exact counterpart of max(angle, rate) for
      # positive motion and min(angle, rate) for negative motion.
      proposed = max(angle_root, rate_root)
      if (
        not oriented_low < proposed < oriented_high
        or proposed == oriented_current
        or not math.isfinite(proposed)
      ):
        proposed = 0.5 * (oriented_low + oriented_high)

      request = motion_direction * proposed
      self.twin.predict_constant_request_sensitivity_into(
        source,
        duration,
        request,
        disturbance_torque,
        self.arrival_align_terms,
        self.arrival_state,
        self.arrival_sensitivity,
        DT_CTRL,
      )
      proposed_angle_error = motion_direction * (
        self.arrival_state.angle_deg - target_angle_value
      )
      proposed_rate_error = (
        motion_direction * (
          self.arrival_state.rate_deg_s - target_rate_value
        )
        if enforce_rate else math.inf
      )
      proposed_angle_sensitivity = (
        self.arrival_sensitivity.angle_per_torque
      )
      proposed_rate_sensitivity = (
        self.arrival_sensitivity.rate_per_torque
      )
      angle_close = (
        proposed_angle_error >= 0.0
        or (
          proposed_angle_sensitivity > 0.0
          and proposed_angle_error
          >= -proposed_angle_sensitivity * torque_resolution
        )
      )
      rate_close = not enforce_rate or (
        proposed_rate_error >= 0.0
        or (
          proposed_rate_sensitivity > 0.0
          and proposed_rate_error
          >= -proposed_rate_sensitivity * torque_resolution
        )
      )
      if angle_close and rate_close and (
        abs(proposed_angle_error)
        <= proposed_angle_sensitivity * torque_resolution
        or (
          enforce_rate
          and abs(proposed_rate_error)
          <= proposed_rate_sensitivity * torque_resolution
        )
      ):
        return request

      if proposed_angle_error >= 0.0 and proposed_rate_error >= 0.0:
        oriented_high = proposed
        high_angle_error = proposed_angle_error
        high_rate_error = proposed_rate_error
      else:
        oriented_low = proposed
        low_angle_error = proposed_angle_error
        low_rate_error = proposed_rate_error
      oriented_current = proposed
      angle_error = proposed_angle_error
      rate_error = proposed_rate_error
      angle_sensitivity = proposed_angle_sensitivity
      rate_sensitivity = proposed_rate_sensitivity

    # The selected branch is affine. Use its last exact sensitivity for the
    # final sub-count estimate without spending another complete rack rollout;
    # clamp to the proven bracket if the estimate crosses a branch boundary.
    if angle_sensitivity > 0.0 and math.isfinite(angle_sensitivity):
      angle_root = oriented_current - angle_error / angle_sensitivity
    elif high_angle_error > low_angle_error:
      angle_root = oriented_low + (
        -low_angle_error
        * (oriented_high - oriented_low)
        / (high_angle_error - low_angle_error)
      )
    else:
      angle_root = 0.5 * (oriented_low + oriented_high)
    if not enforce_rate:
      rate_root = oriented_low
    elif rate_sensitivity > 0.0 and math.isfinite(rate_sensitivity):
      rate_root = oriented_current - rate_error / rate_sensitivity
    elif high_rate_error > low_rate_error:
      rate_root = oriented_low + (
        -low_rate_error
        * (oriented_high - oriented_low)
        / (high_rate_error - low_rate_error)
      )
    else:
      rate_root = 0.5 * (oriented_low + oriented_high)
    final_request = max(angle_root, rate_root)
    if not math.isfinite(final_request):
      final_request = 0.5 * (oriented_low + oriented_high)
    final_request = min(max(
      final_request,
      oriented_low,
    ), oriented_high)
    hardware_count = 1.0 / self.twin.params.steer_max
    if abs(final_request - oriented_current) > hardware_count:
      # A branch extrapolation larger than one command count is not locally
      # trustworthy. Verify only those rare cases, then extrapolate once more
      # from the observed branch. Ordinary frames retain the fixed fast path.
      request = motion_direction * final_request
      self.twin.predict_constant_request_sensitivity_into(
        source,
        duration,
        request,
        disturbance_torque,
        self.arrival_align_terms,
        self.arrival_state,
        self.arrival_sensitivity,
        DT_CTRL,
      )
      angle_error = motion_direction * (
        self.arrival_state.angle_deg - target_angle_value
      )
      rate_error = (
        motion_direction * (
          self.arrival_state.rate_deg_s - target_rate_value
        )
        if enforce_rate else math.inf
      )
      angle_sensitivity = self.arrival_sensitivity.angle_per_torque
      rate_sensitivity = self.arrival_sensitivity.rate_per_torque
      if angle_error >= 0.0 and rate_error >= 0.0:
        oriented_high = final_request
        high_angle_error = angle_error
        high_rate_error = rate_error
      else:
        oriented_low = final_request
        low_angle_error = angle_error
        low_rate_error = rate_error
      if angle_sensitivity > 0.0 and math.isfinite(angle_sensitivity):
        angle_root = final_request - angle_error / angle_sensitivity
      elif high_angle_error > low_angle_error:
        angle_root = oriented_low + (
          -low_angle_error
          * (oriented_high - oriented_low)
          / (high_angle_error - low_angle_error)
        )
      else:
        angle_root = 0.5 * (oriented_low + oriented_high)
      if not enforce_rate:
        rate_root = oriented_low
      elif rate_sensitivity > 0.0 and math.isfinite(rate_sensitivity):
        rate_root = final_request - rate_error / rate_sensitivity
      elif high_rate_error > low_rate_error:
        rate_root = oriented_low + (
          -low_rate_error
          * (oriented_high - oriented_low)
          / (high_rate_error - low_rate_error)
        )
      else:
        rate_root = 0.5 * (oriented_low + oriented_high)
      final_request = max(angle_root, rate_root)
      if not math.isfinite(final_request):
        final_request = 0.5 * (oriented_low + oriented_high)
      final_request = min(max(
        final_request,
        oriented_low,
      ), oriented_high)
    return motion_direction * final_request

  def _action_point_feedforward(
    self,
    action_time: float,
    physical_effect_time: float,
    desired_start_angle: float,
    desired_start_rate: float,
    desired_action_angle: float,
    desired_action_rate: float,
    action_speed: float,
    local_feedforward: float,
    align_inputs: AlignInputs,
    disturbance_torque: float,
  ) -> float:
    """Solve planned torque that carries the desired rack to the scalar.

    Feedforward starts on the model-authored rack trajectory, not on the
    measured rack. This keeps requested-path torque independent of tracking
    error; the one physical inverse-residual term is added exactly once by
    ``compute``. The actuator state remains the real queued state, so the solve
    still owns the torque needed to overcome current slew lag.
    """
    duration = float(action_time) - float(physical_effect_time)
    if duration <= 0.0:
      return float(local_feedforward)
    source = self.arrival_source_state
    source.angle_deg = float(desired_start_angle)
    source.rate_deg_s = float(desired_start_rate)
    source.applied_torque = self.predicted_state.applied_torque
    source.v_ego = float(action_speed)
    # This is a synthetic point on the authored trajectory, not a measured
    # held rack. Static load cannot be imported from the current physical
    # position into it.
    source.held_static_load = self.twin.params.t_breakaway
    frame_count = int(math.ceil(duration / DT_CTRL))
    self.arrival_low_request = self._constant_request_endpoint(
      source.applied_torque,
      -1.0,
      frame_count,
    )
    self.arrival_high_request = self._constant_request_endpoint(
      source.applied_torque,
      1.0,
      frame_count,
    )
    self.twin.prepare_align_runtime_terms(
      source.v_ego,
      align_inputs,
      self.arrival_align_terms,
    )
    motion = float(desired_action_angle) - float(desired_start_angle)
    if motion == 0.0:
      motion = float(desired_action_rate) - float(desired_start_rate)
    enforce_rate = motion != 0.0
    return self._solve_terminal_state_request(
      source,
      duration,
      desired_action_angle,
      desired_action_rate,
      local_feedforward,
      disturbance_torque,
      motion,
      enforce_rate,
    )

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
    friction = self.twin.inverse_friction_torque(
      rack_state, required_acceleration,
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
    state.held_static_load = self.predicted_state.held_static_load
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
      desired_action_rate = self._interpolate(
        self.workspace.desired_rates, action_position, count,
      )
      action_point_speed = self._interpolate(
        self.workspace.reference_speeds, action_position, count,
      )
      # Compensate only the physical control-to-rack delay. Commands emitted
      # during that interval are already queued, so replay their actual
      # trajectory rather than pretending the newest torque was present for
      # the whole delay. This is state estimation, not path lead: the position
      # target remains the scalar at its authored action timestamp.
      self._remember_applied_torque(state.applied_torque)
      self._predict_to_effect_time(
        state,
        physical_prediction_delay,
        align_inputs,
        float(disturbance_torque),
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
      action_feedforward = self._action_point_feedforward(
        action_sample_time,
        tracking_sample_time,
        desired_angle,
        desired_rate,
        desired_action_angle,
        desired_action_rate,
        action_point_speed,
        self.local_terms.feedforward,
        align_inputs,
        disturbance_torque,
      )
      action_target = min(max(
        action_feedforward + self.local_terms.feedback,
        -1.0,
      ), 1.0)
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
      result.held_static_load = state.held_static_load
      result.rack_stationary = bool(
        state.rate_deg_s == 0.0
        and state.held_static_load > self.twin.params.t_breakaway
      )
      result.status = CandidateStatus.OK
      result.candidate_count = 1
      result.available_schedule_count = 1
      result.optimality_residual = 0.0
      return result
    except (ValueError, OverflowError):
      self.command_history_start = 0
      self.command_history_count = 0
      result.invalidate(state.applied_torque, CandidateStatus.INPUT_INVALID)
      return result


# Compatibility name for route-audit tooling that loads the shared numerical
# artifact by its tournament-era symbol. Both names are the same class/bytes.
InverseEpsLQIFallback = InverseEpsActionController
