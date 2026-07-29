"""Shared deterministic numerical core for BLaTv2 shadow telemetry.

The on-device daemon and the route-audit harness import this file directly.
Environment measurements, including wall-clock compute time and messaging
transport health, deliberately live outside this core.

The hot path owns fixed numpy buffers, two reusable plant states/alignment
input sets, one PlantTwin, and one mutable result. ``compute`` therefore
returns a borrowed result object whose fields are overwritten on the next
call; callers that retain history must copy the scalar fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from openpilot.selfdrive.controls.lib.blatv2.controller import (
  CandidateStatus,
  ControllerParams,
  ObserverStatus,
)
from openpilot.selfdrive.controls.lib.blatv2.candidate_common import CandidateWorkspace
from openpilot.selfdrive.controls.lib.blatv2.fallback import InverseEpsLQIFallback
from openpilot.selfdrive.controls.lib.blatv2.mpc import ModelFollowingTorqueMPC
from openpilot.selfdrive.controls.lib.blatv2.observer import DisturbanceObserver
from openpilot.selfdrive.controls.lib.blatv2.plant import (
  AlignInputs,
  AlignParams,
  PlantParams,
  PlantState,
  PlantTwin,
)
from openpilot.selfdrive.controls.lib.blatv2.reference import (
  build_reference_into,
  horizon,
  interpolate_buffer,
  model_action_time,
  plan_curvatures_from_model_into,
  torque_demand,
)
from openpilot.selfdrive.modeld.constants import ModelConstants


@dataclass(slots=True)
class ShadowResult:
  valid: bool = False
  reference_curvature: float = 0.0
  torque_demand: float = 0.0
  feasible_torque: float = 0.0
  plant_residual: float = 0.0
  scalar_plan_disagreement: float = 0.0
  horizon: float = 0.0
  v_ego: float = 0.0
  aligning_torque: float = 0.0
  align_inputs_valid: bool = False
  disturbance_estimate: float = 0.0
  observer_status: int = int(ObserverStatus.RESET_LATERAL_INVALID)
  observer_unconstrained_update: float = 0.0
  mpc_command_torque: float = 0.0
  mpc_status: int = int(CandidateStatus.INPUT_INVALID)
  mpc_candidate_count: int = 0
  mpc_available_schedule_count: int = 0
  mpc_optimality_residual: float = 0.0
  fallback_command_torque: float = 0.0
  fallback_status: int = int(CandidateStatus.INPUT_INVALID)
  fallback_candidate_count: int = 0
  fallback_optimality_residual: float = 0.0


class ShadowCore:
  """Preallocated shadow core; state exists only for one-step residuals."""

  def __init__(
    self,
    seed_params: PlantParams,
    torque_params: Any,
    car_params: Any,
    controller_params: ControllerParams,
  ):
    self.seed_params = seed_params
    self.torque_params = torque_params
    self.align_params = AlignParams.from_car_params(car_params, torque_params)
    self.twin = PlantTwin(seed_params, self.align_params)
    self.controller_params = controller_params
    self.observer = DisturbanceObserver(seed_params, controller_params)
    self.candidate_workspace = CandidateWorkspace()
    self.mpc = ModelFollowingTorqueMPC(
      self.twin, controller_params, self.candidate_workspace,
    )
    self.fallback = InverseEpsLQIFallback(
      self.twin, controller_params, self.candidate_workspace,
    )
    self.default_horizon = horizon(seed_params)

    capacity = len(ModelConstants.T_IDXS)
    self.plan_times = np.empty(capacity, dtype=np.float64)
    self.plan_curvatures = np.empty(capacity, dtype=np.float64)
    self.reference_times = np.empty(capacity, dtype=np.float64)
    self.reference_curvatures = np.empty(capacity, dtype=np.float64)
    self.reference_speeds = np.empty(capacity, dtype=np.float64)

    self.previous_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.current_state = PlantState(0.0, 0.0, 0.0, 0.0)
    self.previous_align_inputs = AlignInputs(
      0.0, 0.0, 1.0, self.align_params.nominal_steer_ratio, False,
    )
    self.current_align_inputs = AlignInputs(
      0.0, 0.0, 1.0, self.align_params.nominal_steer_ratio, False,
    )
    self.has_previous_state = False
    self.result = ShadowResult(horizon=self.default_horizon)
    self.reference_count = 0
    self.horizon_seconds = self.default_horizon
    self.actuation_delay = self.seed_params.actuation_delay
    self.action_time = model_action_time(self.actuation_delay)
    self.frame_prepared = False
    self.candidate_workspace_valid = False

  def reset(self) -> None:
    self.has_previous_state = False
    self.frame_prepared = False
    self.candidate_workspace_valid = False
    self.observer.reset()
    self.mpc.reset()
    self.fallback.reset()

  def invalid_result(self) -> ShadowResult:
    self.reset()
    result = self.result
    result.valid = False
    result.reference_curvature = 0.0
    result.torque_demand = 0.0
    result.feasible_torque = 0.0
    result.plant_residual = 0.0
    result.scalar_plan_disagreement = 0.0
    result.horizon = self.default_horizon
    result.v_ego = 0.0
    result.aligning_torque = 0.0
    result.align_inputs_valid = False
    result.disturbance_estimate = 0.0
    result.observer_status = int(self.observer.status)
    result.observer_unconstrained_update = 0.0
    result.mpc_command_torque = 0.0
    result.mpc_status = int(CandidateStatus.INPUT_INVALID)
    result.mpc_candidate_count = 0
    result.mpc_available_schedule_count = 0
    result.mpc_optimality_residual = 0.0
    result.fallback_command_torque = 0.0
    result.fallback_status = int(CandidateStatus.INPUT_INVALID)
    result.fallback_candidate_count = 0
    result.fallback_optimality_residual = 0.0
    return result

  def frame_actuation_delay(self, lateral_delay: float, lateral_delay_valid: bool) -> float:
    delay = float(lateral_delay)
    if lateral_delay_valid and math.isfinite(delay) and delay >= 0.0:
      return delay
    return self.seed_params.actuation_delay

  def _set_align_inputs(self, target: AlignInputs, live_parameters: Any, live_parameters_valid: bool) -> None:
    if live_parameters_valid:
      roll = float(live_parameters.roll)
      angle_offset = float(live_parameters.angleOffsetDeg)
      stiffness_factor = float(live_parameters.stiffnessFactor)
      steer_ratio = float(live_parameters.steerRatio)
      valid = (
        math.isfinite(roll)
        and math.isfinite(angle_offset)
        and math.isfinite(stiffness_factor)
        and math.isfinite(steer_ratio)
        and stiffness_factor > 0.0
        and steer_ratio > 0.0
      )
      if valid:
        target.roll = roll
        target.angle_offset_deg = angle_offset
        target.stiffness_factor = stiffness_factor
        target.steer_ratio = steer_ratio
        target.valid = True
        return

    # Never carry live alignment inputs across an invalid frame.
    target.roll = 0.0
    target.angle_offset_deg = 0.0
    target.stiffness_factor = 1.0
    target.steer_ratio = self.align_params.nominal_steer_ratio
    target.valid = False

  def _build_reference_speeds(
    self,
    model: Any,
    plan_count: int,
    reference_count: int,
    horizon_seconds: float,
    v_ego: float,
  ) -> None:
    """Preserve the model's speed changes on the scalar-pinned time grid.

    The model plan is camera-time based. Its first predicted speed is therefore
    shifted to the measured current speed, matching the established v14
    convention, while every future delta remains model-authored.
    """
    model_speeds = model.velocity.x
    if len(model_speeds) < plan_count:
      raise ValueError("model speed plan is shorter than its curvature plan")
    first_speed = float(model_speeds[0])
    speed_offset = float(v_ego) - first_speed
    output_index = 0
    for index in range(plan_count):
      time_value = float(self.plan_times[index])
      if 0.0 <= time_value <= horizon_seconds:
        if output_index >= reference_count:
          raise ValueError("reference speed count exceeds reference curvature count")
        speed_value = float(model_speeds[index]) + speed_offset
        if not math.isfinite(speed_value):
          raise ValueError("model speed plan must be finite")
        self.reference_speeds[output_index] = max(speed_value, 0.0)
        output_index += 1
    if output_index != reference_count:
      raise ValueError("reference speed and curvature grids do not align")

  def begin_frame(
    self,
    model: Any,
    car_state: Any,
    car_control: Any,
    car_output: Any,
    live_parameters: Any,
    live_parameters_valid: bool,
    lateral_delay: float,
    lateral_delay_valid: bool,
    model_valid: bool,
  ) -> ShadowResult:
    if self.frame_prepared:
      raise RuntimeError("previous BLaTv2 shadow frame was not finalized")
    actuation_delay = self.frame_actuation_delay(lateral_delay, lateral_delay_valid)

    scalar = float(model.action.desiredCurvature)
    v_ego = float(car_state.vEgo)
    signed_plan_count = plan_curvatures_from_model_into(
      model,
      scalar,
      self.plan_times,
      self.plan_curvatures,
    )
    plan_valid = signed_plan_count > 0
    plan_count = abs(signed_plan_count)
    action_time = model_action_time(actuation_delay)
    horizon_seconds = horizon(self.seed_params, actuation_delay)
    reference_count = build_reference_into(
      scalar,
      self.plan_times,
      self.plan_curvatures,
      plan_count,
      action_time,
      horizon_seconds,
      self.reference_times,
      self.reference_curvatures,
    )
    self._build_reference_speeds(
      model,
      plan_count,
      reference_count,
      horizon_seconds,
      v_ego,
    )
    reference_now_delay = interpolate_buffer(
      self.reference_times,
      self.reference_curvatures,
      reference_count,
      action_time,
    )
    plan_at_action = interpolate_buffer(
      self.plan_times,
      self.plan_curvatures,
      plan_count,
      action_time,
    )
    disagreement = plan_at_action - scalar

    self._set_align_inputs(self.current_align_inputs, live_parameters, live_parameters_valid)
    demand = torque_demand(
      reference_now_delay,
      v_ego,
      self.current_align_inputs.roll,
      self.torque_params,
    )
    applied = float(car_output.actuatorsOutput.torque)
    feasible = self.twin.apply_slew(applied, demand)

    state = self.current_state
    state.angle_deg = float(car_state.steeringAngleDeg)
    state.rate_deg_s = float(car_state.steeringRateDeg)
    state.applied_torque = applied
    state.v_ego = v_ego
    if not (
      math.isfinite(state.angle_deg)
      and math.isfinite(state.rate_deg_s)
      and math.isfinite(state.applied_torque)
      and math.isfinite(state.v_ego)
    ):
      raise ValueError("plant state must be finite")

    residual_valid = self.has_previous_state
    residual = 0.0
    if residual_valid:
      residual_aligning_torque = self.twin.aligning_torque(
        self.previous_state,
        self.previous_align_inputs,
      )
      residual = self.twin.one_step_residual(
        self.previous_state,
        self.previous_state.applied_torque,
        state,
        self.previous_align_inputs,
      )
      residual_v_ego = self.previous_state.v_ego
      residual_align_inputs_valid = self.previous_align_inputs.valid
    else:
      residual_aligning_torque = self.twin.aligning_torque(state, self.current_align_inputs)
      residual_v_ego = state.v_ego
      residual_align_inputs_valid = self.current_align_inputs.valid

    result = self.result
    result.valid = bool(
      model_valid
      and plan_valid
      and residual_valid
      and math.isfinite(reference_now_delay)
      and math.isfinite(demand)
      and math.isfinite(feasible)
      and math.isfinite(residual)
      and math.isfinite(disagreement)
      and math.isfinite(horizon_seconds)
      and math.isfinite(residual_aligning_torque)
    )
    result.reference_curvature = float(reference_now_delay)
    result.torque_demand = float(demand)
    result.feasible_torque = float(feasible)
    result.plant_residual = float(residual)
    result.scalar_plan_disagreement = float(disagreement)
    result.horizon = float(horizon_seconds)
    # These fields describe state_t and its inputs for the reported
    # state_t -> state_t1 residual. The bootstrap frame reports current state.
    result.v_ego = float(residual_v_ego)
    result.aligning_torque = float(residual_aligning_torque)
    result.align_inputs_valid = residual_align_inputs_valid
    requested = float(car_control.actuators.torque)
    recorded_constraint_active = not math.isfinite(requested) or requested != applied
    disturbance = self.observer.update(
      residual,
      residual_valid,
      bool(car_control.latActive),
      bool(car_state.steeringPressed),
      bool(car_state.standstill),
      bool(model_valid and plan_valid),
      recorded_constraint_active,
    )
    result.disturbance_estimate = disturbance
    result.observer_status = int(self.observer.status)
    result.observer_unconstrained_update = self.observer.unconstrained_update
    result.mpc_command_torque = applied
    result.mpc_status = int(CandidateStatus.INPUT_INVALID)
    result.mpc_candidate_count = 0
    result.mpc_available_schedule_count = 0
    result.mpc_optimality_residual = 0.0
    result.fallback_command_torque = applied
    result.fallback_status = int(CandidateStatus.INPUT_INVALID)
    result.fallback_candidate_count = 0
    result.fallback_optimality_residual = 0.0

    self.reference_count = reference_count
    self.horizon_seconds = horizon_seconds
    self.actuation_delay = actuation_delay
    self.action_time = action_time
    self.candidate_workspace_valid = False
    if result.valid:
      try:
        self.candidate_workspace.fill(
          self.twin,
          self.current_state,
          self.current_align_inputs,
          self.reference_times,
          self.reference_curvatures,
          self.reference_count,
          self.horizon_seconds,
          result.disturbance_estimate,
          self.controller_params.kinetic_friction,
          self.reference_speeds,
        )
        self.candidate_workspace_valid = True
      except (ValueError, OverflowError):
        pass
    self.frame_prepared = True
    return result

  def compute_mpc(self) -> ShadowResult:
    if not self.frame_prepared:
      raise RuntimeError("BLaTv2 shadow frame has not been prepared")
    result = self.result
    if result.valid and self.candidate_workspace_valid:
      if self.observer.status.reset:
        self.mpc.reset()
      candidate = self.mpc.compute(
        self.current_state,
        self.current_align_inputs,
        self.reference_times,
        self.reference_curvatures,
        self.reference_count,
        self.horizon_seconds,
        self.actuation_delay,
        result.disturbance_estimate,
        workspace_prepared=True,
      )
      result.mpc_command_torque = candidate.command_torque
      result.mpc_status = int(candidate.status)
      result.mpc_candidate_count = candidate.candidate_count
      result.mpc_available_schedule_count = candidate.available_schedule_count
      result.mpc_optimality_residual = candidate.optimality_residual
    else:
      self.mpc.reset()
      self.mpc.result.invalidate(self.current_state.applied_torque)
    return result

  def compute_fallback(self) -> ShadowResult:
    if not self.frame_prepared:
      raise RuntimeError("BLaTv2 shadow frame has not been prepared")
    result = self.result
    if result.valid and self.candidate_workspace_valid:
      candidate = self.fallback.compute(
        self.current_state,
        self.current_align_inputs,
        self.reference_times,
        self.reference_curvatures,
        self.reference_count,
        self.horizon_seconds,
        self.actuation_delay,
        result.disturbance_estimate,
        self.observer.status,
        workspace_prepared=True,
        action_time=self.action_time,
      )
      result.fallback_command_torque = candidate.command_torque
      result.fallback_status = int(candidate.status)
      result.fallback_candidate_count = candidate.candidate_count
      result.fallback_optimality_residual = candidate.optimality_residual
    else:
      self.fallback.reset()
    return result

  def end_frame(self) -> ShadowResult:
    if not self.frame_prepared:
      raise RuntimeError("BLaTv2 shadow frame has not been prepared")
    self.previous_state, self.current_state = self.current_state, self.previous_state
    self.previous_align_inputs, self.current_align_inputs = self.current_align_inputs, self.previous_align_inputs
    self.has_previous_state = True
    self.frame_prepared = False
    return self.result

  def compute(
    self,
    model: Any,
    car_state: Any,
    car_control: Any,
    car_output: Any,
    live_parameters: Any,
    live_parameters_valid: bool,
    lateral_delay: float,
    lateral_delay_valid: bool,
    model_valid: bool = True,
  ) -> ShadowResult:
    self.begin_frame(
      model,
      car_state,
      car_control,
      car_output,
      live_parameters,
      live_parameters_valid,
      lateral_delay,
      lateral_delay_valid,
      model_valid,
    )
    self.compute_mpc()
    self.compute_fallback()
    return self.end_frame()
