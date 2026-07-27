"""Analytic inverse-EPS feedforward plus finite-horizon LQI shadow candidate.

This candidate shares the scalar-pinned reference, plant, conservative full
``t_breakaway`` friction treatment, disturbance estimate, and the three
provisional owner feel-dials with MPC. It adds no independent gains: the LQI
weights are derived from ``SIGMA_Y``, ``SIGMA_HEADING``, and
``SIGMA_TORQUE_RATE``.

The persistent lateral-error integral follows the observer lifecycle. It resets
on lateral/model invalidity, steering press, standstill, and engagement; it
freezes whenever the recorded actuator is constrained. It therefore cannot
wind up during exactly the frames in which observer learning is suspended.
"""

from __future__ import annotations

import math

import numpy as np

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.blatv2.candidate_common import CandidateWorkspace
from openpilot.selfdrive.controls.lib.blatv2.controller import (
  DECISION_DT,
  CandidateResult,
  CandidateStatus,
  ControllerParams,
  ObserverStatus,
)
from openpilot.selfdrive.controls.lib.blatv2.plant import AlignInputs, PlantState, PlantTwin


class InverseEpsLQIFallback:
  STATE_DIM = 3

  def __init__(self, twin: PlantTwin, controller_params: ControllerParams):
    self.twin = twin
    self.params = controller_params
    self.workspace = CandidateWorkspace()
    self.result = CandidateResult()
    self.integral_lateral_error = 0.0

    self.a = np.zeros((self.STATE_DIM, self.STATE_DIM), dtype=np.float64)
    self.p = np.zeros((self.STATE_DIM, self.STATE_DIM), dtype=np.float64)
    self.next_p = np.zeros((self.STATE_DIM, self.STATE_DIM), dtype=np.float64)
    self.ap = np.zeros((self.STATE_DIM, self.STATE_DIM), dtype=np.float64)
    self.b = np.zeros(self.STATE_DIM, dtype=np.float64)
    self.pb = np.zeros(self.STATE_DIM, dtype=np.float64)
    self.bpa = np.zeros(self.STATE_DIM, dtype=np.float64)
    self.gain = np.zeros(self.STATE_DIM, dtype=np.float64)
    self.q = np.zeros(self.STATE_DIM, dtype=np.float64)

  def reset(self) -> None:
    self.integral_lateral_error = 0.0
    self.result.invalidate()

  def _update_integral(
    self,
    observer_status: ObserverStatus,
    curvature_error: float,
    v_ego: float,
  ) -> None:
    if observer_status.reset:
      self.integral_lateral_error = 0.0
    elif not observer_status.frozen:
      update = float(v_ego) * float(v_ego) * float(curvature_error) * DT_CTRL * DT_CTRL
      bound = self.params.sigma_y
      self.integral_lateral_error = min(max(self.integral_lateral_error + update, -bound), bound)

  def _compute_gain(
    self,
    state: PlantState,
    align_inputs: AlignInputs,
    horizon_count: int,
  ) -> float:
    dt = DECISION_DT
    rate_decay = 1.0 - self.twin.params.b_steer * dt
    torque_to_rate = self.twin.params.k_t * dt
    curvature = self.twin.curvature_from_angle(state.angle_deg, state.v_ego, align_inputs)
    curvature_plus_degree = self.twin.curvature_from_angle(state.angle_deg + 1.0, state.v_ego, align_inputs)
    curvature_per_degree = curvature_plus_degree - curvature

    self.a.fill(0.0)
    self.a[0, 0] = 1.0
    self.a[0, 1] = dt * rate_decay
    self.a[1, 1] = rate_decay
    self.a[2, 0] = state.v_ego * state.v_ego * curvature_per_degree * dt * dt
    self.a[2, 2] = 1.0
    self.b[0] = dt * torque_to_rate
    self.b[1] = torque_to_rate
    self.b[2] = 0.0

    heading_angle_scale = state.v_ego * curvature_per_degree * dt / self.params.sigma_heading
    heading_rate_scale = state.v_ego * curvature_per_degree * dt * dt / self.params.sigma_heading
    self.q[0] = heading_angle_scale * heading_angle_scale
    self.q[1] = heading_rate_scale * heading_rate_scale
    self.q[2] = 1.0 / (self.params.sigma_y * self.params.sigma_y)
    control_weight = 1.0 / (self.params.sigma_torque_rate * dt) ** 2

    self.p.fill(0.0)
    for index in range(self.STATE_DIM):
      self.p[index, index] = self.q[index]

    gain_residual = 0.0
    for _ in range(horizon_count):
      for row in range(self.STATE_DIM):
        value = 0.0
        for column in range(self.STATE_DIM):
          value += self.p[row, column] * self.b[column]
        self.pb[row] = value

      denominator = control_weight
      for index in range(self.STATE_DIM):
        denominator += self.b[index] * self.pb[index]
      if not math.isfinite(denominator) or denominator <= 0.0:
        return math.inf

      for column in range(self.STATE_DIM):
        value = 0.0
        for row in range(self.STATE_DIM):
          value += self.pb[row] * self.a[row, column]
        self.bpa[column] = value
        self.gain[column] = value / denominator

      for row in range(self.STATE_DIM):
        for column in range(self.STATE_DIM):
          value = 0.0
          for inner in range(self.STATE_DIM):
            value += self.p[row, inner] * self.a[inner, column]
          self.ap[row, column] = value

      for row in range(self.STATE_DIM):
        for column in range(self.STATE_DIM):
          value = self.q[row] if row == column else 0.0
          for inner in range(self.STATE_DIM):
            value += self.a[inner, row] * self.ap[inner, column]
          value -= self.bpa[row] * self.bpa[column] / denominator
          self.next_p[row, column] = value

      self.p, self.next_p = self.next_p, self.p

    for index in range(self.STATE_DIM):
      gain_residual = max(gain_residual, abs(self.gain[index] * denominator - self.bpa[index]))
    return gain_residual

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
  ) -> CandidateResult:
    result = self.result
    try:
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
      sample_position = min(
        max(float(actuation_delay) / DECISION_DT, 0.0),
        float(self.workspace.decision_count - 1),
      )
      lower_index = int(sample_position)
      upper_index = min(lower_index + 1, self.workspace.decision_count - 1)
      fraction = sample_position - lower_index
      desired_angle = float(self.workspace.desired_angles[lower_index]) + fraction * (
        float(self.workspace.desired_angles[upper_index]) - float(self.workspace.desired_angles[lower_index])
      )
      desired_rate = float(self.workspace.desired_rates[lower_index]) + fraction * (
        float(self.workspace.desired_rates[upper_index]) - float(self.workspace.desired_rates[lower_index])
      )
      reference_curvature = float(self.workspace.reference_curvatures[0])
      measured_curvature = self.twin.curvature_from_angle(state.angle_deg, state.v_ego, align_inputs)
      curvature_error = measured_curvature - reference_curvature
      self._update_integral(observer_status, curvature_error, state.v_ego)

      residual = self._compute_gain(state, align_inputs, self.workspace.decision_count)
      if not math.isfinite(residual):
        result.invalidate(state.applied_torque, CandidateStatus.NON_CONVERGED)
        result.candidate_count = 1
        return result

      angle_error = state.angle_deg - desired_angle
      rate_error = state.rate_deg_s - desired_rate
      correction = -(
        float(self.gain[0]) * angle_error
        + float(self.gain[1]) * rate_error
        + float(self.gain[2]) * self.integral_lateral_error
      )
      feedforward = float(self.workspace.feedforward[lower_index]) + fraction * (
        float(self.workspace.feedforward[upper_index]) - float(self.workspace.feedforward[lower_index])
      )
      raw_command = feedforward + correction
      if not math.isfinite(raw_command):
        result.invalidate(state.applied_torque, CandidateStatus.NON_CONVERGED)
        result.candidate_count = 1
        return result

      result.command_torque = self.twin.apply_slew(state.applied_torque, min(max(raw_command, -1.0), 1.0))
      result.status = CandidateStatus.OK
      result.candidate_count = 1
      result.optimality_residual = residual
      return result
    except (ValueError, OverflowError):
      result.invalidate(state.applied_torque, CandidateStatus.INPUT_INVALID)
      return result
