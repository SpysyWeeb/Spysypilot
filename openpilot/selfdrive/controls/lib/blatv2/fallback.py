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
    self.b = np.zeros(self.STATE_DIM, dtype=np.float64)
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

    a00 = float(self.a[0, 0])
    a01 = float(self.a[0, 1])
    a02 = float(self.a[0, 2])
    a10 = float(self.a[1, 0])
    a11 = float(self.a[1, 1])
    a12 = float(self.a[1, 2])
    a20 = float(self.a[2, 0])
    a21 = float(self.a[2, 1])
    a22 = float(self.a[2, 2])
    b0 = float(self.b[0])
    b1 = float(self.b[1])
    b2 = float(self.b[2])
    q0 = float(self.q[0])
    q1 = float(self.q[1])
    q2 = float(self.q[2])
    p00, p01, p02 = q0, 0.0, 0.0
    p10, p11, p12 = 0.0, q1, 0.0
    p20, p21, p22 = 0.0, 0.0, q2

    for _ in range(horizon_count):
      pb0 = 0.0
      pb0 += p00 * b0
      pb0 += p01 * b1
      pb0 += p02 * b2
      pb1 = 0.0
      pb1 += p10 * b0
      pb1 += p11 * b1
      pb1 += p12 * b2
      pb2 = 0.0
      pb2 += p20 * b0
      pb2 += p21 * b1
      pb2 += p22 * b2

      denominator = control_weight
      denominator += b0 * pb0
      denominator += b1 * pb1
      denominator += b2 * pb2
      if not math.isfinite(denominator) or denominator <= 0.0:
        return math.inf

      bpa0 = 0.0
      bpa0 += pb0 * a00
      bpa0 += pb1 * a10
      bpa0 += pb2 * a20
      bpa1 = 0.0
      bpa1 += pb0 * a01
      bpa1 += pb1 * a11
      bpa1 += pb2 * a21
      bpa2 = 0.0
      bpa2 += pb0 * a02
      bpa2 += pb1 * a12
      bpa2 += pb2 * a22
      gain0 = bpa0 / denominator
      gain1 = bpa1 / denominator
      gain2 = bpa2 / denominator

      ap00 = 0.0
      ap00 += p00 * a00
      ap00 += p01 * a10
      ap00 += p02 * a20
      ap01 = 0.0
      ap01 += p00 * a01
      ap01 += p01 * a11
      ap01 += p02 * a21
      ap02 = 0.0
      ap02 += p00 * a02
      ap02 += p01 * a12
      ap02 += p02 * a22
      ap10 = 0.0
      ap10 += p10 * a00
      ap10 += p11 * a10
      ap10 += p12 * a20
      ap11 = 0.0
      ap11 += p10 * a01
      ap11 += p11 * a11
      ap11 += p12 * a21
      ap12 = 0.0
      ap12 += p10 * a02
      ap12 += p11 * a12
      ap12 += p12 * a22
      ap20 = 0.0
      ap20 += p20 * a00
      ap20 += p21 * a10
      ap20 += p22 * a20
      ap21 = 0.0
      ap21 += p20 * a01
      ap21 += p21 * a11
      ap21 += p22 * a21
      ap22 = 0.0
      ap22 += p20 * a02
      ap22 += p21 * a12
      ap22 += p22 * a22

      next_p00 = q0
      next_p00 += a00 * ap00
      next_p00 += a10 * ap10
      next_p00 += a20 * ap20
      next_p00 -= bpa0 * bpa0 / denominator
      next_p01 = 0.0
      next_p01 += a00 * ap01
      next_p01 += a10 * ap11
      next_p01 += a20 * ap21
      next_p01 -= bpa0 * bpa1 / denominator
      next_p02 = 0.0
      next_p02 += a00 * ap02
      next_p02 += a10 * ap12
      next_p02 += a20 * ap22
      next_p02 -= bpa0 * bpa2 / denominator
      next_p10 = 0.0
      next_p10 += a01 * ap00
      next_p10 += a11 * ap10
      next_p10 += a21 * ap20
      next_p10 -= bpa1 * bpa0 / denominator
      next_p11 = q1
      next_p11 += a01 * ap01
      next_p11 += a11 * ap11
      next_p11 += a21 * ap21
      next_p11 -= bpa1 * bpa1 / denominator
      next_p12 = 0.0
      next_p12 += a01 * ap02
      next_p12 += a11 * ap12
      next_p12 += a21 * ap22
      next_p12 -= bpa1 * bpa2 / denominator
      next_p20 = 0.0
      next_p20 += a02 * ap00
      next_p20 += a12 * ap10
      next_p20 += a22 * ap20
      next_p20 -= bpa2 * bpa0 / denominator
      next_p21 = 0.0
      next_p21 += a02 * ap01
      next_p21 += a12 * ap11
      next_p21 += a22 * ap21
      next_p21 -= bpa2 * bpa1 / denominator
      next_p22 = q2
      next_p22 += a02 * ap02
      next_p22 += a12 * ap12
      next_p22 += a22 * ap22
      next_p22 -= bpa2 * bpa2 / denominator
      p00, p01, p02 = next_p00, next_p01, next_p02
      p10, p11, p12 = next_p10, next_p11, next_p12
      p20, p21, p22 = next_p20, next_p21, next_p22

    self.gain[0] = gain0
    self.gain[1] = gain1
    self.gain[2] = gain2
    gain_residual = 0.0
    gain_residual = max(gain_residual, abs(gain0 * denominator - bpa0))
    gain_residual = max(gain_residual, abs(gain1 * denominator - bpa1))
    gain_residual = max(gain_residual, abs(gain2 * denominator - bpa2))
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
