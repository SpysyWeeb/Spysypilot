"""Model-Following Torque MPC shadow candidate.

The decision variable is requested normalized Hyundai steering torque on a
50 ms grid. Each decision is zero-order held and expanded to the 100 Hz plant
grid, where :class:`PlantTwin` applies the exact asymmetric runtime limiter,
including decay-before-build sign crossings.

The convex subproblem for each deterministic torque-sign schedule is a
tridiagonal quadratic solved by scalar IEEE-754 binary64 LDLT. No BLAS,
platform-selected factorization, or unordered collection participates.
Equal-cost schedules retain the lowest index by updating the winner only for a
strictly smaller cost.

Friction is conservative by construction while ``t_breakaway`` remains
provisional: the rollout always uses the full ``±t_breakaway`` Coulomb model
opposing predicted motion and the inverse feedforward never credits friction as
helpful. The independent recorded-response disturbance estimate is separately
bounded to ``±t_breakaway``. The resulting deliberate design tolerance is two
breakaway-equivalent loads; event-anchored identification may tighten it later,
but argument or feel may not.
"""

from __future__ import annotations

import math

import numpy as np

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.blatv2.candidate_common import (
  MAX_CONTROL_STEPS,
  MAX_DECISION_STEPS,
  MAX_SIGN_SCHEDULES,
  CandidateWorkspace,
)
from openpilot.selfdrive.controls.lib.blatv2.controller import (
  DECISION_DT,
  CandidateResult,
  CandidateStatus,
  ControllerParams,
)
from openpilot.selfdrive.controls.lib.blatv2.plant import AlignInputs, PlantState, PlantTwin


class ModelFollowingTorqueMPC:
  def __init__(self, twin: PlantTwin, controller_params: ControllerParams):
    self.twin = twin
    self.params = controller_params
    self.workspace = CandidateWorkspace()
    self.result = CandidateResult()

    self.schedules = np.zeros((MAX_SIGN_SCHEDULES, MAX_DECISION_STEPS), dtype=np.int8)
    self.diagonal = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.off_diagonal = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.rhs = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.work_rhs = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.ldlt_diagonal = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.ldlt_lower = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.ldlt_y = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.solution = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.best_solution = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.fixed = np.zeros(MAX_DECISION_STEPS, dtype=np.bool_)
    self.fixed_value = np.zeros(MAX_DECISION_STEPS, dtype=np.float64)

    self.requested_100hz = np.empty(MAX_CONTROL_STEPS, dtype=np.float64)
    self.applied_100hz = np.empty(MAX_CONTROL_STEPS, dtype=np.float64)
    self.angle_100hz = np.empty(MAX_CONTROL_STEPS, dtype=np.float64)
    self.rate_100hz = np.empty(MAX_CONTROL_STEPS, dtype=np.float64)
    self._state = PlantState(0.0, 0.0, 0.0, 0.0)
    self._align_inputs = AlignInputs(
      0.0, 0.0, 1.0, twin.align_params.nominal_steer_ratio, False,
    )
    self._solve_residual = 0.0
    self.winning_schedule_index = -1

  @staticmethod
  def _sign(value: float) -> int:
    return 1 if value > 0.0 else (-1 if value < 0.0 else 0)

  def _build_schedules(self, count: int) -> int:
    base = self.schedules[0]
    previous_nonzero = 0
    crossing_count = 0
    crossing_indices = self.ldlt_diagonal  # borrowed integer-exact float workspace
    for index in range(count):
      sign = self._sign(float(self.workspace.feedforward[index]))
      base[index] = sign
      if sign != 0:
        if previous_nonzero != 0 and sign != previous_nonzero:
          if crossing_count < MAX_DECISION_STEPS:
            crossing_indices[crossing_count] = index
          crossing_count += 1
        previous_nonzero = sign

    required = 1 + 2 * crossing_count
    if required > MAX_SIGN_SCHEDULES:
      return -required

    schedule_count = 1
    for crossing_number in range(crossing_count):
      crossing = int(crossing_indices[crossing_number])
      old_sign = int(base[crossing - 1])
      new_sign = int(base[crossing])

      earlier = self.schedules[schedule_count]
      later = self.schedules[schedule_count + 1]
      for index in range(count):
        value = base[index]
        earlier[index] = value
        later[index] = value
      earlier[crossing - 1] = new_sign
      later[crossing] = old_sign
      schedule_count += 2
    return schedule_count

  def _assemble_quadratic(self, count: int, previous_torque: float) -> None:
    rate_weight = 1.0 / (self.params.sigma_torque_rate * DECISION_DT) ** 2
    speed = abs(self._state.v_ego)
    for index in range(count):
      remaining = (count - index) * DECISION_DT
      heading_scale = speed * remaining / self.params.sigma_heading
      lateral_scale = speed * remaining * remaining * 0.5 / self.params.sigma_y
      path_weight = 1.0 + heading_scale * heading_scale + lateral_scale * lateral_scale
      diagonal = path_weight
      if index > 0:
        diagonal += rate_weight
      if index + 1 < count:
        diagonal += rate_weight
      self.diagonal[index] = diagonal
      self.off_diagonal[index] = -rate_weight if index + 1 < count else 0.0
      self.rhs[index] = path_weight * float(self.workspace.feedforward[index])

    self.diagonal[0] += rate_weight
    self.rhs[0] += rate_weight * float(previous_torque)

  def _solve_free_segments(self, count: int) -> bool:
    index = 0
    while index < count:
      if self.fixed[index]:
        self.solution[index] = self.fixed_value[index]
        index += 1
        continue
      start = index
      while index + 1 < count and not self.fixed[index + 1]:
        index += 1
      end = index

      for row in range(start, end + 1):
        rhs = self.rhs[row]
        if row == start and row > 0 and self.fixed[row - 1]:
          rhs -= self.off_diagonal[row - 1] * self.fixed_value[row - 1]
        if row == end and row + 1 < count and self.fixed[row + 1]:
          rhs -= self.off_diagonal[row] * self.fixed_value[row + 1]
        self.work_rhs[row] = rhs

      diagonal = self.diagonal[start]
      if not math.isfinite(diagonal) or diagonal <= 0.0:
        return False
      self.ldlt_diagonal[start] = diagonal
      self.ldlt_y[start] = self.work_rhs[start]
      for row in range(start + 1, end + 1):
        lower = self.off_diagonal[row - 1] / self.ldlt_diagonal[row - 1]
        diagonal = self.diagonal[row] - lower * self.off_diagonal[row - 1]
        if not math.isfinite(diagonal) or diagonal <= 0.0:
          return False
        self.ldlt_lower[row] = lower
        self.ldlt_diagonal[row] = diagonal
        self.ldlt_y[row] = self.work_rhs[row] - lower * self.ldlt_y[row - 1]

      self.solution[end] = self.ldlt_y[end] / self.ldlt_diagonal[end]
      for row in range(end - 1, start - 1, -1):
        self.solution[row] = self.ldlt_y[row] / self.ldlt_diagonal[row] - self.ldlt_lower[row + 1] * self.solution[row + 1]
      index += 1
    return True

  def _solve_schedule(self, schedule_index: int, count: int) -> CandidateStatus:
    self._solve_residual = 0.0
    signs = self.schedules[schedule_index]
    for index in range(count):
      self.fixed[index] = signs[index] == 0
      self.fixed_value[index] = 0.0

    scale = 0.0
    for index in range(count):
      scale = max(scale, abs(self.rhs[index]))
    tolerance = count * math.ulp(1.0 + scale)

    # At most one constraint changes per iteration. Lowest-index selection for
    # equal violations is part of the deterministic active-set contract.
    for _ in range(2 * count + 2):
      if not self._solve_free_segments(count):
        return CandidateStatus.NON_CONVERGED

      violation_index = -1
      violation_value = 0.0
      for index in range(count):
        if self.fixed[index]:
          continue
        value = float(self.solution[index])
        sign = int(signs[index])
        if not math.isfinite(value):
          return CandidateStatus.NON_CONVERGED
        if value > 1.0:
          violation_index = index
          violation_value = 1.0
        elif value < -1.0:
          violation_index = index
          violation_value = -1.0
        elif sign > 0 and value < 0.0:
          violation_index = index
          violation_value = 0.0
        elif sign < 0 and value > 0.0:
          violation_index = index
          violation_value = 0.0
        if violation_index >= 0:
          break

      if violation_index >= 0:
        self.fixed[violation_index] = True
        self.fixed_value[violation_index] = violation_value
        continue

      residual = 0.0
      release_index = -1
      release_violation = tolerance
      for index in range(count):
        gradient = self.diagonal[index] * self.solution[index] - self.rhs[index]
        if index > 0:
          gradient += self.off_diagonal[index - 1] * self.solution[index - 1]
        if index + 1 < count:
          gradient += self.off_diagonal[index] * self.solution[index + 1]
        if not self.fixed[index]:
          violation = abs(gradient)
        elif signs[index] == 0:
          # A zero-valued sign schedule is an equality, not a releasable bound.
          violation = 0.0
        elif self.fixed_value[index] == 1.0:
          violation = max(gradient, 0.0)
        elif self.fixed_value[index] == -1.0:
          violation = max(-gradient, 0.0)
        elif signs[index] > 0:
          violation = max(-gradient, 0.0)
        else:
          violation = max(gradient, 0.0)
        residual = max(residual, violation)
        if self.fixed[index] and signs[index] != 0 and violation > release_violation:
          release_violation = violation
          release_index = index

      if release_index >= 0:
        self.fixed[release_index] = False
        self.fixed_value[release_index] = 0.0
        continue

      self._solve_residual = residual
      return CandidateStatus.OK
    return CandidateStatus.NON_CONVERGED

  def _rollout_cost(self, disturbance_torque: float, actuation_delay: float) -> float:
    self.workspace.expand_decisions_zoh(self.solution, self.requested_100hz)
    control_count = self.workspace.control_count
    self.twin.predict_into(
      self._state,
      self.requested_100hz,
      control_count,
      DT_CTRL,
      self._align_inputs,
      self.applied_100hz,
      self.angle_100hz,
      self.rate_100hz,
      disturbance_torque,
      actuation_delay,
    )

    heading_error = 0.0
    lateral_error = 0.0
    previous_applied = self._state.applied_torque
    cost = 0.0
    for index in range(control_count):
      predicted_curvature = self.twin.curvature_from_angle(
        float(self.angle_100hz[index]),
        self._state.v_ego,
        self._align_inputs,
      )
      curvature_error = predicted_curvature - float(self.workspace.control_reference[index])
      heading_error += self._state.v_ego * curvature_error * DT_CTRL
      lateral_error += self._state.v_ego * heading_error * DT_CTRL
      torque_rate = (float(self.applied_100hz[index]) - previous_applied) / DT_CTRL
      previous_applied = float(self.applied_100hz[index])
      cost += (
        (lateral_error / self.params.sigma_y) ** 2
        + (heading_error / self.params.sigma_heading) ** 2
        + (torque_rate / self.params.sigma_torque_rate) ** 2
      )
    return cost

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
  ) -> CandidateResult:
    result = self.result
    self._state = state
    self._align_inputs = align_inputs
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
      count = self.workspace.decision_count
      schedule_count = self._build_schedules(count)
      if schedule_count < 0:
        result.invalidate(state.applied_torque, CandidateStatus.ENUMERATION_EXHAUSTED)
        result.candidate_count = -schedule_count
        return result
      self._assemble_quadratic(count, state.applied_torque)
    except (ValueError, OverflowError):
      result.invalidate(state.applied_torque, CandidateStatus.INPUT_INVALID)
      return result

    best_cost = math.inf
    best_residual = 0.0
    self.winning_schedule_index = -1
    solved_count = 0
    non_converged_count = 0
    for schedule_index in range(schedule_count):
      status = self._solve_schedule(schedule_index, count)
      solved_count += 1
      if status != CandidateStatus.OK:
        non_converged_count += 1
        continue
      try:
        cost = self._rollout_cost(disturbance_torque, actuation_delay)
      except (ValueError, OverflowError):
        non_converged_count += 1
        continue
      # Strict comparison pins equal-cost tie-breaking to the lower schedule.
      if math.isfinite(cost) and cost < best_cost:
        best_cost = cost
        best_residual = self._solve_residual
        self.winning_schedule_index = schedule_index
        for index in range(count):
          self.best_solution[index] = self.solution[index]

    if not math.isfinite(best_cost):
      failure = CandidateStatus.NON_CONVERGED if non_converged_count == solved_count else CandidateStatus.INFEASIBLE
      result.invalidate(state.applied_torque, failure)
      result.candidate_count = solved_count
      return result

    result.command_torque = self.twin.apply_slew(state.applied_torque, float(self.best_solution[0]))
    result.status = CandidateStatus.OK
    result.candidate_count = solved_count
    result.optimality_residual = best_residual
    return result
