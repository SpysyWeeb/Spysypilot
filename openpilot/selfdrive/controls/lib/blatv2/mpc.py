"""Model-Following Torque MPC shadow candidate.

The decision variable is requested normalized Hyundai steering torque on a
50 ms grid. The first decision is sampled on every 100 Hz controller frame,
where :class:`PlantTwin` applies the exact asymmetric runtime limiter,
including decay-before-build sign crossings.

The convex subproblem for each deterministic torque-sign schedule is a
tridiagonal quadratic solved by scalar IEEE-754 binary64 LDLT. No BLAS,
platform-selected factorization, or unordered collection participates.

Solving and rolling out every plausible sign-crossing placement made runtime
scale linearly on 6.84% of an ordinary field route (nine schedules cost 59 ms
median). Runtime is therefore hard-bounded to one active-set solve per frame.
The selected schedule reuses the previous frame's winning schedule index when
that early/late crossing ordinal still exists; bootstrap, lifecycle resets, and
an out-of-range prior index use the base schedule. This O(1) warm start
preserves exact limiter semantics for the selected schedule without making wall
time data-dependent.
Telemetry reports both the one evaluated candidate and the number of schedules
that were available before the bound.

Friction is conservative by construction while ``t_breakaway`` remains
provisional: the inverse feedforward charges the full ``±t_breakaway`` Coulomb
model against intended motion and never credits friction as helpful. The
independent recorded-response disturbance estimate is separately bounded to
``±t_breakaway``. The resulting deliberate design tolerance is two
breakaway-equivalent loads; event-anchored identification may tighten it later,
but argument or feel may not.
"""

from __future__ import annotations

import math

import numpy as np

from openpilot.selfdrive.controls.lib.blatv2.candidate_common import (
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

    self.schedules = np.zeros((MAX_SIGN_SCHEDULES, MAX_DECISION_STEPS), dtype=np.int8)
    self.crossing_indices = np.empty(MAX_DECISION_STEPS, dtype=np.int16)
    self.schedule_change_indices = np.full(
      MAX_SIGN_SCHEDULES, -1, dtype=np.int16,
    )
    self.schedule_change_signs = np.zeros(MAX_SIGN_SCHEDULES, dtype=np.int8)
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

    self._state = PlantState(0.0, 0.0, 0.0, 0.0)
    self._solve_residual = 0.0
    self.winning_schedule_index = -1
    self.has_warm_start = False

  def reset(self) -> None:
    self.result.invalidate()
    self.winning_schedule_index = -1
    self.has_warm_start = False

  @staticmethod
  def _sign(value: float) -> int:
    return 1 if value > 0.0 else (-1 if value < 0.0 else 0)

  def _build_schedules(self, count: int) -> int:
    base = self.schedules[0]
    previous_nonzero = 0
    crossing_count = 0
    for index in range(count):
      sign = self._sign(float(self.workspace.feedforward[index]))
      base[index] = sign
      if sign != 0:
        if previous_nonzero != 0 and sign != previous_nonzero:
          if crossing_count < MAX_DECISION_STEPS:
            self.crossing_indices[crossing_count] = index
          crossing_count += 1
        previous_nonzero = sign

    required = 1 + 2 * crossing_count
    if required > MAX_SIGN_SCHEDULES:
      return -required

    self.schedule_change_indices[0] = -1
    self.schedule_change_signs[0] = 0
    schedule_count = 1
    for crossing_number in range(crossing_count):
      crossing = int(self.crossing_indices[crossing_number])
      old_sign = int(base[crossing - 1])
      new_sign = int(base[crossing])

      self.schedule_change_indices[schedule_count] = crossing - 1
      self.schedule_change_signs[schedule_count] = new_sign
      self.schedule_change_indices[schedule_count + 1] = crossing
      self.schedule_change_signs[schedule_count + 1] = old_sign
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

  def _select_warm_schedule(self, schedule_count: int) -> int:
    """Reuse the prior winning ordinal in O(1), or bootstrap from base."""
    if (
      self.has_warm_start
      and 0 <= self.winning_schedule_index < schedule_count
    ):
      return self.winning_schedule_index
    return 0

  def _materialize_schedule(
    self,
    schedule_index: int,
    count: int,
  ) -> None:
    if schedule_index == 0:
      return
    selected = self.schedules[schedule_index]
    base = self.schedules[0]
    for index in range(count):
      selected[index] = base[index]
    changed_index = int(self.schedule_change_indices[schedule_index])
    selected[changed_index] = self.schedule_change_signs[schedule_index]

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
    *,
    workspace_prepared: bool = False,
  ) -> CandidateResult:
    result = self.result
    self._state = state
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
      count = self.workspace.decision_count
      schedule_count = self._build_schedules(count)
      if schedule_count < 0:
        result.invalidate(state.applied_torque, CandidateStatus.ENUMERATION_EXHAUSTED)
        result.candidate_count = -schedule_count
        result.available_schedule_count = -schedule_count
        return result
      self._assemble_quadratic(count, state.applied_torque)
    except (ValueError, OverflowError):
      result.invalidate(state.applied_torque, CandidateStatus.INPUT_INVALID)
      return result

    selected_schedule = self._select_warm_schedule(
      schedule_count,
    )
    self._materialize_schedule(selected_schedule, count)
    self.winning_schedule_index = -1
    status = self._solve_schedule(selected_schedule, count)
    if status != CandidateStatus.OK:
      result.invalidate(state.applied_torque, status)
      result.candidate_count = 1
      result.available_schedule_count = schedule_count
      self.has_warm_start = False
      return result

    best_residual = self._solve_residual
    self.winning_schedule_index = selected_schedule
    for index in range(count):
      self.best_solution[index] = self.solution[index]
    self.has_warm_start = True

    result.command_torque = self.twin.apply_slew(state.applied_torque, float(self.best_solution[0]))
    result.status = CandidateStatus.OK
    result.candidate_count = 1
    result.available_schedule_count = schedule_count
    result.optimality_residual = best_residual
    return result
