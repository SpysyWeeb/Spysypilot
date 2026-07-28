import json
import math
from pathlib import Path

import numpy as np

from openpilot.selfdrive.controls.lib.blatv2.candidate_common import (
  CandidateWorkspace,
  linear_rate_coulomb_direction,
)
from openpilot.selfdrive.controls.lib.blatv2.controller import (
  DECISION_DT,
  CandidateStatus,
  ControllerParams,
  ObserverStatus,
)
from openpilot.selfdrive.controls.lib.blatv2.reference import interpolate_buffer
from openpilot.selfdrive.controls.lib.blatv2.fallback import InverseEpsLQIFallback
from openpilot.selfdrive.controls.lib.blatv2.mpc import ModelFollowingTorqueMPC
from openpilot.selfdrive.controls.lib.blatv2.plant import (
  AlignInputs,
  AlignParams,
  PlantParams,
  PlantState,
  PlantTwin,
)


PLANT_PARAMS = PlantParams(
  4000.0,
  10.0,
  0.09,
  0.0,
  409,
  4,
  7,
  1,
  True,
  (2.5, 5.5, 8.5, 12.0, 16.5, 21.0),
  (0.85, 0.39, 0.38, 0.36, 0.286, 0.288),
)
ALIGN_PARAMS = AlignParams(
  2000.0, 3.0, 1.2, 100000.0, 110000.0, 15.0, 0.0, 0.0
)
CONTROLLER_PARAMS = ControllerParams(0.05, 0.01, 0.5, 0.5, True)
ALIGN_INPUTS = AlignInputs(0.0, 0.0, 1.0, 15.0, True)
REFERENCE_TIMES = np.asarray((0.0, 0.5, 1.0), dtype=np.float64)
REFERENCE_CURVATURES = np.asarray((0.001, 0.001, 0.001), dtype=np.float64)


def feasible_steady_state(twin: PlantTwin) -> PlantState:
  angle = twin.angle_from_curvature(0.001, 10.0, ALIGN_INPUTS)
  initial = PlantState(angle, 0.0, 0.0, 10.0)
  return PlantState(angle, 0.0, twin.aligning_torque(initial, ALIGN_INPUTS), 10.0)


def test_candidate_status_values_are_distinct_and_stable():
  assert [int(status) for status in CandidateStatus] == [0, 1, 2, 3, 4]


def test_shared_sigma_values_are_marked_as_provisional_owner_feel_dials():
  payload = json.loads(
    Path("openpilot/selfdrive/controls/lib/blatv2/controller_seed_params.json").read_text()
  )
  assert payload["provisional"]
  assert {
    name: (entry["value"], entry["owner_feel_dial"], entry["provisional"])
    for name, entry in payload.items()
    if name.startswith("sigma_")
  } == {
    "sigma_y": (0.05, True, True),
    "sigma_heading": (0.01, True, True),
    "sigma_torque_rate": (0.5, True, True),
  }


def test_mpc_transparency_for_already_feasible_smooth_reference():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = feasible_steady_state(twin)
  candidate = ModelFollowingTorqueMPC(twin, CONTROLLER_PARAMS)
  result = candidate.compute(
    state, ALIGN_INPUTS, REFERENCE_TIMES, REFERENCE_CURVATURES, 3, 1.0, 0.0, 0.0,
  )
  assert result.status == CandidateStatus.OK
  assert math.isclose(result.command_torque, candidate.workspace.feedforward[0], abs_tol=1e-9)


def test_fallback_transparency_for_already_feasible_smooth_reference():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = feasible_steady_state(twin)
  candidate = InverseEpsLQIFallback(twin, CONTROLLER_PARAMS)
  result = candidate.compute(
    state,
    ALIGN_INPUTS,
    REFERENCE_TIMES,
    REFERENCE_CURVATURES,
    3,
    1.0,
    0.0,
    0.0,
    ObserverStatus.ACTIVE,
  )
  assert result.status == CandidateStatus.OK
  assert math.isclose(result.command_torque, candidate.workspace.feedforward[0], abs_tol=1e-9)


def test_monotonic_workspace_interpolation_is_bit_exact_to_scalar_contract():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = feasible_steady_state(twin)
  times = np.asarray((0.0, 0.1, 0.3, 0.8, 1.0), dtype=np.float64)
  curvatures = np.asarray(
    (0.0, 0.001, -0.002, 0.003, 0.001),
    dtype=np.float64,
  )
  workspace = CandidateWorkspace()
  workspace.fill(
    twin,
    state,
    ALIGN_INPUTS,
    times,
    curvatures,
    len(times),
    1.0,
    0.0,
  )

  expected_decisions = np.asarray([
    interpolate_buffer(
      times, curvatures, len(times), index * DECISION_DT,
    )
    for index in range(workspace.decision_count)
  ], dtype=np.float64)
  assert (
    workspace.reference_curvatures[:workspace.decision_count].tobytes()
    == expected_decisions.tobytes()
  )


def test_coulomb_feedforward_crosses_zero_without_hard_sign_flip():
  directions = (
    linear_rate_coulomb_direction(2.0, 1.0, 0.0),
    linear_rate_coulomb_direction(2.0, -2.0, 0.0),
    linear_rate_coulomb_direction(2.0, -6.0, 0.0),
  )
  assert directions == (1.0, 0.0, -0.5)
  assert max(
    abs(right - left)
    for left, right in zip(directions, directions[1:], strict=False)
  ) == 1.0


def test_coulomb_feedforward_retains_full_breakaway_from_stiction():
  assert linear_rate_coulomb_direction(0.0, 0.0, 2.0) == 1.0
  assert linear_rate_coulomb_direction(0.0, 0.0, -2.0) == -1.0
  assert linear_rate_coulomb_direction(0.0, 0.0, 0.0) == 0.0


def test_coulomb_feedforward_rejects_non_finite_inputs():
  for values in (
    (math.nan, 0.0, 1.0),
    (0.0, math.inf, 1.0),
    (0.0, 0.0, -math.inf),
  ):
    with np.testing.assert_raises(ValueError):
      linear_rate_coulomb_direction(*values)


def test_mpc_solves_only_one_warm_selected_schedule():
  class MultiScheduleMPC(ModelFollowingTorqueMPC):
    solved_indices: list[int]

    def __init__(self, twin: PlantTwin, params: ControllerParams):
      super().__init__(twin, params)
      self.solved_indices = []

    def _build_schedules(self, count: int) -> int:
      for schedule_index in range(3):
        for index in range(count):
          self.schedules[schedule_index, index] = 1
      return 3

    def _solve_schedule(self, schedule_index: int, count: int) -> CandidateStatus:
      self.solved_indices.append(schedule_index)
      return super()._solve_schedule(schedule_index, count)

  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = feasible_steady_state(twin)
  candidate = MultiScheduleMPC(twin, CONTROLLER_PARAMS)
  result = candidate.compute(
    state,
    ALIGN_INPUTS,
    REFERENCE_TIMES,
    REFERENCE_CURVATURES,
    3,
    1.0,
    0.0,
    0.0,
  )
  assert result.status == CandidateStatus.OK
  assert result.candidate_count == 1
  assert result.available_schedule_count == 3
  assert candidate.solved_indices == [0]
  assert candidate.winning_schedule_index == 0


def test_mpc_warm_schedule_reuses_prior_winner_or_falls_back_to_base():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  candidate = ModelFollowingTorqueMPC(twin, CONTROLLER_PARAMS)
  candidate.has_warm_start = True
  candidate.winning_schedule_index = 2

  assert candidate._select_warm_schedule(3) == 2
  assert candidate._select_warm_schedule(2) == 0
  candidate.reset()
  assert candidate._select_warm_schedule(3) == 0


def test_enumeration_exhaustion_is_legible():
  class ExhaustedMPC(ModelFollowingTorqueMPC):
    def _build_schedules(self, _count: int) -> int:
      return -17

  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = feasible_steady_state(twin)
  result = ExhaustedMPC(twin, CONTROLLER_PARAMS).compute(
    state, ALIGN_INPUTS, REFERENCE_TIMES, REFERENCE_CURVATURES, 3, 1.0, 0.0, 0.0,
  )
  assert result.status == CandidateStatus.ENUMERATION_EXHAUSTED


def test_fallback_integral_freezes_and_resets_with_observer_lifecycle():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = feasible_steady_state(twin)
  state.angle_deg += 1.0
  candidate = InverseEpsLQIFallback(twin, CONTROLLER_PARAMS)

  candidate.compute(
    state, ALIGN_INPUTS, REFERENCE_TIMES, REFERENCE_CURVATURES, 3, 1.0, 0.0, 0.0, ObserverStatus.ACTIVE,
  )
  learned = candidate.integral_lateral_error
  assert learned != 0.0
  candidate.compute(
    state,
    ALIGN_INPUTS,
    REFERENCE_TIMES,
    REFERENCE_CURVATURES,
    3,
    1.0,
    0.0,
    0.0,
    ObserverStatus.FROZEN_RECORDED_CONSTRAINT,
  )
  assert candidate.integral_lateral_error == learned
  candidate.compute(
    state,
    ALIGN_INPUTS,
    REFERENCE_TIMES,
    REFERENCE_CURVATURES,
    3,
    1.0,
    0.0,
    0.0,
    ObserverStatus.RESET_STEERING_PRESSED,
  )
  assert candidate.integral_lateral_error == 0.0
