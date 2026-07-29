import json
import math
from pathlib import Path

import numpy as np

from openpilot.selfdrive.controls.lib.blatv2.candidate_common import (
  CandidateWorkspace,
  decision_cell_coulomb_direction,
  decision_cell_friction,
  measured_rack_friction,
)
from openpilot.selfdrive.controls.lib.blatv2.controller import (
  DECISION_DT,
  CandidateStatus,
  ControllerParams,
  ObserverStatus,
)
from openpilot.selfdrive.controls.lib.blatv2.reference import interpolate_buffer
from openpilot.selfdrive.controls.lib.blatv2.fallback import (
  InverseEpsActionController,
  InverseEpsLQIFallback,
)
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
ACTION_PARAMS = ControllerParams(
  0.05, 0.01, 0.5, 0.5, True, 0.00091683, 0.03,
)
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
    "sigma_curvature": (0.00091683, True, True),
  }
  assert payload["kinetic_friction"] == {
    "value": 0.03,
    "units": "normalized torque",
    "owner_feel_dial": False,
    "provisional": True,
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
    decision_cell_coulomb_direction(2.0, 1.0, 0.0),
    decision_cell_coulomb_direction(1.0, -1.0, 0.0),
    decision_cell_coulomb_direction(-1.0, -2.0, 0.0),
  )
  assert directions == (1.0, 0.0, -1.0)
  assert max(
    abs(right - left)
    for left, right in zip(directions, directions[1:], strict=False)
  ) == 1.0


def test_coulomb_feedforward_retains_full_breakaway_from_stiction():
  assert decision_cell_coulomb_direction(0.0, 0.0, 2.0) == 1.0
  assert decision_cell_coulomb_direction(0.0, 0.0, -2.0) == -1.0
  assert decision_cell_coulomb_direction(0.0, 0.0, 0.0) == 0.0


def test_feedforward_uses_static_breakaway_only_until_rack_moves():
  static = decision_cell_friction(0.0, 0.0, 2.0, 0.09, 0.03)
  moving = decision_cell_friction(2.0, 1.0, 0.0, 0.09, 0.03)
  crossing = decision_cell_friction(1.0, -3.0, 0.0, 0.09, 0.03)
  assert static == 0.09
  assert moving == 0.03
  assert crossing == -0.015


def test_measured_rack_stick_to_slip_transition_is_continuous():
  values = tuple(
    measured_rack_friction(rate, 1.0, 0.09, 0.03)
    for rate in (0.0, 1.0, 2.0, 3.0, 4.0, 8.0)
  )
  assert values == (0.09, 0.075, 0.06, 0.045, 0.03, 0.03)
  assert measured_rack_friction(0.0, -1.0, 0.09, 0.03) == -0.09


def test_action_computed_torque_places_error_poles_at_physical_rack_rate():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = PlantState(0.0, 0.0, 0.0, 5.0)
  result = InverseEpsActionController(twin, CONTROLLER_PARAMS).compute(
    state,
    ALIGN_INPUTS,
    REFERENCE_TIMES,
    np.full(3, 0.001, dtype=np.float64),
    3,
    1.0,
    0.0,
    0.0,
    ObserverStatus.ACTIVE,
    action_time=0.20,
  )
  expected = (
    result.desired_acceleration_deg_s2
    + 2.0 * PLANT_PARAMS.b_steer
      * (result.desired_rate_deg_s - result.predicted_rate_deg_s)
    + PLANT_PARAMS.b_steer * PLANT_PARAMS.b_steer
      * (result.desired_angle_deg - result.predicted_angle_deg)
  )
  assert math.isclose(result.required_acceleration_deg_s2, expected, abs_tol=1e-12)


def test_action_horizon_changes_torque_reachability_not_position_authority():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = PlantState(0.0, 0.0, 0.0, 5.0)
  times = np.arange(0.0, 1.05, 0.05, dtype=np.float64)
  base = np.full(len(times), 0.00001, dtype=np.float64)
  mutated = base.copy()
  mutated[times >= 0.35] = np.linspace(
    0.00001, 0.15, np.count_nonzero(times >= 0.35),
  )
  first = InverseEpsActionController(twin, ACTION_PARAMS).compute(
    state,
    ALIGN_INPUTS,
    times,
    base,
    len(times),
    1.0,
    0.05,
    0.0,
    ObserverStatus.ACTIVE,
    action_time=0.20,
  )
  first_position = first.desired_angle_deg
  first_command = first.command_torque
  second = InverseEpsActionController(twin, ACTION_PARAMS).compute(
    state,
    ALIGN_INPUTS,
    times,
    mutated,
    len(times),
    1.0,
    0.05,
    0.0,
    ObserverStatus.ACTIVE,
    action_time=0.20,
  )
  assert second.desired_angle_deg == first_position
  assert second.horizon_assist_active
  assert second.horizon_demand_time_seconds > 0.20
  assert abs(second.command_torque) > abs(first_command)
  assert abs(second.raw_command_torque) == 1.0


def test_action_breakaway_requires_persistent_large_error():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = PlantState(0.0, 0.0, 0.0, 5.0)
  candidate = InverseEpsActionController(twin, ACTION_PARAMS)
  curvatures = np.full(3, 0.01, dtype=np.float64)

  for frame in range(1, 5):
    result = candidate.compute(
      state,
      ALIGN_INPUTS,
      REFERENCE_TIMES,
      curvatures,
      3,
      1.0,
      0.0,
      0.0,
      ObserverStatus.ACTIVE,
      action_time=0.20,
    )
    assert not result.breakaway_active
    assert result.breakaway_persistence_frames == frame
    assert result.friction_torque == 0.0

  result = candidate.compute(
    state,
    ALIGN_INPUTS,
    REFERENCE_TIMES,
    curvatures,
    3,
    1.0,
    0.0,
    0.0,
    ObserverStatus.ACTIVE,
    action_time=0.20,
  )
  assert result.breakaway_active
  assert result.breakaway_persistence_frames == 5
  assert abs(result.friction_torque) == PLANT_PARAMS.t_breakaway


def test_action_subthreshold_center_noise_never_arms_breakaway():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = PlantState(0.0, 0.0, 0.0, 17.0)
  candidate = InverseEpsActionController(twin, ACTION_PARAMS)
  for frame in range(20):
    curvature = 0.0001 if frame % 2 == 0 else -0.0001
    result = candidate.compute(
      state,
      ALIGN_INPUTS,
      REFERENCE_TIMES,
      np.full(3, curvature, dtype=np.float64),
      3,
      1.0,
      0.0,
      0.0,
      ObserverStatus.ACTIVE,
      action_time=0.20,
    )
    assert not result.breakaway_active
    assert result.friction_torque == 0.0


def test_action_breakaway_transitions_continuously_to_kinetic_friction():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = PlantState(0.0, 0.0, 0.0, 5.0)
  candidate = InverseEpsActionController(twin, ACTION_PARAMS)
  curvatures = np.full(3, 0.01, dtype=np.float64)
  for _ in range(5):
    result = candidate.compute(
      state, ALIGN_INPUTS, REFERENCE_TIMES, curvatures, 3, 1.0, 0.0,
      0.0, ObserverStatus.ACTIVE, action_time=0.20,
    )
  assert abs(result.friction_torque) == 0.09

  state.rate_deg_s = -2.0
  result = candidate.compute(
    state, ALIGN_INPUTS, REFERENCE_TIMES, curvatures, 3, 1.0, 0.0,
    0.0, ObserverStatus.ACTIVE, action_time=0.20,
  )
  assert math.isclose(abs(result.friction_torque), 0.06)
  assert result.breakaway_active

  state.rate_deg_s = -4.0
  result = candidate.compute(
    state, ALIGN_INPUTS, REFERENCE_TIMES, curvatures, 3, 1.0, 0.0,
    0.0, ObserverStatus.ACTIVE, action_time=0.20,
  )
  assert math.isclose(abs(result.friction_torque), 0.03)
  assert not result.breakaway_active


def test_action_horizon_assist_is_clamped_before_predicted_path_lead():
  fast_params = PlantParams(
    400000.0,
    PLANT_PARAMS.b_steer,
    0.0,
    PLANT_PARAMS.actuation_delay,
    PLANT_PARAMS.steer_max,
    PLANT_PARAMS.delta_up,
    PLANT_PARAMS.delta_down,
    PLANT_PARAMS.steer_step,
    PLANT_PARAMS.provisional,
    PLANT_PARAMS.torque_per_lataccel_speed_nodes,
    PLANT_PARAMS.torque_per_lataccel_values,
  )
  twin = PlantTwin(fast_params, ALIGN_PARAMS)
  state = PlantState(0.0, 0.0, 0.0, 3.0)
  times = np.arange(0.0, 1.05, 0.05, dtype=np.float64)
  curvatures = np.full(len(times), 0.00001, dtype=np.float64)
  curvatures[times >= 0.35] = np.linspace(
    0.00001, 0.15, np.count_nonzero(times >= 0.35),
  )
  candidate = InverseEpsActionController(twin, ACTION_PARAMS)
  result = candidate.compute(
    state, ALIGN_INPUTS, times, curvatures, len(times), 1.0, 0.05, 0.0,
    ObserverStatus.ACTIVE, action_time=0.20,
  )
  assert result.horizon_assist_active
  assert result.no_lead_limited
  next_angle = candidate._next_angle(result.command_torque, ALIGN_INPUTS, 0.0)
  next_desired = candidate._interpolate(
    candidate.workspace.desired_angles,
    (0.20 + 0.01) / DECISION_DT,
    candidate.workspace.decision_count,
  )
  direction = math.copysign(1.0, result.desired_angle_deg)
  assert direction * (next_angle - next_desired) <= 1e-9
  assert result.command_torque == twin.apply_slew(
    state.applied_torque, result.raw_command_torque,
  )


def test_action_controller_uses_full_authority_for_large_low_speed_error():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = PlantState(0.0, 0.0, 0.0, 3.0)
  result = InverseEpsActionController(twin, CONTROLLER_PARAMS).compute(
    state,
    ALIGN_INPUTS,
    REFERENCE_TIMES,
    np.full(3, 0.15, dtype=np.float64),
    3,
    1.0,
    0.05,
    0.0,
    ObserverStatus.ACTIVE,
    action_time=0.20,
  )
  assert result.status == CandidateStatus.OK
  assert abs(result.raw_command_torque) == 1.0
  assert result.command_torque == twin.apply_slew(
    state.applied_torque, result.raw_command_torque,
  )
  assert result.slew_constrained


def test_action_inverse_cancels_rack_damping_at_predicted_state():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = PlantState(0.0, 0.0, 0.0, 8.0)
  curvatures = np.full(3, 0.00003, dtype=np.float64)
  candidate = InverseEpsActionController(twin, CONTROLLER_PARAMS)
  result = candidate.compute(
    state,
    ALIGN_INPUTS,
    REFERENCE_TIMES,
    curvatures,
    3,
    1.0,
    0.05,
    0.0,
    ObserverStatus.ACTIVE,
    action_time=0.20,
  )
  recovered_acceleration = (
    result.dynamic_torque * PLANT_PARAMS.k_t
    - PLANT_PARAMS.b_steer * result.predicted_rate_deg_s
  )
  assert math.isclose(
    recovered_acceleration,
    result.required_acceleration_deg_s2,
    abs_tol=1e-12,
  )


def test_coulomb_feedforward_rejects_non_finite_inputs():
  for values in (
    (math.nan, 0.0, 1.0),
    (0.0, math.inf, 1.0),
    (0.0, 0.0, -math.inf),
  ):
    with np.testing.assert_raises(ValueError):
      decision_cell_coulomb_direction(*values)


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


def test_action_controller_has_no_observer_writable_state():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = feasible_steady_state(twin)
  state.angle_deg += 1.0
  candidate = InverseEpsActionController(twin, CONTROLLER_PARAMS)
  active = candidate.compute(
    state, ALIGN_INPUTS, REFERENCE_TIMES, REFERENCE_CURVATURES, 3, 1.0, 0.0, 0.0, ObserverStatus.ACTIVE,
  )
  active_values = (
    active.command_torque,
    active.raw_command_torque,
    active.required_acceleration_deg_s2,
  )
  frozen = candidate.compute(
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
  assert active_values == (
    frozen.command_torque,
    frozen.raw_command_torque,
    frozen.required_acceleration_deg_s2,
  )
  reset = candidate.compute(
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
  assert active_values == (
    reset.command_torque,
    reset.raw_command_torque,
    reset.required_acceleration_deg_s2,
  )
