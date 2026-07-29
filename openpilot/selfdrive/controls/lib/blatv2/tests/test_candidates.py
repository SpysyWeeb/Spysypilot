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
  0.05, 0.01, 0.5, 0.5, True, 0.00091683, 0.03, 0.025, 1.0,
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
  assert payload["tracking_stiffness"] == {
    "value": 0.025,
    "units": "normalized torque/deg",
    "owner_feel_dial": True,
    "provisional": True,
    "provenance": (
      "v217 restores the calm small-error stiffness after route bc showed "
      "that a stronger global proportional gain amplified routine plan error "
      "into correction activity."
    ),
  }
  assert payload["authority_transition_error_deg"] == {
    "value": 1.0,
    "units": "steering-wheel deg",
    "owner_feel_dial": True,
    "provisional": True,
    "provenance": (
      "v217 route-bc and 9f sweep selected 1.0 degree: it materially improves "
      "every unsaturated bc named turn, matches v14's 9f direct-handoff "
      "delivered fraction, and remains calmer than v14 in the 15.6-20.1 m/s "
      "band. It adds no timer, latch, integral, or second command path."
    ),
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


def test_action_feedback_separates_position_stiffness_from_rate_damping():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = PlantState(0.0, 0.0, 0.0, 5.0)
  result = InverseEpsActionController(twin, ACTION_PARAMS).compute(
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
  angle_error = result.desired_angle_deg - result.predicted_angle_deg
  transition_fraction = min(
    abs(angle_error) / ACTION_PARAMS.authority_transition_error_deg, 1.0,
  )
  smoothstep = transition_fraction * transition_fraction * (
    3.0 - 2.0 * transition_fraction
  )
  static_authority = max(
    PLANT_PARAMS.t_breakaway
    - ACTION_PARAMS.tracking_stiffness
      * ACTION_PARAMS.authority_transition_error_deg,
    0.0,
  )
  position_feedback = math.copysign(
    ACTION_PARAMS.tracking_stiffness * abs(angle_error)
    + static_authority * smoothstep,
    angle_error,
  )
  expected = (
    result.desired_acceleration_deg_s2
    + PLANT_PARAMS.b_steer
      * (result.desired_rate_deg_s - result.predicted_rate_deg_s)
    + PLANT_PARAMS.k_t * (
      position_feedback
      + (PLANT_PARAMS.delta_up / PLANT_PARAMS.steer_max / 4.0)
        * (result.desired_rate_deg_s - result.predicted_rate_deg_s)
    )
  )
  assert math.isclose(result.required_acceleration_deg_s2, expected, abs_tol=1e-12)


def test_tracking_stiffness_changes_small_error_feedback_without_changing_feedforward():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = PlantState(0.0, 0.0, 0.0, 5.0)
  reference = np.full(3, 0.001, dtype=np.float64)
  calm = InverseEpsActionController(twin, ACTION_PARAMS).compute(
    state,
    ALIGN_INPUTS,
    REFERENCE_TIMES,
    reference,
    3,
    1.0,
    0.0,
    0.0,
    ObserverStatus.ACTIVE,
    action_time=0.20,
  )
  calm_feedforward = calm.feedforward_torque
  calm_feedback = calm.feedback_torque
  stiffer_params = ControllerParams(
    0.05, 0.01, 0.5, 0.5, True, 0.00091683, 0.03, 0.05, 1.0,
  )
  stiffer = InverseEpsActionController(twin, stiffer_params).compute(
    state,
    ALIGN_INPUTS,
    REFERENCE_TIMES,
    reference,
    3,
    1.0,
    0.0,
    0.0,
    ObserverStatus.ACTIVE,
    action_time=0.20,
  )
  assert stiffer.feedforward_torque == calm_feedforward
  assert abs(stiffer.feedback_torque) > abs(calm_feedback)


def test_tracking_stiffness_must_be_finite_and_positive():
  for value in (0.0, -1.0, math.inf, math.nan):
    params = ControllerParams(
      0.05, 0.01, 0.5, 0.5, True, 0.00091683, 0.03, value,
    )
    try:
      params.validate()
    except ValueError:
      pass
    else:
      raise AssertionError(f"accepted invalid tracking stiffness {value!r}")


def test_authority_transition_error_must_be_finite_and_positive():
  for value in (0.0, -1.0, math.inf, math.nan):
    params = ControllerParams(
      0.05, 0.01, 0.5, 0.5, True, 0.00091683, 0.03, 0.025,
      value,
    )
    try:
      params.validate()
    except ValueError:
      pass
    else:
      raise AssertionError(f"accepted invalid authority transition {value!r}")


def test_rate_feedback_quantum_is_bounded_to_one_torque_build_step():
  rate_damping = (
    PLANT_PARAMS.delta_up
    / PLANT_PARAMS.steer_max
    / 4.0
  )
  assert math.isclose(
    rate_damping * 4.0,
    PLANT_PARAMS.delta_up / PLANT_PARAMS.steer_max,
    abs_tol=1e-15,
  )


def test_action_far_future_does_not_pull_current_torque_or_position():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = PlantState(0.0, 0.0, 0.0, 5.0)
  times = np.arange(0.0, 1.05, 0.05, dtype=np.float64)
  base = np.full(len(times), 0.00001, dtype=np.float64)
  mutated = base.copy()
  mutated[times >= 0.40] = np.linspace(
    0.00001, 0.15, np.count_nonzero(times >= 0.40),
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
  assert second.command_torque == first_command
  assert second.raw_command_torque == first.raw_command_torque
  assert not second.horizon_assist_active
  assert second.horizon_demand_time_seconds == 0.0


def test_action_authority_curve_reaches_static_load_at_transition():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = PlantState(0.0, 0.0, 0.0, 5.0)
  curvature = twin.curvature_from_angle(
    ACTION_PARAMS.authority_transition_error_deg, 5.0, ALIGN_INPUTS,
  )
  curvatures = np.full(3, curvature, dtype=np.float64)
  result = InverseEpsActionController(twin, ACTION_PARAMS).compute(
    state, ALIGN_INPUTS, REFERENCE_TIMES, curvatures, 3, 1.0, 0.0,
    0.0, ObserverStatus.ACTIVE, action_time=0.20,
  )
  rate_feedback = (
    PLANT_PARAMS.delta_up / PLANT_PARAMS.steer_max / 4.0
  ) * (result.desired_rate_deg_s - result.predicted_rate_deg_s)
  assert math.isclose(
    result.feedback_torque - rate_feedback,
    PLANT_PARAMS.t_breakaway,
    abs_tol=1e-12,
  )
  assert not result.breakaway_active
  assert result.breakaway_persistence_frames == 0


def test_action_authority_curve_is_continuous_at_transition():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = PlantState(0.0, 0.0, 0.0, 5.0)
  transition = ACTION_PARAMS.authority_transition_error_deg
  epsilon = 1e-5
  position_feedback = []
  for angle in (transition - epsilon, transition, transition + epsilon):
    curvature = twin.curvature_from_angle(angle, 5.0, ALIGN_INPUTS)
    result = InverseEpsActionController(twin, ACTION_PARAMS).compute(
      state,
      ALIGN_INPUTS,
      REFERENCE_TIMES,
      np.full(3, curvature, dtype=np.float64),
      3,
      1.0,
      0.0, 0.0, ObserverStatus.ACTIVE, action_time=0.20,
    )
    rate_feedback = (
      PLANT_PARAMS.delta_up / PLANT_PARAMS.steer_max / 4.0
    ) * (result.desired_rate_deg_s - result.predicted_rate_deg_s)
    position_feedback.append(result.feedback_torque - rate_feedback)
  assert position_feedback[0] < position_feedback[1] < position_feedback[2]
  assert math.isclose(
    (position_feedback[2] - position_feedback[1]) / epsilon,
    ACTION_PARAMS.tracking_stiffness,
    rel_tol=1e-9,
  )
  assert math.isclose(
    (position_feedback[1] - position_feedback[0]) / epsilon,
    ACTION_PARAMS.tracking_stiffness,
    rel_tol=1e-4,
  )


def test_action_authority_curve_is_stateless_across_reversals():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = PlantState(0.0, 0.0, 0.0, 5.0)
  candidate = InverseEpsActionController(twin, ACTION_PARAMS)
  curvature = twin.curvature_from_angle(1.0, 5.0, ALIGN_INPUTS)
  positive = candidate.compute(
    state, ALIGN_INPUTS, REFERENCE_TIMES,
    np.full(3, curvature, dtype=np.float64), 3, 1.0, 0.0, 0.0,
    ObserverStatus.ACTIVE, action_time=0.20,
  )
  positive_raw = positive.raw_command_torque
  negative = candidate.compute(
    state, ALIGN_INPUTS, REFERENCE_TIMES,
    np.full(3, -curvature, dtype=np.float64), 3, 1.0, 0.0, 0.0,
    ObserverStatus.RESET_STEERING_PRESSED, action_time=0.20,
  )
  assert math.isclose(negative.raw_command_torque, -positive_raw, abs_tol=1e-12)


def test_action_moving_rack_uses_signed_kinetic_friction_only():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = PlantState(0.0, -4.0, 0.0, 5.0)
  result = InverseEpsActionController(twin, ACTION_PARAMS).compute(
    state,
    ALIGN_INPUTS,
    REFERENCE_TIMES,
    REFERENCE_CURVATURES,
    3,
    1.0,
    0.0,
    0.0,
    ObserverStatus.ACTIVE,
    action_time=0.20,
  )
  assert result.friction_torque == -ACTION_PARAMS.kinetic_friction
  assert not result.breakaway_active


def test_action_requested_load_is_evaluated_at_desired_rack_position():
  twin = PlantTwin(PLANT_PARAMS, ALIGN_PARAMS)
  state = PlantState(25.0, 0.0, 0.0, 10.0)
  curvatures = np.full(3, 0.005, dtype=np.float64)
  result = InverseEpsActionController(twin, ACTION_PARAMS).compute(
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
  desired_load = twin.aligning_torque_values(
    result.desired_angle_deg, result.action_speed_mps, ALIGN_INPUTS,
  )
  measured_load = twin.aligning_torque_values(
    result.predicted_angle_deg, result.action_speed_mps, ALIGN_INPUTS,
  )
  assert result.aligning_torque == desired_load
  assert result.aligning_torque != measured_load


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
  first = InverseEpsActionController(twin, ACTION_PARAMS).compute(
    state, ALIGN_INPUTS, REFERENCE_TIMES, REFERENCE_CURVATURES, 3, 1.0, 0.0, 0.0, ObserverStatus.ACTIVE,
  )
  first_values = (
    first.command_torque,
    first.raw_command_torque,
    first.required_acceleration_deg_s2,
  )
  candidate = InverseEpsActionController(twin, ACTION_PARAMS)
  active = candidate.compute(
    state, ALIGN_INPUTS, REFERENCE_TIMES, REFERENCE_CURVATURES, 3,
    1.0, 0.0, 0.0, ObserverStatus.ACTIVE,
  )
  active_values = (
    active.command_torque,
    active.raw_command_torque,
    active.required_acceleration_deg_s2,
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
  assert first_values == active_values
  assert active_values == (
    reset.command_torque,
    reset.raw_command_torque,
    reset.required_acceleration_deg_s2,
  )
