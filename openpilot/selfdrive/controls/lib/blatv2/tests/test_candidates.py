import json
import math
from pathlib import Path
import struct

import numpy as np

from openpilot.selfdrive.controls.lib.blatv2.candidate_common import (
  CandidateWorkspace,
  decision_cell_coulomb_direction,
  planned_rack_friction,
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
  0.05, 0.01, 0.5, 0.5, True, 0.00091683, 0.03, 0.025,
)
ALIGN_INPUTS = AlignInputs(0.0, 0.0, 1.0, 15.0, True)
REFERENCE_TIMES = np.asarray((0.0, 0.5, 1.0), dtype=np.float64)
REFERENCE_CURVATURES = np.asarray((0.001, 0.001, 0.001), dtype=np.float64)


def action_twin(
  plant_params: PlantParams = PLANT_PARAMS,
) -> PlantTwin:
  return PlantTwin(
    plant_params,
    ALIGN_PARAMS,
    kinetic_friction=ACTION_PARAMS.kinetic_friction,
  )


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
      "v219 uses the existing calm slope only for linear measured-rack " +
      "residual correction at the physical-effect time. Requested-path " +
      "feedforward and its exact state/slew-feasible trajectory own build, hold, " +
      "release, and full authority."
    ),
  }
  assert "authority_transition_error_deg" not in payload


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
    np.asarray(
      workspace.reference_curvatures[:workspace.decision_count],
      dtype=np.float64,
    ).tobytes()
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


def test_planned_feedforward_never_guesses_static_rack_state():
  static_reference = planned_rack_friction(
    0.0, 0.0, 2.0, 0.03,
  )
  moving = planned_rack_friction(2.0, 1.0, 0.0, 0.03)
  crossing = planned_rack_friction(1.0, -3.0, 0.0, 0.03)
  assert static_reference == 0.03
  assert moving == 0.03
  assert crossing == -0.015


def test_planned_friction_has_no_zero_to_four_degree_quantum_step():
  stuck_reference = planned_rack_friction(
    0.0, 0.0, 1.0, 0.03,
  )
  moving_reference = planned_rack_friction(
    4.0, 4.0, 0.0, 0.03,
  )
  assert stuck_reference == moving_reference == 0.03


def test_action_inverse_uses_measured_held_load_without_angle_gate():
  twin = action_twin()
  times = np.arange(0.0, 1.05, 0.05, dtype=np.float64)
  for angle in (0.0, 450.0):
    desired_angle = angle - 1.0
    reference = np.full(
      len(times),
      twin.curvature_from_angle(
        desired_angle, 0.0, ALIGN_INPUTS,
      ),
      dtype=np.float64,
    )
    result = InverseEpsActionController(twin, ACTION_PARAMS).compute(
      PlantState(angle, 0.0, 0.30, 0.0, 0.30),
      ALIGN_INPUTS,
      times,
      reference,
      len(times),
      1.0,
      0.12,
      0.0,
      ObserverStatus.ACTIVE,
    )
    assert result.friction_torque == -0.30


def test_action_controller_rejects_a_second_kinetic_friction_law():
  mismatched = PlantTwin(
    PLANT_PARAMS, ALIGN_PARAMS, kinetic_friction=0.09,
  )
  try:
    InverseEpsActionController(mismatched, ACTION_PARAMS)
  except ValueError as error:
    assert (
      "controller and plant must share one kinetic-friction value"
      in str(error)
    )
  else:
    raise AssertionError("mismatched kinetic-friction law was accepted")


def test_action_feedback_is_linear_and_separates_position_from_rate_damping():
  twin = action_twin()
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
  rate_error = result.desired_rate_deg_s - result.predicted_rate_deg_s
  position_feedback = ACTION_PARAMS.tracking_stiffness * angle_error
  expected = (
    result.desired_acceleration_deg_s2
    + PLANT_PARAMS.b_steer * rate_error
    + PLANT_PARAMS.k_t * (
      position_feedback
      + (PLANT_PARAMS.delta_up / PLANT_PARAMS.steer_max / 4.0)
        * rate_error
    )
  )
  assert math.isclose(result.required_acceleration_deg_s2, expected, abs_tol=1e-12)


def test_tracking_stiffness_changes_small_error_feedback_without_changing_feedforward():
  twin = action_twin()
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
    0.05, 0.01, 0.5, 0.5, True, 0.00091683, 0.03, 0.05,
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


def test_trajectory_transition_time_matches_exact_asymmetric_limits():
  twin = action_twin()
  candidate = InverseEpsActionController(twin, ACTION_PARAMS)
  frame_seconds = 0.01
  assert math.isclose(
    candidate._torque_transition_time(0.0, 1.0),
    PLANT_PARAMS.steer_max / PLANT_PARAMS.delta_up * frame_seconds,
    abs_tol=1e-15,
  )
  assert math.isclose(
    candidate._torque_transition_time(1.0, 0.0),
    PLANT_PARAMS.steer_max / PLANT_PARAMS.delta_down * frame_seconds,
    abs_tol=1e-15,
  )
  assert math.isclose(
    candidate._torque_transition_time(0.5, -0.5),
    (
      0.5 * PLANT_PARAMS.steer_max / PLANT_PARAMS.delta_down
      + 0.5 * PLANT_PARAMS.steer_max / PLANT_PARAMS.delta_up
    ) * frame_seconds,
    abs_tol=1e-15,
  )


def test_trajectory_predecessor_is_latest_value_that_reaches_next_cell():
  twin = action_twin()
  candidate = InverseEpsActionController(twin, ACTION_PARAMS)
  predecessor = candidate._latest_feasible_predecessor(
    0.0, 1.0, DECISION_DT,
  )
  expected = 1.0 - (
    DECISION_DT / 0.01
  ) * PLANT_PARAMS.delta_up / PLANT_PARAMS.steer_max
  assert math.isclose(predecessor, expected, abs_tol=1e-7)
  assert candidate._torque_transition_time(
    predecessor, 1.0,
  ) <= DECISION_DT + 1e-12
  assert candidate._torque_transition_time(
    predecessor - 1e-5, 1.0,
  ) > DECISION_DT


def test_future_slew_projection_uses_only_planned_inverse_feedforward():
  twin = action_twin()
  workspace = CandidateWorkspace()
  state = PlantState(0.0, 0.0, 0.0, 5.0)
  workspace.fill(
    twin,
    state,
    ALIGN_INPUTS,
    REFERENCE_TIMES,
    np.asarray((0.0, 0.004, -0.002), dtype=np.float64),
    3,
    1.0,
    0.0,
  )
  candidate = InverseEpsActionController(
    twin, ACTION_PARAMS, workspace,
  )
  action_target = 0.123
  candidate._project_inverse_trajectory(
    0.0, 0.0, action_target, 0.20,
  )
  assert candidate.trajectory_demands[0] == action_target
  for index in range(1, workspace.decision_count):
    assert math.isclose(
      candidate.trajectory_demands[index],
      workspace.feedforward[index],
      abs_tol=1e-15,
    )


def test_action_point_feedforward_reaches_authored_angle_or_uses_full_authority():
  twin = action_twin()
  candidate = InverseEpsActionController(twin, ACTION_PARAMS)
  candidate.predicted_state.applied_torque = 0.0
  target = candidate._action_point_feedforward(
    0.30,
    0.0,
    0.0,
    0.0,
    3.0,
    0.0,
    5.0,
    0.0,
    ALIGN_INPUTS,
    0.0,
  )
  source = PlantState(0.0, 0.0, 0.0, 5.0)
  reached = PlantState(0.0, 0.0, 0.0, 5.0)
  twin.predict_constant_request_into(
    source, 0.30, target, ALIGN_INPUTS, 0.0, reached, 0.01,
  )
  assert abs(reached.angle_deg - 3.0) < 0.01
  assert candidate._action_point_feedforward(
    0.30,
    0.0,
    0.0,
    0.0,
    100.0,
    0.0,
    5.0,
    0.0,
    ALIGN_INPUTS,
    0.0,
  ) == 1.0


def test_constant_request_endpoint_matches_exact_repeated_limiter():
  twin = action_twin()
  candidate = InverseEpsActionController(twin, ACTION_PARAMS)
  for source in (-1.0, -0.5, -2 / 409, 0.0, 2 / 409, 0.5, 1.0):
    for direction in (-1.0, 1.0):
      for frame_count in (0, 1, 2, 7, 31, 200):
        expected = source
        for _ in range(frame_count):
          expected = twin.apply_slew(expected, direction)
        actual = candidate._constant_request_endpoint(
          source, direction, frame_count,
        )
        assert math.isclose(actual, expected, abs_tol=2e-15)


def test_action_feedforward_is_independent_of_measured_rack_error():
  twin = action_twin()
  candidate = InverseEpsActionController(twin, ACTION_PARAMS)
  candidate.predicted_state.applied_torque = 0.25
  candidate.predicted_state.angle_deg = -50.0
  candidate.predicted_state.rate_deg_s = -20.0
  first = candidate._action_point_feedforward(
    0.30,
    0.0,
    1.0,
    2.0,
    4.0,
    8.0,
    5.0,
    0.1,
    ALIGN_INPUTS,
    0.0,
  )
  candidate.predicted_state.angle_deg = 50.0
  candidate.predicted_state.rate_deg_s = 20.0
  second = candidate._action_point_feedforward(
    0.30,
    0.0,
    1.0,
    2.0,
    4.0,
    8.0,
    5.0,
    0.1,
    ALIGN_INPUTS,
    0.0,
  )
  assert first == second


def test_action_feedforward_preserves_terminal_rack_momentum():
  twin = action_twin()
  candidate = InverseEpsActionController(twin, ACTION_PARAMS)
  candidate.predicted_state.applied_torque = 0.0
  target = candidate._action_point_feedforward(
    0.30,
    0.0,
    0.0,
    0.0,
    1.0,
    8.0,
    5.0,
    0.0,
    ALIGN_INPUTS,
    0.0,
  )
  source = PlantState(0.0, 0.0, 0.0, 5.0)
  reached = PlantState(0.0, 0.0, 0.0, 5.0)
  twin.predict_constant_request_into(
    source, 0.30, target, ALIGN_INPUTS, 0.0, reached, 0.01,
  )
  assert reached.angle_deg >= 1.0 - 0.01
  assert reached.rate_deg_s >= 8.0 - 0.01

  reverse_target = candidate._action_point_feedforward(
    0.30,
    0.0,
    0.0,
    0.0,
    -1.0,
    -8.0,
    5.0,
    0.0,
    ALIGN_INPUTS,
    0.0,
  )
  twin.predict_constant_request_into(
    source,
    0.30,
    reverse_target,
    ALIGN_INPUTS,
    0.0,
    reached,
    0.01,
  )
  assert reached.angle_deg <= -1.0 + 0.01
  assert reached.rate_deg_s <= -8.0 + 0.01


def test_action_effect_prediction_replays_queued_commands_not_latest_hold():
  delayed_params = PLANT_PARAMS.with_actuation_delay(0.03)
  twin = action_twin(delayed_params)
  candidate = InverseEpsActionController(twin, ACTION_PARAMS)
  state = PlantState(0.0, 0.0, 0.0, 5.0)
  candidate._remember_applied_torque(0.1)
  candidate._remember_applied_torque(0.2)
  candidate._remember_applied_torque(0.3)
  candidate._predict_to_effect_time(state, 0.03, ALIGN_INPUTS, 0.0)

  expected = PlantState(0.0, 0.0, 0.0, 5.0)
  for applied in (0.1, 0.2, 0.3):
    expected = twin.advance_applied(
      expected, applied, 0.01, ALIGN_INPUTS,
    )
  held = PlantState(0.0, 0.0, 0.0, 5.0)
  twin.predict_held_state_into(
    state, 0.03, ALIGN_INPUTS, 0.0, held, 0.01,
  )
  assert math.isclose(
    candidate.predicted_state.angle_deg, expected.angle_deg, abs_tol=1e-12,
  )
  assert math.isclose(
    candidate.predicted_state.rate_deg_s, expected.rate_deg_s, abs_tol=1e-12,
  )
  assert candidate.predicted_state.applied_torque == expected.applied_torque
  assert candidate.predicted_state.v_ego == expected.v_ego
  assert candidate.predicted_state != held


def test_action_reset_discards_queued_commands_at_regime_boundary():
  delayed_params = PLANT_PARAMS.with_actuation_delay(0.03)
  candidate = InverseEpsActionController(
    action_twin(delayed_params), ACTION_PARAMS,
  )
  candidate._remember_applied_torque(0.1)
  candidate._remember_applied_torque(0.2)
  assert candidate.command_history_count == 2
  candidate.reset()
  assert candidate.command_history_count == 0
  assert candidate.command_history_start == 0


def test_action_tracks_reference_at_physical_effect_time_not_later_scalar_time():
  twin = action_twin()
  state = PlantState(0.0, 0.0, 0.0, 5.0)
  times = np.arange(0.0, 0.55, 0.05, dtype=np.float64)
  curvatures = np.linspace(0.0, 0.02, len(times), dtype=np.float64)
  workspace = CandidateWorkspace()
  workspace.fill(
    twin,
    state,
    ALIGN_INPUTS,
    times,
    curvatures,
    len(times),
    0.5,
    0.0,
  )
  candidate = InverseEpsActionController(
    twin, ACTION_PARAMS, workspace,
  )
  physical_time = 0.05
  scalar_time = 0.20
  expected_physical = candidate._interpolate(
    workspace.desired_angles,
    physical_time / DECISION_DT,
    workspace.decision_count,
  )
  later_scalar_position = candidate._interpolate(
    workspace.desired_angles,
    scalar_time / DECISION_DT,
    workspace.decision_count,
  )
  result = candidate.compute(
    state,
    ALIGN_INPUTS,
    times,
    curvatures,
    len(times),
    0.5,
    physical_time,
    0.0,
    ObserverStatus.ACTIVE,
    workspace_prepared=True,
    action_time=scalar_time,
  )
  assert expected_physical != later_scalar_position
  assert result.desired_angle_deg == expected_physical
  assert result.action_time_seconds == scalar_time
  assert result.prediction_delay_seconds == physical_time


def test_steady_turn_holding_torque_survives_zero_tracking_error():
  twin = action_twin()
  state = feasible_steady_state(twin)
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
  assert result.feedback_torque == 0.0
  assert result.feedforward_torque == state.applied_torque
  assert result.raw_command_torque == state.applied_torque
  assert result.command_torque == state.applied_torque


def test_action_uses_future_torque_only_when_exact_slew_requires_it():
  twin = action_twin()
  state = PlantState(0.0, 0.0, 0.0, 5.0)
  times = np.arange(0.0, 1.05, 0.05, dtype=np.float64)
  base = np.full(len(times), 0.00001, dtype=np.float64)
  mutated = base.copy()
  mutated[times >= 0.40] = 0.15
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
  first_raw = first.raw_command_torque
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
  # Both requests exceed one build step from zero, so their first emitted
  # command is intentionally the same exact actuator limit.
  assert second.command_torque == first_command
  assert second.raw_command_torque != first_raw
  assert second.horizon_assist_active
  # The five-point inverse stencil sees the 0.40 s curvature step beginning at
  # 0.30 s; that derived physical demand is the first binding cell.
  assert second.horizon_demand_time_seconds >= 0.30


def test_future_turn_cannot_move_rack_before_time_aligned_reference():
  twin = action_twin()
  state = PlantState(0.0, 0.0, 0.0, 5.0)
  times = np.arange(0.0, 1.05, 0.05, dtype=np.float64)
  for direction in (-1.0, 1.0):
    curvatures = np.zeros(len(times), dtype=np.float64)
    curvatures[times >= 0.40] = direction * 0.15
    result = InverseEpsActionController(twin, ACTION_PARAMS).compute(
      state,
      ALIGN_INPUTS,
      times,
      curvatures,
      len(times),
      1.0,
      0.05,
      0.0,
      ObserverStatus.ACTIVE,
      action_time=0.20,
    )
    assert result.desired_angle_deg == 0.0
    assert result.command_torque == 0.0
    assert result.raw_command_torque == 0.0
    assert result.no_lead_limited


def test_action_raw_request_reproduces_reported_slew_command_bit_exactly():
  twin = action_twin()
  times = np.arange(0.0, 1.05, 0.05, dtype=np.float64)
  random = np.random.default_rng(219)
  for _ in range(32):
    state = PlantState(
      float(random.uniform(-20.0, 20.0)),
      float(random.uniform(-12.0, 12.0)),
      float(random.uniform(-1.0, 1.0)),
      float(random.uniform(2.0, 25.0)),
    )
    curvatures = np.cumsum(
      random.normal(0.0, 0.0015, len(times)),
    )
    result = InverseEpsActionController(twin, ACTION_PARAMS).compute(
      state,
      ALIGN_INPUTS,
      times,
      curvatures,
      len(times),
      1.0,
      float(random.uniform(0.05, 0.20)),
      0.0,
      ObserverStatus.ACTIVE,
      action_time=float(random.uniform(0.25, 0.60)),
    )
    replayed = twin.apply_slew(
      state.applied_torque, result.raw_command_torque,
    )
    assert struct.pack("<d", replayed) == struct.pack(
      "<d", result.command_torque,
    )


def test_action_linear_residual_is_stateless_across_reversals():
  twin = action_twin()
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
  twin = action_twin()
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
  twin = action_twin()
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
  twin = action_twin()
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
