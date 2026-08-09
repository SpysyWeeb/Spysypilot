from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from opendbc.car.hyundai.steering_request import (
  MAX_ANGLE,
  MAX_ANGLE_FRAMES,
  apply_steering_request_fault_avoidance,
)

from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
)
from openpilot.selfdrive.controls.lib.blatv2 import horizon as horizon_module
from openpilot.selfdrive.controls.lib.blatv2.horizon import (
  CONTROL_DT_SECONDS,
  HORIZON_SAMPLE_COUNT,
  HORIZON_SECONDS,
  IMMEDIATE_SECONDS,
  PREPARATION_SECONDS,
  HorizonController,
  HorizonPolicy,
  HorizonStatus,
  horizon_confidence,
)
from openpilot.selfdrive.controls.lib.blatv2.plant import (
  ComputedTorque,
  RackState,
  TrackingPolicy,
  step_plant,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  DEFAULT_SPEED_NODES_MPS,
  PhysicalParameters,
  ProfileNode,
  VehicleProfile,
)


POLICY_PATH = Path(__file__).parents[1] / "lib" / "blatv2" / "provisional_horizon_policy.json"
REVIEWED_POLICY_SHA256 = "2b95ca8466b898165976af56564fe86b8130e280d22762c67d99e8432de3abe6"


def horizon_policy() -> HorizonPolicy:
  return HorizonPolicy.from_json_file(POLICY_PATH)


def limits() -> RuntimeTorqueLimits:
  return RuntimeTorqueLimits(
    steer_max=409,
    delta_up=4,
    delta_down=7,
    steer_step=1,
    driver_allowance=50,
    driver_multiplier=2,
    driver_factor=1,
    production_envelope_verified=True,
  )


def mapping() -> RackMappingSnapshot:
  return RackMappingSnapshot(
    mass_kg=2100.0,
    wheelbase_m=2.9,
    center_to_front_m=1.2,
    center_to_rear_m=1.7,
    tire_stiffness_front=100000.0,
    tire_stiffness_rear=110000.0,
    steer_ratio_rear=0.0,
    steer_ratio=15.0,
    roll_rad=0.0,
    angle_offset_deg=0.0,
    valid=True,
  )


def profile() -> VehicleProfile:
  parameters = PhysicalParameters(
    torque_per_lateral_accel=0.39298251926074423,
    rack_gain_deg_s2_per_torque=4000.0,
    rack_damping_per_s=10.0,
    transport_delay_s=0.1,
    static_friction_torque=0.1301424652338028,
    kinetic_friction_torque=0.1301424652338028,
    rack_rate_resolution_deg_s=4.0,
    confidence=0.0,
    qualified=False,
  )
  return VehicleProfile(
    vehicle_identity="HYUNDAI_PALISADE",
    revision=0,
    provenance="synthetic provisional horizon test",
    nodes=tuple(
      ProfileNode(
        speed_mps=speed,
        parameters=parameters,
        clean_support_s=0.0,
        sample_count=0,
        cross_fit_route_count=0,
        full_fit_candidate_rms=0.0,
      )
      for speed in DEFAULT_SPEED_NODES_MPS
    ),
  )


def planner(policy: HorizonPolicy | None = None) -> HorizonController:
  return HorizonController(
    fixed_dt_s=CONTROL_DT_SECONDS,
    limits=limits(),
    profile=profile(),
    tracking_policy=TrackingPolicy(10.0, 1.0),
    horizon_policy=horizon_policy() if policy is None else policy,
    nominal_mapping=mapping(),
  )


def trajectory(
  angles_deg: list[float],
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
  if len(angles_deg) != HORIZON_SAMPLE_COUNT:
    raise ValueError("test trajectory must cover exactly two seconds")
  rates = [0.0] * HORIZON_SAMPLE_COUNT
  accelerations = [0.0] * HORIZON_SAMPLE_COUNT
  for index in range(1, HORIZON_SAMPLE_COUNT - 1):
    rates[index] = (angles_deg[index + 1] - angles_deg[index - 1]) / (2.0 * CONTROL_DT_SECONDS)
  rates[0] = (angles_deg[1] - angles_deg[0]) / CONTROL_DT_SECONDS
  rates[-1] = (angles_deg[-1] - angles_deg[-2]) / CONTROL_DT_SECONDS
  for index in range(1, HORIZON_SAMPLE_COUNT - 1):
    accelerations[index] = (rates[index + 1] - rates[index - 1]) / (2.0 * CONTROL_DT_SECONDS)
  accelerations[0] = (rates[1] - rates[0]) / CONTROL_DT_SECONDS
  accelerations[-1] = (rates[-1] - rates[-2]) / CONTROL_DT_SECONDS
  return (
    [0.0] * HORIZON_SAMPLE_COUNT,
    angles_deg,
    rates,
    accelerations,
    [0.0] * HORIZON_SAMPLE_COUNT,
  )


def run(
  controller: HorizonController,
  angles_deg: list[float],
  *,
  initial_angle_deg: float = 0.0,
  previous_counts: int = 0,
  driver_torque: float = 0.0,
  steering_pressed: bool = False,
  lateral_active: bool = True,
  current_steering_angle_deg: float | None = None,
  steering_request_counter: int = 0,
  steering_request_state_valid: bool = True,
  transport_delay_s: float = 0.0,
  committed_command_time_angles_deg: list[float] | None = None,
):
  curvatures, angles, rates, accelerations, speeds = trajectory(angles_deg)
  return controller.update(
    desired_curvatures=curvatures,
    desired_angles_deg=angles,
    desired_rates_deg_s=rates,
    desired_accelerations_deg_s2=accelerations,
    planned_speeds_mps=speeds,
    initial_state=RackState(
      initial_angle_deg,
      0.0,
      previous_counts / limits().steer_max,
    ),
    previous_applied_counts=previous_counts,
    driver_torque=driver_torque,
    steering_pressed=steering_pressed,
    lateral_active=lateral_active,
    current_steering_angle_deg=(initial_angle_deg if current_steering_angle_deg is None else current_steering_angle_deg),
    steering_request_fault_avoidance_counter=steering_request_counter,
    steering_request_state_valid=steering_request_state_valid,
    live_mapping=None,
    lateral_accel_offset_mps2=0.0,
    disturbance_torque=0.0,
    transport_delay_s=transport_delay_s,
    committed_command_time_angles_deg=(committed_command_time_angles_deg),
  )


def ramp(
  *,
  initial: float,
  final: float,
  start_index: int,
  duration: int,
) -> list[float]:
  values = [initial] * HORIZON_SAMPLE_COUNT
  for index in range(start_index, HORIZON_SAMPLE_COUNT):
    fraction = min((index - start_index) / duration, 1.0)
    values[index] = initial + fraction * (final - initial)
  return values


def test_confidence_has_three_continuous_zones() -> None:
  policy = horizon_policy()
  assert horizon_confidence(0.0, policy) == 1.0
  assert horizon_confidence(IMMEDIATE_SECONDS, policy) == 1.0
  assert (
    horizon_confidence(
      PREPARATION_SECONDS,
      policy,
    )
    == policy.preparation_confidence
  )
  assert (
    horizon_confidence(
      HORIZON_SECONDS,
      policy,
    )
    == policy.reserve_confidence
  )
  values = [horizon_confidence(index * CONTROL_DT_SECONDS, policy) for index in range(HORIZON_SAMPLE_COUNT)]
  assert all(left >= right for left, right in zip(values, values[1:], strict=False))


def test_confidence_slack_creates_sign_preserving_reachability_bands() -> None:
  policy = horizon_policy()
  runtime_limits = limits()
  for terminal in (-12.0, 12.0):
    controller = planner(policy)
    result = run(
      controller,
      ramp(initial=0.0, final=terminal, start_index=51, duration=20),
    )
    assert result.valid
    assert (controller.band_lower_counts[0], controller.band_upper_counts[0]) == (0, 0)
    for index in (50, 100, 160):
      requested = min(
        max(int(round(controller.raw_torques[index] * runtime_limits.steer_max)), -runtime_limits.steer_max),
        runtime_limits.steer_max,
      )
      slack = round((1.0 - controller.confidences[index]) * policy.maximum_torque_slack * runtime_limits.steer_max)
      assert requested * terminal > 0.0
      if requested > 0:
        expected = (max(0, requested - slack), requested)
        assert 0 <= controller.band_lower_counts[index] <= controller.band_upper_counts[index]
        reachability_target = expected[0]
      else:
        expected = (requested, min(0, requested + slack))
        assert controller.band_lower_counts[index] <= controller.band_upper_counts[index] <= 0
        reachability_target = expected[1]
      assert (controller.band_lower_counts[index], controller.band_upper_counts[index]) == expected
      assert controller.projector.authored_counts[index] == reachability_target
    assert controller.band_lower_counts[50] == controller.band_upper_counts[50]


def test_confidence_slack_reduces_future_preparation_without_changing_raw_path() -> None:
  policy = horizon_policy()
  with_slack = planner(policy)
  without_slack = planner(replace(policy, maximum_torque_slack=0.0))
  angles = ramp(initial=0.0, final=-12.0, start_index=52, duration=20)

  with_result = run(with_slack, angles)
  without_result = run(without_slack, angles)

  assert with_slack.raw_torques == without_slack.raw_torques
  assert with_slack.reactive_counts == without_slack.reactive_counts
  assert with_slack.reactive_angles_deg == without_slack.reactive_angles_deg
  assert with_slack.reactive_rates_deg_s == without_slack.reactive_rates_deg_s
  assert with_slack.raw_torques[:51] == [0.0] * 51
  assert (with_slack.projector.witness_counts[0], without_slack.projector.witness_counts[0]) == (0, -4)
  assert (with_result.planned_counts, without_result.planned_counts) == (0, -4)
  assert not with_result.preparation_active
  assert without_result.preparation_active
  assert with_result.maximum_authority_required == without_result.maximum_authority_required
  assert not with_result.maximum_authority_active
  assert not without_result.maximum_authority_active


def test_committed_horizon_policy_is_explicit_and_hash_pinned() -> None:
  first = horizon_policy()
  second = horizon_policy()
  assert first == second
  assert first.provisional
  assert first.revision == 0
  assert first.sha256 == REVIEWED_POLICY_SHA256
  assert first.to_json() == second.to_json()


def test_settled_path_holds_zero_without_artificial_motion() -> None:
  controller = planner()
  result = run(controller, [0.0] * HORIZON_SAMPLE_COUNT)
  assert result.valid
  assert result.status == HorizonStatus.OK
  assert result.raw_torque == 0.0
  assert result.planned_counts == 0
  assert not result.preparation_active
  assert not result.preparation_scheduled
  assert controller.reactive_counts == [0] * HORIZON_SAMPLE_COUNT


def test_sharp_turn_preloads_friction_without_early_path() -> None:
  controller = planner()
  start = int(IMMEDIATE_SECONDS / CONTROL_DT_SECONDS)
  result = run(
    controller,
    ramp(initial=0.0, final=8.0, start_index=start, duration=30),
  )
  assert result.valid
  assert result.preparation_scheduled
  assert not result.preparation_active
  assert result.planned_counts == 0
  assert result.reactive_counts == 0
  assert all(angle <= 4.0 * CONTROL_DT_SECONDS + 1e-12 for angle in controller.reactive_angles_deg[:start])
  # Future counts are reachability evidence, not a second command sequence.
  # Only index zero may actuate; a fresh plan is built on the next frame.
  first_preparation = next(
    index
    for index, (witness, reactive) in enumerate(
      zip(
        controller.projector.witness_counts,
        controller.reactive_counts,
        strict=True,
      )
    )
    if witness != reactive
  )
  assert 0 < first_preparation < start
  assert result.maximum_path_lead_deg <= (profile().nodes[0].parameters.rack_rate_resolution_deg_s * CONTROL_DT_SECONDS)


def test_release_and_reversal_do_not_unwind_before_authored_path() -> None:
  start = int(IMMEDIATE_SECONDS / CONTROL_DT_SECONDS)
  for terminal in (0.0, -8.0):
    controller = planner()
    result = run(
      controller,
      ramp(
        initial=8.0,
        final=terminal,
        start_index=start,
        duration=40,
      ),
      initial_angle_deg=8.0,
    )
    assert result.valid
    tolerance = profile().nodes[0].parameters.rack_rate_resolution_deg_s * CONTROL_DT_SECONDS
    assert all(angle >= 8.0 - tolerance - 1e-12 for angle in controller.reactive_angles_deg[:start])


def test_high_angle_request_cut_preserves_counts_and_zeros_rack_torque() -> None:
  controller = planner()
  high_angle = MAX_ANGLE + 1.0
  result = run(
    controller,
    [high_angle] * HORIZON_SAMPLE_COUNT,
    initial_angle_deg=high_angle,
    previous_counts=200,
    steering_request_counter=MAX_ANGLE_FRAMES - 1,
  )

  counter = MAX_ANGLE_FRAMES - 1
  expected_active: list[bool] = []
  expected_counters: list[int] = []
  for angle in controller.reactive_angles_deg[:4]:
    counter, active = apply_steering_request_fault_avoidance(
      angle,
      True,
      counter,
    )
    expected_active.append(active)
    expected_counters.append(counter)
  assert expected_active == [True, False, False, True]
  assert controller.steering_request_active[:4] == expected_active
  assert controller.steering_request_counters[:4] == expected_counters
  assert result.first_request_suppression_index == 1

  suppressed_index = 1
  transmitted_counts = controller.reactive_counts[suppressed_index]
  assert transmitted_counts != 0
  state = RackState(
    controller.reactive_angles_deg[suppressed_index],
    controller.reactive_rates_deg_s[suppressed_index],
    controller.reactive_counts[suppressed_index - 1] / limits().steer_max,
  )
  parameters = controller.profile.parameters_at(0.0).parameters
  zero_step = step_plant(
    state,
    0.0,
    0.0,
    controller.nominal_mapping,
    controller.nominal_mapping,
    0.0,
    parameters,
    0.0,
    CONTROL_DT_SECONDS,
  ).state
  transmitted_step = step_plant(
    state,
    transmitted_counts / limits().steer_max,
    0.0,
    controller.nominal_mapping,
    controller.nominal_mapping,
    0.0,
    parameters,
    0.0,
    CONTROL_DT_SECONDS,
  ).state
  assert controller.reactive_angles_deg[suppressed_index + 1] == zero_step.angle_deg
  assert controller.reactive_rates_deg_s[suppressed_index + 1] == zero_step.rate_deg_s
  assert zero_step != transmitted_step


def test_current_observed_angle_owns_imminent_request_cut() -> None:
  controller = planner()
  result = run(
    controller,
    [0.0] * HORIZON_SAMPLE_COUNT,
    current_steering_angle_deg=MAX_ANGLE + 1.0,
    steering_request_counter=MAX_ANGLE_FRAMES,
  )

  assert result.valid
  assert result.first_request_suppression_index == 0
  assert not controller.steering_request_active[0]
  assert controller.steering_request_active[1]


def test_request_projection_uses_command_time_across_transport_delay() -> None:
  controller = planner()
  transport_delay = 0.1
  committed_angles = [0.0] * HORIZON_SAMPLE_COUNT
  result = run(
    controller,
    [MAX_ANGLE + 1.0] * HORIZON_SAMPLE_COUNT,
    initial_angle_deg=MAX_ANGLE + 1.0,
    current_steering_angle_deg=0.0,
    transport_delay_s=transport_delay,
    committed_command_time_angles_deg=committed_angles,
  )

  counter = 0
  expected_active: list[bool] = []
  expected_counters: list[int] = []
  for index in range(HORIZON_SAMPLE_COUNT):
    command_time_s = index * CONTROL_DT_SECONDS
    if index == 0:
      angle = 0.0
    elif command_time_s <= transport_delay:
      angle = committed_angles[index]
    else:
      effect_index = round(
        (command_time_s - transport_delay) / CONTROL_DT_SECONDS,
      )
      angle = controller.reactive_angles_deg[effect_index]
    counter, active = apply_steering_request_fault_avoidance(
      angle,
      True,
      counter,
    )
    expected_active.append(active)
    expected_counters.append(counter)

  assert result.valid
  assert controller.steering_request_active == expected_active
  assert controller.steering_request_counters == expected_counters
  assert controller.steering_request_counters[1] == 0
  assert controller.reactive_angles_deg[1] > MAX_ANGLE
  assert result.first_request_suppression_index == expected_active.index(False)


def test_request_cut_changes_high_angle_reversal_rollout_not_authored_path() -> None:
  authored = ramp(
    initial=MAX_ANGLE + 1.0,
    final=-(MAX_ANGLE + 1.0),
    start_index=0,
    duration=40,
  )
  original = tuple(authored)
  normal = planner()
  near_cut = planner()

  run(
    normal,
    authored,
    initial_angle_deg=authored[0],
    previous_counts=200,
  )
  cut_result = run(
    near_cut,
    authored,
    initial_angle_deg=authored[0],
    previous_counts=200,
    steering_request_counter=MAX_ANGLE_FRAMES - 1,
  )

  assert tuple(authored) == original
  assert cut_result.first_request_suppression_index == 1
  assert near_cut.reactive_angles_deg[2] != normal.reactive_angles_deg[2]
  assert near_cut.reactive_rates_deg_s[2] != normal.reactive_rates_deg_s[2]


def test_inherited_future_path_lead_is_reported_and_not_extended() -> None:
  controller = planner()
  initial_angle = 7.9
  result = run(
    controller,
    ramp(
      initial=8.0,
      final=-8.0,
      start_index=int(IMMEDIATE_SECONDS / CONTROL_DT_SECONDS),
      duration=40,
    ),
    initial_angle_deg=initial_angle,
  )
  tolerance = profile().nodes[0].parameters.rack_rate_resolution_deg_s * horizon_policy().no_lead_position_tolerance_s
  inherited_lead = 8.0 - initial_angle
  assert result.valid
  assert result.planned_counts == result.reactive_counts
  assert tolerance < result.maximum_path_lead_deg <= inherited_lead
  assert result.path_lead_constrained_samples == 1


def test_maximum_authority_reports_future_need_and_live_activation_separately() -> None:
  controller = planner()
  start = int(IMMEDIATE_SECONDS / CONTROL_DT_SECONDS)
  result = run(
    controller,
    ramp(initial=0.0, final=30.0, start_index=start, duration=20),
  )
  assert result.valid
  assert result.maximum_authority_required
  assert not result.maximum_authority_active

  live = run(
    planner(),
    [10000.0] * HORIZON_SAMPLE_COUNT,
    previous_counts=limits().steer_max - limits().delta_up,
  )
  assert live.valid
  assert live.maximum_authority_required
  assert live.maximum_authority_active
  assert live.planned_counts == limits().steer_max


def test_maximum_authority_still_obeys_no_lead_gate(monkeypatch) -> None:
  monkeypatch.setattr(
    horizon_module,
    "compute_inverse_torque",
    lambda *args, **kwargs: ComputedTorque(
      raw_torque=2.0,
      aligning_torque=0.0,
      friction_torque=0.0,
      motion_feedforward_torque=0.0,
      position_feedback_torque=0.0,
      rate_feedback_torque=0.0,
      disturbance_torque=0.0,
      position_error_deg=0.0,
      rate_error_deg_s=0.0,
      required_acceleration_deg_s2=0.0,
    ),
  )
  controller = planner()
  result = run(
    controller,
    [index * 0.005 for index in range(HORIZON_SAMPLE_COUNT)],
    previous_counts=100,
  )

  tolerance = profile().nodes[0].parameters.rack_rate_resolution_deg_s * horizon_policy().no_lead_position_tolerance_s
  assert result.valid
  assert abs(result.raw_torque) >= 1.0
  assert controller.reactive_counts[0] == 104
  assert result.planned_counts < controller.reactive_counts[0]
  assert result.maximum_path_lead_deg <= tolerance
  assert not result.maximum_authority_active


def test_driver_override_suppresses_future_preparation() -> None:
  for driver_torque in (-51.0, 51.0):
    controller = planner()
    result = run(
      controller,
      ramp(initial=0.0, final=30.0, start_index=100, duration=10),
      driver_torque=driver_torque,
    )
    assert result.valid
    assert result.status == HorizonStatus.DRIVER_OVERRIDE
    assert result.driver_suppressed
    assert not result.future_band_reachable
    assert result.planned_counts == result.reactive_counts


def test_steering_pressed_suppresses_even_with_neutral_torque() -> None:
  controller = planner()
  result = run(
    controller,
    ramp(initial=0.0, final=30.0, start_index=100, duration=10),
    steering_pressed=True,
  )
  assert result.valid
  assert result.status == HorizonStatus.DRIVER_OVERRIDE
  assert result.planned_counts == result.reactive_counts


def test_constructor_rejects_non_control_grid_and_unverified_limits() -> None:
  try:
    HorizonController(
      fixed_dt_s=0.03,
      limits=limits(),
      profile=profile(),
      tracking_policy=TrackingPolicy(10.0, 1.0),
      horizon_policy=horizon_policy(),
      nominal_mapping=mapping(),
    )
  except ValueError:
    pass
  else:
    raise AssertionError("non-control horizon grid was accepted")
  unverified = RuntimeTorqueLimits(
    steer_max=409,
    delta_up=4,
    delta_down=7,
    steer_step=1,
    driver_allowance=50,
    driver_multiplier=2,
    driver_factor=1,
    production_envelope_verified=False,
  )
  try:
    HorizonController(
      fixed_dt_s=CONTROL_DT_SECONDS,
      limits=unverified,
      profile=profile(),
      tracking_policy=TrackingPolicy(10.0, 1.0),
      horizon_policy=horizon_policy(),
      nominal_mapping=mapping(),
    )
  except ValueError:
    pass
  else:
    raise AssertionError("unverified production envelope was accepted")


def test_invalid_inputs_fail_without_stale_command() -> None:
  for invalid in (None, math.nan, math.inf, -math.inf, 1e308, "bad"):
    controller = planner()
    valid = run(controller, [0.0] * HORIZON_SAMPLE_COUNT)
    assert valid.valid
    curvatures, angles, rates, accelerations, speeds = trajectory(
      [0.0] * HORIZON_SAMPLE_COUNT,
    )
    angles[100] = invalid  # type: ignore[assignment]
    result = controller.update(
      desired_curvatures=curvatures,
      desired_angles_deg=angles,
      desired_rates_deg_s=rates,
      desired_accelerations_deg_s2=accelerations,
      planned_speeds_mps=speeds,
      initial_state=RackState(0.0, 0.0, 0.0),
      previous_applied_counts=0,
      driver_torque=0.0,
      steering_pressed=False,
      lateral_active=True,
      current_steering_angle_deg=0.0,
      steering_request_fault_avoidance_counter=0,
      steering_request_state_valid=True,
      live_mapping=None,
      lateral_accel_offset_mps2=0.0,
      disturbance_torque=0.0,
    )
    assert not result.valid
    assert result.status == HorizonStatus.INVALID_INPUT
    assert result.planned_counts == 0
    assert controller.reactive_counts == [0] * HORIZON_SAMPLE_COUNT

  class BrokenSequence:
    def __len__(self) -> int:
      raise RuntimeError("broken sequence")

  controller = planner()
  _, angles, rates, accelerations, speeds = trajectory([0.0] * HORIZON_SAMPLE_COUNT)
  result = controller.update(
    desired_curvatures=BrokenSequence(),  # type: ignore[arg-type]
    desired_angles_deg=angles,
    desired_rates_deg_s=rates,
    desired_accelerations_deg_s2=accelerations,
    planned_speeds_mps=speeds,
    initial_state=RackState(0.0, 0.0, 0.0),
    previous_applied_counts=0,
    driver_torque=0.0,
    steering_pressed=False,
    lateral_active=True,
    current_steering_angle_deg=0.0,
    steering_request_fault_avoidance_counter=0,
    steering_request_state_valid=True,
    live_mapping=None,
    lateral_accel_offset_mps2=0.0,
    disturbance_torque=0.0,
  )
  assert not result.valid
  assert result.status == HorizonStatus.INVALID_INPUT

  for counter, valid in ((0, False), (-1, True), (MAX_ANGLE_FRAMES + 2, True)):
    result = run(
      controller,
      [0.0] * HORIZON_SAMPLE_COUNT,
      steering_request_counter=counter,
      steering_request_state_valid=valid,
    )
    assert not result.valid
    assert result.status == HorizonStatus.INVALID_INPUT
    assert result.planned_counts == 0


def test_input_sequences_must_cover_exactly_two_seconds() -> None:
  controller = planner()
  curvatures, angles, rates, accelerations, speeds = trajectory(
    [0.0] * HORIZON_SAMPLE_COUNT,
  )
  result = controller.update(
    desired_curvatures=curvatures[:-1],
    desired_angles_deg=angles,
    desired_rates_deg_s=rates,
    desired_accelerations_deg_s2=accelerations,
    planned_speeds_mps=speeds,
    initial_state=RackState(0.0, 0.0, 0.0),
    previous_applied_counts=0,
    driver_torque=0.0,
    steering_pressed=False,
    lateral_active=True,
    current_steering_angle_deg=0.0,
    steering_request_fault_avoidance_counter=0,
    steering_request_state_valid=True,
    live_mapping=None,
    lateral_accel_offset_mps2=0.0,
    disturbance_torque=0.0,
  )
  assert not result.valid
  assert result.status == HorizonStatus.INVALID_INPUT


def test_repeated_plans_are_byte_stable() -> None:
  controller = planner()
  angles = ramp(initial=0.0, final=-12.0, start_index=70, duration=50)
  snapshots = []
  for _ in range(3):
    result = run(controller, angles)
    snapshots.append(
      (
        result.snapshot(),
        tuple(controller.raw_torques),
        tuple(controller.band_lower_counts),
        tuple(controller.band_upper_counts),
        tuple(controller.reactive_counts),
        tuple(controller.reactive_angles_deg),
        tuple(controller.reactive_rates_deg_s),
      )
    )
  assert snapshots[0] == snapshots[1] == snapshots[2]
