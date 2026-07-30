from __future__ import annotations

import inspect
import math
from pathlib import Path
import unittest

from openpilot.selfdrive.controls.lib.blatv2.core import (
  CoreResult,
  CoreStatus,
  ModularControllerCore,
)
from openpilot.selfdrive.controls.lib.blatv2.intent import (
  INTENT_CAPACITY,
  adapt_model_intent_into,
)
from openpilot.selfdrive.controls.lib.blatv2.observer import (
  ObserverPolicy,
  ObserverStatus,
)
from openpilot.selfdrive.controls.lib.blatv2.plant import (
  RackState,
  TrackingPolicy,
  compute_inverse_torque,
  step_plant,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
  map_reference,
)
from openpilot.selfdrive.controls.lib.blatv2.reference import (
  sample_reference,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  DEFAULT_SPEED_NODES_MPS,
  PhysicalParameters,
  ProfileNode,
  VehicleProfile,
)


DT = 0.01


def mapping(*, valid: bool = True) -> RackMappingSnapshot:
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
    valid=valid,
  )


def vehicle_profile(
  *,
  qualified: bool = True,
  delays: tuple[float, ...] | None = None,
  torque_gains: tuple[float, ...] | None = None,
) -> VehicleProfile:
  selected_delays = (
    (0.0,) * len(DEFAULT_SPEED_NODES_MPS)
    if delays is None
    else delays
  )
  selected_gains = (
    (0.30,) * len(DEFAULT_SPEED_NODES_MPS)
    if torque_gains is None
    else torque_gains
  )
  if (
    len(selected_delays) != len(DEFAULT_SPEED_NODES_MPS)
    or len(selected_gains) != len(DEFAULT_SPEED_NODES_MPS)
  ):
    raise ValueError("test profile fields must match the speed nodes")
  nodes = []
  for index, speed in enumerate(DEFAULT_SPEED_NODES_MPS):
    params = PhysicalParameters(
      torque_per_lateral_accel=selected_gains[index],
      rack_gain_deg_s2_per_torque=1500.0,
      rack_damping_per_s=8.0,
      transport_delay_s=selected_delays[index],
      static_friction_torque=0.09,
      kinetic_friction_torque=0.03,
      rack_rate_resolution_deg_s=4.0,
      confidence=1.0 if qualified else 0.0,
      qualified=qualified,
    )
    nodes.append(ProfileNode(
      speed_mps=speed,
      parameters=params,
      clean_support_s=500.0 if qualified else 0.0,
      sample_count=50000 if qualified else 0,
      validation_count=25000 if qualified else 0,
      validation_rms=0.01 if qualified else 0.0,
    ))
  return VehicleProfile(
    vehicle_identity="synthetic-vehicle",
    revision=1 if qualified else 0,
    provenance="synthetic core test",
    nodes=tuple(nodes),
  )


def plan_values(
  *,
  planned_speed: float,
  constant_curvature: float | None = None,
  constant_speed: bool = False,
) -> tuple[list[float], list[float], list[float], list[float]]:
  times = [index * 0.05 for index in range(21)]
  speeds = (
    [planned_speed] * len(times)
    if constant_speed
    else [
      planned_speed + 0.25 * time + 0.03 * time * time
      for time in times
    ]
  )
  if constant_curvature is None:
    curvatures = [
      0.002 + 0.009 * time + 0.003 * time * time
      for time in times
    ]
  else:
    curvatures = [constant_curvature] * len(times)
  rates = [
    curvature * speed
    for curvature, speed in zip(curvatures, speeds, strict=True)
  ]
  return times, rates, speeds, curvatures


def adapted_intent(
  profile: VehicleProfile,
  *,
  current_speed: float = 10.0,
  scalar_curvature: float = 0.012,
  action_time_s: float = 0.25,
  plan_time_now_s: float = 0.20,
  state_age_s: float = 0.0,
  constant_curvature: float | None = None,
  constant_speed: bool = False,
  malformed_future: bool = False,
  physical_delay_override: float | None = None,
  message_valid: bool = True,
):
  times, rates, speeds, curvatures = plan_values(
    planned_speed=max(current_speed, 1.0),
    constant_curvature=constant_curvature,
    constant_speed=constant_speed,
  )
  native_speeds = list(speeds)
  if malformed_future:
    native_speeds[4] = 0.0
  outputs = tuple([0.0] * INTENT_CAPACITY for _ in range(4))
  delay = (
    profile.parameters_at(current_speed).parameters.transport_delay_s
    if physical_delay_override is None
    else physical_delay_override
  )
  origin_ns = 10_000_000_000
  control_ns = origin_ns + round(plan_time_now_s * 1e9)
  state_sample_ns = control_ns - round(state_age_s * 1e9)
  publication_ns = control_ns - 10_000_000
  adaptation = adapt_model_intent_into(
    state_sample_mono_ns=state_sample_ns,
    control_witness_mono_ns=control_ns,
    model_publication_mono_ns=publication_ns,
    plan_origin_mono_ns=origin_ns,
    model_frame_id=42,
    message_valid=message_valid,
    message_alive=True,
    scalar_desired_curvature=scalar_curvature,
    published_desired_curvature_time_s=action_time_s,
    native_plan_times_s=times,
    native_orientation_rates_z=rates,
    native_velocities_x=native_speeds,
    current_v_ego_m_s=current_speed,
    physical_transport_delay_s=delay,
    output_plan_times_s=outputs[0],
    output_orientation_rates_z=outputs[1],
    output_velocities_x=outputs[2],
    output_plan_curvatures=outputs[3],
  )
  return adaptation, outputs, (times, rates, native_speeds, curvatures)


def make_core(
  profile: VehicleProfile,
  *,
  observer_policy: ObserverPolicy | None = None,
) -> ModularControllerCore:
  return ModularControllerCore(
    fixed_dt_s=DT,
    profile=profile,
    tracking_policy=TrackingPolicy(6.0),
    observer_policy=observer_policy,
    nominal_mapping=mapping(),
    plan_capacity=INTENT_CAPACITY,
  )


def update_core(
  core: ModularControllerCore,
  adaptation,
  outputs,
  *,
  scalar_curvature: float = 0.012,
  current_speed: float = 10.0,
  measured_angle: float = 0.0,
  measured_rate: float = 0.0,
  measured_acceleration: float = 0.0,
  applied_torque: float = 0.0,
  live: RackMappingSnapshot | None = None,
  lateral_active: bool = True,
  lateral_valid: bool = True,
  engagement_boundary: bool = False,
  live_parameters_valid: bool = True,
  steering_pressed: bool = False,
  actuator_constrained: bool = False,
  output_constrained: bool = False,
  standstill: bool = False,
):
  return core.update(
    frame=adaptation.frame,
    intent_status=adaptation.status,
    intent_plan_times_s=outputs[0],
    intent_orientation_rates_z=outputs[1],
    intent_velocities_x=outputs[2],
    scalar_curvature=scalar_curvature,
    current_v_ego_m_s=current_speed,
    measured_rack_angle_deg=measured_angle,
    measured_rack_rate_deg_s=measured_rate,
    measured_rack_acceleration_deg_s2=measured_acceleration,
    recorded_applied_torque=applied_torque,
    lateral_accel_offset=0.0,
    live_mapping=mapping() if live is None else live,
    lateral_active=lateral_active,
    lateral_valid=lateral_valid,
    engagement_boundary=engagement_boundary,
    live_parameters_valid=live_parameters_valid,
    steering_pressed=steering_pressed,
    actuator_constrained=actuator_constrained,
    output_constrained=output_constrained,
    standstill=standstill,
  )


class TestBLaTv2Core(unittest.TestCase):
  def test_reference_query_before_at_and_after_scalar_action_is_anchored(self) -> None:
    profile = vehicle_profile()
    scalar = 0.012
    action = 0.25
    for query_time in (0.15, action, 0.35):
      with self.subTest(query_time=query_time):
        adaptation, outputs, native = adapted_intent(
          profile,
          scalar_curvature=scalar,
          action_time_s=action,
          plan_time_now_s=query_time,
        )
        result = update_core(
          make_core(profile),
          adaptation,
          outputs,
          scalar_curvature=scalar,
        )
        expected_query_time = adaptation.status.plan_time_now_s
        expected = sample_reference(
          native[0],
          native[1],
          native[2],
          scalar,
          action,
          expected_query_time,
          10.0,
          (expected_query_time,),
        )
        self.assertTrue(result.valid)
        self.assertEqual(
          result.physical_effect_plan_s,
          expected_query_time,
        )
        self.assertEqual(
          result.desired_curvature,
          expected.curvatures[0],
        )
        if query_time == action:
          self.assertEqual(result.desired_curvature, scalar)
        elif query_time < action:
          self.assertLess(result.desired_curvature, scalar)
        else:
          self.assertGreater(result.desired_curvature, scalar)

  def test_effect_time_is_owned_by_profile_delay_without_action_lead(self) -> None:
    profile = vehicle_profile(delays=(0.12,) * 6)
    adaptation, outputs, _ = adapted_intent(profile)
    core = make_core(profile)
    result = None
    for _ in range(12):
      result = update_core(core, adaptation, outputs)
    self.assertIsNotNone(result)
    self.assertTrue(result.valid)
    expected_effect = adaptation.status.plan_time_now_s + 0.12
    self.assertEqual(result.physical_effect_plan_s, expected_effect)
    self.assertEqual(
      result.physical_effect_plan_s,
      adaptation.status.physical_effect_plan_s,
    )
    self.assertNotEqual(
      result.physical_effect_plan_s,
      adaptation.frame.timing.scalar_action_plan_s,
    )
    self.assertGreater(
      result.physical_effect_plan_s,
      adaptation.frame.timing.scalar_action_plan_s,
    )

  def test_state_age_extends_plant_horizon_not_reference_query_time(self) -> None:
    profile = vehicle_profile(delays=(0.02,) * 6)
    current_adaptation, current_outputs, _ = adapted_intent(
      profile,
      state_age_s=0.0,
    )
    aged_adaptation, aged_outputs, _ = adapted_intent(
      profile,
      state_age_s=0.005,
    )
    current_core = make_core(profile)
    aged_core = make_core(profile)
    current = aged = None
    for torque in (0.1, 0.2, 0.3):
      current = update_core(
        current_core,
        current_adaptation,
        current_outputs,
        applied_torque=torque,
      )
      aged = update_core(
        aged_core,
        aged_adaptation,
        aged_outputs,
        applied_torque=torque,
      )
    self.assertIsNotNone(current)
    self.assertIsNotNone(aged)
    self.assertTrue(current.valid)
    self.assertTrue(aged.valid)
    self.assertEqual(current.state_age_s, 0.0)
    self.assertEqual(current.total_prediction_horizon_s, 0.02)
    self.assertEqual(aged.state_age_s, 0.005)
    self.assertEqual(aged.total_prediction_horizon_s, 0.025)
    self.assertEqual(
      current.physical_effect_plan_s,
      aged.physical_effect_plan_s,
    )
    self.assertEqual(current.prediction_history_count, 2)
    self.assertEqual(aged.prediction_history_count, 3)
    self.assertAlmostEqual(aged.prediction_fractional_dt_s, 0.005)

  def test_scalar_only_fallback_is_finite_and_not_stale(self) -> None:
    profile = vehicle_profile()
    adaptation, outputs, _ = adapted_intent(
      profile, malformed_future=True,
    )
    self.assertTrue(adaptation.status.scalar_only)
    core = make_core(profile)
    result = update_core(core, adaptation, outputs)
    self.assertTrue(result.valid)
    self.assertEqual(result.status, CoreStatus.DEGRADED_SCALAR_ONLY)
    self.assertTrue(result.reference_scalar_only)
    self.assertEqual(result.desired_curvature, 0.012)
    self.assertEqual(result.desired_curvature_rate, 0.0)
    self.assertEqual(result.desired_curvature_acceleration, 0.0)
    self.assertTrue(math.isfinite(result.raw_torque))

  def test_existing_reference_mapper_and_inverse_reach_core_unchanged(self) -> None:
    profile = vehicle_profile()
    adaptation, outputs, native = adapted_intent(profile)
    core = make_core(profile)
    result = update_core(
      core,
      adaptation,
      outputs,
      measured_angle=2.0,
      measured_rate=1.5,
      applied_torque=0.1,
    )
    self.assertTrue(result.valid)

    query = sample_reference(
      native[0],
      native[1],
      native[2],
      0.012,
      adaptation.frame.timing.scalar_action_plan_s,
      adaptation.status.plan_time_now_s,
      10.0,
      (adaptation.status.plan_time_now_s,),
    )
    mapped = map_reference(
      query.curvatures[0],
      query.curvature_rates[0],
      query.curvature_accelerations[0],
      query.planned_speeds[0],
      query.planned_speed_rates[0],
      query.planned_speed_accelerations[0],
      mapping(),
      mapping(),
    )
    params = profile.parameters_at(query.planned_speeds[0]).parameters
    expected = compute_inverse_torque(
      RackState(2.0, 1.5, 0.1),
      query.curvatures[0],
      mapped.angle_deg,
      mapped.rate_deg_s,
      mapped.acceleration_deg_s2,
      query.planned_speeds[0],
      0.0,
      0.0,
      params,
      TrackingPolicy(6.0),
      0.0,
    )
    self.assertEqual(result.desired_curvature, query.curvatures[0])
    self.assertEqual(result.desired_curvature_rate, query.curvature_rates[0])
    self.assertEqual(
      result.desired_curvature_acceleration,
      query.curvature_accelerations[0],
    )
    self.assertEqual(result.desired_angle_deg, mapped.angle_deg)
    self.assertEqual(result.desired_rate_deg_s, mapped.rate_deg_s)
    self.assertEqual(
      result.desired_acceleration_deg_s2,
      mapped.acceleration_deg_s2,
    )
    self.assertEqual(result.raw_torque, expected.raw_torque)
    self.assertEqual(
      result.motion_feedforward_torque,
      expected.motion_feedforward_torque,
    )
    self.assertEqual(
      result.position_feedback_torque,
      expected.position_feedback_torque,
    )
    self.assertEqual(
      result.rate_feedback_torque,
      expected.rate_feedback_torque,
    )

  def test_current_raw_request_cannot_affect_own_prediction(self) -> None:
    profile = vehicle_profile(delays=(0.02,) * 6)
    first_core = make_core(profile)
    second_core = make_core(profile)
    common_adaptation, common_outputs, _ = adapted_intent(
      profile, scalar_curvature=0.0,
    )
    for core in (first_core, second_core):
      update_core(
        core,
        common_adaptation,
        common_outputs,
        scalar_curvature=0.0,
        applied_torque=0.10,
      )

    left_adaptation, left_outputs, _ = adapted_intent(
      profile, scalar_curvature=0.03,
    )
    right_adaptation, right_outputs, _ = adapted_intent(
      profile, scalar_curvature=-0.03,
    )
    left = update_core(
      first_core,
      left_adaptation,
      left_outputs,
      scalar_curvature=0.03,
      applied_torque=0.20,
    )
    right = update_core(
      second_core,
      right_adaptation,
      right_outputs,
      scalar_curvature=-0.03,
      applied_torque=0.20,
    )
    self.assertTrue(left.valid)
    self.assertTrue(right.valid)
    self.assertEqual(left.predicted_angle_deg, right.predicted_angle_deg)
    self.assertEqual(left.predicted_rate_deg_s, right.predicted_rate_deg_s)
    self.assertEqual(
      left.prediction_last_applied_torque,
      right.prediction_last_applied_torque,
    )
    self.assertNotEqual(left.raw_torque, right.raw_torque)
    self.assertNotIn(
      "raw_torque",
      inspect.signature(first_core.update).parameters,
    )

  def test_variable_delay_history_is_oldest_to_newest(self) -> None:
    profile = vehicle_profile(
      delays=(0.02, 0.025, 0.03, 0.035, 0.04, 0.045),
    )
    core = make_core(profile)
    high_adaptation, high_outputs, _ = adapted_intent(
      profile, current_speed=30.0,
    )
    result = None
    for torque in (0.10, 0.20, 0.30, 0.40, 0.50):
      result = update_core(
        core,
        high_adaptation,
        high_outputs,
        current_speed=30.0,
        applied_torque=torque,
      )
    self.assertIsNotNone(result)
    self.assertTrue(result.valid)
    self.assertEqual(core.history_capacity, INTENT_CAPACITY)
    self.assertEqual(result.prediction_history_count, 5)
    self.assertAlmostEqual(result.prediction_fractional_dt_s, 0.005)
    self.assertEqual(result.prediction_first_applied_torque, 0.10)
    self.assertEqual(result.prediction_last_applied_torque, 0.50)

    low_adaptation, low_outputs, _ = adapted_intent(
      profile, current_speed=0.0,
    )
    result = update_core(
      core,
      low_adaptation,
      low_outputs,
      current_speed=0.0,
      applied_torque=0.60,
    )
    self.assertTrue(result.valid)
    self.assertEqual(result.prediction_history_count, 2)
    self.assertEqual(result.prediction_fractional_dt_s, 0.0)
    self.assertEqual(result.prediction_first_applied_torque, 0.50)
    self.assertEqual(result.prediction_last_applied_torque, 0.60)

  def test_profile_interpolation_is_continuous_at_node(self) -> None:
    gains = (0.6, 0.5, 0.4, 0.35, 0.3, 0.28)
    profile = vehicle_profile(torque_gains=gains)
    results = []
    for speed in (5.0 - 1e-7, 5.0, 5.0 + 1e-7):
      adaptation, outputs, _ = adapted_intent(
        profile,
        current_speed=speed,
        constant_curvature=0.01,
      )
      core = make_core(profile)
      results.append(update_core(
        core,
        adaptation,
        outputs,
        current_speed=speed,
        scalar_curvature=0.01,
      ).snapshot())
    gain_index = CoreResult.__slots__.index(
      "torque_per_lateral_accel",
    )
    raw_index = CoreResult.__slots__.index("raw_torque")
    self.assertAlmostEqual(results[0][gain_index], results[1][gain_index])
    self.assertAlmostEqual(results[1][gain_index], results[2][gain_index])
    self.assertLess(abs(results[0][raw_index] - results[1][raw_index]), 1e-6)
    self.assertLess(abs(results[1][raw_index] - results[2][raw_index]), 1e-6)

  def test_observer_uses_recorded_response_and_obeys_lifecycle(self) -> None:
    profile = vehicle_profile()
    core = make_core(
      profile,
      observer_policy=ObserverPolicy(0.01, 0.5),
    )
    adaptation, outputs, _ = adapted_intent(
      profile,
      current_speed=0.0,
      scalar_curvature=0.0,
      constant_curvature=0.0,
    )
    params = profile.parameters_at(0.0).parameters
    rack_rate = 10.0
    applied = 0.20
    friction = params.kinetic_friction_torque
    disturbance = 0.10
    acceleration = (
      params.rack_gain_deg_s2_per_torque
      * (applied - friction - disturbance)
      - params.rack_damping_per_s * rack_rate
    )
    active = update_core(
      core,
      adaptation,
      outputs,
      scalar_curvature=0.0,
      current_speed=0.0,
      measured_rate=rack_rate,
      measured_acceleration=acceleration,
      applied_torque=applied,
    )
    self.assertEqual(active.observer_status, int(ObserverStatus.ACTIVE))
    self.assertAlmostEqual(
      active.observer_instantaneous_disturbance_torque,
      disturbance,
      places=14,
    )
    estimate = active.observer_estimated_disturbance_torque
    self.assertEqual(active.disturbance_torque, estimate)

    frozen = update_core(
      core,
      adaptation,
      outputs,
      scalar_curvature=0.0,
      current_speed=0.0,
      measured_rate=rack_rate,
      measured_acceleration=-900.0,
      applied_torque=-0.8,
      steering_pressed=True,
    )
    self.assertEqual(
      frozen.observer_status,
      int(ObserverStatus.FROZEN_STEERING_PRESSED),
    )
    self.assertEqual(
      frozen.observer_estimated_disturbance_torque, estimate,
    )

    reset = update_core(
      core,
      adaptation,
      outputs,
      scalar_curvature=0.0,
      current_speed=0.0,
      measured_rate=rack_rate,
      measured_acceleration=acceleration,
      applied_torque=applied,
      engagement_boundary=True,
    )
    self.assertEqual(
      reset.observer_status,
      int(ObserverStatus.RESET_ENGAGEMENT_BOUNDARY),
    )
    self.assertEqual(reset.observer_estimated_disturbance_torque, 0.0)

  def test_unqualified_profile_is_shadow_visible_not_activation_claim(self) -> None:
    profile = vehicle_profile(qualified=False)
    core = make_core(
      profile,
      observer_policy=ObserverPolicy(0.1, 0.5),
    )
    adaptation, outputs, _ = adapted_intent(profile)
    result = update_core(core, adaptation, outputs)
    self.assertTrue(result.valid)
    self.assertFalse(result.profile_qualified)
    self.assertEqual(
      result.status, CoreStatus.SHADOW_UNQUALIFIED_PROFILE,
    )
    self.assertEqual(
      result.observer_status,
      int(ObserverStatus.DISABLED_UNQUALIFIED_PROFILE),
    )
    self.assertFalse(hasattr(result, "field_eligible"))

  def test_invalid_runtime_input_is_finite_and_clears_prior_output(self) -> None:
    profile = vehicle_profile()
    core = make_core(profile)
    adaptation, outputs, _ = adapted_intent(profile)
    valid = update_core(core, adaptation, outputs)
    self.assertTrue(valid.valid)
    self.assertNotEqual(valid.raw_torque, 0.0)
    invalid = update_core(
      core,
      adaptation,
      outputs,
      measured_acceleration=math.nan,
    )
    self.assertIs(invalid, valid)
    self.assertFalse(invalid.valid)
    self.assertEqual(invalid.status, CoreStatus.INVALID_MEASUREMENT)
    self.assertEqual(invalid.raw_torque, 0.0)
    self.assertTrue(
      all(
        math.isfinite(value)
        for value in invalid.snapshot()
        if isinstance(value, float)
      ),
    )

    invalid_adaptation, invalid_outputs, _ = adapted_intent(
      profile, message_valid=False,
    )
    invalid_intent = update_core(
      core, invalid_adaptation, invalid_outputs,
    )
    self.assertFalse(invalid_intent.valid)
    self.assertEqual(invalid_intent.status, CoreStatus.INVALID_INTENT)
    self.assertEqual(invalid_intent.raw_torque, 0.0)

  def test_transport_clock_disagreement_is_legible(self) -> None:
    profile = vehicle_profile(delays=(0.12,) * 6)
    adaptation, outputs, _ = adapted_intent(
      profile, physical_delay_override=0.10,
    )
    core = make_core(profile)
    result = update_core(core, adaptation, outputs)
    self.assertFalse(result.valid)
    self.assertEqual(
      result.status, CoreStatus.TRANSPORT_TIME_MISMATCH,
    )
    self.assertEqual(result.raw_torque, 0.0)

  def test_repeated_trace_snapshots_are_byte_stable(self) -> None:
    profile = vehicle_profile()

    def replay() -> bytes:
      core = make_core(
        profile,
        observer_policy=ObserverPolicy(0.08, 0.5),
      )
      encoded = bytearray()
      for index in range(100):
        scalar = 0.01 * math.sin(index * 0.07)
        adaptation, outputs, _ = adapted_intent(
          profile, scalar_curvature=scalar,
        )
        result = update_core(
          core,
          adaptation,
          outputs,
          scalar_curvature=scalar,
          measured_rate=3.0 * math.sin(index * 0.11),
          measured_acceleration=20.0 * math.cos(index * 0.09),
          applied_torque=0.1 * math.sin(index * 0.05),
        )
        encoded.extend(repr(result.snapshot()).encode("ascii"))
        encoded.append(10)
      return bytes(encoded)

    expected = replay()
    for _ in range(3):
      self.assertEqual(replay(), expected)

  def test_forward_inverse_reciprocity_on_reference(self) -> None:
    profile = vehicle_profile()
    scalar = 0.01
    speed = 10.0
    desired = map_reference(
      scalar, 0.0, 0.0, speed, 0.0, 0.0, mapping(), mapping(),
    )
    adaptation, outputs, _ = adapted_intent(
      profile,
      current_speed=speed,
      scalar_curvature=scalar,
      constant_curvature=scalar,
      constant_speed=True,
    )
    core = make_core(profile)
    result = update_core(
      core,
      adaptation,
      outputs,
      scalar_curvature=scalar,
      current_speed=speed,
      measured_angle=desired.angle_deg,
      measured_rate=0.0,
      measured_acceleration=0.0,
      applied_torque=0.0,
    )
    self.assertTrue(result.valid)
    self.assertEqual(result.position_error_deg, 0.0)
    self.assertEqual(result.rate_error_deg_s, 0.0)
    state = RackState(
      desired.angle_deg, 0.0, result.raw_torque,
    )
    stepped = step_plant(
      state,
      result.raw_torque,
      speed,
      mapping(),
      mapping(),
      0.0,
      profile.parameters_at(speed).parameters,
      0.0,
      DT,
    )
    self.assertEqual(stepped.state.angle_deg, state.angle_deg)
    self.assertEqual(stepped.state.rate_deg_s, 0.0)

  def test_result_is_reused_and_forbidden_control_clocks_are_absent(self) -> None:
    profile = vehicle_profile()
    adaptation, outputs, _ = adapted_intent(profile)
    core = make_core(profile)
    first = update_core(core, adaptation, outputs)
    second = update_core(core, adaptation, outputs)
    self.assertIs(first, second)
    source = Path(inspect.getfile(ModularControllerCore)).read_text()
    for forbidden in (
      "DT_MDL",
      "LAT_SMOOTH_SECONDS",
      "liveDelay",
      "BLaTv1",
      "v14",
    ):
      self.assertNotIn(forbidden, source)
    signature = inspect.signature(ModularControllerCore)
    self.assertIs(signature.parameters["tracking_policy"].default, inspect.Parameter.empty)
    self.assertIs(signature.parameters["observer_policy"].default, inspect.Parameter.empty)


if __name__ == "__main__":
  unittest.main()
