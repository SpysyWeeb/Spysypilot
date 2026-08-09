from __future__ import annotations

from dataclasses import replace
import inspect
import math
from pathlib import Path
import unittest

from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
  apply_torque_envelope_counts,
)
from openpilot.selfdrive.controls.lib.blatv2.bootstrap import (
  ControllerSelection,
  EngagementDecision,
  profile_sha256,
)
from openpilot.selfdrive.controls.lib.blatv2.controller import (
  CandidateStatus,
  ModularControllerCandidate,
)
from openpilot.selfdrive.controls.lib.blatv2.core import (
  ModularControllerCore,
)
from openpilot.selfdrive.controls.lib.blatv2.horizon import HorizonPolicy
from openpilot.selfdrive.controls.lib.blatv2.intent import (
  INTENT_CAPACITY,
  adapt_model_intent_into,
)
from openpilot.selfdrive.controls.lib.blatv2.live_safety import (
  LiveSafetyState,
  RECOVERY_OK_FRAMES,
)
from openpilot.selfdrive.controls.lib.blatv2.plant import TrackingPolicy
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  PhysicalParameters,
  ProfileNode,
  VehicleProfile,
)


LIMITS = RuntimeTorqueLimits(
  409,
  4,
  7,
  1,
  50,
  2,
  1,
  production_envelope_verified=True,
)
DT = 0.01
POLICY_PATH = Path(__file__).parents[1] / "lib" / "blatv2" / "provisional_horizon_policy.json"
_TEST_CASE = unittest.TestCase()


def mapping(
  *,
  valid: bool = True,
  vehicle_identity_offset: float = 0.0,
) -> RackMappingSnapshot:
  return RackMappingSnapshot(
    mass_kg=2100.0 + vehicle_identity_offset,
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


def profile(
  *,
  qualified: bool = True,
  revision: int = 7,
  identity: str = "candidate-car",
  transport_delay_s: float = 0.0,
) -> VehicleProfile:
  parameters = PhysicalParameters(
    torque_per_lateral_accel=0.30,
    rack_gain_deg_s2_per_torque=1500.0,
    rack_damping_per_s=8.0,
    transport_delay_s=transport_delay_s,
    static_friction_torque=0.09,
    kinetic_friction_torque=0.03,
    rack_rate_resolution_deg_s=4.0,
    confidence=1.0 if qualified else 0.0,
    qualified=qualified,
  )
  return VehicleProfile(
    vehicle_identity=identity,
    revision=revision,
    provenance="controller facade test",
    nodes=(
      ProfileNode(
        0.0,
        parameters,
        500.0 if qualified else 0.0,
        50000 if qualified else 0,
        25000 if qualified else 0,
        0.01 if qualified else 0.0,
      ),
      ProfileNode(
        30.0,
        parameters,
        500.0 if qualified else 0.0,
        50000 if qualified else 0,
        25000 if qualified else 0,
        0.01 if qualified else 0.0,
      ),
    ),
  )


def candidate(
  selected_profile: VehicleProfile,
  *,
  limits: RuntimeTorqueLimits = LIMITS,
) -> ModularControllerCandidate:
  core = ModularControllerCore(
    fixed_dt_s=DT,
    profile=selected_profile,
    tracking_policy=TrackingPolicy(6.0),
    observer_policy=None,
    nominal_mapping=mapping(),
    runtime_limits=limits,
    horizon_policy=HorizonPolicy.from_json_file(POLICY_PATH),
    plan_capacity=INTENT_CAPACITY,
  )
  return ModularControllerCandidate(
    core=core,
  )


def decision(
  selected_profile: VehicleProfile | None,
  *,
  selection: ControllerSelection | None = None,
  selected_hash: str | None = None,
  provisional: bool = True,
) -> EngagementDecision:
  actual_selection = (
    ControllerSelection.STOCK
    if selected_profile is None
    else ControllerSelection.MODULAR
  ) if selection is None else selection
  return EngagementDecision(
    selection=actual_selection,
    profile=selected_profile,
    profile_sha256=(
      ""
      if selected_profile is None
      else profile_sha256(selected_profile)
    ) if selected_hash is None else selected_hash,
    provisional=(
      False if actual_selection == ControllerSelection.STOCK else provisional
    ),
  )


def intent(
  selected_profile: VehicleProfile,
  *,
  scalar: float = 0.012,
  malformed_future: bool = False,
):
  times = [index * 0.1 for index in range(INTENT_CAPACITY)]
  speeds = [10.0] * len(times)
  curvatures = [
    scalar if malformed_future else scalar + 0.002 * time
    for time in times
  ]
  rates = [
    curvature * speed
    for curvature, speed in zip(curvatures, speeds, strict=True)
  ]
  if malformed_future:
    speeds[4] = 0.0
  outputs = tuple([0.0] * INTENT_CAPACITY for _ in range(4))
  adaptation = adapt_model_intent_into(
    state_sample_mono_ns=10_200_000_000,
    control_witness_mono_ns=10_200_000_000,
    model_publication_mono_ns=10_190_000_000,
    plan_origin_mono_ns=10_000_000_000,
    model_frame_id=42,
    message_valid=True,
    message_alive=True,
    scalar_desired_curvature=scalar,
    published_desired_curvature_time_s=0.25,
    native_plan_times_s=times,
    native_orientation_rates_z=rates,
    native_velocities_x=speeds,
    current_v_ego_m_s=10.0,
    physical_transport_delay_s=(
      selected_profile.parameters_at(10.0)
      .parameters.transport_delay_s
    ),
    output_plan_times_s=outputs[0],
    output_orientation_rates_z=outputs[1],
    output_velocities_x=outputs[2],
    output_plan_curvatures=outputs[3],
  )
  return adaptation, outputs


def update(
  selected_candidate: ModularControllerCandidate,
  selected_decision: EngagementDecision,
  *,
  scalar: float = 0.012,
  applied_counts: int = 0,
  recorded_applied_torque: float = 0.0,
  driver_torque: float = 0.0,
  measured_acceleration: float = 0.0,
  lateral_active: bool = True,
  lateral_valid: bool = True,
  malformed_future: bool = False,
  live: RackMappingSnapshot | None = None,
  steering_pressed: bool = False,
  steering_request_counter: int = 0,
  steering_request_state_valid: bool = True,
):
  adaptation, outputs = intent(
    selected_candidate.core.profile,
    scalar=scalar,
    malformed_future=malformed_future,
  )
  return selected_candidate.update(
    engagement_decision=selected_decision,
    previous_command_counts=applied_counts,
    recorded_applied_torque=recorded_applied_torque,
    driver_torque=driver_torque,
    frame=adaptation.frame,
    intent_status=adaptation.status,
    intent_plan_times_s=outputs[0],
    intent_orientation_rates_z=outputs[1],
    intent_velocities_x=outputs[2],
    scalar_curvature=scalar,
    current_v_ego_m_s=10.0,
    measured_rack_angle_deg=0.0,
    measured_rack_rate_deg_s=0.0,
    measured_rack_acceleration_deg_s2=measured_acceleration,
    lateral_accel_offset=0.0,
    live_mapping=mapping() if live is None else live,
    lateral_active=lateral_active,
    lateral_valid=lateral_valid,
    engagement_boundary=False,
    live_parameters_valid=True,
    steering_pressed=steering_pressed,
    steering_request_fault_avoidance_counter=steering_request_counter,
    steering_request_state_valid=steering_request_state_valid,
    actuator_constrained=False,
    output_constrained=False,
    standstill=False,
  )


def test_stock_default_shadows_once_and_never_exposes_actuation() -> None:
  selected_profile = profile()
  selected_candidate = candidate(selected_profile)
  stock = decision(None)
  selected_candidate.begin_engagement(stock)
  calls = 0
  original_update = selected_candidate.core.update

  def counted_update(**kwargs):
    nonlocal calls
    calls += 1
    return original_update(**kwargs)

  selected_candidate.core.update = counted_update  # type: ignore[method-assign]
  result = update(selected_candidate, stock)
  assert calls == 1
  assert result.status == CandidateStatus.SHADOW_STOCK
  assert result.shadow_valid
  assert not result.command_available
  assert not result.command_envelope_applied
  assert result.command_torque == 0.0
  assert result.core_result is not None
  assert result.controls_valid and result.car_control_valid
  assert selected_candidate.guard.invalid_frames == 0


def test_exact_profile_object_content_hash_and_vehicle_activate() -> None:
  selected_profile = profile()
  selected_candidate = candidate(selected_profile)
  modular = decision(selected_profile)
  selected_candidate.begin_engagement(modular)
  result = update(selected_candidate, modular)
  assert result.status == CandidateStatus.MODULAR_OK
  assert result.command_available
  assert result.command_envelope_applied
  assert result.selected_profile_sha256 == profile_sha256(
    selected_profile,
  )
  assert result.selected_profile_revision == selected_profile.revision


def test_unverified_opendbc_envelope_cannot_construct_horizon_core() -> None:
  selected_profile = profile()
  unverified = replace(
    LIMITS,
    production_envelope_verified=False,
  )
  with _TEST_CASE.assertRaisesRegex(
    ValueError,
    "verified 100 Hz envelope",
  ):
    candidate(selected_profile, limits=unverified)


def test_candidate_uses_the_horizon_cores_exact_limits_object() -> None:
  selected = candidate(profile())
  assert selected.runtime_limits is selected.core.runtime_limits
  assert selected.runtime_limits is selected.core.horizon.limits


def _modular_activation_rejects_profile_binding_error(
  malformed: str,
) -> None:
  selected_profile = profile()
  selected_candidate = candidate(selected_profile)
  if malformed == "missing":
    bad = decision(
      None,
      selection=ControllerSelection.MODULAR,
      selected_hash="missing",
    )
  elif malformed == "unqualified":
    incomplete = profile(qualified=False)
    bad = decision(incomplete)
  elif malformed == "object":
    equal_but_distinct = replace(selected_profile)
    assert equal_but_distinct == selected_profile
    assert equal_but_distinct is not selected_profile
    bad = decision(equal_but_distinct)
  elif malformed == "content":
    bad = decision(replace(selected_profile, revision=8))
  elif malformed == "hash":
    bad = decision(selected_profile, selected_hash="wrong")
  else:
    bad = decision(profile(identity="other"))
  with _TEST_CASE.assertRaises(ValueError):
    selected_candidate.begin_engagement(bad)
  assert not selected_candidate.engaged


def test_modular_activation_rejects_every_profile_binding_error() -> None:
  for malformed in (
    "missing",
    "unqualified",
    "object",
    "content",
    "hash",
    "vehicle",
  ):
    _modular_activation_rejects_profile_binding_error(malformed)


def test_stock_decision_cannot_smuggle_profile_or_provisional_state() -> None:
  selected_profile = profile()
  selected_candidate = candidate(selected_profile)
  for bad in (
    EngagementDecision(
      ControllerSelection.STOCK,
      selected_profile,
      profile_sha256(selected_profile),
      False,
    ),
    EngagementDecision(
      ControllerSelection.STOCK,
      None,
      "",
      True,
    ),
  ):
    with _TEST_CASE.assertRaises(ValueError):
      selected_candidate.begin_engagement(bad)


def _mid_engagement_decision_change_faults_until_explicit_end(
  change: str,
) -> None:
  selected_profile = profile()
  selected_candidate = candidate(selected_profile)
  modular = decision(selected_profile)
  selected_candidate.begin_engagement(modular)
  assert update(
    selected_candidate, modular,
  ).status == CandidateStatus.MODULAR_OK

  if change == "selection":
    changed_decision = decision(None)
  elif change == "profile":
    changed_decision = decision(replace(selected_profile, revision=8))
  elif change == "hash":
    changed_decision = decision(
      selected_profile,
      selected_hash="changed",
    )
  elif change == "core_profile":
    selected_candidate.core.profile = profile(revision=8)
    changed_decision = modular
  else:
    changed_decision = replace(modular)
  changed = update(selected_candidate, changed_decision)
  assert changed.status == CandidateStatus.ENGAGEMENT_DECISION_CHANGED
  assert not changed.shadow_valid
  assert changed.command_available
  assert changed.safety_state == LiveSafetyState.HOLDING_FIRST_INVALID
  assert changed.controls_valid

  faulted = update(selected_candidate, modular)
  assert faulted.status == CandidateStatus.ENGAGEMENT_BINDING_FAULTED
  assert faulted.command_available
  assert faulted.safety_state == LiveSafetyState.DECAYING_INVALID
  selected_candidate.end_engagement(modular)


def test_mid_engagement_decision_change_faults_until_explicit_end() -> None:
  for change in (
    "selection",
    "profile",
    "hash",
    "copied_decision",
    "core_profile",
  ):
    _mid_engagement_decision_change_faults_until_explicit_end(change)


def test_end_requires_bound_decision_and_calls_after_end_are_rejected() -> None:
  selected_profile = profile()
  selected_candidate = candidate(selected_profile)
  modular = decision(selected_profile)
  selected_candidate.begin_engagement(modular)
  copied = replace(modular)
  with _TEST_CASE.assertRaises(ValueError):
    selected_candidate.end_engagement(copied)
  assert not selected_candidate.engaged
  after = update(selected_candidate, modular)
  assert after.status == CandidateStatus.NOT_ENGAGED
  assert not after.command_available
  assert not after.controls_valid


def test_valid_live_command_agrees_exactly_with_read_only_projection() -> None:
  selected_profile = profile()
  selected_candidate = candidate(selected_profile)
  modular = decision(selected_profile)
  selected_candidate.begin_engagement(modular)
  result = update(
    selected_candidate,
    modular,
    scalar=0.08,
    applied_counts=37,
  )
  assert result.command_available
  assert result.feasibility_status.value == 0
  assert result.core_result is not None
  core = result.core_result
  assert result.raw_torque == core.raw_torque
  assert result.requested_counts == core.planned_counts
  assert result.feasible_counts is not None
  assert result.feasible_counts == core.planned_counts
  assert result.command_torque == result.feasible_torque
  assert result.command_torque == core.planned_torque
  assert not result.constraint_active
  assert (
    round(result.command_torque * LIMITS.steer_max)
    == result.feasible_counts
  )
  repeated = apply_torque_envelope_counts(
    LIMITS,
    result.feasible_counts,
    37,
    0.0,
  )
  assert repeated == result.feasible_counts


def test_transport_prime_keeps_command_and_physical_state_separate() -> None:
  selected_profile = profile(transport_delay_s=0.02)
  request_off = candidate(selected_profile)
  request_off.prime_transport_state(100, 0.0)
  assert request_off.core._history_count == request_off.core.history_capacity
  assert set(request_off.core._applied_history) == {0.0}

  request_active = candidate(selected_profile)
  applied = 100 / LIMITS.steer_max
  request_active.prime_transport_state(100, applied)
  assert set(request_active.core._applied_history) == {applied}

  with _TEST_CASE.assertRaisesRegex(ValueError, "physical torque"):
    request_off.prime_transport_state(100, math.nan)


def test_scalar_only_fails_closed_and_nominal_mapping_remains_explicit() -> None:
  selected_profile = profile()
  modular = decision(selected_profile)
  malformed_candidate = candidate(selected_profile)
  malformed_candidate.begin_engagement(modular)
  malformed = update(
    malformed_candidate,
    modular,
    malformed_future=True,
  )
  assert malformed.status == CandidateStatus.MODULAR_CORE_INVALID
  assert not malformed.shadow_valid
  assert malformed.command_available
  assert malformed.safety_state == LiveSafetyState.HOLDING_FIRST_INVALID

  nominal_candidate = candidate(selected_profile)
  nominal_candidate.begin_engagement(modular)
  nominal = update(
    nominal_candidate,
    modular,
    live=mapping(valid=False),
  )
  assert nominal.status == CandidateStatus.MODULAR_DEGRADED_NOMINAL_MAPPING
  assert nominal.shadow_valid
  assert nominal.command_available
  assert nominal.controls_valid


def test_unqualified_profile_is_shadow_only() -> None:
  incomplete = profile(qualified=False)
  selected_candidate = candidate(incomplete)
  with _TEST_CASE.assertRaises(ValueError):
    selected_candidate.begin_engagement(decision(incomplete))
  stock = decision(None)
  selected_candidate.begin_engagement(stock)
  result = update(selected_candidate, stock)
  assert result.status == CandidateStatus.SHADOW_UNQUALIFIED_PROFILE
  assert result.shadow_valid
  assert not result.command_available


def test_invalid_sequence_holds_decays_latches_and_recovers_ten_frames() -> None:
  selected_profile = profile()
  selected_candidate = candidate(selected_profile)
  modular = decision(selected_profile)
  selected_candidate.begin_engagement(modular)

  initial = update(
    selected_candidate,
    modular,
    scalar=0.08,
    applied_counts=100,
  )
  assert initial.status == CandidateStatus.MODULAR_OK
  first = update(
    selected_candidate,
    modular,
    applied_counts=100,
    measured_acceleration=math.nan,
  )
  assert first.status == CandidateStatus.MODULAR_CORE_INVALID
  assert first.command_torque == 100 / LIMITS.steer_max
  assert first.safety_state == LiveSafetyState.HOLDING_FIRST_INVALID

  second = update(
    selected_candidate,
    modular,
    applied_counts=100,
    measured_acceleration=math.nan,
  )
  assert second.command_torque == (
    100 - LIMITS.delta_down
  ) / LIMITS.steer_max
  assert second.safety_state == LiveSafetyState.DECAYING_INVALID

  current = second
  for _ in range(selected_candidate.guard.invalid_latch_frames - 2):
    current = update(
      selected_candidate,
      modular,
      applied_counts=round(
        current.command_torque * LIMITS.steer_max,
      ),
      measured_acceleration=math.nan,
    )
  assert current.safety_state == LiveSafetyState.COMM_ISSUE_LATCHED
  assert current.command_torque == 0.0
  assert not current.controls_valid
  assert not current.car_control_valid

  for frame_index in range(RECOVERY_OK_FRAMES - 1):
    current = update(
      selected_candidate,
      modular,
      scalar=0.08,
      applied_counts=0,
    )
    assert current.recovery_ok_frames == frame_index + 1
    assert current.status == CandidateStatus.MODULAR_OK
    assert current.safety_state == LiveSafetyState.COMM_ISSUE_LATCHED
    assert not current.controls_valid
  recovered = update(
    selected_candidate,
    modular,
    scalar=0.08,
    applied_counts=0,
  )
  assert recovered.safety_state == LiveSafetyState.OK
  assert recovered.controls_valid and recovered.car_control_valid
  assert recovered.command_torque == recovered.feasible_torque


def test_driver_override_uses_same_envelope_for_diagnostic_and_live() -> None:
  selected_profile = profile()
  selected_candidate = candidate(selected_profile)
  modular = decision(selected_profile)
  selected_candidate.begin_engagement(modular)
  result = update(
    selected_candidate,
    modular,
    scalar=0.08,
    applied_counts=0,
    driver_torque=-200.0,
    steering_pressed=True,
  )
  assert result.command_torque == result.feasible_torque
  assert result.feasible_counts == apply_torque_envelope_counts(
    LIMITS,
    result.requested_counts,
    0,
    -200.0,
  )
  assert result.core_result is not None
  core = result.core_result
  assert core.driver_suppressed
  assert core.raw_to_planned_constrained
  assert result.command_torque == core.planned_torque
  assert result.feasible_counts == core.planned_counts
  assert not result.constraint_active


def test_snapshot_is_deterministic_and_result_is_reused() -> None:
  selected_profile = profile()
  modular = decision(selected_profile)

  def replay() -> tuple[bytes, int]:
    selected_candidate = candidate(selected_profile)
    selected_candidate.begin_engagement(modular)
    encoded = bytearray()
    identity = 0
    for index in range(50):
      result = update(
        selected_candidate,
        modular,
        scalar=0.02 * math.sin(index * 0.1),
        applied_counts=index % 7,
      )
      if index == 0:
        identity = id(result)
      else:
        assert id(result) == identity
      encoded.extend(repr(result.snapshot()).encode("ascii"))
      encoded.append(10)
    return bytes(encoded), identity

  expected, _ = replay()
  for _ in range(3):
    actual, _ = replay()
    assert actual == expected


def test_facade_neither_imports_stock_nor_mutates_reference_inputs() -> None:
  selected_profile = profile()
  selected_candidate = candidate(selected_profile)
  modular = decision(selected_profile)
  selected_candidate.begin_engagement(modular)
  adaptation, outputs = intent(selected_profile)
  before = tuple(tuple(values) for values in outputs)
  selected_candidate.update(
    engagement_decision=modular,
    previous_command_counts=0,
    recorded_applied_torque=0.0,
    driver_torque=0.0,
    frame=adaptation.frame,
    intent_status=adaptation.status,
    intent_plan_times_s=outputs[0],
    intent_orientation_rates_z=outputs[1],
    intent_velocities_x=outputs[2],
    scalar_curvature=0.012,
    current_v_ego_m_s=10.0,
    measured_rack_angle_deg=0.0,
    measured_rack_rate_deg_s=0.0,
    measured_rack_acceleration_deg_s2=0.0,
    lateral_accel_offset=0.0,
    live_mapping=mapping(),
    lateral_active=True,
    lateral_valid=True,
    engagement_boundary=False,
    live_parameters_valid=True,
    steering_pressed=False,
    steering_request_fault_avoidance_counter=0,
    steering_request_state_valid=True,
    actuator_constrained=False,
    output_constrained=False,
    standstill=False,
  )
  assert tuple(tuple(values) for values in outputs) == before

  source = Path(
    inspect.getfile(ModularControllerCandidate),
  ).read_text()
  for forbidden in (
    "LatControlTorque",
    "BLaTv1",
    "v14",
    "409",
    "DT_MDL",
    "LAT_SMOOTH_SECONDS",
  ):
    assert forbidden not in source
  signature = inspect.signature(ModularControllerCandidate)
  assert "runtime_limits" not in signature.parameters
