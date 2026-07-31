from __future__ import annotations

from dataclasses import replace
import inspect
import math
from pathlib import Path
from unittest.mock import patch

from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.selfdrive.controls.lib.blatv2.stock_bootstrap import (
  fresh_stock_torque_controller,
)
from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
)
from openpilot.selfdrive.controls.lib.blatv2.approved_artifact import (
  ApprovedProfileArtifact,
  ArtifactDiagnostic,
)
from openpilot.selfdrive.controls.lib.blatv2.bootstrap import (
  ControllerSelection,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  make_calibration_seed_profile,
)
from openpilot.selfdrive.controls.lib.blatv2.controller import (
  CandidateStatus,
)
from openpilot.selfdrive.controls.lib.blatv2.core import CoreStatus
from openpilot.selfdrive.controls.lib.blatv2.live_adapter import (
  INTENT_CAPACITY,
  LiveAdapterStatus,
  LiveInputAdapter,
  exact_applied_torque_counts,
)
from openpilot.selfdrive.controls.lib.blatv2 import live_controller as live_module
from openpilot.selfdrive.controls.lib.blatv2.live_controller import (
  LiveEligibility,
  ModularLiveController,
)
from openpilot.selfdrive.controls.lib.blatv2.live_safety import (
  LiveSafetyState,
  RECOVERY_OK_FRAMES,
)
from openpilot.selfdrive.controls.lib.blatv2.live_telemetry import (
  build_modular_lateral_state,
)
from openpilot.selfdrive.controls.lib.blatv2.policy import ControllerPolicy
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  RuntimeVehicleBundle,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  PhysicalParameters,
  ProfileNode,
  VehicleProfile,
)


SOURCE_COMMIT = "1" * 40
OPENDBC_COMMIT = "2" * 40
EVIDENCE_HASH = "3" * 64
HARNESS_COMMIT = "4" * 40
BASE_NS = 10_000_000_000


def synthetic_cp() -> car.CarParams:
  cp = car.CarParams.new_message()
  cp.carFingerprint = "synthetic-torque-platform"
  cp.steerControlType = car.CarParams.SteerControlType.torque
  cp.mass = 2100.0
  cp.rotationalInertia = 3000.0
  cp.wheelbase = 2.9
  cp.centerToFront = 1.2
  cp.steerRatio = 15.0
  cp.steerRatioRear = 0.0
  cp.tireStiffnessFront = 100000.0
  cp.tireStiffnessRear = 110000.0
  cp.tireStiffnessFactor = 1.0
  cp.steerActuatorDelay = 0.12
  cp.maxLateralAccel = 4.0
  cp.lateralTuning.init("torque")
  cp.lateralTuning.torque.latAccelFactor = 3.0
  cp.lateralTuning.torque.latAccelOffset = 0.0
  cp.lateralTuning.torque.friction = 0.09
  return cp


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


def qualified_profile(
  *,
  identity: str = "synthetic-torque-platform",
) -> VehicleProfile:
  parameters = PhysicalParameters(
    torque_per_lateral_accel=1.0 / 3.0,
    rack_gain_deg_s2_per_torque=4000.0,
    rack_damping_per_s=10.0,
    transport_delay_s=0.12,
    static_friction_torque=0.09,
    kinetic_friction_torque=0.03,
    rack_rate_resolution_deg_s=4.0,
    confidence=0.95,
    qualified=True,
  )
  return VehicleProfile(
    vehicle_identity=identity,
    revision=1,
    provenance="synthetic replay-qualified profile",
    nodes=(
      ProfileNode(0.0, parameters, 200.0, 20000, 5000, 0.01),
      ProfileNode(30.0, parameters, 600.0, 60000, 12000, 0.02),
    ),
  )


def policy() -> ControllerPolicy:
  return ControllerPolicy(
    revision=1,
    provenance="synthetic accepted policy",
    provisional=False,
    natural_frequency_per_s=8.0,
    damping_ratio=1.0,
    observer_time_constant_s=None,
    observer_max_abs_disturbance_torque=None,
  )


def runtime_bundle(
  *,
  verified: bool = True,
) -> RuntimeVehicleBundle:
  profile = qualified_profile()
  calibration_seed = make_calibration_seed_profile(
    vehicle_identity=profile.vehicle_identity,
    torque_callback_slope=1.0 / 3.0,
    stock_friction_torque=0.09,
    transport_delay_s=0.12,
    rack_rate_resolution_deg_s=4.0,
    speed_nodes_mps=(0.0, 30.0),
  )
  return RuntimeVehicleBundle(
    vehicle_identity=profile.vehicle_identity,
    car_fingerprint=profile.vehicle_identity,
    provisional_rack_provenance="synthetic explicit seed",
    torque_limits=RuntimeTorqueLimits(
      409,
      4,
      7,
      1,
      50,
      2,
      1,
      production_envelope_verified=verified,
    ),
    nominal_rack_mapping=mapping(),
    calibration_seed_profile=calibration_seed,
    seed_profile=profile,
    stock_lateral_accel_offset_mps2=0.0,
    torque_callback_slope=1.0 / 3.0,
    torque_callback_max_abs_residual=0.0,
    torque_callback_representation_tolerance=1e-14,
  )


def artifact(bundle: RuntimeVehicleBundle) -> ApprovedProfileArtifact:
  return ApprovedProfileArtifact(
    vehicle_profile=qualified_profile(
      identity=bundle.vehicle_identity,
    ),
    controller_policy=policy(),
    runtime_vehicle_identity_sha256=bundle.identity_sha256,
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
    learner_evidence_sha256=EVIDENCE_HASH,
    replay_harness_commit=HARNESS_COMMIT,
    replay_passed=True,
    delivered_replay_passed=True,
    safety_passed=True,
    deterministic_aa_passed=True,
    device_timing_passed=True,
  )


def live(
  *,
  bundle: RuntimeVehicleBundle | None = None,
  selected_artifact: ApprovedProfileArtifact | None = None,
  diagnostic: ArtifactDiagnostic = ArtifactDiagnostic.OK,
) -> ModularLiveController:
  cp = synthetic_cp()
  actual_bundle = runtime_bundle() if bundle is None else bundle
  actual_artifact = (
    artifact(actual_bundle)
    if selected_artifact is None and diagnostic == ArtifactDiagnostic.OK
    else selected_artifact
  )
  return ModularLiveController(
    car_params=cp,
    runtime_bundle=actual_bundle,
    artifact=actual_artifact,
    activation_provisional=True,
    artifact_diagnostic=diagnostic,
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
  )


def model(
  *,
  curvature: float = 0.012,
  same_grid: bool = True,
) -> object:
  message = log.ModelDataV2.new_message()
  times = [index * 0.05 for index in range(INTENT_CAPACITY)]
  speeds = [10.0] * len(times)
  curvatures = [curvature + 0.001 * time for time in times]
  message.frameId = 42
  message.timestampEof = BASE_NS
  message.orientationRate.t = times
  message.orientationRate.z = [
    value * speed
    for value, speed in zip(curvatures, speeds, strict=True)
  ]
  message.velocity.t = (
    times
    if same_grid
    else [time + 0.001 for time in times]
  )
  message.velocity.x = speeds
  message.action.desiredCurvature = curvature
  message.action.desiredCurvatureTime = 0.35
  return message


def vehicle_messages() -> tuple[object, object, object]:
  state = car.CarState.new_message()
  state.vEgo = 10.0
  state.steeringAngleDeg = 0.0
  state.steeringRateDeg = 0.0
  state.steeringTorque = 0.0
  state.steeringPressed = False
  state.standstill = False

  output = car.CarOutput.new_message()
  output.actuatorsOutput.torqueOutputCan = 37.0

  live_params = log.LiveParametersData.new_message()
  live_params.valid = True
  live_params.angleOffsetValid = True
  live_params.steerRatioValid = True
  live_params.stiffnessFactorValid = True
  live_params.stiffnessFactor = 1.0
  live_params.steerRatio = 15.0
  live_params.angleOffsetDeg = 0.0
  live_params.roll = 0.0
  return state, output, live_params


class FakeClock:
  def __init__(self):
    self.now_ns = BASE_NS + 250_000_000

  def advance(self) -> None:
    self.now_ns += 10_000_000


def bind_modular(
  selected: ModularLiveController,
  clock: FakeClock,
  state: object,
  output: object,
) -> None:
  selected.update_engagement(
    enabled=False,
    lateral_active=False,
    lateral_maneuver_active=False,
  )
  selected.observe_inactive_state(
    state_sample_mono_ns=clock.now_ns - 15_000_000,
    car_state=state,
    inputs_valid=True,
  )
  assert selected.observe_previous_applied(output)
  assert selected.update_engagement(
    enabled=True,
    lateral_active=True,
    lateral_maneuver_active=False,
  ) == ControllerSelection.MODULAR


def step(
  selected: ModularLiveController,
  clock: FakeClock,
  state: object,
  output: object,
  live_params: object,
  *,
  message_valid: bool = True,
  live_parameters_valid: bool = True,
  maneuver_active: bool = False,
):
  selected.observe_previous_applied(output)
  result = selected.update_modular(
    state_sample_mono_ns=clock.now_ns - 5_000_000,
    model_publication_mono_ns=clock.now_ns - 10_000_000,
    model_message=model(),
    car_state=state,
    live_parameters=live_params,
    model_message_valid=message_valid,
    model_message_alive=True,
    vehicle_inputs_valid=True,
    live_parameters_inputs_valid=live_parameters_valid,
    lateral_active=True,
    actuator_constrained_previous=False,
    lateral_maneuver_active=maneuver_active,
  )
  output.actuatorsOutput.torqueOutputCan = float(
    round(result.command_torque * 409),
  )
  clock.advance()
  return result


def test_absent_invalid_mismatched_and_unverified_are_exact_stock() -> None:
  bundle = runtime_bundle()
  absent = live(
    bundle=bundle,
    selected_artifact=None,
    diagnostic=ArtifactDiagnostic.ABSENT,
  )
  mismatched = live(
    bundle=bundle,
    selected_artifact=replace(
      artifact(bundle),
      runtime_vehicle_identity_sha256="f" * 64,
    ),
  )
  unverified_bundle = runtime_bundle(verified=False)
  unverified = live(
    bundle=unverified_bundle,
    selected_artifact=artifact(unverified_bundle),
  )
  malformed = live(
    bundle=bundle,
    selected_artifact=None,
    diagnostic=ArtifactDiagnostic.STATE_INVALID,
  )

  for controller in (absent, mismatched, unverified, malformed):
    state, output, _ = vehicle_messages()
    controller.observe_previous_applied(output)
    selection = controller.update_engagement(
      enabled=True,
      lateral_active=True,
      lateral_maneuver_active=False,
    )
    assert selection == ControllerSelection.STOCK
    assert controller.candidate_result is None
    assert controller.messages_valid
    assert controller.selection == ControllerSelection.STOCK


def test_exact_artifact_is_only_modular_selection_and_boundary_is_immutable() -> None:
  selected = live()
  state, output, _ = vehicle_messages()
  clock = FakeClock()
  bind_modular(selected, clock, state, output)
  bound_decision = selected.decision
  assert selected.eligibility == LiveEligibility.ELIGIBLE

  # Brief lateral loss and a maneuver signal cannot switch an enabled session.
  assert selected.update_engagement(
    enabled=True,
    lateral_active=False,
    lateral_maneuver_active=True,
  ) == ControllerSelection.MODULAR
  assert selected.decision is bound_decision

  # Enabled falling while AOL/MADS keeps lateral active also cannot switch.
  assert selected.update_engagement(
    enabled=False,
    lateral_active=True,
    lateral_maneuver_active=False,
  ) == ControllerSelection.MODULAR
  assert selected.decision is bound_decision

  # Only both false end the session.
  assert selected.update_engagement(
    enabled=False,
    lateral_active=False,
    lateral_maneuver_active=False,
  ) == ControllerSelection.STOCK
  assert not selected.enabled_bound


def test_lateral_only_session_can_bind_modular_and_enabled_toggle_is_immutable(
) -> None:
  selected = live()
  _, output, _ = vehicle_messages()
  selected.observe_previous_applied(output)
  decision = selected.update_engagement(
    enabled=False,
    lateral_active=True,
    lateral_maneuver_active=False,
  )
  assert decision == ControllerSelection.MODULAR
  exact_decision = selected.decision
  assert exact_decision is not None

  # An enabled transition within the same AOL/MADS lateral session is not a
  # second binding boundary and cannot switch or reconstruct the controller.
  assert selected.update_engagement(
    enabled=True,
    lateral_active=True,
    lateral_maneuver_active=False,
  ) == ControllerSelection.MODULAR
  assert selected.decision is exact_decision


def test_maneuver_mode_binds_stock_for_complete_lateral_session() -> None:
  selected = live()
  _, output, _ = vehicle_messages()
  selected.observe_previous_applied(output)
  assert selected.update_engagement(
    enabled=True,
    lateral_active=True,
    lateral_maneuver_active=True,
  ) == ControllerSelection.STOCK
  # Clearing maneuver mode while either lifetime signal remains true cannot
  # hot-switch the immutable session.
  assert selected.update_engagement(
    enabled=False,
    lateral_active=True,
    lateral_maneuver_active=False,
  ) == ControllerSelection.STOCK
  assert selected.candidate_result is None


def test_exact_count_source_rejects_fractional_normalized_reconstruction() -> None:
  selected = live()
  _, output, _ = vehicle_messages()
  limits = selected.runtime_bundle.torque_limits
  assert exact_applied_torque_counts(output, limits) == 37
  output.actuatorsOutput.torqueOutputCan = 37.25
  assert exact_applied_torque_counts(output, limits) is None
  assert not selected.observe_previous_applied(output)
  assert selected.last_exact_applied_counts is None


def test_native_grid_and_derivative_gap_contracts() -> None:
  cp = synthetic_cp()
  selected_profile = qualified_profile()
  adapter = LiveInputAdapter(
    car_params=cp,
    profile=selected_profile,
  )
  state, _, live_params = vehicle_messages()
  adapter.observe_inactive_state(
    state_sample_mono_ns=BASE_NS + 200_000_000,
    car_state=state,
    inputs_valid=True,
  )
  prepared = adapter.prepare(
    state_sample_mono_ns=BASE_NS + 210_000_000,
    control_witness_mono_ns=BASE_NS + 215_000_000,
    model_publication_mono_ns=BASE_NS + 205_000_000,
    model_message=model(same_grid=False),
    car_state=state,
    live_parameters=live_params,
    model_message_valid=True,
    model_message_alive=True,
    vehicle_inputs_valid=True,
    live_parameters_inputs_valid=True,
  )
  assert prepared.rack_derivative_valid
  assert prepared.adaptation.status.scalar_only
  assert math.isclose(prepared.scalar_curvature, 0.012, rel_tol=1e-6, abs_tol=1e-12)

  gapped = adapter.prepare(
    state_sample_mono_ns=BASE_NS + 230_000_001,
    control_witness_mono_ns=BASE_NS + 235_000_000,
    model_publication_mono_ns=BASE_NS + 225_000_000,
    model_message=model(),
    car_state=state,
    live_parameters=live_params,
    model_message_valid=True,
    model_message_alive=True,
    vehicle_inputs_valid=True,
    live_parameters_inputs_valid=True,
  )
  assert gapped.status == LiveAdapterStatus.INVALID_RACK_DERIVATIVE
  assert not gapped.rack_derivative_valid
  assert math.isnan(gapped.measured_rack_acceleration_deg_s2)


def test_nonzero_delay_zoh_prime_makes_first_active_frame_valid(
  monkeypatch,
) -> None:
  selected = live()
  state, output, live_params = vehicle_messages()
  clock = FakeClock()
  assert selected.candidate is not None
  prime_values: list[float] = []
  original_prime = selected.candidate.core.prime_applied_history

  def counted_prime(value: float) -> None:
    prime_values.append(float(value))
    original_prime(value)

  selected.candidate.core.prime_applied_history = counted_prime
  monkeypatch.setattr(
    live_module,
    "control_witness_mono_ns",
    lambda: clock.now_ns,
  )
  bind_modular(selected, clock, state, output)
  assert prime_values == [37 / 409]
  assert selected.candidate.core.history_capacity >= 12
  assert selected.candidate.core._history_count == (
    selected.candidate.core.history_capacity
  )
  first = step(selected, clock, state, output, live_params)
  assert first.status == CandidateStatus.MODULAR_OK
  assert first.core_result is not None and first.core_result.valid
  assert first.core_result.prediction_history_count >= 12
  assert first.command_available
  assert first.safety_state == LiveSafetyState.OK
  assert first.controls_valid
  assert selected.last_exact_applied_counts == 37
  # A valid first witness establishes cadence; it does not prime twice.
  assert prime_values == [37 / 409]


def test_control_witness_discontinuity_reprimes_and_uses_guard_recovery(
) -> None:
  for cadence_failure in ("repeated", "gap"):
    selected = live()
    state, output, live_params = vehicle_messages()
    clock = FakeClock()
    assert selected.candidate is not None
    prime_values: list[float] = []
    original_prime = selected.candidate.core.prime_applied_history

    def counted_prime(
      value: float,
      values: list[float] = prime_values,
      prime=original_prime,
    ) -> None:
      values.append(float(value))
      prime(value)

    selected.candidate.core.prime_applied_history = counted_prime
    with patch.object(
      live_module,
      "control_witness_mono_ns",
      side_effect=lambda selected_clock=clock: selected_clock.now_ns,
    ):
      bind_modular(selected, clock, state, output)
      valid = step(selected, clock, state, output, live_params)
      assert valid.status == CandidateStatus.MODULAR_OK
      assert len(prime_values) == 1

      if cadence_failure == "repeated":
        clock.now_ns -= 10_000_000
      else:
        clock.now_ns += 10_000_000
      exact_counts_before_failure = round(
        output.actuatorsOutput.torqueOutputCan,
      )
      invalid = step(selected, clock, state, output, live_params)
      assert not selected.control_cadence_valid
      assert selected.transport_reprimed
      assert len(prime_values) == 2
      assert prime_values[-1] == (
        exact_counts_before_failure / 409
      )
      assert invalid.status == CandidateStatus.MODULAR_CORE_INVALID
      assert invalid.safety_state == LiveSafetyState.HOLDING_FIRST_INVALID

      for frame in range(RECOVERY_OK_FRAMES - 1):
        recovering = step(selected, clock, state, output, live_params)
        assert selected.control_cadence_valid
        assert not selected.transport_reprimed
        assert recovering.safety_state == LiveSafetyState.RECOVERING
        assert recovering.recovery_ok_frames == frame + 1
      recovered = step(selected, clock, state, output, live_params)
      assert recovered.status == CandidateStatus.MODULAR_OK
      assert recovered.safety_state == LiveSafetyState.OK
      assert selected.messages_valid
      assert len(prime_values) == 2


def test_invalid_live_parameters_are_diagnostic_only_and_guard_actuation(
) -> None:
  for failure in ("message", "subscription"):
    selected = live()
    state, output, live_params = vehicle_messages()
    clock = FakeClock()
    with patch.object(
      live_module,
      "control_witness_mono_ns",
      side_effect=lambda selected_clock=clock: selected_clock.now_ns,
    ):
      bind_modular(selected, clock, state, output)
      valid = step(selected, clock, state, output, live_params)
      assert valid.status == CandidateStatus.MODULAR_OK
      prior_command = output.actuatorsOutput.torqueOutputCan / 409.0

      if failure == "message":
        live_params.valid = False
      invalid = step(
        selected,
        clock,
        state,
        output,
        live_params,
        live_parameters_valid=failure != "subscription",
      )
      assert selected.prepared_input is not None
      assert (
        selected.prepared_input.status
        == LiveAdapterStatus.INVALID_LIVE_PARAMETERS
      )
      assert not selected.prepared_input.live_parameters_valid
      assert not selected.prepared_input.lateral_valid
      assert invalid.core_result is not None
      assert (
        invalid.core_result.status
        == CoreStatus.DEGRADED_NOMINAL_MAPPING
      )
      assert invalid.status == CandidateStatus.MODULAR_CORE_INVALID
      assert invalid.safety_state == LiveSafetyState.HOLDING_FIRST_INVALID
      assert invalid.command_torque == prior_command
      assert selected.messages_valid


def test_stock_maneuver_selection_never_primes_or_calls_modular_core() -> None:
  selected = live()
  assert selected.candidate is not None
  calls = 0
  original = selected.candidate.core.prime_applied_history

  def counted(value):
    nonlocal calls
    calls += 1
    return original(value)

  selected.candidate.core.prime_applied_history = counted
  _, output, _ = vehicle_messages()
  selected.observe_previous_applied(output)
  assert selected.update_engagement(
    enabled=True,
    lateral_active=True,
    lateral_maneuver_active=True,
  ) == ControllerSelection.STOCK
  assert calls == 0
  assert selected.previous_control_witness_ns is None


def test_invalid_guard_propagates_hold_decay_latch_and_ten_ok_recovery(
  monkeypatch,
) -> None:
  selected = live()
  state, output, live_params = vehicle_messages()
  clock = FakeClock()
  monkeypatch.setattr(
    live_module,
    "control_witness_mono_ns",
    lambda: clock.now_ns,
  )
  bind_modular(selected, clock, state, output)
  valid = step(selected, clock, state, output, live_params)
  assert valid.safety_state == LiveSafetyState.OK
  previous = output.actuatorsOutput.torqueOutputCan / 409.0

  first = step(
    selected,
    clock,
    state,
    output,
    live_params,
    message_valid=False,
  )
  assert first.safety_state == LiveSafetyState.HOLDING_FIRST_INVALID
  assert first.command_torque == previous
  assert selected.messages_valid

  current = first
  latch_frames = selected.candidate.guard.invalid_latch_frames
  for _ in range(latch_frames - 1):
    current = step(
      selected,
      clock,
      state,
      output,
      live_params,
      message_valid=False,
    )
  assert current.safety_state == LiveSafetyState.COMM_ISSUE_LATCHED
  assert current.command_torque == 0.0
  assert not current.controls_valid
  assert not selected.messages_valid

  for frame in range(RECOVERY_OK_FRAMES - 1):
    recovering = step(
      selected,
      clock,
      state,
      output,
      live_params,
    )
    assert recovering.recovery_ok_frames == frame + 1
    assert not selected.messages_valid
  recovered = step(selected, clock, state, output, live_params)
  assert recovered.safety_state == LiveSafetyState.OK
  assert recovered.controls_valid
  assert selected.messages_valid


def test_binding_mismatch_poison_lasts_until_both_false_session_boundary(
  monkeypatch,
) -> None:
  selected = live()
  state, output, live_params = vehicle_messages()
  clock = FakeClock()
  monkeypatch.setattr(
    live_module,
    "control_witness_mono_ns",
    lambda: clock.now_ns,
  )
  bind_modular(selected, clock, state, output)
  step(selected, clock, state, output, live_params)
  exact_decision = selected.decision
  assert exact_decision is not None
  selected.decision = replace(exact_decision)

  current = step(selected, clock, state, output, live_params)
  assert current.status == CandidateStatus.ENGAGEMENT_DECISION_CHANGED
  assert current.safety_state == LiveSafetyState.HOLDING_FIRST_INVALID
  for _ in range(selected.candidate.guard.invalid_latch_frames - 1):
    current = step(selected, clock, state, output, live_params)
  assert current.safety_state == LiveSafetyState.COMM_ISSUE_LATCHED
  assert not selected.messages_valid

  # Returning the exact object cannot recover a poisoned identity binding in
  # the same session, even after more than the ordinary ten-OK recovery span.
  selected.decision = exact_decision
  for _ in range(RECOVERY_OK_FRAMES + 2):
    current = step(selected, clock, state, output, live_params)
    assert current.status == CandidateStatus.ENGAGEMENT_BINDING_FAULTED
    assert current.safety_state == LiveSafetyState.COMM_ISSUE_LATCHED
    assert not selected.messages_valid

  # Only the actual AOL/MADS session end resets the poisoned binding.  A new
  # latActive-only session may then bind the same exact artifact and starts
  # cleanly through the normal transport-history prime.
  assert selected.update_engagement(
    enabled=False,
    lateral_active=False,
    lateral_maneuver_active=False,
  ) == ControllerSelection.STOCK
  selected.observe_inactive_state(
    state_sample_mono_ns=clock.now_ns - 15_000_000,
    car_state=state,
    inputs_valid=True,
  )
  assert selected.observe_previous_applied(output)
  assert selected.update_engagement(
    enabled=False,
    lateral_active=True,
    lateral_maneuver_active=False,
  ) == ControllerSelection.MODULAR
  current = step(selected, clock, state, output, live_params)
  assert current.status == CandidateStatus.MODULAR_OK
  assert current.safety_state == LiveSafetyState.OK
  assert selected.messages_valid


def test_envelope_output_raw_scalar_timing_and_schema_round_trip(
  monkeypatch,
) -> None:
  selected = live()
  state, output, live_params = vehicle_messages()
  clock = FakeClock()
  monkeypatch.setattr(
    live_module,
    "control_witness_mono_ns",
    lambda: clock.now_ns,
  )
  bind_modular(selected, clock, state, output)
  result = step(selected, clock, state, output, live_params)
  prepared = selected.prepared_input
  assert prepared is not None
  assert math.isclose(prepared.scalar_curvature, 0.012, rel_tol=1e-6, abs_tol=1e-12)
  assert math.isclose(
    prepared.desired_curvature_time_s,
    float(model().action.desiredCurvatureTime),
    rel_tol=1e-6,
    abs_tol=1e-12,
  )
  assert prepared.state_sample_mono_ns == BASE_NS + 245_000_000
  assert prepared.control_witness_mono_ns == BASE_NS + 250_000_000
  assert result.feasible_counts is not None
  assert result.command_torque == result.feasible_torque
  assert (
    round(result.command_torque * 409)
    == result.feasible_counts
  )

  telemetry = build_modular_lateral_state(
    selected,
    lateral_active=True,
    measured_curvature=0.01,
    v_ego_m_s=10.0,
  )
  encoded = telemetry.to_bytes()
  with log.ControlsState.LateralTorqueState.from_bytes(encoded) as decoded:
    assert decoded.modularArchitecture
    assert decoded.modularSelection == ControllerSelection.MODULAR
    assert decoded.modularArtifactHash == selected.artifact_sha256
    assert decoded.modularPolicyHash == selected.policy_sha256
    assert decoded.modularRuntimeIdentityHash == (
      selected.runtime_identity_sha256
    )
    assert decoded.modularControlWitnessMonoTime == (
      BASE_NS + 250_000_000
    )
    assert decoded.modularStateSampleMonoTime == (
      BASE_NS + 245_000_000
    )
    assert math.isclose(decoded.modularRawScalarCurvature, 0.012, rel_tol=1e-6, abs_tol=1e-12)
    assert decoded.modularCommandTorque == result.command_torque
    assert decoded.modularCommandEnvelopeApplied
    assert decoded.modularControlsValid


def test_hot_path_and_engagement_have_no_params_or_file_io() -> None:
  sources = "\n".join((
    inspect.getsource(ModularLiveController.update_engagement),
    inspect.getsource(ModularLiveController.update_modular),
    inspect.getsource(LiveInputAdapter.prepare),
  ))
  forbidden = (
    "Params(",
    ".get(",
    ".put(",
    "open(",
    "read_text(",
    "write_text(",
  )
  for token in forbidden:
    assert token not in sources


def test_modular_boundary_reconstructs_every_hidden_stock_state() -> None:
  cp = synthetic_cp()
  cp.steerLimitTimer = 0.8
  cp = cp.as_reader()

  class FakeInterface:
    @staticmethod
    def torque_from_lateral_accel():
      return lambda lateral_accel, tuning: (
        lateral_accel / tuning.latAccelFactor
      )

    @staticmethod
    def lateral_accel_from_torque():
      return lambda torque, tuning: torque * tuning.latAccelFactor

  stale = fresh_stock_torque_controller(cp, FakeInterface())
  stale.pid.i = 0.4
  stale.pid.p = 0.3
  stale.lat_accel_request_buffer.append(2.0)
  stale.jerk_filter.x = 9.0
  stale.sat_time = 0.7

  fresh = fresh_stock_torque_controller(cp, FakeInterface())
  assert fresh is not stale
  assert fresh.pid.i == 0.0
  assert fresh.pid.p == 0.0
  assert fresh.jerk_filter.x == 0.0
  assert fresh.sat_time == 0.0
  assert set(fresh.lat_accel_request_buffer) == {0.0}


def test_stock_arm_remains_the_unmodified_stock_update_shape() -> None:
  source = (
    Path(__file__).parents[1] / "controlsd.py"
  ).read_text()
  stock_start = source.index(
    "if controller_selection == ControllerSelection.STOCK:",
  )
  modular_start = source.index(
    "    else:\n      result = self.blatv2_live.update_modular(",
    stock_start,
  )
  stock_arm = source[stock_start:modular_start]
  for required in (
    "lateralManeuverPlan",
    "model_v2.action.desiredCurvature",
    "clip_curvature(",
    'self.sm[\"liveDelay\"].lateralDelay + LAT_SMOOTH_SECONDS',
    "self.LaC.update(",
    "actuators.torque = float(steer)",
  ):
    assert required in stock_arm
  assert "update_modular" not in stock_arm
  assert "blatv2_messages_valid = True" in stock_arm


def test_production_sources_have_no_platform_literals_or_legacy_mechanisms() -> None:
  root = Path(__file__).parents[1]
  sources = "\n".join(
    (root / "lib" / "blatv2" / name).read_text()
    for name in (
      "live_adapter.py",
      "live_controller.py",
      "live_telemetry.py",
    )
  )
  for forbidden in (
    "hyundai",
    "palisade",
    "telluride",
    "UnwindPhaseTracker",
    "HyundaiLowSpeedTorqueDamping",
    "LQI",
    "MPC",
  ):
    assert forbidden.lower() not in sources.lower()
