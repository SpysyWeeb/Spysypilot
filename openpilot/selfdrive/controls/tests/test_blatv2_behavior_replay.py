from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import ast
import math
from pathlib import Path
from unittest.mock import patch

from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel

from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
  apply_torque_envelope_counts,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  ReplayArtifactIdentity,
  ReplayCoreIdentity,
  ReplayRole,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorControlResponse,
  BehaviorSourceIdentity,
  SparseModelBehaviorIntent,
  derive_behavior_reference,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import BehaviorPolicy
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay import (
  ReplayFrameInput,
  SOURCE_LAT_SMOOTH_SECONDS,
  make_behavior_route_evidence_decoder,
  make_exact_stock_behavior_replay_core,
  make_modular_behavior_replay_core,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_transaction import (
  CanonicalBehaviorControlInput,
  ControllerReplayRequest,
  DecodedBehaviorRoute,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  CalibrationParameters,
  CalibrationProfileNode,
  VehicleCalibrationProfile,
)
from openpilot.selfdrive.controls.lib.blatv2.core import ModularControllerCore
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_spool import (
  _encode_frame,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  MeasuredLearningFrame,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import RackMappingSnapshot
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  ControlsWitness,
  DrivingEventLocator,
  LateralManeuverPlanPublication,
  LiveDelayPublication,
  LiveTorqueParametersPublication,
  ModelPublication,
  RouteEvidenceArtifact,
  RouteEvidenceSourceIdentity,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
)
from openpilot.selfdrive.controls.lib.blatv2.stock_bootstrap import (
  fresh_stock_torque_controller,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  DEFAULT_SPEED_NODES_MPS,
)


BASE_NS = 50_000_000_000
FINGERPRINT = "synthetic-behavior-replay"


class CarControllerParams:
  STEER_MAX = 409
  STEER_DELTA_UP = 4
  STEER_DELTA_DOWN = 7
  STEER_STEP = 1
  STEER_DRIVER_ALLOWANCE = 50
  STEER_DRIVER_MULTIPLIER = 2
  STEER_DRIVER_FACTOR = 1
  BLATV2_RUNTIME_ENVELOPE_COMPATIBLE = True
  BLATV2_RACK_RATE_RESOLUTION_DEG_S = 4.0

  def __init__(self, CP) -> None:
    del CP


class _FakeController:
  pass


class _FakeInterface:
  CarController = _FakeController

  def __init__(self, CP) -> None:
    self.CP = CP

  @staticmethod
  def torque_from_lateral_accel():
    return lambda lateral_accel, torque_params: (
      lateral_accel / torque_params.latAccelFactor
    )

  @staticmethod
  def lateral_accel_from_torque():
    return lambda torque, torque_params: torque * torque_params.latAccelFactor


INTERFACES = {FINGERPRINT: _FakeInterface}


def synthetic_cp() -> object:
  cp = car.CarParams.new_message()
  cp.carFingerprint = FINGERPRINT
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
  cp.steerActuatorDelay = 0.0
  cp.maxLateralAccel = 4.0
  cp.steerLimitTimer = 1.0
  cp.lateralTuning.init("torque")
  cp.lateralTuning.torque.latAccelFactor = 3.0
  cp.lateralTuning.torque.latAccelOffset = 0.0
  cp.lateralTuning.torque.friction = 0.09
  cp.lateralTuning.torque.steeringAngleDeadzoneDeg = 0.0
  return cp


def nominal_mapping(cp: object) -> RackMappingSnapshot:
  return RackMappingSnapshot.from_vehicle_model(
    VehicleModel(cp),
    roll_rad=0.0,
    angle_offset_deg=0.0,
    valid=True,
  )


def physical_profile() -> VehicleCalibrationProfile:
  parameters = CalibrationParameters(
    torque_per_lateral_accel=1.0 / 3.0,
    lateral_accel_offset_correction_mps2=0.0,
    kinetic_friction_torque=0.03,
    static_breakaway_torque=0.09,
    transport_delay_s=0.0,
    rack_rate_resolution_deg_s=4.0,
    confidence=1.0,
    qualified=True,
  )
  return VehicleCalibrationProfile(
    vehicle_identity=FINGERPRINT,
    revision=1,
    provenance="synthetic exact behavior replay",
    nodes=tuple(
      CalibrationProfileNode(
        speed_mps=speed,
        parameters=parameters,
        base_support_s=600.0,
        base_sample_count=60000,
        moving_support_s=300.0,
        moving_sample_count=30000,
        breakaway_support_s=30.0,
        breakaway_sample_count=3000,
        validation_count=30000,
        inverse_calibration_validation_rms=0.01,
        breakaway_validation_rms=0.01,
      )
      for speed in DEFAULT_SPEED_NODES_MPS
    ),
  )


def provisional() -> ProvisionalRackDynamics:
  return ProvisionalRackDynamics(
    rack_gain_deg_s2_per_torque=1500.0,
    rack_damping_per_s=8.0,
    rack_rate_resolution_deg_s=4.0,
    provenance="synthetic explicit replay seed",
  )


def behavior_policy() -> BehaviorPolicy:
  return BehaviorPolicy(
    natural_frequency_per_s=8.0,
    damping_ratio=1.0,
  )


def core_identity(name: str, token: str) -> ReplayCoreIdentity:
  return ReplayCoreIdentity(
    controller_name=name,
    core_artifact_sha256=token * 64,
    source_openpilot_commit="1" * 40,
    opendbc_commit="2" * 40,
    panda_commit="3" * 40,
  )


def frame_input(
  index: int,
  cp_bytes: bytes,
  *,
  recorded_angle: float = 0.0,
  recorded_rate: float = 0.0,
  recorded_counts: int = 0,
) -> ReplayFrameInput:
  return ReplayFrameInput(
    physical_record_index=index,
    state_sample_mono_time_ns=BASE_NS + index * 10_000_000,
    model_frame_id=index,
    recorded_rack_angle_deg=recorded_angle,
    recorded_rack_rate_deg_s=recorded_rate,
    recorded_rack_acceleration_deg_s2=0.0,
    recorded_applied_torque=recorded_counts / 409.0,
    recorded_applied_counts=recorded_counts,
    recorded_raw_request_torque=recorded_counts / 409.0,
    driver_torque=0.0,
    stiffness_factor=1.0,
    steer_ratio=15.0,
    live_torque_parameters_publication_index=-1,
    live_torque_lat_accel_factor=3.0,
    live_torque_lat_accel_offset=0.0,
    live_torque_friction=0.09,
    lateral_delay_publication_index=-1,
    lateral_delay_seconds=0.0,
    lateral_maneuver_plan_publication_index=-1,
    lateral_maneuver_desired_curvature=0.0,
    applied_count_valid=True,
    witness_resolved=True,
    control_cadence_valid=True,
    model_message_valid=True,
    model_message_alive=True,
    live_parameters_inputs_valid=True,
    live_torque_health_exact=True,
    live_torque_inputs_valid=False,
    live_torque_use_params=False,
    lateral_delay_inputs_valid=False,
    lateral_maneuver_plan_valid=False,
    intervention_onset_uncertain=False,
    standstill=False,
    car_params_bytes=cp_bytes if index == 0 else None,
  )


def decoded_route(
  *,
  physical_overrides: dict[int, tuple[float, float, int]] | None = None,
  intervention_index: int | None = None,
  second_episode: bool = False,
) -> DecodedBehaviorRoute:
  cp = synthetic_cp()
  cp_bytes = cp.to_bytes()
  mapping = nominal_mapping(cp)
  models = []
  controls = []
  count = 36
  for index in range(count):
    mono_ns = BASE_NS + index * 10_000_000
    curvature = 0.0 if index == 0 else 0.012
    times = tuple(sample * 0.05 for sample in range(20))
    models.append(SparseModelBehaviorIntent(
      plan_origin_mono_time_ns=mono_ns,
      publication_mono_time_ns=mono_ns,
      model_frame_id=index,
      plan_valid=True,
      scalar_curvature_1pm=curvature,
      scalar_action_plan_s=0.25,
      native_times_s=times,
      orientation_rates_z=tuple(curvature * 8.0 for _ in times),
      velocities_x=tuple(8.0 for _ in times),
    ))
    active = index > 0
    if second_episode and index in (18, 19):
      active = False
    override = (0.0, 0.0, 0)
    if physical_overrides is not None:
      override = physical_overrides.get(index, override)
    encoded = frame_input(
      index,
      cp_bytes,
      recorded_angle=override[0],
      recorded_rate=override[1],
      recorded_counts=override[2],
    ).to_bytes()
    onset = index == intervention_index
    controls.append(CanonicalBehaviorControlInput(
      mono_time_ns=mono_ns,
      route_time_s=index * 0.01,
      speed_mps=8.0,
      model_publication_index=index,
      live_rack_mapping=mapping,
      nominal_rack_mapping=mapping,
      core_input=encoded,
      inputs_valid=True,
      lateral_active=active,
      steering_pressed=onset,
      platform_fault=False,
      driver_intervention_onset=onset,
    ))
  return DecodedBehaviorRoute(
    route_id="synthetic-route",
    route_evidence_sha256="a" * 64,
    vehicle_identity=FINGERPRINT,
    recorded_source=BehaviorSourceIdentity(
      controller_name="stock_canonical",
      controller_artifact_sha256="b" * 64,
      source_openpilot_commit="1" * 40,
      opendbc_commit="2" * 40,
      panda_commit="3" * 40,
      evidence_schema_version=2,
    ),
    model_publications=tuple(models),
    control_inputs=tuple(controls),
    event_locators=(),
  )


def route_evidence_with_inactive_premodel_prefix() -> RouteEvidenceArtifact:
  cp_bytes = synthetic_cp().to_bytes()
  physical_frames = tuple(
    MeasuredLearningFrame(
      sample_mono_ns=BASE_NS + index * 10_000_000 - 2_000_000,
      response_mono_ns=BASE_NS + index * 10_000_000 - 1_000_000,
      applied_report_mono_ns=BASE_NS + index * 10_000_000 - 3_000_000,
      applied_effective_mono_ns=BASE_NS + index * 10_000_000 - 13_000_000,
      speed_mps=8.0,
      steering_angle_deg=float(index),
      steering_rate_deg_s=1.0,
      steering_torque=0.0,
      applied_torque=index / 409.0,
      steering_pressed=False,
      standstill=False,
      steer_fault_temporary=False,
      steer_fault_permanent=False,
      can_valid=True,
      can_timeout=False,
      lateral_active=index > 0,
      live_parameters_valid=True,
      angle_offset_valid=True,
      steer_ratio_valid=True,
      stiffness_factor_valid=True,
      angle_offset_deg=0.0,
      steer_ratio=15.0,
      stiffness_factor=1.0,
      roll_rad=0.0,
      inputs_valid=True,
    )
    for index in range(4)
  )
  models = tuple(
    ModelPublication(
      segment_index=0,
      ordinal=index,
      mono_time_ns=BASE_NS + (index + 1) * 10_000_000 - 5_000_000,
      frame_id=100 + index,
      timestamp_eof_ns=BASE_NS + (index + 1) * 10_000_000 - 6_000_000,
      scalar_curvature=0.01 + index * 0.001,
      desired_curvature_time_s=0.25,
      plan_times=(0.0, 0.05, 0.1),
      orientation_rate_z=(0.08, 0.088, 0.096),
      velocity_x=(8.0, 8.0, 8.0),
      message_valid=True,
      native_grid_valid=True,
    )
    for index in range(3)
  )
  witnesses = tuple(
    ControlsWitness(
      segment_index=0,
      ordinal=index,
      mono_time_ns=BASE_NS + index * 10_000_000,
      physical_record_index=index,
      model_publication_index=index - 1,
      live_torque_parameters_index=0,
      live_delay_index=0,
      lateral_maneuver_plan_index=0,
      poll_mono_time_ns=BASE_NS + index * 10_000_000 - 1_000_000,
      state_sample_mono_ns=BASE_NS + index * 10_000_000 - 2_000_000,
      live_parameters_mono_ns=BASE_NS + index * 10_000_000 - 2_500_000,
      car_output_report_mono_ns=BASE_NS + index * 10_000_000 - 3_000_000,
      car_output_effective_mono_ns=BASE_NS + index * 10_000_000 - 13_000_000,
      car_control_mono_ns=BASE_NS + index * 10_000_000,
      raw_request_torque=index / 409.0,
      measured_curvature=0.001 * index,
      desired_curvature=0.01,
      envelope_headroom=1.0 - index / 409.0,
      torque_output_can_count=index,
      message_valid=True,
      model_message_alive=index > 0,
      model_link_valid=index > 0,
      inputs_valid=True,
      lateral_active=index > 0,
      driver_intervening=False,
      steer_fault=False,
      intervention_onset=False,
      intervention_onset_uncertain=False,
      race_unresolved=False,
      gap_from_previous=False,
      car_control_paired=True,
      torque_output_can_valid=True,
      maneuver_plan_available=True,
      live_torque_parameters_available=True,
      live_delay_available=True,
      live_torque_parameters_checks_passed=True,
      live_torque_parameters_health_exact=True,
    )
    for index in range(4)
  )
  source = RouteEvidenceSourceIdentity(
    route_id="synthetic-evidence-route",
    route_time_origin_mono_ns=BASE_NS,
    route_segment_sha256=("e" * 64,),
    route_segment_size_bytes=(1234,),
    source_superproject_commit="1" * 40,
    source_opendbc_commit="2" * 40,
    source_panda_commit="3" * 40,
    controller_source_kind="stock_canonical",
    controller_artifact_sha256="4" * 64,
    behavior_eligible=True,
    behavior_ineligible_reason="eligible",
    vehicle_identity=FINGERPRINT,
    runtime_identity="5" * 64,
    schema_versions={"extractor": 3, "route_evidence": 2},
    preparation_provenance={"canonical": True},
    physical_plane_encoding_id="blatv2-measured-learning-frame-v1",
    physical_record_count=4,
    preparation_cache_key="6" * 64,
    controls_witness_count=4,
    unresolved_witness_count=0,
    gap_count=0,
    model_link_failure_count=1,
  )
  return RouteEvidenceArtifact(
    source,
    cp_bytes,
    b"".join(_encode_frame(frame) for frame in physical_frames),
    models,
    witnesses,
    live_torque_parameters=(LiveTorqueParametersPublication(
      0, 0, BASE_NS - 20_000_000, 3.0, 0.0, 0.09, 1,
      True, True, True,
    ),),
    live_delays=(LiveDelayPublication(
      0, 0, BASE_NS - 19_000_000, 0.12, 1, True, "valid",
    ),),
    lateral_maneuver_plans=(LateralManeuverPlanPublication(
      0, 0, BASE_NS - 18_000_000, 0.011, True,
    ),),
    event_locators=(DrivingEventLocator(
      0, 0, BASE_NS + 40_000_000, BASE_NS + 20_000_000,
      1.0, 1.0, "event-1", "lat.turnStopTurn", "warning", True,
    ),),
  )


def request(
  route: DecodedBehaviorRoute,
  *,
  modular: bool,
) -> ControllerReplayRequest:
  profile = physical_profile()
  references = []
  for control in route.control_inputs:
    model = route.model_publications[control.model_publication_index]
    references.append(derive_behavior_reference(
      model,
      BehaviorControlResponse(
        mono_time_ns=control.mono_time_ns,
        route_time_s=control.route_time_s,
        speed_mps=control.speed_mps,
        transport_delay_s=0.0,
        live_rack_mapping=control.live_rack_mapping,
        nominal_rack_mapping=control.nominal_rack_mapping,
        measured_curvature_1pm=0.0,
        measured_rack_angle_deg=0.0,
        measured_rack_rate_deg_s=0.0,
        measured_rack_accel_deg_s2=0.0,
        raw_requested_torque=0.0,
        envelope_applied_torque=0.0,
        torque_headroom=1.0,
        actuator_constrained=False,
        lateral_active=control.lateral_active,
        inputs_valid=control.inputs_valid,
        steering_pressed=control.steering_pressed,
        controller_fault=False,
        driver_intervention_onset=control.driver_intervention_onset,
      ),
    ))
  policy = behavior_policy() if modular else None
  identity = core_identity("modular" if modular else "stock", "c" if modular else "d")
  return ControllerReplayRequest(
    artifact_identity=ReplayArtifactIdentity.compose(
      ReplayRole.CANDIDATE if modular else ReplayRole.EXACT_STOCK,
      identity,
      policy,
    ),
    policy=policy,
    route=route,
    references=tuple(references),
    physical_profile=profile,
  )


def modular_core():
  return make_modular_behavior_replay_core(
    core_identity("modular", "c"),
    provisional_dynamics=provisional(),
    interface_registry=INTERFACES,
  )


def stock_core():
  return make_exact_stock_behavior_replay_core(
    core_identity("stock", "d"),
    provisional_dynamics=provisional(),
    interface_registry=INTERFACES,
  )


def test_replay_frame_input_roundtrip_is_exact_and_strict() -> None:
  cp_bytes = synthetic_cp().to_bytes()
  original = frame_input(0, cp_bytes, recorded_angle=-0.0)
  encoded = original.to_bytes()
  restored = ReplayFrameInput.from_bytes(encoded)

  assert restored == original
  assert math.copysign(1.0, restored.recorded_rack_angle_deg) == -1.0
  assert restored.to_bytes() == encoded
  with patch("json.loads", return_value={"schemaVersion": 1}):
    try:
      ReplayFrameInput.from_bytes(encoded)
    except Exception as exc:
      assert "keys" in str(exc)
    else:
      raise AssertionError("strict input schema accepted missing fields")


def test_decoder_preserves_inactive_premodel_prefix_and_exact_links() -> None:
  artifact = route_evidence_with_inactive_premodel_prefix()
  decoder = make_behavior_route_evidence_decoder(
    provisional_dynamics=provisional(),
    interface_registry=INTERFACES,
  )
  decoded = decoder(artifact, physical_profile())

  assert len(decoded.control_inputs) == 4
  assert decoded.control_inputs[0].model_publication_index is None
  assert not decoded.control_inputs[0].inputs_valid
  assert not decoded.control_inputs[0].lateral_active
  assert tuple(
    control.model_publication_index for control in decoded.control_inputs[1:]
  ) == (0, 1, 2)
  first = ReplayFrameInput.from_bytes(decoded.control_inputs[0].core_input)
  active = ReplayFrameInput.from_bytes(decoded.control_inputs[1].core_input)
  assert first.car_params_bytes == synthetic_cp().to_bytes()
  assert active.car_params_bytes is None
  assert active.recorded_applied_counts == 1
  assert active.live_torque_health_exact
  assert active.live_torque_inputs_valid
  assert decoded.event_locators[0].event_type == "lat.turnStopTurn"
  assert decoded.route_evidence_sha256 == artifact.sha256


def test_exact_stock_timing_scalar_is_pinned_to_source() -> None:
  modeld_path = Path(__file__).parents[2] / "modeld" / "modeld.py"
  tree = ast.parse(modeld_path.read_text(encoding="utf-8"))
  assignments = {
    node.targets[0].id: ast.literal_eval(node.value)
    for node in tree.body
    if isinstance(node, ast.Assign)
    and len(node.targets) == 1
    and isinstance(node.targets[0], ast.Name)
    and node.targets[0].id == "LAT_SMOOTH_SECONDS"
  }
  assert assignments["LAT_SMOOTH_SECONDS"] == SOURCE_LAT_SMOOTH_SECONDS


def test_count_envelope_preserves_asymmetry_and_sign_crossing_budget() -> None:
  limits = RuntimeTorqueLimits(409, 4, 7, 1, 50, 2, 1, True)
  assert apply_torque_envelope_counts(limits, 409, 0, 0.0) == 4
  assert apply_torque_envelope_counts(limits, 0, 100, 0.0) == 93
  assert apply_torque_envelope_counts(limits, -409, 3, 0.0) == -4
  assert apply_torque_envelope_counts(limits, 409, -3, 0.0) == 4


def test_modular_replay_calls_existing_core_and_is_exact_aa() -> None:
  route = decoded_route()
  replay_request = request(route, modular=True)
  original_update = ModularControllerCore.update
  calls = []

  def traced_update(self, *args, **kwargs):
    calls.append(kwargs["scalar_curvature"])
    return original_update(self, *args, **kwargs)

  with patch.object(ModularControllerCore, "update", traced_update):
    first = tuple(modular_core().replay_route(replay_request))
  second = tuple(modular_core().replay_route(replay_request))

  assert len(calls) == len(route.control_inputs) - 1
  assert first == second
  assert all(math.isfinite(output.raw_requested_torque) for output in first)
  assert any(output.response_eligible for output in first)


def test_stock_replay_constructs_actual_source_controller() -> None:
  replay_request = request(decoded_route(), modular=False)
  with patch(
    "openpilot.selfdrive.controls.lib.blatv2.behavior_replay.fresh_stock_torque_controller",
    wraps=fresh_stock_torque_controller,
  ) as constructor:
    outputs = tuple(stock_core().replay_route(replay_request))

  assert constructor.call_count == 1
  assert any(output.response_eligible for output in outputs)
  assert all(math.isfinite(output.raw_requested_torque) for output in outputs)


def test_recorded_response_after_bootstrap_cannot_leash_candidate() -> None:
  baseline_route = decoded_route()
  perturbed_route = decoded_route(physical_overrides={
    index: (300.0 - index, -80.0 + index, 350)
    for index in range(2, 36)
  })
  baseline = tuple(modular_core().replay_route(request(baseline_route, modular=True)))
  perturbed = tuple(modular_core().replay_route(request(perturbed_route, modular=True)))

  assert baseline[1:] == perturbed[1:]


def test_intervention_censors_episode_and_next_boundary_rebootstraps() -> None:
  route = decoded_route(intervention_index=10, second_episode=True)
  outputs = tuple(modular_core().replay_route(request(route, modular=True)))

  assert all(output.response_eligible for output in outputs[1:10])
  assert all(not output.response_eligible for output in outputs[10:20])
  assert outputs[20].response_eligible


def test_uncertain_intervention_onset_conservatively_censors_episode() -> None:
  route = decoded_route(second_episode=True)
  controls = list(route.control_inputs)
  uncertain = ReplayFrameInput.from_bytes(controls[10].core_input)
  controls[10] = replace(
    controls[10],
    core_input=replace(
      uncertain,
      intervention_onset_uncertain=True,
    ).to_bytes(),
  )
  route = replace(route, control_inputs=tuple(controls))
  outputs = tuple(modular_core().replay_route(request(route, modular=True)))

  assert all(output.response_eligible for output in outputs[1:10])
  assert all(not output.response_eligible for output in outputs[10:20])
  assert outputs[20].response_eligible


def test_exact_stock_faults_when_submaster_health_is_not_exact() -> None:
  route = decoded_route()
  controls = list(route.control_inputs)
  input_one = ReplayFrameInput.from_bytes(controls[1].core_input)
  controls[1] = replace(
    controls[1],
    core_input=replace(
      input_one,
      live_torque_health_exact=False,
    ).to_bytes(),
  )
  route = replace(route, control_inputs=tuple(controls))
  outputs = tuple(stock_core().replay_route(request(route, modular=False)))

  assert outputs[1].controller_fault
  assert not outputs[1].response_eligible


def test_whole_route_callback_resets_and_is_thread_safe() -> None:
  replay_request = request(decoded_route(), modular=True)
  replay = modular_core()
  expected = tuple(replay.replay_route(replay_request))
  tuple(replay.replay_route(request(decoded_route(physical_overrides={1: (5.0, 0.0, 20)}), modular=True)))
  assert tuple(replay.replay_route(replay_request)) == expected

  with ThreadPoolExecutor(max_workers=4) as executor:
    results = tuple(executor.map(
      lambda _: tuple(replay.replay_route(replay_request)),
      range(8),
    ))
  assert all(result == expected for result in results)


def test_nonfinite_core_output_is_finite_fault_and_not_eligible() -> None:
  route = decoded_route()
  replay_request = request(route, modular=True)
  original_update = ModularControllerCore.update
  call_count = 0

  def one_nonfinite(self, *args, **kwargs):
    nonlocal call_count
    result = original_update(self, *args, **kwargs)
    call_count += 1
    if call_count == 5:
      result.raw_torque = math.nan
    return result

  with patch.object(ModularControllerCore, "update", one_nonfinite):
    outputs = tuple(modular_core().replay_route(replay_request))

  failed = outputs[5]
  assert math.isfinite(failed.raw_requested_torque)
  assert failed.controller_fault
  assert not failed.response_eligible


def test_core_input_ignores_later_recorded_request_and_applied_values() -> None:
  route = decoded_route()
  controls = list(route.control_inputs)
  for index in range(2, len(controls)):
    decoded = ReplayFrameInput.from_bytes(controls[index].core_input)
    controls[index] = replace(
      controls[index],
      core_input=replace(
        decoded,
        recorded_applied_torque=-0.9,
        recorded_applied_counts=-368,
        recorded_raw_request_torque=0.95,
      ).to_bytes(),
    )
  perturbed = replace(route, control_inputs=tuple(controls))
  baseline = tuple(stock_core().replay_route(request(route, modular=False)))
  changed = tuple(stock_core().replay_route(request(perturbed, modular=False)))

  assert baseline[1:] == changed[1:]
