from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import ast
import hashlib
import math
from pathlib import Path
import struct
from unittest.mock import patch

from opendbc.car.structs import car
from opendbc.car.hyundai.steering_request import MAX_ANGLE, MAX_ANGLE_FRAMES
from opendbc.car.vehicle_model import VehicleModel

from openpilot.selfdrive.modeld.constants import ModelConstants
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
  BehaviorReferenceAtControl,
  BehaviorSourceIdentity,
  SparseModelBehaviorIntent,
  derive_behavior_reference,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import BehaviorPolicy
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay import (
  BehaviorReplayStepper,
  BehaviorReplayError,
  ReplayFrameInput,
  SOURCE_LAT_SMOOTH_SECONDS,
  behavior_scenario_set_identity,
  make_behavior_route_evidence_decoder,
  make_behavior_scenario_route_evidence_decoder,
  make_exact_stock_behavior_replay_core,
  make_modular_behavior_replay_core,
  reviewed_replay_core_identity,
  validate_behavior_scenario_active_frame,
  validate_reviewed_behavior_replay_core,
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
from openpilot.selfdrive.controls.lib.blatv2.counterfactual_plant import (
  CounterfactualPlantMember,
  step_counterfactual_plant,
)
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
        cross_fit_route_count=30000,
        full_fit_candidate_rms=0.01,
        breakaway_full_fit_candidate_rms=0.01,
      )
      for speed in DEFAULT_SPEED_NODES_MPS
    ),
  )


def physical_profile_with_delay(delay_s: float) -> VehicleCalibrationProfile:
  base = physical_profile()
  return replace(
    base,
    nodes=tuple(
      replace(node, parameters=replace(node.parameters, transport_delay_s=delay_s))
      for node in base.nodes
    ),
  )


def provisional() -> ProvisionalRackDynamics:
  return ProvisionalRackDynamics(
    rack_gain_deg_s2_per_torque=1500.0,
    rack_damping_per_s=8.0,
    rack_rate_resolution_deg_s=4.0,
    provenance="synthetic explicit replay seed",
  )


def plant_member(
  *,
  gain: float = 1500.0,
  damping: float = 8.0,
  delay: float = 0.0,
  load: float = 0.0,
) -> CounterfactualPlantMember:
  return CounterfactualPlantMember.create(
    rack_gain_deg_s2_per_torque=gain,
    rack_damping_per_s=damping,
    delay_offset_s=delay,
    unresolved_load_torque=load,
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


def reviewed_core_identity(*, modular: bool) -> ReplayCoreIdentity:
  return reviewed_replay_core_identity(
    exact_stock=not modular,
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
    times = tuple(ModelConstants.T_IDXS)
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
      plan_times=tuple(ModelConstants.T_IDXS),
      orientation_rate_z=tuple(
        (0.01 + index * 0.001) * 8.0 for _ in ModelConstants.T_IDXS
      ),
      velocity_x=tuple(8.0 for _ in ModelConstants.T_IDXS),
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
      lateral_maneuver_plan_index=-1,
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
      steering_request_fault_avoidance_counter=0,
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
      maneuver_plan_available=False,
      live_torque_parameters_available=True,
      live_delay_available=True,
      live_torque_parameters_checks_passed=True,
      live_torque_parameters_health_exact=True,
      steering_request_active=True,
      steering_request_active_valid=True,
      steering_request_fault_avoidance_counter_valid=True,
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
    schema_versions={"extractor": 3, "route_evidence": 4},
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
    lateral_maneuver_plans=(),
    event_locators=(DrivingEventLocator(
      0, 0, BASE_NS + 40_000_000, BASE_NS + 20_000_000,
      1.0, 1.0, "event-1", "lat.turnStopTurn", "warning", True,
    ),),
  )


def replace_route_evidence(
  artifact: RouteEvidenceArtifact,
  *,
  source: RouteEvidenceSourceIdentity | None = None,
  witnesses: tuple[ControlsWitness, ...] | None = None,
) -> RouteEvidenceArtifact:
  return RouteEvidenceArtifact(
    artifact.source_identity if source is None else source,
    bytes(artifact.car_params_bytes),
    bytes(artifact.physical_bytes),
    artifact.model_publications,
    artifact.control_witnesses if witnesses is None else witnesses,
    artifact.live_torque_parameters,
    artifact.live_delays,
    artifact.lateral_maneuver_plans,
    artifact.event_locators,
  )


def output_bytes(outputs) -> bytes:
  return b"".join(
    struct.pack(
      "<Q9d5?",
      output.mono_time_ns,
      output.measured_curvature_1pm,
      output.measured_rack_angle_deg,
      output.measured_rack_rate_deg_s,
      output.measured_rack_accel_deg_s2,
      output.raw_requested_torque,
      output.planned_requested_torque,
      output.reachable_envelope_torque,
      output.envelope_applied_torque,
      output.torque_headroom,
      output.actuator_constrained,
      output.steering_request_active,
      output.maximum_authority_required,
      output.controller_fault,
      output.response_eligible,
    )
    for output in outputs
  )


def request(
  route: DecodedBehaviorRoute,
  *,
  modular: bool,
) -> ControllerReplayRequest:
  profile = physical_profile()
  references = []
  for control in route.control_inputs:
    if control.model_publication_index is None:
      references.append(BehaviorReferenceAtControl(
        model_publication_mono_time_ns=0,
        plan_time_now_s=0.0,
        physical_effect_plan_s=0.0,
        scalar_curvature_1pm=0.0,
        anchored_curvature_1pm=0.0,
        anchored_curvature_rate_1pm_s=0.0,
        anchored_curvature_accel_1pm_s2=0.0,
        desired_rack_angle_deg=0.0,
        desired_rack_rate_deg_s=0.0,
        desired_rack_accel_deg_s2=0.0,
        valid=False,
      ))
      continue
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
        planned_requested_torque=0.0,
        reachable_envelope_torque=0.0,
        envelope_applied_torque=0.0,
        torque_headroom=1.0,
        actuator_constrained=False,
        steering_request_active=True,
        maximum_authority_required=False,
        lateral_active=control.lateral_active,
        inputs_valid=control.inputs_valid,
        steering_pressed=control.steering_pressed,
        controller_fault=False,
        driver_intervention_onset=control.driver_intervention_onset,
      ),
    ))
  policy = behavior_policy() if modular else None
  identity = reviewed_core_identity(modular=modular)
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
    reviewed_core_identity(modular=True),
    provisional_dynamics=provisional(),
    interface_registry=INTERFACES,
  )


def stock_core():
  return make_exact_stock_behavior_replay_core(
    reviewed_core_identity(modular=False),
    provisional_dynamics=provisional(),
    interface_registry=INTERFACES,
  )


def test_reviewed_core_authority_rejects_injected_vehicle_interface() -> None:
  identity = reviewed_core_identity(modular=False)
  injected = make_exact_stock_behavior_replay_core(
    identity,
    provisional_dynamics=provisional(),
    interface_registry=INTERFACES,
  )
  try:
    validate_reviewed_behavior_replay_core(injected, exact_stock=True)
  except BehaviorReplayError as error:
    assert "reviewed execution adapter" in str(error)
  else:
    raise AssertionError("injected vehicle interface acquired stock authority")

  production = make_exact_stock_behavior_replay_core(
    identity,
    provisional_dynamics=provisional(),
  )
  validate_reviewed_behavior_replay_core(production, exact_stock=True)


def streaming_stepper(
  replay_request: ControllerReplayRequest,
  *,
  plant_member: CounterfactualPlantMember | None = None,
) -> tuple[BehaviorReplayStepper, tuple[ReplayFrameInput, ...]]:
  inputs = tuple(
    ReplayFrameInput.from_bytes(control.core_input)
    for control in replay_request.route.control_inputs
  )
  return BehaviorReplayStepper(
    vehicle_identity=replay_request.route.vehicle_identity,
    physical_profile=replay_request.physical_profile,
    policy=replay_request.policy,
    first_frame_input=inputs[0],
    nominal_rack_mapping=replay_request.route.control_inputs[0].nominal_rack_mapping,
    provisional_dynamics=provisional(),
    plant_member=plant_member,
    interface_registry=INTERFACES,
  ), inputs


def stream_request(
  replay_request: ControllerReplayRequest,
  stepper: BehaviorReplayStepper | None = None,
) -> tuple:
  if stepper is None:
    stepper, inputs = streaming_stepper(replay_request)
  else:
    inputs = tuple(
      ReplayFrameInput.from_bytes(control.core_input)
      for control in replay_request.route.control_inputs
    )
  outputs = []
  for control, frame, reference in zip(
    replay_request.route.control_inputs,
    inputs,
    replay_request.references,
    strict=True,
  ):
    model = (
      None
      if control.model_publication_index is None
      else replay_request.route.model_publications[control.model_publication_index]
    )
    outputs.append(stepper.step(
      control=control,
      frame_input=frame,
      model_intent=model,
      reference=reference,
    ))
  return tuple(outputs)


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


def test_replay_frame_input_validates_live_torque_values_only_when_consumed() -> None:
  cp_bytes = synthetic_cp().to_bytes()
  ignored = replace(
    frame_input(0, cp_bytes),
    live_torque_friction=-3.0,
    live_torque_inputs_valid=False,
    live_torque_use_params=True,
  )
  restored = ReplayFrameInput.from_bytes(ignored.to_bytes())
  assert restored.live_torque_friction == -3.0
  assert not restored.live_torque_inputs_valid

  def route_with_ignored_friction(friction: float):
    route = decoded_route()
    controls = list(route.control_inputs)
    payload = ReplayFrameInput.from_bytes(controls[1].core_input)
    controls[1] = replace(
      controls[1],
      core_input=replace(
        payload,
        live_torque_friction=friction,
        live_torque_inputs_valid=False,
        live_torque_use_params=True,
      ).to_bytes(),
    )
    return replace(route, control_inputs=tuple(controls))

  first = tuple(stock_core().replay_route(request(route_with_ignored_friction(-3.0), modular=False)))
  second = tuple(stock_core().replay_route(request(route_with_ignored_friction(-30.0), modular=False)))
  assert first == second

  try:
    replace(ignored, live_torque_inputs_valid=True)
  except ValueError as error:
    assert "consumed live torque friction" in str(error)
  else:
    raise AssertionError("consumed negative live torque friction was accepted")


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


def test_scenario_decoder_preserves_unverified_recorded_source_without_relabeling() -> None:
  base = route_evidence_with_inactive_premodel_prefix()
  source = replace(
    base.source_identity,
    controller_source_kind="ineligible",
    behavior_eligible=False,
    behavior_ineligible_reason="unverified_stock_composition",
  )
  artifact = replace_route_evidence(base, source=source)
  strict_decoder = make_behavior_route_evidence_decoder(
    provisional_dynamics=provisional(),
    interface_registry=INTERFACES,
  )
  scenario_decoder = make_behavior_scenario_route_evidence_decoder(
    provisional_dynamics=provisional(),
    interface_registry=INTERFACES,
  )

  try:
    strict_decoder(artifact, physical_profile())
  except BehaviorReplayError as error:
    assert "behavior-ineligible" in str(error)
  else:
    raise AssertionError("strict behavior decoder admitted ineligible evidence")

  decoded = scenario_decoder(artifact, physical_profile())
  provenance = decoded.scenario_provenance
  assert provenance is not None
  assert not provenance.recorded_behavior_eligible
  assert provenance.recorded_behavior_ineligible_reason == "unverified_stock_composition"
  assert provenance.recorded_source.controller_name == "ineligible"
  assert provenance.recorded_source.controller_name != "stock_canonical"
  assert provenance.route_evidence_sha256 == artifact.sha256


def test_active_scenario_validation_matches_cert_v5_physical_and_finite_contract() -> None:
  artifact = route_evidence_with_inactive_premodel_prefix()
  physical = tuple(artifact.iter_physical_frames())[1]
  witness = artifact.control_witnesses[1]
  model = artifact.model_publications[0]
  torque = artifact.live_torque_parameters[0]
  delay = artifact.live_delays[0]

  def corrupted(value, field: str, replacement=math.nan):
    result = replace(value)
    object.__setattr__(result, field, replacement)
    return result

  def validate(**changes) -> None:
    validate_behavior_scenario_active_frame(
      physical_record_index=1,
      rack_acceleration_valid=True,
      witness=changes.get("witness", witness),
      physical=changes.get("physical", physical),
      model=changes.get("model", model),
      live_torque=changes.get("live_torque", torque),
      live_delay=changes.get("live_delay", delay),
      maneuver=None,
    )

  validate()
  rejected = (
    {
      "witness": replace(
        witness,
        car_control_paired=False,
        car_control_mono_ns=-1,
      ),
    },
    {"physical": replace(physical, inputs_valid=False)},
    {
      "witness": replace(witness, inputs_valid=False),
      "physical": replace(physical, inputs_valid=False),
    },
    {"model": corrupted(model, "desired_curvature_time_s")},
    {"model": corrupted(model, "plan_times", (math.nan, 0.05, 0.1))},
    {"live_torque": corrupted(torque, "lat_accel_factor")},
    {"live_torque": corrupted(torque, "lat_accel_offset")},
    {"live_torque": corrupted(torque, "friction")},
    {"live_delay": corrupted(delay, "lateral_delay_s")},
  )
  for values in rejected:
    try:
      validate(**values)
    except BehaviorReplayError:
      pass
    else:
      raise AssertionError(f"cert-v5-invalid active scenario was admitted: {values}")


def test_scenario_identity_binds_ordered_recorded_sources() -> None:
  base = route_evidence_with_inactive_premodel_prefix()
  source = replace(
    base.source_identity,
    controller_source_kind="ineligible",
    behavior_eligible=False,
    behavior_ineligible_reason="unverified_stock_composition",
  )
  artifact = replace_route_evidence(base, source=source)
  decoder = make_behavior_scenario_route_evidence_decoder(
    provisional_dynamics=provisional(),
    interface_registry=INTERFACES,
  )
  first = decoder(artifact, physical_profile())
  assert first.scenario_provenance is not None
  second_provenance = replace(
    first.scenario_provenance,
    route_id="synthetic-evidence-route-2",
    route_evidence_sha256="7" * 64,
  )
  second = replace(
    first,
    route_id=second_provenance.route_id,
    route_evidence_sha256=second_provenance.route_evidence_sha256,
    scenario_provenance=second_provenance,
  )

  forward = behavior_scenario_set_identity((first, second))
  reverse = behavior_scenario_set_identity((second, first))
  assert forward.sha256 != reverse.sha256
  assert [
    value["recordedBehaviorIneligibleReason"]
    for value in forward.to_dict()["scenarioSources"]
  ] == ["unverified_stock_composition", "unverified_stock_composition"]


def test_recorded_controller_reason_cannot_veto_valid_scenario_replay() -> None:
  base = route_evidence_with_inactive_premodel_prefix()
  decoder = make_behavior_scenario_route_evidence_decoder(
    provisional_dynamics=provisional(),
    interface_registry=INTERFACES,
  )
  recorded_ineligible_source = replace(
    base.source_identity,
    controller_source_kind="ineligible",
    behavior_eligible=False,
    behavior_ineligible_reason="lateral_maneuver_plan_missing",
  )
  recorded_ineligible = replace_route_evidence(base, source=recorded_ineligible_source)
  decoded = decoder(recorded_ineligible, physical_profile())
  assert decoded.scenario_provenance is not None
  assert not decoded.scenario_provenance.recorded_behavior_eligible
  assert (
    decoded.scenario_provenance.recorded_behavior_ineligible_reason
    == "lateral_maneuver_plan_missing"
  )
  for core, modular in ((stock_core(), False), (modular_core(), True)):
    outputs = tuple(core.replay_route(request(decoded, modular=modular)))
    assert len(outputs) == len(decoded.control_inputs)
    assert any(output.response_eligible for output in outputs)

  scenario_source = replace(
    base.source_identity,
    controller_source_kind="ineligible",
    behavior_eligible=False,
    behavior_ineligible_reason="unverified_stock_composition",
    unresolved_witness_count=1,
  )
  unresolved_witnesses = list(base.control_witnesses)
  unresolved_witnesses[1] = replace(unresolved_witnesses[1], race_unresolved=True)
  unresolved = replace_route_evidence(
    base,
    source=scenario_source,
    witnesses=tuple(unresolved_witnesses),
  )
  try:
    decoder(unresolved, physical_profile())
  except BehaviorReplayError as error:
    assert "unresolved" in str(error)
  else:
    raise AssertionError("scenario decoder admitted unresolved active evidence")

  inactive_witnesses = tuple(
    replace(witness, lateral_active=False)
    for witness in base.control_witnesses
  )
  inactive = replace_route_evidence(
    base,
    source=replace(scenario_source, unresolved_witness_count=0),
    witnesses=inactive_witnesses,
  )
  try:
    decoder(inactive, physical_profile())
  except BehaviorReplayError as error:
    assert "no active lateral" in str(error)
  else:
    raise AssertionError("scenario decoder admitted evidence without lateral activity")


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
  if "LAT_SMOOTH_SECONDS" not in assignments:
    assert any(
      isinstance(node, ast.ImportFrom)
      and node.module == "openpilot.selfdrive.modeld.timing"
      and any(alias.name == "LAT_SMOOTH_SECONDS" for alias in node.names)
      for node in tree.body
    )
    timing_path = modeld_path.with_name("timing.py")
    timing_tree = ast.parse(timing_path.read_text(encoding="utf-8"))
    assignments = {
      node.targets[0].id: ast.literal_eval(node.value)
      for node in timing_tree.body
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
  requests = []

  def traced_update(self, *args, **kwargs):
    calls.append(kwargs["scalar_curvature"])
    result = original_update(self, *args, **kwargs)
    requests.append((result.raw_torque, result.planned_torque))
    return result

  with patch.object(ModularControllerCore, "update", traced_update):
    first = tuple(modular_core().replay_route(replay_request))
  second = tuple(modular_core().replay_route(replay_request))

  assert len(calls) == len(route.control_inputs) - 1
  assert first == second
  assert any(raw != planned for raw, planned in requests)
  assert tuple(output.raw_requested_torque for output in first[1:]) == tuple(
    raw for raw, _ in requests
  )
  assert tuple(output.planned_requested_torque for output in first[1:]) == tuple(
    planned for _, planned in requests
  )
  assert any(
    output.raw_requested_torque != output.planned_requested_torque
    for output in first[1:]
  )
  limits = RuntimeTorqueLimits(409, 4, 7, 1, 50, 2, 1, True)
  previous_applied_counts = round(first[0].envelope_applied_torque * limits.steer_max)
  for control, output in zip(route.control_inputs[1:], first[1:], strict=True):
    frame = ReplayFrameInput.from_bytes(control.core_input)
    expected_applied_counts = apply_torque_envelope_counts(
      limits,
      round(output.planned_requested_torque * limits.steer_max),
      previous_applied_counts,
      frame.driver_torque,
    )
    assert output.envelope_applied_torque == expected_applied_counts / limits.steer_max
    previous_applied_counts = expected_applied_counts
  assert all(math.isfinite(output.raw_requested_torque) for output in first)
  assert any(output.response_eligible for output in first)


def test_replay_preserves_raw_planned_and_applied_as_distinct_values() -> None:
  original_update = ModularControllerCore.update

  def forced_requests(self, *args, **kwargs):
    result = original_update(self, *args, **kwargs)
    if result.valid:
      result.raw_torque = -1.0
      result.planned_torque = -0.001
    return result

  replay_request = request(decoded_route(), modular=True)
  with patch.object(ModularControllerCore, "update", forced_requests):
    outputs = tuple(modular_core().replay_route(replay_request))

  output = outputs[1]
  assert output.raw_requested_torque == -1.0
  assert output.planned_requested_torque == -0.001
  assert output.envelope_applied_torque == 0.0
  assert len({
    output.raw_requested_torque,
    output.planned_requested_torque,
    output.envelope_applied_torque,
  }) == 3
  assert output.reachable_envelope_torque == -4 / 409
  assert output.maximum_authority_required


def test_whole_route_adapters_match_streaming_core_and_legacy_bytes() -> None:
  # These bytes bind raw, planned, reachable, applied, and request-state
  # semantics; a field loss or reinterpretation must change the golden.
  contract_sha256 = {
    False: "b492d5bde172ed7e26ddedd53ba38af4ce9f685f0f058e4d073335b1f123d3a8",
    True: "f565652bd4c9ce34a097b5efcd2d4f2cd1ff106747137f574c3150c1a3535dc0",
  }
  for replay_core, modular in ((stock_core(), False), (modular_core(), True)):
    replay_request = request(decoded_route(), modular=modular)
    whole_route = tuple(replay_core.replay_route(replay_request))
    streamed = stream_request(replay_request)
    assert output_bytes(streamed) == output_bytes(whole_route)
    assert hashlib.sha256(output_bytes(streamed)).hexdigest() == contract_sha256[modular]


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


def test_exact_stock_replay_uses_detected_vehicle_envelope() -> None:
  outputs = tuple(stock_core().replay_route(request(decoded_route(), modular=False)))

  # The synthetic detected port declares the Palisade 409/4/7 contract. The
  # stock request exceeds one build step, so its first applied command proves
  # replay used that runtime envelope rather than a controller-side literal.
  assert outputs[1].envelope_applied_torque == -4 / 409
  assert outputs[1].actuator_constrained


def test_recorded_response_after_bootstrap_cannot_leash_candidate() -> None:
  baseline_route = decoded_route()
  perturbed_route = decoded_route(physical_overrides={
    index: (300.0 - index, -80.0 + index, 350)
    for index in range(2, 36)
  })
  baseline = tuple(modular_core().replay_route(request(baseline_route, modular=True)))
  perturbed = tuple(modular_core().replay_route(request(perturbed_route, modular=True)))

  assert baseline[1:] == perturbed[1:]


def test_controller_sees_can_torque_while_rack_response_obeys_positive_delay() -> None:
  replay_request = request(decoded_route(), modular=True)
  inputs = tuple(
    ReplayFrameInput.from_bytes(control.core_input)
    for control in replay_request.route.control_inputs
  )
  stepper = BehaviorReplayStepper(
    vehicle_identity=replay_request.route.vehicle_identity,
    physical_profile=physical_profile_with_delay(0.02),
    policy=replay_request.policy,
    first_frame_input=inputs[0],
    nominal_rack_mapping=replay_request.route.control_inputs[0].nominal_rack_mapping,
    provisional_dynamics=provisional(),
    plant_member=plant_member(delay=0.0),
    interface_registry=INTERFACES,
  )
  observed_controller_applied: list[float] = []
  original_update = ModularControllerCore.update

  def traced_update(self, *args, **kwargs):
    observed_controller_applied.append(kwargs["recorded_applied_torque"])
    return original_update(self, *args, **kwargs)

  outputs = []
  rack_effective = []
  with patch.object(ModularControllerCore, "update", traced_update):
    for control, frame_input, reference in zip(
      replay_request.route.control_inputs,
      inputs,
      replay_request.references,
      strict=True,
    ):
      model = (
        None
        if control.model_publication_index is None
        else replay_request.route.model_publications[control.model_publication_index]
      )
      outputs.append(stepper.step(
        control=control,
        frame_input=frame_input,
        model_intent=model,
        reference=reference,
      ))
      rack_effective.append(
        None
        if stepper.state is None
        else stepper.delay_line.latest_rack_effective_torque
      )

  assert observed_controller_applied == [
    output.envelope_applied_torque for output in outputs[:-1]
  ]
  assert any(
    effective is not None and effective != output.envelope_applied_torque
    for effective, output in zip(rack_effective[1:], outputs[1:], strict=True)
  )


def test_replay_request_cut_carries_can_count_but_commits_zero_torque() -> None:
  replay_request = request(
    decoded_route(physical_overrides={1: (MAX_ANGLE + 1.0, 0.0, 200)}),
    modular=True,
  )
  stepper, inputs = streaming_stepper(replay_request)

  for index in (0, 1):
    control = replay_request.route.control_inputs[index]
    stepper.step(
      control=control,
      frame_input=inputs[index],
      model_intent=replay_request.route.model_publications[index],
      reference=replay_request.references[index],
    )
  stepper.steering_request_fault_avoidance_counter = MAX_ANGLE_FRAMES

  control = replay_request.route.control_inputs[2]
  output = stepper.step(
    control=control,
    frame_input=inputs[2],
    model_intent=replay_request.route.model_publications[2],
    reference=replay_request.references[2],
  )

  assert output.envelope_applied_torque != 0.0
  assert stepper.previous_applied_counts != 0
  assert not stepper.previous_steering_request_active
  assert stepper.delay_line.latest_rack_effective_torque == 0.0


def test_independent_plant_member_changes_response_without_changing_controller_bytes() -> None:
  replay_request = request(decoded_route(), modular=True)
  slow, _ = streaming_stepper(
    replay_request,
    plant_member=plant_member(gain=500.0, damping=12.0),
  )
  fast, _ = streaming_stepper(
    replay_request,
    plant_member=plant_member(gain=3000.0, damping=4.0),
  )
  slow_outputs = stream_request(replay_request, slow)
  fast_outputs = stream_request(replay_request, fast)
  assert slow.plant_member.member_id != fast.plant_member.member_id
  assert output_bytes(slow_outputs) != output_bytes(fast_outputs)


def test_platform_fault_latches_and_freezes_until_next_episode() -> None:
  replay_request = request(decoded_route(), modular=True)
  controls = list(replay_request.route.control_inputs)
  controls[7] = replace(controls[7], platform_fault=True)
  replay_request = replace(
    replay_request,
    route=replace(replay_request.route, control_inputs=tuple(controls)),
  )
  outputs = stream_request(replay_request)
  assert all(output.controller_fault for output in outputs[7:])
  assert all(not output.response_eligible for output in outputs[7:])
  assert all(
    (output.measured_rack_angle_deg, output.measured_rack_rate_deg_s)
    == (outputs[7].measured_rack_angle_deg, outputs[7].measured_rack_rate_deg_s)
    for output in outputs[8:]
  )


def test_intervention_censors_episode_and_next_boundary_rebootstraps() -> None:
  route = decoded_route(intervention_index=10, second_episode=True)
  outputs = tuple(modular_core().replay_route(request(route, modular=True)))

  assert all(output.response_eligible for output in outputs[1:10])
  assert all(not output.response_eligible for output in outputs[10:20])
  assert outputs[20].response_eligible


def test_streaming_episode_boundary_intervention_and_rebootstrap() -> None:
  route = decoded_route(
    physical_overrides={20: (12.0, -3.0, 80)},
    intervention_index=10,
    second_episode=True,
  )
  outputs = stream_request(request(route, modular=True))

  assert all(output.response_eligible for output in outputs[1:10])
  assert all(not output.response_eligible for output in outputs[10:20])
  assert outputs[20].response_eligible
  assert outputs[20].measured_rack_angle_deg == 12.0
  assert outputs[20].measured_rack_rate_deg_s == -3.0


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


def test_streaming_invalid_gap_faults_until_a_fresh_episode() -> None:
  route = decoded_route(second_episode=True)
  controls = list(route.control_inputs)
  invalid = ReplayFrameInput.from_bytes(controls[7].core_input)
  controls[7] = replace(
    controls[7],
    core_input=replace(invalid, control_cadence_valid=False).to_bytes(),
  )
  route = replace(route, control_inputs=tuple(controls))
  outputs = stream_request(request(route, modular=True))

  assert all(output.controller_fault for output in outputs[7:18])
  assert all(not output.response_eligible for output in outputs[7:20])
  assert outputs[20].response_eligible
  assert not outputs[20].controller_fault


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
      result.planned_torque = math.nan
    return result

  with patch.object(ModularControllerCore, "update", one_nonfinite):
    outputs = tuple(modular_core().replay_route(replay_request))

  failed = outputs[5]
  assert math.isfinite(failed.raw_requested_torque)
  assert failed.controller_fault
  assert not failed.response_eligible


def test_final_frame_plant_failure_is_reported_on_that_frame() -> None:
  replay_request = request(decoded_route(), modular=True)
  call_count = 0
  expected_calls = len(replay_request.route.control_inputs) - 1

  def fail_last(**kwargs):
    nonlocal call_count
    call_count += 1
    if call_count == expected_calls:
      raise ValueError("synthetic final plant failure")
    return step_counterfactual_plant(**kwargs)

  with patch(
    "openpilot.selfdrive.controls.lib.blatv2.behavior_replay.step_counterfactual_plant",
    side_effect=fail_last,
  ):
    outputs = tuple(modular_core().replay_route(replay_request))

  assert call_count == expected_calls
  assert outputs[-1].controller_fault
  assert not outputs[-1].response_eligible


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


def test_recorded_request_cannot_change_stock_or_candidate_counterfactual_bytes() -> None:
  route = decoded_route(second_episode=True)
  controls = list(route.control_inputs)
  for index, control in enumerate(controls):
    decoded = ReplayFrameInput.from_bytes(control.core_input)
    controls[index] = replace(
      control,
      core_input=replace(
        decoded,
        recorded_raw_request_torque=(-0.99 if index % 2 else 0.99),
      ).to_bytes(),
    )
  perturbed = replace(route, control_inputs=tuple(controls))

  for replay_core, modular in ((stock_core(), False), (modular_core(), True)):
    baseline = tuple(replay_core.replay_route(request(route, modular=modular)))
    changed = tuple(replay_core.replay_route(request(perturbed, modular=modular)))
    assert output_bytes(baseline) == output_bytes(changed)


def test_streaming_recorded_request_is_metamorphically_inert() -> None:
  route = decoded_route(second_episode=True)
  controls = list(route.control_inputs)
  for index, control in enumerate(controls):
    decoded = ReplayFrameInput.from_bytes(control.core_input)
    controls[index] = replace(
      control,
      core_input=replace(
        decoded,
        recorded_raw_request_torque=(-0.87 if index % 3 else 0.91),
      ).to_bytes(),
    )
  perturbed = replace(route, control_inputs=tuple(controls))

  for modular in (False, True):
    baseline = stream_request(request(route, modular=modular))
    changed = stream_request(request(perturbed, modular=modular))
    assert output_bytes(baseline) == output_bytes(changed)


def test_streaming_reset_is_fresh_route_aa() -> None:
  for modular in (False, True):
    replay_request = request(decoded_route(second_episode=True), modular=modular)
    stepper, _ = streaming_stepper(replay_request)
    first = stream_request(replay_request, stepper)
    stepper.reset()
    second = stream_request(replay_request, stepper)
    assert output_bytes(first) == output_bytes(second)
