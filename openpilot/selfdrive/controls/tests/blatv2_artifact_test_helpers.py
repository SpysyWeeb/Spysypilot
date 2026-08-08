"""Deterministic proof builders for approved-artifact tests only."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from openpilot.selfdrive.controls.lib.blatv2.approved_artifact import (
  CalibrationSelectionManifest,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  BEHAVIOR_FINALIZATION_SCHEMA_VERSION,
  BehaviorLearningFinalization,
  FinalizationReason,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import BehaviorPolicy
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  CalibrationParameters,
  CalibrationProfileNode,
  VehicleCalibrationProfile,
)
from openpilot.selfdrive.controls.lib.blatv2.device_acceptance import (
  FAILURE_REASONS,
  MODULAR_ARCHITECTURE,
  PERCENTILE_METHOD,
  DeviceAcceptanceReceipt,
)
from openpilot.selfdrive.controls.lib.blatv2.policy import ControllerPolicy
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  VehicleProfile,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_spool import (
  _encode_frame,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  MeasuredLearningFrame,
)
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  ControlsWitness,
  RouteEvidenceArtifact,
  RouteEvidenceSourceIdentity,
)


def route_evidence_for_frames(
  route_name: str,
  frames: tuple[MeasuredLearningFrame, ...],
  provenance: Mapping[str, object],
  *,
  car_params_bytes: bytes = b"test-canonical-car-params",
  runtime_identity: str | None = None,
) -> RouteEvidenceArtifact:
  """Build complete but behavior-ineligible v4 evidence for unit fixtures."""
  route_hash = hashlib.sha256(route_name.encode()).hexdigest()
  source = RouteEvidenceSourceIdentity(
    route_id=route_name,
    route_time_origin_mono_ns=(frames[0].sample_mono_ns if frames else 1),
    route_segment_sha256=(route_hash,),
    route_segment_size_bytes=(len(frames),),
    source_superproject_commit="1" * 40,
    source_opendbc_commit="2" * 40,
    source_panda_commit="3" * 40,
    controller_source_kind="ineligible",
    controller_artifact_sha256="0" * 64,
    behavior_eligible=False,
    behavior_ineligible_reason="test_fixture_has_no_behavior_plane",
    vehicle_identity="test-vehicle",
    runtime_identity=(
      hashlib.sha256(b"test-runtime").hexdigest()
      if runtime_identity is None
      else runtime_identity
    ),
    schema_versions={"route_evidence": 4},
    preparation_provenance=dict(provenance),
    physical_plane_encoding_id="blatv2-measured-learning-frame-v1",
    physical_record_count=len(frames),
    preparation_cache_key=hashlib.sha256(
      f"test:{route_name}".encode(),
    ).hexdigest(),
    controls_witness_count=len(frames),
    unresolved_witness_count=0,
    gap_count=0,
    model_link_failure_count=len(frames),
  )
  controls = tuple(
    ControlsWitness(
      segment_index=0, ordinal=index, mono_time_ns=frame.sample_mono_ns,
      physical_record_index=index, model_publication_index=-1,
      live_torque_parameters_index=-1, live_delay_index=-1,
      lateral_maneuver_plan_index=-1, poll_mono_time_ns=frame.sample_mono_ns,
      state_sample_mono_ns=frame.response_mono_ns,
      live_parameters_mono_ns=frame.response_mono_ns,
      car_output_report_mono_ns=frame.applied_report_mono_ns,
      car_output_effective_mono_ns=frame.applied_effective_mono_ns,
      car_control_mono_ns=frame.sample_mono_ns,
      raw_request_torque=frame.applied_torque,
      measured_curvature=0.0, desired_curvature=0.0,
      envelope_headroom=max(0.0, 1.0 - abs(frame.applied_torque)),
      torque_output_can_count=round(frame.applied_torque * 409.0),
      steering_request_fault_avoidance_counter=0,
      message_valid=True, model_message_alive=False,
      model_link_valid=False, inputs_valid=frame.inputs_valid,
      lateral_active=frame.lateral_active,
      driver_intervening=frame.steering_pressed,
      steer_fault=frame.steer_fault_temporary or frame.steer_fault_permanent,
      intervention_onset=False, intervention_onset_uncertain=False,
      race_unresolved=False, gap_from_previous=False,
      car_control_paired=True, torque_output_can_valid=True,
      maneuver_plan_available=False,
      live_torque_parameters_available=False, live_delay_available=False,
      live_torque_parameters_checks_passed=False,
      live_torque_parameters_health_exact=True,
      steering_request_active=True,
      steering_request_active_valid=True,
      steering_request_fault_avoidance_counter_valid=True,
    )
    for index, frame in enumerate(frames)
  )
  return RouteEvidenceArtifact(
    source, car_params_bytes,
    b"".join(_encode_frame(frame) for frame in frames), (), controls,
  )


def passing_device_acceptance_receipt(
  *,
  vehicle_identity: str,
  runtime_identity_sha256: str,
  profile_sha256: str,
  controller_policy_sha256: str,
  horizon_policy_sha256: str,
  source_openpilot_commit: str,
  opendbc_commit: str,
  panda_commit: str,
) -> DeviceAcceptanceReceipt:
  return DeviceAcceptanceReceipt(
    route_evidence_sha256s=("f" * 64,),
    device_type="tici",
    vehicle_identity=vehicle_identity,
    controller_architecture=MODULAR_ARCHITECTURE,
    source_openpilot_commit=source_openpilot_commit,
    opendbc_commit=opendbc_commit,
    panda_commit=panda_commit,
    live_artifact_sha256="",
    runtime_identity_sha256=runtime_identity_sha256,
    profile_sha256=profile_sha256,
    controller_policy_sha256=controller_policy_sha256,
    horizon_policy_sha256=horizon_policy_sha256,
    sample_count=1,
    percentile_method=PERCENTILE_METHOD,
    compute_p50_seconds=0.001,
    compute_p90_seconds=0.001,
    compute_p99_seconds=0.001,
    compute_max_seconds=0.001,
    drop_count=0,
    failure_counts=tuple((reason, 0) for reason in FAILURE_REASONS),
  )


def passed_behavior_finalization(
  controller_policy: ControllerPolicy,
  *,
  namespace: str = "approved-artifact-test",
) -> BehaviorLearningFinalization:
  behavior = BehaviorPolicy.from_controller_policy(controller_policy)

  def identity(label: str) -> str:
    return hashlib.sha256(f"{namespace}:{label}".encode()).hexdigest()

  gate_spec = identity("gate-spec")
  route_partition = identity("route-partition")
  recorded_source = identity("recorded-source")
  training = identity("training")
  validation = identity("validation")
  selection = hashlib.sha256(json.dumps({
    "finalBehaviorPolicySha256": behavior.sha256,
    "gateSpecSha256": gate_spec,
    "recordedSourceIdentitySha256": recorded_source,
    "routePartitionSha256": route_partition,
    "trainingSelectionSha256": training,
    "validationSha256": validation,
  }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  return BehaviorLearningFinalization(
    schema_version=BEHAVIOR_FINALIZATION_SCHEMA_VERSION,
    gate_spec_sha256=gate_spec,
    route_partition_sha256=route_partition,
    recorded_source_identity_sha256=recorded_source,
    training_selection_sha256=training,
    validation_sha256=validation,
    smooth_passed=True,
    swift_passed=True,
    strong_passed=True,
    target_materially_improved=True,
    final_behavior_policy=behavior,
    final_behavior_policy_sha256=behavior.sha256,
    behavior_selection_sha256=selection,
    reasons=(FinalizationReason.PASSED,),
  )


def calibration_selection_manifest(
  selected_profile: VehicleProfile,
  *,
  learner_evidence_sha256: str,
  qualification_manifest_sha256: str,
  calibration_profile: VehicleCalibrationProfile | None = None,
) -> CalibrationSelectionManifest:
  calibration = (
    calibration_profile
    if calibration_profile is not None
    else calibration_profile_for_controller(selected_profile)
  )
  profile_sha = hashlib.sha256(selected_profile.to_json().encode()).hexdigest()
  calibration_sha = hashlib.sha256(calibration.to_json().encode()).hexdigest()
  return CalibrationSelectionManifest(
    selected_controller_profile_sha256=profile_sha,
    candidate_calibration_profile_sha256=calibration_sha,
    learner_evidence_sha256=learner_evidence_sha256,
    qualification_manifest_sha256=qualification_manifest_sha256,
    all_nodes_qualified=True,
  )


def calibration_profile_for_controller(
  profile: VehicleProfile,
) -> VehicleCalibrationProfile:
  """Project a controller profile into exact observable test evidence."""
  nodes = tuple(
    CalibrationProfileNode(
      speed_mps=node.speed_mps,
      parameters=CalibrationParameters(
        torque_per_lateral_accel=node.parameters.torque_per_lateral_accel,
        lateral_accel_offset_correction_mps2=(
          node.parameters.lateral_accel_offset_correction_mps2
        ),
        kinetic_friction_torque=node.parameters.kinetic_friction_torque,
        static_breakaway_torque=node.parameters.static_friction_torque,
        transport_delay_s=node.parameters.transport_delay_s,
        rack_rate_resolution_deg_s=node.parameters.rack_rate_resolution_deg_s,
        confidence=node.parameters.confidence,
        qualified=node.parameters.qualified,
      ),
      base_support_s=node.clean_support_s,
      base_sample_count=node.sample_count,
      moving_support_s=0.0,
      moving_sample_count=0,
      breakaway_support_s=0.0,
      breakaway_sample_count=0,
      cross_fit_route_count=node.cross_fit_route_count,
      full_fit_candidate_rms=node.full_fit_candidate_rms,
      breakaway_full_fit_candidate_rms=None,
    )
    for node in profile.nodes
  )
  return VehicleCalibrationProfile(
    vehicle_identity=profile.vehicle_identity,
    revision=profile.revision,
    provenance="test-only observable projection",
    nodes=nodes,
  )
