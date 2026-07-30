"""Cap'n Proto boundary for the active modular lateral controller."""

from __future__ import annotations

import math
from typing import Any

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.blatv2.bootstrap import (
  ControllerSelection,
)
from openpilot.selfdrive.controls.lib.blatv2.live_controller import (
  MODULAR_LIVE_ARCHITECTURE,
  MODULAR_LIVE_VERSION,
  ModularLiveController,
)


def _finite_or_zero(value: object) -> float:
  try:
    numeric = float(value)
  except (TypeError, ValueError, OverflowError):
    return 0.0
  return numeric if math.isfinite(numeric) else 0.0


def _uint_or_zero(value: object, bits: int) -> int:
  try:
    numeric = int(value)
  except (TypeError, ValueError, OverflowError):
    return 0
  maximum = (1 << bits) - 1
  return numeric if 0 <= numeric <= maximum else 0


def build_modular_lateral_state(
  live: ModularLiveController,
  *,
  lateral_active: bool,
  measured_curvature: float,
  v_ego_m_s: float,
) -> Any:
  """Build a normal LateralTorqueState plus append-only modular telemetry."""
  state = log.ControlsState.LateralTorqueState.new_message()
  candidate = live.candidate_result
  prepared = live.prepared_input
  core = None if candidate is None else candidate.core_result

  command = 0.0 if candidate is None else candidate.command_torque
  raw_torque = 0.0 if candidate is None else candidate.raw_torque
  desired_curvature = (
    0.0 if core is None else core.desired_curvature
  )
  current_speed = _finite_or_zero(v_ego_m_s)
  state.active = bool(lateral_active)
  state.error = _finite_or_zero(
    0.0 if core is None else core.position_error_deg,
  )
  state.errorRate = _finite_or_zero(
    0.0 if core is None else core.rate_error_deg_s,
  )
  state.p = _finite_or_zero(
    0.0 if core is None else core.position_feedback_torque,
  )
  state.i = 0.0
  state.d = _finite_or_zero(
    0.0 if core is None else core.rate_feedback_torque,
  )
  state.f = _finite_or_zero(
    0.0
    if core is None
    else (
      core.aligning_torque
      + core.friction_torque
      + core.motion_feedforward_torque
      + core.disturbance_torque
    ),
  )
  state.output = _finite_or_zero(command)
  state.saturated = bool(
    candidate is not None and candidate.constraint_active,
  )
  state.actualLateralAccel = _finite_or_zero(
    measured_curvature * current_speed * current_speed,
  )
  effect_speed = (
    current_speed if core is None else core.effect_speed_mps
  )
  state.desiredLateralAccel = _finite_or_zero(
    desired_curvature * effect_speed * effect_speed,
  )
  state.desiredLateralJerk = 0.0
  state.version = int(MODULAR_LIVE_VERSION)

  state.modularArchitecture = MODULAR_LIVE_ARCHITECTURE
  state.modularControllerVersion = int(MODULAR_LIVE_VERSION)
  state.modularSelection = int(live.selection)
  state.modularBindingReason = int(live.binding_reason)
  state.modularCandidateStatus = int(
    0 if candidate is None else candidate.status,
  )
  state.modularCoreStatus = int(
    0 if core is None else core.status,
  )
  state.modularArtifactHash = str(live.artifact_sha256)
  state.modularProfileHash = str(live.profile_sha256)
  state.modularPolicyHash = str(live.policy_sha256)
  state.modularRuntimeIdentityHash = str(
    live.runtime_identity_sha256,
  )
  state.modularSourceOpenpilotCommit = str(
    live.source_openpilot_commit,
  )
  state.modularOpendbcCommit = str(live.opendbc_commit)
  state.modularControlWitnessMonoTime = _uint_or_zero(
    live.control_witness_ns,
    64,
  )
  state.modularStateSampleMonoTime = _uint_or_zero(
    0 if prepared is None else prepared.state_sample_mono_ns,
    64,
  )
  state.modularModelPublicationMonoTime = _uint_or_zero(
    0 if prepared is None else prepared.model_publication_mono_ns,
    64,
  )
  state.modularModelTimestampEof = _uint_or_zero(
    0 if prepared is None else prepared.model_timestamp_eof_ns,
    64,
  )
  state.modularDesiredCurvatureTimeSeconds = _finite_or_zero(
    0.0 if prepared is None else prepared.desired_curvature_time_s,
  )
  state.modularRawScalarCurvature = _finite_or_zero(
    0.0 if prepared is None else prepared.scalar_curvature,
  )
  state.modularReferenceCurvature = _finite_or_zero(
    desired_curvature,
  )
  state.modularRawTorque = _finite_or_zero(raw_torque)
  state.modularCommandTorque = _finite_or_zero(command)
  state.modularFeasibleTorque = _finite_or_zero(
    0.0 if candidate is None else candidate.feasible_torque,
  )
  state.modularAligningTorque = _finite_or_zero(
    0.0 if core is None else core.aligning_torque,
  )
  state.modularFrictionTorque = _finite_or_zero(
    0.0 if core is None else core.friction_torque,
  )
  state.modularMotionFeedforwardTorque = _finite_or_zero(
    0.0 if core is None else core.motion_feedforward_torque,
  )
  state.modularPositionFeedbackTorque = _finite_or_zero(
    0.0 if core is None else core.position_feedback_torque,
  )
  state.modularRateFeedbackTorque = _finite_or_zero(
    0.0 if core is None else core.rate_feedback_torque,
  )
  state.modularDisturbanceTorque = _finite_or_zero(
    0.0 if core is None else core.disturbance_torque,
  )
  state.modularDesiredAngleDeg = _finite_or_zero(
    0.0 if core is None else core.desired_angle_deg,
  )
  state.modularDesiredRateDegS = _finite_or_zero(
    0.0 if core is None else core.desired_rate_deg_s,
  )
  state.modularDesiredAccelerationDegS2 = _finite_or_zero(
    0.0 if core is None else core.desired_acceleration_deg_s2,
  )
  state.modularMeasuredAngleDeg = _finite_or_zero(
    0.0 if core is None else core.measured_angle_deg,
  )
  state.modularMeasuredRateDegS = _finite_or_zero(
    0.0 if core is None else core.measured_rate_deg_s,
  )
  state.modularMeasuredAccelerationDegS2 = _finite_or_zero(
    (
      0.0
      if prepared is None
      else prepared.measured_rack_acceleration_deg_s2
    ),
  )
  state.modularPredictedAngleDeg = _finite_or_zero(
    0.0 if core is None else core.predicted_angle_deg,
  )
  state.modularPredictedRateDegS = _finite_or_zero(
    0.0 if core is None else core.predicted_rate_deg_s,
  )
  previous_counts = (
    0
    if live.last_exact_applied_counts is None
    else live.last_exact_applied_counts
  )
  state.modularPreviousAppliedCounts = int(previous_counts)
  state.modularPreviousAppliedTorque = _finite_or_zero(
    (
      0.0
      if live.runtime_bundle is None
      else previous_counts
      / live.runtime_bundle.torque_limits.steer_max
    ),
  )
  state.modularDriverTorque = _finite_or_zero(
    0.0 if prepared is None else prepared.driver_torque,
  )
  state.modularConstraintActive = bool(
    candidate is not None and candidate.constraint_active,
  )
  state.modularConstraintReason = int(
    0 if candidate is None else candidate.constraint_reason,
  )
  state.modularFeasibilityStatus = int(
    0 if candidate is None else candidate.feasibility_status,
  )
  state.modularSafetyState = int(
    0 if candidate is None else candidate.safety_state,
  )
  state.modularControlsValid = bool(
    candidate is None or candidate.controls_valid,
  )
  state.modularCarControlValid = bool(
    candidate is None or candidate.car_control_valid,
  )
  state.modularInvalidFrames = _uint_or_zero(
    0 if candidate is None else candidate.invalid_frames,
    16,
  )
  state.modularRecoveryOkFrames = _uint_or_zero(
    0 if candidate is None else candidate.recovery_ok_frames,
    8,
  )
  state.modularPreviousOutputConstrained = bool(
    live.last_output_constrained_input,
  )
  state.modularPreviousActuatorConstrained = bool(
    live.last_actuator_constrained_input,
  )
  state.modularVehicleStateValid = bool(
    prepared is not None and prepared.vehicle_state_valid,
  )
  state.modularLiveParametersValid = bool(
    prepared is not None and prepared.live_parameters_valid,
  )
  state.modularIntentStatus = int(
    (
      0
      if prepared is None or prepared.adaptation is None
      else prepared.adaptation.status.code
    ),
  )
  state.modularComputeTimeSeconds = _finite_or_zero(
    live.compute_time_seconds,
  )
  state.modularStateAgeSeconds = _finite_or_zero(
    0.0 if core is None else core.state_age_s,
  )
  state.modularTotalPredictionHorizonSeconds = _finite_or_zero(
    0.0 if core is None else core.total_prediction_horizon_s,
  )
  state.modularTransportDelaySeconds = _finite_or_zero(
    0.0 if core is None else core.transport_delay_s,
  )
  state.modularCommandEnvelopeApplied = bool(
    candidate is not None and candidate.command_envelope_applied,
  )
  state.modularManeuverForcedStock = bool(
    live.maneuver_forced_stock,
  )
  state.modularProductionEnvelopeVerified = bool(
    live.runtime_bundle is not None
    and live.runtime_bundle.torque_limits.production_envelope_verified
  )
  state.modularSelectionBound = bool(
    live.enabled_bound
    and live.selection == ControllerSelection.MODULAR
  )
  return state
