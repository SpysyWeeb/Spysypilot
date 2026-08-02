from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
  apply_torque_envelope,
)
from openpilot.selfdrive.controls.blatv2_learnerd import (
  PUBLISHED_SERVICES,
  SUBSCRIBED_SERVICES,
  BlatV2LearnerDaemon,
  _CanonicalSourceHistory,
  assert_no_actuation_publishers,
  default_learning_storage_root,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_coordinator import (
  CalibrationLearningFinalization,
  CalibrationLearningLifecycleState,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_learner import (
  CalibrationFitStatus,
  CalibrationInterpolationQualificationReport,
  CalibrationLearningResult,
  CalibrationPairedLossDiagnostic,
  CalibrationQualificationReason,
  CalibrationSampleAccounting,
  CalibrationSampleDisposition,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  _MeasuredEnvelopeConstraint,
  LearningRestoreError,
  MeasuredLearningFrame,
  PersistentLearningRuntime,
  build_detected_runtime_bundle,
  build_persistent_learning_runtime,
)
from openpilot.selfdrive.controls.lib.blatv2.learner import ActuatorBoundary
from openpilot.selfdrive.controls.lib.blatv2.learning_operation_status import (
  LEARNING_OPERATION_STATUS_PARAM,
  LearningOperationState,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_status import (
  LEARNING_STATUS_PARAM,
  DriveEvidenceBaseline,
  build_learning_status_bytes,
  build_learning_status_payload,
  decode_learning_status,
  validate_learning_status_payload,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
)


GENERIC_FINGERPRINT = "GENERIC NON-HYUNDAI TORQUE PLATFORM"
GENERIC_CONTROLLER_MODULE = "test_blatv2_generic_controller"


def _test_route_identity(label: str) -> str:
  return hashlib.sha256(f"blatv2-test-route:{label}".encode()).hexdigest()


class GenericControllerParams:
  STEER_MAX = 211
  STEER_DELTA_UP = 5
  STEER_DELTA_DOWN = 8
  STEER_STEP = 1
  STEER_DRIVER_ALLOWANCE = 37
  STEER_DRIVER_MULTIPLIER = 2
  STEER_DRIVER_FACTOR = 1

  def __init__(self, _car_params) -> None:
    pass


class GenericCarController:
  pass


GenericCarController.__module__ = GENERIC_CONTROLLER_MODULE


class GenericCarInterface:
  CarController = GenericCarController

  def __init__(self, car_params) -> None:
    self.car_params = car_params

  @staticmethod
  def torque_from_lateral_accel():
    return lambda lateral_accel, _torque_tuning: 0.37 * lateral_accel


def rack_dynamics() -> ProvisionalRackDynamics:
  return ProvisionalRackDynamics(
    rack_gain_deg_s2_per_torque=4000.0,
    rack_damping_per_s=10.0,
    rack_rate_resolution_deg_s=4.0,
    provenance="test-only provisional physical seed",
  )


def generic_car_params():
  cp = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
  cp.carFingerprint = GENERIC_FINGERPRINT
  cp.brand = "generic-test-brand"
  return cp


def generic_registry():
  return {GENERIC_FINGERPRINT: GenericCarInterface}


def build_generic_bundle(generic_controller_module):
  cp = generic_car_params()
  bundle, car_interface, controller_params = build_detected_runtime_bundle(
    car_params=cp,
    provisional_rack_dynamics=rack_dynamics(),
    interface_registry=generic_registry(),
  )
  return cp, bundle, car_interface, controller_params


def build_generic_runtime(
  directory: Path,
  generic_controller_module,
) -> PersistentLearningRuntime:
  cp = generic_car_params()
  return build_persistent_learning_runtime(
    car_params=cp,
    storage_root=directory,
    provisional_rack_dynamics=rack_dynamics(),
    interface_registry=generic_registry(),
  )


def measured_frame(
  cp,
  sample_mono_ns: int,
  **overrides,
) -> MeasuredLearningFrame:
  values = {
    "sample_mono_ns": sample_mono_ns,
    "response_mono_ns": sample_mono_ns - 1_000_000,
    "applied_report_mono_ns": sample_mono_ns - 500_000,
    "applied_effective_mono_ns": sample_mono_ns - 10_500_000,
    "speed_mps": 10.0,
    "steering_angle_deg": 5.0 * sample_mono_ns * 1e-9,
    "steering_rate_deg_s": 5.0,
    "steering_torque": 0.0,
    "steering_pressed": False,
    "standstill": False,
    "steer_fault_temporary": False,
    "steer_fault_permanent": False,
    "can_valid": True,
    "can_timeout": False,
    "applied_torque": 0.0,
    "lateral_active": True,
    "live_parameters_valid": True,
    "angle_offset_valid": True,
    "steer_ratio_valid": True,
    "stiffness_factor_valid": True,
    "angle_offset_deg": 0.0,
    "steer_ratio": float(cp.steerRatio),
    "stiffness_factor": float(cp.tireStiffnessFactor),
    "roll_rad": 0.0,
    "inputs_valid": True,
  }
  values.update(overrides)
  return MeasuredLearningFrame(**values)


FRAME_DT_NS = 10_000_000


def warm_runtime_to_first_causal_sample(
  runtime: PersistentLearningRuntime,
  cp,
  base_mono_ns: int,
  *,
  start_index: int = 0,
  **overrides,
) -> int:
  """Advance through command delay plus signed-derivative warm-up."""
  accepted_before = runtime.coordinator.accepted_sample_count
  delay_s = runtime.runtime_bundle.calibration_seed_profile.parameters_at(10.0).parameters.transport_delay_s
  maximum_frames = math.ceil(delay_s / (FRAME_DT_NS * 1e-9)) + 6
  first_valid_command = measured_frame(
    cp,
    base_mono_ns + (start_index + 1) * FRAME_DT_NS,
    **overrides,
  )
  earliest_causal_response_ns = first_valid_command.applied_effective_mono_ns + round(delay_s * 1e9)
  for offset in range(maximum_frames):
    index = start_index + offset
    frame_overrides = dict(overrides)
    if start_index == 0 and offset == 0:
      frame_overrides["applied_effective_mono_ns"] = 0
    frame = measured_frame(
      cp,
      base_mono_ns + index * FRAME_DT_NS,
      **frame_overrides,
    )
    accepted = runtime.ingest(frame)
    if accepted:
      assert frame.response_mono_ns >= earliest_causal_response_ns
      assert runtime.coordinator.accepted_sample_count == accepted_before + 1
      return index + 1
    assert runtime.coordinator.accepted_sample_count == accepted_before
  raise AssertionError("causal learner did not accept within computed bound")


def _test_palisade_409_4_7_boundary_classifier() -> None:
  limits = RuntimeTorqueLimits(
    steer_max=409,
    delta_up=4,
    delta_down=7,
    steer_step=1,
    driver_allowance=50,
    driver_multiplier=2,
    driver_factor=1,
  )
  classifier = _MeasuredEnvelopeConstraint(limits)
  time_s = 1.0

  first = classifier.update(
    sample_time_s=time_s,
    applied_torque=0.0,
    driver_torque=0.0,
    inputs_valid=True,
  )
  assert not first.valid
  time_s += 0.01
  interior = classifier.update(
    sample_time_s=time_s,
    applied_torque=0.0,
    driver_torque=0.0,
    inputs_valid=True,
  )
  assert interior.valid and not interior.constrained

  applied = 0.0
  while applied < 1.0:
    previous = applied
    applied = apply_torque_envelope(
      limits,
      1.0,
      previous,
      0.0,
    ).applied_torque
    time_s += 0.01
    observation = classifier.update(
      sample_time_s=time_s,
      # Exercise the exact Float32 value present in an rlog.
      applied_torque=float(np.float32(applied)),
      driver_torque=0.0,
      inputs_valid=True,
    )
    assert observation.valid and observation.constrained
    assert observation.boundary & ActuatorBoundary.SLEW_BUILD
  assert observation.boundary & ActuatorBoundary.MAGNITUDE
  assert observation.magnitude_boundary_dwell_s == 0.0

  for hold_index in range(13):
    time_s += 0.01
    held = classifier.update(
      sample_time_s=time_s,
      applied_torque=1.0,
      driver_torque=0.0,
      inputs_valid=True,
    )
    assert held.valid and held.constrained
    assert held.boundary == ActuatorBoundary.MAGNITUDE
    assert abs(held.magnitude_boundary_dwell_s - 0.01 * (hold_index + 1)) < 1e-12

  applied = apply_torque_envelope(
    limits,
    -1.0,
    applied,
    0.0,
  ).applied_torque
  time_s += 0.01
  release = classifier.update(
    sample_time_s=time_s,
    applied_torque=applied,
    driver_torque=0.0,
    inputs_valid=True,
  )
  assert release.valid and release.constrained
  assert release.boundary == ActuatorBoundary.SLEW_RELEASE

  saw_crossing = False
  saw_negative_build = False
  while applied > -1.0:
    previous = applied
    applied = apply_torque_envelope(
      limits,
      -1.0,
      previous,
      0.0,
    ).applied_torque
    time_s += 0.01
    negative = classifier.update(
      sample_time_s=time_s,
      applied_torque=applied,
      driver_torque=0.0,
      inputs_valid=True,
    )
    assert negative.valid and negative.constrained
    if previous >= 0.0 and applied < 0.0:
      saw_crossing = True
      assert negative.boundary & ActuatorBoundary.SLEW_RELEASE
      assert negative.boundary & ActuatorBoundary.SLEW_BUILD
    elif abs(applied) > abs(previous):
      saw_negative_build = True
      assert negative.boundary & ActuatorBoundary.SLEW_BUILD
  assert saw_crossing and saw_negative_build
  assert negative.boundary & ActuatorBoundary.MAGNITUDE
  time_s += 0.01
  negative_held = classifier.update(
    sample_time_s=time_s,
    applied_torque=-1.0,
    driver_torque=0.0,
    inputs_valid=True,
  )
  assert negative_held.valid
  assert negative_held.boundary == ActuatorBoundary.MAGNITUDE

  saw_positive_crossing = False
  while applied < 1.0:
    previous = applied
    applied = apply_torque_envelope(
      limits,
      1.0,
      previous,
      0.0,
    ).applied_torque
    time_s += 0.01
    positive_again = classifier.update(
      sample_time_s=time_s,
      applied_torque=applied,
      driver_torque=0.0,
      inputs_valid=True,
    )
    assert positive_again.valid and positive_again.constrained
    if previous <= 0.0 and applied > 0.0:
      saw_positive_crossing = True
      assert positive_again.boundary & ActuatorBoundary.SLEW_RELEASE
      assert positive_again.boundary & ActuatorBoundary.SLEW_BUILD
  assert saw_positive_crossing

  # Every count that a Float32 rlog can carry survives count-grid validation;
  # 267 counts is the worst round-trip error for this 409-count envelope.
  for count in range(-409, 410):
    grid = _MeasuredEnvelopeConstraint(limits)
    encoded = float(np.float32(count / 409))
    grid.update(
      sample_time_s=1.0,
      applied_torque=encoded,
      driver_torque=0.0,
      inputs_valid=True,
    )
    same = grid.update(
      sample_time_s=1.01,
      applied_torque=encoded,
      driver_torque=0.0,
      inputs_valid=True,
    )
    assert same.valid

  # A driver-limited endpoint is reachable but not vehicle-only evidence.
  driver_classifier = _MeasuredEnvelopeConstraint(limits)
  previous = 390 / 409
  driver_classifier.update(
    sample_time_s=2.0,
    applied_torque=previous,
    driver_torque=0.0,
    inputs_valid=True,
  )
  driver_endpoint = apply_torque_envelope(
    limits,
    1.0,
    previous,
    -60.0,
  ).applied_torque
  driver_bound = driver_classifier.update(
    sample_time_s=2.01,
    applied_torque=driver_endpoint,
    driver_torque=-60.0,
    inputs_valid=True,
  )
  assert not driver_bound.valid
  assert driver_bound.boundary & ActuatorBoundary.DRIVER

  # Real card ordering can pair the current driver torque with the prior
  # carOutput. Reject both adjacent samples conservatively; otherwise these
  # collapsed endpoints are indistinguishable without request intent.
  for previous, driver in ((7 / 409, -500.0), (-7 / 409, 500.0)):
    collapsed = _MeasuredEnvelopeConstraint(limits)
    collapsed.update(
      sample_time_s=2.0,
      applied_torque=previous,
      driver_torque=0.0,
      inputs_valid=True,
    )
    endpoint = apply_torque_envelope(
      limits,
      1.0 if driver < 0.0 else -1.0,
      previous,
      driver,
    ).applied_torque
    observation = collapsed.update(
      sample_time_s=2.01,
      applied_torque=endpoint,
      driver_torque=driver,
      inputs_valid=True,
    )
    assert not observation.valid
    assert observation.boundary == ActuatorBoundary.DRIVER

  dwell = _MeasuredEnvelopeConstraint(limits)
  dwell.update(
    sample_time_s=5.0,
    applied_torque=1.0,
    driver_torque=0.0,
    inputs_valid=True,
  )
  for index in range(14):
    settled = dwell.update(
      sample_time_s=5.01 + index * 0.01,
      applied_torque=1.0,
      driver_torque=0.0,
      inputs_valid=True,
    )
  assert abs(settled.magnitude_boundary_dwell_s - 0.13) < 1e-12
  contaminated = dwell.update(
    sample_time_s=5.15,
    applied_torque=1.0,
    driver_torque=60.0,
    inputs_valid=True,
  )
  assert not contaminated.valid
  assert contaminated.magnitude_boundary_dwell_s == 0.0
  # The adjacent-driver veto consumes one extra frame for card/carOutput
  # scheduling ambiguity, then magnitude dwell starts over from zero.
  adjacent = dwell.update(
    sample_time_s=5.16,
    applied_torque=1.0,
    driver_torque=0.0,
    inputs_valid=True,
  )
  assert not adjacent.valid
  restarted = dwell.update(
    sample_time_s=5.17,
    applied_torque=1.0,
    driver_torque=0.0,
    inputs_valid=True,
  )
  assert restarted.valid
  assert restarted.magnitude_boundary_dwell_s == 0.0
  lifecycle_reset = dwell.update(
    sample_time_s=5.18,
    applied_torque=1.0,
    driver_torque=0.0,
    inputs_valid=False,
  )
  assert not lifecycle_reset.valid
  first_after_reset = dwell.update(
    sample_time_s=5.19,
    applied_torque=1.0,
    driver_torque=0.0,
    inputs_valid=True,
  )
  assert not first_after_reset.valid

  for bad_time, bad_torque in (
    (3.0, 0.5),
    (3.01, 410 / 409),
    (3.02, float("nan")),
  ):
    invalid_classifier = _MeasuredEnvelopeConstraint(limits)
    invalid_classifier.update(
      sample_time_s=2.99,
      applied_torque=0.0,
      driver_torque=0.0,
      inputs_valid=True,
    )
    invalid = invalid_classifier.update(
      sample_time_s=bad_time,
      applied_torque=bad_torque,
      driver_torque=0.0,
      inputs_valid=True,
    )
    assert not invalid.valid

  gap_classifier = _MeasuredEnvelopeConstraint(limits)
  gap_classifier.update(
    sample_time_s=4.0,
    applied_torque=0.0,
    driver_torque=0.0,
    inputs_valid=True,
  )
  gap = gap_classifier.update(
    sample_time_s=4.016,
    applied_torque=0.0,
    driver_torque=0.0,
    inputs_valid=True,
  )
  assert not gap.valid


def _test_generic_non_hyundai_construction_uses_detected_opendbc_limits(
  generic_controller_module,
) -> None:
  cp, bundle, car_interface, controller_params = build_generic_bundle(
    generic_controller_module,
  )
  assert cp.brand == "generic-test-brand"
  assert type(car_interface) is GenericCarInterface
  assert type(controller_params) is GenericControllerParams
  assert bundle.car_fingerprint == GENERIC_FINGERPRINT
  assert bundle.vehicle_identity == GENERIC_FINGERPRINT
  assert (
    bundle.torque_limits.steer_max,
    bundle.torque_limits.delta_up,
    bundle.torque_limits.delta_down,
  ) == (211, 5, 8)
  assert bundle.torque_callback_slope == 0.37
  assert "hyundai" not in type(car_interface).__module__.lower()


def _test_runtime_collects_only_onroad_and_persists_only_offroad(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  runtime = build_generic_runtime(tmp_path, generic_controller_module)
  cp = runtime.car_params
  artifact_root = runtime.artifact_paths.root
  assert runtime.coordinator.state is CalibrationLearningLifecycleState.OFFROAD
  assert not artifact_root.exists()

  runtime.transition_onroad(
    _test_route_identity("collect-and-persist"), route_counter=0,
  )
  with patch(
    "openpilot.selfdrive.controls.lib.blatv2.calibration_coordinator._atomic_write_bytes",
  ) as write:
    warm_runtime_to_first_causal_sample(runtime, cp, 1_000_000_000)
    write.assert_not_called()
  assert not artifact_root.exists()

  finalization = runtime.transition_offroad_and_persist()
  assert runtime.coordinator.state is CalibrationLearningLifecycleState.OFFROAD
  assert runtime.artifact_paths.evidence.read_bytes() == (finalization.evidence_bytes)
  assert runtime.artifact_paths.manifest.read_bytes() == (finalization.manifest_bytes)
  assert finalization.candidate_profile_json is None
  assert not runtime.artifact_paths.candidates.exists()


def _test_restart_restores_exact_cross_drive_evidence(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  first = build_generic_runtime(tmp_path, generic_controller_module)
  cp = first.car_params
  first.transition_onroad(_test_route_identity("restart-first"), route_counter=0)
  warm_runtime_to_first_causal_sample(first, cp, 1_000_000_000)
  saved = first.transition_offroad_and_persist()

  restored = build_generic_runtime(tmp_path, generic_controller_module)
  assert restored.coordinator.state is CalibrationLearningLifecycleState.OFFROAD
  assert restored.coordinator.finalize().evidence_bytes == (saved.evidence_bytes)
  assert restored.coordinator.finalize().manifest_bytes == (saved.manifest_bytes)
  restored.transition_onroad(_test_route_identity("restart-second"), route_counter=1)
  warm_runtime_to_first_causal_sample(restored, cp, 2_000_000_000)
  second = restored.transition_offroad_and_persist()
  assert second.evidence_bytes != saved.evidence_bytes


def _test_restore_corruption_and_seed_mismatch_fail_closed(
  tmp_path: Path,
  generic_controller_module,
  test_case: unittest.TestCase,
) -> None:
  runtime = build_generic_runtime(tmp_path, generic_controller_module)
  runtime.transition_onroad(
    _test_route_identity("corruption-restore"), route_counter=0,
  )
  warm_runtime_to_first_causal_sample(
    runtime,
    runtime.car_params,
    1_000_000_000,
  )
  runtime.transition_offroad_and_persist()
  paths = runtime.artifact_paths
  original = paths.evidence.read_bytes()
  paths.evidence.write_bytes(original[:-1])
  with test_case.assertRaisesRegex(LearningRestoreError, "canonical restore"):
    PersistentLearningRuntime.restore(
      car_params=runtime.car_params,
      runtime_bundle=runtime.runtime_bundle,
      artifact_paths=paths,
    )
  assert paths.evidence.read_bytes() == original[:-1]

  paths.evidence.write_bytes(original)
  different_seed = replace(
    runtime.runtime_bundle.calibration_seed_profile,
    provenance="different exact seed provenance",
  )
  different_bundle = replace(
    runtime.runtime_bundle,
    calibration_seed_profile=different_seed,
  )
  with test_case.assertRaisesRegex(LearningRestoreError, "canonical restore"):
    PersistentLearningRuntime.restore(
      car_params=runtime.car_params,
      runtime_bundle=different_bundle,
      artifact_paths=paths,
    )


def _test_clean_frame_filters_reject_invalid_but_include_limit_boundaries(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  runtime = build_generic_runtime(tmp_path, generic_controller_module)
  cp = runtime.car_params
  runtime.transition_onroad(
    _test_route_identity("clean-frame-filters"), route_counter=0,
  )
  base = 1_000_000_000

  frame_index = warm_runtime_to_first_causal_sample(runtime, cp, base)
  accepted = runtime.coordinator.accepted_sample_count

  assert not runtime.ingest(
    measured_frame(
      cp,
      base + frame_index * FRAME_DT_NS,
      steering_pressed=True,
    )
  )
  assert runtime.coordinator.sample_accounting.count(
    CalibrationSampleDisposition.DRIVER_OVERRIDE_OR_ALLOWANCE,
  ) == 1
  frame_index += 1
  assert not runtime.ingest(
    measured_frame(
      cp,
      base + frame_index * FRAME_DT_NS,
      lateral_active=False,
    )
  )
  assert runtime.coordinator.sample_accounting.count(
    CalibrationSampleDisposition.LATERAL_INACTIVE,
  ) == 1
  frame_index += 1
  assert not runtime.ingest(
    measured_frame(
      cp,
      base + frame_index * FRAME_DT_NS,
      standstill=True,
    )
  )
  assert runtime.coordinator.sample_accounting.count(
    CalibrationSampleDisposition.STANDSTILL_OR_BELOW_MIN_STEER_SPEED,
  ) == 1
  frame_index += 1
  assert not runtime.ingest(
    measured_frame(
      cp,
      base + frame_index * FRAME_DT_NS,
      live_parameters_valid=False,
    )
  )
  assert runtime.coordinator.sample_accounting.count(
    CalibrationSampleDisposition.LIVE_RACK_MAPPING_INVALID,
  ) == 1
  frame_index += 1
  assert not runtime.ingest(
    measured_frame(
      cp,
      base + frame_index * FRAME_DT_NS,
      can_valid=False,
    )
  )
  assert runtime.coordinator.sample_accounting.count(
    CalibrationSampleDisposition.VEHICLE_INPUT_INVALID,
  ) == 1
  frame_index += 1
  assert not runtime.ingest(
    measured_frame(
      cp,
      base + frame_index * FRAME_DT_NS,
      response_mono_ns=0,
    )
  )
  assert runtime.coordinator.sample_accounting.count(
    CalibrationSampleDisposition.INVALID_NUMERIC_OR_TIMESTAMP,
  ) == 1
  frame_index += 1
  assert not runtime.ingest(
    measured_frame(
      cp,
      base + frame_index * FRAME_DT_NS,
      steering_torque=float("nan"),
    )
  )
  assert runtime.coordinator.sample_accounting.count(
    CalibrationSampleDisposition.INVALID_NUMERIC_OR_TIMESTAMP,
  ) == 2
  frame_index += 1
  assert not runtime.ingest(
    measured_frame(
      cp,
      base + frame_index * FRAME_DT_NS,
      speed_mps=float("nan"),
    )
  )
  assert runtime.coordinator.sample_accounting.count(
    CalibrationSampleDisposition.INVALID_NUMERIC_OR_TIMESTAMP,
  ) == 3
  frame_index += 1
  assert runtime.coordinator.accepted_sample_count == accepted

  # Envelope, causal command history, and signed derivative warm independently.
  frame_index = warm_runtime_to_first_causal_sample(
    runtime,
    cp,
    base,
    start_index=frame_index,
  )
  accepted += 1

  upper_boundary = runtime.runtime_bundle.torque_limits.delta_up / runtime.runtime_bundle.torque_limits.steer_max
  assert runtime.ingest(
    measured_frame(
      cp,
      base + frame_index * FRAME_DT_NS,
      applied_torque=upper_boundary,
    )
  )
  frame_index += 1
  accepted += 1

  # The boundary becomes response evidence only after the physical transport
  # delay. The current-frame command must not be mislabeled as causal input.
  constrained_seen = False
  for _ in range(12):
    assert runtime.ingest(
      measured_frame(
        cp,
        base + frame_index * FRAME_DT_NS,
      )
    )
    frame_index += 1
    accepted += 1
    constrained_seen |= runtime.last_actuator_constrained
  assert constrained_seen
  assert runtime.coordinator.accepted_sample_count == accepted

  # A finite value that could not have crossed the vehicle-owned envelope in
  # one frame is corrupt input, not constrained evidence.
  assert not runtime.ingest(
    measured_frame(
      cp,
      base + frame_index * FRAME_DT_NS,
      applied_torque=0.5,
    )
  )
  assert runtime.coordinator.sample_accounting.count(
    CalibrationSampleDisposition.CAUSAL_COMMAND_ALIGNMENT_UNAVAILABLE,
  ) >= 1
  frame_index += 1
  assert runtime.last_actuator_constrained
  assert runtime.coordinator.accepted_sample_count == accepted

  # A dropped controls frame cannot be compressed into a 10 ms derivative.
  assert not runtime.ingest(
    measured_frame(
      cp,
      base + (frame_index + 1) * FRAME_DT_NS,
    )
  )
  assert runtime.coordinator.accepted_sample_count == accepted
  accounting = runtime.coordinator.sample_accounting
  assert accounting.count(
    CalibrationSampleDisposition.MEASUREMENT_WARMUP_OR_DISCONTINUITY,
  ) >= 1
  assert accounting.ingested_sample_count == (
    accounting.accepted_sample_count + accounting.rejected_sample_count
  )


def _test_vehicle_owned_slew_and_full_torque_are_accepted_evidence(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  runtime = build_generic_runtime(tmp_path, generic_controller_module)
  cp = runtime.car_params
  runtime.transition_onroad(
    _test_route_identity("vehicle-envelope"), route_counter=0,
  )
  base = 1_000_000_000

  # Prime the measured envelope and derivative using ordinary interior input.
  frame_index = warm_runtime_to_first_causal_sample(runtime, cp, base)
  accepted = runtime.coordinator.accepted_sample_count

  applied = 0.0
  while applied < 1.0:
    applied = apply_torque_envelope(
      runtime.runtime_bundle.torque_limits,
      1.0,
      applied,
      0.0,
    ).applied_torque
    assert runtime.ingest(
      measured_frame(
        cp,
        base + frame_index * FRAME_DT_NS,
        applied_torque=applied,
      )
    )
    accepted += 1
    frame_index += 1

  # Hold long enough for the full-magnitude command to become the aligned
  # input. All source commands remain exact, vehicle-owned envelope points.
  assert applied == 1.0
  full_magnitude_seen = False
  for _ in range(14):
    assert runtime.ingest(
      measured_frame(
        cp,
        base + frame_index * FRAME_DT_NS,
        applied_torque=1.0,
      )
    )
    frame_index += 1
    accepted += 1
    full_magnitude_seen |= any(
      runtime.coordinator._learner.evidence_for_node(node_index).authority_magnitude_sample_count > 0
      for node_index in range(
        len(runtime.runtime_bundle.calibration_seed_profile.nodes),
      )
    )
  assert full_magnitude_seen
  assert runtime.coordinator.accepted_sample_count == accepted


def _test_input_contract_and_subscriptions_exclude_intent_and_requests() -> None:
  assert_no_actuation_publishers()
  assert PUBLISHED_SERVICES == ()
  assert "carControl" in SUBSCRIBED_SERVICES
  assert "modelV2" not in SUBSCRIBED_SERVICES
  forbidden = ("desired", "request", "candidate", "model", "reference")
  frame_fields = tuple(MeasuredLearningFrame.__dataclass_fields__)
  assert not any(token in field.lower() for field in frame_fields for token in forbidden)


def _test_canonical_history_resolves_one_update_race_and_rejects_stale() -> None:
  older = SimpleNamespace(name="older")
  first = SimpleNamespace(name="first")
  future = SimpleNamespace(name="future")
  history = _CanonicalSourceHistory()
  history.update(
    message=older,
    mono_ns=80_000_000,
    valid=True,
    alive=True,
  )
  history.update(
    message=first,
    mono_ns=90_000_000,
    valid=True,
    alive=True,
  )
  history.update(
    message=future,
    mono_ns=110_000_000,
    valid=True,
    alive=True,
  )
  selected = history.select(
    witness_mono_ns=100_000_000,
    maximum_age_ns=15_000_000,
  )
  assert selected is not None
  assert selected.message.name == "first"
  assert selected.mono_ns == 90_000_000
  assert selected.previous_mono_ns == 80_000_000

  late_insert = _CanonicalSourceHistory()
  late_insert.update(
    message=older,
    mono_ns=80_000_000,
    valid=True,
    alive=True,
  )
  late_insert.update(
    message=future,
    mono_ns=110_000_000,
    valid=True,
    alive=True,
  )
  late_insert.update(
    message=first,
    mono_ns=90_000_000,
    valid=True,
    alive=True,
  )
  inserted = late_insert.select(
    witness_mono_ns=100_000_000,
    maximum_age_ns=15_000_000,
  )
  assert inserted is not None
  assert inserted.message.name == "first"
  assert inserted.previous_mono_ns == 80_000_000
  inserted_current = late_insert.select(
    witness_mono_ns=115_000_000,
    maximum_age_ns=15_000_000,
  )
  assert inserted_current is not None
  assert inserted_current.message.name == "future"
  assert inserted_current.previous_mono_ns == 90_000_000

  future_only = _CanonicalSourceHistory()
  future_only.update(
    message=future,
    mono_ns=110_000_000,
    valid=True,
    alive=True,
  )
  assert (
    future_only.select(
      witness_mono_ns=100_000_000,
      maximum_age_ns=15_000_000,
    )
    is None
  )

  stale = _CanonicalSourceHistory()
  stale.update(
    message=first,
    mono_ns=84_000_000,
    valid=True,
    alive=True,
  )
  assert (
    stale.select(
      witness_mono_ns=100_000_000,
      maximum_age_ns=15_000_000,
    )
    is None
  )


def _test_runtime_uses_delayed_command_not_same_frame_torque(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  runtime = build_generic_runtime(tmp_path, generic_controller_module)
  cp = runtime.car_params
  runtime.transition_onroad(
    _test_route_identity("delayed-command"), route_counter=0,
  )
  base = 1_000_000_000
  captured = []
  original_ingest = runtime.coordinator.ingest

  def capture(sample, **kwargs):
    captured.append(sample)
    return original_ingest(sample, **kwargs)

  runtime.coordinator.ingest = capture
  frames = []
  for index in range(18):
    overrides = {
      "applied_torque": (index / runtime.runtime_bundle.torque_limits.steer_max),
    }
    if index == 0:
      overrides["applied_effective_mono_ns"] = 0
    frame = measured_frame(
      cp,
      base + index * FRAME_DT_NS,
      **overrides,
    )
    frames.append(frame)
    runtime.ingest(frame)

  valid_index = next(index for index, sample in enumerate(captured) if sample.valid)
  valid_sample = captured[valid_index]
  response_time_s = frames[valid_index].response_mono_ns * 1e-9
  delay_s = runtime.runtime_bundle.calibration_seed_profile.parameters_at(
    frames[valid_index].speed_mps,
  ).parameters.transport_delay_s
  causal_target_s = response_time_s - delay_s
  eligible_indices = [
    index for index, frame in enumerate(frames[: valid_index + 1]) if (index > 0 and frame.applied_effective_mono_ns * 1e-9 <= causal_target_s + 1e-12)
  ]
  expected_index = max(eligible_indices)

  assert valid_sample.applied_torque == frames[expected_index].applied_torque
  assert expected_index < valid_index
  assert valid_sample.applied_torque != frames[valid_index].applied_torque


def _test_runtime_retains_unsigned_reversal_without_fitting_it(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  runtime = build_generic_runtime(tmp_path, generic_controller_module)
  cp = runtime.car_params
  runtime.transition_onroad(
    _test_route_identity("unsigned-reversal"), route_counter=0,
  )
  base = 1_000_000_000
  next_index = warm_runtime_to_first_causal_sample(
    runtime,
    cp,
    base,
  )
  forward_mono_ns = base + next_index * FRAME_DT_NS
  forward = measured_frame(
    cp,
    forward_mono_ns,
    steering_rate_deg_s=8.0,
  )
  assert runtime.ingest(forward)

  before_reversals = sum(item.rack_reversals for item in runtime.coordinator.support_diagnostics)
  before_supported = sum(item.supported_sample_count for item in runtime.coordinator.support_diagnostics)
  before_moving = sum(item.moving_sample_count for item in runtime.coordinator.support_diagnostics)
  before_breakaway = sum(item.breakaway_sample_count for item in runtime.coordinator.support_diagnostics)
  before_clean = runtime.coordinator.clean_sample_count
  before_accepted = runtime.coordinator.accepted_sample_count
  reversal = measured_frame(
    cp,
    forward_mono_ns + FRAME_DT_NS,
    steering_angle_deg=forward.steering_angle_deg - 0.1,
    # The positive raw magnitude models an unsigned platform signal.
    steering_rate_deg_s=8.0,
  )
  assert runtime.ingest(reversal)

  after = runtime.coordinator.support_diagnostics
  assert sum(item.rack_reversals for item in after) > before_reversals
  assert sum(item.supported_sample_count for item in after) > before_supported
  assert sum(item.moving_sample_count for item in after) > before_moving
  assert sum(item.breakaway_sample_count for item in after) == before_breakaway
  assert runtime.coordinator.clean_sample_count == before_clean + 1
  assert runtime.coordinator.accepted_sample_count == before_accepted + 1


class FakeParams:
  def __init__(self, root: Path, car_params_bytes: bytes = b"car-params"):
    self.root = root
    # Production keeps this identity across manager restarts; CarParams itself
    # is manager-cleared and cannot be the learner's sole restore source.
    self.values: dict[str, object] = {
      "CarParamsPersistent": car_params_bytes,
      "CurrentRoute": "00000000--0000000000",
      "IsOffroad": False,
    }
    self.puts: list[tuple[str, object, bool]] = []
    self.removes: list[str] = []
    self.remove_failures = 0
    self.before_put = None

  def get_param_path(self) -> str:
    return str(self.root / "params" / "d")

  def get(self, key: str, block: bool = False):
    assert block is False
    return self.values.get(key)

  def get_bool(self, key: str, block: bool = False):
    assert block is False
    return self.values.get(key)

  def put(self, key: str, value, block: bool = False):
    # JSON Params reject pre-encoded bytes at the Python/C++ boundary.
    assert type(value) is dict
    if self.before_put is not None:
      self.before_put(key, value)
    self.values[key] = value
    self.puts.append((key, value, block))

  def remove(self, key: str):
    if self.remove_failures > 0:
      self.remove_failures -= 1
      raise OSError("injected Params remove failure")
    self.values.pop(key, None)
    self.removes.append(key)


class FakeLogger:
  def __init__(self) -> None:
    self.exceptions: list[str] = []

  def exception(self, message: str) -> None:
    self.exceptions.append(message)


class FakeSubMaster:
  def __init__(self) -> None:
    self.data = {service: SimpleNamespace() for service in SUBSCRIBED_SERVICES}
    self.updated = dict.fromkeys(SUBSCRIBED_SERVICES, False)
    self.seen = dict.fromkeys(SUBSCRIBED_SERVICES, False)
    self.valid = dict.fromkeys(SUBSCRIBED_SERVICES, False)
    self.alive = dict.fromkeys(SUBSCRIBED_SERVICES, False)
    self.logMonoTime = dict.fromkeys(SUBSCRIBED_SERVICES, 0)
    self.timeouts: list[int] = []

  def __getitem__(self, service: str):
    return self.data[service]

  def update(self, timeout: int) -> None:
    self.timeouts.append(timeout)

  def publish(self, messages: dict[str, tuple[object, int]]) -> None:
    self.updated = dict.fromkeys(SUBSCRIBED_SERVICES, False)
    for service, (message, mono_ns) in messages.items():
      self.data[service] = message
      self.updated[service] = True
      self.seen[service] = True
      self.valid[service] = True
      self.alive[service] = True
      self.logMonoTime[service] = mono_ns


def fake_messages(cp, *, started: bool, active: bool = True):
  return {
    "deviceState": SimpleNamespace(started=started),
    "controlsState": SimpleNamespace(),
    "carControl": SimpleNamespace(latActive=active),
    "carState": SimpleNamespace(
      vEgo=10.0,
      steeringAngleDeg=0.0,
      steeringRateDeg=5.0,
      steeringTorque=0.0,
      steeringPressed=False,
      standstill=False,
      steerFaultTemporary=False,
      steerFaultPermanent=False,
      canValid=True,
      canTimeout=False,
    ),
    "carOutput": SimpleNamespace(
      actuatorsOutput=SimpleNamespace(torque=0.0),
    ),
    "liveParameters": SimpleNamespace(
      valid=True,
      angleOffsetValid=True,
      steerRatioValid=True,
      stiffnessFactorValid=True,
      angleOffsetDeg=0.0,
      steerRatio=cp.steerRatio,
      stiffnessFactor=cp.tireStiffnessFactor,
      roll=0.0,
    ),
  }


def publish_frame(sm: FakeSubMaster, cp, mono_ns: int) -> None:
  messages = fake_messages(cp, started=True)
  messages["carState"].steeringAngleDeg = 5.0 * mono_ns * 1e-9
  sm.publish({service: (messages[service], mono_ns) for service in SUBSCRIBED_SERVICES})


def publish_causal_frames(
  daemon: BlatV2LearnerDaemon,
  sm: FakeSubMaster,
  cp,
  *,
  base_mono_ns: int,
  start_index: int = 0,
) -> int:
  if daemon.runtime is None:
    raise AssertionError("daemon runtime must be prepared before warm-up")
  accepted_before = daemon.accepted_sample_count
  delay_s = daemon.runtime.runtime_bundle.calibration_seed_profile.parameters_at(10.0).parameters.transport_delay_s
  maximum_frames = math.ceil(delay_s / (FRAME_DT_NS * 1e-9)) + 7
  for offset in range(maximum_frames):
    index = start_index + offset
    publish_frame(sm, cp, base_mono_ns + index * FRAME_DT_NS)
    daemon.step()
    if daemon.accepted_sample_count > accepted_before:
      return index + 1
  raise AssertionError("daemon did not accept within computed causal bound")


def _test_daemon_observes_offroad_without_onroad_poll_and_restores(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  cp = generic_car_params()
  sm = FakeSubMaster()
  params = FakeParams(tmp_path)
  assert "CarParams" not in params.values
  assert params.values["CarParamsPersistent"] == b"car-params"
  params.values["CarParamsPersistent"] = b"stale-persistent-car-params"
  params.values["CarParams"] = b"current-route-car-params"
  logger = FakeLogger()
  created: list[PersistentLearningRuntime] = []
  decoded_values: list[bytes] = []

  def runtime_factory(car_params, storage_root):
    runtime = build_persistent_learning_runtime(
      car_params=car_params,
      storage_root=storage_root,
      provisional_rack_dynamics=rack_dynamics(),
      interface_registry=generic_registry(),
    )
    created.append(runtime)
    return runtime

  daemon = BlatV2LearnerDaemon(
    sm=sm,
    params=params,
    storage_root=tmp_path / "learning",
    car_params_decoder=lambda encoded: decoded_values.append(encoded) or cp,
    runtime_factory=runtime_factory,
    logger=logger,
    route_owned_persistence=False,
  )
  assert default_learning_storage_root(params) == (tmp_path / "params" / "blatv2-learning")

  # The always-running process prepares and authenticates evidence offroad;
  # qualification is never smuggled into the first onroad frame.
  offroad_messages = fake_messages(cp, started=False)
  sm.publish(
    {
      "deviceState": (offroad_messages["deviceState"], 800_000_000),
    }
  )
  daemon.step()
  assert len(created) == 1
  assert decoded_values == [b"current-route-car-params"]
  assert created[0].coordinator.state is CalibrationLearningLifecycleState.OFFROAD
  assert not (tmp_path / "learning").exists()

  params.values["CarParams"] = b"current-route-car-params"
  start_messages = fake_messages(cp, started=True)
  sm.publish(
    {
      "deviceState": (start_messages["deviceState"], 900_000_000),
    }
  )
  daemon.step()
  assert len(created) == 1
  assert created[0].coordinator.state is CalibrationLearningLifecycleState.ONROAD
  assert not (tmp_path / "learning").exists()

  daemon_next_index = publish_causal_frames(
    daemon,
    sm,
    cp,
    base_mono_ns=1_000_000_000,
  )
  assert daemon.controls_witness_count == daemon_next_index
  assert daemon.accepted_sample_count == 1
  assert not (tmp_path / "learning").exists()

  def assert_artifacts_precede_status(key, _value):
    if key == LEARNING_OPERATION_STATUS_PARAM:
      return
    assert key == LEARNING_STATUS_PARAM
    assert created[0].artifact_paths.evidence.is_file()
    assert created[0].artifact_paths.manifest.is_file()

  params.before_put = assert_artifacts_precede_status
  stop_messages = fake_messages(cp, started=False)
  sm.publish(
    {
      "deviceState": (stop_messages["deviceState"], 2_000_000_000),
    }
  )
  daemon.step()
  assert created[0].coordinator.state is CalibrationLearningLifecycleState.OFFROAD
  assert created[0].artifact_paths.evidence.is_file()
  assert created[0].artifact_paths.manifest.is_file()
  status = params.values[LEARNING_STATUS_PARAM]
  assert type(status) is dict
  assert status["last_drive_complete"] is True
  assert all(node["last_drive_clean_support_s"] is not None and node["last_drive_accepted_sample_count"] is not None for node in status["nodes"])
  assert params.puts[-1][0] == LEARNING_OPERATION_STATUS_PARAM
  assert params.puts[-1][1]["state"] == "idle"
  assert params.puts[-1][1]["diagnostic"] == "evidence_ready"
  assert any(key == LEARNING_STATUS_PARAM for key, _, _ in params.puts)
  assert params.puts[-1][2] is True
  assert all(timeout == 100 for timeout in sm.timeouts)
  assert logger.exceptions == []

  # A reconstructed daemon/runtime restores the exact persisted evidence.
  restored = runtime_factory(cp, tmp_path / "learning")
  assert restored.coordinator.finalize().evidence_bytes == (created[0].coordinator.finalize().evidence_bytes)

  # Manager-start clears the cache; verified offroad restore republishes the
  # cumulative state without inventing unknowable per-drive deltas.
  restart_params = FakeParams(tmp_path / "restart")
  restart_sm = FakeSubMaster()
  restarted = BlatV2LearnerDaemon(
    sm=restart_sm,
    params=restart_params,
    storage_root=tmp_path / "learning",
    car_params_decoder=lambda _encoded: cp,
    runtime_factory=runtime_factory,
    logger=FakeLogger(),
    route_owned_persistence=False,
  )
  restart_sm.publish(
    {
      "deviceState": (offroad_messages["deviceState"], 3_000_000_000),
    }
  )
  restarted.step()
  restored_status = restart_params.values[LEARNING_STATUS_PARAM]
  assert restored_status["last_drive_complete"] is False
  assert all(node["last_drive_clean_support_s"] is None and node["last_drive_accepted_sample_count"] is None for node in restored_status["nodes"])


def _test_mid_drive_start_skips_collection_until_offroad_preparation(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  cp = generic_car_params()
  sm = FakeSubMaster()
  params = FakeParams(tmp_path)
  logger = FakeLogger()
  factory_calls: list[bool] = []

  def runtime_factory(car_params, storage_root):
    factory_calls.append(True)
    return build_persistent_learning_runtime(
      car_params=car_params,
      storage_root=storage_root,
      provisional_rack_dynamics=rack_dynamics(),
      interface_registry=generic_registry(),
    )

  daemon = BlatV2LearnerDaemon(
    sm=sm,
    params=params,
    storage_root=tmp_path / "learning",
    car_params_decoder=lambda _encoded: cp,
    runtime_factory=runtime_factory,
    logger=logger,
    route_owned_persistence=False,
  )
  started = fake_messages(cp, started=True)
  sm.publish({"deviceState": (started["deviceState"], 1_000_000_000)})
  daemon.step()
  assert factory_calls == []
  assert daemon.runtime is None

  publish_frame(sm, cp, 1_010_000_000)
  daemon.step()
  assert daemon.accepted_sample_count == 0
  assert not (tmp_path / "learning").exists()
  assert LEARNING_STATUS_PARAM not in params.values

  stopped = fake_messages(cp, started=False)
  sm.publish({"deviceState": (stopped["deviceState"], 2_000_000_000)})
  daemon.step()
  assert factory_calls == [True]
  assert daemon.runtime is not None
  assert daemon.runtime.coordinator.state is CalibrationLearningLifecycleState.OFFROAD

  params.values["CarParams"] = b"car-params"
  sm.publish({"deviceState": (started["deviceState"], 3_000_000_000)})
  daemon.step()
  assert daemon.runtime.coordinator.state is CalibrationLearningLifecycleState.ONROAD


def _test_car_params_precedence_fallback_and_transient_absence(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  cp = generic_car_params()

  missing_params = FakeParams(tmp_path / "missing")
  missing_params.values.pop("CarParamsPersistent")
  preserved_status = {"preserved_during_unknown_identity": True}
  missing_params.values[LEARNING_STATUS_PARAM] = preserved_status
  missing_sm = FakeSubMaster()
  missing = BlatV2LearnerDaemon(
    sm=missing_sm,
    params=missing_params,
    storage_root=tmp_path / "missing-learning",
    car_params_decoder=lambda _encoded: cp,
    runtime_factory=lambda _cp, _root: (_ for _ in ()).throw(
      AssertionError("runtime factory needs a concrete CarParams"),
    ),
    logger=FakeLogger(),
    route_owned_persistence=False,
  )
  stopped = fake_messages(cp, started=False)
  missing_sm.publish(
    {
      "deviceState": (stopped["deviceState"], 1_000_000_000),
    }
  )
  missing.step()
  assert missing.runtime is None
  assert missing_params.values[LEARNING_STATUS_PARAM] is preserved_status
  assert missing_params.removes == []

  fallback_params = FakeParams(tmp_path / "fallback")
  fallback_params.values.pop("CarParamsPersistent")
  fallback_params.values["CarParams"] = b"manager-scoped-car-params"
  fallback_sm = FakeSubMaster()
  calls: list[bytes] = []

  def runtime_factory(car_params, storage_root):
    calls.append(fallback_params.values["CarParams"])
    return build_persistent_learning_runtime(
      car_params=car_params,
      storage_root=storage_root,
      provisional_rack_dynamics=rack_dynamics(),
      interface_registry=generic_registry(),
    )

  fallback = BlatV2LearnerDaemon(
    sm=fallback_sm,
    params=fallback_params,
    storage_root=tmp_path / "fallback-learning",
    car_params_decoder=lambda _encoded: cp,
    runtime_factory=runtime_factory,
    logger=FakeLogger(),
    route_owned_persistence=False,
  )
  fallback_sm.publish(
    {
      "deviceState": (stopped["deviceState"], 2_000_000_000),
    }
  )
  fallback.step()
  assert calls == [b"manager-scoped-car-params"]
  assert fallback.runtime is not None
  first_runtime = fallback.runtime

  # The model fingerprint alone is not the runtime identity. A changed exact
  # canonical CarParams for the same fingerprint rebuilds the prepared owner.
  fallback_params.values["CarParams"] = b"changed-same-fingerprint"
  fallback_sm.publish(
    {
      "deviceState": (stopped["deviceState"], 2_100_000_000),
    }
  )
  fallback.step()
  assert calls == [
    b"manager-scoped-car-params",
    b"changed-same-fingerprint",
  ]
  assert fallback.runtime is not first_runtime
  assert fallback._prepared_car_params_bytes == b"changed-same-fingerprint"


def _test_live_identity_late_and_mismatch_never_cross_contaminate(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  cp_a = generic_car_params()
  cp_b = SimpleNamespace(carFingerprint="DIFFERENT LIVE VEHICLE")

  def decoder(encoded):
    return cp_b if encoded == b"vehicle-b" else cp_a

  def runtime_factory(car_params, storage_root):
    return build_persistent_learning_runtime(
      car_params=car_params,
      storage_root=storage_root,
      provisional_rack_dynamics=rack_dynamics(),
      interface_registry=generic_registry(),
    )

  # A persistent identity prepares A, but a confirmed live B makes the entire
  # drive ineligible. Changing the live Param back to A cannot reopen it.
  mismatch_params = FakeParams(
    tmp_path / "mismatch",
    car_params_bytes=b"vehicle-a",
  )
  mismatch_sm = FakeSubMaster()
  mismatch = BlatV2LearnerDaemon(
    sm=mismatch_sm,
    params=mismatch_params,
    storage_root=tmp_path / "mismatch-learning",
    car_params_decoder=decoder,
    runtime_factory=runtime_factory,
    logger=FakeLogger(),
    route_owned_persistence=False,
  )
  stopped = fake_messages(cp_a, started=False)
  started = fake_messages(cp_a, started=True)
  mismatch_sm.publish(
    {
      "deviceState": (stopped["deviceState"], 1_000_000_000),
    }
  )
  mismatch.step()
  assert mismatch.runtime is not None
  mismatch_params.values["CarParams"] = b"vehicle-b"
  mismatch_sm.publish(
    {
      "deviceState": (started["deviceState"], 2_000_000_000),
    }
  )
  mismatch.step()
  assert mismatch._runtime_unavailable_this_drive
  assert not mismatch._live_identity_bound
  assert mismatch.runtime.coordinator.state is CalibrationLearningLifecycleState.OFFROAD
  mismatch_params.values["CarParams"] = b"vehicle-a"
  for mono_ns in (2_010_000_000, 2_020_000_000, 2_030_000_000):
    publish_frame(mismatch_sm, cp_a, mono_ns)
    mismatch.step()
  assert mismatch.accepted_sample_count == 0
  assert mismatch.runtime.coordinator.accepted_sample_count == 0

  # A late live CarParams is a startup race, not a rejected drive. Witnesses
  # remain unresolved without crashing until the identity is bound.
  late_params = FakeParams(
    tmp_path / "late",
    car_params_bytes=b"vehicle-a",
  )
  late_sm = FakeSubMaster()
  late = BlatV2LearnerDaemon(
    sm=late_sm,
    params=late_params,
    storage_root=tmp_path / "late-learning",
    car_params_decoder=decoder,
    runtime_factory=runtime_factory,
    logger=FakeLogger(),
    route_owned_persistence=False,
  )
  late_sm.publish(
    {
      "deviceState": (stopped["deviceState"], 3_000_000_000),
    }
  )
  late.step()
  late_sm.publish(
    {
      "deviceState": (started["deviceState"], 4_000_000_000),
    }
  )
  late.step()
  assert late.runtime is not None
  assert late.runtime.coordinator.state is CalibrationLearningLifecycleState.OFFROAD
  for mono_ns in (4_010_000_000, 4_020_000_000):
    publish_frame(late_sm, cp_a, mono_ns)
    late.step()
  assert late.controls_witness_count == 2
  assert late.unresolved_witness_count == 2
  assert late.accepted_sample_count == 0
  assert not late._runtime_unavailable_this_drive

  late_params.values["CarParams"] = b"vehicle-a"
  late_next_index = publish_causal_frames(
    late,
    late_sm,
    cp_a,
    base_mono_ns=4_030_000_000,
  )
  assert late._live_identity_bound
  assert late.runtime.coordinator.state is CalibrationLearningLifecycleState.ONROAD
  accepted_before_identity_change = late.accepted_sample_count
  assert accepted_before_identity_change >= 1

  # A confirmed identity change after binding closes collection before that
  # witness reaches A's evidence, and the drive cannot reopen.
  late_params.values["CarParams"] = b"vehicle-b"
  publish_frame(
    late_sm,
    cp_a,
    4_030_000_000 + late_next_index * FRAME_DT_NS,
  )
  late.step()
  assert late._runtime_unavailable_this_drive
  assert not late._live_identity_bound
  assert late.accepted_sample_count == accepted_before_identity_change
  unresolved_after_change = late.unresolved_witness_count
  late_params.values["CarParams"] = b"vehicle-a"
  publish_frame(
    late_sm,
    cp_a,
    4_030_000_000 + (late_next_index + 1) * FRAME_DT_NS,
  )
  late.step()
  assert late.accepted_sample_count == accepted_before_identity_change
  assert late.unresolved_witness_count == unresolved_after_change + 1


def _test_learning_status_is_canonical_strict_and_drive_local(
  tmp_path: Path,
  generic_controller_module,
  test_case: unittest.TestCase,
) -> None:
  runtime = build_generic_runtime(tmp_path, generic_controller_module)
  before = runtime.coordinator.finalize()
  baseline = DriveEvidenceBaseline.from_support_diagnostics(
    runtime.coordinator.support_diagnostics,
  )
  empty_payload = build_learning_status_payload(
    finalization=before,
    runtime_bundle=runtime.runtime_bundle,
    drive_baseline=None,
  )
  empty_bytes = build_learning_status_bytes(
    finalization=before,
    runtime_bundle=runtime.runtime_bundle,
    drive_baseline=None,
  )
  assert empty_bytes == build_learning_status_bytes(
    finalization=before,
    runtime_bundle=runtime.runtime_bundle,
    drive_baseline=None,
  )
  assert decode_learning_status(empty_bytes) == empty_payload
  assert empty_payload["last_drive_complete"] is False
  assert empty_payload["all_nodes_qualified"] is False
  assert empty_payload["candidate_profile_sha256"] is None
  assert empty_payload["candidate_profile_revision"] is None
  assert [node["node_index"] for node in empty_payload["nodes"]] == (list(range(len(runtime.runtime_bundle.calibration_seed_profile.nodes))))
  assert all(node["reasons"] for node in empty_payload["nodes"])
  assert all(
    node["candidate_parameters"] is None
    and node["last_drive_clean_support_s"] is None
    and node["last_drive_base_support_s"] is None
    and node["last_drive_moving_support_s"] is None
    and node["last_drive_breakaway_support_s"] is None
    and node["last_drive_authority_support_s"] is None
    and node["last_drive_accepted_sample_count"] is None
    for node in empty_payload["nodes"]
  )
  with test_case.assertRaisesRegex(ValueError, "not canonical"):
    decode_learning_status(empty_bytes + b" ")
  invalid_confidence = copy.deepcopy(empty_payload)
  invalid_confidence["nodes"][0]["confidence"] = 1.1
  with test_case.assertRaisesRegex(ValueError, "must not exceed one"):
    validate_learning_status_payload(invalid_confidence)
  invalid_reason = copy.deepcopy(empty_payload)
  invalid_reason["nodes"][0]["reasons"] = ["learned"]
  with test_case.assertRaisesRegex(ValueError, "reasons disagree"):
    validate_learning_status_payload(invalid_reason)

  authority_reports = (
    replace(
      before.learning_result.node_reports[0],
      reasons=(CalibrationQualificationReason.AUTHORITY_VALIDATION_REGRESSION,),
    ),
    *before.learning_result.node_reports[1:],
  )
  authority_finalization = replace(
    before,
    learning_result=CalibrationLearningResult(
      node_reports=authority_reports,
      candidate_profile=None,
    ),
  )
  authority_payload = build_learning_status_payload(
    finalization=authority_finalization,
    runtime_bundle=runtime.runtime_bundle,
    drive_baseline=None,
  )
  assert authority_payload["nodes"][0]["reasons"] == [
    CalibrationQualificationReason.AUTHORITY_VALIDATION_REGRESSION.value,
  ]
  validate_learning_status_payload(authority_payload)

  learned_nodes = tuple(
    replace(
      node,
      parameters=replace(
        node.parameters,
        torque_per_lateral_accel=0.4,
        lateral_accel_offset_correction_mps2=-0.03,
        kinetic_friction_torque=0.03,
        static_breakaway_torque=0.09,
        confidence=1.0,
        qualified=True,
      ),
    )
    for node in runtime.runtime_bundle.calibration_seed_profile.nodes
  )
  learned_profile = replace(
    runtime.runtime_bundle.calibration_seed_profile,
    revision=runtime.runtime_bundle.calibration_seed_profile.revision + 1,
    provenance="test-only fully qualified evidence",
    nodes=learned_nodes,
  )
  paired_loss = CalibrationPairedLossDiagnostic(
    2,
    -0.01,
    0.002,
    -0.012,
    -0.008,
    1e-14,
  )
  learned_reports = tuple(
    replace(
      before.learning_result.node_reports[index],
      minimum_support_s=150.0,
      clean_support_s=180.0,
      supported_sample_count=18000,
      training_count=14400,
      validation_count=3600,
      validation_support_s=36.0,
      base_support_s=60.0,
      base_sample_count=6000,
      moving_support_s=60.0,
      moving_sample_count=6000,
      moving_training_count=4800,
      moving_validation_count=1200,
      breakaway_support_s=60.0,
      breakaway_sample_count=6000,
      breakaway_training_count=4800,
      breakaway_validation_count=1200,
      lateral_accel_span_mps2=1.0,
      lateral_accel_rms_mps2=0.3,
      rack_travel_deg=240.0,
      applied_torque_span=0.4,
      rack_reversals=8,
      lateral_accel_directions=2,
      applied_torque_directions=2,
      seed_validation_rms=0.2,
      candidate_validation_rms=0.1,
      moving_seed_validation_rms=0.2,
      moving_candidate_validation_rms=0.1,
      breakaway_seed_validation_rms=0.2,
      breakaway_candidate_validation_rms=0.1,
      confidence=1.0,
      reasons=(CalibrationQualificationReason.LEARNED,),
      candidate_parameters=node.parameters,
      fit_diagnostics=tuple(
        replace(
          diagnostic,
          status=CalibrationFitStatus.IDENTIFIABLE,
          moving_rank=diagnostic.moving_parameter_count,
          condition_estimate=1.0,
          breakaway_rank=diagnostic.breakaway_parameter_count,
        )
        for diagnostic in before.learning_result.node_reports[index].fit_diagnostics
      ),
      training_paired_loss=paired_loss,
      validation_paired_loss=paired_loss,
      training_outcome=CalibrationQualificationReason.LEARNED,
    )
    for index, node in enumerate(learned_nodes)
  )
  candidate_json = learned_profile.to_json().encode("utf-8")
  interpolation_reports = tuple(
    CalibrationInterpolationQualificationReport(
      interval_index=index,
      lower_speed_mps=learned_nodes[index].speed_mps,
      upper_speed_mps=learned_nodes[index + 1].speed_mps,
      training_paired_loss=paired_loss,
      validation_paired_loss=paired_loss,
      reasons=(CalibrationQualificationReason.QUALIFIED,),
    )
    for index in range(len(learned_nodes) - 1)
  )
  qualified_finalization = CalibrationLearningFinalization(
    manifest_bytes=b"test manifest",
    manifest_sha256="b" * 64,
    evidence_bytes=b"test evidence",
    evidence_sha256="c" * 64,
    selected_profile_json=candidate_json,
    selected_profile_sha256=hashlib.sha256(candidate_json).hexdigest(),
    candidate_profile_json=candidate_json,
    candidate_profile_sha256=hashlib.sha256(candidate_json).hexdigest(),
    learning_result=CalibrationLearningResult(
      node_reports=learned_reports,
      candidate_profile=learned_profile,
      interpolation_reports=interpolation_reports,
      selected_profile=learned_profile,
    ),
    sample_accounting=CalibrationSampleAccounting.empty(),
  )
  qualified = build_learning_status_payload(
    finalization=qualified_finalization,
    runtime_bundle=runtime.runtime_bundle,
    drive_baseline=None,
  )
  assert qualified["all_nodes_qualified"] is True
  assert qualified["candidate_profile_revision"] == learned_profile.revision
  assert all(node["qualified"] for node in qualified["nodes"])
  assert all(
    set(node["candidate_parameters"])
    == {
      "torque_per_lateral_accel",
      "lateral_accel_offset_correction_mps2",
      "kinetic_friction_torque",
      "static_breakaway_torque",
    }
    for node in qualified["nodes"]
  )
  assert all(
    node["candidate_parameters"]["lateral_accel_offset_correction_mps2"]
    == -0.03
    for node in qualified["nodes"]
  )

  runtime.transition_onroad(
    _test_route_identity("status-drive-local"), route_counter=0,
  )
  cp = runtime.car_params
  warm_runtime_to_first_causal_sample(runtime, cp, 1_000_000_000)
  after = runtime.transition_offroad_and_persist()
  driven = build_learning_status_payload(
    finalization=after,
    runtime_bundle=runtime.runtime_bundle,
    drive_baseline=baseline,
  )
  assert driven["last_drive_complete"] is True
  assert all(
    node["last_drive_clean_support_s"] is not None
    and node["last_drive_base_support_s"] is not None
    and node["last_drive_moving_support_s"] is not None
    and node["last_drive_breakaway_support_s"] is not None
    and node["last_drive_authority_support_s"] is not None
    and node["last_drive_accepted_sample_count"] is not None
    for node in driven["nodes"]
  )
  assert sum(node["last_drive_accepted_sample_count"] for node in driven["nodes"]) >= 1

  mismatched_seed = replace(
    runtime.runtime_bundle.calibration_seed_profile,
    nodes=runtime.runtime_bundle.calibration_seed_profile.nodes[:-1],
  )
  mismatched_bundle = replace(
    runtime.runtime_bundle,
    calibration_seed_profile=mismatched_seed,
  )
  with test_case.assertRaisesRegex(
    ValueError,
    "runtime speed grid differ",
  ):
    build_learning_status_payload(
      finalization=after,
      runtime_bundle=mismatched_bundle,
      drive_baseline=baseline,
    )


def _test_status_write_failures_and_corrupt_restore_fail_closed(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  cp = generic_car_params()
  sm = FakeSubMaster()
  params = FakeParams(tmp_path)
  logger = FakeLogger()

  def runtime_factory(car_params, storage_root):
    return build_persistent_learning_runtime(
      car_params=car_params,
      storage_root=storage_root,
      provisional_rack_dynamics=rack_dynamics(),
      interface_registry=generic_registry(),
    )

  daemon = BlatV2LearnerDaemon(
    sm=sm,
    params=params,
    storage_root=tmp_path / "learning",
    car_params_decoder=lambda _encoded: cp,
    runtime_factory=runtime_factory,
    logger=logger,
    route_owned_persistence=False,
  )
  stopped = fake_messages(cp, started=False)
  started = fake_messages(cp, started=True)
  sm.publish({"deviceState": (stopped["deviceState"], 1_000_000_000)})
  daemon.step()
  params.values["CarParams"] = b"car-params"
  sm.publish({"deviceState": (started["deviceState"], 2_000_000_000)})
  daemon.step()
  publish_frame(sm, cp, 2_010_000_000)
  daemon.step()
  publish_frame(sm, cp, 2_020_000_000)
  daemon.step()
  publish_frame(sm, cp, 2_030_000_000)
  daemon.step()
  prior = {"preserved": True}
  params.values[LEARNING_STATUS_PARAM] = prior

  assert daemon.runtime is not None
  with patch.object(
    daemon.runtime,
    "transition_offroad_and_persist",
    side_effect=OSError("injected persistence failure"),
  ):
    sm.publish({"deviceState": (stopped["deviceState"], 3_000_000_000)})
    daemon.step()
  assert params.values[LEARNING_STATUS_PARAM] is prior
  assert any("offroad persist failed" in item for item in logger.exceptions)

  corrupt_root = tmp_path / "corrupt"
  corrupt_paths = runtime_factory(cp, corrupt_root).artifact_paths
  corrupt_paths.root.mkdir(parents=True)
  corrupt_paths.evidence.write_bytes(b"corrupt")
  corrupt_paths.manifest.write_bytes(b"corrupt")
  corrupt_params = FakeParams(tmp_path / "corrupt-params")
  previous_status = {"previous_valid_status": True}
  corrupt_params.values[LEARNING_STATUS_PARAM] = previous_status
  # The step retries once after restore; keep both attempts failed so the next
  # offroad poll proves retry state survives the whole cycle.
  corrupt_params.remove_failures = 2
  corrupt_sm = FakeSubMaster()
  corrupt = BlatV2LearnerDaemon(
    sm=corrupt_sm,
    params=corrupt_params,
    storage_root=corrupt_root,
    car_params_decoder=lambda _encoded: cp,
    runtime_factory=runtime_factory,
    logger=FakeLogger(),
    route_owned_persistence=False,
  )
  corrupt_sm.publish(
    {
      "deviceState": (stopped["deviceState"], 4_000_000_000),
    }
  )
  corrupt.step()
  assert corrupt.runtime is None
  assert corrupt_params.values[LEARNING_STATUS_PARAM] is previous_status
  assert corrupt._learning_status_clear_pending
  assert not corrupt_params.removes

  # Restore attempts remain suppressed for the known-bad fingerprint, but the
  # non-authoritative stale-cache removal retries independently.
  corrupt_sm.publish(
    {
      "deviceState": (stopped["deviceState"], 4_100_000_000),
    }
  )
  corrupt.step()
  assert LEARNING_STATUS_PARAM not in corrupt_params.values
  assert not any(key == LEARNING_STATUS_PARAM for key, _, _ in corrupt_params.puts)
  assert any(
    key == LEARNING_OPERATION_STATUS_PARAM and value["state"] == "failed" and value["diagnostic"] == "runtime_restore_failed"
    for key, value, _ in corrupt_params.puts
  )
  assert corrupt_params.removes == [LEARNING_STATUS_PARAM]
  assert not corrupt._learning_status_clear_pending


def _test_route_owned_learner_never_overwrites_offroad_backfill_status(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  cp = generic_car_params()

  def make_runtime_factory(created):
    def runtime_factory(car_params, storage_root):
      runtime = build_persistent_learning_runtime(
        car_params=car_params,
        storage_root=storage_root,
        provisional_rack_dynamics=rack_dynamics(),
        interface_registry=generic_registry(),
      )
      created.append(runtime)
      return runtime

    return runtime_factory

  # Cold restore and a subsequent CURRENT revision reload both happen while
  # the backfill process exclusively owns the offroad operation projection.
  cold_params = FakeParams(tmp_path / "cold")
  cold_status = {"owner": "backfill-cold-scan"}
  cold_params.values[LEARNING_OPERATION_STATUS_PARAM] = cold_status
  cold_sm = FakeSubMaster()
  cold_created: list[PersistentLearningRuntime] = []
  cold = BlatV2LearnerDaemon(
    sm=cold_sm,
    params=cold_params,
    storage_root=tmp_path / "cold-learning",
    car_params_decoder=lambda _encoded: cp,
    runtime_factory=make_runtime_factory(cold_created),
    logger=FakeLogger(),
  )
  revisions = iter((b"generation-a", b"generation-b", b"generation-b"))
  cold._artifact_revision = lambda _runtime: next(revisions)
  stopped = fake_messages(cp, started=False)
  cold_sm.publish(
    {
      "deviceState": (stopped["deviceState"], 1_000_000_000),
    }
  )
  cold.step()
  assert len(cold_created) == 1
  assert cold_params.values[LEARNING_OPERATION_STATUS_PARAM] is cold_status
  assert not any(key == LEARNING_OPERATION_STATUS_PARAM for key, _, _ in cold_params.puts)

  pointer_flip_status = {"owner": "backfill-pointer-flip"}
  cold_params.values[LEARNING_OPERATION_STATUS_PARAM] = pointer_flip_status
  cold_sm.publish(
    {
      "deviceState": (stopped["deviceState"], 1_100_000_000),
    }
  )
  cold.step()
  assert len(cold_created) == 2
  assert cold_params.values[LEARNING_OPERATION_STATUS_PARAM] is pointer_flip_status
  assert not any(key == LEARNING_OPERATION_STATUS_PARAM for key, _, _ in cold_params.puts)

  # A confirmed exact live-CarParams mismatch leaves the runtime unbound.
  # Returning offroad with a changed persistent identity must not let learnerd
  # overwrite the status that backfilld publishes between those transitions.
  mismatch_params = FakeParams(
    tmp_path / "failed-bind",
    car_params_bytes=b"prepared-car-params",
  )
  mismatch_sm = FakeSubMaster()
  mismatch_created: list[PersistentLearningRuntime] = []
  mismatch = BlatV2LearnerDaemon(
    sm=mismatch_sm,
    params=mismatch_params,
    storage_root=tmp_path / "failed-bind-learning",
    car_params_decoder=lambda _encoded: cp,
    runtime_factory=make_runtime_factory(mismatch_created),
    logger=FakeLogger(),
  )
  mismatch_sm.publish(
    {
      "deviceState": (stopped["deviceState"], 2_000_000_000),
    }
  )
  mismatch.step()
  mismatch_params.values["CarParams"] = b"live-mismatch"
  started = fake_messages(cp, started=True)
  mismatch_sm.publish(
    {
      "deviceState": (started["deviceState"], 3_000_000_000),
    }
  )
  mismatch.step()
  assert mismatch._runtime_unavailable_this_drive
  assert mismatch_params.values[LEARNING_OPERATION_STATUS_PARAM]["state"] == "drive_skipped_identity_mismatch"

  offroad_owner_status = {"owner": "backfill-after-skipped-drive"}
  mismatch_params.values[LEARNING_OPERATION_STATUS_PARAM] = offroad_owner_status
  mismatch_params.values["CarParamsPersistent"] = b"replacement-identity"
  prior_operation_puts = sum(key == LEARNING_OPERATION_STATUS_PARAM for key, _, _ in mismatch_params.puts)
  mismatch_sm.publish(
    {
      "deviceState": (stopped["deviceState"], 4_000_000_000),
    }
  )
  mismatch.step()
  assert len(mismatch_created) == 2
  assert mismatch_params.values[LEARNING_OPERATION_STATUS_PARAM] is offroad_owner_status
  assert sum(key == LEARNING_OPERATION_STATUS_PARAM for key, _, _ in mismatch_params.puts) == prior_operation_puts

  # Manager flips IsOffroad before the last deviceState update reaches the
  # live process. Its stale local `_onroad` observation must not let that
  # process overwrite the newly active backfill owner.
  mismatch._onroad = True
  mismatch_params.values["IsOffroad"] = True
  transition_owner_status = {"owner": "backfill-manager-transition"}
  mismatch_params.values[LEARNING_OPERATION_STATUS_PARAM] = transition_owner_status
  prior_operation_puts = sum(key == LEARNING_OPERATION_STATUS_PARAM for key, _, _ in mismatch_params.puts)
  mismatch._publish_operation_status(
    state=LearningOperationState.COLLECTING,
    diagnostic="collecting_current_drive",
  )
  assert mismatch_params.values[LEARNING_OPERATION_STATUS_PARAM] is transition_owner_status
  assert sum(key == LEARNING_OPERATION_STATUS_PARAM for key, _, _ in mismatch_params.puts) == prior_operation_puts


class TestBlatV2LearnerDaemon(unittest.TestCase):
  def setUp(self):
    self.temporary_directory = tempfile.TemporaryDirectory()
    self.tmp_path = Path(self.temporary_directory.name)
    module = ModuleType(GENERIC_CONTROLLER_MODULE)
    module.CarControllerParams = GenericControllerParams
    self.module_patcher = patch.dict(
      sys.modules,
      {GENERIC_CONTROLLER_MODULE: module},
    )
    self.module_patcher.start()

  def tearDown(self):
    self.module_patcher.stop()
    self.temporary_directory.cleanup()

  def test_palisade_409_4_7_boundary_classifier(self):
    _test_palisade_409_4_7_boundary_classifier()

  def test_generic_non_hyundai_construction_uses_detected_opendbc_limits(self):
    _test_generic_non_hyundai_construction_uses_detected_opendbc_limits(None)

  def test_runtime_collects_only_onroad_and_persists_only_offroad(self):
    _test_runtime_collects_only_onroad_and_persists_only_offroad(
      self.tmp_path,
      None,
    )

  def test_restart_restores_exact_cross_drive_evidence(self):
    _test_restart_restores_exact_cross_drive_evidence(self.tmp_path, None)

  def test_restore_corruption_and_seed_mismatch_fail_closed(self):
    _test_restore_corruption_and_seed_mismatch_fail_closed(
      self.tmp_path,
      None,
      self,
    )

  def test_clean_frame_filters_reject_invalid_but_include_limit_boundaries(self):
    _test_clean_frame_filters_reject_invalid_but_include_limit_boundaries(
      self.tmp_path,
      None,
    )

  def test_vehicle_owned_slew_and_full_torque_are_accepted_evidence(self):
    _test_vehicle_owned_slew_and_full_torque_are_accepted_evidence(
      self.tmp_path,
      None,
    )

  def test_input_contract_and_subscriptions_exclude_intent_and_requests(self):
    _test_input_contract_and_subscriptions_exclude_intent_and_requests()

  def test_canonical_history_resolves_one_update_race_and_rejects_stale(self):
    _test_canonical_history_resolves_one_update_race_and_rejects_stale()

  def test_runtime_uses_delayed_command_not_same_frame_torque(self):
    _test_runtime_uses_delayed_command_not_same_frame_torque(
      self.tmp_path,
      None,
    )

  def test_runtime_retains_unsigned_reversal_without_fitting_it(self):
    _test_runtime_retains_unsigned_reversal_without_fitting_it(
      self.tmp_path,
      None,
    )

  def test_daemon_observes_offroad_without_onroad_poll_and_restores(self):
    _test_daemon_observes_offroad_without_onroad_poll_and_restores(
      self.tmp_path,
      None,
    )

  def test_mid_drive_start_skips_collection_until_offroad_preparation(self):
    _test_mid_drive_start_skips_collection_until_offroad_preparation(
      self.tmp_path,
      None,
    )

  def test_car_params_precedence_fallback_and_transient_absence(self):
    _test_car_params_precedence_fallback_and_transient_absence(
      self.tmp_path,
      None,
    )

  def test_live_identity_late_and_mismatch_never_cross_contaminate(self):
    _test_live_identity_late_and_mismatch_never_cross_contaminate(
      self.tmp_path,
      None,
    )

  def test_learning_status_is_canonical_strict_and_drive_local(self):
    _test_learning_status_is_canonical_strict_and_drive_local(
      self.tmp_path,
      None,
      self,
    )

  def test_status_write_failures_and_corrupt_restore_fail_closed(self):
    _test_status_write_failures_and_corrupt_restore_fail_closed(
      self.tmp_path,
      None,
    )

  def test_route_owned_learner_never_overwrites_offroad_backfill_status(self):
    _test_route_owned_learner_never_overwrites_offroad_backfill_status(
      self.tmp_path,
      None,
    )
