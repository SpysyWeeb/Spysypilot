from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from types import SimpleNamespace
import unittest

from openpilot.selfdrive.controls.lib.blatv2.calibration_coordinator import (
  CalibrationLearningFinalization,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_learner import (
  CalibrationLearningResult,
  CalibrationNodeQualificationReport,
  CalibrationQualificationReason,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  CalibrationProfileNode,
  VehicleCalibrationProfile,
  make_calibration_seed_profile,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_status import (
  DriveEvidenceBaseline,
  LEARNING_STATUS_SCHEMA_VERSION,
  build_learning_status_bytes,
  build_learning_status_payload,
  decode_learning_status,
  validate_learning_status_payload,
)


@dataclass(frozen=True)
class _RuntimeBundle:
  vehicle_identity: str
  calibration_identity_sha256: str
  calibration_seed_profile: VehicleCalibrationProfile
  # A deliberately different legacy identity proves the projection selects
  # the calibration identity rather than the retired physical-profile one.
  identity_sha256: str = "f" * 64


def _seed() -> VehicleCalibrationProfile:
  return make_calibration_seed_profile(
    vehicle_identity="status-v2-car",
    torque_callback_slope=0.42,
    stock_friction_torque=0.06,
    transport_delay_s=0.12,
    rack_rate_resolution_deg_s=4.0,
    speed_nodes_mps=(0.0, 10.0),
  )


def _candidate(seed: VehicleCalibrationProfile) -> VehicleCalibrationProfile:
  nodes = tuple(
    CalibrationProfileNode(
      speed_mps=node.speed_mps,
      parameters=replace(
        node.parameters,
        torque_per_lateral_accel=0.34 + 0.01 * index,
        lateral_accel_offset_correction_mps2=-0.04 + 0.01 * index,
        kinetic_friction_torque=0.03,
        static_breakaway_torque=0.09,
        confidence=0.8,
        qualified=True,
      ),
      base_support_s=20.0,
      base_sample_count=200,
      moving_support_s=12.0,
      moving_sample_count=120,
      breakaway_support_s=8.0,
      breakaway_sample_count=80,
      validation_count=80,
      inverse_calibration_validation_rms=0.05,
      breakaway_validation_rms=0.04,
    )
    for index, node in enumerate(seed.nodes)
  )
  return VehicleCalibrationProfile(
    vehicle_identity=seed.vehicle_identity,
    revision=3,
    provenance="test-only observable calibration",
    nodes=nodes,
  )


def _report(
  node_index: int,
  speed_mps: float,
  parameters,
  *,
  qualified: bool = True,
) -> CalibrationNodeQualificationReport:
  reasons = (
    (CalibrationQualificationReason.QUALIFIED,)
    if qualified
    else (CalibrationQualificationReason.INSUFFICIENT_BREAKAWAY_EVIDENCE,)
  )
  return CalibrationNodeQualificationReport(
    node_index=node_index,
    speed_mps=speed_mps,
    minimum_support_s=30.0,
    clean_support_s=40.0,
    supported_sample_count=400,
    training_count=320,
    validation_count=80,
    validation_support_s=8.0,
    base_support_s=20.0,
    base_sample_count=200,
    moving_support_s=12.0,
    moving_sample_count=120,
    moving_training_count=96,
    moving_validation_count=24,
    breakaway_support_s=8.0,
    breakaway_sample_count=80,
    breakaway_training_count=64,
    breakaway_validation_count=16,
    lateral_accel_span_mps2=1.2,
    lateral_accel_rms_mps2=0.35,
    rack_travel_deg=240.0,
    applied_torque_span=0.75,
    rack_reversals=12,
    lateral_accel_directions=2,
    applied_torque_directions=2,
    seed_validation_rms=0.11,
    candidate_validation_rms=0.06,
    moving_seed_validation_rms=0.12,
    moving_candidate_validation_rms=0.07,
    breakaway_seed_validation_rms=0.13,
    breakaway_candidate_validation_rms=0.08,
    confidence=0.8,
    reasons=reasons,
    candidate_parameters=parameters,
    authority_support_s=5.0,
    authority_sample_count=50,
    authority_fit_support_s=3.0,
    authority_fit_sample_count=30,
    authority_training_count=24,
    authority_validation_count=6,
    authority_seed_validation_rms=0.14,
    authority_candidate_validation_rms=0.09,
  )


def _fixtures(*, qualified: bool = True):
  seed = _seed()
  candidate = _candidate(seed)
  reports = tuple(
    _report(
      index,
      node.speed_mps,
      candidate.nodes[index].parameters,
      qualified=qualified,
    )
    for index, node in enumerate(seed.nodes)
  )
  profile = candidate if qualified else None
  candidate_json = None if profile is None else profile.to_json().encode()
  finalization = CalibrationLearningFinalization(
    manifest_bytes=b"status-v2-manifest",
    manifest_sha256="a" * 64,
    evidence_bytes=b"status-v2-evidence",
    evidence_sha256="b" * 64,
    candidate_profile_json=candidate_json,
    candidate_profile_sha256=(
      None if candidate_json is None else hashlib.sha256(candidate_json).hexdigest()
    ),
    learning_result=CalibrationLearningResult(reports, profile),
  )
  runtime = _RuntimeBundle(seed.vehicle_identity, "c" * 64, seed)
  return runtime, finalization


def _baseline() -> DriveEvidenceBaseline:
  diagnostics = tuple(
    SimpleNamespace(
      node_index=index,
      speed_mps=speed,
      clean_support_s=33.0,
      supported_sample_count=330,
      base_support_s=18.0,
      base_sample_count=180,
      moving_support_s=9.0,
      moving_sample_count=90,
      breakaway_support_s=6.0,
      breakaway_sample_count=60,
      authority_support_s=4.0,
      authority_sample_count=40,
      authority_fit_support_s=2.0,
      authority_fit_sample_count=20,
    )
    for index, speed in enumerate((0.0, 10.0))
  )
  return DriveEvidenceBaseline.from_support_diagnostics(diagnostics)


def test_schema_v2_roundtrip_identity_observable_parameters_and_deltas() -> None:
  runtime, finalization = _fixtures()
  baseline = _baseline()
  payload = build_learning_status_payload(
    finalization=finalization,
    runtime_bundle=runtime,
    drive_baseline=baseline,
  )
  encoded = build_learning_status_bytes(
    finalization=finalization,
    runtime_bundle=runtime,
    drive_baseline=baseline,
  )

  assert payload["schema_version"] == LEARNING_STATUS_SCHEMA_VERSION == 2
  assert payload["runtime_identity_sha256"] == runtime.calibration_identity_sha256
  assert payload["runtime_identity_sha256"] != runtime.identity_sha256
  assert payload["seed_profile_sha256"] == hashlib.sha256(
    runtime.calibration_seed_profile.to_json().encode(),
  ).hexdigest()
  assert payload["all_nodes_qualified"] is True
  assert payload["candidate_profile_sha256"] == finalization.candidate_profile_sha256
  assert decode_learning_status(encoded) == payload
  with unittest.TestCase().assertRaisesRegex(ValueError, "not canonical"):
    decode_learning_status(encoded + b" ")
  assert encoded == json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  ).encode()

  node = payload["nodes"][0]
  assert node["candidate_parameters"] == {
    "kinetic_friction_torque": 0.03,
    "lateral_accel_offset_correction_mps2": -0.04,
    "static_breakaway_torque": 0.09,
    "torque_per_lateral_accel": 0.34,
  }
  assert node["last_drive_clean_support_s"] == 7.0
  assert node["last_drive_accepted_sample_count"] == 70
  assert node["last_drive_base_support_s"] == 2.0
  assert node["last_drive_base_sample_count"] == 20
  assert node["last_drive_moving_support_s"] == 3.0
  assert node["last_drive_moving_sample_count"] == 30
  assert node["last_drive_breakaway_support_s"] == 2.0
  assert node["last_drive_breakaway_sample_count"] == 20
  assert node["last_drive_authority_support_s"] == 1.0
  assert node["last_drive_authority_sample_count"] == 10
  assert node["last_drive_authority_fit_support_s"] == 1.0
  assert node["last_drive_authority_fit_sample_count"] == 10


def test_decoder_rejects_legacy_schema_rack_fields_and_unknown_reasons() -> None:
  test_case = unittest.TestCase()
  runtime, finalization = _fixtures()
  payload = build_learning_status_payload(
    finalization=finalization,
    runtime_bundle=runtime,
    drive_baseline=None,
  )

  legacy = json.loads(json.dumps(payload))
  legacy["schema_version"] = 1
  with test_case.assertRaisesRegex(ValueError, "version/authority marker"):
    decode_learning_status(json.dumps(legacy, sort_keys=True, separators=(",", ":")))

  rack_field = json.loads(json.dumps(payload))
  rack_field["nodes"][0]["candidate_parameters"]["rack_gain_deg_s2_per_torque"] = 4000.0
  with test_case.assertRaisesRegex(ValueError, "candidate_parameters schema differs"):
    validate_learning_status_payload(rack_field)

  unknown_reason = json.loads(json.dumps(payload))
  unknown_reason["nodes"][0]["qualified"] = False
  unknown_reason["nodes"][0]["reasons"] = ["invented_reason"]
  unknown_reason["all_nodes_qualified"] = False
  unknown_reason["candidate_profile_sha256"] = None
  unknown_reason["candidate_profile_revision"] = None
  with test_case.assertRaisesRegex(ValueError, "qualification reason is unknown"):
    validate_learning_status_payload(unknown_reason)


def test_baseline_rejects_any_population_moving_backwards() -> None:
  test_case = unittest.TestCase()
  runtime, finalization = _fixtures()
  baseline = _baseline()
  first = finalization.learning_result.node_reports[0]
  backwards = replace(first, authority_fit_sample_count=19)
  bad_result = CalibrationLearningResult(
    (backwards, *finalization.learning_result.node_reports[1:]),
    finalization.learning_result.candidate_profile,
  )
  bad_finalization = replace(finalization, learning_result=bad_result)

  with test_case.assertRaisesRegex(ValueError, "moved backwards"):
    build_learning_status_payload(
      finalization=bad_finalization,
      runtime_bundle=runtime,
      drive_baseline=baseline,
    )


def test_candidate_identity_requires_every_node_qualified() -> None:
  test_case = unittest.TestCase()
  runtime, finalization = _fixtures(qualified=False)
  assert build_learning_status_payload(
    finalization=finalization,
    runtime_bundle=runtime,
    drive_baseline=None,
  )["candidate_profile_sha256"] is None

  forged = replace(finalization, candidate_profile_sha256="d" * 64)
  with test_case.assertRaisesRegex(ValueError, "candidate hash and node qualification disagree"):
    build_learning_status_payload(
      finalization=forged,
      runtime_bundle=runtime,
      drive_baseline=None,
    )
