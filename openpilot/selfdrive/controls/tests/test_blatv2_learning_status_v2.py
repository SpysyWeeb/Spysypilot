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
  CalibrationFitStatus,
  CalibrationCrossFitModelDiagnostic,
  CalibrationInterpolationQualificationReport,
  CalibrationIntervalStratum,
  CalibrationIntervalStratumDiagnostic,
  CalibrationIndependentRouteCounts,
  CalibrationCrossFitStatus,
  CalibrationLearningResult,
  CalibrationModelFitDiagnostic,
  CalibrationModelId,
  CalibrationNodeQualificationReport,
  CalibrationPairedLossDiagnostic,
  CalibrationQualificationReason,
  CalibrationSampleAccounting,
  CalibrationSampleDisposition,
  MIN_STRATUM_TRAINING_ROWS,
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


# Exact support populations from the first four-worker device generation.
# Repeated binary64 accumulation puts the independently tracked clean total a
# few ULPs away from the sum of its mutually exclusive populations. The
# authoritative evidence validator accepts this ordinary rounding pattern,
# so the display-only projection must accept the same bytes.
_DEVICE_SUPPORT_POPULATIONS_HEX = (
  ("0x1.0a44139e49b93p+8", "0x1.d3042eaf59bcfp+7", "0x1.fd8f2a8005819p+4", "0x1.d2133d3905d82p-1"),
  ("0x1.62b638afc1237p+9", "0x1.3bfb958f8bbd7p+9", "0x1.2dba9292cfb84p+6", "0x1.0350cddb6e06dp+1"),
  ("0x1.03a6a45be81c4p+10", "0x1.d8618befd2dcap+9", "0x1.6d15043517f6ap+6", "0x1.491c415a53a3bp+1"),
  ("0x1.22eb64ebea10ep+10", "0x1.0e8ec3b25b6e8p+10", "0x1.3be940d524edfp+6", "0x1.3c1a5878ad3f9p+1"),
  ("0x1.8de1131e56d21p+9", "0x1.77a8f0c2c4053p+9", "0x1.5a9fb39f2b29fp+5", "0x1.1c4e434035d3dp+0"),
  ("0x1.21c8b4ef5f334p+8", "0x1.1730de571cafcp+8", "0x1.38815f1017158p+3", "0x1.a7973f83963a6p-1"),
)
_DEVICE_SUPPORT_POPULATIONS = tuple(
  tuple(float.fromhex(value) for value in population)
  for population in _DEVICE_SUPPORT_POPULATIONS_HEX
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
      cross_fit_route_count=80,
      full_fit_candidate_rms=0.05,
      breakaway_full_fit_candidate_rms=0.04,
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
    (CalibrationQualificationReason.LEARNED,)
    if qualified
    else (CalibrationQualificationReason.INSUFFICIENT_BREAKAWAY_EVIDENCE,)
  )
  paired_loss = CalibrationPairedLossDiagnostic(
    2, -0.01, 0.002, -0.012, -0.008, 1e-14
  )
  neutral_loss = CalibrationPairedLossDiagnostic(
    2, 0.0, 0.0, 0.0, 0.0, 1e-14
  )
  fit_diagnostics = tuple(
    CalibrationModelFitDiagnostic(
      model=model,
      status=CalibrationFitStatus.IDENTIFIABLE,
      moving_rank=rank,
      moving_parameter_count=rank,
      condition_estimate=1.0,
      breakaway_rank=1,
      breakaway_parameter_count=1,
    )
    for model, rank in (
      (CalibrationModelId.STATIC_ONLY, 0),
      (CalibrationModelId.FRICTION_MAP, 1),
      (CalibrationModelId.OFFSET_AND_FRICTION, 2),
      (CalibrationModelId.FULL_MAP, 3),
    )
  )
  return CalibrationNodeQualificationReport(
    node_index=node_index,
    speed_mps=speed_mps,
    minimum_support_s=30.0,
    clean_support_s=40.0,
    supported_sample_count=400,
    full_fit_count=320,
    cross_fit_route_count=2 if qualified else 0,
    base_support_s=20.0,
    base_sample_count=200,
    moving_support_s=12.0,
    moving_sample_count=120,
    moving_full_fit_count=96,
    moving_cross_fit_route_count=2,
    breakaway_support_s=8.0,
    breakaway_sample_count=80,
    breakaway_full_fit_count=64,
    breakaway_cross_fit_route_count=2,
    breakaway_episode_full_fit_count=64 if qualified else 0,
    breakaway_episode_cross_fit_route_count=2 if qualified else 0,
    lateral_accel_span_mps2=1.2,
    lateral_accel_rms_mps2=0.35,
    rack_travel_deg=240.0,
    applied_torque_span=0.75,
    rack_reversals=12,
    lateral_accel_directions=2,
    applied_torque_directions=2,
    full_fit_seed_rms=0.11,
    full_fit_candidate_rms=0.06,
    moving_full_fit_seed_rms=0.12,
    moving_full_fit_candidate_rms=0.07,
    breakaway_full_fit_seed_rms=0.13,
    breakaway_full_fit_candidate_rms=0.08,
    confidence=0.8,
    reasons=reasons,
    candidate_parameters=parameters,
    authority_support_s=5.0,
    authority_sample_count=50,
    authority_fit_support_s=3.0,
    authority_fit_sample_count=30,
    authority_full_fit_count=24,
    authority_cross_fit_route_count=2,
    authority_full_fit_seed_rms=0.14,
    authority_full_fit_candidate_rms=0.09,
    fit_diagnostics=fit_diagnostics,
    full_fit_paired_loss=(
      CalibrationPairedLossDiagnostic(2, -0.02, 0.005, -0.025, -0.015, 1e-14)
      if qualified
      else None
    ),
    cross_fit_paired_loss=(
      paired_loss
      if qualified
      else None
    ),
    selection_outcome=(
      CalibrationQualificationReason.LEARNED if qualified else None
    ),
    independent_route_counts=CalibrationIndependentRouteCounts(2, 2, 2, 2, 2, 2),
    cross_fit_diagnostics=tuple(
      CalibrationCrossFitModelDiagnostic(
        model=diagnostic.model,
        status=(
          CalibrationCrossFitStatus.SCORED
          if qualified and diagnostic.model is CalibrationModelId.FULL_MAP
          else CalibrationCrossFitStatus.NO_ROBUST_IMPROVEMENT
        ),
        contributing_route_count=2,
        successful_fold_count=2,
        failed_fold_count=0,
        paired_loss=(
          paired_loss
          if qualified and diagnostic.model is CalibrationModelId.FULL_MAP
          else neutral_loss
        ),
      )
      for diagnostic in fit_diagnostics
    ),
    full_fit_diagnostic=fit_diagnostics[-1] if qualified else None,
    unresolved_diagnostics=(),
    full_fit_stratum_paired_losses=(paired_loss,) * 4 if qualified else (),
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
  interpolation_reports = (
    (
      CalibrationInterpolationQualificationReport(
        interval_index=0,
        lower_speed_mps=seed.nodes[0].speed_mps,
        upper_speed_mps=seed.nodes[1].speed_mps,
        stratum_diagnostics=(CalibrationIntervalStratumDiagnostic(
          stratum=CalibrationIntervalStratum.BASE,
          full_fit_paired_loss=CalibrationPairedLossDiagnostic(
            2, -0.02, 0.005, -0.025, -0.015, 1e-14
          ),
          cross_fit_paired_loss=CalibrationPairedLossDiagnostic(
            2, -0.01, 0.002, -0.012, -0.008, 1e-14
          ),
          contributing_route_count=2,
          successful_fold_count=2,
          failed_fold_count=0,
          cross_fit_status=CalibrationCrossFitStatus.SCORED,
        ),),
        reasons=(CalibrationQualificationReason.QUALIFIED,),
        contributing_route_count=2,
        successful_fold_count=2,
        failed_fold_count=0,
        cross_fit_status=CalibrationCrossFitStatus.SCORED,
      ),
    )
    if qualified
    else ()
  )
  candidate_json = None if profile is None else profile.to_json().encode()
  sample_accounting = (
    CalibrationSampleAccounting.empty()
    .with_disposition(CalibrationSampleDisposition.ACCEPTED)
    .with_disposition(CalibrationSampleDisposition.LATERAL_INACTIVE)
  )
  finalization = CalibrationLearningFinalization(
    manifest_bytes=b"status-v2-manifest",
    manifest_sha256="a" * 64,
    evidence_bytes=b"status-v2-evidence",
    evidence_sha256="b" * 64,
    selected_profile_json=candidate_json,
    selected_profile_sha256=(
      None if candidate_json is None else hashlib.sha256(candidate_json).hexdigest()
    ),
    candidate_profile_json=candidate_json,
    candidate_profile_sha256=(
      None if candidate_json is None else hashlib.sha256(candidate_json).hexdigest()
    ),
    learning_result=CalibrationLearningResult(
      reports,
      profile,
      interpolation_reports,
      selected_profile=profile,
    ),
    sample_accounting=sample_accounting,
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


def test_schema_v7_roundtrip_identity_observable_parameters_and_deltas() -> None:
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

  assert payload["schema_version"] == LEARNING_STATUS_SCHEMA_VERSION == 9
  assert payload["runtime_identity_sha256"] == runtime.calibration_identity_sha256
  assert payload["runtime_identity_sha256"] != runtime.identity_sha256
  assert payload["seed_profile_sha256"] == hashlib.sha256(
    runtime.calibration_seed_profile.to_json().encode(),
  ).hexdigest()
  assert payload["all_nodes_qualified"] is True
  assert payload["all_nodes_evaluated"] is True
  assert payload["all_intervals_qualified"] is True
  assert payload["candidate_profile_available"] is True
  assert payload["candidate_profile_sha256"] == finalization.candidate_profile_sha256
  assert payload["sample_accounting"] == finalization.sample_accounting.to_payload()
  assert payload["nodes"][0]["independent_route_counts"]["all"] == 2
  assert len(payload["nodes"][0]["cross_fit_diagnostics"]) == 4
  assert payload["nodes"][0]["full_fit_diagnostic"]["model"] == "full_map"
  assert payload["interpolation_reports"][0]["stratum_diagnostics"][0]["stratum"] == "base"
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
  assert node["evaluation_status"] == "learned"
  assert node["selection_outcome"] == "learned"
  assert len(node["fit_diagnostics"]) == 4
  assert node["fit_diagnostics"][-1]["moving_rank"] == 3
  assert node["full_fit_paired_loss"]["route_count"] == 2
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


def test_fully_evaluated_cross_fit_regression_is_not_reported_as_pending() -> None:
  runtime, finalization = _fixtures(qualified=False)
  regressed_loss = CalibrationPairedLossDiagnostic(
    2, 0.02, 0.005, 0.015, 0.025, 1e-14
  )
  rejected_reports = tuple(
    replace(
      report,
      reasons=(CalibrationQualificationReason.CROSS_FIT_REGRESSION,),
      selection_outcome=CalibrationQualificationReason.LEARNED,
      breakaway_episode_full_fit_count=MIN_STRATUM_TRAINING_ROWS,
      full_fit_paired_loss=regressed_loss,
      full_fit_diagnostic=report.fit_diagnostics[-1],
      full_fit_stratum_paired_losses=(regressed_loss,) * 4,
    )
    for report in finalization.learning_result.node_reports
  )
  rejected = replace(
    finalization,
    learning_result=CalibrationLearningResult(rejected_reports, None, ()),
  )

  payload = build_learning_status_payload(
    finalization=rejected,
    runtime_bundle=runtime,
    drive_baseline=None,
  )

  assert payload["all_nodes_evaluated"] is True
  assert payload["all_nodes_qualified"] is False
  assert payload["all_intervals_qualified"] is False
  assert payload["candidate_profile_available"] is False
  assert all(
    node["evaluation_status"] == "cross_fit_regressed"
    for node in payload["nodes"]
  )


def test_qualified_stratum_proof_is_identity_ordered_and_non_regressing() -> None:
  test_case = unittest.TestCase()
  runtime, finalization = _fixtures()
  payload = build_learning_status_payload(
    finalization=finalization,
    runtime_bundle=runtime,
    drive_baseline=None,
  )

  regressed = json.loads(json.dumps(payload))
  loss = regressed["nodes"][0]["full_fit_stratum_paired_losses"][0]["paired_loss"]
  loss.update({
    "lower_bound_mse": 0.01,
    "mean_candidate_minus_seed_mse": 0.02,
    "uncertainty_mse": 0.01,
    "upper_bound_mse": 0.03,
  })
  with test_case.assertRaisesRegex(ValueError, "qualified node carries failed proof"):
    validate_learning_status_payload(regressed)

  reordered = json.loads(json.dumps(payload))
  losses = reordered["nodes"][0]["full_fit_stratum_paired_losses"]
  losses[0], losses[1] = losses[1], losses[0]
  with test_case.assertRaisesRegex(ValueError, "stratum ordering"):
    validate_learning_status_payload(reordered)

  wrong_population = json.loads(json.dumps(payload))
  wrong_population["nodes"][0]["full_fit_stratum_paired_losses"][0][
    "paired_loss"
  ]["route_count"] += 1
  with test_case.assertRaisesRegex(ValueError, "route population disagrees"):
    validate_learning_status_payload(wrong_population)

def test_device_accumulation_rounding_projects_without_hiding_snapshot() -> None:
  diagnostics = tuple(
    SimpleNamespace(
      node_index=index,
      speed_mps=speed,
      clean_support_s=population[0],
      supported_sample_count=3,
      base_support_s=population[1],
      base_sample_count=1,
      moving_support_s=population[2],
      moving_sample_count=1,
      breakaway_support_s=population[3],
      breakaway_sample_count=1,
      authority_support_s=1.0,
      authority_sample_count=1,
      authority_fit_support_s=0.5,
      authority_fit_sample_count=1,
    )
    for index, (speed, population) in enumerate(zip(
      (0.0, 5.0, 10.0, 15.0, 20.0, 30.0),
      _DEVICE_SUPPORT_POPULATIONS,
      strict=True,
    ))
  )
  baseline = DriveEvidenceBaseline.from_support_diagnostics(diagnostics)
  assert tuple(node.clean_support_s for node in baseline.nodes) == tuple(
    population[0] for population in _DEVICE_SUPPORT_POPULATIONS
  )

  runtime, finalization = _fixtures()
  payload = build_learning_status_payload(
    finalization=finalization,
    runtime_bundle=runtime,
    drive_baseline=None,
  )
  node = payload["nodes"][0]
  for clean, base, moving, breakaway in _DEVICE_SUPPORT_POPULATIONS:
    node["clean_support_s"] = clean
    node["base_support_s"] = base
    node["moving_support_s"] = moving
    node["breakaway_support_s"] = breakaway
    validate_learning_status_payload(payload)


def test_display_rejects_material_support_population_disagreement() -> None:
  runtime, finalization = _fixtures()
  payload = build_learning_status_payload(
    finalization=finalization,
    runtime_bundle=runtime,
    drive_baseline=None,
  )
  node = payload["nodes"][0]
  node["clean_support_s"] = (
    node["base_support_s"]
    + node["moving_support_s"]
    + node["breakaway_support_s"]
    + 1e-6
  )
  with unittest.TestCase().assertRaisesRegex(
    ValueError,
    "clean support populations disagree",
  ):
    validate_learning_status_payload(payload)


def test_decoder_rejects_legacy_schema_rack_fields_and_unknown_reasons() -> None:
  test_case = unittest.TestCase()
  runtime, finalization = _fixtures()
  payload = build_learning_status_payload(
    finalization=finalization,
    runtime_bundle=runtime,
    drive_baseline=None,
  )

  legacy = json.loads(json.dumps(payload))
  legacy["schema_version"] = 2
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

  bad_accounting = json.loads(json.dumps(payload))
  bad_accounting["sample_accounting"]["rejection_reasons"].pop(
    CalibrationSampleDisposition.LATERAL_INACTIVE.value,
  )
  with test_case.assertRaisesRegex(ValueError, "accounting is invalid"):
    validate_learning_status_payload(bad_accounting)

  inconsistent_accounting = json.loads(json.dumps(payload))
  inconsistent_accounting["sample_accounting"]["ingested_sample_count"] = 1
  with test_case.assertRaisesRegex(ValueError, "totals disagree"):
    validate_learning_status_payload(inconsistent_accounting)

  missing_proof = json.loads(json.dumps(payload))
  missing_proof["nodes"][0]["cross_fit_diagnostics"] = []
  with test_case.assertRaisesRegex(ValueError, "nonempty"):
    validate_learning_status_payload(missing_proof)


def test_schema_v7_rejects_cross_fit_semantic_tampering() -> None:
  test_case = unittest.TestCase()
  runtime, finalization = _fixtures()
  payload = build_learning_status_payload(
    finalization=finalization,
    runtime_bundle=runtime,
    drive_baseline=None,
  )

  moving_legacy = json.loads(json.dumps(payload))
  moving_legacy["nodes"][0]["moving_cross_fit_route_count"] += 1
  with test_case.assertRaisesRegex(ValueError, "moving_cross_fit_route_count disagrees"):
    validate_learning_status_payload(moving_legacy)

  moving_population = json.loads(json.dumps(payload))
  moving_population["nodes"][0]["independent_route_counts"]["moving"] = 1
  with test_case.assertRaisesRegex(ValueError, "moving_cross_fit_route_count disagrees"):
    validate_learning_status_payload(moving_population)

  loss_routes = json.loads(json.dumps(payload))
  selected = loss_routes["nodes"][0]["cross_fit_diagnostics"][-1]
  selected["paired_loss"]["route_count"] = 1
  selected["paired_loss"]["uncertainty_mse"] = None
  selected["paired_loss"]["lower_bound_mse"] = None
  selected["paired_loss"]["upper_bound_mse"] = None
  with test_case.assertRaisesRegex(ValueError, "paired-loss route count disagrees"):
    validate_learning_status_payload(loss_routes)

  selected_status = json.loads(json.dumps(payload))
  selected_node = selected_status["nodes"][0]
  selected_diagnostic = selected_node["cross_fit_diagnostics"][-1]
  selected_diagnostic["status"] = CalibrationCrossFitStatus.NO_ROBUST_IMPROVEMENT.value
  selected_diagnostic["paired_loss"].update({
    "mean_candidate_minus_seed_mse": 0.0,
    "uncertainty_mse": 0.0,
    "lower_bound_mse": 0.0,
    "upper_bound_mse": 0.0,
  })
  selected_node["cross_fit_paired_loss"] = json.loads(json.dumps(
    selected_diagnostic["paired_loss"]
  ))
  with test_case.assertRaisesRegex(ValueError, "selected family status contradicts outcome"):
    validate_learning_status_payload(selected_status)

  interval_status = json.loads(json.dumps(payload))
  interval_status["interpolation_reports"][0]["qualified"] = False
  interval_status["interpolation_reports"][0]["reasons"] = [
    CalibrationQualificationReason.CROSS_FIT_INCONCLUSIVE.value
  ]
  stratum = interval_status["interpolation_reports"][0]["stratum_diagnostics"][0]
  stratum["cross_fit_status"] = (
    CalibrationCrossFitStatus.NO_ROBUST_IMPROVEMENT.value
  )
  stratum["cross_fit_paired_loss"].update({
    "mean_candidate_minus_seed_mse": 0.0,
    "uncertainty_mse": 0.1,
    "lower_bound_mse": -0.1,
    "upper_bound_mse": 0.1,
  })
  interval_status["all_intervals_qualified"] = False
  with test_case.assertRaisesRegex(ValueError, "aggregate status contradicts strata"):
    validate_learning_status_payload(interval_status)

  interval_reason = json.loads(json.dumps(interval_status))
  interval_reason["interpolation_reports"][0]["cross_fit_status"] = (
    CalibrationCrossFitStatus.NO_ROBUST_IMPROVEMENT.value
  )
  interval_reason["interpolation_reports"][0]["reasons"] = [
    CalibrationQualificationReason.INTERPOLATION_CROSS_FIT_REGRESSION.value
  ]
  with test_case.assertRaisesRegex(ValueError, "reasons contradict stratum diagnostics"):
    validate_learning_status_payload(interval_reason)

  unqualified_runtime, unqualified_finalization = _fixtures(qualified=False)
  node_reason = build_learning_status_payload(
    finalization=unqualified_finalization,
    runtime_bundle=unqualified_runtime,
    drive_baseline=None,
  )
  node_reason["nodes"][0]["reasons"] = [
    CalibrationQualificationReason.SINGULAR_FIT.value
  ]
  with test_case.assertRaisesRegex(ValueError, "reasons contradict carried diagnostics"):
    validate_learning_status_payload(node_reason)


def test_baseline_rejects_any_population_moving_backwards() -> None:
  test_case = unittest.TestCase()
  runtime, finalization = _fixtures()
  baseline = _baseline()
  first = finalization.learning_result.node_reports[0]
  backwards = replace(first, authority_fit_sample_count=19)
  bad_result = CalibrationLearningResult(
    (backwards, *finalization.learning_result.node_reports[1:]),
    finalization.learning_result.candidate_profile,
    finalization.learning_result.interpolation_reports,
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
  with test_case.assertRaisesRegex(ValueError, "candidate hash and profile availability disagree"):
    build_learning_status_payload(
      finalization=forged,
      runtime_bundle=runtime,
      drive_baseline=None,
    )


def test_unqualified_reason_set_is_derived_from_carried_prerequisites() -> None:
  runtime, finalization = _fixtures(qualified=False)
  payload = build_learning_status_payload(
    finalization=finalization,
    runtime_bundle=runtime,
    drive_baseline=None,
  )

  fabricated = json.loads(json.dumps(payload))
  fabricated["nodes"][0]["reasons"].append(
    CalibrationQualificationReason.INSUFFICIENT_SUPPORT.value,
  )
  with unittest.TestCase().assertRaisesRegex(
    ValueError,
    "reasons contradict carried diagnostics",
  ):
    validate_learning_status_payload(fabricated)

  missing = json.loads(json.dumps(payload))
  missing["nodes"][0]["reasons"] = [
    CalibrationQualificationReason.CROSS_FIT_INCONCLUSIVE.value,
  ]
  with unittest.TestCase().assertRaisesRegex(
    ValueError,
    "reasons contradict carried diagnostics",
  ):
    validate_learning_status_payload(missing)

  for extra in (
    CalibrationQualificationReason.CROSS_FIT_REGRESSION,
    CalibrationQualificationReason.MOVING_CROSS_FIT_REGRESSION,
    CalibrationQualificationReason.RANK_DEFICIENT_FIT,
    CalibrationQualificationReason.INVALID_PARAMETERS,
  ):
    poisoned = json.loads(json.dumps(payload))
    poisoned["nodes"][0]["reasons"].append(extra.value)
    with unittest.TestCase().assertRaisesRegex(
      ValueError,
      "reasons contradict carried diagnostics",
    ):
      validate_learning_status_payload(poisoned)

  ordered = json.loads(json.dumps(payload))
  ordered_node = ordered["nodes"][0]
  ordered_node["clean_support_s"] = ordered_node["minimum_support_s"] - 1.0
  ordered_node["base_support_s"] = (
    ordered_node["clean_support_s"]
    - ordered_node["moving_support_s"]
    - ordered_node["breakaway_support_s"]
  )
  ordered_node["reasons"] = [
    CalibrationQualificationReason.INSUFFICIENT_SUPPORT.value,
    CalibrationQualificationReason.INSUFFICIENT_BREAKAWAY_EVIDENCE.value,
  ]
  validate_learning_status_payload(ordered)
  reordered = json.loads(json.dumps(ordered))
  reordered["nodes"][0]["reasons"].reverse()
  with unittest.TestCase().assertRaisesRegex(
    ValueError,
    "reasons contradict carried diagnostics",
  ):
    validate_learning_status_payload(reordered)
  missing_legitimate = json.loads(json.dumps(ordered))
  missing_legitimate["nodes"][0]["reasons"].pop()
  with unittest.TestCase().assertRaisesRegex(
    ValueError,
    "reasons contradict carried diagnostics",
  ):
    validate_learning_status_payload(missing_legitimate)

  unresolved_poison = json.loads(json.dumps(payload))
  unresolved_poison["nodes"][0]["unresolved_diagnostics"] = [
    CalibrationQualificationReason.INVALID_PARAMETERS.value,
  ]
  with unittest.TestCase().assertRaisesRegex(
    ValueError,
    "unresolved diagnostics contradict proof",
  ):
    validate_learning_status_payload(unresolved_poison)


def test_breakaway_population_and_reason_cannot_be_mutated_independently() -> None:
  runtime, finalization = _fixtures(qualified=False)
  payload = build_learning_status_payload(
    finalization=finalization,
    runtime_bundle=runtime,
    drive_baseline=None,
  )
  payload["nodes"][0]["breakaway_episode_full_fit_count"] = (
    MIN_STRATUM_TRAINING_ROWS
  )

  with unittest.TestCase().assertRaisesRegex(
    ValueError,
    "reasons contradict carried diagnostics",
  ):
    validate_learning_status_payload(payload)


def test_invalid_parameter_reason_is_derived_from_complete_predicate() -> None:
  runtime, finalization = _fixtures(qualified=False)
  payload = build_learning_status_payload(
    finalization=finalization,
    runtime_bundle=runtime,
    drive_baseline=None,
  )
  mutations = (
    ("torque_per_lateral_accel", 0.0),
    ("torque_per_lateral_accel", -0.1),
    ("kinetic_friction_torque", -0.1),
    ("kinetic_friction_torque", 0.2),
    ("lateral_accel_offset_correction_mps2", float("inf")),
  )
  for field, value in mutations:
    poisoned = json.loads(json.dumps(payload))
    poisoned["nodes"][0]["candidate_parameters"][field] = value
    with unittest.TestCase().assertRaisesRegex(
      ValueError,
      "reasons contradict carried diagnostics",
    ):
      validate_learning_status_payload(poisoned)


def test_all_seed_qualified_result_has_no_candidate_artifact() -> None:
  runtime, finalization = _fixtures()
  zero_loss = CalibrationPairedLossDiagnostic(
    2, 0.0, 0.0, 0.0, 0.0, 1e-14
  )
  reports = tuple(
    replace(
      report,
      reasons=(CalibrationQualificationReason.SEED_RETAINED,),
      candidate_parameters=runtime.calibration_seed_profile.nodes[index].parameters,
      selection_outcome=CalibrationQualificationReason.SEED_RETAINED,
      full_fit_paired_loss=zero_loss,
      cross_fit_paired_loss=zero_loss,
      cross_fit_diagnostics=tuple(
        replace(
          diagnostic,
          status=CalibrationCrossFitStatus.NO_ROBUST_IMPROVEMENT,
          paired_loss=zero_loss,
        )
        for diagnostic in report.cross_fit_diagnostics
      ),
      full_fit_stratum_paired_losses=(),
    )
    for index, report in enumerate(finalization.learning_result.node_reports)
  )
  intervals = tuple(
    replace(
      report,
      stratum_diagnostics=tuple(
        replace(
          diagnostic,
          full_fit_paired_loss=zero_loss,
          cross_fit_paired_loss=zero_loss,
        )
        for diagnostic in report.stratum_diagnostics
      ),
    )
    for report in finalization.learning_result.interpolation_reports
  )
  all_seed = replace(
    finalization,
    candidate_profile_json=None,
    candidate_profile_sha256=None,
    learning_result=CalibrationLearningResult(reports, None, intervals),
  )
  payload = build_learning_status_payload(
    finalization=all_seed,
    runtime_bundle=runtime,
    drive_baseline=None,
  )
  assert payload["all_nodes_evaluated"] is True
  assert payload["all_intervals_qualified"] is True
  assert payload["all_nodes_qualified"] is True
  assert payload["candidate_profile_available"] is False
  assert payload["candidate_profile_sha256"] is None
  assert all(node["evaluation_status"] == "seed_retained" for node in payload["nodes"])
  assert all(node["cross_fit_paired_loss"]["route_count"] == 2 for node in payload["nodes"])
  assert all(node["full_fit_diagnostic"] is not None for node in payload["nodes"])


def test_failure_classifications_remain_distinct() -> None:
  runtime, finalization = _fixtures(qualified=False)
  cases = (
    (
      CalibrationQualificationReason.INSUFFICIENT_EXCITATION,
      "evidence_insufficient",
      None,
      None,
    ),
    (
      CalibrationQualificationReason.RANK_DEFICIENT_FIT,
      "rank_deficient",
      CalibrationFitStatus.RANK_DEFICIENT,
      None,
    ),
    (
      CalibrationQualificationReason.ILL_CONDITIONED_FIT,
      "ill_conditioned",
      CalibrationFitStatus.ILL_CONDITIONED,
      None,
    ),
    (
      CalibrationQualificationReason.CROSS_FIT_REGRESSION,
      "cross_fit_regressed",
      None,
      CalibrationQualificationReason.LEARNED,
    ),
  )
  for reason, expected, fit_status, selection_outcome in cases:
    reports = list(finalization.learning_result.node_reports)
    diagnostics = reports[0].fit_diagnostics
    if fit_status is not None:
      diagnostics = tuple(
        replace(
          diagnostic,
          status=(
            CalibrationFitStatus.NO_SOLUTION
            if fit_status is CalibrationFitStatus.RANK_DEFICIENT
            and diagnostic.moving_parameter_count == 0
            else fit_status
          ),
          moving_rank=(
            diagnostic.moving_rank - 1
            if fit_status is CalibrationFitStatus.RANK_DEFICIENT
            and diagnostic.moving_parameter_count > 0
            else diagnostic.moving_rank
          ),
          breakaway_rank=(
            diagnostic.breakaway_rank
          ),
        )
        for diagnostic in diagnostics
      )
    reports[0] = replace(
      reports[0],
      reasons=(reason,),
      fit_diagnostics=diagnostics,
      selection_outcome=selection_outcome,
      breakaway_episode_full_fit_count=(
        0
        if reason is CalibrationQualificationReason.INSUFFICIENT_BREAKAWAY_EVIDENCE
        else MIN_STRATUM_TRAINING_ROWS
      ),
      lateral_accel_span_mps2=(
        0.0
        if reason is CalibrationQualificationReason.INSUFFICIENT_EXCITATION
        else reports[0].lateral_accel_span_mps2
      ),
      full_fit_paired_loss=(
        CalibrationPairedLossDiagnostic(
          2, 0.02, 0.005, 0.015, 0.025, 1e-14
        )
        if reason is CalibrationQualificationReason.CROSS_FIT_REGRESSION
        else reports[0].full_fit_paired_loss
      ),
      full_fit_diagnostic=(
        reports[0].fit_diagnostics[-1]
        if reason is CalibrationQualificationReason.CROSS_FIT_REGRESSION
        else reports[0].full_fit_diagnostic
      ),
      full_fit_stratum_paired_losses=(
        (CalibrationPairedLossDiagnostic(
          2, 0.02, 0.005, 0.015, 0.025, 1e-14
        ),) * 4
        if reason is CalibrationQualificationReason.CROSS_FIT_REGRESSION
        else reports[0].full_fit_stratum_paired_losses
      ),
    )
    classified = replace(
      finalization,
      learning_result=CalibrationLearningResult(tuple(reports), None),
    )
    payload = build_learning_status_payload(
      finalization=classified,
      runtime_bundle=runtime,
      drive_baseline=None,
    )
    assert payload["nodes"][0]["evaluation_status"] == expected
