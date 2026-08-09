from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.lib.blatv2.device_acceptance import (
  DeviceAcceptanceError,
  DeviceAcceptanceReceipt,
  PERCENTILE_METHOD,
  build_device_acceptance_receipt,
)
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  ControlsWitness,
  RouteEvidenceArtifact,
  RouteEvidenceError,
  RouteEvidenceSourceIdentity,
)
from openpilot.selfdrive.controls.tests.test_blatv2_route_evidence import (
  artifact as route_artifact,
)


def write_artifact(
  path: Path,
  *,
  source: RouteEvidenceSourceIdentity | None = None,
  witnesses: tuple[ControlsWitness, ...] | None = None,
) -> Path:
  base = route_artifact()
  rebuilt = RouteEvidenceArtifact(
    base.source_identity if source is None else source,
    base.car_params_bytes,
    base.physical_bytes,
    base.model_publications,
    base.control_witnesses if witnesses is None else witnesses,
    base.live_torque_parameters,
    base.live_delays,
    base.lateral_maneuver_plans,
    base.event_locators,
  )
  path.write_bytes(rebuilt.canonical_bytes)
  return path


def write_three_witness_artifact(
  path: Path,
  witnesses: tuple[ControlsWitness, ControlsWitness, ControlsWitness],
) -> Path:
  base = route_artifact()
  physical = bytes(base.physical_bytes)
  record_size = len(physical) // len(base.control_witnesses)
  rebuilt = RouteEvidenceArtifact(
    replace(
      base.source_identity,
      physical_record_count=3,
      controls_witness_count=3,
    ),
    base.car_params_bytes,
    physical + physical[-record_size:],
    base.model_publications,
    witnesses,
    base.live_torque_parameters,
    base.live_delays,
    base.lateral_maneuver_plans,
    base.event_locators,
  )
  path.write_bytes(rebuilt.canonical_bytes)
  return path


def test_receipt_is_canonical_deduplicated_and_uses_nearest_rank(
  tmp_path: Path,
) -> None:
  path = write_artifact(tmp_path / "route.route-evidence")
  receipt = build_device_acceptance_receipt((path, path))
  assert receipt.passed
  assert receipt.sample_count == 2
  assert receipt.percentile_method == PERCENTILE_METHOD
  assert receipt.compute_p50_seconds == 0.001
  assert receipt.compute_p90_seconds == 0.002
  assert receipt.compute_p99_seconds == 0.002
  assert receipt.compute_max_seconds == 0.002
  assert receipt.drop_count == 0
  assert len(receipt.route_evidence_sha256s) == 1
  assert DeviceAcceptanceReceipt.from_json(receipt.to_json()) == receipt


@pytest.mark.parametrize(
  ("changes", "reason"),
  (
    ({"modular_compute_time_seconds": 0.010}, "compute_budget_exceeded"),
    ({"modular_control_cadence_valid": False}, "control_cadence_invalid"),
    ({"modular_intent_status": 6}, "intent_not_ok"),
    ({"modular_safety_state": 2, "modular_invalid_frames": 1}, "invalid_frames_nonzero"),
    ({"modular_recovery_ok_frames": 1}, "recovery_frames_nonzero"),
    ({"modular_selection_bound": False}, "modular_binding_invalid"),
    ({"modular_controls_valid": False}, "controls_state_invalid"),
    ({"modular_car_control_valid": False}, "car_control_invalid"),
    ({"modular_vehicle_state_valid": False}, "vehicle_state_invalid"),
    ({"modular_live_parameters_valid": False}, "live_parameters_invalid"),
    ({"modular_horizon_valid": False}, "horizon_invalid"),
    ({"modular_adapter_exception": True}, "adapter_exception"),
    ({"modular_production_envelope_verified": False}, "production_envelope_unverified"),
    (
      {
        "torque_output_can_count": 0,
        "torque_output_can_valid": False,
        "modular_final_expected_counts": 0,
      },
      "actuator_correspondence_invalid",
    ),
    (
      {
        "inputs_valid": False,
        "steering_request_active": False,
        "steering_request_active_valid": False,
        "steering_request_fault_avoidance_counter_valid": False,
      },
      "actuator_correspondence_invalid",
    ),
    (
      {
        "inputs_valid": False,
        "steering_request_fault_avoidance_counter_valid": False,
      },
      "actuator_correspondence_invalid",
    ),
    (
      {
        "modular_final_expected_counts": 0,
        "modular_final_count_match_valid": False,
      },
      "actuator_correspondence_invalid",
    ),
    ({"modular_final_expected_counts": 0}, "actuator_correspondence_invalid"),
    (
      {
        "modular_final_count_residual": 1,
        "modular_final_limiter_altered": True,
      },
      "actuator_correspondence_invalid",
    ),
    ({"modular_final_limiter_altered": True}, "actuator_correspondence_invalid"),
  ),
)
def test_any_active_frame_failure_vetoes(
  tmp_path: Path,
  changes: dict[str, object],
  reason: str,
) -> None:
  base = route_artifact()
  witnesses = (replace(base.control_witnesses[0], **changes), base.control_witnesses[1])
  path = write_artifact(tmp_path / "failed.route-evidence", witnesses=witnesses)
  receipt = build_device_acceptance_receipt((path,))
  assert not receipt.passed
  assert receipt.failure_count(reason) > 0


def test_high_angle_request_suppression_is_not_clean_receipt_evidence(
  tmp_path: Path,
) -> None:
  base = route_artifact()
  suppressed = replace(
    base.control_witnesses[0],
    inputs_valid=False,
    steering_request_active=False,
  )
  receipt = build_device_acceptance_receipt(
    (
      write_artifact(
        tmp_path / "high-angle-suppression.route-evidence",
        witnesses=(suppressed, base.control_witnesses[1]),
      ),
    )
  )
  assert not receipt.passed
  assert receipt.failure_count("actuator_correspondence_invalid") == 1
  assert sum(count for _, count in receipt.failure_counts) == 1


@pytest.mark.parametrize(
  "changes",
  (
    {
      "modular_selection": 0,
      "modular_selection_bound": False,
      "modular_active": True,
    },
    {"lateral_active": False},
  ),
)
def test_contradictory_modular_claims_cannot_be_filtered_out(
  tmp_path: Path,
  changes: dict[str, object],
) -> None:
  base = route_artifact()
  witnesses = (
    replace(base.control_witnesses[0], **changes),
    base.control_witnesses[1],
  )
  receipt = build_device_acceptance_receipt(
    (
      write_artifact(
        tmp_path / "contradictory.route-evidence",
        witnesses=witnesses,
      ),
    )
  )
  assert not receipt.passed
  assert receipt.failure_count("modular_binding_invalid") == 1


def test_active_session_cannot_silently_drop_modular_claim(
  tmp_path: Path,
) -> None:
  base = route_artifact()
  omitted = replace(
    base.control_witnesses[1],
    modular_selection=0,
    modular_selection_bound=False,
    modular_active=False,
  )
  receipt = build_device_acceptance_receipt(
    (
      write_artifact(
        tmp_path / "omitted-active.route-evidence",
        witnesses=(base.control_witnesses[0], omitted),
      ),
    )
  )
  assert not receipt.passed
  assert receipt.failure_count("modular_binding_invalid") == 1


def test_active_nonmodular_frame_does_not_reset_modular_gap_history(
  tmp_path: Path,
) -> None:
  base = route_artifact()
  first, second = base.control_witnesses
  middle = replace(
    second,
    modular_selection=0,
    modular_selection_bound=False,
    modular_active=False,
  )
  final = replace(
    second,
    ordinal=2,
    mono_time_ns=first.mono_time_ns + 20,
    physical_record_index=2,
    modular_control_witness_mono_ns=(first.modular_control_witness_mono_ns + 20_000_000),
  )
  receipt = build_device_acceptance_receipt(
    (
      write_three_witness_artifact(
        tmp_path / "active-gap.route-evidence",
        (first, middle, final),
      ),
    )
  )
  assert not receipt.passed
  assert receipt.failure_count("inferred_cadence_gap") == 1


def test_inactive_boundary_resets_gap_and_proves_one_bootstrap_frame(
  tmp_path: Path,
) -> None:
  base = route_artifact()
  first, second = base.control_witnesses
  inactive = replace(
    second,
    lateral_active=False,
    modular_selection=0,
    modular_selection_bound=False,
    modular_active=False,
  )
  final = replace(
    second,
    ordinal=2,
    mono_time_ns=first.mono_time_ns + 30,
    physical_record_index=2,
    modular_control_witness_mono_ns=(first.modular_control_witness_mono_ns + 30_000_000),
    modular_final_expected_counts=0,
    modular_final_count_residual=0,
    modular_final_count_match_valid=False,
    modular_final_limiter_altered=False,
  )
  receipt = build_device_acceptance_receipt(
    (
      write_three_witness_artifact(
        tmp_path / "inactive-bootstrap.route-evidence",
        (first, inactive, final),
      ),
    )
  )
  assert receipt.passed
  assert receipt.sample_count == 2
  assert receipt.drop_count == 0


def test_inactive_modular_binding_neither_bootstraps_nor_hides_gap(
  tmp_path: Path,
) -> None:
  base = route_artifact()
  first, second = base.control_witnesses
  inactive_bound = replace(
    second,
    lateral_active=False,
    modular_selection=1,
    modular_selection_bound=True,
    modular_active=False,
  )
  final = replace(
    second,
    ordinal=2,
    mono_time_ns=first.mono_time_ns + 30,
    physical_record_index=2,
    modular_control_witness_mono_ns=(first.modular_control_witness_mono_ns + 30_000_000),
    modular_final_expected_counts=0,
    modular_final_count_residual=0,
    modular_final_count_match_valid=False,
    modular_final_limiter_altered=False,
  )
  receipt = build_device_acceptance_receipt(
    (
      write_three_witness_artifact(
        tmp_path / "inactive-bound-gap.route-evidence",
        (first, inactive_bound, final),
      ),
    )
  )
  assert not receipt.passed
  assert receipt.failure_count("modular_binding_invalid") == 1
  assert receipt.failure_count("actuator_correspondence_invalid") == 1
  assert receipt.failure_count("inferred_cadence_gap") == 1


@pytest.mark.parametrize(
  ("changes", "reason", "has_drop"),
  (
    ({"unresolved_witness_count": 1}, "unresolved_witness_summary_nonzero", False),
    ({"gap_count": 1}, "route_gap_summary_nonzero", True),
    (
      {
        "controls_witness_count": 3,
        "unresolved_witness_count": 1,
        "pre_poll_dropped_timestamps_ns": (999,),
      },
      "pre_poll_witness_dropped",
      True,
    ),
  ),
)
def test_nonzero_route_summary_defects_veto_receipt(
  tmp_path: Path,
  changes: dict[str, object],
  reason: str,
  has_drop: bool,
) -> None:
  base = route_artifact()
  path = write_artifact(
    tmp_path / "summary-defect.route-evidence",
    source=replace(base.source_identity, **changes),
  )
  receipt = build_device_acceptance_receipt((path,))
  assert not receipt.passed
  assert receipt.failure_count(reason) > 0
  assert (receipt.drop_count > 0) is has_drop


def test_recorded_and_inferred_cadence_gaps_veto(tmp_path: Path) -> None:
  base = route_artifact()
  recorded = replace(base.control_witnesses[1], gap_from_previous=True)
  receipt = build_device_acceptance_receipt(
    (
      write_artifact(
        tmp_path / "recorded.route-evidence",
        witnesses=(base.control_witnesses[0], recorded),
      ),
    )
  )
  assert not receipt.passed
  assert receipt.drop_count == 1
  assert receipt.failure_count("recorded_cadence_gap") == 1

  inferred = replace(
    base.control_witnesses[1],
    modular_control_witness_mono_ns=(base.control_witnesses[0].modular_control_witness_mono_ns + 20_000_000),
  )
  receipt = build_device_acceptance_receipt(
    (
      write_artifact(
        tmp_path / "inferred.route-evidence",
        witnesses=(base.control_witnesses[0], inferred),
      ),
    )
  )
  assert not receipt.passed
  assert receipt.drop_count == 1
  assert receipt.failure_count("inferred_cadence_gap") == 1


def test_missing_telemetry_and_non_comma_device_are_ineligible(
  tmp_path: Path,
) -> None:
  base = route_artifact()
  missing = tuple(
    replace(
      witness,
      modular_compute_time_seconds=0.0,
      modular_control_witness_mono_ns=0,
      modular_invalid_frames=0,
      modular_recovery_ok_frames=0,
      modular_intent_status=0,
      modular_safety_state=0,
      modular_telemetry_available=False,
      modular_controls_valid=False,
      modular_car_control_valid=False,
      modular_vehicle_state_valid=False,
      modular_live_parameters_valid=False,
      modular_horizon_valid=False,
      modular_control_cadence_valid=False,
      modular_adapter_exception=False,
      modular_production_envelope_verified=False,
      modular_final_expected_counts=0,
      modular_final_count_residual=0,
      modular_final_count_match_valid=False,
      modular_final_limiter_altered=False,
    )
    for witness in base.control_witnesses
  )
  receipt = build_device_acceptance_receipt(
    (
      write_artifact(
        tmp_path / "legacy.route-evidence",
        witnesses=missing,
      ),
    )
  )
  assert not receipt.passed
  assert receipt.failure_count("telemetry_unavailable") == 2

  pc_source = replace(base.source_identity, device_type="pc")
  receipt = build_device_acceptance_receipt(
    (
      write_artifact(
        tmp_path / "pc.route-evidence",
        source=pc_source,
      ),
    )
  )
  assert not receipt.passed
  assert receipt.failure_count("non_comma_device") == 1


def test_identity_mismatch_and_tampering_fail_closed(tmp_path: Path) -> None:
  base = route_artifact()
  first = write_artifact(tmp_path / "first.route-evidence")
  second_source = replace(
    base.source_identity,
    recorded_profile_sha256="b" * 64,
  )
  second = write_artifact(
    tmp_path / "second.route-evidence",
    source=second_source,
  )
  receipt = build_device_acceptance_receipt((first, second))
  assert not receipt.passed
  assert receipt.failure_count("profile_identity_mismatch") == 1

  tampered = bytearray(first.read_bytes())
  tampered[-1] ^= 1
  first.write_bytes(tampered)
  with pytest.raises(RouteEvidenceError):
    build_device_acceptance_receipt((first,))

  noncanonical = json.dumps(
    receipt.to_param(),
    sort_keys=False,
    separators=(",", ":"),
  )
  with pytest.raises(DeviceAcceptanceError):
    DeviceAcceptanceReceipt.from_json(noncanonical)

  invalid_route_identity = receipt.to_param()
  invalid_route_identity["routeEvidenceSha256s"] = [1]
  with pytest.raises(DeviceAcceptanceError):
    DeviceAcceptanceReceipt.from_json(
      json.dumps(
        invalid_route_identity,
        sort_keys=True,
        separators=(",", ":"),
      )
    )
