from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import struct
import tempfile

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_spool import _encode_frame
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import MeasuredLearningFrame
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  ControlsWitness,
  DrivingEventLocator,
  LateralManeuverPlanPublication,
  LiveDelayPublication,
  LiveTorqueParametersPublication,
  ModelPublication,
  RouteEvidenceArtifact,
  RouteEvidenceError,
  RouteEvidenceSourceIdentity,
  RouteEvidenceStore,
  inspect_route_evidence_file,
)


def frame(mono: int) -> MeasuredLearningFrame:
  return MeasuredLearningFrame(
    sample_mono_ns=mono, response_mono_ns=mono - 1,
    applied_report_mono_ns=mono - 2, applied_effective_mono_ns=mono - 3,
    speed_mps=5.0, steering_angle_deg=2.0, steering_rate_deg_s=1.0,
    steering_torque=0.1, applied_torque=0.2, steering_pressed=False,
    standstill=False, steer_fault_temporary=False,
    steer_fault_permanent=False, can_valid=True, can_timeout=False,
    lateral_active=True, live_parameters_valid=True, angle_offset_valid=True,
    steer_ratio_valid=True, stiffness_factor_valid=True, angle_offset_deg=0.0,
    steer_ratio=15.0, stiffness_factor=1.0, roll_rad=0.0, inputs_valid=True,
  )


FRAMES = (frame(1_000), frame(1_010))
PHYSICAL = b"".join(_encode_frame(value) for value in FRAMES)
CP = b"canonical-car-params"


def source() -> RouteEvidenceSourceIdentity:
  return RouteEvidenceSourceIdentity(
    route_id="000000b7--a6b3b1f175", route_time_origin_mono_ns=900,
    route_segment_sha256=("a" * 64,), route_segment_size_bytes=(1234,),
    source_superproject_commit="1" * 40, source_opendbc_commit="2" * 40,
    source_panda_commit="3" * 40, controller_source_kind="stock_canonical",
    controller_artifact_sha256="4" * 64, behavior_eligible=True,
    behavior_ineligible_reason="eligible", vehicle_identity="HYUNDAI_PALISADE",
    runtime_identity="5" * 64,
    schema_versions={"extractor": 3, "route_evidence": 2},
    preparation_provenance={"extractor_schema_version": 3},
    physical_plane_encoding_id="blatv2-measured-learning-frame-v1",
    physical_record_count=2, preparation_cache_key="6" * 64,
    controls_witness_count=2, unresolved_witness_count=0, gap_count=0,
    model_link_failure_count=0,
  )


def model(index: int) -> ModelPublication:
  return ModelPublication(
    segment_index=0, ordinal=index, mono_time_ns=950 + index,
    frame_id=100 + index, timestamp_eof_ns=940 + index,
    scalar_curvature=-0.0 if index == 0 else 0.01,
    desired_curvature_time_s=0.2, plan_times=(0.0, 0.05),
    orientation_rate_z=(0.0, 0.01), velocity_x=(5.0, 5.0),
    message_valid=True, native_grid_valid=True,
  )


def witness(index: int, raw: float = -0.0) -> ControlsWitness:
  return ControlsWitness(
    segment_index=0, ordinal=index, mono_time_ns=1_000 + index * 10,
    physical_record_index=index, model_publication_index=index,
    live_torque_parameters_index=0, live_delay_index=0,
    lateral_maneuver_plan_index=0, poll_mono_time_ns=990 + index * 10,
    state_sample_mono_ns=980 + index * 10, live_parameters_mono_ns=970,
    car_output_report_mono_ns=965 + index * 10,
    car_output_effective_mono_ns=955 + index * 10,
    car_control_mono_ns=1_001 + index * 10, raw_request_torque=raw,
    measured_curvature=-0.01, desired_curvature=-0.012,
    envelope_headroom=1.0 - abs(raw), torque_output_can_count=20 + index,
    message_valid=True, model_message_alive=True, model_link_valid=True,
    inputs_valid=True, lateral_active=True, driver_intervening=False,
    steer_fault=False, intervention_onset=False,
    intervention_onset_uncertain=False, race_unresolved=False,
    gap_from_previous=False, car_control_paired=True,
    torque_output_can_valid=True, maneuver_plan_available=True,
    live_torque_parameters_available=True, live_delay_available=True,
    live_torque_parameters_checks_passed=True,
    live_torque_parameters_health_exact=True,
  )


TORQUE = (LiveTorqueParametersPublication(0, 0, 940, 2.5, 0.0, 0.1, 1, True, True, True),)
DELAY = (LiveDelayPublication(0, 0, 941, 0.12, 1, True, "valid"),)
MANEUVER = (LateralManeuverPlanPublication(0, 0, 942, -0.012, True),)
EVENTS = (DrivingEventLocator(0, 0, 1_020, 1_000, 6.0, 2.0, "event-1", "lat.turnStopTurn", "warning", True),)


def artifact(raw: float = -0.0) -> RouteEvidenceArtifact:
  return RouteEvidenceArtifact(
    source(), CP, PHYSICAL, (model(0), model(1)),
    (witness(0, raw), witness(1, 0.25)), TORQUE, DELAY, MANEUVER, EVENTS,
  )


def test_deterministic_roundtrip_and_exact_planes() -> None:
  first = artifact()
  restored = RouteEvidenceArtifact.from_bytes(first.canonical_bytes)
  assert restored.canonical_bytes == first.canonical_bytes
  assert restored.sha256 == hashlib.sha256(first.canonical_bytes).hexdigest()
  assert bytes(restored.car_params_bytes) == CP
  assert bytes(restored.physical_bytes) == PHYSICAL
  assert tuple(restored.iter_physical_frames()) == FRAMES
  assert restored.model_publications == first.model_publications
  assert restored.control_witnesses == first.control_witnesses
  assert restored.live_torque_parameters == TORQUE
  assert restored.live_delays == DELAY
  assert restored.lateral_maneuver_plans == MANEUVER
  assert restored.event_locators == EVENTS
  assert struct.pack("<d", restored.model_publications[0].scalar_curvature) == struct.pack("<Q", 1 << 63)
  assert struct.pack("<d", restored.control_witnesses[0].raw_request_torque) == struct.pack("<Q", 1 << 63)


def test_invalid_native_model_grid_is_preserved_empty() -> None:
  invalid = replace(
    model(0), plan_times=(), orientation_rate_z=(), velocity_x=(),
    native_grid_valid=False, message_valid=False,
  )
  evidence = RouteEvidenceArtifact(
    replace(source(), model_link_failure_count=1), CP, PHYSICAL,
    (invalid, model(1)), (witness(0), witness(1)), TORQUE, DELAY,
    MANEUVER, EVENTS,
  )
  restored = RouteEvidenceArtifact.from_bytes(evidence.canonical_bytes)
  assert restored.model_publications[0].native_grid_valid is False
  assert restored.model_publications[0].plan_times == ()

  with pytest.raises(RouteEvidenceError, match="must be empty"):
    replace(invalid, plan_times=(0.0,))


def test_rejects_old_spool_and_corrupt_sections() -> None:
  with pytest.raises(RouteEvidenceError, match="unsupported"):
    RouteEvidenceArtifact.from_bytes(b"BLATSP01" + b"\0" * 100)
  encoded = bytearray(artifact().canonical_bytes)
  encoded[-1] ^= 1
  with pytest.raises(RouteEvidenceError):
    RouteEvidenceArtifact.from_bytes(encoded)
  with pytest.raises(RouteEvidenceError):
    RouteEvidenceArtifact.from_bytes(artifact().canonical_bytes + b"x")


def test_indices_counts_and_exact_can_count_are_closed() -> None:
  with pytest.raises(RouteEvidenceError):
    RouteEvidenceArtifact(
      source(), CP, PHYSICAL, (model(0), model(1)),
      (replace(witness(0), model_publication_index=9), witness(1)),
      TORQUE, DELAY, MANEUVER, EVENTS,
    )
  with pytest.raises(RouteEvidenceError, match="canonical zero"):
    replace(witness(0), torque_output_can_valid=False)
  with pytest.raises(RouteEvidenceError, match="size/count"):
    RouteEvidenceArtifact(
      source(), CP, PHYSICAL[:-1], (model(0), model(1)),
      (witness(0), witness(1)), TORQUE, DELAY, MANEUVER, EVENTS,
    )


def test_store_requires_exact_aa_and_is_immutable(tmp_path: Path) -> None:
  store = RouteEvidenceStore(tmp_path / "evidence")
  with pytest.raises(RouteEvidenceError, match="A/A mismatch"):
    store.publish(artifact(), artifact(0.1))
  assert not store.root.exists()

  expected = artifact()
  store.publish(expected, artifact())
  assert store.lookup(expected.source_key).canonical_bytes == expected.canonical_bytes  # type: ignore[union-attr]
  assert store.load(expected.sha256).canonical_bytes == expected.canonical_bytes
  object_path = store.root / "objects" / f"{expected.sha256}.route-evidence"
  object_path.write_bytes(object_path.read_bytes()[:-1] + b"x")
  with pytest.raises(RouteEvidenceError, match="hash mismatch"):
    store.load(expected.sha256)


def test_store_rejects_symlinks(tmp_path: Path) -> None:
  target = tmp_path / "target"
  target.mkdir()
  linked = tmp_path / "linked"
  linked.symlink_to(target, target_is_directory=True)
  with pytest.raises(RouteEvidenceError, match="unsafe"):
    RouteEvidenceStore(linked).publish(artifact(), artifact())


def test_streamed_inspection_authenticates_shape_and_store_object(tmp_path: Path) -> None:
  expected = artifact()
  first = tmp_path / "first.route-evidence"
  second = tmp_path / "second.route-evidence"
  first.write_bytes(expected.canonical_bytes)
  second.write_bytes(expected.canonical_bytes)
  summary = inspect_route_evidence_file(first)
  assert summary.sha256 == expected.sha256
  assert summary.source_identity == expected.source_identity
  assert summary.physical_size == len(PHYSICAL)

  store = RouteEvidenceStore(tmp_path / "store")
  store.publish_files(
    first,
    second,
    sha256=expected.sha256,
    source_key=expected.source_key,
  )
  assert store.inspect(expected.sha256).sha256 == expected.sha256


def test_streamed_inspection_rejects_manifest_count_section_disagreement(tmp_path: Path) -> None:
  encoded = artifact().canonical_bytes.replace(
    b'"model_publication_count":2',
    b'"model_publication_count":9',
  )
  assert encoded != artifact().canonical_bytes
  path = tmp_path / "invalid-count.route-evidence"
  path.write_bytes(encoded)
  with pytest.raises(RouteEvidenceError, match="model section size/count"):
    inspect_route_evidence_file(path)


def test_publish_files_rejects_same_size_mutation_after_aa(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  expected = artifact()
  first = tmp_path / "first.route-evidence"
  second = tmp_path / "second.route-evidence"
  first.write_bytes(expected.canonical_bytes)
  second.write_bytes(expected.canonical_bytes)
  store = RouteEvidenceStore(tmp_path / "store")

  original_mkstemp = tempfile.mkstemp
  mutated = False

  def mutate_after_aa(*args: object, **kwargs: object) -> tuple[int, str]:
    nonlocal mutated
    result = original_mkstemp(*args, **kwargs)
    if not mutated:
      mutated = True
      with first.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(bytes([expected.canonical_bytes[-1] ^ 1]))
        stream.flush()
        os.fsync(stream.fileno())
    return result

  monkeypatch.setattr(tempfile, "mkstemp", mutate_after_aa)
  with pytest.raises(RouteEvidenceError, match="changed during publication"):
    store.publish_files(
      first,
      second,
      sha256=expected.sha256,
      source_key=expected.source_key,
    )
  assert not (store.root / "objects" / f"{expected.sha256}.route-evidence").exists()
