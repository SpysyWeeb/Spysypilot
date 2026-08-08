from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import tracemalloc

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.lib.blatv2 import route_evidence as route_evidence_module
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
  RouteEvidenceFileSummary,
  RouteEvidenceSourceIdentity,
  RouteEvidenceStreamReader,
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
    source_panda_commit="3" * 40, controller_source_kind="modular_artifact",
    controller_artifact_sha256="4" * 64, behavior_eligible=True,
    behavior_ineligible_reason="eligible", vehicle_identity="HYUNDAI_PALISADE",
    runtime_identity="5" * 64,
    schema_versions={"extractor": 3, "route_evidence": 4},
    preparation_provenance={"extractor_schema_version": 3},
    physical_plane_encoding_id="blatv2-measured-learning-frame-v1",
    physical_record_count=2, preparation_cache_key="6" * 64,
    controls_witness_count=2, unresolved_witness_count=0, gap_count=0,
    model_link_failure_count=0,
    device_type="tici",
    controller_architecture="blatv2.modular.preview-rack",
    recorded_source_openpilot_commit="1" * 40,
    recorded_opendbc_commit="2" * 40,
    live_artifact_sha256="7" * 64,
    recorded_runtime_identity_sha256="5" * 64,
    recorded_profile_sha256="8" * 64,
    recorded_controller_policy_sha256="9" * 64,
    recorded_horizon_policy_sha256="a" * 64,
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
    steering_request_fault_avoidance_counter=0,
    message_valid=True, model_message_alive=True, model_link_valid=True,
    inputs_valid=True, lateral_active=True, driver_intervening=False,
    steer_fault=False, intervention_onset=False,
    intervention_onset_uncertain=False, race_unresolved=False,
    gap_from_previous=False, car_control_paired=True,
    torque_output_can_valid=True, maneuver_plan_available=True,
    live_torque_parameters_available=True, live_delay_available=True,
    live_torque_parameters_checks_passed=True,
    live_torque_parameters_health_exact=True,
    steering_request_active=True,
    steering_request_active_valid=True,
    steering_request_fault_avoidance_counter_valid=True,
    modular_compute_time_seconds=0.001 + index * 0.001,
    modular_control_witness_mono_ns=1_000_000_000 + index * 10_000_000,
    modular_selection=1,
    modular_invalid_frames=0,
    modular_recovery_ok_frames=0,
    modular_intent_status=0,
    modular_safety_state=1,
    modular_telemetry_available=True,
    modular_active=True,
    modular_selection_bound=True,
    modular_controls_valid=True,
    modular_car_control_valid=True,
    modular_vehicle_state_valid=True,
    modular_live_parameters_valid=True,
    modular_horizon_valid=True,
    modular_control_cadence_valid=True,
    modular_adapter_exception=False,
    modular_production_envelope_verified=True,
    modular_final_expected_counts=20 + index,
    modular_final_count_residual=0,
    modular_final_count_match_valid=True,
    modular_final_limiter_altered=False,
  )


TORQUE = (LiveTorqueParametersPublication(0, 0, 940, 2.5, 0.0, 0.1, 1, True, True, True),)
DELAY = (LiveDelayPublication(0, 0, 941, 0.12, 1, True, "valid"),)
MANEUVER = (LateralManeuverPlanPublication(0, 0, 942, -0.012, True),)
EVENTS = (DrivingEventLocator(0, 0, 1_020, 1_000, 6.0, 2.0, "event-1", "lat.turnStopTurn", "warning", True),)
_ROUTE_HEADER = struct.Struct("<8sHH9Q")
_SECTION_NAMES = (
  "manifest", "car_params", "physical", "models", "controls",
  "live_torque", "live_delay", "maneuvers", "events",
)


def artifact(raw: float = -0.0) -> RouteEvidenceArtifact:
  return RouteEvidenceArtifact(
    source(), CP, PHYSICAL, (model(0), model(1)),
    (witness(0, raw), witness(1, 0.25)), TORQUE, DELAY, MANEUVER, EVENTS,
  )


def _manifest(encoded: bytes) -> dict[str, object]:
  sizes = _ROUTE_HEADER.unpack_from(encoded)[3:]
  payload = json.loads(encoded[_ROUTE_HEADER.size:_ROUTE_HEADER.size + sizes[0]])
  assert isinstance(payload, dict)
  return payload


def _mutate_section(
  encoded: bytes,
  name: str,
  mutate: Callable[[bytearray], None],
) -> bytes:
  sizes = _ROUTE_HEADER.unpack_from(encoded)[3:]
  section_index = _SECTION_NAMES.index(name)
  start = _ROUTE_HEADER.size + sum(sizes[:section_index])
  end = start + sizes[section_index]
  section = bytearray(encoded[start:end])
  mutate(section)
  assert len(section) == sizes[section_index]
  manifest = _manifest(encoded)
  section_hashes = manifest["section_sha256"]
  assert isinstance(section_hashes, dict)
  section_hashes[name] = hashlib.sha256(section).hexdigest()
  if name == "physical":
    manifest["physical_plane_sha256"] = hashlib.sha256(section).hexdigest()
  rebuilt_manifest = json.dumps(
    manifest, allow_nan=False, separators=(",", ":"), sort_keys=True,
  ).encode()
  assert len(rebuilt_manifest) == sizes[0]
  manifest_end = _ROUTE_HEADER.size + sizes[0]
  return b"".join((
    encoded[:_ROUTE_HEADER.size],
    rebuilt_manifest,
    encoded[manifest_end:start],
    section,
    encoded[end:],
  ))


def _unchecked_summary(
  path: Path,
  trusted: RouteEvidenceFileSummary,
  encoded: bytes,
) -> RouteEvidenceFileSummary:
  info = path.stat()
  return replace(
    trusted,
    path=path,
    sha256=hashlib.sha256(encoded).hexdigest(),
    manifest=_manifest(encoded),
    st_dev=info.st_dev,
    st_ino=info.st_ino,
    st_size=info.st_size,
    st_mtime_ns=info.st_mtime_ns,
    st_ctime_ns=info.st_ctime_ns,
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
  assert restored.control_witnesses[1].modular_final_expected_counts == 21
  assert restored.control_witnesses[1].modular_final_count_match_valid
  assert restored.live_torque_parameters == TORQUE
  assert restored.live_delays == DELAY
  assert restored.lateral_maneuver_plans == MANEUVER
  assert restored.event_locators == EVENTS
  assert struct.pack("<d", restored.model_publications[0].scalar_curvature) == struct.pack("<Q", 1 << 63)
  assert struct.pack("<d", restored.control_witnesses[0].raw_request_torque) == struct.pack("<Q", 1 << 63)


def test_request_cut_roundtrip_preserves_command_and_fit_censor() -> None:
  cut = replace(
    witness(0),
    torque_output_can_count=193,
    inputs_valid=False,
    steering_request_active=False,
    steering_request_active_valid=True,
    steering_request_fault_avoidance_counter=90,
    steering_request_fault_avoidance_counter_valid=True,
  )
  encoded = route_evidence_module._encode_controls((cut,))
  restored = route_evidence_module._decode_controls(
    memoryview(encoded),
    1,
  )[0]
  assert restored.torque_output_can_count == 193
  assert restored.steering_request_active_valid
  assert not restored.steering_request_active
  assert restored.steering_request_fault_avoidance_counter == 90
  assert restored.steering_request_fault_avoidance_counter_valid
  assert not restored.inputs_valid


def test_v4_control_wire_retains_exact_final_correspondence() -> None:
  expected = replace(
    witness(0),
    modular_final_expected_counts=19,
    modular_final_count_residual=1,
    modular_final_count_match_valid=True,
    modular_final_limiter_altered=True,
  )
  encoded = route_evidence_module._encode_controls((expected,))
  assert len(encoded) == route_evidence_module._CONTROL.size == 185
  assert route_evidence_module._decode_controls(
    memoryview(encoded),
    1,
  ) == (expected,)


def test_physically_valid_control_requires_explicit_active_request() -> None:
  with pytest.raises(RouteEvidenceError, match="active steering request"):
    replace(
      witness(0),
      steering_request_active=False,
      steering_request_active_valid=False,
      steering_request_fault_avoidance_counter=0,
      steering_request_fault_avoidance_counter_valid=False,
    )


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

  version_three = bytearray(artifact().canonical_bytes)
  struct.pack_into("<8sH", version_three, 0, b"BLATRE03", 3)
  with pytest.raises(RouteEvidenceError, match="unsupported"):
    RouteEvidenceArtifact.from_bytes(version_three)


@pytest.mark.parametrize("mutation", ("noncanonical_bool", "nonfinite_numeric"))
def test_physical_records_are_canonical_in_eager_inspector_and_stream_paths(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  mutation: str,
) -> None:
  expected = artifact()
  path = tmp_path / f"physical-{mutation}.route-evidence"
  path.write_bytes(expected.canonical_bytes)
  trusted = inspect_route_evidence_file(path)

  def mutate_physical(section: bytearray) -> None:
    if mutation == "noncanonical_bool":
      # Four int64 clocks and nine float64 values precede the twelve bools;
      # lateral_active is the seventh bool.
      section[struct.calcsize("<4q9d") + 6] = 2
    else:
      struct.pack_into("<d", section, struct.calcsize("<4q"), math.nan)

  forged = _mutate_section(expected.canonical_bytes, "physical", mutate_physical)
  with pytest.raises(RouteEvidenceError, match="physical"):
    RouteEvidenceArtifact.from_bytes(forged)

  path.write_bytes(forged)
  with pytest.raises(RouteEvidenceError, match="physical"):
    inspect_route_evidence_file(path)

  summary = _unchecked_summary(path, trusted, forged)
  monkeypatch.setattr(
    route_evidence_module,
    "inspect_route_evidence_file",
    lambda _: summary,
  )
  with RouteEvidenceStreamReader(path) as stream:
    with pytest.raises(RouteEvidenceError, match="physical"):
      tuple(stream.iter_physical_frames())


def test_event_ids_are_unique_in_direct_eager_inspector_and_stream_paths(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  second = replace(
    EVENTS[0],
    ordinal=1,
    publication_mono_time_ns=1_021,
    occurred_mono_time_ns=1_001,
    event_id="event-2",
  )
  duplicate = replace(second, event_id=EVENTS[0].event_id)
  with pytest.raises(RouteEvidenceError, match="not unique"):
    RouteEvidenceArtifact(
      source(), CP, PHYSICAL, (model(0), model(1)),
      (witness(0), witness(1)), TORQUE, DELAY, MANEUVER,
      (EVENTS[0], duplicate),
    )

  expected = RouteEvidenceArtifact(
    source(), CP, PHYSICAL, (model(0), model(1)),
    (witness(0), witness(1)), TORQUE, DELAY, MANEUVER,
    (EVENTS[0], second),
  )
  path = tmp_path / "duplicate-event-id.route-evidence"
  path.write_bytes(expected.canonical_bytes)
  trusted = inspect_route_evidence_file(path)

  def duplicate_event_id(section: bytearray) -> None:
    assert section.count(b"event-1") == 1
    assert section.count(b"event-2") == 1
    section[:] = section.replace(b"event-2", b"event-1")

  forged = _mutate_section(expected.canonical_bytes, "events", duplicate_event_id)
  with pytest.raises(RouteEvidenceError, match="not unique"):
    RouteEvidenceArtifact.from_bytes(forged)

  path.write_bytes(forged)
  with pytest.raises(RouteEvidenceError, match="not unique"):
    inspect_route_evidence_file(path)

  summary = _unchecked_summary(path, trusted, forged)
  monkeypatch.setattr(
    route_evidence_module,
    "inspect_route_evidence_file",
    lambda _: summary,
  )
  with RouteEvidenceStreamReader(path) as stream:
    with pytest.raises(RouteEvidenceError, match="not unique"):
      tuple(stream.iter_event_locators())


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
  with store.open_stream(expected.sha256) as stream:
    assert tuple(stream.iter_control_witnesses()) == expected.control_witnesses


def test_stream_reader_matches_every_eager_plane_frame_for_frame(tmp_path: Path) -> None:
  expected = artifact()
  path = tmp_path / "stream.route-evidence"
  path.write_bytes(expected.canonical_bytes)

  with RouteEvidenceStreamReader(path) as stream:
    assert stream.summary.sha256 == expected.sha256
    assert stream.read_car_params_bytes() == bytes(expected.car_params_bytes)
    assert tuple(stream.iter_physical_frames()) == tuple(expected.iter_physical_frames())
    assert tuple(stream.iter_model_publications()) == expected.model_publications
    assert tuple(stream.iter_control_witnesses()) == expected.control_witnesses
    assert tuple(stream.iter_live_torque_parameters()) == expected.live_torque_parameters
    assert tuple(stream.iter_live_delays()) == expected.live_delays
    assert tuple(stream.iter_lateral_maneuver_plans()) == expected.lateral_maneuver_plans
    assert tuple(stream.iter_event_locators()) == expected.event_locators

  with pytest.raises(RouteEvidenceError, match="closed"):
    stream.read_car_params_bytes()


@pytest.mark.parametrize("mutation", ("tamper", "truncate"))
def test_stream_reader_rejects_changes_to_held_artifact(
  tmp_path: Path,
  mutation: str,
) -> None:
  expected = artifact()
  path = tmp_path / f"{mutation}.route-evidence"
  path.write_bytes(expected.canonical_bytes)

  with RouteEvidenceStreamReader(path) as stream:
    with path.open("r+b") as output:
      if mutation == "tamper":
        output.seek(stream.summary.physical_offset)
        output.write(b"\xff")
      else:
        output.truncate(stream.summary.physical_offset + 1)
      output.flush()
      os.fsync(output.fileno())
    with pytest.raises(RouteEvidenceError, match="changed during streaming|truncated"):
      tuple(stream.iter_physical_frames())


def test_stream_reader_peak_memory_does_not_scale_with_record_count(tmp_path: Path) -> None:
  def evidence_path(name: str, count: int) -> Path:
    frames = tuple(frame(1_000 + index * 10) for index in range(count))
    physical = b"".join(_encode_frame(value) for value in frames)
    controls = tuple(
      replace(
        witness(0),
        ordinal=index,
        mono_time_ns=1_000 + index * 10,
        physical_record_index=index,
      )
      for index in range(count)
    )
    identity = replace(
      source(),
      physical_record_count=count,
      controls_witness_count=count,
    )
    value = RouteEvidenceArtifact(
      identity, CP, physical, (model(0),), controls,
      TORQUE, DELAY, MANEUVER, EVENTS,
    )
    path = tmp_path / f"{name}.route-evidence"
    path.write_bytes(value.canonical_bytes)
    return path

  small = evidence_path("small", 20)
  large = evidence_path("large", 5_000)
  gc.collect()

  def peak(path: Path) -> int:
    tracemalloc.start()
    try:
      with RouteEvidenceStreamReader(path) as stream:
        assert sum(1 for _ in stream.iter_physical_frames()) > 0
        assert sum(1 for _ in stream.iter_control_witnesses()) > 0
      return tracemalloc.get_traced_memory()[1]
    finally:
      tracemalloc.stop()

  small_peak = peak(small)
  large_peak = peak(large)
  assert large_peak <= small_peak + 256 * 1024


@pytest.mark.parametrize("pre_poll_count", (1, 3))
def test_streamed_inspection_accounts_for_pre_poll_witnesses(
  tmp_path: Path,
  pre_poll_count: int,
) -> None:
  timestamps = tuple(100 + index for index in range(pre_poll_count))
  identity = replace(
    source(),
    controls_witness_count=2 + pre_poll_count,
    unresolved_witness_count=pre_poll_count,
    pre_poll_dropped_timestamps_ns=timestamps,
  )
  expected = RouteEvidenceArtifact(
    identity,
    CP,
    PHYSICAL,
    (model(0), model(1)),
    (witness(0), witness(1)),
    TORQUE,
    DELAY,
    MANEUVER,
    EVENTS,
  )
  path = tmp_path / f"pre-poll-{pre_poll_count}.route-evidence"
  path.write_bytes(expected.canonical_bytes)
  assert inspect_route_evidence_file(path).sha256 == expected.sha256


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
