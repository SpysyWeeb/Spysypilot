from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest  # noqa: TID251

from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from openpilot.selfdrive.controls.lib.blatv2 import learning_backfill
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  BACKFILL_LEDGER_SCHEMA_VERSION,
  CANONICAL_JOIN_SCHEMA_VERSION,
  BackfillError,
  BuildDescriptor,
  BuildDescriptorRegistry,
  HistoricalLearningBackfill,
  PreparedRoute,
  ReplayResult,
  RouteCandidate,
  RouteRejected,
  discover_complete_route_candidates,
  extend_ledger,
  load_ledger,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_operation_status import (
  LEARNING_OPERATION_STATUS_PARAM,
  LearningOperationStatusPublisher,
  route_identity_sha256,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_progress import (
  BACKFILL_PROGRESS_PARAM,
  BackfillProgressPublisher,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  MeasuredLearningFrame,
  build_persistent_learning_runtime,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
)
from openpilot.selfdrive.controls.tests.blatv2_artifact_test_helpers import (
  route_evidence_for_frames,
)


CAUSAL_ROUTE_FRAME_COUNT = 15


class FakeParams:
  def __init__(self) -> None:
    self.values: dict[str, object] = {}
    self.puts: list[tuple[str, dict[str, object], bool]] = []

  def put(
    self,
    key: str,
    value: dict[str, object],
    *,
    block: bool,
  ) -> None:
    self.values[key] = dict(value)
    self.puts.append((key, dict(value), block))

  def remove(self, key: str) -> None:
    self.values.pop(key, None)


class FaultInjectedProgressParams(FakeParams):
  def __init__(self, failure_mode: str) -> None:
    super().__init__()
    self.failure_mode = failure_mode
    self.remove_count = 0

  def put(
    self,
    key: str,
    value: dict[str, object],
    *,
    block: bool,
  ) -> None:
    if self.failure_mode == "put":
      raise OSError("injected display-only progress write failure")
    super().put(key, value, block=block)

  def remove(self, key: str) -> None:
    self.remove_count += 1
    if self.failure_mode == "final_clear" and self.remove_count >= 2:
      raise OSError("injected display-only progress clear failure")
    super().remove(key)


def descriptor() -> BuildDescriptor:
  return BuildDescriptor(
    superproject_commit="1" * 40,
    opendbc_commit="2" * 40,
    panda_commit="3" * 40,
    log_schema_blob="4" * 40,
    supported_vehicle_identity=str(CAR.HYUNDAI_PALISADE),
    steer_max=409,
    steer_delta_up=4,
    steer_delta_down=7,
    steer_step=1,
    driver_allowance=50,
    driver_multiplier=2,
    driver_factor=1,
    production_envelope_verified=True,
    rack_rate_resolution_deg_s=4.0,
  )


def dynamics() -> ProvisionalRackDynamics:
  return ProvisionalRackDynamics(
    rack_gain_deg_s2_per_torque=4000.0,
    rack_damping_per_s=10.0,
    rack_rate_resolution_deg_s=4.0,
    provenance="engine test",
  )


def route_frame(
  cp,
  mono_ns: int,
  *,
  first_in_route: bool = False,
) -> MeasuredLearningFrame:
  return MeasuredLearningFrame(
    sample_mono_ns=mono_ns,
    response_mono_ns=mono_ns - 1_000_000,
    applied_report_mono_ns=mono_ns - 500_000,
    applied_effective_mono_ns=(
      0 if first_in_route else mono_ns - 10_500_000
    ),
    speed_mps=10.0,
    steering_angle_deg=5.0 * mono_ns * 1e-9,
    steering_rate_deg_s=5.0,
    steering_torque=0.0,
    steering_pressed=False,
    standstill=False,
    steer_fault_temporary=False,
    steer_fault_permanent=False,
    can_valid=True,
    can_timeout=False,
    applied_torque=0.0,
    lateral_active=True,
    live_parameters_valid=True,
    angle_offset_valid=True,
    steer_ratio_valid=True,
    stiffness_factor_valid=True,
    angle_offset_deg=0.0,
    steer_ratio=float(cp.steerRatio),
    stiffness_factor=float(cp.tireStiffnessFactor),
    roll_rad=0.0,
    inputs_valid=True,
  )


def add_route(
  log_root: Path,
  counter: int,
  *,
  locked: bool = False,
  segment_count: int = 1,
) -> tuple[str, Path]:
  route_name = f"{counter:08x}--{counter:010x}"
  first_rlog: Path | None = None
  for segment_index in range(segment_count):
    directory = log_root / f"{route_name}--{segment_index}"
    directory.mkdir(parents=True)
    rlog = directory / "rlog"
    rlog.write_bytes(
      f"route-{counter}-segment-{segment_index}".encode("ascii"),
    )
    if segment_index == 0:
      first_rlog = rlog
    if locked:
      (directory / "rlog.lock").touch()
  assert first_rlog is not None
  return route_name, first_rlog


def make_engine(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  *,
  pending_route_identity: str | None = None,
  with_progress: bool = False,
  replay_worker_count: int = 1,
  route_discovery=None,
  prepared_route_source=None,
) -> tuple[
  HistoricalLearningBackfill,
  FakeParams,
  Counter[str],
  object,
]:
  cp = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
  cp.carVin = "KM8R74HE0LU000001"
  storage_root = tmp_path / "learning"

  def runtime_factory():
    return build_persistent_learning_runtime(
      car_params=cp,
      storage_root=storage_root,
      provisional_rack_dynamics=dynamics(),
    )

  prepare_calls: Counter[str] = Counter()

  def fake_prepare(route, **kwargs):
    prepare_calls[route.route_name] += 1
    for position, segment in enumerate(route.segments, start=1):
      if kwargs.get("segment_started") is not None:
        kwargs["segment_started"](
          segment,
          position,
          len(route.segments),
        )
      if kwargs.get("segment_completed") is not None:
        kwargs["segment_completed"](
          segment,
          position,
          len(route.segments),
        )
    base = (route.route_counter + 1) * 1_000_000_000
    frames = tuple(
        route_frame(
          cp,
          base + index * 10_000_000,
          first_in_route=index == 0,
        )
        for index in range(CAUSAL_ROUTE_FRAME_COUNT)
      )
    provenance = {
        "canonical_join_schema_version": CANONICAL_JOIN_SCHEMA_VERSION,
        "car_params_sha256": hashlib.sha256(b"car-params").hexdigest(),
        "dongle_id_sha256": hashlib.sha256(b"dongle").hexdigest(),
        "extractor_schema_version": (
          learning_backfill.NATIVE_EXTRACTOR_SCHEMA_VERSION
        ),
        "log_schema_blob": "4" * 40,
        "opendbc_commit": "2" * 40,
        "panda_commit": "3" * 40,
        "physical_compatibility_sha256": hashlib.sha256(
          b"physical",
        ).hexdigest(),
        "route_version": "test-version",
        "selected_event_stream_sha256": hashlib.sha256(
          route.route_name.encode("ascii"),
        ).hexdigest(),
        "superproject_commit": "1" * 40,
      }
    return PreparedRoute(
      frames=frames,
      controls_witness_count=CAUSAL_ROUTE_FRAME_COUNT,
      unresolved_witness_count=0,
      gap_count=0,
      provenance=provenance,
      route_evidence=route_evidence_for_frames(
        route.route_name, frames, provenance,
      ),
    )

  monkeypatch.setattr(learning_backfill, "prepare_route", fake_prepare)
  extractor = tmp_path / "extractor"
  extractor.write_bytes(b"reviewed native extractor")
  extractor.chmod(0o755)
  params = FakeParams()
  progress_clock = iter(range(1_000_000_000, 10_000_000_000, 100_000_000))
  engine = HistoricalLearningBackfill(
    log_root=tmp_path / "logs",
    extractor_path=extractor,
    current_car_params=cp,
    runtime_factory=runtime_factory,
    route_bundle_factory=lambda *_args: (_ for _ in ()).throw(
      AssertionError("fake prepared route must bypass bundle construction"),
    ),
    car_params_decoder=lambda _encoded: cp,
    descriptor_registry=BuildDescriptorRegistry((descriptor(),)),
    expected_dongle_id="dongle",
    operation_status=LearningOperationStatusPublisher(params),
    backfill_progress=(
      BackfillProgressPublisher(params) if with_progress else None
    ),
    progress_monotonic_ns=lambda: next(progress_clock),
    abort_requested=lambda: False,
    pending_route_identity=pending_route_identity,
    replay_worker_count=replay_worker_count,
    route_discovery=route_discovery,
    prepared_route_source=prepared_route_source,
  )
  return engine, params, prepare_calls, runtime_factory


def test_extractor_identity_is_pinned_across_entire_transaction(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  add_route(tmp_path / "logs", 0x10)
  add_route(tmp_path / "logs", 0x20)
  engine, _, _, runtime_factory = make_engine(tmp_path, monkeypatch)
  fake_prepare = learning_backfill.prepare_route
  prepared_count = 0

  def checked_prepare(route, **kwargs):
    nonlocal prepared_count
    expected = kwargs["expected_extractor_sha256"]
    extractor = learning_backfill.open_verified_extractor(
      engine.extractor_path,
      expected_sha256=expected,
    )
    try:
      learning_backfill.verify_open_extractor(extractor)
    finally:
      os.close(extractor.descriptor)
    prepared = fake_prepare(route, **kwargs)
    prepared_count += 1
    if prepared_count == 1:
      replacement = engine.extractor_path.with_suffix(".replacement")
      replacement.write_bytes(b"different native extractor bytes")
      replacement.chmod(0o755)
      os.replace(replacement, engine.extractor_path)
    return prepared

  monkeypatch.setattr(learning_backfill, "prepare_route", checked_prepare)

  with pytest.raises(BackfillError) as raised:
    engine.run_once()

  assert raised.value.diagnostic == "backfill_reader_unavailable"
  assert prepared_count == 1
  assert not runtime_factory().artifact_paths.backfill_pointer.exists()


@pytest.mark.parametrize("replay_worker_count", (1, 2, 4))
def test_replay_worker_count_accepts_only_supported_integers(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  replay_worker_count: int,
) -> None:
  engine, _params, _calls, _runtime_factory = make_engine(
    tmp_path,
    monkeypatch,
    replay_worker_count=replay_worker_count,
  )

  assert engine.replay_worker_count == replay_worker_count
  assert type(engine.replay_worker_count) is int


@pytest.mark.parametrize(
  "replay_worker_count",
  (False, True, 0, 3, 5, 1.0, "2", None),
)
def test_replay_worker_count_rejects_unsupported_values(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  replay_worker_count: object,
) -> None:
  with pytest.raises(
    ValueError,
    match="backfill replay worker count is outside its bound",
  ):
    make_engine(
      tmp_path,
      monkeypatch,
      replay_worker_count=replay_worker_count,
    )


def published_artifact_snapshot(runtime_factory) -> dict[str, object]:
  runtime = runtime_factory()
  unresolved_paths = runtime.artifact_paths
  paths = unresolved_paths.resolved()
  candidates = (
    tuple(
      (candidate.name, candidate.read_bytes())
      for candidate in sorted(paths.candidates.iterdir())
      if candidate.is_file()
    )
    if paths.candidates.is_dir()
    else ()
  )
  return {
    "pointer": unresolved_paths.backfill_pointer.read_bytes(),
    "evidence": paths.evidence.read_bytes(),
    "manifest": paths.manifest.read_bytes(),
    "candidates": candidates,
    "ledger": paths.backfill_ledger.read_bytes(),
    "provenance": paths.backfill_provenance.read_bytes(),
    "commit": paths.backfill_commit.read_bytes(),
  }


def read_atomic_pipe_records(read_descriptor: int) -> list[str]:
  encoded = bytearray()
  while True:
    chunk = os.read(read_descriptor, 4096)
    if not chunk:
      break
    encoded.extend(chunk)
  os.close(read_descriptor)
  return encoded.decode("ascii").splitlines()


def test_optional_sources_match_default_and_preserve_a_a_authority_order(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  baseline_root = tmp_path / "baseline"
  add_route(baseline_root / "logs", 0x10)
  add_route(baseline_root / "logs", 0x20)
  (
    baseline_engine,
    _,
    baseline_prepare_calls,
    baseline_runtime_factory,
  ) = make_engine(baseline_root, monkeypatch)

  baseline = baseline_engine.run_once()
  assert baseline.publication is not None
  baseline_artifacts = published_artifact_snapshot(
    baseline_runtime_factory,
  )

  injected_root = tmp_path / "injected"
  add_route(injected_root / "logs", 0x10)
  add_route(injected_root / "logs", 0x20)
  candidates = discover_complete_route_candidates(
    injected_root / "logs",
  )
  discovery_abort_callbacks = []
  source_calls = []

  def injected_discovery(abort_requested):
    discovery_abort_callbacks.append(abort_requested)
    return learning_backfill.FullRlogDiscovery(candidates, False)

  def injected_source(authority_index, route, abort_requested):
    source_calls.append((
      authority_index,
      route.route_name,
      route.display_identity,
      abort_requested,
    ))
    assert not abort_requested()
    return learning_backfill.prepare_route(route)

  (
    injected_engine,
    _,
    injected_prepare_calls,
    injected_runtime_factory,
  ) = make_engine(
    injected_root,
    monkeypatch,
    route_discovery=injected_discovery,
    prepared_route_source=injected_source,
  )

  injected = injected_engine.run_once()

  assert injected.publication is not None
  assert discovery_abort_callbacks == [injected_engine.abort_requested]
  expected_route_identity = [
    (route.route_name, route.display_identity)
    for route in candidates
  ]
  assert [
    (route_name, route_identity)
    for _, route_name, route_identity, _ in source_calls
  ] == 2 * expected_route_identity
  assert [authority for authority, *_ in source_calls] == [1, 1, 2, 2]
  assert all(
    callback is injected_engine.abort_requested
    for *_, callback in source_calls
  )
  assert baseline_prepare_calls == injected_prepare_calls
  assert published_artifact_snapshot(injected_runtime_factory) == (
    baseline_artifacts
  )
  assert injected.publication.finalization == (
    baseline.publication.finalization
  )


def test_injected_discovery_wrong_type_fails_closed_without_publication(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  add_route(tmp_path / "logs", 0x10)
  engine, _, _, runtime_factory = make_engine(
    tmp_path,
    monkeypatch,
    route_discovery=lambda _abort_requested: (),
  )

  with pytest.raises(BackfillError) as raised:
    engine.run_once()

  assert raised.value.diagnostic == "backfill_route_incompatible"
  assert not runtime_factory().artifact_paths.backfill_pointer.exists()


@pytest.mark.parametrize("wrong_kind", ("object", "subclass"))
def test_injected_preparation_wrong_type_fails_closed_without_publication(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  wrong_kind: str,
) -> None:
  add_route(tmp_path / "logs", 0x10)

  class DerivedPreparedRoute(PreparedRoute):
    pass

  def invalid_source(_authority, route, _abort_requested):
    if wrong_kind == "object":
      return object()
    prepared = learning_backfill.prepare_route(route)
    return DerivedPreparedRoute(
      frames=prepared.frames,
      controls_witness_count=prepared.controls_witness_count,
      unresolved_witness_count=prepared.unresolved_witness_count,
      gap_count=prepared.gap_count,
      provenance=prepared.provenance,
    )

  engine, _, _, runtime_factory = make_engine(
    tmp_path,
    monkeypatch,
    prepared_route_source=invalid_source,
  )

  with pytest.raises(BackfillError) as raised:
    engine.run_once()

  assert raised.value.diagnostic == "backfill_route_incompatible"
  assert not runtime_factory().artifact_paths.backfill_pointer.exists()


def test_injected_prepared_route_spool_exact_type_is_accepted(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  route_name, _ = add_route(tmp_path / "logs", 0x10)
  source_calls = []

  def injected_source(authority_index, route, abort_requested):
    source_calls.append((authority_index, route.route_name))
    prepared = learning_backfill.prepare_route(route)
    spool_root = tmp_path / f"remote-spool-{authority_index}"
    spool_root.mkdir(mode=0o700, exist_ok=True)
    descriptor = learning_backfill.write_prepared_route_spool(
      spool_root,
      route.route_name,
      prepared.frames,
      controls_witness_count=prepared.controls_witness_count,
      unresolved_witness_count=prepared.unresolved_witness_count,
      gap_count=prepared.gap_count,
      provenance=prepared.provenance,
        max_frames=learning_backfill.MAXIMUM_ROUTE_FRAMES,
        abort_requested=abort_requested,
        route_evidence=prepared.route_evidence,
      )
    return learning_backfill.open_prepared_route_spool(
      spool_root,
      descriptor,
      expected_route_name=route.route_name,
      max_frames=learning_backfill.MAXIMUM_ROUTE_FRAMES,
    )

  engine, _, _, _ = make_engine(
    tmp_path,
    monkeypatch,
    prepared_route_source=injected_source,
  )

  result = engine.run_once()

  assert result.publication is not None
  assert source_calls == [(1, route_name), (2, route_name)]
  assert not tuple(tmp_path.glob("remote-spool-*/*.blatspool"))


@pytest.mark.parametrize("abort_position", ("before", "after"))
def test_injected_preparation_abort_checks_bracket_callback_and_never_publish(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  abort_position: str,
) -> None:
  add_route(tmp_path / "logs", 0x10)
  abort_state = {"requested": abort_position == "before"}
  source_calls = 0

  def abort_requested() -> bool:
    return abort_state["requested"]

  def injected_source(authority_index, route, callback_abort_requested):
    nonlocal source_calls
    source_calls += 1
    assert authority_index == 1
    assert callback_abort_requested is abort_requested
    prepared = learning_backfill.prepare_route(route)
    abort_state["requested"] = abort_position == "after"
    return prepared

  engine, _, _, runtime_factory = make_engine(
    tmp_path,
    monkeypatch,
    prepared_route_source=injected_source,
  )
  route = discover_complete_route_candidates(tmp_path / "logs")[0]

  with pytest.raises(BackfillError) as raised:
    engine._prepare(
      runtime_factory(),
      route,
      authority_index=1,
      abort_requested=abort_requested,
    )

  assert raised.value.diagnostic == "unexpected_error"
  assert source_calls == (0 if abort_position == "before" else 1)
  assert not runtime_factory().artifact_paths.backfill_pointer.exists()


@pytest.mark.parametrize("abort_position", ("before", "after"))
def test_injected_discovery_abort_checks_bracket_callback_and_never_publish(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  abort_position: str,
) -> None:
  add_route(tmp_path / "logs", 0x10)
  abort_state = {"requested": False}
  discovery_calls = 0

  def abort_requested() -> bool:
    return abort_state["requested"]

  engine, _, _, runtime_factory = make_engine(tmp_path, monkeypatch)
  original_runtime_factory = engine.runtime_factory
  runtime_factory_calls = 0

  def cancellation_aware_runtime_factory():
    nonlocal runtime_factory_calls
    runtime_factory_calls += 1
    runtime = original_runtime_factory()
    if runtime_factory_calls == 2 and abort_position == "before":
      abort_state["requested"] = True
    return runtime

  def injected_discovery(callback_abort_requested):
    nonlocal discovery_calls
    discovery_calls += 1
    assert callback_abort_requested is abort_requested
    candidates = discover_complete_route_candidates(tmp_path / "logs")
    abort_state["requested"] = abort_position == "after"
    return learning_backfill.FullRlogDiscovery(candidates, False)

  engine.runtime_factory = cancellation_aware_runtime_factory
  engine.abort_requested = abort_requested
  engine.route_discovery = injected_discovery

  with pytest.raises(BackfillError) as raised:
    engine.run_once()

  assert raised.value.diagnostic == "unexpected_error"
  assert discovery_calls == (0 if abort_position == "before" else 1)
  assert not runtime_factory().artifact_paths.backfill_pointer.exists()


def test_injected_route_rejection_keeps_stable_ledger_semantics(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  rejected_name, _ = add_route(tmp_path / "logs", 0x10)
  accepted_name, _ = add_route(tmp_path / "logs", 0x20)
  source_calls = []

  def injected_source(authority_index, route, abort_requested):
    source_calls.append((authority_index, route.route_name))
    assert not abort_requested()
    if route.route_name == rejected_name:
      raise RouteRejected(
        "remote_route_rejected",
        "deterministic injected preparation rejection",
      )
    return learning_backfill.prepare_route(route)

  engine, _, _, runtime_factory = make_engine(
    tmp_path,
    monkeypatch,
    prepared_route_source=injected_source,
  )

  result = engine.run_once()

  assert result.publication is not None
  assert result.publication.diagnostic == (
    "backfill_complete_with_rejections"
  )
  assert source_calls == [
    (1, rejected_name),
    (1, accepted_name),
    (2, rejected_name),
    (2, accepted_name),
  ]
  runtime = runtime_factory()
  ledger = load_ledger(
    runtime.artifact_paths,
    runtime_identity_sha256=(
      runtime.runtime_bundle.calibration_identity_sha256
    ),
  )
  assert [entry["route_name"] for entry in ledger["entries"]] == [
    rejected_name,
    accepted_name,
  ]
  assert [entry["disposition"] for entry in ledger["entries"]] == [
    "rejected",
    "ingested",
  ]
  assert ledger["entries"][0]["diagnostic"] == "remote_route_rejected"
  assert ledger["entries"][0]["provenance"] is None
  assert ledger["entries"][0]["accepted_sample_count"] == 0
  assert ledger["entries"][0]["rejected_sample_count"] == 0


def test_bootstrap_then_watermark_late_skip_and_hash_exactly_once(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log_root = tmp_path / "logs"
  first_name, first_rlog = add_route(log_root, 0x10)
  second_name, _ = add_route(log_root, 0x20)
  engine, params, prepare_calls, runtime_factory = make_engine(
    tmp_path,
    monkeypatch,
  )

  first_run = engine.run_once()

  assert first_run.publication is not None
  assert not first_run.pending_logger_close
  assert prepare_calls == Counter({first_name: 2, second_name: 2})
  runtime = runtime_factory()
  ledger = load_ledger(
    runtime.artifact_paths,
    runtime_identity_sha256=runtime.runtime_bundle.calibration_identity_sha256,
  )
  assert [entry["route_name"] for entry in ledger["entries"]] == [
    first_name,
    second_name,
  ]
  assert [entry["disposition"] for entry in ledger["entries"]] == [
    "ingested",
    "ingested",
  ]
  assert ledger["watermark_route_counter"] == 0x20
  operation_updates = [
    value
    for key, value, _ in params.puts
    if key == LEARNING_OPERATION_STATUS_PARAM
  ]
  replay_counts = [
    value["accepted_sample_count"]
    for value in operation_updates
    if value["diagnostic"] == "replaying_route"
  ]
  assert replay_counts == [0, 1, 1, 2]

  late_name, _ = add_route(log_root, 0x05)
  new_name, _ = add_route(log_root, 0x30)
  second_run = engine.run_once()
  assert second_run.publication is not None
  assert prepare_calls == Counter({
    first_name: 2,
    second_name: 2,
    new_name: 2,
  })
  runtime = runtime_factory()
  extended = load_ledger(
    runtime.artifact_paths,
    runtime_identity_sha256=runtime.runtime_bundle.calibration_identity_sha256,
  )
  assert [entry["route_name"] for entry in extended["entries"]] == [
    first_name,
    second_name,
    late_name,
    new_name,
  ]
  assert [entry["disposition"] for entry in extended["entries"]] == [
    "ingested",
    "ingested",
    "late_older_skipped",
    "ingested",
  ]
  assert extended["watermark_route_counter"] == 0x30

  no_op = engine.run_once()
  assert no_op.publication is None
  assert not no_op.pending_logger_close
  assert prepare_calls == Counter({
    first_name: 2,
    second_name: 2,
    new_name: 2,
  })

  first_rlog.write_bytes(b"mutated-known-route")
  with pytest.raises(BackfillError) as changed:
    engine.run_once()
  assert changed.value.diagnostic == "backfill_untracked_evidence"


def test_progress_reports_both_passes_without_double_counting(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log_root = tmp_path / "logs"
  add_route(log_root, 0x10, segment_count=2)
  add_route(log_root, 0x20)
  engine, params, prepare_calls, _ = make_engine(
    tmp_path,
    monkeypatch,
    with_progress=True,
  )

  result = engine.run_once()

  assert result.publication is not None
  assert result.publication.accepted_sample_count == 2
  assert prepare_calls == Counter({
    "00000010--0000000010": 2,
    "00000020--0000000020": 2,
  })
  updates = [
    value
    for key, value, _ in params.puts
    if key == BACKFILL_PROGRESS_PARAM
  ]
  assert [
    (
      value["pass_index"],
      value["current_route_index"],
      value["current_segment_index"],
      value["current_route_segment_count"],
    )
    for value in updates
    if value["phase"] in {"reading_segment", "applying_route"}
  ] == [
    (1, 1, 1, 2),
    (1, 1, 2, 2),
    (1, 1, 2, 2),
    (1, 2, 1, 1),
    (1, 2, 1, 1),
    (2, 1, 1, 2),
    (2, 1, 2, 2),
    (2, 1, 2, 2),
    (2, 2, 1, 1),
    (2, 2, 1, 1),
  ]
  assert [value["phase"] for value in updates[-2:]] == [
    "comparing",
    "publishing",
  ]
  assert updates[-1]["completed_replay_segment_count"] == 6
  assert updates[-1]["completed_work_units"] == updates[-1][
    "total_work_units"
  ]
  replay_updates = [
    value
    for value in updates
    if value["phase"] in {"reading_segment", "applying_route"}
  ]
  assert any(
    value["approximate_remaining_seconds"] is not None
    for value in replay_updates
  )
  assert all(
    left["completed_work_units"] <= right["completed_work_units"]
    for left, right in zip(updates, updates[1:], strict=False)
  )
  assert BACKFILL_PROGRESS_PARAM not in params.values
  operation_updates = [
    value
    for key, value, _ in params.puts
    if key == LEARNING_OPERATION_STATUS_PARAM
  ]
  assert operation_updates[-1]["accepted_sample_count"] == 2


def test_rejected_route_resolves_progress_without_eta_samples(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log_root = tmp_path / "logs"
  rejected_name, _ = add_route(log_root, 0x10, segment_count=2)
  add_route(log_root, 0x20)
  engine, params, _, _ = make_engine(
    tmp_path,
    monkeypatch,
    with_progress=True,
  )
  original_prepare = engine._prepare

  def reject_first_route(runtime, route, **kwargs):
    if route.route_name == rejected_name:
      kwargs["segment_started"](
        route.segments[0],
        1,
        len(route.segments),
      )
      raise RouteRejected(
        "invalid_route_version",
        "representative rejection after one segment began",
      )
    return original_prepare(runtime, route, **kwargs)

  engine._prepare = reject_first_route
  result = engine.run_once()

  assert result.publication is not None
  assert result.publication.diagnostic == (
    "backfill_complete_with_rejections"
  )
  updates = [
    value
    for key, value, _ in params.puts
    if key == BACKFILL_PROGRESS_PARAM
  ]
  good_route_starts = [
    value
    for value in updates
    if (
      value["phase"] == "reading_segment"
      and value["current_route_index"] == 2
      and value["current_segment_index"] == 1
    )
  ]
  assert [value["completed_replay_segment_count"] for value in good_route_starts] == [
    2,
    5,
  ]
  assert all(
    value["approximate_remaining_seconds"] is None
    for value in updates
    if value["phase"] in {"reading_segment", "applying_route"}
  )
  assert updates[-1]["completed_replay_segment_count"] == updates[-1][
    "total_replay_segment_count"
  ]
  assert updates[-1]["completed_work_units"] == updates[-1][
    "total_work_units"
  ]


def test_progress_projection_does_not_change_replay_artifacts(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  baseline_root = tmp_path / "baseline"
  add_route(baseline_root / "logs", 0x10, segment_count=2)
  baseline_engine, _, _, baseline_runtime_factory = make_engine(
    baseline_root,
    monkeypatch,
  )
  baseline = baseline_engine.run_once()
  assert baseline.publication is not None
  baseline_runtime = baseline_runtime_factory()
  baseline_ledger = baseline_runtime.artifact_paths.backfill_ledger.read_bytes()

  observed_root = tmp_path / "observed"
  add_route(observed_root / "logs", 0x10, segment_count=2)
  observed_engine, _, _, observed_runtime_factory = make_engine(
    observed_root,
    monkeypatch,
    with_progress=True,
  )
  observed = observed_engine.run_once()
  assert observed.publication is not None
  observed_runtime = observed_runtime_factory()
  observed_ledger = observed_runtime.artifact_paths.backfill_ledger.read_bytes()

  assert observed.publication.finalization == (
    baseline.publication.finalization
  )
  assert observed_ledger == baseline_ledger
  assert observed.publication.generation_sha256 == (
    baseline.publication.generation_sha256
  )
  assert observed.publication.ledger_sha256 == (
    baseline.publication.ledger_sha256
  )


@pytest.mark.parametrize("parallel_worker_count", (2, 4))
def test_parallel_workers_publish_exact_single_worker_artifacts(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  parallel_worker_count: int,
) -> None:
  serial_root = tmp_path / "serial"
  add_route(serial_root / "logs", 0x10, segment_count=2)
  add_route(serial_root / "logs", 0x20)
  add_route(serial_root / "logs", 0x30, segment_count=3)
  serial_engine, _, _, serial_runtime_factory = make_engine(
    serial_root,
    monkeypatch,
    replay_worker_count=1,
  )
  serial = serial_engine.run_once()
  assert serial.publication is not None
  serial_artifacts = published_artifact_snapshot(serial_runtime_factory)

  parallel_root = tmp_path / "parallel"
  add_route(parallel_root / "logs", 0x10, segment_count=2)
  add_route(parallel_root / "logs", 0x20)
  add_route(parallel_root / "logs", 0x30, segment_count=3)
  parallel_engine, _, _, parallel_runtime_factory = make_engine(
    parallel_root,
    monkeypatch,
    replay_worker_count=parallel_worker_count,
  )
  parallel = parallel_engine.run_once()
  assert parallel.publication is not None
  parallel_artifacts = published_artifact_snapshot(
    parallel_runtime_factory,
  )

  assert parallel_artifacts == serial_artifacts
  assert parallel.publication.finalization.evidence_bytes == (
    serial.publication.finalization.evidence_bytes
  )
  assert parallel.publication.finalization.manifest_bytes == (
    serial.publication.finalization.manifest_bytes
  )
  assert parallel.publication.finalization.candidate_profile_json == (
    serial.publication.finalization.candidate_profile_json
  )
  assert (
    parallel.publication.accepted_sample_count,
    parallel.publication.rejected_sample_count,
  ) == (
    serial.publication.accepted_sample_count,
    serial.publication.rejected_sample_count,
  )
  assert parallel.publication.generation_sha256 == (
    serial.publication.generation_sha256
  )
  assert parallel.publication.ledger_sha256 == (
    serial.publication.ledger_sha256
  )


def test_four_worker_two_route_topology_has_four_independent_prepare_lanes(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  first_name, _ = add_route(tmp_path / "logs", 0x10)
  second_name, _ = add_route(tmp_path / "logs", 0x20)
  engine, _params, _calls, _runtime_factory = make_engine(
    tmp_path,
    monkeypatch,
    replay_worker_count=4,
  )
  deterministic_prepare = learning_backfill.prepare_route
  read_descriptor, write_descriptor = os.pipe()

  def record_prepare(route, **kwargs):
    process = learning_backfill.multiprocessing.current_process()
    record = "|".join((
      process.name,
      str(os.getpid()),
      str(os.getppid()),
      route.route_name,
    )).encode("ascii") + b"\n"
    assert len(record) < 4096
    assert os.write(write_descriptor, record) == len(record)
    return deterministic_prepare(route, **kwargs)

  monkeypatch.setattr(learning_backfill, "prepare_route", record_prepare)
  try:
    result = engine.run_once()
  finally:
    os.close(write_descriptor)
  assert result.publication is not None

  records = [
    (name, int(pid), int(ppid), route_name)
    for name, pid, ppid, route_name in (
      record.split("|")
      for record in read_atomic_pipe_records(read_descriptor)
    )
  ]
  parent_name = learning_backfill.multiprocessing.current_process().name
  assert len(records) == 4
  assert {record[0] for record in records} == {
    parent_name,
    "blatv2-replay-2",
    "blatv2-prepare-1",
    "blatv2-prepare-2",
  }
  assert len({record[1] for record in records}) == 4
  assert Counter(record[3] for record in records) == Counter({
    first_name: 2,
    second_name: 2,
  })
  assert {
    record[0]: record[3]
    for record in records
  } == {
    parent_name: first_name,
    "blatv2-replay-2": first_name,
    "blatv2-prepare-1": second_name,
    "blatv2-prepare-2": second_name,
  }
  by_name = {record[0]: record for record in records}
  assert by_name[parent_name][1] == os.getpid()
  assert by_name["blatv2-replay-2"][2] == os.getpid()
  assert by_name["blatv2-prepare-1"][2] == os.getpid()
  assert by_name["blatv2-prepare-2"][2] == (
    by_name["blatv2-replay-2"][1]
  )


def test_four_worker_single_route_starts_no_prepare_helpers(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  route_name, _ = add_route(tmp_path / "logs", 0x10)
  engine, _params, _calls, runtime_factory = make_engine(
    tmp_path,
    monkeypatch,
    replay_worker_count=4,
  )
  deterministic_prepare = learning_backfill.prepare_route
  read_descriptor, write_descriptor = os.pipe()

  def record_prepare(route, **kwargs):
    process = learning_backfill.multiprocessing.current_process()
    record = (
      f"{process.name}|{os.getpid()}|{route.route_name}\n"
    ).encode("ascii")
    assert os.write(write_descriptor, record) == len(record)
    return deterministic_prepare(route, **kwargs)

  monkeypatch.setattr(learning_backfill, "prepare_route", record_prepare)
  try:
    result = engine.run_once()
  finally:
    os.close(write_descriptor)
  assert result.publication is not None

  records = [record.split("|") for record in read_atomic_pipe_records(
    read_descriptor,
  )]
  assert len(records) == 2
  assert {record[0] for record in records} == {
    learning_backfill.multiprocessing.current_process().name,
    "blatv2-replay-2",
  }
  assert {record[2] for record in records} == {route_name}
  artifact_root = runtime_factory().artifact_paths.root
  assert not tuple(artifact_root.glob(
    f"{learning_backfill.BACKFILL_SPOOL_DIRECTORY_PREFIX}*",
  ))


def test_four_worker_prefetched_rejection_matches_serial_and_keeps_later_route(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def run(worker_count: int, root: Path):
    add_route(root / "logs", 0x10)
    rejected_name, _ = add_route(root / "logs", 0x20)
    later_name, _ = add_route(root / "logs", 0x30)
    engine, _params, _calls, runtime_factory = make_engine(
      root,
      monkeypatch,
      replay_worker_count=worker_count,
    )
    deterministic_prepare = learning_backfill.prepare_route

    def reject_middle(route, **kwargs):
      if route.route_name == rejected_name:
        raise RouteRejected(
          "invalid_route_version",
          "deterministic prefetched-route rejection",
        )
      return deterministic_prepare(route, **kwargs)

    monkeypatch.setattr(learning_backfill, "prepare_route", reject_middle)
    result = engine.run_once()
    assert result.publication is not None
    ledger = load_ledger(
      runtime_factory().artifact_paths,
      runtime_identity_sha256=(
        runtime_factory().runtime_bundle.calibration_identity_sha256
      ),
    )
    assert [entry["route_name"] for entry in ledger["entries"]] == [
      "00000010--0000000010",
      rejected_name,
      later_name,
    ]
    assert [entry["disposition"] for entry in ledger["entries"]] == [
      "ingested",
      "rejected",
      "ingested",
    ]
    assert ledger["entries"][1]["diagnostic"] == "invalid_route_version"
    return result, published_artifact_snapshot(runtime_factory)

  serial_result, serial_artifacts = run(1, tmp_path / "serial")
  parallel_result, parallel_artifacts = run(4, tmp_path / "parallel")

  assert parallel_artifacts == serial_artifacts
  assert parallel_result.publication is not None
  assert serial_result.publication is not None
  assert parallel_result.publication.finalization == (
    serial_result.publication.finalization
  )
  assert (
    parallel_result.publication.accepted_sample_count,
    parallel_result.publication.rejected_sample_count,
  ) == (
    serial_result.publication.accepted_sample_count,
    serial_result.publication.rejected_sample_count,
  )


def test_four_worker_helper_provenance_difference_is_unpublished(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  add_route(tmp_path / "logs", 0x10)
  add_route(tmp_path / "logs", 0x20)
  engine, _params, _calls, runtime_factory = make_engine(
    tmp_path,
    monkeypatch,
    replay_worker_count=4,
  )
  deterministic_prepare = learning_backfill.prepare_route

  def helper_distinct_prepare(route, **kwargs):
    prepared = deterministic_prepare(route, **kwargs)
    process_name = learning_backfill.multiprocessing.current_process().name
    if process_name.startswith("blatv2-prepare-"):
      provenance = dict(prepared.provenance)
      provenance["selected_event_stream_sha256"] = hashlib.sha256(
        process_name.encode("ascii"),
      ).hexdigest()
      return replace(
        prepared,
        provenance=provenance,
        route_evidence=route_evidence_for_frames(
          route.route_name, prepared.frames, provenance,
        ),
      )
    return prepared

  monkeypatch.setattr(
    learning_backfill,
    "prepare_route",
    helper_distinct_prepare,
  )

  with pytest.raises(BackfillError) as raised:
    engine.run_once()

  assert raised.value.diagnostic == "backfill_nondeterministic"
  paths = runtime_factory().artifact_paths
  assert not paths.backfill_pointer.exists()
  assert not paths.evidence.exists()
  assert not paths.manifest.exists()
  assert not paths.backfill_ledger.exists()


def test_route_evidence_is_not_durable_until_both_authorities_match(
  tmp_path: Path,
) -> None:
  route_name = "00000010--0000000010"
  route = RouteCandidate(
    route_name=route_name,
    route_counter=0x10,
    segments=(learning_backfill.RouteSegment(
      index=0,
      path=tmp_path / "not-read",
      sha256="a" * 64,
      size_bytes=1,
    ),),
  )
  evidence = route_evidence_for_frames(
    route_name,
    (route_frame(
      CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE),
      1_000_000_000,
      first_in_route=True,
    ),),
    {"test": "first-authority-publication-boundary"},
  )
  result = ReplayResult(
    route=route,
    disposition="ingested",
    diagnostic="ingested",
    provenance={"test": "first-authority-publication-boundary"},
    accepted_sample_count=0,
    rejected_sample_count=1,
    controls_witness_count=1,
    unresolved_witness_count=0,
    route_evidence_sha256=evidence.sha256,
    route_evidence_source_key=evidence.source_key,
  )

  learning_backfill._stage_route_evidence(
    root=tmp_path,
    authority_index=1,
    artifact=evidence,
  )
  assert not (tmp_path / "route_evidence_v2").exists()

  learning_backfill._stage_route_evidence(
    root=tmp_path,
    authority_index=2,
    artifact=evidence,
  )
  replay_pass = SimpleNamespace(results=(result,))
  learning_backfill._publish_route_evidence_after_aa(
    root=tmp_path,
    first=replay_pass,
    second=replay_pass,
  )

  stored = learning_backfill.RouteEvidenceStore(
    tmp_path / "route_evidence_v2",
  ).load(evidence.sha256)
  assert stored.canonical_bytes == evidence.canonical_bytes
  assert not (tmp_path / ".route-evidence-staging-v2").exists()


def test_stale_route_evidence_staging_is_bounded_and_quarantined(
  tmp_path: Path,
) -> None:
  route_name = "00000010--0000000010"
  evidence = route_evidence_for_frames(
    route_name,
    (route_frame(
      CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE),
      1_000_000_000,
      first_in_route=True,
    ),),
    {"test": "crash-recovery-staging"},
  )
  learning_backfill._stage_route_evidence(
    root=tmp_path,
    authority_index=1,
    artifact=evidence,
  )
  staging = tmp_path / learning_backfill.ROUTE_EVIDENCE_STAGING_DIRECTORY
  staged_file = next((staging / "authority-1").iterdir())
  staged_bytes = staged_file.read_bytes()
  second = staging / "authority-2"
  second.mkdir(mode=0o700)
  partial = second / f".{evidence.sha256}.route-evidence.deadbeef.partial"
  partial.write_bytes(b"authenticated-owner-incomplete-copy")
  partial.chmod(0o600)

  with pytest.raises(BackfillError) as raised:
    learning_backfill.cleanup_stale_prepared_route_spools(tmp_path)

  assert raised.value.diagnostic == "backfill_spool_invalid"
  quarantine = (
    tmp_path / learning_backfill.ROUTE_EVIDENCE_STAGING_QUARANTINE
  )
  assert not staging.exists()
  assert (
    quarantine / "authority-1" / staged_file.name
  ).read_bytes() == staged_bytes
  assert (
    quarantine / "authority-2" / partial.name
  ).read_bytes() == b"authenticated-owner-incomplete-copy"
  with pytest.raises(BackfillError, match="operator recovery"):
    learning_backfill.cleanup_stale_prepared_route_spools(tmp_path)


def test_stale_route_evidence_staging_unknown_child_fails_without_move(
  tmp_path: Path,
) -> None:
  staging = tmp_path / learning_backfill.ROUTE_EVIDENCE_STAGING_DIRECTORY
  authority = staging / "authority-1"
  authority.mkdir(mode=0o700, parents=True)
  (authority / "foreign-file").write_bytes(b"do not delete")

  with pytest.raises(BackfillError, match="unknown child"):
    learning_backfill.cleanup_stale_prepared_route_spools(tmp_path)

  assert (authority / "foreign-file").read_bytes() == b"do not delete"
  assert not (
    tmp_path / learning_backfill.ROUTE_EVIDENCE_STAGING_QUARANTINE
  ).exists()


def test_stale_prepared_route_scratch_is_quarantined_selectively(
  tmp_path: Path,
) -> None:
  artifact_root = tmp_path / "artifact"
  artifact_root.mkdir()
  stale = artifact_root / (
    f"{learning_backfill.BACKFILL_SPOOL_DIRECTORY_PREFIX}1-deadbeef"
  )
  stale.mkdir()
  stale.chmod(0o700)
  unrelated = artifact_root / ".another-component-scratch"
  unrelated.mkdir()
  (unrelated / "keep").write_bytes(b"owned elsewhere")

  with pytest.raises(BackfillError, match="operator recovery"):
    learning_backfill.cleanup_stale_prepared_route_spools(artifact_root)

  assert not stale.exists()
  quarantine = (
    artifact_root / learning_backfill.PREPARED_ROUTE_SCRATCH_QUARANTINE
  )
  assert quarantine.is_dir()
  assert unrelated.is_dir()
  assert (unrelated / "keep").read_bytes() == b"owned elsewhere"
  with pytest.raises(BackfillError, match="operator recovery"):
    learning_backfill.cleanup_stale_prepared_route_spools(artifact_root)


def test_stale_prepared_route_unknown_child_is_never_moved_or_deleted(
  tmp_path: Path,
) -> None:
  artifact_root = tmp_path / "artifact"
  artifact_root.mkdir()
  stale = artifact_root / (
    f"{learning_backfill.BACKFILL_SPOOL_DIRECTORY_PREFIX}1-deadbeef"
  )
  foreign = stale / "foreign"
  foreign.mkdir(parents=True, mode=0o700)
  stale.chmod(0o700)
  sentinel = foreign / "must-survive"
  sentinel.write_bytes(b"not owned by the learner")

  with pytest.raises(BackfillError, match="unknown child"):
    learning_backfill.cleanup_stale_prepared_route_spools(artifact_root)

  assert sentinel.read_bytes() == b"not owned by the learner"
  assert stale.is_dir()
  assert not (
    artifact_root / learning_backfill.PREPARED_ROUTE_SCRATCH_QUARANTINE
  ).exists()


def test_stale_prepared_route_path_swap_preserves_replacement_sentinel(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  artifact_root = tmp_path / "artifact"
  artifact_root.mkdir()
  stale = artifact_root / (
    f"{learning_backfill.BACKFILL_SPOOL_DIRECTORY_PREFIX}1-deadbeef"
  )
  stale.mkdir(mode=0o700)
  displaced = artifact_root / "displaced-owned-scratch"
  real_inventory = learning_backfill._bounded_directory_names
  swapped = False

  def swap_after_inventory(target, maximum, **kwargs):
    nonlocal swapped
    names = real_inventory(target, maximum, **kwargs)
    if not swapped and kwargs["purpose"] == "prepared-route scratch":
      swapped = True
      stale.rename(displaced)
      stale.mkdir(mode=0o700)
      (stale / "must-survive").write_bytes(b"replacement")
    return names

  monkeypatch.setattr(
    learning_backfill,
    "_bounded_directory_names",
    swap_after_inventory,
  )
  with pytest.raises(BackfillError, match="path changed"):
    learning_backfill.cleanup_stale_prepared_route_spools(artifact_root)

  assert (stale / "must-survive").read_bytes() == b"replacement"
  assert displaced.is_dir()
  assert not (
    artifact_root / learning_backfill.PREPARED_ROUTE_SCRATCH_QUARANTINE
  ).exists()


@pytest.mark.parametrize("unsafe_kind", ("file", "symlink"))
def test_stale_prepared_route_scratch_unsafe_type_fails_closed(
  tmp_path: Path,
  unsafe_kind: str,
) -> None:
  artifact_root = tmp_path / "artifact"
  artifact_root.mkdir()
  unsafe = artifact_root / (
    f"{learning_backfill.BACKFILL_SPOOL_DIRECTORY_PREFIX}2-bad0cafe"
  )
  target = tmp_path / "outside"
  target.mkdir()
  if unsafe_kind == "file":
    unsafe.write_bytes(b"not a private directory")
  else:
    unsafe.symlink_to(target, target_is_directory=True)

  with pytest.raises(BackfillError) as raised:
    learning_backfill.cleanup_stale_prepared_route_spools(artifact_root)

  assert raised.value.diagnostic == "backfill_spool_invalid"
  assert os.path.lexists(unsafe)
  assert target.is_dir()


def make_empty_prefetch_lane(
  artifact_root: Path,
) -> learning_backfill._PrefetchingRoutePreparer:
  return learning_backfill._PrefetchingRoutePreparer(
    authority_index=1,
    routes=(),
    local_prepare=lambda _route: None,
    helper_prepare=lambda _route, _abort: None,
    scratch_parent=artifact_root,
    abort_requested=lambda: False,
  )


def test_live_prepared_route_close_removes_only_exact_empty_lane(
  tmp_path: Path,
) -> None:
  lane = make_empty_prefetch_lane(tmp_path)
  scratch = lane._ensure_scratch_directory()

  lane.close()

  assert not scratch.exists()


def test_live_prepared_route_close_preserves_unknown_child(
  tmp_path: Path,
) -> None:
  lane = make_empty_prefetch_lane(tmp_path)
  scratch = lane._ensure_scratch_directory()
  sentinel = scratch / "foreign-sentinel"
  sentinel.write_bytes(b"never recursively delete")

  with pytest.raises(BackfillError, match="unknown child"):
    lane.close()

  assert sentinel.read_bytes() == b"never recursively delete"
  assert not (
    tmp_path / learning_backfill.PREPARED_ROUTE_SCRATCH_QUARANTINE
  ).exists()


def test_live_prepared_route_close_path_swap_preserves_sentinel(
  tmp_path: Path,
) -> None:
  lane = make_empty_prefetch_lane(tmp_path)
  scratch = lane._ensure_scratch_directory()
  displaced = tmp_path / "displaced-live-scratch"
  scratch.rename(displaced)
  scratch.mkdir(mode=0o700)
  sentinel = scratch / "must-survive"
  sentinel.write_bytes(b"replacement")

  with pytest.raises(BackfillError, match="inode changed"):
    lane.close()

  assert sentinel.read_bytes() == b"replacement"
  assert displaced.is_dir()


def test_four_worker_helper_failure_reaps_every_child_and_publishes_nothing(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  add_route(tmp_path / "logs", 0x10)
  add_route(tmp_path / "logs", 0x20)
  engine, _params, _calls, runtime_factory = make_engine(
    tmp_path,
    monkeypatch,
    replay_worker_count=4,
  )
  deterministic_prepare = learning_backfill.prepare_route
  children_before = {
    child.pid
    for child in learning_backfill.multiprocessing.active_children()
  }

  def fail_in_helper(route, **kwargs):
    if learning_backfill.multiprocessing.current_process().name.startswith(
      "blatv2-prepare-",
    ):
      raise BackfillError(
        "injected_helper_failure",
        f"helper rejected {route.route_name}",
      )
    return deterministic_prepare(route, **kwargs)

  monkeypatch.setattr(learning_backfill, "prepare_route", fail_in_helper)

  with pytest.raises(BackfillError) as raised:
    engine.run_once()

  assert raised.value.diagnostic == "injected_helper_failure"
  assert {
    child.pid
    for child in learning_backfill.multiprocessing.active_children()
  } == children_before
  paths = runtime_factory().artifact_paths
  assert not paths.backfill_pointer.exists()
  assert not paths.evidence.exists()
  assert not paths.manifest.exists()
  assert not paths.backfill_ledger.exists()
  assert not tuple(paths.root.glob(
    f"{learning_backfill.BACKFILL_SPOOL_DIRECTORY_PREFIX}*",
  ))


@pytest.mark.parametrize("parallel_worker_count", (2, 4))
def test_parallel_verification_keeps_existing_progress_schema_consistent(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  parallel_worker_count: int,
) -> None:
  baseline_root = tmp_path / "baseline"
  add_route(baseline_root / "logs", 0x10, segment_count=2)
  add_route(baseline_root / "logs", 0x20)
  baseline_engine, _, _, baseline_runtime_factory = make_engine(
    baseline_root,
    monkeypatch,
    replay_worker_count=parallel_worker_count,
  )
  baseline = baseline_engine.run_once()
  assert baseline.publication is not None
  baseline_artifacts = published_artifact_snapshot(
    baseline_runtime_factory,
  )

  observed_root = tmp_path / "observed"
  add_route(observed_root / "logs", 0x10, segment_count=2)
  add_route(observed_root / "logs", 0x20)
  engine, params, _, observed_runtime_factory = make_engine(
    observed_root,
    monkeypatch,
    with_progress=True,
    replay_worker_count=parallel_worker_count,
  )

  result = engine.run_once()

  assert result.publication is not None
  assert published_artifact_snapshot(observed_runtime_factory) == (
    baseline_artifacts
  )
  updates = [
    value
    for key, value, _ in params.puts
    if key == BACKFILL_PROGRESS_PARAM
  ]
  assert [value["phase"] for value in updates[-2:]] == [
    "comparing",
    "publishing",
  ]
  assert updates[-1]["completed_replay_segment_count"] == updates[-1][
    "total_replay_segment_count"
  ]
  assert updates[-1]["completed_work_units"] == updates[-1][
    "total_work_units"
  ]
  assert updates[-2]["pass_index"] == 2
  assert all(
    left["completed_work_units"] <= right["completed_work_units"]
    for left, right in zip(updates, updates[1:], strict=False)
  )
  assert all(
    value["pass_index"] == 1
    for value in updates
    if value["phase"] in {"reading_segment", "applying_route"}
  )
  assert {value["pass_count"] for value in updates} == {2}


def test_second_process_output_difference_is_nondeterministic_and_unpublished(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log_root = tmp_path / "logs"
  add_route(log_root, 0x10)
  engine, _, _, runtime_factory = make_engine(
    tmp_path,
    monkeypatch,
    replay_worker_count=2,
  )
  parent_pid = os.getpid()
  deterministic_prepare = learning_backfill.prepare_route

  def process_distinct_prepare(route, **kwargs):
    prepared = deterministic_prepare(route, **kwargs)
    if os.getpid() == parent_pid:
      return prepared
    provenance = dict(prepared.provenance)
    provenance["selected_event_stream_sha256"] = hashlib.sha256(
      f"worker-{os.getpid()}".encode("ascii"),
    ).hexdigest()
    return replace(prepared, provenance=provenance)

  monkeypatch.setattr(
    learning_backfill,
    "prepare_route",
    process_distinct_prepare,
  )

  with pytest.raises(BackfillError) as raised:
    engine.run_once()

  assert raised.value.diagnostic == "backfill_nondeterministic"
  artifact_paths = runtime_factory().artifact_paths
  assert not artifact_paths.backfill_pointer.exists()
  assert not artifact_paths.evidence.exists()
  assert not artifact_paths.manifest.exists()
  assert not artifact_paths.backfill_ledger.exists()


@pytest.mark.parametrize("failure_mode", ("put", "final_clear"))
def test_broken_progress_projection_is_fail_open(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  failure_mode: str,
) -> None:
  baseline_root = tmp_path / "baseline"
  add_route(baseline_root / "logs", 0x10, segment_count=2)
  baseline_engine, _, _, baseline_runtime_factory = make_engine(
    baseline_root,
    monkeypatch,
  )
  baseline = baseline_engine.run_once()
  assert baseline.publication is not None
  baseline_runtime = baseline_runtime_factory()
  baseline_ledger = baseline_runtime.artifact_paths.backfill_ledger.read_bytes()

  observed_root = tmp_path / "observed"
  add_route(observed_root / "logs", 0x10, segment_count=2)
  observed_engine, operation_params, _, observed_runtime_factory = (
    make_engine(
      observed_root,
      monkeypatch,
    )
  )
  observed_engine.backfill_progress = BackfillProgressPublisher(
    FaultInjectedProgressParams(failure_mode),
  )
  observed = observed_engine.run_once()
  assert observed.publication is not None
  observed_runtime = observed_runtime_factory()
  observed_ledger = observed_runtime.artifact_paths.backfill_ledger.read_bytes()

  assert observed.publication.finalization == (
    baseline.publication.finalization
  )
  assert observed_ledger == baseline_ledger
  assert observed.publication.generation_sha256 == (
    baseline.publication.generation_sha256
  )
  assert observed.publication.ledger_sha256 == (
    baseline.publication.ledger_sha256
  )
  operation = operation_params.values[LEARNING_OPERATION_STATUS_PARAM]
  assert type(operation) is dict
  assert operation["state"] == "idle"
  assert operation["terminal"] is True


def test_publication_plus_locked_route_leaves_truthful_finalizing_status(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log_root = tmp_path / "logs"
  complete_name, _ = add_route(log_root, 0x10)
  pending_name, _ = add_route(log_root, 0x20, locked=True)
  engine, params, prepare_calls, _ = make_engine(
    tmp_path,
    monkeypatch,
    pending_route_identity=route_identity_sha256(pending_name),
  )

  result = engine.run_once()

  assert result.publication is not None
  assert result.pending_logger_close
  assert prepare_calls == Counter({complete_name: 2})
  final = params.values[LEARNING_OPERATION_STATUS_PARAM]
  assert type(final) is dict
  assert final["state"] == "finalizing"
  assert final["diagnostic"] == "finalizing_drive"
  assert final["terminal"] is False
  assert final["last_route_identity"] == route_identity_sha256(pending_name)
  assert final["evidence_sha256"] == (
    result.publication.finalization.evidence_sha256
  )
  assert final["ledger_sha256"] == result.publication.ledger_sha256


def test_locked_route_gets_full_poll_after_first_unlocked_discovery(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log_root = tmp_path / "logs"
  route_name, _ = add_route(log_root, 0x20, locked=True)
  lock = log_root / f"{route_name}--0" / "rlog.lock"
  engine, _params, prepare_calls, _ = make_engine(
    tmp_path,
    monkeypatch,
    pending_route_identity=route_identity_sha256(route_name),
  )
  real_discovery = learning_backfill.discover_full_rlog_state
  discovery_calls = 0

  def remove_lock_after_snapshot(*args, **kwargs):
    nonlocal discovery_calls
    discovery_calls += 1
    snapshot = real_discovery(*args, **kwargs)
    if discovery_calls == 1:
      assert snapshot.pending_logger_close
      assert snapshot.candidates == ()
      lock.unlink()
    return snapshot

  monkeypatch.setattr(
    learning_backfill,
    "discover_full_rlog_state",
    remove_lock_after_snapshot,
  )

  first = engine.run_once()
  assert prepare_calls[route_name] == 0
  second = engine.run_once()
  assert prepare_calls[route_name] == 0
  third = engine.run_once()

  assert first.publication is None
  assert first.pending_logger_close
  assert second.publication is None
  assert second.pending_logger_close
  assert third.publication is not None
  assert not third.pending_logger_close
  assert discovery_calls == 3
  assert prepare_calls[route_name] == 2


def test_current_unlocked_route_gets_one_logger_quiescence_cycle(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log_root = tmp_path / "logs"
  route_name, _ = add_route(log_root, 0x20)
  engine, _params, prepare_calls, _ = make_engine(
    tmp_path,
    monkeypatch,
    pending_route_identity=route_identity_sha256(route_name),
  )

  deferred = engine.run_once()
  replayed = engine.run_once()

  assert deferred.publication is None
  assert deferred.pending_logger_close
  assert replayed.publication is not None
  assert not replayed.pending_logger_close
  assert prepare_calls[route_name] == 2


def test_backfill_status_write_aborts_after_onroad_handoff(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  engine, params, _calls, runtime_factory = make_engine(
    tmp_path,
    monkeypatch,
  )
  live_status = {"owner": "live-after-onroad-handoff"}
  params.values[LEARNING_OPERATION_STATUS_PARAM] = live_status
  engine.abort_requested = lambda: True

  with pytest.raises(BackfillError) as raised:
    engine._publish(
      runtime_factory(),
      state=learning_backfill.LearningOperationState.BACKFILLING,
      diagnostic="scanning_routes",
    )

  assert raised.value.diagnostic == "unexpected_error"
  assert params.values[LEARNING_OPERATION_STATUS_PARAM] is live_status


class AbortCoordinator:
  ingested_sample_count = 0
  accepted_sample_count = 0

  def finalize(self):
    raise AssertionError("aborted replay must not finalize")


class AbortRuntime:
  def __init__(self) -> None:
    self.coordinator = AbortCoordinator()
    self.ingest_calls = 0
    self.routes: list[tuple[int, str, str | None]] = []

  def transition_onroad(
    self,
    route_identity_sha256: str,
    route_content_sha256: str | None,
    *,
    route_counter: int,
  ) -> None:
    self.routes.append((route_counter, route_identity_sha256, route_content_sha256))

  def ingest(self, _frame: object) -> None:
    self.ingest_calls += 1

  def transition_offroad_without_persist(self) -> None:
    return None


def test_forked_replay_worker_cancel_reaps_child() -> None:
  context = learning_backfill.multiprocessing.get_context("fork")
  entered = context.Event()
  exited = context.Event()
  idle_wait = context.Event()

  def cancellable_replay(worker_abort_requested):
    entered.set()
    try:
      while not worker_abort_requested():
        idle_wait.wait(timeout=0.01)
      raise BackfillError(
        "unexpected_error",
        "test worker observed cancellation",
      )
    finally:
      exited.set()

  worker = learning_backfill._ForkedReplayWorker(
    replay=cancellable_replay,
    abort_requested=lambda: False,
  )
  process = worker._process
  assert entered.wait(timeout=5.0)

  worker.cancel()

  assert exited.wait(timeout=5.0)
  assert not process.is_alive()
  assert process.exitcode == 0


def test_forked_replay_worker_propagates_error_and_reaps_child() -> None:
  def failing_replay(_worker_abort_requested):
    raise BackfillError(
      "injected_worker_failure",
      "deterministic worker failure",
    )

  worker = learning_backfill._ForkedReplayWorker(
    replay=failing_replay,
    abort_requested=lambda: False,
  )
  process = worker._process

  with pytest.raises(BackfillError) as raised:
    worker.result()

  assert raised.value.diagnostic == "injected_worker_failure"
  assert not process.is_alive()
  assert process.exitcode == 0


@pytest.mark.parametrize("parallel_worker_count", (2, 4))
def test_parallel_engine_closes_inherited_writer_lock_in_every_child(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  parallel_worker_count: int,
) -> None:
  add_route(tmp_path / "logs", 0x10)
  add_route(tmp_path / "logs", 0x20)
  engine, _, _, _ = make_engine(
    tmp_path,
    monkeypatch,
    replay_worker_count=parallel_worker_count,
  )
  parent_pid = os.getpid()
  deterministic_prepare = learning_backfill.prepare_route
  forked_worker = learning_backfill._ForkedReplayWorker
  inherited_writer: list[tuple[int, int, int]] = []

  def capture_writer_lock(**kwargs):
    inherited_close_fds = kwargs["inherited_close_fds"]
    assert len(inherited_close_fds) == 1
    descriptor = inherited_close_fds[0]
    lock_stat = os.fstat(descriptor)
    inherited_writer.append((
      descriptor,
      lock_stat.st_dev,
      lock_stat.st_ino,
    ))
    return forked_worker(**kwargs)

  def assert_lock_absent_in_child(route, **kwargs):
    if os.getpid() != parent_pid:
      descriptor, expected_device, expected_inode = inherited_writer[0]
      try:
        child_stat = os.fstat(descriptor)
      except OSError:
        child_stat = None
      if child_stat is not None and (
        child_stat.st_dev,
        child_stat.st_ino,
      ) == (expected_device, expected_inode):
        raise BackfillError(
          "inherited_writer_lock_open",
          "replay child retained the parent's writer lock",
        )
    return deterministic_prepare(route, **kwargs)

  monkeypatch.setattr(
    learning_backfill,
    "_ForkedReplayWorker",
    capture_writer_lock,
  )
  monkeypatch.setattr(
    learning_backfill,
    "prepare_route",
    assert_lock_absent_in_child,
  )

  result = engine.run_once()

  assert result.publication is not None
  assert len(inherited_writer) == 1


def test_forked_replay_process_start_failure_has_no_child(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  context = learning_backfill.multiprocessing.get_context("fork")
  process_type = type(context.Process())
  children_before = {
    child.pid
    for child in learning_backfill.multiprocessing.active_children()
  }

  def fail_start(_process) -> None:
    raise OSError("injected process start failure")

  monkeypatch.setattr(process_type, "start", fail_start)

  with pytest.raises(BackfillError) as raised:
    learning_backfill._ForkedReplayWorker(
      replay=lambda _worker_abort_requested: None,
      abort_requested=lambda: False,
    )

  assert raised.value.diagnostic == "unexpected_error"
  assert isinstance(raised.value.__cause__, OSError)
  assert {
    child.pid
    for child in learning_backfill.multiprocessing.active_children()
  } == children_before


def test_forked_replay_pre_ready_failure_reaps_started_child() -> None:
  children_before = {
    child.pid
    for child in learning_backfill.multiprocessing.active_children()
  }

  with pytest.raises(BackfillError) as raised:
    learning_backfill._ForkedReplayWorker(
      replay=lambda _worker_abort_requested: None,
      abort_requested=lambda: False,
      inherited_close_fds=(-1,),
    )

  assert raised.value.diagnostic == "unexpected_error"
  assert {
    child.pid
    for child in learning_backfill.multiprocessing.active_children()
  } == children_before


def test_parent_runtime_restore_failure_starts_no_worker(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  add_route(tmp_path / "logs", 0x10)
  engine, _, _, runtime_factory = make_engine(
    tmp_path,
    monkeypatch,
    replay_worker_count=2,
  )
  runtime_calls = 0
  worker_starts = 0

  def fail_parent_replay_runtime():
    nonlocal runtime_calls
    runtime_calls += 1
    if runtime_calls == 3:
      raise BackfillError(
        "injected_restore_failure",
        "primary replay runtime restore failed",
      )
    return runtime_factory()

  def count_worker_start(**_kwargs):
    nonlocal worker_starts
    worker_starts += 1
    raise AssertionError("worker started before parent runtime restored")

  engine.runtime_factory = fail_parent_replay_runtime
  monkeypatch.setattr(
    learning_backfill,
    "_ForkedReplayWorker",
    count_worker_start,
  )

  with pytest.raises(BackfillError) as raised:
    engine.run_once()

  assert raised.value.diagnostic == "injected_restore_failure"
  assert worker_starts == 0
  assert not runtime_factory().artifact_paths.backfill_pointer.exists()


def test_primary_replay_failure_cancels_and_reaps_parallel_worker(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  add_route(tmp_path / "logs", 0x10)
  engine, _, _, runtime_factory = make_engine(
    tmp_path,
    monkeypatch,
    replay_worker_count=2,
  )
  context = learning_backfill.multiprocessing.get_context("fork")
  child_entered = context.Event()
  child_exited = context.Event()
  idle_wait = context.Event()
  parent_pid = os.getpid()

  def fail_primary_while_child_waits(_route, **kwargs):
    if os.getpid() == parent_pid:
      assert child_entered.wait(timeout=5.0)
      raise BackfillError(
        "injected_primary_failure",
        "primary replay failed after verification worker started",
      )
    worker_abort_requested = kwargs["abort_requested"]
    child_entered.set()
    try:
      while not worker_abort_requested():
        idle_wait.wait(timeout=0.01)
      raise BackfillError(
        "unexpected_error",
        "verification worker observed cancellation",
      )
    finally:
      child_exited.set()

  monkeypatch.setattr(
    learning_backfill,
    "prepare_route",
    fail_primary_while_child_waits,
  )

  with pytest.raises(BackfillError) as raised:
    engine.run_once()

  assert raised.value.diagnostic == "injected_primary_failure"
  assert child_exited.wait(timeout=5.0)
  assert not runtime_factory().artifact_paths.backfill_pointer.exists()


def test_replay_aborts_periodically_before_publication() -> None:
  route = RouteCandidate(
    route_name="0000002a--000000002a",
    route_counter=0x2A,
    segments=(),
  )
  runtime = AbortRuntime()
  abort_checks = 0

  def abort_requested() -> bool:
    nonlocal abort_checks
    abort_checks += 1
    return abort_checks >= 3

  with pytest.raises(BackfillError):
    learning_backfill.replay_routes(
      runtime=runtime,
      routes=(route,),
      prepare=lambda _route: PreparedRoute(
        frames=tuple(range(600)),
        controls_witness_count=600,
        unresolved_witness_count=0,
        gap_count=0,
        provenance={},
      ),
      abort_requested=abort_requested,
    )
  assert runtime.ingest_calls == 256
  assert runtime.routes == [(route.route_counter, route.display_identity, None)]


def test_replay_requires_one_sample_disposition_per_prepared_frame() -> None:
  route = RouteCandidate(
    route_name="0000002a--000000002a",
    route_counter=0x2A,
    segments=(),
  )
  runtime = AbortRuntime()
  with pytest.raises(BackfillError) as raised:
    learning_backfill.replay_routes(
      runtime=runtime,
      routes=(route,),
      prepare=lambda _route: PreparedRoute(
        frames=(object(),),
        controls_witness_count=1,
        unresolved_witness_count=0,
        gap_count=0,
        provenance={},
      ),
    )
  assert raised.value.diagnostic == "backfill_nondeterministic"
  assert runtime.ingest_calls == 1


def test_route_local_rejection_does_not_block_later_good_route(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log_root = tmp_path / "logs"
  bad_name, _ = add_route(log_root, 0x10)
  good_name, _ = add_route(log_root, 0x20)
  _engine, _params, _calls, runtime_factory = make_engine(
    tmp_path,
    monkeypatch,
  )
  cp = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
  routes = discover_complete_route_candidates(log_root)
  baseline = runtime_factory().coordinator.finalize().evidence_sha256
  runtime = runtime_factory()

  def prepare(route):
    if route.route_name == bad_name:
      raise RouteRejected(
        "invalid_route_version",
        "representative malformed historical InitData version",
      )
    return PreparedRoute(
      frames=tuple(
        route_frame(
          cp,
          1_000_000_000 + index * 10_000_000,
          first_in_route=index == 0,
        )
        for index in range(CAUSAL_ROUTE_FRAME_COUNT)
      ),
      controls_witness_count=CAUSAL_ROUTE_FRAME_COUNT,
      unresolved_witness_count=0,
      gap_count=0,
      provenance={
        "canonical_join_schema_version": CANONICAL_JOIN_SCHEMA_VERSION,
        "car_params_sha256": hashlib.sha256(b"car-params").hexdigest(),
        "dongle_id_sha256": hashlib.sha256(b"dongle").hexdigest(),
        "extractor_schema_version": (
          learning_backfill.NATIVE_EXTRACTOR_SCHEMA_VERSION
        ),
        "log_schema_blob": "4" * 40,
        "opendbc_commit": "2" * 40,
        "panda_commit": "3" * 40,
        "physical_compatibility_sha256": hashlib.sha256(
          b"physical",
        ).hexdigest(),
        "route_version": "test-version",
        "selected_event_stream_sha256": hashlib.sha256(
          route.route_name.encode("ascii"),
        ).hexdigest(),
        "superproject_commit": "1" * 40,
      },
    )

  replay = learning_backfill.replay_routes(
    runtime=runtime,
    routes=routes,
    prepare=prepare,
  )

  assert [result.route.route_name for result in replay.results] == [
    bad_name,
    good_name,
  ]
  rejected, ingested = replay.results
  assert rejected.disposition == "rejected"
  assert rejected.diagnostic == "invalid_route_version"
  assert rejected.provenance is None
  assert (
    rejected.accepted_sample_count,
    rejected.rejected_sample_count,
    rejected.controls_witness_count,
    rejected.unresolved_witness_count,
  ) == (0, 0, 0, 0)
  assert ingested.disposition == "ingested"
  assert ingested.accepted_sample_count > 0
  assert replay.finalization.evidence_sha256 != baseline

  runtime_identity = runtime.runtime_bundle.calibration_identity_sha256
  ledger = extend_ledger(
    {
      "entries": [],
      "runtime_identity_sha256": runtime_identity,
      "schema_version": BACKFILL_LEDGER_SCHEMA_VERSION,
      "watermark_route_counter": None,
    },
    late_routes=(),
    replay_results=replay.results,
  )
  assert [entry["disposition"] for entry in ledger["entries"]] == [
    "rejected",
    "ingested",
  ]


class SimpleRoute:
  route_name = "00000001--0000000001"
  route_counter = 1
  segments = ()

  @property
  def display_identity(self) -> str:
    return route_identity_sha256(self.route_name)
