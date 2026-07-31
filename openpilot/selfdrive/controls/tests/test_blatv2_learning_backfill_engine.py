from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import os
from pathlib import Path

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
    return PreparedRoute(
      frames=tuple(
        route_frame(
          cp,
          base + index * 10_000_000,
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
        "extractor_schema_version": 1,
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
  )
  return engine, params, prepare_calls, runtime_factory


@pytest.mark.parametrize("replay_worker_count", (0, 3))
def test_replay_worker_count_is_bounded(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  replay_worker_count: int,
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


def test_two_replay_workers_publish_exact_single_worker_artifacts(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  serial_root = tmp_path / "serial"
  add_route(serial_root / "logs", 0x10, segment_count=2)
  add_route(serial_root / "logs", 0x20)
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
  parallel_engine, _, _, parallel_runtime_factory = make_engine(
    parallel_root,
    monkeypatch,
    replay_worker_count=2,
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


def test_parallel_verification_keeps_existing_progress_schema_consistent(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  baseline_root = tmp_path / "baseline"
  add_route(baseline_root / "logs", 0x10, segment_count=2)
  baseline_engine, _, _, baseline_runtime_factory = make_engine(
    baseline_root,
    monkeypatch,
    replay_worker_count=2,
  )
  baseline = baseline_engine.run_once()
  assert baseline.publication is not None
  baseline_artifacts = published_artifact_snapshot(
    baseline_runtime_factory,
  )

  observed_root = tmp_path / "observed"
  add_route(observed_root / "logs", 0x10, segment_count=2)
  engine, params, _, observed_runtime_factory = make_engine(
    observed_root,
    monkeypatch,
    with_progress=True,
    replay_worker_count=2,
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

  def transition_onroad(self) -> None:
    return None

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


def test_parallel_engine_closes_inherited_writer_lock_in_child(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  add_route(tmp_path / "logs", 0x10)
  engine, _, _, _ = make_engine(
    tmp_path,
    monkeypatch,
    replay_worker_count=2,
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
          "verification worker retained the parent's writer lock",
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
  route = SimpleRoute()
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
        "extractor_schema_version": 1,
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
