from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import pytest  # noqa: TID251

from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from openpilot.selfdrive.controls.lib.blatv2 import learning_backfill
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  BACKFILL_LEDGER_SCHEMA_VERSION,
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
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  MeasuredLearningFrame,
  build_persistent_learning_runtime,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
)


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


def route_frame(cp, mono_ns: int) -> MeasuredLearningFrame:
  return MeasuredLearningFrame(
    sample_mono_ns=mono_ns,
    speed_mps=10.0,
    steering_angle_deg=0.0,
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
) -> tuple[str, Path]:
  route_name = f"{counter:08x}--{counter:010x}"
  directory = log_root / f"{route_name}--0"
  directory.mkdir(parents=True)
  rlog = directory / "rlog"
  rlog.write_bytes(f"route-{counter}".encode("ascii"))
  if locked:
    (directory / "rlog.lock").touch()
  return route_name, rlog


def make_engine(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  *,
  pending_route_identity: str | None = None,
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

  def fake_prepare(route, **_kwargs):
    prepare_calls[route.route_name] += 1
    base = (route.route_counter + 1) * 1_000_000_000
    return PreparedRoute(
      frames=tuple(
        route_frame(cp, base + index * 10_000_000)
        for index in range(3)
      ),
      controls_witness_count=3,
      unresolved_witness_count=0,
      gap_count=0,
      provenance={
        "canonical_join_schema_version": 1,
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
    abort_requested=lambda: False,
    pending_route_identity=pending_route_identity,
  )
  return engine, params, prepare_calls, runtime_factory


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
    runtime_identity_sha256=runtime.runtime_bundle.identity_sha256,
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
    runtime_identity_sha256=runtime.runtime_bundle.identity_sha256,
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
        )
        for index in range(3)
      ),
      controls_witness_count=3,
      unresolved_witness_count=0,
      gap_count=0,
      provenance={
        "canonical_join_schema_version": 1,
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

  runtime_identity = runtime.runtime_bundle.identity_sha256
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
