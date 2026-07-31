from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest  # noqa: TID251

from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import (
  CAR,
  HyundaiFlags,
  HyundaiSafetyFlags,
)
from opendbc.car.structs import car
from openpilot.selfdrive.controls.lib.blatv2 import learning_backfill
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  BuildDescriptor,
  BuildDescriptorRegistry,
  ExtractedEvent,
  RouteCandidate,
  RouteRejected,
  RouteSegment,
  _validate_route_bundle,
  discover_complete_route_candidates,
  has_pending_full_rlog,
  prepare_route,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  build_detected_runtime_bundle,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
  build_runtime_vehicle_bundle,
)


ROOT_COMMIT = "1" * 40
DONGLE_ID = "device-dongle"


def rack_dynamics() -> ProvisionalRackDynamics:
  return ProvisionalRackDynamics(
    rack_gain_deg_s2_per_torque=4000.0,
    rack_damping_per_s=10.0,
    rack_rate_resolution_deg_s=4.0,
    provenance="backfill route test seed",
  )


def reviewed_descriptor(**overrides: object) -> BuildDescriptor:
  fields: dict[str, object] = {
    "driver_allowance": 50,
    "driver_factor": 1,
    "driver_multiplier": 2,
    "log_schema_blob": "4" * 40,
    "opendbc_commit": "2" * 40,
    "panda_commit": "3" * 40,
    "production_envelope_verified": True,
    "rack_rate_resolution_deg_s": 4.0,
    "steer_delta_down": 7,
    "steer_delta_up": 4,
    "steer_max": 409,
    "steer_step": 1,
    "superproject_commit": ROOT_COMMIT,
    "supported_vehicle_identity": str(CAR.HYUNDAI_PALISADE),
  }
  fields.update(overrides)
  return BuildDescriptor(**fields)


class FakeCarParamsPayload:
  def __init__(self, encoded: bytes) -> None:
    self.encoded = encoded

  def as_builder(self) -> FakeCarParamsPayload:
    return self

  def to_bytes(self) -> bytes:
    return self.encoded


class FakeEvent:
  def __init__(
    self,
    which: str,
    mono_ns: int,
    *,
    valid: bool = True,
    payload: object | None = None,
  ) -> None:
    self._which = which
    self.logMonoTime = mono_ns
    self.valid = valid
    if which == "initData":
      self.initData = payload
    elif which == "sentinel":
      self.sentinel = SimpleNamespace(type=payload)
    elif which == "carParams":
      self.carParams = FakeCarParamsPayload(bytes(payload))
    elif which in learning_backfill._SOURCE_SERVICES:
      setattr(self, which, payload)

  def which(self) -> str:
    return self._which


class FakeEventContext(AbstractContextManager[FakeEvent]):
  def __init__(self, event: FakeEvent) -> None:
    self.event = event

  def __enter__(self) -> FakeEvent:
    return self.event

  def __exit__(self, *_exc_info: object) -> None:
    return None


def source_message(service: str, cp: car.CarParams) -> object:
  if service == "carControl":
    return SimpleNamespace(latActive=True)
  if service == "carState":
    return SimpleNamespace(
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
    )
  if service == "carOutput":
    return SimpleNamespace(
      actuatorsOutput=SimpleNamespace(torque=0.0),
    )
  if service == "liveParameters":
    return SimpleNamespace(
      valid=True,
      angleOffsetValid=True,
      steerRatioValid=True,
      stiffnessFactorValid=True,
      angleOffsetDeg=0.0,
      steerRatio=float(cp.steerRatio),
      stiffnessFactor=float(cp.tireStiffnessFactor),
      roll=0.0,
    )
  raise AssertionError(service)


def extracted_fixture(
  cp: car.CarParams,
  *,
  control_count: int = 200,
  source_mode: str = "all",
  invalid_kind: str | None = None,
  dirty: bool = False,
  dongle_id: str = DONGLE_ID,
  root_commit: str = ROOT_COMMIT,
  route_version: str = "test-version",
  duplicate_car_params: bool = False,
  structure: str = "normal",
) -> tuple[
  tuple[ExtractedEvent, ...],
  dict[bytes, FakeEvent],
]:
  events: list[FakeEvent] = []

  def add(
    which: str,
    mono_ns: int,
    *,
    payload: object | None = None,
    valid: bool = True,
  ) -> None:
    events.append(FakeEvent(
      which,
      mono_ns,
      valid=valid and invalid_kind != which,
      payload=payload,
    ))

  base = 1_000_000_000
  if structure != "missing_init":
    add(
      "initData",
      base - 30_000_000,
      payload=SimpleNamespace(
        gitCommit=root_commit,
        dirty=dirty,
        dongleId=dongle_id,
        version=route_version,
      ),
    )
  if structure == "duplicate_init":
    add(
      "initData",
      base - 25_000_000,
      payload=SimpleNamespace(
        gitCommit=root_commit,
        dirty=dirty,
        dongleId=dongle_id,
        version=route_version,
      ),
    )
  if structure == "payload_before_start":
    add("controlsState", base - 22_000_000)
  add(
    "sentinel",
    base - 20_000_000,
    payload="startOfRoute",
  )
  add(
    "carParams",
    base - 10_000_000,
    payload=b"canonical-route-car-params",
  )
  if duplicate_car_params:
    add(
      "carParams",
      base - 5_000_000,
      payload=b"canonical-route-car-params",
    )

  for index in range(control_count):
    timestamp = base + index * 10_000_000
    if source_mode == "gap" and index >= control_count // 2:
      timestamp += 30_000_000
    publish_sources = (
      source_mode in {"all", "gap"}
      or source_mode == "stop" and index == 0
      or source_mode == "late" and index >= control_count // 2
      or source_mode == "control_before_equal" and index > 0
    )
    if source_mode == "control_before_equal" and index == 0:
      add("controlsState", timestamp)
      for service in learning_backfill._SOURCE_SERVICES:
        add(
          service,
          timestamp,
          payload=source_message(service, cp),
        )
      continue
    if publish_sources:
      for service in learning_backfill._SOURCE_SERVICES:
        add(
          service,
          timestamp,
          payload=source_message(service, cp),
        )
    add("controlsState", timestamp)

  end_time = base + control_count * 10_000_000 + 40_000_000
  add("sentinel", end_time, payload="endOfRoute")
  if structure == "payload_after_end":
    add(
      "carState",
      end_time + 1,
      payload=source_message("carState", cp),
    )

  mapping: dict[bytes, FakeEvent] = {}
  records = []
  for ordinal, event in enumerate(events):
    encoded = f"event-{ordinal}".encode("ascii")
    mapping[encoded] = event
    records.append(ExtractedEvent(
      which=learning_backfill._EVENT_WHICH[event.which()],
      mono_ns=event.logMonoTime,
      ordinal=ordinal,
      encoded=encoded,
    ))
  return tuple(records), mapping


def prepare_fixture(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  *,
  records: tuple[ExtractedEvent, ...],
  events: dict[bytes, FakeEvent],
  cp: car.CarParams,
  descriptor: BuildDescriptor | None = None,
  car_params_decoder=lambda _encoded: None,
  route_bundle_factory=None,
):
  segment_path = tmp_path / "rlog"
  segment_path.write_bytes(b"immutable-rlog")
  segment = RouteSegment(
    index=0,
    path=segment_path,
    sha256=hashlib.sha256(segment_path.read_bytes()).hexdigest(),
    size_bytes=segment_path.stat().st_size,
  )
  route = RouteCandidate(
    route_name="00000001--0000000001",
    route_counter=1,
    segments=(segment,),
  )
  current_bundle, _, _ = build_detected_runtime_bundle(
    car_params=cp,
    provisional_rack_dynamics=rack_dynamics(),
  )
  selected_descriptor = descriptor or reviewed_descriptor()
  monkeypatch.setattr(
    learning_backfill,
    "extract_segment_events",
    lambda *_args, **_kwargs: records,
  )
  decoder = (
    (lambda _encoded: cp)
    if car_params_decoder is None
    else car_params_decoder
  )
  bundle_factory = (
    (lambda _route_cp, _descriptor: current_bundle)
    if route_bundle_factory is None
    else route_bundle_factory
  )
  return prepare_route(
    route,
    extractor_path=tmp_path / "unused-extractor",
    event_reader=lambda encoded: FakeEventContext(events[encoded]),
    car_params_decoder=decoder,
    descriptor_registry=BuildDescriptorRegistry((selected_descriptor,)),
    route_bundle_factory=bundle_factory,
    current_car_params=cp,
    current_bundle=current_bundle,
    expected_dongle_id=DONGLE_ID,
  )


@pytest.fixture
def palisade_cp() -> car.CarParams:
  cp = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
  cp.carVin = "KM8R74HE0LU000001"
  return cp


def test_equal_timestamp_join_uses_original_record_order(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  palisade_cp: car.CarParams,
) -> None:
  records, events = extracted_fixture(
    palisade_cp,
    source_mode="control_before_equal",
  )
  prepared = prepare_fixture(
    tmp_path,
    monkeypatch,
    records=records,
    events=events,
    cp=palisade_cp,
    car_params_decoder=None,
  )

  assert prepared.controls_witness_count == 200
  assert prepared.unresolved_witness_count == 1
  assert len(prepared.frames) == 199
  assert prepared.frames[0].sample_mono_ns == 1_010_000_000


@pytest.mark.parametrize("invalid_kind", ["initData", "carParams", "sentinel"])
def test_invalid_structural_events_are_rejected(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  palisade_cp: car.CarParams,
  invalid_kind: str,
) -> None:
  records, events = extracted_fixture(
    palisade_cp,
    invalid_kind=invalid_kind,
  )
  with pytest.raises(RouteRejected) as raised:
    prepare_fixture(
      tmp_path,
      monkeypatch,
      records=records,
      events=events,
      cp=palisade_cp,
      car_params_decoder=None,
    )
  assert raised.value.reason == "invalid_provenance_event"


@pytest.mark.parametrize(
  "structure",
  [
    "missing_init",
    "duplicate_init",
    "payload_before_start",
    "payload_after_end",
  ],
)
def test_segment_structure_is_exact_and_fail_closed(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  palisade_cp: car.CarParams,
  structure: str,
) -> None:
  records, events = extracted_fixture(
    palisade_cp,
    structure=structure,
  )
  with pytest.raises(RouteRejected) as raised:
    prepare_fixture(
      tmp_path,
      monkeypatch,
      records=records,
      events=events,
      cp=palisade_cp,
      car_params_decoder=None,
    )
  assert raised.value.reason == "route_structure_mismatch"


def test_identical_duplicate_car_params_remain_route_wide_compatible(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  palisade_cp: car.CarParams,
) -> None:
  records, events = extracted_fixture(
    palisade_cp,
    duplicate_car_params=True,
  )
  prepared = prepare_fixture(
    tmp_path,
    monkeypatch,
    records=records,
    events=events,
    cp=palisade_cp,
    car_params_decoder=None,
  )
  assert len(prepared.frames) == 200


@pytest.mark.parametrize("source_mode", ["late", "stop", "gap"])
def test_quality_gate_covers_all_controls_and_route_tail(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  palisade_cp: car.CarParams,
  source_mode: str,
) -> None:
  records, events = extracted_fixture(
    palisade_cp,
    source_mode=source_mode,
  )
  with pytest.raises(RouteRejected) as raised:
    prepare_fixture(
      tmp_path,
      monkeypatch,
      records=records,
      events=events,
      cp=palisade_cp,
      car_params_decoder=None,
    )
  assert raised.value.reason == "measurement_continuity_failed"


@pytest.mark.parametrize(
  ("fixture_overrides", "expected_reason"),
  [
    ({"dirty": True}, "dirty_build"),
    ({"dongle_id": "other-device"}, "dongle_mismatch"),
    ({"root_commit": "9" * 40}, "unreviewed_build"),
  ],
)
def test_build_and_device_provenance_are_strict(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  palisade_cp: car.CarParams,
  fixture_overrides: dict[str, object],
  expected_reason: str,
) -> None:
  records, events = extracted_fixture(
    palisade_cp,
    **fixture_overrides,
  )
  with pytest.raises(RouteRejected) as raised:
    prepare_fixture(
      tmp_path,
      monkeypatch,
      records=records,
      events=events,
      cp=palisade_cp,
      car_params_decoder=None,
    )
  assert raised.value.reason == expected_reason


@pytest.mark.parametrize(
  "route_version",
  ("", " leading-space", "x" * 257, "\x00"),
)
def test_route_version_is_bounded_printable_and_route_local(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  palisade_cp: car.CarParams,
  route_version: str,
) -> None:
  records, events = extracted_fixture(
    palisade_cp,
    route_version=route_version,
  )
  with pytest.raises(RouteRejected) as raised:
    prepare_fixture(
      tmp_path,
      monkeypatch,
      records=records,
      events=events,
      cp=palisade_cp,
      car_params_decoder=None,
    )

  assert raised.value.reason == "invalid_route_version"


def test_unresolved_controls_population_hits_route_bound_during_decode(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  palisade_cp: car.CarParams,
) -> None:
  records, events = extracted_fixture(
    palisade_cp,
    control_count=3,
    source_mode="none",
  )
  monkeypatch.setattr(learning_backfill, "MAXIMUM_ROUTE_FRAMES", 2)

  with pytest.raises(RouteRejected) as raised:
    prepare_fixture(
      tmp_path,
      monkeypatch,
      records=records,
      events=events,
      cp=palisade_cp,
      car_params_decoder=None,
    )

  assert raised.value.reason == "route_too_large"


def test_route_local_decode_and_runtime_failures_are_isolated_rejections(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  palisade_cp: car.CarParams,
) -> None:
  records, events = extracted_fixture(palisade_cp)
  with pytest.raises(RouteRejected) as cp_failure:
    prepare_fixture(
      tmp_path,
      monkeypatch,
      records=records,
      events=events,
      cp=palisade_cp,
      car_params_decoder=lambda _encoded: (_ for _ in ()).throw(
        ValueError("corrupt CP"),
      ),
    )
  assert cp_failure.value.reason == "car_params_decode_failed"

  with pytest.raises(RouteRejected) as runtime_failure:
    prepare_fixture(
      tmp_path,
      monkeypatch,
      records=records,
      events=events,
      cp=palisade_cp,
      car_params_decoder=None,
      route_bundle_factory=lambda *_args: (_ for _ in ()).throw(
        KeyError("unsupported fingerprint"),
      ),
    )
  assert runtime_failure.value.reason == "route_runtime_unsupported"


def test_obsolete_384_envelope_is_rejected(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  palisade_cp: car.CarParams,
) -> None:
  records, events = extracted_fixture(palisade_cp)
  obsolete = reviewed_descriptor(
    steer_max=384,
    steer_delta_up=3,
  )
  with pytest.raises(RouteRejected) as raised:
    prepare_fixture(
      tmp_path,
      monkeypatch,
      records=records,
      events=events,
      cp=palisade_cp,
      descriptor=obsolete,
      car_params_decoder=None,
    )
  assert raised.value.reason == "controller_limits_mismatch"


def test_descriptor_proxy_recovers_reviewed_envelope_from_old_cp_flags(
  palisade_cp: car.CarParams,
) -> None:
  descriptor = reviewed_descriptor()
  current_bundle, _, _ = build_detected_runtime_bundle(
    car_params=palisade_cp,
    provisional_rack_dynamics=rack_dynamics(),
  )
  with car.CarParams.from_bytes(palisade_cp.to_bytes()) as reader:
    old_cp = reader.as_builder()
  old_cp.flags = int(old_cp.flags) & ~int(HyundaiFlags.BLATV2_HIGH_LIMITS)
  old_cp.safetyConfigs[-1].safetyParam = (
    int(old_cp.safetyConfigs[-1].safetyParam)
    & ~int(HyundaiSafetyFlags.BLATV2_HIGH_LIMITS)
  )

  flag_derived, old_interface, _ = build_detected_runtime_bundle(
    car_params=old_cp,
    provisional_rack_dynamics=rack_dynamics(),
  )
  assert (
    flag_derived.torque_limits.steer_max,
    flag_derived.torque_limits.delta_up,
    flag_derived.torque_limits.production_envelope_verified,
  ) == (384, 3, False)

  descriptor_bundle = build_runtime_vehicle_bundle(
    car_params=old_cp,
    car_interface_or_callback=old_interface,
    controller_params=descriptor.controller_params_proxy(),
    vehicle_identity=str(old_cp.carFingerprint),
    provisional_rack_dynamics=rack_dynamics(),
  )
  assert descriptor_bundle.torque_limits == current_bundle.torque_limits
  assert _validate_route_bundle(
    route_car_params=old_cp,
    route_bundle=descriptor_bundle,
    current_car_params=palisade_cp,
    current_bundle=current_bundle,
    descriptor=descriptor,
  )


def _segment(
  root: Path,
  route: str,
  index: int,
  *,
  filename: str = "rlog",
  lock: str | None = None,
) -> Path:
  directory = root / f"{route}--{index}"
  directory.mkdir()
  (directory / filename).write_bytes(
    f"{route}:{index}:{filename}".encode("ascii"),
  )
  if lock is not None:
    (directory / lock).touch()
  return directory


def test_device_route_discovery_requires_contiguous_unlocked_full_rlogs(
  tmp_path: Path,
) -> None:
  newest = "00000020--0000000020"
  oldest = "00000010--0000000010"
  _segment(tmp_path, newest, 0, filename="rlog")
  newest_one = _segment(tmp_path, newest, 1, filename="rlog")
  (newest_one / "rlog.zst").write_bytes(b"preferred-compressed")
  _segment(tmp_path, oldest, 0, filename="rlog.zst")
  _segment(tmp_path, "00000030--0000000030", 0, lock="other.lock")
  _segment(tmp_path, "00000040--0000000040", 0)
  _segment(tmp_path, "00000040--0000000040", 2)
  _segment(tmp_path, "00000050--0000000050", 0, filename="qlog.zst")

  discovered = discover_complete_route_candidates(tmp_path)

  assert [route.route_name for route in discovered] == [oldest, newest]
  assert [segment.index for segment in discovered[1].segments] == [0, 1]
  assert discovered[1].segments[1].path.name == "rlog.zst"
  assert discovered[1].segments[1].sha256 == hashlib.sha256(
    b"preferred-compressed",
  ).hexdigest()
  assert has_pending_full_rlog(tmp_path)


def test_route_pruning_during_discovery_does_not_block_other_routes(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  pruned = "00000010--0000000010"
  retained = "00000020--0000000020"
  pruned_directory = _segment(tmp_path, pruned, 0, filename="rlog")
  _segment(tmp_path, retained, 0, filename="rlog")
  original_hash = learning_backfill._sha256_file

  def prune_then_hash(path: Path, **kwargs) -> str:
    if path.parent == pruned_directory:
      path.unlink()
    return original_hash(path, **kwargs)

  monkeypatch.setattr(
    learning_backfill,
    "_sha256_file",
    prune_then_hash,
  )

  discovered = discover_complete_route_candidates(tmp_path)

  assert [route.route_name for route in discovered] == [retained]


def test_route_pruning_after_discovery_is_route_local_rejection(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  palisade_cp: car.CarParams,
) -> None:
  records, events = extracted_fixture(palisade_cp)
  original_hash = learning_backfill._sha256_file
  hash_calls = 0

  def disappear_before_second_hash(path: Path, **kwargs) -> str:
    nonlocal hash_calls
    hash_calls += 1
    if hash_calls == 2:
      path.unlink()
    return original_hash(path, **kwargs)

  monkeypatch.setattr(
    learning_backfill,
    "_sha256_file",
    disappear_before_second_hash,
  )
  with pytest.raises(RouteRejected) as raised:
    prepare_fixture(
      tmp_path,
      monkeypatch,
      records=records,
      events=events,
      cp=palisade_cp,
      car_params_decoder=None,
    )

  assert raised.value.reason == "segment_changed"
