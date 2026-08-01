from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest  # noqa: TID251

from opendbc.car.structs import car
from openpilot.selfdrive.controls.blatv2_backfilld import (
  BlatV2BackfillDaemon,
  _decode_car_params,
  _serialize_owned_car_params,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  BackfillError,
  BackfillRunResult,
  BuildDescriptorRegistry,
  discover_complete_route_candidates,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_operation_status import (
  LEARNING_OPERATION_STATUS_PARAM,
  LearningOperationState,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_progress import (
  BACKFILL_PROGRESS_PARAM,
)
from openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill import (
  BridgeFallbackUnavailableError,
)
from openpilot.selfdrive.controls.lib.blatv2.offdevice_protocol import (
  BridgeCorruptError,
  BridgeIncompatibleError,
  BridgeRemoteError,
  BridgeUnavailableError,
)
from openpilot.selfdrive.controls.lib.blatv2.offdevice_progress import (
  OFFDEVICE_PROGRESS_PARAM,
  OffdeviceFallbackReason,
)


RUNTIME_IDENTITY = hashlib.sha256(b"runtime").hexdigest()
ROUTE_IDENTITY = hashlib.sha256(b"route").hexdigest()


class FakeParams:
  def __init__(self) -> None:
    self.values: dict[str, object] = {"IsOffroad": True}

  def put(
    self,
    key: str,
    value: dict[str, object],
    *,
    block: bool,
  ) -> None:
    assert block is True
    self.values[key] = dict(value)

  def get(self, key: str, *, block: bool) -> object | None:
    assert block is False
    return self.values.get(key)

  def get_bool(self, key: str, block: bool = False) -> object:
    assert block is False
    return self.values.get(key)

  def remove(self, key: str) -> None:
    self.values.pop(key, None)


def test_terminal_failure_preserves_same_operation_progress(
  tmp_path: Path,
) -> None:
  params = FakeParams()
  daemon = BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
    extractor_path=tmp_path / "extractor",
    descriptor_path=tmp_path / "descriptors.json",
  )
  daemon.operation_status.publish(
    state=LearningOperationState.FINALIZING,
    diagnostic="publishing_backfill",
    new_operation=True,
    accepted_sample_count=17,
    rejected_sample_count=4,
    retry_count=2,
    runtime_identity_sha256=RUNTIME_IDENTITY,
    vehicle_identity="CAR",
    last_route_identity=ROUTE_IDENTITY,
  )
  before = daemon.operation_status.last_payload
  assert before is not None
  params.values[BACKFILL_PROGRESS_PARAM] = {"stale": "progress"}
  params.values[OFFDEVICE_PROGRESS_PARAM] = {"stale": "offdevice"}

  daemon._publish_failure(
    "backfill_publish_failed",
    SimpleNamespace(carFingerprint="CHANGED-CONTEXT-MUST-NOT-WIN"),
  )

  failed = params.values[LEARNING_OPERATION_STATUS_PARAM]
  assert type(failed) is dict
  assert failed["state"] == "failed"
  assert failed["diagnostic"] == "backfill_publish_failed"
  assert failed["terminal"] is True
  assert failed["operation_id"] == before["operation_id"]
  assert failed["sequence"] == before["sequence"] + 1
  assert failed["accepted_sample_count"] == 17
  assert failed["rejected_sample_count"] == 4
  assert failed["retry_count"] == 2
  assert failed["runtime_identity_sha256"] == RUNTIME_IDENTITY
  assert failed["vehicle_identity"] == "CAR"
  assert failed["current_route_identity"] is None
  assert failed["current_route_index"] is None
  assert failed["total_route_count"] is None
  assert BACKFILL_PROGRESS_PARAM not in params.values
  assert OFFDEVICE_PROGRESS_PARAM not in params.values


def test_terminal_failure_cannot_overwrite_live_onroad_owner(
  tmp_path: Path,
) -> None:
  params = FakeParams()
  daemon = BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
    extractor_path=tmp_path / "extractor",
    descriptor_path=tmp_path / "descriptors.json",
  )
  live_status = {"owner": "live-manager-transition"}
  params.values[LEARNING_OPERATION_STATUS_PARAM] = live_status
  params.values["IsOffroad"] = False

  daemon._publish_failure(
    "backfill_publish_failed",
    SimpleNamespace(carFingerprint="CAR"),
  )

  assert params.values[LEARNING_OPERATION_STATUS_PARAM] is live_status


def test_dirty_current_build_fails_before_replay(tmp_path: Path) -> None:
  daemon = BlatV2BackfillDaemon(
    params=FakeParams(),
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
    extractor_path=tmp_path / "extractor",
    descriptor_path=tmp_path / "descriptors.json",
    current_build_clean=lambda: False,
  )

  with pytest.raises(BackfillError) as raised:
    daemon._build_engine(SimpleNamespace(carFingerprint="CAR"))

  assert raised.value.diagnostic == "backfill_route_incompatible"
  assert "dirty" in str(raised.value)


def test_remote_contract_reuses_owned_car_params_wire_bytes(
  tmp_path: Path,
) -> None:
  params = FakeParams()
  params.values["GitCommit"] = "a" * 40
  daemon = BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
    descriptor_path=tmp_path / "descriptors.json",
  )
  original = car.CarParams.new_message()
  original.carFingerprint = "TEST CAR"
  serialized_car_params = bytes(original.to_bytes())
  car_params = _decode_car_params(serialized_car_params)
  owned_wire_bytes = _serialize_owned_car_params(car_params)
  runtime = SimpleNamespace(
    runtime_bundle=SimpleNamespace(calibration_identity_sha256="d" * 64),
  )
  engine = SimpleNamespace(
    descriptor_registry=SimpleNamespace(identity_sha256="e" * 64),
    expected_dongle_id="f" * 16,
    runtime_factory=MagicMock(return_value=runtime),
  )
  with (
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.get_commit",
      return_value="b" * 40,
    ),
    patch.object(
      BuildDescriptorRegistry,
      "from_json_file",
      return_value=SimpleNamespace(identity_sha256="f" * 64),
    ),
  ):
    first = daemon._remote_contract(engine, car_params, owned_wire_bytes)
    second = daemon._remote_contract(engine, car_params, owned_wire_bytes)

  assert base64.b64decode(first["car_params_b64"]) == serialized_car_params
  assert first == second
  assert first["vehicle_fingerprint"] == "TEST CAR"


def test_missing_remote_config_falls_back_to_local_without_current(
  tmp_path: Path,
) -> None:
  params = FakeParams()
  params.values[BACKFILL_PROGRESS_PARAM] = {"stale": "remote"}
  storage = tmp_path / "learning"
  daemon = BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=storage,
  )
  daemon._remote_contract = MagicMock(return_value={"source_commit": "a" * 40})
  engine = SimpleNamespace(expected_dongle_id="f" * 16)
  with (
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.default_bridge_config_directory",
      return_value=tmp_path / "missing-config",
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.load_bridge_secret",
      side_effect=BridgeUnavailableError("not configured"),
    ),
  ):
    assert daemon._prepare_remote(engine, object(), b"encoded") is None

  assert BACKFILL_PROGRESS_PARAM not in params.values
  offdevice = params.values[OFFDEVICE_PROGRESS_PARAM]
  assert type(offdevice) is dict
  assert offdevice["phase"] == "local_fallback"
  assert offdevice["fallback_reason_code"] == "worker_unavailable"
  assert not (storage / "CURRENT").exists()


def test_offdevice_progress_io_failure_cannot_block_local_fallback(
  tmp_path: Path,
) -> None:
  class FailingProgressParams(FakeParams):
    def put(self, key: str, value: dict[str, object], *, block: bool) -> None:
      if key == OFFDEVICE_PROGRESS_PARAM:
        raise OSError("display storage unavailable")
      super().put(key, value, block=block)

    def remove(self, key: str) -> None:
      if key == OFFDEVICE_PROGRESS_PARAM:
        raise OSError("display storage unavailable")
      super().remove(key)

  daemon = BlatV2BackfillDaemon(
    params=FailingProgressParams(),
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
  )
  daemon._remote_contract = MagicMock(return_value={"source_commit": "a" * 40})
  with (
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.default_bridge_config_directory",
      return_value=tmp_path / "missing-config",
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.load_bridge_secret",
      side_effect=BridgeUnavailableError("not configured"),
    ),
  ):
    assert daemon._prepare_remote(
      SimpleNamespace(expected_dongle_id="f" * 16),
      object(),
      b"encoded",
    ) is None


def test_remote_preparation_passes_protected_worker_host_to_discovery(
  tmp_path: Path,
) -> None:
  daemon = BlatV2BackfillDaemon(
    params=FakeParams(),
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
  )
  daemon._remote_contract = MagicMock(return_value={"source_commit": "a" * 40})
  runtime = SimpleNamespace(artifact_paths=SimpleNamespace(root=tmp_path))
  engine = SimpleNamespace(
    expected_dongle_id="f" * 16,
    runtime_factory=MagicMock(return_value=runtime),
  )
  session = object()
  discovery = MagicMock(return_value=object())
  with (
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.default_bridge_config_directory",
      return_value=tmp_path / "config",
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.load_bridge_secret",
      return_value=b"s" * 32,
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.load_bridge_worker_host",
      return_value="192.168.1.241",
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.discover_worker",
      discovery,
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.OffdeviceBridgeClient",
      return_value=object(),
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.prepare_remote_session",
      return_value=session,
    ),
  ):
    assert daemon._prepare_remote(engine, object(), b"encoded") is session

  assert discovery.call_args.kwargs["configured_host"] == "192.168.1.241"


@pytest.mark.parametrize(
  "code",
  [
    "artifact_not_found",
    "busy",
    "internal_error",
    "job_failed",
    "job_not_found",
    "route_unavailable",
  ],
)
def test_authenticated_worker_availability_error_falls_back_local(
  tmp_path: Path,
  code: str,
) -> None:
  params = FakeParams()
  params.values[BACKFILL_PROGRESS_PARAM] = {"active": "remote"}
  daemon = BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
  )
  daemon._remote_contract = MagicMock(return_value={"source_commit": "a" * 40})
  runtime = SimpleNamespace(artifact_paths=SimpleNamespace(root=tmp_path))
  engine = SimpleNamespace(
    expected_dongle_id="f" * 16,
    runtime_factory=MagicMock(return_value=runtime),
  )
  with (
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.default_bridge_config_directory",
      return_value=tmp_path / "config",
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.load_bridge_secret",
      return_value=b"s" * 32,
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.load_bridge_worker_host",
      return_value=None,
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.discover_worker",
      return_value=object(),
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.OffdeviceBridgeClient",
      return_value=object(),
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.prepare_remote_session",
      side_effect=BridgeRemoteError(code, "retry locally"),
    ),
  ):
    assert daemon._prepare_remote(engine, object(), b"encoded") is None

  assert BACKFILL_PROGRESS_PARAM not in params.values
  offdevice = params.values[OFFDEVICE_PROGRESS_PARAM]
  assert type(offdevice) is dict
  assert offdevice["phase"] == "local_fallback"
  expected_reason = {
    "artifact_not_found": "remote_artifact_unavailable",
    "busy": "worker_busy",
    "internal_error": "remote_job_failed",
    "job_failed": "remote_job_failed",
    "job_not_found": "network_interrupted",
    "route_unavailable": "remote_artifact_unavailable",
  }[code]
  assert offdevice["fallback_reason_code"] == expected_reason


def test_reason_carrying_unavailable_error_survives_local_fallback(
  tmp_path: Path,
) -> None:
  params = FakeParams()
  daemon = BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
  )
  daemon._remote_contract = MagicMock(return_value={"source_commit": "a" * 40})
  runtime = SimpleNamespace(artifact_paths=SimpleNamespace(root=tmp_path))
  engine = SimpleNamespace(
    expected_dongle_id="f" * 16,
    runtime_factory=MagicMock(return_value=runtime),
  )
  with (
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.default_bridge_config_directory",
      return_value=tmp_path / "config",
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.load_bridge_secret",
      return_value=b"s" * 32,
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.load_bridge_worker_host",
      return_value=None,
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.discover_worker",
      return_value=object(),
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.OffdeviceBridgeClient",
      return_value=object(),
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.prepare_remote_session",
      side_effect=BridgeFallbackUnavailableError(
        "route limit",
        OffdeviceFallbackReason.REMOTE_ROUTE_LIMIT,
      ),
    ),
  ):
    assert daemon._prepare_remote(engine, object(), b"encoded") is None

  offdevice = params.values[OFFDEVICE_PROGRESS_PARAM]
  assert type(offdevice) is dict
  assert offdevice["phase"] == "local_fallback"
  assert offdevice["fallback_reason_code"] == "remote_route_limit"


@pytest.mark.parametrize(
  ("remote_error", "diagnostic"),
  [
    (BridgeIncompatibleError("wrong source"), "backfill_route_incompatible"),
    (BridgeCorruptError("bad HMAC result"), "backfill_corrupt_log"),
    (
      BridgeRemoteError("source_mismatch", "wrong source"),
      "backfill_route_incompatible",
    ),
    (
      BridgeRemoteError("contract_mismatch", "wrong runtime"),
      "backfill_route_incompatible",
    ),
    (
      BridgeRemoteError("job_conflict", "idempotency conflict"),
      "backfill_reader_unavailable",
    ),
    (
      BridgeRemoteError("upload_invalid", "invalid upload"),
      "backfill_reader_unavailable",
    ),
    (
      BridgeRemoteError("artifact_bound_exceeded", "oversized artifact"),
      "backfill_reader_unavailable",
    ),
  ],
)
def test_authenticated_remote_contract_failure_maps_without_current(
  tmp_path: Path,
  remote_error: Exception,
  diagnostic: str,
) -> None:
  storage = tmp_path / "learning"
  daemon = BlatV2BackfillDaemon(
    params=FakeParams(),
    log_root=tmp_path / "logs",
    storage_root=storage,
  )
  daemon._remote_contract = MagicMock(return_value={"source_commit": "a" * 40})
  engine = SimpleNamespace(expected_dongle_id="f" * 16)
  with (
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.default_bridge_config_directory",
      return_value=tmp_path / "config",
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.load_bridge_secret",
      return_value=b"s" * 32,
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.load_bridge_worker_host",
      return_value=None,
    ),
    patch(
      "openpilot.selfdrive.controls.blatv2_backfilld.discover_worker",
      side_effect=remote_error,
    ),
  ):
    with pytest.raises(BackfillError) as raised:
      daemon._prepare_remote(engine, object(), b"encoded")

  assert raised.value.diagnostic == diagnostic
  assert not (storage / "CURRENT").exists()


def test_remote_session_closes_after_preserving_transaction_state(
  tmp_path: Path,
) -> None:
  params = FakeParams()
  daemon = BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
  )
  result = BackfillRunResult(publication=None, pending_logger_close=False)
  local_engine = SimpleNamespace(name="local")
  remote_engine = SimpleNamespace(run_once=MagicMock(return_value=result))
  events: list[str] = []
  session = MagicMock()
  session.build_engine.return_value = remote_engine
  session.preserve_transaction_state.side_effect = (
    lambda _engine: events.append("preserve")
  )
  session.close.side_effect = lambda: events.append("close")
  car_params = SimpleNamespace(
    carFingerprint="CAR",
    to_bytes=MagicMock(return_value=b"encoded"),
  )
  daemon._wait_for_car_params = MagicMock(return_value=car_params)
  daemon._build_engine = MagicMock(return_value=local_engine)
  daemon._prepare_remote = MagicMock(return_value=session)
  daemon._project_learning_status = MagicMock()

  def transition_onroad(_seconds: float) -> None:
    assert events == ["preserve", "close"]
    params.values["IsOffroad"] = False

  with patch(
    "openpilot.selfdrive.controls.blatv2_backfilld.time.sleep",
    side_effect=transition_onroad,
  ):
    daemon.run()

  session.preserve_transaction_state.assert_called_once_with(remote_engine)
  session.close.assert_called_once_with()
  daemon._project_learning_status.assert_called_once_with(remote_engine, None)


def test_remote_session_closes_when_remote_engine_construction_fails(
  tmp_path: Path,
) -> None:
  params = FakeParams()
  daemon = BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
  )
  car_params = SimpleNamespace(
    carFingerprint="CAR",
    to_bytes=MagicMock(return_value=b"encoded"),
  )
  local_engine = SimpleNamespace(name="local")
  session = MagicMock()
  session.build_engine.side_effect = RuntimeError("constructor failed")
  daemon._wait_for_car_params = MagicMock(return_value=car_params)
  daemon._build_engine = MagicMock(return_value=local_engine)
  daemon._prepare_remote = MagicMock(return_value=session)

  daemon.run()

  session.close.assert_called_once_with()
  session.preserve_transaction_state.assert_not_called()
  failure = params.values[LEARNING_OPERATION_STATUS_PARAM]
  assert type(failure) is dict
  assert failure["state"] == "failed"
  assert failure["diagnostic"] == "unexpected_error"


def test_backfill_prefers_current_route_car_params(
  tmp_path: Path,
) -> None:
  params = FakeParams()
  params.values["CarParamsPersistent"] = b"stale-384"
  params.values["CarParams"] = b"current-409"
  daemon = BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
    extractor_path=tmp_path / "extractor",
    descriptor_path=tmp_path / "descriptors.json",
  )

  with patch(
    "openpilot.selfdrive.controls.blatv2_backfilld._decode_car_params",
    side_effect=lambda encoded: encoded,
  ) as decode:
    selected = daemon._read_car_params()

  assert selected == b"current-409"
  decode.assert_called_once_with(b"current-409")


def test_noop_existing_generation_restores_learning_display(
  tmp_path: Path,
) -> None:
  params = FakeParams()
  daemon = BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
    extractor_path=tmp_path / "extractor",
    descriptor_path=tmp_path / "descriptors.json",
  )
  pointer = tmp_path / "learning" / "backfill_current.json"
  pointer.parent.mkdir()
  pointer.write_bytes(b"authenticated-by-runtime-factory")
  finalization = SimpleNamespace(name="authenticated-finalization")
  runtime_bundle = SimpleNamespace(name="runtime-bundle")
  runtime = SimpleNamespace(
    artifact_paths=SimpleNamespace(backfill_pointer=pointer),
    coordinator=SimpleNamespace(finalize=lambda: finalization),
    runtime_bundle=runtime_bundle,
  )
  engine = SimpleNamespace(runtime_factory=lambda: runtime)
  projected = {"authenticated_projection": True}

  with patch(
    "openpilot.selfdrive.controls.blatv2_backfilld.build_learning_status_payload",
    return_value=projected,
  ) as build:
    daemon._project_learning_status(engine, None)

  build.assert_called_once_with(
    finalization=finalization,
    runtime_bundle=runtime_bundle,
    drive_baseline=None,
  )
  assert params.values["BLaTv2LearningStatus"] == projected


@pytest.mark.parametrize("transition_point", ("finalize", "payload"))
def test_learning_display_cannot_cross_onroad_handoff(
  tmp_path: Path,
  transition_point: str,
) -> None:
  params = FakeParams()
  daemon = BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
    extractor_path=tmp_path / "extractor",
    descriptor_path=tmp_path / "descriptors.json",
  )
  pointer = tmp_path / "learning" / "backfill_current.json"
  pointer.parent.mkdir()
  pointer.write_bytes(b"authenticated-by-runtime-factory")
  finalization = SimpleNamespace(name="authenticated-finalization")

  def finalize():
    if transition_point == "finalize":
      params.values["IsOffroad"] = False
    return finalization

  runtime = SimpleNamespace(
    artifact_paths=SimpleNamespace(backfill_pointer=pointer),
    coordinator=SimpleNamespace(finalize=finalize),
    runtime_bundle=SimpleNamespace(name="runtime-bundle"),
  )
  engine = SimpleNamespace(runtime_factory=lambda: runtime)

  def build_payload(**_kwargs):
    if transition_point == "payload":
      params.values["IsOffroad"] = False
    return {"must_not_be_published": True}

  with patch(
    "openpilot.selfdrive.controls.blatv2_backfilld.build_learning_status_payload",
    side_effect=build_payload,
  ):
    daemon._project_learning_status(engine, None)

  assert "BLaTv2LearningStatus" not in params.values


def test_route_discovery_aborts_during_chunked_hash(
  tmp_path: Path,
) -> None:
  segment = tmp_path / "00000001--0000000001--0"
  segment.mkdir()
  (segment / "rlog").write_bytes(b"x" * (3 * 1024 * 1024))
  abort_checks = 0

  def abort_during_hash() -> bool:
    nonlocal abort_checks
    abort_checks += 1
    # Discovery reaches this threshold after opening the rlog and hashing its
    # first chunk, proving cancellation is checked within rather than after
    # the potentially multi-gigabyte file read.
    return abort_checks >= 8

  with pytest.raises(BackfillError) as raised:
    discover_complete_route_candidates(
      tmp_path,
      abort_requested=abort_during_hash,
    )

  assert raised.value.diagnostic == "unexpected_error"
  assert "hashing" in str(raised.value)


def test_daemon_stays_healthy_after_noop_without_rescanning(
  tmp_path: Path,
) -> None:
  params = FakeParams()
  daemon = BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
    extractor_path=tmp_path / "extractor",
    descriptor_path=tmp_path / "descriptors.json",
  )
  engine = SimpleNamespace(
    run_once=MagicMock(
      return_value=BackfillRunResult(
        publication=None,
        pending_logger_close=False,
      ),
    ),
  )
  car_params = SimpleNamespace(
    carFingerprint="CAR",
    to_bytes=MagicMock(return_value=b"encoded"),
  )
  daemon._wait_for_car_params = MagicMock(return_value=car_params)
  daemon._build_engine = MagicMock(return_value=engine)
  daemon._prepare_remote = MagicMock(return_value=None)
  daemon._project_learning_status = MagicMock()

  def transition_onroad(_seconds: float) -> None:
    assert engine.run_once.call_count == 1
    params.values["IsOffroad"] = False

  with patch(
    "openpilot.selfdrive.controls.blatv2_backfilld.time.sleep",
    side_effect=transition_onroad,
  ):
    daemon.run()

  engine.run_once.assert_called_once_with()
  daemon._project_learning_status.assert_called_once_with(engine, None)


def test_onroad_handoff_clears_progress_without_overwriting_live_status(
  tmp_path: Path,
) -> None:
  params = FakeParams()
  daemon = BlatV2BackfillDaemon(
    params=params,
    log_root=tmp_path / "logs",
    storage_root=tmp_path / "learning",
    extractor_path=tmp_path / "extractor",
    descriptor_path=tmp_path / "descriptors.json",
  )
  params.values[BACKFILL_PROGRESS_PARAM] = {"active": "offroad-progress"}
  params.values[OFFDEVICE_PROGRESS_PARAM] = {"active": "pc-progress"}
  engine = SimpleNamespace(
    run_once=MagicMock(
      return_value=BackfillRunResult(
        publication=None,
        pending_logger_close=False,
      ),
    ),
  )
  car_params = SimpleNamespace(
    carFingerprint="CAR",
    to_bytes=MagicMock(return_value=b"encoded"),
  )
  daemon._wait_for_car_params = MagicMock(return_value=car_params)
  daemon._build_engine = MagicMock(return_value=engine)
  daemon._prepare_remote = MagicMock(return_value=None)
  live_status = {"owner": "live-manager-transition"}

  def transition_onroad(
    _engine: object,
    _publication: object,
  ) -> None:
    params.values["IsOffroad"] = False
    params.values[LEARNING_OPERATION_STATUS_PARAM] = live_status

  daemon._project_learning_status = transition_onroad
  daemon.run()

  assert BACKFILL_PROGRESS_PARAM not in params.values
  assert OFFDEVICE_PROGRESS_PARAM not in params.values
  assert params.values[LEARNING_OPERATION_STATUS_PARAM] is live_status


def test_daemon_waits_full_poll_after_first_unlocked_discovery(
  tmp_path: Path,
) -> None:
  params = FakeParams()
  log_root = tmp_path / "logs"
  segment = log_root / "00000001--0000000001--0"
  segment.mkdir(parents=True)
  (segment / "rlog").write_bytes(b"complete-after-lock-release")
  lock = segment / "rlog.lock"
  lock.touch()
  daemon = BlatV2BackfillDaemon(
    params=params,
    log_root=log_root,
    storage_root=tmp_path / "learning",
    extractor_path=tmp_path / "extractor",
    descriptor_path=tmp_path / "descriptors.json",
  )
  publication = SimpleNamespace(name="publication")
  engine = SimpleNamespace(
    run_once=MagicMock(side_effect=(
      BackfillRunResult(
        publication=None,
        pending_logger_close=True,
      ),
      BackfillRunResult(
        publication=None,
        pending_logger_close=True,
      ),
      BackfillRunResult(
        publication=publication,
        pending_logger_close=False,
      ),
    )),
  )
  car_params = SimpleNamespace(
    carFingerprint="CAR",
    to_bytes=MagicMock(return_value=b"encoded"),
  )
  daemon._wait_for_car_params = MagicMock(return_value=car_params)
  daemon._build_engine = MagicMock(return_value=engine)
  daemon._prepare_remote = MagicMock(return_value=None)
  daemon._project_learning_status = MagicMock()
  sleep_calls = 0

  def advance_logger_or_transition(_seconds: float) -> None:
    nonlocal sleep_calls
    sleep_calls += 1
    if sleep_calls == 1:
      # The pending wait must not invoke run_once/hash old routes again.
      assert engine.run_once.call_count == 1
      lock.unlink()
    elif sleep_calls == 2:
      assert engine.run_once.call_count == 2
    else:
      assert engine.run_once.call_count == 3
      params.values["IsOffroad"] = False

  with patch(
    "openpilot.selfdrive.controls.blatv2_backfilld.time.sleep",
    side_effect=advance_logger_or_transition,
  ):
    daemon.run()

  assert engine.run_once.call_count == 3
  car_params.to_bytes.assert_called_once_with()
  assert daemon._prepare_remote.call_args_list == [
    ((engine, car_params, b"encoded"),),
    ((engine, car_params, b"encoded"),),
    ((engine, car_params, b"encoded"),),
  ]
  assert daemon._project_learning_status.call_args_list == [
    ((engine, None),),
    ((engine, None),),
    ((engine, publication),),
  ]
