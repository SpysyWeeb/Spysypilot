from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.blatv2_backfilld import (
  BlatV2BackfillDaemon,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  BackfillError,
  BackfillRunResult,
  discover_complete_route_candidates,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_operation_status import (
  LEARNING_OPERATION_STATUS_PARAM,
  LearningOperationState,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_progress import (
  BACKFILL_PROGRESS_PARAM,
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
  daemon._wait_for_car_params = MagicMock(return_value=object())
  daemon._build_engine = MagicMock(return_value=engine)
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
  daemon._wait_for_car_params = MagicMock(return_value=object())
  daemon._build_engine = MagicMock(return_value=engine)
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
  assert daemon._project_learning_status.call_args_list == [
    ((engine, None),),
    ((engine, None),),
    ((engine, publication),),
  ]
