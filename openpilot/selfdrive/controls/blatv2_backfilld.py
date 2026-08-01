#!/usr/bin/env python3
"""Manager-forked, offroad-only owner for BLaTv2 full-rlog evidence."""

from __future__ import annotations

from collections.abc import Callable
import base64
import hashlib
import json
import os
import signal
from pathlib import Path
import time
from typing import Any

from opendbc.car.structs import car
from opendbc.car.car_helpers import interfaces
from openpilot.common.basedir import BASEDIR
from openpilot.common.git import get_commit
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.blatv2_learnerd import (
  PROVISIONAL_RACK_DYNAMICS_PATH,
  default_learning_storage_root,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  BackfillPublication,
  BackfillError,
  BuildDescriptorRegistry,
  HistoricalLearningBackfill,
  build_current_historical_descriptor,
  has_pending_full_rlog,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_operation_status import (
  LEARNING_OPERATION_STATUS_PARAM,
  LearningOperationState,
  LearningOperationStatusPublisher,
  decode_learning_operation_status,
  route_identity_sha256,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_progress import (
  BackfillProgressPublisher,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  build_persistent_learning_runtime,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_status import (
  LEARNING_STATUS_PARAM,
  build_learning_status_payload,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_learning_status import (
  BehaviorLearningStatusPublisher,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_pipeline import (
  BehaviorPipelineResult,
  OffroadBehaviorLearningPipeline,
)
from openpilot.selfdrive.controls.lib.blatv2.offdevice_backfill import (
  ArchitectureVerificationError,
  BridgeFallbackUnavailableError,
  RemotePreparationSession,
  prepare_remote_session,
  remote_error_fallback_reason,
  remote_error_is_unavailable,
)
from openpilot.selfdrive.controls.lib.blatv2.offdevice_client import (
  OffdeviceBridgeClient,
  default_bridge_config_directory,
  discover_worker,
  load_bridge_secret,
  load_bridge_worker_host,
)
from openpilot.selfdrive.controls.lib.blatv2.offdevice_progress import (
  OFFDEVICE_PROGRESS_PARAM,
  OffdeviceFallbackReason,
  OffdeviceProgressPhase,
  OffdeviceProgressPublisher,
  decode_offdevice_progress,
)
from openpilot.selfdrive.controls.lib.blatv2.offdevice_protocol import (
  BridgeAbortedError,
  BridgeCorruptError,
  BridgeIncompatibleError,
  BridgeRemoteError,
  BridgeUnavailableError,
)
from openpilot.selfdrive.controls.lib.blatv2.preparation_identity import (
  PreparationIdentityError,
  numerical_environment_sha256,
  preparation_implementation_sha256,
)
from openpilot.selfdrive.controls.lib.blatv2.preparation_contract import (
  HISTORICAL_BUILD_DESCRIPTORS,
  NATIVE_EXTRACTOR_PATH,
  PROVISIONAL_RACK_DYNAMICS_PATH,
  decode_car_params as _decode_car_params,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
  build_runtime_vehicle_bundle,
)


PENDING_ROUTE_POLL_S = 1.0
# A decoded CE-sized route can approach one GiB.  The optional PC worker keeps
# four-way throughput, but comma's complete local fallback intentionally owns
# one route at a time so loss of the worker cannot OOM the offroad daemon.
DEVICE_LOCAL_BACKFILL_REPLAY_WORKER_COUNT = 1


class _BestEffortOffdeviceProgress:
  """Keep an informational Params failure outside the learning transaction."""

  def __init__(self, publisher: OffdeviceProgressPublisher) -> None:
    self._publisher = publisher
    self._disabled = False

  @property
  def last_payload(self) -> dict[str, object] | None:
    return None if self._disabled else self._publisher.last_payload

  def clear(self) -> None:
    self._disabled = False
    try:
      self._publisher.clear()
    except Exception:
      self._disabled = True
      cloudlog.exception("blatv2 off-device progress write disabled")

  def publish(self, **fields: object) -> bytes | None:
    if self._disabled:
      return None
    try:
      return self._publisher.publish(**fields)
    except Exception:
      self._disabled = True
      cloudlog.exception("blatv2 off-device progress write disabled")
      return None


def _manager_declares_clean_build() -> bool:
  # manager.py exports CLEAN only after its full build-metadata dirty check.
  return os.environ.get("CLEAN") == "1"


def _decode_text(value: object) -> str | None:
  if type(value) is str:
    return value
  if isinstance(value, (bytes, bytearray, memoryview)):
    return bytes(value).decode("utf-8")
  return None


def _serialize_owned_car_params(car_params: car.CarParams) -> bytes:
  """Serialize the owned builder once before any retryable remote work."""
  return bytes(car_params.to_bytes())


class BlatV2BackfillDaemon:
  def __init__(
    self,
    *,
    params: Any | None = None,
    log_root: str | Path | None = None,
    storage_root: str | Path | None = None,
    extractor_path: str | Path = NATIVE_EXTRACTOR_PATH,
    descriptor_path: str | Path = HISTORICAL_BUILD_DESCRIPTORS,
    current_build_clean: Callable[[], bool] = (
      _manager_declares_clean_build
    ),
    behavior_pipeline_factory: Callable[..., Any] = (
      OffroadBehaviorLearningPipeline
    ),
  ) -> None:
    self.params = Params() if params is None else params
    self.log_root = Path(Paths.log_root() if log_root is None else log_root)
    self.storage_root = (
      default_learning_storage_root(self.params)
      if storage_root is None
      else Path(storage_root)
    )
    self.extractor_path = Path(extractor_path)
    self.descriptor_path = Path(descriptor_path)
    self.current_build_clean = current_build_clean
    self.behavior_pipeline_factory = behavior_pipeline_factory
    self.operation_status = LearningOperationStatusPublisher(self.params)
    self.behavior_status = BehaviorLearningStatusPublisher(self.params)
    self.backfill_progress = BackfillProgressPublisher(self.params)
    self.offdevice_progress = _BestEffortOffdeviceProgress(
      OffdeviceProgressPublisher(self.params),
    )
    self._stopping = False
    self._recover_interrupted_architecture_verification()

  def _recover_interrupted_architecture_verification(self) -> None:
    """Replace a SIGKILL-stale verifying projection with a terminal fact."""
    try:
      raw_progress = self.params.get(OFFDEVICE_PROGRESS_PARAM, block=False)
      raw_operation = self.params.get(
        LEARNING_OPERATION_STATUS_PARAM,
        block=False,
      )
      if raw_progress is None or raw_operation is None:
        return
      try:
        progress = decode_offdevice_progress(raw_progress)
      except ValueError:
        # Schema-v1 verification progress can survive a process-only restart.
        # It is display-only; recognize only its terminally meaningful phase.
        legacy = json.loads(raw_progress)
        if type(legacy) is not dict or legacy.get("phase") != "arm_certifying":
          return
        progress = {"phase": "arm_certifying"}
      operation = decode_learning_operation_status(raw_operation)
      if (
        progress["phase"] != OffdeviceProgressPhase.ARM_CERTIFYING.value
        or operation["terminal"] is True
      ):
        return
      self.operation_status.publish(
        state=LearningOperationState.FAILED,
        diagnostic="architecture_verification_interrupted",
        new_operation=True,
        accepted_sample_count=operation["accepted_sample_count"],
        rejected_sample_count=operation["rejected_sample_count"],
        retry_count=operation["retry_count"],
        runtime_identity_sha256=operation["runtime_identity_sha256"],
        vehicle_identity=operation["vehicle_identity"],
      )
      self.offdevice_progress.clear()
    except Exception:
      # Recovery is display-only and must not prevent the next fresh offroad
      # transaction from reporting its own authoritative failure.
      cloudlog.exception("blatv2 interrupted verification recovery failed")

  def stop(self, *_args: object) -> None:
    self._stopping = True

  def _abort_requested(self) -> bool:
    if self._stopping:
      return True
    try:
      is_offroad = self.params.get_bool("IsOffroad")
    except (
      AttributeError,
      KeyError,
      TypeError,
      ValueError,
      RuntimeError,
      OSError,
    ):
      return True
    return type(is_offroad) is not bool or not is_offroad

  def _read_car_params(self) -> car.CarParams | None:
    # Prefer the CP that produced the just-finished route; persistent CP can
    # lag a software/envelope update and is only a cold-boot fallback.
    for key in ("CarParams", "CarParamsPersistent"):
      try:
        encoded = self.params.get(key, block=False)
        if isinstance(encoded, (bytes, bytearray, memoryview)):
          return _decode_car_params(bytes(encoded))
      except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        OSError,
      ):
        cloudlog.exception(f"blatv2 backfill could not decode {key}")
    return None

  def _wait_for_car_params(self) -> car.CarParams | None:
    preparing_started = False
    while not self._abort_requested():
      car_params = self._read_car_params()
      if car_params is not None:
        # Manager-start deliberately clears every display projection. Publish
        # before runtime construction, route discovery, worker inventory, or
        # uploads can begin so a cold boot never spends that bounded-but-long
        # preflight interval claiming that learner status is unavailable.
        # When CarParams arrived after a wait, advance the operation that
        # already owns ``waiting_for_car_params`` instead of inventing a
        # second operation for the same boot transaction.
        try:
          if self._abort_requested():
            return None
          self.operation_status.publish(
            state=LearningOperationState.PREPARING,
            diagnostic="restoring_runtime",
            new_operation=not preparing_started,
            vehicle_identity=str(car_params.carFingerprint),
          )
        except Exception:
          # Status is informational and must never become a prerequisite for
          # the authoritative replay. The outer failure path will still own
          # any real preflight failure.
          cloudlog.exception("blatv2 backfill preflight status write failed")
        return car_params
      if not preparing_started:
        try:
          if self._abort_requested():
            return None
          self.operation_status.publish(
            state=LearningOperationState.PREPARING,
            diagnostic="waiting_for_car_params",
            new_operation=True,
          )
        except Exception:
          cloudlog.exception("blatv2 backfill status write failed")
        preparing_started = True
      time.sleep(PENDING_ROUTE_POLL_S)
    return None

  def _build_engine(
    self,
    car_params: car.CarParams,
  ) -> HistoricalLearningBackfill:
    if not self.current_build_clean():
      raise BackfillError(
        "backfill_route_incompatible",
        "current build is dirty; historical replay is disabled",
      )
    dynamics = ProvisionalRackDynamics.from_json_file(
      PROVISIONAL_RACK_DYNAMICS_PATH,
    )

    def runtime_factory():
      return build_persistent_learning_runtime(
        car_params=car_params,
        storage_root=self.storage_root,
        provisional_rack_dynamics=dynamics,
      )

    def route_bundle_factory(route_car_params, descriptor):
      fingerprint = str(route_car_params.carFingerprint)
      try:
        interface_class = interfaces[fingerprint]
      except KeyError as exc:
        raise RuntimeError(
          "historical route fingerprint has no current interpretation",
        ) from exc
      car_interface = interface_class(route_car_params)
      return build_runtime_vehicle_bundle(
        car_params=route_car_params,
        car_interface_or_callback=car_interface,
        controller_params=descriptor.controller_params_proxy(),
        vehicle_identity=fingerprint,
        provisional_rack_dynamics=dynamics,
      )

    current_runtime = runtime_factory()
    root_commit = _decode_text(self.params.get("GitCommit", block=False))
    opendbc_commit = get_commit(str(Path(BASEDIR) / "opendbc_repo"))
    panda_commit = get_commit(str(Path(BASEDIR) / "panda"))
    if (
      root_commit is None
      or len(root_commit) != 40
      or len(opendbc_commit) != 40
      or len(panda_commit) != 40
    ):
      raise BackfillError(
        "backfill_route_incompatible",
        "current build provenance is unavailable",
      )
    try:
      current_descriptor = build_current_historical_descriptor(
        source_commit=root_commit,
        opendbc_commit=opendbc_commit,
        panda_commit=panda_commit,
        log_schema_path=Path(BASEDIR) / "openpilot" / "cereal" / "log.capnp",
        current_car_params=car_params,
        current_runtime_bundle=current_runtime.runtime_bundle,
      )
    except ValueError as exc:
      raise BackfillError(
        "backfill_route_incompatible",
        str(exc),
      ) from exc
    registry = BuildDescriptorRegistry.from_json_file(
      self.descriptor_path,
    ).with_descriptor(current_descriptor)
    dongle_id = _decode_text(self.params.get("DongleId", block=False))
    if dongle_id is None or not dongle_id:
      raise BackfillError(
        "backfill_route_incompatible",
        "current dongle identity is unavailable",
      )
    current_route = _decode_text(
      self.params.get("CurrentRoute", block=False),
    )
    pending_route_identity = (
      None
      if current_route is None
      else route_identity_sha256(current_route)
    )
    return HistoricalLearningBackfill(
      log_root=self.log_root,
      extractor_path=self.extractor_path,
      current_car_params=car_params,
      runtime_factory=runtime_factory,
      route_bundle_factory=route_bundle_factory,
      car_params_decoder=_decode_car_params,
      descriptor_registry=registry,
      expected_dongle_id=dongle_id,
      operation_status=self.operation_status,
      backfill_progress=self.backfill_progress,
      abort_requested=self._abort_requested,
      pending_route_identity=pending_route_identity,
      replay_worker_count=DEVICE_LOCAL_BACKFILL_REPLAY_WORKER_COUNT,
    )

  def _remote_contract(
    self,
    engine: HistoricalLearningBackfill,
    car_params: car.CarParams,
    encoded_car_params: bytes,
  ) -> dict[str, object]:
    """Bind one remote preparation job to this exact device runtime."""
    root_commit = _decode_text(self.params.get("GitCommit", block=False))
    opendbc_commit = get_commit(str(Path(BASEDIR) / "opendbc_repo"))
    panda_commit = get_commit(str(Path(BASEDIR) / "panda"))
    if (
      root_commit is None
      or len(root_commit) != 40
      or len(opendbc_commit) != 40
      or len(panda_commit) != 40
    ):
      raise BackfillError(
        "backfill_route_incompatible",
        "remote preparation provenance is unavailable",
      )
    runtime = engine.runtime_factory()
    return {
      "car_params_b64": base64.b64encode(encoded_car_params).decode("ascii"),
      "car_params_sha256": hashlib.sha256(encoded_car_params).hexdigest(),
      "descriptor_registry_sha256": engine.descriptor_registry.identity_sha256,
      "historical_descriptor_registry_sha256": (
        BuildDescriptorRegistry.from_json_file(
          self.descriptor_path,
        ).identity_sha256
      ),
      "dongle_id": engine.expected_dongle_id,
      "opendbc_commit": opendbc_commit,
      "panda_commit": panda_commit,
      "runtime_identity_sha256": (
        runtime.runtime_bundle.calibration_identity_sha256
      ),
      "source_commit": root_commit,
      "vehicle_fingerprint": str(car_params.carFingerprint),
    }

  def _prepare_remote(
    self,
    engine: HistoricalLearningBackfill,
    car_params: car.CarParams,
    encoded_car_params: bytes,
  ) -> RemotePreparationSession | None:
    """Return a PC-prepared session, or None for normal local fallback."""
    try:
      self.offdevice_progress.clear()
    except Exception:
      cloudlog.exception("blatv2 off-device progress cleanup failed")
    contract = self._remote_contract(engine, car_params, encoded_car_params)
    try:
      preparation_identity = preparation_implementation_sha256(
        BASEDIR,
        opendbc_commit=str(contract["opendbc_commit"]),
        panda_commit=str(contract["panda_commit"]),
      )
      device_environment_identity = numerical_environment_sha256()
    except PreparationIdentityError as exc:
      raise BackfillError(
        "backfill_route_incompatible",
        f"route-preparation implementation identity is unavailable: {exc}",
      ) from exc
    try:
      bridge_config = default_bridge_config_directory(self.params)
      secret = load_bridge_secret(bridge_config)
      worker = discover_worker(
        secret=secret,
        client_id=f"comma-{engine.expected_dongle_id}",
        expected_source_commit=str(contract["source_commit"]),
        abort_requested=self._abort_requested,
        configured_host=load_bridge_worker_host(bridge_config),
      )
      client = OffdeviceBridgeClient(
        worker=worker,
        secret=secret,
        client_id=f"comma-{engine.expected_dongle_id}",
        abort_requested=self._abort_requested,
      )
      runtime = engine.runtime_factory()
      session = prepare_remote_session(
        engine=engine,
        client=client,
        contract=contract,
        preparation_implementation_sha256=preparation_identity,
        device_numerical_environment_sha256=device_environment_identity,
        scratch_parent=runtime.artifact_paths.root,
        abort_requested=self._abort_requested,
        offdevice_progress=self.offdevice_progress,
      )
      exclusions = tuple(getattr(session, "unverified_exclusions", ()))
      if exclusions:
        cloudlog.info(
          "blatv2 excluded %d unverified PC-only route rejection(s)",
          len(exclusions),
        )
        for exclusion in exclusions:
          cloudlog.info(
            "blatv2 unverified PC-only exclusion route=%s reason=%s",
            exclusion.route_identity_sha256,
            exclusion.rejection_reason,
          )
      return session
    except BridgeUnavailableError as exc:
      # Absence, a busy worker, or an interrupted connection is not a learner
      # failure. The device immediately retains its original local backend.
      cloudlog.info(f"blatv2 remote worker unavailable; using local replay: {exc}")
      try:
        self.backfill_progress.clear()
      except Exception:
        pass
      self._publish_local_fallback(
        self._bridge_unavailable_reason(exc),
      )
      return None
    except BridgeAbortedError as exc:
      raise BackfillError(
        "unexpected_error",
        "offroad ownership ended during remote preparation",
      ) from exc
    except ArchitectureVerificationError as exc:
      raise BackfillError(
        "architecture_verification_failed",
        f"bounded architecture verification failed: {exc.reason}",
      ) from exc
    except BridgeIncompatibleError as exc:
      raise BackfillError(
        "backfill_route_incompatible",
        f"remote preparation contract is incompatible: {exc}",
      ) from exc
    except BridgeCorruptError as exc:
      raise BackfillError(
        "backfill_corrupt_log",
        f"remote preparation data is corrupt: {exc}",
      ) from exc
    except BridgeRemoteError as exc:
      if remote_error_is_unavailable(exc):
        cloudlog.info(
          f"blatv2 remote worker busy; using local replay: {exc}",
        )
        try:
          self.backfill_progress.clear()
        except Exception:
          pass
        self._publish_local_fallback(
          remote_error_fallback_reason(exc),
        )
        return None
      diagnostic = (
        "backfill_route_incompatible"
        if exc.code in {"source_mismatch", "contract_mismatch"}
        else "backfill_reader_unavailable"
      )
      raise BackfillError(
        diagnostic,
        f"remote preparation worker rejected the job: {exc}",
      ) from exc

  def _bridge_unavailable_reason(
    self,
    error: BridgeUnavailableError,
  ) -> OffdeviceFallbackReason:
    """Classify a fallback by its last completed display-only phase."""
    if isinstance(error, BridgeFallbackUnavailableError):
      return error.fallback_reason
    last = self.offdevice_progress.last_payload
    if last is None:
      return OffdeviceFallbackReason.WORKER_UNAVAILABLE
    phase = OffdeviceProgressPhase(last["phase"])
    if phase is OffdeviceProgressPhase.DOWNLOADING:
      return OffdeviceFallbackReason.NETWORK_INTERRUPTED
    if phase is OffdeviceProgressPhase.ARM_CERTIFYING:
      return OffdeviceFallbackReason.REMOTE_CERTIFICATION_UNAVAILABLE
    return OffdeviceFallbackReason.REMOTE_PREPARATION_UNAVAILABLE

  def _publish_local_fallback(
    self,
    reason: OffdeviceFallbackReason,
  ) -> None:
    """Project fallback without granting the display any learner authority."""
    try:
      last = self.offdevice_progress.last_payload
      certification_fields: dict[str, object] = {}
      if last is not None and last["certified_route_count"] is not None:
        certification_fields = {
          "certified_domain_count": last["certified_domain_count"],
          "certified_route_count": last["certified_route_count"],
          "remote_only_rejection_excluded_count": (
            last["remote_only_rejection_excluded_count"]
          ),
          "total_certification_domain_count": (
            last["total_certification_domain_count"]
          ),
          "total_certification_route_count": (
            last["total_certification_route_count"]
          ),
        }
      self.offdevice_progress.publish(
        phase=OffdeviceProgressPhase.LOCAL_FALLBACK,
        new_session=last is None,
        fallback_reason_code=reason,
        **certification_fields,
      )
    except Exception:
      # This projection is informational. A Params/UI failure must never
      # change whether the authoritative local replay runs.
      cloudlog.exception("blatv2 off-device fallback status write failed")

  def _publish_failure(
    self,
    diagnostic: str,
    car_params: car.CarParams | None,
  ) -> None:
    # A failure discovered after the manager's onroad transition belongs to
    # the cancelled offroad transaction and must not overwrite live status.
    if self._abort_requested():
      return
    try:
      self.backfill_progress.clear()
    except Exception:
      cloudlog.exception("blatv2 backfill progress cleanup failed")
    try:
      self.offdevice_progress.clear()
    except Exception:
      cloudlog.exception("blatv2 off-device progress cleanup failed")
    last = self.operation_status.last_payload
    continuing_operation = last is not None and last["terminal"] is False
    context: dict[str, object] = {
      "accepted_sample_count": (
        last["accepted_sample_count"] if continuing_operation else 0
      ),
      "rejected_sample_count": (
        last["rejected_sample_count"] if continuing_operation else 0
      ),
      "retry_count": (
        last["retry_count"] if continuing_operation else 0
      ),
    }
    if continuing_operation:
      # Preserve same-operation identities and monotonic counters while
      # deliberately dropping current-route progress, which is forbidden on
      # a terminal failure payload.
      context["vehicle_identity"] = last["vehicle_identity"]
      context["runtime_identity_sha256"] = (
        last["runtime_identity_sha256"]
      )
    elif car_params is not None:
      context["vehicle_identity"] = str(car_params.carFingerprint)
    try:
      self.operation_status.publish(
        state=LearningOperationState.FAILED,
        diagnostic=diagnostic,
        new_operation=not continuing_operation,
        **context,
      )
    except Exception:
      cloudlog.exception("blatv2 backfill failure status write failed")

  def _project_learning_status(
    self,
    engine: HistoricalLearningBackfill,
    publication: BackfillPublication | None,
  ) -> None:
    try:
      if self._abort_requested():
        return
      runtime = engine.runtime_factory()
      if publication is None:
        # A no-op cold-boot scan still owns offroad display projection. Only
        # an authenticated CURRENT generation is eligible for reconstruction.
        if not runtime.artifact_paths.backfill_pointer.is_file():
          return
        finalization = runtime.coordinator.finalize()
      else:
        finalization = publication.finalization
      payload = build_learning_status_payload(
        finalization=finalization,
        runtime_bundle=runtime.runtime_bundle,
        drive_baseline=None,
      )
      if self._abort_requested():
        return
      self.params.put(LEARNING_STATUS_PARAM, payload, block=True)
    except Exception:
      # Display projection is non-authoritative; CURRENT is already a complete
      # authenticated generation (or restore failed closed above).
      cloudlog.exception(
        "blatv2 backfill learning display projection failed",
      )

  def _offroad_confirmed(self) -> bool:
    """Independent publication guard for informational behavior artifacts."""
    if self._stopping:
      return False
    try:
      value = self.params.get_bool("IsOffroad")
    except (
      AttributeError,
      KeyError,
      TypeError,
      ValueError,
      RuntimeError,
      OSError,
    ):
      return False
    return type(value) is bool and value

  def _run_behavior_learning(
    self,
    engine: HistoricalLearningBackfill,
  ) -> BehaviorPipelineResult | None:
    """Run the independent behavior stage without reclassifying calibration.

    Physical evidence has already committed before this method is reachable.
    A behavior failure therefore retains stock and is reported through its own
    Params status; it must not rewrite the successful physical operation as a
    backfill failure.
    """
    if self._abort_requested():
      return None
    try:
      source_commit = _decode_text(self.params.get("GitCommit", block=False))
      opendbc_commit = get_commit(str(Path(BASEDIR) / "opendbc_repo"))
      panda_commit = get_commit(str(Path(BASEDIR) / "panda"))
      if (
        source_commit is None
        or len(source_commit) != 40
        or len(opendbc_commit) != 40
        or len(panda_commit) != 40
      ):
        raise ValueError("behavior replay build provenance is unavailable")
      dynamics = ProvisionalRackDynamics.from_json_file(
        PROVISIONAL_RACK_DYNAMICS_PATH,
      )
      pipeline = self.behavior_pipeline_factory(
        params=self.params,
        status_publisher=self.behavior_status,
        provisional_dynamics=dynamics,
        source_openpilot_commit=source_commit,
        opendbc_commit=opendbc_commit,
        panda_commit=panda_commit,
        abort_requested=self._abort_requested,
        offroad_confirmed=self._offroad_confirmed,
        logger=cloudlog,
      )
      return pipeline.run(engine.runtime_factory())
    except Exception:
      # The pipeline itself projects a fail-closed status for expected
      # transaction failures. This catch protects physical evidence from
      # construction/provenance errors before the pipeline can own a status.
      if not self._abort_requested():
        cloudlog.exception("blatv2 behavior learning failed unexpectedly")
      return None

  def run(self) -> None:
    car_params = None
    try:
      car_params = self._wait_for_car_params()
      if car_params is None:
        return
      local_engine = self._build_engine(car_params)
      # ``_decode_car_params`` returns an owned builder. Pycapnp builders are
      # write-once unless their internal flag is reset, while pending-logger
      # handling can retry remote preparation. Freeze the wire bytes exactly
      # once and reuse them across every retry.
      encoded_car_params = _serialize_owned_car_params(car_params)
      while not self._abort_requested():
        session = self._prepare_remote(
          local_engine,
          car_params,
          encoded_car_params,
        )
        remote_engine: HistoricalLearningBackfill | None = None
        try:
          if session is not None:
            remote_engine = session.build_engine()
          engine = local_engine if remote_engine is None else remote_engine
          result = engine.run_once()
          self._project_learning_status(engine, result.publication)
          if not result.pending_logger_close:
            try:
              self.offdevice_progress.clear()
            except Exception:
              cloudlog.exception("blatv2 off-device progress cleanup failed")
        finally:
          if session is not None:
            try:
              if remote_engine is not None:
                session.preserve_transaction_state(remote_engine)
            finally:
              # Downloaded spools are private transaction scratch. They must
              # not survive a constructor, replay, projection, or preserve
              # failure, and are never inputs to a later daemon operation.
              session.close()
        if not result.pending_logger_close:
          # Close remote preparation/network scratch before behavior replay
          # forks. Behavioral inputs come only from the durable A/A route
          # store and the authenticated physical CURRENT generation.
          self._run_behavior_learning(local_engine)
        if not result.pending_logger_close:
          # Stay healthy under manager's offroad predicate without rescanning.
          # A new drive requires an onroad transition, which aborts this
          # interruptible idle wait and lets manager start a fresh process on
          # the next offroad transition.
          while not self._abort_requested():
            time.sleep(PENDING_ROUTE_POLL_S)
          return
        # Keep the FINALIZING operation stable while loggerd owns a lock or
        # while the engine requests its post-unlock quiescence interval.
        # Replaying run_once inside the wait would rehash prior routes.
        while not self._abort_requested():
          # A locked route first waits until its marker vanishes. The next
          # engine scan discovers that candidate unlocked and returns pending
          # once more, so re-entering this sleep establishes one complete poll
          # interval after the first unlocked discovery.
          time.sleep(PENDING_ROUTE_POLL_S)
          if (
            self._abort_requested()
            or not has_pending_full_rlog(
              self.log_root,
              abort_requested=self._abort_requested,
            )
          ):
            break
    except BackfillError as exc:
      if not self._abort_requested():
        self._publish_failure(exc.diagnostic, car_params)
        cloudlog.exception(f"blatv2 backfill failed: {exc}")
    except Exception:
      if not self._abort_requested():
        self._publish_failure("unexpected_error", car_params)
        cloudlog.exception("blatv2 backfill failed unexpectedly")
    finally:
      # Progress belongs only to this offroad process. Clear it even when an
      # onroad handoff or SIGTERM deliberately suppresses operation-status
      # writes, so stale route coordinates cannot survive into a drive.
      try:
        self.backfill_progress.clear()
      except Exception:
        cloudlog.exception("blatv2 backfill progress cleanup failed")
      try:
        self.offdevice_progress.clear()
      except Exception:
        cloudlog.exception("blatv2 off-device progress cleanup failed")


def main() -> None:
  daemon = BlatV2BackfillDaemon()
  signal.signal(signal.SIGINT, daemon.stop)
  signal.signal(signal.SIGTERM, daemon.stop)
  daemon.run()


if __name__ == "__main__":
  main()
