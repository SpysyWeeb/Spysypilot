#!/usr/bin/env python3
"""Manager-forked, offroad-only owner for BLaTv2 full-rlog evidence."""

from __future__ import annotations

from collections.abc import Callable
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
  MAXIMUM_EVENT_TRAVERSAL_WORDS,
  BackfillPublication,
  BackfillError,
  BuildDescriptor,
  BuildDescriptorRegistry,
  HistoricalLearningBackfill,
  git_blob_sha1,
  has_pending_full_rlog,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_operation_status import (
  LearningOperationState,
  LearningOperationStatusPublisher,
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
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
  build_runtime_vehicle_bundle,
)


HISTORICAL_BUILD_DESCRIPTORS = (
  Path(__file__).resolve().parent
  / "lib"
  / "blatv2"
  / "historical_build_descriptors.json"
)
NATIVE_EXTRACTOR_PATH = (
  Path(BASEDIR)
  / "openpilot"
  / "selfdrive"
  / "controls"
  / "blatv2_rlog_extract"
)
PENDING_ROUTE_POLL_S = 1.0


def _manager_declares_clean_build() -> bool:
  # manager.py exports CLEAN only after its full build-metadata dirty check.
  return os.environ.get("CLEAN") == "1"


def _decode_text(value: object) -> str | None:
  if type(value) is str:
    return value
  if isinstance(value, (bytes, bytearray, memoryview)):
    return bytes(value).decode("utf-8")
  return None


def _decode_car_params(encoded: bytes) -> car.CarParams:
  with car.CarParams.from_bytes(
    encoded,
    traversal_limit_in_words=MAXIMUM_EVENT_TRAVERSAL_WORDS,
    nesting_limit=64,
  ) as reader:
    return reader.as_builder()


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
    self.operation_status = LearningOperationStatusPublisher(self.params)
    self.backfill_progress = BackfillProgressPublisher(self.params)
    self._stopping = False

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
    limits = current_runtime.runtime_bundle.torque_limits
    rack_resolutions = {
      float(node.parameters.rack_rate_resolution_deg_s)
      for node in current_runtime.runtime_bundle.calibration_seed_profile.nodes
    }
    if len(rack_resolutions) != 1:
      raise BackfillError(
        "backfill_route_incompatible",
        "current runtime has inconsistent rack-rate resolution",
      )
    current_descriptor = BuildDescriptor(
      superproject_commit=root_commit,
      opendbc_commit=opendbc_commit,
      panda_commit=panda_commit,
      log_schema_blob=git_blob_sha1(
        Path(BASEDIR) / "openpilot" / "cereal" / "log.capnp",
      ),
      supported_vehicle_identity=str(car_params.carFingerprint),
      steer_max=limits.steer_max,
      steer_delta_up=limits.delta_up,
      steer_delta_down=limits.delta_down,
      steer_step=limits.steer_step,
      driver_allowance=limits.driver_allowance,
      driver_multiplier=limits.driver_multiplier,
      driver_factor=limits.driver_factor,
      production_envelope_verified=(
        limits.production_envelope_verified
      ),
      rack_rate_resolution_deg_s=rack_resolutions.pop(),
    )
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
    )

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

  def run(self) -> None:
    car_params = None
    try:
      car_params = self._wait_for_car_params()
      if car_params is None:
        return
      engine = self._build_engine(car_params)
      while not self._abort_requested():
        result = engine.run_once()
        self._project_learning_status(engine, result.publication)
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


def main() -> None:
  daemon = BlatV2BackfillDaemon()
  signal.signal(signal.SIGINT, daemon.stop)
  signal.signal(signal.SIGTERM, daemon.stop)
  daemon.run()


if __name__ == "__main__":
  main()
