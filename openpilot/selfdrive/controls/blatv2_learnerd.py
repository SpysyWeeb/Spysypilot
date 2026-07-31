#!/usr/bin/env python3
"""Measured-only learning adapter retained for offline/harness use.

The stock-only field manager does not launch this module. Its finite-poll
adapter remains available to deterministic tests and offline harnesses. It
publishes nothing and subscribes to no model intent. The sole ``carControl``
field it reads is the canonical ``latActive`` validity witness;
requested/candidate torque cannot enter the learning sample.

Production durable evidence is owned exclusively by ``blatv2_backfilld``,
which replays complete closed full rlogs offroad. This process may restore
that evidence to compute an onroad preview, but discards the preview at the
onroad-to-offroad boundary and never commits it.

The shared storage root convention is a dedicated sibling of the active
Params data directory (normally ``/data/params/blatv2-learning``) and is
injectable for tests.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from opendbc.car.structs import car
from openpilot.cereal.services import SERVICE_LIST
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  MeasuredLearningFrame,
  PersistentLearningRuntime,
  build_persistent_learning_runtime,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_operation_status import (
  LearningOperationState,
  LearningOperationStatusPublisher,
  route_identity_sha256,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_frame import (
  CanonicalSourceHistory as _CanonicalSourceHistory,
  CanonicalSourceSnapshot as _SourceSnapshot,
  maximum_source_age_ns as _maximum_source_age_ns,
  measured_learning_frame,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_status import (
  LEARNING_STATUS_PARAM,
  DriveEvidenceBaseline,
  build_learning_status_payload,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
)


PUBLISHED_SERVICES: tuple[str, ...] = ()
SUBSCRIBED_SERVICES = (
  "deviceState",
  "controlsState",
  "carControl",
  "carState",
  "carOutput",
  "liveParameters",
)
MEASUREMENT_SERVICES = (
  "carControl",
  "carState",
  "carOutput",
  "liveParameters",
)
PROVISIONAL_RACK_DYNAMICS_PATH = (
  Path(__file__).resolve().parent
  / "lib"
  / "blatv2"
  / "provisional_rack_dynamics.json"
)
STEP_TIMEOUT_MS = 100
OPERATION_STATUS_REFRESH_WITNESSES = int(
  round(2.0 * SERVICE_LIST["controlsState"].frequency),
)


def assert_no_actuation_publishers(
  services: tuple[str, ...] = PUBLISHED_SERVICES,
) -> None:
  assert services == ()
  assert "carControl" not in services


def default_learning_storage_root(params: Any) -> Path:
  """Return a reversible convention-relative storage root without creating it."""
  active_params_directory = Path(params.get_param_path())
  return active_params_directory.parent / "blatv2-learning"


class BlatV2LearnerDaemon:
  """Finite-poll message adapter around ``PersistentLearningRuntime``."""

  def __init__(
    self,
    *,
    sm: Any | None = None,
    params: Any | None = None,
    storage_root: str | Path | None = None,
    car_params_decoder: Callable[[bytes], car.CarParams] | None = None,
    runtime_factory: Callable[
      [car.CarParams, Path],
      PersistentLearningRuntime,
    ] | None = None,
    logger: Any | None = None,
    route_owned_persistence: bool = True,
  ) -> None:
    assert_no_actuation_publishers()
    if sm is None or params is None or car_params_decoder is None:
      import openpilot.cereal.messaging as messaging
      from openpilot.common.params import Params

      if sm is None:
        # With no single poll socket, update() wakes for deviceState offroad
        # and for the 100 Hz controls witness onroad.
        sm = messaging.SubMaster(list(SUBSCRIBED_SERVICES))
      if params is None:
        params = Params()
      if car_params_decoder is None:
        def decode_car_params(encoded):
          return messaging.log_from_bytes(encoded, car.CarParams)
        car_params_decoder = decode_car_params
    if logger is None:
      from openpilot.common.swaglog import cloudlog

      logger = cloudlog

    self.sm = sm
    self.params = params
    self.storage_root = (
      Path(storage_root)
      if storage_root is not None
      else default_learning_storage_root(params)
    )
    self.car_params_decoder = car_params_decoder
    self.runtime_factory = (
      runtime_factory
      if runtime_factory is not None
      else self._production_runtime_factory
    )
    self.logger = logger
    self.route_owned_persistence = bool(route_owned_persistence)
    self.operation_status = LearningOperationStatusPublisher(self.params)
    self.runtime: PersistentLearningRuntime | None = None
    self._runtime_fingerprint: str | None = None
    self._prepared_car_params_bytes: bytes | None = None
    self._prepared_artifact_revision: bytes | None = None
    self._onroad: bool | None = None
    self._runtime_unavailable_this_drive = False
    self._suppress_offroad_operation_status = self.route_owned_persistence
    self._live_identity_bound = False
    self._offroad_restore_failed_identity: tuple[str, bytes] | None = None
    self._pending_offroad_persist = False
    self._learning_status_clear_pending = False
    self._drive_baseline: DriveEvidenceBaseline | None = None
    self._current_route_identity: str | None = None
    self._persist_retry_count = 0
    self._histories = {
      service: _CanonicalSourceHistory()
      for service in MEASUREMENT_SERVICES
    }
    self.controls_witness_count = 0
    self.unresolved_witness_count = 0
    self.accepted_sample_count = 0
    self._last_collecting_status_witness_count = 0
    if not self.route_owned_persistence:
      self._publish_operation_status(
        state=LearningOperationState.PREPARING,
        diagnostic="waiting_for_car_params",
        new_operation=True,
      )

  @staticmethod
  def _production_runtime_factory(
    car_params: car.CarParams,
    storage_root: Path,
  ) -> PersistentLearningRuntime:
    dynamics = ProvisionalRackDynamics.from_json_file(
      PROVISIONAL_RACK_DYNAMICS_PATH,
    )
    return build_persistent_learning_runtime(
      car_params=car_params,
      storage_root=storage_root,
      provisional_rack_dynamics=dynamics,
    )

  def _log_exception(self, message: str) -> None:
    exception = getattr(self.logger, "exception", None)
    if callable(exception):
      exception(message)

  def _operation_runtime_context(self) -> dict[str, object]:
    if self.runtime is None:
      return {}
    return {
      "vehicle_identity": self.runtime.runtime_bundle.vehicle_identity,
      "runtime_identity_sha256": (
        self.runtime.runtime_bundle.identity_sha256
      ),
    }

  def _publish_operation_status(
    self,
    *,
    state: LearningOperationState,
    diagnostic: str,
    new_operation: bool = False,
    finalization: Any | None = None,
    **context: object,
  ) -> None:
    """Best-effort UI projection that cannot affect learning authority."""
    if self.route_owned_persistence:
      # Historical full-rlog replay is the sole offroad status owner. Keep
      # both observations at the write boundary: deviceState can lag the
      # manager's IsOffroad transition while learnerd is being stopped.
      if self._onroad is not True:
        return
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
        return
      if type(is_offroad) is not bool or is_offroad:
        return
    fields: dict[str, object] = self._operation_runtime_context()
    fields.update({
      "accepted_sample_count": self.accepted_sample_count,
      "rejected_sample_count": max(
        0,
        self.controls_witness_count - self.accepted_sample_count,
      ),
      "retry_count": self._persist_retry_count,
    })
    if finalization is not None:
      fields["evidence_sha256"] = finalization.evidence_sha256
    fields.update(context)
    try:
      self.operation_status.publish(
        state=state,
        diagnostic=diagnostic,
        new_operation=new_operation,
        **fields,
      )
    except (KeyError, TypeError, ValueError, RuntimeError, OSError):
      self._log_exception("blatv2 learner operation status write failed")

  def _capture_current_route_identity(self) -> None:
    if self._onroad is not True or self._current_route_identity is not None:
      return
    try:
      encoded = self.params.get("CurrentRoute", block=False)
      if encoded is None:
        return
      if type(encoded) is str:
        route_name = encoded
      elif isinstance(encoded, (bytes, bytearray, memoryview)):
        route_name = bytes(encoded).decode("utf-8")
      else:
        raise TypeError("CurrentRoute must be text or UTF-8 bytes")
      self._current_route_identity = route_identity_sha256(route_name)
      if self._live_identity_bound and self.runtime is not None:
        self._publish_operation_status(
          state=LearningOperationState.COLLECTING,
          diagnostic="collecting_current_drive",
          current_route_identity=self._current_route_identity,
        )
    except (
      AttributeError,
      KeyError,
      TypeError,
      UnicodeDecodeError,
      ValueError,
      RuntimeError,
      OSError,
    ):
      self._log_exception("blatv2 learner could not identify CurrentRoute")

  def _read_offroad_car_params(
    self,
  ) -> tuple[car.CarParams, bytes] | None:
    # The just-finished route's manager-scoped CP is authoritative after a
    # software/envelope change. Persistent CP is only the cold-boot fallback.
    for key in ("CarParams", "CarParamsPersistent"):
      try:
        encoded = self.params.get(key, block=False)
        if encoded is not None:
          canonical = bytes(encoded)
          return self.car_params_decoder(canonical), canonical
      except Exception:
        self._log_exception(f"blatv2 learner could not decode {key}")
    return None

  @staticmethod
  def _artifact_revision(
    runtime: PersistentLearningRuntime,
  ) -> bytes | None:
    try:
      pointer = runtime.artifact_paths.backfill_pointer
      return pointer.read_bytes() if pointer.is_file() else None
    except OSError:
      return None

  def _read_live_car_params(self) -> tuple[car.CarParams, bytes] | None:
    try:
      encoded = self.params.get("CarParams", block=False)
      if encoded is None:
        return None
      canonical = bytes(encoded)
      return self.car_params_decoder(canonical), canonical
    except Exception:
      self._log_exception("blatv2 learner could not decode live CarParams")
      return None

  def _clear_learning_status_if_pending(self) -> None:
    if self._onroad is not False or not self._learning_status_clear_pending:
      return
    try:
      self.params.remove(LEARNING_STATUS_PARAM)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError):
      self._log_exception("blatv2 learner could not clear stale status")
      return
    self._learning_status_clear_pending = False

  def _prepare_offroad_runtime(self) -> None:
    """Restore/qualify persistent state only while the device is offroad."""
    if self._onroad is not False:
      return
    resolved = self._read_offroad_car_params()
    if resolved is None:
      return
    car_params, encoded_car_params = resolved
    fingerprint = str(car_params.carFingerprint)
    if (
      self.runtime is not None
      and self._runtime_fingerprint == fingerprint
      and self._prepared_car_params_bytes == encoded_car_params
      and self._prepared_artifact_revision == self._artifact_revision(
        self.runtime,
      )
    ):
      return
    if self.runtime is not None:
      # A definitive exact identity change invalidates the previous runtime
      # and its display cache before any replacement is attempted.
      self.runtime = None
      self._runtime_fingerprint = None
      self._prepared_car_params_bytes = None
      self._prepared_artifact_revision = None
      self._learning_status_clear_pending = True
      self._clear_learning_status_if_pending()
    restore_identity = (fingerprint, encoded_car_params)
    if self._offroad_restore_failed_identity == restore_identity:
      return
    if (
      not self._runtime_unavailable_this_drive
      and not self._suppress_offroad_operation_status
    ):
      last_operation = self.operation_status.last_payload
      self._publish_operation_status(
        state=LearningOperationState.PREPARING,
        diagnostic="restoring_runtime",
        new_operation=bool(
          last_operation is not None
          and last_operation["terminal"] is True
        ),
        vehicle_identity=fingerprint,
      )
    try:
      runtime = self.runtime_factory(car_params, self.storage_root)
    except Exception:
      # Corrupt artifacts are never replaced with an empty learner.
      self.runtime = None
      self._runtime_fingerprint = None
      self._prepared_car_params_bytes = None
      self._prepared_artifact_revision = None
      self._offroad_restore_failed_identity = restore_identity
      # This concrete vehicle's artifacts failed authentication. A display
      # snapshot left by a process-only restart is therefore stale and must
      # disappear rather than masquerade as current learning evidence.
      self._learning_status_clear_pending = True
      self._clear_learning_status_if_pending()
      if (
        not self._runtime_unavailable_this_drive
        and not self._suppress_offroad_operation_status
      ):
        self._publish_operation_status(
          state=LearningOperationState.FAILED,
          diagnostic="runtime_restore_failed",
          vehicle_identity=fingerprint,
        )
      self._log_exception("blatv2 learner runtime restore failed closed")
      return
    self.runtime = runtime
    self._runtime_fingerprint = fingerprint
    self._prepared_car_params_bytes = encoded_car_params
    self._prepared_artifact_revision = self._artifact_revision(runtime)
    self._offroad_restore_failed_identity = None
    if self._suppress_offroad_operation_status:
      return
    # Manager-start clears the display cache. Repopulate it only from an
    # exact artifact set that the runtime just restored and authenticated.
    # A first-ever empty learner has no persisted authority to project yet.
    if (
      runtime.artifact_paths.evidence.is_file()
      and runtime.artifact_paths.manifest.is_file()
    ):
      try:
        finalization = runtime.coordinator.finalize()
        self._write_learning_status(
          finalization,
          drive_baseline=None,
        )
        if not self._runtime_unavailable_this_drive:
          self._publish_operation_status(
            state=LearningOperationState.IDLE,
            diagnostic="evidence_ready",
            finalization=finalization,
          )
      except Exception:
        self._log_exception(
          "blatv2 learner status restore projection failed",
        )
    else:
      # A successfully authenticated empty owner has no learning state to
      # display. Remove any process-restart cache left by an older owner.
      self._learning_status_clear_pending = True
      self._clear_learning_status_if_pending()
      if (
        not self._runtime_unavailable_this_drive
        and not self._suppress_offroad_operation_status
      ):
        self._publish_operation_status(
          state=LearningOperationState.READY_NO_EVIDENCE,
          diagnostic="ready_for_first_drive",
        )

  def _start_prepared_onroad_runtime(self) -> None:
    """Activate only a bundle already restored during an offroad period."""
    if self._runtime_unavailable_this_drive:
      return
    resolved = self._read_live_car_params()
    if resolved is None:
      # CarParams may appear shortly after deviceState.started. Waiting is not
      # an incompatibility and cannot admit an unbound measurement frame.
      return
    car_params, encoded_car_params = resolved
    fingerprint = str(car_params.carFingerprint)
    if (
      self.runtime is None
      or self._runtime_fingerprint != fingerprint
      or self._prepared_car_params_bytes is None
      or encoded_car_params != self._prepared_car_params_bytes
    ):
      # A concrete live identity mismatch invalidates this drive's collection,
      # but does not mutate the prepared offroad evidence.
      self._runtime_unavailable_this_drive = True
      self._live_identity_bound = False
      mismatch_context: dict[str, object] = {}
      if self.runtime is not None:
        mismatch_context = self._operation_runtime_context()
      else:
        mismatch_context["vehicle_identity"] = fingerprint
      self._publish_operation_status(
        state=(
          LearningOperationState.DRIVE_SKIPPED_IDENTITY_MISMATCH
        ),
        diagnostic="car_params_identity_mismatch",
        last_route_identity=self._current_route_identity,
        **mismatch_context,
      )
      return
    if self._live_identity_bound:
      return
    if self.runtime.coordinator.state.value == "onroad":
      raise RuntimeError("onroad learner runtime lacks a live identity binding")
    self._drive_baseline = DriveEvidenceBaseline.from_support_diagnostics(
      self.runtime.coordinator.support_diagnostics,
    )
    self.runtime.transition_onroad()
    self._live_identity_bound = True
    self._publish_operation_status(
      state=LearningOperationState.COLLECTING,
      diagnostic="collecting_current_drive",
      current_route_identity=self._current_route_identity,
    )

  def _transition(self, started: bool) -> None:
    if self._onroad is not None and bool(started) == self._onroad:
      return
    was_onroad = self._onroad is True
    was_offroad = self._onroad is False
    self._onroad = bool(started)
    if self._onroad:
      self._runtime_unavailable_this_drive = False
      self._live_identity_bound = False
      self._pending_offroad_persist = False
      self._drive_baseline = None
      self._current_route_identity = None
      self._persist_retry_count = 0
      self._last_collecting_status_witness_count = 0
      if was_offroad:
        self.controls_witness_count = 0
        self.unresolved_witness_count = 0
        self.accepted_sample_count = 0
      self._publish_operation_status(
        state=LearningOperationState.PREPARING,
        diagnostic="waiting_for_car_params",
        new_operation=True,
      )
      self._capture_current_route_identity()
      self._start_prepared_onroad_runtime()
    elif (
      was_onroad
      and self.runtime is not None
      and self.runtime.coordinator.state.value == "onroad"
    ):
      if self.route_owned_persistence:
        # Full-rlog replay is the sole durable owner from first boot onward.
        # Live collection remains a useful preview but cannot race or
        # duplicate the same route in canonical evidence.
        self.runtime.transition_offroad_without_persist()
        self._drive_baseline = None
        self._live_identity_bound = False
        # Discard the preview coordinator immediately. The same offroad step
        # restores a fresh immutable committed generation, so no unledgered
        # sample can leak into a later drive while logger completion is
        # delayed.
        self.runtime = None
        self._runtime_fingerprint = None
        self._prepared_car_params_bytes = None
        self._prepared_artifact_revision = None
        self._suppress_offroad_operation_status = True
        return
      try:
        self._publish_operation_status(
          state=LearningOperationState.FINALIZING,
          diagnostic="finalizing_drive",
          last_route_identity=self._current_route_identity,
        )
        finalization = self.runtime.transition_offroad_and_persist()
        # Params is the last atomic display operation after evidence,
        # optional candidate, and manifest persistence have all succeeded.
        self._write_learning_status(
          finalization,
          drive_baseline=self._drive_baseline,
        )
        self._pending_offroad_persist = False
        self._drive_baseline = None
        self._publish_operation_status(
          state=LearningOperationState.IDLE,
          diagnostic="evidence_ready",
          finalization=finalization,
          last_route_identity=self._current_route_identity,
        )
      except Exception:
        self._pending_offroad_persist = True
        self._persist_retry_count += 1
        self._publish_operation_status(
          state=LearningOperationState.RETRY_PENDING,
          diagnostic="persist_retry_pending",
          last_route_identity=self._current_route_identity,
        )
        self._log_exception("blatv2 learner offroad persist failed")
      finally:
        self._live_identity_bound = False

  def _write_learning_status(
    self,
    finalization,
    *,
    drive_baseline: DriveEvidenceBaseline | None,
  ) -> None:
    if self._onroad is not False or self.runtime is None:
      raise RuntimeError("learning display status may be written only offroad")
    payload = build_learning_status_payload(
      finalization=finalization,
      runtime_bundle=self.runtime.runtime_bundle,
      drive_baseline=drive_baseline,
    )
    self.params.put(LEARNING_STATUS_PARAM, payload, block=True)
    self._learning_status_clear_pending = False

  def _retry_pending_persist(self) -> None:
    if (
      self._onroad is not False
      or not self._pending_offroad_persist
      or self.runtime is None
    ):
      return
    try:
      finalization = self.runtime.persist_offroad()
      self._write_learning_status(
        finalization,
        drive_baseline=self._drive_baseline,
      )
      self._pending_offroad_persist = False
      self._drive_baseline = None
      self._publish_operation_status(
        state=LearningOperationState.IDLE,
        diagnostic="evidence_ready",
        finalization=finalization,
        last_route_identity=self._current_route_identity,
      )
    except Exception:
      self._persist_retry_count += 1
      self._publish_operation_status(
        state=LearningOperationState.RETRY_PENDING,
        diagnostic="persist_retry_pending",
        last_route_identity=self._current_route_identity,
      )
      self._log_exception("blatv2 learner offroad persist retry failed")

  def _capture_sources(self) -> None:
    for service, history in self._histories.items():
      if self.sm.updated[service]:
        history.update(
          message=self.sm[service],
          mono_ns=int(self.sm.logMonoTime[service]),
          valid=bool(self.sm.valid[service]),
          alive=bool(self.sm.alive[service]),
        )

  def _select_sources(
    self,
    witness_mono_ns: int,
  ) -> dict[str, _SourceSnapshot] | None:
    selected: dict[str, _SourceSnapshot] = {}
    for service, history in self._histories.items():
      snapshot = history.select(
        witness_mono_ns=witness_mono_ns,
        maximum_age_ns=_maximum_source_age_ns(service),
      )
      if snapshot is None:
        return None
      selected[service] = snapshot
    return selected

  @staticmethod
  def _measured_frame(
    witness_mono_ns: int,
    sources: dict[str, _SourceSnapshot],
  ) -> MeasuredLearningFrame:
    car_state = sources["carState"].message
    car_control = sources["carControl"].message
    car_output = sources["carOutput"].message
    live_parameters = sources["liveParameters"].message
    return measured_learning_frame(
      witness_mono_ns=witness_mono_ns,
      car_state=car_state,
      car_control=car_control,
      car_output=car_output,
      live_parameters=live_parameters,
    )

  def _process_controls_witness(self) -> None:
    self.controls_witness_count += 1
    witness_mono_ns = int(self.sm.logMonoTime["controlsState"])
    controls_valid = (
      witness_mono_ns > 0
      and bool(self.sm.valid["controlsState"])
      and bool(self.sm.alive["controlsState"])
    )
    sources = (
      self._select_sources(witness_mono_ns)
      if controls_valid
      else None
    )
    if (
      sources is None
      or self.runtime is None
      or not self._live_identity_bound
      or self._runtime_unavailable_this_drive
      or self.runtime.coordinator.state.value != "onroad"
    ):
      self.unresolved_witness_count += 1
      self._refresh_collecting_operation_status()
      return
    try:
      frame = self._measured_frame(witness_mono_ns, sources)
      accepted = self.runtime.ingest(frame)
    except (AttributeError, TypeError, ValueError, OverflowError):
      self.unresolved_witness_count += 1
      self._refresh_collecting_operation_status()
      return
    if accepted:
      self.accepted_sample_count += 1
    self._refresh_collecting_operation_status()

  def _refresh_collecting_operation_status(self) -> None:
    if (
      self._live_identity_bound
      and (
        self.controls_witness_count
        - self._last_collecting_status_witness_count
      ) >= OPERATION_STATUS_REFRESH_WITNESSES
    ):
      self._publish_operation_status(
        state=LearningOperationState.COLLECTING,
        diagnostic="collecting_current_drive",
        current_route_identity=self._current_route_identity,
      )
      self._last_collecting_status_witness_count = (
        self.controls_witness_count
      )

  def step(self, timeout_ms: int = STEP_TIMEOUT_MS) -> None:
    self.sm.update(timeout_ms)
    self._capture_sources()
    self._capture_current_route_identity()

    if (
      self.sm.seen["deviceState"]
      and self.sm.valid["deviceState"]
      and self.sm.alive["deviceState"]
    ):
      self._transition(bool(self.sm["deviceState"].started))
    if self._onroad is False:
      if self.sm.updated["deviceState"]:
        self._clear_learning_status_if_pending()
        self._retry_pending_persist()
      self._prepare_offroad_runtime()
      self._clear_learning_status_if_pending()
    if self._onroad is True:
      self._start_prepared_onroad_runtime()
      if self.sm.updated["controlsState"]:
        self._process_controls_witness()

  def run(self) -> None:
    while True:
      self.step()


def main() -> None:
  BlatV2LearnerDaemon().run()


if __name__ == "__main__":
  main()
