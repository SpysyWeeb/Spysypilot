#!/usr/bin/env python3
"""Always-running, measured-only persistence process for modular BLaTv2.

The process polls every subscribed socket with a finite timeout, including
``deviceState``.  It therefore observes the onroad-to-offroad transition even
after all onroad-only publishers stop.  It publishes nothing and subscribes to
no model intent. The sole ``carControl`` field it reads is the canonical
``latActive`` validity witness; requested/candidate torque cannot enter the
learning sample.

Production storage is a dedicated sibling of the active Params data
directory (normally ``/data/params/blatv2-learning``).  The root is injectable
and is not created until an offroad persist operation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from opendbc.car.structs import car
from openpilot.cereal.services import SERVICE_LIST
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  MeasuredLearningFrame,
  PersistentLearningRuntime,
  build_persistent_learning_runtime,
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
# Each source may be at most one-and-a-half of its declared publication
# periods older than the controlsState witness. This is a data-alignment
# bound, not a controller or feel parameter.
MAX_SOURCE_AGE_PERIODS = 1.5
STEP_TIMEOUT_MS = 100


def assert_no_actuation_publishers(
  services: tuple[str, ...] = PUBLISHED_SERVICES,
) -> None:
  assert services == ()
  assert "carControl" not in services


def default_learning_storage_root(params: Any) -> Path:
  """Return a reversible convention-relative storage root without creating it."""
  active_params_directory = Path(params.get_param_path())
  return active_params_directory.parent / "blatv2-learning"


def _copy_message(message: Any) -> Any:
  builder = getattr(message, "as_builder", None)
  return builder() if callable(builder) else message


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
  message: Any
  mono_ns: int
  valid: bool
  alive: bool


class _CanonicalSourceHistory:
  """Retain the two newest snapshots and select one preceding a witness."""

  __slots__ = ("_current", "_previous")

  def __init__(self) -> None:
    self._current: _SourceSnapshot | None = None
    self._previous: _SourceSnapshot | None = None

  def update(
    self,
    *,
    message: Any,
    mono_ns: int,
    valid: bool,
    alive: bool,
  ) -> None:
    timestamp = int(mono_ns)
    if timestamp <= 0:
      return
    snapshot = _SourceSnapshot(
      message=_copy_message(message),
      mono_ns=timestamp,
      valid=bool(valid),
      alive=bool(alive),
    )
    if self._current is None or timestamp > self._current.mono_ns:
      self._previous = self._current
      self._current = snapshot
    elif timestamp == self._current.mono_ns:
      self._current = snapshot
    elif (
      self._previous is None
      or timestamp >= self._previous.mono_ns
    ):
      self._previous = snapshot

  def select(
    self,
    *,
    witness_mono_ns: int,
    maximum_age_ns: int,
  ) -> _SourceSnapshot | None:
    witness = int(witness_mono_ns)
    maximum_age = int(maximum_age_ns)
    if witness <= 0 or maximum_age < 0:
      return None
    eligible = tuple(
      snapshot
      for snapshot in (self._current, self._previous)
      if (
        snapshot is not None
        and snapshot.mono_ns <= witness
        and witness - snapshot.mono_ns <= maximum_age
      )
    )
    if not eligible:
      return None
    selected = max(eligible, key=lambda snapshot: snapshot.mono_ns)
    if not selected.valid or not selected.alive:
      return None
    return selected


def _maximum_source_age_ns(service: str) -> int:
  frequency = float(SERVICE_LIST[service].frequency)
  if not math.isfinite(frequency) or frequency <= 0.0:
    raise ValueError("learning source must have a positive declared frequency")
  return int(round(MAX_SOURCE_AGE_PERIODS * 1e9 / frequency))


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
    self.runtime: PersistentLearningRuntime | None = None
    self._runtime_fingerprint: str | None = None
    self._onroad: bool | None = None
    self._runtime_unavailable_this_drive = False
    self._offroad_restore_failed_fingerprint: str | None = None
    self._pending_offroad_persist = False
    self._histories = {
      service: _CanonicalSourceHistory()
      for service in MEASUREMENT_SERVICES
    }
    self.controls_witness_count = 0
    self.unresolved_witness_count = 0
    self.accepted_sample_count = 0

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

  def _read_car_params(self) -> car.CarParams | None:
    try:
      encoded = self.params.get("CarParams", block=False)
      if encoded is None:
        return None
      return self.car_params_decoder(encoded)
    except Exception:
      self._log_exception("blatv2 learner could not decode CarParams")
      return None

  def _prepare_offroad_runtime(self) -> None:
    """Restore/qualify persistent state only while the device is offroad."""
    if self._onroad is not False:
      return
    car_params = self._read_car_params()
    if car_params is None:
      return
    fingerprint = str(car_params.carFingerprint)
    if (
      self.runtime is not None
      and self._runtime_fingerprint == fingerprint
    ):
      return
    if self._offroad_restore_failed_fingerprint == fingerprint:
      return
    try:
      runtime = self.runtime_factory(car_params, self.storage_root)
    except Exception:
      # Corrupt artifacts are never replaced with an empty learner.
      self._offroad_restore_failed_fingerprint = fingerprint
      self._log_exception("blatv2 learner runtime restore failed closed")
      return
    self.runtime = runtime
    self._runtime_fingerprint = fingerprint
    self._offroad_restore_failed_fingerprint = None

  def _start_prepared_onroad_runtime(self) -> None:
    """Activate only a bundle already restored during an offroad period."""
    if (
      self.runtime is not None
      and self.runtime.coordinator.state.value == "onroad"
    ):
      return
    if self._runtime_unavailable_this_drive:
      return
    car_params = self._read_car_params()
    if car_params is None:
      return
    fingerprint = str(car_params.carFingerprint)
    if (
      self.runtime is not None
      and self._runtime_fingerprint == fingerprint
    ):
      self.runtime.transition_onroad()
      return
    # Qualifying/restoring evidence is offroad work. A process that first
    # appears mid-drive skips that drive and prepares at the next stop.
    self._runtime_unavailable_this_drive = True

  def _transition(self, started: bool) -> None:
    if self._onroad is not None and bool(started) == self._onroad:
      return
    was_onroad = self._onroad is True
    self._onroad = bool(started)
    if self._onroad:
      self._runtime_unavailable_this_drive = False
      self._pending_offroad_persist = False
      self._start_prepared_onroad_runtime()
    elif was_onroad and self.runtime is not None:
      try:
        self.runtime.transition_offroad_and_persist()
        self._pending_offroad_persist = False
      except Exception:
        self._pending_offroad_persist = True
        self._log_exception("blatv2 learner offroad persist failed")

  def _retry_pending_persist(self) -> None:
    if (
      self._onroad is not False
      or not self._pending_offroad_persist
      or self.runtime is None
    ):
      return
    try:
      self.runtime.persist_offroad()
      self._pending_offroad_persist = False
    except Exception:
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
    return MeasuredLearningFrame(
      sample_mono_ns=int(witness_mono_ns),
      speed_mps=float(car_state.vEgo),
      steering_angle_deg=float(car_state.steeringAngleDeg),
      steering_rate_deg_s=float(car_state.steeringRateDeg),
      steering_torque=float(car_state.steeringTorque),
      steering_pressed=bool(car_state.steeringPressed),
      standstill=bool(car_state.standstill),
      steer_fault_temporary=bool(car_state.steerFaultTemporary),
      steer_fault_permanent=bool(car_state.steerFaultPermanent),
      can_valid=bool(car_state.canValid),
      can_timeout=bool(car_state.canTimeout),
      applied_torque=float(car_output.actuatorsOutput.torque),
      # Deliberately read no actuator/request field from carControl. The
      # learner needs only the canonical proof that lateral actuation was
      # actually enabled for this measured response frame.
      lateral_active=bool(car_control.latActive),
      live_parameters_valid=bool(live_parameters.valid),
      angle_offset_valid=bool(live_parameters.angleOffsetValid),
      steer_ratio_valid=bool(live_parameters.steerRatioValid),
      stiffness_factor_valid=bool(
        live_parameters.stiffnessFactorValid,
      ),
      angle_offset_deg=float(live_parameters.angleOffsetDeg),
      steer_ratio=float(live_parameters.steerRatio),
      stiffness_factor=float(live_parameters.stiffnessFactor),
      roll_rad=float(live_parameters.roll),
      inputs_valid=True,
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
    if sources is None or self.runtime is None:
      self.unresolved_witness_count += 1
      return
    try:
      frame = self._measured_frame(witness_mono_ns, sources)
      accepted = self.runtime.ingest(frame)
    except (AttributeError, TypeError, ValueError, OverflowError):
      self.unresolved_witness_count += 1
      return
    if accepted:
      self.accepted_sample_count += 1

  def step(self, timeout_ms: int = STEP_TIMEOUT_MS) -> None:
    self.sm.update(timeout_ms)
    self._capture_sources()

    if (
      self.sm.seen["deviceState"]
      and self.sm.valid["deviceState"]
      and self.sm.alive["deviceState"]
    ):
      self._transition(bool(self.sm["deviceState"].started))
    if self._onroad is False and self.sm.updated["deviceState"]:
      self._retry_pending_persist()
      self._prepare_offroad_runtime()
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
