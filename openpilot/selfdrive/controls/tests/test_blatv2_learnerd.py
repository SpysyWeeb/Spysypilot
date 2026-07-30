from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from openpilot.selfdrive.controls.blatv2_learnerd import (
  PUBLISHED_SERVICES,
  SUBSCRIBED_SERVICES,
  BlatV2LearnerDaemon,
  _CanonicalSourceHistory,
  assert_no_actuation_publishers,
  default_learning_storage_root,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_coordinator import (
  LearningFinalization,
  LearningLifecycleState,
)
from openpilot.selfdrive.controls.lib.blatv2.learner import (
  LearningResult,
  NodeQualificationReport,
  QualificationReason,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  LearningRestoreError,
  MeasuredLearningFrame,
  PersistentLearningRuntime,
  build_detected_runtime_bundle,
  build_persistent_learning_runtime,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_status import (
  LEARNING_STATUS_PARAM,
  DriveEvidenceBaseline,
  build_learning_status_bytes,
  build_learning_status_payload,
  decode_learning_status,
  validate_learning_status_payload,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
)


GENERIC_FINGERPRINT = "GENERIC NON-HYUNDAI TORQUE PLATFORM"
GENERIC_CONTROLLER_MODULE = "test_blatv2_generic_controller"


class GenericControllerParams:
  STEER_MAX = 211
  STEER_DELTA_UP = 5
  STEER_DELTA_DOWN = 8
  STEER_STEP = 1
  STEER_DRIVER_ALLOWANCE = 37
  STEER_DRIVER_MULTIPLIER = 2
  STEER_DRIVER_FACTOR = 1

  def __init__(self, _car_params) -> None:
    pass


class GenericCarController:
  pass


GenericCarController.__module__ = GENERIC_CONTROLLER_MODULE


class GenericCarInterface:
  CarController = GenericCarController

  def __init__(self, car_params) -> None:
    self.car_params = car_params

  @staticmethod
  def torque_from_lateral_accel():
    return lambda lateral_accel, _torque_tuning: 0.37 * lateral_accel


def rack_dynamics() -> ProvisionalRackDynamics:
  return ProvisionalRackDynamics(
    rack_gain_deg_s2_per_torque=4000.0,
    rack_damping_per_s=10.0,
    rack_rate_resolution_deg_s=4.0,
    provenance="test-only provisional physical seed",
  )


def generic_car_params():
  cp = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
  cp.carFingerprint = GENERIC_FINGERPRINT
  cp.brand = "generic-test-brand"
  return cp


def generic_registry():
  return {GENERIC_FINGERPRINT: GenericCarInterface}


def build_generic_bundle(generic_controller_module):
  cp = generic_car_params()
  bundle, car_interface, controller_params = build_detected_runtime_bundle(
    car_params=cp,
    provisional_rack_dynamics=rack_dynamics(),
    interface_registry=generic_registry(),
  )
  return cp, bundle, car_interface, controller_params


def build_generic_runtime(
  directory: Path,
  generic_controller_module,
) -> PersistentLearningRuntime:
  cp = generic_car_params()
  return build_persistent_learning_runtime(
    car_params=cp,
    storage_root=directory,
    provisional_rack_dynamics=rack_dynamics(),
    interface_registry=generic_registry(),
  )


def measured_frame(
  cp,
  sample_mono_ns: int,
  **overrides,
) -> MeasuredLearningFrame:
  values = {
    "sample_mono_ns": sample_mono_ns,
    "speed_mps": 10.0,
    "steering_angle_deg": 0.0,
    "steering_rate_deg_s": 5.0,
    "steering_torque": 0.0,
    "steering_pressed": False,
    "standstill": False,
    "steer_fault_temporary": False,
    "steer_fault_permanent": False,
    "can_valid": True,
    "can_timeout": False,
    "applied_torque": 0.0,
    "lateral_active": True,
    "live_parameters_valid": True,
    "angle_offset_valid": True,
    "steer_ratio_valid": True,
    "stiffness_factor_valid": True,
    "angle_offset_deg": 0.0,
    "steer_ratio": float(cp.steerRatio),
    "stiffness_factor": float(cp.tireStiffnessFactor),
    "roll_rad": 0.0,
    "inputs_valid": True,
  }
  values.update(overrides)
  return MeasuredLearningFrame(**values)


def _test_generic_non_hyundai_construction_uses_detected_opendbc_limits(
  generic_controller_module,
) -> None:
  cp, bundle, car_interface, controller_params = build_generic_bundle(
    generic_controller_module,
  )
  assert cp.brand == "generic-test-brand"
  assert type(car_interface) is GenericCarInterface
  assert type(controller_params) is GenericControllerParams
  assert bundle.car_fingerprint == GENERIC_FINGERPRINT
  assert bundle.vehicle_identity == GENERIC_FINGERPRINT
  assert (
    bundle.torque_limits.steer_max,
    bundle.torque_limits.delta_up,
    bundle.torque_limits.delta_down,
  ) == (211, 5, 8)
  assert bundle.torque_callback_slope == 0.37
  assert "hyundai" not in type(car_interface).__module__.lower()


def _test_runtime_collects_only_onroad_and_persists_only_offroad(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  runtime = build_generic_runtime(tmp_path, generic_controller_module)
  cp = runtime.car_params
  artifact_root = runtime.artifact_paths.root
  assert runtime.coordinator.state is LearningLifecycleState.OFFROAD
  assert not artifact_root.exists()

  runtime.transition_onroad()
  with patch(
    "openpilot.selfdrive.controls.lib.blatv2.learning_coordinator._atomic_write_bytes",
  ) as write:
    assert not runtime.ingest(measured_frame(cp, 1_000_000_000))
    assert not runtime.ingest(measured_frame(cp, 1_010_000_000))
    assert runtime.ingest(measured_frame(cp, 1_020_000_000))
    write.assert_not_called()
  assert not artifact_root.exists()

  finalization = runtime.transition_offroad_and_persist()
  assert runtime.coordinator.state is LearningLifecycleState.OFFROAD
  assert runtime.artifact_paths.evidence.read_bytes() == (
    finalization.evidence_bytes
  )
  assert runtime.artifact_paths.manifest.read_bytes() == (
    finalization.manifest_bytes
  )
  assert finalization.candidate_profile_json is None
  assert not runtime.artifact_paths.candidates.exists()


def _test_restart_restores_exact_cross_drive_evidence(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  first = build_generic_runtime(tmp_path, generic_controller_module)
  cp = first.car_params
  first.transition_onroad()
  assert not first.ingest(measured_frame(cp, 1_000_000_000))
  assert not first.ingest(measured_frame(cp, 1_010_000_000))
  assert first.ingest(measured_frame(cp, 1_020_000_000))
  saved = first.transition_offroad_and_persist()

  restored = build_generic_runtime(tmp_path, generic_controller_module)
  assert restored.coordinator.state is LearningLifecycleState.OFFROAD
  assert restored.coordinator.finalize().evidence_bytes == (
    saved.evidence_bytes
  )
  assert restored.coordinator.finalize().manifest_bytes == (
    saved.manifest_bytes
  )
  restored.transition_onroad()
  assert not restored.ingest(measured_frame(cp, 2_000_000_000))
  assert not restored.ingest(measured_frame(cp, 2_010_000_000))
  assert restored.ingest(measured_frame(cp, 2_020_000_000))
  second = restored.transition_offroad_and_persist()
  assert second.evidence_bytes != saved.evidence_bytes


def _test_restore_corruption_and_seed_mismatch_fail_closed(
  tmp_path: Path,
  generic_controller_module,
  test_case: unittest.TestCase,
) -> None:
  runtime = build_generic_runtime(tmp_path, generic_controller_module)
  runtime.transition_onroad()
  runtime.ingest(measured_frame(runtime.car_params, 1_000_000_000))
  runtime.ingest(measured_frame(runtime.car_params, 1_010_000_000))
  runtime.ingest(measured_frame(runtime.car_params, 1_020_000_000))
  runtime.transition_offroad_and_persist()
  paths = runtime.artifact_paths
  original = paths.evidence.read_bytes()
  paths.evidence.write_bytes(original[:-1])
  with test_case.assertRaisesRegex(LearningRestoreError, "canonical restore"):
    PersistentLearningRuntime.restore(
      car_params=runtime.car_params,
      runtime_bundle=runtime.runtime_bundle,
      artifact_paths=paths,
    )
  assert paths.evidence.read_bytes() == original[:-1]

  paths.evidence.write_bytes(original)
  different_seed = replace(
    runtime.runtime_bundle.seed_profile,
    provenance="different exact seed provenance",
  )
  different_bundle = replace(
    runtime.runtime_bundle,
    seed_profile=different_seed,
  )
  with test_case.assertRaisesRegex(LearningRestoreError, "canonical restore"):
    PersistentLearningRuntime.restore(
      car_params=runtime.car_params,
      runtime_bundle=different_bundle,
      artifact_paths=paths,
    )


def _test_clean_frame_filters_reject_driver_inactive_invalid_gap_and_constraint(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  runtime = build_generic_runtime(tmp_path, generic_controller_module)
  cp = runtime.car_params
  runtime.transition_onroad()
  base = 1_000_000_000

  assert not runtime.ingest(measured_frame(cp, base))
  assert not runtime.ingest(measured_frame(cp, base + 10_000_000))
  assert runtime.ingest(measured_frame(cp, base + 20_000_000))
  accepted = runtime.coordinator.accepted_sample_count

  assert not runtime.ingest(measured_frame(
    cp,
    base + 30_000_000,
    steering_pressed=True,
  ))
  assert not runtime.ingest(measured_frame(
    cp,
    base + 40_000_000,
    lateral_active=False,
  ))
  assert not runtime.ingest(measured_frame(
    cp,
    base + 50_000_000,
    standstill=True,
  ))
  assert not runtime.ingest(measured_frame(
    cp,
    base + 60_000_000,
    live_parameters_valid=False,
  ))
  assert not runtime.ingest(measured_frame(
    cp,
    base + 70_000_000,
    can_valid=False,
  ))
  assert runtime.coordinator.accepted_sample_count == accepted

  # Envelope history and the measured derivative each warm independently.
  assert not runtime.ingest(measured_frame(cp, base + 80_000_000))
  assert not runtime.ingest(measured_frame(cp, base + 90_000_000))
  assert runtime.ingest(measured_frame(cp, base + 100_000_000))
  accepted += 1

  upper_boundary = (
    runtime.runtime_bundle.torque_limits.delta_up
    / runtime.runtime_bundle.torque_limits.steer_max
  )
  assert not runtime.ingest(measured_frame(
    cp,
    base + 110_000_000,
    applied_torque=upper_boundary,
  ))
  assert runtime.last_actuator_constrained
  assert runtime.coordinator.accepted_sample_count == accepted

  # A dropped controls frame cannot be compressed into a 10 ms derivative.
  assert not runtime.ingest(measured_frame(cp, base + 140_000_000))
  assert runtime.coordinator.accepted_sample_count == accepted


def _test_input_contract_and_subscriptions_exclude_intent_and_requests() -> None:
  assert_no_actuation_publishers()
  assert PUBLISHED_SERVICES == ()
  assert "carControl" in SUBSCRIBED_SERVICES
  assert "modelV2" not in SUBSCRIBED_SERVICES
  forbidden = ("desired", "request", "candidate", "model", "reference")
  frame_fields = tuple(MeasuredLearningFrame.__dataclass_fields__)
  assert not any(
    token in field.lower()
    for field in frame_fields
    for token in forbidden
  )


def _test_canonical_history_resolves_one_update_race_and_rejects_stale() -> None:
  first = SimpleNamespace(name="first")
  future = SimpleNamespace(name="future")
  history = _CanonicalSourceHistory()
  history.update(
    message=first,
    mono_ns=90_000_000,
    valid=True,
    alive=True,
  )
  history.update(
    message=future,
    mono_ns=110_000_000,
    valid=True,
    alive=True,
  )
  selected = history.select(
    witness_mono_ns=100_000_000,
    maximum_age_ns=15_000_000,
  )
  assert selected is not None
  assert selected.message.name == "first"
  assert selected.mono_ns == 90_000_000

  future_only = _CanonicalSourceHistory()
  future_only.update(
    message=future,
    mono_ns=110_000_000,
    valid=True,
    alive=True,
  )
  assert future_only.select(
    witness_mono_ns=100_000_000,
    maximum_age_ns=15_000_000,
  ) is None

  stale = _CanonicalSourceHistory()
  stale.update(
    message=first,
    mono_ns=84_000_000,
    valid=True,
    alive=True,
  )
  assert stale.select(
    witness_mono_ns=100_000_000,
    maximum_age_ns=15_000_000,
  ) is None


class FakeParams:
  def __init__(self, root: Path, car_params_bytes: bytes = b"car-params"):
    self.root = root
    # Production keeps this identity across manager restarts; CarParams itself
    # is manager-cleared and cannot be the learner's sole restore source.
    self.values: dict[str, object] = {
      "CarParamsPersistent": car_params_bytes,
    }
    self.puts: list[tuple[str, object, bool]] = []
    self.removes: list[str] = []
    self.remove_failures = 0
    self.before_put = None

  def get_param_path(self) -> str:
    return str(self.root / "params" / "d")

  def get(self, key: str, block: bool = False):
    assert block is False
    return self.values.get(key)

  def put(self, key: str, value, block: bool = False):
    # JSON Params reject pre-encoded bytes at the Python/C++ boundary.
    assert type(value) is dict
    if self.before_put is not None:
      self.before_put(key, value)
    self.values[key] = value
    self.puts.append((key, value, block))

  def remove(self, key: str):
    if self.remove_failures > 0:
      self.remove_failures -= 1
      raise OSError("injected Params remove failure")
    self.values.pop(key, None)
    self.removes.append(key)


class FakeLogger:
  def __init__(self) -> None:
    self.exceptions: list[str] = []

  def exception(self, message: str) -> None:
    self.exceptions.append(message)


class FakeSubMaster:
  def __init__(self) -> None:
    self.data = {
      service: SimpleNamespace()
      for service in SUBSCRIBED_SERVICES
    }
    self.updated = dict.fromkeys(SUBSCRIBED_SERVICES, False)
    self.seen = dict.fromkeys(SUBSCRIBED_SERVICES, False)
    self.valid = dict.fromkeys(SUBSCRIBED_SERVICES, False)
    self.alive = dict.fromkeys(SUBSCRIBED_SERVICES, False)
    self.logMonoTime = dict.fromkeys(SUBSCRIBED_SERVICES, 0)
    self.timeouts: list[int] = []

  def __getitem__(self, service: str):
    return self.data[service]

  def update(self, timeout: int) -> None:
    self.timeouts.append(timeout)

  def publish(self, messages: dict[str, tuple[object, int]]) -> None:
    self.updated = dict.fromkeys(SUBSCRIBED_SERVICES, False)
    for service, (message, mono_ns) in messages.items():
      self.data[service] = message
      self.updated[service] = True
      self.seen[service] = True
      self.valid[service] = True
      self.alive[service] = True
      self.logMonoTime[service] = mono_ns


def fake_messages(cp, *, started: bool, active: bool = True):
  return {
    "deviceState": SimpleNamespace(started=started),
    "controlsState": SimpleNamespace(),
    "carControl": SimpleNamespace(latActive=active),
    "carState": SimpleNamespace(
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
    ),
    "carOutput": SimpleNamespace(
      actuatorsOutput=SimpleNamespace(torque=0.0),
    ),
    "liveParameters": SimpleNamespace(
      valid=True,
      angleOffsetValid=True,
      steerRatioValid=True,
      stiffnessFactorValid=True,
      angleOffsetDeg=0.0,
      steerRatio=cp.steerRatio,
      stiffnessFactor=cp.tireStiffnessFactor,
      roll=0.0,
    ),
  }


def publish_frame(sm: FakeSubMaster, cp, mono_ns: int) -> None:
  messages = fake_messages(cp, started=True)
  sm.publish({
    service: (messages[service], mono_ns)
    for service in SUBSCRIBED_SERVICES
  })


def _test_daemon_observes_offroad_without_onroad_poll_and_restores(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  cp = generic_car_params()
  sm = FakeSubMaster()
  params = FakeParams(tmp_path)
  assert "CarParams" not in params.values
  assert params.values["CarParamsPersistent"] == b"car-params"
  params.values["CarParams"] = b"manager-scoped-must-not-win"
  logger = FakeLogger()
  created: list[PersistentLearningRuntime] = []
  decoded_values: list[bytes] = []

  def runtime_factory(car_params, storage_root):
    runtime = build_persistent_learning_runtime(
      car_params=car_params,
      storage_root=storage_root,
      provisional_rack_dynamics=rack_dynamics(),
      interface_registry=generic_registry(),
    )
    created.append(runtime)
    return runtime

  daemon = BlatV2LearnerDaemon(
    sm=sm,
    params=params,
    storage_root=tmp_path / "learning",
    car_params_decoder=lambda encoded: (
      decoded_values.append(encoded) or cp
    ),
    runtime_factory=runtime_factory,
    logger=logger,
  )
  assert default_learning_storage_root(params) == (
    tmp_path / "params" / "blatv2-learning"
  )

  # The always-running process prepares and authenticates evidence offroad;
  # qualification is never smuggled into the first onroad frame.
  offroad_messages = fake_messages(cp, started=False)
  sm.publish({
    "deviceState": (offroad_messages["deviceState"], 800_000_000),
  })
  daemon.step()
  assert len(created) == 1
  assert decoded_values == [b"car-params"]
  assert created[0].coordinator.state is LearningLifecycleState.OFFROAD
  assert not (tmp_path / "learning").exists()

  params.values["CarParams"] = b"car-params"
  start_messages = fake_messages(cp, started=True)
  sm.publish({
    "deviceState": (start_messages["deviceState"], 900_000_000),
  })
  daemon.step()
  assert len(created) == 1
  assert created[0].coordinator.state is LearningLifecycleState.ONROAD
  assert not (tmp_path / "learning").exists()

  publish_frame(sm, cp, 1_000_000_000)
  daemon.step()
  publish_frame(sm, cp, 1_010_000_000)
  daemon.step()
  publish_frame(sm, cp, 1_020_000_000)
  daemon.step()
  assert daemon.controls_witness_count == 3
  assert daemon.accepted_sample_count == 1
  assert not (tmp_path / "learning").exists()

  def assert_artifacts_precede_status(key, _value):
    assert key == LEARNING_STATUS_PARAM
    assert created[0].artifact_paths.evidence.is_file()
    assert created[0].artifact_paths.manifest.is_file()

  params.before_put = assert_artifacts_precede_status
  stop_messages = fake_messages(cp, started=False)
  sm.publish({
    "deviceState": (stop_messages["deviceState"], 2_000_000_000),
  })
  daemon.step()
  assert created[0].coordinator.state is LearningLifecycleState.OFFROAD
  assert created[0].artifact_paths.evidence.is_file()
  assert created[0].artifact_paths.manifest.is_file()
  status = params.values[LEARNING_STATUS_PARAM]
  assert type(status) is dict
  assert status["last_drive_complete"] is True
  assert all(
    node["last_drive_clean_support_s"] is not None
    and node["last_drive_accepted_sample_count"] is not None
    for node in status["nodes"]
  )
  assert params.puts[-1][0] == LEARNING_STATUS_PARAM
  assert params.puts[-1][2] is True
  assert all(timeout == 100 for timeout in sm.timeouts)
  assert logger.exceptions == []

  # A reconstructed daemon/runtime restores the exact persisted evidence.
  restored = runtime_factory(cp, tmp_path / "learning")
  assert restored.coordinator.finalize().evidence_bytes == (
    created[0].coordinator.finalize().evidence_bytes
  )

  # Manager-start clears the cache; verified offroad restore republishes the
  # cumulative state without inventing unknowable per-drive deltas.
  restart_params = FakeParams(tmp_path / "restart")
  restart_sm = FakeSubMaster()
  restarted = BlatV2LearnerDaemon(
    sm=restart_sm,
    params=restart_params,
    storage_root=tmp_path / "learning",
    car_params_decoder=lambda _encoded: cp,
    runtime_factory=runtime_factory,
    logger=FakeLogger(),
  )
  restart_sm.publish({
    "deviceState": (offroad_messages["deviceState"], 3_000_000_000),
  })
  restarted.step()
  restored_status = restart_params.values[LEARNING_STATUS_PARAM]
  assert restored_status["last_drive_complete"] is False
  assert all(
    node["last_drive_clean_support_s"] is None
    and node["last_drive_accepted_sample_count"] is None
    for node in restored_status["nodes"]
  )


def _test_mid_drive_start_skips_collection_until_offroad_preparation(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  cp = generic_car_params()
  sm = FakeSubMaster()
  params = FakeParams(tmp_path)
  logger = FakeLogger()
  factory_calls: list[bool] = []

  def runtime_factory(car_params, storage_root):
    factory_calls.append(True)
    return build_persistent_learning_runtime(
      car_params=car_params,
      storage_root=storage_root,
      provisional_rack_dynamics=rack_dynamics(),
      interface_registry=generic_registry(),
    )

  daemon = BlatV2LearnerDaemon(
    sm=sm,
    params=params,
    storage_root=tmp_path / "learning",
    car_params_decoder=lambda _encoded: cp,
    runtime_factory=runtime_factory,
    logger=logger,
  )
  started = fake_messages(cp, started=True)
  sm.publish({"deviceState": (started["deviceState"], 1_000_000_000)})
  daemon.step()
  assert factory_calls == []
  assert daemon.runtime is None

  publish_frame(sm, cp, 1_010_000_000)
  daemon.step()
  assert daemon.accepted_sample_count == 0
  assert not (tmp_path / "learning").exists()
  assert LEARNING_STATUS_PARAM not in params.values

  stopped = fake_messages(cp, started=False)
  sm.publish({"deviceState": (stopped["deviceState"], 2_000_000_000)})
  daemon.step()
  assert factory_calls == [True]
  assert daemon.runtime is not None
  assert daemon.runtime.coordinator.state is LearningLifecycleState.OFFROAD

  params.values["CarParams"] = b"car-params"
  sm.publish({"deviceState": (started["deviceState"], 3_000_000_000)})
  daemon.step()
  assert daemon.runtime.coordinator.state is LearningLifecycleState.ONROAD


def _test_car_params_precedence_fallback_and_transient_absence(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  cp = generic_car_params()

  missing_params = FakeParams(tmp_path / "missing")
  missing_params.values.pop("CarParamsPersistent")
  preserved_status = {"preserved_during_unknown_identity": True}
  missing_params.values[LEARNING_STATUS_PARAM] = preserved_status
  missing_sm = FakeSubMaster()
  missing = BlatV2LearnerDaemon(
    sm=missing_sm,
    params=missing_params,
    storage_root=tmp_path / "missing-learning",
    car_params_decoder=lambda _encoded: cp,
    runtime_factory=lambda _cp, _root: (_ for _ in ()).throw(
      AssertionError("runtime factory needs a concrete CarParams"),
    ),
    logger=FakeLogger(),
  )
  stopped = fake_messages(cp, started=False)
  missing_sm.publish({
    "deviceState": (stopped["deviceState"], 1_000_000_000),
  })
  missing.step()
  assert missing.runtime is None
  assert missing_params.values[LEARNING_STATUS_PARAM] is preserved_status
  assert missing_params.removes == []

  fallback_params = FakeParams(tmp_path / "fallback")
  fallback_params.values.pop("CarParamsPersistent")
  fallback_params.values["CarParams"] = b"manager-scoped-car-params"
  fallback_sm = FakeSubMaster()
  calls: list[bytes] = []

  def runtime_factory(car_params, storage_root):
    calls.append(fallback_params.values["CarParams"])
    return build_persistent_learning_runtime(
      car_params=car_params,
      storage_root=storage_root,
      provisional_rack_dynamics=rack_dynamics(),
      interface_registry=generic_registry(),
    )

  fallback = BlatV2LearnerDaemon(
    sm=fallback_sm,
    params=fallback_params,
    storage_root=tmp_path / "fallback-learning",
    car_params_decoder=lambda _encoded: cp,
    runtime_factory=runtime_factory,
    logger=FakeLogger(),
  )
  fallback_sm.publish({
    "deviceState": (stopped["deviceState"], 2_000_000_000),
  })
  fallback.step()
  assert calls == [b"manager-scoped-car-params"]
  assert fallback.runtime is not None
  first_runtime = fallback.runtime

  # The model fingerprint alone is not the runtime identity. A changed exact
  # canonical CarParams for the same fingerprint rebuilds the prepared owner.
  fallback_params.values["CarParams"] = b"changed-same-fingerprint"
  fallback_sm.publish({
    "deviceState": (stopped["deviceState"], 2_100_000_000),
  })
  fallback.step()
  assert calls == [
    b"manager-scoped-car-params",
    b"changed-same-fingerprint",
  ]
  assert fallback.runtime is not first_runtime
  assert (
    fallback._prepared_car_params_bytes
    == b"changed-same-fingerprint"
  )


def _test_live_identity_late_and_mismatch_never_cross_contaminate(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  cp_a = generic_car_params()
  cp_b = SimpleNamespace(carFingerprint="DIFFERENT LIVE VEHICLE")

  def decoder(encoded):
    return cp_b if encoded == b"vehicle-b" else cp_a

  def runtime_factory(car_params, storage_root):
    return build_persistent_learning_runtime(
      car_params=car_params,
      storage_root=storage_root,
      provisional_rack_dynamics=rack_dynamics(),
      interface_registry=generic_registry(),
    )

  # A persistent identity prepares A, but a confirmed live B makes the entire
  # drive ineligible. Changing the live Param back to A cannot reopen it.
  mismatch_params = FakeParams(
    tmp_path / "mismatch",
    car_params_bytes=b"vehicle-a",
  )
  mismatch_params.values["CarParams"] = b"vehicle-b"
  mismatch_sm = FakeSubMaster()
  mismatch = BlatV2LearnerDaemon(
    sm=mismatch_sm,
    params=mismatch_params,
    storage_root=tmp_path / "mismatch-learning",
    car_params_decoder=decoder,
    runtime_factory=runtime_factory,
    logger=FakeLogger(),
  )
  stopped = fake_messages(cp_a, started=False)
  started = fake_messages(cp_a, started=True)
  mismatch_sm.publish({
    "deviceState": (stopped["deviceState"], 1_000_000_000),
  })
  mismatch.step()
  assert mismatch.runtime is not None
  mismatch_sm.publish({
    "deviceState": (started["deviceState"], 2_000_000_000),
  })
  mismatch.step()
  assert mismatch._runtime_unavailable_this_drive
  assert not mismatch._live_identity_bound
  assert (
    mismatch.runtime.coordinator.state
    is LearningLifecycleState.OFFROAD
  )
  mismatch_params.values["CarParams"] = b"vehicle-a"
  for mono_ns in (2_010_000_000, 2_020_000_000, 2_030_000_000):
    publish_frame(mismatch_sm, cp_a, mono_ns)
    mismatch.step()
  assert mismatch.accepted_sample_count == 0
  assert mismatch.runtime.coordinator.accepted_sample_count == 0

  # A late live CarParams is a startup race, not a rejected drive. Witnesses
  # remain unresolved without crashing until the identity is bound.
  late_params = FakeParams(
    tmp_path / "late",
    car_params_bytes=b"vehicle-a",
  )
  late_sm = FakeSubMaster()
  late = BlatV2LearnerDaemon(
    sm=late_sm,
    params=late_params,
    storage_root=tmp_path / "late-learning",
    car_params_decoder=decoder,
    runtime_factory=runtime_factory,
    logger=FakeLogger(),
  )
  late_sm.publish({
    "deviceState": (stopped["deviceState"], 3_000_000_000),
  })
  late.step()
  late_sm.publish({
    "deviceState": (started["deviceState"], 4_000_000_000),
  })
  late.step()
  assert late.runtime is not None
  assert late.runtime.coordinator.state is LearningLifecycleState.OFFROAD
  for mono_ns in (4_010_000_000, 4_020_000_000):
    publish_frame(late_sm, cp_a, mono_ns)
    late.step()
  assert late.controls_witness_count == 2
  assert late.unresolved_witness_count == 2
  assert late.accepted_sample_count == 0
  assert not late._runtime_unavailable_this_drive

  late_params.values["CarParams"] = b"vehicle-a"
  for mono_ns in (4_030_000_000, 4_040_000_000, 4_050_000_000):
    publish_frame(late_sm, cp_a, mono_ns)
    late.step()
  assert late._live_identity_bound
  assert late.runtime.coordinator.state is LearningLifecycleState.ONROAD
  assert late.accepted_sample_count == 1

  # A confirmed identity change after binding closes collection before that
  # witness reaches A's evidence, and the drive cannot reopen.
  late_params.values["CarParams"] = b"vehicle-b"
  publish_frame(late_sm, cp_a, 4_060_000_000)
  late.step()
  assert late._runtime_unavailable_this_drive
  assert not late._live_identity_bound
  assert late.accepted_sample_count == 1
  unresolved_after_change = late.unresolved_witness_count
  late_params.values["CarParams"] = b"vehicle-a"
  publish_frame(late_sm, cp_a, 4_070_000_000)
  late.step()
  assert late.accepted_sample_count == 1
  assert late.unresolved_witness_count == unresolved_after_change + 1


def _test_learning_status_is_canonical_strict_and_drive_local(
  tmp_path: Path,
  generic_controller_module,
  test_case: unittest.TestCase,
) -> None:
  runtime = build_generic_runtime(tmp_path, generic_controller_module)
  before = runtime.coordinator.finalize()
  baseline = DriveEvidenceBaseline.from_support_diagnostics(
    runtime.coordinator.support_diagnostics,
  )
  empty_payload = build_learning_status_payload(
    finalization=before,
    runtime_bundle=runtime.runtime_bundle,
    drive_baseline=None,
  )
  empty_bytes = build_learning_status_bytes(
    finalization=before,
    runtime_bundle=runtime.runtime_bundle,
    drive_baseline=None,
  )
  assert empty_bytes == build_learning_status_bytes(
    finalization=before,
    runtime_bundle=runtime.runtime_bundle,
    drive_baseline=None,
  )
  assert decode_learning_status(empty_bytes) == empty_payload
  assert empty_payload["last_drive_complete"] is False
  assert empty_payload["all_nodes_qualified"] is False
  assert empty_payload["candidate_profile_sha256"] is None
  assert empty_payload["candidate_profile_revision"] is None
  assert [node["node_index"] for node in empty_payload["nodes"]] == (
    list(range(len(runtime.runtime_bundle.seed_profile.nodes)))
  )
  assert all(node["reasons"] for node in empty_payload["nodes"])
  assert all(
    node["candidate_parameters"] is None
    and node["last_drive_clean_support_s"] is None
    and node["last_drive_accepted_sample_count"] is None
    for node in empty_payload["nodes"]
  )
  with test_case.assertRaisesRegex(ValueError, "not canonical"):
    decode_learning_status(empty_bytes + b" ")
  invalid_confidence = copy.deepcopy(empty_payload)
  invalid_confidence["nodes"][0]["confidence"] = 1.1
  with test_case.assertRaisesRegex(ValueError, "must not exceed one"):
    validate_learning_status_payload(invalid_confidence)
  invalid_reason = copy.deepcopy(empty_payload)
  invalid_reason["nodes"][0]["reasons"] = ["qualified"]
  with test_case.assertRaisesRegex(ValueError, "reasons disagree"):
    validate_learning_status_payload(invalid_reason)

  learned_nodes = tuple(
    replace(
      node,
      parameters=replace(
        node.parameters,
        torque_per_lateral_accel=0.4,
        rack_gain_deg_s2_per_torque=1500.0,
        rack_damping_per_s=8.0,
        kinetic_friction_torque=0.03,
        confidence=1.0,
        qualified=True,
      ),
    )
    for node in runtime.runtime_bundle.seed_profile.nodes
  )
  learned_profile = replace(
    runtime.runtime_bundle.seed_profile,
    revision=runtime.runtime_bundle.seed_profile.revision + 1,
    provenance="test-only fully qualified evidence",
    nodes=learned_nodes,
  )
  learned_reports = tuple(
    NodeQualificationReport(
      node_index=index,
      speed_mps=node.speed_mps,
      minimum_support_s=150.0,
      clean_support_s=180.0,
      supported_sample_count=18000,
      training_count=14400,
      validation_count=3600,
      validation_support_s=36.0,
      lateral_accel_span_mps2=1.0,
      lateral_accel_rms_mps2=0.3,
      rack_travel_deg=240.0,
      applied_torque_span=0.4,
      rack_reversals=8,
      seed_validation_rms=0.2,
      candidate_validation_rms=0.1,
      confidence=1.0,
      reasons=(QualificationReason.QUALIFIED,),
      candidate_parameters=node.parameters,
    )
    for index, node in enumerate(learned_nodes)
  )
  candidate_json = learned_profile.to_json().encode("utf-8")
  qualified_finalization = LearningFinalization(
    manifest_bytes=b"test manifest",
    manifest_sha256="b" * 64,
    evidence_bytes=b"test evidence",
    evidence_sha256="c" * 64,
    candidate_profile_json=candidate_json,
    candidate_profile_sha256=hashlib.sha256(candidate_json).hexdigest(),
    learning_result=LearningResult(
      node_reports=learned_reports,
      candidate_profile=learned_profile,
    ),
  )
  qualified = build_learning_status_payload(
    finalization=qualified_finalization,
    runtime_bundle=runtime.runtime_bundle,
    drive_baseline=None,
  )
  assert qualified["all_nodes_qualified"] is True
  assert qualified["candidate_profile_revision"] == learned_profile.revision
  assert all(node["qualified"] for node in qualified["nodes"])
  assert all(
    set(node["candidate_parameters"]) == {
      "torque_per_lateral_accel",
      "rack_gain_deg_s2_per_torque",
      "rack_damping_per_s",
      "kinetic_friction_torque",
    }
    for node in qualified["nodes"]
  )

  runtime.transition_onroad()
  cp = runtime.car_params
  assert not runtime.ingest(measured_frame(cp, 1_000_000_000))
  assert not runtime.ingest(measured_frame(cp, 1_010_000_000))
  assert runtime.ingest(measured_frame(cp, 1_020_000_000))
  after = runtime.transition_offroad_and_persist()
  driven = build_learning_status_payload(
    finalization=after,
    runtime_bundle=runtime.runtime_bundle,
    drive_baseline=baseline,
  )
  assert driven["last_drive_complete"] is True
  assert all(
    node["last_drive_clean_support_s"] is not None
    and node["last_drive_accepted_sample_count"] is not None
    for node in driven["nodes"]
  )
  assert sum(
    node["last_drive_accepted_sample_count"]
    for node in driven["nodes"]
  ) >= 1

  mismatched_seed = replace(
    runtime.runtime_bundle.seed_profile,
    nodes=runtime.runtime_bundle.seed_profile.nodes[:-1],
  )
  mismatched_bundle = replace(
    runtime.runtime_bundle,
    seed_profile=mismatched_seed,
  )
  with test_case.assertRaisesRegex(
    ValueError,
    "runtime speed grid differ",
  ):
    build_learning_status_payload(
      finalization=after,
      runtime_bundle=mismatched_bundle,
      drive_baseline=baseline,
    )


def _test_status_write_failures_and_corrupt_restore_fail_closed(
  tmp_path: Path,
  generic_controller_module,
) -> None:
  cp = generic_car_params()
  sm = FakeSubMaster()
  params = FakeParams(tmp_path)
  logger = FakeLogger()

  def runtime_factory(car_params, storage_root):
    return build_persistent_learning_runtime(
      car_params=car_params,
      storage_root=storage_root,
      provisional_rack_dynamics=rack_dynamics(),
      interface_registry=generic_registry(),
    )

  daemon = BlatV2LearnerDaemon(
    sm=sm,
    params=params,
    storage_root=tmp_path / "learning",
    car_params_decoder=lambda _encoded: cp,
    runtime_factory=runtime_factory,
    logger=logger,
  )
  stopped = fake_messages(cp, started=False)
  started = fake_messages(cp, started=True)
  sm.publish({"deviceState": (stopped["deviceState"], 1_000_000_000)})
  daemon.step()
  params.values["CarParams"] = b"car-params"
  sm.publish({"deviceState": (started["deviceState"], 2_000_000_000)})
  daemon.step()
  publish_frame(sm, cp, 2_010_000_000)
  daemon.step()
  publish_frame(sm, cp, 2_020_000_000)
  daemon.step()
  publish_frame(sm, cp, 2_030_000_000)
  daemon.step()
  prior = {"preserved": True}
  params.values[LEARNING_STATUS_PARAM] = prior

  assert daemon.runtime is not None
  with patch.object(
    daemon.runtime,
    "transition_offroad_and_persist",
    side_effect=OSError("injected persistence failure"),
  ):
    sm.publish({"deviceState": (stopped["deviceState"], 3_000_000_000)})
    daemon.step()
  assert params.values[LEARNING_STATUS_PARAM] is prior
  assert any("offroad persist failed" in item for item in logger.exceptions)

  corrupt_root = tmp_path / "corrupt"
  corrupt_paths = runtime_factory(cp, corrupt_root).artifact_paths
  corrupt_paths.root.mkdir(parents=True)
  corrupt_paths.evidence.write_bytes(b"corrupt")
  corrupt_paths.manifest.write_bytes(b"corrupt")
  corrupt_params = FakeParams(tmp_path / "corrupt-params")
  previous_status = {"previous_valid_status": True}
  corrupt_params.values[LEARNING_STATUS_PARAM] = previous_status
  # The step retries once after restore; keep both attempts failed so the next
  # offroad poll proves retry state survives the whole cycle.
  corrupt_params.remove_failures = 2
  corrupt_sm = FakeSubMaster()
  corrupt = BlatV2LearnerDaemon(
    sm=corrupt_sm,
    params=corrupt_params,
    storage_root=corrupt_root,
    car_params_decoder=lambda _encoded: cp,
    runtime_factory=runtime_factory,
    logger=FakeLogger(),
  )
  corrupt_sm.publish({
    "deviceState": (stopped["deviceState"], 4_000_000_000),
  })
  corrupt.step()
  assert corrupt.runtime is None
  assert corrupt_params.values[LEARNING_STATUS_PARAM] is previous_status
  assert corrupt._learning_status_clear_pending
  assert not corrupt_params.removes

  # Restore attempts remain suppressed for the known-bad fingerprint, but the
  # non-authoritative stale-cache removal retries independently.
  corrupt_sm.publish({
    "deviceState": (stopped["deviceState"], 4_100_000_000),
  })
  corrupt.step()
  assert LEARNING_STATUS_PARAM not in corrupt_params.values
  assert not corrupt_params.puts
  assert corrupt_params.removes == [LEARNING_STATUS_PARAM]
  assert not corrupt._learning_status_clear_pending


class TestBlatV2LearnerDaemon(unittest.TestCase):
  def setUp(self):
    self.temporary_directory = tempfile.TemporaryDirectory()
    self.tmp_path = Path(self.temporary_directory.name)
    module = ModuleType(GENERIC_CONTROLLER_MODULE)
    module.CarControllerParams = GenericControllerParams
    self.module_patcher = patch.dict(
      sys.modules,
      {GENERIC_CONTROLLER_MODULE: module},
    )
    self.module_patcher.start()

  def tearDown(self):
    self.module_patcher.stop()
    self.temporary_directory.cleanup()

  def test_generic_non_hyundai_construction_uses_detected_opendbc_limits(self):
    _test_generic_non_hyundai_construction_uses_detected_opendbc_limits(None)

  def test_runtime_collects_only_onroad_and_persists_only_offroad(self):
    _test_runtime_collects_only_onroad_and_persists_only_offroad(
      self.tmp_path,
      None,
    )

  def test_restart_restores_exact_cross_drive_evidence(self):
    _test_restart_restores_exact_cross_drive_evidence(self.tmp_path, None)

  def test_restore_corruption_and_seed_mismatch_fail_closed(self):
    _test_restore_corruption_and_seed_mismatch_fail_closed(
      self.tmp_path,
      None,
      self,
    )

  def test_clean_frame_filters_reject_driver_inactive_invalid_gap_and_constraint(self):
    _test_clean_frame_filters_reject_driver_inactive_invalid_gap_and_constraint(
      self.tmp_path,
      None,
    )

  def test_input_contract_and_subscriptions_exclude_intent_and_requests(self):
    _test_input_contract_and_subscriptions_exclude_intent_and_requests()

  def test_canonical_history_resolves_one_update_race_and_rejects_stale(self):
    _test_canonical_history_resolves_one_update_race_and_rejects_stale()

  def test_daemon_observes_offroad_without_onroad_poll_and_restores(self):
    _test_daemon_observes_offroad_without_onroad_poll_and_restores(
      self.tmp_path,
      None,
    )

  def test_mid_drive_start_skips_collection_until_offroad_preparation(self):
    _test_mid_drive_start_skips_collection_until_offroad_preparation(
      self.tmp_path,
      None,
    )

  def test_car_params_precedence_fallback_and_transient_absence(self):
    _test_car_params_precedence_fallback_and_transient_absence(
      self.tmp_path,
      None,
    )

  def test_live_identity_late_and_mismatch_never_cross_contaminate(self):
    _test_live_identity_late_and_mismatch_never_cross_contaminate(
      self.tmp_path,
      None,
    )

  def test_learning_status_is_canonical_strict_and_drive_local(self):
    _test_learning_status_is_canonical_strict_and_drive_local(
      self.tmp_path,
      None,
      self,
    )

  def test_status_write_failures_and_corrupt_restore_fail_closed(self):
    _test_status_write_failures_and_corrupt_restore_fail_closed(
      self.tmp_path,
      None,
    )
