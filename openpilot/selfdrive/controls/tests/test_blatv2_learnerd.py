from __future__ import annotations

from dataclasses import replace
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
  LearningLifecycleState,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  LearningRestoreError,
  MeasuredLearningFrame,
  PersistentLearningRuntime,
  build_detected_runtime_bundle,
  build_persistent_learning_runtime,
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
    self.car_params_bytes = car_params_bytes

  def get_param_path(self) -> str:
    return str(self.root / "params" / "d")

  def get(self, key: str, block: bool = False):
    assert key == "CarParams"
    assert block is False
    return self.car_params_bytes


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
  logger = FakeLogger()
  created: list[PersistentLearningRuntime] = []

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
    car_params_decoder=lambda _encoded: cp,
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
  assert created[0].coordinator.state is LearningLifecycleState.OFFROAD
  assert not (tmp_path / "learning").exists()

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

  stop_messages = fake_messages(cp, started=False)
  sm.publish({
    "deviceState": (stop_messages["deviceState"], 2_000_000_000),
  })
  daemon.step()
  assert created[0].coordinator.state is LearningLifecycleState.OFFROAD
  assert created[0].artifact_paths.evidence.is_file()
  assert created[0].artifact_paths.manifest.is_file()
  assert all(timeout == 100 for timeout in sm.timeouts)
  assert logger.exceptions == []

  # A reconstructed daemon/runtime restores the exact persisted evidence.
  restored = runtime_factory(cp, tmp_path / "learning")
  assert restored.coordinator.finalize().evidence_bytes == (
    created[0].coordinator.finalize().evidence_bytes
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

  stopped = fake_messages(cp, started=False)
  sm.publish({"deviceState": (stopped["deviceState"], 2_000_000_000)})
  daemon.step()
  assert factory_calls == [True]
  assert daemon.runtime is not None
  assert daemon.runtime.coordinator.state is LearningLifecycleState.OFFROAD

  sm.publish({"deviceState": (started["deviceState"], 3_000_000_000)})
  daemon.step()
  assert daemon.runtime.coordinator.state is LearningLifecycleState.ONROAD


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
