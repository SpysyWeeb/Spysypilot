from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

from openpilot.selfdrive.controls.blatv2_profiled import (
  PUBLISHED_SERVICES,
  SUBSCRIBED_SERVICES,
  BlatV2ProfileDaemon,
  ProfileLifecycleContext,
  RuntimeCommits,
  assert_passive_process_contract,
  resolve_runtime_commits,
)
from openpilot.selfdrive.controls.lib.blatv2.approved_artifact import (
  ACTIVATION_STATE_PARAM,
  APPROVED_ARTIFACT_PARAM,
  ApprovedArtifactReader,
  ApprovedProfileArtifact,
  ArtifactDiagnostic,
  PersistentProfileActivation,
)
from openpilot.selfdrive.controls.lib.blatv2.bootstrap import (
  ControllerSelection,
)
from openpilot.selfdrive.controls.lib.blatv2.lifecycle_status import (
  LIFECYCLE_STATUS_PARAM,
  build_lifecycle_status_bytes,
  build_lifecycle_status_payload,
)
from openpilot.selfdrive.controls.lib.blatv2.policy import ControllerPolicy
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  PhysicalParameters,
  ProfileNode,
  VehicleProfile,
)
from openpilot.selfdrive.controls.tests.blatv2_artifact_test_helpers import (
  calibration_profile_for_controller,
  calibration_selection_manifest,
  passing_device_acceptance_receipt,
  passed_behavior_finalization,
)


VEHICLE = "GENERIC PORTABLE TORQUE VEHICLE"
RUNTIME_HASH = "1" * 64
SOURCE_COMMIT = "2" * 40
OPENDBC_COMMIT = "3" * 40
PANDA_COMMIT = "4" * 40
EVIDENCE_HASH = "4" * 64
CALIBRATION_MANIFEST_HASH = "5" * 64
HARNESS_COMMIT = "5" * 40
HORIZON_POLICY_HASH = "6" * 64


class MemoryParams:
  def __init__(self):
    self.values = {
      "CarParamsPersistent": b"generic-car-params",
      "GitCommit": SOURCE_COMMIT,
    }
    self.puts: list[tuple[str, object, bool]] = []
    self.removes: list[str] = []
    self.onroad = False
    self.onroad_writes: list[str] = []

  def get(self, key, block=False):
    assert block is False
    return copy.deepcopy(self.values.get(key))

  def put(self, key, value, block=False):
    if self.onroad:
      self.onroad_writes.append(key)
      raise AssertionError("profile daemon attempted an onroad Params write")
    if key == LIFECYCLE_STATUS_PARAM:
      assert type(value) is dict
    self.values[key] = copy.deepcopy(value)
    self.puts.append((key, copy.deepcopy(value), block))

  def remove(self, key):
    if self.onroad:
      self.onroad_writes.append(key)
      raise AssertionError("profile daemon attempted an onroad Params remove")
    self.values.pop(key, None)
    self.removes.append(key)


class FakeSubMaster:
  def __init__(self):
    self.data = {
      "deviceState": SimpleNamespace(started=False),
      "selfdriveState": SimpleNamespace(enabled=False),
      "controlsState": SimpleNamespace(),
    }
    self.seen = dict.fromkeys(SUBSCRIBED_SERVICES, False)
    self.valid = dict.fromkeys(SUBSCRIBED_SERVICES, False)
    self.alive = dict.fromkeys(SUBSCRIBED_SERVICES, False)
    self.updated = dict.fromkeys(SUBSCRIBED_SERVICES, False)
    self.timeouts: list[int] = []

  def __getitem__(self, service):
    return self.data[service]

  def update(self, timeout):
    self.timeouts.append(timeout)

  def publish(
    self,
    *,
    started: bool | None = None,
    enabled: bool | None = None,
    modular: bool | None = None,
    artifact_hash: str | None = None,
  ):
    self.updated = dict.fromkeys(SUBSCRIBED_SERVICES, False)
    if started is not None:
      self.data["deviceState"] = SimpleNamespace(started=started)
      self.seen["deviceState"] = True
      self.valid["deviceState"] = True
      self.alive["deviceState"] = True
      self.updated["deviceState"] = True
    if enabled is not None:
      self.data["selfdriveState"] = SimpleNamespace(enabled=enabled)
      self.seen["selfdriveState"] = True
      self.valid["selfdriveState"] = True
      self.alive["selfdriveState"] = True
      self.updated["selfdriveState"] = True
    if modular is not None:
      torque_state = SimpleNamespace(
        modularSelection=(
          int(ControllerSelection.MODULAR)
          if modular
          else int(ControllerSelection.STOCK)
        ),
        modularSelectionBound=modular,
        modularArtifactHash=(
          artifact().artifact_sha256
          if artifact_hash is None
          else artifact_hash
        ),
      )
      lateral_state = SimpleNamespace(
        torqueState=torque_state,
        which=lambda: "torqueState",
      )
      self.data["controlsState"] = SimpleNamespace(
        lateralControlState=lateral_state,
      )
      self.seen["controlsState"] = True
      self.valid["controlsState"] = True
      self.alive["controlsState"] = True
      self.updated["controlsState"] = True


class FakeLogger:
  def __init__(self):
    self.exceptions: list[str] = []

  def exception(self, message):
    self.exceptions.append(message)


def profile(revision=1):
  parameters = PhysicalParameters(
    torque_per_lateral_accel=0.31,
    rack_gain_deg_s2_per_torque=1600.0,
    rack_damping_per_s=7.0,
    transport_delay_s=0.12,
    static_friction_torque=0.09,
    kinetic_friction_torque=0.03,
    rack_rate_resolution_deg_s=4.0,
    confidence=0.9,
    qualified=True,
  )
  return VehicleProfile(
    vehicle_identity=VEHICLE,
    revision=revision,
    provenance="qualified generic evidence",
    nodes=(
      ProfileNode(0.0, parameters, 180.0, 18000, 4000, 0.02),
      ProfileNode(30.0, parameters, 600.0, 60000, 12000, 0.03),
    ),
  )


def artifact(revision=1, *, runtime_hash=RUNTIME_HASH):
  selected_profile = profile(revision)
  calibration_profile = calibration_profile_for_controller(selected_profile)
  selected_policy = ControllerPolicy(
    revision=1,
    provenance="replay-qualified generic policy",
    provisional=False,
    natural_frequency_per_s=8.0,
    damping_ratio=1.0,
    observer_time_constant_s=None,
    observer_max_abs_disturbance_torque=None,
  )
  profile_hash = hashlib.sha256(
    selected_profile.to_json().encode(),
  ).hexdigest()
  return ApprovedProfileArtifact(
    vehicle_profile=selected_profile,
    calibration_profile=calibration_profile,
    controller_policy=selected_policy,
    horizon_policy_sha256=HORIZON_POLICY_HASH,
    runtime_vehicle_identity_sha256=runtime_hash,
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
    panda_commit=PANDA_COMMIT,
    calibration_selection_manifest=calibration_selection_manifest(
      selected_profile,
      learner_evidence_sha256=EVIDENCE_HASH,
      qualification_manifest_sha256=CALIBRATION_MANIFEST_HASH,
      calibration_profile=calibration_profile,
    ),
    behavior_finalization=passed_behavior_finalization(selected_policy),
    replay_harness_commit=HARNESS_COMMIT,
    replay_passed=True,
    delivered_replay_passed=True,
    deterministic_aa_passed=True,
    device_acceptance_receipt=passing_device_acceptance_receipt(
      vehicle_identity=selected_profile.vehicle_identity,
      runtime_identity_sha256=runtime_hash,
      profile_sha256=profile_hash,
      controller_policy_sha256=selected_policy.sha256,
      horizon_policy_sha256=HORIZON_POLICY_HASH,
      source_openpilot_commit=SOURCE_COMMIT,
      opendbc_commit=OPENDBC_COMMIT,
      panda_commit=PANDA_COMMIT,
    ),
    smooth_passed=True,
    swift_passed=True,
    strong_passed=True,
  )


def context_for(
  params,
  *,
  envelope_verified=True,
):
  runtime_bundle = SimpleNamespace(
    vehicle_identity=VEHICLE,
    identity_sha256=RUNTIME_HASH,
    torque_limits=SimpleNamespace(
      production_envelope_verified=envelope_verified,
    ),
  )
  commits = RuntimeCommits(SOURCE_COMMIT, OPENDBC_COMMIT, PANDA_COMMIT)
  return ProfileLifecycleContext(
    runtime_bundle=runtime_bundle,
    commits=commits,
    reader=ApprovedArtifactReader(params),
    activation=PersistentProfileActivation(
      params,
      expected_vehicle_identity=VEHICLE,
      expected_runtime_vehicle_identity_sha256=RUNTIME_HASH,
      expected_source_openpilot_commit=SOURCE_COMMIT,
      expected_opendbc_commit=OPENDBC_COMMIT,
      expected_panda_commit=PANDA_COMMIT,
      production_envelope_verified=envelope_verified,
    ),
  )


def daemon_for(
  params,
  sm,
  *,
  envelope_verified=True,
):
  generic_cp = SimpleNamespace(carFingerprint=VEHICLE)
  return BlatV2ProfileDaemon(
    sm=sm,
    params=params,
    car_params_decoder=lambda _encoded: generic_cp,
    context_factory=lambda _cp, lifecycle_params: context_for(
      lifecycle_params,
      envelope_verified=envelope_verified,
    ),
    logger=FakeLogger(),
  )


def step(
  daemon,
  params,
  sm,
  *,
  started=None,
  enabled=None,
  modular=None,
  artifact_hash=None,
):
  if started is not None:
    params.onroad = started
  if enabled is True and modular is None:
    modular = True
  sm.publish(
    started=started,
    enabled=enabled,
    modular=modular,
    artifact_hash=artifact_hash,
  )
  daemon.step()


def lifecycle_payload(activation):
  return build_lifecycle_status_payload(
    activation=activation,
    vehicle_identity=VEHICLE,
    runtime_identity_sha256=RUNTIME_HASH,
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
    panda_commit=PANDA_COMMIT,
  )


def test_lifecycle_display_projection_maps_current_stock_state():
  params = MemoryParams()
  manager = context_for(params).activation
  stock = lifecycle_payload(manager)
  assert stock["controller_state"] == "stock"
  assert stock["effective_controller"] == "stock"
  assert stock["active_profile"] is None
  assert stock["staged_profile"] is None
  assert stock["activation_state_sha256"] == manager.state_sha256
  assert build_lifecycle_status_bytes(
    activation=manager,
    vehicle_identity=VEHICLE,
    runtime_identity_sha256=RUNTIME_HASH,
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
    panda_commit=PANDA_COMMIT,
  ) == build_lifecycle_status_bytes(
    activation=manager,
    vehicle_identity=VEHICLE,
    runtime_identity_sha256=RUNTIME_HASH,
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
    panda_commit=PANDA_COMMIT,
  )


def test_lifecycle_display_invalid_and_unapproved_fail_closed():
  malformed_params = MemoryParams()
  malformed_params.values[ACTIVATION_STATE_PARAM] = {"not": "canonical"}
  malformed = context_for(malformed_params).activation
  invalid = lifecycle_payload(malformed)
  assert invalid["controller_state"] == "unavailable"
  assert invalid["effective_controller"] == "stock"
  assert invalid["diagnostic"] == ArtifactDiagnostic.STATE_INVALID.value
  assert invalid["active_profile"] is None
  assert invalid["activation_state_sha256"] is None

  blocked_params = MemoryParams()
  blocked_params.values[APPROVED_ARTIFACT_PARAM] = artifact().to_param()
  sm = FakeSubMaster()
  blocked = daemon_for(blocked_params, sm)
  step(blocked, blocked_params, sm, started=False)
  assert blocked.context.activation.active_artifact is None
  assert (
    blocked.last_artifact_diagnostic
    is ArtifactDiagnostic.EXTERNAL_SAFETY_AUTHORITY_UNAVAILABLE
  )
  status = blocked_params.values[LIFECYCLE_STATUS_PARAM]
  assert status["controller_state"] == "unavailable"
  assert status["effective_controller"] == "stock"
  assert (
    status["diagnostic"]
    == ArtifactDiagnostic.EXTERNAL_SAFETY_AUTHORITY_UNAVAILABLE.value
  )
  assert ACTIVATION_STATE_PARAM not in blocked_params.values


def test_lifecycle_cache_skips_unchanged_offroad_writes_but_tracks_transition():
  params = MemoryParams()
  sm = FakeSubMaster()
  daemon = daemon_for(params, sm)
  step(daemon, params, sm, started=False)
  lifecycle_puts = [
    put for put in params.puts if put[0] == LIFECYCLE_STATUS_PARAM
  ]
  assert len(lifecycle_puts) == 1
  assert lifecycle_puts[0][1]["controller_state"] == "stock"

  for _ in range(3):
    step(daemon, params, sm, started=False)
  assert len([
    put for put in params.puts if put[0] == LIFECYCLE_STATUS_PARAM
  ]) == 1

  params.values[APPROVED_ARTIFACT_PARAM] = artifact().to_param()
  step(daemon, params, sm, started=False)
  lifecycle_puts = [
    put for put in params.puts if put[0] == LIFECYCLE_STATUS_PARAM
  ]
  assert len(lifecycle_puts) == 2
  assert lifecycle_puts[-1][1]["controller_state"] == "unavailable"
  assert (
    lifecycle_puts[-1][1]["diagnostic"]
    == ArtifactDiagnostic.EXTERNAL_SAFETY_AUTHORITY_UNAVAILABLE.value
  )
  assert (
    daemon.last_artifact_diagnostic
    is ArtifactDiagnostic.EXTERNAL_SAFETY_AUTHORITY_UNAVAILABLE
  )
  assert ACTIVATION_STATE_PARAM not in params.values


def _fingerprint_daemon(params, sm, *, fail_fingerprint):
  def decode(encoded):
    return SimpleNamespace(carFingerprint=encoded.decode("ascii"))

  def context_factory(cp, lifecycle_params):
    if cp.carFingerprint == fail_fingerprint:
      return None
    return context_for(lifecycle_params)

  return BlatV2ProfileDaemon(
    sm=sm,
    params=params,
    car_params_decoder=decode,
    context_factory=context_factory,
    logger=FakeLogger(),
  )


def test_known_fingerprint_failure_clears_old_context_and_cache_offroad():
  params = MemoryParams()
  params.values["CarParamsPersistent"] = b"vehicle-a"
  sm = FakeSubMaster()
  daemon = _fingerprint_daemon(
    params,
    sm,
    fail_fingerprint="vehicle-b",
  )
  step(daemon, params, sm, started=False)
  original_context = daemon.context
  original_status = copy.deepcopy(params.values[LIFECYCLE_STATUS_PARAM])

  # Temporary loss of both identities preserves the already-validated owner
  # and its cache.
  params.values.pop("CarParamsPersistent")
  params.values.pop("CarParams", None)
  step(daemon, params, sm, started=False)
  assert daemon.context is original_context
  assert params.values[LIFECYCLE_STATUS_PARAM] == original_status

  # A definitively decoded different fingerprint that cannot construct a
  # current owner invalidates both.
  params.values["CarParamsPersistent"] = b"vehicle-b"
  step(daemon, params, sm, started=False)
  assert daemon.context is None
  assert LIFECYCLE_STATUS_PARAM not in params.values
  assert LIFECYCLE_STATUS_PARAM in params.removes
  assert daemon._last_lifecycle_status_bytes is None


def test_same_fingerprint_changed_exact_carparams_rebuilds_context():
  params = MemoryParams()
  params.values["CarParamsPersistent"] = b"same-model-runtime-a"
  sm = FakeSubMaster()
  factory_inputs: list[bytes] = []

  def context_factory(_cp, lifecycle_params):
    factory_inputs.append(lifecycle_params.values["CarParamsPersistent"])
    return context_for(lifecycle_params)

  daemon = BlatV2ProfileDaemon(
    sm=sm,
    params=params,
    car_params_decoder=lambda _encoded: SimpleNamespace(
      carFingerprint="SAME MODEL",
    ),
    context_factory=context_factory,
    logger=FakeLogger(),
  )
  step(daemon, params, sm, started=False)
  first_context = daemon.context
  assert factory_inputs == [b"same-model-runtime-a"]
  assert daemon._context_car_params_bytes == b"same-model-runtime-a"

  params.values["CarParamsPersistent"] = b"same-model-runtime-b"
  step(daemon, params, sm, started=False)
  assert factory_inputs == [
    b"same-model-runtime-a",
    b"same-model-runtime-b",
  ]
  assert daemon.context is not first_context
  assert daemon._context_car_params_bytes == b"same-model-runtime-b"
  assert LIFECYCLE_STATUS_PARAM in params.removes
  lifecycle_puts = [
    put for put in params.puts if put[0] == LIFECYCLE_STATUS_PARAM
  ]
  assert len(lifecycle_puts) == 2


def test_restart_factory_failure_removes_preseeded_stale_lifecycle_cache():
  params = MemoryParams()
  params.values["CarParamsPersistent"] = b"vehicle-b"
  params.values[LIFECYCLE_STATUS_PARAM] = {"stale": "display cache"}
  sm = FakeSubMaster()
  daemon = _fingerprint_daemon(
    params,
    sm,
    fail_fingerprint="vehicle-b",
  )
  step(daemon, params, sm, started=False)
  assert daemon.context is None
  assert LIFECYCLE_STATUS_PARAM not in params.values
  assert params.removes == [LIFECYCLE_STATUS_PARAM]


def test_known_identity_failure_onroad_defers_cache_clear_until_offroad():
  params = MemoryParams()
  params.values["CarParamsPersistent"] = b"vehicle-a"
  sm = FakeSubMaster()
  daemon = _fingerprint_daemon(
    params,
    sm,
    fail_fingerprint="vehicle-b",
  )
  step(daemon, params, sm, started=False)
  status = copy.deepcopy(params.values[LIFECYCLE_STATUS_PARAM])
  step(daemon, params, sm, started=True)

  params.values["CarParamsPersistent"] = b"vehicle-b"
  step(daemon, params, sm, started=True)
  assert daemon.context is None
  assert params.values[LIFECYCLE_STATUS_PARAM] == status
  assert params.onroad_writes == []
  assert daemon._lifecycle_status_clear_pending

  step(daemon, params, sm, started=False)
  assert LIFECYCLE_STATUS_PARAM not in params.values
  assert params.onroad_writes == []
  assert not daemon._lifecycle_status_clear_pending


def test_stale_mismatched_malformed_and_unverified_artifacts_remain_stock():
  malformed_params = MemoryParams()
  malformed_params.values[APPROVED_ARTIFACT_PARAM] = {
    "automaticApproval": True,
  }
  sm = FakeSubMaster()
  malformed = daemon_for(malformed_params, sm)
  step(malformed, malformed_params, sm, started=False)
  assert malformed.context.activation.active_artifact is None
  assert malformed.last_artifact_diagnostic is ArtifactDiagnostic.MALFORMED

  mismatch_params = MemoryParams()
  mismatch_params.values[APPROVED_ARTIFACT_PARAM] = artifact(
    runtime_hash="a" * 64,
  ).to_param()
  sm = FakeSubMaster()
  mismatch = daemon_for(mismatch_params, sm)
  step(mismatch, mismatch_params, sm, started=False)
  assert mismatch.context.activation.active_artifact is None
  assert (
    mismatch.last_artifact_diagnostic
    is ArtifactDiagnostic.RUNTIME_VEHICLE_MISMATCH
  )

  unverified_params = MemoryParams()
  unverified_params.values[APPROVED_ARTIFACT_PARAM] = artifact().to_param()
  sm = FakeSubMaster()
  unverified = daemon_for(
    unverified_params,
    sm,
    envelope_verified=False,
  )
  step(unverified, unverified_params, sm, started=False)
  assert unverified.context.activation.active_artifact is None
  assert (
    unverified.last_artifact_diagnostic
    is ArtifactDiagnostic.EXTERNAL_SAFETY_AUTHORITY_UNAVAILABLE
  )


def test_runtime_commit_resolution_requires_full_exact_agreement(tmp_path):
  params = MemoryParams()
  (tmp_path / ".git").mkdir()
  (tmp_path / "opendbc_repo").mkdir()
  (tmp_path / "panda").mkdir()
  commits = {
    str(tmp_path): SOURCE_COMMIT,
    str(tmp_path / "opendbc_repo"): OPENDBC_COMMIT,
    str(tmp_path / "panda"): PANDA_COMMIT,
  }
  resolved = resolve_runtime_commits(
    params,
    basedir=tmp_path,
    repository_commit=lambda path: commits[path],
  )
  assert resolved == RuntimeCommits(SOURCE_COMMIT, OPENDBC_COMMIT, PANDA_COMMIT)

  params.values["GitCommit"] = SOURCE_COMMIT[:-1]
  assert resolve_runtime_commits(
    params,
    basedir=tmp_path,
    repository_commit=lambda path: commits[path],
  ) is None
  params.values["GitCommit"] = SOURCE_COMMIT
  commits[str(tmp_path)] = "9" * 40
  assert resolve_runtime_commits(
    params,
    basedir=tmp_path,
    repository_commit=lambda path: commits[path],
  ) is None


def test_unresolved_runtime_identity_never_stages_or_writes():
  params = MemoryParams()
  params.values[APPROVED_ARTIFACT_PARAM] = artifact().to_param()
  sm = FakeSubMaster()
  generic_cp = SimpleNamespace(carFingerprint=VEHICLE)
  daemon = BlatV2ProfileDaemon(
    sm=sm,
    params=params,
    car_params_decoder=lambda _encoded: generic_cp,
    context_factory=lambda _cp, _params: None,
    logger=FakeLogger(),
  )
  step(daemon, params, sm, started=False)
  assert daemon.context is None
  assert params.puts == []
  assert ACTIVATION_STATE_PARAM not in params.values


def test_process_is_passive_portable_and_not_manager_registered():
  assert_passive_process_contract()
  assert PUBLISHED_SERVICES == ()
  assert SUBSCRIBED_SERVICES == (
    "deviceState",
    "selfdriveState",
    "controlsState",
  )
  source = (
    Path(__file__).resolve().parents[1] / "blatv2_profiled.py"
  ).read_text()
  assert "hyundai" not in source.lower()
  assert "carControl" not in SUBSCRIBED_SERVICES

  process_config = (
    Path(__file__).resolve().parents[3]
    / "system"
    / "manager"
    / "process_config.py"
  ).read_text()
  assert 'PythonProcess("blatv2_profiled",' not in process_config
