from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

from openpilot.selfdrive.controls.blatv2_profiled import (
  PUBLISHED_SERVICES,
  STEP_TIMEOUT_MS,
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
  ProfileIdentity,
)
from openpilot.selfdrive.controls.lib.blatv2.bootstrap import (
  ControllerSelection,
)
from openpilot.selfdrive.controls.lib.blatv2.feedback import (
  FEEDBACK_REQUEST_PARAM,
  FEEDBACK_RESPONSE_PARAM,
  FeedbackChoice,
  FeedbackRequest,
  FeedbackResponse,
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
VEHICLE = "GENERIC PORTABLE TORQUE VEHICLE"
RUNTIME_HASH = "1" * 64
SOURCE_COMMIT = "2" * 40
OPENDBC_COMMIT = "3" * 40
EVIDENCE_HASH = "4" * 64
HARNESS_COMMIT = "5" * 40


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
  return ApprovedProfileArtifact(
    vehicle_profile=profile(revision),
    controller_policy=ControllerPolicy(
      revision=1,
      provenance="replay-qualified generic policy",
      provisional=False,
      natural_frequency_per_s=8.0,
      damping_ratio=1.0,
      observer_time_constant_s=None,
      observer_max_abs_disturbance_torque=None,
    ),
    runtime_vehicle_identity_sha256=runtime_hash,
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
    learner_evidence_sha256=EVIDENCE_HASH,
    replay_harness_commit=HARNESS_COMMIT,
    replay_passed=True,
    delivered_replay_passed=True,
    safety_passed=True,
    deterministic_aa_passed=True,
    device_timing_passed=True,
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
  commits = RuntimeCommits(SOURCE_COMMIT, OPENDBC_COMMIT)
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


def prepare_provisional_daemon():
  params = MemoryParams()
  params.values[APPROVED_ARTIFACT_PARAM] = artifact().to_param()
  sm = FakeSubMaster()
  daemon = daemon_for(params, sm)
  step(daemon, params, sm, started=False)
  assert daemon.context is not None
  assert daemon.context.activation.provisional
  return params, sm, daemon


def lifecycle_payload(activation):
  return build_lifecycle_status_payload(
    activation=activation,
    vehicle_identity=VEHICLE,
    runtime_identity_sha256=RUNTIME_HASH,
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
  )


def test_lifecycle_display_projection_maps_every_effective_state():
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
  ) == build_lifecycle_status_bytes(
    activation=manager,
    vehicle_identity=VEHICLE,
    runtime_identity_sha256=RUNTIME_HASH,
    source_openpilot_commit=SOURCE_COMMIT,
    opendbc_commit=OPENDBC_COMMIT,
  )

  manager.stage(artifact(), offroad=True)
  staged = lifecycle_payload(manager)
  assert staged["controller_state"] == "staged"
  assert staged["effective_controller"] == "stock"
  assert staged["active_profile"] is None
  assert staged["staged_profile"]["profile_revision"] == 1

  manager.prepare_offroad(offroad=True)
  provisional = lifecycle_payload(manager)
  assert provisional["controller_state"] == "provisional"
  assert provisional["effective_controller"] == "modular"
  assert provisional["active_profile"]["profile_revision"] == 1
  assert provisional["staged_profile"] is None

  active = manager.active_artifact
  identity = ProfileIdentity.from_artifact(active)
  request = FeedbackRequest(
    artifact_sha256=identity.artifact_sha256,
    profile_sha256=identity.profile_sha256,
    profile_revision=identity.profile_revision,
  )
  params.values[FEEDBACK_REQUEST_PARAM] = request.to_param()
  params.values[FEEDBACK_RESPONSE_PARAM] = (
    FeedbackResponse.for_request(
      request,
      FeedbackChoice.WORSE,
    ).to_param()
  )
  assert manager.consume_feedback(offroad=True) is FeedbackChoice.WORSE
  rollback = lifecycle_payload(manager)
  assert rollback["controller_state"] == "rollback_pending"
  assert rollback["effective_controller"] == "stock"
  assert rollback["rejected_profile_count"] == 1

  manager.prepare_offroad(offroad=True)
  rolled_back = lifecycle_payload(manager)
  assert rolled_back["controller_state"] == "stock"
  assert rolled_back["effective_controller"] == "stock"

  approved_params = MemoryParams()
  approved_manager = context_for(approved_params).activation
  approved_manager.stage(artifact(), offroad=True)
  approved_manager.prepare_offroad(offroad=True)
  approved_active = approved_manager.active_artifact
  approved_identity = ProfileIdentity.from_artifact(approved_active)
  approved_request = FeedbackRequest(
    artifact_sha256=approved_identity.artifact_sha256,
    profile_sha256=approved_identity.profile_sha256,
    profile_revision=approved_identity.profile_revision,
  )
  approved_params.values[FEEDBACK_REQUEST_PARAM] = approved_request.to_param()
  approved_params.values[FEEDBACK_RESPONSE_PARAM] = (
    FeedbackResponse.for_request(
      approved_request,
      FeedbackChoice.BETTER,
    ).to_param()
  )
  assert (
    approved_manager.consume_feedback(offroad=True)
    is FeedbackChoice.BETTER
  )
  approved = lifecycle_payload(approved_manager)
  assert approved["controller_state"] == "approved"
  assert approved["effective_controller"] == "modular"


def test_lifecycle_display_invalid_stale_and_unverified_fail_closed():
  malformed_params = MemoryParams()
  malformed_params.values[ACTIVATION_STATE_PARAM] = {"not": "canonical"}
  malformed = context_for(malformed_params).activation
  invalid = lifecycle_payload(malformed)
  assert invalid["controller_state"] == "unavailable"
  assert invalid["effective_controller"] == "stock"
  assert invalid["diagnostic"] == ArtifactDiagnostic.STATE_INVALID.value
  assert invalid["active_profile"] is None
  assert invalid["activation_state_sha256"] is None

  stale_params = MemoryParams()
  old_source = "6" * 40
  old_artifact = ApprovedProfileArtifact(
    vehicle_profile=artifact().vehicle_profile,
    controller_policy=artifact().controller_policy,
    runtime_vehicle_identity_sha256=RUNTIME_HASH,
    source_openpilot_commit=old_source,
    opendbc_commit=OPENDBC_COMMIT,
    learner_evidence_sha256=EVIDENCE_HASH,
    replay_harness_commit=HARNESS_COMMIT,
    replay_passed=True,
    delivered_replay_passed=True,
    safety_passed=True,
    deterministic_aa_passed=True,
    device_timing_passed=True,
  )
  old_manager = PersistentProfileActivation(
    stale_params,
    expected_vehicle_identity=VEHICLE,
    expected_runtime_vehicle_identity_sha256=RUNTIME_HASH,
    expected_source_openpilot_commit=old_source,
    expected_opendbc_commit=OPENDBC_COMMIT,
    production_envelope_verified=True,
  )
  old_manager.stage(old_artifact, offroad=True)
  old_manager.prepare_offroad(offroad=True)
  stale = context_for(stale_params).activation
  stale_status = lifecycle_payload(stale)
  assert stale_status["controller_state"] == "unavailable"
  assert stale_status["diagnostic"] == ArtifactDiagnostic.STATE_STALE_BUILD.value
  assert stale_status["effective_controller"] == "stock"

  unverified_params = MemoryParams()
  prepared = context_for(unverified_params).activation
  prepared.stage(artifact(), offroad=True)
  prepared.prepare_offroad(offroad=True)
  unverified = context_for(
    unverified_params,
    envelope_verified=False,
  ).activation
  unverified_status = lifecycle_payload(unverified)
  assert unverified_status["controller_state"] == "unavailable"
  assert unverified_status["effective_controller"] == "stock"
  assert (
    unverified_status["diagnostic"]
    == ArtifactDiagnostic.UNVERIFIED_ACTUATION_ENVELOPE.value
  )
  assert unverified_status["active_profile"] is None


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
  assert lifecycle_puts[-1][1]["controller_state"] == "provisional"


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


def test_exact_lifecycle_is_prepared_offroad_and_live_path_is_read_only():
  params, sm, daemon = prepare_provisional_daemon()
  activation = daemon.context.activation
  display = params.values[LIFECYCLE_STATUS_PARAM]
  assert type(display) is dict
  assert display["controller_state"] == "provisional"
  assert display["effective_controller"] == "modular"
  assert params.puts[-1][0] == LIFECYCLE_STATUS_PARAM
  assert params.puts[-1][2] is True
  selected = activation.active_artifact
  assert selected is not None
  assert activation.staged_artifact is None

  puts_before_drive = len(params.puts)
  step(daemon, params, sm, started=True, enabled=False)
  step(daemon, params, sm, enabled=True)
  step(daemon, params, sm, enabled=True)
  assert len(params.puts) == puts_before_drive
  assert params.onroad_writes == []

  step(daemon, params, sm, started=False)
  request = FeedbackRequest.from_param(
    params.values[FEEDBACK_REQUEST_PARAM],
  )
  assert request.profile_sha256 == selected.vehicle_profile_sha256
  assert request.profile_revision == selected.vehicle_profile.revision
  assert all(timeout == STEP_TIMEOUT_MS for timeout in sm.timeouts)

  # The same state object subsequently binds without touching Params.
  put_count = len(params.puts)
  decision = activation.begin_engagement()
  assert decision.selection is ControllerSelection.MODULAR
  assert decision.artifact == selected
  activation.end_engagement()
  assert len(params.puts) == put_count


def test_onroad_period_without_enabled_lateral_does_not_prompt():
  params, sm, daemon = prepare_provisional_daemon()
  step(daemon, params, sm, started=True, enabled=False)
  step(daemon, params, sm, enabled=False)
  step(daemon, params, sm, started=False)
  assert FEEDBACK_REQUEST_PARAM not in params.values


def test_enabled_but_forced_stock_drive_does_not_prompt():
  params, sm, daemon = prepare_provisional_daemon()
  step(
    daemon,
    params,
    sm,
    started=True,
    enabled=True,
    modular=False,
  )
  step(daemon, params, sm, enabled=True, modular=False)
  step(daemon, params, sm, started=False)
  assert FEEDBACK_REQUEST_PARAM not in params.values


def test_modular_telemetry_for_another_artifact_does_not_prompt():
  params, sm, daemon = prepare_provisional_daemon()
  step(
    daemon,
    params,
    sm,
    started=True,
    enabled=True,
    modular=True,
    artifact_hash="f" * 64,
  )
  step(daemon, params, sm, started=False)
  assert FEEDBACK_REQUEST_PARAM not in params.values


def _feedback_outcome_is_exact_and_offroad(choice):
  params, sm, daemon = prepare_provisional_daemon()
  step(daemon, params, sm, started=True, enabled=True)
  step(daemon, params, sm, started=False)
  request = FeedbackRequest.from_param(
    params.values[FEEDBACK_REQUEST_PARAM],
  )
  params.values[FEEDBACK_RESPONSE_PARAM] = (
    FeedbackResponse.for_request(request, choice).to_param()
  )
  step(daemon, params, sm, started=False)

  activation = daemon.context.activation
  assert FEEDBACK_REQUEST_PARAM not in params.values
  assert FEEDBACK_RESPONSE_PARAM not in params.values
  if choice is FeedbackChoice.WORSE:
    assert activation.active_artifact is None
    assert not activation.rollback_pending
    assert len(activation.rejected_profile_identities) == 1
    assert (
      activation.begin_engagement().selection
      is ControllerSelection.STOCK
    )
    activation.end_engagement()
  elif choice in (FeedbackChoice.BETTER, FeedbackChoice.ABOUT_SAME):
    assert activation.active_artifact is not None
    assert not activation.provisional
  else:
    assert activation.active_artifact is not None
    assert activation.provisional
    # NOT_SURE permits a fresh prompt only after another enabled drive.
    step(daemon, params, sm, started=True, enabled=True)
    step(daemon, params, sm, started=False)
    assert FeedbackRequest.from_param(
      params.values[FEEDBACK_REQUEST_PARAM],
    ) == request


def test_all_feedback_outcomes_are_exact_and_offroad():
  for choice in FeedbackChoice:
    _feedback_outcome_is_exact_and_offroad(choice)


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
    is ArtifactDiagnostic.UNVERIFIED_ACTUATION_ENVELOPE
  )


def test_daemon_retires_canonical_old_build_then_prepares_current_artifact():
  params = MemoryParams()
  old_source = "6" * 40
  old_artifact = copy.copy(artifact())
  old_artifact = ApprovedProfileArtifact(
    vehicle_profile=old_artifact.vehicle_profile,
    controller_policy=old_artifact.controller_policy,
    runtime_vehicle_identity_sha256=RUNTIME_HASH,
    source_openpilot_commit=old_source,
    opendbc_commit=OPENDBC_COMMIT,
    learner_evidence_sha256=EVIDENCE_HASH,
    replay_harness_commit=HARNESS_COMMIT,
    replay_passed=True,
    delivered_replay_passed=True,
    safety_passed=True,
    deterministic_aa_passed=True,
    device_timing_passed=True,
  )
  old_manager = PersistentProfileActivation(
    params,
    expected_vehicle_identity=VEHICLE,
    expected_runtime_vehicle_identity_sha256=RUNTIME_HASH,
    expected_source_openpilot_commit=old_source,
    expected_opendbc_commit=OPENDBC_COMMIT,
    production_envelope_verified=True,
  )
  old_manager.stage(old_artifact, offroad=True)
  old_manager.prepare_offroad(offroad=True)
  params.values[APPROVED_ARTIFACT_PARAM] = artifact().to_param()

  sm = FakeSubMaster()
  daemon = daemon_for(params, sm)
  step(daemon, params, sm, started=False)
  activation = daemon.context.activation
  assert not activation.stale_build_state
  assert activation.active_artifact == artifact()
  assert activation.provisional


def test_rejected_exact_identity_never_reactivates_from_approved_param():
  params, sm, daemon = prepare_provisional_daemon()
  selected = daemon.context.activation.active_artifact
  step(daemon, params, sm, started=True, enabled=True)
  step(daemon, params, sm, started=False)
  request = FeedbackRequest.from_param(
    params.values[FEEDBACK_REQUEST_PARAM],
  )
  params.values[FEEDBACK_RESPONSE_PARAM] = (
    FeedbackResponse.for_request(
      request,
      FeedbackChoice.WORSE,
    ).to_param()
  )
  step(daemon, params, sm, started=False)
  identity = ProfileIdentity.from_artifact(selected)
  assert identity in daemon.context.activation.rejected_profile_identities
  for _ in range(3):
    step(daemon, params, sm, started=False)
  assert daemon.context.activation.active_artifact is None


def test_worse_restores_the_exact_previous_artifact_before_next_drive():
  params = MemoryParams()
  manager = context_for(params).activation
  first = artifact(1)
  second = artifact(2)
  manager.stage(first, offroad=True)
  manager.prepare_offroad(offroad=True)
  first_request = FeedbackRequest(
    first.artifact_sha256,
    first.vehicle_profile_sha256,
    first.vehicle_profile.revision,
  )
  params.values[FEEDBACK_REQUEST_PARAM] = first_request.to_param()
  params.values[FEEDBACK_RESPONSE_PARAM] = (
    FeedbackResponse.for_request(
      first_request,
      FeedbackChoice.BETTER,
    ).to_param()
  )
  assert manager.consume_feedback(offroad=True) is FeedbackChoice.BETTER
  manager.stage(second, offroad=True)
  manager.prepare_offroad(offroad=True)
  assert manager.active_artifact == second

  params.values[APPROVED_ARTIFACT_PARAM] = second.to_param()
  sm = FakeSubMaster()
  daemon = daemon_for(params, sm)
  step(
    daemon,
    params,
    sm,
    started=True,
    enabled=True,
    artifact_hash=second.artifact_sha256,
  )
  step(daemon, params, sm, started=False)
  request = FeedbackRequest.from_param(
    params.values[FEEDBACK_REQUEST_PARAM],
  )
  params.values[FEEDBACK_RESPONSE_PARAM] = (
    FeedbackResponse.for_request(
      request,
      FeedbackChoice.WORSE,
    ).to_param()
  )
  step(daemon, params, sm, started=False)
  assert daemon.context.activation.active_artifact == first
  decision = daemon.context.activation.begin_engagement()
  assert decision.artifact == first
  assert decision.selection is ControllerSelection.MODULAR
  daemon.context.activation.end_engagement()


def test_daemon_restart_mid_enabled_drive_publishes_request_on_stop():
  params = MemoryParams()
  manager = context_for(params).activation
  manager.stage(artifact(), offroad=True)
  manager.prepare_offroad(offroad=True)
  assert manager.provisional

  sm = FakeSubMaster()
  restarted = daemon_for(params, sm)
  step(restarted, params, sm, started=True, enabled=True)
  assert restarted.context.activation.provisional
  assert params.onroad_writes == []
  step(restarted, params, sm, started=False)
  assert FeedbackRequest.from_param(
    params.values[FEEDBACK_REQUEST_PARAM],
  ).profile_sha256 == artifact().vehicle_profile_sha256


def test_runtime_commit_resolution_requires_full_exact_agreement(tmp_path):
  params = MemoryParams()
  (tmp_path / ".git").mkdir()
  (tmp_path / "opendbc_repo").mkdir()
  commits = {
    str(tmp_path): SOURCE_COMMIT,
    str(tmp_path / "opendbc_repo"): OPENDBC_COMMIT,
  }
  resolved = resolve_runtime_commits(
    params,
    basedir=tmp_path,
    repository_commit=lambda path: commits[path],
  )
  assert resolved == RuntimeCommits(SOURCE_COMMIT, OPENDBC_COMMIT)

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


def test_process_is_passive_portable_and_registered_across_road_state():
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
  assert "def blatv2_learning(started: bool, params: Params, CP: car.CarParams)" in process_config
  assert "return not CP.notCar" in process_config
  assert (
    "".join((
      'PythonProcess("blatv2_profiled", ',
      '"openpilot.selfdrive.controls.blatv2_profiled", ',
      "blatv2_learning, restart_if_crash=True)",
    ))
  ) in process_config


def test_activation_state_writes_are_offroad_and_atomic_json():
  params, sm, daemon = prepare_provisional_daemon()
  activation_puts = [
    put for put in params.puts if put[0] == ACTIVATION_STATE_PARAM
  ]
  assert activation_puts
  assert all(block is True for _, _, block in activation_puts)
  assert all(type(payload) is dict for _, payload, _ in activation_puts)
