#!/usr/bin/env python3
"""Passive offroad lifecycle owner for approved modular BLaTv2 profiles.

This process publishes nothing and polls ``deviceState`` while passively
observing ``selfdriveState.enabled`` plus the exact modular-selection witness
in ``controlsState``. It stays alive across the onroad/offroad boundary so
every blocking Params mutation is performed offroad:

* validate and stage an externally-approved exact artifact;
* atomically prepare promotion or exact rollback;
* publish a profile-bound driver-feedback request after a provisional drive;
* consume only a matching explicit response.

It never constructs an approval, marks a gate passed, consumes learner
candidates, infers feedback from interventions, or publishes actuation.
Onroad work is limited to read-only runtime restoration and an in-memory
record of the already-prepared provisional profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
from collections.abc import Callable, Mapping

from opendbc.car.structs import car
from openpilot.common.basedir import BASEDIR
from openpilot.common.git import get_commit
from openpilot.selfdrive.controls.lib.blatv2.approved_artifact import (
  ApprovedArtifactReader,
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
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  build_detected_runtime_bundle,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
  RuntimeVehicleBundle,
)


PUBLISHED_SERVICES: tuple[str, ...] = ()
SUBSCRIBED_SERVICES = (
  "deviceState",
  "selfdriveState",
  "controlsState",
)
STEP_TIMEOUT_MS = 1000
PROVISIONAL_RACK_DYNAMICS_PATH = (
  Path(__file__).resolve().parent
  / "lib"
  / "blatv2"
  / "provisional_rack_dynamics.json"
)
_FULL_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


def assert_passive_process_contract(
  *,
  publishers: tuple[str, ...] = PUBLISHED_SERVICES,
  subscribers: tuple[str, ...] = SUBSCRIBED_SERVICES,
) -> None:
  assert publishers == ()
  assert subscribers == (
    "deviceState",
    "selfdriveState",
    "controlsState",
  )
  assert "carControl" not in publishers
  assert "carControl" not in subscribers


@dataclass(frozen=True, slots=True)
class RuntimeCommits:
  source_openpilot_commit: str
  opendbc_commit: str


@dataclass(slots=True)
class ProfileLifecycleContext:
  runtime_bundle: RuntimeVehicleBundle
  commits: RuntimeCommits
  reader: ApprovedArtifactReader
  activation: PersistentProfileActivation


def _exact_commit(value: object) -> str | None:
  if type(value) is not str or _FULL_COMMIT_RE.fullmatch(value) is None:
    return None
  return value


def resolve_runtime_commits(
  params: Any,
  *,
  basedir: str | Path = BASEDIR,
  repository_commit: Callable[[str | None], str] = get_commit,
) -> RuntimeCommits | None:
  """Resolve exact deployed source identities or fail closed.

  Manager's full ``GitCommit`` Param is the authoritative runtime build
  identity. When the superproject repository metadata is present, it must
  agree. Opendbc's checked-out repository HEAD is authoritative for the
  production vehicle interface used by this process. Abbreviated, missing, or
  disagreeing identities are never guessed.
  """
  try:
    runtime_source = _exact_commit(
      params.get("GitCommit", block=False),
    )
    if runtime_source is None:
      return None
    root = Path(basedir)
    if (root / ".git").exists():
      repository_source = _exact_commit(repository_commit(str(root)))
      if repository_source is None or repository_source != runtime_source:
        return None
    opendbc = _exact_commit(repository_commit(str(root / "opendbc_repo")))
  except (KeyError, TypeError, ValueError, RuntimeError, OSError):
    return None
  if opendbc is None:
    return None
  return RuntimeCommits(runtime_source, opendbc)


def build_profile_lifecycle_context(
  *,
  car_params: car.CarParams,
  params: Any,
  provisional_rack_dynamics: ProvisionalRackDynamics,
  interface_registry: Mapping[str, type] | None = None,
  commit_resolver: Callable[[Any], RuntimeCommits | None] = (
    resolve_runtime_commits
  ),
) -> ProfileLifecycleContext | None:
  """Build the same generic runtime identity used by shadow and learning."""
  commits = commit_resolver(params)
  if commits is None:
    return None
  try:
    bundle, _, _ = build_detected_runtime_bundle(
      car_params=car_params,
      provisional_rack_dynamics=provisional_rack_dynamics,
      interface_registry=interface_registry,
    )
    activation = PersistentProfileActivation(
      params,
      expected_vehicle_identity=bundle.vehicle_identity,
      expected_runtime_vehicle_identity_sha256=bundle.identity_sha256,
      expected_source_openpilot_commit=(
        commits.source_openpilot_commit
      ),
      expected_opendbc_commit=commits.opendbc_commit,
      production_envelope_verified=(
        bundle.torque_limits.production_envelope_verified
      ),
    )
  except (KeyError, TypeError, ValueError, RuntimeError, OSError):
    return None
  return ProfileLifecycleContext(
    runtime_bundle=bundle,
    commits=commits,
    reader=ApprovedArtifactReader(params),
    activation=activation,
  )


class BlatV2ProfileDaemon:
  """Finite-poll adapter for offroad-only profile lifecycle persistence."""

  def __init__(
    self,
    *,
    sm: Any | None = None,
    params: Any | None = None,
    car_params_decoder: Callable[[bytes], car.CarParams] | None = None,
    context_factory: Callable[
      [car.CarParams, Any],
      ProfileLifecycleContext | None,
    ] | None = None,
    logger: Any | None = None,
  ) -> None:
    assert_passive_process_contract()
    if sm is None or params is None or car_params_decoder is None:
      import openpilot.cereal.messaging as messaging
      from openpilot.common.params import Params

      if sm is None:
        sm = messaging.SubMaster(
          list(SUBSCRIBED_SERVICES),
          poll="deviceState",
        )
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
    self.car_params_decoder = car_params_decoder
    self.context_factory = (
      context_factory
      if context_factory is not None
      else self._production_context_factory
    )
    self.logger = logger
    self.context: ProfileLifecycleContext | None = None
    self._context_fingerprint: str | None = None
    self._onroad: bool | None = None
    self._onroad_provisional_identity: ProfileIdentity | None = None
    self._onroad_selfdrive_enabled = False
    self._onroad_profile_exercised = False
    self._completed_provisional_drive: ProfileIdentity | None = None
    self._hold_staged_until_onroad = False
    self.last_artifact_diagnostic = ArtifactDiagnostic.ABSENT

  @staticmethod
  def _production_context_factory(
    car_params: car.CarParams,
    params: Any,
  ) -> ProfileLifecycleContext | None:
    dynamics = ProvisionalRackDynamics.from_json_file(
      PROVISIONAL_RACK_DYNAMICS_PATH,
    )
    return build_profile_lifecycle_context(
      car_params=car_params,
      params=params,
      provisional_rack_dynamics=dynamics,
    )

  def _log_exception(self, message: str) -> None:
    exception = getattr(self.logger, "exception", None)
    if callable(exception):
      exception(message)

  def _read_car_params(self) -> car.CarParams | None:
    for key in ("CarParamsPersistent", "CarParams"):
      try:
        encoded = self.params.get(key, block=False)
        if encoded is not None:
          return self.car_params_decoder(encoded)
      except Exception:
        self._log_exception(
          f"blatv2 profile lifecycle could not decode {key}",
        )
    return None

  def _ensure_context(self) -> None:
    car_params = self._read_car_params()
    if car_params is None:
      return
    fingerprint = str(car_params.carFingerprint)
    if (
      self.context is not None
      and self._context_fingerprint == fingerprint
    ):
      return
    try:
      context = self.context_factory(car_params, self.params)
    except Exception:
      context = None
      self._log_exception(
        "blatv2 profile lifecycle runtime construction failed closed",
      )
    if context is None:
      return
    self.context = context
    self._context_fingerprint = fingerprint

  def _bind_onroad_identity(self) -> None:
    if self._onroad_provisional_identity is not None or self.context is None:
      return
    activation = self.context.activation
    active = activation.active_artifact
    if (
      active is not None
      and activation.provisional
      and activation.production_envelope_verified
      and not activation.rollback_pending
    ):
      self._onroad_provisional_identity = (
        ProfileIdentity.from_artifact(active)
      )

  def _observe_enabled_profile(self) -> None:
    if (
      self._onroad is True
      and self.sm.updated["selfdriveState"]
      and self.sm.valid["selfdriveState"]
      and self.sm.alive["selfdriveState"]
      and bool(self.sm["selfdriveState"].enabled)
    ):
      self._onroad_selfdrive_enabled = True
    if (
      self._onroad is not True
      or self._onroad_provisional_identity is None
      or not self._onroad_selfdrive_enabled
      or not self.sm.updated["controlsState"]
      or not self.sm.valid["controlsState"]
      or not self.sm.alive["controlsState"]
    ):
      return
    try:
      lateral = self.sm["controlsState"].lateralControlState
      if lateral.which() != "torqueState":
        return
      torque_state = lateral.torqueState
      exact_selection = (
        int(torque_state.modularSelection)
        == int(ControllerSelection.MODULAR)
      )
      exact_artifact = (
        str(torque_state.modularArtifactHash)
        == self._onroad_provisional_identity.artifact_sha256
      )
      if (
        exact_selection
        and bool(torque_state.modularSelectionBound)
        and exact_artifact
      ):
        self._onroad_profile_exercised = True
    except (AttributeError, TypeError, ValueError, OverflowError):
      return

  def _remove_param(self, key: str) -> bool:
    try:
      self.params.remove(key)
      return True
    except (KeyError, TypeError, ValueError, RuntimeError, OSError):
      return False

  def _publish_completed_drive_request(self) -> None:
    identity = self._completed_provisional_drive
    if identity is None or self.context is None:
      return
    activation = self.context.activation
    active = activation.active_artifact
    if (
      active is None
      or not activation.provisional
      or activation.rollback_pending
      or ProfileIdentity.from_artifact(active) != identity
    ):
      self._completed_provisional_drive = None
      return

    request = FeedbackRequest(
      artifact_sha256=identity.artifact_sha256,
      profile_sha256=identity.profile_sha256,
      profile_revision=identity.profile_revision,
    )
    try:
      existing = FeedbackRequest.from_param(
        self.params.get(FEEDBACK_REQUEST_PARAM, block=False),
      )
    except (KeyError, TypeError, ValueError, RuntimeError, OSError):
      existing = None
    if existing != request:
      # A response belongs to the request it names. Remove it before replacing
      # the request so stale exact feedback cannot suppress this completed
      # drive's prompt.
      self._remove_param(FEEDBACK_RESPONSE_PARAM)
      try:
        self.params.put(
          FEEDBACK_REQUEST_PARAM,
          request.to_param(),
          block=True,
        )
      except (KeyError, TypeError, ValueError, RuntimeError, OSError):
        return
    self._completed_provisional_drive = None

  def _consume_matching_feedback(self) -> FeedbackChoice | None:
    if self.context is None:
      return None
    return self.context.activation.consume_feedback(offroad=True)

  def _stage_exact_approved_artifact(self) -> None:
    if self.context is None:
      return
    context = self.context
    bundle = context.runtime_bundle
    commits = context.commits
    result = context.reader.read(
      expected_vehicle_identity=bundle.vehicle_identity,
      expected_runtime_vehicle_identity_sha256=bundle.identity_sha256,
      expected_source_openpilot_commit=commits.source_openpilot_commit,
      expected_opendbc_commit=commits.opendbc_commit,
    )
    self.last_artifact_diagnostic = result.diagnostic
    candidate = result.artifact
    if candidate is None:
      return
    if not bundle.torque_limits.production_envelope_verified:
      # Shadowing and learning remain portable, but live activation requires
      # opendbc to assert that its production CarController uses this exact
      # command envelope.
      self.last_artifact_diagnostic = (
        ArtifactDiagnostic.UNVERIFIED_ACTUATION_ENVELOPE
      )
      return

    activation = context.activation
    identity = ProfileIdentity.from_artifact(candidate)
    if identity in activation.rejected_profile_identities:
      return
    active = activation.active_artifact
    staged = activation.staged_artifact
    if active is not None and active.artifact_sha256 == candidate.artifact_sha256:
      return
    if staged is not None and staged.artifact_sha256 == candidate.artifact_sha256:
      return
    if (
      active is not None
      and candidate.vehicle_profile.revision
      <= active.vehicle_profile.revision
    ):
      return
    if (
      staged is not None
      and candidate.vehicle_profile.revision
      <= staged.vehicle_profile.revision
    ):
      return
    try:
      activation.stage(candidate, offroad=True)
    except (TypeError, ValueError, RuntimeError):
      return

  def _offroad_cycle(self) -> None:
    if self.context is None:
      return
    activation = self.context.activation
    if activation.stale_build_state:
      # Canonical state from another exact build is safe to retire offroad.
      # Malformed state never exposes this flag and remains fail-closed.
      if not activation.retire_stale_build_offroad(offroad=True):
        return
    if activation.diagnostic not in (
      ArtifactDiagnostic.OK,
      ArtifactDiagnostic.ABSENT,
    ):
      return
    # Feedback may alter or reject the active identity, so consume it before
    # publishing a new request or considering a newer artifact.
    choice = self._consume_matching_feedback()
    if choice == FeedbackChoice.WORSE or activation.rollback_pending:
      # Exact rollback is the only transition prepared in this offroad
      # session. A separately-staged revision cannot leapfrog explicit WORSE
      # feedback before the rollback target gets one real engagement.
      activation.prepare_offroad(offroad=True)
      self._hold_staged_until_onroad = True
      self._completed_provisional_drive = None
      return
    if self._hold_staged_until_onroad:
      return
    self._publish_completed_drive_request()
    self._stage_exact_approved_artifact()
    activation.prepare_offroad(offroad=True)

  def _transition(self, started: bool) -> None:
    state = bool(started)
    if self._onroad is not None and state == self._onroad:
      if state:
        self._bind_onroad_identity()
        self._observe_enabled_profile()
      return
    was_onroad = self._onroad is True
    completed_identity = self._onroad_provisional_identity
    completed_exercised = self._onroad_profile_exercised
    self._onroad = state
    if state:
      self._hold_staged_until_onroad = False
      self._onroad_provisional_identity = None
      self._onroad_selfdrive_enabled = False
      self._onroad_profile_exercised = False
      self._bind_onroad_identity()
      self._observe_enabled_profile()
    else:
      if was_onroad and completed_exercised:
        self._completed_provisional_drive = completed_identity
      self._onroad_provisional_identity = None
      self._onroad_selfdrive_enabled = False
      self._onroad_profile_exercised = False
      self._offroad_cycle()

  def step(self, timeout_ms: int = STEP_TIMEOUT_MS) -> None:
    self.sm.update(timeout_ms)
    if not (
      self.sm.seen["deviceState"]
      and self.sm.valid["deviceState"]
      and self.sm.alive["deviceState"]
    ):
      return
    self._ensure_context()
    self._transition(bool(self.sm["deviceState"].started))
    if self._onroad is False and self.sm.updated["deviceState"]:
      self._offroad_cycle()

  def run(self) -> None:
    while True:
      self.step()


def main() -> None:
  BlatV2ProfileDaemon().run()


if __name__ == "__main__":
  main()
