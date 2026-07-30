"""Validated, display-only projection of BLaTv2 profile lifecycle state.

Only ``blatv2_profiled`` may write this cache, and only while offroad after
``PersistentProfileActivation`` has validated the authoritative activation
state against the current vehicle, runtime, source, and opendbc identities.
The cache is never read by controller selection, approval, feedback, rollback,
or safety code.  The authoritative state remains ``BLaTv2ActivationState``.
"""

from __future__ import annotations

from enum import StrEnum
import json
import re

from openpilot.selfdrive.controls.lib.blatv2.approved_artifact import (
  ArtifactDiagnostic,
  PersistentProfileActivation,
  ProfileIdentity,
)


LIFECYCLE_STATUS_PARAM = "BLaTv2LifecycleStatus"
LIFECYCLE_STATUS_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class DisplayControllerState(StrEnum):
  STOCK = "stock"
  STAGED = "staged"
  PROVISIONAL = "provisional"
  APPROVED = "approved"
  ROLLBACK_PENDING = "rollback_pending"
  UNAVAILABLE = "unavailable"


class EffectiveController(StrEnum):
  STOCK = "stock"
  MODULAR = "modular"


def canonical_lifecycle_status_bytes(payload: object) -> bytes:
  """Return the canonical cache identity for write de-duplication/tests."""
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  ).encode("utf-8")


def _identity_payload(artifact) -> dict[str, object] | None:
  if artifact is None:
    return None
  identity = ProfileIdentity.from_artifact(artifact)
  return {
    "artifact_sha256": identity.artifact_sha256,
    "profile_revision": identity.profile_revision,
    "profile_sha256": identity.profile_sha256,
  }


def build_lifecycle_status_payload(
  *,
  activation: PersistentProfileActivation,
  vehicle_identity: str,
  runtime_identity_sha256: str,
  source_openpilot_commit: str,
  opendbc_commit: str,
) -> dict[str, object]:
  """Map one already-validated activation owner into an inert UI snapshot."""
  if type(vehicle_identity) is not str or not vehicle_identity.strip():
    raise ValueError("lifecycle display vehicle identity must be nonempty")
  if (
    type(runtime_identity_sha256) is not str
    or _SHA256_RE.fullmatch(runtime_identity_sha256) is None
  ):
    raise ValueError("lifecycle runtime identity must be a SHA-256")
  for value, name in (
    (source_openpilot_commit, "source openpilot commit"),
    (opendbc_commit, "opendbc commit"),
  ):
    if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
      raise ValueError(f"lifecycle {name} must be an exact commit")

  diagnostic = activation.diagnostic
  active = activation.active_artifact
  staged = activation.staged_artifact
  valid_diagnostic = diagnostic in (
    ArtifactDiagnostic.OK,
    ArtifactDiagnostic.ABSENT,
  )
  envelope_verified = activation.production_envelope_verified
  effective = EffectiveController.STOCK
  if not valid_diagnostic:
    state = DisplayControllerState.UNAVAILABLE
    active = None
    staged = None
  elif active is not None and not envelope_verified:
    # A validated artifact without a production envelope is still ineligible
    # for actuation and must never receive an active-looking badge.
    diagnostic = ArtifactDiagnostic.UNVERIFIED_ACTUATION_ENVELOPE
    state = DisplayControllerState.UNAVAILABLE
    active = None
  elif activation.rollback_pending:
    # begin_engagement deliberately refuses modular while rollback is pending.
    state = DisplayControllerState.ROLLBACK_PENDING
  elif active is not None and activation.provisional:
    state = DisplayControllerState.PROVISIONAL
    effective = EffectiveController.MODULAR
  elif active is not None:
    state = DisplayControllerState.APPROVED
    effective = EffectiveController.MODULAR
  elif staged is not None:
    # Staging is lifecycle progress, not live authority.
    state = DisplayControllerState.STAGED
  else:
    state = DisplayControllerState.STOCK

  state_sha256 = activation.state_sha256
  if valid_diagnostic and state_sha256 is None:
    raise ValueError("validated activation state lacks an identity")
  payload = {
    "activation_state_sha256": state_sha256,
    "active_profile": _identity_payload(active),
    "controller_state": state.value,
    "diagnostic": diagnostic.value,
    "effective_controller": effective.value,
    "informational_only": True,
    "opendbc_commit": opendbc_commit,
    "production_envelope_verified": envelope_verified,
    "rejected_profile_count": len(
      activation.rejected_profile_identities,
    ),
    "runtime_identity_sha256": runtime_identity_sha256,
    "schema_version": LIFECYCLE_STATUS_SCHEMA_VERSION,
    "source_openpilot_commit": source_openpilot_commit,
    "staged_profile": _identity_payload(staged),
    "vehicle_identity": vehicle_identity,
  }
  return payload


def build_lifecycle_status_bytes(
  *,
  activation: PersistentProfileActivation,
  vehicle_identity: str,
  runtime_identity_sha256: str,
  source_openpilot_commit: str,
  opendbc_commit: str,
) -> bytes:
  return canonical_lifecycle_status_bytes(build_lifecycle_status_payload(
    activation=activation,
    vehicle_identity=vehicle_identity,
    runtime_identity_sha256=runtime_identity_sha256,
    source_openpilot_commit=source_openpilot_commit,
    opendbc_commit=opendbc_commit,
  ))
