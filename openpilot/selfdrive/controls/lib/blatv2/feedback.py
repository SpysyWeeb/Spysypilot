"""Profile-bound driver-feedback contract for the modular lateral controller.

The UI is only a collector for explicit driver feedback. It never interprets
driver interventions and never changes controller selection. A producer
creates a request for one exact profile artifact; the offroad UI writes one
response bound to that same artifact. Both Params keys are persistent JSON so
manager restarts and onroad/offroad transitions cannot silently consume a
pending request.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Protocol


FEEDBACK_SCHEMA_VERSION = 2
FEEDBACK_REQUEST_PARAM = "BLaTv2FeedbackRequest"
FEEDBACK_RESPONSE_PARAM = "BLaTv2FeedbackResponse"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REQUEST_KEYS = frozenset(
  (
    "schemaVersion",
    "artifactSha256",
    "profileSha256",
    "profileRevision",
  ),
)
_RESPONSE_KEYS = frozenset(
  (
    "schemaVersion",
    "artifactSha256",
    "profileSha256",
    "profileRevision",
    "choice",
  ),
)


class FeedbackChoice(StrEnum):
  BETTER = "BETTER"
  ABOUT_SAME = "ABOUT_SAME"
  WORSE = "WORSE"
  NOT_SURE = "NOT_SURE"


class FeedbackValidationError(ValueError):
  """Raised when feedback JSON violates the exact wire contract."""


class ParamsLike(Protocol):
  def get(self, key: str, block: bool = False): ...

  def put(self, key: str, value, block: bool = False): ...


def _validate_sha256(value: object, name: str) -> str:
  if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
    raise FeedbackValidationError(f"{name} must be a lowercase SHA-256")
  return value


def _validate_profile_identity(
  artifact_sha256: object,
  profile_sha256: object,
  profile_revision: object,
) -> tuple[str, str, int]:
  artifact_identity = _validate_sha256(
    artifact_sha256,
    "artifactSha256",
  )
  profile_identity = _validate_sha256(
    profile_sha256,
    "profileSha256",
  )
  if type(profile_revision) is not int or profile_revision < 1:
    raise FeedbackValidationError("profileRevision must be a positive integer")
  return artifact_identity, profile_identity, profile_revision


def _validate_payload(value: object, expected_keys: frozenset[str]) -> dict:
  if type(value) is not dict:
    raise FeedbackValidationError("feedback payload must be a JSON object")
  if frozenset(value) != expected_keys:
    raise FeedbackValidationError("feedback payload keys are not canonical")
  if type(value["schemaVersion"]) is not int or value["schemaVersion"] != FEEDBACK_SCHEMA_VERSION:
    raise FeedbackValidationError("unsupported feedback schemaVersion")
  return value


@dataclass(frozen=True, slots=True)
class FeedbackRequest:
  artifact_sha256: str
  profile_sha256: str
  profile_revision: int

  def __post_init__(self) -> None:
    _validate_profile_identity(
      self.artifact_sha256,
      self.profile_sha256,
      self.profile_revision,
    )

  @classmethod
  def from_param(cls, value: object) -> FeedbackRequest:
    payload = _validate_payload(value, _REQUEST_KEYS)
    (
      artifact_sha256,
      profile_sha256,
      profile_revision,
    ) = _validate_profile_identity(
      payload["artifactSha256"],
      payload["profileSha256"],
      payload["profileRevision"],
    )
    return cls(artifact_sha256, profile_sha256, profile_revision)

  def to_param(self) -> dict:
    # Insertion order is fixed so Params' JSON encoder writes one canonical form.
    return {
      "schemaVersion": FEEDBACK_SCHEMA_VERSION,
      "artifactSha256": self.artifact_sha256,
      "profileSha256": self.profile_sha256,
      "profileRevision": self.profile_revision,
    }


@dataclass(frozen=True, slots=True)
class FeedbackResponse:
  artifact_sha256: str
  profile_sha256: str
  profile_revision: int
  choice: FeedbackChoice

  def __post_init__(self) -> None:
    _validate_profile_identity(
      self.artifact_sha256,
      self.profile_sha256,
      self.profile_revision,
    )
    if type(self.choice) is not FeedbackChoice:
      raise FeedbackValidationError("choice must be a FeedbackChoice")

  @classmethod
  def from_param(cls, value: object) -> FeedbackResponse:
    payload = _validate_payload(value, _RESPONSE_KEYS)
    (
      artifact_sha256,
      profile_sha256,
      profile_revision,
    ) = _validate_profile_identity(
      payload["artifactSha256"],
      payload["profileSha256"],
      payload["profileRevision"],
    )
    try:
      choice = FeedbackChoice(payload["choice"])
    except (TypeError, ValueError) as exc:
      raise FeedbackValidationError("unknown feedback choice") from exc
    return cls(artifact_sha256, profile_sha256, profile_revision, choice)

  @classmethod
  def for_request(
    cls,
    request: FeedbackRequest,
    choice: FeedbackChoice,
  ) -> FeedbackResponse:
    return cls(
      request.artifact_sha256,
      request.profile_sha256,
      request.profile_revision,
      choice,
    )

  def to_param(self) -> dict:
    return {
      "schemaVersion": FEEDBACK_SCHEMA_VERSION,
      "artifactSha256": self.artifact_sha256,
      "profileSha256": self.profile_sha256,
      "profileRevision": self.profile_revision,
      "choice": self.choice.value,
    }

  def matches(self, request: FeedbackRequest) -> bool:
    return (
      self.artifact_sha256 == request.artifact_sha256
      and self.profile_sha256 == request.profile_sha256
      and self.profile_revision == request.profile_revision
    )


def pending_feedback_request(params: ParamsLike) -> FeedbackRequest | None:
  """Return only a strictly-valid request without a matching response.

  A malformed request cannot produce a UI action. A malformed response cannot
  acknowledge a valid request, so the valid request remains pending. Neither
  key is removed or rewritten here.
  """
  try:
    request = FeedbackRequest.from_param(params.get(FEEDBACK_REQUEST_PARAM))
  except (FeedbackValidationError, KeyError, TypeError, ValueError, RuntimeError, OSError):
    return None

  try:
    response = FeedbackResponse.from_param(params.get(FEEDBACK_RESPONSE_PARAM))
  except (FeedbackValidationError, KeyError, TypeError, ValueError, RuntimeError, OSError):
    response = None
  return None if response is not None and response.matches(request) else request


def write_feedback_response(
  params: ParamsLike,
  request: FeedbackRequest,
  choice: FeedbackChoice,
) -> bool:
  """Atomically persist a response if ``request`` is still the pending request."""
  pending = pending_feedback_request(params)
  if pending != request:
    return False
  response = FeedbackResponse.for_request(request, choice)
  # Params' blocking write fsyncs a temporary file and atomically renames it.
  params.put(FEEDBACK_RESPONSE_PARAM, response.to_param(), block=True)
  return True


class FeedbackPromptState:
  """Pure UI lifecycle state, independent of the rendering framework."""

  def __init__(self) -> None:
    self._presented_request: FeedbackRequest | None = None

  @property
  def presented_request(self) -> FeedbackRequest | None:
    return self._presented_request

  def update(
    self,
    params: ParamsLike,
    *,
    offroad: bool,
  ) -> FeedbackRequest | None:
    """Return a request exactly once per offroad presentation lifecycle."""
    if not offroad:
      self._presented_request = None
      return None
    request = pending_feedback_request(params)
    if request is None:
      self._presented_request = None
      return None
    if request == self._presented_request:
      return None
    self._presented_request = request
    return request

  def submit(
    self,
    params: ParamsLike,
    choice: FeedbackChoice,
    *,
    offroad: bool,
  ) -> bool:
    """Write only the exact request currently represented by the modal."""
    if not offroad or self._presented_request is None:
      return False
    try:
      written = write_feedback_response(
        params,
        self._presented_request,
        choice,
      )
    except (FeedbackValidationError, KeyError, TypeError, ValueError, RuntimeError, OSError):
      return False
    if written:
      self._presented_request = None
    return written
