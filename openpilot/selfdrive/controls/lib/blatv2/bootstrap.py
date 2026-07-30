"""Atomic stock-bootstrap and learned-profile activation policy.

This module does not run either controller. It decides which already-built
artifact is eligible at an engagement boundary. With no approved complete
profile, stock remains active and the modular controller may only shadow.

No profile changes while engaged. A candidate is tied to the exact hash of
its deterministic JSON and requires an external replay/safety approval for
that hash. Driver feedback is evidence after a provisional drive; ``Worse``
rolls back before the next engagement without deleting the rejected data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib

from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  VehicleProfile,
)


class ControllerSelection(IntEnum):
  STOCK = 0
  MODULAR = 1


class DriverFeedback(IntEnum):
  BETTER = 0
  ABOUT_SAME = 1
  WORSE = 2
  NOT_SURE = 3


@dataclass(frozen=True, slots=True)
class ProfileApproval:
  profile_sha256: str
  source_commit: str
  opendbc_commit: str
  replay_passed: bool
  safety_passed: bool
  deterministic_aa_passed: bool

  @property
  def passed(self) -> bool:
    return (
      self.replay_passed
      and self.safety_passed
      and self.deterministic_aa_passed
    )


@dataclass(frozen=True, slots=True)
class EngagementDecision:
  selection: ControllerSelection
  profile: VehicleProfile | None
  profile_sha256: str
  provisional: bool


def profile_sha256(profile: VehicleProfile) -> str:
  return hashlib.sha256(profile.to_json().encode("utf-8")).hexdigest()


class ProfileActivationManager:
  """Lifecycle state; called only at staging/engagement boundaries."""

  def __init__(self, vehicle_identity: str):
    if not vehicle_identity:
      raise ValueError("vehicle identity must not be empty")
    self.vehicle_identity = vehicle_identity
    self._engaged = False
    self._active_profile: VehicleProfile | None = None
    self._active_hash = ""
    self._previous_profile: VehicleProfile | None = None
    self._previous_hash = ""
    self._staged_profile: VehicleProfile | None = None
    self._staged_hash = ""
    self._rollback_requested = False
    self._provisional = False

  @property
  def engaged(self) -> bool:
    return self._engaged

  @property
  def active_profile(self) -> VehicleProfile | None:
    return self._active_profile

  @property
  def staged_profile(self) -> VehicleProfile | None:
    return self._staged_profile

  def stage_candidate(
    self,
    profile: VehicleProfile,
    approval: ProfileApproval,
    *,
    onroad: bool,
  ) -> None:
    """Stage an exact, complete, gate-approved offroad profile."""
    if onroad or self._engaged:
      raise RuntimeError("profiles may be staged only while offroad")
    if profile.vehicle_identity != self.vehicle_identity:
      raise ValueError("profile belongs to a different vehicle")
    if not profile.qualified:
      raise ValueError("incomplete speed profile cannot be staged")
    candidate_hash = profile_sha256(profile)
    if approval.profile_sha256 != candidate_hash:
      raise ValueError("approval does not identify this exact profile")
    if not approval.source_commit or not approval.opendbc_commit:
      raise ValueError("approval source identities must not be empty")
    if not approval.passed:
      raise ValueError("profile acceptance gates did not all pass")
    if (
      self._active_profile is not None
      and profile.revision <= self._active_profile.revision
    ):
      raise ValueError("candidate profile revision must advance")
    self._staged_profile = profile
    self._staged_hash = candidate_hash

  def begin_engagement(self) -> EngagementDecision:
    if self._engaged:
      raise RuntimeError("engagement already active")
    if self._rollback_requested:
      self._active_profile = self._previous_profile
      self._active_hash = self._previous_hash
      self._previous_profile = None
      self._previous_hash = ""
      self._rollback_requested = False
      self._provisional = False
    elif self._staged_profile is not None:
      self._previous_profile = self._active_profile
      self._previous_hash = self._active_hash
      self._active_profile = self._staged_profile
      self._active_hash = self._staged_hash
      self._staged_profile = None
      self._staged_hash = ""
      self._provisional = True

    self._engaged = True
    selection = (
      ControllerSelection.MODULAR
      if self._active_profile is not None
      else ControllerSelection.STOCK
    )
    return EngagementDecision(
      selection=selection,
      profile=self._active_profile,
      profile_sha256=self._active_hash,
      provisional=self._provisional,
    )

  def end_engagement(self) -> None:
    if not self._engaged:
      raise RuntimeError("no active engagement")
    self._engaged = False

  def record_feedback(
    self,
    feedback: DriverFeedback,
    *,
    offroad: bool,
  ) -> None:
    if not offroad or self._engaged:
      raise RuntimeError("driver feedback is accepted only while offroad")
    if not self._provisional or self._active_profile is None:
      raise RuntimeError("no provisional profile awaits feedback")
    if feedback == DriverFeedback.WORSE:
      self._rollback_requested = True
    elif feedback in (
      DriverFeedback.BETTER,
      DriverFeedback.ABOUT_SAME,
    ):
      self._provisional = False
      self._previous_profile = None
      self._previous_hash = ""
    elif feedback == DriverFeedback.NOT_SURE:
      # Keep the current profile provisional; no automatic judgment.
      pass
    else:
      raise ValueError("unknown driver feedback")
