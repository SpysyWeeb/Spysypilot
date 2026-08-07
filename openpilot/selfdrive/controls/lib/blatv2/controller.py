"""Activation-bound facade for one modular BLaTv2 controller candidate.

This module composes existing artifacts; it adds no steering mechanism. The
core remains the sole owner of the unconstrained torque request, feasibility
remains read-only diagnostics, and ``InvalidOutputGuard`` plus the production
actuator envelope remain the sole owners of a command that may be sent to
opendbc.

Stock bootstrap is intentionally outside this facade. A stock engagement
still computes the modular core in shadow, but it never exposes a modular
command. The caller continues to invoke the existing stock controller
separately.

Scalar-only reference and nominal-rack-mapping degradation are accepted for
live use because the existing core marks both results valid and deterministic.
An unqualified profile remains shadow-only unless the caller supplies the
explicit development authorization and a provisional engagement decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
import math
from numbers import Integral

from openpilot.selfdrive.controls.lib.blatv2.actuator import (
  RuntimeTorqueLimits,
)
from openpilot.selfdrive.controls.lib.blatv2.bootstrap import (
  ControllerSelection,
  EngagementDecision,
  profile_sha256,
)
from openpilot.selfdrive.controls.lib.blatv2.contracts import CanonicalFrame
from openpilot.selfdrive.controls.lib.blatv2.core import (
  CoreResult,
  CoreStatus,
  ModularControllerCore,
)
from openpilot.selfdrive.controls.lib.blatv2.feasibility import (
  ConstraintReason,
  FeasibilityStatus,
  inspect_current_torque_feasibility,
)
from openpilot.selfdrive.controls.lib.blatv2.intent import IntentBuildStatus
from openpilot.selfdrive.controls.lib.blatv2.live_safety import (
  InvalidOutputGuard,
  LiveSafetyState,
  SafeCommand,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
)


class CandidateStatus(IntEnum):
  """Primary facade state; detailed core and safety states remain separate."""

  NOT_ENGAGED = 0
  SHADOW_STOCK = 1
  SHADOW_STOCK_CORE_INVALID = 2
  SHADOW_UNQUALIFIED_PROFILE = 3
  MODULAR_INACTIVE = 4
  MODULAR_OK = 5
  MODULAR_DEGRADED_SCALAR_ONLY = 6
  MODULAR_DEGRADED_NOMINAL_MAPPING = 7
  MODULAR_CORE_INVALID = 8
  ENGAGEMENT_DECISION_CHANGED = 9
  ENGAGEMENT_BINDING_FAULTED = 10


class CandidateResult:
  """Reused facade output; snapshot before the candidate's next update."""

  __slots__ = (
    "status",
    "shadow_valid",
    "command_available",
    "command_envelope_applied",
    "command_torque",
    "raw_torque",
    "feasible_torque",
    "unmet_torque",
    "requested_counts",
    "feasible_counts",
    "constraint_active",
    "constraint_reason",
    "feasibility_status",
    "safety_constrained",
    "safety_state",
    "controls_valid",
    "car_control_valid",
    "invalid_frames",
    "recovery_ok_frames",
    "selected_profile_sha256",
    "selected_profile_revision",
    "core_status",
    "core_result",
  )

  def __init__(self) -> None:
    self.clear(
      CandidateStatus.NOT_ENGAGED,
      selected_profile_sha256="",
      selected_profile_revision=0,
    )

  def clear(
    self,
    status: CandidateStatus,
    *,
    selected_profile_sha256: str,
    selected_profile_revision: int,
  ) -> None:
    self.status = status
    self.shadow_valid = False
    self.command_available = False
    self.command_envelope_applied = False
    self.command_torque = 0.0
    self.raw_torque = 0.0
    self.feasible_torque = math.nan
    self.unmet_torque = math.nan
    self.requested_counts: int | None = None
    self.feasible_counts: int | None = None
    self.constraint_active = False
    self.constraint_reason = ConstraintReason.INVALID_REQUEST
    self.feasibility_status = FeasibilityStatus.INVALID_REQUEST
    self.safety_constrained = False
    self.safety_state = LiveSafetyState.INACTIVE
    self.controls_valid = False
    self.car_control_valid = False
    self.invalid_frames = 0
    self.recovery_ok_frames = 0
    self.selected_profile_sha256 = selected_profile_sha256
    self.selected_profile_revision = selected_profile_revision
    self.core_status = CoreStatus.INVALID_MEASUREMENT
    self.core_result: CoreResult | None = None

  def snapshot(self) -> tuple[object, ...]:
    """Allocate an immutable value copy, including the reused core result."""
    values: list[object] = []
    for name in self.__slots__:
      value = getattr(self, name)
      values.append(
        value.snapshot()
        if name == "core_result" and value is not None
        else value
      )
    return tuple(values)


_LIVE_CORE_STATUSES = (
  CoreStatus.OK,
  CoreStatus.DEGRADED_SCALAR_ONLY,
  CoreStatus.DEGRADED_NOMINAL_MAPPING,
)


class ModularControllerCandidate:
  """One exact profile/core/limits candidate with engagement-bound authority."""

  def __init__(
    self,
    *,
    core: ModularControllerCore,
    runtime_limits: RuntimeTorqueLimits,
    development_unqualified_profile_authorized: bool = False,
  ) -> None:
    if not isinstance(core, ModularControllerCore):
      raise TypeError("candidate requires a ModularControllerCore")
    if not isinstance(runtime_limits, RuntimeTorqueLimits):
      raise TypeError("candidate requires explicit runtime torque limits")
    self.core = core
    self.runtime_limits = runtime_limits
    self.development_unqualified_profile_authorized = bool(
      development_unqualified_profile_authorized,
    )
    self.guard = InvalidOutputGuard(core.fixed_dt_s)
    self.result = CandidateResult()
    self._core_profile = core.profile
    self._core_profile_hash = profile_sha256(self._core_profile)
    self._engaged = False
    self._decision: EngagementDecision | None = None
    self._binding_faulted = False

  @property
  def engaged(self) -> bool:
    return self._engaged

  @property
  def engagement_decision(self) -> EngagementDecision | None:
    return self._decision

  def prime_transport_state(
    self,
    previous_applied_counts: int,
  ) -> None:
    """Prime only the rack-transport history from an exact measured count."""
    if self._engaged:
      raise RuntimeError("transport state may be primed only while disengaged")
    self._prime_transport_counts(previous_applied_counts)

  def reprime_transport_state(
    self,
    previous_applied_counts: int,
  ) -> None:
    """Resynchronize transport history after a detected controller-frame gap.

    The frame carrying the discontinuity is invalid and never runs the core.
    Repeating the exact current command-layer count across the delay ring
    prevents the next valid frame from treating missing 100 Hz commands as a
    continuous history.  This changes transport state only; engagement
    identity and the invalid-output guard remain intact.
    """
    if not self._engaged or self._decision is None:
      raise RuntimeError("transport resynchronization requires engagement")
    if self._decision.selection != ControllerSelection.MODULAR:
      raise RuntimeError("stock selection has no live transport history")
    self._prime_transport_counts(previous_applied_counts)

  def _prime_transport_counts(
    self,
    previous_applied_counts: int,
  ) -> None:
    if isinstance(previous_applied_counts, bool) or not isinstance(
      previous_applied_counts,
      Integral,
    ):
      raise TypeError("transport prime requires an exact integer count")
    counts = int(previous_applied_counts)
    if abs(counts) > self.runtime_limits.steer_max:
      raise ValueError("transport prime exceeds the runtime torque envelope")
    self.core.prime_applied_history(
      counts / self.runtime_limits.steer_max,
    )

  def _validate_begin_decision(
    self,
    decision: EngagementDecision,
  ) -> None:
    if not isinstance(decision, EngagementDecision):
      raise TypeError("candidate engagement requires EngagementDecision")
    if self.core.profile is not self._core_profile:
      raise ValueError("core profile changed after candidate construction")
    if decision.selection == ControllerSelection.STOCK:
      if (
        decision.profile is not None
        or decision.profile_sha256 != ""
        or decision.provisional
      ):
        raise ValueError(
          "stock decision must not carry modular profile state",
        )
      return
    if decision.selection != ControllerSelection.MODULAR:
      raise ValueError("unknown controller selection")
    if not self.runtime_limits.production_envelope_verified:
      raise ValueError(
        "modular actuation requires an opendbc-verified runtime envelope",
      )
    if decision.profile is None:
      raise ValueError("modular decision is missing its profile")
    unqualified_authorized = (
      self.development_unqualified_profile_authorized
      and decision.provisional
    )
    if (
      (not decision.profile.qualified or not self._core_profile.qualified)
      and not unqualified_authorized
    ):
      raise ValueError("unqualified profile cannot own live actuation")
    if (
      decision.profile.vehicle_identity
      != self._core_profile.vehicle_identity
    ):
      raise ValueError("modular decision vehicle identity differs from core")
    if decision.profile != self._core_profile:
      raise ValueError("modular decision profile content differs from core")
    if decision.profile is not self._core_profile:
      raise ValueError(
        "modular decision must carry the core's exact profile object",
      )
    if (
      decision.profile_sha256 != self._core_profile_hash
      or profile_sha256(decision.profile) != self._core_profile_hash
    ):
      raise ValueError("modular decision profile hash differs from core")

  def begin_engagement(self, decision: EngagementDecision) -> None:
    """Bind exactly one immutable bootstrap decision until explicit end."""
    if self._engaged:
      raise RuntimeError("candidate engagement already active")
    self._validate_begin_decision(decision)
    self.guard.reset()
    self._decision = decision
    self._binding_faulted = False
    self._engaged = True
    self._clear_result_for_bound_decision()

  def end_engagement(self, decision: EngagementDecision) -> None:
    """End the exact bound decision and reset all live-safety state."""
    if not self._engaged or self._decision is None:
      raise RuntimeError("candidate has no active engagement")
    matches = decision is self._decision
    self.guard.reset()
    self._engaged = False
    self._decision = None
    self._binding_faulted = False
    self.result.clear(
      CandidateStatus.NOT_ENGAGED,
      selected_profile_sha256="",
      selected_profile_revision=0,
    )
    if not matches:
      raise ValueError(
        "engagement end decision differs from the bound decision",
      )

  def _decision_matches(self, decision: EngagementDecision) -> bool:
    common_match = (
      self._decision is not None
      and decision is self._decision
      and decision.selection == self._decision.selection
      and decision.profile is self._decision.profile
      and decision.profile_sha256 == self._decision.profile_sha256
      and decision.provisional == self._decision.provisional
      and self.core.profile is self._core_profile
    )
    if not common_match:
      return False
    if decision.selection == ControllerSelection.STOCK:
      return decision.profile is None and decision.profile_sha256 == ""
    profile_authorized = (
      self._core_profile.qualified
      or (
        self.development_unqualified_profile_authorized
        and decision.provisional
      )
    )
    return (
      decision.profile is self._core_profile
      and decision.profile_sha256 == self._core_profile_hash
      and profile_authorized
    )

  def _clear_result_for_bound_decision(self) -> None:
    decision = self._decision
    if (
      decision is None
      or decision.selection != ControllerSelection.MODULAR
      or decision.profile is None
    ):
      selected_hash = ""
      selected_revision = 0
    else:
      selected_hash = decision.profile_sha256
      selected_revision = decision.profile.revision
    self.result.clear(
      CandidateStatus.NOT_ENGAGED,
      selected_profile_sha256=selected_hash,
      selected_profile_revision=selected_revision,
    )

  def _copy_core_and_feasibility(
    self,
    core_result: CoreResult,
    previous_applied_counts: int,
    driver_torque: float,
  ) -> None:
    self.result.core_result = core_result
    self.result.core_status = core_result.status
    self.result.shadow_valid = core_result.valid
    self.result.raw_torque = core_result.raw_torque
    feasibility = inspect_current_torque_feasibility(
      self.runtime_limits,
      previous_applied_counts,
      core_result.raw_torque,
      driver_torque,
    )
    self.result.feasibility_status = feasibility.status
    self.result.feasible_torque = feasibility.feasible_applied_torque
    self.result.unmet_torque = feasibility.unmet_torque
    self.result.requested_counts = feasibility.requested_counts
    self.result.feasible_counts = feasibility.feasible_counts
    self.result.constraint_active = feasibility.constraint_active
    self.result.constraint_reason = feasibility.constraint_reason

  def _copy_safe_command(self, safe_command: SafeCommand) -> None:
    self.result.command_available = True
    self.result.command_envelope_applied = True
    self.result.command_torque = safe_command.torque
    self.result.safety_constrained = safe_command.constrained
    self.result.safety_state = safe_command.state
    self.result.controls_valid = safe_command.controls_valid
    self.result.car_control_valid = safe_command.car_control_valid
    self.result.invalid_frames = safe_command.invalid_frames
    self.result.recovery_ok_frames = safe_command.recovery_ok_frames

  def update(
    self,
    *,
    engagement_decision: EngagementDecision,
    previous_applied_counts: int,
    driver_torque: float,
    frame: CanonicalFrame | None,
    intent_status: IntentBuildStatus,
    intent_plan_times_s: Sequence[float],
    intent_orientation_rates_z: Sequence[float],
    intent_velocities_x: Sequence[float],
    scalar_curvature: float,
    current_v_ego_m_s: float,
    measured_rack_angle_deg: float,
    measured_rack_rate_deg_s: float,
    measured_rack_acceleration_deg_s2: float,
    lateral_accel_offset: float,
    live_mapping: RackMappingSnapshot | None,
    lateral_active: bool,
    lateral_valid: bool,
    engagement_boundary: bool,
    live_parameters_valid: bool,
    steering_pressed: bool,
    actuator_constrained: bool,
    output_constrained: bool,
    standstill: bool,
  ) -> CandidateResult:
    """Compute one shadow frame and, only when authorized, one safe command."""
    if isinstance(previous_applied_counts, bool) or not isinstance(
      previous_applied_counts,
      Integral,
    ):
      raise TypeError("previous applied torque must be an integer count")
    driver = float(driver_torque)
    if not math.isfinite(driver):
      raise ValueError("driver torque must be finite")

    self._clear_result_for_bound_decision()
    if not self._engaged or self._decision is None:
      self.guard.reset()
      return self.result

    applied_counts = int(previous_applied_counts)
    applied_torque = applied_counts / self.runtime_limits.steer_max
    decision_matches = self._decision_matches(engagement_decision)
    if self._binding_faulted or not decision_matches:
      newly_faulted = not self._binding_faulted
      self._binding_faulted = True
      safe_command = self.guard.update(
        active=bool(lateral_active),
        core_ok=False,
        raw_torque=math.nan,
        applied_torque=applied_torque,
        driver_torque=driver,
        limits=self.runtime_limits,
      )
      self._copy_safe_command(safe_command)
      self.result.status = (
        CandidateStatus.ENGAGEMENT_DECISION_CHANGED
        if newly_faulted
        else CandidateStatus.ENGAGEMENT_BINDING_FAULTED
      )
      return self.result

    core_result = self.core.update(
      frame=frame,
      intent_status=intent_status,
      intent_plan_times_s=intent_plan_times_s,
      intent_orientation_rates_z=intent_orientation_rates_z,
      intent_velocities_x=intent_velocities_x,
      scalar_curvature=scalar_curvature,
      current_v_ego_m_s=current_v_ego_m_s,
      measured_rack_angle_deg=measured_rack_angle_deg,
      measured_rack_rate_deg_s=measured_rack_rate_deg_s,
      measured_rack_acceleration_deg_s2=(
        measured_rack_acceleration_deg_s2
      ),
      recorded_applied_torque=applied_torque,
      lateral_accel_offset=lateral_accel_offset,
      live_mapping=live_mapping,
      lateral_active=lateral_active,
      lateral_valid=lateral_valid,
      engagement_boundary=engagement_boundary,
      live_parameters_valid=live_parameters_valid,
      steering_pressed=steering_pressed,
      actuator_constrained=actuator_constrained,
      output_constrained=output_constrained,
      standstill=standstill,
    )
    self._copy_core_and_feasibility(
      core_result,
      applied_counts,
      driver,
    )

    if self._decision.selection == ControllerSelection.STOCK:
      self.guard.reset()
      self.result.controls_valid = True
      self.result.car_control_valid = True
      self.result.safety_state = LiveSafetyState.INACTIVE
      if core_result.status == CoreStatus.SHADOW_UNQUALIFIED_PROFILE:
        self.result.status = (
          CandidateStatus.SHADOW_UNQUALIFIED_PROFILE
        )
      elif core_result.valid:
        self.result.status = CandidateStatus.SHADOW_STOCK
      else:
        self.result.status = CandidateStatus.SHADOW_STOCK_CORE_INVALID
      return self.result

    if not lateral_active:
      safe_command = self.guard.update(
        active=False,
        core_ok=False,
        raw_torque=core_result.raw_torque,
        applied_torque=applied_torque,
        driver_torque=driver,
        limits=self.runtime_limits,
      )
      self.result.status = CandidateStatus.MODULAR_INACTIVE
      self.result.safety_state = safe_command.state
      self.result.controls_valid = safe_command.controls_valid
      self.result.car_control_valid = safe_command.car_control_valid
      return self.result

    profile_authorized = (
      core_result.profile_qualified
      or (
        self.development_unqualified_profile_authorized
        and self._decision.provisional
        and core_result.status == CoreStatus.SHADOW_UNQUALIFIED_PROFILE
      )
    )
    core_status_live = (
      core_result.status in _LIVE_CORE_STATUSES
      or (
        self.development_unqualified_profile_authorized
        and core_result.status == CoreStatus.SHADOW_UNQUALIFIED_PROFILE
      )
    )
    core_live_valid = (
      core_result.valid
      and profile_authorized
      and core_status_live
      and lateral_valid
    )
    safe_command = self.guard.update(
      active=True,
      core_ok=core_live_valid,
      raw_torque=core_result.raw_torque,
      applied_torque=applied_torque,
      driver_torque=driver,
      limits=self.runtime_limits,
    )
    self._copy_safe_command(safe_command)

    if core_live_valid:
      if safe_command.state == LiveSafetyState.OK:
        if (
          self.result.feasibility_status != FeasibilityStatus.OK
          or self.result.feasible_counts is None
        ):
          raise AssertionError(
            "valid core request has no valid envelope projection",
          )
        safe_counts = int(round(
          safe_command.torque * self.runtime_limits.steer_max,
        ))
        if (
          safe_counts != self.result.feasible_counts
          or safe_command.torque != self.result.feasible_torque
        ):
          raise AssertionError(
            "live safety and feasibility disagree on the sole envelope",
          )
      if core_result.status == CoreStatus.DEGRADED_SCALAR_ONLY:
        self.result.status = (
          CandidateStatus.MODULAR_DEGRADED_SCALAR_ONLY
        )
      elif core_result.status == CoreStatus.DEGRADED_NOMINAL_MAPPING:
        self.result.status = (
          CandidateStatus.MODULAR_DEGRADED_NOMINAL_MAPPING
        )
      else:
        self.result.status = CandidateStatus.MODULAR_OK
    else:
      self.result.status = CandidateStatus.MODULAR_CORE_INVALID
    return self.result

  def update_invalid_frame(
    self,
    *,
    engagement_decision: EngagementDecision,
    previous_applied_counts: int,
    driver_torque: float,
    lateral_active: bool,
  ) -> CandidateResult:
    """Route an adapter/runtime failure through the same live-safety guard.

    This is not a fallback controller and it does not calculate a substitute
    request.  It exists so a malformed live message or an exception at a
    cereal-to-core boundary cannot either crash into an unannounced steering
    drop or hot-switch to stock.  The first frame holds, subsequent frames
    decay through the one production envelope, and the existing commIssue
    validity path latches after the pinned interval.
    """
    if isinstance(previous_applied_counts, bool) or not isinstance(
      previous_applied_counts,
      Integral,
    ):
      raise TypeError("previous applied torque must be an integer count")
    driver = float(driver_torque)
    if not math.isfinite(driver):
      driver = 0.0

    self._clear_result_for_bound_decision()
    if not self._engaged or self._decision is None:
      self.guard.reset()
      return self.result
    applied_torque = (
      int(previous_applied_counts) / self.runtime_limits.steer_max
    )
    decision_matches = self._decision_matches(engagement_decision)
    if self._binding_faulted or not decision_matches:
      newly_faulted = not self._binding_faulted
      self._binding_faulted = True
      safe_command = self.guard.update(
        active=bool(lateral_active),
        core_ok=False,
        raw_torque=math.nan,
        applied_torque=applied_torque,
        driver_torque=driver,
        limits=self.runtime_limits,
      )
      self._copy_safe_command(safe_command)
      self.result.status = (
        CandidateStatus.ENGAGEMENT_DECISION_CHANGED
        if newly_faulted
        else CandidateStatus.ENGAGEMENT_BINDING_FAULTED
      )
      return self.result
    if self._decision.selection != ControllerSelection.MODULAR:
      self.guard.reset()
      self.result.status = CandidateStatus.SHADOW_STOCK_CORE_INVALID
      self.result.controls_valid = True
      self.result.car_control_valid = True
      return self.result

    safe_command = self.guard.update(
      active=bool(lateral_active),
      core_ok=False,
      raw_torque=math.nan,
      applied_torque=applied_torque,
      driver_torque=driver,
      limits=self.runtime_limits,
    )
    self._copy_safe_command(safe_command)
    self.result.raw_torque = math.nan
    self.result.status = (
      CandidateStatus.MODULAR_CORE_INVALID
      if lateral_active
      else CandidateStatus.MODULAR_INACTIVE
    )
    return self.result
