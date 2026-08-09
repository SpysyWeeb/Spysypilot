"""Engagement-bound live composition for the modular BLaTv2 controller.

The existing stock ``LatControlTorque`` remains outside this module and is the
sole actuator unless an exact, already-active approved artifact or an explicit
development trial is eligible at the lateral-session boundary. This module
performs no profile staging, feedback processing, file/Params writes, or hot
fallback.

A lateral session (``enabled OR lateral_active``) owns one immutable selection.
A bound modular
controller routes every invalid frame through ``InvalidOutputGuard``; it
never switches to stock while that session remains active.  A stock engagement
never calls the modular numerical core and modular telemetry/invalidity cannot
affect its command or message validity.
"""

from __future__ import annotations

from enum import IntEnum
import math
from pathlib import Path
import time
from typing import Any

from opendbc.car.structs import car

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.blatv2.approved_artifact import (
  ArtifactDiagnostic,
  ApprovedProfileArtifact,
  PersistentProfileActivation,
)
from openpilot.selfdrive.controls.lib.blatv2.bootstrap import (
  ControllerSelection,
  EngagementDecision,
  profile_sha256,
)
from openpilot.selfdrive.controls.lib.blatv2.controller import (
  CandidateResult,
  ModularControllerCandidate,
)
from openpilot.selfdrive.controls.lib.blatv2.core import (
  ModularControllerCore,
)
from openpilot.selfdrive.controls.lib.blatv2.horizon import HorizonPolicy
from openpilot.selfdrive.controls.lib.blatv2.intent import INTENT_CAPACITY
from openpilot.selfdrive.controls.lib.blatv2.live_adapter import (
  LiveInputAdapter,
  MAX_RECORDED_FRAME_GAP_NS,
  PreparedLiveInput,
  exact_applied_torque_counts,
  exact_steering_request_state,
)
from openpilot.selfdrive.controls.lib.blatv2.policy import ControllerPolicy
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
  RuntimeVehicleBundle,
  build_runtime_vehicle_bundle,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  compose_controller_profile,
)


MODULAR_LIVE_ARCHITECTURE = "blatv2.modular.preview-rack"
MODULAR_REACTIVE_ARCHITECTURE = "blatv2.modular.inverse-rack"
MODULAR_LIVE_VERSION = 2
EXPERIMENTAL_CONTROLLER_PARAM = "BLaTv2ExperimentalController"
PROVISIONAL_RACK_DYNAMICS_PATH = (
  Path(__file__).resolve().parent / "provisional_rack_dynamics.json"
)
PROVISIONAL_CONTROLLER_POLICY_PATH = (
  Path(__file__).resolve().parent / "provisional_controller_policy.json"
)
PROVISIONAL_HORIZON_POLICY_PATH = (
  Path(__file__).resolve().parent / "provisional_horizon_policy.json"
)
APPROVED_HORIZON_POLICY_PATH = (
  Path(__file__).resolve().parent / "approved_horizon_policy.json"
)


class LiveEligibility(IntEnum):
  ELIGIBLE = 0
  NO_RUNTIME_BUNDLE = 1
  NO_ACTIVE_ARTIFACT = 2
  ACTIVATION_STATE_INVALID = 3
  VEHICLE_IDENTITY_MISMATCH = 4
  RUNTIME_IDENTITY_MISMATCH = 5
  SOURCE_COMMIT_MISMATCH = 6
  OPENDBC_COMMIT_MISMATCH = 7
  PANDA_COMMIT_MISMATCH = 8
  PROVISIONAL_POLICY = 9
  UNQUALIFIED_PROFILE = 10
  UNVERIFIED_PRODUCTION_ENVELOPE = 11
  CONSTRUCTION_FAILED = 12
  NO_EXACT_APPLIED_COUNT = 13
  LATERAL_MANEUVER_MODE = 14
  CALIBRATION_COMPOSITION_MISMATCH = 15
  EXPERIMENTAL_PLATFORM_UNSUPPORTED = 16
  EXPERIMENTAL_CONFIGURATION_INVALID = 17
  NO_EXACT_STEERING_REQUEST_STATE = 18
  HORIZON_CONFIGURATION_INVALID = 19


def control_witness_mono_ns() -> int:
  """Capture the Event.logMonoTime-compatible clock at precompute entry."""
  clock_id = getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC)
  return time.clock_gettime_ns(clock_id)


class ModularLiveController:
  """One preconstructed candidate and immutable engagement binding."""

  __slots__ = (
    "car_params",
    "runtime_bundle",
    "artifact",
    "activation",
    "activation_provisional",
    "artifact_diagnostic",
    "experimental_requested",
    "experimental_platform_compatible",
    "experimental_active",
    "controller_profile",
    "controller_policy",
    "horizon_policy",
    "source_openpilot_commit",
    "opendbc_commit",
    "panda_commit",
    "eligibility",
    "binding_reason",
    "candidate",
    "adapter",
    "selection",
    "decision",
    "enabled_bound",
    "maneuver_forced_stock",
    "last_exact_applied_counts",
    "last_applied_count_valid",
    "last_steering_request_active",
    "last_steering_request_counter",
    "last_steering_request_valid",
    "last_car_output_mono_ns",
    "car_output_fresh",
    "car_output_cadence_valid",
    "transport_reprime_required",
    "pending_command_counts",
    "last_expected_command_counts",
    "final_count_residual",
    "final_count_match_valid",
    "previous_output_constrained",
    "previous_lateral_active",
    "last_output_constrained_input",
    "last_actuator_constrained_input",
    "prepared_input",
    "candidate_result",
    "compute_time_seconds",
    "control_witness_ns",
    "previous_control_witness_ns",
    "control_cadence_valid",
    "transport_reprimed",
    "adapter_exception",
  )

  def __init__(
    self,
    *,
    car_params: car.CarParams,
    runtime_bundle: RuntimeVehicleBundle | None,
    artifact: ApprovedProfileArtifact | None,
    activation_provisional: bool,
    artifact_diagnostic: ArtifactDiagnostic,
    source_openpilot_commit: str,
    opendbc_commit: str,
    panda_commit: str,
    activation: PersistentProfileActivation | None = None,
    experimental_requested: bool = False,
    experimental_platform_compatible: bool = False,
    experimental_policy: ControllerPolicy | None = None,
    horizon_policy: HorizonPolicy | None = None,
  ) -> None:
    self.car_params = car_params
    self.runtime_bundle = runtime_bundle
    self.artifact = artifact
    self.activation = activation
    self.activation_provisional = bool(activation_provisional)
    self.artifact_diagnostic = artifact_diagnostic
    self.experimental_requested = bool(experimental_requested)
    self.experimental_platform_compatible = bool(
      experimental_platform_compatible,
    )
    self.experimental_active = False
    self.controller_profile = (
      None if artifact is None else artifact.vehicle_profile
    )
    self.controller_policy = (
      experimental_policy
      if artifact is None
      else artifact.controller_policy
    )
    self.horizon_policy = horizon_policy
    if (
      artifact is None
      and self.experimental_requested
      and runtime_bundle is not None
    ):
      try:
        self.controller_profile = compose_controller_profile(
          runtime_bundle.calibration_seed_profile,
          runtime_bundle.seed_profile,
        )
      except (TypeError, ValueError, OverflowError):
        self.controller_profile = None
    self.source_openpilot_commit = str(source_openpilot_commit)
    self.opendbc_commit = str(opendbc_commit)
    self.panda_commit = str(panda_commit)
    self.eligibility = self._validate_eligibility()
    self.experimental_active = bool(
      self.eligibility == LiveEligibility.ELIGIBLE
      and artifact is None
      and self.experimental_requested
    )
    self.binding_reason = self.eligibility
    self.candidate: ModularControllerCandidate | None = None
    self.adapter: LiveInputAdapter | None = None

    if self.eligibility == LiveEligibility.ELIGIBLE:
      if (
        runtime_bundle is None
        or self.controller_profile is None
        or self.controller_policy is None
        or self.horizon_policy is None
      ):
        raise AssertionError("eligible live controller lacks a candidate")
      try:
        core = ModularControllerCore(
          fixed_dt_s=DT_CTRL,
          profile=self.controller_profile,
          tracking_policy=self.controller_policy.tracking_policy,
          observer_policy=self.controller_policy.observer_policy,
          nominal_mapping=runtime_bundle.nominal_rack_mapping,
          runtime_limits=runtime_bundle.torque_limits,
          horizon_policy=self.horizon_policy,
          plan_capacity=INTENT_CAPACITY,
          development_reactive_only=self.experimental_active,
        )
        self.candidate = ModularControllerCandidate(
          core=core,
          development_unqualified_profile_authorized=(
            self.experimental_active
          ),
        )
        self.adapter = LiveInputAdapter(
          car_params=car_params,
          profile=self.controller_profile,
        )
      except Exception:
        self.eligibility = LiveEligibility.CONSTRUCTION_FAILED
        self.experimental_active = False
        self.candidate = None
        self.adapter = None

    self.selection = ControllerSelection.STOCK
    self.decision: EngagementDecision | None = None
    self.enabled_bound = False
    self.maneuver_forced_stock = False
    self.last_exact_applied_counts: int | None = None
    self.last_applied_count_valid = False
    self.last_steering_request_active = False
    self.last_steering_request_counter = 0
    self.last_steering_request_valid = False
    self.last_car_output_mono_ns: int | None = None
    self.car_output_fresh = False
    self.car_output_cadence_valid = False
    self.transport_reprime_required = False
    self.pending_command_counts: int | None = None
    self.last_expected_command_counts: int | None = None
    self.final_count_residual = 0
    self.final_count_match_valid = False
    self.previous_output_constrained = False
    self.previous_lateral_active = False
    self.last_output_constrained_input = False
    self.last_actuator_constrained_input = False
    self.prepared_input: PreparedLiveInput | None = None
    self.candidate_result: CandidateResult | None = None
    self.compute_time_seconds = 0.0
    self.control_witness_ns = 0
    self.previous_control_witness_ns: int | None = None
    self.control_cadence_valid = True
    self.transport_reprimed = False
    self.adapter_exception = False

  @classmethod
  def from_persistent(
    cls,
    *,
    car_params: car.CarParams,
    car_interface: object,
    params: Any,
    source_openpilot_commit: str,
    opendbc_commit: str,
    panda_commit: str,
  ) -> ModularLiveController:
    """Read/validate once at process construction; never writes activation."""
    runtime_bundle: RuntimeVehicleBundle | None = None
    artifact: ApprovedProfileArtifact | None = None
    activation: PersistentProfileActivation | None = None
    activation_provisional = False
    diagnostic = ArtifactDiagnostic.ABSENT
    experimental_requested = False
    experimental_platform_compatible = False
    experimental_policy: ControllerPolicy | None = None
    approved_horizon_policy: HorizonPolicy | None = None
    horizon_policy: HorizonPolicy | None = None
    try:
      controller = car_interface.CC
      controller_params = controller.params
      dynamics = ProvisionalRackDynamics.from_json_file(
        PROVISIONAL_RACK_DYNAMICS_PATH,
      )
      runtime_bundle = build_runtime_vehicle_bundle(
        car_params=car_params,
        car_interface_or_callback=car_interface,
        controller_params=controller_params,
        vehicle_identity=str(car_params.carFingerprint),
        provisional_rack_dynamics=dynamics,
      )
    except Exception:
      diagnostic = ArtifactDiagnostic.STATE_INVALID
      runtime_bundle = None

    if runtime_bundle is not None:
      try:
        approved_horizon_policy = HorizonPolicy.from_json_file(
          APPROVED_HORIZON_POLICY_PATH,
        )
      except (OSError, TypeError, ValueError, OverflowError):
        approved_horizon_policy = None
      horizon_policy = approved_horizon_policy
      try:
        experimental_requested = params.get_bool(
          EXPERIMENTAL_CONTROLLER_PARAM,
        )
      except Exception:
        experimental_requested = False
      if experimental_requested:
        experimental_platform_compatible = (
          getattr(
            controller_params,
            "BLATV2_CONTEXTUAL_DYNAMICS_COMPATIBLE",
            False,
          )
          is True
        )
        try:
          horizon_policy = HorizonPolicy.from_json_file(
            PROVISIONAL_HORIZON_POLICY_PATH,
          )
        except (OSError, TypeError, ValueError, OverflowError):
          horizon_policy = None
        try:
          experimental_policy = ControllerPolicy.from_json_file(
            PROVISIONAL_CONTROLLER_POLICY_PATH,
          )
        except (OSError, TypeError, ValueError, OverflowError):
          experimental_policy = None

    if runtime_bundle is not None:
      try:
        # PersistentProfileActivation validates the complete state and every
        # embedded artifact during construction. Its begin/end calls are now a
        # read-only binding over state already prepared by profiled offroad.
        activation = PersistentProfileActivation(
          params,
          expected_vehicle_identity=runtime_bundle.vehicle_identity,
          expected_runtime_vehicle_identity_sha256=(
            runtime_bundle.identity_sha256
          ),
          expected_source_openpilot_commit=source_openpilot_commit,
          expected_opendbc_commit=opendbc_commit,
          expected_panda_commit=panda_commit,
          production_envelope_verified=(
            runtime_bundle.torque_limits.production_envelope_verified
          ),
        )
        diagnostic = activation.diagnostic
        if diagnostic == ArtifactDiagnostic.OK:
          artifact = activation.active_artifact
          activation_provisional = activation.provisional
          if artifact is not None:
            horizon_policy = approved_horizon_policy
      except Exception:
        diagnostic = ArtifactDiagnostic.STATE_INVALID
        activation = None
        artifact = None
        activation_provisional = False
    return cls(
      car_params=car_params,
      runtime_bundle=runtime_bundle,
      artifact=artifact,
      activation=activation,
      activation_provisional=activation_provisional,
      artifact_diagnostic=diagnostic,
      source_openpilot_commit=source_openpilot_commit,
      opendbc_commit=opendbc_commit,
      panda_commit=panda_commit,
      experimental_requested=experimental_requested,
      experimental_platform_compatible=experimental_platform_compatible,
      experimental_policy=experimental_policy,
      horizon_policy=horizon_policy,
    )

  def _validate_eligibility(self) -> LiveEligibility:
    bundle = self.runtime_bundle
    artifact = self.artifact
    if bundle is None:
      return LiveEligibility.NO_RUNTIME_BUNDLE
    if artifact is None:
      if self.experimental_requested:
        if not self.experimental_platform_compatible:
          return LiveEligibility.EXPERIMENTAL_PLATFORM_UNSUPPORTED
        if not bundle.torque_limits.production_envelope_verified:
          return LiveEligibility.UNVERIFIED_PRODUCTION_ENVELOPE
        if (
          self.controller_profile is None
          or self.controller_policy is None
          or not self.controller_policy.provisional
          or self.controller_profile.qualified
        ):
          return LiveEligibility.EXPERIMENTAL_CONFIGURATION_INVALID
        if (
          self.horizon_policy is None
          or not self.horizon_policy.provisional
        ):
          return LiveEligibility.HORIZON_CONFIGURATION_INVALID
        return LiveEligibility.ELIGIBLE
      if self.artifact_diagnostic not in (
        ArtifactDiagnostic.OK,
        ArtifactDiagnostic.ABSENT,
      ):
        return LiveEligibility.ACTIVATION_STATE_INVALID
      return LiveEligibility.NO_ACTIVE_ARTIFACT
    if self.artifact_diagnostic != ArtifactDiagnostic.OK:
      return LiveEligibility.ACTIVATION_STATE_INVALID
    if (
      self.horizon_policy is None
      or self.horizon_policy.provisional
      or self.horizon_policy.sha256 != artifact.horizon_policy_sha256
    ):
      # The exact source commit owns this fixed planner policy. Historical
      # artifacts cannot silently acquire the provisional development policy.
      return LiveEligibility.HORIZON_CONFIGURATION_INVALID
    if (
      artifact.vehicle_profile.vehicle_identity
      != bundle.vehicle_identity
    ):
      return LiveEligibility.VEHICLE_IDENTITY_MISMATCH
    if (
      artifact.runtime_vehicle_identity_sha256
      != bundle.identity_sha256
    ):
      return LiveEligibility.RUNTIME_IDENTITY_MISMATCH
    try:
      expected_profile = compose_controller_profile(
        artifact.calibration_profile,
        bundle.seed_profile,
      )
    except (TypeError, ValueError, OverflowError):
      return LiveEligibility.CALIBRATION_COMPOSITION_MISMATCH
    if artifact.vehicle_profile != expected_profile:
      return LiveEligibility.CALIBRATION_COMPOSITION_MISMATCH
    if artifact.source_openpilot_commit != self.source_openpilot_commit:
      return LiveEligibility.SOURCE_COMMIT_MISMATCH
    if artifact.opendbc_commit != self.opendbc_commit:
      return LiveEligibility.OPENDBC_COMMIT_MISMATCH
    if artifact.panda_commit != self.panda_commit:
      return LiveEligibility.PANDA_COMMIT_MISMATCH
    if artifact.controller_policy.provisional:
      return LiveEligibility.PROVISIONAL_POLICY
    if not artifact.vehicle_profile.qualified:
      return LiveEligibility.UNQUALIFIED_PROFILE
    if not bundle.torque_limits.production_envelope_verified:
      return LiveEligibility.UNVERIFIED_PRODUCTION_ENVELOPE
    return LiveEligibility.ELIGIBLE

  @property
  def artifact_sha256(self) -> str:
    return "" if self.artifact is None else self.artifact.artifact_sha256

  @property
  def profile_sha256(self) -> str:
    return (
      ""
      if self.controller_profile is None
      else profile_sha256(self.controller_profile)
    )

  @property
  def policy_sha256(self) -> str:
    return (
      ""
      if self.controller_policy is None
      else self.controller_policy.sha256
    )

  @property
  def horizon_policy_sha256(self) -> str:
    return (
      ""
      if self.horizon_policy is None
      else self.horizon_policy.sha256
    )

  @property
  def last_recorded_applied_torque(self) -> float | None:
    """Return the request-aware actuator input represented by CarOutput."""
    if (
      self.runtime_bundle is None
      or self.last_exact_applied_counts is None
      or not self.last_applied_count_valid
      or not self.last_steering_request_valid
    ):
      return None
    if not self.last_steering_request_active:
      return 0.0
    return (
      self.last_exact_applied_counts
      / self.runtime_bundle.torque_limits.steer_max
    )

  @property
  def runtime_identity_sha256(self) -> str:
    return (
      ""
      if self.runtime_bundle is None
      else self.runtime_bundle.identity_sha256
    )

  @property
  def final_limiter_altered(self) -> bool:
    return self.final_count_match_valid and self.final_count_residual != 0

  def record_requested_command(self, torque: float) -> bool:
    """Bind this control request to the next new CarOutput witness."""
    if self.runtime_bundle is None or self.selection != ControllerSelection.MODULAR:
      self.pending_command_counts = None
      return False
    try:
      normalized = float(torque)
    except (TypeError, ValueError, OverflowError):
      self.pending_command_counts = None
      return False
    steer_max = self.runtime_bundle.torque_limits.steer_max
    if not math.isfinite(normalized):
      self.pending_command_counts = None
      return False
    counts = int(round(normalized * steer_max))
    if abs(counts) > steer_max:
      self.pending_command_counts = None
      return False
    if self.transport_reprime_required:
      return False
    if self.pending_command_counts is not None:
      # A duplicate output leaves command/output correspondence unknown. Keep
      # the original unmatched command until one new output discards it during
      # transport resynchronization; never relabel it as this newer request.
      return False
    self.pending_command_counts = counts
    return True

  def observe_previous_applied(
    self,
    car_output: Any,
    *,
    car_output_mono_ns: int | None = None,
  ) -> bool:
    """Update distinct command-count and physical-request witnesses."""
    self.last_applied_count_valid = False
    self.last_steering_request_valid = False
    self.final_count_match_valid = False
    self.car_output_fresh = False
    self.car_output_cadence_valid = False
    if self.runtime_bundle is None:
      return False
    timestamped = car_output_mono_ns is not None
    if timestamped and (
      type(car_output_mono_ns) is not int
      or car_output_mono_ns < 0
      or (
        self.last_car_output_mono_ns is not None
        and car_output_mono_ns <= self.last_car_output_mono_ns
      )
    ):
      self.transport_reprime_required = True
      return False

    prior_output_mono_ns = self.last_car_output_mono_ns
    self.car_output_fresh = True
    self.car_output_cadence_valid = (
      not timestamped
      or prior_output_mono_ns is None
      or car_output_mono_ns - prior_output_mono_ns
      <= MAX_RECORDED_FRAME_GAP_NS
    )
    if timestamped:
      self.last_car_output_mono_ns = car_output_mono_ns
    correspondence_valid = (
      self.car_output_cadence_valid
      and not self.transport_reprime_required
    )
    expected_counts = (
      self.pending_command_counts if correspondence_valid else None
    )
    self.pending_command_counts = None
    self.last_expected_command_counts = expected_counts
    if not self.car_output_cadence_valid:
      self.transport_reprime_required = True
    counts = exact_applied_torque_counts(
      car_output,
      self.runtime_bundle.torque_limits,
    )
    if counts is None:
      self.transport_reprime_required = True
      return False
    self.last_exact_applied_counts = counts
    self.last_applied_count_valid = True
    if expected_counts is not None:
      self.final_count_residual = counts - expected_counts
      self.final_count_match_valid = True
    request_state = exact_steering_request_state(car_output)
    if request_state is None:
      self.transport_reprime_required = True
      return False
    self.last_steering_request_active = request_state[0]
    self.last_steering_request_counter = request_state[1]
    self.last_steering_request_valid = True
    return not self.transport_reprime_required

  def observe_inactive_state(
    self,
    *,
    state_sample_mono_ns: int,
    car_state: Any,
    live_parameters: Any,
    inputs_valid: bool,
    live_parameters_inputs_valid: bool,
  ) -> None:
    """Warm only the measured-rate derivative while no engagement is bound."""
    if self.enabled_bound or self.adapter is None:
      return
    self.adapter.observe_inactive_state(
      state_sample_mono_ns=state_sample_mono_ns,
      car_state=car_state,
      live_parameters=live_parameters,
      inputs_valid=inputs_valid,
      live_parameters_inputs_valid=live_parameters_inputs_valid,
    )

  def _stock_decision(self) -> EngagementDecision:
    return EngagementDecision(
      selection=ControllerSelection.STOCK,
      profile=None,
      profile_sha256="",
      provisional=False,
    )

  def _modular_decision(self) -> EngagementDecision:
    if self.controller_profile is None:
      raise AssertionError("modular decision has no controller profile")
    return EngagementDecision(
      selection=ControllerSelection.MODULAR,
      profile=self.controller_profile,
      profile_sha256=self.profile_sha256,
      provisional=(
        self.experimental_active or self.activation_provisional
      ),
    )

  def update_engagement(
    self,
    *,
    enabled: bool,
    lateral_active: bool,
    lateral_maneuver_active: bool,
  ) -> ControllerSelection:
    """Bind one immutable lateral-control session.

    AOL/MADS can make lateral active while ``enabled`` is false, so ``enabled
    OR lateral_active`` is both the authorization boundary and the lifetime.
    An eligible approved artifact or authorized trial may therefore bind
    MODULAR at a lateral-only boundary, and later ``enabled`` toggles cannot
    switch controllers while lateral steering remains active.
    """
    enabled_now = bool(enabled)
    lateral_active_now = bool(lateral_active)
    session_active = enabled_now or lateral_active_now
    if not session_active:
      if self.enabled_bound:
        if (
          self.selection == ControllerSelection.MODULAR
          and self.candidate is not None
          and self.candidate.engagement_decision is not None
        ):
          # End with the candidate's exact bound object.  If the wrapper-side
          # decision identity was corrupted, that mismatch has already
          # poisoned the session; it must not prevent the explicit
          # both-false boundary from resetting the candidate for a new one.
          self.candidate.end_engagement(
            self.candidate.engagement_decision,
          )
        if self.activation is not None and self.activation.engaged:
          self.activation.end_engagement()
        if self.adapter is not None:
          self.adapter.reset_derivative()
      self.enabled_bound = False
      self.selection = ControllerSelection.STOCK
      self.decision = None
      self.maneuver_forced_stock = False
      self.binding_reason = self.eligibility
      self.previous_output_constrained = False
      self.previous_lateral_active = False
      self.last_output_constrained_input = False
      self.last_actuator_constrained_input = False
      self.candidate_result = None
      self.previous_control_witness_ns = None
      self.control_cadence_valid = True
      self.transport_reprimed = False
      self.transport_reprime_required = False
      self.pending_command_counts = None
      return self.selection

    if self.enabled_bound:
      # Selection and exact decision identity are immutable until disabled.
      return self.selection

    self.enabled_bound = True
    self.maneuver_forced_stock = bool(lateral_maneuver_active)
    self.binding_reason = self.eligibility
    activation_allows_modular = self.experimental_active
    if not self.experimental_active:
      activation_allows_modular = self.artifact is not None
    if not self.experimental_active and self.activation is not None:
      try:
        approved_decision = self.activation.begin_engagement()
        activation_allows_modular = (
          approved_decision.selection == ControllerSelection.MODULAR
          and approved_decision.artifact is self.artifact
          and approved_decision.provisional
          == self.activation_provisional
        )
      except Exception:
        activation_allows_modular = False
        self.binding_reason = LiveEligibility.ACTIVATION_STATE_INVALID
    modular_eligible = (
      self.eligibility == LiveEligibility.ELIGIBLE
      and activation_allows_modular
      and not self.maneuver_forced_stock
      and self.last_exact_applied_counts is not None
      and self.last_applied_count_valid
      and self.last_steering_request_valid
      and self.car_output_fresh
      and self.car_output_cadence_valid
      and not self.transport_reprime_required
      and self.candidate is not None
    )
    if not modular_eligible:
      if self.maneuver_forced_stock:
        self.binding_reason = LiveEligibility.LATERAL_MANEUVER_MODE
      elif (
        self.eligibility == LiveEligibility.ELIGIBLE
        and not activation_allows_modular
      ):
        self.binding_reason = LiveEligibility.ACTIVATION_STATE_INVALID
      elif (
        self.eligibility == LiveEligibility.ELIGIBLE
        and (
          self.last_exact_applied_counts is None
          or not self.last_applied_count_valid
        )
      ):
        self.binding_reason = LiveEligibility.NO_EXACT_APPLIED_COUNT
      elif (
        self.eligibility == LiveEligibility.ELIGIBLE
        and not self.last_steering_request_valid
      ):
        self.binding_reason = (
          LiveEligibility.NO_EXACT_STEERING_REQUEST_STATE
        )
      self.selection = ControllerSelection.STOCK
      self.decision = self._stock_decision()
      return self.selection

    decision = self._modular_decision()
    try:
      recorded_applied = self.last_recorded_applied_torque
      if recorded_applied is None:
        raise ValueError("request-aware transport state is unavailable")
      # Transport-state initialization only: the previous emitted request is
      # held backward across the fixed delay ring. A carried count whose
      # steering request bit was off contributes zero actuator input.
      self.candidate.prime_transport_state(
        self.last_exact_applied_counts,
        recorded_applied,
      )
      self.candidate.begin_engagement(decision)
    except Exception:
      self.binding_reason = LiveEligibility.CONSTRUCTION_FAILED
      self.selection = ControllerSelection.STOCK
      self.decision = self._stock_decision()
      return self.selection
    self.selection = ControllerSelection.MODULAR
    self.decision = decision
    self.previous_control_witness_ns = None
    self.control_cadence_valid = True
    self.transport_reprimed = False
    return self.selection

  def _observe_control_cadence(self, witness_ns: int) -> bool:
    """Validate one BOOTTIME control witness against the previous frame."""
    current = int(witness_ns)
    previous = self.previous_control_witness_ns
    self.previous_control_witness_ns = current
    if current < 0:
      return False
    if previous is None:
      return True
    gap_ns = current - previous
    return 0 < gap_ns <= MAX_RECORDED_FRAME_GAP_NS

  def update_modular(
    self,
    *,
    state_sample_mono_ns: int,
    model_publication_mono_ns: int,
    model_message: Any,
    car_state: Any,
    live_parameters: Any,
    model_message_valid: bool,
    model_message_alive: bool,
    vehicle_inputs_valid: bool,
    live_parameters_inputs_valid: bool,
    lateral_active: bool,
    actuator_constrained_previous: bool,
    lateral_maneuver_active: bool,
  ) -> CandidateResult:
    """Compute one bound modular frame or route failure through the guard."""
    if (
      self.selection != ControllerSelection.MODULAR
      or not self.enabled_bound
      or self.candidate is None
      or self.adapter is None
      or self.decision is None
      or self.last_exact_applied_counts is None
    ):
      raise RuntimeError("modular update called without a bound candidate")

    engagement_boundary = (
      bool(lateral_active) != self.previous_lateral_active
    )
    self.previous_lateral_active = bool(lateral_active)
    self.last_output_constrained_input = (
      self.previous_output_constrained
    )
    self.last_actuator_constrained_input = bool(
      actuator_constrained_previous
      or not self.last_steering_request_active,
    )
    driver_torque = 0.0
    try:
      driver_torque = float(car_state.steeringTorque)
      if not math.isfinite(driver_torque):
        driver_torque = 0.0
    except (AttributeError, TypeError, ValueError, OverflowError):
      driver_torque = 0.0

    self.control_witness_ns = control_witness_mono_ns()
    start = time.perf_counter()
    self.adapter_exception = False
    self.transport_reprimed = False
    try:
      self.control_cadence_valid = self._observe_control_cadence(
        self.control_witness_ns,
      )
      if not self.control_cadence_valid:
        self.transport_reprime_required = True
      prepared = self.adapter.prepare(
        state_sample_mono_ns=state_sample_mono_ns,
        control_witness_mono_ns=self.control_witness_ns,
        model_publication_mono_ns=model_publication_mono_ns,
        model_message=model_message,
        car_state=car_state,
        live_parameters=live_parameters,
        model_message_valid=model_message_valid,
        model_message_alive=model_message_alive,
        vehicle_inputs_valid=vehicle_inputs_valid,
        live_parameters_inputs_valid=live_parameters_inputs_valid,
      )
      self.prepared_input = prepared
      adaptation = prepared.adaptation
      if adaptation is None:
        raise ValueError("live adapter returned no intent status")

      exact_physical_state_valid = (
        self.last_applied_count_valid
        and self.last_steering_request_valid
        and self.car_output_fresh
      )
      recorded_applied = (
        self.last_recorded_applied_torque
        if exact_physical_state_valid
        else None
      )
      reprime_this_frame = self.transport_reprime_required
      if reprime_this_frame and recorded_applied is not None:
        # The missing/repeated controller interval has unknown 100 Hz command
        # history. Re-seed the ring from the exact command/request pair, then
        # make this frame invalid; the guard owns hold/decay.
        self.candidate.reprime_transport_state(
          self.last_exact_applied_counts,
          recorded_applied,
        )
        self.transport_reprimed = True
        self.transport_reprime_required = False

      # A lateral maneuver process appearing after the enabled boundary is
      # treated as invalid input.  Hot-switching to stock would violate the
      # immutable engagement contract; the normal guard safely decays instead.
      runtime_inputs_valid = (
        exact_physical_state_valid
        and self.car_output_cadence_valid
        and self.control_cadence_valid
        and not reprime_this_frame
        and not lateral_maneuver_active
      )
      if not runtime_inputs_valid:
        result = self.candidate.update_invalid_frame(
          engagement_decision=self.decision,
          previous_command_counts=self.last_exact_applied_counts,
          driver_torque=driver_torque,
          lateral_active=lateral_active,
        )
      else:
        result = self.candidate.update(
          engagement_decision=self.decision,
          previous_command_counts=self.last_exact_applied_counts,
          recorded_applied_torque=recorded_applied,
          driver_torque=prepared.driver_torque,
          frame=adaptation.frame,
          intent_status=adaptation.status,
          intent_plan_times_s=self.adapter.plan_times_s,
          intent_orientation_rates_z=self.adapter.orientation_rates_z,
          intent_velocities_x=self.adapter.velocities_x,
          # Raw model action: clip_curvature remains stock-path-only.
          scalar_curvature=prepared.scalar_curvature,
          current_v_ego_m_s=prepared.current_v_ego_m_s,
          measured_rack_angle_deg=prepared.measured_rack_angle_deg,
          measured_rack_rate_deg_s=prepared.measured_rack_rate_deg_s,
          measured_rack_acceleration_deg_s2=(
            prepared.measured_rack_acceleration_deg_s2
          ),
          lateral_accel_offset=float(
            self.car_params.lateralTuning.torque.latAccelOffset,
          ),
          live_mapping=prepared.live_mapping,
          lateral_active=lateral_active,
          lateral_valid=prepared.lateral_valid,
          engagement_boundary=engagement_boundary,
          live_parameters_valid=prepared.live_parameters_valid,
          steering_pressed=prepared.steering_pressed,
          steering_request_fault_avoidance_counter=(
            self.last_steering_request_counter
          ),
          steering_request_state_valid=(
            self.last_steering_request_valid
          ),
          # These describe the previous frame/recorded response.  The newly
          # calculated request cannot constrain the observer retrospectively.
          actuator_constrained=(
            actuator_constrained_previous
            or not self.last_steering_request_active
          ),
          output_constrained=self.previous_output_constrained,
          standstill=prepared.standstill,
        )
    except Exception:
      self.adapter_exception = True
      result = self.candidate.update_invalid_frame(
        engagement_decision=self.decision,
        previous_command_counts=self.last_exact_applied_counts,
        driver_torque=driver_torque,
        lateral_active=lateral_active,
      )
    self.compute_time_seconds = time.perf_counter() - start
    self.candidate_result = result
    self.previous_output_constrained = bool(
      result.safety_constrained,
    )
    return result

  @property
  def messages_valid(self) -> bool:
    """Only a bound modular guard may invalidate control messages."""
    if (
      self.selection != ControllerSelection.MODULAR
      or self.candidate_result is None
    ):
      return True
    return bool(
      self.candidate_result.controls_valid
      and self.candidate_result.car_control_valid
    )


def construct_modular_live_controller(
  *,
  car_params: car.CarParams,
  car_interface: object,
  params: Any,
  source_openpilot_commit: str,
  opendbc_commit: str,
  panda_commit: str,
) -> ModularLiveController:
  """Fail closed to stock for any optional modular construction error."""
  try:
    return ModularLiveController.from_persistent(
      car_params=car_params,
      car_interface=car_interface,
      params=params,
      source_openpilot_commit=source_openpilot_commit,
      opendbc_commit=opendbc_commit,
      panda_commit=panda_commit,
    )
  except Exception:
    return ModularLiveController(
      car_params=car_params,
      runtime_bundle=None,
      artifact=None,
      activation=None,
      activation_provisional=False,
      artifact_diagnostic=ArtifactDiagnostic.STATE_INVALID,
      source_openpilot_commit=source_openpilot_commit,
      opendbc_commit=opendbc_commit,
      panda_commit=panda_commit,
    )
