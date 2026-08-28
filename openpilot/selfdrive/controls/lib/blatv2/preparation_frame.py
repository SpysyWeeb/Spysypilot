"""Architecture-neutral measured-frame contract used by route preparation.

Keeping this wire-facing dataclass separate from the learner lifecycle lets a
controller, fitter, or UI-only change leave the cross-architecture preparation
certificate intact.  The production learner imports the same class; there is
no adapter or duplicate representation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MeasuredLearningFrame:
  """One time-aligned physical-response frame; no command intent is accepted."""

  # controlsState is the canonical race witness. Physical regression uses the
  # independent response and effective applied-command clocks below.
  sample_mono_ns: int
  response_mono_ns: int
  applied_report_mono_ns: int
  applied_effective_mono_ns: int
  speed_mps: float
  steering_angle_deg: float
  steering_rate_deg_s: float
  steering_torque: float
  steering_pressed: bool
  standstill: bool
  steer_fault_temporary: bool
  steer_fault_permanent: bool
  can_valid: bool
  can_timeout: bool
  applied_torque: float
  lateral_active: bool
  live_parameters_valid: bool
  angle_offset_valid: bool
  steer_ratio_valid: bool
  stiffness_factor_valid: bool
  angle_offset_deg: float
  steer_ratio: float
  stiffness_factor: float
  roll_rad: float
  inputs_valid: bool
