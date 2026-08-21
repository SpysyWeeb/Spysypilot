"""Shared immutable contracts for Palisade rack-trajectory control."""
from __future__ import annotations

from dataclasses import dataclass

RESPONSE_TIME_S = .4


@dataclass(frozen=True, slots=True)
class PathTarget:
  curvature: float
  speed_mps: float
  angle_deg: float
  rate_deg_s: float


@dataclass(frozen=True, slots=True)
class MotionLimits:
  max_rate_deg_s: float
  max_acceleration_deg_s2: float
  max_jerk_deg_s3: float
  response_time_s: float = RESPONSE_TIME_S


@dataclass(frozen=True, slots=True)
class RackTarget:
  position_deg: float
  rate_deg_s: float


@dataclass(frozen=True, slots=True)
class RackPlan:
  position_deg: float
  rate_deg_s: float
  acceleration_deg_s2: float
  rate_limited: bool
  acceleration_limited: bool
  jerk_limited: bool


@dataclass(frozen=True, slots=True)
class RackTrajectoryOutput:
  torque: float
  target_curvature: float
  target_angle_deg: float
  target_rate_deg_s: float
  planned_angle_deg: float
  planned_rate_deg_s: float
  planned_acceleration_deg_s2: float
  measured_rate_deg_s: float
  lateral_accel_error: float
  rate_error_deg_s: float
  position_feedback_torque: float
  rate_feedback_torque: float
  feedforward_torque: float
  desired_lateral_accel: float
  actual_lateral_accel: float
  desired_lateral_jerk: float
  feedback_torque: float
  feedback_limited: bool
  motion_limited: bool
  torque_limited: bool
  rate_limit_deg_s: float
  acceleration_limit_deg_s2: float
  jerk_limit_deg_s3: float
  profile_transition: bool
  path_limited: bool
  infeasible: bool
  saturated: bool
