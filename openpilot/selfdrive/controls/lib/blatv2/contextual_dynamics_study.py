"""Route-held-out diagnostic for the BLaTv2 transient rack seed.

This module evaluates the existing physical rack equation against recorded
measured response. Recorded controller requests are inputs to the physical
plant, never behavior labels. The result is intentionally diagnostic and has
no artifact, Params, or controller-selection surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SPEED_NODES_MPS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)
PHASES = ("steady", "turn_in", "unwind")
DIRECTIONS = ("left", "right")


@dataclass(frozen=True, slots=True, order=True)
class DynamicsCandidate:
  rack_gain_deg_s2_per_torque: float
  rack_damping_per_s: float

  def __post_init__(self) -> None:
    if (
      not math.isfinite(self.rack_gain_deg_s2_per_torque)
      or self.rack_gain_deg_s2_per_torque <= 0.0
      or not math.isfinite(self.rack_damping_per_s)
      or self.rack_damping_per_s < 0.0
    ):
      raise ValueError("rack candidate must be finite and physical")


@dataclass(frozen=True, slots=True)
class WindowBatch:
  initial_angle_deg: np.ndarray
  initial_rate_deg_s: np.ndarray
  applied_torque: np.ndarray
  aligning_torque: np.ndarray
  dt_s: np.ndarray
  measured_final_angle_deg: np.ndarray
  speed_node_mps: np.ndarray
  phase: tuple[str, ...]
  direction: tuple[str, ...]

  def __post_init__(self) -> None:
    count = len(self.initial_angle_deg)
    if (
      self.initial_angle_deg.shape != (count,)
      or self.initial_rate_deg_s.shape != (count,)
      or self.applied_torque.ndim != 2
      or self.aligning_torque.shape != self.applied_torque.shape
      or self.dt_s.shape != self.applied_torque.shape
      or self.applied_torque.shape[0] != count
      or self.measured_final_angle_deg.shape != (count,)
      or self.speed_node_mps.shape != (count,)
      or len(self.phase) != count
      or len(self.direction) != count
    ):
      raise ValueError("window batch dimensions disagree")
    arrays = (
      self.initial_angle_deg,
      self.initial_rate_deg_s,
      self.applied_torque,
      self.aligning_torque,
      self.dt_s,
      self.measured_final_angle_deg,
      self.speed_node_mps,
    )
    if any(not np.isfinite(value).all() for value in arrays):
      raise ValueError("window batch must be finite")
    if (self.dt_s <= 0.0).any():
      raise ValueError("window sample periods must be positive")
    if any(value not in PHASES for value in self.phase):
      raise ValueError("window phase is invalid")
    if any(value not in DIRECTIONS for value in self.direction):
      raise ValueError("window direction is invalid")

  @property
  def count(self) -> int:
    return len(self.initial_angle_deg)


def nearest_speed_node(speed_mps: np.ndarray) -> np.ndarray:
  speeds = np.asarray(speed_mps, dtype=np.float64)
  if speeds.ndim != 1 or not np.isfinite(speeds).all() or (speeds < 0.0).any():
    raise ValueError("speeds must be one-dimensional, finite, and non-negative")
  nodes = np.asarray(SPEED_NODES_MPS, dtype=np.float64)
  return nodes[np.abs(speeds[:, None] - nodes[None, :]).argmin(axis=1)]


def maneuver_context(
  initial_lateral_accel: float,
  final_lateral_accel: float,
  *,
  phase_threshold_mps2: float = 0.08,
) -> tuple[str, str] | None:
  initial = float(initial_lateral_accel)
  final = float(final_lateral_accel)
  threshold = float(phase_threshold_mps2)
  if (
    not math.isfinite(initial)
    or not math.isfinite(final)
    or not math.isfinite(threshold)
    or threshold <= 0.0
  ):
    raise ValueError("maneuver context inputs are invalid")
  representative = final if abs(final) >= abs(initial) else initial
  if abs(representative) < threshold:
    return None
  direction = "left" if representative > 0.0 else "right"
  if initial * final < 0.0:
    return None
  magnitude_change = abs(final) - abs(initial)
  if magnitude_change > threshold:
    phase = "turn_in"
  elif magnitude_change < -threshold:
    phase = "unwind"
  else:
    phase = "steady"
  return phase, direction


def _step_rack_batch(
  angle_deg: np.ndarray,
  rate_deg_s: np.ndarray,
  applied_torque: np.ndarray,
  aligning_torque: np.ndarray,
  dt_s: np.ndarray,
  candidate: DynamicsCandidate,
  *,
  static_friction_torque: float,
  kinetic_friction_torque: float,
  rack_rate_resolution_deg_s: float,
) -> tuple[np.ndarray, np.ndarray]:
  static = float(static_friction_torque)
  kinetic = float(kinetic_friction_torque)
  resolution = float(rack_rate_resolution_deg_s)
  if (
    not all(math.isfinite(value) for value in (static, kinetic, resolution))
    or static < 0.0
    or kinetic < 0.0
    or kinetic > static
    or resolution < 0.0
  ):
    raise ValueError("friction parameters are outside their physical domain")

  net_before_friction = applied_torque - aligning_torque
  if resolution == 0.0:
    moving_fraction = (rate_deg_s != 0.0).astype(np.float64)
  else:
    moving_fraction = np.minimum(np.abs(rate_deg_s) / resolution, 1.0)
  friction_magnitude = static + moving_fraction * (kinetic - static)
  at_rest = rate_deg_s == 0.0
  held = at_rest & (np.abs(net_before_friction) <= static)
  direction = np.where(at_rest, np.sign(net_before_friction), np.sign(rate_deg_s))
  friction = np.where(held, net_before_friction, direction * friction_magnitude)
  effective_torque = net_before_friction - friction
  acceleration = (
    candidate.rack_gain_deg_s2_per_torque * effective_torque
    - candidate.rack_damping_per_s * rate_deg_s
  )
  next_rate = rate_deg_s + acceleration * dt_s

  crossing = rate_deg_s * next_rate < 0.0
  supports_reversal = (
    (np.abs(net_before_friction) > static)
    & (np.sign(net_before_friction) == -np.sign(rate_deg_s))
  )
  unsupported = crossing & ~supports_reversal
  next_rate = np.where(unsupported, 0.0, next_rate)

  supported = crossing & supports_reversal
  safe_acceleration = np.where(acceleration == 0.0, 1.0, acceleration)
  time_to_zero = np.clip(-rate_deg_s / safe_acceleration, 0.0, dt_s)
  post_effective = net_before_friction - np.sign(net_before_friction) * static
  post_rate = (
    candidate.rack_gain_deg_s2_per_torque
    * post_effective
    * (dt_s - time_to_zero)
  )
  next_rate = np.where(supported, post_rate, next_rate)
  return angle_deg + next_rate * dt_s, next_rate


def predict_final_angles(
  windows: WindowBatch,
  candidate: DynamicsCandidate,
  *,
  static_friction_torque: float,
  kinetic_friction_torque: float,
  rack_rate_resolution_deg_s: float,
) -> np.ndarray:
  angle = windows.initial_angle_deg.copy()
  rate = windows.initial_rate_deg_s.copy()
  for index in range(windows.applied_torque.shape[1]):
    angle, rate = _step_rack_batch(
      angle,
      rate,
      windows.applied_torque[:, index],
      windows.aligning_torque[:, index],
      windows.dt_s[:, index],
      candidate,
      static_friction_torque=static_friction_torque,
      kinetic_friction_torque=kinetic_friction_torque,
      rack_rate_resolution_deg_s=rack_rate_resolution_deg_s,
    )
  return angle


def squared_angle_error(
  windows: WindowBatch,
  candidate: DynamicsCandidate,
  *,
  static_friction_torque: float,
  kinetic_friction_torque: float,
  rack_rate_resolution_deg_s: float,
) -> np.ndarray:
  predicted = predict_final_angles(
    windows,
    candidate,
    static_friction_torque=static_friction_torque,
    kinetic_friction_torque=kinetic_friction_torque,
    rack_rate_resolution_deg_s=rack_rate_resolution_deg_s,
  )
  return np.square(predicted - windows.measured_final_angle_deg)
