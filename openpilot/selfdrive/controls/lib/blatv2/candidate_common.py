"""Preallocated reference/feedforward workspace shared by both candidates."""

from __future__ import annotations

import math

import numpy as np

from openpilot.selfdrive.controls.lib.blatv2.controller import DECISION_DT
from openpilot.selfdrive.controls.lib.blatv2.plant import AlignInputs, PlantState, PlantTwin
from openpilot.selfdrive.modeld.constants import ModelConstants


MAX_REFERENCE_TIME = float(ModelConstants.T_IDXS[-1])
MAX_DECISION_STEPS = int(math.ceil(MAX_REFERENCE_TIME / DECISION_DT)) + 2
# One reference schedule plus early/late placement for every possible adjacent
# sign crossing. Capacity is derived from the runtime horizon grid, not tuned.
MAX_SIGN_SCHEDULES = 1 + 2 * (MAX_DECISION_STEPS - 1)
# Palisade steering-rate reports in 4 deg/s quanta. This is a measurement
# resolution, not a feel dial. Blending across one quantum makes static-to-slip
# compensation continuous without adding filter state or hysteresis timing.
RACK_RATE_QUANTUM_DEG_S = 4.0


def decision_cell_coulomb_direction(
  left_rate: float,
  right_rate: float,
  departure_direction: float,
) -> float:
  """Return the exact mean Coulomb direction over one linear-rate cell.

  Away from a zero crossing this is exactly ``±1``. Across a crossing it is
  the positive-motion fraction minus the negative-motion fraction, so the
  inverse feedforward passes continuously through zero instead of commanding
  an instantaneous ``+t_breakaway`` to ``-t_breakaway`` flip. When the rack is
  stationary for the whole cell, the requested departure direction retains
  full breakaway authority; a zero departure remains in stiction.

  The transition width is the existing decision cell, not a new tuning
  constant or filter state.
  """
  left = float(left_rate)
  right = float(right_rate)
  departure = float(departure_direction)
  if not all(math.isfinite(value) for value in (left, right, departure)):
    raise ValueError("Coulomb direction inputs must be finite")
  magnitude = abs(left) + abs(right)
  if magnitude == 0.0:
    if departure == 0.0:
      return 0.0
    return math.copysign(1.0, departure)
  return (left + right) / magnitude


def decision_cell_friction(
  left_rate: float,
  right_rate: float,
  departure_direction: float,
  static_breakaway: float,
  kinetic_friction: float,
) -> float:
  """Return distinct stick/slip feedforward friction.

  A stationary cell departing stiction receives full static breakaway. Any
  cell containing rack motion receives the response-surface kinetic value,
  including the exact zero-crossing average. Plant stiction and the observer
  clamp remain independent at the static ``t_breakaway`` value.
  """
  direction = decision_cell_coulomb_direction(
    left_rate, right_rate, departure_direction,
  )
  stationary = abs(float(left_rate)) + abs(float(right_rate)) == 0.0
  magnitude = float(static_breakaway) if stationary and direction != 0.0 else float(kinetic_friction)
  if not math.isfinite(magnitude) or magnitude < 0.0:
    raise ValueError("friction magnitudes must be finite and non-negative")
  return magnitude * direction


def measured_rack_friction(
  measured_rate: float,
  departure_direction: float,
  static_breakaway: float,
  kinetic_friction: float,
) -> float:
  """Continuous stick/slip compensation from measured rack motion.

  The current command is governed by whether the rack is physically stuck,
  not whether the future reference happens to have an exactly-zero derivative.
  At zero measured rate it receives full breakaway compensation. Across the
  first sensor quantum the magnitude linearly blends to kinetic friction.
  """
  rate = float(measured_rate)
  departure = float(departure_direction)
  static = float(static_breakaway)
  kinetic = float(kinetic_friction)
  if not all(math.isfinite(value) for value in (rate, departure, static, kinetic)):
    raise ValueError("measured rack friction inputs must be finite")
  if static < 0.0 or kinetic < 0.0:
    raise ValueError("friction magnitudes must be non-negative")

  if abs(rate) > 0.0:
    direction = math.copysign(1.0, rate)
  elif departure != 0.0:
    direction = math.copysign(1.0, departure)
  else:
    return 0.0
  slip_fraction = min(abs(rate) / RACK_RATE_QUANTUM_DEG_S, 1.0)
  magnitude = static + slip_fraction * (kinetic - static)
  return direction * magnitude


class CandidateWorkspace:
  """Fixed buffers for the shared scalar-pinned reference and inverse EPS map."""

  def __init__(self) -> None:
    self.decision_times = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.reference_curvatures = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.reference_speeds = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.desired_angles = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.desired_rates = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.desired_accelerations = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.friction_directions = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.friction_torques = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.feedforward = np.empty(MAX_DECISION_STEPS, dtype=np.float64)
    self.decision_count = 0

  def fill(
    self,
    twin: PlantTwin,
    state: PlantState,
    align_inputs: AlignInputs,
    reference_times: np.ndarray,
    reference_curvatures: np.ndarray,
    reference_count: int,
    horizon_seconds: float,
    disturbance_torque: float,
    kinetic_friction: float | None = None,
    reference_speeds: np.ndarray | None = None,
  ) -> None:
    """Build the 50 ms decision grid and inverse-EPS feedforward.

    Full ``t_breakaway`` is retained only for departure from stiction. Moving
    cells use the independently supplied kinetic-friction response-surface
    value. A cell containing a rack-rate zero crossing uses the exact
    cell-average direction. With no override, kinetic friction defaults to
    ``t_breakaway`` and reproduces the shipped controller behavior.
    """
    horizon = min(float(horizon_seconds), MAX_REFERENCE_TIME)
    if not math.isfinite(horizon) or horizon < 0.0:
      raise ValueError("candidate horizon must be finite and non-negative")

    decision_count = min(int(math.floor(horizon / DECISION_DT)) + 1, MAX_DECISION_STEPS)
    if decision_count < 2:
      raise ValueError("candidate horizon is too short")

    if reference_speeds is not None and len(reference_speeds) < reference_count:
      raise ValueError("reference speed buffer is shorter than the reference")

    reference_index = 0
    first_reference_time = float(reference_times[0])
    last_reference_time = float(reference_times[reference_count - 1])
    for index in range(decision_count):
      time_value = index * DECISION_DT
      if time_value <= first_reference_time:
        curvature = float(reference_curvatures[0])
      elif time_value >= last_reference_time:
        curvature = float(reference_curvatures[reference_count - 1])
      else:
        while (
          reference_index + 1 < reference_count
          and float(reference_times[reference_index + 1]) <= time_value
        ):
          reference_index += 1
        upper_index = reference_index + 1
        lower_time = float(reference_times[reference_index])
        upper_time = float(reference_times[upper_index])
        fraction = (time_value - lower_time) / (upper_time - lower_time)
        lower_value = float(reference_curvatures[reference_index])
        curvature = lower_value + fraction * (
          float(reference_curvatures[upper_index]) - lower_value
        )
      self.decision_times[index] = time_value
      self.reference_curvatures[index] = curvature
      if reference_speeds is None:
        speed = state.v_ego
      elif time_value <= first_reference_time:
        speed = float(reference_speeds[0])
      elif time_value >= last_reference_time:
        speed = float(reference_speeds[reference_count - 1])
      else:
        lower_time = float(reference_times[reference_index])
        upper_time = float(reference_times[reference_index + 1])
        fraction = (time_value - lower_time) / (upper_time - lower_time)
        speed = float(reference_speeds[reference_index]) + fraction * (
          float(reference_speeds[reference_index + 1])
          - float(reference_speeds[reference_index])
        )
      self.reference_speeds[index] = max(speed, 0.0)
      self.desired_angles[index] = twin.angle_from_curvature(
        curvature, self.reference_speeds[index], align_inputs,
      )

    # The model replans at 20 Hz. A three-point second difference made the
    # action controller respond to individual plan samples as acceleration,
    # which route bb exposed as near-center activity. Interior cells use the
    # derivative of one quadratic least-squares fit over five native model
    # samples. This is a fixed numerical stencil, not a feel/timing filter:
    # position remains the untouched scalar-pinned reference and the fit only
    # makes its local rate/acceleration coherent.
    for index in range(decision_count):
      if 2 <= index < decision_count - 2:
        center_angle = self.desired_angles[index]
        rate = (
          -2.0 * (self.desired_angles[index - 2] - center_angle)
          - (self.desired_angles[index - 1] - center_angle)
          + (self.desired_angles[index + 1] - center_angle)
          + 2.0 * (self.desired_angles[index + 2] - center_angle)
        ) / (10.0 * DECISION_DT)
      elif index == 0:
        rate = (self.desired_angles[1] - self.desired_angles[0]) / DECISION_DT
      elif index == decision_count - 1:
        rate = (self.desired_angles[index] - self.desired_angles[index - 1]) / DECISION_DT
      else:
        rate = (self.desired_angles[index + 1] - self.desired_angles[index - 1]) / (2.0 * DECISION_DT)
      self.desired_rates[index] = rate

    for index in range(decision_count):
      if 2 <= index < decision_count - 2:
        center_angle = self.desired_angles[index]
        acceleration = (
          2.0 * (self.desired_angles[index - 2] - center_angle)
          - (self.desired_angles[index - 1] - center_angle)
          - (self.desired_angles[index + 1] - center_angle)
          + 2.0 * (self.desired_angles[index + 2] - center_angle)
        ) / (7.0 * DECISION_DT * DECISION_DT)
      elif index == 0:
        acceleration = (self.desired_rates[1] - self.desired_rates[0]) / DECISION_DT
      elif index == decision_count - 1:
        acceleration = (self.desired_rates[index] - self.desired_rates[index - 1]) / DECISION_DT
      else:
        acceleration = (self.desired_rates[index + 1] - self.desired_rates[index - 1]) / (2.0 * DECISION_DT)

      rate = self.desired_rates[index]
      self.desired_accelerations[index] = acceleration
      left_rate = (
        rate
        if index == 0
        else 0.5 * (self.desired_rates[index - 1] + rate)
      )
      right_rate = (
        rate
        if index == decision_count - 1
        else 0.5 * (rate + self.desired_rates[index + 1])
      )
      departure = acceleration
      if departure == 0.0:
        departure = self.desired_angles[index] - state.angle_deg
      direction = decision_cell_coulomb_direction(
        left_rate, right_rate, departure
      )
      self.friction_directions[index] = direction
      moving_friction = (
        twin.params.t_breakaway
        if kinetic_friction is None
        else float(kinetic_friction)
      )
      friction = decision_cell_friction(
        left_rate,
        right_rate,
        departure,
        twin.params.t_breakaway,
        moving_friction,
      )
      self.friction_torques[index] = friction
      aligning = twin.aligning_torque_values(
        self.desired_angles[index], self.reference_speeds[index], align_inputs,
      )
      dynamic = (acceleration + twin.params.b_steer * rate) / twin.params.k_t
      demand = aligning + dynamic + friction + float(disturbance_torque)
      self.feedforward[index] = min(max(demand, -1.0), 1.0)

    self.decision_count = decision_count
