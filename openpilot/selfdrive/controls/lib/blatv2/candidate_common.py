"""Preallocated reference/feedforward workspace shared by both candidates."""

from __future__ import annotations

from bisect import bisect_left
import math

import numpy as np

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.controls.lib.blatv2.controller import DECISION_DT
from openpilot.selfdrive.controls.lib.blatv2.plant import (
  AlignBatchTerms,
  AlignInputs,
  PlantState,
  PlantTwin,
)
from openpilot.selfdrive.modeld.constants import ModelConstants


MAX_REFERENCE_TIME = float(ModelConstants.T_IDXS[-1])
MAX_DECISION_STEPS = int(math.ceil(MAX_REFERENCE_TIME / DECISION_DT)) + 2
# One reference schedule plus early/late placement for every possible adjacent
# sign crossing. Capacity is derived from the runtime horizon grid, not tuned.
MAX_SIGN_SCHEDULES = 1 + 2 * (MAX_DECISION_STEPS - 1)
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


def planned_rack_friction(
  left_rate: float,
  right_rate: float,
  departure_direction: float,
  kinetic_friction: float,
) -> float:
  """Return the one moving-friction feedforward over a reference cell.

  A reference derivative cannot establish whether the physical rack is
  stuck. Static load therefore belongs exclusively to ``PlantTwin``'s
  measured state. The planned inverse uses kinetic friction for every cell,
  including departures and the exact zero-crossing average; the plant solve
  supplies whatever additional request is physically needed to break the
  measured rack free.
  """
  direction = decision_cell_coulomb_direction(
    left_rate, right_rate, departure_direction,
  )
  magnitude = float(kinetic_friction)
  if not math.isfinite(magnitude) or magnitude < 0.0:
    raise ValueError(
      "kinetic friction must be finite and non-negative"
    )
  return magnitude * direction


class CandidateWorkspace:
  """Fixed buffers for the shared scalar-pinned reference and inverse EPS map."""

  def __init__(self) -> None:
    # Python-float buffers are measurably faster than per-element numpy scalar
    # access on comma hardware. They remain fixed-capacity binary64 storage;
    # no list or element count changes on the hot path.
    self.decision_times = [0.0] * MAX_DECISION_STEPS
    self.reference_lower_indices = [0] * MAX_DECISION_STEPS
    self.reference_fractions = [0.0] * MAX_DECISION_STEPS
    self.cached_reference_times = [0.0] * MAX_DECISION_STEPS
    self.reference_curvatures = [0.0] * MAX_DECISION_STEPS
    self.reference_speeds = [0.0] * MAX_DECISION_STEPS
    self.desired_angles = [0.0] * MAX_DECISION_STEPS
    self.desired_rates = [0.0] * MAX_DECISION_STEPS
    self.desired_accelerations = [0.0] * MAX_DECISION_STEPS
    self.friction_directions = [0.0] * MAX_DECISION_STEPS
    self.friction_torques = [0.0] * MAX_DECISION_STEPS
    self.aligning_torques = [0.0] * MAX_DECISION_STEPS
    self.feedforward = [0.0] * MAX_DECISION_STEPS
    self.align_batch_terms = AlignBatchTerms()
    self.decision_count = 0
    self.cached_reference_count = 0
    self.cached_decision_count = 0
    for index in range(MAX_DECISION_STEPS):
      self.decision_times[index] = index * DECISION_DT

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
    reference_speeds: np.ndarray | None = None,
    reference_times_changed: bool | None = None,
  ) -> None:
    """Build the 50 ms decision grid and inverse-EPS feedforward.

    Every planned cell uses the plant twin's single kinetic-friction value.
    Static load is a measured plant state and is handled only by the plant
    rollouts. A cell containing a rack-rate zero crossing uses the exact
    cell-average direction.
    """
    horizon = min(float(horizon_seconds), MAX_REFERENCE_TIME)
    if not math.isfinite(horizon) or horizon < 0.0:
      raise ValueError("candidate horizon must be finite and non-negative")

    decision_count = min(int(math.floor(horizon / DECISION_DT)) + 1, MAX_DECISION_STEPS)
    if decision_count < 2:
      raise ValueError("candidate horizon is too short")

    if reference_speeds is not None and len(reference_speeds) < reference_count:
      raise ValueError("reference speed buffer is shorter than the reference")
    twin.prepare_align_batch_terms(
      align_inputs,
      self.align_batch_terms,
    )

    cache_matches = (
      self.cached_reference_count == reference_count
      and self.cached_decision_count == decision_count
    )
    if reference_times_changed is True:
      cache_matches = False
    elif cache_matches and reference_times_changed is None:
      for index in range(reference_count):
        if (
          self.cached_reference_times[index]
          != float(reference_times[index])
        ):
          cache_matches = False
          break
    if not cache_matches:
      reference_index = 0
      first_reference_time = float(reference_times[0])
      last_reference_time = float(
        reference_times[reference_count - 1]
      )
      previous_time = -math.inf
      for index in range(reference_count):
        reference_time = float(reference_times[index])
        if reference_time <= previous_time:
          raise ValueError(
            "candidate reference times must be strictly increasing"
          )
        self.cached_reference_times[index] = reference_time
        previous_time = reference_time
      for index in range(decision_count):
        time_value = float(self.decision_times[index])
        if time_value <= first_reference_time:
          lower_index = 0
          fraction = 0.0
        elif time_value >= last_reference_time:
          lower_index = reference_count - 1
          fraction = 0.0
        else:
          while (
            reference_index + 1 < reference_count
            and float(reference_times[reference_index + 1])
            <= time_value
          ):
            reference_index += 1
          lower_index = reference_index
          lower_time = float(reference_times[lower_index])
          upper_time = float(reference_times[lower_index + 1])
          fraction = (
            (time_value - lower_time) / (upper_time - lower_time)
          )
        self.reference_lower_indices[index] = lower_index
        self.reference_fractions[index] = fraction
      self.cached_reference_count = reference_count
      self.cached_decision_count = decision_count

    lower_indices = self.reference_lower_indices
    fractions = self.reference_fractions
    reference_curvature_grid = self.reference_curvatures
    reference_speed_grid = self.reference_speeds
    desired_angles = self.desired_angles
    aligning_torques = self.aligning_torques
    batch = self.align_batch_terms
    plant_align = twin.align_params
    slip_factor = batch.slip_factor
    use_roll_compensation = abs(slip_factor) >= 1e-6
    inverse_slip = (
      1.0 / slip_factor if use_roll_compensation else 0.0
    )
    roll_numerator = ACCELERATION_DUE_TO_GRAVITY * batch.roll
    roll_gravity = batch.roll * ACCELERATION_DUE_TO_GRAVITY
    steer_ratio = batch.steer_ratio
    angle_offset = batch.angle_offset_deg
    lat_accel_offset = plant_align.lat_accel_offset
    steer_ratio_rear = plant_align.steer_ratio_rear
    wheelbase = plant_align.wheelbase
    torque_nodes = twin.params.torque_per_lataccel_speed_nodes
    torque_values = twin.params.torque_per_lataccel_values

    # This is the same scalar vehicle-model/alignment arithmetic as
    # ``prepare_align_speed_terms`` + the two prepared transforms, fused into
    # one fixed-buffer pass. A 100 Hz controller cannot afford three Python
    # calls and a mutable terms object for every one of the ~39 horizon cells.
    for index in range(decision_count):
      lower_index = int(lower_indices[index])
      upper_index = min(lower_index + 1, reference_count - 1)
      fraction = float(fractions[index])
      lower_value = float(reference_curvatures[lower_index])
      curvature = lower_value + fraction * (
        float(reference_curvatures[upper_index]) - lower_value
      )
      reference_curvature_grid[index] = curvature
      if reference_speeds is None:
        speed = state.v_ego
      else:
        lower_speed = float(reference_speeds[lower_index])
        speed = lower_speed + fraction * (
          float(reference_speeds[upper_index]) - lower_speed
        )
      speed = max(speed, 0.0)
      if not math.isfinite(speed):
        raise ValueError("vehicle speed must be finite")
      reference_speed_grid[index] = speed
      speed_squared = speed * speed
      curvature_denominator = 1.0 - slip_factor * speed_squared
      if curvature_denominator == 0.0:
        raise ValueError("vehicle-model curvature denominator is zero")
      curvature_factor = (
        (1.0 - steer_ratio_rear)
        / curvature_denominator
        / wheelbase
      )
      if use_roll_compensation:
        roll_denominator = inverse_slip - speed_squared
        if roll_denominator == 0.0:
          raise ValueError("vehicle-model roll denominator is zero")
        roll_compensation = roll_numerator / roll_denominator
      else:
        roll_compensation = 0.0
      if curvature_factor == 0.0:
        raise ValueError("vehicle-model curvature factor is zero")
      desired_angle = (
        math.degrees(
          (-curvature - roll_compensation)
          * steer_ratio
          / curvature_factor
        )
        + angle_offset
      )
      desired_angles[index] = desired_angle
      if speed <= torque_nodes[0]:
        torque_gain = torque_values[0]
      else:
        torque_index = bisect_left(torque_nodes, speed, 1)
        if torque_index >= len(torque_nodes):
          torque_gain = torque_values[-1]
        else:
          torque_fraction = (
            (speed - torque_nodes[torque_index - 1])
            / (
              torque_nodes[torque_index]
              - torque_nodes[torque_index - 1]
            )
          )
          torque_gain = (
            torque_values[torque_index - 1]
            + torque_fraction
            * (
              torque_values[torque_index]
              - torque_values[torque_index - 1]
            )
          )
      aligning_torques[index] = -(
        (
          curvature * speed_squared
          - roll_gravity
          - lat_accel_offset
        )
        * torque_gain
      )

    # The model replans at 20 Hz. A three-point second difference made the
    # action controller respond to individual plan samples as acceleration,
    # which route bb exposed as near-center activity. Interior cells use the
    # derivative of one quadratic least-squares fit over five native model
    # samples. This is a fixed numerical stencil, not a feel/timing filter:
    # position remains the untouched scalar-pinned reference and the fit only
    # makes its local rate/acceleration coherent.
    moving_friction = twin.kinetic_friction
    if not math.isfinite(moving_friction) or moving_friction < 0.0:
      raise ValueError(
        "kinetic friction must be finite and non-negative"
      )
    damping = twin.params.b_steer
    torque_acceleration = twin.params.k_t
    disturbance = float(disturbance_torque)
    if not math.isfinite(disturbance):
      raise ValueError("disturbance torque must be finite")

    rates = self.desired_rates
    accelerations = self.desired_accelerations
    friction_directions = self.friction_directions
    friction_torques = self.friction_torques
    feedforward = self.feedforward
    decision_dt = DECISION_DT
    inverse_ten_dt = 1.0 / (10.0 * decision_dt)
    inverse_two_dt = 1.0 / (2.0 * decision_dt)
    inverse_seven_dt_squared = (
      1.0 / (7.0 * decision_dt * decision_dt)
    )
    inverse_dt = 1.0 / decision_dt
    rate_previous = 0.0
    rate_current = (
      desired_angles[1] - desired_angles[0]
    ) * inverse_dt
    if decision_count == 2:
      rate_next = (
        desired_angles[1] - desired_angles[0]
      ) * inverse_dt
    else:
      rate_next = (
        desired_angles[2] - desired_angles[0]
      ) * inverse_two_dt
    for index in range(decision_count):
      rate = rate_current
      rates[index] = rate
      if 2 <= index < decision_count - 2:
        center_angle = desired_angles[index]
        acceleration = (
          2.0 * (desired_angles[index - 2] - center_angle)
          - (desired_angles[index - 1] - center_angle)
          - (desired_angles[index + 1] - center_angle)
          + 2.0 * (desired_angles[index + 2] - center_angle)
        ) * inverse_seven_dt_squared
      elif index == 0:
        acceleration = (rate_next - rate) * inverse_dt
      elif index == decision_count - 1:
        acceleration = (rate - rate_previous) * inverse_dt
      else:
        acceleration = (
          rate_next - rate_previous
        ) * inverse_two_dt

      accelerations[index] = acceleration
      left_rate = (
        rate
        if index == 0
        else 0.5 * (rate_previous + rate)
      )
      right_rate = (
        rate
        if index == decision_count - 1
        else 0.5 * (rate + rate_next)
      )
      departure = acceleration
      if departure == 0.0:
        departure = desired_angles[index] - state.angle_deg
      magnitude = abs(left_rate) + abs(right_rate)
      if magnitude == 0.0:
        direction = (
          0.0 if departure == 0.0
          else math.copysign(1.0, departure)
        )
      else:
        direction = (left_rate + right_rate) / magnitude
      friction_directions[index] = direction
      friction = moving_friction * direction
      friction_torques[index] = friction
      dynamic = (
        acceleration + damping * rate
      ) / torque_acceleration
      demand = (
        aligning_torques[index]
        + dynamic
        + friction
        + disturbance
      )
      if not math.isfinite(demand):
        raise ValueError("inverse feedforward must be finite")
      if demand < -1.0:
        demand = -1.0
      elif demand > 1.0:
        demand = 1.0
      feedforward[index] = demand

      following_index = index + 2
      rate_previous = rate
      rate_current = rate_next
      if following_index < decision_count:
        if following_index == decision_count - 1:
          rate_next = (
            desired_angles[following_index]
            - desired_angles[following_index - 1]
          ) * inverse_dt
        elif 2 <= following_index < decision_count - 2:
          center_angle = desired_angles[following_index]
          rate_next = (
            -2.0 * (
              desired_angles[following_index - 2] - center_angle
            )
            - (
              desired_angles[following_index - 1] - center_angle
            )
            + (
              desired_angles[following_index + 1] - center_angle
            )
            + 2.0 * (
              desired_angles[following_index + 2] - center_angle
            )
          ) * inverse_ten_dt
        else:
          rate_next = (
            desired_angles[following_index + 1]
            - desired_angles[following_index - 1]
          ) * inverse_two_dt

    self.decision_count = decision_count
