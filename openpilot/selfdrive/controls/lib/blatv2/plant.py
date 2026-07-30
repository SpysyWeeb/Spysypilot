"""Deterministic BLaTv2 steering-rack plant twin.

Numerical contract
------------------
* Plant arithmetic is Python ``float`` (IEEE-754 binary64); no BLAS or
  platform-selected math kernels are used.
* ``PlantState`` is steering-wheel angle in degrees, steering-wheel rate in
  degrees/second, normalized torque on [-1, 1], vehicle speed in m/s, and the
  measured lower bound on static rack load at the current held position.
* Hyundai ``SAS_Speed`` is an unsigned 4 deg/s magnitude. ``SignedRackRate``
  restores direction from consecutive steering-angle measurements before the
  value enters any plant, friction, observer, or controller arithmetic.
* ``k_t`` is deg/s² per normalized effective torque, ``b_steer`` is 1/s, and
  ``t_breakaway`` is normalized torque.
* Tire self-aligning load preserves frozen-v14's measurement convention:
  offset-corrected steering angle enters the vehicle model with live roll,
  stiffness, and steer ratio; measured curvature becomes lateral acceleration;
  and roll gravity plus ``latAccelOffset`` are removed. A seed-file schedule
  then maps that acceleration to normalized torque with linear interpolation
  between speed nodes and flat extrapolation beyond them. Calibration friction
  is deliberately excluded because ``t_breakaway`` owns Coulomb friction.
* Aligning torque is subtracted from applied torque before ``k_t``. Invalid live
  parameters use zero roll/angle offset and nominal stiffness/steer ratio for
  that frame only; inputs are explicit and never retained by the plant.
* Integration is semi-implicit Euler: rate is advanced before angle.
* At exactly zero measured motion, the current applied/alignment/disturbance
  equilibrium supplies a lower bound on the static load at that rack
  position. It is never inferred from a planned rate and never injected as a
  second feedforward command. Above that held envelope, motion transitions to
  the one kinetic-friction law. A numerical rate sign crossing is clamped to
  zero so friction cannot create a one-frame oscillation.
* ``predict`` applies the exact asymmetric limiter to requested torque and
  models the pure actuation delay with zero-order-held samples. Before the
  delayed sequence arrives, the state's measured applied torque is held.
* ``one_step_residual`` consumes already-applied ``carOutput`` torque and does
  not delay it a second time.
* The limiter is the float64 counterpart of frozen-BLaT's
  ``torque_transition_time``: same-sign growth uses ``delta_up``, same-sign
  release uses ``delta_down``, and a sign crossing spends decay budget reaching
  zero before using the remaining frame fraction to build the new sign.
* Seed delay is the physical command-transport delay used by the rack rollout.
  ``liveDelay`` is an independent end-to-end reference-timing input and must
  never replace it. The frozen ``PlantParams`` and ``PlantTwin`` are never
  reconstructed on the hot path.

These rules are shared verbatim by the device shadow and route-audit replay.
Changing any rule is a behavior change and requires a shadow-version bump.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.realtime import DT_CTRL


# Hyundai reports steering-wheel rate in 4 deg/s quanta. This is a sensor
# resolution, not a feel dial. The plant's presliding friction transition uses
# exactly one observable quantum so static load cannot disappear
# discontinuously when predicted motion begins.
RACK_RATE_QUANTUM_DEG_S = 4.0


@dataclass(frozen=True)
class PlantParams:
  k_t: float
  b_steer: float
  t_breakaway: float
  actuation_delay: float
  steer_max: int
  delta_up: int
  delta_down: int
  steer_step: int
  provisional: bool
  torque_per_lataccel_speed_nodes: tuple[float, ...]
  torque_per_lataccel_values: tuple[float, ...]

  @classmethod
  def from_seed_file(cls, path: str | Path, controller_params: Any) -> PlantParams:
    with Path(path).open(encoding="utf-8") as stream:
      seed = json.load(stream)

    steady_state = seed["steady_state_torque_per_lateral_accel"]
    params = cls(
      k_t=float(seed["k_t"]),
      b_steer=float(seed["b_steer"]),
      t_breakaway=float(seed["t_breakaway"]),
      actuation_delay=float(seed["actuation_delay"]),
      steer_max=int(controller_params.STEER_MAX),
      delta_up=int(controller_params.STEER_DELTA_UP),
      delta_down=int(controller_params.STEER_DELTA_DOWN),
      steer_step=int(controller_params.STEER_STEP),
      provisional=bool(seed["provisional"]),
      torque_per_lataccel_speed_nodes=tuple(
        float(value) for value in steady_state["speed_nodes_mps"]
      ),
      torque_per_lataccel_values=tuple(
        float(value) for value in steady_state["torque_per_mps2"]
      ),
    )
    params._validate()
    return params

  def with_actuation_delay(self, actuation_delay: float) -> PlantParams:
    updated = replace(self, actuation_delay=float(actuation_delay))
    updated._validate()
    return updated

  def _validate(self) -> None:
    scalars = (self.k_t, self.b_steer, self.t_breakaway, self.actuation_delay)
    if not all(math.isfinite(value) for value in scalars):
      raise ValueError("plant parameters must be finite")
    if self.k_t <= 0.0 or self.b_steer < 0.0 or self.t_breakaway < 0.0 or self.actuation_delay < 0.0:
      raise ValueError("plant physical parameters are outside their valid domain")
    if self.steer_max <= 0 or self.delta_up <= 0 or self.delta_down <= 0 or self.steer_step <= 0:
      raise ValueError("actuator limits must be positive")
    nodes = self.torque_per_lataccel_speed_nodes
    values = self.torque_per_lataccel_values
    if len(nodes) < 2 or len(nodes) != len(values):
      raise ValueError("steady-state torque schedule must have equal-length nodes and values")
    if not all(math.isfinite(value) for value in (*nodes, *values)):
      raise ValueError("steady-state torque schedule must be finite")
    if nodes[0] < 0.0 or any(right <= left for left, right in zip(nodes, nodes[1:], strict=False)):
      raise ValueError("steady-state torque speed nodes must be non-negative and strictly increasing")
    if any(value <= 0.0 for value in values):
      raise ValueError("steady-state torque gains must be positive")

  def torque_per_lateral_accel(self, v_ego: float) -> float:
    """Return the calibrated steady-state gain with pinned flat extrapolation."""
    speed = abs(float(v_ego))
    if not math.isfinite(speed):
      raise ValueError("vehicle speed must be finite")
    nodes = self.torque_per_lataccel_speed_nodes
    values = self.torque_per_lataccel_values
    if speed <= nodes[0]:
      return values[0]
    index = bisect_left(nodes, speed, 1)
    if index >= len(nodes):
      return values[-1]
    fraction = (
      (speed - nodes[index - 1])
      / (nodes[index] - nodes[index - 1])
    )
    return (
      values[index - 1]
      + fraction * (values[index] - values[index - 1])
    )


@dataclass(frozen=True, slots=True)
class AlignParams:
  mass: float
  wheelbase: float
  center_to_front: float
  tire_stiffness_front: float
  tire_stiffness_rear: float
  nominal_steer_ratio: float
  steer_ratio_rear: float
  lat_accel_offset: float

  @classmethod
  def from_car_params(cls, car_params: Any, torque_params: Any) -> AlignParams:
    params = cls(
      mass=float(car_params.mass),
      wheelbase=float(car_params.wheelbase),
      center_to_front=float(car_params.centerToFront),
      tire_stiffness_front=float(car_params.tireStiffnessFront),
      tire_stiffness_rear=float(car_params.tireStiffnessRear),
      nominal_steer_ratio=float(car_params.steerRatio),
      steer_ratio_rear=float(car_params.steerRatioRear),
      lat_accel_offset=float(torque_params.latAccelOffset),
    )
    params._validate()
    return params

  def _validate(self) -> None:
    values = (
      self.mass,
      self.wheelbase,
      self.center_to_front,
      self.tire_stiffness_front,
      self.tire_stiffness_rear,
      self.nominal_steer_ratio,
      self.steer_ratio_rear,
      self.lat_accel_offset,
    )
    if not all(math.isfinite(value) for value in values):
      raise ValueError("alignment parameters must be finite")
    if (
      self.mass <= 0.0
      or self.wheelbase <= 0.0
      or not 0.0 < self.center_to_front < self.wheelbase
      or self.tire_stiffness_front <= 0.0
      or self.tire_stiffness_rear <= 0.0
      or self.nominal_steer_ratio <= 0.0
    ):
      raise ValueError("alignment parameters are outside their physical domain")


@dataclass(slots=True)
class AlignInputs:
  roll: float
  angle_offset_deg: float
  stiffness_factor: float
  steer_ratio: float
  valid: bool

  def validate(self) -> None:
    if not (
      math.isfinite(self.roll)
      and math.isfinite(self.angle_offset_deg)
      and math.isfinite(self.stiffness_factor)
      and math.isfinite(self.steer_ratio)
    ):
      raise ValueError("alignment inputs must be finite")
    if self.stiffness_factor <= 0.0 or self.steer_ratio <= 0.0:
      raise ValueError("alignment stiffness and steer ratio must be positive")


@dataclass(slots=True)
class PlantState:
  angle_deg: float
  rate_deg_s: float
  applied_torque: float
  v_ego: float
  held_static_load: float = 0.0

  def __post_init__(self) -> None:
    if not all(math.isfinite(value) for value in (
      self.angle_deg,
      self.rate_deg_s,
      self.applied_torque,
      self.v_ego,
      self.held_static_load,
    )):
      raise ValueError("plant state must be finite")
    if self.held_static_load < 0.0:
      raise ValueError("held static load must be non-negative")


@dataclass(slots=True)
class PlantSensitivity:
  """Terminal state derivative with respect to one constant request."""

  angle_per_torque: float = 0.0
  rate_per_torque: float = 0.0


@dataclass(slots=True)
class AlignRuntimeTerms:
  """Caller-owned invariants for repeated dynamics at one speed/alignment.

  Terminal inverse solves advance the same state through several 10 ms steps
  and repeat that rollout for multiple requests. These vehicle-model terms are
  constant throughout. Caller-owned storage keeps the live path allocation
  free while removing redundant validation and coefficient construction.
  """

  speed_squared: float = 0.0
  curvature_factor: float = 0.0
  roll_compensation: float = 0.0
  steer_ratio: float = 1.0
  angle_offset_deg: float = 0.0
  roll_gravity: float = 0.0
  torque_gain: float = 0.0
  torque_per_angle: float = 0.0
  aligning_torque_offset: float = 0.0


@dataclass(slots=True)
class AlignBatchTerms:
  """Alignment invariants shared by every speed cell in one frame."""

  slip_factor: float = 0.0
  roll: float = 0.0
  steer_ratio: float = 1.0
  angle_offset_deg: float = 0.0


@dataclass(slots=True)
class SignedRackRate:
  """Recover Hyundai rack-rate direction without filtering its magnitude.

  Hyundai's classic ``SAS11.SAS_Speed`` field contains only an unsigned
  magnitude. Steering angle supplies the missing direction. When angle
  quantization produces a zero delta while the magnitude is nonzero, the last
  observed motion direction is retained; a zero magnitude reports zero rate
  without forgetting that direction. An explicitly negative input remains
  negative so the shared artifact also accepts platforms/tests that already
  provide a signed rate.
  """

  previous_angle_deg: float = 0.0
  direction: float = 0.0
  initialized: bool = False
  stationary: bool = True

  def reset(self) -> None:
    self.previous_angle_deg = 0.0
    self.direction = 0.0
    self.initialized = False
    self.stationary = True

  def update(self, angle_deg: float, reported_rate_deg_s: float) -> float:
    angle = float(angle_deg)
    reported = float(reported_rate_deg_s)
    if not math.isfinite(angle) or not math.isfinite(reported):
      raise ValueError("rack-rate inputs must be finite")

    magnitude = abs(reported)
    angle_delta = (
      angle - self.previous_angle_deg if self.initialized else 0.0
    )
    if reported < 0.0:
      self.direction = -1.0
    elif self.initialized:
      if angle_delta != 0.0:
        self.direction = math.copysign(1.0, angle_delta)

    # A zero Hyundai SAS_Speed sample is not proof of stiction when the
    # independently measured angle changed. This boolean is used only to
    # decide whether the current equilibrium can identify held static load;
    # rate magnitude remains the unfiltered platform measurement.
    self.stationary = bool(magnitude == 0.0 and angle_delta == 0.0)
    self.previous_angle_deg = angle
    self.initialized = True
    if magnitude == 0.0 or self.direction == 0.0:
      return 0.0
    return self.direction * magnitude


class PlantTwin:
  def __init__(
    self,
    params: PlantParams,
    align_params: AlignParams,
    residual_dt: float = DT_CTRL,
    kinetic_friction: float | None = None,
  ):
    if not math.isfinite(residual_dt) or residual_dt <= 0.0:
      raise ValueError("residual_dt must be finite and positive")
    kinetic = (
      params.t_breakaway
      if kinetic_friction is None
      else float(kinetic_friction)
    )
    if (
      not math.isfinite(kinetic)
      or kinetic < 0.0
      or kinetic > params.t_breakaway
    ):
      raise ValueError(
        "kinetic friction must be finite and within static breakaway"
      )
    self.params = params
    self.align_params = align_params
    self.kinetic_friction = kinetic
    self.residual_dt = float(residual_dt)
    self.nominal_align_inputs = AlignInputs(
      roll=0.0,
      angle_offset_deg=0.0,
      stiffness_factor=1.0,
      steer_ratio=align_params.nominal_steer_ratio,
      valid=False,
    )

  @staticmethod
  def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)

  def apply_slew(self, prev_torque: float, requested: float) -> float:
    prev = self._clip(prev_torque, -1.0, 1.0)
    target = self._clip(requested, -1.0, 1.0)
    if target == prev:
      return target

    build = self.params.delta_up / self.params.steer_max
    decay = self.params.delta_down / self.params.steer_max

    if prev * target >= 0.0:
      budget = build if abs(target) > abs(prev) else decay
      return prev + math.copysign(min(abs(target - prev), budget), target - prev)

    # Spend the frame fraction reaching zero at the decay rate. Only the
    # remaining fraction can build torque in the requested direction.
    decay_fraction = abs(prev) / decay
    if decay_fraction >= 1.0:
      return math.copysign(abs(prev) - decay, prev)

    remaining_fraction = 1.0 - decay_fraction
    built = min(abs(target), build * remaining_fraction)
    return math.copysign(built, target)

  def aligning_torque(self, state: PlantState, align_inputs: AlignInputs) -> float:
    """Return the frozen-v14 steady torque required at the measured wheel angle."""
    return self._aligning_torque(state.angle_deg, state.v_ego, align_inputs)

  def aligning_torque_values(self, angle_deg: float, v_ego: float, align_inputs: AlignInputs) -> float:
    """Allocation-free scalar form used by both shadow candidates."""
    return self._aligning_torque(float(angle_deg), float(v_ego), align_inputs)

  def observed_held_static_load(
    self,
    state: PlantState,
    align_inputs: AlignInputs,
    disturbance_torque: float,
    rack_stationary: bool,
    eligible: bool,
  ) -> float:
    """Return the measured lower bound on load holding the rack stationary.

    Static load is not observable from a planned path derivative. It is
    observable, without a fitted angle schedule, when the measured rack did
    not move: applied torque minus the modelled aligning and disturbance loads
    is the load that the rack just held. The value is a lower bound—not an
    identified maximum—so it is recomputed from the current physical state
    every frame and is retained only through the first observable rack-rate
    quantum while the plant transitions continuously to kinetic friction.

    Driver interaction, inactive control, invalid alignment, and measured
    motion make the equilibrium ineligible. The physical seed remains the
    deterministic fallback. The load is deliberately not clipped to actuator
    authority: aligning load plus scrub can exceed normalized torque 1.0, and
    that is precisely the condition in which the inverse must request full
    authority rather than predict motion the rack cannot make.
    """
    disturbance = float(disturbance_torque)
    if not math.isfinite(disturbance):
      raise ValueError("disturbance torque must be finite")
    base = self.params.t_breakaway
    if (
      not eligible
      or not rack_stationary
      or state.rate_deg_s != 0.0
      or not align_inputs.valid
    ):
      return base
    aligning = self.aligning_torque(state, align_inputs)
    equilibrium = abs(
      self._clip(state.applied_torque, -1.0, 1.0)
      - aligning
      - disturbance
    )
    return max(base, equilibrium)

  def _initial_static_load(self, state: PlantState) -> float:
    return max(
      float(state.held_static_load),
      self.params.t_breakaway,
    )

  def _advanced_static_load(
    self,
    static_load: float,
    rate: float,
  ) -> float:
    return (
      self.params.t_breakaway
      if abs(rate) >= RACK_RATE_QUANTUM_DEG_S
      else static_load
    )

  def inverse_friction_torque(
    self,
    state: PlantState,
    departure_direction: float,
  ) -> float:
    """Return the exact friction term inverted by the controller.

    This is the inverse view of :meth:`_friction_effective_torque`, not a
    second compensation mechanism. A physically stationary rack uses its
    measured held-load state in the requested departure direction. A moving
    rack uses the same continuous presliding-to-kinetic law as every forward
    rollout. Planned future cells never call this method because they have no
    measured rack state and use kinetic friction only.
    """
    departure = float(departure_direction)
    if not math.isfinite(departure):
      raise ValueError("departure direction must be finite")
    rate = float(state.rate_deg_s)
    direction = rate if rate != 0.0 else departure
    if direction == 0.0:
      return 0.0
    static_load = self._initial_static_load(state)
    slip_fraction = min(
      abs(rate) / RACK_RATE_QUANTUM_DEG_S, 1.0,
    )
    friction = static_load + slip_fraction * (
      self.kinetic_friction - static_load
    )
    return math.copysign(friction, direction)

  def _friction_effective_torque(
    self,
    net_torque: float,
    rate: float,
    static_load: float,
  ) -> float:
    """Apply the single stick/slip law used by every plant rollout."""
    if rate == 0.0:
      if abs(net_torque) <= static_load:
        return 0.0
      direction = net_torque
    else:
      direction = rate
    slip_fraction = min(abs(rate) / RACK_RATE_QUANTUM_DEG_S, 1.0)
    friction = static_load + slip_fraction * (
      self.kinetic_friction - static_load
    )
    return net_torque - math.copysign(friction, direction)

  def prepare_align_runtime_terms(
    self,
    v_ego: float,
    align_inputs: AlignInputs,
    target: AlignRuntimeTerms,
  ) -> None:
    """Prepare invariant vehicle-model terms for repeated rack rollouts."""
    align_inputs.validate()
    self._prepare_align_runtime_terms(
      v_ego,
      align_inputs,
      target,
    )

  def prepare_align_batch_terms(
    self,
    align_inputs: AlignInputs,
    target: AlignBatchTerms,
  ) -> None:
    """Prepare frame-wide alignment invariants for a speed trajectory."""
    align_inputs.validate()
    p = self.align_params
    front_stiffness = (
      align_inputs.stiffness_factor * p.tire_stiffness_front
    )
    rear_stiffness = (
      align_inputs.stiffness_factor * p.tire_stiffness_rear
    )
    center_to_rear = p.wheelbase - p.center_to_front
    target.slip_factor = (
      p.mass
      * (
        front_stiffness * p.center_to_front
        - rear_stiffness * center_to_rear
      )
      / (
        p.wheelbase
        * p.wheelbase
        * front_stiffness
        * rear_stiffness
      )
    )
    target.roll = align_inputs.roll
    target.steer_ratio = align_inputs.steer_ratio
    target.angle_offset_deg = align_inputs.angle_offset_deg

  def prepare_align_speed_terms(
    self,
    v_ego: float,
    batch: AlignBatchTerms,
    target: AlignRuntimeTerms,
  ) -> None:
    """Prepare speed-varying terms from frame-wide alignment invariants."""
    p = self.align_params
    speed = float(v_ego)
    if not math.isfinite(speed):
      raise ValueError("vehicle speed must be finite")
    slip_factor = batch.slip_factor
    curvature_denominator = 1.0 - slip_factor * speed * speed
    if curvature_denominator == 0.0:
      raise ValueError("vehicle-model curvature denominator is zero")
    target.curvature_factor = (
      (1.0 - p.steer_ratio_rear)
      / curvature_denominator
      / p.wheelbase
    )
    if abs(slip_factor) < 1e-6:
      target.roll_compensation = 0.0
    else:
      roll_denominator = 1.0 / slip_factor - speed * speed
      if roll_denominator == 0.0:
        raise ValueError("vehicle-model roll denominator is zero")
      target.roll_compensation = (
        ACCELERATION_DUE_TO_GRAVITY
        * batch.roll
        / roll_denominator
      )
    target.speed_squared = speed * speed
    target.steer_ratio = batch.steer_ratio
    target.angle_offset_deg = batch.angle_offset_deg
    target.roll_gravity = batch.roll * ACCELERATION_DUE_TO_GRAVITY
    target.torque_gain = self.params.torque_per_lateral_accel(speed)
    target.torque_per_angle = (
      target.curvature_factor
      * math.radians(1.0)
      / batch.steer_ratio
      * target.speed_squared
      * target.torque_gain
    )
    zero_angle_curvature = -(
      target.curvature_factor
      * math.radians(-target.angle_offset_deg)
      / target.steer_ratio
      + target.roll_compensation
    )
    target.aligning_torque_offset = -(
      (
        zero_angle_curvature * target.speed_squared
        - target.roll_gravity
        - self.align_params.lat_accel_offset
      )
      * target.torque_gain
    )

  def _prepare_align_runtime_terms(
    self,
    v_ego: float,
    align_inputs: AlignInputs,
    target: AlignRuntimeTerms,
  ) -> None:
    """Prepared-term implementation for callers validating once per batch."""
    p = self.align_params
    speed = float(v_ego)
    if not math.isfinite(speed):
      raise ValueError("vehicle speed must be finite")
    front_stiffness = (
      align_inputs.stiffness_factor * p.tire_stiffness_front
    )
    rear_stiffness = (
      align_inputs.stiffness_factor * p.tire_stiffness_rear
    )
    center_to_rear = p.wheelbase - p.center_to_front
    slip_factor = (
      p.mass
      * (
        front_stiffness * p.center_to_front
        - rear_stiffness * center_to_rear
      )
      / (
        p.wheelbase
        * p.wheelbase
        * front_stiffness
        * rear_stiffness
      )
    )
    curvature_denominator = 1.0 - slip_factor * speed * speed
    if curvature_denominator == 0.0:
      raise ValueError("vehicle-model curvature denominator is zero")
    curvature_factor = (
      (1.0 - p.steer_ratio_rear)
      / curvature_denominator
      / p.wheelbase
    )
    if abs(slip_factor) < 1e-6:
      roll_compensation = 0.0
    else:
      roll_denominator = 1.0 / slip_factor - speed * speed
      if roll_denominator == 0.0:
        raise ValueError("vehicle-model roll denominator is zero")
      roll_compensation = (
        ACCELERATION_DUE_TO_GRAVITY
        * align_inputs.roll
        / roll_denominator
      )

    target.speed_squared = speed * speed
    target.curvature_factor = curvature_factor
    target.roll_compensation = roll_compensation
    target.steer_ratio = align_inputs.steer_ratio
    target.angle_offset_deg = align_inputs.angle_offset_deg
    target.roll_gravity = (
      align_inputs.roll * ACCELERATION_DUE_TO_GRAVITY
    )
    target.torque_gain = self.params.torque_per_lateral_accel(speed)
    target.torque_per_angle = (
      curvature_factor
      * math.radians(1.0)
      / align_inputs.steer_ratio
      * target.speed_squared
      * target.torque_gain
    )
    zero_angle_curvature = -(
      curvature_factor
      * math.radians(-target.angle_offset_deg)
      / target.steer_ratio
      + roll_compensation
    )
    target.aligning_torque_offset = -(
      (
        zero_angle_curvature * target.speed_squared
        - target.roll_gravity
        - self.align_params.lat_accel_offset
      )
      * target.torque_gain
    )

  @staticmethod
  def angle_from_curvature_prepared(
    curvature: float,
    terms: AlignRuntimeTerms,
  ) -> float:
    """Exact inverse vehicle-model angle from prepared invariant terms."""
    if terms.curvature_factor == 0.0:
      raise ValueError("vehicle-model curvature factor is zero")
    steering_angle_rad = (
      (-float(curvature) - terms.roll_compensation)
      * terms.steer_ratio
      / terms.curvature_factor
    )
    return (
      math.degrees(steering_angle_rad) + terms.angle_offset_deg
    )

  def _aligning_torque_prepared(
    self,
    angle_deg: float,
    terms: AlignRuntimeTerms,
  ) -> float:
    """Evaluate the exact aligning map from prepared invariant terms."""
    return (
      terms.torque_per_angle * float(angle_deg)
      + terms.aligning_torque_offset
    )

  def aligning_torque_from_curvature_prepared(
    self,
    curvature: float,
    terms: AlignRuntimeTerms,
  ) -> float:
    """Steady torque for a desired curvature on the same prepared map."""
    gravity_adjusted = (
      float(curvature) * terms.speed_squared
      - terms.roll_gravity
      - self.align_params.lat_accel_offset
    )
    return -(gravity_adjusted * terms.torque_gain)

  def curvature_from_angle(self, angle_deg: float, v_ego: float, align_inputs: AlignInputs) -> float:
    """Mirror the frozen-v14 offset/roll vehicle-model measurement pipeline."""
    align_inputs.validate()
    p = self.align_params
    speed = float(v_ego)
    front_stiffness = align_inputs.stiffness_factor * p.tire_stiffness_front
    rear_stiffness = align_inputs.stiffness_factor * p.tire_stiffness_rear
    center_to_rear = p.wheelbase - p.center_to_front
    slip_factor = (
      p.mass * (front_stiffness * p.center_to_front - rear_stiffness * center_to_rear)
      / (p.wheelbase * p.wheelbase * front_stiffness * rear_stiffness)
    )
    curvature_denominator = 1.0 - slip_factor * speed * speed
    if curvature_denominator == 0.0:
      raise ValueError("vehicle-model curvature denominator is zero")
    curvature_factor = (1.0 - p.steer_ratio_rear) / curvature_denominator / p.wheelbase
    if abs(slip_factor) < 1e-6:
      roll_compensation = 0.0
    else:
      roll_denominator = 1.0 / slip_factor - speed * speed
      if roll_denominator == 0.0:
        raise ValueError("vehicle-model roll denominator is zero")
      roll_compensation = ACCELERATION_DUE_TO_GRAVITY * align_inputs.roll / roll_denominator
    steering_angle_rad = math.radians(float(angle_deg) - align_inputs.angle_offset_deg)
    return -(curvature_factor * steering_angle_rad / align_inputs.steer_ratio + roll_compensation)

  def angle_from_curvature(self, curvature: float, v_ego: float, align_inputs: AlignInputs) -> float:
    """Exact scalar inverse of :meth:`curvature_from_angle`."""
    align_inputs.validate()
    p = self.align_params
    speed = float(v_ego)
    front_stiffness = align_inputs.stiffness_factor * p.tire_stiffness_front
    rear_stiffness = align_inputs.stiffness_factor * p.tire_stiffness_rear
    center_to_rear = p.wheelbase - p.center_to_front
    slip_factor = (
      p.mass * (front_stiffness * p.center_to_front - rear_stiffness * center_to_rear)
      / (p.wheelbase * p.wheelbase * front_stiffness * rear_stiffness)
    )
    curvature_denominator = 1.0 - slip_factor * speed * speed
    if curvature_denominator == 0.0:
      raise ValueError("vehicle-model curvature denominator is zero")
    curvature_factor = (1.0 - p.steer_ratio_rear) / curvature_denominator / p.wheelbase
    if curvature_factor == 0.0:
      raise ValueError("vehicle-model curvature factor is zero")
    if abs(slip_factor) < 1e-6:
      roll_compensation = 0.0
    else:
      roll_denominator = 1.0 / slip_factor - speed * speed
      if roll_denominator == 0.0:
        raise ValueError("vehicle-model roll denominator is zero")
      roll_compensation = ACCELERATION_DUE_TO_GRAVITY * align_inputs.roll / roll_denominator
    steering_angle_rad = (-float(curvature) - roll_compensation) * align_inputs.steer_ratio / curvature_factor
    return math.degrees(steering_angle_rad) + align_inputs.angle_offset_deg

  def _aligning_torque(self, angle_deg: float, v_ego: float, align_inputs: AlignInputs) -> float:
    align_inputs.validate()
    p = self.align_params
    speed = float(v_ego)
    front_stiffness = align_inputs.stiffness_factor * p.tire_stiffness_front
    rear_stiffness = align_inputs.stiffness_factor * p.tire_stiffness_rear
    center_to_rear = p.wheelbase - p.center_to_front
    slip_factor = (
      p.mass * (front_stiffness * p.center_to_front - rear_stiffness * center_to_rear)
      / (p.wheelbase * p.wheelbase * front_stiffness * rear_stiffness)
    )
    curvature_denominator = 1.0 - slip_factor * speed * speed
    if curvature_denominator == 0.0:
      raise ValueError("vehicle-model curvature denominator is zero")
    curvature_factor = (1.0 - p.steer_ratio_rear) / curvature_denominator / p.wheelbase
    if abs(slip_factor) < 1e-6:
      roll_compensation = 0.0
    else:
      roll_denominator = 1.0 / slip_factor - speed * speed
      if roll_denominator == 0.0:
        raise ValueError("vehicle-model roll denominator is zero")
      roll_compensation = ACCELERATION_DUE_TO_GRAVITY * align_inputs.roll / roll_denominator

    steering_angle_rad = math.radians(angle_deg - align_inputs.angle_offset_deg)
    vehicle_model_curvature = curvature_factor * steering_angle_rad / align_inputs.steer_ratio + roll_compensation
    measured_curvature = -vehicle_model_curvature
    measured_lateral_accel = measured_curvature * speed * speed
    gravity_adjusted = (
      measured_lateral_accel
      - align_inputs.roll * ACCELERATION_DUE_TO_GRAVITY
      - p.lat_accel_offset
    )
    # Platform output is the negative of the signed log-space fit. The gain is
    # a magnitude; rate-sign hysteresis is modeled separately by t_breakaway.
    return -(
      gravity_adjusted
      * self.params.torque_per_lateral_accel(speed)
    )

  def _next_rate(
    self,
    angle: float,
    rate: float,
    applied_torque: float,
    v_ego: float,
    align_inputs: AlignInputs,
    dt: float,
    disturbance_torque: float = 0.0,
    static_load: float | None = None,
  ) -> float:
    torque = self._clip(applied_torque, -1.0, 1.0)
    aligning_torque = self._aligning_torque(angle, v_ego, align_inputs)
    disturbance = float(disturbance_torque)
    if not math.isfinite(disturbance):
      raise ValueError("disturbance torque must be finite")
    net_torque = torque - aligning_torque - disturbance
    breakaway = (
      self.params.t_breakaway
      if static_load is None
      else float(static_load)
    )
    if not math.isfinite(breakaway) or breakaway < self.params.t_breakaway:
      raise ValueError("static load must be finite and at least breakaway")
    effective_torque = self._friction_effective_torque(
      net_torque, rate, breakaway,
    )

    acceleration = self.params.k_t * effective_torque - self.params.b_steer * rate
    next_rate = rate + acceleration * dt
    if rate != 0.0 and next_rate * rate < 0.0:
      next_rate = 0.0
    return next_rate

  def _next_rate_prepared(
    self,
    angle: float,
    rate: float,
    applied_torque: float,
    dt: float,
    disturbance_torque: float,
    terms: AlignRuntimeTerms,
    static_load: float,
  ) -> float:
    """Hot-path counterpart of :meth:`_next_rate` for prepared rollouts."""
    aligning_torque = self._aligning_torque_prepared(angle, terms)
    net_torque = (
      applied_torque - aligning_torque - disturbance_torque
    )
    effective_torque = self._friction_effective_torque(
      net_torque, rate, static_load,
    )

    acceleration = (
      self.params.k_t * effective_torque
      - self.params.b_steer * rate
    )
    next_rate = rate + acceleration * dt
    if rate != 0.0 and next_rate * rate < 0.0:
      next_rate = 0.0
    return next_rate

  def _apply_slew_bounded(
    self,
    prev_torque: float,
    target: float,
  ) -> float:
    """Exact limiter for inputs already proven to be within [-1, 1]."""
    if target == prev_torque:
      return target

    build = self.params.delta_up / self.params.steer_max
    decay = self.params.delta_down / self.params.steer_max
    if prev_torque * target >= 0.0:
      budget = (
        build if abs(target) > abs(prev_torque) else decay
      )
      return prev_torque + math.copysign(
        min(abs(target - prev_torque), budget),
        target - prev_torque,
      )

    decay_fraction = abs(prev_torque) / decay
    if decay_fraction >= 1.0:
      return math.copysign(
        abs(prev_torque) - decay, prev_torque,
      )
    remaining_fraction = 1.0 - decay_fraction
    built = min(abs(target), build * remaining_fraction)
    return math.copysign(built, target)

  def _advance(
    self,
    angle: float,
    rate: float,
    applied_torque: float,
    v_ego: float,
    align_inputs: AlignInputs,
    dt: float,
    disturbance_torque: float = 0.0,
    static_load: float | None = None,
  ) -> tuple[float, float]:
    next_rate = self._next_rate(
      angle,
      rate,
      applied_torque,
      v_ego,
      align_inputs,
      dt,
      disturbance_torque,
      static_load,
    )
    next_angle = angle + next_rate * dt
    return next_angle, next_rate

  def advance_applied(
    self,
    state: PlantState,
    applied_torque: float,
    dt: float,
    align_inputs: AlignInputs,
    disturbance_torque: float = 0.0,
  ) -> PlantState:
    """Advance one already-applied sample for delivered-path reconstruction.

    The caller owns actuator slew and pure-delay history. This method therefore
    applies neither a second limiter nor a second delay; it is the public,
    allocation-minimal counterpart of the one-step residual dynamics used by
    the promotion harness.
    """
    if not math.isfinite(dt) or dt <= 0.0:
      raise ValueError("dt must be finite and positive")
    static_load = self._initial_static_load(state)
    angle, rate = self._advance(
      float(state.angle_deg),
      float(state.rate_deg_s),
      float(applied_torque),
      float(state.v_ego),
      align_inputs,
      float(dt),
      float(disturbance_torque),
      static_load,
    )
    static_load = self._advanced_static_load(static_load, rate)
    return PlantState(
      angle_deg=angle,
      rate_deg_s=rate,
      applied_torque=float(applied_torque),
      v_ego=float(state.v_ego),
      held_static_load=static_load,
    )

  def predict_held_state_into(
    self,
    state: PlantState,
    duration: float,
    align_inputs: AlignInputs,
    disturbance_torque: float,
    target: PlantState,
    max_step: float,
  ) -> None:
    """Predict the measured rack to the actuator action time.

    This is delay compensation, not future-path preview: the already-applied
    torque is held while the measured rack state advances only through the
    latency that separates this control decision from rack response. The
    caller supplies reusable storage so the live path allocates nothing.
    """
    remaining = float(duration)
    step_limit = float(max_step)
    if not math.isfinite(remaining) or remaining < 0.0:
      raise ValueError("prediction duration must be finite and non-negative")
    if not math.isfinite(step_limit) or step_limit <= 0.0:
      raise ValueError("prediction step must be finite and positive")
    angle = float(state.angle_deg)
    rate = float(state.rate_deg_s)
    static_load = self._initial_static_load(state)
    while remaining > 0.0:
      step = min(remaining, step_limit)
      angle, rate = self._advance(
        angle,
        rate,
        state.applied_torque,
        state.v_ego,
        align_inputs,
        step,
        disturbance_torque,
        static_load,
      )
      static_load = self._advanced_static_load(static_load, rate)
      remaining -= step
    target.angle_deg = angle
    target.rate_deg_s = rate
    target.applied_torque = float(state.applied_torque)
    target.v_ego = float(state.v_ego)
    target.held_static_load = static_load

  def predict_applied_history_into(
    self,
    state: PlantState,
    duration: float,
    applied_history: Any,
    history_start: int,
    history_count: int,
    align_inputs: AlignInputs,
    disturbance_torque: float,
    target: PlantState,
    max_step: float,
  ) -> None:
    """Predict to this request's effect time through already-queued torque.

    ``state`` is the measured rack now. Commands emitted during the preceding
    pure-delay interval are already committed and will reach the rack before
    the command being computed now can do so. They are supplied oldest first
    through a caller-owned circular buffer. If engagement has not yet filled
    the delay line, the unknown prefix holds the currently applied torque.

    History samples are already slew-limited actuator commands. This method
    therefore applies neither another limiter nor another delay, and allocates
    no temporary sequence.
    """
    duration_value = float(duration)
    step_limit = float(max_step)
    start = int(history_start)
    count = int(history_count)
    capacity = len(applied_history)
    if not math.isfinite(duration_value) or duration_value < 0.0:
      raise ValueError("prediction duration must be finite and non-negative")
    if not math.isfinite(step_limit) or step_limit <= 0.0:
      raise ValueError("prediction step must be finite and positive")
    if count < 0 or count > capacity:
      raise ValueError("applied history count is outside buffer bounds")
    if capacity == 0 and count != 0:
      raise ValueError("empty applied history cannot contain samples")
    if capacity and not 0 <= start < capacity:
      raise ValueError("applied history start is outside buffer bounds")

    step_count = int(math.ceil(duration_value / step_limit))
    retained_count = min(count, step_count)
    missing_count = step_count - retained_count
    retained_start = count - retained_count
    angle = float(state.angle_deg)
    rate = float(state.rate_deg_s)
    applied = float(state.applied_torque)
    static_load = self._initial_static_load(state)
    for step_index in range(step_count):
      step = (
        step_limit
        if step_index + 1 < step_count
        else duration_value - step_limit * (step_count - 1)
      )
      if step_index >= missing_count:
        logical_index = retained_start + step_index - missing_count
        physical_index = (start + logical_index) % capacity
        applied = float(applied_history[physical_index])
      if not math.isfinite(applied):
        raise ValueError("applied history must be finite")
      angle, rate = self._advance(
        angle,
        rate,
        applied,
        state.v_ego,
        align_inputs,
        step,
        disturbance_torque,
        static_load,
      )
      static_load = self._advanced_static_load(static_load, rate)
    target.angle_deg = angle
    target.rate_deg_s = rate
    target.applied_torque = applied
    target.v_ego = float(state.v_ego)
    target.held_static_load = static_load

  def predict_applied_history_prepared_into(
    self,
    state: PlantState,
    duration: float,
    applied_history: Any,
    history_start: int,
    history_count: int,
    disturbance_torque: float,
    runtime_terms: AlignRuntimeTerms,
    target: PlantState,
    max_step: float,
  ) -> None:
    """Prepared hot-path form of :meth:`predict_applied_history_into`.

    The vehicle speed and live alignment inputs are invariant throughout the
    pure-delay rollout. Preparing their exact coefficients once removes a
    complete vehicle-model reconstruction from every 10 ms history sample;
    limiter, friction, integration, and history-selection semantics remain
    identical.
    """
    duration_value = float(duration)
    step_limit = float(max_step)
    start = int(history_start)
    count = int(history_count)
    capacity = len(applied_history)
    disturbance = float(disturbance_torque)
    if not math.isfinite(duration_value) or duration_value < 0.0:
      raise ValueError(
        "prediction duration must be finite and non-negative"
      )
    if not math.isfinite(step_limit) or step_limit <= 0.0:
      raise ValueError("prediction step must be finite and positive")
    if not math.isfinite(disturbance):
      raise ValueError("disturbance torque must be finite")
    if count < 0 or count > capacity:
      raise ValueError("applied history count is outside buffer bounds")
    if capacity == 0 and count != 0:
      raise ValueError("empty applied history cannot contain samples")
    if capacity and not 0 <= start < capacity:
      raise ValueError("applied history start is outside buffer bounds")

    step_count = int(math.ceil(duration_value / step_limit))
    retained_count = min(count, step_count)
    missing_count = step_count - retained_count
    retained_start = count - retained_count
    angle = float(state.angle_deg)
    rate = float(state.rate_deg_s)
    applied = float(state.applied_torque)
    static_load = self._initial_static_load(state)
    for step_index in range(step_count):
      step = (
        step_limit
        if step_index + 1 < step_count
        else duration_value - step_limit * (step_count - 1)
      )
      if step_index >= missing_count:
        logical_index = retained_start + step_index - missing_count
        physical_index = (start + logical_index) % capacity
        applied = float(applied_history[physical_index])
      if not math.isfinite(applied):
        raise ValueError("applied history must be finite")
      rate = self._next_rate_prepared(
        angle,
        rate,
        applied,
        step,
        disturbance,
        runtime_terms,
        static_load,
      )
      angle += rate * step
      static_load = self._advanced_static_load(static_load, rate)
    target.angle_deg = angle
    target.rate_deg_s = rate
    target.applied_torque = applied
    target.v_ego = float(state.v_ego)
    target.held_static_load = static_load

  def predict_constant_request_into(
    self,
    state: PlantState,
    duration: float,
    requested_torque: float,
    align_inputs: AlignInputs,
    disturbance_torque: float,
    target: PlantState,
    max_step: float = DT_CTRL,
  ) -> None:
    """Roll out one constant request through the exact 409/4/7 limiter.

    This is the scalar numerical primitive for the live action-point inverse.
    It allocates no trajectory and applies one actuator update at the start of
    each controller interval before advancing the rack with ZOH torque.
    """
    remaining = float(duration)
    step_limit = float(max_step)
    request = float(requested_torque)
    if not math.isfinite(remaining) or remaining < 0.0:
      raise ValueError("prediction duration must be finite and non-negative")
    if not math.isfinite(step_limit) or step_limit <= 0.0:
      raise ValueError("prediction step must be finite and positive")
    if not math.isfinite(request):
      raise ValueError("requested torque must be finite")

    angle = float(state.angle_deg)
    rate = float(state.rate_deg_s)
    applied = float(state.applied_torque)
    static_load = self._initial_static_load(state)
    while remaining > 0.0:
      step = min(remaining, step_limit)
      applied = self.apply_slew(applied, request)
      angle, rate = self._advance(
        angle,
        rate,
        applied,
        state.v_ego,
        align_inputs,
        step,
        disturbance_torque,
        static_load,
      )
      static_load = self._advanced_static_load(static_load, rate)
      remaining -= step
    target.angle_deg = angle
    target.rate_deg_s = rate
    target.applied_torque = applied
    target.v_ego = float(state.v_ego)
    target.held_static_load = static_load

  def predict_constant_request_prepared_into(
    self,
    state: PlantState,
    duration: float,
    requested_torque: float,
    disturbance_torque: float,
    runtime_terms: AlignRuntimeTerms,
    target: PlantState,
    max_step: float = DT_CTRL,
  ) -> None:
    """Prepared form of :meth:`predict_constant_request_into`.

    Limiter, integration, stiction, and zero-rate semantics are unchanged.
    Only invariant validation and vehicle-model coefficient construction move
    outside the terminal inverse's repeated request evaluations.
    """
    remaining = float(duration)
    step_limit = float(max_step)
    request = float(requested_torque)
    if request < -1.0:
      request = -1.0
    elif request > 1.0:
      request = 1.0
    disturbance = float(disturbance_torque)
    if not math.isfinite(remaining) or remaining < 0.0:
      raise ValueError(
        "prediction duration must be finite and non-negative"
      )
    if not math.isfinite(step_limit) or step_limit <= 0.0:
      raise ValueError("prediction step must be finite and positive")
    if not math.isfinite(request) or not math.isfinite(disturbance):
      raise ValueError(
        "requested and disturbance torque must be finite"
      )

    angle = float(state.angle_deg)
    rate = float(state.rate_deg_s)
    applied = float(state.applied_torque)
    if applied < -1.0:
      applied = -1.0
    elif applied > 1.0:
      applied = 1.0
    build = self.params.delta_up / self.params.steer_max
    decay = self.params.delta_down / self.params.steer_max
    static_load = self._initial_static_load(state)
    torque_gain = self.params.k_t
    damping = self.params.b_steer
    align_slope = runtime_terms.torque_per_angle
    align_offset = runtime_terms.aligning_torque_offset
    while remaining > 0.0:
      step = step_limit if remaining > step_limit else remaining
      if request != applied:
        if applied * request >= 0.0:
          budget = (
            build if abs(request) > abs(applied) else decay
          )
          difference = request - applied
          if abs(difference) <= budget:
            applied = request
          else:
            applied += math.copysign(budget, difference)
        else:
          decay_fraction = abs(applied) / decay
          if decay_fraction >= 1.0:
            applied = math.copysign(
              abs(applied) - decay, applied,
            )
          else:
            build_limit = build * (1.0 - decay_fraction)
            if abs(request) <= build_limit:
              applied = request
            else:
              applied = math.copysign(build_limit, request)

      aligning_torque = align_slope * angle + align_offset
      net_torque = applied - aligning_torque - disturbance
      effective_torque = self._friction_effective_torque(
        net_torque, rate, static_load,
      )
      acceleration = torque_gain * effective_torque - damping * rate
      next_rate = rate + acceleration * step
      if rate != 0.0 and next_rate * rate < 0.0:
        next_rate = 0.0
      rate = next_rate
      static_load = self._advanced_static_load(static_load, rate)
      angle += rate * step
      remaining -= step
    target.angle_deg = angle
    target.rate_deg_s = rate
    target.applied_torque = applied
    target.v_ego = float(state.v_ego)
    target.held_static_load = static_load

  def predict_constant_request_sensitivity_into(
    self,
    state: PlantState,
    duration: float,
    requested_torque: float,
    disturbance_torque: float,
    runtime_terms: AlignRuntimeTerms,
    target: PlantState,
    sensitivity: PlantSensitivity,
    max_step: float = DT_CTRL,
  ) -> None:
    """Prepared rollout plus its exact local request sensitivity.

    The rack and limiter are piecewise affine. This propagates the derivative
    through the same selected slew, stiction, and zero-crossing branches as
    the ordinary prediction. A safeguarded terminal inverse can therefore
    jump directly to a branch's solution instead of rebuilding the full plant
    twelve times by blind bisection.
    """
    remaining = float(duration)
    step_limit = float(max_step)
    request = float(requested_torque)
    if request < -1.0:
      request = -1.0
    elif request > 1.0:
      request = 1.0
    disturbance = float(disturbance_torque)
    if not math.isfinite(remaining) or remaining < 0.0:
      raise ValueError(
        "prediction duration must be finite and non-negative"
      )
    if not math.isfinite(step_limit) or step_limit <= 0.0:
      raise ValueError("prediction step must be finite and positive")
    if not math.isfinite(request) or not math.isfinite(disturbance):
      raise ValueError(
        "requested and disturbance torque must be finite"
      )

    angle = float(state.angle_deg)
    rate = float(state.rate_deg_s)
    applied = float(state.applied_torque)
    if applied < -1.0:
      applied = -1.0
    elif applied > 1.0:
      applied = 1.0
    angle_sensitivity = 0.0
    rate_sensitivity = 0.0
    applied_sensitivity = 0.0
    build = self.params.delta_up / self.params.steer_max
    decay = self.params.delta_down / self.params.steer_max
    build_over_decay = build / decay
    static_load = self._initial_static_load(state)
    torque_gain = self.params.k_t
    damping = self.params.b_steer
    align_slope = runtime_terms.torque_per_angle
    align_offset = runtime_terms.aligning_torque_offset
    while remaining > 0.0:
      step = step_limit if remaining > step_limit else remaining

      if request == applied:
        applied = request
        applied_sensitivity = 1.0
      elif applied * request >= 0.0:
        budget = build if abs(request) > abs(applied) else decay
        if abs(request - applied) <= budget:
          applied = request
          applied_sensitivity = 1.0
        else:
          applied += math.copysign(budget, request - applied)
      else:
        decay_fraction = abs(applied) / decay
        if decay_fraction >= 1.0:
          applied = math.copysign(
            abs(applied) - decay, applied,
          )
        else:
          remaining_fraction = 1.0 - decay_fraction
          build_limit = build * remaining_fraction
          if abs(request) <= build_limit:
            applied = request
            applied_sensitivity = 1.0
          else:
            applied = math.copysign(build_limit, request)
            applied_sensitivity *= build_over_decay

      aligning_torque = align_slope * angle + align_offset
      net_torque = applied - aligning_torque - disturbance
      net_sensitivity = (
        applied_sensitivity
        - align_slope * angle_sensitivity
      )
      if rate == 0.0 and abs(net_torque) <= static_load:
        effective_torque = 0.0
        effective_sensitivity = 0.0
      else:
        friction_direction = net_torque if rate == 0.0 else rate
        slip_fraction = min(
          abs(rate) / RACK_RATE_QUANTUM_DEG_S, 1.0,
        )
        friction = static_load + slip_fraction * (
          self.kinetic_friction - static_load
        )
        effective_torque = net_torque - math.copysign(
          friction, friction_direction,
        )
        if abs(rate) < RACK_RATE_QUANTUM_DEG_S and rate != 0.0:
          friction_rate_slope = (
            self.kinetic_friction - static_load
          ) / RACK_RATE_QUANTUM_DEG_S
          effective_sensitivity = (
            net_sensitivity
            - friction_rate_slope * rate_sensitivity
          )
        else:
          effective_sensitivity = net_sensitivity

      acceleration = (
        torque_gain * effective_torque
        - damping * rate
      )
      acceleration_sensitivity = (
        torque_gain * effective_sensitivity
        - damping * rate_sensitivity
      )
      next_rate = rate + acceleration * step
      next_rate_sensitivity = (
        rate_sensitivity + acceleration_sensitivity * step
      )
      if rate != 0.0 and next_rate * rate < 0.0:
        next_rate = 0.0
        next_rate_sensitivity = 0.0
      rate = next_rate
      rate_sensitivity = next_rate_sensitivity
      static_load = self._advanced_static_load(static_load, rate)
      angle += rate * step
      angle_sensitivity += rate_sensitivity * step
      remaining -= step

    target.angle_deg = angle
    target.rate_deg_s = rate
    target.applied_torque = applied
    target.v_ego = float(state.v_ego)
    target.held_static_load = static_load
    sensitivity.angle_per_torque = angle_sensitivity
    sensitivity.rate_per_torque = rate_sensitivity

  def predict(
    self,
    state: PlantState,
    torque_sequence: Sequence[float],
    dt: float,
    align_inputs: AlignInputs | None = None,
  ) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not math.isfinite(dt) or dt <= 0.0:
      raise ValueError("dt must be finite and positive")

    requested = tuple(float(value) for value in torque_sequence)
    if not all(math.isfinite(value) for value in requested):
      raise ValueError("torque_sequence must be finite")

    applied_sequence: list[float] = []
    applied = self._clip(state.applied_torque, -1.0, 1.0)
    for demand in requested:
      applied = self.apply_slew(applied, demand)
      applied_sequence.append(applied)

    angle = float(state.angle_deg)
    rate = float(state.rate_deg_s)
    static_load = self._initial_static_load(state)
    inputs = self.nominal_align_inputs if align_inputs is None else align_inputs
    angles: list[float] = []
    rates: list[float] = []
    for index in range(len(requested)):
      delayed_time = index * dt - self.params.actuation_delay
      if delayed_time < 0.0:
        delayed_torque = state.applied_torque
      else:
        delayed_index = min(int(delayed_time / dt), len(applied_sequence) - 1)
        delayed_torque = applied_sequence[delayed_index]
      angle, rate = self._advance(
        angle,
        rate,
        delayed_torque,
        state.v_ego,
        inputs,
        dt,
        static_load=static_load,
      )
      static_load = self._advanced_static_load(static_load, rate)
      angles.append(angle)
      rates.append(rate)

    return tuple(angles), tuple(rates)

  def predict_into(
    self,
    state: PlantState,
    torque_sequence: Sequence[float],
    count: int,
    dt: float,
    align_inputs: AlignInputs,
    applied_out: Any,
    angle_out: Any,
    rate_out: Any,
    disturbance_torque: float = 0.0,
    actuation_delay: float | None = None,
  ) -> None:
    """Allocation-free exact limiter/delay/plant rollout into caller buffers."""
    if not math.isfinite(dt) or dt <= 0.0:
      raise ValueError("dt must be finite and positive")
    if count <= 0 or count > len(torque_sequence) or count > len(applied_out) or count > len(angle_out) or count > len(rate_out):
      raise ValueError("prediction count is outside buffer bounds")

    delay = self.params.actuation_delay if actuation_delay is None else float(actuation_delay)
    if not math.isfinite(delay) or delay < 0.0:
      raise ValueError("actuation delay must be finite and non-negative")

    applied = self._clip(state.applied_torque, -1.0, 1.0)
    for index in range(count):
      demand = float(torque_sequence[index])
      if not math.isfinite(demand):
        raise ValueError("torque_sequence must be finite")
      applied = self.apply_slew(applied, demand)
      applied_out[index] = applied

    angle = float(state.angle_deg)
    rate = float(state.rate_deg_s)
    static_load = self._initial_static_load(state)
    for index in range(count):
      delayed_time = index * dt - delay
      if delayed_time < 0.0:
        delayed_torque = state.applied_torque
      else:
        delayed_index = min(int(delayed_time / dt), count - 1)
        delayed_torque = float(applied_out[delayed_index])
      next_rate = self._next_rate(
        angle,
        rate,
        delayed_torque,
        state.v_ego,
        align_inputs,
        dt,
        disturbance_torque,
        static_load,
      )
      angle += next_rate * dt
      rate = next_rate
      static_load = self._advanced_static_load(static_load, rate)
      angle_out[index] = angle
      rate_out[index] = rate

  def one_step_residual(
    self,
    state_t: PlantState,
    applied_torque_t: float,
    state_t1: PlantState,
    align_inputs: AlignInputs | None = None,
  ) -> float:
    inputs = self.nominal_align_inputs if align_inputs is None else align_inputs
    predicted_rate = self._next_rate(
      state_t.angle_deg,
      state_t.rate_deg_s,
      float(applied_torque_t),
      state_t.v_ego,
      inputs,
      self.residual_dt,
      static_load=self._initial_static_load(state_t),
    )
    return float(state_t1.rate_deg_s - predicted_rate)
