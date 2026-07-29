"""Deterministic BLaTv2 steering-rack plant twin.

Numerical contract
------------------
* Plant arithmetic is Python ``float`` (IEEE-754 binary64); no BLAS or
  platform-selected math kernels are used.
* ``PlantState`` is steering-wheel angle in degrees, steering-wheel rate in
  degrees/second, normalized torque on [-1, 1], and vehicle speed in m/s.
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
* At exactly zero rate, torque inside the breakaway envelope produces no
  motion. Above it, Coulomb torque is removed in the demand direction. While
  moving, Coulomb torque opposes motion; a numerical rate sign crossing is
  clamped to zero so friction cannot create a one-frame oscillation.
* ``predict`` applies the exact asymmetric limiter to requested torque and
  models the pure actuation delay with zero-order-held samples. Before the
  delayed sequence arrives, the state's measured applied torque is held.
* ``one_step_residual`` consumes already-applied ``carOutput`` torque and does
  not delay it a second time.
* The limiter is the float64 counterpart of frozen-BLaT's
  ``torque_transition_time``: same-sign growth uses ``delta_up``, same-sign
  release uses ``delta_down``, and a sign crossing spends decay budget reaching
  zero before using the remaining frame fraction to build the new sign.
* Seed delay is a deterministic fallback. A valid ``liveDelay`` value is passed
  explicitly to the allocation-free rollout; the frozen ``PlantParams`` and
  ``PlantTwin`` are never reconstructed on the hot path.

These rules are shared verbatim by the device shadow and route-audit replay.
Changing any rule is a behavior change and requires a shadow-version bump.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.realtime import DT_CTRL


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
    for index in range(1, len(nodes)):
      if speed <= nodes[index]:
        fraction = (speed - nodes[index - 1]) / (nodes[index] - nodes[index - 1])
        return values[index - 1] + fraction * (values[index] - values[index - 1])
    return values[-1]


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

  def __post_init__(self) -> None:
    if not all(math.isfinite(value) for value in (self.angle_deg, self.rate_deg_s, self.applied_torque, self.v_ego)):
      raise ValueError("plant state must be finite")


class PlantTwin:
  def __init__(self, params: PlantParams, align_params: AlignParams, residual_dt: float = DT_CTRL):
    if not math.isfinite(residual_dt) or residual_dt <= 0.0:
      raise ValueError("residual_dt must be finite and positive")
    self.params = params
    self.align_params = align_params
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
  ) -> float:
    torque = self._clip(applied_torque, -1.0, 1.0)
    aligning_torque = self._aligning_torque(angle, v_ego, align_inputs)
    disturbance = float(disturbance_torque)
    if not math.isfinite(disturbance):
      raise ValueError("disturbance torque must be finite")
    net_torque = torque - aligning_torque - disturbance
    if rate == 0.0:
      if abs(net_torque) <= self.params.t_breakaway:
        effective_torque = 0.0
      else:
        effective_torque = net_torque - math.copysign(self.params.t_breakaway, net_torque)
    else:
      effective_torque = net_torque - math.copysign(self.params.t_breakaway, rate)

    acceleration = self.params.k_t * effective_torque - self.params.b_steer * rate
    next_rate = rate + acceleration * dt
    if rate != 0.0 and next_rate * rate < 0.0:
      next_rate = 0.0
    return next_rate

  def _advance(
    self,
    angle: float,
    rate: float,
    applied_torque: float,
    v_ego: float,
    align_inputs: AlignInputs,
    dt: float,
    disturbance_torque: float = 0.0,
  ) -> tuple[float, float]:
    next_rate = self._next_rate(angle, rate, applied_torque, v_ego, align_inputs, dt, disturbance_torque)
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
    angle, rate = self._advance(
      float(state.angle_deg),
      float(state.rate_deg_s),
      float(applied_torque),
      float(state.v_ego),
      align_inputs,
      float(dt),
      float(disturbance_torque),
    )
    return PlantState(
      angle_deg=angle,
      rate_deg_s=rate,
      applied_torque=float(applied_torque),
      v_ego=float(state.v_ego),
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
      )
      remaining -= step
    target.angle_deg = angle
    target.rate_deg_s = rate
    target.applied_torque = float(state.applied_torque)
    target.v_ego = float(state.v_ego)

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
      )
      remaining -= step
    target.angle_deg = angle
    target.rate_deg_s = rate
    target.applied_torque = applied
    target.v_ego = float(state.v_ego)

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
      angle, rate = self._advance(angle, rate, delayed_torque, state.v_ego, inputs, dt)
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
      )
      angle += next_rate * dt
      rate = next_rate
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
    )
    return float(state_t1.rate_deg_s - predicted_rate)
