"""Deterministic BLaTv2 steering-rack plant twin.

Numerical contract
------------------
* Plant arithmetic is Python ``float`` (IEEE-754 binary64); no BLAS or
  platform-selected math kernels are used.
* ``PlantState`` is steering-wheel angle in degrees, steering-wheel rate in
  degrees/second, normalized torque on [-1, 1], and vehicle speed in m/s.
* ``k_t`` is deg/s² per normalized effective torque, ``b_steer`` is 1/s, and
  ``t_breakaway`` is normalized torque.
* Tire self-aligning load mirrors frozen-v14's measurement convention exactly:
  offset-corrected steering angle enters the vehicle model with live roll,
  stiffness, and steer ratio; measured curvature becomes lateral acceleration;
  roll gravity and ``latAccelOffset`` are removed; and ``latAccelFactor`` maps
  that acceleration to normalized torque. Calibration friction is deliberately
  excluded because ``t_breakaway`` owns Coulomb friction.
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
* Seed delay is a deterministic fallback. A valid ``liveDelay`` value replaces
  it by constructing a per-frame ``PlantParams`` copy.

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

  @classmethod
  def from_seed_file(cls, path: str | Path, controller_params: Any) -> PlantParams:
    with Path(path).open(encoding="utf-8") as stream:
      seed = json.load(stream)

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


@dataclass(frozen=True, slots=True)
class AlignParams:
  mass: float
  wheelbase: float
  center_to_front: float
  tire_stiffness_front: float
  tire_stiffness_rear: float
  nominal_steer_ratio: float
  steer_ratio_rear: float
  lat_accel_factor: float
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
      lat_accel_factor=float(torque_params.latAccelFactor),
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
      self.lat_accel_factor,
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
      or self.lat_accel_factor <= 0.0
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
    # Controller output is the negative of torque_from_lateral_accel.
    return -(gravity_adjusted / p.lat_accel_factor)

  def _next_rate(
    self,
    angle: float,
    rate: float,
    applied_torque: float,
    v_ego: float,
    align_inputs: AlignInputs,
    dt: float,
  ) -> float:
    torque = self._clip(applied_torque, -1.0, 1.0)
    aligning_torque = self._aligning_torque(angle, v_ego, align_inputs)
    net_torque = torque - aligning_torque
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
  ) -> tuple[float, float]:
    next_rate = self._next_rate(angle, rate, applied_torque, v_ego, align_inputs, dt)
    next_angle = angle + next_rate * dt
    return next_angle, next_rate

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
