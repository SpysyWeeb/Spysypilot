"""Deterministic BLaTv2 steering-rack plant twin.

Numerical contract
------------------
* All arithmetic is Python ``float`` (IEEE-754 binary64); no numpy, BLAS, or
  platform-selected math kernels are used.
* ``PlantState`` is steering-wheel angle in degrees, steering-wheel rate in
  degrees/second, and normalized torque on [-1, 1].
* ``k_t`` is deg/s² per normalized effective torque, ``b_steer`` is 1/s, and
  ``t_breakaway`` is normalized torque.
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


@dataclass(frozen=True)
class PlantState:
  angle_deg: float
  rate_deg_s: float
  applied_torque: float

  def __post_init__(self) -> None:
    if not all(math.isfinite(value) for value in (self.angle_deg, self.rate_deg_s, self.applied_torque)):
      raise ValueError("plant state must be finite")


class PlantTwin:
  def __init__(self, params: PlantParams, residual_dt: float = DT_CTRL):
    if not math.isfinite(residual_dt) or residual_dt <= 0.0:
      raise ValueError("residual_dt must be finite and positive")
    self.params = params
    self.residual_dt = float(residual_dt)

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

  def _advance(self, angle: float, rate: float, applied_torque: float, dt: float) -> tuple[float, float]:
    torque = self._clip(applied_torque, -1.0, 1.0)
    if rate == 0.0:
      if abs(torque) <= self.params.t_breakaway:
        effective_torque = 0.0
      else:
        effective_torque = torque - math.copysign(self.params.t_breakaway, torque)
    else:
      effective_torque = torque - math.copysign(self.params.t_breakaway, rate)

    acceleration = self.params.k_t * effective_torque - self.params.b_steer * rate
    next_rate = rate + acceleration * dt
    if rate != 0.0 and next_rate * rate < 0.0:
      next_rate = 0.0
    next_angle = angle + next_rate * dt
    return next_angle, next_rate

  def predict(self, state: PlantState, torque_sequence: Sequence[float], dt: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
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
    angles: list[float] = []
    rates: list[float] = []
    for index in range(len(requested)):
      delayed_time = index * dt - self.params.actuation_delay
      if delayed_time < 0.0:
        delayed_torque = state.applied_torque
      else:
        delayed_index = min(int(delayed_time / dt), len(applied_sequence) - 1)
        delayed_torque = applied_sequence[delayed_index]
      angle, rate = self._advance(angle, rate, delayed_torque, dt)
      angles.append(angle)
      rates.append(rate)

    return tuple(angles), tuple(rates)

  def one_step_residual(self, state_t: PlantState, applied_torque_t: float, state_t1: PlantState) -> float:
    _, predicted_rate = self._advance(
      state_t.angle_deg,
      state_t.rate_deg_s,
      float(applied_torque_t),
      self.residual_dt,
    )
    return float(state_t1.rate_deg_s - predicted_rate)
