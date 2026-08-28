"""Exact runtime torque envelope shared with opendbc.

The controller contains no vehicle torque/rate literals. Limits are copied
once from the detected vehicle's ``CarControllerParams`` and this module calls
opendbc's production limiter arithmetic directly. Opendbc/panda enforce the
same limits again at the vehicle boundary; an identical request and previous
count therefore pass unchanged rather than encountering a second shaping law.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from opendbc.car.lateral import apply_driver_steer_torque_limits


@dataclass(frozen=True, slots=True)
class RuntimeTorqueLimits:
  steer_max: int
  delta_up: int
  delta_down: int
  steer_step: int
  driver_allowance: int
  driver_multiplier: int
  driver_factor: int
  production_envelope_verified: bool = False

  def __post_init__(self) -> None:
    if (
      self.steer_max <= 0
      or self.delta_up <= 0
      or self.delta_down <= 0
      or self.steer_step <= 0
      or self.driver_allowance < 0
      or self.driver_multiplier < 0
      or self.driver_factor < 0
    ):
      raise ValueError("runtime torque limits are outside their valid domain")

  @classmethod
  def from_controller_params(cls, params: Any) -> RuntimeTorqueLimits:
    return cls(
      steer_max=int(params.STEER_MAX),
      delta_up=int(params.STEER_DELTA_UP),
      delta_down=int(params.STEER_DELTA_DOWN),
      steer_step=int(params.STEER_STEP),
      driver_allowance=int(params.STEER_DRIVER_ALLOWANCE),
      driver_multiplier=int(params.STEER_DRIVER_MULTIPLIER),
      driver_factor=int(params.STEER_DRIVER_FACTOR),
      production_envelope_verified=(
        getattr(
          params,
          "BLATV2_RUNTIME_ENVELOPE_COMPATIBLE",
          False,
        )
        is True
      ),
    )

  # opendbc's shared limiter deliberately consumes the platform-style names.
  @property
  def STEER_MAX(self) -> int:
    return self.steer_max

  @property
  def STEER_DELTA_UP(self) -> int:
    return self.delta_up

  @property
  def STEER_DELTA_DOWN(self) -> int:
    return self.delta_down

  @property
  def STEER_DRIVER_ALLOWANCE(self) -> int:
    return self.driver_allowance

  @property
  def STEER_DRIVER_MULTIPLIER(self) -> int:
    return self.driver_multiplier

  @property
  def STEER_DRIVER_FACTOR(self) -> int:
    return self.driver_factor

  def driver_exceeds_allowance(self, driver_torque: float) -> bool:
    """Whether measured human torque makes rack input ambiguous."""
    torque = float(driver_torque)
    if not math.isfinite(torque):
      return True
    return max(
      abs(torque),
      abs(torque * self.driver_factor),
    ) > self.driver_allowance


@dataclass(frozen=True, slots=True)
class EnvelopeResult:
  requested_torque: float
  applied_torque: float
  requested_counts: int
  applied_counts: int
  constrained: bool


def apply_torque_envelope_counts(
  limits: RuntimeTorqueLimits,
  requested_counts: int,
  previous_applied_counts: int,
  driver_torque: float,
) -> int:
  """Apply the production opendbc magnitude/driver/rate rules in count space."""
  driver = float(driver_torque)
  if not math.isfinite(driver):
    raise ValueError("driver torque must be finite")
  return apply_driver_steer_torque_limits(
    int(requested_counts),
    int(previous_applied_counts),
    driver,
    limits,
  )


def apply_torque_envelope(
  limits: RuntimeTorqueLimits,
  requested_torque: float,
  previous_applied_torque: float,
  driver_torque: float,
) -> EnvelopeResult:
  """Quantize and constrain one normalized request exactly as opendbc does."""
  requested = float(requested_torque)
  previous = float(previous_applied_torque)
  if not math.isfinite(requested) or not math.isfinite(previous):
    raise ValueError("normalized torque inputs must be finite")

  requested_counts = int(round(requested * limits.steer_max))
  previous_counts = int(round(previous * limits.steer_max))
  applied_counts = apply_torque_envelope_counts(
    limits,
    requested_counts,
    previous_counts,
    driver_torque,
  )
  applied = applied_counts / limits.steer_max
  return EnvelopeResult(
    requested_torque=requested,
    applied_torque=applied,
    requested_counts=requested_counts,
    applied_counts=applied_counts,
    constrained=applied_counts != requested_counts,
  )
