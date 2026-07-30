"""Immutable contracts shared by BLaTv2 live control and replay.

Timing fields deliberately describe different clocks instead of collapsing
them into one ambiguous delay:

* ``state_sample_mono_ns`` is the monotonic time of the vehicle-state sample.
* ``control_witness_mono_ns`` is captured immediately before computation.
* ``plan_origin_mono_ns`` is the model trajectory origin (`timestampEof`).
* ``plan_publication_mono_ns`` is the later message publication time and is
  used only to measure freshness.
* ``scalar_action_plan_s`` is the action timestamp published with the scalar,
  on the model plan's native relative-time grid.
* ``scalar_action_effect_time_s`` is therefore an authored model time.
* ``plant_effect_time_s`` is when a command issued at the control witness is
  expected to affect the rack.
* ``total_prediction_horizon_s`` spans the age of the sampled rack state plus
  physical command transport delay.

Live code must consume the model-published ``scalar_action_plan_s``. It must
not reconstruct that timestamp from ``DT_MDL``, live delay, or model smoothing
constants. This keeps model intent and actuator response as separate facts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral


def _require_finite_nonnegative(name: str, value: float) -> None:
  if not math.isfinite(value) or value < 0.0:
    raise ValueError(f"{name} must be finite and non-negative")


def _require_mono_ns(name: str, value: int) -> None:
  if (
    isinstance(value, bool)
    or not isinstance(value, Integral)
    or value < 0
    or value > (1 << 64) - 1
  ):
    raise ValueError(f"{name} must be a UInt64 monotonic timestamp")


@dataclass(frozen=True, slots=True)
class FrameTiming:
  """Canonical timestamps for one controller frame."""

  state_sample_mono_ns: int
  control_witness_mono_ns: int
  plan_origin_mono_ns: int
  plan_publication_mono_ns: int
  scalar_action_plan_s: float
  transport_delay_s: float

  def __post_init__(self) -> None:
    _require_mono_ns("state_sample_mono_ns", self.state_sample_mono_ns)
    _require_mono_ns(
      "control_witness_mono_ns",
      self.control_witness_mono_ns,
    )
    _require_mono_ns("plan_origin_mono_ns", self.plan_origin_mono_ns)
    _require_mono_ns(
      "plan_publication_mono_ns",
      self.plan_publication_mono_ns,
    )
    _require_finite_nonnegative(
      "scalar_action_plan_s",
      self.scalar_action_plan_s,
    )
    _require_finite_nonnegative(
      "transport_delay_s",
      self.transport_delay_s,
    )
    if self.plan_origin_mono_ns > self.plan_publication_mono_ns:
      raise ValueError(
        "plan origin cannot follow its publication time",
      )
    if self.plan_publication_mono_ns > self.control_witness_mono_ns:
      raise ValueError(
        "plan publication cannot follow the control witness",
      )
    if self.state_sample_mono_ns > self.control_witness_mono_ns:
      raise ValueError(
        "vehicle-state sample cannot follow the control witness",
      )

  @property
  def plan_age_s(self) -> float:
    return (
      self.control_witness_mono_ns - self.plan_publication_mono_ns
    ) * 1e-9

  @property
  def state_age_s(self) -> float:
    return (
      self.control_witness_mono_ns - self.state_sample_mono_ns
    ) * 1e-9

  @property
  def plan_time_now_s(self) -> float:
    return (
      self.control_witness_mono_ns - self.plan_origin_mono_ns
    ) * 1e-9

  @property
  def scalar_action_effect_time_s(self) -> float:
    return self.plan_origin_mono_ns * 1e-9 + self.scalar_action_plan_s

  @property
  def plant_effect_time_s(self) -> float:
    return (
      self.control_witness_mono_ns * 1e-9 + self.transport_delay_s
    )

  @property
  def physical_effect_plan_s(self) -> float:
    return self.plan_time_now_s + self.transport_delay_s

  @property
  def total_prediction_horizon_s(self) -> float:
    return self.state_age_s + self.transport_delay_s


@dataclass(frozen=True, slots=True)
class FrameValidity:
  """Validity facts; policy remains with the consuming controller."""

  model_valid: bool
  plan_valid: bool
  vehicle_state_valid: bool
  calibration_valid: bool

  @property
  def reference_valid(self) -> bool:
    return self.model_valid and self.plan_valid

  @property
  def all_valid(self) -> bool:
    return self.reference_valid and self.vehicle_state_valid and self.calibration_valid


@dataclass(frozen=True, slots=True)
class CanonicalFrame:
  """One immutable timing/validity snapshot shared by all modules."""

  timing: FrameTiming
  validity: FrameValidity


@dataclass(frozen=True, slots=True)
class ReferenceBuildStatus:
  """Populated prefix and validity returned by the caller-buffer API."""

  count: int
  valid: bool
  scalar_only: bool

  def __post_init__(self) -> None:
    if self.count <= 0:
      raise ValueError("reference count must be positive")
    if self.valid and self.scalar_only:
      raise ValueError("a valid plan reference cannot be scalar-only")
    if not self.valid and not self.scalar_only:
      raise ValueError("an invalid reference must be scalar-only")

  @property
  def degraded(self) -> bool:
    """Whether the model shape was rejected in favor of scalar-only output."""
    return self.scalar_only


@dataclass(frozen=True, slots=True)
class ReferenceQueryOutput:
  """Immutable result from explicit plan-relative reference queries.

  Unlike :class:`ReferenceOutput`, query results do not require the caller to
  include either anchor time. When an anchor time is queried exactly, the
  corresponding output is nevertheless required to equal its live anchor
  exactly: scalar curvature at ``scalar_action_plan_s`` and measured speed at
  ``plan_time_now_s``.
  """

  times_s: tuple[float, ...]
  curvatures: tuple[float, ...]
  curvature_rates: tuple[float, ...]
  curvature_accelerations: tuple[float, ...]
  planned_speeds: tuple[float, ...]
  planned_speed_rates: tuple[float, ...]
  planned_speed_accelerations: tuple[float, ...]
  scalar_curvature: float
  scalar_action_plan_s: float
  plan_time_now_s: float
  measured_v_ego: float
  valid: bool
  scalar_only: bool

  def __post_init__(self) -> None:
    count = len(self.times_s)
    if count <= 0:
      raise ValueError("reference query must contain at least one sample")
    if any(
      len(values) != count
      for values in (
        self.curvatures,
        self.curvature_rates,
        self.curvature_accelerations,
        self.planned_speeds,
        self.planned_speed_rates,
        self.planned_speed_accelerations,
      )
    ):
      raise ValueError("reference query fields must have equal lengths")
    scalars = (
      *self.times_s,
      *self.curvatures,
      *self.curvature_rates,
      *self.curvature_accelerations,
      *self.planned_speeds,
      *self.planned_speed_rates,
      *self.planned_speed_accelerations,
      self.scalar_curvature,
      self.scalar_action_plan_s,
      self.plan_time_now_s,
      self.measured_v_ego,
    )
    if not all(math.isfinite(value) for value in scalars):
      raise ValueError("reference query output must be finite")
    _require_finite_nonnegative(
      "scalar_action_plan_s",
      self.scalar_action_plan_s,
    )
    _require_finite_nonnegative("plan_time_now_s", self.plan_time_now_s)
    _require_finite_nonnegative("measured_v_ego", self.measured_v_ego)
    if any(right <= left for left, right in zip(self.times_s, self.times_s[1:], strict=False)):
      raise ValueError("reference query times must be strictly increasing")
    if self.valid == self.scalar_only:
      raise ValueError("valid and scalar_only must be logical opposites")

    for index, query_time_s in enumerate(self.times_s):
      if query_time_s == self.scalar_action_plan_s and self.curvatures[index] != self.scalar_curvature:
        raise ValueError("reference must equal scalar at the action time")
      if query_time_s == self.plan_time_now_s and self.planned_speeds[index] != self.measured_v_ego:
        raise ValueError("reference speed must equal vEgo at plan-now")
      if self.scalar_only and (
        self.curvatures[index] != self.scalar_curvature
        or self.curvature_rates[index] != 0.0
        or self.curvature_accelerations[index] != 0.0
        or self.planned_speeds[index] != self.measured_v_ego
        or self.planned_speed_rates[index] != 0.0
        or self.planned_speed_accelerations[index] != 0.0
      ):
        raise ValueError("degraded reference must be scalar-only")

  @property
  def degraded(self) -> bool:
    """Whether every query was replaced by the safe scalar-only reference."""
    return self.scalar_only


@dataclass(frozen=True, slots=True)
class ReferenceOutput:
  """Immutable convenience result from the stateless reference compiler."""

  times_s: tuple[float, ...]
  curvatures: tuple[float, ...]
  curvature_rates: tuple[float, ...]
  curvature_accelerations: tuple[float, ...]
  planned_speeds: tuple[float, ...]
  scalar_curvature: float
  scalar_action_plan_s: float
  valid: bool
  scalar_only: bool

  def __post_init__(self) -> None:
    count = len(self.times_s)
    if count <= 0:
      raise ValueError("reference output must contain at least one sample")
    if any(
      len(values) != count
      for values in (
        self.curvatures,
        self.curvature_rates,
        self.curvature_accelerations,
        self.planned_speeds,
      )
    ):
      raise ValueError("reference output fields must have equal lengths")
    scalars = (
      *self.times_s,
      *self.curvatures,
      *self.curvature_rates,
      *self.curvature_accelerations,
      *self.planned_speeds,
      self.scalar_curvature,
      self.scalar_action_plan_s,
    )
    if not all(math.isfinite(value) for value in scalars):
      raise ValueError("reference output must be finite")
    if self.scalar_action_plan_s < 0.0:
      raise ValueError("scalar action plan time must be non-negative")
    if any(right <= left for left, right in zip(self.times_s, self.times_s[1:], strict=False)):
      raise ValueError("reference output times must be strictly increasing")
    try:
      action_index = self.times_s.index(self.scalar_action_plan_s)
    except ValueError as exc:
      raise ValueError("reference output must contain the scalar action time") from exc
    if self.curvatures[action_index] != self.scalar_curvature:
      raise ValueError("reference must equal the scalar at its action time")
    if self.valid == self.scalar_only:
      raise ValueError("valid and scalar_only must be logical opposites")

  @property
  def degraded(self) -> bool:
    """Whether the model shape was rejected in favor of scalar-only output."""
    return self.scalar_only
