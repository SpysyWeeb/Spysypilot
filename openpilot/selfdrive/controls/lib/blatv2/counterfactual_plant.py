"""Causal plant-member execution for offline controller comparisons.

The live controller owns a nominal rack model.  Counterfactual evaluation must
not prove that controller against the same model: this module applies each
controller's CAN torque through one independently identified plant member.

The actuator and rack clocks are intentionally separate.  ``commit_and_sample``
records the torque which the production envelope placed on CAN this frame,
then returns the zero-order-held torque whose transport delay has elapsed at
the start of this plant interval.  A zero delay therefore uses the new torque;
any positive sub-frame delay uses the previous torque.  Episode bootstrap
fills the unknown pre-engagement history with the exact recorded CAN-applied
anchor.  Changing speed may change the selected delay, but never rewrites
history.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re

from openpilot.selfdrive.controls.lib.blatv2.plant import (
  PlantStep,
  RackState,
  step_plant,
)
from openpilot.selfdrive.controls.lib.blatv2.rack_mapper import (
  RackMappingSnapshot,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  PhysicalParameters,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class CounterfactualPlantMember:
  """One independently identified transient rack model.

  Steady aligning load, friction, and their speed schedules remain owned by
  the authenticated physical profile.  A member varies only the transient
  quantities the route-separated identified set could not collapse to one
  value.
  """

  member_id: str
  rack_gain_deg_s2_per_torque: float
  rack_damping_per_s: float
  delay_offset_s: float
  unresolved_load_torque: float

  @staticmethod
  def identity_for(
    *,
    rack_gain_deg_s2_per_torque: float,
    rack_damping_per_s: float,
    delay_offset_s: float,
    unresolved_load_torque: float,
  ) -> str:
    payload = {
      "delayOffsetS": float(delay_offset_s).hex(),
      "domain": "blatv2-global-transient-member-v2",
      "loadUncertaintyTorque": float(unresolved_load_torque).hex(),
      "rackDampingPerS": float(rack_damping_per_s).hex(),
      "rackGainDegS2PerTorque": float(rack_gain_deg_s2_per_torque).hex(),
    }
    return hashlib.sha256(
      json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode(),
    ).hexdigest()

  @classmethod
  def create(
    cls,
    *,
    rack_gain_deg_s2_per_torque: float,
    rack_damping_per_s: float,
    delay_offset_s: float,
    unresolved_load_torque: float,
  ) -> CounterfactualPlantMember:
    return cls(
      member_id=cls.identity_for(
        rack_gain_deg_s2_per_torque=rack_gain_deg_s2_per_torque,
        rack_damping_per_s=rack_damping_per_s,
        delay_offset_s=delay_offset_s,
        unresolved_load_torque=unresolved_load_torque,
      ),
      rack_gain_deg_s2_per_torque=rack_gain_deg_s2_per_torque,
      rack_damping_per_s=rack_damping_per_s,
      delay_offset_s=delay_offset_s,
      unresolved_load_torque=unresolved_load_torque,
    )

  def __post_init__(self) -> None:
    values = (
      self.rack_gain_deg_s2_per_torque,
      self.rack_damping_per_s,
      self.delay_offset_s,
      self.unresolved_load_torque,
    )
    if (
      type(self.member_id) is not str
      or _SHA256_RE.fullmatch(self.member_id) is None
      or not all(math.isfinite(value) for value in values)
      or self.rack_gain_deg_s2_per_torque <= 0.0
      or self.rack_damping_per_s <= 0.0
      or self.member_id != self.identity_for(
        rack_gain_deg_s2_per_torque=self.rack_gain_deg_s2_per_torque,
        rack_damping_per_s=self.rack_damping_per_s,
        delay_offset_s=self.delay_offset_s,
        unresolved_load_torque=self.unresolved_load_torque,
      )
    ):
      raise ValueError("counterfactual plant member is outside its domain")

  def parameters_for(self, base: PhysicalParameters) -> PhysicalParameters:
    """Substitute transient dynamics without changing steady calibration."""
    if not isinstance(base, PhysicalParameters):
      raise TypeError("plant member requires physical base parameters")
    return replace(
      base,
      rack_gain_deg_s2_per_torque=self.rack_gain_deg_s2_per_torque,
      rack_damping_per_s=self.rack_damping_per_s,
    )

  def effective_delay_s(self, base_transport_delay_s: float) -> float:
    delay = float(base_transport_delay_s) + self.delay_offset_s
    if not math.isfinite(delay) or delay < 0.0:
      raise ValueError("counterfactual plant delay is outside its domain")
    return delay


class AppliedTorqueDelayLine:
  """Fixed-capacity ZOH history between CAN-applied and rack-effective torque."""

  def __init__(self, *, fixed_dt_s: float, maximum_delay_s: float) -> None:
    dt = float(fixed_dt_s)
    maximum_delay = float(maximum_delay_s)
    if (
      not math.isfinite(dt)
      or dt <= 0.0
      or not math.isfinite(maximum_delay)
      or maximum_delay < 0.0
    ):
      raise ValueError("delay-line timing is outside its domain")
    self.fixed_dt_s = dt
    self.maximum_delay_s = maximum_delay
    self.capacity = self._delay_frame_count(maximum_delay) + 1
    self._history = [0.0] * self.capacity
    self._write_index = 0
    self._initialized = False
    self._latest_can_applied_torque = 0.0
    self._latest_rack_effective_torque = 0.0

  def _delay_frame_count(self, delay_s: float) -> int:
    delay = float(delay_s)
    if not math.isfinite(delay) or delay < 0.0:
      raise ValueError("transport delay is outside its domain")
    tolerance = 8.0 * math.ulp(max(delay, self.fixed_dt_s))
    if delay > self.maximum_delay_s + tolerance:
      raise ValueError("transport delay exceeds the fixed history")
    delay = min(delay, self.maximum_delay_s)
    full_frames = int(math.floor(delay / self.fixed_dt_s))
    remainder = delay - full_frames * self.fixed_dt_s
    if remainder <= tolerance:
      return full_frames
    if self.fixed_dt_s - remainder <= tolerance:
      return full_frames + 1
    return full_frames + 1

  @property
  def latest_can_applied_torque(self) -> float:
    if not self._initialized:
      raise RuntimeError("delay line has not been primed")
    return self._latest_can_applied_torque

  @property
  def latest_rack_effective_torque(self) -> float:
    if not self._initialized:
      raise RuntimeError("delay line has not been primed")
    return self._latest_rack_effective_torque

  def reset(self, can_applied_torque: float) -> None:
    """Prime unknown pre-engagement history with one measured CAN anchor."""
    applied = float(can_applied_torque)
    if not math.isfinite(applied):
      raise ValueError("delay-line prime must be finite")
    for index in range(self.capacity):
      self._history[index] = applied
    self._write_index = 0
    self._latest_can_applied_torque = applied
    self._latest_rack_effective_torque = applied
    self._initialized = True

  def commit_and_sample(
    self,
    can_applied_torque: float,
    effective_delay_s: float,
  ) -> float:
    """Commit this frame's envelope output and sample rack-effective torque."""
    if not self._initialized:
      raise RuntimeError("delay line has not been primed")
    applied = float(can_applied_torque)
    if not math.isfinite(applied):
      raise ValueError("CAN-applied torque must be finite")
    delayed_frames = self._delay_frame_count(effective_delay_s)
    current_index = self._write_index
    self._history[current_index] = applied
    self._write_index = (current_index + 1) % self.capacity
    effective = self._history[(current_index - delayed_frames) % self.capacity]
    self._latest_can_applied_torque = applied
    self._latest_rack_effective_torque = effective
    return effective


def step_counterfactual_plant(
  *,
  state: RackState,
  rack_effective_torque: float,
  speed_mps: float,
  mapping: RackMappingSnapshot,
  nominal_mapping: RackMappingSnapshot,
  lateral_accel_offset: float,
  base_parameters: PhysicalParameters,
  member: CounterfactualPlantMember,
  dt: float,
) -> PlantStep:
  """Advance one independent member without modifying controller parameters."""
  if not isinstance(member, CounterfactualPlantMember):
    raise TypeError("counterfactual plant requires an identified member")
  return step_plant(
    state,
    rack_effective_torque,
    speed_mps,
    mapping,
    nominal_mapping,
    lateral_accel_offset,
    member.parameters_for(base_parameters),
    member.unresolved_load_torque,
    dt,
  )
