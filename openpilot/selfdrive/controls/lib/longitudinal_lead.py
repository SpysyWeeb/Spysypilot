"""Shared, side-effect-free lead physics for BLoTv2.

The planner and the 100 Hz final-stop controller run in different processes,
so they cannot share state. They can, however, share one definition of a
usable lead and one relative-motion calculation. Keeping that math here
prevents the two policy layers from disagreeing about the same vehicle.
"""
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class LeadObservation:
  """Finite, filtered lead values captured from one valid radarState sample."""

  present: bool = False
  distance: float = math.inf
  speed: float = 0.0
  acceleration: float = 0.0
  model_prob: float = 0.0

  @classmethod
  def from_radar(cls, lead: Any, service_valid: bool) -> "LeadObservation":
    if not service_valid or lead is None or not bool(getattr(lead, "present", False)):
      return cls()

    try:
      distance = float(lead.dRel)
      speed = float(lead.vLeadK)
      acceleration = float(lead.aLeadK)
      model_prob = float(lead.modelProb)
    except (AttributeError, TypeError, ValueError):
      return cls()

    if not all(math.isfinite(value) for value in (distance, speed, acceleration, model_prob)) or distance <= 0.0:
      return cls()

    return cls(
      present=True,
      distance=distance,
      speed=max(speed, 0.0),
      acceleration=min(max(acceleration, -10.0), 5.0),
      model_prob=min(max(model_prob, 0.0), 1.0),
    )


def closing_speed(v_ego: float, lead: LeadObservation) -> float:
  """Positive only when ego is gaining on a usable lead."""
  return max(float(v_ego) - lead.speed, 0.0) if lead.present else 0.0


def closing_decel_requirement(v_ego: float, lead: LeadObservation, stop_distance: float,
                              min_gap_budget: float) -> float:
  """Constant deceleration needed to shed relative closing speed before a margin.

  This is deliberately relative-frame physics. Equal-speed queue motion therefore
  requires no extra final-stop pressure, while a stopped lead produces the familiar
  ``v² / 2d`` result.
  """
  if not lead.present:
    return 0.0

  gap_budget = max(lead.distance - stop_distance, min_gap_budget)
  closing = closing_speed(v_ego, lead)
  return closing * closing / (2.0 * gap_budget)


def total_decel_requirement(v_ego: float, lead: LeadObservation, stop_distance: float,
                            min_gap_budget: float) -> float:
  """Match measured lead braking and shed closing energy before the margin."""
  if not lead.present:
    return 0.0
  return max(-lead.acceleration, 0.0) + closing_decel_requirement(
    v_ego, lead, stop_distance, min_gap_budget,
  )


def time_to_collision(v_ego: float, lead: LeadObservation, min_closing_speed: float = 0.3) -> float:
  if not lead.present:
    return math.inf
  closing = closing_speed(v_ego, lead)
  return lead.distance / closing if closing > min_closing_speed else math.inf
