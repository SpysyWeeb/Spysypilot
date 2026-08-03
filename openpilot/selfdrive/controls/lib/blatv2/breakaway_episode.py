"""Deterministic vehicle-global breakaway episode reconstruction.

The Palisade's steering-rate signal resolves motion in 4 deg/s quanta while
its raw steering-angle signal commonly resolves motion earlier.  Treating
every zero-rate frame as physically stationary therefore places the
breakaway row after the rack has already started moving.  This module keeps
the two measurements distinct:

* unresolved rate plus sub-threshold adjacent angle motion is ``BASE``;
* angle-resolved motion after a continuous dwell is ``PENDING`` until a
  same-direction rate quantum confirms it;
* the confirming frame is ``BREAKAWAY`` and carries the complete immutable
  episode; and
* all other resolved motion is ``MOVING``.

The angle-motion threshold is exactly half one declared rate quantum
integrated over the measured interval.  Dwell and confirmation use the
vehicle's transport delay.  No steering-feel or fitted constants live here.
The state is vehicle-global rather than speed-node-local; speed weighting is
the learner's later responsibility and cannot duplicate one physical onset.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

from openpilot.selfdrive.controls.lib.blatv2.learner import (
  LearningSample,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_source import (
  CalibrationIngestionCoordinate,
)


class BreakawayCategory(StrEnum):
  BASE = "base"
  MOVING = "moving"
  PENDING = "pending"
  BREAKAWAY = "breakaway"
  DISCARDED = "discarded"


@dataclass(frozen=True, slots=True)
class BreakawayEpisodePoint:
  """Measured values retained at one physical episode boundary."""

  speed_mps: float
  dt_s: float
  applied_torque: float
  measured_lateral_accel_mps2: float
  measured_rack_angle_deg: float
  rack_rate_deg_s: float
  rack_acceleration_deg_s2: float
  actuator_constrained: bool
  source_coordinate: CalibrationIngestionCoordinate | None = None

  @classmethod
  def from_sample(
    cls,
    sample: LearningSample,
    source_coordinate: CalibrationIngestionCoordinate | None = None,
  ) -> BreakawayEpisodePoint:
    return cls(
      speed_mps=sample.speed_mps,
      dt_s=sample.dt_s,
      applied_torque=sample.applied_torque,
      measured_lateral_accel_mps2=(
        sample.measured_lateral_accel_mps2
      ),
      measured_rack_angle_deg=sample.measured_rack_angle_deg,
      rack_rate_deg_s=sample.rack_rate_deg_s,
      rack_acceleration_deg_s2=sample.rack_acceleration_deg_s2,
      actuator_constrained=sample.actuator_constrained,
      source_coordinate=source_coordinate,
    )


@dataclass(frozen=True, slots=True)
class BreakawayEpisode:
  """One confirmed transition from a physically stuck rack to motion."""

  direction: int
  onset_speed_mps: float
  dwell_s: float
  last_stuck: BreakawayEpisodePoint
  first_motion: BreakawayEpisodePoint
  rate_confirmation: BreakawayEpisodePoint
  angle_assisted: bool

  def __post_init__(self) -> None:
    if self.direction not in (-1, 1):
      raise ValueError("breakaway direction must be signed")
    if (
      not math.isfinite(self.onset_speed_mps)
      or self.onset_speed_mps < 0.0
      or not math.isfinite(self.dwell_s)
      or self.dwell_s < 0.0
    ):
      raise ValueError("breakaway episode values must be finite and physical")
    coordinates = (
      self.last_stuck.source_coordinate,
      self.first_motion.source_coordinate,
      self.rate_confirmation.source_coordinate,
    )
    if any(coordinate is not None for coordinate in coordinates):
      if any(coordinate is None for coordinate in coordinates):
        raise ValueError("breakaway coordinates must be complete")
      complete = tuple(coordinate for coordinate in coordinates if coordinate is not None)
      if len({coordinate.route_content_sha256 for coordinate in complete}) != 1:
        raise ValueError("breakaway coordinates must belong to one route")
      keys = tuple(coordinate.ordering_key for coordinate in complete)
      if not (keys[0] <= keys[1] <= keys[2]):
        raise ValueError("breakaway coordinates are not ordered")


@dataclass(frozen=True, slots=True)
class BreakawayDecision:
  category: BreakawayCategory
  direction: int = 0
  episode: BreakawayEpisode | None = None

  def __post_init__(self) -> None:
    if self.direction not in (-1, 0, 1):
      raise ValueError("breakaway decision direction must be signed")
    if (self.category is BreakawayCategory.BREAKAWAY) != (
      self.episode is not None
    ):
      raise ValueError("only a breakaway decision may carry an episode")
    if self.episode is not None and self.direction != self.episode.direction:
      raise ValueError("breakaway decision and episode directions disagree")
    if self.category in (
      BreakawayCategory.MOVING,
      BreakawayCategory.PENDING,
      BreakawayCategory.BREAKAWAY,
    ) and self.direction == 0:
      raise ValueError("resolved motion requires a direction")
    if self.category in (
      BreakawayCategory.BASE,
      BreakawayCategory.DISCARDED,
    ) and self.direction != 0:
      raise ValueError("non-motion decision cannot carry a direction")


@dataclass(frozen=True, slots=True)
class _PendingAngleOnset:
  direction: int
  onset_speed_mps: float
  stationary_dwell_s: float
  last_stuck_point: BreakawayEpisodePoint
  first_motion_point: BreakawayEpisodePoint
  elapsed_s: float
  confirmation_limit_s: float


class BreakawayEpisodeDetector:
  """Reconstruct one physical breakaway episode before speed weighting."""

  __slots__ = (
    "_last_stuck_point",
    "_pending",
    "_previous_point",
    "_stationary_dwell_s",
  )

  def __init__(self) -> None:
    self.reset()

  def reset(self) -> None:
    self._previous_point: BreakawayEpisodePoint | None = None
    self._last_stuck_point: BreakawayEpisodePoint | None = None
    self._stationary_dwell_s = 0.0
    self._pending: _PendingAngleOnset | None = None

  @staticmethod
  def _direction(value: float, threshold: float) -> int:
    return 1 if value >= threshold else -1 if value <= -threshold else 0

  @staticmethod
  def _valid_sample(sample: LearningSample) -> bool:
    return (
      sample._base_valid
      and math.isfinite(sample.measured_rack_angle_deg)
    )

  def _clear_motion_state(
    self,
    point: BreakawayEpisodePoint,
  ) -> None:
    self._previous_point = point
    self._last_stuck_point = None
    self._stationary_dwell_s = 0.0
    self._pending = None

  def _moving(
    self,
    point: BreakawayEpisodePoint,
    direction: int,
  ) -> BreakawayDecision:
    if direction not in (-1, 1):
      raise ValueError("moving decision requires a direction")
    self._clear_motion_state(point)
    return BreakawayDecision(BreakawayCategory.MOVING, direction)

  def _discard(
    self,
    point: BreakawayEpisodePoint,
  ) -> BreakawayDecision:
    self._clear_motion_state(point)
    return BreakawayDecision(BreakawayCategory.DISCARDED)

  def update(
    self,
    sample: LearningSample,
    rack_rate_resolution_deg_s: float,
    transport_delay_s: float,
    source_coordinate: CalibrationIngestionCoordinate | None = None,
  ) -> BreakawayDecision:
    """Classify one sample and emit an episode only after physical proof.

    Configuration errors raise after resetting state.  Invalid lifecycle or
    discontinuous ``LearningSample`` input resets state and returns ``BASE``;
    the caller already owns the sample-validity decision, while this detector
    guarantees that no pre-gap dwell can leak into a later onset.
    """
    if not isinstance(sample, LearningSample):
      self.reset()
      raise TypeError("breakaway detector requires LearningSample")
    resolution = float(rack_rate_resolution_deg_s)
    transport_delay = float(transport_delay_s)
    if (
      not math.isfinite(resolution)
      or resolution <= 0.0
      or not math.isfinite(transport_delay)
      or transport_delay < 0.0
    ):
      self.reset()
      raise ValueError("breakaway detector configuration is not physical")
    if not self._valid_sample(sample):
      self.reset()
      return BreakawayDecision(BreakawayCategory.BASE)

    point = BreakawayEpisodePoint.from_sample(sample, source_coordinate)
    previous = self._previous_point
    if previous is None:
      self._previous_point = point
      if self._direction(point.rack_rate_deg_s, resolution) == 0:
        self._last_stuck_point = point
        self._stationary_dwell_s = 0.0
        return BreakawayDecision(BreakawayCategory.BASE)
      return BreakawayDecision(
        BreakawayCategory.MOVING,
        self._direction(point.rack_rate_deg_s, resolution),
      )

    angle_delta = (
      point.measured_rack_angle_deg
      - previous.measured_rack_angle_deg
    )
    angle_threshold = 0.5 * resolution * point.dt_s
    angle_direction = self._direction(angle_delta, angle_threshold)
    rate_direction = self._direction(
      point.rack_rate_deg_s,
      resolution,
    )

    pending = self._pending
    if pending is not None:
      elapsed = pending.elapsed_s + point.dt_s
      if elapsed > pending.confirmation_limit_s:
        if (
          rate_direction != 0
          and angle_direction != 0
          and rate_direction != angle_direction
        ):
          return self._discard(point)
        observed_direction = rate_direction or angle_direction
        return (
          self._moving(point, observed_direction)
          if observed_direction != 0
          else self._discard(point)
        )
      if (
        angle_direction != 0
        and angle_direction != pending.direction
      ):
        if rate_direction != 0 and rate_direction != angle_direction:
          return self._discard(point)
        return self._moving(point, angle_direction)
      if rate_direction != 0:
        if rate_direction != pending.direction:
          return self._moving(point, rate_direction)
        episode = BreakawayEpisode(
          direction=pending.direction,
          onset_speed_mps=pending.onset_speed_mps,
          dwell_s=pending.stationary_dwell_s,
          last_stuck=pending.last_stuck_point,
          first_motion=pending.first_motion_point,
          rate_confirmation=point,
          angle_assisted=True,
        )
        self._clear_motion_state(point)
        return BreakawayDecision(
          BreakawayCategory.BREAKAWAY,
          episode.direction,
          episode,
        )
      self._pending = _PendingAngleOnset(
        direction=pending.direction,
        onset_speed_mps=pending.onset_speed_mps,
        stationary_dwell_s=pending.stationary_dwell_s,
        last_stuck_point=pending.last_stuck_point,
        first_motion_point=pending.first_motion_point,
        elapsed_s=elapsed,
        confirmation_limit_s=pending.confirmation_limit_s,
      )
      self._previous_point = point
      return BreakawayDecision(
        BreakawayCategory.PENDING,
        pending.direction,
      )

    if rate_direction != 0:
      if (
        angle_direction != 0
        and angle_direction != rate_direction
      ):
        return self._discard(point)
      if (
        self._last_stuck_point is not None
        and self._stationary_dwell_s >= transport_delay
      ):
        episode = BreakawayEpisode(
          direction=rate_direction,
          onset_speed_mps=point.speed_mps,
          dwell_s=self._stationary_dwell_s,
          last_stuck=self._last_stuck_point,
          first_motion=point,
          rate_confirmation=point,
          angle_assisted=False,
        )
        self._clear_motion_state(point)
        return BreakawayDecision(
          BreakawayCategory.BREAKAWAY,
          episode.direction,
          episode,
        )
      return self._moving(point, rate_direction)

    if angle_direction != 0:
      if (
        self._last_stuck_point is not None
        and self._stationary_dwell_s >= transport_delay
      ):
        self._pending = _PendingAngleOnset(
          direction=angle_direction,
          onset_speed_mps=point.speed_mps,
          stationary_dwell_s=self._stationary_dwell_s,
          last_stuck_point=self._last_stuck_point,
          first_motion_point=point,
          elapsed_s=0.0,
          confirmation_limit_s=transport_delay,
        )
        self._previous_point = point
        self._last_stuck_point = None
        self._stationary_dwell_s = 0.0
        return BreakawayDecision(
          BreakawayCategory.PENDING,
          angle_direction,
        )
      return self._moving(point, angle_direction)

    self._stationary_dwell_s += point.dt_s
    self._last_stuck_point = point
    self._previous_point = point
    return BreakawayDecision(BreakawayCategory.BASE)
