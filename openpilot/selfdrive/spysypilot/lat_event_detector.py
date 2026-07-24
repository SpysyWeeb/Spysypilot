"""Pure lateral driving-event detection and signal conditioning.

This module intentionally owns no messaging, Params, filesystem, UI, or process
lifecycle behavior. It retains the original detector thresholds and classifications.
"""
import math
from collections import deque
from dataclasses import dataclass


# Version 2 intentionally uses per-event-type cooldowns. This retains a distinct
# failure in the same physical episode instead of letting one lateral type hide it.
DETECTOR_VERSION = 2
EVENT_COOLDOWN = 8.0


@dataclass
class LateralSample:
  mono_time: float
  active: bool = False
  v_ego: float = 0.0
  steering_angle_deg: float = 0.0
  steering_rate_deg: float = 0.0
  driver_torque: float = 0.0
  steering_pressed: bool = False
  desired_lateral_accel: float = 0.0
  actual_lateral_accel: float = 0.0
  request_torque: float = 0.0
  applied_torque: float = 0.0
  p_term: float = 0.0
  controller_version: int = 0
  reference_version: int = 0
  reference_rate: float = 0.0
  reference_target_torque: float = 0.0
  reference_unwind_scale: float = 0.0
  reference_sustained_unwind_scale: float = 0.0
  unwind_effective_phase: float = 0.0
  unwind_overspeed: float = 0.0
  unwind_same_episode: bool = False
  road_confounded: bool = False

  @property
  def applied_target_gap(self) -> float:
    return abs(self.applied_torque - self.reference_target_torque)

  @property
  def driver_confounded(self) -> bool:
    return self.steering_pressed or abs(self.driver_torque) > 50.0


@dataclass(frozen=True)
class LateralDetection:
  event_type: str
  severity: str
  confidence: float
  reason: str


class SteeringRateFilter:
  """Reconstruct signed steering rate with the legacy ~80 ms low-pass."""

  def __init__(self):
    self.previous_angle: float | None = None
    self.previous_time: float | None = None
    self.filtered_rate = 0.0

  def update(self, angle: float, now: float) -> float:
    dt = now - self.previous_time if self.previous_time is not None else 0.01
    if self.previous_angle is not None and 0.002 < dt < 0.1:
      raw_rate = max(-800.0, min(800.0, (angle - self.previous_angle) / dt))
      alpha = dt / (0.08 + dt)
      self.filtered_rate += alpha * (raw_rate - self.filtered_rate)
    self.previous_angle = angle
    self.previous_time = now
    return self.filtered_rate


class RoadBumpClassifier:
  def __init__(self):
    self.baseline_z: float | None = None
    self.confounded_until = -math.inf

  def update(self, z_accel: float, now: float) -> bool:
    self.baseline_z = z_accel if self.baseline_z is None else 0.995 * self.baseline_z + 0.005 * z_accel
    if abs(z_accel - self.baseline_z) > 1.25:
      self.confounded_until = now + 0.75
    return now < self.confounded_until


class LateralEventDetector:
  """Conservative, auditable detector for known BLaT failure shapes."""

  def __init__(self, cooldown: float = EVENT_COOLDOWN):
    self.cooldown = cooldown
    self.last_event_times: dict[str, float] = {}
    self.prev_phase = 0.0
    self.prev_same_episode = False
    self.stationary_since: float | None = None
    self.was_stalled = False
    self.late_unwind_since: float | None = None
    self.authority_since: float | None = None
    self.releases: deque[float] = deque()

  def reset_temporal(self) -> None:
    self.stationary_since = None
    self.was_stalled = False
    self.late_unwind_since = None
    self.authority_since = None
    self.releases.clear()

  def update(self, sample: LateralSample) -> LateralDetection | None:
    phase_handoff = self.prev_phase > 0.6 and (
      sample.unwind_effective_phase < 0.25 or (self.prev_same_episode and not sample.unwind_same_episode)
    )
    self.prev_phase = sample.unwind_effective_phase
    self.prev_same_episode = sample.unwind_same_episode

    if not sample.active or sample.steering_pressed:
      self.reset_temporal()
      return None

    detection: LateralDetection | None = None
    if phase_handoff and abs(sample.steering_rate_deg) > 60.0 and sample.applied_target_gap > 0.18:
      detection = LateralDetection(
        "handoffMismatch", "warning", 0.95,
        "unwind ownership changed while wheel motion and applied-target torque gap remained high",
      )

    if detection is None and (
      abs(sample.steering_angle_deg) < 15.0
      and abs(sample.steering_rate_deg) > 120.0
      and sample.unwind_effective_phase > 0.2
      and sample.unwind_overspeed > 0.3
      and sample.applied_target_gap > 0.12
    ):
      detection = LateralDetection(
        "centerOvershoot", "warning", 0.90,
        "wheel crossed center quickly while unwind braking still had an applied-target gap",
      )

    unwind_expected = (
      sample.reference_sustained_unwind_scale > 0.6
      and sample.unwind_effective_phase > 0.6
      and abs(sample.steering_angle_deg) > 25.0
      and abs(sample.reference_rate) > 0.2
    )
    if unwind_expected and abs(sample.steering_rate_deg) < 8.0:
      if self.late_unwind_since is None:
        self.late_unwind_since = sample.mono_time
    else:
      self.late_unwind_since = None
    if detection is None and self.late_unwind_since is not None and sample.mono_time - self.late_unwind_since > 1.0:
      detection = LateralDetection(
        "lateUnwind", "warning", 0.85,
        "future path called for unwind but the steering wheel remained stalled for over one second",
      )
      self.late_unwind_since = None

    tracking_active = abs(sample.reference_rate) > 0.2 or abs(sample.desired_lateral_accel - sample.actual_lateral_accel) > 0.15
    stationary = abs(sample.steering_rate_deg) < 8.0 and tracking_active
    if stationary:
      if self.stationary_since is None:
        self.stationary_since = sample.mono_time
      self.was_stalled = sample.mono_time - self.stationary_since > 0.15
    elif abs(sample.steering_rate_deg) > 30.0:
      if self.was_stalled:
        self.releases.append(sample.mono_time)
      self.stationary_since = None
      self.was_stalled = False
    while self.releases and sample.mono_time - self.releases[0] > 6.0:
      self.releases.popleft()
    if detection is None and len(self.releases) >= 3:
      detection = LateralDetection(
        "stallRelease", "warning", 0.85,
        "three steering stall-release cycles occurred within six seconds",
      )
      self.releases.clear()

    authority_limited = (
      abs(sample.request_torque) > 0.95
      and abs(sample.applied_torque) > 0.95
      and abs(sample.desired_lateral_accel - sample.actual_lateral_accel) > 0.35
    )
    if authority_limited:
      if self.authority_since is None:
        self.authority_since = sample.mono_time
    else:
      self.authority_since = None
    if detection is None and self.authority_since is not None and sample.mono_time - self.authority_since > 1.0:
      detection = LateralDetection(
        "torqueAuthority", "critical", 0.80,
        "requested and applied torque stayed saturated while lateral tracking error continued growing",
      )
      self.authority_since = None

    if detection is None:
      return None
    if sample.mono_time - self.last_event_times.get(detection.event_type, -math.inf) < self.cooldown:
      return None
    self.last_event_times[detection.event_type] = sample.mono_time
    return detection
