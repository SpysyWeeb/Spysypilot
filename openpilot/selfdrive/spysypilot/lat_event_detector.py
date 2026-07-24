"""Pure lateral driving-event detection and signal conditioning.

This module intentionally owns no messaging, Params, filesystem, UI, or process
lifecycle behavior. It retains the original detector thresholds and classifications.
"""
import math
from collections import deque
from dataclasses import dataclass


# Version 3 resets a stall arm when tracking becomes inactive and adds compact
# evidence. Detection thresholds and the three-release/six-second policy are unchanged.
DETECTOR_VERSION = 3
EVENT_COOLDOWN = 8.0
STALL_RELEASE_WINDOW_S = 6.0
STALL_TRANSITION_HYSTERESIS_S = 0.25
EVIDENCE_HISTORY_S = 15.0

DRIVER_CONFOUND_TORQUE = 1
DRIVER_CONFOUND_STEERING_PRESSED = 2


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
  steering_torque_eps: float = 0.0
  damping_applied: float = 0.0
  damping_state: str = "inactive"
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
class LateralEvidence:
  evidence_start_mono_time: float
  evidence_end_mono_time: float
  driver_confounded_fraction: float
  max_abs_driver_torque: float
  steering_pressed_any: bool
  road_confounded_fraction: float
  driver_confound_reason: int
  episode_start_mono_time: float
  episode_key: str
  analysis_window_before_s: float
  analysis_window_after_s: float
  stall_release_count: int = 0
  release_offsets_s: tuple[float, ...] = ()
  stall_durations_s: tuple[float, ...] = ()
  release_peak_rates_deg: tuple[float, ...] = ()
  stall_episode_phase: str = ""
  late_unwind_duration_s: float = 0.0
  previous_unwind_effective_phase: float = 0.0
  previous_unwind_same_episode: bool = False
  tracking_inactive_time_s: float = 0.0

  @property
  def driver_confounded_any(self) -> bool:
    return self.driver_confound_reason != 0

  @property
  def road_confounded_any(self) -> bool:
    return self.road_confounded_fraction > 0.0


@dataclass
class StallRelease:
  mono_time: float
  stall_duration_s: float
  peak_rate_deg: float
  phase: str


@dataclass(frozen=True)
class LateralDetection:
  event_type: str
  severity: str
  confidence: float
  reason: str
  evidence: LateralEvidence | None = None


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
    self.stall_armed_until = -math.inf
    self.armed_stall_duration = 0.0
    self.late_unwind_since: float | None = None
    self.authority_since: float | None = None
    self.releases: deque[StallRelease] = deque()
    self.history: deque[tuple[float, bool, float, bool, bool, bool]] = deque()
    self.episode_start: float | None = None
    self.episode_last_activity = -math.inf

  def reset_temporal(self) -> None:
    self.stationary_since = None
    self.was_stalled = False
    self.stall_armed_until = -math.inf
    self.armed_stall_duration = 0.0
    self.late_unwind_since = None
    self.authority_since = None
    self.releases.clear()
    self.episode_start = None
    self.episode_last_activity = -math.inf

  def _update_history(self, sample: LateralSample) -> None:
    tracking_active = abs(sample.reference_rate) > 0.2 or abs(sample.desired_lateral_accel - sample.actual_lateral_accel) > 0.15
    self.history.append((
      sample.mono_time,
      sample.driver_confounded,
      sample.driver_torque,
      sample.steering_pressed,
      sample.road_confounded,
      tracking_active,
    ))
    while self.history and sample.mono_time - self.history[0][0] > EVIDENCE_HISTORY_S:
      self.history.popleft()

  def _update_episode(self, sample: LateralSample, tracking_active: bool) -> None:
    active_shape = (
      abs(sample.steering_angle_deg) > 15.0
      or abs(sample.steering_rate_deg) > 8.0
      or sample.unwind_effective_phase > 0.2
      or tracking_active
    )
    if active_shape:
      if self.episode_start is None or sample.mono_time - self.episode_last_activity > 2.0:
        self.episode_start = sample.mono_time
      self.episode_last_activity = sample.mono_time
    elif self.episode_start is not None and sample.mono_time - self.episode_last_activity > 2.0:
      self.episode_start = None

  @staticmethod
  def _analysis_window(event_type: str) -> tuple[float, float]:
    return {
      "stallRelease": (7.0, 2.0),
      "lateUnwind": (3.0, 2.0),
      "centerOvershoot": (2.0, 2.0),
      "handoffMismatch": (2.0, 2.0),
      "torqueAuthority": (2.0, 2.0),
    }[event_type]

  def _evidence(self, sample: LateralSample, event_type: str, evidence_start: float,
                previous_phase: float, previous_same_episode: bool,
                stall_releases: list[StallRelease] | None = None,
                late_unwind_duration: float = 0.0) -> LateralEvidence:
    samples = [entry for entry in self.history if entry[0] >= evidence_start]
    if not samples:
      samples = [(
        sample.mono_time,
        sample.driver_confounded,
        sample.driver_torque,
        sample.steering_pressed,
        sample.road_confounded,
        True,
      )]
    driver_count = sum(entry[1] for entry in samples)
    road_count = sum(entry[4] for entry in samples)
    steering_pressed_any = any(entry[3] for entry in samples)
    max_abs_driver_torque = max(abs(entry[2]) for entry in samples)
    tracking_inactive_time = sum(
      max(0.0, samples[index + 1][0] - entry[0])
      for index, entry in enumerate(samples[:-1])
      if not entry[5]
    )
    driver_reason = 0
    if max_abs_driver_torque > 50.0:
      driver_reason |= DRIVER_CONFOUND_TORQUE
    if steering_pressed_any:
      driver_reason |= DRIVER_CONFOUND_STEERING_PRESSED

    before, after = self._analysis_window(event_type)
    episode_start = self.episode_start if self.episode_start is not None else evidence_start
    releases = stall_releases or []
    phases = {release.phase for release in releases}
    stall_phase = next(iter(phases)) if len(phases) == 1 else ("mixed" if phases else "")
    return LateralEvidence(
      evidence_start_mono_time=evidence_start,
      evidence_end_mono_time=sample.mono_time,
      driver_confounded_fraction=driver_count / len(samples),
      max_abs_driver_torque=max_abs_driver_torque,
      steering_pressed_any=steering_pressed_any,
      road_confounded_fraction=road_count / len(samples),
      driver_confound_reason=driver_reason,
      episode_start_mono_time=episode_start,
      episode_key=f"lat:{round(episode_start * 1e9)}",
      analysis_window_before_s=before,
      analysis_window_after_s=after,
      stall_release_count=len(releases),
      release_offsets_s=tuple(release.mono_time - sample.mono_time for release in releases),
      stall_durations_s=tuple(release.stall_duration_s for release in releases),
      release_peak_rates_deg=tuple(release.peak_rate_deg for release in releases),
      stall_episode_phase=stall_phase,
      late_unwind_duration_s=late_unwind_duration,
      previous_unwind_effective_phase=previous_phase,
      previous_unwind_same_episode=previous_same_episode,
      tracking_inactive_time_s=tracking_inactive_time,
    )

  def update(self, sample: LateralSample) -> LateralDetection | None:
    previous_phase = self.prev_phase
    previous_same_episode = self.prev_same_episode
    phase_handoff = previous_phase > 0.6 and (
      sample.unwind_effective_phase < 0.25 or (previous_same_episode and not sample.unwind_same_episode)
    )
    self.prev_phase = sample.unwind_effective_phase
    self.prev_same_episode = sample.unwind_same_episode
    self._update_history(sample)

    if not sample.active or sample.steering_pressed:
      self.reset_temporal()
      return None

    tracking_active = abs(sample.reference_rate) > 0.2 or abs(sample.desired_lateral_accel - sample.actual_lateral_accel) > 0.15
    self._update_episode(sample, tracking_active)
    detection: LateralDetection | None = None
    stall_releases: list[StallRelease] = []
    late_unwind_duration = 0.0
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
      late_unwind_duration = sample.mono_time - self.late_unwind_since
      detection = LateralDetection(
        "lateUnwind", "warning", 0.85,
        "future path called for unwind but the steering wheel remained stalled for over one second",
      )
      self.late_unwind_since = None

    stationary = abs(sample.steering_rate_deg) < 8.0 and tracking_active
    if stationary:
      if self.stationary_since is None:
        self.stationary_since = sample.mono_time
      self.armed_stall_duration = sample.mono_time - self.stationary_since
      self.was_stalled = self.armed_stall_duration > 0.15
      if self.was_stalled:
        self.stall_armed_until = sample.mono_time + STALL_TRANSITION_HYSTERESIS_S
    elif not tracking_active:
      self.stationary_since = None
      self.was_stalled = False
      self.stall_armed_until = -math.inf
      self.armed_stall_duration = 0.0
    elif abs(sample.steering_rate_deg) > 30.0:
      if self.was_stalled and sample.mono_time <= self.stall_armed_until:
        phase = "unwind" if (
          sample.unwind_effective_phase > 0.2 or sample.reference_sustained_unwind_scale > 0.2
        ) else "turnIn"
        self.releases.append(StallRelease(
          sample.mono_time,
          self.armed_stall_duration,
          abs(sample.steering_rate_deg),
          phase,
        ))
      self.stationary_since = None
      self.was_stalled = False
      self.stall_armed_until = -math.inf
      self.armed_stall_duration = 0.0
    elif self.was_stalled and sample.mono_time > self.stall_armed_until:
      self.stationary_since = None
      self.was_stalled = False
      self.armed_stall_duration = 0.0

    for release in self.releases:
      if sample.mono_time - release.mono_time <= STALL_TRANSITION_HYSTERESIS_S:
        release.peak_rate_deg = max(release.peak_rate_deg, abs(sample.steering_rate_deg))
    while self.releases and sample.mono_time - self.releases[0].mono_time > STALL_RELEASE_WINDOW_S:
      self.releases.popleft()
    if detection is None and len(self.releases) >= 3:
      stall_releases = list(self.releases)
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
    evidence_start = {
      "stallRelease": sample.mono_time - STALL_RELEASE_WINDOW_S,
      "lateUnwind": sample.mono_time - 3.0,
      "centerOvershoot": sample.mono_time - 2.0,
      "handoffMismatch": sample.mono_time - 2.0,
      "torqueAuthority": sample.mono_time - 2.0,
    }[detection.event_type]
    evidence = self._evidence(
      sample,
      detection.event_type,
      evidence_start,
      previous_phase,
      previous_same_episode,
      stall_releases,
      late_unwind_duration,
    )
    return LateralDetection(
      detection.event_type,
      detection.severity,
      detection.confidence,
      detection.reason,
      evidence,
    )
