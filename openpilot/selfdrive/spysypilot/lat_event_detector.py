"""Pure lateral driving-event detection and signal conditioning.

This module intentionally owns no messaging, Params, filesystem, UI, process
lifecycle, or actuation behavior. Detector v4 distinguishes an intentionally
requested center crossing from an unrequested overshoot and keeps driver/road
interaction as attribution evidence rather than an event-suppression rule.
"""
import math
from collections import deque
from dataclasses import dataclass


DETECTOR_VERSION = 4
EVENT_COOLDOWN = 8.0
STALL_RELEASE_WINDOW_S = 6.0
STALL_TRANSITION_HYSTERESIS_S = 0.25
EVIDENCE_HISTORY_S = 15.0

CENTER_ESTABLISHED_DEG = 25.0
CENTER_RATE_THRESHOLD_DEG_S = 120.0
CENTER_UNWIND_PHASE_THRESHOLD = 0.2
CENTER_OVERSPEED_THRESHOLD = 0.3
CENTER_APPLIED_TARGET_GAP = 0.12
HANDOFF_APPLIED_TARGET_GAP = 0.18
HANDOFF_RATE_THRESHOLD_DEG_S = 60.0
HANDOFF_CONSOLIDATION_S = 0.5
DESIRED_REVERSAL_EPS = 0.12
DESIRED_REVERSAL_LOOKBACK_S = 0.75
DESIRED_REVERSAL_PRECOMMIT_S = 0.15
DESIRED_REVERSAL_COMMIT_S = 0.18
TRACKING_ERROR_THRESHOLD = 0.35
TRACKING_ERROR_PERSIST_S = 0.15
DRIVER_RAW_TORQUE_THRESHOLD = 50.0
ROAD_SUBSTANTIAL_FRACTION = 0.25

DRIVER_CONFOUND_TORQUE = 1
DRIVER_CONFOUND_STEERING_PRESSED = 2

DRIVER_INTERACTION_NONE = "none"
DRIVER_INTERACTION_POSSIBLE_RAW_TORQUE = "possibleRawTorque"
DRIVER_INTERACTION_CONFIRMED_STEERING_PRESSED = "confirmedSteeringPressed"

ROAD_INTERACTION_NONE = "none"
ROAD_INTERACTION_TRANSIENT = "transient"
ROAD_INTERACTION_SUBSTANTIAL = "substantial"


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
  damping_applied: float = 0.0  # Legacy snapshot only; never used as v4 actual damping evidence.
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
  vertical_accel_deviation: float = 0.0

  # Optional observer-only carOutput evidence. None means unavailable/non-Hyundai.
  actual_damping_amount: float | None = None
  actual_damping_state: str | None = None
  turn_in_blocked: bool | None = None
  breakaway_latch: float | None = None
  sustain_floor_contribution: float | None = None
  damping_version: int | None = None

  @property
  def applied_target_gap(self) -> float:
    return abs(self.applied_torque - self.reference_target_torque)

  @property
  def driver_confounded(self) -> bool:
    return self.steering_pressed or abs(self.driver_torque) > DRIVER_RAW_TORQUE_THRESHOLD


@dataclass(frozen=True)
class StallReleaseEvidence:
  mono_time: float
  offset_from_trigger_s: float
  stall_duration_s: float
  peak_signed_rate_deg: float
  peak_abs_rate_deg: float
  steering_angle_deg: float
  v_ego: float
  desired_lateral_accel: float
  actual_lateral_accel: float
  request_torque: float
  applied_torque: float
  reference_target_torque: float
  actual_damping_amount: float | None
  actual_damping_state: str | None
  turn_in_blocked: bool | None
  breakaway_latch: float | None
  sustain_floor_contribution: float | None
  damping_version: int | None
  driver_evidence_active: bool
  road_evidence_active: bool
  phase: str


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

  raw_torque_exceeded_fraction: float = 0.0
  longest_raw_torque_duration_s: float = 0.0
  steering_pressed_fraction: float = 0.0
  driver_torque_active_at_trigger: bool = False
  steering_pressed_at_trigger: bool = False
  driver_interaction: str = DRIVER_INTERACTION_NONE
  road_confounded_at_trigger: bool = False
  max_vertical_accel_deviation: float = 0.0
  longest_road_confounded_duration_s: float = 0.0
  road_interaction: str = ROAD_INTERACTION_NONE

  center_crossing_mono_time: float | None = None
  phase_handoff_mono_time: float | None = None
  committed_reversal: bool = False
  peak_abs_center_rate_deg: float = 0.0
  signed_center_rate_deg: float = 0.0
  established_outside_center: bool = False
  desired_lateral_accel_before_crossing: float = 0.0
  desired_lateral_accel_after_crossing: float = 0.0
  desired_reversal_commit_mono_time: float | None = None
  tracking_error_at_crossing: float = 0.0
  applied_target_gap_at_crossing: float = 0.0
  requested_torque_at_crossing: float = 0.0
  applied_torque_at_crossing: float = 0.0
  reference_target_torque_at_crossing: float = 0.0
  peak_tracking_error: float = 0.0
  peak_applied_target_gap: float = 0.0
  handoff_consolidated: bool = False
  handoff_center_delta_s: float = 0.0
  stall_releases: tuple[StallReleaseEvidence, ...] = ()

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
  peak_signed_rate_deg: float
  peak_abs_rate_deg: float
  phase: str
  steering_angle_deg: float
  v_ego: float
  desired_lateral_accel: float
  actual_lateral_accel: float
  request_torque: float
  applied_torque: float
  reference_target_torque: float
  actual_damping_amount: float | None
  actual_damping_state: str | None
  turn_in_blocked: bool | None
  breakaway_latch: float | None
  sustain_floor_contribution: float | None
  damping_version: int | None
  driver_evidence_active: bool
  road_evidence_active: bool

  @property
  def peak_rate_deg(self) -> float:
    """Compatibility alias for the v2/v3 absolute peak-rate field."""
    return self.peak_abs_rate_deg


@dataclass
class PendingCrossing:
  mono_time: float
  old_side: int
  new_side: int
  episode_start: float
  previous_phase: float
  previous_same_episode: bool
  signed_rate_deg: float
  peak_abs_rate_deg: float
  desired_lateral_accel_before: float
  desired_lateral_accel_after: float
  peak_tracking_error: float
  tracking_error_at_crossing: float
  tracking_error_since: float | None
  longest_tracking_error_s: float
  peak_applied_target_gap: float
  applied_target_gap_at_crossing: float
  requested_torque_at_crossing: float
  applied_torque_at_crossing: float
  reference_target_torque_at_crossing: float
  gap_since: float | None
  longest_gap_s: float
  unwind_phase: float
  unwind_overspeed: float
  associated_handoff_time: float | None = None
  reversal_commit_mono_time: float | None = None


@dataclass
class PendingHandoff:
  mono_time: float
  episode_start: float
  previous_phase: float
  previous_same_episode: bool
  peak_abs_rate_deg: float
  peak_tracking_error: float
  peak_applied_target_gap: float


@dataclass(frozen=True)
class LateralDetection:
  event_type: str
  severity: str
  confidence: float
  reason: str
  evidence: LateralEvidence | None = None
  occurred_mono_time: float | None = None
  detected_mono_time: float | None = None


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
    self.last_deviation = 0.0

  def update(self, z_accel: float, now: float) -> bool:
    self.baseline_z = z_accel if self.baseline_z is None else 0.995 * self.baseline_z + 0.005 * z_accel
    self.last_deviation = abs(z_accel - self.baseline_z)
    if self.last_deviation > 1.25:
      self.confounded_until = now + 0.75
    return now < self.confounded_until


class LateralEventDetector:
  """Conservative, auditable detector for known BLaT failure shapes."""

  def __init__(self, cooldown: float = EVENT_COOLDOWN):
    self.cooldown = cooldown
    self.last_event_times: dict[str, float] = {}
    self.prev_phase = 0.0
    self.prev_same_episode = False
    self.previous_sample: LateralSample | None = None
    self.established_center_side = 0
    self.pending_crossing: PendingCrossing | None = None
    self.pending_handoff: PendingHandoff | None = None
    self.stationary_since: float | None = None
    self.was_stalled = False
    self.stall_armed_until = -math.inf
    self.armed_stall_duration = 0.0
    self.late_unwind_since: float | None = None
    self.authority_since: float | None = None
    self.releases: deque[StallRelease] = deque()
    # time, driver-any, torque, pressed, road, tracking, vertical deviation,
    # desired lateral acceleration, tracking error, and applied-target gap
    self.history: deque[tuple[float, bool, float, bool, bool, bool, float, float, float, float]] = deque()
    self.episode_start: float | None = None
    self.episode_last_activity = -math.inf

  def reset_temporal(self) -> None:
    self.previous_sample = None
    self.established_center_side = 0
    self.pending_crossing = None
    self.pending_handoff = None
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
      abs(sample.vertical_accel_deviation),
      sample.desired_lateral_accel,
      abs(sample.desired_lateral_accel - sample.actual_lateral_accel),
      sample.applied_target_gap,
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
      "committedHandoffHarshness": (2.0, 2.0),
      "handoffMismatch": (2.0, 2.0),
      "torqueAuthority": (2.0, 2.0),
    }[event_type]

  @staticmethod
  def _longest_true_duration(samples: list[tuple], truth_index: int) -> float:
    longest = 0.0
    started: float | None = None
    for entry in samples:
      if entry[truth_index]:
        started = entry[0] if started is None else started
      elif started is not None:
        longest = max(longest, entry[0] - started)
        started = None
    if started is not None:
      longest = max(longest, samples[-1][0] - started)
    return longest

  def _trigger_entry(self, occurred_time: float) -> tuple:
    # Driver/road classifications are held observations. Use the last value
    # known at the interpolated trigger, never a future sample.
    previous = [entry for entry in self.history if entry[0] <= occurred_time]
    return previous[-1] if previous else self.history[0]

  def _evidence(self, sample: LateralSample, event_type: str, evidence_start: float,
                previous_phase: float, previous_same_episode: bool,
                stall_releases: list[StallRelease] | None = None,
                late_unwind_duration: float = 0.0,
                occurred_time: float | None = None,
                crossing: PendingCrossing | None = None,
                handoff_time: float | None = None) -> LateralEvidence:
    samples = [entry for entry in self.history if entry[0] >= evidence_start]
    if not samples:
      samples = [(
        sample.mono_time,
        sample.driver_confounded,
        sample.driver_torque,
        sample.steering_pressed,
        sample.road_confounded,
        True,
        abs(sample.vertical_accel_deviation),
        sample.desired_lateral_accel,
        abs(sample.desired_lateral_accel - sample.actual_lateral_accel),
        sample.applied_target_gap,
      )]
    driver_count = sum(entry[1] for entry in samples)
    raw_torque_count = sum(abs(entry[2]) > DRIVER_RAW_TORQUE_THRESHOLD for entry in samples)
    road_count = sum(entry[4] for entry in samples)
    steering_pressed_count = sum(entry[3] for entry in samples)
    steering_pressed_any = steering_pressed_count > 0
    max_abs_driver_torque = max(abs(entry[2]) for entry in samples)
    tracking_inactive_time = sum(
      max(0.0, samples[index + 1][0] - entry[0])
      for index, entry in enumerate(samples[:-1])
      if not entry[5]
    )
    driver_reason = 0
    if max_abs_driver_torque > DRIVER_RAW_TORQUE_THRESHOLD:
      driver_reason |= DRIVER_CONFOUND_TORQUE
    if steering_pressed_any:
      driver_reason |= DRIVER_CONFOUND_STEERING_PRESSED
    driver_interaction = (
      DRIVER_INTERACTION_CONFIRMED_STEERING_PRESSED if steering_pressed_any
      else DRIVER_INTERACTION_POSSIBLE_RAW_TORQUE if driver_reason & DRIVER_CONFOUND_TORQUE
      else DRIVER_INTERACTION_NONE
    )

    trigger_time = sample.mono_time if occurred_time is None else occurred_time
    trigger = self._trigger_entry(trigger_time)
    longest_road = self._longest_true_duration(samples, 4)
    road_fraction = road_count / len(samples)
    road_interaction = (
      ROAD_INTERACTION_SUBSTANTIAL
      if road_fraction >= ROAD_SUBSTANTIAL_FRACTION
      else ROAD_INTERACTION_TRANSIENT if road_count else ROAD_INTERACTION_NONE
    )

    before, after = self._analysis_window(event_type)
    episode_start = self.episode_start if self.episode_start is not None else evidence_start
    releases = stall_releases or []
    phases = {release.phase for release in releases}
    stall_phase = next(iter(phases)) if len(phases) == 1 else ("mixed" if phases else "")
    structured_releases = tuple(
      StallReleaseEvidence(
        mono_time=release.mono_time,
        offset_from_trigger_s=release.mono_time - trigger_time,
        stall_duration_s=release.stall_duration_s,
        peak_signed_rate_deg=release.peak_signed_rate_deg,
        peak_abs_rate_deg=release.peak_abs_rate_deg,
        steering_angle_deg=release.steering_angle_deg,
        v_ego=release.v_ego,
        desired_lateral_accel=release.desired_lateral_accel,
        actual_lateral_accel=release.actual_lateral_accel,
        request_torque=release.request_torque,
        applied_torque=release.applied_torque,
        reference_target_torque=release.reference_target_torque,
        actual_damping_amount=release.actual_damping_amount,
        actual_damping_state=release.actual_damping_state,
        turn_in_blocked=release.turn_in_blocked,
        breakaway_latch=release.breakaway_latch,
        sustain_floor_contribution=release.sustain_floor_contribution,
        damping_version=release.damping_version,
        driver_evidence_active=release.driver_evidence_active,
        road_evidence_active=release.road_evidence_active,
        phase=release.phase,
      )
      for release in releases
    )
    return LateralEvidence(
      evidence_start_mono_time=evidence_start,
      evidence_end_mono_time=sample.mono_time,
      driver_confounded_fraction=driver_count / len(samples),
      max_abs_driver_torque=max_abs_driver_torque,
      steering_pressed_any=steering_pressed_any,
      road_confounded_fraction=road_fraction,
      driver_confound_reason=driver_reason,
      episode_start_mono_time=episode_start,
      episode_key=f"lat:{round(episode_start * 1e9)}",
      analysis_window_before_s=before,
      analysis_window_after_s=after,
      stall_release_count=len(releases),
      release_offsets_s=tuple(release.mono_time - trigger_time for release in releases),
      stall_durations_s=tuple(release.stall_duration_s for release in releases),
      release_peak_rates_deg=tuple(release.peak_abs_rate_deg for release in releases),
      stall_episode_phase=stall_phase,
      late_unwind_duration_s=late_unwind_duration,
      previous_unwind_effective_phase=previous_phase,
      previous_unwind_same_episode=previous_same_episode,
      tracking_inactive_time_s=tracking_inactive_time,
      raw_torque_exceeded_fraction=raw_torque_count / len(samples),
      longest_raw_torque_duration_s=self._longest_true_duration(
        [(entry[0], abs(entry[2]) > DRIVER_RAW_TORQUE_THRESHOLD) for entry in samples], 1,
      ),
      steering_pressed_fraction=steering_pressed_count / len(samples),
      driver_torque_active_at_trigger=abs(trigger[2]) > DRIVER_RAW_TORQUE_THRESHOLD,
      steering_pressed_at_trigger=bool(trigger[3]),
      driver_interaction=driver_interaction,
      road_confounded_at_trigger=bool(trigger[4]),
      max_vertical_accel_deviation=max(entry[6] for entry in samples),
      longest_road_confounded_duration_s=longest_road,
      road_interaction=road_interaction,
      center_crossing_mono_time=crossing.mono_time if crossing is not None else None,
      phase_handoff_mono_time=handoff_time,
      committed_reversal=event_type == "committedHandoffHarshness",
      peak_abs_center_rate_deg=crossing.peak_abs_rate_deg if crossing is not None else 0.0,
      signed_center_rate_deg=crossing.signed_rate_deg if crossing is not None else 0.0,
      established_outside_center=crossing is not None,
      desired_lateral_accel_before_crossing=(
        crossing.desired_lateral_accel_before if crossing is not None else 0.0
      ),
      desired_lateral_accel_after_crossing=(
        crossing.desired_lateral_accel_after if crossing is not None else 0.0
      ),
      desired_reversal_commit_mono_time=(
        crossing.reversal_commit_mono_time if crossing is not None else None
      ),
      tracking_error_at_crossing=(
        crossing.tracking_error_at_crossing if crossing is not None else 0.0
      ),
      applied_target_gap_at_crossing=(
        crossing.applied_target_gap_at_crossing if crossing is not None else 0.0
      ),
      requested_torque_at_crossing=(
        crossing.requested_torque_at_crossing if crossing is not None else 0.0
      ),
      applied_torque_at_crossing=(
        crossing.applied_torque_at_crossing if crossing is not None else 0.0
      ),
      reference_target_torque_at_crossing=(
        crossing.reference_target_torque_at_crossing if crossing is not None else 0.0
      ),
      peak_tracking_error=crossing.peak_tracking_error if crossing is not None else 0.0,
      peak_applied_target_gap=crossing.peak_applied_target_gap if crossing is not None else 0.0,
      handoff_consolidated=crossing is not None and handoff_time is not None,
      handoff_center_delta_s=(
        abs(crossing.mono_time - handoff_time)
        if crossing is not None and handoff_time is not None else 0.0
      ),
      stall_releases=structured_releases,
    )

  def _reversal_commit_time(self, crossing: PendingCrossing, now: float) -> float | None:
    if now - crossing.mono_time < DESIRED_REVERSAL_COMMIT_S:
      return None
    entries = [
      entry for entry in self.history
      if crossing.mono_time - DESIRED_REVERSAL_LOOKBACK_S <= entry[0] <= now
    ]
    transitions: list[tuple[tuple, tuple]] = []
    for index, current in enumerate(entries):
      if abs(current[7]) < DESIRED_REVERSAL_EPS:
        continue
      current_sign = 1 if current[7] > 0.0 else -1
      previous_index = next((
        earlier_index for earlier_index in range(index - 1, -1, -1)
        if abs(entries[earlier_index][7]) >= DESIRED_REVERSAL_EPS
      ), None)
      if previous_index is None:
        continue
      previous = entries[previous_index]
      previous_sign = 1 if previous[7] > 0.0 else -1
      if previous_sign == current_sign or current[0] > crossing.mono_time + 0.05:
        continue
      run_start = previous[0]
      for earlier in reversed(entries[:previous_index]):
        if (
          abs(earlier[7]) < DESIRED_REVERSAL_EPS
          or (1 if earlier[7] > 0.0 else -1) != previous_sign
        ):
          break
        run_start = earlier[0]
      if previous[0] - run_start >= DESIRED_REVERSAL_PRECOMMIT_S:
        transitions.append((previous, current))
    for previous, first_new in reversed(transitions):
      new_sign = 1 if first_new[7] > 0.0 else -1
      if now - first_new[0] < DESIRED_REVERSAL_COMMIT_S:
        continue
      if any(
        entry[0] >= first_new[0]
        and (
          abs(entry[7]) < DESIRED_REVERSAL_EPS
          or (1 if entry[7] > 0.0 else -1) != new_sign
        )
        for entry in entries
      ):
        continue
      crossing.desired_lateral_accel_before = previous[7]
      crossing.desired_lateral_accel_after = first_new[7]
      return first_new[0] + DESIRED_REVERSAL_COMMIT_S
    return None

  @staticmethod
  def _same_episode(crossing: PendingCrossing, handoff: PendingHandoff) -> bool:
    return abs(crossing.episode_start - handoff.episode_start) < 1e-6

  def _emit(self, sample: LateralSample, event_type: str, severity: str, confidence: float,
            reason: str, occurred_time: float, previous_phase: float,
            previous_same_episode: bool, stall_releases: list[StallRelease] | None = None,
            late_unwind_duration: float = 0.0, crossing: PendingCrossing | None = None,
            handoff_time: float | None = None) -> LateralDetection | None:
    if sample.mono_time - self.last_event_times.get(event_type, -math.inf) < self.cooldown:
      return None
    self.last_event_times[event_type] = sample.mono_time
    evidence_start = {
      "stallRelease": sample.mono_time - STALL_RELEASE_WINDOW_S,
      "lateUnwind": sample.mono_time - 3.0,
      "centerOvershoot": occurred_time - 2.0,
      "committedHandoffHarshness": occurred_time - 2.0,
      "handoffMismatch": occurred_time - 2.0,
      "torqueAuthority": sample.mono_time - 2.0,
    }[event_type]
    evidence = self._evidence(
      sample,
      event_type,
      evidence_start,
      previous_phase,
      previous_same_episode,
      stall_releases,
      late_unwind_duration,
      occurred_time,
      crossing,
      handoff_time,
    )
    return LateralDetection(
      event_type,
      severity,
      confidence,
      reason,
      evidence,
      occurred_time,
      sample.mono_time,
    )

  def _continuous_above_start(self, value_index: int, threshold: float, now: float) -> float | None:
    started: float | None = None
    for entry in reversed(self.history):
      if entry[0] > now:
        continue
      if entry[value_index] < threshold:
        break
      started = entry[0]
    return started

  def _longest_above_duration(self, value_index: int, threshold: float,
                              start: float, end: float) -> float:
    samples = [
      (entry[0], entry[value_index] >= threshold)
      for entry in self.history
      if start <= entry[0] <= end
    ]
    return self._longest_true_duration(samples, 1) if samples else 0.0

  def _observe_center_crossing(self, sample: LateralSample, previous_phase: float,
                               previous_same_episode: bool) -> None:
    previous = self.previous_sample
    side = 1 if sample.steering_angle_deg > 0.0 else -1 if sample.steering_angle_deg < 0.0 else 0
    if previous is None:
      if abs(sample.steering_angle_deg) >= CENTER_ESTABLISHED_DEG:
        self.established_center_side = side
      return
    previous_side = 1 if previous.steering_angle_deg > 0.0 else -1 if previous.steering_angle_deg < 0.0 else 0
    changed_side = previous_side != 0 and side != 0 and previous_side != side
    if not changed_side or self.established_center_side != previous_side:
      if abs(sample.steering_angle_deg) >= CENTER_ESTABLISHED_DEG:
        self.established_center_side = side
      return
    dt = sample.mono_time - previous.mono_time
    if dt <= 0.0:
      return
    fraction = abs(previous.steering_angle_deg) / (
      abs(previous.steering_angle_deg) + abs(sample.steering_angle_deg)
    )
    crossing_time = previous.mono_time + fraction * dt
    error = abs(sample.desired_lateral_accel - sample.actual_lateral_accel)
    gap = sample.applied_target_gap
    tracking_error_start = self._continuous_above_start(8, TRACKING_ERROR_THRESHOLD, sample.mono_time)
    gap_start = self._continuous_above_start(9, CENTER_APPLIED_TARGET_GAP, sample.mono_time)
    pre_window_start = crossing_time - max(TRACKING_ERROR_PERSIST_S * 2.0, 0.3)
    prior_tracking_error_duration = self._longest_above_duration(
      8, TRACKING_ERROR_THRESHOLD, pre_window_start, sample.mono_time,
    )
    prior_gap_duration = self._longest_above_duration(
      9, CENTER_APPLIED_TARGET_GAP, pre_window_start, sample.mono_time,
    )
    signed_rate = (
      previous.steering_rate_deg
      if abs(previous.steering_rate_deg) >= abs(sample.steering_rate_deg)
      else sample.steering_rate_deg
    )
    episode_start = self.episode_start if self.episode_start is not None else previous.mono_time
    crossing = PendingCrossing(
      mono_time=crossing_time,
      old_side=previous_side,
      new_side=side,
      episode_start=episode_start,
      previous_phase=previous_phase,
      previous_same_episode=previous_same_episode,
      signed_rate_deg=signed_rate,
      peak_abs_rate_deg=max(abs(previous.steering_rate_deg), abs(sample.steering_rate_deg)),
      desired_lateral_accel_before=previous.desired_lateral_accel,
      desired_lateral_accel_after=sample.desired_lateral_accel,
      peak_tracking_error=max(
        abs(previous.desired_lateral_accel - previous.actual_lateral_accel),
        error,
      ),
      tracking_error_at_crossing=max(
        abs(previous.desired_lateral_accel - previous.actual_lateral_accel),
        error,
      ),
      tracking_error_since=tracking_error_start,
      longest_tracking_error_s=max(
        prior_tracking_error_duration,
        sample.mono_time - tracking_error_start if tracking_error_start is not None else 0.0,
      ),
      peak_applied_target_gap=max(previous.applied_target_gap, gap),
      applied_target_gap_at_crossing=max(previous.applied_target_gap, gap),
      requested_torque_at_crossing=(
        previous.request_torque + fraction * (sample.request_torque - previous.request_torque)
      ),
      applied_torque_at_crossing=(
        previous.applied_torque + fraction * (sample.applied_torque - previous.applied_torque)
      ),
      reference_target_torque_at_crossing=(
        previous.reference_target_torque
        + fraction * (sample.reference_target_torque - previous.reference_target_torque)
      ),
      gap_since=gap_start,
      longest_gap_s=max(
        prior_gap_duration,
        sample.mono_time - gap_start if gap_start is not None else 0.0,
      ),
      unwind_phase=max(previous.unwind_effective_phase, sample.unwind_effective_phase),
      unwind_overspeed=max(previous.unwind_overspeed, sample.unwind_overspeed),
    )
    if (
      self.pending_handoff is not None
      and abs(crossing_time - self.pending_handoff.mono_time) <= HANDOFF_CONSOLIDATION_S
      and self._same_episode(crossing, self.pending_handoff)
    ):
      crossing.associated_handoff_time = self.pending_handoff.mono_time
      crossing.peak_abs_rate_deg = max(crossing.peak_abs_rate_deg, self.pending_handoff.peak_abs_rate_deg)
      crossing.peak_tracking_error = max(crossing.peak_tracking_error, self.pending_handoff.peak_tracking_error)
      crossing.peak_applied_target_gap = max(crossing.peak_applied_target_gap, self.pending_handoff.peak_applied_target_gap)
      self.pending_handoff = None
    self.pending_crossing = crossing
    self.established_center_side = side if abs(sample.steering_angle_deg) >= CENTER_ESTABLISHED_DEG else 0

  def _update_pending_crossing(self, sample: LateralSample) -> LateralDetection | None:
    crossing = self.pending_crossing
    if crossing is None:
      return None
    error = abs(sample.desired_lateral_accel - sample.actual_lateral_accel)
    crossing.peak_abs_rate_deg = max(crossing.peak_abs_rate_deg, abs(sample.steering_rate_deg))
    crossing.peak_tracking_error = max(crossing.peak_tracking_error, error)
    crossing.peak_applied_target_gap = max(crossing.peak_applied_target_gap, sample.applied_target_gap)
    if error >= TRACKING_ERROR_THRESHOLD:
      crossing.tracking_error_since = (
        sample.mono_time if crossing.tracking_error_since is None else crossing.tracking_error_since
      )
      crossing.longest_tracking_error_s = max(
        crossing.longest_tracking_error_s,
        sample.mono_time - crossing.tracking_error_since,
      )
    else:
      if crossing.tracking_error_since is not None:
        crossing.longest_tracking_error_s = max(
          crossing.longest_tracking_error_s,
          sample.mono_time - crossing.tracking_error_since,
        )
      crossing.tracking_error_since = None
    if sample.applied_target_gap >= CENTER_APPLIED_TARGET_GAP:
      crossing.gap_since = sample.mono_time if crossing.gap_since is None else crossing.gap_since
      crossing.longest_gap_s = max(crossing.longest_gap_s, sample.mono_time - crossing.gap_since)
    else:
      if crossing.gap_since is not None:
        crossing.longest_gap_s = max(
          crossing.longest_gap_s,
          sample.mono_time - crossing.gap_since,
        )
      crossing.gap_since = None
    if (
      self.pending_handoff is not None
      and abs(crossing.mono_time - self.pending_handoff.mono_time) <= HANDOFF_CONSOLIDATION_S
      and self._same_episode(crossing, self.pending_handoff)
    ):
      crossing.associated_handoff_time = self.pending_handoff.mono_time
      crossing.peak_abs_rate_deg = max(crossing.peak_abs_rate_deg, self.pending_handoff.peak_abs_rate_deg)
      crossing.peak_tracking_error = max(crossing.peak_tracking_error, self.pending_handoff.peak_tracking_error)
      crossing.peak_applied_target_gap = max(crossing.peak_applied_target_gap, self.pending_handoff.peak_applied_target_gap)
      self.pending_handoff = None

    # Wait through the full consolidation window so a following phase handoff can
    # be folded into the same physical transition.
    if sample.mono_time - crossing.mono_time < HANDOFF_CONSOLIDATION_S:
      return None
    crossing.reversal_commit_mono_time = self._reversal_commit_time(crossing, sample.mono_time)
    committed = crossing.reversal_commit_mono_time is not None
    fast = crossing.peak_abs_rate_deg >= CENTER_RATE_THRESHOLD_DEG_S
    tracking_persistent = (
      crossing.longest_tracking_error_s >= TRACKING_ERROR_PERSIST_S
    )
    gap_persistent = (
      crossing.longest_gap_s >= TRACKING_ERROR_PERSIST_S
    )
    harsh = (
      fast
      or tracking_persistent
      or gap_persistent
    )
    legacy_overshoot_shape = (
      fast
      and crossing.unwind_phase > CENTER_UNWIND_PHASE_THRESHOLD
      and crossing.unwind_overspeed > CENTER_OVERSPEED_THRESHOLD
      and crossing.peak_applied_target_gap > CENTER_APPLIED_TARGET_GAP
    )
    self.pending_crossing = None
    if committed and harsh:
      return self._emit(
        sample,
        "committedHandoffHarshness",
        "warning",
        0.92,
        "requested trajectory reversal crossed center with excessive transition harshness",
        crossing.mono_time,
        crossing.previous_phase,
        crossing.previous_same_episode,
        crossing=crossing,
        handoff_time=crossing.associated_handoff_time,
      )
    if not committed and legacy_overshoot_shape:
      return self._emit(
        sample,
        "centerOvershoot",
        "warning",
        0.90,
        "wheel crossed center quickly without a committed requested reversal",
        crossing.mono_time,
        crossing.previous_phase,
        crossing.previous_same_episode,
        crossing=crossing,
        handoff_time=crossing.associated_handoff_time,
      )
    return None

  def _update_pending_handoff(self, sample: LateralSample) -> LateralDetection | None:
    handoff = self.pending_handoff
    if handoff is None or sample.mono_time - handoff.mono_time < HANDOFF_CONSOLIDATION_S:
      return None
    self.pending_handoff = None
    return self._emit(
      sample,
      "handoffMismatch",
      "warning",
      0.95,
      "unwind ownership changed while wheel motion and applied-target torque gap remained high",
      handoff.mono_time,
      handoff.previous_phase,
      handoff.previous_same_episode,
      handoff_time=handoff.mono_time,
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

    if not sample.active:
      self.reset_temporal()
      return None

    tracking_active = abs(sample.reference_rate) > 0.2 or abs(sample.desired_lateral_accel - sample.actual_lateral_accel) > 0.15
    self._update_episode(sample, tracking_active)
    self._observe_center_crossing(sample, previous_phase, previous_same_episode)
    if phase_handoff and abs(sample.steering_rate_deg) > HANDOFF_RATE_THRESHOLD_DEG_S and sample.applied_target_gap > HANDOFF_APPLIED_TARGET_GAP:
      episode_start = self.episode_start if self.episode_start is not None else sample.mono_time
      self.pending_handoff = PendingHandoff(
        sample.mono_time,
        episode_start,
        previous_phase,
        previous_same_episode,
        abs(sample.steering_rate_deg),
        abs(sample.desired_lateral_accel - sample.actual_lateral_accel),
        sample.applied_target_gap,
      )
      if (
        self.pending_crossing is not None
        and abs(self.pending_crossing.mono_time - sample.mono_time) <= HANDOFF_CONSOLIDATION_S
        and self._same_episode(self.pending_crossing, self.pending_handoff)
      ):
        self.pending_crossing.associated_handoff_time = sample.mono_time

    detection = self._update_pending_crossing(sample)
    if detection is None:
      detection = self._update_pending_handoff(sample)

    stall_releases: list[StallRelease] = []
    late_unwind_duration = 0.0
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
      detection = self._emit(
        sample,
        "lateUnwind",
        "warning",
        0.85,
        "future path called for unwind but the steering wheel remained stalled for over one second",
        sample.mono_time,
        previous_phase,
        previous_same_episode,
        late_unwind_duration=late_unwind_duration,
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
          mono_time=sample.mono_time,
          stall_duration_s=self.armed_stall_duration,
          peak_signed_rate_deg=sample.steering_rate_deg,
          peak_abs_rate_deg=abs(sample.steering_rate_deg),
          phase=phase,
          steering_angle_deg=sample.steering_angle_deg,
          v_ego=sample.v_ego,
          desired_lateral_accel=sample.desired_lateral_accel,
          actual_lateral_accel=sample.actual_lateral_accel,
          request_torque=sample.request_torque,
          applied_torque=sample.applied_torque,
          reference_target_torque=sample.reference_target_torque,
          actual_damping_amount=sample.actual_damping_amount,
          actual_damping_state=sample.actual_damping_state,
          turn_in_blocked=sample.turn_in_blocked,
          breakaway_latch=sample.breakaway_latch,
          sustain_floor_contribution=sample.sustain_floor_contribution,
          damping_version=sample.damping_version,
          driver_evidence_active=sample.driver_confounded,
          road_evidence_active=sample.road_confounded,
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
        if abs(sample.steering_rate_deg) > release.peak_abs_rate_deg:
          release.peak_signed_rate_deg = sample.steering_rate_deg
          release.peak_abs_rate_deg = abs(sample.steering_rate_deg)
    while self.releases and sample.mono_time - self.releases[0].mono_time > STALL_RELEASE_WINDOW_S:
      self.releases.popleft()
    if detection is None and len(self.releases) >= 3:
      stall_releases = list(self.releases)
      detection = self._emit(
        sample,
        "stallRelease",
        "warning",
        0.85,
        "three steering stall-release cycles occurred within six seconds",
        sample.mono_time,
        previous_phase,
        previous_same_episode,
        stall_releases=stall_releases,
      )
      self.releases.clear()

    authority_limited = (
      abs(sample.request_torque) > 0.95
      and abs(sample.applied_torque) > 0.95
      and abs(sample.desired_lateral_accel - sample.actual_lateral_accel) > TRACKING_ERROR_THRESHOLD
    )
    if authority_limited:
      if self.authority_since is None:
        self.authority_since = sample.mono_time
    else:
      self.authority_since = None
    if detection is None and self.authority_since is not None and sample.mono_time - self.authority_since > 1.0:
      detection = self._emit(
        sample,
        "torqueAuthority",
        "critical",
        0.80,
        "requested and applied torque stayed saturated while lateral tracking error continued growing",
        sample.mono_time,
        previous_phase,
        previous_same_episode,
      )
      self.authority_since = None

    # Keep the last non-zero angle so an exactly-zero wheel sample does not
    # erase the bracketing sample needed to interpolate the physical crossing.
    if abs(sample.steering_angle_deg) > 1e-9:
      self.previous_sample = sample
    return detection
