"""Pure low-speed stop-landing jolt detector.

This module owns no messaging, Params, filesystem, UI, planner, controller, or
vehicle behavior. Car-state samples advance the episode state at 100 Hz. IMU
samples are accepted only on real livePose updates and retain their own
monotonic timestamps.
"""
import math
from collections import deque
from dataclasses import dataclass


DETECTOR_VERSION = 1

ARM_SPEED_MPS = 1.5
APPROACH_SPEED_MPS = 0.5
REARM_SPEED_MPS = 1.0
STOP_SPEED_MPS = 0.05
STOP_CONFIRM_S = 0.2
POST_STOP_S = 0.45
ANALYSIS_BEFORE_STANDSTILL_S = 2.0
HISTORY_S = 6.0

SMOOTH_WINDOW_S = 0.30
JERK_WINDOW_S = 0.25
MIN_JERK_WINDOW_S = 0.18
PEAK_MATCH_S = 0.30

IMU_WARNING_JERK = 3.0
AEGO_WARNING_JERK = 2.5
IMU_CRITICAL_JERK = 5.0
MATCHED_ACCEL_CHANGE = 0.5
MIN_MATCHED_ACCEL_CHANGE = 0.3
POST_STOP_RELEASE_AEGO_JERK = 0.75
POST_STOP_RELEASE_AEGO_CHANGE = 0.17
POST_STOP_RELEASE_DELAY_S = 0.20
FALLBACK_AEGO_JERK = 4.0
FALLBACK_ACCEL_CHANGE = 0.8
LOW_SPEED_CRITICAL_DECEL = -1.0


@dataclass(frozen=True)
class StopJoltCarSample:
  t: float
  required_valid: bool
  long_active: bool
  should_stop: bool
  v_ego: float
  a_ego: float
  standstill: bool
  brake_pressed: bool
  gas_pressed: bool
  brake_hold_active: bool
  plan_a_target: float
  requested_accel: float
  applied_accel: float
  long_control_state: str
  lead_present: bool
  d_rel: float
  v_lead_k: float
  radar_valid: bool
  road_confounded: bool = False


@dataclass(frozen=True)
class StopJoltImuSample:
  t: float
  accel_x: float
  valid: bool
  road_confounded: bool = False


@dataclass(frozen=True)
class StopJoltEvent:
  classification: str
  severity: str
  confidence: float
  reason: str
  attribution: str
  episode_start_mono_time: float
  standstill_mono_time: float
  peak_jolt_mono_time: float
  detection_mono_time: float
  imu_jerk: float
  a_ego_jerk: float
  imu_accel_before: float
  imu_accel_after: float
  a_ego_accel_before: float
  a_ego_accel_after: float
  accel_change: float
  accel_at_02_mps: float
  v_ego_at_peak: float
  plan_a_target: float
  requested_accel: float
  applied_accel: float
  plan_accel_change: float
  requested_accel_change: float
  applied_accel_change: float
  should_stop_before: bool
  should_stop_at_peak: bool
  should_stop_after: bool
  long_control_state_before: str
  long_control_state_at_peak: str
  long_control_state_after: str
  lead_present: bool
  d_rel: float
  v_lead_k: float
  brake_pressed: bool
  gas_pressed: bool
  brake_hold_active: bool
  radar_valid: bool
  imu_valid: bool
  road_confounded: bool

  @property
  def abs_imu_jerk(self) -> float:
    return abs(self.imu_jerk)

  @property
  def abs_a_ego_jerk(self) -> float:
    return abs(self.a_ego_jerk)


@dataclass(frozen=True)
class _SmoothedPoint:
  t: float
  value: float


@dataclass(frozen=True)
class _JerkPoint:
  t: float
  jerk: float
  accel_before: float
  accel_after: float

  @property
  def accel_change(self) -> float:
    return self.accel_after - self.accel_before


@dataclass(frozen=True)
class _Candidate:
  sign: int
  imu: _JerkPoint | None
  a_ego: _JerkPoint
  fallback: bool
  calibrated_release: bool
  critical: bool
  road_confounded: bool

  @property
  def peak_time(self) -> float:
    if self.imu is None:
      return self.a_ego.t
    imu_score = abs(self.imu.jerk) / IMU_WARNING_JERK
    a_ego_score = abs(self.a_ego.jerk) / AEGO_WARNING_JERK
    return self.imu.t if imu_score >= a_ego_score else self.a_ego.t

  @property
  def score(self) -> float:
    if self.imu is None:
      return abs(self.a_ego.jerk) / FALLBACK_AEGO_JERK
    a_ego_threshold = POST_STOP_RELEASE_AEGO_JERK if self.calibrated_release else AEGO_WARNING_JERK
    return min(abs(self.imu.jerk) / IMU_WARNING_JERK,
               abs(self.a_ego.jerk) / a_ego_threshold)


def _smooth_series(samples: list[tuple[float, float]], window_s: float = SMOOTH_WINDOW_S) -> list[_SmoothedPoint]:
  """Trailing sample mean with an actual-time window."""
  result: list[_SmoothedPoint] = []
  values: deque[tuple[float, float]] = deque()
  total = 0.0
  for t, value in samples:
    values.append((t, value))
    total += value
    while values and t - values[0][0] > window_s:
      total -= values.popleft()[1]
    if values:
      result.append(_SmoothedPoint(t, total / len(values)))
  return result


def _rolling_jerk(points: list[_SmoothedPoint], window_s: float = JERK_WINDOW_S) -> list[_JerkPoint]:
  """Least-squares acceleration slope over an actual-time rolling window."""
  result: list[_JerkPoint] = []
  window: deque[_SmoothedPoint] = deque()
  for point in points:
    window.append(point)
    while window and point.t - window[0].t > window_s:
      window.popleft()
    if len(window) < 3 or window[-1].t - window[0].t < MIN_JERK_WINDOW_S:
      continue

    mean_t = sum(item.t for item in window) / len(window)
    mean_value = sum(item.value for item in window) / len(window)
    denominator = sum((item.t - mean_t) ** 2 for item in window)
    if denominator <= 1e-12:
      continue
    slope = sum((item.t - mean_t) * (item.value - mean_value) for item in window) / denominator
    result.append(_JerkPoint(
      t=(window[0].t + window[-1].t) * 0.5,
      jerk=slope,
      accel_before=window[0].value,
      accel_after=window[-1].value,
    ))
  return result


def _nearest_car(samples: list[StopJoltCarSample], t: float) -> StopJoltCarSample:
  return min(samples, key=lambda sample: abs(sample.t - t))


def _nearest_smoothed(points: list[_SmoothedPoint], t: float) -> float:
  return min(points, key=lambda point: abs(point.t - t)).value


def _sign_matches(value: float, sign: int) -> bool:
  return value * sign > 0.0


class StopJoltDetector:
  """Detect a brake grab or release snap only during the final stop landing."""

  def __init__(self):
    self.car_history: deque[StopJoltCarSample] = deque()
    self.imu_history: deque[StopJoltImuSample] = deque()
    self.last_car_time = -math.inf
    self.last_imu_time = -math.inf
    self.phase = "idle"
    self.approach_seen = False
    self.landing_start: float | None = None
    self.episode_start: float | None = None
    self.low_speed_since: float | None = None
    self.standstill_time: float | None = None

  def _prune(self, now: float) -> None:
    while self.car_history and now - self.car_history[0].t > HISTORY_S:
      self.car_history.popleft()
    while self.imu_history and now - self.imu_history[0].t > HISTORY_S:
      self.imu_history.popleft()

  def _reset_episode(self, phase: str = "idle") -> None:
    self.phase = phase
    self.approach_seen = False
    self.landing_start = None
    self.episode_start = None
    self.low_speed_since = None
    self.standstill_time = None

  def update_imu(self, sample: StopJoltImuSample) -> None:
    """Accept each livePose update once; duplicate/held timestamps are ignored."""
    if not math.isfinite(sample.t) or sample.t <= self.last_imu_time:
      return
    self.last_imu_time = sample.t
    if math.isfinite(sample.accel_x):
      self.imu_history.append(sample)
    self._prune(sample.t)

  def update_car(self, sample: StopJoltCarSample) -> StopJoltEvent | None:
    if not math.isfinite(sample.t) or sample.t <= self.last_car_time:
      return None
    self.last_car_time = sample.t
    if math.isfinite(sample.v_ego) and math.isfinite(sample.a_ego):
      self.car_history.append(sample)
    self._prune(sample.t)

    if not sample.long_active:
      self._reset_episode()
      return None

    if self.phase == "blocked":
      if sample.v_ego <= REARM_SPEED_MPS:
        return None
      self._reset_episode()

    driver_intervened = sample.brake_pressed or sample.gas_pressed
    if self.phase == "armed":
      if driver_intervened or not sample.required_valid:
        self._reset_episode("blocked")
        return None

      stopped = sample.standstill or sample.v_ego <= STOP_SPEED_MPS
      if stopped:
        if self.low_speed_since is None:
          self.low_speed_since = sample.t
        if self.standstill_time is None and sample.t - self.low_speed_since >= STOP_CONFIRM_S:
          self.standstill_time = self.low_speed_since
      else:
        self.low_speed_since = None

      if self.standstill_time is not None and sample.t - self.standstill_time >= POST_STOP_S:
        event = self._finalize(sample.t)
        self._reset_episode("blocked")
        return event
      return None

    approach_eligible = (
      sample.required_valid
      and sample.long_active
      and not driver_intervened
    )
    if not approach_eligible:
      self.approach_seen = False
      return None

    if sample.v_ego > ARM_SPEED_MPS:
      self.approach_seen = True
      self.landing_start = None
    elif sample.v_ego > APPROACH_SPEED_MPS:
      self.approach_seen = True
    if self.approach_seen and sample.should_stop and STOP_SPEED_MPS < sample.v_ego <= ARM_SPEED_MPS:
      if self.landing_start is None:
        self.landing_start = sample.t
      self.phase = "armed"
      self.episode_start = self.landing_start
    elif self.approach_seen and STOP_SPEED_MPS < sample.v_ego <= ARM_SPEED_MPS and self.landing_start is None:
      self.landing_start = sample.t
    return None

  @staticmethod
  def _road_confounded(car_samples: list[StopJoltCarSample],
                       imu_samples: list[StopJoltImuSample], peak_time: float) -> bool:
    return (
      any(sample.road_confounded and abs(sample.t - peak_time) <= 0.4 for sample in car_samples)
      or any(sample.road_confounded and abs(sample.t - peak_time) <= 0.4 for sample in imu_samples)
    )

  @staticmethod
  def _matched_candidate(sign: int, imu_jerks: list[_JerkPoint], a_ego_jerks: list[_JerkPoint],
                         car_samples: list[StopJoltCarSample], imu_samples: list[StopJoltImuSample],
                         accel_at_02_mps: float, standstill_time: float) -> _Candidate | None:
    pairs: list[tuple[float, _JerkPoint, _JerkPoint, bool]] = []
    for imu in imu_jerks:
      if not _sign_matches(imu.jerk, sign) or abs(imu.jerk) < IMU_WARNING_JERK:
        continue
      for a_ego in a_ego_jerks:
        if not _sign_matches(a_ego.jerk, sign):
          continue
        if abs(imu.t - a_ego.t) > PEAK_MATCH_S:
          continue
        imu_change = abs(imu.accel_change)
        a_ego_change = abs(a_ego.accel_change)
        standard_match = (
          abs(a_ego.jerk) >= AEGO_WARNING_JERK
          and max(imu_change, a_ego_change) >= MATCHED_ACCEL_CHANGE
          and min(imu_change, a_ego_change) >= MIN_MATCHED_ACCEL_CHANGE
        )
        # Route replay exposed a distinct release-snap shape: the device IMU
        # sees a >=3 m/s^3 rebound at standstill, while the wheel-speed aEgo
        # estimate responds 0.1-0.3 s later with a smaller but sustained
        # change. This calibrated path remains IMU-confirmed and applies only
        # after standstill; it cannot turn an aEgo-only estimator snap into an
        # event.
        calibrated_release = (
          sign > 0
          and a_ego.t >= standstill_time + POST_STOP_RELEASE_DELAY_S
          and abs(a_ego.jerk) >= POST_STOP_RELEASE_AEGO_JERK
          and imu_change >= MATCHED_ACCEL_CHANGE
          and a_ego_change >= POST_STOP_RELEASE_AEGO_CHANGE
        )
        if not standard_match and not calibrated_release:
          continue
        a_ego_threshold = POST_STOP_RELEASE_AEGO_JERK if calibrated_release and not standard_match else AEGO_WARNING_JERK
        score = min(abs(imu.jerk) / IMU_WARNING_JERK, abs(a_ego.jerk) / a_ego_threshold)
        pairs.append((score, imu, a_ego, calibrated_release and not standard_match))
    if not pairs:
      return None

    # A full two-signal match carries stronger evidence than the calibrated
    # IMU-dominant release path, even if the latter's normalized score is
    # numerically larger because its aEgo threshold is deliberately lower.
    _, imu, a_ego, calibrated_release = max(
      pairs,
      key=lambda pair: (not pair[3], pair[0], abs(pair[1].jerk) + abs(pair[2].jerk)),
    )
    peak_time = imu.t if abs(imu.jerk) / IMU_WARNING_JERK >= abs(a_ego.jerk) / AEGO_WARNING_JERK else a_ego.t
    critical = abs(imu.jerk) >= IMU_CRITICAL_JERK or accel_at_02_mps <= LOW_SPEED_CRITICAL_DECEL
    return _Candidate(
      sign,
      imu,
      a_ego,
      fallback=False,
      calibrated_release=calibrated_release,
      critical=critical,
      road_confounded=StopJoltDetector._road_confounded(car_samples, imu_samples, peak_time),
    )

  @staticmethod
  def _fallback_candidate(sign: int, a_ego_jerks: list[_JerkPoint],
                          car_samples: list[StopJoltCarSample],
                          imu_samples: list[StopJoltImuSample]) -> _Candidate | None:
    candidates = [
      point for point in a_ego_jerks
      if _sign_matches(point.jerk, sign)
      and abs(point.jerk) >= FALLBACK_AEGO_JERK
      and abs(point.accel_change) >= FALLBACK_ACCEL_CHANGE
    ]
    if not candidates:
      return None
    point = max(candidates, key=lambda candidate: abs(candidate.jerk))
    return _Candidate(
      sign,
      None,
      point,
      fallback=True,
      calibrated_release=False,
      critical=False,
      road_confounded=StopJoltDetector._road_confounded(car_samples, imu_samples, point.t),
    )

  def _finalize(self, detection_time: float) -> StopJoltEvent | None:
    assert self.episode_start is not None
    assert self.standstill_time is not None
    analysis_start = self.standstill_time - ANALYSIS_BEFORE_STANDSTILL_S
    smoothing_start = analysis_start - SMOOTH_WINDOW_S
    car_samples = [
      sample for sample in self.car_history
      if smoothing_start <= sample.t <= detection_time
    ]
    imu_samples = [
      sample for sample in self.imu_history
      if smoothing_start <= sample.t <= detection_time
    ]
    if len(car_samples) < 3:
      return None

    jerk_start = max(analysis_start, self.episode_start)
    a_ego_smoothed = _smooth_series([(sample.t, sample.a_ego) for sample in car_samples])
    a_ego_jerks = [
      point for point in _rolling_jerk(a_ego_smoothed)
      if jerk_start <= point.t <= detection_time
      and _nearest_car(car_samples, point.t).v_ego <= ARM_SPEED_MPS
    ]
    valid_imu_samples = [sample for sample in imu_samples if sample.valid]
    imu_smoothed = _smooth_series([(sample.t, sample.accel_x) for sample in valid_imu_samples])
    imu_jerks = [
      point for point in _rolling_jerk(imu_smoothed)
      if jerk_start <= point.t <= detection_time
      and _nearest_car(car_samples, point.t).v_ego <= ARM_SPEED_MPS
    ]
    imu_available = (
      len(valid_imu_samples) >= 5
      and valid_imu_samples[-1].t - valid_imu_samples[0].t >= JERK_WINDOW_S
      and bool(imu_jerks)
    )
    if not a_ego_jerks:
      return None

    low_speed_samples = [
      sample for sample in car_samples
      if sample.t <= self.standstill_time and sample.v_ego <= APPROACH_SPEED_MPS
    ]
    low_speed_sample = min(low_speed_samples, key=lambda sample: abs(sample.v_ego - 0.2)) if low_speed_samples else car_samples[-1]
    accel_at_02_mps = _nearest_smoothed(a_ego_smoothed, low_speed_sample.t)

    candidates: list[_Candidate] = []
    for sign in (-1, 1):
      candidate = (
        self._matched_candidate(
          sign, imu_jerks, a_ego_jerks, car_samples, imu_samples, accel_at_02_mps,
          self.standstill_time,
        )
        if imu_available
        else self._fallback_candidate(sign, a_ego_jerks, car_samples, imu_samples)
      )
      if candidate is None:
        continue
      # Ordinary bump-correlated warnings are noise. Preserve only severe
      # evidence, explicitly marked and confidence-reduced.
      if candidate.road_confounded and not candidate.critical:
        continue
      candidates.append(candidate)
    if not candidates:
      return None

    negative = next((candidate for candidate in candidates if candidate.sign < 0), None)
    positive = next((candidate for candidate in candidates if candidate.sign > 0), None)
    if negative is not None and positive is not None:
      classification = "grabAndRebound"
    elif negative is not None:
      classification = "brakeGrab"
    else:
      classification = "releaseSnap"

    primary = max(candidates, key=lambda candidate: (candidate.critical, candidate.score))
    peak_time = primary.peak_time
    at_peak = _nearest_car(car_samples, peak_time)
    before = _nearest_car(car_samples, peak_time - JERK_WINDOW_S)
    after = _nearest_car(car_samples, peak_time + JERK_WINDOW_S)
    plan_change = after.plan_a_target - before.plan_a_target
    requested_change = after.requested_accel - before.requested_accel
    applied_change = after.applied_accel - before.applied_accel

    command_change = max(abs(plan_change), abs(requested_change))
    if command_change >= MATCHED_ACCEL_CHANGE and (
      _sign_matches(plan_change, primary.sign) or _sign_matches(requested_change, primary.sign)
    ):
      attribution = "controller"
    elif command_change < 0.25 and abs(applied_change) < 0.25:
      attribution = "vehicle"
    else:
      attribution = "mixed"

    severity = "critical" if primary.critical else "warning"
    confidence = (
      0.60 if primary.fallback
      else (0.98 if primary.critical else (0.82 if primary.calibrated_release else 0.90))
    )
    if primary.road_confounded:
      confidence *= 0.65
    imu_jerk = primary.imu.jerk if primary.imu is not None else 0.0
    imu_before = primary.imu.accel_before if primary.imu is not None else 0.0
    imu_after = primary.imu.accel_after if primary.imu is not None else 0.0
    primary_change = primary.imu.accel_change if primary.imu is not None else primary.a_ego.accel_change
    reason = (
      f"{classification}: stop landing jerk imu={imu_jerk:+.2f} m/s^3 " +
      f"aEgo={primary.a_ego.jerk:+.2f} m/s^3"
    )

    return StopJoltEvent(
      classification=classification,
      severity=severity,
      confidence=confidence,
      reason=reason,
      attribution=attribution,
      episode_start_mono_time=self.episode_start,
      standstill_mono_time=self.standstill_time,
      peak_jolt_mono_time=peak_time,
      detection_mono_time=detection_time,
      imu_jerk=imu_jerk,
      a_ego_jerk=primary.a_ego.jerk,
      imu_accel_before=imu_before,
      imu_accel_after=imu_after,
      a_ego_accel_before=primary.a_ego.accel_before,
      a_ego_accel_after=primary.a_ego.accel_after,
      accel_change=primary_change,
      accel_at_02_mps=accel_at_02_mps,
      v_ego_at_peak=at_peak.v_ego,
      plan_a_target=at_peak.plan_a_target,
      requested_accel=at_peak.requested_accel,
      applied_accel=at_peak.applied_accel,
      plan_accel_change=plan_change,
      requested_accel_change=requested_change,
      applied_accel_change=applied_change,
      should_stop_before=before.should_stop,
      should_stop_at_peak=at_peak.should_stop,
      should_stop_after=after.should_stop,
      long_control_state_before=before.long_control_state,
      long_control_state_at_peak=at_peak.long_control_state,
      long_control_state_after=after.long_control_state,
      lead_present=at_peak.lead_present,
      d_rel=at_peak.d_rel,
      v_lead_k=at_peak.v_lead_k,
      brake_pressed=at_peak.brake_pressed,
      gas_pressed=at_peak.gas_pressed,
      brake_hold_active=at_peak.brake_hold_active,
      radar_valid=at_peak.radar_valid,
      imu_valid=primary.imu is not None,
      road_confounded=primary.road_confounded,
    )
