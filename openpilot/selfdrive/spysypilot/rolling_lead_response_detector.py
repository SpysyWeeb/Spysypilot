"""Pure observer for late responses to a rolling lead pulling away.

The detector owns no messaging, filesystem, controller, planner, or vehicle
behavior. Callers provide synchronized monotonic samples and translate the
returned snake-case event type through the driving-event name normalizer.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import fmean


DETECTOR_NAME = "rollingLeadResponseDetector"
DETECTOR_VERSION = 1

MIN_EGO_SPEED_MPS = 1.5
MAX_EGO_SPEED_MPS = 25.0
CANCEL_EGO_SPEED_MPS = 1.0
MIN_LEAD_DISTANCE_M = 4.0
MAX_LEAD_DISTANCE_M = 60.0
TRACK_CONFIRM_S = 0.75
LEAD_LOSS_GRACE_S = 0.30
RADAR_DISTANCE_JUMP_M = 1.5

BASELINE_WINDOW_S = 0.75
LEAD_SPEED_RISE_WINDOW_S = 1.25
LEAD_SPEED_RISE_MPS = 2.0
LEAD_ACCEL_FILTER_S = 0.20
LEAD_ACCEL_THRESHOLD_MPS2 = 1.0
LEAD_ACCEL_CONFIRM_S = 0.40
LEAD_RELATIVE_SPEED_MPS = 1.5
MIN_COMMIT_GAP_GROWTH_M = 0.30
MIN_COMMIT_SPEED_RISE_MPS = 0.30

RESPONSE_CONFIRM_S = 0.20
PLANNER_BASELINE_RISE_MPS2 = 0.50
OUTPUT_BASELINE_RISE_MPS2 = 0.50
EGO_BASELINE_RISE_MPS2 = 0.40
RESPONSE_ABSOLUTE_ACCEL_MPS2 = 0.30

EVALUATION_WINDOW_S = 2.50
MIN_EVENT_GAP_GROWTH_M = 3.0
PLANNER_BAD_LATENCY_S = 0.75
EGO_BAD_LATENCY_S = 1.0
CONTROLLER_BAD_FOLLOW_S = 0.25
VEHICLE_BAD_FOLLOW_S = 0.50
HIGH_RELATIVE_SPEED_MPS = 3.0
LOW_EGO_ACCEL_MPS2 = 0.50
RAPID_GAP_GROWTH_M = 5.0
RAPID_GAP_GROWTH_WINDOW_S = 2.0

EPISODE_COOLDOWN_S = 25.0
SETTLED_RELATIVE_SPEED_MPS = 0.8
SETTLED_GAP_CHANGE_M = 0.5
SETTLED_CONFIRM_S = 0.75
HISTORY_S = 3.0


@dataclass(frozen=True)
class RollingLeadSample:
  t: float
  long_active: bool
  v_ego: float
  a_ego: float
  gas_pressed: bool
  brake_pressed: bool
  lead_present: bool
  lead_valid: bool
  radar_valid: bool
  radar_track_id: int
  d_rel: float
  v_lead: float
  v_lead_k: float
  a_lead_k: float
  plan_valid: bool
  a_target: float
  should_stop: bool
  output_valid: bool
  output_accel: float


@dataclass(frozen=True)
class RollingLeadOnsetSnapshot:
  kind: str
  t: float
  d_rel: float
  v_lead: float
  v_lead_k: float
  a_lead_k: float
  v_ego: float
  a_ego: float
  a_target: float
  output_accel: float
  should_stop: bool
  gas_pressed: bool
  brake_pressed: bool


@dataclass(frozen=True)
class RollingLeadResponseEvent:
  event_type: str
  attribution_detail: str
  detail: str
  severity: int
  confidence: float
  occurred_mono_time: float
  detected_mono_time: float
  lead_commit_mono_time: float
  planner_response_mono_time: float | None
  controller_response_mono_time: float | None
  ego_response_mono_time: float | None
  lead_to_plan_s: float | None
  lead_to_command_s: float | None
  lead_to_ego_s: float | None
  baseline_lead_speed: float
  peak_lead_speed: float
  baseline_ego_speed: float
  peak_ego_speed: float
  peak_relative_speed: float
  baseline_gap: float
  final_gap: float
  max_gap_growth: float
  baseline_lead_accel: float
  peak_lead_accel: float
  baseline_planner_accel: float
  peak_planner_accel: float
  baseline_output_accel: float
  peak_output_accel: float
  baseline_ego_accel: float
  peak_ego_accel: float
  radar_track_id: int
  radar_discontinuity: bool
  driver_confounded: bool
  onsets: tuple[RollingLeadOnsetSnapshot, ...]
  episode_start_mono_time: float
  analysis_window_before_s: float = 5.0
  analysis_window_after_s: float = 8.0


@dataclass(frozen=True)
class _Baseline:
  lead_speed: float
  ego_speed: float
  gap: float
  lead_accel: float
  planner_accel: float
  output_accel: float
  ego_accel: float


def _mean(samples: list[RollingLeadSample], field: str) -> float:
  return fmean(getattr(sample, field) for sample in samples)


class RollingLeadResponseDetector:
  """Detect one sparse late-response event per continuous lead pull-away."""

  def __init__(self) -> None:
    self._history: deque[RollingLeadSample] = deque()
    self._track_id: int | None = None
    self._track_start: float | None = None
    self._last_lead_seen: float | None = None
    self._last_sample_t: float | None = None
    self._last_radar_t: float | None = None
    self._last_d_rel: float | None = None
    self._phase = "tracking"
    self._cooldown_until = 0.0
    self._settled_since: float | None = None
    self._discontinuity_blocked = False
    self._discontinuity_settled_since: float | None = None
    self._commit_t: float | None = None
    self._baseline: _Baseline | None = None
    self._commit_snapshot: RollingLeadOnsetSnapshot | None = None
    self._condition_starts: dict[str, float] = {}
    self._condition_snapshots: dict[str, RollingLeadOnsetSnapshot] = {}
    self._onsets: dict[str, float] = {}
    self._onset_snapshots: dict[str, RollingLeadOnsetSnapshot] = {}
    self._peak_lead_speed = 0.0
    self._peak_ego_speed = 0.0
    self._peak_relative_speed = 0.0
    self._peak_relative_slow_ego = 0.0
    self._peak_lead_accel = 0.0
    self._peak_planner_accel = 0.0
    self._peak_output_accel = 0.0
    self._peak_ego_accel = 0.0
    self._max_gap_growth = 0.0
    self._rapid_gap_growth = False

  @staticmethod
  def _snapshot(kind: str, sample: RollingLeadSample) -> RollingLeadOnsetSnapshot:
    return RollingLeadOnsetSnapshot(
      kind=kind,
      t=sample.t,
      d_rel=sample.d_rel,
      v_lead=sample.v_lead,
      v_lead_k=sample.v_lead_k,
      a_lead_k=sample.a_lead_k,
      v_ego=sample.v_ego,
      a_ego=sample.a_ego,
      a_target=sample.a_target,
      output_accel=sample.output_accel,
      should_stop=sample.should_stop,
      gas_pressed=sample.gas_pressed,
      brake_pressed=sample.brake_pressed,
    )

  @staticmethod
  def _lead_valid(sample: RollingLeadSample) -> bool:
    return sample.lead_present and sample.lead_valid and sample.radar_valid

  @staticmethod
  def _required_valid(sample: RollingLeadSample) -> bool:
    return sample.plan_valid and sample.output_valid

  @staticmethod
  def _candidate_conditions(sample: RollingLeadSample) -> bool:
    return (
      sample.long_active
      and MIN_EGO_SPEED_MPS <= sample.v_ego <= MAX_EGO_SPEED_MPS
      and not sample.gas_pressed
      and not sample.brake_pressed
      and RollingLeadResponseDetector._lead_valid(sample)
      and RollingLeadResponseDetector._required_valid(sample)
      and MIN_LEAD_DISTANCE_M <= sample.d_rel <= MAX_LEAD_DISTANCE_M
    )

  def _clear_track(self, *, radar_discontinuity: bool = False) -> None:
    self._history.clear()
    self._track_id = None
    self._track_start = None
    self._last_lead_seen = None
    self._last_radar_t = None
    self._last_d_rel = None
    self._discontinuity_blocked = self._discontinuity_blocked or radar_discontinuity
    self._discontinuity_settled_since = None
    self._clear_episode()

  def _clear_episode(self) -> None:
    self._phase = "tracking"
    self._commit_t = None
    self._baseline = None
    self._commit_snapshot = None
    self._condition_starts.clear()
    self._condition_snapshots.clear()
    self._onsets.clear()
    self._onset_snapshots.clear()

  def _lock_episode(self, now: float) -> None:
    self._phase = "locked"
    self._cooldown_until = now + EPISODE_COOLDOWN_S
    self._settled_since = None
    self._commit_t = None
    self._baseline = None
    self._condition_starts.clear()
    self._condition_snapshots.clear()
    self._onsets.clear()
    self._onset_snapshots.clear()

  def _append(self, sample: RollingLeadSample) -> bool:
    if self._last_sample_t is not None and sample.t <= self._last_sample_t:
      return False
    self._last_sample_t = sample.t
    self._history.append(sample)
    while self._history and sample.t - self._history[0].t > HISTORY_S:
      self._history.popleft()
    return True

  def _track_sample(self, sample: RollingLeadSample) -> bool:
    """Update exact-track continuity; return False on a hard discontinuity."""
    if not self._lead_valid(sample):
      if self._last_lead_seen is None or sample.t - self._last_lead_seen > LEAD_LOSS_GRACE_S:
        self._clear_track()
        return False
      return True

    if self._track_id is not None and sample.radar_track_id != self._track_id:
      self._clear_track(radar_discontinuity=True)
      self._track_id = sample.radar_track_id
      self._track_start = sample.t
      self._last_lead_seen = sample.t
      self._last_radar_t = sample.t
      self._last_d_rel = sample.d_rel
      return False

    if (
      self._last_radar_t is not None
      and self._last_d_rel is not None
      and sample.t - self._last_radar_t <= LEAD_LOSS_GRACE_S
      and abs(sample.d_rel - self._last_d_rel) > RADAR_DISTANCE_JUMP_M
    ):
      self._clear_track(radar_discontinuity=True)
      self._track_id = sample.radar_track_id
      self._track_start = sample.t
      self._last_lead_seen = sample.t
      self._last_radar_t = sample.t
      self._last_d_rel = sample.d_rel
      return False

    if self._track_id is None:
      self._track_id = sample.radar_track_id
      self._track_start = sample.t
    self._last_lead_seen = sample.t
    self._last_radar_t = sample.t
    self._last_d_rel = sample.d_rel
    return True

  def _filtered_lead_accels(self) -> list[tuple[RollingLeadSample, float]]:
    result: list[tuple[RollingLeadSample, float]] = []
    samples = [sample for sample in self._history if self._lead_valid(sample)]
    left = 0
    for right, sample in enumerate(samples):
      while left < right and sample.t - samples[left].t > LEAD_ACCEL_FILTER_S:
        left += 1
      result.append((sample, fmean(item.a_lead_k for item in samples[left:right + 1])))
    return result

  def _commit_evidence(self, sample: RollingLeadSample) -> tuple[float, _Baseline] | None:
    if self._track_start is None or sample.t - self._track_start < TRACK_CONFIRM_S:
      return None

    history = [item for item in self._history if self._lead_valid(item)]
    rise_window = [item for item in history if sample.t - LEAD_SPEED_RISE_WINDOW_S <= item.t <= sample.t]
    if not rise_window:
      return None
    oldest_speed = rise_window[0].v_lead_k
    speed_rise = sample.v_lead_k - oldest_speed

    accel_start: float | None = None
    current_start: float | None = None
    for item, filtered in self._filtered_lead_accels():
      if item.t < sample.t - LEAD_SPEED_RISE_WINDOW_S:
        continue
      if filtered > LEAD_ACCEL_THRESHOLD_MPS2:
        current_start = item.t if current_start is None else current_start
        if item.t - current_start + 1e-9 >= LEAD_ACCEL_CONFIRM_S:
          accel_start = current_start
      else:
        current_start = None

    speed_onset: float | None = None
    if speed_rise >= LEAD_SPEED_RISE_MPS:
      speed_onset = next((
        item.t for item in rise_window
        if item.v_lead_k - oldest_speed >= MIN_COMMIT_SPEED_RISE_MPS
      ), sample.t)

    if speed_onset is None and accel_start is None:
      return None

    tentative_onset = accel_start if speed_onset is None else speed_onset
    if accel_start is not None:
      tentative_onset = min(tentative_onset, accel_start)

    baseline_samples = [
      item for item in history
      if tentative_onset - BASELINE_WINDOW_S <= item.t <= tentative_onset
    ]
    if len(baseline_samples) < 2 or baseline_samples[-1].t - baseline_samples[0].t < BASELINE_WINDOW_S * 0.65:
      return None
    baseline = _Baseline(
      lead_speed=_mean(baseline_samples, "v_lead_k"),
      ego_speed=_mean(baseline_samples, "v_ego"),
      gap=_mean(baseline_samples, "d_rel"),
      lead_accel=_mean(baseline_samples, "a_lead_k"),
      planner_accel=_mean(baseline_samples, "a_target"),
      output_accel=_mean(baseline_samples, "output_accel"),
      ego_accel=_mean(baseline_samples, "a_ego"),
    )
    relative_speed = sample.v_lead_k - sample.v_ego
    gap_growth = sample.d_rel - baseline.gap
    if relative_speed < LEAD_RELATIVE_SPEED_MPS or gap_growth < MIN_COMMIT_GAP_GROWTH_M:
      return None
    return tentative_onset, baseline

  def _set_onset(self, name: str, condition: bool, sample: RollingLeadSample) -> None:
    if name in self._onsets:
      return
    if not condition:
      self._condition_starts.pop(name, None)
      self._condition_snapshots.pop(name, None)
      return
    start = self._condition_starts.setdefault(name, sample.t)
    self._condition_snapshots.setdefault(name, self._snapshot(name, sample))
    if sample.t - start + 1e-9 >= RESPONSE_CONFIRM_S:
      self._onsets[name] = start
      self._onset_snapshots[name] = self._condition_snapshots[name]

  def _update_responses(self, sample: RollingLeadSample) -> None:
    assert self._baseline is not None
    self._set_onset(
      "plan",
      sample.plan_valid and (
        sample.a_target - self._baseline.planner_accel >= PLANNER_BASELINE_RISE_MPS2
        or sample.a_target >= RESPONSE_ABSOLUTE_ACCEL_MPS2
      ),
      sample,
    )
    self._set_onset(
      "command",
      sample.output_valid and (
        sample.output_accel - self._baseline.output_accel >= OUTPUT_BASELINE_RISE_MPS2
        or sample.output_accel >= RESPONSE_ABSOLUTE_ACCEL_MPS2
      ),
      sample,
    )
    self._set_onset(
      "ego",
      sample.a_ego - self._baseline.ego_accel >= EGO_BASELINE_RISE_MPS2
      or sample.a_ego >= RESPONSE_ABSOLUTE_ACCEL_MPS2,
      sample,
    )

  def _start_episode(self, commit_t: float, baseline: _Baseline) -> None:
    history = [item for item in self._history if self._lead_valid(item)]
    commit_sample = min(history, key=lambda item: abs(item.t - commit_t))
    self._phase = "committed"
    self._commit_t = commit_t
    self._baseline = baseline
    self._commit_snapshot = self._snapshot("leadCommit", commit_sample)
    self._peak_lead_speed = baseline.lead_speed
    self._peak_ego_speed = baseline.ego_speed
    self._peak_relative_speed = 0.0
    self._peak_relative_slow_ego = 0.0
    self._peak_lead_accel = baseline.lead_accel
    self._peak_planner_accel = baseline.planner_accel
    self._peak_output_accel = baseline.output_accel
    self._peak_ego_accel = baseline.ego_accel
    self._max_gap_growth = 0.0
    self._rapid_gap_growth = False
    for item in history:
      if item.t >= commit_t:
        self._observe_committed(item)

  def _observe_committed(self, sample: RollingLeadSample) -> None:
    assert self._baseline is not None
    assert self._commit_t is not None
    self._update_responses(sample)
    relative_speed = sample.v_lead_k - sample.v_ego
    gap_growth = sample.d_rel - self._baseline.gap
    self._peak_lead_speed = max(self._peak_lead_speed, sample.v_lead_k)
    self._peak_ego_speed = max(self._peak_ego_speed, sample.v_ego)
    self._peak_relative_speed = max(self._peak_relative_speed, relative_speed)
    if sample.a_ego < LOW_EGO_ACCEL_MPS2:
      self._peak_relative_slow_ego = max(self._peak_relative_slow_ego, relative_speed)
    self._peak_lead_accel = max(self._peak_lead_accel, sample.a_lead_k)
    self._peak_planner_accel = max(self._peak_planner_accel, sample.a_target)
    self._peak_output_accel = max(self._peak_output_accel, sample.output_accel)
    self._peak_ego_accel = max(self._peak_ego_accel, sample.a_ego)
    self._max_gap_growth = max(self._max_gap_growth, gap_growth)
    if sample.t - self._commit_t <= RAPID_GAP_GROWTH_WINDOW_S and gap_growth >= RAPID_GAP_GROWTH_M:
      self._rapid_gap_growth = True

  def _latency(self, name: str) -> float | None:
    if self._commit_t is None or name not in self._onsets:
      return None
    return self._onsets[name] - self._commit_t

  def _bad_response(self) -> bool:
    plan_latency = self._latency("plan")
    ego_latency = self._latency("ego")
    return (
      plan_latency is None
      or plan_latency > PLANNER_BAD_LATENCY_S
      or ego_latency is None
      or ego_latency > EGO_BAD_LATENCY_S
      or self._peak_relative_slow_ego > HIGH_RELATIVE_SPEED_MPS
      or self._rapid_gap_growth
    )

  def _build_event(self, sample: RollingLeadSample) -> RollingLeadResponseEvent:
    assert self._commit_t is not None
    assert self._baseline is not None
    plan_t = self._onsets.get("plan")
    command_t = self._onsets.get("command")
    ego_t = self._onsets.get("ego")
    plan_latency = self._latency("plan")
    command_latency = self._latency("command")
    ego_latency = self._latency("ego")

    if plan_latency is None or plan_latency > PLANNER_BAD_LATENCY_S:
      attribution = "planner"
      attribution_detail = "planner response was missing or late after lead commitment"
    elif command_t is None or plan_t is None or command_t - plan_t > CONTROLLER_BAD_FOLLOW_S:
      attribution = "controller"
      attribution_detail = "applied output followed a timely planner response too slowly"
    elif ego_t is None or ego_t - command_t > VEHICLE_BAD_FOLLOW_S:
      attribution = "vehicle"
      attribution_detail = "measured ego response followed a timely command too slowly"
    else:
      attribution = "vehicle"
      attribution_detail = "conservative downstream fallback: gap growth persisted despite nominal response-onset timing"
    event_type = f"late_rolling_lead_response_{attribution}"

    measured_latency = {
      "planner": plan_latency,
      "controller": None if command_t is None or plan_t is None else command_t - plan_t,
      "vehicle": None if ego_t is None or command_t is None else ego_t - command_t,
    }[attribution]
    latency_text = "unresolved" if measured_latency is None else f"{measured_latency:.2f} s"
    onsets = [self._commit_snapshot] if self._commit_snapshot is not None else []
    onsets.extend(self._onset_snapshots[name] for name in ("plan", "command", "ego") if name in self._onset_snapshots)
    severity = 3 if self._rapid_gap_growth or self._peak_relative_speed >= 4.5 else 2
    return RollingLeadResponseEvent(
      event_type=event_type,
      attribution_detail=attribution_detail,
      detail=(
        f"Late rolling lead response - {attribution} ({latency_text})"
        if "fallback" not in attribution_detail
        else "Rolling lead response remained insufficient after nominal response onsets"
      ),
      severity=severity,
      confidence=0.95,
      occurred_mono_time=self._commit_t,
      detected_mono_time=sample.t,
      lead_commit_mono_time=self._commit_t,
      planner_response_mono_time=plan_t,
      controller_response_mono_time=command_t,
      ego_response_mono_time=ego_t,
      lead_to_plan_s=plan_latency,
      lead_to_command_s=command_latency,
      lead_to_ego_s=ego_latency,
      baseline_lead_speed=self._baseline.lead_speed,
      peak_lead_speed=self._peak_lead_speed,
      baseline_ego_speed=self._baseline.ego_speed,
      peak_ego_speed=self._peak_ego_speed,
      peak_relative_speed=self._peak_relative_speed,
      baseline_gap=self._baseline.gap,
      final_gap=sample.d_rel,
      max_gap_growth=self._max_gap_growth,
      baseline_lead_accel=self._baseline.lead_accel,
      peak_lead_accel=self._peak_lead_accel,
      baseline_planner_accel=self._baseline.planner_accel,
      peak_planner_accel=self._peak_planner_accel,
      baseline_output_accel=self._baseline.output_accel,
      peak_output_accel=self._peak_output_accel,
      baseline_ego_accel=self._baseline.ego_accel,
      peak_ego_accel=self._peak_ego_accel,
      radar_track_id=sample.radar_track_id,
      radar_discontinuity=False,
      driver_confounded=False,
      onsets=tuple(onsets),
      episode_start_mono_time=self._commit_t,
    )

  def _update_lock(self, sample: RollingLeadSample) -> None:
    recent = [
      item for item in self._history
      if self._lead_valid(item) and sample.t - item.t <= BASELINE_WINDOW_S
    ]
    gap_change = max((item.d_rel for item in recent), default=sample.d_rel) - min(
      (item.d_rel for item in recent), default=sample.d_rel,
    )
    settled = abs(sample.v_lead_k - sample.v_ego) <= SETTLED_RELATIVE_SPEED_MPS and gap_change <= SETTLED_GAP_CHANGE_M
    if settled:
      self._settled_since = sample.t if self._settled_since is None else self._settled_since
    else:
      self._settled_since = None
    if (
      sample.t >= self._cooldown_until
      and self._settled_since is not None
      and sample.t - self._settled_since >= SETTLED_CONFIRM_S
    ):
      self._clear_track()

  def _update_discontinuity_block(self, sample: RollingLeadSample) -> bool:
    if not self._discontinuity_blocked:
      return False
    settled = (
      abs(sample.v_lead_k - sample.v_ego) <= SETTLED_RELATIVE_SPEED_MPS
      and abs(sample.a_lead_k) < LEAD_ACCEL_THRESHOLD_MPS2 * 0.5
    )
    if settled:
      self._discontinuity_settled_since = (
        sample.t if self._discontinuity_settled_since is None else self._discontinuity_settled_since
      )
    else:
      self._discontinuity_settled_since = None
    if (
      self._discontinuity_settled_since is not None
      and sample.t - self._discontinuity_settled_since >= SETTLED_CONFIRM_S
    ):
      self._discontinuity_blocked = False
      self._discontinuity_settled_since = None
      self._history.clear()
      self._track_start = sample.t
      return False
    return True

  def update(self, sample: RollingLeadSample) -> RollingLeadResponseEvent | None:
    if not self._append(sample):
      return None

    if self._phase == "locked":
      self._update_lock(sample)
      return None

    if sample.gas_pressed or sample.brake_pressed or not sample.long_active or sample.v_ego < CANCEL_EGO_SPEED_MPS:
      self._clear_track()
      return None
    if not self._required_valid(sample):
      self._clear_track()
      return None
    if not self._track_sample(sample):
      return None
    # A brief radar dropout keeps continuity alive, but held/invalid lead values
    # must not advance commitment, response, or finalization evidence.
    if not self._lead_valid(sample):
      return None

    if self._update_discontinuity_block(sample):
      return None

    if self._phase == "tracking":
      if not self._candidate_conditions(sample):
        return None
      evidence = self._commit_evidence(sample)
      if evidence is None:
        return None
      self._start_episode(*evidence)

    if self._phase != "committed" or self._commit_t is None:
      return None

    # The current sample was already replayed from history when commitment began.
    if not self._onset_snapshots or all(onset.t != sample.t for onset in self._onset_snapshots.values()):
      self._observe_committed(sample)

    elapsed = sample.t - self._commit_t
    if elapsed < EVALUATION_WINDOW_S:
      return None

    should_emit = self._max_gap_growth >= MIN_EVENT_GAP_GROWTH_M and self._bad_response()
    if not should_emit:
      self._lock_episode(sample.t)
      return None
    event = self._build_event(sample)
    self._lock_episode(sample.t)
    return event
