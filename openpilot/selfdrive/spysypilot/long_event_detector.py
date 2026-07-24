"""Lightweight on-road longitudinal event detection.

This module deliberately has no messaging or filesystem dependencies. driving_eventd
feeds it the latest planner, radar, controller, and vehicle state and publishes its
results through the shared drivingEvent envelope.
"""
from dataclasses import dataclass


MIN_STANDSTILL_S = 0.5
SIGNAL_CONFIRM_S = 0.15
FORECAST_CONFIRM_S = 0.2
LEAD_MOVING_SPEED = 0.25
EGO_MOVING_SPEED = 0.1
POSITIVE_ACCEL = 0.05
BAD_LAUNCH_LAG_S = 0.25
SEVERE_LAUNCH_LAG_S = 1.5
LAUNCH_STALL_S = 3.0
LEAD_LOSS_GRACE_S = 0.3
RADAR_JUMP_M = 1.5
DETECTOR_VERSION = 2


@dataclass(frozen=True)
class LaunchSample:
  t: float
  active: bool
  standstill: bool
  v_ego: float
  lead_present: bool
  radar_valid: bool
  d_rel: float
  v_lead: float
  v_lead_k: float
  radar_track_id: int
  plan_valid: bool
  plan_should_stop: bool
  output_valid: bool
  output_accel: float
  forecast_valid: bool
  predicted_lead_v_2s: float
  a_ego: float = 0.0
  brake_pressed: bool = False
  brake_hold_active: bool = False


@dataclass(frozen=True)
class OnsetSnapshot:
  kind: str
  t: float
  d_rel: float
  v_lead: float
  v_ego: float
  a_ego: float
  output_accel: float
  brake_pressed: bool
  brake_hold_active: bool


@dataclass(frozen=True)
class LongEvent:
  event_type: str
  detail: str
  severity: int
  confidence: float
  lead_to_ego_s: float = 0.0
  command_to_ego_s: float = 0.0
  plan_to_lead_s: float = 0.0
  command_to_lead_s: float = 0.0
  forecast_to_lead_s: float = 0.0
  radar_discontinuity: bool = False
  onsets: tuple[OnsetSnapshot, ...] = ()
  episode_start_mono_time: float = 0.0
  analysis_window_before_s: float = 5.0
  analysis_window_after_s: float = 3.0
  attribution_detail: str = ""


class LeadLaunchDetector:
  """Detect late launches and attribute the delay to plan, command, or vehicle response."""
  def __init__(self):
    self.reset()

  def reset(self) -> None:
    self._candidate_start: float | None = None
    self._last_lead_seen: float | None = None
    self._condition_starts: dict[str, float] = {}
    self._condition_snapshots: dict[str, OnsetSnapshot] = {}
    self._onsets: dict[str, float] = {}
    self._onset_snapshots: dict[str, OnsetSnapshot] = {}
    self._candidate_snapshot: OnsetSnapshot | None = None
    self._prev_radar_t: float | None = None
    self._prev_d_rel: float | None = None
    self._prev_track_id: int | None = None
    self._radar_discontinuity = False
    self._emitted = False

  @staticmethod
  def _snapshot(kind: str, sample: LaunchSample) -> OnsetSnapshot:
    return OnsetSnapshot(
      kind,
      sample.t,
      sample.d_rel,
      sample.v_lead,
      sample.v_ego,
      sample.a_ego,
      sample.output_accel,
      sample.brake_pressed,
      sample.brake_hold_active,
    )

  def _onset(self, name: str, condition: bool, sample: LaunchSample, confirm_s: float) -> float | None:
    if name in self._onsets:
      return self._onsets[name]
    if not condition:
      self._condition_starts.pop(name, None)
      self._condition_snapshots.pop(name, None)
      return None

    start = self._condition_starts.setdefault(name, sample.t)
    self._condition_snapshots.setdefault(name, self._snapshot(name, sample))
    if sample.t - start + 1e-9 >= confirm_s:
      self._onsets[name] = start
      self._onset_snapshots[name] = self._condition_snapshots[name]
      return start
    return None

  def _event_onsets(self) -> tuple[OnsetSnapshot, ...]:
    snapshots = []
    if self._candidate_snapshot is not None:
      snapshots.append(self._candidate_snapshot)
    snapshots.extend(self._onset_snapshots[name] for name in (
      "forecast", "plan", "command", "lead", "ego", "egoAcceleration"
    ) if name in self._onset_snapshots)
    return tuple(snapshots)

  def _update_radar_quality(self, sample: LaunchSample) -> None:
    if not (sample.lead_present and sample.radar_valid):
      return
    if self._prev_radar_t is not None and self._prev_d_rel is not None:
      dt = sample.t - self._prev_radar_t
      if 0.0 < dt < 0.2 and abs(sample.d_rel - self._prev_d_rel) > RADAR_JUMP_M:
        self._radar_discontinuity = True
    if self._prev_track_id is not None and sample.radar_track_id >= 0 and self._prev_track_id >= 0:
      if sample.radar_track_id != self._prev_track_id:
        self._radar_discontinuity = True
    self._prev_radar_t = sample.t
    self._prev_d_rel = sample.d_rel
    self._prev_track_id = sample.radar_track_id

  @staticmethod
  def _severity(lag_s: float) -> int:
    if lag_s >= SEVERE_LAUNCH_LAG_S:
      return 3
    if lag_s >= 0.75:
      return 2
    return 1

  def _build_late_launch(self, t_lead: float, t_ego: float) -> LongEvent:
    lag = t_ego - t_lead
    t_plan = self._onsets.get("plan")
    t_command = self._onsets.get("command")
    t_forecast = self._onsets.get("forecast")

    if t_plan is None or t_plan - t_lead > BAD_LAUNCH_LAG_S:
      cause = "planner"
      event_type = "late_lead_launch_planner"
    elif t_command is None or t_command - t_lead > BAD_LAUNCH_LAG_S:
      cause = "controller"
      event_type = "late_lead_launch_controller"
    else:
      cause = "downstream/vehicle response"
      event_type = "late_lead_launch_vehicle"

    confidence = 0.55 if self._radar_discontinuity else 0.95
    return LongEvent(
      event_type=event_type,
      detail=f"Late launch +{lag:.1f} s - {cause}",
      severity=self._severity(lag),
      confidence=confidence,
      lead_to_ego_s=lag,
      command_to_ego_s=(t_ego - t_command) if t_command is not None else 0.0,
      plan_to_lead_s=(t_plan - t_lead) if t_plan is not None else 0.0,
      command_to_lead_s=(t_command - t_lead) if t_command is not None else 0.0,
      forecast_to_lead_s=(t_forecast - t_lead) if t_forecast is not None else 0.0,
      radar_discontinuity=self._radar_discontinuity,
      onsets=self._event_onsets(),
      episode_start_mono_time=self._candidate_start if self._candidate_start is not None else t_lead,
      attribution_detail=cause,
    )

  def update(self, sample: LaunchSample) -> LongEvent | None:
    if not sample.active:
      self.reset()
      return None

    lead_valid = sample.lead_present and sample.radar_valid
    if sample.standstill and lead_valid and self._candidate_start is None:
      self._candidate_start = sample.t
      self._candidate_snapshot = self._snapshot("candidate", sample)

    if self._candidate_start is None:
      return None

    if lead_valid:
      self._last_lead_seen = sample.t
      self._update_radar_quality(sample)
    elif self._last_lead_seen is None or sample.t - self._last_lead_seen > LEAD_LOSS_GRACE_S:
      self.reset()
      return None

    armed = sample.t - self._candidate_start >= MIN_STANDSTILL_S
    if not armed:
      return None

    self._onset("forecast", sample.forecast_valid and sample.predicted_lead_v_2s > 0.3,
                sample, FORECAST_CONFIRM_S)
    t_plan = self._onset("plan", sample.plan_valid and not sample.plan_should_stop,
                         sample, SIGNAL_CONFIRM_S)
    t_command = self._onset("command", sample.output_valid and sample.output_accel > POSITIVE_ACCEL,
                            sample, SIGNAL_CONFIRM_S)
    t_lead = self._onset("lead", lead_valid and sample.v_lead > LEAD_MOVING_SPEED,
                         sample, SIGNAL_CONFIRM_S)
    t_ego = self._onset("ego", sample.v_ego > EGO_MOVING_SPEED,
                        sample, SIGNAL_CONFIRM_S)
    self._onset("egoAcceleration", sample.a_ego > POSITIVE_ACCEL, sample, 0.0)

    if self._emitted:
      if t_ego is not None or not sample.standstill:
        self.reset()
      return None

    if t_lead is not None and t_ego is not None:
      lag = t_ego - t_lead
      if lag > BAD_LAUNCH_LAG_S:
        self._emitted = True
        return self._build_late_launch(t_lead, t_ego)
      self.reset()
      return None

    if t_lead is not None and sample.t - t_lead >= LAUNCH_STALL_S:
      self._emitted = True
      confidence = 0.55 if self._radar_discontinuity else 0.95
      return LongEvent(
        event_type="lead_launch_stall",
        detail=f"Launch stalled +{sample.t - t_lead:.1f} s",
        severity=3,
        confidence=confidence,
        lead_to_ego_s=sample.t - t_lead,
        command_to_ego_s=(sample.t - t_command) if t_command is not None else 0.0,
        plan_to_lead_s=(t_plan - t_lead) if t_plan is not None else 0.0,
        command_to_lead_s=(t_command - t_lead) if t_command is not None else 0.0,
        forecast_to_lead_s=(self._onsets["forecast"] - t_lead) if "forecast" in self._onsets else 0.0,
        radar_discontinuity=self._radar_discontinuity,
        onsets=self._event_onsets(),
        episode_start_mono_time=self._candidate_start if self._candidate_start is not None else t_lead,
        attribution_detail="downstream response unresolved",
      )

    # Ego moved before radar measured a departure. That is not a late launch; end this
    # candidate and let a future standstill arm a fresh one.
    if t_ego is not None and t_lead is None:
      self.reset()

    return None
