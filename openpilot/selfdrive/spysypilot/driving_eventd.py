#!/usr/bin/env python3
"""Observer-only runtime for manual, lateral, and longitudinal driving events."""
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.spysypilot.lat_event_detector import (
  DETECTOR_VERSION as LAT_DETECTOR_VERSION,
  LateralDetection,
  LateralEventDetector,
  LateralSample,
  RoadBumpClassifier,
  SteeringRateFilter,
)
from openpilot.selfdrive.spysypilot.long_event_detector import (
  DETECTOR_VERSION as LONG_DETECTOR_VERSION,
  LaunchSample,
  LeadLaunchDetector,
  LongEvent,
)


EVENT_VERSION = 2
GROUP_WINDOW_NS = 2_500_000_000
CONTEXT_BEFORE = 2
CONTEXT_AFTER = 1
ACK_RETRY_INTERVAL_NS = 1_000_000_000


@dataclass(frozen=True)
class EventCandidate:
  occurred_mono_time: int
  domain: str
  source: str
  event_type: str
  detector: str
  detector_version: int
  severity: str
  confidence: float
  reason: str
  attribution: str = "unknown"
  driver_confounded: bool = False
  road_confounded: bool = False
  payload: Any | None = None
  detector_evidence: Any | None = None
  detected_mono_time: int = 0
  episode_start_mono_time: int = 0
  analysis_window_before_s: float = 0.0
  analysis_window_after_s: float = 0.0
  episode_key: str = ""


@dataclass(frozen=True)
class AcceptedEvent:
  event_id: str
  group_id: str
  candidate: EventCandidate


@dataclass
class PendingEvent:
  event: AcceptedEvent
  last_sent_ns: int
  attempts: int = 1


class EventRecorder:
  """Assign stable IDs and short-lived physical-episode group IDs."""

  def __init__(self, id_factory: Callable[[], str] | None = None, group_window_ns: int = GROUP_WINDOW_NS):
    self.id_factory = id_factory or (lambda: uuid.uuid4().hex)
    self.group_window_ns = group_window_ns
    self._group_id = ""
    self._group_last_time = -group_window_ns
    self._episode_groups: dict[str, tuple[str, int]] = {}

  def accept(self, candidate: EventCandidate) -> AcceptedEvent:
    if candidate.episode_key:
      episode_group = self._episode_groups.get(candidate.episode_key)
      if episode_group is None:
        if not self._group_id or candidate.occurred_mono_time - self._group_last_time > self.group_window_ns:
          self._group_id = self.id_factory()
        self._episode_groups[candidate.episode_key] = (self._group_id, candidate.occurred_mono_time)
      else:
        self._group_id = episode_group[0]
        self._episode_groups[candidate.episode_key] = (self._group_id, candidate.occurred_mono_time)
    elif not self._group_id or candidate.occurred_mono_time - self._group_last_time > self.group_window_ns:
      self._group_id = self.id_factory()
    self._group_last_time = max(self._group_last_time, candidate.occurred_mono_time)
    self._episode_groups = {
      key: value for key, value in self._episode_groups.items()
      if candidate.occurred_mono_time - value[1] <= 60_000_000_000
    }
    return AcceptedEvent(self.id_factory(), self._group_id, candidate)


def manual_candidate(occurred_mono_time: int) -> EventCandidate:
  return EventCandidate(
    occurred_mono_time=occurred_mono_time,
    domain="manual",
    source="user",
    event_type="manual.general",
    detector="bookmarkButton",
    detector_version=1,
    severity="info",
    confidence=1.0,
    reason="user marked a driving event",
    detected_mono_time=occurred_mono_time,
    episode_start_mono_time=occurred_mono_time,
    analysis_window_before_s=15.0,
    analysis_window_after_s=5.0,
  )


def lateral_candidate(sample: LateralSample, detection: LateralDetection) -> EventCandidate:
  attribution = "actuator" if detection.event_type == "torqueAuthority" else "controller"
  evidence = detection.evidence
  driver_confounded = evidence.driver_confounded_any if evidence is not None else sample.driver_confounded
  road_confounded = evidence.road_confounded_any if evidence is not None else sample.road_confounded
  if driver_confounded or road_confounded:
    attribution = "mixed"
  occurred_mono_time = round(sample.mono_time * 1e9)
  return EventCandidate(
    occurred_mono_time=occurred_mono_time,
    domain="lateral",
    source="automatic",
    event_type=f"lat.{detection.event_type}",
    detector="blatLateralEventDetector",
    detector_version=LAT_DETECTOR_VERSION,
    severity=detection.severity,
    confidence=detection.confidence,
    reason=detection.reason,
    attribution=attribution,
    driver_confounded=driver_confounded,
    road_confounded=road_confounded,
    payload=sample,
    detector_evidence=evidence,
    detected_mono_time=occurred_mono_time,
    episode_start_mono_time=round(evidence.episode_start_mono_time * 1e9) if evidence is not None else occurred_mono_time,
    analysis_window_before_s=evidence.analysis_window_before_s if evidence is not None else 2.0,
    analysis_window_after_s=evidence.analysis_window_after_s if evidence is not None else 2.0,
    episode_key=evidence.episode_key if evidence is not None else "",
  )


def longitudinal_candidate(sample: LaunchSample, event: LongEvent) -> EventCandidate:
  names = {
    "late_lead_launch_planner": ("long.lateLeadLaunchPlanner", "planner"),
    "late_lead_launch_controller": ("long.lateLeadLaunchController", "controller"),
    "late_lead_launch_vehicle": ("long.lateLeadLaunchVehicle", "mixed"),
    "lead_launch_stall": ("long.leadLaunchStall", "unknown"),
  }
  event_type, attribution = names[event.event_type]
  severity = "critical" if event.severity >= 3 else ("warning" if event.severity else "info")
  occurred_mono_time = round(sample.t * 1e9)
  return EventCandidate(
    occurred_mono_time=occurred_mono_time,
    domain="longitudinal",
    source="automatic",
    event_type=event_type,
    detector="leadLaunchDetector",
    detector_version=LONG_DETECTOR_VERSION,
    severity=severity,
    confidence=event.confidence,
    reason=event.detail,
    attribution=attribution,
    payload=event,
    detected_mono_time=occurred_mono_time,
    episode_start_mono_time=round(event.episode_start_mono_time * 1e9) if event.episode_start_mono_time else occurred_mono_time,
    analysis_window_before_s=event.analysis_window_before_s,
    analysis_window_after_s=event.analysis_window_after_s,
    episode_key=f"long:{round(event.episode_start_mono_time * 1e9)}" if event.episode_start_mono_time else "",
  )


class DrivingEventPlatform:
  """Runs detectors independently so one domain cannot disable another."""

  def __init__(self, recorder: EventRecorder | None = None,
               lateral_detector: LateralEventDetector | None = None,
               longitudinal_detector: LeadLaunchDetector | None = None,
               on_error: Callable[[str], None] | None = None):
    self.recorder = recorder or EventRecorder()
    self.lateral_detector = lateral_detector or LateralEventDetector()
    self.longitudinal_detector = longitudinal_detector or LeadLaunchDetector()
    self.on_error = on_error or (lambda domain: cloudlog.exception(f"driving_eventd: {domain} detector failed"))

  def update(self, lateral_sample: LateralSample | None = None, longitudinal_sample: LaunchSample | None = None,
             manual_pressed: bool = False, manual_time_ns: int | None = None) -> list[AcceptedEvent]:
    candidates: list[EventCandidate] = []
    if lateral_sample is not None:
      try:
        detection = self.lateral_detector.update(lateral_sample)
        if detection is not None:
          candidates.append(lateral_candidate(lateral_sample, detection))
      except Exception:
        self.on_error("lateral")

    if longitudinal_sample is not None:
      try:
        event = self.longitudinal_detector.update(longitudinal_sample)
        if event is not None:
          candidates.append(longitudinal_candidate(longitudinal_sample, event))
      except Exception:
        self.on_error("longitudinal")

    if manual_pressed:
      candidates.append(manual_candidate(manual_time_ns if manual_time_ns is not None else time.monotonic_ns()))

    candidates.sort(key=lambda candidate: candidate.occurred_mono_time)
    return [self.recorder.accept(candidate) for candidate in candidates]


def _param_text(params: Params, key: str) -> str:
  value = params.get(key)
  if value is None:
    return ""
  return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def build_message(event: AcceptedEvent, git_commit: str = "", git_branch: str = ""):
  msg = messaging.new_message("drivingEvent", valid=True)
  out = msg.drivingEvent
  candidate = event.candidate
  out.version = EVENT_VERSION
  out.eventId = event.event_id
  out.groupId = event.group_id
  out.occurredMonoTime = candidate.occurred_mono_time
  out.domain = candidate.domain
  out.source = candidate.source
  out.eventType = candidate.event_type
  out.detector = candidate.detector
  out.detectorVersion = candidate.detector_version
  out.severity = candidate.severity
  out.confidence = candidate.confidence
  out.reason = candidate.reason
  out.attribution = candidate.attribution
  out.driverConfounded = candidate.driver_confounded
  out.roadConfounded = candidate.road_confounded
  out.requestedContextBefore = CONTEXT_BEFORE
  out.requestedContextAfter = CONTEXT_AFTER
  out.gitCommit = git_commit
  out.gitBranch = git_branch
  out.detectedMonoTime = candidate.detected_mono_time or candidate.occurred_mono_time
  out.episodeStartMonoTime = candidate.episode_start_mono_time or candidate.occurred_mono_time
  out.analysisWindowBeforeS = candidate.analysis_window_before_s
  out.analysisWindowAfterS = candidate.analysis_window_after_s
  out.episodeKey = candidate.episode_key

  if candidate.domain == "lateral":
    sample: LateralSample = candidate.payload
    evidence = candidate.detector_evidence
    payload = out.payload.init("lateral")
    payload.controllerVersion = sample.controller_version
    payload.referenceVersion = sample.reference_version
    payload.vEgo = sample.v_ego
    payload.steeringAngleDeg = sample.steering_angle_deg
    payload.steeringRateDeg = sample.steering_rate_deg
    payload.desiredLateralAccel = sample.desired_lateral_accel
    payload.actualLateralAccel = sample.actual_lateral_accel
    payload.requestTorque = sample.request_torque
    payload.appliedTorque = sample.applied_torque
    payload.referenceTargetTorque = sample.reference_target_torque
    payload.referenceRate = sample.reference_rate
    payload.referenceUnwindScale = sample.reference_unwind_scale
    payload.referenceSustainedUnwindScale = sample.reference_sustained_unwind_scale
    payload.unwindEffectivePhase = sample.unwind_effective_phase
    payload.unwindOverspeed = sample.unwind_overspeed
    payload.unwindSameEpisode = sample.unwind_same_episode
    payload.appliedTargetGap = sample.applied_target_gap
    payload.pTerm = sample.p_term
    payload.driverTorque = sample.driver_torque
    payload.steeringPressed = sample.steering_pressed
    payload.steeringTorqueEps = sample.steering_torque_eps
    payload.dampingApplied = sample.damping_applied
    payload.dampingState = sample.damping_state
    payload.triggerDriverConfounded = sample.driver_confounded
    payload.triggerRoadConfounded = sample.road_confounded
    if evidence is not None:
      payload.driverConfoundedFraction = evidence.driver_confounded_fraction
      payload.maxAbsDriverTorque = evidence.max_abs_driver_torque
      payload.steeringPressedAny = evidence.steering_pressed_any
      payload.roadConfoundedFraction = evidence.road_confounded_fraction
      payload.driverConfoundReason = evidence.driver_confound_reason
      payload.evidenceStartMonoTime = round(evidence.evidence_start_mono_time * 1e9)
      payload.evidenceEndMonoTime = round(evidence.evidence_end_mono_time * 1e9)
      payload.stallReleaseCount = evidence.stall_release_count
      payload.releaseOffsetsS = list(evidence.release_offsets_s)
      payload.stallDurationsS = list(evidence.stall_durations_s)
      payload.releasePeakRatesDeg = list(evidence.release_peak_rates_deg)
      payload.stallEpisodePhase = evidence.stall_episode_phase
      payload.lateUnwindDurationS = evidence.late_unwind_duration_s
      payload.previousUnwindEffectivePhase = evidence.previous_unwind_effective_phase
      payload.previousUnwindSameEpisode = evidence.previous_unwind_same_episode
      payload.trackingInactiveTimeS = evidence.tracking_inactive_time_s
  elif candidate.domain == "longitudinal":
    long_event: LongEvent = candidate.payload
    payload = out.payload.init("leadLaunch")
    payload.forecastToLeadS = long_event.forecast_to_lead_s
    payload.planToLeadS = long_event.plan_to_lead_s
    payload.commandToLeadS = long_event.command_to_lead_s
    payload.leadToEgoS = long_event.lead_to_ego_s
    payload.commandToEgoS = long_event.command_to_ego_s
    payload.radarDiscontinuity = long_event.radar_discontinuity
    payload.radarConfidence = long_event.confidence
    payload.attributionDetail = long_event.attribution_detail
    onsets = payload.init("onsets", len(long_event.onsets))
    for index, onset in enumerate(long_event.onsets):
      onsets[index].kind = onset.kind
      onsets[index].monoTime = round(onset.t * 1e9)
      onsets[index].dRel = onset.d_rel
      onsets[index].vLead = onset.v_lead
      onsets[index].vEgo = onset.v_ego
      onsets[index].aEgo = onset.a_ego
      onsets[index].outputAccel = onset.output_accel
      onsets[index].brakePressed = onset.brake_pressed
      onsets[index].brakeHoldActive = onset.brake_hold_active
  else:
    out.payload.none = None
  return msg


class EventSubmitter:
  """Retain accepted events until loggerd acknowledges their stable event IDs."""

  def __init__(self, pm: messaging.PubMaster, git_commit: str = "", git_branch: str = "",
               retry_interval_ns: int = ACK_RETRY_INTERVAL_NS):
    self.pm = pm
    self.git_commit = git_commit
    self.git_branch = git_branch
    self.retry_interval_ns = retry_interval_ns
    self.pending: dict[str, PendingEvent] = {}

  def _send(self, pending: PendingEvent, now_ns: int) -> bool:
    pending.last_sent_ns = now_ns
    try:
      self.pm.send("drivingEvent", build_message(pending.event, self.git_commit, self.git_branch))
      return True
    except Exception:
      cloudlog.exception(f"driving_eventd: failed to submit event {pending.event.event_id}")
      return False

  def submit(self, event: AcceptedEvent, now_ns: int | None = None) -> None:
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    pending = PendingEvent(event, now_ns)
    self.pending[event.event_id] = pending
    self._send(pending, now_ns)

  def acknowledge(self, event_id: str) -> bool:
    return self.pending.pop(event_id, None) is not None

  def retry_due(self, now_ns: int | None = None) -> list[str]:
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    retried: list[str] = []
    for event_id, pending in self.pending.items():
      if now_ns - pending.last_sent_ns >= self.retry_interval_ns:
        pending.attempts += 1
        self._send(pending, now_ns)
        retried.append(event_id)
    return retried


def longitudinal_sample(sm: messaging.SubMaster) -> LaunchSample:
  lead = sm["radarState"].leadOne
  leads_v3 = sm["modelV2"].leadsV3
  forecast_valid = sm.valid["modelV2"] and len(leads_v3) > 0 and leads_v3[0].prob > 0.5 and len(leads_v3[0].v) > 1
  predicted_lead_v_2s = 0.0
  if forecast_valid:
    predicted_lead_v_2s = float(lead.vLead) + float(leads_v3[0].v[1]) - float(leads_v3[0].v[0])

  now = sm.logMonoTime["carState"] * 1e-9 if sm.logMonoTime["carState"] else time.monotonic()
  return LaunchSample(
    t=now,
    active=bool(sm.valid["carState"] and sm.valid["carControl"] and sm["carControl"].longActive),
    standstill=bool(sm["carState"].standstill),
    v_ego=float(sm["carState"].vEgo),
    lead_present=bool(lead.status),
    radar_valid=bool(sm.valid["radarState"]),
    d_rel=float(lead.dRel),
    v_lead=float(lead.vLead),
    v_lead_k=float(lead.vLeadK),
    radar_track_id=int(lead.radarTrackId),
    plan_valid=bool(sm.valid["longitudinalPlan"]),
    plan_should_stop=bool(sm["longitudinalPlan"].shouldStop),
    output_valid=bool(sm.valid["carOutput"]),
    output_accel=float(sm["carOutput"].actuatorsOutput.accel),
    forecast_valid=forecast_valid,
    predicted_lead_v_2s=predicted_lead_v_2s,
    a_ego=float(getattr(sm["carState"], "aEgo", 0.0)),
    brake_pressed=bool(getattr(sm["carState"], "brakePressed", False)),
    brake_hold_active=bool(getattr(sm["carState"], "brakeHoldActive", False)),
  )


def lateral_sample(sm: messaging.SubMaster, rate_filter: SteeringRateFilter,
                   bump_classifier: RoadBumpClassifier) -> LateralSample:
  controls_state = sm["controlsState"]
  torque_state = controls_state.lateralControlState.torqueState if controls_state.lateralControlState.which() == "torqueState" else None
  car_state = sm["carState"]
  now = sm.logMonoTime["controlsState"] * 1e-9 if sm.logMonoTime["controlsState"] else time.monotonic()
  angle = float(car_state.steeringAngleDeg)
  road_confounded = sm.valid["livePose"] and bump_classifier.update(float(sm["livePose"].accelerationDevice.z), now)
  damping_applied = float(torque_state.d) if torque_state is not None else 0.0
  damping_state = (
    "blocked" if torque_state is not None and getattr(torque_state, "dampingTurnInBlocked", False)
    else ("applied" if abs(damping_applied) > 1e-6 else "inactive")
  )
  return LateralSample(
    mono_time=now,
    active=bool(torque_state is not None and torque_state.active and sm["carControl"].latActive),
    v_ego=float(car_state.vEgo),
    steering_angle_deg=angle,
    steering_rate_deg=rate_filter.update(angle, now),
    driver_torque=float(car_state.steeringTorque),
    steering_pressed=bool(car_state.steeringPressed),
    desired_lateral_accel=float(torque_state.desiredLateralAccel) if torque_state is not None else 0.0,
    actual_lateral_accel=float(torque_state.actualLateralAccel) if torque_state is not None else 0.0,
    request_torque=float(sm["carControl"].actuators.torque),
    applied_torque=float(sm["carOutput"].actuatorsOutput.torque),
    p_term=float(torque_state.p) if torque_state is not None else 0.0,
    steering_torque_eps=float(car_state.steeringTorqueEps),
    damping_applied=damping_applied,
    damping_state=damping_state,
    controller_version=int(torque_state.version) if torque_state is not None else 0,
    reference_version=int(getattr(torque_state, "referenceVersion", 0)) if torque_state is not None else 0,
    reference_rate=float(getattr(torque_state, "referenceRate", 0.0)) if torque_state is not None else 0.0,
    reference_target_torque=float(getattr(torque_state, "referenceReachableTargetTorque", 0.0)) if torque_state is not None else 0.0,
    reference_unwind_scale=float(getattr(torque_state, "referenceUnwindScale", 0.0)) if torque_state is not None else 0.0,
    reference_sustained_unwind_scale=float(getattr(torque_state, "referenceSustainedUnwindScale", 0.0)) if torque_state is not None else 0.0,
    unwind_effective_phase=float(getattr(torque_state, "unwindEffectivePhase", 0.0)) if torque_state is not None else 0.0,
    unwind_overspeed=float(getattr(torque_state, "unwindPhaseOverspeed", 0.0)) if torque_state is not None else 0.0,
    unwind_same_episode=bool(getattr(torque_state, "unwindSameEpisode", False)) if torque_state is not None else False,
    road_confounded=bool(road_confounded),
  )


def main() -> None:
  params = Params()
  git_commit = _param_text(params, "GitCommit")
  git_branch = _param_text(params, "GitBranch")
  pm = messaging.PubMaster(["drivingEvent"])
  ack_sock = messaging.sub_sock("drivingEventRecorded", conflate=False)
  sm = messaging.SubMaster(
    ["bookmarkButton", "carState", "carControl", "carOutput", "controlsState",
     "radarState", "longitudinalPlan", "modelV2", "livePose"],
  )
  platform = DrivingEventPlatform()
  submitter = EventSubmitter(pm, git_commit, git_branch)
  rate_filter = SteeringRateFilter()
  bump_classifier = RoadBumpClassifier()

  if not pm.wait_for_readers_to_update("drivingEvent", timeout=10):
    cloudlog.warning("driving_eventd: loggerd reader not ready; retaining events for retry")

  while True:
    sm.update(100)
    for ack_msg in messaging.drain_sock(ack_sock):
      if ack_msg.valid and ack_msg.which() == "drivingEventRecorded":
        submitter.acknowledge(ack_msg.drivingEventRecorded.eventId)

    controls_updated = bool(sm.updated["controlsState"])
    car_state_updated = bool(sm.updated["carState"])
    bookmark_updated = bool(sm.updated["bookmarkButton"])
    lat_sample = lateral_sample(sm, rate_filter, bump_classifier) if controls_updated else None
    long_sample = longitudinal_sample(sm) if car_state_updated else None
    events = platform.update(
      lat_sample,
      long_sample,
      manual_pressed=bookmark_updated,
      manual_time_ns=sm.logMonoTime["bookmarkButton"] if bookmark_updated else None,
    )
    for event in events:
      submitter.submit(event)
      cloudlog.event("driving_event", event_id=event.event_id, group_id=event.group_id,
                     domain=event.candidate.domain, event_type=event.candidate.event_type)
    for event_id in submitter.retry_due():
      cloudlog.warning("driving_eventd: retrying unacknowledged event", event_id=event_id)


if __name__ == "__main__":
  main()
