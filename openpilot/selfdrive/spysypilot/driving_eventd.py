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


EVENT_VERSION = 1
GROUP_WINDOW_NS = 2_500_000_000
CONTEXT_BEFORE = 2
CONTEXT_AFTER = 1


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


@dataclass(frozen=True)
class AcceptedEvent:
  event_id: str
  group_id: str
  candidate: EventCandidate


class EventRecorder:
  """Assign stable IDs and short-lived physical-episode group IDs."""

  def __init__(self, id_factory: Callable[[], str] | None = None, group_window_ns: int = GROUP_WINDOW_NS):
    self.id_factory = id_factory or (lambda: uuid.uuid4().hex)
    self.group_window_ns = group_window_ns
    self._group_id = ""
    self._group_last_time = -group_window_ns

  def accept(self, candidate: EventCandidate) -> AcceptedEvent:
    if not self._group_id or candidate.occurred_mono_time - self._group_last_time > self.group_window_ns:
      self._group_id = self.id_factory()
    self._group_last_time = max(self._group_last_time, candidate.occurred_mono_time)
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
  )


def lateral_candidate(sample: LateralSample, detection: LateralDetection) -> EventCandidate:
  attribution = "actuator" if detection.event_type == "torqueAuthority" else "controller"
  return EventCandidate(
    occurred_mono_time=round(sample.mono_time * 1e9),
    domain="lateral",
    source="automatic",
    event_type=f"lat.{detection.event_type}",
    detector="blatLateralEventDetector",
    detector_version=LAT_DETECTOR_VERSION,
    severity=detection.severity,
    confidence=detection.confidence,
    reason=detection.reason,
    attribution=attribution,
    driver_confounded=sample.driver_confounded,
    road_confounded=sample.road_confounded,
    payload=sample,
  )


def longitudinal_candidate(sample: LaunchSample, event: LongEvent) -> EventCandidate:
  names = {
    "late_lead_launch_planner": ("long.lateLeadLaunchPlanner", "planner"),
    "late_lead_launch_controller": ("long.lateLeadLaunchController", "controller"),
    "late_lead_launch_vehicle": ("long.lateLeadLaunchVehicle", "vehicle"),
    "lead_launch_stall": ("long.leadLaunchStall", "unknown"),
  }
  event_type, attribution = names[event.event_type]
  severity = "critical" if event.severity >= 3 else ("warning" if event.severity else "info")
  return EventCandidate(
    occurred_mono_time=round(sample.t * 1e9),
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

  def update(self, lateral_sample: LateralSample, longitudinal_sample: LaunchSample,
             manual_pressed: bool = False, manual_time_ns: int | None = None) -> list[AcceptedEvent]:
    candidates: list[EventCandidate] = []
    try:
      detection = self.lateral_detector.update(lateral_sample)
      if detection is not None:
        candidates.append(lateral_candidate(lateral_sample, detection))
    except Exception:
      self.on_error("lateral")

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

  if candidate.domain == "lateral":
    sample: LateralSample = candidate.payload
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
  else:
    out.payload.none = None
  return msg


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
  )


def lateral_sample(sm: messaging.SubMaster, rate_filter: SteeringRateFilter,
                   bump_classifier: RoadBumpClassifier) -> LateralSample:
  controls_state = sm["controlsState"]
  torque_state = controls_state.lateralControlState.torqueState if controls_state.lateralControlState.which() == "torqueState" else None
  car_state = sm["carState"]
  now = sm.logMonoTime["controlsState"] * 1e-9 if sm.logMonoTime["controlsState"] else time.monotonic()
  angle = float(car_state.steeringAngleDeg)
  road_confounded = sm.valid["livePose"] and bump_classifier.update(float(sm["livePose"].accelerationDevice.z), now)
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
    controller_version=int(torque_state.version) if torque_state is not None else 0,
    reference_version=int(torque_state.referenceVersion) if torque_state is not None else 0,
    reference_rate=float(torque_state.referenceRate) if torque_state is not None else 0.0,
    reference_target_torque=float(torque_state.referenceReachableTargetTorque) if torque_state is not None else 0.0,
    reference_unwind_scale=float(torque_state.referenceUnwindScale) if torque_state is not None else 0.0,
    reference_sustained_unwind_scale=float(torque_state.referenceSustainedUnwindScale) if torque_state is not None else 0.0,
    unwind_effective_phase=float(torque_state.unwindEffectivePhase) if torque_state is not None else 0.0,
    unwind_overspeed=float(torque_state.unwindPhaseOverspeed) if torque_state is not None else 0.0,
    unwind_same_episode=bool(torque_state.unwindSameEpisode) if torque_state is not None else False,
    road_confounded=bool(road_confounded),
  )


def main() -> None:
  params = Params()
  git_commit = _param_text(params, "GitCommit")
  git_branch = _param_text(params, "GitBranch")
  pm = messaging.PubMaster(["drivingEvent"])
  sm = messaging.SubMaster(
    ["bookmarkButton", "carState", "carControl", "carOutput", "controlsState",
     "radarState", "longitudinalPlan", "modelV2", "livePose"],
    poll="carState",
  )
  platform = DrivingEventPlatform()
  rate_filter = SteeringRateFilter()
  bump_classifier = RoadBumpClassifier()

  while True:
    sm.update(1000)
    if not sm.updated["carState"] and not sm.updated["bookmarkButton"]:
      continue
    lat_sample = lateral_sample(sm, rate_filter, bump_classifier)
    long_sample = longitudinal_sample(sm)
    events = platform.update(
      lat_sample,
      long_sample,
      manual_pressed=bool(sm.updated["bookmarkButton"]),
      manual_time_ns=time.monotonic_ns(),
    )
    for event in events:
      pm.send("drivingEvent", build_message(event, git_commit, git_branch))
      cloudlog.event("driving_event", event_id=event.event_id, group_id=event.group_id,
                     domain=event.candidate.domain, event_type=event.candidate.event_type)


if __name__ == "__main__":
  main()
