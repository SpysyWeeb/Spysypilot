#!/usr/bin/env python3
import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from openpilot.cereal import messaging
from openpilot.common.hardware import PC
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

EVENT_VERSION = 1
EVENT_COOLDOWN = 8.0
EVENT_INDEX_MAX_BYTES = 2 * 1024 * 1024
EVENT_INDEX_DIR = Path(Paths.comma_home()) / "lat_events" if PC else Path("/data/community/lat_events")


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


@dataclass
class Detection:
  event_type: str
  severity: str
  confidence: float
  reason: str


class LateralEventDetector:
  """Conservative, auditable detector for known BLaT failure shapes."""

  def __init__(self, cooldown: float = EVENT_COOLDOWN):
    self.cooldown = cooldown
    self.last_event_time = -math.inf
    self.prev_phase = 0.0
    self.prev_same_episode = False
    self.stationary_since: float | None = None
    self.was_stalled = False
    self.late_unwind_since: float | None = None
    self.authority_since: float | None = None
    self.releases: deque[float] = deque()

  def suppress(self, now: float) -> None:
    self.last_event_time = now

  def reset_temporal(self) -> None:
    self.stationary_since = None
    self.was_stalled = False
    self.late_unwind_since = None
    self.authority_since = None
    self.releases.clear()

  def update(self, sample: LateralSample) -> Detection | None:
    phase_handoff = self.prev_phase > 0.6 and (
      sample.unwind_effective_phase < 0.25 or (self.prev_same_episode and not sample.unwind_same_episode)
    )
    self.prev_phase = sample.unwind_effective_phase
    self.prev_same_episode = sample.unwind_same_episode

    if not sample.active or sample.steering_pressed:
      self.reset_temporal()
      return None

    detection: Detection | None = None
    if phase_handoff and abs(sample.steering_rate_deg) > 60.0 and sample.applied_target_gap > 0.18:
      detection = Detection(
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
      detection = Detection(
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
      detection = Detection(
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
      detection = Detection(
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
      detection = Detection(
        "torqueAuthority", "critical", 0.80,
        "requested and applied torque stayed saturated while lateral tracking error continued growing",
      )
      self.authority_since = None

    if detection is None or sample.mono_time - self.last_event_time < self.cooldown:
      return None
    self.last_event_time = sample.mono_time
    return detection


class LateralEventStore:
  def __init__(self, root: Path = EVENT_INDEX_DIR, max_bytes: int = EVENT_INDEX_MAX_BYTES):
    self.root = root
    self.max_bytes = max_bytes
    self.path = root / "lateral_events.jsonl"

  def write(self, record: dict) -> None:
    self.root.mkdir(parents=True, exist_ok=True)
    if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
      rotated = self.path.with_suffix(".jsonl.1")
      os.replace(self.path, rotated)

    payload = (json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n").encode()
    fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
      view = memoryview(payload)
      while view:
        view = view[os.write(fd, view):]
      os.fsync(fd)
    finally:
      os.close(fd)


class RoadBumpClassifier:
  def __init__(self):
    self.baseline_z: float | None = None
    self.confounded_until = -math.inf

  def update(self, z_accel: float, now: float) -> bool:
    self.baseline_z = z_accel if self.baseline_z is None else 0.995 * self.baseline_z + 0.005 * z_accel
    if abs(z_accel - self.baseline_z) > 1.25:
      self.confounded_until = now + 0.75
    return now < self.confounded_until


def current_segment(route: str) -> int:
  if not route:
    return -1
  prefix = f"{route}--"
  segments = []
  try:
    for entry in os.scandir(Paths.log_root()):
      if entry.is_dir() and entry.name.startswith(prefix):
        try:
          segments.append(int(entry.name.removeprefix(prefix)))
        except ValueError:
          pass
  except OSError:
    return -1
  return max(segments, default=-1)


def event_record(params: Params, sample: LateralSample, detection: Detection, source: str) -> dict:
  route = params.get("CurrentRoute", encoding="utf8") or ""
  return {
    "eventVersion": EVENT_VERSION,
    "source": source,
    "type": detection.event_type,
    "severity": detection.severity,
    "confidence": detection.confidence,
    "reason": detection.reason,
    "route": route,
    "segment": current_segment(route),
    "gitCommit": params.get("GitCommit", encoding="utf8") or "",
    "gitBranch": params.get("GitBranch", encoding="utf8") or "",
    **asdict(sample),
    "appliedTargetGap": sample.applied_target_gap,
    "driverConfounded": sample.driver_confounded,
  }


def publish_event(pm: messaging.PubMaster, sample: LateralSample, detection: Detection, source: str) -> None:
  msg = messaging.new_message("lateralEvent", valid=True)
  event = msg.lateralEvent
  event.version = EVENT_VERSION
  event.type = detection.event_type
  event.source = source
  event.severity = detection.severity
  event.confidence = detection.confidence
  event.controllerVersion = sample.controller_version
  event.referenceVersion = sample.reference_version
  event.vEgo = sample.v_ego
  event.steeringAngleDeg = sample.steering_angle_deg
  event.steeringRateDeg = sample.steering_rate_deg
  event.desiredLateralAccel = sample.desired_lateral_accel
  event.actualLateralAccel = sample.actual_lateral_accel
  event.requestTorque = sample.request_torque
  event.appliedTorque = sample.applied_torque
  event.referenceTargetTorque = sample.reference_target_torque
  event.referenceUnwindScale = sample.reference_sustained_unwind_scale
  event.appliedTargetGap = sample.applied_target_gap
  event.roadConfounded = sample.road_confounded
  event.driverConfounded = sample.driver_confounded
  event.reason = detection.reason
  pm.send("lateralEvent", msg)


def main() -> None:
  params = Params()
  store = LateralEventStore()
  detector = LateralEventDetector()
  bump_classifier = RoadBumpClassifier()
  pm = messaging.PubMaster(["lateralEvent"])
  sm = messaging.SubMaster(
    ["controlsState", "carState", "carControl", "carOutput", "livePose", "bookmarkButton"],
    poll="controlsState",
  )

  previous_angle: float | None = None
  filtered_rate = 0.0
  previous_time: float | None = None
  sample = LateralSample(time.monotonic())

  while True:
    sm.update(1000)
    now = sm.logMonoTime["controlsState"] * 1e-9 if sm.logMonoTime["controlsState"] else time.monotonic()
    car_state = sm["carState"]
    angle = float(car_state.steeringAngleDeg)
    dt = now - previous_time if previous_time is not None else 0.01
    if previous_angle is not None and 0.002 < dt < 0.1:
      raw_rate = max(-800.0, min(800.0, (angle - previous_angle) / dt))
      alpha = dt / (0.08 + dt)
      filtered_rate += alpha * (raw_rate - filtered_rate)
    previous_angle = angle
    previous_time = now

    torque_state = None
    controls_state = sm["controlsState"]
    if controls_state.lateralControlState.which() == "torqueState":
      torque_state = controls_state.lateralControlState.torqueState

    road_confounded = False
    if sm.valid["livePose"]:
      road_confounded = bump_classifier.update(float(sm["livePose"].accelerationDevice.z), now)

    sample = LateralSample(
      mono_time=now,
      active=bool(torque_state is not None and torque_state.active and sm["carControl"].latActive),
      v_ego=float(car_state.vEgo),
      steering_angle_deg=angle,
      steering_rate_deg=filtered_rate,
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
      road_confounded=road_confounded,
    )

    automatic = detector.update(sample)
    if sm.updated["bookmarkButton"]:
      detector.suppress(now)
      detections = [(Detection("manual", "info", 1.0, "user marked a lateral event"), "user")]
    elif automatic is not None:
      detections = [(automatic, "automatic")]
    else:
      detections = []

    for detection, source in detections:
      record = event_record(params, sample, detection, source)
      try:
        store.write(record)
      except OSError:
        cloudlog.exception("failed to save lateral event marker")
        continue
      publish_event(pm, sample, detection, source)
      cloudlog.warning(f"lateral event logged: {detection.event_type}: {detection.reason}")


if __name__ == "__main__":
  main()
