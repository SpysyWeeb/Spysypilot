"""Pure lateral driving-event detection and signal conditioning.

This module intentionally owns no messaging, Params, filesystem, UI, process
lifecycle, or actuation behavior. Detector v5 distinguishes an intentionally
requested center crossing from an unrequested overshoot and keeps driver/road
interaction as attribution evidence rather than an event-suppression rule.
"""
import math
from collections import deque
from dataclasses import dataclass


DETECTOR_VERSION = 6
EVENT_COOLDOWN = 8.0
STALL_RELEASE_WINDOW_S = 6.0
STALL_TRANSITION_HYSTERESIS_S = 0.25
EVIDENCE_HISTORY_S = 25.0
SAMPLE_GAP_RESET_S = 1.0

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

UNWIND_ARM_ANGLE_DEG = 45.0
UNWIND_REFERENCE_SCALE = 0.6
UNWIND_PHASE = 0.6
UNWIND_REFERENCE_RATE_MIN = 0.2
UNWIND_PROGRESS_RATIO_MIN = 0.35
UNWIND_DEFICIT_ELIGIBLE_S = 0.10
UNWIND_DRIVER_ASSIST_WHEEL_RATE_DEG_S = 30.0
UNWIND_NO_ASSIST_OBSERVATION_S = 2.0
UNWIND_PROGRESS_RECENT_RATE_LOOKBACK_S = 0.75
UNWIND_PROGRESS_DRIVER_RATE_LOOKBACK_S = 0.25
UNWIND_RAW_DRIVER_ASSIST_S = 0.08
UNWIND_CONFIRMED_DRIVER_TORQUE = 200.0
UNWIND_MAX_EXPECTED_WHEEL_RATE_DEG_S = 400.0
UNWIND_EXPECTED_RATE_LOW_DEG_S = 25.0
UNWIND_EXPECTED_RATE_MID_DEG_S = 50.0
UNWIND_EXPECTED_RATE_HIGH_DEG_S = 112.5
UNWIND_DURATION_LOW_S = 0.8
UNWIND_DURATION_MID_S = 0.6
UNWIND_DURATION_HIGH_S = 0.4
UNWIND_LOW_BAND_MAX_DEG = 120.0
UNWIND_MID_BAND_MAX_DEG = 240.0
UNWIND_END_ANGLE_DEG = 25.0

OLD_TURN_SIGN_PHASE_DIRECTION_MIN = 0.5
CROWN_NEUTRAL_TOLERANCE = 0.05
CROWN_NEUTRAL_HOLD_S = 0.10
HIGH_ANGLE_SCALE_EPSILON = 1e-3
WHEEL_PROGRESS_MARKS_DEG = (5.0, 20.0, 50.0)
UNWIND_REBOUND_MIN_COMPONENT = 0.10

DRIVER_AT_START_WINDOW_S = 0.20
DRIVER_ASSIST_ACCELERATION_MIN_DEG_S = 10.0

DRIVER_CAUSATION_DRIVER_CREATED = "driverCreated"
DRIVER_CAUSATION_INTERVENTION_BACKED = "interventionBacked"
DRIVER_CAUSATION_AUTONOMOUS_ONLY = "autonomousOnly"
DRIVER_CAUSATION_MIXED = "mixed"

EPISODE_CENTER_CLOSE_DEG = 15.0
EPISODE_CENTER_CLOSE_S = 0.5
EPISODE_MAX_LIFETIME_S = 20.0
EPISODE_ACTIVE_RATE_DEG_S = 8.0
EPISODE_ACTIVE_PHASE = 0.2
EPISODE_QUIET_GAP_S = 2.0

TURN_STOP_MOVING_RATE_DEG_S = 30.0
TURN_STOP_ARM_ANGLE_DEG = 25.0
TURN_STOP_ABSOLUTE_DWELL_RATE_DEG_S = 8.0
TURN_STOP_DWELL_RATE_FRACTION = 0.25
TURN_STOP_SLOWING_RATE_FRACTION = 0.70
TURN_STOP_MIN_DWELL_S = 0.30
TURN_STOP_STRONG_DWELL_S = 0.50
TURN_STOP_STRONG_ANGLE_DEG = 90.0
TURN_STOP_STRONG_TORQUE = 0.80
TURN_STOP_TRACKING_ERROR = 0.15
TURN_STOP_RELEASE_OBSERVATION_S = 0.25
TURN_STOP_CLUSTER_SELECTION_S = 2.1
TURN_STOP_MAX_DWELL_S = 2.75
TURN_STOP_MIN_PRE_PROGRESS_DEG = 10.0
TURN_STOP_MIN_POST_PROGRESS_DEG = 10.0

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
  measurement_rate: float = 0.0
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

  # Crown-neutral torque context (stock schema degrades both to 0.0).
  unwind_neutral_torque: float = 0.0
  unwind_phase_direction: float = 0.0
  # Optional high-angle unwind evidence. None means the running schema lacks the fields.
  high_angle_unwind_scale: float | None = None
  torque_command_before_high_angle_exit: float | None = None
  high_angle_unwind_old_torque_correction: float | None = None
  high_angle_unwind_old_direction_torque: float | None = None

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
class DriverCausationSummary:
  """Granular driver-causation evidence (unwindProgressDeficit / turnStopTurn only)."""
  active_before_deficit: bool = False
  active_at_deficit_start: bool = False
  active_during_evaluation: bool = False
  intervened_after_deficit: bool = False
  intervention_accelerated_progress: bool = False
  assist_raw_torque_only: bool = False
  causation: str = ""


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

  unwind_episode_start_mono_time: float = 0.0
  unwind_deficit_start_mono_time: float = 0.0
  unwind_deficit_duration_s: float = 0.0
  initial_steering_angle_deg: float = 0.0
  peak_steering_angle_deg: float = 0.0
  trigger_steering_angle_deg: float = 0.0
  expected_unwind_direction: str = ""
  expected_angle_progress_deg: float = 0.0
  actual_angle_progress_deg: float = 0.0
  unwind_progress_ratio: float = 0.0
  reference_rate_at_trigger: float = 0.0
  measurement_rate_at_trigger: float = 0.0
  peak_reference_measurement_rate_gap: float = 0.0
  requested_torque_neutral_cross_mono_time: float | None = None
  applied_torque_neutral_cross_mono_time: float | None = None
  unwind_command_delay_s: float = 0.0
  driver_assisted_unwind: bool = False
  driver_assist_mono_time: float | None = None
  driver_assist_angle_deg: float = 0.0
  driver_assist_wheel_rate_deg: float = 0.0
  driver_assist_torque: float = 0.0
  wheel_rate_increase_after_assist_deg_s: float = 0.0
  unwind_requested_torque_at_trigger: float = 0.0
  unwind_applied_torque_at_trigger: float = 0.0
  unwind_reference_target_torque_at_trigger: float = 0.0
  unwind_applied_target_gap_at_trigger: float = 0.0
  unwind_peak_applied_target_gap: float = 0.0

  movement_start_mono_time: float = 0.0
  dwell_start_mono_time: float = 0.0
  release_mono_time: float = 0.0
  dwell_duration_s: float = 0.0
  dwell_steering_angle_deg: float = 0.0
  pre_dwell_peak_rate_deg_s: float = 0.0
  minimum_dwell_rate_deg_s: float = 0.0
  release_peak_rate_deg_s: float = 0.0
  rate_reduction_ratio: float = 0.0
  restart_classification: str = ""
  request_torque_during_dwell: float = 0.0
  applied_torque_during_dwell: float = 0.0
  reference_target_during_dwell: float = 0.0
  tracking_active_during_dwell: bool = False
  driver_involved_before_dwell: bool = False
  driver_involved_during_dwell: bool = False
  driver_involved_after_dwell: bool = False
  turn_stop_actual_damping_amount: float | None = None
  turn_stop_actual_damping_state: str | None = None
  turn_stop_turn_in_blocked: bool | None = None
  turn_stop_breakaway_latch: float | None = None
  turn_stop_sustain_floor_contribution: float | None = None
  turn_stop_damping_version: int | None = None

  # v6 crown-neutral trigger snapshots (payload @127-@134).
  unwind_neutral_torque: float = 0.0
  unwind_phase_direction: float = 0.0
  high_angle_evidence_valid: bool = False
  high_angle_unwind_scale: float = 0.0
  torque_command_before_high_angle_exit: float = 0.0
  high_angle_unwind_old_torque_correction: float = 0.0
  high_angle_unwind_old_direction_torque: float = 0.0
  old_turn_sign: float = 0.0
  # v6 per-episode timestamps (payload @135-@149).
  future_unwind_commit_mono_time: float | None = None
  high_angle_exit_first_nonzero_mono_time: float | None = None
  requested_crown_neutral_mono_time: float | None = None
  applied_crown_neutral_mono_time: float | None = None
  unwind_crown_command_delay_s: float = 0.0
  wheel_progress_5_mono_time: float | None = None
  wheel_progress_20_mono_time: float | None = None
  wheel_progress_50_mono_time: float | None = None
  # v6 old-torque rebound tracker snapshot (payload @150-@154).
  unwind_rebound_max_magnitude: float = 0.0
  unwind_rebound_start_mono_time: float | None = None
  unwind_rebound_duration_s: float = 0.0
  unwind_rebound_same_episode: bool = False
  # v6 driver causation (payload @155-@160, @165).
  driver_active_before_deficit: bool = False
  driver_active_at_deficit_start: bool = False
  driver_active_during_evaluation: bool = False
  driver_intervened_after_deficit: bool = False
  driver_intervention_accelerated_progress: bool = False
  driver_causation: str = ""
  # v6 turn-stop lifetime progress (payload @161-@162).
  turn_stop_pre_dwell_progress_deg: float = 0.0
  turn_stop_post_dwell_progress_deg: float = 0.0
  # v6 consolidated road-metrics window start actually covered (payload @163-@164).
  road_evidence_window_start_mono_time: float | None = None
  driver_assist_raw_torque_only: bool = False

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


@dataclass
class UnwindProgressEpisode:
  start_mono_time: float
  initial_steering_angle_deg: float
  peak_steering_angle_deg: float
  expected_direction: int
  expected_rate_deg_s: float
  deficit_start_mono_time: float | None = None
  mature_mono_time: float | None = None
  trigger_steering_angle_deg: float = 0.0
  trigger_expected_progress_deg: float = 0.0
  trigger_actual_progress_deg: float = 0.0
  trigger_progress_ratio: float = 0.0
  trigger_reference_rate: float = 0.0
  trigger_measurement_rate: float = 0.0
  trigger_requested_torque: float = 0.0
  trigger_applied_torque: float = 0.0
  trigger_reference_target_torque: float = 0.0
  trigger_applied_target_gap: float = 0.0
  peak_reference_measurement_rate_gap: float = 0.0
  peak_applied_target_gap: float = 0.0
  requested_neutral_cross_mono_time: float | None = None
  applied_neutral_cross_mono_time: float | None = None
  previous_request_torque: float = 0.0
  previous_applied_torque: float = 0.0
  previous_mono_time: float = 0.0
  recent_toward_rates: deque[tuple[float, float]] | None = None
  steering_pressed_was_active: bool = False
  raw_driver_assist_since: float | None = None
  raw_driver_assist_start_rate_deg_s: float = 0.0
  # v6: episode identity snapshot at arm (A2) and crown-neutral machinery.
  episode_start_snapshot: float = 0.0
  old_turn_sign: float = 0.0
  future_unwind_commit_mono_time: float | None = None
  high_angle_exit_first_nonzero_mono_time: float | None = None
  requested_crown_candidate_start: float | None = None
  requested_crown_neutral_mono_time: float | None = None
  applied_crown_candidate_start: float | None = None
  applied_crown_neutral_mono_time: float | None = None
  wheel_progress_5_mono_time: float | None = None
  wheel_progress_20_mono_time: float | None = None
  wheel_progress_50_mono_time: float | None = None
  # v6: old-torque rebound tracker (armed at first requested crown candidate,
  # but only once a genuine prior old-direction hold has been observed).
  rebound_armed_mono_time: float | None = None
  rebound_start_mono_time: float | None = None
  rebound_max_magnitude: float = 0.0
  rebound_duration_s: float = 0.0
  rebound_same_episode: bool = False
  rebound_previous_above: bool = False
  held_old_direction_before_release: bool = False
  # v6: crown/high-angle trigger snapshots.
  trigger_unwind_neutral_torque: float = 0.0
  trigger_unwind_phase_direction: float = 0.0
  trigger_high_angle_valid: bool = False
  trigger_high_angle_unwind_scale: float = 0.0
  trigger_torque_command_before_high_angle_exit: float = 0.0
  trigger_high_angle_old_torque_correction: float = 0.0
  trigger_high_angle_old_direction_torque: float = 0.0


@dataclass
class TurnStopEpisode:
  state: str
  movement_start_mono_time: float
  movement_direction: int
  pre_dwell_peak_signed_rate_deg_s: float
  pre_dwell_peak_rate_deg_s: float
  driver_involved_before_dwell: bool
  dwell_start_mono_time: float | None = None
  dwell_steering_angle_deg: float = 0.0
  minimum_dwell_rate_deg_s: float = math.inf
  max_request_torque_during_dwell: float = 0.0
  max_applied_torque_during_dwell: float = 0.0
  max_reference_target_during_dwell: float = 0.0
  tracking_active_during_dwell: bool = True
  meaningful_reference_during_dwell: bool = False
  driver_involved_during_dwell: bool = False
  release_mono_time: float | None = None
  release_sample: LateralSample | None = None
  release_peak_rate_deg_s: float = 0.0
  restart_classification: str = ""
  driver_involved_after_dwell: bool = False
  # v6: episode identity snapshot at creation (A2) and lifetime progress.
  episode_start_snapshot: float = 0.0
  creation_angle_deg: float = 0.0
  pre_dwell_progress_deg: float = 0.0


@dataclass
class PendingTurnStopCandidate:
  episode: TurnStopEpisode
  release_sample: LateralSample
  score: float
  deadline_mono_time: float
  previous_phase: float
  previous_same_episode: bool
  post_release_progress_deg: float = 0.0


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
    # desired lateral acceleration, tracking error, applied-target gap, and
    # signed wheel rate.
    self.history: deque[tuple[float, bool, float, bool, bool, bool, float, float, float, float, float]] = deque()
    self.episode_start: float | None = None
    self.episode_last_activity = -math.inf
    self.episode_center_since: float | None = None
    self.episode_center_closed = False
    self.unwind_progress_episode: UnwindProgressEpisode | None = None
    self.unwind_progress_latched = False
    self.turn_stop_episode: TurnStopEpisode | None = None
    self.pending_turn_stop_candidate: PendingTurnStopCandidate | None = None
    # Dedicated route-discontinuity clock; previous_sample skips exactly-zero-angle samples.
    self.last_sample_mono_time: float | None = None

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
    self.episode_center_since = None
    self.episode_center_closed = False
    self.unwind_progress_episode = None
    self.unwind_progress_latched = False
    self.turn_stop_episode = None
    self.pending_turn_stop_candidate = None

  def reset_discontinuity(self) -> None:
    # Backwards time leaves future-stamped cooldowns that would suppress
    # everything, and pre-gap history would contaminate post-gap evidence.
    self.reset_temporal()
    self.history.clear()
    self.last_event_times.clear()

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
      sample.steering_rate_deg,
    ))
    while self.history and sample.mono_time - self.history[0][0] > EVIDENCE_HISTORY_S:
      self.history.popleft()

  def _update_episode(self, sample: LateralSample, tracking_active: bool) -> None:
    in_center_band = abs(sample.steering_angle_deg) <= EPISODE_CENTER_CLOSE_DEG
    if in_center_band:
      if self.episode_center_since is None:
        self.episode_center_since = sample.mono_time
    else:
      self.episode_center_since = None
      self.episode_center_closed = False

    if self.episode_start is not None:
      if (
        self.episode_center_since is not None
        and sample.mono_time - self.episode_center_since >= EPISODE_CENTER_CLOSE_S
      ):
        # Sustained near-center settles the physical maneuver regardless of
        # other activity bits; a fresh episode needs a fresh sustained period.
        self.episode_start = None
        self.episode_center_closed = True
        self.episode_center_since = sample.mono_time
      elif sample.mono_time - self.episode_start > EPISODE_MAX_LIFETIME_S:
        self.episode_start = None

    active_shape = (
      abs(sample.steering_angle_deg) > EPISODE_CENTER_CLOSE_DEG
      or abs(sample.steering_rate_deg) > EPISODE_ACTIVE_RATE_DEG_S
      or sample.unwind_effective_phase > EPISODE_ACTIVE_PHASE
      or tracking_active
    )
    if active_shape:
      if self.episode_start is None:
        # After a center-close, tracking_active alone may not re-arm while the
        # wheel stays settled in the center band (A2 churn guard).
        if (
          not self.episode_center_closed
          or abs(sample.steering_angle_deg) > EPISODE_CENTER_CLOSE_DEG
          or abs(sample.steering_rate_deg) > EPISODE_ACTIVE_RATE_DEG_S
        ):
          self.episode_start = sample.mono_time
          self.episode_center_closed = False
          self.episode_last_activity = sample.mono_time
      elif sample.mono_time - self.episode_last_activity > EPISODE_QUIET_GAP_S:
        self.episode_start = sample.mono_time
        self.episode_last_activity = sample.mono_time
      else:
        self.episode_last_activity = sample.mono_time
    elif self.episode_start is not None and sample.mono_time - self.episode_last_activity > EPISODE_QUIET_GAP_S:
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
      "unwindProgressDeficit": (6.0, 2.0),
      "turnStopTurn": (6.0, 2.0),
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
                handoff_time: float | None = None,
                unwind_episode: UnwindProgressEpisode | None = None,
                turn_stop_episode: TurnStopEpisode | None = None,
                driver_assist_sample: LateralSample | None = None,
                wheel_rate_increase_after_assist: float = 0.0,
                turn_stop_release_sample: LateralSample | None = None,
                turn_stop_restart_classification: str = "",
                episode_start_snapshot: float | None = None,
                driver_causation: DriverCausationSummary | None = None,
                turn_stop_post_progress_deg: float = 0.0) -> LateralEvidence:
    if self.history:
      # The serialized window must equal the window the evidence actually
      # covered (history starts after boot or a route discontinuity).
      evidence_start = max(evidence_start, self.history[0][0])
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
        sample.steering_rate_deg,
      )]
    driver_count = sum(entry[1] for entry in samples)
    raw_torque_count = sum(abs(entry[2]) > DRIVER_RAW_TORQUE_THRESHOLD for entry in samples)
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

    episode_start = (
      episode_start_snapshot if episode_start_snapshot is not None
      else self.episode_start if self.episode_start is not None
      else evidence_start
    )
    # Road metrics cover the consolidated physical-maneuver window, clamped
    # to the oldest history actually available at emission (A6).
    road_window_start = min(episode_start, evidence_start)
    if self.history:
      road_window_start = max(road_window_start, self.history[0][0])
    road_samples = [entry for entry in self.history if entry[0] >= road_window_start] or samples
    road_count = sum(entry[4] for entry in road_samples)
    longest_road = self._longest_true_duration(road_samples, 4)
    road_fraction = road_count / len(road_samples)
    road_interaction = (
      ROAD_INTERACTION_SUBSTANTIAL
      if road_fraction >= ROAD_SUBSTANTIAL_FRACTION
      else ROAD_INTERACTION_TRANSIENT if road_count else ROAD_INTERACTION_NONE
    )

    before, after = self._analysis_window(event_type)
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
      max_vertical_accel_deviation=max(entry[6] for entry in road_samples),
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
      unwind_episode_start_mono_time=(
        unwind_episode.start_mono_time if unwind_episode is not None else 0.0
      ),
      unwind_deficit_start_mono_time=(
        unwind_episode.deficit_start_mono_time
        if unwind_episode is not None and unwind_episode.deficit_start_mono_time is not None else 0.0
      ),
      unwind_deficit_duration_s=(
        sample.mono_time - unwind_episode.deficit_start_mono_time
        if unwind_episode is not None and unwind_episode.deficit_start_mono_time is not None else 0.0
      ),
      initial_steering_angle_deg=(
        unwind_episode.initial_steering_angle_deg if unwind_episode is not None else 0.0
      ),
      peak_steering_angle_deg=(
        unwind_episode.peak_steering_angle_deg if unwind_episode is not None else 0.0
      ),
      trigger_steering_angle_deg=(
        unwind_episode.trigger_steering_angle_deg if unwind_episode is not None else 0.0
      ),
      expected_unwind_direction=(
        "positive" if unwind_episode is not None and unwind_episode.expected_direction > 0
        else "negative" if unwind_episode is not None else ""
      ),
      expected_angle_progress_deg=(
        unwind_episode.trigger_expected_progress_deg if unwind_episode is not None else 0.0
      ),
      actual_angle_progress_deg=(
        unwind_episode.trigger_actual_progress_deg if unwind_episode is not None else 0.0
      ),
      unwind_progress_ratio=(
        unwind_episode.trigger_progress_ratio if unwind_episode is not None else 0.0
      ),
      reference_rate_at_trigger=(
        unwind_episode.trigger_reference_rate if unwind_episode is not None else 0.0
      ),
      measurement_rate_at_trigger=(
        unwind_episode.trigger_measurement_rate if unwind_episode is not None else 0.0
      ),
      peak_reference_measurement_rate_gap=(
        unwind_episode.peak_reference_measurement_rate_gap if unwind_episode is not None else 0.0
      ),
      requested_torque_neutral_cross_mono_time=(
        unwind_episode.requested_neutral_cross_mono_time if unwind_episode is not None else None
      ),
      applied_torque_neutral_cross_mono_time=(
        unwind_episode.applied_neutral_cross_mono_time if unwind_episode is not None else None
      ),
      unwind_command_delay_s=(
        unwind_episode.requested_neutral_cross_mono_time - unwind_episode.start_mono_time
        if (
          unwind_episode is not None
          and unwind_episode.requested_neutral_cross_mono_time is not None
        ) else 0.0
      ),
      driver_assisted_unwind=driver_assist_sample is not None,
      driver_assist_mono_time=(
        driver_assist_sample.mono_time if driver_assist_sample is not None else None
      ),
      driver_assist_angle_deg=(
        driver_assist_sample.steering_angle_deg if driver_assist_sample is not None else 0.0
      ),
      driver_assist_wheel_rate_deg=(
        driver_assist_sample.steering_rate_deg if driver_assist_sample is not None else 0.0
      ),
      driver_assist_torque=(
        driver_assist_sample.driver_torque if driver_assist_sample is not None else 0.0
      ),
      wheel_rate_increase_after_assist_deg_s=wheel_rate_increase_after_assist,
      unwind_requested_torque_at_trigger=(
        unwind_episode.trigger_requested_torque if unwind_episode is not None else 0.0
      ),
      unwind_applied_torque_at_trigger=(
        unwind_episode.trigger_applied_torque if unwind_episode is not None else 0.0
      ),
      unwind_reference_target_torque_at_trigger=(
        unwind_episode.trigger_reference_target_torque if unwind_episode is not None else 0.0
      ),
      unwind_applied_target_gap_at_trigger=(
        unwind_episode.trigger_applied_target_gap if unwind_episode is not None else 0.0
      ),
      unwind_peak_applied_target_gap=(
        unwind_episode.peak_applied_target_gap if unwind_episode is not None else 0.0
      ),
      movement_start_mono_time=(
        turn_stop_episode.movement_start_mono_time if turn_stop_episode is not None else 0.0
      ),
      dwell_start_mono_time=(
        turn_stop_episode.dwell_start_mono_time
        if turn_stop_episode is not None and turn_stop_episode.dwell_start_mono_time is not None else 0.0
      ),
      release_mono_time=(
        turn_stop_episode.release_mono_time
        if turn_stop_episode is not None and turn_stop_episode.release_mono_time is not None else 0.0
      ),
      dwell_duration_s=(
        turn_stop_episode.release_mono_time - turn_stop_episode.dwell_start_mono_time
        if (
          turn_stop_episode is not None
          and turn_stop_episode.dwell_start_mono_time is not None
          and turn_stop_episode.release_mono_time is not None
        ) else 0.0
      ),
      dwell_steering_angle_deg=(
        turn_stop_episode.dwell_steering_angle_deg if turn_stop_episode is not None else 0.0
      ),
      pre_dwell_peak_rate_deg_s=(
        turn_stop_episode.pre_dwell_peak_rate_deg_s if turn_stop_episode is not None else 0.0
      ),
      minimum_dwell_rate_deg_s=(
        turn_stop_episode.minimum_dwell_rate_deg_s
        if turn_stop_episode is not None and math.isfinite(turn_stop_episode.minimum_dwell_rate_deg_s)
        else 0.0
      ),
      release_peak_rate_deg_s=(
        turn_stop_episode.release_peak_rate_deg_s if turn_stop_episode is not None else 0.0
      ),
      rate_reduction_ratio=(
        max(
          0.0,
          1.0 - turn_stop_episode.minimum_dwell_rate_deg_s
          / max(turn_stop_episode.pre_dwell_peak_rate_deg_s, 1e-3),
        )
        if turn_stop_episode is not None and math.isfinite(turn_stop_episode.minimum_dwell_rate_deg_s)
        else 0.0
      ),
      restart_classification=turn_stop_restart_classification,
      request_torque_during_dwell=(
        turn_stop_episode.max_request_torque_during_dwell if turn_stop_episode is not None else 0.0
      ),
      applied_torque_during_dwell=(
        turn_stop_episode.max_applied_torque_during_dwell if turn_stop_episode is not None else 0.0
      ),
      reference_target_during_dwell=(
        turn_stop_episode.max_reference_target_during_dwell if turn_stop_episode is not None else 0.0
      ),
      tracking_active_during_dwell=(
        turn_stop_episode.tracking_active_during_dwell if turn_stop_episode is not None else False
      ),
      driver_involved_before_dwell=(
        turn_stop_episode.driver_involved_before_dwell if turn_stop_episode is not None else False
      ),
      driver_involved_during_dwell=(
        turn_stop_episode.driver_involved_during_dwell if turn_stop_episode is not None else False
      ),
      driver_involved_after_dwell=(
        turn_stop_episode.driver_involved_after_dwell if turn_stop_episode is not None else False
      ),
      turn_stop_actual_damping_amount=(
        turn_stop_release_sample.actual_damping_amount if turn_stop_release_sample is not None else None
      ),
      turn_stop_actual_damping_state=(
        turn_stop_release_sample.actual_damping_state if turn_stop_release_sample is not None else None
      ),
      turn_stop_turn_in_blocked=(
        turn_stop_release_sample.turn_in_blocked if turn_stop_release_sample is not None else None
      ),
      turn_stop_breakaway_latch=(
        turn_stop_release_sample.breakaway_latch if turn_stop_release_sample is not None else None
      ),
      turn_stop_sustain_floor_contribution=(
        turn_stop_release_sample.sustain_floor_contribution
        if turn_stop_release_sample is not None else None
      ),
      turn_stop_damping_version=(
        turn_stop_release_sample.damping_version if turn_stop_release_sample is not None else None
      ),
      unwind_neutral_torque=(
        unwind_episode.trigger_unwind_neutral_torque if unwind_episode is not None else 0.0
      ),
      unwind_phase_direction=(
        unwind_episode.trigger_unwind_phase_direction if unwind_episode is not None else 0.0
      ),
      high_angle_evidence_valid=(
        unwind_episode.trigger_high_angle_valid if unwind_episode is not None else False
      ),
      high_angle_unwind_scale=(
        unwind_episode.trigger_high_angle_unwind_scale if unwind_episode is not None else 0.0
      ),
      torque_command_before_high_angle_exit=(
        unwind_episode.trigger_torque_command_before_high_angle_exit
        if unwind_episode is not None else 0.0
      ),
      high_angle_unwind_old_torque_correction=(
        unwind_episode.trigger_high_angle_old_torque_correction
        if unwind_episode is not None else 0.0
      ),
      high_angle_unwind_old_direction_torque=(
        unwind_episode.trigger_high_angle_old_direction_torque
        if unwind_episode is not None else 0.0
      ),
      old_turn_sign=unwind_episode.old_turn_sign if unwind_episode is not None else 0.0,
      future_unwind_commit_mono_time=(
        unwind_episode.future_unwind_commit_mono_time if unwind_episode is not None else None
      ),
      high_angle_exit_first_nonzero_mono_time=(
        unwind_episode.high_angle_exit_first_nonzero_mono_time
        if unwind_episode is not None else None
      ),
      requested_crown_neutral_mono_time=(
        unwind_episode.requested_crown_neutral_mono_time if unwind_episode is not None else None
      ),
      applied_crown_neutral_mono_time=(
        unwind_episode.applied_crown_neutral_mono_time if unwind_episode is not None else None
      ),
      unwind_crown_command_delay_s=(
        unwind_episode.requested_crown_neutral_mono_time - unwind_episode.start_mono_time
        if (
          unwind_episode is not None
          and unwind_episode.requested_crown_neutral_mono_time is not None
        ) else 0.0
      ),
      wheel_progress_5_mono_time=(
        unwind_episode.wheel_progress_5_mono_time if unwind_episode is not None else None
      ),
      wheel_progress_20_mono_time=(
        unwind_episode.wheel_progress_20_mono_time if unwind_episode is not None else None
      ),
      wheel_progress_50_mono_time=(
        unwind_episode.wheel_progress_50_mono_time if unwind_episode is not None else None
      ),
      unwind_rebound_max_magnitude=(
        unwind_episode.rebound_max_magnitude if unwind_episode is not None else 0.0
      ),
      unwind_rebound_start_mono_time=(
        unwind_episode.rebound_start_mono_time if unwind_episode is not None else None
      ),
      unwind_rebound_duration_s=(
        unwind_episode.rebound_duration_s if unwind_episode is not None else 0.0
      ),
      unwind_rebound_same_episode=(
        unwind_episode.rebound_same_episode if unwind_episode is not None else False
      ),
      driver_active_before_deficit=(
        driver_causation.active_before_deficit if driver_causation is not None else False
      ),
      driver_active_at_deficit_start=(
        driver_causation.active_at_deficit_start if driver_causation is not None else False
      ),
      driver_active_during_evaluation=(
        driver_causation.active_during_evaluation if driver_causation is not None else False
      ),
      driver_intervened_after_deficit=(
        driver_causation.intervened_after_deficit if driver_causation is not None else False
      ),
      driver_intervention_accelerated_progress=(
        driver_causation.intervention_accelerated_progress
        if driver_causation is not None else False
      ),
      driver_causation=driver_causation.causation if driver_causation is not None else "",
      turn_stop_pre_dwell_progress_deg=(
        turn_stop_episode.pre_dwell_progress_deg if turn_stop_episode is not None else 0.0
      ),
      turn_stop_post_dwell_progress_deg=turn_stop_post_progress_deg,
      road_evidence_window_start_mono_time=road_window_start,
      driver_assist_raw_torque_only=(
        driver_causation.assist_raw_torque_only if driver_causation is not None else False
      ),
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
            handoff_time: float | None = None,
            unwind_episode: UnwindProgressEpisode | None = None,
            turn_stop_episode: TurnStopEpisode | None = None,
            driver_assist_sample: LateralSample | None = None,
            wheel_rate_increase_after_assist: float = 0.0,
            turn_stop_release_sample: LateralSample | None = None,
            turn_stop_restart_classification: str = "",
            episode_start_snapshot: float | None = None,
            driver_causation: DriverCausationSummary | None = None,
            turn_stop_post_progress_deg: float = 0.0) -> LateralDetection | None:
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
      "unwindProgressDeficit": occurred_time - 6.0,
      "turnStopTurn": occurred_time - 6.0,
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
      unwind_episode,
      turn_stop_episode,
      driver_assist_sample,
      wheel_rate_increase_after_assist,
      turn_stop_release_sample,
      turn_stop_restart_classification,
      episode_start_snapshot,
      driver_causation,
      turn_stop_post_progress_deg,
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
        episode_start_snapshot=crossing.episode_start,
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
        episode_start_snapshot=crossing.episode_start,
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
      episode_start_snapshot=handoff.episode_start,
    )

  @staticmethod
  def _unwind_duration(angle_deg: float) -> float:
    angle = abs(angle_deg)
    if angle > UNWIND_MID_BAND_MAX_DEG:
      return UNWIND_DURATION_HIGH_S
    if angle > UNWIND_LOW_BAND_MAX_DEG:
      return UNWIND_DURATION_MID_S
    return UNWIND_DURATION_LOW_S

  @staticmethod
  def _unwind_minimum_rate(angle_deg: float) -> float:
    angle = abs(angle_deg)
    if angle > UNWIND_MID_BAND_MAX_DEG:
      return UNWIND_EXPECTED_RATE_HIGH_DEG_S
    if angle > UNWIND_LOW_BAND_MAX_DEG:
      return UNWIND_EXPECTED_RATE_MID_DEG_S
    return UNWIND_EXPECTED_RATE_LOW_DEG_S

  @staticmethod
  def _interpolated_neutral_cross(previous_time: float, previous_value: float,
                                   now: float, value: float) -> float | None:
    if previous_value == value or previous_value * value > 0.0:
      return None
    denominator = abs(previous_value) + abs(value)
    if denominator <= 1e-9:
      return now
    return previous_time + (now - previous_time) * abs(previous_value) / denominator

  @staticmethod
  def _unwind_reference_committed(sample: LateralSample, expected_direction: int) -> bool:
    # Standard openpilot curvature/lateral-acceleration sign is opposite the
    # steering-wheel angle sign. Keep this sign convention isolated here; the
    # reference rate remains a lateral-acceleration rate and is never
    # integrated as wheel angle.
    return (
      sample.reference_sustained_unwind_scale > UNWIND_REFERENCE_SCALE
      and sample.unwind_effective_phase > UNWIND_PHASE
      and sample.reference_rate * expected_direction < -UNWIND_REFERENCE_RATE_MIN
    )

  def _recent_wheel_rate_peak(self, now: float) -> float:
    return max(
      (
        abs(entry[10]) for entry in self.history
        if now - UNWIND_PROGRESS_RECENT_RATE_LOOKBACK_S <= entry[0] <= now
      ),
      default=0.0,
    )

  @staticmethod
  def _snapshot_unwind_trigger(episode: UnwindProgressEpisode, sample: LateralSample,
                               expected_progress: float, actual_progress: float,
                               progress_ratio: float) -> None:
    episode.trigger_steering_angle_deg = sample.steering_angle_deg
    episode.trigger_expected_progress_deg = expected_progress
    episode.trigger_actual_progress_deg = actual_progress
    episode.trigger_progress_ratio = progress_ratio
    episode.trigger_reference_rate = sample.reference_rate
    episode.trigger_measurement_rate = sample.measurement_rate
    episode.trigger_requested_torque = sample.request_torque
    episode.trigger_applied_torque = sample.applied_torque
    episode.trigger_reference_target_torque = sample.reference_target_torque
    episode.trigger_applied_target_gap = sample.applied_target_gap
    episode.trigger_unwind_neutral_torque = sample.unwind_neutral_torque
    episode.trigger_unwind_phase_direction = sample.unwind_phase_direction
    episode.trigger_high_angle_valid = (
      sample.high_angle_unwind_scale is not None
      and sample.torque_command_before_high_angle_exit is not None
      and sample.high_angle_unwind_old_torque_correction is not None
      and sample.high_angle_unwind_old_direction_torque is not None
    )
    episode.trigger_high_angle_unwind_scale = (
      sample.high_angle_unwind_scale if sample.high_angle_unwind_scale is not None else 0.0
    )
    episode.trigger_torque_command_before_high_angle_exit = (
      sample.torque_command_before_high_angle_exit
      if sample.torque_command_before_high_angle_exit is not None else 0.0
    )
    episode.trigger_high_angle_old_torque_correction = (
      sample.high_angle_unwind_old_torque_correction
      if sample.high_angle_unwind_old_torque_correction is not None else 0.0
    )
    episode.trigger_high_angle_old_direction_torque = (
      sample.high_angle_unwind_old_direction_torque
      if sample.high_angle_unwind_old_direction_torque is not None else 0.0
    )

  @staticmethod
  def _old_direction_component(torque: float, neutral: float, old_turn_sign: float) -> float:
    # Torque remaining in the original turn direction, measured against the
    # CURRENT crown-neutral (the neutral slews; never cache a route constant).
    return max((torque - neutral) * old_turn_sign, 0.0)

  def _update_unwind_crown_tracking(self, episode: UnwindProgressEpisode,
                                    sample: LateralSample) -> None:
    if episode.high_angle_exit_first_nonzero_mono_time is None and (
      sample.high_angle_unwind_scale is not None
      and sample.high_angle_unwind_scale > HIGH_ANGLE_SCALE_EPSILON
    ):
      episode.high_angle_exit_first_nonzero_mono_time = sample.mono_time

    requested_component = self._old_direction_component(
      sample.request_torque, sample.unwind_neutral_torque, episode.old_turn_sign,
    )
    if requested_component >= CROWN_NEUTRAL_TOLERANCE:
      # A genuine old-direction hold has now been observed; only after this
      # may the rebound tracker arm (Fix 3 -- otherwise an episode that arms
      # already near crown-neutral would misrepresent ordinary control-loop
      # torque chatter as a "rebound" from a release that never happened).
      episode.held_old_direction_before_release = True
    if requested_component < CROWN_NEUTRAL_TOLERANCE:
      if episode.requested_crown_candidate_start is None:
        episode.requested_crown_candidate_start = sample.mono_time
      if (
        episode.requested_crown_neutral_mono_time is None
        and sample.mono_time - episode.requested_crown_candidate_start >= CROWN_NEUTRAL_HOLD_S
      ):
        episode.requested_crown_neutral_mono_time = episode.requested_crown_candidate_start
    else:
      episode.requested_crown_candidate_start = None

    applied_component = self._old_direction_component(
      sample.applied_torque, sample.unwind_neutral_torque, episode.old_turn_sign,
    )
    if applied_component < CROWN_NEUTRAL_TOLERANCE:
      if episode.applied_crown_candidate_start is None:
        episode.applied_crown_candidate_start = sample.mono_time
      if (
        episode.applied_crown_neutral_mono_time is None
        and sample.mono_time - episode.applied_crown_candidate_start >= CROWN_NEUTRAL_HOLD_S
      ):
        episode.applied_crown_neutral_mono_time = episode.applied_crown_candidate_start
    else:
      episode.applied_crown_candidate_start = None

    # Rebound tracker arms at the first requested crown-neutral candidate (no
    # hold requirement) and then watches for old-direction re-excursions --
    # but only once a genuine prior old-direction hold has actually been
    # observed (Fix 3); otherwise there was never a "release" to rebound from.
    if episode.rebound_armed_mono_time is None:
      if episode.held_old_direction_before_release and requested_component < CROWN_NEUTRAL_TOLERANCE:
        episode.rebound_armed_mono_time = sample.mono_time
      return
    if requested_component > UNWIND_REBOUND_MIN_COMPONENT:
      if episode.rebound_start_mono_time is None:
        episode.rebound_start_mono_time = sample.mono_time
        episode.rebound_same_episode = True
      episode.rebound_max_magnitude = max(episode.rebound_max_magnitude, requested_component)
      if episode.rebound_previous_above:
        episode.rebound_duration_s += max(0.0, sample.mono_time - episode.previous_mono_time)
      episode.rebound_previous_above = True
    else:
      episode.rebound_previous_above = False

  def _unwind_driver_causation(self, episode: UnwindProgressEpisode, sample: LateralSample,
                               assist_confirmed: bool, steering_press_began: bool,
                               raw_assist_confirmed: bool,
                               wheel_rate_increase_after_assist: float) -> DriverCausationSummary:
    deficit_start = episode.deficit_start_mono_time
    if deficit_start is None:
      deficit_start = (
        episode.mature_mono_time if episode.mature_mono_time is not None
        else episode.start_mono_time
      )
    evaluation_end = (
      episode.mature_mono_time if episode.mature_mono_time is not None else sample.mono_time
    )
    active_before = any(
      entry[1] for entry in self.history
      if episode.start_mono_time <= entry[0] < deficit_start
    )
    active_at_start = any(
      entry[1] for entry in self.history
      if deficit_start - DRIVER_AT_START_WINDOW_S <= entry[0] <= deficit_start
    )
    active_during = any(
      entry[1] for entry in self.history
      if deficit_start <= entry[0] <= evaluation_end
    )
    intervened_after = assist_confirmed or (
      not active_at_start
      and any(
        entry[1] for entry in self.history
        if deficit_start < entry[0] <= sample.mono_time
      )
    )
    accelerated = wheel_rate_increase_after_assist >= DRIVER_ASSIST_ACCELERATION_MIN_DEG_S
    if active_before and active_at_start and active_during:
      causation = DRIVER_CAUSATION_DRIVER_CREATED
    elif not active_at_start and (assist_confirmed or (intervened_after and accelerated)):
      causation = DRIVER_CAUSATION_INTERVENTION_BACKED
    elif not (active_before or active_at_start or active_during or intervened_after):
      causation = DRIVER_CAUSATION_AUTONOMOUS_ONLY
    else:
      causation = DRIVER_CAUSATION_MIXED
    return DriverCausationSummary(
      active_before_deficit=active_before,
      active_at_deficit_start=active_at_start,
      active_during_evaluation=active_during,
      intervened_after_deficit=intervened_after,
      intervention_accelerated_progress=accelerated,
      assist_raw_torque_only=raw_assist_confirmed and not steering_press_began,
      causation=causation,
    )

  @staticmethod
  def _turn_stop_driver_causation(episode: TurnStopEpisode) -> DriverCausationSummary:
    involved = (
      episode.driver_involved_before_dwell,
      episode.driver_involved_during_dwell,
      episode.driver_involved_after_dwell,
    )
    if all(involved):
      causation = DRIVER_CAUSATION_DRIVER_CREATED
    elif not any(involved):
      causation = DRIVER_CAUSATION_AUTONOMOUS_ONLY
    else:
      causation = DRIVER_CAUSATION_MIXED
    return DriverCausationSummary(causation=causation)

  def _advance_unwind_episode(self, episode: UnwindProgressEpisode, sample: LateralSample) -> None:
    """Per-sample state advancement for an ACTIVE unwind episode: the literal-
    zero neutral-cross interpolation, the crown-neutral/rebound tracker, and
    the previous_*/previous_mono_time bookkeeping they depend on.

    This must run on EVERY sample while an episode is active, independent of
    which detector (if any) wins the emission-priority slot that sample. It
    is called from two places: normally, inline from `_update_unwind_progress`
    when the priority chain reaches it uncontested; and directly from
    `update()` when a higher-priority detector (pending crossing/handoff)
    claims the emission slot instead, so this bookkeeping is never silently
    skipped -- a skipped sample would otherwise leave `previous_mono_time`
    stale, and the NEXT real invocation would compute its hold/rebound deltas
    across the wrong (too-wide) gap, lumping the skipped sample's real signal
    change into the wrong bucket."""
    if episode.requested_neutral_cross_mono_time is None:
      episode.requested_neutral_cross_mono_time = self._interpolated_neutral_cross(
        episode.previous_mono_time,
        episode.previous_request_torque,
        sample.mono_time,
        sample.request_torque,
      )
    if episode.applied_neutral_cross_mono_time is None:
      episode.applied_neutral_cross_mono_time = self._interpolated_neutral_cross(
        episode.previous_mono_time,
        episode.previous_applied_torque,
        sample.mono_time,
        sample.applied_torque,
      )
    self._update_unwind_crown_tracking(episode, sample)
    episode.previous_request_torque = sample.request_torque
    episode.previous_applied_torque = sample.applied_torque
    episode.previous_mono_time = sample.mono_time

  def _update_unwind_progress(self, sample: LateralSample, previous_phase: float,
                              previous_same_episode: bool) -> LateralDetection | None:
    if (
      self.unwind_progress_latched
      and abs(sample.steering_angle_deg) <= UNWIND_END_ANGLE_DEG
    ):
      self.unwind_progress_latched = False

    episode = self.unwind_progress_episode
    expected_direction = -1 if sample.steering_angle_deg > 0.0 else 1
    committed = self._unwind_reference_committed(sample, expected_direction)
    if episode is None:
      if (
        self.unwind_progress_latched
        or abs(sample.steering_angle_deg) <= UNWIND_ARM_ANGLE_DEG
        or not committed
      ):
        return None
      expected_rate = min(
        UNWIND_MAX_EXPECTED_WHEEL_RATE_DEG_S,
        max(
          self._unwind_minimum_rate(sample.steering_angle_deg),
          self._recent_wheel_rate_peak(sample.mono_time),
        ),
      )
      # The phase-direction field is authoritative for the original turn's
      # torque sign when populated; otherwise fall back to the wheel side.
      old_turn_sign = (
        (1.0 if sample.unwind_phase_direction > 0 else -1.0)
        if abs(sample.unwind_phase_direction) > OLD_TURN_SIGN_PHASE_DIRECTION_MIN
        else (1.0 if sample.steering_angle_deg > 0.0 else -1.0)
      )
      episode = UnwindProgressEpisode(
        start_mono_time=sample.mono_time,
        initial_steering_angle_deg=sample.steering_angle_deg,
        peak_steering_angle_deg=sample.steering_angle_deg,
        expected_direction=expected_direction,
        expected_rate_deg_s=expected_rate,
        previous_request_torque=sample.request_torque,
        previous_applied_torque=sample.applied_torque,
        previous_mono_time=sample.mono_time,
        recent_toward_rates=deque(),
        steering_pressed_was_active=sample.steering_pressed,
        episode_start_snapshot=(
          self.episode_start if self.episode_start is not None else sample.mono_time
        ),
        old_turn_sign=old_turn_sign,
        # Arming requires the committed reference, so the commit is the arm time.
        future_unwind_commit_mono_time=sample.mono_time,
      )
      self.unwind_progress_episode = episode

    # A meaningful reference that changes back toward the existing turn
    # invalidates an apparent unwind before publication.
    if (
      sample.reference_sustained_unwind_scale > UNWIND_REFERENCE_SCALE
      and sample.unwind_effective_phase > UNWIND_PHASE
      and abs(sample.reference_rate) > UNWIND_REFERENCE_RATE_MIN
      and sample.reference_rate * episode.expected_direction >= 0.0
    ):
      self.unwind_progress_episode = None
      return None

    if abs(sample.steering_angle_deg) > abs(episode.peak_steering_angle_deg):
      episode.peak_steering_angle_deg = sample.steering_angle_deg
    episode.peak_reference_measurement_rate_gap = max(
      episode.peak_reference_measurement_rate_gap,
      abs(sample.reference_rate - sample.measurement_rate),
    )
    episode.peak_applied_target_gap = max(
      episode.peak_applied_target_gap,
      sample.applied_target_gap,
    )

    self._advance_unwind_episode(episode, sample)

    elapsed = max(0.0, sample.mono_time - episode.start_mono_time)
    expected_progress = min(
      abs(episode.initial_steering_angle_deg),
      episode.expected_rate_deg_s * elapsed,
    )
    actual_progress = max(
      0.0,
      abs(episode.initial_steering_angle_deg) - abs(sample.steering_angle_deg),
    )
    if episode.wheel_progress_5_mono_time is None and actual_progress >= WHEEL_PROGRESS_MARKS_DEG[0]:
      episode.wheel_progress_5_mono_time = sample.mono_time
    if episode.wheel_progress_20_mono_time is None and actual_progress >= WHEEL_PROGRESS_MARKS_DEG[1]:
      episode.wheel_progress_20_mono_time = sample.mono_time
    if episode.wheel_progress_50_mono_time is None and actual_progress >= WHEEL_PROGRESS_MARKS_DEG[2]:
      episode.wheel_progress_50_mono_time = sample.mono_time
    progress_ratio = actual_progress / max(expected_progress, 1e-3)
    deficit = (
      expected_progress >= UNWIND_DEFICIT_ELIGIBLE_S * episode.expected_rate_deg_s
      and progress_ratio < UNWIND_PROGRESS_RATIO_MIN
    )
    if deficit and episode.deficit_start_mono_time is None:
      episode.deficit_start_mono_time = sample.mono_time
    elif (
      not deficit
      and episode.mature_mono_time is None
      and episode.deficit_start_mono_time is not None
    ):
      episode.deficit_start_mono_time = None

    if (
      episode.deficit_start_mono_time is not None
      and episode.mature_mono_time is None
      and sample.mono_time - episode.deficit_start_mono_time
      >= self._unwind_duration(episode.peak_steering_angle_deg)
    ):
      episode.mature_mono_time = sample.mono_time
      self._snapshot_unwind_trigger(
        episode,
        sample,
        expected_progress,
        actual_progress,
        progress_ratio,
      )

    toward_rate = max(0.0, sample.steering_rate_deg * episode.expected_direction)
    recent_rates = episode.recent_toward_rates
    assert recent_rates is not None
    while recent_rates and sample.mono_time - recent_rates[0][0] > UNWIND_PROGRESS_DRIVER_RATE_LOOKBACK_S:
      recent_rates.popleft()
    previous_toward_rate = max((value for _, value in recent_rates), default=0.0)
    raw_assist_active = (
      sample.driver_torque * episode.expected_direction >= UNWIND_CONFIRMED_DRIVER_TORQUE
      and toward_rate >= UNWIND_DRIVER_ASSIST_WHEEL_RATE_DEG_S
    )
    if raw_assist_active:
      if episode.raw_driver_assist_since is None:
        episode.raw_driver_assist_since = sample.mono_time
        episode.raw_driver_assist_start_rate_deg_s = toward_rate
    else:
      episode.raw_driver_assist_since = None
      episode.raw_driver_assist_start_rate_deg_s = 0.0
    raw_assist_confirmed = (
      episode.raw_driver_assist_since is not None
      and sample.mono_time - episode.raw_driver_assist_since >= UNWIND_RAW_DRIVER_ASSIST_S
    )
    steering_press_began = sample.steering_pressed and not episode.steering_pressed_was_active
    assist_confirmed = (
      episode.deficit_start_mono_time is not None
      and (steering_press_began or raw_assist_confirmed)
    )
    episode.steering_pressed_was_active = sample.steering_pressed
    recent_rates.append((sample.mono_time, toward_rate))

    if assist_confirmed:
      if episode.mature_mono_time is None:
        episode.mature_mono_time = sample.mono_time
        self._snapshot_unwind_trigger(
          episode,
          sample,
          expected_progress,
          actual_progress,
          progress_ratio,
        )
      occurred_time = episode.deficit_start_mono_time
      wheel_rate_increase = max(
        0.0,
        toward_rate - (
          episode.raw_driver_assist_start_rate_deg_s
          if raw_assist_confirmed else previous_toward_rate
        ),
      )
      detection = self._emit(
        sample,
        "unwindProgressDeficit",
        "warning",
        0.95,
        "driver intervention accelerated an unwind that was behind the requested trajectory",
        occurred_time,
        previous_phase,
        previous_same_episode,
        unwind_episode=episode,
        driver_assist_sample=sample,
        wheel_rate_increase_after_assist=wheel_rate_increase,
        episode_start_snapshot=episode.episode_start_snapshot,
        driver_causation=self._unwind_driver_causation(
          episode,
          sample,
          True,
          steering_press_began,
          raw_assist_confirmed,
          wheel_rate_increase,
        ),
      )
      self.unwind_progress_episode = None
      self.unwind_progress_latched = True
      self.late_unwind_since = None
      return detection

    if (
      episode.mature_mono_time is not None
      and sample.mono_time - episode.mature_mono_time >= UNWIND_NO_ASSIST_OBSERVATION_S
    ):
      occurred_time = (
        episode.deficit_start_mono_time
        if episode.deficit_start_mono_time is not None else episode.mature_mono_time
      )
      detection = self._emit(
        sample,
        "unwindProgressDeficit",
        "warning",
        0.90,
        "wheel unwind progress remained materially behind the requested trajectory",
        occurred_time,
        previous_phase,
        previous_same_episode,
        unwind_episode=episode,
        episode_start_snapshot=episode.episode_start_snapshot,
        driver_causation=self._unwind_driver_causation(
          episode,
          sample,
          False,
          False,
          False,
          0.0,
        ),
      )
      self.unwind_progress_episode = None
      self.unwind_progress_latched = True
      self.late_unwind_since = None
      return detection
    return None

  @staticmethod
  def _larger_abs(previous: float, value: float) -> float:
    return value if abs(value) > abs(previous) else previous

  @staticmethod
  def _restart_classification(episode: TurnStopEpisode, sample: LateralSample) -> str:
    restart_direction = 1 if sample.steering_rate_deg > 0.0 else -1
    if restart_direction == episode.movement_direction:
      return "sameDirectionTurn"
    toward_center_direction = -1 if episode.dwell_steering_angle_deg > 0.0 else 1
    if restart_direction == toward_center_direction:
      return "unwind"
    return "reversal"

  def _update_turn_stop_turn(self, sample: LateralSample, tracking_active: bool,
                             previous_phase: float,
                             previous_same_episode: bool) -> LateralDetection | None:
    pending = self.pending_turn_stop_candidate
    if pending is not None:
      pending.post_release_progress_deg = max(
        pending.post_release_progress_deg,
        abs(sample.steering_angle_deg - pending.episode.dwell_steering_angle_deg),
      )
      if sample.mono_time >= pending.deadline_mono_time:
        self.pending_turn_stop_candidate = None
        if pending.post_release_progress_deg < TURN_STOP_MIN_POST_PROGRESS_DEG:
          # The wheel never meaningfully moved on after the release; drop the
          # candidate instead of publishing it.
          return None
        return self._emit(
          sample,
          "turnStopTurn",
          "warning",
          0.90,
          "steering motion slowed into a sustained high-demand dwell before releasing",
          pending.episode.release_mono_time,
          pending.previous_phase,
          pending.previous_same_episode,
          turn_stop_episode=pending.episode,
          turn_stop_release_sample=pending.release_sample,
          turn_stop_restart_classification=pending.episode.restart_classification,
          episode_start_snapshot=pending.episode.episode_start_snapshot,
          driver_causation=self._turn_stop_driver_causation(pending.episode),
          turn_stop_post_progress_deg=pending.post_release_progress_deg,
        )

    episode = self.turn_stop_episode
    rate = sample.steering_rate_deg
    abs_rate = abs(rate)
    meaningful_demand = (
      tracking_active
      or abs(sample.reference_rate) > TURN_STOP_TRACKING_ERROR
      or sample.reference_sustained_unwind_scale > 0.2
    )
    if episode is None:
      if (
        abs_rate <= TURN_STOP_MOVING_RATE_DEG_S
        or (
          abs(sample.steering_angle_deg) <= TURN_STOP_ARM_ANGLE_DEG
          and not meaningful_demand
        )
      ):
        return None
      self.turn_stop_episode = TurnStopEpisode(
        state="moving",
        movement_start_mono_time=sample.mono_time,
        movement_direction=1 if rate > 0.0 else -1,
        pre_dwell_peak_signed_rate_deg_s=rate,
        pre_dwell_peak_rate_deg_s=abs_rate,
        driver_involved_before_dwell=sample.driver_confounded,
        episode_start_snapshot=(
          self.episode_start if self.episode_start is not None else sample.mono_time
        ),
        creation_angle_deg=sample.steering_angle_deg,
      )
      return None

    if episode.state == "released":
      assert episode.release_mono_time is not None
      assert episode.release_sample is not None
      episode.release_peak_rate_deg_s = max(episode.release_peak_rate_deg_s, abs_rate)
      episode.driver_involved_after_dwell |= sample.driver_confounded
      if sample.mono_time - episode.release_mono_time < TURN_STOP_RELEASE_OBSERVATION_S:
        return None
      self.turn_stop_episode = None
      return None

    if episode.state in ("moving", "slowing"):
      current_direction = 1 if rate > 0.0 else -1
      if abs_rate > TURN_STOP_MOVING_RATE_DEG_S and current_direction != episode.movement_direction:
        self.turn_stop_episode = TurnStopEpisode(
          state="moving",
          movement_start_mono_time=sample.mono_time,
          movement_direction=current_direction,
          pre_dwell_peak_signed_rate_deg_s=rate,
          pre_dwell_peak_rate_deg_s=abs_rate,
          driver_involved_before_dwell=sample.driver_confounded,
          episode_start_snapshot=(
            self.episode_start if self.episode_start is not None else sample.mono_time
          ),
          creation_angle_deg=sample.steering_angle_deg,
        )
        return None
      episode.driver_involved_before_dwell |= sample.driver_confounded
      if abs_rate > episode.pre_dwell_peak_rate_deg_s:
        episode.pre_dwell_peak_rate_deg_s = abs_rate
        episode.pre_dwell_peak_signed_rate_deg_s = rate
        episode.movement_direction = 1 if rate > 0.0 else -1
      dwell_limit = max(
        TURN_STOP_ABSOLUTE_DWELL_RATE_DEG_S,
        TURN_STOP_DWELL_RATE_FRACTION * episode.pre_dwell_peak_rate_deg_s,
      )
      if abs_rate <= dwell_limit:
        pre_dwell_progress = abs(sample.steering_angle_deg - episode.creation_angle_deg)
        if pre_dwell_progress < TURN_STOP_MIN_PRE_PROGRESS_DEG:
          # Too little travel since this episode's creation to be a real turn
          # stop; keep the movement bookkeeping without opening a dwell.
          episode.state = "slowing"
          return None
        episode.state = "dwell"
        episode.pre_dwell_progress_deg = pre_dwell_progress
        episode.dwell_start_mono_time = sample.mono_time
        episode.dwell_steering_angle_deg = sample.steering_angle_deg
        episode.minimum_dwell_rate_deg_s = abs_rate
        episode.max_request_torque_during_dwell = sample.request_torque
        episode.max_applied_torque_during_dwell = sample.applied_torque
        episode.max_reference_target_during_dwell = sample.reference_target_torque
        episode.tracking_active_during_dwell = tracking_active
        episode.meaningful_reference_during_dwell = meaningful_demand
        episode.driver_involved_during_dwell = sample.driver_confounded
      elif abs_rate < TURN_STOP_SLOWING_RATE_FRACTION * episode.pre_dwell_peak_rate_deg_s:
        episode.state = "slowing"
      else:
        episode.state = "moving"
      return None

    assert episode.dwell_start_mono_time is not None
    dwell_duration = sample.mono_time - episode.dwell_start_mono_time
    if dwell_duration > TURN_STOP_MAX_DWELL_S:
      # A steady curve that outlives the cap is forgotten entirely; a later
      # movement must never publish it.
      self.turn_stop_episode = None
      return None
    released = (
      dwell_duration >= TURN_STOP_MIN_DWELL_S
      and abs_rate > TURN_STOP_MOVING_RATE_DEG_S
      and abs_rate
      >= max(
        TURN_STOP_MOVING_RATE_DEG_S,
        episode.minimum_dwell_rate_deg_s + TURN_STOP_ABSOLUTE_DWELL_RATE_DEG_S,
      )
    )
    if not released:
      # A1: the strong-release condition has precedence; these resets apply
      # only to samples that do NOT satisfy it.
      if abs(sample.steering_angle_deg) <= TURN_STOP_ARM_ANGLE_DEG and not meaningful_demand:
        # Mirrors the entry gate: the wheel drifted back into the center band
        # without any demand keeping the dwell meaningful.
        self.turn_stop_episode = None
        return None
      current_direction = 1 if rate > 0.0 else -1
      if abs_rate > TURN_STOP_MOVING_RATE_DEG_S and current_direction != episode.movement_direction:
        self.turn_stop_episode = TurnStopEpisode(
          state="moving",
          movement_start_mono_time=sample.mono_time,
          movement_direction=current_direction,
          pre_dwell_peak_signed_rate_deg_s=rate,
          pre_dwell_peak_rate_deg_s=abs_rate,
          driver_involved_before_dwell=sample.driver_confounded,
          episode_start_snapshot=(
            self.episode_start if self.episode_start is not None else sample.mono_time
          ),
          creation_angle_deg=sample.steering_angle_deg,
        )
        return None
      episode.minimum_dwell_rate_deg_s = min(episode.minimum_dwell_rate_deg_s, abs_rate)
      if abs(sample.steering_angle_deg) > abs(episode.dwell_steering_angle_deg):
        episode.dwell_steering_angle_deg = sample.steering_angle_deg
      episode.max_request_torque_during_dwell = self._larger_abs(
        episode.max_request_torque_during_dwell,
        sample.request_torque,
      )
      episode.max_applied_torque_during_dwell = self._larger_abs(
        episode.max_applied_torque_during_dwell,
        sample.applied_torque,
      )
      episode.max_reference_target_during_dwell = self._larger_abs(
        episode.max_reference_target_during_dwell,
        sample.reference_target_torque,
      )
      episode.tracking_active_during_dwell &= tracking_active
      episode.meaningful_reference_during_dwell |= meaningful_demand
      episode.driver_involved_during_dwell |= sample.driver_confounded
      return None

    unsupported_hold = (
      not episode.tracking_active_during_dwell
      and not episode.meaningful_reference_during_dwell
      and abs(episode.max_request_torque_during_dwell) <= TURN_STOP_STRONG_TORQUE
      and abs(episode.max_applied_torque_during_dwell) <= TURN_STOP_STRONG_TORQUE
      and abs(episode.max_reference_target_during_dwell) <= TURN_STOP_STRONG_TORQUE
    )
    shape_context = (
      abs(episode.dwell_steering_angle_deg) > TURN_STOP_ARM_ANGLE_DEG
      or abs(episode.max_request_torque_during_dwell) > TURN_STOP_STRONG_TORQUE
      or abs(episode.max_applied_torque_during_dwell) > TURN_STOP_STRONG_TORQUE
    )
    strong = shape_context and not unsupported_hold and (
      abs(episode.dwell_steering_angle_deg) > TURN_STOP_STRONG_ANGLE_DEG
      or dwell_duration > TURN_STOP_STRONG_DWELL_S
      or abs(episode.max_request_torque_during_dwell) > TURN_STOP_STRONG_TORQUE
      or abs(episode.max_applied_torque_during_dwell) > TURN_STOP_STRONG_TORQUE
      or episode.tracking_active_during_dwell
      or episode.meaningful_reference_during_dwell
    )
    classification = self._restart_classification(episode, sample)
    if not strong:
      self.turn_stop_episode = None
      return None
    episode.state = "released"
    episode.release_mono_time = sample.mono_time
    episode.release_sample = sample
    episode.release_peak_rate_deg_s = abs_rate
    episode.restart_classification = classification
    episode.driver_involved_after_dwell = sample.driver_confounded
    score = (
      10.0 * dwell_duration
      + abs(episode.dwell_steering_angle_deg) / TURN_STOP_STRONG_ANGLE_DEG
      + 2.0 * max(
        abs(episode.max_request_torque_during_dwell),
        abs(episode.max_applied_torque_during_dwell),
        abs(episode.max_reference_target_during_dwell),
      )
      + episode.release_peak_rate_deg_s / 200.0
    )
    release_progress = abs(sample.steering_angle_deg - episode.dwell_steering_angle_deg)
    pending = self.pending_turn_stop_candidate
    if pending is None:
      self.pending_turn_stop_candidate = PendingTurnStopCandidate(
        episode=episode,
        release_sample=sample,
        score=score,
        deadline_mono_time=sample.mono_time + TURN_STOP_CLUSTER_SELECTION_S,
        previous_phase=previous_phase,
        previous_same_episode=previous_same_episode,
        post_release_progress_deg=release_progress,
      )
    elif sample.mono_time <= pending.deadline_mono_time and score > pending.score:
      # Preserve the strongest dwell/release in a short sawtooth cluster.
      # Extend only enough to finish that release's evidence observation; the
      # original selection horizon otherwise remains fixed.
      self.pending_turn_stop_candidate = PendingTurnStopCandidate(
        episode=episode,
        release_sample=sample,
        score=score,
        deadline_mono_time=max(
          pending.deadline_mono_time,
          sample.mono_time + TURN_STOP_RELEASE_OBSERVATION_S,
        ),
        previous_phase=previous_phase,
        previous_same_episode=previous_same_episode,
        post_release_progress_deg=release_progress,
      )
    return None

  def update(self, sample: LateralSample) -> LateralDetection | None:
    if self.last_sample_mono_time is not None and (
      sample.mono_time < self.last_sample_mono_time
      or sample.mono_time - self.last_sample_mono_time > SAMPLE_GAP_RESET_S
    ):
      self.reset_discontinuity()
    self.last_sample_mono_time = sample.mono_time

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
    if detection is not None:
      # A higher-priority detector (crossing/handoff) claimed this sample's
      # emission slot, so _update_unwind_progress never runs -- but an
      # active episode's crown-neutral/rebound bookkeeping must still
      # advance every sample regardless of which detector emits, or it
      # silently skips real signal changes on shared samples (fix: decouple
      # state advancement from emission priority).
      if self.unwind_progress_episode is not None:
        self._advance_unwind_episode(self.unwind_progress_episode, sample)
    else:
      detection = self._update_unwind_progress(
        sample,
        previous_phase,
        previous_same_episode,
      )
    if detection is None:
      detection = self._update_turn_stop_turn(
        sample,
        tracking_active,
        previous_phase,
        previous_same_episode,
      )

    stall_releases: list[StallRelease] = []
    late_unwind_duration = 0.0
    unwind_expected = (
      sample.reference_sustained_unwind_scale > 0.6
      and sample.unwind_effective_phase > 0.6
      and abs(sample.steering_angle_deg) > 25.0
      and abs(sample.reference_rate) > 0.2
      and not self.unwind_progress_latched
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
      self.unwind_progress_episode = None
      self.unwind_progress_latched = True

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
