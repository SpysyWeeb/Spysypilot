"""Conditional Experimental Mode for BLoTv2.

This module detects a generic, lead-free model stop intent and requests the
existing Experimental longitudinal strategy. It never chooses a target speed,
acceleration, brake command, or stop point.

The deployed model does not publish traffic-light or stop-sign classes. Its
available stop evidence is ``action.shouldStop`` plus the finite 10-second
position/velocity trajectory. Confidence below is therefore confidence in a
temporally persistent stop *intent*, not semantic confidence that an object is
a red light or stop sign.
"""

from dataclasses import dataclass
import math
from typing import Any

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL, DT_MDL


# Model-stop observation tuning. All distances are meters and speeds are m/s.
STOP_PREDICTION_HORIZON_S = 5.0
STOP_PATH_MIN_DISTANCE_M = 4.0
STOP_TERMINAL_SPEED_MAX = 1.0
STOP_FALLBACK_ACCEL_MAX = -0.5
STOP_TRAJECTORY_MIN_POINTS = 2

# The strict detector above is intentionally retained for low-speed and
# ambiguous scenes. At urban-road speed, waiting for the model's ten-second
# terminal velocity to reach almost zero can leave less than a comfortable
# stopping distance. The early tier recognizes a sustained *shape* of model
# intent instead: the path has contracted inside a comfort-deceleration
# envelope, the terminal velocity has fallen substantially, and the action is
# already braking. It is limited to straight, unsignaled, lead-free approaches
# by the scene guards below and by the existing entry veto.
STOP_EARLY_MIN_SPEED = 13.0
STOP_EARLY_COMFORT_DECEL = 1.3
STOP_EARLY_RESPONSE_BUFFER_S = 0.5
STOP_EARLY_TERMINAL_SPEED_RATIO = 0.35
STOP_EARLY_TERMINAL_SPEED_MAX = 6.5
STOP_EARLY_DESIRED_ACCEL_MAX = -0.5
STOP_EARLY_MAX_LATERAL_ACCEL = 1.0
STOP_EARLY_MAX_HEADING_CHANGE = math.radians(20.0)

# Weaker evidence may charge the temporal filter but is deliberately below
# STOP_SAMPLE_MIN_CONFIDENCE, so it can never request Experimental by itself.
# This avoids spending the full filter delay after a gradually developing
# high-speed stop becomes strong enough to qualify.
STOP_EARLY_HINT_HORIZON_S = 8.0
STOP_EARLY_HINT_TERMINAL_SPEED_RATIO = 0.55
STOP_EARLY_HINT_DESIRED_ACCEL_MAX = -0.25

# Evidence strengths and temporal qualification.
STOP_DIRECT_CONFIDENCE = 1.0
STOP_TRAJECTORY_CONFIDENCE = 0.85
STOP_EARLY_CONFIDENCE = 0.80
STOP_FALLBACK_CONFIDENCE = 0.70
STOP_EARLY_HINT_CONFIDENCE = 0.45
STOP_SAMPLE_MIN_CONFIDENCE = 0.70
STOP_FILTER_TIME_CONSTANT_S = 0.30
STOP_ENTRY_FILTER_THRESHOLD = 0.55
STOP_RELEASE_FILTER_THRESHOLD = 0.25
STOP_ENTRY_DEBOUNCE_S = 0.20
STOP_RELEASE_HYSTERESIS_S = 0.75

# Latching and release behavior.
MODE_MIN_LATCH_S = 1.0
STANDSTILL_MIN_LATCH_S = 1.0
MODEL_INVALID_RELEASE_S = 0.50
RESUME_RELEASE_SPEED = 0.8
POST_STOP_SUPPRESS_S = 2.0
DRIVER_OVERRIDE_SUPPRESS_S = 2.0

# Entry vetoes keep existing lead and committed-turn behavior in charge.
LEAD_RELEVANCE_MIN_DISTANCE_M = 35.0
LEAD_RELEVANCE_TIME_S = 3.5
LEAD_PATH_MARGIN_M = 10.0
LEAD_RELEASE_HYSTERESIS_S = 3.0
COMMITTED_TURN_MAX_SPEED = 8.0
COMMITTED_TURN_MIN_STEERING_DEG = 35.0
COMMITTED_TURN_MIN_CURVATURE = 0.04


@dataclass(frozen=True)
class StopIntentObservation:
  confidence: float = 0.0
  reason: str = "none"
  path_end_m: float | None = None
  terminal_speed: float | None = None
  relevant_lead: bool = False
  committed_turn: bool = False


def _last_finite(values: Any, minimum_length: int = 1) -> float | None:
  try:
    if len(values) < minimum_length:
      return None
    value = float(values[-1])
  except (AttributeError, IndexError, TypeError, ValueError):
    return None
  return value if math.isfinite(value) else None


def _finite_float(value: Any, default: float = 0.0) -> float:
  try:
    parsed = float(value)
  except (TypeError, ValueError):
    return default
  return parsed if math.isfinite(parsed) else default


def _optional_finite_float(value: Any) -> float | None:
  try:
    parsed = float(value)
  except (TypeError, ValueError):
    return None
  return parsed if math.isfinite(parsed) else None


def observe_model_stop_intent(model: Any, car_state: Any, radar_state: Any) -> StopIntentObservation:
  """Translate current BLoTv2 cereal messages into one stop-intent sample."""
  v_ego = max(_finite_float(getattr(car_state, "vEgo", 0.0)), 0.0)
  action = getattr(model, "action", None)
  should_stop = bool(getattr(action, "shouldStop", False))
  desired_accel = _finite_float(getattr(action, "desiredAcceleration", 0.0))
  desired_curvature = _optional_finite_float(getattr(action, "desiredCurvature", None))

  position = getattr(model, "position", None)
  velocity = getattr(model, "velocity", None)
  orientation = getattr(model, "orientation", None)
  path_end_m = _last_finite(getattr(position, "x", ()), STOP_TRAJECTORY_MIN_POINTS)
  terminal_speed = _last_finite(getattr(velocity, "x", ()), STOP_TRAJECTORY_MIN_POINTS)
  terminal_heading = _last_finite(getattr(orientation, "z", ()), STOP_TRAJECTORY_MIN_POINTS)

  blinker = bool(getattr(car_state, "leftBlinker", False) or getattr(car_state, "rightBlinker", False))
  lateral_accel = abs(desired_curvature) * v_ego ** 2 if desired_curvature is not None else None
  straight_approach = bool(
    not blinker and
    lateral_accel is not None and lateral_accel <= STOP_EARLY_MAX_LATERAL_ACCEL and
    terminal_heading is not None and abs(terminal_heading) <= STOP_EARLY_MAX_HEADING_CHANGE
  )

  distance_limit = max(STOP_PATH_MIN_DISTANCE_M, v_ego * STOP_PREDICTION_HORIZON_S)
  path_in_horizon = path_end_m is not None and 0.0 < path_end_m <= distance_limit
  trajectory_stop = path_in_horizon and terminal_speed is not None and terminal_speed <= STOP_TERMINAL_SPEED_MAX
  fallback_stop = path_in_horizon and terminal_speed is None and desired_accel <= STOP_FALLBACK_ACCEL_MAX

  early_distance_limit = (
    v_ego ** 2 / (2.0 * STOP_EARLY_COMFORT_DECEL) + v_ego * STOP_EARLY_RESPONSE_BUFFER_S
  )
  early_path = path_end_m is not None and 0.0 < path_end_m <= early_distance_limit
  early_stop = bool(
    v_ego >= STOP_EARLY_MIN_SPEED and straight_approach and early_path and
    terminal_speed is not None and terminal_speed <= STOP_EARLY_TERMINAL_SPEED_MAX and
    terminal_speed <= v_ego * STOP_EARLY_TERMINAL_SPEED_RATIO and
    desired_accel <= STOP_EARLY_DESIRED_ACCEL_MAX
  )

  early_hint_path = path_end_m is not None and 0.0 < path_end_m <= v_ego * STOP_EARLY_HINT_HORIZON_S
  early_hint = bool(
    v_ego >= STOP_EARLY_MIN_SPEED and straight_approach and early_hint_path and
    terminal_speed is not None and terminal_speed <= v_ego * STOP_EARLY_HINT_TERMINAL_SPEED_RATIO and
    desired_accel <= STOP_EARLY_HINT_DESIRED_ACCEL_MAX
  )

  confidence = 0.0
  reason = "none"
  if should_stop:
    confidence = STOP_DIRECT_CONFIDENCE
    reason = "shouldStop"
  elif trajectory_stop:
    confidence = STOP_TRAJECTORY_CONFIDENCE
    reason = "trajectory"
  elif early_stop:
    confidence = STOP_EARLY_CONFIDENCE
    reason = "earlyTrajectory"
  elif fallback_stop:
    confidence = STOP_FALLBACK_CONFIDENCE
    reason = "path+braking"
  elif early_hint:
    confidence = STOP_EARLY_HINT_CONFIDENCE
    reason = "earlyHint"

  lead = getattr(radar_state, "leadOne", None)
  lead_present = bool(getattr(lead, "present", getattr(lead, "status", False)))
  lead_distance = _finite_float(getattr(lead, "dRel", math.inf), math.inf)
  lead_limit = max(LEAD_RELEVANCE_MIN_DISTANCE_M, v_ego * LEAD_RELEVANCE_TIME_S)
  if path_end_m is not None:
    lead_limit = max(lead_limit, path_end_m + LEAD_PATH_MARGIN_M)
  relevant_lead = lead_present and 0.0 <= lead_distance <= lead_limit

  steering_angle = abs(_finite_float(getattr(car_state, "steeringAngleDeg", 0.0)))
  committed_turn = bool(
    blinker and v_ego <= COMMITTED_TURN_MAX_SPEED and
    (steering_angle >= COMMITTED_TURN_MIN_STEERING_DEG or
     (desired_curvature is not None and abs(desired_curvature) >= COMMITTED_TURN_MIN_CURVATURE))
  )

  return StopIntentObservation(confidence, reason, path_end_m, terminal_speed, relevant_lead, committed_turn)


class ConditionalExperimentalMode:
  """Filtered and latched owner of BLoTv2's conditional mode request."""

  def __init__(self, control_dt: float = DT_CTRL, model_dt: float = DT_MDL):
    self.control_dt = control_dt
    self.model_dt = model_dt
    self.intent_filter = FirstOrderFilter(0.0, STOP_FILTER_TIME_CONSTANT_S, model_dt)
    self.reset()

  def reset(self) -> None:
    self.experimental_mode = False
    self.intent_filter.x = 0.0
    self.last_observation = StopIntentObservation()
    self._entry_elapsed = 0.0
    self._clear_elapsed = 0.0
    self._active_elapsed = 0.0
    self._standstill_elapsed = 0.0
    self._standstill_seen = False
    self._invalid_elapsed = 0.0
    self._lead_veto_remaining = 0.0
    self._post_stop_remaining = 0.0
    self._override_remaining = 0.0

  @property
  def driver_override_active(self) -> bool:
    return self._override_remaining > 0.0

  @property
  def stop_latched(self) -> bool:
    return self.experimental_mode

  def _clear_evidence(self) -> None:
    self.intent_filter.x = 0.0
    self.last_observation = StopIntentObservation()
    self._entry_elapsed = 0.0
    self._clear_elapsed = 0.0

  def _deactivate(self, suppress_for: float) -> None:
    self.experimental_mode = False
    self._active_elapsed = 0.0
    self._standstill_elapsed = 0.0
    self._standstill_seen = False
    self._post_stop_remaining = max(self._post_stop_remaining, suppress_for)
    self._clear_evidence()

  def _update_model_evidence(self, model: Any, car_state: Any, radar_state: Any, model_valid: bool) -> None:
    observation = observe_model_stop_intent(model, car_state, radar_state) if model_valid else StopIntentObservation()
    self.last_observation = observation

    # A short radar dropout must not transfer a lead-owned slowdown to CEM.
    # Leads and committed turns veto only a new handoff. Once a model stop has
    # qualified, later scene churn cannot flick the mode off while that model
    # stop evidence itself remains present.
    if observation.relevant_lead:
      self._lead_veto_remaining = LEAD_RELEASE_HYSTERESIS_S
    else:
      self._lead_veto_remaining = max(self._lead_veto_remaining - self.model_dt, 0.0)
    entry_veto = self._lead_veto_remaining > 0.0 or observation.committed_turn
    confidence = observation.confidence if self.experimental_mode or not entry_veto else 0.0
    raw_stop = confidence >= STOP_SAMPLE_MIN_CONFIDENCE
    # A filter-only hint can shorten later entry latency, but cannot sustain an
    # already active latch indefinitely after qualifying evidence disappears.
    filter_input = confidence if not self.experimental_mode or raw_stop else 0.0
    filtered_confidence = self.intent_filter.update(filter_input)

    if self.experimental_mode:
      self._entry_elapsed = 0.0
      if not raw_stop and filtered_confidence <= STOP_RELEASE_FILTER_THRESHOLD:
        self._clear_elapsed += self.model_dt
      else:
        self._clear_elapsed = 0.0
    else:
      self._clear_elapsed = 0.0
      entry_ready = raw_stop and filtered_confidence >= STOP_ENTRY_FILTER_THRESHOLD
      if entry_ready and self._post_stop_remaining <= 0.0:
        self._entry_elapsed += self.model_dt
      else:
        self._entry_elapsed = 0.0

  def update(self, model: Any, car_state: Any, radar_state: Any, *,
             controls_enabled: bool, model_updated: bool, model_valid: bool) -> bool:
    """Return the conditional Experimental request for the current control tick."""
    if not controls_enabled:
      self.reset()
      return False

    self._post_stop_remaining = max(self._post_stop_remaining - self.control_dt, 0.0)

    driver_override = bool(getattr(car_state, "gasPressed", False) or getattr(car_state, "brakePressed", False))
    if driver_override:
      self._override_remaining = DRIVER_OVERRIDE_SUPPRESS_S
      self._deactivate(0.0)
      return False

    if self._override_remaining > 0.0:
      self._override_remaining = max(self._override_remaining - self.control_dt, 0.0)
      self._clear_evidence()
      return False

    if model_valid:
      self._invalid_elapsed = 0.0
    else:
      self._invalid_elapsed += self.control_dt

    if model_updated:
      self._update_model_evidence(model, car_state, radar_state, model_valid)

    if not self.experimental_mode and self._invalid_elapsed >= MODEL_INVALID_RELEASE_S:
      self._clear_evidence()

    if self.experimental_mode:
      self._active_elapsed += self.control_dt
      standstill = bool(getattr(car_state, "standstill", False))
      if standstill:
        self._standstill_seen = True
        self._standstill_elapsed += self.control_dt

      v_ego = max(_finite_float(getattr(car_state, "vEgo", 0.0)), 0.0)
      resumed_after_stop = self._standstill_seen and v_ego >= RESUME_RELEASE_SPEED
      invalid_too_long = self._invalid_elapsed >= MODEL_INVALID_RELEASE_S
      standstill_latch_satisfied = not self._standstill_seen or self._standstill_elapsed >= STANDSTILL_MIN_LATCH_S
      stable_clear = (
        self._active_elapsed >= MODE_MIN_LATCH_S and
        standstill_latch_satisfied and
        self._clear_elapsed >= STOP_RELEASE_HYSTERESIS_S
      )

      if resumed_after_stop or stable_clear:
        self._deactivate(POST_STOP_SUPPRESS_S)
      elif invalid_too_long:
        self._deactivate(0.0)
    elif (
      self._post_stop_remaining <= 0.0 and
      self._invalid_elapsed < MODEL_INVALID_RELEASE_S and
      self._entry_elapsed >= STOP_ENTRY_DEBOUNCE_S
    ):
      self.experimental_mode = True
      self._active_elapsed = 0.0
      self._standstill_elapsed = 0.0
      self._standstill_seen = False
      self._entry_elapsed = 0.0

    return self.experimental_mode
