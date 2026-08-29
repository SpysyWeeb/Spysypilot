from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_lead import lead_present
from openpilot.selfdrive.controls.lib.stop_helpers import (MODEL_INVALID_RELEASE_S, STOP_COMMIT_MAX_DISTANCE, STOP_SAMPLE_MIN_CONFIDENCE,
                                                           StopObservation, model_complete_and_finite, observe_model_stop)

# Requests Experimental mode for a confirmed, lead-free model stop so the e2e candidate can plan the
# approach; it never chooses a speed, an acceleration, a brake command or a stop point.
STOP_FILTER_TIME_CONSTANT_S = 0.30
STOP_ENTRY_FILTER_THRESHOLD = 0.55
STOP_RELEASE_FILTER_THRESHOLD = 0.25
STOP_ENTRY_DEBOUNCE_S = 0.20
STOP_RELEASE_HYSTERESIS_S = 0.75
MODE_MIN_LATCH_S = 1.0
STOP_INTENT_HOLD_S = 4.0
STANDSTILL_MIN_LATCH_S = 1.0
RESUME_RELEASE_SPEED = 0.8
POST_STOP_SUPPRESS_S = 2.0
DRIVER_OVERRIDE_SUPPRESS_S = 2.0
LEAD_RELEASE_HYSTERESIS_S = 3.0


class ConditionalExperimentalMode:
  def __init__(self, control_dt=DT_CTRL, model_dt=DT_MDL):
    self.control_dt = control_dt
    self.model_dt = model_dt
    self.intent_filter = FirstOrderFilter(0.0, STOP_FILTER_TIME_CONSTANT_S, model_dt)
    self.reset()

  def reset(self):
    self.experimental_mode = False
    self.intent_filter.x = 0.0
    self.last_observation = StopObservation()
    self._entry_elapsed = 0.0
    self._clear_elapsed = 0.0
    self._active_elapsed = 0.0
    self._intent_hold_remaining = 0.0
    self._standstill_elapsed = 0.0
    self._standstill_seen = False
    self._invalid_elapsed = 0.0
    self._lead_veto_remaining = 0.0
    self._lead_release_active = False
    self._post_stop_remaining = 0.0
    self._override_remaining = 0.0
    self._model_complete = True

  def _clear_evidence(self):
    self.intent_filter.x = 0.0
    self.last_observation = StopObservation()
    self._entry_elapsed = 0.0
    self._clear_elapsed = 0.0
    self._intent_hold_remaining = 0.0
    self._lead_release_active = False

  def _deactivate(self, suppress_for):
    self.experimental_mode = False
    self._active_elapsed = 0.0
    self._standstill_elapsed = 0.0
    self._standstill_seen = False
    self._post_stop_remaining = max(self._post_stop_remaining, suppress_for)
    self._clear_evidence()

  def _update_model_evidence(self, model, car_state, radar_state, model_valid):
    obs = observe_model_stop(model, car_state, radar_state) if model_valid else StopObservation()
    self.last_observation = obs
    self._model_complete = obs.complete or not model_valid

    # a relevant lead vetoes a new handoff and starts a grace; during it one strict frame may mint a revocable release
    # if both raw leads are gone and every model lead hypothesis is outside the stop corridor
    self._lead_veto_remaining = LEAD_RELEASE_HYSTERESIS_S if obs.relevant_lead else max(self._lead_veto_remaining - self.model_dt, 0.0)
    release_safe = (self._lead_veto_remaining > 0.0 and not obs.lead_present and not obs.committed_turn
                    and obs.path_end is not None and obs.path_end > 0.0 and obs.corridor_clear)
    if not release_safe:
      self._lead_release_active = False
    elif self._lead_release_active:
      self._lead_release_active = obs.strict_stop or obs.early_stop
    elif not self.experimental_mode and obs.strict_stop and obs.path_end < STOP_COMMIT_MAX_DISTANCE:
      self._lead_release_active = True
    entry_veto = (self._lead_veto_remaining > 0.0 and not self._lead_release_active) or obs.committed_turn

    confidence = obs.confidence if self.experimental_mode or not entry_veto else 0.0
    raw_stop = confidence >= STOP_SAMPLE_MIN_CONFIDENCE
    if raw_stop:
      self._intent_hold_remaining = STOP_INTENT_HOLD_S
    # a filter-only hint shortens entry, but cannot sustain an active latch after qualifying evidence disappears
    filtered = self.intent_filter.update(confidence if not self.experimental_mode or raw_stop else 0.0)

    if self.experimental_mode:
      self._entry_elapsed = 0.0
      self._clear_elapsed = self._clear_elapsed + self.model_dt if not raw_stop and filtered <= STOP_RELEASE_FILTER_THRESHOLD else 0.0
    else:
      self._clear_elapsed = 0.0
      entry_ready = raw_stop and filtered >= STOP_ENTRY_FILTER_THRESHOLD and self._post_stop_remaining <= 0.0
      self._entry_elapsed = self._entry_elapsed + self.model_dt if entry_ready else 0.0

  def update(self, model, car_state, radar_state, *, controls_enabled, model_updated, model_valid, radar_valid=True):
    if not controls_enabled:
      self.reset()
      return False

    self._post_stop_remaining = max(self._post_stop_remaining - self.control_dt, 0.0)
    self._intent_hold_remaining = max(self._intent_hold_remaining - self.control_dt, 0.0)

    if car_state.gasPressed or car_state.brakePressed:
      self._override_remaining = DRIVER_OVERRIDE_SUPPRESS_S
      self._deactivate(0.0)
      return False
    if self._override_remaining > 0.0:
      self._override_remaining = max(self._override_remaining - self.control_dt, 0.0)
      self._clear_evidence()
      return False

    if model_updated and model_valid and not model_complete_and_finite(model):
      self._model_complete = False
    self._invalid_elapsed = 0.0 if model_valid and self._model_complete else self._invalid_elapsed + self.control_dt
    if lead_present(radar_state) or not model_valid or not radar_valid:
      # a raw lead on any control tick revokes a pending recent-lead release; entry evidence itself is only judged on model frames
      self._lead_release_active = False

    if model_updated:
      self._update_model_evidence(model, car_state, radar_state, model_valid and radar_valid)

    if not self.experimental_mode and self._invalid_elapsed >= MODEL_INVALID_RELEASE_S:
      self._clear_evidence()

    if self.experimental_mode:
      self._active_elapsed += self.control_dt
      if car_state.standstill:
        self._standstill_seen = True
        self._standstill_elapsed += self.control_dt
      resumed_after_stop = self._standstill_seen and car_state.vEgo >= RESUME_RELEASE_SPEED
      standstill_latch_satisfied = not self._standstill_seen or self._standstill_elapsed >= STANDSTILL_MIN_LATCH_S
      stable_clear = (self._active_elapsed >= MODE_MIN_LATCH_S and self._intent_hold_remaining <= 0.0
                      and standstill_latch_satisfied and self._clear_elapsed >= STOP_RELEASE_HYSTERESIS_S)
      if resumed_after_stop or stable_clear:
        self._deactivate(POST_STOP_SUPPRESS_S)
      elif self._invalid_elapsed >= MODEL_INVALID_RELEASE_S:
        self._deactivate(0.0)
    elif (self._post_stop_remaining <= 0.0 and self._invalid_elapsed < MODEL_INVALID_RELEASE_S
          and self._entry_elapsed >= STOP_ENTRY_DEBOUNCE_S):
      self.experimental_mode = True
      self._active_elapsed = 0.0
      self._standstill_elapsed = 0.0
      self._standstill_seen = False
      self._entry_elapsed = 0.0
    return self.experimental_mode
