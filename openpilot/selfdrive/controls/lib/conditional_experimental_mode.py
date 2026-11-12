import math

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_lead import lead_present
from openpilot.selfdrive.controls.lib.stop_helpers import (LEAD_CRAWL_SPEED, MODEL_INVALID_RELEASE_S, STOP_COMMIT_MAX_DISTANCE,
                                                           STOP_EARLY_HINT_ENTRY_CONFIDENCE, STOP_SAMPLE_MIN_CONFIDENCE,
                                                           STOP_TRAJECTORY_CONFIDENCE, StopObservation, model_complete_and_finite,
                                                           observe_model_stop)

# Requests Experimental mode for a confirmed model stop that no lead owns so the e2e candidate can plan the
# approach; it never chooses a speed, an acceleration, a brake command or a stop point.
STOP_FILTER_TIME_CONSTANT_S = 0.30
STOP_ENTRY_FILTER_THRESHOLD = 0.55
STOP_RELEASE_FILTER_THRESHOLD = 0.25
STOP_ENTRY_DEBOUNCE_S = 0.20
STOP_RELEASE_HYSTERESIS_S = 0.75
MODE_MIN_LATCH_S = 1.0
STOP_INTENT_HOLD_S = 4.0
STOP_HOLD_MIN_CONFIDENCE = 0.8  # while active, only early/strict/direct tiers refresh the hold: route 0x2b, lone 0.70-confidence
                                # hint frames refreshed it 35+ times and kept the mode searching 17.5 s past the light
CLEAR_CANCEL_S = 0.75           # s of sustained open road (a long, moving model path) that cancels the hold outright:
                                # the road being clear is positive evidence the stop is behind us, not mere absence
STANDSTILL_MIN_LATCH_S = 1.0
RESUME_RELEASE_SPEED = 0.8
POST_STOP_SUPPRESS_S = 2.0
DRIVER_OVERRIDE_SUPPRESS_S = 2.0
LEAD_RELEASE_HYSTERESIS_S = 0.5  # s, the recent-lead guard while rolling (D48). 3.0 held the model's stop out of the switch for 3 s
                                 # after the lead was lost: routes 0x207 t=346.9-350.4 (a red light after a bend) and 0x201
                                 # t=115.7-120.3 (traffic slowing ahead), the driver braking each time; 3.0 restores the guard as it
                                 # was before D48. It counts down on model frames: the stop evidence of the 10 frames after the
                                 # lead's last one is zeroed (60 at 3.0; the float countdown leaves a residue on the last frame)
LEAD_RELEASE_HYSTERESIS_AT_REST_S = 3.0  # s, the guard a lead armed with the car below GO_ROLL_SPEED (D48). At rest the model's stop bit
                                         # lags the plan opening by about a second once the queue launches, and 0.5 s let that stale bit
                                         # switch the mode at the launch (route 0x1f3 t=22.8-24.0, as if engaged)
# D49, a lead past the model's stop point: LEAD_BEYOND_STOP_MEMORY_S = 0.0 switches it off. Route 0x1fc t=1451.9-1455.8: a yellow light,
# the lead 80-86 m out past the junction; the plan alternated between a strict stop 42-55 m out and a longer one 79-103 m out that made
# the lead relevant, and every longer frame re-armed the guard
LEAD_BEYOND_STOP_MEMORY_S = 1.0  # s, the window over which the model's nearest stop point is remembered (frames with stop evidence)
LEAD_BEYOND_STOP_MARGIN = 5.0    # m, a lead moving at LEAD_CRAWL_SPEED or more and farther than this past that point is no relevant
                                 # lead for the veto: it has gone through the place the model stops
GO_ROLL_SPEED = 1.0  # m/s, below this the open road alone is a go (a green at rest or a crawl); above it the model must launch
GO_ACCEL = 0.5       # m/s^2
GO_CONFIRM_S = 0.5   # s, longer than the model's brief open-road flashes at rest


class ConditionalExperimentalMode:
  def __init__(self, control_dt=DT_CTRL, model_dt=DT_MDL):
    self.control_dt = control_dt
    self.model_dt = model_dt
    self.intent_filter = FirstOrderFilter(0.0, STOP_FILTER_TIME_CONSTANT_S, model_dt)
    self.reset()

  def reset(self):
    self._open_elapsed = 0.0
    self._go_elapsed = 0.0
    self.experimental_mode = False
    self.intent_filter.x = 0.0
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
    self._exit_revocable = False
    self._override_remaining = 0.0
    self._model_complete = True
    self._stop_points = []

  def _clear_evidence(self):
    self.intent_filter.x = 0.0
    self._open_elapsed = 0.0
    self._go_elapsed = 0.0
    self._entry_elapsed = 0.0
    self._clear_elapsed = 0.0
    self._intent_hold_remaining = 0.0
    self._lead_release_active = False
    self._stop_points = []

  def _lead_beyond_stop(self, obs, radar_state, v_ego):
    # D49: remembers the model's stop point on each frame with stop evidence (any tier) over LEAD_BEYOND_STOP_MEMORY_S. A relevant lead
    # is beyond the stop when every present radar lead can be read, is moving (not a queue: a stopped or creeping car past the model's
    # stop point is what the model stops behind, D42) and is more than LEAD_BEYOND_STOP_MARGIN past the nearest remembered point. At a
    # crawl a relevant lead keeps its veto, as before (a queue's creep and launch are the lead logic's)
    frames = round(LEAD_BEYOND_STOP_MEMORY_S / self.model_dt)
    if frames <= 0:
      self._stop_points = []
      return False
    has_point = obs.confidence > 0.0 and obs.path_end is not None and math.isfinite(obs.path_end) and obs.path_end > 0.0
    self._stop_points = (self._stop_points + [obs.path_end if has_point else None])[-frames:]
    points = [p for p in self._stop_points if p is not None]
    if not obs.relevant_lead or not points or not math.isfinite(v_ego) or v_ego < GO_ROLL_SPEED:
      return False
    nearest = min(points)
    leads = [lead for lead in (radar_state.leadOne, radar_state.leadTwo) if lead.present]
    return all(math.isfinite(lead.dRel) and math.isfinite(lead.vLead) and lead.vLead >= LEAD_CRAWL_SPEED
               and lead.dRel > nearest + LEAD_BEYOND_STOP_MARGIN for lead in leads)

  def _deactivate(self, suppress_for):
    self.experimental_mode = False
    self._active_elapsed = 0.0
    self._standstill_elapsed = 0.0
    self._standstill_seen = False
    self._post_stop_remaining = max(self._post_stop_remaining, suppress_for)
    self._exit_revocable = False
    self._clear_evidence()

  @property
  def _entry_suppressed(self):
    return self._post_stop_remaining > 0.0 and not self._exit_revocable

  def _update_model_evidence(self, model, car_state, radar_state, model_valid, radar_valid):
    obs = observe_model_stop(model, car_state, radar_state) if model_valid else StopObservation()
    # completeness is a property of the model frame alone: judging it on radar validity too certified a garbage
    # trajectory as complete on every radar dropout and the MODEL_INVALID_RELEASE_S release never fired
    self._model_complete = obs.complete or not model_valid
    # every tier below is read against the radar leads, so an invalid radar leaves the frame with no stop evidence
    # to judge: it carries the empty observation, exactly as an invalid model does
    obs = obs if radar_valid else StopObservation()

    # a relevant lead vetoes a new handoff and starts a grace (D48: LEAD_RELEASE_HYSTERESIS_S rolling, the longer
    # LEAD_RELEASE_HYSTERESIS_AT_REST_S with the car below GO_ROLL_SPEED); on a near stop (strict, direct) at speed, a tracked lead that
    # does not own it (since D46 any lead but a queue or an unreadable one) ends the veto at once. A moving lead past the model's nearest
    # stop point of the last second is no relevant lead for the veto (D49): it neither vetoes nor re-arms the grace, nor owns the stop,
    # and on a near stop it still ends the veto at once, as D46 has it. During the grace one strict frame may mint a revocable release
    # if both raw leads are gone and every model lead hypothesis is outside the stop corridor
    beyond = self._lead_beyond_stop(obs, radar_state, car_state.vEgo)
    relevant_lead = obs.relevant_lead and not beyond
    lead_owns_stop = obs.lead_owns_stop and not beyond
    near_stop = obs.should_stop or obs.strict_stop
    if relevant_lead and (lead_owns_stop or not near_stop or car_state.vEgo < GO_ROLL_SPEED):
      self._lead_veto_remaining = LEAD_RELEASE_HYSTERESIS_S if car_state.vEgo >= GO_ROLL_SPEED else LEAD_RELEASE_HYSTERESIS_AT_REST_S
    elif obs.relevant_lead and near_stop:
      # a lead beyond the stop reads as moving, readable and the car rolling, so on a near stop this is D46's case whatever D49 says
      self._lead_veto_remaining = 0.0
    else:
      self._lead_veto_remaining = max(self._lead_veto_remaining - self.model_dt, 0.0)
    release_safe = (self._lead_veto_remaining > 0.0 and not obs.lead_present and not obs.committed_turn
                    and obs.path_end is not None and obs.path_end > 0.0 and obs.corridor_clear)
    if not release_safe:
      self._lead_release_active = False
    elif self._lead_release_active:
      self._lead_release_active = obs.strict_stop or obs.early_stop
    elif not self.experimental_mode and obs.strict_stop and obs.path_end < STOP_COMMIT_MAX_DISTANCE:
      self._lead_release_active = True
    entry_veto = (self._lead_veto_remaining > 0.0 and not self._lead_release_active) or obs.committed_turn

    # below the early tiers' speed, with no relevant radar lead (the observation's own reading, D49 aside), a short plan rolling on no
    # faster than the car and at 4.5 m/s or less, the model not asking to speed up (D50)
    slow_plan = obs.slow_plan and not obs.relevant_lead
    evidence = obs.confidence
    if not self.experimental_mode and (obs.slowing_approach or slow_plan):
      # below the early tiers' speed, a slowing, straight plan with the model braking enters at the hint's confidence (D47), and so
      # does a slow, short plan with nobody ahead (D50). Neither is a near stop, so a relevant lead vetoes them (one moving past the
      # model's stop excepted, D49). Entry evidence only: below the hold's 0.8 neither could refresh the hold of an active mode; what
      # holds a mode they entered is below
      evidence = max(evidence, STOP_EARLY_HINT_ENTRY_CONFIDENCE)
    confidence = evidence if self.experimental_mode or not entry_veto else 0.0
    if self.experimental_mode and (obs.slowing or slow_plan):
      # a plan still shedding most of the speed has not reached the stop, even when it relaunches before its end, and a plan still
      # rolling slowly to a short end with nobody ahead has not left it (D50: its end reads as open road at 3 m/s and more, and the
      # open-road count would cancel the hold a second after entry and flicker the mode off and on): trajectory evidence, so neither
      # the hold nor the clear hysteresis runs out and the open-road count cannot start
      confidence = max(confidence, STOP_TRAJECTORY_CONFIDENCE)
    raw_stop = confidence >= STOP_SAMPLE_MIN_CONFIDENCE
    if raw_stop and (not self.experimental_mode or confidence >= STOP_HOLD_MIN_CONFIDENCE):
      # entry evidence arms the hold; once active, a lone borderline hint may not keep the search alive
      self._intent_hold_remaining = STOP_INTENT_HOLD_S
    # sustained open road is positive clear evidence: it cancels the hold, and the ordinary clear hysteresis finishes the exit
    self._open_elapsed = self._open_elapsed + self.model_dt if obs.release_open and not raw_stop else 0.0
    if self._open_elapsed + 1e-9 >= CLEAR_CANCEL_S:
      self._intent_hold_remaining = 0.0
    # a go frame: the open road and the plan no longer slowing, with the car at a crawl or the model asking for real acceleration
    go = (self.experimental_mode and obs.release_open and not obs.slowing
          and (car_state.vEgo < GO_ROLL_SPEED or model.action.desiredAcceleration >= GO_ACCEL))
    self._go_elapsed = self._go_elapsed + self.model_dt if go else 0.0
    # a filter-only hint shortens entry, but cannot sustain an active latch after qualifying evidence disappears
    filtered = self.intent_filter.update(confidence if not self.experimental_mode or raw_stop else 0.0)

    if self.experimental_mode:
      self._entry_elapsed = 0.0
      self._clear_elapsed = self._clear_elapsed + self.model_dt if not raw_stop and filtered <= STOP_RELEASE_FILTER_THRESHOLD else 0.0
    else:
      self._clear_elapsed = 0.0
      entry_ready = raw_stop and filtered >= STOP_ENTRY_FILTER_THRESHOLD and not self._entry_suppressed
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
    if car_state.vEgo >= RESUME_RELEASE_SPEED:
      self._exit_revocable = False
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
      self._update_model_evidence(model, car_state, radar_state, model_valid, radar_valid)

    if not self.experimental_mode and self._invalid_elapsed >= MODEL_INVALID_RELEASE_S:
      self._clear_evidence()

    if self.experimental_mode:
      self._active_elapsed += self.control_dt
      if car_state.standstill:
        self._standstill_seen = True
        self._standstill_elapsed += self.control_dt
      resumed_after_stop = self._standstill_seen and car_state.vEgo >= RESUME_RELEASE_SPEED
      standstill_latch_satisfied = not self._standstill_seen or self._standstill_elapsed >= STANDSTILL_MIN_LATCH_S
      latch_satisfied = self._active_elapsed >= MODE_MIN_LATCH_S and standstill_latch_satisfied
      stable_clear = latch_satisfied and self._intent_hold_remaining <= 0.0 and self._clear_elapsed >= STOP_RELEASE_HYSTERESIS_S
      go_confirmed = latch_satisfied and self._go_elapsed + 1e-9 >= GO_CONFIRM_S
      if resumed_after_stop or stable_clear or go_confirmed:
        self._deactivate(POST_STOP_SUPPRESS_S)
        # an exit at rest hands the stop to Chill on the model's word alone: until the car moves off, the model calling
        # the stop again takes it back, since nothing else can form a hold in Chill
        self._exit_revocable = car_state.vEgo < GO_ROLL_SPEED
      elif self._invalid_elapsed >= MODEL_INVALID_RELEASE_S:
        self._deactivate(0.0)
    elif (not self._entry_suppressed and self._invalid_elapsed < MODEL_INVALID_RELEASE_S
          and self._entry_elapsed >= STOP_ENTRY_DEBOUNCE_S):
      self.experimental_mode = True
      self._active_elapsed = 0.0
      self._standstill_elapsed = 0.0
      self._standstill_seen = False
      self._entry_elapsed = 0.0
    return self.experimental_mode
