from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
from openpilot.selfdrive.controls.lib.stop_helpers import MODEL_INVALID_RELEASE_S, STOP_COMMIT_MAX_DISTANCE

# In Experimental mode the model sometimes plans a stop but never commits: shouldStop dithers and the car
# crawls toward the line. The tell is the planned path, whose endpoint closes in while the intent flickers.
# Force Stops latches the model's own stop point, shapes the approach with a bounded cruise cap, hands the
# committed point to the MPC, and owns the hold at standstill until the model plans a real launch.
MODEL_STOP_TIME = 3.0     # s, path endpoint within v_ego * this reads as "model plans to stop"
LATCH_STOP_TIME = 3.25    # s, commit braking evidence once its filtered stop intent is stable
EARLY_STOP_TIME = 4.5     # s, widened detection window, honored only while the model is actually braking
                          # (route 38 t=306: the model backloads lead-less red lights -- still 28 mph with the line
                          # 39 m out -- and the v*3s window latches too late to shape anything)
A_STOP_ENVELOPE = 0.65    # m/s^2, kinematic approach profile; planner/MPC still choose the acceleration
MPC_PROFILE_OFFSET = 6.0  # m, the profile reaches zero this far before its target
PRE_LATCH_GATE = 0.35     # filtered detector level that turns on shaping of the model's live endpoint
DV_MAX = 2.0              # m/s, the cap may never sit further below current speed while shaping: v_cruise is a target
                          # the MPC erases in seconds, so a collapsing path must not command a slam
LATCH_SETBACK = 3.0       # m, short of the model's endpoint. 5.0 was calibrated for the soft MPC column that overshot its point;
                          # with the committed profile the car stops ~0.6 m short of the point (routes 25/26), and the owner's
                          # own stops sat 1.5-2.7 m short of the model's endpoint: 3 m lands about a metre before them
MIN_STOP_LENGTH = 3.0     # m, floor of the detector window, keeps it alive at crawl speeds
DETECT_RC = 1.0           # s
LATCH_THRESHOLD = 0.55
RELEASE_THRESHOLD = 0.30
LEAD_RC = 1.0             # s
LEAD_GATE = 0.45          # filtered lead level above which stopping is the lead logic's job
RAMP_TIME = 3.0           # s, speed cap = remaining distance / this
GAS_OVERRIDE_S = 10.0     # s, a gas press cancels approach shaping for this long; the hold may still re-enter
STOP_POSITION_HOLD_S = 4.0
EXTEND_RATE = 3.0         # m/s, the latched point may follow the model's endpoint forward this fast: the latch trips
                          # mid-collapse of the path, and a frozen target would park the car short of the real line
EXTEND_DEADBAND = 2.0
DOWN_SPEED = 3.0          # m/s, below this the latched point also follows the endpoint down (route 38 t=351:
                          # a stale latch must not roll into the crosswalk); above it the latch is immune to collapse
DOWN_RATE = 2.0
DOWN_DEADBAND = 1.0
QUALIFY_S = 0.3           # s of consistent strict, lead-free stop evidence on a world-fixed endpoint that commits before the
                          # classic window (was 1.0: the model calls a red light only 4-5 s out, every 0.1 s is 1.4 m at 14 m/s)
QUALIFY_WORLD_TOLERANCE = 5.0
RELEASE_RC = 0.30         # s, filter on the model's launch evidence while holding
RELEASE_OPEN_THRESHOLD = 0.70
RELEASE_OPEN_LENGTH = 30.0    # m, a path this long ...
RELEASE_OPEN_FRAMES = 3       # ... for this many consecutive frames releases the hold at once (a green): the filtered path
                              # needed ~0.4 s more, and a one- or two-frame flash (route 27 t=263) is not a green
RESUME_SPEED = 0.8        # m/s, above this a hold becomes a moving commitment again
REARM_S = 10.0            # s, after a lead or a gas tap breaks a commitment or a hold, stopping with stop evidence re-enters the hold
CLEAR_WINDOW_S = 4.0      # s, fallback release while holding: mostly clear, moving, lead-free model frames
CLEAR_WINDOW_FRACTION = 0.8
NO_CAP = math.inf

# The committed approach profile. The MPC's stop column is a soft quadratic obstacle: it starts braking late and
# hard (route 24, 2026-08-29: +1.45 -> -1.86 over 1.5 s after a commit that needed 1.9 m/s^2). The owner's own
# stops do the opposite: reach the needed deceleration within a second, hold it, ease off at the end. The profile
# below is that shape as a plan candidate: the constant deceleration that lands short of the committed point,
# jerk-limited, faded out at low speed so the column's own easing landing (and the hold) take the last metres.
PROFILE_JERK = 2.0             # m/s^3, the candidate moves at most this fast (the owner builds braking at 1-2 m/s^3)
PROFILE_LANDING = 2.5          # m, the constant-deceleration profile lands this far short of the committed point, so the
                               # column's easing landing has room with the car's actuation lag. The car stops about a metre
                               # past the landing (closed-loop plant), so this margin sets the stop position almost 1:1
PROFILE_MIN_DISTANCE = 0.5     # m, past the committed point the column and the hold own everything
PROFILE_MIN_TIME = 1.0         # s, the profile never plans to reach its landing sooner than this: the need tapers with
                               # the speed (v/2 at the end) instead of blowing up as the landing closes (route 27 t=1052)
PROFILE_MAX_DECEL = 3.0        # m/s^2, past this the approach is no longer a comfort matter; column and e2e remain
PROFILE_HANDOVER_SPEED = 3.0   # m/s, the profile starts fading out here ...
PROFILE_FADE_SPEED = 1.5       # m/s, ... and is gone here


@dataclass(frozen=True, slots=True)
class ForceStopsResult:
  v_cruise_cap: float = NO_CAP
  stop_x: float | None = None
  holding: bool = False
  a_target: float | None = None   # the committed approach profile, a plan candidate while a commitment is moving


class ForceStops:
  def __init__(self, dt=DT_MDL):
    self.dt = dt
    self.detect_filter = FirstOrderFilter(0.0, DETECT_RC, dt)
    self.braking_filter = FirstOrderFilter(0.0, DETECT_RC, dt)
    self.lead_filter = FirstOrderFilter(0.0, LEAD_RC, dt)
    self.release_filter = FirstOrderFilter(0.0, RELEASE_RC, dt)
    self.clear_window = deque(maxlen=round(CLEAR_WINDOW_S / dt))
    self.override_timer = 0.0
    self.invalid_s = 0.0
    self.rearm_remaining = 0.0
    self.reset()

  def reset(self):
    self.detect_filter.x = 0.0
    self.braking_filter.x = 0.0
    self.lead_filter.x = 0.0
    self.release_filter.x = 0.0
    self.clear_window.clear()
    self.forcing = False
    self.holding = False
    self.remaining = 0.0
    self.position_hold_remaining = 0.0
    self.qualified_s = 0.0
    self.qualified_endpoint = None
    self.profile_accel = None
    self._open_frames = 0

  def _profile(self, v_ego, a_ego):
    # constant deceleration to the landing, entered from the car's current acceleration and jerk-limited from there
    if self.remaining <= PROFILE_MIN_DISTANCE or v_ego <= PROFILE_FADE_SPEED:
      self.profile_accel = None
      return None
    distance = max(self.remaining - PROFILE_LANDING, v_ego * PROFILE_MIN_TIME)
    need = min(v_ego ** 2 / (2.0 * distance), PROFILE_MAX_DECEL)
    fade = float(np.clip((v_ego - PROFILE_FADE_SPEED) / (PROFILE_HANDOVER_SPEED - PROFILE_FADE_SPEED), 0.0, 1.0))
    start = self.profile_accel if self.profile_accel is not None else min(a_ego, 0.0)
    step = PROFILE_JERK * self.dt
    self.profile_accel = float(np.clip(-need * fade, start - step, start + step))
    return self.profile_accel

  def _result(self, v_ego, a_ego=0.0):
    if self.holding:
      self.profile_accel = None
      return ForceStopsResult(0.0, max(self.remaining, -STOP_DISTANCE), True)
    if self.forcing:
      # no speed cap once committed: the profile, the column and the hold own the stop. The cap's cruise floor used to land
      # the car at -1.2 m/s^2 down to walking pace once the profile had faded (route 27 t=1053)
      return ForceStopsResult(NO_CAP, max(self.remaining, -STOP_DISTANCE), False, self._profile(v_ego, a_ego))
    self.profile_accel = None
    return ForceStopsResult()

  def _qualify(self, obs, v_ego, tracking_lead):
    # one second of strict, lead-free stop evidence on a world-fixed endpoint commits before the classic window
    if not (obs.strict_stop and not tracking_lead and not obs.committed_turn and 0.0 < obs.path_end < STOP_COMMIT_MAX_DISTANCE):
      self.qualified_s = 0.0
      self.qualified_endpoint = None
      return False
    if self.qualified_endpoint is not None and abs(obs.path_end - (self.qualified_endpoint - v_ego * self.dt)) <= QUALIFY_WORLD_TOLERANCE:
      self.qualified_s += self.dt
      self.qualified_endpoint -= v_ego * self.dt
    else:
      self.qualified_s = 0.0
      self.qualified_endpoint = obs.path_end
    return self.qualified_s + 1e-6 >= QUALIFY_S

  def update(self, obs, CS, experimental_mode, enabled, model_valid):
    if not math.isfinite(CS.vEgo) or CS.brakePressed or not enabled:
      self.reset()
      self.rearm_remaining = 0.0
      return ForceStopsResult()
    v_ego = max(CS.vEgo, 0.0)
    a_ego = CS.aEgo if math.isfinite(CS.aEgo) else 0.0

    if CS.gasPressed:
      self.override_timer = GAS_OVERRIDE_S
      if self.holding:
        self.rearm_remaining = REARM_S
      self.reset()
      return ForceStopsResult()
    self.override_timer = max(self.override_timer - self.dt, 0.0)
    self.rearm_remaining = max(self.rearm_remaining - self.dt, 0.0)

    # the mode gates entry only; a later mode exit never strips a commitment or a hold
    if not (self.forcing or self.holding) and not experimental_mode:
      self.reset()
      return ForceStopsResult()

    self.invalid_s = 0.0 if model_valid and obs.complete else self.invalid_s + self.dt
    if self.invalid_s > 0.0:
      if self.invalid_s >= MODEL_INVALID_RELEASE_S:
        self.reset()
        return ForceStopsResult()
      return self._result(v_ego, a_ego)

    self.lead_filter.update(1.0 if obs.lead_present else 0.0)
    tracking_lead = self.lead_filter.x > LEAD_GATE
    if self.holding:
      return self._hold(obs, v_ego, a_ego)
    if obs.lane_change and not self.holding:
      # the endpoint is about to belong to another lane (route 27 t=379: the through lane's line held a stop that the
      # left-turn lane's line was 15 m beyond); drop shaping and any commitment and re-qualify on the new lane's path
      self.reset()
      return ForceStopsResult()
    if tracking_lead or (obs.lead_present and not self.forcing):
      # a lead while moving hands the stop to the lead logic; a broken commitment may re-form as a hold. A raw lead blocks a
      # new commitment, only a tracked one breaks an existing one: a single radar frame reset a red-light commitment 0.5 s
      # before the driver braked (route 24), and a flickering lead must not mint a commitment between its frames (route 23)
      if self.forcing:
        self.rearm_remaining = REARM_S
      self.reset()
      return ForceStopsResult()

    model_length = obs.path_end
    if model_length <= 0.0:
      self.reset()
      return ForceStopsResult()

    qualified = self._qualify(obs, v_ego, tracking_lead)
    committed_length = max(model_length - LATCH_SETBACK, 0.0)
    stop_time = EARLY_STOP_TIME if obs.braking else MODEL_STOP_TIME
    model_stopping = 0.0 < model_length < max(v_ego * stop_time, MIN_STOP_LENGTH)
    classic_latch_ready = 0.0 < model_length < max(v_ego * MODEL_STOP_TIME, MIN_STOP_LENGTH)
    latch_ready = 0.0 < model_length < max(v_ego * (LATCH_STOP_TIME if obs.braking else MODEL_STOP_TIME), MIN_STOP_LENGTH)
    detected = ((model_stopping or obs.should_stop) and not tracking_lead) or qualified
    self.detect_filter.update(1.0 if detected else 0.0)
    self.braking_filter.update(1.0 if detected and obs.braking else 0.0)
    self.position_hold_remaining = max(self.position_hold_remaining - self.dt, 0.0)
    if detected:
      self.position_hold_remaining = STOP_POSITION_HOLD_S

    if CS.standstill and (self.forcing or (self.rearm_remaining > 0.0 and detected and not obs.committed_turn)):
      # the stop is reached, or re-reached shortly after a lead or a gas tap broke the hold: hold it until the model plans a launch
      self.rearm_remaining = 0.0
      self.holding = True
      self.forcing = False
      self.remaining = min(self.remaining, 0.0) if self.remaining > 0.0 else self.remaining
      self.release_filter.x = 0.0
      self.clear_window.clear()
      self._open_frames = 0
      return self._result(v_ego, a_ego)

    if self.override_timer > 0.0:
      return ForceStopsResult()

    just_committed = False
    if not self.forcing:
      latch_confident = self.detect_filter.x if classic_latch_ready else self.braking_filter.x
      if not obs.committed_turn and (qualified or (latch_confident >= LATCH_THRESHOLD and latch_ready)):
        self.forcing = True
        self.remaining = committed_length
        just_committed = True
      elif obs.early_stop and not tracking_lead or self.detect_filter.x >= PRE_LATCH_GATE:
        # shape the approach on the model's live endpoint; nothing is frozen yet, a green light costs nothing
        profile_distance = max(model_length - MPC_PROFILE_OFFSET, 0.0)
        return ForceStopsResult(max(math.sqrt(2.0 * A_STOP_ENVELOPE * profile_distance), v_ego - DV_MAX))
      else:
        return ForceStopsResult()

    if not just_committed:
      self.remaining -= v_ego * self.dt
    # the green: a long, moving path with no stop evidence releases a moving commitment at once, like the hold's D21
    # release -- the filtered detector plus the 4 s position hold kept the profile braking 1.2-1.7 s after the road had
    # opened, until the owner's own gas ended it (route 0x2c t=1105/1135; the hold was even re-armed by noisy path dips)
    self._open_frames = (self._open_frames + 1
                         if (obs.path_end is not None and obs.path_end > RELEASE_OPEN_LENGTH
                             and not (obs.should_stop or obs.strict_stop)) else 0)
    if self._open_frames >= RELEASE_OPEN_FRAMES:
      self.reset()
      return ForceStopsResult()
    if (self.remaining > 0.0 and detected and self.detect_filter.x >= LATCH_THRESHOLD
        and committed_length > self.remaining + EXTEND_DEADBAND):
      # the latched point follows an endpoint that keeps sitting beyond it while the model still calls the stop; it
      # is not gated on the latch window any more -- a slow forward drift of a far endpoint left a commitment 3 m
      # short and the car heading for a stop ~10 m before the line (route 25 t=1547, field test 3)
      self.remaining = min(self.remaining + EXTEND_RATE * self.dt, committed_length)
    if self.remaining > 0.0 and v_ego < DOWN_SPEED and committed_length < self.remaining - DOWN_DEADBAND:
      self.remaining = max(self.remaining - DOWN_RATE * self.dt, committed_length)
    if self.detect_filter.x < RELEASE_THRESHOLD and self.position_hold_remaining <= 0.0:
      self.forcing = False
      return ForceStopsResult()
    return self._result(v_ego, a_ego)

  def _hold(self, obs, v_ego, a_ego):
    if v_ego >= RESUME_SPEED:
      # rolling again (creep or a grade): back to a moving commitment, the latch survives
      self.holding = False
      self.forcing = True
      return self._result(v_ego, a_ego)
    if obs.relevant_lead:
      self.reset()
      self.rearm_remaining = REARM_S
      return ForceStopsResult()
    self.release_filter.update(1.0 if obs.release_open else 0.0)
    self._open_frames = self._open_frames + 1 if (obs.path_end is not None and obs.path_end > RELEASE_OPEN_LENGTH) else 0
    if self._open_frames >= RELEASE_OPEN_FRAMES:
      self.reset()
      return ForceStopsResult()
    stop_evidence = obs.should_stop or obs.strict_stop
    self.clear_window.append(not stop_evidence and obs.terminal_moving and obs.corridor_clear)
    window_clear = (len(self.clear_window) == self.clear_window.maxlen
                    and sum(self.clear_window) >= CLEAR_WINDOW_FRACTION * self.clear_window.maxlen)
    if self.release_filter.x > RELEASE_OPEN_THRESHOLD or window_clear:
      self.reset()
      return ForceStopsResult()
    return self._result(v_ego, a_ego)
