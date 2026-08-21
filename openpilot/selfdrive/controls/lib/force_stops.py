"""
Original concept from IQPilot's IQForceStops (IQLvbs/openpilot), reimplemented for
Spysypilot by SpysyWeeb (github.com/SpysyWeeb).

Force Stops. In experimental mode the driving model sometimes plans a stop (red light,
stop sign) but never commits: action.shouldStop dithers and the car crawls toward the
line indefinitely. The tell is the model's planned *path*: its endpoint closes in to a
few meters while the stop intent flickers. This module reads that intent directly --
when the model's path ends within a few seconds of travel and there is no lead, latch
the model's own stop point. A bounded cruise-speed cap shapes the approach; after
commit, the planner also gives the remaining point to its native MPC. Force Stops
never commands acceleration or brakes; on combo, Smooth Stops lands the last meter.

Before commitment, a false detection only produces a bounded, plan-shaped slowdown;
the latched point survives brief model dropouts, then unwinds after a bounded hold.
"""
import math

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.conditional_experimental_mode import (
  FORCE_STOP_COMMIT_DISTANCE_M,
  model_stop_release_open,
  model_trajectories_complete_and_finite,
)

MODEL_STOP_TIME = 3.0     # s, path endpoint within v_ego * this reads as "model plans to stop"
LATCH_STOP_TIME = 3.25    # s, commit braking evidence once its filtered stop intent is stable
EARLY_STOP_TIME = 4.5     # s, widened detection window honored only while the model is actually
                          # braking (route 38 t=306: the model backloads lead-less red lights --
                          # still 28mph with the line 39m out -- and the v*3s window latches too
                          # late to shape anything; the brake gate keeps curve-shortened paths,
                          # which the model coasts toward, from tripping the wide window)
EARLY_BRAKE_GATE = -0.5   # m/s^2, model desiredAcceleration below this counts as "braking"
# Match CEM's strong early-stop tier before bypassing Force Stops' own filter.
# This also protects manually selected Experimental mode from curve/turn hints.
EARLY_STOP_MIN_SPEED = 13.0
EARLY_STOP_COMFORT_DECEL = 1.3
EARLY_STOP_RESPONSE_S = 0.5
EARLY_STOP_TERMINAL_RATIO = 0.35
EARLY_STOP_TERMINAL_MAX = 6.5
EARLY_STOP_MAX_LAT_ACCEL = 1.0
EARLY_STOP_MAX_HEADING = math.radians(20.0)
A_STOP_ENVELOPE = 0.65    # m/s^2, StarPilot's model-stop kinematic approach profile --
                          # the lower profile pulls the bounded cruise target down earlier;
                          # planner/MPC still chooses acceleration; DV_MAX bounds only the cap step
MPC_PROFILE_OFFSET_M = 6.0  # m, StarPilot's profile reaches zero this far before its target
PRE_LATCH_GATE = 0.35     # filtered detector level that turns on pre-latch shaping: a LIVE envelope
                          # on the model's current endpoint, following it freely down AND up. The
                          # latch itself stays on the classic window -- freezing the wide window's
                          # early (pre-collapse) path proved counterproductive in replay: the model
                          # pulls its endpoint in as it brakes, and a stale large latch lets the car
                          # roll past the model's final stop point
DV_MAX = 2.0              # m/s, the cap may never sit further below current speed while shaping
                          # (same bounded-error lesson as smooth_approach: v_cruise is a target the
                          # MPC erases in seconds; unbounded, a collapsing path commands a slam).
                          # Inside ~2 m/s of a stop the bound is moot and the commit ramp expresses
                          # fully, so the guaranteed-stop property is untouched
# Apply the setback to every committed endpoint path so later ratchets cannot erase it.
# Pre-latch shaping stays on the raw live endpoint; final-placement tuning must not alter early response.
LATCH_SETBACK = 5.0       # m, route-calibrated distance short of the model's endpoint
MIN_STOP_LENGTH = 3.0     # m, floor of the detector window, keeps it alive at crawl speeds
DETECT_RC = 1.0           # s, filter time constant on the (flickery) detector
LATCH_THRESHOLD = 0.55    # filtered detector level that latches a forced stop
RELEASE_THRESHOLD = 0.30  # hysteresis: unlatch below this (the model wants to go, e.g. green light)
OPEN_RELEASE_RC = 0.30
OPEN_RELEASE_THRESHOLD = 0.25
LEAD_RC = 1.0             # s, filter on radar lead status
LEAD_GATE = 0.45          # filtered lead level above which stopping is the lead logic's job
RAMP_TIME = 3.0           # s, speed cap = remaining distance / this (linear-in-distance ramp to 0)
GAS_OVERRIDE_S = 10.0     # s, a gas press during a forced stop cancels forcing for this long
STOP_POSITION_HOLD_S = 4.0  # s, keep the latched point through brief model dropouts
EXTEND_RATE = 3.0         # m/s, max rate the latched stop point may follow the model's endpoint
                          # forward. The latch trips mid-collapse of the model's path (by
                          # construction: the detector needs pathEnd < v*3s), so when the model
                          # honestly extends its plan toward the true stop line afterward, the
                          # frozen target parks the car short. Bounded following keeps the latch's
                          # purpose -- immunity to dithering -- while letting the car roll up.
EXTEND_DEADBAND = 2.0     # m, endpoint jitter to ignore before following it forward
DOWN_SPEED = 3.0          # m/s, below this the latched point also follows the model's endpoint DOWN
DOWN_RATE = 2.0           # m/s, bounded, so the car cannot roll past the model's own stop point
                          # into a crosswalk on a stale latch (route 38 t=351); above DOWN_SPEED the
                          # latch stays immune to the endpoint collapsing onto the car mid-braking
DOWN_DEADBAND = 1.0       # m
NO_CAP = float('inf')
CEM_QUALIFIED_MAX_AGE_S = 2.0 * DT_MDL


class ForceStops:
  def __init__(self, dt: float = DT_MDL):
    self.dt = dt
    self.detect_filter = FirstOrderFilter(0.0, DETECT_RC, dt)
    self.braking_filter = FirstOrderFilter(0.0, DETECT_RC, dt)
    self.lead_filter = FirstOrderFilter(0.0, LEAD_RC, dt)
    self.open_release_filter = FirstOrderFilter(0.0, OPEN_RELEASE_RC, dt)
    self.forcing = False
    self.remaining = 0.0
    self.override_timer = 0.0
    self.position_hold_remaining = 0.0

  def _reset(self) -> None:
    self.detect_filter.x = 0.0
    self.braking_filter.x = 0.0
    self.lead_filter.x = 0.0
    self.open_release_filter.x = 0.0
    self.forcing = False
    self.remaining = 0.0
    self.position_hold_remaining = 0.0

  def update(self, sm) -> float:
    """Returns a cruise speed cap in m/s; NO_CAP when inactive. min() it into v_cruise."""
    CS = sm['carState']
    if CS.gasPressed:
      self.override_timer = GAS_OVERRIDE_S
      self.detect_filter.x = 0.0
      self.braking_filter.x = 0.0
      self.open_release_filter.x = 0.0
      self.forcing = False
      self.position_hold_remaining = 0.0
      return NO_CAP

    if not math.isfinite(CS.vEgo):
      self._reset()
      return NO_CAP
    v_ego = max(CS.vEgo, 0.0)

    if CS.brakePressed:
      self._reset()
      return NO_CAP

    self.override_timer = max(self.override_timer - self.dt, 0.0)

    if not (sm['selfdriveState'].enabled and sm['selfdriveState'].experimentalMode):
      self._reset()
      return NO_CAP

    if not sm.all_checks(['carState', 'modelV2', 'radarState', 'selfdriveState']):
      self._reset()
      return NO_CAP

    lead_present = bool(sm['radarState'].leadOne.present or sm['radarState'].leadTwo.present)
    self.lead_filter.update(1.0 if lead_present else 0.0)
    tracking_lead = self.lead_filter.x > LEAD_GATE
    if lead_present:
      self.detect_filter.x = 0.0
      self.braking_filter.x = 0.0
      self.open_release_filter.x = 0.0
      self.forcing = False
      self.position_hold_remaining = 0.0
      return NO_CAP

    # the model's planned path ends here; a short endpoint means it is planning a stop,
    # however much its shouldStop bit dithers
    model = sm['modelV2']
    if not model_trajectories_complete_and_finite(model):
      self._reset()
      return NO_CAP

    xs = model.position.x
    model_length = float(xs[-1]) if len(xs) else 0.0
    if len(xs) < 2 or not math.isfinite(model_length) or model_length <= 0.0:
      self._reset()
      return NO_CAP

    action = model.action
    try:
      desired_accel = float(action.desiredAcceleration)
      desired_curvature = float(action.desiredCurvature)
      terminal_speed = float(model.velocity.x[-1]) if len(model.velocity.x) >= 2 else math.inf
      terminal_heading = float(model.orientation.z[-1]) if len(model.orientation.z) >= 2 else math.inf
      qualified_distance = float(getattr(sm['selfdriveState'], 'conditionalStopDistance', 0.0))
    except (TypeError, ValueError, OverflowError):
      self._reset()
      return NO_CAP
    if not math.isfinite(desired_accel):
      self._reset()
      return NO_CAP
    qualified_model_time = int(getattr(sm['selfdriveState'], 'conditionalStopModelMonoTime', 0))
    model_time = int(getattr(sm, 'logMonoTime', {}).get('modelV2', 0))
    qualified_age = (model_time - qualified_model_time) / 1e9
    cem_stop_qualified = (
      bool(getattr(sm['selfdriveState'], 'conditionalStopQualified', False)) and
      math.isfinite(qualified_distance) and 0.0 < qualified_distance < FORCE_STOP_COMMIT_DISTANCE_M and
      0.0 <= qualified_age <= CEM_QUALIFIED_MAX_AGE_S
    )
    qualified_current_distance = qualified_distance - v_ego * qualified_age
    committed_length = max((qualified_current_distance if cem_stop_qualified else model_length) - LATCH_SETBACK, 0.0)
    early_stopping = (
      len(xs) >= 2 and not tracking_lead and v_ego >= EARLY_STOP_MIN_SPEED and
      desired_accel <= EARLY_BRAKE_GATE and
      model_length <= v_ego ** 2 / (2.0 * EARLY_STOP_COMFORT_DECEL) + v_ego * EARLY_STOP_RESPONSE_S and
      math.isfinite(terminal_speed) and terminal_speed <= min(EARLY_STOP_TERMINAL_MAX, v_ego * EARLY_STOP_TERMINAL_RATIO) and
      math.isfinite(desired_curvature) and abs(desired_curvature) * v_ego ** 2 <= EARLY_STOP_MAX_LAT_ACCEL and
      math.isfinite(terminal_heading) and abs(terminal_heading) <= EARLY_STOP_MAX_HEADING and
      not (CS.leftBlinker or CS.rightBlinker)
    )
    braking = desired_accel < EARLY_BRAKE_GATE
    stop_time = EARLY_STOP_TIME if braking else MODEL_STOP_TIME
    model_stopping = 0.0 < model_length < max(v_ego * stop_time, MIN_STOP_LENGTH)
    classic_latch_ready = 0.0 < model_length < max(v_ego * MODEL_STOP_TIME, MIN_STOP_LENGTH)
    latch_time = LATCH_STOP_TIME if braking else MODEL_STOP_TIME
    latch_ready = 0.0 < model_length < max(v_ego * latch_time, MIN_STOP_LENGTH)
    detected = ((model_stopping or action.shouldStop) and not tracking_lead) or cem_stop_qualified
    self.detect_filter.update(1.0 if detected else 0.0)
    self.braking_filter.update(1.0 if detected and braking else 0.0)
    if not self.forcing:
      self.open_release_filter.x = 0.0
    else:
      self.open_release_filter.update(1.0 if model_stop_release_open(model, require_nonbraking=False) else 0.0)
    self.position_hold_remaining = max(self.position_hold_remaining - self.dt, 0.0)
    if detected:
      self.position_hold_remaining = STOP_POSITION_HOLD_S

    if self.override_timer > 0.0:
      return NO_CAP

    # Once the car is actually stopped, this cap's one job -- getting it to commit to
    # the stop despite the model dithering -- is done. Staying in place from here is the
    # hold-clamp's job (via longitudinalPlan.shouldStop, independent of this cap entirely),
    # not this cap's: remaining can only progress by distance traveled, which is zero for
    # as long as this cap is itself the thing pinning speed near zero, so holding on here
    # is a fixed point with no way out except the driver's gas. Step aside instead, every
    # tick, without latching a fresh forcing bout until the car is moving again -- a car
    # already at standstill is never "approaching" a stop, so it's outside this cap's job.
    if CS.standstill:
      self._reset()
      return NO_CAP

    if self.forcing and self.open_release_filter.x > OPEN_RELEASE_THRESHOLD:
      self._reset()
      return NO_CAP

    just_committed = False
    if not self.forcing:
      latch_confident = self.detect_filter.x if classic_latch_ready else self.braking_filter.x
      if cem_stop_qualified or (latch_confident >= LATCH_THRESHOLD and latch_ready):
        # latch the route-calibrated stop point now, while the model is confident; from here we only
        # count down by distance actually traveled, immune to later dithering
        self.forcing = True
        self.remaining = committed_length
        just_committed = True
      elif early_stopping or self.detect_filter.x >= PRE_LATCH_GATE:
        # pre-latch shaping: comfort envelope on the model's LIVE endpoint, so lead-less
        # red lights brake on the owner's curve instead of the model's backloaded ramp;
        # nothing is frozen yet, so a green light or model change of heart costs nothing.
        profile_distance = max(model_length - MPC_PROFILE_OFFSET_M, 0.0)
        return max(math.sqrt(2.0 * A_STOP_ENVELOPE * profile_distance), v_ego - DV_MAX)
      else:
        return NO_CAP

    if not just_committed:
      self.remaining -= v_ego * self.dt
    # forward-ratchet: while the model still confidently plans this stop, follow its endpoint
    # as it extends (bounded rate, never backward -- shrinking happens only by travel above)
    if (self.remaining > 0.0 and detected and latch_ready and self.detect_filter.x >= LATCH_THRESHOLD and
        committed_length > self.remaining + EXTEND_DEADBAND):
      self.remaining = min(self.remaining + EXTEND_RATE * self.dt, committed_length)
    # endgame down-follow: never roll past the live, setback-adjusted endpoint on a stale latch
    if self.remaining > 0.0 and v_ego < DOWN_SPEED and committed_length < self.remaining - DOWN_DEADBAND:
      self.remaining = max(self.remaining - DOWN_RATE * self.dt, committed_length)
    if self.detect_filter.x < RELEASE_THRESHOLD and self.position_hold_remaining <= 0.0:
      self.forcing = False
      return NO_CAP
    # Speed-cap fallback near the point. DV_MAX bounds only this cap step; after
    # commitment, native MPC may choose stronger braking for the position target.
    profile_distance = max(self.remaining - MPC_PROFILE_OFFSET_M, 0.0)
    cap = min(profile_distance / RAMP_TIME, math.sqrt(2.0 * A_STOP_ENVELOPE * profile_distance))
    return max(cap, v_ego - DV_MAX)
