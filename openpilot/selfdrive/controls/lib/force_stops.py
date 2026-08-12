"""
Original concept from IQPilot's IQForceStops (IQLvbs/openpilot), reimplemented for
Spysypilot by SpysyWeeb (github.com/SpysyWeeb).

Force Stops. In experimental mode the driving model sometimes plans a stop (red light,
stop sign) but never commits: action.shouldStop dithers and the car crawls toward the
line indefinitely. The tell is the model's planned *path*: its endpoint closes in to a
few meters while the stop intent flickers. This module reads that intent directly --
when the model's path ends within a few seconds of travel and there is no lead, latch
the model's own stop point and hold the plan to it by capping the cruise speed at
(remaining distance / ramp time). The stock planner converts that cap into its cruise
acceleration candidate while stronger MPC/e2e braking still wins; on combo,
Smooth Stops lands the last meter. Force Stops decides that/where, the planner
shapes, and Smooth Stops lands.

A false detection only produces a gentle, plan-shaped slowdown; the latched point
survives brief model dropouts, then unwinds after a bounded hold. The cap feeds
the stock planner, never a synthetic brake command.
"""
import math

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

MODEL_STOP_TIME = 3.0     # s, path endpoint within v_ego * this reads as "model plans to stop"
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
A_STOP_ENVELOPE = 1.2     # m/s^2, the owner's comfort curve applied to the model's stop point --
                          # remaining/RAMP_TIME alone is a late linear ramp; the sqrt envelope
                          # shapes the whole approach onto his fitted braking profile
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
# Feed every cap path the same physical endpoint; otherwise a pre-latch or ratchet path silently
# erases the owner's stop-line setback.
LATCH_SETBACK = 5.0       # m, route-calibrated distance short of the model's endpoint
MIN_STOP_LENGTH = 3.0     # m, floor of the detector window, keeps it alive at crawl speeds
DETECT_RC = 1.0           # s, filter time constant on the (flickery) detector
LATCH_THRESHOLD = 0.55    # filtered detector level that latches a forced stop
RELEASE_THRESHOLD = 0.30  # hysteresis: unlatch below this (the model wants to go, e.g. green light)
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


class ForceStops:
  def __init__(self, dt: float = DT_MDL):
    self.dt = dt
    self.detect_filter = FirstOrderFilter(0.0, DETECT_RC, dt)
    self.lead_filter = FirstOrderFilter(0.0, LEAD_RC, dt)
    self.forcing = False
    self.remaining = 0.0
    self.override_timer = 0.0
    self.position_hold_remaining = 0.0

  def _reset(self) -> None:
    self.detect_filter.x = 0.0
    self.lead_filter.x = 0.0
    self.forcing = False
    self.remaining = 0.0
    self.position_hold_remaining = 0.0

  def update(self, sm) -> float:
    """Returns a cruise speed cap in m/s; NO_CAP when inactive. min() it into v_cruise."""
    CS = sm['carState']
    v_ego = max(CS.vEgo, 0.0)

    if CS.gasPressed:
      self.override_timer = GAS_OVERRIDE_S
      self.detect_filter.x = 0.0
      self.forcing = False
      self.position_hold_remaining = 0.0
      return NO_CAP

    self.override_timer = max(self.override_timer - self.dt, 0.0)

    if not (sm['selfdriveState'].enabled and sm['selfdriveState'].experimentalMode):
      self._reset()
      return NO_CAP

    valid = getattr(sm, 'valid', {})
    if not (valid.get('modelV2', False) and valid.get('radarState', False)):
      self._reset()
      return NO_CAP

    lead_present = bool(sm['radarState'].leadOne.present or sm['radarState'].leadTwo.present)
    self.lead_filter.update(1.0 if lead_present else 0.0)
    tracking_lead = self.lead_filter.x > LEAD_GATE
    if lead_present:
      self.detect_filter.x = 0.0
      self.forcing = False
      self.position_hold_remaining = 0.0
      return NO_CAP

    # the model's planned path ends here; a short endpoint means it is planning a stop,
    # however much its shouldStop bit dithers
    xs = sm['modelV2'].position.x
    model_length = float(xs[-1]) if len(xs) else 0.0
    if not math.isfinite(model_length) or model_length <= 0.0:
      self._reset()
      return NO_CAP
    committed_length = max(model_length - LATCH_SETBACK, 0.0)

    action = sm['modelV2'].action
    terminal_speed = float(sm['modelV2'].velocity.x[-1]) if len(sm['modelV2'].velocity.x) >= 2 else math.inf
    terminal_heading = float(sm['modelV2'].orientation.z[-1]) if len(sm['modelV2'].orientation.z) >= 2 else math.inf
    early_stopping = (
      len(xs) >= 2 and not tracking_lead and v_ego >= EARLY_STOP_MIN_SPEED and
      math.isfinite(action.desiredAcceleration) and action.desiredAcceleration <= EARLY_BRAKE_GATE and
      model_length <= v_ego ** 2 / (2.0 * EARLY_STOP_COMFORT_DECEL) + v_ego * EARLY_STOP_RESPONSE_S and
      math.isfinite(terminal_speed) and terminal_speed <= min(EARLY_STOP_TERMINAL_MAX, v_ego * EARLY_STOP_TERMINAL_RATIO) and
      math.isfinite(action.desiredCurvature) and abs(action.desiredCurvature) * v_ego ** 2 <= EARLY_STOP_MAX_LAT_ACCEL and
      math.isfinite(terminal_heading) and abs(terminal_heading) <= EARLY_STOP_MAX_HEADING and
      not (CS.leftBlinker or CS.rightBlinker)
    )
    stop_time = EARLY_STOP_TIME if action.desiredAcceleration < EARLY_BRAKE_GATE else MODEL_STOP_TIME
    model_stopping = 0.0 < model_length < max(v_ego * stop_time, MIN_STOP_LENGTH)
    latch_ready = 0.0 < model_length < max(v_ego * MODEL_STOP_TIME, MIN_STOP_LENGTH)
    detected = (model_stopping or action.shouldStop) and not tracking_lead
    self.detect_filter.update(1.0 if detected else 0.0)
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

    if not self.forcing:
      if self.detect_filter.x >= LATCH_THRESHOLD and latch_ready:
        # latch the route-calibrated stop point now, while the model is confident; from here we only
        # count down by distance actually traveled, immune to later dithering
        self.forcing = True
        self.remaining = committed_length
      elif early_stopping or self.detect_filter.x >= PRE_LATCH_GATE:
        # pre-latch shaping: comfort envelope on the model's LIVE endpoint, so lead-less
        # red lights brake on the owner's curve instead of the model's backloaded ramp;
        # nothing is frozen yet, so a green light or model change of heart costs nothing.
        return max(math.sqrt(2.0 * A_STOP_ENVELOPE * model_length), v_ego - DV_MAX)
      else:
        return NO_CAP

    self.remaining = max(self.remaining - v_ego * self.dt, 0.0)
    # forward-ratchet: while the model still confidently plans this stop, follow its endpoint
    # as it extends (bounded rate, never backward -- shrinking happens only by travel above)
    if detected and latch_ready and self.detect_filter.x >= LATCH_THRESHOLD and committed_length > self.remaining + EXTEND_DEADBAND:
      self.remaining = min(self.remaining + EXTEND_RATE * self.dt, committed_length)
    # endgame down-follow: never roll past the live, setback-adjusted endpoint on a stale latch
    if v_ego < DOWN_SPEED and committed_length < self.remaining - DOWN_DEADBAND:
      self.remaining = max(self.remaining - DOWN_RATE * self.dt, committed_length)
    if self.detect_filter.x < RELEASE_THRESHOLD and self.position_hold_remaining <= 0.0:
      self.forcing = False
      return NO_CAP
    # linear commit ramp near the point, owner's comfort envelope shaping the approach into it,
    # bounded so a collapsing target can never demand a slam (the commit ramp lives below the
    # bound only within ~2 m/s of the stop, where it must)
    cap = min(self.remaining / RAMP_TIME, math.sqrt(2.0 * A_STOP_ENVELOPE * self.remaining))
    return max(cap, v_ego - DV_MAX)
