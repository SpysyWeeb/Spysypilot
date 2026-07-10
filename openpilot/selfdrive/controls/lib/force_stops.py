"""
Original concept from IQPilot's IQForceStops (IQLvbs/openpilot), reimplemented for
Spysypilot by SpysyWeeb (github.com/SpysyWeeb).

Force Stops. In experimental mode the driving model sometimes plans a stop (red light,
stop sign) but never commits: action.shouldStop dithers and the car crawls toward the
line indefinitely. The tell is the model's planned *path*: its endpoint closes in to a
few meters while the stop intent flickers. This module reads that intent directly --
when the model's path ends within a few seconds of travel and there is no lead, latch
the model's own stop point and hold the plan to it by capping the cruise speed at
(remaining distance / ramp time). The ACC MPC then shapes the deceleration itself, its
shouldStop asserts as the plan speed falls below vEgoStopping, and Smooth Stops lands
the last meter. Force Stops decides that/where, the MPC shapes, Smooth Stops lands.

A false detection only produces a gentle, plan-shaped slowdown that unwinds when the
filtered detector drops -- the cap drives the real MPC, never a synthetic brake command.
"""
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

MODEL_STOP_TIME = 3.0     # s, path endpoint within v_ego * this reads as "model plans to stop"
MIN_STOP_LENGTH = 3.0     # m, floor of the detector window, keeps it alive at crawl speeds
DETECT_RC = 1.0           # s, filter time constant on the (flickery) detector
LATCH_THRESHOLD = 0.55    # filtered detector level that latches a forced stop
RELEASE_THRESHOLD = 0.30  # hysteresis: unlatch below this (the model wants to go, e.g. green light)
LEAD_RC = 1.0             # s, filter on radar lead status
LEAD_GATE = 0.45          # filtered lead level above which stopping is the lead logic's job
RAMP_TIME = 3.0           # s, speed cap = remaining distance / this (linear-in-distance ramp to 0)
GAS_OVERRIDE_S = 10.0     # s, a gas press during a forced stop cancels forcing for this long
EXTEND_RATE = 3.0         # m/s, max rate the latched stop point may follow the model's endpoint
                          # forward. The latch trips mid-collapse of the model's path (by
                          # construction: the detector needs pathEnd < v*3s), so when the model
                          # honestly extends its plan toward the true stop line afterward, the
                          # frozen target parks the car short. Bounded following keeps the latch's
                          # purpose -- immunity to dithering -- while letting the car roll up.
EXTEND_DEADBAND = 2.0     # m, endpoint jitter to ignore before following it forward
NO_CAP = float('inf')


class ForceStops:
  def __init__(self, dt: float = DT_MDL):
    self.dt = dt
    self.detect_filter = FirstOrderFilter(0.0, DETECT_RC, dt)
    self.lead_filter = FirstOrderFilter(0.0, LEAD_RC, dt)
    self.forcing = False
    self.remaining = 0.0
    self.override_timer = 0.0

  def _reset(self) -> None:
    self.detect_filter.x = 0.0
    self.lead_filter.x = 0.0
    self.forcing = False
    self.remaining = 0.0

  def update(self, sm) -> float:
    """Returns a cruise speed cap in m/s; NO_CAP when inactive. min() it into v_cruise."""
    CS = sm['carState']
    v_ego = max(CS.vEgo, 0.0)

    if not (sm['selfdriveState'].enabled and sm['selfdriveState'].experimentalMode):
      self._reset()
      return NO_CAP

    self.lead_filter.update(1.0 if sm['radarState'].leadOne.status else 0.0)
    tracking_lead = self.lead_filter.x > LEAD_GATE

    # the model's planned path ends here; a short endpoint means it is planning a stop,
    # however much its shouldStop bit dithers
    xs = sm['modelV2'].position.x
    model_length = float(xs[-1]) if len(xs) else 0.0
    model_stopping = 0.0 < model_length < max(v_ego * MODEL_STOP_TIME, MIN_STOP_LENGTH)
    detected = (model_stopping or sm['modelV2'].action.shouldStop) and not tracking_lead
    self.detect_filter.update(1.0 if detected else 0.0)

    # driver gas during (or about to enter) a forced stop: the driver knows better
    if CS.gasPressed and (self.forcing or self.detect_filter.x >= LATCH_THRESHOLD):
      self.override_timer = GAS_OVERRIDE_S
      self.forcing = False
    self.override_timer = max(self.override_timer - self.dt, 0.0)
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
      self.forcing = False
      return NO_CAP

    if not self.forcing:
      if self.detect_filter.x >= LATCH_THRESHOLD and model_stopping:
        # latch the model's stop point now, while it is confident; from here we only
        # count down by distance actually traveled, immune to later dithering
        self.forcing = True
        self.remaining = max(model_length, MIN_STOP_LENGTH)
      else:
        return NO_CAP

    self.remaining = max(self.remaining - v_ego * self.dt, 0.0)
    # forward-ratchet: while the model still confidently plans this stop, follow its endpoint
    # as it extends (bounded rate, never backward -- shrinking happens only by travel above)
    if self.detect_filter.x >= LATCH_THRESHOLD and model_length > self.remaining + EXTEND_DEADBAND:
      self.remaining = min(self.remaining + EXTEND_RATE * self.dt, model_length)
    if self.detect_filter.x < RELEASE_THRESHOLD:
      self.forcing = False
      return NO_CAP
    return self.remaining / RAMP_TIME
