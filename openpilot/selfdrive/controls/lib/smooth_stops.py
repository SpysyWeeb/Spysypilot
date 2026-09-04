"""Smooth Stops: the last 0.3 m/s of a stop, inside longcontrol.

The planner owns the approach and its ease-off; its stop bit (`should_stop`, v < 0.3 m/s) hands the last few
decimetres to this controller, which has exactly two jobs:

* never clamp while rolling -- stock jumps to the `stopping` state (a ramp to stopAccel) the moment the stop bit sets,
  i.e. while the car is still moving; the hold is armed here only once the car has actually stopped;
* keep a small "kiss" of braking on while the stop completes -- the plan's own request may fade to nothing right at the
  end. Harder planner braking always passes straight through. Whether the car is actually still slowing is the planner's
  business: BLoTv3's landing corridor closes that loop on the measured acceleration (its anti-creep press), so this
  controller carries no stall ratchet of its own any more (audit 2026-09-02: two ratchets stacked on one output).

Field audit 2026-08-29 (routes 23-27, 22 stops, CAN-decoded): this window is the last 0.3 m/s and lasts under 1.3 s; a
lead floor and a queue-aware anti-creep ratchet that lived here never triggered, and the planner already owns leads, so
they are gone; below ~0.3 m/s the Palisade's ESP brings the car to rest with its own brake whatever is requested.
"""
from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_CTRL

# Handoff to the hold clamp.
STANDSTILL_SPEED = 0.10        # m/s, hand the stop to the car's own hold here. Below ~0.06 m/s the ESP fades its braking
                               # whatever the request says, and after StopReq it coasts ~1 s before its own clamp bites: a
                               # hand-off at 0.05 (route 0x4b t=402) was 1.5 s of fade plus 1.4 s of coast, felt as a stop
                               # that keeps creeping; at 0.10 (routes 0x33-0x3e, 25+ stops) the clamp lands after 4 cm and
                               # 0.8 s. The Palisade's standstill flag asserts at ~0.6 m/s and is not believed
HOLD_RELEASE_FRAMES = 50       # control frames (0.5 s) of should_stop=False that release the hold on their own. The release
                               # itself is free of delay: the frame the stop bit drops with the plan asking to move lifts
                               # StopReq at once. The count is the backstop for a plan that sits at zero, and it is what
                               # keeps a stop-bit flicker with the plan still braking from lifting the hold (route 0x4b
                               # t=299: four toggles in 0.4 s read as the brakes slipping)

# The landing.
STOP_KISS_DECEL = 0.15         # m/s^2, the least braking kept on while the stop completes; the same number as the planner
                               # corridor's KISS_DECEL (combo asserts the two agree), so the handoff never changes the plan
SETTLE_JERK = 2.5              # m/s^3, smoothness of the landing command (planner braking is never feathered)


class SmoothStopController:
  """The last 0.3 m/s: bound the plan from below by the kiss, hand off to the clamp once stopped."""

  def __init__(self):
    self._no_stop_frames = 0

  def reset(self) -> None:
    # the settle keeps no state between landings; the hold debounce has its own arm
    pass

  def want_hold(self, should_stop: bool, v_ego: float) -> bool:
    # the clamp lands on a stopped car: the kiss carries it down to STANDSTILL_SPEED first
    return bool(should_stop and v_ego <= STANDSTILL_SPEED)

  def arm_hold(self) -> None:
    # every entry into the hold gets a fresh release debounce; reset() runs every frame while holding and must not
    self._no_stop_frames = 0

  def hold_release(self, should_stop: bool, a_target: float) -> bool:
    # a launch is the stop bit dropping with the plan asking to move: released on that frame, no debounce (the owner's
    # ruling: response time is not for sale). A dropped bit with the plan still braking is a flicker until it lasts
    if not should_stop and a_target > 0.0:
      self._no_stop_frames = 0
      return True
    self._no_stop_frames = 0 if should_stop else self._no_stop_frames + 1
    return self._no_stop_frames >= HOLD_RELEASE_FRAMES

  def settle(self, a_target: float, v_ego: float, last_output: float) -> float:
    # the plan, bounded from below by the kiss; harder plan braking passes through
    a_settle = max(min(a_target, -STOP_KISS_DECEL), ACCEL_MIN)

    # one smooth command: the landing may not step, in either direction
    step = SETTLE_JERK * DT_CTRL
    return min(max(a_settle, last_output - step), last_output + step)
