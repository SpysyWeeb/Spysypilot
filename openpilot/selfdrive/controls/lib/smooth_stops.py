"""Smooth Stops: the last 0.3 m/s of a stop, inside longcontrol.

The planner owns the approach and its ease-off; its stop bit (`should_stop`, v < 0.3 m/s) hands the last few
decimetres to this controller, which has exactly two jobs:

* never clamp while rolling -- stock jumps to the `stopping` state (a ramp to stopAccel) the moment the stop bit sets,
  i.e. while the car is still moving; the hold is armed here only once the car has actually stopped;
* guarantee the stop completes, gently -- the plan's own request may fade to nothing right at the end; at least a small
  "kiss" of braking stays on, and if the car stops making progress (a grade, creep torque) the pressure ratchets up
  until it does. Harder planner braking always passes straight through.

Field audit 2026-08-29 (routes 23-27, 22 stops, CAN-decoded): this window is the last 0.3 m/s and lasts under 1.3 s; a
lead floor and a queue-aware anti-creep ratchet that lived here never triggered, and the planner already owns leads, so
they are gone; below ~0.3 m/s the Palisade's ESP brings the car to rest with its own brake whatever is requested.
"""
from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_CTRL

# Handoff to the hold clamp.
STANDSTILL_SPEED = 0.05        # m/s, arm the stopping/hold clamp once the car is essentially stopped
STANDSTILL_HOLD_SPEED = 0.15   # m/s, ceiling for trusting CS.standstill (the Palisade asserts it a hair early)
HOLD_RELEASE_FRAMES = 2        # frames of should_stop=False before the hold releases. The planner's hold already corroborates
                               # a launch; ~22 min of standstill on the audit routes showed no stop-bit flicker at all

# The landing.
STOP_KISS_DECEL = 0.12         # m/s^2, the least braking kept on while the stop completes
STALL_S = 0.5                  # s without progress before the ratchet starts
STALL_RATE = 0.5               # m/s^2 per stalled second added until the car moves again toward the stop
PROGRESS_EPS = 0.02            # m/s, a speed drop below the running minimum counts as progress
SETTLE_JERK = 2.5              # m/s^3, smoothness of the landing command (planner braking is never feathered)


class SmoothStopController:
  """The last 0.3 m/s: bound the plan from below by the kiss, ratchet on a stall, hand off to the clamp once stopped."""

  def __init__(self):
    self._no_stop_frames = 0
    self.reset()

  def reset(self) -> None:
    self._v_min = float('inf')
    self._stall_s = 0.0

  def want_hold(self, should_stop: bool, v_ego: float, standstill: bool) -> bool:
    # the clamp lands on a stopped car: v at or below STANDSTILL_SPEED, or the car's own standstill flag while it is
    # slow enough to be believed
    return bool(should_stop and (v_ego <= STANDSTILL_SPEED or (standstill and v_ego <= STANDSTILL_HOLD_SPEED)))

  def arm_hold(self) -> None:
    # every entry into the hold gets a fresh release debounce; reset() runs every frame while holding and must not
    self._no_stop_frames = 0

  def hold_release(self, should_stop: bool) -> bool:
    self._no_stop_frames = 0 if should_stop else self._no_stop_frames + 1
    return self._no_stop_frames >= HOLD_RELEASE_FRAMES

  def settle(self, a_target: float, v_ego: float, last_output: float) -> float:
    # progress bookkeeping: the running minimum speed, and how long the car has not been getting slower
    if v_ego < self._v_min - PROGRESS_EPS:
      self._v_min = v_ego
      self._stall_s = 0.0
    else:
      self._stall_s += DT_CTRL
    ratchet = STALL_RATE * max(self._stall_s - STALL_S, 0.0)

    # the plan, bounded from below by the kiss (plus whatever a stall has ratcheted in); harder plan braking passes through
    a_settle = min(a_target, -(STOP_KISS_DECEL + ratchet))
    a_settle = max(a_settle, ACCEL_MIN)

    # one smooth command: the landing may not step, in either direction
    step = SETTLE_JERK * DT_CTRL
    return min(max(a_settle, last_output - step), last_output + step)
