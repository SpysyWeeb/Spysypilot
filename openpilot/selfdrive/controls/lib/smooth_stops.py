"""The last decimetres of a stop, inside longcontrol.

The planner's stop bit hands off here: a floor of braking rides under the plan while the car rolls out, and the
hold clamp is armed only once the car has stopped. This carries no anti-creep of its own, so nothing here presses
harder when the car stops making progress under the kiss. The planner and the car's own ESP, which brings the car
to rest below ~0.3 m/s whatever is requested, complete the stop.
"""
from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_CTRL

# The stop bit sets at v < 0.3 m/s with the plan already braking, and stock clamps on that frame, while the car
# still rolls. The hand-off waits for 0.10 m/s instead: below ~0.06 m/s the ESP fades its braking whatever is
# requested, and the car coasts about a second after StopReq before its own clamp bites. The standstill flag
# asserts at ~0.6 m/s and is far too early to use. The release is immediate on a launch, where latency is not
# acceptable; the frame count is the backstop for a plan that sits at zero and keeps a dropped stop bit with the
# plan still braking from lifting the hold.
STANDSTILL_SPEED = 0.10     # m/s, speed at or below which the hold clamp takes over the stop
HOLD_RELEASE_FRAMES = 50    # control frames of should_stop=False that release the hold on their own
STOP_KISS_DECEL = 0.15      # m/s^2, least braking kept on while the stop completes
SETTLE_JERK = 2.5           # m/s^3, rate limit on the landing command


class SmoothStopController:
  """The last 0.3 m/s: bound the plan from below by the kiss, hand off to the clamp once stopped."""

  def __init__(self):
    self._no_stop_frames = 0

  def want_hold(self, should_stop: bool, v_ego: float) -> bool:
    # the clamp lands on a stopped car: the kiss carries it down to STANDSTILL_SPEED first
    return bool(should_stop and v_ego <= STANDSTILL_SPEED)

  def arm_hold(self) -> None:
    # every entry into the hold gets a fresh release debounce
    self._no_stop_frames = 0

  def hold_release(self, should_stop: bool, a_target: float) -> bool:
    # a launch is the stop bit dropping with the plan asking to move: released on that frame, no debounce
    if not should_stop and a_target > 0.0:
      self._no_stop_frames = 0
      return True

    # a dropped bit with the plan still braking is a flicker unless it lasts HOLD_RELEASE_FRAMES
    self._no_stop_frames = 0 if should_stop else self._no_stop_frames + 1
    return self._no_stop_frames >= HOLD_RELEASE_FRAMES

  def settle(self, a_target: float, last_output: float) -> float:
    # the plan, bounded from below by the kiss; harder plan braking passes through
    a_settle = max(min(a_target, -STOP_KISS_DECEL), ACCEL_MIN)

    # one smooth command: the landing may not step, in either direction
    step = SETTLE_JERK * DT_CTRL
    return min(max(a_settle, last_output - step), last_output + step)
