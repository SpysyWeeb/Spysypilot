from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

LANE_CHANGE_SPEED_MIN = 20 * CV.MPH_TO_MS
LANE_CHANGE_TIME_MAX = 10.
LANE_CHANGE_START_TIME = 0.5
LANE_CHANGE_NUDGELESS_DELAY = 0.05  # s, auto-advance from preLaneChange without torque nudge

class DesireHelper:
  def __init__(self):
    self.lane_change_state = LaneChangeState.off
    self.lane_change_direction = LaneChangeDirection.none
    self.lane_change_timer = 0.0
    self.lane_change_wait_timer = 0.0
    self.nudgeless_used = False
    self.brake_cancelled = False
    self.prev_one_blinker = False
    self.prev_signal = False
    self.desire = log.Desire.none

  @staticmethod
  def get_lane_change_direction(CS):
    return LaneChangeDirection.left if CS.leftBlinker else LaneChangeDirection.right

  def update(self, carstate, lateral_active, lane_change_prob):
    v_ego = carstate.vEgo
    one_blinker = carstate.leftBlinker != carstate.rightBlinker
    below_lane_change_speed = v_ego < LANE_CHANGE_SPEED_MIN

    # Nudgeless fires once per signal event: one blinker on, from the frame it comes on until it goes off, whatever
    # lateral control does in between. A lateral dropout neither gives the event another automatic start nor takes
    # away one it has not used. Only a signal that comes on while a change can begin from it (lateral active, above
    # the lane-change speed, no change under way) is offered one; nudgeless_used marks the offer spent or never made.
    # Braking CANCELS auto for the rest of this signal event -- not just while the
    # pedal is down. A driver braking with the blinker on isn't setting up an auto
    # change, and the change firing the instant they lift the pedal is a surprise;
    # they can still nudge the wheel to change, or cycle the blinker to re-arm.
    signal_started = one_blinker and not self.prev_signal
    if signal_started:
      self.nudgeless_used = True
      self.brake_cancelled = False
    if one_blinker and carstate.brakePressed:
      self.brake_cancelled = True

    if not lateral_active or self.lane_change_timer > LANE_CHANGE_TIME_MAX:
      self.lane_change_state = LaneChangeState.off
      self.lane_change_direction = LaneChangeDirection.none
      self.lane_change_timer = 0.0
      self.lane_change_wait_timer = 0.0
    else:
      if self.lane_change_state == LaneChangeState.off and one_blinker and not self.prev_one_blinker and not below_lane_change_speed:
        self.lane_change_state = LaneChangeState.preLaneChange
        self.lane_change_timer = 0.0
        self.lane_change_wait_timer = 0.0
        # back from a lateral dropout with the signal held, the signal event keeps what it had left
        if signal_started:
          self.nudgeless_used = False
        # Initialize lane change direction to prevent UI alert flicker
        self.lane_change_direction = self.get_lane_change_direction(carstate)

      elif self.lane_change_state == LaneChangeState.preLaneChange:
        # Update lane change direction
        self.lane_change_direction = self.get_lane_change_direction(carstate)

        self.lane_change_wait_timer += DT_MDL

        torque_applied = carstate.steeringPressed and \
                         ((carstate.steeringTorque > 0 and self.lane_change_direction == LaneChangeDirection.left) or
                          (carstate.steeringTorque < 0 and self.lane_change_direction == LaneChangeDirection.right))

        blindspot_detected = ((carstate.leftBlindspot and self.lane_change_direction == LaneChangeDirection.left) or
                              (carstate.rightBlindspot and self.lane_change_direction == LaneChangeDirection.right))

        # Subsequent changes in the same signal require a manual nudge so the car
        # can't keep auto-changing lanes.
        auto_allowed = not self.nudgeless_used and not self.brake_cancelled and self.lane_change_wait_timer > LANE_CHANGE_NUDGELESS_DELAY

        if not one_blinker or below_lane_change_speed:
          self.lane_change_state = LaneChangeState.off
          self.lane_change_direction = LaneChangeDirection.none
          self.lane_change_timer = 0.0
          self.lane_change_wait_timer = 0.0
          # the offer ends with the wait: below the lane-change speed stock begins again only on a fresh signal, and a
          # lateral dropout once back above it is not one
          self.nudgeless_used = True
        elif not blindspot_detected and (torque_applied or auto_allowed):
          # A manual start also consumes this blinker event's automatic allowance.
          self.nudgeless_used = True
          self.lane_change_state = LaneChangeState.laneChangeStarting
          self.lane_change_timer = 0.0
          self.lane_change_wait_timer = 0.0

      elif self.lane_change_state == LaneChangeState.laneChangeStarting:
        self.lane_change_timer += DT_MDL

        if lane_change_prob < 0.02 and self.lane_change_timer >= LANE_CHANGE_START_TIME:
          self.lane_change_timer = 0.0
          if one_blinker:
            self.lane_change_state = LaneChangeState.preLaneChange
            self.lane_change_direction = self.get_lane_change_direction(carstate)
          else:
            self.lane_change_state = LaneChangeState.off
            self.lane_change_direction = LaneChangeDirection.none

    self.prev_one_blinker = one_blinker and lateral_active
    self.prev_signal = one_blinker

    self.desire = log.Desire.none
    if self.lane_change_state == LaneChangeState.laneChangeStarting:
      if self.lane_change_direction == LaneChangeDirection.left:
        self.desire = log.Desire.laneChangeLeft
      elif self.lane_change_direction == LaneChangeDirection.right:
        self.desire = log.Desire.laneChangeRight
