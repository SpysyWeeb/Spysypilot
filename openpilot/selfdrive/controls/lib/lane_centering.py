import math
from typing import NamedTuple
import numpy as np

from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL, DT_MDL

LaneChangeState = log.LaneChangeState

LOOKAHEAD_T = 1.0  # s
MIN_LOOKAHEAD = 10.0  # m, below 10 m/s the correction fades with speed squared
GAIN = 0.2  # share of the plan's distance to the lane center that gets closed by the lookahead
HEADING_GAIN = 2.0  # the plan's heading away from the lane center counts for this many lookaheads
MAX_OFFSET = 0.5  # m, so the correction never exceeds GAIN * 2 * MAX_OFFSET / LOOKAHEAD_T**2 = 0.2 m/s^2
SMOOTH_TAU = 0.25  # s

MIN_LANE_PROB = 0.5
MAX_LANE_STD = 0.3  # m
MIN_LANE_WIDTH = 2.5  # m
MAX_LANE_WIDTH = 4.5  # m

# a plan this far from the lane center is a decision, not drift
OFFSET_BP = [0.5, 0.9]  # m
OFFSET_V = [1.0, 0.0]

# with one line gone the lane center comes from the other line and the width learned while both were seen
WIDTH_TAU = 5.0  # s
WIDTH_STEP = 0.4  # m, a width this different for WIDTH_STEP_FRAMES is a new road, not noise
WIDTH_STEP_FRAMES = 5
MAX_WIDTH_AGE = 60.0  # s
MAX_OFFSET_JUMP = 0.25  # m, a lane center that moves this much relative to the plan came from the wrong width

# a driver who pushes the car further the way the model leans agrees with the model, so the correction stands down
NUDGE_SETTLE = 0.5  # s without steering before a push is judged, so a push in several presses counts once
MIN_LEAN = 0.05  # m, of correction against the model before it leans any way
MIN_NUDGE = 0.1  # m, the push has to move the plan
MAX_NUDGE = 1.0  # m, further is another lane
YIELD_OFFSET = 0.25  # m off center the push has to leave the plan
REARM_OFFSET = 0.15  # m, a plan back this close to the center gets centered again
MAX_YIELD_TIME = 120.0  # s


class LaneCenter(NamedTuple):
  curvature: float
  error: float  # of the lane center from where the plan is heading, before the gain
  offset: float  # of the lane center from the plan at the lookahead
  width: float | None  # of the lane at the lookahead, when both lines were seen


def get_lane_center(model_v2, v_ego: float, width: float | None = None) -> LaneCenter | None:
  lines, probs, stds, position = model_v2.laneLines, model_v2.laneLineProbs, model_v2.laneLineStds, model_v2.position
  if min(len(lines), len(probs), len(stds)) < 3:
    return None

  left, right = lines[1], lines[2]
  if any(len(path.x) == 0 or len(path.x) != len(path.y) for path in (left, right, position)):
    return None

  lookahead = max(v_ego * LOOKAHEAD_T, MIN_LOOKAHEAD)
  if position.x[-1] < lookahead:
    return None

  x = np.asarray(position.x)
  if not np.isfinite(x).all():
    return None
  # the plan up to the lookahead, ending exactly there
  x = np.append(x[x < lookahead], lookahead)
  left_y = np.interp(x, left.x, left.y)
  right_y = np.interp(x, right.x, right.y)
  left_valid, right_valid = (probs[i] >= MIN_LANE_PROB and stds[i] <= MAX_LANE_STD for i in (1, 2))
  if left_valid and right_valid:
    width = right_y[-1] - left_y[-1]
    if not MIN_LANE_WIDTH < width < MAX_LANE_WIDTH:
      return None
    center = (left_y + right_y) / 2
  elif left_valid and width is not None:
    center = left_y + width / 2
    width = None
  elif right_valid and width is not None:
    center = right_y - width / 2
    width = None
  else:
    return None

  # the road's curvature is in both the lane center and the plan, only the offset between them is left
  offset = center - np.interp(x, position.x, position.y)
  x_mean, offset_mean = x.mean(), offset.mean()
  x_var = np.dot(x - x_mean, x - x_mean)
  heading = np.dot(x - x_mean, offset - offset_mean) / x_var if x_var > 0 else 0.0

  # a plan heading away from the center will be further off than the lookahead shows, one heading back needs less
  error = offset_mean - heading * x_mean + HEADING_GAIN * heading * lookahead
  # fading on the same error lets a plan that has started to move over go, not only one that is already there
  error = np.clip(error, -MAX_OFFSET, MAX_OFFSET) * np.interp(abs(error), OFFSET_BP, OFFSET_V)

  # curvature of the arc that closes the error at the lookahead
  curvature = GAIN * 2 * error / lookahead**2
  if not math.isfinite(curvature):
    return None
  return LaneCenter(float(curvature), float(error), float(offset[-1]), None if width is None else float(width))


class LaneCentering:
  def __init__(self):
    self.frame_id = None
    self.target = 0.0
    self.curvature = FirstOrderFilter(0.0, SMOOTH_TAU, DT_CTRL)
    self.width: FirstOrderFilter | None = None
    self.width_frame_id = 0
    self.width_steps = 0
    self.offset = 0.0
    self.lean = 0.0
    self.position: float | None = None
    self.nudge: tuple[float, float | None] | None = None
    self.nudge_torque = 0.0
    self.released_frames = 0
    self.yield_frame_id: int | None = None

  def reset(self):
    self.curvature.x = 0.0
    self.lean, self.position = 0.0, None
    self.nudge = None
    self.yield_frame_id = None

  def update(self, lat_active: bool, CS, model_v2) -> float:
    if not lat_active:
      self.reset()
      return 0.0

    # a push starts and ends where the plan was before this model frame
    self.update_nudge(CS.steeringPressed, CS.steeringTorque)
    if model_v2.frameId != self.frame_id:
      self.frame_id = model_v2.frameId
      self.target = self.update_target(model_v2, CS.vEgo)

    changing_lanes = CS.leftBlinker or CS.rightBlinker or model_v2.meta.laneChangeState != LaneChangeState.off
    if changing_lanes:
      self.yield_frame_id = None

    holding_off = CS.steeringPressed or changing_lanes or self.yield_frame_id is not None
    return self.curvature.update(0.0 if holding_off else self.target)

  def update_target(self, model_v2, v_ego: float) -> float:
    if self.yield_frame_id is not None and not 0 <= (model_v2.frameId - self.yield_frame_id) * DT_MDL <= MAX_YIELD_TIME:
      self.yield_frame_id = None

    width_age = (model_v2.frameId - self.width_frame_id) * DT_MDL
    width = self.width.x if self.width is not None and 0 <= width_age <= MAX_WIDTH_AGE else None
    lane = get_lane_center(model_v2, v_ego, width)
    if lane is None or lane.width is None:
      self.width_steps = 0
    self.lean, self.position = 0.0, None
    if lane is None:
      return 0.0

    self.position = -lane.offset
    if abs(lane.offset) < REARM_OFFSET:
      self.yield_frame_id = None

    if lane.width is not None:
      self.learn_width(lane.width)
      self.width_frame_id = model_v2.frameId
      self.offset = lane.offset
    # only the plan can tell that the learned width no longer fits the road
    elif abs(lane.offset - self.offset) > MAX_OFFSET_JUMP:
      return 0.0

    self.lean = -lane.error
    return lane.curvature

  def update_nudge(self, steering_pressed: bool, steering_torque: float):
    if steering_pressed:
      if self.nudge is None:
        self.nudge, self.nudge_torque = (self.lean, self.position), 0.0
      self.nudge_torque += steering_torque
      self.released_frames = 0
    elif self.nudge is not None:
      self.released_frames += 1
      if self.released_frames >= round(NUDGE_SETTLE / DT_CTRL):
        self.judge_nudge(*self.nudge)
        self.nudge = None

  def judge_nudge(self, lean: float, start: float | None):
    if abs(lean) < MIN_LEAN or start is None or self.position is None:
      return
    side = math.copysign(1.0, lean)
    moved = (self.position - start) * side
    if abs(moved) > MAX_NUDGE:
      return
    # the model moves the car too while the driver holds the wheel, only the driver's torque says who pushed: it is
    # positive to the left, the lane is measured to the right
    pushed = -self.nudge_torque * side > 0
    if pushed and moved >= MIN_NUDGE and self.position * side >= YIELD_OFFSET:
      self.yield_frame_id = self.frame_id
    elif moved <= -MIN_NUDGE:
      self.yield_frame_id = None

  def learn_width(self, width: float):
    if self.width is None:
      self.width = FirstOrderFilter(width, WIDTH_TAU, DT_MDL)
    elif abs(width - self.width.x) > WIDTH_STEP:
      self.width_steps += 1
      if self.width_steps >= WIDTH_STEP_FRAMES:
        self.width.x = width
        self.width_steps = 0
    else:
      self.width_steps = 0
      self.width.update(width)
