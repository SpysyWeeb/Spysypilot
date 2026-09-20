import math
import numpy as np

from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL

LaneChangeState = log.LaneChangeState

LOOKAHEAD_T = 1.0  # s
MIN_LOOKAHEAD = 10.0  # m, below 10 m/s the correction fades with speed squared
GAIN = 0.2  # share of the plan's distance to the lane center that gets closed by the lookahead
SMOOTH_TAU = 0.25  # s

MIN_LANE_PROB = 0.5
MAX_LANE_STD = 0.3  # m
MIN_LANE_WIDTH = 2.5  # m
MAX_LANE_WIDTH = 4.5  # m

# a plan this far from the lane center is a decision, not drift
OFFSET_BP = [0.5, 0.9]  # m
OFFSET_V = [1.0, 0.0]


def get_lane_center_curvature(model_v2, v_ego: float) -> float:
  lines, probs, stds, position = model_v2.laneLines, model_v2.laneLineProbs, model_v2.laneLineStds, model_v2.position
  if min(len(lines), len(probs), len(stds)) < 3:
    return 0.0

  left, right = lines[1], lines[2]
  if any(len(path.x) == 0 or len(path.x) != len(path.y) for path in (left, right, position)):
    return 0.0

  if min(probs[1], probs[2]) < MIN_LANE_PROB or max(stds[1], stds[2]) > MAX_LANE_STD:
    return 0.0

  lookahead = max(v_ego * LOOKAHEAD_T, MIN_LOOKAHEAD)
  if position.x[-1] < lookahead:
    return 0.0

  left_y = np.interp(lookahead, left.x, left.y)
  right_y = np.interp(lookahead, right.x, right.y)
  if not MIN_LANE_WIDTH < right_y - left_y < MAX_LANE_WIDTH:
    return 0.0

  # the road's curvature is in both the lane center and the plan, only the offset between them is left
  offset = (left_y + right_y) / 2 - np.interp(lookahead, position.x, position.y)
  offset *= np.interp(abs(offset), OFFSET_BP, OFFSET_V)

  # curvature of the arc that closes the offset at the lookahead
  curvature = GAIN * 2 * offset / lookahead**2
  return float(curvature) if math.isfinite(curvature) else 0.0


class LaneCentering:
  def __init__(self):
    self.frame_id = None
    self.target = 0.0
    self.curvature = FirstOrderFilter(0.0, SMOOTH_TAU, DT_CTRL)

  def reset(self):
    self.curvature.x = 0.0

  def update(self, lat_active: bool, CS, model_v2) -> float:
    if not lat_active:
      self.reset()
      return 0.0

    if model_v2.frameId != self.frame_id:
      self.frame_id = model_v2.frameId
      self.target = get_lane_center_curvature(model_v2, CS.vEgo)

    leaving_lane = CS.steeringPressed or CS.leftBlinker or CS.rightBlinker or \
                   model_v2.meta.laneChangeState != LaneChangeState.off
    return self.curvature.update(0.0 if leaving_lane else self.target)
