import unittest

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.modeld import get_action_from_model


def _previous_action(curvature: float = 0.002) -> log.ModelDataV2.Action:
  return log.ModelDataV2.Action(desiredCurvature=curvature, desiredAcceleration=0.0, shouldStop=False, desiredCurvatureTime=0.0)


class TestActionTiming(unittest.TestCase):

  def test_action_head_publishes_authored_time(self):
    lat_action_t = float(np.float32(0.2375))
    model_output = {"action": np.asarray([[0.04, 0.0]], dtype=np.float32)}
    action = get_action_from_model(model_output, _previous_action(), lat_action_t, 0.3, 2.0)
    self.assertEqual(action.desiredCurvatureTime, np.float32(lat_action_t))

    event = log.Event.new_message()
    event.init("modelV2")
    event.modelV2.action = action
    with log.Event.from_bytes(event.to_bytes()) as decoded:
      self.assertEqual(decoded.modelV2.action.desiredCurvatureTime, np.float32(lat_action_t))

  def test_plan_fallback_publishes_same_authored_time(self):
    lat_action_t = float(np.float32(0.25))
    plan = np.zeros((len(ModelConstants.T_IDXS), Plan.ORIENTATION_RATE.stop), dtype=np.float32)
    plan[:, Plan.VELOCITY][:, 0] = 5.0
    plan[:, Plan.T_FROM_CURRENT_EULER][:, 2] = np.asarray(ModelConstants.T_IDXS, dtype=np.float32) * 0.05
    plan[:, Plan.ORIENTATION_RATE][:, 2] = 0.05

    action = get_action_from_model({"plan": plan[np.newaxis, ...]}, _previous_action(), lat_action_t, 0.3, 5.0)
    self.assertTrue(np.isfinite(action.desiredCurvature))
    self.assertEqual(action.desiredCurvatureTime, np.float32(lat_action_t))
