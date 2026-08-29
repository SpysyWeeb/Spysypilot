import unittest

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan
from openpilot.selfdrive.modeld.modeld import LAT_PREVIEW_OFFSETS, get_action_from_model


def _previous_action(curvature: float = 0.002) -> log.ModelDataV2.Action:
  return log.ModelDataV2.Action(desiredCurvature=curvature, desiredAcceleration=0.0, shouldStop=False, desiredCurvatureTime=0.0)


class TestActionTiming(unittest.TestCase):

  def test_action_head_publishes_authored_time(self):
    lat_action_t = float(np.float32(0.2375))
    model_output = {"action": np.asarray([[0.04, 0.0]], dtype=np.float32)}
    action = get_action_from_model(model_output, _previous_action(), lat_action_t, 0.3, 2.0)
    self.assertEqual(action.desiredCurvatureTime, np.float32(lat_action_t))
    # no plan head, no preview: a consumer treats the empty list as absent
    self.assertEqual(len(action.desiredCurvaturePreview), 0)

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


def _curving_plan(v_ego: float = 12.0) -> np.ndarray:
  # yaw building quadratically: a curve entry, so the preview differs from sample to sample
  t = np.asarray(ModelConstants.T_IDXS, dtype=np.float32)
  plan = np.zeros((len(t), Plan.ORIENTATION_RATE.stop), dtype=np.float32)
  plan[:, Plan.VELOCITY][:, 0] = v_ego
  plan[:, Plan.T_FROM_CURRENT_EULER][:, 2] = 0.01 * t + 0.004 * t ** 2
  plan[:, Plan.ORIENTATION_RATE][:, 2] = 0.01 + 0.008 * t
  return plan


class TestCurvaturePreview(unittest.TestCase):

  def test_preview_is_the_scalar_function_along_the_plan(self):
    lat_action_t = float(np.float32(0.25))
    plan = _curving_plan()
    action = get_action_from_model({"plan": plan[np.newaxis, ...]}, _previous_action(), lat_action_t, 0.3, 12.0)
    times = list(action.desiredCurvaturePreviewTimes)
    preview = list(action.desiredCurvaturePreview)
    self.assertEqual(len(preview), len(LAT_PREVIEW_OFFSETS))
    self.assertEqual(times, [np.float32(lat_action_t + offset) for offset in LAT_PREVIEW_OFFSETS])
    # bit-exact pin: the first sample is the published curvature, in the same float32 field
    self.assertEqual(preview[0], action.desiredCurvature)
    yaws = plan[:, Plan.T_FROM_CURRENT_EULER][:, 2]
    yaw_rates = plan[:, Plan.ORIENTATION_RATE][:, 2]
    for t, curvature in zip(times, preview, strict=True):
      expected = get_curvature_from_plan(yaws, yaw_rates, ModelConstants.T_IDXS, 12.0, float(t))
      self.assertAlmostEqual(curvature, expected, delta=1e-6)
    self.assertGreater(preview[-1], preview[0])

  def test_action_head_preview_carries_the_plan_change(self):
    lat_action_t = float(np.float32(0.25))
    plan = _curving_plan()
    head_curvature = 0.04 / max(1.0, 12.0) ** 2
    action = get_action_from_model({"action": np.asarray([[0.04, 0.0]], dtype=np.float32), "plan": plan[np.newaxis, ...]},
                                   _previous_action(), lat_action_t, 0.3, 12.0)
    preview = list(action.desiredCurvaturePreview)
    self.assertEqual(preview[0], action.desiredCurvature)
    self.assertAlmostEqual(action.desiredCurvature, head_curvature, delta=1e-7)
    yaws = plan[:, Plan.T_FROM_CURRENT_EULER][:, 2]
    yaw_rates = plan[:, Plan.ORIENTATION_RATE][:, 2]
    anchor = get_curvature_from_plan(yaws, yaw_rates, ModelConstants.T_IDXS, 12.0, lat_action_t)
    for t, curvature in zip(action.desiredCurvaturePreviewTimes, preview, strict=True):
      expected = get_curvature_from_plan(yaws, yaw_rates, ModelConstants.T_IDXS, 12.0, float(t)) - anchor
      self.assertAlmostEqual(curvature - preview[0], expected, delta=1e-6)

  def test_low_speed_hold_pins_the_preview_to_the_held_curvature(self):
    plan = _curving_plan(v_ego=0.1)
    action = get_action_from_model({"plan": plan[np.newaxis, ...]}, _previous_action(0.002), 0.25, 0.3, 0.1)
    self.assertEqual(action.desiredCurvature, np.float32(0.002))
    self.assertEqual(action.desiredCurvaturePreview[0], action.desiredCurvature)

  def test_preview_survives_serialization(self):
    action = get_action_from_model({"plan": _curving_plan()[np.newaxis, ...]}, _previous_action(), 0.25, 0.3, 12.0)
    event = log.Event.new_message()
    event.init("modelV2")
    event.modelV2.action = action
    with log.Event.from_bytes(event.to_bytes()) as decoded:
      self.assertEqual(list(decoded.modelV2.action.desiredCurvaturePreview), list(action.desiredCurvaturePreview))
      self.assertEqual(decoded.modelV2.action.desiredCurvaturePreview[0], decoded.modelV2.action.desiredCurvature)

