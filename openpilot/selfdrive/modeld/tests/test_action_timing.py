import ast
from pathlib import Path
from types import FunctionType
import unittest

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.drive_helpers import (
  get_accel_from_plan,
  get_curvature_from_plan,
  should_stop,
  smooth_value,
)
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan


def get_action_from_model() -> FunctionType:
  """Load the exact function without importing modeld's daemon-only binaries."""
  source_path = Path(__file__).parents[1] / "modeld.py"
  tree = ast.parse(source_path.read_text())
  function = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "get_action_from_model"
  )
  namespace = {
    "np": np,
    "log": log,
    "Plan": Plan,
    "ModelConstants": ModelConstants,
    "get_accel_from_plan": get_accel_from_plan,
    "get_curvature_from_plan": get_curvature_from_plan,
    "should_stop": should_stop,
    "smooth_value": smooth_value,
    "LAT_SMOOTH_SECONDS": 0.0,
    "LONG_SMOOTH_SECONDS": 0.3,
    "MIN_LAT_CONTROL_SPEED": 0.3,
  }
  module = ast.Module(body=[function], type_ignores=[])
  exec(compile(module, source_path, "exec"), namespace)
  return namespace["get_action_from_model"]


def _previous_action(curvature: float = 0.002) -> log.ModelDataV2.Action:
  return log.ModelDataV2.Action(
    desiredCurvature=curvature,
    desiredAcceleration=0.0,
    shouldStop=False,
    desiredCurvatureTime=0.0,
  )


class TestActionTiming(unittest.TestCase):
  def setUp(self):
    self.get_action_from_model = get_action_from_model()

  def test_action_head_preserves_curvature_and_publishes_authored_time(self) -> None:
    lat_action_t = float(np.float32(0.2375))
    raw_curvature = 0.01
    previous = _previous_action()
    model_output = {
      "action": np.asarray([[raw_curvature * 4.0, 0.0]], dtype=np.float32),
    }

    unsmoothed = self.get_action_from_model(
      model_output, previous, lat_action_t, 0.3, 2.0,
    )
    self.assertEqual(unsmoothed.desiredCurvature, np.float32(raw_curvature))

    globals_dict = self.get_action_from_model.__globals__
    old_smooth_seconds = globals_dict["LAT_SMOOTH_SECONDS"]
    globals_dict["LAT_SMOOTH_SECONDS"] = 0.1
    try:
      action = self.get_action_from_model(
        model_output, previous, lat_action_t, 0.3, 2.0,
      )
    finally:
      globals_dict["LAT_SMOOTH_SECONDS"] = old_smooth_seconds

    expected_curvature = smooth_value(
      raw_curvature, previous.desiredCurvature, 0.1,
    )
    self.assertEqual(action.desiredCurvature, np.float32(expected_curvature))
    self.assertEqual(action.desiredCurvatureTime, np.float32(lat_action_t))

    event = log.Event.new_message()
    event.init("modelV2")
    event.modelV2.action = action
    with log.Event.from_bytes(event.to_bytes()) as decoded:
      self.assertEqual(decoded.modelV2.action.desiredCurvature, action.desiredCurvature)
      self.assertEqual(
        decoded.modelV2.action.desiredCurvatureTime,
        np.float32(lat_action_t),
      )

  def test_plan_fallback_publishes_same_authored_time(self) -> None:
    lat_action_t = float(np.float32(0.25))
    plan = np.zeros(
      (len(ModelConstants.T_IDXS), Plan.ORIENTATION_RATE.stop),
      dtype=np.float32,
    )
    plan[:, Plan.VELOCITY][:, 0] = 5.0
    plan[:, Plan.T_FROM_CURRENT_EULER][:, 2] = (
      np.asarray(ModelConstants.T_IDXS, dtype=np.float32) * 0.05
    )
    plan[:, Plan.ORIENTATION_RATE][:, 2] = 0.05

    action = self.get_action_from_model(
      {"plan": plan[np.newaxis, ...]},
      _previous_action(),
      lat_action_t,
      0.3,
      5.0,
    )

    self.assertTrue(np.isfinite(action.desiredCurvature))
    self.assertEqual(action.desiredCurvatureTime, np.float32(lat_action_t))
