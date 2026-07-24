from types import SimpleNamespace

import numpy as np

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.modeld.constants import ModelConstants


def model_message(size=ModelConstants.IDX_N, gas_press_probs=(0.1, 0.8)):
  values = np.linspace(0.0, 1.0, size)
  return SimpleNamespace(
    position=SimpleNamespace(x=values),
    velocity=SimpleNamespace(x=2.0 * values),
    acceleration=SimpleNamespace(x=-values),
    meta=SimpleNamespace(
      disengagePredictions=SimpleNamespace(gasPressProbs=gas_press_probs),
    ),
  )


def test_parse_model_preserves_custom_launch_inputs():
  model = model_message()
  x, v, a, j, throttle_prob = LongitudinalPlanner.parse_model(model)

  np.testing.assert_array_equal(x, np.interp(T_IDXS, ModelConstants.T_IDXS, model.position.x))
  np.testing.assert_array_equal(v, np.interp(T_IDXS, ModelConstants.T_IDXS, model.velocity.x))
  np.testing.assert_array_equal(a, np.interp(T_IDXS, ModelConstants.T_IDXS, model.acceleration.x))
  np.testing.assert_array_equal(j, np.zeros(len(T_IDXS)))
  assert throttle_prob == 0.8


def test_parse_model_falls_back_safely_on_invalid_trajectory():
  x, v, a, j, throttle_prob = LongitudinalPlanner.parse_model(model_message(size=1, gas_press_probs=()))

  for trajectory in (x, v, a, j):
    np.testing.assert_array_equal(trajectory, np.zeros(len(T_IDXS)))
  assert throttle_prob == 1.0
