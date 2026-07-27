import math
from types import SimpleNamespace
import unittest

from openpilot.selfdrive.controls.lib.blatv2.plant import AlignParams, PlantParams, PlantTwin
from openpilot.selfdrive.controls.lib.blatv2.reference import build_reference, horizon, torque_demand


def params() -> PlantParams:
  return PlantParams(4000.0, 10.0, 0.05, 0.12, 409, 4, 7, 1, True)


def align_params() -> AlignParams:
  return AlignParams(2000.0, 3.0, 1.2, 100000.0, 110000.0, 15.0, 0.0, 2.5, 0.0)


class TestReference(unittest.TestCase):
  def test_constant_plan_is_exact_scalar_everywhere(self) -> None:
    times, reference = build_reference(0.012, [0.0, 0.5, 1.0, 1.5], [0.02] * 4, 0.3, 1.5)
    self.assertEqual(times, (0.0, 0.5, 1.0, 1.5))
    self.assertEqual(reference, (0.012,) * 4)

  def test_horizon_is_runtime_limit_derived(self) -> None:
    expected = (409 / 7 + 409 / 4) * 0.01 + 0.12 + 0.2
    self.assertTrue(math.isclose(horizon(params()), expected))

  def test_transparency_for_already_feasible_smooth_demand(self) -> None:
    twin = PlantTwin(params(), align_params())
    torque_params = SimpleNamespace(latAccelFactor=3.0, latAccelOffset=0.0, friction=0.0)
    raw = torque_demand(-0.0001, 10.0, 0.0, torque_params)
    self.assertLess(abs(raw), 4 / 409)
    self.assertTrue(math.isclose(twin.apply_slew(0.0, raw), raw, abs_tol=1e-9))
