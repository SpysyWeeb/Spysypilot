import unittest

import numpy as np

from openpilot.selfdrive.controls.lib.blatv2.contextual_dynamics_study import (
  DynamicsCandidate,
  WindowBatch,
  maneuver_context,
  nearest_speed_node,
  predict_final_angles,
)
from openpilot.selfdrive.controls.lib.blatv2.plant import (
  RackState,
  step_rack_dynamics,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  PhysicalParameters,
)
from tools.blatv2_context_study import (
  FIELD_CANDIDATE_SCHEDULE,
  _scheduled_evaluation,
)


class TestBLaTv2ContextualDynamicsStudy(unittest.TestCase):
  def test_maneuver_context_uses_magnitude_change_not_named_turn_state(self):
    self.assertEqual(maneuver_context(0.2, 0.6), ("turn_in", "left"))
    self.assertEqual(maneuver_context(-0.6, -0.2), ("unwind", "right"))
    self.assertEqual(maneuver_context(0.4, 0.42), ("steady", "left"))
    self.assertIsNone(maneuver_context(-0.2, 0.2))
    self.assertIsNone(maneuver_context(0.0, 0.01))

  def test_nearest_speed_node_has_deterministic_lower_tie(self):
    observed = nearest_speed_node(np.asarray([2.5, 7.5, 17.5, 25.0, 40.0]))
    np.testing.assert_array_equal(observed, [0.0, 5.0, 15.0, 20.0, 30.0])

  def test_vector_rollout_matches_live_scalar_kernel(self):
    candidate = DynamicsCandidate(1600.0, 8.0)
    applied = np.asarray([[0.0, 0.2, 0.2, -0.1]], dtype=np.float64)
    aligning = np.asarray([[0.0, 0.02, 0.03, 0.01]], dtype=np.float64)
    dt = np.full_like(applied, 0.01)
    windows = WindowBatch(
      initial_angle_deg=np.asarray([1.0]),
      initial_rate_deg_s=np.asarray([0.0]),
      applied_torque=applied,
      aligning_torque=aligning,
      dt_s=dt,
      measured_final_angle_deg=np.asarray([0.0]),
      speed_node_mps=np.asarray([10.0]),
      phase=("turn_in",),
      direction=("left",),
    )
    predicted = predict_final_angles(
      windows,
      candidate,
      static_friction_torque=0.13,
      kinetic_friction_torque=0.13,
      rack_rate_resolution_deg_s=4.0,
    )[0]

    parameters = PhysicalParameters(
      torque_per_lateral_accel=0.39335,
      rack_gain_deg_s2_per_torque=1600.0,
      rack_damping_per_s=8.0,
      transport_delay_s=0.1,
      static_friction_torque=0.13,
      kinetic_friction_torque=0.13,
      rack_rate_resolution_deg_s=4.0,
      confidence=1.0,
      qualified=True,
    )
    state = RackState(1.0, 0.0, 0.0)
    for index in range(applied.shape[1]):
      state = step_rack_dynamics(
        state,
        applied[0, index],
        aligning[0, index],
        parameters,
        0.0,
        dt[0, index],
      ).state
    self.assertEqual(predicted, state.angle_deg)

  def test_invalid_candidate_and_batch_fail_closed(self):
    with self.assertRaises(ValueError):
      DynamicsCandidate(0.0, 8.0)
    with self.assertRaises(ValueError):
      WindowBatch(
        initial_angle_deg=np.asarray([0.0]),
        initial_rate_deg_s=np.asarray([0.0]),
        applied_torque=np.asarray([[0.0]]),
        aligning_torque=np.asarray([[0.0]]),
        dt_s=np.asarray([[0.0]]),
        measured_final_angle_deg=np.asarray([0.0]),
        speed_node_mps=np.asarray([0.0]),
        phase=("steady",),
        direction=("left",),
      )

  def test_field_candidate_selects_only_its_speed_local_strata(self):
    self.assertEqual(FIELD_CANDIDATE_SCHEDULE, {
      0.0: "4000/10",
      5.0: "4000/10",
      10.0: "3200/14",
      15.0: "3200/14",
      20.0: "3200/14",
      30.0: "3200/14",
    })
    route = {
      "evaluations": {
        "4000/10": {
          "strata": {
            "5.turn_in.left": {"count": 2, "rmseDeg": 4.0},
            "10.turn_in.left": {"count": 3, "rmseDeg": 40.0},
          },
        },
        "3200/14": {
          "strata": {
            "5.turn_in.left": {"count": 2, "rmseDeg": 30.0},
            "10.turn_in.left": {"count": 3, "rmseDeg": 6.0},
          },
        },
      },
    }

    selected = _scheduled_evaluation(route, FIELD_CANDIDATE_SCHEDULE)
    self.assertEqual(selected["count"], 5)
    self.assertEqual(set(selected["strata"]), {
      "5.turn_in.left",
      "10.turn_in.left",
    })
    self.assertAlmostEqual(
      selected["rmseDeg"],
      ((2 * 4.0 ** 2 + 3 * 6.0 ** 2) / 5) ** 0.5,
    )


if __name__ == "__main__":
  unittest.main()
