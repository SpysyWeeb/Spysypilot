import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.constants import steady_follow_distance


@pytest.mark.parametrize("v_ego, t_follow, expected", [
  (29.0, 1.0, 36.0),  # aggressive
  (29.0, 1.45, 49.05),  # standard
  (29.0, 1.75, 57.75),  # relaxed
  (0.0, 1.0, 7.0),  # a stop is STOP_DISTANCE whatever the personality
  (0.0, 1.75, 7.0),
  (10.0, 1.2, 19.0),  # a padded follow time is used as given
])
def test_steady_follow_distance(v_ego, t_follow, expected):
  assert steady_follow_distance(v_ego, t_follow) == pytest.approx(expected)


def test_lead_target_distance_field():
  plan = log.LongitudinalPlan.new_message()
  assert plan.leadTargetDistance == 0.0  # unset reads 0, which the car side treats as no target
  plan.leadTargetDistance = 36.0
  assert plan.leadTargetDistance == 36.0
