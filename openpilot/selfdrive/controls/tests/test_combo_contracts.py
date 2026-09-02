"""Cross-branch contracts that only exist on combo, where the branches meet."""
from openpilot.selfdrive.controls.lib.smooth_stops import STOP_KISS_DECEL
from openpilot.selfdrive.controls.lib.stop_landing import KISS_DECEL


def test_the_handoff_kiss_is_the_corridors_kiss():
  # LongControl's settle bounds the plan by its kiss; the planner corridor pins the plan at its own. If they differ, one side
  # silently changes the other's landing (audit 2026-09-02)
  assert STOP_KISS_DECEL == KISS_DECEL
