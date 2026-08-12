import math
from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.force_stops import (A_STOP_ENVELOPE, DV_MAX, ForceStops, GAS_OVERRIDE_S,
                                                           LATCH_SETBACK, STOP_POSITION_HOLD_S)


DT = 0.05


class FakeSubMaster(dict):
  def __init__(self, *, model_length=20.0, should_stop=True, desired_accel=-0.6,
               v_ego=10.0, lead_present=False, model_valid=True, terminal_speed=0.0):
    super().__init__(
      carState=SimpleNamespace(vEgo=v_ego, gasPressed=False, standstill=False,
                               leftBlinker=False, rightBlinker=False),
      selfdriveState=SimpleNamespace(enabled=True, experimentalMode=True),
      radarState=SimpleNamespace(
        leadOne=SimpleNamespace(present=lead_present),
        leadTwo=SimpleNamespace(present=False),
      ),
      modelV2=SimpleNamespace(
        position=SimpleNamespace(x=[0.0, model_length]),
        velocity=SimpleNamespace(x=[v_ego, terminal_speed]),
        orientation=SimpleNamespace(z=[0.0, 0.0]),
        action=SimpleNamespace(shouldStop=should_stop, desiredAcceleration=desired_accel, desiredCurvature=0.0),
      ),
    )
    self.valid = {"modelV2": model_valid, "radarState": True}


def arm(force_stops, sm):
  for _ in range(30):
    force_stops.update(sm)
  assert force_stops.forcing


def test_cem_qualified_stop_starts_live_shaping_before_latch():
  v_ego = 19.477
  for model_length in (116.146, 150.0):
    expected_cap = max(math.sqrt(2.0 * A_STOP_ENVELOPE * model_length), v_ego - DV_MAX)
    force_stops = ForceStops(dt=DT)
    sm = FakeSubMaster(model_length=model_length, should_stop=False, desired_accel=-0.73,
                       v_ego=v_ego, terminal_speed=4.066)

    assert math.isclose(force_stops.update(sm), expected_cap)
    assert not force_stops.forcing


def test_incomplete_early_trajectory_cannot_shape():
  for field, axis in (("position", "x"), ("velocity", "x"), ("orientation", "z")):
    force_stops = ForceStops(dt=DT)
    sm = FakeSubMaster(model_length=116.146, should_stop=False, desired_accel=-0.73,
                       v_ego=19.477, terminal_speed=4.066)
    trajectory = getattr(sm["modelV2"], field)
    setattr(trajectory, axis, [getattr(trajectory, axis)[-1]])

    assert math.isinf(force_stops.update(sm))


def test_nonfinite_early_action_cannot_shape():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster(model_length=116.146, should_stop=False, desired_accel=-math.inf,
                     v_ego=19.477, terminal_speed=4.066)

  assert math.isinf(force_stops.update(sm))


def test_filtered_lead_blocks_immediate_early_shaping_after_dropout():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster(model_length=116.146, should_stop=False, desired_accel=-0.73,
                     v_ego=19.477, terminal_speed=4.066, lead_present=True)
  for _ in range(30):
    assert math.isinf(force_stops.update(sm))
  sm["radarState"].leadOne.present = False

  assert math.isinf(force_stops.update(sm))


def test_early_shaping_does_not_prime_physical_latch():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster(model_length=116.146, should_stop=False, desired_accel=-0.73,
                     v_ego=19.477, terminal_speed=4.066)
  for _ in range(30):
    assert math.isfinite(force_stops.update(sm))
  sm["modelV2"].position.x[-1] = 58.0

  force_stops.update(sm)

  assert not force_stops.forcing


def test_committed_endpoint_keeps_configured_setback():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster()
  while not force_stops.forcing:
    force_stops.update(sm)
  assert math.isclose(force_stops.remaining, 20.0 - LATCH_SETBACK - sm["carState"].vEgo * DT)

  force_stops.remaining = 3.0
  sm["carState"].vEgo = 4.0
  sm["modelV2"].position.x[-1] = 7.0
  force_stops.update(sm)
  assert math.isclose(force_stops.remaining, 3.0 - 4.0 * DT)

  outward_force_stops = ForceStops(dt=1.0)
  outward_force_stops.forcing = True
  outward_force_stops.detect_filter.x = 1.0
  outward_force_stops.remaining = 6.0
  outward_sm = FakeSubMaster(model_length=10.0, v_ego=4.0)
  outward_force_stops.update(outward_sm)
  assert math.isclose(outward_force_stops.remaining, 10.0 - LATCH_SETBACK)

  force_stops.remaining = 3.5
  sm["carState"].vEgo = 0.0
  sm["modelV2"].position.x[-1] = 5.0
  force_stops.update(sm)
  assert math.isclose(force_stops.remaining, 3.5 - 2.0 * DT)

  near_force_stops = ForceStops(dt=DT)
  near_sm = FakeSubMaster(model_length=1.0, v_ego=1.0)
  cap = math.inf
  while not near_force_stops.forcing:
    cap = near_force_stops.update(near_sm)
  assert near_force_stops.remaining == 0.0
  assert cap == max(0.0, near_sm["carState"].vEgo - DV_MAX)
  assert math.isfinite(cap)


def test_latched_position_survives_brief_model_clear():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster()
  arm(force_stops, sm)

  sm["modelV2"].position.x[-1] = 100.0
  sm["modelV2"].action.shouldStop = False
  sm["modelV2"].action.desiredAcceleration = 0.0

  hold_frames = int((STOP_POSITION_HOLD_S - 0.5) / DT)
  assert all(math.isfinite(force_stops.update(sm)) for _ in range(hold_frames))
  cap = 0.0
  for _ in range(int(1.0 / DT)):
    cap = force_stops.update(sm)
  assert math.isinf(cap)


def test_new_evidence_refreshes_position_hold():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster()
  arm(force_stops, sm)

  sm["modelV2"].position.x[-1] = 100.0
  sm["modelV2"].action.shouldStop = False
  sm["modelV2"].action.desiredAcceleration = 0.0
  for _ in range(int(3.0 / DT)):
    force_stops.update(sm)

  sm["modelV2"].position.x[-1] = 20.0
  sm["modelV2"].action.shouldStop = True
  force_stops.update(sm)
  sm["modelV2"].position.x[-1] = 100.0
  sm["modelV2"].action.shouldStop = False

  assert all(math.isfinite(force_stops.update(sm)) for _ in range(int(3.5 / DT)))


def test_clear_model_cannot_move_latched_point_outward():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster()
  arm(force_stops, sm)
  remaining = force_stops.remaining

  sm["modelV2"].position.x[-1] = 100.0
  sm["modelV2"].action.shouldStop = False
  sm["modelV2"].action.desiredAcceleration = 0.0
  force_stops.update(sm)

  expected = max(remaining - sm["carState"].vEgo * DT, 0.0)
  assert math.isclose(force_stops.remaining, expected)


def test_stale_should_stop_cannot_move_latched_point_to_long_path():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster()
  arm(force_stops, sm)
  remaining = force_stops.remaining

  sm["modelV2"].position.x[-1] = 100.0
  force_stops.update(sm)

  expected = max(remaining - sm["carState"].vEgo * DT, 0.0)
  assert math.isclose(force_stops.remaining, expected)


def test_raw_lead_immediately_releases_latched_position():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster()
  arm(force_stops, sm)

  sm["radarState"].leadOne.present = True

  assert math.isinf(force_stops.update(sm))
  assert not force_stops.forcing


def test_standstill_clears_latch_before_launch():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster()
  arm(force_stops, sm)

  sm["carState"].standstill = True
  assert math.isinf(force_stops.update(sm))
  sm["carState"].standstill = False

  assert math.isinf(force_stops.update(sm))
  assert not force_stops.forcing


def test_immediate_exit_conditions_bypass_position_hold():
  for condition in ("disabled", "gas", "model_invalid", "radar_invalid"):
    force_stops = ForceStops(dt=DT)
    sm = FakeSubMaster()
    arm(force_stops, sm)

    if condition == "disabled":
      sm["selfdriveState"].enabled = False
    elif condition == "gas":
      sm["carState"].gasPressed = True
    else:
      sm.valid["modelV2" if condition == "model_invalid" else "radarState"] = False

    assert math.isinf(force_stops.update(sm))
    assert not force_stops.forcing


def test_gas_bypasses_pre_latch_shaping():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster()
  for _ in range(9):
    force_stops.update(sm)

  sm["carState"].gasPressed = True

  assert math.isinf(force_stops.update(sm))


def test_gas_override_survives_experimental_mode_release():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster()
  sm["selfdriveState"].experimentalMode = False
  sm["carState"].gasPressed = True

  assert math.isinf(force_stops.update(sm))
  assert force_stops.override_timer == GAS_OVERRIDE_S

  sm["carState"].gasPressed = False
  for _ in range(int(1.0 / DT)):
    force_stops.update(sm)
  sm["selfdriveState"].experimentalMode = True

  assert math.isinf(force_stops.update(sm))


def test_secondary_raw_lead_bypasses_pre_latch_shaping():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster()
  for _ in range(9):
    force_stops.update(sm)

  sm["radarState"].leadTwo.present = True

  assert math.isinf(force_stops.update(sm))


def test_invalid_model_cannot_arm_force_stop():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster(model_valid=False)

  assert all(math.isinf(force_stops.update(sm)) for _ in range(30))
  assert not force_stops.forcing


def test_nonfinite_model_position_releases_latched_stop():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster()
  arm(force_stops, sm)

  sm["modelV2"].position.x[-1] = float("nan")

  assert math.isinf(force_stops.update(sm))
  assert not force_stops.forcing
