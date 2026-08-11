import math
from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.force_stops import ForceStops, GAS_OVERRIDE_S, STOP_POSITION_HOLD_S


DT = 0.05


class FakeSubMaster(dict):
  def __init__(self, *, model_length=20.0, should_stop=True, desired_accel=-0.6,
               v_ego=10.0, lead_present=False, model_valid=True):
    super().__init__(
      carState=SimpleNamespace(vEgo=v_ego, gasPressed=False, standstill=False),
      selfdriveState=SimpleNamespace(enabled=True, experimentalMode=True),
      radarState=SimpleNamespace(
        leadOne=SimpleNamespace(present=lead_present),
        leadTwo=SimpleNamespace(present=False),
      ),
      modelV2=SimpleNamespace(
        position=SimpleNamespace(x=[0.0, model_length]),
        action=SimpleNamespace(shouldStop=should_stop, desiredAcceleration=desired_accel),
      ),
    )
    self.valid = {"modelV2": model_valid, "radarState": True}


def arm(force_stops, sm):
  for _ in range(30):
    force_stops.update(sm)
  assert force_stops.forcing


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
