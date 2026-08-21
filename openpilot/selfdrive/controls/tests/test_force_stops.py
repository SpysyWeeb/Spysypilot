import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.force_stops import (A_STOP_ENVELOPE, DV_MAX, ForceStops,
                                                           GAS_OVERRIDE_S, LATCH_SETBACK, LATCH_THRESHOLD,
                                                           STOP_POSITION_HOLD_S)
from openpilot.selfdrive.controls.lib.force_stops import MPC_PROFILE_OFFSET_M
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE, LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner, get_cruise_accel
from openpilot.selfdrive.modeld.constants import ModelConstants


DT = 0.05


def test_force_stop_cruise_candidate_limits_decel_jerk_in_experimental_mode():
  args = (0.0, 10.0, 0.0, 0.0, SimpleNamespace(steerRatio=15.0, wheelbase=2.9), DT, 0.0, True)
  assert math.isclose(get_cruise_accel(True, *args), get_cruise_accel(False, *args))


class FakeSubMaster(dict):
  def __init__(self, *, model_length=20.0, should_stop=True, desired_accel=-0.6,
               v_ego=10.0, lead_present=False, model_valid=True, terminal_speed=0.0,
               conditional_stop_qualified=False, qualified_distance=0.0,
               model_mono_time=2_000_000_000, qualified_model_mono_time=2_000_000_000,
               conditional_stop_latched=False):
    super().__init__(
      carState=SimpleNamespace(vEgo=v_ego, gasPressed=False, brakePressed=False, standstill=False,
                               leftBlinker=False, rightBlinker=False),
      selfdriveState=SimpleNamespace(enabled=True, experimentalMode=True,
                                     conditionalStopQualified=conditional_stop_qualified,
                                     conditionalStopDistance=qualified_distance,
                                     conditionalStopModelMonoTime=qualified_model_mono_time,
                                     conditionalStopLatched=conditional_stop_latched),
      radarState=SimpleNamespace(
        leadOne=SimpleNamespace(present=lead_present),
        leadTwo=SimpleNamespace(present=False),
      ),
      modelV2=SimpleNamespace(
        position=SimpleNamespace(x=[model_length * i / (ModelConstants.IDX_N - 1) for i in range(ModelConstants.IDX_N)]),
        velocity=SimpleNamespace(x=[v_ego + (terminal_speed - v_ego) * i / (ModelConstants.IDX_N - 1)
                                    for i in range(ModelConstants.IDX_N)]),
        acceleration=SimpleNamespace(x=[desired_accel] * ModelConstants.IDX_N),
        orientation=SimpleNamespace(z=[0.0, 0.0]),
        action=SimpleNamespace(shouldStop=should_stop, desiredAcceleration=desired_accel, desiredCurvature=0.0),
      ),
    )
    self.valid = {"carState": True, "modelV2": model_valid, "radarState": True, "selfdriveState": True}
    self.alive = dict.fromkeys(self.valid, True)
    self.freq_ok = dict.fromkeys(self.valid, True)
    self.logMonoTime = {"modelV2": model_mono_time}

  def all_checks(self, services=None):
    services = self.keys() if services is None else services
    return all(self.valid.get(service, False) and self.alive.get(service, False) and self.freq_ok.get(service, False)
               for service in services)


def arm(force_stops, sm):
  for _ in range(30):
    force_stops.update(sm)
  assert force_stops.forcing


def test_force_stops_latches_on_the_bounded_early_horizon():
  sm = FakeSubMaster(model_length=32.149414, v_ego=10.098594, should_stop=False, desired_accel=-0.8)
  force_stops = ForceStops(dt=DT)

  for _ in range(30):
    force_stops.update(sm)

  assert force_stops.forcing
  assert 3.0 * sm["carState"].vEgo < sm["modelV2"].position.x[-1] < 3.25 * sm["carState"].vEgo

  nonbraking = FakeSubMaster(model_length=32.149414, v_ego=10.098594, should_stop=True, desired_accel=0.0)
  force_stops = ForceStops(dt=DT)
  for _ in range(30):
    force_stops.update(nonbraking)
  assert not force_stops.forcing


def test_force_stops_does_not_spend_nonbraking_confidence_on_one_braking_frame():
  sm = FakeSubMaster(model_length=32.149414, v_ego=10.098594, should_stop=True, desired_accel=0.0)
  force_stops = ForceStops(dt=DT)
  while force_stops.detect_filter.x < LATCH_THRESHOLD:
    force_stops.update(sm)
  assert not force_stops.forcing

  sm["modelV2"].action.desiredAcceleration = -0.8
  force_stops.update(sm)
  assert not force_stops.forcing

  sm["modelV2"].action.desiredAcceleration = 0.0
  sm["modelV2"].position.x[-1] = 30.0
  force_stops.update(sm)
  assert force_stops.forcing


def test_cem_qualified_stop_commits_before_classic_horizon():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster(model_length=95.0, v_ego=20.0, should_stop=False,
                     desired_accel=-0.8, terminal_speed=4.0,
                     conditional_stop_qualified=True, qualified_distance=95.0)

  force_stops.update(sm)

  assert force_stops.forcing
  assert math.isclose(force_stops.remaining, 95.0 - LATCH_SETBACK)


def test_widened_latch_requires_stable_braking_but_classic_latch_does_not():
  sm = FakeSubMaster(model_length=32.149414, v_ego=10.098594, should_stop=True, desired_accel=-0.8)
  force_stops = ForceStops(dt=DT)
  for _ in range(30):
    force_stops.update(sm)
  assert force_stops.forcing

  sm = FakeSubMaster(model_length=32.149414, v_ego=10.098594, should_stop=True, desired_accel=0.0)
  force_stops = ForceStops(dt=DT)
  while force_stops.detect_filter.x < LATCH_THRESHOLD:
    force_stops.update(sm)
  sm["modelV2"].action.desiredAcceleration = -0.8
  force_stops.update(sm)
  assert not force_stops.forcing

  sm["modelV2"].action.desiredAcceleration = 0.0
  sm["modelV2"].position.x[-1] = 30.0
  force_stops.update(sm)
  assert force_stops.forcing


def test_cem_qualified_stop_requires_current_valid_bound_authority():
  for mutation in (
    lambda sm: sm.valid.__setitem__("carState", False),
    lambda sm: sm.valid.__setitem__("selfdriveState", False),
    lambda sm: sm.alive.__setitem__("modelV2", False),
    lambda sm: sm.freq_ok.__setitem__("selfdriveState", False),
    lambda sm: setattr(sm["selfdriveState"], "conditionalStopModelMonoTime", sm.logMonoTime["modelV2"] - 500_000_000),
    lambda sm: setattr(sm["selfdriveState"], "conditionalStopDistance", float("nan")),
  ):
    force_stops = ForceStops(dt=DT)
    sm = FakeSubMaster(model_length=50.0, v_ego=20.0, should_stop=False, desired_accel=0.0,
                       terminal_speed=20.0, conditional_stop_qualified=True, qualified_distance=95.0)
    mutation(sm)
    assert math.isinf(force_stops.update(sm))
    assert not force_stops.forcing


def test_brake_and_nonfinite_inputs_release_without_priming():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster(model_length=95.0, v_ego=20.0, should_stop=False,
                     desired_accel=-0.8, terminal_speed=4.0,
                     conditional_stop_qualified=True, qualified_distance=95.0)
  force_stops.update(sm)
  assert force_stops.forcing

  sm["carState"].brakePressed = True
  assert math.isinf(force_stops.update(sm))
  assert not force_stops.forcing

  sm["carState"].brakePressed = False
  sm["selfdriveState"].conditionalStopQualified = False
  sm["modelV2"].action.desiredAcceleration = -math.inf
  for _ in range(30):
    assert math.isinf(force_stops.update(sm))
  assert force_stops.detect_filter.x == 0.0


def test_gas_override_keeps_priority_when_both_pedals_are_pressed():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster()
  sm["carState"].gasPressed = True
  sm["carState"].brakePressed = True
  sm["carState"].vEgo = float("nan")

  assert math.isinf(force_stops.update(sm))
  assert force_stops.override_timer == GAS_OVERRIDE_S


def test_committed_target_stays_fixed_and_mpc_releases_after_crossing():
  force_stops = ForceStops(dt=1.0)
  sm = FakeSubMaster(model_length=100.0, v_ego=4.0, should_stop=False, desired_accel=0.0)
  force_stops.forcing = True
  force_stops.detect_filter.x = 1.0
  force_stops.position_hold_remaining = 4.0
  force_stops.remaining = 6.5
  world_targets = []
  travel = 0.0

  for _ in range(3):
    force_stops.update(sm)
    travel += sm["carState"].vEgo
    world_targets.append(travel + force_stops.remaining)

  np.testing.assert_allclose(world_targets, world_targets[0])
  assert force_stops.remaining < 0.0


def test_cem_qualified_stop_starts_live_shaping_before_latch():
  v_ego = 19.477
  for model_length in (116.146, 150.0):
    expected_cap = max(math.sqrt(2.0 * A_STOP_ENVELOPE * (model_length - MPC_PROFILE_OFFSET_M)), v_ego - DV_MAX)
    force_stops = ForceStops(dt=DT)
    sm = FakeSubMaster(model_length=model_length, should_stop=False, desired_accel=-0.73,
                       v_ego=v_ego, terminal_speed=4.066)

    assert math.isclose(force_stops.update(sm), expected_cap)
    assert not force_stops.forcing


def test_starpilot_profile_uses_six_meter_offset():
  assert A_STOP_ENVELOPE == 0.65
  assert MPC_PROFILE_OFFSET_M == 6.0
  force_stops = ForceStops(dt=DT)
  profile_distance = 20.0
  force_stops.forcing = True
  force_stops.remaining = MPC_PROFILE_OFFSET_M + profile_distance
  sm = FakeSubMaster(model_length=force_stops.remaining + LATCH_SETBACK, v_ego=0.0)

  expected_cap = math.sqrt(2.0 * A_STOP_ENVELOPE * profile_distance)
  assert math.isclose(force_stops.update(sm), expected_cap)


def test_committed_stop_is_the_mpc_obstacle_until_final_landing():
  absent_lead = SimpleNamespace(present=False, dRel=0.0, vLead=0.0, aLeadK=0.0, aLeadTau=1.5, modelProb=0.0)
  radar_state = SimpleNamespace(leadOne=absent_lead, leadTwo=absent_lead)
  mpc = LongitudinalMpc()
  mpc.set_cur_state(10.0, 0.0)
  mpc.update(radar_state)
  baseline_obstacle = mpc.params[:, 2].copy()

  for _ in range(4):
    mpc.update(radar_state, stop_x=30.0)
  np.testing.assert_allclose(mpc.params[:, 2], 30.0 + STOP_DISTANCE)
  assert mpc.source == LongitudinalPlanSource.stop
  assert mpc.a_solution[1] < 0.0
  assert mpc.crash_cnt == 0

  close_lead = SimpleNamespace(present=True, dRel=15.0, vLead=0.0, aLeadK=0.0, aLeadTau=1.5, modelProb=1.0)
  mpc.update(SimpleNamespace(leadOne=close_lead, leadTwo=absent_lead), stop_x=30.0)
  assert mpc.source == LongitudinalPlanSource.lead0
  assert np.all(mpc.params[:, 2] <= 30.0 + STOP_DISTANCE)

  mpc.update(SimpleNamespace(leadOne=absent_lead, leadTwo=close_lead), stop_x=30.0)
  assert mpc.source == LongitudinalPlanSource.lead1
  assert np.all(mpc.params[:, 2] <= 30.0 + STOP_DISTANCE)

  for active_stop_x in (-STOP_DISTANCE, -STOP_DISTANCE + 0.25, 0.0, STOP_DISTANCE):
    mpc.update(radar_state, stop_x=active_stop_x)
    np.testing.assert_allclose(mpc.params[:, 2], active_stop_x + STOP_DISTANCE)
    assert mpc.solution_status == 0
    assert np.all(np.isfinite(mpc.a_solution))

  for released_stop_x in (-STOP_DISTANCE - 0.25, float("-inf"), float("inf"), float("nan")):
    mpc.update(radar_state, stop_x=released_stop_x)
    np.testing.assert_allclose(mpc.params[:, 2], baseline_obstacle)

  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster(model_length=MPC_PROFILE_OFFSET_M + LATCH_SETBACK, v_ego=0.0)
  force_stops.forcing = True
  force_stops.remaining = MPC_PROFILE_OFFSET_M
  assert force_stops.update(sm) == 0.0


def test_committed_stop_stays_with_mpc_until_force_stops_releases():
  v_ego = 6.4
  CP = SimpleNamespace(openpilotLongitudinalControl=True, longitudinalActuatorDelay=0.2,
                       steerRatio=15.0, wheelbase=2.9)
  planner = LongitudinalPlanner(CP, init_v=v_ego)
  planner.curve_speed_limiter.update = lambda model, v_cruise, v_ego=0.0, lateral_active=False, roll=0.0, torque_params=None: v_cruise
  planner.blotv2.update = lambda *args: SimpleNamespace(jerk_scale=1.0, t_follow=1.45, emergency=False)
  planner.lead_departure.update = lambda **kwargs: False
  sm = FakeSubMaster(model_length=100.0, should_stop=False, desired_accel=0.0, v_ego=v_ego)
  sm["carState"].vCruise = 100.0
  sm["carState"].aEgo = 0.0
  sm["carState"].steeringAngleDeg = 0.0
  sm["carState"].steeringPressed = False
  sm["controlsState"] = SimpleNamespace(forceDecel=False, longControlState=LongCtrlState.pid)
  sm["carControl"] = SimpleNamespace(orientationNED=[], latActive=True)
  sm["vehicleParameters"] = SimpleNamespace(angleOffsetDeg=0.0, roll=0.0)
  sm["lateralTorqueParameters"] = SimpleNamespace(useParams=False)
  sm["selfdriveState"].personality = 1
  sm["modelV2"].meta = SimpleNamespace(disengagePredictions=SimpleNamespace(gasPressProbs=[]),
                                        laneChangeState=log.LaneChangeState.off)
  sm["modelV2"].leadsV3 = []
  absent_lead = SimpleNamespace(present=False, dRel=0.0, vLead=0.0, aLeadK=0.0, aLeadTau=1.5, modelProb=0.0)
  sm["radarState"] = SimpleNamespace(leadOne=absent_lead, leadTwo=absent_lead)
  sm.all_checks = lambda services=None: True

  planner.force_stops.forcing = True
  planner.force_stops.remaining = MPC_PROFILE_OFFSET_M + 0.48
  planner.force_stops.position_hold_remaining = 1.0
  planner.update(sm)
  before_remaining = planner.force_stops.remaining
  before_accel = planner.output_a_target
  planner.update(sm)

  assert before_remaining > MPC_PROFILE_OFFSET_M >= planner.force_stops.remaining
  np.testing.assert_allclose(planner.mpc.params[:, 2], planner.force_stops.remaining + STOP_DISTANCE)
  assert planner.output_a_target <= before_accel + 0.25

  sm["carState"].vEgo = 0.2
  planner.update(sm)
  np.testing.assert_allclose(planner.mpc.params[:, 2], planner.force_stops.remaining + STOP_DISTANCE)
  assert planner.output_should_stop

  sm["carState"].vEgo = 1.0
  planner.force_stops.remaining = -STOP_DISTANCE - 1.0
  planner.force_stops.position_hold_remaining = 1.0
  planner.update(sm)
  assert planner.force_stops.forcing
  np.testing.assert_allclose(planner.mpc.params[:, 2], 0.0)

  sm["selfdriveState"].experimentalMode = False
  planner.update(sm)
  assert not planner.force_stops.forcing
  assert np.all(planner.mpc.params[:, 2] > STOP_DISTANCE)

  sm["selfdriveState"].experimentalMode = True
  sm["selfdriveState"].conditionalStopLatched = True
  sm["carState"].vEgo = 0.2
  sm["carState"].standstill = True
  sm["modelV2"].action.desiredAcceleration = 0.0
  planner.a_cruise = 1.0
  planner.output_a_target = 1.0
  planner.v_desired_filter.x = 0.2
  with patch("openpilot.selfdrive.controls.lib.longitudinal_planner.get_accel_from_plan", return_value=0.2):
    planner.update(sm)
  assert not planner.force_stops.forcing
  assert planner.output_should_stop


def test_force_stops_retains_and_clamps_the_crossed_mpc_obstacle():
  force_stops = ForceStops(dt=1.0)
  sm = FakeSubMaster(model_length=20.0, v_ego=1.0, should_stop=False, desired_accel=0.0)
  force_stops.forcing = True
  force_stops.detect_filter.x = 1.0
  force_stops.position_hold_remaining = 1.0
  force_stops.remaining = -STOP_DISTANCE + sm["carState"].vEgo

  assert force_stops.update(sm) == 0.0
  assert force_stops.forcing
  assert force_stops.remaining == -STOP_DISTANCE

  sm["selfdriveState"].experimentalMode = False
  assert math.isinf(force_stops.update(sm))
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
  assert math.isclose(force_stops.remaining, 20.0 - LATCH_SETBACK)

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

  hold_frames = int(0.5 / DT)
  assert all(math.isfinite(force_stops.update(sm)) for _ in range(hold_frames))
  cap = 0.0
  for _ in range(int(STOP_POSITION_HOLD_S / DT) + 1):
    cap = force_stops.update(sm)
  assert math.isinf(cap)


def test_new_evidence_refreshes_position_hold():
  force_stops = ForceStops(dt=DT)
  sm = FakeSubMaster(model_length=1.0, v_ego=0.5)
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


def test_nonfinite_trajectory_samples_cannot_prime_detector_or_latch():
  for axis in ("position", "velocity"):
    for bad_value in (float("nan"), float("inf"), float("-inf")):
      for sample in (10, -1):
        force_stops = ForceStops(dt=DT)
        sm = FakeSubMaster()
        sm["modelV2"].position.x = [20.0 * i / (ModelConstants.IDX_N - 1) for i in range(ModelConstants.IDX_N)]
        sm["modelV2"].velocity.x = [10.0 * (1.0 - i / (ModelConstants.IDX_N - 1)) for i in range(ModelConstants.IDX_N)]
        getattr(sm["modelV2"], axis).x[sample] = bad_value

        for _ in range(30):
          assert math.isinf(force_stops.update(sm))
        assert force_stops.detect_filter.x == 0.0
        assert not force_stops.forcing
        assert force_stops.position_hold_remaining == 0.0

        sm["modelV2"].position.x = [20.0 * i / (ModelConstants.IDX_N - 1) for i in range(ModelConstants.IDX_N)]
        sm["modelV2"].velocity.x = [10.0 * (1.0 - i / (ModelConstants.IDX_N - 1)) for i in range(ModelConstants.IDX_N)]
        assert math.isinf(force_stops.update(sm))
        assert not force_stops.forcing
        arm(force_stops, sm)
