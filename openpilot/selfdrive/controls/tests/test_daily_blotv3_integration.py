import math

import numpy as np
import pytest

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from opendbc.car.interfaces import ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.accel_boost import AccelBoost
from openpilot.selfdrive.controls.lib.force_stops import ForceStopsResult
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_lead import LEAD_T_IDXS, LeadObservation
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LEAD_ABSENCE_FRAMES
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner, get_max_accel
from openpilot.selfdrive.controls.tests.test_longitudinal_lead import model_lead, radar_lead
from openpilot.selfdrive.controls.tests.test_longitudinal_planner import IdealCar, j_cruise_step
from openpilot.selfdrive.modeld.constants import ModelConstants


class PlannerMessages(dict):
  def __init__(self, v_ego=15.0, experimental=True, e2e_accel=-0.2):
    services = ('carState', 'carControl', 'controlsState', 'selfdriveState', 'modelV2', 'radarState', 'vehicleParameters')
    super().__init__((s, getattr(messaging.new_message(s), s)) for s in services)
    self.invalid = set()
    self.logMonoTime = dict.fromkeys(services, 0)
    self['carState'].vEgo = v_ego
    self['carState'].vCruise = 100.0
    self['controlsState'].longControlState = LongCtrlState.pid
    self['selfdriveState'].enabled = True
    self['selfdriveState'].experimentalMode = experimental
    self['selfdriveState'].personality = log.LongitudinalPersonality.standard
    self['modelV2'].action.desiredAcceleration = e2e_accel

  def all_checks(self, service_list=None):
    return not self.invalid.intersection(self if service_list is None else service_list)


def car_params():
  cp = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
  cp.openpilotLongitudinalControl = True
  cp.longitudinalActuatorDelay = 0.5
  return cp


def test_blotv3_launch_envelope_respects_daily_vehicle_ceiling():
  assert math.isclose(get_max_accel(0.0), ACCEL_MAX)


def test_upstream_accelboost_survives_planner_and_publication():
  planner = LongitudinalPlanner(car_params(), init_v=15.0)
  assert isinstance(planner.accel_boost, AccelBoost)
  planner.accel_boost.total_boost = 0.1
  sm = PlannerMessages()
  planner.update(sm)
  assert math.isclose(planner.output_a_target, -0.1, abs_tol=1e-6)

  class Publisher:
    def send(self, service, message):
      self.message = message

  pm = Publisher()
  planner.publish(sm, pm)
  assert math.isclose(pm.message.longitudinalPlan.accelBoost, 0.1, abs_tol=1e-6)
  assert pm.message.longitudinalPlan.longitudinalPlanSource == log.LongitudinalPlan.LongitudinalPlanSource.e2e


def test_e2e_admission_compares_the_boosted_target_with_the_stop_profile(monkeypatch):
  planner = LongitudinalPlanner(car_params(), init_v=15.0)
  planner.accel_boost.total_boost = 0.2
  # The raw model barely clears the profile's urgency margin; the boosted target does not.
  monkeypatch.setattr(planner.force_stops, 'update', lambda *args: ForceStopsResult(a_target=-0.3))
  planner.update(PlannerMessages(e2e_accel=-0.81))
  assert planner.plan_winner == 'profile'
  assert math.isclose(planner.output_a_target, -0.3, abs_tol=1e-6)


@pytest.mark.parametrize('profile,e2e,previous,eligible', [(-0.8, -0.2, 'cruise', False), (-0.3, -0.6, 'cruise', False),
                                                         (-0.3, -0.9, 'cruise', True), (-0.3, -0.35, 'e2e', False)])
def test_accelboost_learns_only_when_the_admitted_model_limits_the_plan(monkeypatch, profile, e2e, previous, eligible):
  planner = LongitudinalPlanner(car_params(), init_v=15.0)
  planner.plan_winner = previous
  sm = PlannerMessages(e2e_accel=e2e)
  monkeypatch.setattr(planner.force_stops, 'update', lambda *args: ForceStopsResult(a_target=profile))
  planner.update(sm)
  assert planner.accel_boost.boost_eligible == eligible
  sm['carState'].gasPressed = True
  planner.update(sm)
  assert (planner.accel_boost.total_boost > 0.0) == eligible


def set_lead(sm, slot, distance):
  lead = getattr(sm['radarState'], slot)
  lead.present = True
  lead.dRel = distance
  lead.vLead = 0.0
  lead.vLeadK = 0.0
  lead.aLeadK = 0.0
  lead.modelProb = 1.0


@pytest.mark.parametrize('slot', ['leadOne', 'leadTwo'])
def test_landing_cannot_ease_urgent_braking_for_either_mpc_lead(slot):
  planner = LongitudinalPlanner(car_params(), init_v=1.0)
  sm = PlannerMessages(v_ego=1.0, e2e_accel=-3.0)
  sm['modelV2'].action.shouldStop = True
  set_lead(sm, 'leadOne', 50.0)
  set_lead(sm, slot, 1.0)
  planner.update(sm)
  assert math.isclose(planner.output_a_target, -3.0, abs_tol=1e-6)
  assert planner.output_should_stop


@pytest.mark.parametrize('slot', ['leadOne', 'leadTwo'])
@pytest.mark.parametrize('dropout', ['absent', 'radar_invalid', 'nonfinite'])
def test_landing_urgency_survives_only_the_bounded_lead_absence_window(slot, dropout):
  planner = LongitudinalPlanner(car_params(), init_v=1.0)
  sm = PlannerMessages(v_ego=1.0, e2e_accel=-3.0)
  sm['modelV2'].action.shouldStop = True
  set_lead(sm, 'leadOne', 50.0)
  set_lead(sm, slot, 1.0)
  planner.update(sm)
  assert math.isclose(planner.output_a_target, -3.0, abs_tol=1e-6)

  if dropout == 'radar_invalid':
    # a frame radarState fails its checks is planned as its message: the lead it still reports is no dropout, its urgency stays
    sm.invalid.add('radarState')
    for _ in range(LEAD_ABSENCE_FRAMES + 1):
      planner.update(sm)
      assert math.isclose(planner.output_a_target, -3.0, abs_tol=1e-6)
      assert planner.output_should_stop and planner.stop_landing.required_decel > planner.stop_landing.bound(1.0)
    return
  if dropout == 'nonfinite':
    getattr(sm['radarState'], slot).aLeadK = math.nan
  else:
    getattr(sm['radarState'], slot).present = False
  for _ in range(LEAD_ABSENCE_FRAMES - 1):
    planner.update(sm)
    assert math.isclose(planner.output_a_target, -3.0, abs_tol=1e-6)
    assert planner.output_should_stop
  planner.update(sm)
  assert planner.stop_landing.required_decel <= planner.stop_landing.bound(1.0)
  allowed = float(np.interp(1.0, *planner.stop_landing.release_jerk)) * planner.dt
  assert -3.0 < planner.output_a_target <= -3.0 + allowed + 1e-9


@pytest.mark.parametrize('invalid_service', [None, 'modelV2', 'radarState'])
def test_both_forecasts_require_fresh_model_and_radar_at_the_real_mpc_boundary(monkeypatch, invalid_service):
  planner = LongitudinalPlanner(car_params(), init_v=10.0)
  sm = PlannerMessages(v_ego=10.0)
  sm['radarState'].leadOne = radar_lead(dRel=25.0, vLead=5.0, vLeadK=5.0)
  sm['radarState'].leadTwo = radar_lead(dRel=40.0, vLead=6.0, vLeadK=6.0)
  sm['modelV2'].leadsV3 = [model_lead(5.0 + 0.5 * LEAD_T_IDXS), model_lead(6.0 + 0.5 * LEAD_T_IDXS)]
  if invalid_service is not None:
    sm.invalid.add(invalid_service)
  actual = []
  process_lead_model = planner.mpc.process_lead_model

  def observe(anchor, radar):
    prediction = process_lead_model(anchor, radar)
    actual.append((anchor, prediction, planner.mpc.process_lead(radar)))
    return prediction

  monkeypatch.setattr(planner.mpc, 'process_lead_model', observe)
  planner.update(sm)
  assert len(actual) == 2
  for anchor, prediction, fallback in actual:
    if invalid_service is None:
      assert anchor is not None
      assert not np.allclose(prediction, fallback)
    else:
      assert anchor is None
      np.testing.assert_array_equal(prediction, fallback)


def test_a_disengage_clears_missing_lead_urgency_before_reengagement():
  planner = LongitudinalPlanner(car_params(), init_v=1.0)
  sm = PlannerMessages(v_ego=1.0, e2e_accel=-3.0)
  sm['modelV2'].action.shouldStop = True
  # Supply a real stop horizon: an empty model path is not sufficient stop intent after a reset.
  times = np.asarray(ModelConstants.T_IDXS)
  moving = np.minimum(times, 2.0)
  sm['modelV2'].position.x = (moving - 0.25 * moving ** 2).tolist()
  sm['modelV2'].velocity.x = np.maximum(1.0 - 0.5 * times, 0.0).tolist()
  set_lead(sm, 'leadTwo', 1.0)
  planner.update(sm)
  assert planner.stop_landing.required_decel > 3.0
  sm['radarState'].leadTwo.present = False
  sm['controlsState'].longControlState = LongCtrlState.off
  sm['selfdriveState'].enabled = False
  planner.update(sm)
  assert planner.stop_landing.required_decel == 0.0
  sm['controlsState'].longControlState = LongCtrlState.pid
  sm['selfdriveState'].enabled = True
  planner.update(sm)
  assert planner.stop_landing.required_decel == 0.0
  assert planner.output_a_target > -1.0


def test_nonfinite_targets_do_not_extend_the_missing_lead_guard():
  law = LongitudinalPlanner(car_params()).stop_landing
  law.update(-3.0, 1.0, LeadObservation(True, 1.0), True)
  assert law.required_decel > 3.0
  for _ in range(LEAD_ABSENCE_FRAMES):
    assert math.isnan(law.update(math.nan, 1.0, LeadObservation(), True))
  assert law.required_decel == 0.0
  assert law.update(-3.0, 1.0, LeadObservation(), True) > -1.0


def test_remembered_urgency_does_not_itself_command_braking_or_prevent_a_launch():
  law = LongitudinalPlanner(car_params()).stop_landing
  law.update(-3.0, 1.0, LeadObservation(True, 1.0), True)
  assert law.update(0.5, 1.0, LeadObservation(), True, launch=True) == 0.5
  assert not law.landing


def test_a_fresh_far_lead_replaces_old_urgency_without_disabling_landing():
  law = LongitudinalPlanner(car_params()).stop_landing
  law.update(-3.0, 1.0, LeadObservation(), True, lead2=LeadObservation(True, 1.0))
  assert law.required_decel > 3.0
  assert -1.0 < law.update(-3.0, 1.0, LeadObservation(), True, lead2=LeadObservation(True, 50.0)) < 0.0
  assert law.landing


@pytest.mark.parametrize('slot', ['leadOne', 'leadTwo'])
def test_urgency_expiry_releases_braking_without_a_step(slot):
  planner = LongitudinalPlanner(car_params(), init_v=1.0)
  sm = PlannerMessages(v_ego=1.0, e2e_accel=-3.0)
  sm['modelV2'].action.shouldStop = True
  set_lead(sm, slot, 1.0)
  planner.update(sm)
  assert planner.output_a_target <= -2.9
  getattr(sm['radarState'], slot).present = False
  allowed = float(np.interp(1.0, *planner.stop_landing.release_jerk)) * planner.dt
  for _ in range(LEAD_ABSENCE_FRAMES + 35):
    previous = planner.output_a_target
    planner.update(sm)
    assert planner.output_a_target - previous <= allowed + 1e-9
  assert planner.stop_landing.required_decel == 0.0
  assert not planner.stop_landing.releasing
  # With fixed input speed/zero aEgo, the existing anti-stall press legitimately remains in the ordinary bound.
  assert math.isclose(planner.output_a_target, -planner.stop_landing.bound(1.0) - planner.stop_landing.integral)
  # Fresh urgency is still allowed to brake immediately after the release.
  set_lead(sm, slot, 1.0)
  planner.update(sm)
  assert planner.output_a_target <= -2.9


@pytest.mark.parametrize('event', ['fresh_urgency', 'disengage'])
def test_an_in_progress_urgency_release_yields_to_new_authority(event):
  planner = LongitudinalPlanner(car_params(), init_v=1.0)
  sm = PlannerMessages(v_ego=1.0, e2e_accel=-3.0)
  sm['modelV2'].action.shouldStop = True
  set_lead(sm, 'leadTwo', 1.0)
  planner.update(sm)
  sm['radarState'].leadTwo.present = False
  for _ in range(LEAD_ABSENCE_FRAMES):
    planner.update(sm)
  assert planner.stop_landing.releasing
  assert planner.output_a_target > -3.0
  if event == 'fresh_urgency':
    set_lead(sm, 'leadTwo', 1.0)
  else:
    sm['controlsState'].longControlState = LongCtrlState.off
    sm['selfdriveState'].enabled = False
  planner.update(sm)
  assert not planner.stop_landing.releasing
  if event == 'fresh_urgency':
    assert planner.output_a_target <= -2.9
  else:
    assert planner.stop_landing.required_decel == 0.0




def landing(lost, experimental=False):
  # an ideal car landing from 3.3 m/s on a car standing 7.6 m ahead, which the radar does not report on the lost frames; in Experimental
  # the model calls the stop at the car asking -2.5. Per frame: the published target, its rise, the speed, whether the MPC keeps the
  # car, the landing's absence count for it, whether the landing is latched, its release ramp, and whether it holds the car below its
  # corridor
  car, frames = IdealCar(3.3), []
  for k in range(round(6.0 / DT_MDL)):
    told, v_ego, d = car.a, car.v, 7.6 - car.x
    model = {'experimental': True, 'should_stop': True, 'path': (max(d, 0.0), 0.0), 'e2e_accel': -2.5} if experimental else {}
    car.step(lead=None if k in lost else (d, 0.0, 0.0), v_cruise=10.0, **model)
    law = car.planner.stop_landing
    frames.append((car.a, car.a - told, v_ego, car.planner.mpc.kept[0], law._lead_missing[0], law.landing, law.releasing,
                   car.a < -(law.bound(v_ego) + law.integral) - 1e-6))
  return frames


def loss_start(frames, v_loss):
  return next(k for k, f in enumerate(frames) if f[5] and f[2] <= v_loss)


def ramps(frames, start):
  # the frames, counted from the loss, on which a release ramp starts while the landing still holds: a lead's room going back. The
  # landing's own exit ramps later
  return [k - start for k, (prev, f) in enumerate(zip(frames, frames[1:], strict=False), start=1) if f[6] and not prev[6] and f[5]]


@pytest.mark.parametrize('v_loss', [3.0, 2.5], ids=['at 3 m/s', 'held beyond the corridor at 2.5 m/s'])
def test_a_four_frame_dropout_in_a_landing_never_steps_the_target(v_loss):
  # the radar loses the car for four frames while the landing brakes the car toward it. The MPC keeps it and the landing reads the lead
  # it keeps; the landing's absence window counts from the radar's loss but cannot end while the keep runs, so nothing hands back: the
  # target lands as without the dropout
  clean = landing(())
  start = loss_start(clean, v_loss)
  assert v_loss - 0.1 <= clean[start][2] and all(f[5] for f in clean[start:start + 4])
  assert clean[start][7] == (v_loss < 3.0), clean[start]
  lost = range(start, start + 4)
  dropout = landing(lost)
  assert all(f[3] and f[4] == k + 1 and f[5] and not f[6] for k, f in enumerate(dropout[lost.start:lost.stop])), dropout[lost.start:lost.stop]
  assert all(abs(d[0] - c[0]) <= 0.02 and d[1] <= c[1] + 0.005 for d, c in zip(dropout, clean, strict=True)), \
    max((d[1] - c[1], k) for k, (d, c) in enumerate(zip(dropout, clean, strict=True)))


def lost_for_good(experimental):
  # the same landing, the car gone for good at 2.5 m/s with its room held below the corridor: the MPC keeps it for the rest of the
  # absence hold, which the landing's window runs alongside, and the room goes back through one release ramp from the frame that keep
  # ends, no frame rising faster than the release jerk
  frames = landing(range(loss_start(landing((), experimental), 2.5), round(6.0 / DT_MDL)), experimental)
  start = loss_start(frames, 2.5)
  kept = [k - start for k, f in enumerate(frames) if f[3]]
  assert kept == list(range(LEAD_ABSENCE_FRAMES - 1)), kept
  assert frames[start + LEAD_ABSENCE_FRAMES - 2][7], frames[start + LEAD_ABSENCE_FRAMES - 2]
  assert ramps(frames, start) == [LEAD_ABSENCE_FRAMES - 1], ramps(frames, start)
  assert all(f[1] <= j_cruise_step(f[2]) + 0.005 for f in frames[start:]), max((f[1] - j_cruise_step(f[2]), k - start) for k, f in enumerate(frames))


def test_a_car_lost_for_good_in_a_landing_hands_its_room_back_once_at_the_release_jerk():
  lost_for_good(experimental=False)


def test_a_car_lost_for_good_in_an_experimental_landing_hands_its_room_back_from_the_frame_the_keep_ends():
  # the same with the model asking -2.5 at the car, which holds the plan below the corridor through the loss
  lost_for_good(experimental=True)


def test_a_car_back_nine_frames_after_it_was_lost_in_an_experimental_landing_does_not_step_the_target():
  # Experimental, the model asking -2.5 at the car: the radar loses it at 2.5 m/s and finds it again three frames after the keep ended.
  # The room the landing held went back from the keep's end, so the car's return finds the ramp running and the target never rises
  # faster than the release jerk
  start = loss_start(landing((), True), 2.5)
  frames = landing(range(start, start + 9), True)
  assert ramps(frames, start) == [LEAD_ABSENCE_FRAMES - 1], ramps(frames, start)
  assert all(f[1] <= j_cruise_step(f[2]) + 0.005 for f in frames[start:]), max((f[1] - j_cruise_step(f[2]), k - start) for k, f in enumerate(frames))
