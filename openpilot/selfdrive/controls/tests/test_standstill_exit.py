import math

import numpy as np
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_lead import LeadObservation
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.constants import STOP_DISTANCE
from openpilot.selfdrive.controls.lib.longitudinal_planner import J_CRUISE_BP, J_CRUISE_VALS
from openpilot.selfdrive.controls.lib.necessity_supervisor import LEAD_DEPARTURE_CANCEL, LEAD_SAME_CAR_JUMP, LEAD_STANDING_NOISE
from openpilot.selfdrive.controls.lib.standstill_exit import LAUNCH_ACCEL, LAUNCH_EXIT_ACCEL, StandstillExit
from openpilot.selfdrive.controls.lib.stop_landing import KISS_DECEL, KISS_SPEED, STANDSTILL_SPEED
from openpilot.selfdrive.controls.tests.test_longitudinal_planner import j_cruise_step

LEAD_GAP = 5.5  # m, where the lead the car rests behind stands
KISS = -KISS_DECEL
NONE = LeadObservation()
LEAD = object()  # the MPC binds the pre-release's lead


def lead_at(distance, track_id=7):
  return LeadObservation(True, distance, 0.0, 0.0, 1.0, track_id)


def step(ex, standstill=True, v_ego=0.0, held=False, owner_hold=False, stop_request=False, level=False, edge=False, plan=KISS,
         lead=None, active=True, obstacle=LEAD, slot=0):
  # one frame; an owner's hold is a hold. level: the pre-release's release this frame; edge: one given once; lead: the
  # pre-release's lead as it last read it (default: standing LEAD_GAP ahead); obstacle: the reading in radar slot `slot` the MPC
  # binds while from rest it leaves no room (default: the pre-release's lead, a standing car within STOP_DISTANCE), None for none
  lead = lead_at(LEAD_GAP) if lead is None else lead
  if obstacle is LEAD:
    obstacle = lead if lead.present and lead.distance <= STOP_DISTANCE else None
  ex.update(active, standstill, v_ego, held or owner_hold, owner_hold, stop_request, level, edge, plan, lead,
            None if obstacle is None else (slot, obstacle))
  return ex.holding, ex.apply(plan, v_ego)


def held_at_rest():
  ex = StandstillExit(DT_MDL, (J_CRUISE_BP, J_CRUISE_VALS))
  for _ in range(10):
    step(ex, owner_hold=True)
  assert ex.at_rest and ex.holding
  return ex


def roll(ex, distance, plan=0.0, speed=0.25, lead=None):
  # the wheels turn and the car rolls distance at speed, the lead reading where it stands
  frames = round(distance / (speed * DT_MDL))
  for k in range(frames):
    step(ex, standstill=False, v_ego=speed, plan=plan, lead=lead_at(LEAD_GAP - (k + 1) * speed * DT_MDL) if lead is None else lead)
  return frames * speed * DT_MDL


class TestHold:
  def test_an_owner_holds_and_a_plan_under_the_exit_level_is_still_a_hold(self):
    ex = held_at_rest()
    bit, out = step(ex, owner_hold=True, plan=0.3)
    assert bit and out == 0.3
    # nothing holds and nothing releases: the MPC's hover above its stop threshold leaves the car held, not in limbo
    for plan in (0.12, 0.30, 0.49):
      bit, out = step(ex, plan=plan)
      assert bit and not ex.launching and out == plan, plan

  @pytest.mark.parametrize('v_ego', [KISS_SPEED, 1.0])
  def test_a_stale_standstill_flag_at_speed_is_not_rest(self, v_ego):
    ex = held_at_rest()
    bit, out = step(ex, standstill=True, v_ego=v_ego, edge=True, level=True, plan=0.2)
    assert not ex.at_rest and not ex.committed and not ex.launching and out == 0.2


class TestCommit:
  @pytest.mark.parametrize('release', ['edge', 'level'])
  @pytest.mark.parametrize('plan', [KISS, 0.0, 0.3, 1.4])
  def test_a_release_commits_on_its_frame_at_the_launch_level(self, release, plan):
    ex = held_at_rest()
    bit, out = step(ex, plan=plan, **{release: True})
    assert ex.committed and ex.launching and not bit
    assert out == max(plan, LAUNCH_ACCEL)

  @pytest.mark.parametrize('plan', [LAUNCH_EXIT_ACCEL, 0.7])
  def test_the_plan_asking_the_exit_level_commits_without_a_release(self, plan):
    ex = held_at_rest()
    bit, out = step(ex, plan=plan)
    assert ex.committed and not bit and out == LAUNCH_ACCEL

  def test_a_commit_waits_for_the_last_hold(self):
    # a Force Stops green while the model's stop bit lags two frames: the green is kept and commits on the frame nothing holds
    ex = held_at_rest()
    for edge in (True, False):
      bit, out = step(ex, edge=edge, held=True, stop_request=True, plan=0.0)
      assert bit and not ex.committed and out == 0.0
    bit, out = step(ex, plan=0.0)
    assert ex.committed and not bit and out == LAUNCH_ACCEL

  def test_a_release_given_while_an_owner_holds_is_not_one(self):
    # Force Stops handing its stop to a lead the pre-release holds for: when that hold goes, nothing has released the car
    ex = held_at_rest()
    step(ex, edge=True, owner_hold=True)
    for _ in range(5):
      step(ex, owner_hold=True)
    for _ in range(10):
      bit, out = step(ex, plan=0.2)
      assert bit and not ex.committed and out == 0.2

  def test_a_release_under_another_candidates_braking_does_not_commit(self):
    # at rest the lift does not override a plan braking past the exit level, whichever candidate asks it
    ex = held_at_rest()
    bits = []
    for _ in range(20):
      bit, out = step(ex, level=True, plan=-LAUNCH_EXIT_ACCEL - 0.1)
      bits.append(bit)
      assert not ex.committed and out == -LAUNCH_EXIT_ACCEL - 0.1
    assert all(bits)


class TestTakeBack:
  @pytest.mark.parametrize('cause', ['owner_hold', 'stop_request', 'braking', 'lapsed'])
  def test_a_launch_is_taken_back_before_the_wheels_turn(self, cause):
    ex = held_at_rest()
    step(ex, level=True, plan=0.2)
    assert ex.launching
    frame, plan = {'owner_hold': ({'level': True, 'owner_hold': True}, 0.2), 'stop_request': ({'level': True, 'stop_request': True, 'held': True}, 0.0),
                   'braking': ({'level': True, 'held': True}, -LAUNCH_EXIT_ACCEL - 0.01), 'lapsed': ({'held': True}, 0.05)}[cause]
    bit, out = step(ex, plan=plan, **frame)
    assert ex.taken_back and bit and not ex.launching and out == plan

  @pytest.mark.parametrize('moved', [0.0, LEAD_STANDING_NOISE + 0.05], ids=['where_it_was', 'a_little_on'])
  @pytest.mark.parametrize('plan', [0.3, 0.0])
  def test_a_lapsed_release_keeps_the_launch_while_no_candidate_asks_for_the_stop(self, plan, moved):
    # the pre-release gives no release, without its own hold: the launch stands while no candidate's stop bit is up, wherever its
    # lead stands and whatever the arbitrated plan (the model asking 0.0 at rest without its stop bit is not a stop)
    ex = held_at_rest()
    step(ex, level=True, plan=0.2)
    lead = lead_at(LEAD_GAP + moved)
    for _ in range(round(2 * LEAD_DEPARTURE_CANCEL / DT_MDL)):
      bit, out = step(ex, plan=plan, lead=lead)
      assert not ex.taken_back and not bit and out == LAUNCH_ACCEL
    bit, out = step(ex, held=True, plan=plan, lead=lead)
    assert ex.taken_back and bit and out == plan

  @pytest.mark.parametrize('dropout', [False, True])
  def test_the_exit_times_no_cancel_of_its_own(self, dropout):
    # the pre-release gives no release while its lead stands where it was released, with or without a frame its lead is missing:
    # the exit counts nothing, and its cancel arrives as the pre-release's hold, which takes the launch back on its frame
    ex = held_at_rest()
    step(ex, level=True, plan=0.3)
    if dropout:
      step(ex, plan=0.3, obstacle=None)
    for _ in range(round(2 * LEAD_DEPARTURE_CANCEL / DT_MDL)):
      bit, out = step(ex, plan=0.3)
      assert not ex.taken_back and not bit and out == LAUNCH_ACCEL
    bit, out = step(ex, owner_hold=True, plan=0.3)
    assert ex.taken_back and bit and out == 0.3

  def test_a_frame_the_pre_release_reports_no_release_is_one(self):
    # the pre-release reports its release on frames its lead is missing as on the others, and the launch stands through radar
    # dropouts on alternate frames whatever stop bit a candidate raises there; a frame it reports none with a stop bit up takes
    # the launch back, its lead missing or not: the exit keeps no release of its own
    for missing in (False, True):
      ex = held_at_rest()
      step(ex, level=True, plan=0.2)
      for k in range(10):
        dropout = k % 2 == 0
        bit, out = step(ex, level=True, held=dropout, plan=0.0, obstacle=None if dropout else LEAD)
        assert not ex.taken_back and not bit and out == LAUNCH_ACCEL, k
      bit, out = step(ex, held=True, plan=0.0, obstacle=None if missing else LEAD)
      assert ex.taken_back and bit and out == 0.0, missing

  def test_a_stop_request_flicker_keeps_the_release_given_once(self):
    ex = held_at_rest()
    step(ex, edge=True, plan=0.0)
    bit, _ = step(ex, stop_request=True, held=True, plan=0.0)
    assert ex.taken_back and bit
    bit, out = step(ex, plan=0.0)
    assert ex.committed and not bit and out == LAUNCH_ACCEL

  def test_the_mpc_replanning_under_a_standing_release_is_not_a_take_back(self):
    # the MPC's own stop bit up for the lead the launch was given for, which leaves no room
    ex = held_at_rest()
    step(ex, edge=True, plan=0.0)
    for plan in (0.05, -0.3, -LAUNCH_EXIT_ACCEL, 0.0):
      bit, out = step(ex, held=True, plan=plan)
      assert not ex.taken_back and not bit and out == LAUNCH_ACCEL, plan

  @pytest.mark.parametrize('given, reading', [(NONE, lead_at(1.0, 11)), (lead_at(LEAD_GAP), lead_at(LEAD_GAP - LEAD_SAME_CAR_JUMP - 0.1)),
                                              (lead_at(3.0), lead_at(3.0 + LEAD_SAME_CAR_JUMP + 0.1, 9))],
                           ids=['where_no_lead_was', 'nearer_on_its_track', 'another_car_beyond'])
  def test_a_lead_with_no_room_the_launch_was_not_given_for_takes_it_back(self, given, reading):
    # a car appearing at rest after a Force Stops green on open road, the lead's track reading a car's jump nearer, or another car
    # taking the slot beyond it within the stop distance: the MPC binding it with no room takes the launch back on its frame,
    # before the MPC's own plan falls to its stop bit
    ex = held_at_rest()
    step(ex, edge=True, plan=0.3, lead=given)
    assert ex.launching
    bit, out = step(ex, plan=0.3, lead=reading, obstacle=reading)
    assert ex.taken_back and bit and out == 0.3

  @pytest.mark.parametrize('release', ['edge', 'level'])
  @pytest.mark.parametrize('reading', [lead_at(3.0), lead_at(3.0 - LEAD_SAME_CAR_JUMP + 0.1, 9), lead_at(3.0 + 3.0)],
                           ids=['where_it_stood', 'another_track_on_it', 'driving_off_on_its_track'])
  def test_the_lead_the_launch_was_given_for_is_not_new(self, reading, release):
    # the same car with no room: its radar track, or a reading within a car's jump of it (radard re-associates a standing car)
    ex = held_at_rest()
    step(ex, plan=0.3, lead=lead_at(3.0), **{release: True})
    bit, out = step(ex, held=True, plan=0.0, lead=reading, level=release == 'level')
    assert not ex.taken_back and not bit and out == LAUNCH_ACCEL

  @pytest.mark.parametrize('release', ['level', 'edge'])
  @pytest.mark.parametrize('reading', [lead_at(5.0 - LEAD_SAME_CAR_JUMP - 0.1), lead_at(3.0 + LEAD_SAME_CAR_JUMP + 0.1, 9)],
                           ids=['its_own_track_a_jump_nearer', 'a_new_track_a_jump_beyond'])
  def test_the_same_car_is_judged_against_its_record_under_any_release(self, reading, release):
    # what a launch is given for is the record taken at its commit, whichever release it is: its own track read more than a car's
    # jump nearer, or a reading on another track more than a jump beyond, with no room, takes the launch back
    ex = held_at_rest()
    at = lead_at(5.0 if reading.track_id == 7 else 3.0)
    step(ex, plan=0.3, lead=at, **{release: True})
    bit, out = step(ex, plan=0.3, lead=reading, level=release == 'level')
    assert ex.taken_back and bit and out == 0.3

  def test_a_lead_in_contact_takes_the_launch_back_whichever_it_is(self):
    # the lead the release is for reads at the bumper (a reading at or behind it is 0.0): its record is no cover, the launch is
    # taken back on that frame and its release is spent
    ex = held_at_rest()
    step(ex, level=True, plan=0.3, lead=lead_at(0.1))
    bit, out = step(ex, level=True, plan=0.3, lead=lead_at(0.0))
    assert ex.taken_back and bit and out == 0.3
    bit, out = step(ex, level=True, plan=LAUNCH_EXIT_ACCEL + 0.1, lead=lead_at(0.3))
    assert bit and not ex.committed

  def test_a_lead_in_contact_while_the_car_is_held_spends_nothing(self):
    # held behind a lead read at the bumper, which its owner then releases: a lead in contact is no obstacle of its own while the
    # car is held, and the release commits
    ex = held_at_rest()
    for _ in range(5):
      step(ex, owner_hold=True, lead=lead_at(0.0))
    bit, out = step(ex, level=True, plan=0.3, lead=lead_at(0.0))
    assert ex.committed and not bit and out == LAUNCH_ACCEL

  @pytest.mark.parametrize('room', [False, True], ids=['no_room', 'room'])
  def test_a_car_appearing_is_new_by_the_mpcs_room_not_by_its_stop_bit(self, room):
    # after a Force Stops green on open road a car appears standing ahead. Leaving no room it is new on the frame the MPC binds it,
    # while the MPC's plan, falling from its seed, is still above its stop threshold; leaving room, the MPC's stop bit up for a frame
    # on its seed is not something new
    ex = held_at_rest()
    step(ex, edge=True, plan=0.3, lead=NONE)
    car = lead_at(STOP_DISTANCE + (1.0 if room else -1.0), 11)
    bit, out = step(ex, plan=0.3, held=room, lead=car, obstacle=None if room else car)
    assert ex.taken_back != room and bit != room
    bit, out = step(ex, plan=0.3, lead=car, obstacle=None if room else car)
    assert ex.holding != room and not ex.committed and out == (LAUNCH_ACCEL if room else 0.3)

  def car_appeared_after_a_green(self):
    # a Force Stops green on open road commits, then a car appears standing 1 m ahead and the MPC binds it with no room: the
    # launch is taken back. The pre-release reads it as its lead from then on, as it last read it while the radar misses it
    ex = held_at_rest()
    step(ex, edge=True, plan=0.3, lead=NONE)
    car = lead_at(1.0, 11)
    step(ex, plan=0.3, lead=car)
    assert ex.taken_back
    return ex, car

  def test_a_car_appearing_after_a_launch_holds_it_while_the_radar_misses_it(self):
    # the MPC's hover behind it dropping its stop bit, a frame the radar misses it with the free road's plan asking to go, or the
    # plan asking to go on a frame it is back where it stood, does not commit on the green again: a car missing has not gone
    ex, car = self.car_appeared_after_a_green()
    for k in range(15):
      seen, stop = k % 3 != 2, k % 3 == 0
      plan = 0.15 if seen else LAUNCH_EXIT_ACCEL + 0.3
      bit, out = step(ex, plan=plan, held=stop, lead=car, obstacle=car if seen else None)
      assert bit and not ex.committed and out == plan, k
    for _ in range(round(2.0 / DT_MDL)):
      bit, _ = step(ex, plan=LAUNCH_EXIT_ACCEL + 0.3, lead=car, obstacle=None)
      assert bit and not ex.committed
    back = lead_at(car.distance + 0.1, 11)
    bit, out = step(ex, plan=LAUNCH_EXIT_ACCEL + 0.3, lead=back)
    assert bit and not ex.committed and out == LAUNCH_EXIT_ACCEL + 0.3

  @pytest.mark.parametrize('nearer', [False, True])
  def test_a_car_beyond_the_car_that_appeared_does_not_end_the_hold(self, nearer):
    # a car nearer than the one that appeared, or another car beyond it in its radar slot, within the MPC's stop distance or
    # beyond it, ends nothing: the green spent, the car holds until a release is given anew, which then commits
    ex, car = self.car_appeared_after_a_green()
    if nearer:
      near = lead_at(0.5, 13)
      bit, _ = step(ex, plan=0.15, lead=near)
      assert bit and not ex.committed
    for beyond in (lead_at(car.distance + 5.0, 12), lead_at(STOP_DISTANCE + 0.5, 12)):
      for _ in range(10):
        bit, out = step(ex, plan=LAUNCH_EXIT_ACCEL + 0.3, lead=beyond)
        assert bit and not ex.committed
    bit, out = step(ex, edge=True, plan=0.15, lead=lead_at(STOP_DISTANCE + 0.5, 12))
    assert ex.committed and not bit and out == LAUNCH_ACCEL

  def test_a_car_beyond_it_in_the_other_radar_slot_does_not_end_the_hold(self):
    # the radar misses the car that appeared in leadOne while it reports a car farther on in leadTwo, which the MPC then binds:
    # the car in front is missing, not gone
    ex, car = self.car_appeared_after_a_green()
    for _ in range(10):
      bit, _ = step(ex, plan=0.3, lead=car, obstacle=lead_at(12.0, 30), slot=1)
      assert bit and not ex.committed

  def test_the_car_that_appeared_driving_off_commits_on_the_pre_releases_release(self):
    # the pre-release's release, about its own lead, stands. A car takes leadOne in front of it, which the pre-release adopts as
    # its lead with no release, and the launch is taken back; the car holds, whatever the plan asks and however far the car reads
    # moving off, until the pre-release releases it: that commits behind it, and the MPC binding it again is no take-back
    ex = held_at_rest()
    step(ex, level=True, plan=0.3)
    car = lead_at(LEAD_GAP - LEAD_SAME_CAR_JUMP - 1.0, 11)
    step(ex, plan=-0.2, lead=car)
    assert ex.taken_back
    for k in range(10):
      bit, out = step(ex, plan=0.15, held=k % 2 == 0, lead=car)
      assert bit and not ex.committed and out == 0.15, k
    off = lead_at(car.distance + LEAD_STANDING_NOISE + 0.05, 11)
    bit, out = step(ex, plan=LAUNCH_EXIT_ACCEL, lead=off)
    assert bit and not ex.committed and out == LAUNCH_EXIT_ACCEL
    bit, out = step(ex, level=True, plan=0.05, lead=off)
    assert ex.committed and not bit and out == LAUNCH_ACCEL
    bit, out = step(ex, level=True, held=True, plan=0.0, lead=off)
    assert not ex.taken_back and not bit and out == LAUNCH_ACCEL
    # held again by its owner and released: the owner's release is given afresh
    step(ex, owner_hold=True, lead=off)
    assert ex.taken_back
    bit, out = step(ex, level=True, plan=0.15, lead=off)
    assert ex.committed and not bit and out == LAUNCH_ACCEL

  def test_a_launch_given_again_keeps_what_its_release_was_given_against(self):
    # a Force Stops green on open road commits; a stop asked for takes it back, and the launch returns on the green while the MPC
    # binds a car pulling in ahead, still moving, that leaves it room. The car stays new: stopping within the MPC's stop distance
    # it takes the launch back
    ex = held_at_rest()
    step(ex, edge=True, plan=0.3, lead=NONE)
    step(ex, held=True, stop_request=True, plan=0.0, lead=NONE)
    assert ex.taken_back
    step(ex, plan=0.3, lead=LeadObservation(True, STOP_DISTANCE - 0.5, 2.0, -2.0, 1.0, 11), obstacle=None)
    assert ex.committed
    stopped = lead_at(STOP_DISTANCE - 0.4, 11)
    bit, out = step(ex, plan=0.05, lead=stopped)
    assert ex.taken_back and bit and out == 0.05

  def test_a_launch_on_the_plan_alone_is_given_for_the_lead_read_at_its_commit(self):
    # the plan asks the exit level by itself behind a lead standing beyond the stop distance and commits; a car cutting in 3 m
    # ahead, which the pre-release now reads, leaves no room: it is not the lead the launch was given for, and takes it back
    ex = held_at_rest()
    step(ex, plan=LAUNCH_EXIT_ACCEL, lead=lead_at(STOP_DISTANCE + 2.0))
    assert ex.committed
    bit, out = step(ex, plan=LAUNCH_EXIT_ACCEL, lead=lead_at(3.0, 9))
    assert ex.taken_back and bit and out == LAUNCH_EXIT_ACCEL

  @pytest.mark.parametrize('reading, new', [(lead_at(LEAD_GAP - 1.0, 30), True), (lead_at(LEAD_GAP, 7), False)],
                           ids=['another_car', 'its_own_track'])
  def test_a_car_in_the_other_radar_slot_is_not_the_lead_bound(self, reading, new):
    # the pre-release's release for the lead in leadOne commits; the MPC then binds a car in leadTwo standing a metre nearer:
    # in another slot only the same radar track is the same car
    ex = held_at_rest()
    step(ex, level=True, plan=0.3)
    bit, out = step(ex, level=True, plan=0.0, obstacle=reading, slot=1)
    assert ex.taken_back == new and bit == new and out == (0.0 if new else LAUNCH_ACCEL)

  def test_the_pre_releases_own_lead_is_not_new_while_its_release_stands(self):
    # a launch commits on the pre-release's release while the MPC binds a car standing in leadTwo; the MPC then binding the
    # pre-release's lead with no room is the release's own business, not something new
    ex = held_at_rest()
    step(ex, level=True, plan=0.3, obstacle=None)
    assert ex.committed
    for _ in range(5):
      bit, out = step(ex, level=True, held=True, plan=0.05)
      assert not ex.taken_back and not bit and out == LAUNCH_ACCEL

  def test_a_commit_on_a_frame_without_the_lead_binds_the_lead_the_release_is_for(self):
    # the pre-release's release stands through a frame its lead is missing and the launch commits on that frame, the MPC binding
    # no lead; the pre-release reports its lead as it last read it. The release then lapses with the MPC stopping for that same
    # lead and is taken back; given again, it commits: the lead it is for is not new
    ex = held_at_rest()
    step(ex, level=True, held=True, stop_request=True, plan=0.0)
    step(ex, level=True, plan=0.3, obstacle=None)
    assert ex.committed
    step(ex, held=True, plan=0.05)
    assert ex.taken_back
    bit, out = step(ex, level=True, plan=0.05)
    assert ex.committed and not bit and out == LAUNCH_ACCEL

  def test_a_release_given_anew_is_given_for_the_lead_the_pre_release_reads_then(self):
    # a launch on the pre-release's release behind a lead 3 m ahead is taken back by its hold, which it hands back when a car
    # standing 6 m ahead takes the slot; that hand-back commits, given for the car the pre-release now reads, whose lack of room is
    # no take-back
    ex = held_at_rest()
    step(ex, level=True, plan=0.3, lead=lead_at(3.0))
    step(ex, owner_hold=True, lead=lead_at(3.0))
    assert ex.taken_back
    beyond = lead_at(6.0, 9)
    step(ex, edge=True, plan=0.0, lead=beyond)
    assert ex.committed
    bit, out = step(ex, plan=0.0, lead=beyond)
    assert not ex.taken_back and not bit and out == LAUNCH_ACCEL

  def test_a_release_given_with_its_lead_back_in_sight_is_given_for_that_lead(self):
    # a hold handed back with the lead gone from the radar commits with no lead given; the same car comes back within the stop
    # distance pulling away and the pre-release releases it on that frame: the release is given for it, and the launch stands
    ex = held_at_rest()
    step(ex, edge=True, plan=0.3, lead=NONE)
    back = LeadObservation(True, LEAD_GAP, 0.6, 0.5, 1.0, 7)
    for _ in range(3):
      bit, out = step(ex, level=True, plan=0.3, lead=back)
      assert not ex.taken_back and not bit and out == LAUNCH_ACCEL

  def test_an_owner_holding_the_car_again_drops_what_its_releases_were_given_against(self):
    # a launch on the pre-release's release behind a lead 8.9 m ahead is taken back by its hold; while it holds the lead reads 3.5 m
    # ahead, and its next release commits: the record of the first launch went with its release
    ex = held_at_rest()
    step(ex, level=True, plan=0.3, lead=lead_at(8.9))
    step(ex, owner_hold=True, lead=lead_at(8.9))
    assert ex.taken_back
    for _ in range(5):
      step(ex, owner_hold=True, plan=0.0, lead=lead_at(3.5))
    bit, out = step(ex, level=True, plan=0.0, lead=lead_at(3.5))
    assert ex.committed and not bit and out == LAUNCH_ACCEL


class TestSpent:
  # a take-back for an owner's hold or for something new spends every release standing: the car holds, the plan's own ask
  # included, until a release is given anew
  @pytest.mark.parametrize('cause', ['owner_hold', 'something_new'])
  def test_a_take_back_for_an_obstacle_spends_the_releases_standing(self, cause):
    for anew in ('level', 'edge'):
      ex = held_at_rest()
      step(ex, level=True, edge=True, plan=0.3)
      new = cause == 'something_new'
      bit, _ = step(ex, level=True, plan=0.3, owner_hold=not new, obstacle=lead_at(2.0, 11) if new else None, slot=1)
      assert ex.taken_back and bit
      for _ in range(10):
        bit, out = step(ex, level=True, plan=LAUNCH_EXIT_ACCEL + 0.1, obstacle=None)
        assert bit and not ex.committed and out == LAUNCH_EXIT_ACCEL + 0.1
      if anew == 'level':
        step(ex, plan=0.0, obstacle=None)
      bit, out = step(ex, plan=0.0, obstacle=None, **{anew: True})
      assert ex.committed and not bit and out == LAUNCH_ACCEL, anew

  def test_an_owner_holding_the_car_again_while_it_is_held_spends_the_release_standing(self):
    # a launch on the pre-release's release is taken back by a stop asked for, which spends nothing; while the car is held an
    # owner holds it again and lets go without a release: the release still standing does not commit, nor does the plan, until a
    # release is given anew
    ex = held_at_rest()
    step(ex, level=True, plan=0.3)
    step(ex, level=True, stop_request=True, held=True, plan=0.0)
    assert ex.taken_back
    step(ex, level=True, owner_hold=True, plan=0.0)
    for _ in range(5):
      bit, out = step(ex, level=True, plan=LAUNCH_EXIT_ACCEL + 0.1)
      assert bit and not ex.committed
    step(ex, plan=0.0)
    bit, out = step(ex, level=True, plan=0.0)
    assert ex.committed and not bit

  def test_a_stop_request_spends_nothing(self):
    # a stop asked for takes the launch back without spending the green, which returns once the stop clears
    ex = held_at_rest()
    step(ex, edge=True, plan=0.0)
    step(ex, stop_request=True, held=True, plan=0.0)
    assert ex.taken_back
    bit, out = step(ex, plan=0.0)
    assert ex.committed and not bit

  def test_something_new_while_the_car_is_held_spends_the_release_standing(self):
    # a green commits and a stop asked for takes it back; while the stop holds the car a car appears in leadTwo with no room: when
    # the stop clears and the radar misses that car, the green does not commit again
    ex = held_at_rest()
    step(ex, edge=True, plan=0.0)
    step(ex, stop_request=True, held=True, plan=0.0)
    step(ex, stop_request=True, held=True, plan=0.0, obstacle=lead_at(2.0, 30), slot=1)
    for _ in range(10):
      bit, out = step(ex, plan=LAUNCH_EXIT_ACCEL + 0.1, obstacle=None)
      assert bit and not ex.committed
    bit, out = step(ex, edge=True, plan=0.0, obstacle=None)
    assert ex.committed and not bit

  def test_a_release_given_anew_with_something_new_still_there_is_spent_too(self):
    # the pre-release releases its lead while a car stands in leadTwo with no room: nothing commits
    ex = held_at_rest()
    for _ in range(5):
      bit, _ = step(ex, level=True, plan=0.3, obstacle=lead_at(3.0, 30), slot=1)
      assert bit and not ex.committed


class TestHandBack:
  def test_the_lift_goes_back_at_the_cruise_jerk_once_the_wheels_turn(self):
    ex = held_at_rest()
    step(ex, edge=True, plan=0.0)
    out_prev, v, outs = LAUNCH_ACCEL, 0.15, []
    while len(outs) < 20:
      _, out = step(ex, standstill=False, v_ego=v, plan=0.3)
      outs.append(out)
      assert math.isclose(out, max(out_prev - j_cruise_step(v), 0.3), abs_tol=1e-9), (len(outs), out_prev, out)
      out_prev, v = out, v + out * DT_MDL
    assert outs[-1] == 0.3 and not ex.releasing

  def test_a_plan_above_the_lift_passes(self):
    ex = held_at_rest()
    step(ex, edge=True, plan=0.0)
    for _ in range(3):
      step(ex, standstill=False, v_ego=0.2, plan=0.3)
    assert ex.releasing
    _, out = step(ex, standstill=False, v_ego=0.25, plan=1.5)
    assert out == 1.5 and not ex.releasing

  def test_braking_passes_at_once(self):
    ex = held_at_rest()
    step(ex, edge=True, plan=0.0)
    _, out = step(ex, standstill=False, v_ego=0.15, plan=-0.4)
    assert out == -0.4 and not ex.releasing
    _, out = step(ex, standstill=False, v_ego=0.15, plan=0.2)
    assert out == 0.2

  def test_a_stop_bit_ends_the_hand_back(self):
    # the stop bit up off rest hands the car to LongControl's stop, which resumes from the plan, not from the lift
    ex = held_at_rest()
    step(ex, edge=True, plan=0.0)
    step(ex, standstill=False, v_ego=0.15, plan=0.3)
    assert ex.releasing
    _, out = step(ex, standstill=False, v_ego=0.16, held=True, plan=0.08)
    assert out == 0.08 and not ex.releasing
    _, out = step(ex, standstill=False, v_ego=0.17, plan=0.3)
    assert out == 0.3

  def test_arriving_at_rest_ends_the_hand_back(self):
    ex = held_at_rest()
    step(ex, edge=True, plan=0.0)
    step(ex, standstill=False, v_ego=0.15, plan=0.2)
    assert ex.releasing
    bit, out = step(ex, owner_hold=True, plan=KISS)
    assert bit and out == KISS and not ex.releasing


class TestUnanswered:
  def launch_and_rest(self, travel=0.3, lead_moved=0.0, plan=0.2):
    # a launch on the pre-release's evidence (or on the plan, at plan >= the exit level) behind the lead standing LEAD_GAP ahead;
    # the car rolls travel and rests again, held, with the lead lead_moved on from where it stood
    ex = held_at_rest()
    step(ex, level=True, plan=plan)
    assert ex.committed
    rolled = roll(ex, travel)
    lead = lead_at(LEAD_GAP - rolled + lead_moved)
    step(ex, owner_hold=True, lead=lead)
    return ex, lead

  def test_a_lead_that_did_not_answer_makes_the_next_launch_wait_for_the_plan(self):
    ex, lead = self.launch_and_rest()
    for _ in range(5):
      bit, out = step(ex, level=True, plan=0.2, lead=lead)
      assert bit and not ex.committed and out == 0.2
    bit, out = step(ex, level=True, plan=LAUNCH_EXIT_ACCEL, lead=lead)
    assert ex.committed and not bit and out == LAUNCH_ACCEL
    # the plan's launch answered it: taken back, the next evidence commits again
    step(ex, owner_hold=True, lead=lead)
    step(ex, level=True, plan=0.2, lead=lead)
    assert ex.committed

  def test_a_standstill_flag_flicker_without_motion_judges_nothing(self):
    # the flag drops for a frame with the wheels held: the launch hands back, but nothing travelled, so nothing is judged
    ex = held_at_rest()
    step(ex, level=True, plan=0.2)
    step(ex, standstill=False, level=True, plan=0.2)
    step(ex, owner_hold=True)
    bit, out = step(ex, level=True, plan=0.2)
    assert ex.committed and not bit and out == LAUNCH_ACCEL

  def test_a_launch_taken_back_is_not_judged_at_the_next_rest(self):
    # a release withdrawn after the ESC let go: the car rolls under the raised stop bit and stops again, which is not a launch
    # its lead failed to answer
    ex = held_at_rest()
    step(ex, level=True, plan=0.2)
    step(ex, owner_hold=True)
    assert ex.taken_back
    for k in range(round(0.3 / (0.25 * DT_MDL))):
      step(ex, standstill=False, v_ego=0.25, held=True, plan=KISS, lead=lead_at(LEAD_GAP - (k + 1) * 0.25 * DT_MDL))
    lead = lead_at(LEAD_GAP - 0.3)
    step(ex, owner_hold=True, lead=lead)
    bit, out = step(ex, level=True, plan=0.2, lead=lead)
    assert ex.committed and not bit and out == LAUNCH_ACCEL

  def test_a_lead_that_later_moves_clears_it(self):
    # the mark clears once the lead reads beyond the standing noise of where it stood at that rest, behind the same car included
    ex, lead = self.launch_and_rest()
    bit, _ = step(ex, level=True, plan=0.2, lead=lead)
    assert bit and not ex.committed
    bit, out = step(ex, level=True, plan=0.2, lead=lead_at(lead.distance + LEAD_STANDING_NOISE + 0.05))
    assert ex.committed and not bit and out == LAUNCH_ACCEL

  def test_a_lead_missing_back_at_rest_is_judged_on_the_frame_it_is_read_again(self):
    # the radar misses the lead on the frames the car comes back to rest: nothing is judged until it is read, standing where it
    # stood, and the next launch on evidence then waits for the plan
    ex = held_at_rest()
    step(ex, level=True, plan=0.2)
    rolled = roll(ex, 0.3)
    for _ in range(5):
      step(ex, owner_hold=True, lead=NONE)
    lead = lead_at(LEAD_GAP - rolled)
    step(ex, owner_hold=True, lead=lead)
    bit, out = step(ex, level=True, plan=0.2, lead=lead)
    assert bit and not ex.committed and out == 0.2

  def test_a_judgement_still_pending_when_the_car_leaves_rest_is_dropped(self):
    # back at rest with the lead missing, the car rolls on before the lead is read: that rest's judgement is dropped
    ex = held_at_rest()
    step(ex, level=True, plan=0.2)
    rolled = roll(ex, 0.3)
    step(ex, owner_hold=True, lead=NONE)
    rolled += roll(ex, 0.3, lead=NONE)
    lead = lead_at(LEAD_GAP - rolled)
    step(ex, owner_hold=True, lead=lead)
    bit, out = step(ex, level=True, plan=0.2, lead=lead)
    assert ex.committed and not bit and out == LAUNCH_ACCEL

  @pytest.mark.parametrize('travel, lead_moved, plan', [(0.3, LEAD_STANDING_NOISE + 0.05, 0.2), (LEAD_SAME_CAR_JUMP + 0.5, 0.0, 0.2),
                                                         (0.3, 0.0, LAUNCH_EXIT_ACCEL)],
                           ids=['the lead answered', 'another stop', "the plan's launch"])
  def test_a_launch_that_was_answered_or_asked_for_does_not_mark_it(self, travel, lead_moved, plan):
    ex, lead = self.launch_and_rest(travel, lead_moved, plan)
    bit, out = step(ex, level=True, plan=0.2, lead=lead)
    assert ex.committed and not bit and out == LAUNCH_ACCEL


class TestReset:
  @pytest.mark.parametrize('cause', ['inactive', 'nonfinite_speed'])
  def test_a_reset_drops_the_launch_and_its_memory(self, cause):
    ex, lead = TestUnanswered().launch_and_rest()
    step(ex, edge=True, held=True, stop_request=True, lead=lead)
    step(ex, active=cause != 'inactive', v_ego=math.nan if cause == 'nonfinite_speed' else 0.0, lead=lead)
    assert not (ex.at_rest or ex.holding or ex.launching or ex.releasing) and ex.lift == 0.0
    step(ex, owner_hold=True, lead=lead)
    bit, out = step(ex, level=True, plan=0.2, lead=lead)
    assert ex.committed and not bit and out == LAUNCH_ACCEL

  @pytest.mark.parametrize('launching', [False, True])
  def test_a_nonfinite_plan_passes_through_without_changing_state(self, launching):
    ex = held_at_rest()
    if launching:
      step(ex, level=True, plan=0.2)
    before = dict(vars(ex))
    ex.update(True, True, 0.0, False, False, False, True, True, math.nan, lead_at(LEAD_GAP), (0, lead_at(1.0, 11)))
    assert math.isnan(ex.apply(math.nan, 0.0))
    assert vars(ex) == {**before, 'committed': False, 'taken_back': False}, (before, vars(ex))


@pytest.mark.parametrize('seed', range(10))
def test_at_rest_the_bit_is_a_hold_or_a_launch_and_the_output_never_undercuts_the_plan(seed):
  # random 2,000-frame sequences of rests and rolls, holds, releases, stops asked for, plans, leads and obstacles
  rng = np.random.default_rng(seed)
  ex = StandstillExit(DT_MDL, (J_CRUISE_BP, J_CRUISE_VALS))
  v, distance, launches = 0.0, LEAD_GAP, 0
  for _ in range(2000):
    if rng.random() < 0.03:
      v = 0.0 if v > 0.0 else float(rng.uniform(0.05, 3.0))
    standstill = v < STANDSTILL_SPEED if rng.random() > 0.02 else not v < STANDSTILL_SPEED
    distance = float(np.clip(distance + rng.normal(0.0, 0.1), 0.5, 40.0))
    owner_hold = rng.random() < 0.3
    held = owner_hold or rng.random() < 0.3
    plan = float(rng.uniform(-1.5, 2.0)) if rng.random() < 0.7 else float(rng.choice([KISS, 0.0, 0.05, 0.2, LAUNCH_EXIT_ACCEL]))
    active = rng.random() > 0.005
    obstacle = (int(rng.choice([0, 1])), lead_at(float(rng.uniform(0.5, STOP_DISTANCE)), int(rng.choice([7, 9])))) if rng.random() < 0.3 else None
    ex.update(active, standstill, v, held, owner_hold, rng.random() < 0.1, rng.random() < 0.3, rng.random() < 0.05, plan,
              lead_at(distance) if rng.random() < 0.9 else LeadObservation(), obstacle)
    launching, releasing = ex.launching, ex.releasing
    out = ex.apply(plan, v)
    launches += ex.committed
    if ex.at_rest and not ex.holding:
      assert out >= LAUNCH_ACCEL, (out, plan)
    assert out >= plan
    if not (launching or releasing):
      assert out == plan
  assert launches > 0
