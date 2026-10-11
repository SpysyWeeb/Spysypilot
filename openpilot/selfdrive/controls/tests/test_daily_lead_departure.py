"""Native planner regressions for standstill ownership across both radar lead slots."""
import pytest

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.force_stops import RELEASE_OPEN_FRAMES
from openpilot.selfdrive.controls.lib.necessity_supervisor import LEAD_DEPARTURE_CONFIRM
from openpilot.selfdrive.controls.lib.stop_helpers import PATH_OPEN_LENGTH
from openpilot.selfdrive.controls.tests.test_longitudinal_planner import PALISADE_CP, frame


SOURCES = (log.LongitudinalPlan.LongitudinalPlanSource.lead0, log.LongitudinalPlan.LongitudinalPlanSource.lead1)


def dual_frame(leads=(None, None), *, tracks=(7, 17), forecasts=(None, None), v=0.0, a=0.0, path=None):
  # Reuse the source suite's real cereal messages and accepted model-forecast construction, never stub the MPC.
  sm = frame(v, a, v_cruise=10.0, path=path)
  model_leads = []
  for slot, lead in enumerate(leads):
    sample = frame(v, lead=lead, forecast=forecasts[slot])
    radar_lead = sample['radarState'].leadOne
    radar_lead.radarTrackId = tracks[slot]
    setattr(sm['radarState'], ('leadOne', 'leadTwo')[slot], radar_lead)
    model_leads.append(sample['modelV2'].leadsV3[0] if lead is not None else log.ModelDataV2.LeadDataV3.new_message())
  sm['modelV2'].leadsV3 = model_leads
  return sm


def slots(slot, lead, other=None):
  return (lead, other) if slot == 0 else (other, lead)


def rest_behind(slot, other=None):
  planner = LongitudinalPlanner(PALISADE_CP, init_v=0.3)
  v, a, x = 0.3, 0.0, 0.0
  for _ in range(400):
    planner.update(dual_frame(slots(slot, (7.5 - x, 0.0, 0.0), other), v=v, a=a))
    a = float(planner.output_a_target)
    v = max(v + a * DT_MDL, 0.0)
    x += v * DT_MDL
    if v < 0.01:
      break
  assert v < 0.01, 'native planner did not stop behind the lead'
  d = 7.5 - x
  # The first measured-standstill frame is when the existing pre-release owns the stop bit.
  planner.update(dual_frame(slots(slot, (d, 0.0, 0.0), other)))
  assert planner.mpc.source == SOURCES[slot]
  assert planner.output_should_stop
  return planner, d


@pytest.mark.parametrize('slot', [0, 1], ids=['primary', 'secondary'])
def test_native_lead_stop_arms_hold_and_inching_cannot_release_it(slot):
  planner, d = rest_behind(slot)
  assert planner.lead_departure.holding
  targets = []
  while d < 9.0:
    d += 0.2 * DT_MDL
    planner.update(dual_frame(slots(slot, (d, 0.2, 0.0))))
    targets.append(planner.mpc_a_target)
    assert planner.output_should_stop and planner.lead_departure.holding
  # Exercise the actual unsafe seam: the MPC no longer requests a stop, so only the hold owns this bit.
  assert not should_stop(0.0, max(targets))
  planner.update(dual_frame(slots(slot, (d, 1.0, 1.0))))
  assert not planner.output_should_stop and not planner.lead_departure.holding


@pytest.mark.parametrize('slot', [0, 1], ids=['primary', 'secondary'])
@pytest.mark.parametrize('kind', ['measured', 'model'])
def test_only_sustained_owned_departure_releases(slot, kind):
  planner, d = rest_behind(slot)
  assert planner.lead_departure.holding
  forecasts = slots(slot, 0.5 if kind == 'model' else None)
  speed = 0.0 if kind == 'model' else 0.3
  for _ in range(round(LEAD_DEPARTURE_CONFIRM / DT_MDL) - 1):
    planner.update(dual_frame(slots(slot, (d, speed, 0.0)), forecasts=forecasts))
    assert planner.output_should_stop and planner.lead_departure.holding
  planner.update(dual_frame(slots(slot, (d, speed, 0.0)), forecasts=forecasts))
  assert not planner.output_should_stop and not planner.lead_departure.holding


@pytest.mark.parametrize('slot', [0, 1], ids=['primary', 'secondary'])
@pytest.mark.parametrize('dropout', ['absent', 'radar_stale', 'model_stale', 'forecast_rejected'])
def test_missing_or_unaccepted_evidence_keeps_the_owned_hold(slot, dropout):
  planner, d = rest_behind(slot)
  for _ in range(round(1.0 / DT_MDL)):
    sm = dual_frame(slots(slot, None if dropout == 'absent' else (d, 0.0, 0.0)), forecasts=slots(slot, 0.5))
    if dropout == 'radar_stale':
      sm.alive['radarState'] = False
    elif dropout == 'model_stale':
      sm.alive['modelV2'] = False
    elif dropout == 'forecast_rejected':
      sm['modelV2'].leadsV3[slot].prob = 0.0
    planner.update(sm)
    assert planner.output_should_stop and planner.lead_departure.holding
  # Reappearance on the same track retains the hold, then a real departure still releases it.
  planner.update(dual_frame(slots(slot, (d, 0.0, 0.0))))
  assert planner.output_should_stop and planner.lead_departure.holding
  planner.update(dual_frame(slots(slot, (d, 1.0, 0.0))))
  assert not planner.output_should_stop


@pytest.mark.parametrize('slot', [0, 1], ids=['primary', 'secondary'])
@pytest.mark.parametrize('other', [(3.1, 1.0, 0.0), (25.0, 0.0, 0.0), (25.0, 5.0, 0.0)],
                         ids=['moving_cut_in', 'far_stopped', 'far_moving'])
def test_a_visible_held_lead_keeps_ownership_when_the_other_slot_changes(slot, other):
  planner, d = rest_behind(slot)
  for _ in range(round(1.0 / DT_MDL)):
    planner.update(dual_frame(slots(slot, (d, 0.0, 0.0), other), forecasts=slots(slot, 0.0, 1.0)))
    assert planner.output_should_stop and planner.lead_departure.holding
    assert not planner.lead_departure.handed_back


@pytest.mark.parametrize('slot', [0, 1], ids=['primary', 'secondary'])
@pytest.mark.parametrize('migrated', [False, True], ids=['same_slot', 'migrated'])
def test_owned_departure_cannot_clear_the_mpcs_stop_for_another_lead(slot, migrated):
  planner, d = rest_behind(slot)
  current_slot = 1 - slot if migrated else slot
  tracks = (17, 7) if migrated else (7, 17)
  for _ in range(round(1.0 / DT_MDL)):
    planner.update(dual_frame(slots(current_slot, (d, 1.0, 0.0), (3.1, 0.0, 0.0)), tracks=tracks))
    assert planner.mpc.source == SOURCES[1 - current_slot]
    assert planner.mpc.binding_obstacle == 'lead' and should_stop(0.0, planner.mpc_a_target)
    assert planner.output_should_stop
    # The original car really departed: it must not acquire the other car's stop as its own hold.
    assert not planner.lead_departure.holding
  # With the blocker gone, the original car's continuing departure still releases normally.
  planner.update(dual_frame(slots(current_slot, (d, 1.0, 0.0)), tracks=tracks))
  assert not planner.output_should_stop and not planner.lead_departure.holding


@pytest.mark.parametrize('slot', [0, 1], ids=['primary', 'secondary'])
@pytest.mark.parametrize('other', [(25.0, 0.0, 0.0), (3.1, 1.0, 0.0)], ids=['far_stationary', 'same_track_departing'])
def test_owned_departure_preserves_release_without_a_different_blocker(slot, other):
  planner, d = rest_behind(slot)
  track = (7, 17)[slot]
  tracks = (track, track) if other[1] > 0.0 else (7, 17)
  planner.update(dual_frame(slots(slot, (d, 1.0, 0.0), other), tracks=tracks))
  assert planner.mpc.source == SOURCES[1 - slot if other[1] > 0.0 else slot]
  # Even a duplicate report in the binding slot is the same departing car, not another stop owner.
  assert should_stop(0.0, planner.mpc_a_target)
  assert not planner.output_should_stop and not planner.lead_departure.holding


@pytest.mark.parametrize('slot', [0, 1], ids=['primary_to_secondary', 'secondary_to_primary'])
def test_held_track_migration_beats_the_other_slots_moving_cut_in(slot):
  planner, d = rest_behind(slot)
  tracks = (7, 17)
  for i in range(round(1.0 / DT_MDL)):
    migrated = i % 2 == 0
    current_slot = 1 - slot if migrated else slot
    planner.update(dual_frame(slots(current_slot, (d, 0.0, 0.0), (3.1, 1.0, 0.0)),
                              tracks=tracks[::-1] if migrated else tracks,
                              forecasts=slots(current_slot, 0.0, 1.0)))
    assert planner.output_should_stop and planner.lead_departure.holding
    assert not planner.lead_departure.handed_back
  # Its accepted forecast follows the migrated track, not the original slot.
  for _ in range(round(LEAD_DEPARTURE_CONFIRM / DT_MDL)):
    planner.update(dual_frame(slots(1 - slot, (d, 0.0, 0.0)), tracks=tracks[::-1], forecasts=slots(1 - slot, 0.5)))
  assert not planner.output_should_stop and not planner.lead_departure.holding


@pytest.mark.parametrize('slot', [0, 1], ids=['primary', 'secondary'])
@pytest.mark.parametrize('moves_slot', [False, True], ids=['other_stays', 'other_migrates'])
def test_missing_held_track_does_not_hand_back_to_the_already_visible_other_lead(slot, moves_slot):
  other = (25.0, 0.0, 0.0)
  planner, _ = rest_behind(slot, other)
  tracks = (7, 17)
  remaining_slot = slot if moves_slot else 1 - slot
  for _ in range(round(1.0 / DT_MDL)):
    planner.update(dual_frame(slots(remaining_slot, other), tracks=tracks[::-1] if moves_slot else tracks))
    assert planner.output_should_stop and planner.lead_departure.holding
    assert not planner.lead_departure.handed_back


@pytest.mark.parametrize('slot', [0, 1], ids=['primary', 'secondary'])
@pytest.mark.parametrize('speed', [0.0, 1.0], ids=['model', 'measured'])
def test_duplicate_track_cannot_borrow_the_other_slots_departure(slot, speed):
  planner, d = rest_behind(slot)
  track = (7, 17)[slot]
  for _ in range(round(1.0 / DT_MDL)):
    planner.update(dual_frame(slots(slot, (d, 0.0, 0.0), (d - 1.0, speed, 0.0)),
                              tracks=(track, track), forecasts=slots(slot, 0.0, 1.0)))
    assert planner.output_should_stop and planner.lead_departure.holding


@pytest.mark.parametrize('slot', [0, 1], ids=['primary', 'secondary'])
@pytest.mark.parametrize('replacement', ['other_slot_near', 'same_slot_far', 'same_slot_cut_in'])
def test_a_new_unrelated_moving_track_cannot_release_a_missing_held_track(slot, replacement):
  planner, d = rest_behind(slot)
  new_slot = 1 - slot if replacement == 'other_slot_near' else slot
  distance = {'other_slot_near': d - 0.1, 'same_slot_far': 25.0, 'same_slot_cut_in': 3.1}[replacement]
  for _ in range(round(1.0 / DT_MDL)):
    planner.update(dual_frame(slots(new_slot, (distance, 1.0, 0.0)), tracks=(99, 99), forecasts=slots(new_slot, 1.0)))
    assert planner.output_should_stop and planner.lead_departure.holding
    assert not planner.lead_departure.handed_back


@pytest.mark.parametrize('slot', [0, 1], ids=['primary', 'secondary'])
def test_open_road_confirmation_starts_only_after_the_owned_lead_disappears(slot):
  planner, d = rest_behind(slot)
  open_road = (2.0 * PATH_OPEN_LENGTH, 5.0)
  for _ in range(round(1.0 / DT_MDL)):
    planner.update(dual_frame(slots(slot, (d, 0.0, 0.0)), path=open_road))
    assert planner.output_should_stop and planner.lead_departure.holding
  for _ in range(RELEASE_OPEN_FRAMES - 1):
    planner.update(dual_frame(path=open_road))
    assert planner.output_should_stop and planner.lead_departure.holding
  planner.update(dual_frame(path=open_road))
  assert planner.lead_departure.handed_back and not planner.lead_departure.holding
  assert not planner.output_should_stop
