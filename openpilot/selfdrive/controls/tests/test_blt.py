"""Unit tests for the BLT necessity supervisor (selfdrive/controls/lib/blt.py).

These lock in the behavior that was originally validated by replaying the real class
against recorded routes (see docs/BLoT.md): trigger arm/disarm semantics, debounce
times, hysteresis holds, the whiplash ratchet, the emergency stand-down, and slew
continuity. Scenario numbers are chosen to isolate one mechanism at a time -- each
test states which gates the other mechanisms are held clear of.
"""
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.blt import (BLTSupervisor, DebouncedTrigger, LeadDeparturePreRelease,
                                                  JERK_SCALE_MIN, JERK_SCALE_RATE,
                                                  LEAD_DEPARTURE_CANCEL, LEAD_DEPARTURE_CONFIRM,
                                                  ONSET_PAD_MAX, ONSET_RATE_UP, ONSET_RATE_DOWN)

T_FOLLOW_BASE = 1.45


class Obj:
  def __init__(self, **kw):
    self.__dict__.update(kw)


class FakeSM(dict):
  def __init__(self, *args, radar_valid=True, **kwargs):
    super().__init__(*args, **kwargs)
    self.valid = {'radarState': radar_valid}


def make_sm(status=True, v_ego=15.0, v_lead=15.0, d_rel=30.0, a_lead=0.0,
            lead_prob=1.0, model_v=None, radar_valid=True, standstill=False,
            v_lead_raw=None):
  """model_v defaults to a flat leadsV3 speed prediction (slope 0 -> model-arm quiet)."""
  if model_v is None:
    model_v = [v_lead] * 6
  sm = FakeSM(radar_valid=radar_valid)
  if v_lead_raw is None:
    v_lead_raw = v_lead
  sm['radarState'] = Obj(leadOne=Obj(present=status, vLead=v_lead_raw, vLeadK=v_lead,
                                    dRel=d_rel, aLeadK=a_lead))
  sm['carState'] = Obj(vEgo=v_ego, standstill=standstill)
  sm['modelV2'] = Obj(leadsV3=[Obj(prob=lead_prob, v=model_v)])
  return sm


def run(sup, sm, a_plan, n=1):
  out = None
  for _ in range(n):
    out = sup.update(sm, a_plan, T_FOLLOW_BASE)
  return out


def run_pre_release(release, sm, n=1, active=True):
  out = False
  for _ in range(n):
    out = release.update(sm, active)
  return out


def frames(seconds):
  return round(seconds / DT_MDL)


class TestDebouncedTrigger:
  def test_arms_only_after_debounce(self):
    # float accumulation: 8 x 0.05 sums just under 0.4, so arming lands one frame later
    trig = DebouncedTrigger(0.4)
    for _ in range(frames(0.4)):
      assert not trig.step(arm=True, disarm=False)
    assert trig.step(arm=True, disarm=False)

  def test_holds_in_hysteresis_band(self):
    trig = DebouncedTrigger(0.4)
    for _ in range(frames(0.4) + 1):
      trig.step(arm=True, disarm=False)
    # in the band (neither arm nor disarm) the accrual holds -- still armed
    assert trig.step(arm=False, disarm=False)
    assert trig.step(arm=False, disarm=False)

  def test_disarm_resets(self):
    trig = DebouncedTrigger(0.4)
    for _ in range(frames(0.4)):
      trig.step(arm=True, disarm=False)
    assert not trig.step(arm=False, disarm=True)
    # accrual restarts from zero
    assert not trig.step(arm=True, disarm=False)


class TestInert:
  def test_no_lead(self):
    sup = BLTSupervisor()
    js, tf = run(sup, make_sm(status=False), a_plan=-3.0, n=frames(2.0))
    assert js == 1.0
    assert tf == T_FOLLOW_BASE

  def test_radar_invalid(self):
    sup = BLTSupervisor()
    js, tf = run(sup, make_sm(radar_valid=False), a_plan=-3.0, n=frames(2.0))
    assert js == 1.0
    assert tf == T_FOLLOW_BASE

  def test_crawl_speed(self):
    sup = BLTSupervisor()
    js, tf = run(sup, make_sm(v_ego=0.5, v_lead=0.0, d_rel=8.0), a_plan=-3.0, n=frames(2.0))
    assert js == 1.0
    assert tf == T_FOLLOW_BASE


class TestRecoveryBoost:
  # steady following (a_req ~ 0), plan holding deep braking -> stale deceleration
  def test_arms_on_sustained_excess(self):
    sup = BLTSupervisor()
    sm = make_sm()  # v_ego == v_lead, no closing, a_req = 0
    js, _ = run(sup, sm, a_plan=-2.0, n=frames(0.35))
    assert js == 1.0  # debounce (0.4s) not yet met -- no movement at all
    js, _ = run(sup, sm, a_plan=-2.0, n=frames(1.0))
    assert js == pytest.approx(JERK_SCALE_MIN)

  def test_ignores_small_trims(self):
    sup = BLTSupervisor()
    js, _ = run(sup, make_sm(), a_plan=-0.5, n=frames(2.0))  # below MIN_BOOST_BRAKE
    assert js == 1.0

  def test_disarms_when_plan_converges(self):
    sup = BLTSupervisor()
    sm = make_sm()
    run(sup, sm, a_plan=-2.0, n=frames(1.5))
    js, _ = run(sup, sm, a_plan=-0.1, n=frames(1.0))  # lead not braking: no ratchet hold
    assert js == 1.0


class TestModelEarlyArm:
  # model forecasts a hard lead decel (leadsV3 0->2s slope) before radar measures one
  def test_arms_on_predicted_hard_decel(self):
    sup = BLTSupervisor()
    sm = make_sm(model_v=[15.0, 11.0, 8.0, 6.0, 5.0, 5.0])  # slope -2.0 m/s^2
    js, _ = run(sup, sm, a_plan=0.0, n=frames(1.0))
    assert js == pytest.approx(JERK_SCALE_MIN)

  def test_silent_when_model_unconfident(self):
    sup = BLTSupervisor()
    sm = make_sm(lead_prob=0.3, model_v=[15.0, 11.0, 8.0, 6.0, 5.0, 5.0])
    js, _ = run(sup, sm, a_plan=0.0, n=frames(1.0))
    assert js == 1.0

  def test_silent_on_mild_prediction(self):
    sup = BLTSupervisor()
    sm = make_sm(model_v=[15.0, 14.4, 13.8, 13.4, 13.0, 12.8])  # slope -0.3, dynamic onset's band
    js, _ = run(sup, sm, a_plan=0.0, n=frames(1.0))
    assert js == 1.0


class TestLaunchBoost:
  # lead pulling away AND accelerating, necessity zero, plan accel lagging the lead's
  LAUNCH = dict(v_ego=3.0, v_lead=5.0, d_rel=10.0, a_lead=2.0)

  def test_arms_on_lagging_launch(self):
    sup = BLTSupervisor()
    sm = make_sm(**self.LAUNCH)
    js, _ = run(sup, sm, a_plan=0.2, n=frames(0.35))
    assert js == 1.0  # debounce not yet met
    js, _ = run(sup, sm, a_plan=0.2, n=frames(1.0))
    assert js == pytest.approx(JERK_SCALE_MIN)

  def test_no_arm_on_constant_speed_pullaway(self):
    sup = BLTSupervisor()
    sm = make_sm(v_ego=3.0, v_lead=8.0, d_rel=15.0, a_lead=0.0)  # receding, not accelerating
    js, _ = run(sup, sm, a_plan=0.0, n=frames(2.0))
    assert js == 1.0

  def test_disarms_when_plan_catches_lead(self):
    sup = BLTSupervisor()
    sm = make_sm(**self.LAUNCH)
    run(sup, sm, a_plan=0.2, n=frames(1.5))
    js, _ = run(sup, sm, a_plan=2.0, n=frames(1.0))  # shortfall ~ 0; lead accelerating: no ratchet
    assert js == 1.0


class TestLeadDeparturePreRelease:
  @staticmethod
  def launch_sm(**kwargs):
    defaults = dict(v_ego=0.0, v_lead=0.0, v_lead_raw=0.0, d_rel=6.0,
                    model_v=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], standstill=True)
    defaults.update(kwargs)
    return make_sm(**defaults)

  def test_requires_sustained_prediction(self):
    release = LeadDeparturePreRelease()
    sm = self.launch_sm()
    for _ in range(frames(LEAD_DEPARTURE_CONFIRM) - 1):
      assert not release.update(sm, active=True)
    assert release.update(sm, active=True)

  def test_uses_radar_anchored_model_delta(self):
    release = LeadDeparturePreRelease()
    # The model's absolute v[1] is above threshold, but its predicted delta is only
    # 0.1 m/s and radar says the lead is stopped: no departure.
    sm = self.launch_sm(model_v=[2.0, 2.1, 2.2, 2.3, 2.4, 2.5])
    for _ in range(frames(1.0)):
      assert not release.update(sm, active=True)

  def test_collapsed_prediction_cancels_smoothly(self):
    release = LeadDeparturePreRelease()
    assert run_pre_release(release, self.launch_sm(), frames(LEAD_DEPARTURE_CONFIRM))
    collapsed = self.launch_sm(model_v=[0.0] * 6)
    for _ in range(frames(LEAD_DEPARTURE_CANCEL) - 1):
      assert release.update(collapsed, active=True)
    assert not release.update(collapsed, active=True)

  def test_measured_motion_releases_without_model_confirmation(self):
    release = LeadDeparturePreRelease()
    moving_lead = self.launch_sm(v_lead=0.4, v_lead_raw=0.4, model_v=[0.0] * 6)
    assert release.update(moving_lead, active=True)

    # A brief radar/model dropout must not pulse the hold while ego waits for brake bleed.
    stopped_reading = self.launch_sm(model_v=[0.0] * 6)
    assert release.update(stopped_reading, active=True)
    assert not release.update(self.launch_sm(standstill=False, v_ego=0.2), active=True)

  def test_reapplies_hold_if_measured_lead_re_stops(self):
    release = LeadDeparturePreRelease()
    moving_lead = self.launch_sm(v_lead=0.4, v_lead_raw=0.4, model_v=[0.0] * 6)
    assert release.update(moving_lead, active=True)
    stopped_lead = self.launch_sm(model_v=[0.0] * 6)
    assert not run_pre_release(release, stopped_lead, frames(LEAD_DEPARTURE_CANCEL))

  @pytest.mark.parametrize("kwargs,active", [
    ({"status": False}, True),
    ({"radar_valid": False}, True),
    ({"standstill": False, "v_ego": 0.2}, True),
    ({}, False),
    ({"lead_prob": 0.3}, True),
  ])
  def test_scope_guards(self, kwargs, active):
    release = LeadDeparturePreRelease()
    sm = self.launch_sm(**kwargs)
    for _ in range(frames(1.0)):
      assert not release.update(sm, active=active)


class TestWhiplashRatchet:
  def _armed_launch(self):
    sup = BLTSupervisor()
    run(sup, make_sm(**TestLaunchBoost.LAUNCH), a_plan=0.2, n=frames(1.5))
    assert sup.jerk_scale == pytest.approx(JERK_SCALE_MIN)
    return sup

  def test_holds_relaxed_while_lead_brakes_and_closing(self):
    sup = self._armed_launch()
    # launch flips to braking: trigger disarms, but the scale must not stiffen
    # (a_req = 1.0 + 1/22 ~ 1.05 < 1.5 -> non-emergency path)
    sm = make_sm(v_ego=5.0, v_lead=4.0, d_rel=15.0, a_lead=-1.0)
    js, _ = run(sup, sm, a_plan=0.0, n=frames(1.0))
    assert js == pytest.approx(JERK_SCALE_MIN)

  def test_releases_when_lead_stops_braking(self):
    sup = self._armed_launch()
    sm = make_sm(v_ego=5.0, v_lead=4.0, d_rel=15.0, a_lead=-1.0)
    run(sup, sm, a_plan=0.0, n=frames(1.0))
    sm = make_sm(v_ego=5.0, v_lead=4.0, d_rel=15.0, a_lead=0.0)
    js, _ = run(sup, sm, a_plan=0.0, n=frames(1.0))
    assert js == 1.0

  def test_emergency_standdown_overrides_ratchet(self):
    sup = self._armed_launch()
    # low TTC AND high necessity: stand-down returns to stock even though the lead
    # is braking hard (ttc = 15/8 ~ 1.9 < 3.5, a_req = 2 + 64/22 ~ 4.9 > 1.5)
    sm = make_sm(v_ego=10.0, v_lead=2.0, d_rel=15.0, a_lead=-2.0)
    js, _ = run(sup, sm, a_plan=-1.0, n=frames(1.0))
    assert js == 1.0


class TestSlewContinuity:
  def test_jerk_scale_never_steps(self):
    sup = BLTSupervisor()
    sm_boost = make_sm()
    sm_off = make_sm(status=False)
    prev = 1.0
    max_step = JERK_SCALE_RATE * DT_MDL + 1e-9
    # arm, then rip the lead away mid-boost: every frame must respect the slew
    for i in range(frames(3.0)):
      js, _ = sup.update(sm_boost if i < frames(1.5) else sm_off, -2.0, T_FOLLOW_BASE)
      assert abs(js - prev) <= max_step
      prev = js
    assert prev == 1.0

  def test_pad_respects_rates(self):
    sup = BLTSupervisor()
    sm = make_sm(v_ego=15.0, v_lead=14.0, d_rel=40.0, a_lead=-0.75)
    prev = 0.0
    for _ in range(frames(2.0)):
      _, tf = sup.update(sm, -0.5, T_FOLLOW_BASE)
      pad = tf - T_FOLLOW_BASE
      assert pad - prev <= ONSET_RATE_UP * DT_MDL + 1e-9
      assert prev - pad <= ONSET_RATE_DOWN * DT_MDL + 1e-9
      prev = pad


class TestDynamicOnset:
  def test_pad_proportional_to_lead_decel(self):
    sup = BLTSupervisor()
    # lead braking at half of ONSET_FULL_DECEL -> half of ONSET_PAD_MAX
    sm = make_sm(v_ego=15.0, v_lead=14.0, d_rel=40.0, a_lead=-0.75)
    _, tf = run(sup, sm, a_plan=-0.5, n=frames(2.0))
    assert tf - T_FOLLOW_BASE == pytest.approx(ONSET_PAD_MAX * 0.5)

  def test_no_pad_while_recovering(self):
    sup = BLTSupervisor()
    # ego already slower than the lead: holding the gap open is pure lag
    sm = make_sm(v_ego=13.0, v_lead=14.0, d_rel=40.0, a_lead=-0.75)
    _, tf = run(sup, sm, a_plan=-0.5, n=frames(2.0))
    assert tf == T_FOLLOW_BASE

  def test_stopped_lead_pad(self):
    sup = BLTSupervisor()
    # stopped lead never trips the decel-based pad; the closing-energy form must
    # (route 3e: ttc here is 3.33 < MIN_TTC but a_req = 1.125 < 1.5 keeps BLT active)
    sm = make_sm(v_ego=6.0, v_lead=0.0, d_rel=20.0, a_lead=0.0)
    _, tf = run(sup, sm, a_plan=-0.5, n=frames(2.0))
    expected = ONSET_PAD_MAX * min(1.125 / 1.2, 1.0)
    assert tf - T_FOLLOW_BASE == pytest.approx(expected, abs=1e-3)

  def test_base_passthrough_same_frame(self):
    sup = BLTSupervisor()
    sm = make_sm()
    sup.update(sm, 0.0, T_FOLLOW_BASE)
    # a personality change moves the output the SAME frame -- the pad slews, the base never
    _, tf = sup.update(sm, 0.0, 1.25)
    assert tf == 1.25
