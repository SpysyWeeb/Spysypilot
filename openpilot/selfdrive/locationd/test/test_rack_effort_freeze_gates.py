#!/usr/bin/env python3
"""One test per freeze gate (shadow_learner_design.md tests_required item 3),
each a minimal synthetic frame sequence tripping only that gate, plus one
confirming the fallback bit is logged but never blocks accumulation in v1.

Gates 1/2/4 are structural: a tripped frame never becomes a steady candidate
at all (SteadyClassifier.step returns None). Gate 3 (paramsMoving) and the
fallback bit are logged-but-conditional: the frame is still a candidate and
still gets a rackEffortFrame, but RackEffortAccumulator.feed()/flush() must
skip (gate 3) or accept (fallback) folding its run into biasHat.
"""
from openpilot.selfdrive.locationd.rack_effort_classifier import (
  FrameSample, RackEffortAccumulator, SteadyClassifier, STEADY_HALF_FRAMES,
  STEER_DRIVER_ALLOWANCE, HANDS_OFF_TORQUE_THRESH_OFFLINE,
  FREEZE_DRIVER_OVERRIDE, FREEZE_PARAMS_MOVING, FREEZE_RACK_FALLBACK,
)

DT_NS = 10_000_000  # 10ms, matches DT_NOMINAL
N = 2 * STEADY_HALF_FRAMES + 10  # enough to fill the window plus a handful of emitted centers


def make_sample(i, **overrides):
  base = {
    "log_mono_time": i * DT_NS, "which_lateral": "rackState", "rack_fallback": False,
    "lat_active": True, "steering_pressed": False, "steering_torque": 5.0,
    "steering_angle_deg": 10.0, "steering_rate_deg": 0.0, "v_ego": 20.0,
    "h_measured": 0.3, "lat_accel": 1.0, "h_prior": 0.25,
    "ltp_cal_perc": 100, "use_params": True,
  }
  base.update(overrides)
  return FrameSample(**base)


def feed_all(clf, sample_fn, n=N):
  return [clf.step(sample_fn(i)) for i in range(n)]


def test_baseline_is_steady_with_no_freeze_bits():
  results = feed_all(SteadyClassifier(), make_sample)
  logged = [r for r in results if r is not None]
  assert len(logged) > 0
  assert all(r.freeze_bits == 0 for r in logged)


def test_gate1_driver_override_steering_pressed():
  results = feed_all(SteadyClassifier(), lambda i: make_sample(i, steering_pressed=True))
  assert all(r is None for r in results)


def test_gate1_driver_override_torque_above_on_device_allowance():
  clf = SteadyClassifier(hands_off_torque_thresh=STEER_DRIVER_ALLOWANCE)
  results = feed_all(clf, lambda i: make_sample(i, steering_torque=STEER_DRIVER_ALLOWANCE + 5))
  assert all(r is None for r in results)


def test_gate1_nan_steering_torque_excluded_like_extracts_vectorized_mask():
  """extract.py's candidate gate is `np.abs(steeringTorque) < THRESH`, which is
  False (excluded) for a NaN steeringTorque; the scalar mirror must exclude a
  NaN frame too, not let it slip through because `>= thresh` is also False."""
  import math
  results = feed_all(SteadyClassifier(), lambda i: make_sample(i, steering_torque=math.nan))
  assert all(r is None for r in results)


def test_gate1_bit_is_meaningful_between_offline_and_on_device_thresholds():
  """A frame between the offline 30 and on-device 50 cut still qualifies as a
  live candidate, but freezeBits bit0 flags it as one that would fail the
  stricter offline definition -- this is steeringTorqueRaw's whole purpose."""
  torque = (HANDS_OFF_TORQUE_THRESH_OFFLINE + STEER_DRIVER_ALLOWANCE) / 2  # 40
  clf = SteadyClassifier(hands_off_torque_thresh=STEER_DRIVER_ALLOWANCE)
  results = feed_all(clf, lambda i: make_sample(i, steering_torque=torque))
  logged = [r for r in results if r is not None]
  assert len(logged) > 0
  assert all(r.freeze_bits & FREEZE_DRIVER_OVERRIDE for r in logged)
  assert all(r.steering_torque_raw == torque for r in logged)


def test_gate2_saturated():
  results = feed_all(SteadyClassifier(), lambda i: make_sample(i, h_measured=0.97))
  assert all(r is None for r in results)


def test_gate4_non_rackstate_lateral_controller_excluded_structurally():
  results = feed_all(SteadyClassifier(), lambda i: make_sample(i, which_lateral="torqueState"))
  assert all(r is None for r in results)


def test_gate3_params_moving_logged_but_never_accumulated():
  results = feed_all(SteadyClassifier(), lambda i: make_sample(i, ltp_cal_perc=40))
  logged = [r for r in results if r is not None]
  assert len(logged) >= 3  # long enough to otherwise form a valid event
  assert all(r.freeze_bits & FREEZE_PARAMS_MOVING for r in logged)

  acc = RackEffortAccumulator()
  for r in logged:
    acc.feed(r)
  acc.flush()
  assert acc.cells == {}, "a run containing a paramsMoving frame must never update biasHat"


def test_gate3_use_params_flip_holds_for_about_one_second():
  clf = SteadyClassifier()
  # flip useParams once, then confirm frames within the ~1s hold are still marked paramsMoving
  results = []
  for i in range(N):
    use_params = False if i < 5 else True
    results.append(clf.step(make_sample(i, use_params=use_params)))
  logged = [(i, r) for i, r in enumerate(results) if r is not None]
  assert any(r.freeze_bits & FREEZE_PARAMS_MOVING for _, r in logged), \
    "frames shortly after a useParams flip should still be flagged paramsMoving"


def test_fallback_bit_logged_but_never_gates_accumulation_in_v1():
  results = feed_all(SteadyClassifier(), lambda i: make_sample(i, rack_fallback=True))
  logged = [r for r in results if r is not None]
  assert len(logged) >= 3
  assert all(r.freeze_bits & FREEZE_RACK_FALLBACK for r in logged)

  acc = RackEffortAccumulator()
  for r in logged:
    acc.feed(r)
  acc.flush()
  assert len(acc.cells) == 1
  cell = next(iter(acc.cells.values()))
  assert cell.n_events == 1
  assert cell.bias_hat != 0.0
