#!/usr/bin/env python3
"""Pure-python core for the Rack Effort Shadow Observer (RESO, step 3-C).

Shared by the on-device process (rack_effort_observer.py) and the offline
mirror in phase4/rack_effort_seed (extract.py's additive decode, verify2.py,
fit.py's replay mode), so the live classifier and the offline reference can
never drift apart silently -- every consumer calls the same scalar math and
windowing logic defined here.

Reproduces /var/home/alex/Documents/spysypilot-route-audit/phase4/
rack_effort_seed/extract.py's vectorized H (steady-frame) definitions
frame-for-frame; see that file for the derivation notes this only restates in
scalar/streaming form. G (moving-frame / rate-gain) bookkeeping is out of
scope for v1 -- see shadow_learner_design.md's "chosen": H-only, log-only,
zero torque authority, zero G bookkeeping.

No cereal/messaging import here on purpose: this module has no path to
carControl/actuators and nothing in it can publish or drive them (see
test_rack_effort_isolation.py).
"""
import hashlib
import math
from collections import deque
from dataclasses import dataclass, field

G = 9.81  # ACCELERATION_DUE_TO_GRAVITY, opendbc/car/__init__.py

# ---- cell grid: byte-identical to extract.py. GRID_VERSION bumps only if these change. ----
GRID_VERSION = 1
V_BANDS = [(0, 2), (2, 4), (4, 7), (7, 10), (10, 15), (15, 20), (20, 30), (30, 1e9)]
ANGLE_BANDS = [(0, 2), (2, 5), (5, 15), (15, 45), (45, 120), (120, 1e9)]
LATACCEL_BIN_WIDTH = 0.5   # m/s^2, signed bins centered on 0
LATACCEL_BIN_CLIP = 8.0    # clip |latAccel| beyond this into the extreme bin
DIRECTION_DEADBAND_DEG = 0.05

# ---- steady (H) frame-selection thresholds: byte-identical to extract.py ----
STEADY_WINDOW_S = 0.30
STEADY_HALF_S = 0.15
STEADY_RATE_THRESH = 2.0            # deg/s
STEADY_ANGLE_RANGE_THRESH = 0.5     # deg, max-min steeringAngleDeg over the window
STEADY_TORQUE_SAT_THRESH = 0.95     # |hMeasured| below this (not saturated)
STEADY_TORQUE_DELTA_THRESH = 0.008  # |per-frame delta hMeasured| below this everywhere in window
GAP_MULT = 3.0                      # a frame-to-frame gap > GAP_MULT*dt_nominal breaks contiguity
MIN_RUN_FRAMES = 3                  # events shorter than this are discarded (learner_note.md sec.2)

# The one deliberate on-device deviation from extract.py's offline definitions
# (learner_note.md section 4): the offline seed's hands-off gate used a more
# conservative 30 native-CAN-unit threshold to bias H_surface.json toward
# clean anchors; the live gate uses the platform's own driver-override
# allowance instead. steeringTorqueRaw is still logged per frame so offline
# can retroactively test either cut.
HANDS_OFF_TORQUE_THRESH_OFFLINE = 30.0  # extract.py's HANDS_OFF_TORQUE_THRESH, exact mirror
STEER_DRIVER_ALLOWANCE = 50.0           # on-device value, learner_note.md section 4

DT_NOMINAL = 0.01  # controlsState cadence (100 Hz / DT_CTRL) -- fixed on a live stream, unlike
                    # extract.py which derives it per-route from decoded timestamps.
STEADY_HALF_FRAMES = max(1, round(STEADY_HALF_S / DT_NOMINAL))  # 15

PARAMS_MOVING_HOLD_S = 1.0  # freeze gate 3's hysteresis window after a torqued useParams flip
N_CAP_H = 20                # learner_note.md section 3 ("count-based EMA, capped")
ROUTE_SKETCH_CAP = 16       # shadow_learner_panel.json capnp_additions: routeSketch capped 16

FRAME_VERSION = 1
SNAPSHOT_VERSION = 1
DEF_VERSION = 1  # bumps only when the mask/hPrior definition itself changes (e.g. a future
                 # fallback exclusion) -- see shadow_learner_design.md "chosen".

# freezeBits (RackEffortFrame.freezeBits, see log.capnp)
FREEZE_DRIVER_OVERRIDE = 1 << 0  # steeringPressed, or |steeringTorque| >= the OFFLINE 30 cut
FREEZE_SATURATED = 1 << 1        # |hMeasured| >= STEADY_TORQUE_SAT_THRESH
FREEZE_PARAMS_MOVING = 1 << 2    # torqued's own calPerc<100 or a recent useParams flip
FREEZE_RACK_FALLBACK = 1 << 3    # audit-only in v1: logged, never gates (see design "chosen")


# ---------------------------------------------------------------------------
# Scalar physics: opendbc VehicleModel.calc_curvature closed form + torqued's
# linear hold-torque model, reproduced scalar-for-scalar from extract.py's
# calc_curvature_vec / h_prior_vec (see that file for the full derivation
# notes on both the sign convention and why friction is not folded into h_prior).
# ---------------------------------------------------------------------------

def calc_curvature(sa_rad, u, roll, stiffness_factor, steer_ratio,
                    mass, center_to_front, wheelbase, steer_ratio_rear,
                    tire_stiffness_front, tire_stiffness_rear):
  aF = center_to_front
  l = wheelbase
  aR = l - aF
  cF = stiffness_factor * tire_stiffness_front
  cR = stiffness_factor * tire_stiffness_rear
  sf = mass * (cF * aF - cR * aR) / (l ** 2 * cF * cR)
  denom = 1.0 - sf * u ** 2
  if denom == 0.0:
    return float("nan")
  curvature_factor = (1.0 - steer_ratio_rear) / denom / l
  roll_comp = 0.0 if abs(sf) < 1e-6 else G * roll / (1.0 / sf - u ** 2)
  return curvature_factor * sa_rad / steer_ratio + roll_comp


def lateral_accel_signed(curvature, v_ego):
  # sign convention: LatControlTorque.update() computes measured_curvature = -VM.calc_curvature(...)
  # before multiplying by v^2 -- see extract.py's calc_curvature_vec docstring.
  return -curvature * v_ego ** 2


def h_prior(lat_accel_signed, roll, lat_accel_factor, lat_accel_offset):
  net = lat_accel_signed - G * roll - lat_accel_offset
  if not math.isfinite(lat_accel_factor) or lat_accel_factor == 0.0:
    return float("nan")
  return -(net / lat_accel_factor)


def band_index(value, bands):
  av = abs(value)
  for i, (lo, hi) in enumerate(bands):
    if lo <= av < hi:
      return i
  return -1


def latAccel_bin(lat_accel):
  clipped = max(-LATACCEL_BIN_CLIP, min(LATACCEL_BIN_CLIP, lat_accel))
  return int(math.floor(clipped / LATACCEL_BIN_WIDTH))


def direction_of(steering_angle_deg):
  if steering_angle_deg > DIRECTION_DEADBAND_DEG:
    return 1
  if steering_angle_deg < -DIRECTION_DEADBAND_DEG:
    return -1
  return 0


def effective_steer_ratio(vp_valid, steer_ratio_live, cp_steer_ratio):
  return steer_ratio_live if (vp_valid and steer_ratio_live > 1.0) else cp_steer_ratio


def effective_stiffness_factor(vp_valid, stiffness_factor_live):
  return stiffness_factor_live if (vp_valid and stiffness_factor_live > 0.1) else 1.0


def effective_roll(vp_valid, roll_live):
  return roll_live if vp_valid else 0.0


def effective_angle_offset_deg(vp_valid, angle_offset_deg_live):
  return angle_offset_deg_live if vp_valid else 0.0


def effective_torque_params(ltp_valid, lat_accel_factor_filtered, lat_accel_offset_filtered):
  if ltp_valid and math.isfinite(lat_accel_factor_filtered) and abs(lat_accel_factor_filtered) > 1e-3:
    laf = lat_accel_factor_filtered
  else:
    laf = float("nan")
  lao = lat_accel_offset_filtered if ltp_valid else float("nan")
  return laf, lao


def build_frame_sample(*, log_mono_time, which_lateral, rack_fallback, lat_active, steering_pressed,
                        steering_torque, steering_angle_deg, steering_rate_deg, v_ego, h_measured,
                        vp_valid, roll_live, angle_offset_deg_live, stiffness_factor_live, steer_ratio_live,
                        ltp_valid, ltp_cal_perc, use_params, lat_accel_factor_filtered, lat_accel_offset_filtered,
                        cp_mass, cp_center_to_front, cp_wheelbase, cp_steer_ratio_rear,
                        cp_tire_stiffness_front, cp_tire_stiffness_rear, cp_steer_ratio) -> "FrameSample":
  """Assembles one FrameSample from raw signal values, applying the same
  vehicleParameters/lateralTorqueParameters validity gating extract.py's
  assemble_route uses before calling calc_curvature_vec/h_prior_vec. This is
  the single place that logic lives -- the live process, the bit-exact test,
  and the offline decoders all call it instead of re-deriving it."""
  steer_ratio = effective_steer_ratio(vp_valid, steer_ratio_live, cp_steer_ratio)
  stiffness_factor = effective_stiffness_factor(vp_valid, stiffness_factor_live)
  roll = effective_roll(vp_valid, roll_live)
  angle_offset_deg = effective_angle_offset_deg(vp_valid, angle_offset_deg_live)
  sa_rad = math.radians(steering_angle_deg - angle_offset_deg)
  curvature = calc_curvature(sa_rad, v_ego, roll, stiffness_factor, steer_ratio,
                              cp_mass, cp_center_to_front, cp_wheelbase, cp_steer_ratio_rear,
                              cp_tire_stiffness_front, cp_tire_stiffness_rear)
  lat_accel = lateral_accel_signed(curvature, v_ego)
  laf, lao = effective_torque_params(ltp_valid, lat_accel_factor_filtered, lat_accel_offset_filtered)
  hp = h_prior(lat_accel, roll, laf, lao)
  return FrameSample(
    log_mono_time=log_mono_time, which_lateral=which_lateral, rack_fallback=rack_fallback,
    lat_active=lat_active, steering_pressed=steering_pressed, steering_torque=steering_torque,
    steering_angle_deg=steering_angle_deg, steering_rate_deg=steering_rate_deg, v_ego=v_ego,
    h_measured=h_measured, lat_accel=lat_accel, h_prior=hp, ltp_cal_perc=ltp_cal_perc, use_params=use_params,
  )


def route_id_hash(route_str: str) -> int:
  """Stable (cross-process, cross-run) 64-bit hash of a route identity string.

  Python's builtin hash() is salted per-process (PYTHONHASHSEED), which would
  make routeIdHash meaningless across restarts -- use a fixed digest instead.
  """
  digest = hashlib.sha256(route_str.encode()).digest()
  return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# Per-frame input / classified output
# ---------------------------------------------------------------------------

@dataclass
class FrameSample:
  """One controlsState-anchored frame's worth of raw signal (see
  docs/RACK_EFFORT_OBSERVER.md for the field-to-message mapping)."""
  log_mono_time: int
  which_lateral: str        # controlsState.lateralControlState.which()
  rack_fallback: bool       # rackState.fallback
  lat_active: bool          # carControl.latActive
  steering_pressed: bool
  steering_torque: float    # carState.steeringTorque
  steering_angle_deg: float
  steering_rate_deg: float
  v_ego: float
  h_measured: float         # carOutput.actuatorsOutput.torque
  lat_accel: float          # derived: lateral_accel_signed(calc_curvature(...), v_ego)
  h_prior: float            # derived: h_prior(lat_accel, roll, latAccelFactorFiltered, latAccelOffsetFiltered)
  ltp_cal_perc: int         # lateralTorqueParameters.calPerc
  use_params: bool          # lateralTorqueParameters.useParams


@dataclass
class ClassifiedFrame:
  version: int
  run_id: int
  v_band_idx: int
  angle_band_idx: int
  lat_accel_bin_idx: int
  direction: int
  h_measured: float
  h_prior: float
  h_residual: float
  freeze_bits: int
  steering_torque_raw: float
  cal_perc: int
  v_ego: float
  log_mono_time: int


@dataclass
class _WinFrame:
  candidate: bool
  contiguity_run: int
  steering_angle_deg: float
  sample: FrameSample
  h_residual: float


class SteadyClassifier:
  """Streaming (causal, fixed-lag) port of extract.py's steady-frame (H)
  classifier: a centered rolling window means every result is delayed by
  STEADY_HALF_FRAMES frames behind the input it was computed from.

  `hands_off_torque_thresh` defaults to the on-device STEER_DRIVER_ALLOWANCE;
  pass HANDS_OFF_TORQUE_THRESH_OFFLINE to reproduce extract.py's own gate
  exactly (used by the bit-exact replay test).
  """

  def __init__(self, hands_off_torque_thresh: float = STEER_DRIVER_ALLOWANCE):
    self.hands_off_torque_thresh = hands_off_torque_thresh
    self.half = STEADY_HALF_FRAMES
    self.win_len = 2 * self.half + 1
    self._buf: deque[_WinFrame] = deque(maxlen=self.win_len)
    self._contiguity_run = 0
    self._last_lm = None
    self._prev_h_measured = None
    self._last_use_params = None
    self._last_use_params_flip_lm = None
    # event/run bookkeeping over the *emitted* (steady) sequence
    self._event_run_id = 0
    self._prev_emitted_mask = False
    self._prev_emitted_cell = None

  def _candidate(self, s: FrameSample) -> bool:
    if s.which_lateral != "rackState":
      return False
    if not s.lat_active or s.steering_pressed:
      return False
    # written as "not (< thresh)" rather than ">= thresh" so a NaN steering_torque
    # excludes the frame exactly like extract.py's `np.abs(x) < THRESH` mask does
    # (NaN comparisons are always False either way; ">=" would let NaN slip through)
    if not abs(s.steering_torque) < self.hands_off_torque_thresh:
      return False
    if not math.isfinite(s.h_measured) or abs(s.h_measured) >= STEADY_TORQUE_SAT_THRESH:
      return False
    if not math.isfinite(s.steering_rate_deg) or abs(s.steering_rate_deg) >= STEADY_RATE_THRESH:
      return False
    if self._prev_h_measured is not None and math.isfinite(self._prev_h_measured):
      if abs(s.h_measured - self._prev_h_measured) >= STEADY_TORQUE_DELTA_THRESH:
        return False
    return True

  def _params_moving(self, s: FrameSample) -> bool:
    if s.ltp_cal_perc < 100:
      return True
    if self._last_use_params_flip_lm is None:
      return False
    return (s.log_mono_time - self._last_use_params_flip_lm) * 1e-9 < PARAMS_MOVING_HOLD_S

  def step(self, s: FrameSample) -> ClassifiedFrame | None:
    """Feed one new frame; returns a ClassifiedFrame for the *center* of the
    window (STEADY_HALF_FRAMES frames behind `s`) iff that center frame
    qualifies as a steady candidate, else None."""
    # contiguity run id: breaks on a time gap > GAP_MULT * DT_NOMINAL
    if self._last_lm is not None:
      dt = (s.log_mono_time - self._last_lm) * 1e-9
      if dt > GAP_MULT * DT_NOMINAL:
        self._contiguity_run += 1
    self._last_lm = s.log_mono_time

    if self._last_use_params is not None and s.use_params != self._last_use_params:
      self._last_use_params_flip_lm = s.log_mono_time
    self._last_use_params = s.use_params

    params_moving = self._params_moving(s)
    candidate = self._candidate(s)
    h_residual = s.h_measured - s.h_prior if math.isfinite(s.h_prior) else float("nan")
    self._prev_h_measured = s.h_measured

    self._buf.append(_WinFrame(candidate=candidate, contiguity_run=self._contiguity_run,
                                steering_angle_deg=s.steering_angle_deg, sample=s, h_residual=h_residual))
    if len(self._buf) < self.win_len:
      return None

    center = self._buf[self.half]
    win_candidate = all(f.candidate for f in self._buf)
    win_same_run = all(f.contiguity_run == center.contiguity_run for f in self._buf)
    angles = [f.steering_angle_deg for f in self._buf]
    angle_ptp = max(angles) - min(angles)
    mask = (win_candidate and win_same_run and angle_ptp < STEADY_ANGLE_RANGE_THRESH
            and math.isfinite(center.sample.h_prior))

    if not mask:
      self._prev_emitted_mask = False
      self._prev_emitted_cell = None
      return None

    cs = center.sample
    v_band_idx = band_index(cs.v_ego, V_BANDS)
    angle_band_idx = band_index(cs.steering_angle_deg, ANGLE_BANDS)
    lat_bin_idx = latAccel_bin(cs.lat_accel)
    direction = direction_of(cs.steering_angle_deg)
    cell = (v_band_idx, angle_band_idx, lat_bin_idx, direction)

    if self._prev_emitted_mask and cell == self._prev_emitted_cell:
      pass  # same event continues
    else:
      self._event_run_id += 1
    self._prev_emitted_mask = True
    self._prev_emitted_cell = cell

    freeze_bits = 0
    if cs.steering_pressed or abs(cs.steering_torque) >= HANDS_OFF_TORQUE_THRESH_OFFLINE:
      freeze_bits |= FREEZE_DRIVER_OVERRIDE
    if abs(cs.h_measured) >= STEADY_TORQUE_SAT_THRESH:
      freeze_bits |= FREEZE_SATURATED
    if params_moving:
      freeze_bits |= FREEZE_PARAMS_MOVING
    if cs.rack_fallback:
      freeze_bits |= FREEZE_RACK_FALLBACK

    return ClassifiedFrame(
      version=FRAME_VERSION, run_id=self._event_run_id,
      v_band_idx=v_band_idx, angle_band_idx=angle_band_idx,
      lat_accel_bin_idx=lat_bin_idx, direction=direction,
      h_measured=cs.h_measured, h_prior=cs.h_prior, h_residual=center.h_residual,
      freeze_bits=freeze_bits, steering_torque_raw=cs.steering_torque,
      cal_perc=cs.ltp_cal_perc, v_ego=cs.v_ego, log_mono_time=cs.log_mono_time,
    )


# ---------------------------------------------------------------------------
# Event -> per-cell EMA accumulation (learner_note.md sections 2-3): one
# update per contiguous run (>= MIN_RUN_FRAMES), sampled as median(hResidual)
# over the run, folded in via a count-capped EMA. Skips (does not accumulate)
# any run containing a paramsMoving-flagged frame -- freeze gate 3. The
# rackFallback bit never gates accumulation in v1 (see FREEZE_RACK_FALLBACK).
# ---------------------------------------------------------------------------

@dataclass
class CellState:
  bias_hat: float = 0.0
  n_events: int = 0
  route_sketch: dict = field(default_factory=dict)  # routeIdHash -> nEvents, capped ROUTE_SKETCH_CAP
  last_update_mono_time: int = 0


class RackEffortAccumulator:
  """Folds a stream of ClassifiedFrame (grouped into runs by run_id + cell)
  into per-cell bias_hat state -- the on-device logic and fit.py's offline
  replay mode both call this so they can never disagree by construction."""

  def __init__(self, cells: dict | None = None):
    self.cells: dict[tuple, CellState] = cells if cells is not None else {}
    self._run_id = None
    self._run_cell = None
    self._run_frames: list[ClassifiedFrame] = []

  def _cell_key(self, f: ClassifiedFrame):
    return (f.v_band_idx, f.angle_band_idx, f.lat_accel_bin_idx, f.direction)

  def _finalize_run(self, route_hash: int | None):
    frames = self._run_frames
    self._run_frames = []
    if len(frames) < MIN_RUN_FRAMES:
      return
    if any(f.freeze_bits & FREEZE_PARAMS_MOVING for f in frames):
      return  # freeze gate 3: torqued's own params were moving during this run
    key = self._cell_key(frames[0])
    residuals = sorted(f.h_residual for f in frames)
    mid = len(residuals) // 2
    if len(residuals) % 2:
      median_residual = residuals[mid]
    else:
      median_residual = 0.5 * (residuals[mid - 1] + residuals[mid])

    cell = self.cells.setdefault(key, CellState())
    cell.n_events += 1
    alpha = 1.0 / min(cell.n_events, N_CAP_H)
    cell.bias_hat += alpha * (median_residual - cell.bias_hat)
    cell.last_update_mono_time = frames[-1].log_mono_time
    if route_hash is not None:
      if route_hash in cell.route_sketch:
        cell.route_sketch[route_hash] += 1
      elif len(cell.route_sketch) < ROUTE_SKETCH_CAP:
        cell.route_sketch[route_hash] = 1
      # else: sketch is full -- documented lower-bound undercount, not an error

  def feed(self, f: ClassifiedFrame, route_hash: int | None = None):
    key = self._cell_key(f)
    if self._run_id is not None and (f.run_id != self._run_id or key != self._run_cell):
      self._finalize_run(route_hash)
    self._run_id = f.run_id
    self._run_cell = key
    self._run_frames.append(f)

  def flush(self, route_hash: int | None = None):
    """Finalize any in-progress run (call at shutdown / snapshot boundary)."""
    if self._run_frames:
      self._finalize_run(route_hash)
    self._run_id = None
    self._run_cell = None
