#!/usr/bin/env python3
"""Step 3-C: Rack Effort Shadow Observer (RESO).

Manager sibling of torqued: a log-only hold-torque (H) shadow learner for the
BLaTv3 rack controller. It has zero torque authority -- it never subscribes
to carControl/actuators and publishes only rackEffortFrame (on qualifying
candidate frames) and rackEffortSnapshot (every 60s and at shutdown). See
docs/RACK_EFFORT_OBSERVER.md and Documents/spysypilot-route-audit/phase3/
step3c_design/shadow_learner_design.md for the full design and rationale.

Restore/invalidate mirrors TorqueEstimator (torqued.py) exactly in shape: a
(carFingerprint, GRID_VERSION, DEF_VERSION, SNAPSHOT_VERSION) restore key
checked against the cached CarParamsPrevRoute + LiveRackEffortObserver Params
on boot; any mismatch or decode error discards the cache and starts empty
(same try/except-and-remove() pattern torqued.py uses).
"""
import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.locationd.rack_effort_classifier import (
  CellState, RackEffortAccumulator, SteadyClassifier, build_frame_sample, route_id_hash,
  GRID_VERSION, DEF_VERSION, SNAPSHOT_VERSION,
)

SNAPSHOT_FRAMES = 6000  # 60s at controlsState's 100 Hz, matching the design's "1/60 Hz" cadence


def get_restore_key(CP, grid_version, def_version, snapshot_version):
  return (CP.carFingerprint, grid_version, def_version, snapshot_version)


def restore_state(CP: car.CarParams) -> dict:
  params = Params()
  params_cache = params.get("CarParamsPrevRoute")
  snap_cache = params.get("LiveRackEffortObserver")
  if params_cache is None or snap_cache is None:
    return {}
  try:
    with log.Event.from_bytes(snap_cache) as log_evt:
      cache_snap = log_evt.rackEffortSnapshot
    with car.CarParams.from_bytes(params_cache) as msg:
      cache_CP = msg
    cached_key = get_restore_key(cache_CP, cache_snap.gridVersion, cache_snap.defVersion, cache_snap.version)
    if cached_key == get_restore_key(CP, GRID_VERSION, DEF_VERSION, SNAPSHOT_VERSION):
      cells = {}
      for c in cache_snap.cells:
        key = (c.vBandIdx, c.angleBandIdx, c.latAccelBinIdx, c.direction)
        cells[key] = CellState(bias_hat=c.biasHat, n_events=c.nEvents,
                                route_sketch={rc.routeIdHash: rc.nEvents for rc in c.routeSketch},
                                last_update_mono_time=c.lastUpdateMonoTime)
      cloudlog.info("restored rack effort observer state from cache")
      return cells
  except Exception:
    cloudlog.exception("failed to restore cached rack effort observer state")
    params.remove("LiveRackEffortObserver")
  return {}


def snapshot_msg(cells: dict, epoch_mono_time: int):
  msg = messaging.new_message('rackEffortSnapshot')
  snap = msg.rackEffortSnapshot
  snap.version = SNAPSHOT_VERSION
  snap.gridVersion = GRID_VERSION
  snap.defVersion = DEF_VERSION
  snap.epoch = epoch_mono_time
  cell_list = snap.init('cells', len(cells))
  for cell_msg, (key, state) in zip(cell_list, cells.items(), strict=True):
    cell_msg.vBandIdx, cell_msg.angleBandIdx, cell_msg.latAccelBinIdx, cell_msg.direction = key
    cell_msg.biasHat = state.bias_hat
    cell_msg.nEvents = min(state.n_events, 65535)
    cell_msg.lastUpdateMonoTime = state.last_update_mono_time
    rs_list = cell_msg.init('routeSketch', len(state.route_sketch))
    for rc_msg, (route_hash, n) in zip(rs_list, state.route_sketch.items(), strict=True):
      rc_msg.routeIdHash = route_hash
      rc_msg.nEvents = min(n, 65535)
  return msg


def frame_msg(result):
  msg = messaging.new_message('rackEffortFrame')
  f = msg.rackEffortFrame
  f.version = result.version
  f.runId = result.run_id
  f.vBandIdx = result.v_band_idx
  f.angleBandIdx = result.angle_band_idx
  f.latAccelBinIdx = result.lat_accel_bin_idx
  f.direction = result.direction
  f.hMeasured = result.h_measured
  f.hPrior = result.h_prior
  f.freezeBits = result.freeze_bits
  f.steeringTorqueRaw = result.steering_torque_raw
  f.calPerc = result.cal_perc
  f.vEgo = result.v_ego
  return msg


def build_sample_from_sm(sm, CP):
  cs = sm['controlsState']
  rack = cs.lateralControlState.rackState
  cc = sm['carControl']
  co = sm['carOutput']
  s = sm['carState']
  vp = sm['vehicleParameters']
  ltp = sm['lateralTorqueParameters']
  return build_frame_sample(
    log_mono_time=sm.logMonoTime['controlsState'],
    which_lateral='rackState', rack_fallback=bool(rack.fallback),
    lat_active=bool(cc.latActive), steering_pressed=bool(s.steeringPressed),
    steering_torque=float(s.steeringTorque), steering_angle_deg=float(s.steeringAngleDeg),
    steering_rate_deg=float(s.steeringRateDeg), v_ego=float(s.vEgo),
    h_measured=float(co.actuatorsOutput.torque),
    vp_valid=bool(vp.valid), roll_live=float(vp.roll), angle_offset_deg_live=float(vp.angleOffsetDeg),
    stiffness_factor_live=float(vp.stiffnessFactor), steer_ratio_live=float(vp.steerRatio),
    ltp_valid=bool(ltp.valid), ltp_cal_perc=int(ltp.calPerc), use_params=bool(ltp.useParams),
    lat_accel_factor_filtered=float(ltp.latAccelFactorFiltered),
    lat_accel_offset_filtered=float(ltp.latAccelOffsetFiltered),
    cp_mass=CP.mass, cp_center_to_front=CP.centerToFront, cp_wheelbase=CP.wheelbase,
    cp_steer_ratio_rear=CP.steerRatioRear, cp_tire_stiffness_front=CP.tireStiffnessFront,
    cp_tire_stiffness_rear=CP.tireStiffnessRear, cp_steer_ratio=CP.steerRatio,
  )


def main():
  config_realtime_process([0, 1, 2, 3], 5)

  params = Params()
  # Read once, like torqued.py -- CP does not change mid-drive, so this is a blocking
  # Params read rather than an ongoing SubMaster subscription (see docs/RACK_EFFORT_OBSERVER.md).
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)

  pm = messaging.PubMaster(['rackEffortFrame', 'rackEffortSnapshot'])
  # Never subscribes to carControl.actuators or anything upstream of it: carControl is read
  # only for latActive (extract.py's own steady-candidate gate), never for a torque value.
  sm = messaging.SubMaster(['carState', 'carControl', 'carOutput', 'controlsState',
                             'vehicleParameters', 'lateralTorqueParameters'], poll='controlsState')

  classifier = SteadyClassifier()  # on-device default: STEER_DRIVER_ALLOWANCE
  accumulator = RackEffortAccumulator(restore_state(CP))

  route_str = params.get("CurrentRoute") or ""
  route_hash = route_id_hash(route_str)

  def publish_snapshot():
    msg = snapshot_msg(accumulator.cells, sm.logMonoTime['controlsState'])
    pm.send('rackEffortSnapshot', msg)
    params.put("LiveRackEffortObserver", msg.to_bytes())

  try:
    while True:
      sm.update()

      if sm.updated['controlsState'] and sm.all_checks():
        cs = sm['controlsState']
        if cs.lateralControlState.which() == 'rackState':
          cur_route = params.get("CurrentRoute") or ""
          if cur_route != route_str:
            route_str = cur_route
            route_hash = route_id_hash(route_str)

          sample = build_sample_from_sm(sm, CP)
          result = classifier.step(sample)
          if result is not None:
            accumulator.feed(result, route_hash=route_hash)
            pm.send('rackEffortFrame', frame_msg(result))

      # Snapshot cadence only *reads* accumulator.cells -- it must never flush() the
      # in-progress run here, or a periodic snapshot boundary would fragment one real
      # dwell-event into two artificially short ones (see test_rack_effort_snapshot.py).
      if sm.frame > 0 and sm.frame % SNAPSHOT_FRAMES == 0:
        publish_snapshot()
  finally:
    # flush() only at shutdown, per the design's "60s + shutdown" cadence.
    accumulator.flush(route_hash=route_hash)
    publish_snapshot()


if __name__ == "__main__":
  main()
