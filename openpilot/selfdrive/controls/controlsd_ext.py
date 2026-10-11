from opendbc.car.structs import car
import openpilot.cereal.messaging as messaging

PANDA_PERMISSION_MAX_AGE_NS = 250_000_000  # 2.5 reports at pandaStates' 10 Hz rate.


class ControlsExt:
  def __init__(self):
    self._aol_requested = False
    self._aol_request_time = 0
    self._aol_first_panda_time = 0

  def get_lat_active(self, sm: messaging.SubMaster) -> bool:
    """Determine if lateral (steering) control should be active.

    AOL may steer independently of ACC, but only after Panda confirms permission.
    """
    if not sm.alive['spysydriveStateSP'] or not sm.valid['spysydriveStateSP']:
      self._aol_requested = False
      return False

    sp_state = sm['spysydriveStateSP']
    if sp_state.aol.available:
      if sp_state.aol.active and not self._aol_requested:
        self._aol_request_time = sm.logMonoTime['spysydriveStateSP']
        self._aol_first_panda_time = 0
      self._aol_requested = bool(sp_state.aol.active)
      panda_time = sm.logMonoTime['pandaStates']
      # The independently freshness-checked 100 Hz AOL state bounds permission age.
      if not self._aol_requested or not sm.all_checks(['pandaStates']) or \
         sm.logMonoTime['spysydriveStateSP'] - panda_time >= PANDA_PERMISSION_MAX_AGE_NS:
        self._aol_first_panda_time = 0
        return False
      if panda_time <= self._aol_request_time:
        return False
      if not self._aol_first_panda_time:
        self._aol_first_panda_time = panda_time
      # Keep the host's torque and high-angle histories reset until Panda can
      # accept them. pandad samples health BEFORE sending its heartbeat, so the
      # first post-request sample can precede a queued off. Independent lateral
      # needs a later sample; controlsAllowed is its own grant. The heartbeat
      # still uses aol.enabled, not this gate. Ignore non-actuating Pandas.
      pandas = [p for p in sm['pandaStates'] if p.safetyModel not in
                (car.CarParams.SafetyModel.silent, car.CarParams.SafetyModel.noOutput)]
      return bool(pandas) and all(p.safetyModel in (car.CarParams.SafetyModel.hyundai,
                                                  car.CarParams.SafetyModel.hyundaiLegacy,
                                                  car.CarParams.SafetyModel.hyundaiCanfd) and
                                 (p.controlsAllowed or (p.lateralAllowed and panda_time > self._aol_first_panda_time)) for p in pandas)

    # Fallback to stock behavior if AOL state unavailable
    self._aol_requested = False
    return bool(sm['selfdriveState'].active)
