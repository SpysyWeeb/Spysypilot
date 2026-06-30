import openpilot.cereal.messaging as messaging
from openpilot.spysypilot.selfdrive.controls.lib.blinker_pause_lateral import BlinkerPauseLateral


class ControlsExt:
  def __init__(self):
    self.blinker_pause_lateral = BlinkerPauseLateral()

  def get_lat_active(self, sm: messaging.SubMaster) -> bool:
    """Determine if lateral (steering) control should be active.

    When MADS is available, use mads.active instead of selfdriveState.active.
    This allows steering when ACC is off (always-on-lateral).
    """
    if self.blinker_pause_lateral.update(sm['carState']):
      return False

    sp_state = sm['spysydriveStateSP']
    if sp_state.aol.available:
      return bool(sp_state.aol.active)

    # Fallback to stock behavior if MADS state unavailable
    return bool(sm['selfdriveState'].active)