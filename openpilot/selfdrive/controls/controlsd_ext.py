import openpilot.cereal.messaging as messaging


class ControlsExt:
  def get_lat_active(self, sm: messaging.SubMaster) -> bool:
    """Determine if lateral (steering) control should be active.

    When AOL is available, use aol.active instead of selfdriveState.active.
    This allows steering when ACC is off (always-on-lateral).
    """
    sp_state = sm['spysydriveStateSP']
    if sp_state.aol.available:
      return bool(sp_state.aol.active)

    # Fallback to stock behavior if AOL state unavailable
    return bool(sm['selfdriveState'].active)
