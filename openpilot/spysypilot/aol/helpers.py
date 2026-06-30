# ALT_EXP flag value (must match opendbc/safety/spysypilot/aol_types.h)
ALT_EXP_AOL_ENABLE = 1024  # 0x400 — enables Always-On-Lateral in panda safety model


def set_alternative_experience(CP) -> None:
  """Set panda alternativeExperience to enable AOL lateral control.

  REMAIN_ACTIVE: only ALT_EXP_AOL_ENABLE is set. Panda allows steering
  regardless of brake or gas state.
  """
  CP.alternativeExperience |= ALT_EXP_AOL_ENABLE


def is_hyundai_always_allow(CP) -> bool:
  """Hyundai with LDA button or CAN-FD (e.g. Palisade) can activate AOL
  without requiring cruise to be available first."""
  try:
    from opendbc.car.hyundai.values import HyundaiFlags
    return bool(CP.flags & (HyundaiFlags.HAS_LDA_BUTTON | HyundaiFlags.CANFD))
  except Exception:
    return CP.brand == 'hyundai'