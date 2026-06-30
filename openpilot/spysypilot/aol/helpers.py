def is_hyundai_always_allow(CP) -> bool:
  """Hyundai with LDA button or CAN-FD (e.g. Palisade) can activate AOL
  without requiring cruise to be available first."""
  try:
    from opendbc.car.hyundai.values import HyundaiFlags
    return bool(CP.flags & (HyundaiFlags.HAS_LDA_BUTTON | HyundaiFlags.CANFD))
  except Exception:
    return CP.brand == 'hyundai'
