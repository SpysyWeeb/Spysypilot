from opendbc.car.structs import car

from openpilot.selfdrive.controls.lib.blatv2.preparation_contract import decode_car_params


def test_decode_car_params_returns_owned_reusable_message() -> None:
  original = car.CarParams.new_message()
  original.carFingerprint = "TEST_CAR"
  original.mass = 2_000.0
  original.wheelbase = 3.0
  original.lateralTuning.init("torque")
  encoded = original.to_bytes()

  decoded = decode_car_params(encoded)
  # The historical borrowed reader exhausted its cumulative traversal budget
  # at roughly 1.68 million accesses during the two-pass physical replay.
  for _ in range(2_000_000):
    float(decoded.lateralTuning.torque.latAccelFactor)

  assert decoded.carFingerprint == "TEST_CAR"
  assert decoded.mass == 2_000.0
  assert decoded.wheelbase == 3.0
  assert decoded.as_builder().to_bytes() == encoded
