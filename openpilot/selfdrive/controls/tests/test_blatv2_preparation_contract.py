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
  for _ in range(100_000):
    float(decoded.lateralTuning.torque.latAccelFactor)

  assert decoded.carFingerprint == "TEST_CAR"
  assert decoded.mass == 2_000.0
  assert decoded.wheelbase == 3.0
  assert decoded.as_builder().to_bytes() == encoded
