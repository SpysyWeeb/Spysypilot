import openpilot.cereal.messaging as messaging


def test_model_v2_big_field():
  msg = messaging.new_message("modelV2")
  msg.modelV2.big = False
  assert msg.modelV2.big is False
