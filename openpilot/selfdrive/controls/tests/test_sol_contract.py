from openpilot.cereal import custom, log
from openpilot.cereal.services import SERVICE_LIST


def test_sol_wire_contract():
  # SOL uses the fork-reserved service/fields, without renumbering stock schemas.
  assert 'spysydriveStateSP' in SERVICE_LIST
  assert hasattr(custom, 'AolState')
  event = log.Event.new_message()
  state = event.init('spysydriveStateSP').aol
  state.available = True
  state.enabled = True
  state.active = True
  assert state.available and state.enabled and state.active
  assert log.PandaState.new_message(lateralAllowed=True).lateralAllowed
