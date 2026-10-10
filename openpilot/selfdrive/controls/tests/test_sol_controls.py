import pytest

from openpilot.cereal import messaging

from openpilot.selfdrive.controls.controlsd_ext import ControlsExt


class SubMaster(messaging.SubMaster):
  def __init__(self, *, fresh=True, valid=True, available=True, active=True, stock_active=False):
    super().__init__(['spysydriveStateSP', 'selfdriveState'])
    sol = messaging.new_message('spysydriveStateSP')
    sol.spysydriveStateSP.aol.available = available
    sol.spysydriveStateSP.aol.active = active
    stock = messaging.new_message('selfdriveState')
    stock.selfdriveState.active = stock_active
    self.data['spysydriveStateSP'] = sol.spysydriveStateSP.as_reader()
    self.data['selfdriveState'] = stock.selfdriveState.as_reader()
    self.alive = {'spysydriveStateSP': fresh}
    self.valid = {'spysydriveStateSP': valid}


@pytest.mark.parametrize('fresh,valid', [(False, True), (True, False), (False, False)])
@pytest.mark.parametrize('stock_active', [False, True])
def test_stale_sol_cannot_keep_steering_or_fall_back_to_stock(fresh, valid, stock_active):
  sm = SubMaster(fresh=fresh, valid=valid, stock_active=stock_active)
  assert not ControlsExt().get_lat_active(sm)


def test_fresh_sol_decouples_lateral_and_stock_fallback():
  assert ControlsExt().get_lat_active(SubMaster())
  assert not ControlsExt().get_lat_active(SubMaster(active=False, stock_active=True))
  assert ControlsExt().get_lat_active(SubMaster(available=False, stock_active=True))
