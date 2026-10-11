from types import SimpleNamespace

import pytest

import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveD
from openpilot.selfdrive.selfdrived.tests.test_conditional_experimental_mode import selfdrived


@pytest.mark.parametrize('initialized,passive', [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize('manual', [False, True])
def test_cem_runs_after_sol_and_resets_even_when_sol_update_is_gated(initialized, passive, manual):
  sd = selfdrived(manual, False, enabled=False)
  sd.CP.passive = passive
  sd.initialized = initialized
  sd.events = []
  sd.active = False
  cs = messaging.new_message('carState').carState
  calls = []

  def conditional(*args, **kwargs):
    calls.append('cem')
    return kwargs['controls_enabled']

  sd.conditional_experimental_mode.update = conditional
  sd.state_machine = SimpleNamespace(update=lambda _: (True, True))
  sd.data_sample = lambda: cs
  sd.update_events = lambda _: None
  sd.update_experimental_mode = lambda state: SelfdriveD.update_experimental_mode(sd, state)
  sd.update_alerts = lambda _: calls.append('alerts')
  sd.publish_selfdriveState = lambda _: calls.append(('state', sd.experimental_mode))
  sd.pm = object()
  sd.aol = SimpleNamespace(update_events=lambda _: calls.append('sol-events'), update=lambda _: calls.append('sol-update'),
                           publish=lambda _: calls.append('sol-publish'))
  SelfdriveD.step(sd)
  sol_active = initialized and not passive
  expected = ['sol-events', 'sol-update'] if sol_active else []
  expected += ['cem', 'alerts', ('state', manual or sol_active), 'sol-publish']
  assert calls == expected
  assert sd.CS_prev is cs
