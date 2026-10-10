import importlib.util

import pytest

from openpilot.cereal import custom


def test_sol_independent_lateral_activation():
  assert importlib.util.find_spec('openpilot.spysypilot') is not None, 'SOL state machine is not imported'
  from openpilot.spysypilot.aol.state import AolStateMachine
  sm = AolStateMachine(0.01)
  assert sm.update() == (False, False)
  sm.add_event('lkasEnable')
  assert sm.update() == (True, True)
  assert sm.state == custom.AolState.AolStateEnum.enabled
  sm.clear_events()
  sm.add_event('userDisable')
  assert sm.update() == (False, False)


@pytest.mark.parametrize('initial', ['disabled', 'enabled', 'overriding', 'softDisabling', 'paused'])
@pytest.mark.parametrize('event', ['canPause', 'softDisable', 'immediateDisable', 'userDisable'])
def test_fault_and_pause_take_priority_over_override_and_enable(initial, event):
  from openpilot.spysypilot.aol.state import AolStateMachine, State
  sm = AolStateMachine(0.01)
  sm.state = getattr(State, initial)
  sm.soft_disable_timer = 1.0
  for e in ('overrideLateral', 'lkasEnable', event):
    sm.add_event(e)
  enabled, active = sm.update()
  if event in ('immediateDisable', 'userDisable'):
    assert not enabled and not active
  elif event == 'canPause':
    assert enabled and not active
    assert sm.state == State.paused
  elif initial in ('disabled', 'paused'):
    assert not active  # a fault must not allow engagement or resume
  else:
    assert sm.state == State.softDisabling
    assert sm.soft_disable_timer <= (1.0 if initial == 'softDisabling' else 3.0)
