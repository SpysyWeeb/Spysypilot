import inspect
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import openpilot.cereal.messaging as messaging
from opendbc.car.hyundai.values import CAR as HyundaiCAR, HyundaiFlags
from opendbc.car.structs import car
from opendbc.car.toyota.values import CAR as ToyotaCAR
from openpilot.common.params import Params
from openpilot.selfdrive.controls.tests.test_conditional_experimental_mode import model, stop_model
from openpilot.selfdrive.selfdrived import selfdrived as selfdrived_module
from openpilot.selfdrive.selfdrived.selfdrived import ButtonType, EventName, SelfdriveD


@pytest.fixture
def selfdrived_factory(monkeypatch, tmp_path):
  # Keep the real constructor, SubMaster messages, Params, events and mode owners.
  # Only IPC and daemon/platform setup are replaced; no publishers/devices are needed.
  params = Params(str(tmp_path))
  monkeypatch.setattr(selfdrived_module, 'Params', lambda: params)
  monkeypatch.setattr(selfdrived_module, 'get_build_metadata', lambda: SimpleNamespace(
    openpilot=SimpleNamespace(comma_remote=False), tested_channel=False))
  monkeypatch.setattr(selfdrived_module, 'HARDWARE', SimpleNamespace(get_device_type=lambda: 'pc'))
  monkeypatch.setattr(selfdrived_module, 'set_offroad_alert', Mock())
  monkeypatch.setattr(selfdrived_module, 'Ratekeeper', lambda *args, **kwargs: SimpleNamespace(lagging=False))
  monkeypatch.setattr(messaging, 'PubMaster', Mock())
  monkeypatch.setattr(messaging, 'sub_sock', Mock())

  def make(*, manual=False, long_control=True, initialized=True, passive=False, brand='mock', flags=0, car_fingerprint=''):
    params.put_bool('ExperimentalMode', manual, block=True)
    params.put('LongitudinalPersonality', 1, block=True)
    cp = car.CarParams.new_message(brand=brand, flags=flags, carFingerprint=car_fingerprint,
                                  openpilotLongitudinalControl=long_control, passive=passive)
    sd = SelfdriveD(CP=cp)
    sd.initialized = initialized
    sd.enabled = True
    for health in (sd.sm.alive, sd.sm.valid, sd.sm.freq_ok):
      health.update(dict.fromkeys(sd.sm.services, True))
    # Spies still persist to the real temporary Params store.
    monkeypatch.setattr(params, 'put_bool', Mock(wraps=params.put_bool))
    monkeypatch.setattr(params, 'put', Mock(wraps=params.put))
    return sd

  return make


def tick(sd, pressed=None, *, can_valid=True):
  cs = car.CarState.new_message(canValid=can_valid, vEgo=10.0)
  if pressed is not None:
    cs.buttonEvents = [car.CarState.ButtonEvent.new_message(type=ButtonType.gapAdjustCruise, pressed=pressed)]
  sd.sm.frame += 1
  sd.sm.updated['modelV2'] = sd.sm.frame % 5 == 0
  sd.update_events(cs)
  sd.update_experimental_mode(cs)
  sd.CS_prev = cs
  return cs


def assert_persisted(sd, key, expected):
  # Runtime writes are queued; read a separate handle rather than assuming a flush.
  params = Params(sd.params.d)
  deadline = time.monotonic() + 5.0
  while params.get(key) != expected and time.monotonic() < deadline:
    time.sleep(0.001)
  assert params.get(key) == expected


def poll_params(sd, monkeypatch):
  # One complete real config-thread iteration, without a wall-clock sleep.
  stop = threading.Event()
  with monkeypatch.context() as patcher:
    patcher.setattr(selfdrived_module.time, 'sleep', lambda _: stop.set())
    sd.params_thread(stop)


@pytest.mark.parametrize('manual', [False, True])
def test_read_started_before_hold_cannot_restore_old_manual_mode(selfdrived_factory, monkeypatch, manual):
  sd = selfdrived_factory(manual=manual)
  read_started = threading.Event()
  finish_read = threading.Event()
  original_get_bool = sd.params.get_bool

  def delayed_get_bool(key, block=False):
    value = original_get_bool(key, block)
    if key == 'ExperimentalMode':
      read_started.set()
      assert finish_read.wait(5), 'stale read was not released'
    return value

  monkeypatch.setattr(sd.params, 'get_bool', delayed_get_bool)
  with ThreadPoolExecutor(max_workers=1) as pool:
    reader = pool.submit(poll_params, sd, monkeypatch)
    try:
      assert read_started.wait(5), 'Params reader did not reach the barrier'
      tick(sd, True)
      for _ in range(49):
        tick(sd)
      assert sd.manual_experimental_mode is not manual
      assert sd.experimental_mode is not manual
    finally:
      finish_read.set()
    reader.result(timeout=5)

  # Complete the stale poll before resolving the next effective mode.
  tick(sd)
  assert sd.manual_experimental_mode is not manual
  assert sd.experimental_mode is not manual


@pytest.mark.parametrize('manual', [False, True])
def test_poll_cannot_undo_toggle_while_persistence_is_pending(selfdrived_factory, monkeypatch, manual):
  sd = selfdrived_factory(manual=manual)
  original_put_bool = sd.params.put_bool
  main_thread = threading.get_ident()

  def delayed_put_bool(key, value, block=False):
    # Model a backed-up asynchronous Params queue: queued writes are not visible yet.
    if block:
      assert threading.get_ident() != main_thread, 'disk persistence ran on the main loop'
      original_put_bool(key, value, block=True)

  monkeypatch.setattr(sd.params, 'put_bool', delayed_put_bool)
  tick(sd, True)
  for _ in range(49):
    tick(sd)
  assert sd.manual_experimental_mode is not manual
  assert Params(sd.params.d).get_bool('ExperimentalMode') is manual

  with ThreadPoolExecutor(max_workers=1) as pool:
    pool.submit(poll_params, sd, monkeypatch).result(timeout=5)
  tick(sd)
  assert sd.manual_experimental_mode is not manual
  assert sd.experimental_mode is not manual
  assert Params(sd.params.d).get_bool('ExperimentalMode') is not manual


@pytest.mark.parametrize('manual', [False, True])
def test_poll_generation_check_and_assignment_are_serialized_with_hold(selfdrived_factory, monkeypatch, manual):
  sd = selfdrived_factory(manual=manual)
  assign_ready = threading.Event()
  finish_assignment = threading.Event()
  lines, first_line = inspect.getsourcelines(SelfdriveD.params_thread)
  assignment_line = first_line + next(i for i, line in enumerate(lines) if 'self.manual_experimental_mode = manual_mode' in line)

  def pause_before_assignment(frame, event, arg):
    if event == 'line' and frame.f_code is SelfdriveD.params_thread.__code__ and frame.f_lineno == assignment_line:
      # The generation comparison has already succeeded. Pausing the read alone
      # would not catch a check-then-store race if the assignment lost its lock.
      assign_ready.set()
      assert finish_assignment.wait(5), 'assignment barrier was not released'
    return pause_before_assignment

  def poll_at_barrier():
    sys.settrace(pause_before_assignment)
    try:
      poll_params(sd, monkeypatch)
    finally:
      sys.settrace(None)

  def hold():
    tick(sd, True)
    for _ in range(49):
      tick(sd)

  with ThreadPoolExecutor(max_workers=2) as pool:
    reader = pool.submit(poll_at_barrier)
    try:
      assert assign_ready.wait(5), 'poll did not reach the assignment barrier'
      assignment_locked = sd.manual_mode_lock.locked()
      button = pool.submit(hold)
      if not assignment_locked:
        # Expose the losing interleaving rather than racing the two workers.
        button.result(timeout=5)
    finally:
      finish_assignment.set()
    reader.result(timeout=5)
    button.result(timeout=5)
  tick(sd)
  assert sd.manual_experimental_mode is not manual
  assert sd.experimental_mode is not manual


@pytest.mark.parametrize('manual', [False, True])
@pytest.mark.parametrize('conditional', [False, True])
@pytest.mark.parametrize('newer_toggles', [1, 2])
def test_delayed_persistence_preserves_latest_hold_and_cem(selfdrived_factory, monkeypatch, manual, conditional, newer_toggles):
  sd = selfdrived_factory(manual=manual)
  sd.sm.data['modelV2'] = stop_model() if conditional else model()
  for _ in range(100):
    tick(sd)
  assert sd.conditional_experimental_mode.experimental_mode is conditional

  original_put_bool = sd.params.put_bool
  write_started = threading.Event()
  finish_write = threading.Event()
  main_thread = threading.get_ident()

  def delayed_put_bool(key, value, block=False):
    assert threading.get_ident() != main_thread, 'persistence ran on the main loop'
    assert block, 'a queued write is not an acknowledgment of persistence'
    write_started.set()
    assert finish_write.wait(5), 'writer was not released'
    original_put_bool(key, value, block=block)

  monkeypatch.setattr(sd.params, 'put_bool', delayed_put_bool)
  tick(sd, True)
  for _ in range(49):
    tick(sd)
  assert sd.manual_experimental_mode is not manual
  assert sd.experimental_mode is ((not manual) or conditional)

  with ThreadPoolExecutor(max_workers=1) as pool:
    writer = pool.submit(poll_params, sd, monkeypatch)
    try:
      assert write_started.wait(5), 'writer did not reach the persistence barrier'
      assert Params(sd.params.d).get_bool('ExperimentalMode') is manual
      # Main-loop frames and further complete holds must not wait for disk I/O.
      for _ in range(100):
        tick(sd)
        assert sd.manual_experimental_mode is not manual
        assert sd.experimental_mode is ((not manual) or conditional)
      latest = not manual
      for _ in range(newer_toggles):
        tick(sd, False)
        tick(sd, True)
        for _ in range(49):
          tick(sd)
        latest = not latest
        assert sd.manual_experimental_mode is latest
        assert sd.experimental_mode is (latest or conditional)
    finally:
      finish_write.set()
    writer.result(timeout=5)
    tick(sd)
    assert sd.manual_experimental_mode is latest
    assert sd.experimental_mode is (latest or conditional)
    # A stale write completion must not retire a newer request, even if values match.
    pool.submit(poll_params, sd, monkeypatch).result(timeout=5)

  assert Params(sd.params.d).get_bool('ExperimentalMode') is latest
  assert sd.manual_experimental_mode is latest
  assert original_put_bool.call_count == 2
  assert sd.personality == 1
  sd.params.put.assert_not_called()


@pytest.mark.parametrize('manual', [False, True])
@pytest.mark.parametrize('conditional', [False, True])
def test_ui_polling_resumes_after_hold_persistence(selfdrived_factory, monkeypatch, manual, conditional):
  sd = selfdrived_factory(manual=manual)
  sd.sm.data['modelV2'] = stop_model() if conditional else model()
  for _ in range(100):
    tick(sd)
  tick(sd, True)
  for _ in range(49):
    tick(sd)
  poll_params(sd, monkeypatch)
  assert sd.manual_experimental_mode is not manual
  tick(sd, False)

  # Separate Params handle models the existing UI, not the selfdrived writer.
  ui_params = Params(sd.params.d)
  for requested in (manual, not manual):
    ui_params.put_bool('ExperimentalMode', requested, block=True)
    poll_params(sd, monkeypatch)
    tick(sd)
    assert sd.manual_experimental_mode is requested
    assert sd.experimental_mode is (requested or conditional)
    assert sd.conditional_experimental_mode.experimental_mode is conditional
  sd.params.put_bool.assert_called_once_with('ExperimentalMode', not manual, block=True)


@pytest.mark.parametrize('held_frames', [1, 49])
def test_short_tap_cycles_personality_once(selfdrived_factory, held_frames):
  sd = selfdrived_factory()
  tick(sd, True)
  for _ in range(held_frames - 1):
    tick(sd)
    assert EventName.personalityChanged not in sd.events.names
  assert sd.personality == 1
  tick(sd, False)
  assert sd.personality == 0
  assert EventName.personalityChanged in sd.events.names
  sd.params.put.assert_called_once_with('LongitudinalPersonality', 0)
  assert_persisted(sd, 'LongitudinalPersonality', 0)
  for _ in range(100):
    tick(sd)
    assert EventName.personalityChanged not in sd.events.names
  sd.params.put_bool.assert_not_called()
  assert not Params(sd.params.d).get_bool('ExperimentalMode')
  assert sd.personality == 0


@pytest.mark.parametrize('manual', [False, True])
@pytest.mark.parametrize('flags', [HyundaiFlags.CANFD, HyundaiFlags.CANFD | HyundaiFlags.CANFD_ALT_BUTTONS])
def test_hyundai_canfd_stalled_release_cannot_toggle_mode(selfdrived_factory, monkeypatch, manual, flags):
  sd = selfdrived_factory(manual=manual, brand='hyundai', flags=int(flags), car_fingerprint=HyundaiCAR.HYUNDAI_IONIQ_5)
  # CAN-FD CRUISE_BUTTONS can stop transmitting for ~0.5s while canValid stays true.
  # The release of a physical short tap is not visible until the message resumes.
  tick(sd, True)
  for _ in range(50):
    tick(sd)
    assert sd.manual_experimental_mode is manual
    assert sd.experimental_mode is manual
  poll_params(sd, monkeypatch)
  sd.params.put_bool.assert_not_called()
  assert Params(sd.params.d).get_bool('ExperimentalMode') is manual
  assert sd.personality == 1

  tick(sd, False)
  assert sd.personality == 0
  assert EventName.personalityChanged in sd.events.names
  sd.params.put.assert_called_once_with('LongitudinalPersonality', 0)
  assert_persisted(sd, 'LongitudinalPersonality', 0)
  # Normal short releases retain the same behavior after the stalled release.
  tick(sd, True)
  tick(sd, False)
  assert sd.personality == 2
  assert EventName.personalityChanged in sd.events.names
  sd.params.put_bool.assert_not_called()


@pytest.mark.parametrize(('brand', 'flags', 'fingerprint'), [
  ('hyundai', int(HyundaiCAR.HYUNDAI_PALISADE.config.flags), HyundaiCAR.HYUNDAI_PALISADE),
  ('toyota', int(HyundaiFlags.CANFD), ToyotaCAR.TOYOTA_RAV4_TSS2),
])
def test_supported_hold_is_not_gated_by_brand_or_unrelated_flags(selfdrived_factory, monkeypatch, brand, flags, fingerprint):
  # Classic-CAN Hyundai (including Palisade) and other brands retain hold-to-mode.
  sd = selfdrived_factory(brand=brand, flags=flags, car_fingerprint=fingerprint)
  tick(sd, True)
  for _ in range(49):
    tick(sd)
  assert sd.manual_experimental_mode is True
  assert sd.experimental_mode is True
  poll_params(sd, monkeypatch)
  assert_persisted(sd, 'ExperimentalMode', True)
  tick(sd, False)
  assert sd.personality == 1
  sd.params.put.assert_not_called()
  tick(sd, True)
  tick(sd, False)
  assert sd.personality == 0
  assert EventName.personalityChanged in sd.events.names


@pytest.mark.parametrize('held_frames', [49, 50])
@pytest.mark.parametrize('invalid_edge', [None, True, False])
def test_invalid_can_cancels_hold_evidence_without_rearming_completed_hold(selfdrived_factory, held_frames, invalid_edge):
  sd = selfdrived_factory()
  tick(sd, True)
  for _ in range(held_frames - 1):
    tick(sd)
  toggled = held_frames == 50
  assert sd.manual_experimental_mode is toggled

  tick(sd, invalid_edge, can_valid=False)
  assert sd.manual_experimental_mode is toggled
  assert sd.personality == 1
  assert EventName.personalityChanged not in sd.events.names
  for _ in range(100):
    tick(sd)
    assert sd.manual_experimental_mode is toggled
  # A completed hold must still swallow its eventual valid release, not cycle too.
  tick(sd, False)
  assert sd.personality == (1 if toggled else 0)
  assert (EventName.personalityChanged in sd.events.names) is not toggled
  tick(sd, True)
  for _ in range(49):
    tick(sd)
  assert sd.manual_experimental_mode is not toggled


def test_long_press_toggles_once_without_personality_and_rearms_on_release(selfdrived_factory, monkeypatch):
  sd = selfdrived_factory()
  tick(sd, True)
  for _ in range(48):
    tick(sd)
  sd.params.put_bool.assert_not_called()
  assert not Params(sd.params.d).get_bool('ExperimentalMode')

  # The press frame counts as frame one; frame 50 still held is the threshold.
  tick(sd)
  assert sd.manual_experimental_mode is True
  sd.params.put_bool.assert_not_called()
  poll_params(sd, monkeypatch)
  sd.params.put_bool.assert_called_once_with('ExperimentalMode', True, block=True)
  assert_persisted(sd, 'ExperimentalMode', True)
  for _ in range(100):
    tick(sd)
  tick(sd, False)
  assert EventName.personalityChanged not in sd.events.names
  for _ in range(20):
    tick(sd)
  assert sd.personality == 1
  sd.params.put.assert_not_called()
  assert sd.params.put_bool.call_count == 1

  tick(sd, True)
  for _ in range(48):
    tick(sd)
  assert sd.params.put_bool.call_count == 1
  tick(sd)
  assert sd.manual_experimental_mode is False
  poll_params(sd, monkeypatch)
  assert sd.params.put_bool.call_count == 2
  sd.params.put_bool.assert_called_with('ExperimentalMode', False, block=True)
  assert_persisted(sd, 'ExperimentalMode', False)
  tick(sd, False)
  assert sd.personality == 1
  sd.params.put.assert_not_called()

  # Releasing a long press also re-arms the normal tap action.
  tick(sd, True)
  tick(sd, False)
  assert sd.personality == 0
  sd.params.put.assert_called_once_with('LongitudinalPersonality', 0)
  assert sd.params.put_bool.call_count == 2


@pytest.mark.parametrize('manual', [False, True])
@pytest.mark.parametrize('conditional', [False, True])
def test_hold_toggles_manual_preference_before_same_tick_cem_resolution(selfdrived_factory, monkeypatch, manual, conditional):
  sd = selfdrived_factory(manual=manual)
  sd.sm.data['modelV2'] = stop_model() if conditional else model()
  for _ in range(100):
    tick(sd)
  assert sd.conditional_experimental_mode.experimental_mode is conditional
  assert sd.experimental_mode is (manual or conditional)

  tick(sd, True)
  for _ in range(48):
    tick(sd)
  sd.params.put_bool.assert_not_called()
  assert sd.manual_experimental_mode is manual

  tick(sd)
  sd.params.put_bool.assert_not_called()
  # tick() has already run the real resolver, before the 10 Hz Params poll.
  assert sd.experimental_mode is ((not manual) or conditional)
  assert sd.manual_experimental_mode is not manual
  poll_params(sd, monkeypatch)
  sd.params.put_bool.assert_called_once_with('ExperimentalMode', not manual, block=True)
  assert_persisted(sd, 'ExperimentalMode', not manual)
  assert sd.conditional_experimental_mode.experimental_mode is conditional

  # The existing Params reader must retain the new manual preference, not CEM.
  poll_params(sd, monkeypatch)
  tick(sd)
  assert sd.manual_experimental_mode is not manual
  assert sd.experimental_mode is ((not manual) or conditional)

  for _ in range(100):
    tick(sd)
  tick(sd, False)
  assert sd.params.put_bool.call_count == 1
  sd.params.put.assert_not_called()
  assert sd.personality == 1
  assert EventName.personalityChanged not in sd.events.names


@pytest.mark.parametrize('manual', [False, True])
@pytest.mark.parametrize('gate', ['long_control', 'initialized', 'passive'])
def test_existing_gates_ignore_taps_and_holds(selfdrived_factory, manual, gate):
  sd = selfdrived_factory(manual=manual, **{gate: gate == 'passive'})
  persisted_before = Params(sd.params.d).get('ExperimentalMode')
  manual_before = sd.manual_experimental_mode
  tick(sd, True)
  for _ in range(100):
    tick(sd)
  tick(sd, False)
  tick(sd, True)
  tick(sd, False)
  assert sd.personality == 1
  sd.params.put.assert_not_called()
  sd.params.put_bool.assert_not_called()
  assert Params(sd.params.d).get('ExperimentalMode') is persisted_before
  assert sd.manual_experimental_mode is manual_before
  assert sd.experimental_mode is manual_before
  assert EventName.personalityChanged not in sd.events.names
