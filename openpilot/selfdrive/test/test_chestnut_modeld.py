import base64
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from tinygrad import Context, Tensor, TinyJit
from tinygrad_repo.examples.openpilot.helpers import allocate_inputs, dump_pickle, load_pickle

from openpilot.selfdrive.modeld import helpers, modeld


@pytest.mark.parametrize('size, expected', [(None, False), (134, False), (1 << 20, False), ((1 << 20) + 1, True)])
def test_chestnut_model_admission(tmp_path, monkeypatch, size, expected):
  monkeypatch.setattr(helpers, 'MODELS_DIR', tmp_path)
  if size is not None:
    with helpers.modeld_pkl_path(True).open('wb') as f:
      f.truncate(size)
  for size_name in ('1344x760', '1928x1208'):
    (tmp_path / f'big_driving_warp_{size_name}_tinygrad.pkl').touch()
  assert helpers.chestnut_compiled() is expected
  # The camera-device warps are built locally, not the big-device LFS warps.
  for path in tmp_path.glob('big_driving_warp*'):
    path.unlink()
  assert helpers.chestnut_compiled() is expected


def make_host_artifacts(path, model_device):
  # Small, test-owned artifacts. CPU:1 is a second host device, NOT a USB/QCOM GPU.
  warp_specs = {'input_frame': ((2, 16), 'uint8', 'CPU'), 'M_inv': ((2, 3, 3), 'float32', 'CPU')}

  @TinyJit
  def warp(input_frame, M_inv):
    return (input_frame[:, :4] + M_inv[:, 0, 2].reshape(2, 1).cast('uint8')).contiguous().realize()

  with Context(DEV='CPU:LLVM'):
    inputs = allocate_inputs(warp_specs)
    for _ in range(3):
      warp(**inputs)
    (path / 'driving_warp_4x2_tinygrad.pkl').write_bytes(pickle.dumps({'run': warp, 'input_specs': warp_specs}))

  input_specs = {'new_img': ((2, 4), 'uint8', model_device),
                 'desire': ((8,), 'float32', model_device), 'traffic_convention': ((2,), 'float32', model_device),
                 'action_t': ((2,), 'float32', model_device), 'history': ((1,), 'float32', model_device)}
  output_specs = {'outputs': ((1, 8), 'float32', model_device), 'next_history': ((1,), 'float32', model_device)}

  @TinyJit
  def run(output_buffers, new_img, desire, traffic_convention, action_t, history):
    values = Tensor.stack(new_img[0].float().sum(), new_img[1].float().sum(), desire.sum(), traffic_convention.sum(),
                          action_t[0], action_t[1], history[0], history[0] + 1).reshape(1, 8)
    Tensor.realize(output_buffers['outputs'].assign(values), output_buffers['next_history'].assign(history + 1))

  with Context(DEV='CPU:LLVM'):
    inputs, outputs = allocate_inputs(input_specs), allocate_inputs(output_specs)
    for _ in range(3):
      run(output_buffers=outputs, **inputs)
    artifact = {'run': run, 'input_specs': input_specs, 'output_specs': output_specs,
                'metadata': {'output_shapes': {'outputs': (1, 8), 'next_history': (1,)},
                             'metadata': {'output_slices': base64.b64encode(pickle.dumps({'probe': slice(0, 8)}))}}}
    dump_pickle(artifact, path / 'driving_tinygrad.pkl')


@pytest.mark.parametrize('model_device', ['CPU', 'CPU:1'])
def test_model_upload_run_and_warmup(tmp_path, monkeypatch, model_device):
  with Context(DEV='CPU:LLVM'):
    make_host_artifacts(tmp_path, model_device)
    monkeypatch.setattr(modeld, 'MODELS_DIR', tmp_path)
    monkeypatch.setattr(modeld, 'modeld_pkl_path', lambda _: tmp_path / 'driving_tinygrad.pkl')
    monkeypatch.setattr(modeld, 'load_oob', lambda path, _: load_pickle(path, out_of_band=True))
    monkeypatch.setattr(modeld, 'get_nv12_info', lambda *_: (4, 2, 2, 16))
    monkeypatch.setattr(modeld, 'Parser', lambda: SimpleNamespace(parse_outputs=lambda result: result))
    model = modeld.ModelState(4, 2, model_device != 'CPU')
    frames = {'img': np.arange(16, dtype=np.uint8), 'big_img': np.arange(16, 32, dtype=np.uint8)}
    transforms = {name: np.eye(3, dtype=np.float32) for name in frames}
    transforms['big_img'][0, 2] = 2
    pulse = np.zeros(8, dtype=np.float32)
    pulse[1] = 1
    inputs = {'desire_pulse': pulse, 'traffic_convention': np.array([0, 1], dtype=np.float32),
              'action_t': np.array([0.5, 1.5], dtype=np.float32)}
    called = []
    first = model.run(frames, transforms, inputs, lambda: called.append(True))['probe']
    np.testing.assert_array_equal(first, [[6, 78, 1, 1, 0.5, 1.5, 0, 1]])
    second = model.run(frames, transforms, inputs)['probe']
    np.testing.assert_array_equal(second, [[6, 78, 0, 1, 0.5, 1.5, 1, 2]])
    assert called == [True]
    assert model.state_pairs == {'history': 'next_history'}
    assert model.input_queues['history']._buffer().base is model.outputs['next_history']._buffer().base
    assert model.frame_input.device.device == 'CPU'
    assert model.model_input.device.device == model_device
    assert (model.frame_input is model.model_input) == (model_device == 'CPU')
    np.testing.assert_array_equal(model.frames, np.stack(list(frames.values())))
    for upload in (model.model_input, model.frame_input):
      assert upload.npy_size % 128 == 0
      for value in upload.views.values():
        assert value._buffer().offset % 128 == 0
    model.warmup()
    for upload in (model.model_input, model.frame_input):
      assert not upload.packed.any()
    assert not model.prev_desire.any()
    np.testing.assert_array_equal(model.input_queues['history'].numpy(), [0])
    np.testing.assert_array_equal(model.run(frames, transforms, inputs)['probe'], first)


@pytest.mark.parametrize('retry', [False, True], ids=['first_boot', 'retry_boot'])
@pytest.mark.parametrize('scenario', ['ready', 'cable_only', 'late_link', 'no_link', 'load_error', 'no_chestnut'])
def test_startup_enumeration_link_and_standby(monkeypatch, scenario, retry):
  calls, values = [], {'ChestnutRetry': True} if retry else {}
  clock = [0.0]

  class StartupDone(Exception):
    pass

  class Params:
    def get_bool(self, key):
      return values.get(key, False)

    def put_bool(self, key, value, block=False):
      values[key] = value

    def remove(self, key):
      values.pop(key, None)

  class Vision:
    width, height, buffer_len = 4, 2, 16

    def __init__(self, *_):
      pass

    @staticmethod
    def available_streams(*_, **__):
      return [modeld.VisionStreamType.VISION_STREAM_NARROW_ROAD]

    def connect(self, *_):
      return True

  class Model:
    def __init__(self, _w, _h, chestnut):
      self.chestnut = chestnut
      calls.append('big' if chestnut else 'small')
      if chestnut and scenario == 'load_error':
        raise RuntimeError('test load failure')

    def warmup(self):
      calls.append('warm_big' if self.chestnut else 'warm_small')

  class Thread:
    def __init__(self, target, daemon):
      assert daemon
      self.target = target

    def start(self):
      calls.append('loader')
      self.target()

    def join(self, timeout):
      assert timeout == modeld.BIG_MODEL_TIMEOUT

    def is_alive(self):
      return False

  def enumeration():
    calls.append('enumerate')
    if scenario == 'cable_only':
      clock[0] += 10.1
      raise TimeoutError('test no device')

  def link():
    calls.append('link')
    return scenario != 'no_link' and (scenario != 'late_link' or clock[0] >= 2)

  def sleep(seconds):
    clock[0] += seconds
    assert clock[0] <= modeld.BIG_MODEL_TIMEOUT + 1

  def realtime(*_):
    assert values['ChestnutLoading'] is False
    raise StartupDone

  monkeypatch.setattr(modeld, 'Params', Params)
  monkeypatch.setattr(modeld, 'VisionIpcClient', Vision)
  monkeypatch.setattr(modeld, 'ModelState', Model)
  monkeypatch.setattr(modeld, 'threading', SimpleNamespace(Thread=Thread))
  monkeypatch.setattr(modeld, 'time', SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
  monkeypatch.setattr(modeld, 'gc', SimpleNamespace(disable=lambda: None))
  monkeypatch.setattr(modeld, 'cloudlog', SimpleNamespace(warning=lambda *_: None, error=lambda *_: None,
                                                        exception=lambda *_: None))
  monkeypatch.setattr(modeld, 'chestnut_compiled', lambda: scenario != 'no_chestnut')
  monkeypatch.setattr(modeld, 'chestnut_present', lambda: scenario != 'cable_only')
  monkeypatch.setattr(modeld, 'cable_connected', lambda: True)
  monkeypatch.setattr(modeld, 'wait_for_chestnut', enumeration)
  monkeypatch.setattr(modeld, 'link_up', link, raising=False)
  monkeypatch.setattr(modeld, 'config_realtime_process', realtime)
  with pytest.raises(StartupDone):
    modeld.main()

  success = scenario in ('ready', 'late_link')
  if scenario == 'no_chestnut':
    assert calls == ['small']
    assert 'ChestnutActive' not in values
    assert 'OnroadCycleRequested' not in values
    return
  assert calls[0] == 'enumerate'
  if success or retry:
    assert values['ChestnutActive'] is success
    assert values.get('ChestnutRetry', False) is retry
    assert 'OnroadCycleRequested' not in values
  else:
    # a first failure retries once through an onroad cycle; the retry boot decides ChestnutActive
    assert values['ChestnutRetry'] is True
    assert values['OnroadCycleRequested'] is True
    assert 'ChestnutActive' not in values
  if scenario == 'cable_only':
    assert calls == ['enumerate', 'small']
    assert clock[0] < 11
  elif scenario == 'no_link':
    assert 'loader' not in calls
    assert clock[0] == modeld.BIG_MODEL_TIMEOUT
  else:
    assert calls.index('link') < calls.index('loader')
    if scenario == 'late_link':
      assert calls.count('link') == 3
    if success:
      assert calls[-3:] == ['warm_big', 'small', 'warm_small']
    else:
      assert calls[-1] == 'small'
      assert 'warm_small' not in calls


def test_firmware_flash_starts_immediately_and_bounds_startup_hold(monkeypatch):
  import ast
  import inspect
  import threading
  from openpilot.common.hardware.usb import CHESTNUT_USB_IDS
  from openpilot.system.hardware import hardwared

  clock = [1.0]
  started, release = threading.Event(), threading.Event()
  chestnut = hardwared.Chestnut()
  usb = [dict(zip(('vendorId', 'productId'), CHESTNUT_USB_IDS[0], strict=True), product='old firmware')]

  def flash():
    started.set()
    assert release.wait(5), 'test must release its real worker'

  monkeypatch.setattr(chestnut, 'flash', flash)
  monkeypatch.setattr(hardwared, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
  try:
    chestnut.update(False, usb)
    assert chestnut.thread is None, 'never start firmware flashing onroad'
    chestnut.update(True, usb)
    assert started.wait(1), 'first offroad attempt must not wait for the retry interval'
    assert chestnut.thread is not None
    assert chestnut.attempts == 1
    chestnut.update(True, usb)
    assert chestnut.attempts == 1, 'do not launch a second live worker'

    # Execute the real hardware-thread admission assignment, not a test-owned copy of its predicate.
    tree = ast.parse(inspect.getsource(hardwared.hardware_thread))
    gates: list[ast.stmt] = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
      isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == 'startup_conditions' and
      isinstance(t.slice, ast.Constant) and t.slice.value == 'not_flashing_chestnut' for t in n.targets)]
    assert len(gates) == 1
    gate = compile(ast.Module(body=gates, type_ignores=[]), hardwared.__file__, 'exec')
    for elapsed, held in [(0., True), (29.999, True), (30., False), (31., False)]:
      clock[0] = chestnut.last_attempt + elapsed
      assert chestnut.flashing is held
      state = {'chestnut': chestnut, 'startup_conditions': {}}
      exec(gate, state)
      assert state['startup_conditions']['not_flashing_chestnut'] is not held
    assert chestnut.thread.is_alive(), 'timeout releases admission; it does not cancel the worker'
  finally:
    release.set()
    if chestnut.thread is not None:
      chestnut.thread.join(2)
      assert not chestnut.thread.is_alive()
  assert not chestnut.flashing
