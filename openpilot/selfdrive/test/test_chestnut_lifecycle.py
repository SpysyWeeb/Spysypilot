"""Host-only lifecycle tests through the real manager launcher and process boundary."""
import copy
import json
import multiprocessing
import os
from pathlib import Path
import signal
import threading
import time
from types import SimpleNamespace

import pytest
import numpy as np
from tinygrad import Context, Tensor
from tinygrad.codegen import to_program_cache
from tinygrad.engine import worker
from tinygrad.engine.realize import run_linear
from tinygrad.helpers import PARALLEL
from tinygrad.uop.ops import Ops

from opendbc.car.structs import car
from openpilot.common.params import Params, ParamKeyFlag, ParamKeyType
from openpilot.selfdrive.modeld import modeld
from openpilot.system.manager import manager
from openpilot.system.manager.process import PythonProcess, ensure_running
from openpilot.system.manager.process_config import managed_processes, only_onroad

pytestmark = pytest.mark.skipif(multiprocessing.get_start_method() != 'fork', reason='manager uses inherited fork fixtures')


@pytest.mark.parametrize('lat_active', [False, True])
@pytest.mark.parametrize('speed', [0., 0.3, 0.31, 5.])
@pytest.mark.parametrize('curvature', [-0.027, 0.027])
def test_main_action_tracks_inactive_curvature_without_changing_active_or_moving(lifecycle, monkeypatch, lat_active, speed, curvature):
  lifecycle.scenario.stall = None
  monkeypatch.setattr(modeld, 'BIG_MODEL_TIMEOUT', 2)
  subscribed, actions = [], []

  class SubMaster(dict):
    updated = {'extrinsicsCalibration': False}

    def __init__(self, services):
      super().__init__(driverMonitoringState=SimpleNamespace(isRHD=False), narrowRoadCameraState=SimpleNamespace(frameId=1),
                       carState=SimpleNamespace(vEgo=speed), lateralDelay=SimpleNamespace(lateralDelay=0),
                       carControl=SimpleNamespace(latActive=lat_active), controlsState=SimpleNamespace(curvature=curvature))
      subscribed.extend(services)

    def update(self, *_args, **_kwargs):
      pass

  class FrameCaptured(Exception):
    pass

  def capture(_msg, _model_output, action, *_args):
    actions.append(action)
    raise FrameCaptured

  monkeypatch.setattr(modeld, 'SubMaster', SubMaster)
  monkeypatch.setattr(modeld.ModelState, 'run', lambda *_: {'action': np.array([[0.02, -0.1]])})
  monkeypatch.setattr(modeld, 'fill_model_msg', capture)
  with pytest.raises(FrameCaptured):
    modeld.main()
  assert len(actions) == 1
  expected = (curvature if not lat_active else 0.) if speed <= 0.3 else 0.02 / max(1., speed) ** 2
  assert actions[0].desiredCurvature == pytest.approx(expected)
  assert 'controlsState' in subscribed, 'the seed must have a live messaging subscription'


def test_model_replay_receives_curvature_seed():
  import ast
  import inspect
  from openpilot.selfdrive.test.process_replay import model_replay, process_replay

  config = next(c for c in process_replay.CONFIGS if c.proc_name == 'modeld')
  assert 'controlsState' in config.pubs
  tree = ast.parse(inspect.getsource(model_replay.model_replay))
  trims = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and
           isinstance(n.value.func, ast.Name) and n.value.func.id == 'trim_logs' and any(
             isinstance(t, ast.Name) and t.id == 'modeld_logs' for t in n.targets)]
  assert len(trims) == 1
  assert 'controlsState' in ast.literal_eval(trims[0].args[-1])


@pytest.mark.parametrize('configured, exitcode, restart', [(True, os.EX_TEMPFAIL, True), (True, 1, False), (True, 0, False),
                                                          (True, -signal.SIGKILL, False), (False, os.EX_TEMPFAIL, False)])
def test_manager_restarts_only_explicit_modeld_timeout(tmp_path, monkeypatch, configured, exitcode, restart):
  params = Params(str(tmp_path / 'params'))
  cp = car.CarParams.new_message()

  def exit_modeld():
    if exitcode < 0:
      os.kill(os.getpid(), -exitcode)
    os._exit(exitcode)

  monkeypatch.setattr(modeld, 'main', exit_modeld)
  proc = (copy.copy(managed_processes['modeld']) if configured else
          PythonProcess('test_modeld', 'openpilot.selfdrive.modeld.modeld', only_onroad))
  procs = {proc.name: proc}
  try:
    ensure_running(procs.values(), True, params, cp)
    first = proc.proc
    first.join(10)
    assert first.exitcode == exitcode
    ensure_running(procs.values(), True, params, cp)
    assert (proc.proc is not first) is restart
    if restart:
      assert proc.proc.pid != first.pid
      proc.proc.join(10)
      assert proc.proc.exitcode == exitcode
    else:
      for _ in range(3):
        ensure_running(procs.values(), True, params, cp)
      assert proc.proc is first
    # Ordinary offroad cleanup remains the reset boundary for other exit codes.
    ensure_running(procs.values(), False, params, cp)
    assert proc.proc is None
  finally:
    proc.stop(sig=signal.SIGKILL)


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
  params_path = str(tmp_path / 'params')
  params = Params(params_path)
  cp = car.CarParams.new_message()
  params.put('CarParams', cp.to_bytes(), block=True)
  trace = tmp_path / 'trace.txt'
  blocked, ran = (multiprocessing.Event() for _ in range(2))
  # A killed multiprocessing.Condition waiter can deadlock Event.set(); a pipe has no userspace waiter handshake.
  release_read, release_write = os.pipe()
  release = SimpleNamespace(set=lambda: os.write(release_write, b'1'))
  scenario = SimpleNamespace(stall='load', compile=None)

  def record(event):
    with trace.open('a') as f:
      f.write(f'{os.getpid()} {event}\n')

  def stall(where):
    if scenario.stall == where:
      blocked.set()
      os.read(release_read, 1)
      record(f'late_{where}')

  class Vision:
    width, height, buffer_len = 4, 2, 16
    frame_id, timestamp_sof, timestamp_eof = 1, 1_000_000_000, 1_000_000_010

    def __init__(self, *_):
      pass

    @staticmethod
    def available_streams(*_, **__):
      return [modeld.VisionStreamType.VISION_STREAM_NARROW_ROAD]

    def connect(self, *_):
      return True

    def recv(self):
      return object()

  class Model:
    vision_input_names = ('img', 'big_img')

    def __init__(self, _width, _height, chestnut):
      self.chestnut = chestnut
      record('big' if chestnut else 'small')
      if scenario.compile is not None:
        scenario.compile('load' if chestnut else 'small')
      if chestnut:
        stall('load')

    def warmup(self):
      record('warm_big' if self.chestnut else 'warm_small')
      if scenario.compile is not None:
        scenario.compile('warmup' if self.chestnut else 'warm_small')
      if self.chestnut:
        stall('warmup')

    def run(self, *_):
      record('run_big' if self.chestnut else 'run_small')
      ran.set()
      # Keep modeld alive after reaching actual main-loop model selection/invocation.
      threading.Event().wait()

  class SubMaster(dict):
    updated = {'extrinsicsCalibration': False}

    def __init__(self, *_):
      super().__init__(driverMonitoringState=SimpleNamespace(isRHD=False), narrowRoadCameraState=SimpleNamespace(frameId=1),
                       carState=SimpleNamespace(vEgo=0), lateralDelay=SimpleNamespace(lateralDelay=0))

    def update(self, *_, **__):
      pass

  monkeypatch.setattr(modeld, 'Params', lambda: Params(params_path))
  monkeypatch.setattr(modeld, 'VisionIpcClient', Vision)
  monkeypatch.setattr(modeld, 'ModelState', Model)
  monkeypatch.setattr(modeld, 'SubMaster', SubMaster)
  monkeypatch.setattr(modeld, 'PubMaster', lambda *_: SimpleNamespace(send=lambda *_: None))
  monkeypatch.setattr(modeld, 'BIG_MODEL_TIMEOUT', 0.05)
  monkeypatch.setattr(modeld, 'chestnut_compiled', lambda: True)
  monkeypatch.setattr(modeld, 'chestnut_present', lambda: True)
  monkeypatch.setattr(modeld, 'cable_connected', lambda: True)
  monkeypatch.setattr(modeld, 'wait_for_chestnut', lambda: record('enumerate'))
  monkeypatch.setattr(modeld, 'link_up', lambda: True)
  monkeypatch.setattr(modeld, 'config_realtime_process', lambda *_: record('realtime'))
  proc = copy.copy(managed_processes['modeld'])
  procs = {'modeld': proc}

  def tick(started=True):
    ensure_running(procs.values(), started, params, cp)

  def events(pid):
    return [line.split()[1] for line in trace.read_text().splitlines() if line.split()[0] == str(pid)]

  yield SimpleNamespace(params=params, proc=proc, tick=tick, events=events, blocked=blocked, release=release, ran=ran,
                        scenario=scenario, params_path=params_path)
  proc.stop(sig=signal.SIGKILL)
  os.close(release_read)
  os.close(release_write)


@pytest.mark.parametrize('stall_at', ['load', 'warmup'])
def test_timeout_kills_loader_before_small_and_restarts_small_only(lifecycle, stall_at):
  life = lifecycle
  life.scenario.stall = stall_at
  life.tick()
  first = life.proc.proc
  assert life.blocked.wait(10)
  first.join(3)
  print(f'timeout pid={first.pid}, exitcode={first.exitcode}, events={life.events(first.pid)}')
  assert first.exitcode == os.EX_TEMPFAIL
  assert not first.is_alive()
  assert 'small' not in life.events(first.pid)
  assert 'realtime' not in life.events(first.pid)
  assert life.params.get_bool('ChestnutLoadTimedOut')
  assert life.params.get('ChestnutActive') is False
  assert life.params.get('ChestnutLoading') is False

  life.release.set()
  before = life.events(first.pid)
  for _ in range(2):
    life.ran.clear()
    life.tick()
    recovered = life.proc.proc
    assert recovered.pid != first.pid
    assert life.ran.wait(10), 'restarted process must run the small model, not merely construct it'
    assert recovered.is_alive()
    assert life.events(recovered.pid) == ['small', 'realtime', 'run_small']
    assert life.events(first.pid) == before, 'no late work may survive in the timed-out process'
    assert life.params.get_bool('ChestnutLoadTimedOut')
    assert life.params.get('ChestnutActive') is False
    assert life.params.get('ChestnutLoading') is False
    print(f'recovered pid={recovered.pid}, events={life.events(recovered.pid)}')
    # Explicit manager stop/start in this session must not retry the big model.
    life.proc.stop(sig=signal.SIGKILL)
    assert not recovered.is_alive()
    assert life.proc.proc is None


@pytest.mark.parametrize('outcome', ['ready', 'load_error', 'warmup_error', 'timeout_load', 'timeout_warmup'])
def test_loader_compilation_is_serial_and_restores_caller_policy(lifecycle, monkeypatch, tmp_path, outcome):
  life = lifecycle
  reports = tmp_path / 'compilation.jsonl'
  life.scenario.stall = outcome.removeprefix('timeout_') if outcome.startswith('timeout_') else None
  monkeypatch.setattr(modeld, 'BIG_MODEL_TIMEOUT', 5)
  # A previous host test may have a pool; a newly launched modeld must not inherit that test-owned object.
  monkeypatch.setattr(worker, 'worker_pool', None)

  def compile_model_phase(phase):
    report = {'pid': os.getpid(), 'phase': phase, 'parallel': PARALLEL.value}
    if phase in ('load', 'warmup'):
      # Both constructor lower_and_compile calls and first-run dynamic kernels use this real backend path.
      # No inner PARALLEL override: only the caller baseline and production loader policy may set it.
      x = Tensor(np.arange(16, dtype=np.float32), device='CPU')
      add, mul = (x + 11).contiguous(), (x * 13).contiguous()
      to_program_cache.clear()
      if phase == 'load':
        compiled = modeld.lower_and_compile(add.schedule_linear(mul))
        programs = [c.body for c in compiled.src if c.body.op is Ops.PROGRAM]
        run_linear(compiled)
      else:
        # First-run warmup enters lower_and_compile dynamically through Tensor.realize/run_linear.
        Tensor.realize(add, mul)
        programs = list(to_program_cache.values())
      assert len(programs) == 2 and all(p.src[-1].op is Ops.BINARY for p in programs)
      np.testing.assert_array_equal(add.numpy(), np.arange(16, dtype=np.float32) + 11)
      np.testing.assert_array_equal(mul.numpy(), np.arange(16, dtype=np.float32) * 13)
      pool = worker.worker_pool
      report.update(kernels=len(programs), pool=pool is not None, children=[p.pid for p in multiprocessing.active_children()])
      assert worker.get_worker_pool() is pool
      # Keep RED safe: record the real pool before closing only our own workers, before any forced timeout.
      if pool is not None:
        worker.terminate_worker_pool()
        pool.join()
      report['children_after_cleanup'] = [p.pid for p in multiprocessing.active_children()]
      report['worker_states_after_cleanup'] = {pid: Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
                                               for pid in report['children'] if Path(f'/proc/{pid}/stat').exists()}
    with reports.open('a') as f:
      f.write(json.dumps(report) + '\n')
    if outcome == f'{phase}_error':
      raise RuntimeError(f'test {phase} failure after real compilation')

  life.scenario.compile = compile_model_phase
  # Bound the RED pool to two, outside ModelState; production must override it through construction AND warmup.
  with Context(PARALLEL=2, DEV='CPU:LLVM', CACHELEVEL=0):
    life.tick()
    first = life.proc.proc
    if outcome.startswith('timeout_'):
      assert life.blocked.wait(10)
      first.join(10)
      assert first.exitcode == os.EX_TEMPFAIL
      assert not first.is_alive()
      assert 'small' not in life.events(first.pid)
      assert 'realtime' not in life.events(first.pid)
      assert life.params.get_bool('ChestnutLoadTimedOut')
      assert life.params.get('ChestnutActive') is False
      assert life.params.get('ChestnutLoading') is False
      before = life.events(first.pid)
      life.release.set()
      life.tick()
      recovered = life.proc.proc
      assert recovered.pid != first.pid
      assert life.ran.wait(10)
      assert life.events(recovered.pid) == ['small', 'realtime', 'run_small']
      assert life.events(first.pid) == before
      assert recovered.is_alive()
    else:
      assert life.ran.wait(10)
      assert first.is_alive()
      assert ('run_big' if outcome == 'ready' else 'run_small') in life.events(first.pid)
      assert not life.params.get_bool('ChestnutLoadTimedOut')
    assert PARALLEL == 2

  observed = [json.loads(line) for line in reports.read_text().splitlines()]
  print(f'compilation outcome={outcome}: {observed}')
  big = [r for r in observed if r['phase'] in ('load', 'warmup')]
  assert [r['phase'] for r in big] == (['load'] if outcome in ('load_error', 'timeout_load') else ['load', 'warmup'])
  assert all(r['kernels'] == 2 and not r['children_after_cleanup'] and not r['worker_states_after_cleanup'] for r in big)
  assert all(not r['pool'] and not r['children'] for r in big), 'timeout-governed compilation must never create child workers'
  assert all(r['parallel'] == 0 for r in big)
  small = [r for r in observed if r['phase'] in ('small', 'warm_small')]
  assert small and all(r['parallel'] == 2 for r in small), 'restore caller policy before small-model continuation'


@pytest.mark.parametrize('failure', ['write_error', 'dropped_write', 'read_error'])
def test_timeout_marker_failure_does_not_request_restart(lifecycle, monkeypatch, failure):
  life = lifecycle
  synchronous_write = multiprocessing.Event()

  class FailedParams(Params):
    wrote_timeout = False

    def put_bool(self, key, val, block=False):
      if key == 'ChestnutLoadTimedOut':
        self.wrote_timeout = True
        if block:
          synchronous_write.set()
        if failure == 'write_error':
          raise OSError('test disk failure')
        if failure == 'dropped_write':
          return
      super().put_bool(key, val, block=block)

    def get_bool(self, key, block=False):
      if key == 'ChestnutLoadTimedOut' and self.wrote_timeout and failure == 'read_error':
        raise OSError('test readback failure')
      return super().get_bool(key, block=block)

  monkeypatch.setattr(modeld, 'Params', lambda: FailedParams(life.params_path))
  life.tick()
  first = life.proc.proc
  assert life.blocked.wait(10)
  first.join(3)
  assert first.exitcode == 1
  assert synchronous_write.is_set()
  assert 'small' not in life.events(first.pid)
  assert 'realtime' not in life.events(first.pid)
  before = life.events(first.pid)
  life.release.set()
  for _ in range(3):
    life.tick()
  assert life.proc.proc is first
  assert life.events(first.pid) == before


def test_manager_start_clears_timeout_latch_and_ready_model_runs(lifecycle, monkeypatch, tmp_path):
  life = lifecycle
  life.params.put_bool('ChestnutLoadTimedOut', True, block=True)
  assert life.params.get_type('ChestnutLoadTimedOut') == ParamKeyType.BOOL
  for flag in (ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_IGNITION_ON):
    life.params.clear_all(flag)
    assert life.params.get_bool('ChestnutLoadTimedOut')

  class ResetComplete(Exception):
    pass

  def stop_before_hardware():
    raise ResetComplete

  # Exercise real manager_init clearing/default initialization, stopping before hardware/registration/network.
  monkeypatch.setattr(manager, 'Params', lambda: Params(life.params_path))
  monkeypatch.setattr(manager, 'save_bootlog', lambda: None)
  monkeypatch.setattr(manager, 'get_build_metadata', lambda: SimpleNamespace(release_channel=False))
  monkeypatch.setattr(manager, 'Paths', SimpleNamespace(shm_path=lambda: str(tmp_path / 'shm')))
  monkeypatch.setattr(manager, 'HARDWARE', SimpleNamespace(get_serial=stop_before_hardware))
  with pytest.raises(ResetComplete):
    manager.manager_init()
  assert life.params.get('ChestnutLoadTimedOut') is None
  # Manager init clears CarParams too; camera/car boundaries are test-owned.
  life.params.put('CarParams', car.CarParams.new_message().to_bytes(), block=True)
  life.scenario.stall = None
  life.tick()
  current = life.proc.proc
  assert life.ran.wait(10)
  assert life.events(current.pid) == ['enumerate', 'enumerate', 'big', 'warm_big', 'small', 'warm_small', 'realtime', 'run_big']
  assert life.params.get('ChestnutLoadTimedOut') is None
  deadline = time.monotonic() + 3
  while life.params.get('ChestnutLoading') is not False and time.monotonic() < deadline:
    time.sleep(0.01)
  assert life.params.get('ChestnutLoading') is False
  assert life.params.get('ChestnutActive') is True
  for _ in range(3):
    life.tick()
  assert life.proc.proc is current
  assert current.is_alive()
  configured = [name for name, proc in managed_processes.items() if getattr(proc, 'restart_on_exit_code', None) is not None]
  assert configured == ['modeld']
  print(f'manager_init reset latch; ready pid={current.pid}, events={life.events(current.pid)}')


@pytest.mark.parametrize('retried', [False, True])
def test_failed_big_load_retries_once_through_onroad_cycle(lifecycle, monkeypatch, tmp_path, retried):
  life = lifecycle
  life.scenario.stall = None
  monkeypatch.setattr(modeld, 'BIG_MODEL_TIMEOUT', 5)
  if retried:
    life.params.put_bool('ChestnutRetry', True, block=True)
  assert life.params.get_type('ChestnutRetry') == ParamKeyType.BOOL

  def fail_big_load(phase):
    if phase == 'load':
      raise RuntimeError('test big model load failure')
  life.scenario.compile = fail_big_load

  def drop_chestnut():
    with (tmp_path / 'trace.txt').open('a') as f:
      f.write(f'{os.getpid()} drop\n')
  monkeypatch.setattr(modeld, 'drop_chestnut', drop_chestnut)

  life.tick()
  current = life.proc.proc
  assert life.ran.wait(10)
  # Both boots drop pending AMD work before the small model is built, then run it.
  assert life.events(current.pid) == ['enumerate', 'enumerate', 'big', 'drop', 'small', 'realtime', 'run_small']
  deadline = time.monotonic() + 3
  while life.params.get('ChestnutLoading') is not False and time.monotonic() < deadline:
    time.sleep(0.01)
  assert life.params.get('ChestnutLoading') is False
  assert life.params.get_bool('ChestnutRetry')
  assert not life.params.get_bool('ChestnutLoadTimedOut')
  if not retried:
    # First failure: request one onroad cycle and leave ChestnutActive unset, so no failure alert fires before the retry.
    assert life.params.get_bool('OnroadCycleRequested')
    assert life.params.get('ChestnutActive') is None
  else:
    # Failure on the retry boot: report the failure and never request another cycle.
    assert life.params.get('ChestnutActive') is False
    assert not life.params.get_bool('OnroadCycleRequested')
  assert current.is_alive()
