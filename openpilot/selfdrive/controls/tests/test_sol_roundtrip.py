"""Host/cereal/pandad/firmware/safety round trip. Board I/O is emulated, not a device test."""
import ctypes
import importlib.util
import re
import subprocess
from pathlib import Path

import capnproto
import pytest

from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.safety.tests.common import CANPackerSafety
from opendbc.safety.tests.libsafety import libsafety_py
from openpilot.cereal import log
from openpilot.selfdrive.controls.tests.test_sol_driver import make_driver, step
from openpilot.spysypilot.aol.aol import AolDriver

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope='module')
def wire(tmp_path_factory):
  # Reuse the native test's exact firmware handler, watchdog, health flags and full
  # compiled safety implementation. Only hardware reads/writes are substituted.
  spec = importlib.util.spec_from_file_location('sol_native', ROOT / 'panda/tests/usbprotocol/test_aol.py')
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  native = module.native.__wrapped__(tmp_path_factory)
  native.safety_rx_hook.argtypes = [ctypes.c_void_p]
  native.safety_rx_hook.restype = ctypes.c_bool
  native.get_controls_allowed.restype = ctypes.c_bool

  pandad = (ROOT / 'openpilot/selfdrive/pandad/pandad.cc').read_text()
  panda = (ROOT / 'openpilot/selfdrive/pandad/panda.cc').read_text()
  heartbeat = re.search(r'void Panda::send_heartbeat\([^}]+}', panda).group()
  enabled = re.search(r'      engaged_aol = [^;]+;', pandad).group()
  fill = pandad[pandad.index('void fill_panda_state('):pandad.index('void fill_panda_can_state(')]
  source = r'''
#include <cstring>
#include <initializer_list>
#include <capnp/serialize.h>
#include "cereal/gen/cpp/log.capnp.h"
#include "panda/board/health.h"
struct Handle {
  uint16_t command = 0, param1 = 0, param2 = 0;
  void control_write(uint16_t c, uint16_t a, uint16_t b) { command=c; param1=a; param2=b; }
};
struct Panda { Handle *handle; void send_heartbeat(bool engaged, bool engaged_aol); };
''' + heartbeat + fill + r'''
struct Sm {
  cereal::Event::Reader event;
  bool fresh;
  bool allAliveAndValid(std::initializer_list<const char *>) { return fresh; }
  cereal::Event::Reader operator[](const char *) { return event; }
};
extern "C" uint32_t encode(const void *data, size_t size, bool fresh) {
  auto words = kj::heapArray<capnp::word>(size / sizeof(capnp::word));
  memcpy(words.begin(), data, size);
  capnp::FlatArrayMessageReader reader(words.asPtr());
  Sm sm{reader.getRoot<cereal::Event>(), fresh};
  bool engaged_aol = false;
''' + enabled + r'''
  Handle h; Panda p{&h}; p.send_heartbeat(false, engaged_aol);
  return h.command == 0xf3 ? (uint32_t(h.param1) << 16) | h.param2 : 0xffffffff;
}
extern "C" size_t decode(uint16_t flags, void *out) {
  capnp::MallocMessageBuilder msg;
  auto event = msg.initRoot<cereal::Event>();
  event.setValid(true);
  auto ps = event.initPandaStates(1)[0];
  health_t health{}; health.flags_pkt = flags;
  health.safety_mode_pkt = uint8_t(cereal::CarParams::SafetyModel::HYUNDAI);
  fill_panda_state(ps, cereal::PandaState::PandaType::RED_PANDA, health);
  auto words = capnp::messageToFlatArray(msg);
  auto bytes = words.asBytes();
  memcpy(out, bytes.begin(), bytes.size());
  return bytes.size();
}
'''
  directory = tmp_path_factory.mktemp('sol_cpp_wire')
  cpp, so = directory / 'wire.cc', directory / 'wire.so'
  cpp.write_text(source)
  generated = ROOT / 'openpilot/cereal/gen/cpp'
  subprocess.run(['c++', '-shared', '-fPIC', '-std=c++20', '-Werror', str(cpp),
                  *map(str, sorted(generated.glob('*.c++'))), '-o', str(so),
                  '-I', str(ROOT), '-I', str(ROOT / 'openpilot'), '-I', str(generated), '-I', capnproto.INCLUDE_DIR,
                  '-L', capnproto.LIB_DIR, '-Wl,-rpath,' + capnproto.LIB_DIR, '-lcapnp', '-lkj'], check=True)
  cpp_wire = ctypes.CDLL(str(so))
  cpp_wire.encode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_bool]
  cpp_wire.encode.restype = ctypes.c_uint32
  cpp_wire.decode.argtypes = [ctypes.c_uint16, ctypes.c_void_p]
  cpp_wire.decode.restype = ctypes.c_size_t
  return native, cpp_wire


def exchange(wire, driver, fresh=True):
  native, cpp = wire
  captured = []

  class Publisher:
    @staticmethod
    def send(topic, msg):
      assert topic == 'spysydriveStateSP'
      captured.append(msg.to_bytes())

  driver.publish(Publisher())
  data = captured[0]
  params = cpp.encode(data, len(data), fresh)
  assert params >> 16 == 0  # Independent steering must never engage longitudinal.
  native.receive(params >> 16, params & 0xffff)
  flags = native.health_flags(native.get_lateral_allowed(), (1 << 7) | (1 << 8))
  buf = ctypes.create_string_buffer(4096)
  size = cpp.decode(flags, buf)
  with log.Event.from_bytes(buf.raw[:size]) as event:
    state = event.pandaStates[0]
    assert state.nmiReset and state.hardfaultReset
    assert not state.controlsAllowed
    driver.sd.sm['pandaStates'] = [state.as_builder()]
  return params


def rx(native, packet):
  assert native.safety_rx_hook(int(libsafety_py.ffi.cast('uintptr_t', packet)))


@pytest.mark.parametrize('lfa', [False, True])
def test_two_button_cycles_and_stale_heartbeat_are_coupled(wire, lfa):
  native, _ = wire
  native.reset_test()
  native.set_safety_hooks(8, 0)
  native.set_alternative_experience(1024)
  driver, cs = make_driver()
  if lfa:
    driver.CP.flags |= int(HyundaiFlags.HAS_LDA_BUTTON)
    driver = AolDriver(driver.sd)
  packer = CANPackerSafety('hyundai_can_generated')
  counter = 0

  def button(pressed):
    nonlocal counter
    packet = (libsafety_py.make_CANPacket(0x391, 0, bytes([16 if pressed else 0]) + bytes(7)) if lfa else
              packer.make_can_msg_safety('CLU11', 0, {'CF_Clu_CruiseSwMain': int(pressed), 'CF_Clu_AliveCnt1': counter}))
    counter += 1
    rx(native, packet)
    cs.buttonEvents = [{'type': 'lkas' if lfa else 'mainCruise', 'pressed': pressed}]
    step(driver, cs)
    exchange(wire, driver)

  exchange(wire, driver)
  assert not native.get_lateral_allowed()
  button(True)
  button(False)
  assert driver.active and native.get_lateral_allowed()
  assert not native.get_controls_allowed()
  button(True)
  button(False)
  assert not driver.active and not native.get_lateral_allowed()
  button(True)
  button(False)
  assert driver.active and native.get_lateral_allowed()
  exchange(wire, driver, fresh=False)
  assert not native.get_lateral_allowed()
  cs.buttonEvents = []
  for _ in range(205):
    step(driver, cs)
  assert not driver.active  # The decoded health report reaches mismatch shutdown.
