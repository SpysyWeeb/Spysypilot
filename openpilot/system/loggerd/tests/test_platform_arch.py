import hashlib
import os
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]


def test_comma_hardware_arch_contract():
  sources = [
    ROOT / "SConstruct",
    ROOT / "openpilot/selfdrive/modeld/SConscript",
    ROOT / "openpilot/selfdrive/ui/SConscript",
    ROOT / "openpilot/system/loggerd/SConscript",
  ]
  text = "\n".join(path.read_text() for path in sources)

  assert "COMMA_HARDWARE = os.path.isfile('/AGNOS')" in text
  assert 'arch = "comma_arm64"' in text
  assert "larch64" not in text


def test_comma_hardware_runtime_contract():
  real_isfile = os.path.isfile
  def marker_isfile(path):
    return path == "/TICI" if path in ("/TICI", "/AGNOS") else real_isfile(path)

  with patch("os.path.isfile", side_effect=marker_isfile):
    hardware = runpy.run_path(str(ROOT / "openpilot/common/hardware/__init__.py"))

  assert hardware["TICI"] is True
  assert hardware["AGNOS"] is False
  assert hardware["COMMA_HARDWARE"] is True
  assert hardware["PC"] is False


def test_runtime_import_contract():
  subprocess.run([
    sys.executable,
    "-c",
    "import openpilot.system.hardware.hardwared; import openpilot.selfdrive.modeld.usbgpu_link",
  ], cwd=ROOT, check=True)


def test_chestnut_firmware_contract():
  firmware = ROOT / "openpilot/system/hardware/chestnut/firmware_wrapped.bin"

  assert firmware.is_file()
  assert hashlib.sha256(firmware.read_bytes()).hexdigest() == "9520fde0bf43d499c07abd0a09b74e94d8a7cc3d610f577b7a4e218ab8a378e9"
