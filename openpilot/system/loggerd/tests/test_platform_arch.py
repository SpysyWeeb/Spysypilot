from pathlib import Path


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
  from openpilot.common.hardware import COMMA_HARDWARE, TICI

  assert COMMA_HARDWARE is TICI


def test_hardwared_import_contract():
  import openpilot.system.hardware.hardwared  # noqa: F401


def test_usbgpu_import_contract():
  import openpilot.selfdrive.modeld.usbgpu_link  # noqa: F401
