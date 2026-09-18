import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpilot.selfdrive.modeld import helpers


class TestHelpers(unittest.TestCase):
  def test_chestnut_present_requires_current_firmware(self):
    cases = [
      ("add1", "0001", "custom ed4e39b7-CLEAN", True),
      ("3801", "0001", "custom ed4e39b7-CLEAN", True),
      ("add1", "0001", "custom stale-CLEAN", False),
    ]
    for vendor, product_id, product, expected in cases:
      with self.subTest(vendor=vendor, product=product):
        with tempfile.TemporaryDirectory() as tmp:
          root = Path(tmp)
          device = root / "1-1"
          device.mkdir()
          (device / "idVendor").write_text(vendor)
          (device / "idProduct").write_text(product_id)
          (device / "product").write_text(product)
          with patch.object(helpers, "USB_DEVICES_PATH", root):
            self.assertIs(helpers.chestnut_present(), expected)

  def test_chestnut_compiled_rejects_lfs_pointer(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      pkl = root / "big_driving_tinygrad.pkl"
      with patch.object(helpers, "MODELS_DIR", root):
        self.assertFalse(helpers.chestnut_compiled())
        pkl.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:76cc\nsize 776634338\n")
        self.assertFalse(helpers.chestnut_compiled())
        pkl.write_bytes(b"\0" * (2 << 20))
        self.assertTrue(helpers.chestnut_compiled())
