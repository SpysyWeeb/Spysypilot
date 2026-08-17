import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpilot.selfdrive.modeld import helpers


class TestHelpers(unittest.TestCase):
  def test_usbgpu_present_requires_current_firmware(self):
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
            self.assertIs(helpers.usbgpu_present(), expected)
