import sys
from pathlib import Path

from openpilot.common.hardware import AGNOS
from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, USB_DEVICES_PATH, is_chestnut_usb_id

MODELS_DIR = Path(__file__).resolve().parent / 'models'


def modeld_pkl_path(chestnut: bool):
  prefix = 'big_' if chestnut else ''
  return MODELS_DIR / f'{prefix}driving_tinygrad.pkl'

def load_oob(path, chestnut=False):
  from tinygrad import Context
  device = 'USB+AMD:LLVM' if chestnut else 'QCOM' if AGNOS else 'METAL' if sys.platform == 'darwin' else 'CPU:LLVM'
  with Context(DEV=device):
    from tinygrad_repo.examples.openpilot.helpers import load_pickle
    return load_pickle(path, out_of_band=True)

def chestnut_present() -> bool:
  for d in USB_DEVICES_PATH.glob("*"):
    try:
      usb_id = (int((d / "idVendor").read_text(), 16), int((d / "idProduct").read_text(), 16))
      product = (d / "product").read_text().strip()
      if is_chestnut_usb_id(*usb_id) and product == CHESTNUT_USB_PRODUCT:
        return True
    except Exception:
      pass
  return False

def chestnut_compiled() -> bool:
  # the model pickle comes through git LFS; an unfetched pointer file is not a model
  path = modeld_pkl_path(chestnut=True)
  return path.is_file() and path.stat().st_size > 1 << 20
