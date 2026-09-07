"""Persist Python crash tracebacks for the on-device error-log viewer."""
import datetime
import os
import traceback

from openpilot.common.swaglog import cloudlog

ERROR_LOG_PATH = "/data/community/crashes/error.log"
ERROR_LOG_MAX_BYTES = 100 * 1024  # keep the file under 100 KB


def save_exception() -> None:
  try:
    os.makedirs(os.path.dirname(ERROR_LOG_PATH), exist_ok=True)
    tb = traceback.format_exc()
    if not tb or tb.strip() == "NoneType: None":
      return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}]\n{tb}\n{'=' * 60}\n\n"
    existing = ""
    if os.path.exists(ERROR_LOG_PATH):
      with open(ERROR_LOG_PATH) as f:
        existing = f.read()
    combined = entry + existing
    if len(combined.encode()) > ERROR_LOG_MAX_BYTES:
      combined = combined.encode()[:ERROR_LOG_MAX_BYTES].decode(errors="ignore")
    with open(ERROR_LOG_PATH, "w") as f:
      f.write(combined)
  except Exception:
    pass


def capture_exception(*args, **kwargs) -> None:
  del args, kwargs
  cloudlog.exception("crash")
  save_exception()
