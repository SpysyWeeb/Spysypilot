"""Install exception handler for process crash."""
import datetime
import os
import traceback
import sentry_sdk
from enum import Enum
from sentry_sdk.integrations.threading import ThreadingIntegration

from openpilot.common.params import Params
from openpilot.system.athena.registration import is_registered_device
from openpilot.common.hardware import HARDWARE, PC
from openpilot.common.swaglog import cloudlog
from openpilot.common.version import get_build_metadata, get_version

ERROR_LOG_PATH = "/data/community/crashes/error.log"
ERROR_LOG_MAX_BYTES = 100 * 1024  # keep the file under 100 KB


class SentryProject(Enum):
  # python project
  SELFDRIVE = "https://6f3c7076c1e14b2aa10f5dde6dda0cc4@o33823.ingest.sentry.io/77924"
  # native project
  SELFDRIVE_NATIVE = "https://3e4b586ed21a4479ad5d85083b639bc6@o33823.ingest.sentry.io/157615"


def report_tombstone(fn: str, message: str, contents: str) -> None:
  cloudlog.error({'tombstone': message})

  with sentry_sdk.configure_scope() as scope:
    scope.set_extra("tombstone_fn", fn)
    scope.set_extra("tombstone", contents)
    sentry_sdk.capture_message(message=message)
    sentry_sdk.flush()


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
    # newest entry at the top; trim if the file grows too large
    combined = entry + existing
    if len(combined.encode()) > ERROR_LOG_MAX_BYTES:
      combined = combined.encode()[:ERROR_LOG_MAX_BYTES].decode(errors="ignore")
    with open(ERROR_LOG_PATH, "w") as f:
      f.write(combined)
  except Exception:
    pass


def capture_exception(*args, **kwargs) -> None:
  cloudlog.error("crash", exc_info=kwargs.get('exc_info', 1))
  save_exception()

  try:
    sentry_sdk.capture_exception(*args, **kwargs)
    sentry_sdk.flush()  # https://github.com/getsentry/sentry-python/issues/291
  except Exception:
    cloudlog.exception("sentry exception")


def set_tag(key: str, value: str) -> None:
  sentry_sdk.set_tag(key, value)


def init(project: SentryProject) -> bool:
  build_metadata = get_build_metadata()
  # forks like to mess with this, so double check
  comma_remote = build_metadata.openpilot.comma_remote and "commaai" in build_metadata.openpilot.git_origin
  if not comma_remote or not is_registered_device() or PC:
    return False

  env = "release" if build_metadata.tested_channel else "master"
  dongle_id = Params().get("DongleId")

  integrations = []
  if project == SentryProject.SELFDRIVE:
    integrations.append(ThreadingIntegration(propagate_hub=True))

  sentry_sdk.init(project.value,
                  default_integrations=False,
                  release=get_version(),
                  integrations=integrations,
                  traces_sample_rate=1.0,
                  max_value_length=8192,
                  environment=env)

  sentry_sdk.set_user({"id": dongle_id})
  sentry_sdk.set_tag("dirty", build_metadata.openpilot.is_dirty)
  sentry_sdk.set_tag("origin", build_metadata.openpilot.git_origin)
  sentry_sdk.set_tag("branch", build_metadata.channel)
  sentry_sdk.set_tag("commit", build_metadata.openpilot.git_commit)
  sentry_sdk.set_tag("device", HARDWARE.get_device_type())

  return True

