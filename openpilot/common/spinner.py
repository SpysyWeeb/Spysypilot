import os
import signal
import subprocess
import time
from openpilot.common.basedir import BASEDIR

BOOT_FIFO = '/tmp/spysypilot_boot.fifo'
BOOT_PID = '/tmp/spysypilot_spinner.pid'


def _spinner_alive() -> int | None:
  """Return spinner PID if alive, else None."""
  try:
    with open(BOOT_PID) as f:
      pid = int(f.read().strip())
    os.kill(pid, 0)
    return pid
  except Exception:
    return None


class Spinner:
  def __init__(self):
    self._fifo_path = BOOT_FIFO
    self._write_file = None
    self._owns_subprocess = False

    alive_pid = _spinner_alive()
    if alive_pid is None:
      # Clean up any stale files and start fresh
      for p in (BOOT_FIFO, BOOT_PID):
        try:
          os.unlink(p)
        except FileNotFoundError:
          pass
      try:
        os.mkfifo(BOOT_FIFO)
      except OSError:
        return

      try:
        proc = subprocess.Popen(
          ["./spinner.py", BOOT_FIFO],
          cwd=os.path.join(BASEDIR, "openpilot/system", "ui"),
          close_fds=True,
          start_new_session=True,  # survive parent process exit
        )
        with open(BOOT_PID, 'w') as f:
          f.write(str(proc.pid))
        self._owns_subprocess = True
      except OSError:
        return

    # Wait until the spinner subprocess opens the FIFO for reading (up to 15s).
    # This blocks here so callers can immediately send messages after __init__ returns.
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
      if self._connect():
        break
      time.sleep(0.05)

  def _connect(self) -> bool:
    """Non-blocking attempt to open FIFO for writing. Returns True on success."""
    if self._write_file is not None:
      return True
    try:
      fd = os.open(self._fifo_path, os.O_WRONLY | os.O_NONBLOCK)
      self._write_file = os.fdopen(fd, 'w', buffering=1)
      return True
    except OSError:
      return False

  def update(self, spinner_text: str):
    if not self._connect():
      return
    try:
      self._write_file.write(spinner_text + '\n')
      self._write_file.flush()
    except (BrokenPipeError, IOError):
      self._write_file = None

  def update_progress(self, cur: float, total: float):
    self.update(str(round(100 * cur / total)))

  def log(self, text: str):
    self.update(f"LOG:{text}")

  def detach(self):
    """Close our write end without killing the spinner subprocess."""
    if self._write_file is not None:
      try:
        self._write_file.close()
      except Exception:
        pass
      self._write_file = None
    self._owns_subprocess = False

  def close(self):
    """Close write end and kill the spinner subprocess."""
    self.detach()
    pid = _spinner_alive()
    if pid is not None:
      try:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
      except Exception:
        pass
    for p in (BOOT_FIFO, BOOT_PID):
      try:
        os.unlink(p)
      except Exception:
        pass

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def __del__(self):
    # Don't kill the subprocess on GC — detach only so next process can connect
    self.detach()


if __name__ == "__main__":
  with Spinner() as s:
    s.update("Spinner text")
    time.sleep(5.0)
  print("gone")
  time.sleep(5.0)
