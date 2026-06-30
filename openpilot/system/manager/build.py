#!/usr/bin/env python3
import os
import select as io_select
import subprocess

# NOTE: Do NOT import anything here that needs be built (e.g. params)
from openpilot.common.basedir import BASEDIR
from openpilot.common.spinner import Spinner
from openpilot.common.text_window import TextWindow
from openpilot.common.hardware import HARDWARE, AGNOS

def build() -> None:
  spinner = Spinner()
  spinner.update_progress(0, 100)
  spinner.log("Checking build...")

  HARDWARE.set_power_save(False)
  if AGNOS:
    os.sched_setaffinity(0, range(8))  # ensure we can use the isolcpus cores

  # building with all cores can result in using too much memory, so retry serially
  compile_output: list[bytes] = []
  for attempt, parallelism in enumerate(([], ["-j4"], ["-j1"])):
    compile_output.clear()
    if attempt > 0:
      spinner.log(f"Retrying build (attempt {attempt + 1})...")
    with subprocess.Popen(["scons", *parallelism], cwd=BASEDIR, env={**os.environ, "PWD": BASEDIR},
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as scons:
      assert scons.stdout is not None
      assert scons.stderr is not None

      # Read from both stdout (compilation lines) and stderr (progress + warnings)
      while scons.poll() is None:
        try:
          rlist, _, _ = io_select.select([scons.stdout, scons.stderr], [], [], 0.1)
          for f in rlist:
            line = f.readline()
            if not line:
              continue
            line = line.rstrip()

            if f is scons.stderr:
              prefix = b'progress: '
              if line.startswith(prefix):
                progress = float(line[len(prefix):])
                spinner.update_progress(100 * min(1., progress / 100.), 100.)
              elif len(line):
                compile_output.append(line)
                line_str = line.decode('utf8', 'replace')
                spinner.log(line_str)
                print(line_str)
            else:
              # stdout: actual scons compilation lines
              if len(line):
                compile_output.append(line)
                line_str = line.decode('utf8', 'replace')
                spinner.log(line_str)
                print(line_str)
        except Exception:
          pass

      # Drain both pipes before retrying or returning
      for f in (scons.stdout, scons.stderr):
        for line in f.read().split(b'\n'):
          line = line.rstrip()
          if len(line):
            compile_output.append(line)

    if scons.returncode == 0:
      if compile_output:
        spinner.log("Build complete.")
      else:
        spinner.log("Build up to date.")
      break

  if scons.returncode != 0:
    # Build failed log errors
    error_s = b"\n".join(compile_output).decode('utf8', 'replace')
    spinner.log("Build FAILED.")

    # Kill the boot spinner before showing the TextWindow
    spinner.close()
    if not os.getenv("CI"):
      with TextWindow("openpilot failed to build\n \n" + error_s) as t:
        t.wait_for_exit()
    exit(1)
  # On success, detach() is called implicitly by __del__ — spinner subprocess survives

if __name__ == "__main__":
  build()
