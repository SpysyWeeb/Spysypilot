"""Atomic, explicitly offroad persistence for BLaTv2 learner evidence."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from openpilot.selfdrive.controls.lib.blatv2.learner import (
  ProfileLearner,
  learner_evidence_sha256,
)
from openpilot.selfdrive.controls.lib.blatv2.vehicle_profile import (
  VehicleProfile,
)


def write_learner_evidence(
  path: str | os.PathLike[str],
  learner: ProfileLearner,
  *,
  offroad: bool,
) -> str:
  """Atomically persist evidence and return its canonical SHA-256 identity."""
  if offroad is not True:
    raise RuntimeError("learner evidence may be written only while offroad")
  if not isinstance(learner, ProfileLearner):
    raise TypeError("write_learner_evidence requires a ProfileLearner")

  target = Path(path)
  parent = target.parent
  encoded = learner.export_evidence()
  temporary_fd, temporary_name = tempfile.mkstemp(
    dir=parent,
    prefix=f".{target.name}.",
    suffix=".tmp",
  )
  try:
    with os.fdopen(temporary_fd, "wb") as temporary:
      temporary_fd = -1
      temporary.write(encoded)
      temporary.flush()
      os.fsync(temporary.fileno())
    os.replace(temporary_name, target)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(parent, directory_flags)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  except BaseException:
    if temporary_fd >= 0:
      os.close(temporary_fd)
    try:
      os.unlink(temporary_name)
    except FileNotFoundError:
      pass
    raise
  return learner_evidence_sha256(encoded)


def read_learner_evidence(
  path: str | os.PathLike[str],
  seed_profile: VehicleProfile,
) -> ProfileLearner:
  """Read, authenticate, and restore evidence for the exact seed profile."""
  encoded = Path(path).read_bytes()
  return ProfileLearner.from_evidence(seed_profile, encoded)
