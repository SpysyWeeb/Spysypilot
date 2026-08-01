"""Content and runtime identity for cross-architecture route preparation.

The live PC session still requires an exact superproject checkout.  Durable
certificates use a narrower identity: only the reviewed source/data closure
that can alter prepared frames or certification-vector bytes, plus a paired
ARM/x86 numerical-environment fingerprint.  Controller, learner-selection,
UI, transport, and status changes therefore cannot evict a valid proof.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Final, Mapping


PREPARATION_IMPLEMENTATION_SCHEMA_VERSION: Final = 2
NUMERICAL_ENVIRONMENT_SCHEMA_VERSION: Final = 2
_IDENTITY_DOMAIN: Final = b"blatv2-preparation-implementation-v2\0"
_ENVIRONMENT_DOMAIN: Final = b"blatv2-numerical-environment-v2\0"
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_GIT: Final = "/usr/bin/git"

# Reviewed closure of code/data that can change preparation/vector bytes.
# Frame/decode contracts are factored into small modules so controller and
# learner-policy edits do not enter this identity.
PREPARATION_IMPLEMENTATION_DEPENDENCIES: Final = (
  "openpilot/cereal/log.capnp",
  "openpilot/cereal/services.py",
  "openpilot/selfdrive/controls/lib/blatv2/actuator.py",
  "openpilot/selfdrive/controls/lib/blatv2/calibration_profile.py",
  "openpilot/selfdrive/controls/lib/blatv2/certification_vector.py",
  "openpilot/selfdrive/controls/lib/blatv2/historical_build_descriptors.json",
  "openpilot/selfdrive/controls/lib/blatv2/learning_backfill.py",
  "openpilot/selfdrive/controls/lib/blatv2/learning_backfill_spool.py",
  "openpilot/selfdrive/controls/lib/blatv2/preparation_contract.py",
  "openpilot/selfdrive/controls/lib/blatv2/preparation_frame.py",
  "openpilot/selfdrive/controls/lib/blatv2/preparation_identity.py",
  "openpilot/selfdrive/controls/lib/blatv2/provisional_rack_dynamics.json",
  "openpilot/selfdrive/controls/lib/blatv2/rack_mapper.py",
  "openpilot/selfdrive/controls/lib/blatv2/route_evidence.py",
  "openpilot/selfdrive/controls/lib/blatv2/runtime_vehicle.py",
  "openpilot/selfdrive/controls/lib/blatv2/vehicle_profile.py",
)

# These are the native/runtime implementations actually exercised by bounded
# Cap'n Proto decode, Float64 vector math, and VehicleModel reconstruction.
_NUMERICAL_MODULES: Final = (
  "capnp.lib.capnp",
  "numpy._core._multiarray_umath",
  "opendbc.car.structs",
  "opendbc.car.vehicle_model",
  "openpilot.cereal",
)


class PreparationIdentityError(RuntimeError):
  """The committed preparation or numerical environment is unauthenticated."""


def _frame(digest: object, value: bytes) -> None:
  digest.update(len(value).to_bytes(8, "big"))
  digest.update(value)


def _canonical_json(value: object) -> bytes:
  try:
    return json.dumps(
      value,
      allow_nan=False,
      separators=(",", ":"),
      sort_keys=True,
    ).encode("utf-8")
  except (TypeError, ValueError) as exc:
    raise PreparationIdentityError("identity payload is not canonical JSON") from exc


def build_preparation_implementation_sha256(
  entries: Mapping[str, tuple[str, bytes]],
  *,
  opendbc_commit: str,
  panda_commit: str,
) -> str:
  """Hash an already-authenticated manifest; kept pure for mutation tests."""
  if (
    _COMMIT_RE.fullmatch(opendbc_commit) is None
    or _COMMIT_RE.fullmatch(panda_commit) is None
  ):
    raise PreparationIdentityError("preparation submodule identity is invalid")
  if tuple(sorted(entries)) != PREPARATION_IMPLEMENTATION_DEPENDENCIES:
    raise PreparationIdentityError("preparation dependency manifest is incomplete")
  digest = hashlib.sha256()
  digest.update(_IDENTITY_DOMAIN)
  _frame(digest, str(PREPARATION_IMPLEMENTATION_SCHEMA_VERSION).encode("ascii"))
  _frame(digest, opendbc_commit.encode("ascii"))
  _frame(digest, panda_commit.encode("ascii"))
  for relative in PREPARATION_IMPLEMENTATION_DEPENDENCIES:
    mode, encoded = entries[relative]
    if mode not in {"100644", "100755"} or type(encoded) is not bytes:
      raise PreparationIdentityError("preparation dependency entry is invalid")
    _frame(digest, relative.encode("utf-8"))
    _frame(digest, mode.encode("ascii"))
    _frame(digest, encoded)
  return digest.hexdigest()


def build_numerical_environment_fingerprint(payload: Mapping[str, object]) -> str:
  """Hash a validated architecture-specific runtime description."""
  if type(payload) is not dict or set(payload) != {
    "architecture",
    "byteorder",
    "implementation",
    "modules",
    "native_runtime_artifacts",
    "python_executable",
    "python_version",
    "schema_version",
  }:
    raise PreparationIdentityError("numerical environment keys are invalid")
  if payload["schema_version"] != NUMERICAL_ENVIRONMENT_SCHEMA_VERSION:
    raise PreparationIdentityError("numerical environment version is invalid")
  for key in ("architecture", "byteorder", "implementation"):
    value = payload[key]
    if type(value) is not str or not value or len(value) > 128:
      raise PreparationIdentityError("numerical environment text is invalid")
  version = payload["python_version"]
  if (
    type(version) is not list
    or len(version) != 3
    or any(type(value) is not int or value < 0 for value in version)
  ):
    raise PreparationIdentityError("numerical Python version is invalid")
  modules = payload["modules"]
  if type(modules) is not dict or tuple(sorted(modules)) != _NUMERICAL_MODULES:
    raise PreparationIdentityError("numerical module inventory is invalid")
  for name in _NUMERICAL_MODULES:
    record = modules[name]
    if type(record) is not dict or set(record) != {
      "artifact_name", "artifact_sha256", "version",
    }:
      raise PreparationIdentityError("numerical module record is invalid")
    if (
      type(record["artifact_name"]) is not str
      or not record["artifact_name"]
      or type(record["artifact_sha256"]) is not str
      or _SHA256_RE.fullmatch(record["artifact_sha256"]) is None
      or type(record["version"]) is not str
    ):
      raise PreparationIdentityError("numerical module identity is invalid")
  executable = payload["python_executable"]
  if type(executable) is not dict or set(executable) != {
    "artifact_name", "artifact_sha256",
  }:
    raise PreparationIdentityError("Python executable identity is invalid")
  if (
    type(executable["artifact_name"]) is not str
    or not executable["artifact_name"]
    or type(executable["artifact_sha256"]) is not str
    or _SHA256_RE.fullmatch(executable["artifact_sha256"]) is None
  ):
    raise PreparationIdentityError("Python executable identity is invalid")
  native = payload["native_runtime_artifacts"]
  if type(native) is not dict or not native or len(native) > 512:
    raise PreparationIdentityError("native runtime inventory is invalid")
  for key, record in native.items():
    if (
      type(key) is not str
      or not key
      or type(record) is not dict
      or set(record) != {"artifact_name", "artifact_sha256"}
      or type(record["artifact_name"]) is not str
      or not record["artifact_name"]
      or type(record["artifact_sha256"]) is not str
      or _SHA256_RE.fullmatch(record["artifact_sha256"]) is None
    ):
      raise PreparationIdentityError("native runtime identity is invalid")
  return hashlib.sha256(_ENVIRONMENT_DOMAIN + _canonical_json(payload)).hexdigest()


def _hash_regular_file(path: Path) -> str:
  try:
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
      raise PreparationIdentityError("numerical module artifact is unsafe")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
  except PreparationIdentityError:
    raise
  except OSError as exc:
    raise PreparationIdentityError("numerical module artifact is unavailable") from exc
  digest = hashlib.sha256()
  try:
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
      raise PreparationIdentityError("numerical module changed during open")
    while chunk := os.read(descriptor, 1024 * 1024):
      digest.update(chunk)
    after = os.fstat(descriptor)
    if (
      (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
      != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
    ):
      raise PreparationIdentityError("numerical module changed while hashing")
  finally:
    os.close(descriptor)
  return digest.hexdigest()


def numerical_environment_payload() -> dict[str, object]:
  """Measure the local architecture/runtime without comparing ARM to x86."""
  modules: dict[str, object] = {}
  for name in _NUMERICAL_MODULES:
    try:
      module = importlib.import_module(name)
      path_value = getattr(module, "__file__", None)
      if type(path_value) is not str:
        raise PreparationIdentityError("numerical module lacks an artifact")
      path = Path(path_value).resolve(strict=True)
    except PreparationIdentityError:
      raise
    except Exception as exc:
      raise PreparationIdentityError(
        f"numerical module is unavailable: {name}",
      ) from exc
    version = getattr(module, "__version__", "")
    modules[name] = {
      "artifact_name": path.name,
      "artifact_sha256": _hash_regular_file(path),
      "version": "" if version is None else str(version),
    }
  executable_path = Path(sys.executable).resolve(strict=True)
  native_paths: set[Path] = {executable_path}
  try:
    maps = Path("/proc/self/maps").read_text(encoding="utf-8")
  except OSError as exc:
    raise PreparationIdentityError("native runtime map is unavailable") from exc
  for line in maps.splitlines():
    fields = line.split(maxsplit=5)
    if len(fields) == 6 and fields[5].startswith("/"):
      try:
        candidate = Path(fields[5]).resolve(strict=True)
        info = candidate.lstat()
      except OSError:
        continue
      if stat.S_ISREG(info.st_mode) and (
        ".so" in candidate.name or candidate == executable_path
      ):
        native_paths.add(candidate)
  native_runtime_artifacts: dict[str, object] = {}
  for path in sorted(native_paths, key=str):
    identity = _hash_regular_file(path)
    native_runtime_artifacts[f"{path.name}:{identity}"] = {
      "artifact_name": path.name,
      "artifact_sha256": identity,
    }
  return {
    "architecture": platform.machine(),
    "byteorder": sys.byteorder,
    "implementation": f"{sys.implementation.name}:{sys.implementation.cache_tag}",
    "modules": modules,
    "native_runtime_artifacts": native_runtime_artifacts,
    "python_executable": {
      "artifact_name": executable_path.name,
      "artifact_sha256": _hash_regular_file(executable_path),
    },
    "python_version": list(sys.version_info[:3]),
    "schema_version": NUMERICAL_ENVIRONMENT_SCHEMA_VERSION,
  }


def numerical_environment_sha256() -> str:
  return build_numerical_environment_fingerprint(numerical_environment_payload())


def _git(root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
  environment = {
    key: value for key, value in os.environ.items() if not key.startswith("GIT_")
  }
  environment.update({"LANG": "C", "LC_ALL": "C"})
  process = subprocess.run(
    (_GIT, "-C", str(root), *arguments),
    input=input_bytes,
    capture_output=True,
    check=False,
    env=environment,
  )
  if process.returncode:
    raise PreparationIdentityError("preparation Git identity is unavailable")
  return process.stdout


def preparation_implementation_sha256(
  source_root: str | Path,
  *,
  opendbc_commit: str,
  panda_commit: str,
) -> str:
  """Authenticate and hash the explicit dependency closure from one checkout."""
  try:
    root = Path(source_root).resolve(strict=True)
    root_info = root.lstat()
  except OSError as exc:
    raise PreparationIdentityError("preparation source root is unavailable") from exc
  if not stat.S_ISDIR(root_info.st_mode) or root.is_symlink():
    raise PreparationIdentityError("preparation source root is unsafe")

  index = _git(
    root,
    "ls-files",
    "--stage",
    "-z",
    "--",
    *PREPARATION_IMPLEMENTATION_DEPENDENCIES,
  )
  records = [record for record in index.split(b"\0") if record]
  if len(records) != len(PREPARATION_IMPLEMENTATION_DEPENDENCIES):
    raise PreparationIdentityError("preparation dependency is not tracked")
  staged: dict[str, tuple[str, str]] = {}
  for record in records:
    try:
      metadata, raw_path = record.split(b"\t", 1)
      mode, object_id, stage = metadata.decode("ascii").split()
      relative = raw_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
      raise PreparationIdentityError("preparation Git entry is malformed") from exc
    if stage != "0" or mode not in {"100644", "100755"}:
      raise PreparationIdentityError("preparation Git entry mode is unsupported")
    staged[relative] = (mode, object_id)
  if tuple(sorted(staged)) != PREPARATION_IMPLEMENTATION_DEPENDENCIES:
    raise PreparationIdentityError("preparation Git manifest does not match")

  entries: dict[str, tuple[str, bytes]] = {}
  for relative in PREPARATION_IMPLEMENTATION_DEPENDENCIES:
    path = root / relative
    try:
      info = path.lstat()
      encoded = path.read_bytes()
    except OSError as exc:
      raise PreparationIdentityError(
        f"preparation dependency is unreadable: {relative}",
      ) from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
      raise PreparationIdentityError(
        f"preparation dependency is unsafe: {relative}",
      )
    mode, object_id = staged[relative]
    # Let Git honor the repository's object format (SHA-1 or SHA-256) instead
    # of reimplementing blob IDs with a hard-coded digest.
    observed = _git(root, "hash-object", "--stdin", input_bytes=encoded).decode(
      "ascii",
    ).strip()
    if observed != object_id:
      raise PreparationIdentityError(
        f"preparation dependency differs from the Git index: {relative}",
      )
    entries[relative] = (mode, encoded)
  return build_preparation_implementation_sha256(
    entries,
    opendbc_commit=opendbc_commit,
    panda_commit=panda_commit,
  )
