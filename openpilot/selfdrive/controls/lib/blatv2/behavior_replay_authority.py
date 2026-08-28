"""Authenticated, route-major BLaTv2 comparison authority.

This is the only shared operation allowed to turn trainer route evidence into
a verified stock/candidate comparison.  It starts from the trainer's immutable
content-addressed import manifest, reopens and authenticates every selected
route object, loads the committed metric/segmentation/gate artifacts, executes
the reviewed stock and modular adapters, aggregates their compact route
results, and independently repeats the complete operation before issuing a
receipt.

The public boundary accepts no prepared routes, replay outputs, scorecards, or
aggregates.  Those values are data rather than execution authority and cannot
be exchanged for a receipt.  This module is PC-only and has no Params,
publication, controller, schema, process, or actuation path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys

from openpilot.selfdrive.controls.lib.blatv2.behavior_aggregate import (
  BehaviorAggregateEvaluation,
  BehaviorAggregateSpec,
  BehaviorRouteSplit,
  aggregate_behavior_route_results,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_configuration import (
  BEHAVIOR_GATE_SPEC_PATH,
  BEHAVIOR_SEGMENTATION_CONFIG_PATH,
  load_behavior_gate_spec,
  load_behavior_segmentation_config,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  BehaviorGateSpec,
  ReplayRole,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorScenarioProvenance,
  BehaviorScenarioSetIdentity,
  canonical_json,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import (
  BehaviorPolicy,
  build_candidate_grid,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_replay import (
  behavior_scenario_provenance_from_route_source,
  reviewed_replay_core_identity,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_route_evaluator import (
  BehaviorRouteEvaluation,
  BehaviorRouteEvaluationError,
  _evaluate_behavior_route_policies_with_registry_for_test,
  evaluate_behavior_route_policies,
)
from openpilot.selfdrive.controls.lib.blatv2.calibration_profile import (
  VehicleCalibrationProfile,
)
from openpilot.selfdrive.controls.lib.blatv2.certification_vector import (
  CERTIFICATION_VECTOR_MAX_BYTES,
  CERTIFICATION_VECTOR_SCHEMA_VERSION,
  CertificationVector,
  CertificationVectorError,
)
from openpilot.selfdrive.controls.lib.blatv2.policy import ControllerPolicy
from openpilot.selfdrive.controls.lib.blatv2.preparation_contract import (
  BLATV2_LIBRARY_ROOT,
  PROVISIONAL_RACK_DYNAMICS_PATH,
)
from openpilot.selfdrive.controls.lib.blatv2.route_evidence import (
  RouteEvidenceError,
  RouteEvidenceFileSummary,
  inspect_route_evidence_file,
)
from openpilot.selfdrive.controls.lib.blatv2.runtime_vehicle import (
  ProvisionalRackDynamics,
)


BEHAVIOR_REPLAY_AUTHORITY_SCHEMA_VERSION = 2
TRAINER_IMPORT_MANIFEST_SCHEMA_VERSION = 2
REVIEWED_REPLAY_SOURCE_SCHEMA_VERSION = 2
PROVISIONAL_CONTROLLER_POLICY_PATH = (
  BLATV2_LIBRARY_ROOT / "provisional_controller_policy.json"
)

_IMPORT_MANIFEST_MAXIMUM_BYTES = 16 * 1024 * 1024
_PHYSICAL_PROFILE_MAXIMUM_BYTES = 4 * 1024 * 1024
_MAXIMUM_MANIFEST_ROUTES = 4096
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_JOB_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_ROUTE_ID_RE = re.compile(r"[0-9a-f]{8}--[0-9a-f]{10}\Z")
_REJECTION_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SOURCE_COMPOSITION_DOMAIN = b"blatv2-shared-execution-source-v1\0"
_RUNTIME_IDENTITY_DOMAIN = b"blatv2-shared-execution-runtime-v2\0"
_MODULE_CLOSURE_DOMAIN = b"blatv2-executing-module-closure-v2\0"
_GIT_TIMEOUT_SECONDS = 30.0


class BehaviorReplayAuthorityError(RuntimeError):
  """The immutable experiment cannot produce an authenticated receipt."""


def _sha256(value: object, name: str) -> str:
  if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
    raise BehaviorReplayAuthorityError(f"{name} must be lowercase SHA-256")
  return value


def _commit(value: object, name: str) -> str:
  if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
    raise BehaviorReplayAuthorityError(f"{name} must be a full lowercase commit")
  return value


def _uint(value: object, name: str, maximum: int | None = None) -> int:
  if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
    raise BehaviorReplayAuthorityError(f"{name} must be a bounded nonnegative integer")
  return value


def _canonical_sha256(value: object) -> str:
  return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ReviewedReplaySource:
  """Expected clean-checkout/runtime identity verified against execution.

  Production derives these fields again from the executing checkout, gitlinks,
  and numerical runtime. The caller value is an equality expectation, never
  authority by assertion. This keeps trainer implementation out of the shared
  numerical package while binding the exact bytes that executed replay.
  """

  source_openpilot_commit: str
  opendbc_commit: str
  panda_commit: str
  source_composition_sha256: str
  runtime_identity_sha256: str
  module_closure_sha256: str
  schema_version: int = REVIEWED_REPLAY_SOURCE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    _commit(self.source_openpilot_commit, "reviewed openpilot commit")
    _commit(self.opendbc_commit, "reviewed opendbc commit")
    _commit(self.panda_commit, "reviewed panda commit")
    _sha256(self.source_composition_sha256, "reviewed source composition")
    _sha256(self.runtime_identity_sha256, "reviewed runtime identity")
    _sha256(self.module_closure_sha256, "reviewed module closure")
    if self.schema_version != REVIEWED_REPLAY_SOURCE_SCHEMA_VERSION:
      raise ValueError("reviewed replay source schema is incompatible")

  def to_dict(self) -> dict[str, object]:
    return {
      "opendbcCommit": self.opendbc_commit,
      "moduleClosureSha256": self.module_closure_sha256,
      "pandaCommit": self.panda_commit,
      "runtimeIdentitySha256": self.runtime_identity_sha256,
      "sourceCompositionSha256": self.source_composition_sha256,
      "sourceOpenpilotCommit": self.source_openpilot_commit,
      "schemaVersion": self.schema_version,
    }


def _git(root: Path, *arguments: str) -> bytes:
  executable = shutil.which("git")
  if executable is None:
    raise BehaviorReplayAuthorityError("git executable is unavailable")
  environment = os.environ.copy()
  for name in tuple(environment):
    if name.startswith("GIT_"):
      del environment[name]
  environment.update({
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "LC_ALL": "C",
  })
  try:
    result = subprocess.run(
      (
        executable,
        "--no-optional-locks",
        "-c", "core.fsmonitor=false",
        "-c", "core.untrackedCache=false",
        "-c", f"core.hooksPath={os.devnull}",
        "-C", os.fspath(root),
        *arguments,
      ),
      stdin=subprocess.DEVNULL,
      capture_output=True,
      check=False,
      shell=False,
      env=environment,
      timeout=_GIT_TIMEOUT_SECONDS,
    )
  except (OSError, subprocess.TimeoutExpired) as error:
    raise BehaviorReplayAuthorityError("source checkout cannot be inspected") from error
  if result.returncode != 0:
    raise BehaviorReplayAuthorityError("source checkout failed Git inspection")
  return result.stdout


def _git_text(root: Path, *arguments: str) -> str:
  try:
    return _git(root, *arguments).decode("utf-8", errors="strict").strip()
  except UnicodeDecodeError as error:
    raise BehaviorReplayAuthorityError("source checkout Git output is not UTF-8") from error


def _require_clean_checkout(root: Path, label: str) -> None:
  status = _git(
    root,
    "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none",
  )
  if status:
    raise BehaviorReplayAuthorityError(f"{label} worktree is not clean")
  for entry in (value for value in _git(root, "ls-files", "-v", "-z").split(b"\0") if value):
    flag, separator, _ = entry.partition(b" ")
    if separator != b" " or flag != b"H":
      raise BehaviorReplayAuthorityError(f"{label} index has hidden worktree state")


def _gitlink(root: Path, relative: str) -> str:
  records = tuple(value for value in _git(root, "ls-tree", "-z", "HEAD", "--", relative).split(b"\0") if value)
  if len(records) != 1:
    raise BehaviorReplayAuthorityError(f"{relative} Git link is unavailable")
  try:
    header, path = records[0].split(b"\t", 1)
    mode, kind, commit = header.decode("ascii").split(" ", 2)
    decoded_path = path.decode("utf-8", errors="strict")
  except (UnicodeDecodeError, ValueError) as error:
    raise BehaviorReplayAuthorityError(f"{relative} Git link is malformed") from error
  if (mode, kind, decoded_path) != ("160000", "commit", relative):
    raise BehaviorReplayAuthorityError(f"{relative} is not a Git submodule")
  return _commit(commit, f"{relative} Git link")


def _stable_regular_sha256(path: Path, purpose: str) -> str:
  try:
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
      raise BehaviorReplayAuthorityError(f"{purpose} is not a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
      while block := stream.read(1024 * 1024):
        digest.update(block)
    after = path.stat()
  except (OSError, ValueError) as error:
    raise BehaviorReplayAuthorityError(f"{purpose} cannot be hashed") from error
  if (
    before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns,
  ) != (
    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns,
  ):
    raise BehaviorReplayAuthorityError(f"{purpose} changed while hashing")
  return digest.hexdigest()


def _runtime_module_identity(name: str) -> dict[str, str]:
  try:
    module = importlib.import_module(name)
    path_value = getattr(module, "__file__", None)
    if type(path_value) is not str:
      raise ValueError("module has no file")
    path = Path(path_value).resolve(strict=True)
    version = str(getattr(module, "__version__", "unversioned"))
  except (ImportError, OSError, ValueError) as error:
    raise BehaviorReplayAuthorityError(f"required runtime module {name} is unavailable") from error
  return {
    "fileSha256": _stable_regular_sha256(path, f"runtime module {name}"),
    "name": name,
    "origin": path.as_posix(),
    "version": version,
  }


def _committed_module_identity(
  name: str,
  module: object,
  superproject_root: Path,
  opendbc_root: Path,
) -> dict[str, str]:
  path_value = getattr(module, "__file__", None)
  if type(path_value) is not str:
    raise BehaviorReplayAuthorityError(f"executing module {name} has no file origin")
  unresolved = Path(path_value)
  try:
    path = unresolved.resolve(strict=True)
  except OSError as error:
    raise BehaviorReplayAuthorityError(f"executing module {name} is unavailable") from error
  if not unresolved.is_absolute() or unresolved != path:
    raise BehaviorReplayAuthorityError(f"executing module {name} origin is not canonical")
  if name == "openpilot" or name.startswith("openpilot."):
    expected_root = superproject_root / "openpilot"
    repository = superproject_root
  elif name == "opendbc" or name.startswith("opendbc."):
    expected_root = opendbc_root / "opendbc"
    repository = opendbc_root
  else:
    raise BehaviorReplayAuthorityError(f"executing module {name} is outside the reviewed closure")
  try:
    path.relative_to(expected_root)
    relative = path.relative_to(repository).as_posix()
  except ValueError as error:
    raise BehaviorReplayAuthorityError(
      f"executing module {name} escaped its reviewed repository",
    ) from error
  actual_sha256 = _stable_regular_sha256(path, f"executing module {name}")
  if path.suffix == ".py":
    try:
      committed = _git(repository, "show", f"HEAD:{relative}")
    except BehaviorReplayAuthorityError as error:
      raise BehaviorReplayAuthorityError(
        f"executing Python module {name} is not tracked by the reviewed tree",
      ) from error
    if hashlib.sha256(committed).hexdigest() != actual_sha256:
      raise BehaviorReplayAuthorityError(
        f"executing Python module {name} differs from the reviewed tree",
      )
    kind = "committed-python"
  else:
    # Generated Python loaders and native extensions are not represented by a
    # source-tree blob. Their exact bytes are therefore explicit runtime
    # identity, not inferred from a package version string.
    kind = "runtime-bytes"
  return {
    "fileSha256": actual_sha256,
    "kind": kind,
    "name": name,
    "origin": relative,
  }


def _executing_module_closure(root: Path) -> tuple[dict[str, str], ...]:
  # Importing car_helpers loads the complete registered vehicle interface and
  # controller population. This makes later route activation unable to expand
  # the reviewed opendbc numerical closure silently.
  importlib.import_module("opendbc.car.car_helpers")
  selected_names = tuple(sorted(
    name
    for name, module in sys.modules.items()
    if module is not None
    and getattr(module, "__file__", None) is not None
    and (
      name in {
        "openpilot.selfdrive.controls.lib.blatv2",
        "openpilot.common.realtime",
        "openpilot.selfdrive.controls.lib.drive_helpers",
        "openpilot.selfdrive.controls.lib.latcontrol",
        "openpilot.selfdrive.controls.lib.latcontrol_torque",
        "opendbc",
        "opendbc.can",
      }
      or name.startswith((
        "openpilot.selfdrive.controls.lib.blatv2.",
        "opendbc.car.",
        "opendbc.can.",
      ))
    )
  ))
  if not selected_names:
    raise BehaviorReplayAuthorityError("executing numerical module closure is empty")
  opendbc_root = root / "opendbc_repo"
  return tuple(
    _committed_module_identity(
      name,
      sys.modules[name],
      root,
      opendbc_root,
    )
    for name in selected_names
  )


def inspect_current_replay_source() -> ReviewedReplaySource:
  """Derive the exact clean checkout and numerical runtime executing replay."""
  root = Path(__file__).resolve().parents[5]
  if root != root.resolve(strict=True):
    raise BehaviorReplayAuthorityError("shared source root is not canonical")
  _require_clean_checkout(root, "openpilot")
  openpilot_commit = _commit(_git_text(root, "rev-parse", "HEAD"), "openpilot HEAD")
  tree = _git_text(root, "rev-parse", "HEAD^{tree}")
  links: dict[str, str] = {}
  submodule_trees: dict[str, str] = {}
  for relative in ("opendbc_repo", "panda"):
    commit = _gitlink(root, relative)
    submodule = root / relative
    _require_clean_checkout(submodule, relative)
    if _git_text(submodule, "rev-parse", "HEAD") != commit:
      raise BehaviorReplayAuthorityError(f"{relative} worktree differs from its Git link")
    links[relative] = commit
    submodule_trees[relative] = _git_text(submodule, "rev-parse", "HEAD^{tree}")
  composition = hashlib.sha256(_SOURCE_COMPOSITION_DOMAIN + canonical_json({
    "openpilotCommit": openpilot_commit,
    "openpilotTree": tree,
    "submoduleCommits": links,
    "submoduleTrees": submodule_trees,
  }).encode("utf-8")).hexdigest()
  module_closure = _executing_module_closure(root)
  module_closure_sha256 = hashlib.sha256(
    _MODULE_CLOSURE_DOMAIN + canonical_json(module_closure).encode("utf-8"),
  ).hexdigest()
  executable = Path(sys.executable).resolve(strict=True)
  runtime = hashlib.sha256(_RUNTIME_IDENTITY_DOMAIN + canonical_json({
    "byteorder": sys.byteorder,
    "cacheTag": sys.implementation.cache_tag,
    "executableSha256": _stable_regular_sha256(executable, "Python executable"),
    "implementation": sys.implementation.name,
    "modules": [
      _runtime_module_identity("numpy"),
      _runtime_module_identity("numpy._core._multiarray_umath"),
      _runtime_module_identity("capnp"),
      _runtime_module_identity("capnp.lib.capnp"),
    ],
    "moduleClosureSha256": module_closure_sha256,
    "version": list(sys.version_info[:5]),
  }).encode("utf-8")).hexdigest()
  return ReviewedReplaySource(
    source_openpilot_commit=openpilot_commit,
    opendbc_commit=links["opendbc_repo"],
    panda_commit=links["panda"],
    source_composition_sha256=composition,
    runtime_identity_sha256=runtime,
    module_closure_sha256=module_closure_sha256,
  )


def _verify_replay_source(expected: ReviewedReplaySource) -> ReviewedReplaySource:
  if not isinstance(expected, ReviewedReplaySource):
    raise TypeError("replay_source must be a ReviewedReplaySource")
  observed = inspect_current_replay_source()
  if observed != expected:
    raise BehaviorReplayAuthorityError("reviewed replay source differs from executing bytes")
  return observed


@dataclass(frozen=True, slots=True)
class _RegularFileIdentity:
  device: int
  inode: int
  size: int
  mtime_ns: int
  ctime_ns: int

  @classmethod
  def from_stat(cls, value: os.stat_result) -> _RegularFileIdentity:
    return cls(
      value.st_dev,
      value.st_ino,
      value.st_size,
      value.st_mtime_ns,
      value.st_ctime_ns,
    )


def _safe_directory(path: Path, purpose: str) -> Path:
  if not isinstance(path, Path) or not path.is_absolute():
    raise BehaviorReplayAuthorityError(f"{purpose} must be an absolute pathlib.Path")
  try:
    metadata = path.lstat()
    resolved = path.resolve(strict=True)
  except OSError as error:
    raise BehaviorReplayAuthorityError(f"{purpose} is unavailable") from error
  if not stat.S_ISDIR(metadata.st_mode) or path != resolved:
    raise BehaviorReplayAuthorityError(f"{purpose} is not a canonical directory")
  return path


def _read_immutable_regular(
  path: Path,
  purpose: str,
  maximum_bytes: int,
) -> tuple[bytes, _RegularFileIdentity]:
  if not isinstance(path, Path) or not path.is_absolute():
    raise BehaviorReplayAuthorityError(f"{purpose} path must be absolute")
  try:
    metadata = path.lstat()
    resolved = path.resolve(strict=True)
  except OSError as error:
    raise BehaviorReplayAuthorityError(f"{purpose} is unavailable") from error
  if (
    path != resolved
    or path.is_symlink()
    or not stat.S_ISREG(metadata.st_mode)
    or metadata.st_mode & 0o222
    or not 0 < metadata.st_size <= maximum_bytes
  ):
    raise BehaviorReplayAuthorityError(f"{purpose} is not an immutable regular file")
  expected = _RegularFileIdentity.from_stat(metadata)
  descriptor = -1
  try:
    descriptor = os.open(
      path,
      os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or _RegularFileIdentity.from_stat(opened) != expected:
      raise BehaviorReplayAuthorityError(f"{purpose} changed while opening")
    output = bytearray()
    while len(output) < expected.size:
      block = os.read(descriptor, min(1024 * 1024, expected.size - len(output)))
      if not block:
        break
      output.extend(block)
    if len(output) != expected.size or os.read(descriptor, 1):
      raise BehaviorReplayAuthorityError(f"{purpose} size changed while reading")
    if _RegularFileIdentity.from_stat(os.fstat(descriptor)) != expected:
      raise BehaviorReplayAuthorityError(f"{purpose} changed while reading")
    return bytes(output), expected
  except BehaviorReplayAuthorityError:
    raise
  except OSError as error:
    raise BehaviorReplayAuthorityError(f"{purpose} cannot be read") from error
  finally:
    if descriptor >= 0:
      os.close(descriptor)


def _validate_immutable_regular_metadata(
  path: Path,
  purpose: str,
  expected_size: int,
) -> None:
  """Validate a large immutable object without copying it into memory."""
  if not isinstance(path, Path) or not path.is_absolute():
    raise BehaviorReplayAuthorityError(f"{purpose} path must be absolute")
  try:
    metadata = path.lstat()
    resolved = path.resolve(strict=True)
  except OSError as error:
    raise BehaviorReplayAuthorityError(f"{purpose} is unavailable") from error
  if (
    path != resolved
    or path.is_symlink()
    or not stat.S_ISREG(metadata.st_mode)
    or metadata.st_mode & 0o222
    or metadata.st_size != expected_size
  ):
    raise BehaviorReplayAuthorityError(f"{purpose} is not an immutable regular file")


def _parse_canonical_file(
  path: Path,
  purpose: str,
  maximum_bytes: int,
) -> tuple[dict[str, object], str]:
  encoded, _ = _read_immutable_regular(path, purpose, maximum_bytes)
  try:
    payload: object = json.loads(encoded)
    if type(payload) is not dict or encoded != (canonical_json(payload) + "\n").encode("utf-8"):
      raise ValueError("not canonical")
  except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
    raise BehaviorReplayAuthorityError(f"{purpose} is not canonical JSON") from error
  return payload, hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class _ImportedRoute:
  route_id: str
  object_path: Path
  artifact_sha256: str
  size_bytes: int
  source_row: Mapping[str, object]
  scenario: BehaviorScenarioProvenance


@dataclass(frozen=True, slots=True)
class _AuthenticatedImport:
  manifest_sha256: str
  scenario_source_set_identity: str
  inspector_identity_sha256: str
  imported_routes: tuple[_ImportedRoute, ...]


def _strict_import_inspector(value: object) -> dict[str, object]:
  keys = {
    "runtimeIdentitySha256",
    "sourceCompositionSha256",
    "sourceIdentitySha256",
    "sourceOpenpilotCommit",
  }
  if type(value) is not dict or set(value) != keys:
    raise BehaviorReplayAuthorityError("import inspector identity is malformed")
  for name in (
    "runtimeIdentitySha256",
    "sourceCompositionSha256",
    "sourceIdentitySha256",
  ):
    _sha256(value[name], f"import inspector {name}")
  _commit(value["sourceOpenpilotCommit"], "import inspector source commit")
  return value


def _validate_remote_worker(value: object) -> None:
  keys = {
    "jobId",
    "requestSha256",
    "workerExtractorSha256",
    "workerImplementationCommit",
    "workerImplementationSha256",
    "workerInstanceId",
  }
  if type(value) is not dict or set(value) != keys:
    raise BehaviorReplayAuthorityError("import remote-worker identity is malformed")
  if type(value["jobId"]) is not str or _JOB_ID_RE.fullmatch(value["jobId"]) is None:
    raise BehaviorReplayAuthorityError("import remote-worker job ID is malformed")
  for name in (
    "requestSha256",
    "workerExtractorSha256",
    "workerImplementationSha256",
    "workerInstanceId",
  ):
    _sha256(value[name], f"import remote-worker {name}")
  _commit(value["workerImplementationCommit"], "remote-worker implementation commit")


def _source_row(summary: RouteEvidenceFileSummary) -> dict[str, object]:
  value = summary.source_identity.manifest_dict()
  value["routeId"] = value.pop("route_id")
  return value


def _validate_certification_vector_descriptor(
  store_root: Path,
  route_id: str,
  value: object,
  source: Mapping[str, object],
  route_evidence_sha256: str,
) -> None:
  keys = {
    "authorityArtifactIds",
    "path",
    "schemaVersion",
    "selectionIdentitySha256",
    "sha256",
    "sizeBytes",
  }
  if type(value) is not dict or set(value) != keys:
    raise BehaviorReplayAuthorityError(
      f"route {route_id} certification vector is malformed",
    )
  vector_authority_ids = value["authorityArtifactIds"]
  if type(vector_authority_ids) is not list or len(vector_authority_ids) != 2:
    raise BehaviorReplayAuthorityError(
      f"route {route_id} certification authority is malformed",
    )
  for index, authority_id in enumerate(vector_authority_ids):
    _sha256(authority_id, f"route {route_id} certification authority {index + 1}")
  vector_sha256 = _sha256(value["sha256"], f"route {route_id} certification vector")
  selection_sha256 = _sha256(
    value["selectionIdentitySha256"],
    f"route {route_id} certification selection",
  )
  size = _uint(
    value["sizeBytes"],
    f"route {route_id} certification vector size",
    CERTIFICATION_VECTOR_MAX_BYTES,
  )
  if size == 0 or value["schemaVersion"] != CERTIFICATION_VECTOR_SCHEMA_VERSION:
    raise BehaviorReplayAuthorityError(
      f"route {route_id} certification version is incompatible",
    )
  expected_path = f"objects/{vector_sha256}.cert-vector"
  if value["path"] != expected_path:
    raise BehaviorReplayAuthorityError(
      f"route {route_id} certification vector is not content-addressed",
    )
  try:
    encoded, _ = _read_immutable_regular(
      store_root / "objects" / f"{vector_sha256}.cert-vector",
      f"route {route_id} certification vector",
      CERTIFICATION_VECTOR_MAX_BYTES,
    )
    vector = CertificationVector.from_bytes(encoded)
  except (BehaviorReplayAuthorityError, CertificationVectorError) as error:
    raise BehaviorReplayAuthorityError(
      f"route {route_id} certification vector failed authentication",
    ) from error
  if len(encoded) != size or vector.sha256 != vector_sha256:
    raise BehaviorReplayAuthorityError(
      f"route {route_id} certification identity differs",
    )
  manifest = vector.manifest
  if (
    manifest.get("route_name") != route_id
    or manifest.get("selection_identity_sha256") != selection_sha256
  ):
    raise BehaviorReplayAuthorityError(
      f"route {route_id} certification provenance differs",
    )
  expected_segments = [
    {"index": index, "sha256": sha256, "size_bytes": size_bytes}
    for index, (sha256, size_bytes) in enumerate(zip(
      source.get("route_segment_sha256", ()),
      source.get("route_segment_size_bytes", ()),
      strict=True,
    ))
  ]
  if manifest.get("source_manifest") != expected_segments:
    raise BehaviorReplayAuthorityError(
      f"route {route_id} certification segments differ",
    )
  behavior_eligible = source.get("behavior_eligible")
  behavior_reason = source.get("behavior_ineligible_reason")
  if (
    type(behavior_eligible) is not bool
    or type(behavior_reason) is not str
    or behavior_eligible != (behavior_reason == "eligible")
  ):
    raise BehaviorReplayAuthorityError(
      f"route {route_id} recorded-controller eligibility is malformed",
    )
  segment_results = manifest.get("segment_results")
  if type(segment_results) is not list or not segment_results:
    raise BehaviorReplayAuthorityError(
      f"route {route_id} certification coverage is empty",
    )
  for segment in segment_results:
    behavior = segment.get("behavior_plane") if type(segment) is dict else None
    scenario = segment.get("scenario_plane") if type(segment) is dict else None
    source_hashes = (
      segment.get("encoded_source_plane_sha256")
      if type(segment) is dict
      else None
    )
    if (
      type(source_hashes) is not dict
      or source_hashes.get("route_evidence_complete")
      != route_evidence_sha256
    ):
      raise BehaviorReplayAuthorityError(
        f"route {route_id} certification does not bind its route evidence",
      )
    if (
      type(behavior) is not dict
      or behavior.get("source_eligible") is not behavior_eligible
      or behavior.get("source_eligibility_reason") != behavior_reason
    ):
      raise BehaviorReplayAuthorityError(
        f"route {route_id} recorded-controller certification provenance differs",
      )
    if (
      type(scenario) is not dict
      or type(scenario.get("controls_retained")) is not int
      or scenario["controls_retained"] < 0
      or type(scenario.get("active_controls_retained")) is not int
      or not 0 <= scenario["active_controls_retained"] <= scenario["controls_retained"]
      or type(scenario.get("proof_eligible")) is not bool
      or scenario["proof_eligible"] is not (scenario["controls_retained"] > 0)
    ):
      raise BehaviorReplayAuthorityError(
        f"route {route_id} selected certification segment lacks scenario input proof",
      )
  proof = manifest.get("scenario_proof")
  if type(proof) is not dict or set(proof) != {
    "active_controls_retained",
    "controls_retained",
    "proof_eligible",
    "selected_inputs_sha256",
  }:
    raise BehaviorReplayAuthorityError(f"route {route_id} scenario proof is malformed")
  controls_retained = proof["controls_retained"]
  active_controls_retained = proof["active_controls_retained"]
  if (
    type(controls_retained) is not int
    or controls_retained <= 0
    or type(active_controls_retained) is not int
    or not 0 <= active_controls_retained <= controls_retained
    or proof["proof_eligible"] is not True
  ):
    raise BehaviorReplayAuthorityError(
      f"route {route_id} lacks cert-v5 scenario input proof",
    )
  _sha256(proof["selected_inputs_sha256"], f"route {route_id} selected scenario inputs")


def _validate_imported_object(
  store_root: Path,
  route_id: str,
  artifact: object,
  source: object,
) -> _ImportedRoute:
  if type(artifact) is not dict or set(artifact) != {
    "authorityArtifactIds", "path", "sha256", "sizeBytes",
  }:
    raise BehaviorReplayAuthorityError(f"route {route_id} artifact row is malformed")
  artifact_sha256 = _sha256(artifact["sha256"], f"route {route_id} artifact")
  size_bytes = _uint(
    artifact["sizeBytes"],
    f"route {route_id} artifact size",
    1 << 40,
  )
  authority_ids = artifact["authorityArtifactIds"]
  if type(authority_ids) is not list or len(authority_ids) != 2:
    raise BehaviorReplayAuthorityError(f"route {route_id} authority pair is malformed")
  for index, value in enumerate(authority_ids):
    _sha256(value, f"route {route_id} authority artifact {index + 1}")
  expected_relative = f"objects/{artifact_sha256}.route-evidence"
  if artifact["path"] != expected_relative:
    raise BehaviorReplayAuthorityError(f"route {route_id} object path is not content-addressed")
  relative = PurePosixPath(expected_relative)
  if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
    raise BehaviorReplayAuthorityError(f"route {route_id} object path is unsafe")
  path = store_root.joinpath(*relative.parts)
  _validate_immutable_regular_metadata(
    path,
    f"route {route_id} evidence object",
    size_bytes,
  )
  try:
    summary = inspect_route_evidence_file(path)
  except RouteEvidenceError as error:
    raise BehaviorReplayAuthorityError(
      f"route {route_id} evidence object failed authentication",
    ) from error
  if summary.sha256 != artifact_sha256 or summary.st_size != size_bytes:
    raise BehaviorReplayAuthorityError(f"route {route_id} evidence identity mismatch")
  if type(source) is not dict or canonical_json(source) != canonical_json(_source_row(summary)):
    raise BehaviorReplayAuthorityError(f"route {route_id} source differs from evidence")
  try:
    scenario = behavior_scenario_provenance_from_route_source(
      summary.source_identity,
      summary.sha256,
    )
  except (TypeError, ValueError, RuntimeError) as error:
    raise BehaviorReplayAuthorityError(
      f"route {route_id} is not a comparable controller scenario",
    ) from error
  if scenario.route_id != route_id:
    raise BehaviorReplayAuthorityError(f"route {route_id} source route ID disagrees")
  return _ImportedRoute(
    route_id=route_id,
    object_path=path,
    artifact_sha256=artifact_sha256,
    size_bytes=size_bytes,
    source_row=source,
    scenario=scenario,
  )


def _authenticate_import(store_root: Path, manifest_sha256: str) -> _AuthenticatedImport:
  root = _safe_directory(store_root, "route-evidence store")
  _safe_directory(root / "imports", "route-evidence manifest directory")
  _safe_directory(root / "objects", "route-evidence object directory")
  selected_sha256 = _sha256(manifest_sha256, "import manifest")
  path = root / "imports" / f"{selected_sha256}.json"
  manifest, observed_sha256 = _parse_canonical_file(
    path,
    "import manifest",
    _IMPORT_MANIFEST_MAXIMUM_BYTES,
  )
  if observed_sha256 != selected_sha256:
    raise BehaviorReplayAuthorityError("import manifest content address is invalid")
  if set(manifest) != {
    "importedRouteCount",
    "inspector",
    "jobStateSha256",
    "rejectedRouteCount",
    "remoteWorker",
    "routes",
    "scenarioSourceSetIdentity",
    "schemaVersion",
  } or manifest["schemaVersion"] != TRAINER_IMPORT_MANIFEST_SCHEMA_VERSION:
    raise BehaviorReplayAuthorityError("import manifest shape/version is incompatible")
  inspector = _strict_import_inspector(manifest["inspector"])
  _validate_remote_worker(manifest["remoteWorker"])
  _sha256(manifest["jobStateSha256"], "import job state")
  declared_source_set = _sha256(
    manifest["scenarioSourceSetIdentity"],
    "import scenario source set",
  )
  rows = manifest["routes"]
  if type(rows) is not list or not rows or len(rows) > _MAXIMUM_MANIFEST_ROUTES:
    raise BehaviorReplayAuthorityError("import route population is invalid")
  imported_count = _uint(
    manifest["importedRouteCount"],
    "imported route count",
    _MAXIMUM_MANIFEST_ROUTES,
  )
  rejected_count = _uint(
    manifest["rejectedRouteCount"],
    "rejected route count",
    _MAXIMUM_MANIFEST_ROUTES,
  )
  if imported_count + rejected_count != len(rows):
    raise BehaviorReplayAuthorityError("import route counts disagree")
  route_ids: list[str] = []
  scenario_source_rows: list[dict[str, object]] = []
  imported: list[_ImportedRoute] = []
  observed_rejected = 0
  for row in rows:
    if type(row) is not dict or set(row) != {
      "artifact",
      "archiveContentSha256",
      "rejectionReasons",
      "routeId",
      "source",
      "status",
    }:
      raise BehaviorReplayAuthorityError("import route row is malformed")
    route_id = row["routeId"]
    if type(route_id) is not str or _ROUTE_ID_RE.fullmatch(route_id) is None:
      raise BehaviorReplayAuthorityError("import route ID is malformed")
    route_ids.append(route_id)
    archive_sha256 = row["archiveContentSha256"]
    if archive_sha256 is not None:
      _sha256(archive_sha256, f"route {route_id} archive content")
    reasons = row["rejectionReasons"]
    if type(reasons) is not list or any(
      type(reason) is not str or _REJECTION_RE.fullmatch(reason) is None
      for reason in reasons
    ) or len(set(reasons)) != len(reasons):
      raise BehaviorReplayAuthorityError(f"route {route_id} rejection reasons are malformed")
    if row["status"] == "rejected":
      observed_rejected += 1
      if not reasons or row["artifact"] is not None or row["source"] is not None:
        raise BehaviorReplayAuthorityError(f"route {route_id} rejection row is inconsistent")
      continue
    if row["status"] != "imported" or reasons:
      raise BehaviorReplayAuthorityError(f"route {route_id} import status is invalid")
    if archive_sha256 is None:
      raise BehaviorReplayAuthorityError(f"route {route_id} lacks raw-archive identity")
    artifact = row["artifact"]
    if type(artifact) is not dict or set(artifact) != {
      "authorityArtifactIds", "certificationVector", "path", "sha256", "sizeBytes",
    } or type(row["source"]) is not dict:
      raise BehaviorReplayAuthorityError(f"route {route_id} schema-2 artifact row is malformed")
    authority_ids = artifact["authorityArtifactIds"]
    if type(authority_ids) is not list or len(authority_ids) != 2:
      raise BehaviorReplayAuthorityError(f"route {route_id} authority pair is malformed")
    for index, value in enumerate(authority_ids):
      _sha256(value, f"route {route_id} authority artifact {index + 1}")
    artifact_sha256 = _sha256(
      artifact["sha256"],
      f"route {route_id} route-evidence artifact",
    )
    _validate_certification_vector_descriptor(
      root,
      route_id,
      artifact["certificationVector"],
      row["source"],
      artifact_sha256,
    )
    imported_route = _validate_imported_object(
      root,
      route_id,
      {key: value for key, value in artifact.items() if key != "certificationVector"},
      row["source"],
    )
    imported.append(imported_route)
    scenario_source_rows.append({"routeId": route_id, "source": row["source"]})
  if route_ids != sorted(set(route_ids)):
    raise BehaviorReplayAuthorityError("import routes must be unique and sorted")
  if len(imported) != imported_count or observed_rejected != rejected_count:
    raise BehaviorReplayAuthorityError("observed import dispositions disagree")
  if len(imported) < 2:
    raise BehaviorReplayAuthorityError("authenticated comparison needs multiple routes")
  expected_source_set = _canonical_sha256({
    "domain": "blatv2-trainer-scenario-source-set-v1",
    "routes": scenario_source_rows,
  })
  if declared_source_set != expected_source_set:
    raise BehaviorReplayAuthorityError("import scenario source-set identity differs")
  scenarios = BehaviorScenarioSetIdentity(tuple(route.scenario for route in imported))
  if len({scenario.vehicle_identity for scenario in scenarios.sources}) != 1:
    raise BehaviorReplayAuthorityError("imported scenarios mix vehicle identities")
  return _AuthenticatedImport(
    manifest_sha256=selected_sha256,
    scenario_source_set_identity=declared_source_set,
    inspector_identity_sha256=_canonical_sha256(inspector),
    imported_routes=tuple(imported),
  )


def _load_physical_profile(
  path: Path,
  expected_vehicle_identity: str,
) -> tuple[VehicleCalibrationProfile, str]:
  encoded, _ = _read_immutable_regular(
    path,
    "physical profile",
    _PHYSICAL_PROFILE_MAXIMUM_BYTES,
  )
  try:
    profile = VehicleCalibrationProfile.from_json(
      encoded,
      expected_vehicle_identity,
    )
  except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
    raise BehaviorReplayAuthorityError("physical profile is invalid") from error
  canonical = profile.to_json().encode("utf-8")
  if encoded not in (canonical, canonical + b"\n"):
    raise BehaviorReplayAuthorityError("physical profile is not canonical JSON")
  if not profile.qualified:
    raise BehaviorReplayAuthorityError("physical profile is not fully qualified")
  return profile, hashlib.sha256(canonical).hexdigest()


def _candidate_from_committed_grid(
  candidate: BehaviorPolicy,
  gate_spec: BehaviorGateSpec,
) -> tuple[BehaviorPolicy, str]:
  if not isinstance(candidate, BehaviorPolicy):
    raise TypeError("candidate must be a BehaviorPolicy")
  try:
    controller_seed = ControllerPolicy.from_json_file(
      PROVISIONAL_CONTROLLER_POLICY_PATH,
    )
    center = BehaviorPolicy.from_controller_policy(controller_seed)
    candidates = build_candidate_grid(gate_spec.candidate_grid.policy_grid(center))
  except (OSError, TypeError, ValueError) as error:
    raise BehaviorReplayAuthorityError("committed candidate-grid authority is invalid") from error
  if candidate not in tuple(value.policy for value in candidates):
    raise BehaviorReplayAuthorityError("candidate is outside the committed policy grid")
  return center, controller_seed.sha256


@dataclass(frozen=True, slots=True)
class BehaviorAuthorityRun:
  """Compact deterministic output of one complete authenticated execution."""

  schema_version: int
  import_manifest_sha256: str
  import_scenario_source_set_identity: str
  import_inspector_identity_sha256: str
  replay_source: ReviewedReplaySource
  controller_seed_sha256: str
  search_center_policy: BehaviorPolicy
  candidate_policy: BehaviorPolicy
  aggregate_spec: BehaviorAggregateSpec
  stock_training: BehaviorAggregateEvaluation
  stock_validation: BehaviorAggregateEvaluation
  candidate_training: BehaviorAggregateEvaluation
  candidate_validation: BehaviorAggregateEvaluation
  production_mode: bool

  def __post_init__(self) -> None:
    if self.schema_version != BEHAVIOR_REPLAY_AUTHORITY_SCHEMA_VERSION:
      raise ValueError("behavior authority run schema is incompatible")
    for name, value in (
      ("import manifest", self.import_manifest_sha256),
      ("import source set", self.import_scenario_source_set_identity),
      ("import inspector", self.import_inspector_identity_sha256),
      ("controller seed", self.controller_seed_sha256),
    ):
      _sha256(value, name)
    if not isinstance(self.replay_source, ReviewedReplaySource):
      raise TypeError("behavior authority run requires reviewed source")
    if not isinstance(self.search_center_policy, BehaviorPolicy):
      raise TypeError("behavior authority run requires search-center policy")
    if not isinstance(self.candidate_policy, BehaviorPolicy):
      raise TypeError("behavior authority run requires candidate policy")
    if not isinstance(self.aggregate_spec, BehaviorAggregateSpec):
      raise TypeError("behavior authority run requires aggregate specification")
    if type(self.production_mode) is not bool:
      raise TypeError("behavior authority run production mode must be boolean")
    expected = (
      (self.stock_training, BehaviorRouteSplit.TRAINING, ReplayRole.EXACT_STOCK),
      (self.stock_validation, BehaviorRouteSplit.VALIDATION, ReplayRole.EXACT_STOCK),
      (self.candidate_training, BehaviorRouteSplit.TRAINING, ReplayRole.CANDIDATE),
      (self.candidate_validation, BehaviorRouteSplit.VALIDATION, ReplayRole.CANDIDATE),
    )
    for evaluation, split, role in expected:
      if not isinstance(evaluation, BehaviorAggregateEvaluation):
        raise TypeError("behavior authority run contains an invalid evaluation")
      if (
        evaluation.identity.aggregate_spec_sha256 != self.aggregate_spec.sha256
        or evaluation.identity.split is not split
        or evaluation.identity.replay_artifact.role is not role
      ):
        raise ValueError("behavior authority evaluation identity disagrees")
    if self.stock_training.policy is not None or self.stock_validation.policy is not None:
      raise ValueError("exact stock cannot carry a behavior policy")
    if (
      self.candidate_training.policy != self.candidate_policy
      or self.candidate_validation.policy != self.candidate_policy
    ):
      raise ValueError("candidate evaluations use another policy")

  def to_dict(self) -> dict[str, object]:
    return {
      "activationEligible": False,
      "aggregateSpecSha256": self.aggregate_spec.sha256,
      "candidatePolicy": self.candidate_policy.to_dict(),
      "candidateTrainingEvaluationSha256": self.candidate_training.sha256,
      "candidateValidationEvaluationSha256": self.candidate_validation.sha256,
      "controllerSeedSha256": self.controller_seed_sha256,
      "importInspectorIdentitySha256": self.import_inspector_identity_sha256,
      "importManifestSha256": self.import_manifest_sha256,
      "importScenarioSourceSetIdentity": self.import_scenario_source_set_identity,
      "replaySource": self.replay_source.to_dict(),
      "productionMode": self.production_mode,
      "schemaVersion": self.schema_version,
      "searchCenterPolicy": self.search_center_policy.to_dict(),
      "stockTrainingEvaluationSha256": self.stock_training.sha256,
      "stockValidationEvaluationSha256": self.stock_validation.sha256,
    }

  def to_json(self) -> str:
    return canonical_json(self.to_dict())

  @property
  def sha256(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


class AuthenticatedBehaviorReplayReceipt:
  """Opaque in-process proof of two bit-identical authority executions."""

  __slots__ = ("_first", "_independent_aa_sha256")

  def __init__(
    self,
    *_: object,
    **__: object,
  ) -> None:
    raise TypeError("behavior replay receipts are issued only by production authority")

  @property
  def result(self) -> BehaviorAuthorityRun:
    return self._first

  @property
  def independent_aa_sha256(self) -> str:
    return self._independent_aa_sha256

  def to_dict(self) -> dict[str, object]:
    if (
      not self._first.production_mode
      or self._independent_aa_sha256 != self._first.sha256
    ):
      raise BehaviorReplayAuthorityError("behavior replay receipt invariants are invalid")
    before = _verify_replay_source(self._first.replay_source)
    payload = {
      "activationEligible": False,
      "bitExactIndependentAA": True,
      "independentAASha256": self._independent_aa_sha256,
      "result": self._first.to_dict(),
      "schemaVersion": BEHAVIOR_REPLAY_AUTHORITY_SCHEMA_VERSION,
    }
    after = _verify_replay_source(self._first.replay_source)
    if before != after:
      raise BehaviorReplayAuthorityError("replay source changed during receipt serialization")
    return payload

  @property
  def canonical_sha256(self) -> str:
    return hashlib.sha256(
      b"blatv2-authenticated-behavior-replay-v1\0"
      + canonical_json(self.to_dict()).encode("utf-8"),
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class NonAuthoritativeBehaviorReplayResult:
  """Test-only A/A result which cannot serialize as authority evidence."""

  first: BehaviorAuthorityRun
  independent_aa_sha256: str

  def __post_init__(self) -> None:
    if (
      not isinstance(self.first, BehaviorAuthorityRun)
      or self.independent_aa_sha256 != self.first.sha256
    ):
      raise ValueError("non-authoritative replay result state is inconsistent")

  @property
  def authoritative(self) -> bool:
    return False


def _execute_authority_once_common(
  *,
  evidence_store_root: Path,
  import_manifest_sha256: str,
  physical_profile_path: Path,
  candidate: BehaviorPolicy,
  replay_source: ReviewedReplaySource,
  interface_registry: Mapping[str, type] | None,
  verify_execution_source: bool,
) -> BehaviorAuthorityRun:
  # These are one execution mode, not two independently composable options.
  # In particular, an injected interface registry must never be relabeled as
  # a production run merely by setting the source-verification flag.
  production_mode = interface_registry is None
  if type(verify_execution_source) is not bool or verify_execution_source != production_mode:
    raise BehaviorReplayAuthorityError(
      "replay execution mode and interface authority disagree",
    )
  authenticated = _authenticate_import(evidence_store_root, import_manifest_sha256)
  scenarios = BehaviorScenarioSetIdentity(tuple(
    route.scenario for route in authenticated.imported_routes
  ))
  vehicle_identity = scenarios.sources[0].vehicle_identity
  try:
    gate_spec = load_behavior_gate_spec(BEHAVIOR_GATE_SPEC_PATH)
    segmentation = load_behavior_segmentation_config(
      BEHAVIOR_SEGMENTATION_CONFIG_PATH,
    )
    dynamics = ProvisionalRackDynamics.from_json_file(
      PROVISIONAL_RACK_DYNAMICS_PATH,
    )
  except (OSError, TypeError, ValueError) as error:
    raise BehaviorReplayAuthorityError("committed replay configuration is invalid") from error
  profile, profile_sha256 = _load_physical_profile(
    physical_profile_path,
    vehicle_identity,
  )
  search_center, controller_seed_sha256 = _candidate_from_committed_grid(
    candidate,
    gate_spec,
  )
  try:
    exact_stock_core = reviewed_replay_core_identity(
      exact_stock=True,
      source_openpilot_commit=replay_source.source_openpilot_commit,
      opendbc_commit=replay_source.opendbc_commit,
      panda_commit=replay_source.panda_commit,
    )
    modular_core = reviewed_replay_core_identity(
      exact_stock=False,
      source_openpilot_commit=replay_source.source_openpilot_commit,
      opendbc_commit=replay_source.opendbc_commit,
      panda_commit=replay_source.panda_commit,
    )
    spec = BehaviorAggregateSpec.freeze(
      scenarios=scenarios,
      gate_spec=gate_spec,
      exact_stock_core=exact_stock_core,
      modular_core=modular_core,
      physical_profile_sha256=profile_sha256,
      provisional_dynamics_sha256=dynamics.identity_sha256,
      segmentation_config_sha256=segmentation.sha256,
    )
  except (TypeError, ValueError) as error:
    raise BehaviorReplayAuthorityError(
      "authenticated scenarios cannot form the committed experiment",
    ) from error
  stock_results: dict[BehaviorRouteSplit, list[BehaviorRouteEvaluation]] = {
    BehaviorRouteSplit.TRAINING: [],
    BehaviorRouteSplit.VALIDATION: [],
  }
  candidate_results: dict[BehaviorRouteSplit, list[BehaviorRouteEvaluation]] = {
    BehaviorRouteSplit.TRAINING: [],
    BehaviorRouteSplit.VALIDATION: [],
  }
  split_by_route = {
    assignment.route_id: assignment.split
    for assignment in spec.partition.assignments
  }
  for route in authenticated.imported_routes:
    try:
      before = _verify_replay_source(replay_source) if verify_execution_source else None
      evaluate = (
        evaluate_behavior_route_policies
        if interface_registry is None
        else _evaluate_behavior_route_policies_with_registry_for_test
      )
      keyword: dict[str, object] = {
        "opponent_roles": (ReplayRole.EXACT_STOCK, ReplayRole.CANDIDATE),
        "core_identities": (exact_stock_core, modular_core),
        "segmentation_config": segmentation,
      }
      if interface_registry is not None:
        keyword["interface_registry"] = interface_registry
      preparation, evaluations = evaluate(
        route.object_path,
        None,
        profile,
        dynamics,
        (None, candidate),
        gate_spec.metric_config,
        **keyword,
      )
      after = _verify_replay_source(replay_source) if verify_execution_source else None
      if verify_execution_source and (before != replay_source or after != replay_source):
        raise BehaviorReplayAuthorityError(
          f"route {route.route_id} executing source changed during replay",
        )
      if preparation.scenario != route.scenario:
        raise BehaviorReplayAuthorityError(
          f"route {route.route_id} preparation changed scenario provenance",
        )
      stock, modular = evaluations
    except BehaviorReplayAuthorityError:
      raise
    except (BehaviorRouteEvaluationError, RouteEvidenceError, TypeError, ValueError) as error:
      raise BehaviorReplayAuthorityError(
        f"route {route.route_id} failed authenticated preparation or replay: {error}",
      ) from error
    split = split_by_route[route.route_id]
    stock_results[split].append(stock)
    candidate_results[split].append(modular)
  try:
    stock_training = aggregate_behavior_route_results(
      spec,
      BehaviorRouteSplit.TRAINING,
      stock_results[BehaviorRouteSplit.TRAINING],
      None,
    )
    stock_validation = aggregate_behavior_route_results(
      spec,
      BehaviorRouteSplit.VALIDATION,
      stock_results[BehaviorRouteSplit.VALIDATION],
      None,
    )
    candidate_training = aggregate_behavior_route_results(
      spec,
      BehaviorRouteSplit.TRAINING,
      candidate_results[BehaviorRouteSplit.TRAINING],
      candidate,
    )
    candidate_validation = aggregate_behavior_route_results(
      spec,
      BehaviorRouteSplit.VALIDATION,
      candidate_results[BehaviorRouteSplit.VALIDATION],
      candidate,
    )
  except (TypeError, ValueError) as error:
    raise BehaviorReplayAuthorityError("deterministic route aggregation failed") from error
  return BehaviorAuthorityRun(
    schema_version=BEHAVIOR_REPLAY_AUTHORITY_SCHEMA_VERSION,
    import_manifest_sha256=authenticated.manifest_sha256,
    import_scenario_source_set_identity=(
      authenticated.scenario_source_set_identity
    ),
    import_inspector_identity_sha256=(
      authenticated.inspector_identity_sha256
    ),
    replay_source=replay_source,
    controller_seed_sha256=controller_seed_sha256,
    search_center_policy=search_center,
    candidate_policy=candidate,
    aggregate_spec=spec,
    stock_training=stock_training,
    stock_validation=stock_validation,
    candidate_training=candidate_training,
    candidate_validation=candidate_validation,
    production_mode=production_mode,
  )


def _execute_authority_once(
  *,
  evidence_store_root: Path,
  import_manifest_sha256: str,
  physical_profile_path: Path,
  candidate: BehaviorPolicy,
  replay_source: ReviewedReplaySource,
) -> BehaviorAuthorityRun:
  """Execute one production epoch with no injectable numerical interface."""
  return _execute_authority_once_common(
    evidence_store_root=evidence_store_root,
    import_manifest_sha256=import_manifest_sha256,
    physical_profile_path=physical_profile_path,
    candidate=candidate,
    replay_source=replay_source,
    interface_registry=None,
    verify_execution_source=True,
  )


def _execute_authority_once_for_test(
  *,
  evidence_store_root: Path,
  import_manifest_sha256: str,
  physical_profile_path: Path,
  candidate: BehaviorPolicy,
  replay_source: ReviewedReplaySource,
  interface_registry: Mapping[str, type],
) -> BehaviorAuthorityRun:
  return _execute_authority_once_common(
    evidence_store_root=evidence_store_root,
    import_manifest_sha256=import_manifest_sha256,
    physical_profile_path=physical_profile_path,
    candidate=candidate,
    replay_source=replay_source,
    interface_registry=interface_registry,
    verify_execution_source=False,
  )


def _run_authenticated_behavior_replay(
  *,
  evidence_store_root: Path,
  import_manifest_sha256: str,
  physical_profile_path: Path,
  candidate: BehaviorPolicy,
  replay_source: ReviewedReplaySource,
) -> tuple[BehaviorAuthorityRun, str]:
  if not isinstance(replay_source, ReviewedReplaySource):
    raise TypeError("replay_source must be a ReviewedReplaySource")
  arguments: dict[str, object] = {
    "evidence_store_root": evidence_store_root,
    "import_manifest_sha256": import_manifest_sha256,
    "physical_profile_path": physical_profile_path,
    "candidate": candidate,
    "replay_source": replay_source,
  }
  _verify_replay_source(replay_source)
  first = _execute_authority_once(**arguments)
  _verify_replay_source(replay_source)
  independent = _execute_authority_once(**arguments)
  _verify_replay_source(replay_source)
  if first.to_json() != independent.to_json():
    raise BehaviorReplayAuthorityError(
      "independent A/A replay differs; no receipt was issued",
    )
  return first, independent.sha256


def run_authenticated_behavior_replay(
  *,
  evidence_store_root: Path,
  import_manifest_sha256: str,
  physical_profile_path: Path,
  candidate: BehaviorPolicy,
  replay_source: ReviewedReplaySource,
) -> AuthenticatedBehaviorReplayReceipt:
  """Run the complete production authority with the detected car interface.

  The operation intentionally has no configuration-path, route-list, replay
  callback, aggregate, receipt-token, or interface-registry argument.
  """
  observed_source = _verify_replay_source(replay_source)
  first, independent_sha256 = _run_authenticated_behavior_replay(
    evidence_store_root=evidence_store_root,
    import_manifest_sha256=import_manifest_sha256,
    physical_profile_path=physical_profile_path,
    candidate=candidate,
    replay_source=observed_source,
  )
  _verify_replay_source(observed_source)
  if independent_sha256 != first.sha256:
    raise BehaviorReplayAuthorityError("replay result is not eligible for a receipt")
  # Python object privacy is not a security boundary. The authority is the
  # clean reviewed source and its verified transcript; construction lives here
  # solely to keep the production and injectable test workflows disjoint.
  receipt = object.__new__(AuthenticatedBehaviorReplayReceipt)
  receipt._first = first
  receipt._independent_aa_sha256 = independent_sha256
  receipt.to_dict()
  _verify_replay_source(observed_source)
  return receipt


def _run_authenticated_behavior_replay_with_registry_for_test(
  *,
  evidence_store_root: Path,
  import_manifest_sha256: str,
  physical_profile_path: Path,
  candidate: BehaviorPolicy,
  replay_source: ReviewedReplaySource,
  interface_registry: Mapping[str, type],
) -> NonAuthoritativeBehaviorReplayResult:
  """Exercise A/A with a synthetic port without receipt authority."""
  arguments: dict[str, object] = {
    "evidence_store_root": evidence_store_root,
    "import_manifest_sha256": import_manifest_sha256,
    "physical_profile_path": physical_profile_path,
    "candidate": candidate,
    "replay_source": replay_source,
    "interface_registry": interface_registry,
  }
  first = _execute_authority_once_for_test(**arguments)
  independent = _execute_authority_once_for_test(**arguments)
  if first.to_json() != independent.to_json():
    raise BehaviorReplayAuthorityError("independent test replay A/A differs")
  return NonAuthoritativeBehaviorReplayResult(first, independent.sha256)
