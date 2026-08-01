"""Immutable, offroad-only persistence for behavior-learning generations.

This module is deliberately a terminal evidence store, not an activation
mechanism.  It cannot write Params, approve a controller, or affect live
actuation.  One generation records the exact physical authority, committed
configuration files, route population, source build, independent A/A
transaction, finalization, and (only when qualified) selected behavior policy.

Publication uses a content-addressed immutable directory followed by one
atomically replaced ``CURRENT`` pointer.  Every externally supplied identity
is authenticated again while loading; a directory is accepted only when its
file inventory is exact and every byte is covered by ``commit.json``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any

from openpilot.selfdrive.controls.lib.blatv2.behavior_configuration import (
  BEHAVIOR_GATE_SPEC_PATH,
  BEHAVIOR_SEGMENTATION_CONFIG_PATH,
  load_behavior_gate_spec,
  load_behavior_segmentation_config,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_coordinator import (
  BehaviorGateSpec,
  BehaviorLearningFinalization,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_evidence import (
  BehaviorSourceIdentity,
  canonical_json,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_policy import BehaviorPolicy
from openpilot.selfdrive.controls.lib.blatv2.behavior_segmentation import (
  SegmentationConfig,
)
from openpilot.selfdrive.controls.lib.blatv2.behavior_transaction import (
  BehaviorLearningTransactionResult,
)


BEHAVIOR_GENERATION_SCHEMA_VERSION = 1
BEHAVIOR_GENERATION_POINTER_SCHEMA_VERSION = 1
BEHAVIOR_ROUTE_SET_SCHEMA_VERSION = 1
MAX_BEHAVIOR_GENERATION_FILE_BYTES = 32 * 1024 * 1024

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GENERATION_FILES = frozenset((
  "commit.json",
  "finalization.json",
  "gate_spec.json",
  "recorded_source.json",
  "route_evidence_set.json",
  "segmentation_config.json",
  "transaction.json",
))
_POLICY_FILE = "policy.json"
_COMMIT_KEYS = frozenset((
  "artifactFileSha256s",
  "finalizationSha256",
  "gateSpecFileSha256",
  "gateSpecSha256",
  "physicalGenerationSha256",
  "physicalProfileSha256",
  "recordedSourceIdentitySha256",
  "routeEvidenceSetSha256",
  "schemaVersion",
  "segmentationConfigFileSha256",
  "segmentationConfigSha256",
  "selectedPolicySha256",
  "transactionSha256",
))
_POINTER_KEYS = frozenset(("generationSha256", "schemaVersion"))
_ROUTE_SET_KEYS = frozenset(("routes", "schemaVersion"))
_ROUTE_KEYS = frozenset(("routeId", "sha256"))
_SOURCE_KEYS = frozenset((
  "controllerArtifactSha256",
  "controllerName",
  "evidenceSchemaVersion",
  "opendbcCommit",
  "pandaCommit",
  "sourceOpenpilotCommit",
))
_POLICY_KEYS = frozenset(("dampingRatio", "naturalFrequencyPerS"))


class BehaviorGenerationError(RuntimeError):
  """A generation is unsafe, nondeterministic, corrupt, or inconsistent."""


@dataclass(frozen=True, slots=True)
class LoadedBehaviorGeneration:
  """Fully authenticated behavior evidence; never an activation object."""

  generation_sha256: str
  physical_generation_sha256: str
  physical_profile_sha256: str
  gate_spec: BehaviorGateSpec
  segmentation_config: SegmentationConfig
  transaction: BehaviorLearningTransactionResult
  finalization: BehaviorLearningFinalization
  route_evidence_sha256s: tuple[tuple[str, str], ...]
  recorded_source: BehaviorSourceIdentity
  selected_policy: BehaviorPolicy | None

  @property
  def stock_retained(self) -> bool:
    return self.selected_policy is None


def _sha256(encoded: bytes) -> str:
  return hashlib.sha256(encoded).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
  return canonical_json(payload).encode("utf-8")


def _strict_sha256(value: object, name: str) -> str:
  if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
    raise BehaviorGenerationError(f"{name} must be a lowercase SHA-256")
  return value


def _strict_object(
  value: object,
  expected_keys: frozenset[str],
  name: str,
) -> dict[str, Any]:
  if type(value) is not dict or frozenset(value) != expected_keys:
    raise BehaviorGenerationError(f"{name} keys do not match the schema")
  return value


def _decode_canonical_object(
  encoded: bytes,
  expected_keys: frozenset[str],
  name: str,
) -> dict[str, Any]:
  try:
    payload = json.loads(encoded)
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise BehaviorGenerationError(f"{name} is not valid JSON") from exc
  root = _strict_object(payload, expected_keys, name)
  if encoded != _canonical_bytes(root):
    raise BehaviorGenerationError(f"{name} is not canonical JSON")
  return root


def _safe_directory(path: Path, *, create: bool) -> Path:
  try:
    info = path.lstat()
  except FileNotFoundError:
    if not create:
      raise BehaviorGenerationError(
        f"required directory is absent: {path.name}",
      ) from None
    try:
      path.mkdir(mode=0o700)
      info = path.lstat()
    except OSError as exc:
      raise BehaviorGenerationError(f"cannot create directory: {path.name}") from exc
  except OSError as exc:
    raise BehaviorGenerationError(f"cannot inspect directory: {path.name}") from exc
  if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
    raise BehaviorGenerationError(f"unsafe directory type: {path.name}")
  return path


def _safe_root(path: str | Path, *, create: bool) -> Path:
  root = Path(path)
  if not root.is_absolute():
    root = root.absolute()
  if create:
    # The caller owns the parent artifact directory.  Requiring it to exist
    # prevents a recursive mkdir from silently traversing symlinked parents.
    _safe_directory(root.parent, create=False)
  return _safe_directory(root, create=create)


def _read_regular_file(path: Path) -> bytes:
  try:
    before = path.lstat()
  except OSError as exc:
    raise BehaviorGenerationError(f"required artifact is unavailable: {path.name}") from exc
  if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
    raise BehaviorGenerationError(f"unsafe artifact type: {path.name}")
  if before.st_size > MAX_BEHAVIOR_GENERATION_FILE_BYTES:
    raise BehaviorGenerationError(f"artifact exceeds size bound: {path.name}")
  flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
  try:
    descriptor = os.open(path, flags)
  except OSError as exc:
    raise BehaviorGenerationError(f"cannot safely open artifact: {path.name}") from exc
  try:
    opened = os.fstat(descriptor)
    if (
      not stat.S_ISREG(opened.st_mode)
      or opened.st_dev != before.st_dev
      or opened.st_ino != before.st_ino
      or opened.st_size != before.st_size
    ):
      raise BehaviorGenerationError(f"artifact changed while opening: {path.name}")
    chunks: list[bytes] = []
    remaining = opened.st_size
    while remaining:
      chunk = os.read(descriptor, min(remaining, 1024 * 1024))
      if not chunk:
        raise BehaviorGenerationError(f"artifact was truncated: {path.name}")
      chunks.append(chunk)
      remaining -= len(chunk)
    if os.read(descriptor, 1):
      raise BehaviorGenerationError(f"artifact grew while reading: {path.name}")
    return b"".join(chunks)
  finally:
    os.close(descriptor)


def _fsync_directory(path: Path) -> None:
  descriptor = os.open(
    path,
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
  )
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)


def _write_fsynced(path: Path, encoded: bytes) -> None:
  descriptor = os.open(
    path,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
    0o600,
  )
  try:
    view = memoryview(encoded)
    while view:
      count = os.write(descriptor, view)
      if count <= 0:
        raise OSError("short behavior-generation write")
      view = view[count:]
    os.fsync(descriptor)
  finally:
    os.close(descriptor)


def _publication_guard(
  abort_requested: Callable[[], bool],
  offroad_confirmed: Callable[[], bool],
  phase: str,
) -> None:
  try:
    aborted = abort_requested()
    offroad = offroad_confirmed()
  except BaseException as exc:
    raise BehaviorGenerationError(f"publication guard failed {phase}") from exc
  if type(aborted) is not bool or type(offroad) is not bool:
    raise BehaviorGenerationError("publication guards must return booleans")
  if aborted:
    raise BehaviorGenerationError(f"behavior publication aborted {phase}")
  if not offroad:
    raise BehaviorGenerationError(f"behavior publication requires offroad state {phase}")


def _parse_gate_spec(path: Path, encoded: bytes) -> BehaviorGateSpec:
  try:
    result = load_behavior_gate_spec(path)
  except (OSError, UnicodeDecodeError, ValueError) as exc:
    raise BehaviorGenerationError("behavior gate spec is invalid") from exc
  if _read_regular_file(path) != encoded:
    raise BehaviorGenerationError("behavior gate spec changed while reading")
  if encoded != (result.to_json() + "\n").encode("utf-8"):
    raise BehaviorGenerationError("behavior gate spec bytes are not canonical")
  return result


def _parse_segmentation_config(path: Path, encoded: bytes) -> SegmentationConfig:
  try:
    result = load_behavior_segmentation_config(path)
  except (OSError, UnicodeDecodeError, ValueError) as exc:
    raise BehaviorGenerationError("behavior segmentation config is invalid") from exc
  if _read_regular_file(path) != encoded:
    raise BehaviorGenerationError("behavior segmentation config changed while reading")
  expected = (canonical_json(result.to_dict()) + "\n").encode("utf-8")
  if encoded != expected:
    raise BehaviorGenerationError("behavior segmentation config bytes are not canonical")
  return result


def _parse_transaction(encoded: bytes) -> BehaviorLearningTransactionResult:
  try:
    text = encoded.decode("utf-8")
    result = BehaviorLearningTransactionResult.from_json(text)
  except (UnicodeDecodeError, ValueError) as exc:
    raise BehaviorGenerationError("behavior transaction is invalid") from exc
  if result.to_json().encode("utf-8") != encoded:
    raise BehaviorGenerationError("behavior transaction is not canonical")
  return result


def _parse_finalization(encoded: bytes) -> BehaviorLearningFinalization:
  try:
    text = encoded.decode("utf-8")
    result = BehaviorLearningFinalization.from_json(text)
  except (UnicodeDecodeError, ValueError) as exc:
    raise BehaviorGenerationError("behavior finalization is invalid") from exc
  if result.to_json().encode("utf-8") != encoded:
    raise BehaviorGenerationError("behavior finalization is not canonical")
  return result


def _parse_policy(encoded: bytes) -> BehaviorPolicy:
  root = _decode_canonical_object(encoded, _POLICY_KEYS, "behavior policy")
  if type(root["naturalFrequencyPerS"]) not in (int, float) or type(
    root["dampingRatio"],
  ) not in (int, float):
    raise BehaviorGenerationError("behavior policy values must be numeric")
  try:
    result = BehaviorPolicy(
      natural_frequency_per_s=float(root["naturalFrequencyPerS"]),
      damping_ratio=float(root["dampingRatio"]),
    )
  except (TypeError, ValueError, OverflowError) as exc:
    raise BehaviorGenerationError("behavior policy values are invalid") from exc
  if result.to_json().encode("utf-8") != encoded:
    raise BehaviorGenerationError("behavior policy is not canonical")
  return result


def _parse_source(encoded: bytes) -> BehaviorSourceIdentity:
  root = _decode_canonical_object(encoded, _SOURCE_KEYS, "recorded source")
  text_keys = _SOURCE_KEYS - {"evidenceSchemaVersion"}
  if any(type(root[key]) is not str for key in text_keys):
    raise BehaviorGenerationError("recorded source text fields are invalid")
  if type(root["evidenceSchemaVersion"]) is not int:
    raise BehaviorGenerationError("recorded source schema version must be integer")
  try:
    result = BehaviorSourceIdentity(
      controller_name=root["controllerName"],
      controller_artifact_sha256=root["controllerArtifactSha256"],
      source_openpilot_commit=root["sourceOpenpilotCommit"],
      opendbc_commit=root["opendbcCommit"],
      panda_commit=root["pandaCommit"],
      evidence_schema_version=root["evidenceSchemaVersion"],
    )
  except (TypeError, ValueError) as exc:
    raise BehaviorGenerationError("recorded source identity is invalid") from exc
  if result.to_json().encode("utf-8") != encoded:
    raise BehaviorGenerationError("recorded source identity is not canonical")
  return result


def _route_set_bytes(
  route_evidence_sha256s: tuple[tuple[str, str], ...],
) -> bytes:
  return _canonical_bytes({
    "routes": [
      {"routeId": route_id, "sha256": digest}
      for route_id, digest in route_evidence_sha256s
    ],
    "schemaVersion": BEHAVIOR_ROUTE_SET_SCHEMA_VERSION,
  })


def _parse_route_set(encoded: bytes) -> tuple[tuple[str, str], ...]:
  root = _decode_canonical_object(encoded, _ROUTE_SET_KEYS, "route evidence set")
  if root["schemaVersion"] != BEHAVIOR_ROUTE_SET_SCHEMA_VERSION or type(
    root["schemaVersion"],
  ) is not int:
    raise BehaviorGenerationError("route evidence set schema is incompatible")
  values = root["routes"]
  if type(values) is not list:
    raise BehaviorGenerationError("route evidence set routes must be an array")
  routes: list[tuple[str, str]] = []
  for value in values:
    route = _strict_object(value, _ROUTE_KEYS, "route evidence identity")
    if type(route["routeId"]) is not str or not route["routeId"].strip():
      raise BehaviorGenerationError("route evidence ID must be non-empty text")
    routes.append((route["routeId"], _strict_sha256(route["sha256"], "route evidence")))
  result = tuple(routes)
  route_ids = tuple(route_id for route_id, _ in result)
  if not result or route_ids != tuple(sorted(set(route_ids))):
    raise BehaviorGenerationError("route evidence set must be non-empty, unique, and sorted")
  return result


def _validate_transaction_bindings(
  *,
  transaction: BehaviorLearningTransactionResult,
  physical_profile_sha256: str,
  gate_spec: BehaviorGateSpec,
  segmentation_config: SegmentationConfig,
  recorded_source: BehaviorSourceIdentity,
) -> None:
  if transaction.physical_profile_sha256 != physical_profile_sha256:
    raise BehaviorGenerationError("physical profile identity mismatch")
  if transaction.segmentation_config_sha256 != segmentation_config.sha256:
    raise BehaviorGenerationError("segmentation configuration identity mismatch")
  if transaction.finalization.gate_spec_sha256 != gate_spec.sha256:
    raise BehaviorGenerationError("behavior gate-spec identity mismatch")
  final_source = transaction.finalization.recorded_source_identity_sha256
  if final_source is None:
    raise BehaviorGenerationError(
      "behavior finalization does not expose a recorded source identity",
    )
  if final_source != recorded_source.sha256:
    raise BehaviorGenerationError("recorded source identity mismatch")


def _commit_bytes(
  *,
  physical_generation_sha256: str,
  physical_profile_sha256: str,
  gate_spec: BehaviorGateSpec,
  segmentation_config: SegmentationConfig,
  transaction: BehaviorLearningTransactionResult,
  recorded_source: BehaviorSourceIdentity,
  selected_policy: BehaviorPolicy | None,
  artifact_files: Mapping[str, bytes],
) -> bytes:
  route_set = artifact_files["route_evidence_set.json"]
  return _canonical_bytes({
    "artifactFileSha256s": {
      name: _sha256(encoded)
      for name, encoded in sorted(artifact_files.items())
    },
    "finalizationSha256": transaction.finalization.sha256,
    "gateSpecFileSha256": _sha256(artifact_files["gate_spec.json"]),
    "gateSpecSha256": gate_spec.sha256,
    "physicalGenerationSha256": physical_generation_sha256,
    "physicalProfileSha256": physical_profile_sha256,
    "recordedSourceIdentitySha256": recorded_source.sha256,
    "routeEvidenceSetSha256": _sha256(route_set),
    "schemaVersion": BEHAVIOR_GENERATION_SCHEMA_VERSION,
    "segmentationConfigFileSha256": _sha256(
      artifact_files["segmentation_config.json"],
    ),
    "segmentationConfigSha256": segmentation_config.sha256,
    "selectedPolicySha256": (
      None if selected_policy is None else selected_policy.sha256
    ),
    "transactionSha256": transaction.sha256,
  })


def _existing_generation_matches(
  generation: Path,
  expected_files: Mapping[str, bytes],
) -> bool:
  try:
    _safe_directory(generation, create=False)
    if frozenset(entry.name for entry in generation.iterdir()) != frozenset(expected_files):
      return False
    return all(
      _read_regular_file(generation / name) == encoded
      for name, encoded in expected_files.items()
    )
  except (BehaviorGenerationError, OSError):
    return False


def _atomic_replace_pointer(
  path: Path,
  encoded: bytes,
  *,
  abort_requested: Callable[[], bool],
  offroad_confirmed: Callable[[], bool],
) -> None:
  if path.exists() or path.is_symlink():
    _read_regular_file(path)
  _publication_guard(abort_requested, offroad_confirmed, "before pointer staging")
  descriptor, temporary_name = tempfile.mkstemp(
    dir=path.parent,
    prefix=f".{path.name}.",
    suffix=".tmp",
  )
  temporary = Path(temporary_name)
  try:
    os.fchmod(descriptor, 0o600)
    view = memoryview(encoded)
    while view:
      count = os.write(descriptor, view)
      if count <= 0:
        raise OSError("short CURRENT write")
      view = view[count:]
    os.fsync(descriptor)
    os.close(descriptor)
    descriptor = -1
    _publication_guard(abort_requested, offroad_confirmed, "before pointer publication")
    os.replace(temporary, path)
    _fsync_directory(path.parent)
  finally:
    if descriptor >= 0:
      os.close(descriptor)
    try:
      temporary.unlink()
    except FileNotFoundError:
      pass


def publish_behavior_generation(
  *,
  behavior_root: str | Path,
  first_authority: BehaviorLearningTransactionResult,
  second_authority: BehaviorLearningTransactionResult,
  physical_generation_sha256: str,
  physical_profile_sha256: str,
  recorded_source: BehaviorSourceIdentity,
  abort_requested: Callable[[], bool],
  offroad_confirmed: Callable[[], bool],
  gate_spec_path: str | Path = BEHAVIOR_GATE_SPEC_PATH,
  segmentation_config_path: str | Path = BEHAVIOR_SEGMENTATION_CONFIG_PATH,
) -> str:
  """Publish one A/A-verified behavior report and return its generation SHA.

  A failed qualification is intentionally publishable: it records that exact
  evidence retained stock and omits ``policy.json``.  Nothing here can promote
  even a successful policy into a live controller.
  """
  if not isinstance(first_authority, BehaviorLearningTransactionResult) or not isinstance(
    second_authority,
    BehaviorLearningTransactionResult,
  ):
    raise BehaviorGenerationError("both behavior authorities must be transaction results")
  first_bytes = first_authority.to_json().encode("utf-8")
  second_bytes = second_authority.to_json().encode("utf-8")
  if first_bytes != second_bytes:
    raise BehaviorGenerationError("independent behavior authorities are not byte-identical")
  transaction = _parse_transaction(first_bytes)
  physical_generation = _strict_sha256(
    physical_generation_sha256,
    "physical generation identity",
  )
  physical_profile = _strict_sha256(
    physical_profile_sha256,
    "physical profile identity",
  )
  if not isinstance(recorded_source, BehaviorSourceIdentity):
    raise BehaviorGenerationError("recorded source must be an exact source identity")

  gate_path = Path(gate_spec_path)
  segmentation_path = Path(segmentation_config_path)
  gate_bytes = _read_regular_file(gate_path)
  segmentation_bytes = _read_regular_file(segmentation_path)
  gate_spec = _parse_gate_spec(gate_path, gate_bytes)
  segmentation_config = _parse_segmentation_config(
    segmentation_path,
    segmentation_bytes,
  )
  _validate_transaction_bindings(
    transaction=transaction,
    physical_profile_sha256=physical_profile,
    gate_spec=gate_spec,
    segmentation_config=segmentation_config,
    recorded_source=recorded_source,
  )

  selected_policy = transaction.selected_policy
  route_set = _route_set_bytes(transaction.route_evidence_sha256s)
  artifact_files: dict[str, bytes] = {
    "finalization.json": transaction.finalization.to_json().encode("utf-8"),
    "gate_spec.json": gate_bytes,
    "recorded_source.json": recorded_source.to_json().encode("utf-8"),
    "route_evidence_set.json": route_set,
    "segmentation_config.json": segmentation_bytes,
    "transaction.json": first_bytes,
  }
  if selected_policy is not None:
    artifact_files[_POLICY_FILE] = selected_policy.to_json().encode("utf-8")
  commit = _commit_bytes(
    physical_generation_sha256=physical_generation,
    physical_profile_sha256=physical_profile,
    gate_spec=gate_spec,
    segmentation_config=segmentation_config,
    transaction=transaction,
    recorded_source=recorded_source,
    selected_policy=selected_policy,
    artifact_files=artifact_files,
  )
  generation_sha256 = _sha256(commit)
  artifact_files["commit.json"] = commit

  _publication_guard(abort_requested, offroad_confirmed, "before staging")
  root = _safe_root(behavior_root, create=True)
  generations = _safe_directory(root / "generations", create=True)
  staging: Path | None = Path(tempfile.mkdtemp(
    dir=generations,
    prefix=".staging-",
  ))
  try:
    assert staging is not None
    for name, encoded in sorted(artifact_files.items()):
      _write_fsynced(staging / name, encoded)
    _fsync_directory(staging)
    _publication_guard(abort_requested, offroad_confirmed, "before generation publication")
    generation = generations / generation_sha256
    try:
      if generation.exists() or generation.is_symlink():
        if not _existing_generation_matches(generation, artifact_files):
          raise BehaviorGenerationError(
            "behavior generation identity collision is not byte-identical",
          )
        shutil.rmtree(staging)
        staging = None
      else:
        os.rename(staging, generation)
        staging = None
    except OSError as exc:
      if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
        raise
      if not _existing_generation_matches(generation, artifact_files):
        raise BehaviorGenerationError(
          "behavior generation identity collision is not byte-identical",
        ) from exc
      shutil.rmtree(staging)
      staging = None
    _fsync_directory(generations)
    pointer = _canonical_bytes({
      "generationSha256": generation_sha256,
      "schemaVersion": BEHAVIOR_GENERATION_POINTER_SCHEMA_VERSION,
    })
    _atomic_replace_pointer(
      root / "CURRENT",
      pointer,
      abort_requested=abort_requested,
      offroad_confirmed=offroad_confirmed,
    )
    _fsync_directory(root)
  except BaseException as exc:
    if staging is not None and staging.exists():
      shutil.rmtree(staging)
    if isinstance(exc, BehaviorGenerationError):
      raise
    raise BehaviorGenerationError("immutable behavior publication failed") from exc
  return generation_sha256


def _parse_commit(encoded: bytes) -> dict[str, Any]:
  root = _decode_canonical_object(encoded, _COMMIT_KEYS, "behavior generation commit")
  if type(root["schemaVersion"]) is not int or root[
    "schemaVersion"
  ] != BEHAVIOR_GENERATION_SCHEMA_VERSION:
    raise BehaviorGenerationError("behavior generation schema is incompatible")
  for key in _COMMIT_KEYS - {
    "artifactFileSha256s",
    "schemaVersion",
    "selectedPolicySha256",
  }:
    _strict_sha256(root[key], key)
  if root["selectedPolicySha256"] is not None:
    _strict_sha256(root["selectedPolicySha256"], "selected policy identity")
  hashes = root["artifactFileSha256s"]
  if type(hashes) is not dict or any(type(name) is not str for name in hashes):
    raise BehaviorGenerationError("artifact file hash inventory is invalid")
  for name, digest in hashes.items():
    if name == "commit.json" or name.startswith(".") or "/" in name:
      raise BehaviorGenerationError("artifact file inventory contains an unsafe name")
    _strict_sha256(digest, f"artifact file {name}")
  return root


def load_behavior_generation(
  behavior_root: str | Path,
  generation_sha256: str,
) -> LoadedBehaviorGeneration:
  """Authenticate one immutable generation without granting activation."""
  identity = _strict_sha256(generation_sha256, "behavior generation identity")
  root = _safe_root(behavior_root, create=False)
  generations = _safe_directory(root / "generations", create=False)
  generation = _safe_directory(generations / identity, create=False)
  commit_bytes = _read_regular_file(generation / "commit.json")
  if _sha256(commit_bytes) != identity:
    raise BehaviorGenerationError("behavior generation directory identity mismatch")
  commit = _parse_commit(commit_bytes)
  selected_sha = commit["selectedPolicySha256"]
  expected_names = set(_GENERATION_FILES)
  if selected_sha is not None:
    expected_names.add(_POLICY_FILE)
  try:
    actual_names = frozenset(entry.name for entry in generation.iterdir())
  except OSError as exc:
    raise BehaviorGenerationError("behavior generation inventory is unavailable") from exc
  if actual_names != frozenset(expected_names):
    raise BehaviorGenerationError("behavior generation file inventory is not exact")
  artifact_hashes = commit["artifactFileSha256s"]
  expected_artifact_names = frozenset(expected_names - {"commit.json"})
  if frozenset(artifact_hashes) != expected_artifact_names:
    raise BehaviorGenerationError("committed artifact hash inventory is not exact")
  artifacts = {
    name: _read_regular_file(generation / name)
    for name in sorted(expected_artifact_names)
  }
  for name, encoded in artifacts.items():
    if _sha256(encoded) != artifact_hashes[name]:
      raise BehaviorGenerationError(f"artifact hash mismatch: {name}")

  transaction = _parse_transaction(artifacts["transaction.json"])
  finalization = _parse_finalization(artifacts["finalization.json"])
  gate_spec = _parse_gate_spec(generation / "gate_spec.json", artifacts["gate_spec.json"])
  segmentation = _parse_segmentation_config(
    generation / "segmentation_config.json",
    artifacts["segmentation_config.json"],
  )
  recorded_source = _parse_source(artifacts["recorded_source.json"])
  routes = _parse_route_set(artifacts["route_evidence_set.json"])
  selected_policy = (
    None if selected_sha is None else _parse_policy(artifacts[_POLICY_FILE])
  )

  if finalization != transaction.finalization:
    raise BehaviorGenerationError("finalization disagrees with transaction")
  if routes != transaction.route_evidence_sha256s:
    raise BehaviorGenerationError("route evidence set disagrees with transaction")
  if selected_policy != transaction.selected_policy:
    raise BehaviorGenerationError("selected policy disagrees with transaction")
  physical_profile = _strict_sha256(
    commit["physicalProfileSha256"],
    "physical profile identity",
  )
  _validate_transaction_bindings(
    transaction=transaction,
    physical_profile_sha256=physical_profile,
    gate_spec=gate_spec,
    segmentation_config=segmentation,
    recorded_source=recorded_source,
  )
  checks = (
    (commit["transactionSha256"], transaction.sha256, "transaction"),
    (commit["finalizationSha256"], finalization.sha256, "finalization"),
    (commit["gateSpecSha256"], gate_spec.sha256, "gate spec"),
    (commit["gateSpecFileSha256"], _sha256(artifacts["gate_spec.json"]), "gate spec file"),
    (commit["segmentationConfigSha256"], segmentation.sha256, "segmentation config"),
    (
      commit["segmentationConfigFileSha256"],
      _sha256(artifacts["segmentation_config.json"]),
      "segmentation config file",
    ),
    (
      commit["recordedSourceIdentitySha256"],
      recorded_source.sha256,
      "recorded source",
    ),
    (
      commit["routeEvidenceSetSha256"],
      _sha256(artifacts["route_evidence_set.json"]),
      "route evidence set",
    ),
    (
      selected_sha,
      None if selected_policy is None else selected_policy.sha256,
      "selected policy",
    ),
  )
  for committed, computed, name in checks:
    if committed != computed:
      raise BehaviorGenerationError(f"{name} identity mismatch")
  return LoadedBehaviorGeneration(
    generation_sha256=identity,
    physical_generation_sha256=_strict_sha256(
      commit["physicalGenerationSha256"],
      "physical generation identity",
    ),
    physical_profile_sha256=physical_profile,
    gate_spec=gate_spec,
    segmentation_config=segmentation,
    transaction=transaction,
    finalization=finalization,
    route_evidence_sha256s=routes,
    recorded_source=recorded_source,
    selected_policy=selected_policy,
  )


def load_current_behavior_generation(
  behavior_root: str | Path,
) -> LoadedBehaviorGeneration:
  """Resolve the canonical pointer once, then authenticate that generation."""
  root = _safe_root(behavior_root, create=False)
  pointer = _read_regular_file(root / "CURRENT")
  payload = _decode_canonical_object(pointer, _POINTER_KEYS, "behavior CURRENT")
  if type(payload["schemaVersion"]) is not int or payload[
    "schemaVersion"
  ] != BEHAVIOR_GENERATION_POINTER_SCHEMA_VERSION:
    raise BehaviorGenerationError("behavior CURRENT schema is incompatible")
  identity = _strict_sha256(payload["generationSha256"], "CURRENT generation")
  return load_behavior_generation(root, identity)
