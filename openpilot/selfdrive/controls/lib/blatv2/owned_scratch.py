"""Bounded cleanup for exact process-owned publication scratch.

Recursive pathname deletion is deliberately forbidden here.  A publication
records the inode of the private root when it creates it and supplies the
complete set of file bytes it may contain.  Cleanup opens that exact inode
with ``O_NOFOLLOW``, authenticates the complete shallow inventory, then
unlinks only those regular files through held directory descriptors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat


@dataclass(frozen=True, slots=True)
class OwnedDirectoryIdentity:
  st_dev: int
  st_ino: int


class OwnedScratchError(RuntimeError):
  """Scratch ownership, inventory, or identity changed before cleanup."""


def capture_owned_directory(path: Path) -> OwnedDirectoryIdentity:
  """Capture one private, creator-owned directory before it gains children."""
  try:
    info = path.lstat()
  except OSError as exc:
    raise OwnedScratchError("owned scratch root is unavailable") from exc
  if (
    path.is_symlink()
    or not stat.S_ISDIR(info.st_mode)
    or info.st_uid != os.geteuid()
    or stat.S_IMODE(info.st_mode) != 0o700
  ):
    raise OwnedScratchError("owned scratch root is not private")
  return OwnedDirectoryIdentity(info.st_dev, info.st_ino)


def _safe_relative_file(value: str) -> tuple[str, str | None]:
  parts = value.split("/")
  if (
    len(parts) not in (1, 2)
    or any(not part or part in {".", ".."} for part in parts)
    or any("\0" in part for part in parts)
  ):
    raise OwnedScratchError("owned scratch allowlist path is invalid")
  return parts[-1], None if len(parts) == 1 else parts[0]


def _open_directory_at(name: str, parent_fd: int) -> int:
  return os.open(
    name,
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0),
    dir_fd=parent_fd,
  )


def _bounded_directory_names(
  directory_fd: int,
  maximum: int,
) -> tuple[str, ...]:
  if maximum < 0:
    raise OwnedScratchError("owned scratch directory bound is invalid")
  names: list[str] = []
  try:
    with os.scandir(directory_fd) as entries:
      for entry in entries:
        if len(names) >= maximum:
          raise OwnedScratchError(
            "owned scratch directory exceeds its population bound",
          )
        names.append(entry.name)
  except OwnedScratchError:
    raise
  except OSError as exc:
    raise OwnedScratchError(
      "owned scratch directory inventory is unavailable",
    ) from exc
  return tuple(names)


def _validate_directory(info: os.stat_result, *, private: bool) -> None:
  invalid_mode = (
    stat.S_IMODE(info.st_mode) != 0o700
    if private
    else bool(stat.S_IMODE(info.st_mode) & 0o022)
  )
  if (
    not stat.S_ISDIR(info.st_mode)
    or info.st_uid != os.geteuid()
    or invalid_mode
  ):
    raise OwnedScratchError("owned scratch directory identity is unsafe")


def _validate_file(
  directory_fd: int,
  name: str,
  expected: bytes,
  *,
  maximum_file_bytes: int,
) -> tuple[int, os.stat_result]:
  descriptor = -1
  try:
    before = os.stat(
      name,
      dir_fd=directory_fd,
      follow_symlinks=False,
    )
    descriptor = os.open(
      name,
      os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
      dir_fd=directory_fd,
    )
    opened = os.fstat(descriptor)
    if (
      not stat.S_ISREG(opened.st_mode)
      or opened.st_uid != os.geteuid()
      or stat.S_IMODE(opened.st_mode) & 0o022
      or opened.st_dev != before.st_dev
      or opened.st_ino != before.st_ino
      or opened.st_size != before.st_size
      or opened.st_size != len(expected)
      or opened.st_size > maximum_file_bytes
    ):
      raise OwnedScratchError("owned scratch file identity is unsafe")
    observed = hashlib.sha256()
    while block := os.read(descriptor, 1024 * 1024):
      observed.update(block)
    after = os.fstat(descriptor)
    if (
      after.st_dev != opened.st_dev
      or after.st_ino != opened.st_ino
      or after.st_size != opened.st_size
      or after.st_mtime_ns != opened.st_mtime_ns
      or after.st_ctime_ns != opened.st_ctime_ns
      or observed.digest() != hashlib.sha256(expected).digest()
    ):
      raise OwnedScratchError("owned scratch file changed during validation")
    result = (descriptor, opened)
    descriptor = -1
    return result
  except OwnedScratchError:
    raise
  except OSError as exc:
    raise OwnedScratchError("owned scratch file cannot be opened safely") from exc
  finally:
    if descriptor >= 0:
      os.close(descriptor)


def _unlink_held_file(
  directory_fd: int,
  name: str,
  descriptor: int,
  expected: os.stat_result,
) -> None:
  try:
    rebound = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    held = os.fstat(descriptor)
    if (
      rebound.st_dev != expected.st_dev
      or rebound.st_ino != expected.st_ino
      or rebound.st_size != expected.st_size
      or rebound.st_mtime_ns != expected.st_mtime_ns
      or rebound.st_ctime_ns != expected.st_ctime_ns
      or held.st_dev != expected.st_dev
      or held.st_ino != expected.st_ino
      or held.st_size != expected.st_size
      or held.st_mtime_ns != expected.st_mtime_ns
      or held.st_ctime_ns != expected.st_ctime_ns
    ):
      raise OwnedScratchError("owned scratch file changed before unlink")
    os.unlink(name, dir_fd=directory_fd)
  except OwnedScratchError:
    raise
  except OSError as exc:
    raise OwnedScratchError("owned scratch file unlink failed") from exc


def remove_owned_allowlisted_tree(
  path: Path,
  identity: OwnedDirectoryIdentity,
  allowed_files: Mapping[str, bytes],
  *,
  maximum_file_bytes: int,
  maximum_total_bytes: int,
  require_complete: bool,
) -> None:
  """Remove only an authenticated root with a shallow exact file allowlist.

  The tree may contain direct files and one level of named directories.  When
  ``require_complete`` is false, a failed publication may contain any subset
  of the allowlist, but never an unknown object or a partial byte sequence.
  """
  if (
    maximum_file_bytes < 0
    or maximum_total_bytes < 0
    or len(allowed_files) > 32
  ):
    raise OwnedScratchError("owned scratch cleanup bounds are invalid")
  expected_by_directory: dict[str | None, dict[str, bytes]] = {}
  total_allowed_bytes = 0
  for relative, encoded in allowed_files.items():
    if type(relative) is not str or type(encoded) is not bytes:
      raise OwnedScratchError("owned scratch allowlist is invalid")
    name, directory = _safe_relative_file(relative)
    expected_by_directory.setdefault(directory, {})[name] = encoded
    total_allowed_bytes += len(encoded)
  if total_allowed_bytes > maximum_total_bytes:
    raise OwnedScratchError("owned scratch allowlist exceeds its byte bound")

  parent_fd = -1
  root_fd = -1
  child_fds: dict[str, int] = {}
  held_files: list[tuple[int, str, int, os.stat_result]] = []
  observed_names: dict[str | None, tuple[str, ...]] = {}
  root_validated = False
  try:
    parent_fd = os.open(
      path.parent,
      os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
      | getattr(os, "O_NOFOLLOW", 0),
    )
    parent_info = os.fstat(parent_fd)
    _validate_directory(parent_info, private=False)
    root_fd = os.open(
      path.name,
      os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
      | getattr(os, "O_NOFOLLOW", 0),
      dir_fd=parent_fd,
    )
    opened = os.fstat(root_fd)
    _validate_directory(opened, private=True)
    if opened.st_dev != identity.st_dev or opened.st_ino != identity.st_ino:
      raise OwnedScratchError("owned scratch root inode changed")

    root_files = expected_by_directory.get(None, {})
    expected_root_names = set(root_files) | {
      directory for directory in expected_by_directory if directory is not None
    }
    root_names = _bounded_directory_names(
      root_fd,
      len(expected_root_names) + 1,
    )
    observed_names[None] = root_names
    if (
      not set(root_names).issubset(expected_root_names)
      or (require_complete and set(root_names) != expected_root_names)
    ):
      raise OwnedScratchError("owned scratch root contains an unknown child")

    for directory, expected_files in expected_by_directory.items():
      if directory is None:
        continue
      if directory not in root_names:
        if require_complete:
          raise OwnedScratchError("owned scratch directory is missing")
        continue
      child_fd = _open_directory_at(directory, root_fd)
      child_fds[directory] = child_fd
      child_info = os.fstat(child_fd)
      _validate_directory(child_info, private=False)
      names = _bounded_directory_names(child_fd, len(expected_files) + 1)
      observed_names[directory] = names
      if (
        not set(names).issubset(expected_files)
        or (require_complete and set(names) != set(expected_files))
      ):
        raise OwnedScratchError("owned scratch directory contains an unknown child")

    for name in observed_names[None]:
      if name in child_fds:
        continue
      descriptor, info = _validate_file(
        root_fd,
        name,
        root_files[name],
        maximum_file_bytes=maximum_file_bytes,
      )
      held_files.append((root_fd, name, descriptor, info))
    for directory, child_fd in child_fds.items():
      for name in observed_names[directory]:
        descriptor, info = _validate_file(
          child_fd,
          name,
          expected_by_directory[directory][name],
          maximum_file_bytes=maximum_file_bytes,
        )
        held_files.append((child_fd, name, descriptor, info))

    # Inventory and every byte are authenticated before the first unlink.
    for directory_fd, name, descriptor, info in held_files:
      _unlink_held_file(directory_fd, name, descriptor, info)
    for directory, child_fd in child_fds.items():
      os.fsync(child_fd)
      rebound = os.stat(
        directory,
        dir_fd=root_fd,
        follow_symlinks=False,
      )
      opened_child = os.fstat(child_fd)
      if (
        rebound.st_dev != opened_child.st_dev
        or rebound.st_ino != opened_child.st_ino
      ):
        raise OwnedScratchError("owned scratch child directory changed")
      os.rmdir(directory, dir_fd=root_fd)
    os.fsync(root_fd)
    root_validated = True
  except OwnedScratchError:
    raise
  except OSError as exc:
    raise OwnedScratchError("owned scratch cleanup failed") from exc
  finally:
    for _, _, descriptor, _ in held_files:
      os.close(descriptor)
    for descriptor in child_fds.values():
      os.close(descriptor)
    if root_fd >= 0:
      os.close(root_fd)
    if not root_validated and parent_fd >= 0:
      os.close(parent_fd)
      parent_fd = -1

  try:
    rebound = os.stat(
      path.name,
      dir_fd=parent_fd,
      follow_symlinks=False,
    )
    if rebound.st_dev != identity.st_dev or rebound.st_ino != identity.st_ino:
      raise OwnedScratchError("owned scratch path changed")
    os.rmdir(path.name, dir_fd=parent_fd)
    os.fsync(parent_fd)
  except OwnedScratchError:
    raise
  except OSError as exc:
    raise OwnedScratchError("owned scratch root cleanup failed") from exc
  finally:
    if parent_fd >= 0:
      os.close(parent_fd)
