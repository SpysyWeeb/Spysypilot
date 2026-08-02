from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openpilot.selfdrive.controls.lib.blatv2 import owned_scratch
from openpilot.selfdrive.controls.lib.blatv2.owned_scratch import (
  OwnedScratchError,
  capture_owned_directory,
  remove_owned_allowlisted_tree,
)


def write_publication_staging(
  root: Path,
  files: dict[str, bytes],
) -> None:
  for relative, encoded in files.items():
    path = root / relative
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(encoded)


def remove(root: Path, files: dict[str, bytes], *, complete: bool) -> None:
  identity = capture_owned_directory(root)
  remove_owned_allowlisted_tree(
    root,
    identity,
    files,
    maximum_file_bytes=1024,
    maximum_total_bytes=4096,
    require_complete=complete,
  )


class TestOwnedPublicationScratch(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)

  def tearDown(self) -> None:
    self.temporary.cleanup()

  def test_cleanup_removes_exact_shallow_manifest(self) -> None:
    staging = self.root / "staging"
    staging.mkdir(mode=0o700)
    files = {
      "commit.json": b"commit",
      "evidence.json": b"evidence",
      "selected_profiles/abc.json": b"profile",
    }
    write_publication_staging(staging, files)

    remove(staging, files, complete=True)

    self.assertFalse(staging.exists())

  def test_cleanup_preserves_unknown_nested_child(self) -> None:
    staging = self.root / "staging"
    staging.mkdir(mode=0o700)
    files = {"commit.json": b"commit"}
    write_publication_staging(staging, files)
    sentinel = staging / "foreign" / "must-survive"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"not publication-owned")
    identity = capture_owned_directory(staging)

    with self.assertRaisesRegex(OwnedScratchError, "unknown child"):
      remove_owned_allowlisted_tree(
        staging,
        identity,
        files,
        maximum_file_bytes=1024,
        maximum_total_bytes=4096,
        require_complete=True,
      )

    self.assertEqual(sentinel.read_bytes(), b"not publication-owned")
    self.assertEqual((staging / "commit.json").read_bytes(), b"commit")

  def test_path_swap_preserves_replacement_sentinel(self) -> None:
    staging = self.root / "staging"
    staging.mkdir(mode=0o700)
    files = {"commit.json": b"commit"}
    write_publication_staging(staging, files)
    identity = capture_owned_directory(staging)
    displaced = self.root / "displaced-owned-staging"
    staging.rename(displaced)
    staging.mkdir(mode=0o700)
    sentinel = staging / "must-survive"
    sentinel.write_bytes(b"replacement")

    with self.assertRaisesRegex(OwnedScratchError, "inode changed"):
      remove_owned_allowlisted_tree(
        staging,
        identity,
        files,
        maximum_file_bytes=1024,
        maximum_total_bytes=4096,
        require_complete=True,
      )

    self.assertEqual(sentinel.read_bytes(), b"replacement")
    self.assertEqual((displaced / "commit.json").read_bytes(), b"commit")

  def test_final_root_replacement_is_not_removed(self) -> None:
    staging = self.root / "staging"
    staging.mkdir(mode=0o700)
    files = {"commit.json": b"commit"}
    write_publication_staging(staging, files)
    identity = capture_owned_directory(staging)
    displaced = self.root / "displaced-after-validation"
    real_stat = os.stat
    swapped = False

    def swap_before_final_stat(path, *args, **kwargs):
      nonlocal swapped
      if (
        not swapped
        and path == staging.name
        and kwargs.get("dir_fd") is not None
      ):
        swapped = True
        staging.rename(displaced)
        staging.mkdir(mode=0o700)
        (staging / "must-survive").write_bytes(b"replacement")
      return real_stat(path, *args, **kwargs)

    with (
      patch.object(owned_scratch.os, "stat", swap_before_final_stat),
      self.assertRaisesRegex(OwnedScratchError, "path changed"),
    ):
      remove_owned_allowlisted_tree(
        staging,
        identity,
        files,
        maximum_file_bytes=1024,
        maximum_total_bytes=4096,
        require_complete=True,
      )

    self.assertEqual(
      (staging / "must-survive").read_bytes(),
      b"replacement",
    )
    self.assertTrue(displaced.is_dir())


if __name__ == "__main__":
  unittest.main()
