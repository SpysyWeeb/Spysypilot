from __future__ import annotations

from pathlib import Path

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_spool import (
  PreparedRouteSpoolDescriptor,
  SpoolFormatError,
  open_prepared_route_spool,
  write_prepared_route_spool,
)
from openpilot.selfdrive.controls.tests.test_blatv2_route_evidence import (
  FRAMES,
  artifact,
)


ROUTE = "000000b7--a6b3b1f175"


def write(root: Path, *, filename: str | None = None) -> PreparedRouteSpoolDescriptor:
  evidence = artifact()
  return write_prepared_route_spool(
    root, ROUTE, FRAMES,
    controls_witness_count=evidence.source_identity.controls_witness_count,
    unresolved_witness_count=evidence.source_identity.unresolved_witness_count,
    gap_count=evidence.source_identity.gap_count,
    provenance=evidence.source_identity.preparation_provenance,
    max_frames=10, abort_requested=lambda: False, filename=filename,
    route_evidence=evidence,
  )


def test_spool_is_exact_full_route_evidence_and_iterates_physical(tmp_path: Path) -> None:
  descriptor = write(tmp_path)
  evidence = artifact()
  assert (tmp_path / descriptor.filename).read_bytes() == evidence.canonical_bytes
  opened = open_prepared_route_spool(
    tmp_path, descriptor, expected_route_name=ROUTE, max_frames=10,
  )
  assert opened.route_evidence.canonical_bytes == evidence.canonical_bytes
  assert tuple(opened.iter_frames()) == FRAMES
  assert opened.provenance == evidence.source_identity.preparation_provenance


def test_physical_only_v1_writer_and_reader_fail_closed(tmp_path: Path) -> None:
  with pytest.raises(SpoolFormatError, match="complete shared"):
    write_prepared_route_spool(
      tmp_path, ROUTE, FRAMES, controls_witness_count=2,
      unresolved_witness_count=0, gap_count=0, provenance={}, max_frames=10,
      abort_requested=lambda: False,
    )
  old = tmp_path / "old.blatspool"
  old.write_bytes(b"BLATSP01" + b"\0" * 100)
  descriptor = PreparedRouteSpoolDescriptor(
    route_name=ROUTE, filename=old.name,
    sha256=__import__("hashlib").sha256(old.read_bytes()).hexdigest(),
    size_bytes=old.stat().st_size, frame_count=2,
  )
  with pytest.raises(SpoolFormatError, match="unsupported"):
    open_prepared_route_spool(
      tmp_path, descriptor, expected_route_name=ROUTE, max_frames=10,
    )


def test_writer_rejects_mismatched_facts_and_abort(tmp_path: Path) -> None:
  evidence = artifact()
  with pytest.raises(SpoolFormatError, match="facts disagree"):
    write_prepared_route_spool(
      tmp_path, ROUTE, FRAMES, controls_witness_count=3,
      unresolved_witness_count=0, gap_count=0,
      provenance=evidence.source_identity.preparation_provenance,
      max_frames=10, abort_requested=lambda: False, route_evidence=evidence,
    )
  with pytest.raises(SpoolFormatError, match="cancelled"):
    write_prepared_route_spool(
      tmp_path, ROUTE, FRAMES,
      controls_witness_count=evidence.source_identity.controls_witness_count,
      unresolved_witness_count=0, gap_count=0,
      provenance=evidence.source_identity.preparation_provenance,
      max_frames=10, abort_requested=lambda: True, route_evidence=evidence,
    )


def test_open_and_iteration_detect_corruption(tmp_path: Path) -> None:
  descriptor = write(tmp_path)
  path = tmp_path / descriptor.filename
  encoded = bytearray(path.read_bytes())
  encoded[-1] ^= 1
  path.write_bytes(encoded)
  with pytest.raises(SpoolFormatError, match="identity"):
    open_prepared_route_spool(
      tmp_path, descriptor, expected_route_name=ROUTE, max_frames=10,
    )

  descriptor = write(tmp_path, filename="second.route-evidence")
  opened = open_prepared_route_spool(
    tmp_path, descriptor, expected_route_name=ROUTE, max_frames=10,
  )
  path = tmp_path / descriptor.filename
  path.write_bytes(path.read_bytes()[:-1] + b"x")
  with pytest.raises(SpoolFormatError, match="changed"):
    tuple(opened.iter_frames())


def test_cleanup_is_identity_checked(tmp_path: Path) -> None:
  descriptor = write(tmp_path)
  descriptor.cleanup(tmp_path)
  assert not (tmp_path / descriptor.filename).exists()


def test_symlink_directory_and_file_are_rejected(tmp_path: Path) -> None:
  target = tmp_path / "target"
  target.mkdir(mode=0o700)
  linked = tmp_path / "linked"
  linked.symlink_to(target, target_is_directory=True)
  with pytest.raises(SpoolFormatError, match="private"):
    write(linked)

  descriptor = write(target)
  outside = tmp_path / "outside"
  outside.write_bytes((target / descriptor.filename).read_bytes())
  (target / descriptor.filename).unlink()
  (target / descriptor.filename).symlink_to(outside)
  with pytest.raises(SpoolFormatError, match="regular"):
    open_prepared_route_spool(
      target, descriptor, expected_route_name=ROUTE, max_frames=10,
    )
