from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import pickle
import struct

import pytest  # noqa: TID251

from openpilot.selfdrive.controls.lib.blatv2.learning_backfill_spool import (
  MAX_METADATA_BYTES,
  PreparedRouteSpoolDescriptor,
  SPOOL_HEADER_SIZE,
  SPOOL_RECORD_SIZE,
  SpoolFormatError,
  open_prepared_route_spool,
  write_prepared_route_spool,
)
from openpilot.selfdrive.controls.lib.blatv2.learning_runtime import (
  MeasuredLearningFrame,
)


ROUTE_NAME = "000000b7--a6b3b1f175"


def private_directory(tmp_path: Path, name: str = "spool") -> Path:
  directory = tmp_path / name
  directory.mkdir(mode=0o700)
  directory.chmod(0o700)
  return directory


def measured_frame(seed: int = 0) -> MeasuredLearningFrame:
  return MeasuredLearningFrame(
    sample_mono_ns=9_000_000_000 + seed,
    response_mono_ns=8_999_000_000 + seed,
    applied_report_mono_ns=8_999_500_000 + seed,
    applied_effective_mono_ns=8_989_500_000 + seed,
    speed_mps=-0.0 if seed == 0 else 4.25 + seed,
    steering_angle_deg=-456.125 + seed,
    steering_rate_deg_s=4.0 + seed,
    steering_torque=-1.25 + seed,
    steering_pressed=bool(seed & 1),
    standstill=bool(seed & 2),
    steer_fault_temporary=bool(seed & 4),
    steer_fault_permanent=bool(seed & 8),
    can_valid=not bool(seed & 1),
    can_timeout=bool(seed & 2),
    applied_torque=0.125 + seed,
    lateral_active=not bool(seed & 4),
    live_parameters_valid=not bool(seed & 8),
    angle_offset_valid=bool(seed & 1),
    steer_ratio_valid=not bool(seed & 2),
    stiffness_factor_valid=bool(seed & 4),
    angle_offset_deg=-0.25 + seed,
    steer_ratio=13.75 + seed,
    stiffness_factor=1.125 + seed,
    roll_rad=-0.03125 + seed,
    inputs_valid=not bool(seed & 8),
  )


def write_spool(
  directory: Path,
  *,
  frames: tuple[MeasuredLearningFrame, ...] | None = None,
  filename: str = "route.spool",
  abort_requested=lambda: False,
) -> PreparedRouteSpoolDescriptor:
  selected_frames = frames or (measured_frame(0), measured_frame(5))
  return write_prepared_route_spool(
    directory,
    ROUTE_NAME,
    selected_frames,
    controls_witness_count=len(selected_frames) + 3,
    unresolved_witness_count=2,
    gap_count=1,
    provenance={
      "build": "a" * 40,
      "nested": {"policy": "full-rlog", "version": 3},
    },
    max_frames=10,
    abort_requested=abort_requested,
    filename=filename,
  )


def updated_descriptor(
  descriptor: PreparedRouteSpoolDescriptor,
  path: Path,
  **updates,
) -> PreparedRouteSpoolDescriptor:
  encoded = path.read_bytes()
  return replace(
    descriptor,
    size_bytes=len(encoded),
    sha256=hashlib.sha256(encoded).hexdigest(),
    **updates,
  )


def float_bits(value: float) -> bytes:
  return struct.pack("<d", value)


def test_exact_round_trip_preserves_every_field_and_negative_zero(
  tmp_path: Path,
) -> None:
  directory = private_directory(tmp_path)
  original = (measured_frame(0), measured_frame(5))
  descriptor = write_spool(directory, frames=original)
  spool = open_prepared_route_spool(
    directory,
    descriptor,
    expected_route_name=ROUTE_NAME,
    max_frames=10,
  )

  assert descriptor.route == ROUTE_NAME
  assert descriptor.frame_count == 2
  assert descriptor.size_bytes == (
    (directory / descriptor.filename).stat().st_size
  )
  assert spool.route_name == ROUTE_NAME
  assert spool.controls_witness_count == 5
  assert spool.unresolved_witness_count == 2
  assert spool.gap_count == 1
  assert spool.provenance == {
    "build": "a" * 40,
    "nested": {"policy": "full-rlog", "version": 3},
  }

  restored = tuple(spool.iter_frames())
  assert restored == original
  assert tuple(field.name for field in fields(restored[0])) == tuple(
    field.name for field in fields(original[0])
  )
  for expected, actual in zip(original, restored, strict=True):
    for field in fields(expected):
      expected_value = getattr(expected, field.name)
      actual_value = getattr(actual, field.name)
      if type(expected_value) is float:
        assert float_bits(actual_value) == float_bits(expected_value)
      else:
        assert actual_value == expected_value
  assert float_bits(restored[0].speed_mps) == struct.pack("<Q", 1 << 63)
  assert pickle.loads(pickle.dumps(descriptor)) == descriptor


def test_output_bytes_and_hash_are_deterministic(tmp_path: Path) -> None:
  first_directory = private_directory(tmp_path, "first")
  second_directory = private_directory(tmp_path, "second")
  first = write_spool(first_directory)
  second = write_spool(second_directory)

  assert first == second
  assert (
    first_directory / first.filename
  ).read_bytes() == (
    second_directory / second.filename
  ).read_bytes()
  assert first.sha256 == hashlib.sha256(
    (first_directory / first.filename).read_bytes(),
  ).hexdigest()


@pytest.mark.parametrize("mutation", ["corrupt", "truncate", "trailing"])
def test_reader_rejects_corruption_truncation_and_trailing_bytes(
  tmp_path: Path,
  mutation: str,
) -> None:
  directory = private_directory(tmp_path)
  descriptor = write_spool(directory)
  path = directory / descriptor.filename
  encoded = bytearray(path.read_bytes())
  if mutation == "corrupt":
    encoded[-1] ^= 0x01
    path.write_bytes(encoded)
    candidate = descriptor
  elif mutation == "truncate":
    path.write_bytes(encoded[:-1])
    candidate = updated_descriptor(descriptor, path)
  else:
    path.write_bytes(encoded + b"x")
    candidate = updated_descriptor(descriptor, path)

  with pytest.raises(SpoolFormatError):
    open_prepared_route_spool(
      directory,
      candidate,
      expected_route_name=ROUTE_NAME,
      max_frames=10,
    )


def test_iterator_rejects_mutation_after_reader_validation(
  tmp_path: Path,
) -> None:
  directory = private_directory(tmp_path)
  descriptor = write_spool(directory)
  spool = open_prepared_route_spool(
    directory,
    descriptor,
    expected_route_name=ROUTE_NAME,
    max_frames=10,
  )
  path = directory / descriptor.filename
  path.write_bytes(path.read_bytes()[:-1])

  with pytest.raises(SpoolFormatError, match="changed size"):
    tuple(spool)


def test_reader_rejects_wrong_route_and_descriptor_identity(
  tmp_path: Path,
) -> None:
  directory = private_directory(tmp_path)
  descriptor = write_spool(directory)

  with pytest.raises(SpoolFormatError, match="descriptor route mismatch"):
    open_prepared_route_spool(
      directory,
      descriptor,
      expected_route_name="000000ff--ffffffffffff",
      max_frames=10,
    )
  with pytest.raises(SpoolFormatError, match="hash mismatch"):
    open_prepared_route_spool(
      directory,
      replace(descriptor, sha256="0" * 64),
      expected_route_name=ROUTE_NAME,
      max_frames=10,
    )
  with pytest.raises(SpoolFormatError, match="size mismatch"):
    open_prepared_route_spool(
      directory,
      replace(descriptor, size_bytes=descriptor.size_bytes + 1),
      expected_route_name=ROUTE_NAME,
      max_frames=10,
    )


@pytest.mark.parametrize(
  "filename",
  ("../escape", "/absolute", "two/levels", "", ".", "unsafe name"),
)
def test_writer_rejects_unsafe_filename(
  tmp_path: Path,
  filename: str,
) -> None:
  directory = private_directory(tmp_path)
  with pytest.raises(SpoolFormatError, match="filename is unsafe"):
    write_spool(directory, filename=filename)


def test_reader_rejects_symlink(tmp_path: Path) -> None:
  directory = private_directory(tmp_path)
  descriptor = write_spool(directory)
  alias = directory / "alias.spool"
  alias.symlink_to(descriptor.filename)

  with pytest.raises(SpoolFormatError, match="non-symlink"):
    open_prepared_route_spool(
      directory,
      replace(descriptor, filename=alias.name),
      expected_route_name=ROUTE_NAME,
      max_frames=10,
    )


def test_reader_rejects_oversized_count_and_metadata(tmp_path: Path) -> None:
  count_directory = private_directory(tmp_path, "count")
  count_descriptor = write_spool(count_directory)
  count_path = count_directory / count_descriptor.filename
  count_encoded = bytearray(count_path.read_bytes())
  struct.pack_into("<Q", count_encoded, 12, 11)
  count_path.write_bytes(count_encoded)
  count_descriptor = updated_descriptor(
    count_descriptor,
    count_path,
    frame_count=11,
  )
  with pytest.raises(SpoolFormatError, match="frame count exceeds"):
    open_prepared_route_spool(
      count_directory,
      count_descriptor,
      expected_route_name=ROUTE_NAME,
      max_frames=10,
    )

  metadata_directory = private_directory(tmp_path, "metadata")
  metadata_descriptor = write_spool(metadata_directory)
  metadata_path = metadata_directory / metadata_descriptor.filename
  metadata_encoded = bytearray(metadata_path.read_bytes())
  struct.pack_into("<I", metadata_encoded, 20, MAX_METADATA_BYTES + 1)
  metadata_path.write_bytes(metadata_encoded)
  metadata_descriptor = updated_descriptor(
    metadata_descriptor,
    metadata_path,
  )
  with pytest.raises(SpoolFormatError, match="metadata exceeds"):
    open_prepared_route_spool(
      metadata_directory,
      metadata_descriptor,
      expected_route_name=ROUTE_NAME,
      max_frames=10,
    )


def test_reader_rejects_noncanonical_metadata(tmp_path: Path) -> None:
  directory = private_directory(tmp_path)
  descriptor = write_spool(directory)
  path = directory / descriptor.filename
  encoded = bytearray(path.read_bytes())
  metadata_size = struct.unpack_from("<I", encoded, 20)[0]
  metadata_end = SPOOL_HEADER_SIZE + metadata_size
  metadata = json.loads(encoded[SPOOL_HEADER_SIZE:metadata_end])
  noncanonical = json.dumps(
    {key: metadata[key] for key in reversed(metadata)},
    separators=(",", ":"),
  ).encode()
  assert len(noncanonical) == metadata_size
  encoded[SPOOL_HEADER_SIZE:metadata_end] = noncanonical
  path.write_bytes(encoded)
  descriptor = updated_descriptor(descriptor, path)

  with pytest.raises(SpoolFormatError, match="not canonical"):
    open_prepared_route_spool(
      directory,
      descriptor,
      expected_route_name=ROUTE_NAME,
      max_frames=10,
    )


def test_writer_cancellation_removes_partial_and_preserves_published_file(
  tmp_path: Path,
) -> None:
  directory = private_directory(tmp_path)
  descriptor = write_spool(directory)
  original = (directory / descriptor.filename).read_bytes()
  calls = 0

  def abort_requested() -> bool:
    nonlocal calls
    calls += 1
    return calls >= 4

  with pytest.raises(SpoolFormatError, match="cancelled"):
    write_spool(
      directory,
      frames=tuple(measured_frame(index) for index in range(8)),
      abort_requested=abort_requested,
    )

  assert (directory / descriptor.filename).read_bytes() == original
  assert tuple(directory.glob("*.partial")) == ()
  assert tuple(directory.glob(".*.partial")) == ()


def test_cleanup_deletes_only_exact_descriptor_file(tmp_path: Path) -> None:
  directory = private_directory(tmp_path)
  descriptor = write_spool(directory)
  spool = open_prepared_route_spool(
    directory,
    descriptor,
    expected_route_name=ROUTE_NAME,
    max_frames=10,
  )
  other = directory / "other.spool"
  other.write_bytes(b"keep")
  spool.cleanup()
  assert not (directory / descriptor.filename).exists()
  assert other.read_bytes() == b"keep"

  stale = write_spool(directory)
  replacement = write_spool(
    directory,
    frames=(measured_frame(7),),
  )
  assert stale.filename == replacement.filename
  with pytest.raises(SpoolFormatError, match="identity"):
    stale.cleanup(directory)
  assert (directory / replacement.filename).exists()


def test_writer_enforces_tuple_frame_bound_and_private_directory(
  tmp_path: Path,
) -> None:
  directory = private_directory(tmp_path)
  with pytest.raises(TypeError, match="must be a tuple"):
    write_prepared_route_spool(
      directory,
      ROUTE_NAME,
      [measured_frame(0)],  # type: ignore[arg-type]
      controls_witness_count=1,
      unresolved_witness_count=0,
      gap_count=0,
      provenance={},
      max_frames=1,
      abort_requested=lambda: False,
    )
  with pytest.raises(SpoolFormatError, match="frame count exceeds"):
    write_prepared_route_spool(
      directory,
      ROUTE_NAME,
      (measured_frame(0), measured_frame(1)),
      controls_witness_count=2,
      unresolved_witness_count=0,
      gap_count=0,
      provenance={},
      max_frames=1,
      abort_requested=lambda: False,
    )

  public_directory = tmp_path / "public"
  public_directory.mkdir(mode=0o755)
  public_directory.chmod(0o755)
  with pytest.raises(SpoolFormatError, match="not private"):
    write_spool(public_directory)


def test_record_contract_is_fixed_little_endian_layout() -> None:
  assert SPOOL_RECORD_SIZE == (4 * 8) + (9 * 8) + 12
