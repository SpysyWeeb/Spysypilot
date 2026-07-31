from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import pytest  # noqa: TID251
import zstandard as zstd

from openpilot.cereal import log
from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.controls.lib.blatv2 import learning_backfill
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  MAXIMUM_EVENT_BYTES,
  BackfillError,
  RouteRejected,
  extract_segment_events,
)


NATIVE_EXTRACTOR = (
  Path(BASEDIR)
  / "openpilot"
  / "selfdrive"
  / "controls"
  / "blatv2_rlog_extract"
)

# Generated once with the reviewed historical schemas pinned by descriptor:
#   superproject 02bf07c412b4ae92889c7a977fec61328a300c66
#   log.capnp blob 5de6084958601c76ded50380097da4ee0b213a43
#   opendbc 7ab2f7f85f3f9acb6167b7a4d472ea513aa27609
# It contains InitData, start, an unselected DeviceState, old CarParams,
# ControlsState, and end. Keeping the old wire bytes here makes schema
# compatibility independent of the current Python builder.
REVIEWED_LEGACY_RLOG_SHA256 = (
  "20c484c1d6a7f287681b66835f470361bb19d7ba04c0b0b7d830aa399bc55190"
)
REVIEWED_LEGACY_RLOG_BASE64 = "".join((
  "AAAAACYAAAAAAAAAAgABAGQAAAAAAAAAAAAAAAAAAAAAAAAAAgAVAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAYQAAAHIAAABlAAAAmgAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMQAAAEoBAAAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwMmJmMDdjNDEyYjRhZTky",
  "ODg5YzdhOTc3ZmVjNjEzMjhhMzAwYzY2AAAAAAAAAABsZWdhY3ktZGV2aWNlAAAA",
  "bGVnYWN5LXJldmlld2VkLXYxAAAAAAAAAAAAAAUAAAAAAAAAAgABAG4AAAAAAAAA",
  "RwAAAAAAAAAAAAAAAQAAAAMAAAAAAAAAAAAAAB4AAAAAAAAAAgABAHMAAAAAAAAA",
  "BQAAAAAAAAAAAAAADwALAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJAAAAAAAAAACAAEA",
  "dgAAAAAAAABDAAAAAAAAAAAAAAASAA4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAiAAAA",
  "AAAAAAIAAQB4AAAAAAAAAAYAAAAAAAAAAAAAABgABgAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABQAAAAAAAAACAAEA",
  "ggAAAAAAAABHAAAAAAAAAAAAAAABAAAAAQAAAAAAAAA=",
))


@pytest.fixture(scope="module")
def native_extractor() -> Path:
  assert NATIVE_EXTRACTOR.is_file(), (
    "native extractor is not built; run "
    + "`scons openpilot/selfdrive/controls/blatv2_rlog_extract`"
  )
  assert os.access(NATIVE_EXTRACTOR, os.X_OK)
  return NATIVE_EXTRACTOR


def reviewed_legacy_rlog() -> bytes:
  encoded = base64.b64decode(REVIEWED_LEGACY_RLOG_BASE64)
  assert hashlib.sha256(encoded).hexdigest() == REVIEWED_LEGACY_RLOG_SHA256
  return encoded


def independent_capnp_messages(encoded: bytes) -> tuple[bytes, ...]:
  """Split flat-array messages without using the native implementation."""
  messages: list[bytes] = []
  offset = 0
  while offset < len(encoded):
    assert len(encoded) - offset >= 8
    segment_count = struct.unpack_from("<I", encoded, offset)[0] + 1
    assert 0 < segment_count <= 512
    table_u32 = 1 + segment_count
    padded_table_u32 = (table_u32 + 1) & ~1
    table_bytes = padded_table_u32 * 4
    assert len(encoded) - offset >= table_bytes
    content_words = sum(
      struct.unpack_from("<I", encoded, offset + 4 * (1 + index))[0]
      for index in range(segment_count)
    )
    message_bytes = table_bytes + content_words * 8
    assert message_bytes > 0
    assert offset + message_bytes <= len(encoded)
    messages.append(encoded[offset:offset + message_bytes])
    offset += message_bytes
  assert offset == len(encoded)
  return tuple(messages)


def independently_selected(
  encoded: bytes,
) -> tuple[tuple[int, int, bytes], ...]:
  selected = []
  for message in independent_capnp_messages(encoded):
    with log.Event.from_bytes(
      message,
      traversal_limit_in_words=MAXIMUM_EVENT_BYTES // 8,
      nesting_limit=64,
    ) as event:
      which = event.which()
      if which in learning_backfill._EVENT_WHICH:
        selected.append((
          learning_backfill._EVENT_WHICH[which],
          int(event.logMonoTime),
          message,
        ))
  return tuple(selected)


@pytest.mark.parametrize("compressed", (False, True))
def test_native_matches_independent_pycapnp_on_reviewed_legacy_fixture(
  native_extractor: Path,
  tmp_path: Path,
  compressed: bool,
) -> None:
  raw = reviewed_legacy_rlog()
  messages = independent_capnp_messages(raw)
  with log.Event.from_bytes(messages[0]) as init_event:
    assert init_event.which() == "initData"
    assert (
      init_event.initData.gitCommit
      == "02bf07c412b4ae92889c7a977fec61328a300c66"
    )
    assert init_event.initData.version == "legacy-reviewed-v1"
  segment = tmp_path / ("rlog.zst" if compressed else "rlog")
  segment.write_bytes(
    zstd.ZstdCompressor(level=1).compress(raw)
    if compressed
    else raw
  )

  extracted = extract_segment_events(native_extractor, segment)

  assert tuple(
    (record.which, record.mono_ns, record.encoded)
    for record in extracted
  ) == independently_selected(raw)
  assert [record.ordinal for record in extracted] == list(
    range(len(extracted)),
  )
  assert len(extracted) == 5


def current_event(which: str, mono_ns: int) -> bytes:
  event = log.Event.new_message(valid=True, logMonoTime=mono_ns)
  event.init(which)
  return event.to_bytes()


@pytest.mark.parametrize(
  "corrupt",
  (
    lambda raw: raw[:-1],
    lambda raw: raw + b"\x00",
    lambda raw: raw + current_event("deviceState", 140),
  ),
  ids=("truncated", "trailing-byte", "unselected-event-after-end"),
)
def test_native_rejects_raw_truncation_and_post_terminal_data(
  native_extractor: Path,
  tmp_path: Path,
  corrupt,
) -> None:
  segment = tmp_path / "rlog"
  segment.write_bytes(corrupt(reviewed_legacy_rlog()))

  with pytest.raises(RouteRejected):
    extract_segment_events(native_extractor, segment)


@pytest.mark.parametrize(
  "tail",
  (
    lambda compressor: compressor.compress(b""),
    lambda _compressor: b"\x50\x2a\x4d\x18\x00\x00\x00\x00",
    lambda _compressor: b"trailing-garbage",
  ),
  ids=("concatenated-empty-frame", "skippable-frame", "garbage"),
)
def test_native_rejects_every_byte_after_single_zstd_frame(
  native_extractor: Path,
  tmp_path: Path,
  tail,
) -> None:
  compressor = zstd.ZstdCompressor(level=1)
  frame = compressor.compress(reviewed_legacy_rlog())
  segment = tmp_path / "rlog.zst"
  segment.write_bytes(frame + tail(compressor))

  with pytest.raises(RouteRejected):
    extract_segment_events(native_extractor, segment)


def test_native_rejects_truncated_zstd(
  native_extractor: Path,
  tmp_path: Path,
) -> None:
  frame = zstd.ZstdCompressor(level=1).compress(reviewed_legacy_rlog())
  segment = tmp_path / "rlog.zst"
  segment.write_bytes(frame[:-1])

  with pytest.raises(RouteRejected):
    extract_segment_events(native_extractor, segment)


def run_native(
  native_extractor: Path,
  segment: Path,
) -> subprocess.CompletedProcess[bytes]:
  return subprocess.run(
    (str(native_extractor), str(segment)),
    stdin=subprocess.DEVNULL,
    capture_output=True,
    check=False,
    timeout=5,
  )


def test_native_enforces_message_and_zstd_window_bounds(
  native_extractor: Path,
  tmp_path: Path,
) -> None:
  oversized_message = tmp_path / "oversized-rlog"
  oversized_message.write_bytes(struct.pack(
    "<II",
    0,
    MAXIMUM_EVENT_BYTES // 8,
  ))
  message_result = run_native(native_extractor, oversized_message)
  assert message_result.returncode != 0
  assert b"message exceeds size bound" in message_result.stderr

  normal_frame = bytearray(
    zstd.ZstdCompressor(
      level=1,
      write_content_size=False,
    ).compress(b"not-capnp"),
  )
  assert normal_frame[:5] == b"\x28\xb5\x2f\xfd\x00"
  # Window descriptor exponent 28 requests 256 MiB, beyond the helper's
  # explicitly configured 128 MiB decoder maximum.
  normal_frame[5] = 0x90
  oversized_window = tmp_path / "oversized-window.zst"
  oversized_window.write_bytes(normal_frame)
  window_result = run_native(native_extractor, oversized_window)
  assert window_result.returncode != 0
  assert (
    b"window" in window_result.stderr.lower()
    or b"memory" in window_result.stderr.lower()
  )


def fake_extractor(
  tmp_path: Path,
  output: bytes,
  *,
  exit_code: int = 0,
  error: bytes = b"",
) -> Path:
  executable = tmp_path / (
    f"fake-extractor-{hashlib.sha256(output + error).hexdigest()[:12]}"
  )
  executable.write_text(
    f"""#!{sys.executable}
import base64
import sys
sys.stdout.buffer.write(base64.b64decode({base64.b64encode(output)!r}))
sys.stdout.buffer.flush()
sys.stderr.buffer.write(base64.b64decode({base64.b64encode(error)!r}))
raise SystemExit({exit_code})
""",
  )
  executable.chmod(0o755)
  return executable


@pytest.mark.parametrize(
  ("output", "exit_code", "error", "reason"),
  (
    (
      learning_backfill._STREAM_HEADER.pack(
        learning_backfill._STREAM_MAGIC,
        learning_backfill.NATIVE_EXTRACTOR_SCHEMA_VERSION,
        0,
      )
      + learning_backfill._RECORD_HEADER.pack(
        0,
        learning_backfill._END_RECORD,
        1,
      ),
      0,
      b"",
      "extractor_trailer_mismatch",
    ),
    (
      learning_backfill._STREAM_HEADER.pack(
        learning_backfill._STREAM_MAGIC,
        learning_backfill.NATIVE_EXTRACTOR_SCHEMA_VERSION,
        0,
      )
      + learning_backfill._RECORD_HEADER.pack(
        0,
        learning_backfill._END_RECORD,
        0,
      )
      + b"x",
      0,
      b"",
      "extractor_trailing_output",
    ),
    (
      learning_backfill._STREAM_HEADER.pack(
        learning_backfill._STREAM_MAGIC,
        learning_backfill.NATIVE_EXTRACTOR_SCHEMA_VERSION,
        0,
      )
      + learning_backfill._RECORD_HEADER.pack(
        0,
        learning_backfill._END_RECORD,
        0,
      ),
      9,
      b"injected native failure",
      "extractor_failed",
    ),
    (
      learning_backfill._STREAM_HEADER.pack(
        learning_backfill._STREAM_MAGIC,
        learning_backfill.NATIVE_EXTRACTOR_SCHEMA_VERSION,
        0,
      )
      + b"x",
      0,
      b"",
      "extractor_truncated",
    ),
    (
      learning_backfill._STREAM_HEADER.pack(
        learning_backfill._STREAM_MAGIC,
        learning_backfill.NATIVE_EXTRACTOR_SCHEMA_VERSION,
        0,
      )
      + learning_backfill._RECORD_HEADER.pack(
        MAXIMUM_EVENT_BYTES + 1,
        0,
        1,
      ),
      0,
      b"",
      "event_too_large",
    ),
  ),
  ids=(
    "trailer-count",
    "trailing-output",
    "nonzero-exit-and-stderr",
    "truncated-record",
    "event-size-bound",
  ),
)
def test_python_protocol_requires_finite_verified_trailer_and_exit(
  tmp_path: Path,
  output: bytes,
  exit_code: int,
  error: bytes,
  reason: str,
) -> None:
  segment = tmp_path / "segment"
  segment.write_bytes(b"unused")
  extractor = fake_extractor(
    tmp_path,
    output,
    exit_code=exit_code,
    error=error,
  )

  with pytest.raises(RouteRejected) as raised:
    extract_segment_events(extractor, segment)

  assert raised.value.reason == reason


def test_python_protocol_enforces_selected_record_bound(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  header = learning_backfill._STREAM_HEADER.pack(
    learning_backfill._STREAM_MAGIC,
    learning_backfill.NATIVE_EXTRACTOR_SCHEMA_VERSION,
    0,
  )
  record = learning_backfill._RECORD_HEADER.pack(1, 0, 1) + b"x"
  trailer = learning_backfill._RECORD_HEADER.pack(
    0,
    learning_backfill._END_RECORD,
    2,
  )
  extractor = fake_extractor(tmp_path, header + record + record + trailer)
  segment = tmp_path / "segment"
  segment.write_bytes(b"unused")
  monkeypatch.setattr(
    learning_backfill,
    "MAXIMUM_SELECTED_RECORDS_PER_SEGMENT",
    1,
  )

  with pytest.raises(RouteRejected) as raised:
    extract_segment_events(extractor, segment)

  assert raised.value.reason == "extractor_output_too_large"


def test_cancellation_kills_and_reaps_extractor(
  tmp_path: Path,
) -> None:
  segment = tmp_path / "segment"
  segment.write_bytes(b"unused")
  extractor = tmp_path / "hanging-extractor"
  header = learning_backfill._STREAM_HEADER.pack(
    learning_backfill._STREAM_MAGIC,
    learning_backfill.NATIVE_EXTRACTOR_SCHEMA_VERSION,
    0,
  )
  extractor.write_text(
    f"""#!{sys.executable}
import base64
from pathlib import Path
import os
import sys
import time
Path(sys.argv[1] + ".pid").write_text(str(os.getpid()))
sys.stdout.buffer.write(base64.b64decode({base64.b64encode(header)!r}))
sys.stdout.buffer.flush()
time.sleep(60)
""",
  )
  extractor.chmod(0o755)
  abort_checks = 0

  def abort_requested() -> bool:
    nonlocal abort_checks
    abort_checks += 1
    return abort_checks >= 3

  started = time.monotonic()
  with pytest.raises(BackfillError) as raised:
    extract_segment_events(
      extractor,
      segment,
      abort_requested=abort_requested,
    )
  elapsed = time.monotonic() - started

  assert raised.value.diagnostic == "unexpected_error"
  assert elapsed < 5
  pid = int(Path(f"{segment}.pid").read_text())
  with pytest.raises(ProcessLookupError):
    os.kill(pid, 0)
