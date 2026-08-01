from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

import pytest  # noqa: TID251
import zstandard as zstd

from openpilot.cereal import log
from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.controls.lib.blatv2 import learning_backfill
from openpilot.selfdrive.controls.lib.blatv2.learning_backfill import (
  MAXIMUM_EVENT_BYTES,
  BackfillError,
  RouteSegment,
  RouteRejected,
  extract_segment_events,
  open_verified_extractor,
  verify_open_extractor,
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

  extractor = open_verified_extractor(native_extractor)
  descriptor = os.open(segment, os.O_RDONLY | os.O_NOFOLLOW)
  try:
    extracted = extract_segment_events(
      native_extractor,
      segment,
      extractor_fd=extractor.descriptor,
      segment_fd=descriptor,
    )
    verify_open_extractor(extractor)
  finally:
    os.close(descriptor)
    os.close(extractor.descriptor)

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


def test_native_retains_complete_shared_evidence_input_set(
  native_extractor: Path,
  tmp_path: Path,
) -> None:
  """The shared preparation pass owns both learning evidence planes."""
  raw = b"".join((
    current_event("modelV2", 100),
    current_event("selfdriveState", 110),
    current_event("liveTorqueParameters", 120),
    current_event("liveDelay", 130),
    current_event("lateralManeuverPlan", 140),
    current_event("drivingEvent", 150),
  ))
  segment = tmp_path / "behavior-rlog"
  segment.write_bytes(raw)

  extracted = extract_segment_events(native_extractor, segment)

  assert tuple(
    (record.which, record.mono_ns, record.encoded)
    for record in extracted
  ) == independently_selected(raw)
  assert [record.which for record in extracted] == [
    learning_backfill._EVENT_WHICH["modelV2"],
    learning_backfill._EVENT_WHICH["selfdriveState"],
    learning_backfill._EVENT_WHICH["liveTorqueParameters"],
    learning_backfill._EVENT_WHICH["liveDelay"],
    learning_backfill._EVENT_WHICH["lateralManeuverPlan"],
    learning_backfill._EVENT_WHICH["drivingEvent"],
  ]


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


def test_held_segment_fd_closes_child_open_toctou(
  tmp_path: Path,
) -> None:
  """The child consumes the inode already hashed, never a swapped pathname."""
  segment_path = tmp_path / "rlog.zst"
  saved_path = tmp_path / "rlog.saved"
  ready_path = tmp_path / "extractor.ready"
  go_path = tmp_path / "extractor.go"
  read_path = tmp_path / "extractor.read"
  expected = b"expected immutable route bytes"
  alternate = b"alternate bytes consumed only by the vulnerable path"
  segment_path.write_bytes(expected)
  segment = RouteSegment(
    index=0,
    path=segment_path,
    sha256=hashlib.sha256(expected).hexdigest(),
    size_bytes=len(expected),
  )
  extractor = tmp_path / "race-extractor"
  extractor.write_text(
    f"""#!{sys.executable}
from pathlib import Path
import struct
import sys
import time
ready = Path({str(ready_path)!r})
go = Path({str(go_path)!r})
read = Path({str(read_path)!r})
ready.touch()
while not go.exists():
  time.sleep(0.001)
payload = Path(sys.argv[1]).read_bytes()
read.touch()
out = sys.stdout.buffer
out.write(struct.pack("<8sII", b"BLATV2R1", {learning_backfill.NATIVE_EXTRACTOR_SCHEMA_VERSION}, 0))
out.write(struct.pack("<IIQ", len(payload), 1, 123))
out.write(payload)
out.write(struct.pack("<IIQ", 0, 0xffffffff, 1))
out.flush()
""",
  )
  extractor.chmod(0o755)
  descriptor, opened_stat = learning_backfill._open_verified_route_segment(
    segment,
    abort_requested=lambda: False,
  )
  swap_error: list[BaseException] = []

  def swap_path_during_child_open() -> None:
    try:
      assert ready_path.exists() or _wait_for_path(ready_path)
      segment_path.rename(saved_path)
      segment_path.write_bytes(alternate)
      go_path.touch()
      assert read_path.exists() or _wait_for_path(read_path)
      segment_path.unlink()
      saved_path.rename(segment_path)
    except BaseException as exc:
      swap_error.append(exc)
      go_path.touch()

  thread = threading.Thread(target=swap_path_during_child_open)
  thread.start()
  try:
    records = extract_segment_events(
      extractor,
      segment_path,
      segment_fd=descriptor,
    )
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not swap_error
    assert records[0].encoded == expected
    assert records[0].encoded != alternate
    # Even though the child read the correct held inode, any path mutation is
    # rejected rather than silently accepted as an immutable route snapshot.
    with pytest.raises(RouteRejected) as raised:
      learning_backfill._verify_open_route_segment(
        segment,
        descriptor,
        opened_stat,
        abort_requested=lambda: False,
      )
    assert raised.value.reason == "segment_changed"
  finally:
    go_path.touch()
    thread.join(timeout=5)
    os.close(descriptor)


def test_held_extractor_fd_closes_child_exec_toctou(
  tmp_path: Path,
) -> None:
  """A timed pathname swap cannot execute bytes outside the held hash."""
  extractor_path = tmp_path / "extractor"
  saved_path = tmp_path / "extractor.saved"
  alternate_path = tmp_path / "extractor.alternate"
  original_started = tmp_path / "original.started"
  alternate_started = tmp_path / "alternate.started"
  go_path = tmp_path / "extractor.go"
  segment = tmp_path / "segment"
  segment.write_bytes(b"unused")

  def script(marker: Path) -> str:
    return f"""#!{sys.executable}
from pathlib import Path
import struct
import sys
import time
Path({str(marker)!r}).touch()
while not Path({str(go_path)!r}).exists():
  time.sleep(0.001)
out = sys.stdout.buffer
out.write(struct.pack("<8sII", b"BLATV2R1", {learning_backfill.NATIVE_EXTRACTOR_SCHEMA_VERSION}, 0))
out.write(struct.pack("<IIQ", 0, 0xffffffff, 0))
out.flush()
"""

  extractor_path.write_text(script(original_started))
  alternate_path.write_text(script(alternate_started))
  extractor_path.chmod(0o755)
  alternate_path.chmod(0o755)
  expected_sha256 = hashlib.sha256(extractor_path.read_bytes()).hexdigest()
  extractor = open_verified_extractor(
    extractor_path,
    expected_sha256=expected_sha256,
  )
  swapped = threading.Event()
  swap_error: list[BaseException] = []

  def swap_only_while_child_starts() -> None:
    try:
      extractor_path.rename(saved_path)
      alternate_path.rename(extractor_path)
      swapped.set()
      deadline = time.monotonic() + 5.0
      while (
        not original_started.exists()
        and not alternate_started.exists()
        and time.monotonic() < deadline
      ):
        time.sleep(0.001)
      assert original_started.exists() or alternate_started.exists()
      extractor_path.rename(alternate_path)
      saved_path.rename(extractor_path)
      go_path.touch()
    except BaseException as exc:
      swap_error.append(exc)
      swapped.set()
      go_path.touch()

  thread = threading.Thread(target=swap_only_while_child_starts)
  thread.start()
  try:
    assert swapped.wait(timeout=5)
    records = extract_segment_events(
      extractor_path,
      segment,
      extractor_fd=extractor.descriptor,
    )
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not swap_error
    assert records == ()
    assert original_started.exists()
    assert not alternate_started.exists()
    assert hashlib.sha256(extractor_path.read_bytes()).hexdigest() == (
      expected_sha256
    )
    with pytest.raises(BackfillError) as raised:
      verify_open_extractor(extractor)
    assert raised.value.diagnostic == "backfill_reader_unavailable"
  finally:
    go_path.touch()
    thread.join(timeout=5)
    os.close(extractor.descriptor)


def _wait_for_path(path: Path) -> bool:
  deadline = time.monotonic() + 5.0
  while time.monotonic() < deadline:
    if path.exists():
      return True
    time.sleep(0.001)
  return False


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
