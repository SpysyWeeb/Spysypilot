"""
Seekable local playback engine for the on-device Clip Viewer feature.

This adapts pieces of tools/clip/run.py -- load_logs_parallel(), patch_submaster(),
FrameQueue/iter_segment_frames(), and the VisionIpcServer("camerad") setup -- so they can feed
the real onroad renderer (AugmentedRoadView) from *inside* the already-running on-device UI
process, instead of run.py's standalone linear offline render. Read tools/clip/run.py first;
this module deliberately mirrors its structure so the two stay easy to compare.

Differences from tools/clip/run.py (which is unmodified -- see that file for the original):
 - No RECORD/RECORD_OUTPUT/OFFSCREEN env vars, no gui_app.init_window()/ffmpeg process: we render
   into the existing app's one shared window via a Widget. We never own the window.
 - Arbitrary seek instead of one fixed [start, end] linear pass: segments/messages/camera frames
   are (re)loaded in a small rolling window around the current playback position instead of once
   up front for the whole clip. See _reload().
 - load_logs_windowed() (our load_logs_parallel equivalent) uses a ThreadPoolExecutor for the
   parse/chunk step instead of multiprocessing.Pool. run.py only ever calls this *before* opening
   its raylib window, so fork()ing there is harmless; here, the shared UI window/GL context and
   several background threads (mouse polling, screen brightness, params refresh) are already
   alive by the time a route is loaded, and fork()ing a multi-threaded, windowed process to spin
   up worker processes is a real (if generally low-probability) hazard worth just not taking.
   Threads are slower for this CPU-bound parsing step (GIL contention), but 1-2 segments' worth
   of local logs is small enough that this is not a practical problem for a manual seek.
 - The ui_state.sm.update monkeypatch (patch_submaster in run.py) is reworked so the replayed
   frame index is read live from this instance (self._displayed_frame_idx / self._window_seg_start)
   instead of captured once over a fixed message_chunks list -- a window reload (seek) then just
   needs to update that state, not reinstall the patch.
 - Message replay is keyed off self._displayed_frame_idx (the real idx of whatever camera frame
   was last actually sent to the preview, set in _pump_camera_frames), NOT self._route_frame (the
   wall-clock decode target run.py's equivalent linear loop would use). CONFIRMED IN THE FIELD:
   keying replay off the wall-clock target instead of the displayed frame produced visibly unsynced
   video/overlay -- the background camera decoder can fall well behind wall clock (e.g. the
   qcamera path below blocks on decoding an entire 60s segment before yielding its first frame)
   with nothing coupling the two clocks back together. self._route_frame still exists and still
   drives _reload()/window positioning (i.e. what to decode *towards*); self._displayed_frame_idx
   is what actually gets shown and replayed, so overlay data always matches the video frame
   underneath it, at the cost of playback running slower than real time when decode lags.
 - Camera frames are only ever displayed once wall-clock time says they're due (see _drain_ready).
   run.py's loop is 1 render tick = 1 decoded frame, always true there since it owns a dedicated
   render loop paced at FRAMERATE; not true here, where the render loop's rate is independent of
   the video's native 20fps. CONFIRMED IN THE FIELD: without this cap, a 20fps clip visibly played
   back far faster than 20fps whenever decode had a backlog ready and the render loop ticked
   faster than 20Hz (which it now often can). A one-frame lookahead holds a decoded-but-not-yet-due
   frame between calls instead of showing it early or dropping it.
 - The idx passed to VisionIpcServer.send() is NOT the raw route-frame counter run.py uses --
   see NUM_VIPC_BUFFERS below. CONFIRMED IN THE FIELD: shipping this with run.py's unbounded idx
   froze the device (silent, unrecoverable, no OOM-killer log, no panic, no crash log -- consistent
   with a wedged GPU driver) after ~14s/~280 frames of playback. run.py never surfaces this because
   it's a short-lived process that exits right after rendering one clip.
 - Also serves VISION_STREAM_WIDE_ROAD (matching run.py's conditional wide-camera setup) instead
   of road-only: AugmentedRoadView can switch to wide mid-replay (this fork's own
   experimental-mode hot-swap feature makes replayed experimentalMode=True routes common), and a
   road-only server leaves that switch stuck retrying an unthrottled per-frame reconnect forever.

SAFETY-CRITICAL, read before touching this file:
ui_state (selfdrive/ui/ui_state.py) is a hard process-wide singleton. selfdrive/ui/ui.py's main
loop calls the *real* ui_state.update() every single frame, unconditionally, regardless of the
nav stack -- it is not gated on whether our screen is the visible/top widget. ui_state.update()
calls self.sm.update(0) internally, so for as long as ui_state.sm.update is monkeypatched here,
*every frame of the whole app* -- not just frames where our screen renders -- derives
ui_state.started/ignition/status (and fires ui_state's offroad/engaged transition callbacks) from
replayed data instead of live IPC. This is contained to in-process, cosmetic UI state (screen
brightness/wakefulness, which layout is selected, alert dismissal bookkeeping): actual vehicle
control (controlsd/selfdrived/card/radard/plannerd) runs in separate OS processes with their own
independent onroad detection and never sees this process's ui_state object, and per
system/manager/process_config.py those processes (and camerad) are not even running while we've
verified we're offroad. But it still means the *display* would not fall through to the real
onroad screen on its own while this patch is installed and our screen occludes MainLayout on the
nav stack (MainLayout's own onroad-transition handling simply does not run while anything else is
on top of the nav stack) -- hence the independent ignition watchdog below, which must be polled
every frame our screen is visible, loaded route or not, and which never reads ui_state.sm (the
thing this module lies to) to make that determination.
"""
import os
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain

import numpy as np

from openpilot.cereal import messaging
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.spysypilot.clip_routes import RouteSummary
from openpilot.selfdrive.test.process_replay.migration import migrate_all
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.tools.lib.filereader import FileReader
from openpilot.tools.lib.framereader import FrameReader, ffprobe
from openpilot.tools.lib.logreader import _LogFileReader
from openpilot.tools.lib.route import FileName
from msgq.visionipc import VisionIpcServer, VisionStreamType

FRAMERATE = 20  # fixed message/frame grid baked into recorded logs -- matches tools/clip/run.py
SEG_SECONDS = 60  # nominal segment length
WINDOW_SEGMENTS = 2  # how many ~60s segments to keep parsed/decoded around the playhead
MAX_FRAMES_PER_TICK = 10  # cap catch-up work per UI frame after a stall/seek

# Number of physical buffers VisionIpcServer.create_buffers() allocates per stream. CameraView
# (selfdrive/ui/onroad/cameraview.py -- shared with the live onroad screen, not modified by this
# feature) caches one GPU/EGL/DMA-buf import per distinct frame.idx it sees, keyed by whatever idx
# value send() was called with, and never evicts that cache short of the widget being fully closed.
# tools/clip/run.py passes a route-relative frame counter (0..thousands) as idx and gets away with
# it because it's a short-lived process that exits after rendering one clip -- any accumulation
# dies with the process. This runs inside the long-lived, cached-for-the-app's-lifetime ui process
# instead (see clip_viewer_button.py), so an unbounded, ever-growing idx would mean an unbounded,
# never-freed GPU resource per frame -- a real, confirmed-in-the-field device freeze (silent, no
# OOM-killer log, no panic, no crash log: consistent with a wedged GPU driver, not a clean crash).
# Fix: cycle idx through [0, NUM_VIPC_BUFFERS) so CameraView's cache legitimately reuses at most
# this many entries, mirroring how the small physical buffer ring is actually sized.
NUM_VIPC_BUFFERS = 4


def _resolve_segment(log_root: str, seg_dir_name: str) -> dict[str, str | None]:
  """Find the known on-disk filenames for one segment directory.

  Deliberately NOT using tools.lib.route.Route(name, data_dir=...) here. Verified by direct
  testing (not just reading) that it cannot find real on-device routes: Route._get_segments_local
  matches candidate directory names with `segment_name.startswith(self.name.canonical_name)`, but
  RouteName always normalizes canonical_name to use '|' as the dongle_id/timestamp delimiter,
  while real on-device segment directories are named with '_'
  (<dongle_id>_<YYYY-MM-DD--HH-MM-SS>--<segment_num>). '_...'.startswith('...|...') is always
  False, so Route(name, data_dir=log_root) raises "Could not find segments for route ..." for
  every real recorded route. We resolve paths ourselves instead, reusing only the FileName
  filename constants from tools.lib.route for consistency.
  """
  seg_path = os.path.join(log_root, seg_dir_name)

  def find(names: tuple[str, ...]) -> str | None:
    for name in names:
      p = os.path.join(seg_path, name)
      if os.path.isfile(p):
        return p
    return None

  return {
    'log_path': find(FileName.RLOG),
    'camera_path': find(FileName.FCAMERA),
    'qcamera_path': find(FileName.QCAMERA),
    'ecamera_path': find(FileName.ECAMERA),
  }


def _download_segment(path: str) -> bytes:
  with FileReader(path) as f:
    return bytes(f.read())


def _parse_and_chunk_segment(raw_data: bytes, fps: int) -> list[dict]:
  """Same chunking behavior as tools/clip/run.py's _parse_and_chunk_segment, just called
  directly (per-segment) instead of through a multiprocessing.Pool worker wrapper."""
  messages = migrate_all(list(_LogFileReader("", dat=raw_data, sort_by_time=True)))
  if not messages:
    return []

  dt_ns, chunks, current, next_time = 1e9 / fps, [], {}, messages[0].logMonoTime + 1e9 / fps
  for msg in messages:
    if msg.logMonoTime >= next_time:
      chunks.append(current)
      current, next_time = {}, next_time + dt_ns * ((msg.logMonoTime - next_time) // dt_ns + 1)
    current[msg.which()] = msg
  return chunks + [current] if current else chunks


def load_logs_windowed(log_paths: list[str], fps: int = FRAMERATE) -> list[dict]:
  """Equivalent to tools/clip/run.py's load_logs_parallel, but thread-pooled end to end instead
  of process-pooled for the parse step -- see module docstring for why."""
  if not log_paths:
    return []
  num_workers = min(16, len(log_paths))

  with ThreadPoolExecutor(max_workers=num_workers) as pool:
    futures = {pool.submit(_download_segment, path): idx for idx, path in enumerate(log_paths)}
    raw_data = {futures[f]: f.result() for f in as_completed(futures)}

  with ThreadPoolExecutor(max_workers=num_workers) as pool:
    futures = [pool.submit(_parse_and_chunk_segment, raw_data[i], fps) for i in range(len(log_paths))]
    results = [f.result() for f in futures]

  return list(chain.from_iterable(results))


def _get_frame_dimensions(camera_path: str) -> tuple[int, int]:
  probe = ffprobe(camera_path)
  stream = probe["streams"][0]
  return stream["width"], stream["height"]


def _iter_segment_frames(camera_paths, start_time, end_time, fps=FRAMERATE, use_qcam=False,
                          frame_size: tuple[int, int] | None = None):
  """Same as tools/clip/run.py's iter_segment_frames, except it stops the generator quietly
  instead of raising when it runs past the last recorded segment/frame -- a nominal
  num_segments*60s route duration can overrun the real (possibly shorter) last segment, and that
  is expected here (we don't parse logs just to find the exact last frame for a list display)."""
  frames_per_seg = fps * SEG_SECONDS
  start_frame, end_frame = int(start_time * fps), int(end_time * fps)
  current_seg = -1
  seg_frames: FrameReader | np.ndarray | None = None

  for global_idx in range(start_frame, end_frame):
    seg_idx, local_idx = global_idx // frames_per_seg, global_idx % frames_per_seg

    if seg_idx != current_seg:
      current_seg = seg_idx
      path = camera_paths[seg_idx] if seg_idx < len(camera_paths) else None
      if not path:
        return

      if use_qcam:
        w, h = frame_size or _get_frame_dimensions(path)
        with FileReader(path) as f:
          # "-i pipe:0", not "-i -": CONFIRMED ON-DEVICE (live SSH, 2026-07-01) this device's
          # ffmpeg is a minimal build (--disable-autodetect, --enable-protocol='file,pipe' only --
          # see `ffmpeg -version`'s configuration line) that does not resolve the "-" stdin
          # shorthand the way a typical full-featured ffmpeg build does -- it fails immediately
          # with "Error opening input: Protocol not found". tools/clip/run.py has this same "-i -"
          # invocation but is normally run on a dev PC with a full ffmpeg build, so this never
          # surfaced there. "-v error" (was "quiet"): quiet was swallowing the real error text
          # above, which is exactly what made this fail silently instead of self-diagnosing.
          result = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "nv12", "-"],
                                  input=f.read(), capture_output=True)
        if result.returncode != 0:
          cloudlog.warning(f"clip_playback: ffmpeg qcamera decode failed: {result.stderr.decode()}")
          return
        seg_frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, w * h * 3 // 2)
      else:
        seg_frames = FrameReader(path, pix_fmt="nv12")

    assert seg_frames is not None
    try:
      frame = seg_frames[local_idx] if use_qcam else seg_frames.get(local_idx)
    except Exception:
      return  # past the end of this segment's actual recorded frames

    yield global_idx, frame


class FrameQueue:
  """Background camera-frame decoder/prefetcher -- behavior matches tools/clip/run.py's
  FrameQueue exactly. Strictly sequential/forward over [start_time, end_time); it cannot rewind
  or jump ahead, so ClipPlayer recreates a fresh one on every seek (see ClipPlayer._reload)."""
  def __init__(self, camera_paths, start_time, end_time, fps=FRAMERATE, prefetch_count=60, use_qcam=False):
    first_path = next((p for p in camera_paths if p), None)
    if not first_path:
      raise RuntimeError("No valid camera paths")
    self.frame_w, self.frame_h = _get_frame_dimensions(first_path)

    self._queue: queue.Queue = queue.Queue(maxsize=prefetch_count)
    self._stop = threading.Event()
    self._error: Exception | None = None
    self._thread = threading.Thread(
      target=self._worker,
      args=(camera_paths, start_time, end_time, fps, use_qcam, (self.frame_w, self.frame_h)),
      daemon=True,
    )
    self._thread.start()

  def _worker(self, camera_paths, start_time, end_time, fps, use_qcam, frame_size):
    try:
      for idx, data in _iter_segment_frames(camera_paths, start_time, end_time, fps, use_qcam, frame_size):
        if self._stop.is_set():
          break
        self._queue.put((idx, data.tobytes()))
    except Exception as e:
      cloudlog.exception("clip_playback: frame decode error")
      self._error = e
    finally:
      self._queue.put(None)

  def get_nowait(self):
    """Non-blocking. Raises queue.Empty if nothing is ready yet, StopIteration once the window
    is exhausted. Only non-blocking access is needed here -- unlike run.py's dedicated render
    loop, we must never stall the shared UI render thread waiting on the decoder."""
    if self._error:
      raise self._error
    result = self._queue.get_nowait()
    if result is None:
      raise StopIteration("No more frames")
    return result

  def stop(self):
    self._stop.set()
    while not self._queue.empty():
      try:
        self._queue.get_nowait()
      except queue.Empty:
        break
    self._thread.join(timeout=2.0)


class ClipPlayer:
  """Owns one route's worth of seekable playback state: parsed message chunks for a small
  rolling window of segments, a background camera-frame decoder for the same window, and the
  VisionIpcServer("camerad") that feeds AugmentedRoadView. See module docstring for the
  ui_state.sm monkeypatch lifecycle and the independent ignition watchdog."""

  def __init__(self):
    self._log_root = Paths.log_root()
    self.route: RouteSummary | None = None
    self._log_paths: list[str | None] = []
    self._camera_paths: list[str | None] = []
    self._ecamera_paths: list[str | None] = []
    self._use_qcam = False

    self._window_seg_start = -1
    self._message_chunks: list[dict] = []

    self._frame_queue: FrameQueue | None = None
    self._wide_frame_queue: FrameQueue | None = None
    self._road_pending: tuple[int, bytes] | None = None  # see _drain_ready
    self._wide_pending: tuple[int, bytes] | None = None
    self._vipc: VisionIpcServer | None = None
    self._frame_send_count = 0  # diagnostic: logged once so a future test is diagnosable via cloudlog

    self.total_frames = 0
    self._route_frame = 0  # wall-clock DECODE TARGET -- drives _reload()/window positioning only
    # Real (route-relative) idx of the last camera frame actually sent to the preview. Message
    # replay and UI-facing position (current_time_s/progress) key off THIS, not _route_frame --
    # CONFIRMED ON-DEVICE (2026-07-01): video and overlay were visibly unsynced. Root cause: the
    # camera frame shown at any given tick is whatever the background FrameQueue decoder thread
    # has managed to produce so far (which can fall well behind wall clock, e.g. the qcamera path
    # blocks on decoding an entire 60s/1200-frame segment before yielding its first frame), while
    # _route_frame/mock_update marched forward on wall clock regardless -- two independent clocks
    # with no coupling between them. Keying replay off the actually-displayed frame instead means
    # overlay data (lane lines, alerts, HUD) always matches what's on screen, at the cost of
    # playback running slower than real time when decode is the bottleneck -- correct pairing over
    # correct speed, which is the right trade for a review feature.
    self._displayed_frame_idx = 0
    self.playing = False
    self._play_wall_start = 0.0
    self._play_frame_start = 0
    self._speed = 1.0  # always 1x today; pacing math below is structured to make changing this easy

    self._patched = False
    self._orig_sm_update = None

    # Deliberately independent of ui_state.sm -- see module docstring. SubMaster is not a
    # singleton (confirmed against e.g. selfdrive/car/card.py, selfdrive/controls/radard.py), so
    # this is a normal, cheap, safe construction.
    self._watchdog_sm = messaging.SubMaster(["deviceState"])
    self.ignition_tripped = False

  # -- loading ------------------------------------------------------------------------------
  def load_route(self, route: RouteSummary):
    self.close()  # tear down whatever was loaded before and restore any stale patch first
    self.ignition_tripped = False
    self.route = route
    self._log_paths = []
    self._camera_paths = []
    self._ecamera_paths = []
    qcamera_paths: list[str | None] = []
    for seg_name in route.segments:
      paths = _resolve_segment(self._log_root, seg_name)
      self._log_paths.append(paths['log_path'])
      self._camera_paths.append(paths['camera_path'])
      self._ecamera_paths.append(paths['ecamera_path'])
      qcamera_paths.append(paths['qcamera_path'])

    # Prefer qcamera (low-res) whenever every segment has it, rather than only as a fallback for
    # missing full-res segments. CONFIRMED ON-DEVICE (live SSH session, 2026-07-01): with the
    # EGL bypass (see cameraview.py's use_egl), every displayed frame now goes through a real CPU
    # texture upload (the exact cost the zero-copy path exists to avoid) at fcamera's full
    # 1928x1208, and that upload is the dominant remaining cost -- FPS logged chronically at
    # 3-9 (should be ~20) for the entire duration of playback, not just during reloads. qcamera is
    # roughly an order of magnitude fewer pixels; for a route-review preview (not the primary
    # driving display) that's a reasonable trade against a large, continuous performance cost.
    self._use_qcam = all(qcamera_paths) or (not all(self._camera_paths) and any(qcamera_paths))
    if self._use_qcam:
      self._camera_paths = qcamera_paths

    self._frame_send_count = 0
    self.total_frames = route.num_segments * SEG_SECONDS * FRAMERATE
    self._route_frame = 0
    self._displayed_frame_idx = 0
    self.playing = False

    self._reload(0)
    if self.total_frames > 0:
      self._install_patch()

  def _reload(self, target_frame: int):
    """(Re)load whatever's needed to render at target_frame: message chunks for the segment
    window containing it (skipped if that window is already loaded), and a fresh
    FrameQueue/VisionIpcServer starting exactly at target_frame (always recreated -- FrameQueue
    can only go forward, so any seek needs a new one; local-disk only, so this is cheap)."""
    num_segs = len(self._log_paths)
    if num_segs == 0:
      return
    seg_start = max(0, min(target_frame // (SEG_SECONDS * FRAMERATE), num_segs - 1))
    seg_end = min(num_segs, seg_start + WINDOW_SEGMENTS)

    if seg_start != self._window_seg_start:
      window_log_paths = [p for p in self._log_paths[seg_start:seg_end] if p]
      try:
        self._message_chunks = load_logs_windowed(window_log_paths, fps=FRAMERATE)
      except Exception:
        cloudlog.exception("clip_playback: failed to load logs")
        self._message_chunks = []
      self._window_seg_start = seg_start

    self._teardown_frame_feed()

    target_s = target_frame / FRAMERATE
    window_end_s = seg_end * SEG_SECONDS
    if target_s < window_end_s and any(self._camera_paths[seg_start:seg_end]):
      try:
        fq = FrameQueue(self._camera_paths, target_s, window_end_s, fps=FRAMERATE, use_qcam=self._use_qcam)
        vipc = VisionIpcServer("camerad")
        vipc.create_buffers(VisionStreamType.VISION_STREAM_ROAD, NUM_VIPC_BUFFERS, fq.frame_w, fq.frame_h)

        # Also serve VISION_STREAM_WIDE_ROAD when the route has it (mirrors tools/clip/run.py).
        # Without this, AugmentedRoadView._switch_stream_if_needed can decide to switch to wide
        # (e.g. replayed experimentalMode + low replayed vEgo -- this fork ships an
        # experimental-mode hot-swap button, so this is a real, not hypothetical, combination) and
        # get stuck target-connecting to a stream we never provisioned. Not used with the qcamera
        # fallback, matching run.py -- qcam-only routes don't have a wide segment to decode either.
        wfq = None
        if not self._use_qcam and any(self._ecamera_paths[seg_start:seg_end]):
          try:
            wfq = FrameQueue(self._ecamera_paths, target_s, window_end_s, fps=FRAMERATE)
            vipc.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, NUM_VIPC_BUFFERS, wfq.frame_w, wfq.frame_h)
          except Exception:
            cloudlog.exception("clip_playback: failed to start wide camera feed, road-only")
            if wfq is not None:
              wfq.stop()  # create_buffers can fail after the decoder thread already started
            wfq = None

        vipc.start_listener()
        self._frame_queue = fq
        self._wide_frame_queue = wfq
        self._vipc = vipc
        cloudlog.debug(f"clip_playback: serving ROAD{'+WIDE' if wfq else ''} buffers "
                        f"({fq.frame_w}x{fq.frame_h}) for {self.route.name if self.route else '?'}")
      except Exception:
        cloudlog.exception("clip_playback: failed to start camera feed")
        self._frame_queue = None
        self._wide_frame_queue = None
        self._vipc = None

  # -- transport ------------------------------------------------------------------------------
  def play(self):
    if not self.route or self.total_frames == 0:
      return
    if self._route_frame >= self.total_frames - 1:
      self.seek_to_frame(0)
    self.playing = True
    self._play_wall_start = time.monotonic()
    self._play_frame_start = self._route_frame

  def pause(self):
    self.playing = False

  def toggle_play_pause(self):
    self.pause() if self.playing else self.play()

  def seek_to_frame(self, frame: int):
    if not self.route or self.total_frames == 0:
      return
    frame = max(0, min(frame, self.total_frames - 1))
    # Defense in depth against duplicate/bursty seek calls (see clip_seek_bar.py's
    # _handle_mouse_event docstring) -- _reload() unconditionally tears down and recreates the
    # VisionIpcServer/FrameQueue/GL textures, so a no-op duplicate seek to the exact frame we're
    # already serving should skip that entirely rather than repeat the expensive rebuild.
    if frame == self._route_frame and self._frame_queue is not None:
      return
    self._route_frame = frame
    self._reload(frame)
    if self.playing:
      self._play_wall_start = time.monotonic()
      self._play_frame_start = frame

  def seek_fraction(self, frac: float):
    frac = max(0.0, min(frac, 1.0))
    self.seek_to_frame(int(frac * max(self.total_frames - 1, 0)))

  # -- per-frame update ------------------------------------------------------------------------
  def tick(self) -> bool:
    """Call once per UI frame *whenever our screen is the visible/top widget*, loaded route or
    not. Returns False if the ignition watchdog just tripped -- caller must leave the screen
    immediately. This poll must never be skipped while our screen occludes MainLayout: see module
    docstring, MainLayout's own onroad-transition handling does not run at all while anything
    else is on top of the nav stack, patched sm or not, loaded route or not."""
    self._watchdog_sm.update(0)
    if self._watchdog_sm['deviceState'].started:
      self.ignition_tripped = True

    if self.ignition_tripped:
      # Never let a teardown failure here propagate: this runs from Widget._update_state() inside
      # the shared UI's main render loop, and this exact path is what's responsible for getting
      # us off screen the moment the car is actually being driven -- an uncaught exception here
      # would instead crash the whole UI process at the worst possible time. close() already
      # guarantees the ui_state.sm patch is restored via its own try/finally; this just ensures
      # tick() itself always returns False once tripped, no matter what.
      try:
        self.close()
      except Exception:
        cloudlog.exception("clip_playback: error while closing after ignition trip")
      return False

    if not self.route:
      return True

    if self.playing and self.total_frames:
      elapsed = time.monotonic() - self._play_wall_start
      target = self._play_frame_start + int(elapsed * FRAMERATE * self._speed)
      target = min(target, self.total_frames - 1)
      if target // (SEG_SECONDS * FRAMERATE) != self._window_seg_start:
        self._reload(target)
      self._route_frame = target
      if target >= self.total_frames - 1:
        self.playing = False

    self._pump_camera_frames()
    return True

  def _drain_ready(self, fq: FrameQueue, pending_attr: str, max_idx: int) -> tuple[tuple[int, bytes] | None, bool]:
    """Drain decoded frames with idx <= max_idx from one decoder queue, keeping only the most
    recent due one -- the vision client conflates to the latest buffer anyway, and the UI's own
    render loop is what actually paints, so intermediate frames on a slow tick don't need sending
    individually. Bounded so a big stall/seek can't spend unbounded time here.

    CONFIRMED IN THE FIELD (2026-07-01): without the max_idx cap, this always returned whatever
    the decoder had most recently produced, with no relation to wall-clock time. FrameQueue's
    background thread decodes as fast as it can (only blocked by its queue filling up, never
    throttled to real time), and this method gets called once per UI render tick -- so whenever
    the render loop ticks faster than the source's native 20fps (it now often can, after the
    earlier performance fixes), playback raced through frames far faster than real time (a 20fps
    clip visibly playing back at whatever rate decoder+render-loop could sustain together, well
    above 20fps). max_idx is the current wall-clock playback target (self._route_frame, computed
    for real-time pacing regardless of render rate) -- a frame is only ever handed to the caller
    once its idx is actually due. A decoded frame that's ahead of schedule is held in
    getattr(self, pending_attr) (a one-frame lookahead) rather than being shown early or dropped,
    so it's simply used on a later call once max_idx catches up to it -- this is also exactly what
    lets playback correctly run *slower* than real time when decode is the bottleneck (round 5's
    fix), rather than skipping ahead to resync.

    Returns (frame_or_None, exhausted) -- the caller drops its reference to fq when exhausted."""
    latest = getattr(self, pending_attr)
    if latest is not None:
      if latest[0] > max_idx:
        return None, False  # already holding a not-yet-due frame; nothing new to show this tick
      setattr(self, pending_attr, None)

    for _ in range(MAX_FRAMES_PER_TICK):
      try:
        item = fq.get_nowait()
      except queue.Empty:
        break
      except StopIteration:
        return latest, True
      except Exception:
        cloudlog.exception("clip_playback: frame queue error")
        return latest, True
      if item[0] <= max_idx:
        latest = item
      else:
        setattr(self, pending_attr, item)  # ahead of schedule -- hold for a later tick
        break
    return latest, False

  def _pump_camera_frames(self):
    """Forward the newest *due* decoded road (and wide, if being served) frame to the preview
    VisionIpcServer -- see _drain_ready for the wall-clock-pacing reasoning. The idx passed to
    send() is deliberately NOT the raw route-frame counter -- see NUM_VIPC_BUFFERS' module-level
    comment: CameraView caches one GPU/EGL image per distinct idx forever, so an ever-growing idx
    is an ever-growing, never-freed GPU resource in this long-lived process. pts is still computed
    from the real (unbounded) frame counter so playback timing/ordering stays correct -- only the
    buffer-slot idx is bounded."""
    if self._vipc is None:
      return

    max_idx = self._route_frame  # wall-clock target; see _drain_ready

    if self._frame_queue is not None:
      road, exhausted = self._drain_ready(self._frame_queue, '_road_pending', max_idx)
      if exhausted:
        self._frame_queue = None
      if road is not None:
        idx, frame_bytes = road
        pts = int(idx * 5e7)  # matches tools/clip/run.py's synthetic 50ms-per-frame timestamp spacing
        self._vipc.send(VisionStreamType.VISION_STREAM_ROAD, frame_bytes, idx % NUM_VIPC_BUFFERS, pts, pts)
        self._displayed_frame_idx = idx  # mock_update keys off this -- see module/field docstring
        self._frame_send_count += 1
        if self._frame_send_count == 1:
          route_name = self.route.name if self.route else '?'
          cloudlog.debug(f"clip_playback: sent first ROAD frame (route idx {idx}) for {route_name}")

    if self._wide_frame_queue is not None:
      wide, exhausted = self._drain_ready(self._wide_frame_queue, '_wide_pending', max_idx)
      if exhausted:
        self._wide_frame_queue = None
      if wide is not None:
        idx, frame_bytes = wide
        pts = int(idx * 5e7)
        self._vipc.send(VisionStreamType.VISION_STREAM_WIDE_ROAD, frame_bytes, idx % NUM_VIPC_BUFFERS, pts, pts)

  # -- ui_state.sm monkeypatch (see module docstring) ------------------------------------------
  def _install_patch(self):
    if self._patched:
      return
    self._orig_sm_update = ui_state.sm.update
    # Reset started_frame so alerts render correctly (recv_frame must be >= started_frame),
    # matching tools/clip/run.py's patch_submaster.
    ui_state.started_frame = 0
    ui_state.started_time = time.monotonic()

    player = self

    def mock_update(timeout=None):
      sm = ui_state.sm
      t = time.monotonic()
      sm.updated = dict.fromkeys(sm.services, False)
      chunks = player._message_chunks
      idx = player._displayed_frame_idx - player._window_seg_start * SEG_SECONDS * FRAMERATE
      if chunks and 0 <= idx < len(chunks):
        for svc, msg in chunks[idx].items():
          if svc in sm.data:
            sm.seen[svc] = sm.updated[svc] = sm.alive[svc] = sm.valid[svc] = True
            sm.data[svc] = getattr(msg.as_builder(), svc)
            sm.logMonoTime[svc], sm.recv_time[svc], sm.recv_frame[svc] = msg.logMonoTime, t, sm.frame
      sm.frame += 1

    ui_state.sm.update = mock_update
    self._patched = True

  def _restore_patch(self):
    if not self._patched:
      return
    try:
      if self._orig_sm_update is not None:
        ui_state.sm.update = self._orig_sm_update
    finally:
      self._patched = False
      self._orig_sm_update = None

  # -- teardown ------------------------------------------------------------------------------
  def _teardown_frame_feed(self):
    if self._frame_queue is not None:
      self._frame_queue.stop()
      self._frame_queue = None
    if self._wide_frame_queue is not None:
      self._wide_frame_queue.stop()
      self._wide_frame_queue = None
    # A pending lookahead frame belongs to the FrameQueue that decoded it -- never carry it over
    # to whatever gets created next (_reload always tears down before creating fresh queues).
    self._road_pending = None
    self._wide_pending = None
    # No explicit close()/stop() API observed on VisionIpcServer anywhere in this codebase's
    # usage (e.g. CameraView.close() just drops its VisionIpcClient reference the same way) --
    # dropping the reference lets its destructor release the shared buffers/listener socket.
    self._vipc = None

  def close(self):
    """Tear down everything and restore the real ui_state.sm.update. Safe to call repeatedly /
    when nothing is loaded. Must run on every exit path (back button, route switch, ignition
    watchdog trip, screen hide) -- guaranteed via try/finally regardless of what fails above."""
    try:
      self.playing = False
      self.route = None
      self._log_paths = []
      self._camera_paths = []
      self._ecamera_paths = []
      self.total_frames = 0
      self._route_frame = 0
      self._displayed_frame_idx = 0
      self._teardown_frame_feed()
      self._message_chunks = []
      self._window_seg_start = -1
    finally:
      self._restore_patch()

  # -- convenience for the UI ------------------------------------------------------------------
  # current_time_s/progress key off _displayed_frame_idx (what's actually on screen), not
  # _route_frame (the wall-clock decode target) -- so the seek bar's position always matches the
  # video/overlay pair actually being shown, not where playback is nominally "supposed" to be.
  @property
  def current_time_s(self) -> float:
    return self._displayed_frame_idx / FRAMERATE

  @property
  def total_time_s(self) -> float:
    return self.total_frames / FRAMERATE

  @property
  def progress(self) -> float:
    if not self.total_frames:
      return 0.0
    return self._displayed_frame_idx / max(self.total_frames - 1, 1)

  @property
  def is_loaded(self) -> bool:
    return self.route is not None
