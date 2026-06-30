import collections
import subprocess
import threading
import pyray as rl

try:
    import zmq
    import msgpack
    _HAS_ZMQ = True
except ImportError:
    _HAS_ZMQ = False

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget

_BG     = rl.Color(15, 15, 15, 255)
_GREEN  = rl.Color(140, 255, 140, 255)
_YELLOW = rl.Color(255, 230, 100, 255)
_RED    = rl.Color(255, 100, 100, 255)
_DIM    = rl.Color(160, 160, 160, 120)
_BORDER = rl.Color(255, 255, 255, 25)

MAX_LINES = 50
FONT_SIZE  = 22
LINE_H     = 28
PAD        = 16

SWAGLOG_SOCKET = "ipc:///tmp/swaglog"

_SWAGLOG_COLORS = {
    'DEBUG':    _DIM,
    'INFO':     _GREEN,
    'WARNING':  _YELLOW,
    'ERROR':    _RED,
    'CRITICAL': _RED,
}

# journalctl patterns to suppress (PAM session spam)
_JOURNAL_NOISE = r'pam_unix.*session\|session opened for\|session closed for\|COMMAND='


class TerminalWidget(Widget):
    def __init__(self):
        super().__init__()
        self._lines: collections.deque[tuple[str, rl.Color]] = collections.deque(maxlen=MAX_LINES)
        self._lock = threading.Lock()
        self._running = False
        self._journal_proc: subprocess.Popen | None = None
        self._grep_proc: subprocess.Popen | None = None
        self._background_tap_callback = None

    def set_background_tap_callback(self, cb) -> None:
        self._background_tap_callback = cb

    def show_event(self) -> None:
        super().show_event()
        self._start()

    def hide_event(self) -> None:
        super().hide_event()
        self._stop()

    def _start(self) -> None:
        self._stop()
        self._running = True
        with self._lock:
            self._lines.clear()

        # journalctl → grep (filter noise) → reader thread
        try:
            self._journal_proc = subprocess.Popen(
                ['journalctl', '-f', '-n', '20', '--no-pager', '-o', 'short-monotonic', '--no-hostname'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
            self._grep_proc = subprocess.Popen(
                ['grep', '--line-buffered', '-Ev', _JOURNAL_NOISE],
                stdin=self._journal_proc.stdout, stdout=subprocess.PIPE, text=True, bufsize=1,
            )
            threading.Thread(target=self._journal_reader, daemon=True).start()
        except FileNotFoundError:
            with self._lock:
                self._lines.append(('[journalctl not found]', _DIM))

        # swaglog ZMQ subscriber — openpilot internal logs
        if _HAS_ZMQ:
            threading.Thread(target=self._swaglog_reader, daemon=True).start()
        else:
            with self._lock:
                self._lines.append(('[zmq/msgpack not available — openpilot logs hidden]', _YELLOW))

    def _stop(self) -> None:
        self._running = False
        for proc in (self._grep_proc, self._journal_proc):
            if proc is not None:
                proc.terminate()
        self._grep_proc = None
        self._journal_proc = None

    def _journal_reader(self) -> None:
        proc = self._grep_proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            if not self._running:
                break
            text = line.rstrip('\n')
            if text:
                with self._lock:
                    self._lines.append((text, _DIM))

    def _swaglog_reader(self) -> None:
        try:
            ctx = zmq.Context()
            sock = ctx.socket(zmq.SUB)
            sock.connect(SWAGLOG_SOCKET)
            sock.setsockopt(zmq.SUBSCRIBE, b"")
            sock.setsockopt(zmq.RCVTIMEO, 200)
        except Exception:
            return

        while self._running:
            try:
                data = sock.recv()
                msg = msgpack.unpackb(data, raw=False)
                level = msg.get('levelname', 'INFO')
                name  = msg.get('name', '?')
                text  = str(msg.get('message', ''))
                line  = f"[{level[0]}] {name}: {text}"
                color = _SWAGLOG_COLORS.get(level, _GREEN)
                with self._lock:
                    self._lines.append((line, color))
            except zmq.error.Again:
                pass
            except Exception:
                pass

        try:
            sock.close()
        except Exception:
            pass

    def _handle_mouse_release(self, mouse_pos) -> None:
        if self._background_tap_callback:
            self._background_tap_callback()

    def _render(self, rect: rl.Rectangle) -> None:
        rl.draw_rectangle_rounded(rect, 0.025, 10, _BG)
        rl.draw_rectangle_rounded_lines_ex(rect, 0.025, 10, 1, _BORDER)

        with self._lock:
            lines = list(self._lines)

        fn = gui_app.font(FontWeight.NORMAL)

        if not lines:
            rl.draw_text_ex(fn, 'Waiting for output...', rl.Vector2(int(rect.x + PAD), int(rect.y + PAD)),
                            FONT_SIZE, 0, _DIM)
            return

        max_visible = max(1, int((rect.height - PAD * 2) / LINE_H))
        visible = lines[-max_visible:]
        max_chars = max(1, int((rect.width - PAD * 2) / (FONT_SIZE * 0.56)))

        rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
        y = int(rect.y + PAD)
        x = int(rect.x + PAD)
        for text, color in visible:
            rl.draw_text_ex(fn, text[:max_chars], rl.Vector2(x, y), FONT_SIZE, 0, color)
            y += LINE_H
        rl.end_scissor_mode()
