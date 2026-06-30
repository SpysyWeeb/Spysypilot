import collections
import subprocess
import threading
import pyray as rl

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

LIVE_LOG   = '/tmp/spysy_terminal.log'

# journalctl patterns to suppress (PAM session spam)
_JOURNAL_NOISE = r'pam_unix.*session\|session opened for\|session closed for\|COMMAND='


def _level_color(line: str) -> rl.Color:
    if line.startswith('[W]') or line.startswith('[WARNING]'):
        return _YELLOW
    if line.startswith('[E]') or line.startswith('[ERROR]') or line.startswith('[C]') or line.startswith('[CRITICAL]'):
        return _RED
    return _GREEN


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

        # Thread 1: openpilot swaglog feed (tail the live plaintext file)
        threading.Thread(target=self._openpilot_reader, daemon=True).start()

        # Thread 2: filtered system journal (SSH connections, kernel, etc.)
        try:
            self._journal_proc = subprocess.Popen(
                ['journalctl', '-f', '-n', '5', '--no-pager', '-o', 'short-monotonic', '--no-hostname'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
            self._grep_proc = subprocess.Popen(
                ['grep', '--line-buffered', '-Ev', _JOURNAL_NOISE],
                stdin=self._journal_proc.stdout, stdout=subprocess.PIPE, text=True, bufsize=1,
            )
            threading.Thread(target=self._journal_reader, daemon=True).start()
        except FileNotFoundError:
            pass

    def _stop(self) -> None:
        self._running = False
        for proc in (self._grep_proc, self._journal_proc):
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        self._grep_proc = None
        self._journal_proc = None

    def _openpilot_reader(self) -> None:
        try:
            proc = subprocess.Popen(
                ['tail', '-f', '-n', '50', LIVE_LOG],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except FileNotFoundError:
            with self._lock:
                self._lines.append(('[tail not found]', _YELLOW))
            return

        for line in proc.stdout:
            if not self._running:
                proc.terminate()
                break
            text = line.rstrip('\n')
            if text:
                with self._lock:
                    self._lines.append((text, _level_color(text)))

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
                    self._lines.append(('[sys] ' + text, _DIM))

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
            rl.draw_text_ex(fn, 'Waiting for openpilot output...',
                            rl.Vector2(int(rect.x + PAD), int(rect.y + PAD)), FONT_SIZE, 0, _DIM)
            return

        max_visible = max(1, int((rect.height - PAD * 2) / LINE_H))
        visible = lines[-max_visible:]

        rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
        y = int(rect.y + PAD)
        x = int(rect.x + PAD)
        for text, color in visible:
            rl.draw_text_ex(fn, text, rl.Vector2(x, y), FONT_SIZE, 0, color)
            y += LINE_H
        rl.end_scissor_mode()
