from __future__ import annotations

import os
import re
import sys
from pathlib import Path


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_RESET = "\x1b[0m"
_UNSET = object()
_STYLE_CODES = {
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
}


class Console:
    """Small console-output helper with optional ANSI color and mirrored logging."""

    def __init__(
        self,
        stream=None,
        log_path: str | Path | None = "dyana.log",
        use_color: bool | None = None,
        append_log: bool = False,
    ):
        self.stream = stream or sys.stdout
        self.log_path = Path(log_path) if log_path is not None else None
        self.append_log = bool(append_log)
        self.use_color = _supports_color(self.stream) if use_color is None else bool(use_color)
        self._log_file = None

    def configure(
        self,
        *,
        stream=None,
        log_path: str | Path | None | object = _UNSET,
        use_color: bool | None = None,
        append_log: bool | None = None,
    ) -> None:
        """Reconfigure stream, log path, and color behavior."""
        if stream is not None:
            self.stream = stream

        if log_path is not _UNSET:
            self.close()
            self.log_path = Path(log_path) if log_path is not None else None

        if append_log is not None:
            self.append_log = bool(append_log)

        if use_color is None:
            self.use_color = _supports_color(self.stream)
        else:
            self.use_color = bool(use_color)

    def close(self) -> None:
        """Close the mirrored log file if it is open."""
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def capture_state(self) -> tuple[object, Path | None, bool, bool]:
        """Capture the current console configuration for later restoration."""
        return (self.stream, self.log_path, self.use_color, self.append_log)

    def restore_state(self, state: tuple[object, Path | None, bool, bool]) -> None:
        """Restore a previously captured console configuration."""
        stream, log_path, use_color, append_log = state
        self.close()
        self.stream = stream
        self.log_path = log_path
        self.use_color = use_color
        self.append_log = append_log

    def plain(self, message: str = "") -> None:
        self.emit(message)

    def prompt(self, message: str) -> None:
        """Write an interactive prompt without a trailing newline."""
        self.emit(message, end=" ")

    def info(self, message: str) -> None:
        self.emit(message, prefix="> ", style="dim")

    def success(self, message: str) -> None:
        self.emit(message, prefix="+ ", style="green")

    def warn(self, message: str) -> None:
        self.emit(message, prefix="! ", style="yellow")

    def error(self, message: str) -> None:
        self.emit(message, prefix="x ", style="red")

    def progress(self, message: str) -> None:
        self.emit(message, prefix="... ", style="cyan")

    def section(self, title: str) -> None:
        self.emit("")
        self.emit(title, style=("bold", "cyan"))

    def header(self, title: str, lines: list[str] | tuple[str, ...] | None = None) -> None:
        self.emit(title, style=("bold", "cyan"))
        self.emit("=" * len(title), style="cyan")
        for line in lines or ():
            self.emit(line)

    def key_value(self, key: str, value: object, *, indent: int = 0, style: str | tuple[str, ...] | None = None) -> None:
        self.emit(f"{key}: {value}", indent=indent, style=style)

    def emit(
        self,
        message: str = "",
        *,
        prefix: str = "",
        indent: int = 0,
        style: str | tuple[str, ...] | None = None,
        end: str = "\n",
    ) -> None:
        """Write one logical console message and mirror it into the console log."""
        lines = message.splitlines() or [""]
        rendered_lines = []
        plain_lines = []
        for line in lines:
            plain_text = (" " * indent) + prefix + line
            plain_lines.append(plain_text)
            rendered_lines.append(_apply_style(plain_text, style, enabled=self.use_color))

        rendered_text = "\n".join(rendered_lines)
        plain_text = "\n".join(plain_lines)

        self.stream.write(rendered_text + end)
        flush = getattr(self.stream, "flush", None)
        if callable(flush):
            flush()
        self._write_log(plain_text + end)

    def _write_log(self, text: str) -> None:
        if self.log_path is None:
            return

        if self._log_file is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if self.append_log else "w"
            self._log_file = open(self.log_path, mode, buffering=1, encoding="utf-8")

        self._log_file.write(_strip_ansi(text))
        self._log_file.flush()

    def log_reply(self, reply: str) -> None:
        """Append one interactive reply to the mirrored console log."""
        self._write_log(f"{reply}\n")

def _apply_style(text: str, style: str | tuple[str, ...] | None, *, enabled: bool) -> str:
    if not enabled or style is None:
        return text

    if isinstance(style, str):
        styles = (style,)
    else:
        styles = style

    prefix = "".join(_STYLE_CODES[name] for name in styles if name in _STYLE_CODES)
    if not prefix:
        return text
    return f"{prefix}{text}{_RESET}"


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from console text."""
    return _strip_ansi(text)


def _supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False

    isatty = getattr(stream, "isatty", None)
    if not callable(isatty) or not isatty():
        return False

    if os.name != "nt":
        return True

    return _enable_windows_ansi(stream)


def _enable_windows_ansi(stream) -> bool:
    if stream not in (sys.stdout, sys.stderr):
        return False

    try:
        import ctypes
    except ImportError:
        return False

    kernel32 = ctypes.windll.kernel32
    handle_id = -11 if stream is sys.stdout else -12
    handle = kernel32.GetStdHandle(handle_id)
    if handle == 0:
        return False

    mode = ctypes.c_uint()
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
        return False

    enable_vt = 0x0004
    if mode.value & enable_vt:
        return True

    return kernel32.SetConsoleMode(handle, mode.value | enable_vt) != 0


console = Console(log_path=None)
