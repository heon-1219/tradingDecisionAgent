"""Update keys in a `.env` file in place, atomically. Used by `PUT /api/settings` and the kill switch.

`update_env_file(path, {"NAME": "value" | None})`:
- The first line that sets `NAME` (optionally `export NAME`, any case, since pydantic-settings reads
  env names case-insensitively) is replaced in place; later lines that set it again are dropped, so
  the file ends up with exactly one. A missing name is appended at the end. `None` removes it.
- Every other line (comments, blank lines, unrelated keys) is kept byte for byte, along with the
  line-ending style (`\\r\\n` if the file uses it) and a UTF-8 BOM.
- Values are checked first (`check_env_value`): newlines, control characters, `"`, `\\` and `${`
  are refused, so a value can never break out of its line, escape its quotes, or be expanded by
  python-dotenv's `${VAR}` interpolation. A value made only of `[A-Za-z0-9_./:@+-]` is written bare;
  anything else is double-quoted. Nothing is written if any value or name is refused.
- The write is atomic: a temp file in the same directory, flushed and fsynced, then `os.replace`.
  An existing file's permission bits are copied to it; a new file keeps `mkstemp`'s owner-only mode.
- Values are never logged and never appear in an exception message.
"""
from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

_NAME_RE = re.compile(r"[A-Z_][A-Z0-9_]*")
_BARE_VALUE_RE = re.compile(r"[A-Za-z0-9_./:@+\-]+")
_LINE_KEY_RE = re.compile(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
_BOM = "﻿"


class EnvFileError(ValueError):
    """A name or value that cannot be written safely. The message names the problem, never the
    value."""


def check_env_value(value: str) -> None:
    """Raise `EnvFileError` unless `value` can be written as one safe `.env` value."""
    if not isinstance(value, str):
        raise EnvFileError("the value must be text")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise EnvFileError("the value contains a line break or another control character")
    if '"' in value or "\\" in value:
        raise EnvFileError('the value contains a double quote (") or a backslash (\\)')
    if "${" in value:
        raise EnvFileError("the value contains '${', which the .env reader would expand")


def _format(name: str, value: str) -> str:
    if _BARE_VALUE_RE.fullmatch(value):
        return f"{name}={value}"
    return f'{name}="{value}"'


def update_env_file(path: Path | str, updates: Mapping[str, str | None]) -> None:
    """Apply `updates` to the env file at `path` (created if missing). See the module docstring."""
    path = Path(path)
    for name, value in updates.items():
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise EnvFileError("invalid variable name")
        if value is not None:
            check_env_value(value)

    text = path.read_bytes().decode("utf-8") if path.exists() else ""
    bom = text.startswith(_BOM)
    if bom:
        text = text[len(_BOM):]
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)

    wanted = {name.upper(): name for name in updates}
    done: set[str] = set()
    out: list[str] = []
    for line in lines:
        match = _LINE_KEY_RE.match(line)
        key = match.group(1).upper() if match else None
        if key is None or key not in wanted:
            out.append(line)
            continue
        name = wanted[key]
        if key in done or updates[name] is None:
            done.add(key)
            continue  # a duplicate, or a removal
        ending = line[len(line.rstrip("\r\n")):] or newline
        out.append(_format(name, updates[name]) + ending)
        done.add(key)

    missing = [name for name in updates if name.upper() not in done and updates[name] is not None]
    if missing and out and not out[-1].endswith(("\n", "\r")):
        out[-1] += newline
    out.extend(_format(name, updates[name]) + newline for name in missing)

    data = ((_BOM if bom else "") + "".join(out)).encode("utf-8")
    _atomic_write(path, data)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else None
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
