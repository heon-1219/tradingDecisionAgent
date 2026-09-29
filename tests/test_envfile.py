"""Tests for `finagent.envfile`: the in-place, atomic `.env` updater behind `PUT /api/settings`.

Every test works on a tmp file; the real `.env` is never touched.
"""
from __future__ import annotations

import os

import pytest

from finagent.config import Settings
from finagent.envfile import EnvFileError, check_env_value, update_env_file

SECRET = "sk-TEST-envfile-0123456789abcdef"

ORIGINAL = (
    "# my settings\n"
    "LLM_PROVIDER=gemini\n"
    "\n"
    "ALPACA_API_KEY_ID=OLDKEYID\n"
    "# trading\n"
    "TRADING_ENABLED=true\n"
    "OTHER=keep me  # a comment\n"
)


def _write(path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def _read(path) -> str:
    return path.read_bytes().decode("utf-8")


def test_replaces_a_line_in_place_and_keeps_everything_else(tmp_path):
    env = tmp_path / ".env"
    _write(env, ORIGINAL)
    update_env_file(env, {"ALPACA_API_KEY_ID": "NEWKEYID123", "TRADING_ENABLED": "false"})
    assert _read(env) == (
        "# my settings\n"
        "LLM_PROVIDER=gemini\n"
        "\n"
        "ALPACA_API_KEY_ID=NEWKEYID123\n"
        "# trading\n"
        "TRADING_ENABLED=false\n"
        "OTHER=keep me  # a comment\n"
    )


def test_appends_a_missing_key_at_the_end(tmp_path):
    env = tmp_path / ".env"
    _write(env, "A=1")  # no trailing newline
    update_env_file(env, {"TAVILY_API_KEY": "tvly-abc"})
    assert _read(env) == "A=1\nTAVILY_API_KEY=tvly-abc\n"


def test_none_removes_the_key_and_every_duplicate(tmp_path):
    env = tmp_path / ".env"
    _write(env, "TAVILY_API_KEY=a\nB=2\nexport TAVILY_API_KEY=b\n")
    update_env_file(env, {"TAVILY_API_KEY": None})
    assert _read(env) == "B=2\n"


def test_duplicates_collapse_into_the_first_line(tmp_path):
    env = tmp_path / ".env"
    _write(env, "X=0\nSEC_USER_AGENT=old\nY=1\nSEC_USER_AGENT=older\n")
    update_env_file(env, {"SEC_USER_AGENT": "Jane Doe jane@example.com"})
    assert _read(env) == 'X=0\nSEC_USER_AGENT="Jane Doe jane@example.com"\nY=1\n'


def test_names_match_case_insensitively_and_through_export(tmp_path):
    # pydantic-settings reads env names case-insensitively, so a lower-case line is the same key.
    env = tmp_path / ".env"
    _write(env, "alpaca_api_key_id=old\n  export ALPACA_API_SECRET_KEY = old2\n")
    update_env_file(env, {"ALPACA_API_KEY_ID": "N1", "ALPACA_API_SECRET_KEY": "N2"})
    assert _read(env) == "ALPACA_API_KEY_ID=N1\nALPACA_API_SECRET_KEY=N2\n"


def test_a_similar_prefix_is_not_a_match(tmp_path):
    env = tmp_path / ".env"
    _write(env, "TAVILY_API_KEY_OLD=x\n#TAVILY_API_KEY=commented\n")
    update_env_file(env, {"TAVILY_API_KEY": "new"})
    assert _read(env) == "TAVILY_API_KEY_OLD=x\n#TAVILY_API_KEY=commented\nTAVILY_API_KEY=new\n"


def test_crlf_line_endings_are_kept(tmp_path):
    env = tmp_path / ".env"
    _write(env, "A=1\r\nTRADING_ENABLED=true\r\n")
    update_env_file(env, {"TRADING_ENABLED": "false", "B": "2"})
    assert _read(env) == "A=1\r\nTRADING_ENABLED=false\r\nB=2\r\n"


def test_a_utf8_bom_is_kept_and_the_first_line_still_matches(tmp_path):
    env = tmp_path / ".env"
    env.write_bytes("﻿TRADING_ENABLED=true\n".encode("utf-8"))
    update_env_file(env, {"TRADING_ENABLED": "false"})
    assert env.read_bytes() == "﻿TRADING_ENABLED=false\n".encode("utf-8")


def test_creates_the_file_when_missing(tmp_path):
    env = tmp_path / "sub" / ".env"
    update_env_file(env, {"TRADING_ENABLED": "false"})
    assert _read(env) == "TRADING_ENABLED=false\n"


def test_values_with_spaces_or_hashes_are_double_quoted(tmp_path):
    env = tmp_path / ".env"
    update_env_file(env, {"SEC_USER_AGENT": "Jane Doe #1 jane@example.com", "K": "a'b"})
    assert _read(env) == 'SEC_USER_AGENT="Jane Doe #1 jane@example.com"\nK="a\'b"\n'


@pytest.mark.parametrize("bad", [
    "line\nbreak", "carriage\rreturn", 'a"quote', "back\\slash", "nul\x00", "tab\tchar",
    "interp${HOME}", "del\x7f",
])
def test_unsafe_values_are_refused_and_nothing_is_written(tmp_path, bad):
    env = tmp_path / ".env"
    _write(env, ORIGINAL)
    with pytest.raises(EnvFileError) as info:
        update_env_file(env, {"TAVILY_API_KEY": "ok", "SEC_USER_AGENT": bad})
    assert _read(env) == ORIGINAL
    assert bad not in str(info.value)
    assert list(tmp_path.iterdir()) == [env]


def test_check_env_value_error_never_contains_the_value():
    with pytest.raises(EnvFileError) as info:
        check_env_value(SECRET + "\n")
    assert SECRET not in str(info.value)
    check_env_value(SECRET)  # a normal key is fine


def test_invalid_names_are_refused(tmp_path):
    env = tmp_path / ".env"
    for name in ("", "1ABC", "A B", "A=B", "a-b"):
        with pytest.raises(EnvFileError):
            update_env_file(env, {name: "x"})
    assert not env.exists()


def test_a_failed_replace_leaves_the_original_and_no_temp_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    _write(env, ORIGINAL)

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        update_env_file(env, {"TRADING_ENABLED": "false"})
    assert _read(env) == ORIGINAL
    assert list(tmp_path.iterdir()) == [env]


def test_no_temp_file_is_left_behind(tmp_path):
    env = tmp_path / ".env"
    _write(env, ORIGINAL)
    update_env_file(env, {"TRADING_ENABLED": "false"})
    assert list(tmp_path.iterdir()) == [env]


def test_settings_read_back_exactly_what_was_written(tmp_path, monkeypatch):
    for name in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY", "TAVILY_API_KEY", "SEC_USER_AGENT",
                 "TRADING_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    _write(env, ORIGINAL)
    update_env_file(env, {
        "ALPACA_API_KEY_ID": "PKABC123",
        "ALPACA_API_SECRET_KEY": SECRET,
        "TAVILY_API_KEY": "tvly-x_y.z",
        "SEC_USER_AGENT": "Jane Doe jane@example.com #1 $5 'quoted'",
        "TRADING_ENABLED": "false",
    })
    s = Settings(_env_file=env)
    assert s.alpaca_api_key_id == "PKABC123"
    assert s.alpaca_api_secret_key == SECRET
    assert s.tavily_api_key == "tvly-x_y.z"
    assert s.sec_user_agent == "Jane Doe jane@example.com #1 $5 'quoted'"
    assert s.trading_enabled is False
