"""Disk guard and filesystem safety tests (CLAUDE.md §3, §11, §21).

The disk guard has to name the drive it refused to write to, because on this
machine the difference between C: and D: is the whole point.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from visualresearcher.utils.disk import (
    DiskSpaceError,
    check_free_space,
    drive_of,
    free_gb,
    usage_for,
)
from visualresearcher.utils.files import (
    atomic_write_bytes,
    atomic_write_text,
    backup_file,
    dir_size_bytes,
    is_within,
    resolve_within,
    sanitize_filename,
    slugify,
)
from visualresearcher.utils.hashing import cache_key, normalize_params, sha256_bytes

# ---------------------------------------------------------------------------
# Disk guard
# ---------------------------------------------------------------------------


def test_usage_reports_something_plausible(tmp_path):
    usage = usage_for(tmp_path)
    assert usage.total_gb > 0
    assert 0 <= usage.free_gb <= usage.total_gb
    assert usage.drive


def test_usage_works_for_a_path_that_does_not_exist_yet(tmp_path):
    """Stages check headroom before creating their output folder."""
    usage = usage_for(tmp_path / "not" / "created" / "yet")
    assert usage.total_gb > 0


def test_guard_passes_when_there_is_room(tmp_path):
    check_free_space(tmp_path, 0.0001, stage="test")


def test_guard_aborts_below_the_threshold_and_names_the_drive(tmp_path):
    """§21: guard aborts below threshold naming the drive."""
    impossible = free_gb(tmp_path) + 10_000
    with pytest.raises(DiskSpaceError) as excinfo:
        check_free_space(tmp_path, impossible, stage="image download")

    message = str(excinfo.value)
    expected_drive = drive_of(tmp_path)
    assert expected_drive in message, f"error must name the drive; got: {message}"
    assert "image download" in message, "error must name the stage that refused to run"
    assert "GB free" in message
    assert f"{impossible:.2f}" in message, "error must state the requirement"


def test_drive_of_returns_a_windows_drive_letter(tmp_path):
    drive = drive_of(tmp_path)
    assert drive.endswith(":") or drive in {"/", "\\"}


def test_dir_size_counts_nested_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.bin").write_bytes(b"0" * 100)
    (tmp_path / "y.bin").write_bytes(b"0" * 50)
    assert dir_size_bytes(tmp_path) == 150
    assert dir_size_bytes(tmp_path / "missing") == 0


# ---------------------------------------------------------------------------
# Filename sanitisation (§11: never trust a remote filename)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "../../../../windows/system32/config/sam",
        "..\\..\\..\\secrets.txt",
        "....//....//etc/passwd",
        "/absolute/path.jpg",
        "C:\\Windows\\evil.jpg",
        'normal<>:"|?*.jpg',
        "trailing.dots...",
        "trailing spaces   ",
        ".hidden",
        "..",
        ".",
        "",
        "con.jpg",
        "PRN",
        "lpt1.png",
        "nul.txt",
        "with\x00null.jpg",
        "new\nline.jpg",
    ],
)
def test_sanitize_filename_yields_one_safe_component(hostile):
    safe = sanitize_filename(hostile)
    assert safe, "must never return an empty name"
    assert "/" not in safe and "\\" not in safe, f"{safe!r} still contains a separator"
    assert ".." not in safe, f"{safe!r} still contains a traversal sequence"
    assert not safe.startswith("."), f"{safe!r} starts with a dot"
    assert safe == safe.rstrip(". "), f"{safe!r} has a trailing dot or space"
    assert not any(ord(c) < 32 for c in safe), f"{safe!r} contains a control character"
    assert safe.split(".")[0].lower() not in {
        "con",
        "prn",
        "aux",
        "nul",
        "com1",
        "lpt1",
    }, f"{safe!r} is a reserved Windows device name"


def test_sanitize_filename_keeps_ordinary_names_intact():
    assert sanitize_filename("031_1_darth_baras.jpg") == "031_1_darth_baras.jpg"


def test_sanitize_filename_truncates_but_keeps_the_extension():
    safe = sanitize_filename("x" * 400 + ".jpeg")
    assert len(safe) <= 120
    assert safe.endswith(".jpeg")


@pytest.mark.parametrize(
    "attack",
    [
        ("..", "..", "etc", "passwd"),
        ("../../escape.txt",),
        ("subdir", "..", "..", "outside.jpg"),
    ],
)
def test_resolve_within_blocks_path_traversal(tmp_path, attack):
    """§11: block path traversal."""
    base = tmp_path / "images"
    base.mkdir()
    resolved = resolve_within(base, *attack)
    assert is_within(base, resolved), (
        f"resolve_within escaped its base: {resolved} is outside {base}"
    )


def test_resolve_within_allows_a_legitimate_name(tmp_path):
    base = tmp_path / "images"
    base.mkdir()
    out = resolve_within(base, "031_03.jpg")
    assert out == base / "031_03.jpg"


def test_is_within_rejects_a_sibling_directory(tmp_path):
    base = tmp_path / "a"
    other = tmp_path / "ab"
    base.mkdir()
    other.mkdir()
    assert not is_within(base, other / "x.txt"), (
        "prefix matching must not treat 'ab' as being inside 'a'"
    )


# ---------------------------------------------------------------------------
# Slugs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Darth Baras Dark Council", "darth_baras_dark_council"),
        ("  Spaced   Out  ", "spaced_out"),
        ("Ünïcödé Trïckery", "unicode_trickery"),
        ("!!!", "untitled"),
        ("", "untitled"),
        ("Star Wars: The Old Republic", "star_wars_the_old_republic"),
    ],
)
def test_slugify(raw, expected):
    assert slugify(raw) == expected


def test_slugify_respects_max_length_and_snaps_to_a_word():
    slug = slugify("darth baras confronts the dark council on korriban", max_len=30)
    assert len(slug) <= 30
    assert not slug.endswith("_")
    assert "_" in slug


# ---------------------------------------------------------------------------
# Atomic writes (§12.6: never leave a partial file)
# ---------------------------------------------------------------------------


def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = tmp_path / "out.json"
    atomic_write_text(target, '{"a": 1}')
    assert target.read_text(encoding="utf-8") == '{"a": 1}'
    assert not list(tmp_path.glob("*.partial")), "a .partial file was left behind"


def test_atomic_write_is_utf8(tmp_path):
    """§2.12: encoding='utf-8' always."""
    target = tmp_path / "out.txt"
    atomic_write_text(target, "Ünïcödé — em dash and ellipsis…")
    assert target.read_text(encoding="utf-8") == "Ünïcödé — em dash and ellipsis…"
    assert b"\xc3\x9c" in target.read_bytes(), "file should be UTF-8 on disk"


def test_backup_is_taken_before_overwrite(tmp_path):
    """§2.8: .bak before overwrite."""
    target = tmp_path / "data.json"
    atomic_write_text(target, "original")
    atomic_write_text(target, "replacement", backup=True)
    assert target.read_text(encoding="utf-8") == "replacement"
    assert (tmp_path / "data.json.bak").read_text(encoding="utf-8") == "original"


def test_backup_of_a_missing_file_is_a_no_op(tmp_path):
    assert backup_file(tmp_path / "nothing.txt") is None


def test_atomic_write_bytes_round_trips(tmp_path):
    target = tmp_path / "blob.bin"
    payload = bytes(range(256))
    atomic_write_bytes(target, payload)
    assert target.read_bytes() == payload


# ---------------------------------------------------------------------------
# Cache keys (§20)
# ---------------------------------------------------------------------------


def test_cache_key_ignores_dict_ordering():
    a = cache_key("ddgs", "search", {"q": "baras", "n": 30})
    b = cache_key("ddgs", "search", {"n": 30, "q": "baras"})
    assert a == b, "key ordering must not create two cache entries for one call"


def test_cache_key_distinguishes_provider_and_params():
    base = cache_key("ddgs", "search", {"q": "baras"})
    assert base != cache_key("wikimedia", "search", {"q": "baras"})
    assert base != cache_key("ddgs", "images", {"q": "baras"})
    assert base != cache_key("ddgs", "search", {"q": "vowrawn"})


def test_normalize_params_is_stable_and_compact():
    assert normalize_params({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_sha256_matches_a_known_value():
    assert sha256_bytes(b"") == ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


def test_sha256_file_matches_bytes(tmp_path: Path):
    from visualresearcher.utils.hashing import sha256_file

    target = tmp_path / "f.bin"
    payload = b"visualresearcher" * 1000
    target.write_bytes(payload)
    assert sha256_file(target) == sha256_bytes(payload)


# ---------------------------------------------------------------------------
# The Windows rename retry (§2.12)
# ---------------------------------------------------------------------------


def test_the_atomic_rename_retries_a_transient_lock(tmp_path, monkeypatch):
    """§2.12: Windows holds brand-new files open for milliseconds.

    A search indexer or antivirus scanner can still have a handle on a file
    created moments ago, and `os.replace` then fails with WinError 5 or 32.
    Without a retry that surfaces as a pipeline stage crashing partway through
    a run -- which is what happened under two concurrent jobs.
    """
    import os as os_mod

    from visualresearcher.utils import files as files_mod

    target = tmp_path / "out.json"
    calls = {"n": 0}
    real_replace = os_mod.replace

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "Access is denied", str(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(files_mod.os, "replace", flaky_replace)
    monkeypatch.setattr(files_mod, "REPLACE_BACKOFF_S", 0.001)

    files_mod.atomic_write_text(target, '{"ok": true}')

    assert calls["n"] == 3, "it should have retried twice then succeeded"
    assert target.read_text(encoding="utf-8") == '{"ok": true}'
    assert not list(tmp_path.glob("*.partial")), "the staging file must be gone"


def test_a_persistent_lock_eventually_raises(tmp_path, monkeypatch):
    """Retrying forever would hang a run; it gives up and says so."""
    from visualresearcher.utils import files as files_mod

    def always_denied(src, dst):
        raise PermissionError(13, "Access is denied", str(dst))

    monkeypatch.setattr(files_mod.os, "replace", always_denied)
    monkeypatch.setattr(files_mod, "REPLACE_BACKOFF_S", 0.001)

    with pytest.raises(OSError, match="could not replace"):
        files_mod.atomic_write_text(tmp_path / "out.json", "x")

    assert not list(tmp_path.glob("*.partial")), "a failed write must not leave a .partial"


def test_a_real_error_is_not_retried(tmp_path, monkeypatch):
    """A full disk or bad path should be reported at once, not after six sleeps."""
    from visualresearcher.utils import files as files_mod

    calls = {"n": 0}

    def disk_full(src, dst):
        calls["n"] += 1
        raise OSError(28, "No space left on device", str(dst))

    monkeypatch.setattr(files_mod.os, "replace", disk_full)

    with pytest.raises(OSError, match="could not replace"):
        files_mod.atomic_write_text(tmp_path / "out.json", "x")

    assert calls["n"] == 1, "a non-transient error must not be retried"
