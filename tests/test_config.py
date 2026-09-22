"""Configuration tests (CLAUDE.md §3, §16).

The cache-location tests exist because getting them wrong has a concrete
consequence on this machine: C: has under 200 MB free, and one unguarded model
download fills it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from visualresearcher.config import (
    ENV_DEFAULT_KEYS,
    Settings,
    load_env_defaults,
    load_settings,
    offline_mode,
)

# ---------------------------------------------------------------------------
# Defaults and YAML overrides
# ---------------------------------------------------------------------------


def test_defaults_match_section_16(sandbox):
    cfg = load_settings(config_path=sandbox / "absent.yaml", root=sandbox)
    assert cfg.segmentation.target_s == 8.0
    assert (cfg.segmentation.min_s, cfg.segmentation.max_s) == (6.0, 10.0)
    assert cfg.images.candidates_per_segment == 30
    assert cfg.images.keep_per_segment == 8
    assert cfg.images.max_bytes == 15_728_640
    assert cfg.clips.max_full_download_minutes == 15
    assert cfg.confidence.high == 0.85
    assert cfg.confidence.medium == 0.60
    assert cfg.output.min_free_gb == 5
    assert cfg.watch.filename_prefix == "Tight"
    assert cfg.worker.max_concurrent == 2
    assert cfg.compute.max_concurrent_cpu_heavy == 1


def test_the_shipped_config_yaml_parses():
    """The real config.yaml must stay loadable."""
    root = Path(__file__).resolve().parents[1]
    cfg = load_settings(config_path=root / "config.yaml", root=root)
    assert cfg.ranking.weights.clip == 0.45
    assert cfg.images.providers == ["ddgs", "wikimedia"]


def test_yaml_overrides_only_the_keys_it_names(sandbox):
    path = sandbox / "config.yaml"
    path.write_text(yaml.safe_dump({"segmentation": {"target_s": 12.0}}), encoding="utf-8")
    cfg = load_settings(config_path=path, root=sandbox)
    assert cfg.segmentation.target_s == 12.0
    assert cfg.segmentation.min_s == 6.0, "unnamed keys must keep their defaults"


def test_a_relative_output_root_resolves_under_the_install_root(sandbox):
    """A relative path must never resolve against the shell's cwd (§3)."""
    path = sandbox / "config.yaml"
    path.write_text(yaml.safe_dump({"output": {"root": "projects"}}), encoding="utf-8")
    cfg = load_settings(config_path=path, root=sandbox)
    assert cfg.projects_dir == sandbox / "projects"
    assert cfg.projects_dir.is_absolute()


def test_an_absolute_output_root_is_respected(sandbox, tmp_path):
    elsewhere = tmp_path / "somewhere"
    path = sandbox / "config.yaml"
    path.write_text(yaml.safe_dump({"output": {"root": str(elsewhere)}}), encoding="utf-8")
    cfg = load_settings(config_path=path, root=sandbox)
    assert cfg.projects_dir == elsewhere


def test_a_non_mapping_config_is_rejected(sandbox):
    path = sandbox / "config.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_settings(config_path=path, root=sandbox)


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError):
        Settings.model_validate({"segmentation": {"min_s": 0}})


def test_derived_paths_all_sit_under_the_install_root(sandbox):
    cfg = load_settings(config_path=sandbox / "absent.yaml", root=sandbox)
    for path in (cfg.data_dir, cfg.cache_dir, cfg.tmp_dir, cfg.projects_dir, cfg.db_path):
        assert str(path).startswith(str(sandbox)), f"{path} escaped the install root"


def test_ensure_dirs_creates_what_it_promises(sandbox):
    cfg = load_settings(config_path=sandbox / "absent.yaml", root=sandbox)
    cfg.ensure_dirs()
    for path in (cfg.data_dir, cfg.cache_dir, cfg.tmp_dir, cfg.projects_dir):
        assert path.is_dir()


# ---------------------------------------------------------------------------
# Offline switch (§2.9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_offline_mode_recognises_truthy_values(monkeypatch, value):
    monkeypatch.setenv("VR_OFFLINE", value)
    assert offline_mode() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "", "off"])
def test_offline_mode_recognises_falsy_values(monkeypatch, value):
    monkeypatch.setenv("VR_OFFLINE", value)
    assert offline_mode() is False


# ---------------------------------------------------------------------------
# Cache locations (§3) -- nothing may touch C:
# ---------------------------------------------------------------------------


def test_env_defaults_fill_in_unset_cache_variables(sandbox, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    (sandbox / ".env").write_text("HF_HOME=caches/hf\nTORCH_HOME=caches/torch\n", encoding="utf-8")
    applied = load_env_defaults(sandbox)
    assert applied["HF_HOME"] == str(sandbox / "caches" / "hf")
    assert os.environ["HF_HOME"] == str(sandbox / "caches" / "hf")
    assert (sandbox / "caches" / "hf").is_dir(), "the cache directory should exist"


def test_an_existing_environment_value_is_never_overwritten(sandbox, monkeypatch):
    monkeypatch.setenv("HF_HOME", "D:/explicit/choice")
    (sandbox / ".env").write_text("HF_HOME=caches/hf\n", encoding="utf-8")
    applied = load_env_defaults(sandbox)
    assert "HF_HOME" not in applied
    assert os.environ["HF_HOME"] == "D:/explicit/choice"


def test_env_defaults_ignore_keys_outside_the_allowlist(sandbox, monkeypatch):
    """This file must not become a general config or credential channel."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (sandbox / ".env").write_text(
        "OPENAI_API_KEY=sk-should-never-be-read\nVR_OFFLINE=0\n", encoding="utf-8"
    )
    applied = load_env_defaults(sandbox)
    assert applied == {}
    assert "OPENAI_API_KEY" not in os.environ


def test_missing_env_file_is_not_an_error(sandbox):
    assert load_env_defaults(sandbox) == {}


def test_the_shipped_env_file_keeps_every_cache_off_c_drive():
    """The real .env must not point any cache at C: (§3)."""
    root = Path(__file__).resolve().parents[1]
    env_file = root / ".env"
    assert env_file.exists(), ".env should ship with the repo"

    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        assert key.strip() in ENV_DEFAULT_KEYS, f"{key} is not an allowed key"
        assert not value.strip().upper().startswith("C:"), f"{key} points at C:, which §3 forbids"


def test_the_shipped_env_file_contains_no_secrets():
    root = Path(__file__).resolve().parents[1]
    text = root / ".env"
    content = text.read_text(encoding="utf-8")
    for marker in ("sk-", "API_KEY", "TOKEN", "SECRET", "PASSWORD"):
        for raw in content.splitlines():
            if raw.strip().startswith("#"):
                continue
            assert marker not in raw, f".env appears to contain a secret: {raw!r}"
