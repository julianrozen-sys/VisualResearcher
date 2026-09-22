"""Configuration (CLAUDE.md §16).

Defaults live here in Pydantic models; ``config.yaml`` at the repo root
overrides them key by key. Anything path-shaped is resolved against the install
root so a relative config value can never land on C: by accident.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

__all__ = ["Settings", "load_settings", "install_root", "offline_mode"]


def install_root() -> Path:
    """Repo root: ``src/visualresearcher/config.py`` -> three parents up."""
    return Path(__file__).resolve().parents[2]


#: Variables ``.env`` is allowed to supply. Restricted on purpose: this
#: mechanism exists to keep model and package caches off C: (§3), not as a
#: general configuration channel, and it must never carry credentials.
ENV_DEFAULT_KEYS = frozenset(
    {"HF_HOME", "TORCH_HOME", "UV_CACHE_DIR", "PIP_CACHE_DIR", "XDG_CACHE_HOME"}
)


def load_local_env(root: Path | None = None) -> list[str]:
    """Load ``<root>/.env.local`` into the environment. Returns the keys set.

    Unlike ``.env``, this file is gitignored and may hold secrets and personal
    data -- it is where `providers/llm/openai.py` already tells people to put
    ``OPENAI_API_KEY``. Nothing actually read it until the first live run
    needed ``VR_CONTACT``, so that instruction was pointing at a file the code
    ignored: put your key there as told, and authentication still fails with
    no clue why.

    Any ``KEY=VALUE`` is accepted (no allow-list -- that is ``.env``'s job, and
    its restriction exists to stop cache paths turning into a config channel).
    A value already in the real environment always wins.
    """
    base = Path(root).resolve() if root else install_root()
    env_file = base / ".env.local"
    applied: list[str] = []
    if not env_file.exists():
        return applied

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value or os.environ.get(key):
            continue
        os.environ[key] = value
        applied.append(key)
    return applied


def load_env_defaults(root: Path | None = None) -> dict[str, str]:
    """Fill in unset cache-location variables from ``<root>/.env``.

    §3 says caches come from ``HF_HOME``/``TORCH_HOME`` and that no cache path
    may be hardcoded. Both hold here: the code only ever reads the environment,
    and the machine-specific values live in an editable file outside the
    source. Existing variables are never overwritten -- a value already in the
    environment always wins.

    This matters more than it looks: with ~160 MB free on C:, a single
    unguarded model download would fill the system drive.
    """
    base = Path(root).resolve() if root else install_root()
    env_file = base / ".env"
    applied: dict[str, str] = {}
    if not env_file.exists():
        return applied

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key not in ENV_DEFAULT_KEYS or not value:
            continue
        if os.environ.get(key):
            continue  # an explicit environment value always wins
        resolved = Path(value)
        if not resolved.is_absolute():
            resolved = base / resolved
        resolved.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(resolved)
        applied[key] = str(resolved)
    return applied


def offline_mode() -> bool:
    """``VR_OFFLINE=1`` forces every provider to its fake (CLAUDE.md §2.9)."""
    return os.environ.get("VR_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


class SegmentationSettings(BaseModel):
    target_s: float = 8.0
    min_s: float = 6.0
    max_s: float = 10.0

    @field_validator("min_s")
    @classmethod
    def _min_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("segmentation.min_s must be > 0")
        return v


class ImageSettings(BaseModel):
    candidates_per_segment: int = 30
    keep_per_segment: int = 8
    max_bytes: int = 15_728_640
    min_width: int = 800
    phash_distance: int = 6
    providers: list[str] = Field(default_factory=lambda: ["ddgs", "wikimedia"])
    #: Per-provider cap on how many of a segment's queries it is asked.
    #:
    #: Call count is queries x providers, so this is the only knob that reduces
    #: a rate-limited provider's load; `candidates_per_segment` sets `limit=`
    #: inside each call and does not change how many calls are made.
    #:
    #: Commons is capped at 2 rather than dropped: it is the only provider that
    #: returns a real licence (ddgs answered 68 of 68 with `license=unknown`),
    #: so every segment keeps some attributable candidates in its ranking pool.
    max_queries_per_provider: dict[str, int] = Field(
        default_factory=lambda: {"wikimedia": 2}
    )


class ClipSettings(BaseModel):
    enabled: bool = True
    max_per_segment: int = 1
    padding_s: float = 3.0
    max_height: int = 1080
    max_full_download_minutes: float = 15.0
    max_project_gb: float = 15.0
    concurrency: int = 2


class RankingWeights(BaseModel):
    clip: float = 0.45
    entity: float = 0.2
    source: float = 0.1
    resolution: float = 0.1
    sharpness: float = 0.1
    repetition: float = -0.15


class RankingSettings(BaseModel):
    model: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "auto"
    seed: int = 1729
    weights: RankingWeights = Field(default_factory=RankingWeights)


class OutputSettings(BaseModel):
    root: Path = Field(default_factory=lambda: install_root() / "projects")
    slug_max_len: int = 40
    write_srt: bool = True
    min_free_gb: float = 5.0


class ConfidenceSettings(BaseModel):
    high: float = 0.85
    medium: float = 0.60


class WatchSettings(BaseModel):
    dir: Path = Path("D:/shared/incoming")
    filename_prefix: str = "Tight"
    stable_secs: float = 3.0
    poll_interval_s: float = 5.0


class WorkerSettings(BaseModel):
    max_concurrent: int = 2


class ComputeSettings(BaseModel):
    max_concurrent_cpu_heavy: int = 1


class TranscriptionSettings(BaseModel):
    """Not in §16's sample but needed to pick a provider without editing code."""

    provider: str = "faster_whisper"
    model: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None


class Settings(BaseModel):
    segmentation: SegmentationSettings = Field(default_factory=SegmentationSettings)
    images: ImageSettings = Field(default_factory=ImageSettings)
    clips: ClipSettings = Field(default_factory=ClipSettings)
    ranking: RankingSettings = Field(default_factory=RankingSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    confidence: ConfidenceSettings = Field(default_factory=ConfidenceSettings)
    watch: WatchSettings = Field(default_factory=WatchSettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    compute: ComputeSettings = Field(default_factory=ComputeSettings)
    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)

    # Not user-facing; set so tests can redirect state without touching the real tree.
    root: Path = Field(default_factory=install_root)

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "visualresearcher.db"

    @property
    def cache_dir(self) -> Path:
        return self.root / ".cache"

    @property
    def tmp_dir(self) -> Path:
        return self.root / ".tmp"

    @property
    def domain_packs_dir(self) -> Path:
        return self.root / "domain_packs"

    @property
    def projects_dir(self) -> Path:
        return self.output.root

    def project_dir(self, name: str) -> Path:
        return self.projects_dir / name

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.cache_dir, self.tmp_dir, self.projects_dir):
            path.mkdir(parents=True, exist_ok=True)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings(config_path: Path | None = None, *, root: Path | None = None) -> Settings:
    """Build settings from ``config.yaml``, falling back to the defaults above.

    Args:
        config_path: explicit YAML file; defaults to ``<root>/config.yaml``.
        root: install root override, used by tests to sandbox all state.
    """
    base_root = Path(root).resolve() if root else install_root()
    path = Path(config_path) if config_path else base_root / "config.yaml"
    raw: dict = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}")
        raw = loaded

    raw = _deep_merge({}, raw)
    raw["root"] = base_root
    # A relative output root must resolve under the install root, never the cwd.
    out = raw.setdefault("output", {})
    if isinstance(out, dict):
        if "root" in out and out["root"]:
            candidate = Path(str(out["root"]))
            out["root"] = candidate if candidate.is_absolute() else (base_root / candidate)
        else:
            out["root"] = base_root / "projects"
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return load_settings()


def get_settings(refresh: bool = False) -> Settings:
    if refresh:
        _cached_settings.cache_clear()
    return _cached_settings()
