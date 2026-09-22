"""Fixture-backed image search (CLAUDE.md §2.9).

This fake does something the other fakes do not: it writes **real image files**
to disk and returns ``file://`` URLs to them. That is deliberate. If it
returned fabricated metadata, the offline run would skip the download
validation, the Pillow decode check, the hashing and the dedupe pass -- which
is most of what §11 actually specifies. Generating real bytes means
``VR_OFFLINE=1`` exercises all of it.

The generated set is built to give the dedupe stage genuine work:

* unique images per query,
* a shared pool that several queries return, as real providers do,
* **exact duplicates** (byte-identical, caught by SHA256),
* **resized, cropped and recompressed variants** (caught by pHash/dHash),
* a small number of deliberately invalid files -- too small, undecodable, and
  an SVG -- so the rejection paths run too.

Everything is deterministic: the same query always produces the same set.
"""

from __future__ import annotations

import colorsys
import hashlib
import random
from pathlib import Path

from ...logging_setup import get_logger
from ..base import Availability
from .base import ImageCandidate, ImageSearchProvider

__all__ = ["FakeImageSearchProvider", "generate_image"]

log = get_logger("providers.images.fake")

#: Near-duplicate kinds, cheapest-to-detect first.
_DUPLICATE_KINDS = ("exact duplicate", "resized", "cropped", "recompressed")

_LICENCES = [
    ("CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/"),
    ("CC BY 2.0", "https://creativecommons.org/licenses/by/2.0/"),
    ("Public domain", "https://creativecommons.org/publicdomain/mark/1.0/"),
    ("unknown", ""),
]

_DOMAINS = [
    "commons.wikimedia.org",
    "starwars.fandom.com",
    "swtor.com",
    "static.wikia.nocookie.net",
    "i.imgur.com",
]


def _seed_for(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def generate_image(
    path: Path,
    seed: int,
    *,
    width: int = 1200,
    height: int = 800,
    quality: int = 88,
    label: str = "",
) -> Path:
    """Draw a deterministic, visually distinctive JPEG at ``path``.

    Distinctiveness matters: perceptual hashing must be able to tell two
    different generated images apart, or the dedupe tests would pass for the
    wrong reason.
    """
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    hue = rng.random()
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)

    # Vertical gradient base.
    top = colorsys.hsv_to_rgb(hue, 0.55, 0.85)
    bottom = colorsys.hsv_to_rgb((hue + 0.12) % 1.0, 0.7, 0.25)
    for y in range(height):
        blend = y / max(1, height - 1)
        draw.line(
            [(0, y), (width, y)],
            fill=tuple(int(255 * (top[i] * (1 - blend) + bottom[i] * blend)) for i in range(3)),
        )

    # Shapes, so the perceptual hash of each image is genuinely different.
    for _ in range(rng.randint(6, 12)):
        x0 = rng.randint(0, width - 1)
        y0 = rng.randint(0, height - 1)
        x1 = min(width, x0 + rng.randint(width // 12, width // 3))
        y1 = min(height, y0 + rng.randint(height // 12, height // 3))
        shade = colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.3, 0.9), rng.uniform(0.3, 1.0))
        colour = tuple(int(255 * c) for c in shade)
        if rng.random() < 0.5:
            draw.rectangle([x0, y0, x1, y1], fill=colour)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=colour)

    if label:
        draw.text((16, 16), label[:60], fill=(255, 255, 255))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=quality)
    return path


class FakeImageSearchProvider(ImageSearchProvider):
    name = "fake"
    is_fake = True

    def __init__(self, *, cache_dir: Path | None = None, include_bad: bool = True, **_ignored):
        # Falls back to a temp-free location under the install root.
        if cache_dir is None:
            from ...config import install_root

            cache_dir = install_root() / ".cache" / "fake_images"
        self.cache_dir = Path(cache_dir)
        self.include_bad = include_bad

    def availability(self) -> Availability:
        return Availability.available("generates real images locally", offline_safe=True)

    # -- generation ---------------------------------------------------------

    def _ensure(self, name: str, seed: int, **kwargs) -> Path:
        path = self.cache_dir / name
        if not path.exists():
            generate_image(path, seed, **kwargs)
        return path

    def _shared_pool(self, count: int = 8) -> list[tuple[str, Path]]:
        """Images several queries return, as real search engines do."""
        out = []
        for i in range(count):
            name = f"shared_{i:02d}.jpg"
            path = self._ensure(name, seed=1000 + i, width=1280, height=720, label=name)
            out.append((name, path))
        return out

    def _candidate(
        self,
        path: Path,
        *,
        query: str,
        index: int,
        note: str = "",
    ) -> ImageCandidate:
        rng = random.Random(_seed_for(f"{query}:{index}"))
        licence, licence_url = _LICENCES[rng.randrange(len(_LICENCES))]
        domain = _DOMAINS[rng.randrange(len(_DOMAINS))]
        width = height = 0
        try:
            from PIL import Image

            with Image.open(path) as probe:
                width, height = probe.size
        except Exception:  # noqa: BLE001 - a deliberately broken file
            pass
        return ImageCandidate(
            image_url=path.resolve().as_uri(),
            source_page=f"https://{domain}/wiki/{query.replace(' ', '_')}",
            title=f"{query} ({note})" if note else query,
            width=width,
            height=height,
            creator=f"Contributor {rng.randrange(100):02d}",
            license=licence,
            license_url=licence_url,
            provider=self.name,
            query=query,
            thumbnail=path.resolve().as_uri(),
            extra={"synthetic": True, "note": note},
        )

    def _bad_candidates(self, query: str) -> list[ImageCandidate]:
        """Files that must be rejected by download validation (§11)."""
        out: list[ImageCandidate] = []
        slug = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]

        # 1. Real image, but under images.min_width.
        tiny = self.cache_dir / f"{slug}_tiny.jpg"
        if not tiny.exists():
            generate_image(tiny, _seed_for(query + "tiny"), width=320, height=200)
        out.append(self._candidate(tiny, query=query, index=900, note="too small"))

        # 2. Claims to be a JPEG, is not decodable.
        broken = self.cache_dir / f"{slug}_broken.jpg"
        if not broken.exists():
            broken.parent.mkdir(parents=True, exist_ok=True)
            broken.write_bytes(b"\xff\xd8\xff\xe0 this is not actually a jpeg body")
        out.append(self._candidate(broken, query=query, index=901, note="undecodable"))

        # 3. SVG, which §11 rejects outright.
        svg = self.cache_dir / f"{slug}_vector.svg"
        if not svg.exists():
            svg.parent.mkdir(parents=True, exist_ok=True)
            svg.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800"></svg>',
                encoding="utf-8",
            )
        out.append(self._candidate(svg, query=query, index=902, note="svg"))
        return out

    # -- search -------------------------------------------------------------

    def search(self, query: str, *, limit: int = 30, **kwargs) -> list[ImageCandidate]:
        from PIL import Image

        seed = _seed_for(query)
        rng = random.Random(seed)
        slug = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]

        candidates: list[ImageCandidate] = []

        # Duplicates and rejects are a *proportion* of the result set, not a
        # fixed count. A fixed count is realistic at limit=30 but swamps a
        # small request: at limit=10 it left one usable image per query, and a
        # segment ended up with nine candidates where §22 P3 requires fifteen.
        n_bad = (3 if limit >= 24 else 1) if self.include_bad else 0
        n_dupes = min(4, max(1, limit // 6))
        n_shared = min(3, max(1, limit // 10))
        unique_target = max(1, limit - n_bad - n_dupes - n_shared)

        # Images unique to this query.
        originals: list[Path] = []
        for i in range(unique_target):
            name = f"{slug}_{i:02d}.jpg"
            width = rng.choice([1000, 1200, 1400, 1600])
            height = int(width * rng.choice([0.5625, 0.6667, 0.75, 1.0]))
            path = self._ensure(
                name, seed=seed + i, width=width, height=height, label=f"{query[:28]} #{i}"
            )
            originals.append(path)
            candidates.append(self._candidate(path, query=query, index=i))

        # Near-duplicates, so dedupe has real work (§11). Kinds are taken in a
        # fixed order so a small n_dupes still covers the cheapest case first.
        if originals and n_dupes:
            base = originals[0]
            for offset, kind in enumerate(_DUPLICATE_KINDS[:n_dupes]):
                path = self._make_duplicate(base, slug, kind, Image)
                candidates.append(self._candidate(path, query=query, index=800 + offset, note=kind))

        # The invalid files come before the shared pool so that, if the caller
        # asked for very few results, the rejection paths still get exercised.
        if n_bad:
            bad = self._bad_candidates(query)
            if n_bad < len(bad):
                # Rotate by query so that, across a segment's several queries,
                # all three rejection paths still run at least once.
                start = _seed_for(query) % len(bad)
                bad = [bad[(start + i) % len(bad)] for i in range(n_bad)]
            candidates.extend(bad)

        # A few from the shared pool, which other queries also return.
        for name, path in self._shared_pool()[:n_shared]:
            candidates.append(self._candidate(path, query=query, index=700, note=f"shared {name}"))

        # Scatter the duplicates and the invalid files through the result set.
        # Built in order they sit at the end, and any caller that stops early
        # -- as the download stage does once it has its budget -- would never
        # reach them, so dedupe and the rejection paths would silently never
        # run. Real search results are not sorted by how well-formed they are.
        random.Random(seed ^ 0x5EED).shuffle(candidates)

        log.debug("fake image search %r -> %d candidate(s)", query, len(candidates))
        return candidates[:limit]

    def _make_duplicate(self, base: Path, slug: str, kind: str, Image) -> Path:
        """Create (once) a near-duplicate of ``base`` of the given kind."""
        path = self.cache_dir / f"{slug}_dup_{kind.replace(' ', '_')}.jpg"
        if path.exists():
            return path

        if kind == "exact duplicate":
            path.write_bytes(base.read_bytes())  # SHA256 catches this
        elif kind == "resized":
            with Image.open(base) as source:
                source.resize(
                    (int(source.width * 0.7), int(source.height * 0.7)), Image.LANCZOS
                ).save(path, "JPEG", quality=85)
        elif kind == "cropped":
            with Image.open(base) as source:
                inset_x, inset_y = int(source.width * 0.04), int(source.height * 0.04)
                source.crop(
                    (inset_x, inset_y, source.width - inset_x, source.height - inset_y)
                ).save(path, "JPEG", quality=88)
        elif kind == "recompressed":
            with Image.open(base) as source:
                source.save(path, "JPEG", quality=35)
        return path
