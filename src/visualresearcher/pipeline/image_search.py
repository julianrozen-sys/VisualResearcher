"""Stage 7: image search (CLAUDE.md §11).

Runs each segment's queries across every configured provider and collects
candidates. Two rules shape it:

* **one failing provider never stops a segment.** A provider that raises is
  logged, the segment is noted, and the remaining providers still run;
* **the candidate budget is spread across query kinds**, not spent on the
  first query. The exact-event query is the most likely to find the right
  shot, but if all thirty slots go to it, a segment whose exact-event query
  finds nothing ends up with nothing at all.

Results are interleaved by query so that the kept set keeps the variety §11's
diversity rule will later need.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..config import Settings
from ..logging_setup import get_logger
from ..providers.base import ProviderError
from ..providers.images.base import ImageCandidate, ImageSearchProvider
from ..providers.registry import resolve
from ..schemas import ImageRecord, Segment
from ..utils.files import atomic_write_text

__all__ = [
    "search_segment",
    "collect_providers",
    "write_candidate_manifest",
    "read_candidate_manifest",
]

log = get_logger("pipeline.image_search")


def collect_providers(settings: Settings) -> list[ImageSearchProvider]:
    """The configured image providers, offline-substituted where needed.

    Under ``VR_OFFLINE=1`` the registry maps every name to the fake, so this
    is de-duplicated by identity to avoid running the same fake three times.
    """
    providers: list[ImageSearchProvider] = []
    seen: set[str] = set()
    for name in settings.images.providers:
        try:
            provider = resolve("images", name, min_width=settings.images.min_width)
        except ProviderError as exc:
            log.warning("image provider %r unusable: %s", name, exc)
            continue
        if provider.name in seen:
            continue
        seen.add(provider.name)
        providers.append(provider)  # type: ignore[arg-type]

    if not providers:
        log.error("no image providers available at all")
    else:
        log.info("image providers: %s", ", ".join(p.name for p in providers))
    return providers


def _query_budget(provider: ImageSearchProvider, settings: Settings) -> int:
    """How many of a segment's queries this provider may be asked.

    Unlimited by default; ``images.max_queries_per_provider`` narrows named
    providers. Keeps every provider present in every segment, just not at the
    same call volume.
    """
    caps = getattr(settings.images, "max_queries_per_provider", None) or {}
    return int(caps.get(provider.name, 1_000_000))


def search_segment(
    segment: Segment,
    providers: list[ImageSearchProvider],
    settings: Settings,
) -> tuple[list[ImageCandidate], list[str]]:
    """Gather candidates for one segment. Returns ``(candidates, notes)``.

    ``notes`` records any provider that failed, so the segment can be marked
    degraded rather than silently returning fewer results.
    """
    budget = settings.images.candidates_per_segment
    queries = segment.queries or []
    notes: list[str] = []

    if not queries:
        return [], ["no queries for this segment"]
    if not providers:
        return [], ["no image providers available"]

    # Spread the budget across queries and providers, with a little headroom
    # so that dedupe and validation losses do not starve the segment.
    per_call = max(4, math.ceil(budget * 1.6 / (len(queries) * len(providers))))

    #: query index -> candidates, so results can be interleaved afterwards.
    buckets: list[list[ImageCandidate]] = [[] for _ in queries]
    seen: set[str] = set()

    for query_index, query in enumerate(queries):
        for provider in providers:
            if query_index >= _query_budget(provider, settings):
                # Rate-limit triage, measured on the first live run. Call count
                # is queries x providers, so the only lever that cuts a slow
                # provider's load is calling it on fewer queries -- lowering
                # `candidates_per_segment` changes `limit=` per call and not
                # the number of calls, which is what earns the 429s.
                #
                # Commons is capped rather than dropped because it is the only
                # provider that returns a real licence: ddgs answered 68 of 68
                # with `license=unknown`. Demoting it to a fallback would make
                # sources.csv mostly unattributable, which is §21's whole point.
                continue
            try:
                hits = provider.search(query.text, limit=per_call)
            except ProviderError as exc:
                message = f"{provider.name} failed on {query.text!r}: {exc}"
                log.warning("segment %03d: %s", segment.index, message)
                notes.append(message)
                continue
            except Exception as exc:  # noqa: BLE001 - a provider must never kill a segment
                message = f"{provider.name} raised on {query.text!r}: {exc}"
                log.warning("segment %03d: %s", segment.index, message)
                notes.append(message)
                continue

            for hit in hits:
                key = hit.dedupe_key()
                if not hit.image_url or key in seen:
                    continue
                seen.add(key)
                hit.query = query.text
                hit.query_kind = str(query.kind)
                hit.segment_index = getattr(hit, "segment_index", segment.index)
                buckets[query_index].append(hit)

    # Interleave: one from each query in turn, so the budget is not eaten by
    # whichever query happened to return most.
    interleaved: list[ImageCandidate] = []
    for position in range(max((len(b) for b in buckets), default=0)):
        for bucket in buckets:
            if position < len(bucket):
                interleaved.append(bucket[position])

    log.info(
        "segment %03d: %d candidate(s) from %d quer(y/ies) across %d provider(s)",
        segment.index,
        len(interleaved),
        len(queries),
        len(providers),
    )
    return interleaved, notes


def _relative(record_dump: dict, root: Path | None) -> dict:
    """Rewrite ``local_path`` as a project-relative POSIX path.

    The manifest is part of the deliverable folder, so it must survive that
    folder being moved or copied to another machine. An absolute path also
    makes two otherwise-identical runs differ, which hides real regressions
    when comparing a resumed run against an uninterrupted one.
    """
    if not root:
        return record_dump
    raw = record_dump.get("local_path") or ""
    if not raw:
        return record_dump
    try:
        record_dump["local_path"] = Path(raw).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        pass  # outside the project; leave it as it is rather than mangling it
    return record_dump


def write_candidate_manifest(path, records, rejected=None, duplicates=None, root=None) -> None:
    """``images/manifest.json`` -- every record, kept and discarded (§11).

    Discards stay in the file. A manifest that only lists survivors cannot
    answer "why is there no picture of X", which is the question the review UI
    exists to answer.
    """
    payload = {
        "kept": [_relative(r.model_dump(), root) for r in records],
        "duplicates": [_relative(r.model_dump(), root) for r in (duplicates or [])],
        "rejected": [
            {
                "image_url": o.candidate.image_url if o.candidate else "",
                "source_page": o.candidate.source_page if o.candidate else "",
                "provider": o.candidate.provider if o.candidate else "",
                "query": o.candidate.query if o.candidate else "",
                "reason": o.reason,
                "detail": o.detail,
            }
            for o in (rejected or [])
        ],
        "counts": {
            "kept": len(records),
            "duplicates": len(duplicates or []),
            "rejected": len(rejected or []),
        },
    }
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def read_candidate_manifest(path, root=None) -> dict[str, list[ImageRecord]]:
    """Read a manifest back, restoring absolute paths from ``root``.

    The mirror of :func:`write_candidate_manifest`. Later stages (ranking,
    the review UI, collection) need to open the files, so the relative paths
    stored on disk are resolved here in one place rather than at each use.
    """
    from ..schemas import ImageRecord

    path = Path(path)
    if not path.exists():
        return {"kept": [], "duplicates": []}
    payload = json.loads(path.read_text(encoding="utf-8"))

    def restore(items):
        out = []
        for item in items or []:
            record = ImageRecord.model_validate(item)
            if root and record.local_path and not Path(record.local_path).is_absolute():
                record.local_path = str(Path(root) / record.local_path)
            out.append(record)
        return out

    return {
        "kept": restore(payload.get("kept")),
        "duplicates": restore(payload.get("duplicates")),
    }
