"""Stage 4: entity resolution (CLAUDE.md §10) -- the highest-value component.

A transcriber hears "Barriss" where the narrator said "Darth Baras". Searching
for "Barriss" returns a Jedi from a different era and every image is wrong. So
this stage exists, and §10 constrains it tightly:

1. Never blindly replace. Every correction carries original, resolved,
   confidence and reason.
2. **The transcript on disk keeps the ORIGINAL words.** Corrections apply only
   to search queries. Nothing in this module writes to transcript.json.
3. Applied at confidence >= 0.75. Below that, both spellings are searched and
   ranking decides.
4. Order: domain pack (exact, then fuzzy) -> phonetic -> LLM.
5. ``entity_overrides.yaml`` is user-edited, always wins, permanently.

The ordering in rule four is not a preference, it is a cost and reliability
ranking: an exact alias is free and certain, phonetics are free and plausible,
the LLM costs money and can hallucinate. Each pass only sees what the previous
one could not resolve.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..logging_setup import get_logger
from ..providers.base import ProviderError
from ..providers.llm.base import LLMProvider
from ..schemas import EntityCorrection
from ..utils.files import atomic_write_text
from ..utils.phonetics import metaphone, normalize_name, similarity

__all__ = [
    "PackEntity",
    "DomainPack",
    "load_domain_packs",
    "select_packs",
    "Overrides",
    "load_overrides",
    "write_overrides_template",
    "EntityResolver",
    "resolve_entities",
    "apply_corrections",
    "extract_candidates",
    "APPLY_THRESHOLD",
    "FUZZY_THRESHOLD",
]

log = get_logger("pipeline.entities")

#: §10.3 -- corrections apply at or above this confidence.
APPLY_THRESHOLD = 0.75

#: Edit-similarity needed before two spellings are considered the same name.
#: 0.86 accepts "Tattooine"/"Tatooine" and rejects "Baras"/"Barris" being the
#: same as unrelated short names.
FUZZY_THRESHOLD = 0.86

#: Multipliers applied to a pack entry's confidence, by how the match was made.
#: An exact alias is what the pack author wrote down; everything else is
#: inference and is discounted accordingly.
_METHOD_WEIGHT = {
    "override": 1.0,
    "domain_pack_exact": 1.0,
    "domain_pack_fuzzy": 0.92,
    "phonetic": 0.88,
    "llm": 1.0,
}

_PROPER_NOUN = re.compile(r"\b[A-Z][A-Za-z'’-]{2,}(?:\s+[A-Z][A-Za-z'’-]{2,})?\b")

#: Capitalised words that are usually just the start of a sentence.
_STOPWORDS = frozenset(
    """
    the a an and but or when what where who whom whose that this these those
    he she they it his her their its there here every none nobody everyone
    someone anything nothing for from with without before after during if
    because while as at by in into on of to up down then now so yet still even
    one two three four five first second last next also however although
    though since until unless whether both each other another such same
    just only very much many most more less least well back over under about
    above below between through against within across behind beyond
    i you we me us him them my your our mine yours ours
    is are was were be been being have has had do does did will would can
    could should may might must shall
    """.split()
)


# ---------------------------------------------------------------------------
# Domain packs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackEntity:
    canonical: str
    type: str = "character"
    aliases: tuple[str, ...] = ()
    confidence: float = 0.8
    reason: str = ""
    evidence: tuple[str, ...] = ()
    search_hints: tuple[str, ...] = ()
    also_known_as: tuple[str, ...] = ()

    def all_spellings(self) -> tuple[str, ...]:
        return (self.canonical, *self.aliases)


@dataclass(frozen=True)
class DomainPack:
    name: str
    title: str = ""
    franchise: str = ""
    era: str = ""
    #: Short qualifier for search queries, e.g. "SWTOR" (§7's examples).
    search_tag: str = ""
    triggers: tuple[str, ...] = ()
    min_triggers: int = 2
    entities: tuple[PackEntity, ...] = ()
    terminology: tuple[str, ...] = ()
    works: tuple[dict, ...] = ()
    path: Path | None = None

    def trigger_hits(self, text: str) -> list[str]:
        """Which triggers appear in ``text``, matched case-insensitively on word boundaries."""
        lowered = text.lower()
        hits = []
        for trigger in self.triggers:
            pattern = r"\b" + re.escape(trigger.lower()) + r"\b"
            if re.search(pattern, lowered):
                hits.append(trigger)
        return hits


def _as_tuple(value) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def load_domain_packs(directory: Path) -> list[DomainPack]:
    """Load every ``*.yaml`` in ``directory``.

    A malformed pack is logged and skipped rather than raised: one bad file
    must not stop a job (§8).
    """
    packs: list[DomainPack] = []
    directory = Path(directory)
    if not directory.exists():
        log.warning("no domain_packs directory at %s", directory)
        return packs

    for path in sorted(directory.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                raise ValueError("pack must be a YAML mapping")
            entities = tuple(
                PackEntity(
                    canonical=str(item["canonical"]),
                    type=str(item.get("type", "character")),
                    aliases=tuple(str(a) for a in _as_tuple(item.get("aliases"))),
                    confidence=float(item.get("confidence", 0.8)),
                    reason=str(item.get("reason", "")),
                    evidence=tuple(str(e) for e in _as_tuple(item.get("evidence"))),
                    search_hints=tuple(str(h) for h in _as_tuple(item.get("search_hints"))),
                    also_known_as=tuple(str(a) for a in _as_tuple(item.get("also_known_as"))),
                )
                for item in raw.get("entities", [])
            )
            packs.append(
                DomainPack(
                    name=str(raw.get("name") or path.stem),
                    title=str(raw.get("title", "")),
                    franchise=str(raw.get("franchise", "")),
                    era=str(raw.get("era", "")),
                    search_tag=str(raw.get("search_tag", "")),
                    triggers=tuple(str(t) for t in _as_tuple(raw.get("triggers"))),
                    min_triggers=int(raw.get("min_triggers", 2)),
                    entities=entities,
                    terminology=tuple(str(t) for t in _as_tuple(raw.get("terminology"))),
                    works=tuple(raw.get("works") or ()),
                    path=path,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad pack must not stop a job
            log.error("skipping domain pack %s: %s", path.name, exc)
    log.info("loaded %d domain pack(s) from %s", len(packs), directory)
    return packs


def select_packs(text: str, packs: list[DomainPack]) -> list[DomainPack]:
    """Activate the packs whose triggers appear often enough, most specific first.

    A pack that does not fire contributes nothing -- loading star_wars.yaml for
    a cookery video must not start renaming people to Darth Vader.

    Ordering matters as much as selection. Both ``star_wars`` and ``swtor``
    fire on a SWTOR narration, and whichever comes first sets the project's
    subject, which is then appended to every search query. Trigger-hit count
    is the specificity signal: the pack that matched the narration more
    closely leads. Without this the subject tag reads "Star Wars" rather than
    "Star Wars: The Old Republic", and every query loses an era's worth of
    precision.
    """
    scored: list[tuple[int, int, DomainPack]] = []
    for index, pack in enumerate(packs):
        hits = pack.trigger_hits(text)
        if len(hits) >= pack.min_triggers:
            scored.append((len(hits), index, pack))
            log.info(
                "domain pack %r active (%d triggers: %s)",
                pack.name,
                len(hits),
                ", ".join(hits[:5]),
            )
        else:
            log.debug("domain pack %r not active (%d triggers)", pack.name, len(hits))

    # Most hits first; original order breaks ties so selection stays deterministic.
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [pack for _, _, pack in scored]


# ---------------------------------------------------------------------------
# User overrides (§10.5)
# ---------------------------------------------------------------------------


@dataclass
class Overrides:
    """``projects/<name>/entity_overrides.yaml``. Always wins, permanently."""

    corrections: dict[str, str] = field(default_factory=dict)
    never_correct: set[str] = field(default_factory=set)
    path: Path | None = None

    def lookup(self, original: str) -> str | None:
        return self.corrections.get(normalize_name(original))

    def is_blocked(self, original: str) -> bool:
        return normalize_name(original) in self.never_correct


def load_overrides(path: Path) -> Overrides:
    """Read the user's overrides. A missing or broken file is never fatal."""
    path = Path(path)
    if not path.exists():
        return Overrides(path=path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("entity_overrides.yaml must be a YAML mapping")
        corrections: dict[str, str] = {}
        for item in raw.get("corrections") or []:
            if isinstance(item, dict) and item.get("original") and item.get("resolved"):
                corrections[normalize_name(str(item["original"]))] = str(item["resolved"])
        never = {normalize_name(str(n)) for n in (raw.get("never_correct") or [])}
        log.info(
            "loaded %d override(s) and %d never-correct entr(y/ies) from %s",
            len(corrections),
            len(never),
            path.name,
        )
        return Overrides(corrections=corrections, never_correct=never, path=path)
    except Exception as exc:  # noqa: BLE001
        log.error("ignoring malformed %s: %s", path.name, exc)
        return Overrides(path=path)


OVERRIDES_TEMPLATE = """\
# Entity overrides for this project. You edit this file; it always wins, and
# it keeps winning on every re-run (CLAUDE.md §10.5).
#
# Anything listed here is applied at full confidence, ahead of the domain
# packs, the phonetic matcher and the LLM.
#
# corrections:
#   - original: Barriss        # what the transcript says
#     resolved: Darth Baras    # what to search for instead
#
# never_correct:               # leave these spellings exactly as transcribed
#   - Vette

corrections: []
never_correct: []
"""


def write_overrides_template(path: Path) -> bool:
    """Create the overrides file if it is absent. Never overwrites user edits."""
    path = Path(path)
    if path.exists():
        return False
    atomic_write_text(path, OVERRIDES_TEMPLATE)
    return True


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------


def extract_candidates(text: str) -> dict[str, int]:
    """Proper-noun-looking phrases and how often each occurs.

    A word that only ever appears at the start of a sentence, and is a common
    English word, is not a name. Getting this wrong means trying to "resolve"
    the word "The", which wastes an LLM call and risks a nonsense correction.
    """
    counts: dict[str, int] = {}
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for match in _PROPER_NOUN.finditer(sentence):
            phrase = match.group(0).strip()
            words = phrase.split()
            # Drop a leading stopword: "The Dark Council" -> "Dark Council".
            while words and words[0].lower() in _STOPWORDS:
                words = words[1:]
            if not words:
                continue
            # A single capitalised stopword at sentence start is not a name.
            if len(words) == 1 and words[0].lower() in _STOPWORDS:
                continue
            phrase = " ".join(words)
            if len(phrase) < 3:
                continue
            counts[phrase] = counts.get(phrase, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class EntityResolver:
    """Runs §10.4's four passes in order, each seeing only what is still unresolved."""

    def __init__(
        self,
        packs: list[DomainPack],
        *,
        overrides: Overrides | None = None,
        llm: LLMProvider | None = None,
        apply_threshold: float = APPLY_THRESHOLD,
    ) -> None:
        self.packs = packs
        self.overrides = overrides or Overrides()
        self.llm = llm
        self.apply_threshold = apply_threshold

        # Flat indexes across active packs.
        self._by_alias: dict[str, tuple[PackEntity, str]] = {}
        self._by_phonetic: dict[str, list[tuple[PackEntity, str]]] = {}
        self._canonical: set[str] = set()
        for pack in packs:
            for entity in pack.entities:
                self._canonical.add(normalize_name(entity.canonical))
                for spelling in entity.all_spellings():
                    key = normalize_name(spelling)
                    self._by_alias.setdefault(key, (entity, pack.name))
                    phon = metaphone(spelling)
                    if phon:
                        self._by_phonetic.setdefault(phon, []).append((entity, pack.name))

    # -- individual passes -------------------------------------------------

    def _from_override(self, candidate: str) -> EntityCorrection | None:
        resolved = self.overrides.lookup(candidate)
        if not resolved or normalize_name(resolved) == normalize_name(candidate):
            return None
        return EntityCorrection(
            original=candidate,
            resolved=resolved,
            confidence=1.0,
            reason="User override in entity_overrides.yaml.",
            evidence=["entity_overrides.yaml"],
            method="override",
            # §10.5: an override always wins, permanently. It is applied
            # regardless of apply_threshold -- the user has already decided.
            applied=True,
        )

    def _from_pack_exact(self, candidate: str) -> EntityCorrection | None:
        hit = self._by_alias.get(normalize_name(candidate))
        if hit is None:
            return None
        entity, pack_name = hit
        if normalize_name(entity.canonical) == normalize_name(candidate):
            return None  # already canonical; nothing to correct
        return self._build(candidate, entity, pack_name, "domain_pack_exact")

    def _from_pack_fuzzy(self, candidate: str) -> EntityCorrection | None:
        best: tuple[float, PackEntity, str, str] | None = None
        for key, (entity, pack_name) in self._by_alias.items():
            ratio = similarity(candidate, key)
            if ratio >= FUZZY_THRESHOLD and (best is None or ratio > best[0]):
                best = (ratio, entity, pack_name, key)
        if best is None:
            return None
        ratio, entity, pack_name, matched = best
        if normalize_name(entity.canonical) == normalize_name(candidate):
            return None
        return self._build(
            candidate,
            entity,
            pack_name,
            "domain_pack_fuzzy",
            extra_reason=f"Spelling is {ratio:.0%} similar to the known form {matched!r}.",
            scale=ratio,
        )

    def _from_phonetic(self, candidate: str) -> EntityCorrection | None:
        key = metaphone(candidate)
        if not key:
            return None
        matches = self._by_phonetic.get(key)
        if not matches:
            return None
        entity, pack_name = matches[0]
        if normalize_name(entity.canonical) == normalize_name(candidate):
            return None
        return self._build(
            candidate,
            entity,
            pack_name,
            "phonetic",
            extra_reason=f"Same metaphone key ({key}) as the known form.",
        )

    def _build(
        self,
        candidate: str,
        entity: PackEntity,
        pack_name: str,
        method: str,
        *,
        extra_reason: str = "",
        scale: float = 1.0,
    ) -> EntityCorrection:
        confidence = min(1.0, entity.confidence * _METHOD_WEIGHT[method] * scale)
        reason = entity.reason or f"Known {entity.type} in the {pack_name} domain pack."
        if extra_reason:
            reason = f"{reason} {extra_reason}"
        evidence = list(entity.evidence) or [f"domain_pack:{pack_name}"]
        return EntityCorrection(
            original=candidate,
            resolved=entity.canonical,
            confidence=round(confidence, 3),
            reason=reason,
            evidence=evidence,
            method=method,
            applied=confidence >= self.apply_threshold,
        )

    # -- the LLM pass ------------------------------------------------------

    def _from_llm(self, candidates: list[str], transcript_text: str) -> list[EntityCorrection]:
        """Ask the model about whatever the cheap passes could not resolve.

        One call for all remaining candidates, with the full transcript as
        context (§10.4). A provider failure degrades to "no corrections" rather
        than failing the job (§8).
        """
        if not self.llm or not candidates:
            return []

        system = (
            "You correct speech-to-text mishearings of proper nouns in a video "
            "essay narration. You are given the full transcript and a list of "
            "names as transcribed. For each name that is clearly a mishearing "
            "of a different, specific, real name in this subject matter, give "
            "the corrected spelling, a confidence from 0 to 1, and a one-line "
            "reason citing the evidence in the transcript. If a name is already "
            "correct, or you are not confident which name was meant, omit it "
            "entirely. Never guess. Returning fewer corrections is always "
            "better than returning a wrong one."
        )
        user = f"TRANSCRIPT:\n{transcript_text}\n\nNAMES AS TRANSCRIBED: {json.dumps(candidates)}"
        schema = {
            "corrections": [
                {
                    "original": "string",
                    "resolved": "string",
                    "confidence": 0.0,
                    "reason": "string",
                }
            ]
        }
        try:
            response = self.llm.complete_json(
                system=system, user=user, task="entity_resolution", schema_hint=schema
            )
        except ProviderError as exc:
            log.warning("entity resolution: LLM unavailable, continuing without it (%s)", exc)
            return []

        out: list[EntityCorrection] = []
        for item in response.get("corrections", []) or []:
            try:
                original = str(item["original"]).strip()
                resolved = str(item["resolved"]).strip()
                confidence = float(item.get("confidence", 0.0))
            except (KeyError, TypeError, ValueError):
                log.debug("entity resolution: skipping malformed LLM item %r", item)
                continue
            if not original or not resolved:
                continue
            if normalize_name(original) == normalize_name(resolved):
                continue
            out.append(
                EntityCorrection(
                    original=original,
                    resolved=resolved,
                    confidence=round(min(max(confidence, 0.0), 1.0), 3),
                    reason=str(item.get("reason", "")).strip() or "Resolved by language model.",
                    evidence=["llm"],
                    method="llm",
                    applied=confidence >= self.apply_threshold,
                )
            )
        return out

    # -- entry point -------------------------------------------------------

    def resolve(self, transcript_text: str) -> list[EntityCorrection]:
        """All corrections for a transcript, ordered by descending confidence."""
        candidates = extract_candidates(transcript_text)
        log.info("entity resolution: %d distinct candidate name(s)", len(candidates))

        corrections: list[EntityCorrection] = []
        unresolved: list[str] = []

        for candidate in sorted(candidates, key=lambda c: (-candidates[c], c)):
            if self.overrides.is_blocked(candidate):
                log.debug("candidate %r is in never_correct; leaving it alone", candidate)
                continue

            correction = (
                self._from_override(candidate)
                or self._from_pack_exact(candidate)
                or self._from_pack_fuzzy(candidate)
                or self._from_phonetic(candidate)
            )
            if correction is not None:
                corrections.append(correction)
                continue
            # Already canonical means nothing to resolve, and nothing to ask about.
            if normalize_name(candidate) not in self._canonical:
                unresolved.append(candidate)

        llm_corrections = self._from_llm(unresolved, transcript_text)
        # A cheap pass already spoke for these; do not let the LLM overrule it.
        seen = {normalize_name(c.original) for c in corrections}
        corrections.extend(c for c in llm_corrections if normalize_name(c.original) not in seen)

        corrections.sort(key=lambda c: (-c.confidence, c.original))
        applied = sum(1 for c in corrections if c.applied)
        log.info(
            "entity resolution: %d correction(s), %d applied at >= %.2f confidence",
            len(corrections),
            applied,
            self.apply_threshold,
        )
        for correction in corrections:
            log.debug(
                "  %-22s -> %-22s %.2f  %-18s %s",
                correction.original,
                correction.resolved,
                correction.confidence,
                correction.method,
                "APPLIED" if correction.applied else "search both spellings",
            )
        return corrections


def resolve_entities(
    transcript_text: str,
    packs: list[DomainPack],
    *,
    overrides: Overrides | None = None,
    llm: LLMProvider | None = None,
    apply_threshold: float = APPLY_THRESHOLD,
) -> list[EntityCorrection]:
    """Convenience wrapper around :class:`EntityResolver`."""
    return EntityResolver(
        packs, overrides=overrides, llm=llm, apply_threshold=apply_threshold
    ).resolve(transcript_text)


# ---------------------------------------------------------------------------
# Applying corrections -- to QUERIES only (§10.2)
# ---------------------------------------------------------------------------


def apply_corrections(text: str, corrections: list[EntityCorrection]) -> str:
    """Rewrite ``text`` using the applied corrections.

    **Only ever called on search query text.** The transcript on disk keeps the
    original words; §10.2 is not negotiable and there is a test that fails if
    this function is ever pointed at transcript.json.

    Replacement is whole-word and case-insensitive, longest original first so
    that "Lord Drog" is handled before "Drog".
    """
    applied = [c for c in corrections if c.applied]
    if not applied:
        return text
    for correction in sorted(applied, key=lambda c: -len(c.original)):
        pattern = re.compile(rf"\b{re.escape(correction.original)}\b", re.IGNORECASE)
        text = pattern.sub(correction.resolved, text)
    return text


def both_spellings(text: str, corrections: list[EntityCorrection]) -> list[str]:
    """For sub-threshold corrections, the variants worth searching (§10.3).

    Returns the original text first, then one variant per unapplied correction
    whose original appears in the text. Ranking decides between them.
    """
    variants = [text]
    for correction in corrections:
        if correction.applied:
            continue
        pattern = re.compile(rf"\b{re.escape(correction.original)}\b", re.IGNORECASE)
        if pattern.search(text):
            variant = pattern.sub(correction.resolved, text)
            if variant not in variants:
                variants.append(variant)
    return variants
