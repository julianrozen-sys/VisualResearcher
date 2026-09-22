"""Entity resolution tests (CLAUDE.md §10, §21).

§21 names four requirements: the four known pairs resolve, sub-0.75 is NOT
applied, overrides always win, and the transcript is never overwritten. All
four are here, plus the ordering guarantees from §10.4.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from visualresearcher.pipeline.entities import (
    APPLY_THRESHOLD,
    DomainPack,
    Overrides,
    PackEntity,
    apply_corrections,
    both_spellings,
    extract_candidates,
    load_domain_packs,
    load_overrides,
    resolve_entities,
    select_packs,
    write_overrides_template,
)
from visualresearcher.providers.base import Availability, ProviderError
from visualresearcher.providers.llm.base import LLMProvider, LLMResponse

PACK_DIR = Path(__file__).resolve().parents[1] / "domain_packs"

# The narration the fake transcriber produces: contains all four mishearings.
SWTOR_TEXT = (
    "When the Sith Warrior first walks into the chamber, the Dark Council is "
    "already waiting. Barriss stands at the centre of the circle, wrapped in "
    "that heavy mask he never removes. Valron watches from the far side, "
    "saying nothing. Sanx had given up the location of the safe house. "
    "Drog fell on Hoth, and with him went the last witness. The duel plays "
    "out beneath the Korriban academy."
)


@pytest.fixture
def packs() -> list[DomainPack]:
    return load_domain_packs(PACK_DIR)


@pytest.fixture
def active(packs) -> list[DomainPack]:
    return select_packs(SWTOR_TEXT, packs)


# ---------------------------------------------------------------------------
# Domain packs
# ---------------------------------------------------------------------------


def test_the_shipped_packs_load(packs):
    names = {p.name for p in packs}
    assert {"swtor", "star_wars"} <= names
    swtor = next(p for p in packs if p.name == "swtor")
    assert swtor.franchise == "Star Wars"
    assert len(swtor.entities) >= 8


def test_every_pack_entity_has_a_reason_and_a_confidence(packs):
    """§10.1 -- a correction without a stated reason is a blind replacement."""
    for pack in packs:
        for entity in pack.entities:
            assert entity.reason.strip(), f"{pack.name}:{entity.canonical} has no reason"
            assert 0.0 < entity.confidence <= 1.0, (
                f"{pack.name}:{entity.canonical} has confidence {entity.confidence}"
            )


def test_a_malformed_pack_is_skipped_not_raised(tmp_path):
    (tmp_path / "good.yaml").write_text(
        yaml.safe_dump(
            {"name": "good", "entities": [{"canonical": "X", "reason": "r", "confidence": 0.9}]}
        ),
        encoding="utf-8",
    )
    (tmp_path / "broken.yaml").write_text("entities: [{no_canonical: true}]", encoding="utf-8")
    (tmp_path / "notamapping.yaml").write_text("- a\n- b\n", encoding="utf-8")

    packs = load_domain_packs(tmp_path)
    assert [p.name for p in packs] == ["good"], "one bad pack must not stop the others"


def test_packs_activate_only_when_triggers_appear(packs):
    assert {p.name for p in select_packs(SWTOR_TEXT, packs)} >= {"swtor"}
    unrelated = "Today we are making a sourdough loaf with a rye starter."
    assert select_packs(unrelated, packs) == [], (
        "a cookery narration must not activate the Star Wars packs"
    )


def test_a_single_trigger_is_not_enough(packs):
    """min_triggers guards against one incidental mention hijacking a project."""
    barely = "He mentioned Korriban once, in passing, during an unrelated talk."
    assert select_packs(barely, packs) == []


# ---------------------------------------------------------------------------
# §21: the four known pairs resolve, with reasons
# ---------------------------------------------------------------------------

KNOWN_PAIRS = [
    ("Barriss", "Darth Baras"),
    ("Valron", "Darth Vowrawn"),
    ("Sanx", "Colonel Senks"),
    ("Drog", "Lord Draahg"),
]


@pytest.mark.parametrize(("original", "resolved"), KNOWN_PAIRS)
def test_each_known_mishearing_resolves(active, original, resolved):
    corrections = resolve_entities(SWTOR_TEXT, active)
    match = next((c for c in corrections if c.original == original), None)
    assert match is not None, (
        f"{original!r} was not resolved at all; got "
        f"{[(c.original, c.resolved) for c in corrections]}"
    )
    assert match.resolved == resolved
    assert match.confidence >= APPLY_THRESHOLD
    assert match.applied is True


@pytest.mark.parametrize(("original", "resolved"), KNOWN_PAIRS)
def test_each_known_correction_states_a_reason_and_evidence(active, original, resolved):
    """§10.1 -- store original, resolved, confidence, reason."""
    corrections = resolve_entities(SWTOR_TEXT, active)
    match = next(c for c in corrections if c.original == original)
    assert match.reason.strip(), "a correction with no reason is a blind replacement"
    assert match.evidence, "a correction must cite its evidence"
    assert match.method in {"domain_pack_exact", "domain_pack_fuzzy", "phonetic", "llm"}


def test_all_four_resolve_in_a_single_pass(active):
    """The P2 acceptance criterion, stated as one assertion."""
    corrections = resolve_entities(SWTOR_TEXT, active)
    got = {c.original: c.resolved for c in corrections if c.applied}
    for original, resolved in KNOWN_PAIRS:
        assert got.get(original) == resolved, (
            f"expected {original} -> {resolved}, got {got.get(original)!r}"
        )


# ---------------------------------------------------------------------------
# §21: below 0.75 is NOT applied
# ---------------------------------------------------------------------------


def _pack_with(confidence: float) -> list[DomainPack]:
    return [
        DomainPack(
            name="test",
            triggers=("testing",),
            min_triggers=1,
            entities=(
                PackEntity(
                    canonical="Correct Name",
                    aliases=("Wrongname",),
                    confidence=confidence,
                    reason="test entry",
                ),
            ),
        )
    ]


def test_a_low_confidence_correction_is_recorded_but_not_applied():
    """§10.3 -- below 0.75 both spellings are searched and ranking decides."""
    corrections = resolve_entities("Wrongname appeared again.", _pack_with(0.60))
    match = next(c for c in corrections if c.original == "Wrongname")
    assert match.confidence < APPLY_THRESHOLD
    assert match.applied is False, "a sub-threshold correction must not be applied"


def test_a_high_confidence_correction_is_applied():
    corrections = resolve_entities("Wrongname appeared again.", _pack_with(0.90))
    match = next(c for c in corrections if c.original == "Wrongname")
    assert match.applied is True


@pytest.mark.parametrize("confidence", [0.74, 0.749])
def test_just_below_the_threshold_is_not_applied(confidence):
    corrections = resolve_entities("Wrongname appeared again.", _pack_with(confidence))
    assert next(c for c in corrections if c.original == "Wrongname").applied is False


def test_exactly_at_the_threshold_is_applied():
    corrections = resolve_entities("Wrongname appeared again.", _pack_with(0.75))
    assert next(c for c in corrections if c.original == "Wrongname").applied is True


def test_an_unapplied_correction_still_yields_both_spellings():
    corrections = resolve_entities("Wrongname appeared again.", _pack_with(0.60))
    variants = both_spellings("images of Wrongname", corrections)
    assert "images of Wrongname" in variants
    assert "images of Correct Name" in variants, (
        "§10.3 requires both spellings to be searched below the threshold"
    )


def test_an_applied_correction_does_not_produce_a_second_variant():
    corrections = resolve_entities("Wrongname appeared again.", _pack_with(0.95))
    assert both_spellings("images of Wrongname", corrections) == ["images of Wrongname"]


# ---------------------------------------------------------------------------
# §21: overrides always win
# ---------------------------------------------------------------------------


def test_an_override_beats_the_domain_pack(active):
    overrides = Overrides(corrections={"barriss": "Barriss Offee"})
    corrections = resolve_entities(SWTOR_TEXT, active, overrides=overrides)
    match = next(c for c in corrections if c.original == "Barriss")
    assert match.resolved == "Barriss Offee", "the user's override must win over the pack"
    assert match.confidence == 1.0
    assert match.method == "override"
    assert match.applied is True


def test_an_override_wins_even_below_the_packs_confidence(active):
    overrides = Overrides(corrections={"drog": "Something Else Entirely"})
    corrections = resolve_entities(SWTOR_TEXT, active, overrides=overrides)
    match = next(c for c in corrections if c.original == "Drog")
    assert match.resolved == "Something Else Entirely"


def test_never_correct_leaves_a_spelling_alone(active):
    overrides = Overrides(never_correct={"barriss"})
    corrections = resolve_entities(SWTOR_TEXT, active, overrides=overrides)
    assert not any(c.original == "Barriss" for c in corrections), (
        "a never_correct entry must suppress the correction entirely"
    )
    # The others are untouched.
    assert any(c.original == "Drog" for c in corrections)


def test_overrides_file_round_trips(tmp_path):
    path = tmp_path / "entity_overrides.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "corrections": [{"original": "Barriss", "resolved": "Darth Baras"}],
                "never_correct": ["Vette"],
            }
        ),
        encoding="utf-8",
    )
    overrides = load_overrides(path)
    assert overrides.lookup("barriss") == "Darth Baras"
    assert overrides.lookup("BARRISS") == "Darth Baras"
    assert overrides.is_blocked("vette") is True


def test_a_malformed_overrides_file_is_ignored_not_fatal(tmp_path):
    path = tmp_path / "entity_overrides.yaml"
    path.write_text("corrections: this is not a list\n", encoding="utf-8")
    overrides = load_overrides(path)
    assert overrides.corrections == {}


def test_the_overrides_template_is_written_once_and_never_clobbers(tmp_path):
    path = tmp_path / "entity_overrides.yaml"
    assert write_overrides_template(path) is True
    path.write_text("corrections:\n  - original: A\n    resolved: B\n", encoding="utf-8")
    assert write_overrides_template(path) is False, "must never overwrite user edits"
    assert load_overrides(path).lookup("a") == "B"


def test_the_written_template_is_valid_yaml(tmp_path):
    path = tmp_path / "entity_overrides.yaml"
    write_overrides_template(path)
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert parsed == {"corrections": [], "never_correct": []}


# ---------------------------------------------------------------------------
# §21: the transcript is never overwritten (§10.2)
# ---------------------------------------------------------------------------


def test_corrections_apply_to_query_text_only(active):
    corrections = resolve_entities(SWTOR_TEXT, active)
    query = apply_corrections("Barriss on the Dark Council", corrections)
    assert query == "Darth Baras on the Dark Council"
    # The source text handed in is untouched -- strings are immutable, but the
    # point is that nothing in this module writes it back anywhere.
    assert "Barriss" in SWTOR_TEXT


def test_apply_corrections_is_whole_word_only(active):
    corrections = resolve_entities(SWTOR_TEXT, active)
    # "Drogher" contains "Drog" but is a different word.
    assert apply_corrections("Drogher", corrections) == "Drogher"
    assert apply_corrections("Drog", corrections) == "Lord Draahg"


def test_apply_corrections_is_case_insensitive(active):
    corrections = resolve_entities(SWTOR_TEXT, active)
    assert apply_corrections("BARRISS speaks", corrections) == "Darth Baras speaks"


def test_longer_originals_are_replaced_first():
    packs = [
        DomainPack(
            name="t",
            triggers=("x",),
            min_triggers=1,
            entities=(
                PackEntity(canonical="Lord Draahg", aliases=("Drog",), confidence=0.9, reason="r"),
                PackEntity(
                    canonical="Draahg the Elder",
                    aliases=("Lord Drog",),
                    confidence=0.95,
                    reason="r",
                ),
            ),
        )
    ]
    corrections = resolve_entities("Lord Drog and Drog were both there.", packs)
    result = apply_corrections("Lord Drog", corrections)
    assert result == "Draahg the Elder", (
        "the longer original must win, or 'Lord Drog' becomes 'Lord Lord Draahg'"
    )


def test_unapplied_corrections_do_not_rewrite_text():
    corrections = resolve_entities("Wrongname was there.", _pack_with(0.5))
    assert apply_corrections("a photo of Wrongname", corrections) == "a photo of Wrongname"


def test_the_transcribe_stage_never_calls_apply_corrections():
    """A structural guard on §10.2: the transcript writer must not import this."""
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "visualresearcher"
        / "pipeline"
        / "transcribe.py"
    ).read_text(encoding="utf-8")
    assert "apply_corrections" not in source
    assert "entities" not in source, (
        "transcribe.py must not reach into entity resolution; §10.2 requires the "
        "transcript on disk to keep the original words"
    )


# ---------------------------------------------------------------------------
# §10.4 ordering
# ---------------------------------------------------------------------------


def test_an_exact_alias_beats_a_phonetic_match(active):
    """Exact alias is free and certain; phonetics are inference."""
    corrections = resolve_entities(SWTOR_TEXT, active)
    assert next(c for c in corrections if c.original == "Barriss").method == "domain_pack_exact"


def test_phonetics_catch_a_spelling_the_pack_never_listed():
    """'Boriss' is not an alias, but it sounds like Baras."""
    packs = [
        DomainPack(
            name="t",
            triggers=("x",),
            min_triggers=1,
            entities=(
                PackEntity(canonical="Baras", aliases=(), confidence=0.9, reason="test entry"),
            ),
        )
    ]
    corrections = resolve_entities("Boriss took the chair.", packs)
    match = next((c for c in corrections if c.original == "Boriss"), None)
    assert match is not None, "phonetic matching should have caught this"
    assert match.resolved == "Baras"
    assert match.method in {"phonetic", "domain_pack_fuzzy"}


def test_a_canonical_name_is_never_corrected_to_itself(active):
    corrections = resolve_entities("Korriban is the Sith homeworld. " + SWTOR_TEXT, active)
    assert not any(c.original.lower() == c.resolved.lower() for c in corrections)
    assert not any(c.original == "Korriban" for c in corrections)


def test_candidate_extraction_ignores_sentence_opening_stopwords():
    counts = extract_candidates("The council waited. When Baras arrived, everyone stood.")
    assert "The" not in counts
    assert "When" not in counts
    assert "Baras" in counts


def test_candidate_extraction_strips_a_leading_article():
    counts = extract_candidates("He joined the Dark Council. The Dark Council agreed.")
    assert "Dark Council" in counts
    assert "The Dark Council" not in counts


# ---------------------------------------------------------------------------
# The LLM pass
# ---------------------------------------------------------------------------


class _StubLLM(LLMProvider):
    name = "stub"

    def __init__(self, payload: dict | None = None, *, fail: bool = False):
        self.payload = payload or {"corrections": []}
        self.fail = fail
        self.calls: list[dict] = []

    def availability(self) -> Availability:
        return Availability.available("stub")

    def complete_json(self, *, system, user, task, schema_hint=None, max_tokens=4096):
        self.calls.append({"task": task, "user": user})
        if self.fail:
            raise ProviderError("stub failure")
        return LLMResponse(self.payload, model="stub")


def test_the_llm_only_sees_names_the_cheap_passes_could_not_resolve(active):
    llm = _StubLLM()
    resolve_entities(SWTOR_TEXT, active, llm=llm)
    assert len(llm.calls) == 1, "§20 requires batching, not one call per name"
    asked = llm.calls[0]["user"]
    assert "Barriss" not in asked.split("NAMES AS TRANSCRIBED:")[1], (
        "a name already resolved by the domain pack must not be sent to the model"
    )


def test_an_llm_failure_degrades_instead_of_failing_the_job(active):
    llm = _StubLLM(fail=True)
    corrections = resolve_entities(SWTOR_TEXT, active, llm=llm)
    assert len(corrections) >= 4, "the domain pack results must survive an LLM failure"


def test_llm_corrections_are_accepted_when_well_formed():
    llm = _StubLLM(
        {
            "corrections": [
                {
                    "original": "Zorbo",
                    "resolved": "Zorba",
                    "confidence": 0.9,
                    "reason": "named later in the transcript",
                }
            ]
        }
    )
    corrections = resolve_entities("Zorbo arrived at the gate.", [], llm=llm)
    match = next(c for c in corrections if c.original == "Zorbo")
    assert match.resolved == "Zorba"
    assert match.method == "llm"
    assert match.applied is True


def test_malformed_llm_items_are_skipped():
    llm = _StubLLM(
        {
            "corrections": [
                {"original": "Zorbo"},  # no resolved
                {"resolved": "Zorba"},  # no original
                {"original": "Same", "resolved": "Same", "confidence": 0.9},  # no-op
                {"original": "Good", "resolved": "Better", "confidence": 0.8, "reason": "r"},
            ]
        }
    )
    corrections = resolve_entities("Zorbo and Good arrived.", [], llm=llm)
    assert [c.original for c in corrections] == ["Good"]


def test_an_llm_correction_cannot_overrule_a_domain_pack(active):
    llm = _StubLLM(
        {
            "corrections": [
                {
                    "original": "Barriss",
                    "resolved": "Barriss Offee",
                    "confidence": 0.99,
                    "reason": "model preference",
                }
            ]
        }
    )
    corrections = resolve_entities(SWTOR_TEXT, active, llm=llm)
    barriss = [c for c in corrections if c.original == "Barriss"]
    assert len(barriss) == 1, "there must be exactly one correction per name"
    assert barriss[0].resolved == "Darth Baras"


def test_the_offline_fake_llm_invents_no_corrections():
    """An offline stand-in that guessed would violate §10.1."""
    from visualresearcher.providers.llm.fake import FakeLLMProvider

    corrections = resolve_entities(
        "Zorbo and Klaxton arrived.", [], llm=FakeLLMProvider(fixture_dir=Path("nowhere"))
    )
    assert corrections == []
