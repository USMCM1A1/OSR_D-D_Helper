"""Tests for the SRD stat-block lookup module.

Covers:
- Index build: every entry has a non-empty body and an Armor Class field
  (we only index things that look like stat blocks, not rules sections).
- `scan`: word-boundary exact match, plurals, longest-name precedence,
  multi-word names. The fuzzy pass was removed — it produced too many
  false positives on ordinary English ("Fight" → Wight, "death" →
  Dretch, "light" → Knight). Misspelt names now silently fail to
  match, which is easier to notice than a confidently-wrong hit.
- find / names accessors.
"""

from __future__ import annotations

import pytest

import srd_lookup
from srd_lookup import Match, StatBlock


@pytest.fixture(scope="module")
def index() -> dict:
    return srd_lookup._index()


class TestIndex:
    def test_index_non_empty(self, index: dict) -> None:
        # We expect ~300 creatures across the two source files.
        assert len(index) > 50

    def test_every_entry_has_armor_class(self, index: dict) -> None:
        for sb in index.values():
            assert "**Armor Class**" in sb.body

    def test_skeleton_present_with_full_body(self) -> None:
        sb = srd_lookup.find("Skeleton")
        assert sb is not None
        assert sb.name == "Skeleton"
        assert "**Armor Class** 13" in sb.body
        assert "**Hit Points** 13" in sb.body


class TestExactScan:
    def test_singular(self) -> None:
        hits = srd_lookup.scan("A lone Skeleton stands here.")
        assert len(hits) == 1
        assert hits[0].statblock.name == "Skeleton"
        assert hits[0].source == "exact"
        assert hits[0].original == "Skeleton"

    def test_plural(self) -> None:
        hits = srd_lookup.scan("3 Skeletons (MM p.272) stand dormant.")
        assert len(hits) == 1
        assert hits[0].statblock.name == "Skeleton"
        assert hits[0].source == "exact"

    def test_longest_name_wins(self) -> None:
        hits = srd_lookup.scan(
            "A minotaur skeleton stands beside a smaller skeleton."
        )
        names = [h.statblock.name for h in hits]
        assert "Minotaur Skeleton" in names
        # The plain Skeleton should also match the second occurrence.
        assert "Skeleton" in names

    def test_word_boundary_avoids_substring_false_positive(self) -> None:
        # "skeletal" must not match Skeleton — that would be a regex bug.
        # No fuzzy fallback exists, so the scan returns nothing.
        hits = srd_lookup.scan("The skeletal remains crumble.")
        assert hits == []

    def test_no_match_returns_empty(self) -> None:
        assert srd_lookup.scan("Just some prose, nothing dangerous here.") == []


class TestNoFuzzyMatching:
    """Misspelt or near-miss words must not silently match a creature.
    These tests pin the post-removal behaviour: if you typo a creature
    name, you get nothing — and the empty stat block area in the editor
    is the prompt to fix the spelling."""

    def test_misspelled_zombie_does_not_match(self) -> None:
        # Previously fuzzy-matched Zombie at ratio ~0.73.
        assert srd_lookup.scan("3 zombys ambush the party.") == []

    def test_misspelled_skeleton_does_not_match(self) -> None:
        # Previously fuzzy-matched Skeleton at ratio ~0.83.
        assert srd_lookup.scan("A skelaton emerges from the niche.") == []

    def test_ordinary_english_does_not_match(self) -> None:
        # The bug that motivated the removal: prose words that fuzzy-
        # matched creatures. "Fight" → Wight (~0.83), "death" → Dretch
        # (~0.73), "light" → Knight (~0.86), "count" → Scout (~0.83).
        prose = (
            "Fight to the death, will not pursue beyond room exits. "
            "Stone pegs count against difficult terrain. The room is "
            "dim with no light source, and is unusual in its width."
        )
        names = [h.statblock.name for h in srd_lookup.scan(prose)]
        for ghost in ("Wight", "Dretch", "Knight", "Scout", "Wraith"):
            assert ghost not in names


class TestParseEncounterDeclarations:
    """`parse_encounter_declarations` is anchored on the formal
    `<count> <Name>(s) (MM p.<page>)` pattern. It's the source of truth
    for both room enrichment and the encounter simulator's monster
    list; descriptive prose between declarations must not bleed in."""

    def test_single_declaration(self):
        entries = srd_lookup.parse_encounter_declarations(
            "3 Skeletons (MM p.272). Unaware — standing dormant."
        )
        assert len(entries) == 1
        assert entries[0].statblock.name == "Skeleton"
        assert entries[0].count_expr == "3"

    def test_multi_word_name(self):
        entries = srd_lookup.parse_encounter_declarations(
            "1 Giant Scorpion (MM p.327). Unaware — dormant in a "
            "hollow it has scraped in the pit's northern shadow."
        )
        # The "shadow" in the prose must NOT add a Shadow creature.
        names = [e.statblock.name for e in entries]
        assert names == ["Giant Scorpion"]
        assert entries[0].count_expr == "1"

    def test_singular_and_plural_resolve(self):
        a = srd_lookup.parse_encounter_declarations("1 Wight (MM p.300).")
        b = srd_lookup.parse_encounter_declarations("2 Wights (MM p. 300).")
        assert a[0].statblock.name == "Wight"
        assert b[0].statblock.name == "Wight"

    def test_lizardfolk_no_plural_s(self):
        # Lizardfolk is its own plural; the parser must handle that.
        entries = srd_lookup.parse_encounter_declarations(
            "4 Lizardfolk (MM p.204). Territorial."
        )
        assert len(entries) == 1
        assert entries[0].statblock.name == "Lizardfolk"
        assert entries[0].count_expr == "4"

    def test_multiple_declarations_one_line(self):
        entries = srd_lookup.parse_encounter_declarations(
            "1 Wight (MM p. 300) + 2 Ghouls (MM p. 148). All hostile."
        )
        pairs = [(e.statblock.name, e.count_expr) for e in entries]
        assert pairs == [("Wight", "1"), ("Ghoul", "2")]

    def test_dice_count(self):
        entries = srd_lookup.parse_encounter_declarations(
            "1d4 Goblins (MM p.166) ambush from the rafters."
        )
        assert entries[0].count_expr == "1d4"
        assert entries[0].statblock.name == "Goblin"

    def test_page_ref_without_p(self):
        entries = srd_lookup.parse_encounter_declarations(
            "2 Ghouls (MM 148). Snarling."
        )
        assert len(entries) == 1
        assert entries[0].statblock.name == "Ghoul"

    def test_prose_alone_returns_nothing(self):
        # No declaration → no entries, even if the prose contains
        # creature names.
        entries = srd_lookup.parse_encounter_declarations(
            "Three skeletons are described, but no formal declaration."
        )
        assert entries == []

    def test_shadow_in_prose_does_not_add_shadow(self):
        # Exact regression for the user's reported bug.
        entries = srd_lookup.parse_encounter_declarations(
            "1 Giant Scorpion (MM p.327). Unaware — dormant in a hollow "
            "it has scraped in the pit's northern shadow (DC 13 passive "
            "Perception to notice the bulk before it stirs). Becomes "
            "hostile the moment direct light strikes it."
        )
        names = [e.statblock.name for e in entries]
        assert names == ["Giant Scorpion"]
        for ghost in ("Shadow", "Knight", "Wight"):
            assert ghost not in names

    def test_bare_declaration_without_count_defaults_to_one(self):
        # Encounter text written without a leading count — "Giant
        # Scorpion (MM p.327)" rather than "1 Giant Scorpion (MM p.327)"
        # — must still resolve, with count_expr defaulting to "1".
        entries = srd_lookup.parse_encounter_declarations(
            "Giant Scorpion (MM p.327). Unaware — dormant in a hollow "
            "it has scraped in the pit's northern shadow."
        )
        assert len(entries) == 1
        assert entries[0].statblock.name == "Giant Scorpion"
        assert entries[0].count_expr == "1"

    def test_title_case_prose_word_before_name_is_peeled(self):
        # If a Title-Case prose word ends up adjacent to the declaration
        # ("Unaware Giant Scorpion (MM p.327)"), the parser peels it off
        # until the remainder resolves to a known stat block.
        entries = srd_lookup.parse_encounter_declarations(
            "Unaware Giant Scorpion (MM p.327) lurks in the pit."
        )
        assert len(entries) == 1
        assert entries[0].statblock.name == "Giant Scorpion"

    def test_unknown_creature_name_skipped(self):
        # If a declaration's name doesn't resolve, it's silently
        # dropped — the empty stat-block area is the cue to fix.
        entries = srd_lookup.parse_encounter_declarations(
            "1 Florbnax (MM p.999). A creature that doesn't exist."
        )
        assert entries == []

    def test_empty_text(self):
        assert srd_lookup.parse_encounter_declarations("") == []
        assert srd_lookup.parse_encounter_declarations("   \n") == []


class TestFindAndNames:
    def test_find_case_insensitive(self) -> None:
        a = srd_lookup.find("aboleth")
        b = srd_lookup.find("Aboleth")
        c = srd_lookup.find("ABOLETH")
        assert a is not None and a == b == c

    def test_find_unknown_returns_none(self) -> None:
        assert srd_lookup.find("Bogus Beast") is None

    def test_names_sorted_unique(self) -> None:
        all_names = srd_lookup.names()
        assert all_names == sorted(all_names)
        assert len(all_names) == len(set(all_names))
