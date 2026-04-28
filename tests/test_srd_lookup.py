"""Tests for the SRD stat-block lookup module.

Covers:
- Index build: every entry has a non-empty body and an Armor Class field
  (we only index things that look like stat blocks, not rules sections).
- Exact `scan` pass: word-boundary, plurals, longest-name precedence,
  multi-word names, returns Match.source == "exact".
- Fuzzy `scan` pass: deterministic difflib.get_close_matches catches
  typo'd names (`zomby` → `Zombie`, `skelaton` → `Skeleton`) without
  matching unrelated words, returns Match.source == "fuzzy".
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

    def test_word_boundary_avoids_exact_substring_false_positive(self) -> None:
        # "skeletal" must not match Skeleton via the exact pass — that
        # would be a regex bug. The fuzzy pass may surface it as a
        # caveat-flagged hit; the DM is the final filter for those.
        hits = srd_lookup.scan("The skeletal remains crumble.")
        # No exact-source hits: the regex must not false-positive on
        # word substrings like "skeletal".
        assert all(h.source == "fuzzy" for h in hits)

    def test_no_match_returns_empty(self) -> None:
        assert srd_lookup.scan("Just some prose, nothing dangerous here.") == []


class TestFuzzyScan:
    def test_misspelled_zombie(self) -> None:
        hits = srd_lookup.scan("3 zombys ambush the party.")
        assert len(hits) == 1
        assert hits[0].statblock.name == "Zombie"
        assert hits[0].source == "fuzzy"
        assert hits[0].original == "zombys"

    def test_misspelled_skeleton(self) -> None:
        hits = srd_lookup.scan("A skelaton emerges from the niche.")
        assert len(hits) == 1
        assert hits[0].statblock.name == "Skeleton"
        assert hits[0].source == "fuzzy"

    def test_unrelated_word_does_not_fuzzy_match(self) -> None:
        # Generic prose with no creature-adjacent words should produce
        # no hits even with the fuzzy pass.
        hits = srd_lookup.scan(
            "The corridor stretches on, dust thick in the air."
        )
        assert hits == []

    def test_short_token_skipped(self) -> None:
        # 3-letter tokens are below the fuzzy threshold to avoid noise.
        # "imp" is a real creature, but exact matching catches it; we
        # check that a token shorter than the cutoff doesn't fire.
        hits = srd_lookup.scan("ape")  # length 3, below cutoff
        # The exact pass would still match "Ape" → it's a valid creature
        # name. So we expect an EXACT match, not a fuzzy one.
        if hits:
            assert all(h.source == "exact" for h in hits)


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
