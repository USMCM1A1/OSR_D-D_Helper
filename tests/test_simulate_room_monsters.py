"""Tests for editor_server._build_monsters_from_room.

The simulator pulls its monster list from the formal
`<count> <Name>(s) (MM p.<page>)` declarations in the room's
encounter_text via srd_lookup.parse_encounter_declarations. Descriptive
prose between declarations must not bleed into the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import editor_server
import srd_lookup


@dataclass
class _StubRoom:
    """Minimal room shape. statblocks is unused by the new simulator
    pipeline — included only so the StubRoom matches the real shape if
    we ever broaden the function."""
    encounter_text: str
    statblocks: str = ""


class TestDeclarationsOnly:
    def test_single_declaration(self):
        room = _StubRoom(
            encounter_text="3 Skeletons (MM p.272). Unaware — standing dormant."
        )
        monsters = editor_server._build_monsters_from_room(room)
        assert len(monsters) == 3
        assert all(m.name == "Skeleton" for m in monsters)

    def test_giant_scorpion_does_not_pull_phantom_shadow(self):
        # Exact reproduction of the user-reported bug.
        room = _StubRoom(
            encounter_text=(
                "1 Giant Scorpion (MM p.327). Unaware — dormant in a "
                "hollow it has scraped in the pit's northern shadow "
                "(DC 13 passive Perception to notice the bulk before "
                "it stirs). Becomes hostile the moment direct light "
                "strikes it or a creature drops into the pit."
            ),
        )
        monsters = editor_server._build_monsters_from_room(room)
        names = sorted(set(m.name for m in monsters))
        assert names == ["Giant Scorpion"]
        assert len(monsters) == 1

    def test_multiple_declarations(self):
        room = _StubRoom(
            encounter_text=(
                "1 Wight (MM p. 300) + 2 Ghouls (MM p. 148). All hostile."
            ),
        )
        monsters = editor_server._build_monsters_from_room(room)
        names = sorted(m.name for m in monsters)
        assert names == ["Ghoul", "Ghoul", "Wight"]

    def test_lizardfolk_plural(self):
        room = _StubRoom(
            encounter_text="4 Lizardfolk (MM p.204). Territorial."
        )
        monsters = editor_server._build_monsters_from_room(room)
        assert len(monsters) == 4
        assert all(m.name == "Lizardfolk" for m in monsters)

    def test_dice_count_is_deterministic(self):
        room = _StubRoom(
            encounter_text="1d4 Goblins (MM p.166) jeer from the rafters."
        )
        a = editor_server._build_monsters_from_room(room)
        b = editor_server._build_monsters_from_room(room)
        assert len(a) == len(b)
        assert 1 <= len(a) <= 4
        assert all(m.name == "Goblin" for m in a)

    def test_no_declaration_returns_empty(self):
        # Prose alone — no MM page reference — produces no monsters.
        # The DM's signal is the empty stat-block area in the editor.
        room = _StubRoom(
            encounter_text="A lone Ghoul stands silently in the corner."
        )
        assert editor_server._build_monsters_from_room(room) == []

    def test_empty_text_returns_empty(self):
        assert editor_server._build_monsters_from_room(_StubRoom("")) == []

    def test_prose_words_do_not_inflate(self):
        # The combined nightmare: every prose word that has ever
        # fuzzy- or scan-matched a creature is in this text, but only
        # the formal declaration has an MM page reference.
        room = _StubRoom(
            encounter_text=(
                "3 Skeletons (MM p.272). Fight to the death. The room "
                "is dim with no light source, and is unusual in its "
                "width. A shadow falls across the floor; the count of "
                "alcoves is seven."
            ),
        )
        monsters = editor_server._build_monsters_from_room(room)
        names = sorted(set(m.name for m in monsters))
        assert names == ["Skeleton"]
        assert len(monsters) == 3
