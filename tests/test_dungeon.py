"""Phase 1 test gate — schema and graph validation for dungeon JSON.

Updated for the multi-level JSON structure: dungeon-level metadata
stays at the top, with rooms/corridors/WM rules nested inside `levels[]`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import dungeon
from dungeon import DungeonValidationError


EXAMPLE_PATH = Path(__file__).parent.parent / "data" / "example_dungeon.json"


@pytest.fixture
def base() -> dict:
    """A minimal valid multi-level dungeon dict that tests mutate."""
    return {
        "dungeon_name": "Test Dungeon",
        "party_level": 3,
        "current_level": 1,
        "party": {
            "size": 1,
            "characters": [{"name": "Solo", "darkvision": False, "exhaustion": 0}],
        },
        "levels": [
            {
                "level_number": 1,
                "display_name": "Level 1 — Test",
                "map_image": "target_dungeon_maps/test1.png",
                "map_image_scale": 1.0,
                "wm_check_method": "d20",
                "wm_check_threshold": 18,
                "wm_check_frequency": "every_turn",
                "wandering_monster_table": [
                    {"roll": 1, "encounter": "Goblin"},
                    {"roll": 2, "encounter": "Skeleton"},
                ],
                "rooms": [
                    {"id": "A", "name": "Alpha", "state": "unexplored", "tags": ["empty"]},
                    {"id": "B", "name": "Beta",  "state": "unexplored", "tags": ["encounter"]},
                ],
                "corridors": [
                    {"from": "A", "to": "B", "distance_ft": 20, "tags": []},
                ],
            },
        ],
    }


def _load_dict(d: dict) -> dungeon.Dungeon:
    return dungeon._from_dict(d, source="<test>")


def _level0(d: dict) -> dict:
    """The first level dict — most field-level mutations happen inside it."""
    return d["levels"][0]


# --- Happy paths -------------------------------------------------------------


class TestExampleDungeon:
    def test_example_loads(self) -> None:
        d = dungeon.load(EXAMPLE_PATH)
        assert d.name == "The Tomb of the Iron Lich"
        assert d.party_level == 3
        assert len(d.levels) == 2
        assert d.current_level == 1

    def test_level1_rooms_and_corridors(self) -> None:
        d = dungeon.load(EXAMPLE_PATH)
        l1 = d.get_level(1)
        assert len(l1.rooms) == 7
        assert len(l1.corridors) == 7

    def test_example_room_ids_unique_per_level(self) -> None:
        d = dungeon.load(EXAMPLE_PATH)
        for level in d.levels:
            ids = [r.id for r in level.rooms]
            assert len(set(ids)) == len(ids)

    def test_example_rooms_by_id_lookup(self) -> None:
        d = dungeon.load(EXAMPLE_PATH)
        assert d.get_level(1).rooms_by_id["R07"].name == "Lich's Sanctum"

    def test_neighbors_undirected_for_normal_edges(self) -> None:
        l1 = dungeon.load(EXAMPLE_PATH).get_level(1)
        assert "R02" in l1.neighbors("R01")
        assert "R01" in l1.neighbors("R02")

    def test_neighbors_one_way_is_directional(self) -> None:
        l1 = dungeon.load(EXAMPLE_PATH).get_level(1)
        # R07 → R01 is one-way: R01 ∈ R07's neighbors but not vice-versa via that edge.
        assert "R01" in l1.neighbors("R07")
        assert "R07" not in l1.neighbors("R01")

    def test_stairs_tags_present(self) -> None:
        d = dungeon.load(EXAMPLE_PATH)
        assert d.get_level(1).stairs_down_room_id() == "R04"
        assert d.get_level(2).stairs_up_room_id() == "R10"

    def test_current_level_property(self) -> None:
        d = dungeon.load(EXAMPLE_PATH)
        assert d.current.level_number == 1
        assert "Entry Vaults" in d.current.display_name


class TestMinimalValid:
    def test_minimal_loads(self, base: dict) -> None:
        d = _load_dict(base)
        assert d.name == "Test Dungeon"
        assert len(d.levels) == 1
        assert len(d.get_level(1).rooms) == 2

    def test_single_room_level_is_valid(self, base: dict) -> None:
        _level0(base)["rooms"] = [
            {"id": "A", "name": "Only", "state": "unexplored", "tags": ["empty"]}
        ]
        _level0(base)["corridors"] = []
        d = _load_dict(base)
        assert len(d.get_level(1).rooms) == 1


# --- Top-level required fields -----------------------------------------------


class TestRequiredFields:
    @pytest.mark.parametrize("field", [
        "dungeon_name", "party_level", "current_level", "party", "levels",
    ])
    def test_missing_top_level_field_raises(self, base: dict, field: str) -> None:
        del base[field]
        with pytest.raises(DungeonValidationError, match=field):
            _load_dict(base)

    @pytest.mark.parametrize("field", [
        "level_number", "display_name", "map_image", "map_image_scale",
        "wm_check_method", "wm_check_threshold", "wm_check_frequency",
        "wandering_monster_table", "rooms", "corridors",
    ])
    def test_missing_per_level_field_raises(self, base: dict, field: str) -> None:
        del _level0(base)[field]
        with pytest.raises(DungeonValidationError, match=field):
            _load_dict(base)

    def test_empty_levels_array_raises(self, base: dict) -> None:
        base["levels"] = []
        with pytest.raises(DungeonValidationError, match="levels"):
            _load_dict(base)

    def test_current_level_not_in_levels_raises(self, base: dict) -> None:
        base["current_level"] = 99
        with pytest.raises(DungeonValidationError, match="current_level"):
            _load_dict(base)

    def test_duplicate_level_number_raises(self, base: dict) -> None:
        base["levels"].append(dict(_level0(base), level_number=1))
        with pytest.raises(DungeonValidationError, match="duplicate level_number"):
            _load_dict(base)


# --- Schema-level validation -------------------------------------------------


class TestEnumValidation:
    def test_invalid_wm_method_raises(self, base: dict) -> None:
        _level0(base)["wm_check_method"] = "d100"
        with pytest.raises(DungeonValidationError, match="wm_check_method"):
            _load_dict(base)

    def test_invalid_room_state_raises(self, base: dict) -> None:
        _level0(base)["rooms"][0]["state"] = "smoldering"
        with pytest.raises(DungeonValidationError, match="state"):
            _load_dict(base)

    def test_invalid_room_tag_raises(self, base: dict) -> None:
        _level0(base)["rooms"][0]["tags"] = ["mysterious"]
        with pytest.raises(DungeonValidationError, match="tag"):
            _load_dict(base)

    def test_invalid_corridor_tag_raises(self, base: dict) -> None:
        _level0(base)["corridors"][0]["tags"] = ["weird"]
        with pytest.raises(DungeonValidationError, match="tag"):
            _load_dict(base)

    def test_empty_room_tags_raises(self, base: dict) -> None:
        _level0(base)["rooms"][0]["tags"] = []
        with pytest.raises(DungeonValidationError, match="tags"):
            _load_dict(base)

    def test_stairs_tags_accepted(self, base: dict) -> None:
        _level0(base)["rooms"][0]["tags"] = ["empty", "stairs_down"]
        d = _load_dict(base)
        assert "stairs_down" in d.get_level(1).rooms[0].tags


class TestRoomValidation:
    def test_duplicate_room_id_raises(self, base: dict) -> None:
        _level0(base)["rooms"][1]["id"] = "A"
        with pytest.raises(DungeonValidationError, match="duplicate room id"):
            _load_dict(base)

    def test_room_missing_field_raises(self, base: dict) -> None:
        del _level0(base)["rooms"][0]["name"]
        with pytest.raises(DungeonValidationError, match="name"):
            _load_dict(base)


class TestCorridorValidation:
    def test_corridor_unknown_from_raises(self, base: dict) -> None:
        _level0(base)["corridors"][0]["from"] = "ZZZ"
        with pytest.raises(DungeonValidationError, match="unknown room id"):
            _load_dict(base)

    def test_corridor_unknown_to_raises(self, base: dict) -> None:
        _level0(base)["corridors"][0]["to"] = "ZZZ"
        with pytest.raises(DungeonValidationError, match="unknown room id"):
            _load_dict(base)

    def test_corridor_self_loop_raises(self, base: dict) -> None:
        c = _level0(base)["corridors"][0]
        c["to"] = c["from"]
        with pytest.raises(DungeonValidationError, match="self-loops"):
            _load_dict(base)

    def test_corridor_zero_distance_raises(self, base: dict) -> None:
        _level0(base)["corridors"][0]["distance_ft"] = 0
        with pytest.raises(DungeonValidationError, match="distance_ft"):
            _load_dict(base)


# --- Graph-level validation --------------------------------------------------


class TestGraphConnectivity:
    """With the PNG-driven paradigm, connectivity is implicit in the image
    rather than the corridor list. Orphan rooms are *allowed* — the runtime
    can still reveal them by clicking the corresponding region. We only
    check that corridor endpoints exist."""

    def test_orphan_room_is_allowed(self, base: dict) -> None:
        _level0(base)["rooms"].append(
            {"id": "C", "name": "Lonely", "state": "unexplored", "tags": ["empty"]}
        )
        d = _load_dict(base)
        assert "C" in d.get_level(1).rooms_by_id

    def test_corridor_to_unknown_room_raises(self, base: dict) -> None:
        _level0(base)["corridors"].append(
            {"from": "A", "to": "ZZZ", "distance_ft": 10, "tags": []}
        )
        with pytest.raises(DungeonValidationError, match="unknown room id"):
            _load_dict(base)

    def test_one_way_edge_loads(self, base: dict) -> None:
        _level0(base)["corridors"][0]["tags"] = ["one-way"]
        d = _load_dict(base)
        assert len(d.get_level(1).rooms) == 2


class TestPartyValidation:
    def test_party_size_mismatch_raises(self, base: dict) -> None:
        base["party"]["size"] = 5
        with pytest.raises(DungeonValidationError, match="party.size"):
            _load_dict(base)

    def test_duplicate_character_name_raises(self, base: dict) -> None:
        base["party"]["size"] = 2
        base["party"]["characters"].append({"name": "Solo"})
        with pytest.raises(DungeonValidationError, match="duplicate character name"):
            _load_dict(base)

    def test_exhaustion_out_of_range_raises(self, base: dict) -> None:
        base["party"]["characters"][0]["exhaustion"] = 7
        with pytest.raises(DungeonValidationError, match="exhaustion"):
            _load_dict(base)


# --- Loader-level errors -----------------------------------------------------


class TestLevelChallengeRating:
    """challenge_rating is a free-text DM-authored hint at the level scope.
    Defaults to '' for legacy JSON; must round-trip through dump/load."""

    def test_default_empty(self, base: dict) -> None:
        d = _load_dict(base)
        assert d.get_level(1).challenge_rating == ""

    def test_round_trip(self, base: dict, tmp_path: Path) -> None:
        base["levels"][0]["challenge_rating"] = "CR 1/4–1 (standard)"
        d = _load_dict(base)
        out = tmp_path / "rt.json"
        dungeon.dump(d, out)
        d2 = dungeon.load(out)
        assert d2.get_level(1).challenge_rating == "CR 1/4–1 (standard)"


class TestNarrativeFields:
    """box_text / encounter_text / treasure_text / special_text — the
    per-room narrative fields edited via the browser editor. Must round-
    trip through dump/load and default to '' when absent (legacy JSON)."""

    def test_defaults_are_empty(self, base: dict) -> None:
        d = _load_dict(base)
        room = d.get_level(1).rooms[0]
        assert room.box_text == ""
        assert room.encounter_text == ""
        assert room.treasure_text == ""
        assert room.special_text == ""

    def test_round_trip(self, base: dict, tmp_path: Path) -> None:
        _level0(base)["rooms"][0]["box_text"] = "Read me aloud."
        _level0(base)["rooms"][0]["encounter_text"] = "1 ghoul"
        _level0(base)["rooms"][0]["treasure_text"] = "200 gp"
        _level0(base)["rooms"][0]["special_text"] = "Pulsing rune on floor"
        d = _load_dict(base)
        out = tmp_path / "rt.json"
        dungeon.dump(d, out)
        d2 = dungeon.load(out)
        room = d2.get_level(1).rooms[0]
        assert room.box_text == "Read me aloud."
        assert room.encounter_text == "1 ghoul"
        assert room.treasure_text == "200 gp"
        assert room.special_text == "Pulsing rune on floor"


class TestLoaderErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            dungeon.load(tmp_path / "nope.json")

    def test_malformed_json_raises_validation_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(DungeonValidationError, match="invalid JSON"):
            dungeon.load(bad)

    def test_top_level_not_object_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.json"
        bad.write_text("[]")
        with pytest.raises(DungeonValidationError, match="top level"):
            dungeon.load(bad)
