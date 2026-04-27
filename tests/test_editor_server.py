"""Tests for editor_server — GET renders the form, POST mutates JSON."""

from __future__ import annotations

import json
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlencode

import pytest

import dungeon as dungeon_mod
import editor_server


SEED_DUNGEON = {
    "dungeon_name": "Smoke Dungeon",
    "party_level": 1,
    "current_level": 1,
    "party": {"size": 1, "characters": [{"name": "Solo"}]},
    "levels": [
        {
            "level_number": 1,
            "display_name": "Level 1",
            "map_image": "target_dungeon_maps/none.png",
            "map_image_scale": 1.0,
            "wm_check_method": "d20",
            "wm_check_threshold": 18,
            "wm_check_frequency": "every_turn",
            "wandering_monster_table": [
                {"roll": 1, "encounter": "Goblin"},
            ],
            "rooms": [
                {
                    "id": "R01", "name": "R01", "state": "unexplored",
                    "tags": ["empty"],
                    "image_region": {"kind": "rect",
                                     "x": 0, "y": 0,
                                     "width": 100, "height": 100},
                },
                {
                    "id": "R02", "name": "R02", "state": "unexplored",
                    "tags": ["empty"],
                    "image_region": {"kind": "rect",
                                     "x": 200, "y": 200,
                                     "width": 100, "height": 100},
                },
            ],
            "corridors": [],
        },
    ],
}


@pytest.fixture
def dungeon_json(tmp_path: Path) -> Path:
    p = tmp_path / "smoke.json"
    p.write_text(json.dumps(SEED_DUNGEON, indent=2))
    return p


@pytest.fixture
def server(dungeon_json: Path):
    """Run the editor server on an ephemeral port for the duration of one
    test. Daemon thread; we shut the server down after each test."""
    srv, _thread = editor_server.start_editor_server(dungeon_json, port=0)
    host, port = srv.server_address[:2]
    try:
        yield srv, host, port
    finally:
        srv.shutdown()


def _http_get(host: str, port: int, path: str = "/"):
    conn = HTTPConnection(host, port, timeout=2)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp, body


def _http_post(host: str, port: int, path: str, form: dict[str, list[str] | str],
               extra_headers: dict[str, str] | None = None):
    """form values may be str or list[str]; we encode multi-valued ones via
    urllib.parse.urlencode(..., doseq=True)."""
    body = urlencode(form, doseq=True)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Content-Length": str(len(body)),
    }
    if extra_headers:
        headers.update(extra_headers)
    conn = HTTPConnection(host, port, timeout=2)
    conn.request("POST", path, body=body, headers=headers)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp, raw


# --- GET ---------------------------------------------------------------------


class TestGet:
    def test_healthz(self, server):
        _, host, port = server
        resp, body = _http_get(host, port, "/healthz")
        assert resp.status == 200
        assert body == b"ok\n"

    def test_root_returns_form_with_room_ids(self, server):
        _, host, port = server
        resp, body = _http_get(host, port, "/")
        assert resp.status == 200
        text = body.decode("utf-8")
        assert "R01" in text
        assert "R02" in text
        assert 'name="box_text"' in text
        assert 'name="notes"' in text

    def test_root_when_dungeon_missing_404(self, tmp_path):
        p = tmp_path / "nope.json"
        srv, _ = editor_server.start_editor_server(p, port=0)
        host, port = srv.server_address[:2]
        try:
            resp, _ = _http_get(host, port, "/")
            assert resp.status == 404
        finally:
            srv.shutdown()


# --- POST --------------------------------------------------------------------


class TestPost:
    def test_save_writes_narrative_fields(self, server, dungeon_json):
        _, host, port = server
        resp, _ = _http_post(host, port, "/room", {
            "level_number": "1",
            "room_id": "R01",
            "name": "Entrance",
            "box_text": "A cold draught greets you.",
            "notes": "",
            "encounter_text": "",
            "treasure_text": "",
            "special_text": "Glowing rune in the floor.",
            "encounter_ref": "",
            "treasure_tier": "",
            "tags": ["empty"],
        })
        assert resp.status == 303  # Post/Redirect/Get
        assert resp.getheader("Location") == "/?saved=R01"

        d = dungeon_mod.load(dungeon_json)
        room = d.get_level(1).rooms_by_id["R01"]
        assert room.name == "Entrance"
        assert room.box_text == "A cold draught greets you."
        assert room.special_text == "Glowing rune in the floor."

    def test_save_preserves_image_region(self, server, dungeon_json):
        _, host, port = server
        before = dungeon_mod.load(dungeon_json).get_level(1).rooms_by_id["R01"]
        before_region = before.image_region

        _http_post(host, port, "/room", {
            "level_number": "1",
            "room_id": "R01",
            "name": "Updated",
            "box_text": "x", "notes": "",
            "encounter_text": "", "treasure_text": "",
            "encounter_ref": "", "treasure_tier": "",
            "tags": ["empty"],
        })

        after = dungeon_mod.load(dungeon_json).get_level(1).rooms_by_id["R01"]
        assert after.image_region.kind == before_region.kind
        assert after.image_region.rect == before_region.rect

    def test_save_updates_tags(self, server, dungeon_json):
        _, host, port = server
        _http_post(host, port, "/room", {
            "level_number": "1",
            "room_id": "R02",
            "name": "R02",
            "box_text": "", "notes": "",
            "encounter_text": "", "treasure_text": "",
            "encounter_ref": "", "treasure_tier": "",
            "tags": ["encounter", "treasure"],
        })
        room = dungeon_mod.load(dungeon_json).get_level(1).rooms_by_id["R02"]
        assert set(room.tags) == {"encounter", "treasure"}

    def test_save_unknown_room_400(self, server):
        _, host, port = server
        resp, _ = _http_post(host, port, "/room", {
            "level_number": "1", "room_id": "ZZZ",
            "name": "x", "box_text": "", "notes": "",
            "encounter_text": "", "treasure_text": "",
            "encounter_ref": "", "treasure_tier": "",
            "tags": ["empty"],
        })
        assert resp.status == 400

    def test_save_drops_unknown_tags(self, server, dungeon_json):
        """Server must filter incoming tags to the known set so a malformed
        request can't introduce arbitrary strings into the JSON."""
        _, host, port = server
        _http_post(host, port, "/room", {
            "level_number": "1", "room_id": "R01",
            "name": "R01", "box_text": "", "notes": "",
            "encounter_text": "", "treasure_text": "",
            "encounter_ref": "", "treasure_tier": "",
            "tags": ["encounter", "evil-tag-injection"],
        })
        room = dungeon_mod.load(dungeon_json).get_level(1).rooms_by_id["R01"]
        assert "encounter" in room.tags
        assert "evil-tag-injection" not in room.tags

    def test_save_empty_tags_falls_back_to_empty(self, server, dungeon_json):
        """A room must always have at least one tag; if the user un-checks
        every tag, default back to ['empty'] so JSON validation passes."""
        _, host, port = server
        _http_post(host, port, "/room", {
            "level_number": "1", "room_id": "R01",
            "name": "R01", "box_text": "", "notes": "",
            "encounter_text": "", "treasure_text": "",
            "encounter_ref": "", "treasure_tier": "",
        })
        room = dungeon_mod.load(dungeon_json).get_level(1).rooms_by_id["R01"]
        assert room.tags == ("empty",)


class TestLevelEndpoint:
    """POST /level mutates the LEVEL fields (display name, challenge
    rating, WM rules + table). Rooms on that level are untouched."""

    def test_get_includes_level_card(self, server):
        _, host, port = server
        _, body = _http_get(host, port, "/")
        text = body.decode("utf-8")
        assert "level-card" in text
        assert 'name="challenge_rating"' in text
        assert 'name="wm_check_threshold"' in text
        assert 'name="wm_check_every_n_turns"' in text
        assert 'name="wm_roll"' in text
        assert 'name="wm_encounter"' in text

    def test_save_level_basic_fields(self, server, dungeon_json):
        _, host, port = server
        resp, _ = _http_post(host, port, "/level", {
            "level_number": "1",
            "display_name": "The Entry Vaults",
            "challenge_rating": "CR 1/4–1 (standard)",
            "wm_check_method": "d20",
            "wm_check_threshold": "17",
            "wm_check_every_n_turns": "3",
            "wm_roll": ["1", "2"],
            "wm_encounter": ["Skeleton", "Giant Rat"],
        })
        assert resp.status == 303
        assert resp.getheader("Location") == "/?saved=L1"
        d = dungeon_mod.load(dungeon_json)
        lv = d.get_level(1)
        assert lv.display_name == "The Entry Vaults"
        assert lv.challenge_rating == "CR 1/4–1 (standard)"
        assert lv.wm_check_threshold == 17
        assert lv.wm_check_every_n_turns == 3

    def test_save_level_replaces_wm_table(self, server, dungeon_json):
        _, host, port = server
        _http_post(host, port, "/level", {
            "level_number": "1",
            "display_name": "Level 1",
            "challenge_rating": "",
            "wm_check_method": "d20",
            "wm_check_threshold": "18",
            "wm_check_every_n_turns": "1",
            "wm_roll": ["1", "3", "5"],
            "wm_encounter": ["Wight", "Mummy", "Ghast"],
        })
        d = dungeon_mod.load(dungeon_json)
        rolls = [(e.roll, e.encounter)
                 for e in d.get_level(1).wandering_monster_table]
        assert rolls == [(1, "Wight"), (3, "Mummy"), (5, "Ghast")]

    def test_save_level_drops_blank_rows(self, server, dungeon_json):
        _, host, port = server
        _http_post(host, port, "/level", {
            "level_number": "1",
            "display_name": "Level 1",
            "challenge_rating": "",
            "wm_check_method": "d20",
            "wm_check_threshold": "18",
            "wm_check_every_n_turns": "1",
            "wm_roll":      ["1", "2", "",   "3"],
            "wm_encounter": ["X", "",  "Y",  "Z"],  # row 1 blank enc, row 2 blank roll
        })
        d = dungeon_mod.load(dungeon_json)
        rolls = [(e.roll, e.encounter)
                 for e in d.get_level(1).wandering_monster_table]
        assert rolls == [(1, "X"), (3, "Z")]

    def test_save_level_keeps_existing_table_when_all_rows_invalid(
        self, server, dungeon_json,
    ):
        _, host, port = server
        before = dungeon_mod.load(dungeon_json).get_level(1).wandering_monster_table
        _http_post(host, port, "/level", {
            "level_number": "1",
            "display_name": "Level 1",
            "challenge_rating": "",
            "wm_check_method": "d20",
            "wm_check_threshold": "18",
            "wm_check_every_n_turns": "1",
            "wm_roll": ["", ""],
            "wm_encounter": ["", ""],
        })
        after = dungeon_mod.load(dungeon_json).get_level(1).wandering_monster_table
        # Empty submission preserves the existing table (loader requires
        # at least one row).
        assert after == before

    def test_save_level_preserves_rooms(self, server, dungeon_json):
        _, host, port = server
        _http_post(host, port, "/level", {
            "level_number": "1",
            "display_name": "Renamed",
            "challenge_rating": "",
            "wm_check_method": "d20",
            "wm_check_threshold": "18",
            "wm_check_every_n_turns": "1",
            "wm_roll": ["1"],
            "wm_encounter": ["Goblin"],
        })
        d = dungeon_mod.load(dungeon_json)
        # The fixture has R01 and R02 with rect regions; both must survive.
        ids = {r.id for r in d.get_level(1).rooms}
        assert ids == {"R01", "R02"}


class TestFragmentSave:
    """When the request carries `X-Editor-Fragment: 1`, save endpoints
    return 200 OK with the freshly-rendered card HTML so the JS client
    can splice it in without a full page reload (which would erase
    typed-but-unsaved edits in sibling forms — the original bug)."""

    def test_room_fragment_response_is_200_with_just_that_card(
        self, server, dungeon_json
    ):
        _, host, port = server
        resp, body = _http_post(host, port, "/room", {
            "level_number": "1", "room_id": "R01",
            "name": "Saved Name",
            "box_text": "Box text here.",
            "notes": "", "encounter_text": "", "treasure_text": "",
            "special_text": "", "encounter_ref": "", "treasure_tier": "",
            "tags": ["empty"],
        }, extra_headers={"X-Editor-Fragment": "1"})
        assert resp.status == 200
        ct = resp.getheader("Content-Type") or ""
        assert ct.startswith("text/html")
        text = body.decode("utf-8")
        # Card markup includes the room id in its container id.
        assert 'id="room-R01"' in text
        # Other rooms do NOT bleed into the fragment response.
        assert 'id="room-R02"' not in text
        # The post still wrote to disk.
        room = dungeon_mod.load(dungeon_json).get_level(1).rooms_by_id["R01"]
        assert room.name == "Saved Name"
        assert room.box_text == "Box text here."

    def test_level_fragment_response_is_200_with_level_card(
        self, server, dungeon_json
    ):
        _, host, port = server
        resp, body = _http_post(host, port, "/level", {
            "level_number": "1",
            "display_name": "Renamed via Fragment",
            "challenge_rating": "",
            "wm_check_method": "d20",
            "wm_check_threshold": "18",
            "wm_check_every_n_turns": "1",
            "wm_roll": ["1"],
            "wm_encounter": ["Goblin"],
        }, extra_headers={"X-Editor-Fragment": "1"})
        assert resp.status == 200
        text = body.decode("utf-8")
        assert 'id="level-1"' in text
        # Fragment includes the display name we just saved.
        assert "Renamed via Fragment" in text
        # No `<html>` shell in a fragment.
        assert "<!doctype html>" not in text.lower()

    def test_room_post_without_header_still_303s(self, server):
        """The header-less path stays the legacy 303 redirect — needed
        for non-JS form posts and back-compat with existing tooling."""
        _, host, port = server
        resp, _ = _http_post(host, port, "/room", {
            "level_number": "1", "room_id": "R01",
            "name": "x", "box_text": "", "notes": "",
            "encounter_text": "", "treasure_text": "",
            "encounter_ref": "", "treasure_tier": "",
            "tags": ["empty"],
        })
        assert resp.status == 303
