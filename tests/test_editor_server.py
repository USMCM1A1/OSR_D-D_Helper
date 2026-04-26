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


def _http_post(host: str, port: int, path: str, form: dict[str, list[str] | str]):
    """form values may be str or list[str]; we encode multi-valued ones via
    urllib.parse.urlencode(..., doseq=True)."""
    body = urlencode(form, doseq=True)
    conn = HTTPConnection(host, port, timeout=2)
    conn.request("POST", path, body=body, headers={
        "Content-Type": "application/x-www-form-urlencoded",
        "Content-Length": str(len(body)),
    })
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
    def test_save_writes_box_text(self, server, dungeon_json):
        _, host, port = server
        resp, _ = _http_post(host, port, "/room", {
            "level_number": "1",
            "room_id": "R01",
            "name": "Entrance",
            "box_text": "A cold draught greets you.",
            "notes": "",
            "encounter_text": "",
            "treasure_text": "",
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
