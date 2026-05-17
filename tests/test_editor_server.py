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

    def test_root_returns_spa_shell(self, server):
        # GET / is now the SPA shell with tab strip + iframes.
        # The room-editor content lives at /editor (an iframe target).
        _, host, port = server
        resp, body = _http_get(host, port, "/")
        assert resp.status == 200
        text = body.decode("utf-8")
        assert 'data-tab="editor"' in text
        assert 'data-tab="assistant"' in text
        assert 'src="/editor"' in text

    def test_editor_returns_form_with_room_ids(self, server):
        _, host, port = server
        resp, body = _http_get(host, port, "/editor")
        assert resp.status == 200
        text = body.decode("utf-8")
        assert "R01" in text
        assert "R02" in text
        assert 'name="box_text"' in text
        assert 'name="notes"' in text

    def test_player_route_renders_html(self, server):
        # /player is the screen-shareable Player View. It's a tiny HTML
        # wrapper that auto-refreshes /player.png on a 2 s interval.
        _, host, port = server
        resp, body = _http_get(host, port, "/player")
        assert resp.status == 200
        text = body.decode("utf-8")
        assert "/player.png" in text
        assert "Player View" in text

    def test_player_png_404_when_missing(self, server, monkeypatch, tmp_path):
        # When pygame hasn't written the PNG yet, /player.png returns
        # 404 so the client-side refresh polls without crashing.
        monkeypatch.setattr(editor_server, "_PLAYER_PNG_PATH",
                            tmp_path / "nonexistent.png")
        _, host, port = server
        resp, _ = _http_get(host, port, "/player.png")
        assert resp.status == 404

    def test_player_png_serves_bytes_when_present(self, server, monkeypatch, tmp_path):
        # When the PNG exists, /player.png returns its bytes with the
        # right content-type. Use a tiny placeholder — the route doesn't
        # inspect the contents.
        fake = tmp_path / "player_map.png"
        fake.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake-bytes-for-test")
        monkeypatch.setattr(editor_server, "_PLAYER_PNG_PATH", fake)
        _, host, port = server
        resp, body = _http_get(host, port, "/player.png")
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "image/png"
        assert body == fake.read_bytes()

    def test_editor_when_dungeon_missing_404(self, tmp_path):
        p = tmp_path / "nope.json"
        srv, _ = editor_server.start_editor_server(p, port=0)
        host, port = srv.server_address[:2]
        try:
            resp, _ = _http_get(host, port, "/editor")
            assert resp.status == 404
        finally:
            srv.shutdown()

    def test_room_fragment_endpoint_returns_one_card(self, server):
        """GET /room?level_number=N&room_id=R01 returns just that
        room's card — used by the editor's BroadcastChannel listener
        to swap a single card in place when /assistant applies a room."""
        _, host, port = server
        resp, body = _http_get(
            host, port, "/room?level_number=1&room_id=R01")
        assert resp.status == 200
        text = body.decode("utf-8")
        assert 'id="room-R01"' in text
        assert 'name="level_number" value="1"' in text
        # Only the card, no full-page chrome.
        assert "<html" not in text.lower()

    def test_room_fragment_endpoint_unknown_room_404(self, server):
        _, host, port = server
        resp, _ = _http_get(
            host, port, "/room?level_number=1&room_id=GHOST")
        assert resp.status == 404

    def test_room_fragment_endpoint_missing_args_400(self, server):
        _, host, port = server
        resp, _ = _http_get(host, port, "/room?level_number=1")
        assert resp.status == 400


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
        assert resp.getheader("Location") == "/editor?saved=R01"

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
        _, body = _http_get(host, port, "/editor")
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
        assert resp.getheader("Location") == "/editor?saved=L1"
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


# --- Assistant page + apply --------------------------------------------------


def _http_post_json(host, port, path, payload):
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    conn = HTTPConnection(host, port, timeout=2)
    conn.request("POST", path, body=body, headers=headers)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp, raw


class TestAssistantPage:
    def test_get_renders_setup_card_when_cli_missing(self, server, monkeypatch):
        # Force the CLI-missing branch by making which() return None.
        import shutil
        monkeypatch.setattr(shutil, "which", lambda _: None)
        _, host, port = server
        resp, body = _http_get(host, port, "/assistant")
        assert resp.status == 200
        assert b"Setup needed" in body
        assert b"claude.com/download" in body
        # No chat form is rendered when the CLI isn't available.
        assert b'id="theme"' not in body

    def test_get_renders_chat_form_when_cli_present(self, server, monkeypatch):
        # Pretend the CLI is on PATH.
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/" + name)
        _, host, port = server
        resp, body = _http_get(host, port, "/assistant")
        assert resp.status == 200
        assert b'id="theme"' in body
        assert b'id="start-btn"' in body

    def test_apply_without_session_returns_400(self, server, monkeypatch):
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/" + name)
        _, host, port = server
        # Drop any leftover sessions from prior tests.
        editor_server._assistant_sessions.clear()
        resp, raw = _http_post_json(host, port, "/assistant/apply",
                                    {"room_id": "R01"})
        assert resp.status == 400
        payload = json.loads(raw)
        assert payload["error"] == "no_proposal"

    def test_apply_writes_room_and_returns_card(self, server, dungeon_json):
        # Inject a session with a pre-populated proposal so we can test
        # the apply path without invoking the real CLI.
        import dungeon_assistant as da
        import dungeon as dungeon_mod
        d = dungeon_mod.load(dungeon_json)
        sess = da.AssistantSession(
            dungeon_path=dungeon_json,
            dungeon=d,
            theme="test",
            level_number=1,
            party_level=3,
            runner=lambda args: {  # never invoked on the apply path
                "is_error": False, "result": "", "structured_output": {},
                "session_id": "sesn-x",
            },
        )
        sess._proposals["R01"] = da.RoomProposal(
            id="R01", name="The Threshold",
            tags=("encounter",),
            reaction_required=True,
            box_text="A vaulted hall.",
            encounter_text="2 Skeletons (MM p.272). Hostile.",
            notes="They charge on first sight.",
        )
        editor_server._assistant_sessions[dungeon_json.resolve()] = sess

        _, host, port = server
        resp, body = _http_post_json(host, port, "/assistant/apply",
                                     {"room_id": "R01"})
        assert resp.status == 200, body
        # Returns a room-card HTML fragment.
        assert b'id="room-R01"' in body
        assert b"The Threshold" in body

        # The dungeon JSON on disk now reflects the proposal.
        d2 = dungeon_mod.load(dungeon_json)
        room = d2.levels[0].rooms_by_id["R01"]
        assert room.name == "The Threshold"
        assert room.box_text == "A vaulted hall."
        assert "encounter" in room.tags

        # A backup file exists in the same folder.
        baks = list(dungeon_json.parent.glob(dungeon_json.name + ".*.bak"))
        assert len(baks) >= 1

    def test_reset_clears_session(self, server, dungeon_json):
        editor_server._assistant_sessions[dungeon_json.resolve()] = "sentinel"
        _, host, port = server
        resp, _ = _http_post_json(host, port, "/assistant/reset", {})
        assert resp.status == 204
        assert dungeon_json.resolve() not in editor_server._assistant_sessions


class TestAssistantMultiDungeon:
    """Multi-dungeon picker: ?dungeon=<folder> on GET, dungeon_folder
    in POST bodies. The handler resolves a folder name under
    dungeons_dir against the bound default and rejects anything that
    doesn't live under that root."""

    @pytest.fixture
    def multi_dungeon_root(self, tmp_path):
        """Build two dungeon folders under one root."""
        root = tmp_path / "dungeons"
        root.mkdir()
        for name, party in [("alpha", 3), ("beta", 5)]:
            folder = root / name
            folder.mkdir()
            payload = json.loads(json.dumps(SEED_DUNGEON))
            payload["dungeon_name"] = f"The {name.title()} Dungeon"
            payload["party_level"] = party
            (folder / "dungeon.json").write_text(
                json.dumps(payload, indent=2)
            )
        return root

    @pytest.fixture
    def multi_server(self, multi_dungeon_root):
        bound = multi_dungeon_root / "alpha" / "dungeon.json"
        srv, _ = editor_server.start_editor_server(
            bound, port=0, dungeons_dir=multi_dungeon_root,
        )
        host, port = srv.server_address[:2]
        try:
            yield srv, host, port, multi_dungeon_root
        finally:
            srv.shutdown()

    def test_picker_lists_all_dungeons_in_root(self, multi_server, monkeypatch):
        import shutil
        monkeypatch.setattr(shutil, "which", lambda n: "/usr/local/bin/" + n)
        _, host, port, _ = multi_server
        resp, body = _http_get(host, port, "/assistant")
        assert resp.status == 200
        # Both dungeons should appear in the picker dropdown.
        assert b"Alpha Dungeon" in body
        assert b"Beta Dungeon" in body
        # Folder names round-trip in the option values.
        assert b'value="alpha"' in body
        assert b'value="beta"' in body

    def test_get_with_dungeon_query_renders_that_dungeon(
        self, multi_server, monkeypatch
    ):
        import shutil
        monkeypatch.setattr(shutil, "which", lambda n: "/usr/local/bin/" + n)
        _, host, port, _ = multi_server
        resp, body = _http_get(host, port, "/assistant?dungeon=beta")
        assert resp.status == 200
        # H1 reflects the picked dungeon.
        assert b"Beta Dungeon" in body
        # The hidden current-dungeon-folder input is "beta".
        assert b'id="current-dungeon-folder" value="beta"' in body
        # Party level default mirrors the picked dungeon.
        assert b'value="5"' in body

    def test_get_with_unknown_dungeon_404s(self, multi_server, monkeypatch):
        import shutil
        monkeypatch.setattr(shutil, "which", lambda n: "/usr/local/bin/" + n)
        _, host, port, _ = multi_server
        resp, _ = _http_get(host, port, "/assistant?dungeon=ghost")
        assert resp.status == 404

    def test_get_rejects_path_traversal(self, multi_server, monkeypatch):
        import shutil
        monkeypatch.setattr(shutil, "which", lambda n: "/usr/local/bin/" + n)
        _, host, port, _ = multi_server
        # `..` would otherwise resolve outside dungeons_dir.
        resp, _ = _http_get(host, port, "/assistant?dungeon=../etc")
        assert resp.status == 404

    def test_assistant_page_marks_levels_as_ready_or_not(
        self, multi_server, monkeypatch
    ):
        """Level dropdown options carry data-ready/data-n-annotated/
        data-image-present attributes the JS uses to drive the
        readiness banner. Levels with rooms but no map image are
        flagged not-ready, levels with image+rooms are ready."""
        import shutil
        monkeypatch.setattr(shutil, "which", lambda n: "/usr/local/bin/" + n)
        _, host, port, _ = multi_server
        # SEED_DUNGEON's level 1 has a couple rooms but no PNG file
        # in tmp_path → image_present should be 0, ready should be 0.
        resp, body = _http_get(host, port, "/assistant?dungeon=alpha")
        assert resp.status == 200
        text = body.decode("utf-8")
        assert 'data-ready="0"' in text
        assert 'data-image-present="0"' in text
        # The "(not ready)" suffix shows in the option label too.
        assert "(not ready)" in text

    def test_assistant_start_refuses_when_level_has_no_annotated_rooms(
        self, multi_server, monkeypatch
    ):
        """Start is gated server-side too — even if a JS-disabled
        client POSTed to /assistant/start, an empty level returns 400
        with a level_not_ready error and the message that explains
        what to fix."""
        import shutil
        monkeypatch.setattr(shutil, "which", lambda n: "/usr/local/bin/" + n)
        _, host, port, root = multi_server
        # Strip image_region from every room on alpha's L1 so it counts
        # as un-annotated.
        alpha_path = root / "alpha" / "dungeon.json"
        d = dungeon_mod.load(alpha_path)
        for r in d.levels[0].rooms:
            r.image_region = None
        dungeon_mod.dump(d, alpha_path)

        resp, raw = _http_post_json(host, port, "/assistant/start", {
            "theme": "tomb of test",
            "level_number": 1,
            "party_level": 3,
            "model": "claude-sonnet-4-6",
            "dungeon_folder": "alpha",
        })
        assert resp.status == 400
        payload = json.loads(raw)
        assert payload["error"] == "level_not_ready"
        assert "annotated" in payload["message"]
        assert "annotation" in payload["message"]

    def test_assistant_page_renders_theme_textarea_not_input(
        self, multi_server, monkeypatch,
    ):
        """The Theme field is a multi-line textarea so the DM can paste
        a paragraph-length concept (level structure, current events,
        key NPCs)."""
        import shutil
        monkeypatch.setattr(shutil, "which", lambda n: "/usr/local/bin/" + n)
        _, host, port, _ = multi_server
        resp, body = _http_get(host, port, "/assistant")
        assert resp.status == 200
        text = body.decode("utf-8")
        # Textarea, not single-line input.
        assert '<textarea id="theme"' in text
        # Field label hints at the richer purpose.
        assert "Theme &amp; concept" in text or "Theme & concept" in text
        # Old single-line input shape is gone.
        assert '<input type="text" id="theme"' not in text

    def test_assistant_start_accepts_multiline_theme(
        self, multi_server, monkeypatch,
    ):
        """A multi-paragraph theme should round-trip through the start
        endpoint to the AssistantSession (verified by the level-not-ready
        guard firing AFTER the theme is read — proves the body parsed)."""
        import shutil
        monkeypatch.setattr(shutil, "which", lambda n: "/usr/local/bin/" + n)
        _, host, port, root = multi_server
        # Strip image_region so start fails on level_not_ready instead
        # of blowing budget on a real CLI call. We only care that the
        # multi-line theme parsed.
        alpha_path = root / "alpha" / "dungeon.json"
        d = dungeon_mod.load(alpha_path)
        for r in d.levels[0].rooms:
            r.image_region = None
        dungeon_mod.dump(d, alpha_path)

        long_theme = (
            "A long-lost assassins cult.\n\n"
            "Three levels: living quarters, torture and dungeon "
            "areas, then the temple and holy relics.\n\n"
            "Recently a desert genie has burst in and is conducting a "
            "ritual on the third level."
        )
        resp, raw = _http_post_json(host, port, "/assistant/start", {
            "theme": long_theme,
            "level_number": 1,
            "party_level": 3,
            "model": "claude-sonnet-4-6",
            "dungeon_folder": "alpha",
        })
        # Falls through to the level_not_ready guard, which is exactly
        # the post-parse path — proves the body parsed cleanly.
        assert resp.status == 400
        payload = json.loads(raw)
        assert payload["error"] == "level_not_ready"

    def test_apply_with_dungeon_folder_targets_picked_dungeon(
        self, multi_server, monkeypatch
    ):
        # Inject a session against `beta` (not the bound default `alpha`)
        # and confirm Apply writes to beta's dungeon.json.
        import dungeon_assistant as da
        import shutil
        monkeypatch.setattr(shutil, "which", lambda n: "/usr/local/bin/" + n)
        _, host, port, root = multi_server
        beta_path = root / "beta" / "dungeon.json"
        d_beta = dungeon_mod.load(beta_path)
        sess = da.AssistantSession(
            dungeon_path=beta_path,
            dungeon=d_beta,
            theme="x",
            level_number=1,
            party_level=5,
            runner=lambda args: {"is_error": False, "result": "",
                                  "structured_output": {},
                                  "session_id": "sesn-x"},
        )
        sess._proposals["R01"] = da.RoomProposal(
            id="R01", name="Beta Hall",
            tags=("encounter",),
            box_text="A beta-only room.",
        )
        editor_server._assistant_sessions[beta_path.resolve()] = sess

        resp, body = _http_post_json(host, port, "/assistant/apply",
                                     {"room_id": "R01",
                                      "dungeon_folder": "beta"})
        assert resp.status == 200, body
        assert b"Beta Hall" in body
        # Beta's JSON got the change; alpha's didn't.
        d_beta_after = dungeon_mod.load(beta_path)
        assert d_beta_after.levels[0].rooms_by_id["R01"].name == "Beta Hall"
        d_alpha_after = dungeon_mod.load(root / "alpha" / "dungeon.json")
        assert d_alpha_after.levels[0].rooms_by_id["R01"].name != "Beta Hall"


class TestWorkflowStatus:
    """`_workflow_status` is the source of truth for the Workflow tab.
    Six steps, each with a state that should advance from todo → done
    as the user finishes setup. The tests walk the dungeon through the
    lifecycle and assert each step flips when its precondition is met."""

    def test_fresh_dungeon_reports_todo_states(self, dungeon_json):
        # SEED_DUNGEON has 2 annotated rooms but no map PNG on disk,
        # no character JSONs, and no encounter text. So:
        #   step 0 (map):        todo (PNG missing)
        #   step 1 (annotate):   done (2 rooms with image_region)
        #   step 2 (populate):   todo (rooms empty)
        #   step 3 (characters): todo
        #   step 4 (simulate):   blocked (no chars, no encounters)
        #   step 5 (play):       ready (always)
        d = dungeon_mod.load(dungeon_json)
        steps = editor_server._workflow_status(d, dungeon_json)
        states = {s["n"]: s["state"] for s in steps}
        assert states[0] == "todo"
        assert states[1] == "done"
        assert states[2] == "todo"
        assert states[3] == "todo"
        assert states[4] == "blocked"
        assert states[5] == "ready"

    def test_map_step_done_when_png_exists(self, tmp_path):
        # Build a minimal dungeon whose map_image points at a PNG we
        # actually write to disk.
        json_path = tmp_path / "d.json"
        png_path = tmp_path / "level1.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"X" * 50)
        seed = dict(SEED_DUNGEON)
        seed["levels"] = [dict(SEED_DUNGEON["levels"][0],
                               map_image="level1.png")]
        json_path.write_text(json.dumps(seed))
        d = dungeon_mod.load(json_path)
        steps = editor_server._workflow_status(d, json_path)
        assert steps[0]["state"] == "done"

    def test_populate_done_when_rooms_have_text(self, tmp_path):
        seed = dict(SEED_DUNGEON)
        rooms = [dict(r) for r in SEED_DUNGEON["levels"][0]["rooms"]]
        for r in rooms:
            r["encounter_text"] = "2 Skeletons (MM p.272). Hostile."
        seed["levels"] = [dict(SEED_DUNGEON["levels"][0], rooms=rooms)]
        json_path = tmp_path / "d.json"
        json_path.write_text(json.dumps(seed))
        d = dungeon_mod.load(json_path)
        steps = editor_server._workflow_status(d, json_path)
        # Step 2 (populate) flips done; step 4 (simulate) still blocked
        # because there are no characters yet.
        assert steps[2]["state"] == "done"
        assert steps[4]["state"] == "blocked"

    def test_simulate_ready_when_chars_and_encounters_present(self, tmp_path):
        seed = dict(SEED_DUNGEON)
        rooms = [dict(r) for r in SEED_DUNGEON["levels"][0]["rooms"]]
        rooms[0]["encounter_text"] = "1 Wight (MM p. 300)."
        seed["levels"] = [dict(SEED_DUNGEON["levels"][0], rooms=rooms)]
        json_path = tmp_path / "d.json"
        json_path.write_text(json.dumps(seed))
        # Drop a character JSON in the conventional place.
        chars_dir = tmp_path / "characters"
        chars_dir.mkdir()
        (chars_dir / "hero.json").write_text(
            json.dumps({"name": "Hero", "level": 1})
        )
        d = dungeon_mod.load(json_path)
        steps = editor_server._workflow_status(d, json_path)
        assert steps[3]["state"] == "done"
        assert steps[4]["state"] == "ready"

    def test_shell_defaults_to_workflow_for_fresh_dungeon(self, tmp_path):
        # No image_region on any room → shell should auto-pick the
        # workflow tab as the default-active iframe.
        seed = dict(SEED_DUNGEON)
        rooms = [dict(r) for r in SEED_DUNGEON["levels"][0]["rooms"]]
        for r in rooms:
            r.pop("image_region", None)  # un-annotate
        seed["levels"] = [dict(SEED_DUNGEON["levels"][0], rooms=rooms)]
        json_path = tmp_path / "d.json"
        json_path.write_text(json.dumps(seed))
        srv, _ = editor_server.start_editor_server(json_path, port=0)
        host, port = srv.server_address[:2]
        try:
            resp, body = _http_get(host, port, "/")
            assert resp.status == 200
            text = body.decode("utf-8")
            # Workflow tab has the active class.
            assert ('id="tab-workflow" data-tab="workflow"\n     class="active"'
                    in text)
            assert 'id="frame-workflow"' in text
        finally:
            srv.shutdown()


class TestNewDungeonEndpoint:
    """POST /workflow/new_dungeon scaffolds a new dungeon under the
    server's dungeons_dir. Doesn't try to switch the running pygame
    onto it — that's the launcher's job."""

    @pytest.fixture
    def server_with_dungeons_dir(self, tmp_path, dungeon_json):
        # Re-use the seed dungeon fixture as the "current" dungeon,
        # but place dungeons_dir at tmp_path/dungeons so the scaffolder
        # writes into a known empty root.
        dungeons_root = tmp_path / "dungeons"
        dungeons_root.mkdir()
        srv, _ = editor_server.start_editor_server(
            dungeon_json, port=0, dungeons_dir=dungeons_root,
        )
        host, port = srv.server_address[:2]
        try:
            yield srv, host, port, dungeons_root
        finally:
            srv.shutdown()

    def test_scaffolds_new_dungeon(self, server_with_dungeons_dir):
        _, host, port, root = server_with_dungeons_dir
        resp, raw = _http_post_json(
            host, port, "/workflow/new_dungeon",
            {"name": "The Whispering Vault", "party_level": 5,
             "party_size": 4},
        )
        data = json.loads(raw.decode("utf-8"))
        assert resp.status == 200, data
        assert data["ok"] is True
        assert data["folder"] == "the-whispering-vault"
        target = root / "the-whispering-vault"
        assert target.exists()
        # Scaffolded dungeon.json should have the requested fields.
        scaffold = json.loads((target / "dungeon.json").read_text())
        assert scaffold["dungeon_name"] == "The Whispering Vault"
        assert scaffold["party_level"] == 5
        assert len(scaffold["levels"]) == 1
        assert scaffold["levels"][0]["rooms"] == []

    def test_rejects_empty_name(self, server_with_dungeons_dir):
        _, host, port, _ = server_with_dungeons_dir
        resp, raw = _http_post_json(
            host, port, "/workflow/new_dungeon",
            {"name": "   ", "party_level": 3},
        )
        data = json.loads(raw.decode("utf-8"))
        assert resp.status == 400
        assert data["ok"] is False
        assert "required" in data["error"].lower()

    def test_rejects_slug_collision(self, server_with_dungeons_dir):
        _, host, port, root = server_with_dungeons_dir
        # Pre-create a folder so the second scaffold collides.
        (root / "duplicate-name").mkdir()
        resp, raw = _http_post_json(
            host, port, "/workflow/new_dungeon",
            {"name": "Duplicate Name", "party_level": 3},
        )
        data = json.loads(raw.decode("utf-8"))
        assert resp.status == 409
        assert data["ok"] is False
        assert "already exists" in data["error"].lower()

    def test_punctuation_name_falls_back_to_untitled(self, server_with_dungeons_dir):
        # slugify_dungeon_name returns "untitled-dungeon" when the
        # name has nothing slugable in it. Document that behavior
        # here so a future tightening of the slugger surfaces in tests.
        _, host, port, root = server_with_dungeons_dir
        resp, raw = _http_post_json(
            host, port, "/workflow/new_dungeon",
            {"name": "!!!", "party_level": 3},
        )
        data = json.loads(raw.decode("utf-8"))
        assert resp.status == 200, data
        assert data["folder"] == "untitled-dungeon"
        assert (root / "untitled-dungeon").exists()
