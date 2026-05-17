"""Smoke tests for the PNG-based renderer with annotation regions.

The renderer now builds its label map at runtime by rasterizing each
room's `image_region` (rect or polygon) — there's no analyzer pipeline.
These tests cover that rasterization path, the fog-mask helpers, and a
headless MapView smoke run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame
import pytest

import dungeon as dungeon_mod
from dungeon import ImageRegion
from renderer import (
    Camera,
    MapView,
    _alpha_to_fog_surface,
    _build_fog_alpha,
    _point_in_polygon,
    _resolve_image_path,
    _topmost_room_at,
    build_revealed_mask,
)
from session import Session


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORREL_JSON = PROJECT_ROOT / "data" / "torrel.json"
TEMPLE_PNG = PROJECT_ROOT / "target_dungeon_maps" / "ancient-temple-of-torrel-level1.png"


@pytest.fixture(scope="module", autouse=True)
def _pygame_init():
    pygame.init()
    pygame.font.init()
    yield
    pygame.quit()


@pytest.fixture
def torrel(tmp_path):
    """Make a temp copy of torrel.json + a level with two annotated rooms,
    so tests don't depend on an authored dungeon and don't mutate the repo
    file on save."""
    src = dungeon_mod.load(TORREL_JSON)
    # Drop levels 2 & 3 to keep the test focused; add two rect rooms on L1.
    l1 = src.get_level(1)
    from dungeon import Room
    rooms = (
        Room(id="R01", name="R01", state="unexplored", tags=("empty",),
             image_region=ImageRegion(kind="rect", rect=(100, 100, 300, 200))),
        Room(id="R02", name="R02", state="unexplored", tags=("encounter",),
             image_region=ImageRegion(kind="polygon",
                                      points=((600, 600), (900, 600),
                                              (900, 900), (700, 900)))),
    )
    l1.rooms = rooms
    l1.rooms_by_id = {r.id: r for r in rooms}
    src.levels = (l1,)
    src.levels_by_number = {1: l1}
    src.current_level = 1
    out = tmp_path / "test_torrel.json"
    dungeon_mod.dump(src, out)
    return out


@pytest.fixture
def session(tmp_path, torrel):
    d = dungeon_mod.load(torrel)
    s = Session.create(tmp_path / "renderer.db", d, torrel, rng_seed=42)
    yield s
    s.close()


# --- ImageRegion --------------------------------------------------------------


class TestImageRegion:
    def test_rect_to_dict_round_trip(self):
        r = ImageRegion(kind="rect", rect=(10, 20, 30, 40))
        d = r.to_dict()
        assert d == {"kind": "rect", "x": 10, "y": 20, "width": 30, "height": 40}

    def test_polygon_to_dict_round_trip(self):
        r = ImageRegion(kind="polygon", points=((0, 0), (10, 0), (10, 10), (0, 10)))
        d = r.to_dict()
        assert d["kind"] == "polygon"
        assert d["points"] == [[0, 0], [10, 0], [10, 10], [0, 10]]

    def test_rect_centroid(self):
        r = ImageRegion(kind="rect", rect=(0, 0, 100, 50))
        assert r.centroid() == (50, 25)

    def test_polygon_centroid(self):
        r = ImageRegion(kind="polygon", points=((0, 0), (10, 0), (10, 10), (0, 10)))
        cx, cy = r.centroid()
        assert (cx, cy) == (5, 5)


# --- dump round-trip ----------------------------------------------------------


class TestDumpRoundTrip:
    def test_dump_then_load_preserves_image_region(self, tmp_path):
        d = dungeon_mod.load(TORREL_JSON)
        from dungeon import Room
        l1 = d.get_level(1)
        rooms = (Room(id="R01", name="Test", state="unexplored", tags=("empty",),
                      image_region=ImageRegion(kind="rect", rect=(5, 6, 7, 8))),)
        l1.rooms = rooms
        l1.rooms_by_id = {"R01": rooms[0]}
        out = tmp_path / "round.json"
        dungeon_mod.dump(d, out)
        d2 = dungeon_mod.load(out)
        r2 = d2.get_level(1).rooms_by_id["R01"]
        assert r2.image_region is not None
        assert r2.image_region.kind == "rect"
        assert r2.image_region.rect == (5, 6, 7, 8)


# --- Camera -------------------------------------------------------------------


class TestCamera:
    def test_round_trip(self):
        c = Camera(zoom=1.5, offset_x=40, offset_y=-20)
        for wx, wy in [(0, 0), (10, 50), (-30, 100)]:
            sx, sy = c.world_to_screen(wx, wy)
            wx2, wy2 = c.screen_to_world(sx, sy)
            assert wx2 == pytest.approx(wx)
            assert wy2 == pytest.approx(wy)

    def test_zoom_at_preserves_cursor(self):
        c = Camera()
        cursor = (300, 200)
        wx_before, wy_before = c.screen_to_world(*cursor)
        c.zoom_at(*cursor, factor=1.5)
        wx_after, wy_after = c.screen_to_world(*cursor)
        assert wx_after == pytest.approx(wx_before)
        assert wy_after == pytest.approx(wy_before)


# --- Label map rasterization --------------------------------------------------


def _make_level(rooms):
    from dungeon import Level
    return Level(
        level_number=1, display_name="L", map_image="x.png",
        map_image_scale=1.0, wm_check_method="d20", wm_check_threshold=18,
        wm_check_frequency="every_turn", wandering_monster_table=(),
        rooms=tuple(rooms), corridors=(),
    )


class TestRevealedMask:
    def test_unrevealed_rooms_yield_empty_mask(self):
        from dungeon import Room
        rooms = [Room(id="R01", name="A", state="unexplored", tags=("empty",),
                      image_region=ImageRegion(kind="rect", rect=(5, 5, 10, 10)))]
        level = _make_level(rooms)
        mask = build_revealed_mask(level, (50, 50), revealed_room_ids=set())
        assert not mask.any()

    def test_revealed_rect_fills_only_its_pixels(self):
        from dungeon import Room
        rooms = [Room(id="R01", name="A", state="unexplored", tags=("empty",),
                      image_region=ImageRegion(kind="rect", rect=(5, 5, 10, 10)))]
        level = _make_level(rooms)
        mask = build_revealed_mask(level, (50, 50), revealed_room_ids={"R01"})
        assert mask[10, 10]
        assert not mask[0, 0]
        assert not mask[40, 40]

    def test_overlapping_rooms_both_reveal_their_pixels(self):
        """The bug we fixed: a later-drawn room *used* to obscure an
        earlier room on overlapping pixels because of label-overwrite.
        Now every revealed room contributes; OR is idempotent."""
        from dungeon import Room
        rooms = [
            Room(id="R01", name="A", state="unexplored", tags=("empty",),
                 image_region=ImageRegion(kind="rect", rect=(0, 0, 20, 20))),
            Room(id="R02", name="B", state="unexplored", tags=("empty",),
                 image_region=ImageRegion(kind="rect", rect=(10, 10, 20, 20))),
        ]
        level = _make_level(rooms)
        # Reveal only R01. The overlap region (10..20, 10..20) belongs to
        # both rooms; with R01 revealed, those pixels must be in the mask
        # — even though R02 was drawn later.
        mask = build_revealed_mask(level, (40, 40), revealed_room_ids={"R01"})
        assert mask[5, 5]      # R01 only
        assert mask[15, 15]    # overlap — must still reveal because R01 covers it
        assert not mask[25, 25]  # R02 only — not revealed


class TestHitTesting:
    def test_topmost_wins_overlap(self):
        from dungeon import Room
        rooms = [
            Room(id="R01", name="A", state="unexplored", tags=("empty",),
                 image_region=ImageRegion(kind="rect", rect=(0, 0, 30, 30))),
            Room(id="R02", name="B", state="unexplored", tags=("empty",),
                 image_region=ImageRegion(kind="rect", rect=(10, 10, 30, 30))),
        ]
        level = _make_level(rooms)
        # Inside both rects: most recently drawn (R02) wins.
        assert _topmost_room_at(level, 15, 15).id == "R02"
        # Inside R01 only.
        assert _topmost_room_at(level, 5, 5).id == "R01"
        # Inside R02 only.
        assert _topmost_room_at(level, 35, 35).id == "R02"
        # Outside both.
        assert _topmost_room_at(level, 100, 100) is None

    @pytest.mark.parametrize("x,y,inside", [
        (5, 5, True), (15, 5, True), (15, 15, True),  # inside square
        (-1, 5, False), (25, 5, False), (5, 25, False),  # outside
    ])
    def test_point_in_polygon_axis_aligned(self, x, y, inside):
        pts = ((0, 0), (20, 0), (20, 20), (0, 20))
        assert _point_in_polygon(x, y, pts) is inside


class TestFog:
    def test_alpha_zero_where_revealed(self):
        mask = np.array([[True, False, True], [False, False, True]])
        alpha = _build_fog_alpha(mask, base_alpha=200)
        assert alpha[0, 0] == 0
        assert alpha[0, 1] == 200
        assert alpha[1, 2] == 0

    def test_alpha_to_surface_is_srcalpha(self):
        alpha = np.full((10, 10), 100, dtype=np.uint8)
        surf = _alpha_to_fog_surface(alpha, (0, 0, 0))
        assert surf.get_size() == (10, 10)
        assert surf.get_flags() & pygame.SRCALPHA


# --- Image path resolution ---------------------------------------------------


class TestResolveImagePath:
    """`_resolve_image_path` keeps a dungeon folder portable: a level's
    `map_image` is first looked up next to the dungeon JSON, then falls
    back to PROJECT_ROOT for the legacy in-repo layout."""

    def _level(self, map_image: str):
        from dungeon import Level
        return Level(
            level_number=1, display_name="L1",
            map_image=map_image, map_image_scale=1.0,
            wm_check_method="d20", wm_check_threshold=18,
            wm_check_frequency="every_turn",
            wandering_monster_table=(),
            rooms=(), corridors=(),
        )

    def test_dungeon_dir_takes_precedence(self, tmp_path):
        # Create the file inside dungeon_dir so the JSON-relative branch hits.
        (tmp_path / "level1.png").write_bytes(b"fake")
        path = _resolve_image_path(self._level("level1.png"), tmp_path)
        assert path == tmp_path / "level1.png"

    def test_falls_back_to_project_root_when_missing_in_dungeon_dir(self, tmp_path):
        # No file in tmp_path → resolver returns PROJECT_ROOT/<path> as fallback.
        path = _resolve_image_path(self._level("legacy/foo.png"), tmp_path)
        assert path == PROJECT_ROOT / "legacy" / "foo.png"

    def test_absolute_path_passes_through(self, tmp_path):
        abs_png = tmp_path / "abs.png"
        abs_png.write_bytes(b"fake")
        path = _resolve_image_path(self._level(str(abs_png)), tmp_path / "other")
        assert path == abs_png

    def test_no_dungeon_dir_uses_project_root(self):
        path = _resolve_image_path(self._level("data/x.png"), None)
        assert path == PROJECT_ROOT / "data" / "x.png"


# --- MapView smoke ------------------------------------------------------------


class TestMapView:
    def test_loads_image(self, session):
        view = MapView(session)
        assert view.image is not None

    def test_click_inside_region_cycles_state(self, session):
        view = MapView(session)
        pygame.display.set_mode((640, 480))
        view._fit_to_window()
        room = session.dungeon.current.rooms[0]
        new_state = view.cycle_room_state(room.id)
        assert new_state == "known"

    def test_add_room_via_annotation_appends(self, session, tmp_path):
        view = MapView(session, dungeon_path=tmp_path / "annotated.json")
        before = len(view.level.rooms)
        view._add_room_with_region(
            ImageRegion(kind="rect", rect=(200, 300, 300, 200))
        )
        assert len(view.level.rooms) == before + 1
        assert view.level.rooms[-1].image_region.kind == "rect"

    def test_annotation_mode_polygon_creates_room(self, session, tmp_path):
        view = MapView(session, dungeon_path=tmp_path / "annotated.json")
        before = len(view.level.rooms)
        view.toggle_annotation_mode()
        view._annot_tool = "polygon"
        view._annot_polygon_points = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]
        view._commit_polygon()
        assert len(view.level.rooms) == before + 1
        new_room = view.level.rooms[-1]
        assert new_room.image_region.kind == "polygon"

    def test_delete_hovered_room_removes_it(self, session, tmp_path):
        view = MapView(session, dungeon_path=tmp_path / "annotated.json")
        view.toggle_annotation_mode()
        view._annot_hovered_room_id = view.level.rooms[0].id
        before = len(view.level.rooms)
        view._delete_hovered_room()
        assert len(view.level.rooms) == before - 1

    def test_delete_with_no_hover_is_noop(self, session, tmp_path):
        view = MapView(session, dungeon_path=tmp_path / "annotated.json")
        view.toggle_annotation_mode()
        view._annot_hovered_room_id = None
        before = len(view.level.rooms)
        view._delete_hovered_room()
        assert len(view.level.rooms) == before  # still all there

    def test_mouse_motion_in_annotation_updates_hover(
        self, session, tmp_path,
    ):
        """The MOUSEMOTION handler in annotation mode tracks which
        room is under the cursor, so hover+Del knows what to delete."""
        view = MapView(session, dungeon_path=tmp_path / "annotated.json")
        pygame.display.set_mode((1280, 800))
        view._fit_to_window()
        view.toggle_annotation_mode()
        room = view.level.rooms[0]
        # Use the room's centroid as a guaranteed-inside-the-region
        # world point, then convert to screen coords.
        cx, cy = room.image_region.centroid()
        sx, sy = view.camera.world_to_screen(cx, cy)
        ev = pygame.event.Event(pygame.MOUSEMOTION,
                                pos=(int(sx), int(sy)),
                                rel=(0, 0), buttons=(0, 0, 0))
        view._handle_annotation_event(ev)
        assert view._annot_hovered_room_id == room.id

        # Move off the room → hover clears.
        ev2 = pygame.event.Event(pygame.MOUSEMOTION,
                                 pos=(0, 0),
                                 rel=(0, 0), buttons=(0, 0, 0))
        view._handle_annotation_event(ev2)
        assert view._annot_hovered_room_id is None


class TestNewDungeonModal:
    """The pygame Options menu's 'New Dungeon…' modal: scaffolds
    `dungeons/<slug>/dungeon.json` from a typed name and queues a
    ReloadRequest so main.py swaps onto it."""

    def test_modal_starts_closed(self, session, tmp_path):
        view = MapView(session, dungeons_dir=tmp_path / "dungeons")
        assert view._new_dungeon_open is False
        assert view._new_dungeon_name == ""

    def test_open_resets_form_state(self, session, tmp_path):
        view = MapView(session, dungeons_dir=tmp_path / "dungeons")
        view._new_dungeon_name = "old text"
        view._new_dungeon_error = "stale error"
        view.open_new_dungeon_modal()
        assert view._new_dungeon_open
        assert view._new_dungeon_name == ""
        assert view._new_dungeon_party_level == 3
        assert view._new_dungeon_focus == "name"
        assert view._new_dungeon_error == ""

    def test_open_noops_without_dungeons_dir(self, session):
        view = MapView(session, dungeons_dir=None)
        view.open_new_dungeon_modal()
        assert view._new_dungeon_open is False

    def test_keydown_appends_printable_to_name(self, session, tmp_path):
        view = MapView(session, dungeons_dir=tmp_path / "dungeons")
        view.open_new_dungeon_modal()
        for ch in "Tomb of Test":
            ev = pygame.event.Event(
                pygame.KEYDOWN, key=pygame.K_a, unicode=ch, mod=0,
            )
            view._handle_new_dungeon_keydown(ev)
        assert view._new_dungeon_name == "Tomb of Test"

    def test_keydown_backspace_trims_name(self, session, tmp_path):
        view = MapView(session, dungeons_dir=tmp_path / "dungeons")
        view.open_new_dungeon_modal()
        view._new_dungeon_name = "abcde"
        ev = pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_BACKSPACE, unicode="", mod=0,
        )
        view._handle_new_dungeon_keydown(ev)
        assert view._new_dungeon_name == "abcd"

    def test_tab_swaps_focus(self, session, tmp_path):
        view = MapView(session, dungeons_dir=tmp_path / "dungeons")
        view.open_new_dungeon_modal()
        ev = pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_TAB, unicode="\t", mod=0,
        )
        view._handle_new_dungeon_keydown(ev)
        assert view._new_dungeon_focus == "party_level"
        view._handle_new_dungeon_keydown(ev)
        assert view._new_dungeon_focus == "name"

    def test_party_level_arrow_keys_clamp(self, session, tmp_path):
        view = MapView(session, dungeons_dir=tmp_path / "dungeons")
        view.open_new_dungeon_modal()
        view._new_dungeon_focus = "party_level"
        view._new_dungeon_party_level = 1
        # Down at floor stays at 1.
        ev_down = pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_DOWN, unicode="", mod=0,
        )
        view._handle_new_dungeon_keydown(ev_down)
        assert view._new_dungeon_party_level == 1
        # Up steps; ceiling is 20.
        ev_up = pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_UP, unicode="", mod=0,
        )
        for _ in range(25):
            view._handle_new_dungeon_keydown(ev_up)
        assert view._new_dungeon_party_level == 20

    def test_create_with_empty_name_shows_error(self, session, tmp_path):
        view = MapView(session, dungeons_dir=tmp_path / "dungeons")
        view.open_new_dungeon_modal()
        view._do_create_new_dungeon()
        assert view._new_dungeon_error
        assert view._pending_reload is None

    def test_create_with_collision_shows_error(self, session, tmp_path):
        # Pre-create a colliding folder.
        dungeons = tmp_path / "dungeons"
        dungeons.mkdir()
        (dungeons / "test-thing").mkdir()
        view = MapView(session, dungeons_dir=dungeons)
        view.open_new_dungeon_modal()
        view._new_dungeon_name = "Test Thing"
        view._do_create_new_dungeon()
        assert "already exists" in view._new_dungeon_error
        assert view._pending_reload is None

    def test_create_success_scaffolds_and_queues_reload(
        self, session, tmp_path,
    ):
        dungeons = tmp_path / "dungeons"
        dungeons.mkdir()
        view = MapView(session, dungeons_dir=dungeons)
        view.open_new_dungeon_modal()
        view._new_dungeon_name = "Brand New Dungeon"
        view._new_dungeon_party_level = 5
        view._do_create_new_dungeon()

        assert view._new_dungeon_error == ""
        assert view._new_dungeon_open is False
        # Folder + JSON exist on disk.
        target = dungeons / "brand-new-dungeon"
        assert (target / "dungeon.json").exists()
        # Reload queued onto the new folder.
        assert view._pending_reload is not None
        assert view._pending_reload.folder.resolve() == target.resolve()
        assert view._pending_reload.do_full_reset is False
        # Loaded dungeon picks up the typed party_level + name.
        d = dungeon_mod.load(target / "dungeon.json")
        assert d.name == "Brand New Dungeon"
        assert d.party_level == 5


class TestExternalReload:
    """The renderer polls the dungeon JSON's mtime and merges metadata
    fields when an external process (the editor server) edits the file."""

    def test_reload_updates_metadata(self, session, torrel):
        view = MapView(session, dungeon_path=torrel)
        view._add_room_with_region(
            ImageRegion(kind="rect", rect=(10, 10, 20, 20))
        )
        rid = view.level.rooms[-1].id
        view.cycle_room_state(rid)
        assert view.level.rooms_by_id[rid].state == "known"

        # Simulate an external edit (the editor server pattern):
        d = dungeon_mod.load(torrel)
        d.get_level(1).rooms_by_id[rid].box_text = "External edit applied."
        dungeon_mod.dump(d, torrel)

        view._reload_dungeon_metadata()
        live = view.level.rooms_by_id[rid]
        assert live.box_text == "External edit applied."
        # Reveal state preserved.
        assert live.state == "known"
        # Region preserved (geometry shouldn't flow through the metadata path).
        assert live.image_region is not None
        assert live.image_region.kind == "rect"

    def test_reload_does_not_reset_state_for_other_rooms(self, session, torrel):
        view = MapView(session, dungeon_path=torrel)
        view._add_room_with_region(
            ImageRegion(kind="rect", rect=(10, 10, 20, 20))
        )
        view._add_room_with_region(
            ImageRegion(kind="rect", rect=(50, 50, 20, 20))
        )
        for r in view.level.rooms:
            view.cycle_room_state(r.id)
        # External edit on one room.
        d = dungeon_mod.load(torrel)
        d.get_level(1).rooms[0].notes = "edited"
        dungeon_mod.dump(d, torrel)
        view._reload_dungeon_metadata()
        for r in view.level.rooms:
            assert r.state == "known"


class TestActionButtons:
    """Bottom-strip session actions exposed via MapView. Verify
    pre/postconditions are wired correctly so the buttons match what the
    user sees in the strip."""

    def test_advance_turn_increments(self, session):
        view = MapView(session)
        before = session.tracker.turn
        view.action_advance_turn()
        assert session.tracker.turn == before + 1

    def test_light_torch_consumes_supply_and_adds_source(self, session):
        view = MapView(session)
        before_torches = session.get_supplies()["torches"]
        before_lights = len(session.tracker.light_sources)
        view.action_light_torch()
        assert session.get_supplies()["torches"] == before_torches - 1
        assert len(session.tracker.light_sources) == before_lights + 1
        assert session.tracker.light_sources[-1].kind == "torch"

    def test_light_torch_disabled_when_stash_empty(self, session):
        view = MapView(session)
        session.set_supply_count("torches", 0)
        assert view._can_light_torch() is False
        view.action_light_torch()
        assert len(session.tracker.light_sources) == 0  # no-op

    def test_refill_lantern_when_no_lantern_disabled(self, session):
        view = MapView(session)
        # Fresh session has 0 active lanterns.
        assert view._can_refill_lantern() is False

    def test_refill_lantern_resets_to_full_duration(self, session):
        view = MapView(session)
        view.action_light_lantern()
        ls = view._active_lantern()
        # Burn it down a bit by simulating turns.
        ls.turns_remaining = 5
        before_oil = session.get_supplies()["oil_flask"]
        view.action_refill_lantern()
        # 36 turns is the hooded_lantern duration in config.LIGHT_DURATIONS_TURNS.
        assert ls.turns_remaining == 36
        assert session.get_supplies()["oil_flask"] == before_oil - 1

    def test_lighting_lantern_does_not_consume_lantern(self, session):
        """Regression: the lantern is a permanent object. Lighting it
        consumes oil, not the lantern itself."""
        view = MapView(session)
        before = session.get_supplies()
        before_l = before["hooded_lantern"]
        before_o = before["oil_flask"]
        view.action_light_lantern()
        after = session.get_supplies()
        assert after["hooded_lantern"] == before_l       # lantern preserved
        assert after["oil_flask"] == before_o - 1        # oil consumed

    def test_cannot_light_second_lantern_when_only_one_owned(self, session):
        """L:1 + 1 active lantern means we can't light another (you'd be
        lighting the same physical lantern twice)."""
        view = MapView(session)
        view.action_light_lantern()  # 1 active
        assert view._can_light_lantern() is False
        view.action_light_lantern()  # no-op
        assert sum(1 for ls in session.tracker.light_sources
                   if ls.kind == "hooded_lantern") == 1

    def test_toggle_noisy_flips(self, session):
        view = MapView(session)
        assert session.tracker.noisy is False
        view.action_toggle_noisy()
        assert session.tracker.noisy is True
        view.action_toggle_noisy()
        assert session.tracker.noisy is False

    def test_manual_wm_roll_does_not_advance_turn(self, session):
        view = MapView(session)
        before = session.tracker.turn
        view.action_manual_wm_roll()
        assert session.tracker.turn == before
        assert session.tracker.last_wm is not None


class TestUndo:
    def test_undo_add_removes_added_room(self, session, tmp_path):
        view = MapView(session, dungeon_path=tmp_path / "annotated.json")
        before = len(view.level.rooms)
        view._add_room_with_region(
            ImageRegion(kind="rect", rect=(100, 100, 50, 50))
        )
        assert len(view.level.rooms) == before + 1
        assert view.undo_last_annotation() is True
        assert len(view.level.rooms) == before

    def test_undo_delete_restores_room_with_state(self, session, tmp_path):
        view = MapView(session, dungeon_path=tmp_path / "annotated.json")
        view.cycle_room_state("R01")  # → known
        assert view.level.rooms_by_id["R01"].state == "known"
        # Now delete it.
        view._annot_hovered_room_id = "R01"
        view._delete_hovered_room()
        assert "R01" not in view.level.rooms_by_id
        # Undo restores it AND its prior state.
        view.undo_last_annotation()
        assert "R01" in view.level.rooms_by_id
        assert view.level.rooms_by_id["R01"].state == "known"

    def test_undo_returns_false_on_empty_stack(self, session, tmp_path):
        view = MapView(session, dungeon_path=tmp_path / "annotated.json")
        # Drain any actions that the fixture may have left.
        view._undo_stack.clear()
        assert view.undo_last_annotation() is False

    def test_undo_capped_at_max_depth(self, session, tmp_path):
        from renderer import MAX_UNDO_DEPTH
        view = MapView(session, dungeon_path=tmp_path / "annotated.json")
        view._undo_stack.clear()
        for i in range(MAX_UNDO_DEPTH + 5):
            view._add_room_with_region(
                ImageRegion(kind="rect", rect=(i, i, 10, 10))
            )
        assert len(view._undo_stack) == MAX_UNDO_DEPTH


# --- Level switching reloads assets -------------------------------------------


class TestLevelSwitch:
    @pytest.mark.skipif(
        not (PROJECT_ROOT / "data" / "torrel.json").exists(),
        reason="needs the multi-level torrel.json",
    )
    def test_switch_loads_new_level_assets(self, tmp_path):
        d = dungeon_mod.load(TORREL_JSON)
        if len(d.levels) < 2:
            pytest.skip("fixture has fewer than 2 levels")
        s = Session.create(tmp_path / "ls.db", d, TORREL_JSON, rng_seed=42)
        try:
            pygame.display.set_mode((320, 240))
            view = MapView(s)
            original_image_id = id(view.image)
            s.switch_level(+1)
            view.switch_to_current_level()
            assert id(view.image) != original_image_id
        finally:
            s.close()


class TestTooltipResolver:
    """The tooltip layer's dwell logic is `_resolve_tooltip`. Tests run
    against the staticmethod without a pygame display — pure timing /
    rect-hit logic, all the heavy plate-drawing is elsewhere."""

    DWELL = 400

    def test_no_rect_under_cursor_returns_none(self):
        rects = [(pygame.Rect(0, 0, 10, 10), "first")]
        text, _, _ = MapView._resolve_tooltip(
            rects, mouse_pos=(50, 50),
            hover_rect=None, hover_started_ms=None,
            now_ms=1000, dwell_ms=self.DWELL,
        )
        assert text is None

    def test_hover_before_dwell_no_tooltip(self):
        r = pygame.Rect(0, 0, 10, 10)
        rects = [(r, "first")]
        # Enter the rect at t=0; dwell threshold not yet reached.
        text, hover, started = MapView._resolve_tooltip(
            rects, mouse_pos=(5, 5),
            hover_rect=None, hover_started_ms=None,
            now_ms=0, dwell_ms=self.DWELL,
        )
        assert text is None
        assert hover == r
        assert started == 0
        # Still inside at t=200 (< 400) — still no tooltip.
        text, hover, started = MapView._resolve_tooltip(
            rects, mouse_pos=(5, 5),
            hover_rect=hover, hover_started_ms=started,
            now_ms=200, dwell_ms=self.DWELL,
        )
        assert text is None

    def test_hover_past_dwell_returns_text(self):
        r = pygame.Rect(0, 0, 10, 10)
        rects = [(r, "the-tip")]
        text, hover, started = MapView._resolve_tooltip(
            rects, mouse_pos=(5, 5),
            hover_rect=r, hover_started_ms=0,
            now_ms=self.DWELL, dwell_ms=self.DWELL,
        )
        assert text == "the-tip"

    def test_overlapping_rects_inner_wins(self):
        # Outer registered first, inner registered second — last wins
        # because the more specific control draws on top.
        outer = pygame.Rect(0, 0, 100, 100)
        inner = pygame.Rect(40, 40, 20, 20)
        rects = [(outer, "outer"), (inner, "inner")]
        text, hover, _ = MapView._resolve_tooltip(
            rects, mouse_pos=(50, 50),
            hover_rect=inner, hover_started_ms=0,
            now_ms=self.DWELL, dwell_ms=self.DWELL,
        )
        assert text == "inner"
        assert hover == inner

    def test_moving_to_new_rect_restarts_dwell(self):
        a = pygame.Rect(0, 0, 10, 10)
        b = pygame.Rect(20, 0, 10, 10)
        rects = [(a, "A"), (b, "B")]
        # Was hovering A past dwell; cursor moves into B at t=600.
        text, hover, started = MapView._resolve_tooltip(
            rects, mouse_pos=(25, 5),
            hover_rect=a, hover_started_ms=0,
            now_ms=600, dwell_ms=self.DWELL,
        )
        # New rect — timer restarted, no tooltip yet.
        assert text is None
        assert hover == b
        assert started == 600
