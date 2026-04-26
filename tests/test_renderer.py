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

    def test_delete_selected_room_removes_it(self, session, tmp_path):
        view = MapView(session, dungeon_path=tmp_path / "annotated.json")
        view.toggle_annotation_mode()
        view._annot_selected_room_id = view.level.rooms[0].id
        before = len(view.level.rooms)
        view._delete_selected_room()
        assert len(view.level.rooms) == before - 1


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
        view._annot_selected_room_id = "R01"
        view._delete_selected_room()
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
