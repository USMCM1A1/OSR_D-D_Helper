"""Pygame DM editor over a pre-drawn map PNG.

The image *is* the dungeon. We don't draw rooms, corridors, or walls —
those are already in `level.map_image`. The editor:

    1. Loads the level's PNG and its companion label map (.npz from
       map_analysis.py — per-pixel component id → which room a click hits).
    2. Builds a fog-of-war mask: opaque over pixels whose component is
       unexplored, transparent over revealed components.
    3. Composites image + fog on screen for the DM, with markers showing
       per-component state (so the DM can see what's revealed even where
       the fog is clear).
    4. On click: looks up the cursor's image-px label, cycles that room
       through unexplored → known → cleared.
    5. After every state or level change, snapshots two PNGs into
       render_output/ — `dm_map.png` (no fog) and `player_map.png` (fog
       opaque over unrevealed). The HTML wrapper auto-refreshes them in
       the player browser tab.

Public API:
    Camera                 -- world↔screen transforms, zoom, pan
    MapView                -- pygame interaction + drawing + snapshotting
    run(session, on_change, on_open_browser)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pygame
from PIL import Image, ImageDraw

import dungeon as dungeon_mod
from dungeon import Dungeon, ImageRegion, Level, Room
from session import Session


# How often to check the dungeon JSON mtime for external edits.
MTIME_POLL_SECONDS = 1.0


# --- Aesthetic constants ----------------------------------------------------

BG_OUTSIDE       = (26, 26, 26)         # window space outside the image
FOG_COLOR        = (12, 8, 4)           # near-black fog
FOG_ALPHA_PLAYER = 240                  # fully obscure unrevealed for players
FOG_ALPHA_DM     = 110                  # DM sees through fog at half-opacity
MARKER_RADIUS    = 18                   # per-component marker
MARKER_RING      = 4
STATE_COLORS = {
    "unexplored": (200, 200, 200, 230),
    "known":      (60, 200, 80, 230),
    "cleared":    (130, 110, 80, 230),
}
ALERT_RED        = (168, 32, 26)
PARTY_DOT        = (255, 220, 60)
PARTY_DOT_RING   = (0, 0, 0)
LABEL_INK        = (40, 40, 40)
HELP_INK         = (110, 110, 110)


DRAG_THRESHOLD_PX = 5
ZOOM_MIN  = 0.05
ZOOM_MAX  = 4.0
ZOOM_STEP = 1.15

MAX_UNDO_DEPTH = 50

STATE_CYCLE = ("unexplored", "known", "cleared")

PROJECT_ROOT = Path(__file__).resolve().parent
RENDER_OUTPUT = PROJECT_ROOT / "render_output"
DM_PNG = RENDER_OUTPUT / "dm_map.png"
PLAYER_PNG = RENDER_OUTPUT / "player_map.png"


# --- Camera ------------------------------------------------------------------


@dataclass
class Camera:
    """2D camera with uniform zoom and pan offset.

    Image-space (world) px → screen px:
        screen = world * zoom + offset
    """
    zoom: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        return (wx * self.zoom + self.offset_x, wy * self.zoom + self.offset_y)

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return ((sx - self.offset_x) / self.zoom, (sy - self.offset_y) / self.zoom)

    def zoom_at(self, sx: float, sy: float, factor: float) -> None:
        wx, wy = self.screen_to_world(sx, sy)
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom * factor))
        if new_zoom == self.zoom:
            return
        self.zoom = new_zoom
        nsx, nsy = self.world_to_screen(wx, wy)
        self.offset_x += sx - nsx
        self.offset_y += sy - nsy

    def pan(self, dx: float, dy: float) -> None:
        self.offset_x += dx
        self.offset_y += dy


# --- Asset loading -----------------------------------------------------------


def _resolve_image_path(level: Level) -> Path:
    """Resolve the level's map_image path against the project root."""
    p = Path(level.map_image)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _load_level_image(level: Level) -> pygame.Surface:
    """Load the PNG for `level`. Returns the raw pygame Surface (callers
    can call .convert() once a display mode is set; we don't here so the
    loader works headless for tests)."""
    img_path = _resolve_image_path(level)
    if not img_path.exists():
        raise FileNotFoundError(f"Map image missing: {img_path}")
    image = pygame.image.load(str(img_path))
    if pygame.display.get_surface() is not None:
        image = image.convert()
    return image


def build_revealed_mask(
    level: Level,
    image_size: tuple[int, int],
    revealed_room_ids: set[str],
) -> np.ndarray:
    """Boolean (H, W) mask: True wherever *any* revealed room's region
    covers the pixel. Overlap is handled correctly because we OR each
    room's pixels into the same mask — pixels under multiple rooms stay
    True as long as at least one of them is revealed."""
    w, h = image_size
    canvas = Image.new("1", (w, h), 0)  # 1-bit binary image
    draw = ImageDraw.Draw(canvas)
    for room in level.rooms:
        if room.id not in revealed_room_ids:
            continue
        region = room.image_region
        if region is None:
            continue
        _draw_region_to(draw, region, fill=1)
    return np.asarray(canvas, dtype=bool)


def _draw_region_to(draw: ImageDraw.ImageDraw, region: ImageRegion, *, fill) -> None:
    if region.kind == "rect" and region.rect is not None:
        x, y, w, h = region.rect
        draw.rectangle([x, y, x + w - 1, y + h - 1], fill=fill)
    elif region.kind == "polygon" and region.points:
        draw.polygon([(int(p[0]), int(p[1])) for p in region.points], fill=fill)


def _region_contains_point(region: ImageRegion, x: float, y: float) -> bool:
    """Geometric hit-test for a rect or polygon region in image-pixel coords."""
    if region.kind == "rect" and region.rect is not None:
        rx, ry, rw, rh = region.rect
        return rx <= x < rx + rw and ry <= y < ry + rh
    if region.kind == "polygon" and region.points:
        return _point_in_polygon(x, y, region.points)
    return False


def _point_in_polygon(x: float, y: float,
                      points: tuple[tuple[int, int], ...]) -> bool:
    """Standard ray-casting: count edge crossings along a horizontal ray
    from (x, y) to +∞. Odd → inside, even → outside."""
    inside = False
    n = len(points)
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            else:
                xinters = p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _topmost_room_at(level: Level, x: float, y: float) -> Room | None:
    """Most recently drawn room whose region contains (x, y) in image
    pixels — newer rooms (later in the list) win overlap."""
    for room in reversed(level.rooms):
        if room.image_region is None:
            continue
        if _region_contains_point(room.image_region, x, y):
            return room
    return None


# --- Fog mask ---------------------------------------------------------------


def _build_fog_alpha(
    revealed_mask: np.ndarray,
    *,
    base_alpha: int,
) -> np.ndarray:
    """Build an (H, W) uint8 alpha channel from a boolean revealed mask:
    0 where revealed (no fog), `base_alpha` where not (fogged)."""
    return np.where(revealed_mask, np.uint8(0), np.uint8(base_alpha))


def _alpha_to_fog_surface(alpha: np.ndarray, color: tuple[int, int, int]) -> pygame.Surface:
    """Build an SRCALPHA pygame Surface from an (H, W) alpha array filled
    with `color`."""
    H, W = alpha.shape
    rgba = np.empty((H, W, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = alpha
    # pygame surfarray expects (W, H, 4) for make_surface, but we'll use
    # frombuffer which expects rows-first (H, W, 4) bytes.
    surf = pygame.image.frombuffer(rgba.tobytes(), (W, H), "RGBA")
    if pygame.display.get_surface() is not None:
        surf = surf.convert_alpha()
    return surf


# --- MapView -----------------------------------------------------------------


class MapView:
    """The DM editor's runtime model for one level."""

    def __init__(
        self,
        session: Session,
        *,
        dungeon_path: Path | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.session = session
        self.dungeon: Dungeon = session.dungeon
        self._dungeon_path = dungeon_path  # used by annotation mode to persist
        self._on_change = on_change

        self.camera = Camera()
        self.image: pygame.Surface | None = None
        # Pre-baked fog Surfaces, invalidated on any state or annotation change.
        self._fog_dm: pygame.Surface | None = None
        self._fog_player: pygame.Surface | None = None

        # Drag/pan/hover state for play mode.
        self._mouse_down_pos: tuple[int, int] | None = None
        self._panning = False
        self._pan_anchor: tuple[int, int] | None = None

        # Annotation mode state.
        self._annot_mode = False
        self._annot_tool = "rect"  # "rect" | "polygon"
        self._annot_drag_start_world: tuple[float, float] | None = None
        self._annot_drag_current_world: tuple[float, float] | None = None
        self._annot_polygon_points: list[tuple[float, float]] = []
        self._annot_selected_room_id: str | None = None
        # Undo stack — tuples of (action_kind, level_number, room_snapshot,
        # original_index). action_kind ∈ {"add", "delete"}. We snapshot the
        # full Room (so undoing a delete restores name/state/tags/region).
        self._undo_stack: list[tuple[str, int, Room, int]] = []

        # Track the dungeon JSON's mtime so we can detect external edits
        # (the browser editor server writes the same file). _last_self_mtime
        # is updated after every self-write so we don't react to our own
        # dumps; _last_poll_time throttles the os.stat call to ~1 Hz.
        self._last_self_mtime: float = 0.0
        self._last_poll_time: float = 0.0
        if dungeon_path is not None and Path(dungeon_path).exists():
            self._last_self_mtime = Path(dungeon_path).stat().st_mtime

        # Options menu (toggled with O). Hit-test rects are recomputed each
        # frame in _draw_options_menu so resizing the window stays in sync.
        self._options_open: bool = False
        self._options_button_rects: list[tuple[pygame.Rect,
                                               Callable[[], None],
                                               bool]] = []
        # Filled in by set_menu_actions(); see run().
        self._action_open_editor: Callable[[], None] | None = None
        self._action_open_player: Callable[[], None] | None = None
        self._action_ascend: Callable[[], None] | None = None
        self._action_descend: Callable[[], None] | None = None
        self._action_quit: Callable[[], None] | None = None

        self.help_font = pygame.font.SysFont("monospace,courier", 12)
        self.title_font = pygame.font.SysFont("georgia,serif", 18, bold=True)

        self._load_current_level()

    # -- Asset / fog management ----------------------------------------------

    @property
    def level(self) -> Level:
        return self.dungeon.current

    def _load_current_level(self) -> None:
        self.image = _load_level_image(self.level)
        self._invalidate_fog()
        self._fit_to_window()

    def _fit_to_window(self) -> None:
        """Scale & center the image so the whole map is visible in the
        current display surface."""
        surf = pygame.display.get_surface()
        if surf is None or self.image is None:
            return
        sw, sh = surf.get_size()
        iw, ih = self.image.get_width(), self.image.get_height()
        scale = min(sw / iw, sh / ih) * 0.95
        scale = max(ZOOM_MIN, min(ZOOM_MAX, scale))
        self.camera.zoom = scale
        self.camera.offset_x = (sw - iw * scale) / 2
        self.camera.offset_y = (sh - ih * scale) / 2

    def _revealed_room_ids(self) -> set[str]:
        return {r.id for r in self.level.rooms if r.state in ("known", "cleared")}

    def _ensure_fog(self) -> tuple[pygame.Surface, pygame.Surface]:
        if self._fog_dm is None or self._fog_player is None:
            assert self.image is not None
            size = (self.image.get_width(), self.image.get_height())
            revealed_mask = build_revealed_mask(
                self.level, size, self._revealed_room_ids(),
            )
            dm_alpha = _build_fog_alpha(revealed_mask, base_alpha=FOG_ALPHA_DM)
            player_alpha = _build_fog_alpha(revealed_mask, base_alpha=FOG_ALPHA_PLAYER)
            self._fog_dm = _alpha_to_fog_surface(dm_alpha, FOG_COLOR)
            self._fog_player = _alpha_to_fog_surface(player_alpha, FOG_COLOR)
        return self._fog_dm, self._fog_player

    def _invalidate_fog(self) -> None:
        self._fog_dm = None
        self._fog_player = None

    # -- Hit-testing ---------------------------------------------------------

    def _room_at_screen(self, screen_pos: tuple[int, int]) -> Room | None:
        """Topmost (most recently drawn) room whose region contains the
        cursor. Iterates rooms in reverse so newer regions win overlap."""
        if self.image is None:
            return None
        wx, wy = self.camera.screen_to_world(*screen_pos)
        return _topmost_room_at(self.level, wx, wy)

    def _is_meaningful_drag(self, pos: tuple[int, int]) -> bool:
        if self._mouse_down_pos is None:
            return False
        dx = pos[0] - self._mouse_down_pos[0]
        dy = pos[1] - self._mouse_down_pos[1]
        return (dx * dx + dy * dy) > DRAG_THRESHOLD_PX ** 2

    # -- State cycling -------------------------------------------------------

    def cycle_room_state(self, room_id: str) -> str:
        room = self.level.rooms_by_id[room_id]
        idx = STATE_CYCLE.index(room.state) if room.state in STATE_CYCLE else 0
        new_state = STATE_CYCLE[(idx + 1) % len(STATE_CYCLE)]
        self.session.update_room_state(room_id, new_state)
        self._invalidate_fog()
        if self._on_change is not None:
            self._on_change()
        return new_state

    # -- Level switching -----------------------------------------------------

    def switch_to_current_level(self) -> None:
        """Re-bind to the level the session currently points at — used after
        session.switch_level(...)."""
        self._load_current_level()
        if self._on_change is not None:
            self._on_change()

    # -- Event loop ----------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        # Options menu intercepts mouse clicks while open so the underlying
        # map doesn't receive them (otherwise clicking a menu button would
        # also try to reveal a room behind it).
        if self._options_open and event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self._handle_options_click(event.pos):
                    return
        if self._annot_mode:
            self._handle_annotation_event(event)
            return
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self._mouse_down_pos = event.pos
            elif event.button == 2:
                self._panning = True
                self._pan_anchor = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if self._mouse_down_pos is not None and not self._is_meaningful_drag(event.pos):
                    room = self._room_at_screen(event.pos)
                    if room is not None:
                        self.cycle_room_state(room.id)
                self._mouse_down_pos = None
            elif event.button == 2:
                self._panning = False
                self._pan_anchor = None
        elif event.type == pygame.MOUSEMOTION:
            if self._panning and self._pan_anchor is not None:
                dx = event.pos[0] - self._pan_anchor[0]
                dy = event.pos[1] - self._pan_anchor[1]
                self.camera.pan(dx, dy)
                self._pan_anchor = event.pos
        elif event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            factor = ZOOM_STEP if event.y > 0 else 1 / ZOOM_STEP
            self.camera.zoom_at(mx, my, factor)
        elif event.type == pygame.VIDEORESIZE:
            self._fit_to_window()

    # -- Drawing -------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        # Cheap once-per-second poll for external edits to the dungeon JSON
        # (the browser editor server writes via dungeon.dump). When detected
        # we merge metadata fields back onto the in-memory rooms without
        # disturbing reveal state or annotated regions.
        self._poll_dungeon_mtime()

        surface.fill(BG_OUTSIDE)
        if self.image is None:
            return
        # In annotation mode show the bare image (no fog) so the DM can see
        # what's there to outline.
        if self._annot_mode:
            self._draw_image(surface, fog=None)
            self._draw_annotation_overlay(surface)
        else:
            self._draw_image(surface, fog=self._ensure_fog()[0])
            self._draw_room_markers(surface)
            self._draw_party_marker(surface)
        self._draw_chrome(surface)
        # Options panel renders last so it sits above everything else.
        self._draw_options_menu(surface)

    def _draw_image(self, surface: pygame.Surface, fog: pygame.Surface | None) -> None:
        """Composite image (+ optional fog) at the camera's zoom."""
        zoom = self.camera.zoom
        iw, ih = self.image.get_width(), self.image.get_height()
        sw = max(1, int(iw * zoom))
        sh = max(1, int(ih * zoom))
        if fog is None:
            composite = self.image
        else:
            composite = self.image.copy()
            composite.blit(fog, (0, 0))
        scaled = pygame.transform.smoothscale(composite, (sw, sh))
        surface.blit(scaled, (int(self.camera.offset_x), int(self.camera.offset_y)))

    def _draw_room_markers(self, surface: pygame.Surface) -> None:
        """Per-room state dot at each annotated region's centroid."""
        zoom = self.camera.zoom
        for r in self.level.rooms:
            wx, wy = self._room_centroid_world(r)
            if wx is None:
                continue
            sx, sy = self.camera.world_to_screen(wx, wy)
            color = STATE_COLORS.get(r.state, STATE_COLORS["unexplored"])
            radius = max(4, int(MARKER_RADIUS * zoom))
            ring = max(1, int(MARKER_RING * zoom))
            pygame.draw.circle(surface, color[:3], (int(sx), int(sy)), radius)
            border = ALERT_RED if "encounter" in r.tags else (0, 0, 0)
            pygame.draw.circle(surface, border, (int(sx), int(sy)), radius, ring)

    def _room_centroid_world(self, room: Room) -> tuple[int, int] | tuple[None, None]:
        """Centroid of the room's image_region. Returns (None, None) if the
        room isn't annotated yet."""
        if room.image_region is None:
            return (None, None)
        cx, cy = room.image_region.centroid()
        return (cx, cy)

    def _draw_annotation_overlay(self, surface: pygame.Surface) -> None:
        """In annotation mode: outline every existing region (red for the
        selected one), and preview the in-progress shape under the cursor."""
        for r in self.level.rooms:
            if r.image_region is None:
                continue
            color = (210, 60, 60) if r.id == self._annot_selected_room_id else (60, 130, 220)
            self._draw_region_outline(surface, r.image_region, color, width=3)
            # Label in image-space at centroid.
            cx, cy = r.image_region.centroid()
            sx, sy = self.camera.world_to_screen(cx, cy)
            text = self.help_font.render(r.id, True, (10, 10, 10))
            plate = pygame.Surface(
                (text.get_width() + 6, text.get_height() + 4), pygame.SRCALPHA,
            )
            plate.fill((244, 228, 193, 220))
            surface.blit(plate, (int(sx - plate.get_width() / 2),
                                 int(sy - plate.get_height() / 2)))
            surface.blit(text, (int(sx - text.get_width() / 2),
                                int(sy - text.get_height() / 2)))

        # In-progress rectangle.
        if (self._annot_tool == "rect" and
                self._annot_drag_start_world is not None and
                self._annot_drag_current_world is not None):
            x0, y0 = self._annot_drag_start_world
            x1, y1 = self._annot_drag_current_world
            sx0, sy0 = self.camera.world_to_screen(min(x0, x1), min(y0, y1))
            sx1, sy1 = self.camera.world_to_screen(max(x0, x1), max(y0, y1))
            rect = pygame.Rect(int(sx0), int(sy0),
                               int(max(1, sx1 - sx0)), int(max(1, sy1 - sy0)))
            pygame.draw.rect(surface, (60, 200, 60), rect, 3)

        # In-progress polygon.
        if self._annot_tool == "polygon" and self._annot_polygon_points:
            screen_pts = [self.camera.world_to_screen(x, y)
                          for x, y in self._annot_polygon_points]
            for i, (sx, sy) in enumerate(screen_pts):
                pygame.draw.circle(surface, (60, 200, 60), (int(sx), int(sy)), 5)
                if i > 0:
                    p = screen_pts[i - 1]
                    pygame.draw.line(surface, (60, 200, 60),
                                     (int(p[0]), int(p[1])), (int(sx), int(sy)), 2)

    def _draw_region_outline(self, surface: pygame.Surface, region: ImageRegion,
                             color: tuple[int, int, int], *, width: int) -> None:
        if region.kind == "rect" and region.rect is not None:
            x, y, w, h = region.rect
            sx0, sy0 = self.camera.world_to_screen(x, y)
            sx1, sy1 = self.camera.world_to_screen(x + w, y + h)
            rect = pygame.Rect(int(sx0), int(sy0),
                               int(sx1 - sx0), int(sy1 - sy0))
            pygame.draw.rect(surface, color, rect, width)
        elif region.kind == "polygon" and region.points:
            pts = [self.camera.world_to_screen(p[0], p[1]) for p in region.points]
            pygame.draw.polygon(surface, color,
                                [(int(p[0]), int(p[1])) for p in pts], width)

    def _draw_party_marker(self, surface: pygame.Surface) -> None:
        try:
            pid = self.session.get_party_position()
        except LookupError:
            return
        room = self.level.rooms_by_id.get(pid)
        if room is None:
            return
        wx, wy = self._room_centroid_world(room)
        if wx is None:
            return
        sx, sy = self.camera.world_to_screen(wx, wy)
        radius = max(6, int((MARKER_RADIUS + 4) * self.camera.zoom))
        pygame.draw.circle(surface, PARTY_DOT, (int(sx), int(sy)), radius)
        pygame.draw.circle(surface, PARTY_DOT_RING, (int(sx), int(sy)), radius, 3)

    def _draw_chrome(self, surface: pygame.Surface) -> None:
        lv = self.level
        ascend = "⌘↑ ascend" if self.session.can_ascend() else "—"
        descend = "⌘↓ descend" if self.session.can_descend() else "—"
        mode = f"  [ANNOTATION · {self._annot_tool}]" if self._annot_mode else ""
        title = f"L{lv.level_number}  {lv.display_name}    [{ascend} | {descend}]{mode}"
        text = self.title_font.render(title, True, LABEL_INK)
        pad = 4
        plate = pygame.Surface(
            (text.get_width() + pad * 2, text.get_height() + pad * 2),
            pygame.SRCALPHA,
        )
        plate.fill((244, 228, 193, 220))
        surface.blit(plate, (6, 6))
        surface.blit(text, (6 + pad, 6 + pad))

        if self._annot_mode:
            msg = ("A: exit · drag: rect · P: polygon · click+Del: remove · "
                   "Enter: close poly · ⌘Z: undo · ⌘+/⌘-: zoom · ⌘0: fit · Esc: cancel/exit")
        else:
            msg = ("O: options menu · click room: cycle state · middle-drag: pan · "
                   "scroll or ⌘+/⌘-: zoom · ⌘0: fit · Esc: quit")
        help_text = self.help_font.render(msg, True, HELP_INK)
        surface.blit(help_text, (8, surface.get_height() - help_text.get_height() - 6))

    # -- Annotation mode -----------------------------------------------------

    def toggle_annotation_mode(self) -> None:
        """Flip between play mode (click reveals) and annotation mode (draw
        rectangles/polygons to define rooms)."""
        self._annot_mode = not self._annot_mode
        self._reset_annotation_state()

    def _reset_annotation_state(self) -> None:
        self._annot_drag_start_world = None
        self._annot_drag_current_world = None
        self._annot_polygon_points = []
        self._annot_selected_room_id = None

    def _handle_annotation_event(self, event: pygame.event.Event) -> None:
        # Pan + zoom still work in annotation mode (middle-drag, scroll).
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 2:
                self._panning = True
                self._pan_anchor = event.pos
                return
            if event.button == 1:
                self._annot_mouse_down(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self._annot_mouse_up(event.pos)
            elif event.button == 2:
                self._panning = False
                self._pan_anchor = None
        elif event.type == pygame.MOUSEMOTION:
            if self._panning and self._pan_anchor is not None:
                dx = event.pos[0] - self._pan_anchor[0]
                dy = event.pos[1] - self._pan_anchor[1]
                self.camera.pan(dx, dy)
                self._pan_anchor = event.pos
            elif self._annot_drag_start_world is not None:
                self._annot_drag_current_world = self.camera.screen_to_world(*event.pos)
        elif event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            factor = ZOOM_STEP if event.y > 0 else 1 / ZOOM_STEP
            self.camera.zoom_at(mx, my, factor)

    def _annot_mouse_down(self, pos: tuple[int, int]) -> None:
        if self._annot_tool == "rect":
            # Click on an existing region selects it (Delete then removes it).
            room = self._room_at_screen(pos)
            if room is not None:
                self._annot_selected_room_id = room.id
                return
            self._annot_selected_room_id = None
            self._annot_drag_start_world = self.camera.screen_to_world(*pos)
            self._annot_drag_current_world = self._annot_drag_start_world

    def _annot_mouse_up(self, pos: tuple[int, int]) -> None:
        if self._annot_tool == "rect":
            if self._annot_drag_start_world is not None:
                self._commit_rect(pos)
        elif self._annot_tool == "polygon":
            # Each click in polygon mode appends a vertex. Press Enter to
            # close the shape; Esc to cancel.
            world = self.camera.screen_to_world(*pos)
            self._annot_polygon_points.append(world)

    def _commit_rect(self, pos: tuple[int, int]) -> None:
        if self._annot_drag_start_world is None:
            return
        end = self.camera.screen_to_world(*pos)
        x0, y0 = self._annot_drag_start_world
        x1, y1 = end
        # Skip trivial drags (just selection clicks that didn't move).
        if abs(x1 - x0) < 6 or abs(y1 - y0) < 6:
            self._annot_drag_start_world = None
            self._annot_drag_current_world = None
            return
        x = int(min(x0, x1))
        y = int(min(y0, y1))
        w = int(abs(x1 - x0))
        h = int(abs(y1 - y0))
        self._add_room_with_region(ImageRegion(kind="rect", rect=(x, y, w, h)))
        self._annot_drag_start_world = None
        self._annot_drag_current_world = None

    def _commit_polygon(self) -> None:
        if len(self._annot_polygon_points) < 3:
            self._annot_polygon_points = []
            return
        pts = tuple((int(p[0]), int(p[1])) for p in self._annot_polygon_points)
        self._add_room_with_region(ImageRegion(kind="polygon", points=pts))
        self._annot_polygon_points = []

    def _next_room_id(self) -> str:
        """Pick the next R## id not in use on the current level."""
        used = {r.id for r in self.level.rooms}
        for n in range(1, 1000):
            rid = f"R{n:02d}"
            if rid not in used:
                return rid
        raise RuntimeError("ran out of room ids")

    def _add_room_with_region(self, region: ImageRegion) -> None:
        room = self._insert_new_room(region)
        # Push to undo stack: undoing an add = deleting this room.
        self._push_undo("add", self.level.level_number, room,
                        len(self.level.rooms) - 1)
        self._after_annotation_change()

    def _insert_new_room(self, region: ImageRegion) -> Room:
        """Common helper: create the Room, append, write SQLite rows.
        Used by both initial add and undo-of-delete."""
        is_first_on_level = len(self.level.rooms) == 0
        rid = self._next_room_id()
        new_room = Room(
            id=rid,
            name=rid,
            state="unexplored",
            tags=("empty",),
            image_region=region,
        )
        self.level.rooms = self.level.rooms + (new_room,)
        self.level.rooms_by_id[rid] = new_room
        with self.session.conn:
            self.session.conn.execute(
                "INSERT OR IGNORE INTO room_state "
                "(session_id, level_number, room_id, state, notes) "
                "VALUES (?, ?, ?, 'unexplored', '')",
                (self.session.session_id, self.level.level_number, rid),
            )
            if is_first_on_level:
                self.session.conn.execute(
                    "INSERT OR REPLACE INTO party_position "
                    "(session_id, level_number, room_id) VALUES (?, ?, ?)",
                    (self.session.session_id, self.level.level_number, rid),
                )
        return new_room

    def _delete_selected_room(self) -> None:
        rid = self._annot_selected_room_id
        if rid is None:
            return
        # Snapshot for undo before mutating.
        idx = next((i for i, r in enumerate(self.level.rooms) if r.id == rid), None)
        if idx is None:
            return
        snapshot = self.level.rooms[idx]
        self._remove_room(rid)
        self._push_undo("delete", self.level.level_number, snapshot, idx)
        self._annot_selected_room_id = None
        self._after_annotation_change()

    def _remove_room(self, rid: str) -> None:
        self.level.rooms = tuple(r for r in self.level.rooms if r.id != rid)
        self.level.rooms_by_id.pop(rid, None)
        self.session.conn.execute(
            "DELETE FROM room_state WHERE session_id = ? AND level_number = ? AND room_id = ?",
            (self.session.session_id, self.level.level_number, rid),
        )
        self.session.conn.commit()

    def _restore_room_at(self, level_number: int, room: Room, index: int) -> None:
        """Re-insert a previously-deleted room at the given index, restoring
        its full state. Used by undo."""
        # Switch to that level if we're elsewhere — undo always operates on
        # the level the action originally happened on.
        if self.dungeon.current_level != level_number:
            self.session.set_current_level(level_number)
            self.switch_to_current_level()
        rooms = list(self.level.rooms)
        rooms.insert(min(index, len(rooms)), room)
        self.level.rooms = tuple(rooms)
        self.level.rooms_by_id[room.id] = room
        with self.session.conn:
            self.session.conn.execute(
                "INSERT OR REPLACE INTO room_state "
                "(session_id, level_number, room_id, state, notes) "
                "VALUES (?, ?, ?, ?, ?)",
                (self.session.session_id, level_number, room.id,
                 room.state, room.notes),
            )

    def _push_undo(self, kind: str, level_number: int,
                   room: Room, index: int) -> None:
        self._undo_stack.append((kind, level_number, room, index))
        if len(self._undo_stack) > MAX_UNDO_DEPTH:
            self._undo_stack.pop(0)

    def undo_last_annotation(self) -> bool:
        """Reverse the most recent annotation action. Returns True on
        success, False if the stack was empty."""
        if not self._undo_stack:
            return False
        kind, level_number, room, index = self._undo_stack.pop()
        if kind == "add":
            # Undoing an add = delete the room.
            if self.dungeon.current_level != level_number:
                self.session.set_current_level(level_number)
                self.switch_to_current_level()
            self._remove_room(room.id)
        elif kind == "delete":
            self._restore_room_at(level_number, room, index)
        self._after_annotation_change()
        return True

    def _after_annotation_change(self) -> None:
        """Bookkeeping shared between add, delete, undo: rebuild fog,
        re-save the JSON, fire on_change so the browser snapshots refresh."""
        self._invalidate_fog()
        self._save_dungeon()
        if self._on_change is not None:
            self._on_change()

    def _save_dungeon(self) -> None:
        """Write the dungeon back to its source JSON so annotations persist
        across sessions on the same dungeon file."""
        if self._dungeon_path is None:
            return
        dungeon_mod.dump(self.dungeon, self._dungeon_path)
        # Record our own write so the mtime poller doesn't react to it.
        try:
            self._last_self_mtime = Path(self._dungeon_path).stat().st_mtime
        except OSError:
            pass

    # -- Options menu --------------------------------------------------------

    def set_menu_actions(
        self,
        *,
        open_editor: Callable[[], None] | None = None,
        open_player: Callable[[], None] | None = None,
        ascend: Callable[[], None] | None = None,
        descend: Callable[[], None] | None = None,
        quit_app: Callable[[], None] | None = None,
    ) -> None:
        """Wire the actions the options menu (and matching keyboard
        shortcuts) can invoke. Called from run() so callbacks have closure
        over session + view + the running flag."""
        self._action_open_editor = open_editor
        self._action_open_player = open_player
        self._action_ascend = ascend
        self._action_descend = descend
        self._action_quit = quit_app

    def toggle_options_menu(self) -> None:
        self._options_open = not self._options_open

    def _options_items(self) -> list[tuple[str, str, Callable[[], None] | None, bool]]:
        """Return the menu rows: (label, key_hint, action, enabled)."""
        return [
            ("Annotation mode", "A",
             self.toggle_annotation_mode, True),
            ("Open Room Editor", "E",
             self._action_open_editor,
             self._action_open_editor is not None),
            ("Open Player Map", "M",
             self._action_open_player,
             self._action_open_player is not None),
            ("Ascend Level", "⌘↑",
             self._action_ascend,
             self.session.can_ascend() and self._action_ascend is not None),
            ("Descend Level", "⌘↓",
             self._action_descend,
             self.session.can_descend() and self._action_descend is not None),
            ("Quit", "Esc",
             self._action_quit,
             self._action_quit is not None),
        ]

    def _draw_options_menu(self, surface: pygame.Surface) -> None:
        """Draw the modal options panel. Recomputes button rects so the
        click handler stays accurate after window resizes."""
        self._options_button_rects = []
        if not self._options_open:
            return

        # Dim backdrop so the map fades behind the menu.
        backdrop = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        backdrop.fill((0, 0, 0, 130))
        surface.blit(backdrop, (0, 0))

        items = self._options_items()
        panel_w = 380
        btn_h = 46
        btn_gap = 8
        title_h = 56
        panel_pad = 22
        panel_h = title_h + panel_pad + len(items) * (btn_h + btn_gap) + panel_pad
        panel_x = (surface.get_width() - panel_w) // 2
        panel_y = max(40, (surface.get_height() - panel_h) // 2)

        # Panel background.
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        panel_bg = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel_bg.fill((244, 228, 193, 250))
        surface.blit(panel_bg, panel_rect.topleft)
        pygame.draw.rect(surface, (26, 26, 26), panel_rect, 3,
                         border_radius=6)

        # Title bar.
        title = self.title_font.render("OPTIONS", True, (26, 26, 26))
        surface.blit(title, (panel_x + (panel_w - title.get_width()) // 2,
                             panel_y + 16))

        # Buttons.
        btn_x = panel_x + 20
        btn_w = panel_w - 40
        btn_y = panel_y + title_h + panel_pad
        for label, key_hint, action, enabled in items:
            rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
            if enabled:
                fill = (252, 252, 250)
                edge = (26, 26, 26)
                ink = (26, 26, 26)
            else:
                fill = (220, 215, 200)
                edge = (130, 110, 80)
                ink = (130, 110, 80)
            pygame.draw.rect(surface, fill, rect, border_radius=4)
            pygame.draw.rect(surface, edge, rect, 2, border_radius=4)
            label_surf = self.title_font.render(label, True, ink)
            key_surf = self.help_font.render(key_hint, True, ink)
            surface.blit(label_surf, (rect.x + 16,
                                      rect.y + (btn_h - label_surf.get_height()) // 2))
            surface.blit(key_surf, (rect.right - 16 - key_surf.get_width(),
                                    rect.y + (btn_h - key_surf.get_height()) // 2))
            self._options_button_rects.append((rect, action, enabled))
            btn_y += btn_h + btn_gap

    def _handle_options_click(self, pos: tuple[int, int]) -> bool:
        """Return True if the click was consumed by the options menu."""
        if not self._options_open:
            return False
        for rect, action, enabled in self._options_button_rects:
            if rect.collidepoint(pos):
                if enabled and action is not None:
                    self._options_open = False
                    action()
                return True
        # Click outside any button (and the menu is open) → close.
        self._options_open = False
        return True

    # -- External-edit detection (mtime poll → metadata merge) ---------------

    def _poll_dungeon_mtime(self) -> None:
        """If the dungeon JSON has been modified externally since our last
        write, re-parse it and merge metadata fields onto the in-memory
        Rooms. Cheap (≤ 1 Hz; an os.stat call)."""
        if self._dungeon_path is None:
            return
        now = time.monotonic()
        if now - self._last_poll_time < MTIME_POLL_SECONDS:
            return
        self._last_poll_time = now
        try:
            mtime = os.stat(self._dungeon_path).st_mtime
        except OSError:
            return
        if mtime <= self._last_self_mtime:
            return
        self._reload_dungeon_metadata()
        self._last_self_mtime = mtime

    def _reload_dungeon_metadata(self) -> None:
        """Re-parse the JSON and copy *metadata* fields onto each in-memory
        Room. Preserves `state` (runtime fog reveal) and `image_region`
        (annotated geometry) — those flow through pygame, not the editor."""
        try:
            fresh = dungeon_mod.load(self._dungeon_path)
        except (FileNotFoundError, dungeon_mod.DungeonValidationError):
            return
        for fresh_lv in fresh.levels:
            live_lv = self.dungeon.levels_by_number.get(fresh_lv.level_number)
            if live_lv is None:
                continue
            for fresh_r in fresh_lv.rooms:
                live_r = live_lv.rooms_by_id.get(fresh_r.id)
                if live_r is None:
                    continue
                live_r.name = fresh_r.name
                live_r.tags = fresh_r.tags
                live_r.reaction_required = fresh_r.reaction_required
                live_r.notes = fresh_r.notes
                live_r.encounter_ref = fresh_r.encounter_ref
                live_r.treasure_tier = fresh_r.treasure_tier
                live_r.box_text = fresh_r.box_text
                live_r.encounter_text = fresh_r.encounter_text
                live_r.treasure_text = fresh_r.treasure_text
                # Deliberately NOT copied: state, image_region.
        self._invalidate_fog()
        if self._on_change is not None:
            self._on_change()

    # -- Snapshots (for browser auto-refresh) --------------------------------

    def snapshot_to_disk(self) -> None:
        """Save dm_map.png (DM fog) and player_map.png (player fog) to
        render_output/. Called by main.py whenever state changes."""
        if self.image is None:
            return
        RENDER_OUTPUT.mkdir(parents=True, exist_ok=True)
        fog_dm, fog_player = self._ensure_fog()
        for path, fog in ((DM_PNG, fog_dm), (PLAYER_PNG, fog_player)):
            composite = self.image.copy()
            composite.blit(fog, (0, 0))
            pygame.image.save(composite, str(path))


# --- Run loop ---------------------------------------------------------------


def run(
    session: Session,
    *,
    dungeon_path: Path | None = None,
    window_size: tuple[int, int] = (1280, 800),
    on_change: Callable[[], None] | None = None,
    on_open_browser: Callable[[], None] | None = None,
    on_open_editor: Callable[[], None] | None = None,
    on_open_player: Callable[[], None] | None = None,
) -> None:
    """Run the pygame event loop. on_change fires after any state change.
    The on_open_* hooks are individual tab openers used by the options
    menu; on_open_browser is the legacy "open all tabs" shortcut for V."""
    pygame.init()
    pygame.font.init()
    surface = pygame.display.set_mode(window_size, pygame.RESIZABLE)
    pygame.display.set_caption(f"OSR Dungeon — Editor — {session.dungeon.name}")
    clock = pygame.time.Clock()

    def _on_change() -> None:
        view.snapshot_to_disk()
        if on_change is not None:
            on_change()

    view = MapView(session, dungeon_path=dungeon_path, on_change=_on_change)
    view.snapshot_to_disk()

    # Closures for the options-menu actions. We keep them here so they can
    # close over `running` (for quit) and the level-switch sequence (which
    # touches both session and view).
    running = True

    def _ascend() -> None:
        if session.can_ascend():
            session.switch_level(-1)
            view.switch_to_current_level()

    def _descend() -> None:
        if session.can_descend():
            session.switch_level(+1)
            view.switch_to_current_level()

    def _quit() -> None:
        nonlocal running
        running = False

    view.set_menu_actions(
        open_editor=on_open_editor,
        open_player=on_open_player,
        ascend=_ascend,
        descend=_descend,
        quit_app=_quit,
    )

    # On macOS the conventional shortcut modifier is Cmd, which pygame
    # reports as KMOD_META; on Linux/Windows it's Ctrl (KMOD_CTRL). Accept
    # either so the same shortcut sheet works cross-platform.
    cmd_mask = pygame.KMOD_CTRL | pygame.KMOD_META

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                cmd = bool(event.mod & cmd_mask)
                if event.key == pygame.K_ESCAPE:
                    if view._options_open:
                        view._options_open = False
                    elif view._annot_mode and (
                        view._annot_drag_start_world is not None
                        or view._annot_polygon_points
                    ):
                        view._reset_annotation_state()
                    elif view._annot_mode:
                        view.toggle_annotation_mode()
                    else:
                        running = False
                elif event.key == pygame.K_v and on_open_browser is not None:
                    on_open_browser()
                elif event.key == pygame.K_o and not cmd and not view._annot_mode:
                    view.toggle_options_menu()
                elif event.key == pygame.K_e and not cmd and not view._annot_mode:
                    if on_open_editor is not None:
                        on_open_editor()
                elif event.key == pygame.K_m and not cmd and not view._annot_mode:
                    if on_open_player is not None:
                        on_open_player()
                elif event.key == pygame.K_a and not cmd:
                    view.toggle_annotation_mode()
                elif cmd and event.key == pygame.K_z:
                    view.undo_last_annotation()
                elif cmd and event.key in (pygame.K_EQUALS, pygame.K_PLUS,
                                           pygame.K_KP_PLUS):
                    surface_size = surface.get_size()
                    cx, cy = surface_size[0] // 2, surface_size[1] // 2
                    view.camera.zoom_at(cx, cy, ZOOM_STEP)
                elif cmd and event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    surface_size = surface.get_size()
                    cx, cy = surface_size[0] // 2, surface_size[1] // 2
                    view.camera.zoom_at(cx, cy, 1 / ZOOM_STEP)
                elif cmd and event.key == pygame.K_0:
                    # Cmd+0 = fit map back to window (handy after zooming).
                    view._fit_to_window()
                elif view._annot_mode and event.key == pygame.K_p:
                    view._annot_tool = "polygon" if view._annot_tool == "rect" else "rect"
                    view._reset_annotation_state()
                elif view._annot_mode and event.key == pygame.K_RETURN:
                    if view._annot_tool == "polygon":
                        view._commit_polygon()
                elif view._annot_mode and event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                    view._delete_selected_room()
                elif cmd and event.key == pygame.K_UP:
                    _ascend()
                elif cmd and event.key == pygame.K_DOWN:
                    _descend()
                else:
                    view.handle_event(event)
            else:
                view.handle_event(event)
        view.draw(surface)
        pygame.display.flip()
        clock.tick(60)

    session.close()
    pygame.quit()
