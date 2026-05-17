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
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pygame
from PIL import Image, ImageDraw

import config
import dungeon as dungeon_mod
import journal as journal_mod
from dungeon import Dungeon, ImageRegion, Level, Room
from session import DungeonInfo, Session
from tracker import LightSource


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

# --- Bottom strip (always-visible turn / resources / actions) ---------------
STATUS_BAR_HEIGHT       = 22
ACTION_BUTTON_HEIGHT    = 30
ACTION_BUTTON_WIDTH     = 130
ACTION_BUTTON_GAP       = 6
STRIP_PAD_BOTTOM        = 30  # leave room for the existing help line below
STATUS_INK              = (40, 40, 40)
STATUS_PLATE            = (244, 228, 193, 230)
STATUS_PLATE_NOISY      = (224, 184, 120, 240)  # warm tan when Noisy is on
STATUS_LOW_INK          = (168, 32, 26)         # red — torch <= 2 turns

# Single-letter labels used in the resource summary so the strip stays compact.
SUPPLY_ABBREV = {
    "torches":        "T",
    "hooded_lantern": "L",
    "oil_flask":      "O",
    "ration":         "R",
    "water_gallon":   "W",
}
# Order of supplies in the strip; matches DEFAULT_SUPPLY_KINDS in session.py.
SUPPLY_DISPLAY_ORDER = ("torches", "hooded_lantern", "oil_flask",
                        "ration", "water_gallon")

STATE_CYCLE = ("unexplored", "known", "cleared")


@dataclass(frozen=True)
class ReloadRequest:
    """Signal from the renderer back to main.py: tear down the editor
    server + session and re-enter run() pointed at `folder`. If
    `do_full_reset` is True, run Session.full_reset(folder) before
    reopening so the dungeon is wiped to its level skeleton."""
    folder: Path
    do_full_reset: bool = False

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


def _resolve_image_path(level: Level,
                        dungeon_dir: Path | None = None) -> Path:
    """Resolve the level's map_image path.

    Resolution order:
      1. If `level.map_image` is absolute → use as-is.
      2. If `dungeon_dir` is given and `<dungeon_dir>/map_image` exists →
         use that (preferred — keeps a dungeon folder portable).
      3. Else fall back to `PROJECT_ROOT / map_image` (legacy layout
         where map_image was relative to the project root).
    """
    p = Path(level.map_image)
    if p.is_absolute():
        return p
    if dungeon_dir is not None:
        candidate = Path(dungeon_dir) / p
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / p


def _load_level_image(level: Level,
                      dungeon_dir: Path | None = None) -> pygame.Surface:
    """Load the PNG for `level`. Returns the raw pygame Surface (callers
    can call .convert() once a display mode is set; we don't here so the
    loader works headless for tests).

    `dungeon_dir` is the directory containing the dungeon JSON. When set,
    `level.map_image` is resolved relative to that folder first (the
    portable / per-dungeon layout)."""
    img_path = _resolve_image_path(level, dungeon_dir)
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
        dungeons_dir: Path | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.session = session
        self.dungeon: Dungeon = session.dungeon
        self._dungeon_path = dungeon_path  # used by annotation mode to persist
        # Root directory enumerated by the in-app dungeon picker. None disables
        # the "Open Different Dungeon…" menu row.
        self._dungeons_dir = Path(dungeons_dir) if dungeons_dir else None
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
        # Most recent room the cursor was over, in either play or
        # annotation mode. Used for keyboard parity: pressing Enter
        # cycles the hovered room's state and I opens its info modal,
        # so the DM doesn't have to mouse-target every action.
        self._hovered_room_id: str | None = None

        # Annotation mode state.
        self._annot_mode = False
        self._annot_tool = "rect"  # "rect" | "polygon"
        self._annot_drag_start_world: tuple[float, float] | None = None
        self._annot_drag_current_world: tuple[float, float] | None = None
        self._annot_polygon_points: list[tuple[float, float]] = []
        # The room currently under the cursor in annotation mode.
        # MOUSEMOTION updates this; rendering highlights it red so the
        # DM can see what they're about to delete; Delete/Backspace
        # acts on it. Replaces the older click-to-select model.
        self._annot_hovered_room_id: str | None = None
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
        # When True the options modal swaps its contents for the dungeon
        # picker (list of folders under self._dungeons_dir).
        self._picker_open: bool = False
        self._picker_rows: list[tuple[pygame.Rect, Path]] = []
        # Full-reset confirmation: when open, the modal shows a destructive
        # confirm dialog. Click rects are populated each draw.
        self._reset_confirm_open: bool = False
        self._reset_confirm_rects: list[tuple[pygame.Rect,
                                              Callable[[], None]]] = []
        # Reset-progress confirmation (lighter than full reset — keeps
        # annotations, just wipes runtime state). Separate flag so the
        # two confirm flows don't share state.
        self._progress_confirm_open: bool = False
        self._progress_confirm_rects: list[tuple[pygame.Rect,
                                                 Callable[[], None]]] = []
        # New-Dungeon modal: collects a free-text name + party level,
        # scaffolds dungeons/<slug>/dungeon.json, and reload-switches
        # the running app to the new folder via ReloadRequest. The
        # text field is the focused element while the modal is open;
        # all KEYDOWN events are routed to it (the run loop checks
        # _new_dungeon_open before any other key bindings).
        self._new_dungeon_open: bool = False
        self._new_dungeon_name: str = ""
        self._new_dungeon_party_level: int = 3
        # Which field has keyboard focus: 'name' | 'party_level'
        self._new_dungeon_focus: str = "name"
        self._new_dungeon_error: str = ""
        self._new_dungeon_rects: list[tuple[pygame.Rect,
                                            Callable[[], None]]] = []
        # Per-field click rects so Tab/click switches focus.
        self._new_dungeon_field_rects: dict[str, pygame.Rect] = {}
        # Set when the user asks to switch dungeons or full-reset. The run()
        # loop exits cleanly when this is non-None and returns it to main.py.
        self._pending_reload: ReloadRequest | None = None
        # Bottom-strip action buttons. Same recompute-each-frame pattern.
        self._action_button_rects: list[tuple[pygame.Rect,
                                              Callable[[], None],
                                              bool]] = []
        # Startup warning overlay (Phase 5): a one-shot dismissable panel
        # shown only when there's something the DM needs to see before
        # play (e.g. dungeon.json edited externally between sessions).
        # `_startup_warning_lines` is empty in the happy path so the
        # overlay never draws. Dismissed by any keypress or click; the
        # dismiss event is consumed so it doesn't double as a room click.
        self._startup_warning_lines: list[str] = []
        self._startup_warning_panel_rect: pygame.Rect | None = None

        # Room-info modal: right-click a room to inspect its JSON metadata
        # (box text, encounter, treasure, special, DM notes). Click-rects are
        # recomputed each draw so window resize stays in sync.
        self._room_info_open: bool = False
        self._room_info_room_id: str | None = None
        self._room_info_close_rect: pygame.Rect | None = None
        self._room_info_panel_rect: pygame.Rect | None = None
        # Scroll state: y-offset in pixels into the body (clamped to
        # [0, max_scroll]); max_scroll is updated each draw so resizing the
        # window or switching rooms with longer notes stays consistent.
        self._room_info_scroll_y: float = 0.0
        self._room_info_max_scroll: float = 0.0
        self._room_info_body_rect: pygame.Rect | None = None
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
        dungeon_dir = (Path(self._dungeon_path).parent
                       if self._dungeon_path is not None else None)
        self.image = _load_level_image(self.level, dungeon_dir)
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
        # Startup warning overlay intercepts the next input event of any
        # kind and dismisses itself. The event is consumed so a click
        # used to dismiss doesn't also reveal a room underneath.
        if self._startup_warning_lines:
            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                self._dismiss_startup_warnings()
                return
        # Options menu intercepts mouse clicks while open so the underlying
        # map doesn't receive them (otherwise clicking a menu button would
        # also try to reveal a room behind it).
        if self._options_open and event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self._handle_options_click(event.pos):
                    return
        # Room-info modal intercepts clicks the same way: a left-click on
        # the close button or outside the panel dismisses; clicks inside
        # the panel are swallowed so the map underneath doesn't see them.
        # Mouse wheel events scroll the modal body when the cursor is
        # over it, otherwise fall through to map zoom.
        if self._room_info_open:
            if event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if (self._room_info_panel_rect is not None
                        and self._room_info_panel_rect.collidepoint(mx, my)):
                    # 60 px per wheel notch — feels right with macOS trackpad
                    # natural-scroll on (event.y is positive going up).
                    self._room_info_scroll_y = max(
                        0.0,
                        min(self._room_info_max_scroll,
                            self._room_info_scroll_y - event.y * 60),
                    )
                    return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._handle_room_info_click(event.pos)
                    return
                if event.button == 3:
                    # Right-clicking again over a different room re-targets
                    # the modal; otherwise just swallow.
                    room = self._room_at_screen(event.pos)
                    if room is not None and room.id != self._room_info_room_id:
                        self._room_info_room_id = room.id
                        self._room_info_scroll_y = 0.0
                    return
        # Bottom-strip action buttons intercept too (otherwise clicking
        # "Advance Turn" would also click-through to a room behind it).
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and not self._annot_mode):
            if self._handle_action_button_click(event.pos):
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
            elif event.button == 3:
                # Right-click → open the room-info modal for the room
                # under the cursor (DM-only — pygame window is DM view).
                room = self._room_at_screen(event.pos)
                if room is not None:
                    self._room_info_open = True
                    self._room_info_room_id = room.id
                    self._room_info_scroll_y = 0.0
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if self._panning:
                    # Left-drag was promoted to a pan — swallow the click.
                    self._panning = False
                    self._pan_anchor = None
                elif self._mouse_down_pos is not None and not self._is_meaningful_drag(event.pos):
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
            elif (self._mouse_down_pos is not None
                  and self._is_meaningful_drag(event.pos)):
                # Promote a left-drag past the click threshold to a pan
                # (Macs without a middle mouse button can't use button 2).
                self._panning = True
                self._pan_anchor = event.pos
            else:
                # Update hovered room for keyboard parity.
                room = self._room_at_screen(event.pos)
                self._hovered_room_id = room.id if room is not None else None
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
        # Room-info modal renders even later so it can stack above the
        # options panel if the DM ever opens both (we don't expect this,
        # but the ordering is unambiguous).
        self._draw_room_info_modal(surface)
        # Startup warning overlay sits on top of everything until the
        # user dismisses it (one-shot, no-op when no warnings queued).
        self._draw_startup_warnings(surface)

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
        """Per-room state dot at each annotated region's centroid. Sized to
        match the party marker so a non-party room reads at the same scale
        as the party room (otherwise the party dot draws on top of the
        smaller state dot and the party room looks bigger than the rest)."""
        zoom = self.camera.zoom
        radius = max(6, int((MARKER_RADIUS + 4) * zoom))
        ring = max(1, int(MARKER_RING * zoom))
        for r in self.level.rooms:
            wx, wy = self._room_centroid_world(r)
            if wx is None:
                continue
            sx, sy = self.camera.world_to_screen(wx, wy)
            color = STATE_COLORS.get(r.state, STATE_COLORS["unexplored"])
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
            color = (210, 60, 60) if r.id == self._annot_hovered_room_id else (60, 130, 220)
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
        # Compact level label — full display_name + ascend/descend hints
        # are reachable through the options menu (O), so the on-map chrome
        # stays out of the way.
        lv = self.level
        mode = f"  ·  ANNOTATING ({self._annot_tool})" if self._annot_mode else ""
        title = f"Lvl {lv.level_number}{mode}"
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
            msg = ("A: exit · drag: rect · space-drag: pan · "
                   "P: polygon · hover+Del: remove room · Enter: close poly · "
                   "⌘Z: undo · ⌘+/⌘-: zoom · ⌘0: fit · Esc: cancel/exit")
        else:
            msg = ("O: options · click or Enter: cycle hovered · "
                   "right-click or I: notes · drag: pan · "
                   "scroll or ⌘+/⌘-: zoom · ⌘0: fit · Esc: quit")
        help_text = self.help_font.render(msg, True, HELP_INK)
        surface.blit(help_text, (8, surface.get_height() - help_text.get_height() - 6))

        # Bottom strip lives only outside annotation mode (annotation has
        # its own dedicated chrome above, and drawing actions over the map
        # when the DM is sketching room outlines would just be clutter).
        if not self._annot_mode:
            self._draw_status_strip(surface)
            self._draw_action_buttons(surface)

    # -- Bottom strip drawing -----------------------------------------------

    def _draw_status_strip(self, surface: pygame.Surface) -> None:
        """Single-line text summary above the action buttons: turn,
        elapsed time, light sources, supply pool, last WM, Noisy."""
        tracker = self.session.tracker
        turn = tracker.turn
        h, m = tracker.elapsed_hm
        # Light source summary: count active + flag any with ≤ 2 turns.
        actives = list(tracker.light_sources)
        if actives:
            lights_str = " ".join(
                f"{ls.label.split()[0]}{ls.turns_remaining}t"
                + ("⚠" if ls.turns_remaining <= config.LIGHT_LOW_WARNING_TURNS else "")
                for ls in actives
            )
        else:
            lights_str = "—"
        # Compact supply abbreviations (T:4 L:1 O:3 R:24 W:6).
        supplies = self.session.get_supplies()
        sup_str = " ".join(
            f"{SUPPLY_ABBREV[k]}:{supplies.get(k, 0)}"
            for k in SUPPLY_DISPLAY_ORDER
        )
        wm = tracker.last_wm
        if wm is None:
            wm_str = "—"
        else:
            mark = "✗" if wm.triggered else "✓"
            wm_str = f"{wm.method}={wm.roll}{mark}"
            if wm.triggered and wm.encounter:
                wm_str += f" {wm.encounter}"
        noisy_str = "  · NOISY" if tracker.noisy else ""

        text = (f"Turn {turn}  {h}h{m:02d}m"
                f"  ·  Lights: {lights_str}"
                f"  ·  Stash: {sup_str}"
                f"  ·  WM: {wm_str}{noisy_str}")
        surf = self.help_font.render(text, True, STATUS_INK)
        # Plate underneath so the strip stays readable over the PNG.
        bar_y = (surface.get_height() - STRIP_PAD_BOTTOM
                 - ACTION_BUTTON_HEIGHT - STATUS_BAR_HEIGHT - 8)
        plate_color = STATUS_PLATE_NOISY if tracker.noisy else STATUS_PLATE
        plate = pygame.Surface((surface.get_width(), STATUS_BAR_HEIGHT),
                               pygame.SRCALPHA)
        plate.fill(plate_color)
        surface.blit(plate, (0, bar_y))
        surface.blit(surf, (10, bar_y + (STATUS_BAR_HEIGHT - surf.get_height()) // 2))

    def _action_buttons(self) -> list[tuple[str, Callable[[], None], bool]]:
        """List of (label, action, enabled) tuples shown in the bottom row."""
        return [
            ("Advance Turn", self.action_advance_turn, True),
            ("Roll WM",      self.action_manual_wm_roll, True),
            ("+ Torch",      self.action_light_torch,
             self._can_light_torch()),
            ("+ Lantern",    self.action_light_lantern,
             self._can_light_lantern()),
            ("Refill",       self.action_refill_lantern,
             self._can_refill_lantern()),
            ("Noisy",        self.action_toggle_noisy, True),
        ]

    def _draw_action_buttons(self, surface: pygame.Surface) -> None:
        """Row of clickable buttons. Recomputes hit-test rects each frame
        so window resizes don't desync clicks."""
        self._action_button_rects = []
        items = self._action_buttons()
        total_w = (len(items) * ACTION_BUTTON_WIDTH
                   + (len(items) - 1) * ACTION_BUTTON_GAP)
        # Centered horizontally, just above the help line.
        x = max(10, (surface.get_width() - total_w) // 2)
        y = surface.get_height() - STRIP_PAD_BOTTOM - ACTION_BUTTON_HEIGHT
        for label, action, enabled in items:
            rect = pygame.Rect(x, y, ACTION_BUTTON_WIDTH, ACTION_BUTTON_HEIGHT)
            # Highlight Noisy button when active.
            is_noisy_active = (label == "Noisy" and self.session.tracker.noisy)
            if is_noisy_active:
                fill = (200, 100, 60)
                edge = (120, 50, 20)
                ink = (255, 250, 240)
            elif enabled:
                fill = (252, 252, 250)
                edge = (26, 26, 26)
                ink = (26, 26, 26)
            else:
                fill = (220, 215, 200)
                edge = (130, 110, 80)
                ink = (130, 110, 80)
            pygame.draw.rect(surface, fill, rect, border_radius=4)
            pygame.draw.rect(surface, edge, rect, 2, border_radius=4)
            text = self.help_font.render(label, True, ink)
            surface.blit(text, (rect.x + (rect.w - text.get_width()) // 2,
                                rect.y + (rect.h - text.get_height()) // 2))
            self._action_button_rects.append((rect, action, enabled))
            x += ACTION_BUTTON_WIDTH + ACTION_BUTTON_GAP

    def _handle_action_button_click(self, pos: tuple[int, int]) -> bool:
        """If `pos` lands on an enabled action button, run it. Returns True
        if the click was consumed (caller skips room hit-test)."""
        for rect, action, enabled in self._action_button_rects:
            if rect.collidepoint(pos):
                if enabled:
                    action()
                return True
        return False

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
        self._annot_hovered_room_id = None

    def _handle_annotation_event(self, event: pygame.event.Event) -> None:
        # Pan + zoom still work in annotation mode. Three pan gestures
        # are accepted, in order of convention:
        #   - Space + left-drag  (the universal graphics-tool gesture —
        #     Photoshop/Figma/Illustrator all use this; most discoverable)
        #   - Option/Alt + left-drag (kept for muscle memory)
        #   - Middle-mouse-button drag (3-button mouse only)
        # Plain left-drag stays reserved for drawing room rectangles.
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 2:
                self._panning = True
                self._pan_anchor = event.pos
                return
            if event.button == 1:
                mods = pygame.key.get_mods()
                space_held = pygame.key.get_pressed()[pygame.K_SPACE]
                if space_held or (mods & pygame.KMOD_ALT):
                    self._panning = True
                    self._pan_anchor = event.pos
                    return
                self._annot_mouse_down(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if self._panning:
                    self._panning = False
                    self._pan_anchor = None
                else:
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
            else:
                # Idle motion: update hovered room so it highlights red
                # and Delete/Backspace can act on it.
                room = self._room_at_screen(event.pos)
                self._annot_hovered_room_id = room.id if room is not None else None
        elif event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            factor = ZOOM_STEP if event.y > 0 else 1 / ZOOM_STEP
            self.camera.zoom_at(mx, my, factor)

    def _annot_mouse_down(self, pos: tuple[int, int]) -> None:
        if self._annot_tool == "rect":
            # Click on an existing region: no-op. The hover state
            # already highlights it (red), and Delete/Backspace will
            # remove the hovered room. We deliberately don't start a
            # new rectangle on top of an existing one — overlapping
            # rooms are confusing in fog rendering and almost always
            # accidental.
            room = self._room_at_screen(pos)
            if room is not None:
                return
            # Empty space: start a new rectangle drag.
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

    def _delete_hovered_room(self) -> None:
        """Remove the room currently under the cursor in annotation
        mode. No-op if nothing is hovered. Wired to Delete/Backspace
        in the run loop's annotation-mode key handler."""
        rid = self._annot_hovered_room_id
        if rid is None:
            return
        # Snapshot for undo before mutating.
        idx = next((i for i, r in enumerate(self.level.rooms) if r.id == rid), None)
        if idx is None:
            return
        snapshot = self.level.rooms[idx]
        self._remove_room(rid)
        self._push_undo("delete", self.level.level_number, snapshot, idx)
        self._annot_hovered_room_id = None
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

    # -- Session actions (bottom strip + keyboard shortcuts) ----------------

    def action_advance_turn(self) -> None:
        """Tick the turn counter, light timers, and (per level cadence)
        roll a wandering monster check."""
        self.session.advance_turn()

    def action_manual_wm_roll(self) -> None:
        """Roll a wandering monster check WITHOUT advancing the turn."""
        self.session.tracker.roll_wm()
        self.session.save()

    def _can_light_torch(self) -> bool:
        return self.session.get_supplies().get("torches", 0) > 0

    def action_light_torch(self) -> None:
        if not self._can_light_torch():
            return
        self.session.add_light_source("torch")
        self.session.consume_supply("torches", 1)

    def _active_lantern_count(self) -> int:
        return sum(1 for ls in self.session.tracker.light_sources
                   if ls.kind in ("hooded_lantern", "bullseye_lantern"))

    def _can_light_lantern(self) -> bool:
        """A lantern is a *permanent object*: you can light one only if you
        have a lantern not already lit AND have oil to fuel it."""
        s = self.session.get_supplies()
        total_lanterns = s.get("hooded_lantern", 0)
        return (total_lanterns > self._active_lantern_count()
                and s.get("oil_flask", 0) > 0)

    def action_light_lantern(self) -> None:
        if not self._can_light_lantern():
            return
        self.session.add_light_source("hooded_lantern")
        # The lantern itself is not consumed — only the oil is. The stash
        # count `L:N` reflects total lanterns owned; whether one is lit is
        # tracked in tracker.light_sources.
        self.session.consume_supply("oil_flask", 1)

    def _active_lantern(self) -> "LightSource | None":
        """The active lantern with the LEAST turns remaining — refilling
        the dimmest one is what a DM almost always wants."""
        lanterns = [ls for ls in self.session.tracker.light_sources
                    if ls.kind in ("hooded_lantern", "bullseye_lantern")]
        if not lanterns:
            return None
        return min(lanterns, key=lambda ls: ls.turns_remaining)

    def _can_refill_lantern(self) -> bool:
        return (self._active_lantern() is not None
                and self.session.get_supplies().get("oil_flask", 0) > 0)

    def action_refill_lantern(self) -> None:
        lantern = self._active_lantern()
        if lantern is None:
            return
        if self.session.get_supplies().get("oil_flask", 0) <= 0:
            return
        # Reset to a full flask's duration.
        lantern.turns_remaining = config.LIGHT_DURATIONS_TURNS[lantern.kind]
        self.session.consume_supply("oil_flask", 1)
        self.session.tracker.journal.record(
            self.session.tracker.turn, journal_mod.KIND_NOTE,
            f"{lantern.label}: refilled "
            f"({lantern.turns_remaining} turns)",
        )
        self.session.save()

    def action_toggle_noisy(self) -> None:
        self.session.tracker.set_noisy(not self.session.tracker.noisy)
        self.session.save()

    def action_reset_progress(self) -> None:
        """Reset all play state to a fresh-session baseline: turn=0,
        fog restored, supplies refilled to defaults, exhaustion zeroed,
        light sources extinguished, journal cleared, party at level 1's
        first room. Annotations and dungeon metadata are preserved."""
        self.session.reset_progress()
        # The journal was cleared inside reset_progress — drop a single
        # note line so the DM has a marker for "this run started here".
        self.session.tracker.journal.record(
            self.session.tracker.turn, journal_mod.KIND_NOTE,
            "Dungeon progress reset — fresh run starting.",
        )
        self.session.save()
        # Reload the level so the view re-binds to whichever level
        # reset_progress moved the party to (the shallowest).
        self.switch_to_current_level()
        self._invalidate_fog()
        if self._on_change is not None:
            self._on_change()

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
            ("New Dungeon…", "",
             self.open_new_dungeon_modal,
             self._dungeons_dir is not None),
            ("Open Different Dungeon…", "",
             self.open_dungeon_picker,
             self._dungeons_dir is not None),
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
            ("Reset Dungeon Progress…", "",
             self.open_progress_confirm, True),
            ("Reset Dungeon…", "",
             self.open_reset_confirm,
             self._dungeon_path is not None),
            ("Quit", "Esc",
             self._action_quit,
             self._action_quit is not None),
        ]

    def open_reset_confirm(self) -> None:
        """Switch the options modal into reset-confirm mode. Caller is the
        menu row handler — the menu must already be open."""
        if self._dungeon_path is None:
            return
        self._reset_confirm_open = True
        self._options_open = True

    def close_reset_confirm(self) -> None:
        self._reset_confirm_open = False

    # -- New Dungeon modal -----------------------------------------------

    def open_new_dungeon_modal(self) -> None:
        """Open the New Dungeon modal. Caller is the menu row handler;
        the options menu must already be open."""
        if self._dungeons_dir is None:
            return
        self._new_dungeon_open = True
        self._new_dungeon_name = ""
        self._new_dungeon_party_level = 3
        self._new_dungeon_focus = "name"
        self._new_dungeon_error = ""
        self._options_open = True

    def close_new_dungeon_modal(self) -> None:
        self._new_dungeon_open = False
        self._new_dungeon_error = ""

    def _do_create_new_dungeon(self) -> None:
        """Validate the form, scaffold the dungeon, and queue a
        ReloadRequest so main.py swaps the running app onto it."""
        if self._dungeons_dir is None:
            return
        from session import Session  # local import — avoid circular
        name = self._new_dungeon_name.strip()
        if not name:
            self._new_dungeon_error = "Enter a name for the dungeon."
            return
        slug = Session.slugify_dungeon_name(name)
        target = (self._dungeons_dir / slug).resolve()
        if target.exists():
            self._new_dungeon_error = (
                f"A dungeon already exists at dungeons/{slug}/. "
                f"Pick a different name."
            )
            return
        try:
            Session.scaffold_dungeon(
                target, name=name,
                party_level=self._new_dungeon_party_level,
            )
        except (ValueError, FileExistsError, OSError) as e:
            self._new_dungeon_error = f"Couldn't create dungeon: {e}"
            return
        # Success — close all modals and reload into the new folder.
        # session.save() flushes the current dungeon's state before the
        # run loop returns; main.py then opens the new folder.
        self.session.save()
        self._new_dungeon_open = False
        self._options_open = False
        self._pending_reload = ReloadRequest(folder=target,
                                             do_full_reset=False)

    def _handle_new_dungeon_keydown(self, event: pygame.event.Event) -> bool:
        """Route a keydown to the focused field. Returns True if the
        event was consumed (so the run loop doesn't also process it)."""
        if not self._new_dungeon_open:
            return False
        if event.key == pygame.K_ESCAPE:
            self.close_new_dungeon_modal()
            return True
        if event.key == pygame.K_TAB:
            self._new_dungeon_focus = (
                "party_level" if self._new_dungeon_focus == "name" else "name"
            )
            return True
        if event.key == pygame.K_RETURN:
            self._do_create_new_dungeon()
            return True
        if self._new_dungeon_focus == "name":
            if event.key == pygame.K_BACKSPACE:
                self._new_dungeon_name = self._new_dungeon_name[:-1]
                self._new_dungeon_error = ""
                return True
            ch = event.unicode
            if ch and ch.isprintable() and len(self._new_dungeon_name) < 64:
                self._new_dungeon_name += ch
                self._new_dungeon_error = ""
                return True
        elif self._new_dungeon_focus == "party_level":
            if event.key == pygame.K_BACKSPACE:
                # Reset to 1 on backspace so the field is never blank
                # (we display ints, not free-text).
                self._new_dungeon_party_level = 1
                return True
            if event.key in (pygame.K_UP, pygame.K_RIGHT, pygame.K_PLUS,
                             pygame.K_EQUALS):
                self._new_dungeon_party_level = min(
                    20, self._new_dungeon_party_level + 1)
                return True
            if event.key in (pygame.K_DOWN, pygame.K_LEFT, pygame.K_MINUS,
                             pygame.K_KP_MINUS):
                self._new_dungeon_party_level = max(
                    1, self._new_dungeon_party_level - 1)
                return True
            ch = event.unicode
            if ch and ch.isdigit():
                # Replace-on-type: 1 → 12 → 3 (typing replaces, easier
                # than tracking digit-buffer state for a 1-2 digit field).
                self._new_dungeon_party_level = max(1, min(20, int(ch)))
                return True
        return True  # swallow everything else while modal is open

    def _handle_new_dungeon_click(self, pos: tuple[int, int]) -> bool:
        """Click handler for the New Dungeon modal. Returns True if
        consumed."""
        if not self._new_dungeon_open:
            return False
        for rect, action in self._new_dungeon_rects:
            if rect.collidepoint(pos):
                action()
                return True
        # Click on a field switches focus.
        for field, rect in self._new_dungeon_field_rects.items():
            if rect.collidepoint(pos):
                self._new_dungeon_focus = field
                return True
        return True  # swallow clicks outside; user must Cancel/Esc

    def open_progress_confirm(self) -> None:
        """Switch the options modal into reset-progress confirm mode.
        Lighter than the full-reset flow — preserves annotations and
        only zeros runtime state."""
        self._progress_confirm_open = True
        self._options_open = True

    def close_progress_confirm(self) -> None:
        self._progress_confirm_open = False

    def _do_reset_progress(self) -> None:
        """Execute the progress reset and close everything modal."""
        self.action_reset_progress()
        self._progress_confirm_open = False
        self._options_open = False

    def _do_full_reset(self) -> None:
        """Signal main.py to wipe the current dungeon and reopen it. The
        actual full_reset runs in main.py once the run loop has exited so
        the SQLite handle is closed and the editor server stops before we
        rewrite the JSON."""
        assert self._dungeon_path is not None
        folder = Path(self._dungeon_path).parent
        self._pending_reload = ReloadRequest(folder=folder,
                                             do_full_reset=True)

    def open_dungeon_picker(self) -> None:
        """Switch the options modal into picker mode. The menu must already
        be open (this is invoked from a menu row) — we just flip the flag."""
        if self._dungeons_dir is None:
            return
        self._picker_open = True
        self._options_open = True

    def close_dungeon_picker(self) -> None:
        self._picker_open = False

    def _switch_to_dungeon(self, folder: Path) -> None:
        """Signal main.py to reload into `folder`. The pygame run loop
        catches the request, returns it, and main.py rebuilds the session
        and editor server in-process — the SDL window stays alive (an
        os.execv here breaks the macOS WindowServer attachment)."""
        self.session.save()
        self._pending_reload = ReloadRequest(folder=folder,
                                             do_full_reset=False)

    def _draw_options_menu(self, surface: pygame.Surface) -> None:
        """Draw the modal options panel. Recomputes button rects so the
        click handler stays accurate after window resizes."""
        self._options_button_rects = []
        self._picker_rows = []
        self._reset_confirm_rects = []
        self._progress_confirm_rects = []
        if not self._options_open:
            return

        # Dim backdrop so the map fades behind the menu.
        backdrop = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        backdrop.fill((0, 0, 0, 130))
        surface.blit(backdrop, (0, 0))

        if self._picker_open:
            self._draw_dungeon_picker(surface)
            return
        if self._new_dungeon_open:
            self._draw_new_dungeon_modal(surface)
            return
        if self._progress_confirm_open:
            self._draw_progress_confirm(surface)
            return
        if self._reset_confirm_open:
            self._draw_reset_confirm(surface)
            return

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

    def _draw_dungeon_picker(self, surface: pygame.Surface) -> None:
        """Inside the options modal: list of dungeon folders. Click → switch.
        Caller (_draw_options_menu) has already drawn the dim backdrop."""
        try:
            infos = (Session.list_dungeons(self._dungeons_dir)
                     if self._dungeons_dir is not None else [])
        except Exception:
            infos = []

        current_folder: Path | None = None
        if self._dungeon_path is not None:
            current_folder = Path(self._dungeon_path).parent.resolve()

        panel_w = 520
        row_h = 56
        row_gap = 6
        title_h = 56
        panel_pad = 22
        n_rows = max(1, len(infos))
        panel_h = (title_h + panel_pad
                   + n_rows * (row_h + row_gap) + panel_pad)
        panel_x = (surface.get_width() - panel_w) // 2
        panel_y = max(40, (surface.get_height() - panel_h) // 2)

        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        panel_bg = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel_bg.fill((244, 228, 193, 250))
        surface.blit(panel_bg, panel_rect.topleft)
        pygame.draw.rect(surface, (26, 26, 26), panel_rect, 3,
                         border_radius=6)

        title = self.title_font.render("OPEN DUNGEON", True, (26, 26, 26))
        surface.blit(title, (panel_x + (panel_w - title.get_width()) // 2,
                             panel_y + 16))

        row_x = panel_x + 20
        row_w = panel_w - 40
        row_y = panel_y + title_h + panel_pad

        if not infos:
            msg = self.help_font.render(
                f"No dungeons found in {self._dungeons_dir}.",
                True, (60, 50, 30),
            )
            surface.blit(msg, (row_x, row_y + 8))
            return

        for info in infos:
            rect = pygame.Rect(row_x, row_y, row_w, row_h)
            is_current = (current_folder is not None
                          and info.folder.resolve() == current_folder)
            if is_current:
                fill = (220, 215, 200)
                edge = (130, 110, 80)
                ink = (130, 110, 80)
            else:
                fill = (252, 252, 250)
                edge = (26, 26, 26)
                ink = (26, 26, 26)
            pygame.draw.rect(surface, fill, rect, border_radius=4)
            pygame.draw.rect(surface, edge, rect, 2, border_radius=4)

            top = self.title_font.render(info.name, True, ink)
            surface.blit(top, (rect.x + 14, rect.y + 6))

            if info.has_session:
                stats = (f"L{info.current_level} of {info.n_levels}"
                         f" · turn {info.current_turn}"
                         f" · saved {info.last_saved_at[:19]}")
            else:
                stats = f"{info.n_levels} levels · no progress yet"
            if is_current:
                stats = f"(current) · {stats}"
            sub = self.help_font.render(stats, True, ink)
            surface.blit(sub, (rect.x + 14, rect.y + 30))

            # Only enqueue clickable rows for non-current entries.
            if not is_current:
                self._picker_rows.append((rect, info.folder))

            row_y += row_h + row_gap

    def _draw_new_dungeon_modal(self, surface: pygame.Surface) -> None:
        """Modal collecting a new dungeon's name + party level.
        Renders inside the options-modal stack — caller (the options
        menu draw) has already drawn the dim backdrop."""
        from session import Session  # for slug preview
        ink = (26, 26, 26)
        muted = (130, 110, 80)
        focus_ring = (90, 72, 48)
        panel_w = 540
        title_h = 56
        panel_pad = 22
        body_h = 220
        btn_h = 46
        btn_gap = 12
        panel_h = title_h + panel_pad + body_h + btn_h + panel_pad
        panel_x = (surface.get_width() - panel_w) // 2
        panel_y = max(40, (surface.get_height() - panel_h) // 2)

        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        panel_bg = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel_bg.fill((244, 228, 193, 250))
        surface.blit(panel_bg, panel_rect.topleft)
        pygame.draw.rect(surface, ink, panel_rect, 3, border_radius=6)

        title = self.title_font.render("NEW DUNGEON", True, ink)
        surface.blit(title, (panel_x + (panel_w - title.get_width()) // 2,
                             panel_y + 16))

        # Name field
        label_y = panel_y + title_h + panel_pad - 6
        name_label = self.help_font.render(
            "Dungeon name (free text):", True, muted)
        surface.blit(name_label, (panel_x + 24, label_y))
        name_rect = pygame.Rect(panel_x + 24, label_y + 18,
                                panel_w - 48, 32)
        bg_color = (252, 252, 250)
        pygame.draw.rect(surface, bg_color, name_rect, border_radius=3)
        edge = focus_ring if self._new_dungeon_focus == "name" else (130, 110, 80)
        edge_w = 2 if self._new_dungeon_focus == "name" else 1
        pygame.draw.rect(surface, edge, name_rect, edge_w, border_radius=3)

        body_font = pygame.font.SysFont("georgia,serif", 16)
        text_surf = body_font.render(self._new_dungeon_name, True, ink)
        surface.blit(text_surf, (name_rect.x + 8, name_rect.y + 6))
        # Caret indicator when focused.
        if self._new_dungeon_focus == "name":
            caret_x = name_rect.x + 8 + text_surf.get_width()
            pygame.draw.line(
                surface, ink,
                (caret_x, name_rect.y + 5),
                (caret_x, name_rect.y + name_rect.height - 5),
                1,
            )
        self._new_dungeon_field_rects["name"] = name_rect

        # Slug preview
        slug = Session.slugify_dungeon_name(self._new_dungeon_name)
        slug_msg = (f"Folder will be: dungeons/{slug}/"
                    if self._new_dungeon_name.strip() else
                    "Folder name is generated from the dungeon name.")
        slug_surf = self.help_font.render(slug_msg, True, muted)
        surface.blit(slug_surf, (panel_x + 24, name_rect.y + name_rect.height + 6))

        # Party-level field
        party_label_y = name_rect.y + name_rect.height + 30
        party_label = self.help_font.render(
            "Party level (1–20, ↑/↓ or type a digit):", True, muted)
        surface.blit(party_label, (panel_x + 24, party_label_y))
        party_rect = pygame.Rect(panel_x + 24, party_label_y + 18, 80, 32)
        pygame.draw.rect(surface, bg_color, party_rect, border_radius=3)
        edge_p = focus_ring if self._new_dungeon_focus == "party_level" else (130, 110, 80)
        edge_pw = 2 if self._new_dungeon_focus == "party_level" else 1
        pygame.draw.rect(surface, edge_p, party_rect, edge_pw, border_radius=3)
        pl_surf = body_font.render(
            str(self._new_dungeon_party_level), True, ink)
        surface.blit(pl_surf,
                     (party_rect.x + (party_rect.width - pl_surf.get_width()) // 2,
                      party_rect.y + 6))
        self._new_dungeon_field_rects["party_level"] = party_rect

        # Error (only when present)
        if self._new_dungeon_error:
            err_surf = self.help_font.render(
                self._new_dungeon_error, True, ALERT_RED)
            surface.blit(err_surf, (panel_x + 24, party_rect.y + 42))

        # Hint line right above buttons.
        hint = self.help_font.render(
            "After creating, drop your level1.png into the new folder, "
            "then press A to annotate rooms.",
            True, muted,
        )
        surface.blit(hint,
                     (panel_x + 24, panel_y + panel_h - panel_pad - btn_h - 22))

        # Buttons
        btn_total_w = panel_w - 48
        each_w = (btn_total_w - btn_gap) // 2
        btn_y = panel_y + panel_h - panel_pad - btn_h
        cancel_rect = pygame.Rect(panel_x + 24, btn_y, each_w, btn_h)
        create_rect = pygame.Rect(panel_x + 24 + each_w + btn_gap, btn_y,
                                  each_w, btn_h)

        pygame.draw.rect(surface, (252, 252, 250), cancel_rect, border_radius=4)
        pygame.draw.rect(surface, ink, cancel_rect, 2, border_radius=4)
        cancel_label = self.title_font.render("Cancel", True, ink)
        surface.blit(cancel_label,
                     (cancel_rect.x + (cancel_rect.width - cancel_label.get_width()) // 2,
                      cancel_rect.y + (btn_h - cancel_label.get_height()) // 2))

        pygame.draw.rect(surface, ink, create_rect, border_radius=4)
        create_label = self.title_font.render("Create dungeon",
                                              True, (244, 228, 193))
        surface.blit(create_label,
                     (create_rect.x + (create_rect.width - create_label.get_width()) // 2,
                      create_rect.y + (btn_h - create_label.get_height()) // 2))

        self._new_dungeon_rects = [
            (cancel_rect, self.close_new_dungeon_modal),
            (create_rect, self._do_create_new_dungeon),
        ]

    def _draw_progress_confirm(self, surface: pygame.Surface) -> None:
        """Confirm modal for Reset Dungeon Progress. Less alarming than
        the full-reset modal (no red panel border, no "destructive"
        copy) since annotations are preserved — just runtime state."""
        ink = (26, 26, 26)
        panel_w = 540
        title_h = 56
        panel_pad = 22
        body_h = 196
        btn_h = 46
        btn_gap = 12
        panel_h = title_h + panel_pad + body_h + btn_h + panel_pad
        panel_x = (surface.get_width() - panel_w) // 2
        panel_y = max(40, (surface.get_height() - panel_h) // 2)

        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        panel_bg = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel_bg.fill((244, 228, 193, 250))
        surface.blit(panel_bg, panel_rect.topleft)
        pygame.draw.rect(surface, ink, panel_rect, 3, border_radius=6)

        title = self.title_font.render("RESET PROGRESS", True, ink)
        surface.blit(title, (panel_x + (panel_w - title.get_width()) // 2,
                             panel_y + 16))

        body_lines = [
            "This resets the play state to a fresh run:",
            "  • turn counter back to 0",
            "  • fog of war restored on every level",
            "  • supplies refilled to defaults",
            "  • exhaustion zeroed; lights extinguished",
            "  • journal cleared; party returned to level 1",
            "",
            "Annotations and dungeon metadata are preserved.",
        ]
        ty = panel_y + title_h + panel_pad - 6
        for line in body_lines:
            surf = self.help_font.render(line, True, (40, 30, 20))
            surface.blit(surf, (panel_x + 24, ty))
            ty += surf.get_height() + 4

        btn_total_w = panel_w - 48
        each_w = (btn_total_w - btn_gap) // 2
        btn_y = panel_y + panel_h - panel_pad - btn_h
        cancel_rect = pygame.Rect(panel_x + 24, btn_y, each_w, btn_h)
        confirm_rect = pygame.Rect(panel_x + 24 + each_w + btn_gap, btn_y,
                                   each_w, btn_h)

        # Cancel — neutral parchment.
        pygame.draw.rect(surface, (252, 252, 250), cancel_rect, border_radius=4)
        pygame.draw.rect(surface, ink, cancel_rect, 2, border_radius=4)
        cancel_label = self.title_font.render("Cancel", True, ink)
        surface.blit(cancel_label,
                     (cancel_rect.x + (cancel_rect.width - cancel_label.get_width()) // 2,
                      cancel_rect.y + (btn_h - cancel_label.get_height()) // 2))

        # Confirm — dark fill (matches the Advance Turn / save buttons),
        # not red, since annotations survive.
        pygame.draw.rect(surface, ink, confirm_rect, border_radius=4)
        confirm_label = self.title_font.render("Reset progress",
                                               True, (244, 228, 193))
        surface.blit(confirm_label,
                     (confirm_rect.x + (confirm_rect.width - confirm_label.get_width()) // 2,
                      confirm_rect.y + (btn_h - confirm_label.get_height()) // 2))

        self._progress_confirm_rects = [
            (cancel_rect, self.close_progress_confirm),
            (confirm_rect, self._do_reset_progress),
        ]

    def _draw_reset_confirm(self, surface: pygame.Surface) -> None:
        """Inside the options modal: destructive confirm for full reset."""
        panel_w = 540
        title_h = 56
        panel_pad = 22
        body_h = 180
        btn_h = 46
        btn_gap = 12
        panel_h = title_h + panel_pad + body_h + btn_h + panel_pad
        panel_x = (surface.get_width() - panel_w) // 2
        panel_y = max(40, (surface.get_height() - panel_h) // 2)

        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        panel_bg = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel_bg.fill((244, 228, 193, 250))
        surface.blit(panel_bg, panel_rect.topleft)
        pygame.draw.rect(surface, ALERT_RED, panel_rect, 3, border_radius=6)

        title = self.title_font.render("RESET DUNGEON", True, ALERT_RED)
        surface.blit(title, (panel_x + (panel_w - title.get_width()) // 2,
                             panel_y + 16))

        body_lines = [
            "This will delete every annotated room and corridor on every level,",
            "and reset turn / fog / supplies. Level metadata and wandering monster",
            "tables are preserved.",
            "",
            "A timestamped .bak of the current dungeon.json is saved next to it",
            "before anything is overwritten.",
        ]
        ty = panel_y + title_h + panel_pad - 6
        for line in body_lines:
            surf = self.help_font.render(line, True, (40, 30, 20))
            surface.blit(surf, (panel_x + 24, ty))
            ty += surf.get_height() + 4

        # Two side-by-side buttons: Cancel (neutral) | Wipe (destructive red).
        btn_total_w = panel_w - 48
        each_w = (btn_total_w - btn_gap) // 2
        btn_y = panel_y + panel_h - panel_pad - btn_h
        cancel_rect = pygame.Rect(panel_x + 24, btn_y, each_w, btn_h)
        wipe_rect = pygame.Rect(panel_x + 24 + each_w + btn_gap, btn_y,
                                each_w, btn_h)

        # Cancel button (parchment fill, black ink).
        pygame.draw.rect(surface, (252, 252, 250), cancel_rect, border_radius=4)
        pygame.draw.rect(surface, (26, 26, 26), cancel_rect, 2, border_radius=4)
        cancel_label = self.title_font.render("Cancel", True, (26, 26, 26))
        surface.blit(cancel_label,
                     (cancel_rect.x + (cancel_rect.width - cancel_label.get_width()) // 2,
                      cancel_rect.y + (btn_h - cancel_label.get_height()) // 2))

        # Wipe button (red fill, white ink — destructive emphasis).
        pygame.draw.rect(surface, ALERT_RED, wipe_rect, border_radius=4)
        pygame.draw.rect(surface, (26, 26, 26), wipe_rect, 2, border_radius=4)
        wipe_label = self.title_font.render("Wipe annotations + session",
                                            True, (255, 255, 255))
        surface.blit(wipe_label,
                     (wipe_rect.x + (wipe_rect.width - wipe_label.get_width()) // 2,
                      wipe_rect.y + (btn_h - wipe_label.get_height()) // 2))

        self._reset_confirm_rects = [
            (cancel_rect, self.close_reset_confirm),
            (wipe_rect, self._do_full_reset),
        ]

    def _handle_options_click(self, pos: tuple[int, int]) -> bool:
        """Return True if the click was consumed by the options menu."""
        if not self._options_open:
            return False
        if self._new_dungeon_open:
            return self._handle_new_dungeon_click(pos)
        if self._progress_confirm_open:
            for rect, action in self._progress_confirm_rects:
                if rect.collidepoint(pos):
                    action()
                    return True
            # Click outside the dialog cancels.
            self._progress_confirm_open = False
            return True
        if self._reset_confirm_open:
            for rect, action in self._reset_confirm_rects:
                if rect.collidepoint(pos):
                    action()
                    return True
            # Click outside the dialog cancels.
            self._reset_confirm_open = False
            self._options_open = False
            return True
        if self._picker_open:
            for rect, folder in self._picker_rows:
                if rect.collidepoint(pos):
                    self._switch_to_dungeon(folder)
                    return True  # unreachable — execv replaces process
            # Click outside a row closes the picker AND the menu.
            self._picker_open = False
            self._options_open = False
            return True
        for rect, action, enabled in self._options_button_rects:
            if rect.collidepoint(pos):
                if enabled and action is not None:
                    self._options_open = False
                    action()
                return True
        # Click outside any button (and the menu is open) → close.
        self._options_open = False
        return True

    # -- Room-info modal -----------------------------------------------------

    def _wrap_text(self, text: str, font: pygame.font.Font,
                   max_width: int) -> list[str]:
        """Greedy word-wrap honouring explicit \\n breaks. Empty string
        returns []; pure-whitespace lines are preserved as blank lines so
        paragraph breaks survive."""
        out: list[str] = []
        for raw_line in text.split("\n"):
            if raw_line == "":
                out.append("")
                continue
            words = raw_line.split(" ")
            current = ""
            for w in words:
                trial = w if current == "" else current + " " + w
                if font.size(trial)[0] <= max_width:
                    current = trial
                else:
                    if current:
                        out.append(current)
                    current = w
            if current:
                out.append(current)
        return out

    # Inline markdown parser: splits text into (style, text) tokens, where
    # style is one of 'normal', 'bold', 'italic', 'bi'. The SRD source uses
    # `_**...**_` for bold-italic (e.g. attack names), `**...**` for plain
    # bold, and `_..._` for italic. Anything not inside those markers is
    # 'normal'. Order matters: longest delimiter wins.
    _MD_BI = "_**"
    _MD_BI_END = "**_"
    _MD_BOLD = "**"
    _MD_ITALIC = "_"

    def _parse_inline_md(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        i = 0
        n = len(text)
        while i < n:
            if text.startswith(self._MD_BI, i):
                end = text.find(self._MD_BI_END, i + 3)
                if end != -1:
                    out.append(("bi", text[i + 3:end]))
                    i = end + 3
                    continue
            if text.startswith(self._MD_BOLD, i):
                end = text.find(self._MD_BOLD, i + 2)
                if end != -1:
                    out.append(("bold", text[i + 2:end]))
                    i = end + 2
                    continue
            if text[i] == self._MD_ITALIC:
                end = text.find(self._MD_ITALIC, i + 1)
                if end != -1:
                    out.append(("italic", text[i + 1:end]))
                    i = end + 1
                    continue
            # Walk forward to the next marker (or end of string).
            nxt = n
            for marker in (self._MD_BI, self._MD_BOLD, self._MD_ITALIC):
                pos = text.find(marker, i + 1)
                if pos != -1 and pos < nxt:
                    nxt = pos
            if nxt == i:
                # No special marker accepted; consume one char and continue
                # so we never loop forever on malformed input.
                out.append(("normal", text[i]))
                i += 1
            else:
                out.append(("normal", text[i:nxt]))
                i = nxt
        return out

    def _layout_markdown(self, text: str, fonts: dict, line_h: int,
                         max_width: int, ink: tuple[int, int, int]
                         ) -> list[dict]:
        """Convert a markdown stat-block into a flat list of layout rows.
        Recognised constructs: ### / #### headers, **bold**, _italic_,
        _**bold-italic**_, bullets (- ), and table lines (starting with |
        — kept verbatim in monospace so column alignment survives)."""
        rows: list[dict] = []
        body_font = fonts["normal"]
        h3_font, h4_font, mono_font = fonts["h3"], fonts["h4"], fonts["mono"]
        h3_h, h4_h = h3_font.get_linesize(), h4_font.get_linesize()
        mono_h = mono_font.get_linesize()

        for raw in text.split("\n"):
            stripped = raw.rstrip()
            if not stripped:
                rows.append({"kind": "vspace", "h": max(4, line_h // 2)})
                continue
            if stripped.startswith("|"):
                rows.append({"kind": "mono", "text": stripped, "h": mono_h})
                continue
            if stripped.startswith("#### "):
                rows.append({"kind": "vspace", "h": 4})
                rows.append({"kind": "rich", "color": ink, "h": h4_h,
                             "segs": [(h4_font, stripped[5:].strip())]})
                rows.append({"kind": "vspace", "h": 2})
                continue
            if stripped.startswith("### "):
                rows.append({"kind": "vspace", "h": 6})
                rows.append({"kind": "rich", "color": ink, "h": h3_h,
                             "segs": [(h3_font, stripped[4:].strip())]})
                rows.append({"kind": "vspace", "h": 2})
                continue
            bullet = stripped.startswith("- ")
            inline = stripped[2:] if bullet else stripped
            segments = self._parse_inline_md(inline)
            current: list[tuple[pygame.font.Font, str]] = []
            current_w = 0
            indent_str = "• " if bullet else ""
            cont_indent = "  " if bullet else ""
            if indent_str:
                current.append((body_font, indent_str))
                current_w = body_font.size(indent_str)[0]
            for style, seg_text in segments:
                font = fonts.get(style, body_font)
                # Tokenise into runs of whitespace and non-whitespace so we
                # can break at word boundaries while preserving the spaces.
                for tok in re.findall(r"\S+|\s+", seg_text):
                    tw = font.size(tok)[0]
                    if tok.isspace():
                        if current and current_w + tw <= max_width:
                            current.append((font, tok))
                            current_w += tw
                        continue
                    if current_w + tw > max_width and current:
                        rows.append({"kind": "rich", "color": ink,
                                     "h": line_h, "segs": list(current)})
                        current = []
                        current_w = 0
                        if cont_indent:
                            current.append((body_font, cont_indent))
                            current_w = body_font.size(cont_indent)[0]
                    current.append((font, tok))
                    current_w += tw
            if current:
                rows.append({"kind": "rich", "color": ink, "h": line_h,
                             "segs": list(current)})
        return rows

    def _draw_room_info_modal(self, surface: pygame.Surface) -> None:
        """Modal overlay listing the JSON-side room metadata (box text,
        encounter, treasure, special, DM notes). Opened via right-click on
        a room; closed by Esc, the [×] button, or click outside the panel.
        The SRD stat blocks section is rendered with inline markdown
        (headers, bold, italic, bullets, monospace table rows)."""
        self._room_info_close_rect = None
        self._room_info_panel_rect = None
        if not self._room_info_open or self._room_info_room_id is None:
            return
        room = self.level.rooms_by_id.get(self._room_info_room_id)
        if room is None:
            self._room_info_open = False
            return

        # Dim backdrop (consistent with options menu).
        backdrop = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        backdrop.fill((0, 0, 0, 130))
        surface.blit(backdrop, (0, 0))

        ink = (26, 26, 26)
        muted = (130, 110, 80)
        body_font = pygame.font.SysFont("georgia,serif", 14)
        header_font = pygame.font.SysFont("georgia,serif", 13, bold=True)
        meta_font = self.help_font

        # Markdown-aware font palette used for the SRD stat-block section.
        md_fonts = {
            "normal": body_font,
            "bold": pygame.font.SysFont("georgia,serif", 14, bold=True),
            "italic": pygame.font.SysFont("georgia,serif", 14, italic=True),
            "bi": pygame.font.SysFont("georgia,serif", 14,
                                       bold=True, italic=True),
            "mono": pygame.font.SysFont("menlo,monaco,courier", 12),
            "h3": pygame.font.SysFont("georgia,serif", 17, bold=True),
            "h4": pygame.font.SysFont("georgia,serif", 14, bold=True),
        }

        sw, sh = surface.get_size()
        panel_w = min(620, sw - 40)
        text_max_w = panel_w - 48
        title_h = 56
        section_gap = 14
        section_pad = 6
        line_h = body_font.get_linesize()
        header_h = header_font.get_linesize()

        # Build the section list. The third element flags whether the body
        # is markdown (so far only the SRD stat blocks).
        sections: list[tuple[str, str, bool]] = []
        if room.box_text.strip():
            sections.append(("Box text (read aloud)", room.box_text, False))
        enc_parts: list[str] = []
        if room.encounter_text.strip():
            enc_parts.append(room.encounter_text)
        if room.encounter_ref:
            enc_parts.append(f"D&D Beyond: {room.encounter_ref}")
        if enc_parts:
            sections.append(("Encounter", "\n\n".join(enc_parts), False))
        if room.statblocks.strip():
            sections.append(("SRD stat blocks", room.statblocks, True))
        trsr_parts: list[str] = []
        if room.treasure_text.strip():
            trsr_parts.append(room.treasure_text)
        if room.treasure_tier:
            trsr_parts.append(f"Tier: {room.treasure_tier}")
        if trsr_parts:
            sections.append(("Treasure", "\n\n".join(trsr_parts), False))
        if room.special_text.strip():
            sections.append(("Special", room.special_text, False))
        if room.notes.strip():
            sections.append(("DM notes", room.notes, False))

        # Build a single flat list of layout rows for the body. Each row
        # carries its own height so visibility and scroll math stay simple.
        rows: list[dict] = []
        for header, text, is_md in sections:
            rows.append({"kind": "section_header", "text": header,
                         "h": header_h + section_pad})
            if is_md:
                rows.extend(self._layout_markdown(
                    text, md_fonts, line_h, text_max_w, ink))
            else:
                for line in self._wrap_text(text, body_font, text_max_w):
                    rows.append({"kind": "rich", "color": ink, "h": line_h,
                                 "segs": [(body_font, line)] if line else []})
            rows.append({"kind": "vspace", "h": section_gap})

        # Placeholder when the room has no metadata at all.
        if not rows:
            placeholder = (
                "(No notes for this room yet — open the editor with E to add "
                "box text, encounter, treasure, special, or DM notes.)"
            )
            for line in self._wrap_text(placeholder, body_font, text_max_w):
                rows.append({"kind": "rich", "color": muted, "h": line_h,
                             "segs": [(body_font, line)]})

        body_h = sum(r["h"] for r in rows)

        meta_h = meta_font.get_linesize() + 6
        if room.reaction_required:
            meta_h += meta_font.get_linesize() + 4

        # Panel height: prefer enough room for the content, but cap at 85%
        # of the window. When the content is taller, the body scrolls and
        # a thin scrollbar appears on the right edge.
        chrome_h = title_h + meta_h + 16 + 24
        max_h = int(sh * 0.85)
        panel_h = min(max_h, chrome_h + body_h)
        panel_x = (sw - panel_w) // 2
        panel_y = max(40, (sh - panel_h) // 2)
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

        panel_bg = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel_bg.fill((244, 228, 193, 250))
        surface.blit(panel_bg, panel_rect.topleft)
        pygame.draw.rect(surface, ink, panel_rect, 3, border_radius=6)
        self._room_info_panel_rect = panel_rect

        # Title and close button.
        title_text = f"{room.id} — {room.name}" if room.name else room.id
        title_surf = self.title_font.render(title_text, True, ink)
        surface.blit(title_surf, (panel_x + 24, panel_y + 18))

        close_rect = pygame.Rect(panel_x + panel_w - 38, panel_y + 14, 24, 24)
        pygame.draw.rect(surface, ink, close_rect, 2, border_radius=3)
        x_label = self.title_font.render("×", True, ink)
        surface.blit(x_label,
                     (close_rect.x + (close_rect.width - x_label.get_width()) // 2 + 1,
                      close_rect.y + (close_rect.height - x_label.get_height()) // 2 - 2))
        self._room_info_close_rect = close_rect

        sep_y = panel_y + title_h
        pygame.draw.line(surface, ink,
                         (panel_x + 16, sep_y), (panel_x + panel_w - 16, sep_y), 1)

        # Meta row.
        meta_y = sep_y + 8
        tag_str = "Tags: " + (", ".join(room.tags) if room.tags else "—")
        meta_surf = meta_font.render(tag_str, True, muted)
        surface.blit(meta_surf, (panel_x + 24, meta_y))
        if room.reaction_required:
            rr_y = meta_y + meta_font.get_linesize() + 2
            rr_surf = meta_font.render("REACTION REQUIRED on first entry",
                                       True, ALERT_RED)
            surface.blit(rr_surf, (panel_x + 24, rr_y))

        # Body region.
        body_top = sep_y + 8 + meta_h + 8
        body_bottom = panel_y + panel_h - 16
        body_view_h = max(0, body_bottom - body_top)
        body_rect = pygame.Rect(panel_x + 16, body_top,
                                panel_w - 32, body_view_h)
        self._room_info_body_rect = body_rect

        content_h = body_h

        self._room_info_max_scroll = max(0.0, content_h - body_view_h)
        self._room_info_scroll_y = max(
            0.0, min(self._room_info_max_scroll, self._room_info_scroll_y),
        )

        prior_clip = surface.get_clip()
        surface.set_clip(body_rect)
        try:
            y = body_top - int(self._room_info_scroll_y)
            text_x = panel_x + 24
            for row in rows:
                row_h = row["h"]
                # Skip drawing rows that are fully outside the visible area
                # (still advance y so layout stays in sync).
                if y + row_h < body_top or y > body_bottom:
                    y += row_h
                    continue
                kind = row["kind"]
                if kind == "section_header":
                    hs = header_font.render(row["text"], True, ink)
                    surface.blit(hs, (text_x, y))
                elif kind == "rich":
                    x = text_x
                    for font, txt in row["segs"]:
                        if not txt:
                            continue
                        seg_surf = font.render(txt, True, row["color"])
                        surface.blit(seg_surf, (x, y))
                        x += seg_surf.get_width()
                elif kind == "mono":
                    ms = md_fonts["mono"].render(row["text"], True, ink)
                    surface.blit(ms, (text_x, y))
                # vspace: nothing to draw, just advance.
                y += row_h
        finally:
            surface.set_clip(prior_clip)

        # Scrollbar.
        if self._room_info_max_scroll > 0:
            track_w = 6
            track_x = body_rect.right - track_w - 2
            track_rect = pygame.Rect(track_x, body_top, track_w, body_view_h)
            pygame.draw.rect(surface, (214, 202, 168), track_rect,
                             border_radius=3)
            thumb_h = max(28, int(body_view_h * body_view_h / max(1, content_h)))
            travel = body_view_h - thumb_h
            thumb_y = body_top + int(travel * (
                self._room_info_scroll_y / self._room_info_max_scroll
            ))
            thumb_rect = pygame.Rect(track_x, thumb_y, track_w, thumb_h)
            pygame.draw.rect(surface, ink, thumb_rect, border_radius=3)

    def _handle_room_info_click(self, pos: tuple[int, int]) -> None:
        """Close-button / outside-panel click handling. Always swallows the
        click — the caller has already gated on _room_info_open."""
        if (self._room_info_close_rect is not None
                and self._room_info_close_rect.collidepoint(pos)):
            self.close_room_info()
            return
        if (self._room_info_panel_rect is not None
                and not self._room_info_panel_rect.collidepoint(pos)):
            self.close_room_info()
            return
        # Click inside the panel (but not on [×]) — keep it open.

    def close_room_info(self) -> None:
        self._room_info_open = False
        self._room_info_room_id = None
        self._room_info_scroll_y = 0.0
        self._room_info_max_scroll = 0.0

    # -- Startup warning overlay --------------------------------------------

    def show_startup_warnings(self, lines: list[str]) -> None:
        """Queue a one-shot warning panel. Visible until the user
        dismisses it with any keypress or click. Empty list = no
        overlay (the happy path)."""
        self._startup_warning_lines = list(lines)

    def _dismiss_startup_warnings(self) -> None:
        self._startup_warning_lines = []
        self._startup_warning_panel_rect = None

    @property
    def has_startup_warnings(self) -> bool:
        return bool(self._startup_warning_lines)

    def _draw_startup_warnings(self, surface: pygame.Surface) -> None:
        """Centered parchment panel listing each warning. Dim backdrop
        behind it so it reads as modal."""
        self._startup_warning_panel_rect = None
        if not self._startup_warning_lines:
            return
        ink = (26, 26, 26)
        muted = (130, 110, 80)
        title_font = self.title_font
        body_font = pygame.font.SysFont("georgia,serif", 14)
        hint_font = self.help_font

        sw, sh = surface.get_size()
        panel_w = min(560, sw - 60)
        text_max_w = panel_w - 48

        # Wrap each warning, plus a leading "⚠ " glyph on the first line.
        wrapped: list[tuple[str, list[str]]] = []
        line_h = body_font.get_linesize()
        body_h = 0
        for w in self._startup_warning_lines:
            lines = self._wrap_text("⚠  " + w, body_font, text_max_w)
            wrapped.append((w, lines))
            body_h += line_h * len(lines) + 10  # paragraph gap

        title_h = 50
        bottom_pad = 56  # room for the dismiss hint
        panel_h = title_h + body_h + bottom_pad
        panel_x = (sw - panel_w) // 2
        panel_y = max(60, (sh - panel_h) // 2)
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

        # Dim backdrop.
        backdrop = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        backdrop.fill((0, 0, 0, 130))
        surface.blit(backdrop, (0, 0))

        # Panel.
        panel_bg = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel_bg.fill((244, 228, 193, 250))
        surface.blit(panel_bg, panel_rect.topleft)
        pygame.draw.rect(surface, ink, panel_rect, 3, border_radius=6)
        self._startup_warning_panel_rect = panel_rect

        # Title.
        title_surf = title_font.render("Heads up", True, ink)
        surface.blit(title_surf, (panel_x + 24, panel_y + 14))
        sep_y = panel_y + title_h
        pygame.draw.line(surface, ink, (panel_x + 16, sep_y),
                         (panel_x + panel_w - 16, sep_y), 1)

        # Warnings.
        y = sep_y + 12
        for _, lines in wrapped:
            for line in lines:
                surface.blit(body_font.render(line, True, ink),
                             (panel_x + 24, y))
                y += line_h
            y += 10

        # Dismiss hint.
        hint = hint_font.render(
            "Press any key or click to dismiss.", True, muted)
        surface.blit(hint, (panel_x + 24, panel_y + panel_h - 28))

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
                live_r.special_text = fresh_r.special_text
                live_r.statblocks = fresh_r.statblocks
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
    dungeons_dir: Path | None = None,
    window_size: tuple[int, int] = (1280, 800),
    on_change: Callable[[], None] | None = None,
    on_open_browser: Callable[[], None] | None = None,
    on_open_editor: Callable[[], None] | None = None,
    on_open_player: Callable[[], None] | None = None,
    startup_warnings: list[str] | None = None,
) -> ReloadRequest | None:
    """Run the pygame event loop. on_change fires after any state change.
    The on_open_* hooks are individual tab openers used by the options
    menu; on_open_browser is the legacy "open all tabs" shortcut for V.

    Return value: None on a clean quit; a ReloadRequest if the user
    asked to switch dungeons or full-reset. main.py is expected to
    handle the reload (rebuild session + editor server, optionally run
    Session.full_reset) and call run() again with the new session. We
    deliberately do NOT pygame.quit() in the reload path so the SDL
    window survives across the reload — a fresh pygame init after
    quit() on macOS often fails to reattach to the WindowServer."""
    pygame.init()
    pygame.font.init()
    surface = pygame.display.set_mode(window_size, pygame.RESIZABLE)
    pygame.display.set_caption(f"OSR Dungeon — Editor — {session.dungeon.name}")
    clock = pygame.time.Clock()

    def _on_change() -> None:
        view.snapshot_to_disk()
        if on_change is not None:
            on_change()

    view = MapView(session,
                   dungeon_path=dungeon_path,
                   dungeons_dir=dungeons_dir,
                   on_change=_on_change)
    if startup_warnings:
        view.show_startup_warnings(startup_warnings)
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
                # Startup warning overlay swallows the next keydown to
                # dismiss itself. Skip the rest of the per-key dispatch
                # so e.g. Space doesn't also Advance Turn.
                if view.has_startup_warnings:
                    view._dismiss_startup_warnings()
                    continue
                # New Dungeon modal owns ALL keydowns while open — text
                # entry must not double as Advance Turn / hotkeys.
                if view._new_dungeon_open:
                    if view._handle_new_dungeon_keydown(event):
                        continue
                cmd = bool(event.mod & cmd_mask)
                if event.key == pygame.K_ESCAPE:
                    if view._room_info_open:
                        view.close_room_info()
                    elif view._new_dungeon_open:
                        view.close_new_dungeon_modal()
                    elif view._progress_confirm_open:
                        view._progress_confirm_open = False
                    elif view._reset_confirm_open:
                        view._reset_confirm_open = False
                    elif view._picker_open:
                        view._picker_open = False
                    elif view._options_open:
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
                    view._delete_hovered_room()
                elif cmd and event.key == pygame.K_UP:
                    _ascend()
                elif cmd and event.key == pygame.K_DOWN:
                    _descend()
                # --- Hovered-room keyboard parity (Phase 6) ---
                # Enter cycles the hovered room's reveal state, I opens
                # the info modal for it. Both are no-ops if the cursor
                # isn't over a room.
                elif (event.key == pygame.K_RETURN and not cmd
                      and not view._annot_mode
                      and view._hovered_room_id is not None):
                    view.cycle_room_state(view._hovered_room_id)
                elif (event.key == pygame.K_i and not cmd
                      and not view._annot_mode
                      and view._hovered_room_id is not None):
                    view._room_info_open = True
                    view._room_info_room_id = view._hovered_room_id
                    view._room_info_scroll_y = 0.0
                # --- Session action shortcuts (turn / resources) ---
                elif (event.key == pygame.K_SPACE and not view._annot_mode):
                    view.action_advance_turn()
                elif event.key == pygame.K_r and not cmd and not view._annot_mode:
                    view.action_manual_wm_roll()
                elif event.key == pygame.K_t and not cmd and not view._annot_mode:
                    view.action_light_torch()
                elif (event.key == pygame.K_l and not cmd
                      and not view._annot_mode):
                    shift = bool(event.mod & pygame.KMOD_SHIFT)
                    if shift:
                        view.action_refill_lantern()
                    else:
                        view.action_light_lantern()
                elif event.key == pygame.K_n and not cmd and not view._annot_mode:
                    view.action_toggle_noisy()
                else:
                    view.handle_event(event)
            else:
                view.handle_event(event)
        view.draw(surface)
        pygame.display.flip()
        clock.tick(60)
        # If a click handler queued a reload (switch dungeon / full reset),
        # exit cleanly so main.py can rebuild the world. Save first so the
        # next session sees up-to-date state.
        if view._pending_reload is not None:
            session.save()
            session.close()
            return view._pending_reload

    session.close()
    pygame.quit()
    return None
