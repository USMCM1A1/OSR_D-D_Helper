"""Graph data model and JSON loader for hand-authored or generated dungeons.

A Dungeon is a stack of self-contained Levels. Each Level owns its own
room graph, corridor graph, wandering-monster table, WM check rules,
and (per the spec) a `map_image` reference for browser rendering. Only
one level is rendered / played at a time — `Dungeon.current_level`
selects which.

Public API:
    load(path) -> Dungeon
    DungeonValidationError

The loader validates structure aggressively and raises
DungeonValidationError with a clear message on any issue.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config


class DungeonValidationError(ValueError):
    """Raised when a dungeon JSON file fails schema or graph validation."""


# --- Dataclasses -------------------------------------------------------------


@dataclass
class Character:
    """Per-character runtime state.

    `name` and `darkvision` are stable for the session; `exhaustion` is
    mutated by the DM as conditions warrant (PHB p. 291).
    """
    name: str
    darkvision: bool = False
    exhaustion: int = 0


@dataclass(frozen=True)
class Party:
    size: int
    characters: tuple[Character, ...]


@dataclass(frozen=True)
class WMTableEntry:
    roll: int
    encounter: str


@dataclass
class ImageRegion:
    """Reveal region for a room — either an axis-aligned rectangle or a
    polygon, in image-pixel coordinates.

    `kind == "rect"`    → use `rect = (x, y, w, h)`
    `kind == "polygon"` → use `points = ((x, y), (x, y), …)`
    """
    kind: str
    rect: tuple[int, int, int, int] | None = None
    points: tuple[tuple[int, int], ...] | None = None

    def centroid(self) -> tuple[int, int]:
        if self.kind == "rect" and self.rect is not None:
            x, y, w, h = self.rect
            return (x + w // 2, y + h // 2)
        if self.kind == "polygon" and self.points:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            return (sum(xs) // len(xs), sum(ys) // len(ys))
        return (0, 0)

    def to_dict(self) -> dict:
        if self.kind == "rect":
            assert self.rect is not None
            x, y, w, h = self.rect
            return {"kind": "rect", "x": int(x), "y": int(y),
                    "width": int(w), "height": int(h)}
        if self.kind == "polygon":
            assert self.points is not None
            return {"kind": "polygon",
                    "points": [[int(p[0]), int(p[1])] for p in self.points]}
        raise ValueError(f"unknown region kind: {self.kind!r}")


@dataclass
class Room:
    id: str
    name: str
    state: str
    tags: tuple[str, ...]
    reaction_required: bool = False
    notes: str = ""
    encounter_ref: str | None = None
    treasure_tier: str | None = None
    image_region: ImageRegion | None = None
    # Free-form narrative content edited via the browser-tab room editor.
    # All default to "" so legacy JSON without these keys still loads.
    box_text: str = ""             # read aloud to players
    encounter_text: str = ""       # encounter / monsters / tactics
    treasure_text: str = ""        # treasure / loot details


@dataclass(frozen=True)
class Corridor:
    src: str          # room id (`from` is a Python keyword)
    dst: str          # room id
    distance_ft: int
    tags: tuple[str, ...] = ()


@dataclass
class Level:
    """A self-contained dungeon level with its own graph and WM rules."""
    level_number: int
    display_name: str
    map_image: str
    map_image_scale: float
    wm_check_method: str
    wm_check_threshold: int
    wm_check_frequency: str
    wandering_monster_table: tuple[WMTableEntry, ...]
    rooms: tuple[Room, ...]
    corridors: tuple[Corridor, ...]
    rooms_by_id: dict[str, Room] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rooms_by_id:
            self.rooms_by_id = {r.id: r for r in self.rooms}

    def neighbors(self, room_id: str) -> list[str]:
        """Room IDs reachable in one step from `room_id`. Honors `one-way`."""
        out: list[str] = []
        for c in self.corridors:
            if c.src == room_id:
                out.append(c.dst)
            elif c.dst == room_id and "one-way" not in c.tags:
                out.append(c.src)
        return out

    def stairs_up_room_id(self) -> str | None:
        """First room tagged `stairs_up` on this level, if any."""
        for r in self.rooms:
            if "stairs_up" in r.tags:
                return r.id
        return None

    def stairs_down_room_id(self) -> str | None:
        for r in self.rooms:
            if "stairs_down" in r.tags:
                return r.id
        return None


@dataclass
class Dungeon:
    name: str
    party_level: int
    party: Party
    levels: tuple[Level, ...]
    current_level: int
    levels_by_number: dict[int, Level] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.levels_by_number:
            self.levels_by_number = {lv.level_number: lv for lv in self.levels}

    @property
    def current(self) -> Level:
        """The currently active level."""
        return self.levels_by_number[self.current_level]

    def get_level(self, level_number: int) -> Level:
        return self.levels_by_number[level_number]

    @property
    def deepest_level_number(self) -> int:
        return max(lv.level_number for lv in self.levels)

    @property
    def shallowest_level_number(self) -> int:
        return min(lv.level_number for lv in self.levels)


# --- Loader ------------------------------------------------------------------


def load(path: str | Path) -> Dungeon:
    """Load a dungeon JSON file, validate, and return a Dungeon."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise DungeonValidationError(f"{p}: invalid JSON: {e.msg} (line {e.lineno})") from e
    return _from_dict(raw, source=str(p))


def dump(dungeon: Dungeon, path: str | Path) -> None:
    """Serialize a Dungeon back to JSON. Used by the annotation editor to
    persist drawn room regions to disk so they survive across sessions."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_to_dict(dungeon), indent=2) + "\n")


def _to_dict(d: Dungeon) -> dict:
    return {
        "dungeon_name": d.name,
        "party_level": d.party_level,
        "current_level": d.current_level,
        "party": {
            "size": d.party.size,
            "characters": [
                {"name": c.name, "darkvision": c.darkvision, "exhaustion": c.exhaustion}
                for c in d.party.characters
            ],
        },
        "levels": [_level_to_dict(lv) for lv in d.levels],
    }


def _level_to_dict(lv: Level) -> dict:
    return {
        "level_number": lv.level_number,
        "display_name": lv.display_name,
        "map_image": lv.map_image,
        "map_image_scale": lv.map_image_scale,
        "wm_check_method": lv.wm_check_method,
        "wm_check_threshold": lv.wm_check_threshold,
        "wm_check_frequency": lv.wm_check_frequency,
        "wandering_monster_table": [
            {"roll": e.roll, "encounter": e.encounter}
            for e in lv.wandering_monster_table
        ],
        "rooms": [_room_to_dict(r) for r in lv.rooms],
        "corridors": [
            {"from": c.src, "to": c.dst, "distance_ft": c.distance_ft,
             "tags": list(c.tags)}
            for c in lv.corridors
        ],
    }


def _room_to_dict(r: Room) -> dict:
    out: dict = {
        "id": r.id,
        "name": r.name,
        "state": r.state,
        "tags": list(r.tags),
        "reaction_required": r.reaction_required,
        "notes": r.notes,
        "encounter_ref": r.encounter_ref,
        "treasure_tier": r.treasure_tier,
        "box_text": r.box_text,
        "encounter_text": r.encounter_text,
        "treasure_text": r.treasure_text,
    }
    if r.image_region is not None:
        out["image_region"] = r.image_region.to_dict()
    return out


def _from_dict(raw: Any, source: str = "<dict>") -> Dungeon:
    if not isinstance(raw, dict):
        raise DungeonValidationError(f"{source}: top level must be an object")

    required = ("dungeon_name", "party_level", "current_level", "party", "levels")
    missing = [k for k in required if k not in raw]
    if missing:
        raise DungeonValidationError(f"{source}: missing required field(s): {', '.join(missing)}")

    name = _require_str(raw["dungeon_name"], "dungeon_name", source)
    party_level = _require_int(raw["party_level"], "party_level", source)
    if party_level < 1:
        raise DungeonValidationError(f"{source}: party_level must be >= 1, got {party_level}")
    current_level = _require_int(raw["current_level"], "current_level", source)
    party = _parse_party(raw["party"], source)

    levels_raw = raw["levels"]
    if not isinstance(levels_raw, list) or not levels_raw:
        raise DungeonValidationError(f"{source}: levels must be a non-empty list")

    levels: list[Level] = []
    seen_numbers: set[int] = set()
    for i, lv_raw in enumerate(levels_raw):
        level = _parse_level(lv_raw, i, source)
        if level.level_number in seen_numbers:
            raise DungeonValidationError(
                f"{source}: duplicate level_number {level.level_number}"
            )
        seen_numbers.add(level.level_number)
        levels.append(level)

    if current_level not in seen_numbers:
        raise DungeonValidationError(
            f"{source}: current_level {current_level} not present in levels[] "
            f"(have {sorted(seen_numbers)})"
        )

    return Dungeon(
        name=name,
        party_level=party_level,
        party=party,
        levels=tuple(levels),
        current_level=current_level,
    )


# --- Section parsers ---------------------------------------------------------


def _parse_level(raw: Any, idx: int, source: str) -> Level:
    if not isinstance(raw, dict):
        raise DungeonValidationError(f"{source}: levels[{idx}] must be an object")

    required = (
        "level_number", "display_name", "map_image", "map_image_scale",
        "wm_check_method", "wm_check_threshold", "wm_check_frequency",
        "wandering_monster_table", "rooms", "corridors",
    )
    missing = [k for k in required if k not in raw]
    if missing:
        raise DungeonValidationError(
            f"{source}: levels[{idx}] missing required field(s): {', '.join(missing)}"
        )

    level_number = _require_int(raw["level_number"], f"levels[{idx}].level_number", source)
    display_name = _require_str(raw["display_name"], f"levels[{idx}].display_name", source)
    map_image = _require_str(raw["map_image"], f"levels[{idx}].map_image", source)
    scale_raw = raw["map_image_scale"]
    if isinstance(scale_raw, bool) or not isinstance(scale_raw, (int, float)):
        raise DungeonValidationError(
            f"{source}: levels[{idx}].map_image_scale must be a number"
        )
    map_image_scale = float(scale_raw)

    method = _require_str(raw["wm_check_method"], f"levels[{idx}].wm_check_method", source)
    if method not in config.WM_METHODS:
        raise DungeonValidationError(
            f"{source}: levels[{idx}].wm_check_method must be one of {config.WM_METHODS}, "
            f"got {method!r}"
        )
    threshold = _require_int(raw["wm_check_threshold"], f"levels[{idx}].wm_check_threshold", source)
    frequency = _require_str(raw["wm_check_frequency"], f"levels[{idx}].wm_check_frequency", source)
    if frequency not in config.WM_FREQUENCIES:
        raise DungeonValidationError(
            f"{source}: levels[{idx}].wm_check_frequency must be one of {config.WM_FREQUENCIES}, "
            f"got {frequency!r}"
        )

    wm_table = _parse_wm_table(raw["wandering_monster_table"], f"levels[{idx}]", source)
    rooms = _parse_rooms(raw["rooms"], f"levels[{idx}]", source)
    corridors = _parse_corridors(raw["corridors"], rooms, f"levels[{idx}]", source)
    _validate_graph(rooms, corridors, f"{source} levels[{idx}]")

    return Level(
        level_number=level_number,
        display_name=display_name,
        map_image=map_image,
        map_image_scale=map_image_scale,
        wm_check_method=method,
        wm_check_threshold=threshold,
        wm_check_frequency=frequency,
        wandering_monster_table=wm_table,
        rooms=rooms,
        corridors=corridors,
    )


def _parse_wm_table(raw: Any, prefix: str, source: str) -> tuple[WMTableEntry, ...]:
    if not isinstance(raw, list) or not raw:
        raise DungeonValidationError(
            f"{source}: {prefix}.wandering_monster_table must be a non-empty list"
        )
    out: list[WMTableEntry] = []
    seen_rolls: set[int] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or "roll" not in entry or "encounter" not in entry:
            raise DungeonValidationError(
                f"{source}: {prefix}.wandering_monster_table[{i}] must have 'roll' and 'encounter'"
            )
        roll = _require_int(entry["roll"], f"{prefix}.wandering_monster_table[{i}].roll", source)
        if roll in seen_rolls:
            raise DungeonValidationError(
                f"{source}: duplicate roll {roll} in {prefix}.wandering_monster_table"
            )
        seen_rolls.add(roll)
        encounter = _require_str(
            entry["encounter"], f"{prefix}.wandering_monster_table[{i}].encounter", source,
        )
        out.append(WMTableEntry(roll=roll, encounter=encounter))
    return tuple(out)


def _parse_party(raw: Any, source: str) -> Party:
    if not isinstance(raw, dict) or "size" not in raw or "characters" not in raw:
        raise DungeonValidationError(f"{source}: party must have 'size' and 'characters'")
    size = _require_int(raw["size"], "party.size", source)
    chars_raw = raw["characters"]
    if not isinstance(chars_raw, list):
        raise DungeonValidationError(f"{source}: party.characters must be a list")
    if len(chars_raw) != size:
        raise DungeonValidationError(
            f"{source}: party.size ({size}) does not match len(party.characters) ({len(chars_raw)})"
        )
    chars: list[Character] = []
    seen_names: set[str] = set()
    for i, c in enumerate(chars_raw):
        if not isinstance(c, dict) or "name" not in c:
            raise DungeonValidationError(f"{source}: party.characters[{i}] must have 'name'")
        cname = _require_str(c["name"], f"party.characters[{i}].name", source)
        if cname in seen_names:
            raise DungeonValidationError(f"{source}: duplicate character name {cname!r}")
        seen_names.add(cname)
        darkvision = bool(c.get("darkvision", False))
        exhaustion = int(c.get("exhaustion", 0))
        if not (config.EXHAUSTION_MIN <= exhaustion <= config.EXHAUSTION_MAX):
            raise DungeonValidationError(
                f"{source}: {cname} exhaustion {exhaustion} out of range "
                f"[{config.EXHAUSTION_MIN}, {config.EXHAUSTION_MAX}]"
            )
        chars.append(Character(name=cname, darkvision=darkvision, exhaustion=exhaustion))
    return Party(size=size, characters=tuple(chars))


def _parse_rooms(raw: Any, prefix: str, source: str) -> tuple[Room, ...]:
    """Parse the rooms[] array. Empty is allowed — a level can start with
    zero annotated rooms; the DM will draw them in annotation mode."""
    if not isinstance(raw, list):
        raise DungeonValidationError(f"{source}: {prefix}.rooms must be a list")
    rooms: list[Room] = []
    seen_ids: set[str] = set()
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            raise DungeonValidationError(f"{source}: {prefix}.rooms[{i}] must be an object")
        for k in ("id", "name", "state", "tags"):
            if k not in r:
                raise DungeonValidationError(
                    f"{source}: {prefix}.rooms[{i}] missing field {k!r}"
                )
        rid = _require_str(r["id"], f"{prefix}.rooms[{i}].id", source)
        if rid in seen_ids:
            raise DungeonValidationError(f"{source}: duplicate room id {rid!r} in {prefix}")
        seen_ids.add(rid)
        rname = _require_str(r["name"], f"{prefix}.rooms[{i}].name", source)
        state = _require_str(r["state"], f"{prefix}.rooms[{i}].state", source)
        if state not in config.ROOM_STATES:
            raise DungeonValidationError(
                f"{source}: room {rid!r} state must be one of {config.ROOM_STATES}, got {state!r}"
            )
        tags_raw = r["tags"]
        if not isinstance(tags_raw, list) or not tags_raw:
            raise DungeonValidationError(
                f"{source}: room {rid!r} tags must be a non-empty list"
            )
        for t in tags_raw:
            if t not in config.ROOM_TAGS:
                raise DungeonValidationError(
                    f"{source}: room {rid!r} has unknown tag {t!r}; "
                    f"allowed: {config.ROOM_TAGS}"
                )
        region = _parse_image_region(r.get("image_region"), rid, source)
        rooms.append(Room(
            id=rid,
            name=rname,
            state=state,
            tags=tuple(tags_raw),
            reaction_required=bool(r.get("reaction_required", False)),
            notes=str(r.get("notes", "")),
            encounter_ref=r.get("encounter_ref"),
            treasure_tier=r.get("treasure_tier"),
            image_region=region,
            box_text=str(r.get("box_text", "")),
            encounter_text=str(r.get("encounter_text", "")),
            treasure_text=str(r.get("treasure_text", "")),
        ))
    return tuple(rooms)


def _parse_image_region(raw: Any, room_id: str, source: str) -> ImageRegion | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise DungeonValidationError(
            f"{source}: room {room_id!r} image_region must be an object"
        )
    kind = raw.get("kind")
    if kind == "rect":
        for k in ("x", "y", "width", "height"):
            if k not in raw:
                raise DungeonValidationError(
                    f"{source}: room {room_id!r} rect image_region missing {k!r}"
                )
        return ImageRegion(
            kind="rect",
            rect=(int(raw["x"]), int(raw["y"]),
                  int(raw["width"]), int(raw["height"])),
        )
    if kind == "polygon":
        pts = raw.get("points")
        if not isinstance(pts, list) or len(pts) < 3:
            raise DungeonValidationError(
                f"{source}: room {room_id!r} polygon image_region needs ≥3 points"
            )
        parsed_pts: list[tuple[int, int]] = []
        for i, pt in enumerate(pts):
            if (not isinstance(pt, (list, tuple)) or len(pt) != 2):
                raise DungeonValidationError(
                    f"{source}: room {room_id!r} polygon point[{i}] must be [x, y]"
                )
            parsed_pts.append((int(pt[0]), int(pt[1])))
        return ImageRegion(kind="polygon", points=tuple(parsed_pts))
    raise DungeonValidationError(
        f"{source}: room {room_id!r} image_region.kind must be 'rect' or 'polygon'"
    )


def _parse_corridors(raw: Any, rooms: tuple[Room, ...], prefix: str, source: str) -> tuple[Corridor, ...]:
    if not isinstance(raw, list):
        raise DungeonValidationError(f"{source}: {prefix}.corridors must be a list")
    room_ids = {r.id for r in rooms}
    out: list[Corridor] = []
    for i, c in enumerate(raw):
        if not isinstance(c, dict):
            raise DungeonValidationError(f"{source}: {prefix}.corridors[{i}] must be an object")
        for k in ("from", "to", "distance_ft"):
            if k not in c:
                raise DungeonValidationError(
                    f"{source}: {prefix}.corridors[{i}] missing field {k!r}"
                )
        src = _require_str(c["from"], f"{prefix}.corridors[{i}].from", source)
        dst = _require_str(c["to"], f"{prefix}.corridors[{i}].to", source)
        if src not in room_ids:
            raise DungeonValidationError(
                f"{source}: {prefix}.corridors[{i}] references unknown room id {src!r} (from)"
            )
        if dst not in room_ids:
            raise DungeonValidationError(
                f"{source}: {prefix}.corridors[{i}] references unknown room id {dst!r} (to)"
            )
        if src == dst:
            raise DungeonValidationError(
                f"{source}: {prefix}.corridors[{i}] has from==to ({src!r}); self-loops not allowed"
            )
        distance = _require_int(c["distance_ft"], f"{prefix}.corridors[{i}].distance_ft", source)
        if distance <= 0:
            raise DungeonValidationError(
                f"{source}: {prefix}.corridors[{i}] distance_ft must be > 0, got {distance}"
            )
        tags_raw = c.get("tags", []) or []
        if not isinstance(tags_raw, list):
            raise DungeonValidationError(f"{source}: {prefix}.corridors[{i}].tags must be a list")
        for t in tags_raw:
            if t not in config.CORRIDOR_TAGS:
                raise DungeonValidationError(
                    f"{source}: {prefix}.corridors[{i}] has unknown tag {t!r}; "
                    f"allowed: {config.CORRIDOR_TAGS}"
                )
        out.append(Corridor(src=src, dst=dst, distance_ft=distance, tags=tuple(tags_raw)))
    return tuple(out)


# --- Graph validation --------------------------------------------------------


def _validate_graph(
    rooms: tuple[Room, ...],
    corridors: tuple[Corridor, ...],
    source: str,
) -> None:
    """Validate that every corridor's endpoints exist in `rooms`.

    Strict graph connectivity used to be enforced here, but with auto-
    derived dungeons (rooms come from image segmentation) the corridor
    edge list is best-effort — connectivity is implicit in the image.
    Orphan rooms are allowed; the runtime can still reveal them
    independently by clicking the corresponding region.
    """
    room_ids = {r.id for r in rooms}
    for c in corridors:
        if c.src not in room_ids or c.dst not in room_ids:
            raise DungeonValidationError(
                f"{source}: corridor {c.src} → {c.dst} references an unknown room"
            )


# --- Small helpers -----------------------------------------------------------


def _require_str(v: Any, field_name: str, source: str) -> str:
    if not isinstance(v, str) or not v:
        raise DungeonValidationError(f"{source}: {field_name} must be a non-empty string")
    return v


def _require_int(v: Any, field_name: str, source: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise DungeonValidationError(f"{source}: {field_name} must be an integer")
    return v
