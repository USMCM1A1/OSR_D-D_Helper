"""Pure-Python layout utilities — no pygame dependency.

Operates on a single Level (one floor of a dungeon). Both the pygame
editor (renderer.py) and the SVG renderer (svg_render.py) call these
helpers to compute room positions for the *current* level.
"""

from __future__ import annotations

from collections import defaultdict, deque

from dungeon import Level


def auto_layout(
    level: Level,
    *,
    x_spacing: float = 320.0,
    y_spacing: float = 220.0,
    x0: float = 220.0,
    y0: float = 220.0,
) -> dict[str, tuple[float, float]]:
    """BFS-tree level layout for one Level. Deterministic given the Level."""
    if not level.rooms:
        return {}
    adj: dict[str, list[str]] = {r.id: [] for r in level.rooms}
    for c in level.corridors:
        adj[c.src].append(c.dst)
        adj[c.dst].append(c.src)
    levels: dict[str, int] = {level.rooms[0].id: 0}
    queue: deque[str] = deque([level.rooms[0].id])
    while queue:
        rid = queue.popleft()
        for nid in adj[rid]:
            if nid not in levels:
                levels[nid] = levels[rid] + 1
                queue.append(nid)
    by_level: dict[int, list[str]] = defaultdict(list)
    for rid, lvl in levels.items():
        by_level[lvl].append(rid)
    positions: dict[str, tuple[float, float]] = {}
    for lvl in sorted(by_level):
        ids = sorted(by_level[lvl])
        for i, rid in enumerate(ids):
            positions[rid] = (x0 + lvl * x_spacing, y0 + i * y_spacing)
    return positions


def effective_positions(session, level_number: int | None = None) -> dict[str, tuple[float, float]]:
    """Return saved session positions for `level_number` (defaults to the
    session's current level), falling back to auto_layout for unset rooms."""
    if level_number is None:
        level_number = session.current_level
    level = session.dungeon.levels_by_number[level_number]
    saved = session.get_room_positions(level_number)
    positions = auto_layout(level)
    for rid, xy in saved.items():
        if xy is not None:
            positions[rid] = xy
    return positions
