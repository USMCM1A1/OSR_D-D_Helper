"""Project-wide constants for the OSR Dungeon System.

Values are sourced from the 5E PHB and DMG; see CLAUDE.md for citations.
This module has no runtime dependencies and may be imported by any other module.
"""

from __future__ import annotations

# --- Time --------------------------------------------------------------------

# 1 dungeon turn = 10 minutes of in-game time (PHB p. 182).
TURN_MINUTES = 10

SHORT_REST_TURNS = 6   # PHB p. 186, 1-hour minimum.
LONG_REST_TURNS = 48   # PHB p. 186, 8 hours.

# --- Light sources (PHB p. 183) ----------------------------------------------

# Duration in turns for each light source (one fill / one item).
LIGHT_DURATIONS_TURNS: dict[str, int] = {
    "torch": 6,
    "candle": 6,
    "hooded_lantern": 36,
    "bullseye_lantern": 36,
}

# Bright / dim light radii in feet for each light source.
LIGHT_RADII_FT: dict[str, tuple[int, int]] = {
    "torch":            (20, 20),
    "candle":           (5,  5),
    "hooded_lantern":   (30, 30),
    "bullseye_lantern": (60, 60),  # cone, not sphere — renderer handles.
}

LIGHT_LOW_WARNING_TURNS = 2  # show ⚠ when this many turns or fewer remain.

# --- Wandering monsters (DMG p. 82) -----------------------------------------

WM_METHOD_D20 = "d20"
WM_METHOD_D6 = "d6"
WM_METHODS = (WM_METHOD_D20, WM_METHOD_D6)

# Default thresholds — overridable per-dungeon in the JSON.
WM_DEFAULT_D20_THRESHOLD = 18  # encounter on 18+ (i.e. 18, 19, 20).
WM_DEFAULT_D6_THRESHOLD = 1    # encounter on a 1.

# --- Exhaustion (PHB p. 291) -------------------------------------------------

EXHAUSTION_MIN = 0
EXHAUSTION_MAX = 6
EXHAUSTION_DANGER = 3  # red highlight at this level or higher.

# --- Party-level CR scaling (CLAUDE.md "Party Level Scaling") ---------------

# (low_cr, high_cr) inclusive band of standard-encounter CRs by party level.
# Use float for fractional CRs (1/8, 1/4, 1/2 → 0.125, 0.25, 0.5).
CR_BANDS_BY_PARTY_LEVEL: dict[tuple[int, int], tuple[float, float]] = {
    (1, 2):  (0.125, 0.5),
    (3, 4):  (0.25,  1.0),
    (5, 6):  (0.5,   2.0),
    (7, 8):  (1.0,   3.0),
    (9, 10): (2.0,   5.0),
    (11, 99):(4.0,   8.0),
}

# (low_cr, high_cr) inclusive band of "deadly room" CRs by party level.
DEADLY_CR_BANDS_BY_PARTY_LEVEL: dict[tuple[int, int], tuple[float, float]] = {
    (1, 2):  (1.0,  1.0),
    (3, 4):  (2.0,  3.0),
    (5, 6):  (4.0,  5.0),
    (7, 8):  (6.0,  7.0),
    (9, 10): (8.0,  10.0),
    (11, 99):(10.0, 99.0),
}

# DMG individual treasure tier labels — referenced by room.treasure_tier.
TREASURE_TIERS = ("cr0-4", "cr5-10", "cr11-16", "cr17+")

# --- Schema-valid value sets -------------------------------------------------

ROOM_STATES = ("unexplored", "known", "cleared")

ROOM_TAGS = ("encounter", "trap", "treasure", "special", "empty",
             "stairs_up", "stairs_down")

CORRIDOR_TAGS = ("secret", "locked", "trapped", "one-way")

WM_FREQUENCIES = ("every_turn",)  # extension point; only one for now.


def cr_band_for_party_level(party_level: int) -> tuple[float, float]:
    """Return the (low, high) standard CR band for a given party level."""
    for (lo, hi), band in CR_BANDS_BY_PARTY_LEVEL.items():
        if lo <= party_level <= hi:
            return band
    raise ValueError(f"party_level {party_level} out of range")


def deadly_cr_band_for_party_level(party_level: int) -> tuple[float, float]:
    """Return the (low, high) deadly-room CR band for a given party level."""
    for (lo, hi), band in DEADLY_CR_BANDS_BY_PARTY_LEVEL.items():
        if lo <= party_level <= hi:
            return band
    raise ValueError(f"party_level {party_level} out of range")
