"""Structured extraction from SRD monster statblock markdown.

`srd_lookup.find()` returns a `StatBlock(name, body)` whose body is the
hand-curated markdown lifted from the SRD source files. The encounter
simulator needs *numbers* — AC, HP, attack bonuses, damage dice, save
DCs — so this module regex-parses the markdown into a `ParsedMonster`.

The format is consistent enough across the SRD that a half-dozen
patterns cover ~all attack lines we care about. Anything we can't parse
is silently dropped from the attacks list rather than raising — a
half-described monster is still a usable opponent (it just attacks less
often). The parser does NOT model conditions (paralysis, frightened,
etc.) at the simulation level; rider clauses on attacks are captured as
free text on `MonsterAttack.rider_text` for the trace, but the
simulator ignores their mechanical effect.

Also exports `roll_dice` and `roll_count` because the wandering-monster
table strings ("2d6 Skeletons") need the same NdN±M parsing and no
existing module provides it.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

import srd_lookup


# ---------------------------------------------------------------------------
# Dice parsing
# ---------------------------------------------------------------------------

# Matches NdN with an optional ± modifier. Whitespace around the operator
# is tolerated because the SRD writes "2d6 + 2" but our own callers will
# pass "2d6+2" — we accept both.
_DICE_RE = re.compile(
    r"(?P<count>\d+)\s*d\s*(?P<sides>\d+)\s*(?:(?P<sign>[+-])\s*(?P<mod>\d+))?",
    re.IGNORECASE,
)


def roll_dice(expr: str, rng: random.Random) -> int:
    """Roll one NdN±M expression and return the total. Empty/static
    expressions like "5" return that integer; "0" returns 0.

    Negative totals clamp to 0 so a damage roll can never heal.
    """
    s = expr.strip()
    if not s:
        return 0
    m = _DICE_RE.search(s)
    if not m:
        # Plain integer fallback.
        try:
            return max(0, int(s))
        except ValueError:
            return 0
    count = int(m.group("count"))
    sides = int(m.group("sides"))
    mod = 0
    if m.group("sign"):
        mod = int(m.group("mod")) * (1 if m.group("sign") == "+" else -1)
    total = sum(rng.randint(1, sides) for _ in range(count)) + mod
    return max(0, total)


_COUNT_PREFIX_RE = re.compile(r"^\s*(\d+d\d+(?:\s*[+-]\s*\d+)?|\d+)", re.IGNORECASE)


def roll_count(expr: str, rng: random.Random) -> int:
    """Roll the leading dice/integer count from a WM-table style string,
    e.g. "2d6 Skeletons" → 2-12, "Ghoul" → 1, "Shadows (1d3)" → first
    matched count. Always returns at least 1 so a "monster is present"
    string never spawns zero combatants."""
    if not expr:
        return 1
    # Try explicit parenthesised dice first ("Shadows (1d3)").
    paren = re.search(r"\((\d+d\d+(?:\s*[+-]\s*\d+)?)\)", expr, re.IGNORECASE)
    if paren:
        n = roll_dice(paren.group(1), rng)
        return max(1, n)
    m = _COUNT_PREFIX_RE.match(expr)
    if not m:
        return 1
    n = roll_dice(m.group(1), rng)
    return max(1, n)


# ---------------------------------------------------------------------------
# Parsed monster types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonsterAttack:
    name: str
    to_hit: int
    damage_dice: str           # "2d6+2"; suitable for roll_dice
    damage_type: str           # "piercing", "slashing", "bludgeoning", ...
    range: str                 # "melee" | "ranged"
    save: dict | None = None   # {"ability": "con", "dc": 10, "half_on_save": False}
    rider_text: str = ""       # free-text descriptor of secondary effects (paralyze, etc.)


@dataclass(frozen=True)
class ParsedMonster:
    name: str
    ac: int
    hp_avg: int                 # the parenthesised average — what we use as starting HP
    hp_dice: str                # "5d8" or "2d8+4"
    speed: int                  # walking speed in ft.
    cr: str                     # "1", "1/4", "1/8", "9", ...
    creature_type: str          # "undead" | "humanoid" | "fiend" | ... (for paladin smite logic)
    attacks: tuple[MonsterAttack, ...]
    multiattack: str = ""       # raw multiattack prose ("makes two morningstar attacks")
    traits: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Body-level field patterns
# ---------------------------------------------------------------------------

_AC_RE = re.compile(r"\*\*Armor Class\*\*\s*(\d+)")
_HP_RE = re.compile(
    r"\*\*Hit Points\*\*\s*(?P<avg>\d+)\s*\((?P<dice>[^)]+)\)"
)
_SPEED_RE = re.compile(r"\*\*Speed\*\*\s*(\d+)\s*ft")
_CR_RE = re.compile(r"\*\*Challenge\*\*\s*([\d/]+)")
# Type line: "_Medium undead, chaotic evil_" → captures "undead". The size
# tokens are fixed; everything after the size up to the next comma is the
# creature type (sometimes followed by a parenthesised tag like "(orc)"
# which we strip).
_TYPE_RE = re.compile(
    r"_(?:Tiny|Small|Medium|Large|Huge|Gargantuan)\s+([a-z]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Action patterns
# ---------------------------------------------------------------------------

# An attack action line example (Ghoul):
#   _**Bite.**_ _Melee Weapon Attack:_ +2 to hit, reach 5 ft., one creature.
#   _Hit:_ 9 (2d6 + 2) piercing damage.
_ACTION_HEADER_RE = re.compile(
    r"_\*\*(?P<name>[^.*]+)\.\*\*_\s+"
    r"_(?P<kind>Melee|Ranged|Melee or Ranged)\s+(?:Weapon|Spell)\s+Attack:_\s+"
    r"\+(?P<to_hit>\d+)\s+to\s+hit,",
    re.IGNORECASE,
)
_HIT_RE = re.compile(
    r"_Hit:_\s*\d+\s*\((?P<dice>[^)]+)\)\s+(?P<dtype>[a-z]+)\s+damage\.?",
    re.IGNORECASE,
)
# Save rider on an attack ("DC 10 Constitution saving throw or be paralyzed
# for 1 minute"). We capture the DC, the ability, and the rider text as a
# free-form description for the combat trace.
_SAVE_RIDER_RE = re.compile(
    r"DC\s+(?P<dc>\d+)\s+(?P<ability>Strength|Dexterity|Constitution|"
    r"Intelligence|Wisdom|Charisma)\s+saving throw",
    re.IGNORECASE,
)
_MULTIATTACK_RE = re.compile(
    r"_\*\*Multiattack\.\*\*_\s+(?P<text>[^_]+?)(?=_\*\*|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_ABILITY_TO_KEY = {
    "strength": "str", "dexterity": "dex", "constitution": "con",
    "intelligence": "int", "wisdom": "wis", "charisma": "cha",
}


# Multiattack prose uses English number words rather than digits. We
# stop at six because no SRD multiattack goes higher; if a future
# monster does we'll fall back to digit parsing or default 1.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
}


def parse_multiattack_count(text: str) -> int:
    """Pull the attack count from a multiattack prose line.

    Examples:
      'The giant makes two morningstar attacks.' → 2
      'The lizardfolk makes two melee attacks, each with a different
       weapon.' → 2
      'X makes 3 attacks: one with Y and two with Z.' → 3 (total)

    Default 1 if nothing parses — that keeps a malformed statblock
    runnable as a single-attack monster rather than crashing."""
    if not text:
        return 1
    t = text.lower()
    # English number words: "makes two/three/... attacks".
    m = re.search(r"\bmakes\s+(\w+)\s+", t)
    if m:
        word = m.group(1)
        if word in _NUMBER_WORDS:
            return _NUMBER_WORDS[word]
        if word.isdigit():
            try:
                return max(1, int(word))
            except ValueError:
                pass
    # Bare digit fallback: 'makes 3 attacks'.
    m = re.search(r"\bmakes\s+(\d+)\s+", t)
    if m:
        try:
            return max(1, int(m.group(1)))
        except ValueError:
            pass
    return 1


def select_multiattack_indices(
    attacks: tuple["MonsterAttack", ...], multiattack_text: str,
) -> tuple[int, ...]:
    """Return indices into `attacks` listing which attack to use on
    each swing of one multiattack turn.

    Three cases, tried in order:

      1. A specific attack name appears in the prose ('two morningstar
         attacks' → Morningstar × 2). The first matching attack name
         is repeated `count` times.

      2. A range qualifier appears ('two melee attacks' / 'two ranged
         attacks'). Filter attacks by range, cycle through them up to
         `count`.

      3. Generic 'makes N attacks' with no qualifier. Take the first
         N attacks from the list — this matches the SRD convention of
         listing the most-used attack first.

    Returns an empty tuple if `attacks` is empty.
    """
    if not attacks:
        return ()
    if not multiattack_text or not multiattack_text.strip():
        # No multiattack defined for this monster — caller falls back
        # to the single-action path.
        return ()
    count = parse_multiattack_count(multiattack_text)
    if count <= 0:
        return ()
    text_lower = multiattack_text.lower()

    # Case 1: specific weapon name. Skip the implicit Multiattack
    # entry (we exclude it from attacks already); look for the first
    # real attack whose name appears in the prose.
    for i, atk in enumerate(attacks):
        if atk.name.lower() in text_lower:
            return tuple([i] * count)

    # Case 2: range qualifier.
    if "melee attack" in text_lower:
        candidates = [i for i, a in enumerate(attacks) if a.range == "melee"]
        if candidates:
            # Cycle through the available melee attacks to reach count.
            return tuple((candidates * ((count // len(candidates)) + 1))[:count])
    if "ranged attack" in text_lower:
        candidates = [i for i, a in enumerate(attacks) if a.range == "ranged"]
        if candidates:
            return tuple((candidates * ((count // len(candidates)) + 1))[:count])

    # Case 3: generic — first N attacks in order.
    return tuple(range(min(count, len(attacks))))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse(statblock: srd_lookup.StatBlock) -> ParsedMonster:
    """Convert an SRD statblock body to a ParsedMonster.

    Best-effort: any field we can't extract gets a sensible default
    (AC 10, HP 1, no attacks). Garbage input produces a runnable but
    weak monster rather than a crash."""
    body = statblock.body

    ac = _first_int(_AC_RE, body, default=10)
    hp_avg, hp_dice = _parse_hp(body)
    speed = _first_int(_SPEED_RE, body, default=30)
    cr = _first_str(_CR_RE, body, default="0")
    creature_type = _first_str(_TYPE_RE, body, default="unknown").lower()

    actions_section = _slice_actions(body)
    multiattack_match = _MULTIATTACK_RE.search(actions_section)
    multiattack = multiattack_match.group("text").strip() if multiattack_match else ""

    attacks = _parse_attacks(actions_section)

    return ParsedMonster(
        name=statblock.name,
        ac=ac,
        hp_avg=hp_avg,
        hp_dice=hp_dice,
        speed=speed,
        cr=cr,
        creature_type=creature_type,
        attacks=tuple(attacks),
        multiattack=multiattack,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _first_int(pat: re.Pattern, text: str, *, default: int) -> int:
    m = pat.search(text)
    if not m:
        return default
    try:
        return int(m.group(1))
    except (ValueError, IndexError):
        return default


def _first_str(pat: re.Pattern, text: str, *, default: str) -> str:
    m = pat.search(text)
    return m.group(1).strip() if m else default


def _parse_hp(body: str) -> tuple[int, str]:
    m = _HP_RE.search(body)
    if not m:
        return (1, "1d4")
    try:
        avg = int(m.group("avg"))
    except ValueError:
        avg = 1
    dice = m.group("dice").strip().replace(" ", "")
    return (avg, dice)


_ACTIONS_HEADER_RE = re.compile(r"^(###|####)\s+Actions\b", re.MULTILINE)


def _slice_actions(body: str) -> str:
    """Return everything from the Actions header to the next sibling
    (or higher) header. Traits and Reactions sections are excluded.

    Two header levels exist in the SRD source files:
      - `#### Actions` for Monsters.txt (creatures under a `## Group`)
      - `### Actions`  for Miscellaneous-creatures.txt (creatures under
        a top-level `## Creature` header). Both must work."""
    m = _ACTIONS_HEADER_RE.search(body)
    if m is None:
        return ""
    rest = body[m.start():]
    # Stop at the next non-Actions header at the same or higher level.
    # Same level matches \n###... or \n####...; lower levels are inside
    # the section and shouldn't break the slice.
    stop = re.search(r"\n(##+)\s+(?!Actions\b)", rest[1:])
    if stop:
        return rest[:stop.start() + 1]
    return rest


def _parse_attacks(actions_text: str) -> list[MonsterAttack]:
    """Walk the actions section and pull every attack line.

    An attack is detected by `_ACTION_HEADER_RE`. The hit/damage clause
    follows in the same paragraph; we look for the first `_Hit:_` after
    the header. Save riders (paralysis etc.) are captured as descriptive
    text for the trace; the simulator ignores their mechanical effect
    in v1.
    """
    out: list[MonsterAttack] = []
    if not actions_text:
        return out

    for header in _ACTION_HEADER_RE.finditer(actions_text):
        name = header.group("name").strip()
        if name.lower() == "multiattack":
            continue
        try:
            to_hit = int(header.group("to_hit"))
        except ValueError:
            continue
        kind = header.group("kind").lower()
        # "Melee or Ranged" → prefer melee for engagement logic; the
        # simulator's positioning model is binary anyway.
        rng = "ranged" if kind == "ranged" else "melee"

        # Look for the matching _Hit:_ in the next paragraph (next ~2 lines).
        tail = actions_text[header.end():]
        # Stop scanning at the next action header so we don't bleed damage
        # from a later action into this one.
        next_action = _ACTION_HEADER_RE.search(tail)
        clip = tail[:next_action.start()] if next_action else tail

        hit = _HIT_RE.search(clip)
        if not hit:
            continue
        dice = hit.group("dice").replace(" ", "")
        dtype = hit.group("dtype").strip().lower()

        save: dict | None = None
        rider = ""
        save_match = _SAVE_RIDER_RE.search(clip)
        if save_match:
            try:
                dc = int(save_match.group("dc"))
            except ValueError:
                dc = 10
            ability = _ABILITY_TO_KEY.get(
                save_match.group("ability").lower(), "con"
            )
            half_on_save = "half as much" in clip.lower()
            save = {"ability": ability, "dc": dc, "half_on_save": half_on_save}
            # Descriptive snippet — first sentence after the save phrase.
            rider = clip[save_match.start():].splitlines()[0].strip()

        out.append(MonsterAttack(
            name=name,
            to_hit=to_hit,
            damage_dice=dice,
            damage_type=dtype,
            range=rng,
            save=save,
            rider_text=rider,
        ))
    return out
