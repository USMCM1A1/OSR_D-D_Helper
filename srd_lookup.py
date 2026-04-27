"""SRD stat-block lookup.

Parses the markdown files under `srd-resources/` once on first use and
exposes:

    find(name)  -> StatBlock | None       (exact name, case-insensitive)
    names()     -> list[str]              (all known creatures)
    scan(text)  -> list[StatBlock]        (every creature mentioned in text)

The two source files use different markdown levels for creature names:

    Monsters.txt              Miscellaneous-creatures.txt
      ## Group (e.g. Skeletons)   ## Creature (e.g. Ape)
        ### Creature (e.g. Skeleton)
          #### Actions / Traits

We detect creature sections by header level *and* by the presence of the
**Armor Class** marker in the body — header text alone is ambiguous
(e.g. `### Burrow` is a rules subsection, not a monster).

`scan` matches longest names first so "Minotaur Skeleton" wins over plain
"Skeleton" when both appear in the same string. Plurals are honoured
(`Skeletons` matches the `Skeleton` block) and matches are word-bounded
so "skeletal" doesn't pull in "Skeleton".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

SRD_DIR = Path(__file__).parent / "srd-resources"
MONSTERS_PATH = SRD_DIR / "Monsters.txt"
MISC_PATH = SRD_DIR / "Miscellaneous-creatures.txt"


class StatBlock(NamedTuple):
    name: str
    body: str  # markdown including the heading line, trimmed


def _sections_at_level(text: str, level: int) -> list[tuple[str, str]]:
    """Split markdown by headers at exactly `level` (#'s). The body of a
    section extends until the next header at level <= the current one,
    so subsections (deeper headers) are kept inside the body."""
    pat = re.compile(r"^(#{1,6}) ([^\n]+)$", re.MULTILINE)
    matches = list(pat.finditer(text))
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        if len(m.group(1)) != level:
            continue
        end = len(text)
        for j in range(i + 1, len(matches)):
            if len(matches[j].group(1)) <= level:
                end = matches[j].start()
                break
        out.append((m.group(2).strip(), text[m.start():end].strip()))
    return out


def _looks_like_statblock(body: str) -> bool:
    return "**Armor Class**" in body and "**Hit Points**" in body


def _build_index() -> dict[str, StatBlock]:
    index: dict[str, StatBlock] = {}
    if MONSTERS_PATH.exists():
        text = MONSTERS_PATH.read_text(encoding="utf-8")
        for name, body in _sections_at_level(text, level=3):
            if _looks_like_statblock(body):
                index[name.lower()] = StatBlock(name=name, body=body)
    if MISC_PATH.exists():
        text = MISC_PATH.read_text(encoding="utf-8")
        for name, body in _sections_at_level(text, level=2):
            if _looks_like_statblock(body):
                index[name.lower()] = StatBlock(name=name, body=body)
    return index


_INDEX: dict[str, StatBlock] | None = None


def _index() -> dict[str, StatBlock]:
    global _INDEX
    if _INDEX is None:
        _INDEX = _build_index()
    return _INDEX


def find(name: str) -> StatBlock | None:
    return _index().get(name.lower().strip())


def names() -> list[str]:
    return sorted(s.name for s in _index().values())


def scan(text: str) -> list[StatBlock]:
    """Find every creature whose name appears in `text`. Returns a
    de-duplicated list ordered by where the match starts in the text."""
    if not text:
        return []
    haystack = text.lower()
    # Longest names first so "Minotaur Skeleton" claims its span before
    # plain "Skeleton" gets a chance to match the inner word.
    candidates = sorted(_index().values(), key=lambda s: -len(s.name))
    found: list[tuple[int, StatBlock]] = []
    consumed: list[tuple[int, int]] = []
    seen: set[str] = set()
    for sb in candidates:
        pat = re.compile(r"\b" + re.escape(sb.name.lower()) + r"s?\b")
        for m in pat.finditer(haystack):
            span = (m.start(), m.end())
            if any(span[0] < e and span[1] > s for s, e in consumed):
                continue
            consumed.append(span)
            if sb.name not in seen:
                seen.add(sb.name)
                found.append((span[0], sb))
            break  # one statblock per creature is enough
    found.sort(key=lambda t: t[0])
    return [sb for _, sb in found]
