"""SRD stat-block lookup.

Parses the markdown files under `srd-resources/` once on first use and
exposes:

    find(name)  -> StatBlock | None       (exact name, case-insensitive)
    names()     -> list[str]              (all known creatures)
    scan(text)  -> list[Match]            (every creature mentioned in
                                            text by exact word-boundary
                                            match)

The two source files use different markdown levels for creature names:

    Monsters.txt              Miscellaneous-creatures.txt
      ## Group (e.g. Skeletons)   ## Creature (e.g. Ape)
        ### Creature (e.g. Skeleton)
          #### Actions / Traits

We detect creature sections by header level *and* by the presence of the
**Armor Class** marker in the body — header text alone is ambiguous
(e.g. `### Burrow` is a rules subsection, not a monster).

`scan` does a single exact pass: word-boundary regex (longest name wins,
plural-`s` accepted). A previous version of this module also did a
difflib-based fuzzy pass to catch misspellings (`zomby`→`Zombie`), but
in practice the DM rarely mistypes monster names while the false-
positive rate against ordinary English is unacceptable: the prose
"Fight to the death" used to silently inject Wight, Dretch and Knight
into a room. The fuzzy pass was removed in favor of a clean
"misspelt → no match" failure mode, which is easier to notice and fix.

No LLM. Deterministic and offline.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, NamedTuple

SRD_DIR = Path(__file__).parent / "srd-resources"
MONSTERS_PATH = SRD_DIR / "Monsters.txt"
MISC_PATH = SRD_DIR / "Miscellaneous-creatures.txt"


class StatBlock(NamedTuple):
    name: str
    body: str  # markdown including the heading line, trimmed


class Match(NamedTuple):
    """One creature reference found in input text.

    `source` is kept on the type for backwards compatibility with prior
    callers; it is now always `"exact"`. The fuzzy matcher was removed
    after it produced too many false positives on ordinary English.
    """
    statblock: StatBlock
    source: Literal["exact", "fuzzy"]
    original: str  # the substring of the input text that matched


class EncounterEntry(NamedTuple):
    """One formal monster declaration parsed from a room's encounter
    text — the `<count> <Name>(s) (MM p.<page>)` pattern. Anchored on
    the parenthesised page reference so descriptive prose can mention
    creature-adjacent words ("shadow", "light", "death") without
    inflating the encounter."""
    count_expr: str   # the literal count token: "3", "1d4", "1d6+1"
    statblock: StatBlock
    original: str     # the full matched substring


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


_DECLARATION_RE = re.compile(
    # count: integer or NdN±M dice expression. Optional — a bare
    # "Giant Scorpion (MM p.327)" is read as a single creature.
    r"\b"
    r"(?:(?P<count>\d+(?:d\d+(?:[+-]\d+)?)?)\s+)?"
    # name: one or more capitalised words, no punctuation. The capital
    # requirement keeps the match from extending backwards into
    # lowercase prose ("in the hollow Giant Scorpion ..."). The
    # greediness is bounded by the required parenthesised page ref
    # that follows. Title-Case prose words that *do* sit adjacent to
    # the declaration ("The Giant Scorpion (MM p.327)") are peeled
    # off in `parse_encounter_declarations` until the remainder
    # resolves to a known stat block.
    r"(?P<name>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)"
    # MM page ref. "p." is optional; spacing is flexible: "(MM p.272)",
    # "(MM p. 272)", "(MM 272)" all match.
    r"\s*\(MM\s*(?:p\.?\s*)?\d+\s*\)",
)


def parse_encounter_declarations(text: str) -> list[EncounterEntry]:
    """Pull formal monster declarations from `text`. Each declaration
    is a `<count> <Name>(s) (MM p.<page>)` triple. The MM page
    reference is the anchor that distinguishes a declaration from
    descriptive prose — without it, a sentence like "1 Giant Scorpion
    ... dormant in a hollow it has scraped in the pit's northern
    shadow" would inflate the encounter with a phantom Shadow.

    Names are resolved against the SRD index via `find`, with a fall-
    back that strips trailing 's' for simple plurals. Declarations
    whose name doesn't resolve to a known stat block are skipped (the
    empty stat-block area in the editor is the DM's cue to fix the
    spelling).
    """
    if not text:
        return []
    out: list[EncounterEntry] = []
    seen_names: set[str] = set()
    for m in _DECLARATION_RE.finditer(text):
        raw_name = m.group("name").strip()
        # Peel leading Title-Case words off the captured name until the
        # remainder resolves. Handles prose words that the regex pulls
        # in alongside the real creature name, e.g. "The Giant Scorpion
        # (MM p.327)" → try "The Giant Scorpion" → "Giant Scorpion" ✓.
        sb = None
        words = raw_name.split()
        for i in range(len(words)):
            candidate = " ".join(words[i:])
            sb = find(candidate) or find(candidate.rstrip("sS"))
            if sb is not None:
                break
        if sb is None:
            continue
        if sb.name in seen_names:
            # Same creature declared twice: keep the first declaration's
            # count and skip the rest. (If the DM really wants
            # heterogeneous counts they can scale the first.)
            continue
        seen_names.add(sb.name)
        out.append(EncounterEntry(
            count_expr=m.group("count") or "1",
            statblock=sb,
            original=m.group(0),
        ))
    return out


def scan(text: str) -> list[Match]:
    """Find every creature whose name appears in `text`. Returns a
    de-duplicated list ordered by where the match starts.

    Exact word-boundary regex only, with optional trailing 's' for
    plurals. Longest creature names are tried first so a multi-word
    name ("Minotaur Skeleton") claims its span before "Skeleton" can
    match the inner word.

    Note: for room enrichment and encounter simulation, prefer
    `parse_encounter_declarations` — that function anchors on the
    formal `(MM p.<page>)` declaration syntax and ignores prose
    mentions of creature-adjacent words.
    """
    if not text:
        return []
    haystack = text.lower()

    candidates = sorted(_index().values(), key=lambda s: -len(s.name))
    consumed: list[tuple[int, int]] = []
    found: list[tuple[int, Match]] = []
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
                found.append((span[0], Match(
                    statblock=sb, source="exact",
                    original=text[span[0]:span[1]],
                )))
            break  # one statblock per creature

    found.sort(key=lambda t: t[0])
    return [match for _, match in found]
