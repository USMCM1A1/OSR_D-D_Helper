"""SRD stat-block lookup.

Parses the markdown files under `srd-resources/` once on first use and
exposes:

    find(name)  -> StatBlock | None       (exact name, case-insensitive)
    names()     -> list[str]              (all known creatures)
    scan(text)  -> list[Match]            (every creature mentioned in text;
                                            each Match flags whether it was
                                            found by exact regex or fuzzy
                                            similarity so the editor can
                                            surface the difference)

The two source files use different markdown levels for creature names:

    Monsters.txt              Miscellaneous-creatures.txt
      ## Group (e.g. Skeletons)   ## Creature (e.g. Ape)
        ### Creature (e.g. Skeleton)
          #### Actions / Traits

We detect creature sections by header level *and* by the presence of the
**Armor Class** marker in the body — header text alone is ambiguous
(e.g. `### Burrow` is a rules subsection, not a monster).

`scan` is two-pass:
  Pass 1 — exact word-boundary regex (longest name wins, plural-`s`
           accepted). Catches the common case with zero false positives.
  Pass 2 — `difflib`-based fuzzy match on residual capitalised tokens
           the regex missed. Catches misspellings (`zomby`→`Zombie`,
           `skelaton`→`Skeleton`) and typo'd plurals. Cutoff 0.85;
           tokens shorter than 4 chars are skipped to avoid noise.
           Tagged `'fuzzy'` so the editor can render a caveat.

No LLM. Both passes are deterministic and offline.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Literal, NamedTuple

SRD_DIR = Path(__file__).parent / "srd-resources"
MONSTERS_PATH = SRD_DIR / "Monsters.txt"
MISC_PATH = SRD_DIR / "Miscellaneous-creatures.txt"

# Fuzzy-match tuning. `cutoff` is difflib's similarity ratio threshold
# (1.0 = identical, 0.0 = nothing in common). Empirically, 0.72 catches
# the typo cases we care about (`zomby`→Zombie at 0.727, `skelaton`,
# `goblen`, `kobald` all ≥ 0.83) while staying above unrelated pairs.
# Min token length 5 keeps short words ("from", "lone", "here") out of
# fuzzy territory — the regex pass handles correctly-spelt short names.
# The fuzzy pass will sometimes surface a debatable match (e.g.
# "skeletal" → Skeleton at 0.75) — that's by design: the editor renders
# fuzzy hits with a caveat so the DM can reject false positives.
FUZZY_CUTOFF = 0.72
FUZZY_MIN_TOKEN_LEN = 5


class StatBlock(NamedTuple):
    name: str
    body: str  # markdown including the heading line, trimmed


class Match(NamedTuple):
    """One creature reference found in input text. `source` records how
    the match was made so the UI can flag fuzzy hits for human review."""
    statblock: StatBlock
    source: Literal["exact", "fuzzy"]
    original: str  # the substring of the input text that matched


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


def scan(text: str) -> list[Match]:
    """Find every creature whose name appears in `text`. Returns a
    de-duplicated list ordered by where the match starts in the text.
    Two-pass: exact regex first (zero false positives), then fuzzy
    string similarity on residual tokens to catch misspellings."""
    if not text:
        return []
    haystack = text.lower()

    # ----- Pass 1: exact word-boundary regex with plural-s -----
    # Longest names first so "Minotaur Skeleton" claims its span before
    # plain "Skeleton" matches the inner word.
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

    # ----- Pass 2: fuzzy match on residual word tokens -----
    # Walk through every word in the original text. Skip tokens that
    # overlap an already-consumed exact span (those are accounted for).
    # Skip short tokens (≤3 chars) — too prone to noise. Apply
    # difflib.get_close_matches against the lowercase name index; accept
    # the top hit if its ratio is ≥ FUZZY_CUTOFF.
    name_index_lower = [s.name.lower() for s in _index().values()]
    name_lookup = {s.name.lower(): s for s in _index().values()}
    for m in re.finditer(r"\b[A-Za-z][A-Za-z'\-]+\b", text):
        span = (m.start(), m.end())
        token = m.group(0)
        if len(token) < FUZZY_MIN_TOKEN_LEN:
            continue
        if any(span[0] < e and span[1] > s for s, e in consumed):
            continue  # exact pass already claimed this region
        # Strip trailing 's' before fuzzy compare so "zombys" lines up
        # with "zombie" (otherwise the pluralisation drops the ratio).
        probe = token.lower().rstrip("s")
        if len(probe) < FUZZY_MIN_TOKEN_LEN:
            continue
        hits = difflib.get_close_matches(
            probe, name_index_lower, n=1, cutoff=FUZZY_CUTOFF,
        )
        if not hits:
            continue
        sb = name_lookup[hits[0]]
        if sb.name in seen:
            continue
        seen.add(sb.name)
        consumed.append(span)
        found.append((span[0], Match(
            statblock=sb, source="fuzzy", original=token,
        )))

    found.sort(key=lambda t: t[0])
    return [match for _, match in found]
