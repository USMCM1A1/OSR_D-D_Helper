"""LLM-assisted character sheet → JSON ingestion.

Pipeline:
    PDF file
      → `pdf_to_text` (pypdf, deterministic offline)
      → `extract_character` (subprocess `claude --json-schema ... text`)
      → `save_character` (atomic write to dungeons/<name>/characters/)

The CLI subprocess pattern is the same one `dungeon_assistant.py` uses,
so the user pays no extra auth cost (rides on their existing Claude
Code subscription) and tests can inject a fake `runner` to stay fully
offline.

The schema is intentionally narrow — only fields the encounter
simulator reads. Anything the model can't find on the sheet falls back
to defaults (zero bonus, empty attack list); a partial extraction still
yields a runnable PC rather than an exception.

Failures we surface specifically:
  IngesterUnavailable  — claude CLI missing / not signed in / pypdf
                          missing. The editor server should render a
                          setup card instead of a generic error.
  RuntimeError         — CLI ran but failed (auth error, schema
                          violation, network). Editor surfaces the
                          message to the DM verbatim.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dungeon import atomic_write_text


# Default model. Sonnet is plenty for character-sheet extraction —
# Opus is overkill for narrow structured output.
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_CLI_NAME = "claude"

# Plenty of headroom for a sheet up to ~20 pages.
SUBPROCESS_TIMEOUT_SEC = 240

CliRunner = Callable[[list[str]], dict[str, Any]]


# ---------------------------------------------------------------------------
# JSON schema — the simulator's PC representation
# ---------------------------------------------------------------------------


CHARACTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "class": {
            "type": "string",
            "description": (
                "Lowercase D&D 5e class name. One of: fighter, cleric, rogue, "
                "wizard, sorcerer, paladin, ranger, barbarian, bard, druid, "
                "warlock, monk. If multi-class, pick the dominant class."
            ),
        },
        "level": {"type": "integer", "minimum": 1, "maximum": 20},
        "ac": {"type": "integer"},
        "hp_max": {"type": "integer"},
        "speed": {"type": "integer"},
        "init_bonus": {"type": "integer"},
        "ability_mods": {
            "type": "object",
            "properties": {
                "str": {"type": "integer"}, "dex": {"type": "integer"},
                "con": {"type": "integer"}, "int": {"type": "integer"},
                "wis": {"type": "integer"}, "cha": {"type": "integer"},
            },
            "required": ["str", "dex", "con", "int", "wis", "cha"],
        },
        "saves": {
            "type": "object",
            "properties": {
                "str": {"type": "integer"}, "dex": {"type": "integer"},
                "con": {"type": "integer"}, "int": {"type": "integer"},
                "wis": {"type": "integer"}, "cha": {"type": "integer"},
            },
        },
        "attacks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "to_hit": {"type": "integer"},
                    "damage": {
                        "type": "string",
                        "description": "Dice expression like '1d8+3'.",
                    },
                    "damage_type": {"type": "string"},
                    "range": {"type": "string", "enum": ["melee", "ranged"]},
                },
                "required": ["name", "to_hit", "damage", "range"],
            },
        },
        "spells": {
            "type": "object",
            "properties": {
                "slots": {
                    "type": "object",
                    "description": (
                        "Spell slots by level, e.g. {\"1\": 4, \"2\": 2}. "
                        "Omit for non-casters."
                    ),
                },
                "memorized": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "level": {"type": "integer"},
                            "type": {
                                "type": "string",
                                "enum": ["heal", "save_attack", "attack"],
                            },
                            "amount": {
                                "type": "string",
                                "description": (
                                    "Healing dice for heal-type spells "
                                    "(e.g. '1d8+3')."
                                ),
                            },
                            "save": {"type": "string"},
                            "dc": {"type": "integer"},
                            "damage": {"type": "string"},
                            "half_on_save": {"type": "boolean"},
                            "aoe": {"type": "boolean"},
                        },
                        "required": ["name", "level", "type"],
                    },
                },
            },
        },
        "features": {
            "type": "object",
            "properties": {
                "sneak_attack_dice": {"type": "integer"},
                "second_wind": {"type": "boolean"},
                "action_surge": {"type": "boolean"},
                "divine_smite_slots_usable": {"type": "boolean"},
                "rage": {"type": "boolean"},
            },
        },
        "tactics_notes": {"type": "string"},
    },
    "required": ["name", "class", "ac", "hp_max", "attacks"],
}


EXTRACTION_PROMPT = """\
You are extracting a D&D 5e character sheet into a fixed JSON schema for
a dungeon-encounter simulator. The simulator does not need every field
on the sheet — only the combat-relevant numbers in the schema.

Rules:

1. Always populate every REQUIRED field. If a value is genuinely absent
   from the sheet (not just hard to find), use a sensible default
   rather than leaving it out — for missing AC use 10, for missing
   hp_max use 8 × level, for missing damage use "1d4".

2. The `attacks` array should list every weapon attack the character
   can make in combat with its **to-hit bonus** (proficiency + ability
   mod) and **damage dice including ability modifier** (e.g. a longsword
   wielded two-handed by a Str 16 fighter is "1d10+3"). Skip noncombat
   tools.

3. For the `spells` block, list ONLY the spells the character has
   prepared/known that are useful in combat. Categorise each as:
     - `heal`     — single-target healing (Cure Wounds, Healing Word).
                    Set `amount` to the dice formula.
     - `save_attack` — DC-save damage spells (Sacred Flame, Burning
                    Hands, Fireball, Hold Person). Set `save`, `dc`,
                    `damage`, and `half_on_save`. Set `aoe: true` for
                    cones, spheres, or any spell hitting multiple
                    targets.
     - `attack`   — spell attack rolls (Eldritch Blast, Fire Bolt
                    treated as `save_attack` with no save is fine — use
                    `attack` only for rolled-to-hit spells).

4. The `features` block is a small set of booleans/integers that
   modify combat:
     - `sneak_attack_dice`: rogue Sneak Attack die count (e.g. 2 at
       level 3).
     - `second_wind`: fighter Second Wind available.
     - `divine_smite_slots_usable`: paladin can spend slots for smite.
     - Other features the simulator ignores can be omitted.

5. `class` is lowercase, single-word.

6. Output ONLY the JSON object. No prose.
"""


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


class IngesterUnavailable(RuntimeError):
    """Raised when the ingester can't run end-to-end (CLI missing,
    Claude Code not signed in, or pypdf not installed). The editor
    server catches this to render setup instructions."""


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------


def pdf_to_text(path: str | Path) -> str:
    """Extract a flat text dump from `path` using pypdf. Raises
    IngesterUnavailable if the dep isn't installed.

    Two-pass extraction:
      1. Page content streams (`page.extract_text()`) — picks up the
         static template labels of a fillable sheet.
      2. Form widget annotations (`/Widget` subtypes with `/T` field
         name + `/V` value) — picks up the *filled-in values*, which
         live OUTSIDE the page content stream on D&D Beyond /
         Mythweavers / official 5e fillable PDFs. Without this pass
         the LLM sees only the blank template and fills the schema
         with defaults.
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise IngesterUnavailable(
            "pypdf is not installed. Run `pip install pypdf>=4.0`."
        ) from e
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")
    reader = PdfReader(str(p))
    chunks: list[str] = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue
    page_text = "\n\n".join(c for c in chunks if c.strip())

    fields_text = _extract_form_field_lines(reader)
    if fields_text:
        return f"{page_text}\n\n=== FILLED FORM FIELDS ===\n{fields_text}"
    return page_text


# Skip widget values that mean "unchecked" or "no value". `/Off` and
# `/No` are PDF name objects pypdf renders as "/Off" or "/No"; we also
# treat empty strings and bare placeholders as no-value.
_EMPTY_FIELD_VALUES = frozenset({"", "/Off", "/No", "Off", "No", "--"})


def _extract_form_field_lines(reader) -> str:
    """Walk every page's annotations and emit `<field name>: <value>`
    lines for filled widgets. Resolves IndirectObjects as needed.
    Returns "" if no filled widgets are found.

    This is the path that recovers AcroForm field values for fillable
    character sheets — pypdf's `reader.get_fields()` returns {} on
    several common templates (D&D Beyond export, Mythweavers) because
    the fields aren't registered in the AcroForm root, only on the
    page widgets themselves.
    """
    try:
        from pypdf.generic import IndirectObject
    except ImportError:
        IndirectObject = ()  # type: ignore

    out: list[str] = []
    seen_names: set[str] = set()
    for page in reader.pages:
        annots = page.get("/Annots") if hasattr(page, "get") else None
        if annots is None:
            continue
        if isinstance(annots, IndirectObject):
            try:
                annots = annots.get_object()
            except Exception:
                continue
        try:
            iter_annots = list(annots)
        except TypeError:
            continue
        for a in iter_annots:
            try:
                obj = a.get_object() if hasattr(a, "get_object") else a
            except Exception:
                continue
            if not hasattr(obj, "get"):
                continue
            if obj.get("/Subtype") != "/Widget":
                continue
            name = obj.get("/T")
            value = obj.get("/V")
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str in _EMPTY_FIELD_VALUES:
                continue
            name_str = str(name).strip() if name is not None else ""
            # Some sheets repeat the same field on multiple pages
            # (header repeats character name etc.); de-dupe by name.
            key = f"{name_str}::{value_str}"
            if key in seen_names:
                continue
            seen_names.add(key)
            if name_str:
                out.append(f"{name_str}: {value_str}")
            else:
                out.append(value_str)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


def check_cli_available(cli_name: str = DEFAULT_CLI_NAME) -> None:
    if shutil.which(cli_name) is None:
        raise IngesterUnavailable(
            f"`{cli_name}` CLI not found on PATH. Character extraction "
            f"runs through your existing Claude Code subscription — "
            f"install Claude Code from claude.com/download and run "
            f"`{cli_name} login` if not already signed in."
        )


def _default_runner(args: list[str]) -> dict:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True,
            timeout=SUBPROCESS_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"claude CLI timed out after {SUBPROCESS_TIMEOUT_SEC}s"
        )
    except FileNotFoundError:
        raise IngesterUnavailable(
            "`claude` CLI vanished between availability check and call — "
            "is your PATH stable?"
        )
    if result.returncode != 0:
        snippet = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"claude CLI exited {result.returncode}: {snippet[:500]}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"claude CLI returned non-JSON: {e.msg}; "
            f"stdout starts with: {result.stdout[:200]!r}"
        )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_character(pdf_text: str, *, runner: CliRunner | None = None,
                      model: str = DEFAULT_MODEL,
                      cli_name: str = DEFAULT_CLI_NAME) -> dict:
    """Send the PDF text to Claude with the schema, return the parsed
    JSON dict. Raises RuntimeError on CLI failure; IngesterUnavailable
    if the CLI isn't installed."""
    if not pdf_text or not pdf_text.strip():
        raise ValueError("pdf_text is empty — nothing to extract")

    if runner is None:
        check_cli_available(cli_name)
        runner = _default_runner

    args = [
        cli_name,
        "--print",
        "--output-format", "json",
        "--model", model,
        "--system-prompt", EXTRACTION_PROMPT,
        "--json-schema", json.dumps(CHARACTER_SCHEMA),
        f"<character-sheet>\n{pdf_text}\n</character-sheet>\n\n"
        f"Extract this character into the schema.",
    ]
    envelope = runner(args)
    if envelope.get("is_error"):
        msg = envelope.get("result") or "claude CLI reported an error"
        raise RuntimeError(str(msg))

    structured = envelope.get("structured_output")
    if not isinstance(structured, dict):
        raise RuntimeError(
            "claude CLI did not return structured_output; got: "
            f"{type(structured).__name__}"
        )
    return structured


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    s = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return s or "character"


def characters_dir(dungeon_path: str | Path) -> Path:
    """The directory where per-dungeon character JSONs live. Created
    on demand by save_character."""
    p = Path(dungeon_path)
    # dungeon_path may be the dungeon folder OR the dungeon.json file
    # inside it; normalise to the folder.
    if p.is_file() or p.suffix == ".json":
        p = p.parent
    return p / "characters"


def save_character(character: dict, dungeon_path: str | Path,
                   *, filename: str | None = None) -> Path:
    """Write `character` JSON into the dungeon's characters/ folder.
    Atomic. Returns the file path."""
    name = str(character.get("name") or "Character")
    target_name = filename or f"{_slug(name)}.json"
    if not target_name.endswith(".json"):
        target_name += ".json"
    out_dir = characters_dir(dungeon_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / target_name
    payload = json.dumps(character, indent=2) + "\n"
    atomic_write_text(path, payload)
    return path


def load_characters(dungeon_path: str | Path) -> list[dict]:
    """Return every character JSON in the dungeon's characters/ folder,
    sorted by filename. Empty list if the folder doesn't exist."""
    d = characters_dir(dungeon_path)
    if not d.exists():
        return []
    out: list[dict] = []
    for path in sorted(d.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            out.append(data)
    return out
