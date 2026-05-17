"""LLM-backed dungeon assistant — Claude Code CLI backend.

Subprocesses the local `claude` CLI rather than calling the Anthropic
API directly. The user's existing Claude Code subscription auth powers
each call (no separate `ANTHROPIC_API_KEY` needed; cost bills against
the subscription).

Per-turn flow:

    claude --print --output-format json --model <m>
           --system-prompt <SYSTEM_PROMPT>     (first turn only)
           --json-schema <PROPOSE_ROOMS_SCHEMA>
           --resume <session_id>               (subsequent turns)
           "<user prompt>"

The CLI returns a JSON envelope with:
  - `result`         — the model's prose reply (assistant chat text)
  - `structured_output` — the JSON that satisfies our schema (proposals)
  - `session_id`     — opaque token we pass to --resume next turn
  - `usage`, `total_cost_usd` — surfaced for the chat usage line

Vision is handled by Claude Code's built-in read tool. We don't
base64-encode the level PNG; we just include its absolute path in the
prompt and the CLI loads it natively.

Tests inject a fake `runner` callable so unit tests run offline
without the CLI installed. The runner takes the argv list and returns
a dict shaped like the CLI's --output-format=json envelope.

Caveats vs the raw SDK:
  - Each turn pays subprocess startup overhead (~250-500 ms).
  - Prompt-cache breakpoints aren't configurable; the CLI handles
    caching internally and we surface usage so the DM can see it.
  - On structured-output failure (model emits invalid JSON for the
    schema), the CLI retries internally before reporting; we treat any
    `is_error: true` as a hard failure.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import config
import srd_lookup
from dungeon import Dungeon


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default model. User explicitly chose Sonnet 4.6 in the planning round.
# The CLI accepts aliases ("sonnet", "opus") or full IDs ("claude-sonnet-4-6").
DEFAULT_MODEL = "claude-sonnet-4-6"

# CLI binary name. Tests can override via AssistantSession(cli_name=...).
DEFAULT_CLI_NAME = "claude"

# Per-turn subprocess wall-clock cap. The first turn is the heaviest:
# system prompt + dungeon context + level image (vision) + structured
# output for ~6-10 rooms, all with no prefix cache yet — empirically
# 60-300 s on Sonnet 4.6, longer if the model is exploring. Subsequent
# refinement turns ride the prefix cache and finish in 5-30 s.
# 480 s (8 min) gives even slow first turns comfortable headroom while
# staying well below the SDK's "should be streaming" threshold (~10
# min). If you hit this ceiling regularly, either narrow the level
# (fewer rooms per batch) or switch to a faster model from the dropdown.
SUBPROCESS_TIMEOUT_SEC = 480


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# The system prompt is split into a stable base + a dynamically-built
# SRD creature list. The list comes from srd_lookup.names() at module
# load so the assistant is constrained to the same creature index the
# editor's enrichment and the encounter simulator can resolve — no more
# "the assistant suggested a Banshee that Enrich silently drops".


_SYSTEM_PROMPT_BASE = """\
You are a dungeon design assistant for D&D 5E using an old-school Renaissance
(OSR) exploration paradigm. The campaign emphasises logistics, resource
management, and emergent storytelling over scripted narrative. Telling a grand
narrative where the PCs are the main character is not the primary activity —
exploration, discovery, and player decision-making are.

You help populate per-room content for a dungeon JSON file the DM is
authoring. The dungeon, theme, level number, and party level are provided in
the dungeon-context block of the first user message; the level's annotated
map image path is included in the prompt — read it before proposing.

THEME / CONCEPT — the DM's "Theme" field may be a tag-line ("catacombs of
an exiled priest-king") OR a paragraph-length concept that establishes the
dungeon's premise, its level structure, key NPCs, current events, or
ongoing rituals (for example: "long-lost assassins' cult, three levels —
living quarters, torture chambers, temple/relics — a desert genie has
burst in and is conducting a ritual on the third level"). Treat whatever
the DM writes as authoritative. When the concept names per-level themes,
align proposals to the level you're working on. When it names a current
event or NPC, weave it into the level it touches and reference it in
neighbouring rooms (rumours, evidence, reaction). Do not invent
contradictions to the concept — if it says level 1 is residential, your
encounters and special features should fit residential.

VOICE — box_text (read-aloud):
Terse, atmospheric, second-person present tense. Two to four sentences max.
Describe only what the senses immediately perceive — smell, sound, light,
obvious features. Never describe hidden things, monster intentions, or
mechanical information.
Example: "The smell of old water reaches you before you see the room. A wide
stone chamber, its floor slick with moisture. Iron rings are set into the far
wall at chest height, their purpose unclear."

VOICE — encounter_text:
Concrete and practical. Reference the SRD by creature name (e.g. "3 Skeletons
(MM p.272)"), number appearing, initial disposition (hostile/neutral/unaware),
and one tactical note. Keep it under 4 sentences.
Example: "2 Ghouls (MM p.148). Unaware — feeding on a corpse in the northeast
corner. Will not pursue beyond the room."

The formal declaration syntax — `<count> <Name>(s) (MM p.<page>)` — is
load-bearing. The room editor's enrichment button and the encounter
simulator both parse encounter_text by anchoring on the parenthesised MM
page reference. Without it, the declaration is invisible to both tools.
Always include the MM page ref (page numbers may be approximate; the
parser only cares that one is present). The creature name MUST come
from the SRD CREATURE LIST below — see the constraint at the bottom of
this prompt.

VOICE — treasure_text:
Specific. Mundane container first, then contents. Mix coin with one or two
interesting objects. Avoid magic items unless party level warrants it
(level 1-4: rare; level 5+: occasional).
Example: "A leather satchel slumped against the wall contains 34 gp, a brass
compass (non-magical, worth 15 gp), and a folded letter in a language the
party does not recognise."

VOICE — special_text (paired with the `special` tag):
Concrete description of the feature plus how it works mechanically. Cover
what the feature does, how the party can interact, relevant DCs, and
consequences.

SCHEMA — populate ONLY these fields per room:
  name              — 1-3 evocative words (not "Room 1"; e.g. "Flooded
                      Antechamber" or "The Chained Door").
  tags              — subset of [encounter, trap, treasure, special, empty,
                      stairs_up, stairs_down]. Multiple tags allowed
                      (e.g. ["encounter", "treasure"] for a guarded hoard).
  reaction_required — true if monsters are present and disposition is not
                      immediately obvious.
  notes             — DM-only mechanical notes: secret doors, trap mechanics,
                      monster tactics, connections to other rooms.
  box_text, encounter_text, treasure_text, special_text — as above. Leave
                      empty when the corresponding tag is absent.

DO NOT touch: id, state, image_region, encounter_ref, treasure_tier,
statblocks. The DM's UI manages these.

POPULATION FORMULA (B-Series) — apply when populating an empty level unless
the DM specifies otherwise:
  1/3 of rooms — monsters (half with treasure, half without).
  1/6 of rooms — traps (1/3 of those protecting treasure).
  1/6 of rooms — special features.
  1/3 of rooms — empty (1/6 of those with hidden treasure).
Round to whole rooms; document the breakdown in your `summary` field so the
DM can see the apportionment.

WORKFLOW: emit proposals via the structured-output JSON schema. Each call
carries a `summary` (1-2 sentences for chat) and a `rooms` array. Include
ONLY the rooms you're proposing this turn — never list a room you aren't
touching.

Before proposing, check the dungeon-context block for which rooms are already
populated. NEVER overwrite a populated room without explicit DM permission.
If the DM asks to revise a specific room, propose only that room. Stay within
the room IDs given to you — do not invent rooms.

If the level's room set is too small to apply the B-Series formula cleanly
(e.g. fewer than 6 rooms), say so in your prose reply and suggest a sensible
custom apportionment before proposing.
"""


def _build_system_prompt() -> str:
    """Append the live SRD creature list as a hard constraint. Called
    at module load — the index is stable across the process's lifetime,
    so a single computation is fine."""
    names = srd_lookup.names()
    creature_block = ", ".join(names)
    constraint = f"""
SRD CREATURE CONSTRAINT (HARD):
The DM's encounter-enrichment and combat-simulator pipeline can ONLY
resolve creatures whose names are in the local SRD index. You MUST
choose every monster from the list below; do not invent names, do not
use creatures from non-SRD sources (Volo's, Mordenkainen's, Fizban's,
etc.), and do not paraphrase ("a kind of ghoul" → use "Ghoul"). When
the level's theme calls for a creature that isn't on the list, pick
the closest analogue from the list and tune the prose to fit.

Use the canonical capitalisation shown. Plural form is fine
(`3 Skeletons (MM p.272)`); some creatures are their own plural
(`4 Lizardfolk (MM p.204)`).

Available creatures (alphabetical, {len(names)} total):
{creature_block}
"""
    return _SYSTEM_PROMPT_BASE + constraint


SYSTEM_PROMPT = _build_system_prompt()


# ---------------------------------------------------------------------------
# Structured output schema (replaces tool use)
# ---------------------------------------------------------------------------

PROPOSE_ROOMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "1-2 sentence cover note for the chat log (B-Series "
                "breakdown if first batch)."
            ),
        },
        "rooms": {
            "type": "array",
            "description": "Rooms in this proposal batch.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(config.ROOM_TAGS),
                        },
                    },
                    "reaction_required": {"type": "boolean"},
                    "notes": {"type": "string"},
                    "box_text": {"type": "string"},
                    "encounter_text": {"type": "string"},
                    "treasure_text": {"type": "string"},
                    "special_text": {"type": "string"},
                },
                "required": ["id", "name", "tags"],
            },
        },
    },
    "required": ["summary", "rooms"],
}


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class AssistantUnavailable(RuntimeError):
    """Raised when the assistant can't run end-to-end (CLI missing,
    Claude Code not signed in). The editor server catches this and
    renders a setup-instructions panel instead of the chat form."""


@dataclass
class RoomProposal:
    id: str
    name: str
    tags: tuple[str, ...]
    reaction_required: bool = False
    notes: str = ""
    box_text: str = ""
    encounter_text: str = ""
    treasure_text: str = ""
    special_text: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "tags": list(self.tags),
            "reaction_required": self.reaction_required,
            "notes": self.notes,
            "box_text": self.box_text,
            "encounter_text": self.encounter_text,
            "treasure_text": self.treasure_text,
            "special_text": self.special_text,
        }


@dataclass
class AssistantTurn:
    """One assistant turn — text reply, the proposed rooms, the model's
    summary, token usage, and the per-turn dollar cost reported by the
    CLI. Sent over the JSON wire to the browser."""
    text: str
    proposals: list[RoomProposal]
    summary: str = ""
    usage: dict | None = None
    cost_usd: float | None = None
    rejected: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Context block — same as before, no image encoding (CLI reads natively)
# ---------------------------------------------------------------------------


def build_dungeon_context_block(dungeon: Dungeon, level_number: int) -> str:
    """Compact text snapshot of one level — what the assistant needs to
    know to propose without overwriting populated rooms or inventing
    missing ones. Regenerated whenever an Apply mutates the dungeon so
    subsequent turns reflect the new state."""
    level = dungeon.levels_by_number.get(level_number)
    if level is None:
        return f"Level {level_number} not found in dungeon {dungeon.name!r}."

    lines: list[str] = [
        f"Dungeon: {dungeon.name}",
        f"Party level: {dungeon.party_level} (party size {dungeon.party.size})",
        f"Level {level.level_number}: {level.display_name}",
        f"WM check: {level.wm_check_method} threshold {level.wm_check_threshold}",
        "",
        "Wandering monster table:",
    ]
    for entry in level.wandering_monster_table:
        lines.append(f"  {entry.roll}: {entry.encounter}")

    lines.append("")
    lines.append("Rooms (ID — status — current tags — current name):")
    for r in level.rooms:
        has_content = bool(
            r.box_text.strip()
            or r.encounter_text.strip()
            or r.treasure_text.strip()
            or r.special_text.strip()
            or r.notes.strip()
            or (r.name and r.name != r.id)
        )
        status = "POPULATED" if has_content else "empty"
        tag_str = ", ".join(r.tags) if r.tags else "—"
        lines.append(f"  {r.id} — {status} — [{tag_str}] — {r.name!r}")

    if level.corridors:
        lines.append("")
        lines.append("Corridors:")
        for c in level.corridors:
            extras = f" tags={list(c.tags)}" if c.tags else ""
            lines.append(f"  {c.src} -> {c.dst}: {c.distance_ft} ft{extras}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


_FORBIDDEN_PROPOSAL_FIELDS = frozenset({
    "state", "image_region", "encounter_ref", "treasure_tier", "statblocks",
})


def validate_proposal(raw: dict, dungeon: Dungeon, level_number: int
                      ) -> RoomProposal:
    """Convert one raw room dict (from the CLI's structured_output) into
    a validated RoomProposal. Raises ValueError on unknown room id,
    unknown tags, or any forbidden field set."""
    bad_keys = _FORBIDDEN_PROPOSAL_FIELDS & raw.keys()
    if bad_keys:
        raise ValueError(
            f"proposal sets forbidden fields: {sorted(bad_keys)}"
        )

    rid = raw.get("id")
    if not isinstance(rid, str) or not rid:
        raise ValueError("proposal missing room id")

    level = dungeon.levels_by_number.get(level_number)
    if level is None:
        raise ValueError(f"level {level_number} not in dungeon")
    if rid not in level.rooms_by_id:
        raise ValueError(
            f"unknown room id {rid!r} for level {level_number}"
        )

    raw_tags = raw.get("tags", []) or []
    if not isinstance(raw_tags, list):
        raise ValueError("tags must be a list")
    bad_tags = [t for t in raw_tags if t not in config.ROOM_TAGS]
    if bad_tags:
        raise ValueError(f"unknown tags: {bad_tags}")
    tags = tuple(raw_tags) if raw_tags else ("empty",)

    return RoomProposal(
        id=rid,
        name=str(raw.get("name") or rid),
        tags=tags,
        reaction_required=bool(raw.get("reaction_required", False)),
        notes=str(raw.get("notes") or ""),
        box_text=str(raw.get("box_text") or ""),
        encounter_text=str(raw.get("encounter_text") or ""),
        treasure_text=str(raw.get("treasure_text") or ""),
        special_text=str(raw.get("special_text") or ""),
    )


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


CliRunner = Callable[[list[str]], dict]


def check_cli_available(cli_name: str = DEFAULT_CLI_NAME) -> None:
    """Raise AssistantUnavailable when `claude` CLI isn't on PATH. We
    don't probe `claude --version` here because it occasionally talks
    to an update server; the page render already pays a small cost
    and we want it to stay snappy. Auth failures surface naturally on
    the first turn."""
    if shutil.which(cli_name) is None:
        raise AssistantUnavailable(
            f"`{cli_name}` CLI not found on PATH. The dungeon assistant "
            f"runs through your existing Claude Code subscription — "
            f"install Claude Code from claude.com/download and run "
            f"`{cli_name} login` if not already signed in."
        )


def _default_runner(args: list[str]) -> dict:
    """Run the CLI with timeout + JSON parsing. Errors are wrapped as
    RuntimeError so the editor server can surface them in the chat."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"claude CLI timed out after {SUBPROCESS_TIMEOUT_SEC}s"
        )
    except FileNotFoundError:
        raise AssistantUnavailable(
            "`claude` CLI vanished between availability check and call — "
            "is your PATH stable?"
        )

    if result.returncode != 0:
        # Auth issues, schema validation failures, network errors all
        # land here. stderr is the most useful chunk.
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
# Session
# ---------------------------------------------------------------------------


@dataclass
class AssistantSession:
    """One DM-driven conversation, scoped to a dungeon path.

    Lifecycle:
        construct → start() to seed the conversation and get the first
        batch of proposals → send(text) for refinement turns → optionally
        refresh_dungeon_context() after the editor applies a proposal.

    The CLI handles message-history retention server-side (via the
    `session_id` we pass to --resume); we keep latest_proposals locally
    so /assistant/apply can mutate the JSON without round-tripping
    back to the LLM.
    """

    dungeon_path: Path
    dungeon: Dungeon
    theme: str
    level_number: int
    party_level: int
    model: str = DEFAULT_MODEL
    cli_name: str = DEFAULT_CLI_NAME
    # Inject for tests; otherwise use the real subprocess runner.
    runner: CliRunner | None = None

    _session_id: str | None = field(default=None, init=False, repr=False)
    _proposals: dict[str, RoomProposal] = field(
        default_factory=dict, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        if self.runner is None:
            check_cli_available(self.cli_name)
            self.runner = _default_runner

    # -- public API -----------------------------------------------------

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def latest_proposals(self) -> dict[str, RoomProposal]:
        return dict(self._proposals)

    def reset(self) -> None:
        """Drop the CLI session pointer and any queued proposals. The
        next start() will create a fresh CLI session."""
        self._session_id = None
        self._proposals.clear()

    def start(self) -> AssistantTurn:
        """Seed turn 1: include dungeon context + level image path +
        opening request. The CLI's read tool loads the image natively."""
        self.reset()
        return self._run(self._build_opening_prompt(), fresh=True)

    def send(self, user_text: str) -> AssistantTurn:
        """Append a follow-up turn. Reuses the CLI session via --resume,
        so the model has the full conversation context for free."""
        if self._session_id is None:
            raise RuntimeError("session not started — call start() first")
        text = user_text.strip()
        if not text:
            raise ValueError("empty user message")
        return self._run(text, fresh=False)

    def refresh_dungeon_context(self, fresh: Dungeon) -> None:
        """Update the in-memory dungeon snapshot after Apply mutates the
        JSON. The CLI session retains the original context block in its
        history; subsequent refinement turns operate against the model's
        memory of the prior proposals, which is what we want — it knows
        which rooms it already proposed."""
        self.dungeon = fresh

    # -- internals ------------------------------------------------------

    def _build_opening_prompt(self) -> str:
        level = self.dungeon.levels_by_number.get(self.level_number)
        if level is None:
            raise ValueError(
                f"level {self.level_number} not in dungeon {self.dungeon.name!r}"
            )
        context = build_dungeon_context_block(self.dungeon, self.level_number)
        image_path = (self.dungeon_path.parent / level.map_image).resolve()
        return (
            f"<dungeon-context>\n{context}\n</dungeon-context>\n\n"
            f"Theme: {self.theme}\n"
            f"Level: {self.level_number}\n"
            f"Party level: {self.party_level}\n\n"
            f"Reference the level's annotated map image: {image_path}\n\n"
            "Propose populated content for the empty rooms in this level "
            "using the structured-output schema. Apply the B-Series formula "
            "(or explain why a custom apportionment fits better). Skip any "
            "rooms marked POPULATED in the dungeon-context block — I'll "
            "ask you to revise those explicitly if I want changes."
        )

    def _build_args(self, prompt: str, *, fresh: bool) -> list[str]:
        args = [
            self.cli_name,
            "--print",
            "--output-format", "json",
            "--model", self.model,
            "--json-schema", json.dumps(PROPOSE_ROOMS_SCHEMA),
        ]
        if fresh:
            args.extend(["--system-prompt", SYSTEM_PROMPT])
        else:
            assert self._session_id is not None  # gated by send()
            args.extend(["--resume", self._session_id])
        args.append(prompt)
        return args

    def _run(self, prompt: str, *, fresh: bool) -> AssistantTurn:
        args = self._build_args(prompt, fresh=fresh)
        envelope = self.runner(args)

        if envelope.get("is_error"):
            err_msg = envelope.get("result") or "claude CLI reported an error"
            raise RuntimeError(str(err_msg))

        sid = envelope.get("session_id")
        if sid:
            self._session_id = sid

        text = str(envelope.get("result") or "")
        structured = envelope.get("structured_output") or {}
        if not isinstance(structured, dict):
            structured = {}
        summary = str(structured.get("summary", "")).strip()

        proposals: list[RoomProposal] = []
        rejected: list[dict] = []
        for raw_room in structured.get("rooms", []) or []:
            try:
                prop = validate_proposal(
                    raw_room, self.dungeon, self.level_number,
                )
            except ValueError as e:
                rejected.append({"raw": raw_room, "error": str(e)})
                continue
            proposals.append(prop)
            self._proposals[prop.id] = prop

        usage = envelope.get("usage")
        if usage is not None and not isinstance(usage, dict):
            usage = None
        cost = envelope.get("total_cost_usd")
        if cost is not None and not isinstance(cost, (int, float)):
            cost = None

        return AssistantTurn(
            text=text,
            proposals=proposals,
            summary=summary,
            usage=usage,
            cost_usd=float(cost) if cost is not None else None,
            rejected=rejected,
        )
