# D&D 5E Dungeon Exploration Tracker — CLAUDE.md

## Project Overview

A local pygame application that runs on macOS and can be projected to a second screen during tabletop D&D 5E sessions. The app enforces **5E dungeon exploration procedures** as written in the Player's Handbook and Dungeon Master's Guide — turn-based time tracking, resource depletion, wandering monster rolls, and a graph-based dungeon map with fog of war.

The campaign focus is old-school exploration and logistics: mapping unknown spaces, managing light and supplies, and surviving a dangerous environment. No novel mechanics are introduced. Everything the app tracks has a source in the 5E rulebooks.

This is a **Dungeon Master tool**, not a player-facing VTT. Character management and combat resolution are handled externally in D&D Beyond. This app handles dungeon state.

**Rules references**: PHB Chapter 8 (Adventuring), DMG Chapter 5 (Adventure Environments), DMG p. 82-83 (random encounters), DMG p. 136-137 (treasure by CR).

---

## UX Principle — app-first, no terminal escape hatches

User-facing workflows must complete inside the GUI. Telling the user to *"quit pygame and re-pick from the launcher,"* *"open a terminal and run …,"* *"restart to apply changes,"* or otherwise leave the app to finish a task is a **design failure**, not an acceptable workaround. The point of building this as an application is that it behaves like one.

When a feature needs to span the pygame DM View ↔ the editor-server SPA (separate threads in the same process), build the IPC channel rather than asking the user to bridge it manually. The renderer already exposes a `_pending_reload` / `ReloadRequest` mechanism for in-place dungeon switching — wire new flows into it instead of telling the user to relaunch.

CLI flags and shell scripts are fine as plumbing (automation, tests, the launcher). They are not acceptable as the **only** path for a user-facing operation. If a proposed solution starts with *"quit X and then …"* — that's the signal that the architecture, not the copy, needs to change.

---

## Dungeon Authoring

A dungeon is a self-contained folder under `dungeons/`:

```
dungeons/<name>/
├── dungeon.json    # metadata + levels + rooms + WM tables
├── level1.png      # one PNG per level; map_image paths are relative
├── level2.png
└── session.db      # per-dungeon SQLite — fog, turn, supplies, journal
```

The folder is the unit of work. Copy it, zip it, hand it to a friend — the JSON, the level PNGs, and the in-progress session state all travel together. To switch dungeons, point the app at a different folder. To start a fresh playthrough, delete `session.db` (the JSON + PNGs are preserved).

The **map images are the ground truth**. Workflow:

1. Find or draw a set of map PNGs (one per level) and drop them in `dungeons/<name>/`.
2. Write a starter `dungeon.json` next to them — top-level metadata (`dungeon_name`, `current_level`, `party`) plus one entry in `levels[]` per PNG. Each level needs a `level_number`, a `map_image` (bare filename, e.g. `level1.png`), and a `wandering_monster_table`. Rooms can start as `[]` — they're added in the editor.
3. Open the dungeon in the app (`python main.py dungeons/<name>`). Use **annotation mode** (`A`) to draw rectangles over the map image for each room; each rectangle becomes a `Room` with an `image_region`. Use the browser-tab editor (`E`) to fill in box text, encounters, treasure, and notes per room. Edits autosave to `dungeon.json`.
4. Play. Click rooms to cycle reveal state — the per-room rectangles drive the fog-of-war mask painted over the PNG.

There is no procedural generator. Every room exists because the DM drew it. The schema's `treasure_tier` field survives as a free-text label (e.g. `"DMG CR 0-4 individual"`) for the DM's own reference.

---

## Core 5E Exploration Procedures

### The Dungeon Turn
- **1 turn = 10 minutes of in-game time** (PHB p. 182, consistent with short rest minimum and light source durations)
- The DM advances the turn manually (button press or keyboard shortcut)
- Each turn triggers automatically:
  1. Tick down all active resource timers (torches, lanterns, spell durations)
  2. Roll for wandering monsters (DMG p. 82)
  3. Log the turn event in the session journal

### Travel Pace
Slow pace for cautious dungeon exploration: 200 ft/minute → 2,000 ft per 10-minute turn (PHB p. 182). The app does not auto-calculate distance; the DM advances turns based on fictional positioning and narration.

### Wandering Monster Check
- Per DMG p. 82: check every turn by rolling a **d20**; encounter on 18, 19, or 20
- Alternatively: **1d6, encounter on a 1** (also per DMG); DM selects method in dungeon JSON config
- App auto-rolls and displays result prominently
- On an encounter: visible alert displayed, DM resolves via D&D Beyond Encounters or ad hoc
- **Wandering Monster Table**: defined in dungeon JSON; app rolls on it when encounter is triggered

### Light Sources
Per PHB p. 183, rules as written:
- **Torch**: bright light 20 ft, dim 20 ft beyond; **6 turns (1 hour)**
- **Hooded lantern**: bright light 30 ft, dim 30 ft beyond; **36 turns (6 hours)** per oil flask
- **Bullseye lantern**: bright light 60 ft cone, dim 60 ft beyond; **36 turns** per oil flask
- **Candle**: bright light 5 ft, dim 5 ft beyond; **6 turns**
- **Oil flask**: one flask = one lantern fill
- **Darkvision**: per-character flag, no timer
- Each light source tracked individually with turn countdown; warning at 2 turns remaining

### Short Rest
Per PHB p. 186:
- Minimum **6 turns (1 hour)**
- WM checks continue each turn during rest
- Torches and lanterns continue to burn
- **Short Rest Mode**: DM clicks "Begin Short Rest"; app auto-advances 6 turns, rolling WM checks and burning resources for each, then alerts DM
- HP recovery and spell slots resolved in D&D Beyond; app tracks only time and resource cost

### Long Rest
Per PHB p. 186:
- **48 turns (8 hours)**; WM checks continue every turn
- Handled same as Short Rest Mode but for 48 turns
- Rarely used in-dungeon; included for completeness

### Exhaustion
Per PHB p. 291:
- Tracked as a simple counter (0-6) per character
- Causes: no food/water (DMG p. 185), forced march (PHB p. 182)
- DM increments manually; app displays as reminder
- Effects resolved by DM and D&D Beyond

### Rations and Supplies
Per DMG p. 111:
- **Rations**: 1 per character per day; DM decrements manually
- **Water**: 1 gallon per character per day
- **Other consumables** (rope, spikes, torches carried): simple counts, DM decrements manually

### Noise Modifier
Per DMG encounter frequency guidance:
- DM toggles **Noisy** flag (e.g., after combat, failed stealth, triggered trap)
- Increases WM check threshold for a configurable number of turns
- No automatic detection; DM applies judgment

---

## Dungeon Map — Graph-Based System

### Data Model
- **Rooms** = nodes:
  - Unique ID, short name
  - State: `unexplored` | `known` | `cleared`
  - Tags: `encounter`, `trap`, `treasure`, `special`, `empty`, `stairs_up`, `stairs_down` (multiple allowed; a room may carry both stair tags for a two-way connection)
  - `reaction_required`: boolean — if true, app prompts reaction roll on entry
  - Notes field (DM only)
  - `encounter_ref`: name of pre-built D&D Beyond encounter (optional)
  - `treasure_tier`: free-text label for the DM's treasure-table reference (optional)
- **Corridors** = edges:
  - Distance in feet
  - Tags: `secret`, `locked`, `trapped`, `one-way`

### Fog of War
- Rooms start `unexplored` (hidden in Player View)
- DM reveals rooms/corridors by clicking
- **DM View**: all rooms visible with full notes
- **Player View**: revealed rooms only, no notes
- Reveal state is recorded **per level**. Switching levels does not reset fog of war — a party returning to Level 1 from Level 2 sees Level 1 exactly as they left it.

### Map Rendering
- Rooms as thick-outlined black rectangles on parchment; corridors as thick black lines. See **Aesthetic Direction** for the full pen-and-ink spec.
- **Room state** is conveyed by fill, not hue:
  - `unexplored` (DM View only): rectangle filled with crosshatch, no label, no icon.
  - `known`: parchment fill with room label and tag icon(s) stamped inside.
  - `cleared`: parchment fill with a faint diagonal cross-out through the room.
- **Tag icons** stamped inside known rooms: skull (encounter), chest (treasure), `X` (trap), star (special). Empty rooms have no icon.
- **Active alert overlay** (red ink): encounter triggered → red outline + flashing skull; reaction-required → red outline on the rectangle.
- **Door symbols** on corridor edges: T-bar tick for `locked` or `trapped`; dashed line for `secret`; arrowhead for `one-way`.
- Party position marker on current room (small black ink figure or filled dot).
- DM can drag nodes (cosmetic); map is zoomable and pannable.

### Multi-Level Dungeons

A dungeon is a stack of self-contained levels. Each level has its own map image, room graph, corridor graph, wandering monster table, and WM check rules. Only the **current level** is loaded and rendered at a time; non-current levels stay on disk until visited.

- **`current_level`** at the top of the JSON selects which level is active when the session opens.
- **`levels[]`** is an ordered array, indexed by `level_number` (1 = surface entry, increasing values = deeper). The party config (`party`, `party_level`) and dungeon-level metadata stay at the top of the JSON.
- **Map image per level**: each level supplies a `map_image` path (resolved relative to the dungeon folder, so a portable dungeon directory carries its own PNGs) and a `map_image_scale`. Images are loaded on demand — the previous level's image is released when the new one is loaded so memory stays low.
- **Level switching**: the DM ascends with `Ctrl+Up` / `[▲ Ascend]` and descends with `Ctrl+Down` / `[▼ Descend]`. On switch:
  1. Current level state (room reveals, party position, active effects) is flushed to SQLite immediately.
  2. The new level's image and graph are loaded.
  3. Party position is set to a room tagged `stairs_down` (when descending) or `stairs_up` (when ascending) on the destination level. If no such room is tagged, the party drops onto the first room in the destination's room list.
  4. The journal logs the transition inline, e.g. `Turn 14 — Party descends to Level 2 — The Burial Halls`.
- **Disabled controls at boundaries**: ascend is disabled on level 1; descend is disabled on the deepest level.
- **Per-level node alignment**: each level has its own room positions. Alignment data is keyed in SQLite by `(dungeon_name, level_number, room_id)`. Entering alignment mode (`Ctrl+A`) only affects the currently displayed level.
- **Per-level fog of war**: see the Fog of War subsection above. Reveal state is preserved per level.
- **Single continuous journal**: the `turn_log` table is a single stream across all levels — level transitions appear as inline entries; the log is *not* partitioned by level.

### Dungeon Definition File Schema

```json
{
  "dungeon_name": "The Tomb of the Iron Lich",
  "party_level": 3,
  "current_level": 1,
  "party": {
    "size": 4,
    "characters": [
      {"name": "Thorin", "darkvision": true,  "exhaustion": 0},
      {"name": "Mira",   "darkvision": true,  "exhaustion": 0},
      {"name": "Aldric", "darkvision": false, "exhaustion": 0},
      {"name": "Sera",   "darkvision": false, "exhaustion": 0}
    ]
  },
  "levels": [
    {
      "level_number": 1,
      "display_name": "Level 1 — The Entry Vaults",
      "map_image": "level1.png",
      "map_image_scale": 1.0,
      "wm_check_method": "d20",
      "wm_check_threshold": 18,
      "wm_check_frequency": "every_turn",
      "wandering_monster_table": [
        {"roll": 1, "encounter": "2d6 Skeletons"},
        {"roll": 2, "encounter": "1d4 Zombies"},
        {"roll": 3, "encounter": "Giant Spider"},
        {"roll": 4, "encounter": "Gelatinous Cube"},
        {"roll": 5, "encounter": "Shadows (1d3)"},
        {"roll": 6, "encounter": "Ghoul"}
      ],
      "rooms": [
        {
          "id": "R01",
          "name": "Entrance Hall",
          "state": "unexplored",
          "tags": ["empty"],
          "reaction_required": false,
          "notes": "10x30 ft. Bas-relief carvings. Dust undisturbed.",
          "encounter_ref": null,
          "treasure_tier": null
        },
        {
          "id": "R02",
          "name": "Guard Room",
          "state": "unexplored",
          "tags": ["encounter", "treasure"],
          "reaction_required": true,
          "notes": "4 skeletons, inactive. Footlocker with 50 gp.",
          "encounter_ref": "Skeleton Guard Room",
          "treasure_tier": "cr0-4"
        },
        {
          "id": "R05",
          "name": "Spiral Stair Down",
          "state": "unexplored",
          "tags": ["empty", "stairs_down"],
          "reaction_required": false,
          "notes": "Stone spiral stair descending into damp air.",
          "encounter_ref": null,
          "treasure_tier": null
        }
      ],
      "corridors": [
        {"from": "R01", "to": "R02", "distance_ft": 30, "tags": ["locked"]},
        {"from": "R02", "to": "R05", "distance_ft": 60, "tags": []}
      ]
    },
    {
      "level_number": 2,
      "display_name": "Level 2 — The Burial Halls",
      "map_image": "level2.png",
      "map_image_scale": 1.0,
      "wm_check_method": "d20",
      "wm_check_threshold": 18,
      "wm_check_frequency": "every_turn",
      "wandering_monster_table": [
        {"roll": 1, "encounter": "Ghoul"},
        {"roll": 2, "encounter": "Wight"},
        {"roll": 3, "encounter": "1d6 Specters"}
      ],
      "rooms": [
        {
          "id": "R10",
          "name": "Stair Landing",
          "state": "unexplored",
          "tags": ["empty", "stairs_up"],
          "reaction_required": false,
          "notes": "Foot of the spiral stair from Level 1.",
          "encounter_ref": null,
          "treasure_tier": null
        }
      ],
      "corridors": []
    }
  ]
}
```

---

## Display Modes

| Mode | For | Shows |
|---|---|---|
| **DM View** | DM laptop | All rooms, notes, tags, resource panel, WM results, journal |
| **Player View** | Projected screen | Revealed rooms only, no notes, no DM alerts, party position |

Two separate windows when projecting; toggled by `Tab` on single screen.

---

## Session Journal

Scrolling log (DM View only), auto-populated:
- `Turn 7 — Elapsed: 1h 10m`
- `Turn 7 — Torch #2: 2 turns remaining ⚠`
- `Turn 7 — WM Check: rolled 19 — ENCOUNTER → Ghoul`
- `Turn 7 — Entered R02 — Reaction roll required`
- `Turns 8–13 — Short Rest. 6 WM checks. No encounters.`
- `Turn 14 — Party descends to Level 2 — The Burial Halls`
- Manual DM notes at any time

Saved to SQLite, exportable to plain text. The journal is a single continuous log across all levels — level transitions are logged inline, not partitioned per level.

---

## Persistence — SQLite Backend

| Table | Contents | Scoped by |
|---|---|---|
| `sessions` | Session metadata (dungeon file, date, party level, notes, `current_level`) | — |
| `room_state` | Per-room state overrides + node-alignment positions | `(session_id, level_number, room_id)` |
| `resources` | Light source instances with turns remaining; supply counts | `session_id` |
| `characters` | Per-character exhaustion and darkvision | `session_id` |
| `active_effects` | Spell durations, noise flag with turn expiry | `(session_id, level_number)` |
| `turn_log` | Full journal — single continuous stream across all levels | `session_id` |
| `party_position` | Current room ID, per visited level | `(session_id, level_number)` |

`level_number` columns scope per-level state so each level's reveal map, node alignment, and active effects are remembered independently. The `turn_log` is *not* level-scoped — level transitions are noted inline.

Startup: load dungeon JSON, merge with SQLite session state for the `current_level` only. Sessions resume seamlessly across multiple play dates.

---

## UI Layout (DM View)

```
┌─────────────────────────────────────┬──────────────────────┐
│                                     │  LEVEL               │
│                                     │  L2 — Burial Halls   │
│                                     │  [▲ Ascend]  [▼ Desc]│
│         DUNGEON MAP                 │──────────────────────│
│   (current level only —             │  TURN TRACKER        │
│    map_image + graph overlay)       │  Turn: 14            │
│                                     │  Elapsed: 2h 20m     │
│                                     │  Party Level: 3      │
│                                     │──────────────────────│
│                                     │  LIGHT SOURCES       │
│                                     │  Torch 1:  2t ⚠      │
│                                     │  Torch 2:  5t        │
│                                     │  Lantern:  31t       │
│                                     │──────────────────────│
│                                     │  SUPPLIES            │
│                                     │  Oil: 2   Rations:24 │
│                                     │  Water: 4 gal        │
│                                     │──────────────────────│
│                                     │  EXHAUSTION          │
│                                     │  Thorin: 0  Mira: 0  │
│                                     │  Aldric: 1  Sera: 0  │
│                                     │──────────────────────│
│                                     │  [ADVANCE TURN]      │
│                                     │  [SHORT REST]        │
│                                     │  [NOISY] toggle      │
│                                     │  Last WM: T13 → 11 ✓ │
├─────────────────────────────────────┴──────────────────────┤
│  JOURNAL                                                    │
│  Turn 14 — Party descends to Level 2 — The Burial Halls    │
│  Turn 14 — Torch #1: 2 turns remaining ⚠                   │
│  Turn 13 — WM Check: rolled 11 — No encounter              │
│  Turn 12 — WM Check: rolled 19 — ENCOUNTER → Giant Spider  │
│  Turn 11 — Entered R02 — Reaction roll required            │
└─────────────────────────────────────────────────────────────┘
```

The **LEVEL** panel at the top of the side rail shows the current level's number and display name plus the ascend/descend buttons. `[▲ Ascend]` is disabled on level 1; `[▼ Descend]` is disabled on the deepest level.

---

## Controls & Keyboard Shortcuts

| Key / Button | Action |
|---|---|
| `Space` / `Enter` | Advance one turn |
| `Tab` | Toggle DM / Player View |
| `R` | Manual WM roll (no turn advance) |
| `N` | Add manual journal note |
| `S` | Begin Short Rest (auto-advances 6 turns) |
| `L` | Begin Long Rest (auto-advances 48 turns) |
| `Click room` | Reveal room / cycle state (DM View) |
| `Drag room` | Reposition node (cosmetic) |
| `Scroll` | Zoom map |
| `Middle drag` | Pan map |
| `Ctrl+S` | Force save to SQLite |
| `Ctrl+L` | Load dungeon JSON |
| `Ctrl+Up` / `[▲ Ascend]` | Ascend one level (disabled on level 1) |
| `Ctrl+Down` / `[▼ Descend]` | Descend one level (disabled on deepest level) |
| `Ctrl+A` | Enter node-alignment mode for the current level |

---

## Aesthetic Direction

Pen-and-ink dungeon cartography on parchment — the look of a hand-inked Dyson-Logos map, not a digital UI. Reference: `dungeon-art-1.png`, `dungeon-art-2.png` in the project root.

### Map surface
- **Background**: parchment / aged paper, `#f4e4c1`. No dark background anywhere — app shell, map canvas, and panels all share this surface.
- **Ink**: pure black `#1a1a1a` for all map linework — room outlines, corridors, icons, door symbols, room labels.
- **Two-tone rule**: black ink on parchment. **Red** (`#a8201a` or similar muted ink red) is reserved exclusively for active alerts — encounter triggered, torch ≤ 2 turns, exhaustion ≥ 3, reaction-required outline. No other color appears on the map.
- **Rooms**: thick-outlined rectangles (not circles), heavy black stroke. Label rendered in a small serif or slab-serif font inside the room or just below it.
- **Corridors**: thick single black lines connecting room rectangles.
- **Wall fill**: dense crosshatch (or solid dark fill where crosshatch is too costly) in the negative space immediately around rooms and along corridors, approximating inked wall mass.
- **Door symbols**: perpendicular T-bar tick across corridor lines for `locked` or `trapped` edges; classic dungeon-map door glyph.
- **Tag icons**: small stamped black-ink glyphs inside each room, keyed to its tag — skull (encounter), chest outline (treasure), `X` (trap), star/asterisk (special). Icons replace per-state node coloring.
- **Reaction-required**: red ink outline around the room rectangle (only place red appears outside an alert state).

### Typography
- **Room labels (on the map)**: serif or slab-serif. Nothing modern or sans-serif on the map surface itself.
- **UI panels** (turn tracker, resource panel, journal): monospace.
- No decorative chrome; every element still serves a function.

### What this overrides
This replaces the prior "war room / dark background / amber + sans-serif" direction. The dark theme is retired across the whole app, not just the map.

---

## Tech Stack

- **Python 3.11+**
- **pygame** for rendering and input
- **SQLite** via `sqlite3` stdlib for persistence
- **JSON** for dungeon definition files
- Fully local, no network required for play, annotation, or editing
- **Optional**: the dungeon assistant at
  `http://127.0.0.1:8765/assistant` (chat-style room population from
  a theme + level + party level, with vision on the level PNG)
  subprocesses the local `claude` CLI and rides on Claude Code's
  existing subscription auth — no separate API key, no extra
  Python deps. Install Claude Code from claude.com/download and run
  `claude login` if you haven't already. The assistant is the only
  feature that calls out to a remote service, and only when the DM
  clicks Start session. Without Claude Code installed, the assistant
  page renders a setup-instructions card and the rest of the app
  behaves identically.
- **Optional**: `pip install -e ".[desktop]"` pulls in `pywebview`
  so the editor and Player view open as native desktop windows
  (system WebKit on macOS, no Chromium bundle) instead of browser
  tabs. Each window runs in its own subprocess to avoid the macOS
  main-thread conflict with pygame. Without pywebview installed the
  app falls back to `webbrowser.open` and everything still works.
- Target OS: macOS (cross-platform compatible)

---

## File Structure

```
dungeon-tracker/
├── CLAUDE.md
├── main.py                       ← entry point: open / list / reset a dungeon
├── dungeon.py                    ← graph data model, JSON loader
├── session.py                    ← SQLite session state manager
├── tracker.py                    ← turn engine, resource timers, WM rolls
├── renderer.py                   ← pygame map + UI rendering, annotation mode
├── editor_server.py              ← localhost browser-tab room/level editor
├── journal.py                    ← session log management
├── config.py                     ← constants (turn duration, light durations)
├── data/
│   └── example_dungeon.json      ← synthetic test fixture (no PNGs)
├── dungeons/                     ← one folder per dungeon (the unit of work)
│   └── ancient-temple-of-torrel/
│       ├── dungeon.json          ← metadata + levels + rooms + WM tables
│       ├── level1.png            ← map_image paths are bare filenames
│       ├── level2.png
│       ├── level3.png
│       └── session.db            ← per-dungeon SQLite (fog/turn/supplies)
├── render_output/                ← snapshot PNGs for the browser tabs
└── assets/
    └── fonts/
```

---

## Out of Scope (Do Not Build)

- Character sheet management — D&D Beyond
- Combat resolution — D&D Beyond Encounters
- Token-level fog of war, line of sight, per-token lighting
- Network or multiplayer features
- Audio
- Import from external map tools (Dungeon Scrawl, donjon, etc.)
- Any mechanic not present in the 5E PHB or DMG

---

## MVP Build Order

Each phase ends at a test gate and is independently runnable before proceeding. **pytest** for logic, **manual smoke** for visual/UX work.

### Phase 1 — Foundations & Data Model
- **Ships**: `dungeon.py` (graph data model, JSON loader, validator), `config.py` (turn duration, light durations, tag/state vocabularies), `data/example_dungeon.json`.
- **pytest**: JSON parses; graph is connected (no orphan rooms, every edge endpoint exists); invalid JSON is rejected with a clear error.
- **Gate**: `pytest -k dungeon` green; `python -c "from dungeon import load; load('data/example_dungeon.json')"` succeeds.

### Phase 2 — Turn Engine (headless)
- **Ships**: `tracker.py` (turn counter, elapsed-time, light timers, WM roll), `journal.py` (in-memory event log), CLI runner in `main.py` printing journal lines.
- **pytest**: turn advance ticks all timers; WM thresholds (d20 ≥ 18 hits, 1d6 = 1 hits) over a seeded RNG; torch expires after 6 turns; short-rest helper advances 6 turns and emits 6 WM checks.
- **Gate**: simulate a 50-turn run from a seeded RNG; journal output matches a recorded golden file.

### Phase 3 — SQLite Persistence
- **Ships**: `session.py` (schema + state manager: `sessions`, `room_state`, `resources`, `characters`, `active_effects`, `turn_log`, `party_position`); turn engine writes through it.
- **pytest**: save → close → reload restores tracker, journal, and resources byte-equal; plain-text journal export round-trips.
- **Gate**: kill mid-session, reload, advance one more turn — state is intact.

### Phase 4 — Map Renderer (graph layout)
- **Ships**: `renderer.py` skeleton — pygame window, parchment background `#f4e4c1`, thick black rectangles at node positions, thick corridor lines, click-to-reveal, drag-to-reposition, zoom + pan. Plain serif room labels.
- **Manual smoke**: example dungeon renders, every room visible in DM view, click reveals, drag persists position to session, zoom/pan responsive.
- **Gate**: visual checklist signed off; rendering does not regress test gates from Phases 1–3.

### Phase 5 — Map Renderer (pen-and-ink styling)
- **Ships**: crosshatch wall fill around rooms and corridors; tag icons (skull / chest / X / star); T-bar door symbol on `locked` and `trapped` edges; dashed line for `secret`; arrowhead for `one-way`; cleared-room cross-out; red outline overlay for active alerts and reaction-required rooms.
- **Manual smoke**: side-by-side comparison against `dungeon-art-1.png` and `dungeon-art-2.png` — does it read as Dyson?
- **Gate**: screenshot review; **Aesthetic Direction** spec satisfied point-by-point.

### Phase 6 — DM View Integration
- **Ships**: right panel (turn tracker, light sources, supplies, exhaustion, action buttons), bottom journal, all keyboard shortcuts wired (Space, R, N, S, L, Tab, Ctrl+S, Ctrl+L).
- **Manual smoke**: 30-turn playtest checklist — load → advance → WM rolls → torch warnings → manual notes → save/reload.
- **Gate**: DM playtest checklist passes end-to-end with no console errors.

### Phase 7 — Player View
- **Ships**: second pygame window (or Tab-toggle in single-window mode); fog-of-war filter — only revealed rooms, no notes, no DM alerts.
- **Manual smoke**: project to second display; confirm zero DM-only info leaks across views.
- **Gate**: DM/Player side-by-side comparison passes.

### Phase 8 — Polish (sub-phases, each independently mergeable)
- **8a Short Rest mode** — auto-advance 6 turns with WM checks; UI prompt. *pytest*: rest helper produces 6 WM events; *manual*: button works.
- **8b Long Rest mode** — auto-advance 48 turns. *pytest*: rest helper produces 48 WM events.
- **8c Exhaustion tracker** — per-character counter 0–6; red highlight at ≥3. *manual*: increment/decrement per character.
- **8d Noise flag** — toggles WM threshold for N turns. *pytest*: flag inflates threshold, expires after N.
- **8e Lantern + oil tracking** — flask burns down; refill from supplies. *pytest*: 36-turn lantern duration; oil decrement on refill.
- **8f Reaction-roll prompt** — modal on entering `reaction_required` rooms; logs DM-entered result. *manual*: prompt appears, journal entry recorded.
- **8g WM table rolls** — on encounter trigger, auto-roll on dungeon's WM table and display. *pytest*: roll picks correct table row.
