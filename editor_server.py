"""Tiny stdlib HTTP server for editing per-room metadata in a browser.

Architecture (per the plan):

    Pygame editor                Browser tab (this server)
        │                                │
        ▼                                ▼
        ──── data/<dungeon>.json (single source of truth) ────

The server holds **no** in-memory copy of the Dungeon — it re-reads the
JSON from disk on every request. The pygame editor polls the file mtime
once per second and reloads metadata fields when it changes. This means
the two processes never share mutable state.

`GET  /`         → render the editor form (one card per room)
`POST /room`     → urlencoded form, mutate the matching room, dump JSON,
                    303 redirect back to /
`GET  /healthz`  → 200 OK (used by tests)

Running:
    server, thread = start_editor_server(Path("data/torrel.json"), port=8765)
    # ... pygame loop ...
    server.shutdown()  # optional — daemon thread exits with the process

The handler class stores its config (dungeon path) on a class attribute
which `start_editor_server` populates before binding. We don't subclass
HTTPServer for that since it'd be more boilerplate.
"""

from __future__ import annotations

import html
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import character_ingester
import config
import dungeon as dungeon_mod
import dungeon_assistant
import encounter_simulator
import srd_lookup
import statblock_parser
from dungeon import WMTableEntry
from dungeon_assistant import AssistantSession, AssistantUnavailable


DEFAULT_PORT = 8765

# Path to the player-view PNG written by renderer.py. The /player
# route renders an HTML wrapper around this image; /player.png streams
# its bytes. The PNG is regenerated whenever fog state changes.
_PLAYER_PNG_PATH = Path(__file__).resolve().parent / "render_output" / "player_map.png"

# Player-view HTML. Black background fills the projector / TV when
# the map's aspect ratio doesn't match the screen. The img tag busts
# its own cache every 2 s so newly revealed rooms appear without the
# DM having to refresh anything.
_PLAYER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OSR Dungeon — Player Map</title>
<style>
  html, body { margin: 0; height: 100vh; background: #1a1a1a; overflow: hidden; }
  #map { width: 100vw; height: 100vh; object-fit: contain; display: block; }
  #status {
    position: fixed; bottom: 8px; right: 12px;
    font: 12px Georgia, 'Times New Roman', serif; color: #f4e4c1;
    background: rgba(0, 0, 0, 0.55);
    padding: 4px 8px; border-radius: 3px;
    pointer-events: none;
  }
  #empty {
    position: fixed; inset: 0;
    display: flex; align-items: center; justify-content: center;
    color: #826e50; font: 14px Georgia, serif;
  }
</style>
</head>
<body>
<img id="map" alt="Player map">
<div id="status">Player View · live</div>
<div id="empty" hidden>Waiting for the DM to reveal the first room…</div>
<script>
  var img = document.getElementById('map');
  var empty = document.getElementById('empty');
  function refresh() {
    var probe = new Image();
    probe.onload = function () {
      img.src = probe.src;
      empty.hidden = true;
    };
    probe.onerror = function () {
      empty.hidden = false;
      img.removeAttribute('src');
    };
    probe.src = '/player.png?t=' + Date.now();
  }
  refresh();
  setInterval(refresh, 2000);
</script>
</body>
</html>"""

# Tags shown as checkboxes; order matches CLAUDE.md.
TAG_OPTIONS = list(config.ROOM_TAGS)

# In-memory dungeon-assistant sessions, keyed by absolute dungeon path.
# Module-level on purpose: one editor_server process serves one dungeon
# folder, so this is effectively a single-slot cache. Surviving across
# requests is the point — losing the conversation on every refresh
# would be brutal. Cleared on /assistant/reset.
_assistant_sessions: dict[Path, AssistantSession] = {}


def _get_or_none_assistant(path: Path) -> AssistantSession | None:
    return _assistant_sessions.get(path.resolve())


# --- HTML rendering ----------------------------------------------------------


# Outer single-page-app shell. GET / returns this; the three real
# pages (room editor, assistant, characters) live inside iframes so
# their existing CSS/JS keeps working unchanged. The simulator is a
# fourth on-demand iframe loaded when the user clicks a room's
# "Simulate" button — it's not a persistent tab in the strip.
#
# Why iframes instead of in-page panel swaps:
#   - Each page has its own substantial CSS + JS already; isolating
#     them avoids selector and script-namespace clashes.
#   - All three persistent iframes stay loaded when the user tabs
#     away, so the existing BroadcastChannel-driven editor refresh
#     after an assistant Apply keeps working without surgery.
#   - Wrapping the whole thing in a pywebview window later doesn't
#     require any in-frame changes.
SHELL_CSS = """
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; background: #f4e4c1; }
body { font: 14px/1.4 Georgia, 'Times New Roman', serif; color: #1a1a1a; }

.tab-strip {
  display: flex;
  background: #1a1a1a;
  padding: 0;
  position: sticky; top: 0; z-index: 10;
  border-bottom: 2px solid #1a1a1a;
}
.tab-strip a {
  color: #d6caa8;
  text-decoration: none;
  padding: 0.65em 1.4em;
  font: 13px/1 Georgia, 'Times New Roman', serif;
  letter-spacing: 0.04em;
  border-right: 1px solid #3a2f25;
  cursor: pointer;
  user-select: none;
}
.tab-strip a:hover:not(.active) { background: #3a2f25; color: #f4e4c1; }
.tab-strip a.active {
  background: #f4e4c1;
  color: #1a1a1a;
  font-weight: bold;
}
.tab-strip a.close-tab {
  margin-left: 0.6em;
  padding: 0.65em 0.8em;
  color: #d6caa8;
  font-weight: bold;
}
.tab-strip a.close-tab:hover { color: #a8201a; background: #3a2f25; }
.tab-strip .dungeon-label {
  margin-left: auto;
  padding: 0.65em 1.2em;
  color: #826e50;
  font-style: italic;
  font-size: 12px;
}

.frame-wrap {
  position: relative;
  height: calc(100vh - 36px);
  overflow: hidden;
}
.frame-wrap iframe {
  position: absolute; inset: 0;
  width: 100%; height: 100%;
  border: 0;
  display: none;
  background: #f4e4c1;
}
.frame-wrap iframe.active { display: block; }
"""


# Bridge JS injected into every in-frame page so:
#   - clicks on internal anchors (/, /editor, /assistant, /characters,
#     /simulate?…) switch the parent shell's active tab instead of
#     navigating the iframe out of the shell, and
#   - the page's own Simulate buttons can ask the parent to open the
#     simulator tab via window.parent.postMessage.
# If the page is loaded outside the shell (window.parent === window),
# the bridge is a no-op and links behave normally.
FRAME_BRIDGE_JS = r"""
(function () {
  if (window.parent === window) return;  // not embedded in shell
  function isInternal(href) {
    if (!href) return false;
    return (href === '/' || href === '/editor'
            || href.startsWith('/editor?')
            || href === '/assistant'
            || href.startsWith('/assistant?')
            || href === '/characters'
            || href.startsWith('/characters?')
            || href.startsWith('/simulate?'));
  }
  document.addEventListener('click', function (e) {
    var a = e.target && e.target.closest && e.target.closest('a');
    if (!a) return;
    var href = a.getAttribute('href');
    if (!isInternal(href)) return;
    e.preventDefault();
    var target = (href === '/') ? '/editor' : href;
    window.parent.postMessage({type: 'nav', target: target}, '*');
  });
})();
"""


# Server-rendered shell. The iframes for Editor / Assistant / Characters
# start loaded so cross-tab broadcasts (e.g. assistant→editor refresh)
# work without any cold-start latency on first tab click.
def _render_app_shell(dungeon_name: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OSR - DM Assistant Tools</title>
<style>{SHELL_CSS}</style>
</head>
<body>
<nav class="tab-strip">
  <a id="tab-editor" data-tab="editor" class="active">Editor</a>
  <a id="tab-assistant" data-tab="assistant">Assistant</a>
  <a id="tab-characters" data-tab="characters">Characters</a>
  <a id="tab-simulate" data-tab="simulate" hidden>Simulator</a>
  <a id="tab-simulate-close" class="close-tab" hidden title="Close simulator">×</a>
  <span class="dungeon-label">{_esc(dungeon_name)}</span>
</nav>
<div class="frame-wrap">
  <iframe id="frame-editor" data-tab="editor" class="active" src="/editor"></iframe>
  <iframe id="frame-assistant" data-tab="assistant" src="/assistant"></iframe>
  <iframe id="frame-characters" data-tab="characters" src="/characters"></iframe>
  <iframe id="frame-simulate" data-tab="simulate" src="about:blank"></iframe>
</div>
<script>
(function () {{
  var tabEls = document.querySelectorAll('.tab-strip a[data-tab]');
  var frameEls = document.querySelectorAll('.frame-wrap iframe');
  var simTab = document.getElementById('tab-simulate');
  var simClose = document.getElementById('tab-simulate-close');
  var simFrame = document.getElementById('frame-simulate');

  function show(name) {{
    tabEls.forEach(function (t) {{
      t.classList.toggle('active', t.dataset.tab === name);
    }});
    frameEls.forEach(function (f) {{
      f.classList.toggle('active', f.dataset.tab === name);
    }});
    var hash = (name === 'editor') ? '' : '#tab=' + name;
    if (location.hash !== hash) {{
      history.replaceState(null, '', location.pathname + hash);
    }}
  }}

  function openSimulator(url) {{
    simFrame.src = url;
    simTab.hidden = false;
    simClose.hidden = false;
    show('simulate');
  }}

  function closeSimulator() {{
    simFrame.src = 'about:blank';
    simTab.hidden = true;
    simClose.hidden = true;
    show('editor');
  }}

  tabEls.forEach(function (t) {{
    t.addEventListener('click', function (e) {{
      e.preventDefault();
      show(t.dataset.tab);
    }});
  }});
  simClose.addEventListener('click', function (e) {{
    e.preventDefault();
    closeSimulator();
  }});

  // Honor #tab=<name> on first load (the simulator tab is excluded —
  // it needs a target URL from a child message, not just a name).
  var m = /^#tab=([a-z]+)/.exec(location.hash);
  if (m && m[1] !== 'simulate') show(m[1]);

  // Child-frame → shell bridge. Children postMessage({{type:'nav',target}}).
  window.addEventListener('message', function (ev) {{
    var d = ev.data || {{}};
    if (d.type !== 'nav' || !d.target) return;
    var t = d.target;
    if (t.startsWith('/simulate?')) {{
      openSimulator(t);
      return;
    }}
    var name =
      (t === '/editor' || t.startsWith('/editor?')) ? 'editor'
      : (t === '/assistant' || t.startsWith('/assistant?')) ? 'assistant'
      : (t === '/characters' || t.startsWith('/characters?')) ? 'characters'
      : null;
    if (name) show(name);
  }});
}})();
</script>
</body>
</html>
"""


PAGE_CSS = """
* { box-sizing: border-box; }
body {
  font: 14px/1.45 Georgia, 'Times New Roman', serif;
  background: #f4e4c1;
  color: #1a1a1a;
  max-width: 960px;
  margin: 0 auto;
  padding: 1em 1.5em 4em;
}
h1 {
  font-size: 1.6em;
  margin: 0.6em 0 0.3em;
}
h2.level {
  font-size: 1.2em;
  margin: 2em 0 0.6em;
  padding-bottom: 0.3em;
  border-bottom: 2px solid #1a1a1a;
}
.empty-level {
  color: #826e50;
  font-style: italic;
  margin: 0.5em 0 1em;
}
.level-card {
  background: #f0e3bf;
  border: 1px solid #826e50;
  border-radius: 6px;
  padding: 1em 1.2em;
  margin: 0.5em 0 1.2em;
}
.level-card .row { gap: 1em; align-items: flex-end; }
.level-card .row label { flex: 1; }
.wm-table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.4em 0 0.6em;
}
.wm-table th, .wm-table td {
  text-align: left;
  padding: 4px 6px;
  border-bottom: 1px solid #d6caa8;
}
.wm-table th {
  font-size: 0.82em;
  color: #826e50;
  font-weight: normal;
}
.wm-table td:first-child { width: 70px; }
.wm-table td:last-child  { width: 40px; text-align: center; }
.wm-table input { width: 100%; }
.wm-row-del {
  background: transparent;
  color: #a8201a;
  border: 0;
  font-size: 1.2em;
  cursor: pointer;
  padding: 0 6px;
}
.wm-row-del:hover { color: #1a1a1a; }
.wm-add-btn {
  background: transparent;
  color: #1a1a1a;
  border: 1px solid #826e50;
  padding: 0.3em 0.8em;
  border-radius: 3px;
  cursor: pointer;
  font: inherit;
}
.wm-add-btn:hover { background: #d6caa8; }
.room {
  background: #fffaf0;
  border: 1px solid #c8a96e;
  border-radius: 6px;
  padding: 1em 1.2em;
  margin: 1em 0;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.room-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.6em;
}
.room-id {
  font-family: 'Courier New', Courier, monospace;
  font-weight: bold;
  font-size: 1.1em;
  color: #826e50;
}
label {
  display: block;
  margin-top: 0.7em;
  font-weight: bold;
  font-size: 0.92em;
  letter-spacing: 0.02em;
}
.help {
  font-weight: normal;
  color: #826e50;
  font-style: italic;
  margin-left: 0.4em;
}
input[type=text], select, textarea {
  width: 100%;
  padding: 0.45em 0.55em;
  font: inherit;
  border: 1px solid #c8a96e;
  border-radius: 3px;
  background: #fffefb;
}
textarea { min-height: 4.5em; resize: vertical; line-height: 1.4; }
.tags {
  display: flex; flex-wrap: wrap; gap: 0.4em 0.9em;
  font-weight: normal;
}
.tags label {
  display: inline-flex; gap: 0.3em; align-items: center;
  font-weight: normal; margin-top: 0;
}
.row { display: flex; gap: 1em; }
.row > * { flex: 1; }
.save-row {
  display: flex; justify-content: flex-end; align-items: center;
  margin-top: 1em; gap: 0.8em;
}
.save-row .saved { color: #527a3e; font-style: italic; }
button {
  background: #1a1a1a;
  color: #f4e4c1;
  border: 0;
  padding: 0.55em 1.4em;
  font: inherit;
  font-weight: bold;
  border-radius: 3px;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}
button:hover { background: #3a2f25; }
/* Disabled / "saved" state: grey, no pointer. The page-load default is
 * disabled — any input event on the form switches the button back to
 * the dark active style. */
button:disabled, button.saved {
  background: #d6caa8;
  color: #826e50;
  cursor: not-allowed;
}
button:disabled:hover, button.saved:hover { background: #d6caa8; }
.enrich-row {
  display: flex; align-items: flex-start; gap: 0.8em;
  margin-top: 0.4em; flex-wrap: wrap;
}
.enrich-btn {
  background: transparent;
  color: #1a1a1a;
  border: 1px solid #826e50;
  padding: 0.4em 0.9em;
  border-radius: 3px;
  cursor: pointer;
  font: inherit;
  font-size: 0.9em;
  white-space: nowrap;
  transition: background 0.12s;
}
.enrich-btn:hover { background: #d6caa8; }
.enrich-btn:disabled { background: transparent; color: #826e50; cursor: progress; }
.simulate-btn {
  background: transparent;
  color: #a8201a;
  border: 1px solid #a8201a;
  padding: 0.4em 0.9em;
  border-radius: 3px;
  cursor: pointer;
  font: inherit;
  font-size: 0.9em;
  white-space: nowrap;
  transition: background 0.12s;
}
.simulate-btn:hover { background: #f0d6d2; }
.simulate-btn:disabled { background: transparent; color: #826e50; cursor: not-allowed; border-color: #c9b886; }
.enrich-empty { color: #826e50; font-style: italic; align-self: center; }
.statblocks-block {
  flex: 1 1 100%;
  border: 1px solid #c8a96e;
  border-radius: 4px;
  background: #fffaf0;
  padding: 0.4em 0.6em;
  margin-top: 0.4em;
}
.statblocks-block summary {
  cursor: pointer; font-weight: bold; color: #5a4830;
  padding: 0.2em 0;
}
.statblocks-block summary:hover { color: #1a1a1a; }
.statblocks-body {
  max-height: 420px;
  overflow-y: auto;
  margin: 0.4em 0 0;
  padding: 0.5em 0.7em;
  font: 12.5px/1.45 'Menlo', 'Monaco', 'Courier New', monospace;
  white-space: pre-wrap;
  background: #fbf6e6;
  border-top: 1px solid #d6caa8;
}
""".strip()


def _esc(s: str | None) -> str:
    """HTML-escape a string for safe interpolation."""
    return html.escape(s if s is not None else "", quote=True)


def _checkbox(name: str, value: str, checked: bool, label: str) -> str:
    c = " checked" if checked else ""
    return (f'<label><input type="checkbox" name="{name}" '
            f'value="{value}"{c}> {label}</label>')


def _select(name: str, options: tuple[str, ...], current: str | None) -> str:
    parts = [f'<select name="{name}">']
    for opt in options:
        sel = " selected" if (current or "") == opt else ""
        label = "—" if opt == "" else opt
        parts.append(f'  <option value="{_esc(opt)}"{sel}>{_esc(label)}</option>')
    parts.append('</select>')
    return "\n".join(parts)


def _render_statblocks_block(statblocks: str) -> str:
    """Render the stat-blocks panel under the Enrich button. Empty string
    when no enrichment has been pulled yet (so the form layout doesn't
    leave an empty <details>)."""
    if not statblocks.strip():
        return ('<span class="enrich-empty">'
                'No stat blocks pulled yet for this room.'
                '</span>')
    # Count creatures by counting top-level headings ("### " for Monsters
    # entries and "## " for Misc-creatures entries).
    n = sum(1 for line in statblocks.splitlines()
            if line.startswith("### ") or
            (line.startswith("## ") and not line.startswith("### ")))
    label = f"{n} creature{'' if n == 1 else 's'}" if n else "stat blocks"
    return (f'<details class="statblocks-block">'
            f'<summary>SRD stat blocks ({label})</summary>'
            f'<pre class="statblocks-body">{_esc(statblocks)}</pre>'
            f'</details>')


def _render_room(level_number: int, room) -> str:
    tag_checks = "\n".join(
        _checkbox("tags", t, t in room.tags, t) for t in TAG_OPTIONS
    )
    return f"""
<div class="room" id="room-{_esc(room.id)}">
  <form method="post" action="/room">
    <input type="hidden" name="level_number" value="{level_number}">
    <input type="hidden" name="room_id" value="{_esc(room.id)}">
    <div class="room-head">
      <span class="room-id">{_esc(room.id)}</span>
    </div>

    <label>Name
      <input type="text" name="name" value="{_esc(room.name)}">
    </label>

    <label>Tags
      <div class="tags">{tag_checks}</div>
    </label>

    <label class="tags" style="margin-top: 0.5em;">
      <input type="checkbox" name="reaction_required" value="1"
        {"checked" if room.reaction_required else ""}>
      Reaction required (prompt for reaction roll on first entry)
    </label>

    <label>Box text <span class="help">read aloud to players</span>
      <textarea name="box_text" rows="4">{_esc(room.box_text)}</textarea>
    </label>

    <label>DM notes <span class="help">private — moods, secrets, foreshadowing</span>
      <textarea name="notes" rows="3">{_esc(room.notes)}</textarea>
    </label>

    <label>Encounter details <span class="help">monsters, tactics, dispositions</span>
      <textarea name="encounter_text" rows="3">{_esc(room.encounter_text)}</textarea>
    </label>
    <div class="enrich-row">
      <button type="button" class="enrich-btn" onclick="enrichRoom(this)"
        title="Scan the encounter text + encounter ref for SRD creature names and pull their stat blocks. Replaces any prior stat blocks for this room."
        >⚡ Enrich from SRD</button>
      <button type="button" class="simulate-btn" onclick="simulateRoom(this)"
        {"" if room.statblocks.strip() else "disabled"}
        title="Run a Monte Carlo combat simulation of this encounter against your party. Requires SRD enrichment first, plus character sheets uploaded on the Characters page."
        >🗡️ Simulate</button>
      {_render_statblocks_block(room.statblocks)}
    </div>

    <label>Treasure details <span class="help">items, gold, location, hidden vs obvious</span>
      <textarea name="treasure_text" rows="3">{_esc(room.treasure_text)}</textarea>
    </label>

    <label>Special details <span class="help">unique features, mechanisms, lore — paired with the `special` tag</span>
      <textarea name="special_text" rows="3">{_esc(room.special_text)}</textarea>
    </label>

    <div class="row">
      <label>Encounter ref <span class="help">D&amp;D Beyond encounter name</span>
        <input type="text" name="encounter_ref" value="{_esc(room.encounter_ref or "")}">
      </label>
      <label>Treasure tier <span class="help">free-text label, e.g. "DMG CR 0–4 individual"</span>
        <input type="text" name="treasure_tier" value="{_esc(room.treasure_tier or "")}">
      </label>
    </div>

    <div class="save-row">
      <button type="submit" disabled>Save</button>
    </div>
  </form>
</div>
""".strip()


# ---------------------------------------------------------------------------
# Encounter simulator + character pages
# ---------------------------------------------------------------------------


def _build_monsters_from_room(room) -> list[statblock_parser.ParsedMonster]:
    """Extract a flat list of ParsedMonster for a room's encounter.

    Reads the formal `<count> <Name>(s) (MM p.<page>)` declarations
    out of `room.encounter_text` via `parse_encounter_declarations`.
    Descriptive prose between declarations is ignored — that's the
    whole point of the page-reference anchor.

    Dice counts ("1d4 Goblins") roll once with a fixed seed so the
    same room always produces the same monster list; re-rolling on
    every visit would make Monte Carlo results impossible to compare.
    """
    rng = __import__("random").Random(0)
    entries = srd_lookup.parse_encounter_declarations(room.encounter_text or "")
    monsters: list[statblock_parser.ParsedMonster] = []
    for entry in entries:
        count = statblock_parser.roll_count(entry.count_expr, rng)
        parsed = statblock_parser.parse(entry.statblock)
        monsters.extend([parsed] * count)
    return monsters


def _render_characters_page_body(*, dungeon_path: Path,
                                 characters: list[dict],
                                 cli_ok: bool, cli_error: str) -> str:
    folder = dungeon_path.parent.name
    cards = "\n".join(_render_character_card(c) for c in characters) \
            if characters else (
        '<p class="empty-level">No characters uploaded yet. '
        'Upload one PDF per party member below.</p>'
    )
    setup_card = "" if cli_ok else f"""
<div class="setup-card">
  <h3>Claude Code CLI not available</h3>
  <p>{_esc(cli_error)}</p>
  <p>Once installed and signed in, return to this page to upload character
  sheets. The simulator will work with hand-edited JSON files in
  <code>{_esc(str(dungeon_path.parent / "characters"))}</code> in the
  meantime.</p>
</div>
""".strip()
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Characters — {_esc(folder)}</title>
<style>
{PAGE_CSS}
.setup-card {{
  background: #fdf3df; border: 1px solid #c9b886; border-radius: 6px;
  padding: 0.8em 1em; margin: 1em 0;
}}
.setup-card h3 {{ margin: 0 0 0.3em; color: #826e50; }}
.character-card {{
  background: #f0e3bf; border: 1px solid #826e50; border-radius: 6px;
  padding: 0.8em 1em; margin: 0.6em 0;
}}
.character-card h3 {{ margin: 0 0 0.3em; }}
.character-card .meta {{ color: #5a4830; font-size: 0.92em; }}
.upload-card {{
  background: #f0e3bf; border: 2px dashed #826e50; border-radius: 6px;
  padding: 1em 1.2em; margin: 1em 0;
}}
.upload-card progress {{ width: 100%; }}
.upload-result {{ margin-top: 0.6em; font-size: 0.95em; }}
.upload-result.ok {{ color: #1a6b1a; }}
.upload-result.err {{ color: #a8201a; }}
.delete-form {{ display: inline; margin: 0; }}
.delete-form button {{
  background: transparent; color: #a8201a; border: 1px solid #a8201a;
  font-size: 0.85em; padding: 0.2em 0.6em; border-radius: 3px;
  cursor: pointer;
}}
</style>
</head>
<body>
<h1>Characters — <em>{_esc(folder)}</em></h1>
<p><a href="/">← back to room editor</a></p>

{setup_card}

<h2 class="level">Party</h2>
{cards}

<h2 class="level">Upload character sheet</h2>
<div class="upload-card">
  <form id="upload-form">
    <label>PDF file
      <input type="file" id="pdf-input" accept="application/pdf,.pdf" required>
    </label>
    <label>Filename override <span class="help">optional — defaults to a slug of the character's name</span>
      <input type="text" id="filename-input" placeholder="e.g. thorin">
    </label>
    <div class="save-row">
      <button type="submit" id="upload-btn">Extract &amp; save</button>
    </div>
  </form>
  <progress id="upload-progress" hidden></progress>
  <div id="upload-result" class="upload-result"></div>
</div>

<script>
(function () {{
  var form = document.getElementById('upload-form');
  var fileInput = document.getElementById('pdf-input');
  var nameInput = document.getElementById('filename-input');
  var btn = document.getElementById('upload-btn');
  var progress = document.getElementById('upload-progress');
  var result = document.getElementById('upload-result');
  form.addEventListener('submit', function (e) {{
    e.preventDefault();
    var file = fileInput.files[0];
    if (!file) {{
      result.textContent = 'Pick a PDF file first.';
      result.className = 'upload-result err';
      return;
    }}
    var name = (nameInput.value || '').trim();
    var url = '/characters/upload';
    if (name) url += '?filename=' + encodeURIComponent(name);
    btn.disabled = true;
    btn.textContent = 'Extracting (LLM call, ~30–60s)…';
    progress.hidden = false;
    result.textContent = '';
    fetch(url, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/octet-stream'}},
      body: file,
    }})
    .then(function (resp) {{
      return resp.json().then(function (data) {{
        return {{ status: resp.status, data: data }};
      }});
    }})
    .then(function (out) {{
      progress.hidden = true;
      btn.disabled = false;
      btn.textContent = 'Extract & save';
      if (out.data.ok) {{
        result.textContent = 'Saved ' + out.data.filename + ' (' + (out.data.name || 'unnamed') + '). Reloading…';
        result.className = 'upload-result ok';
        setTimeout(function () {{ location.reload(); }}, 700);
      }} else {{
        result.textContent = 'Failed: ' + (out.data.error || 'unknown error');
        result.className = 'upload-result err';
      }}
    }})
    .catch(function (err) {{
      progress.hidden = true;
      btn.disabled = false;
      btn.textContent = 'Extract & save';
      result.textContent = 'Upload error: ' + err.message;
      result.className = 'upload-result err';
    }});
  }});
}})();
</script>
<script>{FRAME_BRIDGE_JS}</script>
</body>
</html>
"""


def _render_character_card(c: dict) -> str:
    name = _esc(str(c.get("name") or "Unnamed"))
    cls = _esc(str(c.get("class") or "?"))
    level = c.get("level") or "?"
    ac = c.get("ac") or "?"
    hp = c.get("hp_max") or "?"
    attacks = c.get("attacks") or []
    atk_lines = "; ".join(
        f"{_esc(a.get('name', '?'))} (+{a.get('to_hit', 0)}, {_esc(a.get('damage', '?'))})"
        for a in attacks
    ) or "<em>none</em>"
    spells = c.get("spells", {}).get("memorized", [])
    spell_count = len(spells)
    filename = (
        character_ingester._slug(str(c.get("name", "character"))) + ".json"
    )
    pretty = _esc(json.dumps(c, indent=2))
    return f"""
<div class="character-card">
  <h3>{name} <span class="meta">— {cls} {level}</span></h3>
  <div class="meta">AC {ac}, HP {hp}, {spell_count} memorised spells</div>
  <div class="meta">Attacks: {atk_lines}</div>
  <details>
    <summary>Full JSON</summary>
    <pre style="font-size:11.5px; overflow-x:auto;">{pretty}</pre>
  </details>
  <form class="delete-form" method="post" action="/characters/delete"
        onsubmit="return confirm('Delete {name}?')">
    <input type="hidden" name="filename" value="{_esc(filename)}">
    <button type="submit">Delete</button>
  </form>
</div>
""".strip()


def _render_simulate_page_body(*, dungeon_name: str, level_number: int,
                               room, characters: list[dict],
                               monsters: list[statblock_parser.ParsedMonster],
                               trials: int,
                               party_level_override: int | None = None,
                               seed: int = 0) -> str:
    """Run the Monte Carlo if both sides are present; otherwise render
    a clear setup card explaining what's missing.

    When `party_level_override` is set, the supplied `characters` have
    already been scaled to that level by the caller; we surface the
    fact on the results page so the DM knows what they're seeing.

    `seed` is the Monte Carlo base seed. Rendered as the form's
    Seed field, blank-by-default so the next Re-run randomises
    again; user can paste a specific seed to reproduce a run."""
    title = (f"Simulate — {dungeon_name} · L{level_number} · "
             f"{room.id} {room.name}")

    if not characters:
        return _render_simulate_setup(
            title=title,
            level_number=level_number, room_id=room.id,
            problem="No party characters uploaded yet.",
            fix=('Upload character sheets on the '
                 '<a href="/characters">Characters</a> page first.'),
        )
    if not monsters:
        return _render_simulate_setup(
            title=title,
            level_number=level_number, room_id=room.id,
            problem=(
                "No monsters could be parsed from this room's encounter "
                "text."
            ),
            fix=(
                'Edit the encounter text on the <a href="/">room editor</a>, '
                'then click "⚡ Enrich from SRD". The simulator pulls '
                'creatures whose names appear in the encounter text and '
                'reads their counts (e.g. "2 Ghouls", "1d4 Skeletons").'
            ),
        )

    # Run the MC with the (possibly randomised) seed.
    report = encounter_simulator.monte_carlo(
        characters, monsters, trials=trials, base_seed=seed,
    )

    party_lines = "\n".join(
        f"<li>{_esc(c.get('name', '?'))} — {_esc(c.get('class', '?'))} "
        f"{c.get('level', '?')}, AC {c.get('ac', '?')}, "
        f"HP {c.get('hp_max', '?')}</li>"
        for c in characters
    )
    monster_counts: dict[str, int] = {}
    for m in monsters:
        monster_counts[m.name] = monster_counts.get(m.name, 0) + 1
    monster_lines = "\n".join(
        f"<li>{count}× {_esc(name)}</li>"
        for name, count in sorted(monster_counts.items())
    )

    trace_html = "\n".join(_esc(line) for line in report.sample_trace)

    win_class = ("good" if report.party_win_pct >= 70.0
                 else "warn" if report.party_win_pct >= 30.0
                 else "bad")
    tpk_class = ("bad" if report.tpk_pct >= 30.0
                 else "warn" if report.tpk_pct >= 5.0
                 else "good")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(title)}</title>
<style>
{PAGE_CSS}
.stats {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1em; margin: 1em 0;
}}
.stat {{
  background: #f0e3bf; border: 1px solid #826e50; border-radius: 6px;
  padding: 0.8em 1em;
}}
.stat .label {{ font-size: 0.85em; color: #826e50; text-transform: uppercase; letter-spacing: 0.04em; }}
.stat .value {{ font-size: 1.6em; font-weight: bold; margin-top: 0.2em; }}
.stat.good .value {{ color: #1a6b1a; }}
.stat.warn .value {{ color: #b8861f; }}
.stat.bad  .value {{ color: #a8201a; }}
.setup-grid {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em; margin-bottom: 1em;
}}
.setup-grid h3 {{ margin: 0 0 0.4em; }}
.setup-grid ul {{ margin: 0; padding-left: 1.2em; }}
pre.trace {{
  background: #fbf6e6; border: 1px solid #d6caa8;
  padding: 0.8em 1em; max-height: 70vh; overflow-y: auto;
  font: 12px/1.5 'Menlo', 'Monaco', 'Courier New', monospace;
  white-space: pre-wrap;
}}
.params-form {{
  background: #f0e3bf; border: 1px solid #826e50; border-radius: 6px;
  padding: 0.6em 1em; margin: 1em 0;
}}
.params-form label {{ display: inline-block; margin-right: 1em; }}
.params-form button {{
  background: transparent; color: #1a1a1a; border: 1px solid #826e50;
  padding: 0.4em 0.9em; border-radius: 3px; cursor: pointer;
}}
.params-form .help {{
  display: block; color: #826e50; font-size: 0.8em; margin-top: 2px;
}}
.scale-note {{
  background: #fdf3df; border: 1px solid #c9b886; border-radius: 6px;
  padding: 0.6em 1em; margin: 0.8em 0; font-size: 0.92em;
}}
.scale-note strong {{ color: #826e50; }}
</style>
</head>
<body>
<h1>{_esc(title)}</h1>
<p><a href="/">← back to room editor</a> · <a href="/characters">manage characters</a></p>

<form class="params-form" method="get" action="/simulate">
  <input type="hidden" name="level_number" value="{level_number}">
  <input type="hidden" name="room_id" value="{_esc(room.id)}">
  <label>Trials
    <input type="number" name="trials" value="{trials}" min="1" max="500" style="width: 70px;">
  </label>
  <label>Party level
    <input type="number" name="party_level" value="{party_level_override or ''}" min="1" max="20" placeholder="auto" style="width: 70px;">
    <span class="help">leave blank to use each character's stored level</span>
  </label>
  <label>Seed
    <input type="number" name="seed" value="" placeholder="random" style="width: 110px;">
    <span class="help">this run used <code>{seed}</code> — paste it back here to reproduce; leave blank to roll fresh dice</span>
  </label>
  <button type="submit">Re-run</button>
</form>

{_render_scale_note(party_level_override) if party_level_override else ''}

<div class="setup-grid">
  <div>
    <h3>Party ({len(characters)})</h3>
    <ul>{party_lines}</ul>
  </div>
  <div>
    <h3>Monsters ({len(monsters)})</h3>
    <ul>{monster_lines}</ul>
  </div>
</div>

<div class="stats">
  <div class="stat {win_class}">
    <div class="label">Party win</div>
    <div class="value">{report.party_win_pct:.0f}%</div>
  </div>
  <div class="stat {tpk_class}">
    <div class="label">TPK</div>
    <div class="value">{report.tpk_pct:.0f}%</div>
  </div>
  <div class="stat">
    <div class="label">Avg party HP</div>
    <div class="value">{report.avg_party_hp_pct:.0f}%</div>
  </div>
  <div class="stat">
    <div class="label">Avg rounds</div>
    <div class="value">{report.avg_rounds:.1f}</div>
  </div>
  <div class="stat">
    <div class="label">MVP</div>
    <div class="value" style="font-size:1.2em;">{_esc(report.mvp_name)}</div>
    <div class="label">{report.mvp_avg_damage:.1f} dmg/trial</div>
  </div>
</div>

<details open>
  <summary><strong>Sample combat trace (trial #0)</strong></summary>
  <pre class="trace">{trace_html}</pre>
</details>

<p style="color:#826e50; font-size:0.88em;">
Aggregated over {report.trials} independent trials. Combat abstractions:
no positioning grid, no opportunity attacks, no concentration, no
condition effects (paralysis riders are noted in the trace but not
applied). At 0&nbsp;HP a PC is "down" and out of the fight.
</p>
<script>{FRAME_BRIDGE_JS}</script>
</body>
</html>
"""


def _render_scale_note(party_level: int) -> str:
    """Banner shown above the stats when the party has been scaled to
    a non-stored level. Names what the scaler does and doesn't change
    so the DM doesn't over-interpret the result."""
    return f"""
<div class="scale-note">
  <strong>Party scaled to level {party_level}.</strong>
  HP, Rogue sneak-attack dice, spell slots, and Fighter/Paladin/Ranger
  Extra Attack at L≥5 have been adjusted. <em>Spells known and
  attack-to-hit bonuses are unchanged</em> — a caster scaled to L5
  has L3 slots but still casts the spells listed in their JSON, so
  they're slightly underpowered vs. a true L5 sheet.
</div>
""".strip()


def _render_simulate_setup(*, title: str, level_number: int, room_id: str,
                           problem: str, fix: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(title)}</title>
<style>
{PAGE_CSS}
.setup-card {{
  background: #fdf3df; border: 1px solid #c9b886; border-radius: 6px;
  padding: 1em 1.2em; margin: 1em 0;
}}
.setup-card h3 {{ margin: 0 0 0.4em; color: #826e50; }}
</style>
</head>
<body>
<h1>{_esc(title)}</h1>
<p><a href="/">← back to room editor</a></p>
<div class="setup-card">
  <h3>Cannot simulate yet</h3>
  <p>{problem}</p>
  <p>{fix}</p>
</div>
<script>{FRAME_BRIDGE_JS}</script>
</body>
</html>
"""


PAGE_JS = """
// Dirty-tracking + AJAX save. Both use event delegation off document.body
// so they keep working after we splice in a freshly-rendered card; we
// don't need to re-bind anything when DOM is replaced.
//
// The reason for AJAX: a 303 + full-page-reload after a single-room save
// would re-render every other form from disk, blowing away in-progress
// edits the DM has typed but not yet saved. With AJAX we swap only the
// affected card.

document.body.addEventListener('input', function (e) {
  var form = e.target.closest && e.target.closest('form');
  if (!form) return;
  var btn = form.querySelector('button[type=submit]');
  if (btn) { btn.disabled = false; btn.classList.remove('saved'); }
});
document.body.addEventListener('change', function (e) {
  var form = e.target.closest && e.target.closest('form');
  if (!form) return;
  var btn = form.querySelector('button[type=submit]');
  if (btn) { btn.disabled = false; btn.classList.remove('saved'); }
});

document.body.addEventListener('submit', function (e) {
  var form = e.target;
  if (!form || form.tagName !== 'FORM') return;
  e.preventDefault();
  var btn = form.querySelector('button[type=submit]');
  var origLabel = btn ? btn.textContent : '';
  if (btn) { btn.disabled = true; btn.textContent = 'Saving…'; }

  // Build URL-encoded body (the server uses parse_qs).
  var fd = new FormData(form);
  var params = new URLSearchParams();
  fd.forEach(function (v, k) { params.append(k, v); });

  fetch(form.action, {
    method: form.method || 'POST',
    body: params.toString(),
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      'X-Editor-Fragment': '1'
    }
  }).then(function (resp) {
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    return resp.text();
  }).then(function (htmlFrag) {
    // Replace the saved card (.room or .level-card) with the freshly
    // rendered fragment. Sibling cards in the page are untouched, so
    // their unsaved edits survive.
    var container = form.closest('.room, .level-card');
    if (container) {
      var tmp = document.createElement('div');
      tmp.innerHTML = htmlFrag.trim();
      var fresh = tmp.firstElementChild;
      if (fresh) {
        container.replaceWith(fresh);
        var newBtn = fresh.querySelector('button[type=submit]');
        if (newBtn) {
          var saved = origLabel || newBtn.textContent;
          newBtn.classList.add('saved');
          newBtn.textContent = 'Saved ✓';
          newBtn.disabled = true;
          setTimeout(function () {
            newBtn.classList.remove('saved');
            newBtn.textContent = saved;
          }, 1400);
        }
      }
    }
  }).catch(function (err) {
    if (btn) { btn.disabled = false; btn.textContent = origLabel; }
    alert('Save failed: ' + err.message);
  });
});

// WM-table row editing for the level-settings card. addWmRow appends
// an empty row to the same table the clicked button belongs to;
// removeWmRow strips its row from the DOM. Both flag the form dirty.
function addWmRow(btn) {
  var form = btn.closest('form');
  var tbody = form.querySelector('table.wm-table tbody');
  var tr = document.createElement('tr');
  tr.innerHTML =
    '<td><input name="wm_roll" type="number" min="1"></td>' +
    '<td><input name="wm_encounter" type="text"></td>' +
    '<td><button type="button" class="wm-row-del" ' +
        'onclick="removeWmRow(this)" title="remove row">×</button></td>';
  tbody.appendChild(tr);
  form.querySelector('button[type=submit]').disabled = false;
}

function removeWmRow(btn) {
  var tr = btn.closest('tr');
  var form = tr.closest('form');
  tr.remove();
  form.querySelector('button[type=submit]').disabled = false;
}

// Enrich button — POSTs to /enrich with the room form's hidden ids,
// then swaps the room card with the freshly rendered fragment so the
// new stat blocks appear inline. Skips the form's own submit handler.
function enrichRoom(btn) {
  var card = btn.closest('.room');
  var form = card && card.querySelector('form');
  if (!form) return;
  var levelInput = form.querySelector('input[name="level_number"]');
  var roomInput  = form.querySelector('input[name="room_id"]');
  if (!levelInput || !roomInput) return;
  var origLabel = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Enriching…';

  var params = new URLSearchParams();
  params.append('level_number', levelInput.value);
  params.append('room_id', roomInput.value);

  fetch('/enrich', {
    method: 'POST',
    body: params.toString(),
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      'X-Editor-Fragment': '1'
    }
  }).then(function (resp) {
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    return resp.text();
  }).then(function (htmlFrag) {
    var tmp = document.createElement('div');
    tmp.innerHTML = htmlFrag.trim();
    var fresh = tmp.firstElementChild;
    if (fresh) card.replaceWith(fresh);
  }).catch(function (err) {
    btn.disabled = false;
    btn.textContent = origLabel;
    alert('Enrich failed: ' + err.message);
  });
}

// Simulate button — open the encounter simulation in the shell's
// on-demand "Simulator" tab. When loaded inside the SPA shell we
// postMessage the URL to the parent; when loaded standalone we fall
// back to opening a new browser tab.
function simulateRoom(btn) {
  if (btn.disabled) return;
  var card = btn.closest('.room');
  var form = card && card.querySelector('form');
  if (!form) return;
  var levelInput = form.querySelector('input[name="level_number"]');
  var roomInput  = form.querySelector('input[name="room_id"]');
  if (!levelInput || !roomInput) return;
  var url = '/simulate?level_number=' + encodeURIComponent(levelInput.value) +
            '&room_id=' + encodeURIComponent(roomInput.value);
  if (window.parent !== window) {
    window.parent.postMessage({type: 'nav', target: url}, '*');
  } else {
    window.open(url, '_blank');
  }
}

// Listen on the cross-tab BroadcastChannel so when the /assistant tab
// applies a room, the editor page swaps that one card in place
// without a full reload (which would wipe any unsaved sibling edits).
(function () {
  if (typeof BroadcastChannel !== 'function') return;
  var bc = new BroadcastChannel('osr-dungeon-editor');
  bc.addEventListener('message', function (e) {
    var data = e.data || {};
    if (data.type !== 'room-applied' || !data.room_id) return;
    var card = document.getElementById('room-' + data.room_id);
    if (!card) return;
    var levelForm = card.querySelector('input[name="level_number"]');
    var levelNum = levelForm ? levelForm.value : '';
    if (!levelNum) return;
    fetch('/room?level_number=' + encodeURIComponent(levelNum) +
          '&room_id=' + encodeURIComponent(data.room_id))
      .then(function (resp) {
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        return resp.text();
      })
      .then(function (htmlFrag) {
        var tmp = document.createElement('div');
        tmp.innerHTML = htmlFrag.trim();
        var fresh = tmp.firstElementChild;
        if (!fresh) return;
        // Brief highlight so the DM sees what just changed.
        fresh.style.transition = 'background 1.2s';
        fresh.style.background = '#e8f0d8';
        card.replaceWith(fresh);
        setTimeout(function () { fresh.style.background = ''; }, 1500);
      })
      .catch(function () { /* silent — editor stays as-is */ });
  });
})();
""".strip()


def _render_level_card(level) -> str:
    """Settings card at the top of each level: display name, challenge
    rating, WM rules, full WM table editor."""
    method_options = "".join(
        f'<option value="{_esc(m)}"'
        f'{" selected" if level.wm_check_method == m else ""}>{_esc(m)}</option>'
        for m in config.WM_METHODS
    )
    table_rows = "\n".join(
        _render_wm_row(e.roll, e.encounter)
        for e in level.wandering_monster_table
    )
    return f"""
<div class="level-card" id="level-{level.level_number}">
  <form method="post" action="/level" class="level-form">
    <input type="hidden" name="level_number" value="{level.level_number}">

    <div class="row">
      <label>Display name
        <input type="text" name="display_name"
               value="{_esc(level.display_name)}">
      </label>
      <label>Challenge rating <span class="help">free-text reminder</span>
        <input type="text" name="challenge_rating"
               value="{_esc(level.challenge_rating)}"
               placeholder="e.g. CR 1/4–1 (standard) · CR 2 (deadly)">
      </label>
    </div>

    <div class="row">
      <label>WM check method
        <select name="wm_check_method">{method_options}</select>
      </label>
      <label>Threshold <span class="help">d20: encounter on ≥ ; d6: ≤</span>
        <input type="number" name="wm_check_threshold" min="1" max="20"
               value="{level.wm_check_threshold}">
      </label>
      <label>Check every N turns
        <input type="number" name="wm_check_every_n_turns" min="1"
               value="{level.wm_check_every_n_turns}">
      </label>
    </div>

    <label>Wandering monster table
      <span class="help">roll 1..N; rolls auto-sorted on save</span>
    </label>
    <table class="wm-table">
      <thead><tr><th>Roll</th><th>Encounter</th><th></th></tr></thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
    <button type="button" class="wm-add-btn" onclick="addWmRow(this)">+ Add row</button>

    <div class="save-row">
      <button type="submit" disabled>Save Level Settings</button>
    </div>
  </form>
</div>
""".strip()


def _render_wm_row(roll: int, encounter: str) -> str:
    return (
        '        <tr>'
        f'<td><input name="wm_roll" type="number" min="1" value="{int(roll)}"></td>'
        f'<td><input name="wm_encounter" type="text" value="{_esc(encounter)}"></td>'
        '<td><button type="button" class="wm-row-del" '
        'onclick="removeWmRow(this)" title="remove row">×</button></td>'
        '</tr>'
    )


def _render_page(d, *, saved_room_id: str | None = None,
                 saved_level_number: int | None = None) -> str:
    """Build the full HTML page from a Dungeon."""
    sections = []
    for level in d.levels:
        sections.append(
            f'<h2 class="level">Level {level.level_number}</h2>'
        )
        sections.append(_render_level_card(level))
        if not level.rooms:
            sections.append(
                '<p class="empty-level">'
                'No rooms annotated yet. Draw rectangles in the pygame editor '
                '(press <code>A</code>) and they will appear here.'
                '</p>'
            )
            continue
        for room in level.rooms:
            sections.append(_render_room(level.level_number, room))

    saved_banner = ""
    saved_label = saved_room_id
    if saved_level_number is not None:
        saved_label = f"Level {saved_level_number}"
    if saved_label:
        saved_banner = (
            f'<p class="saved" style="text-align:right;color:#527a3e;'
            f'font-style:italic;">Saved {_esc(saved_label)}.</p>'
        )

    body = "\n".join(sections)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Room Editor — {_esc(d.name)}</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<h1>{_esc(d.name)}</h1>
<p style="color:#826e50;">
  Edit per-room metadata. Changes save to <code>{_esc(str(EditorHandler.dungeon_path))}</code>;
  the pygame editor reloads them within a second.
  &nbsp;·&nbsp;
  <a href="/assistant" style="color:#5a4830;">Open dungeon assistant →</a>
  &nbsp;·&nbsp;
  <a href="/characters" style="color:#5a4830;">Manage characters →</a>
</p>
{saved_banner}
{body}
<script>{PAGE_JS}</script>
<script>{FRAME_BRIDGE_JS}</script>
</body>
</html>
"""


# --- Assistant page rendering -----------------------------------------------


def _turn_to_json(turn) -> dict:
    """Serialise an AssistantTurn for the JSON wire."""
    return {
        "text": turn.text,
        "summary": turn.summary,
        "usage": turn.usage,
        "cost_usd": turn.cost_usd,
        "proposals": [p.to_dict() for p in turn.proposals],
        "rejected": turn.rejected,
    }


ASSISTANT_PAGE_CSS = """
* { box-sizing: border-box; }
body {
  font: 14px/1.45 Georgia, 'Times New Roman', serif;
  background: #f4e4c1;
  color: #1a1a1a;
  max-width: 1100px;
  margin: 0 auto;
  padding: 1em 1.5em 4em;
}
h1 { font-size: 1.6em; margin: 0.6em 0 0.3em; }
.subtle { color: #826e50; }
a { color: #5a4830; }

.assistant-bar {
  background: #f0e3bf;
  border: 1px solid #826e50;
  border-radius: 6px;
  padding: 1em 1.2em;
  margin: 0.5em 0 1em;
  display: flex; flex-wrap: wrap; gap: 0.8em 1.2em;
  align-items: flex-end;
}
.assistant-bar label {
  font-weight: bold; font-size: 0.92em;
  display: flex; flex-direction: column; gap: 0.2em;
}
.assistant-bar input, .assistant-bar select, .assistant-bar textarea {
  font: inherit;
  padding: 0.4em 0.55em;
  border: 1px solid #c8a96e;
  border-radius: 3px;
  background: #fffefb;
  min-width: 14em;
}
.assistant-bar .theme-field {
  flex: 1 1 100%;
}
.assistant-bar .theme-field textarea {
  width: 100%;
  min-height: 4.5em;
  resize: vertical;
  font: inherit;
  line-height: 1.4;
}
.assistant-bar button {
  background: #1a1a1a; color: #f4e4c1;
  border: 0; padding: 0.55em 1.2em;
  font: inherit; font-weight: bold;
  border-radius: 3px; cursor: pointer;
}
.assistant-bar button:hover { background: #3a2f25; }
.assistant-bar button.secondary {
  background: transparent; color: #1a1a1a;
  border: 1px solid #826e50;
}
.assistant-bar button.secondary:hover { background: #d6caa8; }
.assistant-bar button:disabled {
  background: #d6caa8; color: #826e50; cursor: not-allowed;
}

.chat-log {
  background: #fffaf0;
  border: 1px solid #c8a96e;
  border-radius: 6px;
  padding: 1em 1.2em;
  min-height: 280px;
  margin: 1em 0;
}
.chat-log:empty::before {
  content: 'Set theme/level/party level above and press Start session.';
  color: #826e50; font-style: italic;
}
.turn { margin: 0 0 1.4em; }
.turn-header {
  font-weight: bold; font-size: 0.9em;
  color: #5a4830;
  margin-bottom: 0.3em;
  text-transform: uppercase; letter-spacing: 0.06em;
}
.turn.user .turn-header { color: #2a4a6a; }
.turn-text {
  white-space: pre-wrap;
  border-left: 3px solid #c8a96e;
  padding: 0.2em 0 0.2em 0.8em;
  margin: 0 0 0.6em;
}
.turn.user .turn-text { border-left-color: #6a8aaa; }

.proposal {
  border: 1px solid #c8a96e;
  background: #fbf6e6;
  border-radius: 4px;
  padding: 0.7em 0.9em;
  margin: 0.6em 0;
}
.proposal.fuzzy { border-color: #a8201a; }
.proposal-head {
  display: flex; justify-content: space-between;
  align-items: baseline; gap: 0.6em;
  margin-bottom: 0.4em;
}
.proposal-id {
  font-family: 'Courier New', Courier, monospace;
  font-weight: bold; color: #5a4830;
}
.proposal-name { font-weight: bold; }
.proposal-tags {
  font-size: 0.85em; color: #826e50;
  font-family: 'Courier New', Courier, monospace;
}
.proposal-field {
  margin: 0.3em 0;
}
.proposal-field-label {
  font-size: 0.8em; font-weight: bold;
  color: #5a4830; text-transform: uppercase;
  letter-spacing: 0.05em;
}
.proposal-field-body {
  white-space: pre-wrap;
  margin: 0.1em 0 0.4em;
}
.proposal-actions {
  display: flex; gap: 0.5em; margin-top: 0.4em;
}
.proposal-actions button {
  font: inherit; font-size: 0.9em;
  padding: 0.35em 0.85em;
  border-radius: 3px; cursor: pointer;
}
.proposal-actions .apply-btn {
  background: #1a1a1a; color: #f4e4c1; border: 0;
}
.proposal-actions .apply-btn:hover { background: #3a2f25; }
.proposal-actions .skip-btn {
  background: transparent; border: 1px solid #826e50;
}
.proposal-actions .skip-btn:hover { background: #d6caa8; }
.proposal.applied {
  background: #e8f0d8; border-color: #527a3e;
}

.usage-line {
  font-size: 0.8em; color: #826e50;
  margin-top: 0.4em;
}
.turn.pending .turn-text {
  border-left-color: #826e50;
  font-style: italic;
  color: #5a4830;
}
.turn.pending .dots {
  animation: pulse 1.2s ease-in-out infinite;
  margin-right: 0.4em;
  color: #826e50;
}
.turn.pending .pending-elapsed {
  margin-left: 0.4em;
  font-variant-numeric: tabular-nums;
  color: #826e50;
}
@keyframes pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 1; }
}

.compose {
  display: flex; gap: 0.6em; align-items: stretch;
  margin-top: 0.8em;
}
.compose textarea {
  flex: 1;
  font: inherit; padding: 0.5em 0.7em;
  border: 1px solid #c8a96e; border-radius: 4px;
  background: #fffefb;
  min-height: 4em; resize: vertical;
}
.compose button {
  background: #1a1a1a; color: #f4e4c1;
  border: 0; padding: 0.55em 1.4em;
  font: inherit; font-weight: bold;
  border-radius: 3px; cursor: pointer;
}
.compose button:disabled {
  background: #d6caa8; color: #826e50; cursor: not-allowed;
}

.setup-card {
  background: #fffaf0;
  border: 1px solid #c8a96e;
  border-radius: 6px;
  padding: 1.4em 1.6em;
  margin: 2em 0;
  max-width: 720px;
}
.readiness-banner {
  background: #fbf2dd;
  border: 1px solid #c8a96e;
  border-left: 4px solid #b8821a;
  border-radius: 6px;
  padding: 0.9em 1.2em;
  margin: 1em 0;
}
.readiness-banner h3 {
  margin: 0 0 0.3em 0;
  font-size: 1.05em;
  color: #5a4830;
}
.readiness-banner ol {
  margin: 0.4em 0 0 0;
  padding-left: 1.3em;
}
.readiness-banner li {
  margin: 0.25em 0;
}
.readiness-banner code {
  background: #f0e3bf;
  padding: 0.05em 0.4em;
  border-radius: 2px;
  font-size: 0.92em;
}
.setup-card pre {
  background: #f0e3bf; padding: 0.6em 0.9em;
  border-radius: 3px; overflow-x: auto;
  font: 13px/1.4 'Menlo', monospace;
}
.error-banner {
  background: #fbe6e2; border: 1px solid #a8201a;
  color: #5a1818; padding: 0.6em 0.9em;
  border-radius: 4px; margin: 0.8em 0;
}
""".strip()


def _render_assistant_html(d, unavailable_msg: str | None,
                            *, dungeon_path: Path,
                            dungeons: list) -> str:
    """Full HTML for the /assistant chat page. When unavailable_msg is
    set we render a setup-instructions card instead of the chat form.

    `dungeons` is a list of DungeonInfo for the picker; selecting a
    different one navigates to /assistant?dungeon=<folder>."""
    if unavailable_msg is not None:
        return _render_assistant_setup_html(d, unavailable_msg, dungeon_path)

    # Per-level readiness — a level is "ready" when its map PNG exists
    # and at least one room has image_region geometry. The form
    # disables Start for non-ready levels and the JS shows a checklist
    # explaining what's missing. We pass these flags via data-*
    # attributes so the JS doesn't have to re-fetch.
    dungeon_dir = dungeon_path.parent
    level_readiness: dict[int, dict] = {}
    for lv in d.levels:
        png_path = dungeon_dir / lv.map_image
        n_rooms = len(lv.rooms)
        n_annotated = sum(
            1 for r in lv.rooms if r.image_region is not None
        )
        level_readiness[lv.level_number] = {
            "n_rooms": n_rooms,
            "n_annotated": n_annotated,
            "image_present": png_path.exists(),
            "image_filename": lv.map_image,
        }

    level_options_parts = []
    for lv in d.levels:
        info = level_readiness[lv.level_number]
        ready = info["image_present"] and info["n_annotated"] > 0
        suffix = (
            f"({info['n_annotated']} rooms)"
            if ready else "(not ready)"
        )
        level_options_parts.append(
            f'<option value="{lv.level_number}"'
            f' data-n-rooms="{info["n_rooms"]}"'
            f' data-n-annotated="{info["n_annotated"]}"'
            f' data-image-present="{"1" if info["image_present"] else "0"}"'
            f' data-image-filename="{_esc(info["image_filename"])}"'
            f' data-ready="{"1" if ready else "0"}">'
            f'Level {lv.level_number} — {_esc(lv.display_name)} {suffix}'
            f'</option>'
        )
    level_options = "\n".join(level_options_parts)
    current_folder = dungeon_path.parent.resolve()
    dungeon_options_parts = []
    for info in dungeons:
        is_current = info.folder.resolve() == current_folder
        sel = " selected" if is_current else ""
        n_levels = info.n_levels
        dungeon_options_parts.append(
            f'<option value="{_esc(info.folder.name)}"{sel}>'
            f'{_esc(info.name)} ({n_levels} level'
            f'{"s" if n_levels != 1 else ""})</option>'
        )
    dungeon_options = "\n".join(dungeon_options_parts)
    # The folder name is what we round-trip in URL ?dungeon= params and
    # POST `dungeon_folder` fields.
    current_folder_name = current_folder.name
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Assistant — {_esc(d.name)}</title>
<style>{ASSISTANT_PAGE_CSS}</style>
</head>
<body>
<h1>Dungeon Assistant — <span id="dungeon-name">{_esc(d.name)}</span></h1>
<p class="subtle">
  Chat-style room population. Choose a dungeon + theme + level + party
  level, then iterate. Per-room Apply writes to
  <code id="dungeon-path">{_esc(str(dungeon_path))}</code> with a backup
  taken first. <a href="/">Back to room editor</a>
</p>

<div class="assistant-bar">
  <label class="theme-field">Theme &amp; concept
    <textarea id="theme" rows="4"
      placeholder="A short tag-line works ('catacombs of an exiled priest-king') but the assistant gets much better when you describe the concept fully. For example: 'A long-lost assassins cult. Three levels: living quarters, torture and dungeon areas, then the temple and holy relics. Recently a desert genie has burst in and is conducting a ritual on the third level — cultists are scrambling, and the relic chamber is partially collapsed.' Mention per-level themes, current events, or key NPCs and the assistant will weave them through proposals."></textarea>
  </label>
</div>

<div class="assistant-bar">
  <label>Dungeon
    <select id="dungeon-picker">{dungeon_options}</select>
  </label>
  <label>Level
    <select id="level">{level_options}</select>
  </label>
  <label>Party level
    <input type="number" id="party-level" min="1" max="20" value="{d.party_level}">
  </label>
  <label>Model
    <select id="model">
      <option value="claude-sonnet-4-6" selected>Sonnet 4.6 (default)</option>
      <option value="claude-opus-4-7">Opus 4.7 (high quality)</option>
      <option value="claude-haiku-4-5">Haiku 4.5 (cheap, may miss voice)</option>
    </select>
  </label>
  <button id="start-btn">Start session</button>
  <button id="reset-btn" class="secondary">Reset</button>
</div>

<input type="hidden" id="current-dungeon-folder" value="{_esc(current_folder_name)}">

<div id="readiness-banner" class="readiness-banner" hidden>
  <h3>This level isn't ready yet</h3>
  <p class="subtle">The assistant fills in <em>content</em> for rooms
  you've already drawn — it can't draw rooms or supply maps. Finish
  these steps first, then refresh this page:</p>
  <ol id="readiness-steps"></ol>
</div>

<div id="chat-log" class="chat-log"></div>

<div class="compose">
  <textarea id="user-input"
    placeholder="Refinement message (e.g. 'make R05 darker, less generic') — Cmd+Enter to send"
    disabled></textarea>
  <button id="send-btn" disabled>Send</button>
</div>

<script>
const $ = (id) => document.getElementById(id);
const log = $('chat-log');
let inFlight = false;
// Broadcast channel notifies the / editor tab when a room is applied
// here so it can swap that room's card in place without reloading
// (BroadcastChannel is widely supported; we feature-detect anyway).
const assistantBroadcast =
  typeof BroadcastChannel === 'function'
    ? new BroadcastChannel('osr-dungeon-editor')
    : null;

function setBusy(b) {{
  inFlight = b;
  $('start-btn').disabled = b;
  $('send-btn').disabled = b || $('user-input').disabled;
}}

// Live "thinking…" placeholder shown while a turn is in flight. The
// CLI subprocess can run 30-180 s on a fresh cache, so we surface a
// running counter rather than leave the page silent. The placeholder
// is removed by the caller once the response arrives (success or fail).
let pendingPlaceholder = null;
let pendingTimer = null;
function showPending(label) {{
  hidePending();
  const wrap = document.createElement('div');
  wrap.className = 'turn assistant pending';
  wrap.innerHTML = `
    <div class="turn-header">Assistant</div>
    <div class="turn-text"><span class="dots">●</span>
      <span class="pending-label"></span>
      <span class="pending-elapsed">0s</span></div>`;
  wrap.querySelector('.pending-label').textContent = label;
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
  pendingPlaceholder = wrap;
  const startTime = Date.now();
  pendingTimer = setInterval(() => {{
    const sec = Math.round((Date.now() - startTime) / 1000);
    if (pendingPlaceholder) {{
      pendingPlaceholder.querySelector('.pending-elapsed').textContent =
        sec + 's';
    }}
  }}, 1000);
}}
function hidePending() {{
  if (pendingTimer) {{ clearInterval(pendingTimer); pendingTimer = null; }}
  if (pendingPlaceholder) {{ pendingPlaceholder.remove(); pendingPlaceholder = null; }}
}}

function append(html) {{
  const div = document.createElement('div');
  div.innerHTML = html;
  log.appendChild(div.firstElementChild);
  log.scrollTop = log.scrollHeight;
}}

function renderUserTurn(text) {{
  const wrap = document.createElement('div');
  wrap.className = 'turn user';
  wrap.innerHTML = `
    <div class="turn-header">You</div>
    <div class="turn-text"></div>`;
  wrap.querySelector('.turn-text').textContent = text;
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
}}

function renderAssistantTurn(turn) {{
  const wrap = document.createElement('div');
  wrap.className = 'turn assistant';
  const header = document.createElement('div');
  header.className = 'turn-header';
  header.textContent = 'Assistant';
  wrap.appendChild(header);
  if (turn.text) {{
    const tx = document.createElement('div');
    tx.className = 'turn-text';
    tx.textContent = turn.text;
    wrap.appendChild(tx);
  }}
  if (turn.summary) {{
    const sm = document.createElement('div');
    sm.style.fontStyle = 'italic';
    sm.style.color = '#5a4830';
    sm.style.margin = '0.2em 0 0.6em';
    sm.textContent = turn.summary;
    wrap.appendChild(sm);
  }}
  for (const p of (turn.proposals || [])) {{
    wrap.appendChild(buildProposalCard(p));
  }}
  if ((turn.rejected || []).length > 0) {{
    const rej = document.createElement('div');
    rej.className = 'error-banner';
    rej.textContent = (
      'Rejected ' + turn.rejected.length + ' proposal(s) due to ' +
      'validation errors. First: ' + turn.rejected[0].error);
    wrap.appendChild(rej);
  }}
  if (turn.usage || turn.cost_usd != null) {{
    const u = document.createElement('div');
    u.className = 'usage-line';
    const parts = [];
    if (turn.usage) {{
      const cr = turn.usage.cache_read_input_tokens || 0;
      const cw = turn.usage.cache_creation_input_tokens || 0;
      const it = turn.usage.input_tokens || 0;
      const ot = turn.usage.output_tokens || 0;
      parts.push('tokens — input ' + it + ' (+ ' + cr + ' cached), output ' + ot);
      if (cw > 0) parts.push('cache write ' + cw);
    }}
    if (turn.cost_usd != null) {{
      parts.push('cost $' + turn.cost_usd.toFixed(4));
    }}
    u.textContent = parts.join(' · ');
    wrap.appendChild(u);
  }}
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
}}

function buildProposalCard(p) {{
  const card = document.createElement('div');
  card.className = 'proposal';
  card.dataset.roomId = p.id;
  // Each proposal is a fresh decision. Even if an earlier turn's
  // proposal for the same room was applied, this NEW version needs
  // its own Apply click to overwrite the file with the new content.
  // Applied state is per-card from this point on (the button only
  // flips after the user clicks Apply on this particular card).

  const head = document.createElement('div');
  head.className = 'proposal-head';
  head.innerHTML = `
    <span><span class="proposal-id">${{p.id}}</span>
          &nbsp;<span class="proposal-name"></span></span>
    <span class="proposal-tags"></span>`;
  head.querySelector('.proposal-name').textContent = p.name || '';
  head.querySelector('.proposal-tags').textContent =
    (p.tags || []).join(', ');
  card.appendChild(head);

  const fields = [
    ['Box text', p.box_text],
    ['Encounter', p.encounter_text],
    ['Treasure', p.treasure_text],
    ['Special', p.special_text],
    ['DM notes', p.notes],
  ];
  for (const [label, body] of fields) {{
    if (!body || !body.trim()) continue;
    const f = document.createElement('div');
    f.className = 'proposal-field';
    const lbl = document.createElement('div');
    lbl.className = 'proposal-field-label';
    lbl.textContent = label;
    const bd = document.createElement('div');
    bd.className = 'proposal-field-body';
    bd.textContent = body;
    f.appendChild(lbl);
    f.appendChild(bd);
    card.appendChild(f);
  }}
  if (p.reaction_required) {{
    const rr = document.createElement('div');
    rr.style.fontWeight = 'bold';
    rr.style.color = '#a8201a';
    rr.style.fontSize = '0.85em';
    rr.textContent = 'Reaction required on first entry';
    card.appendChild(rr);
  }}

  const actions = document.createElement('div');
  actions.className = 'proposal-actions';
  const apply = document.createElement('button');
  apply.className = 'apply-btn';
  apply.textContent = 'Apply';
  apply.disabled = false;
  apply.addEventListener('click', () => applyProposal(p.id, apply));
  const skip = document.createElement('button');
  skip.className = 'skip-btn';
  skip.textContent = 'Skip';
  skip.addEventListener('click', () => card.remove());
  actions.appendChild(apply);
  actions.appendChild(skip);
  card.appendChild(actions);
  return card;
}}

async function applyProposal(roomId, btn) {{
  btn.disabled = true; btn.textContent = 'Applying…';
  try {{
    const r = await fetch('/assistant/apply', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        room_id: roomId,
        dungeon_folder: currentDungeonFolder(),
      }}),
    }});
    if (!r.ok) {{
      const err = await r.json().catch(() => ({{error: 'HTTP ' + r.status}}));
      throw new Error(err.error || ('HTTP ' + r.status));
    }}
    // Flip JUST this card to applied — earlier cards for the same
    // room (from prior turns) keep their own state. The user can see
    // history this way without losing track of which version was
    // applied last.
    btn.textContent = 'Applied ✓';
    const card = btn.closest('.proposal');
    if (card) card.classList.add('applied');
    // Broadcast to other tabs (specifically the / editor) so they can
    // refresh that room's card without a full page reload.
    if (assistantBroadcast) {{
      assistantBroadcast.postMessage({{
        type: 'room-applied',
        dungeon_folder: currentDungeonFolder(),
        room_id: roomId,
      }});
    }}
  }} catch (e) {{
    btn.disabled = false; btn.textContent = 'Apply';
    alert('Apply failed: ' + e.message);
  }}
}}

function currentDungeonFolder() {{
  return $('current-dungeon-folder').value;
}}

async function postJSON(path, body) {{
  // Always tag requests with the dungeon folder so the server hits
  // the right session even when several pages are open across dungeons.
  const augmented = {{...(body || {{}}), dungeon_folder: currentDungeonFolder()}};
  const r = await fetch(path, {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(augmented),
  }});
  const txt = await r.text();
  let data = null;
  try {{ data = txt ? JSON.parse(txt) : null; }} catch {{ data = null; }}
  if (!r.ok) {{
    throw new Error((data && (data.message || data.error)) || ('HTTP ' + r.status));
  }}
  return data;
}}

// Picker — switching the selected dungeon reloads the page so the
// level dropdown reflects the new dungeon's levels.
$('dungeon-picker').addEventListener('change', (e) => {{
  const folder = e.target.value;
  if (folder && folder !== currentDungeonFolder()) {{
    location.href = '/assistant?dungeon=' + encodeURIComponent(folder);
  }}
}});

// Level readiness — update banner + Start button whenever the level
// dropdown changes (and once on page load).
function updateReadinessBanner() {{
  const sel = $('level');
  const opt = sel.options[sel.selectedIndex];
  if (!opt) return;
  const ready = opt.dataset.ready === '1';
  const nAnnotated = parseInt(opt.dataset.nAnnotated, 10) || 0;
  const imagePresent = opt.dataset.imagePresent === '1';
  const imageFilename = opt.dataset.imageFilename || 'level.png';

  const banner = $('readiness-banner');
  const steps = $('readiness-steps');
  const startBtn = $('start-btn');

  if (ready) {{
    banner.hidden = true;
    startBtn.disabled = false;
    startBtn.title = '';
    return;
  }}

  // Build a tailored checklist for what's missing.
  steps.innerHTML = '';
  const items = [];
  if (!imagePresent) {{
    const li = document.createElement('li');
    li.innerHTML = (
      'Drop your level\\'s map image into the dungeon folder as ' +
      '<code></code>. The pygame app picks it up on the next load.'
    );
    li.querySelector('code').textContent = imageFilename;
    items.push(li);
  }}
  if (nAnnotated === 0) {{
    const li = document.createElement('li');
    li.innerHTML = (
      'In the pygame window, press <code>A</code> to enter annotation ' +
      'mode, then drag rectangles over each room you want to populate. ' +
      'Press <code>A</code> again to exit. Each rectangle becomes a room ' +
      'the assistant can fill in.'
    );
    items.push(li);
  }}
  const lastLi = document.createElement('li');
  lastLi.textContent = 'Refresh this page (Cmd+R) to pick up the changes.';
  items.push(lastLi);
  for (const li of items) steps.appendChild(li);

  banner.hidden = false;
  startBtn.disabled = true;
  startBtn.title = 'This level needs at least one annotated room first.';
}}

$('level').addEventListener('change', updateReadinessBanner);
updateReadinessBanner();  // run once on page load

$('start-btn').addEventListener('click', async () => {{
  const theme = $('theme').value.trim();
  if (!theme) {{ alert('Theme/concept is required.'); return; }}
  const payload = {{
    theme,
    level_number: parseInt($('level').value, 10),
    party_level: parseInt($('party-level').value, 10),
    model: $('model').value,
  }};
  log.innerHTML = '';
  setBusy(true);
  // Render the kickoff as: a "theme & concept" block (preserves
  // newlines from the textarea) followed by a metadata line. For long
  // pasted concepts this reads much better than crammed into one line.
  const meta = `Level ${{payload.level_number}} · Party ${{payload.party_level}} · ${{payload.model}}`;
  renderUserTurn(theme + '\\n\\n— ' + meta);
  showPending('Thinking… first turn pays the prefix-cache write and runs ~30–180 s.');
  try {{
    const turn = await postJSON('/assistant/start', payload);
    hidePending();
    renderAssistantTurn(turn);
    $('user-input').disabled = false;
  }} catch (e) {{
    hidePending();
    append('<div class="error-banner"></div>');
    log.lastElementChild.textContent = 'Start failed: ' + e.message;
  }} finally {{
    setBusy(false);
  }}
}});

$('send-btn').addEventListener('click', sendMessage);
$('user-input').addEventListener('keydown', (e) => {{
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') sendMessage();
}});

async function sendMessage() {{
  const text = $('user-input').value.trim();
  if (!text || inFlight) return;
  setBusy(true);
  renderUserTurn(text);
  $('user-input').value = '';
  showPending('Thinking… refinement turns ride the prefix cache and usually finish in 5–30 s.');
  try {{
    const turn = await postJSON('/assistant/message', {{text}});
    hidePending();
    renderAssistantTurn(turn);
  }} catch (e) {{
    hidePending();
    append('<div class="error-banner"></div>');
    log.lastElementChild.textContent = 'Send failed: ' + e.message;
  }} finally {{
    setBusy(false);
  }}
}}

$('reset-btn').addEventListener('click', async () => {{
  if (!confirm('Reset the assistant session for this dungeon?')) return;
  await fetch('/assistant/reset', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{dungeon_folder: currentDungeonFolder()}}),
  }});
  log.innerHTML = '';
  $('user-input').disabled = true;
}});
</script>
<script>{FRAME_BRIDGE_JS}</script>
</body>
</html>"""


def _render_assistant_setup_html(d, message: str,
                                  dungeon_path: Path | None = None) -> str:
    # `dungeon_path` is informational only — the setup card doesn't need
    # the picker since the CLI is what's missing, not the dungeon.
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Assistant — {_esc(d.name)}</title>
<style>{ASSISTANT_PAGE_CSS}</style>
</head>
<body>
<h1>Dungeon Assistant</h1>
<p class="subtle">
  <a href="/">Back to room editor</a>
</p>

<div class="setup-card">
  <h2 style="margin-top:0;">Setup needed</h2>
  <p>{_esc(message)}</p>
  <p>The assistant runs through your existing Claude Code subscription
     — no separate API key, no extra billing. To enable it:</p>
  <ol>
    <li>Install Claude Code from
      <a href="https://claude.com/download">claude.com/download</a>
      if you don't already have it.</li>
    <li>From a terminal, sign in (only needed once per machine):
      <pre>claude login</pre>
    </li>
    <li>Confirm the CLI is on your PATH where you ran <code>main.py</code>:
      <pre>which claude
claude --version</pre>
    </li>
    <li>Restart <code>python main.py …</code> — the assistant page will
      load the chat form.</li>
  </ol>
  <p class="subtle">
    The assistant is the only feature that calls out to a remote
    service, and only when you click Start session. Each turn is
    billed against your Claude Code subscription, not a separate API
    console account. The rest of the app stays fully offline.
  </p>
</div>
<script>{FRAME_BRIDGE_JS}</script>
</body>
</html>"""


# --- Server handler ---------------------------------------------------------


class EditorHandler(BaseHTTPRequestHandler):
    """One handler instance per request. Reads the JSON fresh on every
    request — never holds a long-lived in-memory dungeon."""

    # Set by start_editor_server before binding. `dungeon_path` is the
    # default dungeon (the one the pygame app was launched against);
    # `dungeons_dir` is the root for the assistant's per-page picker so
    # the DM can target a different dungeon without restarting the app.
    dungeon_path: Path = Path()
    dungeons_dir: Path | None = None

    # Silence the default per-request stderr log (pygame stdout is noisy).
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    # -- GET ---------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == "/healthz":
            self._respond(HTTPStatus.OK, b"ok\n", "text/plain; charset=utf-8")
            return
        # Player view: a separate window the DM screen-shares / projects.
        # /player is the HTML wrapper that auto-refreshes /player.png.
        if self.path == "/player":
            self._respond(HTTPStatus.OK,
                          _PLAYER_HTML.encode("utf-8"),
                          "text/html; charset=utf-8")
            return
        if self.path == "/player.png" or self.path.startswith("/player.png?"):
            self._serve_player_png()
            return
        # GET / returns the SPA shell; each tab is an iframe pointing at
        # the underlying route. The room editor itself lives at /editor.
        if self.path == "/":
            self._render_shell()
            return
        if self.path == "/editor":
            self._render()
            return
        if self.path.startswith("/editor?saved="):
            saved_id = self.path.split("=", 1)[1]
            # Level saves use the form `?saved=L<number>`; rooms use the
            # plain id. We dispatch on the leading 'L' so we can show the
            # right banner.
            if saved_id.startswith("L"):
                try:
                    self._render(saved_level_number=int(saved_id[1:]))
                    return
                except ValueError:
                    pass
            self._render(saved_room_id=saved_id)
            return
        # GET /room?level_number=N&room_id=R01 → fresh card fragment.
        # Used by the / editor tab when it gets a BroadcastChannel
        # message from /assistant after an Apply, so the on-screen
        # card can refresh without a full page reload.
        if self.path.startswith("/room?"):
            from urllib.parse import parse_qs as _pq
            try:
                _, query = self.path.split("?", 1)
                qs = _pq(query)
                level_number = int(qs.get("level_number", ["0"])[0])
                room_id = qs.get("room_id", [""])[0]
                if not room_id:
                    raise ValueError("missing room_id")
            except (ValueError, KeyError):
                self.send_error(HTTPStatus.BAD_REQUEST, "Bad request")
                return
            self._respond_room_fragment(level_number, room_id)
            return
        if self.path == "/assistant" or self.path.startswith("/assistant?"):
            # Pull out the optional ?dungeon=<folder> override.
            folder = None
            if "?" in self.path:
                from urllib.parse import parse_qs
                _, query = self.path.split("?", 1)
                qs = parse_qs(query)
                folder_list = qs.get("dungeon", [])
                if folder_list:
                    folder = folder_list[0]
            self._render_assistant_page(dungeon_folder=folder)
            return
        if self.path == "/characters" or self.path.startswith("/characters?"):
            self._render_characters_page()
            return
        if self.path.startswith("/simulate?"):
            from urllib.parse import parse_qs as _pq
            try:
                _, query = self.path.split("?", 1)
                qs = _pq(query)
                level_number = int(qs.get("level_number", ["0"])[0])
                room_id = qs.get("room_id", [""])[0]
                trials = int(qs.get("trials", ["100"])[0])
                # party_level is optional. Empty/missing → no scaling
                # (use each character's stored level).
                party_level_raw = qs.get("party_level", [""])[0].strip()
                party_level: int | None
                if party_level_raw:
                    party_level = max(1, min(20, int(party_level_raw)))
                else:
                    party_level = None
                # seed is optional. Empty/missing → randomise so each
                # Re-run gives fresh dice. Provide an explicit number
                # to reproduce a specific run (handy for sharing a
                # trace).
                seed_raw = qs.get("seed", [""])[0].strip()
                if seed_raw:
                    seed = int(seed_raw)
                else:
                    import random as _r
                    seed = _r.randrange(2**31)
                if not room_id:
                    raise ValueError("missing room_id")
            except (ValueError, KeyError):
                self.send_error(HTTPStatus.BAD_REQUEST, "Bad request")
                return
            self._render_simulate_page(level_number, room_id, trials,
                                       party_level, seed)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    # -- POST --------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        # Character-upload endpoint takes the raw PDF body. Read it
        # before parse_qs eats it as form-encoded.
        if self.path.startswith("/characters/upload"):
            raw = self.rfile.read(length) if length > 0 else b""
            self._handle_character_upload(raw)
            return
        raw = self.rfile.read(length) if length > 0 else b""
        # Assistant endpoints take JSON bodies (cleaner for nested
        # message lists); the existing form endpoints still parse
        # urlencoded bodies via parse_qs below.
        ctype = (self.headers.get("Content-Type") or "").lower()
        if self.path.startswith("/assistant/") and "json" in ctype:
            self._handle_assistant_post(raw)
            return
        form = parse_qs(raw.decode("utf-8"), keep_blank_values=True)
        if self.path == "/characters/delete":
            filename = form.get("filename", [""])[0]
            self._handle_character_delete(filename)
            return

        # When the JS fetch path POSTs with this header, return the freshly
        # rendered HTML *fragment* for the saved card (200 OK) so the browser
        # can splice it into the existing page without a full reload —
        # otherwise unsaved edits in sibling forms would be wiped. Plain
        # HTML form submits (no JS) keep the legacy 303 redirect path so
        # they still work, and the existing tests still pass.
        is_fragment = self.headers.get("X-Editor-Fragment") == "1"

        if self.path == "/room":
            try:
                level_number = int(form.get("level_number", ["0"])[0])
                room_id = form.get("room_id", [""])[0]
                if not room_id:
                    raise ValueError("missing room_id")
                self._mutate_room_and_save(level_number, room_id, form)
            except (ValueError, KeyError, dungeon_mod.DungeonValidationError) as e:
                self.send_error(HTTPStatus.BAD_REQUEST, f"Bad request: {e}")
                return
            if is_fragment:
                self._respond_room_fragment(level_number, room_id)
            else:
                self._respond(HTTPStatus.SEE_OTHER, b"",
                              content_type="text/plain",
                              extra_headers={"Location": f"/editor?saved={room_id}"})
            return

        if self.path == "/level":
            try:
                level_number = int(form.get("level_number", ["0"])[0])
                self._mutate_level_and_save(level_number, form)
            except (ValueError, KeyError, dungeon_mod.DungeonValidationError) as e:
                self.send_error(HTTPStatus.BAD_REQUEST, f"Bad request: {e}")
                return
            if is_fragment:
                self._respond_level_fragment(level_number)
            else:
                self._respond(HTTPStatus.SEE_OTHER, b"",
                              content_type="text/plain",
                              extra_headers={"Location": f"/editor?saved=L{level_number}"})
            return

        if self.path == "/enrich":
            try:
                level_number = int(form.get("level_number", ["0"])[0])
                room_id = form.get("room_id", [""])[0]
                if not room_id:
                    raise ValueError("missing room_id")
                self._enrich_room_and_save(level_number, room_id)
            except (ValueError, KeyError, dungeon_mod.DungeonValidationError) as e:
                self.send_error(HTTPStatus.BAD_REQUEST, f"Bad request: {e}")
                return
            # Enrich is always called via JS fetch — return the fragment so
            # the page can splice in the updated card without losing
            # in-progress edits in other rooms.
            self._respond_room_fragment(level_number, room_id)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def _respond_room_fragment(self, level_number: int, room_id: str) -> None:
        """Re-read JSON, render just the one room card, return as 200 HTML."""
        d = dungeon_mod.load(self.dungeon_path)
        level = d.levels_by_number.get(level_number)
        room = level.rooms_by_id.get(room_id) if level is not None else None
        if room is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Room missing after save")
            return
        body = _render_room(level_number, room).encode("utf-8")
        self._respond(HTTPStatus.OK, body, "text/html; charset=utf-8")

    # -- Assistant ---------------------------------------------------------

    def _list_assistant_dungeons(self) -> list:
        """Return DungeonInfo entries for the assistant's picker.
        Includes the server's bound dungeon even if it lives outside
        dungeons_dir (so the page never appears empty)."""
        from session import Session  # local import — avoid circular
        infos = []
        if self.dungeons_dir is not None:
            try:
                infos = Session.list_dungeons(self.dungeons_dir)
            except Exception:
                infos = []
        # If the bound dungeon isn't in the dungeons_dir scan, prepend
        # it so the user can always operate on what they launched with.
        bound_folder = self.dungeon_path.parent.resolve()
        if not any(i.folder.resolve() == bound_folder for i in infos):
            try:
                d = dungeon_mod.load(self.dungeon_path)
                from session import DungeonInfo
                infos = [DungeonInfo(
                    folder=bound_folder, name=d.name, n_levels=len(d.levels),
                    has_session=False, current_level=d.current_level,
                    current_turn=0, last_saved_at="",
                )] + list(infos)
            except Exception:
                pass
        return infos

    def _resolve_dungeon_path(self, folder_name: str | None) -> Path:
        """Map an optional folder-name override to a concrete
        dungeon.json path. Falls back to the server's bound dungeon
        when the override is missing or empty. Validates the resolved
        path stays under dungeons_dir to avoid path traversal."""
        if not folder_name:
            return self.dungeon_path
        if self.dungeons_dir is None:
            return self.dungeon_path
        # Strip any leading slashes/dots; only the basename or a
        # single relative segment under dungeons_dir is allowed.
        clean = folder_name.strip().strip("/").strip("\\")
        if not clean or "/" in clean or "\\" in clean or ".." in clean:
            raise ValueError(f"invalid dungeon folder {folder_name!r}")
        candidate = (self.dungeons_dir / clean / "dungeon.json").resolve()
        # Confirm the resolved path is genuinely inside dungeons_dir.
        try:
            candidate.relative_to(self.dungeons_dir.resolve())
        except ValueError:
            raise ValueError(f"dungeon {folder_name!r} not under dungeons_dir")
        if not candidate.exists():
            raise ValueError(f"dungeon {folder_name!r} not found")
        return candidate

    def _handle_assistant_post(self, raw: bytes) -> None:
        """Dispatch JSON-bodied /assistant/* requests."""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError as e:
            self._respond_json(HTTPStatus.BAD_REQUEST,
                               {"error": f"invalid JSON: {e}"})
            return

        if self.path == "/assistant/start":
            self._assistant_start(payload)
        elif self.path == "/assistant/message":
            self._assistant_message(payload)
        elif self.path == "/assistant/apply":
            self._assistant_apply(payload)
        elif self.path == "/assistant/reset":
            self._assistant_reset(payload)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def _assistant_start(self, payload: dict) -> None:
        try:
            theme = str(payload.get("theme", "")).strip()
            level_number = int(payload.get("level_number", 0))
            party_level = int(payload.get("party_level", 0))
            model = str(payload.get("model")
                        or dungeon_assistant.DEFAULT_MODEL)
            folder = payload.get("dungeon_folder")
            if not theme:
                raise ValueError("theme is required")
            target_path = self._resolve_dungeon_path(folder)
        except (TypeError, ValueError) as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            return

        try:
            d = dungeon_mod.load(target_path)
            level = d.levels_by_number.get(level_number)
            if level is None:
                raise ValueError(
                    f"level {level_number} not in this dungeon"
                )
            n_annotated = sum(
                1 for r in level.rooms if r.image_region is not None
            )
            if n_annotated == 0:
                self._respond_json(HTTPStatus.BAD_REQUEST, {
                    "error": "level_not_ready",
                    "message": (
                        f"Level {level_number} has no annotated rooms yet. "
                        "Drop the level's map image into the dungeon "
                        "folder and use pygame's annotation mode (A) to "
                        "draw rooms before starting an assistant session."
                    ),
                })
                return
            session = AssistantSession(
                dungeon_path=target_path,
                dungeon=d,
                theme=theme,
                level_number=level_number,
                party_level=party_level,
                model=model,
            )
            turn = session.start()
        except AssistantUnavailable as e:
            self._respond_json(HTTPStatus.SERVICE_UNAVAILABLE,
                               {"error": "assistant_unavailable",
                                "message": str(e)})
            return
        except (ValueError, dungeon_mod.DungeonValidationError) as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            return
        except Exception as e:  # network / API errors
            self._respond_json(HTTPStatus.BAD_GATEWAY,
                               {"error": "api_error", "message": str(e)})
            return

        _assistant_sessions[target_path.resolve()] = session
        self._respond_json(HTTPStatus.OK, _turn_to_json(turn))

    def _assistant_message(self, payload: dict) -> None:
        text = str(payload.get("text", "")).strip()
        if not text:
            self._respond_json(HTTPStatus.BAD_REQUEST,
                               {"error": "empty message"})
            return
        try:
            target_path = self._resolve_dungeon_path(
                payload.get("dungeon_folder"))
        except ValueError as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            return
        session = _get_or_none_assistant(target_path)
        if session is None:
            self._respond_json(HTTPStatus.BAD_REQUEST,
                               {"error": "no_session",
                                "message": "Start a session first."})
            return
        try:
            turn = session.send(text)
        except AssistantUnavailable as e:
            self._respond_json(HTTPStatus.SERVICE_UNAVAILABLE,
                               {"error": "assistant_unavailable",
                                "message": str(e)})
            return
        except Exception as e:
            self._respond_json(HTTPStatus.BAD_GATEWAY,
                               {"error": "api_error", "message": str(e)})
            return
        self._respond_json(HTTPStatus.OK, _turn_to_json(turn))

    def _assistant_apply(self, payload: dict) -> None:
        room_id = str(payload.get("room_id", "")).strip()
        if not room_id:
            self._respond_json(HTTPStatus.BAD_REQUEST,
                               {"error": "missing room_id"})
            return
        try:
            target_path = self._resolve_dungeon_path(
                payload.get("dungeon_folder"))
        except ValueError as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            return
        session = _get_or_none_assistant(target_path)
        if session is None or room_id not in session.latest_proposals:
            self._respond_json(HTTPStatus.BAD_REQUEST,
                               {"error": "no_proposal",
                                "message": (
                                    f"No active proposal for {room_id}.")})
            return
        proposal = session.latest_proposals[room_id]
        try:
            d = dungeon_mod.load(target_path)
            level = d.levels_by_number.get(session.level_number)
            if level is None:
                raise KeyError(f"unknown level {session.level_number}")
            room = level.rooms_by_id.get(room_id)
            if room is None:
                raise KeyError(
                    f"room {room_id} missing on level {session.level_number}"
                )
            # Apply only the editable fields. Leave id, state,
            # image_region, encounter_ref, treasure_tier, statblocks
            # untouched.
            room.name = proposal.name
            room.tags = proposal.tags
            room.reaction_required = proposal.reaction_required
            room.notes = proposal.notes
            room.box_text = proposal.box_text
            room.encounter_text = proposal.encounter_text
            room.treasure_text = proposal.treasure_text
            room.special_text = proposal.special_text

            # Backup → atomic write (same shape as /enrich).
            dungeon_mod.backup_dungeon_json(target_path, keep_last=3)
            dungeon_mod.dump(d, target_path)

            # Refresh the session's snapshot so the next turn's
            # dungeon-context reflects the populated room.
            fresh = dungeon_mod.load(target_path)
            session.refresh_dungeon_context(fresh)
        except (KeyError, dungeon_mod.DungeonValidationError) as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            return

        # Return the freshly-rendered room card so the main editor page
        # can splice it in (DM may have / open in a sibling tab).
        body = _render_room(session.level_number,
                            level.rooms_by_id[room_id]).encode("utf-8")
        self._respond(HTTPStatus.OK, body, "text/html; charset=utf-8")

    def _assistant_reset(self, payload: dict | None = None) -> None:
        try:
            target_path = self._resolve_dungeon_path(
                (payload or {}).get("dungeon_folder"))
        except ValueError:
            target_path = self.dungeon_path
        _assistant_sessions.pop(target_path.resolve(), None)
        self._respond(HTTPStatus.NO_CONTENT, b"", "text/plain")

    def _render_assistant_page(self,
                                dungeon_folder: str | None = None) -> None:
        # Resolve which dungeon this page is for. Bad ?dungeon= values
        # get a clear 404 rather than silently falling back, so the DM
        # notices typos in the URL.
        try:
            target_path = self._resolve_dungeon_path(dungeon_folder)
        except ValueError as e:
            self.send_error(HTTPStatus.NOT_FOUND, str(e))
            return
        try:
            d = dungeon_mod.load(target_path)
        except (FileNotFoundError, dungeon_mod.DungeonValidationError) as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR,
                            f"Dungeon load failed: {e}")
            return
        # The page renders even when the CLI isn't installed — in that
        # case it shows the setup-instructions card. The probe is just
        # a `which claude` check; auth failures surface naturally on
        # the first turn so we don't try to detect them up front.
        unavailable_msg: str | None = None
        try:
            dungeon_assistant.check_cli_available()
        except AssistantUnavailable as e:
            unavailable_msg = str(e)
        body = _render_assistant_html(
            d, unavailable_msg,
            dungeon_path=target_path,
            dungeons=self._list_assistant_dungeons(),
        ).encode("utf-8")
        self._respond(HTTPStatus.OK, body, "text/html; charset=utf-8")

    def _respond_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._respond(status, body, "application/json; charset=utf-8")

    def _respond_level_fragment(self, level_number: int) -> None:
        d = dungeon_mod.load(self.dungeon_path)
        level = d.levels_by_number.get(level_number)
        if level is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Level missing after save")
            return
        body = _render_level_card(level).encode("utf-8")
        self._respond(HTTPStatus.OK, body, "text/html; charset=utf-8")

    # -- Internals ---------------------------------------------------------

    def _serve_player_png(self) -> None:
        """Stream the player-view PNG bytes. Returns 404 with a tiny
        placeholder when pygame hasn't written the file yet (cold
        start), so the client-side auto-refresh keeps polling without
        a console error."""
        try:
            data = _PLAYER_PNG_PATH.read_bytes()
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND, "Player map not generated yet")
            return
        self._respond(
            HTTPStatus.OK, data, "image/png",
            # No-cache so the client's cache-busting query string isn't
            # the only thing keeping refreshes fresh; some intermediaries
            # ignore querystrings in their cache key.
            extra_headers={"Cache-Control": "no-store, max-age=0"},
        )

    def _render_shell(self) -> None:
        """GET / — the SPA shell with tab strip and per-tab iframes.
        Each iframe loads its underlying route (/editor, /assistant,
        /characters) so the existing page code keeps working unchanged."""
        try:
            d = dungeon_mod.load(self.dungeon_path)
            dungeon_name = d.name
        except (FileNotFoundError, dungeon_mod.DungeonValidationError):
            # The shell itself doesn't need a valid dungeon — the inner
            # iframes will surface their own errors. Fall back to the
            # folder name so the title bar still has something useful.
            dungeon_name = self.dungeon_path.parent.name
        body = _render_app_shell(dungeon_name).encode("utf-8")
        self._respond(HTTPStatus.OK, body, "text/html; charset=utf-8")

    def _render(self, *, saved_room_id: str | None = None,
                saved_level_number: int | None = None) -> None:
        try:
            d = dungeon_mod.load(self.dungeon_path)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND,
                            f"Dungeon JSON missing: {self.dungeon_path}")
            return
        except dungeon_mod.DungeonValidationError as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR,
                            f"Dungeon JSON invalid: {e}")
            return
        body = _render_page(
            d,
            saved_room_id=saved_room_id,
            saved_level_number=saved_level_number,
        ).encode("utf-8")
        self._respond(HTTPStatus.OK, body, "text/html; charset=utf-8")

    def _mutate_room_and_save(self, level_number: int, room_id: str,
                              form: dict[str, list[str]]) -> None:
        d = dungeon_mod.load(self.dungeon_path)
        level = d.levels_by_number.get(level_number)
        if level is None:
            raise KeyError(f"unknown level {level_number}")
        room = level.rooms_by_id.get(room_id)
        if room is None:
            raise KeyError(f"unknown room {room_id} on level {level_number}")

        room.name = form.get("name", [room.id])[0] or room.id
        room.notes = form.get("notes", [""])[0]
        room.box_text = form.get("box_text", [""])[0]
        room.encounter_text = form.get("encounter_text", [""])[0]
        room.treasure_text = form.get("treasure_text", [""])[0]
        room.special_text = form.get("special_text", [""])[0]
        room.encounter_ref = (form.get("encounter_ref", [""])[0] or None)
        tier = form.get("treasure_tier", [""])[0]
        room.treasure_tier = tier or None
        room.reaction_required = "reaction_required" in form

        # Tags: incoming list under "tags". Filter to known + non-empty.
        raw_tags = form.get("tags", []) or []
        valid = [t for t in raw_tags if t in config.ROOM_TAGS]
        room.tags = tuple(valid) if valid else ("empty",)

        dungeon_mod.dump(d, self.dungeon_path)

    def _enrich_room_and_save(self, level_number: int, room_id: str) -> None:
        """Parse formal `<count> <Name>(s) (MM p.<page>)` declarations
        from the room's encounter text, pull the matching SRD stat
        blocks, and write the concatenation to room.statblocks.
        Replaces any prior enrichment wholesale — the DM edits
        encounter_text and re-clicks Enrich to refresh.

        Only formal declarations count. Descriptive prose that mentions
        creature-adjacent words ("shadow", "light", "death") is
        ignored, so a Giant Scorpion lurking "in the pit's northern
        shadow" doesn't summon a phantom Shadow into the room.

        A timestamped backup of the dungeon.json is written first and
        the last 3 backups are retained."""
        d = dungeon_mod.load(self.dungeon_path)
        level = d.levels_by_number.get(level_number)
        if level is None:
            raise KeyError(f"unknown level {level_number}")
        room = level.rooms_by_id.get(room_id)
        if room is None:
            raise KeyError(f"unknown room {room_id} on level {level_number}")

        entries = srd_lookup.parse_encounter_declarations(room.encounter_text)
        if not entries:
            room.statblocks = ""
        else:
            room.statblocks = "\n\n".join(e.statblock.body for e in entries)
        # Back up before any mutating write so a bad Enrich is recoverable.
        dungeon_mod.backup_dungeon_json(self.dungeon_path, keep_last=3)
        dungeon_mod.dump(d, self.dungeon_path)

    # -- Encounter simulator + character ingester -----------------------

    def _render_simulate_page(self, level_number: int, room_id: str,
                              trials: int,
                              party_level: int | None = None,
                              seed: int = 0) -> None:
        """Run a Monte Carlo simulation for the given room's encounter
        against the uploaded character sheets. Returns a complete HTML
        page (opened in a new browser tab by simulateRoom() in the JS).

        `party_level` (when set) scales every PC to that level via
        level_scaler.scale_party — useful for stress-testing deeper
        dungeon levels without having to re-upload character PDFs.

        `seed` is the Monte Carlo `base_seed`. The do_GET handler
        randomises it when the query string omits the `seed=`
        parameter, so Re-run produces fresh dice by default."""
        try:
            d = dungeon_mod.load(self.dungeon_path)
        except Exception as e:
            self._respond(HTTPStatus.INTERNAL_SERVER_ERROR,
                          f"failed to load dungeon: {e}".encode("utf-8"),
                          "text/plain; charset=utf-8")
            return
        level = d.levels_by_number.get(level_number)
        room = level.rooms_by_id.get(room_id) if level else None
        if room is None:
            self._respond(HTTPStatus.NOT_FOUND,
                          f"room {room_id} on level {level_number} not found"
                          .encode("utf-8"),
                          "text/plain; charset=utf-8")
            return

        characters = character_ingester.load_characters(self.dungeon_path)
        if party_level is not None and characters:
            import level_scaler
            characters = level_scaler.scale_party(characters, party_level)
        monsters = _build_monsters_from_room(room)

        body = _render_simulate_page_body(
            dungeon_name=d.name,
            level_number=level_number,
            room=room,
            characters=characters,
            monsters=monsters,
            trials=max(1, min(trials, 500)),
            party_level_override=party_level,
            seed=seed,
        ).encode("utf-8")
        self._respond(HTTPStatus.OK, body, "text/html; charset=utf-8")

    def _render_characters_page(self) -> None:
        """List uploaded character JSONs + render an upload form."""
        characters = character_ingester.load_characters(self.dungeon_path)
        try:
            character_ingester.check_cli_available()
            cli_ok = True
            cli_error = ""
        except character_ingester.IngesterUnavailable as e:
            cli_ok = False
            cli_error = str(e)
        body = _render_characters_page_body(
            dungeon_path=self.dungeon_path,
            characters=characters,
            cli_ok=cli_ok,
            cli_error=cli_error,
        ).encode("utf-8")
        self._respond(HTTPStatus.OK, body, "text/html; charset=utf-8")

    def _handle_character_upload(self, raw: bytes) -> None:
        """Receive a raw PDF body, run extraction, save the JSON.
        Responds with JSON {ok, name, path} or {ok: false, error: ...}."""
        from urllib.parse import parse_qs as _pq
        # Optional filename hint comes via query string.
        explicit_name = ""
        if "?" in self.path:
            _, query = self.path.split("?", 1)
            qs = _pq(query)
            explicit_name = qs.get("filename", [""])[0].strip()

        if not raw:
            self._respond_json(HTTPStatus.BAD_REQUEST,
                               {"ok": False, "error": "empty upload"})
            return

        # Persist the bytes to a temp file so pypdf can open it. Atomic
        # write isn't necessary — the temp file is throwaway.
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False
            ) as tmp:
                tmp.write(raw)
                tmp_path = Path(tmp.name)
        except OSError as e:
            self._respond_json(HTTPStatus.INTERNAL_SERVER_ERROR,
                               {"ok": False, "error": f"temp write: {e}"})
            return

        try:
            try:
                pdf_text = character_ingester.pdf_to_text(tmp_path)
            except character_ingester.IngesterUnavailable as e:
                self._respond_json(HTTPStatus.SERVICE_UNAVAILABLE,
                                   {"ok": False, "error": str(e)})
                return
            if not pdf_text.strip():
                self._respond_json(HTTPStatus.BAD_REQUEST,
                                   {"ok": False,
                                    "error": "no text could be extracted "
                                             "from the PDF"})
                return
            try:
                character = character_ingester.extract_character(pdf_text)
            except character_ingester.IngesterUnavailable as e:
                self._respond_json(HTTPStatus.SERVICE_UNAVAILABLE,
                                   {"ok": False, "error": str(e)})
                return
            except RuntimeError as e:
                self._respond_json(HTTPStatus.BAD_GATEWAY,
                                   {"ok": False, "error": str(e)})
                return
            if explicit_name and not character.get("name"):
                character["name"] = explicit_name
            saved = character_ingester.save_character(
                character, self.dungeon_path,
                filename=explicit_name or None,
            )
            self._respond_json(HTTPStatus.OK, {
                "ok": True,
                "name": character.get("name", ""),
                "filename": saved.name,
            })
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    def _handle_character_delete(self, filename: str) -> None:
        """Remove a character JSON from the dungeon's characters/ folder.
        Returns 303 redirect back to /characters."""
        if not filename or "/" in filename or "\\" in filename or ".." in filename:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid filename")
            return
        target = (
            character_ingester.characters_dir(self.dungeon_path) / filename
        )
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR,
                            f"delete failed: {e}")
            return
        self._respond(HTTPStatus.SEE_OTHER, b"",
                      content_type="text/plain",
                      extra_headers={"Location": "/characters"})

    def _mutate_level_and_save(self, level_number: int,
                               form: dict[str, list[str]]) -> None:
        d = dungeon_mod.load(self.dungeon_path)
        level = d.levels_by_number.get(level_number)
        if level is None:
            raise KeyError(f"unknown level {level_number}")

        new_name = form.get("display_name", [""])[0].strip()
        if new_name:
            level.display_name = new_name
        level.challenge_rating = form.get("challenge_rating", [""])[0]

        method = form.get("wm_check_method", [level.wm_check_method])[0]
        if method in config.WM_METHODS:
            level.wm_check_method = method

        try:
            threshold = int(form.get("wm_check_threshold",
                                      [str(level.wm_check_threshold)])[0])
            level.wm_check_threshold = max(1, threshold)
        except ValueError:
            pass

        try:
            every_n = int(form.get("wm_check_every_n_turns",
                                    [str(level.wm_check_every_n_turns)])[0])
            level.wm_check_every_n_turns = max(1, every_n)
        except ValueError:
            pass

        # Wandering monster table — paired wm_roll[] / wm_encounter[] inputs.
        rolls = form.get("wm_roll", []) or []
        encounters = form.get("wm_encounter", []) or []
        rebuilt: list[WMTableEntry] = []
        seen_rolls: set[int] = set()
        for roll_str, enc in zip(rolls, encounters):
            try:
                roll = int(roll_str)
            except ValueError:
                continue
            enc = enc.strip()
            if not enc or roll in seen_rolls or roll < 1:
                continue
            seen_rolls.add(roll)
            rebuilt.append(WMTableEntry(roll=roll, encounter=enc))
        # Validation: dungeon loader requires at least one row. If the
        # user emptied the table, keep the existing one.
        if rebuilt:
            level.wandering_monster_table = tuple(
                sorted(rebuilt, key=lambda e: e.roll)
            )

        dungeon_mod.dump(d, self.dungeon_path)

    def _respond(self, status: HTTPStatus, body: bytes,
                 content_type: str = "text/plain; charset=utf-8",
                 *, extra_headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)


# --- Public API --------------------------------------------------------------


def start_editor_server(
    dungeon_path: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    dungeons_dir: Path | str | None = None,
) -> tuple[HTTPServer, threading.Thread]:
    """Start the editor server in a daemon thread. Returns (server, thread).

    Pass `port=0` for an ephemeral port (the bound port can be read from
    `server.server_address[1]`); useful for tests.

    `dungeons_dir` enables the assistant's multi-dungeon picker — it's
    the root scanned by `Session.list_dungeons()`. When omitted the
    assistant page falls back to the single dungeon at `dungeon_path`.
    """
    EditorHandler.dungeon_path = Path(dungeon_path)
    EditorHandler.dungeons_dir = Path(dungeons_dir) if dungeons_dir else None
    server = HTTPServer((host, port), EditorHandler)
    thread = threading.Thread(
        target=server.serve_forever, name="editor-server", daemon=True,
    )
    thread.start()
    return server, thread


# --- CLI smoke (for ad-hoc testing without pygame) -------------------------


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("dungeon_json", type=Path)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = p.parse_args()
    server, _ = start_editor_server(args.dungeon_json, port=args.port)
    print(f"editor server: http://127.0.0.1:{server.server_address[1]}")
    try:
        server.serve_forever()  # blocks until Ctrl+C
    except KeyboardInterrupt:
        server.shutdown()
