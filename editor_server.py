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
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import config
import dungeon as dungeon_mod
import srd_lookup
from dungeon import WMTableEntry


DEFAULT_PORT = 8765

# Tags shown as checkboxes; order matches CLAUDE.md.
TAG_OPTIONS = list(config.ROOM_TAGS)


# --- HTML rendering ----------------------------------------------------------


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
</p>
{saved_banner}
{body}
<script>{PAGE_JS}</script>
</body>
</html>
"""


# --- Server handler ---------------------------------------------------------


class EditorHandler(BaseHTTPRequestHandler):
    """One handler instance per request. Reads the JSON fresh on every
    request — never holds a long-lived in-memory dungeon."""

    # Set by start_editor_server before binding.
    dungeon_path: Path = Path()

    # Silence the default per-request stderr log (pygame stdout is noisy).
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    # -- GET ---------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == "/healthz":
            self._respond(HTTPStatus.OK, b"ok\n", "text/plain; charset=utf-8")
            return
        if self.path.startswith("/?saved="):
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
        if self.path == "/":
            self._render()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    # -- POST --------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b""
        form = parse_qs(raw.decode("utf-8"), keep_blank_values=True)

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
                              extra_headers={"Location": f"/?saved={room_id}"})
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
                              extra_headers={"Location": f"/?saved=L{level_number}"})
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

    def _respond_level_fragment(self, level_number: int) -> None:
        d = dungeon_mod.load(self.dungeon_path)
        level = d.levels_by_number.get(level_number)
        if level is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Level missing after save")
            return
        body = _render_level_card(level).encode("utf-8")
        self._respond(HTTPStatus.OK, body, "text/html; charset=utf-8")

    # -- Internals ---------------------------------------------------------

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
        """Scan encounter_text + encounter_ref for monster names, pull the
        matching SRD stat blocks, and write the concatenation to
        room.statblocks. Replaces any prior enrichment wholesale — the DM
        edits encounter_text and re-clicks Enrich to refresh.

        Fuzzy hits (misspelt creature names) are prefixed with a small
        markdown caveat so the DM can spot and reject false positives.
        A timestamped backup of the dungeon.json is written first and
        the last 3 backups are retained."""
        d = dungeon_mod.load(self.dungeon_path)
        level = d.levels_by_number.get(level_number)
        if level is None:
            raise KeyError(f"unknown level {level_number}")
        room = level.rooms_by_id.get(room_id)
        if room is None:
            raise KeyError(f"unknown room {room_id} on level {level_number}")

        haystack = " ".join(filter(None, (
            room.encounter_text, room.encounter_ref, room.name)))
        matches = srd_lookup.scan(haystack)
        if not matches:
            room.statblocks = ""
        else:
            blocks: list[str] = []
            for m in matches:
                if m.source == "fuzzy":
                    blocks.append(
                        f"_(fuzzy match — input said {m.original!r})_\n"
                        f"{m.statblock.body}"
                    )
                else:
                    blocks.append(m.statblock.body)
            room.statblocks = "\n\n".join(blocks)
        # Back up before any mutating write so a bad Enrich is recoverable.
        dungeon_mod.backup_dungeon_json(self.dungeon_path, keep_last=3)
        dungeon_mod.dump(d, self.dungeon_path)

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
) -> tuple[HTTPServer, threading.Thread]:
    """Start the editor server in a daemon thread. Returns (server, thread).

    Pass `port=0` for an ephemeral port (the bound port can be read from
    `server.server_address[1]`); useful for tests.
    """
    EditorHandler.dungeon_path = Path(dungeon_path)
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
