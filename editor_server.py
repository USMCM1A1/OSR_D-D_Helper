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


DEFAULT_PORT = 8765

# Tags shown as checkboxes; order matches CLAUDE.md.
TAG_OPTIONS = list(config.ROOM_TAGS)
# Treasure tiers for the dropdown — `""` means "no tier".
TREASURE_TIERS = ("", *config.TREASURE_TIERS)


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

    <label>Treasure details <span class="help">items, gold, location, hidden vs obvious</span>
      <textarea name="treasure_text" rows="3">{_esc(room.treasure_text)}</textarea>
    </label>

    <div class="row">
      <label>Encounter ref <span class="help">D&amp;D Beyond encounter name</span>
        <input type="text" name="encounter_ref" value="{_esc(room.encounter_ref or "")}">
      </label>
      <label>Treasure tier <span class="help">CR band for procedural lookup</span>
        {_select("treasure_tier", TREASURE_TIERS, room.treasure_tier)}
      </label>
    </div>

    <div class="save-row">
      <button type="submit" disabled>Save</button>
    </div>
  </form>
</div>
""".strip()


PAGE_JS = """
// Dirty-tracking: Save button starts disabled; any input/change in the
// form re-enables it. On submit the page re-loads and the buttons reset.
(function () {
  document.querySelectorAll('form').forEach(function (form) {
    var btn = form.querySelector('button[type=submit]');
    if (!btn) return;
    function markDirty() {
      btn.disabled = false;
    }
    form.addEventListener('input', markDirty);
    form.addEventListener('change', markDirty);
  });
})();
""".strip()


def _render_page(d, *, saved_room_id: str | None = None) -> str:
    """Build the full HTML page from a Dungeon."""
    sections = []
    for level in d.levels:
        sections.append(
            f'<h2 class="level">Level {level.level_number} — '
            f'{_esc(level.display_name)}</h2>'
        )
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
    if saved_room_id:
        saved_banner = (
            f'<p class="saved" style="text-align:right;color:#527a3e;'
            f'font-style:italic;">Saved {_esc(saved_room_id)}.</p>'
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
            self._render(saved_room_id=saved_id)
            return
        if self.path == "/":
            self._render()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    # -- POST --------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/room":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b""
        form = parse_qs(raw.decode("utf-8"), keep_blank_values=True)

        try:
            level_number = int(form.get("level_number", ["0"])[0])
            room_id = form.get("room_id", [""])[0]
            if not room_id:
                raise ValueError("missing room_id")
            self._mutate_and_save(level_number, room_id, form)
        except (ValueError, KeyError, dungeon_mod.DungeonValidationError) as e:
            self.send_error(HTTPStatus.BAD_REQUEST, f"Bad request: {e}")
            return

        # 303 See Other → browser GET / so the form re-renders fresh and
        # refresh-on-back is idempotent (Post/Redirect/Get pattern).
        self._respond(HTTPStatus.SEE_OTHER, b"",
                      content_type="text/plain",
                      extra_headers={"Location": f"/?saved={room_id}"})

    # -- Internals ---------------------------------------------------------

    def _render(self, *, saved_room_id: str | None = None) -> None:
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
        body = _render_page(d, saved_room_id=saved_room_id).encode("utf-8")
        self._respond(HTTPStatus.OK, body, "text/html; charset=utf-8")

    def _mutate_and_save(self, level_number: int, room_id: str,
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
        room.encounter_ref = (form.get("encounter_ref", [""])[0] or None)
        tier = form.get("treasure_tier", [""])[0]
        room.treasure_tier = tier or None
        room.reaction_required = "reaction_required" in form

        # Tags: incoming list under "tags". Filter to known + non-empty.
        raw_tags = form.get("tags", []) or []
        valid = [t for t in raw_tags if t in config.ROOM_TAGS]
        room.tags = tuple(valid) if valid else ("empty",)

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
