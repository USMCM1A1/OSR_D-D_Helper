"""Tests for character_ingester. The CLI is mocked so tests run offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import character_ingester as ci


# ---------- Fake CLI runner --------------------------------------------


class _RecordingRunner:
    """Mirrors tests/test_dungeon_assistant.py:_RecordingRunner."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> dict:
        self.calls.append(list(args))
        if not self._responses:
            raise AssertionError("no more canned responses")
        return self._responses.pop(0)


def _envelope(*, structured: dict | None = None,
              is_error: bool = False, result: str = "ok") -> dict:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "result": result,
        "session_id": "sesn-test",
        "structured_output": structured,
        "usage": {"input_tokens": 100, "output_tokens": 50,
                  "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 0},
        "total_cost_usd": 0.05,
    }


_THORIN = {
    "name": "Thorin",
    "class": "fighter",
    "level": 3,
    "ac": 18,
    "hp_max": 28,
    "speed": 30,
    "init_bonus": 1,
    "ability_mods": {"str": 3, "dex": 1, "con": 2,
                     "int": 0, "wis": 1, "cha": 0},
    "saves": {"str": 5, "con": 4, "dex": 1,
              "int": 0, "wis": 1, "cha": 0},
    "attacks": [
        {"name": "Longsword", "to_hit": 5, "damage": "1d10+3",
         "damage_type": "slashing", "range": "melee"},
    ],
    "spells": {"slots": {}, "memorized": []},
    "features": {"second_wind": True, "sneak_attack_dice": 0},
}


# ---------- extract_character ------------------------------------------


class TestExtractCharacter:
    def test_happy_path(self):
        runner = _RecordingRunner([_envelope(structured=_THORIN)])
        result = ci.extract_character("dummy pdf text", runner=runner)
        assert result == _THORIN
        assert len(runner.calls) == 1

    def test_args_include_schema_and_system_prompt(self):
        runner = _RecordingRunner([_envelope(structured=_THORIN)])
        ci.extract_character("some text", runner=runner)
        argv = runner.calls[0]
        assert "--json-schema" in argv
        schema_idx = argv.index("--json-schema")
        schema = json.loads(argv[schema_idx + 1])
        assert "name" in schema["properties"]
        assert "attacks" in schema["properties"]
        assert "--system-prompt" in argv
        prompt_idx = argv.index("--system-prompt")
        assert "5e character sheet" in argv[prompt_idx + 1]

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ci.extract_character("", runner=_RecordingRunner([]))

    def test_cli_error_propagates(self):
        runner = _RecordingRunner([
            _envelope(is_error=True, result="auth failed", structured=None),
        ])
        with pytest.raises(RuntimeError, match="auth failed"):
            ci.extract_character("some pdf text", runner=runner)

    def test_missing_structured_output_raises(self):
        runner = _RecordingRunner([_envelope(structured=None)])
        with pytest.raises(RuntimeError, match="structured_output"):
            ci.extract_character("some pdf text", runner=runner)


# ---------- save_character / load_characters ---------------------------


class TestPersistence:
    def test_save_creates_dir_and_file(self, tmp_path: Path):
        dungeon = tmp_path / "test-dungeon"
        dungeon.mkdir()
        path = ci.save_character(_THORIN, dungeon)
        assert path.parent == dungeon / "characters"
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == _THORIN

    def test_save_uses_slug_filename(self, tmp_path: Path):
        dungeon = tmp_path / "d"
        dungeon.mkdir()
        path = ci.save_character({"name": "Sera Quickfingers"}, dungeon)
        assert path.name == "sera-quickfingers.json"

    def test_save_explicit_filename(self, tmp_path: Path):
        dungeon = tmp_path / "d"
        dungeon.mkdir()
        path = ci.save_character(_THORIN, dungeon, filename="boss")
        assert path.name == "boss.json"

    def test_save_path_pointing_at_json_file_uses_parent(self, tmp_path: Path):
        # Caller hands us dungeons/foo/dungeon.json — we should still
        # resolve characters/ next to it, not inside it.
        dungeon = tmp_path / "d"
        dungeon.mkdir()
        json_path = dungeon / "dungeon.json"
        json_path.write_text("{}")
        path = ci.save_character(_THORIN, json_path)
        assert path.parent == dungeon / "characters"

    def test_save_atomic(self, tmp_path: Path):
        dungeon = tmp_path / "d"
        dungeon.mkdir()
        ci.save_character(_THORIN, dungeon)
        # Re-save with a different value; atomic write should leave no
        # .tmp residue.
        ci.save_character({**_THORIN, "ac": 20}, dungeon)
        residues = list((dungeon / "characters").glob("*.tmp"))
        assert residues == []

    def test_load_round_trips(self, tmp_path: Path):
        dungeon = tmp_path / "d"
        dungeon.mkdir()
        ci.save_character(_THORIN, dungeon)
        ci.save_character({**_THORIN, "name": "Mira"}, dungeon)
        loaded = ci.load_characters(dungeon)
        names = sorted(c["name"] for c in loaded)
        assert names == ["Mira", "Thorin"]

    def test_load_empty_when_no_dir(self, tmp_path: Path):
        dungeon = tmp_path / "d"
        dungeon.mkdir()
        assert ci.load_characters(dungeon) == []

    def test_load_skips_invalid_json(self, tmp_path: Path):
        dungeon = tmp_path / "d"
        dungeon.mkdir()
        char_dir = dungeon / "characters"
        char_dir.mkdir()
        (char_dir / "good.json").write_text(json.dumps(_THORIN))
        (char_dir / "bad.json").write_text("not json {")
        loaded = ci.load_characters(dungeon)
        assert len(loaded) == 1
        assert loaded[0]["name"] == "Thorin"


# ---------- Form-field extraction (AcroForm widgets) ------------------


class _FakeAnnotObj(dict):
    """Dict that satisfies pypdf's annotation-object protocol enough
    for _extract_form_field_lines: dict-like .get() and a no-op
    .get_object() that returns self."""

    def get_object(self):
        return self


class _FakePage:
    def __init__(self, annots):
        self._annots = annots

    def get(self, key):
        if key == "/Annots":
            return self._annots
        return None


class _FakeReader:
    def __init__(self, pages):
        self.pages = pages


class TestFormFieldExtraction:
    """Fillable character sheets (D&D Beyond export, Mythweavers, etc.)
    often store every value in form widget annotations whose values
    don't appear in `extract_text()`. _extract_form_field_lines walks
    those widgets directly. This test verifies the walk."""

    def test_walks_widgets_emits_name_value_pairs(self):
        page = _FakePage([
            _FakeAnnotObj({
                "/Subtype": "/Widget",
                "/T": "CharacterName",
                "/V": "Dorcas",
            }),
            _FakeAnnotObj({
                "/Subtype": "/Widget",
                "/T": "CLASS  LEVEL",
                "/V": "Cleric 1",
            }),
        ])
        text = ci._extract_form_field_lines(_FakeReader([page]))
        assert "CharacterName: Dorcas" in text
        assert "CLASS  LEVEL: Cleric 1" in text

    def test_skips_unchecked_and_empty_values(self):
        page = _FakePage([
            _FakeAnnotObj({"/Subtype": "/Widget", "/T": "BoxA", "/V": "/Off"}),
            _FakeAnnotObj({"/Subtype": "/Widget", "/T": "BoxB", "/V": ""}),
            _FakeAnnotObj({"/Subtype": "/Widget", "/T": "BoxC", "/V": "--"}),
            _FakeAnnotObj({"/Subtype": "/Widget", "/T": "BoxD", "/V": "Yes"}),
        ])
        text = ci._extract_form_field_lines(_FakeReader([page]))
        assert "BoxA" not in text
        assert "BoxB" not in text
        assert "BoxC" not in text
        assert "BoxD: Yes" in text

    def test_skips_non_widget_subtypes(self):
        page = _FakePage([
            _FakeAnnotObj({
                "/Subtype": "/Link",  # not a form widget
                "/T": "X",
                "/V": "Y",
            }),
        ])
        text = ci._extract_form_field_lines(_FakeReader([page]))
        assert text == ""

    def test_dedupes_repeated_fields_across_pages(self):
        # The D&D Beyond template puts CharacterName on every page.
        annot = _FakeAnnotObj({
            "/Subtype": "/Widget", "/T": "CharacterName", "/V": "Dorcas",
        })
        text = ci._extract_form_field_lines(
            _FakeReader([_FakePage([annot]), _FakePage([annot])])
        )
        assert text.count("Dorcas") == 1

    def test_no_pages_returns_empty(self):
        assert ci._extract_form_field_lines(_FakeReader([])) == ""

    def test_value_without_field_name_still_emitted(self):
        page = _FakePage([
            _FakeAnnotObj({"/Subtype": "/Widget", "/V": "Lone Value"}),
        ])
        assert ci._extract_form_field_lines(_FakeReader([page])) \
            == "Lone Value"


# ---------- check_cli_available ----------------------------------------


class TestCliPresence:
    def test_raises_when_missing(self, monkeypatch):
        import shutil
        monkeypatch.setattr(shutil, "which", lambda _: None)
        with pytest.raises(ci.IngesterUnavailable, match="CLI not found"):
            ci.check_cli_available()

    def test_no_raise_when_present(self, monkeypatch):
        import shutil
        monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/claude")
        ci.check_cli_available()  # no exception
