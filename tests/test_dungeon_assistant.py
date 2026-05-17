"""Tests for dungeon_assistant.

The assistant now subprocesses the `claude` CLI rather than calling the
Anthropic API directly. Tests inject a fake `runner` callable so the
suite runs offline without the CLI installed.

Coverage:
- validate_proposal — happy path, unknown ids, bad tags, forbidden fields
- build_dungeon_context_block — stable text snapshot
- AssistantSession — start/send round-trip with a recording mock runner;
  --json-schema is forwarded; --resume reuses the session id; usage and
  cost are propagated to AssistantTurn
- check_cli_available — raises when binary is missing
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import config
import dungeon as dungeon_mod
import dungeon_assistant as da


EXAMPLE_PATH = Path(__file__).parent.parent / "data" / "example_dungeon.json"


@pytest.fixture
def example() -> dungeon_mod.Dungeon:
    return dungeon_mod.load(EXAMPLE_PATH, check_image_files=False)


# ----- Validation --------------------------------------------------------


class TestValidateProposal:
    def test_happy_path(self, example):
        prop = da.validate_proposal(
            {
                "id": "R01",
                "name": "Threshold",
                "tags": ["encounter"],
                "reaction_required": True,
                "box_text": "A dim room.",
                "encounter_text": "2 Skeletons (MM p.272). Hostile.",
            },
            example,
            level_number=1,
        )
        assert prop.id == "R01"
        assert prop.name == "Threshold"
        assert prop.tags == ("encounter",)
        assert prop.reaction_required is True
        assert prop.box_text == "A dim room."

    def test_unknown_room_id_raises(self, example):
        with pytest.raises(ValueError, match="unknown room id"):
            da.validate_proposal(
                {"id": "BOGUS", "name": "x", "tags": ["empty"]},
                example, level_number=1,
            )

    def test_unknown_level_raises(self, example):
        with pytest.raises(ValueError, match="level 999"):
            da.validate_proposal(
                {"id": "R01", "name": "x", "tags": ["empty"]},
                example, level_number=999,
            )

    def test_bad_tags_raise(self, example):
        with pytest.raises(ValueError, match="unknown tags"):
            da.validate_proposal(
                {"id": "R01", "name": "x", "tags": ["nonexistent"]},
                example, level_number=1,
            )

    @pytest.mark.parametrize("forbidden", [
        "state", "image_region", "encounter_ref", "treasure_tier",
        "statblocks",
    ])
    def test_forbidden_fields_raise(self, example, forbidden):
        with pytest.raises(ValueError, match="forbidden"):
            da.validate_proposal(
                {
                    "id": "R01", "name": "x", "tags": ["empty"],
                    forbidden: "anything",
                },
                example, level_number=1,
            )

    def test_missing_id_raises(self, example):
        with pytest.raises(ValueError, match="missing room id"):
            da.validate_proposal(
                {"name": "x", "tags": ["empty"]},
                example, level_number=1,
            )

    def test_empty_tags_default_to_empty(self, example):
        prop = da.validate_proposal(
            {"id": "R01", "name": "x"},
            example, level_number=1,
        )
        assert prop.tags == ("empty",)

    def test_proposal_serialises_to_dict(self, example):
        prop = da.validate_proposal(
            {"id": "R01", "name": "n", "tags": ["empty"]},
            example, level_number=1,
        )
        d = prop.to_dict()
        assert d["id"] == "R01"
        assert d["tags"] == ["empty"]
        # All editable fields are present even when empty.
        assert "encounter_text" in d
        assert "treasure_text" in d


# ----- Context block -----------------------------------------------------


class TestContextBlock:
    def test_includes_dungeon_metadata(self, example):
        text = da.build_dungeon_context_block(example, 1)
        assert example.name in text
        assert "Party level" in text
        assert "Wandering monster table" in text

    def test_marks_populated_vs_empty(self, example):
        for room in example.levels[0].rooms:
            room.name = room.id
            room.notes = ""
            room.box_text = ""
            room.encounter_text = ""
            room.treasure_text = ""
            room.special_text = ""
        example.levels[0].rooms[0].box_text = "A vault."
        example.levels[0].rooms[0].name = "Vault"
        text = da.build_dungeon_context_block(example, 1)
        lines = text.splitlines()
        r01_line = next(line for line in lines if line.strip().startswith("R01"))
        r02_line = next(line for line in lines if line.strip().startswith("R02"))
        assert "POPULATED" in r01_line
        assert "empty" in r02_line

    def test_unknown_level_returns_message(self, example):
        text = da.build_dungeon_context_block(example, 999)
        assert "999" in text
        assert "not found" in text


# ----- CLI availability --------------------------------------------------


class TestCliAvailable:
    def test_raises_when_binary_missing(self, monkeypatch):
        # Force shutil.which to report missing.
        import shutil
        monkeypatch.setattr(shutil, "which", lambda _: None)
        with pytest.raises(da.AssistantUnavailable, match="CLI not found"):
            da.check_cli_available()


# ----- Session round-trip with mocked runner -----------------------------


class _RecordingRunner:
    """A fake CLI runner. Records every argv; pops responses in order."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> dict:
        self.calls.append(list(args))
        if not self._responses:
            raise AssertionError("no more canned responses")
        return self._responses.pop(0)


def _envelope(*, structured: dict, text: str = "ok",
              session_id: str = "sesn-test", cost: float = 0.05,
              usage: dict | None = None, is_error: bool = False) -> dict:
    """Build a CLI JSON envelope shaped like `claude --output-format=json`."""
    return {
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "result": text,
        "session_id": session_id,
        "structured_output": structured,
        "usage": usage or {
            "input_tokens": 100, "output_tokens": 50,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        "total_cost_usd": cost,
    }


def _make_session(example, *, responses, theme="catacombs"):
    return da.AssistantSession(
        dungeon_path=EXAMPLE_PATH,
        dungeon=example,
        theme=theme,
        level_number=1,
        party_level=3,
        runner=_RecordingRunner(responses),
    )


class TestAssistantSession:
    def test_start_parses_text_summary_and_proposals(self, example):
        sess = _make_session(example, responses=[
            _envelope(
                text="Plan: 4 rooms in batch one.",
                structured={
                    "summary": "Batch 1: R01.",
                    "rooms": [{
                        "id": "R01", "name": "Threshold",
                        "tags": ["encounter"],
                        "box_text": "Cold air.",
                        "encounter_text": "2 Skeletons.",
                    }],
                },
                cost=0.07,
            ),
        ])
        turn = sess.start()
        assert "batch one" in turn.text
        assert turn.summary == "Batch 1: R01."
        assert len(turn.proposals) == 1
        assert turn.proposals[0].id == "R01"
        assert turn.proposals[0].box_text == "Cold air."
        assert turn.cost_usd == 0.07
        assert turn.rejected == []
        assert sess.session_id == "sesn-test"

    def test_invalid_proposals_go_to_rejected_not_accepted(self, example):
        sess = _make_session(example, responses=[
            _envelope(structured={
                "summary": "",
                "rooms": [
                    {"id": "R01", "name": "Good", "tags": ["empty"]},
                    {"id": "GHOST", "name": "Bad", "tags": ["empty"]},
                    {"id": "R02", "name": "BadTag", "tags": ["nonsense"]},
                ],
            }),
        ])
        turn = sess.start()
        assert [p.id for p in turn.proposals] == ["R01"]
        assert len(turn.rejected) == 2
        assert "GHOST" in turn.rejected[0]["error"]

    def test_first_call_includes_system_prompt_and_schema(self, example):
        sess = _make_session(example, responses=[
            _envelope(structured={
                "summary": "x",
                "rooms": [{"id": "R01", "name": "x", "tags": ["empty"]}],
            }),
        ])
        sess.start()
        argv = sess.runner.calls[0]
        assert "--print" in argv
        assert "--output-format" in argv and "json" in argv
        assert "--system-prompt" in argv
        assert "--json-schema" in argv
        # Schema is the JSON-serialised structured-output spec.
        schema_idx = argv.index("--json-schema")
        schema = json.loads(argv[schema_idx + 1])
        assert "rooms" in schema["properties"]
        # Model is forwarded.
        assert "--model" in argv

    def test_system_prompt_includes_srd_creature_constraint(self, example):
        """The assistant's system prompt must list every SRD creature
        so the model only proposes encounters that the editor's
        enrichment and the simulator can resolve."""
        sess = _make_session(example, responses=[
            _envelope(structured={
                "summary": "x",
                "rooms": [{"id": "R01", "name": "x", "tags": ["empty"]}],
            }),
        ])
        sess.start()
        argv = sess.runner.calls[0]
        prompt = argv[argv.index("--system-prompt") + 1]
        # Constraint header is present.
        assert "SRD CREATURE CONSTRAINT" in prompt
        # Sample SRD creatures appear in the list.
        import srd_lookup
        names = srd_lookup.names()
        # Sample five — head, middle, tail.
        sample = [names[0], names[len(names) // 4], names[len(names) // 2],
                  names[3 * len(names) // 4], names[-1]]
        for n in sample:
            assert n in prompt, f"expected {n} in system prompt"
        # Banshee is a well-known SRD omission; it must NOT appear in
        # the constraint list (otherwise the assistant could propose
        # encounters Enrich would silently drop).
        # We check that "Banshee" appears nowhere as a list entry —
        # i.e. not surrounded by ", " or at end of list.
        assert ", Banshee," not in prompt
        assert ", Banshee\n" not in prompt

    def test_send_uses_resume_with_session_id(self, example):
        sess = _make_session(example, responses=[
            _envelope(structured={
                "summary": "first",
                "rooms": [{"id": "R01", "name": "x", "tags": ["empty"]}],
            }, session_id="sesn-abc"),
            _envelope(structured={
                "summary": "revised",
                "rooms": [{"id": "R01", "name": "y", "tags": ["empty"]}],
            }, session_id="sesn-abc"),
        ])
        sess.start()
        sess.send("rewrite R01")
        second_argv = sess.runner.calls[1]
        # Second call uses --resume <session_id>, NOT --system-prompt
        # (that lives on the CLI session already).
        assert "--resume" in second_argv
        assert second_argv[second_argv.index("--resume") + 1] == "sesn-abc"
        assert "--system-prompt" not in second_argv
        assert sess.latest_proposals["R01"].name == "y"

    def test_send_before_start_raises(self, example):
        sess = _make_session(example, responses=[])
        with pytest.raises(RuntimeError, match="not started"):
            sess.send("hi")

    def test_empty_user_message_rejected(self, example):
        sess = _make_session(example, responses=[
            _envelope(structured={
                "summary": "",
                "rooms": [{"id": "R01", "name": "x", "tags": ["empty"]}],
            }),
        ])
        sess.start()
        with pytest.raises(ValueError, match="empty"):
            sess.send("   ")

    def test_reset_clears_state(self, example):
        sess = _make_session(example, responses=[
            _envelope(structured={
                "summary": "",
                "rooms": [{"id": "R01", "name": "x", "tags": ["empty"]}],
            }),
        ])
        sess.start()
        assert sess.latest_proposals
        sess.reset()
        assert sess.session_id is None
        assert sess.latest_proposals == {}

    def test_cli_error_envelope_raises(self, example):
        sess = _make_session(example, responses=[
            {"is_error": True, "result": "auth required"},
        ])
        with pytest.raises(RuntimeError, match="auth required"):
            sess.start()

    def test_cost_and_usage_propagate(self, example):
        sess = _make_session(example, responses=[
            _envelope(
                structured={"summary": "", "rooms": []},
                usage={
                    "input_tokens": 10, "output_tokens": 20,
                    "cache_read_input_tokens": 5000,
                    "cache_creation_input_tokens": 100,
                },
                cost=0.123,
            ),
        ])
        turn = sess.start()
        assert turn.cost_usd == 0.123
        assert turn.usage["cache_read_input_tokens"] == 5000

    def test_opening_prompt_includes_image_path(self, example):
        sess = _make_session(example, responses=[
            _envelope(structured={"summary": "", "rooms": []}),
        ])
        sess.start()
        argv = sess.runner.calls[0]
        # Last arg is the user prompt; it should mention the level's
        # map_image filename for the CLI's read tool to pick up.
        prompt = argv[-1]
        assert example.levels[0].map_image.split("/")[-1] in prompt


# ----- Default-runner subprocess plumbing --------------------------------


class TestDefaultRunner:
    def test_timeout_wraps_as_runtime_error(self, monkeypatch):
        import subprocess as _sp
        def _boom(*a, **kw):
            raise _sp.TimeoutExpired(cmd="claude", timeout=1)
        monkeypatch.setattr(_sp, "run", _boom)
        with pytest.raises(RuntimeError, match="timed out"):
            da._default_runner(["claude", "--print", "hi"])

    def test_nonzero_exit_wraps_as_runtime_error(self, monkeypatch):
        import subprocess as _sp
        from types import SimpleNamespace as NS
        monkeypatch.setattr(_sp, "run", lambda *a, **kw: NS(
            returncode=1, stdout="", stderr="not signed in",
        ))
        with pytest.raises(RuntimeError, match="not signed in"):
            da._default_runner(["claude", "--print", "hi"])

    def test_non_json_stdout_wraps(self, monkeypatch):
        import subprocess as _sp
        from types import SimpleNamespace as NS
        monkeypatch.setattr(_sp, "run", lambda *a, **kw: NS(
            returncode=0, stdout="not json at all", stderr="",
        ))
        with pytest.raises(RuntimeError, match="non-JSON"):
            da._default_runner(["claude", "--print", "hi"])
