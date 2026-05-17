"""Tests for main.py --scaffold.

The flag is the safe argv-driven path the bash launcher (and tests)
use to scaffold a new dungeon without piping Python into Python.
It exits 0 after printing the new folder path to stdout, or 2 with
a stderr message on validation / collision errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import main as main_mod


def test_scaffold_creates_folder_and_prints_path(tmp_path, capsys):
    dungeons_dir = tmp_path / "dungeons"
    dungeons_dir.mkdir()
    rc = main_mod.main([
        "--scaffold", "The Test Hold",
        "--party-level", "5",
        "--party-size", "3",
        "--dungeons-dir", str(dungeons_dir),
    ])
    assert rc == 0
    captured = capsys.readouterr()
    out_path = Path(captured.out.strip())
    assert out_path == (dungeons_dir / "the-test-hold").resolve()
    assert (out_path / "dungeon.json").exists()
    scaffold = json.loads((out_path / "dungeon.json").read_text())
    assert scaffold["dungeon_name"] == "The Test Hold"
    assert scaffold["party_level"] == 5
    # party_size flows through to the party.characters list length.
    assert len(scaffold["party"]["characters"]) == 3


def test_scaffold_rejects_empty_name(tmp_path, capsys):
    dungeons_dir = tmp_path / "dungeons"
    dungeons_dir.mkdir()
    rc = main_mod.main([
        "--scaffold", "   ",
        "--dungeons-dir", str(dungeons_dir),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "non-empty" in err
    # Nothing got created.
    assert list(dungeons_dir.iterdir()) == []


def test_scaffold_punctuation_name_falls_back_to_untitled(tmp_path, capsys):
    # slugify_dungeon_name has an "untitled-dungeon" fallback for
    # names that would otherwise slug to empty (pure punctuation).
    # Document the resulting folder here so changes to the slugger
    # are caught.
    dungeons_dir = tmp_path / "dungeons"
    dungeons_dir.mkdir()
    rc = main_mod.main([
        "--scaffold", "!!! ???",
        "--dungeons-dir", str(dungeons_dir),
    ])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out.endswith("untitled-dungeon")
    assert (dungeons_dir / "untitled-dungeon").exists()


def test_scaffold_rejects_collision(tmp_path, capsys):
    dungeons_dir = tmp_path / "dungeons"
    dungeons_dir.mkdir()
    (dungeons_dir / "collision").mkdir()
    rc = main_mod.main([
        "--scaffold", "Collision",
        "--dungeons-dir", str(dungeons_dir),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "already exists" in err
