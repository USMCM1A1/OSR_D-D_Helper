"""Headless CLI runner for the dungeon turn engine.

Usage:
    python cli.py data/example_dungeon.json --turns 50 --seed 42
    python cli.py data/example_dungeon.json --interactive

For the rendered DM view run `python main.py` (Phase 4+). This CLI is
kept for headless reproduction of seeded runs and debugging.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import config
from dungeon import load
from journal import format_entry
from tracker import Tracker


def _print_header(t: Tracker) -> None:
    d = t.dungeon
    cmp_word = "≥" if d.wm_check_method == config.WM_METHOD_D20 else "≤"
    print(f"=== {d.name} ===")
    print(
        f"Party level {d.party_level} ({d.party.size} characters). "
        f"WM check: {d.wm_check_method} {cmp_word} {d.wm_check_threshold} every turn."
    )
    if t.light_sources:
        sources = ", ".join(ls.label for ls in t.light_sources)
        print(f"Lit: {sources}")
    print()


def _print_footer(t: Tracker) -> None:
    h, m = t.elapsed_hm
    print()
    print(f"=== Stopped at turn {t.turn} ({h}h{m:02d}m) ===")
    if t.light_sources:
        for ls in t.light_sources:
            print(f"  {ls.label}: {ls.turns_remaining} turns remaining")
    else:
        print("  No active light sources.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless dungeon turn engine.")
    p.add_argument("dungeon", type=Path, help="Path to dungeon JSON file.")
    p.add_argument("--turns", type=int, default=10,
                   help="Turns to simulate in batch mode (default: 10).")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed RNG for reproducibility.")
    p.add_argument("--torches", type=int, default=2,
                   help="Starting torch count (default: 2).")
    p.add_argument("--lanterns", type=int, default=0,
                   help="Starting hooded lantern count (default: 0).")
    p.add_argument("--interactive", action="store_true",
                   help="Press Enter to advance each turn instead of batch.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    d = load(args.dungeon)
    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    t = Tracker(d, rng=rng)

    for _ in range(args.torches):
        t.add_light_source("torch")
    for _ in range(args.lanterns):
        t.add_light_source("hooded_lantern")

    _print_header(t)

    if args.interactive:
        try:
            while True:
                input("[Enter to advance turn, Ctrl-C to exit] ")
                for entry in t.advance_turn():
                    print(format_entry(entry))
        except (EOFError, KeyboardInterrupt):
            print()
    else:
        for _ in range(args.turns):
            for entry in t.advance_turn():
                print(format_entry(entry))

    _print_footer(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
