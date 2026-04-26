"""Test configuration shared across the suite."""

from __future__ import annotations

import os

# Force pygame to use a headless SDL driver so tests work in CI / over SSH /
# on machines without a display server. Must be set before pygame import.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
