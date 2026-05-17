"""Single pywebview window — invoked as a subprocess by main.py.

Background: macOS requires both pygame (SDL2) and pywebview to own
the main thread. Running them in the same process means one of them
loses. Spawning a fresh Python subprocess per webview window
side-steps the conflict cleanly — each window owns its process's
main thread without touching pygame's.

Usage:
    python osr_webview_window.py --url URL --title TITLE
                                  [--width N] [--height N]

When pywebview is unavailable the script exits 2 with a message;
main.py uses that as the signal to fall back to webbrowser.open.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Single OSR pywebview window.")
    parser.add_argument("--url", required=True,
                        help="URL the window should load")
    parser.add_argument("--title", default="OSR Dungeon",
                        help="Window title shown in the title bar")
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()

    try:
        import webview
    except ImportError:
        print(
            "pywebview is not installed. "
            "Install with `pip install pywebview` for native windows, "
            "or remove the --webview flag to use the default browser.",
            file=sys.stderr,
        )
        return 2

    webview.create_window(
        args.title,
        args.url,
        width=args.width,
        height=args.height,
        resizable=True,
        text_select=True,
    )
    # Block until the window closes. The parent process doesn't wait
    # on this subprocess (start_new_session=True in main.py), so the
    # user can close the window without affecting pygame.
    webview.start(debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
