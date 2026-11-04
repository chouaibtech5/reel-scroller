"""reel-scroller command line: study | export | advice."""
from __future__ import annotations

import sys

USAGE = """usage: reel-scroller <command> [options]

commands:
  export   convert an Instagram data export (saved_posts.json) into a URL list
  study    download (free yt-dlp, Apify optional) and analyze reels locally; --local for mp4 files; dry run by default
  advice   render an advice library JSON to a static HTML page (optional)
  dash     local dashboard: watch a run live and browse analyzed reels
  scout    find your watchlist creators' best new reels and queue them for study
  ideas    turn studied reels into patterns and new reel ideas for your niche
  auto     all of it in one go: scout, study and ideas, with the dashboard open

Run `reel-scroller <command> --help` for options."""


def utf8_output() -> None:
    """Captions are full of emoji and non-Latin text; Windows consoles and pipes default to cp1252 and would crash."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main() -> int:
    utf8_output()
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        return 0
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "study":
        from .study import main as run
    elif cmd == "export":
        from .ig_export import main as run
    elif cmd == "advice":
        from .advice_library import main as run
    elif cmd == "dash":
        from .dashboard import main as run
    elif cmd == "scout":
        from .scout import main as run
    elif cmd == "ideas":
        from .ideas import main as run
    elif cmd == "auto":
        from .auto import main as run
    else:
        print(USAGE)
        return 2
    return run(rest)


if __name__ == "__main__":
    sys.exit(main())
