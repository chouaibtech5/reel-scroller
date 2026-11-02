"""reel-scroller command line: study | export | advice."""
from __future__ import annotations

import sys

USAGE = """usage: reel-scroller <command> [options]

commands:
  export   convert an Instagram data export (saved_posts.json) into a URL list
  study    fetch (Apify) and analyze saved reels locally; --local analyzes mp4 files; dry run by default
  advice   render an advice library JSON to a static HTML page (optional)
  dash     local dashboard: watch a run live and browse analyzed reels

Run `reel-scroller <command> --help` for options."""


def main() -> int:
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
    else:
        print(USAGE)
        return 2
    return run(rest)


if __name__ == "__main__":
    sys.exit(main())
