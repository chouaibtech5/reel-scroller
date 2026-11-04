"""One command for the whole routine: scout your watchlist, study the best new reels, turn them into ideas,
with the dashboard open the whole time.

Free unless you say otherwise: scouting goes through Meta's API, downloads through yt-dlp. Apify is only used
with --allow-paid (and an APIFY_TOKEN).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from . import dashboard, ideas, scout, sources, study


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reel-scroller auto",
                                 description="Scout your watchlist, study the best new reels and turn them into ideas, in one go.")
    ap.add_argument("--creators", required=True, help="text file: one handle or profile URL per line")
    ap.add_argument("--out-root", default="./reel-scroller-out")
    ap.add_argument("--niche", default=os.environ.get("REEL_SCROLLER_CONTEXT", ""), help="who the ideas are for")
    ap.add_argument("--backend", help="vision model: ollama (local, default), gemini (fast, needs GEMINI_API_KEY), openai, mlx")
    ap.add_argument("--model", help="exact vision model name")
    ap.add_argument("--ideas-model", help="text model for the ideas step (default: the vision model)")
    ap.add_argument("--per-creator", type=int, default=12)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--min-outlier", type=float, default=1.5)
    ap.add_argument("--count", type=int, default=10, help="how many ideas")
    ap.add_argument("--limit", type=int, default=0, help="study at most N reels this run")
    ap.add_argument("--allow-paid", action="store_true", help="let Apify fill in what the free routes cannot get (needs APIFY_TOKEN)")
    ap.add_argument("--no-dash", action="store_true", help="do not open the dashboard")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    out = Path(a.out_root).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if not sources.meta_config() and not a.allow_paid:
        print("auto scouts through Meta's free API: set META_IG_USER_ID and META_ACCESS_TOKEN (see README, Free sources),\n"
              "or pass --allow-paid to scout through Apify instead.")
        return 2
    if not a.no_dash:
        dashboard.serve_in_background(out, a.port)

    t0 = time.time()
    print("\n== 1/3 scout", flush=True)
    rc = scout.main(["--creators", a.creators, "--out-root", str(out), "--run", "--per-creator", str(a.per_creator),
                     "--top", str(a.top), "--min-outlier", str(a.min_outlier)] + ([] if a.allow_paid else ["--source", "meta"]))
    if rc:
        return rc

    print("\n== 2/3 study", flush=True)
    model_args = (["--backend", a.backend] if a.backend else []) + (["--model", a.model] if a.model else [])
    if (out / "urls.tsv").exists():
        rc = study.main(["--input", str(out / "urls.tsv"), "--out-root", str(out), "--run",
                         "--fetcher", "auto" if a.allow_paid else "ytdlp"] + model_args + (["--limit", str(a.limit)] if a.limit else []))
        if rc:
            return rc
    else:
        print("nothing queued yet")

    print("\n== 3/3 ideas", flush=True)
    idea_args = ["--out-root", str(out), "--count", str(a.count)] + (["--niche", a.niche] if a.niche else [])
    idea_args += (["--backend", a.backend] if a.backend else []) + (["--model", a.ideas_model or a.model] if (a.ideas_model or a.model) else [])
    rc = ideas.main(idea_args)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min", flush=True)
    if not a.no_dash:
        print(f"dashboard: http://127.0.0.1:{a.port}  (Ideas tab; Ctrl+C to quit)", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return rc
