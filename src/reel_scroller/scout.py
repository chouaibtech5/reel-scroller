"""Creator watchlist: fetch the newest public reels of creators you follow, rank each by how far it beat
that creator's usual views, and add the best ones to <out-root>/urls.tsv for `study`.

No login, no cookies: public profile data through the same Apify actor `study` uses. Dry run by default.

  creators.txt   one handle or profile URL per line (# comments allowed)
"""
from __future__ import annotations

import argparse
import re
import statistics
import time
from pathlib import Path

from . import study as _study

HANDLE_RE = re.compile(r"^[A-Za-z0-9._]{1,30}$")
USD_PER_RESULT = 0.0023  # apify/instagram-scraper per post; check current pricing yourself


def parse_creators(path: Path) -> list[str]:
    """Handles from a text file: '@name', 'name' or a profile URL per line. Duplicates and junk are dropped."""
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        m = re.search(r"instagram\.com/([A-Za-z0-9._]+)", s)
        h = (m.group(1) if m else s).lstrip("@").strip("/").lower()
        if h in ("reel", "reels", "p", "tv", "explore", "stories"):
            continue
        if HANDLE_RE.match(h) and h not in out:
            out.append(h)
    return out


def views(it: dict) -> int | None:
    for k in ("videoPlayCount", "videoViewCount"):
        v = it.get(k)
        if isinstance(v, (int, float)) and v >= 0:
            return int(v)
    return None


def is_reel(it: dict) -> bool:
    return it.get("type") == "Video" or it.get("productType") == "clips" or "/reel/" in (it.get("url") or "")


def rank(items: list[dict]) -> list[dict]:
    """Reels grouped by creator, each scored against that creator's median views ("outlier"), best first."""
    by: dict[str, list[dict]] = {}
    for it in items:
        if not is_reel(it) or not it.get("shortCode"):
            continue
        by.setdefault((it.get("ownerUsername") or "?").lower(), []).append(it)
    out = []
    for handle, its in by.items():
        vs = [v for v in (views(i) for i in its) if v]
        med = statistics.median(vs) if vs else None
        for it in its:
            v = views(it)
            out.append({"code": it["shortCode"], "url": f"https://www.instagram.com/reel/{it['shortCode']}/", "creator": handle,
                        "views": v, "median_views": round(med) if med else None,
                        "outlier": round(v / med, 2) if v and med else None, "posted": str(it.get("timestamp") or "")[:10],
                        "caption": (it.get("caption") or "")[:140].replace("\n", " ")})
    return sorted(out, key=lambda r: (r["outlier"] or 0, r["views"] or 0), reverse=True)


def pick(ranked: list[dict], done: set[str], min_outlier: float, top: int) -> list[dict]:
    """Not yet studied, at least `min_outlier` x the creator's usual views, at most `top` per creator."""
    per: dict[str, int] = {}
    out = []
    for r in ranked:
        if r["code"] in done:
            continue
        if min_outlier and (r["outlier"] or 0) < min_outlier:
            continue
        if top and per.get(r["creator"], 0) >= top:
            continue
        per[r["creator"]] = per.get(r["creator"], 0) + 1
        out.append(r)
    return out


def known_codes(out_root: Path) -> set[str]:
    """Codes already studied, failed, or already queued in urls.tsv."""
    codes = {p.stem for p in (out_root / "study").glob("*.json")} | _study.failed_codes(out_root / "study")
    tsv = out_root / "urls.tsv"
    if tsv.exists():
        codes |= {i["code"] for i in _study.parse_input(tsv)}
    return codes


def append_tsv(out_root: Path, picks: list[dict]) -> Path:
    tsv = out_root / "urls.tsv"
    with open(tsv, "a", encoding="utf-8") as f:
        for r in picks:
            f.write(f"{r['url']}\tcreator:{r['creator']}\n")
    return tsv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reel-scroller scout",
                                 description="Find your watchlist creators' best new reels and queue them for `study`. Dry run unless --run.")
    ap.add_argument("--creators", required=True, help="text file: one handle or profile URL per line")
    ap.add_argument("--out-root", default="./reel-scroller-out")
    ap.add_argument("--per-creator", type=int, default=12, help="newest posts to look at per creator (default 12)")
    ap.add_argument("--top", type=int, default=3, help="queue at most this many reels per creator (default 3, 0 = all)")
    ap.add_argument("--min-outlier", type=float, default=1.5,
                    help="only queue reels with at least this many times the creator's median views (default 1.5, 0 = any)")
    ap.add_argument("--run", action="store_true", help="call Apify (paid, small) and write the picks to urls.tsv")
    ap.add_argument("--apify-cap-usd", type=float, default=1.0, help="hard spending cap for this scout run")
    a = ap.parse_args(argv)

    creators = parse_creators(Path(a.creators))
    if not creators:
        print(f"no handles found in {a.creators}")
        return 2
    out_root = Path(a.out_root).expanduser()
    (out_root / "study").mkdir(parents=True, exist_ok=True)
    est = len(creators) * a.per_creator * USD_PER_RESULT
    print(f"{len(creators)} creators x {a.per_creator} posts = up to {len(creators) * a.per_creator} results, about ${est:.2f} (cap ${a.apify_cap_usd:.2f})")
    if not a.run:
        for h in creators:
            print(f"  would scan @{h}")
        print("dry run; pass --run to fetch (Apify is paid). Then analyze the picks with: reel-scroller study --input "
              f"{out_root / 'urls.tsv'} --run")
        return 0
    token = _study.apify_token()
    if not token:
        print("APIFY_TOKEN is not set. Export it in your shell (never put it in a file in this repo).")
        return 2
    try:
        items, run = _study.apify_fetch([f"https://www.instagram.com/{h}/" for h in creators], token, a.apify_cap_usd,
                                        results_limit=a.per_creator)
    except (_study.ApifyError, OSError) as e:
        print(f"apify failed: {e}")
        return 4
    _study.append_jsonl(out_root / "apify_cost.jsonl", {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "scout": len(creators), **run})
    ranked = rank(items)
    picks = pick(ranked, known_codes(out_root), a.min_outlier, a.top)
    _study.append_jsonl(out_root / "scout.jsonl", {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "creators": creators,
                                                   "seen": len(ranked), "picked": [p["code"] for p in picks], "ranked": ranked})
    print(f"apify: {run['items']} posts, ${run['usd']}. {len(ranked)} reels, {len(picks)} new picks:")
    for r in picks:
        ratio = f"{r['outlier']}x" if r["outlier"] else "?"
        print(f"  {ratio:>7}  @{r['creator']:<20} {r['views'] or '?':>9} views  {r['url']}  {r['caption'][:60]}")
    if picks:
        tsv = append_tsv(out_root, picks)
        print(f"added to {tsv}. Next: reel-scroller study --input {tsv} --out-root {out_root} --run --dash")
    else:
        print("nothing new above the outlier bar; try --min-outlier 0 or a larger --per-creator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
