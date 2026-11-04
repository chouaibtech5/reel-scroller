"""Ideas: read every study, find what the strongest reels have in common, and turn it into new reel
ideas for your niche. Text only, so any local text model works (for example qwen3:8b in Ollama).

Writes <out-root>/ideas/ideas-<date>.json and <out-root>/ideas/latest.json (shown in the dashboard).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from . import model as _model
from .backends.vision import OllamaVision, VisionError, make_vision

PLACEHOLDERS = {"short label", "name", "label", "pattern", "your own 2 to 4 word name for it", "the name of the pattern it uses"}

NEAR_DUPLICATE = 0.7  # word overlap (numbers ignored) above which two ideas count as one

IDEAS_PROMPT = """You are a short-form video strategist. The viewer is {niche}.
Below are reels the viewer studied, strongest first (outlier = views divided by that creator's usual views; higher is better).
Each line is one reel as JSON: code, author, views, outlier, format, hook type and text, call to action, why it worked, takeaway.

STATS ACROSS ALL STUDIED REELS:
{stats}

REELS:
{reels}

Return ONLY a JSON object with two keys, "patterns" and "ideas".

Each pattern has: "name" (your own 2 to 4 word name for it), "what" (the repeatable mechanism in one sentence),
"evidence" (list of reel codes from REELS that show it).

Each idea has: "title" (working title), "hook" (the first spoken sentence, under 15 words), "on_screen_hook" (different words
from the spoken hook, under 8 words), "format", "beats" (3 to 5 concrete steps of the video, each saying what is shown or said),
"cta", "pattern" (the name of the pattern it uses), "inspired_by" (reel codes from REELS), "why" (one sentence).

Example of ONE idea at the right level of detail, for a cooking creator (a different niche on purpose; do not reuse its topic):
{{"title": "The 9 euro dinner that beats takeout", "hook": "I spent 9 euros and this tastes better than my 30 euro takeout.",
 "on_screen_hook": "9 EUR vs 30 EUR", "format": "talking head with b-roll",
 "beats": ["Receipt on the counter, total 9 euros", "Cook in 3 fast cuts, one tip per cut",
           "Side by side bite of takeout vs homemade", "Verdict and the one ingredient that made it"],
 "cta": "comment RECIPE and I send the shopping list", "pattern": "price contrast", "inspired_by": ["CODE1"],
 "why": "A concrete number plus a head to head test makes people stay for the result."}}

Rules:
- {n_patterns} patterns that really repeat across the reels, each with the codes that show it.
- {count} ideas for the viewer's niche, each a different angle: vary the format, the hook type and the payoff. No two ideas share a title or hook.
- Every idea names something concrete: a real tool, a number, a before and after, or a result. No vague beats like "watch this" or "learn how".
- Reuse the mechanism that worked (hook shape, structure, call to action), never the creator's topic or wording.
- inspired_by and evidence may only contain codes from the REELS list.
- Plain language, no hashtags, no emojis."""


def load_studies(out_root: Path) -> list[dict]:
    out = []
    for f in sorted((out_root / "study").glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("code") and isinstance(rec.get("analysis"), dict) and not rec["analysis"].get("parse_error"):
            out.append(rec)
    return out


def outliers(out_root: Path) -> dict[str, float]:
    """Outlier ratios recorded by `scout`, latest run wins."""
    ratios: dict[str, float] = {}
    f = out_root / "scout.jsonl"
    if not f.exists():
        return ratios
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            for r in json.loads(line).get("ranked") or []:
                if r.get("outlier"):
                    ratios[r["code"]] = r["outlier"]
        except (ValueError, AttributeError):
            pass
    return ratios


def digest(rec: dict, outlier: float | None = None) -> dict:
    """One study cut down to what the strategist needs (keeps the prompt small)."""
    a = rec["analysis"]
    hook = a.get("hook") if isinstance(a.get("hook"), dict) else {}
    cta = a.get("cta") if isinstance(a.get("cta"), dict) else {}
    tk = a.get("takeaway") if isinstance(a.get("takeaway"), dict) else {}
    g = rec.get("giveaway") or {}
    post = rec.get("post") or {}
    d = {"code": rec["code"], "author": post.get("author"), "views": post.get("views"), "outlier": outlier,
         "format": a.get("format"), "hook_type": hook.get("type"),
         "hook": (hook.get("spoken") or hook.get("on_screen") or "")[:160],
         "cta": ("comment " + g["keyword"]) if g.get("offered") and g.get("keyword") else (cta.get("spoken") or cta.get("on_screen") or "")[:100],
         "why": (a.get("why_it_works") or "")[:200], "takeaway": (tk.get("insight") or "")[:160],
         "cuts_per_10s": (rec.get("facts") or {}).get("cuts_per_10s")}
    return {k: v for k, v in d.items() if v not in (None, "")}


def strength(d: dict) -> tuple:
    return (d.get("outlier") or 0, d.get("views") or 0)


def stats(ds: list[dict]) -> dict:
    def top(key, n=5):
        return [f"{k} ({v})" for k, v in Counter(d[key] for d in ds if d.get(key)).most_common(n)]
    cuts = [d["cuts_per_10s"] for d in ds if isinstance(d.get("cuts_per_10s"), (int, float))]
    return {"reels": len(ds), "formats": top("format"), "hook_types": top("hook_type"),
            "comment_giveaways": sum(1 for d in ds if str(d.get("cta", "")).startswith("comment ")),
            "median_cuts_per_10s": round(statistics.median(cuts), 1) if cuts else None}


def clean(res, codes: set[str]) -> dict:
    """Keep well-formed patterns and ideas; drop source codes the model made up."""
    res = res if isinstance(res, dict) else {}

    def real(xs):
        return [c for c in (xs or []) if isinstance(c, str) and c in codes]

    def key(s) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
    pats = []
    for p in res.get("patterns") or []:
        if not isinstance(p, dict) or not (p.get("name") or p.get("what")):
            continue
        if not p.get("name") or key(p.get("name")) in PLACEHOLDERS:  # small models sometimes echo the field description
            p["name"] = " ".join(str(p.get("what", "")).split()[:4])
        pats.append({**p, "evidence": real(p.get("evidence"))})
    ideas: list[dict] = []
    kept: list[set[str]] = []
    hooks: set[str] = set()
    for i in res.get("ideas") or []:
        if not isinstance(i, dict) or not (i.get("title") and i.get("hook")):
            continue
        words = set(re.sub(r"\d+", " ", key(f"{i['title']} {i['hook']}")).split())
        if key(i["hook"]) in hooks or any(len(words & w) / max(1, len(words | w)) >= NEAR_DUPLICATE for w in kept):
            continue  # same idea with another number ("in 10 seconds" / "in 20 seconds")
        kept.append(words)
        hooks.add(key(i["hook"]))
        if key(i.get("pattern")) in PLACEHOLDERS:
            i.pop("pattern", None)
        i["inspired_by"] = real(i.get("inspired_by"))
        i["beats"] = [b for b in (i.get("beats") or []) if isinstance(b, str)]
        ideas.append(i)
    return {"patterns": pats, "ideas": ideas}


def load_model(backend: str, model: str, temperature: float):
    """The text model. On Ollama, thinking is switched off (slow, and not needed for this)."""
    if backend == "ollama":
        m = OllamaVision(model)
        m.temperature = temperature
        if "thinking" in m.capabilities():
            m.think = False
        return m
    m = make_vision(backend, model)
    if hasattr(m, "temperature"):
        m.temperature = temperature
    return m


def generate(out_root: Path, vision, niche: str, count: int, limit: int, model_name: str) -> dict:
    studies = load_studies(out_root)
    if not studies:
        raise SystemExit(f"no studies in {out_root / 'study'} yet; run `reel-scroller study` first")
    ratios = outliers(out_root)
    ds = sorted((digest(r, ratios.get(r["code"])) for r in studies), key=strength, reverse=True)
    chosen = ds[:limit]
    st = stats(ds)
    ask = count + (count + 1) // 2  # small models repeat themselves; ask for more and keep the distinct ones
    prompt = IDEAS_PROMPT.format(niche=niche, stats=json.dumps(st, ensure_ascii=False),
                                 reels="\n".join(json.dumps(d, ensure_ascii=False) for d in chosen),
                                 n_patterns=min(6, max(2, len(chosen) // 2)), count=ask)
    t0 = time.time()
    res = vision.ask(prompt, None, max_tokens=400 + 260 * ask)
    out = clean(res, {d["code"] for d in chosen})
    out["ideas"] = out["ideas"][:count]
    out.update(generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"), model=model_name, niche=niche, seconds=round(time.time() - t0, 1),
               stats=st, sources=chosen)
    if isinstance(res, dict) and res.get("parse_error"):
        out["error"] = "the model did not return valid JSON; try again or use a larger model"
    return out


def write(out_root: Path, result: dict) -> Path:
    d = out_root / "ideas"
    d.mkdir(parents=True, exist_ok=True)
    body = json.dumps(result, ensure_ascii=False, indent=1)
    path = d / f"ideas-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(body, encoding="utf-8")
    tmp = d / "latest.json.tmp"
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(d / "latest.json")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reel-scroller ideas",
                                 description="Turn your studied reels into patterns and new reel ideas for your niche. Local model, no cost.")
    ap.add_argument("--out-root", default="./reel-scroller-out")
    ap.add_argument("--niche", default=os.environ.get("REEL_SCROLLER_CONTEXT", ""),
                    help='who the ideas are for, e.g. "an AI and video-editing creator" (env REEL_SCROLLER_CONTEXT)')
    ap.add_argument("--count", type=int, default=10, help="how many ideas (default 10)")
    ap.add_argument("--limit", type=int, default=40, help="use the strongest N studied reels (default 40)")
    ap.add_argument("--backend", choices=list(_model.BACKENDS), help="model backend (default: same as study)")
    ap.add_argument("--model", default=os.environ.get("REEL_SCROLLER_IDEAS_MODEL"),
                    help="text model, e.g. qwen3:8b (env REEL_SCROLLER_IDEAS_MODEL; default: the study vision model)")
    ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args(argv)
    out_root = Path(a.out_root).expanduser()
    c = _model.configure(None, None, a.backend)
    name = a.model or c.vlm
    niche = a.niche or "a creator who wants to learn from these reels and make their own"
    try:
        vision = load_model(c.backend, name, a.temperature)
    except VisionError as e:
        print(e)
        return 3
    print(f"ideas: {c.backend} {name}, niche: {niche}", flush=True)
    try:
        result = generate(out_root, vision, niche, a.count, a.limit, name)
    except VisionError as e:
        print(f"stopped: {e}")
        return 6
    path = write(out_root, result)
    for p in result["patterns"]:
        print(f"\n[pattern] {p['name']}: {p.get('what', '')}  ({', '.join(p['evidence']) or 'no source'})")
    for n, i in enumerate(result["ideas"], 1):
        print(f"\n{n}. {i['title']}\n   hook: {i['hook']}\n   from: {', '.join(i['inspired_by']) or '-'}")
    if result.get("error"):
        print(f"\n{result['error']}")
    print(f"\n{len(result['ideas'])} ideas in {result['seconds']}s, saved to {path}. Open them with: reel-scroller dash --out-root {out_root}")
    return 0 if result["ideas"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
