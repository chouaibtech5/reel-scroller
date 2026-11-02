"""Live run state for the dashboard, written next to the study output:

  <out-root>/_live/state.json     run totals, current reel, last finished reel
  <out-root>/_live/events.jsonl   one line per log event
  <out-root>/_live/frames/<code>/ small thumbnails of the frames being analyzed

Every function is a no-op until begin() is called, and never raises: a broken
dashboard must not stop an overnight run.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

THUMB = (270, 480)
MAX_TRANSCRIPT_WORDS = 400

_LIVE: _Live | None = None


class _Live:
    def __init__(self, out_root: Path, mode: str, total: int, setup: dict):
        self.dir = Path(out_root) / "_live"
        shutil.rmtree(self.dir, ignore_errors=True)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.durations: list[float] = []
        self.state = {
            "run": {"started_at": time.time(), "finished_at": None, "mode": mode, "total": total, "done": 0,
                    "failed": 0, "avg_s": None, "setup": setup},
            "current": None,
            "last": None,
        }
        self._write()

    def _write(self) -> None:
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.dir / "state.json")

    def event(self, msg: str, tag: str = "") -> None:
        line = json.dumps({"t": time.time(), "tag": tag, "msg": msg}, ensure_ascii=False)
        with open(self.dir / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _safe(fn):
    def wrap(*a, **k):
        if _LIVE is None:
            return None
        try:
            with _LIVE.lock:
                return fn(_LIVE, *a, **k)
        except Exception:  # noqa: BLE001
            return None
    return wrap


def begin(out_root: Path, mode: str, total: int, setup: dict) -> None:
    global _LIVE
    try:
        _LIVE = _Live(out_root, mode, total, setup)
    except Exception:  # noqa: BLE001
        _LIVE = None


def active() -> bool:
    return _LIVE is not None


@_safe
def event(lv: _Live, msg: str, tag: str = "") -> None:
    lv.event(msg, tag)


@_safe
def item_start(lv: _Live, item: dict, meta: dict) -> None:
    prev = (lv.state.get("last") or {}).get("code")
    for d in (lv.dir / "frames").iterdir():  # keep only the previous reel's thumbnails
        if d.name != prev:
            shutil.rmtree(d, ignore_errors=True)
    lv.state["current"] = {"code": item["code"], "url": item.get("url"), "kind": item.get("kind"),
                           "collections": item.get("collections") or [], "author": (meta or {}).get("author"),
                           "started_at": time.time(), "stage": "starting", "frames": [], "transcript": None, "facts": {}}
    lv._write()
    lv.event(f"start {item['code']}", "run")


@_safe
def stage(lv: _Live, text: str, **facts) -> None:
    cur = lv.state.get("current")
    if cur is None:
        return
    cur["stage"] = text
    cur["facts"].update(facts)
    lv._write()


@_safe
def frames(lv: _Live, shots: list[tuple[float | int, Path]]) -> None:
    """Thumbnails of the frames (t seconds, or slide number) the model is about to see."""
    from PIL import Image

    cur = lv.state.get("current")
    if cur is None:
        return
    d = lv.dir / "frames" / cur["code"]
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for n, (t, p) in enumerate(shots):
        name = f"{n:02d}.jpg"
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail(THUMB)
            im.save(d / name, quality=80)
        label = f"slide {t}" if isinstance(t, int) else f"{t:.1f}s"  # JSON cannot tell 1.0 seconds from slide 1
        out.append({"t": t, "label": label, "file": f"{cur['code']}/{name}"})
    cur["frames"] = out
    lv._write()


@_safe
def transcript(lv: _Live, tr: dict) -> None:
    cur = lv.state.get("current")
    if cur is None:
        return
    cur["transcript"] = {"language": tr.get("language"), "words": (tr.get("words") or [])[:MAX_TRANSCRIPT_WORDS]}
    lv._write()


@_safe
def item_done(lv: _Live, code: str, secs: float) -> None:
    run = lv.state["run"]
    run["done"] += 1
    lv.durations.append(secs)
    run["avg_s"] = round(sum(lv.durations) / len(lv.durations), 1)
    cur = lv.state.get("current") or {}
    lv.state["last"] = {"code": code, "secs": round(secs, 1), "frames": cur.get("frames") or []}
    lv.state["current"] = None
    lv._write()
    lv.event(f"{code} analyzed in {secs:.0f}s", "done")


@_safe
def item_failed(lv: _Live, code: str, error: str) -> None:
    lv.state["run"]["failed"] += 1
    lv.state["current"] = None
    lv._write()
    lv.event(f"{code} FAILED: {error[:200]}", "error")


@_safe
def end(lv: _Live) -> None:
    lv.state["run"]["finished_at"] = time.time()
    lv.state["current"] = None
    lv._write()
    lv.event(f"run finished: {lv.state['run']['done']} analyzed, {lv.state['run']['failed']} failed", "run")
