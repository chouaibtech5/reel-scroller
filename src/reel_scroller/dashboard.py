"""Local dashboard: one page that shows a run live and browses everything analyzed so far.

Standard library only. Binds to 127.0.0.1 and reads <out-root>/_live and <out-root>/study; it never
writes, never starts a run and never calls a model or the network.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

CODE_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
EVENTS_TAIL = 400
SYS_TTL_S = 2.0


def page() -> bytes:
    return resources.files("reel_scroller").joinpath("dashboard.html").read_bytes()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tail_events(path: Path, since: int) -> tuple[list[dict], int]:
    """Events after line `since` (at most the last EVENTS_TAIL), and the new line count."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], 0
    start = max(since, len(lines) - EVENTS_TAIL) if since <= len(lines) else max(0, len(lines) - EVENTS_TAIL)
    out = []
    for l in lines[start:]:
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out, len(lines)


def card(rec: dict) -> dict:
    """The few fields the library grid needs from one study JSON."""
    a = rec.get("analysis") if isinstance(rec.get("analysis"), dict) else {}
    hook = a.get("hook") if isinstance(a.get("hook"), dict) else {}
    post = rec.get("post") or {}
    facts = rec.get("facts") or {}
    give = rec.get("giveaway") or {}
    return {
        "code": rec.get("code"), "url": rec.get("url"), "kind": rec.get("kind"), "keyframe": rec.get("keyframe"),
        "collections": rec.get("collections") or [], "author": post.get("author"), "views": post.get("views"),
        "likes": post.get("likes"), "post_date": post.get("post_date"), "format": a.get("format"),
        "hook_type": hook.get("type"), "hook_text": hook.get("on_screen") or hook.get("spoken") or "",
        "why_it_works": a.get("why_it_works"), "duration_s": facts.get("duration_s"), "cuts_per_10s": facts.get("cuts_per_10s"),
        "giveaway": give.get("keyword") if give.get("offered") else None, "analyzed_at": rec.get("analyzed_at"),
        "wall_s": rec.get("wall_s"),
    }


class Library:
    """study/*.json as cards, re-read only when a file changed."""

    def __init__(self, study: Path):
        self.study = study
        self.cache: dict[str, tuple[float, dict]] = {}
        self.lock = threading.Lock()

    def cards(self) -> list[dict]:
        with self.lock:
            seen = set()
            for f in self.study.glob("*.json"):
                code = f.stem
                seen.add(code)
                try:
                    m = f.stat().st_mtime
                except OSError:
                    continue
                if code not in self.cache or self.cache[code][0] != m:
                    rec = read_json(f)
                    if isinstance(rec, dict) and rec.get("code"):
                        self.cache[code] = (m, card(rec))
            for gone in set(self.cache) - seen:
                del self.cache[gone]
            out = [c for _, c in self.cache.values()]
        return sorted(out, key=lambda c: c.get("analyzed_at") or "", reverse=True)


_sys_cache: dict = {"at": 0.0, "data": {}}


def system_stats() -> dict:
    """RAM, and NVIDIA GPU use when nvidia-smi answers. Cached for a couple of seconds."""
    if time.time() - _sys_cache["at"] < SYS_TTL_S:
        return _sys_cache["data"]
    data: dict = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        data["ram_used_gb"] = round((vm.total - vm.available) / 1024**3, 1)
        data["ram_total_gb"] = round(vm.total / 1024**3, 1)
        data["cpu_pct"] = psutil.cpu_percent(interval=None)
    except Exception:  # noqa: BLE001
        pass
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            p = subprocess.run([exe, "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3)
            if p.returncode == 0 and p.stdout.strip():
                u, used, tot = [x.strip() for x in p.stdout.splitlines()[0].split(",")]
                data.update(gpu_pct=int(u), vram_used_gb=round(int(used) / 1024, 1), vram_total_gb=round(int(tot) / 1024, 1))
        except Exception:  # noqa: BLE001
            pass
    _sys_cache.update(at=time.time(), data=data)
    return data


def safe_file(base: Path, rel: str) -> Path | None:
    """`base/rel` if it stays inside base and exists, else None."""
    try:
        p = (base / rel).resolve()
        p.relative_to(base.resolve())
    except (ValueError, OSError):
        return None
    return p if p.is_file() else None


def make_handler(out_root: Path):
    out_root = Path(out_root)
    live_dir, study = out_root / "_live", out_root / "study"
    library = Library(study)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet: the run already prints its own log
            pass

        def send(self, body: bytes, ctype: str, status: int = 200, cache: str = "no-store") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj, status: int = 200) -> None:
            self.send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            path, qs = u.path, parse_qs(u.query)
            if path in ("/", "/index.html"):
                return self.send(page(), "text/html; charset=utf-8")
            if path == "/api/live":
                since = int((qs.get("since") or ["0"])[0] or 0)
                events, n = tail_events(live_dir / "events.jsonl", since)
                return self.json({"state": read_json(live_dir / "state.json"), "events": events, "next": n,
                                  "sys": system_stats(), "now": time.time()})
            if path == "/api/library":
                return self.json({"reels": library.cards(), "out_root": str(out_root.resolve())})
            m = re.fullmatch(r"/api/reel/([^/]+)", path)
            if m:
                code = m.group(1)
                rec = read_json(study / f"{code}.json") if CODE_RE.match(code) else None
                return self.json(rec) if isinstance(rec, dict) else self.json({"error": "not found"}, 404)
            for prefix, base in (("/img/study/", study), ("/img/live/", live_dir / "frames")):
                if path.startswith(prefix):
                    f = safe_file(base, path[len(prefix):])
                    if f and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        ctype = "image/png" if f.suffix.lower() == ".png" else "image/jpeg"
                        return self.send(f.read_bytes(), ctype, cache="max-age=60")
                    return self.send(b"not found", "text/plain", 404)
            return self.send(b"not found", "text/plain", 404)

    return Handler


def make_server(out_root: Path, port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(out_root))
    srv.daemon_threads = True
    return srv


def serve_in_background(out_root: Path, port: int, open_browser: bool = True) -> ThreadingHTTPServer | None:
    """Start the dashboard on a daemon thread (used by `study --dash`). Returns None if the port is taken."""
    try:
        srv = make_server(out_root, port)
    except OSError as e:
        print(f"dashboard not started: port {port} is busy ({e}). Use --port, or open the one already running.", flush=True)
        return None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    print(f"dashboard: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    return srv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reel-scroller dash",
                                 description="Local dashboard: watch a run live and browse analyzed reels. Read only, localhost only.")
    ap.add_argument("--out-root", default="./reel-scroller-out", help="the --out-root you gave to `study`")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="do not open a browser tab")
    a = ap.parse_args(argv)
    out_root = Path(a.out_root).expanduser()
    if not (out_root / "study").is_dir() and not (out_root / "_live").is_dir():
        print(f"nothing analyzed yet in {out_root} (no study/ or _live/). The dashboard will fill in once a run starts.")
    try:
        srv = make_server(out_root, a.port)
    except OSError as e:
        print(f"port {a.port} is busy ({e}); try --port {a.port + 1}")
        return 2
    url = f"http://127.0.0.1:{a.port}"
    print(f"dashboard: {url}  (Ctrl+C to stop)", flush=True)
    if not a.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0
