"""Free ways to get reels, so Apify becomes optional:

- Meta Business Discovery (official Graph API): recent posts of any public Business or Creator account,
  with Reel view counts. Needs YOUR professional Instagram linked to a Facebook Page, a Meta app and a Page
  access token: META_IG_USER_ID and META_ACCESS_TOKEN. About 200 calls per hour, one call per creator.
- yt-dlp: downloads one public reel by its link, without logging in. Instagram sometimes refuses or slows
  anonymous downloads; callers fall back to Apify when a token is set.

Neither uses your browser cookies or logs in to Instagram.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GRAPH = "https://graph.facebook.com/v25.0"
MEDIA_FIELDS = "id,media_product_type,media_type,timestamp,view_count,like_count,comments_count,permalink,media_url,caption"
RATE_LIMIT_CODES = {4, 17, 32, 613}
YTDLP_TIMEOUT_S = 180


class MetaError(Exception):
    pass


class MetaRateLimited(MetaError):
    pass


class NotProfessional(MetaError):
    """Personal, age-gated or missing account: Business Discovery cannot see it."""


def meta_config(env=None) -> tuple[str, str] | None:
    env = os.environ if env is None else env
    uid, tok = (env.get("META_IG_USER_ID") or "").strip(), (env.get("META_ACCESS_TOKEN") or "").strip()
    return (uid, tok) if uid and tok else None


def shortcode(permalink: str) -> str | None:
    parts = [p for p in urllib.parse.urlparse(permalink or "").path.split("/") if p]
    for i, p in enumerate(parts[:-1]):
        if p in ("reel", "reels", "p", "tv"):
            return parts[i + 1]
    return None


def meta_to_item(m: dict, username: str) -> dict | None:
    """One Graph API media object in the shape scout already ranks (Apify's field names)."""
    code = shortcode(m.get("permalink") or "")
    if not code:
        return None
    video = m.get("media_product_type") == "REELS" or m.get("media_type") == "VIDEO"
    return {"shortCode": code, "ownerUsername": username, "type": "Video" if video else m.get("media_type", "Image").title(),
            "productType": "clips" if m.get("media_product_type") == "REELS" else None,
            "videoPlayCount": m.get("view_count"), "likesCount": m.get("like_count"), "commentsCount": m.get("comments_count"),
            "timestamp": m.get("timestamp"), "caption": m.get("caption"), "url": m.get("permalink"), "mediaUrl": m.get("media_url")}


def meta_discover(handle: str, ig_user_id: str, token: str, limit: int = 12, timeout: float = 30) -> list[dict]:
    """Recent posts of @handle through Business Discovery. The token goes in a header, never in the URL."""
    fields = f"business_discovery.username({handle}){{username,media.limit({min(limit, 50)}){{{MEDIA_FIELDS}}}}}"
    url = f"{GRAPH}/{ig_user_id}?" + urllib.parse.urlencode({"fields": fields})
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "User-Agent": "reel-scroller"})
    for attempt in (1, 2):  # graph.facebook.com connections sometimes time out; one retry, never more
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read().decode("utf-8") or "{}")
            break
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode("utf-8") or "{}")
            except ValueError:
                body = {"error": {"message": f"HTTP {e.code}"}}
            break
        except (urllib.error.URLError, OSError):
            if attempt == 2:
                raise
    d = body.get("business_discovery")
    if d:
        name = d.get("username") or handle
        return [x for x in (meta_to_item(m, name) for m in (d.get("media") or {}).get("data") or []) if x]
    err = body.get("error") or {}
    msg, code = err.get("message") or "no answer", err.get("code")
    if code in RATE_LIMIT_CODES or "limit reached" in msg.lower():
        raise MetaRateLimited(f"Meta's hourly limit is reached ({msg}); try again in an hour")
    if code == 190:
        raise MetaError("Meta access token expired or was revoked; make a new one and update META_ACCESS_TOKEN")
    if code == 110 or any(s in msg.lower() for s in ("invalid user id", "cannot find user", "does not exist", "not a business")):
        raise NotProfessional(f"@{handle} is not a public Business/Creator account, so Meta cannot read it")
    raise MetaError(f"Meta API: {msg}")


def ytdlp_args(url: str, dest: Path) -> list[str]:
    return [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings", "--socket-timeout", "45", "--retries", "1",
            "-f", "b[ext=mp4]/bv*+ba/b", "--merge-output-format", "mp4", "--write-info-json",
            "-o", str(dest / "%(id)s.%(ext)s"), "--", url]  # "--": a URL can never be read as an option


def ytdlp_meta(info: dict) -> dict:
    """yt-dlp's info JSON in the shape study's Apify path produces (study.apify_meta)."""
    def pos(v):
        return v if isinstance(v, (int, float)) and v >= 0 else None
    date = info.get("upload_date") or ""
    return {"caption": (info.get("description") or "")[:2000], "author": info.get("channel") or info.get("uploader_id") or info.get("uploader"),
            "author_id": info.get("channel_id") or info.get("uploader_id"), "likes": pos(info.get("like_count")),
            "views": pos(info.get("view_count")), "comments": pos(info.get("comment_count")),
            "post_date": f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) == 8 else None,
            "duration_s": info.get("duration"), "music_info": None}


def ytdlp_fetch(url: str, dest: Path, timeout: float = YTDLP_TIMEOUT_S) -> tuple[Path, dict]:
    """Download one public reel to dest. Returns (mp4 path, meta). Raises RuntimeError with yt-dlp's reason."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        p = subprocess.run(ytdlp_args(url, dest), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"yt-dlp timed out after {timeout:.0f}s") from e
    videos = sorted(f for f in dest.iterdir() if f.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv"))
    if p.returncode != 0 or not videos:
        reason = (p.stderr or p.stdout).strip().splitlines()
        raise RuntimeError(f"yt-dlp: {reason[-1][:240] if reason else 'no video downloaded'}")
    info_files = sorted(dest.glob("*.info.json"))
    info = json.loads(info_files[0].read_text(encoding="utf-8")) if info_files else {}
    for f in info_files:
        f.unlink(missing_ok=True)
    return videos[0], ytdlp_meta(info)
