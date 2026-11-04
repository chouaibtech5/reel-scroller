"""Free sources (Meta Business Discovery, yt-dlp) and how study/scout route between them and Apify.
No network: urlopen and subprocess are faked."""
import io
import json
import urllib.error
import urllib.parse

import pytest

from reel_scroller import scout, sources
from reel_scroller import study as rs


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("META_IG_USER_ID", "META_ACCESS_TOKEN", "APIFY_TOKEN"):
        monkeypatch.delenv(k, raising=False)


def media(code, views, product="REELS"):
    return {"id": code, "media_product_type": product, "media_type": "VIDEO" if product == "REELS" else "IMAGE",
            "view_count": views, "like_count": 5, "comments_count": 1, "timestamp": "2026-10-01T00:00:00+0000",
            "permalink": f"https://www.instagram.com/{'reel' if product == 'REELS' else 'p'}/{code}/", "caption": "hi",
            "media_url": "https://cdn/x.mp4"}


class FakeOpen:
    def __init__(self, *answers):
        self.answers, self.requests = list(answers), []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        status, body = self.answers.pop(0)
        raw = json.dumps(body).encode()
        if status != 200:
            raise urllib.error.HTTPError(req.full_url, status, "err", {}, io.BytesIO(raw))
        return io.BytesIO(raw)


def test_meta_config_and_shortcode():
    assert sources.meta_config({}) is None
    assert sources.meta_config({"META_IG_USER_ID": "1", "META_ACCESS_TOKEN": "t"}) == ("1", "t")
    assert sources.shortcode("https://www.instagram.com/reel/AbC_1/?igsh=x") == "AbC_1"
    assert sources.shortcode("https://www.instagram.com/p/Xy/") == "Xy"
    assert sources.shortcode("https://example.com/") is None


def test_meta_discover_maps_posts_and_keeps_the_token_out_of_the_url(monkeypatch):
    fake = FakeOpen((200, {"business_discovery": {"username": "Creator", "media": {"data": [media("R1", 900), media("P1", None, "FEED")]}}}))
    monkeypatch.setattr(sources.urllib.request, "urlopen", fake)
    items = sources.meta_discover("creator", "123", "SECRET", limit=12)
    req = fake.requests[0]
    assert "SECRET" not in req.full_url and req.get_header("Authorization") == "Bearer SECRET"
    assert "business_discovery.username(creator)" in urllib.parse.unquote(req.full_url) and "media.limit(12)" in urllib.parse.unquote(req.full_url)
    assert items[0] == {"shortCode": "R1", "ownerUsername": "Creator", "type": "Video", "productType": "clips", "videoPlayCount": 900,
                        "likesCount": 5, "commentsCount": 1, "timestamp": "2026-10-01T00:00:00+0000", "caption": "hi",
                        "url": "https://www.instagram.com/reel/R1/", "mediaUrl": "https://cdn/x.mp4"}
    assert items[1]["type"] == "Image"
    assert [r["code"] for r in scout.rank(items)] == ["R1"]  # scout ranks Meta items like Apify ones


@pytest.mark.parametrize("code,msg,exc", [
    (4, "Application request limit reached", sources.MetaRateLimited),
    (110, "Invalid user id", sources.NotProfessional),
    (190, "Error validating access token", sources.MetaError),
    (100, "Something else", sources.MetaError),
])
def test_meta_discover_errors(monkeypatch, code, msg, exc):
    monkeypatch.setattr(sources.urllib.request, "urlopen", FakeOpen((400, {"error": {"code": code, "message": msg}})))
    with pytest.raises(exc):
        sources.meta_discover("x", "1", "t")


def test_ytdlp_args_and_meta(tmp_path):
    args = sources.ytdlp_args("https://www.instagram.com/reel/A/", tmp_path)
    assert args[1:3] == ["-m", "yt_dlp"] and args[-2:] == ["--", "https://www.instagram.com/reel/A/"]
    assert "--cookies" not in args and "--cookies-from-browser" not in args
    m = sources.ytdlp_meta({"description": "cap", "channel": "maker", "like_count": 7, "view_count": -1, "upload_date": "20261002", "duration": 21.5})
    assert m == {"caption": "cap", "author": "maker", "author_id": None, "likes": 7, "views": None, "comments": None,
                 "post_date": "2026-10-02", "duration_s": 21.5, "music_info": None}


def fake_ytdlp(monkeypatch, ok_codes):
    def run(args, **kw):
        url, dest = args[-1], __import__("pathlib").Path(args[args.index("-o") + 1]).parent
        code = sources.shortcode(url)

        class P:
            returncode, stdout, stderr = 0, "", ""
        if code in ok_codes:
            (dest / f"{code}.mp4").write_bytes(b"v" * 2048)
            (dest / f"{code}.info.json").write_text(json.dumps({"channel": "maker", "view_count": 50}), encoding="utf-8")
        else:
            P.returncode, P.stderr = 1, "ERROR: [Instagram] X: Requested content is not available, rate-limit reached or login required"
        return P()
    monkeypatch.setattr(sources.subprocess, "run", run)


def test_ytdlp_fetch(tmp_path, monkeypatch):
    fake_ytdlp(monkeypatch, {"OK"})
    video, meta = sources.ytdlp_fetch("https://www.instagram.com/reel/OK/", tmp_path / "w")
    assert video.name == "OK.mp4" and meta["views"] == 50 and not list((tmp_path / "w").glob("*.info.json"))
    with pytest.raises(RuntimeError, match="login required"):
        sources.ytdlp_fetch("https://www.instagram.com/reel/NO/", tmp_path / "w2")


def batch(*codes):
    return [{"url": f"https://www.instagram.com/{'p' if c.startswith('C') else 'reel'}/{c}/", "code": c, "collections": [],
             "kind": "carousel" if c.startswith("C") else "reel"} for c in codes]


def test_fetch_batch_free_first_then_apify_only_for_the_rest(tmp_path, monkeypatch):
    fake_ytdlp(monkeypatch, {"OK"})
    sent = []

    def apify(b, *a):
        sent.append([x["code"] for x in b])
        return [{"item": x, "meta": {}, "work": tmp_path, "media": []} for x in b]
    monkeypatch.setattr(rs, "prepare_apify_batch", apify)
    units = rs.fetch_batch(batch("OK", "NO", "CAR"), tmp_path, tmp_path / "w", "auto", "TOKEN", 2.0)
    assert sent == [["NO", "CAR"]]
    assert [u["item"]["code"] for u in units] == ["OK", "NO", "CAR"] and units[0]["meta"]["author"] == "maker"


def test_fetch_batch_without_token_never_pays(tmp_path, monkeypatch):
    fake_ytdlp(monkeypatch, {"OK"})
    monkeypatch.setattr(rs, "prepare_apify_batch", lambda *a: (_ for _ in ()).throw(AssertionError("no token, no Apify")))
    units = rs.fetch_batch(batch("OK", "NO", "CAR"), tmp_path, tmp_path / "w", "auto", None, 2.0)
    assert "error" not in units[0]
    assert units[1]["error"].startswith("ytdlp: yt-dlp: ERROR") and "photo slides need Apify" in units[2]["error"]
    assert not (tmp_path / "w" / "NO").exists()  # failed downloads leave nothing behind


def test_study_run_without_apify_token_is_allowed_and_free(tmp_path, capsys):
    tsv = tmp_path / "urls.tsv"
    tsv.write_text("https://www.instagram.com/reel/A1/\n", encoding="utf-8")
    assert rs.main(["--input", str(tsv), "--out-root", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "yt-dlp (free)" in out and "(free)" in out.splitlines()[-1]
    assert rs.main(["--input", str(tsv), "--out-root", str(tmp_path / "o"), "--fetcher", "apify", "--run"]) == 2


def test_scout_uses_meta_then_apify_for_accounts_meta_cannot_see(tmp_path, monkeypatch, capsys):
    f = tmp_path / "c.txt"
    f.write_text("pro\npersonal\n", encoding="utf-8")
    monkeypatch.setenv("META_IG_USER_ID", "1")
    monkeypatch.setenv("META_ACCESS_TOKEN", "t")
    monkeypatch.setenv("APIFY_TOKEN", "TOKEN")

    def meta(h, uid, tok, limit):
        if h == "personal":
            raise sources.NotProfessional("not visible")
        return [sources.meta_to_item(media("M1", 100), h), sources.meta_to_item(media("M2", 900), h)]
    monkeypatch.setattr(sources, "meta_discover", meta)
    asked = {}

    def apify(urls, token, cap, **k):
        asked["urls"] = urls
        return [{"shortCode": "A1", "ownerUsername": "personal", "type": "Video", "videoPlayCount": 10}], {"run_id": "r", "usd": 0.01, "status": "SUCCEEDED", "items": 1}
    monkeypatch.setattr(rs, "apify_fetch", apify)
    out = tmp_path / "out"
    assert scout.main(["--creators", str(f), "--out-root", str(out), "--run", "--min-outlier", "0"]) == 0
    assert asked["urls"] == ["https://www.instagram.com/personal/"]
    assert {i["code"] for i in rs.parse_input(out / "urls.tsv")} == {"M1", "M2", "A1"}
    assert "free" in capsys.readouterr().out


def test_scout_meta_only_never_calls_apify(tmp_path, monkeypatch, capsys):
    f = tmp_path / "c.txt"
    f.write_text("personal\n", encoding="utf-8")
    monkeypatch.setenv("META_IG_USER_ID", "1")
    monkeypatch.setenv("META_ACCESS_TOKEN", "t")
    monkeypatch.setattr(sources, "meta_discover", lambda *a: (_ for _ in ()).throw(sources.NotProfessional("no")))
    monkeypatch.setattr(rs, "apify_fetch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no token, no Apify")))
    assert scout.main(["--creators", str(f), "--out-root", str(tmp_path / "o"), "--run"]) == 0
    assert "skipped 1 creator" in capsys.readouterr().out


def test_meta_discover_retries_one_dropped_connection(monkeypatch):
    calls = []

    def flaky(req, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError("WinError 10060 timed out")
        return io.BytesIO(json.dumps({"business_discovery": {"username": "c", "media": {"data": []}}}).encode())
    monkeypatch.setattr(sources.urllib.request, "urlopen", flaky)
    assert sources.meta_discover("c", "1", "t") == [] and len(calls) == 2
    monkeypatch.setattr(sources.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("down")))
    with pytest.raises(urllib.error.URLError):
        sources.meta_discover("c", "1", "t")


def test_fetch_batch_fills_views_from_scout(tmp_path, monkeypatch):
    fake_ytdlp(monkeypatch, {"OK"})
    monkeypatch.setattr(sources, "ytdlp_meta", lambda info: {"views": None})
    rs.append_jsonl(tmp_path / "scout.jsonl", {"ranked": [{"code": "OK", "views": 754}]})
    units = rs.fetch_batch(batch("OK"), tmp_path, tmp_path / "w", "ytdlp", None, 2.0)
    assert units[0]["meta"]["views"] == 754


def test_quota_stops_the_run_without_marking_reels_failed(tmp_path, monkeypatch, capsys):
    from reel_scroller.backends.vision import VisionQuotaError
    tsv = tmp_path / "urls.tsv"
    tsv.write_text("https://www.instagram.com/reel/A1/\nhttps://www.instagram.com/reel/A2/\n", encoding="utf-8")
    out = tmp_path / "o"
    monkeypatch.setattr(rs, "load_vision", lambda: object())
    monkeypatch.setattr(rs.media, "preflight", lambda: "")
    monkeypatch.setattr(rs, "fetch_batch", lambda b, *a: [{"item": x, "meta": {}, "work": tmp_path / x["code"], "media": []} for x in b])
    calls = []

    def analyze(vision, unit, out_root):
        calls.append(unit["item"]["code"])
        raise VisionQuotaError("gemini: the API quota is used up")
    monkeypatch.setattr(rs, "analyze_unit", analyze)
    assert rs.main(["--input", str(tsv), "--out-root", str(out), "--run", "--fetcher", "ytdlp"]) == 6
    assert calls == ["A1"]  # stopped at the first reel
    assert not (out / "study" / "_failed.jsonl").exists()  # nothing marked failed, so the next run retries them
    assert "quota is used up" in capsys.readouterr().out
