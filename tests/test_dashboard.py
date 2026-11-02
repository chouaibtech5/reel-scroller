"""Live state writer and the local dashboard server (no models, no network beyond 127.0.0.1)."""
import json
import threading
import urllib.error
import urllib.request

import pytest
from PIL import Image

from reel_scroller import dashboard, live


@pytest.fixture(autouse=True)
def no_live():
    live._LIVE = None
    yield
    live._LIVE = None


def jpg(path, color="red"):
    Image.new("RGB", (720, 1280), color).save(path)
    return path


def test_live_is_a_noop_before_begin(tmp_path):
    assert live.active() is False
    live.event("x")
    live.item_start({"code": "A"}, {})
    live.item_done("A", 1.0)
    assert not (tmp_path / "_live").exists()


def test_live_records_a_run(tmp_path):
    live.begin(tmp_path, "local", 2, {"vlm": "qwen2.5vl:7b"})
    live.item_start({"code": "R1", "url": "local:r1.mp4", "kind": "reel"}, {"author": "me"})
    live.stage("extracting frames", duration_s=12.0, cuts=3)
    live.frames([(0.0, jpg(tmp_path / "a.jpg")), (1.0, jpg(tmp_path / "b.jpg", "blue"))])
    live.transcript({"language": "en", "words": [{"w": "hi", "s": 0.1, "e": 0.3}]})
    st = json.loads((tmp_path / "_live" / "state.json").read_text(encoding="utf-8"))
    cur = st["current"]
    assert cur["code"] == "R1" and cur["author"] == "me" and cur["stage"] == "extracting frames"
    assert cur["facts"] == {"duration_s": 12.0, "cuts": 3}
    assert [f["label"] for f in cur["frames"]] == ["0.0s", "1.0s"]
    with Image.open(tmp_path / "_live" / "frames" / cur["frames"][1]["file"]) as im:
        assert max(im.size) <= 480
    assert cur["transcript"]["words"][0]["w"] == "hi"

    live.item_done("R1", 40.0)
    live.item_start({"code": "R2"}, {})
    live.item_failed("R2", "boom")
    live.end()
    st = json.loads((tmp_path / "_live" / "state.json").read_text(encoding="utf-8"))
    assert st["run"]["done"] == 1 and st["run"]["failed"] == 1 and st["run"]["avg_s"] == 40.0
    assert st["last"]["code"] == "R1" and st["current"] is None and st["run"]["finished_at"]
    tags = [json.loads(l)["tag"] for l in (tmp_path / "_live" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert tags[0] == "run" and "done" in tags and "error" in tags and tags[-1] == "run"


def test_live_never_raises(tmp_path):
    live.begin(tmp_path, "local", 1, {})
    live.item_start({"code": "R"}, {})
    live.frames([(0.0, tmp_path / "missing.jpg")])  # unreadable image: swallowed
    live.item_done("R", 1.0)


def test_carousel_frames_get_slide_labels(tmp_path):
    live.begin(tmp_path, "apify", 1, {})
    live.item_start({"code": "C"}, {})
    live.frames([(1, jpg(tmp_path / "s1.jpg")), (2, jpg(tmp_path / "s2.jpg"))])
    st = json.loads((tmp_path / "_live" / "state.json").read_text(encoding="utf-8"))
    assert [f["label"] for f in st["current"]["frames"]] == ["slide 1", "slide 2"]


def test_tail_events(tmp_path):
    f = tmp_path / "e.jsonl"
    f.write_text("".join(json.dumps({"msg": str(i)}) + "\n" for i in range(5)) + "not json\n", encoding="utf-8")
    ev, n = dashboard.tail_events(f, 3)
    assert n == 6 and [e["msg"] for e in ev] == ["3", "4"]
    assert dashboard.tail_events(f, 99)[0][0]["msg"] == "0"  # a new run restarted the file
    assert dashboard.tail_events(tmp_path / "none.jsonl", 0) == ([], 0)


STUDY = {
    "code": "ABC_1", "url": "https://www.instagram.com/reel/ABC_1/", "kind": "reel", "keyframe": "ABC_1.jpg",
    "collections": ["Hooks"], "post": {"author": "creator", "views": 12000}, "analyzed_at": "2026-11-02T10:00:00",
    "facts": {"duration_s": 20.0, "cuts_per_10s": 3.5},
    "analysis": {"format": "talking head", "hook": {"type": "bold claim", "on_screen": "My AI watched 800 reels"},
                 "why_it_works": "big number"},
    "giveaway": {"offered": True, "keyword": "REELS"},
}


@pytest.fixture
def server(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    (study / "ABC_1.json").write_text(json.dumps(STUDY), encoding="utf-8")
    jpg(study / "ABC_1.jpg")
    (tmp_path / "secret.txt").write_text("no", encoding="utf-8")
    live.begin(tmp_path, "local", 1, {"vlm": "m"})
    live.item_start({"code": "ABC_1"}, {})
    live.frames([(0.0, jpg(tmp_path / "f.jpg"))])
    srv = dashboard.make_server(tmp_path, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def status(url):
    try:
        return get(url)[0]
    except urllib.error.HTTPError as e:
        return e.code


def test_server_routes(server):
    s, ctype, body = get(server + "/")
    assert s == 200 and ctype.startswith("text/html") and b"reel-scroller" in body

    d = json.loads(get(server + "/api/live?since=0")[2])
    assert d["state"]["current"]["code"] == "ABC_1" and d["next"] == 1 and isinstance(d["sys"], dict)
    assert json.loads(get(server + f"/api/live?since={d['next']}")[2])["events"] == []

    reels = json.loads(get(server + "/api/library")[2])["reels"]
    assert reels == [dashboard.card(STUDY)]
    assert reels[0]["giveaway"] == "REELS" and reels[0]["hook_text"] == "My AI watched 800 reels"

    assert json.loads(get(server + "/api/reel/ABC_1")[2])["post"]["author"] == "creator"
    assert get(server + "/img/study/ABC_1.jpg")[1] == "image/jpeg"
    frame = d["state"]["current"]["frames"][0]["file"]
    assert get(server + "/img/live/" + frame)[1] == "image/jpeg"


def test_server_refuses_paths_outside_its_folders(server):
    assert status(server + "/api/reel/..%2Fsecret") == 404
    assert status(server + "/api/reel/nope") == 404
    assert status(server + "/img/study/../secret.txt") == 404
    assert status(server + "/img/study/..%2F..%2Fsecret.txt") == 404
    assert status(server + "/img/live/../../secret.txt") == 404
    assert status(server + "/elsewhere") == 404


def test_card_tolerates_a_parse_error_analysis():
    c = dashboard.card({"code": "X", "analysis": {"parse_error": True, "raw": "?"}})
    assert c["code"] == "X" and c["format"] is None and c["giveaway"] is None
