"""scout (creator watchlist) and ideas (studies to new reel ideas). No network, no models."""
import json

from reel_scroller import ideas, scout
from reel_scroller import study as rs


def item(code, owner, plays, kind="Video"):
    return {"shortCode": code, "ownerUsername": owner, "type": kind, "videoPlayCount": plays, "timestamp": "2026-10-01T00:00:00.000Z",
            "caption": f"caption {code}"}


def test_parse_creators(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("# my niche\n@ItsJakeBruce\nhttps://www.instagram.com/lokopic/\nlokopic\n\ninstagram.com/reel/ABC/\nbad name!\n", encoding="utf-8")
    assert scout.parse_creators(f) == ["itsjakebruce", "lokopic"]


def test_rank_scores_against_each_creators_median():
    items = [item("A1", "a", 1000), item("A2", "a", 1000), item("A3", "a", 9000), item("A4", "a", 500),
             item("B1", "b", 100), item("B2", "b", 300), item("PIC", "b", None, kind="Image")]
    r = scout.rank(items)
    assert [x["code"] for x in r] == ["A3", "B2", "A1", "A2", "A4", "B1"]  # photo dropped; 0.5x tie goes to more views
    top = r[0]
    assert top["outlier"] == 9.0 and top["median_views"] == 1000 and top["creator"] == "a"
    assert top["url"] == "https://www.instagram.com/reel/A3/"


def test_pick_skips_known_and_respects_bar_and_per_creator_cap():
    r = scout.rank([item("A1", "a", 100), item("A2", "a", 400), item("A3", "a", 500), item("A4", "a", 100), item("B1", "b", 50)])
    picks = scout.pick(r, done={"A3"}, min_outlier=1.5, top=1)
    assert [p["code"] for p in picks] == ["A2"]
    assert [p["code"] for p in scout.pick(r, set(), 0, 0)] == ["A3", "A2", "B1", "A1", "A4"]


def test_known_codes_and_append_tsv_round_trip(tmp_path):
    (tmp_path / "study").mkdir()
    (tmp_path / "study" / "DONE.json").write_text("{}", encoding="utf-8")
    rs.append_jsonl(tmp_path / "study" / "_failed.jsonl", {"code": "BAD"})
    r = scout.rank([item("NEW", "a", 100)])
    scout.append_tsv(tmp_path, r)
    assert scout.known_codes(tmp_path) == {"DONE", "BAD", "NEW"}
    queued = rs.parse_input(tmp_path / "urls.tsv")
    assert queued == [{"url": "https://www.instagram.com/reel/NEW/", "code": "NEW", "collections": ["creator:a"], "kind": "reel"}]


def test_scout_dry_run_spends_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("META_IG_USER_ID", raising=False)
    f = tmp_path / "c.txt"
    f.write_text("a\nb\n", encoding="utf-8")
    monkeypatch.setattr(rs, "apify_fetch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call Apify")))
    assert scout.main(["--creators", str(f), "--out-root", str(tmp_path / "out")]) == 0
    assert "dry run" in capsys.readouterr().out
    assert not (tmp_path / "out" / "urls.tsv").exists()


def test_scout_run_queues_outliers(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("META_IG_USER_ID", raising=False)
    f = tmp_path / "c.txt"
    f.write_text("a\n", encoding="utf-8")
    monkeypatch.setenv("APIFY_TOKEN", "TOKEN")
    seen = {}

    def fake(urls, token, cap, **k):
        seen.update(urls=urls, cap=cap, **k)
        return [item("X1", "a", 100), item("X2", "a", 100), item("X3", "a", 1000)], {"run_id": "r", "usd": 0.01, "status": "SUCCEEDED", "items": 3}
    monkeypatch.setattr(rs, "apify_fetch", fake)
    out = tmp_path / "out"
    assert scout.main(["--creators", str(f), "--out-root", str(out), "--run", "--per-creator", "5"]) == 0
    assert seen == {"urls": ["https://www.instagram.com/a/"], "cap": 1.0, "results_limit": 5}
    assert [i["code"] for i in rs.parse_input(out / "urls.tsv")] == ["X3"]
    assert "TOKEN" not in (out / "apify_cost.jsonl").read_text(encoding="utf-8")
    assert ideas.outliers(out) == {"X3": 10.0, "X1": 1.0, "X2": 1.0}


def study(code, fmt="talking head", views=100, give=None):
    return {"code": code, "post": {"author": "c", "views": views}, "facts": {"cuts_per_10s": 2.0},
            "giveaway": {"offered": bool(give), "keyword": give},
            "analysis": {"format": fmt, "hook": {"type": "bold claim", "spoken": f"hook {code}"}, "why_it_works": "clear",
                         "takeaway": {"insight": "do x"}}}


class FakeModel:
    def __init__(self, res):
        self.res, self.prompts = res, []

    def ask(self, prompt, image=None, max_tokens=500):
        self.prompts.append((prompt, image, max_tokens))
        return self.res


def write_studies(out, *recs):
    (out / "study").mkdir(parents=True)
    for r in recs:
        (out / "study" / f"{r['code']}.json").write_text(json.dumps(r), encoding="utf-8")
    (out / "study" / "BROKEN.json").write_text(json.dumps({"code": "BROKEN", "analysis": {"parse_error": True}}), encoding="utf-8")


def test_ideas_ranks_cleans_and_writes(tmp_path):
    write_studies(tmp_path, study("LOW", views=10), study("HIGH", views=5000, give="GUIDE"), study("MID", fmt="screen recording", views=900))
    res = {"patterns": [{"name": "comment bait", "what": "ask for a word", "evidence": ["HIGH", "MADEUP"]}, {"what": "no name"}],
           "ideas": [{"title": "T1", "hook": "Stop scrolling", "beats": ["a", 3, "b"], "inspired_by": ["HIGH", "FAKE"]},
                     {"title": "no hook"}, "junk"]}
    fake = FakeModel(res)
    out = ideas.generate(tmp_path, fake, "an editor", count=3, limit=2, model_name="m")
    prompt, image, tokens = fake.prompts[0]
    assert image is None and "an editor" in prompt and tokens == 400 + 260 * 5 and "5 ideas" in prompt  # 3 asked, 5 requested
    assert '"code": "HIGH"' in prompt and '"code": "MID"' in prompt and '"code": "LOW"' not in prompt  # strongest 2 only
    assert "BROKEN" not in prompt
    assert out["patterns"] == [{"name": "comment bait", "what": "ask for a word", "evidence": ["HIGH"]},
                               {"name": "no name", "what": "no name", "evidence": []}]  # unnamed: named from its description
    assert out["ideas"] == [{"title": "T1", "hook": "Stop scrolling", "beats": ["a", "b"], "inspired_by": ["HIGH"]}]
    assert out["stats"]["reels"] == 3 and out["stats"]["comment_giveaways"] == 1
    path = ideas.write(tmp_path, out)
    assert json.loads((tmp_path / "ideas" / "latest.json").read_text(encoding="utf-8")) == json.loads(path.read_text(encoding="utf-8"))


def test_ideas_prefers_scout_outliers_over_raw_views(tmp_path):
    write_studies(tmp_path, study("BIG", views=90000), study("SMALL", views=800))
    rs.append_jsonl(tmp_path / "scout.jsonl", {"ranked": [{"code": "SMALL", "outlier": 12.0}, {"code": "BIG", "outlier": 0.8}]})
    fake = FakeModel({"ideas": []})
    out = ideas.generate(tmp_path, fake, "x", 1, 1, "m")
    assert [s["code"] for s in out["sources"]] == ["SMALL"]


def test_ideas_reports_a_bad_model_answer(tmp_path):
    write_studies(tmp_path, study("A"))
    out = ideas.generate(tmp_path, FakeModel({"parse_error": True, "raw": "?"}), "x", 1, 5, "m")
    assert out["ideas"] == [] and "valid JSON" in out["error"]


def test_clean_drops_duplicates_and_echoed_placeholders():
    res = {"patterns": [{"name": "short label", "what": "bold number in the first second", "evidence": ["A"]}],
           "ideas": [{"title": "AI edits like a pro", "hook": "h1", "pattern": "short label"},
                     {"title": "AI edits like a PRO!", "hook": "h2"},
                     {"title": "Other", "hook": "H1"},
                     {"title": "Third", "hook": "h3", "pattern": "price contrast"}]}
    out = ideas.clean(res, {"A"})
    assert out["patterns"][0]["name"] == "bold number in the"
    assert [i["title"] for i in out["ideas"]] == ["AI edits like a pro", "Third"]
    assert "pattern" not in out["ideas"][0] and out["ideas"][1]["pattern"] == "price contrast"


def test_clean_drops_ideas_that_only_change_a_number():
    res = {"ideas": [{"title": "AI edits my reel in 10 seconds", "hook": "AI edits my reel in 10 seconds, here is how"},
                     {"title": "AI edits my reel in 20 seconds", "hook": "AI edits my reel in 20 seconds, here is how"},
                     {"title": "3 AI tools for faster editing", "hook": "3 AI tools that cut my editing time in half"}]}
    assert [i["title"] for i in ideas.clean(res, set())["ideas"]] == ["AI edits my reel in 10 seconds", "3 AI tools for faster editing"]
