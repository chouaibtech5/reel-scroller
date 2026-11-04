"""auto: runs scout, study and ideas in order, free unless --allow-paid. The three steps are faked."""
import pytest

from reel_scroller import auto


@pytest.fixture
def steps(monkeypatch, tmp_path):
    calls = []
    for name in ("scout", "study", "ideas"):
        monkeypatch.setattr(getattr(auto, name), "main", lambda argv, n=name: calls.append((n, argv)) or 0)
    monkeypatch.setattr(auto.dashboard, "serve_in_background", lambda *a, **k: calls.append(("dash", a)))
    monkeypatch.setenv("META_IG_USER_ID", "1")
    monkeypatch.setenv("META_ACCESS_TOKEN", "t")
    (tmp_path / "c.txt").write_text("a\n", encoding="utf-8")
    return calls, tmp_path


def run(tmp_path, *extra):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    (out / "urls.tsv").write_text("https://www.instagram.com/reel/A/\n", encoding="utf-8")
    return auto.main(["--creators", str(tmp_path / "c.txt"), "--out-root", str(out), "--no-dash", *extra])


def test_runs_the_three_steps_for_free(steps):
    calls, tmp = steps
    assert run(tmp, "--backend", "gemini", "--niche", "editors", "--limit", "5") == 0
    assert [c[0] for c in calls] == ["scout", "study", "ideas"]
    scout_args, study_args, idea_args = (c[1] for c in calls)
    assert "--run" in scout_args and scout_args[scout_args.index("--source") + 1] == "meta"
    assert study_args[study_args.index("--fetcher") + 1] == "ytdlp" and "--run" in study_args
    assert study_args[study_args.index("--backend") + 1] == "gemini" and study_args[study_args.index("--limit") + 1] == "5"
    assert idea_args[idea_args.index("--niche") + 1] == "editors" and "gemini" in idea_args


def test_allow_paid_lets_apify_fill_in(steps):
    calls, tmp = steps
    assert run(tmp, "--allow-paid") == 0
    assert "--source" not in calls[0][1]
    assert calls[1][1][calls[1][1].index("--fetcher") + 1] == "auto"


def test_without_meta_it_stops_instead_of_paying(steps, monkeypatch, capsys):
    calls, tmp = steps
    monkeypatch.delenv("META_IG_USER_ID")
    assert run(tmp) == 2 and calls == []
    assert "--allow-paid" in capsys.readouterr().out


def test_a_failed_step_stops_the_run(steps, monkeypatch):
    calls, tmp = steps
    monkeypatch.setattr(auto.scout, "main", lambda argv: calls.append(("scout", argv)) or 4)
    assert run(tmp) == 4 and [c[0] for c in calls] == ["scout"]
