"""Vision backends against a fake HTTP layer (no server, no model)."""
import base64
import io
import json
import urllib.error

import pytest

from reel_scroller.backends import vision as v


class FakeHTTP:
    """Records requests; answers from a queue of dicts (or raises an exception instance)."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []

    def __call__(self, req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        body = None if isinstance(req, str) or req.data is None else json.loads(req.data)
        headers = {} if isinstance(req, str) else dict(req.header_items())
        self.requests.append({"url": url, "body": body, "headers": headers})
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return io.BytesIO(json.dumps(a).encode("utf-8"))


TAGS = {"models": [{"name": "qwen2.5vl:7b", "model": "qwen2.5vl:7b"}]}


def chat(text):
    return {"message": {"role": "assistant", "content": text}}


def test_ollama_sends_image_and_parses(tmp_path, monkeypatch):
    img = tmp_path / "sheet.jpg"
    img.write_bytes(b"\xff\xd8jpeg")
    http = FakeHTTP(TAGS, chat('noise {"hook": "x"} noise'))
    monkeypatch.setattr(v.urllib.request, "urlopen", http)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    out = v.make_vision("ollama", "qwen2.5vl:7b").ask("describe", img, max_tokens=300)
    assert out == {"hook": "x"}
    req = http.requests[1]
    assert req["url"] == "http://localhost:11434/api/chat"
    msg = req["body"]["messages"][0]
    assert base64.b64decode(msg["images"][0]) == b"\xff\xd8jpeg"
    assert req["body"]["options"]["num_predict"] == 300 and req["body"]["options"]["temperature"] == 0
    assert req["body"]["options"]["num_ctx"] >= 8192 and req["body"]["stream"] is False


def test_ollama_retries_with_hint_then_gives_up(monkeypatch):
    http = FakeHTTP(TAGS, chat("not json"), chat("still not"))
    monkeypatch.setattr(v.urllib.request, "urlopen", http)
    out = v.OllamaVision("qwen2.5vl:7b", host="127.0.0.1:11434").ask("p", max_tokens=100)
    assert out["parse_error"] is True and out["raw"] == "still not"
    second = http.requests[2]["body"]
    assert second["options"]["num_predict"] == 200 and second["options"]["repeat_penalty"] == 1.15
    assert "IMPORTANT" in second["messages"][0]["content"]
    assert http.requests[0]["url"] == "http://127.0.0.1:11434/api/tags"


def test_ollama_not_running_or_model_missing(monkeypatch):
    monkeypatch.setattr(v.urllib.request, "urlopen", FakeHTTP(urllib.error.URLError("refused")))
    with pytest.raises(v.VisionError, match="ollama pull qwen2.5vl:7b"):
        v.OllamaVision("qwen2.5vl:7b")
    monkeypatch.setattr(v.urllib.request, "urlopen", FakeHTTP({"models": []}))
    with pytest.raises(v.VisionError, match="does not have"):
        v.OllamaVision("qwen2.5vl:7b")


def test_openai_compatible(tmp_path, monkeypatch):
    img = tmp_path / "f.png"
    img.write_bytes(b"png")
    http = FakeHTTP({"choices": [{"message": {"content": '{"a": 1}'}}]})
    monkeypatch.setattr(v.urllib.request, "urlopen", http)
    out = v.OpenAIVision("local-model", base="http://localhost:8080/v1/", key="k").ask("p", img)
    assert out == {"a": 1}
    req = http.requests[0]
    assert req["url"] == "http://localhost:8080/v1/chat/completions"
    assert req["headers"].get("Authorization") == "Bearer k"
    url = req["body"]["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_openai_server_down(monkeypatch):
    monkeypatch.setattr(v.urllib.request, "urlopen", FakeHTTP(urllib.error.URLError("refused")))
    with pytest.raises(v.VisionError, match="REEL_SCROLLER_OPENAI_BASE"):
        v.OpenAIVision("m", base="http://localhost:1234/v1", key="").ask("p")


def test_is_local():
    assert v._is_local("http://localhost:1234/v1")
    assert v._is_local("http://127.0.0.1:1234")
    assert v._is_local("http://192.168.1.20:8000/v1")
    assert not v._is_local("https://api.example.com/v1")


def test_unknown_backend():
    with pytest.raises(ValueError):
        v.make_vision("cloud", "x")
