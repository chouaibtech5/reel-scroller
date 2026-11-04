"""Vision backends against a fake HTTP layer (no server, no model)."""
import base64
import io
import json
import types
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


class FakeConn:
    """Stands in for http.client connections; answers from a shared queue of (status, body) or exceptions."""
    made: list = []

    def __init__(self, answers, sent):
        self.answers, self.sent = answers, sent

    def request(self, method, path, body=None, headers=None):
        self.sent.append({"path": path, "body": json.loads(body), "headers": headers, "conn": id(self)})

    def getresponse(self):
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        status, payload, *hdrs = a
        return types.SimpleNamespace(status=status, getheaders=lambda: list((hdrs[0] if hdrs else {}).items()),
                                     read=lambda: json.dumps(payload).encode())

    def close(self):
        pass


def fake_connections(monkeypatch, *answers):
    answers, sent, made = list(answers), [], []

    def connect(scheme, host, port, timeout):
        made.append((scheme, host, port))
        return FakeConn(answers, sent)
    monkeypatch.setattr(v, "_connection", connect)
    return sent, made


OK = (200, {"choices": [{"message": {"content": '{"a": 1}'}}]})


def test_openai_compatible_reuses_one_connection(tmp_path, monkeypatch):
    img = tmp_path / "f.png"
    img.write_bytes(b"png")
    sent, made = fake_connections(monkeypatch, OK, OK)
    m = v.OpenAIVision("local-model", base="http://localhost:8080/v1/", key="k")
    assert m.ask("p", img) == {"a": 1} and m.ask("q") == {"a": 1}
    assert made == [("http", "localhost", 8080)] and sent[0]["conn"] == sent[1]["conn"]  # one connection for the run
    assert sent[0]["path"] == "/v1/chat/completions" and sent[0]["headers"]["Authorization"] == "Bearer k"
    assert sent[0]["body"]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_reconnects_once_when_a_kept_connection_dropped(monkeypatch):
    sent, made = fake_connections(monkeypatch, ConnectionResetError("closed"), OK)
    assert v.OpenAIVision("m", base="https://api.example.com/v1", key="k").ask("p") == {"a": 1}
    assert len(made) == 2 and made[0] == ("https", "api.example.com", None)


def test_openai_server_down(monkeypatch):
    fake_connections(monkeypatch, ConnectionRefusedError("refused"), ConnectionRefusedError("refused"))
    with pytest.raises(v.VisionError, match="REEL_SCROLLER_OPENAI_BASE"):
        v.OpenAIVision("m", base="http://localhost:1234/v1", key="").ask("p")


def test_rate_limit_waits_then_retries(monkeypatch):
    waits = []
    monkeypatch.setattr(v.time, "sleep", waits.append)
    fake_connections(monkeypatch, (429, {"error": {"message": "quota"}}, {"Retry-After": "7"}), (503, {}), OK)
    assert v.OpenAIVision("m", base="http://localhost:1/v1", key="").ask("p") == {"a": 1}
    assert waits == [7.0, 10]


def test_rejected_key_and_bad_request_show_the_providers_message(monkeypatch):
    fake_connections(monkeypatch, (401, [{"error": {"message": "API key not valid"}}]))
    with pytest.raises(v.VisionError, match="rejected the API key.*API key not valid"):
        v.OpenAIVision("m", base="http://localhost:1/v1", key="bad").ask("p")
    fake_connections(monkeypatch, (400, {"error": {"message": "Unknown name frequency_penalty"}}))
    with pytest.raises(v.VisionError, match="HTTP 400: Unknown name"):
        v.OpenAIVision("m", base="http://localhost:1/v1", key="").ask("p")


def test_gemini_settings(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(v.VisionError, match="GEMINI_API_KEY is not set"):
        v.make_vision("gemini", "gemini-2.5-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "G")
    sent, made = fake_connections(monkeypatch, (200, {"choices": [{"message": {"content": "no json"}}]}), OK)
    assert v.make_vision("gemini", "gemini-2.5-flash").ask("p", max_tokens=100) == {"a": 1}
    first, retry = sent[0], sent[1]
    assert made == [("https", "generativelanguage.googleapis.com", None)] and first["path"] == "/v1beta/openai/chat/completions"
    assert first["headers"]["Authorization"] == "Bearer G" and "G" not in first["path"]
    assert first["body"]["reasoning_effort"] == "none" and first["body"]["response_format"] == {"type": "json_object"}
    assert "frequency_penalty" not in retry["body"] and retry["body"]["max_tokens"] == 200
    assert v.GeminiVision("gemini-2.5-pro", key="G").extra["reasoning_effort"] == "low"


def test_is_local():
    assert v._is_local("http://localhost:1234/v1")
    assert v._is_local("http://127.0.0.1:1234")
    assert v._is_local("http://192.168.1.20:8000/v1")
    assert not v._is_local("https://api.example.com/v1")


def test_unknown_backend():
    with pytest.raises(ValueError):
        v.make_vision("cloud", "x")


DAILY = (429, {"error": {"code": 429, "message": "You exceeded your current quota", "details": [
    {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]},
    {"retryDelay": "30438s"}]}})


def test_daily_quota_stops_at_once_without_retrying(monkeypatch):
    waits = []
    monkeypatch.setattr(v.time, "sleep", waits.append)
    sent, _ = fake_connections(monkeypatch, DAILY)
    with pytest.raises(v.VisionQuotaError, match="20 requests a day.*8 h"):
        v.GeminiVision("gemini-2.5-flash", key="G").ask("p")
    assert waits == [] and len(sent) == 1


def test_per_minute_limit_waits_as_long_as_google_says(monkeypatch):
    waits = []
    monkeypatch.setattr(v.time, "sleep", waits.append)
    fake_connections(monkeypatch, (429, {"error": {"details": [{"retryDelay": "34s"}]}}), OK)
    assert v.GeminiVision("gemini-3.5-flash-lite", key="G").ask("p") == {"a": 1}
    assert waits == [34.0]


def test_rejected_thinking_setting_is_dropped_once(monkeypatch):
    sent, _ = fake_connections(monkeypatch, (400, {"error": {"message": "Request contains an invalid argument."}}), OK, OK)
    m = v.GeminiVision("gemini-9-flash", key="G")
    assert m.ask("p") == {"a": 1} and m.ask("q") == {"a": 1}
    assert sent[0]["body"]["reasoning_effort"] == "minimal"
    assert "reasoning_effort" not in sent[1]["body"] and "reasoning_effort" not in sent[2]["body"]


def test_gemini_effort_per_model():
    assert v.gemini_effort("gemini-2.5-flash") == "none"
    assert v.gemini_effort("gemini-2.5-pro") == "low"
    assert v.gemini_effort("gemini-3.5-flash-lite") == "minimal"
