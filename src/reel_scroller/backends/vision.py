"""Vision-model backends. Each has `ask(prompt, image=None, max_tokens=500)` returning parsed JSON
(or a parse_error dict). `image` is a path to a JPEG/PNG file.

ollama  local Ollama server (default everywhere but Apple Silicon with MLX installed)
openai  any OpenAI-compatible server: LM Studio, llama.cpp server, vLLM (local), or a hosted one you choose
mlx     mlx-vlm on Apple Silicon (the original reel-watcher path)
gemini  Google's Gemini API with GEMINI_API_KEY. Opt-in: frames and transcripts leave this machine.
"""
from __future__ import annotations

import base64
import http.client
import ipaddress
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..model import RETRY_HINT, parse_json
from ._util import log

TIMEOUT_S = 1800  # a contact sheet on CPU only can take minutes
NUM_CTX = 16384  # one contact sheet is a few thousand image tokens; Ollama's default context is too small


class VisionError(RuntimeError):
    pass


class VisionQuotaError(VisionError):
    """The provider's quota is used up for hours (e.g. a free daily limit). Stop the run; the reels are not at fault."""


class _Base:
    def ask(self, prompt, image=None, max_tokens=500):
        # Second attempt uses a stricter prompt: busy screens can make the model transcribe text until it runs out of tokens.
        attempts = [(prompt, max_tokens, None), (prompt + RETRY_HINT, max_tokens * 2, 1.15)]
        out = ""
        for text, limit, penalty in attempts:
            out = self._generate(text, image, limit, penalty)
            parsed = parse_json(out)
            if parsed is not None:
                return parsed
        return {"parse_error": True, "raw": str(out)[:1000]}

    def _generate(self, text: str, image, max_tokens: int, penalty: float | None) -> str:
        raise NotImplementedError


def _post_json(url: str, body: dict, headers: dict | None = None, timeout: float = TIMEOUT_S) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _b64(image) -> str:
    return base64.b64encode(Path(image).read_bytes()).decode("ascii")


class OllamaVision(_Base):
    def __init__(self, model: str, host: str | None = None):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if "://" not in self.host:
            self.host = "http://" + self.host
        self.temperature = 0.0
        self.think: bool | None = None  # only sent when set: models without thinking reject the field
        self._check()

    def capabilities(self) -> set[str]:
        """What Ollama says this model can do (vision, thinking, ...); empty if it cannot tell."""
        try:
            return set(_post_json(self.host + "/api/show", {"model": self.model}, timeout=20).get("capabilities") or [])
        except (urllib.error.URLError, OSError, ValueError):
            return set()

    def _check(self) -> None:
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=10) as r:
                tags = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as e:
            raise VisionError(f"Ollama is not reachable at {self.host} ({e}). Start Ollama (the app, or `ollama serve`), "
                              f"then run: ollama pull {self.model}") from e
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise VisionError(f"Ollama does not have {self.model}. Run: ollama pull {self.model}")

    def _generate(self, text, image, max_tokens, penalty):
        msg = {"role": "user", "content": text}
        if image:
            msg["images"] = [_b64(image)]
        options = {"temperature": self.temperature, "num_predict": max_tokens,
                   "num_ctx": int(os.environ.get("REEL_SCROLLER_NUM_CTX") or NUM_CTX)}
        if penalty:
            options["repeat_penalty"] = penalty
        body = {"model": self.model, "messages": [msg], "stream": False, "format": "json", "options": options, "keep_alive": "30m"}
        if self.think is not None:
            body["think"] = self.think
        r = _post_json(self.host + "/api/chat", body)
        return (r.get("message") or {}).get("content") or ""


def _is_local(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    if host in ("localhost",) or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


RETRY_STATUS = {429, 500, 502, 503, 504}  # busy or rate limited: wait and try again
MAX_RETRIES = 4


LONG_WAIT_S = 600  # a provider asking us to wait longer than this means a daily quota, not a busy minute


def _error_body(data: bytes) -> dict:
    try:
        body = json.loads(data.decode("utf-8", "replace") or "{}")
    except ValueError:
        return {}
    if isinstance(body, list):
        body = body[0] if body and isinstance(body[0], dict) else {}
    err = body.get("error", {}) if isinstance(body, dict) else {}
    return err if isinstance(err, dict) else {"message": str(err)}


def _error_message(status: int, data: bytes) -> str:
    return f"HTTP {status}: {_error_body(data).get('message') or 'no details'}"[:300]


def _seconds(v) -> float:
    try:
        return float(str(v).rstrip("s"))
    except (TypeError, ValueError):
        return 0.0


def _quota(err: dict) -> tuple[float, bool, str | None]:
    """(seconds the provider asks us to wait, whether it is a daily quota, the limit) from a Google-style 429 body."""
    wait, daily, limit = 0.0, False, None
    for d in err.get("details") or []:
        if not isinstance(d, dict):
            continue
        wait = max(wait, _seconds(d.get("retryDelay")))
        for v in d.get("violations") or []:
            if "PerDay" in str(v.get("quotaId", "")):
                daily, limit = True, v.get("quotaValue")
    return wait, daily or wait > LONG_WAIT_S, limit


def _connection(scheme: str, host: str, port: int | None, timeout: float):
    cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    return cls(host, port, timeout=timeout)


class _KeepAlive:
    """One HTTP connection reused for every call of a run. On some networks opening a connection takes 5 to 60 s
    while the request itself takes 5 s, so reconnecting per call made hosted models slower than they are."""

    def __init__(self, base: str, timeout: float = TIMEOUT_S):
        u = urllib.parse.urlparse(base)
        self.scheme, self.host, self.port, self.prefix, self.timeout = u.scheme, u.hostname, u.port, u.path.rstrip("/"), timeout
        self.conn = None

    def post(self, path: str, body: bytes, headers: dict) -> tuple[int, dict, bytes]:
        for attempt in (1, 2):  # a kept connection the server closed meanwhile fails once; reconnect and send again
            if self.conn is None:
                self.conn = _connection(self.scheme, self.host, self.port, self.timeout)
            try:
                self.conn.request("POST", self.prefix + path, body=body, headers=headers)
                r = self.conn.getresponse()
                return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
            except (http.client.HTTPException, OSError):
                self.close()
                if attempt == 2:
                    raise
        raise OSError("unreachable")

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
        self.conn = None


class OpenAIVision(_Base):
    """Chat Completions on an OpenAI-compatible server (local by default)."""
    send_penalty = True  # llama.cpp and vLLM accept frequency_penalty; some hosted APIs reject it
    extra: dict = {}

    def __init__(self, model: str, base: str | None = None, key: str | None = None):
        self.model = model
        self.base = (base or os.environ.get("REEL_SCROLLER_OPENAI_BASE") or "http://localhost:1234/v1").rstrip("/")
        self.key = key if key is not None else os.environ.get("REEL_SCROLLER_OPENAI_KEY", "")
        self.temperature = 0.0
        self.dropped: set[str] = set()
        self.http = _KeepAlive(self.base)
        if not _is_local(self.base):
            log(f"note: {self.base} is not a local address; frames and transcripts are sent there")

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.key}"} if self.key else {})}
        data = json.dumps(body).encode("utf-8")
        for attempt in range(MAX_RETRIES + 1):
            try:
                status, rh, raw = self.http.post("/chat/completions", data, headers)
            except (http.client.HTTPException, OSError) as e:
                raise VisionError(f"no OpenAI-compatible server at {self.base} ({e}). Start LM Studio's server or "
                                  "llama.cpp, or set REEL_SCROLLER_OPENAI_BASE") from e
            if status < 400:
                return json.loads(raw.decode("utf-8") or "{}")
            err = _error_body(raw)
            if status == 429:
                wait, daily, limit = _quota(err)
                if daily:
                    hours = f"; it resets in about {wait / 3600:.0f} h" if wait else ""
                    raise VisionQuotaError(f"{self.model}: the API quota is used up"
                                           f"{f' ({limit} requests a day on this key)' if limit else ''}{hours}. "
                                           "Try another model with --model, run locally with --backend ollama, or enable billing.")
            if status in RETRY_STATUS and attempt < MAX_RETRIES:
                wait = _quota(err)[0] if status == 429 else 0.0
                wait = wait or _seconds(rh.get("retry-after")) or min(60, 5 * 2 ** attempt)
                log(f"{self.model}: HTTP {status}, waiting {wait:.0f}s (free tiers allow a few requests per minute)")
                time.sleep(wait)
                continue
            if status == 400 and "reasoning_effort" in body and "reasoning" not in self.dropped:
                # newer models reject some thinking settings; ask again without it, once
                log(f"{self.model} rejected reasoning_effort={body['reasoning_effort']!r}; retrying without it")
                self.dropped.add("reasoning")
                self.extra = {k: v for k, v in self.extra.items() if k != "reasoning_effort"}
                body = {k: v for k, v in body.items() if k != "reasoning_effort"}
                data = json.dumps(body).encode("utf-8")
                continue
            msg = _error_message(status, raw)
            if status in (401, 403):
                raise VisionError(f"{self.base} rejected the API key ({msg})")
            raise VisionError(f"{self.base}: {msg}")
        raise VisionError(f"{self.base}: still busy after {MAX_RETRIES} retries")

    def _generate(self, text, image, max_tokens, penalty):
        content: list[dict] = [{"type": "text", "text": text}]
        if image:
            mime = "image/png" if Path(image).suffix.lower() == ".png" else "image/jpeg"
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{_b64(image)}"}})
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature, "max_tokens": max_tokens, **self.extra}
        if penalty and self.send_penalty:
            body["frequency_penalty"] = round(penalty - 1, 2)
        choices = self._post(body).get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content") or ""


GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"


def gemini_effort(model: str) -> str:
    """Least thinking each model accepts: 2.5 Flash can turn it off, Pro cannot, 3.x accepts 'minimal' but not 'none'."""
    if "2.5" in model:
        return "low" if "pro" in model else "none"
    return "minimal"


class GeminiVision(OpenAIVision):
    """Google Gemini through its OpenAI-compatible endpoint. Thinking is off: it used the token budget and slowed every call."""
    send_penalty = False  # Gemini rejects frequency_penalty

    def __init__(self, model: str, key: str | None = None):
        key = key if key is not None else (os.environ.get("GEMINI_API_KEY") or "").strip()
        if not key:
            raise VisionError("GEMINI_API_KEY is not set. Get a key at https://aistudio.google.com/apikey and set it in your shell "
                              "(never in a file in this repo)")
        self.model, self.base, self.key, self.temperature = model, GEMINI_BASE, key, 0.0
        self.dropped: set[str] = set()
        self.http = _KeepAlive(self.base)
        self.extra = {"reasoning_effort": gemini_effort(model), "response_format": {"type": "json_object"}}
        log(f"note: using Google's Gemini API ({model}): frames and transcripts are sent to Google")


class MLXVision(_Base):
    """Loads the VLM once (Apple Silicon only)."""

    def __init__(self, model: str):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.model, self.processor = load(model)
        self.config = load_config(model)

    def _generate(self, text, image, max_tokens, penalty):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        images = [str(image)] if image else []
        formatted = apply_chat_template(self.processor, self.config, text, num_images=len(images), enable_thinking=False)
        kwargs = {"repetition_penalty": penalty} if penalty else {}
        out = generate(self.model, self.processor, formatted, image=images or None,
                       max_tokens=max_tokens, temperature=0.0, verbose=False, **kwargs)
        return getattr(out, "text", out)


def make_vision(backend: str, model: str):
    if backend == "ollama":
        return OllamaVision(model)
    if backend == "openai":
        return OpenAIVision(model)
    if backend == "gemini":
        return GeminiVision(model)
    if backend == "mlx":
        try:
            return MLXVision(model)
        except ImportError as e:
            raise VisionError("the mlx backend needs Apple Silicon and `uv sync --extra mac`") from e
    raise ValueError(f"unknown vision backend {backend!r}")
