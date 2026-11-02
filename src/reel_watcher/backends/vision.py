"""Vision-model backends. Each has `ask(prompt, image=None, max_tokens=500)` returning parsed JSON
(or a parse_error dict). `image` is a path to a JPEG/PNG file.

ollama  local Ollama server (default everywhere but Apple Silicon with MLX installed)
openai  any OpenAI-compatible LOCAL server: LM Studio, llama.cpp server, vLLM
mlx     mlx-vlm on Apple Silicon (the original reel-watcher path)
"""
from __future__ import annotations

import base64
import ipaddress
import json
import os
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
        self._check()

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
        options = {"temperature": 0, "num_predict": max_tokens,
                   "num_ctx": int(os.environ.get("REEL_WATCHER_NUM_CTX") or NUM_CTX)}
        if penalty:
            options["repeat_penalty"] = penalty
        r = _post_json(self.host + "/api/chat", {"model": self.model, "messages": [msg], "stream": False,
                                                 "format": "json", "options": options, "keep_alive": "30m"})
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


class OpenAIVision(_Base):
    """Chat Completions on a local OpenAI-compatible server."""

    def __init__(self, model: str, base: str | None = None, key: str | None = None):
        self.model = model
        self.base = (base or os.environ.get("REEL_WATCHER_OPENAI_BASE") or "http://localhost:1234/v1").rstrip("/")
        self.key = key if key is not None else os.environ.get("REEL_WATCHER_OPENAI_KEY", "")
        if not _is_local(self.base):
            log(f"warning: {self.base} is not a local address; frames will leave this machine")

    def _generate(self, text, image, max_tokens, penalty):
        content: list[dict] = [{"type": "text", "text": text}]
        if image:
            mime = "image/png" if Path(image).suffix.lower() == ".png" else "image/jpeg"
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{_b64(image)}"}})
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "temperature": 0, "max_tokens": max_tokens}
        if penalty:
            body["frequency_penalty"] = round(penalty - 1, 2)
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        try:
            r = _post_json(self.base + "/chat/completions", body, headers)
        except (urllib.error.URLError, OSError) as e:
            raise VisionError(f"no OpenAI-compatible server at {self.base} ({e}). Start LM Studio's server or "
                              "llama.cpp, or set REEL_WATCHER_OPENAI_BASE") from e
        choices = r.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content") or ""


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
    if backend == "mlx":
        try:
            return MLXVision(model)
        except ImportError as e:
            raise VisionError("the mlx backend needs Apple Silicon and `uv sync --extra mac`") from e
    raise ValueError(f"unknown vision backend {backend!r}")
