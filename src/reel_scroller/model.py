"""Model choice and JSON parsing. The backends themselves live in `backends/`.

Choice (highest priority first):
  --backend / REEL_SCROLLER_BACKEND     ollama | openai | mlx (default: mlx on Apple Silicon when installed, else ollama)
  --model / REEL_SCROLLER_VLM           exact vision model name for that backend
  --model-size / REEL_SCROLLER_MODEL    small | large | auto
  auto: large with an NVIDIA GPU of 20 GB+ or Apple Silicon with 32 GB+ RAM, else small
Speech to text is chosen in backends/asr.py (REEL_SCROLLER_ASR, REEL_SCROLLER_WHISPER).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
from typing import NamedTuple

from .backends import hw
from .backends._util import is_apple_silicon

VLM_MODELS = {
    "ollama": {"small": "qwen2.5vl:7b", "large": "qwen2.5vl:32b"},
    "openai": {"small": "qwen2.5-vl-7b-instruct", "large": "qwen2.5-vl-32b-instruct"},
    "mlx": {"small": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit", "large": "mlx-community/Qwen3.6-35B-A3B-4bit"},
}
BACKENDS = tuple(VLM_MODELS)


class Choice(NamedTuple):
    backend: str
    vlm: str
    asr: str
    whisper: str
    size: str


def default_backend(env=None, apple_silicon=None, mlx_installed=None) -> str:
    env = os.environ if env is None else env
    b = env.get("REEL_SCROLLER_BACKEND")
    if b:
        return b
    apple = is_apple_silicon() if apple_silicon is None else apple_silicon
    if mlx_installed is None:
        mlx_installed = apple and importlib.util.find_spec("mlx_vlm") is not None
    return "mlx" if apple and mlx_installed else "ollama"


def select_models(size=None, model=None, backend=None, env=None, ram_bytes=None, vram_bytes=None, apple_silicon=None) -> Choice:
    """No downloads, no model loading."""
    from .backends.asr import resolve_asr

    env = os.environ if env is None else env
    apple = is_apple_silicon() if apple_silicon is None else apple_silicon
    backend = backend or default_backend(env, apple)
    if backend not in VLM_MODELS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}, got {backend!r}")
    size = size or env.get("REEL_SCROLLER_MODEL") or "auto"
    if size == "auto":
        ram = hw.total_ram_bytes() if ram_bytes is None else ram_bytes
        vram = hw.gpu_vram_bytes() if vram_bytes is None else vram_bytes
        size = hw.auto_size(ram, vram, apple)
    if size not in ("small", "large"):
        raise ValueError(f"model size must be small, large or auto, got {size!r}")
    vlm = model or env.get("REEL_SCROLLER_VLM") or VLM_MODELS[backend][size]
    asr, whisper = resolve_asr(size, env, apple)
    return Choice(backend, vlm, asr, whisper, size)


CHOICE: Choice | None = None


def configure(size=None, model=None, backend=None) -> Choice:
    """Set the module-wide model choice (called once from the CLI before models load)."""
    global CHOICE
    CHOICE = select_models(size, model, backend)
    return CHOICE


def choice() -> Choice:
    return CHOICE or configure()


RETRY_HINT = (
    "\n\nIMPORTANT: output must be short, valid JSON. Keep every text field under 200 characters: "
    "summarize visible text, never copy long passages, code, numbers or grids."
)


def parse_json(text: str):
    """Extract the outermost JSON object from model output, or None."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start: end + 1])
    except json.JSONDecodeError:
        return None
