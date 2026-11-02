# Cross-platform port: shared contract

This is the interface every part of the port agrees on. Windows, Linux and macOS are all first-class.
The CLI (`study`, `export`, `advice`), the study JSON schema in `README.md`, the Apify dry-run default,
the per-batch spend cap and resumability do not change.

## Package layout

```
src/reel_watcher/
  backends/
    __init__.py     empty
    _util.py        log(msg) -> prints to stderr; is_apple_silicon() -> bool
    vision.py       vision-model backends (Ollama, OpenAI-compatible local server, MLX)
    hw.py           RAM / GPU detection and model-size choice
    asr.py          speech to text with word timing (faster-whisper, MLX on Mac)
    ocr.py          on-screen text (RapidOCR, Apple Vision on Mac)
  model.py          keeps select_models/configure/parse_json/RETRY_HINT, delegates to backends
  study.py          pipeline, calls backends only through the functions below
```

Heavy libraries (faster_whisper, rapidocr, onnxruntime, mlx_*, Vision/pyobjc) are imported lazily
inside functions, never at module top level, so `import reel_watcher.study` works on any OS with only
numpy and pillow installed. Tests never download a model or touch the network.

## backends/asr.py

```python
DEFAULT_MODELS = {
    "faster-whisper": {"small": "small", "large": "large-v3-turbo"},
    "mlx": {"small": "mlx-community/whisper-small-mlx", "large": "mlx-community/whisper-large-v3-turbo"},
}

def resolve_asr(size: str, env: dict | None = None, apple_silicon: bool | None = None) -> tuple[str, str]:
    """(backend, model). backend: env REEL_WATCHER_ASR (faster-whisper|mlx|auto, default auto);
    auto = "mlx" when apple_silicon and mlx_whisper is importable, else "faster-whisper".
    model: env REEL_WATCHER_WHISPER, else DEFAULT_MODELS[backend][size]."""

def transcribe_words(video, model: str, backend: str = "faster-whisper") -> dict:
    """{"language": str|None, "text": str, "segments": [{"start","end","text"}], "words": [{"w","s","e"}]}"""
```

- Times rounded to 2 decimals. Skip segments with `no_speech_prob >= 0.6` or empty text.
  `text` is the kept segments' text joined by single spaces. Same shape as the original mlx version.
- faster-whisper: `word_timestamps=True`, `condition_on_previous_text=False`, no VAD filter.
  Device from env `REEL_WATCHER_DEVICE` (auto|cpu|cuda, default auto): auto = cuda when
  `ctranslate2.get_cuda_device_count() > 0`, else cpu. compute_type float16 on cuda, int8 on cpu.
  If loading on cuda fails, log it and retry on cpu. Cache loaded models per (model, device) for the process.
- mlx: same call the original code made (`mlx_whisper.transcribe(..., word_timestamps=True,
  condition_on_previous_text=False, no_speech_threshold=0.6, verbose=None)`).

## backends/ocr.py

```python
SECRET_RE = re.compile(r"\b(?:sk|pk|ghp|xox[a-z]|AKIA|AIza)[-_A-Za-z0-9]{12,}\b|\b[A-Fa-f0-9]{32,}\b|\b[A-Za-z0-9+/_-]{40,}\b")

def ocr_backend(env: dict | None = None) -> str:   # "apple" | "rapidocr" | "none"
def ocr_image(path) -> str:
```

- Choice: env `REEL_WATCHER_OCR` (auto|apple|rapidocr|none, default auto). auto = "apple" on macOS when
  `Vision` (pyobjc) imports, else "rapidocr" when it imports, else "none".
- Output: text lines ordered top to bottom (then left to right), joined by spaces, `SECRET_RE` replaced by
  `[redacted]`, whitespace collapsed, stripped, max 500 chars. Returns "" when nothing is found or no
  backend is available (log the reason once per process, not per frame).
- RapidOCR: support the `rapidocr` package (v2/v3, with onnxruntime) and fall back to the older
  `rapidocr_onnxruntime` import if that is what is installed. Create the engine once; guard calls with a lock.
- Apple: move the original Apple Vision code here unchanged.

## backends/vision.py

```python
def make_vision(backend: str, model: str):  # returns an object with:
    def ask(self, prompt: str, image=None, max_tokens: int = 500) -> dict
```

`ask` returns parsed JSON (via `model.parse_json`) or `{"parse_error": True, "raw": "..."}`, with the same
two attempts as the original (second attempt adds `RETRY_HINT`, doubles tokens). temperature 0.
Backends: `ollama` (default; `OLLAMA_HOST`, default `http://localhost:11434`, `/api/chat`, images base64,
`format: "json"`), `openai` (any OpenAI-compatible LOCAL server such as LM Studio, llama.cpp, vLLM;
`REEL_WATCHER_OPENAI_BASE` default `http://localhost:1234/v1`, optional `REEL_WATCHER_OPENAI_KEY`),
`mlx` (the original code, Apple Silicon only). Standard library HTTP only (urllib).

## Model choice (backends/hw.py + model.py)

| size | ollama | openai | mlx | whisper |
|---|---|---|---|---|
| small | `qwen2.5vl:7b` (about 6 GB) | user sets the model name | `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` | see asr |
| large | `qwen2.5vl:32b` (about 21 GB) | user sets the model name | `mlx-community/Qwen3.6-35B-A3B-4bit` | see asr |

`--model-size auto` (default) picks `large` only with an NVIDIA GPU of 20 GB VRAM or more, or Apple Silicon
with 32 GB RAM or more; otherwise `small`. RAM via psutil, VRAM via `nvidia-smi` when present.

## CLI and environment

New flag: `--backend ollama|openai|mlx` (env `REEL_WATCHER_BACKEND`; default `mlx` on Apple Silicon when
mlx_vlm imports, else `ollama`). Existing flags unchanged. Environment variables:

| Variable | Meaning |
|---|---|
| `REEL_WATCHER_BACKEND` | vision backend |
| `REEL_WATCHER_MODEL` | size preset small/large/auto |
| `REEL_WATCHER_VLM` | exact vision model name for the chosen backend |
| `OLLAMA_HOST` | Ollama address |
| `REEL_WATCHER_OPENAI_BASE`, `REEL_WATCHER_OPENAI_KEY` | OpenAI-compatible local server |
| `REEL_WATCHER_ASR`, `REEL_WATCHER_WHISPER`, `REEL_WATCHER_DEVICE` | speech to text |
| `REEL_WATCHER_OCR` | OCR backend |
| `APIFY_TOKEN`, `REEL_WATCHER_CONTEXT`, `REEL_WATCHER_WORKFLOW_RE` | unchanged |

## Install

`uv sync` installs the cross-platform set: numpy, pillow, psutil, faster-whisper, rapidocr, onnxruntime.
`uv sync --extra mac` adds mlx-vlm, mlx-whisper and pyobjc-framework-Vision (macOS only).
`uv sync --extra cuda` documents GPU use for faster-whisper (CUDA 12 + cuDNN 9 runtime from NVIDIA).
ffmpeg on PATH: `winget install Gyan.FFmpeg` (Windows), `sudo apt install ffmpeg` (Debian/Ubuntu),
`brew install ffmpeg` (macOS). Ollama from https://ollama.com, then `ollama pull qwen2.5vl:7b`.
