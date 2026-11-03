"""Speech-to-text backends (faster-whisper and mlx-whisper)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from . import _util

DEFAULT_MODELS = {
    "faster-whisper": {"small": "small", "large": "large-v3-turbo"},
    "mlx": {"small": "mlx-community/whisper-small-mlx", "large": "mlx-community/whisper-large-v3-turbo"},
}

_MODELS: dict[tuple[str, str], object] = {}


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, AttributeError):
        return sys.modules.get(name) is not None


def resolve_asr(size: str, env: dict | None = None, apple_silicon: bool | None = None) -> tuple[str, str]:
    """Return (backend, model) from size and environment overrides."""
    env = os.environ if env is None else env
    if apple_silicon is None:
        apple_silicon = _util.is_apple_silicon()
    backend = (env.get("REEL_SCROLLER_ASR") or "auto").strip().lower()
    if backend == "auto":
        backend = "mlx" if (apple_silicon and _has_module("mlx_whisper")) else "faster-whisper"
    if backend not in DEFAULT_MODELS:
        raise ValueError(f"unknown ASR backend: {backend!r}")
    if size not in DEFAULT_MODELS[backend]:
        raise ValueError(f"unknown ASR model size: {size!r}")
    model = (env.get("REEL_SCROLLER_WHISPER") or "").strip() or DEFAULT_MODELS[backend][size]
    return backend, model


def _field(obj, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _map_segments(raw_segments, language: str | None) -> dict:
    """Normalize raw Whisper segments (dicts or objects) into the study transcript dict."""
    words, segs = [], []
    for s in raw_segments or []:
        no_speech = _field(s, "no_speech_prob", 0) or 0
        txt = (_field(s, "text", "") or "").strip()
        if no_speech >= 0.6 or not txt:
            continue
        segs.append({
            "start": round(float(_field(s, "start", 0.0)), 2),
            "end": round(float(_field(s, "end", 0.0)), 2),
            "text": txt,
        })
        for w in _field(s, "words") or []:
            words.append({
                "w": (_field(w, "word", "") or "").strip(),
                "s": round(float(_field(w, "start", 0.0)), 2),
                "e": round(float(_field(w, "end", 0.0)), 2),
            })
    return {"language": language, "text": " ".join(s["text"] for s in segs), "segments": segs, "words": words}


def _resolve_device(env: dict | None = None) -> str:
    env = os.environ if env is None else env
    dev = (env.get("REEL_SCROLLER_DEVICE") or "auto").strip().lower()
    if dev == "auto":
        try:
            import ctranslate2
            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            return "cpu"
    if dev not in ("cpu", "cuda"):
        raise ValueError(f"unknown device: {dev!r}")
    return dev


CUDA_COMPUTE_PREFERENCE = ("float16", "int8_float16", "int8_float32", "int8", "float32")


def _compute_type(device: str) -> str:
    """float16 on modern GPUs; older ones (e.g. Pascal laptops like the MX150) only do int8/float32."""
    if device != "cuda":
        return "int8"
    try:
        import ctranslate2
        ok = set(ctranslate2.get_supported_compute_types("cuda"))
    except Exception:  # noqa: BLE001
        return "float16"
    return next((t for t in CUDA_COMPUTE_PREFERENCE if t in ok), "float16")


def _add_cuda_dll_dirs() -> list[str]:
    """Windows: `uv sync --extra cuda` puts cuBLAS/cuDNN DLLs in site-packages/nvidia/*/bin, which Windows does not search."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return []
    added = []
    for base in sys.path:
        for d in sorted(Path(base).glob("nvidia/*/bin")) if base and Path(base).is_dir() else []:
            try:
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
                added.append(str(d))
            except OSError:
                pass
    return added


def _get_whisper_model(model: str, device: str):
    if (model, device) in _MODELS:
        return _MODELS[(model, device)]
    from faster_whisper import WhisperModel
    if device == "cuda":
        _add_cuda_dll_dirs()
    compute_type = _compute_type(device)
    try:
        inst = WhisperModel(model, device=device, compute_type=compute_type)
        _MODELS[(model, device)] = inst
        return inst
    except Exception as e:
        if device != "cuda":
            raise
        _util.log(f"faster-whisper cuda failed ({e}); retrying on cpu")
        if (model, "cpu") not in _MODELS:
            _MODELS[(model, "cpu")] = WhisperModel(model, device="cpu", compute_type="int8")
        _MODELS[(model, "cuda")] = _MODELS[(model, "cpu")]
        return _MODELS[(model, "cpu")]


def _decode_audio(video, sample_rate: int = 16000):
    """Mono float32 samples via ffmpeg. faster-whisper's own decoder (PyAV) breaks across PyAV releases."""
    import subprocess

    import numpy as np
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sample_rate),
                        "-f", "s16le", "-"], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read audio: {p.stderr.decode('utf-8', 'replace').strip()[-300:]}")
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _faster_whisper(video, model: str, device: str) -> dict:
    wm = _get_whisper_model(model, device)
    segs, info = wm.transcribe(_decode_audio(video), word_timestamps=True, condition_on_previous_text=False, vad_filter=False)
    return _map_segments(segs, getattr(info, "language", None))  # segs is lazy: decoding happens here


def transcribe_words(video, model: str, backend: str = "faster-whisper") -> dict:
    """Transcribe spoken words with timestamps using faster-whisper or mlx-whisper."""
    if backend == "faster-whisper":
        device = _resolve_device()
        try:
            return _faster_whisper(video, model, device)
        except Exception as e:
            # Missing cuDNN/cuBLAS DLLs only fail once decoding starts, not when the model loads
            if device != "cuda":
                raise
            _util.log(f"faster-whisper on cuda failed ({e}); using cpu from now on")
            _MODELS[(model, "cuda")] = _get_whisper_model(model, "cpu")
            return _faster_whisper(video, model, "cpu")
    if backend == "mlx":
        import mlx_whisper
        r = mlx_whisper.transcribe(
            str(video),
            path_or_hf_repo=model,
            word_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            verbose=None,
        )
        return _map_segments(r.get("segments", []), r.get("language"))
    raise ValueError(f"unknown ASR backend: {backend!r}")
