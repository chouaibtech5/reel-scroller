"""Model selection logic (no downloads, no model loading)."""
import pytest

from reel_scroller import media
from reel_scroller.backends import hw
from reel_scroller.model import VLM_MODELS, default_backend, select_models

GB = 1024**3
PC = {"env": {}, "apple_silicon": False}


def test_auto_size_rules():
    assert hw.auto_size(64 * GB, 0, apple_silicon=False) == "small"  # lots of RAM, no GPU: 32B on CPU is too slow
    assert hw.auto_size(16 * GB, 24 * GB, apple_silicon=False) == "large"
    assert hw.auto_size(16 * GB, 12 * GB, apple_silicon=False) == "small"
    assert hw.auto_size(32 * GB, 0, apple_silicon=True) == "large"
    assert hw.auto_size(16 * GB, 0, apple_silicon=True) == "small"


def test_default_backend():
    assert default_backend({}, apple_silicon=False) == "ollama"
    assert default_backend({}, apple_silicon=True, mlx_installed=True) == "mlx"
    assert default_backend({}, apple_silicon=True, mlx_installed=False) == "ollama"
    assert default_backend({"REEL_SCROLLER_BACKEND": "openai"}, apple_silicon=True, mlx_installed=True) == "openai"


def test_auto_on_a_pc():
    c = select_models(ram_bytes=16 * GB, vram_bytes=8 * GB, **PC)
    assert (c.backend, c.vlm, c.size) == ("ollama", VLM_MODELS["ollama"]["small"], "small")
    assert c.asr == "faster-whisper" and c.whisper == "small"
    c = select_models(ram_bytes=16 * GB, vram_bytes=24 * GB, **PC)
    assert (c.vlm, c.size, c.whisper) == (VLM_MODELS["ollama"]["large"], "large", "large-v3-turbo")


def test_explicit_size_and_backend():
    assert select_models("large", vram_bytes=0, ram_bytes=8 * GB, **PC).size == "large"
    c = select_models("small", backend="mlx", **PC)
    assert c.vlm == VLM_MODELS["mlx"]["small"]


def test_env_size_and_flag_precedence():
    assert select_models(env={"REEL_SCROLLER_MODEL": "small"}, vram_bytes=48 * GB, apple_silicon=False).size == "small"
    assert select_models("large", env={"REEL_SCROLLER_MODEL": "small"}, apple_silicon=False).size == "large"


def test_model_overrides():
    assert select_models("small", model="llava:13b", **PC).vlm == "llava:13b"
    env = {"REEL_SCROLLER_VLM": "a:b", "REEL_SCROLLER_WHISPER": "medium"}
    c = select_models("small", env=env, apple_silicon=False)
    assert (c.vlm, c.whisper) == ("a:b", "medium")
    assert select_models("small", model="x:y", env={"REEL_SCROLLER_VLM": "a:b"}, apple_silicon=False).vlm == "x:y"


def test_bad_size_and_backend():
    with pytest.raises(ValueError):
        select_models("huge", **PC)
    with pytest.raises(ValueError):
        select_models("small", backend="cloud", **PC)


def test_gpu_vram_parses_nvidia_smi(monkeypatch):
    class P:
        returncode, stdout = 0, "8192\n24576\n"
    monkeypatch.setattr(hw.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(hw.subprocess, "run", lambda *a, **k: P())
    assert hw.gpu_vram_bytes() == 24576 * 1024**2
    P.returncode = 9  # "GPU is lost" and similar driver errors
    assert hw.gpu_vram_bytes() == 0
    monkeypatch.setattr(hw.shutil, "which", lambda _: None)
    assert hw.gpu_vram_bytes() == 0


def test_preflight_hint_per_os(monkeypatch):
    monkeypatch.setattr(media.shutil, "which", lambda _: None)
    assert "winget" in media.preflight("win32")
    assert "apt" in media.preflight("linux")
    assert "brew" in media.preflight("darwin")
    monkeypatch.setattr(media.shutil, "which", lambda b: "/bin/" + b)
    assert media.preflight("win32") == ""
