"""Tests for cross-platform speech-to-text (ASR) and OCR backends (no network, no model downloads)."""
from __future__ import annotations

import types
import pytest

from reel_scroller.backends import asr, ocr


def test_resolve_asr_precedence_and_auto(monkeypatch):
    # Explicit backend via env, default model per size
    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", None)
    assert asr.resolve_asr("small", env={"REEL_SCROLLER_ASR": "faster-whisper"}, apple_silicon=True) == (
        "faster-whisper",
        "small",
    )
    assert asr.resolve_asr("large", env={"REEL_SCROLLER_ASR": "faster-whisper"}, apple_silicon=False) == (
        "faster-whisper",
        "large-v3-turbo",
    )
    assert asr.resolve_asr("small", env={"REEL_SCROLLER_ASR": "mlx"}, apple_silicon=False) == (
        "mlx",
        "mlx-community/whisper-small-mlx",
    )
    assert asr.resolve_asr("large", env={"REEL_SCROLLER_ASR": "mlx"}, apple_silicon=False) == (
        "mlx",
        "mlx-community/whisper-large-v3-turbo",
    )

    # Env model override wins over size preset
    assert asr.resolve_asr(
        "small",
        env={"REEL_SCROLLER_ASR": "faster-whisper", "REEL_SCROLLER_WHISPER": "custom/whisper"},
        apple_silicon=False,
    ) == ("faster-whisper", "custom/whisper")

    # Auto on Apple Silicon when mlx_whisper is importable -> mlx
    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", types.ModuleType("mlx_whisper"))
    assert asr.resolve_asr("small", env={}, apple_silicon=True) == (
        "mlx",
        "mlx-community/whisper-small-mlx",
    )
    # Auto off Apple Silicon even when mlx_whisper is importable -> faster-whisper
    assert asr.resolve_asr("large", env={}, apple_silicon=False) == (
        "faster-whisper",
        "large-v3-turbo",
    )
    # Auto on Apple Silicon when mlx_whisper is NOT importable -> faster-whisper
    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", None)
    assert asr.resolve_asr("small", env={}, apple_silicon=True) == (
        "faster-whisper",
        "small",
    )

    # Unknown backend raises ValueError
    with pytest.raises(ValueError):
        asr.resolve_asr("small", env={"REEL_SCROLLER_ASR": "whisper-cpp"}, apple_silicon=False)


def test_transcribe_words_mlx_and_segment_mapping(monkeypatch, tmp_path):
    calls = []

    def fake_transcribe(video, **kwargs):
        calls.append((video, kwargs))
        return {
            "language": "en",
            "segments": [
                {
                    "start": 0.1234,
                    "end": 1.9876,
                    "text": "  Hello world  ",
                    "no_speech_prob": 0.1,
                    "words": [
                        {"word": " Hello ", "start": 0.1234, "end": 0.555},
                        {"word": "world", "start": 0.6, "end": 1.9876},
                    ],
                },
                {
                    "start": 2.0,
                    "end": 3.0,
                    "text": "noise segment",
                    "no_speech_prob": 0.6,  # >= 0.6 must be skipped
                    "words": [{"word": "noise", "start": 2.0, "end": 3.0}],
                },
                {
                    "start": 3.0,
                    "end": 4.0,
                    "text": "   ",  # whitespace-only must be skipped
                    "no_speech_prob": 0.0,
                    "words": [{"word": "blank", "start": 3.0, "end": 4.0}],
                },
                {
                    "start": 4.1,
                    "end": 5.256,
                    "text": "Second line",
                    "no_speech_prob": 0.2,
                    "words": [{"word": "Second", "start": 4.1, "end": 4.5}, {"word": "line ", "start": 4.5, "end": 5.256}],
                },
            ],
        }

    fake_mlx = types.SimpleNamespace(transcribe=fake_transcribe)
    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", fake_mlx)

    vid = tmp_path / "clip.mp4"
    out = asr.transcribe_words(vid, "mlx-community/whisper-small-mlx", backend="mlx")
    assert calls == [
        (
            str(vid),
            {
                "path_or_hf_repo": "mlx-community/whisper-small-mlx",
                "word_timestamps": True,
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.6,
                "verbose": None,
            },
        )
    ]
    assert out == {
        "language": "en",
        "text": "Hello world Second line",
        "segments": [
            {"start": 0.12, "end": 1.99, "text": "Hello world"},
            {"start": 4.1, "end": 5.26, "text": "Second line"},
        ],
        "words": [
            {"w": "Hello", "s": 0.12, "e": 0.56},
            {"w": "world", "s": 0.6, "e": 1.99},
            {"w": "Second", "s": 4.1, "e": 4.5},
            {"w": "line", "s": 4.5, "e": 5.26},
        ],
    }


def test_transcribe_words_faster_whisper_caching_and_cuda_fallback(monkeypatch, tmp_path, capsys):
    init_calls = []
    transcribe_calls = []

    class FakeWhisperModel:
        def __init__(self, model_name, device="cpu", compute_type="int8"):
            init_calls.append((model_name, device, compute_type))
            if device == "cuda":
                raise RuntimeError("CUDA driver library missing")

        def transcribe(self, video, **kwargs):
            transcribe_calls.append((video, kwargs))
            seg1 = types.SimpleNamespace(
                start=0.0,
                end=1.234,
                text=" Test faster ",
                no_speech_prob=0.05,
                words=[
                    types.SimpleNamespace(word=" Test", start=0.0, end=0.5),
                    types.SimpleNamespace(word="faster ", start=0.5, end=1.234),
                ],
            )
            seg_skip = types.SimpleNamespace(
                start=1.5,
                end=2.0,
                text="uh",
                no_speech_prob=0.85,
                words=[types.SimpleNamespace(word="uh", start=1.5, end=2.0)],
            )
            info = types.SimpleNamespace(language="en")
            return (s for s in [seg1, seg_skip]), info

    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setitem(__import__("sys").modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: 1))
    monkeypatch.delenv("REEL_SCROLLER_DEVICE", raising=False)
    monkeypatch.setattr(asr, "_decode_audio", lambda video: f"samples of {video}")

    vid = tmp_path / "reel.mp4"
    res1 = asr.transcribe_words(vid, "small", backend="faster-whisper")
    err = capsys.readouterr().err
    assert "cuda" in err.lower()
    assert init_calls == [("small", "cuda", "float16"), ("small", "cpu", "int8")]
    assert transcribe_calls[-1] == (
        f"samples of {vid}",
        {"word_timestamps": True, "condition_on_previous_text": False, "vad_filter": False},
    )
    assert res1 == {
        "language": "en",
        "text": "Test faster",
        "segments": [{"start": 0.0, "end": 1.23, "text": "Test faster"}],
        "words": [{"w": "Test", "s": 0.0, "e": 0.5}, {"w": "faster", "s": 0.5, "e": 1.23}],
    }

    # Second call on cpu uses the cached instance without re-instantiating WhisperModel
    monkeypatch.setenv("REEL_SCROLLER_DEVICE", "cpu")
    res2 = asr.transcribe_words(vid, "small", backend="faster-whisper")
    assert len(init_calls) == 2
    assert res2["text"] == "Test faster"


def test_ocr_backend_choice(monkeypatch):
    # Explicit env values
    assert ocr.ocr_backend(env={"REEL_SCROLLER_OCR": "apple"}) == "apple"
    assert ocr.ocr_backend(env={"REEL_SCROLLER_OCR": "rapidocr"}) == "rapidocr"
    assert ocr.ocr_backend(env={"REEL_SCROLLER_OCR": "none"}) == "none"
    with pytest.raises(ValueError):
        ocr.ocr_backend(env={"REEL_SCROLLER_OCR": "tesseract"})

    # Auto on darwin with Vision -> apple
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "Vision" else None)
    assert ocr.ocr_backend(env={}) == "apple"

    # Auto on darwin without Vision, with rapidocr -> rapidocr
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "rapidocr" else None)
    assert ocr.ocr_backend(env={}) == "rapidocr"

    # Auto on non-darwin even if Vision exists -> rapidocr (via rapidocr_onnxruntime fallback)
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: object() if name in ("Vision", "rapidocr_onnxruntime") else None,
    )
    assert ocr.ocr_backend(env={}) == "rapidocr"

    # Auto when nothing is available -> none
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert ocr.ocr_backend(env={}) == "none"


def test_ocr_image_rapidocr_v3_ordering_secrets_and_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("REEL_SCROLLER_OCR", "rapidocr")
    init_count = 0
    secret_key = "sk-proj-1234567890abcdefghijklmnop"

    class FakeRapidOCRv3:
        def __init__(self):
            nonlocal init_count
            init_count += 1

        def __call__(self, path):
            # Boxes out of order: bottom line first, then top-right, then top-left, plus a low-score box
            return types.SimpleNamespace(
                boxes=[
                    [[10, 80], [100, 80], [100, 95], [10, 95]],   # bottom line (top_y=80, left_x=10)
                    [[120, 10], [200, 10], [200, 25], [120, 25]], # top-right (top_y=10, left_x=120)
                    [[15, 10], [100, 10], [100, 25], [15, 25]],   # top-left (top_y=10, left_x=15)
                    [[10, 50], [90, 50], [90, 60], [10, 60]],     # low score -> dropped
                ],
                txts=[
                    f"Key: {secret_key}   " + ("word " * 120),
                    "world",
                    "Hello",
                    "ghost text",
                ],
                scores=[0.95, 0.88, 0.99, 0.49],
            )

    monkeypatch.setitem(__import__("sys").modules, "rapidocr", types.SimpleNamespace(RapidOCR=FakeRapidOCRv3))
    img = tmp_path / "frame.jpg"
    out1 = ocr.ocr_image(img)
    out2 = ocr.ocr_image(img)
    assert init_count == 1  # Engine created once
    assert out1 == out2
    assert out1.startswith("Hello world Key: [redacted] ")
    assert secret_key not in out1
    assert "ghost text" not in out1
    assert len(out1) == 500


def test_ocr_image_rapidocr_v3_empty_or_none(monkeypatch, tmp_path):
    monkeypatch.setenv("REEL_SCROLLER_OCR", "rapidocr")

    class FakeRapidOCRNone:
        def __call__(self, path):
            return types.SimpleNamespace(boxes=None, txts=None, scores=None)

    monkeypatch.setitem(__import__("sys").modules, "rapidocr", types.SimpleNamespace(RapidOCR=FakeRapidOCRNone))
    assert ocr.ocr_image(tmp_path / "empty.jpg") == ""


def test_ocr_image_rapidocr_onnxruntime_fallback_and_none(monkeypatch, tmp_path):
    monkeypatch.setenv("REEL_SCROLLER_OCR", "rapidocr")
    monkeypatch.setitem(__import__("sys").modules, "rapidocr", None)

    results_queue = [
        (
            [
                [[[5, 40], [80, 40], [80, 55], [5, 55]], "Bottom line", 0.91],
                [[[5, 10], [80, 10], [80, 25], [5, 25]], "Top   line", 0.5],
                [[[5, 25], [80, 25], [80, 35], [5, 35]], "Low score", 0.3],
            ],
            [0.01, 0.02],
        ),
        (None, None),
    ]

    class FakeOldRapidOCR:
        def __call__(self, path):
            return results_queue.pop(0)

    monkeypatch.setitem(
        __import__("sys").modules,
        "rapidocr_onnxruntime",
        types.SimpleNamespace(RapidOCR=FakeOldRapidOCR),
    )
    assert ocr.ocr_image(tmp_path / "f1.jpg") == "Top line Bottom line"
    assert ocr.ocr_image(tmp_path / "f2.jpg") == ""


def test_ocr_image_none_backend_logs_once(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("REEL_SCROLLER_OCR", "none")
    assert ocr.ocr_image(tmp_path / "1.jpg") == ""
    err1 = capsys.readouterr().err
    assert err1.strip() != ""
    assert ocr.ocr_image(tmp_path / "2.jpg") == ""
    err2 = capsys.readouterr().err
    assert err2 == ""


def test_ocr_image_apple_vision(monkeypatch, tmp_path):
    monkeypatch.setenv("REEL_SCROLLER_OCR", "apple")

    class FakeCandidate:
        def __init__(self, s):
            self._s = s

        def string(self):
            return self._s

    class FakeObs:
        def __init__(self, y, text):
            self._y = y
            self._text = text

        def boundingBox(self):
            return types.SimpleNamespace(origin=types.SimpleNamespace(y=self._y))

        def topCandidates_(self, n):
            return [FakeCandidate(self._text)] if self._text else []

    class FakeReq:
        def alloc(self):
            return self

        def init(self):
            return self

        def setRecognitionLevel_(self, level):
            pass

        def setUsesLanguageCorrection_(self, flag):
            pass

        def results(self):
            # In Apple Vision, higher origin.y is higher on screen
            return [FakeObs(0.2, "bottom"), FakeObs(0.8, "top"), FakeObs(0.5, "")]

    class FakeHandler:
        def alloc(self):
            return self

        def initWithURL_options_(self, url, opts):
            return self

        def performRequests_error_(self, reqs, err):
            return (True, None)

    fake_vision = types.SimpleNamespace(
        VNRecognizeTextRequest=FakeReq(),
        VNRequestTextRecognitionLevelAccurate=1,
        VNImageRequestHandler=FakeHandler(),
    )
    fake_foundation = types.SimpleNamespace(
        NSURL=types.SimpleNamespace(fileURLWithPath_=lambda p: p),
        NSDictionary=types.SimpleNamespace(dictionary=lambda: {}),
    )
    monkeypatch.setitem(__import__("sys").modules, "Vision", fake_vision)
    monkeypatch.setitem(__import__("sys").modules, "Foundation", fake_foundation)

    assert ocr.ocr_image(tmp_path / "mac.jpg") == "top bottom"



def test_faster_whisper_cuda_fails_during_decoding(monkeypatch, tmp_path, capsys):
    """Windows without cuDNN: the model loads on cuda, then decoding raises. Fall back to cpu and stay there."""
    inits = []

    class FakeWhisperModel:
        def __init__(self, model_name, device="cpu", compute_type="int8"):
            inits.append(device)
            self.device = device

        def transcribe(self, video, **kwargs):
            def gen():
                if self.device == "cuda":
                    raise RuntimeError("Could not load library cudnn_ops64_9.dll")
                yield types.SimpleNamespace(start=0, end=1, text="hi", no_speech_prob=0, words=[])
            return gen(), types.SimpleNamespace(language="en")

    monkeypatch.setattr(asr, "_MODELS", {})
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setenv("REEL_SCROLLER_DEVICE", "cuda")
    monkeypatch.setattr(asr, "_decode_audio", lambda video: None)
    assert asr.transcribe_words(tmp_path / "a.mp4", "small")["text"] == "hi"
    assert asr.transcribe_words(tmp_path / "b.mp4", "small")["text"] == "hi"
    assert inits == ["cuda", "cpu"]
    assert "cudnn" in capsys.readouterr().err


def test_ocr_image_engine_error_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("REEL_SCROLLER_OCR", "rapidocr")
    monkeypatch.setattr(ocr, "_LOGGED_NO_OCR", False)

    def broken():
        raise ImportError("No module named 'rapidocr'")
    monkeypatch.setattr(ocr, "_get_rapid_engine", broken)
    assert ocr.ocr_image(tmp_path / "f.jpg") == ""


def test_decode_audio_uses_ffmpeg_not_pyav(monkeypatch):
    """faster-whisper's PyAV decoder broke on PyAV 19 (metadata_errors); we decode with ffmpeg instead."""
    import struct
    calls = []

    class P:
        returncode, stdout, stderr = 0, struct.pack("<3h", 0, 16384, -32768), b""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return P()
    monkeypatch.setattr("subprocess.run", fake_run)
    a = asr._decode_audio("clip.mp4")
    assert calls[0][:2] == ["ffmpeg", "-nostdin"] and "16000" in calls[0] and "s16le" in calls[0]
    assert a.dtype.name == "float32" and list(a) == [0.0, 0.5, -1.0]
    P.returncode, P.stderr = 1, b"clip.mp4: Invalid data found"
    with pytest.raises(RuntimeError, match="Invalid data"):
        asr._decode_audio("clip.mp4")
