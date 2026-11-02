"""On-screen text recognition backends (RapidOCR and Apple Vision)."""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import threading

from . import _util

SECRET_RE = re.compile(
    r"\b(?:sk|pk|ghp|xox[a-z]|AKIA|AIza)[-_A-Za-z0-9]{12,}\b|\b[A-Fa-f0-9]{32,}\b|\b[A-Za-z0-9+/_-]{40,}\b"
)

_LOCK = threading.Lock()
_RAPID_ENGINE: tuple[object, object] | None = None
_LOGGED_NO_OCR = False


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, AttributeError):
        return sys.modules.get(name) is not None


def ocr_backend(env: dict | None = None) -> str:
    """Return 'apple', 'rapidocr', or 'none' based on REEL_SCROLLER_OCR and installed packages."""
    env = os.environ if env is None else env
    choice = (env.get("REEL_SCROLLER_OCR") or "auto").strip().lower()
    if choice in ("apple", "rapidocr", "none"):
        return choice
    if choice != "auto":
        raise ValueError(f"unknown OCR backend: {choice!r}")
    if sys.platform == "darwin" and _has_module("Vision"):
        return "apple"
    if _has_module("rapidocr") or _has_module("rapidocr_onnxruntime"):
        return "rapidocr"
    return "none"


def _parse_rapidocr(raw) -> list[tuple[float, float, str]]:
    """Normalize RapidOCR v2/v3 or rapidocr_onnxruntime output into (top_y, left_x, text) tuples."""
    if raw is None:
        return []
    if hasattr(raw, "txts") or hasattr(raw, "boxes"):
        boxes = getattr(raw, "boxes", None)
        txts = getattr(raw, "txts", None)
        scores = getattr(raw, "scores", None)
        if boxes is None or txts is None or len(txts) == 0:
            return []
        if scores is None:
            scores = [1.0] * len(txts)
        entries = zip(boxes, txts, scores)
    else:
        if isinstance(raw, tuple) and len(raw) == 2 and (raw[0] is None or not isinstance(raw[1], str)):
            raw = raw[0]
        if not raw:
            return []
        entries = raw
    out: list[tuple[float, float, str]] = []
    for box, txt, score in entries:
        if score is not None and float(score) < 0.5:
            continue
        if txt is None or not str(txt).strip():
            continue
        top_y = float(min(pt[1] for pt in box))
        left_x = float(min(pt[0] for pt in box))
        out.append((top_y, left_x, str(txt)))
    return out


def _get_rapid_engine():
    global _RAPID_ENGINE
    try:
        from rapidocr import RapidOCR as cls
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR as cls
    if _RAPID_ENGINE is None or _RAPID_ENGINE[0] is not cls:
        _RAPID_ENGINE = (cls, cls())
    return _RAPID_ENGINE[1]


def _apple_ocr_lines(path) -> list[str]:
    import Vision
    from Foundation import NSDictionary, NSURL
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(True)
    h = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
        NSURL.fileURLWithPath_(str(path)), NSDictionary.dictionary()
    )
    ok = h.performRequests_error_([req], None)
    if not (ok[0] if isinstance(ok, tuple) else ok):
        return []
    obs = sorted(req.results() or [], key=lambda o: -o.boundingBox().origin.y)
    return [str(c[0].string()) for o in obs if (c := o.topCandidates_(1))]


def ocr_image(path) -> str:
    """Recognize text top-to-bottom (then left-to-right), redact secrets, collapse whitespace, cap at 500 chars."""
    global _LOGGED_NO_OCR
    backend = ocr_backend()
    if backend == "none":
        if not _LOGGED_NO_OCR:
            _LOGGED_NO_OCR = True
            _util.log("OCR unavailable (no backend); on_screen_text stays empty")
        return ""
    try:
        with _LOCK:
            if backend == "apple":
                lines = _apple_ocr_lines(path)
            else:
                engine = _get_rapid_engine()
                items = _parse_rapidocr(engine(str(path)))
                lines = [txt for _, _, txt in sorted(items, key=lambda x: (x[0], x[1]))]
    except Exception as e:  # one unreadable frame (or a broken install) must not stop the reel
        if not _LOGGED_NO_OCR:
            _LOGGED_NO_OCR = True
            _util.log(f"OCR failed ({type(e).__name__}: {e}); on_screen_text stays empty")
        return ""
    if not lines:
        return ""
    txt = SECRET_RE.sub("[redacted]", " ".join(lines))
    return re.sub(r"\s+", " ", txt).strip()[:500]
