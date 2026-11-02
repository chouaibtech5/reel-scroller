# Agent guide for reel-scroller

reel-scroller analyzes a user's saved Instagram reels with local models. This is the cross-platform port
(Windows, Linux, macOS) of jakeb144/reel-watcher (MIT, by Jacob Bruce), which was Apple Silicon only.
Read `README.md` for behavior and the output schema, and `docs/CONTRACT.md` for the backend interfaces.

## Stack

Python 3.11+, managed with uv. ffmpeg/ffprobe on PATH. Local backends:
vision model via Ollama (default), any OpenAI-compatible local server, or MLX on Apple Silicon;
speech to text via faster-whisper (or mlx-whisper on Mac); OCR via RapidOCR (or Apple Vision on Mac).

## Run

```bash
uv sync
uv run pytest                                                   # must pass before and after any change
uv run reel-scroller study --local clip.mp4 --out-root ./out     # no network except the local model server, no cost
uv run reel-scroller study --input urls.tsv --out-root ./out     # dry run (default)
uv run reel-scroller study --input urls.tsv --out-root ./out --run --limit 3
```

## Code map (`src/reel_scroller/`)

| File | Role |
|---|---|
| `cli.py` | `reel-scroller study|export|advice` dispatcher |
| `study.py` | The pipeline: Apify fetch, frame sampling, OCR, dedupe, contact-sheet VLM call, whisper, giveaway detection, archive, JSON writing, `main()` |
| `model.py` | Model-size choice, `parse_json`, `RETRY_HINT`; delegates to `backends/` |
| `backends/vision.py` | Ollama, OpenAI-compatible and MLX vision backends (`make_vision`) |
| `backends/asr.py` | `resolve_asr`, `transcribe_words` (faster-whisper, mlx-whisper) |
| `backends/ocr.py` | `ocr_backend`, `ocr_image` (RapidOCR, Apple Vision) |
| `backends/hw.py` | RAM and GPU detection |
| `media.py` | ffprobe/ffmpeg helpers, scene cuts, Instagram URL parsing, preflight |
| `ig_export.py` | Instagram data export (saved_posts JSON/HTML) to URL list |
| `advice_library.py` | Optional renderer: advice library JSON to one static HTML page |

Tests live in `tests/`; they use no network and no models (backends are faked with monkeypatch).

## Conventions

- Match the existing style: short functions, type hints, `from __future__ import annotations`,
  one-line docstrings, terse comments only where the why is not obvious.
- Import heavy libraries (faster_whisper, rapidocr, onnxruntime, mlx_*, pyobjc) lazily inside functions.
- Must work on Windows: use `pathlib`, never shell strings, decode subprocess output as UTF-8 with
  `errors="replace"`, never assume `/tmp`, `brew`, `sysctl` or a POSIX shell.
- No em dash characters anywhere (a test enforces it in Python sources).

## Rules

- Never spend money without the user's explicit OK. `--run` calls Apify (about $0.002 per reel). Always
  show the dry run and the count first, start with `--limit 3`, keep the per-batch cap.
- `APIFY_TOKEN` is read from the environment only. Never write it to a file, a log, a URL or a commit.
- Never log in to Instagram, use browser cookies, or automate the Instagram UI. Reel URLs come from the
  user's official data export (`reel-scroller export`) or a plain URL list.
- Do not commit media, frames, transcripts or study output. No real reel data in tests or examples.
- Keep analysis local. Do not add a hosted-LLM or paid-API dependency. The `openai` backend is for local
  OpenAI-compatible servers only.
- Keep the original author's credit (LICENSE, README) intact.
- Do not publish or push anything without the owner's approval.

## Extend

- New field in the verdict: edit `SYNTH_PROMPT` in `study.py`, then the schema block in `README.md`.
- New backend: implement the interface in `docs/CONTRACT.md` and add a test with a fake.
- Add a test for every pure function you change.
