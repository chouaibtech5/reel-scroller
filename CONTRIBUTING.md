# Contributing to reel-watcher

Thank you for helping improve reel-watcher.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and ffmpeg for your operating system.
2. Clone the repository and install dependencies:

```bash
uv sync
```

On macOS with Apple Silicon, use `uv sync --extra mac` to include MLX dependencies.
On systems with an NVIDIA GPU, use `uv sync --extra cuda` for CUDA runtime packages.

## Running tests

Run the test suite before and after making any changes:

```bash
uv run pytest
```

Tests use monkeypatched fakes. They do not download models, touch the network, or require running background services.

## Development rules

- **Keep analysis local**: Do not add hosted-LLM or paid-API dependencies. The `openai` backend is for local OpenAI-compatible servers only.
- **No real reel data**: Never commit reel media, frames, transcripts, or study output. Tests and examples must use synthetic fixtures only.
- **Protect credentials**: Never commit API keys or tokens. `APIFY_TOKEN` must be read from the environment only.
- **No Instagram logins**: Never automate Instagram login flows, scrape with browser cookies, or touch the Instagram UI.
- **No em dash characters**: Do not use em dash characters anywhere in code, tests, or documentation. Use commas, colons, hyphens, or parentheses instead.
- **Cross-platform compatibility**: Code must work on Windows, Linux, and macOS. Use `pathlib` for paths, decode subprocess outputs as UTF-8 with `errors="replace"`, and avoid shell-specific assumptions.
- **Test coverage**: Add a test with fakes for every pure function you change or add.
- **CI verification**: Pull requests must pass CI on Ubuntu, Windows, and macOS.
