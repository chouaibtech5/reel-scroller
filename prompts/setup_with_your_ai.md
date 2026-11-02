# Send this to your AI coding assistant

Copy everything below the line into Claude Code, Codex or a similar assistant that can run commands on your machine.

---

Set up reel-scroller on this machine and run it on a small sample. Work in the repo folder you were opened in (clone it if needed). Follow `AGENTS.md` and `README.md`.

1. Detect the operating system, RAM and GPU:
   - Windows (PowerShell): check RAM with `(Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum / 1GB` and GPU with `nvidia-smi` (if present).
   - Linux (bash): check RAM with `free -h` and GPU with `nvidia-smi` (if present).
   - macOS: check architecture with `uname -m` (arm64 for Apple Silicon) and RAM with `sysctl hw.memsize`.
   - Explain the model choice: `--model-size auto` selects `large` only with an NVIDIA GPU of 20 GB VRAM or more, or Apple Silicon with 32 GB RAM or more. Otherwise it defaults to `small`.
2. Install what is missing:
   - Windows: `winget install Gyan.FFmpeg`, `winget install astral-sh.uv`, install Ollama from https://ollama.com.
   - Linux: `sudo apt install ffmpeg`, `curl -LsSf https://astral.sh/uv/install.sh | sh`, `curl -fsSL https://ollama.com/install.sh | sh`.
   - macOS: `brew install ffmpeg uv`, install Ollama from https://ollama.com (or use MLX).
   - Pull the default vision model: `ollama pull qwen2.5vl:7b`.
   - Run `uv sync`. On Apple Silicon, use `uv sync --extra mac`. With an NVIDIA GPU, use `uv sync --extra cuda`.
   - Run `uv run pytest` and show me the result.
3. Smoke test with no network and no cost:
   - Ask me for one mp4 I own.
   - Run `uv run reel-scroller study --local <file> --out-root ./out`.
   - Summarize `out/study/<name>.json` in five lines.
   - The first run downloads speech and OCR models, which can take time. Tell me before you start that it will.
4. Saved reels:
   - Explain how to download my Instagram data (README section "Getting your saved reels").
   - Wait for me to put `saved_posts.json` (or HTML export) somewhere.
   - Run `uv run reel-scroller export <file> -o urls.tsv` and tell me how many reels it found.
5. Apify:
   - I will create an account and an API token myself.
   - Tell me to set the token in my own shell:
     - PowerShell: `$env:APIFY_TOKEN = "your_token_here"`
     - Bash: `export APIFY_TOKEN="your_token_here"`
   - Never ask me to paste the token into chat, and never write it to a file.
6. Show the dry run and confirm cost:
   - Run `uv run reel-scroller study --input urls.tsv --out-root ./out` to show what would be processed.
   - Report the estimated cost at about $0.002 per reel.
   - Do NOT pass `--run` until I say yes to that specific cost.
   - Once approved, run with `--run --limit 5` first, and show me the results before doing the rest.
7. Rules:
   - Never log in to Instagram for me.
   - Never use my browser cookies.
   - Never commit my data.
   - Keep all analysis local.
