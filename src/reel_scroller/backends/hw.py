"""RAM and GPU detection, and the auto model size."""
from __future__ import annotations

import shutil
import subprocess

GB = 1024**3
LARGE_VRAM = 20 * GB  # an NVIDIA GPU this big runs the large vision model well
LARGE_APPLE_RAM = 32 * GB  # Apple Silicon shares RAM with the GPU


def total_ram_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        return 0


def gpu_vram_bytes() -> int:
    """Largest NVIDIA GPU memory via nvidia-smi, 0 when there is none or it fails."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return 0
    try:
        p = subprocess.run([exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
    except Exception:
        return 0
    if p.returncode != 0:
        return 0
    mib = [int(x) for x in p.stdout.split() if x.strip().isdigit()]
    return max(mib) * 1024**2 if mib else 0


def auto_size(ram_bytes: int, vram_bytes: int, apple_silicon: bool) -> str:
    if vram_bytes >= LARGE_VRAM or (apple_silicon and ram_bytes >= LARGE_APPLE_RAM):
        return "large"
    return "small"
