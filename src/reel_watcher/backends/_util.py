"""Small helpers shared by the backends."""
from __future__ import annotations

import platform
import sys


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64")
