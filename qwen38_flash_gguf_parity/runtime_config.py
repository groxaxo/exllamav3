"""Pure, CPU-only runtime configuration and reproducibility guards.

Extracted from exl3_qwen38_server.py so tests can validate configuration
semantics without importing the server (which boots a dispatch thread and
pulls FastAPI/torch at import time).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

VALID_MULTI_GPU = ("pipeline", "tensor")
VALID_CACHE_KV_BITS = (0, 2, 3, 4, 6, 8)


def validate_multi_gpu(raw: str) -> str:
    """Normalize and validate MULTI_GPU. Unknown values are startup errors:
    a benchmark topology must never be silently reinterpreted."""
    value = raw.strip().lower()
    if value not in VALID_MULTI_GPU:
        raise ValueError(f"MULTI_GPU must be 'pipeline' or 'tensor', got {raw!r}")
    return value


def validate_cache_kv_bits(raw: int) -> int:
    """Validate CACHE_KV_BITS. Invalid bit widths are startup errors rather
    than silent falls back to fp16 (same philosophy as MULTI_GPU)."""
    if raw not in VALID_CACHE_KV_BITS:
        raise ValueError(
            f"CACHE_KV_BITS must be one of {VALID_CACHE_KV_BITS}; got {raw}"
        )
    return raw


def resolve_ngram_stream(ngram_ram: bool) -> bool:
    """Effective ngram_stream_from_disk for a given NGRAM_RAM setting.

    NGRAM_RAM=1 -> RAM (stream off), NGRAM_RAM=0 -> disk (stream on),
    regardless of EXL3_NGRAM_STREAM, so A/B runs are deterministic.
    """
    return not ngram_ram


def check_expected_root(package_file: str, expected_root: str) -> Path:
    """Return the resolved package path, raising when it lives outside
    expected_root. Uses Path.is_relative_to so sibling directories such as
    '/root-old' cannot pass a prefix check."""
    pkg_path = Path(package_file).resolve()
    if expected_root:
        expected = Path(expected_root).resolve()
        if not pkg_path.is_relative_to(expected):
            raise RuntimeError(
                f"exllamav3 imported from {pkg_path}, expected under {expected}. "
                "Fix the env / editable install before benchmarking."
            )
    return pkg_path


def git_identity(package_file: str) -> tuple[str, str]:
    """(sha, clean|dirty|unknown) of the checkout containing package_file.

    Uses `git status --porcelain --untracked-files=all` so staged and
    untracked files count as dirty, not just unstaged edits.
    """
    try:
        root = Path(package_file).resolve().parents[1]
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, timeout=5,
        ).strip()
        status = subprocess.check_output(
            [
                "git", "-C", str(root),
                "status", "--porcelain=v1", "--untracked-files=all",
            ],
            text=True, timeout=5,
        )
        return sha, "dirty" if status.strip() else "clean"
    except Exception:
        return "unknown", "unknown"
