"""CPU-only tests for runtime_config: the reproducibility guards the Phase 0
gate relies on. No torch, no server import, no GPUs."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from runtime_config import (  # noqa: E402
    check_expected_root,
    git_identity,
    resolve_ngram_stream,
    validate_cache_kv_bits,
    validate_multi_gpu,
)


# --------------------------------------------------------------- MULTI_GPU

@pytest.mark.parametrize("raw,expected", [
    ("pipeline", "pipeline"),
    ("tensor", "tensor"),
    ("  PIPELINE ", "pipeline"),
    ("Tensor", "tensor"),
])
def test_multi_gpu_valid(raw, expected):
    assert validate_multi_gpu(raw) == expected


@pytest.mark.parametrize("raw", ["tp", "", "PIPE", "native", "layer-split", "0"])
def test_multi_gpu_invalid_raises(raw):
    with pytest.raises(ValueError, match="MULTI_GPU"):
        validate_multi_gpu(raw)


# --------------------------------------------------------- CACHE_KV_BITS

@pytest.mark.parametrize("bits", [0, 2, 3, 4, 6, 8])
def test_cache_kv_bits_valid(bits):
    assert validate_cache_kv_bits(bits) == bits


@pytest.mark.parametrize("bits", [5, 1, 7, -1, 16])
def test_cache_kv_bits_invalid_raises(bits):
    with pytest.raises(ValueError, match="CACHE_KV_BITS"):
        validate_cache_kv_bits(bits)


# ------------------------------------------------------------- NGRAM

def test_ngram_ram_true_disables_disk_stream():
    # NGRAM_RAM=1 -> resident RAM regardless of EXL3_NGRAM_STREAM
    assert resolve_ngram_stream(True) is False


def test_ngram_ram_zero_is_disk_even_when_ambient_env_disables_stream(monkeypatch):
    # NGRAM_RAM=0 + EXL3_NGRAM_STREAM=0 must still resolve to disk streaming:
    # the mapping must not depend on ambient environment variables.
    monkeypatch.setenv("EXL3_NGRAM_STREAM", "0")
    assert resolve_ngram_stream(False) is True


# ------------------------------------------------------ checkout guard

def _pkg(root: Path) -> str:
    return str(root / "exllamav3" / "__init__.py")


def test_expected_root_exact_passes(tmp_path):
    root = tmp_path / "fork"
    check_expected_root(_pkg(root), str(root))  # no raise


def test_expected_root_subdir_passes(tmp_path):
    root = tmp_path / "fork"
    check_expected_root(_pkg(root), str(tmp_path))  # ancestor root passes


def test_expected_root_sibling_suffix_fails(tmp_path):
    # "/x/fork-old" must NOT pass a check for "/x/fork" (startswith bug class)
    root = tmp_path / "fork"
    evil = tmp_path / "fork-old"
    with pytest.raises(RuntimeError, match="expected under"):
        check_expected_root(_pkg(evil), str(root))


def test_expected_root_empty_disables_guard(tmp_path):
    check_expected_root(_pkg(tmp_path / "anywhere"), "")


# ------------------------------------------------------ git identity

def _git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "exllamav3").mkdir(parents=True)
    (root / "exllamav3" / "__init__.py").write_text("")
    for args in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "init"],
    ):
        subprocess.run(args, cwd=root, check=True)
    return root


def test_git_identity_clean(tmp_path):
    root = _git_repo(tmp_path)
    sha, tree = git_identity(_pkg(root))
    assert tree == "clean"
    assert len(sha) == 40


def test_git_identity_unstaged_is_dirty(tmp_path):
    root = _git_repo(tmp_path)
    (root / "exllamav3" / "__init__.py").write_text("x = 1\n")
    _, tree = git_identity(_pkg(root))
    assert tree == "dirty"


def test_git_identity_staged_is_dirty(tmp_path):
    root = _git_repo(tmp_path)
    (root / "exllamav3" / "__init__.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    _, tree = git_identity(_pkg(root))
    assert tree == "dirty"


def test_git_identity_untracked_is_dirty(tmp_path):
    root = _git_repo(tmp_path)
    (root / "exllamav3" / "new_file.py").write_text("")
    _, tree = git_identity(_pkg(root))
    assert tree == "dirty"


def test_git_identity_non_repo_is_unknown(tmp_path):
    root = tmp_path / "plain"
    (root / "exllamav3").mkdir(parents=True)
    (root / "exllamav3" / "__init__.py").write_text("")
    sha, tree = git_identity(_pkg(root))
    assert (sha, tree) == ("unknown", "unknown")
