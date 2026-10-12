"""Model downloads and on-disk state.

Each variant lives in a readable folder, e.g. ``models/BiRefNet_HR/``, holding the
weights (``model.safetensors``), the model code (``birefnet.py``,
``BiRefNet_config.py``), ``config.json``, the original model card (``README.md``)
and a ``.revision`` file with the downloaded commit.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from . import MODELS_DIR
from .catalog import ModelSpec

FILES = ["birefnet.py", "BiRefNet_config.py", "config.json", "model.safetensors", "README.md"]


def model_dir(spec: ModelSpec) -> Path:
    return MODELS_DIR / spec.repo.split("/", 1)[1].replace("-", "_")


def is_downloaded(spec: ModelSpec) -> bool:
    d = model_dir(spec)
    rev = d / ".revision"
    return rev.exists() and rev.read_text(encoding="utf-8").strip() == spec.revision and all((d / f).exists() for f in FILES)


def downloaded_bytes(spec: ModelSpec) -> int:
    """Bytes currently in the model folder (partial files included)."""
    d = model_dir(spec)
    if not d.exists():
        return 0
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())


def download(spec: ModelSpec, force: bool = False) -> Path:
    """Download (or complete) the variant at its pinned revision. Idempotent."""
    from huggingface_hub import snapshot_download

    d = model_dir(spec)
    if is_downloaded(spec) and not force:
        return d
    d.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=spec.repo, revision=spec.revision, local_dir=d, allow_patterns=FILES)
    missing = [f for f in FILES if not (d / f).exists()]
    if missing:
        raise RuntimeError(f"Incomplete download of {spec.repo}: missing {missing}")
    (d / ".revision").write_text(spec.revision, encoding="utf-8")
    return d


def delete(spec: ModelSpec) -> None:
    d = model_dir(spec)
    if d.exists():
        shutil.rmtree(d)
