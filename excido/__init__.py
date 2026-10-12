"""Excido — local, open-source background removal powered by BiRefNet.

Importing this package configures where data lives. It must therefore be imported
before ``transformers`` / ``huggingface_hub`` (every entry point does so).

Data directory (models, examples, outputs):
  * ``EXCIDO_HOME`` if set;
  * otherwise the source checkout (the folder that contains ``pyproject.toml``);
  * otherwise ``~/.excido`` (e.g. when installed as a regular package).
"""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

PACKAGE_DIR = Path(__file__).resolve().parent


def _default_root() -> Path:
    if os.environ.get("EXCIDO_HOME"):
        return Path(os.environ["EXCIDO_HOME"]).expanduser().resolve()
    if (PACKAGE_DIR.parent / "pyproject.toml").exists():
        return PACKAGE_DIR.parent
    return Path.home() / ".excido"


ROOT = _default_root()
MODELS_DIR = ROOT / "models"
EXAMPLES_DIR = ROOT / "examples"
OUTPUTS_DIR = ROOT / "outputs"
# Extra folder of your own test images, shown as a second gallery in the UI.
GALLERY_DIR = Path(os.environ.get("EXCIDO_GALLERY") or ROOT / "demo").expanduser().resolve()

# Keep the Hugging Face cache (remote code modules included) inside the data directory.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Apple Silicon: run any op that lacks a Metal kernel on the CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
