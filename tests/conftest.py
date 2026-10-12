"""Test setup: an isolated data directory, so tests never touch your models or outputs."""

import os
import tempfile
from pathlib import Path

# Must happen before `excido` is imported anywhere.
_HOME = Path(tempfile.mkdtemp(prefix="excido-test-"))
os.environ["EXCIDO_HOME"] = str(_HOME)
os.environ["EXCIDO_GALLERY"] = str(_HOME / "gallery")
os.environ["EXCIDO_MCP_ROOT"] = str(_HOME / "workspace")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402


@pytest.fixture(scope="session")
def home() -> Path:
    return _HOME


@pytest.fixture(scope="session")
def sample_image(home: Path) -> Path:
    """A 320x240 image with a bright disc on a dark background, saved as an 'official example'."""
    yy, xx = np.mgrid[0:240, 0:320]
    disc = ((xx - 160) ** 2 + (yy - 120) ** 2) < 80**2
    rgb = np.zeros((240, 320, 3), np.uint8)
    rgb[..., 2] = 60
    rgb[disc] = (230, 200, 40)
    path = home / "examples" / "official" / "Disc.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path
