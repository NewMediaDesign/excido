"""Layered PSD export: the cutout becomes an editable layer mask.

Layers, bottom to top:
    Background  (only when the background is a color, a blur or an image)
    Original    the untouched photo, hidden, for reference
    Subject     the photo with refined edge colors + a layer mask holding the alpha

With a transparent background the document is RGBA and its merged preview keeps the
transparency, so file browsers and thumbnails show the cutout correctly. Photoshop,
Affinity, GIMP and Photopea rebuild the image from the layers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import postprocess as pp

LAYER_NAMES = {
    "en": {"subject": "Subject", "original": "Original", "background": "Background"},
    "it": {"subject": "Soggetto", "original": "Originale", "background": "Sfondo"},
}


def write_psd(
    path: Path,
    image: Image.Image,
    alpha: np.ndarray,
    foreground: Image.Image,
    background: Image.Image | None,
    lang: str = "en",
) -> Path:
    """Write the layered PSD to ``path``.

    image: original RGB; alpha: float32 HxW in [0, 1]; foreground: RGB with refined
    edge colors (or ``image`` itself); background: RGB or None for transparency.
    """
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer

    names = LAYER_NAMES.get(lang, LAYER_NAMES["en"])
    transparent = background is None
    psd = PSDImage.new("RGBA" if transparent else "RGB", image.size)
    if transparent:
        # Merged preview with real transparency: composite without a matte color, and
        # flag the extra channel as document transparency (negative layer count).
        if hasattr(psd, "_background_color"):
            psd._background_color = None
        if hasattr(psd, "_merged_alpha"):
            psd._merged_alpha = True
    else:
        PixelLayer.frompil(background.convert("RGB"), psd, names["background"])

    original = PixelLayer.frompil(image.convert("RGB"), psd, names["original"])
    original.visible = False

    subject = PixelLayer.frompil(pp.subject_pixels(image, foreground, alpha), psd, names["subject"])
    subject.create_mask(pp.alpha_to_image(alpha))

    path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(path)
    return path
