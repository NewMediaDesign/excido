"""Post-processing of the mask produced by BiRefNet.

Steps: threshold / invert, foreground color estimation (``refine_foreground``),
compositing over the chosen background, export.

``refine_foreground`` implements the blur-fusion foreground estimation of
M. Forte and F. Pitié, "Approximate Fast Foreground Colour Estimation" (ICIP 2021),
via Photoroom's "fast-foreground-estimation", as used in ``image_proc.py`` of the official BiRefNet
repository (https://github.com/ZhengPeng7/BiRefNet, MIT, Copyright (c) 2024 ZhengPeng).
This is a rewrite: a single PyTorch implementation (CUDA, MPS or CPU, no OpenCV),
O(1)-per-pixel box filters, and a configurable second-pass radius (fixed to 6 upstream).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

GROW_LIMIT = 100.0  # max |grow| in pixels
FEATHER_LIMIT = 100.0  # max feather radius in pixels
BRUSH_LIMIT = 2000  # max brush diameter in pixels


@dataclass
class PostParams:
    mask_mode: str = "soft"  # "soft" (continuous alpha) | "binary"
    threshold: float = 0.5
    invert: bool = False
    grow: float = 0.0  # pixels: > 0 expands the selection, < 0 contracts it
    feather: float = 0.0  # pixels: Gaussian softening of the mask edge
    refine: bool = True
    refine_r1: int = 90
    refine_r2: int = 6
    background: str = "transparent"  # "transparent" | "color" | "blur" | "image"
    bg_color: str = "#ffffff"
    bg_blur: int = 20
    export_format: str = "png"  # "png" | "webp"

    @classmethod
    def from_dict(cls, d: dict | None) -> PostParams:
        d = d or {}
        p = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        p.mask_mode = p.mask_mode if p.mask_mode in ("soft", "binary") else "soft"
        p.threshold = float(min(max(float(p.threshold), 0.0), 1.0))
        p.grow = float(min(max(float(p.grow), -GROW_LIMIT), GROW_LIMIT))
        p.feather = float(min(max(float(p.feather), 0.0), FEATHER_LIMIT))
        p.refine_r1 = int(min(max(int(p.refine_r1), 1), 255))
        p.refine_r2 = int(min(max(int(p.refine_r2), 1), 63))
        p.background = p.background if p.background in ("transparent", "color", "blur", "image") else "transparent"
        p.bg_blur = int(min(max(int(p.bg_blur), 1), 200))
        p.export_format = p.export_format if p.export_format in ("png", "webp") else "png"
        return p


# --------------------------------------------------------------------------- mask


def apply_mask_ops(mask: np.ndarray, p: PostParams, device: str = "cpu") -> np.ndarray:
    """mask: float32 HxW in [0, 1] -> alpha float32 HxW in [0, 1].

    Order: threshold (binary mode) -> invert -> grow/shrink -> feather, so growing
    always enlarges what is kept, whether that is the subject or (inverted) the
    background, and feathering softens the final edge. Brush strokes (see
    ``paint_strokes``) are applied afterwards, on top of all of this.
    """
    alpha = mask.astype(np.float32, copy=True)
    binary = p.mask_mode == "binary"
    if binary:
        alpha = (alpha >= p.threshold).astype(np.float32)
    if p.invert:
        alpha = 1.0 - alpha
    if p.grow:
        alpha = grow_shrink(alpha, p.grow, device, binary=binary)
    if p.feather:
        alpha = feather(alpha, p.feather, device)
    return alpha


def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


@torch.inference_mode()
def feather(alpha: np.ndarray, radius: float, device: str = "cpu") -> np.ndarray:
    """Soften the mask edge with a Gaussian blur of ``radius`` pixels (used as sigma).

    Large radii are computed on a downscaled mask and scaled back up: the result is
    smooth anyway, and it keeps big feathers fast on the CPU.
    """
    if radius <= 0:
        return alpha
    x = torch.from_numpy(np.ascontiguousarray(alpha, dtype=np.float32))[None, None].to(device)
    h, w = x.shape[-2:]
    factor = max(1, int(radius // 3))
    if factor > 1:
        x = F.interpolate(x, size=(max(1, h // factor), max(1, w // factor)), mode="area")
    k = _gaussian_kernel(radius / factor, x.device)
    r = k.numel() // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    if factor > 1:
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x[0, 0].clamp_(0, 1).cpu().numpy()


# ---------------------------------------------------------------------- brush strokes


def _segment_coverage(xs: np.ndarray, ys: np.ndarray, a, b, radius: float, hardness: float) -> np.ndarray:
    """Brush coverage (0-1) of pixel centers (xs, ys) for a round brush dragged from a to b."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 == 0:
        d = np.hypot(xs - ax, ys - ay)
    else:
        t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / length2, 0.0, 1.0)
        d = np.hypot(xs - (ax + t * dx), ys - (ay + t * dy))
    inner = radius * hardness
    # Fully painted inside the hard core, linear falloff to 0 at the brush edge.
    return np.clip((radius - d) / max(radius - inner, 0.5), 0.0, 1.0).astype(np.float32)


def paint_strokes(alpha: np.ndarray, strokes: list[dict]) -> np.ndarray:
    """Apply brush strokes in order. Each stroke: ``{"mode": "add" | "erase",
    "size": diameter_px, "hardness": 0..1, "points": [[x, y], ...]}`` in image pixels.

    Within one stroke the coverage is the maximum over its segments, so overlapping
    dabs don't build up; successive strokes compose like layers of paint.
    """
    if not strokes:
        return alpha
    out = alpha.astype(np.float32, copy=True)
    h, w = out.shape
    for st in strokes:
        pts = st.get("points") or []
        if not pts:
            continue
        radius = max(0.5, min(float(st.get("size", 20)), BRUSH_LIMIT) / 2)
        hardness = min(max(float(st.get("hardness", 0.8)), 0.0), 1.0)
        target = 0.0 if st.get("mode") == "erase" else 1.0
        arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        pad = radius + 1
        x0 = int(max(0, np.floor(arr[:, 0].min() - pad)))
        x1 = int(min(w, np.ceil(arr[:, 0].max() + pad)))
        y0 = int(max(0, np.floor(arr[:, 1].min() - pad)))
        y1 = int(min(h, np.ceil(arr[:, 1].max() + pad)))
        if x0 >= x1 or y0 >= y1:
            continue
        cov = np.zeros((y1 - y0, x1 - x0), np.float32)
        segments = list(zip(arr[:-1], arr[1:], strict=True)) if len(arr) > 1 else [(arr[0], arr[0])]
        for a, b in segments:
            # Only the pixels near this segment: its bounding box grown by the radius.
            sx0 = int(max(x0, np.floor(min(a[0], b[0]) - pad)))
            sx1 = int(min(x1, np.ceil(max(a[0], b[0]) + pad)))
            sy0 = int(max(y0, np.floor(min(a[1], b[1]) - pad)))
            sy1 = int(min(y1, np.ceil(max(a[1], b[1]) + pad)))
            if sx0 >= sx1 or sy0 >= sy1:
                continue
            ys, xs = np.mgrid[sy0:sy1, sx0:sx1]
            c = _segment_coverage(xs + 0.5, ys + 0.5, a, b, radius, hardness)
            view = cov[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0]
            np.maximum(view, c, out=view)
        region = out[y0:y1, x0:x1]
        region += (target - region) * cov
    return out


def _dilate_step(x: torch.Tensor, square: bool) -> torch.Tensor:
    """One-pixel grayscale dilation (max filter): 3x3 square or 4-neighbour cross."""
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    if square:
        return F.max_pool2d(p, 3, stride=1)
    vertical = torch.maximum(p[..., :-2, 1:-1], p[..., 2:, 1:-1])
    horizontal = torch.maximum(p[..., 1:-1, :-2], p[..., 1:-1, 2:])
    return torch.maximum(p[..., 1:-1, 1:-1], torch.maximum(vertical, horizontal))


@torch.inference_mode()
def grow_shrink(alpha: np.ndarray, px: float, device: str = "cpu", binary: bool = False) -> np.ndarray:
    """Expand (px > 0) or contract (px < 0) the selection by ``px`` pixels.

    Grayscale morphology on the soft alpha: soft edges (hair, transparency) keep
    their profile and move as a whole. Alternating cross / square steps build an
    octagon, close to a disk, so the edge moves by about the same distance in every
    direction. Fractional values blend the two nearest whole radii (half-pixel
    control); binary masks move by whole pixels and stay binary.
    """
    if binary:
        px = float(round(px))
    if px == 0:
        return alpha
    r = abs(px)
    whole = int(math.floor(r))
    frac = r - whole
    x = torch.from_numpy(np.ascontiguousarray(alpha, dtype=np.float32))[None, None].to(device)
    if px < 0:
        x = 1.0 - x  # erosion = complement of the dilation of the complement
    prev = x
    for i in range(whole + (1 if frac > 0 else 0)):
        prev = x
        x = _dilate_step(x, square=i % 2 == 1)  # cross first: 1 px moves exactly 1 px along the axes
    if frac > 0:
        x = prev + (x - prev) * frac
    if px < 0:
        x = 1.0 - x
    return x[0, 0].clamp_(0, 1).cpu().numpy()


def alpha_to_image(alpha: np.ndarray) -> Image.Image:
    return Image.fromarray(np.round(alpha * 255.0).astype(np.uint8), mode="L")


# ---------------------------------------------------------------- refine_foreground


def _box_sum_1d(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    """Sum over a window of ``k`` along ``dim`` (3 = W, 2 = H) with replicated borders.

    Uses cumulative sums: O(1) per pixel whatever ``k`` is (avg_pool2d costs O(k^2),
    which is very slow on CPU with r=90 on a 24 MP photo). Accumulates in float64
    where available (MPS has no float64, float32 is accurate enough there).
    """
    lo, hi = (k // 2 - 1, k // 2) if k % 2 == 0 else (k // 2, k // 2)
    n = x.shape[dim]
    pad = (lo, hi, 0, 0) if dim == 3 else (0, 0, lo, hi)
    acc = torch.float32 if x.device.type == "mps" else torch.float64
    c = torch.cumsum(F.pad(x, pad, mode="replicate").to(acc), dim=dim)
    c = F.pad(c, (1, 0, 0, 0) if dim == 3 else (0, 0, 1, 0))
    return (c.narrow(dim, k, n) - c.narrow(dim, 0, n)).float()


def _mean_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    """Equivalent of cv2.blur((k, k)) on [1, C, H, W] tensors (one channel at a time to save memory)."""
    out = [_box_sum_1d(_box_sum_1d(x[:, i : i + 1], k, 3), k, 2) for i in range(x.shape[1])]
    return torch.cat(out, dim=1) / float(k * k)


def _fb_blur_fusion(image, fg, bg, alpha, r):
    blurred_alpha = _mean_blur(alpha, r)
    blurred_fg = _mean_blur(fg * alpha, r) / (blurred_alpha + 1e-5)
    blurred_bg = _mean_blur(bg * (1 - alpha), r) / ((1 - blurred_alpha) + 1e-5)
    fg = blurred_fg + alpha * (image - alpha * blurred_fg - (1 - alpha) * blurred_bg)
    return fg.clamp(0, 1), blurred_bg


@torch.inference_mode()
def refine_foreground(image: Image.Image, alpha: np.ndarray, r1: int = 90, r2: int = 6, device: str = "cpu") -> Image.Image:
    """Estimate the "clean" foreground colors (removes background halos on soft edges).

    image: RGB; alpha: float32 HxW in [0, 1] with the same size.
    """
    img = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    a = torch.from_numpy(alpha.astype(np.float32))[None, None].to(device)
    fg, blur_bg = _fb_blur_fusion(img, img, img, a, r1)
    fg, _ = _fb_blur_fusion(img, fg, blur_bg, a, r2)
    out = (fg[0].permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(np.ascontiguousarray(out), mode="RGB")


# ---------------------------------------------------------------------- compositing


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return (255, 255, 255)


def _cover(bg: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale and center-crop to cover ``size`` (like CSS object-fit: cover)."""
    w, h = size
    scale = max(w / bg.width, h / bg.height)
    nw, nh = max(w, round(bg.width * scale)), max(h, round(bg.height * scale))
    bg = bg.convert("RGB").resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return bg.crop((left, top, left + w, top + h))


def compose(
    image: Image.Image,
    foreground: Image.Image,
    alpha: np.ndarray,
    p: PostParams,
    bg_image: Image.Image | None = None,
) -> Image.Image:
    """Final result: RGBA when the background is transparent, RGB otherwise."""
    mask = alpha_to_image(alpha)
    bg = background_image(image, p, bg_image)
    if bg is None:
        out = foreground.copy()
        out.putalpha(mask)
        return out
    # Image.composite = fg * a + bg * (1 - a) in C, ~5x faster than float numpy.
    return Image.composite(foreground, bg, mask)


def background_image(image: Image.Image, p: PostParams, bg_image: Image.Image | None = None) -> Image.Image | None:
    """The background behind the subject, or None when it is transparent."""
    if p.background == "color":
        return Image.new("RGB", image.size, _hex_to_rgb(p.bg_color))
    if p.background == "blur":
        return image.filter(ImageFilter.GaussianBlur(p.bg_blur))
    if p.background == "image" and bg_image is not None:
        return _cover(bg_image, image.size)
    return None


def subject_pixels(image: Image.Image, foreground: Image.Image, alpha: np.ndarray) -> Image.Image:
    """Pixels for an editable subject layer: refined colors where the mask shows them,
    the untouched original elsewhere, so painting the mask back in reveals the real photo."""
    if foreground is image:
        return image
    keep = (alpha > 0)[..., None]
    return Image.fromarray(np.where(keep, np.asarray(foreground), np.asarray(image)), mode="RGB")


def overlay(image: Image.Image, alpha: np.ndarray, color=(226, 59, 38), opacity: float = 0.55) -> Image.Image:
    """Mask tinted over the original (rubylith-style), to inspect the edges."""
    a = Image.fromarray(np.round(alpha * (255.0 * opacity)).astype(np.uint8), mode="L")
    return Image.composite(Image.new("RGB", image.size, color), image, a)


# --------------------------------------------------------------------------- export


def encode(img: Image.Image, fmt: str = "png") -> bytes:
    buf = io.BytesIO()
    if fmt == "webp":
        # Lossless at minimum effort: method=4 takes ~8 s on an 8 MP photo, method=0 ~0.2 s.
        img.save(buf, format="WEBP", lossless=True, quality=0, method=0)
    elif fmt == "jpeg":
        img.convert("RGB").save(buf, format="JPEG", quality=90)
    else:
        img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()
