import numpy as np
import pytest
import torch
from PIL import Image

from excido import postprocess as pp


def naive_box_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Reference cv2.blur((k, k)) with replicated borders, computed the slow way."""
    lo, hi = (k // 2 - 1, k // 2) if k % 2 == 0 else (k // 2, k // 2)
    padded = np.pad(x, ((lo, hi), (lo, hi)), mode="edge")
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = padded[i : i + k, j : j + k].mean()
    return out


@pytest.mark.parametrize("k", [1, 2, 5, 6, 9, 30])
def test_mean_blur_matches_reference(k):
    rng = np.random.default_rng(k)
    x = rng.random((17, 23)).astype(np.float32)
    got = pp._mean_blur(torch.from_numpy(x)[None, None], k)[0, 0].numpy()
    np.testing.assert_allclose(got, naive_box_mean(x, k), atol=1e-5)


def test_mask_ops():
    m = np.array([[0.2, 0.5, 0.9]], np.float32)
    np.testing.assert_allclose(pp.apply_mask_ops(m, pp.PostParams()), m)
    assert pp.apply_mask_ops(m, pp.PostParams(mask_mode="binary", threshold=0.5)).tolist() == [[0, 1, 1]]
    np.testing.assert_allclose(pp.apply_mask_ops(m, pp.PostParams(invert=True)), [[0.8, 0.5, 0.1]], atol=1e-6)


def dot(size=9):
    m = np.zeros((size, size), np.float32)
    m[size // 2, size // 2] = 1
    return m


def test_grow_shapes_are_exact_at_pixel_scale():
    # 1 px: 4-neighbour cross (the edge moves exactly 1 px along the axes)
    assert pp.grow_shrink(dot(), 1).sum() == 5
    # 2 px: cross then square -> octagon of 21 pixels (a 5x5 square minus its 4 corners)
    g2 = pp.grow_shrink(dot(), 2)
    assert g2.sum() == 21 and g2[2, 2] == 0 and g2[2, 4] == 1


def test_shrink_is_the_mirror_of_grow():
    sq = np.zeros((11, 11), np.float32)
    sq[2:9, 2:9] = 1  # 7x7
    assert pp.grow_shrink(sq, -1).sum() == 25  # 5x5
    assert pp.grow_shrink(sq, -10).sum() == 0
    assert pp.grow_shrink(sq, 0) is sq


def test_half_pixel_blends_neighbouring_radii():
    soft = np.random.default_rng(1).random((20, 20)).astype(np.float32)
    half = pp.grow_shrink(soft, 1.5)
    np.testing.assert_allclose(half, (pp.grow_shrink(soft, 1) + pp.grow_shrink(soft, 2)) / 2, atol=1e-6)
    assert pp.grow_shrink(soft, -0.5).mean() < soft.mean() < pp.grow_shrink(soft, 0.5).mean()


def test_binary_grow_uses_whole_pixels_and_stays_binary():
    out = pp.grow_shrink(dot(), 1.4, binary=True)
    assert set(np.unique(out)) == {0.0, 1.0} and out.sum() == 5


def test_grow_applies_after_invert():
    m = np.zeros((21, 21), np.float32)
    m[5:16, 5:16] = 1
    kept = pp.apply_mask_ops(m, pp.PostParams(invert=True, grow=2)).sum()
    assert kept > (1 - m).sum()  # growing an inverted selection enlarges the kept background


def test_params_are_clamped():
    p = pp.PostParams.from_dict({"threshold": 3, "refine_r1": 0, "background": "plaid", "export_format": "gif", "junk": 1, "grow": -500})
    assert (p.threshold, p.refine_r1, p.background, p.export_format, p.grow) == (1.0, 1, "transparent", "png", -100.0)


def test_refine_keeps_opaque_pixels():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (40, 50, 3), dtype=np.uint8))
    fg = pp.refine_foreground(img, np.ones((40, 50), np.float32), r1=9, r2=3, device="cpu")
    assert fg.size == img.size and fg.mode == "RGB"
    assert np.abs(np.asarray(fg, int) - np.asarray(img, int)).max() <= 1


@pytest.mark.parametrize(
    ("background", "mode"),
    [("transparent", "RGBA"), ("color", "RGB"), ("blur", "RGB"), ("image", "RGB")],
)
def test_compose_modes(background, mode):
    img = Image.new("RGB", (30, 20), (10, 20, 30))
    alpha = np.zeros((20, 30), np.float32)
    alpha[5:15, 5:25] = 1
    bg = Image.new("RGB", (7, 7), (0, 255, 0))
    out = pp.compose(img, img, alpha, pp.PostParams(background=background, bg_color="#ff0000"), bg)
    assert out.mode == mode and out.size == (30, 20)
    if background == "color":
        assert out.getpixel((0, 0)) == (255, 0, 0)
        assert out.getpixel((10, 10)) == (10, 20, 30)
    if background == "image":
        assert out.getpixel((0, 0)) == (0, 255, 0)


def test_encode_formats():
    img = Image.new("RGBA", (8, 8), (1, 2, 3, 128))
    for fmt, magic in (("png", b"\x89PNG"), ("webp", b"RIFF")):
        data = pp.encode(img, fmt)
        assert data.startswith(magic)
        assert Image.open(__import__("io").BytesIO(data)).mode == "RGBA"
