import pytest
from PIL import Image

from excido import catalog
from excido.engine import EngineError, InferParams, available_devices, effective_precision, resolve_device, resolve_size, snap32, to_rgb


def test_snap32_rounds_down_and_clamps():
    assert snap32(1024) == 1024
    assert snap32(1050) == 1024
    assert snap32(100) == 256
    assert snap32(10_000) == 4096


@pytest.mark.parametrize(
    ("model", "params", "img", "expected"),
    [
        ("general", {}, (3000, 2000), (1024, 1024)),
        ("general-hr", {}, (500, 500), (2048, 2048)),
        ("general-lite-2k", {}, (800, 600), (2560, 1440)),
        ("general", {"resolution": "custom", "width": 1000, "height": 700}, (10, 10), (992, 672)),
        ("general", {"resolution": "original", "max_side": 1536}, (3000, 2000), (1536, 1024)),
        ("general-dynamic", {"max_side": 2048}, (3577, 2163), (2048, 1216)),
        ("general-dynamic", {"max_side": 2048}, (640, 480), (640, 480)),
    ],
)
def test_resolve_size(model, params, img, expected):
    p = InferParams.from_dict({"model": model, **params})
    assert resolve_size(catalog.get(model), p, img) == expected


def test_params_validation():
    p = InferParams.from_dict({"precision": "fp8", "device": "tpu", "batch_size": 99, "max_side": 5})
    assert (p.precision, p.device, p.batch_size, p.max_side) == ("fp16", "auto", 8, 256)
    with pytest.raises(EngineError) as e:
        InferParams.from_dict({"model": "nope"})
    assert e.value.code == "unknown_model"


def test_devices():
    devices = available_devices()
    assert devices[-1] == "cpu"
    assert resolve_device("auto") == devices[0]
    assert resolve_device("cpu") == "cpu"
    assert effective_precision("cpu", "fp16") == "fp32"
    assert effective_precision("mps", "bf16") == "fp16"


def test_to_rgb_flattens_transparency_on_white():
    img = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
    assert to_rgb(img).getpixel((0, 0)) == (255, 255, 255)
    assert to_rgb(Image.new("L", (2, 2), 7)).mode == "RGB"


def test_catalog_consistency():
    assert set(catalog.CORE_IDS) <= set(catalog.MODELS)
    for m in catalog.MODELS.values():
        assert len(m.revision) == 40
        assert m.group in ("general", "matting", "benchmark")
        assert m.native is None or all(v % 32 == 0 for v in m.native)
