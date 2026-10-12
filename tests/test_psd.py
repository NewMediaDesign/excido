"""Layered PSD export: layers, layer mask, merged preview, API and MCP."""

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from psd_tools import PSDImage

from excido import postprocess as pp
from excido.psd_export import write_psd


@pytest.fixture()
def scene():
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (40, 60, 3), dtype=np.uint8))
    alpha = np.zeros((40, 60), np.float32)
    alpha[10:30, 15:45] = 1.0
    alpha[10:30, 14] = 0.5  # a soft edge
    foreground = Image.fromarray(np.full((40, 60, 3), 200, np.uint8))
    return image, alpha, foreground


def layers(psd):
    return [(layer.name, layer.visible, layer.has_mask()) for layer in psd]


def test_transparent_psd_has_editable_mask_and_transparent_preview(tmp_path, scene):
    image, alpha, fg = scene
    psd = PSDImage.open(write_psd(tmp_path / "a.psd", image, alpha, fg, None))
    assert layers(psd) == [("Original", False, False), ("Subject", True, True)]
    mask = np.asarray(psd[1].mask.topil())
    assert np.array_equal(mask, np.asarray(pp.alpha_to_image(alpha)))
    merged = psd.numpy()
    assert merged.shape[2] == 4
    assert merged[0, 0, 3] == 0 and merged[20, 30, 3] == 1  # transparent outside, opaque on the subject


def test_subject_layer_keeps_the_real_photo_outside_the_mask(tmp_path, scene):
    image, alpha, fg = scene
    psd = PSDImage.open(write_psd(tmp_path / "a.psd", image, alpha, fg, None))
    pixels = np.asarray(psd[1].topil().convert("RGB"))
    assert np.array_equal(pixels[0, 0], np.asarray(image)[0, 0])  # alpha 0: original
    assert np.array_equal(pixels[20, 30], [200, 200, 200])  # alpha 1: refined colors


def test_psd_with_background_and_italian_names(tmp_path, scene):
    image, alpha, fg = scene
    bg = Image.new("RGB", image.size, (0, 177, 64))
    psd = PSDImage.open(write_psd(tmp_path / "b.psd", image, alpha, fg, bg, lang="it"))
    assert layers(psd) == [("Sfondo", True, False), ("Originale", False, False), ("Soggetto", True, True)]
    assert psd.topil().convert("RGB").getpixel((0, 0)) == (0, 177, 64)


def test_psd_endpoint(monkeypatch, sample_image):
    from excido import server
    from tests.test_api import fake_predict

    monkeypatch.setattr(server.engine, "predict", fake_predict)
    client = TestClient(server.app)
    post = json.dumps({"background": "color", "bg_color": "#ffffff"})
    run = client.post("/api/segment", data={"source": "examples/official/Disc.png", "post": post}).json()
    r = client.get(f"/api/runs/{run['id']}/psd?lang=it")
    assert r.status_code == 200 and r.content.startswith(b"8BPS")
    assert 'filename="Disc.psd"' in r.headers["content-disposition"]
    psd = PSDImage.open(io.BytesIO(r.content))
    assert [layer.name for layer in psd] == ["Sfondo", "Originale", "Soggetto"]
    assert client.get("/api/runs/20000101-000000-dead/psd").status_code == 404
