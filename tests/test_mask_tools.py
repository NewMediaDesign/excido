"""Feather, brush strokes (with undo/redo), history management."""

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from excido import postprocess as pp
from excido import server
from tests.test_api import fake_predict

# ------------------------------------------------------------------- feather


def test_feather_softens_edges_and_keeps_shape():
    m = np.zeros((80, 80), np.float32)
    m[20:60, 20:60] = 1
    out = pp.feather(m, 3)
    assert out.shape == m.shape
    assert out[40, 40] > 0.99 and out[2, 2] < 0.01  # centre and far background untouched
    assert 0.2 < out[40, 20] < 0.8  # the edge became soft
    assert abs(out.mean() - m.mean()) < 0.01
    assert pp.feather(m, 0) is m


def test_large_feather_uses_the_fast_path():
    m = np.zeros((300, 300), np.float32)
    m[100:200, 100:200] = 1
    out = pp.feather(m, 30)
    assert out.shape == m.shape and 0 <= out.min() and out.max() <= 1
    assert out[150, 150] > out[150, 105] > out[150, 60]


# --------------------------------------------------------------------- brush


def test_add_stroke_paints_a_disk():
    out = pp.paint_strokes(np.zeros((100, 100), np.float32), [{"mode": "add", "size": 20, "hardness": 1, "points": [[50, 50]]}])
    assert out[50, 50] == 1 and out[50, 75] == 0
    assert abs((out > 0.5).sum() - np.pi * 10**2) < 40


def test_erase_and_order_of_strokes():
    alpha = np.ones((60, 60), np.float32)
    line = [[10, 30], [50, 30]]
    erased = pp.paint_strokes(alpha, [{"mode": "erase", "size": 10, "hardness": 1, "points": line}])
    assert erased[30, 30] == 0 and erased[5, 30] == 1
    back = pp.paint_strokes(
        alpha,
        [
            {"mode": "erase", "size": 10, "hardness": 1, "points": line},
            {"mode": "add", "size": 10, "hardness": 1, "points": line},
        ],
    )
    assert back[30, 30] == 1  # later strokes win


def test_soft_brush_has_a_falloff_and_strokes_outside_are_ignored():
    soft = pp.paint_strokes(np.zeros((60, 60), np.float32), [{"mode": "add", "size": 30, "hardness": 0, "points": [[30, 30]]}])
    assert soft[30, 30] > 0.9 and 0 < soft[30, 40] < soft[30, 33]
    outside = pp.paint_strokes(np.zeros((10, 10), np.float32), [{"mode": "add", "size": 4, "points": [[500, 500]]}])
    assert outside.sum() == 0


# ------------------------------------------------------------------- the API


@pytest.fixture()
def client(monkeypatch, sample_image):
    monkeypatch.setattr(server.engine, "predict", fake_predict)
    return TestClient(server.app)


def kept(client, desc):
    mask = Image.open(io.BytesIO(client.get(desc["urls"]["mask"]).content))
    return sum(mask.histogram()[128:])


def test_edits_undo_redo_clear_and_rerun(client):
    run = client.post("/api/segment", data={"source": "examples/official/Disc.png"}).json()
    base = kept(client, run)
    stroke = {"mode": "add", "size": 40, "hardness": 1, "points": [[20, 20], [60, 30]]}

    added = client.post(f"/api/runs/{run['id']}/edits", json={"op": "add", "stroke": stroke}).json()
    assert added["edits_count"] == 1 and kept(client, added) > base

    undone = client.post(f"/api/runs/{run['id']}/edits", json={"op": "undo"}).json()
    assert undone["edits_count"] == 0 and undone["redo_count"] == 1 and kept(client, undone) == base

    redone = client.post(f"/api/runs/{run['id']}/edits", json={"op": "redo"}).json()
    assert redone["edits_count"] == 1 and kept(client, redone) == kept(client, added)

    # Strokes survive a re-run of the model on the same image, and reach the PSD.
    again = client.post("/api/segment", data={"source": f"run:{run['id']}"}).json()
    assert again["edits_count"] == 1
    assert client.get(f"/api/runs/{again['id']}/psd").status_code == 200

    cleared = client.post(f"/api/runs/{run['id']}/edits", json={"op": "clear"}).json()
    assert cleared["edits_count"] == 0 and kept(client, cleared) == base


@pytest.mark.parametrize("body", [{"op": "add", "stroke": {"points": []}}, {"op": "add", "stroke": {"points": "x"}}, {"op": "explode"}])
def test_bad_edits_are_rejected(client, body):
    run = client.post("/api/segment", data={"source": "examples/official/Disc.png"}).json()
    assert client.post(f"/api/runs/{run['id']}/edits", json=body).status_code == 400


def test_feather_through_postprocess(client):
    run = client.post("/api/segment", data={"source": "examples/official/Disc.png"}).json()
    desc = client.post(f"/api/runs/{run['id']}/postprocess", json={"feather": 4}).json()
    mask = np.asarray(Image.open(io.BytesIO(client.get(desc["urls"]["mask"]).content)))
    assert desc["post"]["feather"] == 4 and len(np.unique(mask)) > 20  # soft, many grey levels


def test_delete_and_clear_history(client):
    a = client.post("/api/segment", data={"source": "examples/official/Disc.png"}).json()
    b = client.post("/api/segment", data={"source": "examples/official/Disc.png"}).json()
    assert client.get("/api/storage").json()["runs"] >= 2

    assert client.delete(f"/api/runs/{a['id']}").status_code == 200
    assert client.post(f"/api/runs/{a['id']}/postprocess", json={}).status_code == 404
    assert client.delete(f"/api/runs/{a['id']}").status_code == 404
    assert client.delete("/api/runs/../../etc").status_code in (404, 405)

    assert client.delete("/api/runs").json()["deleted"] >= 1
    assert client.get("/api/runs").json()["runs"] == []
    assert client.post(f"/api/runs/{b['id']}/postprocess", json={}).status_code == 404
    assert json.loads(client.get("/api/storage").content)["runs"] == 0
