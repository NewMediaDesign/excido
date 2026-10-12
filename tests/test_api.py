"""API tests with a fake engine: no model download, no GPU."""

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from excido import server
from excido.engine import InferParams


def fake_predict(images, p: InferParams):
    masks = []
    for im in images:
        w, h = im.size
        yy, xx = np.mgrid[0:h, 0:w]
        masks.append((((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < (min(w, h) / 3) ** 2).astype(np.float32))
    info = {
        "model": p.model,
        "model_label": p.model,
        "device": "cpu",
        "precision": "fp32",
        "input_sizes": [[1024, 1024]] * len(images),
        "batches": 1,
        "timings_ms": {"load": 0.0, "preprocess": 1.0, "inference": 2.0, "resize": 0.5},
        "memory_peak_mb": None,
        "params": vars(p),
    }
    return masks, info


@pytest.fixture()
def client(monkeypatch, sample_image):
    monkeypatch.setattr(server.engine, "predict", fake_predict)
    return TestClient(server.app)


def test_system_models_examples(client):
    sysinfo = client.get("/api/system").json()
    assert sysinfo["devices"][-1]["id"] == "cpu"
    models = client.get("/api/models").json()
    assert len(models["models"]) == 16 and models["default"] == "general"
    groups = client.get("/api/examples").json()["groups"]
    assert groups[0]["items"][0]["source"] == "examples/official/Disc.png"


def test_segment_and_postprocess(client):
    r = client.post("/api/segment", data={"source": "examples/official/Disc.png", "params": json.dumps({"model": "general"})})
    assert r.status_code == 200, r.text
    run = r.json()
    assert (run["width"], run["height"]) == (320, 240)
    for key in ("original", "cutout", "mask", "overlay", "thumb"):
        assert client.get(run["urls"][key]).status_code == 200

    r = client.post(
        f"/api/runs/{run['id']}/postprocess",
        json={"mask_mode": "binary", "background": "color", "bg_color": "#00b140", "export_format": "webp"},
    )
    assert r.status_code == 200
    assert r.json()["cutout_file"] == "cutout.webp"

    r = client.post("/api/segment", data={"source": f"run:{run['id']}"})
    assert r.status_code == 200 and r.json()["name"] == "Disc"


def test_grow_and_shrink_via_postprocess(client):
    from io import BytesIO

    from PIL import Image

    run = client.post("/api/segment", data={"source": "examples/official/Disc.png"}).json()

    def kept(grow):
        desc = client.post(f"/api/runs/{run['id']}/postprocess", json={"grow": grow}).json()
        assert desc["post"]["grow"] == grow
        mask = Image.open(BytesIO(client.get(desc["urls"]["mask"]).content))
        return sum(mask.histogram()[128:])

    assert kept(-3) < kept(0) < kept(0.5) < kept(3)


def test_upload(client, sample_image):
    with open(sample_image, "rb") as f:
        r = client.post("/api/segment", files={"file": ("photo.png", f, "image/png")})
    assert r.status_code == 200 and r.json()["name"] == "photo"


@pytest.mark.parametrize(
    ("data", "code"),
    [
        ({"url": "ftp://example.com/x.jpg"}, "invalid_url"),
        ({"source": "examples/../../secret.txt"}, "example_not_found"),
        ({}, "no_image"),
        ({"source": "examples/official/Disc.png", "params": json.dumps({"model": "nope"})}, "unknown_model"),
    ],
)
def test_errors_have_codes(client, data, code):
    r = client.post("/api/segment", data=data)
    assert r.status_code == 422 and r.json()["code"] == code


def test_not_an_image(client):
    r = client.post("/api/segment", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 422 and r.json()["code"] == "invalid_image"


def test_batch(client):
    r = client.post(
        "/api/batch",
        data={"sources": json.dumps({"items": ["examples/official/Disc.png"] * 3}), "params": json.dumps({"batch_size": 2})},
    )
    assert r.status_code == 200
    job_id = r.json()["id"]
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert job["status"] == "done" and job["done"] == 3 and not job["errors"]
    assert sorted(r["index"] for r in job["results"]) == [0, 1, 2]
    assert client.get(job["zip"]).status_code == 200
