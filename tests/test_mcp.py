"""MCP server tests: workspace sandbox, tools (with a fake backend) and the HTTP endpoint."""

import io
import json
import os
from pathlib import Path

import anyio
import numpy as np
import pytest
from fastapi.testclient import TestClient
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image

from excido.mcp_server import OUTPUT_DIRNAME, Settings, Workspace, build_server


def png_bytes(mode="RGB", size=(64, 48), color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color if mode != "L" else 255).save(buf, format="PNG")
    return buf.getvalue()


class FakeBackend:
    """Stands in for the web app: returns a fixed run without touching a model."""

    def __init__(self):
        self.files = {"/outputs/runs/r1/cutout.png": png_bytes("RGBA", color=(1, 2, 3, 128)), "/outputs/runs/r1/mask.png": png_bytes("L")}
        self.calls = []

    def _desc(self, name, post):
        return {
            "id": "r1",
            "name": Path(name).stem,
            "width": 64,
            "height": 48,
            "cutout_file": "cutout.png",
            "post": {"background": post.get("background", "transparent")},
            "info": {
                "model": "general",
                "device": "cpu",
                "precision": "fp32",
                "input_sizes": [[1024, 1024]],
                "timings_ms": {"inference": 5},
            },
            "urls": {"cutout": "/outputs/runs/r1/cutout.png?v=1", "mask": "/outputs/runs/r1/mask.png?v=1"},
        }

    def segment(self, data, name, params, post):
        self.calls.append(("segment", name, params, post))
        return self._desc(name, post)

    def recompose(self, run_id, post):
        self.calls.append(("recompose", run_id, post))
        return self._desc("again.png", post)

    def background(self, data):
        return "bg1"

    def read_file(self, url):
        return self.files[url.split("?")[0]]

    def models(self):
        return [{"id": "general", "label": "General", "downloaded": True}]

    def system(self):
        return {"devices": [{"id": "cpu", "name": "CPU"}]}

    def download(self, model_id):
        self.calls.append(("download", model_id))

    def psd(self, run_id):
        self.calls.append(("psd", run_id))
        return b"8BPS-fake"


@pytest.fixture()
def ws(tmp_path) -> Workspace:
    root = tmp_path / "ws"
    (root / "sub").mkdir(parents=True)
    (root / "a.png").write_bytes(png_bytes())
    (root / "sub" / "b.jpg").write_bytes(png_bytes())
    (root / "notes.txt").write_text("hi")
    (tmp_path / "secret.png").write_bytes(png_bytes())
    return Workspace(root)


def call(server, name, **args):
    return anyio.run(lambda: server.call_tool(name, args))


# ----------------------------------------------------------------- sandbox


def test_workspace_accepts_inside_paths(ws):
    assert ws.resolve_image("a.png").name == "a.png"
    assert ws.resolve_image("sub/b.jpg").name == "b.jpg"
    assert ws.resolve_image(str(ws.root / "a.png")).name == "a.png"
    assert [ws.rel(p) for p in ws.images()] == ["a.png", "sub/b.jpg"]


@pytest.mark.parametrize("ref", ["../secret.png", "sub/../../secret.png", "notes.txt", "missing.png", ""])
def test_workspace_rejects(ws, ref):
    with pytest.raises(ToolError):
        ws.resolve_image(ref)


def test_workspace_rejects_absolute_outside(ws):
    with pytest.raises(ToolError, match="outside|not found"):
        ws.resolve_image(str(ws.root.parent / "secret.png"))


def test_workspace_rejects_symlink_escape(ws):
    link = ws.root / "link.png"
    try:
        os.symlink(ws.root.parent / "secret.png", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")
    with pytest.raises(ToolError, match="outside"):
        ws.resolve_image("link.png")


def test_outputs_are_never_listed_and_never_overwritten(ws):
    first = ws.unique("x.png")
    first.write_bytes(b"1")
    assert ws.unique("x.png").name == "x-1.png"
    assert first.parent.name == OUTPUT_DIRNAME
    assert all(OUTPUT_DIRNAME not in ws.rel(p) for p in ws.images())


# ------------------------------------------------------------------- tools


def test_remove_background_writes_into_workspace(ws):
    be = FakeBackend()
    server = build_server(Settings(ws.root), backend=be)
    result = call(server, "remove_background", image="a.png", background="#00ff00", preview=True)
    summary = result.structured_content
    assert summary["cutout"].startswith(f"{OUTPUT_DIRNAME}/a_")
    assert (ws.root / summary["cutout"]).exists() and (ws.root / summary["mask"]).exists()
    assert summary["foreground_ratio"] == 1.0
    assert [c.type for c in result.content] == ["text", "image"]
    assert be.calls[0][3]["background"] == "color"


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("remove_background", {"image": "../secret.png"}, "outside the Excido workspace"),
        ("remove_background", {"image": "https://example.com/x.jpg"}, "URLs are disabled"),
        ("remove_background", {"image": "a.png", "background": "../secret.png"}, "outside the Excido workspace"),
        ("remove_background_batch", {"folder": ".."}, "not found in the workspace"),
        ("download_model", {"model": "general"}, "downloads are disabled"),
    ],
)
def test_tools_refuse_outside_paths_and_disabled_features(ws, tool, args, message):
    server = build_server(Settings(ws.root), backend=FakeBackend())
    with pytest.raises(ToolError, match=message):
        call(server, tool, **args)


def test_psd_output_is_written_into_the_workspace(ws):
    be = FakeBackend()
    server = build_server(Settings(ws.root), backend=be)
    summary = call(server, "remove_background", image="a.png", psd=True).structured_content
    assert summary["psd"] == f"{OUTPUT_DIRNAME}/a_layers.psd"
    assert (ws.root / summary["psd"]).read_bytes().startswith(b"8BPS")


def test_downloads_when_enabled(ws):
    be = FakeBackend()
    server = build_server(Settings(ws.root, allow_downloads=True), backend=be)
    assert not call(server, "download_model", model="general").is_error
    assert ("download", "general") in be.calls


def test_batch_recompose_and_listing(ws):
    be = FakeBackend()
    server = build_server(Settings(ws.root), backend=be)
    listing = json.loads(call(server, "list_images").content[0].text)
    assert listing["images"] == ["a.png", "sub/b.jpg"]
    batch = json.loads(call(server, "remove_background_batch", folder="", recursive=True).content[0].text)
    assert batch["processed"] == 2 and not batch["errors"]
    again = call(server, "recompose", run_id="r1", background="blur:10")
    assert not again.is_error and be.calls[-1][0] == "recompose"
    info = json.loads(call(server, "workspace_info").content[0].text)
    assert info["urls_allowed"] is False and Path(info["workspace"]) == ws.root


# --------------------------------------------------------------- HTTP endpoint


def test_http_endpoint_and_host_check():
    from excido import server

    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
    }
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    with TestClient(server.app, base_url="http://127.0.0.1:7860") as client:
        r = client.post("/mcp", json=init, headers=headers)
        assert r.status_code == 200, r.text
        assert "excido" in r.text
        # DNS rebinding: a page on another domain pointing at 127.0.0.1 is refused.
        assert client.get("/api/system", headers={"Host": "evil.example"}).status_code == 400
        assert client.post("/mcp", json=init, headers={**headers, "Host": "evil.example"}).status_code == 400


def test_preview_shows_transparency():
    from excido.mcp_server import _preview

    img = _preview(png_bytes("RGBA", size=(1000, 500), color=(0, 0, 0, 0)))
    decoded = Image.open(io.BytesIO(img.data))
    assert decoded.size == (512, 256)
    assert len(set(np.asarray(decoded).reshape(-1, 3)[:, 0].tolist())) > 1  # checkerboard, not flat
