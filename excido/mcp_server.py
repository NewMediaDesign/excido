"""Excido MCP server: background removal tools for Claude and other MCP clients.

Two ways to run it:

* inside the web app (``excido serve``) as a Streamable HTTP endpoint at
  ``http://127.0.0.1:7860/mcp``: it shares the app's model and history, so a single
  process owns the GPU and what an agent does shows up in the web UI;
* ``excido mcp``: a stdio server for clients that launch local commands (e.g. Claude
  Desktop). If the web app is running it forwards the work to it, otherwise it loads
  the model in its own process.

Sandbox: the tools only read images inside one workspace folder (default
``~/Pictures/Excido``) and only write into ``<workspace>/excido-output/``.
Image URLs and model downloads are off unless explicitly enabled.
"""

from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import anyio
import requests
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from PIL import Image as PILImage
from pydantic import Field

from . import __version__, catalog

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
OUTPUT_DIRNAME = "excido-output"
MAX_BYTES = 100 * 2**20

HINTS = {
    "download_model": "Download it with the download_model tool (if enabled) or run `excido download <model>`.",
    "lower_resolution": "Lower the resolution or max_side.",
    "use_fp16": "Use precision='fp16'.",
    "try_lite": "Try a Lite model (general-lite).",
    "lower_batch": "Process fewer images at once.",
    "enable_cpu_fallback": "Retry with device='cpu' (much slower).",
}

INSTRUCTIONS = """\
Excido removes image backgrounds locally with BiRefNet models.
Work only with images inside the Excido workspace folder (see workspace_info); paths are
relative to it. Results are written to its excido-output/ subfolder.
Typical flow: list_images -> remove_background (preview=true to look at the result) ->
recompose to change background/threshold without re-running the model.
Model choice: 'general' for most images, 'general-hr' for large photos and fine detail,
'matting' / 'matting-hr' for hair, fur and transparency, 'portrait' for people,
'general-lite' when speed or memory matters."""


# --------------------------------------------------------------------------- settings


def default_root() -> Path:
    return Path(os.environ.get("EXCIDO_MCP_ROOT") or Path.home() / "Pictures" / "Excido").expanduser()


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    root: Path
    allow_urls: bool = False
    allow_downloads: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        return cls(default_root(), _flag("EXCIDO_MCP_ALLOW_URLS"), _flag("EXCIDO_MCP_ALLOW_DOWNLOADS"))


class Workspace:
    """The only folder the MCP tools may read from (and, below it, write to)."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.out = self.root / OUTPUT_DIRNAME

    def ensure(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)

    def _inside(self, p: Path) -> bool:
        return p == self.root or p.is_relative_to(self.root)

    def resolve_image(self, ref: str) -> Path:
        """Resolve a user-supplied path; symlinks and '..' can't escape the workspace."""
        if not ref or "\x00" in ref:
            raise ToolError("Give the image path relative to the workspace folder.")
        p = Path(ref).expanduser()
        p = p if p.is_absolute() else self.root / p
        try:
            p = p.resolve(strict=True)
        except (OSError, RuntimeError):
            raise ToolError(f"Image not found in the workspace: {ref}") from None
        if not self._inside(p):
            raise ToolError(f"Access denied: '{ref}' is outside the Excido workspace ({self.root}).")
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            raise ToolError(f"Not an image file: {ref}")
        return p

    def resolve_folder(self, ref: str) -> Path:
        p = (self.root / ref).resolve() if ref else self.root
        if not self._inside(p) or not p.is_dir():
            raise ToolError(f"Folder not found in the workspace: {ref}")
        return p

    def rel(self, p: Path) -> str:
        return p.resolve().relative_to(self.root).as_posix()

    def images(self, folder: str = "", recursive: bool = True) -> list[Path]:
        base = self.resolve_folder(folder)
        it = base.rglob("*") if recursive else base.iterdir()
        out = []
        for p in sorted(it):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and OUTPUT_DIRNAME not in p.relative_to(self.root).parts:
                out.append(p)
        return out

    def unique(self, name: str) -> Path:
        self.ensure()
        stem, suffix = os.path.splitext(name)
        p, i = self.out / name, 1
        while p.exists():
            p = self.out / f"{stem}-{i}{suffix}"
            i += 1
        return p


# --------------------------------------------------------------------------- backends


def _engine_error_to_tool_error(err: dict) -> ToolError:
    msg = err.get("error") or "Excido error."
    hints = [HINTS.get(h, h) for h in err.get("hints", [])]
    return ToolError(msg + (" " + " ".join(hints) if hints else ""))


class LocalBackend:
    """Runs in this process, sharing the web app's engine and history when mounted in it."""

    def __init__(self) -> None:
        from . import server  # heavy import (torch): only when actually needed

        self.s = server

    def _call(self, fn, *args):
        from .engine import EngineError

        try:
            return fn(*args)
        except EngineError as e:
            raise _engine_error_to_tool_error(e.to_dict()) from None

    def segment(self, data: bytes, name: str, params: dict, post: dict) -> dict:
        return self._call(self.s.segment_bytes, data, name, params, post)

    def recompose(self, run_id: str, post: dict) -> dict:
        try:
            return self._call(self.s.recompose_run, run_id, post)
        except KeyError:
            raise ToolError(f"Unknown run_id: {run_id}") from None

    def background(self, data: bytes) -> str:
        return self._call(self.s.save_background, data)

    def read_file(self, url: str) -> bytes:
        return self.s.output_file(url).read_bytes()

    def psd(self, run_id: str) -> bytes:
        path, _ = self.s.psd_for_run(run_id)
        return path.read_bytes()

    def models(self) -> list[dict]:
        return self.s.api_models()["models"]

    def system(self) -> dict:
        return self.s.engine.system_info()

    def download(self, model_id: str) -> None:
        from . import store

        store.download(catalog.get(model_id))


class HttpBackend:
    """Forwards the work to a running Excido web app (one process owns the GPU)."""

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")

    def _req(self, method: str, path: str, **kw) -> Any:
        r = requests.request(method, self.base + path, timeout=kw.pop("timeout", 3600), **kw)
        if r.status_code == 422:
            raise _engine_error_to_tool_error(r.json())
        if r.status_code == 404:
            raise ToolError(r.json().get("detail", "Not found"))
        r.raise_for_status()
        return r

    def segment(self, data: bytes, name: str, params: dict, post: dict) -> dict:
        form = {"params": json.dumps(params), "post": json.dumps(post)}
        return self._req("POST", "/api/segment", files={"file": (name, data)}, data=form).json()

    def recompose(self, run_id: str, post: dict) -> dict:
        return self._req("POST", f"/api/runs/{run_id}/postprocess", json=post).json()

    def background(self, data: bytes) -> str:
        return self._req("POST", "/api/backgrounds", files={"file": ("background.png", data)}).json()["id"]

    def read_file(self, url: str) -> bytes:
        return self._req("GET", url).content

    def psd(self, run_id: str) -> bytes:
        return self._req("GET", f"/api/runs/{run_id}/psd").content

    def models(self) -> list[dict]:
        return self._req("GET", "/api/models").json()["models"]

    def system(self) -> dict:
        return self._req("GET", "/api/system").json()

    def download(self, model_id: str) -> None:
        job = self._req("POST", f"/api/models/{model_id}/download").json()
        while job["status"] in ("queued", "running"):
            time.sleep(1)  # runs in a worker thread
            job = self._req("GET", f"/api/jobs/{job['id']}").json()
        if job["status"] != "done":
            raise ToolError(job.get("message") or "Download failed.")


def running_app_url(port: int | None = None) -> str | None:
    """URL of a running Excido web app on this machine, if any."""
    port = port or int(os.environ.get("EXCIDO_PORT", "7860"))
    url = f"http://127.0.0.1:{port}"
    try:
        r = requests.get(url + "/api/system", timeout=1.5)
        if r.ok and "devices" in r.json():
            return url
    except (requests.RequestException, ValueError):
        pass
    return None


def pick_backend() -> LocalBackend | HttpBackend:
    if not _flag("EXCIDO_MCP_STANDALONE"):
        url = running_app_url()
        if url:
            return HttpBackend(url)
    return LocalBackend()


# --------------------------------------------------------------------------- helpers


def _fetch_url(url: str) -> tuple[bytes, str]:
    if not url.lower().startswith(("http://", "https://")):
        raise ToolError("Only http:// and https:// URLs are allowed.")
    try:
        with requests.get(url, stream=True, timeout=20, headers={"User-Agent": f"Excido/{__version__}"}) as r:
            r.raise_for_status()
            data = bytearray()
            for chunk in r.iter_content(1 << 16):
                data += chunk
                if len(data) > MAX_BYTES:
                    raise ToolError("Image larger than 100 MB.")
    except requests.RequestException as e:
        raise ToolError(f"Could not download the image: {e}") from None
    return bytes(data), url.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or "url"


def _resolution(value: str) -> dict:
    v = (value or "auto").strip().lower()
    if v in ("auto", "original"):
        return {"resolution": v}
    try:
        w, h = (int(x) for x in v.split("x"))
    except ValueError:
        raise ToolError("resolution must be 'auto', 'original' or WIDTHxHEIGHT (e.g. 1024x1024).") from None
    return {"resolution": "custom", "width": w, "height": h}


def _checker(size: tuple[int, int], cell: int = 12) -> PILImage.Image:
    w, h = size
    tile = PILImage.new("RGB", (cell * 2, cell * 2), (235, 235, 238))
    tile.paste((205, 206, 210), (0, 0, cell, cell))
    tile.paste((205, 206, 210), (cell, cell, cell * 2, cell * 2))
    bg = PILImage.new("RGB", size)
    for y in range(0, h, cell * 2):
        for x in range(0, w, cell * 2):
            bg.paste(tile, (x, y))
    return bg


def _preview(data: bytes, max_side: int = 512) -> Image:
    """Small JPEG of a result; transparency is shown on a checkerboard."""
    img = PILImage.open(io.BytesIO(data))
    img.thumbnail((max_side, max_side))
    if img.mode in ("RGBA", "LA"):
        bg = _checker(img.size)
        bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").getchannel("A"))
        img = bg
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return Image(data=buf.getvalue(), format="jpeg")


def _foreground_ratio(mask_png: bytes) -> float:
    m = PILImage.open(io.BytesIO(mask_png)).convert("L")
    hist = m.histogram()
    return round(sum(hist[128:]) / max(1, sum(hist)), 4)


def _result(summary: dict, preview: Image | None) -> CallToolResult:
    content: list = [TextContent(type="text", text=json.dumps(summary, indent=2))]
    if preview is not None:
        content.append(preview.to_image_content())
    return CallToolResult(content=content, structured_content=summary)


# --------------------------------------------------------------------------- server

Model = Annotated[str, Field(description="Model id, see list_models (e.g. general, general-hr, matting, portrait, general-lite).")]
Resolution = Annotated[
    str, Field(description="'auto' (model's native size), 'original' (image proportions, capped by max_side) or WIDTHxHEIGHT.")
]
MaxSide = Annotated[int, Field(description="Long-side cap for 'original' resolution and dynamic models.", ge=256, le=4096)]
Precision = Annotated[str, Field(description="fp16 (fast, default), bf16 or fp32.")]
Device = Annotated[str, Field(description="auto, cuda, mps or cpu.")]
MaskMode = Annotated[str, Field(description="'soft' keeps a continuous alpha (hair, transparency); 'binary' cuts at the threshold.")]
Threshold = Annotated[float, Field(description="Cut-off for binary masks (0-1).", ge=0, le=1)]
Grow = Annotated[
    float,
    Field(
        description="Expand (+) or contract (-) the selection by this many pixels (0.5 px steps). "
        "Use small values (1-3) to clean halos or recover thin edges.",
        ge=-100,
        le=100,
    ),
]
Background = Annotated[
    str,
    Field(
        description="'transparent', a color like '#ffffff', 'blur' / 'blur:30', "
        "or the path of a workspace image to place behind the subject."
    ),
]
OutputFormat = Annotated[str, Field(description="png or webp (lossless, keeps transparency).")]
Preview = Annotated[bool, Field(description="Also return a small JPEG preview of the result (costs tokens).")]
Psd = Annotated[bool, Field(description="Also write <name>_layers.psd: subject with an editable layer mask, hidden original, background.")]


def build_server(settings: Settings | None = None, backend: LocalBackend | HttpBackend | None = None) -> MCPServer:
    settings = settings or Settings.from_env()
    ws = Workspace(settings.root)
    state: dict[str, Any] = {"backend": backend}

    def be() -> LocalBackend | HttpBackend:
        if state["backend"] is None:
            state["backend"] = pick_backend()
        return state["backend"]

    mcp = MCPServer(name="excido", title="Excido", version=__version__, instructions=INSTRUCTIONS)

    def post_params(mask_mode, threshold, invert, grow, refine, background, output_format) -> dict:
        post = {
            "mask_mode": mask_mode,
            "threshold": threshold,
            "invert": invert,
            "grow": grow,
            "refine": refine,
            "export_format": output_format,
        }
        bg = (background or "transparent").strip()
        if bg == "transparent":
            post["background"] = "transparent"
        elif bg.startswith("#"):
            post.update(background="color", bg_color=bg)
        elif bg == "blur" or bg.startswith("blur:"):
            post.update(background="blur", bg_blur=int(bg.partition(":")[2] or 20))
        else:
            post.update(background="image", bg_image_id=be().background(ws.resolve_image(bg).read_bytes()))
        return post

    def load_input(image: str) -> tuple[bytes, str, str]:
        """(bytes, file name, how to report the source)."""
        if image.lower().startswith(("http://", "https://")):
            if not settings.allow_urls:
                raise ToolError("Image URLs are disabled. Put the image in the workspace, or start Excido with --mcp-allow-urls.")
            data, name = _fetch_url(image)
            return data, name, image
        path = ws.resolve_image(image)
        return path.read_bytes(), path.name, ws.rel(path)

    def save_outputs(desc: dict, source: str, preview: bool, psd: bool = False) -> CallToolResult:
        b = be()
        stem = Path(desc["name"]).stem
        transparent = desc["post"].get("background") == "transparent"
        ext = desc["cutout_file"].rsplit(".", 1)[-1]
        cutout = b.read_file(desc["urls"]["cutout"])
        mask = b.read_file(desc["urls"]["mask"])
        cut_path = ws.unique(f"{stem}_{'cutout' if transparent else 'composite'}.{ext}")
        mask_path = ws.unique(f"{stem}_mask.png")
        cut_path.write_bytes(cutout)
        mask_path.write_bytes(mask)
        info = desc.get("info", {})
        summary = {
            "run_id": desc["id"],
            "source": source,
            "cutout": ws.rel(cut_path),
            "mask": ws.rel(mask_path),
            "size": [desc["width"], desc["height"]],
            "model": info.get("model"),
            "device": info.get("device"),
            "precision": info.get("precision"),
            "input_size": (info.get("input_sizes") or [None])[0],
            "inference_ms": info.get("timings_ms", {}).get("inference"),
            "foreground_ratio": _foreground_ratio(mask),
            "workspace": str(ws.root),
        }
        if psd:
            psd_path = ws.unique(f"{stem}_layers.psd")
            psd_path.write_bytes(b.psd(desc["id"]))
            summary["psd"] = ws.rel(psd_path)
        return _result(summary, _preview(cutout) if preview else None)

    # ------------------------------------------------------------------ tools

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    async def workspace_info() -> dict:
        """Where Excido reads images from and writes results to, and which options are enabled."""
        ws.ensure()
        return {
            "workspace": str(ws.root),
            "output_folder": str(ws.out),
            "images": len(ws.images()),
            "urls_allowed": settings.allow_urls,
            "model_downloads_allowed": settings.allow_downloads,
        }

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    async def list_images(
        folder: Annotated[str, Field(description="Subfolder of the workspace; empty for the whole workspace.")] = "",
        recursive: bool = True,
        limit: Annotated[int, Field(ge=1, le=2000)] = 200,
    ) -> dict:
        """List the images in the Excido workspace (paths relative to it). Results in excido-output/ are excluded."""
        ws.ensure()
        files = ws.images(folder, recursive)
        return {"workspace": str(ws.root), "count": len(files), "images": [ws.rel(p) for p in files[:limit]]}

    @mcp.tool(annotations=ToolAnnotations(destructive_hint=False, idempotent_hint=False, open_world_hint=False))
    async def remove_background(
        image: Annotated[str, Field(description="Image path relative to the workspace (or an http(s) URL if enabled).")],
        model: Model = "general",
        resolution: Resolution = "auto",
        max_side: MaxSide = 2048,
        precision: Precision = "fp16",
        device: Device = "auto",
        mask_mode: MaskMode = "soft",
        threshold: Threshold = 0.5,
        invert: Annotated[bool, Field(description="Keep the background and remove the subject.")] = False,
        grow: Grow = 0.0,
        refine: Annotated[bool, Field(description="Clean background halos from soft edges (refine_foreground).")] = True,
        background: Background = "transparent",
        output_format: OutputFormat = "png",
        preview: Preview = False,
        psd: Psd = False,
    ) -> CallToolResult:
        """Remove the background of one image. Writes <name>_cutout.png (or _composite) and <name>_mask.png
        into excido-output/ and returns their paths, a run_id for recompose, timings and the share of
        foreground pixels. Use preview=true to see the result."""
        data, name, source = load_input(image)
        params = {"model": model, "max_side": max_side, "precision": precision, "device": device, **_resolution(resolution)}
        post = post_params(mask_mode, threshold, invert, grow, refine, background, output_format)

        def work():
            return save_outputs(be().segment(data, name, params, post), source, preview, psd)

        return await anyio.to_thread.run_sync(work)

    @mcp.tool(annotations=ToolAnnotations(destructive_hint=False, idempotent_hint=False, open_world_hint=False))
    async def remove_background_batch(
        images: Annotated[list[str] | None, Field(description="Image paths relative to the workspace.")] = None,
        folder: Annotated[
            str | None, Field(description="Or: process every image in this workspace subfolder ('' = whole workspace).")
        ] = None,
        recursive: bool = False,
        model: Model = "general",
        resolution: Resolution = "auto",
        max_side: MaxSide = 2048,
        precision: Precision = "fp16",
        device: Device = "auto",
        mask_mode: MaskMode = "soft",
        threshold: Threshold = 0.5,
        invert: bool = False,
        grow: Grow = 0.0,
        refine: bool = True,
        background: Background = "transparent",
        output_format: OutputFormat = "png",
        psd: Psd = False,
    ) -> dict:
        """Remove the background of several images with the same settings. Returns one entry per image."""
        if images:
            paths = [ws.resolve_image(i) for i in images]
        elif folder is not None:
            paths = ws.images(folder, recursive)
        else:
            raise ToolError("Pass either images or folder.")
        if not paths:
            raise ToolError("No images to process.")
        params = {"model": model, "max_side": max_side, "precision": precision, "device": device, **_resolution(resolution)}
        post = post_params(mask_mode, threshold, invert, grow, refine, background, output_format)

        def work():
            results, errors = [], []
            for p in paths:
                try:
                    desc = be().segment(p.read_bytes(), p.name, params, post)
                    results.append(save_outputs(desc, ws.rel(p), False, psd).structured_content)
                except ToolError as e:
                    errors.append({"image": ws.rel(p), "error": str(e)})
            return {"processed": len(results), "errors": errors, "results": results, "output_folder": str(ws.out)}

        return await anyio.to_thread.run_sync(work)

    @mcp.tool(annotations=ToolAnnotations(destructive_hint=False, idempotent_hint=False, open_world_hint=False))
    async def recompose(
        run_id: Annotated[str, Field(description="run_id returned by remove_background.")],
        mask_mode: MaskMode = "soft",
        threshold: Threshold = 0.5,
        invert: bool = False,
        grow: Grow = 0.0,
        refine: bool = True,
        background: Background = "transparent",
        output_format: OutputFormat = "png",
        preview: Preview = False,
        psd: Psd = False,
    ) -> CallToolResult:
        """Re-render a previous result with new mask/background settings without running the model again (fast)."""
        post = post_params(mask_mode, threshold, invert, grow, refine, background, output_format)

        def work():
            desc = be().recompose(run_id, post)
            return save_outputs(desc, f"run:{run_id}", preview, psd)

        return await anyio.to_thread.run_sync(work)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    async def list_models() -> dict:
        """The BiRefNet variants: what each is for, native resolution, size, and whether it is downloaded."""
        models = await anyio.to_thread.run_sync(lambda: be().models())
        keep = ("id", "label", "task", "notes", "native", "dynamic", "size_bytes", "downloaded", "loaded", "license")
        return {"models": [{k: m.get(k) for k in keep} for m in models], "downloads_allowed": settings.allow_downloads}

    @mcp.tool(annotations=ToolAnnotations(destructive_hint=False, idempotent_hint=True, open_world_hint=True))
    async def download_model(model: Model) -> dict:
        """Download a model from the official catalog (pinned Hugging Face revisions). Disabled unless enabled at startup."""
        if not settings.allow_downloads:
            raise ToolError(
                f"Model downloads are disabled. Ask the user to run `excido download {model}` "
                "or to start Excido with --mcp-allow-downloads."
            )
        if model not in catalog.MODELS:
            raise ToolError(f"Unknown model: {model}. Use list_models.")
        await anyio.to_thread.run_sync(lambda: be().download(model))
        return {"model": model, "downloaded": True}

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    async def system_info() -> dict:
        """Compute devices (CUDA / Apple Metal / CPU), memory and the model currently loaded."""
        return await anyio.to_thread.run_sync(lambda: be().system())

    return mcp


def run_stdio(settings: Settings | None = None) -> None:
    """Entry point of `excido mcp`."""
    build_server(settings).run("stdio")
