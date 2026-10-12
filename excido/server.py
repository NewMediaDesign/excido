"""Excido local web server (FastAPI).

Listens on 127.0.0.1 by default. Sync endpoints run in FastAPI's thread pool;
GPU access is serialized by the Engine lock. The MCP endpoint (``/mcp``) is mounted
on the same app, so the web UI and MCP clients share one model and one history.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import traceback
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import EXAMPLES_DIR, GALLERY_DIR, OUTPUTS_DIR, PACKAGE_DIR, __version__, catalog, store
from . import postprocess as pp
from .engine import Engine, EngineError, InferParams, resolve_device
from .runs import BACKGROUNDS_DIR, RunStore, export_psd, new_id, render, safe_stem

STATIC_DIR = PACKAGE_DIR / "static"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MAX_BYTES = 100 * 2**20  # per uploaded or fetched image

engine = Engine()
runs = RunStore()
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
mcp_server = None  # set at the bottom of this module when the MCP endpoint is enabled


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if mcp_server is not None:
        async with mcp_server.session_manager.run():
            yield
    else:
        yield


app = FastAPI(title="Excido", version=__version__, docs_url="/api/docs", redoc_url=None, lifespan=lifespan)
# Only answer requests addressed to this machine: blocks DNS-rebinding attacks, where a web
# page you visit resolves its own domain to 127.0.0.1 to drive the local app.
ALLOWED_HOSTS = [h.strip() for h in os.environ.get("EXCIDO_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",") if h.strip()]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


@app.middleware("http")
async def revalidate_ui(request, call_next):
    """Make browsers revalidate the UI files (cheap 304s), so an update never shows a stale interface."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


@app.exception_handler(EngineError)
def _engine_error(_, exc: EngineError):
    return JSONResponse(status_code=422, content=exc.to_dict())


# ---------------------------------------------------------------------- helpers


def open_image(data: bytes) -> tuple[Image.Image, str | None, bool]:
    """Return (image with EXIF orientation applied, format, rotated?)."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise EngineError("invalid_image", f"Not a readable image: {e}", {"detail": str(e)}) from None
    orientation = img.getexif().get(0x0112, 1)
    if orientation not in (1, None):
        return ImageOps.exif_transpose(img), img.format, True
    return img, img.format, False


def read_upload(file: UploadFile) -> bytes:
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise EngineError("too_large", "Image larger than 100 MB.", {"limit_mb": MAX_BYTES // 2**20})
    return data


def fetch_url(url: str) -> tuple[bytes, str]:
    if not url.lower().startswith(("http://", "https://")):
        raise EngineError("invalid_url", "Only http:// and https:// URLs are allowed.")
    try:
        with requests.get(url, stream=True, timeout=20, headers={"User-Agent": f"Excido/{__version__}"}) as r:
            r.raise_for_status()
            data = bytearray()
            for chunk in r.iter_content(1 << 16):
                data += chunk
                if len(data) > MAX_BYTES:
                    raise EngineError("too_large", "Image larger than 100 MB.", {"limit_mb": MAX_BYTES // 2**20})
    except requests.RequestException as e:
        raise EngineError("url_fetch_failed", f"Could not download the image: {e}", {"detail": str(e)}) from None
    name = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or "url"
    return bytes(data), name


def resolve_local(source: str) -> Path:
    """Path of an example ("examples/...", "gallery/...") or of a run's original ("run:<id>")."""
    if source.startswith("run:"):
        try:
            run = runs.get(source[4:])
        except KeyError:
            raise EngineError("run_not_found", "That run no longer exists.") from None
        return run.dir / run.original_file
    for prefix, base in (("examples/", EXAMPLES_DIR), ("gallery/", GALLERY_DIR)):
        if source.startswith(prefix):
            path = (base / source[len(prefix) :]).resolve()
            if path.is_file() and path.is_relative_to(base.resolve()):
                return path
    raise EngineError("example_not_found", f"Example not found: {source}", {"source": source})


def load_background(bg_id: str | None) -> Image.Image | None:
    if not bg_id:
        return None
    path = (BACKGROUNDS_DIR / f"{safe_stem(bg_id)}.png").resolve()
    return Image.open(path).convert("RGB") if path.exists() else None


def parse_json(text: str | None) -> dict:
    try:
        return json.loads(text) if text else {}
    except ValueError:
        raise HTTPException(400, "Invalid JSON parameters") from None


def new_job(kind: str, **extra) -> dict:
    job = {"id": new_id(), "kind": kind, "status": "queued", "progress": 0.0, "message": "", "created": time.time(), **extra}
    with jobs_lock:
        jobs[job["id"]] = job
    return job


def jobs_snapshot() -> list[dict]:
    with jobs_lock:
        return list(jobs.values())


# ------------------------------------------------------------------- API: system


@app.get("/api/system")
def api_system():
    return engine.system_info()


@app.get("/api/models")
def api_models():
    loaded = engine.key[0] if engine.key else None
    active = {j["model"]: j for j in jobs_snapshot() if j["kind"] == "download" and j["status"] in ("queued", "running")}
    out = []
    for spec in catalog.MODELS.values():
        d = spec.to_dict()
        d["downloaded"] = store.is_downloaded(spec)
        d["loaded"] = spec.id == loaded
        d["download_job"] = active.get(spec.id, {}).get("id")
        out.append(d)
    return {"models": out, "default": catalog.DEFAULT_ID, "core": list(catalog.CORE_IDS)}


@app.post("/api/models/{model_id}/download")
def api_download(model_id: str):
    if model_id not in catalog.MODELS:
        raise EngineError("unknown_model", f"Unknown model: {model_id}", {"model": model_id})
    spec = catalog.get(model_id)
    for j in jobs_snapshot():
        if j["kind"] == "download" and j["model"] == model_id and j["status"] in ("queued", "running"):
            return j
    job = new_job("download", model=model_id, total_bytes=spec.size_bytes)

    def work():
        job["status"] = "running"
        done = threading.Event()

        def monitor():
            while not done.wait(0.5):
                job["progress"] = min(0.99, store.downloaded_bytes(spec) / spec.size_bytes)

        threading.Thread(target=monitor, daemon=True).start()
        try:
            store.download(spec)
            job.update(status="done", progress=1.0, message=f"{spec.label} downloaded.")
        except Exception as e:  # noqa: BLE001 - reported to the UI
            job.update(status="error", message=f"Download failed: {e}")
        finally:
            done.set()

    threading.Thread(target=work, daemon=True).start()
    return job


@app.post("/api/models/unload")
def api_unload():
    with engine.lock:
        engine.unload()
    return engine.system_info()


@app.get("/api/examples")
def api_examples():
    def items(base: Path, prefix: str) -> list[dict]:
        if not base.exists():
            return []
        files = sorted(p for p in base.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())
        out = []
        for p in files:
            rel = p.relative_to(base).as_posix()
            # The official demo runs the "My_*" portraits with Matting HR at 2048.
            hint = {"model": "matting-hr"} if prefix == "examples/" and p.name.startswith("My_") else None
            out.append({"name": p.stem, "source": prefix + rel, "url": f"/{prefix}{rel}", "bytes": p.stat().st_size, "hint": hint})
        return out

    return {
        "groups": [
            {"id": "official", "items": items(EXAMPLES_DIR, "examples/")},
            {"id": "gallery", "folder": GALLERY_DIR.name, "items": items(GALLERY_DIR, "gallery/")},
        ]
    }


# ------------------------------------------------- services (shared with the MCP server)


def segment_bytes(data: bytes, name: str, params: dict | None, post: dict | None) -> dict:
    """Run the model on an encoded image, store the run, render it. Returns the run description."""
    post = post or {}
    ip = InferParams.from_dict(params)
    image, fmt, rotated = open_image(data)
    masks, info = engine.predict([image], ip)
    run = runs.create(name, image.convert("RGB"), None if rotated else data, fmt, masks[0], info)
    with engine.lock:
        return render(run, pp.PostParams.from_dict(post), info["device"], load_background(post.get("bg_image_id")))


def recompose_run(run_id: str, post: dict | None) -> dict:
    """Re-render a stored run with new post-processing settings (no model run)."""
    post = post or {}
    run = runs.get(run_id)  # KeyError if unknown
    device = resolve_device(run.info.get("device", "auto"))
    with engine.lock:
        return render(run, pp.PostParams.from_dict(post), device, load_background(post.get("bg_image_id")))


def save_background(data: bytes) -> str:
    """Store a background image for compositing; returns its id."""
    image, _, _ = open_image(data)
    BACKGROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    bg_id = new_id()
    image = image.convert("RGB")
    image.thumbnail((4096, 4096))
    image.save(BACKGROUNDS_DIR / f"{bg_id}.png")
    return bg_id


def output_file(url: str) -> Path:
    """Local path of a file served under /outputs/ (as returned in run descriptions)."""
    rel = url.split("?", 1)[0].removeprefix("/outputs/")
    path = (OUTPUTS_DIR / rel).resolve()
    if not path.is_relative_to(OUTPUTS_DIR.resolve()):
        raise ValueError(url)
    return path


# ---------------------------------------------------------------- API: inference


@app.post("/api/segment")
def api_segment(
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    source: str | None = Form(None),
    params: str | None = Form(None),
    post: str | None = Form(None),
):
    if file is not None:
        data, name = read_upload(file), file.filename or "upload"
    elif source and source.startswith("run:"):
        path = resolve_local(source)
        data, name = path.read_bytes(), runs.get(source[4:]).name
    elif source:
        path = resolve_local(source)
        data, name = path.read_bytes(), path.name
    elif url:
        data, name = fetch_url(url.strip())
    else:
        raise EngineError("no_image", "No image: upload a file, pick an example or enter a URL.")
    return segment_bytes(data, name, parse_json(params), parse_json(post))


@app.post("/api/runs/{run_id}/postprocess")
def api_postprocess(run_id: str, body: dict):
    try:
        return recompose_run(run_id, body)
    except KeyError:
        raise HTTPException(404, "Run not found") from None


def psd_for_run(run_id: str, lang: str = "en") -> tuple[Path, str]:
    """Build the layered PSD of a run with the post-processing it was last rendered with."""
    run = runs.get(run_id)  # KeyError if unknown
    device = resolve_device(run.info.get("device", "auto"))
    with engine.lock:
        path = export_psd(run, pp.PostParams.from_dict(run.post), device, load_background(run.post.get("bg_image_id")), lang)
    return path, f"{run.name}.psd"


@app.get("/api/runs/{run_id}/psd")
def api_psd(run_id: str, lang: str = "en"):
    """Layered Photoshop file: subject with an editable layer mask, hidden original, background."""
    try:
        path, filename = psd_for_run(run_id, lang)
    except KeyError:
        raise HTTPException(404, "Run not found") from None
    return FileResponse(path, media_type="image/vnd.adobe.photoshop", filename=filename)


@app.get("/api/runs")
def api_runs(limit: int = 30):
    return {"runs": runs.recent(limit)}


@app.post("/api/backgrounds")
def api_background(file: UploadFile = File(...)):
    bg_id = save_background(read_upload(file))
    return {"id": bg_id, "url": f"/outputs/backgrounds/{bg_id}.png", "name": file.filename}


# -------------------------------------------------------------------- API: batch


@app.post("/api/batch")
def api_batch(
    files: list[UploadFile] = File(default=[]),
    sources: str | None = Form(None),
    params: str | None = Form(None),
    post: str | None = Form(None),
):
    # Uploaded files first, then examples: the UI relies on this order for result indices.
    items: list[tuple[str, bytes]] = [(f.filename or f"img{i}", read_upload(f)) for i, f in enumerate(files)]
    for src in parse_json(sources).get("items", []) if sources else []:
        path = resolve_local(src)
        items.append((path.name, path.read_bytes()))
    if not items:
        raise EngineError("batch_empty", "The batch is empty.")

    ip = InferParams.from_dict(parse_json(params))
    post_dict = parse_json(post)
    pparams = pp.PostParams.from_dict(post_dict)
    job = new_job("batch", total=len(items), done=0, results=[], errors=[], zip=None)
    out_dir = OUTPUTS_DIR / "batch" / job["id"]

    def work():
        job["status"] = "running"
        out_dir.mkdir(parents=True, exist_ok=True)
        bg = load_background(post_dict.get("bg_image_id"))
        used: set[str] = set()
        t_start = time.perf_counter()
        for start in range(0, len(items), ip.batch_size):
            chunk = items[start : start + ip.batch_size]
            try:
                opened = [open_image(data) for _, data in chunk]
                masks, info = engine.predict([o[0] for o in opened], ip)
            except Exception as e:  # noqa: BLE001
                err = e.to_dict() if isinstance(e, EngineError) else {"error": str(e), "code": "internal"}
                job["errors"].extend({"index": start + k, "name": n, **err} for k, (n, _) in enumerate(chunk))
                job["done"] += len(chunk)
                job["progress"] = job["done"] / job["total"]
                continue
            for k, ((name, data), (image, fmt, rotated), mask) in enumerate(zip(chunk, opened, masks, strict=True)):
                index = start + k
                try:
                    run = runs.create(name, image.convert("RGB"), None if rotated else data, fmt, mask, info)
                    with engine.lock:
                        desc = render(run, pparams, info["device"], bg)
                    stem = run.name
                    while stem in used:
                        stem += "_1"
                    used.add(stem)
                    (out_dir / f"{stem}.{pparams.export_format}").write_bytes((run.dir / desc["cutout_file"]).read_bytes())
                    (out_dir / f"{stem}_mask.png").write_bytes((run.dir / "mask.png").read_bytes())
                    job["results"].append({**desc, "index": index})
                except Exception as e:  # noqa: BLE001
                    job["errors"].append({"index": index, "name": name, "error": str(e), "code": "internal"})
                job["done"] += 1
                job["progress"] = job["done"] / job["total"]
        zip_path = out_dir / "results.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for f in sorted(out_dir.iterdir()):
                if f != zip_path:
                    zf.write(f, f.name)
        job.update(
            status="done",
            zip=f"/outputs/batch/{job['id']}/results.zip",
            elapsed_s=round(time.perf_counter() - t_start, 2),
            message=f"{len(job['results'])} images processed, {len(job['errors'])} errors.",
        )

    def safe_work():
        try:
            work()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            job.update(status="error", message=str(e))

    threading.Thread(target=safe_work, daemon=True).start()
    return job


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


# ------------------------------------------------------------------ static files


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/examples", StaticFiles(directory=EXAMPLES_DIR), name="examples")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
if GALLERY_DIR.exists():
    app.mount("/gallery", StaticFiles(directory=GALLERY_DIR), name="gallery")


# ------------------------------------------------------------------ MCP endpoint
# Streamable HTTP at /mcp. Mounted last, at the root, so it only receives paths no other
# route claimed. Its own transport security accepts localhost requests only.
if os.environ.get("EXCIDO_MCP", "1") != "0":
    from .mcp_server import LocalBackend, build_server

    mcp_server = build_server(backend=LocalBackend())
    app.mount("/", mcp_server.streamable_http_app(streamable_http_path="/mcp"))
