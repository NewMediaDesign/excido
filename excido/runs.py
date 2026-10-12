"""Runs: original image + raw mask + rendered results.

Each run lives in ``outputs/runs/<run_id>/``:
    original.<ext>   input image (EXIF orientation already applied)
    mask_raw.npy     soft mask from the model (float16, original size)
    mask.png         mask after threshold/invert (8-bit)
    cutout.png|webp  final result (RGBA when the background is transparent)
    overlay.jpg      mask tinted over the original
    thumb.png        thumbnail for the history
    meta.json        parameters, metrics, timings

The raw mask stays in memory (LRU), so changing the threshold, background or
refine settings never re-runs the model.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import OUTPUTS_DIR
from . import postprocess as pp
from .engine import empty_cache, is_oom

RUNS_DIR = OUTPUTS_DIR / "runs"
BACKGROUNDS_DIR = OUTPUTS_DIR / "backgrounds"
WEB_FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif", "BMP": "bmp"}
RUN_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4}")


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


def safe_stem(name: str) -> str:
    stem = Path(name or "image").stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:60] or "image"


@dataclass
class Run:
    id: str
    name: str
    image: Image.Image  # RGB, original size
    mask: np.ndarray  # float32 HxW in [0, 1]
    info: dict
    original_file: str
    post: dict = field(default_factory=dict)
    edits: list = field(default_factory=list)  # brush strokes, applied in order
    redo: list = field(default_factory=list)  # strokes removed by undo
    edits_rev: int = 0  # bumped on every edit, part of the foreground cache key
    fg_cache: dict = field(default_factory=dict)  # (r1, r2, alpha key) -> PIL RGB

    @property
    def dir(self) -> Path:
        return RUNS_DIR / self.id


class RunStore:
    def __init__(self, capacity: int = 12) -> None:
        self.capacity = capacity
        self._mem: OrderedDict[str, Run] = OrderedDict()
        self._lock = threading.Lock()

    def create(
        self,
        name: str,
        image: Image.Image,
        original_bytes: bytes | None,
        fmt: str | None,
        mask: np.ndarray,
        info: dict,
        edits: list | None = None,
    ) -> Run:
        run_id = new_id()
        d = RUNS_DIR / run_id
        d.mkdir(parents=True, exist_ok=True)
        ext = WEB_FORMATS.get(fmt or "")
        if original_bytes is not None and ext:
            original_file = f"original.{ext}"
            (d / original_file).write_bytes(original_bytes)
        else:  # format browsers can't show, or EXIF orientation was applied
            original_file = "original.jpg"
            image.save(d / original_file, quality=95)
        np.save(d / "mask_raw.npy", mask.astype(np.float16))
        run = Run(run_id, safe_stem(name), image, mask, info, original_file, edits=list(edits or []))
        self._put(run)
        return run

    def get(self, run_id: str) -> Run:
        with self._lock:
            if run_id in self._mem:
                self._mem.move_to_end(run_id)
                return self._mem[run_id]
        d = RUNS_DIR / run_id
        if not RUN_ID.fullmatch(run_id) or not (d / "meta.json").exists():
            raise KeyError(run_id)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        image = Image.open(d / meta["original_file"]).convert("RGB")
        mask = np.load(d / "mask_raw.npy").astype(np.float32)
        run = Run(run_id, meta["name"], image, mask, meta["info"], meta["original_file"], meta.get("post", {}), list(meta.get("edits", [])))
        self._put(run)
        return run

    def _put(self, run: Run) -> None:
        with self._lock:
            self._mem[run.id] = run
            self._mem.move_to_end(run.id)
            while len(self._mem) > self.capacity:
                self._mem.popitem(last=False)

    def delete(self, run_id: str) -> bool:
        """Remove a run from disk and memory. Only well-formed ids inside RUNS_DIR are touched."""
        if not RUN_ID.fullmatch(run_id):
            return False
        d = (RUNS_DIR / run_id).resolve()
        if d.parent != RUNS_DIR.resolve() or not (d / "meta.json").exists():
            return False
        with self._lock:
            self._mem.pop(run_id, None)
        shutil.rmtree(d)
        return True

    def clear(self) -> int:
        """Delete every run in the history. Returns how many were removed."""
        if not RUNS_DIR.exists():
            return 0
        return sum(self.delete(d.name) for d in list(RUNS_DIR.iterdir()) if d.is_dir())

    def recent(self, limit: int = 30) -> list[dict]:
        if not RUNS_DIR.exists():
            return []
        out = []
        for d in sorted(RUNS_DIR.iterdir(), reverse=True):
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    out.append(describe(json.loads(meta_path.read_text(encoding="utf-8"))))
                except (OSError, ValueError, KeyError):
                    continue
            if len(out) >= limit:
                break
        return out


def _foreground(run: Run, alpha: np.ndarray, p: pp.PostParams, device: str) -> Image.Image:
    if not p.refine:
        return run.image
    key = (p.refine_r1, p.refine_r2, p.mask_mode, round(p.threshold, 4), p.invert, p.grow, p.feather, run.edits_rev)
    if key not in run.fg_cache:
        try:
            fg = pp.refine_foreground(run.image, alpha, p.refine_r1, p.refine_r2, device)
        except RuntimeError as e:  # includes torch.OutOfMemoryError
            if not is_oom(e):
                raise
            empty_cache(device)
            fg = pp.refine_foreground(run.image, alpha, p.refine_r1, p.refine_r2, "cpu")
        run.fg_cache.clear()  # keep a single version: large photos are heavy
        run.fg_cache[key] = fg
    return run.fg_cache[key]


def final_alpha(run: Run, p: pp.PostParams, device: str = "cpu") -> np.ndarray:
    """The selection as shown: mask settings (threshold, invert, grow, feather), then brush strokes."""
    try:
        alpha = pp.apply_mask_ops(run.mask, p, device)
    except RuntimeError as e:  # includes torch.OutOfMemoryError
        if not is_oom(e):
            raise
        empty_cache(device)
        alpha = pp.apply_mask_ops(run.mask, p, "cpu")
    return pp.paint_strokes(alpha, run.edits)


MAX_STROKE_POINTS = 20000


def _clean_stroke(stroke: dict) -> dict:
    pts = stroke.get("points")
    if not isinstance(pts, list) or not pts or len(pts) > MAX_STROKE_POINTS:
        raise ValueError("A stroke needs between 1 and 20000 points.")
    points = [[round(float(x), 1), round(float(y), 1)] for x, y in pts]
    return {
        "mode": "erase" if stroke.get("mode") == "erase" else "add",
        "size": round(min(max(float(stroke.get("size", 20)), 1.0), float(pp.BRUSH_LIMIT)), 1),
        "hardness": round(min(max(float(stroke.get("hardness", 0.8)), 0.0), 1.0), 3),
        "points": points,
    }


def edit(run: Run, op: str, stroke: dict | None = None) -> None:
    """Brush history: ``add`` a stroke, ``undo``, ``redo`` or ``clear`` all strokes."""
    if op == "add":
        run.edits.append(_clean_stroke(stroke or {}))
        run.redo.clear()
    elif op == "undo" and run.edits:
        run.redo.append(run.edits.pop())
    elif op == "redo" and run.redo:
        run.edits.append(run.redo.pop())
    elif op == "clear":
        run.edits.clear()
        run.redo.clear()
    elif op not in ("undo", "redo"):
        raise ValueError(f"Unknown edit operation: {op}")
    run.edits_rev += 1


def render(run: Run, p: pp.PostParams, device: str = "cpu", bg_image: Image.Image | None = None) -> dict:
    """Apply post-processing, write the run's files and return its description for the UI."""
    t0 = time.perf_counter()
    alpha = final_alpha(run, p, device)
    t1 = time.perf_counter()
    fg = _foreground(run, alpha, p, device)
    t2 = time.perf_counter()
    result = pp.compose(run.image, fg, alpha, p, bg_image)
    t3 = time.perf_counter()

    d = run.dir
    for old in d.glob("cutout.*"):
        old.unlink()
    cutout_file = f"cutout.{p.export_format}"
    (d / cutout_file).write_bytes(pp.encode(result, p.export_format))
    pp.alpha_to_image(alpha).save(d / "mask.png")
    pp.overlay(run.image, alpha).save(d / "overlay.jpg", quality=85)
    thumb = result.copy()
    thumb.thumbnail((256, 256))
    thumb.save(d / "thumb.png")
    t4 = time.perf_counter()

    run.post = asdict(p)
    post_ms = {"mask": t1 - t0, "refine": t2 - t1, "compose": t3 - t2, "save": t4 - t3}
    meta = {
        "id": run.id,
        "name": run.name,
        "width": run.image.width,
        "height": run.image.height,
        "original_file": run.original_file,
        "cutout_file": cutout_file,
        "info": run.info,
        "post": run.post,
        "edits": run.edits,
        "redo_count": len(run.redo),
        "post_timings_ms": {k: round(v * 1000, 1) for k, v in post_ms.items()},
        "updated": time.time(),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return describe(meta)


def export_psd(run: Run, p: pp.PostParams, device: str = "cpu", bg_image: Image.Image | None = None, lang: str = "en") -> Path:
    """Write ``layers.psd`` for the run (subject + editable layer mask). Returns its path."""
    from .psd_export import write_psd

    alpha = final_alpha(run, p, device)
    fg = _foreground(run, alpha, p, device)
    background = pp.background_image(run.image, p, bg_image)
    return write_psd(run.dir / "layers.psd", run.image, alpha, fg, background, lang)


def describe(meta: dict) -> dict:
    """Run metadata + file URLs (with a cache-busting parameter)."""
    v = int(meta.get("updated", time.time()) * 1000)
    base = f"/outputs/runs/{meta['id']}"
    edits = meta.get("edits", [])
    return {
        **{k: v for k, v in meta.items() if k != "edits"},
        "edits_count": len(edits),
        "redo_count": meta.get("redo_count", 0),
        "urls": {
            "original": f"{base}/{meta['original_file']}",
            "cutout": f"{base}/{meta['cutout_file']}?v={v}",
            "mask": f"{base}/mask.png?v={v}",
            "overlay": f"{base}/overlay.jpg?v={v}",
            "thumb": f"{base}/thumb.png?v={v}",
        },
    }
