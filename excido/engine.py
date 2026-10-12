"""BiRefNet inference engine.

Pipeline (identical to the official code):
    RGB -> Resize((H, W)) -> ToTensor -> Normalize(ImageNet) -> model(x)[-1].sigmoid()
The soft mask in [0, 1] is then resized back to the original image size.

Devices: NVIDIA (``cuda``), Apple Silicon (``mps``) and ``cpu``; ``auto`` picks the
best one available. At most one model stays in memory and inference is serialized
by a lock, so concurrent requests never fight over GPU memory.
"""

from __future__ import annotations

import gc
import os
import platform
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from . import __version__, catalog, store

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MIN_SIDE = 256
MAX_SIDE = 4096


class EngineError(Exception):
    """An error meant for the user.

    ``code`` and ``params`` let the UI show a translated message; ``hints`` are
    hint codes (translated by the UI too). ``message`` is the English fallback.
    """

    def __init__(self, code: str, message: str, params: dict | None = None, hints: list[str] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.params = params or {}
        self.hints = hints or []

    def to_dict(self) -> dict:
        return {"error": self.message, "code": self.code, "params": self.params, "hints": self.hints}


# --------------------------------------------------------------------------- devices


def available_devices() -> list[str]:
    """Usable devices, best first."""
    out = []
    if torch.cuda.is_available():
        out.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        out.append("mps")
    out.append("cpu")
    return out


def resolve_device(requested: str) -> str:
    """``auto`` (or an unavailable device) becomes the best available one."""
    devices = available_devices()
    return requested if requested in devices else devices[0]


def effective_precision(device: str, precision: str) -> str:
    if device == "cpu":
        return "fp32"
    if precision == "bf16" and (device == "mps" or not torch.cuda.is_bf16_supported()):
        return "fp16"
    return precision


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def empty_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def is_oom(exc: BaseException) -> bool:
    """CUDA raises OutOfMemoryError; MPS raises RuntimeError('MPS backend out of memory')."""
    return isinstance(exc, torch.OutOfMemoryError) or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


def _cpu_name() -> str:
    return platform.processor() or platform.machine() or "CPU"


# --------------------------------------------------------------------------- params


@dataclass
class InferParams:
    model: str = catalog.DEFAULT_ID
    resolution: str = "auto"  # "auto" (model's native) | "custom" | "original"
    width: int = 1024
    height: int = 1024
    max_side: int = 2048  # long-side cap for "original" and for dynamic models
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    precision: str = "fp16"  # "fp16" | "bf16" | "fp32"
    matmul_precision: str = "high"  # "high" | "highest" (fp32 only)
    keep_loaded: bool = True
    oom_fallback: str = "error"  # "error" | "cpu"
    mask_interp: str = "bilinear"  # "bilinear" | "bicubic" | "lanczos"
    batch_size: int = 1

    @classmethod
    def from_dict(cls, d: dict | None) -> InferParams:
        d = d or {}
        p = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if p.model not in catalog.MODELS:
            raise EngineError("unknown_model", f"Unknown model: {p.model}", {"model": p.model})
        p.resolution = p.resolution if p.resolution in ("auto", "custom", "original") else "auto"
        p.width, p.height = int(p.width), int(p.height)
        p.max_side = int(min(max(int(p.max_side), MIN_SIDE), MAX_SIDE))
        p.device = p.device if p.device in ("auto", "cuda", "mps", "cpu") else "auto"
        p.precision = p.precision if p.precision in ("fp16", "bf16", "fp32") else "fp16"
        p.matmul_precision = p.matmul_precision if p.matmul_precision in ("high", "highest") else "high"
        p.oom_fallback = p.oom_fallback if p.oom_fallback in ("error", "cpu") else "error"
        p.mask_interp = p.mask_interp if p.mask_interp in ("bilinear", "bicubic", "lanczos") else "bilinear"
        p.batch_size = int(min(max(int(p.batch_size), 1), 8))
        return p


def snap32(v: float) -> int:
    """Largest multiple of 32 not above ``v``, within [MIN_SIDE, MAX_SIDE] (architecture constraint)."""
    return int(min(max(int(v) // 32 * 32, MIN_SIDE), MAX_SIDE))


def resolve_size(spec: catalog.ModelSpec, p: InferParams, img_size: tuple[int, int]) -> tuple[int, int]:
    """(width, height) the image is resized to before entering the model."""
    if p.resolution == "custom":
        return snap32(p.width), snap32(p.height)
    if p.resolution == "original" or spec.dynamic:
        w, h = img_size
        s = min(1.0, p.max_side / max(w, h))
        return snap32(w * s), snap32(h * s)
    return spec.native  # type: ignore[return-value]


def to_rgb(img: Image.Image) -> Image.Image:
    """RGB; transparent images are flattened onto white (like the official demo)."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


# --------------------------------------------------------------------------- engine


class Engine:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model: torch.nn.Module | None = None
        self.key: tuple[str, str, str] | None = None  # (model_id, device, weight dtype)
        self._normalize = transforms.Normalize(MEAN, STD)
        self.vram_limit_mb: int | None = None

    # ------------------------------------------------------------------ system

    def system_info(self) -> dict:
        devices = available_devices()
        info: dict = {
            "version": __version__,
            "platform": f"{platform.system()} {platform.machine()}",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "devices": [],
            "default_device": devices[0],
            "cuda_version": torch.version.cuda,
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            "memory_used_mb": None,
            "memory_total_mb": None,
            "vram_limit_mb": self.vram_limit_mb,
            "loaded": None,
        }
        for dev in devices:
            if dev == "cuda":
                total = round(torch.cuda.get_device_properties(0).total_memory / 2**20)
                info["devices"].append({"id": "cuda", "name": torch.cuda.get_device_name(0), "memory_mb": total})
                info["memory_total_mb"] = total
                info["memory_used_mb"] = round(torch.cuda.memory_reserved(0) / 2**20)
            elif dev == "mps":
                total = round(torch.mps.recommended_max_memory() / 2**20) if hasattr(torch.mps, "recommended_max_memory") else None
                info["devices"].append({"id": "mps", "name": "Apple GPU (Metal)", "memory_mb": total})
                if info["memory_total_mb"] is None:
                    info["memory_total_mb"] = total
                    info["memory_used_mb"] = round(torch.mps.driver_allocated_memory() / 2**20)
            else:
                info["devices"].append({"id": "cpu", "name": _cpu_name(), "memory_mb": None})
        if self.key:
            info["loaded"] = {"model": self.key[0], "device": self.key[1], "dtype": self.key[2]}
        return info

    def _limit_vram(self) -> None:
        """Hard cap on the CUDA memory PyTorch may use.

        On Windows, when VRAM runs out the NVIDIA driver silently spills into system
        RAM ("Sysmem Fallback") instead of failing, and inference goes from seconds
        to minutes. With a cap just below the free memory, PyTorch raises
        OutOfMemoryError right away and the app handles it (message or CPU fallback).
        Override the fraction (0-1) with the EXCIDO_VRAM_FRACTION environment variable.
        """
        if self.vram_limit_mb is not None or not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
        env = os.environ.get("EXCIDO_VRAM_FRACTION")
        fraction = float(env) if env else (free - 512 * 2**20) / total
        fraction = min(max(fraction, 0.3), 0.95)
        torch.cuda.set_per_process_memory_fraction(fraction)
        self.vram_limit_mb = round(fraction * total / 2**20)

    # ------------------------------------------------------------------ models

    def load(self, model_id: str, device: str, precision: str) -> float:
        """Load the model unless it is already the one in memory. Returns the seconds spent."""
        spec = catalog.get(model_id)
        if not store.is_downloaded(spec):
            raise EngineError(
                "model_not_downloaded",
                f"The {spec.label} model is not downloaded yet.",
                {"model": spec.id, "label": spec.label},
                hints=["download_model"],
            )
        dtype = torch.float16 if (device != "cpu" and precision == "fp16") else torch.float32
        key = (model_id, device, str(dtype).replace("torch.", ""))
        if self.key == key and self.model is not None:
            return 0.0
        self.unload()
        if device == "cuda":
            self._limit_vram()
        t0 = time.perf_counter()
        from transformers import AutoModelForImageSegmentation
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        model = AutoModelForImageSegmentation.from_pretrained(str(store.model_dir(spec)), trust_remote_code=True)
        model.to(device=device, dtype=dtype).eval()
        self.model, self.key = model, key
        if device != "cpu":
            # Warm-up: initializes kernels so the first run's "inference" time is realistic.
            with torch.inference_mode():
                model(torch.zeros(1, 3, 512, 512, device=device, dtype=dtype))
            _sync(device)
        return time.perf_counter() - t0

    def unload(self) -> None:
        device = self.key[1] if self.key else None
        self.model = None
        self.key = None
        gc.collect()
        for dev in ("cuda", "mps") if device is None else (device,):
            if dev in available_devices():
                empty_cache(dev)

    # --------------------------------------------------------------- inference

    def predict(self, images: list[Image.Image], p: InferParams) -> tuple[list[np.ndarray], dict]:
        """One float32 HxW mask in [0, 1] per image (original size), plus metrics."""
        with self.lock:
            device = resolve_device(p.device)
            try:
                return self._predict(images, p, device)
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if not is_oom(e):
                    raise
            # Only reached after an OOM; outside the except block the traceback no
            # longer pins the tensors in GPU memory.
            self.unload()
            if p.oom_fallback == "cpu" and device != "cpu":
                masks, info = self._predict(images, p, "cpu")
                info["fallback"] = "cpu"
                return masks, info
            spec = catalog.get(p.model)
            hints = ["lower_resolution", "use_fp16"]
            hints.append("try_lite" if spec.backbone == "swin_v1_l" else "lower_batch")
            hints.append("enable_cpu_fallback")
            raise EngineError("out_of_memory", f"Out of GPU memory on {device}.", {"device": device}, hints)

    def _predict(self, images: list[Image.Image], p: InferParams, device: str) -> tuple[list[np.ndarray], dict]:
        spec = catalog.get(p.model)
        precision = effective_precision(device, p.precision)
        torch.set_float32_matmul_precision(p.matmul_precision)

        timings = {"load": self.load(spec.id, device, precision), "preprocess": 0.0, "inference": 0.0, "resize": 0.0}
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        rgb = [to_rgb(im) for im in images]
        sizes = [resolve_size(spec, p, im.size) for im in rgb]
        masks: list[np.ndarray | None] = [None] * len(rgb)

        # Batch together images that share the same input resolution.
        chunks: list[list[int]] = []
        for i in sorted(range(len(rgb)), key=lambda i: sizes[i]):
            if chunks and sizes[chunks[-1][0]] == sizes[i] and len(chunks[-1]) < p.batch_size:
                chunks[-1].append(i)
            else:
                chunks.append([i])

        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
        in_dtype = torch.float16 if precision == "fp16" else torch.float32
        mps_mem = 0
        assert self.model is not None
        for chunk in chunks:
            w, h = sizes[chunk[0]]
            t0 = time.perf_counter()
            x = torch.stack([self._normalize(transforms.functional.to_tensor(rgb[i].resize((w, h), Image.BILINEAR))) for i in chunk])
            x = x.to(device=device, dtype=in_dtype)
            t1 = time.perf_counter()
            with torch.inference_mode(), autocast:
                preds = self.model(x)[-1].sigmoid().float()
            _sync(device)
            if device == "mps":
                mps_mem = max(mps_mem, torch.mps.driver_allocated_memory())
            t2 = time.perf_counter()
            for j, i in enumerate(chunk):
                masks[i] = self._resize_mask(preds[j : j + 1], rgb[i].size, p.mask_interp)
            t3 = time.perf_counter()
            timings["preprocess"] += t1 - t0
            timings["inference"] += t2 - t1
            timings["resize"] += t3 - t2
            del x, preds

        if device == "cuda":
            peak = round(torch.cuda.max_memory_allocated() / 2**20)
        elif device == "mps":
            peak = round(mps_mem / 2**20)
        else:
            peak = None
        info = {
            "model": spec.id,
            "model_label": spec.label,
            "device": device,
            "precision": precision,
            "input_sizes": [list(s) for s in sizes],
            "batches": len(chunks),
            "timings_ms": {k: round(v * 1000, 1) for k, v in timings.items()},
            "memory_peak_mb": peak,
            "params": asdict(p),
        }
        if not p.keep_loaded:
            self.unload()
        else:
            empty_cache(device)  # hand activation caches (GBs at 2048) back to the driver
        return masks, info  # type: ignore[return-value]

    @staticmethod
    def _resize_mask(pred: torch.Tensor, size: tuple[int, int], interp: str) -> np.ndarray:
        """pred: [1, 1, h, w] -> float32 HxW at the original ``size`` (width, height)."""
        w, h = size
        if interp == "lanczos":
            m = Image.fromarray(pred[0, 0].cpu().numpy(), mode="F").resize((w, h), Image.LANCZOS)
            return np.clip(np.asarray(m, dtype=np.float32), 0.0, 1.0)
        out = F.interpolate(pred, size=(h, w), mode=interp, align_corners=False)
        return out[0, 0].clamp_(0, 1).cpu().numpy()
