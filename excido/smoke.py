"""End-to-end check: load each downloaded model, segment a sample, measure time and memory.

Don't run it while the web app is processing images on the same GPU: two processes
competing for VRAM give meaningless numbers (and on Windows can spill into system RAM).
"""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from . import EXAMPLES_DIR, OUTPUTS_DIR, catalog, store
from . import postprocess as pp
from .engine import Engine, EngineError, InferParams


def run_case(engine: Engine, image: Image.Image, label: str, out: Path, **params) -> dict:
    ip = InferParams.from_dict(params)
    t0 = time.perf_counter()
    try:
        _, first = engine.predict([image], ip)  # warm-up: load + kernels
        masks, info = engine.predict([image], ip)
    except EngineError as e:
        return {"case": label, "ok": False, "error": e.message}
    mask = masks[0]
    fg_ratio = float((mask > 0.5).mean())
    ok = mask.shape == (image.height, image.width) and float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0 and 0.005 < fg_ratio < 0.995
    alpha = pp.apply_mask_ops(mask, pp.PostParams())
    t1 = time.perf_counter()
    fg = pp.refine_foreground(image, alpha, device=info["device"])
    refine_ms = (time.perf_counter() - t1) * 1000
    safe = label.replace(" ", "_").replace("/", "-")
    pp.compose(image, fg, alpha, pp.PostParams()).save(out / f"{safe}.png")
    pp.alpha_to_image(alpha).save(out / f"{safe}_mask.png")
    t = info["timings_ms"]
    return {
        "case": label,
        "ok": ok,
        "input": "x".join(map(str, info["input_sizes"][0])),
        "load_s": first["timings_ms"]["load"] / 1000,
        "infer_ms": t["inference"],
        "refine_ms": refine_ms,
        "mem_mb": info["memory_peak_mb"],
        "fg": fg_ratio,
        "total_s": time.perf_counter() - t0,
        "device": info["device"],
        "precision": info["precision"],
    }


def run(models: list[str] | None, image_path: str | None, precisions: list[str], device: str, stress: bool) -> int:
    ids = models or [s.id for s in catalog.MODELS.values() if store.is_downloaded(s)]
    if not ids:
        print("No models downloaded: run `excido download --set core` (or `excido download general-lite`).")
        return 1
    path = Path(image_path) if image_path else EXAMPLES_DIR / "official" / "Helicopter.jpg"
    if not path.exists():
        print(f"Sample image not found: {path}. Run `excido examples` first.")
        return 1
    image = Image.open(path).convert("RGB")
    out = OUTPUTS_DIR / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    engine = Engine()
    names = {d["id"]: d["name"] for d in engine.system_info()["devices"]}
    print(f"Image: {path.name} ({image.width}x{image.height}) | devices: {', '.join(f'{k} ({v})' for k, v in names.items())}\n")

    rows = []
    for mid in ids:
        for prec in precisions:
            rows.append(run_case(engine, image, f"{mid} {prec}", out, model=mid, precision=prec, device=device))
            print(rows[-1], flush=True)
        if stress:
            for side in (2048, 2304):
                size = {"resolution": "custom", "width": side, "height": side}
                rows.append(run_case(engine, image, f"{mid} {side}", out, model=mid, device=device, **size))
                print(rows[-1], flush=True)
        engine.unload()

    print("\n" + "-" * 104)
    print(f"{'case':<26}{'result':<8}{'input':<11}{'load':>7}{'inference':>11}{'refine':>9}{'memory':>11}{'foreground':>12}")
    for r in rows:
        if "error" in r:
            print(f"{r['case']:<26}{'ERROR':<8}{r['error']}")
            continue
        mem = f"{r['mem_mb'] / 1024:.2f} GB" if r["mem_mb"] else "-"
        print(
            f"{r['case']:<26}{'OK' if r['ok'] else 'FAIL':<8}{r['input']:<11}{r['load_s']:>6.1f}s"
            f"{r['infer_ms']:>9.0f}ms{r['refine_ms']:>7.0f}ms{mem:>11}{r['fg'] * 100:>11.1f}%"
        )
    failed = [r for r in rows if "error" not in r and not r["ok"]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} cases OK ({sum('error' in r for r in rows)} errors). Output: {out}")
    return 1 if failed else 0
