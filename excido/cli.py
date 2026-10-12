"""The ``excido`` command.

    excido                      start the web app (same as ``excido serve``)
    excido setup                download the sample images and the core models
    excido download ...         download models (``--list`` shows the catalog)
    excido examples             download the official sample images
    excido remove IMAGES ...    remove backgrounds from files or folders
    excido mcp                  MCP server over stdio (for Claude Desktop and similar clients)
    excido smoke                end-to-end check of the downloaded models

Heavy imports (torch, transformers) happen inside the handlers, so ``--help`` is instant.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys
import time
from pathlib import Path

from . import __version__

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ------------------------------------------------------------------------- serve


def _mcp_env(a: argparse.Namespace) -> None:
    """MCP options travel as environment variables (read when the MCP server is built)."""
    if a.mcp_root:
        os.environ["EXCIDO_MCP_ROOT"] = str(Path(a.mcp_root).expanduser().resolve())
    if a.mcp_allow_urls:
        os.environ["EXCIDO_MCP_ALLOW_URLS"] = "1"
    if a.mcp_allow_downloads:
        os.environ["EXCIDO_MCP_ALLOW_DOWNLOADS"] = "1"


def cmd_serve(a: argparse.Namespace) -> int:
    if a.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"  # must be set before huggingface_hub is imported
    _mcp_env(a)
    if a.no_mcp:
        os.environ["EXCIDO_MCP"] = "0"
    import threading
    import urllib.request
    import webbrowser

    import uvicorn

    try:
        loopback = ipaddress.ip_address(a.host).is_loopback
    except ValueError:
        loopback = a.host == "localhost"
    if not loopback:
        print(
            f"WARNING: listening on {a.host}. Excido has no authentication: anyone who can reach this\n"
            "address can use your GPU, read your results and make the server fetch arbitrary URLs.\n"
            "(The MCP endpoint keeps accepting requests from this computer only.)",
            file=sys.stderr,
        )
        extra = "*" if a.host in ("0.0.0.0", "::") else a.host
        os.environ["EXCIDO_ALLOWED_HOSTS"] = f"127.0.0.1,localhost,{extra}"
    shown = "127.0.0.1" if a.host in ("0.0.0.0", "::") else a.host
    url = f"http://{shown}:{a.port}/"
    print(f"Excido {__version__} -> {url}  (Ctrl+C to quit)")
    if not a.no_mcp:
        from .mcp_server import default_root

        print(f"MCP endpoint -> {url}mcp  (workspace: {default_root()})")

    if not a.no_browser:

        def open_when_ready() -> None:
            for _ in range(240):
                try:
                    urllib.request.urlopen(url + "api/system", timeout=1)
                    webbrowser.open(url)
                    return
                except OSError:
                    time.sleep(0.5)

        threading.Thread(target=open_when_ready, daemon=True).start()
    uvicorn.run("excido.server:app", host=a.host, port=a.port, log_level="warning")
    return 0


# ---------------------------------------------------------------------- download


def cmd_download(a: argparse.Namespace) -> int:
    from . import catalog, store

    if a.list:
        for spec in catalog.MODELS.values():
            status = "OK " if store.is_downloaded(spec) else "-- "
            core = "core" if spec.id in catalog.CORE_IDS else "    "
            lic = "" if spec.license == "MIT" else f"  (license: {spec.license})"
            print(f"{status} {core} {spec.id:<18} {spec.repo:<38} {spec.size_bytes / 1e6:6.0f} MB{lic}")
        return 0

    ids = list(a.ids)
    if a.set == "core":
        ids += list(catalog.CORE_IDS)
    elif a.set == "all":
        ids += list(catalog.MODELS)
    if not ids:
        print("Name some model ids or use --set core|all (see --list).", file=sys.stderr)
        return 2
    try:
        specs = [catalog.get(i) for i in dict.fromkeys(ids)]
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 2
    todo = [s for s in specs if a.force or not store.is_downloaded(s)]
    print(f"Models requested: {len(specs)} | to download: {len(todo)} (~{sum(s.size_bytes for s in todo) / 1e9:.2f} GB)")
    for spec in specs:
        if spec not in todo:
            print(f"[present] {spec.id} -> {store.model_dir(spec)}")
            continue
        print(f"[download] {spec.id} ({spec.repo} @ {spec.revision[:8]}, {spec.size_bytes / 1e6:.0f} MB)", flush=True)
        t0 = time.perf_counter()
        path = store.download(spec, force=a.force)
        print(f"    done in {time.perf_counter() - t0:.0f}s -> {path}")
    return 0


def cmd_examples(_: argparse.Namespace) -> int:
    from .examples import download_examples

    download_examples()
    return 0


def cmd_setup(a: argparse.Namespace) -> int:
    from . import ROOT

    print(f"Data directory: {ROOT}")
    print("\n==> Sample images")
    cmd_examples(a)
    if a.models != "none":
        print(f"\n==> Models ({a.models})")
        cmd_download(argparse.Namespace(list=False, ids=[], set=a.models, force=False))
    print("\n==> Hardware")
    from .engine import Engine

    for d in Engine().system_info()["devices"]:
        print(f"  {d['id']:<5} {d['name']}")
    print("\nSetup complete. Start the app with:  excido   (or: uv run excido)")
    return 0


# ------------------------------------------------------------------------ remove


def _collect(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"[skipped] {raw}: not found", file=sys.stderr)
    return files


def cmd_remove(a: argparse.Namespace) -> int:
    from PIL import Image, ImageOps

    from . import postprocess as pp
    from .engine import Engine, EngineError, InferParams

    files = _collect(a.inputs)
    if not files:
        print("No images to process.", file=sys.stderr)
        return 1

    infer = {
        "model": a.model,
        "max_side": a.max_side,
        "device": a.device,
        "precision": a.precision,
        "matmul_precision": a.matmul_precision,
        "batch_size": a.batch_size,
        "oom_fallback": "cpu" if a.oom_cpu else "error",
        "mask_interp": a.mask_interp,
    }
    if a.resolution in ("auto", "original"):
        infer["resolution"] = a.resolution
    else:
        try:
            w, h = (int(v) for v in a.resolution.lower().split("x"))
        except ValueError:
            print("--resolution must be auto, original or WxH (e.g. 1024x1024).", file=sys.stderr)
            return 2
        infer.update(resolution="custom", width=w, height=h)
    ip = InferParams.from_dict(infer)

    post = {
        "mask_mode": a.mask_mode,
        "threshold": a.threshold,
        "invert": a.invert,
        "grow": a.grow,
        "feather": a.feather,
        "refine": not a.no_refine,
        "refine_r1": a.r1,
        "refine_r2": a.r2,
        "export_format": a.format,
    }
    bg_image = None
    if a.background.startswith("#"):
        post.update(background="color", bg_color=a.background)
    elif a.background.startswith("blur"):
        post.update(background="blur", bg_blur=int(a.background.partition(":")[2] or 20))
    elif a.background != "transparent":
        post["background"] = "image"
        bg_image = Image.open(a.background).convert("RGB")
    pparams = pp.PostParams.from_dict(post)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from .engine import effective_precision, resolve_device

    engine = Engine()
    device = resolve_device(ip.device)
    print(f"{len(files)} images | model {ip.model} | {device} {effective_precision(device, ip.precision)} | output -> {out}")
    t_all = time.perf_counter()
    for start in range(0, len(files), ip.batch_size):
        chunk = files[start : start + ip.batch_size]
        images = [ImageOps.exif_transpose(Image.open(f)).convert("RGB") for f in chunk]
        try:
            masks, info = engine.predict(images, ip)
        except EngineError as e:
            print(f"[error] {', '.join(f.name for f in chunk)}: {e.message}", file=sys.stderr)
            continue
        for f, image, mask, size in zip(chunk, images, masks, info["input_sizes"], strict=True):
            alpha = pp.apply_mask_ops(mask, pparams, info["device"])
            fg = pp.refine_foreground(image, alpha, pparams.refine_r1, pparams.refine_r2, info["device"]) if pparams.refine else image
            result = pp.compose(image, fg, alpha, pparams, bg_image)
            if a.psd:
                from .psd_export import write_psd

                background = pp.background_image(image, pparams, bg_image)
                write_psd(out / f"{f.stem}.psd", image, alpha, fg, background)
            (out / f"{f.stem}.{pparams.export_format}").write_bytes(pp.encode(result, pparams.export_format))
            pp.alpha_to_image(alpha).save(out / f"{f.stem}_mask.png")
            print(f"[ok] {f.name:<32} input {size[0]}x{size[1]}  {info['device']} {info['timings_ms']['inference']:.0f} ms")
    print(f"Done in {time.perf_counter() - t_all:.1f}s")
    return 0


def cmd_smoke(a: argparse.Namespace) -> int:
    from .smoke import run

    return run(a.models, a.image, a.precisions, a.device, a.stress)


def cmd_mcp(a: argparse.Namespace) -> int:
    # stdout carries the MCP protocol: everything else must go to stderr.
    _mcp_env(a)
    os.environ["EXCIDO_PORT"] = str(a.port)
    if a.standalone:
        os.environ["EXCIDO_MCP_STANDALONE"] = "1"
    from .mcp_server import Settings, run_stdio

    settings = Settings.from_env()
    print(f"Excido MCP (stdio) | workspace: {settings.root}", file=sys.stderr)
    run_stdio(settings)
    return 0


def _add_mcp_options(s: argparse.ArgumentParser) -> None:
    s.add_argument("--mcp-root", help="workspace folder the MCP tools may use (default: ~/Pictures/Excido)")
    s.add_argument("--mcp-allow-urls", action="store_true", help="let MCP clients pass http(s) image URLs")
    s.add_argument("--mcp-allow-downloads", action="store_true", help="let MCP clients download models")


# ------------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="excido", description="Local background removal powered by BiRefNet.")
    ap.add_argument("--version", action="version", version=f"excido {__version__}")
    sub = ap.add_subparsers(dest="command")

    s = sub.add_parser("serve", help="start the web app (default command)")
    s.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1, this computer only)")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--no-browser", action="store_true", help="don't open the browser")
    s.add_argument("--offline", action="store_true", help="never contact Hugging Face; use downloaded models only")
    s.add_argument("--no-mcp", action="store_true", help="don't expose the MCP endpoint at /mcp")
    _add_mcp_options(s)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser(
        "mcp",
        help="MCP server over stdio (Claude Desktop and similar clients)",
        description="Uses the running web app if there is one (shared GPU and history), otherwise loads the model itself.",
    )
    _add_mcp_options(s)
    s.add_argument("--port", type=int, default=7860, help="port of the web app to reuse, if running")
    s.add_argument("--standalone", action="store_true", help="never use the running web app")
    s.set_defaults(func=cmd_mcp)

    s = sub.add_parser("setup", help="download the sample images and the core models")
    s.add_argument("--models", choices=["core", "all", "none"], default="core")
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("download", help="download models")
    s.add_argument("ids", nargs="*", help="model ids (see --list)")
    s.add_argument("--set", choices=["core", "all"], help="predefined group of models")
    s.add_argument("--list", action="store_true", help="show the catalog and what is downloaded")
    s.add_argument("--force", action="store_true", help="download again even if present")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("examples", help="download the official sample images")
    s.set_defaults(func=cmd_examples)

    s = sub.add_parser(
        "remove",
        help="remove backgrounds from files or folders",
        description="Writes <name>.png (cutout or composite) and <name>_mask.png for every image.",
    )
    s.add_argument("inputs", nargs="+", help="image files or folders")
    s.add_argument("--out", default="outputs/cli", help="output folder (default: outputs/cli)")
    s.add_argument("--model", default="general", help="model id (see `excido download --list`)")
    s.add_argument("--resolution", default="auto", help="auto | original | WxH (e.g. 1024x1024)")
    s.add_argument("--max-side", type=int, default=2048, help="long-side cap for original/dynamic")
    s.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    s.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    s.add_argument("--matmul-precision", default="high", choices=["high", "highest"])
    s.add_argument("--batch-size", type=int, default=1)
    s.add_argument("--oom-cpu", action="store_true", help="fall back to the CPU when GPU memory runs out")
    s.add_argument("--mask-interp", default="bilinear", choices=["bilinear", "bicubic", "lanczos"])
    s.add_argument("--mask-mode", default="soft", choices=["soft", "binary"])
    s.add_argument("--threshold", type=float, default=0.5)
    s.add_argument("--invert", action="store_true", help="keep the background, drop the subject")
    s.add_argument("--grow", type=float, default=0.0, help="expand (+) or contract (-) the selection, in pixels (0.5 steps)")
    s.add_argument("--feather", type=float, default=0.0, help="soften the mask edge, in pixels")
    s.add_argument("--no-refine", action="store_true", help="disable refine_foreground")
    s.add_argument("--r1", type=int, default=90, help="refine_foreground wide radius")
    s.add_argument("--r2", type=int, default=6, help="refine_foreground fine radius")
    s.add_argument("--background", default="transparent", help="transparent | #RRGGBB | blur:<radius> | path to an image")
    s.add_argument("--format", default="png", choices=["png", "webp"])
    s.add_argument("--psd", action="store_true", help="also write a layered PSD (subject with an editable layer mask)")
    s.set_defaults(func=cmd_remove)

    s = sub.add_parser("smoke", help="end-to-end check of the downloaded models")
    s.add_argument("--models", nargs="*", help="model ids (default: every downloaded model)")
    s.add_argument("--image", help="test image (default: examples/official/Helicopter.jpg)")
    s.add_argument("--precisions", nargs="*", default=["fp16"], choices=["fp16", "bf16", "fp32"])
    s.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    s.add_argument("--stress", action="store_true", help="also try 2048 and 2304 to probe memory limits")
    s.set_defaults(func=cmd_smoke)
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:  # bare `excido` = serve with defaults
        args = parser.parse_args(["serve", *(argv or sys.argv[1:])])
    return args.func(args)
