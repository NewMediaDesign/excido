# Parameters

English · [Italiano](it/PARAMETRI.md)

Settings come in two families, and each section of the right-hand panel carries its tag:

- **run** settings change what goes into the model. Applying them needs a new inference: the *Run* button changes to *Re-run to apply*.
- **live** settings act on the mask that is already computed and kept in memory. They apply instantly, without running the model again.

The same names are used by the API (`/api/segment`, `/api/runs/{id}/postprocess`), by the MCP tools and, with matching options, by `excido remove`.

## Model (run)

| Parameter | Values | Default | Notes |
|---|---|---|---|
| `model` | an id from [MODELS.md](MODELS.md) | `general` | Sets the backbone and the native resolution. Only one model stays in memory; switching reloads in a few seconds. |

## Input resolution (run)

BiRefNet resizes the image to a fixed size (**H** × **W**), **without keeping its proportions**, exactly like the official code. It then resizes the mask back to the original size.

| Parameter | Values | Default | Notes |
|---|---|---|---|
| `resolution` | `auto` · `custom` · `original` | `auto` | See the three modes below. |
| `width`, `height` | 256–4096 | 1024 | Used with `custom` only. |
| `max_side` | 256–4096 | 2048 | Long-side cap for `original`, and for dynamic models with `auto`. |

The three modes:
- **`auto`** uses the model's native size: 1024², 2048², 512² or 2560×1440. Dynamic models instead use the image's own size, scaled down until the long side fits `max_side`.
- **`custom`** uses the width and height you set.
- **`original`** keeps the image proportions, with the long side capped at `max_side`. It is meant for dynamic models; it works with the others, but quality drops away from their native size.

**Architecture constraint.** Width and height must be **multiples of 32**, because the decoder splits the input into patches aligned with the 1/32 feature map. They are always rounded down, with a minimum of 256.

**Memory** grows with the pixel count: 2048² needs about four times the memory of 1024². The measurements are in the README.

## Performance (run)

| Parameter | Values | Default | Effect |
|---|---|---|---|
| `device` | `auto` · `cuda` · `mps` · `cpu` | `auto` | `auto` picks NVIDIA CUDA, then Apple Metal (MPS), then the CPU. A device that isn't available falls back to the best one that is. |
| `precision` | `fp16` · `bf16` · `fp32` | `fp16` | See the three options below. |
| `matmul_precision` | `high` · `highest` | `high` | `torch.set_float32_matmul_precision`: `high` lets FP32 matmuls use TF32 tensor cores. It matters only in FP32. |
| `batch_size` | 1–8 | 1 | Batch tab only: images with the same input size are processed together. Not the `Config.batch_size` baked into the weights. |
| `oom_fallback` | `error` · `cpu` | `error` | When GPU memory runs out: report it with hints, or retry on the CPU (slow). |
| `keep_loaded` | yes / no | yes | Keeps the model in memory between runs. When off, memory is freed and every run reloads the model. |
| `mask_interp` | `bilinear` · `bicubic` · `lanczos` | `bilinear` | How the mask is resized back to the original size. `bilinear` matches `inference.py`; the model card uses PIL's bicubic. |

The three options of `precision`:
- **`fp16`** puts weights and input in half precision (`model.half()`, like the model card). It is the fastest option and uses the least memory, with practically no quality loss according to the author. Measured: 0.40 s and 1.6 GB at 1024², 1.5 s and 5.0 GB at 2048².
- **`bf16`** keeps FP32 weights and runs `torch.autocast(bfloat16)`, like the repository's `inference.py`. It is available on NVIDIA Ampere or newer; on Metal it falls back to `fp16`. Measured: 0.56 s and 3.0 GB.
- **`fp32`** is full precision. Measured: 0.72 s and 3.1 GB at 1024²; above 8 GB at 2048². On the CPU it is always used.

The author quotes 3.5 GB (FP16) and 4.8 GB (FP32) at 1024²; those are process-reserved figures. The numbers above are tensor peaks (`max_memory_allocated`) on an RTX 4070 Laptop.

**VRAM cap.** On the first GPU load, Excido caps its own memory at the free memory minus 512 MB. Beyond that you get an immediate out-of-memory error instead of the driver's silent "sysmem fallback" on Windows. Override it with `EXCIDO_VRAM_FRACTION` (e.g. `0.9`).

## Mask (live)

| Parameter | Values | Default | Effect |
|---|---|---|---|
| `mask_mode` | `soft` · `binary` | `soft` | `soft` uses the sigmoid as alpha (0–255): soft edges, hair, transparency. `binary` gives 0 or 255 only. |
| `threshold` | 0–1 | 0.5 | Cut-off for `binary`; raising it makes the subject tighter. |
| `invert` | yes / no | no | Inverts the mask: keeps the background and drops the subject. |
| `grow` | −100…+100 px, steps of 0.5 (1 in binary mode) | 0 | Expands (+) or contracts (−) the selection. Soft edges keep their profile and move as a whole, so hair stays soft. 1–3 px cleans background halos or recovers thin parts. |
| `feather` | 0–100 px, steps of 0.5 | 0 | Softens the mask edge with a Gaussian blur of this radius. |

The official code applies no threshold: the soft mask is BiRefNet's native output.

The order is fixed: threshold → invert → grow/shrink → feather → brush strokes.

**How grow/shrink works.** It applies grayscale dilation (grow) or erosion (shrink) to the alpha. The steps alternate a 4-neighbour cross and a 3×3 square, which approximates a disk, so the edge moves by the same distance in every direction: exactly 1 px along the axes for `±1`. A half-pixel value blends the two nearest whole radii.

It runs after threshold and invert, so a positive value always enlarges what is kept. The web UI shows it as a slider from −25 to +25 px, with −/+ buttons, a typed value (also beyond ±25) and ↑ ↓ keys (Shift multiplies the step by 10). Double-clicking the slider resets it to 0.

**Feather.** The radius is the sigma of the Gaussian, so the transition is about twice the radius on each side of the edge. It runs after grow/shrink: grow by a few pixels, then feather, to get a soft edge that doesn't eat into the subject. From 6 px up, the blur runs on a downscaled mask and is scaled back up, so even 100 px costs a fraction of a second. The web UI slider goes from 0 to 25 px; the API, the CLI (`--feather`) and the MCP tools accept up to 100.

## Brush (live)

The brush fixes the selection by hand where the model got it wrong: paint **Add** to bring back a missed part, **Erase** to drop a leftover piece of background.

| Setting | Values | Default | Effect |
|---|---|---|---|
| mode | off · add · erase | off | While a mode is on, dragging on the image paints instead of panning. |
| size | 2–400 px in the UI (up to 2000 through the API) | 40 | Brush diameter in **image** pixels, so a stroke covers the same area at any zoom. |
| hardness | 0–100% | 80% | 100% is a hard disk; lower values fade the brush from its inner core to the rim. |

Keyboard shortcuts: <kbd>B</kbd> add, <kbd>E</kbd> erase, <kbd>Esc</kbd> off, <kbd>[</kbd> <kbd>]</kbd> smaller/larger, <kbd>Space</kbd>+drag (or the middle button) pans while painting, <kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>Z</kbd> undo, <kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd> or <kbd>Ctrl</kbd>+<kbd>Y</kbd> redo. Painting needs a single image under the pointer, so turning the brush on switches the before/after and side-by-side views to the overlay.

How the strokes are kept:
- Strokes are applied **in order, on top of** threshold, invert, grow and feather, so later strokes win and changing a slider never loses them.
- They are saved with the result (`meta.json`), with undo and redo; **Clear strokes** removes them all.
- Re-running the model on the same image (for example with another model or resolution) **carries the strokes over**.
- They reach every output: cutout, mask, overlay and the layered PSD.

Through the API, `POST /api/runs/{id}/edits` takes `{"op": "add", "stroke": {"mode": "add" | "erase", "size": 40, "hardness": 0.8, "points": [[x, y], …]}}`, or `{"op": "undo" | "redo" | "clear"}`. Coordinates are in pixels of the original image; a stroke holds at most 20,000 points.

## Edge colors: `refine_foreground` (live)

| Parameter | Values | Default | Effect |
|---|---|---|---|
| `refine` | yes / no | yes | Estimates the clean foreground colors on semi-transparent edges and removes the old background's halo. |
| `refine_r1` | 1–255 | 90 | Radius of the first (wide) estimate. |
| `refine_r2` | 1–63 | 6 | Radius of the second (fine) pass; fixed to 6 upstream. |

This is the blur-fusion foreground estimation of Forte & Pitié (ICIP 2021), as used by the official demo. It is reimplemented with cumulative sums, so its cost is O(1) per pixel whatever the radius. It matters most for hair over a transparent background.

## Background (live)

| Parameter | Values | Default | Effect |
|---|---|---|---|
| `background` | `transparent` · `color` · `blur` · `image` | `transparent` | See the four cases below. |
| `bg_color` | `#RRGGBB` | `#ffffff` | Presets include green screen `#00b140` and blue screen `#0047bb`. |
| `bg_blur` | 1–100 px | 20 | Gaussian blur radius for `blur`. |
| `bg_image_id` | id from `/api/backgrounds` | — | Background image, fitted like CSS `object-fit: cover`. |

The four cases of `background`:
- **`transparent`** produces an RGBA PNG or WebP.
- **`color`** composites the subject over a solid color.
- **`blur`** composites it over the blurred original, for a portrait look.
- **`image`** composites it over an uploaded image.

## Export (live)

| Parameter | Values | Default | Notes |
|---|---|---|---|
| `export_format` | `png` · `webp` | `png` | WebP is lossless, so it keeps the alpha channel. The mask is always an 8-bit PNG. |

**Layered PSD.** The **PSD** button (or `excido remove --psd`, or `psd=true` in the MCP tools) writes a Photoshop file. Its layers, bottom to top, are **Background** (only with a color, blur or image background), **Original** (hidden, for reference) and **Subject**: the photo with an editable **layer mask** holding the alpha. Inside the mask the subject pixels carry the refined edge colors; outside it they are the untouched photo, so painting the mask back in reveals the real image. The export uses the mask and background settings currently shown, and layer names follow the interface language. With a transparent background the merged preview keeps its transparency. It opens in Photoshop, Affinity, GIMP and Photopea.

## Not exposed, and why

| Item | Reason |
|---|---|
| Backbone, `mul_scl_ipt`, `dec_att`, `dec_blk`, `cxt_num`, `out_ref`, … | They define the trained architecture: changing them breaks weight loading. |
| `Config.size` | Never read by the model code: the input tensor decides the resolution. |
| `SDPA_enabled` | Buggy in the Hugging Face code (it ignores the position bias and the window mask), so it stays off. |
| `torch.compile` | The author advises against it for inference: little gain, a very long first run, and it needs Triton on Windows. |
| ONNX / TensorRT | ONNX is about 90% slower than PyTorch with Swin-L; TensorRT exists only as third-party projects. |
| Normalization | The ImageNet mean and standard deviation are fixed by training. |
