# Models

English · [Italiano](it/MODELLI.md)

This catalog covers the 16 official BiRefNet variants published by **ZhengPeng7** on Hugging Face. Code and weights are MIT, except `BiRefNet-DIS5K-TR_TEs`, whose model card declares no license.

Each variant is **pinned to a commit** (`revision` in `excido/catalog.py`), so the remote `birefnet.py` that runs with `trust_remote_code` never changes. Revisions and sizes were read from the Hugging Face API on 2026-10-12.

Models are stored in `models/<Repo>/` (for example `models/BiRefNet_HR/`):

| File | Contents |
|---|---|
| `model.safetensors` | the weights |
| `birefnet.py`, `BiRefNet_config.py`, `config.json` | the model code, readable |
| `README.md` | the original model card |
| `.revision` | the downloaded commit |

## Variants

★ marks the models `excido setup` downloads (the `core` set). Get the others with **Download** in the app's Model panel, or with `uv run excido download <id>`.

| ★ | id | Hugging Face repo | Backbone | Resolution | Weights | Use it for |
|---|---|---|---|---|---|---|
| ★ | `general` | ZhengPeng7/BiRefNet | Swin-L | 1024×1024 | 444 MB fp16 | **Default.** General background removal, including people and transparency. |
| ★ | `general-hr` | ZhengPeng7/BiRefNet_HR | Swin-L | 2048×2048 | 444 MB fp16 | Large photos and fine detail. Use it at 2048: at 1024 it does worse than `general`. |
| ★ | `general-lite` | ZhengPeng7/BiRefNet_lite | Swin-T | 1024×1024 | 178 MB fp32 | About 5× fewer parameters: faster, slightly lower quality. |
| ★ | `general-dynamic` | ZhengPeng7/BiRefNet_dynamic | Swin-L | any (256–2304) | 444 MB fp16 | Trained on random sizes: works at the image's own proportions. |
| ★ | `matting` | ZhengPeng7/BiRefNet-matting | Swin-L | 1024×1024 | 885 MB fp32 | Trimap-free matting: hair, fur, glass, smoke. |
| ★ | `matting-hr` | ZhengPeng7/BiRefNet_HR-matting | Swin-L | 2048×2048 | 444 MB fp16 | High-resolution matting; the official demo uses it for portraits. |
| ★ | `portrait` | ZhengPeng7/BiRefNet-portrait | Swin-L | 1024×1024 | 885 MB fp32 | Portraits (P3M-10k + TR-humans). |
|  | `matting-dynamic` | ZhengPeng7/BiRefNet_dynamic-matting | Swin-L | any | 444 MB fp16 | Matting at any resolution. |
|  | `matting-lite` | ZhengPeng7/BiRefNet_lite-matting | Swin-T | 1024 (inferred) | 89 MB fp16 | Light matting; the card states no resolution. |
|  | `general-512` | ZhengPeng7/BiRefNet_512x512 | Swin-L | 512×512 | 444 MB fp16 | Small inputs and fast previews. |
|  | `general-lite-2k` | ZhengPeng7/BiRefNet_lite-2K | Swin-T | 2560×1440 | 178 MB fp32 | Landscape 2K images with a light backbone. |
|  | `general-legacy` | ZhengPeng7/BiRefNet-legacy | Swin-L | 1024×1024 | 885 MB fp32 | Older general model without matting data: no transparency. |
|  | `dis` | ZhengPeng7/BiRefNet-DIS5K | Swin-L | 1024×1024 | 885 MB fp32 | Paper model for Dichotomous Image Segmentation (very fine structures). |
|  | `dis-tr-tes` | ZhengPeng7/BiRefNet-DIS5K-TR_TEs | Swin-L | 1024×1024 | 885 MB fp32 | Like `dis`, also trained on the test sets. **No license declared.** |
|  | `hrsod` | ZhengPeng7/BiRefNet-HRSOD | Swin-L | 1024×1024 | 885 MB fp32 | High-resolution salient object detection. |
|  | `cod` | ZhengPeng7/BiRefNet-COD | Swin-L | 1024×1024 | 885 MB fp32 | Camouflaged object detection. |

```bash
uv run excido download --list
```

```bash
uv run excido download --set all
```

## Which one?

- **Everyday use.** `general`, with resolution Auto (1024²).
- **Best quality and detail** (jewelry, metal structures, big photos):
  - `general-hr`, with Auto (2048², FP16). It needs about 1.5 s and 5 GB of VRAM on an RTX 4070 Laptop.
  - Alternatively, `general-dynamic` with a long-side limit of 1536–2048, which keeps the proportions.
- **Hair, fur, transparency.** `matting-hr` (same cost as `general-hr`) or `matting`.
- **People.** `portrait` or `matting`.
- **Speed or little memory.** `general-lite`; `general-512` for previews.

## Published results

S-measure on DIS-VD, higher is better:

| Model | 512 | 1024 | 2048 |
|---|---|---|---|
| BiRefNet (general) | — | 0.912 | 0.898 |
| BiRefNet_HR | 0.634 | 0.893 | **0.927** |

**Training resolution matters.** That is why `auto` always uses the native one.

## About the remote code

- The `Config` in `birefnet.py` is fixed, and almost every field defines the trained architecture. This includes `batch_size`, which decides whether the BatchNorm layers exist. None of them are exposed.
- `SDPA_enabled` is `False` in the Hugging Face code, and its SDPA branch is buggy. Leave it off.
- `handler.py`, used by Hugging Face Inference Endpoints, is not downloaded.
