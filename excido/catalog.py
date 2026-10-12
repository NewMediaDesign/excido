"""Catalog of the official BiRefNet variants published by ZhengPeng7 on Hugging Face.

Every entry is pinned to a commit (``revision``), so the remote ``birefnet.py`` that
runs with ``trust_remote_code`` never changes under our feet. Revisions and sizes
were read from the Hugging Face API on 2026-10-12.

User-facing texts here are English; the web UI ships its own translations
(``static/i18n.js``), keyed by model id.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Downloaded by ``excido setup``; the others are one click away in the UI.
CORE_IDS = (
    "general",
    "general-hr",
    "general-lite",
    "general-dynamic",
    "matting",
    "matting-hr",
    "portrait",
)

DEFAULT_ID = "general"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    repo: str
    revision: str
    label: str
    group: str  # "general" | "matting" | "benchmark"
    task: str
    backbone: str  # swin_v1_l (Swin-L, 220.7M params) or swin_v1_t (Swin-T, 44.4M)
    native: tuple[int, int] | None  # (width, height); None = dynamic resolution
    size_bytes: int
    weights_dtype: str
    notes: str = ""
    license: str = "MIT"

    @property
    def dynamic(self) -> bool:
        return self.native is None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dynamic"] = self.dynamic
        d["core"] = self.id in CORE_IDS
        return d


_L = "swin_v1_l"
_T = "swin_v1_t"
_SL = 444_473_596  # Swin-L, fp16 safetensors
_SL32 = 884_878_856  # Swin-L, fp32 safetensors
_ST32 = 177_634_392  # Swin-T, fp32 safetensors

# fmt: off
MODELS: dict[str, ModelSpec] = {m.id: m for m in [
    # General use
    ModelSpec("general", "ZhengPeng7/BiRefNet", "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
              "General", "general", "General background removal / segmentation",
              _L, (1024, 1024), _SL, "fp16", "Recommended for most images."),
    ModelSpec("general-hr", "ZhengPeng7/BiRefNet_HR", "a7a562f6fd16021180f2f4348f4de003a2d3d1e1",
              "General HR", "general", "High-resolution general segmentation",
              _L, (2048, 2048), _SL, "fp16", "Trained at 2048x2048; loses quality at 1024. Needs ~5 GB of VRAM."),
    ModelSpec("general-lite", "ZhengPeng7/BiRefNet_lite", "aa62cd87eafb9cc43056d08ef3615a14628b831d",
              "General Lite", "general", "General segmentation, lightweight backbone",
              _T, (1024, 1024), _ST32, "fp32", "Swin-T backbone: faster and lighter, slightly lower quality."),
    ModelSpec("general-dynamic", "ZhengPeng7/BiRefNet_dynamic", "280306042f57b7a33854319da62fd86aaa89ec4c",
              "General Dynamic", "general", "General segmentation at any resolution",
              _L, None, _SL, "fp16", "Trained on random sizes 256-2304: works at the image's own aspect ratio."),
    ModelSpec("general-512", "ZhengPeng7/BiRefNet_512x512", "678c37f9158280a2b3000b566ecd12bfbfd1a09c",
              "General 512", "general", "Low-resolution general segmentation",
              _L, (512, 512), _SL, "fp16", "Tuned for 512x512 input: fast previews and small images."),
    ModelSpec("general-lite-2k", "ZhengPeng7/BiRefNet_lite-2K", "3f789d2ba403e2ee59ce08ee9250d4c1c590b418",
              "General Lite 2K", "general", "General segmentation for 2K input",
              _T, (2560, 1440), _ST32, "fp32", "Lightweight backbone trained at 2560x1440 (W x H)."),
    ModelSpec("general-legacy", "ZhengPeng7/BiRefNet-legacy", "39bdbfc112efe2b2da0da75522fb3d5c51993f3a",
              "General legacy", "general", "Previous general-use model",
              _L, (1024, 1024), _SL32, "fp32", "Trained without portrait/matting data: no transparency."),
    # Matting (hair, fur, glass, smoke)
    ModelSpec("matting", "ZhengPeng7/BiRefNet-matting", "eccde0a8cbdce7ac5fecfeb06340fe7b949e85d9",
              "Matting", "matting", "Trimap-free general matting",
              _L, (1024, 1024), _SL32, "fp32", "Soft alpha for hair, fur and transparent objects."),
    ModelSpec("matting-hr", "ZhengPeng7/BiRefNet_HR-matting", "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f",
              "Matting HR", "matting", "High-resolution matting",
              _L, (2048, 2048), _SL, "fp16", "Trained at 2048x2048. Needs ~5 GB of VRAM."),
    ModelSpec("matting-dynamic", "ZhengPeng7/BiRefNet_dynamic-matting", "074df545be87034e74a96bf71566ecbbc4c15f0a",
              "Matting Dynamic", "matting", "Matting at any resolution",
              _L, None, _SL, "fp16", "Trained on random sizes 256-2304."),
    ModelSpec("matting-lite", "ZhengPeng7/BiRefNet_lite-matting", "99c33412e3f58e1f33187abdc8c435c645243690",
              "Matting Lite", "matting", "Matting, lightweight backbone",
              _T, (1024, 1024), 88_965_896, "fp16", "The model card states no resolution; 1024 is inferred."),
    ModelSpec("portrait", "ZhengPeng7/BiRefNet-portrait", "b6561965a70070d9143fd9e558f6ca3c481510db",
              "Portrait", "matting", "Portrait matting",
              _L, (1024, 1024), _SL32, "fp32", "Trained on P3M-10k + TR-humans: people and portraits."),
    # Paper benchmarks
    ModelSpec("dis", "ZhengPeng7/BiRefNet-DIS5K", "343602b3219a0e905a86c5012935d5a4a0882390",
              "DIS5K", "benchmark", "Dichotomous Image Segmentation (DIS5K-TR)",
              _L, (1024, 1024), _SL32, "fp32", "Paper model for objects with very fine structures."),
    ModelSpec("dis-tr-tes", "ZhengPeng7/BiRefNet-DIS5K-TR_TEs", "487f440314ea7ab8ea7d184861953a4010b55587",
              "DIS5K TR+TEs", "benchmark", "DIS trained on DIS5K-TR + test sets",
              _L, (1024, 1024), _SL32, "fp32", "The model card declares no license.", license="unspecified"),
    ModelSpec("hrsod", "ZhengPeng7/BiRefNet-HRSOD", "8a0b1035861383b8a63a622a89a015419194e68d",
              "HRSOD", "benchmark", "High-resolution salient object detection",
              _L, (1024, 1024), _SL32, "fp32", "Trained on DUTS / HRSOD / UHRSD."),
    ModelSpec("cod", "ZhengPeng7/BiRefNet-COD", "6aa747035f7b70caa1c66cd5c128d6e9c654ddd0",
              "COD", "benchmark", "Camouflaged object detection",
              _L, (1024, 1024), _SL32, "fp32", "For objects that blend into their background."),
]}
# fmt: on


def get(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id]
    except KeyError:
        raise KeyError(f"Unknown model {model_id!r}. Available: {', '.join(MODELS)}") from None
