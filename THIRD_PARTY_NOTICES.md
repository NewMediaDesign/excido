# Third-party notices

Excido is an independent project. It is **not affiliated with or endorsed by** the
BiRefNet authors, Hugging Face, Photoroom or Adobe.

## BiRefNet (code derived from)

`excido/postprocess.py` (`refine_foreground`) is a rewrite of the function of the same
name in `image_proc.py` of <https://github.com/ZhengPeng7/BiRefNet>, distributed under
the following license:

```
MIT License

Copyright (c) 2024 ZhengPeng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Paper: Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang,
Nicu Sebe — *Bilateral Reference for High-Resolution Dichotomous Image Segmentation*,
CAAI Artificial Intelligence Research, 2024.

The foreground estimation algorithm is described in M. Forte and F. Pitié,
*Approximate Fast Foreground Colour Estimation*, ICIP 2021, and implemented in
<https://github.com/Photoroom/fast-foreground-estimation>. Excido contains no code from
that repository.

## Bundled fonts

The web interface ships these fonts in `excido/static/fonts/`, each under the
SIL Open Font License 1.1 (full text next to each font file):

| Font | Files | Copyright |
|---|---|---|
| Michroma | `Michroma-latin-400.woff2`, `OFL-Michroma.txt` | © 2011 The Michroma Project Authors |
| Geist | `Geist-latin-var.woff2`, `OFL-Geist.txt` | © 2024 The Geist Project Authors |
| Geist Mono | `GeistMono-latin-var.woff2`, `OFL-GeistMono.txt` | © 2024 The Geist Project Authors |

The WOFF2 subsets come from the Fontsource packages (`@fontsource/michroma`,
`@fontsource-variable/geist`, `@fontsource-variable/geist-mono`, version 5.3.0).

## Model weights (downloaded, not redistributed)

Excido downloads the BiRefNet weights and model code (`birefnet.py`) from
<https://huggingface.co/ZhengPeng7> at pinned revisions, on the user's request. They
are not part of this repository. Their model cards declare the MIT license, except
`ZhengPeng7/BiRefNet-DIS5K-TR_TEs`, whose card declares none.

The models were trained on public datasets (DIS5K, P3M-10k, AM-2k, DUTS, HRSOD, …)
that carry their own terms, some of them research-only. If you plan commercial use,
check those terms yourself.

## Sample images (downloaded, not redistributed)

`excido examples` fetches the sample photos of the official demo from the Hugging Face
Space <https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo>. They belong to their
authors and are used only for local testing; they are not included in this repository.

## Python dependencies

Installed by the package manager, not bundled: PyTorch and torchvision (BSD-3-Clause),
Transformers and huggingface_hub (Apache-2.0), timm (Apache-2.0), Kornia (Apache-2.0),
einops (MIT), Accelerate (Apache-2.0), NumPy (BSD-3-Clause), Pillow (MIT-CMU),
FastAPI (MIT), Uvicorn (BSD-3-Clause), python-multipart (Apache-2.0), Requests (Apache-2.0),
MCP Python SDK (MIT), psd-tools (MIT).
