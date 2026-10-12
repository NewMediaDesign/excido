"""Download the 7 sample images of the official BiRefNet demo.

The GitHub repository ships no sample images: the official ones live in the
Hugging Face Space ``ZhengPeng7/BiRefNet_demo`` (``examples/``). They are fetched
into ``examples/official/`` and are not redistributed with Excido.
"""

from __future__ import annotations

import requests

from . import EXAMPLES_DIR, __version__

SPACE = "ZhengPeng7/BiRefNet_demo"
FILES = [
    "Helicopter.jpg",
    "Jewelry.jpg",
    "Windmill.jpg",
    "My_Love.jpg",
    "My_Love_1.jpg",
    "My_Love_2.jpg",
    "My_MiSheng.jpg",
]


def download_examples(verbose: bool = True) -> list[str]:
    out = EXAMPLES_DIR / "official"
    out.mkdir(parents=True, exist_ok=True)
    fetched = []
    for name in FILES:
        dest = out / name
        if dest.exists() and dest.stat().st_size > 0:
            if verbose:
                print(f"[present] {name}")
            continue
        url = f"https://huggingface.co/spaces/{SPACE}/resolve/main/examples/{name}"
        r = requests.get(url, timeout=60, headers={"User-Agent": f"Excido/{__version__}"})
        r.raise_for_status()
        if not r.headers.get("content-type", "").startswith("image/"):
            raise RuntimeError(f"{url} is not an image ({r.headers.get('content-type')})")
        dest.write_bytes(r.content)
        fetched.append(name)
        if verbose:
            print(f"[ok] {name} ({len(r.content) / 1e3:.0f} KB)")
    return fetched
