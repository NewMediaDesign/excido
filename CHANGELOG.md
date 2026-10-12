# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-10-13

### Added

- **Mask brush.**
  - Paint to add to or erase from the selection, with an adjustable size (in image pixels) and hardness.
  - Undo, redo and clear, with keyboard shortcuts: <kbd>B</kbd>, <kbd>E</kbd>, <kbd>Esc</kbd>, <kbd>[</kbd> <kbd>]</kbd>, <kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>Z</kbd>, and <kbd>Space</kbd>+drag to pan while painting.
  - Strokes are applied on top of the mask settings and are saved with the result. They carry over when you re-run the model on the same image, and reach the cutout, the mask and the layered PSD.
- **Feather.** Softens the mask edge with a Gaussian blur of 0–100 px, applied after grow/shrink. It is available in the web UI, the API, `excido remove --feather` and the MCP tools.
- **History management.**
  - Delete a single result, or clear the whole history after an inline confirmation.
  - The disk space used by `outputs/` is shown above the list.
- **New API endpoints.** `POST /api/runs/{id}/edits`, `DELETE /api/runs/{id}`, `DELETE /api/runs` and `GET /api/storage`.

### Changed

- New README screenshot, made with a photo by Zoltan Tasi on Unsplash. The credit is in `THIRD_PARTY_NOTICES.md`.

### Fixed

- At startup, the loaded-model chip showed the model id (`general-hr`) instead of its name until the next refresh.

## [0.1.0] - 2026-10-13

First public version.

### Added

- **Web app.**
  - Before/after slider; cutout, mask, overlay and side-by-side views; zoom and pan; history; batch processing with ZIP export.
  - English/Italian interface, English by default.
  - Monochrome light and dark themes with bundled fonts: Michroma for the wordmark, Geist for text, Geist Mono for titles and values.
- **Runtime settings.**
  - **Model input**: resolution (auto, custom, original), device (auto, CUDA, Apple Metal, CPU) and precision (FP16, BF16, FP32).
  - **Mask**: threshold, invert, and grow/shrink of the selection in half-pixel steps (grayscale morphology that keeps soft edges soft).
  - **Output**: `refine_foreground` radii; transparent, color, blurred or image backgrounds; PNG or WebP.
- **Layered PSD export.** The subject comes with an editable layer mask, plus the hidden original and the background (when there is one); the merged preview stays transparent. It is available from the web UI, `excido remove --psd`, the MCP tools and `GET /api/runs/{id}/psd`.
- **Models.** The 16 official BiRefNet variants are pinned to Hugging Face revisions. `excido setup` downloads the 7 core models.
- **MCP server.**
  - Streamable HTTP at `/mcp` inside the web app, plus `excido mcp` over stdio.
  - The tools are sandboxed to a workspace folder (default `~/Pictures/Excido`).
  - Image URLs and model downloads are opt-in.
- **`excido` command.** Subcommands `serve`, `setup`, `download`, `examples`, `remove`, `mcp` and `smoke`. Configuration through environment variables (`EXCIDO_HOME`, `EXCIDO_MODELS_DIR`, `EXCIDO_GALLERY`, `EXCIDO_MCP_ROOT`, …).
- **Safety.**
  - A hard VRAM cap prevents the slow Windows "sysmem fallback".
  - DNS-rebinding protection covers the whole app.
  - Uploads and downloads are capped at 100 MB.
  - UI files are always revalidated, so updates never show a stale interface.
- **Cross-platform install.** `uv` with a lockfile for Windows, Linux (CUDA 12.8) and macOS (Apple Silicon).
- **Quality.** A test suite and CI on all three systems, including a real CPU inference with the Lite model.
