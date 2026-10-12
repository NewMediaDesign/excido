# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-12

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
- **`excido` command.** Subcommands `serve`, `setup`, `download`, `examples`, `remove`, `mcp` and `smoke`.
- **Safety.**
  - A hard VRAM cap prevents the slow Windows "sysmem fallback".
  - DNS-rebinding protection covers the whole app.
  - Uploads and downloads are capped at 100 MB.
  - UI files are always revalidated, so updates never show a stale interface.
- **Cross-platform install.** `uv` with a lockfile for Windows, Linux (CUDA 12.8) and macOS (Apple Silicon).
- **Quality.** A test suite and CI on all three systems, including a real CPU inference with the Lite model.
