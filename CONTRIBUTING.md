# Contributing to Excido

English · [Italiano](#contribuire-a-excido)

Thanks for helping! Bug reports, model/platform test results (especially **Apple Silicon** and **Linux + NVIDIA**), translations and code are all welcome.

## Set up

1. Install [uv](https://docs.astral.sh/uv/).
2. Clone the repository.
3. Install the dependencies and the development tools:

```bash
uv sync
```

4. Download the sample images and the smallest model:

```bash
uv run excido examples
```

```bash
uv run excido download general-lite
```

## Before opening a pull request

```bash
uv run ruff check
```

```bash
uv run ruff format
```

```bash
uv run pytest
```

- The test suite needs no GPU and no downloads. It uses a fake engine and an isolated data folder.
- If you touch inference or devices, also run a real check:

```bash
uv run excido smoke --models general-lite
```

- Keep the UI strings in `excido/static/i18n.js`, in **both** English and Italian. The app defaults to English.
- Code, comments and commit messages are in English.

## Guidelines

- **Keep it local.**
  - No network calls except downloads the user asked for.
  - No telemetry.
  - No CDN in the web UI (fonts and scripts are bundled).
- **Respect the sandbox.** MCP tools must stay inside the workspace folder.
- **Keep it cross-platform.** Use `pathlib`, avoid shell-specific code, and add `.ps1`/`.sh` variants for scripts.
- **Pin model revisions.** New model variants go in `excido/catalog.py` with a pinned `revision`.
- **Don't commit downloaded data.** Model weights and sample images are downloaded at runtime and must never be committed.

## Reporting bugs

Please include:
- the output of `uv run excido --version`;
- your OS, GPU and driver;
- the model and settings you used;
- what you expected and what happened.

`GET /api/system` (or the `system_info` MCP tool) prints most of this.

---

# Contribuire a Excido

[English](#contributing-to-excido) · Italiano

Grazie dell'aiuto! Sono benvenute le segnalazioni di bug, i risultati dei test su modelli e piattaforme (soprattutto **Apple Silicon** e **Linux + NVIDIA**), le traduzioni e il codice.

**Preparazione.** Installa [uv](https://docs.astral.sh/uv/), clona la repository, poi esegui `uv sync`, `uv run excido examples` e `uv run excido download general-lite`.

**Prima di una pull request.** Esegui `uv run ruff check`, `uv run ruff format` e `uv run pytest`. I test non richiedono GPU né download. Se tocchi l'inferenza esegui anche `uv run excido smoke --models general-lite`. I testi dell'interfaccia vanno in `excido/static/i18n.js`, in inglese **e** in italiano. Codice e commenti sono in inglese.

**Principi.**
- Tutto resta in locale: niente telemetria e niente CDN.
- Gli strumenti MCP restano dentro la cartella di lavoro.
- Il codice deve funzionare su Windows, macOS e Linux.
- I modelli si aggiungono in `excido/catalog.py` con una revisione bloccata.
- Pesi e immagini di esempio non si inseriscono mai nella repository.
