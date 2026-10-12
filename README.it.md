<p align="center"><img src="docs/assets/excido-banner.jpg" width="100%" alt="Excido"></p>
<p align="center"><a href="README.md">English</a> · <b>Italiano</b></p>
<p align="center"><a href="https://github.com/NewMediaDesign/excido/actions/workflows/ci.yml"><img src="https://github.com/NewMediaDesign/excido/actions/workflows/ci.yml/badge.svg" alt="CI"></a> <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT"></p>

Excido è uno strumento locale per scontornare le immagini, basato su [BiRefNet](https://github.com/ZhengPeng7/BiRefNet), lo stato dell'arte nella segmentazione dicotomica ad alta risoluzione (Zheng Peng et al., CAAI AIR 2024). Il modello gira sulla tua GPU o CPU e l'interfaccia si apre nel browser. Le immagini non escono mai dal tuo computer.

<p align="center"><img src="docs/assets/screenshot.jpg" width="100%" alt="App web di Excido: vista prima/dopo di una bicicletta scontornata"></p>

- **App web.** Ha uno slider prima/dopo, il ritaglio su scacchiera, le viste maschera e overlay, zoom e pan, cronologia, batch con esportazione ZIP e interfaccia in inglese e italiano.
- **Tutti i parametri a runtime.** Controlli risoluzione, precisione, dispositivo, soglia, espansione/contrazione della selezione a mezzo pixel, pulizia dei bordi (`refine_foreground`), sfondo e formato ([docs/it/PARAMETRI.md](docs/it/PARAMETRI.md)).
- **16 varianti ufficiali di BiRefNet.** Ci sono General, HR, Lite, Dynamic, Matting, Portrait e i modelli del paper. Sette si scaricano durante il setup, le altre con un clic ([docs/it/MODELLI.md](docs/it/MODELLI.md)).
- **Server MCP.** Claude e gli altri client MCP possono scontornare le immagini per te, dentro una cartella che scegli tu ([docs/it/MCP.md](docs/it/MCP.md)).
- **Export PSD a livelli.** Il soggetto ha una maschera di livello modificabile, più l'originale nascosto e lo sfondo.
- **Riga di comando.** `excido remove` elabora file o cartelle con gli stessi parametri dell'app web.
- **Multipiattaforma.** Gira su NVIDIA (CUDA) con Windows e Linux, su Apple Silicon (Metal) con macOS, oppure sulla CPU ovunque.

## Requisiti

| | Windows 10/11 | Linux | macOS |
|---|---|---|---|
| Accelerazione | NVIDIA, driver 570+ (CUDA 12.8) | NVIDIA, driver 570+ (CUDA 12.8) | Apple Silicon (M1 o successivi), Metal |
| Senza GPU | CPU (lento) | CPU (lento) | I Mac Intel non sono supportati (mancano build recenti di PyTorch) |
| Disco | ~10 GB (PyTorch e i 7 modelli principali) | uguale | ~5 GB |

Serve anche [uv](https://docs.astral.sh/uv/), che installa Python 3.12 e le dipendenze in una `.venv` dentro il progetto. La connessione internet serve solo durante il setup.

## Avvio rapido

**1. Installa uv** (una volta sola).

Windows (PowerShell):

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS e Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Scarica il codice e installa Excido.** Scarica circa 3 GB di PyTorch e 3,7 GB di modelli, quindi servono 10–30 minuti.

```bash
git clone https://github.com/NewMediaDesign/excido.git
```

```bash
cd excido
```

```bash
uv sync
```

```bash
uv run excido setup
```

`excido setup --models all` scarica tutti i 16 modelli (circa 10 GB). `--models none` non ne scarica nessuno: li scarichi poi dall'app. Su Windows puoi usare `.\setup.ps1`, su macOS e Linux `bash setup.sh`.

**3. Avvialo.** L'app apre <http://127.0.0.1:7860> nel browser.

```bash
uv run excido
```

Opzioni utili:
- `--offline` non contatta mai Hugging Face.
- `--port 8000` cambia la porta.
- `--no-browser` non apre il browser.

Gli script `.\run.ps1` e `bash run.sh` avviano l'app allo stesso modo.

## Come si usa

1. **Apri un'immagine.** Clicca un esempio, trascina un file nella finestra, incollalo con <kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>V</kbd> oppure indica un URL.
2. Il modello la elabora e il risultato compare con lo **slider prima/dopo**.
3. **Regola i parametri a destra.**
   - Le sezioni **live** (Maschera, Colori del bordo, Sfondo, Esportazione) si aggiornano subito.
   - Le sezioni **esegui** (Modello, Risoluzione, Prestazioni) richiedono una nuova esecuzione: il pulsante diventa *Riesegui per applicare*.
4. **Scarica** il ritaglio (PNG/WebP con trasparenza, oppure un composito), la maschera, oppure un **PSD a livelli** con maschera di livello modificabile.
5. **Batch.** Nella scheda Batch elabori molte immagini insieme e scarichi uno ZIP.

| Tasto | Azione |
|---|---|
| <kbd>1</kbd>–<kbd>5</kbd> | Prima/Dopo, Ritaglio, Maschera, Overlay, Affiancati |
| <kbd>Spazio</kbd> (tieni premuto) | Mostra l'originale |
| <kbd>F</kbd> / <kbd>0</kbd> | Adatta / 100% |
| rotella, trascina, doppio clic | Zoom, spostamento, adatta |
| <kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>Invio</kbd> | Esegui (o avvia il batch) |

L'interfaccia parte in inglese. Il selettore **EN | IT** in alto ricorda la tua scelta.

## Riga di comando

```bash
uv run excido remove foto/ --out risultati/ --model general-hr
```

```bash
uv run excido remove foto.jpg --background "#ffffff" --format webp
```

```bash
uv run excido download --list
```

`excido --help` e `excido <comando> --help` elencano tutte le opzioni. Il comando `excido smoke` fa una verifica end-to-end dei modelli scaricati, con tempi e memoria.

## Server MCP (Claude e altri agenti)

Mentre l'app è aperta espone un endpoint MCP su `http://127.0.0.1:7860/mcp`. Gli strumenti leggono immagini solo da una **cartella di lavoro**, di default `~/Pictures/Excido`, e scrivono solo nella sua sottocartella `excido-output/`. URL e download dei modelli restano disattivati finché non li abiliti.

Per collegarlo a Claude Code:

```bash
claude mcp add --transport http excido http://127.0.0.1:7860/mcp
```

Claude Desktop, e qualunque altro client che avvia comandi locali, usa `excido mcp`. Aggiungi questo a `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "excido": {
      "command": "uv",
      "args": ["run", "--directory", "/percorso/di/excido", "excido", "mcp"]
    }
  }
}
```

Strumenti, opzioni e note di sicurezza sono in [docs/it/MCP.md](docs/it/MCP.md).

## Prestazioni misurate

Le misure vengono da una RTX 4070 Laptop (8 GB) con PyTorch 2.11 e CUDA 12.8, su un'immagine 3577×2163, dopo il warm-up.

| Modello | Input | Precisione | Inferenza | Memoria picco | Richiesta completa\* |
|---|---|---|---|---|---|
| General | 1024² | FP16 | 0,40 s | 1,6 GB | 1,4 s |
| General | 1024² | BF16 | 0,56 s | 3,0 GB | 1,6 s |
| General | 1024² | FP32 | 0,72 s | 3,1 GB | 1,9 s |
| General Lite | 1024² | FP16 | 0,15 s | 0,8 GB | 1,1 s |
| Matting / Portrait | 1024² | FP16 | 0,41 s | 1,6 GB | 1,4 s |
| General Dynamic | 2048×1216 | FP16 | 0,87 s | 3,1 GB | 1,9 s |
| General HR / Matting HR | 2048² | FP16 | 1,5 s | 5,0 GB | 2,6 s |
| General HR | 2304² (FP16) o 2048² (FP32) | | | oltre il limite | errore OOM chiaro in 3–4 s |
| General Lite su CPU | 1024² | FP32 | 6–12 s | — | |

\* La richiesta completa comprende decodifica, inferenza, `refine_foreground`, composizione e salvataggio dei PNG.

## Struttura del progetto

```
excido/            pacchetto Python: motore, post-processing, server web, server MCP, CLI
excido/static/     interfaccia web (HTML, CSS, JS; font inclusi, funziona offline)
tests/             test pytest (non servono GPU né download)
docs/              architettura, parametri, modelli, MCP (inglese); docs/it/ in italiano
pyproject.toml     dipendenze; uv.lock le blocca per Windows, Linux e macOS
models/            pesi scaricati               (creata a runtime, esclusa da git)
examples/          esempi ufficiali             (creata a runtime, esclusa da git)
outputs/           risultati e cronologia       (creata a runtime, esclusa da git)
demo/              tue immagini di prova, seconda galleria (facoltativa, esclusa da git)
```

## Configurazione

| Variabile | Effetto |
|---|---|
| `EXCIDO_HOME` | Cartella dei dati: modelli, esempi, risultati. Di default è il progetto (o `~/.excido` se installato come pacchetto). |
| `EXCIDO_MODELS_DIR` | Dove salvare i modelli (default `<EXCIDO_HOME>/models`), per esempio su un disco più capiente. |
| `EXCIDO_GALLERY` | Cartella mostrata come seconda galleria (default `demo/`). |
| `EXCIDO_VRAM_FRACTION` | Limite alla memoria GPU, tra 0 e 1 (vedi [docs/it/ARCHITETTURA.md](docs/it/ARCHITETTURA.md)). |
| `EXCIDO_MCP_ROOT` | Cartella di lavoro MCP (default `~/Pictures/Excido`). |
| `EXCIDO_MCP` | Impostala a `0` per disattivare l'endpoint `/mcp`. |
| `HF_TOKEN` | Token Hugging Face facoltativo, per download più veloci. |

## Risoluzione dei problemi

| Problema | Soluzione |
|---|---|
| Il chip del dispositivo dice "Solo CPU" su un PC NVIDIA | Controlla `nvidia-smi`, aggiorna il driver alla 570 o successiva e rilancia `uv sync`. |
| "Memoria GPU esaurita" | Abbassa la risoluzione, usa FP16, prova un modello Lite o attiva il ripiego su CPU. |
| Un'elaborazione dura minuti invece di secondi | Un altro programma sta usando la memoria della GPU. Su Windows il driver NVIDIA sposta allora la memoria nella RAM di sistema e tutto diventa lentissimo. Chiudi l'altro programma e riavvia Excido. |
| "Il modello non è ancora scaricato" | Usa **Scarica** nel pannello Modello, oppure lancia `uv run excido download <id>`. |
| La porta 7860 è occupata | `uv run excido serve --port 8000` |

## Sviluppo

```bash
uv run pytest
```

```bash
uv run ruff check
```

```bash
uv run ruff format
```

La CI esegue lint, test e un'inferenza reale su CPU con il modello Lite, su Windows, Linux e macOS. Vedi [CONTRIBUTING.md](CONTRIBUTING.md) e il [codice di condotta](CODE_OF_CONDUCT.md). I problemi di sicurezza vanno segnalati in privato come spiegato in [SECURITY.md](SECURITY.md).

## Licenza e crediti

Excido è rilasciato con licenza [MIT](LICENSE). È un progetto indipendente, non affiliato agli autori di BiRefNet, a Hugging Face o ad Adobe.

- **BiRefNet.** Codice e pesi sono di Peng Zheng et al., licenza MIT. I pesi vengono scaricati da [huggingface.co/ZhengPeng7](https://huggingface.co/ZhengPeng7) e non sono ridistribuiti qui.
- **Terze parti.** L'attribuzione di `refine_foreground`, i font inclusi (Michroma, Geist, Geist Mono, tutti SIL OFL) e le note su dataset e immagini di esempio sono in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
