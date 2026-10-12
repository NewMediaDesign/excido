# Server MCP

[English](../MCP.md) · Italiano

Excido parla il [Model Context Protocol](https://modelcontextprotocol.io), quindi Claude (Claude Code, Claude Desktop) e gli altri client MCP possono scontornare immagini al posto tuo. Puoi chiedere cose come *"scontorna tutte le foto della cartella scarpe e mettile su bianco"* oppure *"i capelli sono venuti male, riprova con un modello di matting"*. Tutto continua a girare sul tuo computer.

## Due modi per collegarsi

| Modalità | Comando | Ideale per |
|---|---|---|
| **HTTP**, integrata nell'app web | `uv run excido`, poi collegati a `http://127.0.0.1:7860/mcp` | Claude Code e i client che supportano Streamable HTTP. Condivide modello e cronologia dell'app, quindi quello che fa l'agente compare nell'interfaccia web. |
| **stdio** | `uv run excido mcp` | Claude Desktop e i client che avviano un comando locale. Se l'app web è aperta le passa il lavoro (un solo processo usa la GPU); altrimenti carica il modello da sé. |

### Claude Code

```bash
claude mcp add --transport http excido http://127.0.0.1:7860/mcp
```

Oppure, senza tenere aperta l'app web:

```bash
claude mcp add excido -- uv run --directory /percorso/di/excido excido mcp
```

### Claude Desktop

Modifica `claude_desktop_config.json` (*Impostazioni → Sviluppatore → Modifica configurazione*):

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

Su Windows scrivi il percorso come `C:\\Users\\tu\\excido`. Se Claude Desktop non trova `uv`, metti in `command` il percorso completo dell'eseguibile `uv`.

## La cartella di lavoro

Gli strumenti possono **leggere immagini solo da una cartella** e **scrivere solo nella sua sottocartella `excido-output/`**.

- **Default**: `~/Pictures/Excido` (`C:\Users\<tu>\Pictures\Excido` su Windows). Viene creata al primo utilizzo.
- **Per cambiarla**: usa `--mcp-root <cartella>` con `excido serve` o `excido mcp`, oppure la variabile `EXCIDO_MCP_ROOT`.
- **Come viene applicata**:
  - I percorsi sono relativi alla cartella.
  - `..`, percorsi assoluti altrove e collegamenti simbolici che puntano fuori vengono rifiutati.
  - I risultati non sovrascrivono mai file esistenti: viene aggiunto un suffisso `-1`, `-2`…
- **L'app web non è toccata da questo limite**: continua ad accettare quello che carichi a mano.

Metti le foto nella cartella di lavoro e chiedi all'agente di lavorarci. I risultati compaiono in `excido-output/`, come `<nome>_cutout.png` o `<nome>_composite.png`, più `<nome>_mask.png`.

## Strumenti

| Strumento | Cosa fa |
|---|---|
| `workspace_info` | Mostra il percorso della cartella di lavoro e di quella di output, il numero di immagini e le opzioni attive. |
| `list_images` | Elenca le immagini nella cartella di lavoro o in una sua sottocartella. |
| `remove_background` | Scontorna un'immagine, con `model`, `resolution`, `max_side`, `precision`, `device`, `mask_mode`, `threshold`, `invert`, `grow` (±px, passi di 0,5), `refine`, `background`, `output_format` e `preview`. Restituisce i percorsi dei file prodotti, un `run_id`, i tempi e `foreground_ratio`. |
| `remove_background_batch` | Come sopra, per una lista di immagini o una cartella intera. |
| `recompose` | Cambia maschera o sfondo di un risultato precedente, `grow` compreso, senza rieseguire il modello (veloce). |
| `list_models` | Elenca le 16 varianti: a cosa servono, dimensione, se sono scaricate. |
| `download_model` | Scarica un modello dal catalogo ufficiale bloccato alle revisioni note. È **disattivato di default**. |
| `system_info` | Mostra i dispositivi (CUDA, Metal, CPU), la memoria e il modello caricato. |

`background` accetta quattro valori:
- `transparent`;
- un colore come `#ffffff`;
- `blur` oppure `blur:30`;
- il percorso di un'altra immagine della cartella di lavoro, da mettere dietro il soggetto.

`psd=true` scrive anche `<nome>_layers.psd` in `excido-output/`: il soggetto con maschera di livello modificabile, l'originale nascosto e lo sfondo. Funziona con `remove_background`, `remove_background_batch` e `recompose`.

`preview=true` restituisce anche un piccolo JPEG del risultato, con la trasparenza mostrata su scacchiera. Così l'agente vede il ritaglio e può decidere di riprovare, per esempio con un modello di matting. Le anteprime costano token, quindi sono disattivate di default.

## Opzioni

| Opzione (`serve` / `mcp`) | Variabile d'ambiente | Default |
|---|---|---|
| `--mcp-root <cartella>` | `EXCIDO_MCP_ROOT` | `~/Pictures/Excido` |
| `--mcp-allow-urls` | `EXCIDO_MCP_ALLOW_URLS=1` | disattivato: solo file della cartella di lavoro |
| `--mcp-allow-downloads` | `EXCIDO_MCP_ALLOW_DOWNLOADS=1` | disattivato: l'utente usa `excido download` |
| `--no-mcp` (solo `serve`) | `EXCIDO_MCP=0` | l'endpoint `/mcp` è attivo |
| `--standalone` (solo `mcp`) | `EXCIDO_MCP_STANDALONE=1` | riusa l'app web già aperta |

## Note di sicurezza

- **Tutto resta sul computer che esegue Excido.** Pubblicare o clonare la repository non espone nulla: ogni utente esegue la propria copia, sul proprio computer.
- **L'endpoint `/mcp` risponde solo a richieste da `localhost`.** Controlla gli header `Host` e `Origin`, il che blocca gli attacchi DNS rebinding da pagine web. Vale anche se l'app web è avviata con `--host 0.0.0.0`.
- **Tutta l'app web rifiuta header `Host` estranei**, per lo stesso motivo.
- **I client MCP di solito chiedono conferma prima di ogni chiamata a uno strumento.** Tienila attiva per gli strumenti che scrivono file.
