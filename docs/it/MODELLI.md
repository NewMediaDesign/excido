# Modelli

[English](../MODELS.md) · Italiano

Il catalogo comprende le 16 varianti ufficiali di BiRefNet pubblicate da **ZhengPeng7** su Hugging Face. Codice e pesi sono MIT, tranne `BiRefNet-DIS5K-TR_TEs`, il cui model card non dichiara una licenza.

Ogni variante è **bloccata a un commit preciso** (`revision` in `excido/catalog.py`), così il `birefnet.py` remoto eseguito con `trust_remote_code` non cambia mai. Revisioni e dimensioni sono state lette dall'API di Hugging Face il 2026-10-12.

I modelli vengono salvati in `models/<NomeRepo>/` (per esempio `models/BiRefNet_HR/`):

| File | Contenuto |
|---|---|
| `model.safetensors` | i pesi |
| `birefnet.py`, `BiRefNet_config.py`, `config.json` | il codice del modello, ispezionabile |
| `README.md` | il model card originale |
| `.revision` | il commit scaricato |

## Varianti

★ indica i modelli scaricati da `excido setup` (il set `core`). Gli altri si scaricano con **Scarica** nel pannello Modello dell'app, oppure con `uv run excido download <id>`.

| ★ | id | Repo Hugging Face | Backbone | Risoluzione | Pesi | A cosa serve |
|---|---|---|---|---|---|---|
| ★ | `general` | ZhengPeng7/BiRefNet | Swin-L | 1024×1024 | 444 MB fp16 | **Predefinito.** Rimozione sfondo generale, anche persone e trasparenze. |
| ★ | `general-hr` | ZhengPeng7/BiRefNet_HR | Swin-L | 2048×2048 | 444 MB fp16 | Foto grandi e dettagli fini. Usalo a 2048: a 1024 rende meno di `general`. |
| ★ | `general-lite` | ZhengPeng7/BiRefNet_lite | Swin-T | 1024×1024 | 178 MB fp32 | Circa 5 volte meno parametri: più veloce, qualità un po' inferiore. |
| ★ | `general-dynamic` | ZhengPeng7/BiRefNet_dynamic | Swin-L | qualsiasi (256–2304) | 444 MB fp16 | Addestrato su dimensioni casuali: lavora alle proporzioni reali dell'immagine. |
| ★ | `matting` | ZhengPeng7/BiRefNet-matting | Swin-L | 1024×1024 | 885 MB fp32 | Matting senza trimap: capelli, pelo, vetro, fumo. |
| ★ | `matting-hr` | ZhengPeng7/BiRefNet_HR-matting | Swin-L | 2048×2048 | 444 MB fp16 | Matting ad alta risoluzione; la demo ufficiale lo usa per i ritratti. |
| ★ | `portrait` | ZhengPeng7/BiRefNet-portrait | Swin-L | 1024×1024 | 885 MB fp32 | Ritratti (P3M-10k + TR-humans). |
|  | `matting-dynamic` | ZhengPeng7/BiRefNet_dynamic-matting | Swin-L | qualsiasi | 444 MB fp16 | Matting a risoluzione libera. |
|  | `matting-lite` | ZhengPeng7/BiRefNet_lite-matting | Swin-T | 1024 (dedotta) | 89 MB fp16 | Matting leggero; il model card non indica la risoluzione. |
|  | `general-512` | ZhengPeng7/BiRefNet_512x512 | Swin-L | 512×512 | 444 MB fp16 | Input piccoli e anteprime veloci. |
|  | `general-lite-2k` | ZhengPeng7/BiRefNet_lite-2K | Swin-T | 2560×1440 | 178 MB fp32 | Immagini 2K orizzontali, con backbone leggero. |
|  | `general-legacy` | ZhengPeng7/BiRefNet-legacy | Swin-L | 1024×1024 | 885 MB fp32 | Vecchio modello generale senza dati di matting: niente trasparenze. |
|  | `dis` | ZhengPeng7/BiRefNet-DIS5K | Swin-L | 1024×1024 | 885 MB fp32 | Modello del paper per la Dichotomous Image Segmentation (strutture finissime). |
|  | `dis-tr-tes` | ZhengPeng7/BiRefNet-DIS5K-TR_TEs | Swin-L | 1024×1024 | 885 MB fp32 | Come `dis`, addestrato anche sui test set. **Nessuna licenza dichiarata.** |
|  | `hrsod` | ZhengPeng7/BiRefNet-HRSOD | Swin-L | 1024×1024 | 885 MB fp32 | Salient object detection ad alta risoluzione. |
|  | `cod` | ZhengPeng7/BiRefNet-COD | Swin-L | 1024×1024 | 885 MB fp32 | Camouflaged object detection. |

```bash
uv run excido download --list
```

```bash
uv run excido download --set all
```

## Quale scegliere

- **Uso quotidiano.** `general`, con risoluzione Auto (1024²).
- **Massima qualità e dettaglio** (gioielli, strutture metalliche, foto grandi):
  - `general-hr` con Auto (2048², FP16). Su una RTX 4070 Laptop richiede circa 1,5 s e 5 GB di VRAM.
  - In alternativa `general-dynamic` con lato massimo 1536–2048, che mantiene le proporzioni.
- **Capelli, pelo, trasparenze.** `matting-hr` (stesso costo di `general-hr`) oppure `matting`.
- **Persone.** `portrait` oppure `matting`.
- **Velocità o poca memoria.** `general-lite`; `general-512` per le anteprime.

## Risultati pubblicati

S-measure su DIS-VD, più alto è meglio:

| Modello | 512 | 1024 | 2048 |
|---|---|---|---|
| BiRefNet (general) | — | 0,912 | 0,898 |
| BiRefNet_HR | 0,634 | 0,893 | **0,927** |

**La risoluzione di addestramento conta.** Per questo `auto` usa sempre quella nativa.

## Note sul codice remoto

- Il `Config` in `birefnet.py` è fisso, e quasi tutti i campi definiscono l'architettura addestrata. Tra questi c'è `batch_size`, che decide se esistono i layer BatchNorm. Nessuno di questi campi è esposto.
- `SDPA_enabled` è `False` nel codice Hugging Face, e il suo ramo SDPA ha dei bug. Va lasciato disattivato.
- `handler.py`, usato dagli Inference Endpoints di Hugging Face, non viene scaricato.
