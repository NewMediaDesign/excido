# Parametri

[English](../PARAMETERS.md) · Italiano

I parametri sono di due famiglie, e ogni sezione del pannello destro riporta la sua etichetta:

- **esegui**: cambiano ciò che entra nel modello. Per applicarli serve una nuova inferenza: il pulsante *Esegui* diventa *Riesegui per applicare*.
- **live**: agiscono sulla maschera già calcolata, che resta in memoria. Si applicano subito, senza rieseguire il modello.

Gli stessi nomi si usano nell'API (`/api/segment`, `/api/runs/{id}/postprocess`), negli strumenti MCP e, con opzioni equivalenti, in `excido remove`.

## Modello (esegui)

| Parametro | Valori | Default | Note |
|---|---|---|---|
| `model` | un id di [MODELLI.md](MODELLI.md) | `general` | Determina backbone e risoluzione nativa. In memoria resta un solo modello; cambiarlo lo ricarica in qualche secondo. |

## Risoluzione di input (esegui)

BiRefNet ridimensiona l'immagine a una risoluzione fissa (**H** × **W**), **senza mantenere le proporzioni**, esattamente come il codice ufficiale. Poi riporta la maschera alla dimensione originale.

| Parametro | Valori | Default | Note |
|---|---|---|---|
| `resolution` | `auto` · `custom` · `original` | `auto` | Vedi le tre modalità sotto. |
| `width`, `height` | 256–4096 | 1024 | Usati solo con `custom`. |
| `max_side` | 256–4096 | 2048 | Limite del lato lungo per `original` e per i modelli dinamici con `auto`. |

Le tre modalità:
- **`auto`** usa la risoluzione nativa del modello: 1024², 2048², 512² oppure 2560×1440. I modelli dinamici usano invece la dimensione dell'immagine, ridotta finché il lato lungo non rientra in `max_side`.
- **`custom`** usa larghezza e altezza indicate.
- **`original`** mantiene le proporzioni dell'immagine, con il lato lungo limitato a `max_side`. È pensata per i modelli dinamici; con gli altri funziona, ma lontano dalla risoluzione nativa la qualità cala.

**Vincolo dell'architettura.** Larghezza e altezza devono essere **multipli di 32**, perché il decoder divide l'input in patch allineate alla feature map a 1/32. Vengono sempre arrotondate per difetto, con un minimo di 256.

**La memoria** cresce con il numero di pixel: 2048² ne richiede circa 4 volte più di 1024². Le misure sono nel README.

## Prestazioni (esegui)

| Parametro | Valori | Default | Effetto |
|---|---|---|---|
| `device` | `auto` · `cuda` · `mps` · `cpu` | `auto` | `auto` sceglie NVIDIA CUDA, poi Apple Metal (MPS), poi la CPU. Un dispositivo non disponibile ripiega sul migliore disponibile. |
| `precision` | `fp16` · `bf16` · `fp32` | `fp16` | Vedi le tre opzioni sotto. |
| `matmul_precision` | `high` · `highest` | `high` | `torch.set_float32_matmul_precision`: con `high` le matmul FP32 usano TF32 sui core tensor. Conta solo in FP32. |
| `batch_size` | 1–8 | 1 | Solo scheda Batch: le immagini con la stessa risoluzione di input vengono elaborate insieme. Non è il `Config.batch_size` fissato nei pesi. |
| `oom_fallback` | `error` · `cpu` | `error` | Se la memoria GPU finisce: errore con suggerimenti, oppure nuovo tentativo su CPU (lento). |
| `keep_loaded` | sì / no | sì | Tiene il modello in memoria tra un'esecuzione e l'altra. Se lo disattivi la memoria si libera, ma ogni esecuzione ricarica il modello. |
| `mask_interp` | `bilinear` · `bicubic` · `lanczos` | `bilinear` | Come la maschera viene riportata alla dimensione originale. `bilinear` è quella di `inference.py`; il model card usa il bicubic di PIL. |

Le tre opzioni di `precision`:
- **`fp16`** mette pesi e input a metà precisione (`model.half()`, come nel model card). È la più veloce e usa meno memoria, con una perdita di qualità praticamente nulla secondo l'autore. Misurato: 0,40 s e 1,6 GB a 1024², 1,5 s e 5,0 GB a 2048².
- **`bf16`** mantiene i pesi in FP32 con `torch.autocast(bfloat16)`, come `inference.py` della repo. È disponibile su NVIDIA Ampere o successive; su Metal ripiega su `fp16`. Misurato: 0,56 s e 3,0 GB.
- **`fp32`** è la piena precisione. Misurato: 0,72 s e 3,1 GB a 1024²; oltre 8 GB a 2048². Sulla CPU si usa sempre questa.

L'autore indica 3,5 GB (FP16) e 4,8 GB (FP32) a 1024², ma si tratta della memoria riservata dal processo. I numeri sopra sono il picco dei tensori (`max_memory_allocated`) su una RTX 4070 Laptop.

**Limite di VRAM.** Al primo caricamento su GPU, Excido limita la propria memoria a quella libera meno 512 MB. Oltre il limite ottieni subito un errore di memoria esaurita, invece del lento "sysmem fallback" silenzioso del driver su Windows. Puoi cambiarlo con `EXCIDO_VRAM_FRACTION` (per esempio `0.9`).

## Maschera (live)

| Parametro | Valori | Default | Effetto |
|---|---|---|---|
| `mask_mode` | `soft` · `binary` | `soft` | `soft` usa la sigmoide come alpha (0–255): bordi morbidi, capelli, trasparenze. `binary` produce solo 0 o 255. |
| `threshold` | 0–1 | 0,5 | Soglia per `binary`; alzarla rende il soggetto più stretto. |
| `invert` | sì / no | no | Inverte la maschera: tiene lo sfondo ed elimina il soggetto. |
| `grow` | −100…+100 px, a passi di 0,5 (1 in modalità binaria) | 0 | Espande (+) o contrae (−) la selezione. I bordi morbidi mantengono il loro profilo e si spostano in blocco, quindi i capelli restano sfumati. 1–3 px tolgono gli aloni dello sfondo o recuperano parti sottili. |

Il codice ufficiale non applica soglie: la maschera morbida è l'output nativo di BiRefNet.

**Come funziona espandi/contrai.** Applica all'alpha una dilatazione (espandi) o un'erosione (contrai) in scala di grigi. I passi alternano una croce a 4 vicini e un quadrato 3×3, il che approssima un disco: il bordo si sposta della stessa distanza in tutte le direzioni, esattamente 1 px lungo gli assi per `±1`. Un valore a mezzo pixel fonde i due raggi interi più vicini.

Viene applicato dopo soglia e inversione, quindi un valore positivo allarga sempre ciò che viene tenuto. Nell'interfaccia web è uno slider da −25 a +25 px, con i pulsanti −/+, un valore digitabile (anche oltre ±25) e i tasti ↑ ↓ (Maiusc moltiplica il passo per 10). Il doppio clic sullo slider lo azzera.

## Colori del bordo: `refine_foreground` (live)

| Parametro | Valori | Default | Effetto |
|---|---|---|---|
| `refine` | sì / no | sì | Stima i colori puliti del primo piano sui bordi semitrasparenti e toglie l'alone del vecchio sfondo. |
| `refine_r1` | 1–255 | 90 | Raggio della prima stima (larga). |
| `refine_r2` | 1–63 | 6 | Raggio della seconda passata (fine); nell'originale è fisso a 6. |

È la stima blur-fusion di Forte & Pitié (ICIP 2021), usata dalla demo ufficiale. È reimplementata con somme cumulative, quindi costa O(1) per pixel qualunque sia il raggio. Conta soprattutto per i capelli su sfondo trasparente.

## Sfondo (live)

| Parametro | Valori | Default | Effetto |
|---|---|---|---|
| `background` | `transparent` · `color` · `blur` · `image` | `transparent` | Vedi i quattro casi sotto. |
| `bg_color` | `#RRGGBB` | `#ffffff` | Tra i preset ci sono il green screen `#00b140` e il blue screen `#0047bb`. |
| `bg_blur` | 1–100 px | 20 | Raggio della sfocatura gaussiana per `blur`. |
| `bg_image_id` | id da `/api/backgrounds` | — | Immagine di sfondo, adattata come `object-fit: cover`. |

I quattro casi di `background`:
- **`transparent`** produce un PNG o WebP RGBA.
- **`color`** compone il soggetto su un colore pieno.
- **`blur`** lo compone sull'originale sfocato, con un effetto ritratto.
- **`image`** lo compone su un'immagine caricata.

## Esportazione (live)

| Parametro | Valori | Default | Note |
|---|---|---|---|
| `export_format` | `png` · `webp` | `png` | Il WebP è lossless, quindi conserva l'alpha. La maschera è sempre un PNG a 8 bit. |

**PSD a livelli.** Il pulsante **PSD** (oppure `excido remove --psd`, oppure `psd=true` negli strumenti MCP) produce un file Photoshop. I livelli, dal basso verso l'alto, sono **Sfondo** (solo con sfondo colore, sfocato o immagine), **Originale** (nascosto, come riferimento) e **Soggetto**: la foto con una **maschera di livello** modificabile che contiene l'alpha. Dentro la maschera i pixel del soggetto hanno i colori del bordo ripuliti; fuori sono la foto originale, quindi ridipingendo la maschera riappare l'immagine vera. L'export usa le impostazioni di maschera e sfondo mostrate in quel momento, e i nomi dei livelli seguono la lingua dell'interfaccia. Con sfondo trasparente l'anteprima composita resta trasparente. Si apre in Photoshop, Affinity, GIMP e Photopea.

## Cosa non è esposto, e perché

| Voce | Motivo |
|---|---|
| Backbone, `mul_scl_ipt`, `dec_att`, `dec_blk`, `cxt_num`, `out_ref`, … | Definiscono l'architettura addestrata: cambiarli rompe il caricamento dei pesi. |
| `Config.size` | Il codice del modello non lo legge mai: la risoluzione la decide il tensore in input. |
| `SDPA_enabled` | Nel codice Hugging Face ha dei bug (ignora il bias di posizione e la maschera delle finestre), quindi resta disattivato. |
| `torch.compile` | L'autore lo sconsiglia per l'inferenza: poco guadagno, primo avvio lunghissimo, e su Windows richiede Triton. |
| ONNX / TensorRT | Con Swin-L ONNX è circa il 90% più lento di PyTorch; TensorRT esiste solo in progetti di terzi. |
| Normalizzazione | Media e deviazione standard ImageNet, fisse nell'addestramento. |
