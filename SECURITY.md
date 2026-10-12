# Security policy

English · [Italiano](#politica-di-sicurezza)

## Reporting a vulnerability

Please **don't open a public issue** for security problems. Report them privately with
GitHub's **[Report a vulnerability](https://github.com/NewMediaDesign/excido/security/advisories/new)**
form (Security tab → Advisories). Include what you found, how to reproduce it and which
version you used (`uv run excido --version`).

We'll acknowledge the report as soon as we can, agree on a fix and a disclosure date with
you, and credit you in the advisory unless you prefer otherwise.

## Supported versions

Security fixes go into the latest release. Excido is pre-1.0: please update to the newest
version before reporting.

## Scope and design notes

Excido is meant to run **locally, for one user**:

- The web server listens on `127.0.0.1` by default and has **no authentication**. Starting
  it with `--host 0.0.0.0` exposes it to your network; the app warns you when you do.
- Requests whose `Host` header isn't local are refused (DNS-rebinding protection). The MCP
  endpoint also validates `Origin` and only answers local requests.
- MCP tools can only read images inside one workspace folder and only write into its
  `excido-output/` subfolder. Image URLs and model downloads are off by default.
- Model code (`birefnet.py`) runs with `trust_remote_code` at **pinned Hugging Face
  revisions**; the files stay readable in `models/<Repo>/`.

Reports about any of these protections not working as described are especially welcome.

---

# Politica di sicurezza

[English](#security-policy) · Italiano

**Non aprire issue pubbliche** per i problemi di sicurezza. Segnalali in privato con il modulo
**[Report a vulnerability](https://github.com/NewMediaDesign/excido/security/advisories/new)** di
GitHub (scheda Security → Advisories). Indica cosa hai trovato, come riprodurlo e la versione
usata (`uv run excido --version`).

Risponderemo appena possibile, concorderemo con te correzione e data di pubblicazione, e ti
citeremo nell'avviso se lo desideri. Le correzioni vanno nell'ultima release: aggiorna prima di
segnalare.

Excido è pensato per girare **in locale, per un solo utente**:
- Ascolta su `127.0.0.1` e non ha autenticazione.
- Rifiuta le richieste con `Host` non locale. L'endpoint MCP controlla anche l'`Origin`.
- Gli strumenti MCP sono limitati alla cartella di lavoro.
- Il codice del modello gira a revisioni Hugging Face bloccate.
