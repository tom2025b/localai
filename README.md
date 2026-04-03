# localai

Local AI personal assistant running **Phi-3 Mini** on your machine. No cloud, no API keys.

- PyQt6 dark-theme GUI with live streaming chat
- Rust sidecar (axum) for Ollama proxying + SHA-256 response caching
- CLI fallback (`assistant.py`) for ask / summarize / remind
- Docker setup included (ollama + localai in one compose file)

## Run

```bash
cd ~/localai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Or via the shell alias (any shell):
```bash
phi3
```

## Docker

```bash
xhost +local:docker
docker compose up -d ollama
docker compose exec ollama ollama pull phi3:mini
docker compose up localai
```

## Shortcuts

| Key | Action |
|-----|--------|
| `+` / `=` | Increase font size (+2pt, max 32pt) |
| `-` | Decrease font size (-2pt, min 12pt) |
| `Enter` | Send message |

## Architecture

```
PyQt6 GUI (main.py)
  └─▶ OllamaWorker (QThread) ──▶ Rust sidecar :8080 (cache hit)
                               └─▶ Ollama :11434 (phi3:mini)
```

Confidence badge in sidebar shows response certainty (green ≥80% / orange ≥60% / red below) when Ollama exposes logprobs.
