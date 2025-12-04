# Local LLM API

A REST API for text generation using TinyLlama. Runs locally on your machine.

## Run

```bash
docker compose up -d
```

## Test

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "token: SUPER_SECRET_TOKEN_123" \
  -d '{"prompt": "What is an API?", "max_tokens": 100}'
```

## Stop

```bash
docker compose down
```

## Files

- `docker-compose.yml` - Defines the API container, port (8000), and environment variables
- `api/Dockerfile` - Installs Python, PyTorch, and dependencies
- `api/app.py` - FastAPI server that loads TinyLlama model and exposes `/generate` endpoint
- `training/` - Optional scripts for fine-tuning (not required to run)

## app.py

Loads a pre-trained language model from HuggingFace on first request. Accepts POST requests with a prompt, generates a response using the model, and returns it as JSON. Requires a token header for authentication.
