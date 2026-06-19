# ML Inference Platform

A Ray Serve based multi-model inference platform. A single FastAPI gateway
fronts independently scalable model deployments, each registered through a
lightweight model registry — adding a new model means adding files, not
touching the gateway or deploy logic.

## Architecture

```
                      ┌─────────────────────┐
   HTTP requests ───▶ │   IngressDeployment   │   (FastAPI gateway, 1 replica)
                      │  /sentiment/predict   │
                      │  /summarization/...   │
                      └──────────┬────────────┘
                                 │  DeploymentHandle.predict.remote(...)
              ┌──────────────────┼──────────────────┐
              ▼                                      ▼
   ┌─────────────────────┐               ┌─────────────────────┐
   │ SentimentClassifier  │               │     Summarizer        │
   │ (2 replicas, CPU)    │               │ (1 replica, CPU)      │
   └─────────────────────┘               └─────────────────────┘
```

- **Gateway** (`server/ingress.py`) holds one `DeploymentHandle` per
  registered model and mounts a FastAPI router for each. It never changes
  when a model is added.
- **Models** (`models/<name>/`) are independent Ray Serve deployments —
  scaled, resourced, and versioned separately from each other and from the
  gateway.
- **Registry** (`core/registry.py`) is the seam between the two: models
  self-register on import, the deploy script just iterates the registry.

## Project layout

```
ml_platform/
├── run.py                      # entrypoint: python run.py
├── config/
│   └── settings.py             # env-driven config (pydantic-settings)
├── core/
│   ├── base_deployment.py      # BaseModelDeployment ABC
│   └── registry.py             # model registry
├── models/
│   ├── __init__.py             # imports register all models
│   ├── sentiment/
│   │   ├── deployment.py
│   │   └── schema.py
│   └── summarization/
│       ├── deployment.py
│       └── schema.py
├── api/
│   ├── app.py                  # shared FastAPI() instance
│   └── routes/
│       ├── sentiment.py
│       └── summarization.py
└── server/
    ├── ingress.py               # gateway deployment
    └── deploy.py                 # builds graph + serve.run
```

## Requirements

- Python 3.10+
- `ray[serve]`
- `transformers`
- `fastapi`
- `pydantic` / `pydantic-settings`
- `loguru`
- A backend for `transformers` (`torch` or `tensorflow`)

```bash
pip install "ray[serve]" transformers torch fastapi pydantic-settings loguru
```

## Running locally

```bash
python run.py
```

This will:
1. Initialize Ray (`ray.init`)
2. Start the Serve HTTP proxy on `0.0.0.0:8000` (configurable, see below)
3. Bind and deploy every model registered in `models/__init__.py`
4. Mount the gateway at `/`

On success:

```
Live at http://0.0.0.0:8000
  POST /sentiment/predict
  POST /summarization/predict
  GET  /docs
```

Stop with `Ctrl+C` — this calls `serve.shutdown()` / `ray.shutdown()` cleanly.

## API

### Sentiment

```bash
curl -X POST http://localhost:8000/sentiment/predict \
  -H "Content-Type: application/json" \
  -d '{"text": ["I love this", "This is terrible"]}'
```

```json
{
  "results": [
    {"text": "I love this", "label": "POSITIVE", "score": 0.9998},
    {"text": "This is terrible", "label": "NEGATIVE", "score": 0.9995}
  ]
}
```

### Summarization

```bash
curl -X POST http://localhost:8000/summarization/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "<long article text>", "max_length": 100, "min_length": 30}'
```

```json
{
  "results": [
    {"source_text": "<long article text>", "summary": "<generated summary>"}
  ]
}
```

### Health checks

- `GET /health` — gateway liveness
- `GET /sentiment/health` — sentiment deployment liveness
- `GET /summarization/health` — summarization deployment liveness

### Warmup

Trigger lazy model loading right after startup so end users do not pay first-request latency:

```bash
curl -X POST http://localhost:8000/warmup
```

The warmup runs in the background. Check progress with:

```bash
curl http://localhost:8000/warmup/status
```

Example response:

```json
{
  "status": "started",
  "state": "running",
  "warmed": ["sentiment", "summarization"],
  "failed": {}
}
```

### Docs

Interactive Swagger UI for all mounted routers: `GET /docs`

## Configuration

All settings are environment-driven via `pydantic-settings`, prefixed
`ML_`, with `__` as the nested delimiter.

| Variable | Default | Description |
|---|---|---|
| `ML_HOST` | `0.0.0.0` | HTTP bind host |
| `ML_HTTP_PORT` | `8000` | HTTP bind port |
| `ML_SENTIMENT__MODEL_NAME` | `distilbert-base-uncased-finetuned-sst-2-english` | HF model id |
| `ML_SENTIMENT__NUM_REPLICAS` | `2` | Replica count |
| `ML_SENTIMENT__NUM_CPUS` | `1` | CPUs per replica |
| `ML_SENTIMENT__DEVICE` | `-1` | `-1` = CPU, `0`+ = CUDA device index |
| `ML_SUMMARIZATION__MODEL_NAME` | `sshleifer/distilbart-cnn-12-6` | HF model id |
| `ML_SUMMARIZATION__NUM_REPLICAS` | `1` | Replica count |
| `ML_SUMMARIZATION__NUM_CPUS` | `2` | CPUs per replica |
| `ML_SUMMARIZATION__DEVICE` | `-1` | `-1` = CPU, `0`+ = CUDA device index |

Example, running summarization on GPU 0 with sentiment on CPU:

```bash
ML_SUMMARIZATION__DEVICE=0 python run.py
```

`device` defaults to CPU (`-1`) intentionally — it must be explicitly
opted into GPU per model rather than hardcoded, so the same code runs
unmodified on CPU-only and GPU boxes.

## Adding a new model

1. `models/<name>/schema.py` — request/response Pydantic models.
2. `models/<name>/deployment.py` — subclass `BaseModelDeployment`,
   implement `async def predict(...)`, call `register_model(ModelSpec(...))`
   at module level.
3. `api/routes/<name>.py` — `build_router(handle) -> APIRouter` exposing
   `/predict` and `/health`.
4. `config/settings.py` — add a `<Name>Settings` block and attach it to
   `Settings`.
5. `models/__init__.py` — add `from . import <name>`.

`server/deploy.py` and `server/ingress.py` require **no changes** — they
iterate the registry generically.

## Design notes / tradeoffs

- **Single ingress vs. one ingress per model.** A single FastAPI gateway
  was chosen over per-model HTTP ingresses to avoid running N separate
  HTTP listeners and to give a single `/docs` surface. The tradeoff is
  that gateway deploys/restarts touch all routes at once — acceptable
  here since the gateway is a thin router with no inference logic of its
  own, so it's cheap to redeploy.
- **`predict` as the common entrypoint.** Every model deployment exposes
  `async def predict(...)` so the gateway can call `handle.predict.remote(...)`
  uniformly. Per-model argument shapes (e.g. summarization's
  `max_length`/`min_length`) are handled in that model's own route file,
  not in shared gateway code — keeps the registry/gateway model-agnostic
  without forcing a lowest-common-denominator signature.
- **Resourcing is per-model.** `num_replicas` / `num_cpus` / `device` are
  set independently per model since cost and latency profiles differ
  (e.g. summarization is heavier per-request than sentiment here). These
  defaults are starting points — tune against real traffic and latency
  budgets, not the numbers in this repo.
- **No GPU sharing logic included.** If you need multiple models sharing
  a single GPU, that requires explicit fractional `num_gpus` allocation
  and placement group considerations — not handled by this scaffold as-is.