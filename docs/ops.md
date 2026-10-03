# Operations

## Health
`GET /health` (alias `/api/v1/health`) returns 200 `{"status":"ok"}` or 503 `{"status":"degraded"}` when
the model key is missing or a configured Firestore is unreachable. Fields: `version` (GIT_SHA),
`analyze_ok_last`, `analyze_5xx_15m`, `last_error_class`.

## Environment
| Var | Default | Meaning |
|---|---|---|
| GEMINI_API_KEY | - | model key |
| FIRESTORE_PROJECT | - | enables persistence |
| MAX_UPLOAD_MB | 30 | upload cap: 413 on a declared Content-Length over the cap (before the body is read) and on the bytes read; also interpolated into the browser pre-check |
| DRISHTI_RATE_LIMIT_PER_MIN | 0 (off) | per-IP limit (429); opt-in, see docs/policy.md |
| DRISHTI_TRUSTED_PROXY_HOPS | 1 | proxies that append to X-Forwarded-For; client IP = entry `-hops` from the right |
| DRISHTI_ANALYSIS_REQUIRES_ADMIN | off | gate `/api/analysis/{id}` behind admin basic auth; opt-in |
| DRISHTI_DAILY_RUN_CAP | 0 (off) | global daily analyses (Firestore `usage` collection, atomic `Increment`); when > 0 it fails closed: 429 when exceeded, 503 if the store errors or is not configured |
| DRISHTI_FILES_API_THRESHOLD_MB | 8 | video, or media at least this large, is uploaded once via the Files API |
| GEMINI_TIMEOUT_S | 120 | pipeline timeout (504) |
| DRISHTI_ANALYSES_TTL_DAYS | 90 | analyses retention |

## Observability
- One JSON log line per analysis: `{"event":"analysis","run_id","stage_latency_ms":{plan,tools,report,total,tool:*},"tokens_in","tokens_out","plan_fallback","report_fallback","verdict","http_status","media_type","media_bytes"}` (failures add `stage` and `error_class`).
- The same fields are stored on the Firestore analysis doc. A failed run writes `{status:"error", stage, error_class, http_status}`.
- `/health` includes `analyze_p50_ms` / `analyze_p95_ms` (rolling, per instance). `GET /health/model` runs a `models.get` reachability probe (503 if the model is unreachable).

## Streaming
`POST /api/analyze/stream` returns NDJSON events `plan`, `tool_started`, `tool_done`, `report`, then `result` (or `error`). The UI drives progress from these and marks unused tools SKIPPED.

## Dependencies
`requirements.txt` holds the top-level pins; `requirements.lock` is the full pip-compile lock used by the Dockerfile and CI
(`pip-compile --strip-extras -o requirements.lock requirements.txt`, then `pip-audit -r requirements.lock`).

## Deploy
Build with `--build-arg GIT_SHA=$(git rev-parse --short HEAD)` so /health reports the version.
Enable the analyses TTL once (manual, Sai): `gcloud firestore fields ttls update expire_at --collection-group=analyses --enable-ttl`.

## Access model
`/api/analysis/{id}` is public by default (as before; the datetime 500 is fixed). Set
`DRISHTI_ANALYSIS_REQUIRES_ADMIN=true` to require admin basic auth. See docs/policy.md for the
SAI-ONLY decisions.
