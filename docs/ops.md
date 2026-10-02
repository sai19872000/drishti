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
| MAX_UPLOAD_MB | 30 | upload cap (413) |
| DRISHTI_RATE_LIMIT_PER_MIN | 10 | per-IP limit (429); 0 disables |
| DRISHTI_DAILY_RUN_CAP | 0 | global daily analyses (Firestore `usage` collection); 0 disables |
| GEMINI_TIMEOUT_S | 120 | pipeline timeout (504) |
| DRISHTI_ANALYSES_TTL_DAYS | 90 | analyses retention |

## Deploy
Build with `--build-arg GIT_SHA=$(git rev-parse --short HEAD)` so /health reports the version.

## Access model
`/api/analysis/{id}` requires admin basic auth (previously public, and broken for post-TTL records).
