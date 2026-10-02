# Autonomy boundary and change policy (PROPOSED - needs Sai's approval)

## Proposed `non_goals` line (for Sai to approve into the product registry)

> Drishti is a demo only. It is not a validated GMP / regulatory compliance system, its verdicts are
> advisory model output, and it must not be used as the sole basis for a release, batch or
> regulatory decision.

## Change classes

| Class | Examples | Who decides |
|---|---|---|
| **AUTO** (may be drafted as a PR and auto-merged once CI is green) | Dependency bumps within pinned majors, bug fixes, tests, docs, perf and a11y fixes, log/telemetry fields, fail-closed hardening of existing behaviour | CI gate; operator merge |
| **SAI-ONLY** (decision card, never auto-merged) | Authentication or access-model changes on any endpoint (`/api/analysis/{id}`, `/admin`, `/api/analyze*`); enabling or tuning rate limits or the daily spend cap; changing the model (`MODEL`) or major SDK versions; changing the verdict rules or the "never PASS without evidence" invariant; data retention (TTL) changes; anything that adds user-visible cost, new public endpoints, or new data collection; deploys and Firestore/IAM changes; edits to this policy or `non_goals` | Sai |

## Opt-in switches shipped by the hardening PR (defaults preserve prior behaviour)

| Setting | Default | Decision needed from Sai |
|---|---|---|
| `DRISHTI_RATE_LIMIT_PER_MIN` | `0` (off) | Enable a per-IP limit on `/api/analyze`? Uses the trusted proxy hop (`DRISHTI_TRUSTED_PROXY_HOPS`, default 1 for Cloud Run), never the client-supplied leftmost `X-Forwarded-For` entry. |
| `DRISHTI_ANALYSIS_REQUIRES_ADMIN` | off (public, as before; datetime serialisation bug fixed) | Put `/api/analysis/{id}` behind admin basic auth? |
| `DRISHTI_DAILY_RUN_CAP` | `0` (off) | Set a daily cap? When > 0 it fails closed (503) if Firestore cannot be reached. |

## Verdict invariant (do not weaken without Sai)

A stored verdict is PASS only if no check failed, at least one check passed, the model's
`finalVerdict` is PASS, and the report validates (>= 3 checks, all statuses valid). Anything else is
NEUTRAL with `report_fallback=true` (a recorded FAIL is never downgraded).
