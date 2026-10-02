"""
Drishti — Agentic Vision Orchestrator
FastAPI server-side app. All Gemini calls server-side; no key in browser.
"""

import asyncio
import base64
import contextvars
import html
import io
import json
import logging
import os
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("drishti")

# ---------------------------------------------------------------------------
# Config — never fail-fast on missing env vars
# ---------------------------------------------------------------------------
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
FIRESTORE_PROJECT = os.getenv("FIRESTORE_PROJECT") or os.getenv("GCP_PROJECT")
ADMIN_USER = (
    os.getenv("AURACLE_ADMIN_USER")
    or os.getenv("ADMIN_USER")
    or secrets.token_urlsafe(16)
)
ADMIN_PASS = (
    os.getenv("AURACLE_ADMIN_PASS")
    or os.getenv("ADMIN_PASS")
    or secrets.token_urlsafe(32)
)

GIT_SHA = os.getenv("GIT_SHA", "dev")
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "30"))
MAX_QUERY_CHARS = 2000
MAX_SOPS_CHARS = 50000
# Opt-in: default 0 (disabled). Enabling a public rate limit is a SAI-ONLY decision (docs/policy.md).
RATE_LIMIT_PER_MIN = int(os.getenv("DRISHTI_RATE_LIMIT_PER_MIN", "0"))
# Number of trusted proxy hops that append to X-Forwarded-For (Cloud Run front end = 1).
TRUSTED_PROXY_HOPS = max(1, int(os.getenv("DRISHTI_TRUSTED_PROXY_HOPS", "1")))
# Opt-in: gate /api/analysis/{id} behind admin basic auth (default: public, as before).
ANALYSIS_REQUIRES_ADMIN = os.getenv("DRISHTI_ANALYSIS_REQUIRES_ADMIN", "").strip().lower() in ("1", "true", "yes", "on")
# Media at/above this size (and all video) is uploaded once via the Files API.
FILES_API_THRESHOLD_MB = float(os.getenv("DRISHTI_FILES_API_THRESHOLD_MB", "8"))
MIN_CHECKS = 3
DAILY_RUN_CAP = int(os.getenv("DRISHTI_DAILY_RUN_CAP", "0"))  # 0 = disabled
GEMINI_TIMEOUT_S = float(os.getenv("GEMINI_TIMEOUT_S", "120"))

ALLOWED_TOOLS = ["SOP Analysis", "Posture Analysis", "Object Detection", "Tracking Analysis"]
VALID_STATUS = ("PASS", "FAIL", "NEUTRAL")

# In-process health counters (per instance)
_stats = {"analyze_ok_last": None, "analyze_5xx": [], "last_error_class": None, "latencies_ms": []}
_stats_lock = threading.Lock()


def _record(ok: bool, error_class: Optional[str] = None, latency_ms: Optional[float] = None) -> None:
    now = time.time()
    with _stats_lock:
        if latency_ms is not None:
            _stats["latencies_ms"] = (_stats["latencies_ms"] + [float(latency_ms)])[-200:]
        if ok:
            _stats["analyze_ok_last"] = datetime.now(timezone.utc).isoformat()
        else:
            _stats["analyze_5xx"].append(now)
            _stats["last_error_class"] = error_class
        _stats["analyze_5xx"] = [t for t in _stats["analyze_5xx"] if now - t < 900]


def _percentile(values: list, pct: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round(pct / 100.0 * (len(vals) - 1)))))
    return round(vals[idx], 1)


def _client_ip(request: Request) -> str:
    """Trusted client IP. X-Forwarded-For's leftmost entries are client-controlled; the platform
    (Cloud Run) appends the real peer, so take the entry TRUSTED_PROXY_HOPS from the right."""
    parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    if parts and len(parts) >= TRUSTED_PROXY_HOPS:
        return parts[-TRUSTED_PROXY_HOPS]
    return request.client.host if request.client else "unknown"


# Per-IP sliding-window rate limiter (in-memory, per instance)
_rate: dict = {}
_rate_lock = threading.Lock()


def _rate_limited(ip: str) -> bool:
    if RATE_LIMIT_PER_MIN <= 0:
        return False
    now = time.time()
    with _rate_lock:
        hits = [t for t in _rate.get(ip, []) if now - t < 60]
        if len(hits) >= RATE_LIMIT_PER_MIN:
            _rate[ip] = hits
            return True
        hits.append(now)
        _rate[ip] = hits
        if len(_rate) > 10000:
            for k in [k for k, v in _rate.items() if not v or now - v[-1] >= 60]:
                _rate.pop(k, None)
            if len(_rate) > 10000:  # still over: drop the oldest half
                for k in sorted(_rate, key=lambda k: _rate[k][-1] if _rate[k] else 0)[: len(_rate) // 2]:
                    _rate.pop(k, None)
    return False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Drishti", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Lazy Firestore
# ---------------------------------------------------------------------------
_db = None


def get_db():
    global _db
    if _db is not None:
        return _db
    if not FIRESTORE_PROJECT:
        return None
    try:
        from google.cloud import firestore  # type: ignore

        _db = firestore.Client(project=FIRESTORE_PROJECT)
        logger.info("Firestore connected to project %s", FIRESTORE_PROJECT)
    except Exception as exc:
        logger.warning("Firestore init failed (continuing without): %s", exc)
    return _db


# ---------------------------------------------------------------------------
# Lazy Gemini client
# ---------------------------------------------------------------------------
_gemini = None


def get_gemini():
    global _gemini
    if _gemini is not None:
        return _gemini
    if not GEMINI_KEY:
        return None
    try:
        from google import genai  # type: ignore

        _gemini = genai.Client(api_key=GEMINI_KEY)
        logger.info("Gemini client initialised")
    except Exception as exc:
        logger.warning("Gemini init failed: %s", exc)
    return _gemini


# ---------------------------------------------------------------------------
# Admin auth
# ---------------------------------------------------------------------------
security = HTTPBasic()


def require_admin(credentials: HTTPBasicCredentials = Depends(security)):
    ok_user = secrets.compare_digest(
        credentials.username.encode("utf-8"), ADMIN_USER.encode("utf-8")
    )
    ok_pass = secrets.compare_digest(
        credentials.password.encode("utf-8"), ADMIN_PASS.encode("utf-8")
    )
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ---------------------------------------------------------------------------
# Gemini helpers (sync, called via asyncio.to_thread)
# ---------------------------------------------------------------------------
MODEL = "gemini-2.5-flash"

DEFAULT_QUERY = (
    "Verify aseptic technique compliance during the media fill operation. "
    "Check gowning, container handling, and environmental controls per SOP-AT-2025."
)

DEFAULT_SOPS = """SOP-AT-2025: ASEPTIC TECHNIQUE STANDARD OPERATING PROCEDURE
Version: 3.1  |  Effective: 2025-01-15  |  Review: 2026-01-15

1. GOWNING REQUIREMENTS
   1.1 Sterile gown, gloves, and face shield required for all Grade A/B entries.
   1.2 Double-gloving mandatory for Class I biological safety cabinet work.
   1.3 Gown integrity check required before each entry.

2. CONTAINER HANDLING
   2.1 No touch technique: containers held only at base or neck — never at opening.
   2.2 Immediate capping after use; max open-air exposure 30 seconds Grade B, 5 seconds Grade A.
   2.3 Discard any container dropped or potentially contaminated.

3. ENVIRONMENTAL CONTROLS
   3.1 HVAC pre-operation check: confirm Grade A ≥100 HEPA coverage, viable ≤1 CFU/m³.
   3.2 Continuous particle monitoring ≥0.5 µm; alarm at >3520 particles/m³.
   3.3 Personnel count in Grade A must not exceed 2.

4. MEDIA FILL SPECIFIC
   4.1 Incubation at 20–25 °C for 7 days minimum.
   4.2 100% unit inspection; turbidity failure → full batch investigation.
   4.3 All interventions documented within 15 minutes.
"""


_usage_ctx: contextvars.ContextVar = contextvars.ContextVar("drishti_usage", default=None)


def _add_usage(response) -> None:
    usage = _usage_ctx.get()
    meta = getattr(response, "usage_metadata", None)
    if usage is None or meta is None:
        return
    with _stats_lock:
        usage["calls"] = usage.get("calls", 0) + 1
        for key, attr in (("tokens_in", "prompt_token_count"), ("tokens_out", "candidates_token_count")):
            val = getattr(meta, attr, None)
            if isinstance(val, int):
                usage[key] = usage.get(key, 0) + val


def _call_gemini(client, prompt: str, media_bytes: Optional[bytes] = None,
                 media_mime: Optional[str] = None, json_mode: bool = False,
                 schema: Optional[dict] = None, media_uri: Optional[str] = None) -> str:
    """Sync Gemini call. Returns raw text (or JSON string if json_mode)."""
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    parts = [types.Part.from_text(text=prompt)]
    if media_uri and media_mime:
        parts.append(types.Part.from_uri(file_uri=media_uri, mime_type=media_mime))
    elif media_bytes and media_mime:
        parts.append(types.Part.from_bytes(data=media_bytes, mime_type=media_mime))

    config_kwargs: dict = {}
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"
        if schema:
            config_kwargs["response_schema"] = schema

    response = client.models.generate_content(
        model=MODEL,
        contents=types.Content(role="user", parts=parts),
        config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
    )
    _add_usage(response)
    return response.text or ""


def _parse_json_safe(raw: str, fallback: dict) -> dict:
    try:
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else fallback
    except Exception:
        return fallback


PLAN_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "tools": {"type": "ARRAY", "items": {"type": "STRING", "enum": ALLOWED_TOOLS}},
        "reasoning": {"type": "STRING"},
    },
    "required": ["tools"],
}

REPORT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "executiveSummary": {"type": "STRING"},
        "complianceChecks": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "rule": {"type": "STRING"},
                    "status": {"type": "STRING", "enum": list(VALID_STATUS)},
                    "observation": {"type": "STRING"},
                },
                "required": ["rule", "status", "observation"],
            },
        },
        "visualDetails": {"type": "ARRAY", "items": {"type": "STRING"}},
        "finalVerdict": {"type": "STRING", "enum": list(VALID_STATUS)},
        "detailedAnalysis": {"type": "STRING"},
    },
    "required": ["complianceChecks", "finalVerdict"],
}


def _norm_key(s) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


_TOOL_LOOKUP = {_norm_key(t): t for t in ALLOWED_TOOLS}


def _normalize_tools(tools) -> list:
    """Map planner output onto the allowed tool names; drop anything unknown."""
    if isinstance(tools, str):
        tools = [tools]
    if not isinstance(tools, (list, tuple)):
        return []
    out = []
    for t in tools:
        name = _TOOL_LOOKUP.get(_norm_key(t))
        if name and name not in out:
            out.append(name)
    return out


def compute_verdict(report: dict) -> tuple:
    """Return (verdict, fail_count). PASS only if no check failed, at least one check PASSed,
    and the model's finalVerdict is PASS; anything else is NEUTRAL (fail closed)."""
    checks = report.get("complianceChecks") if isinstance(report, dict) else None
    if not isinstance(checks, list):
        checks = []
    statuses = []
    for c in checks:
        if isinstance(c, dict):
            st = str(c.get("status", "")).strip().upper()
            statuses.append(st if st in VALID_STATUS else "NEUTRAL")
    fail_count = sum(1 for st in statuses if st == "FAIL")
    final = str(report.get("finalVerdict", "")).strip().upper() if isinstance(report, dict) else ""
    if fail_count > 0 or final == "FAIL":
        return "FAIL", fail_count
    if fail_count == 0 and any(st == "PASS" for st in statuses) and final == "PASS":
        return "PASS", fail_count
    return "NEUTRAL", fail_count


def validate_report(report) -> tuple:
    """Return (ok, reason). A valid report is a dict with finalVerdict in VALID_STATUS and at
    least MIN_CHECKS complianceChecks, each a dict with a valid status."""
    if not isinstance(report, dict):
        return False, "report is not an object"
    if str(report.get("finalVerdict", "")).strip().upper() not in VALID_STATUS:
        return False, "finalVerdict missing or invalid"
    checks = report.get("complianceChecks")
    if not isinstance(checks, list) or len(checks) < MIN_CHECKS:
        return False, f"fewer than {MIN_CHECKS} complianceChecks"
    for c in checks:
        if not isinstance(c, dict) or str(c.get("status", "")).strip().upper() not in VALID_STATUS:
            return False, "check with missing or invalid status"
    return True, ""


def evaluate_report(report) -> tuple:
    """Return (verdict, fail_count, report_fallback). An invalid report never yields PASS: it
    defaults to NEUTRAL with report_fallback=True (a recorded FAIL is never downgraded)."""
    verdict, fail_count = compute_verdict(report)
    ok, reason = validate_report(report)
    fallback = (not ok) or (isinstance(report, dict) and bool(report.get("report_fallback")))
    if fallback and verdict == "PASS":
        verdict = "NEUTRAL"
    if not ok:
        logger.info("Report failed validation (%s); report_fallback=true", reason)
    return verdict, fail_count, fallback


def _get_plan(client, query: str, sops: str) -> dict:
    prompt = f"""You are an inspection orchestrator. Given the query and available SOPs, decide which vision-analysis tools to run.

Available tools: ["SOP Analysis", "Posture Analysis", "Object Detection", "Tracking Analysis"]
- SOP Analysis: cross-reference footage against written SOPs (requires SOPs text)
- Posture Analysis: assess body mechanics, gowning posture, ergonomics
- Object Detection: identify equipment, containers, PPE, environmental features
- Tracking Analysis: analyze motion sequences, workflows, time-based compliance

Query: {query}
SOPs present: {"yes" if sops.strip() else "no"}

Return JSON ONLY:
{{"tools": ["<tool1>", ...], "reasoning": "<one sentence rationale>"}}"""

    raw = _call_gemini(client, prompt, json_mode=True, schema=PLAN_SCHEMA)
    fallback = {
        "tools": ["SOP Analysis", "Object Detection", "Posture Analysis"],
        "reasoning": "Default tool selection applied.",
    }
    result = _parse_json_safe(raw, fallback)
    plan_fallback = False
    if result is fallback or "tools" not in result:
        result = fallback
        plan_fallback = True
    result = dict(result)
    tools = _normalize_tools(result.get("tools"))
    if not tools:
        tools = list(fallback["tools"])
        result["reasoning"] = "Default tool selection applied (planner returned no valid tools)."
        plan_fallback = True
    result["plan_fallback"] = plan_fallback
    if sops.strip() and "SOP Analysis" not in tools:
        tools.insert(0, "SOP Analysis")
    result["tools"] = tools
    result["reasoning"] = str(result.get("reasoning", ""))
    return result


def _run_sop_analysis(client, query: str, sops: str, media_bytes: bytes, media_mime: str,
                      media_uri: Optional[str] = None) -> str:
    prompt = f"""You are an SOP compliance expert. Analyze the provided visual evidence against the following Standard Operating Procedures.

QUERY: {query}

STANDARD OPERATING PROCEDURES:
{sops}

Assess each SOP section visible in the evidence. Report specific observations for each relevant rule.
Be precise and clinical. Note any deviations, partial compliance, or confirmations."""
    return _call_gemini(client, prompt, media_bytes=media_bytes, media_mime=media_mime, media_uri=media_uri)


def _run_vision_tool(client, tool_type: str, query: str, media_bytes: bytes, media_mime: str,
                     media_uri: Optional[str] = None) -> str:
    prompt = f"""You are a computer vision analysis system performing {tool_type}.

Inspection context: {query}

Analyze the visual evidence carefully. Report specific, observable findings.
Be precise and clinical — this is a compliance instrument, not a summary."""
    return _call_gemini(client, prompt, media_bytes=media_bytes, media_mime=media_mime, media_uri=media_uri)


def _generate_report(client, query: str, sops: str, results: dict, plan: dict) -> dict:
    results_text = "\n\n".join(
        f"=== {tool} ===\n{output}" for tool, output in results.items()
    )
    prompt = f"""You are a compliance audit report writer. Synthesize the vision analysis findings into a structured audit report.

INSPECTION QUERY: {query}

STANDARD OPERATING PROCEDURES:
{sops if sops.strip() else "(none provided)"}

TOOL RESULTS:
{results_text}

Return a JSON object with EXACTLY this structure:
{{
  "title": "<concise report title>",
  "executiveSummary": "<2-3 sentence summary>",
  "complianceChecks": [
    {{"rule": "<SOP rule or observation category>", "status": "PASS|FAIL|NEUTRAL", "observation": "<specific finding>"}}
  ],
  "visualDetails": ["<specific visual observation 1>", "<specific visual observation 2>"],
  "finalVerdict": "PASS|FAIL",
  "detailedAnalysis": "<paragraph of detailed findings>"
}}

Rules:
- complianceChecks must have at least 3 items
- status is exactly "PASS", "FAIL", or "NEUTRAL"
- finalVerdict is "FAIL" if any check is FAIL, else "PASS"
- Be specific and clinical; no filler phrases"""

    raw = _call_gemini(client, prompt, json_mode=True, schema=REPORT_SCHEMA)
    fallback = {
        "title": "Compliance Audit Report",
        "executiveSummary": "Analysis complete. See detailed findings below.",
        "complianceChecks": [
            {"rule": "Visual evidence review", "status": "NEUTRAL", "observation": "Structured analysis generated."}
        ],
        "visualDetails": ["Evidence processed."],
        "finalVerdict": "NEUTRAL",
        "detailedAnalysis": "Report generated from available visual evidence.",
        "report_fallback": True,
    }
    return _parse_json_safe(raw, fallback)


def _map_upstream_error(exc: Exception) -> HTTPException:
    """Translate google.genai / upstream failures into JSON HTTP errors."""
    code = getattr(exc, "code", None)
    msg = str(exc).lower()
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return HTTPException(status_code=504, detail="Model request timed out. Please retry.")
    if code == 429 or "quota" in msg or "resource_exhausted" in msg or "rate limit" in msg:
        return HTTPException(status_code=429, detail="Model quota exceeded. Please retry later.")
    if code in (400, 413) and ("size" in msg or "too large" in msg or "unsupported" in msg or "mime" in msg):
        return HTTPException(status_code=413, detail="Media is too large or unsupported by the model.")
    return HTTPException(status_code=502, detail="Upstream model request failed.")


def _upload_media(client, media_bytes: bytes, media_mime: str):
    """Upload once via the Files API (video or large media). Returns the file object or None
    (callers fall back to inline bytes)."""
    is_video = (media_mime or "").startswith("video/")
    if not (is_video or len(media_bytes) >= FILES_API_THRESHOLD_MB * 1024 * 1024):
        return None
    files = getattr(client, "files", None)
    if files is None:
        return None
    try:
        f = files.upload(file=io.BytesIO(media_bytes), config={"mime_type": media_mime})
        deadline = time.time() + 60
        while str(getattr(getattr(f, "state", None), "name", getattr(f, "state", ""))) == "PROCESSING" \
                and time.time() < deadline:
            time.sleep(1)
            f = files.get(name=f.name)
        state = str(getattr(getattr(f, "state", None), "name", getattr(f, "state", "")))
        if state == "FAILED" or not getattr(f, "uri", None):
            return None
        return f
    except Exception as exc:
        logger.warning("Files API upload failed, using inline media: %s", exc)
        return None


def _delete_media(client, f) -> None:
    try:
        client.files.delete(name=f.name)
    except Exception as exc:
        logger.debug("Files API delete failed: %s", exc)


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


async def run_pipeline(query: str, sops: str, media_bytes: bytes, media_mime: str,
                       telemetry: Optional[dict] = None, emit=None):
    """Full agentic pipeline. Returns (plan, logs, results, report).

    telemetry (optional dict) is filled with stage, stage_latency_ms, tokens_in/out, plan_fallback,
    report_fallback. emit (optional callable) receives NDJSON-able event dicts."""
    telemetry = telemetry if telemetry is not None else {}
    telemetry.setdefault("stage_latency_ms", {})
    client = get_gemini()
    if not client:
        raise HTTPException(status_code=503, detail="Server is not configured with a model key")
    usage = {"tokens_in": 0, "tokens_out": 0, "calls": 0}
    token = _usage_ctx.set(usage)
    try:
        return await asyncio.wait_for(
            _run_pipeline_inner(client, query, sops, media_bytes, media_mime, telemetry, emit),
            GEMINI_TIMEOUT_S,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Pipeline failed: %s: %s", type(exc).__name__, exc)
        telemetry["error_class"] = type(exc).__name__
        _record(False, type(exc).__name__)
        raise _map_upstream_error(exc)
    finally:
        telemetry["tokens_in"] = usage["tokens_in"]
        telemetry["tokens_out"] = usage["tokens_out"]
        _usage_ctx.reset(token)


async def _run_pipeline_inner(client, query: str, sops: str, media_bytes: bytes, media_mime: str,
                              telemetry: Optional[dict] = None, emit=None):
    telemetry = telemetry if telemetry is not None else {}
    lat = telemetry.setdefault("stage_latency_ms", {})

    def ev(event: str, **kw):
        if emit:
            try:
                emit({"event": event, **kw})
            except Exception:  # never let a stream consumer break the pipeline
                pass

    logs = []

    telemetry["stage"] = "plan"
    logs.append("Orchestrator: Parsing query and evaluating tool requirements...")
    t0 = time.perf_counter()
    plan = await asyncio.to_thread(_get_plan, client, query, sops)
    lat["plan"] = _ms(t0)
    telemetry["plan_fallback"] = bool(plan.get("plan_fallback"))
    logs.append(f"Orchestrator: Plan — tools={plan['tools']}. {plan.get('reasoning', '')}")

    results = {}
    tools = _normalize_tools(plan.get("tools", []))
    skipped = [t for t in ALLOWED_TOOLS if t not in tools or (t == "SOP Analysis" and not sops.strip())]
    ev("plan", tools=[t for t in tools if t not in skipped], skipped=skipped,
       reasoning=str(plan.get("reasoning", "")), plan_fallback=telemetry["plan_fallback"])

    telemetry["stage"] = "tools"
    media_file = None
    uri = None
    jobs = []  # (name, callable, args)
    if tools:
        media_file = await asyncio.to_thread(_upload_media, client, media_bytes, media_mime)
        uri = getattr(media_file, "uri", None) if media_file is not None else None
    if "SOP Analysis" in tools and sops.strip():
        jobs.append(("SOP Analysis", _run_sop_analysis, (client, query, sops, media_bytes, media_mime, uri)))
    for name, desc in (
        ("Posture Analysis", "posture and body mechanics analysis"),
        ("Object Detection", "object detection and equipment identification"),
        ("Tracking Analysis", "motion tracking and sequence compliance analysis"),
    ):
        if name in tools:
            jobs.append((name, _run_vision_tool, (client, desc, query, media_bytes, media_mime, uri)))

    async def _run_job(name, fn, args):
        ev("tool_started", tool=name)
        tt = time.perf_counter()
        out = await asyncio.to_thread(fn, *args)
        ms = _ms(tt)
        lat[f"tool:{name}"] = ms
        ev("tool_done", tool=name, latency_ms=ms, output=out)
        return out

    try:
        if jobs:
            logs.append("Tools: Running " + ", ".join(n for n, _, _ in jobs) + " concurrently...")
            tt = time.perf_counter()
            outputs = await asyncio.gather(*(_run_job(n, fn, a) for n, fn, a in jobs))
            lat["tools"] = _ms(tt)
            for (name, _, _), out in zip(jobs, outputs):
                results[name] = out
                logs.append(f"Tools: {name} complete.")
    finally:
        if media_file is not None:
            await asyncio.to_thread(_delete_media, client, media_file)

    if not results:
        logs.append("Report Generator: No tool produced evidence; verdict is NEUTRAL (inconclusive).")
        report = {
            "title": "Inconclusive Analysis",
            "executiveSummary": "No analysis tool produced evidence, so no compliance determination was made.",
            "complianceChecks": [
                {"rule": "Evidence analysis", "status": "NEUTRAL", "observation": "No tool output available."}
            ],
            "visualDetails": [],
            "finalVerdict": "NEUTRAL",
            "detailedAnalysis": "The planner selected no runnable tools. Re-run the analysis.",
            "report_fallback": True,
        }
        telemetry["report_fallback"] = True
        ev("report", report=report)
        return plan, logs, results, report

    telemetry["stage"] = "report"
    logs.append("Report Generator: Compiling structured compliance audit...")
    t0 = time.perf_counter()
    report = await asyncio.to_thread(_generate_report, client, query, sops, results, plan)
    lat["report"] = _ms(t0)
    verdict, _, rep_fb = evaluate_report(report)
    telemetry["report_fallback"] = rep_fb
    logs.append(f"Analysis complete - Verdict: {verdict}")
    ev("report", report=report)

    return plan, logs, results, report


# ---------------------------------------------------------------------------
# HTML Templates
# ---------------------------------------------------------------------------
FONTS = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,400&family=Hanken+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">"""

BASE_STYLES = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --obsidian:#0B0F14;--obsidian-2:#11161C;--obsidian-3:#171F27;--obsidian-4:#1E2832;
  --warm:#F2A85B;--cool:#46C8DD;--warm-deep:#C97E36;--cool-deep:#2C8E9F;
  --seam:linear-gradient(90deg,#F2A85B 0%,#C9B07A 48%,#46C8DD 100%);
  --white:#EAEDEF;--gray-200:#AEB7C0;--gray-400:#7A8692;--gray-600:#4B5560;--gray-800:#2A323B;
  --green:#74D99A;--red:#E8705D;--amber-status:#F0C04A;
  --line:rgba(255,255,255,.08);--line-2:rgba(255,255,255,.15);
  --line-warm:rgba(242,168,91,.32);--line-cool:rgba(70,200,221,.32);
  --font-serif:'Fraunces',Georgia,serif;
  --font-sans:'Hanken Grotesk',ui-sans-serif,sans-serif;
  --font-mono:'JetBrains Mono',ui-monospace,monospace;
  --tracking-label:0.14em;
  --shadow:0 4px 24px rgba(0,0,0,.45);
}
html{background:var(--obsidian);color:var(--white);font-family:var(--font-sans);font-size:17px;line-height:1.5}
body{min-height:100vh;overflow-x:hidden}
a{color:var(--cool);text-decoration:none}
a:hover{text-decoration:underline}
button{cursor:pointer;font-family:var(--font-sans)}
textarea,input[type=file]{font-family:var(--font-sans)}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""

MAIN_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drishti — Agentic Vision Orchestrator</title>
{fonts}
<style>
{base}
/* ---- Layout ---- */
.page-wrap{{max-width:1280px;margin:0 auto;padding:0 24px}}

/* ---- Nav ---- */
nav{{
  border-bottom:1px solid var(--line);
  padding:18px 0;
  position:sticky;top:0;
  background:rgba(11,15,20,.92);
  backdrop-filter:blur(12px);
  z-index:50;
}}
.nav-inner{{
  display:flex;align-items:center;justify-content:space-between;
  max-width:1280px;margin:0 auto;padding:0 24px;
}}
.wordmark{{
  font-family:var(--font-serif);font-size:22px;font-weight:700;letter-spacing:-.02em;
  background:var(--seam);-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}}
.nav-eyebrow{{
  font-family:var(--font-mono);font-size:10px;font-weight:500;
  letter-spacing:var(--tracking-label);text-transform:uppercase;color:var(--gray-400);
}}

/* ---- Hero ---- */
.hero{{
  padding:72px 0 56px;
  position:relative;
  overflow:hidden;
}}
.hero::before{{
  content:'';position:absolute;inset:0;pointer-events:none;
  background:
    radial-gradient(ellipse 60% 40% at 20% 50%,rgba(242,168,91,.07) 0%,transparent 70%),
    radial-gradient(ellipse 50% 35% at 80% 50%,rgba(70,200,221,.08) 0%,transparent 70%);
}}
.hero-eyebrow{{
  font-family:var(--font-mono);font-size:11px;font-weight:500;
  letter-spacing:var(--tracking-label);text-transform:uppercase;color:var(--gray-400);
  margin-bottom:20px;
}}
.hero h1{{
  font-family:var(--font-serif);font-size:clamp(34px,5vw,52px);
  font-weight:700;line-height:1.1;letter-spacing:-.02em;
  max-width:700px;margin-bottom:20px;
}}
.hero-sub{{
  font-size:18px;line-height:1.65;color:var(--gray-200);max-width:600px;margin-bottom:14px;
}}
.hero-what{{
  font-family:var(--font-mono);font-size:13px;color:var(--gray-400);
}}
.banner{{
  background:rgba(232,112,93,.12);border:1px solid rgba(232,112,93,.3);
  border-radius:8px;padding:12px 16px;
  font-family:var(--font-mono);font-size:12px;color:#E8705D;
  letter-spacing:.04em;margin-top:20px;
}}

/* ---- Main grid ---- */
.app-grid{{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:24px;
  padding-bottom:32px;
}}
@media(max-width:860px){{.app-grid{{grid-template-columns:1fr}}}}

/* ---- Input panel ---- */
.panel{{
  background:var(--obsidian-2);border:1px solid var(--line);border-radius:12px;
  padding:28px;
}}
.panel-label{{
  font-family:var(--font-mono);font-size:10px;font-weight:500;
  letter-spacing:var(--tracking-label);text-transform:uppercase;
  color:var(--gray-400);margin-bottom:20px;
}}
.field{{margin-bottom:20px}}
.field label{{
  display:block;font-size:13px;font-weight:500;color:var(--gray-200);margin-bottom:6px;
}}
.field textarea,.field input[type=file]{{
  width:100%;background:var(--obsidian-4);border:1px solid var(--line-2);
  border-radius:8px;color:var(--white);font-size:14px;
  transition:border-color .15s;
}}
.field textarea{{padding:12px 14px;resize:vertical;min-height:80px;}}
.field textarea:focus{{outline:none;border-color:rgba(70,200,221,.55);}}
.field input[type=file]{{
  padding:10px 14px;cursor:pointer;
  color:var(--gray-200);
}}
.field input[type=file]::file-selector-button{{
  background:var(--obsidian-3);border:1px solid var(--line-2);
  color:var(--white);border-radius:6px;padding:4px 12px;
  font-family:var(--font-sans);font-size:13px;cursor:pointer;margin-right:10px;
}}

.cta{{
  display:inline-block;width:100%;
  padding:14px 28px;border:none;border-radius:8px;
  font-family:var(--font-sans);font-size:15px;font-weight:600;
  background:var(--seam);color:#0B0F14;
  cursor:pointer;transition:opacity .15s,transform .1s;
  margin-top:4px;
}}
.cta:hover{{opacity:.9}}
.cta:active{{transform:scale(.98)}}
.cta:disabled{{opacity:.45;cursor:not-allowed;transform:none}}

/* ---- Right panel: schematic + log ---- */
.right-col{{display:flex;flex-direction:column;gap:16px}}

/* ---- Schematic ---- */
.schematic{{
  background:var(--obsidian-2);border:1px solid var(--line);border-radius:0;
  padding:24px;
}}
.schematic-label{{
  font-family:var(--font-mono);font-size:10px;font-weight:500;
  letter-spacing:var(--tracking-label);text-transform:uppercase;
  color:var(--gray-400);margin-bottom:16px;
}}
.s-nodes{{display:flex;flex-direction:column;gap:0}}
.s-node{{
  display:flex;align-items:center;gap:12px;padding:10px 14px;
  border:1px solid var(--line);background:var(--obsidian-3);
  transition:border-color .2s,background .2s,box-shadow .2s;
}}
.s-node+.s-node{{margin-top:-1px}}
.s-node.active{{
  border-color:var(--warm);background:rgba(242,168,91,.08);
  box-shadow:0 0 12px rgba(242,168,91,.18);
  animation:pulse-warm 1.2s ease-in-out infinite;
}}
.s-node.done{{border-color:var(--line-cool);background:rgba(70,200,221,.06)}}
.s-node.skipped .s-name,.s-node.skipped .s-status{{color:var(--gray-200)}}
.evidence{{margin-top:20px;border:1px solid var(--line-2);border-radius:8px;padding:10px 14px}}
.evidence>summary,.evidence-tool>summary{{cursor:pointer;font-family:var(--font-mono);font-size:12px;color:var(--gray-200)}}
.evidence-tool{{margin-top:8px}}
.evidence-pre{{white-space:pre-wrap;word-break:break-word;font-family:var(--font-mono);font-size:12px;color:var(--gray-200);margin-top:6px}}
.evidence-skipped{{font-family:var(--font-mono);font-size:12px;color:var(--gray-400);margin-top:8px}}
.s-dot{{width:7px;height:7px;border-radius:50%;background:var(--gray-600);flex-shrink:0;transition:background .2s}}
.s-node.active .s-dot{{background:var(--warm)}}
.s-node.done .s-dot{{background:var(--cool)}}
.s-name{{font-family:var(--font-mono);font-size:12px;font-weight:500;letter-spacing:.04em;color:var(--gray-200)}}
.s-node.active .s-name{{color:var(--warm)}}
.s-node.done .s-name{{color:var(--cool)}}
.s-status{{
  margin-left:auto;font-family:var(--font-mono);font-size:10px;
  letter-spacing:var(--tracking-label);text-transform:uppercase;color:var(--gray-200);
}}
.s-node.active .s-status{{color:var(--warm)}}
.s-node.done .s-status{{color:var(--cool)}}
@keyframes pulse-warm{{0%,100%{{box-shadow:0 0 8px rgba(242,168,91,.18)}}50%{{box-shadow:0 0 20px rgba(242,168,91,.35)}}}}

/* ---- Log feed ---- */
.log-panel{{
  background:var(--obsidian-2);border:1px solid var(--line);border-radius:0;
  padding:20px;flex:1;
}}
.log-label{{
  font-family:var(--font-mono);font-size:10px;font-weight:500;
  letter-spacing:var(--tracking-label);text-transform:uppercase;
  color:var(--gray-400);margin-bottom:12px;
}}
.log-feed{{
  font-family:var(--font-mono);font-size:12px;line-height:1.7;
  color:var(--gray-400);max-height:200px;overflow-y:auto;
}}
.log-feed .log-line{{color:var(--gray-200);padding:1px 0}}
.log-feed .log-line.done{{color:var(--cool)}}

/* ---- Report panel ---- */
.report-section{{margin-bottom:32px}}
.report-panel{{
  background:var(--obsidian-2);border:1px solid var(--line);border-radius:12px;
  padding:32px;margin-bottom:40px;
}}
.report-empty{{
  font-family:var(--font-mono);font-size:13px;color:var(--gray-400);text-align:center;padding:40px 0;
}}
.report-header{{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:24px;flex-wrap:wrap;gap:12px}}
.report-title{{font-family:var(--font-serif);font-size:24px;font-weight:600;line-height:1.2}}
.verdict-chip{{
  font-family:var(--font-mono);font-size:11px;font-weight:500;
  letter-spacing:var(--tracking-label);text-transform:uppercase;
  padding:4px 12px;border-radius:4px;
  flex-shrink:0;
}}
.verdict-PASS{{background:rgba(116,217,154,.15);color:#74D99A;border:1px solid rgba(116,217,154,.3)}}
.verdict-FAIL{{background:rgba(232,112,93,.15);color:#E8705D;border:1px solid rgba(232,112,93,.3)}}
.verdict-NEUTRAL{{background:rgba(70,200,221,.12);color:#46C8DD;border:1px solid rgba(70,200,221,.25)}}

.report-summary{{font-size:15px;line-height:1.7;color:var(--gray-200);margin-bottom:24px}}

.checks-label{{
  font-family:var(--font-mono);font-size:10px;font-weight:500;
  letter-spacing:var(--tracking-label);text-transform:uppercase;color:var(--gray-400);
  margin-bottom:12px;
}}
.check-item{{
  display:flex;align-items:flex-start;gap:12px;padding:12px 0;
  border-bottom:1px solid var(--line);
}}
.check-item:last-child{{border-bottom:none}}
.status-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0;margin-top:5px}}
.status-PASS{{background:#74D99A}}
.status-FAIL{{background:#E8705D}}
.status-NEUTRAL{{background:#7A8692}}
.check-rule{{font-size:14px;font-weight:500;color:var(--white);margin-bottom:3px}}
.check-obs{{font-size:13px;color:var(--gray-200);line-height:1.5}}
.check-status-word{{
  margin-left:auto;font-family:var(--font-mono);font-size:10px;font-weight:500;
  letter-spacing:var(--tracking-label);text-transform:uppercase;flex-shrink:0;
}}
.status-word-PASS{{color:#74D99A}}
.status-word-FAIL{{color:#E8705D}}
.status-word-NEUTRAL{{color:#7A8692}}

.detail-label{{
  font-family:var(--font-mono);font-size:10px;font-weight:500;
  letter-spacing:var(--tracking-label);text-transform:uppercase;color:var(--gray-400);
  margin:20px 0 10px;
}}
.detail-text{{font-size:14px;line-height:1.75;color:var(--gray-200)}}

.error-panel{{
  background:rgba(232,112,93,.1);border:1px solid rgba(232,112,93,.3);
  border-radius:12px;padding:24px;
  font-family:var(--font-mono);font-size:13px;color:#E8705D;
}}

/* ---- Footer ---- */
footer{{
  border-top:1px solid var(--line);padding:24px 0;
  font-family:var(--font-mono);font-size:10px;
  letter-spacing:var(--tracking-label);text-transform:uppercase;
  color:var(--gray-400);text-align:center;
}}
</style>
</head>
<body>

<nav>
  <div class="nav-inner">
    <span class="wordmark">Drishti</span>
    <span class="nav-eyebrow">Agentic Vision Orchestrator</span>
  </div>
</nav>

<div class="page-wrap">
  <section class="hero">
    <p class="hero-eyebrow">Agentic Vision Orchestrator</p>
    <h1>See what your SOPs require.<br>Prove what your footage shows.</h1>
    <p class="hero-sub">Drishti plans the inspection, runs the right vision tools over your image or video, cross-references your Standard Operating Procedures, and returns a structured pass/fail audit — autonomously.</p>
    <p class="hero-what">Upload a frame or clip, paste your SOPs, ask a question — the orchestrator decides which tools to run, runs them, and writes the report.</p>
    {banner}
  </section>

  <div class="app-grid">
    <!-- Input panel -->
    <div class="panel">
      <p class="panel-label">Inspection input</p>
      <form id="analyzeForm" enctype="multipart/form-data" aria-busy="false">
        <div class="field">
          <label for="query">Inspection query</label>
          <textarea id="query" name="query" rows="3" required minlength="3" maxlength="2000" placeholder="What should the orchestrator verify?">{default_query}</textarea>
        </div>
        <div class="field">
          <label for="sops">Standard Operating Procedures</label>
          <textarea id="sops" name="sops" rows="6" placeholder="Paste your SOP text here...">{default_sops}</textarea>
        </div>
        <div class="field">
          <label for="media">Image or video evidence</label>
          <input type="file" id="media" name="media" accept="image/*,video/*" required>
        </div>
        <button type="submit" class="cta" id="runBtn">Run Analysis</button>
      </form>
    </div>

    <!-- Schematic + log -->
    <div class="right-col">
      <div class="schematic">
        <p class="schematic-label">Orchestrator workflow</p>
        <div class="s-nodes">
          <div class="s-node" id="sn-input">
            <span class="s-dot"></span>
            <span class="s-name">INPUT PROCESSING</span>
            <span class="s-status" id="ss-input">IDLE</span>
          </div>
          <div class="s-node" id="sn-plan">
            <span class="s-dot"></span>
            <span class="s-name">ORCHESTRATOR PLANNING</span>
            <span class="s-status" id="ss-plan">IDLE</span>
          </div>
          <div class="s-node" id="sn-rag">
            <span class="s-dot"></span>
            <span class="s-name">SOP RAG ANALYSIS</span>
            <span class="s-status" id="ss-rag">IDLE</span>
          </div>
          <div class="s-node" id="sn-vision">
            <span class="s-dot"></span>
            <span class="s-name">VISION TOOLS</span>
            <span class="s-status" id="ss-vision">IDLE</span>
          </div>
          <div class="s-node" id="sn-report">
            <span class="s-dot"></span>
            <span class="s-name">REPORT GENERATION</span>
            <span class="s-status" id="ss-report">IDLE</span>
          </div>
        </div>
      </div>

      <div class="log-panel">
        <p class="log-label">Agent log</p>
        <div class="log-feed" id="logFeed" role="status" aria-live="polite">
          <div class="log-line" style="color:var(--gray-400)">Waiting for analysis run...</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Report panel -->
  <div class="report-section">
    <div class="report-panel" id="reportPanel" role="status" aria-live="polite">
      <p class="report-empty">No analysis yet. Provide evidence and run the orchestrator to generate a compliance report.</p>
    </div>
  </div>
</div>

<footer>
  <div class="page-wrap">Drishti &middot; Agentic Vision &middot; Auracle Factory</div>
</footer>

<script>
const NODE_SEQUENCE = [
  ['sn-input','ss-input'],
  ['sn-plan','ss-plan'],
  ['sn-rag','ss-rag'],
  ['sn-vision','ss-vision'],
  ['sn-report','ss-report'],
];

function setNodeState(idx, state) {{
  const [nodeId, statusId] = NODE_SEQUENCE[idx];
  const node = document.getElementById(nodeId);
  const stat = document.getElementById(statusId);
  node.className = 's-node ' + state;
  stat.textContent = state === 'active' ? 'RUNNING' : state === 'done' ? 'DONE' : state === 'skipped' ? 'SKIPPED' : 'IDLE';
}}

function resetNodes() {{
  NODE_SEQUENCE.forEach((_, i) => setNodeState(i, ''));
}}

function appendLog(msg, done) {{
  const feed = document.getElementById('logFeed');
  const line = document.createElement('div');
  line.className = 'log-line' + (done ? ' done' : '');
  line.textContent = msg;
  feed.appendChild(line);
  feed.scrollTop = feed.scrollHeight;
}}

function clearLog() {{
  const feed = document.getElementById('logFeed');
  feed.innerHTML = '';
}}

function esc(s) {{
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}}

function safeVerdict(v) {{
  const u = String(v ?? '').trim().toUpperCase();
  return ['PASS','FAIL','NEUTRAL'].includes(u) ? u : 'NEUTRAL';
}}

function renderReport(data) {{
  const r = data.report;
  if (!r) {{ return '<p class="report-empty">Report data unavailable.</p>'; }}

  const verdict = safeVerdict(data.verdict || r.finalVerdict);
  const overridden = data.verdict && safeVerdict(r.finalVerdict) !== verdict;
  const overrideNote = overridden
    ? '<p class="report-summary" style="font-size:12px">Model verdict overridden: ' + esc(safeVerdict(r.finalVerdict)) + ' &rarr; ' + esc(verdict) + ' based on the individual checks.</p>'
    : '';
  const checks = (r.complianceChecks || []).map(c => {{
    const s = safeVerdict(c.status);
    return `<div class="check-item">
      <span class="status-dot status-${{s}}"></span>
      <div>
        <div class="check-rule">${{esc(c.rule)}}</div>
        <div class="check-obs">${{esc(c.observation)}}</div>
      </div>
      <span class="check-status-word status-word-${{s}}">${{s}}</span>
    </div>`;
  }}).join('');

  const TOOLS = ['SOP Analysis','Posture Analysis','Object Detection','Tracking Analysis'];
  const results = data.results && typeof data.results === 'object' ? data.results : {{}};
  const evidenceItems = TOOLS.map(t => Object.prototype.hasOwnProperty.call(results, t)
    ? `<details class="evidence-tool"><summary>${{esc(t)}}</summary><pre class="evidence-pre">${{esc(results[t])}}</pre></details>`
    : `<p class="evidence-skipped">${{esc(t)}} &mdash; SKIPPED (not run)</p>`).join('');
  const evidence = `<details class="evidence"><summary>Evidence (per-tool outputs)</summary>${{evidenceItems}}</details>`;

  const visuals = (r.visualDetails || []).map(v => `<li style="margin-bottom:4px">${{esc(v)}}</li>`).join('');

  return `
    <div class="report-header">
      <h2 class="report-title">${{esc(r.title || 'Compliance Audit Report')}}</h2>
      <span class="verdict-chip verdict-${{verdict}}">${{verdict}}</span>
    </div>
    ${{overrideNote}}
    <p class="report-summary">${{esc(r.executiveSummary)}}</p>
    <p class="checks-label">Compliance checks</p>
    ${{checks}}
    <p class="detail-label">Visual findings</p>
    <ul style="padding-left:18px;color:var(--gray-200);font-size:14px;line-height:1.75">${{visuals}}</ul>
    <p class="detail-label">Detailed analysis</p>
    <p class="detail-text">${{esc(r.detailedAnalysis)}}</p>
    ${{evidence}}
    <p style="margin-top:20px;font-family:var(--font-mono);font-size:10px;color:var(--gray-400);letter-spacing:.14em">
      RUN ID: ${{esc(data.id)}} &nbsp;·&nbsp; ${{esc(data.created_at || '')}}
    </p>`;
}}

const VISION = ['Posture Analysis','Object Detection','Tracking Analysis'];

function applyEvent(ev, state) {{
  if (ev.event === 'plan') {{
    setNodeState(0, 'done');
    setNodeState(1, 'done');
    state.expected = new Set(ev.tools || []);
    state.pending = new Set(ev.tools || []);
    appendLog('Orchestrator: plan - ' + ((ev.tools || []).join(', ') || 'no tools') + (ev.reasoning ? '. ' + ev.reasoning : ''));
    const hasRag = state.expected.has('SOP Analysis');
    const hasVision = VISION.some(t => state.expected.has(t));
    setNodeState(2, hasRag ? 'active' : 'skipped');
    setNodeState(3, hasVision ? 'active' : 'skipped');
    if (!hasRag && !hasVision) setNodeState(4, 'active');
  }} else if (ev.event === 'tool_started') {{
    appendLog('Tool started: ' + ev.tool);
  }} else if (ev.event === 'tool_done') {{
    state.pending.delete(ev.tool);
    appendLog('Tool done: ' + ev.tool + ' (' + ev.latency_ms + ' ms)');
    if (ev.tool === 'SOP Analysis') setNodeState(2, 'done');
    if (!VISION.some(t => state.pending.has(t)) && VISION.some(t => state.expected.has(t))) setNodeState(3, 'done');
    if (state.pending.size === 0) setNodeState(4, 'active');
  }} else if (ev.event === 'report') {{
    setNodeState(4, 'done');
    appendLog('Report generated.');
  }}
}}

async function readStream(resp, state) {{
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  let final = null;
  for (;;) {{
    const {{value, done}} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {{stream: true}});
    let nl;
    while ((nl = buf.indexOf('\\n')) >= 0) {{
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      let ev;
      try {{ ev = JSON.parse(line); }} catch (_) {{ continue; }}
      if (ev.event === 'result' || ev.event === 'error') final = ev; else applyEvent(ev, state);
    }}
  }}
  return final;
}}

document.getElementById('analyzeForm').addEventListener('submit', async (e) => {{
  e.preventDefault();
  const form = e.target;
  const btn = document.getElementById('runBtn');
  const reportPanel = document.getElementById('reportPanel');

  const MAX_MB = {max_upload_mb};
  const f = form.querySelector('#media').files[0];
  if (f && f.size > MAX_MB * 1024 * 1024) {{
    reportPanel.innerHTML = `<div class="error-panel" role="alert">File too large (max ${{MAX_MB}} MB).</div>`;
    return;
  }}

  btn.disabled = true;
  form.setAttribute('aria-busy', 'true');
  btn.textContent = 'Running...';
  clearLog();
  resetNodes();
  reportPanel.innerHTML = '<p class="report-empty">Orchestrator running — please wait...</p>';
  setNodeState(0, 'active');
  appendLog('Submitting evidence to orchestrator...');

  const state = {{expected: new Set(), pending: new Set()}};
  const fail = (detail) => {{
    resetNodes();
    appendLog('Error: ' + detail, false);
    reportPanel.innerHTML = `<div class="error-panel" role="alert">Analysis failed: ${{esc(detail)}}</div>`;
  }};

  try {{
    const fd = new FormData(form);
    const resp = await fetch('/api/analyze/stream', {{method:'POST', body: fd}});
    const ctype = resp.headers.get('content-type') || '';

    if (!resp.ok || !ctype.includes('ndjson')) {{
      let data = null;
      if (ctype.includes('application/json')) {{
        data = await resp.json();
      }} else {{
        const txt = await resp.text();
        data = {{detail: 'HTTP ' + resp.status + (txt ? ': ' + txt.slice(0, 200) : '')}};
      }}
      let detail = data.detail;
      if (Array.isArray(detail)) {{
        detail = detail.map(d => (d && d.msg) ? d.msg : String(d)).join('; ');
      }}
      fail(detail || resp.statusText || ('HTTP ' + resp.status));
      return;
    }}

    const final = await readStream(resp, state);
    if (!final) {{ fail('The analysis stream ended unexpectedly.'); return; }}
    if (final.event === 'error') {{ fail(final.detail || ('HTTP ' + final.status)); return; }}

    const data = final.data;
    NODE_SEQUENCE.forEach((_, i) => {{ if (!document.getElementById(NODE_SEQUENCE[i][0]).classList.contains('skipped')) setNodeState(i, 'done'); }});
    clearLog();
    (data.logs || []).forEach((l, i) => appendLog(l, i === (data.logs.length - 1)));
    reportPanel.innerHTML = renderReport(data);

  }} catch (err) {{
    resetNodes();
    appendLog('Network error: ' + err.message);
    reportPanel.innerHTML = `<div class="error-panel" role="alert">Network error: ${{esc(err.message)}}</div>`;
  }} finally {{
    btn.disabled = false;
    form.setAttribute('aria-busy', 'false');
    btn.textContent = 'Run Analysis';
  }}
}});
</script>
</body>
</html>"""

ADMIN_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drishti Admin</title>
{fonts}
<style>
{base}
.wrap{{max-width:1100px;margin:0 auto;padding:32px 24px}}
.admin-header{{display:flex;align-items:baseline;gap:16px;margin-bottom:32px;border-bottom:1px solid var(--line);padding-bottom:20px}}
.wordmark{{font-family:var(--font-serif);font-size:20px;font-weight:700;background:var(--seam);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.admin-tag{{font-family:var(--font-mono);font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--gray-400)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{font-family:var(--font-mono);font-size:10px;font-weight:500;letter-spacing:.14em;text-transform:uppercase;color:var(--gray-400);text-align:left;padding:8px 12px;border-bottom:1px solid var(--line)}}
td{{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top;color:var(--gray-200)}}
tr:hover td{{background:var(--obsidian-3)}}
.chip{{display:inline-block;font-family:var(--font-mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;padding:2px 8px;border-radius:3px}}
.chip-PASS{{background:rgba(116,217,154,.15);color:#74D99A}}
.chip-FAIL{{background:rgba(232,112,93,.15);color:#E8705D}}
.chip-NEUTRAL{{background:rgba(70,200,221,.1);color:#46C8DD}}
.empty{{font-family:var(--font-mono);font-size:13px;color:var(--gray-400);padding:40px 0;text-align:center}}
footer{{border-top:1px solid var(--line);padding:20px 0;font-family:var(--font-mono);font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--gray-400);text-align:center;margin-top:40px}}
</style>
</head>
<body>
<div class="wrap">
  <div class="admin-header">
    <span class="wordmark">Drishti</span>
    <span class="admin-tag">Admin &mdash; Recent Analyses</span>
  </div>
  {content}
</div>
<footer>Drishti &middot; Agentic Vision &middot; Auracle Factory</footer>
</body>
</html>"""


def render_main_page(no_key: bool = False) -> str:
    banner = ""
    if no_key:
        banner = '<div class="banner">GEMINI_API_KEY not configured — /api/analyze will return 503 until a key is supplied.</div>'
    return MAIN_TEMPLATE.format(
        fonts=FONTS,
        base=BASE_STYLES,
        banner=banner,
        max_upload_mb=f"{MAX_UPLOAD_MB:g}",
        default_query=DEFAULT_QUERY.replace("<", "&lt;").replace(">", "&gt;"),
        default_sops=DEFAULT_SOPS.replace("<", "&lt;").replace(">", "&gt;"),
    )


def render_admin_page(rows: list) -> str:
    if not rows:
        content = '<p class="empty">No analyses stored yet.</p>'
    else:
        cells = []
        for r in rows:
            raw_verdict = str(r.get("verdict", "NEUTRAL"))
            verdict = raw_verdict if raw_verdict in ("PASS", "FAIL", "NEUTRAL") else "NEUTRAL"
            raw_query = str(r.get("query", ""))
            query_display = html.escape(raw_query[:60]) + ("…" if len(raw_query) > 60 else "")
            run_id = html.escape(str(r.get("id", "")))
            cells.append(
                f"<tr>"
                f"<td><code style='font-size:11px'>{run_id[:8]}…</code></td>"
                f"<td style='color:var(--gray-400);font-size:12px'>{html.escape(str(r.get('created_at',''))[:19].replace('T',' '))}</td>"
                f"<td>{query_display}</td>"
                f"<td><span style='font-family:var(--font-mono);font-size:11px'>{html.escape(str(r.get('media_type','')))}</span></td>"
                f"<td><span class='chip chip-{verdict}'>{verdict}</span></td>"
                f"<td style='text-align:right'>{int(r.get('fail_count', 0))}</td>"
                f"<td><a href='/api/analysis/{run_id}' style='font-size:12px'>JSON</a></td>"
                f"</tr>"
            )
        content = (
            "<table>"
            "<thead><tr>"
            "<th>ID</th><th>Created</th><th>Query</th>"
            "<th>Media</th><th>Verdict</th><th>Fails</th><th></th>"
            "</tr></thead>"
            "<tbody>" + "".join(cells) + "</tbody>"
            "</table>"
        )
    return ADMIN_TEMPLATE.format(fonts=FONTS, base=BASE_STYLES, content=content)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.middleware("http")
async def _reject_oversize_early(request: Request, call_next):
    """Reject on a declared Content-Length before the multipart body is read."""
    if request.method == "POST" and request.url.path.startswith("/api/analyze"):
        cl = request.headers.get("content-length")
        # Allow headroom for the form fields (query/sops) around the media part.
        limit = int(MAX_UPLOAD_MB * 1024 * 1024) + 512 * 1024
        try:
            too_big = cl is not None and int(cl) > limit
        except ValueError:
            too_big = False
        if too_big:
            return JSONResponse(
                {"detail": f"File too large (max {MAX_UPLOAD_MB:g} MB).", "retryable": False},
                status_code=413,
            )
    return await call_next(request)


@app.get("/health")
@app.get("/api/v1/health")
async def health():
    db = get_db()
    firestore_ok = False
    if db:
        try:
            # lightweight ping
            await asyncio.to_thread(lambda: list(db.collection("analyses").limit(1).stream()))
            firestore_ok = True
        except Exception:
            pass
    with _stats_lock:
        lats = list(_stats["latencies_ms"])
        stats = {
            "analyze_ok_last": _stats["analyze_ok_last"],
            "analyze_5xx_15m": len([t for t in _stats["analyze_5xx"] if time.time() - t < 900]),
            "last_error_class": _stats["last_error_class"],
            "analyze_p50_ms": _percentile(lats, 50),
            "analyze_p95_ms": _percentile(lats, 95),
        }
    # Degraded when a configured store is unreachable or the model key is missing.
    degraded = (bool(FIRESTORE_PROJECT) and not firestore_ok) or not GEMINI_KEY
    body = {
        "status": "degraded" if degraded else "ok",
        "version": GIT_SHA,
        "gemini_key_present": bool(GEMINI_KEY),
        "firestore_ok": firestore_ok,
        **stats,
    }
    return JSONResponse(body, status_code=503 if degraded else 200)


@app.get("/health/model")
async def health_model():
    """Model-reachability probe (models.get). Separate from /health so liveness probes stay free."""
    client = get_gemini()
    if not client:
        return JSONResponse({"status": "degraded", "model": MODEL, "model_ok": False,
                             "error_class": "NoKey"}, status_code=503)
    try:
        await asyncio.wait_for(asyncio.to_thread(client.models.get, model=MODEL), 10)
    except Exception as exc:
        return JSONResponse({"status": "degraded", "model": MODEL, "model_ok": False,
                             "error_class": type(exc).__name__}, status_code=503)
    return JSONResponse({"status": "ok", "model": MODEL, "model_ok": True})


@app.get("/", response_class=HTMLResponse)
async def index():
    return render_main_page(no_key=not bool(GEMINI_KEY))


def _daily_cap_state(db) -> str:
    """Firestore-backed daily run counter. Returns 'ok', 'exceeded' or 'error'.

    Fails CLOSED: when DRISHTI_DAILY_RUN_CAP > 0 and the store is missing or the counter cannot be
    incremented/read, the request is refused ('error'). The increment is atomic (Increment(1)) and
    the returned count is checked, so concurrent requests cannot overshoot the cap."""
    if DAILY_RUN_CAP <= 0:
        return "ok"
    if not db:
        logger.warning("Daily cap enabled but Firestore is not configured; failing closed")
        return "error"
    try:
        from google.cloud import firestore  # type: ignore

        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        ref = db.collection("usage").document(day)
        ref.set({"count": firestore.Increment(1), "day": day}, merge=True)
        snap = ref.get()
        count = (snap.to_dict() or {}).get("count", 0) if snap.exists else 0
        return "exceeded" if count > DAILY_RUN_CAP else "ok"
    except Exception as exc:
        logger.warning("Daily cap check failed (failing closed): %s", exc)
        return "error"


def _daily_cap_exceeded(db) -> bool:
    return _daily_cap_state(db) != "ok"


async def _prepare(request: Request, query: str, sops: str, media: UploadFile) -> dict:
    if not GEMINI_KEY:
        raise HTTPException(status_code=503, detail="Server is not configured with a model key")

    ip = _client_ip(request)
    if _rate_limited(ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute.")
    if not query.strip():
        raise HTTPException(status_code=422, detail="Query must not be blank.")

    max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
    media_bytes = await media.read(max_bytes + 1)
    if len(media_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_MB:g} MB).")

    db = get_db()
    cap = await asyncio.to_thread(_daily_cap_state, db)
    if cap == "exceeded":
        raise HTTPException(status_code=429, detail="Daily analysis limit reached.")
    if cap == "error":
        raise HTTPException(status_code=503, detail="Usage limiter unavailable; try again later.")

    return {
        "query": query, "sops": sops, "media_bytes": media_bytes,
        "media_type": media.content_type or "image/jpeg",
        "media_filename": media.filename or "upload", "db": db,
    }


def _log_telemetry(run_id: str, http_status: int, ctx: dict, telemetry: dict, verdict: Optional[str],
                   total_ms: int) -> dict:
    lat = dict(telemetry.get("stage_latency_ms", {}))
    lat["total"] = total_ms
    fields = {
        "run_id": run_id,
        "http_status": http_status,
        "stage_latency_ms": lat,
        "tokens_in": telemetry.get("tokens_in", 0),
        "tokens_out": telemetry.get("tokens_out", 0),
        "plan_fallback": bool(telemetry.get("plan_fallback")),
        "report_fallback": bool(telemetry.get("report_fallback")),
        "verdict": verdict,
        "media_type": ctx["media_type"],
        "media_bytes": len(ctx["media_bytes"]),
    }
    if telemetry.get("error_class"):
        fields["stage"] = telemetry.get("stage")
        fields["error_class"] = telemetry["error_class"]
    logger.info(json.dumps({"event": "analysis", **fields}, default=str))
    return fields


async def _execute(ctx: dict, emit=None) -> dict:
    t_start = time.perf_counter()
    run_id = uuid.uuid4().hex
    db = ctx["db"]
    telemetry: dict = {"stage": "plan"}
    try:
        plan, logs, results, report = await run_pipeline(
            ctx["query"], ctx["sops"], ctx["media_bytes"], ctx["media_type"],
            telemetry=telemetry, emit=emit,
        )
    except HTTPException as exc:
        total_ms = _ms(t_start)
        telemetry.setdefault("error_class", "HTTPException")
        _log_telemetry(run_id, exc.status_code, ctx, telemetry, None, total_ms)
        if db:
            now = datetime.now(timezone.utc)
            err_doc = {
                "id": run_id, "status": "error", "stage": telemetry.get("stage"),
                "error_class": telemetry["error_class"], "http_status": exc.status_code,
                "created_at": now,
                "expire_at": now + timedelta(days=int(os.environ.get("DRISHTI_ANALYSES_TTL_DAYS", 90))),
                "media_type": ctx["media_type"], "media_bytes": len(ctx["media_bytes"]),
                "verdict": "NEUTRAL", "fail_count": 0, "query": ctx["query"], "source": "web",
            }
            try:
                await asyncio.to_thread(db.collection("analyses").document(run_id).set, err_doc)
            except Exception as werr:
                logger.warning("Firestore error-doc write failed: %s", werr)
        raise

    verdict, fail_count, report_fallback = evaluate_report(report)
    telemetry["report_fallback"] = report_fallback
    total_ms = _ms(t_start)
    _record(True, latency_ms=total_ms)

    now = datetime.now(timezone.utc)
    ttl_days = int(os.environ.get("DRISHTI_ANALYSES_TTL_DAYS", 90))
    expire_at = now + timedelta(days=ttl_days)
    fields = _log_telemetry(run_id, 200, ctx, telemetry, verdict, total_ms)

    doc = {
        "id": run_id,
        "created_at": now.isoformat(),
        "query": ctx["query"],
        "sops": ctx["sops"],
        "media_type": ctx["media_type"],
        "media_filename": ctx["media_filename"],
        "media_bytes": len(ctx["media_bytes"]),
        "plan": plan,
        "logs": logs,
        "results": results,
        "report": report,
        "verdict": verdict,
        "fail_count": fail_count,
        "plan_fallback": fields["plan_fallback"],
        "report_fallback": report_fallback,
        "stage_latency_ms": fields["stage_latency_ms"],
        "tokens_in": fields["tokens_in"],
        "tokens_out": fields["tokens_out"],
        "http_status": 200,
        "status": "ok",
        "source": "web",
    }

    if db:
        try:
            fs_doc = doc.copy()
            fs_doc["created_at"] = now
            fs_doc["expire_at"] = expire_at
            await asyncio.to_thread(db.collection("analyses").document(run_id).set, fs_doc)
        except Exception as exc:
            logger.warning("Firestore write failed: %s", exc)

    return doc


@app.post("/api/analyze")
async def analyze(
    request: Request,
    query: str = Form(..., min_length=3, max_length=MAX_QUERY_CHARS),
    sops: str = Form("", max_length=MAX_SOPS_CHARS),
    media: UploadFile = File(...),
):
    ctx = await _prepare(request, query, sops, media)
    doc = await _execute(ctx)
    return JSONResponse(doc)


@app.post("/api/analyze/stream")
async def analyze_stream(
    request: Request,
    query: str = Form(..., min_length=3, max_length=MAX_QUERY_CHARS),
    sops: str = Form("", max_length=MAX_SOPS_CHARS),
    media: UploadFile = File(...),
):
    """Same as /api/analyze but streams NDJSON events: plan, tool_started, tool_done, report,
    then a final `result` (or `error`) event. Validation errors are plain JSON HTTP errors."""
    ctx = await _prepare(request, query, sops, media)
    queue: asyncio.Queue = asyncio.Queue()

    async def runner():
        try:
            doc = await _execute(ctx, emit=queue.put_nowait)
            queue.put_nowait({"event": "result", "data": doc})
        except HTTPException as exc:
            queue.put_nowait({"event": "error", "status": exc.status_code, "detail": exc.detail})
        except Exception as exc:
            logger.warning("Stream pipeline crashed: %s", exc)
            queue.put_nowait({"event": "error", "status": 500, "detail": "Internal error."})
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(runner())

    async def gen():
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps(item, default=str) + "\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(gen(), media_type="application/x-ndjson")


async def _analysis_access(request: Request):
    """Opt-in admin gate for /api/analysis/{id} (DRISHTI_ANALYSIS_REQUIRES_ADMIN)."""
    if not ANALYSIS_REQUIRES_ADMIN:
        return None
    credentials = await security(request)
    return require_admin(credentials)


@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str, _: Optional[str] = Depends(_analysis_access)):
    db = get_db()
    if not db:
        raise HTTPException(status_code=503, detail="Firestore not configured")
    doc = await asyncio.to_thread(db.collection("analyses").document(analysis_id).get)
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return JSONResponse(jsonable_encoder(doc.to_dict()))


@app.get("/admin", response_class=HTMLResponse)
async def admin(_: str = Depends(require_admin)):
    db = get_db()
    rows = []
    if db:
        try:
            def _list():
                docs = db.collection("analyses").order_by(
                    "created_at", direction="DESCENDING"
                ).limit(50).stream()
                return [d.to_dict() for d in docs]

            rows = await asyncio.to_thread(_list)
        except Exception as exc:
            logger.warning("Admin query failed: %s", exc)
    return render_admin_page(rows)
