"""
Drishti — Agentic Vision Orchestrator
FastAPI server-side app. All Gemini calls server-side; no key in browser.
"""

import asyncio
import base64
import html
import json
import logging
import os
import secrets
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
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


def _call_gemini(client, prompt: str, media_bytes: Optional[bytes] = None,
                 media_mime: Optional[str] = None, json_mode: bool = False) -> str:
    """Sync Gemini call. Returns raw text (or JSON string if json_mode)."""
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    parts = [types.Part.from_text(text=prompt)]
    if media_bytes and media_mime:
        parts.append(types.Part.from_bytes(data=media_bytes, mime_type=media_mime))

    config_kwargs: dict = {}
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    response = client.models.generate_content(
        model=MODEL,
        contents=types.Content(role="user", parts=parts),
        config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
    )
    return response.text or ""


def _parse_json_safe(raw: str, fallback: dict) -> dict:
    try:
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        return json.loads(text)
    except Exception:
        return fallback


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

    raw = _call_gemini(client, prompt, json_mode=True)
    fallback = {
        "tools": ["SOP Analysis", "Object Detection", "Posture Analysis"],
        "reasoning": "Default tool selection applied.",
    }
    result = _parse_json_safe(raw, fallback)
    if "tools" not in result:
        result = fallback
    return result


def _run_sop_analysis(client, query: str, sops: str, media_bytes: bytes, media_mime: str) -> str:
    prompt = f"""You are an SOP compliance expert. Analyze the provided visual evidence against the following Standard Operating Procedures.

QUERY: {query}

STANDARD OPERATING PROCEDURES:
{sops}

Assess each SOP section visible in the evidence. Report specific observations for each relevant rule.
Be precise and clinical. Note any deviations, partial compliance, or confirmations."""
    return _call_gemini(client, prompt, media_bytes=media_bytes, media_mime=media_mime)


def _run_vision_tool(client, tool_type: str, query: str, media_bytes: bytes, media_mime: str) -> str:
    prompt = f"""You are a computer vision analysis system performing {tool_type}.

Inspection context: {query}

Analyze the visual evidence carefully. Report specific, observable findings.
Be precise and clinical — this is a compliance instrument, not a summary."""
    return _call_gemini(client, prompt, media_bytes=media_bytes, media_mime=media_mime)


def _generate_report(client, query: str, sops: str, results: dict, plan: dict) -> dict:
    results_text = "\n\n".join(
        f"=== {tool} ===\n{output}" for tool, output in results.items()
    )
    prompt = f"""You are a compliance audit report writer. Synthesize the vision analysis findings into a structured audit report.

INSPECTION QUERY: {query}

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

    raw = _call_gemini(client, prompt, json_mode=True)
    fallback = {
        "title": "Compliance Audit Report",
        "executiveSummary": "Analysis complete. See detailed findings below.",
        "complianceChecks": [
            {"rule": "Visual evidence review", "status": "NEUTRAL", "observation": "Structured analysis generated."}
        ],
        "visualDetails": ["Evidence processed."],
        "finalVerdict": "NEUTRAL",
        "detailedAnalysis": "Report generated from available visual evidence.",
    }
    return _parse_json_safe(raw, fallback)


async def run_pipeline(query: str, sops: str, media_bytes: bytes, media_mime: str):
    """Full agentic pipeline. Returns (plan, logs, results, report)."""
    client = get_gemini()
    if not client:
        raise HTTPException(status_code=503, detail="Server is not configured with a model key")

    logs = []

    logs.append("Orchestrator: Parsing query and evaluating tool requirements...")
    plan = await asyncio.to_thread(_get_plan, client, query, sops)
    logs.append(f"Orchestrator: Plan — tools={plan['tools']}. {plan.get('reasoning', '')}")

    results = {}
    tools = plan.get("tools", [])

    if "SOP Analysis" in tools and sops.strip():
        logs.append("SOP RAG: Cross-referencing Standard Operating Procedures against evidence...")
        results["SOP Analysis"] = await asyncio.to_thread(
            _run_sop_analysis, client, query, sops, media_bytes, media_mime
        )
        logs.append("SOP RAG: Complete.")

    if "Posture Analysis" in tools:
        logs.append("Vision: Running posture and body mechanics analysis...")
        results["Posture Analysis"] = await asyncio.to_thread(
            _run_vision_tool, client, "posture and body mechanics analysis", query, media_bytes, media_mime
        )
        logs.append("Vision: Posture analysis complete.")

    if "Object Detection" in tools:
        logs.append("Vision: Running object detection and equipment identification...")
        results["Object Detection"] = await asyncio.to_thread(
            _run_vision_tool, client, "object detection and equipment identification", query, media_bytes, media_mime
        )
        logs.append("Vision: Object detection complete.")

    if "Tracking Analysis" in tools:
        logs.append("Vision: Running motion tracking and sequence analysis...")
        results["Tracking Analysis"] = await asyncio.to_thread(
            _run_vision_tool, client, "motion tracking and sequence compliance analysis", query, media_bytes, media_mime
        )
        logs.append("Vision: Tracking analysis complete.")

    logs.append("Report Generator: Compiling structured compliance audit...")
    report = await asyncio.to_thread(_generate_report, client, query, sops, results, plan)
    verdict = report.get("finalVerdict", "UNKNOWN")
    logs.append(f"Analysis complete — Verdict: {verdict}")

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
.s-dot{{width:7px;height:7px;border-radius:50%;background:var(--gray-600);flex-shrink:0;transition:background .2s}}
.s-node.active .s-dot{{background:var(--warm)}}
.s-node.done .s-dot{{background:var(--cool)}}
.s-name{{font-family:var(--font-mono);font-size:12px;font-weight:500;letter-spacing:.04em;color:var(--gray-200)}}
.s-node.active .s-name{{color:var(--warm)}}
.s-node.done .s-name{{color:var(--cool)}}
.s-status{{
  margin-left:auto;font-family:var(--font-mono);font-size:10px;
  letter-spacing:var(--tracking-label);text-transform:uppercase;color:var(--gray-600);
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
  color:var(--gray-600);text-align:center;
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
      <form id="analyzeForm" enctype="multipart/form-data">
        <div class="field">
          <label for="query">Inspection query</label>
          <textarea id="query" name="query" rows="3" placeholder="What should the orchestrator verify?">{default_query}</textarea>
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
        <div class="log-feed" id="logFeed">
          <div class="log-line" style="color:var(--gray-600)">Waiting for analysis run...</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Report panel -->
  <div class="report-section">
    <div class="report-panel" id="reportPanel">
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
  stat.textContent = state === 'active' ? 'RUNNING' : state === 'done' ? 'DONE' : 'IDLE';
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

function animateNodes(logs, callback) {{
  // Animate nodes sequentially while request is in-flight
  let idx = 0;
  const keywords = [
    'input',
    'orchestrator',
    'sop',
    'vision',
    'report',
  ];

  function activateNext() {{
    if (idx >= NODE_SEQUENCE.length) {{
      callback && callback();
      return;
    }}
    if (idx > 0) setNodeState(idx - 1, 'done');
    setNodeState(idx, 'active');
    idx++;
  }}

  activateNext(); // start immediately
  // subsequent nodes advance every ~2s while waiting
  const interval = setInterval(() => {{
    if (idx < NODE_SEQUENCE.length) {{
      activateNext();
    }} else {{
      clearInterval(interval);
    }}
  }}, 2000);
  return interval;
}}

function esc(s) {{
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}}

function safeVerdict(v) {{
  return ['PASS','FAIL','NEUTRAL'].includes(v) ? v : 'NEUTRAL';
}}

function renderReport(data) {{
  const r = data.report;
  if (!r) {{ return '<p class="report-empty">Report data unavailable.</p>'; }}

  const verdict = safeVerdict(r.finalVerdict);
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

  const visuals = (r.visualDetails || []).map(v => `<li style="margin-bottom:4px">${{esc(v)}}</li>`).join('');

  return `
    <div class="report-header">
      <h2 class="report-title">${{esc(r.title || 'Compliance Audit Report')}}</h2>
      <span class="verdict-chip verdict-${{verdict}}">${{verdict}}</span>
    </div>
    <p class="report-summary">${{esc(r.executiveSummary)}}</p>
    <p class="checks-label">Compliance checks</p>
    ${{checks}}
    <p class="detail-label">Visual findings</p>
    <ul style="padding-left:18px;color:var(--gray-200);font-size:14px;line-height:1.75">${{visuals}}</ul>
    <p class="detail-label">Detailed analysis</p>
    <p class="detail-text">${{esc(r.detailedAnalysis)}}</p>
    <p style="margin-top:20px;font-family:var(--font-mono);font-size:10px;color:var(--gray-600);letter-spacing:.14em">
      RUN ID: ${{esc(data.id)}} &nbsp;·&nbsp; ${{new Date().toISOString()}}
    </p>`;
}}

document.getElementById('analyzeForm').addEventListener('submit', async (e) => {{
  e.preventDefault();
  const form = e.target;
  const btn = document.getElementById('runBtn');
  const reportPanel = document.getElementById('reportPanel');

  btn.disabled = true;
  btn.textContent = 'Running...';
  clearLog();
  resetNodes();
  reportPanel.innerHTML = '<p class="report-empty">Orchestrator running — please wait...</p>';

  // Client-side node animation while request is in-flight
  const interval = animateNodes();

  appendLog('Submitting evidence to orchestrator...');

  try {{
    const fd = new FormData(form);
    const resp = await fetch('/api/analyze', {{method:'POST', body: fd}});

    clearInterval(interval);
    // Mark all nodes done
    NODE_SEQUENCE.forEach((_, i) => setNodeState(i, 'done'));

    const data = await resp.json();

    if (!resp.ok) {{
      appendLog('Error: ' + (data.detail || resp.statusText), false);
      reportPanel.innerHTML = `<div class="error-panel">Analysis failed: ${{data.detail || resp.statusText}}</div>`;
      return;
    }}

    // Show logs from server
    clearLog();
    (data.logs || []).forEach((l, i) => appendLog(l, i === (data.logs.length - 1)));
    reportPanel.innerHTML = renderReport(data);

  }} catch (err) {{
    clearInterval(interval);
    NODE_SEQUENCE.forEach((_, i) => setNodeState(i, ''));
    appendLog('Network error: ' + err.message);
    reportPanel.innerHTML = `<div class="error-panel">Network error: ${{err.message}}</div>`;
  }} finally {{
    btn.disabled = false;
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
footer{{border-top:1px solid var(--line);padding:20px 0;font-family:var(--font-mono);font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--gray-600);text-align:center;margin-top:40px}}
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

@app.get("/health")
async def health():
    db = get_db()
    firestore_ok = False
    if db:
        try:
            # lightweight ping
            list(db.collection("analyses").limit(1).stream())
            firestore_ok = True
        except Exception:
            pass
    return {"status": "ok", "gemini_key_present": bool(GEMINI_KEY), "firestore_ok": firestore_ok}


@app.get("/", response_class=HTMLResponse)
async def index():
    return render_main_page(no_key=not bool(GEMINI_KEY))


@app.post("/api/analyze")
async def analyze(
    query: str = Form(...),
    sops: str = Form(""),
    media: UploadFile = File(...),
):
    if not GEMINI_KEY:
        raise HTTPException(status_code=503, detail="Server is not configured with a model key")

    media_bytes = await media.read()
    media_type = media.content_type or "image/jpeg"

    plan, logs, results, report = await run_pipeline(query, sops, media_bytes, media_type)

    checks = report.get("complianceChecks", [])
    fail_count = sum(1 for c in checks if c.get("status") == "FAIL")
    verdict = "FAIL" if fail_count > 0 else report.get("finalVerdict", "PASS")

    run_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()

    doc = {
        "id": run_id,
        "created_at": created_at,
        "query": query,
        "sops": sops,
        "media_type": media_type,
        "media_filename": media.filename or "upload",
        "media_bytes": len(media_bytes),
        "plan": plan,
        "logs": logs,
        "results": results,
        "report": report,
        "verdict": verdict,
        "fail_count": fail_count,
        "source": "web",
    }

    db = get_db()
    if db:
        try:
            db.collection("analyses").document(run_id).set(doc)
        except Exception as exc:
            logger.warning("Firestore write failed: %s", exc)

    return JSONResponse(doc)


@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    db = get_db()
    if not db:
        raise HTTPException(status_code=503, detail="Firestore not configured")
    doc = db.collection("analyses").document(analysis_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return JSONResponse(doc.to_dict())


@app.get("/admin", response_class=HTMLResponse)
async def admin(_: str = Depends(require_admin)):
    db = get_db()
    rows = []
    if db:
        try:
            docs = db.collection("analyses").order_by(
                "created_at", direction="DESCENDING"
            ).limit(50).stream()
            rows = [d.to_dict() for d in docs]
        except Exception as exc:
            logger.warning("Admin query failed: %s", exc)
    return render_admin_page(rows)
