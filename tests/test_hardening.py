import asyncio
import json
import logging
import re
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app as appmod

client = TestClient(appmod.app)
FILES = {"media": ("a.jpg", b"xx", "image/jpeg")}


def report(statuses, final="PASS"):
    return {
        "title": "t", "executiveSummary": "s", "visualDetails": [], "detailedAnalysis": "d",
        "complianceChecks": [{"rule": f"r{i}", "status": s, "observation": "o"} for i, s in enumerate(statuses)],
        "finalVerdict": final,
    }


def fake_pipeline(rep, telemetry_extra=None):
    async def fake(q, s, mb, mm, telemetry=None, emit=None):
        if telemetry is not None:
            telemetry.update(telemetry_extra or {})
            telemetry.setdefault("stage_latency_ms", {})["plan"] = 5
        if emit:
            emit({"event": "plan", "tools": ["Object Detection"], "skipped": ["SOP Analysis"]})
            emit({"event": "tool_started", "tool": "Object Detection"})
            emit({"event": "tool_done", "tool": "Object Detection", "latency_ms": 3, "output": "o"})
            emit({"event": "report", "report": rep})
        return {"tools": ["Object Detection"]}, ["l"], {"Object Detection": "o"}, rep
    return fake


def post(rep, headers=None, **kw):
    with patch("app.run_pipeline", side_effect=fake_pipeline(rep, kw.pop("tel", None))), \
         patch("app.get_db", return_value=kw.pop("db", None)):
        return client.post("/api/analyze", data={"query": "check it"}, files=FILES, headers=headers or {})


GOOD = report(["PASS", "PASS", "PASS"])


# --- trusted client IP / rate limit ---------------------------------------
def test_client_ip_uses_trusted_rightmost_hop():
    req = MagicMock()
    req.headers = {"x-forwarded-for": "6.6.6.6, 7.7.7.7, 203.0.113.9"}
    assert appmod._client_ip(req) == "203.0.113.9"


def test_client_ip_hops_and_fallback(monkeypatch):
    req = MagicMock()
    req.client.host = "10.0.0.1"
    req.headers = {"x-forwarded-for": "1.1.1.1, 2.2.2.2, 3.3.3.3"}
    monkeypatch.setattr(appmod, "TRUSTED_PROXY_HOPS", 2)
    assert appmod._client_ip(req) == "2.2.2.2"
    monkeypatch.setattr(appmod, "TRUSTED_PROXY_HOPS", 5)  # fewer entries than hops
    assert appmod._client_ip(req) == "10.0.0.1"
    req.headers = {}
    assert appmod._client_ip(req) == "10.0.0.1"


def test_spoofed_leftmost_xff_does_not_reset_limit(monkeypatch):
    monkeypatch.setattr(appmod, "RATE_LIMIT_PER_MIN", 2)
    codes = [post(GOOD, headers={"X-Forwarded-For": f"9.9.9.{i}, 203.0.113.9"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]
    assert list(appmod._rate) == ["203.0.113.9"]  # one key, no memory growth per fake IP


def test_rate_limit_disabled_by_default():
    assert appmod.RATE_LIMIT_PER_MIN == 0
    assert all(post(GOOD).status_code == 200 for _ in range(15))


# --- verdict gate / validation --------------------------------------------
def test_all_neutral_with_final_pass_is_neutral():
    assert appmod.compute_verdict(report(["NEUTRAL", "NEUTRAL", "NEUTRAL"], "PASS"))[0] == "NEUTRAL"
    r = post(report(["NEUTRAL", "NEUTRAL", "NEUTRAL"], "PASS"))
    assert r.json()["verdict"] == "NEUTRAL"


@pytest.mark.parametrize("rep,ok", [
    (GOOD, True),
    (report(["PASS", "PASS"]), False),
    (report(["PASS", "PASS", "nope"]), False),
    (report(["PASS", "PASS", "PASS"], final="maybe"), False),
    ([1, 2], False),
    ({"finalVerdict": "PASS", "complianceChecks": "x"}, False),
    ({"finalVerdict": "PASS", "complianceChecks": ["a", "b", "c"]}, False),
])
def test_validate_report(rep, ok):
    assert appmod.validate_report(rep)[0] is ok


def test_fewer_than_three_checks_is_neutral_with_fallback_flag():
    r = post(report(["PASS"], "PASS"))
    body = r.json()
    assert body["verdict"] == "NEUTRAL" and body["report_fallback"] is True


def test_valid_report_has_no_fallback_flag():
    body = post(GOOD).json()
    assert body["verdict"] == "PASS" and body["report_fallback"] is False and body["plan_fallback"] is False


def test_invalid_report_keeps_recorded_fail():
    assert post(report(["PASS", "FAIL"], "PASS")).json()["verdict"] == "FAIL"


def test_generate_report_fallback_sets_flag():
    with patch("app._call_gemini", return_value="not json"):
        rep = appmod._generate_report(MagicMock(), "q", "s", {"t": "o"}, {})
    assert rep["report_fallback"] is True
    assert appmod.evaluate_report(rep) == ("NEUTRAL", 0, True)


def test_plan_fallback_flag():
    with patch("app._call_gemini", return_value='{"tools": ["magic"]}'):
        assert appmod._get_plan(MagicMock(), "q", "")["plan_fallback"] is True
    with patch("app._call_gemini", return_value="garbage"):
        assert appmod._get_plan(MagicMock(), "q", "")["plan_fallback"] is True
    with patch("app._call_gemini", return_value='{"tools": ["Object Detection"]}'):
        assert appmod._get_plan(MagicMock(), "q", "")["plan_fallback"] is False


# --- telemetry -------------------------------------------------------------
def test_one_structured_log_line_and_doc_fields(caplog):
    db = MagicMock()
    with caplog.at_level(logging.INFO, logger="drishti"):
        r = post(GOOD, db=db, tel={"tokens_in": 11, "tokens_out": 7, "plan_fallback": True})
    assert r.status_code == 200
    lines = [json.loads(m) for m in caplog.messages if m.startswith('{"event": "analysis"')]
    assert len(lines) == 1
    line = lines[0]
    for k in ("run_id", "stage_latency_ms", "tokens_in", "tokens_out", "plan_fallback", "report_fallback",
              "verdict", "http_status", "media_type", "media_bytes"):
        assert k in line
    assert line["tokens_in"] == 11 and line["http_status"] == 200 and line["verdict"] == "PASS"
    doc = db.collection.return_value.document.return_value.set.call_args[0][0]
    for k in ("stage_latency_ms", "tokens_in", "tokens_out", "plan_fallback", "report_fallback", "http_status"):
        assert k in doc
    assert doc["tokens_out"] == 7 and doc["plan_fallback"] is True and doc["status"] == "ok"


def test_error_doc_written_on_failure(caplog):
    db = MagicMock()

    async def boom(q, s, mb, mm, telemetry=None, emit=None):
        telemetry["stage"] = "tools"
        telemetry["error_class"] = "RuntimeError"
        raise appmod.HTTPException(status_code=502, detail="Upstream model request failed.")

    with patch("app.run_pipeline", side_effect=boom), patch("app.get_db", return_value=db), \
         caplog.at_level(logging.INFO, logger="drishti"):
        r = client.post("/api/analyze", data={"query": "check it"}, files=FILES)
    assert r.status_code == 502
    doc = db.collection.return_value.document.return_value.set.call_args[0][0]
    assert doc["status"] == "error" and doc["stage"] == "tools" and doc["error_class"] == "RuntimeError"
    assert any('"error_class": "RuntimeError"' in m for m in caplog.messages)


def test_usage_metadata_tokens_collected():
    resp = MagicMock(text="hi")
    resp.usage_metadata.prompt_token_count = 10
    resp.usage_metadata.candidates_token_count = 4
    gem = MagicMock()
    gem.models.generate_content.return_value = resp
    usage = {"tokens_in": 0, "tokens_out": 0, "calls": 0}
    tok = appmod._usage_ctx.set(usage)
    try:
        assert appmod._call_gemini(gem, "p") == "hi"
    finally:
        appmod._usage_ctx.reset(tok)
    assert usage["tokens_in"] == 10 and usage["tokens_out"] == 4


def test_pipeline_fills_telemetry_and_stage_latencies():
    tel = {}
    with patch("app.get_gemini", return_value=MagicMock()), \
         patch("app._get_plan", return_value={"tools": ["Object Detection"], "reasoning": "", "plan_fallback": True}), \
         patch("app._run_vision_tool", return_value="o"), \
         patch("app._generate_report", return_value=report(["PASS"])):
        asyncio.run(appmod.run_pipeline("q", "", b"x", "image/jpeg", telemetry=tel))
    assert tel["plan_fallback"] is True and tel["report_fallback"] is True
    assert {"plan", "tools", "report"} <= set(tel["stage_latency_ms"])


def test_health_reports_latency_percentiles():
    with appmod._stats_lock:
        appmod._stats["latencies_ms"] = []
    for ms in (100, 200, 300, 400, 1000):
        appmod._record(True, latency_ms=ms)
    body = client.get("/health").json()
    assert body["analyze_p50_ms"] == 300 and body["analyze_p95_ms"] == 1000
    assert appmod._percentile([], 50) is None


def test_model_probe():
    gem = MagicMock()
    with patch("app.get_gemini", return_value=gem):
        r = client.get("/health/model")
    assert r.status_code == 200 and r.json()["model_ok"] is True
    gem.models.get.assert_called_once_with(model=appmod.MODEL)
    gem.models.get.side_effect = RuntimeError("404 model retired")
    with patch("app.get_gemini", return_value=gem):
        r = client.get("/health/model")
    assert r.status_code == 503 and r.json()["error_class"] == "RuntimeError"
    with patch("app.get_gemini", return_value=None):
        assert client.get("/health/model").status_code == 503


# --- concurrency -----------------------------------------------------------
def test_four_one_second_tools_finish_under_3_5s():
    def slow(*a, **k):
        time.sleep(1)
        return "out"
    with patch("app.get_gemini", return_value=MagicMock()), \
         patch("app._get_plan", return_value={"tools": list(appmod.ALLOWED_TOOLS), "reasoning": ""}), \
         patch("app._run_sop_analysis", side_effect=slow), \
         patch("app._run_vision_tool", side_effect=slow), \
         patch("app._generate_report", return_value=GOOD):
        t = time.time()
        _, _, results, _ = asyncio.run(appmod.run_pipeline("q", "sops", b"x", "image/jpeg"))
    assert len(results) == 4 and time.time() - t < 3.5


# --- Files API upload ------------------------------------------------------
def test_video_uploaded_once_and_uri_passed_to_all_tools():
    gem = MagicMock()
    f = MagicMock(uri="files/abc", name="files/abc")
    f.state.name = "ACTIVE"
    gem.files.upload.return_value = f
    seen = []

    def tool(*a, **k):
        seen.append(a[-1])
        return "o"
    with patch("app.get_gemini", return_value=gem), \
         patch("app._get_plan", return_value={"tools": list(appmod.ALLOWED_TOOLS), "reasoning": ""}), \
         patch("app._run_sop_analysis", side_effect=tool), patch("app._run_vision_tool", side_effect=tool), \
         patch("app._generate_report", return_value=GOOD):
        asyncio.run(appmod.run_pipeline("q", "sops", b"vid", "video/mp4"))
    assert gem.files.upload.call_count == 1
    assert seen == ["files/abc"] * 4
    gem.files.delete.assert_called_once()


def test_small_image_stays_inline_and_upload_failure_falls_back():
    gem = MagicMock()
    assert appmod._upload_media(gem, b"x", "image/jpeg") is None
    gem.files.upload.assert_not_called()
    gem.files.upload.side_effect = RuntimeError("boom")
    assert appmod._upload_media(gem, b"x", "video/mp4") is None


def test_call_gemini_uses_file_uri():
    gem = MagicMock()
    gem.models.generate_content.return_value = MagicMock(text="ok", usage_metadata=None)
    appmod._call_gemini(gem, "p", media_bytes=b"x", media_mime="video/mp4", media_uri="files/abc")
    parts = gem.models.generate_content.call_args.kwargs["contents"].parts
    assert parts[1].file_data.file_uri == "files/abc" and parts[1].inline_data is None


# --- streaming ---------------------------------------------------------------
def test_stream_emits_ndjson_events_then_result():
    with patch("app.run_pipeline", side_effect=fake_pipeline(GOOD)), patch("app.get_db", return_value=None):
        r = client.post("/api/analyze/stream", data={"query": "check it"}, files=FILES)
    assert r.status_code == 200 and "ndjson" in r.headers["content-type"]
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    names = [e["event"] for e in events]
    assert names == ["plan", "tool_started", "tool_done", "report", "result"]
    assert events[0]["skipped"] == ["SOP Analysis"]
    assert events[-1]["data"]["verdict"] == "PASS"


def test_stream_error_event_and_validation_errors():
    async def boom(*a, **k):
        raise appmod.HTTPException(status_code=429, detail="quota")
    with patch("app.run_pipeline", side_effect=boom), patch("app.get_db", return_value=None):
        r = client.post("/api/analyze/stream", data={"query": "check it"}, files=FILES)
    last = json.loads(r.text.splitlines()[-1])
    assert last == {"event": "error", "status": 429, "detail": "quota"}
    assert client.post("/api/analyze/stream", data={"query": ""}, files=FILES).status_code == 422


def test_pipeline_emits_plan_tool_and_report_events_with_skipped():
    events = []
    with patch("app.get_gemini", return_value=MagicMock()), \
         patch("app._get_plan", return_value={"tools": ["Object Detection"], "reasoning": "r"}), \
         patch("app._run_vision_tool", return_value="o"), patch("app._generate_report", return_value=GOOD):
        asyncio.run(appmod.run_pipeline("q", "", b"x", "image/jpeg", emit=events.append))
    assert [e["event"] for e in events] == ["plan", "tool_started", "tool_done", "report"]
    assert "SOP Analysis" in events[0]["skipped"] and events[0]["tools"] == ["Object Detection"]


# --- upload cap --------------------------------------------------------------
def test_content_length_rejected_before_body_is_read(monkeypatch):
    monkeypatch.setattr(appmod, "MAX_UPLOAD_MB", 0.001)
    # Malformed multipart body: without the early check this would be a 422, not a 413.
    r = client.post("/api/analyze", content=b"x" * (2 * 1024 * 1024),
                    headers={"content-type": "multipart/form-data; boundary=zzz"})
    assert r.status_code == 413 and r.headers["content-type"].startswith("application/json")
    assert "too large" in r.json()["detail"].lower()
    r = client.post("/api/analyze/stream", content=b"x" * (2 * 1024 * 1024),
                    headers={"content-type": "multipart/form-data; boundary=zzz"})
    assert r.status_code == 413


def test_template_interpolates_max_upload_mb(monkeypatch):
    monkeypatch.setattr(appmod, "MAX_UPLOAD_MB", 12.5)
    assert "const MAX_MB = 12.5;" in client.get("/").text
    monkeypatch.setattr(appmod, "MAX_UPLOAD_MB", 30.0)
    assert "const MAX_MB = 30;" in client.get("/").text


# --- UI / accessibility (static) -------------------------------------------
def test_gray_600_never_used_for_text():
    pages = [appmod.render_main_page(), appmod.render_admin_page([]), appmod.BASE_STYLES]
    for page in pages:
        assert not re.search(r"(?<![-\w])color\s*:\s*var\(--gray-600\)", page)
        assert "-webkit-text-fill-color:var(--gray-600)" not in page
    rows = [{"id": "a" * 32, "created_at": "2026-01-01T00:00:00", "query": "q", "verdict": "PASS"}]
    assert "var(--gray-600)" not in appmod.render_admin_page(rows)


def test_ui_streams_and_renders_evidence_and_skipped():
    html = client.get("/").text
    assert "/api/analyze/stream" in html and "SKIPPED" in html
    assert '<details class="evidence">' in html
    assert "animateNodes" not in html  # no timer-driven progress
    assert 'role="status"' in html and 'role="alert"' in html


# --- firestore calls are off the event loop ------------------------------------
def test_firestore_calls_use_to_thread():
    calls = []
    real = asyncio.to_thread

    async def spy(fn, *a, **k):
        calls.append(getattr(fn, "__name__", repr(fn)))
        return await real(fn, *a, **k)

    db = MagicMock()
    doc = MagicMock(exists=True)
    doc.to_dict.return_value = {"id": "abc"}
    db.collection.return_value.document.return_value.get.return_value = doc
    db.collection.return_value.order_by.return_value.limit.return_value.stream.return_value = []
    with patch("app.get_db", return_value=db), patch("app.asyncio.to_thread", spy):
        client.get("/api/analysis/abc")
        client.get("/admin", auth=("admin", "secret"))
        post_calls = len(calls)
        post(GOOD, db=db)
    assert any(".get" in c for c in calls[:post_calls]) and "_list" in calls
    assert any(".set" in c for c in calls[post_calls:])
