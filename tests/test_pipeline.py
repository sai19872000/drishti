import asyncio
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


def post(rep, **data):
    with patch("app.run_pipeline") as rp, patch("app.get_db", return_value=None):
        async def fake(*a, **k):
            return {"tools": []}, ["l"], {"x": "y"}, rep
        rp.side_effect = fake
        return client.post("/api/analyze", data={"query": "check it", **data}, files=FILES)


# --- verdict logic -------------------------------------------------------
@pytest.mark.parametrize("rep,expected", [
    (report(["PASS", "FAIL", "PASS"], "PASS"), "FAIL"),
    (report(["PASS", "fail", "PASS"], "PASS"), "FAIL"),
    (report(["PASS", "PASS", "PASS"], "FAIL"), "FAIL"),
    (report(["PASS", "PASS", "PASS"], "PASS"), "PASS"),
    (report([], "PASS"), "NEUTRAL"),
    ({"complianceChecks": [{"status": "PASS"}]}, "NEUTRAL"),  # missing finalVerdict
    ({"complianceChecks": "nope", "finalVerdict": "PASS"}, "NEUTRAL"),
])
def test_compute_verdict(rep, expected):
    assert appmod.compute_verdict(rep)[0] == expected


def test_verdict_stored_is_fail_when_check_fails():
    r = post(report(["PASS", "FAIL", "PASS"], "PASS"))
    assert r.status_code == 200
    assert r.json()["verdict"] == "FAIL" and r.json()["fail_count"] == 1


def test_template_uses_stored_verdict():
    html = client.get("/").text
    assert "data.verdict" in html
    assert 'role="alert"' in html and 'aria-live="polite"' in html


# --- planner normalisation -----------------------------------------------
def test_normalize_tools():
    assert appmod._normalize_tools(["sop_analysis", "OBJECT DETECTION", "bogus"]) == ["SOP Analysis", "Object Detection"]
    assert appmod._normalize_tools("Posture Analysis") == ["Posture Analysis"]
    assert appmod._normalize_tools(None) == []


def test_get_plan_unknown_tools_defaults_and_adds_sop():
    with patch("app._call_gemini", return_value='{"tools": ["magic"], "reasoning": "x"}'):
        plan = appmod._get_plan(MagicMock(), "q", "SOP text")
    assert "SOP Analysis" in plan["tools"] and set(plan["tools"]) <= set(appmod.ALLOWED_TOOLS)


def test_get_plan_adds_sop_when_provided():
    with patch("app._call_gemini", return_value='{"tools": ["Object Detection"], "reasoning": "x"}'):
        plan = appmod._get_plan(MagicMock(), "q", "SOP text")
    assert plan["tools"][0] == "SOP Analysis"


def test_pipeline_empty_plan_is_not_pass():
    with patch("app.get_gemini", return_value=MagicMock()), \
         patch("app._get_plan", return_value={"tools": [], "reasoning": ""}):
        plan, logs, results, rep = asyncio.run(appmod.run_pipeline("q", "", b"x", "image/jpeg"))
    assert results == {}
    assert appmod.compute_verdict(rep)[0] == "NEUTRAL"


def test_tools_run_concurrently():
    def slow(*a, **k):
        time.sleep(0.5)
        return "out"
    with patch("app.get_gemini", return_value=MagicMock()), \
         patch("app._get_plan", return_value={"tools": list(appmod.ALLOWED_TOOLS), "reasoning": ""}), \
         patch("app._run_sop_analysis", side_effect=slow), \
         patch("app._run_vision_tool", side_effect=slow), \
         patch("app._generate_report", return_value=report(["PASS"], "PASS")):
        t = time.time()
        _, _, results, _ = asyncio.run(appmod.run_pipeline("q", "sops", b"x", "image/jpeg"))
    assert len(results) == 4
    assert time.time() - t < 1.5


# --- JSON parsing --------------------------------------------------------
def test_parse_json_safe():
    fb = {"fb": 1}
    assert appmod._parse_json_safe('```json\n{"a": 1}\n```', fb) == {"a": 1}
    assert appmod._parse_json_safe('[1, 2]', fb) == fb
    assert appmod._parse_json_safe('not json', fb) == fb


def test_list_report_does_not_500():
    with patch("app.get_gemini", return_value=MagicMock()), \
         patch("app._get_plan", return_value={"tools": ["Object Detection"], "reasoning": ""}), \
         patch("app._run_vision_tool", return_value="o"), \
         patch("app._call_gemini", return_value="[1,2]"):
        _, _, _, rep = asyncio.run(appmod.run_pipeline("q", "", b"x", "image/jpeg"))
    assert appmod.compute_verdict(rep)[0] == "NEUTRAL"


# --- error mapping -------------------------------------------------------
class QuotaErr(Exception):
    code = 429


def _boom(exc):
    with patch("app.get_gemini", return_value=MagicMock()), patch("app.get_db", return_value=None), \
         patch("app._get_plan", side_effect=exc):
        return client.post("/api/analyze", data={"query": "check it"}, files=FILES)


def test_gemini_quota_maps_to_429_json():
    r = _boom(QuotaErr("quota"))
    assert r.status_code == 429 and "detail" in r.json()


def test_gemini_generic_maps_to_502_json():
    r = _boom(RuntimeError("kaput"))
    assert r.status_code == 502 and r.headers["content-type"].startswith("application/json")


def test_gemini_too_large_maps_to_413():
    class E(Exception):
        code = 400
    assert _boom(E("file too large")).status_code == 413


def test_timeout_maps_to_504():
    assert appmod._map_upstream_error(asyncio.TimeoutError()).status_code == 504


# --- input validation ----------------------------------------------------
def test_empty_query_422():
    r = client.post("/api/analyze", data={"query": ""}, files=FILES)
    assert r.status_code == 422
    r = client.post("/api/analyze", data={"query": "   "}, files=FILES)
    assert r.status_code == 422


def test_query_too_long_422():
    r = client.post("/api/analyze", data={"query": "x" * 2001}, files=FILES)
    assert r.status_code == 422


def test_oversize_upload_413(monkeypatch):
    monkeypatch.setattr(appmod, "MAX_UPLOAD_MB", 0.0001)
    r = client.post("/api/analyze", data={"query": "check it"},
                    files={"media": ("a.jpg", b"x" * 1000, "image/jpeg")})
    assert r.status_code == 413 and "too large" in r.json()["detail"].lower()


def test_rate_limit_429(monkeypatch):
    monkeypatch.setattr(appmod, "RATE_LIMIT_PER_MIN", 2)
    codes = [post(report(["PASS"], "PASS")).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_daily_cap():
    db = MagicMock()
    snap = MagicMock(exists=True)
    snap.to_dict.return_value = {"count": 5}
    db.collection.return_value.document.return_value.get.return_value = snap
    with patch.object(appmod, "DAILY_RUN_CAP", 5):
        assert appmod._daily_cap_exceeded(db) is True
    snap.to_dict.return_value = {"count": 1}
    with patch.object(appmod, "DAILY_RUN_CAP", 5):
        assert appmod._daily_cap_exceeded(db) is False
    assert appmod._daily_cap_exceeded(None) is False


# --- health --------------------------------------------------------------
def test_health_ok_without_firestore():
    r = client.get("/health")
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "ok" and b["version"] and "analyze_5xx_15m" in b
    assert client.get("/api/v1/health").status_code == 200


def test_health_degraded_when_firestore_down():
    db = MagicMock()
    db.collection.return_value.limit.return_value.stream.side_effect = RuntimeError("down")
    with patch("app.get_db", return_value=db), patch.object(appmod, "FIRESTORE_PROJECT", "p"):
        r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "degraded"


def test_health_degraded_without_key():
    with patch.object(appmod, "GEMINI_KEY", None):
        r = client.get("/health")
    assert r.status_code == 503 and r.json()["gemini_key_present"] is False


def test_health_ok_with_firestore():
    db = MagicMock()
    db.collection.return_value.limit.return_value.stream.return_value = iter([])
    with patch("app.get_db", return_value=db), patch.object(appmod, "FIRESTORE_PROJECT", "p"):
        r = client.get("/health")
    assert r.status_code == 200 and r.json()["firestore_ok"] is True


# --- admin & share -------------------------------------------------------
def test_admin_requires_auth():
    assert client.get("/admin").status_code == 401
    assert client.get("/admin", auth=("admin", "wrong")).status_code == 401


def test_admin_ok_renders_rows():
    db = MagicMock()
    d = MagicMock()
    d.to_dict.return_value = {"id": "a" * 32, "created_at": "2026-01-01T00:00:00", "query": "<b>q</b>",
                              "media_type": "image/jpeg", "verdict": "PASS", "fail_count": 0}
    db.collection.return_value.order_by.return_value.limit.return_value.stream.return_value = [d]
    with patch("app.get_db", return_value=db):
        r = client.get("/admin", auth=("admin", "secret"))
    assert r.status_code == 200 and "&lt;b&gt;" in r.text


def test_admin_empty():
    with patch("app.get_db", return_value=None):
        assert client.get("/admin", auth=("admin", "secret")).status_code == 200


def test_get_analysis_requires_admin():
    assert client.get("/api/analysis/abc").status_code == 401


def test_get_analysis_serializes_datetimes():
    from datetime import datetime, timezone
    db = MagicMock()
    doc = MagicMock(exists=True)
    doc.to_dict.return_value = {"id": "abc", "created_at": datetime.now(timezone.utc)}
    db.collection.return_value.document.return_value.get.return_value = doc
    with patch("app.get_db", return_value=db):
        r = client.get("/api/analysis/abc", auth=("admin", "secret"))
    assert r.status_code == 200 and r.json()["id"] == "abc"


def test_get_analysis_404_and_503():
    with patch("app.get_db", return_value=None):
        assert client.get("/api/analysis/abc", auth=("admin", "secret")).status_code == 503
    db = MagicMock()
    db.collection.return_value.document.return_value.get.return_value = MagicMock(exists=False)
    with patch("app.get_db", return_value=db):
        assert client.get("/api/analysis/abc", auth=("admin", "secret")).status_code == 404


def test_index_renders():
    assert client.get("/").status_code == 200
