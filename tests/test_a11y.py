"""Accessibility gate: axe-core via Playwright against / and /admin.

Requires 0 serious or critical violations. Skipped unless Playwright + axe-playwright-python and a
Chromium build are installed (the `a11y` CI job installs them and sets REQUIRE_A11Y=1 so a missing
toolchain fails instead of skipping)."""
import os
import socket
import threading
import time

import pytest

REQUIRED = os.environ.get("REQUIRE_A11Y") == "1"


def _need(mod):
    try:
        return pytest.importorskip(mod)
    except pytest.skip.Exception:
        if REQUIRED:
            pytest.fail(f"{mod} is required when REQUIRE_A11Y=1")
        raise


@pytest.fixture(scope="module")
def server():
    import uvicorn

    import app as appmod

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(appmod.app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    t.join(5)


@pytest.mark.parametrize("path,auth", [("/", False), ("/admin", True)])
def test_no_serious_or_critical_axe_violations(server, path, auth):
    sync_api = _need("playwright.sync_api")
    axe_mod = _need("axe_playwright_python.sync_playwright")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:
            if REQUIRED:
                raise
            pytest.skip(f"chromium unavailable: {exc}")
        ctx = browser.new_context(http_credentials={"username": "admin", "password": "secret"} if auth else None)
        page = ctx.new_page()
        page.goto(server + path)
        results = axe_mod.Axe().run(page)
        browser.close()
    bad = [v for v in results.response["violations"] if v.get("impact") in ("serious", "critical")]
    assert not bad, [(v["id"], v["impact"], v["help"]) for v in bad]
