import os
import sys

# Hermetic env: set before app import; keep ttl env clean between tests.
os.environ.setdefault("GEMINI_API_KEY", "test_key")
for _k in ("FIRESTORE_PROJECT", "GCP_PROJECT"):
    os.environ.pop(_k, None)
os.environ["AURACLE_ADMIN_USER"] = "admin"
os.environ["AURACLE_ADMIN_PASS"] = "secret"
os.environ["DRISHTI_RATE_LIMIT_PER_MIN"] = "1000"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("DRISHTI_ANALYSES_TTL_DAYS", raising=False)
    import app as appmod
    appmod._rate.clear()
    yield
