import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

# app reads its config from the environment at import time. Set a hermetic env just for the
# import, then restore the process environment so nothing leaks into other tests/modules.
_ENV = {
    "GEMINI_API_KEY": "test_key",
    "AURACLE_ADMIN_USER": "admin",
    "AURACLE_ADMIN_PASS": "secret",
}
_DROP = ("FIRESTORE_PROJECT", "GCP_PROJECT", "GOOGLE_API_KEY", "DRISHTI_RATE_LIMIT_PER_MIN",
         "DRISHTI_DAILY_RUN_CAP", "DRISHTI_ANALYSIS_REQUIRES_ADMIN", "DRISHTI_TRUSTED_PROXY_HOPS",
         "DRISHTI_ANALYSES_TTL_DAYS", "MAX_UPLOAD_MB")
_saved = {k: os.environ.get(k) for k in list(_ENV) + list(_DROP)}
os.environ.update(_ENV)
for _k in _DROP:
    os.environ.pop(_k, None)
import app as _appmod  # noqa: E402,F401

for _k, _v in _saved.items():
    if _v is None:
        os.environ.pop(_k, None)
    else:
        os.environ[_k] = _v


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("DRISHTI_ANALYSES_TTL_DAYS", raising=False)
    _appmod._rate.clear()
    yield
    _appmod._rate.clear()
