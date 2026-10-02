from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app as appmod

PIPELINE_RESULT = ("plan", "logs", "results", {"finalVerdict": "PASS", "complianceChecks": []})


@pytest.fixture
def written_doc():
    """Run one analysis against a mocked Firestore and return the stored doc."""
    client = TestClient(appmod.app)
    db = MagicMock()
    document = db.collection.return_value.document.return_value

    def run():
        with patch("app.get_db", return_value=db), patch("app.run_pipeline", return_value=PIPELINE_RESULT):
            resp = client.post(
                "/api/analyze",
                data={"query": "test query"},
                files={"media": ("test_upload.jpg", b"test content", "image/jpeg")},
            )
        assert resp.status_code == 200
        db.collection.assert_called_with("analyses")
        document.set.assert_called_once()
        return document.set.call_args[0][0]

    return run


def test_ttl_default(written_doc):
    doc = written_doc()
    assert isinstance(doc["created_at"], datetime) and isinstance(doc["expire_at"], datetime)
    delta = doc["expire_at"] - doc["created_at"]
    assert abs(delta.total_seconds() - timedelta(days=90).total_seconds()) < 3600


def test_ttl_custom(written_doc, monkeypatch):
    monkeypatch.setenv("DRISHTI_ANALYSES_TTL_DAYS", "7")
    doc = written_doc()
    delta = doc["expire_at"] - doc["created_at"]
    assert abs(delta.total_seconds() - timedelta(days=7).total_seconds()) < 3600


def test_ttl_env_does_not_leak(written_doc):
    # the custom-TTL test above must not leave DRISHTI_ANALYSES_TTL_DAYS behind
    doc = written_doc()
    assert (doc["expire_at"] - doc["created_at"]).days == 90
