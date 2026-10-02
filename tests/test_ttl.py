import os
import io
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient

from app import app

class TestTTL(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    @patch('app.get_db')
    @patch('app.run_pipeline')
    def test_ttl_default(self, mock_run_pipeline, mock_get_db):
        mock_run_pipeline.return_value = (
            "plan", 
            "logs", 
            "results", 
            {"finalVerdict": "PASS", "complianceChecks": []}
        )
        
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_document = MagicMock()
        mock_collection.document.return_value = mock_document
        
        # Test with default TTL (90 days)
            
        file_content = b"test content"
            
        response = self.client.post(
            "/api/analyze",
            data={"query": "test query"},
            files={"media": ("test_upload.jpg", file_content, "image/jpeg")}
        )
            
        self.assertEqual(response.status_code, 200)
        
        # Assert the doc was written to Firestore
        mock_db.collection.assert_called_with("analyses")
        mock_collection.document.assert_called()
        mock_document.set.assert_called_once()
        
        doc = mock_document.set.call_args[0][0]
        
        self.assertIn("created_at", doc)
        self.assertIn("expire_at", doc)
        
        self.assertTrue(isinstance(doc["created_at"], datetime))
        self.assertTrue(isinstance(doc["expire_at"], datetime))
        
        delta = doc["expire_at"] - doc["created_at"]
        self.assertAlmostEqual(delta.total_seconds(), timedelta(days=90).total_seconds(), delta=3600)
        
    @patch('app.get_db')
    @patch('app.run_pipeline')
    def test_ttl_custom(self, mock_run_pipeline, mock_get_db):
        mock_run_pipeline.return_value = (
            "plan", 
            "logs", 
            "results", 
            {"finalVerdict": "PASS", "complianceChecks": []}
        )
        
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_collection = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_document = MagicMock()
        mock_collection.document.return_value = mock_document
        
        # Test with custom TTL (7 days)
        patcher = patch.dict(os.environ, {'DRISHTI_ANALYSES_TTL_DAYS': '7'})
        patcher.start()
        self.addCleanup(patcher.stop)
            
        file_content = b"test content"
            
        response = self.client.post(
            "/api/analyze",
            data={"query": "test query"},
            files={"media": ("test_upload.jpg", file_content, "image/jpeg")}
        )
            
        self.assertEqual(response.status_code, 200)
        
        doc = mock_document.set.call_args[0][0]
        
        delta = doc["expire_at"] - doc["created_at"]
        self.assertAlmostEqual(delta.total_seconds(), timedelta(days=7).total_seconds(), delta=3600)

if __name__ == '__main__':
    unittest.main()
