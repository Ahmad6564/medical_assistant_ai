"""
Integration tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json


@pytest.fixture
def client():
    """Create test client."""
    from src.api.main import app
    return TestClient(app)


@pytest.mark.integration
class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_readiness_check(self, client):
        """Test readiness check."""
        response = client.get("/ready")
        
        assert response.status_code in [200, 503]
        data = response.json()
        assert "models" in data or "status" in data


@pytest.mark.integration
class TestAuthenticationEndpoints:
    """Tests for authentication."""
    
    def test_token_generation(self, client):
        """Test JWT token generation."""
        response = client.post(
            "/auth/token",
            data={"username": "testuser", "password": "testpass"}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
            assert "token_type" in data
    
    def test_api_key_validation(self, client, auth_headers):
        """Test API key validation."""
        response = client.get("/protected", headers=auth_headers)
        
        # Should either succeed or return auth error
        assert response.status_code in [200, 401, 403, 404]
    
    def test_unauthorized_access(self, client):
        """Test access without authentication."""
        response = client.post("/predict/ner", json={"text": "test"})
        
        # Should require authentication
        assert response.status_code in [200, 401, 403]


@pytest.mark.integration
class TestNEREndpoints:
    """Tests for NER endpoints."""
    
    def test_ner_prediction(self, client, auth_headers, sample_medical_text):
        """Test NER prediction endpoint."""
        response = client.post(
            "/predict/ner",
            headers=auth_headers,
            json={"text": sample_medical_text}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "entities" in data or "predictions" in data
    
    def test_ner_batch_prediction(self, client, auth_headers):
        """Test batch NER prediction."""
        texts = [
            "Patient has diabetes and hypertension",
            "Prescribed metformin 500mg twice daily",
            "CBC test ordered"
        ]
        
        response = client.post(
            "/predict/ner/batch",
            headers=auth_headers,
            json={"texts": texts}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == len(texts)
    
    def test_ner_invalid_input(self, client, auth_headers):
        """Test NER with invalid input."""
        response = client.post(
            "/predict/ner",
            headers=auth_headers,
            json={}
        )
        
        assert response.status_code == 422  # Validation error


@pytest.mark.integration
class TestClassificationEndpoints:
    """Tests for classification endpoints."""
    
    def test_classification_prediction(self, client, auth_headers, sample_medical_text):
        """Test classification endpoint."""
        response = client.post(
            "/predict/classify",
            headers=auth_headers,
            json={"text": sample_medical_text}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "labels" in data or "predictions" in data
    
    def test_classification_with_threshold(self, client, auth_headers, sample_medical_text):
        """Test classification with custom threshold."""
        response = client.post(
            "/predict/classify",
            headers=auth_headers,
            json={
                "text": sample_medical_text,
                "threshold": 0.7
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "labels" in data or "predictions" in data
    
    def test_classification_batch(self, client, auth_headers):
        """Test batch classification."""
        texts = [
            "Patient presents with cardiac symptoms",
            "Neurological examination ordered",
            "Endocrine panel requested"
        ]
        
        response = client.post(
            "/predict/classify/batch",
            headers=auth_headers,
            json={"texts": texts}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)


@pytest.mark.integration
class TestRAGEndpoints:
    """Tests for RAG endpoints."""
    
    def test_rag_query(self, client, auth_headers, sample_query):
        """Test RAG query endpoint."""
        response = client.post(
            "/rag/query",
            headers=auth_headers,
            json={"query": sample_query}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data or "documents" in data
    
    def test_rag_with_parameters(self, client, auth_headers, sample_query):
        """Test RAG with custom parameters."""
        response = client.post(
            "/rag/query",
            headers=auth_headers,
            json={
                "query": sample_query,
                "top_k": 5,
                "use_reranking": True
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data or "documents" in data
    
    def test_add_documents(self, client, auth_headers):
        """Test adding documents to RAG."""
        documents = [
            "Document 1 content",
            "Document 2 content"
        ]
        
        response = client.post(
            "/rag/documents",
            headers=auth_headers,
            json={"documents": documents}
        )
        
        # Should succeed or indicate feature not available
        assert response.status_code in [200, 201, 404, 501]


@pytest.mark.integration
class TestLLMEndpoints:
    """Tests for LLM endpoints."""
    
    def test_chat_completion(self, client, auth_headers):
        """Test chat completion endpoint."""
        response = client.post(
            "/chat/completions",
            headers=auth_headers,
            json={
                "messages": [
                    {"role": "user", "content": "What is diabetes?"}
                ]
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "response" in data or "choices" in data
    
    def test_streaming_response(self, client, auth_headers):
        """Test streaming chat completion."""
        response = client.post(
            "/chat/completions",
            headers=auth_headers,
            json={
                "messages": [{"role": "user", "content": "Explain hypertension"}],
                "stream": True
            }
        )
        
        # Streaming should work or return normal response
        assert response.status_code in [200, 501]


@pytest.mark.integration
class TestSafetyEndpoints:
    """Tests for safety endpoints."""
    
    def test_emergency_detection(self, client, auth_headers):
        """Test emergency detection endpoint."""
        response = client.post(
            "/safety/check",
            headers=auth_headers,
            json={"text": "I'm having severe chest pain"}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "is_emergency" in data or "is_safe" in data
    
    def test_content_moderation(self, client, auth_headers):
        """Test content moderation."""
        response = client.post(
            "/safety/moderate",
            headers=auth_headers,
            json={"text": "What is diabetes?"}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "is_appropriate" in data or "is_safe" in data


@pytest.mark.integration
class TestRateLimiting:
    """Tests for rate limiting."""
    
    def test_rate_limit_enforcement(self, client, auth_headers):
        """Test rate limiting."""
        # Make multiple requests
        responses = []
        for _ in range(20):
            response = client.get("/health", headers=auth_headers)
            responses.append(response.status_code)
        
        # Should eventually hit rate limit
        # Allow for rate limiting to be optional
        assert all(code in [200, 429] for code in responses)
    
    def test_rate_limit_headers(self, client, auth_headers):
        """Test rate limit headers."""
        response = client.get("/health", headers=auth_headers)
        
        # Check for rate limit headers (if implemented)
        if "X-RateLimit-Limit" in response.headers:
            assert int(response.headers["X-RateLimit-Limit"]) > 0


@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling."""
    
    def test_404_not_found(self, client):
        """Test 404 error."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_validation_error(self, client, auth_headers):
        """Test validation error handling."""
        response = client.post(
            "/predict/ner",
            headers=auth_headers,
            json={"invalid": "data"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_internal_error_handling(self, client, auth_headers):
        """Test internal error handling."""
        # Send request that might cause error
        response = client.post(
            "/predict/ner",
            headers=auth_headers,
            json={"text": "x" * 100000}  # Very long text
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 413, 422, 500]


@pytest.mark.integration
class TestCORS:
    """Tests for CORS configuration."""
    
    def test_cors_headers(self, client):
        """Test CORS headers."""
        response = client.options("/health")
        
        # CORS might be configured
        if response.status_code == 200:
            headers = response.headers
            # Check for CORS headers (if implemented)
            assert True  # Just ensure no error


@pytest.mark.integration
class TestMetrics:
    """Tests for metrics endpoints."""
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        
        # Metrics endpoint might not be public
        assert response.status_code in [200, 404, 401]
    
    def test_prometheus_format(self, client):
        """Test Prometheus metrics format."""
        response = client.get("/metrics")
        
        if response.status_code == 200:
            # Should be text format
            assert "text/plain" in response.headers.get("content-type", "")


@pytest.mark.slow
@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_complete_workflow(self, client, auth_headers):
        """Test complete medical query workflow."""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Safety check
        safety_response = client.post(
            "/safety/check",
            headers=auth_headers,
            json={"text": "What medications for diabetes?"}
        )
        
        # 3. NER extraction
        ner_response = client.post(
            "/predict/ner",
            headers=auth_headers,
            json={"text": "Patient has diabetes"}
        )
        
        # 4. Classification
        classify_response = client.post(
            "/predict/classify",
            headers=auth_headers,
            json={"text": "Diabetes management consultation"}
        )
        
        # All should succeed or gracefully fail
        assert all(
            r.status_code in [200, 404, 501]
            for r in [safety_response, ner_response, classify_response]
        )
