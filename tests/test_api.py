"""
API endpoint tests.
"""

import pytest
from httpx import AsyncClient


class TestHealthEndpoint:
    """Health check endpoint tests."""

    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "version" in data


class TestSynthesisEndpoint:
    """Synthesis endpoint tests."""

    def test_synthesize_without_auth(self, client):
        """Test synthesis requires authentication."""
        response = client.post(
            "/v1/synthesize",
            json={
                "text": "Hello world",
                "voice_profile_id": "default",
            },
        )
        assert response.status_code == 401

    def test_synthesize_with_auth(self, client, auth_headers):
        """Test synthesis with valid auth."""
        response = client.post(
            "/v1/synthesize",
            json={
                "text": "Hello world",
                "voice_profile_id": "default",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "audio_base64" in data
        assert "duration_ms" in data
        assert "generation_time_ms" in data

    def test_synthesize_validation_error(self, client, auth_headers):
        """Test synthesis with invalid input."""
        response = client.post(
            "/v1/synthesize",
            json={
                "text": "",  # Empty text
                "voice_profile_id": "default",
            },
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_synthesize_long_text(self, client, auth_headers):
        """Test synthesis with long text."""
        response = client.post(
            "/v1/synthesize",
            json={
                "text": "Hello world. " * 100,
                "voice_profile_id": "default",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200


class TestProfileEndpoints:
    """Voice profile endpoint tests."""

    def test_create_profile(self, client, auth_headers):
        """Test creating a voice profile."""
        response = client.post(
            "/v1/profiles",
            json={
                "name": "Test Voice",
                "agent_id": "agent-123",
                "description": "A test voice profile",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Voice"
        assert data["status"] == "pending"

    def test_list_profiles(self, client, auth_headers):
        """Test listing voice profiles."""
        response = client.get("/v1/profiles", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "profiles" in data
        assert "total" in data

    def test_get_profile_not_found(self, client, auth_headers):
        """Test getting non-existent profile."""
        response = client.get(
            "/v1/profiles/nonexistent",
            headers=auth_headers,
        )
        assert response.status_code == 404


class TestTrainingEndpoints:
    """Training endpoint tests."""

    def test_start_training(self, client, auth_headers):
        """Test starting a training job."""
        response = client.post(
            "/v1/training",
            json={
                "voice_profile_id": "test-profile",
                "use_lora": True,
                "lora_rank": 8,
                "num_epochs": 10,
                "gpu_type": "a100",
            },
            headers=auth_headers,
        )
        # May return 200 or 400 depending on training data
        assert response.status_code in (200, 400)

    def test_get_training_job_not_found(self, client, auth_headers):
        """Test getting non-existent training job."""
        response = client.get(
            "/v1/training/nonexistent",
            headers=auth_headers,
        )
        assert response.status_code == 404


class TestRateLimiting:
    """Rate limiting tests."""

    def test_rate_limit_headers(self, client, auth_headers):
        """Test rate limit headers are present."""
        response = client.post(
            "/v1/synthesize",
            json={
                "text": "Hello",
                "voice_profile_id": "default",
            },
            headers=auth_headers,
        )

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers


class TestMetricsEndpoint:
    """Metrics endpoint tests."""

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "csm_http_requests_total" in response.text
