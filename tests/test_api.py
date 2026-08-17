"""Integration and endpoint tests for the FastAPI prediction service."""

import pytest
from fastapi.testclient import TestClient
from src.inference.api import app


@pytest.fixture(scope="module")
def client():
    """Context-managed TestClient ensuring lifespan startup loads all artifacts."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """Verify that GET /health returns 200 OK and ready status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_teams_endpoint(client: TestClient) -> None:
    """Verify that GET /teams returns a list containing major historical teams."""
    response = client.get("/teams")
    assert response.status_code == 200
    data = response.json()
    assert "teams" in data
    assert isinstance(data["teams"], list)
    assert len(data["teams"]) > 50
    assert "Argentina" in data["teams"]
    assert "France" in data["teams"]


def test_prediction_endpoint_valid(client: TestClient) -> None:
    """Verify that POST /prediction computes coherent 3-way probabilities and scorelines."""
    payload = {
        "home_team": "Argentina",
        "away_team": "France",
        "neutral": True,
        "iterations": 3
    }
    response = client.post("/prediction", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "home_win_chance" in data
    assert "draw_chance" in data
    assert "away_win_chance" in data
    assert "top_results" in data

    # Probabilities should sum to approximately 1.0
    total_prob = data["home_win_chance"] + data["draw_chance"] + data["away_win_chance"]
    assert 0.95 <= total_prob <= 1.05

    # Should return requested iterations
    assert len(data["top_results"]) == 3


def test_prediction_endpoint_historical_date(client: TestClient) -> None:
    """Verify that POST /prediction supports point-in-time historical dates."""
    payload = {
        "home_team": "Germany",
        "away_team": "Argentina",
        "neutral": True,
        "iterations": 2,
        "date": "2014-07-13"
    }
    response = client.post("/prediction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["top_results"]) == 2


def test_prediction_endpoint_identical_teams(client: TestClient) -> None:
    """Verify that POST /prediction rejects identical home and away selections with HTTP 400."""
    payload = {
        "home_team": "Brazil",
        "away_team": "Brazil",
        "neutral": True
    }
    response = client.post("/prediction", json=payload)
    assert response.status_code == 400
    assert "cannot be the same" in response.json()["detail"]


def test_prediction_endpoint_invalid_team(client: TestClient) -> None:
    """Verify that POST /prediction rejects unrecorded team names with HTTP 400."""
    payload = {
        "home_team": "Atlantis FC",
        "away_team": "France",
        "neutral": True
    }
    response = client.post("/prediction", json=payload)
    assert response.status_code == 400
