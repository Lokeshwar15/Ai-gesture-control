from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "lstm_loaded" in data


def test_gestures_list():
    with TestClient(app) as client:
        r = client.get("/gestures")
        assert r.status_code == 200
        assert "gestures" in r.json()
