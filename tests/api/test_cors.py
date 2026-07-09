from fastapi.testclient import TestClient

from bashgym.api.routes import create_app
from bashgym.config import reload_settings


def test_dev_cors_allows_live_vite_port(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    reload_settings()

    client = TestClient(create_app())
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5174",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type,x-api-key,x-requested-with",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5174"
