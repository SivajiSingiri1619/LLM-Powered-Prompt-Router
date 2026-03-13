from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_root_serves_chat_ui() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Prompt Router Chat Studio" in response.text
    assert "Route Message" in response.text
