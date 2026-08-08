import time

import pytest
from fastapi.testclient import TestClient

from service.main import SESSION_TOKEN, app

client = TestClient(app)

AUTH = {"X-Session-Token": SESSION_TOKEN}


def _csv(n=25):
    return "date,value\n" + "\n".join(f"2023-01-{i+1:02d},{100+i*2}" for i in range(n))


def _await_job(job_id, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/job/{job_id}")
        assert resp.status_code == 200
        status = resp.json()
        if status["state"] != "running":
            return status
        time.sleep(0.05)
    pytest.fail("job did not finish in time")


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "token" in data


def test_capabilities_endpoint():
    data = client.get("/api/capabilities").json()
    assert "ai_edition" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) == 2


def test_forecast_job_lifecycle():
    started = client.post("/api/forecast", json={"data": _csv(), "horizon": 5}, headers=AUTH)
    assert started.status_code == 200
    job_id = started.json()["job_id"]

    status = _await_job(job_id)
    assert status["state"] == "done", status.get("error")

    report = status["report"]
    assert "winner" in report
    assert "verdict" in report
    assert "ranking" in report
    assert len(report["forecast"]) == 5
    assert "capabilities" in report


def test_job_progress_is_reported():
    """Progress must come from real work, not a timer on the client."""
    started = client.post("/api/forecast", json={"data": _csv(60), "horizon": 5}, headers=AUTH)
    job_id = started.json()["job_id"]

    seen = []
    deadline = time.time() + 90
    while time.time() < deadline:
        status = client.get(f"/api/job/{job_id}").json()
        seen.append(status["pct"])
        if status["state"] != "running":
            break
        time.sleep(0.02)

    assert seen, "no progress samples collected"
    assert max(seen) >= 100.0
    assert seen == sorted(seen), "progress went backwards"


def test_bad_data_surfaces_a_readable_error():
    started = client.post("/api/forecast", json={"data": "1\n2\n3", "horizon": 5}, headers=AUTH)
    status = _await_job(started.json()["job_id"])
    assert status["state"] == "error"
    assert "at least 20 data points" in status["error"]


def test_sync_endpoint_still_works_for_cli():
    resp = client.post("/api/forecast/sync", json={"data": _csv(), "horizon": 5}, headers=AUTH)
    assert resp.status_code == 200
    assert "winner" in resp.json()


def test_samples_are_actually_served():
    """These 404'd before: /samples was shadowed by the catch-all UI mount."""
    for name in (
        "monthly_revenue",
        "daily_rx_count",
        "weekly_inventory_units",
        "active_patients",
    ):
        resp = client.get(f"/samples/{name}.csv")
        assert resp.status_code == 200, f"/samples/{name}.csv is not reachable"
        assert resp.text.startswith("date,value")


def test_unknown_job_is_a_clean_404():
    assert client.get("/api/job/does-not-exist").status_code == 404
