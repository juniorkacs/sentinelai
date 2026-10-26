import pytest
from fastapi.testclient import TestClient

from sentinel.api import create_app

OBS = {
    "timestamp": "2026-03-10T13:00:00Z",
    "requests_per_s": 118,
    "cpu_pct": 46,
    "memory_pct": 55,
    "p95_latency_ms": 98,
    "error_rate_pct": 0.5,
    "net_out_mb": 70,
    "failed_logins": 1,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_model_info_reports_metrics_and_disclaimer(client):
    body = client.get("/api/model").json()
    assert body["dataset"]["synthetic"] is True
    assert set(body["models"]) == {"robust_baseline", "isolation_forest"}
    assert "not a confirmed incident" in body["disclaimer"]


def test_series(client):
    s = client.get("/api/series", params={"metric": "cpu_pct"}).json()
    n = len(s["timestamps"])
    assert n == 7 * 1440
    assert len(s["values"]) == len(s["band_low"]) == len(s["flags"]["isolation_forest"]) == n


def test_series_unknown_metric(client):
    r = client.get("/api/series", params={"metric": "nope"})
    assert r.status_code == 404 and "Choose one of" in r.json()["error"]


def test_alerts_have_explanations(client):
    alerts = client.get("/api/alerts", params={"model": "isolation_forest"}).json()["alerts"]
    assert alerts
    assert all(a["explanation"] and a["peak_score"] >= a["threshold"] for a in alerts)


def test_alerts_reject_unknown_model(client):
    assert client.get("/api/alerts", params={"model": "magic"}).status_code == 422


def test_score_typical_and_extreme_observations(client):
    extreme = OBS | {"failed_logins": 75}
    body = client.post("/api/score", json={"model": "robust_baseline", "observations": [OBS, extreme]}).json()
    normal, attack = body["results"]
    assert normal["is_anomalous"] is False
    assert attack["is_anomalous"] is True
    assert attack["top_factors"][0]["feature"] == "failed_logins"


@pytest.mark.parametrize(
    "payload",
    [
        {"observations": []},
        {"observations": [OBS | {"cpu_pct": 150}]},
        {"observations": [OBS | {"unexpected": 1}]},
        {"observations": [{k: v for k, v in OBS.items() if k != "memory_pct"}]},
        {"model": "magic", "observations": [OBS]},
        {"observations": [OBS] * 1001},
    ],
)
def test_score_validation(client, payload):
    assert client.post("/api/score", json=payload).status_code == 422


def test_dashboard_and_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and "SentinelAI" in r.text
    assert "script-src 'self'" in r.headers["content-security-policy"]
