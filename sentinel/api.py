"""HTTP API and dashboard.

Run ``python -m sentinel serve`` (or ``uvicorn sentinel.api:create_app --factory``).
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from sentinel.data import ANOMALY_DESCRIPTIONS, METRICS
from sentinel.engine import Engine
from sentinel.features import FEATURES, LABELS

STATIC = Path(__file__).parent / "static"
ModelName = Literal["robust_baseline", "isolation_forest"]

DISCLAIMER = (
    "An anomaly score says an observation is unusual compared with this service's training period. "
    "It is not a confirmed incident, and it does not say whether anything malicious happened."
)


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: datetime
    requests_per_s: Annotated[float, Field(ge=0, le=1e6)]
    cpu_pct: Annotated[float, Field(ge=0, le=100)]
    memory_pct: Annotated[float, Field(ge=0, le=100)]
    p95_latency_ms: Annotated[float, Field(ge=0, le=1e6)]
    error_rate_pct: Annotated[float, Field(ge=0, le=100)]
    net_out_mb: Annotated[float, Field(ge=0, le=1e6)]
    failed_logins: Annotated[float, Field(ge=0, le=1e6)]


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: ModelName = "robust_baseline"
    observations: Annotated[list[Observation], Field(min_length=1, max_length=1000)]


def create_app(seed: int = 42) -> FastAPI:
    app = FastAPI(title="SentinelAI", version="1.0.0", docs_url="/api/docs", redoc_url=None)
    engine = Engine(seed=seed)
    app.state.engine = engine

    @app.middleware("http")
    async def headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        resp: Response = await call_next(request)
        if not request.url.path.startswith("/api/docs"):
            resp.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                "object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
            )
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        return resp

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/model")
    def model_info() -> dict[str, object]:
        ds = engine.ds
        return {
            "dataset": {
                "seed": engine.seed,
                "synthetic": True,
                "minutes": len(ds.frame),
                "train_start": ds.frame["timestamp"].iloc[0].isoformat(),
                "test_start": ds.test["timestamp"].iloc[0].isoformat(),
                "test_end": ds.test["timestamp"].iloc[-1].isoformat(),
                "metrics": METRICS,
                "features": [{"name": f, "label": LABELS[f]} for f in FEATURES],
                "anomaly_types": ANOMALY_DESCRIPTIONS,
            },
            "models": {name: asdict(run.report) for name, run in engine.runs.items()},
            "disclaimer": DISCLAIMER,
        }

    @app.get("/api/series")
    def series(metric: Annotated[str, Query(max_length=40)] = "cpu_pct") -> dict[str, object]:
        if metric not in FEATURES or metric not in METRICS:
            raise HTTPException(404, f"Unknown metric. Choose one of: {', '.join(METRICS)}.")
        return engine.series(metric)

    @app.get("/api/alerts")
    def alerts(model: ModelName = "robust_baseline") -> dict[str, object]:
        return {"model": model, "disclaimer": DISCLAIMER, "alerts": engine.alert_payload(model)}

    @app.post("/api/score")
    def score(req: ScoreRequest) -> dict[str, object]:
        df = pd.DataFrame([o.model_dump() for o in req.observations])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        det = engine.detectors[req.model]
        scores = det.score(df)
        expl = engine.explain_rows(req.model, df, top=3)
        return {
            "model": req.model,
            "threshold": round(det.threshold, 4),
            "disclaimer": DISCLAIMER,
            "results": [
                {
                    "timestamp": o.timestamp.isoformat(),
                    "score": round(float(s), 4),
                    "is_anomalous": bool(s > det.threshold),
                    "top_factors": e,
                }
                for o, s, e in zip(req.observations, scores, expl, strict=True)
            ],
        }

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app
