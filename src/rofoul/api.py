"""REST API (FastAPI).

    GET  /health            liveness + whether a model is loaded
    GET  /model             metadata of the loaded model
    POST /model/reload      re-read config/model from disk (hot swap after retraining)
    POST /validate          data-quality report for an uploaded CSV
    POST /normalize         normalized daily series (JSON, or CSV with ?format=csv)
    POST /predict           normalized values + forecast + days-to-clean + alert

Upload the CSV as multipart form field `file`, e.g.
    curl -F "file=@plant.csv" http://localhost:8000/predict
Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from . import __version__
from .config import load_config
from .pipeline import DAILY_OUTPUT_COLS, PipelineResult, load_model, prepare_daily, run
from .validation import InputValidationError, load_and_validate


def create_app(cfg: dict | None = None) -> FastAPI:
    app = FastAPI(title="rofoul - RO fouling early warning", version=__version__,
                  description="Upload RO plant operating data (CSV) to get normalized performance, "
                              "a fouling forecast and the estimated days until cleaning is needed.")
    state = {"cfg": cfg or load_config()}
    state["model"] = load_model(state["cfg"])

    async def _read(file: UploadFile) -> bytes:
        data = await file.read()
        limit = state["cfg"]["api"]["max_upload_mb"] * 1024 * 1024
        if len(data) > limit:
            raise HTTPException(413, f"File larger than {state['cfg']['api']['max_upload_mb']} MB")
        if not data.strip():
            raise HTTPException(422, {"ok": False, "errors": [{"code": "empty_file", "message": "Uploaded file is empty"}]})
        return data

    def _invalid(exc: InputValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content=exc.report.to_dict())

    @app.get("/")
    def root():
        return {"service": "rofoul", "version": __version__, "docs": "/docs"}

    @app.get("/health")
    def health():
        m = state["model"]
        return {"status": "ok", "version": __version__, "model_loaded": m is not None,
                "model_version": m.metadata.get("version") if m else None}

    @app.get("/model")
    def model_info():
        m = state["model"]
        if m is None:
            raise HTTPException(404, "No model loaded; predictions fall back to the linear-trend baseline.")
        return m.metadata

    @app.post("/model/reload")
    def reload_model():
        state["cfg"] = load_config()
        state["model"] = load_model(state["cfg"])
        return health()

    @app.post("/validate")
    async def validate(file: UploadFile = File(...)):
        data = await _read(file)
        try:
            _, report = load_and_validate(data, state["cfg"])
        except InputValidationError as exc:
            return _invalid(exc)
        return report.to_dict()

    @app.post("/normalize")
    async def normalize(file: UploadFile = File(...), format: str = Query("json", pattern="^(json|csv)$")):
        data = await _read(file)
        try:
            daily, report, refs = prepare_daily(data, state["cfg"])
        except InputValidationError as exc:
            return _invalid(exc)
        out = daily[[c for c in DAILY_OUTPUT_COLS if c in daily.columns]]
        if format == "csv":
            buf = io.StringIO()
            out.to_csv(buf, index=False, float_format="%.4f")
            return PlainTextResponse(buf.getvalue(), media_type="text/csv")
        res = PipelineResult(out, out.iloc[0:0], [], report, refs)
        d = res.to_dict(include_daily=True, include_forecast=False)
        d.pop("summary")
        return d

    @app.post("/predict")
    async def predict(file: UploadFile = File(...),
                      include_daily: bool = Query(False, description="include the normalized daily series"),
                      include_forecast: bool = Query(True, description="include the day-by-day forecast")):
        data = await _read(file)
        try:
            res = run(data, state["cfg"], model=state["model"])
        except InputValidationError as exc:
            return _invalid(exc)
        return res.to_dict(include_daily=include_daily, include_forecast=include_forecast)

    return app


app = None  # created lazily by `rofoul serve` / uvicorn factory


def get_app() -> FastAPI:  # uvicorn --factory rofoul.api:get_app
    return create_app()
