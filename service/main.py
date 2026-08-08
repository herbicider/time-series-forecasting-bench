import secrets
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.backtest import run_backtest
from core.ingest import IngestError, parse_and_validate
from core.models import manager
from core.report import create_forecast_report, export_csv, export_pdf

# Weights must be redirected to the per-user cache before torch or
# huggingface_hub is imported anywhere.
manager.configure_model_cache()

SESSION_TOKEN = secrets.token_hex(16)

app = FastAPI(title="Time Series Forecasting Bench")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1", "http://localhost", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_token(x_session_token: Optional[str] = Header(None)):
    # Skip token requirement if not passed in single-user native desktop mode or allow matching
    if x_session_token and x_session_token != SESSION_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid session token")
    return True


class ForecastRequest(BaseModel):
    data: str
    horizon: int = 5
    date_col: Optional[str] = None
    value_col: Optional[str] = None


# ---------------------------------------------------------------------------
# Job registry
#
# Backtests take anywhere from two seconds to several minutes depending on the
# series length and whether a foundation model is loaded. They run on a worker
# thread so the UI can poll for genuine progress and offer a Cancel button,
# instead of the 400 ms fake stage animation it used to show.
# ---------------------------------------------------------------------------

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _new_job() -> str:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "state": "running",
            "stage": "Reading your data",
            "pct": 0.0,
            "message": "Reading and checking your data…",
            "report": None,
            "error": None,
            "cancelled": False,
        }
    return job_id


def _update_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(fields)


def _job_cancelled(job_id: str) -> bool:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return bool(job and job["cancelled"])


class JobCancelled(Exception):
    pass


def _run_job(job_id: str, content, filename: Optional[str], horizon: int,
             date_col: Optional[str], value_col: Optional[str]) -> None:
    try:
        _update_job(job_id, stage="Reading your data", pct=2.0,
                    message="Reading and checking your data…")
        ingest_res = parse_and_validate(
            content=content, filename=filename, date_col=date_col, value_col=value_col
        )

        def progress_cb(stage: str, pct: float, message: str) -> None:
            if _job_cancelled(job_id):
                raise JobCancelled()
            # Reserve the first 5% for ingest and the last 10% for the report.
            _update_job(job_id, stage=stage, pct=5.0 + pct * 0.85, message=message)

        backtest_results, boundaries, extra_warnings = run_backtest(
            y=ingest_res.series,
            horizon=horizon,
            seasonal_period=ingest_res.seasonal_period,
            progress_cb=progress_cb,
        )

        if _job_cancelled(job_id):
            raise JobCancelled()

        _update_job(job_id, stage="Calibrating prediction ranges", pct=92.0,
                    message="Calibrating prediction ranges…")

        report = create_forecast_report(
            ingest_res=ingest_res,
            horizon=horizon,
            backtest_results=backtest_results,
            boundaries=boundaries,
            extra_warnings=extra_warnings,
        )
        report["capabilities"] = manager.capability_report()

        _update_job(job_id, state="done", pct=100.0, stage="Done",
                    message="Finished.", report=report)

    except JobCancelled:
        _update_job(job_id, state="cancelled", message="Cancelled.")
    except IngestError as exc:
        _update_job(job_id, state="error", error=str(exc))
    except Exception as exc:  # noqa: BLE001 — surfaced to the user verbatim
        traceback.print_exc()
        _update_job(job_id, state="error",
                    error=f"Something went wrong while forecasting: {exc}")


def _start_job(content, filename, horizon, date_col, value_col) -> Dict[str, str]:
    job_id = _new_job()
    threading.Thread(
        target=_run_job,
        args=(job_id, content, filename, horizon, date_col, value_col),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/api/health")
def health_check():
    return {"status": "ok", "token": SESSION_TOKEN}


@app.get("/api/capabilities")
def capabilities_endpoint():
    """Which forecasters this build can run, and whether weights are present."""
    return manager.capability_report()


@app.post("/api/forecast")
def run_forecast_endpoint(req: ForecastRequest, authenticated: bool = Depends(verify_token)):
    return _start_job(req.data, None, req.horizon, req.date_col, req.value_col)


@app.post("/api/forecast/file")
async def run_forecast_file_endpoint(
    file: UploadFile = File(...),
    horizon: int = Form(5),
    date_col: Optional[str] = Form(None),
    value_col: Optional[str] = Form(None),
    authenticated: bool = Depends(verify_token),
):
    content = await file.read()
    return _start_job(content, file.filename, horizon, date_col, value_col)


@app.get("/api/job/{job_id}")
def job_status_endpoint(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="That forecast run has expired.")
        snapshot = dict(job)

    # Hand the report over once, then drop it so finished jobs stop accumulating.
    if snapshot["state"] in ("done", "error", "cancelled"):
        with _jobs_lock:
            _jobs.pop(job_id, None)
    return snapshot


@app.post("/api/job/{job_id}/cancel")
def cancel_job_endpoint(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="That forecast run has expired.")
        job["cancelled"] = True
    return {"ok": True}


@app.post("/api/forecast/sync")
def run_forecast_sync_endpoint(req: ForecastRequest, authenticated: bool = Depends(verify_token)):
    """Blocking variant, kept for the CLI and for tests."""
    try:
        ingest_res = parse_and_validate(
            content=req.data, date_col=req.date_col, value_col=req.value_col
        )
    except IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    backtest_results, boundaries, extra_warnings = run_backtest(
        y=ingest_res.series,
        horizon=req.horizon,
        seasonal_period=ingest_res.seasonal_period,
    )
    report = create_forecast_report(
        ingest_res=ingest_res,
        horizon=req.horizon,
        backtest_results=backtest_results,
        boundaries=boundaries,
        extra_warnings=extra_warnings,
    )
    report["capabilities"] = manager.capability_report()
    return report


# ---------------------------------------------------------------------------
# Optional foundation-model weights
# ---------------------------------------------------------------------------

_download_job: Dict[str, Any] = {
    "state": "idle", "pct": 0.0, "message": "", "error": None,
}
_download_lock = threading.Lock()


@app.post("/api/models/download")
def start_model_download():
    with _download_lock:
        if _download_job["state"] == "running":
            return dict(_download_job)
        _download_job.update(state="running", pct=0.0,
                             message="Starting download…", error=None)

    def worker():
        def progress_cb(stage: str, pct: float, message: str) -> None:
            with _download_lock:
                _download_job.update(pct=pct, message=message)

        try:
            result = manager.download_weights(progress_cb=progress_cb)
            with _download_lock:
                if result["ok"]:
                    _download_job.update(state="done", pct=100.0,
                                         message="All models are ready.")
                else:
                    failures = "; ".join(f["error"] for f in result["failed"])
                    _download_job.update(state="error", error=failures)
        except Exception as exc:  # noqa: BLE001
            with _download_lock:
                _download_job.update(state="error", error=str(exc))

    threading.Thread(target=worker, daemon=True).start()
    return dict(_download_job)


@app.get("/api/models/download")
def model_download_status():
    with _download_lock:
        return dict(_download_job)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

@app.post("/api/export/csv")
def export_csv_endpoint(report: Dict[str, Any], authenticated: bool = Depends(verify_token)):
    csv_str = export_csv(report)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=forecast_report.csv"},
    )


@app.post("/api/export/pdf")
def export_pdf_endpoint(report: Dict[str, Any], authenticated: bool = Depends(verify_token)):
    pdf_bytes = export_pdf(report)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=forecast_report.pdf"},
    )


# ---------------------------------------------------------------------------
# Static files. Order matters: /samples must be mounted before the catch-all
# UI mount at "/", which previously shadowed it and made every sample chip 404.
# ---------------------------------------------------------------------------

SAMPLES_DIR = ROOT_DIR / "samples"
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")

UI_DIR = ROOT_DIR / "ui"
if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
