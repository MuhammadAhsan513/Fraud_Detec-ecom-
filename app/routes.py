"""Endpoints.

Single order: GET /health, GET /model-info, POST /analyze-order.
Whole file:   POST /analyze-batch, GET /reports/{id}/rows, GET /reports/{id}/download, GET /template.csv.
Web UI:       GET / (app/static/index.html with the batch limits filled in).
"""
import csv
import io
import json
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Literal

from detect_anomalies import UID_COLUMNS
from fastapi import APIRouter, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, Response

from . import report_html
from .batch import (STATUSES, TEMPLATE_EXAMPLE, ApiError, UploadRejected, check_upload, clean_filename, process,
                    read_header)
from .logging_config import get_logger
from .model_service import ModelService
from .reports import PAGE_SIZE_MAX, ReportStore, safe_download_name
from .schemas import (OPTIONAL_COLUMNS, REQUIRED_COLUMNS, REQUIRED_FIELDS, AnalyzeResponse, ErrorResponse,
                      HealthResponse, ModelInfoResponse, OrderRequest)

log = get_logger("routes")
router = APIRouter()
STATIC_DIR = Path(__file__).resolve().parent / "static"     # web UI files, also mounted at /static
ERRORS = {code: {"model": ErrorResponse} for code in (400, 404, 413, 415, 422, 500, 503)}
FIRST_PAGE_SIZE = 50
NOSNIFF = {"X-Content-Type-Options": "nosniff"}


def _service(request: Request) -> ModelService:
    return request.app.state.model_service


def _store(request: Request) -> ReportStore:
    return request.app.state.report_store


@router.get("/", include_in_schema=False)
def index(request: Request) -> HTMLResponse:
    """The web UI. The batch limits and required columns come from the server, so the page never disagrees."""
    settings = request.app.state.settings
    config = {"max_upload_bytes": settings.max_upload_bytes, "max_upload_rows": settings.max_upload_rows,
              "required_fields": REQUIRED_FIELDS, "required_columns": REQUIRED_COLUMNS,
              "optional_columns": OPTIONAL_COLUMNS}
    page = (STATIC_DIR / "index.html").read_text().replace("{{BATCH_CONFIG}}", escape(json.dumps(config), quote=True))
    return HTMLResponse(page, headers={**NOSNIFF, "Cache-Control": "no-cache"})


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> dict:
    """Liveness. Never touches the model; only reports whether it was loaded at startup."""
    return {"status": "ok", "model_loaded": _service(request).loaded}


@router.get("/model-info", response_model=ModelInfoResponse, responses=ERRORS)
def model_info(request: Request) -> dict:
    return _service(request).info()


@router.post("/analyze-order", response_model=AnalyzeResponse, responses=ERRORS)
def analyze_order(order: OrderRequest, request: Request) -> dict:
    result = _service(request).score(order.model_inputs())
    log.info("scored order_id=%s anomaly_score=%.6f status=%s request_id=%s",
             order.order_id, result["anomaly_score"], result["status"], request.state.request_id)
    return {"order_id": order.order_id, **result}


# ---------------------------------------------------------------- whole file
@router.post("/analyze-batch", responses=ERRORS)
async def analyze_batch(request: Request) -> dict:
    """Check every row of an uploaded CSV (multipart field "file") and save a report."""
    _service(request)._require()                        # 503 before reading anything
    settings = request.app.state.settings
    try:
        form = await request.form(max_files=1, max_fields=20)
    except Exception:                                   # malformed multipart, too many files...
        raise UploadRejected(400, "bad_upload", "The upload couldn't be read. Send one CSV file as "
                                                "multipart/form-data in a field named 'file'.") from None
    try:
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            raise UploadRejected(422, "no_file", "No file was uploaded. Send the CSV in a form field named 'file'.",
                                 [{"field": "file", "reason": "no file was sent"}])
        filename = clean_filename(upload.filename)
        data = await upload.read(settings.max_upload_bytes + 1)
        content_type = upload.content_type
    finally:
        await form.close()                              # removes the multipart spool file
    request.state.upload_filename = filename
    return await run_in_threadpool(_run_batch, request, filename, content_type, data)


def _run_batch(request: Request, filename: str, content_type: str | None, data: bytes) -> dict:
    """Checks -> header -> rows -> model -> report on disk. Runs in a worker thread."""
    settings, service, store = request.app.state.settings, _service(request), _store(request)
    start = time.perf_counter()
    uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    text = check_upload(filename, content_type, data, settings.max_upload_bytes)
    table = read_header(filename, text, data, settings.max_upload_rows)
    bundle = service._require()
    with store.writer() as (start_rows, commit):
        stats = process(table, service, settings.batch_chunk_size, start_rows(table))
        summary = stats.summary(table.n_rows)
        meta = {"filename": filename, "uploaded_at": uploaded_at, "size_bytes": table.size_bytes,
                "sha256": table.sha256, "model_version": bundle["model_version"],
                "threshold": float(bundle["threshold"]), "summary": summary, "columns": table.header,
                "unknown_columns": table.unknown_columns, "ignored_columns": table.ignored_columns,
                "required_columns": REQUIRED_COLUMNS, "required_fields": REQUIRED_FIELDS,
                "duration_ms": round((time.perf_counter() - start) * 1000, 1), **stats.details()}
        report_id = commit(meta)
    first_page = store.rows(report_id, page_size=FIRST_PAGE_SIZE)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    log.info("batch report_id=%s filename=%r rows=%d normal=%d suspicious=%d missing_data=%d invalid_data=%d "
             "duration_ms=%.0f request_id=%s", report_id, filename, summary["total_rows"], summary["normal"],
             summary["suspicious"], summary["missing_data"], summary["invalid_data"], duration_ms,
             request.state.request_id)
    base = f"/reports/{report_id}"
    return {"report_id": report_id, "filename": filename, "uploaded_at": uploaded_at,
            "model_version": meta["model_version"], "threshold": meta["threshold"], "summary": summary,
            "unknown_columns": table.unknown_columns, "ignored_columns": table.ignored_columns,
            "duration_ms": duration_ms, "rows": first_page,
            "links": {"rows": f"{base}/rows", "csv": f"{base}/download?format=csv",
                      "html": f"{base}/download?format=html", "json": f"{base}/download?format=json"}}


@router.get("/reports/{report_id}/rows", responses=ERRORS)
def report_rows(request: Request, report_id: str, status: str | None = None,
                page: int = Query(1, ge=1, le=1_000_000), page_size: int = Query(FIRST_PAGE_SIZE, ge=1, le=PAGE_SIZE_MAX),
                sort: Literal["score", "row"] = "score") -> dict:
    """One page of a report, filtered by status, highest anomaly_score first (sort=row for file order)."""
    status = (status or "").strip() or None
    if status is not None and status not in STATUSES:
        raise ApiError(422, "validation_error", "Request validation failed.",
                       [{"field": "status", "reason": f"must be one of {STATUSES} (or empty for all rows)"}])
    return _store(request).rows(report_id, status=status, page=page, page_size=page_size, sort=sort)


@router.get("/reports/{report_id}/download", responses=ERRORS)
def report_download(request: Request, report_id: str, format: Literal["csv", "html", "json"] = "csv") -> Response:
    """The full report as CSV (Excel-friendly), a self-contained HTML summary, or JSON (summary + all rows)."""
    store = _store(request)
    meta = store.meta(report_id)
    name = safe_download_name(meta["filename"], format)
    if format == "csv":
        return FileResponse(store.csv_path(report_id), media_type="text/csv; charset=utf-8", filename=name,
                            headers=NOSNIFF)
    if format == "json":
        return FileResponse(store.json_path(report_id), media_type="application/json", filename=name,
                            headers=NOSNIFF)
    return HTMLResponse(report_html.render(meta),
                        headers={**NOSNIFF, "Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/template.csv", responses=ERRORS)
def template_csv() -> Response:
    """Header with every expected column plus one example row (utf-8-sig so Excel opens it cleanly)."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(UID_COLUMNS)
    writer.writerow([TEMPLATE_EXAMPLE[c] for c in UID_COLUMNS])
    return Response(("\ufeff" + out.getvalue()).encode("utf-8"), media_type="text/csv; charset=utf-8",
                    headers={**NOSNIFF, "Content-Disposition": 'attachment; filename="order-template.csv"'})
