"""App factory: lifespan (load model once), request middleware and the uniform error handlers.

Run:  uvicorn app.main:app --host $HOST --port $PORT

Single orders: /analyze-order. Whole CSV files: /analyze-batch + /reports/... (see app/batch.py, app/reports.py).

Every error response has the shape
    {"error": {"code": "...", "message": "...", "request_id": "...", "details": [...]}}
"""
import asyncio
import json
import re
import time
import traceback
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .batch import ApiError, human_bytes
from .config import ConfigError, Settings
from .logging_config import get_logger, setup_logging
from .model_service import ModelService, ModelUnavailableError, OrderRejectedError
from .reports import ReportNotFound, ReportStore
from .routes import STATIC_DIR, router

log = get_logger()
access_log = get_logger("access")

REQUEST_ID_HEADER = "x-request-id"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# Routes that require a Content-Type (anything else -> 415).
BODY_MEDIA_TYPES = {("POST", "/analyze-order"): "application/json",
                    ("POST", "/analyze-batch"): "multipart/form-data"}
UPLOAD_ROUTES = {("POST", "/analyze-batch")}        # body limit = MAX_UPLOAD_BYTES (+ multipart overhead)
MULTIPART_OVERHEAD = 64 * 1024
HTTP_CODES = {400: "bad_request", 404: "not_found", 405: "method_not_allowed", 413: "payload_too_large",
              415: "unsupported_media_type", 422: "validation_error", 500: "internal_error",
              503: "model_unavailable"}


# ---------------------------------------------------------------- error envelope
def error_payload(status: int, message: str, request_id: str, details: list | None = None,
                  code: str | None = None) -> dict:
    return {"error": {"code": code or HTTP_CODES.get(status, "error"), "message": message,
                      "request_id": request_id, "details": details or []}}


def error_response(request: Request, status: int, message: str, details: list | None = None,
                   code: str | None = None, headers: dict | None = None) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "") or uuid.uuid4().hex
    return JSONResponse(error_payload(status, message, request_id, details, code), status_code=status,
                        headers=headers)


def _clean_reason(msg: str) -> str:
    return msg.removeprefix("Value error, ")


async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    if any(e.get("type") == "json_invalid" for e in errors):
        return error_response(request, 400, "Request body is not valid JSON.", code="malformed_json")
    if any(tuple(e.get("loc", ())) == ("body",) and e.get("type") == "missing" for e in errors):
        return error_response(request, 400, "Request body is empty.", code="empty_body")
    details = [{"field": ".".join(str(p) for p in e.get("loc", ())[1:]) or "body",
                "reason": _clean_reason(e.get("msg", "invalid"))} for e in errors]
    return error_response(request, 422, "Request validation failed.", details)


async def on_order_rejected(request: Request, exc: OrderRejectedError) -> JSONResponse:
    details = [{"field": r.split(" ", 1)[0], "reason": r} for r in exc.reasons]
    return error_response(request, 422, "Order failed model input validation.", details)


async def on_model_unavailable(request: Request, exc: ModelUnavailableError) -> JSONResponse:
    return error_response(request, 503, "Model is not available. Try again later.")


async def on_api_error(request: Request, exc: ApiError) -> JSONResponse:
    if exc.status in (400, 413, 415, 422):              # file name only; never row contents
        log.info("request rejected code=%s path=%s filename=%r request_id=%s", exc.code, request.url.path,
                 getattr(request.state, "upload_filename", None), getattr(request.state, "request_id", ""))
    return error_response(request, exc.status, exc.message, exc.details, code=exc.code)


async def on_report_not_found(request: Request, exc: ReportNotFound) -> JSONResponse:
    return error_response(request, 404, "Report not found. It may have expired, or the link is wrong.")


async def on_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    messages = {404: "Route not found.", 405: "Method not allowed for this route."}
    message = messages.get(exc.status_code) or (exc.detail if isinstance(exc.detail, str) else "HTTP error.")
    return error_response(request, exc.status_code, message, headers=getattr(exc, "headers", None))


# ---------------------------------------------------------------- middleware
class RequestContextMiddleware:
    """Pure ASGI middleware (outermost app layer).

    * request_id: accepts a valid X-Request-ID, else generates one; echoed in the response header.
    * 415 for JSON routes without Content-Type application/json; 413 for bodies > max_body_bytes
      (checks Content-Length and the actual bytes received).
    * catches any unhandled exception -> logs the traceback, returns a generic 500.
    * logs one line per request: method, path, status, latency_ms, request_id.
    """

    def __init__(self, app, max_body_bytes: int, max_upload_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.max_upload_bytes = max_upload_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start = time.perf_counter()
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        incoming = headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if REQUEST_ID_PATTERN.fullmatch(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        method, path = scope["method"], scope["path"]
        status_holder = {"status": 500, "started": False}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder.update(status=message["status"], started=True)
                message.setdefault("headers", [])
                message["headers"] = [(k, v) for k, v in message["headers"] if k.lower() != b"x-request-id"]
                message["headers"].append((b"x-request-id", request_id.encode()))
            await send(message)

        async def send_error(status: int, message: str) -> None:
            body = json.dumps(error_payload(status, message, request_id), separators=(",", ":")).encode()
            await send_wrapper({"type": "http.response.start", "status": status,
                                "headers": [(b"content-type", b"application/json"),
                                            (b"content-length", str(len(body)).encode())]})
            await send_wrapper({"type": "http.response.body", "body": body})

        try:
            required_media = BODY_MEDIA_TYPES.get((method, path))
            if required_media:
                media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if media_type != required_media:
                    await send_error(415, f"Content-Type must be {required_media}.")
                    return
            is_upload = (method, path) in UPLOAD_ROUTES
            limit = self.max_upload_bytes + MULTIPART_OVERHEAD if is_upload else self.max_body_bytes
            too_large = (f"The file is larger than the {human_bytes(self.max_upload_bytes)} limit." if is_upload
                         else f"Request body exceeds {self.max_body_bytes} bytes.")
            declared = headers.get("content-length")
            if declared is not None and (not declared.isdigit() or int(declared) > limit):
                if declared.isdigit():
                    await send_error(413, too_large)
                else:
                    await send_error(400, "Invalid Content-Length header.")
                return
            # Read the body (bounded) up front so chunked uploads are capped too, then replay it.
            chunks, size, more = [], 0, True
            while more:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                size += len(chunk)
                if size > limit:
                    await send_error(413, too_large)
                    return
                chunks.append(chunk)
                more = message.get("more_body", False)
            body, replayed = b"".join(chunks), False

            async def body_receive():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, body_receive, send_wrapper)
        except Exception:
            log.error("unhandled error request_id=%s method=%s path=%s\n%s",
                      request_id, method, path, traceback.format_exc())
            if not status_holder["started"]:
                await send_error(500, "Internal server error.")
            else:
                status_holder["status"] = 500
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            access_log.info("method=%s path=%s status=%s latency_ms=%.2f request_id=%s",
                            method, path, status_holder["status"], latency_ms, request_id)


# ---------------------------------------------------------------- report cleanup
async def _purge_periodically(store: ReportStore, ttl_minutes: int) -> None:
    """Delete expired reports every few minutes (at most every TTL, at least every 5 minutes)."""
    interval = max(30, min(300, ttl_minutes * 60))
    while True:
        await asyncio.sleep(interval)
        try:
            await run_in_threadpool(store.purge)
        except Exception:
            log.error("report cleanup failed\n%s", traceback.format_exc())


# ---------------------------------------------------------------- app factory
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level)
        if not settings.model_path:
            raise ConfigError("MODEL_PATH is not set. Point it at the Task 1 model bundle, e.g. MODEL_PATH=model.joblib")
        service = ModelService.load(settings.model_path)
        if not service.loaded:
            log.error("model failed to load path=%s reason=%s", settings.model_path, service.error)
            if settings.app_env == "production":
                raise RuntimeError(f"cannot load model from MODEL_PATH={settings.model_path}: {service.error}")
            log.warning("starting in degraded mode (APP_ENV=%s): /model-info and /analyze-order return 503",
                        settings.app_env)
        else:
            for w in service.warnings:
                log.warning("version mismatch: %s", w)
            log.info("model loaded path=%s version=%s feature_count=%d threshold=%.6f env=%s",
                     settings.model_path, service.bundle["model_version"], len(service.bundle["feature_order"]),
                     service.bundle["threshold"], settings.app_env)
        app.state.model_service = service
        store = app.state.report_store
        try:
            await run_in_threadpool(store.init)         # create REPORTS_DIR, delete leftovers + expired reports
        except OSError as e:
            raise ConfigError(f"REPORTS_DIR={settings.reports_dir} is not a writable folder: {e}") from None
        cleaner = asyncio.create_task(_purge_periodically(store, settings.report_ttl_minutes))
        try:
            yield
        finally:
            cleaner.cancel()

    app = FastAPI(title="Order anomaly detection API", version="1.0.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.model_service = ModelService(model_path=settings.model_path, error="not loaded yet")
    app.state.report_store = ReportStore(settings.reports_dir, settings.report_ttl_minutes, settings.max_reports)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")   # web UI assets
    app.add_exception_handler(RequestValidationError, on_validation_error)
    app.add_exception_handler(OrderRejectedError, on_order_rejected)
    app.add_exception_handler(ModelUnavailableError, on_model_unavailable)
    app.add_exception_handler(StarletteHTTPException, on_http_error)
    app.add_exception_handler(ApiError, on_api_error)
    app.add_exception_handler(ReportNotFound, on_report_not_found)
    app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.max_body_bytes,
                       max_upload_bytes=settings.max_upload_bytes)
    return app


app = create_app()
