"""Environment-based settings. Every value comes from an environment variable (see .env.example)."""
import os
from dataclasses import dataclass

LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
APP_ENVS = {"development", "test", "production"}


class ConfigError(RuntimeError):
    """Raised when an environment variable is missing or invalid."""


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


def _dir_env(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip() or default
    if "\x00" in raw:
        raise ConfigError(f"{name} is not a valid path")
    return raw


@dataclass(frozen=True)
class Settings:
    model_path: str | None = None       # required at startup; checked in the lifespan
    log_level: str = "INFO"
    app_env: str = "development"         # production: fail fast if the model cannot be loaded
    max_body_bytes: int = 16384
    host: str = "127.0.0.1"
    port: int = 8000
    # batch CSV checking (POST /analyze-batch) and stored reports
    max_upload_bytes: int = 50 * 1024 * 1024      # larger CSV files -> 413
    max_upload_rows: int = 200_000                # more data rows -> 413
    batch_chunk_size: int = 10_000                # rows scored per model call
    reports_dir: str = "reports"                  # one sub-folder per report, named by its UUID4
    report_ttl_minutes: int = 1440                # reports older than this are deleted
    max_reports: int = 50                         # oldest reports are deleted beyond this many

    @classmethod
    def from_env(cls) -> "Settings":
        log_level = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
        if log_level not in LOG_LEVELS:
            raise ConfigError(f"LOG_LEVEL must be one of {sorted(LOG_LEVELS)}, got {log_level!r}")
        app_env = os.environ.get("APP_ENV", "development").strip().lower()
        if app_env not in APP_ENVS:
            raise ConfigError(f"APP_ENV must be one of {sorted(APP_ENVS)}, got {app_env!r}")
        return cls(
            model_path=os.environ.get("MODEL_PATH", "").strip() or None,
            log_level=log_level,
            app_env=app_env,
            max_body_bytes=_int_env("MAX_BODY_BYTES", 16384, 256, 10_000_000),
            host=os.environ.get("HOST", "127.0.0.1").strip() or "127.0.0.1",
            port=_int_env("PORT", 8000, 1, 65535),
            max_upload_bytes=_int_env("MAX_UPLOAD_BYTES", 50 * 1024 * 1024, 1024, 1024 ** 3),
            max_upload_rows=_int_env("MAX_UPLOAD_ROWS", 200_000, 1, 10_000_000),
            batch_chunk_size=_int_env("BATCH_CHUNK_SIZE", 10_000, 1, 1_000_000),
            reports_dir=_dir_env("REPORTS_DIR", "reports"),
            report_ttl_minutes=_int_env("REPORT_TTL_MINUTES", 1440, 1, 525_600),
            max_reports=_int_env("MAX_REPORTS", 50, 1, 10_000),
        )
