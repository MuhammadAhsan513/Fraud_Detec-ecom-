"""Report storage on disk (no database).

    REPORTS_DIR/<report_id>/meta.json     summary, file details, histogram, top suspicious, breakdowns, problem rows
    REPORTS_DIR/<report_id>/report.csv    every row: original columns + REPORT_COLUMNS (utf-8-sig, formula-safe)
    REPORTS_DIR/<report_id>/report.json   written on the first JSON download, then reused

* report_id is a random UUID4. Every request checks it against REPORT_ID_PATTERN before touching the disk, so an
  id can never be a path (no traversal); unknown, malformed and expired ids all answer 404.
* A report is built in a hidden ".tmp-<uuid>" folder and renamed into place only when complete, so a rejected or
  failed upload leaves nothing behind.
* Reports older than REPORT_TTL_MINUTES are deleted at startup, by a periodic check and when accessed; beyond
  MAX_REPORTS the oldest are deleted.
"""
import csv
import json
import math
import re
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .batch import LIST_SEP, REPORT_COLUMNS, RowResult, Table, report_cells
from .logging_config import get_logger

log = get_logger("reports")

REPORT_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
TMP_PREFIX = ".tmp-"
META, ROWS_CSV, ROWS_JSON = "meta.json", "report.csv", "report.json"
DISPLAY_COLUMNS = ["order_id", "product_id", "category", "order_value", "customer_country", "sales_channel"]
PAGE_SIZE_MAX = 200
CACHE_SIZE = 2                       # loaded reports kept in memory for paging
# Spreadsheet formula injection: a text cell starting with one of these is prefixed with an apostrophe.
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_PLAIN_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", re.ASCII)


class ReportNotFound(LookupError):
    """Unknown, malformed or expired report_id. Mapped to 404."""


def neutralize(value):
    """Make one exported cell safe to open in a spreadsheet.

    Text starting with = + - @ (or tab / CR) gets a leading apostrophe so Excel/Sheets show it as text instead of
    running it as a formula. Plain numbers such as -5 or +1.5 are left alone: a spreadsheet reads them as numbers.
    """
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES) and not _PLAIN_NUMBER.fullmatch(value):
        return "'" + value
    return value


def safe_download_name(filename: str, suffix: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "orders"
    return f"{stem[:80]}-report.{suffix}"


def split_list(cell: str) -> list[str]:
    return [part for part in cell.split(LIST_SEP) if part] if cell else []


def split_errors(cell: str) -> list[dict]:
    return [{"field": f, "reason": r} for f, _, r in (p.partition(": ") for p in split_list(cell))]


class ReportStore:
    def __init__(self, root: str | Path, ttl_minutes: int, max_reports: int):
        self.root = Path(root)
        self.ttl_seconds = ttl_minutes * 60
        self.max_reports = max_reports
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    # ------------------------------------------------------------ lifecycle
    def init(self) -> None:
        """Create the folder, check it is writable, remove leftovers and expired reports."""
        self.root.mkdir(parents=True, exist_ok=True)
        probe = self.root / f"{TMP_PREFIX}probe-{uuid.uuid4()}"
        probe.mkdir()
        probe.rmdir()
        for leftover in self.root.glob(f"{TMP_PREFIX}*"):           # from a crash mid-upload
            shutil.rmtree(leftover, ignore_errors=True)
        self.purge()

    def _report_dirs(self) -> list[Path]:
        return [p for p in self.root.iterdir() if p.is_dir() and REPORT_ID_PATTERN.fullmatch(p.name)]

    @staticmethod
    def _created(path: Path) -> float:
        try:
            return float(json.loads((path / META).read_text())["created_at_epoch"])
        except (OSError, ValueError, KeyError, TypeError):
            return path.stat().st_mtime

    def _delete(self, path: Path) -> None:
        self._cache.pop(path.name, None)
        shutil.rmtree(path, ignore_errors=True)

    def purge(self, keep: str | None = None) -> int:
        """Delete expired reports, then the oldest beyond max_reports. Returns how many were deleted."""
        with self._lock:
            now, deleted = time.time(), 0
            dirs = sorted(((self._created(p), p) for p in self._report_dirs()), key=lambda t: t[0])
            alive = []
            for created, path in dirs:
                if now - created > self.ttl_seconds and path.name != keep:
                    self._delete(path)
                    deleted += 1
                else:
                    alive.append(path)
            excess = len(alive) - self.max_reports
            for path in [p for p in alive if p.name != keep][:max(0, excess)]:
                self._delete(path)
                deleted += 1
        if deleted:
            log.info("purged reports count=%d", deleted)
        return deleted

    # ------------------------------------------------------------ writing
    @contextmanager
    def writer(self):
        """Build a report in a temp folder. Yields (add_row, commit). Anything not committed is removed."""
        report_id = str(uuid.uuid4())
        tmp = self.root / f"{TMP_PREFIX}{report_id}"
        tmp.mkdir()
        handle = open(tmp / ROWS_CSV, "w", encoding="utf-8-sig", newline="")
        committed = False
        try:
            rows = csv.writer(handle)
            state: dict = {}

            def start(table: Table) -> Callable[[RowResult], None]:
                state["table"] = table
                rows.writerow([table.header[i] for i in table.keep] + REPORT_COLUMNS)
                return lambda row: rows.writerow([neutralize(c) for c in report_cells(row, table)])

            def commit(meta: dict) -> str:
                nonlocal committed
                handle.close()
                meta = {**meta, "report_id": report_id, "created_at_epoch": time.time()}
                (tmp / META).write_text(json.dumps(meta, ensure_ascii=False))
                tmp.rename(self.root / report_id)
                committed = True
                self.purge(keep=report_id)
                return report_id

            yield start, commit
        finally:
            if not handle.closed:
                handle.close()
            if not committed:
                shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------ reading
    def path(self, report_id: str) -> Path:
        """Folder of a live report, or ReportNotFound. Expired reports are deleted on access."""
        if not isinstance(report_id, str) or not REPORT_ID_PATTERN.fullmatch(report_id):
            raise ReportNotFound(report_id)
        path = self.root / report_id
        if not (path / META).is_file() or not (path / ROWS_CSV).is_file():
            raise ReportNotFound(report_id)
        if time.time() - self._created(path) > self.ttl_seconds:
            with self._lock:
                self._delete(path)
            raise ReportNotFound(report_id)
        return path

    def meta(self, report_id: str) -> dict:
        return json.loads((self.path(report_id) / META).read_text())

    def _frame(self, report_id: str) -> pd.DataFrame:
        """The report's result columns, sorted by score (highest first, unscored last), cached in memory."""
        path = self.path(report_id)
        with self._lock:
            if report_id in self._cache:
                self._cache.move_to_end(report_id)
                return self._cache[report_id]
        with open(path / ROWS_CSV, encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f))
        cols = [c for c in DISPLAY_COLUMNS if c in header] + REPORT_COLUMNS
        df = pd.read_csv(path / ROWS_CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False, usecols=cols)
        df["row_number"] = df["row_number"].astype(int)
        # Python float() round-trips repr() exactly; pd.to_numeric can be off by one ULP.
        df["score"] = df["anomaly_score"].map(lambda v: float(v) if v else np.nan).astype(float)
        order = np.lexsort((df["row_number"].to_numpy(), -df["score"].fillna(-np.inf).to_numpy()))
        df = df.iloc[order].reset_index(drop=True)
        with self._lock:
            self._cache[report_id] = df
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return df

    def rows(self, report_id: str, status: str | None = None, page: int = 1, page_size: int = 50,
             sort: str = "score") -> dict:
        """One page of rows, filtered by status, sorted by anomaly_score (desc) or by row_number."""
        df = self._frame(report_id)
        if status:
            df = df[df["status"] == status]
        if sort == "row":
            df = df.sort_values("row_number")
        total = len(df)
        pages = max(1, math.ceil(total / page_size))
        part = df.iloc[(page - 1) * page_size: page * page_size]
        return {"report_id": report_id, "status": status, "sort": sort, "page": page, "page_size": page_size,
                "total_rows": total, "total_pages": pages,
                "rows": [self._row_json(r) for r in part.to_dict("records")]}

    @staticmethod
    def _row_json(r: dict) -> dict:
        return {"row_number": int(r["row_number"]), "order_id": r.get("order_id", ""),
                "anomaly_score": None if pd.isna(r["score"]) else float(r["score"]),
                "status": r["status"], "reasons": split_list(r["reasons"]),
                "missing_fields": split_list(r["missing_fields"]),
                "validation_errors": split_errors(r["validation_errors"]),
                "duplicate_order_id": r["duplicate_order_id"] == "yes",
                "values": {c: r[c] for c in DISPLAY_COLUMNS if c in r}}

    # ------------------------------------------------------------ downloads
    def csv_path(self, report_id: str) -> Path:
        return self.path(report_id) / ROWS_CSV

    def json_path(self, report_id: str) -> Path:
        """summary + every row as JSON (built once, then reused). Text cells are formula-safe like the CSV."""
        path = self.path(report_id)
        target = path / ROWS_JSON
        if target.is_file():
            return target
        meta = json.loads((path / META).read_text())
        tmp = path / f"{TMP_PREFIX}{ROWS_JSON}"
        with open(path / ROWS_CSV, encoding="utf-8-sig", newline="") as src, open(tmp, "w", encoding="utf-8") as out:
            reader = csv.DictReader(src)
            original = [c for c in reader.fieldnames if c not in REPORT_COLUMNS]
            head = {k: meta[k] for k in ("report_id", "filename", "uploaded_at", "sha256", "model_version",
                                          "threshold", "summary", "unknown_columns", "ignored_columns")}
            out.write(json.dumps(head, ensure_ascii=False)[:-1] + ', "rows": [')
            for n, rec in enumerate(reader):
                score = rec["anomaly_score"]
                row = {"row_number": int(rec["row_number"]), "anomaly_score": float(score) if score else None,
                       "status": rec["status"], "reasons": split_list(rec["reasons"]),
                       "missing_fields": split_list(rec["missing_fields"]),
                       "validation_errors": split_errors(rec["validation_errors"]),
                       "duplicate_order_id": rec["duplicate_order_id"] == "yes",
                       "values": {c: rec[c] for c in original}}       # already neutralized in report.csv
                out.write(("," if n else "") + json.dumps(row, ensure_ascii=False))
            out.write("]}")
        tmp.replace(target)
        return target
