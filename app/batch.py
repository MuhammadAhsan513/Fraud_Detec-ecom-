"""Batch CSV checking for POST /analyze-batch.

    upload -> file checks -> header check -> row validation -> feature function -> model -> report rows

* File checks (check_upload): .csv name, content type, really text, size, UTF-8, not empty / header-only.
* Header check (read_header, first pass over the file): header names are stripped + lower-cased; duplicate
  names and absent REQUIRED_COLUMNS reject the whole file. The same pass counts rows (MAX_UPLOAD_ROWS) and
  order_ids (duplicate_order_id flag).
* Row validation (check_row): each row goes through the SAME OrderRequest as /analyze-order. A blank
  REQUIRED_FIELDS cell -> "missing_data"; any other schema failure -> "invalid_data". A bad row never stops
  the others.
* Scoring (process, second pass): valid rows are scored BATCH_CHUNK_SIZE at a time by
  ModelService.score_batch (validate_orders -> score_orders -> explain), the same code as /analyze-order.

Nothing here touches the disk: finished rows go to a `sink` callable (app.reports writes them).
Row contents are never logged.
"""
import csv
import hashlib
import heapq
import io
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from pydantic import ValidationError

from .model_service import ModelService
from .schemas import ALLOWED, FLOAT_FIELDS, INT_FIELDS, REQUIRED_COLUMNS, REQUIRED_FIELDS, OrderRequest

STATUSES = ["normal", "suspicious", "missing_data", "invalid_data"]
# Columns a report appends. They are recomputed, so they are ignored when a downloaded report is re-uploaded.
REPORT_COLUMNS = ["row_number", "anomaly_score", "status", "reasons", "missing_fields", "validation_errors",
                  "duplicate_order_id"]
SCHEMA_COLUMNS = list(OrderRequest.model_fields)
# Browsers and curl label CSV files in many ways; anything else (images, PDFs, JSON...) is refused.
ALLOWED_CONTENT_TYPES = {"", "text/csv", "application/csv", "text/x-csv", "application/x-csv", "text/plain",
                         "text/comma-separated-values", "application/vnd.ms-excel", "application/octet-stream"}
LIST_SEP = "; "                     # joins reasons / missing fields / errors inside one CSV cell
SNIFF_CHARS = 65536                 # how much of the text is checked for binary control characters
HIST_MIN, HIST_MAX, HIST_BINS = 0.30, 0.80, 20      # score histogram (documented score range ~0.3-0.8)
TOP_N, MAX_PROBLEM_ROWS = 20, 500
BREAKDOWNS = ["category", "customer_country", "sales_channel"]
_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?|[+-]?(nan|inf|infinity)", re.ASCII | re.IGNORECASE)


class ApiError(Exception):
    """A client error with a plain-language message. Mapped to the standard JSON error envelope."""

    def __init__(self, status: int, code: str, message: str, details: list[dict] | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details or []


class UploadRejected(ApiError):
    """The whole uploaded file is refused; nothing is stored."""


# One real row (orders-sheet.csv row 0) for GET /template.csv, in the CSV's own column order.
TEMPLATE_EXAMPLE = {"order_id": "ORD0020474", "order_timestamp": "2025-09-12 02:16:00", "customer_id": "C009777",
                    "customer_country": "France", "customer_age": "52", "account_age_days": "806",
                    "product_id": "P01005", "product_name": "HOM Product 1005", "category": "Home", "quantity": "1",
                    "unit_price": "22.74", "discount_pct": "20.0", "shipping_cost": "5.37", "tax_amount": "0.25",
                    "platform_fee": "2.79", "order_value": "23.81", "payment_method": "PayPal",
                    "sales_channel": "Mobile App", "device_type": "Android", "ip_country": "Italy",
                    "order_status": "Shipped", "coupon_used": "No"}


# ---------------------------------------------------------------- file checks
def clean_filename(name: str | None) -> str:
    """Base name only, no control characters, at most 200 characters (safe to log and to show)."""
    base = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    base = "".join(ch for ch in base if unicodedata.category(ch)[0] != "C").strip()
    return base[:200] or "upload.csv"


def human_bytes(n: int) -> str:
    return f"{n / 1024 / 1024:.0f} MB" if n >= 1024 * 1024 else f"{n / 1024:.0f} KB"


def check_upload(filename: str, content_type: str | None, data: bytes, max_bytes: int) -> str:
    """Return the decoded text of an uploaded CSV, or raise UploadRejected."""
    if not filename.lower().endswith(".csv"):
        raise UploadRejected(415, "unsupported_file_type",
                             "Only .csv files can be checked. Save the file as CSV (comma separated) and try again.")
    media = (content_type or "").split(";", 1)[0].strip().lower()
    if media not in ALLOWED_CONTENT_TYPES:
        raise UploadRejected(415, "unsupported_file_type",
                             f"The file was sent as '{media[:60]}', not as a CSV file. Upload a .csv file.")
    if len(data) > max_bytes:
        raise UploadRejected(413, "payload_too_large", f"The file is larger than the {human_bytes(max_bytes)} limit.")
    if b"\x00" in data:
        raise UploadRejected(415, "not_text", "This file doesn't look like a CSV text file. It may be a spreadsheet "
                                              "(.xlsx) or another kind of file renamed to .csv.")
    try:
        text = data.decode("utf-8-sig")               # Excel's "CSV UTF-8" adds a BOM
    except UnicodeDecodeError:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise UploadRejected(422, "bad_encoding", "The file isn't UTF-8 text. In Excel, save it as "
                                                      "'CSV UTF-8 (Comma delimited)' and try again.") from None
    if any(unicodedata.category(ch) == "Cc" and ch not in "\t\r\n\f" for ch in text[:SNIFF_CHARS]):
        raise UploadRejected(415, "not_text", "This file doesn't look like a CSV text file. It may be a spreadsheet "
                                              "(.xlsx) or another kind of file renamed to .csv.")
    if not text.strip():
        raise UploadRejected(422, "empty_file", "The file is empty.")
    return text


# ---------------------------------------------------------------- header check (first pass)
def _records(text: str) -> Iterator[tuple[int, list[str]]]:
    """(line number, cells) for each non-blank CSV record. Quoted commas and newlines are handled by csv."""
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        for cells in reader:
            if any(cell.strip() for cell in cells):
                yield reader.line_num, cells
    except csv.Error as e:
        raise UploadRejected(422, "bad_csv", f"The file couldn't be read as CSV near line {reader.line_num}: {e}.") from None


@dataclass
class Table:
    """What the first pass learned about the file. No row contents are kept."""
    filename: str
    text: str
    sha256: str
    size_bytes: int
    header: list[str]                       # normalised names; blank names become unnamed_<n>
    col_index: dict[str, int]               # schema field -> column position
    keep: list[int]                         # positions written back to the report (all but old report columns)
    unknown_columns: list[str]
    ignored_columns: list[str]
    n_rows: int
    order_id_counts: Counter


def read_header(filename: str, text: str, data: bytes, max_rows: int) -> Table:
    """Check the header and count rows. Raises UploadRejected for a file-level problem."""
    records = _records(text)
    first = next(records, None)
    if first is None:
        raise UploadRejected(422, "empty_file", "The file is empty.")
    header = [h.strip().lower() or f"unnamed_{i + 1}" for i, h in enumerate(first[1])]
    duplicated = sorted(name for name, n in Counter(header).items() if n > 1)
    if duplicated:
        raise UploadRejected(422, "duplicate_columns",
                             f"Some column names appear more than once: {', '.join(duplicated)}. "
                             "Each column name must be unique.",
                             [{"field": name, "reason": "column name appears more than once"} for name in duplicated])
    missing = [col for col in REQUIRED_COLUMNS if col not in header]
    if missing:
        raise UploadRejected(422, "missing_columns",
                             f"The file is missing required columns: {', '.join(missing)}. "
                             "Download the template to see every expected column.",
                             [{"field": col, "reason": "required column is missing"} for col in missing])
    col_index = {name: i for i, name in enumerate(header) if name in SCHEMA_COLUMNS}
    oid = col_index["order_id"]
    counts: Counter = Counter()
    n_rows = 0
    for _, cells in records:
        n_rows += 1
        if n_rows > max_rows:
            raise UploadRejected(413, "too_many_rows",
                                 f"The file has more than {max_rows:,} orders. "
                                 f"Split it into files of at most {max_rows:,} orders.")
        order_id = cells[oid].strip() if oid < len(cells) else ""
        if order_id:
            counts[order_id] += 1
    if n_rows == 0:
        raise UploadRejected(422, "header_only", "The file has column names but no orders.")
    return Table(filename=filename, text=text, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
                 header=header, col_index=col_index,
                 keep=[i for i, name in enumerate(header) if name not in REPORT_COLUMNS],
                 unknown_columns=[n for n in header if n not in SCHEMA_COLUMNS and n not in REPORT_COLUMNS],
                 ignored_columns=[n for n in header if n in REPORT_COLUMNS],
                 n_rows=n_rows, order_id_counts=counts)


# ---------------------------------------------------------------- row validation
def to_value(name: str, text: str):
    """CSV cell -> the Python value /analyze-order would receive in JSON. Blank -> None.

    Numbers are parsed only when they look like numbers (ASCII digits, optional sign/decimals/exponent), so a
    cell like '12abc' stays text and the schema reports it as 'must be a number'. Whole-number fields accept
    '52.0' (how pandas writes an int column with blanks) but not '1.5'.
    """
    if text == "":
        return None
    if name not in INT_FIELDS and name not in FLOAT_FIELDS:
        return text
    if not _NUMBER.fullmatch(text):
        return text
    if name in FLOAT_FIELDS:
        return float(text)
    try:
        return int(text)                                     # '806'
    except ValueError:
        number = float(text)                                 # '52.0' -> 52; '1.5', 'nan' stay text
        return int(number) if math.isfinite(number) and number.is_integer() else text


def _num(value) -> str:
    return f"{int(value):,}" if float(value).is_integer() else f"{value:,}"


def plain_error(err: dict) -> dict:
    """One Pydantic error -> {"field", "reason"} in plain language (no values echoed)."""
    loc = err.get("loc") or ("row",)
    name, kind, ctx = str(loc[0]), err.get("type", ""), err.get("ctx") or {}
    if kind in ("int_type", "int_parsing", "int_from_float"):
        reason = "must be a whole number"
    elif kind in ("float_type", "float_parsing"):
        reason = "must be a number"
    elif kind == "finite_number":
        reason = "must be a real number (not NaN or infinity)"
    elif kind == "greater_than_equal":
        reason = f"must be at least {_num(ctx['ge'])}"
    elif kind == "greater_than":
        reason = f"must be more than {_num(ctx['gt'])}"
    elif kind == "less_than_equal":
        reason = f"must be at most {_num(ctx['le'])}"
    elif kind == "less_than":
        reason = f"must be less than {_num(ctx['lt'])}"
    elif kind == "string_pattern_mismatch":
        reason = "may only contain letters, numbers, dashes (-) and underscores (_)"
    elif kind == "string_too_long":
        reason = f"is too long (at most {ctx.get('max_length')} characters)"
    elif kind == "value_error" and name in ALLOWED:
        reason = "is not one of the accepted values: " + ", ".join(ALLOWED[name])
    elif kind == "value_error":
        reason = str(err.get("msg", "is not valid")).removeprefix("Value error, ")
    else:
        reason = str(err.get("msg", "is not valid"))
    return {"field": name, "reason": reason.replace(";", ",")}


@dataclass
class RowResult:
    row_number: int
    cells: list[str]                         # stripped cells, padded to the header width
    order_id: str = ""
    status: str = ""
    score: float | None = None
    reasons: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    duplicate: bool = False
    groups: dict[str, str] = field(default_factory=dict)    # canonical category / country / channel


def check_row(row: RowResult, table: Table) -> dict | None:
    """Validate one row with OrderRequest. Returns the model inputs, or None (status set to missing/invalid)."""
    payload = {}
    for name in SCHEMA_COLUMNS:
        i = table.col_index.get(name)
        payload[name] = to_value(name, row.cells[i]) if i is not None else None
    row.missing = [name for name in REQUIRED_FIELDS if payload[name] is None]
    try:
        order = OrderRequest.model_validate(payload)
    except ValidationError as e:
        row.errors += [plain_error(err) for err in e.errors(include_url=False, include_input=False)
                       if str((err.get("loc") or ("row",))[0]) not in row.missing]
        order = None
    if row.missing:
        row.status = "missing_data"
    elif row.errors or order is None:
        row.status = "invalid_data"
    if row.status:
        return None
    row.groups = {"category": order.category, "customer_country": order.customer_country,
                  "sales_channel": order.sales_channel}
    return order.model_inputs()


# ---------------------------------------------------------------- summary
@dataclass
class Stats:
    """Everything the summary and the HTML report need, collected while rows stream past."""
    threshold: float
    counts: Counter = field(default_factory=Counter)
    hist_normal: list[int] = field(default_factory=lambda: [0] * HIST_BINS)
    hist_suspicious: list[int] = field(default_factory=lambda: [0] * HIST_BINS)
    top: list = field(default_factory=list)                 # min-heap of the TOP_N highest suspicious scores
    groups: dict = field(default_factory=lambda: {dim: {} for dim in BREAKDOWNS})
    problems: list[dict] = field(default_factory=list)
    duplicate_rows: int = 0

    def add(self, row: RowResult) -> None:
        self.counts[row.status] += 1
        self.duplicate_rows += row.duplicate
        order_id = row.order_id
        if row.score is None:
            if len(self.problems) < MAX_PROBLEM_ROWS:
                self.problems.append({"row_number": row.row_number, "order_id": order_id, "status": row.status,
                                      "missing_fields": row.missing, "validation_errors": row.errors})
            return
        flagged = row.status == "suspicious"
        b = min(HIST_BINS - 1, max(0, int((row.score - HIST_MIN) / (HIST_MAX - HIST_MIN) * HIST_BINS)))
        (self.hist_suspicious if flagged else self.hist_normal)[b] += 1
        for dim, value in row.groups.items():
            cell = self.groups[dim].setdefault(value, [0, 0])
            cell[0] += 1
            cell[1] += flagged
        if flagged:
            item = (row.score, -row.row_number, {"row_number": row.row_number, "order_id": order_id,
                                                 "anomaly_score": row.score, "reasons": row.reasons, **row.groups})
            if len(self.top) < TOP_N:
                heapq.heappush(self.top, item)
            elif item[:2] > self.top[0][:2]:
                heapq.heapreplace(self.top, item)

    def summary(self, total: int) -> dict:
        scored = self.counts["normal"] + self.counts["suspicious"]
        return {"total_rows": total, **{s: self.counts[s] for s in STATUSES}, "scored_rows": scored,
                "suspicious_pct": round(100 * self.counts["suspicious"] / scored, 2) if scored else 0.0,
                "duplicate_order_id_rows": self.duplicate_rows}

    def details(self) -> dict:
        width = (HIST_MAX - HIST_MIN) / HIST_BINS
        return {
            "histogram": {"bin_edges": [round(HIST_MIN + i * width, 4) for i in range(HIST_BINS + 1)],
                          "normal": self.hist_normal, "suspicious": self.hist_suspicious},
            "top_suspicious": [item[2] for item in sorted(self.top, reverse=True)],
            "breakdowns": {dim: sorted(({"value": v, "checked": c[0], "suspicious": c[1],
                                         "suspicious_pct": round(100 * c[1] / c[0], 2)} for v, c in values.items()),
                                       key=lambda g: (-g["suspicious"], g["value"]))
                           for dim, values in self.groups.items()},
            "problem_rows": self.problems,
        }


# ---------------------------------------------------------------- second pass: validate + score in chunks
def process(table: Table, service: ModelService, chunk_size: int, sink: Callable[[RowResult], None]) -> Stats:
    """Validate and score every row; each finished row goes to `sink` in file order."""
    stats = Stats(threshold=float(service._require()["threshold"]))
    width = len(table.header)
    records = _records(table.text)
    next(records)                                            # header (already checked)
    oid = table.col_index["order_id"]
    chunk: list[RowResult] = []

    def flush() -> None:
        to_score = [(row, inputs) for row in chunk if (inputs := check_row(row, table)) is not None]
        if to_score:
            for (row, _), result in zip(to_score, service.score_batch([inputs for _, inputs in to_score])):
                if "problems" in result:                     # Task 1's validate_orders refused it
                    row.status = "invalid_data"
                    row.errors = [{"field": p.split(" ", 1)[0], "reason": p.replace(";", ",")}
                                  for p in result["problems"]]
                else:
                    row.status, row.score, row.reasons = result["status"], result["anomaly_score"], result["reasons"]
        for row in chunk:
            stats.add(row)
            sink(row)
        chunk.clear()

    for number, (_, cells) in enumerate(records, start=1):
        cells = [c.strip() for c in cells]
        extra = cells[width:]
        cells = cells[:width] + [""] * (width - len(cells))
        row = RowResult(row_number=number, cells=cells, order_id=cells[oid],
                        duplicate=table.order_id_counts.get(cells[oid], 0) > 1)
        if any(extra):
            row.errors.append({"field": "row", "reason": f"has {width + len(extra)} values but the header has "
                                                         f"{width} columns"})
        chunk.append(row)
        if len(chunk) >= chunk_size:
            flush()
    flush()
    return stats


def report_cells(row: RowResult, table: Table) -> list:
    """One report row: the original columns, then REPORT_COLUMNS."""
    return [row.cells[i] for i in table.keep] + [
        row.row_number,
        repr(row.score) if row.score is not None else "",
        row.status,
        LIST_SEP.join(row.reasons),
        LIST_SEP.join(row.missing),
        LIST_SEP.join(f"{e['field']}: {e['reason']}" for e in row.errors),
        "yes" if row.duplicate else "no",
    ]
