/* Order check: the "Whole file" tab.
 *
 * Calls POST /analyze-batch (multipart field "file"), then GET /reports/{id}/rows for the table and links to
 * GET /reports/{id}/download?format=csv|html|json. Reuses helpers from app.js: $, el, callApi, FIELD_BY_NAME,
 * scorePercent, numberFmt, setStatus. All server text goes into the page with textContent.
 */
"use strict";

const STATUS_INFO = {
  normal: { text: "Looks normal", icon: "✓", tone: "ok" },
  suspicious: { text: "Needs a closer look", icon: "!", tone: "flag" },
  missing_data: { text: "Missing data", icon: "?", tone: "bad" },
  invalid_data: { text: "Invalid data", icon: "✕", tone: "bad" },
};
const NULLABLE_COLUMNS = ["discount_pct", "shipping_cost"];     // required columns whose cells may be blank
const TABLE_PAGE_SIZE = 50;

const batch = { config: readBatchConfig(), file: null, busy: false, report: null, status: "", page: 1, pages: 1 };

// ------------------------------------------------------------------ config from the server (GET / fills it in)
function readBatchConfig() {
  const fallback = { max_upload_bytes: 50 * 1024 * 1024, max_upload_rows: 200000, required_fields: [],
                     required_columns: [], optional_columns: [] };
  try {
    return { ...fallback, ...JSON.parse(document.querySelector('meta[name="batch-config"]').content) };
  } catch {
    return fallback;                                      // page opened as /static/index.html: use defaults
  }
}

function formatBytes(n) {
  if (n >= 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  if (n >= 1024) return `${Math.round(n / 1024)} KB`;
  return `${n} bytes`;
}

function columnLabel(name) {
  return FIELD_BY_NAME[name]?.label || name;
}

/** The plain-language list of columns, built from the server's REQUIRED_COLUMNS. */
function renderColumnHelp() {
  const c = batch.config;
  $("batch-limits").textContent =
    `Up to ${numberFmt.format(c.max_upload_rows)} orders and ${formatBytes(c.max_upload_bytes)} per file. ` +
    "Save from Excel as \"CSV UTF-8 (Comma delimited)\".";
  const list = $("required-columns");
  list.replaceChildren();
  for (const name of c.required_columns) {
    const item = el("li");
    item.append(el("code", {}, name), el("span", {}, ` ${columnLabel(name)}`));
    if (NULLABLE_COLUMNS.includes(name)) item.append(el("span", { class: "col-note" }, " (cells may be blank)"));
    list.append(item);
  }
  if (c.optional_columns.length) {
    $("optional-columns").textContent = `Optional columns: ${c.optional_columns.join(", ")}.`;
  }
}

// ------------------------------------------------------------------ choosing a file (browse or drag and drop)
/** Client-side checks before upload: .csv name, not empty, under the size limit. Returns "" if fine. */
function checkFile(file) {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    return "Only .csv files can be checked. In Excel, use File > Save As > CSV UTF-8 (Comma delimited).";
  }
  if (file.size === 0) return "This file is empty.";
  if (file.size > batch.config.max_upload_bytes) {
    return `This file is ${formatBytes(file.size)}. The limit is ${formatBytes(batch.config.max_upload_bytes)}.`;
  }
  return "";
}

function chooseFile(file) {
  const problem = file ? checkFile(file) : "";
  batch.file = file && !problem ? file : null;
  $("batch-file-info").textContent = file ? `${file.name}, ${formatBytes(file.size)}` : "";
  $("batch-file-error").textContent = problem;
  $("dropzone").classList.toggle("has-error", Boolean(problem));
  $("dropzone").classList.toggle("has-file", Boolean(batch.file));
  updateBatchButton();
}

function setupDropzone() {
  const zone = $("dropzone");
  $("batch-file").addEventListener("change", (e) => chooseFile(e.target.files[0] || null));
  for (const type of ["dragenter", "dragover"]) {
    zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.add("is-dragging"); });
  }
  for (const type of ["dragleave", "drop"]) {
    zone.addEventListener(type, () => zone.classList.remove("is-dragging"));
  }
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    const files = e.dataTransfer?.files || [];
    if (files.length > 1) {
      chooseFile(null);
      $("batch-file-error").textContent = "Drop one file at a time.";
      return;
    }
    chooseFile(files[0] || null);
  });
}

function updateBatchButton() {
  $("batch-submit").disabled = batch.busy || !batch.file || !state.modelInfo;
}

// ------------------------------------------------------------------ upload
function setBatchBusy(busy) {
  batch.busy = busy;
  const button = $("batch-submit");
  button.classList.toggle("is-busy", busy);
  button.setAttribute("aria-busy", busy ? "true" : "false");
  button.querySelector(".btn-label").textContent = busy ? "Checking the file…" : "Check this file";
  $("batch-file").disabled = busy;
  $("batch-progress").textContent = busy
    ? `Checking ${batch.file.name}. A file of 150,000 orders takes about 20 seconds. Please keep this page open.`
    : "";
  updateBatchButton();
}

async function onBatchSubmit(event) {
  event.preventDefault();
  if (batch.busy || !batch.file) return;
  const problem = checkFile(batch.file);
  if (problem) { $("batch-file-error").textContent = problem; return; }
  $("batch-error").hidden = true;
  const form = new FormData();
  form.append("file", batch.file, batch.file.name);
  setBatchBusy(true);
  const resp = await callApi("/analyze-batch", { method: "POST", body: form });
  setBatchBusy(false);
  if (!resp.ok || !resp.body) return showBatchError(resp);
  batch.report = resp.body;
  batch.status = "";
  $("status-filter").value = "";
  renderBatchSummary(resp.body);
  renderRows(resp.body.rows);
  $("batch-result").hidden = false;
  $("batch-result").focus();
}

/** A rejected file or a failed request -> one plain message (+ field details), never raw JSON. */
function showBatchError(resp) {
  const box = $("batch-error");
  box.replaceChildren();
  let message;
  if (resp.offline) {
    setStatus(false, "Service unavailable");
    message = "Cannot reach the service. Check that it is running, then try again.";
  } else if (resp.status === 503) {
    message = "The checking service is not ready. Please try again shortly.";
  } else if (resp.status >= 500 || !resp.body?.error) {
    message = "Sorry, something went wrong while checking this file." +
              (resp.requestId ? ` If it keeps happening, give support this reference: ${resp.requestId}.` : "");
  } else {
    message = String(resp.body.error.message || "The file could not be checked.");
  }
  box.append(el("p", { class: "error-title" }, message));
  const details = resp.body?.error?.details || [];
  if (details.length && resp.status < 500) {
    const list = el("ul");
    for (const d of details.slice(0, 30)) {
      const field = String(d.field || "");
      const label = FIELD_BY_NAME[field] ? `${columnLabel(field)} (${field})` : field;
      list.append(el("li", {}, `${label}: ${String(d.reason || "")}`));
    }
    box.append(list);
  }
  box.hidden = false;
  $("batch-result").hidden = true;
  box.focus?.();
}

// ------------------------------------------------------------------ summary
function renderBatchSummary(report) {
  const s = report.summary;
  $("batch-title").textContent = `Results for ${report.filename}`;
  $("batch-meta").textContent =
    `Checked ${new Date(report.uploaded_at).toLocaleString()} in ${(report.duration_ms / 1000).toFixed(1)} seconds. ` +
    `Model ${report.model_version}, alert line ${report.threshold.toFixed(3)}.`;
  const cards = $("stat-cards");
  cards.replaceChildren();
  const items = [["total", "Total rows", s.total_rows, "", "#"], ...Object.entries(STATUS_INFO)
    .map(([key, info]) => [key, info.text, s[key], info.tone, info.icon])];
  for (const [key, label, count, tone, icon] of items) {
    const card = el("li", { class: "stat-card", "data-tone": tone || "plain" });
    card.append(el("span", { class: "stat-icon", "aria-hidden": "true" }, icon),
                el("span", { class: "stat-count" }, numberFmt.format(count)),
                el("span", { class: "stat-label" }, label));
    if (key === "suspicious") {
      card.append(el("span", { class: "stat-extra" }, `${s.suspicious_pct.toFixed(2)}% of checked orders`));
    }
    cards.append(card);
  }
  const notes = $("batch-notes");
  notes.replaceChildren();
  const lines = [];
  if (s.missing_data + s.invalid_data) {
    lines.push(`${numberFmt.format(s.missing_data + s.invalid_data)} rows could not be checked. ` +
               "Choose Missing data or Invalid data below to see why.");
  }
  if (s.duplicate_order_id_rows) {
    lines.push(`${numberFmt.format(s.duplicate_order_id_rows)} rows share their order reference with another row. ` +
               "Each row was checked on its own.");
  }
  if (report.unknown_columns.length) lines.push(`Columns not used: ${report.unknown_columns.join(", ")}.`);
  if (report.ignored_columns.length) lines.push("Result columns from an earlier report were ignored and recalculated.");
  for (const line of lines) notes.append(el("p", {}, line));
  for (const [id, key] of [["dl-csv", "csv"], ["dl-html", "html"], ["dl-json", "json"]]) {
    $(id).href = report.links[key];
  }
}

// ------------------------------------------------------------------ results table
async function loadRows() {
  const params = new URLSearchParams({ page: String(batch.page), page_size: String(TABLE_PAGE_SIZE) });
  if (batch.status) params.set("status", batch.status);
  $("table-count").textContent = "Loading…";
  const resp = await callApi(`/reports/${encodeURIComponent(batch.report.report_id)}/rows?${params}`);
  if (!resp.ok || !resp.body) {
    $("table-count").textContent = resp.status === 404
      ? "This report has expired. Check the file again to see the rows."
      : "The rows couldn't be loaded. Try again.";
    return;
  }
  renderRows(resp.body);
}

function renderRows(page) {
  batch.page = page.page;
  batch.pages = page.total_pages;
  const body = $("results-body");
  body.replaceChildren();
  for (const row of page.rows) body.append(...rowElements(row));
  if (!page.rows.length) {
    const tr = el("tr");
    tr.append(el("td", { colspan: 4, class: "empty-cell" }, "No rows to show."));
    body.append(tr);
  }
  const shown = page.status ? STATUS_INFO[page.status].text.toLowerCase() : "rows";
  $("table-count").textContent = `${numberFmt.format(page.total_rows)} ${page.status ? `rows: ${shown}` : shown}` +
    (page.total_rows ? `, highest score first.` : ".");
  $("page-info").textContent = `Page ${page.page} of ${page.total_pages}`;
  $("page-prev").disabled = page.page <= 1;
  $("page-next").disabled = page.page >= page.total_pages;
}

/** One table row plus its hidden detail row. The order cell is a button so it works with the keyboard. */
function rowElements(row) {
  const info = STATUS_INFO[row.status];
  const detailId = `detail-${row.row_number}`;
  const tr = el("tr", { class: "result-row", "data-tone": info.tone });
  const toggle = el("button", { type: "button", class: "row-toggle", "aria-expanded": "false",
                                "aria-controls": detailId }, row.order_id || "(no order reference)");
  const orderCell = el("td");
  orderCell.append(toggle);
  tr.append(el("td", { class: "num" }, numberFmt.format(row.row_number)), orderCell, scoreCell(row), statusCell(info));

  const detail = el("tr", { class: "detail-row", id: detailId, hidden: true });
  const cell = el("td", { colspan: 4 });
  cell.append(...rowDetails(row));
  detail.append(cell);

  const flip = () => {
    const open = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    detail.hidden = !open;
  };
  toggle.addEventListener("click", (e) => { e.stopPropagation(); flip(); });
  tr.addEventListener("click", flip);                      // the whole row is clickable too
  return [tr, detail];
}

function scoreCell(row) {
  const td = el("td", { class: "score-cell" });
  if (row.anomaly_score === null) {
    td.append(el("span", { class: "muted" }, "Not checked"));
    return td;
  }
  const bar = el("span", { class: "mini-bar", "aria-hidden": "true" });
  const fill = el("span", { class: "mini-fill" });
  fill.style.width = `${scorePercent(row.anomaly_score)}%`;
  const line = el("span", { class: "mini-line" });
  line.style.left = `${scorePercent(batch.report.threshold)}%`;
  bar.append(fill, line);
  td.append(bar, el("span", { class: "score-num" }, row.anomaly_score.toFixed(3)));
  return td;
}

function statusCell(info) {
  const td = el("td");
  const badge = el("span", { class: "badge", "data-tone": info.tone });
  badge.append(el("span", { class: "badge-icon", "aria-hidden": "true" }, info.icon), el("span", {}, info.text));
  td.append(badge);
  return td;
}

/** Plain-language explanation of one row: its reasons, or what is missing / invalid. */
function rowDetails(row) {
  const out = [];
  const list = (items) => {
    const ul = el("ul");
    for (const item of items) ul.append(el("li", {}, item));
    return ul;
  };
  if (row.status === "missing_data") {
    out.push(el("p", { class: "detail-title" }, "This row was not checked because these required values are blank:"),
             list(row.missing_fields.map((f) => `${columnLabel(f)} (${f})`)));
  }
  if (row.validation_errors.length) {
    out.push(el("p", { class: "detail-title" }, row.status === "invalid_data"
                 ? "This row was not checked because some values can't be used:" : "Other problems in this row:"),
             list(row.validation_errors.map((e) => e.field === "row" ? `The row ${e.reason}.`
                                                 : `${columnLabel(e.field)} (${e.field}) ${e.reason}.`)));
  }
  if (row.anomaly_score !== null) {
    out.push(el("p", { class: "detail-title" }, `What stands out (score ${row.anomaly_score.toFixed(3)}, ` +
                                                `alert line ${batch.report.threshold.toFixed(3)}):`),
             list(row.reasons.length ? row.reasons : ["No explanation is available."]));
    const v = row.values || {};
    const facts = [v.category, v.order_value && `order total ${v.order_value}`, v.customer_country, v.sales_channel]
      .filter(Boolean).join(", ");
    if (facts) out.push(el("p", { class: "muted" }, `Order details: ${facts}.`));
  }
  if (row.duplicate_order_id) {
    out.push(el("p", { class: "muted" }, "Another row in this file has the same order reference. Each row was checked on its own."));
  }
  return out;
}

// ------------------------------------------------------------------ start
function initBatch() {
  renderColumnHelp();
  setupDropzone();
  $("batch-form").addEventListener("submit", onBatchSubmit);
  $("status-filter").addEventListener("change", (e) => { batch.status = e.target.value; batch.page = 1; loadRows(); });
  $("page-prev").addEventListener("click", () => { if (batch.page > 1) { batch.page -= 1; loadRows(); } });
  $("page-next").addEventListener("click", () => { if (batch.page < batch.pages) { batch.page += 1; loadRows(); } });
  $("batch-error").tabIndex = -1;
  updateBatchButton();             // app.js calls it again whenever the service state changes
}

document.addEventListener("DOMContentLoaded", initBatch);
