/* Order check: web UI for the order anomaly API. This file: tabs + the "Single order" tab.
 * batch.js holds the "Whole file" tab and reuses the helpers defined here (el, $, callApi, FIELDS...).
 *
 * Single order talks to the same-origin API: GET /health, GET /model-info, POST /analyze-order.
 * Everything coming back from the server is written with textContent (never innerHTML),
 * and "Recent checks" lives in memory only: nothing is stored in the browser.
 */
"use strict";

// ------------------------------------------------------------------ field list (mirrors app/schemas.py)
// kind: id | text | int | money | percent | datetime | select
// Bounds are the same as the API's; the select options come from /model-info allowed_values.
const MAX_AMOUNT = 1e9;
const ID_PATTERN = /^[A-Za-z0-9_-]+$/;

const FIELDS = [
  // Customer
  { name: "order_id", section: "customer", kind: "id", label: "Order reference",
    hint: "The order number from your shop system, e.g. ORD0020474." },
  { name: "customer_id", section: "customer", kind: "id", optional: true, label: "Customer reference",
    hint: "The customer's number in your shop system." },
  { name: "customer_country", section: "customer", kind: "select", label: "Customer country",
    hint: "The country on the customer's account." },
  { name: "customer_age", section: "customer", kind: "int", optional: true, min: 18, max: 120,
    label: "Customer age", hint: "In years." },
  { name: "account_age_days", section: "customer", kind: "int", min: 0, max: 36500,
    label: "Account age in days", hint: "How many days ago the customer created their account. 0 means today." },
  // Product
  { name: "product_id", section: "product", kind: "id", label: "Product code",
    hint: "The product's code in your catalogue, e.g. P01005." },
  { name: "product_name", section: "product", kind: "text", optional: true, max: 200,
    label: "Product name", hint: "For your own reference. It does not affect the result." },
  { name: "category", section: "product", kind: "select", label: "Product category",
    hint: "The department the product belongs to." },
  { name: "quantity", section: "product", kind: "int", min: 1, max: 1000000,
    label: "Quantity", hint: "How many items of this product were ordered." },
  // Money
  { name: "unit_price", section: "money", kind: "money", positive: true,
    label: "Price per item", hint: "The price of one item before any discount." },
  { name: "discount_pct", section: "money", kind: "percent", optional: true, min: 0, max: 95,
    label: "Discount (%)", hint: "The percentage taken off the item price. Leave empty if unknown." },
  { name: "shipping_cost", section: "money", kind: "money", optional: true,
    label: "Shipping cost", hint: "What the customer paid for delivery. Leave empty if unknown." },
  { name: "tax_amount", section: "money", kind: "money",
    label: "Tax", hint: "The tax charged on the order." },
  { name: "platform_fee", section: "money", kind: "money",
    label: "Platform fee", hint: "The fee the selling platform charged for this order." },
  { name: "order_value", section: "money", kind: "money", positive: true, wide: true,
    label: "Order total", hint: "The final amount the customer paid." },
  // Payment & channel
  { name: "order_timestamp", section: "payment", kind: "datetime",
    label: "Order date and time (UTC)", hint: "When the order was placed, in UTC (the same as UK winter time)." },
  { name: "payment_method", section: "payment", kind: "select",
    label: "Payment method", hint: "How the customer paid." },
  { name: "sales_channel", section: "payment", kind: "select",
    label: "Where the order was placed", hint: "Your website, your mobile app, or a marketplace." },
  { name: "device_type", section: "payment", kind: "select", optional: true,
    label: "Customer's device", hint: "The device used to place the order." },
  { name: "ip_country", section: "payment", kind: "select", optional: true,
    label: "Country the order came from", hint: "Where the customer's internet connection was, based on their IP address." },
  { name: "order_status", section: "payment", kind: "select",
    label: "Order status", hint: "Where the order is now." },
  { name: "coupon_used", section: "payment", kind: "select",
    label: "Coupon used", hint: "Whether the customer entered a coupon code." },
];
const FIELD_BY_NAME = Object.fromEntries(FIELDS.map((f) => [f.name, f]));

// Two real rows from orders-sheet.csv (row 0: normal; row 135: flagged).
// tests/test_ui.py reads the JSON between the markers and checks it against the CSV and the API.
/* EXAMPLES:BEGIN */
const EXAMPLES = {
  "normal": {"order_id": "ORD0020474", "order_timestamp": "2025-09-12 02:16:00", "customer_id": "C009777", "customer_country": "France", "customer_age": 52, "account_age_days": 806, "product_id": "P01005", "product_name": "HOM Product 1005", "category": "Home", "quantity": 1, "unit_price": 22.74, "discount_pct": 20.0, "shipping_cost": 5.37, "tax_amount": 0.25, "platform_fee": 2.79, "order_value": 23.81, "payment_method": "PayPal", "sales_channel": "Mobile App", "device_type": "Android", "ip_country": "Italy", "order_status": "Shipped", "coupon_used": "No"},
  "flagged": {"order_id": "ORD0000170", "order_timestamp": "2025-06-15 20:34:00", "customer_id": "C009652", "customer_country": "UK", "customer_age": 59, "account_age_days": 1558, "product_id": "P00590", "product_name": "OFF Product 0590", "category": "Office", "quantity": 4, "unit_price": 248.48, "discount_pct": 72.0, "shipping_cost": 7.13, "tax_amount": 139.62, "platform_fee": 62.48, "order_value": 2465.67, "payment_method": "Apple Pay", "sales_channel": "Mobile App", "device_type": "Tablet", "ip_country": "Italy", "order_status": "Completed", "coupon_used": "No"}
};
/* EXAMPLES:END */

const SCORE_SCALE = { min: 0.3, max: 0.8 };   // documented anomaly_score range (detect_anomalies.py)
const TOTAL_TOLERANCE = 0.02;                  // rounding slack when comparing totals
const HEALTH_POLL_MS = 30000;

const state = { modelInfo: null, busy: false, recent: [], activeRecent: null };
const $ = (id) => document.getElementById(id);
const numberFmt = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });
const moneyFmt = new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

// ------------------------------------------------------------------ small DOM helpers
/** Create an element with text content (never HTML) and attributes. */
function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === false || value === undefined || value === null) continue;
    node.setAttribute(key, value === true ? "" : String(value));
  }
  if (text !== undefined) node.textContent = text;
  return node;
}

function input(name) {
  return document.querySelector(`[name="${name}"]`);
}

// ------------------------------------------------------------------ building the form
/** Render every field from FIELDS into its fieldset. */
function buildForm() {
  for (const f of FIELDS) {
    const grid = document.querySelector(`fieldset[data-section="${f.section}"] .grid`);
    const wrap = el("div", { class: "field" + (f.wide ? " field-wide" : ""), "data-field": f.name });
    const label = el("label", { for: `f-${f.name}` }, f.label);
    if (f.optional) label.append(el("span", { class: "optional" }, "optional"));
    const hint = el("p", { class: "hint", id: `hint-${f.name}` }, f.hint);
    const error = el("p", { class: "error", id: `err-${f.name}`, "aria-live": "polite" });
    wrap.append(label, hint, makeControl(f), error);
    grid.append(wrap);
  }
}

/** The input/select for one field, with the same limits the API uses. */
function makeControl(f) {
  const common = { id: `f-${f.name}`, name: f.name, "aria-describedby": `hint-${f.name} err-${f.name}`,
                   "aria-required": f.optional ? "false" : "true" };
  if (f.kind === "select") {
    const select = el("select", common);
    select.append(el("option", { value: "" }, f.optional ? "Not known" : "Choose…"));
    return select;
  }
  if (f.kind === "datetime") return el("input", { ...common, type: "datetime-local", step: 60 });
  if (f.kind === "id" || f.kind === "text") {
    return el("input", { ...common, type: "text", maxlength: f.kind === "id" ? 64 : f.max,
                         autocomplete: "off", spellcheck: "false" });
  }
  const isInt = f.kind === "int";
  return el("input", {
    ...common, type: "number", inputmode: isInt ? "numeric" : "decimal",
    min: f.min ?? 0, max: f.max ?? MAX_AMOUNT, step: isInt ? 1 : "0.01",
  });
}

/** Fill the dropdowns from /model-info allowed_values (the model decides, not this file). */
function fillDropdowns(allowed) {
  for (const f of FIELDS.filter((x) => x.kind === "select")) {
    const select = input(f.name);
    const keep = select.value;
    select.length = 1;                                 // keep the placeholder option
    for (const value of allowed[f.name] || []) select.append(el("option", { value }, value));
    select.value = keep;
  }
}

// ------------------------------------------------------------------ validation (same rules as the API)
/** Return a plain-language problem with one field's value, or "" if it is fine. */
function validate(name) {
  const f = FIELD_BY_NAME[name];
  const control = input(name);
  const raw = control.value.trim();

  if (control.validity && control.validity.badInput) return "Enter a number using digits only.";
  if (raw === "") {
    if (f.optional) return "";
    return f.kind === "select" ? "Choose one of the options." : "This is needed to check the order.";
  }
  switch (f.kind) {
    case "id":
      if (raw.length > 64 || !ID_PATTERN.test(raw)) {
        return "Use only letters, numbers, dashes (-) and underscores (_), up to 64 characters.";
      }
      return "";
    case "text":
      return raw.length > f.max ? `Keep this under ${f.max} characters.` : "";
    case "select":
      return (state.modelInfo?.allowed_values?.[name] || []).includes(raw) ? "" : "Choose one of the options.";
    case "int": {
      const n = Number(raw);
      if (!Number.isInteger(n) || n < f.min || n > f.max) {
        return `Enter a whole number from ${numberFmt.format(f.min)} to ${numberFmt.format(f.max)}.`;
      }
      return "";
    }
    case "percent": {
      const n = Number(raw);
      return Number.isFinite(n) && n >= f.min && n <= f.max ? "" : `Enter a percentage from ${f.min} to ${f.max}.`;
    }
    case "money": {
      const n = Number(raw);
      if (!Number.isFinite(n)) return "Enter an amount using digits only.";
      if (f.positive && n <= 0) return "This must be more than 0.";
      if (n < 0) return "This can't be negative. Use 0 if there was none.";
      if (n > MAX_AMOUNT) return "That amount is too large to be a real order.";
      return "";
    }
    case "datetime": {
      const when = parseUtc(raw);
      if (!when) return "Enter a valid date and time.";
      if (when.getTime() > Date.now() + 5 * 60 * 1000) return "The order time can't be in the future.";
      return "";
    }
    default:
      return "";
  }
}

/** Show (or clear) the error under one field and mark it invalid for screen readers. */
function showFieldError(name, message) {
  const control = input(name);
  $(`err-${name}`).textContent = message;
  control.setAttribute("aria-invalid", message ? "true" : "false");
  control.closest(".field").classList.toggle("has-error", Boolean(message));
}

/** Validate every field; returns the names of the invalid ones. */
function validateAll() {
  const bad = [];
  for (const f of FIELDS) {
    const message = validate(f.name);
    showFieldError(f.name, message);
    if (message) bad.push(f.name);
  }
  return bad;
}

function showSummary(text) {
  const box = $("form-summary");
  box.textContent = text;
  box.hidden = !text;
}

// ------------------------------------------------------------------ payload
/** "2025-09-12T02:16" (taken as UTC) -> Date, or null if it doesn't parse. */
function parseUtc(value) {
  const iso = value.length === 16 ? `${value}:00Z` : `${value}Z`;
  const when = new Date(iso);
  return Number.isNaN(when.getTime()) ? null : when;
}

/** Form -> JSON body for POST /analyze-order. Every key is sent; empty optional fields become null.
 *  Runs after validate(), so whole-number fields are already integers (JSON 806, not "806"). */
function buildPayload() {
  const body = {};
  for (const f of FIELDS) {
    const raw = input(f.name).value.trim();
    if (raw === "") body[f.name] = null;
    else if (f.kind === "int" || f.kind === "money" || f.kind === "percent") body[f.name] = Number(raw);
    else if (f.kind === "datetime") body[f.name] = (raw.length === 16 ? `${raw}:00` : raw) + "Z";
    else body[f.name] = raw;
  }
  return body;
}

// ------------------------------------------------------------------ API
/** fetch wrapper: never throws. Returns {ok, status, body, requestId, offline}. */
async function callApi(path, options = {}) {
  try {
    const response = await fetch(path, { headers: { Accept: "application/json" }, ...options });
    let body = null;
    try { body = await response.json(); } catch { /* not JSON: keep null */ }
    return { ok: response.ok, status: response.status, body,
             requestId: response.headers.get("x-request-id") || body?.error?.request_id || "", offline: false };
  } catch {
    return { ok: false, status: 0, body: null, requestId: "", offline: true };
  }
}

// ------------------------------------------------------------------ service status
function setStatus(ready, text) {
  const box = $("service-status");
  box.dataset.state = ready ? "ready" : "down";
  box.querySelector(".status-text").textContent = text;
}

/** Check /health, then load /model-info once (dropdown options + the "How does this work?" facts). */
async function refreshService() {
  const health = await callApi("/health");
  const ready = health.ok && health.body?.model_loaded === true;
  setStatus(ready, ready ? "Service ready" : "Service unavailable");
  if (ready && !state.modelInfo) {
    const info = await callApi("/model-info");
    if (info.ok && info.body) {
      applyModelInfo(info.body);
      if ($("result-body").hidden) hideResult();      // clear the "not ready" message once it recovers
    } else {
      setStatus(false, "Service unavailable");
    }
  }
  if (!state.modelInfo) {                            // the form can't be used yet: say why
    showResultError(health.offline
      ? "Cannot reach the service. Check that it is running, then reload this page."
      : "The checking service is not ready. Please try again shortly. This page checks again every 30 seconds.");
  }
  updateSubmitState();
}

function applyModelInfo(info) {
  state.modelInfo = info;
  fillDropdowns(info.allowed_values || {});
  $("fact-model").textContent = info.model_name;
  $("fact-version").textContent = info.model_version;
  $("fact-threshold").textContent = `${info.threshold.toFixed(3)} (higher scores are flagged)`;
  $("fact-rows").textContent = info.n_training_rows ? numberFmt.format(info.n_training_rows) : "not recorded";
  $("about-facts").hidden = false;
}

function updateSubmitState() {
  $("submit-btn").disabled = state.busy || !state.modelInfo;
  // Examples and file prefill need the dropdown options, which come from /model-info.
  for (const button of document.querySelectorAll("[data-example]")) button.disabled = !state.modelInfo;
  $("prefill-file").disabled = !state.modelInfo;
  if (typeof updateBatchButton === "function") updateBatchButton();     // the Whole file tab (batch.js)
}

// ------------------------------------------------------------------ expected total
/** qty × price × (1 − discount%) + shipping + tax: the same sum the model compares the total with. */
function expectedTotal() {
  const num = (name) => {
    const raw = input(name).value.trim();
    return raw === "" ? null : Number(raw);
  };
  const qty = num("quantity"), price = num("unit_price"), tax = num("tax_amount");
  const shipping = num("shipping_cost"), discount = num("discount_pct") ?? 0;
  if ([qty, price, tax, shipping].some((v) => v === null || !Number.isFinite(v)) || !Number.isFinite(discount)) {
    return null;
  }
  return qty * price * (1 - discount / 100) + shipping + tax;
}

function renderExpected() {
  const expected = expectedTotal();
  const compare = $("expected-compare");
  $("use-expected").disabled = expected === null;
  compare.className = "expected-compare";
  if (expected === null) {
    $("expected-value").textContent = "–";
    compare.textContent = "Fill in quantity, price, shipping and tax to work this out.";
    return;
  }
  $("expected-value").textContent = moneyFmt.format(expected);
  const totalRaw = input("order_value").value.trim();
  const total = Number(totalRaw);
  if (totalRaw === "" || !Number.isFinite(total)) {
    compare.textContent = "";
    return;
  }
  const diff = total - expected;
  if (Math.abs(diff) <= TOTAL_TOLERANCE) {
    compare.textContent = "Matches the order total.";
    compare.classList.add("is-match");
  } else {
    compare.textContent = `The order total is ${moneyFmt.format(Math.abs(diff))} ${diff > 0 ? "more" : "less"} than this.`;
    compare.classList.add("is-diff");
  }
}

function useExpected() {
  const expected = expectedTotal();
  if (expected === null) return;
  input("order_value").value = expected.toFixed(2);
  showFieldError("order_value", validate("order_value"));
  renderExpected();
}

// ------------------------------------------------------------------ errors from the server
/** Turn an error response into messages next to fields (422) or one plain message. */
function renderErrors(resp) {
  hideResult();                                       // never leave an earlier order's verdict next to an error
  if (resp.offline) {
    setStatus(false, "Service unavailable");
    return showResultError("Cannot reach the service. Check that it is running, then try again.");
  }
  if (resp.status === 503) {
    setStatus(false, "Service unavailable");
    return showResultError("The checking service is not ready. Please try again shortly.");
  }
  if (resp.status === 422) {
    const unknown = [];
    let first = null;
    for (const detail of resp.body?.error?.details || []) {
      const name = String(detail.field || "").split(".")[0];
      if (FIELD_BY_NAME[name]) {
        // Prefer our own plain wording; fall back to the server's reason as plain text.
        const message = validate(name) || `The service did not accept this value (${detail.reason}).`;
        showFieldError(name, message);
        first = first || name;
      } else {
        unknown.push(String(detail.reason || "invalid value"));
      }
    }
    showSummary(unknown.length
      ? `The service could not check this order: ${unknown.join("; ")}.`
      : "The service found a problem with some fields. They are marked below.");
    if (first) input(first).focus();
    return;
  }
  const ref = resp.requestId ? ` If it keeps happening, give support this reference: ${resp.requestId}.` : "";
  return showResultError(`Sorry, something went wrong while checking this order.${ref}`);
}

/** Back to the empty result panel (used by Clear and before showing an error). */
function hideResult() {
  $("result-body").hidden = true;
  $("result-error").hidden = true;
  $("result-empty").hidden = false;
  state.activeRecent = null;
  renderRecent();
}

function showResultError(message) {
  $("result-empty").hidden = true;
  const box = $("result-error");
  box.textContent = message;
  box.hidden = false;
}

// ------------------------------------------------------------------ result
/** Draw one result: verdict, score bar, reasons, details. `check` = {result, requestId, checkedAt}. */
function renderResult(check) {
  const { result, requestId, checkedAt } = check;
  const flagged = result.status === "suspicious";
  $("result-empty").hidden = true;
  $("result-error").hidden = true;
  $("result-body").hidden = false;

  const verdict = $("verdict");
  verdict.dataset.tone = flagged ? "flag" : "ok";
  verdict.querySelector(".verdict-icon").textContent = flagged ? "!" : "✓";
  verdict.querySelector(".verdict-text").textContent = flagged ? "Needs a closer look" : "Looks normal";
  $("verdict-order").textContent =
    `Order ${result.order_id}: score ${result.anomaly_score.toFixed(2)}, alert line ${result.threshold_used.toFixed(2)}.`;

  renderMeter(result.anomaly_score, result.threshold_used, flagged);

  const list = $("reasons");
  list.replaceChildren();
  const reasons = result.reasons || [];
  if (!reasons.length) list.append(el("li", {}, "No explanation is available for this order."));
  for (const reason of reasons) list.append(el("li", {}, reason));

  const details = $("details-list");
  details.replaceChildren();
  for (const [term, value] of [
    ["Exact score", result.anomaly_score.toFixed(6)],
    ["Alert line", result.threshold_used.toFixed(6)],
    ["Model version", result.model_version],
    ["Request reference", requestId || "not available"],
    ["Checked at", checkedAt.toLocaleString()],
  ]) {
    details.append(el("dt", {}, term), el("dd", {}, value));
  }
}

/** Position of a score on the 0.3–0.8 bar, as a percentage (clamped). */
function scorePercent(score) {
  const p = (score - SCORE_SCALE.min) / (SCORE_SCALE.max - SCORE_SCALE.min);
  return Math.min(100, Math.max(0, p * 100));
}

function renderMeter(score, threshold, flagged) {
  const pos = scorePercent(score);
  $("meter").dataset.tone = flagged ? "flag" : "ok";
  $("meter-fill").style.width = `${pos}%`;
  $("meter-pin").style.left = `${pos}%`;
  $("meter-line").style.left = `${scorePercent(threshold)}%`;
  $("meter-track").setAttribute("aria-label",
    `Score ${score.toFixed(2)}. The alert line is at ${threshold.toFixed(2)}. ` +
    (flagged ? "The score is past the alert line." : "The score is below the alert line."));
}

// ------------------------------------------------------------------ recent checks (memory only)
function addRecent(check) {
  state.recent.unshift(check);
  state.activeRecent = check;
  renderRecent();
}

function renderRecent() {
  const list = $("recent-list");
  list.replaceChildren();
  $("recent-empty").hidden = state.recent.length > 0;
  for (const check of state.recent) {
    const flagged = check.result.status === "suspicious";
    const button = el("button", { type: "button", class: "recent-item", "data-tone": flagged ? "flag" : "ok",
                                  "aria-current": check === state.activeRecent ? "true" : false });
    button.append(
      el("span", { class: "recent-mark", "aria-hidden": "true" }, flagged ? "!" : "✓"),
      el("span", { class: "recent-id" }, check.result.order_id),
      el("span", { class: "recent-score" }, check.result.anomaly_score.toFixed(2)),
      el("span", { class: "recent-status" }, flagged ? "Needs a closer look" : "Looks normal"),
    );
    button.addEventListener("click", () => {
      state.activeRecent = check;
      renderResult(check);
      renderRecent();
      $("result").focus();
    });
    const item = el("li");
    item.append(button);
    list.append(item);
  }
}

// ------------------------------------------------------------------ actions
function setBusy(busy) {
  state.busy = busy;
  const button = $("submit-btn");
  button.classList.toggle("is-busy", busy);
  button.setAttribute("aria-busy", busy ? "true" : "false");
  button.querySelector(".btn-label").textContent = busy ? "Checking…" : "Check this order";
  updateSubmitState();
}

async function onSubmit(event) {
  event.preventDefault();
  if (state.busy) return;
  showSummary("");
  const bad = validateAll();
  if (bad.length) {
    showSummary(bad.length === 1 ? "One field needs fixing. It is marked below."
                                 : `${bad.length} fields need fixing. They are marked below.`);
    input(bad[0]).focus();
    return;
  }
  setBusy(true);
  const resp = await callApi("/analyze-order", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(buildPayload()),
  });
  setBusy(false);
  if (!resp.ok || !resp.body) return renderErrors(resp);
  const check = { result: resp.body, requestId: resp.requestId, checkedAt: new Date() };
  renderResult(check);
  addRecent(check);
  $("result").focus();
}

function clearForm() {
  $("order-form").reset();
  for (const f of FIELDS) showFieldError(f.name, "");
  showSummary("");
  $("prefill-status").hidden = true;
  renderExpected();
  hideResult();
  input("order_id").focus();
}

/** Load one of the CSV examples into the form (the user still presses "Check this order"). */
function loadExample(key) {
  const example = EXAMPLES[key];
  for (const f of FIELDS) {
    let value = example[f.name];
    if (value === null || value === undefined) value = "";
    if (f.kind === "datetime") value = String(value).replace(" ", "T").slice(0, 16);
    input(f.name).value = String(value);
    showFieldError(f.name, "");
  }
  showSummary("");
  renderExpected();
  $("submit-btn").focus();
}

// ------------------------------------------------------------------ tabs ("Single order" / "Whole file")
const TABS = [{ tab: "tab-single", panel: "panel-single", hash: "" }, { tab: "tab-file", panel: "panel-file", hash: "#file" }];

/** Show one tab panel; the choice is kept in the address (#file) so reloads and links keep it. */
function selectTab(tabId, focus = false) {
  for (const t of TABS) {
    const on = t.tab === tabId;
    const tab = $(t.tab);
    tab.setAttribute("aria-selected", on ? "true" : "false");
    tab.tabIndex = on ? 0 : -1;
    $(t.panel).hidden = !on;
    if (on) {
      if (focus) tab.focus();
      history.replaceState(null, "", t.hash || location.pathname);
    }
  }
}

/** Click or arrow keys / Home / End move between tabs (WAI-ARIA tabs pattern). */
function setupTabs() {
  const ids = TABS.map((t) => t.tab);
  for (const id of ids) {
    $(id).addEventListener("click", () => selectTab(id));
    $(id).addEventListener("keydown", (e) => {
      const i = ids.indexOf(id);
      const next = { ArrowRight: ids[(i + 1) % ids.length], ArrowLeft: ids[(i - 1 + ids.length) % ids.length],
                     Home: ids[0], End: ids[ids.length - 1] }[e.key];
      if (next) { e.preventDefault(); selectTab(next, true); }
    });
  }
  selectTab(location.hash === "#file" ? "tab-file" : "tab-single");
}

// ------------------------------------------------------------------ CSV parsing (shared with batch.js)
/** RFC 4180 CSV -> array of rows (arrays of strings). Handles quotes, "" escapes, CRLF and a UTF-8 BOM. */
function parseCsv(text) {
  const rows = [];
  let row = [], cell = "", quoted = false;
  text = text.replace(/^\uFEFF/, "");
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell); rows.push(row); row = []; cell = "";
    } else cell += ch;
  }
  if (cell !== "" || row.length) { row.push(cell); rows.push(row); }
  return rows.filter((r) => r.some((c) => c.trim() !== ""));          // skip blank lines
}

// ------------------------------------------------------------------ "Upload order details" (prefill, never submits)
const PREFILL_MAX_BYTES = 1024 * 1024;
const REPORT_COLUMNS = ["row_number", "anomaly_score", "status", "reasons", "missing_fields",
                        "validation_errors", "duplicate_order_id"];

/** Read one order from a .json file (object, or array of one object) or a one-row .csv. */
async function readOrderFile(file) {
  const name = file.name.toLowerCase();
  if (file.size > PREFILL_MAX_BYTES) throw new Error("This file is too large for one order. Use the Whole file tab.");
  const text = await file.text();
  if (name.endsWith(".json")) {
    let data;
    try { data = JSON.parse(text); } catch { throw new Error("This JSON file couldn't be read. Check that it is valid JSON."); }
    if (Array.isArray(data)) {
      if (data.length !== 1) throw new Error(`This file has ${data.length} orders. Use the Whole file tab for more than one.`);
      data = data[0];
    }
    if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error("The JSON file must hold one order (an object with field names).");
    return data;
  }
  if (name.endsWith(".csv")) {
    const rows = parseCsv(text);
    if (rows.length < 2) throw new Error("The CSV file needs a row of column names and one order.");
    if (rows.length > 2) throw new Error(`This file has ${rows.length - 1} orders. Use the Whole file tab for more than one.`);
    return Object.fromEntries(rows[0].map((h, i) => [h, rows[1][i] ?? ""]));
  }
  throw new Error("Choose a .json or .csv file.");
}

/** A value from a file -> what the form control can hold, or null if it can't hold it. */
function toControlValue(f, value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "boolean") value = value ? "Yes" : "No";
  const text = String(value).trim();
  if (text === "") return "";
  if (f.kind === "select") {
    const match = (state.modelInfo?.allowed_values?.[f.name] || []).find((v) => v.toLowerCase() === text.toLowerCase());
    return match ?? null;
  }
  if (f.kind === "datetime") {
    // "2025-09-12 02:16:00" is taken as UTC as it is; a time with a zone (Z, +02:00) is converted to UTC.
    let picked = text.replace(" ", "T").slice(0, 16);
    if (/(Z|[+-]\d\d:?\d\d)$/i.test(text)) {
      const when = new Date(text);
      picked = Number.isNaN(when.getTime()) ? "" : when.toISOString().slice(0, 16);
    }
    return /^\d{4}-\d\d-\d\dT\d\d:\d\d$/.test(picked) ? picked : null;
  }
  if (f.kind === "int" || f.kind === "money" || f.kind === "percent") return Number.isFinite(Number(text)) ? text : null;
  return text;
}

/** Fill the form from one order record, then show every problem next to its field. Does not submit. */
function applyOrderRecord(record, fileName) {
  const data = Object.fromEntries(Object.entries(record).map(([k, v]) => [String(k).trim().toLowerCase(), v]));
  const rejected = {};
  for (const f of FIELDS) {
    const raw = data[f.name];
    const value = toControlValue(f, raw);
    input(f.name).value = value ?? "";
    if (value === null) rejected[f.name] = String(raw).slice(0, 60);
  }
  const unknown = Object.keys(data).filter((k) => !FIELD_BY_NAME[k] && !REPORT_COLUMNS.includes(k));
  showSummary("");
  const bad = new Set(validateAll());
  for (const [name, raw] of Object.entries(rejected)) {          // say what the file contained
    const f = FIELD_BY_NAME[name];
    showFieldError(name, f.kind === "select" ? `The file says "${raw}", which isn't one of the options.`
                                             : `The file's value "${raw}" can't be used here.`);
    bad.add(name);
  }
  renderExpected();
  const parts = [`Filled in the form from ${fileName}.`];
  parts.push(bad.size ? `${bad.size} ${bad.size === 1 ? "field needs" : "fields need"} attention (marked below).`
                      : "Review the details, then press Check this order.");
  if (unknown.length) parts.push(`Not used: ${unknown.slice(0, 8).join(", ")}${unknown.length > 8 ? "…" : ""}.`);
  const status = $("prefill-status");
  status.textContent = parts.join(" ");
  status.dataset.tone = bad.size ? "warn" : "ok";
  status.hidden = false;
  const firstBad = FIELDS.find((f) => bad.has(f.name));
  (firstBad ? input(firstBad.name) : $("submit-btn")).focus();
}

async function onPrefillFile(event) {
  const file = event.target.files[0];
  event.target.value = "";                                       // choosing the same file again still works
  if (!file) return;
  const status = $("prefill-status");
  try {
    applyOrderRecord(await readOrderFile(file), file.name);
  } catch (err) {
    status.textContent = err.message;
    status.dataset.tone = "error";
    status.hidden = false;
  }
}

// ------------------------------------------------------------------ start
function init() {
  setupTabs();
  buildForm();
  $("prefill-file").addEventListener("change", onPrefillFile);
  const form = $("order-form");
  form.addEventListener("submit", onSubmit);
  // Errors appear when you leave a field, and clear as soon as the value becomes valid.
  form.addEventListener("focusout", (e) => {
    if (e.target.name && FIELD_BY_NAME[e.target.name]) showFieldError(e.target.name, validate(e.target.name));
  });
  form.addEventListener("input", (e) => {
    const name = e.target.name;
    if (name && e.target.getAttribute("aria-invalid") === "true" && !validate(name)) showFieldError(name, "");
    renderExpected();
  });
  form.addEventListener("change", renderExpected);
  $("use-expected").addEventListener("click", useExpected);
  $("clear-btn").addEventListener("click", clearForm);
  for (const button of document.querySelectorAll("[data-example]")) {
    button.addEventListener("click", () => loadExample(button.dataset.example));
  }
  // Orders can't be in the future: cap the date picker at "now" in UTC.
  input("order_timestamp").max = new Date().toISOString().slice(0, 16);

  renderExpected();
  renderRecent();
  refreshService();
  setInterval(refreshService, HEALTH_POLL_MS);
}

document.addEventListener("DOMContentLoaded", init);
