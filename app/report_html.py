"""The downloadable HTML summary report: one self-contained file (inline CSS + SVG, no scripts, no external requests).

Built from a report's meta.json. Every text value is formula-neutralized and HTML-escaped.
"""
from datetime import datetime, timezone
from html import escape

from .reports import neutralize

STATUS_TEXT = {"normal": "Looks normal", "suspicious": "Needs a closer look",
               "missing_data": "Missing data", "invalid_data": "Invalid data"}
DIM_TEXT = {"category": "Product category", "customer_country": "Customer country", "sales_channel": "Sales channel"}

CSS = """
:root{--bg:#f4f7f9;--surface:#fff;--ink:#17283a;--muted:#4a5b6c;--line:#c3cfd9;--ok:#1d6a42;--flag:#8a4600;
--bad:#a3261b;--accent:#1f4e79;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#0e1720;--surface:#16222d;--ink:#e3eaf0;--muted:#a6b5c3;--line:#2e3e4d;
--ok:#74d6a0;--flag:#f5b968;--bad:#ff9e92;--accent:#8dbce6;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.5 "Atkinson Hyperlegible","Segoe UI",system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}h1{font-size:1.75rem;margin:0 0 4px}
h2{font-size:1.25rem;margin:32px 0 10px}p{margin:0 0 8px}.muted{color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px}
.facts{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;margin:0}.facts dt{color:var(--muted)}
.facts dd{margin:0;overflow-wrap:anywhere}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.stat .n{font-size:1.75rem;font-weight:700;font-variant-numeric:tabular-nums}.stat .l{color:var(--muted)}
.ok{color:var(--ok)}.flag{color:var(--flag)}.bad{color:var(--bad)}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9375rem}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:700}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
svg text{fill:var(--muted);font-size:11px}.note{border-left:4px solid var(--flag);padding:10px 14px}
ul{margin:0;padding-left:1.2em}
"""


def _t(value) -> str:
    """Neutralize spreadsheet formulas, then HTML-escape."""
    return escape(str(neutralize("" if value is None else str(value))))


def _histogram(meta: dict) -> str:
    """Stacked bar chart of scores (normal + suspicious) with the threshold drawn as a line."""
    h = meta["histogram"]
    edges, normal, flagged = h["bin_edges"], h["normal"], h["suspicious"]
    w, ht, pad_l, pad_r, pad_b, pad_t = 760, 228, 48, 24, 40, 12
    plot_w = w - pad_l - pad_r
    top = max([a + b for a, b in zip(normal, flagged)] + [1])
    bw = plot_w / len(normal)
    lo, hi = edges[0], edges[-1]
    y = lambda v: pad_t + (ht - pad_t - pad_b) * (1 - v / top)
    bars = []
    for i, (n, s) in enumerate(zip(normal, flagged)):
        x = pad_l + i * bw
        bars.append(f'<rect x="{x + 1:.1f}" y="{y(n):.1f}" width="{bw - 2:.1f}" height="{y(0) - y(n):.1f}" '
                    f'fill="var(--ok)" opacity=".8"><title>{edges[i]:.3f}-{edges[i + 1]:.3f}: {n} normal</title></rect>')
        if s:
            bars.append(f'<rect x="{x + 1:.1f}" y="{y(n + s):.1f}" width="{bw - 2:.1f}" height="{y(n) - y(n + s):.1f}" '
                        f'fill="var(--flag)"><title>{edges[i]:.3f}-{edges[i + 1]:.3f}: {s} need a closer look</title></rect>')
    thr = min(max(meta["threshold"], lo), hi)
    tx = pad_l + (thr - lo) / (hi - lo) * plot_w
    ticks = "".join(f'<text x="{pad_l + (e - lo) / (hi - lo) * plot_w:.1f}" y="{ht - 22}" text-anchor="middle">'
                    f'{e:.2f}</text>' for e in edges[::4])
    return (f'<svg viewBox="0 0 {w} {ht}" width="100%" role="img" aria-label="Histogram of anomaly scores. '
            f'The alert line is at {meta["threshold"]:.3f}.">'
            f'<line x1="{pad_l}" y1="{y(0):.1f}" x2="{w - pad_r}" y2="{y(0):.1f}" stroke="var(--line)"/>'
            f'<text x="{pad_l - 6}" y="{y(top) + 4:.1f}" text-anchor="end">{top:,}</text>'
            f'<text x="{pad_l - 6}" y="{y(0):.1f}" text-anchor="end">0</text>{"".join(bars)}{ticks}'
            f'<line x1="{tx:.1f}" y1="{pad_t - 4}" x2="{tx:.1f}" y2="{y(0) + 4:.1f}" stroke="var(--ink)" '
            f'stroke-width="2" stroke-dasharray="4 3"/><text x="{tx + 4:.1f}" y="{pad_t + 6}" '
            f'style="fill:var(--ink)">Alert line {meta["threshold"]:.3f}</text>'
            f'<text x="{pad_l + plot_w / 2:.1f}" y="{ht - 4}" text-anchor="middle">anomaly score (higher = more unusual)'
            f'</text></svg>')


def _table(headers: list[tuple[str, bool]], rows: list[list], empty: str) -> str:
    if not rows:
        return f'<p class="muted">{_t(empty)}</p>'
    head = "".join(f'<th scope="col"{" class=num" if num else ""}>{_t(h)}</th>' for h, num in headers)
    body = "".join("<tr>" + "".join(f'<td{" class=num" if headers[i][1] else ""}>{cell}</td>'
                                    for i, cell in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _list(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{_t(i)}</li>" for i in items) + "</ul>" if items else ""


def render(meta: dict) -> str:
    s = meta["summary"]
    uploaded = meta.get("uploaded_at", "")
    cards = [("Total rows", s["total_rows"], ""), ("✓ Looks normal", s["normal"], "ok"),
             ("! Needs a closer look", s["suspicious"], "flag"), ("? Missing data", s["missing_data"], "bad"),
             ("✕ Invalid data", s["invalid_data"], "bad")]
    cards_html = "".join(f'<div class="card stat"><div class="n {cls}">{n:,}</div><div class="l">{_t(label)}</div>'
                         f'</div>' for label, n, cls in cards)
    facts = [("File", meta["filename"]), ("Uploaded", uploaded), ("File size", f'{meta["size_bytes"]:,} bytes'),
             ("SHA-256", meta["sha256"]), ("Model version", meta["model_version"]),
             ("Alert line (threshold)", f'{meta["threshold"]:.6f}'), ("Processing time", f'{meta["duration_ms"]:,.0f} ms'),
             ("Report id", meta["report_id"])]
    if meta.get("unknown_columns"):
        facts.append(("Columns ignored (not used)", ", ".join(meta["unknown_columns"])))
    if meta.get("ignored_columns"):
        facts.append(("Columns from an earlier report (recalculated)", ", ".join(meta["ignored_columns"])))
    facts_html = "".join(f"<dt>{_t(k)}</dt><dd>{_t(v)}</dd>" for k, v in facts)

    top = [[str(i), f'{t["row_number"]:,}', _t(t["order_id"]), f'{t["anomaly_score"]:.4f}', _t(t["category"]),
            _t(t["customer_country"]), _t(t["sales_channel"]), _list(t["reasons"])]
           for i, t in enumerate(meta["top_suspicious"], start=1)]
    breakdowns = "".join(
        f'<div class="card"><h3>{_t(DIM_TEXT[dim])}</h3>' + _table(
            [("Value", False), ("Checked", True), ("Need a closer look", True), ("%", True)],
            [[_t(g["value"]), f'{g["checked"]:,}', f'{g["suspicious"]:,}', f'{g["suspicious_pct"]:.2f}']
             for g in groups], "No checked orders.") + "</div>"
        for dim, groups in meta["breakdowns"].items())
    problem_total = s["missing_data"] + s["invalid_data"]
    problems = [[f'{p["row_number"]:,}', _t(p["order_id"]), _t(STATUS_TEXT[p["status"]]),
                 _t(", ".join(p["missing_fields"])),
                 _list([f'{e["field"]}: {e["reason"]}' for e in p["validation_errors"]])]
                for p in meta["problem_rows"]]
    more = (f'<p class="muted">Showing the first {len(problems):,} of {problem_total:,} problem rows. '
            'The full CSV report lists all of them.</p>') if problem_total > len(problems) else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Order check report: {_t(meta["filename"])}</title><style>{CSS}</style></head>
<body><main>
<h1>Order check report</h1>
<p class="muted">{_t(meta["filename"])}, checked {_t(uploaded)}. Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.</p>
<p class="note card">Orders marked "Needs a closer look" are statistically unusual compared with past orders.
This is a statistical warning, not proof of fraud. A person should review each one.</p>

<h2>Summary</h2>
<div class="cards">{cards_html}</div>
<p style="margin-top:10px">{s["suspicious_pct"]:.2f}% of the {s["scored_rows"]:,} checked orders need a closer look.
{s["duplicate_order_id_rows"]:,} rows share their order reference with another row (each was checked separately).</p>

<h2>File and model</h2>
<div class="card"><dl class="facts">{facts_html}</dl></div>

<h2>Score distribution</h2>
<div class="card">{_histogram(meta)}<p class="muted">Green: looks normal. Amber: needs a closer look (above the alert line).
Scores outside {meta["histogram"]["bin_edges"][0]:.2f}–{meta["histogram"]["bin_edges"][-1]:.2f} are counted in the first or last bar.</p></div>

<h2>Top {len(top)} orders that need a closer look</h2>
{_table([("#", True), ("Row", True), ("Order", False), ("Score", True), ("Category", False), ("Country", False),
         ("Channel", False), ("What stands out", False)], top, "No orders need a closer look.")}

<h2>Where the flagged orders are</h2>
<div class="grid3">{breakdowns}</div>

<h2>Rows that could not be checked</h2>
<p class="muted">Missing data: a required value is blank. Invalid data: a value is the wrong type, out of range or not an accepted option.</p>
{_table([("Row", True), ("Order", False), ("Problem", False), ("Missing", False), ("Invalid values", False)],
        problems, "Every row could be checked.")}{more}
</main></body></html>
"""
